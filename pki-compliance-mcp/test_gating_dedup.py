#!/usr/bin/env python3
"""Offline tests for the cost-gate + dedup logic in compliance_auto_refresh.py.

Runs no network calls, no Anthropic API, no Docker. Redirects the module's
data files to a temp dir and monkeypatches the change detector.
"""
import json
import tempfile
from pathlib import Path

import compliance_auto_refresh as car

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


def with_tmp():
    d = Path(tempfile.mkdtemp())
    car.DATA_DIR = d
    car.REJECTED_FILE = d / "rejected.json"
    car.LAST_RESEARCH_FILE = d / "last_research.json"
    return d


print("== should_run_research gate ==")
car.MAX_DAYS_BETWEEN_RESEARCH = 7

with_tmp()
run, _ = car.should_run_research(force=True)
check("force always runs", run is True)

car.detect_document_changes = lambda: (2, ["cabf_br", "chrome_root_policy"])
car.days_since_last_research = lambda: 1.0
run, reason = car.should_run_research(force=False)
check("doc change -> run", run is True and "changed" in reason)

car.detect_document_changes = lambda: (0, [])
car.days_since_last_research = lambda: 1.0
run, reason = car.should_run_research(force=False)
check("no change + fresh -> skip", run is False)

car.days_since_last_research = lambda: 8.0
run, reason = car.should_run_research(force=False)
check("no change + stale -> run (safety net)", run is True and "safety net" in reason)

car.detect_document_changes = lambda: (-1, [])
car.days_since_last_research = lambda: 1.0
run, reason = car.should_run_research(force=False)
check("detector down + fresh -> skip (protect cost)", run is False)

car.days_since_last_research = lambda: 9.0
run, reason = car.should_run_research(force=False)
check("detector down + stale -> run", run is True)

print("== dedup_changes ==")
d = with_tmp()
car.REJECTED_FILE.write_text(json.dumps({"ids": ["rejected-1"], "signatures": []}))

current_data = {
    "deadlines": [{"id": "existing-1", "title": "Existing One", "date": "2026-01-01"}],
    "cabfDocuments": [{"id": "tls-br", "version": "2.2.6"}],
}
changes = {
    "new_deadlines": [
        {"id": "existing-1", "title": "Existing One", "date": "2026-01-01"},   # dup by id
        {"id": "different-id", "title": "Existing One", "date": "2026-01-01"},  # dup by signature
        {"id": "rejected-1", "title": "Rejected Thing", "date": "2026-02-02"},  # rejected
        {"id": "brand-new", "title": "Brand New", "date": "2026-03-03"},        # keep
    ],
    "document_version_updates": [
        {"id": "tls-br", "new_version": "2.2.6"},        # already current -> drop
        {"id": "ev-guidelines", "new_version": "2.0.3"},  # keep
    ],
    "regulatory_updates": [],
    "needs_human_review": [],
    "summary": "x",
}
filtered, removed = car.dedup_changes(changes, current_data)
kept_ids = [x["id"] for x in filtered["new_deadlines"]]
check("keeps only brand-new deadline", kept_ids == ["brand-new"])
check("drops dup-by-id", "existing-1" not in kept_ids)
check("drops dup-by-signature", "different-id" not in kept_ids)
check("drops rejected", "rejected-1" not in kept_ids)
check("doc update at current version dropped",
      [u["id"] for u in filtered["document_version_updates"]] == ["ev-guidelines"])
check("removed log has 4 entries", len(removed) == 4)

print("== reject_ids persistence ==")
d = with_tmp()
(d / "pending_updates_2026-06-18.json").write_text(json.dumps({
    "new_deadlines": [{"id": "foo", "title": "Foo Thing", "date": "2026-07-01"}]
}))
total = car.reject_ids(["foo"])
saved = json.loads(car.REJECTED_FILE.read_text())
check("foo persisted to rejected ids", "foo" in saved["ids"])
check("signature captured from pending file", "foo thing|2026-07-01" in saved["signatures"])
check("reject returns total count", total == 1)

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
