#!/usr/bin/env python3
"""Offline tests for optional source_url / is_estimated support.

Covers the API serialization in pki_compliance_mcp.py (source_url present,
absent -> null, backwards compatible) and asserts DIFF_SYSTEM_PROMPT in
compliance_auto_refresh.py documents both fields. No network calls.
"""
import json

import pki_compliance_mcp as pki
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


print("== API serialization: source_url ==")
test_entry = {
    "id": "test-source-url-entry",
    "date": "2027-03-01",
    "title": "Test Entry With Source URL",
    "description": "Temporary test entry.",
    "source": "cab-forum",
    "category": "certificates",
    "isMajor": False,
    "source_url": "https://cabforum.org/example-ballot",
    "is_estimated": True,
}
no_url_entry = {
    "id": "test-entry-without-source-url",
    "date": "2027-04-01",
    "title": "Test Entry Without Source URL",
    "description": "Temporary test entry lacking source_url.",
    "source": "cab-forum",
    "category": "certificates",
    "isMajor": False,
}
pki.DEADLINES.append(test_entry)
pki.DEADLINES.append(no_url_entry)
try:
    unified = pki.get_all_deadlines_unified()
    by_id = {d["id"]: d for d in unified}

    added = by_id.get("test-source-url-entry", {})
    check("entry with source_url serializes it",
          added.get("source_url") == "https://cabforum.org/example-ballot")
    check("is_estimated passes through", added.get("is_estimated") is True)

    without = by_id.get("test-entry-without-source-url", {})
    check("entry without source_url gets explicit null",
          "source_url" in without and without["source_url"] is None)

    check("every unified deadline has source_url key (nullable)",
          all("source_url" in d for d in unified))
    check("framework deadlines carry nullable source_url",
          all("source_url" in d for d in unified if d.get("framework_name")))

    # The /api/compliance-data handler json.dumps the unified list directly;
    # missing/None values must serialize cleanly.
    payload = json.dumps(unified)
    check("unified list is JSON-serializable with nulls",
          '"source_url": null' in json.dumps(without))
    check("source_url value survives full JSON round-trip",
          "https://cabforum.org/example-ballot" in payload)
finally:
    pki.DEADLINES.remove(test_entry)
    pki.DEADLINES.remove(no_url_entry)

print("== status is date-consistent (no stale hardcoded values) ==")
from datetime import datetime, timezone
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
unified = pki.get_all_deadlines_unified()
stale_upcoming = [d["id"] for d in unified
                  if d["status"] == "upcoming" and d["date"] < today]
stale_passed = [d["id"] for d in unified
                if d["status"] == "passed" and d["date"] > today]
check(f"no past-dated deadline reports 'upcoming' {stale_upcoming or ''}",
      not stale_upcoming)
check(f"no future-dated deadline reports 'passed' {stale_passed or ''}",
      not stale_passed)

print("== DIFF_SYSTEM_PROMPT documents new fields ==")
prompt = car.DIFF_SYSTEM_PROMPT
check("documents is_estimated", '"is_estimated"' in prompt)
check("documents source_url", '"source_url"' in prompt)
check("forbids guessing a specific day",
      "NEVER guess" in prompt and "explicitly states" in prompt)
check("month precision defaults to estimated",
      "last day of that month" in prompt)
check("quarter/year precision handled", "quarter or year" in prompt)
check("keeps do-not-hallucinate rule", "Do NOT hallucinate" in prompt)
check("keeps human-review rule", "flag for human review" in prompt)

print("== impact is prose, never a bare severity rating ==")
# Bare high/medium/low used to render literally on the frontend as
# "MAJOR IMPACT — high"; converted to prose 2026-07-20. isMajor must be
# stored explicitly wherever impact no longer encodes it.
unified = pki.get_all_deadlines_unified()
bare = [d["id"] for d in unified if d.get("impact") in ("high", "medium", "low")]
check("no deadline carries a bare severity word as impact", bare == [])
check("isMajor always present and boolean",
      all(isinstance(d.get("isMajor"), bool) for d in unified))

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
