#!/usr/bin/env python3
"""Offline tests for loud diff-parse-failure reporting (2026-08-08 decision).

Acceptance condition, verbatim: a run that extracts nothing from an
unparseable response must not render as a clean run anywhere in the email.
Until this change it did — analyze_diff's degrade path wrote a review file
and logged no ERROR line, which is exactly daily_email's "ran clean, review
file written" signature (rendered clean on 2026-08-08 above a run that
extracted zero proposals from a 449-char truncated response).

Also locks the envelope instrumentation: stop_reason/output_tokens are logged
on every diff call and written into the raw-file header on failure — the
2026-08-02/06/08 failures were undiagnosable precisely because stop_reason
was discarded. No network, no Anthropic API, no Docker.
"""
import json
import tempfile
from pathlib import Path

import compliance_auto_refresh as car
import daily_email as de

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
    car.HOLDS_FILE = d / "review_holds.json"
    return d


DATE = "2026-08-08"

# The real 2026-08-08 failure shape (current marker format).
MARKER = ("Diff response unparseable at 2026-08-08T102652Z — 449 chars, "
          "no proposals could be extracted from this run. "
          "Full text: /root/.pki-compliance-mcp/diff_raw_2026-08-08T102652Z.txt "
          "First 400 chars: ```json { \"new_deadlines\": [ ...")
FAILED_PENDING = {
    "summary": "Could not parse diff response — no JSON object found",
    "needs_human_review": [{"id": None, "description": MARKER}],
    "new_deadlines": [], "updated_deadlines": [], "regulatory_updates": [],
    "document_version_updates": [], "content_candidates": [],
}
# The real 2026-08-02 failure shape: pre-marker code, summary is the only
# signal (the marker was an empty raw response, sanitized away).
OLD_SHAPE_PENDING = {
    "summary": "Could not parse diff response",
    "needs_human_review": [], "new_deadlines": [], "updated_deadlines": [],
    "regulatory_updates": [], "document_version_updates": [],
}
CLEAN_PENDING = {
    "summary": "No changes found",
    "needs_human_review": [], "new_deadlines": [], "updated_deadlines": [],
    "regulatory_updates": [], "document_version_updates": [],
    "content_candidates": [],
}

RAN_CLEAN = {"ran": True, "errors": [], "rate_limited_count": 0,
             "review_file_written": True, "skipped": False, "skip_reason": "",
             "in_progress": False}
SKIPPED = {"ran": True, "errors": [], "rate_limited_count": 0,
           "review_file_written": False, "skipped": True,
           "skip_reason": "no document changes; 0.8d since last research",
           "in_progress": False}
DOC_OK = {"ran": True, "checked_at": "2026-08-08T10:30:03",
          "changes_detected": 0, "docs": [], "errors": []}

print("== diff_parse_failure detection ==")
check("current failure shape (marker item) detected",
      de.diff_parse_failure(FAILED_PENDING) == MARKER)
check("2026-08-02 old shape (summary only) detected",
      bool(de.diff_parse_failure(OLD_SHAPE_PENDING)))
check("clean pending -> None", de.diff_parse_failure(CLEAN_PENDING) is None)
check("missing pending file -> None", de.diff_parse_failure(None) is None)
check("unreadable pending file -> None (its own warning already renders)",
      de.diff_parse_failure({"_parse_error": "boom"}) is None)
check("marker prefix stays in sync with compliance_auto_refresh",
      MARKER.startswith(car._DIFF_FAILURE_MARKER_PREFIX))

print("== acceptance: failed parse never renders clean, HTML ==")
html = de.render_html(DATE, FAILED_PENDING, DOC_OK, RAN_CLEAN)
check("research cron never renders clean", "ran clean, review file written" not in html)
check("cron status says FAILED", "FAILED — diff response unparseable" in html)
check("marker detail included", "449 chars" in html)
html_old = de.render_html("2026-08-02", OLD_SHAPE_PENDING, DOC_OK, RAN_CLEAN)
check("old-shape failure also never renders clean",
      "ran clean, review file written" not in html_old)
html_clean = de.render_html(DATE, CLEAN_PENDING, DOC_OK, RAN_CLEAN)
check("genuinely clean run still renders clean",
      "ran clean, review file written" in html_clean)

print("== acceptance: skip-day edge (failed manual run + scheduled skip) ==")
html_skip = de.render_html(DATE, FAILED_PENDING, DOC_OK, SKIPPED)
check("failure dominates the skip line", "FAILED — diff response unparseable" in html_skip)
check("skip fact still attributed to the scheduled run",
      "scheduled run skipped" in html_skip)
html_skip_clean = de.render_html(DATE, None, DOC_OK, SKIPPED)
check("normal skip day unchanged", "research skipped" in html_skip_clean)

print("== acceptance: text render ==")
text = de.render_text(DATE, FAILED_PENDING, DOC_OK, RAN_CLEAN)
check("text status line says FAILED",
      "10:00 UTC research cron: FAILED — diff response unparseable" in text)
check("text includes marker detail", "449 chars" in text)
text_clean = de.render_text(DATE, CLEAN_PENDING, DOC_OK, RAN_CLEAN)
check("clean text status line unchanged", "10:00 UTC research cron: ran" in text_clean)

print("== cron.log signal: ERROR line flips the generic status parser ==")
LOG_FAIL = [
    "[2026-08-08T10:00:01+00:00] PKI Compliance Auto-Refresh starting - 2026-08-08",
    "[2026-08-08T10:00:04+00:00] Research gate: RUN — 1 document(s) changed: cabf_br",
    "[2026-08-08T10:26:52+00:00] ERROR: diff parse failed — no proposals extracted this run (no JSON object found; stop_reason=end_turn, output_tokens=112)",
    "[2026-08-08T10:26:52+00:00] Review file written: /root/.pki-compliance-mcp/pending_updates_2026-08-08.json",
]
ar = de.auto_refresh_summary(LOG_FAIL)
check("ERROR line counted", len(ar["errors"]) == 1)
check("not classified in_progress", not ar["in_progress"])
html_log = de.render_html(DATE, None, DOC_OK, ar)
check("even with no pending file, log signal alone renders 'ran with errors'",
      "ran with errors" in html_log
      and "ran clean, review file written" not in html_log)

print("== analyze_diff: instrumentation + degrade, truncated response ==")
d = with_tmp()


class _FakeResp:
    def __init__(self, text):
        self._text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"content": [{"type": "text", "text": self._text}],
                "stop_reason": "max_tokens", "usage": {"output_tokens": 112}}


class _FakeClient:
    payload = ""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, *a, **k):
        return _FakeResp(_FakeClient.payload)


_real_client = car.httpx.Client
car.httpx.Client = _FakeClient
try:
    # No closing brace at all — the 2026-08-08 signature.
    _FakeClient.payload = '```json\n{\n  "new_deadlines": [\n    {\n      "id": "x'
    out = car.analyze_diff({"q1": "findings"}, None)
    check("degraded return carries the failure marker",
          any(str(m).startswith(car._DIFF_FAILURE_MARKER_PREFIX)
              for m in out["needs_human_review"]))
    log_text = (d / "auto_refresh.log").read_text()
    check("ERROR line logged", "ERROR: diff parse failed" in log_text)
    check("stop_reason logged on the diff call", "stop_reason=max_tokens" in log_text)
    check("output_tokens logged", "output_tokens=112" in log_text)
    raws = list(d.glob("diff_raw_*.txt"))
    check("raw response persisted", len(raws) == 1)
    check("raw-file header names stop_reason",
          "stop_reason=max_tokens" in raws[0].read_text())

    # Braces present but invalid JSON — the 2026-08-06 signature.
    _FakeClient.payload = '{"new_deadlines": [{"id": "x", }, }]}'
    out2 = car.analyze_diff({"q1": "findings"}, None)
    check("malformed-JSON branch also degrades with marker",
          any(str(m).startswith(car._DIFF_FAILURE_MARKER_PREFIX)
              for m in out2["needs_human_review"]))
    check("malformed-JSON branch also logs ERROR",
          (d / "auto_refresh.log").read_text().count("ERROR: diff parse failed") == 2)

    # A clean response must not trip any of it.
    _FakeClient.payload = json.dumps({
        "new_deadlines": [], "updated_deadlines": [],
        "document_version_updates": [], "regulatory_updates": [],
        "needs_human_review": [], "content_candidates": [],
        "summary": "No changes found"})
    out3 = car.analyze_diff({"q1": "findings"}, None)
    check("clean response parses", out3["summary"] == "No changes found")
    check("clean response logs no new ERROR",
          (d / "auto_refresh.log").read_text().count("ERROR: diff parse failed") == 2)
    check("envelope metadata logged on clean calls too",
          (d / "auto_refresh.log").read_text().count("stop_reason=max_tokens") >= 3)
finally:
    car.httpx.Client = _real_client

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
