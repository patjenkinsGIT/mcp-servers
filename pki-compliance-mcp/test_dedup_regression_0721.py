#!/usr/bin/env python3
"""Regression tests for the 2026-07-21 outstanding-list leak.

The 2026-07-21 weekly safety-net run re-proposed six topics whose verdicts
were given in the 2026-07-13 review session, and four leaked back into the
outstanding list. Two causes, both covered here:

1. reject_ids() mapped id -> signature only from the NEWEST pending file,
   which on 07-13 was an empty "Could not parse diff response" stub — and the
   backlog items being verdicted lived in older files anyway. All 10 rejects
   that day persisted as bare ids, which the next run's fresh ids sailed past.
2. The anchor signature space was too coarse for regulatory flags: every
   NIS2 story shared the "anchors:nis2" bucket, so a CJEU-referral flag
   silently collapsed into a transposition flag's outstanding row (looking
   suppressed when it wasn't), and rejecting one NIS2 story would have
   suppressed genuinely-unresolved NEEDS-SOURCE siblings.

Offline: no network, no Anthropic API. Same harness style as
test_gating_dedup.py.
"""
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import compliance_auto_refresh as car
import daily_email

FIXTURE = Path(__file__).parent / "fixtures" / "pending_updates_2026-07-21_regression.json"

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
    daily_email.DATA_DIR = d
    return d


def day(n: int) -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=n)).isoformat()


SIX = json.loads(FIXTURE.read_text())["needs_human_review"]
SIG = {it["id"]: car._review_sig(it) for it in SIX}

print("== anchor buckets for the 2026-07-21 six ==")
check("csc32 discrepancy anchors on the ballot family",
      SIG["cabf-csc32-status-discrepancy"] == "anchors:csc32")
check("DORA CTPP list gets a ctpp qualifier",
      SIG["dora-ctpp-list-published"] == "anchors:ctpp+dora")
check("Czechia/Sweden transposition gets a transposition qualifier",
      SIG["nis2-czechia-sweden-transposition"] == "anchors:nis2+transposition")
# cjeu is a dominant anchor as of 2026-07-31: the co-occurring regulation
# anchors used to ride along in the signature (cjeu+nis2+transposition vs
# cjeu+dora+nis2+transposition), which split one story across four ids.
check("CJEU referral collapses to the durable cjeu bucket",
      SIG["nis2-cjeu-referral-laggard-states"] == "anchors:cjeu")
check("CJEU hold tokens still carry the co-occurring regulations",
      {"cjeu", "nis2", "transposition"}
      <= car.hold_anchors_for(
          [it for it in SIX if it["id"] == "nis2-cjeu-referral-laggard-states"][0]))
check("Commission simplification stays in its dora+nis2 bucket",
      SIG["nis2-commission-simplification-amendments"] == "anchors:dora+nis2")
check("UK ransomware bill anchors instead of falling back to text:",
      SIG["uk-cyber-extortion-ransomware-reporting-bill"] == "anchors:ransomware")
check("CJEU referral no longer shares a bucket with the transposition flag",
      SIG["nis2-cjeu-referral-laggard-states"]
      != SIG["nis2-czechia-sweden-transposition"])
check("all six occupy distinct buckets", len(set(SIG.values())) == 6)

print("== re-worded re-proposals land in the same buckets ==")
check("re-worded CJEU referral matches",
      car._review_sig({"id": "nis2-cjeu-daily-fines",
                       "description": "Commission referral of Ireland, Spain, France and "
                                      "the Netherlands to the CJEU over NIS2 transposition "
                                      "failures, seeking daily fines"})
      == SIG["nis2-cjeu-referral-laggard-states"])
check("re-worded CTPP list matches (singular CTPP)",
      car._review_sig({"id": "dora-critical-ict-provider-list",
                       "description": "ESAs' first CTPP designation list under DORA; "
                                      "needs official announcement URL"})
      == SIG["dora-ctpp-list-published"])
check("re-worded ransomware bill matches",
      car._review_sig({"id": "uk-ransomware-reporting-second-reading",
                       "description": "UK Cyber Extortion and Ransomware (Reporting) Bill "
                                      "second reading; private member's bill"})
      == SIG["uk-cyber-extortion-ransomware-reporting-bill"])
check("'transposed' verb form folds to the transposition anchor",
      car._review_sig({"id": "x", "description": "Sweden transposed NIS2 into national law"})
      == SIG["nis2-czechia-sweden-transposition"])
check("a generic NIS2 flag does NOT collide with the transposition bucket",
      car._review_sig({"id": "x", "description": "New NIS2 implementation guidance published"})
      != SIG["nis2-czechia-sweden-transposition"])

print("== reject_ids maps signatures across the recent window ==")
d = with_tmp()
(d / f"pending_updates_{day(8)}.json").write_text(json.dumps({
    "needs_human_review": [{"id": "review-csc-32-status",
                            "description": "CSC-32 adoption status conflict"}],
    "new_deadlines": [{"id": "some-deadline", "title": "Some Deadline", "date": "2026-09-01"}],
}))
# The newest file is an empty parse-failure stub — exactly the 2026-07-13 state.
(d / f"pending_updates_{day(0)}.json").write_text(json.dumps({
    "summary": "Could not parse diff response",
    "needs_human_review": [], "new_deadlines": [], "updated_deadlines": [],
    "regulatory_updates": [], "document_version_updates": [],
}))
car.reject_ids(["review-csc-32-status", "some-deadline"])
saved = json.loads(car.REJECTED_FILE.read_text())
check("review-flag signature captured from an OLDER file despite empty newest stub",
      "anchors:csc32" in saved["signatures"])
check("proposal signature captured from an older file too",
      "some deadline|2026-09-01" in saved["signatures"])
car.reject_ids(["never-seen-anywhere"])
saved = json.loads(car.REJECTED_FILE.read_text())
check("unmapped id still recorded (id-only)", "never-seen-anywhere" in saved["ids"])
check("unmapped id adds no bogus signature", len(saved["signatures"]) == 2)

d = with_tmp()
(d / "pending_updates_2026-01-05.json").write_text(json.dumps({
    "needs_human_review": [{"id": "ancient-flag", "description": "CSC-32 something"}]}))
car.reject_ids(["ancient-flag"])
saved = json.loads(car.REJECTED_FILE.read_text())
check("files outside the window are not consulted", saved["signatures"] == [])

print("== end-to-end replay of the six-item incident ==")
d = with_tmp()
TODAY = day(0)
(d / f"pending_updates_{day(8)}.json").write_text(FIXTURE.read_text())
# Manual hold, must survive everything below (suppression never lifts a hold).
(d / "review_holds.json").write_text(json.dumps({"smc17": day(-30)}))
# The 07-13 session's four hard REJECTs; the two NIS2 NEEDS-SOURCE items are
# deliberately NOT rejected — unresolved is not handled.
car.reject_ids([
    "cabf-csc32-status-discrepancy",
    "dora-ctpp-list-published",
    "nis2-cjeu-referral-laggard-states",
    "uk-cyber-extortion-ransomware-reporting-bill",
])
saved = json.loads(car.REJECTED_FILE.read_text())
check("all four hard rejects persisted with topic signatures",
      {"anchors:csc32", "anchors:ctpp+dora", "anchors:cjeu",
       "anchors:ransomware"} <= set(saved["signatures"]))

# Next safety-net run re-proposes all six topics, re-worded, under fresh ids.
reproposed = {
    "new_deadlines": [], "updated_deadlines": [],
    "document_version_updates": [], "regulatory_updates": [],
    "needs_human_review": [
        {"id": "csc32-ipr-status-conflict",
         "description": "Ballot CSC-32 status remains ambiguous between IPR review and adopted Code Signing BR v3.11"},
        {"id": "dora-critical-ict-provider-list",
         "description": "ESAs' list of designated critical ICT third-party providers (CTPPs) under DORA; needs official source"},
        {"id": "nis2-cz-se-national-laws",
         "description": "Czechia and Sweden NIS2 transposition laws reportedly in force; still lacking primary sources"},
        {"id": "nis2-cjeu-daily-fines",
         "description": "Commission referral of Ireland, Spain, France and the Netherlands to the CJEU over NIS2 transposition failures"},
        {"id": "nis2-simplification-omnibus",
         "description": "Commission amendments simplifying NIS2 compliance and aligning incident reporting across NIS2/DORA/GDPR"},
        {"id": "uk-ransomware-reporting-second-reading",
         "description": "UK Cyber Extortion and Ransomware (Reporting) Bill second reading scheduled"},
    ],
}
filtered, removed = car.dedup_changes(reproposed, None, exclude_date=TODAY)
reasons = {iid: why for _, iid, why in removed}
check("no re-proposal reaches the new pending file",
      filtered["needs_human_review"] == [])
check("the four verdicted topics are dropped as previously rejected",
      all(reasons.get(i) == "previously rejected" for i in (
          "csc32-ipr-status-conflict", "dora-critical-ict-provider-list",
          "nis2-cjeu-daily-fines", "uk-ransomware-reporting-second-reading")))
check("the two NEEDS-SOURCE topics are dropped only as re-flagged (still live)",
      all((reasons.get(i) or "").startswith("re-flagged") for i in (
          "nis2-cz-se-national-laws", "nis2-simplification-omnibus")))

# The outstanding list must show exactly the two unresolved NEEDS-SOURCE items.
class _NoNet:
    @staticmethod
    def get(*a, **k):
        raise OSError("offline test")
_real_httpx = daily_email.httpx
daily_email.httpx = _NoNet
try:
    outstanding = daily_email.outstanding_review_items(TODAY)
finally:
    daily_email.httpx = _real_httpx
titles = sorted(o["title"] for o in outstanding)
check("outstanding shows exactly the two unresolved NIS2 items",
      titles == ["nis2-commission-simplification-amendments",
                 "nis2-czechia-sweden-transposition"])
check("outstanding keeps the original first-seen date",
      all(o["first_seen"] == day(8) for o in outstanding))
check("manual hold survives the rejects (suppression never lifts a hold)",
      "smc17" in car.held_review_anchors())

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
