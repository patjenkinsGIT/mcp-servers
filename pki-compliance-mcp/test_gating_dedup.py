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
    car.HOLDS_FILE = d / "review_holds.json"
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

print("== sanitize_changes ==")
with_tmp()
changes = {
    "new_deadlines": [
        {"id": "good", "title": "Good", "date": "2026-09-01"},
        {"id": "no-title"},                        # drop: no title
        {},                                        # drop: empty
        "not-a-dict",                              # drop: wrong type
    ],
    "updated_deadlines": [{"id": "upd-ok"}, {"title": "no id"}],
    "regulatory_updates": [
        {"id": "reg-ok", "title": "Reg", "description": "d"},
        {"id": "reg-empty"},                       # drop: no title/description
    ],
    "document_version_updates": [
        {"id": "tls-br", "new_version": "9.9.9"},
        {"id": "no-version"},                      # drop
    ],
    "needs_human_review": [
        {"id": "flag-ok", "description": "desc", "reason": "r"},
        "bare string flag",                        # wrapped into a dict
        {},                                        # drop: empty
    ],
}
sanitized, dropped = car.sanitize_changes(changes)
check("keeps valid new deadline", [x["id"] for x in sanitized["new_deadlines"]] == ["good"])
check("keeps id-only update", [x["id"] for x in sanitized["updated_deadlines"]] == ["upd-ok"])
check("drops empty regulatory", [x["id"] for x in sanitized["regulatory_updates"]] == ["reg-ok"])
check("drops versionless doc bump", [x["id"] for x in sanitized["document_version_updates"]] == ["tls-br"])
check("wraps bare-string review flag",
      any(x.get("description") == "bare string flag" for x in sanitized["needs_human_review"]))
check("drops empty review flag", len(sanitized["needs_human_review"]) == 2)
# 3 new_deadlines + 1 update + 1 regulatory + 1 doc bump + 1 review flag
check("dropped log has 7 entries", len(dropped) == 7)

print("== review-flag dedup across days ==")
d = with_tmp()
from datetime import datetime, timedelta, timezone
TODAY = datetime.now(timezone.utc).date().isoformat()
YDAY = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
# Yesterday's file flagged SC087v2 and CSC-32 under one set of ids…
(d / f"pending_updates_{YDAY}.json").write_text(json.dumps({
    "needs_human_review": [
        {"id": "review-sc087v2-ev-guidelines-serialnumber",
         "description": "Ballot SC087v2 passed, in IPR review"},
        {"id": "review-csc-32-code-signing-oid-ballot",
         "description": "Ballot CSC-32 passed, IPR review ends mid-July"},
    ],
}))
# …today the model re-flags the same topics under brand-new ids.
changes = {
    "new_deadlines": [], "updated_deadlines": [],
    "document_version_updates": [], "regulatory_updates": [],
    "needs_human_review": [
        {"id": "sc087v2-ev-registration-number-ipr-pending",
         "description": "SC087v2 remains in IPR Review Period"},
        {"id": "csc-32-mandatory-policy-oid-ipr-pending",
         "description": "CSC-32 in its IPR Review Period"},
        {"id": "totally-new-thing",
         "description": "A brand new unrelated concern about OCSP responders"},
    ],
}
filtered, removed = car.dedup_changes(changes, None, exclude_date=TODAY)
kept_ids = [x["id"] for x in filtered["needs_human_review"]]
check("re-flagged SC087v2 dropped despite new id",
      "sc087v2-ev-registration-number-ipr-pending" not in kept_ids)
check("re-flagged CSC-32 dropped despite new id",
      "csc-32-mandatory-policy-oid-ipr-pending" not in kept_ids)
check("genuinely new flag kept", kept_ids == ["totally-new-thing"])
check("removal reasons cite first-seen date",
      all(f"first seen {YDAY}" in why for _, _, why in removed))
# Same-day file must be excluded, or a rerun would drop everything.
(d / f"pending_updates_{TODAY}.json").write_text(json.dumps({
    "needs_human_review": [{"id": "totally-new-thing",
                            "description": "A brand new unrelated concern about OCSP responders"}],
}))
changes["needs_human_review"] = [{"id": "totally-new-thing",
                                  "description": "A brand new unrelated concern about OCSP responders"}]
filtered, _ = car.dedup_changes(changes, None, exclude_date=TODAY)
check("today's own file excluded from dedup",
      [x["id"] for x in filtered["needs_human_review"]] == ["totally-new-thing"])

print("== review-flag anchor extraction (2026-07-10 regressions) ==")
# 4-digit zero-padded ballots must anchor (SC0101v2 fell through to text: sig)
sig_a = car._review_sig({"id": "cabf-sc0101v2-adn-transition",
                         "description": "Ballot SC0101v2 passed but IPR review incomplete"})
sig_b = car._review_sig({"id": "review-sc0101v2-adn-clarification",
                         "description": "SC0101v2 (Clarify Authorization Domain Names) reworded flag"})
check("SC0101v2 gets an anchor sig", sig_a.startswith("anchors:"))
check("SC0101v2 reworded flag matches", sig_a == sig_b)
# Space-separated references must anchor ("EO 14412" missed eo-?14412)
check("'EO 14412' with space anchors",
      car._review_sig({"id": "x", "description": "restates existing EO 14412"}).startswith("anchors:"))
check("'M-26-15' and 'M 26 15' agree",
      car._review_sig({"id": "x", "description": "OMB M-26-15 five-phase"})
      == car._review_sig({"id": "y", "description": "OMB memo M 26 15 schedule"}))
# Recurring un-anchored topics now have anchor classes
ms_a = car._review_sig({"id": "microsoft-kernel-driver-crosssign-removal",
                        "description": "removal of trust for kernel drivers signed by the deprecated cross-signed root program"})
ms_b = car._review_sig({"id": "review-microsoft-legacy-driver-trust-removal",
                        "description": "Microsoft reportedly announced removal of trust for kernel drivers, cross-signed, uptime logic change"})
check("MS driver-signing flag gets anchor sig", ms_a.startswith("anchors:"))
check("MS driver-signing reworded flag matches", ms_a == ms_b)
check("SC101v2 and SC0101v2 canonicalize to the same anchor",
      car._review_sig({"id": "review-sc101v2-authorization-domain-names", "description": ""})
      == car._review_sig({"id": "cabf-sc0101v2-adn-transition", "description": ""}))
check("UK CSR spelled-out and abbreviated forms share an anchor",
      car._review_sig({"id": "x", "description": "Cyber Security and Resilience Bill stage"})
      == car._review_sig({"id": "review-uk-csr-bill-stage", "description": ""}))
check("SMC017 and SMC017v2 canonicalize to the same anchor",
      car._review_sig({"id": "x", "description": "Ballot SMC017 passed"})
      == car._review_sig({"id": "y", "description": "SMC017v2 in IPR review"}))
check("Apple root program flag anchors",
      car._review_sig({"id": "review-apple-root-program-github-migration",
                       "description": "Apple Root Program policy publication moved to GitHub"}).startswith("anchors:"))
check("hyphenated apple id alone anchors",
      car._review_sig({"id": "apple-root-program-url-change",
                       "description": "policy URL should be updated"}).startswith("anchors:"))
check("dedicated-TLS flag anchors and matches reworded",
      car._review_sig({"id": "review-chrome-v1-8-dedicated-tls-phaseout-start",
                       "description": "phasing out dedicated TLS violators"})
      == car._review_sig({"id": "chrome-dedicated-tls-enforcement-start",
                          "description": "Chrome dedicated-TLS hierarchy enforcement began June 15"}))

print("== reject_ids persistence ==")
d = with_tmp()
# Relative date: reject_ids only maps signatures from the last 30 days of
# pending files (a hardcoded date here aged out of the window and went stale).
WEEK_AGO = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
(d / f"pending_updates_{WEEK_AGO}.json").write_text(json.dumps({
    "new_deadlines": [{"id": "foo", "title": "Foo Thing", "date": "2026-07-01"}]
}))
total = car.reject_ids(["foo"])
saved = json.loads(car.REJECTED_FILE.read_text())
check("foo persisted to rejected ids", "foo" in saved["ids"])
check("signature captured from pending file", "foo thing|2026-07-01" in saved["signatures"])
check("reject returns total count", total == 1)

print("== hold age-out + manual holds (2026-07-18) ==")
d = with_tmp()
OLD = (datetime.now(timezone.utc).date() - timedelta(days=15)).isoformat()
RECENT = (datetime.now(timezone.utc).date() - timedelta(days=13)).isoformat()
FLAG = {"needs_human_review": [
    {"id": "review-sc0101v2-ipr", "description": "SC0101v2 in IPR review"}]}
(d / f"pending_updates_{OLD}.json").write_text(json.dumps(FLAG))
check("flag-derived hold ages out after 14 days",
      "sc101" not in car.held_review_anchors())
(d / f"pending_updates_{RECENT}.json").write_text(json.dumps(FLAG))
check("flag within window holds", "sc101" in car.held_review_anchors())

d = with_tmp()
(d / f"pending_updates_{OLD}.json").write_text(json.dumps(FLAG))
FUTURE = (datetime.now(timezone.utc).date() + timedelta(days=10)).isoformat()
YDAY2 = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
car.HOLDS_FILE.write_text(json.dumps({
    "SC0101v2": FUTURE,       # spelled like the ballot, not the canonical token
    "smc17": FUTURE,
    "csc32": YDAY2,           # expired yesterday
    "dora": "not-a-date",     # typo must NOT silently lift the hold
}))
held = car.held_review_anchors()
check("manual hold outlives flag age-out", "sc101" in held)
check("manual hold anchor is canonicalized", "sc0101v2" not in held)
check("second manual hold present", "smc17" in held)
check("expired manual hold lifted", "csc32" not in held)
check("unparseable hold-until date keeps hold (fail safe)", "dora" in held)
d = with_tmp()
TODAY2 = datetime.now(timezone.utc).date().isoformat()
car.HOLDS_FILE.write_text(json.dumps({"nis2": TODAY2}))
check("hold-until is inclusive: expires end of that day",
      "nis2" in car.load_manual_holds())
with_tmp()
check("no holds file -> no manual holds", car.load_manual_holds() == set())

print("== tracked-document anchors (2026-07-29/07-31 regression) ==")
# A doc-bump proposal names the document by id; a flag names it in prose. Both
# must resolve to the same hold token, or the bump slips past an open flag.
check("doc id and prose name agree (ev-guidelines)",
      car.document_anchors({"id": "ev-guidelines", "new_version": "2.0.3"})
      == car.document_anchors({"id": "cabf-ev-guidelines-version-discrepancy",
                               "title": "EV Guidelines Version Discrepancy (v2.0.3 vs v2.0.2)"})
      == {"evguidelines"})
check("spelled-out EV name anchors",
      car.document_anchors({"description": "Extended Validation SSL Certificate Guidelines v2.0.3"})
      == {"evguidelines"})
DOC_CASES = [
    ("smimebr", "smime-br", ["S/MIME BR", "S/MIME Baseline Requirements", "SMIME BRs"]),
    ("tlsbr", "tls-br", ["TLS BR", "TLS Baseline Requirements",
                         "Server Certificate Baseline Requirements"]),
    ("codesigningbr", "code-signing-br", ["Code Signing BRs",
                                          "Code Signing Baseline Requirements"]),
    ("netsec", "netsec", ["Network Security Requirements", "NetSec"]),
]
for token, doc_id, prose in DOC_CASES:
    check(f"{doc_id} id anchors to {token}",
          car.document_anchors({"id": doc_id}) == {token})
    for name in prose:
        check(f"{doc_id} prose name {name!r} anchors",
              token in car.document_anchors({"description": f"the {name} were updated"}))
check("unrelated text yields no document anchor",
      car.document_anchors({"description": "Chrome Root Program policy update"}) == set())
check("document anchors stay out of the dedup signature",
      car._review_sig({"id": "review-sc087v2-ev-guidelines-serialnumber",
                       "description": "Ballot SC087v2 passed, in IPR review"})
      == car._review_sig({"id": "sc087v2-ev-registration-number-ipr-pending",
                          "description": "SC087v2 remains in IPR Review Period"}))

d = with_tmp()
(d / f"pending_updates_{RECENT}.json").write_text(json.dumps({"needs_human_review": [
    {"id": "cabf-ev-guidelines-version-discrepancy",
     "title": "EV Guidelines Version Discrepancy (v2.0.3 vs v2.0.2)",
     "description": "Sources disagree on the current EV Guidelines version."}]}))
held = car.held_review_anchors()
check("a prose-only flag with no ballot code still contributes a hold",
      "evguidelines" in held)
check("that flag holds only its own document", "tlsbr" not in held)
d = with_tmp()
car.HOLDS_FILE.write_text(json.dumps({"ev-guidelines": FUTURE}))
check("manual hold written as the doc id lands on the doc token",
      "evguidelines" in car.load_manual_holds())

print("== cjeu is a dominant anchor (recurred under 4 ids: 07-13/21/29/31) ==")
CJEU_A = car._review_sig({"id": "nis2-cjeu-referral-laggard-states",
                          "description": "Commission referral to the CJEU over NIS2 "
                                         "transposition failures"})
CJEU_B = car._review_sig({"id": "nis2-cjeu-daily-fines",
                          "description": "CJEU referral over NIS2 and DORA transposition, "
                                         "seeking daily fines"})
check("cjeu collapses to a single durable bucket", CJEU_A == "anchors:cjeu")
check("co-occurring dora no longer splits the bucket", CJEU_A == CJEU_B)
check("a non-CJEU NIS2 flag keeps its own bucket",
      car._review_sig({"id": "x", "description": "Sweden transposed NIS2"}) != CJEU_A)
d = with_tmp()
car.REJECTED_FILE.write_text(json.dumps({"ids": [], "signatures": [
    "anchors:cjeu+nis2+transposition",              # written before cjeu was dominant
    "anchors:cjeu+dora+nis2+transposition",
    "anchors:csc32",                                # untouched by the rule
    "some plain text sig|2026-07-01",
]}))
rej = car.load_rejected()
check("pre-existing cjeu rejections still match after the rule change",
      CJEU_A in rej["signatures"] and CJEU_B in rej["signatures"])
check("non-dominant stored signatures are untouched",
      {"anchors:csc32", "some plain text sig|2026-07-01"} <= rej["signatures"])
check("collapsing the signature does not narrow the hold set",
      {"cjeu", "nis2", "transposition"} <= car.hold_anchors_for(
          {"id": "nis2-cjeu-referral-laggard-states",
           "description": "CJEU referral over NIS2 transposition failures"}))

print("== deescalate_speculative_urgent ==")
# The verbatim 2026-08-20 item: a research description that called the event
# hypothetical still carried urgent:true into a five-channel content package.
_le = {"id": "letsencrypt-cpcps-missing-attestation-incident",
       "title": "Let's Encrypt CP/CPS Missing Chrome Root Program Attestation",
       "description": ("Let's Encrypt became aware its CP/CPS lacked the required "
                       "attestation of compliance with the Chrome Root Program Policy "
                       "and CCADB Policy by the 2026-06-15 effective date. Community "
                       "discussion (as of Aug 10-12, 2026) is debating whether this "
                       "triggers mandated certificate revocation under Chrome/CCADB "
                       "policy. This could constitute a mass-revocation event "
                       "affecting Let's Encrypt-issued certificates;"),
       "urgent": True}
# MUST keep its flag: the event IS announced; only its scope is uncertain.
_announced = {"id": "ca-x-mass-revoke", "title": "CA X announces mass revocation",
              "description": ("CA X has announced it will revoke 200,000 certificates "
                              "beginning 2026-09-01. It is unclear whether subscribers "
                              "using cross-signed intermediates are affected."),
              "urgent": True}
_distrust = {"id": "root-distrust", "title": "Root distrust",
             "description": "Chrome announced the distrust of Root Y effective 2026-10-01.",
             "urgent": True}
_changes = {"needs_human_review": [_le, _announced], "new_deadlines": [_distrust]}
_hits = car.deescalate_speculative_urgent(_changes)
check("speculative mass-revocation downgraded", _le["urgent"] is False)
check("downgrade records the matched phrase", _le.get("urgent_downgraded") == "debating whether")
check("announced event with uncertain SCOPE keeps urgent", _announced["urgent"] is True)
check("plain announced distrust keeps urgent", _distrust["urgent"] is True)
check("only the speculative item downgraded", len(_hits) == 1)
# The bounded-downside property that lets this run as code at all: the item is
# never moved or dropped, so it still reaches the review queue and the backlog.
check("downgraded item stays in its list", _le in _changes["needs_human_review"])
check("non-urgent items untouched",
      car.deescalate_speculative_urgent(
          {"needs_human_review": [{"id": "x", "description": "could constitute a problem"}]}) == [])

print("== research prompts carry today's date ==")
_today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
_p = car._dated("Search for CA/Browser Forum ballot results from the last 30 days.")
check("today's date injected", _today in _p)
check("original query preserved", "last 30 days" in _p)
check("latest-state instruction present", "LATEST state" in _p or "latest state" in _p.lower())

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
