#!/usr/bin/env python3
"""Offline tests for auto_approve.py tiered classification and source patching.

No network, no git, no service restarts — pure classify/patch logic.
"""
import json
import py_compile
import tempfile
from pathlib import Path

import compliance_auto_refresh as car
import auto_approve as aa

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


d = Path(tempfile.mkdtemp())
car.DATA_DIR = d
car.HOLDS_FILE = d / "review_holds.json"  # isolate from the host's real manual pins
car.REJECTED_FILE = d / "rejected.json"
car.REJECTED_FILE.write_text(json.dumps({"ids": ["rejected-1"], "signatures": []}))

CURRENT = {
    "deadlines": [{"id": "existing-1", "title": "Existing", "date": "2027-01-01"}],
    "cabfDocuments": [{"id": "tls-br", "version": "2.2.6"}],
}


def entry(**over):
    base = {
        "id": "new-ballot-deadline", "date": "2027-06-15",
        "title": "New Ballot Deadline", "description": "Something new.",
        "source": "cab-forum", "category": "certificates", "isMajor": False,
        "source_url": "https://cabforum.org/2026/07/01/ballot-sc-100-something/",
        "feed_confirmed": True,
    }
    base.update(over)
    return base


print("== source_url allowlist ==")
check("cabforum allowed", aa.source_url_allowed("https://cabforum.org/x/"))
check("subdomain allowed", aa.source_url_allowed("https://knowledge.digicert.com/alerts/x"))
check("github only under /cabforum/", aa.source_url_allowed("https://github.com/cabforum/servercert/pull/1"))
check("github elsewhere rejected", not aa.source_url_allowed("https://github.com/evil/repo"))
check("random blog rejected", not aa.source_url_allowed("https://some-pki-blog.com/post"))
check("http (not https) rejected", not aa.source_url_allowed("http://cabforum.org/x/"))

print("== classify: happy path ==")
changes = {"new_deadlines": [entry()], "document_version_updates": [{"id": "ev-guidelines", "new_version": "2.0.3"}],
           "updated_deadlines": [], "regulatory_updates": [], "needs_human_review": []}
auto_new, auto_docs, review, skipped = aa.classify(changes, CURRENT, 5)
check("clean new deadline auto-approves", [x["id"] for x in auto_new] == ["new-ballot-deadline"])
check("doc bump auto-approves", [x["id"] for x in auto_docs] == ["ev-guidelines"])
check("nothing queued", not review and not skipped)

print("== classify: gates ==")
cases = [
    ("missing feed_confirmed queues", entry(feed_confirmed=False)),
    ("no source_url queues", entry(source_url=None)),
    ("off-allowlist source queues", entry(source_url="https://pki-news.example.com/a")),
    ("missing required field queues", entry(description=None)),
    ("bad date format queues", entry(date="March 2027")),
]
for name, item in cases:
    a, _, r, s = aa.classify({"new_deadlines": [item]}, CURRENT, 5)
    check(name, not a and len(r) == 1 and not s)

a, _, r, s = aa.classify({"new_deadlines": [entry(id="existing-1", title="Existing", date="2027-01-01")]}, CURRENT, 5)
check("duplicate of live data skips (not queued)", not a and not r and len(s) == 1)
a, _, r, s = aa.classify({"new_deadlines": [entry(id="rejected-1")]}, CURRENT, 5)
check("previously rejected skips", not a and not r and len(s) == 1)

print("== classify: guards ==")
a, _, r, _ = aa.classify({"new_deadlines": [entry(), entry(title="Same id other content")]}, CURRENT, 5)
check("conflict guard queues both", not a and len(r) == 2)
many = [entry(id=f"new-{i}", title=f"New {i}") for i in range(6)]
a, dv, r, _ = aa.classify({"new_deadlines": many}, CURRENT, 5)
check("daily cap queues everything", not a and not dv and len(r) == 6)
a, _, r, _ = aa.classify({"updated_deadlines": [{"id": "existing-1", "date": "2027-02-01"}]}, CURRENT, 5)
check("updates always queue", not a and len(r) == 1)
a, _, r, _ = aa.classify({"document_version_updates": [{"id": "tls-br", "new_version": "2.2.6"}]}, CURRENT, 5)
check("already-current doc version skipped", not a and not r)

print("== classify: open review hold blocks auto-apply (2026-07-12 regression) ==")
from datetime import datetime, timezone
TODAY = datetime.now(timezone.utc).date().isoformat()
(d / f"pending_updates_{TODAY}.json").write_text(json.dumps({
    "needs_human_review": [
        {"id": "cabf-sc0101v2-adn-transition",
         "description": "Ballot SC0101v2 passed but IPR review has not completed"},
        {"id": "rejected-hold", "description": "Ballot CSC-32 something",
         },
    ],
}))
held_item = entry(id="sc0101v2-adn-mandatory", title="SC0101v2 ADN Rules Become Mandatory",
                  description="Ballot SC0101v2 transition ends.", date="2026-11-15",
                  source_url="https://cabforum.org/2026/07/01/ballot-sc0101v2-clarify-authorization-domain-names/")
a, _, r, s = aa.classify({"new_deadlines": [held_item]}, CURRENT, 5)
check("held-topic proposal queues instead of auto-applying",
      not a and len(r) == 1 and "review hold" in r[0][2])
# The hold lifts when the flag is rejected (handled) — same proposal then applies.
car.REJECTED_FILE.write_text(json.dumps(
    {"ids": ["rejected-1", "cabf-sc0101v2-adn-transition"], "signatures": []}))
a, _, r, s = aa.classify({"new_deadlines": [held_item]}, CURRENT, 5)
check("rejected flag no longer holds the topic", [x["id"] for x in a] == ["sc0101v2-adn-mandatory"])
car.REJECTED_FILE.write_text(json.dumps({"ids": ["rejected-1"], "signatures": []}))
(d / f"pending_updates_{TODAY}.json").unlink()

print("== classify: doc bump held by a prose-named flag (2026-07-29/07-31 regression) ==")
# The flag names the document in prose and carries no ballot code at all, so
# its signature is a "text:" one — ballot anchors alone never connected it to
# the "[doc bump] ev-guidelines" proposal and the bump shipped under the hold.
(d / f"pending_updates_{TODAY}.json").write_text(json.dumps({
    "needs_human_review": [
        {"id": "cabf-ev-guidelines-version-discrepancy",
         "title": "EV Guidelines Version Discrepancy (v2.0.3 vs v2.0.2)",
         "description": "Sources disagree on the current EV Guidelines version."},
    ],
}))
EV_BUMP = {"id": "ev-guidelines", "new_version": "2.0.3"}
a, dv, r, s = aa.classify({"document_version_updates": [EV_BUMP]}, CURRENT, 5)
check("doc bump queues under a prose-named review hold",
      not dv and len(r) == 1 and "review hold" in r[0][2])
check("hold reason names the document token", "evguidelines" in r[0][2])
# An unrelated document is untouched by the EV hold.
a, dv, r, s = aa.classify(
    {"document_version_updates": [{"id": "smime-br", "new_version": "1.0.9"}]}, CURRENT, 5)
check("unrelated doc bump still auto-approves under an EV hold",
      [x["id"] for x in dv] == ["smime-br"] and not r)
# Rejecting the flag (verdict given) lifts the hold; the bump then applies.
car.REJECTED_FILE.write_text(json.dumps(
    {"ids": ["rejected-1", "cabf-ev-guidelines-version-discrepancy"], "signatures": []}))
a, dv, r, s = aa.classify({"document_version_updates": [EV_BUMP]}, CURRENT, 5)
check("rejected flag no longer holds the doc bump", [x["id"] for x in dv] == ["ev-guidelines"])
car.REJECTED_FILE.write_text(json.dumps({"ids": ["rejected-1"], "signatures": []}))

# A ballot flag that names the document in passing holds it too.
(d / f"pending_updates_{TODAY}.json").write_text(json.dumps({
    "needs_human_review": [
        {"id": "review-smc017v2-key-sizes",
         "description": "Ballot SMC017v2 amends the S/MIME Baseline Requirements; IPR review open"},
    ],
}))
a, dv, r, s = aa.classify(
    {"document_version_updates": [{"id": "smime-br", "new_version": "1.0.9"}]}, CURRENT, 5)
check("ballot flag naming S/MIME BR holds the smime-br bump",
      not dv and len(r) == 1 and "smimebr" in r[0][2])
# Deadlines keep the narrower ballot-anchor gate — a document token must not
# queue every deadline that merely cites the document.
a, dv, r, s = aa.classify({"new_deadlines": [entry(
    id="smime-br-citing-deadline", title="New S/MIME BR Requirement",
    description="Applies under the S/MIME Baseline Requirements.")]}, CURRENT, 5)
check("deadline citing a held document is not held by the document token",
      [x["id"] for x in a] == ["smime-br-citing-deadline"])
(d / f"pending_updates_{TODAY}.json").unlink()

print("== patch_source round-trip ==")
mini = '''"""mini"""
DEADLINES = [
    {
        "id": "old-entry",
        "date": "2026-01-01",
        "title": "Old",
        "description": "Old entry.",
        "source": "cab-forum",
        "category": "certificates",
        "isMajor": False,
    },
]
CABF_DOCUMENTS = [
    {"id": "tls-br", "name": "TLS BR", "version": "2.2.6", "date": "Mar 2026"},
]
DATA_FRESHNESS = {
    "fieldVerifications": {
        "deadlines": {"verified": "2026-05-14", "source": "x"},
    },
}
COMPLIANCE_METADATA = {"lastUpdated": "2026-05-14"}
'''
p = d / "mini_module.py"
p.write_text(mini)
applied = aa.patch_source(p, [entry()], [{"id": "tls-br", "new_version": "2.2.7", "new_date": "Jul 2026"}])
py_compile.compile(str(p), doraise=True)
ns = {}
exec(p.read_text(), ns)
ids = [x["id"] for x in ns["DEADLINES"]]
check("new entry inserted", "new-ballot-deadline" in ids and "old-entry" in ids)
new = [x for x in ns["DEADLINES"] if x["id"] == "new-ballot-deadline"][0]
check("source_url carried into entry", new["source_url"].startswith("https://cabforum.org/"))
check("feed_confirmed stripped from entry", "feed_confirmed" not in new)
check("doc version bumped", ns["CABF_DOCUMENTS"][0]["version"] == "2.2.7")
check("doc date bumped", ns["CABF_DOCUMENTS"][0]["date"] == "Jul 2026")
check("lastUpdated bumped", ns["COMPLIANCE_METADATA"]["lastUpdated"] != "2026-05-14")
check("deadlines verification bumped", ns["DATA_FRESHNESS"]["fieldVerifications"]["deadlines"]["verified"] != "2026-05-14")
check("applied ops itemized", len(applied) == 2)

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
