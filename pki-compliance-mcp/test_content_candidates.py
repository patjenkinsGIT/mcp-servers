#!/usr/bin/env python3
"""Offline tests for the content-candidate state (Part A, 2026-08-07) and the
Part B provenance additions (primary_url, rejection-path retention).

No network, no Anthropic API, no Docker. The load-bearing properties:

  1. Emission scoping — the sink ledger suppresses CONTENT emission only.
     Nothing on the deadline path (dedup_changes, auto_approve.classify) reads
     it, so an item that later acquires a date certain becomes a deadline no
     matter what the ledger says.
  2. sink_status semantics — a candidate is suppressed only once "posted".
     A failed or impossible POST leaves the row "pending" and it retries; a
     sink failure never loses the item.
  3. The backstop classifier never eats pipeline-operational flags (ballot
     anchors, tracked-document names, diff-failure markers) or dated items.
"""
import json
import tempfile
from pathlib import Path

import compliance_auto_refresh as car
import content_candidate_sink as sink
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


TODAY = "2026-08-07"

print("== backstop rules: what MUST be classified ==")

eu_item = {"id": "eu-cyber-amendment", "description":
           "Proposal for a Regulation amending the CSA; trilogue negotiations ongoing"}
hit = car._content_rule_match(eu_item)
check("EU legislative procedure item fires eu-legislative-procedure",
      hit is not None and hit[0] == "eu-legislative-procedure")

cjeu_item = {"id": "cjeu-nis2-germany", "description":
             "Commission referred Germany to the Court of Justice over NIS2 transposition delays"}
hit = car._content_rule_match(cjeu_item)
check("CJEU referral fires enforcement-action",
      hit is not None and hit[0] == "enforcement-action")

ipd_item = {"id": "nist-sp-800-x", "description":
            "NIST released the initial public draft of new key management guidance"}
hit = car._content_rule_match(ipd_item)
check("NIST initial public draft fires draft-standard-comment-period",
      hit is not None and hit[0] == "draft-standard-comment-period")

bill_item = {"id": "uk-csr-bill", "description":
             "The Cyber Security and Resilience Bill completed its second reading in the House of Commons"}
hit = car._content_rule_match(bill_item)
check("pre-assent bill fires pre-royal-assent-bill",
      hit is not None and hit[0] == "pre-royal-assent-bill")

print("== backstop rules: what must NOT be classified ==")

check("'received Royal Assent' (adopted law) does not fire",
      car._content_rule_match({"id": "uk-csr-act", "description":
                               "The CSR Act received Royal Assent on 2026-07-01"}) is None)
check("ballot-anchored flag (SC-100) is excluded",
      car._content_rule_match({"id": "sc100-review", "description":
                               "SC-100 ballot passed first reading of IPR review"}) is None)
check("tracked-document flag (EV Guidelines) is excluded",
      car._content_rule_match({"id": "ev-mismatch", "description":
                               "EV Guidelines version discrepancy under committee stage bill review"}) is None)
check("item with a date certain is excluded",
      car._content_rule_match({"id": "x", "date": "2026-11-15", "description":
                               "transposition deadline set by proposal for a directive"}) is None)
check("diff-failure marker is excluded",
      car._content_rule_match({"id": None, "description":
                               "Diff response unparseable at 2026-08-07 — 0 chars"}) is None)
check("non-dict item is excluded", car._content_rule_match("a bare string") is None)

print("== classify_content_candidates plumbing ==")

changes = {
    "new_deadlines": [], "updated_deadlines": [], "document_version_updates": [],
    "regulatory_updates": [dict(cjeu_item)],
    "needs_human_review": [dict(eu_item),
                           {"id": "sc100-review", "description": "SC-100 IPR review open"}],
    "content_candidates": [{"id": "model-emitted", "description": "d",
                            "content_rule": "made-up-rule"}],
}
changes, moved = car.classify_content_candidates(changes)
check("EU item moved out of needs_human_review",
      all(it.get("id") != "eu-cyber-amendment" for it in changes["needs_human_review"]))
check("ballot flag stays in needs_human_review",
      any(it.get("id") == "sc100-review" for it in changes["needs_human_review"]))
check("CJEU item moved out of regulatory_updates", changes["regulatory_updates"] == [])
check("both moves recorded with rule names",
      len(moved) == 2 and {m[2] for m in moved} == {"eu-legislative-procedure", "enforcement-action"})
check("moved items carry content_rule + matched text",
      all(it.get("content_rule") and it.get("content_rule_matched")
          for it in changes["content_candidates"] if it.get("id") != "model-emitted"))
check("model-emitted bogus rule normalized to no-date-certain",
      next(it for it in changes["content_candidates"] if it.get("id") == "model-emitted")
      ["content_rule"] == "no-date-certain")
check("candidates list has all three", len(changes["content_candidates"]) == 3)

print("== sanitize handles content_candidates ==")

s_changes, dropped = car.sanitize_changes({
    "new_deadlines": [], "updated_deadlines": [], "document_version_updates": [],
    "regulatory_updates": [], "needs_human_review": [],
    "content_candidates": [
        "a bare string candidate",
        {"id": "ok", "description": "d", "provenance_urls": ["https://a.example", "bogus"]},
        {},
    ],
})
cc = s_changes["content_candidates"]
check("bare string coerced to dict", cc[0]["description"] == "a bare string candidate")
check("provenance normalized on candidates", cc[1]["provenance_urls"] == ["https://a.example"])
check("empty dict dropped", len(cc) == 2 and any(k == "content_candidates" for k, _, _ in dropped))

print("== primary_url normalization (Part B) ==")

it = car._normalize_provenance({"provenance_urls": ["https://a.example", "https://b.example"],
                                "primary_url": "https://b.example"})
check("primary kept when in list", it["primary_url"] == "https://b.example")
it = car._normalize_provenance({"provenance_urls": ["https://a.example"],
                                "primary_url": "https://c.example"})
check("valid primary missing from list is prepended",
      it["primary_url"] == "https://c.example" and it["provenance_urls"][0] == "https://c.example")
it = car._normalize_provenance({"provenance_urls": ["https://a.example"],
                                "primary_url": "the NIS2 press release"})
check("non-URL primary dropped", "primary_url" not in it)
it = car._normalize_provenance({"provenance_urls": ["https://a.example"]})
check("absent primary stays absent", "primary_url" not in it)
check("primary does not perturb _review_sig",
      car._review_sig({"id": "a", "description": "NIS2 Czechia transposition"})
      == car._review_sig({"id": "a", "description": "NIS2 Czechia transposition",
                          "primary_url": "https://x.example", "provenance_urls": ["https://x.example"]}))

print("== sink ledger: sink_status semantics ==")

with_tmp()
posted_calls = []


def fake_post_ok(payload, base, secret):
    posted_calls.append(payload)
    return "news-123"


def fake_post_fail(payload, base, secret):
    posted_calls.append(payload)
    raise RuntimeError("simulated 500")


real_post, real_config = sink._post_news, sink._news_config
sink._news_config = lambda: ("https://fixmycert.example", "secret")

cand = {"id": "eu-cyber-amendment", "title": "EU cyber amendment",
        "description": "Trilogue ongoing; provisional agreement expected autumn 2026",
        "content_rule": "eu-legislative-procedure",
        "provenance_urls": ["https://eur-lex.europa.eu/x"],
        "primary_url": "https://eur-lex.europa.eu/x"}

sink._post_news = fake_post_ok
stats = sink.process({"content_candidates": [dict(cand)]}, TODAY)
ledger = json.loads((car.DATA_DIR / "content_candidates.json").read_text())
row = next(iter(ledger.values()))
check("new candidate recorded and posted", stats["recorded"] == 1 and stats["posted"] == 1)
check("row is posted with news_id", row["sink_status"] == "posted" and row["news_id"] == "news-123")
check("payload is a draft with cluster + url",
      posted_calls[0]["status"] == "draft" and posted_calls[0]["newsCluster"] == "other"
      and posted_calls[0]["url"] == "https://eur-lex.europa.eu/x")
check("real date lives in the body text (API stamps ingest dates)",
      TODAY in posted_calls[0]["excerpt"])

posted_calls.clear()
stats = sink.process({"content_candidates": [dict(cand)]}, TODAY)
check("unchanged re-encounter suppressed, no second POST",
      stats["suppressed"] == 1 and stats["posted"] == 0 and posted_calls == [])

reworded = dict(cand, description="Trilogue continuing; provisional agreement now expected in the autumn")
stats = sink.process({"content_candidates": [reworded]}, TODAY)
check("text change within cooldown does not re-emit",
      stats["material_change"] == 0 and stats["suppressed"] == 1 and posted_calls == [])

advanced = dict(cand, description="Provisional political agreement reached; formal adoption vote scheduled")
stats = sink.process({"content_candidates": [advanced]}, "2026-08-20")
check("material change after cooldown re-emits",
      stats["material_change"] == 1 and stats["posted"] == 1)

print("== sink ledger: failure keeps the row pending ==")

with_tmp()
posted_calls.clear()
sink._post_news = fake_post_fail
stats = sink.process({"content_candidates": [dict(cand)]}, TODAY)
ledger = json.loads((car.DATA_DIR / "content_candidates.json").read_text())
row = next(iter(ledger.values()))
check("failed POST leaves row pending", row["sink_status"] == "pending" and stats["failed"] == 1)
check("attempt counted", row["attempts"] == 1)

sink._post_news = fake_post_ok
posted_calls.clear()
stats = sink.process({"content_candidates": []}, TODAY)
ledger = json.loads((car.DATA_DIR / "content_candidates.json").read_text())
row = next(iter(ledger.values()))
check("pending row retried on a later run with no new candidates",
      row["sink_status"] == "posted" and stats["posted"] == 1)

with_tmp()
sink._news_config = lambda: ("", "")
stats = sink.process({"content_candidates": [dict(cand)]}, TODAY)
ledger = json.loads((car.DATA_DIR / "content_candidates.json").read_text())
row = next(iter(ledger.values()))
check("missing env: recorded but pending, nothing lost",
      stats["recorded"] == 1 and row["sink_status"] == "pending" and stats["posted"] == 0)

sink._post_news, sink._news_config = real_post, real_config

print("== cluster mapping ==")
check("pqc terms map to pqc cluster", sink._cluster_for("NIST ML-KEM migration draft") == "pqc")
check("default is other", sink._cluster_for("EU cyber amendment trilogue") == "other")

print("== EMISSION SCOPING: ledger never touches the deadline path ==")

d = with_tmp()
# The same topic sits in the ledger as posted...
sink._news_config = lambda: ("https://fixmycert.example", "secret")
sink._post_news = fake_post_ok
sink.process({"content_candidates": [dict(cand)]}, TODAY)
sink._post_news, sink._news_config = real_post, real_config
# ...and later acquires a date certain, arriving as a new_deadline.
adopted = {"id": "eu-cyber-amendment-effective", "date": "2027-10-01",
           "title": "EU cyber amendment effective", "description": "Adopted; applies from 2027-10-01",
           "source": "eur-lex", "category": "regulation"}
deduped, removed = car.dedup_changes(
    {"new_deadlines": [dict(adopted)], "updated_deadlines": [],
     "document_version_updates": [], "regulatory_updates": [], "needs_human_review": []},
    current_data=None)
check("adopted item passes dedup_changes despite posted ledger row",
      len(deduped["new_deadlines"]) == 1 and removed == [])

import auto_approve
auto_new, auto_docs, review, skipped = auto_approve.classify(
    {"content_candidates": [dict(cand)], "new_deadlines": [], "updated_deadlines": [],
     "document_version_updates": [], "regulatory_updates": [], "needs_human_review": []},
    current_data=None, max_auto=5)
check("auto_approve.classify never sees content_candidates (no review queue entry)",
      auto_new == [] and auto_docs == [] and review == [] and skipped == [])

print("== rejection-path provenance retention (Part B) ==")

d = with_tmp()
pending_file = d / f"pending_updates_{TODAY}.json"
pending_file.write_text(json.dumps({
    "needs_human_review": [
        {"id": "nis2-czechia-followup",
         "description": "NIS2 Czechia transposition status unclear",
         "provenance_urls": ["https://nukib.gov.cz/en/news/", "https://example.com/trade-press"],
         "primary_url": "https://nukib.gov.cz/en/news/"},
    ],
}))
car.reject_ids(["nis2-czechia-followup"])
rej = json.loads(car.REJECTED_FILE.read_text())
snap = rej.get("provenance", {}).get("nis2-czechia-followup", {})
check("rejected id keeps its provenance URLs",
      snap.get("provenance_urls") == ["https://nukib.gov.cz/en/news/", "https://example.com/trade-press"])
check("rejected id keeps its primary URL", snap.get("primary_url") == "https://nukib.gov.cz/en/news/")
check("snapshot carries signature + rejection date",
      bool(snap.get("signature")) and bool(snap.get("rejected_at")))
check("load_rejected still returns ids and signatures",
      "nis2-czechia-followup" in car.load_rejected()["ids"])

d = with_tmp()
car.REJECTED_FILE.write_text(json.dumps({"ids": ["old-id"], "signatures": ["text:old thing"]}))
rej = car.load_rejected()
check("old-shape rejected.json (no provenance key) still loads",
      "old-id" in rej["ids"] and "text:old thing" in rej["signatures"])
car.reject_ids(["never-seen-id"])
rej = json.loads(car.REJECTED_FILE.read_text())
check("rejecting an id with no pending provenance adds no snapshot and keeps old data",
      "never-seen-id" in rej["ids"] and "old-id" in rej["ids"]
      and "never-seen-id" not in rej.get("provenance", {}))

print("== daily email surfacing ==")

d = with_tmp()
de.DATA_DIR = d
(d / "content_candidates.json").write_text(json.dumps({
    "anchors:transposition": {"sink_status": "posted", "news_id": "n1",
                              "title": "EU thing", "first_seen": TODAY, "rule": "eu-legislative-procedure"},
    "text:stuck one": {"sink_status": "pending", "title": "Stuck candidate",
                       "first_seen": "2026-08-01", "attempts": 3, "rule": "no-date-certain"},
}))
pending = {"content_candidates": [
    {"id": "x", "title": "EU thing", "description": "NIS2 transposition story",
     "content_rule": "eu-legislative-procedure", "content_rule_matched": "trilogue",
     "provenance_urls": ["https://eur-lex.europa.eu/x"]},
]}
state = de.read_content_candidates(TODAY, pending)
check("today's candidate resolved against ledger",
      len(state["today"]) == 1 and state["today"][0]["sink_status"] in ("posted", "not recorded"))
check("stuck pending row surfaces", len(state["stuck_pending"]) == 1
      and state["stuck_pending"][0]["title"] == "Stuck candidate")

html = de.render_html(TODAY, pending, {"ran": False}, {"ran": False}, None, [], [], state)
check("HTML surfaces the candidate section", "Content candidates" in html and "EU thing" in html)
check("HTML surfaces stuck-delivery warning", "awaiting news-desk delivery" in html)
text = de.render_text(TODAY, pending, {"ran": False}, {"ran": False}, None, [], [], state)
check("text part lists candidates", "CONTENT CANDIDATES" in text)

flag_was = de.SURFACE_CONTENT_CANDIDATES
de.SURFACE_CONTENT_CANDIDATES = False
html_off = de.render_html(TODAY, pending, {"ran": False}, {"ran": False}, None, [], [], state)
check("flag off: today section gone, stuck warning stays",
      "Content candidates (new state" not in html_off and "awaiting news-desk delivery" in html_off)
de.SURFACE_CONTENT_CANDIDATES = flag_was

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
