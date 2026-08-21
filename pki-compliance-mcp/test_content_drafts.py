#!/usr/bin/env python3
"""Offline tests for content_drafts.py and its daily_email.py surface.

Covers urgent-item collection (section coverage, rejected exclusion,
drafted-signature dedup), draft file layout, and the email's drafts section
(HTML, text, subject). No network calls, no git.
"""
import json
import tempfile
from pathlib import Path

import compliance_auto_refresh as car
import content_drafts as cd
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


tmp = Path(tempfile.mkdtemp(prefix="cd_test_"))
cd.DATA_DIR = tmp
cd.LOG_FILE = tmp / "content_drafts.log"
cd.STATE_FILE = tmp / "content_drafted.json"
car.REJECTED_FILE = tmp / "rejected.json"

URGENT_DL = {"id": "test-mass-revocation", "date": "2026-08-01",
             "title": "Mass Revocation Event", "urgent": True,
             "source_url": "https://cabforum.org/x"}
CALM_DL = {"id": "test-calm", "date": "2027-01-01", "title": "Calm Item"}
URGENT_REG = {"id": "test-reg", "date": "2026-08-15",
              "title": "Urgent Regulation", "urgent": True}
URGENT_FLAG = {"id": "test-flag", "title": "Root distrust announced",
               "description": "ballot SC-999", "urgent": True}
PENDING = {
    "summary": "test run",
    "new_deadlines": [URGENT_DL, CALM_DL],
    "updated_deadlines": [],
    "regulatory_updates": [URGENT_REG],
    "needs_human_review": [URGENT_FLAG, {"id": "calm-flag", "title": "calm"}],
}

print("== collect_urgent ==")
got = cd.collect_urgent(PENDING)
check("finds all urgent items", len(got) == 3)
check("skips non-urgent items",
      not any(e["item"].get("id") in ("test-calm", "calm-flag") for e in got))
check("covers proposals and flags",
      {e["kind"] for e in got} == {"deadline", "regulatory", "flagged"})

car.REJECTED_FILE.write_text(json.dumps({"ids": ["test-reg"], "signatures": []}))
got = cd.collect_urgent(PENDING)
check("rejected id excluded", len(got) == 2
      and not any(e["item"].get("id") == "test-reg" for e in got))

cd.STATE_FILE.write_text(json.dumps(
    {"signatures": {car._sig(URGENT_DL): "2026-07-15"}}))
got = cd.collect_urgent(PENDING)
check("already-drafted signature excluded", len(got) == 1
      and got[0]["item"]["id"] == "test-flag")
cd.STATE_FILE.unlink()
car.REJECTED_FILE.write_text(json.dumps({"ids": [], "signatures": []}))

print("== _extract_json ==")
check("plain JSON", cd._extract_json('{"a": 1}') == {"a": 1})
check("JSON with prose around it",
      cd._extract_json('Here you go:\n{"a": 1}\nDone.') == {"a": 1})
check("literal newline inside string tolerated",
      cd._extract_json('{"body": "line one\nline two"}')["body"]
      == "line one\nline two")
try:
    cd._extract_json("no json here")
    check("raises on no JSON", False)
except ValueError:
    check("raises on no JSON", True)

# generate() gained web_search on 2026-08-20 for source verification, so the
# reply can now carry search prose with braces of its own AHEAD of the
# package. The greedy first-brace-to-last-brace span cannot survive that; the
# fallback walks TOP-LEVEL objects and takes the last one that parses.
_SEARCHY = ('Found {"url": "https://example.com/a"} and {"n": 1}.\n\n'
            '{"slug": "x", "source_check": {"verified": true}, '
            '"blog_markdown": "brace { in a string"}')
check("picks the package past brace-bearing search prose",
      cd._extract_json(_SEARCHY)["slug"] == "x")
check("picks the OUTERMOST object, not a nested one",
      cd._extract_json(_SEARCHY)["source_check"] == {"verified": True})
try:
    cd._extract_json("{unbalanced and unparseable")
    check("raises when no candidate parses", False)
except ValueError:
    check("raises when no candidate parses", True)

print("== slugify ==")
check("basic", cd.slugify("Mass Revocation: DigiCert!") == "mass-revocation-digicert")
check("empty fallback", cd.slugify("!!!") == "urgent-item")
check("length cap", len(cd.slugify("x" * 200)) <= 60)

print("== write_drafts ==")
DRAFTS = {
    "slug": "mass-revocation-event",
    "blog_markdown": "# Mass Revocation\n\nBody.",
    "linkedin": "LinkedIn body",
    "tweet": "Tweet body https://fixmycert.com/compliance",
    "youtube": {"title": "Mass Revocation - What To Do",
                "notebooklm_audio_prompt": "audio",
                "notebooklm_visual_prompt": "visual",
                "description": "desc", "pinned_comment": "📌 pin",
                "thumbnail_prompt": "thumb"},
    "kit_email": {"subject": "Mass revocation: 5-day window",
                  "preview_text": "48k certs affected",
                  "body_markdown": "Kit body\n\n- Patrick"},
}
entry = {"kind": "deadline", "sig": car._sig(URGENT_DL), "item": URGENT_DL}
# All five channels, because the assertions below cover the full file layout.
# write_drafts() writes only the channels it is given as of 2026-08-20, so the
# argument is what keeps this the whole-package case rather than a subset.
out_dir = cd.write_drafts(tmp / "repo_drafts", "2026-07-16", DRAFTS, entry,
                          list(cd.CHANNELS))
check("dir named date-slug", out_dir.name == "2026-07-16-mass-revocation-event")
for f in ("blog.md", "linkedin.md", "tweet.md", "youtube.md",
          "kit_broadcast.md", "meta.json"):
    check(f"writes {f}", (out_dir / f).exists())
kit_text = (out_dir / "kit_broadcast.md").read_text()
check("kit has audience scoping note", "19302998" in kit_text
      and "NOT the full list" in kit_text)
check("kit has subject and body",
      "Mass revocation: 5-day window" in kit_text and "Kit body" in kit_text)
check("kit marked draft never auto-send", "never auto-send" in kit_text)
meta = json.loads((out_dir / "meta.json").read_text())
check("meta records signature", meta["signature"] == entry["sig"])
check("meta marked unpublished", "not published" in meta["status"])
# The source-verification result must survive into meta.json, so a drafter
# that caught a stale source_item leaves a record instead of silently
# writing better copy. Absent from DRAFTS here -> degrades, never KeyErrors.
check("meta degrades when model omits source_check",
      meta["source_check"] == {"verified": None}
      and meta["requires_reader_action"] is None)
_checked = cd.write_drafts(
    tmp / "repo_drafts", "2026-07-17",
    dict(DRAFTS, requires_reader_action=False,
         source_check={"verified": True, "drafted_from": "sources",
                       "discrepancies": ["desc called it unresolved; "
                                         "source already answered it"]}),
    entry, list(cd.CHANNELS))
_m = json.loads((_checked / "meta.json").read_text())
check("meta records a source-check discrepancy",
      _m["source_check"]["drafted_from"] == "sources"
      and len(_m["source_check"]["discrepancies"]) == 1)
check("meta records the no-action classification",
      _m["requires_reader_action"] is False)
yt_text = (out_dir / "youtube.md").read_text()
check("youtube.md has all sections",
      all(s in yt_text for s in ("Title", "NotebookLM audio prompt",
                                 "Description", "Pinned comment",
                                 "Thumbnail prompt")))

print("== email drafts surface ==")
de.DRAFTS_DIR = tmp / "repo_drafts"
pkgs = de.read_content_drafts("2026-07-16")
check("read_content_drafts finds package", len(pkgs) == 1)
check("package carries source title", pkgs[0]["title"] == "Mass Revocation Event")
check("package inlines tweet", "Tweet body" in pkgs[0]["tweet"])
check("no packages for other dates", de.read_content_drafts("2026-07-15") == [])
check("missing dir degrades to empty",
      (lambda d: (setattr(de, "DRAFTS_DIR", tmp / "nope"),
                  de.read_content_drafts("2026-07-16"))[1] == [])(None))
de.DRAFTS_DIR = tmp / "repo_drafts"

doc_check = {"ran": True, "changes_detected": 0}
auto_refresh = {"ran": True, "skipped": True, "skip_reason": "gate"}
html = de.render_html("2026-07-16", None, doc_check, auto_refresh, drafts=pkgs)
check("html has drafts section", "Content drafts ready" in html)
check("html inlines tweet", "Tweet body" in html)
check("html points at repo path", "content_drafts/2026-07-16-mass-revocation-event/" in html)
check("html says drafts only", "review before posting" in html)
html_no = de.render_html("2026-07-16", None, doc_check, auto_refresh, drafts=[])
check("no section when no drafts", "Content drafts ready" not in html_no)

text = de.render_text("2026-07-16", None, doc_check, auto_refresh, drafts=pkgs)
check("text has drafts block", "CONTENT DRAFTS READY (1)" in text)
check("text names package dir", "2026-07-16-mass-revocation-event" in text)

print("== notify_urgent ==")
import os
os.environ.pop("NTFY_TOPIC", None)
calls = []
_orig_post = cd.httpx.post
cd.httpx.post = lambda *a, **k: calls.append((a, k)) or (_ for _ in ()).throw(
    AssertionError("should not post without NTFY_TOPIC"))
cd.notify_urgent([{"item": URGENT_DL}])
check("no-op without NTFY_TOPIC", calls == [])


class _FakeResp:
    def raise_for_status(self):
        pass


cd.httpx.post = lambda *a, **k: calls.append((a, k)) or _FakeResp()
os.environ["NTFY_TOPIC"] = "test-topic"
cd.notify_urgent([{"item": URGENT_DL}])
check("posts to topic when set",
      len(calls) == 1 and calls[0][0][0].endswith("/test-topic"))
check("title carries count",
      "1 urgent" in calls[0][1]["headers"]["Title"])
check("body names the item",
      "Mass Revocation Event" in calls[0][1]["content"].decode())


def _boom(*a, **k):
    raise RuntimeError("network down")


cd.httpx.post = _boom
try:
    cd.notify_urgent([{"item": URGENT_DL}])
    check("push failure swallowed", True)
except Exception:
    check("push failure swallowed", False)
cd.httpx.post = _orig_post
os.environ.pop("NTFY_TOPIC", None)

print("== cap ==")
many = {"new_deadlines": [
    {"id": f"u{i}", "date": "2026-08-01", "title": f"Urgent {i}", "urgent": True}
    for i in range(5)]}
check("collect returns all (cap applied in main)", len(cd.collect_urgent(many)) == 5)
check("MAX_DRAFTS_PER_RUN sane", 1 <= cd.MAX_DRAFTS_PER_RUN <= 5)

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
