#!/usr/bin/env python3
"""Provenance retention: the URLs the research model actually read must survive
from the Anthropic response, through sanitize, into the daily email.

Before 2026-07-29 research() kept only `text` content blocks and dropped the
`web_search_tool_result` blocks entirely — so the diff model never saw a URL it
could cite, and every regulatory finding arrived as NEEDS-SOURCE with nothing to
click. Block shapes below are copied from a live claude-sonnet-5 response
(web_search_20260209) captured 2026-07-29.
"""
import compliance_auto_refresh as car
import daily_email as de

passed = failed = 0


def check(label, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {label}")
    else:
        failed += 1
        print(f"  FAIL {label}")


print("== _sources_section extracts URLs from live block shapes ==")

BLOCKS = [
    {"type": "server_tool_use", "input": {"query": "smime br version"}},
    {"type": "web_search_tool_result", "content": [
        {"type": "web_search_result", "url": "https://cabforum.org/working-groups/smime/documents/",
         "title": "S/MIME Baseline Requirements (S/MIME BR)", "page_age": "2 days",
         "encrypted_content": "AAAA" * 500},
        {"type": "web_search_result", "url": "https://nukib.gov.cz/en/news/",
         "title": "NUKIB news", "page_age": None, "encrypted_content": "BBBB" * 500},
    ]},
    {"type": "code_execution_tool_result"},
    {"type": "text", "text": "The current version is 1.0.14."},
]

out = car._sources_section(BLOCKS)
check("emits a SOURCES CONSULTED header", "=== SOURCES CONSULTED ===" in out)
check("carries the first URL", "https://cabforum.org/working-groups/smime/documents/" in out)
check("carries the second URL", "https://nukib.gov.cz/en/news/" in out)
check("carries the title alongside the URL", "S/MIME Baseline Requirements" in out)
check("carries page_age when present", "page age: 2 days" in out)
check("omits page_age when null", "page age: None" not in out)
check("drops encrypted_content (opaque + would balloon the prompt)", "AAAA" not in out)
check("instructs the model not to invent URLs", "do NOT invent" in out)

print("== degenerate shapes ==")
check("no search blocks -> empty string", car._sources_section(
    [{"type": "text", "text": "hi"}]) == "")
check("empty block list -> empty string", car._sources_section([]) == "")

# The failure shape really occurs: block 2 of the live capture was exactly this.
ERR = [{"type": "web_search_tool_result",
        "content": {"type": "web_search_tool_result_error", "error_code": "unavailable"}}]
err_out = car._sources_section(ERR)
check("dict error content does not raise and is reported", "unavailable" in err_out)
check("dict error content emits no sources list", "SOURCES CONSULTED" not in err_out)
check("None content is skipped safely",
      car._sources_section([{"type": "web_search_tool_result", "content": None}]) == "")
check("result entries missing url are skipped", car._sources_section(
    [{"type": "web_search_tool_result", "content": [{"type": "web_search_result", "title": "x"}]}]) == "")

DUPES = [{"type": "web_search_tool_result", "content": [
    {"url": "https://a.example/x", "title": "A"},
    {"url": "https://a.example/x", "title": "A again"},
    {"url": "https://b.example/y", "title": "B"},
]}]
check("duplicate URLs across results are deduped",
      car._sources_section(DUPES).count("https://a.example/x") == 1)
check("distinct URLs both kept", "https://b.example/y" in car._sources_section(DUPES))

print("== _normalize_provenance ==")
check("bare string is coerced to a list",
      car._normalize_provenance({"provenance_urls": "https://a.example"})["provenance_urls"]
      == ["https://a.example"])
check("absent field stays absent (no key invented)",
      "provenance_urls" not in car._normalize_provenance({"id": "x"}))
check("non-URL strings are dropped, not kept",
      car._normalize_provenance({"provenance_urls": ["see the press release", "https://ok.example"]})
      ["provenance_urls"] == ["https://ok.example"])
check("capped at 3",
      len(car._normalize_provenance({"provenance_urls": [f"https://e.example/{i}" for i in range(9)]})
          ["provenance_urls"]) == 3)
check("garbage type -> empty list",
      car._normalize_provenance({"provenance_urls": 42})["provenance_urls"] == [])
check("None entries survive without raising",
      car._normalize_provenance({"provenance_urls": [None, "https://ok.example"]})
      ["provenance_urls"] == ["https://ok.example"])
check("http:// accepted as well as https://",
      car._normalize_provenance({"provenance_urls": ["http://old.example"]})
      ["provenance_urls"] == ["http://old.example"])

print("== provenance must not perturb dedup signatures ==")
# This is the load-bearing property: _review_sig hashes id/title/topic/
# description/reason. If provenance changed the signature, every currently-held
# item would look brand new on the next run and re-flag as a duplicate.
base = {"id": "nis2-czechia", "title": "NIS2 Czechia transposition",
        "description": "Czechia transposed NIS2", "reason": "no primary URL"}
withp = dict(base, provenance_urls=["https://nukib.gov.cz/en/news/"])
check("adding provenance leaves _review_sig unchanged",
      car._review_sig(base) == car._review_sig(withp))
check("changing provenance leaves _review_sig unchanged",
      car._review_sig(withp) == car._review_sig(dict(base, provenance_urls=["https://other.example"])))

print("== sanitize_changes preserves provenance end to end ==")
changes, _dropped = car.sanitize_changes({
    "new_deadlines": [], "updated_deadlines": [], "document_version_updates": [],
    "regulatory_updates": [],
    "needs_human_review": [
        {"id": "a", "description": "d", "provenance_urls": ["https://a.example", "bogus"]},
        {"id": "b", "description": "no provenance at all"},
        "a bare string flag",
    ],
    "summary": "",
})
review = changes["needs_human_review"]
check("provenance survives sanitize", review[0]["provenance_urls"] == ["https://a.example"])
check("item without provenance is still kept", review[1]["id"] == "b")
check("bare-string flags still coerce to dicts", review[2]["description"] == "a bare string flag")

print("== email rendering ==")
html = de.render_html("2026-07-30", None, {"ran": False}, {"ran": False}, None, [
    {"kind": "flagged", "title": "NIS2 Czechia transposition", "date": "",
     "first_seen": "2026-07-21", "urgent": False,
     "provenance_urls": ["https://nukib.gov.cz/en/news/"]},
    {"kind": "flagged", "title": "Legacy item with no provenance", "date": "",
     "first_seen": "2026-07-21", "urgent": False, "provenance_urls": []},
], None)
check("HTML links the provenance URL", "https://nukib.gov.cz/en/news/" in html)
check("HTML renders it as a clickable anchor", "<a href='https://nukib.gov.cz/en/news/'" in html)
check("HTML still renders items lacking provenance", "Legacy item with no provenance" in html)

text = de.render_text("2026-07-30", None, {"ran": False}, {"ran": False}, None, [
    {"kind": "flagged", "title": "NIS2 Czechia transposition", "date": "",
     "first_seen": "2026-07-21", "urgent": False,
     "provenance_urls": ["https://nukib.gov.cz/en/news/"]},
], None)
check("text part lists the source URL", "source: https://nukib.gov.cz/en/news/" in text)

# An entry built before this feature shipped has no provenance_urls key at all.
legacy = de.render_html("2026-07-30", None, {"ran": False}, {"ran": False}, None,
                        [{"kind": "flagged", "title": "Pre-feature item", "date": "",
                          "first_seen": "2026-07-15", "urgent": False}], None)
check("renderer tolerates entries with no provenance_urls key", "Pre-feature item" in legacy)

print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
