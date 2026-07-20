#!/usr/bin/env python3
"""Offline tests for the deadlines CSV export (build_deadlines_csv).

Round-trips the CSV back through csv.reader to prove headers, row count,
flattening (relatedGuides -> joined URLs, consequences -> two columns), and
correct escaping of commas/quotes in descriptions. Also checks the filter
params match get_all_deadlines_unified(). No network calls.
"""
import csv
import io

import pki_compliance_mcp as pki

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


unified = pki.get_all_deadlines_unified()
text = pki.build_deadlines_csv()
rows = list(csv.reader(io.StringIO(text)))
header, data = rows[0], rows[1:]
by_title = {r[header.index("title")]: r for r in data}


print("== structure ==")
check("header matches DEADLINE_CSV_COLUMNS", header == pki.DEADLINE_CSV_COLUMNS)
check("one data row per unified deadline", len(data) == len(unified))
check("every row has full column count", all(len(r) == len(header) for r in data))
check("rows sorted by date ascending",
      [r[0] for r in data] == sorted(r[0] for r in data))

print("== escaping / flattening ==")
# A description containing a comma must round-trip as one field, not split.
comma_desc = next((d for d in unified if "," in d.get("description", "")), None)
if comma_desc:
    row = by_title.get(comma_desc["title"])
    check("comma-laden description survives as a single cell",
          row is not None and row[header.index("description")] == comma_desc["description"])
else:
    check("comma-laden description present to test", False)

ct = by_title.get("Chrome CT Pre-Logging Required")
check("relatedGuides flattened to joined URLs",
      ct is not None and ct[header.index("related_guides")] == "/guides/certificate-transparency")

# A deadline with a consequences dict populates the two consequence columns.
cons_d = next((d for d in unified if (d.get("consequences") or {}).get("scenario")), None)
if cons_d:
    row = by_title.get(cons_d["title"])
    check("consequence_scenario column populated",
          row is not None and row[header.index("consequence_scenario")] == cons_d["consequences"]["scenario"])
else:
    check("consequences present to test", False)

print("== booleans ==")
majors = {r[header.index("title")] for r in data if r[header.index("is_major")] == "true"}
check("is_major true/false only", all(r[header.index("is_major")] in ("true", "false") for r in data))
check("is_major reflects feed isMajor",
      majors == {d["title"] for d in unified if d.get("isMajor")})

print("== filters ==")
chrome_csv = pki.build_deadlines_csv(framework="cabforum")
chrome_rows = list(csv.reader(io.StringIO(chrome_csv)))[1:]
expected = sum(1 for d in unified if d.get("framework_id") == "cabforum")
check("framework filter narrows rows", len(chrome_rows) == expected and expected < len(unified))

upcoming_csv = pki.build_deadlines_csv(status="upcoming")
up_rows = list(csv.reader(io.StringIO(upcoming_csv)))[1:]
check("status filter narrows rows",
      len(up_rows) == sum(1 for d in unified if d.get("status") == "upcoming"))


print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
