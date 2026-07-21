#!/usr/bin/env python3
"""Offline regression tests for the document-hash sanitizer.

Guards against the 2026-07-21 microsoft_root_program false positive: dynamic
learn.microsoft.com page chrome (regenerating "AI Summary" block, daily-
rotating "Additional resources"/Events widget) must not change the sanitized
hash. Fixtures are two captures of the real Learn page differing only in the
Events widget date. No network calls.
"""
import hashlib
from pathlib import Path

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


FIXTURES = Path(__file__).parent / "fixtures"
capture_a = (FIXTURES / "learn_ms_trusted_root_capture_a.html").read_text()
capture_b = (FIXTURES / "learn_ms_trusted_root_capture_b.html").read_text()

print("fixture sanity")
check("captures actually differ raw",
      hashlib.sha256(capture_a.encode()).hexdigest()
      != hashlib.sha256(capture_b.encode()).hexdigest())
check("capture contains events card", "events-card" in capture_a)
check("capture contains ai-summary block", 'data-id="ai-summary"' in capture_a)

print("events widget churn is invisible to the hash")
check("only-Events-date-differs captures hash identically",
      pki.hash_content(capture_a) == pki.hash_content(capture_b))

print("AI Summary regeneration is invisible to the hash")
regen = capture_a.replace(
    "AI Summary: This page describes the",
    "AI Summary: An overview of the")
check("regenerated summary text present in raw", regen != capture_a)
check("regenerated summary hashes identically",
      pki.hash_content(capture_a) == pki.hash_content(regen))

print("substantive document changes still detected")
changed = capture_a.replace(
    "on an annual basis", "on a semi-annual basis")
check("body edit present in raw", changed != capture_a)
check("body edit changes the hash",
      pki.hash_content(capture_a) != pki.hash_content(changed))

print("sanitized content drops the dynamic blocks entirely")
sanitized = capture_a
for pat in pki._HASH_NOISE_PATTERNS:
    sanitized = pat.sub("", sanitized)
for tag, start_pat in pki._HASH_NOISE_BLOCKS:
    sanitized = pki._strip_balanced_blocks(sanitized, tag, start_pat)
check("no events-card left", "events-card" not in sanitized)
check("no ai-summary left", 'data-id="ai-summary"' not in sanitized)
check("no additional-resources rail left",
      'id="ms--additional-resources"' not in sanitized)
check("no Frontier event text left", "Frontier Transformation Week" not in sanitized)
check("document body survives", "Continuing Program Requirements" in sanitized)
check("superseded notice survives (real content, not chrome)",
      "TrustedRootProgram/Program-Requirements" in sanitized)

print("balanced-block stripper edge cases")
nested = ('before<section data-bi-name="events-card"><section>inner'
          '</section>event</section>after')
check("nested same-name tags removed through balanced close",
      pki._strip_balanced_blocks(
          nested, "section", pki._HASH_NOISE_BLOCKS[3][1]) == "beforeafter")
unbalanced = 'x<div data-id="ai-summary">never closed'
check("unbalanced markup terminates, drops opening tag only",
      pki._strip_balanced_blocks(
          unbalanced, "div", pki._HASH_NOISE_BLOCKS[0][1]) == "xnever closed")
check("content without noise blocks passes through unchanged",
      pki._strip_balanced_blocks(
          "<div>plain</div>", "div", pki._HASH_NOISE_BLOCKS[0][1])
      == "<div>plain</div>")

print("existing cloudflare/script rules still hold")
scripted = capture_a.replace(
    "</head>", '<script>var cfToken="abc123";</script></head>')
check("script noise still invisible",
      pki.hash_content(capture_a) == pki.hash_content(scripted))

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
