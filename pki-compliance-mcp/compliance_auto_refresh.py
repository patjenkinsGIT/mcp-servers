#!/usr/bin/env python3
"""
PKI Compliance Hub Auto-Refresh

Automates the manual research-and-update workflow for the FixMyCert PKI
Compliance Hub. Runs as a daily cron job on the DigitalOcean droplet.

Usage:
    python compliance_auto_refresh.py                  # Research only, generate review file
    python compliance_auto_refresh.py --auto-apply     # Research + apply + restart + push
    python compliance_auto_refresh.py --dry-run        # Research only, print to stdout
    python compliance_auto_refresh.py --query-only     # Just run research queries, no diff
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PKI_REPO_PATH = os.environ.get("PKI_REPO_PATH", "/opt/mcp-servers/pki-compliance-mcp")
COMPLIANCE_API_URL = os.environ.get("COMPLIANCE_API_URL", "http://localhost:5000")
DATA_DIR = Path.home() / ".pki-compliance-mcp"
LOG_FILE = DATA_DIR / "auto_refresh.log"
MODEL = "claude-sonnet-5"
INTER_QUERY_DELAY = 15  # seconds between API calls

# --- Cost-control gating -----------------------------------------------------
# The expensive web-search research only runs when (a) a tracked document
# actually changed, (b) the weekly safety net is due, or (c) --force is given.
# This is what stops the daily unconditional Anthropic API spend.
PKI_CONTAINER = os.environ.get("PKI_CONTAINER", "pki-compliance-mcp")
MAX_DAYS_BETWEEN_RESEARCH = int(os.environ.get("MAX_DAYS_BETWEEN_RESEARCH", "7"))
LAST_RESEARCH_FILE = DATA_DIR / "last_research.json"
REJECTED_FILE = DATA_DIR / "rejected.json"
# Manual holds that outlive flag age-out: {"sc101": "2026-08-08", ...}
# (anchor token -> hold-until date, inclusive). See held_review_anchors().
HOLDS_FILE = DATA_DIR / "review_holds.json"

# ---------------------------------------------------------------------------
# Research queries
# ---------------------------------------------------------------------------

RESEARCH_QUERIES = [
    {
        "id": "cabf_ballots",
        "label": "CA/Browser Forum Ballots",
        "prompt": (
            "Search for CA/Browser Forum ballot results from the last 30 days. "
            "Check cabforum.org for any new passed, failed, or proposed ballots "
            "affecting TLS certificate requirements, code signing, or S/MIME. "
            "Include ballot numbers, voting results, and effective dates."
        ),
    },
    {
        "id": "cabf_doc_versions",
        "label": "CA/B Forum Document Versions",
        "prompt": (
            "Check the current version numbers for these CA/Browser Forum documents: "
            "TLS Baseline Requirements (cabforum.org/working-groups/server/baseline-requirements/documents/), "
            "EV Guidelines, Code Signing BRs, S/MIME BRs, Network Security Requirements. "
            "Return the latest version number and date for each."
        ),
    },
    {
        "id": "browser_root_programs",
        "label": "Browser Root Program Updates",
        "prompt": (
            "Search for Chrome Root Program, Mozilla Root Store Policy, Apple Root "
            "Certificate Program, and Microsoft Trusted Root Program updates from "
            "the last 30 days. Check for new policy versions, distrust actions, "
            "deadline changes, or new requirements."
        ),
    },
    {
        "id": "nist_pqc",
        "label": "NIST and PQC Updates",
        "prompt": (
            "Search for NIST post-quantum cryptography updates, NIST SP 800-131A "
            "changes, NIST IR 8547 updates, and NSA CNSA 2.0 guidance changes "
            "from the last 30 days."
        ),
    },
    {
        "id": "regulatory",
        "label": "Regulatory Framework Updates",
        "prompt": (
            "Search for DORA (Digital Operational Resilience Act), NIS2 Directive, "
            "and UK Cyber Security and Resilience Bill updates from the last 30 days. "
            "Include any new implementation deadlines, national transposition updates, "
            "or enforcement actions."
        ),
    },
]

# ---------------------------------------------------------------------------
# Claude API helper
# ---------------------------------------------------------------------------


def _dated(prompt: str) -> str:
    """Prefix a research query with today's date and a recency instruction.

    Every RESEARCH_QUERIES prompt says "the last 30 days", and until
    2026-08-20 nothing ever told the model what today was — no system prompt,
    no date, so the window was unanchored and unenforceable and the model fell
    back on whatever its search results happened to surface. That is the shape
    behind the 2026-08-20 Let's Encrypt description, which hedged itself as
    "as of Aug 10-12, 2026" on a run that executed on 08-20 and then carried
    that eight-day-old framing into a content package.

    The date goes in the USER turn, not a system prompt: it changes daily, and
    a volatile system prefix is what breaks prompt caching if caching is ever
    turned on here.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"Today's date is {today} (UTC). Every relative window below - \"the "
        "last 30 days\" and any similar phrase - is relative to that date, "
        "not to your training data.\n\n"
        "Report the LATEST state of each thing you find, not the first "
        "account of it. Where a source says a matter is unresolved, pending, "
        "disputed, or under discussion, look for a more recent source that "
        "settles it - a follow-up post, a reply further down the same thread, "
        "or a later thread - and report what is true now. If you cannot "
        "establish the current state, say so explicitly and date the claim, "
        "rather than presenting stale information as current.\n\n"
        f"{prompt}"
    )


def research(prompt: str, max_retries: int = 3) -> str:
    """Call Claude API with web_search tool to research PKI updates."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    prompt = _dated(prompt)

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
                response = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": 16000,
                        "tools": [{"type": "web_search_20260209", "name": "web_search"}],
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
        except httpx.TimeoutException:
            log(f"  Timed out after 600s, retrying (attempt {attempt+1}/{max_retries})")
            continue
        if response.status_code == 429:
            wait = 30 * (attempt + 1)
            log(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        data = response.json()
        text = "\n".join(
            block["text"] for block in data["content"] if block["type"] == "text"
        )
        return text + _sources_section(data["content"])
    raise RuntimeError(f"Rate limited or timed out after {max_retries} retries")


def _sources_section(blocks: list) -> str:
    """Render the URLs the search actually returned as an appendix to the text.

    Provenance retention: the model's prose rarely names its sources, and text
    blocks come back with no `citations` array, so the URLs exist ONLY in the
    web_search_tool_result blocks. Dropping them (as this function's caller did
    until 2026-07-29) is why regulatory findings kept arriving as NEEDS-SOURCE:
    the diff model was never shown a URL it could cite, so every such item had
    to be researched again by hand from a blank search.

    `content` is a list of results on success but a bare dict on failure
    (e.g. {"type": "web_search_tool_result_error", "error_code": "unavailable"}),
    so branch on the type before iterating. `encrypted_content` is deliberately
    dropped — it is opaque to us and would balloon the prompt.
    """
    seen, sources, errors = set(), [], []
    for block in blocks:
        if block.get("type") != "web_search_tool_result":
            continue
        content = block.get("content")
        if isinstance(content, dict):
            errors.append(content.get("error_code", "unknown"))
            continue
        if not isinstance(content, list):
            continue
        for result in content:
            url = (result or {}).get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            title = (result.get("title") or "").strip()
            age = (result.get("page_age") or "").strip()
            sources.append(f"- {url}{f' — {title}' if title else ''}{f' (page age: {age})' if age else ''}")
    out = ""
    if sources:
        out += ("\n\n=== SOURCES CONSULTED ===\nURLs returned by web search for this query. "
                "Cite these in provenance_urls; do NOT invent URLs not listed here.\n"
                + "\n".join(sources))
    if errors:
        out += f"\n\n[web search errors this query: {', '.join(sorted(set(errors)))}]"
    return out


# ---------------------------------------------------------------------------
# Fetch current data from local API
# ---------------------------------------------------------------------------


def fetch_current_data() -> dict | None:
    """Fetch current compliance data from the local API."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{COMPLIANCE_API_URL}/api/compliance-data")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        log(f"WARNING: Could not fetch current data from API: {e}")
        return None


# ---------------------------------------------------------------------------
# Diff analysis
# ---------------------------------------------------------------------------

DIFF_SYSTEM_PROMPT = """\
You are a PKI compliance data analyst. You compare research findings against \
existing compliance data and identify CONFIRMED changes only.

Rules:
- Only flag changes with specific sources/URLs
- Do NOT hallucinate deadlines
- For ballot updates: only add if voting is COMPLETE and IPR review has passed
- Match the exact data format of existing entries
- Generate unique id fields following kebab-case naming convention
- If unsure, flag for human review instead of auto-adding

Date precision rules:
- Only emit a day-precise date when a primary source explicitly states that \
exact day (e.g., "effective March 15, 2027"). NEVER guess or invent a specific day.
- If the source only states a month, use the last day of that month and set \
"is_estimated": true.
- If the source only states a quarter or year, use the last day of that \
quarter or year and set "is_estimated": true.
- The "date" field must always be formatted YYYY-MM-DD; "is_estimated" is what \
distinguishes confirmed exact dates from estimated ones.

Return a JSON object with this structure:
{
  "new_deadlines": [...],
  "updated_deadlines": [...],
  "document_version_updates": [...],
  "regulatory_updates": [...],
  "needs_human_review": [...],
  "content_candidates": [...],
  "summary": "brief description of changes found"
}

CONTENT CANDIDATES (a distinct state — not review, not a deadline):
An item that is real, relevant, and well-sourced but has NO future date certain \
belongs in "content_candidates", NOT in needs_human_review or regulatory_updates. \
It will be routed to the content desk instead of the human review queue. Classify \
an item as a content candidate when it matches any of:
- EU ordinary legislative procedure items before adoption (proposals, readings, \
trilogues, Council positions)
- Bills before Royal Assent (or equivalent pre-enactment stages)
- Enforcement actions (CJEU referrals, infringement proceedings)
- Draft standards with open comment periods (NIST ipd/IWD and similar)
- Anything else with no date certain that is still newsworthy for PKI operators

Each content_candidate item:
{
  "id": "kebab-case-id",
  "title": "Short Title",
  "description": "What happened and why it matters, including any dates the \
source states (the content system does not preserve date fields — dates must \
appear in this text)",
  "content_rule": "eu-legislative-procedure|pre-royal-assent-bill|enforcement-action|draft-standard-comment-period|no-date-certain",
  "provenance_urls": [... same rules as needs_human_review ...],
  "primary_url": "the one URL from provenance_urls treated as primary, if any"
}

Do NOT put an item in content_candidates if it has a confirmed future effective \
date — that is a new_deadline (or needs_human_review if unverified). When the \
same story later acquires a date certain, emit it as a new_deadline as normal; \
its earlier content-candidate classification must not stop you.

Each new_deadline should match this format:
{
  "id": "kebab-case-id",
  "date": "YYYY-MM-DD",
  "title": "Short Title",
  "description": "Full description",
  "source": "cab-forum|chrome|mozilla|apple|microsoft|nist|nsa",
  "source_url": "https://... (URL of the primary/authoritative source for this deadline)",
  "category": "certificates|validation|revocation|...",
  "isMajor": true/false,
  "impact": "Brief impact statement",
  "is_estimated": true/false (true unless a primary source explicitly states the exact day),
  "feed_confirmed": true/false (true ONLY when the research findings for this item cite a tracked source document or feed that was flagged as changed; when in doubt, false),
  "urgent": true/false. Default false. Set it true ONLY if the item clears ALL THREE tests:
    1. ANNOUNCED, NOT SPECULATED. The triggering event must be stated as fact by an \
authoritative source - the CA, the root program, or the regulator. If the finding says \
something "could", "may", or "might" happen, that an outcome is unconfirmed or unclear, \
or that a community, forum, or working group is debating or considering whether a rule \
requires it, that is NOT an announcement: urgent stays false. Speculation about a mass \
revocation is not a mass revocation. Report such items normally - just do not escalate them.
    2. READER ACTION, NOT CA HOUSEKEEPING. The obligation must fall on the enterprise \
certificate team. A CA's own compliance problem is NOT urgent for subscribers: a defect \
in its CP/CPS or other policy documents, a missing attestation, an audit finding, or an \
incident report it filed about itself. A DEFECTIVE POLICY DOCUMENT IS NOT A DEFECTIVE \
CERTIFICATE - only mis-issuance of certificates carries subscriber-side revocation.
    3. WEEKS, NOT MONTHS. One of: an announced mass revocation event; an announced \
distrust or removal of a currently trusted root or CA; or a new compliance obligation \
with a date certain falling within ~60 days of today's date, which is given in the user \
message. A date that has already passed does not qualify.
}

needs_human_review items may also carry "urgent": true under those same three tests. \
One nuance on test 1: a report you could not fully verify may still be urgent IF what is \
reported is an announcement that has already happened - "CA X has announced a mass \
revocation", from a source you could not corroborate, still deserves the flag. That is \
different from speculation about whether an event might occur, or from a community \
arguing over what a rule requires; those fail test 1 no matter how confidently they are \
written.

PROVENANCE (required on every needs_human_review item):
Each needs_human_review item MUST carry "provenance_urls": a list of 1-3 URLs you \
actually read for that item, most useful first, copied verbatim from the \
"=== SOURCES CONSULTED ===" appendix of the research findings.
- Include the URL even when it is NOT a primary or authoritative source. A trade-press \
article, a mailing-list post, or a national agency's news page is exactly what makes \
the follow-up a click instead of a fresh search. "Not primary" is a reason to flag the \
item for review, never a reason to omit the URL.
- If the findings genuinely contain no URL for an item, use an empty list []. NEVER \
invent, guess, or reconstruct a URL that does not appear in the appendix.
- This is separate from "source_url" on new_deadlines, which must stay the primary \
authoritative source. new_deadlines may also carry "provenance_urls" when the primary \
source was reached via some other page.
- Items carrying provenance_urls SHOULD also carry "primary_url": the single URL \
from that list you treated as the primary source for the item. It must be one of \
the provenance_urls entries. Omit it if no URL deserves the label; NEVER invent one.

Each document_version_update:
{
  "id": "tls-br|ev-guidelines|code-signing-br|smime-br|netsec",
  "new_version": "X.Y.Z",
  "new_date": "Mon YYYY"
}

If no changes are found, return:
{"new_deadlines":[],"updated_deadlines":[],"document_version_updates":[],"regulatory_updates":[],"needs_human_review":[],"content_candidates":[],"summary":"No changes found"}
"""


def _save_raw_diff(text: str, why: str, stamp: str) -> Path | None:
    """Persist an unparseable diff response so the failure is diagnosable.

    Returns the path written, or None if even that failed — a diagnostic write
    must never be the thing that kills the run."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / f"diff_raw_{stamp}.txt"
        path.write_text(f"# unparseable diff response: {why}\n# {len(text)} chars\n\n{text}")
        log(f"Raw diff response saved for diagnosis: {path} ({why})")
        return path
    except Exception as e:  # pragma: no cover - defensive
        log(f"WARNING: could not save raw diff response ({e})")
        return None


def _diff_failure_marker(text: str, saved: Path | None, stamp: str) -> str:
    """Short human-readable stand-in for the raw response in the review file.

    Leads with the run stamp deliberately. This string becomes a
    needs_human_review item, and _review_sig() falls back to the first 12
    normalized words when an item carries no anchors — so a marker with fixed
    leading words would give two consecutive failures the same signature and
    dedup_changes would drop the second as "re-flagged", silencing a recurring
    fault after day one. Deduping repeated CONTENT proposals is the point of
    that machinery; deduping a repeated FAULT report is not. The stamp is
    unique per run, so every failure reports."""
    where = f" Full text: {saved}" if saved else " Full text could not be saved."
    excerpt = " ".join(text.split())[:400] if text.strip() else "(empty response)"
    return (
        f"Diff response unparseable at {stamp} — {len(text)} chars, "
        f"no proposals could be extracted from this run.{where} "
        f"First 400 chars: {excerpt}"
    )


def analyze_diff(research_results: dict, current_data: dict | None) -> dict:
    """Ask Claude to analyze research findings vs current data."""
    current_summary = "Current data not available (API unreachable)"
    if current_data:
        # Extract just the parts we need for comparison
        current_summary = json.dumps({
            "deadlines": [
                {"id": d["id"], "date": d["date"], "title": d["title"]}
                for d in current_data.get("deadlines", [])
            ],
            "cabfDocuments": current_data.get("cabfDocuments", []),
            "metadata": current_data.get("metadata", {}),
        }, indent=2)

    # Today's date is required, not decorative: the "urgent" rule turns on an
    # obligation "taking effect within ~60 days", and until 2026-08-20 this
    # call was never told what today was, so that test could not actually be
    # applied. It goes in the user turn, after the cacheable system prefix.
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = (
        f"## Today's date\n\n{today} (UTC). Evaluate every relative window "
        "against this date - in particular the \"~60 days\" test for the "
        "`urgent` flag, and whether a date has already passed.\n\n"
        f"## Research Findings\n\n"
        + "\n\n".join(
            f"### {q_id}\n{text}"
            for q_id, text in research_results.items()
        )
        + f"\n\n## Current Compliance Data (for comparison)\n\n{current_summary}"
        + "\n\nAnalyze the research findings against the current data. "
        "Return ONLY the JSON object described in the system prompt."
    )

    with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        response = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": MODEL,
                # claude-sonnet-5 runs adaptive thinking by default when the
                # thinking param is omitted, and thinking tokens count against
                # max_tokens — that exhausted the old 16000 cap on all four
                # 2026-08 diff-parse failures (stop_reason=max_tokens,
                # output_tokens=16000, ~13K of it invisible thinking). Low
                # effort caps the thinking spend; 32000 absorbs the variance
                # that remains while staying under the 600s read timeout at
                # observed throughput. Both are part of the cached prompt
                # prefix — set once, do not vary per run.
                "max_tokens": 32000,
                "output_config": {"effort": "low"},
                "system": DIFF_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        text = "\n".join(
            block["text"] for block in data["content"] if block["type"] == "text"
        )

    # Envelope metadata, logged on EVERY diff call. The 2026-08-02/06/08/12
    # failures were all truncations at max_tokens — the visible text looked
    # far under the cap, but adaptive thinking had consumed the rest of the
    # budget invisibly. stop_reason was discarded right here until 2026-08-12,
    # which is why the cause took four failures to name. Costs nothing to
    # keep; any future failure names itself instead of needing a forensic
    # session.
    stop_reason = data.get("stop_reason")
    out_tokens = (data.get("usage") or {}).get("output_tokens")
    envelope = f"stop_reason={stop_reason}, output_tokens={out_tokens}"
    log(f"Diff response: {len(text)} chars, {envelope}")

    # Extract JSON from response.
    #
    # The brace-less response is the RARE failure; a response that contains
    # braces but is not valid JSON is the likely one, and it used to be
    # unguarded. On 2026-08-06 json.loads raised JSONDecodeError here
    # ("Expecting ',' delimiter: line 87 column 6"), propagated to main, and
    # exited 1 before any review file was written — discarding 192k chars of
    # research that had already been paid for and, because the staleness clock
    # had been reset moments earlier, buying up to 7 days of silence on the
    # day the Microsoft August 2026 deployment notice landed.
    #
    # Both parse failures now land in the same degraded return, and the raw
    # text goes to disk either way: the in-payload copy is truncated (a 7KB
    # blob renders badly in the daily email) and the full text is the
    # diagnostic artifact. 2026-08-02 showed why disk matters — that day's
    # response was empty, the sanitizer dropped it as "malformed (empty)", and
    # the failure left no trace anywhere.
    # Both failure branches log an explicit ERROR line (2026-08-08 decision:
    # loud failure). daily_email's status parser counts "ERROR" lines, so this
    # alone flips the email off the clean path even before its own
    # review-file-based detection — two independent signals, either suffices.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            log(f"ERROR: diff parse failed — no proposals extracted this run "
                f"(JSONDecodeError: {e}; {envelope})")
            saved = _save_raw_diff(text, f"JSONDecodeError: {e}; {envelope}", stamp)
            return {
                "summary": f"Could not parse diff response — malformed JSON ({e})",
                "needs_human_review": [_diff_failure_marker(text, saved, stamp)],
            }
    log(f"ERROR: diff parse failed — no proposals extracted this run "
        f"(no JSON object found; {envelope})")
    saved = _save_raw_diff(text, f"no JSON object found in response; {envelope}", stamp)
    return {
        "summary": "Could not parse diff response — no JSON object found",
        "needs_human_review": [_diff_failure_marker(text, saved, stamp)],
    }


# ---------------------------------------------------------------------------
# File patching (auto-apply mode)
# ---------------------------------------------------------------------------


def apply_changes(changes: dict, repo_path: str):
    """Apply changes to pki_compliance_mcp.py."""
    filepath = Path(repo_path) / "pki_compliance_mcp.py"
    content = filepath.read_text()

    applied = []

    # Apply document version updates
    for update in changes.get("document_version_updates", []):
        doc_id = update["id"]
        new_version = update["new_version"]
        new_date = update.get("new_date", "")

        # Find and replace version string for this document
        pattern = rf'("id":\s*"{doc_id}"[^}}]*"version":\s*")([^"]*)'
        match = re.search(pattern, content)
        if match:
            content = content[:match.start(2)] + new_version + content[match.end(2):]
            applied.append(f"Updated {doc_id} version to {new_version}")

        if new_date:
            pattern = rf'("id":\s*"{doc_id}"[^}}]*"date":\s*")([^"]*)'
            match = re.search(pattern, content)
            if match:
                content = content[:match.start(2)] + new_date + content[match.end(2):]

    # Update DATA_FRESHNESS
    today = datetime.now().strftime("%Y-%m-%d")
    content = re.sub(
        r'"lastFullReview":\s*"[^"]*"',
        f'"lastFullReview": "{today}"',
        content,
    )
    content = re.sub(
        r'"lastUpdated":\s*"[^"]*"',
        f'"lastUpdated": "{today}"',
        content,
    )

    # Calculate next review date (30 days)
    from datetime import timedelta
    next_review = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    content = re.sub(
        r'"nextReviewDue":\s*"[^"]*"',
        f'"nextReviewDue": "{next_review}"',
        content,
    )

    applied.append(f"Updated DATA_FRESHNESS and COMPLIANCE_METADATA dates to {today}")

    filepath.write_text(content)
    return applied


def git_commit_and_push(repo_path: str, message: str):
    """Git commit and push changes."""
    subprocess.run(["git", "add", "pki_compliance_mcp.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo_path, check=True)
    subprocess.run(["git", "push"], cwd=repo_path, check=True)


def restart_api_service():
    """Restart the local API service after updating the data file."""
    subprocess.run(
        ["sudo", "systemctl", "restart", "pki-compliance-api"],
        check=True,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(message: str):
    """Log to file and stdout."""
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"
    print(line)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Derive from DATA_DIR at call time so tests that redirect DATA_DIR
    # don't write into the real log (LOG_FILE is frozen at import).
    with open(DATA_DIR / LOG_FILE.name, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Cost-control gating (decide whether to spend tokens at all)
# ---------------------------------------------------------------------------

# Pure-JSON one-liner run inside the MCP container; reuses the same machinery
# as daily_doc_check.sh. Prints the check_all_documents JSON to stdout.
# persist=False is load-bearing, not a tidy-up. check_all_documents WRITES the
# new hashes and saves unless told otherwise, so this gate — which runs at 10:00
# against the same container the 10:30 daily_doc_check uses — used to consume
# every change before the doc-check could report it to the daily email. The
# email then said "0 doc hash changes" on precisely the days a document changed
# (2026-08-02 apple_root_program, 2026-08-06 microsoft_root_announcements).
# The gate only needs to READ; the 10:30 check is what should consume.
_DETECT_SNIPPET = (
    "import asyncio;"
    "from pki_compliance_mcp import check_all_documents,CheckAllDocumentsInput,ResponseFormat;"
    "print(asyncio.run(check_all_documents("
    "CheckAllDocumentsInput(persist=False,response_format=ResponseFormat.JSON))))"
)


def detect_document_changes() -> tuple[int, list[str]]:
    """Cheap change detection via the container. No Anthropic API calls.

    Returns (changes_detected, changed_doc_ids). changes_detected == -1 means
    detection itself failed (container down, etc.). Slow upstream fetches can
    push a single sweep past the subprocess timeout (seen 2026-07-11), so one
    transient failure gets one retry before we give up for the day.
    """
    for attempt in (1, 2):
        try:
            out = subprocess.run(
                ["docker", "exec", PKI_CONTAINER, "python3", "-c", _DETECT_SNIPPET],
                capture_output=True, text=True, timeout=300, check=True,
            ).stdout.strip()
            data = json.loads(out)
            changed = [d["document_id"] for d in data.get("documents", []) if d.get("changed")]
            return int(data.get("changes_detected", 0)), changed
        except Exception as e:
            log(f"WARNING: document change detection failed (attempt {attempt}/2): {e}")
            if attempt == 1:
                time.sleep(60)
    return -1, []


def days_since_last_research() -> float:
    """Days since research last actually ran. Large number if never."""
    try:
        ts = json.loads(LAST_RESEARCH_FILE.read_text())["timestamp"]
        last = datetime.fromisoformat(ts)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last).total_seconds() / 86400.0
    except Exception:
        return 1e9


def mark_research_ran() -> None:
    LAST_RESEARCH_FILE.write_text(
        json.dumps({"timestamp": datetime.now(timezone.utc).isoformat()})
    )


def should_run_research(force: bool) -> tuple[bool, str]:
    """Gate the expensive research. Returns (run?, human-readable reason)."""
    if force:
        return True, "forced (--force)"

    changes, changed_ids = detect_document_changes()
    if changes > 0:
        return True, f"{changes} document(s) changed: {', '.join(changed_ids)}"

    stale_days = days_since_last_research()
    if stale_days >= MAX_DAYS_BETWEEN_RESEARCH:
        net = "weekly safety net" + (" (detector unavailable)" if changes == -1 else "")
        return True, f"{net}: {stale_days:.1f}d since last research >= {MAX_DAYS_BETWEEN_RESEARCH}d"

    if changes == -1:
        return False, (
            f"change detector unavailable and only {stale_days:.1f}d since last research "
            f"(< {MAX_DAYS_BETWEEN_RESEARCH}d) — skipping to protect cost"
        )
    return False, f"no document changes; {stale_days:.1f}d since last research — skipping"


# ---------------------------------------------------------------------------
# Dedup (don't re-propose already-applied or previously-rejected items)
# ---------------------------------------------------------------------------


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _sig(item: dict) -> str:
    """Fuzzy signature: normalized title + date. Catches dupes with new ids."""
    return f"{_norm(item.get('title'))}|{item.get('date', '')}"


# Review flags have no title/date and the model invents a different id every
# day for the same topic, so they get their own signature: the set of "anchor"
# tokens (ballot numbers, regulation names) found in id+description+reason.
# Regulation names alone (nis2, dora) are too coarse: every NIS2 story would
# share one bucket, so rejecting one would suppress them all, and distinct
# stories collapse to a single outstanding row (2026-07-21: a CJEU-referral
# flag silently merged into a transposition flag's bucket). The event-type
# qualifiers (cjeu, ctpp, transposition, ransomware) split those buckets.
_REVIEW_ANCHOR_RE = re.compile(
    r"\b("
    r"sc[ -]?\d{2,4}(?:v\d+)?"       # server cert ballots: SC087v2, SC101, SC0101v2
    r"|csc[ -]?\d+(?:v\d+)?"         # code signing ballots: CSC-32
    r"|smc[ -]?\d{2,4}(?:v\d+)?"     # S/MIME ballots: SMC017v2
    r"|ns[ -]?\d{2,4}(?:v\d+)?"      # NetSec ballots: NS-010, NS-008v3
    r"|cscwg[ -]?\d+"
    r"|nis[ -]?2"
    r"|dora"
    r"|cjeu"                         # EU Court of Justice referrals
    r"|ctpps?"                       # DORA critical ICT third-party providers
    r"|transpos(?:e[sd]?|ing|itions?)"  # national transposition (NIS2 etc.)
    r"|ransomware"                   # UK ransomware/cyber-extortion reporting bills
    r"|nspm[ -]?12"
    r"|eo[ -]?14\d{3}"               # executive orders: EO 14412, EO 14413
    r"|m[ -]?2[0-9][ -]?\d{2}"       # OMB memos: M-23-02, M-26-15
    r"|ir[ -]?8\d{3}"                # NIST IRs: 8547, 8647
    r"|sp[ -]?800[ -]?\d{2,3}[a-d]?" # NIST SPs: SP 800-73, 800-131A
    r"|secure[ -]?boot"
    r"|mrsp"
    r"|cross[ -]?sign(?:ed|ing)?"    # MS driver-signing / cross-sign trust removals
    r"|kernel[ -]?drivers?"
    r"|apple[ -]root[ -](?:certificate[ -])?program"
    r"|dedicated[ -]tls"
    r"|cyber security and resilience|uk[ -]?csr"
    r")\b",
    re.IGNORECASE,
)


# Grammatical variants of the same topic must produce identical anchor tokens.
_ANCHOR_CANON = {
    "crosssigned": "crosssign",
    "crosssigning": "crosssign",
    "kerneldrivers": "kerneldriver",
    "cybersecurityandresilience": "ukcsr",
    "ctpps": "ctpp",
    "transpose": "transposition",
    "transposes": "transposition",
    "transposed": "transposition",
    "transposing": "transposition",
    "transpositions": "transposition",
}


def _canon_anchor(a: str) -> str:
    """Canonical anchor token: lowercase, no spaces/dashes, variant-folded.
    Ballot references vary day to day in zero-padding and version suffix
    (SC101 / SC0101v2 / SC0101 are the same ballot family)."""
    a = re.sub(r"[\s-]", "", a.lower())
    a = _ANCHOR_CANON.get(a, a)
    b = re.match(r"^(sc|csc|smc|cscwg|ns)0*(\d+)(?:v\d+)?$", a)
    if b:
        a = b.group(1) + b.group(2)
    return a


# Tracked CA/B Forum documents, matched by BOTH the id a proposal uses and the
# prose name a human-review flag uses. A "[doc bump] ev-guidelines" proposal and
# a flag titled "EV Guidelines Version Discrepancy (v2.0.3 vs v2.0.2)" are the
# same topic, but share no ballot code, so ballot anchors alone never connect
# them. (2026-07-29 and again 07-31: the ev-guidelines 2.0.2 -> 2.0.3 bump
# auto-applied with exactly that flag open — the same failure class as the
# 2026-07-12 SC0101v2 incident, benign only because the data happened to be
# right.) Tokens are the canonicalized doc ids, so a manual hold written as the
# doc id ("ev-guidelines") lands on the same token.
_DOC_ANCHOR_RES = [
    ("evguidelines", re.compile(
        r"\bev[\s-]?guidelines\b"
        r"|\bextended[\s-]validation(?:[\s-](?:ssl|tls|certificate))*[\s-]guidelines\b",
        re.IGNORECASE)),
    ("smimebr", re.compile(
        r"\bs[\s/]?mime[\s-]?(?:brs?|baseline[\s-]requirements)\b", re.IGNORECASE)),
    ("tlsbr", re.compile(
        r"\btls[\s-]?(?:brs?|baseline[\s-]requirements)\b"
        r"|\bserver[\s-]certificate[\s-]baseline[\s-]requirements\b", re.IGNORECASE)),
    ("codesigningbr", re.compile(
        r"\bcode[\s-]?signing[\s-]?(?:brs?|baseline[\s-]requirements)\b", re.IGNORECASE)),
    ("netsec", re.compile(
        r"\bnetsec\b|\bnetwork[\s-]security[\s-]requirements\b", re.IGNORECASE)),
]

# Anchors that identify a topic on their own. A CJEU referral is one story
# however many regulations the model names alongside it; letting those co-occur
# in the signature split it into a new bucket every rewording (2026-07-13,
# 07-21, 07-29, 07-31: four ids for one flag, alternating between
# cjeu+nis2+transposition and cjeu+dora+nis2+transposition). Collapsing to the
# dominant token affects the dedup signature ONLY — hold matching still uses the
# full anchor set, so nothing that was held stops being held.
_DOMINANT_ANCHORS = {"cjeu"}


def _anchor_text(item) -> str:
    """The item text anchors are read from (id + human-written prose fields)."""
    if isinstance(item, dict):
        return " ".join(str(item.get(k) or "")
                        for k in ("id", "title", "topic", "description", "reason"))
    return str(item)


def _review_anchors(item) -> set:
    """Every ballot/regulation anchor token in an item's text, uncollapsed."""
    return {_canon_anchor(m.group(1))
            for m in _REVIEW_ANCHOR_RE.finditer(_anchor_text(item))}


def document_anchors(item) -> set:
    """Tracked-document tokens for an item, matched by doc id OR prose name."""
    text = _anchor_text(item)
    return {tok for tok, pat in _DOC_ANCHOR_RES if pat.search(text)}


def hold_anchors_for(item) -> set:
    """Every hold token an item can match: anchors + tracked-document names.

    Wider than the dedup signature on purpose — an unmatched hold ships an
    unreviewed change, an over-matched one only queues something for a human."""
    return _review_anchors(item) | document_anchors(item)


def _review_sig(item) -> str:
    """Signature for a needs_human_review flag, stable across model runs."""
    anchors = _review_anchors(item)
    dominant = anchors & _DOMINANT_ANCHORS
    if dominant:
        anchors = dominant
    if anchors:
        return "anchors:" + "+".join(sorted(anchors))
    return "text:" + " ".join(_norm(_anchor_text(item)).split()[:12])


# ---------------------------------------------------------------------------
# Content candidates (Part A, 2026-08-07)
#
# A fifth outcome next to applied/rejected/held/nothing: real, relevant,
# well-sourced, but no future date certain — so it becomes content, not a
# tracked deadline, and never enters the human review queue. The Compliance
# Tracking Scope Standard (2026-07-22) enforced in code.
#
# EMISSION SCOPING — the load-bearing rule: the content-candidate ledger
# (content_candidate_sink.py) suppresses CONTENT EMISSION ONLY. Nothing on the
# deadline path (dedup_changes, auto_approve.classify) ever reads it, so an
# item that later acquires a date certain re-qualifies as a deadline on its
# next pass regardless of content-candidate history. The two checks live in:
#   - content emission: content_candidate_sink.process() (ledger consult)
#   - deadline path:    dedup_changes() / auto_approve.classify() (unchanged)
# ---------------------------------------------------------------------------

# Code-side backstop for the model-layer classification in DIFF_SYSTEM_PROMPT.
# Rules 1-4 are pattern-matchable; rule 5 ("no date certain") is deliberately
# model-only — code cannot tell "undated because legislative-stage" from
# "undated because it's an operational flag", and a wrong guess here silently
# eats a real Tier 1 item.
CONTENT_CANDIDATE_RULES = [
    # Order matters: "second reading" is stage language in BOTH Westminster
    # and the EU procedure, so the more specific bill rule (which requires
    # "Bill" near the stage words) must run before the EU rule.
    ("pre-royal-assent-bill", re.compile(
        # "received/granted Royal Assent" is an ADOPTED law, not a bill — only
        # pre-assent stage language may match.
        r"(awaiting|before|pending|prior to) royal assent"
        r"|\bbill\b[^.]{0,120}\b(second reading|third reading|committee stage"
        r"|report stage|introduced in (parliament|the house)"
        r"|house of (commons|lords))", re.IGNORECASE)),
    ("eu-legislative-procedure", re.compile(
        r"ordinary legislative procedure|trilogue|council position"
        r"|(first|second) reading|provisional (political )?agreement"
        r"|proposal for a (regulation|directive)|draft (eu )?(regulation|directive)"
        r"|com\(\d{4}\)\s*\d+", re.IGNORECASE)),
    ("enforcement-action", re.compile(
        # cjeu[-\s]: ids arrive hyphenated ("nis2-cjeu-referral-..."), and the
        # 2026-07-21 flag was missed on exactly that (space-only pattern).
        r"cjeu[-\s]referral|referr(ed|al|ing)[^.]{0,80}\b(cjeu|court of justice)"
        r"|infringement proceeding|reasoned opinion|letter of formal notice",
        re.IGNORECASE)),
    ("draft-standard-comment-period", re.compile(
        r"initial public draft|\bipd\b|\biwd\b|internal working draft"
        r"|(public )?comment period (open|opens|closes|closing|ends)"
        r"|draft for (public )?comment|request for (public )?comments? on the draft",
        re.IGNORECASE)),
]

# Fixed vocabulary the model may use for content_rule (rules above + model-only
# rule 5). Anything else is normalized to "no-date-certain" so logs stay greppable.
CONTENT_RULE_NAMES = {name for name, _ in CONTENT_CANDIDATE_RULES} | {"no-date-certain"}

_DIFF_FAILURE_MARKER_PREFIX = "Diff response unparseable"

# Backstop for test 1 of the `urgent` rule in DIFF_SYSTEM_PROMPT ("announced,
# not speculated"). Markers that describe the ESCALATING EVENT as hypothetical.
_URGENT_SPECULATION = re.compile(
    r"\b(could|may|might)\s+(constitute|trigger|require|mandate|force|lead to"
    r"|result in|mean)"
    r"|\b(debat|discuss|argu)\w*\s+(over\s+)?whether"
    r"|\bwhether\s+(this|it|that|the\s+\w+)\s+"
    r"(triggers?|requires?|forces?|constitutes?|mandates?)"
    r"|\b(not|un)\s?confirmed\b"
    r"|\bunclear whether\b"
    r"|\bno (ruling|decision|determination|mandate)\b"
    r"|\b(potential|possible)(ly)?[\s-]+(mass[\s-]?revocation|revocation|distrust)",
    re.IGNORECASE)

# Markers that something authoritative has actually stated the event happened
# or is scheduled. Their presence protects a flag from the de-escalation above.
_URGENT_ANNOUNCEMENT = re.compile(
    r"\b(has |have )?announced\b"
    r"|\bwill (revoke|distrust|remove|be revoked|be distrusted|be removed)\b"
    r"|\brevocation (is |was )?(required|mandated|scheduled|underway|beginning)\b"
    r"|\bis revoking\b|\bhas begun revoking\b|\bnotified subscribers\b"
    r"|\b(effective|takes effect|deadline of|no later than)\s+\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE)


def deescalate_speculative_urgent(changes: dict) -> list[tuple]:
    """Drop `urgent` where the item's own text calls the event hypothetical.

    Why a code backstop exists for this at all: `urgent` is the flag that
    bypasses every calm path. It sets the 🚨 subject prefix, fires the phone
    push, and is the ONLY thing content_drafts.py selects on. On 2026-08-20 a
    research description reading "This could constitute a mass-revocation
    event" carried urgent:true straight into a five-channel content package
    that told readers to inventory their certificates. The model rule already
    said "announced"; this holds it to that word.

    Conservative in both directions, deliberately:
      - It fires only when a speculation marker is present AND no announcement
        marker is, so "CA X announced a mass revocation, unclear which certs
        are affected" keeps its flag — the speculation there is about scope,
        not about whether the event happened.
      - It clears ONLY the urgent flag. The item is not moved, reclassified or
        dropped, so it still reaches the review queue, the daily email and the
        rolling backlog exactly as before. That bounded downside is why this
        can run as code where the content-candidate reclassification could
        not: the cost of a wrong call here is a calmer email, not a lost item.

    Returns [(source_key, id, matched_text)] for logging.
    """
    hits: list[tuple] = []
    for key in ("new_deadlines", "updated_deadlines", "regulatory_updates",
                "needs_human_review"):
        items = changes.get(key)
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict) or not it.get("urgent"):
                continue
            text = _anchor_text(it)
            m = _URGENT_SPECULATION.search(text)
            if not m or _URGENT_ANNOUNCEMENT.search(text):
                continue
            it["urgent"] = False
            it["urgent_downgraded"] = m.group(0)
            iid = it.get("id") or (it.get("title") or "")[:60]
            hits.append((key, iid, m.group(0)))
    return hits


def _content_rule_match(item) -> tuple[str, str] | None:
    """(rule_name, matched_text) for the first backstop rule an item trips,
    or None. Exclusions come first — a wrong reclassification is a silent
    Tier 1 loss, so anything pipeline-operational or date-bearing stays put:
      - ballot/tracked-document anchors: operational review flags (holds!)
      - a parseable date field: has a date certain, deadline path owns it
      - diff-failure markers: fault reports, never content
    """
    if not isinstance(item, dict):
        return None
    text = _anchor_text(item)
    # substring, not startswith: _anchor_text joins id/title/... with spaces,
    # so a marker stored as description arrives with leading whitespace.
    if _DIFF_FAILURE_MARKER_PREFIX in text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(item.get("date") or "")):
        return None
    if any(re.match(r"^(sc|csc|smc|cscwg)\d+$", a) for a in _review_anchors(item)):
        return None
    if document_anchors(item):
        return None
    for name, pat in CONTENT_CANDIDATE_RULES:
        m = pat.search(text)
        if m:
            return name, m.group(0)
    return None


def classify_content_candidates(changes: dict) -> tuple[dict, list]:
    """Move rule-matching items out of the review-bound lists into
    content_candidates. Runs after sanitize and BEFORE dedup, so a previously
    rejected topic (rejected as a deadline, e.g. the CJEU referral) still gets
    its one shot at becoming content instead of being dropped silently.

    Returns (changes, moved) where moved is [(source_key, id, rule, matched)].
    Every decision is logged with the specific rule and the text that fired —
    a misclassification here is a silent failure by nature, so the log is the
    safety net.
    """
    moved: list[tuple] = []
    candidates = changes.get("content_candidates")
    if not isinstance(candidates, list):
        candidates = []
    changes["content_candidates"] = candidates

    # Normalize model-emitted candidates' rule names so logs stay greppable.
    for it in candidates:
        if isinstance(it, dict):
            if it.get("content_rule") not in CONTENT_RULE_NAMES:
                it["content_rule"] = "no-date-certain"
            it.setdefault("content_rule_matched", "(model-classified)")

    for key in ("needs_human_review", "regulatory_updates"):
        items = changes.get(key)
        if not isinstance(items, list):
            continue
        kept = []
        for it in items:
            hit = _content_rule_match(it)
            if hit is None:
                kept.append(it)
                continue
            rule, matched = hit
            it["content_rule"] = rule
            it["content_rule_matched"] = matched
            candidates.append(it)
            iid = it.get("id") or (it.get("title") or it.get("description") or "")[:60]
            moved.append((key, iid, rule, matched))
            log(f"CONTENT CANDIDATE [{rule}] from {key}: {iid!r} — matched {matched!r}")
        changes[key] = kept

    return changes, moved


def load_manual_holds() -> set:
    """Anchors from HOLDS_FILE whose hold-until date hasn't passed.

    Flag-derived holds age out with the 14-day pending-file window, which can
    lift a hold before the underlying event (ballot vote, IPR review end).
    HOLDS_FILE pins a hold to an explicit date instead. An unparseable date
    keeps the hold (fail safe — a typo must not lift a hold silently)."""
    try:
        data = json.loads(HOLDS_FILE.read_text())
    except FileNotFoundError:
        return set()
    except Exception as e:
        print(f"WARNING: unreadable {HOLDS_FILE.name} ({e}) — holding nothing from it")
        return set()
    today = datetime.now(timezone.utc).date()
    anchors = set()
    for anchor, until in data.items():
        try:
            expired = datetime.strptime(str(until), "%Y-%m-%d").date() < today
        except ValueError:
            print(f"WARNING: bad hold-until date {until!r} for {anchor!r} — keeping hold")
            expired = False
        if not expired:
            anchors.add(_canon_anchor(anchor))
    return anchors


def held_review_anchors(days: int = 14) -> set:
    """Hold tokens from non-rejected review flags in recent pending files,
    plus unexpired manual holds from HOLDS_FILE. Tokens are ballot/regulation
    anchors plus the tracked CA/B Forum documents a flag names (see
    hold_anchors_for()).

    A topic a human is still deciding on (e.g. a ballot held for IPR review)
    must not be auto-applied even if a later research run re-proposes it as a
    confirmed deadline — the hold outranks the model's confidence.
    (2026-07-12: auto-approve shipped SC0101v2 three weeks before its IPR
    review ended because it had no view of held topics.)"""
    rejected = load_rejected()
    anchors: set = load_manual_holds()
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    for f in sorted(DATA_DIR.glob("pending_updates_*.json")):
        ds = f.stem.replace("pending_updates_", "")
        try:
            if datetime.strptime(ds, "%Y-%m-%d").date() < cutoff:
                continue
        except ValueError:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for it in data.get("needs_human_review", []):
            if not isinstance(it, dict):
                it = {"description": str(it)}
            if it.get("id") in rejected["ids"] or _review_sig(it) in rejected["signatures"]:
                continue
            # hold_anchors_for(), not the signature: a flag can name a tracked
            # document in prose and carry no ballot code at all, in which case
            # its signature is a "text:" one and would contribute no hold.
            anchors.update(hold_anchors_for(it))
    return anchors


def load_prior_review_sigs(days: int = 14, exclude_date: str | None = None) -> dict:
    """Map review-flag signature -> first-seen date from recent pending files,
    so today's run can drop flags that are just yesterday's items re-worded."""
    sigs: dict[str, str] = {}
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    for f in sorted(DATA_DIR.glob("pending_updates_*.json")):
        ds = f.stem.replace("pending_updates_", "")
        if exclude_date and ds == exclude_date:
            continue
        try:
            if datetime.strptime(ds, "%Y-%m-%d").date() < cutoff:
                continue
        except ValueError:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for it in data.get("needs_human_review", []):
            sigs.setdefault(_review_sig(it), ds)
    return sigs


def _normalize_provenance(item: dict) -> dict:
    """Coerce provenance_urls to a list of http(s) URLs, capped at 3.

    The model sometimes emits a bare string instead of a list. Non-URL strings
    are dropped rather than kept — a hallucinated citation is worse than none,
    since the whole point is that the operator can click it and land on source.
    Deliberately does NOT touch any field _review_sig() hashes (id/title/topic/
    description/reason), so adding provenance cannot re-flag held items.
    """
    raw = item.get("provenance_urls")
    if raw is None:
        return _normalize_primary_url(item)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        item["provenance_urls"] = []
        return _normalize_primary_url(item)
    seen, urls = set(), []
    for u in raw:
        u = str(u or "").strip()
        if u.startswith(("http://", "https://")) and u not in seen:
            seen.add(u)
            urls.append(u)
    item["provenance_urls"] = urls[:3]
    return _normalize_primary_url(item)


def _normalize_primary_url(item: dict) -> dict:
    """Part B: "list of URLs plus which one was treated as primary".

    primary_url must be a real http(s) URL. If it names a URL that survived
    into provenance_urls, keep it; if it's a valid URL the list is missing
    (e.g. the cap cut it), prepend it so the pair stays consistent; anything
    else is dropped — a hallucinated primary is worse than none. Absent field
    stays absent: old records never grow keys they didn't have.
    """
    raw = item.get("primary_url")
    if raw is None:
        return item
    u = str(raw or "").strip()
    if not u.startswith(("http://", "https://")):
        item.pop("primary_url", None)
        return item
    urls = item.get("provenance_urls")
    if isinstance(urls, list) and u not in urls:
        item["provenance_urls"] = ([u] + urls)[:3]
    item["primary_url"] = u
    return item


def sanitize_changes(changes: dict) -> tuple[dict, list]:
    """Drop malformed proposals the diff model sometimes emits (empty dicts,
    items missing id or any human-readable text) before dedup/render."""
    dropped: list[tuple] = []

    def _label(it) -> str:
        return (it.get("id") if isinstance(it, dict) else None) or str(it)[:60]

    for key, required in (
        ("new_deadlines", ("id", "title")),
        ("updated_deadlines", ("id",)),
        ("regulatory_updates", ("id", "title|description")),
        ("document_version_updates", ("id", "new_version")),
    ):
        items = changes.get(key)
        if not isinstance(items, list):
            changes[key] = []
            continue
        kept = []
        for it in items:
            ok = isinstance(it, dict) and all(
                any(it.get(f) for f in field.split("|")) for field in required
            )
            if ok:
                kept.append(it)
            else:
                dropped.append((key, _label(it), f"malformed (missing {'/'.join(required)})"))
        changes[key] = kept

    for key in ("needs_human_review", "content_candidates"):
        items = changes.get(key)
        if not isinstance(items, list):
            changes[key] = []
            continue
        kept = []
        for it in items:
            if isinstance(it, str) and it.strip():
                kept.append({"id": None, "description": it.strip()})
            elif isinstance(it, dict) and (it.get("description") or it.get("reason") or it.get("item")):
                kept.append(_normalize_provenance(it))
            else:
                dropped.append((key, _label(it), "malformed (empty)"))
        changes[key] = kept

    return changes, dropped


def _migrate_sig(sig: str) -> str:
    """Re-collapse a stored anchor signature under today's dominant-anchor rule.

    Rejections are permanent, so a signature written before a rule change must
    keep matching — otherwise the rule change silently un-rejects a topic and it
    resurfaces once. (2026-07-31: making `cjeu` dominant would have stranded the
    stored anchors:cjeu+nis2+transposition and anchors:cjeu+dora+nis2+
    transposition rejections, letting the CJEU flag come back a fifth time.)"""
    if not sig.startswith("anchors:"):
        return sig
    anchors = set(sig[len("anchors:"):].split("+"))
    dominant = anchors & _DOMINANT_ANCHORS
    return "anchors:" + "+".join(sorted(dominant or anchors))


def load_rejected() -> dict:
    try:
        data = json.loads(REJECTED_FILE.read_text())
        return {"ids": set(data.get("ids", [])),
                "signatures": {_migrate_sig(s) for s in data.get("signatures", [])}}
    except Exception:
        return {"ids": set(), "signatures": set()}


def reject_ids(ids: list[str], days: int = 30) -> int:
    """Persist rejected item ids (+ signatures looked up from pending_updates
    files of the last `days` days) so they never resurface. Returns total
    rejected.

    Signatures must be mapped from the whole recent window, not just the
    newest file: verdicts usually target the aggregated outstanding backlog,
    whose items live in older files. (2026-07-13: all 10 rejects that day
    consulted only the newest file — an empty parse-failure stub — persisted
    as bare ids, and the next research run's freshly-invented ids sailed
    straight past them.) A bare id is still recorded, but only a signature
    survives id churn, so unmapped ids get a loud warning.

    `days` is settable from the CLI as --reject-days (2026-08-31). Widen it
    when the flag being rejected is OLDER than the default 30d: NS-010's first
    flag sat in a 31-day-old pending file, so at the default it mapped to no
    signature at all and the rejection would not have survived a rewording."""
    raw = {"ids": [], "signatures": []}
    if REJECTED_FILE.exists():
        raw = json.loads(REJECTED_FILE.read_text())
    idset, sigset = set(raw.get("ids", [])), set(raw.get("signatures", []))
    # Provenance retention on the rejection path (Part B, 2026-08-07): pending
    # files age out at 14 days, so without this snapshot a rejected item's URLs
    # evaporate and every NEEDS-SOURCE follow-up restarts from a blank search
    # (the NIS2 week-of-open-time failure). Old rejected.json files simply lack
    # the key — .get keeps them loading; nothing is backfilled.
    provmap_stored: dict = raw.get("provenance", {}) if isinstance(raw.get("provenance"), dict) else {}

    sigmap: dict[str, str] = {}
    provmap: dict[str, dict] = {}
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    for f in sorted(DATA_DIR.glob("pending_updates_*.json")):  # oldest→newest: newest wording wins
        ds = f.stem.replace("pending_updates_", "")
        try:
            if datetime.strptime(ds, "%Y-%m-%d").date() < cutoff:
                continue
        except ValueError:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue

        def _snapshot(it: dict) -> None:
            urls = [u for u in (it.get("provenance_urls") or []) if isinstance(u, str)]
            primary = it.get("primary_url") or it.get("source_url")
            if urls or primary:
                provmap[it["id"]] = {
                    "title": it.get("title") or it.get("topic") or "",
                    "provenance_urls": urls[:3],
                    "primary_url": primary if isinstance(primary, str) else None,
                    "seen_in": ds,
                }

        for key in ("new_deadlines", "updated_deadlines", "regulatory_updates"):
            for it in data.get(key, []):
                if isinstance(it, dict) and it.get("id"):
                    sigmap[it["id"]] = _sig(it)
                    _snapshot(it)
        for key, sigfn in (("needs_human_review", _review_sig),
                           ("content_candidates", _review_sig)):
            for it in data.get(key, []):
                if isinstance(it, dict) and it.get("id"):
                    sigmap[it["id"]] = sigfn(it)
                    _snapshot(it)

    for iid in ids:
        idset.add(iid)
        if iid in sigmap:
            sigset.add(sigmap[iid])
        else:
            log(f"WARNING: no signature found for {iid!r} in the last {days}d of "
                f"pending files — id-only rejection will not suppress re-worded "
                f"re-proposals of the same topic. If the flag is simply older than "
                f"{days}d, re-run with --reject-days N to widen the window")
        if iid in provmap:
            snap = dict(provmap[iid])
            snap["signature"] = sigmap.get(iid)
            snap["rejected_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            provmap_stored[iid] = snap

    out = {"ids": sorted(idset), "signatures": sorted(sigset)}
    if provmap_stored:
        out["provenance"] = provmap_stored
    REJECTED_FILE.write_text(json.dumps(out, indent=2))
    return len(idset)


def dedup_changes(changes: dict, current_data: dict | None,
                  exclude_date: str | None = None) -> tuple[dict, list]:
    """Drop proposals already in DEADLINES, already at the current doc version,
    or previously rejected. Review flags additionally dedup against the last
    14 days of pending files (by anchor signature, since their ids are not
    stable across runs). regulatory_updates additionally consult the rejected
    ANCHOR signatures — see keep() for why that backstop stops there and is
    deliberately not extended to new_deadlines.
    Returns (filtered_changes, removed[list of tuples])."""
    existing_ids, existing_sigs = set(), set()
    if current_data:
        for d in current_data.get("deadlines", []):
            existing_ids.add(d.get("id"))
            existing_sigs.add(_sig(d))
    rej = load_rejected()
    removed: list[tuple] = []

    def keep(item: dict, kind: str) -> bool:
        iid, sig = item.get("id"), _sig(item)
        if iid in existing_ids or sig in existing_sigs:
            removed.append((kind, iid, "already in DEADLINES"))
            return False
        if iid in rej["ids"] or sig in rej["signatures"]:
            removed.append((kind, iid, "previously rejected"))
            return False
        # Anchor backstop for regulatory_updates ONLY (2026-08-21).
        #
        # _sig() is normalized-title|date and an EXACT match, so a rejected
        # topic returns simply by drifting either half. On 2026-08-06 the CJEU
        # referral did exactly that — title gained one word ("france and
        # netherlands" vs the stored "france netherlands") and the date moved
        # 2026-07-08 -> 2026-07-31 — and landed in regulatory_updates, where
        # nothing consults the anchor signature that was already on file for
        # it. Either drift alone defeats _sig(). rejected.json now carries five
        # ids and four signatures for that one topic, which is the cost of
        # absorbing drift instead of catching it.
        #
        # SCOPED TO regulatory_updates DELIBERATELY, and this asymmetry is the
        # point rather than an oversight: anchors are COARSE (see the 2026-07-21
        # silent-merge bug). A real future CJEU *ruling* carrying a genuine
        # date-certain would hash to the same cjeu+nis2+transposition anchor as
        # the rejected referral. Suppressing that out of new_deadlines would be
        # a WORSE failure than the one this fixes, because a missed deadline is
        # the product's whole job. regulatory_updates carries no date-certain,
        # so a false suppression there costs a note, not a deadline.
        #
        # ANCHOR FORMS ONLY. _review_sig() falls back to "text:" + the first 12
        # normalized words when an item has no anchor tokens, and rejected.json
        # holds 8 such signatures. That fallback drifts with wording exactly
        # like _sig() does, so admitting it here would add coarseness without
        # adding stability. Only "anchors:" participates.
        if kind == "regulatory":
            asig = _review_sig(item)
            if asig.startswith("anchors:") and asig in rej["signatures"]:
                removed.append((kind, iid,
                                f"previously rejected (anchor {asig})"))
                return False
        return True

    for key, kind in (("new_deadlines", "deadline"),
                      ("updated_deadlines", "update"),
                      ("regulatory_updates", "regulatory")):
        if isinstance(changes.get(key), list):
            changes[key] = [x for x in changes[key] if keep(x, kind)]

    if isinstance(changes.get("needs_human_review"), list):
        prior = load_prior_review_sigs(exclude_date=exclude_date)
        kept = []
        for it in changes["needs_human_review"]:
            sig = _review_sig(it)
            iid = it.get("id") if isinstance(it, dict) else str(it)[:60]
            if sig in rej["signatures"]:
                removed.append(("review", iid, "previously rejected"))
            elif sig in prior:
                removed.append(("review", iid, f"re-flagged; first seen {prior[sig]}"))
            else:
                kept.append(it)
        changes["needs_human_review"] = kept

    cur_docs = {d.get("id"): d.get("version") for d in (current_data or {}).get("cabfDocuments", [])}
    if isinstance(changes.get("document_version_updates"), list):
        kept = []
        for u in changes["document_version_updates"]:
            if cur_docs.get(u.get("id")) == u.get("new_version"):
                removed.append(("doc", u.get("id"), "version already current"))
            else:
                kept.append(u)
        changes["document_version_updates"] = kept

    return changes, removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PKI Compliance Auto-Refresh")
    parser.add_argument("--auto-apply", action="store_true", help="Apply changes automatically")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout only, no files")
    parser.add_argument("--query-only", action="store_true", help="Run research queries only")
    parser.add_argument("--force", action="store_true", help="Bypass the change gate and research now")
    parser.add_argument("--reject", nargs="+", metavar="ID", help="Mark item id(s) as rejected so they stop recurring, then exit")
    parser.add_argument("--reject-days", type=int, default=30, metavar="N",
                        help="Days of pending files to search for the rejected ids' signatures "
                             "(default 30). Widen when a flag has aged past the default window — "
                             "an id whose only pending file is older than N maps to NO signature, "
                             "and an id-only rejection does not survive a rewording.")
    parser.add_argument("--list-rejected", action="store_true", help="Print the persisted rejected list and exit")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    # Maintenance subcommands (no API calls)
    if args.list_rejected:
        rej = load_rejected()
        print(json.dumps({"ids": sorted(rej["ids"]), "signatures": sorted(rej["signatures"])}, indent=2))
        return
    if args.reject:
        if args.reject_days < 1:
            parser.error("--reject-days must be at least 1")
        total = reject_ids(args.reject, days=args.reject_days)
        log(f"Rejected {len(args.reject)} item(s) (signature window {args.reject_days}d); "
            f"{total} total on the rejected list")
        return

    log(f"PKI Compliance Auto-Refresh starting - {today}")
    log(f"Mode: {'auto-apply' if args.auto_apply else 'dry-run' if args.dry_run else 'review'}")

    # Step 0: Cost gate — only spend tokens when something actually changed.
    run, reason = should_run_research(args.force)
    log(f"Research gate: {'RUN' if run else 'SKIP'} — {reason}")
    if not run:
        log("No tracked changes. Skipping research — zero Anthropic API calls this run.")
        return

    # Step 1: Run research queries
    research_results = {}
    for i, query in enumerate(RESEARCH_QUERIES):
        log(f"Researching [{i+1}/{len(RESEARCH_QUERIES)}]: {query['label']}")
        try:
            result = research(query["prompt"])
            research_results[query["id"]] = result
            log(f"  Got {len(result)} chars")
        except Exception as e:
            log(f"  ERROR: {e}")
            research_results[query["id"]] = f"Error: {e}"

        if i < len(RESEARCH_QUERIES) - 1:
            time.sleep(INTER_QUERY_DELAY)

    # Reset the staleness clock only if research actually produced results —
    # otherwise the weekly safety net keeps retrying (a retired model 404'ing
    # every call must not count as "research ran").
    #
    # The clock is NOT reset here, only decided here. It is marked once the run
    # has actually produced output, because the same reasoning extends one step
    # further down: on 2026-08-06 research succeeded, the clock was reset, and
    # then the diff step died before writing a review file — so the run
    # produced nothing while still counting as "research ran", and the next
    # six days would have skipped. A run that dies before it has output must
    # leave the safety net armed.
    research_ok = any(not str(v).startswith("Error:") for v in research_results.values())
    if not research_ok:
        log("All research queries failed — NOT marking research as ran (safety net stays armed)")

    if args.query_only:
        log("Query-only mode - printing results")
        for qid, text in research_results.items():
            print(f"\n{'='*60}")
            print(f"  {qid}")
            print(f"{'='*60}")
            print(text)
        # Query-only produced its output on stdout; that counts.
        if research_ok:
            mark_research_ran()
        log("Done (query-only)")
        return

    # Step 2: Fetch current data from local API
    log("Fetching current compliance data from API...")
    current_data = fetch_current_data()

    # Step 3: Analyze diff
    log("Analyzing diff between research and current data...")
    try:
        changes = analyze_diff(research_results, current_data)
    except Exception as e:
        log(f"ERROR analyzing diff: {e}")
        sys.exit(1)

    # Step 3a: Drop malformed proposals (empty dicts, missing id/title) before
    # they reach the review file and render as blank lines in the email.
    changes, malformed = sanitize_changes(changes)
    if malformed:
        log(f"Sanitize dropped {len(malformed)} malformed item(s):")
        for kind, iid, why in malformed:
            log(f"  - [{kind}] {iid}: {why}")

    # Step 3a2: Content-candidate reclassification (Part A). BEFORE dedup on
    # purpose: dedup drops previously-rejected review flags, and a topic
    # rejected as a deadline (e.g. a CJEU referral) is exactly the population
    # that should get its one shot at becoming content instead.
    changes, cc_moved = classify_content_candidates(changes)
    if cc_moved:
        log(f"Content-candidate backstop moved {len(cc_moved)} item(s) out of the review path")

    # Step 3a3: hold the `urgent` rule to its own word "announced". Runs here,
    # after sanitize and before dedup, so a downgraded item still flows through
    # the normal review path — only the escalation is removed, never the item.
    urgent_downgraded = deescalate_speculative_urgent(changes)
    if urgent_downgraded:
        log(f"Urgent backstop downgraded {len(urgent_downgraded)} speculative item(s):")
        for kind, iid, matched in urgent_downgraded:
            log(f"  - [{kind}] {iid}: speculative language {matched!r}, not an announcement")

    # Step 3b: Dedup against live DEADLINES + previously rejected items + the
    # last 14 days of review flags, so the morning email never re-proposes
    # things already handled or re-nags about the same pending ballots.
    # content_candidates are deliberately NOT deduped here — emission dedup is
    # the sink ledger's job, and it must never leak onto the deadline path.
    changes, removed = dedup_changes(changes, current_data, exclude_date=today)
    if removed:
        log(f"Dedup removed {len(removed)} already-known/rejected item(s):")
        for kind, iid, why in removed:
            log(f"  - [{kind}] {iid}: {why}")

    # Step 4: Generate summary
    new_count = len(changes.get("new_deadlines", []))
    updated_count = len(changes.get("updated_deadlines", []))
    doc_count = len(changes.get("document_version_updates", []))
    reg_count = len(changes.get("regulatory_updates", []))
    review_count = len(changes.get("needs_human_review", []))
    candidate_count = len(changes.get("content_candidates", []))
    total_changes = new_count + updated_count + doc_count + reg_count

    summary_lines = [
        f"PKI Compliance Auto-Refresh Summary - {today}",
        "=" * 50,
        f"Research completed: {len(RESEARCH_QUERIES)} queries",
        f"Changes found: {total_changes}",
    ]

    for d in changes.get("new_deadlines", []):
        summary_lines.append(f"  - NEW DEADLINE: {d.get('title', 'unknown')} ({d.get('date', '?')})")
    for d in changes.get("updated_deadlines", []):
        summary_lines.append(f"  - UPDATED: {d.get('id', 'unknown')}")
    for d in changes.get("document_version_updates", []):
        summary_lines.append(f"  - DOC UPDATE: {d.get('id', '?')} -> {d.get('new_version', '?')}")
    for d in changes.get("regulatory_updates", []):
        summary_lines.append(f"  - REGULATORY: {d.get('description', 'unknown')}")
    if review_count:
        summary_lines.append(f"  - NEEDS REVIEW: {review_count} item(s)")
    if candidate_count:
        for c in changes.get("content_candidates", []):
            label = c.get("title") or c.get("id") or (c.get("description") or "")[:60]
            summary_lines.append(
                f"  - CONTENT CANDIDATE [{c.get('content_rule', '?')}]: {label}")
    if removed:
        summary_lines.append(f"  - DEDUP: filtered {len(removed)} already-known/rejected item(s)")

    summary_lines.append(f"Summary: {changes.get('summary', 'N/A')}")

    summary = "\n".join(summary_lines)
    print(summary)

    if args.dry_run:
        log("Dry-run mode - no files written")
        print("\nFull changes JSON:")
        print(json.dumps(changes, indent=2))
        return

    # Step 5: Write review file
    review_file = DATA_DIR / f"pending_updates_{today}.json"
    review_file.write_text(json.dumps(changes, indent=2))
    log(f"Review file written: {review_file}")

    # The run has produced output — now the research counts as having happened.
    # Deliberately after the write, not before: see the note at the gate above.
    # --dry-run never reaches here, so a dry run does not consume the clock.
    if research_ok:
        mark_research_ran()

    # Content-candidate emission (Part A sink). Best-effort by contract: a news
    # API hiccup or missing env must never fail the morning run — failed rows
    # stay `pending` in the ledger and retry on the next pass, and the daily
    # email surfaces them either way.
    try:
        import content_candidate_sink
        content_candidate_sink.process(changes, today)
    except Exception as e:
        log(f"WARNING: content-candidate sink failed ({e}) — candidates remain "
            f"in the review file and ledger; nothing lost, will retry next run")

    if not args.auto_apply:
        summary_lines.append(f"Mode: review")
        summary_lines.append(f"Review file: {review_file}")
        summary_lines.append("Run with --auto-apply to apply changes")
        log("Review mode - changes saved but not applied")
        return

    # Step 6: Auto-apply
    if total_changes == 0:
        log("No changes to apply")
        return

    log("Applying changes...")
    try:
        applied = apply_changes(changes, PKI_REPO_PATH)
        for a in applied:
            log(f"  Applied: {a}")
    except Exception as e:
        log(f"ERROR applying changes: {e}")
        sys.exit(1)

    # Step 7: Git commit + push
    log("Committing and pushing...")
    try:
        commit_msg = f"Auto-refresh compliance data - {today}\n\n{changes.get('summary', '')}"
        git_commit_and_push(PKI_REPO_PATH, commit_msg)
        log("Git push successful")
    except Exception as e:
        log(f"ERROR git push: {e}")

    # Step 8: Restart service
    log("Restarting API service...")
    try:
        restart_api_service()
        log("Service restarted")
    except Exception as e:
        log(f"ERROR restarting service: {e}")
        try:
            subprocess.run(["sudo", "systemctl", "start", "pki-compliance-api"], check=True)
            log("Service started (was stopped)")
        except Exception as e2:
            log(f"ERROR starting service: {e2}")

    log("Done")


if __name__ == "__main__":
    main()