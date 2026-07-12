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


def research(prompt: str, max_retries: int = 3) -> str:
    """Call Claude API with web_search tool to research PKI updates."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

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
        return "\n".join(
            block["text"] for block in data["content"] if block["type"] == "text"
        )
    raise RuntimeError(f"Rate limited or timed out after {max_retries} retries")


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
  "summary": "brief description of changes found"
}

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
  "urgent": true/false (true ONLY for items demanding action on a weeks-not-months timescale: an announced mass revocation event, distrust/removal of a currently trusted root or CA, or a new compliance obligation taking effect within ~60 days; default false)
}

needs_human_review items may also carry "urgent": true under the same criteria \
(e.g. an unverified report of an imminent mass revocation still deserves the flag).

Each document_version_update:
{
  "id": "tls-br|ev-guidelines|code-signing-br|smime-br|netsec",
  "new_version": "X.Y.Z",
  "new_date": "Mon YYYY"
}

If no changes are found, return:
{"new_deadlines":[],"updated_deadlines":[],"document_version_updates":[],"regulatory_updates":[],"needs_human_review":[],"summary":"No changes found"}
"""


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

    prompt = (
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
                "max_tokens": 16000,
                "system": DIFF_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        text = "\n".join(
            block["text"] for block in data["content"] if block["type"] == "text"
        )

    # Extract JSON from response
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json.loads(json_match.group())
    return {"summary": "Could not parse diff response", "needs_human_review": [text]}


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
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Cost-control gating (decide whether to spend tokens at all)
# ---------------------------------------------------------------------------

# Pure-JSON one-liner run inside the MCP container; reuses the same machinery
# as daily_doc_check.sh. Prints the check_all_documents JSON to stdout.
_DETECT_SNIPPET = (
    "import asyncio;"
    "from pki_compliance_mcp import check_all_documents,CheckAllDocumentsInput,ResponseFormat;"
    "print(asyncio.run(check_all_documents("
    "CheckAllDocumentsInput(response_format=ResponseFormat.JSON))))"
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
_REVIEW_ANCHOR_RE = re.compile(
    r"\b("
    r"sc[ -]?\d{2,4}(?:v\d+)?"       # server cert ballots: SC087v2, SC101, SC0101v2
    r"|csc[ -]?\d+(?:v\d+)?"         # code signing ballots: CSC-32
    r"|smc[ -]?\d{2,4}(?:v\d+)?"     # S/MIME ballots: SMC017v2
    r"|cscwg[ -]?\d+"
    r"|nis[ -]?2"
    r"|dora"
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
}


def _review_sig(item) -> str:
    """Signature for a needs_human_review flag, stable across model runs."""
    if isinstance(item, dict):
        text = " ".join(str(item.get(k) or "")
                        for k in ("id", "title", "topic", "description", "reason"))
    else:
        text = str(item)
    anchors = set()
    for m in _REVIEW_ANCHOR_RE.finditer(text):
        a = re.sub(r"[\s-]", "", m.group(1).lower())
        a = _ANCHOR_CANON.get(a, a)
        # Ballot references vary day to day in zero-padding and version suffix
        # (SC101 / SC0101v2 / SC0101 are the same ballot family) — canonicalize.
        b = re.match(r"^(sc|csc|smc|cscwg)0*(\d+)(?:v\d+)?$", a)
        if b:
            a = b.group(1) + b.group(2)
        anchors.add(a)
    if anchors:
        return "anchors:" + "+".join(sorted(anchors))
    return "text:" + " ".join(_norm(text).split()[:12])


def held_review_anchors(days: int = 14) -> set:
    """Anchor tokens from non-rejected review flags in recent pending files.

    A topic a human is still deciding on (e.g. a ballot held for IPR review)
    must not be auto-applied even if a later research run re-proposes it as a
    confirmed deadline — the hold outranks the model's confidence.
    (2026-07-12: auto-approve shipped SC0101v2 three weeks before its IPR
    review ended because it had no view of held topics.)"""
    rejected = load_rejected()
    anchors: set = set()
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
            sig = _review_sig(it)
            if it.get("id") in rejected["ids"] or sig in rejected["signatures"]:
                continue
            if sig.startswith("anchors:"):
                anchors.update(sig[len("anchors:"):].split("+"))
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

    review = changes.get("needs_human_review")
    if not isinstance(review, list):
        changes["needs_human_review"] = []
    else:
        kept = []
        for it in review:
            if isinstance(it, str) and it.strip():
                kept.append({"id": None, "description": it.strip()})
            elif isinstance(it, dict) and (it.get("description") or it.get("reason") or it.get("item")):
                kept.append(it)
            else:
                dropped.append(("needs_human_review", _label(it), "malformed (empty)"))
        changes["needs_human_review"] = kept

    return changes, dropped


def load_rejected() -> dict:
    try:
        data = json.loads(REJECTED_FILE.read_text())
        return {"ids": set(data.get("ids", [])), "signatures": set(data.get("signatures", []))}
    except Exception:
        return {"ids": set(), "signatures": set()}


def reject_ids(ids: list[str]) -> int:
    """Persist rejected item ids (+ signatures looked up from the latest
    pending_updates file) so they never resurface. Returns total rejected."""
    raw = {"ids": [], "signatures": []}
    if REJECTED_FILE.exists():
        raw = json.loads(REJECTED_FILE.read_text())
    idset, sigset = set(raw.get("ids", [])), set(raw.get("signatures", []))

    sigmap: dict[str, str] = {}
    files = sorted(DATA_DIR.glob("pending_updates_*.json"))
    if files:
        try:
            data = json.loads(files[-1].read_text())
            for key in ("new_deadlines", "updated_deadlines", "regulatory_updates"):
                for it in data.get(key, []):
                    if it.get("id"):
                        sigmap[it["id"]] = _sig(it)
            for it in data.get("needs_human_review", []):
                if isinstance(it, dict) and it.get("id"):
                    sigmap[it["id"]] = _review_sig(it)
        except Exception:
            pass

    for iid in ids:
        idset.add(iid)
        if iid in sigmap:
            sigset.add(sigmap[iid])

    REJECTED_FILE.write_text(
        json.dumps({"ids": sorted(idset), "signatures": sorted(sigset)}, indent=2)
    )
    return len(idset)


def dedup_changes(changes: dict, current_data: dict | None,
                  exclude_date: str | None = None) -> tuple[dict, list]:
    """Drop proposals already in DEADLINES, already at the current doc version,
    or previously rejected. Review flags additionally dedup against the last
    14 days of pending files (by anchor signature, since their ids are not
    stable across runs). Returns (filtered_changes, removed[list of tuples])."""
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
        total = reject_ids(args.reject)
        log(f"Rejected {len(args.reject)} item(s); {total} total on the rejected list")
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
    if any(not str(v).startswith("Error:") for v in research_results.values()):
        mark_research_ran()
    else:
        log("All research queries failed — NOT marking research as ran (safety net stays armed)")

    if args.query_only:
        log("Query-only mode - printing results")
        for qid, text in research_results.items():
            print(f"\n{'='*60}")
            print(f"  {qid}")
            print(f"{'='*60}")
            print(text)
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

    # Step 3b: Dedup against live DEADLINES + previously rejected items + the
    # last 14 days of review flags, so the morning email never re-proposes
    # things already handled or re-nags about the same pending ballots.
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