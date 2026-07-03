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
from datetime import datetime, timezone
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
MODEL = "claude-sonnet-4-20250514"
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
        with httpx.Client(timeout=180) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": MODEL,
                    "max_tokens": 4096,
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
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
    raise RuntimeError(f"Rate limited after {max_retries} retries")


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
  "category": "certificates|validation|revocation|...",
  "isMajor": true/false,
  "impact": "Brief impact statement"
}

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

    with httpx.Client(timeout=120) as client:
        response = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": MODEL,
                "max_tokens": 4096,
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
    detection itself failed (container down, etc.).
    """
    try:
        out = subprocess.run(
            ["docker", "exec", PKI_CONTAINER, "python3", "-c", _DETECT_SNIPPET],
            capture_output=True, text=True, timeout=300, check=True,
        ).stdout.strip()
        data = json.loads(out)
        changed = [d["document_id"] for d in data.get("documents", []) if d.get("changed")]
        return int(data.get("changes_detected", 0)), changed
    except Exception as e:
        log(f"WARNING: document change detection failed: {e}")
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


def dedup_changes(changes: dict, current_data: dict | None) -> tuple[dict, list]:
    """Drop proposals already in DEADLINES, already at the current doc version,
    or previously rejected. Returns (filtered_changes, removed[list of tuples])."""
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

    # Reset the staleness clock: research did run this cycle.
    mark_research_ran()

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

    # Step 3b: Dedup against live DEADLINES + previously rejected items, so the
    # morning email never re-proposes things already handled.
    changes, removed = dedup_changes(changes, current_data)
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