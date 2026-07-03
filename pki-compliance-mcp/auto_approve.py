#!/usr/bin/env python3
"""Tiered auto-approval for pending compliance updates.

Runs at 10:15 UTC, between the research cron (10:00, writes
pending_updates_<date>.json) and the daily email (10:35), so the email can
report "N applied automatically, M need your review".

Per run:
  1. Reads ~/.pki-compliance-mcp/pending_updates_<date>.json (real schema:
     new_deadlines / updated_deadlines / document_version_updates /
     regulatory_updates / needs_human_review).
  2. Classifies each item AUTO / REVIEW / SKIP (see criteria below).
  3. Preflight: repo working tree clean AND `git push --dry-run` succeeds.
     If either fails, EVERYTHING queues for review — fail-safe by default.
  4. Backs up pki_compliance_mcp.py to ~/.pki-compliance-mcp/backups/, patches
     the source, py_compiles it, runs both offline test suites. Any failure:
     restore backup, queue everything.
  5. Commits with an itemized message, pushes, restarts pki-compliance-api.
  6. Writes approval_log_<date>.json and review_queue_<date>.json.

Auto-apply criteria (new deadlines — ALL must pass):
  - all required fields present (id, date, title, description, source, category)
  - source_url present, on the primary-source allowlist
  - feed_confirmed is true (research sets it when the feed/doc check flagged
    the source this cycle)
  - not already in live data (id or title+date signature), not previously
    rejected (--reject list)
  - no two auto candidates touch the same id (conflict guard: both queue)
Document version updates auto-apply when the doc id is known and a version is
given. updated_deadlines / regulatory_updates ALWAYS queue for review in v1 —
freeform edits to existing entries never ship unreviewed (this also covers the
estimated-date discipline: date changes require a human eye). Deletes are not
part of the schema; nothing is ever removed automatically.

Daily cap (--max-auto, default 5): exceeding it queues everything — a burst of
"approvable" items usually means an upstream break.

Rollback: git revert the auto-approve commit (every applied op is itemized in
approval_log_<date>.json), or restore the newest ~/.pki-compliance-mcp/backups/
copy, then restart pki-compliance-api.
"""

import argparse
import json
import py_compile
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import compliance_auto_refresh as car  # shared: log, DATA_DIR, load_rejected, _sig, PKI_REPO_PATH

BACKUP_DIR = car.DATA_DIR / "backups"

REQUIRED_FIELDS = ("id", "date", "title", "description", "source", "category")

# Primary-source allowlist. Subdomains of these match; github.com counts only
# under /cabforum/.
ALLOWED_SOURCE_DOMAINS = {
    "cabforum.org",
    "chromium.org",
    "security.googleblog.com",
    "mozilla.org",
    "wiki.mozilla.org",
    "groups.google.com",          # mozilla dev-security-policy archives
    "apple.com",
    "microsoft.com",
    "digicert.com",
    "sectigo.com",
    "nist.gov",
    "csrc.nist.gov",
    "nsa.gov",
    "media.defense.gov",
    "eur-lex.europa.eu",
    "europa.eu",
    "bills.parliament.uk",
    "parliament.uk",
    "gov.uk",
    "legislation.gov.uk",
}

KNOWN_DOC_IDS = {"tls-br", "ev-guidelines", "code-signing-br", "smime-br", "netsec"}


def source_url_allowed(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme != "https" or not parsed.netloc:
        return False
    host = parsed.netloc.lower().lstrip("www.")
    if host == "github.com":
        return parsed.path.startswith("/cabforum/")
    return any(host == d or host.endswith("." + d) for d in ALLOWED_SOURCE_DOMAINS)


def classify(changes: dict, current_data: dict | None, max_auto: int):
    """Split pending changes into (auto_new, auto_docs, review, skipped).

    review/skipped are lists of (kind, item, reason).
    """
    auto_new, auto_docs, review, skipped = [], [], [], []

    existing_ids, existing_sigs = set(), set()
    cur_docs = {}
    if current_data:
        for d in current_data.get("deadlines", []):
            existing_ids.add(d.get("id"))
            existing_sigs.add(car._sig(d))
        cur_docs = {d.get("id"): d.get("version")
                    for d in current_data.get("cabfDocuments", [])}
    rejected = car.load_rejected()

    for item in changes.get("new_deadlines", []):
        iid = item.get("id")
        if iid in existing_ids or car._sig(item) in existing_sigs:
            skipped.append(("deadline", item, "already in live data"))
            continue
        if iid in rejected["ids"] or car._sig(item) in rejected["signatures"]:
            skipped.append(("deadline", item, "previously rejected"))
            continue
        missing = [f for f in REQUIRED_FIELDS if not item.get(f)]
        if missing:
            review.append(("deadline", item, f"missing required fields: {missing}"))
            continue
        if not item.get("source_url"):
            review.append(("deadline", item, "no source_url"))
            continue
        if not source_url_allowed(item["source_url"]):
            review.append(("deadline", item, f"source_url not on allowlist: {item['source_url']}"))
            continue
        if item.get("feed_confirmed") is not True:
            review.append(("deadline", item, "not feed-confirmed"))
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", item.get("date", "")):
            review.append(("deadline", item, f"bad date format: {item.get('date')}"))
            continue
        auto_new.append(item)

    # Conflict guard: two auto candidates with the same id -> both queue.
    seen, dupes = set(), set()
    for item in auto_new:
        (dupes if item["id"] in seen else seen).add(item["id"])
    if dupes:
        for item in [i for i in auto_new if i["id"] in dupes]:
            review.append(("deadline", item, f"conflicting auto candidates for id {item['id']}"))
        auto_new = [i for i in auto_new if i["id"] not in dupes]

    for upd in changes.get("document_version_updates", []):
        if upd.get("id") in KNOWN_DOC_IDS and upd.get("new_version"):
            if cur_docs.get(upd["id"]) == upd["new_version"]:
                skipped.append(("doc", upd, "version already current"))
            else:
                auto_docs.append(upd)
        else:
            review.append(("doc", upd, "unknown document id or missing version"))

    # Freeform edits never ship unreviewed (covers estimated-date discipline).
    for upd in changes.get("updated_deadlines", []):
        review.append(("update", upd, "updates to existing entries always need review"))
    for upd in changes.get("regulatory_updates", []):
        review.append(("regulatory", upd, "regulatory updates always need review"))
    for item in changes.get("needs_human_review", []):
        review.append(("flagged", item, "flagged by research"))

    if len(auto_new) + len(auto_docs) > max_auto:
        for item in auto_new:
            review.append(("deadline", item, f"daily cap exceeded ({max_auto}) — queued"))
        for upd in auto_docs:
            review.append(("doc", upd, f"daily cap exceeded ({max_auto}) — queued"))
        auto_new, auto_docs = [], []

    return auto_new, auto_docs, review, skipped


def format_deadline_entry(d: dict) -> str:
    """Render a DEADLINES entry as source text, matching file conventions."""
    lines = ["    {"]
    order = ["id", "date", "title", "description", "source", "source_url",
             "category", "isMajor", "impact", "is_estimated"]
    for key in order:
        if key not in d or d[key] is None:
            continue
        val = d[key]
        if isinstance(val, bool):
            rendered = "True" if val else "False"
        else:
            rendered = json.dumps(str(val))
        lines.append(f'        "{key}": {rendered},')
    lines.append("    },")
    return "\n".join(lines)


def patch_source(filepath: Path, auto_new: list, auto_docs: list) -> list:
    """Insert new deadlines / bump doc versions in pki_compliance_mcp.py."""
    content = filepath.read_text()
    applied = []

    if auto_new:
        anchor = "DEADLINES = [\n"
        idx = content.index(anchor) + len(anchor)
        block = "\n".join(format_deadline_entry(d) for d in auto_new) + "\n"
        content = content[:idx] + block + content[idx:]
        applied += [f"add deadline {d['id']} ({d['date']})" for d in auto_new]

    for upd in auto_docs:
        pattern = rf'("id":\s*"{upd["id"]}"[^}}]*"version":\s*")([^"]*)'
        m = re.search(pattern, content)
        if m:
            content = content[:m.start(2)] + upd["new_version"] + content[m.end(2):]
            applied.append(f"bump {upd['id']} version to {upd['new_version']}")
        if upd.get("new_date"):
            pattern = rf'("id":\s*"{upd["id"]}"[^}}]*"date":\s*")([^"]*)'
            m = re.search(pattern, content)
            if m:
                content = content[:m.start(2)] + upd["new_date"] + content[m.end(2):]

    # Bump lastUpdated + the deadlines field verification — NOT lastFullReview
    # (auto-apply is not a full review; the 45-day stale clock keeps running).
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = re.sub(r'"lastUpdated":\s*"[^"]*"', f'"lastUpdated": "{today}"', content)
    content = re.sub(
        r'("deadlines":\s*\{"verified":\s*")[^"]*',
        rf'\g<1>{today}', content, count=1)

    filepath.write_text(content)
    return applied


def run(cmd: list, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def preflight(repo: str) -> str | None:
    """Return a reason string if auto-apply must not proceed, else None."""
    r = run(["git", "status", "--porcelain", "--", "pki_compliance_mcp.py"], cwd=repo)
    if r.returncode != 0 or r.stdout.strip():
        return "repo working tree not clean"
    r = run(["git", "push", "--dry-run", "origin", "main"], cwd=repo)
    if r.returncode != 0:
        return f"git push unavailable: {(r.stderr or '').strip().splitlines()[-1] if r.stderr else 'unknown'}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--max-auto", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify and report only; no writes, no git")
    args = parser.parse_args()

    pending_file = car.DATA_DIR / f"pending_updates_{args.date}.json"
    if not pending_file.exists():
        car.log(f"auto_approve: no pending file for {args.date} (research skipped or found nothing)")
        return 0
    changes = json.loads(pending_file.read_text())
    current_data = car.fetch_current_data()

    auto_new, auto_docs, review, skipped = classify(changes, current_data, args.max_auto)

    repo = car.PKI_REPO_PATH
    blocked = None if args.dry_run else preflight(repo)
    if blocked and (auto_new or auto_docs):
        car.log(f"auto_approve: preflight failed ({blocked}) — queueing everything for review")
        review += [("deadline", i, f"preflight failed: {blocked}") for i in auto_new]
        review += [("doc", u, f"preflight failed: {blocked}") for u in auto_docs]
        auto_new, auto_docs = [], []

    applied = []
    if (auto_new or auto_docs) and not args.dry_run:
        filepath = Path(repo) / "pki_compliance_mcp.py"
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / f"pki_compliance_mcp_{args.date}.py"
        shutil.copy2(filepath, backup)
        try:
            applied = patch_source(filepath, auto_new, auto_docs)
            py_compile.compile(str(filepath), doraise=True)
            for test in ("test_gating_dedup.py", "test_source_url_and_estimates.py"):
                r = run([sys.executable, test], cwd=repo)
                if r.returncode != 0:
                    raise RuntimeError(f"{test} failed:\n{r.stdout[-2000:]}")
            msg = "Auto-approve: " + "; ".join(applied) + \
                  "\n\nApplied automatically by auto_approve.py (tiered criteria)."
            for cmd in (["git", "add", "pki_compliance_mcp.py"],
                        ["git", "commit", "-m", msg],
                        ["git", "push", "origin", "main"]):
                r = run(cmd, cwd=repo)
                if r.returncode != 0:
                    raise RuntimeError(f"{' '.join(cmd[:2])} failed: {r.stderr[-500:]}")
            run(["systemctl", "restart", "pki-compliance-api"])
            car.log(f"auto_approve: applied {len(applied)} change(s), pushed, service restarted")
        except Exception as e:
            shutil.copy2(backup, filepath)
            # checkout HEAD resets both index and worktree — a plain
            # "checkout --" restores from the index, which may hold the
            # staged patch if the commit itself failed.
            run(["git", "checkout", "HEAD", "--", "pki_compliance_mcp.py"], cwd=repo)
            car.log(f"auto_approve: APPLY FAILED, rolled back — {e}")
            review += [("deadline", i, f"apply failed, rolled back: {e}") for i in auto_new]
            review += [("doc", u, f"apply failed, rolled back: {e}") for u in auto_docs]
            applied, auto_new, auto_docs = [], [], []

    if args.dry_run:
        car.log(f"auto_approve DRY RUN: would apply {len(auto_new)} deadline(s) + "
                f"{len(auto_docs)} doc bump(s); {len(review)} to review; {len(skipped)} skipped")
        for kind, item, reason in review:
            car.log(f"  REVIEW [{kind}] {item.get('id', item) if isinstance(item, dict) else item}: {reason}")
        return 0

    now = datetime.now(timezone.utc).isoformat()
    (car.DATA_DIR / f"approval_log_{args.date}.json").write_text(json.dumps({
        "timestamp": now,
        "applied": applied,
        "auto_deadlines": [d["id"] for d in auto_new],
        "auto_docs": [u["id"] for u in auto_docs],
        "skipped": [{"kind": k, "id": (i.get("id") if isinstance(i, dict) else str(i)), "reason": r}
                     for k, i, r in skipped],
        "blocked": blocked,
    }, indent=2))
    (car.DATA_DIR / f"review_queue_{args.date}.json").write_text(json.dumps({
        "timestamp": now,
        "items": [{"kind": k, "item": i, "reason": r} for k, i, r in review],
    }, indent=2))
    car.log(f"auto_approve: {len(applied)} applied, {len(review)} queued for review, "
            f"{len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
