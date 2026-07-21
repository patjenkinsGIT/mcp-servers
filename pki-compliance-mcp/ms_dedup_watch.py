#!/usr/bin/env python3
"""Log-only watcher for re-proposals of the 2026-07-21 Microsoft deadlines.

Context: three deadlines were manually applied on 2026-07-21 after the
Microsoft TRP blind-window review (commits d69ecd7/f484f7c) and deliberately
NOT pre-seeded into rejected anchors, as a live test of the dedup-across-runs
gap (which re-flagged six already-verdicted items on 2026-07-21 after 8 days).

Runs from cron after the 10:00 UTC research chain and 10:35 email. Appends one
result block per day to LOG_FILE — including "no re-proposal" and "research
skipped" — and never rejects or modifies anything; Patrick makes the call.

Self-disables after END_DATE (the 14-day anchor-dedup window from the apply
date, plus buffer): logs one final line, after which the cron entry can be
removed.
"""
import json
from datetime import date, datetime, timezone
from pathlib import Path

DATA_DIR = Path.home() / ".pki-compliance-mcp"
LOG_FILE = DATA_DIR / "ms_dedup_watch.log"
CRON_LOG = DATA_DIR / "cron.log"
END_DATE = date(2026, 8, 4)  # 14-day anchor window from 2026-07-21

WATCHED_IDS = [
    "microsoft-april-2026-ctl-notbefore",
    "microsoft-trp-single-purpose-roots",
    "microsoft-pqc-tls-pilot",
]
# Broader net than the ids: the research model generates its own ids, so also
# flag proposals that merely talk about the same events.
KEYWORDS = [
    "trusted root program", "notbefore", "swisssign", "securesign",
    "single-purpose root", "single purpose root", "pqc pilot", "ml-dsa",
    "ancert", "trustedrootprogram",
]


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with LOG_FILE.open("a") as f:
        f.write(f"{stamp} {msg}\n")


def scan_items(path: Path):
    """Yield (item_id, matched_on) for proposals matching ids or keywords."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        yield ("<unreadable>", f"parse error: {e}")
        return
    # Walk every dict in the file regardless of exact schema.
    stack = [data]
    seen = []
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            blob = json.dumps(node).lower()
            item_id = node.get("id") if isinstance(node.get("id"), str) else None
            if item_id or "title" in node:
                hits = [i for i in WATCHED_IDS if i in blob]
                hits += [k for k in KEYWORDS if k in blob]
                if hits and (item_id, tuple(sorted(set(hits)))) not in seen:
                    seen.append((item_id, tuple(sorted(set(hits)))))
                    yield (item_id or node.get("title", "<no id>"),
                           ", ".join(sorted(set(hits))))
                continue  # don't descend into an already-matched item twice
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def main() -> int:
    today = datetime.now(timezone.utc).date()
    if today > END_DATE:
        if f"watch window ended {END_DATE}" not in (
                LOG_FILE.read_text() if LOG_FILE.exists() else ""):
            log(f"watch window ended {END_DATE} — remove the ms_dedup_watch "
                "cron entry")
        return 0

    day = today.isoformat()
    pending = DATA_DIR / f"pending_updates_{day}.json"
    queue = DATA_DIR / f"review_queue_{day}.json"

    if not pending.exists() and not queue.exists():
        # The SKIP marker carries no date; the tail of cron.log is today's run
        # since this fires right after the 10:00 UTC chain.
        tail = CRON_LOG.read_text()[-4000:] if CRON_LOG.exists() else ""
        if "Research gate: SKIP" in tail:
            log(f"{day}: research skipped (cost gate) — no pending file, "
                "nothing to check")
        else:
            log(f"{day}: no pending_updates or review_queue file found — "
                "check cron.log")
        return 0

    matches = []
    for path in (pending, queue):
        if path.exists():
            for item_id, matched_on in scan_items(path):
                matches.append(f"{path.name}: {item_id} (matched: {matched_on})")

    if matches:
        log(f"{day}: RE-PROPOSAL SUSPECTED — {len(matches)} match(es); "
            "review before rejecting:")
        for m in matches:
            log(f"    {m}")
    else:
        log(f"{day}: no re-proposal of the 2026-07-21 Microsoft items — "
            "dedup held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
