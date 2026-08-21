#!/usr/bin/env python3
"""
Daily PKI Compliance summary email.

Composes a concise HTML summary of the two morning crons and sends it via
Resend. Run from host crontab at 10:35 UTC (5 min after daily_doc_check.sh).

EVENT-DRIVEN SINCE 2026-08-20: on a day where nothing happened, this sends
NOTHING. See decide_send() for the exact gate. The change-day path is
untouched — any signal at all and the email fires same-day exactly as before.
Every run records itself in email_runs.json whether or not it sends, and
weekly_rollup.py turns that ledger into the heartbeat that makes the silence
trustworthy. A missing rollup, not a missing daily, is the alarm.

Inputs:
  - /root/.pki-compliance-mcp/pending_updates_YYYY-MM-DD.json (from 10:00 cron)
  - /root/.pki-compliance-mcp/doc_check.log                   (from 10:30 cron)
  - /root/.pki-compliance-mcp/cron.log                        (10:00 cron logs)
  - /root/.pki-compliance-mcp/urgent_notice_YYYY-MM-DD.json   (content_drafts.py)

Outputs:
  - /root/.pki-compliance-mcp/email_runs.json  (run ledger, read by the rollup)

Env required:
  RESEND_API_KEY=re_...

Env optional:
  PKI_EMAIL_TO=patrick@fixmycert.com         (default if unset)
  PKI_EMAIL_FROM=noreply@mail.fixmycert.com  (default if unset)

Usage:
  python3 daily_email.py                 # normal cron run; may suppress
  python3 daily_email.py --force         # send even on a no-op day
  python3 daily_email.py --dry-run       # print the decision + subject, no send,
                                         # no ledger write
  python3 daily_email.py --explain       # print the gate decision only and exit

Exit codes: 0 ok (sent OR deliberately suppressed), 1 API key missing,
2 send failed.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import httpx

DATA_DIR = Path("/root/.pki-compliance-mcp")
DASHBOARD_URL = "https://compliance-api.fixmycert.com/dashboard"
DEFAULT_TO = "patrick@fixmycert.com"
DEFAULT_FROM = "noreply@mail.fixmycert.com"

# Part A safety window (added 2026-08-07): surface content candidates in this
# email while the classifier is new, so misclassified Tier 1 items are caught
# by eye instead of discovered when a deadline passes. Flip to False in this
# one place to stop surfacing them once the classifier has earned trust.
# Rows stuck `pending` in the sink ledger are shown REGARDLESS of this flag —
# a candidate the news desk never received must stay visible somewhere.
SURFACE_CONTENT_CANDIDATES = True


def today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def read_pending_updates(date_str: str) -> dict | None:
    p = DATA_DIR / f"pending_updates_{date_str}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_parse_error": str(e)}


def wait_for_morning_chain(date_str: str, max_wait_s: int = 1500) -> None:
    """Block until the 10:00 research→auto-approve→content-drafts chain ends.

    Research can run past 10:35 on slow days (2026-07-12: 35 minutes, and the
    email reported on a half-done run). content_drafts.py is last in the
    chain (added 2026-07-16) and always writes a log line — even on
    no-urgent-items days — so poll for today's entry there. Safety valve: if
    auto_approve.log shows today but content_drafts.log stays silent for 5
    minutes after, assume the drafts step crashed and don't hold the email.
    Gives up after max_wait_s and lets the email report what it sees.
    """
    drafts_p = DATA_DIR / "content_drafts.log"
    approve_p = DATA_DIR / "auto_approve.log"

    def _has_today(p: Path) -> bool:
        try:
            return any(date_str in ln for ln in p.read_text(errors="replace").splitlines())
        except FileNotFoundError:
            return False

    deadline = time.time() + max_wait_s
    approve_seen_at = None
    while time.time() < deadline:
        if _has_today(drafts_p):
            return
        if approve_seen_at is None and _has_today(approve_p):
            approve_seen_at = time.time()
        if approve_seen_at is not None and time.time() - approve_seen_at > 300:
            return
        time.sleep(30)


def outstanding_review_items(date_str: str, days: int = 14) -> list[dict]:
    """Aggregate every unresolved queued/flagged item from recent pending files.

    Makes each daily email show the full review backlog, so skipping a few
    days of email costs nothing. An item is outstanding unless it was rejected
    (id or signature) or — for proposals — already present in the live
    compliance data (applied by auto-approve or a morning review). Review
    flags collapse across days by anchor signature, keeping first-seen date.
    Reuses the research pipeline's signature machinery so "same item" here
    means exactly what it means to the dedup pass.
    """
    import compliance_auto_refresh as car

    rejected = car.load_rejected()
    live_ids: set = set()
    live_sigs: set = set()
    try:  # live data unavailable -> degrade to showing possibly-extra items
        r = httpx.get("https://compliance-api.fixmycert.com/api/compliance-data", timeout=30)
        r.raise_for_status()
        for dl in r.json().get("deadlines", []):
            live_ids.add(dl.get("id"))
            live_sigs.add(car._sig(dl))
    except Exception:
        pass

    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    seen: dict = {}
    rejected_sigs: set = set()  # a rejected id kills its whole same-topic bucket
    for f in sorted(DATA_DIR.glob("pending_updates_*.json")):
        ds = f.stem.replace("pending_updates_", "")
        if ds < cutoff or ds > date_str:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for key, kind in (("new_deadlines", "deadline"), ("updated_deadlines", "update"),
                          ("regulatory_updates", "regulatory")):
            for it in data.get(key, []):
                if not isinstance(it, dict):
                    continue
                sig = car._sig(it)
                if it.get("id") in rejected["ids"]:
                    rejected_sigs.add(("proposal", sig))
                    continue
                if (sig in rejected["signatures"]
                        or it.get("id") in live_ids or sig in live_sigs):
                    continue
                seen.setdefault(("proposal", sig), {
                    "kind": kind,
                    "title": it.get("title") or it.get("id") or "",
                    "date": it.get("date", ""),
                    "first_seen": ds,
                    "urgent": bool(it.get("urgent")),
                })
        for it in data.get("needs_human_review", []):
            if not isinstance(it, dict):
                it = {"description": str(it)}
            rsig = car._review_sig(it)
            if it.get("id") in rejected["ids"]:
                rejected_sigs.add(("flag", rsig))
                continue
            if rsig in rejected["signatures"]:
                continue
            title = (it.get("title") or it.get("id") or it.get("topic")
                     or (it.get("description") or it.get("reason") or "")[:80])
            if not title:
                continue
            entry = seen.setdefault(("flag", rsig), {
                "kind": "flagged",
                "title": title,
                "date": "",
                "first_seen": ds,
                "urgent": bool(it.get("urgent")),
                "provenance_urls": [],
            })
            # setdefault keeps the earliest sighting (that's the first_seen we
            # want), but provenance may only appear on a later re-flag — an item
            # first raised before 2026-07-29 has none at all. Backfill so the
            # operator gets a link as soon as any run supplies one.
            if not entry.get("provenance_urls"):
                entry["provenance_urls"] = [
                    u for u in (it.get("provenance_urls") or []) if isinstance(u, str)
                ][:3]
    return sorted((v for k, v in seen.items() if k not in rejected_sigs),
                  key=lambda x: (not x["urgent"], x["first_seen"]))


def read_content_candidates(date_str: str, pending: dict | None) -> dict:
    """Content-candidate state for the email: today's candidates (from the
    pending file) with their sink status, plus any ledger rows still pending
    delivery from earlier runs (a stuck row means the news desk never got it).
    """
    try:
        ledger = json.loads((DATA_DIR / "content_candidates.json").read_text())
        if not isinstance(ledger, dict):
            ledger = {}
    except Exception:
        ledger = {}

    import compliance_auto_refresh as car

    today = []
    today_sigs = set()
    if pending and "_parse_error" not in pending:
        for it in pending.get("content_candidates", []):
            if not isinstance(it, dict):
                continue
            sig = car._review_sig(it)
            today_sigs.add(sig)
            row = ledger.get(sig) if isinstance(ledger.get(sig), dict) else {}
            today.append({
                "title": (it.get("title") or it.get("topic") or it.get("id")
                          or (it.get("description") or "")[:80]),
                "rule": it.get("content_rule", "?"),
                "matched": it.get("content_rule_matched", ""),
                "sink_status": row.get("sink_status", "not recorded"),
                "news_id": row.get("news_id", ""),
                "primary_url": it.get("primary_url")
                               or next(iter(it.get("provenance_urls") or []), ""),
            })

    stuck = []
    for sig, row in ledger.items():
        if (isinstance(row, dict) and row.get("sink_status") == "pending"
                and sig not in today_sigs):
            stuck.append({
                "title": row.get("title", sig),
                "rule": row.get("rule", "?"),
                "first_seen": row.get("first_seen", "?"),
                "attempts": row.get("attempts", 0),
            })
    return {"today": today, "stuck_pending": stuck}


def read_log_today(path: Path, date_str: str) -> list[str]:
    """Return lines from `path` whose timestamp prefix matches `date_str`."""
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    # Both logs use ISO-ish prefixes — keep anything containing today's date.
    return [ln for ln in lines if date_str in ln]


def doc_check_summary(log_lines: list[str], date_str: str) -> dict:
    """Parse the most recent daily_doc_check.sh run from doc_check.log.

    Takes the FULL log (not date-filtered lines): only the start/done markers
    carry timestamps — the "changes_detected: N" and per-doc lines don't, so
    filtering by date first would drop them (that bug made the header claim
    "doc-check did not run" while the status section said it ran clean).

    Looks for the block written by the script:
        checked_at: <iso>
        changes_detected: <n>
          <doc_id> <status> hash=<h>
    """
    # Find the LAST "daily_doc_check start" marker with today's date
    starts = [i for i, ln in enumerate(log_lines)
              if "daily_doc_check start" in ln and date_str in ln]
    if not starts:
        return {"ran": False}

    section = log_lines[starts[-1]:]
    # Bound the section at the next run's start marker, if any
    for j, ln in enumerate(section[1:], start=1):
        if "daily_doc_check start" in ln:
            section = section[:j]
            break
    checked_at = None
    changes_detected = None
    docs = []
    for ln in section:
        m = re.match(r"checked_at:\s*(\S+)", ln)
        if m:
            checked_at = m.group(1)
            continue
        m = re.match(r"changes_detected:\s*(\d+)", ln)
        if m:
            changes_detected = int(m.group(1))
            continue
        # Doc status lines: leading whitespace, then "id  status  hash=..."
        m = re.match(r"\s{2,}(\S+)\s+(\S+)\s+hash=(\S+)", ln)
        if m:
            docs.append({"id": m.group(1), "status": m.group(2), "hash": m.group(3)})

    return {
        "ran": True,
        "checked_at": checked_at,
        "changes_detected": changes_detected,
        "docs": docs,
        "errors": [ln for ln in section if "ERROR" in ln or "Traceback" in ln],
    }


def diff_parse_failure(pending: dict | None) -> str | None:
    """Detail string when today's review file records a diff parse failure,
    else None.

    A failed parse degrades into a review file whose needs_human_review holds
    a marker leading "Diff response unparseable ..." (and whose summary starts
    "Could not parse diff response" — the only signal in pre-2026-08-06 files,
    e.g. 2026-08-02's, where the marker was empty and sanitized away). Such a
    run extracts ZERO proposals, so it must never render as clean (2026-08-08
    decision); until this check, it did — the degrade path writes a review
    file and logs no ERROR, which is exactly the "ran clean, review file
    written" signature. Detection reads the review file rather than cron.log
    so the signal survives log rotation and catches failed manual runs too.
    """
    if not pending or "_parse_error" in pending:
        return None
    import compliance_auto_refresh as car
    for it in pending.get("needs_human_review") or []:
        txt = (it.get("description") or "") if isinstance(it, dict) else str(it)
        if txt.startswith(car._DIFF_FAILURE_MARKER_PREFIX):
            return txt
    summary = str(pending.get("summary") or "")
    if summary.startswith("Could not parse diff response"):
        return summary
    return None


def auto_refresh_summary(log_lines: list[str]) -> dict:
    """Parse the most recent compliance_auto_refresh.py run from cron.log."""
    starts = [i for i, ln in enumerate(log_lines) if "Auto-Refresh starting" in ln]
    if not starts:
        return {"ran": False}

    section = log_lines[starts[-1]:]
    errors = [ln for ln in section if "ERROR" in ln]
    rate_limited = sum(1 for ln in section if "Rate limited" in ln)
    review_file_written = any("Review file written" in ln for ln in section)
    # The weekly rollup must state a REAL cron timestamp, never "at unknown",
    # so lift the one the log already carries: car.log() writes
    # "[<iso>] PKI Compliance Auto-Refresh starting - <date>".
    m = re.match(r"\[([^\]]+)\]", log_lines[starts[-1]])
    started_at = m.group(1) if m else None
    # A gate skip is a deliberate no-op (no review file expected), not a failure.
    skip_line = next((ln for ln in section if "Research gate: SKIP" in ln), None)
    skip_reason = ""
    if skip_line and "SKIP — " in skip_line:
        skip_reason = skip_line.split("SKIP — ", 1)[1].strip()
    return {
        "ran": True,
        "started_at": started_at,
        "errors": errors,
        "rate_limited_count": rate_limited,
        "review_file_written": review_file_written,
        "skipped": skip_line is not None,
        "skip_reason": skip_reason,
        # started but neither wrote a review file nor skipped -> mid-run
        "in_progress": not review_file_written and skip_line is None and not errors,
    }


DRAFTS_DIR = Path(os.environ.get("PKI_REPO_PATH",
                                 "/opt/mcp-servers/pki-compliance-mcp")) / "content_drafts"


def read_content_drafts(date_str: str) -> list[dict]:
    """Draft packages content_drafts.py generated today for urgent items.

    Each package dir holds blog.md / linkedin.md / youtube.md / tweet.md +
    meta.json. The email inlines the two short pieces and points at the repo
    for the rest. Drafts only — nothing here has been published.
    """
    out = []
    if not DRAFTS_DIR.exists():
        return out
    for d in sorted(DRAFTS_DIR.glob(f"{date_str}-*")):
        if not d.is_dir():
            continue
        pkg = {"name": d.name}
        for piece in ("tweet", "linkedin"):
            p = d / f"{piece}.md"
            pkg[piece] = p.read_text(errors="replace").strip() if p.exists() else ""
        try:
            meta = json.loads((d / "meta.json").read_text())
            pkg["title"] = (meta.get("source_item", {}).get("title")
                            or meta.get("source_item", {}).get("id") or d.name)
        except Exception:
            pkg["title"] = d.name
        out.append(pkg)
    return out


def read_approval(date_str: str) -> dict | None:
    """Read auto_approve.py output: approval log + review queue.

    auto_approve runs chained after the 10:00 UTC research cron in the same
    crontab line, so it has no fixed clock time — the fixed 10:15 slot was
    removed 2026-07-09 because it raced the 20-30 min research run.
    """
    log_p = DATA_DIR / f"approval_log_{date_str}.json"
    queue_p = DATA_DIR / f"review_queue_{date_str}.json"
    if not log_p.exists() and not queue_p.exists():
        return None
    out = {"applied": [], "blocked": None, "queue": []}
    try:
        if log_p.exists():
            data = json.loads(log_p.read_text())
            out["applied"] = data.get("applied", [])
            out["blocked"] = data.get("blocked")
        if queue_p.exists():
            out["queue"] = json.loads(queue_p.read_text()).get("items", [])
    except Exception as e:
        out["_parse_error"] = str(e)
    return out


def read_urgent_notices(date_str: str) -> dict:
    """Urgent items content_drafts.py notified about today, if any.

    Since 2026-08-20 content_drafts.py is NOTIFY-FIRST: an urgent item
    produces a notification (what it is, why it was flagged, source URLs) and
    NO drafted content. Drafting is an explicit, channel-scoped trigger. This
    reads the notice file that mode writes so the email can carry the
    notification and the exact command that turns it into copy.

    Degrades to empty on anything unreadable — a missing notice file is the
    normal no-urgent-items case, not a fault.
    """
    p = DATA_DIR / f"urgent_notice_{date_str}.json"
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {"items": [], "new": []}
    items = [i for i in (data.get("items") or []) if isinstance(i, dict)]
    return {"items": items, "new": [i for i in items if i.get("is_new")]}


# ---------------------------------------------------------------------------
# Event-driven send gate + run ledger (2026-08-20)
# ---------------------------------------------------------------------------

RUNS_FILE = DATA_DIR / "email_runs.json"
RUNS_KEEP_DAYS = 120
# The rolling backlog window outstanding_review_items() uses. The rollup ages
# items out against the same number, so an item's age-out date in the rollup is
# the day it actually falls out of the daily email's backlog list.
BACKLOG_WINDOW_DAYS = 14


def backlog_signature(outstanding: list[dict] | None) -> str:
    """Stable hash of the backlog's COMPOSITION, not its count.

    The spec is explicit that a swap at equal count must still send, so the
    signature covers each item's identity — kind, title, date, first-seen and
    urgency — and not the length of the list. Cheap by construction: the
    backlog is already aggregated on every run for the email body, so this is
    a hash over data we hold, not a second pass over the pending files.
    Returns "" for a failed aggregation, which decide_send() treats as a
    fault rather than as "unchanged".
    """
    if outstanding is None:
        return ""
    parts = sorted(
        "{kind}|{title}|{date}|{first_seen}|{urgent}".format(
            kind=o.get("kind", ""), title=o.get("title", ""),
            date=o.get("date", ""), first_seen=o.get("first_seen", ""),
            urgent=int(bool(o.get("urgent"))))
        for o in outstanding
    )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def load_runs() -> dict:
    try:
        data = json.loads(RUNS_FILE.read_text())
        if isinstance(data, dict) and isinstance(data.get("runs"), dict):
            return data
    except Exception:
        pass
    return {"runs": {}, "last_rollup": None}


def save_runs(runs: dict) -> None:
    cutoff = (datetime.now(timezone.utc).date()
              - timedelta(days=RUNS_KEEP_DAYS)).isoformat()
    runs["runs"] = {d: v for d, v in runs.get("runs", {}).items() if d >= cutoff}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs, indent=2, sort_keys=True))


def previous_backlog_signature(runs: dict, date_str: str) -> str | None:
    """Backlog signature from the most recent run strictly before date_str.

    None means "no prior run on record" — the first run after deploy, or after
    a gap. decide_send() sends in that case: an unknown baseline is not
    evidence that nothing changed.
    """
    prior = [d for d in runs.get("runs", {}) if d < date_str]
    for d in sorted(prior, reverse=True):
        sig = runs["runs"][d].get("backlog_sig")
        if sig:
            return sig
    return None


def collect_signals(pending, doc_check, auto_refresh, approval, outstanding,
                    drafts, candidates, notices, prev_backlog_sig) -> dict:
    """Everything the gate decides on, in one auditable dict.

    Split into `counts` (did something happen?) and `failures` (did something
    break?). Suppression applies to empty days only — a day the pipeline
    stumbled is never empty, however few items it produced.
    """
    proposed = flagged = 0
    if pending and "_parse_error" not in pending:
        proposed = (len(pending.get("new_deadlines", []))
                    + len(pending.get("updated_deadlines", []))
                    + len(pending.get("document_version_updates", []))
                    + len(pending.get("regulatory_updates", [])))
        flagged = len(pending.get("needs_human_review", []))

    doc_changes = doc_check.get("changes_detected") if doc_check.get("ran") else None
    backlog_sig = backlog_signature(outstanding)

    failures = []
    if not auto_refresh.get("ran"):
        failures.append("10:00 research cron did not run today")
    else:
        if auto_refresh.get("errors"):
            failures.append(f"research cron logged {len(auto_refresh['errors'])} ERROR line(s)")
        if auto_refresh.get("in_progress"):
            failures.append("research cron still running at send time")
    if diff_parse_failure(pending):
        failures.append("diff response unparseable — zero proposals extracted")
    if not doc_check.get("ran"):
        failures.append("10:30 doc-check cron did not run today")
    else:
        if doc_check.get("errors"):
            failures.append(f"doc-check logged {len(doc_check['errors'])} error line(s)")
        if doc_check.get("changes_detected") is None:
            failures.append("doc-check ran but its change count is unavailable")
    if pending and "_parse_error" in pending:
        failures.append(f"pending-updates file unparseable: {pending['_parse_error']}")
    if (pending is None and auto_refresh.get("ran")
            and not auto_refresh.get("skipped")):
        failures.append("research cron ran but wrote no pending file")
    if outstanding is None:
        failures.append("backlog aggregation failed — composition unknown")

    return {
        "counts": {
            "proposed": proposed,
            "flagged": flagged,
            "doc_changes": doc_changes or 0,
            "candidates_today": len((candidates or {}).get("today") or []),
            "candidates_stuck": len((candidates or {}).get("stuck_pending") or []),
            "drafts": len(drafts or []),
            "urgent_new": len(notices.get("new") or []),
            "backlog": len(outstanding or []),
        },
        "backlog_sig": backlog_sig,
        "prev_backlog_sig": prev_backlog_sig,
        "backlog_changed": backlog_sig != (prev_backlog_sig or ""),
        "backlog_baseline_known": prev_backlog_sig is not None,
        "failures": failures,
    }


def decide_send(signals: dict) -> tuple[bool, list[str]]:
    """(send?, reasons). No reasons -> suppress.

    The five quiet-day tests from the 2026-08-20 spec, plus two additions that
    are the same idea applied honestly:

      - content drafts / urgent notices present. An urgent item is by
        construction also a proposal or a flag, so this is belt-and-braces,
        but the flag is the one signal that must never be swallowed by a
        counting bug.
      - anything in `failures`. Silence has to mean "nothing happened", never
        "something broke quietly" — that inversion is the whole risk of going
        event-driven, and it is cheap to close here.

    Deliberately NOT a re-fire trigger: an urgent item that is merely STILL in
    the backlog, unchanged. It fired the day it appeared (composition changed);
    re-nagging it daily is exactly the ritual being cut. It reappears in the
    weekly rollup's held/awaiting-verdict section instead.

    "Urgent" here is `item["urgent"]` as produced by the diff pass and held to
    its word by car.deescalate_speculative_urgent(). There is deliberately no
    second definition in this file.
    """
    c = signals["counts"]
    reasons = []
    if c["proposed"]:
        reasons.append(f"{c['proposed']} proposed change(s)")
    if c["flagged"]:
        reasons.append(f"{c['flagged']} item(s) flagged for review")
    if c["doc_changes"]:
        reasons.append(f"{c['doc_changes']} document hash change(s)")
    if c["candidates_today"]:
        reasons.append(f"{c['candidates_today']} new content candidate(s)")
    if c["candidates_stuck"]:
        reasons.append(f"{c['candidates_stuck']} content candidate(s) stuck undelivered")
    if signals["backlog_changed"]:
        reasons.append("backlog composition changed"
                       if signals["backlog_baseline_known"]
                       else "no prior run on record — baseline unknown")
    if c["urgent_new"]:
        reasons.append(f"{c['urgent_new']} new urgent item(s) notified")
    if c["drafts"]:
        reasons.append(f"{c['drafts']} content draft package(s) written")
    reasons.extend(signals["failures"])
    return bool(reasons), reasons


def record_run(date_str: str, signals: dict, sent: bool, reasons: list[str],
               doc_check: dict, auto_refresh: dict) -> None:
    """Append this run to the ledger. Written whether or not an email went out —
    a suppressed day is exactly the day the rollup has to be able to prove ran.
    """
    runs = load_runs()
    runs.setdefault("runs", {})[date_str] = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "sent": sent,
        "reasons": reasons,
        "counts": signals["counts"],
        "backlog_sig": signals["backlog_sig"],
        "backlog_changed": signals["backlog_changed"],
        "failures": signals["failures"],
        "research": {
            "ran": bool(auto_refresh.get("ran")),
            "started_at": auto_refresh.get("started_at"),
            "skipped": bool(auto_refresh.get("skipped")),
            "skip_reason": auto_refresh.get("skip_reason", ""),
        },
        "doc_check": {
            "ran": bool(doc_check.get("ran")),
            "checked_at": doc_check.get("checked_at"),
            "changes_detected": doc_check.get("changes_detected"),
        },
    }
    save_runs(runs)


def render_html(date_str: str, pending: dict | None, doc_check: dict, auto_refresh: dict, approval: dict | None = None, outstanding: list[dict] | None = None, drafts: list[dict] | None = None, candidates: dict | None = None, notices: dict | None = None) -> str:
    """Build the HTML email body. Plain-text fallback is built separately."""
    parts = []
    parts.append(f"<h2 style='margin:0 0 12px 0;font:600 18px/1.3 -apple-system,system-ui,sans-serif'>PKI Compliance daily report — {date_str}</h2>")

    urgent_items = [o for o in (outstanding or []) if o.get("urgent")]
    if urgent_items:
        parts.append("<p style='font:600 14px/1.4 -apple-system,system-ui,sans-serif;color:#b91c1c'>🚨 "
                     f"{len(urgent_items)} urgent item(s) awaiting review:</p>")
        parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px;color:#b91c1c'>")
        for o in urgent_items:
            parts.append(f"<li><strong>{escape(o['title'])}</strong>{' — ' + escape(o['date']) if o['date'] else ''} (first seen {escape(o['first_seen'])})</li>")
        parts.append("</ul>")

    # Urgent notification — NOT drafted content (2026-08-20). What it is, why
    # it was flagged, and its sources; drafting is a separate explicit trigger
    # with an explicit channel set. On 2026-08-20 the auto-drafting shape
    # produced five channels of copy off a wrong flag, of which two channels
    # were permanently declined and the rest rewritten from scratch.
    note_items = (notices or {}).get("items") or []
    if note_items:
        parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>"
                     f"🔔 Urgent notification ({len(note_items)}) — notify-first, nothing drafted</h3>")
    for note in note_items:
        badge = ("🆕 " if note.get("is_new") else "↻ ")
        parts.append("<div style='border:1px solid #fecaca;background:#fef2f2;border-radius:8px;padding:10px 12px;margin:0 0 10px 0'>")
        parts.append(f"<p style='font:600 13px/1.4 -apple-system,system-ui,sans-serif;margin:0 0 6px 0'>{badge}{escape(str(note.get('title') or note.get('id') or '(untitled)')[:200])}</p>")
        parts.append(f"<p style='font:12px/1.5 -apple-system,system-ui,sans-serif;color:#7f1d1d;margin:0 0 6px 0'><strong>Why flagged:</strong> {escape(str(note.get('why') or '')[:600])}</p>")
        srcs = [u for u in (note.get("source_urls") or []) if isinstance(u, str)]
        if srcs:
            links = ", ".join(
                f"<a href='{escape(u, quote=True)}' style='color:#2563eb'>source {n}</a>"
                for n, u in enumerate(srcs, 1))
            parts.append(f"<p style='font:12px/1.4 -apple-system,system-ui,sans-serif;margin:0 0 6px 0'>↳ {links}</p>")
        else:
            parts.append("<p style='font:12px/1.4 -apple-system,system-ui,sans-serif;color:#7f1d1d;margin:0 0 6px 0'>↳ no source URL on this item — verify before drafting anything.</p>")
        parts.append("<p style='font:12px/1.4 -apple-system,system-ui,sans-serif;color:#666;margin:0'>No content has been drafted. To draft, pick the channels:<br>"
                     f"<code>{escape(str(note.get('draft_command') or ''))}</code></p>")
        parts.append("</div>")

    # Content drafts — written only when drafting was explicitly triggered.
    # Drafts only, nothing is published automatically.
    if drafts:
        parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>📝 Content drafts ready</h3>")
        for pkg in drafts:
            parts.append("<div style='border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;margin:0 0 10px 0'>")
            parts.append(f"<p style='font:600 13px/1.4 -apple-system,system-ui,sans-serif;margin:0 0 6px 0'>{escape(pkg['title'])}</p>")
            if pkg.get("tweet"):
                parts.append(f"<p style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0 0 6px 0'><strong>X:</strong> {escape(pkg['tweet'])}</p>")
            if pkg.get("linkedin"):
                li = pkg["linkedin"]
                li_short = li[:400] + ("…" if len(li) > 400 else "")
                parts.append(f"<p style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0 0 6px 0;white-space:pre-line'><strong>LinkedIn:</strong> {escape(li_short)}</p>")
            parts.append(f"<p style='font:12px/1.4 -apple-system,system-ui,sans-serif;color:#666;margin:0'>Blog post, YouTube package + Kit broadcast: <code>pki-compliance-mcp/content_drafts/{escape(pkg['name'])}/</code> in the repo (git pull). Drafts only — review before posting.</p>")
            parts.append("</div>")

    # Top-line counts
    proposed_count = 0
    needs_review_count = 0
    if pending and "_parse_error" not in pending:
        proposed_count = (
            len(pending.get("new_deadlines", []))
            + len(pending.get("updated_deadlines", []))
            + len(pending.get("document_version_updates", []))
            + len(pending.get("regulatory_updates", []))
        )
        needs_review_count = len(pending.get("needs_human_review", []))

    doc_changes = doc_check.get("changes_detected") if doc_check.get("ran") else None

    parts.append("<p style='font:14px/1.4 -apple-system,system-ui,sans-serif;color:#333'>")
    parts.append(f"<strong>{proposed_count}</strong> proposed change(s) from research cron, ")
    if approval is not None:
        parts.append(f"<strong>{len(approval.get('applied', []))}</strong> applied automatically, ")
        parts.append(f"<strong>{len(approval.get('queue', []))}</strong> queued for your review, ")
    parts.append(f"<strong>{needs_review_count}</strong> flagged for review, ")
    if outstanding is not None:
        parts.append(f"<strong>{len(outstanding)}</strong> outstanding across last 14 days, ")
    if not doc_check.get("ran"):
        parts.append("<strong>doc-check did not run</strong>.")
    elif doc_changes is None:
        parts.append("doc-check ran, <strong>change count unavailable</strong>.")
    else:
        parts.append(f"<strong>{doc_changes}</strong> doc URL hash change(s) detected.")
    parts.append("</p>")

    # Status block
    parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>Cron status</h3>")
    parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
    diff_fail = diff_parse_failure(pending)
    if auto_refresh.get("ran"):
        # Checked before every other outcome: a run whose diff response could
        # not be parsed extracted zero proposals and must report as a FAILURE,
        # whatever else the logs say (2026-08-08 decision).
        if diff_fail:
            parts.append("<li>10:00 UTC research cron: <strong style='color:#b91c1c'>✗ FAILED — diff response unparseable, no proposals extracted from this run's research</strong>")
            if auto_refresh.get("skipped"):
                # The scheduled run skipped, so the failed parse came from
                # another run today (e.g. a manual --force). Both facts shown.
                reason = auto_refresh.get("skip_reason") or "no tracked changes"
                parts.append(f" (scheduled run skipped: {escape(reason)}; the failed parse is from another run today)")
            parts.append(f"<br><span style='color:#64748b;font-size:12px'>{escape(diff_fail[:300])}</span>")
        elif auto_refresh.get("skipped") and not auto_refresh.get("errors"):
            reason = auto_refresh.get("skip_reason") or "no tracked changes"
            parts.append(f"<li>10:00 UTC research cron: ✓ ran — research skipped ({escape(reason)}); no review file expected")
        elif auto_refresh.get("in_progress"):
            parts.append("<li>10:00 UTC research cron: ⏳ still running at send time — today's counts are incomplete; check the dashboard later")
        else:
            ar_ok = auto_refresh.get("review_file_written") and not auto_refresh.get("errors")
            parts.append(f"<li>10:00 UTC research cron: {'✓ ran clean, review file written' if ar_ok else '⚠ ran with errors'}")
        if auto_refresh.get("errors"):
            parts.append(f" — {len(auto_refresh['errors'])} error line(s)")
        if auto_refresh.get("rate_limited_count"):
            parts.append(f", {auto_refresh['rate_limited_count']} rate-limit retry(ies)")
        parts.append("</li>")
    else:
        parts.append("<li>10:00 UTC research cron: <strong style='color:#b91c1c'>did not run today</strong></li>")

    if doc_check.get("ran"):
        dc_ok = not doc_check.get("errors")
        parts.append(f"<li>10:30 UTC doc-check cron: {'✓ ran clean' if dc_ok else '⚠ ran with errors'} at {escape(doc_check.get('checked_at') or 'unknown')}</li>")
    else:
        parts.append("<li>10:30 UTC doc-check cron: <strong style='color:#b91c1c'>did not run today</strong></li>")
    parts.append("</ul>")

    # Proposed changes
    if pending and "_parse_error" not in pending and proposed_count > 0:
        parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>Proposed changes</h3>")
        parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
        for d in pending.get("new_deadlines", []):
            parts.append(f"<li><strong>New deadline</strong>: {escape(d.get('title',''))} <em>({escape(d.get('date',''))})</em></li>")
        for d in pending.get("updated_deadlines", []):
            parts.append(f"<li><strong>Updated deadline</strong>: {escape(d.get('title', d.get('id','')))} <em>({escape(d.get('date',''))})</em></li>")
        for d in pending.get("document_version_updates", []):
            parts.append(f"<li><strong>Doc bump</strong>: {escape(d.get('id',''))} → {escape(d.get('new_version',''))} <em>({escape(d.get('new_date',''))})</em></li>")
        for d in pending.get("regulatory_updates", []):
            reg_title = d.get("title") or d.get("regulation") or d.get("id") or ""
            reg_desc = d.get("description") or d.get("update") or ""
            parts.append(f"<li><strong>Regulatory</strong>: {escape(reg_title)} — {escape(reg_desc)[:160]}</li>")
        parts.append("</ul>")
    elif pending and "_parse_error" in pending:
        parts.append(f"<p style='color:#b91c1c;font:13px/1.4 -apple-system,system-ui,sans-serif'>⚠ Could not parse pending_updates_{date_str}.json: {escape(pending['_parse_error'])}</p>")
    elif pending is None and auto_refresh.get("ran") and not auto_refresh.get("skipped"):
        parts.append(f"<p style='color:#b91c1c;font:13px/1.4 -apple-system,system-ui,sans-serif'>⚠ Research cron ran but pending_updates_{date_str}.json is missing.</p>")

    # Auto-approve outcome (chained after the 10:00 UTC research cron, no fixed slot)
    if approval is not None:
        parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>Auto-approve (after the 10:00 UTC research run)</h3>")
        if approval.get("_parse_error"):
            parts.append(f"<p style='color:#b91c1c;font:13px/1.4 -apple-system,system-ui,sans-serif'>⚠ Could not parse auto-approve output: {escape(approval['_parse_error'])}</p>")
        else:
            if approval.get("blocked"):
                parts.append(f"<p style='color:#b45309;font:13px/1.4 -apple-system,system-ui,sans-serif'>⚠ Auto-apply blocked ({escape(str(approval['blocked']))}) — everything queued for review.</p>")
            if approval.get("applied"):
                parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
                for op in approval["applied"]:
                    parts.append(f"<li>✓ Applied: {escape(op)}</li>")
                parts.append("</ul>")
            if approval.get("queue"):
                parts.append("<p style='font:13px/1.4 -apple-system,system-ui,sans-serif;margin:8px 0 4px 0'><strong>Queued for your review:</strong></p>")
                parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
                for q in approval["queue"]:
                    item = q.get("item", {})
                    label = item.get("title") or item.get("id") or str(item)[:80] if isinstance(item, dict) else str(item)[:80]
                    parts.append(f"<li><strong>[{escape(q.get('kind',''))}]</strong> {escape(label)} — <em>{escape(q.get('reason',''))}</em></li>")
                parts.append("</ul>")
            if not approval.get("applied") and not approval.get("queue"):
                parts.append("<p style='font:13px/1.4 -apple-system,system-ui,sans-serif;color:#666'>Nothing to apply or review.</p>")

    # Items flagged for review
    if pending and pending.get("needs_human_review"):
        parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>Flagged for human review</h3>")
        parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
        for item in pending["needs_human_review"]:
            srcs = []
            if isinstance(item, dict):
                iid = item.get("id") or ""
                desc = item.get("description") or item.get("item") or item.get("reason") or json.dumps(item)
                label = f"{iid}: {desc}" if iid else desc
                srcs = [u for u in (item.get("provenance_urls") or []) if isinstance(u, str)]
            else:
                label = str(item)
            src_part = ""
            if srcs:
                links = ", ".join(
                    f"<a href='{escape(u, quote=True)}' style='color:#2563eb'>source {n}</a>"
                    for n, u in enumerate(srcs, 1)
                )
                src_part = f"<br><span style='color:#64748b'>↳ {links}</span>"
            parts.append(f"<li>{escape(label[:300])}{src_part}</li>")
        parts.append("</ul>")

    # Content candidates — the fifth pipeline state (no date certain → content,
    # not a deadline). Today's classifications are shown while
    # SURFACE_CONTENT_CANDIDATES is on (the Part A safety window); rows the
    # news desk never received (sink_status pending) are shown unconditionally.
    if candidates:
        today_cc = candidates.get("today") or []
        stuck_cc = candidates.get("stuck_pending") or []
        if today_cc and SURFACE_CONTENT_CANDIDATES:
            parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>🗞 Content candidates (new state — check nothing here is a real deadline)</h3>")
            parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
            for c in today_cc:
                sink = c.get("sink_status", "?")
                sink_icon = {"posted": "✅ drafted in news desk", "pending": "⏳ delivery pending"}.get(sink, sink)
                link = ""
                if c.get("primary_url"):
                    link = f" — <a href='{escape(c['primary_url'], quote=True)}' style='color:#2563eb'>source</a>"
                parts.append(
                    f"<li><strong>{escape(str(c['title'])[:200])}</strong> "
                    f"<em>[rule: {escape(c.get('rule', '?'))}]</em> — {escape(sink_icon)}{link}"
                    + (f"<br><span style='color:#64748b;font-size:12px'>matched: {escape(str(c.get('matched'))[:120])}</span>" if c.get("matched") else "")
                    + "</li>")
            parts.append("</ul>")
            parts.append("<p style='font:12px/1.4 -apple-system,system-ui,sans-serif;color:#666;margin-top:4px'>These never enter the review queue. A real Tier 1 item in this list is a misclassification — reply/flag it. Toggle: SURFACE_CONTENT_CANDIDATES in daily_email.py.</p>")
        if stuck_cc:
            parts.append("<p style='font:600 13px/1.4 -apple-system,system-ui,sans-serif;color:#b45309'>⚠ "
                         f"{len(stuck_cc)} content candidate(s) still awaiting news-desk delivery:</p>")
            parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px;color:#b45309'>")
            for c in stuck_cc:
                parts.append(f"<li>{escape(str(c['title'])[:200])} — first seen {escape(str(c['first_seen']))}, {c.get('attempts', 0)} delivery attempt(s). Check NEWS_API_BASE_URL/NEWS_ADMIN_SECRET in the cron env.</li>")
            parts.append("</ul>")

    # Doc URL change detail (when there are any)
    if doc_check.get("ran") and doc_check.get("docs"):
        changed_or_new = [d for d in doc_check["docs"] if d["status"] in ("changed", "new")]
        if changed_or_new:
            parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>Doc URL hash changes</h3>")
            parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
            for d in changed_or_new:
                parts.append(f"<li><code>{escape(d['id'])}</code>: {escape(d['status'])}</li>")
            parts.append("</ul>")
            parts.append("<p style='font:12px/1.4 -apple-system,system-ui,sans-serif;color:#666;margin-top:4px'>Note: hashes are sanitized against dynamic page noise (fixed 2026-07-09); a flagged change is usually a real document update worth a look.</p>")

    # Rolling backlog: everything still awaiting a decision, however old the
    # email that first mentioned it. Makes missed email days cost nothing.
    if outstanding:
        parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>Outstanding review items (last 14 days)</h3>")
        parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
        for o in outstanding:
            urgent_tag = "🚨 " if o.get("urgent") else ""
            date_part = f" <em>({escape(o['date'])})</em>" if o.get("date") else ""
            srcs = "".join(
                f"<a href='{escape(u, quote=True)}' style='color:#2563eb'>source {n}</a>"
                + ("" if n == len(o["provenance_urls"]) else ", ")
                for n, u in enumerate(o.get("provenance_urls") or [], 1)
            )
            src_part = f"<br><span style='color:#64748b'>↳ {srcs}</span>" if srcs else ""
            parts.append(f"<li>{urgent_tag}<strong>[{escape(o['kind'])}]</strong> {escape(o['title'][:200])}{date_part} — first seen {escape(o['first_seen'])}{src_part}</li>")
        parts.append("</ul>")
    elif outstanding is not None:
        parts.append("<p style='font:13px/1.4 -apple-system,system-ui,sans-serif;color:#15803d'>✓ No outstanding review items from the last 14 days.</p>")

    # Footer
    dashboard_token = os.environ.get("DASHBOARD_TOKEN", "")
    dash_link = f"{DASHBOARD_URL}?token={dashboard_token}" if dashboard_token else DASHBOARD_URL
    parts.append("<hr style='border:0;border-top:1px solid #e5e7eb;margin:20px 0'>")
    parts.append("<p style='font:12px/1.4 -apple-system,system-ui,sans-serif;color:#666'>")
    parts.append(f"Dashboard: <a href='{escape(dash_link)}'>{escape(DASHBOARD_URL)}</a><br>")
    parts.append(f"Review file: <code>/root/.pki-compliance-mcp/pending_updates_{date_str}.json</code><br>")
    parts.append("Approval workflow lives in deployment memory under <em>Compliance approval workflow</em>.")
    parts.append("</p>")

    return "<div style='max-width:640px'>" + "".join(parts) + "</div>"


def render_text(date_str: str, pending: dict | None, doc_check: dict, auto_refresh: dict, approval: dict | None = None, outstanding: list[dict] | None = None, drafts: list[dict] | None = None, candidates: dict | None = None, notices: dict | None = None) -> str:
    lines = [f"PKI Compliance daily report — {date_str}", "=" * 60, ""]

    note_items = (notices or {}).get("items") or []
    if note_items:
        lines.append(f"URGENT NOTIFICATION ({len(note_items)}) — notify-first, nothing drafted:")
        for note in note_items:
            tag = "NEW" if note.get("is_new") else "still open"
            lines.append(f"  - [{tag}] {str(note.get('title') or note.get('id') or '(untitled)')[:140]}")
            lines.append(f"      why: {str(note.get('why') or '')[:300]}")
            for u in (note.get("source_urls") or []):
                lines.append(f"      source: {u}")
            lines.append(f"      to draft: {note.get('draft_command') or ''}")
        lines.append("")

    if candidates:
        today_cc = candidates.get("today") or []
        stuck_cc = candidates.get("stuck_pending") or []
        if today_cc and SURFACE_CONTENT_CANDIDATES:
            lines.append(f"CONTENT CANDIDATES ({len(today_cc)}) — new state, check nothing is a real deadline:")
            for c in today_cc:
                lines.append(f"  - [{c.get('rule', '?')}] {str(c['title'])[:120]} ({c.get('sink_status', '?')})")
            lines.append("")
        if stuck_cc:
            lines.append(f"WARNING: {len(stuck_cc)} content candidate(s) awaiting news-desk delivery:")
            for c in stuck_cc:
                lines.append(f"  - {str(c['title'])[:120]} (first seen {c.get('first_seen')}, {c.get('attempts', 0)} attempts)")
            lines.append("")

    if drafts:
        lines.append(f"CONTENT DRAFTS READY ({len(drafts)}):")
        for pkg in drafts:
            lines.append(f"  - {pkg['title']}")
            if pkg.get("tweet"):
                lines.append(f"    X: {pkg['tweet']}")
            lines.append(f"    Full package: pki-compliance-mcp/content_drafts/{pkg['name']}/ (drafts only)")
        lines.append("")

    proposed = 0
    if pending and "_parse_error" not in pending:
        proposed = (
            len(pending.get("new_deadlines", []))
            + len(pending.get("updated_deadlines", []))
            + len(pending.get("document_version_updates", []))
            + len(pending.get("regulatory_updates", []))
        )

    lines.append(f"Proposed changes: {proposed}")
    if approval is not None:
        lines.append(f"Applied automatically: {len(approval.get('applied', []))}")
        lines.append(f"Queued for review: {len(approval.get('queue', []))}")
        if approval.get("blocked"):
            lines.append(f"Auto-apply BLOCKED: {approval['blocked']}")
    lines.append(f"Flagged for review: {len(pending.get('needs_human_review', [])) if pending else 0}")
    lines.append(f"Doc URL hash changes: {doc_check.get('changes_detected', 'n/a')}")
    if outstanding is not None:
        lines.append(f"Outstanding review items (last 14 days): {len(outstanding)}")
        for o in outstanding:
            tag = "URGENT " if o.get("urgent") else ""
            lines.append(f"  - {tag}[{o['kind']}] {o['title'][:120]} (first seen {o['first_seen']})")
            for u in (o.get("provenance_urls") or []):
                lines.append(f"      source: {u}")
    lines.append("")
    diff_fail = diff_parse_failure(pending)
    if diff_fail:
        lines.append("10:00 UTC research cron: FAILED — diff response unparseable, "
                     "no proposals extracted from this run's research")
        lines.append(f"  {diff_fail[:200]}")
    else:
        lines.append(f"10:00 UTC research cron: {'ran' if auto_refresh.get('ran') else 'DID NOT RUN'}")
    lines.append(f"10:30 UTC doc-check cron: {'ran' if doc_check.get('ran') else 'DID NOT RUN'}")
    lines.append("")
    lines.append(f"Dashboard: {DASHBOARD_URL}")
    lines.append(f"Review file: /root/.pki-compliance-mcp/pending_updates_{date_str}.json")
    return "\n".join(lines)


def gather(date_str: str) -> dict:
    """Read every input the email and the gate need. No side effects."""
    pending = read_pending_updates(date_str)
    cron_log = read_log_today(DATA_DIR / "cron.log", date_str)
    # Doc log must NOT be date-filtered: its data lines lack timestamps.
    doc_log_path = DATA_DIR / "doc_check.log"
    doc_log = doc_log_path.read_text(errors="replace").splitlines() if doc_log_path.exists() else []

    auto_refresh = auto_refresh_summary(cron_log)
    doc_check = doc_check_summary(doc_log, date_str)
    approval = read_approval(date_str)

    proposed = 0
    if pending and "_parse_error" not in pending:
        proposed = (
            len(pending.get("new_deadlines", []))
            + len(pending.get("updated_deadlines", []))
            + len(pending.get("document_version_updates", []))
            + len(pending.get("regulatory_updates", []))
        )
    try:
        outstanding = outstanding_review_items(date_str)
    except Exception as e:
        print(f"WARNING: outstanding-items aggregation failed: {e}", file=sys.stderr)
        outstanding = None

    try:
        drafts = read_content_drafts(date_str)
    except Exception as e:
        print(f"WARNING: content-drafts read failed: {e}", file=sys.stderr)
        drafts = []

    try:
        candidates = read_content_candidates(date_str, pending)
    except Exception as e:
        print(f"WARNING: content-candidates read failed: {e}", file=sys.stderr)
        candidates = None

    notices = read_urgent_notices(date_str)

    return {"pending": pending, "doc_check": doc_check,
            "auto_refresh": auto_refresh, "approval": approval,
            "outstanding": outstanding, "drafts": drafts,
            "candidates": candidates, "notices": notices,
            "proposed": proposed}


def build_subject(date_str: str, g: dict) -> str:
    urgent_count = sum(1 for o in (g["outstanding"] or []) if o.get("urgent"))
    cc_today = len((g["candidates"] or {}).get("today") or [])
    subject = f"[PKI Compliance] {date_str} — {g['proposed']} proposed change(s)"
    if g["outstanding"] is not None:
        subject += f", {len(g['outstanding'])} outstanding"
    if g["drafts"]:
        subject += f", {len(g['drafts'])} content draft(s)"
    if cc_today and SURFACE_CONTENT_CANDIDATES:
        subject += f", {cc_today} content candidate(s)"
    if urgent_count:
        subject = f"🚨 URGENT ({urgent_count}) — " + subject
    return subject


def send_email(api_key: str, from_addr: str, to_addr: str, subject: str,
               html: str, text: str) -> int:
    try:
        r = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": from_addr,
                "to": to_addr,
                "subject": subject,
                "html": html,
                "text": text,
            },
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        print(f"sent: id={body.get('id')} subject={subject!r}")
        return 0
    except httpx.HTTPStatusError as e:
        print(f"ERROR: Resend rejected ({e.response.status_code}): {e.response.text[:400]}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="PKI Compliance daily email")
    parser.add_argument("--force", action="store_true",
                        help="Send even when the gate would suppress")
    parser.add_argument("--dry-run", action="store_true",
                        help="Decide and render, but do not send and do not "
                             "write the run ledger")
    parser.add_argument("--explain", action="store_true",
                        help="Print the gate decision and exit; no send, no ledger")
    parser.add_argument("--date", default=None,
                        help="Report on an alternate date (replay/fixtures)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip the end-of-chain poll (replay/testing)")
    args = parser.parse_args()

    date_str = args.date or today_iso()

    if not (args.explain or args.dry_run or args.no_wait):
        wait_for_morning_chain(date_str)

    g = gather(date_str)
    runs = load_runs()
    signals = collect_signals(
        g["pending"], g["doc_check"], g["auto_refresh"], g["approval"],
        g["outstanding"], g["drafts"], g["candidates"], g["notices"],
        previous_backlog_signature(runs, date_str))
    should_send, reasons = decide_send(signals)

    if args.explain:
        print(json.dumps({"date": date_str, "would_send": should_send,
                          "reasons": reasons, "signals": signals}, indent=2))
        return 0

    if not should_send and not args.force:
        # The suppressed path still records itself. This line and the ledger
        # entry are the only evidence the day was checked at all, and the
        # weekly rollup reads the ledger to say so out loud.
        print(f"suppressed: {date_str} — nothing happened "
              f"(0 proposed / 0 flagged / 0 doc changes / backlog unchanged / "
              f"0 new content candidates); no email sent")
        if not args.dry_run:
            record_run(date_str, signals, sent=False, reasons=[],
                       doc_check=g["doc_check"], auto_refresh=g["auto_refresh"])
        return 0

    if args.force and not should_send:
        reasons = ["--force (gate would have suppressed)"]
    print(f"sending: {date_str} — " + "; ".join(reasons))

    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        print("ERROR: RESEND_API_KEY not set", file=sys.stderr)
        return 1
    to_addr = os.environ.get("PKI_EMAIL_TO", DEFAULT_TO)
    from_addr = os.environ.get("PKI_EMAIL_FROM", DEFAULT_FROM)

    subject = build_subject(date_str, g)
    html = render_html(date_str, g["pending"], g["doc_check"], g["auto_refresh"],
                       g["approval"], g["outstanding"], g["drafts"],
                       g["candidates"], g["notices"])
    text = render_text(date_str, g["pending"], g["doc_check"], g["auto_refresh"],
                       g["approval"], g["outstanding"], g["drafts"],
                       g["candidates"], g["notices"])

    if args.dry_run:
        print(f"DRY RUN — would send {subject!r} ({len(html)} bytes HTML)")
        return 0

    rc = send_email(api_key, from_addr, to_addr, subject, html, text)
    record_run(date_str, signals, sent=(rc == 0), reasons=reasons,
               doc_check=g["doc_check"], auto_refresh=g["auto_refresh"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
