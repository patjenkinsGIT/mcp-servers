#!/usr/bin/env python3
"""
Daily PKI Compliance summary email.

Composes a concise HTML summary of the two morning crons and sends it via
Resend. Run from host crontab at 10:35 UTC (5 min after daily_doc_check.sh).

Inputs:
  - /root/.pki-compliance-mcp/pending_updates_YYYY-MM-DD.json (from 10:00 cron)
  - /root/.pki-compliance-mcp/doc_check.log                   (from 10:30 cron)
  - /root/.pki-compliance-mcp/cron.log                        (10:00 cron logs)

Env required:
  RESEND_API_KEY=re_...

Env optional:
  PKI_EMAIL_TO=patrick@fixmycert.com         (default if unset)
  PKI_EMAIL_FROM=noreply@mail.fixmycert.com  (default if unset)

Exit codes: 0 ok, 1 API key missing, 2 send failed.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import httpx

DATA_DIR = Path("/root/.pki-compliance-mcp")
DASHBOARD_URL = "https://compliance-api.fixmycert.com/dashboard"
DEFAULT_TO = "patrick@fixmycert.com"
DEFAULT_FROM = "noreply@mail.fixmycert.com"


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


def auto_refresh_summary(log_lines: list[str]) -> dict:
    """Parse the most recent compliance_auto_refresh.py run from cron.log."""
    starts = [i for i, ln in enumerate(log_lines) if "Auto-Refresh starting" in ln]
    if not starts:
        return {"ran": False}

    section = log_lines[starts[-1]:]
    errors = [ln for ln in section if "ERROR" in ln]
    rate_limited = sum(1 for ln in section if "Rate limited" in ln)
    review_file_written = any("Review file written" in ln for ln in section)
    # A gate skip is a deliberate no-op (no review file expected), not a failure.
    skip_line = next((ln for ln in section if "Research gate: SKIP" in ln), None)
    skip_reason = ""
    if skip_line and "SKIP — " in skip_line:
        skip_reason = skip_line.split("SKIP — ", 1)[1].strip()
    return {
        "ran": True,
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


def render_html(date_str: str, pending: dict | None, doc_check: dict, auto_refresh: dict, approval: dict | None = None, outstanding: list[dict] | None = None, drafts: list[dict] | None = None) -> str:
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

    # Content drafts generated for today's urgent items — drafts only,
    # nothing is published automatically.
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
    if auto_refresh.get("ran"):
        if auto_refresh.get("skipped") and not auto_refresh.get("errors"):
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


def render_text(date_str: str, pending: dict | None, doc_check: dict, auto_refresh: dict, approval: dict | None = None, outstanding: list[dict] | None = None, drafts: list[dict] | None = None) -> str:
    lines = [f"PKI Compliance daily report — {date_str}", "=" * 60, ""]

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
    lines.append(f"10:00 UTC research cron: {'ran' if auto_refresh.get('ran') else 'DID NOT RUN'}")
    lines.append(f"10:30 UTC doc-check cron: {'ran' if doc_check.get('ran') else 'DID NOT RUN'}")
    lines.append("")
    lines.append(f"Dashboard: {DASHBOARD_URL}")
    lines.append(f"Review file: /root/.pki-compliance-mcp/pending_updates_{date_str}.json")
    return "\n".join(lines)


def main() -> int:
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        print("ERROR: RESEND_API_KEY not set", file=sys.stderr)
        return 1

    to_addr = os.environ.get("PKI_EMAIL_TO", DEFAULT_TO)
    from_addr = os.environ.get("PKI_EMAIL_FROM", DEFAULT_FROM)
    date_str = today_iso()

    wait_for_morning_chain(date_str)
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

    urgent_count = sum(1 for o in (outstanding or []) if o.get("urgent"))
    subject = f"[PKI Compliance] {date_str} — {proposed} proposed change(s)"
    if outstanding is not None:
        subject += f", {len(outstanding)} outstanding"
    if drafts:
        subject += f", {len(drafts)} content draft(s)"
    if urgent_count:
        subject = f"🚨 URGENT ({urgent_count}) — " + subject

    html = render_html(date_str, pending, doc_check, auto_refresh, approval, outstanding, drafts)
    text = render_text(date_str, pending, doc_check, auto_refresh, approval, outstanding, drafts)

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


if __name__ == "__main__":
    sys.exit(main())
