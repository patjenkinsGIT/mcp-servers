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


def read_log_today(path: Path, date_str: str) -> list[str]:
    """Return lines from `path` whose timestamp prefix matches `date_str`."""
    if not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    # Both logs use ISO-ish prefixes — keep anything containing today's date.
    return [ln for ln in lines if date_str in ln]


def doc_check_summary(log_lines: list[str]) -> dict:
    """Parse the most recent daily_doc_check.sh run from doc_check.log.

    Looks for the trailing block written by the script:
        checked_at: <iso>
        changes_detected: <n>
          <doc_id> <status> hash=<h>
    """
    # Find the LAST "daily_doc_check start" marker today (if multiple runs)
    starts = [i for i, ln in enumerate(log_lines) if "daily_doc_check start" in ln]
    if not starts:
        return {"ran": False}

    section = log_lines[starts[-1]:]
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
    return {
        "ran": True,
        "errors": errors,
        "rate_limited_count": rate_limited,
        "review_file_written": review_file_written,
    }


def render_html(date_str: str, pending: dict | None, doc_check: dict, auto_refresh: dict) -> str:
    """Build the HTML email body. Plain-text fallback is built separately."""
    parts = []
    parts.append(f"<h2 style='margin:0 0 12px 0;font:600 18px/1.3 -apple-system,system-ui,sans-serif'>PKI Compliance daily report — {date_str}</h2>")

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
    parts.append(f"<strong>{needs_review_count}</strong> flagged for review, ")
    if doc_changes is None:
        parts.append("<strong>doc-check did not run</strong>.")
    else:
        parts.append(f"<strong>{doc_changes}</strong> doc URL hash change(s) detected.")
    parts.append("</p>")

    # Status block
    parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>Cron status</h3>")
    parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
    if auto_refresh.get("ran"):
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
            parts.append(f"<li><strong>Regulatory</strong>: {escape(d.get('regulation',''))} — {escape(d.get('update',''))[:120]}</li>")
        parts.append("</ul>")
    elif pending and "_parse_error" in pending:
        parts.append(f"<p style='color:#b91c1c;font:13px/1.4 -apple-system,system-ui,sans-serif'>⚠ Could not parse pending_updates_{date_str}.json: {escape(pending['_parse_error'])}</p>")
    elif pending is None and auto_refresh.get("ran"):
        parts.append(f"<p style='color:#b91c1c;font:13px/1.4 -apple-system,system-ui,sans-serif'>⚠ Research cron ran but pending_updates_{date_str}.json is missing.</p>")

    # Items flagged for review
    if pending and pending.get("needs_human_review"):
        parts.append("<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 -apple-system,system-ui,sans-serif'>Flagged for human review</h3>")
        parts.append("<ul style='font:13px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding-left:20px'>")
        for item in pending["needs_human_review"]:
            label = item.get("item") or item.get("reason") or json.dumps(item)
            parts.append(f"<li>{escape(label)}</li>")
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
            parts.append("<p style='font:12px/1.4 -apple-system,system-ui,sans-serif;color:#666;margin-top:4px'>Note: 3 URLs (microsoft_root_program, nist_800_131a, nist_800_57) are known-noisy due to dynamic page content; their flips are usually not meaningful.</p>")

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


def render_text(date_str: str, pending: dict | None, doc_check: dict, auto_refresh: dict) -> str:
    lines = [f"PKI Compliance daily report — {date_str}", "=" * 60, ""]

    proposed = 0
    if pending and "_parse_error" not in pending:
        proposed = (
            len(pending.get("new_deadlines", []))
            + len(pending.get("updated_deadlines", []))
            + len(pending.get("document_version_updates", []))
            + len(pending.get("regulatory_updates", []))
        )

    lines.append(f"Proposed changes: {proposed}")
    lines.append(f"Flagged for review: {len(pending.get('needs_human_review', [])) if pending else 0}")
    lines.append(f"Doc URL hash changes: {doc_check.get('changes_detected', 'n/a')}")
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

    pending = read_pending_updates(date_str)
    cron_log = read_log_today(DATA_DIR / "cron.log", date_str)
    doc_log = read_log_today(DATA_DIR / "doc_check.log", date_str)

    auto_refresh = auto_refresh_summary(cron_log)
    doc_check = doc_check_summary(doc_log)

    proposed = 0
    if pending and "_parse_error" not in pending:
        proposed = (
            len(pending.get("new_deadlines", []))
            + len(pending.get("updated_deadlines", []))
            + len(pending.get("document_version_updates", []))
            + len(pending.get("regulatory_updates", []))
        )
    subject = f"[PKI Compliance] {date_str} — {proposed} proposed change(s)"

    html = render_html(date_str, pending, doc_check, auto_refresh)
    text = render_text(date_str, pending, doc_check, auto_refresh)

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
