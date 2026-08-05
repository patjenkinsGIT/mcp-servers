#!/usr/bin/env python3
"""Weekly FixMyCert news-desk gap check — Hub deadlines with no first-party
coverage, plus feed health. Emails the report via Resend.

READ-ONLY: GETs three endpoints and nothing else. No writes, no git, no MCP
port. Replaces the claude.ai cloud routine trig_01KtjShXBbnXayCAKLgC1LJo
(Mondays 15:00 UTC), which could not run at all — the cloud sandbox proxy
denies CONNECT to fixmycert hosts (403, seen 2026-07-23).

Coverage is judged against the ADMIN feed, so results match the news MCP's
`news_get_uncovered` exactly. That matters: 17 first-party items are archived,
10 of them deliberately (the 2026-07-22 editorial pre-screen dropped general
cyber-regulation items to keep /news PKI-core). Archived means "decided", not
"missing" — judging against the public feed instead re-reports those 10 as gaps
every single week. The matching logic below is a copy of server.py's
`_compute_uncovered` and must be kept in step with it.

Items covering a deadline only as an unpublished draft get their own section:
covered for bookkeeping, not yet visible to readers.

Host crontab (root), Mondays 15:00 UTC — needs the admin secret for the read:

    0 15 * * 1 cd /opt/mcp-servers/fixmycert-news-mcp && set -a && . /opt/mcp-servers/.env && set +a && /usr/bin/python3 news_gap_check.py >> /root/.pki-compliance-mcp/news_gap_check.log 2>&1

Env (all already in /opt/mcp-servers/.env):
  NEWS_API_BASE_URL   required, e.g. https://fixmycert.com
  NEWS_ADMIN_SECRET   required — read-only use here (GET /api/news/admin)
  RESEND_API_KEY      required unless --stdout/--json
  NEWS_EMAIL_TO       default patrick@fixmycert.com
  NEWS_EMAIL_FROM     default noreply@mail.fixmycert.com

Flags:
  --stdout   print the text report, send no email (for testing)
  --json     print machine-readable results, send no email
"""

import argparse
import difflib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from html import escape

import httpx

COMPLIANCE_DATA_URL = "https://compliance-api.fixmycert.com/api/compliance-data"
DEFAULT_TO = "patrick@fixmycert.com"
DEFAULT_FROM = "noreply@mail.fixmycert.com"

SINCE_DAYS = 60          # past window; all future deadlines always considered
URGENT_DAYS = 60         # uncovered and due within this many days = urgent
STALL_HOURS = 48         # aggregator updatedAt older than this = stalled
PAGE_SIZE = 500


class FetchError(RuntimeError):
    """An upstream API could not be read. Never downgrade this to a partial
    answer: a half-read feed produces phantom gaps."""


def _redact(s: str) -> str:
    return re.sub(r"(key=)[^&\s]+", r"\1***", s)


# ------------------------------------------------------------------
# Fetch
# ------------------------------------------------------------------

def fetch_deadlines(client: httpx.Client) -> list[dict]:
    try:
        r = client.get(COMPLIANCE_DATA_URL)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise FetchError(f"compliance-api.fixmycert.com unreachable: {_redact(str(e))}") from e
    if not isinstance(data, dict) or "deadlines" not in data:
        raise FetchError("compliance-api returned no 'deadlines' key")
    return data["deadlines"]


def fetch_first_party(client: httpx.Client, base_url: str, secret: str) -> list[dict]:
    """All first-party items from /api/news/admin — drafts and archived included."""
    items: list[dict] = []
    offset = 0
    for _ in range(20):
        try:
            r = client.get(
                f"{base_url}/api/news/admin",
                params={"isFirstParty": "true", "limit": PAGE_SIZE, "offset": offset},
                headers={"Authorization": f"Bearer {secret}"},
            )
            if r.status_code == 401:
                raise FetchError("401 from /api/news/admin — NEWS_ADMIN_SECRET does not match "
                                 "the FixMyCert Replit secret")
            r.raise_for_status()
            data = r.json()
        except FetchError:
            raise
        except Exception as e:
            raise FetchError(f"/api/news/admin unreachable at offset {offset}: {_redact(str(e))}") from e
        if isinstance(data, list):
            page, has_more = data, False
        else:
            page = data.get("items", [])
            has_more = bool(data.get("hasMore")) and bool(page)
        items.extend(page)
        if not has_more:
            break
        offset += len(page)
    return [i for i in items if i.get("isFirstParty")]


def fetch_public_health(client: httpx.Client, base_url: str) -> tuple[int, str | None]:
    """Reader-visible feed size and the aggregator's updatedAt stamp."""
    try:
        r = client.get(f"{base_url}/api/news", params={"limit": 1})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise FetchError(f"public /api/news unreachable: {_redact(str(e))}") from e
    if not isinstance(data, dict):
        raise FetchError("public /api/news returned an unexpected shape")
    return int(data.get("total") or 0), data.get("updatedAt")


# ------------------------------------------------------------------
# Matching — keep in step with server.py::_compute_uncovered
# ------------------------------------------------------------------

def parse_dt(value: str) -> datetime:
    v = (value or "").strip().replace("Z", "+00:00")
    for fmt in None, "%Y-%m-%d":
        try:
            dt = datetime.fromisoformat(v) if fmt is None else datetime.strptime(v, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    raise ValueError(f"Invalid date: {value!r}")


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).strip()


def titles_match(deadline_title: str, item_title: str) -> bool:
    a, b = norm_title(deadline_title), norm_title(item_title)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


def classify(deadlines: list[dict], first_party: list[dict], since_days: int) -> tuple[list[dict], list[tuple[dict, dict]]]:
    """Returns (uncovered deadlines, [(deadline, draft item)] covered only by a draft)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    uncovered: list[dict] = []
    draft_only: list[tuple[dict, dict]] = []
    for d in deadlines:
        try:
            d_date = parse_dt(d.get("date", ""))
        except ValueError:
            continue
        if d_date < cutoff:
            continue
        matches = [i for i in first_party if d.get("id") and i.get("deadlineId") == d.get("id")]
        if not matches:
            matches = [i for i in first_party if titles_match(d.get("title", ""), i.get("title", ""))]
        if not matches:
            uncovered.append(d)
        elif all(i.get("status") == "draft" for i in matches):
            draft_only.append((d, matches[0]))
    uncovered.sort(key=lambda d: d.get("date", ""))
    draft_only.sort(key=lambda p: p[0].get("date", ""))
    return uncovered, draft_only


def is_urgent(deadline: dict) -> bool:
    try:
        return parse_dt(deadline["date"]) <= datetime.now(timezone.utc) + timedelta(days=URGENT_DAYS)
    except (ValueError, KeyError):
        return False


def feed_stalled(updated_at: str | None) -> tuple[bool, str]:
    if not updated_at:
        return True, "feed reported no updatedAt"
    try:
        age = datetime.now(timezone.utc) - parse_dt(updated_at)
    except ValueError:
        return True, f"unparseable updatedAt {updated_at!r}"
    return age.total_seconds() / 3600 > STALL_HOURS, f"{age.total_seconds() / 3600:.0f}h old"


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def _fw(d: dict) -> str:
    return str(d.get("framework_name") or d.get("framework_id") or d.get("category") or "")


def build_report(deadlines, first_party, uncovered, draft_only, public_total, updated_at):
    """Returns (subject, text, html)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    urgent = [d for d in uncovered if is_urgent(d)]
    stalled, age_note = feed_stalled(updated_at)
    published = [i for i in first_party if i.get("status") == "published"]

    subject = f"[FixMyCert news] {today} — {len(uncovered)} uncovered deadline(s)"
    if urgent:
        subject = f"🚨 URGENT ({len(urgent)}) — " + subject

    lines = [f"FixMyCert news-desk gap check — {today}", ""]
    if not uncovered:
        lines.append(f"✅ 0 uncovered. All {len(deadlines)} Hub deadlines in the window "
                     f"(last {SINCE_DAYS} days + all future) have first-party coverage.")
    else:
        lines.append(f"🕳️  {len(uncovered)} Hub deadline(s) with no first-party coverage:")
        lines.append("")
        for d in uncovered:
            flag = "  ⚠️ WITHIN 60 DAYS" if is_urgent(d) else ""
            lines.append(f"- {d.get('date')} [{d.get('id')}] {d.get('title')} ({_fw(d)}){flag}")
        lines.append("")
        lines.append("Close a gap in Claude Code: news_add with that deadlineId, then news_publish.")

    if draft_only:
        lines += ["", f"📝 {len(draft_only)} covered only by an unpublished draft "
                      f"(written, not reader-visible — news_publish to ship):"]
        for d, item in draft_only:
            lines.append(f"- {d.get('date')} [{d.get('id')}] {d.get('title')} "
                         f"→ draft {item.get('id')}")

    lines += [
        "",
        "Feed health",
        f"- Hub deadlines: {len(deadlines)}",
        f"- First-party items: {len(first_party)} ({len(published)} published)",
        f"- Public feed items: {public_total}",
        f"- Aggregator updatedAt: {updated_at or '(absent)'} — {age_note}",
    ]
    if stalled:
        lines.append(f"- ⚠️ RSS aggregator may be stalled (>{STALL_HOURS}h without an update).")
    lines += ["", "Read-only check. Source: fixmycert-news-mcp/news_gap_check.py "
                  "(droplet cron, Mondays 15:00 UTC). Coverage counts archived items as decided, "
                  "matching news_get_uncovered."]
    text = "\n".join(lines)

    rows = "".join(
        f"<li><strong>{escape(str(d.get('date')))}</strong> — {escape(str(d.get('title')))}"
        f"<br><code>{escape(str(d.get('id')))}</code> · {escape(_fw(d))}"
        + (' · <strong style="color:#b91c1c">within 60 days</strong>' if is_urgent(d) else "")
        + "</li>"
        for d in uncovered
    )
    body = (f"<p>✅ <strong>0 uncovered.</strong> All {len(deadlines)} Hub deadlines in the window "
            f"(last {SINCE_DAYS} days + all future) have first-party coverage.</p>"
            if not uncovered else
            f"<p>🕳️ <strong>{len(uncovered)}</strong> Hub deadline(s) with no first-party coverage:</p>"
            f"<ul>{rows}</ul><p>Close a gap in Claude Code: <code>news_add</code> with that "
            f"deadlineId, then <code>news_publish</code>.</p>")
    if draft_only:
        drafts_html = "".join(
            f"<li><strong>{escape(str(d.get('date')))}</strong> — {escape(str(d.get('title')))}"
            f"<br><code>{escape(str(d.get('id')))}</code> → draft <code>{escape(str(i.get('id')))}</code></li>"
            for d, i in draft_only
        )
        body += (f"<p>📝 <strong>{len(draft_only)}</strong> covered only by an unpublished draft "
                 f"(<code>news_publish</code> to ship):</p><ul>{drafts_html}</ul>")
    stall_html = (f'<li style="color:#b91c1c">⚠️ RSS aggregator may be stalled '
                  f"(&gt;{STALL_HOURS}h without an update).</li>" if stalled else "")
    html = (
        f'<div style="font-family:system-ui,sans-serif;max-width:640px">'
        f"<h2>FixMyCert news-desk gap check — {today}</h2>{body}"
        f"<h3>Feed health</h3><ul>"
        f"<li>Hub deadlines: {len(deadlines)}</li>"
        f"<li>First-party items: {len(first_party)} ({len(published)} published)</li>"
        f"<li>Public feed items: {public_total}</li>"
        f"<li>Aggregator updatedAt: {escape(str(updated_at or '(absent)'))} — {age_note}</li>"
        f"{stall_html}</ul>"
        f'<p style="color:#666;font-size:12px">Read-only check. Source: '
        f"<code>fixmycert-news-mcp/news_gap_check.py</code> (droplet cron, Mondays 15:00 UTC). "
        f"Coverage counts archived items as decided, matching <code>news_get_uncovered</code>.</p></div>"
    )
    return subject, text, html


def send(subject: str, text: str, html: str) -> int:
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        print("ERROR: RESEND_API_KEY not set", file=sys.stderr)
        return 1
    try:
        r = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": os.environ.get("NEWS_EMAIL_FROM", DEFAULT_FROM),
                "to": os.environ.get("NEWS_EMAIL_TO", DEFAULT_TO),
                "subject": subject,
                "html": html,
                "text": text,
            },
            timeout=30,
        )
        r.raise_for_status()
        print(f"sent: id={r.json().get('id')} subject={subject!r}")
        return 0
    except httpx.HTTPStatusError as e:
        print(f"ERROR: Resend rejected ({e.response.status_code}): {e.response.text[:400]}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: {_redact(str(e))}", file=sys.stderr)
        return 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly FixMyCert news-desk gap check")
    ap.add_argument("--stdout", action="store_true", help="print the report, send no email")
    ap.add_argument("--json", action="store_true", help="print JSON results, send no email")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"--- news gap check {stamp} ---")

    base_url = os.environ.get("NEWS_API_BASE_URL", "").rstrip("/")
    secret = os.environ.get("NEWS_ADMIN_SECRET", "")
    try:
        if not base_url or not secret:
            raise FetchError("NEWS_API_BASE_URL and NEWS_ADMIN_SECRET must be set "
                             "(source /opt/mcp-servers/.env)")
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            deadlines = fetch_deadlines(client)
            first_party = fetch_first_party(client, base_url, secret)
            public_total, updated_at = fetch_public_health(client, base_url)
    except FetchError as e:
        # Never fail quietly: a fetch failure IS the report, so a silent Monday
        # always means "checked and clean", never "never ran".
        print(f"ERROR: {e}", file=sys.stderr)
        if args.stdout or args.json:
            return 2
        send("⚠️ [FixMyCert news] gap check FAILED — API unreachable",
             f"The weekly gap check could not complete.\n\n{e}\n\nNo coverage conclusion was reached.",
             f"<p>The weekly gap check could not complete.</p><pre>{escape(str(e))}</pre>"
             "<p>No coverage conclusion was reached.</p>")
        return 2

    uncovered, draft_only = classify(deadlines, first_party, SINCE_DAYS)
    subject, text, html = build_report(deadlines, first_party, uncovered, draft_only,
                                       public_total, updated_at)

    if args.json:
        print(json.dumps({
            "checkedAt": stamp,
            "deadlines": len(deadlines),
            "firstPartyItems": len(first_party),
            "publicFeedItems": public_total,
            "feedUpdatedAt": updated_at,
            "uncovered": uncovered,
            "uncoveredCount": len(uncovered),
            "urgentCount": sum(1 for d in uncovered if is_urgent(d)),
            "draftOnly": [{"deadlineId": d.get("id"), "itemId": i.get("id")} for d, i in draft_only],
        }, indent=2, default=str))
        return 0

    print(text)
    if args.stdout:
        return 0
    return send(subject, text, html)


if __name__ == "__main__":
    sys.exit(main())
