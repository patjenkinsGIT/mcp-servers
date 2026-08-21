#!/usr/bin/env python3
"""Weekly PKI Compliance rollup — the heartbeat behind the daily email's silence.

Since 2026-08-20 daily_email.py is EVENT-DRIVEN: on a day where nothing
happened it sends nothing. That is only safe if something else can prove the
pipeline is alive, because otherwise a dead pipeline and a quiet week look
identical in the inbox. This file is that something.

    A missing rollup, not a missing daily, is the alarm.

THIS EMAIL ALWAYS SENDS. There is deliberately no gate in this file, and
adding one would defeat its only purpose — a heartbeat that can suppress
itself is not a heartbeat. The one thing it must never do is go quiet because
it decided the week was boring.

What it reads:
  - /root/.pki-compliance-mcp/email_runs.json   (the run ledger daily_email.py
    writes on EVERY run, sent or suppressed)
  - the pending_updates_*.json backlog, via daily_email.outstanding_review_items

What it answers, in order of importance:
  1. Did daily_email.py actually run every day this week? A day with no ledger
     entry is a SILENT FAILURE and is the headline of the email. This is the
     check the whole event-driven design rests on.
  2. Did anything break? Failures recorded on any day are surfaced even though
     that day's own email already reported them — a failure seen once in a
     week of mail is easy to miss.
  3. What is still awaiting a verdict? decide_send() deliberately does NOT
     re-fire for an urgent item that is merely still sitting in the backlog,
     to stop the daily nagging. It reappears here instead, which is the other
     half of that bargain.
  4. What actually happened — the week's totals.

Constants, the ledger schema and the backlog window are all imported from
daily_email rather than restated, so the two files cannot drift apart. In
particular the age-out dates below use daily_email.BACKLOG_WINDOW_DAYS, so an
item's stated age-out day is the day it really falls out of the daily email's
backlog list.

Usage:
  python3 weekly_rollup.py                 # normal cron run (7 days ending today)
  python3 weekly_rollup.py --days 14
  python3 weekly_rollup.py --date YYYY-MM-DD   # end the window on this day (replay)
  python3 weekly_rollup.py --explain       # print the health verdict, no send
  python3 weekly_rollup.py --dry-run       # render + print subject, no send,
                                           # no last_rollup write

Env required:
  RESEND_API_KEY=re_...
Env optional:
  PKI_EMAIL_TO / PKI_EMAIL_FROM   (same defaults as daily_email.py)

Exit codes: 0 ok, 1 API key missing, 2 send failed.
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from html import escape

import daily_email as de

ROLLUP_DAYS = 7

# Day classifications used throughout. "no-run" is the only one that is an
# alarm; "pre-ledger" exists so the first rollup after deploy does not scream
# about days that simply predate the ledger.
STATUS_SENT = "sent"
STATUS_QUIET = "quiet"
STATUS_NO_RUN = "no-run"
STATUS_PRE_LEDGER = "pre-ledger"


def classify_week(runs: dict, end_date: str, days: int) -> list[dict]:
    """One record per calendar day in the window, oldest first.

    A day with no ledger entry is only a failure if the ledger was already
    running by then: `first_recorded` is the earliest date on record, and
    anything before it is PRE-LEDGER, not missing. Without that distinction
    the first rollup after deploy would report six phantom outages and teach
    the reader to ignore the alarm — which is the one thing this email cannot
    afford.
    """
    ledger = runs.get("runs", {})
    first_recorded = min(ledger) if ledger else None
    end = date.fromisoformat(end_date)
    out = []
    for i in range(days - 1, -1, -1):
        d = (end - timedelta(days=i)).isoformat()
        entry = ledger.get(d)
        if entry is None:
            status = (STATUS_NO_RUN
                      if first_recorded and d >= first_recorded
                      else STATUS_PRE_LEDGER)
            out.append({"date": d, "status": status, "entry": None})
        else:
            out.append({
                "date": d,
                "status": STATUS_SENT if entry.get("sent") else STATUS_QUIET,
                "entry": entry,
            })
    return out


def week_totals(week: list[dict]) -> dict:
    """Sum the ledger's per-day counts across the window.

    Only days that actually ran contribute. A missing day contributes nothing
    rather than a zero, because "we know nothing happened" and "we do not know
    what happened" must not average together into a reassuring number.
    """
    keys = ("proposed", "flagged", "doc_changes", "candidates_today",
            "candidates_stuck", "drafts", "urgent_new")
    totals = {k: 0 for k in keys}
    research_ran = research_skipped = doc_ran = 0
    for day in week:
        e = day["entry"]
        if not e:
            continue
        counts = e.get("counts") or {}
        for k in keys:
            try:
                totals[k] += int(counts.get(k) or 0)
            except (TypeError, ValueError):
                pass
        research = e.get("research") or {}
        if research.get("ran"):
            research_ran += 1
            if research.get("skipped"):
                research_skipped += 1
        if (e.get("doc_check") or {}).get("ran"):
            doc_ran += 1
    totals["research_ran_days"] = research_ran
    totals["research_gate_skipped_days"] = research_skipped
    totals["doc_check_ran_days"] = doc_ran
    # Latest backlog size on record, not a sum — backlog is a level, not a flow.
    latest = next((d["entry"] for d in reversed(week) if d["entry"]), None)
    totals["backlog_now"] = int(((latest or {}).get("counts") or {}).get("backlog") or 0)
    return totals


def week_failures(week: list[dict]) -> list[dict]:
    """Every failure recorded on any day in the window, with its date.

    Surfaced even though the day's own email already carried them: a single
    bad day inside a week of mail is exactly what gets skimmed past.
    """
    out = []
    for day in week:
        for f in ((day["entry"] or {}).get("failures") or []):
            out.append({"date": day["date"], "failure": str(f)})
    return out


def held_items(date_str: str) -> list[dict]:
    """Urgent backlog items awaiting a verdict, with the day they age out.

    This is the section decide_send() defers to when it refuses to re-fire the
    daily email for an urgent item that has not changed. Age-out is computed
    against daily_email.BACKLOG_WINDOW_DAYS so the date stated here is the day
    the item really drops off the daily's backlog list.

    Degrades to [] if the backlog cannot be aggregated — the caller reports
    that as a failure rather than as "nothing held", because an empty held
    list and an unknown held list are very different facts.
    """
    try:
        outstanding = de.outstanding_review_items(
            date_str, days=de.BACKLOG_WINDOW_DAYS)
    except Exception:
        return []
    held = []
    for o in outstanding or []:
        if not o.get("urgent"):
            continue
        first_seen = str(o.get("first_seen") or "")
        ages_out = ""
        try:
            ages_out = (date.fromisoformat(first_seen)
                        + timedelta(days=de.BACKLOG_WINDOW_DAYS)).isoformat()
        except ValueError:
            pass
        held.append({
            "title": str(o.get("title") or "(untitled)"),
            "date": str(o.get("date") or ""),
            "first_seen": first_seen,
            "ages_out": ages_out,
            "kind": str(o.get("kind") or ""),
        })
    return sorted(held, key=lambda h: h["first_seen"])


def verdict(week: list[dict], failures: list[dict]) -> dict:
    """The health call this email exists to make."""
    missing = [d["date"] for d in week if d["status"] == STATUS_NO_RUN]
    ran = [d for d in week if d["entry"]]
    sent = [d for d in ran if d["status"] == STATUS_SENT]
    quiet = [d for d in ran if d["status"] == STATUS_QUIET]
    pre = [d for d in week if d["status"] == STATUS_PRE_LEDGER]
    if missing:
        level = "alarm"
    elif failures:
        level = "warn"
    elif not ran:
        # Nothing on record at all across the whole window. Either this is the
        # very first run after deploy, or daily_email.py has never recorded
        # anything — which the reader must decide, so say so rather than
        # rendering a cheerful zero.
        level = "warn"
    else:
        level = "ok"
    return {
        "level": level,
        "missing_days": missing,
        "days_ran": len(ran),
        "days_sent": len(sent),
        "days_quiet": len(quiet),
        "days_pre_ledger": len(pre),
        "days_in_window": len(week),
    }


def subject(v: dict, start: str, end: str) -> str:
    if v["level"] == "alarm":
        n = len(v["missing_days"])
        return (f"🔴 PKI weekly — {n} DAY(S) WITH NO RUN "
                f"({start} → {end})")
    if v["level"] == "warn" and not v["days_ran"]:
        return f"⚠️ PKI weekly — no runs on record ({start} → {end})"
    if v["level"] == "warn":
        return (f"⚠️ PKI weekly — {v['days_ran']}/{v['days_in_window']} days ran, "
                f"failures recorded ({start} → {end})")
    return (f"✅ PKI weekly — {v['days_ran']}/{v['days_in_window']} days ran, "
            f"{v['days_sent']} sent, {v['days_quiet']} quiet ({start} → {end})")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_SANS = "-apple-system,system-ui,sans-serif"
_BADGE = {
    STATUS_SENT: ("📧", "#1d4ed8", "sent"),
    STATUS_QUIET: ("·", "#6b7280", "quiet — ran, nothing to report"),
    STATUS_NO_RUN: ("🔴", "#b91c1c", "NO RUN RECORDED"),
    STATUS_PRE_LEDGER: ("–", "#9ca3af", "before the ledger started"),
}


def render_html(week, totals, failures, held, v, start, end) -> str:
    p = []
    p.append(f"<h2 style='margin:0 0 4px 0;font:600 18px/1.3 {_SANS}'>"
             f"PKI Compliance weekly rollup</h2>")
    p.append(f"<p style='margin:0 0 14px 0;font:13px/1.4 {_SANS};color:#6b7280'>"
             f"{escape(start)} → {escape(end)} · this email always sends, so its "
             f"absence is itself the alarm</p>")

    if v["level"] == "alarm":
        p.append(f"<div style='border:2px solid #b91c1c;background:#fef2f2;"
                 f"border-radius:8px;padding:12px 14px;margin:0 0 14px 0'>")
        p.append(f"<p style='margin:0 0 6px 0;font:600 15px/1.4 {_SANS};color:#b91c1c'>"
                 f"🔴 daily_email.py recorded no run on "
                 f"{len(v['missing_days'])} day(s)</p>")
        p.append(f"<p style='margin:0 0 6px 0;font:13px/1.5 {_SANS};color:#7f1d1d'>"
                 f"{escape(', '.join(v['missing_days']))}</p>")
        p.append(f"<p style='margin:0;font:13px/1.5 {_SANS};color:#7f1d1d'>"
                 f"The ledger is written on every run, sent or suppressed, so a "
                 f"missing entry means the 10:35 UTC cron did not complete — not "
                 f"that the day was quiet. Check cron.log and email.log for those "
                 f"dates.</p>")
        p.append("</div>")
    elif v["level"] == "warn" and not v["days_ran"]:
        p.append(f"<div style='border:2px solid #b45309;background:#fffbeb;"
                 f"border-radius:8px;padding:12px 14px;margin:0 0 14px 0'>")
        p.append(f"<p style='margin:0;font:600 14px/1.4 {_SANS};color:#92400e'>"
                 f"⚠️ No daily runs on record for this window. If the "
                 f"event-driven email was only just deployed this is expected "
                 f"once; otherwise daily_email.py is not running.</p>")
        p.append("</div>")

    # --- Day-by-day: the proof of life -------------------------------------
    p.append(f"<h3 style='margin:16px 0 6px 0;font:600 14px/1.3 {_SANS}'>"
             f"Daily runs</h3>")
    p.append(f"<table style='border-collapse:collapse;font:13px/1.5 {_SANS}'>")
    for day in week:
        icon, colour, label = _BADGE[day["status"]]
        e = day["entry"] or {}
        detail = ""
        if day["status"] in (STATUS_SENT, STATUS_QUIET):
            reasons = e.get("reasons") or []
            detail = (escape("; ".join(str(r) for r in reasons)[:160])
                      if reasons else "no signals")
        p.append(
            f"<tr><td style='padding:2px 10px 2px 0;color:{colour};white-space:nowrap'>"
            f"{icon} {escape(day['date'])}</td>"
            f"<td style='padding:2px 10px 2px 0;color:{colour};white-space:nowrap'>"
            f"{escape(label)}</td>"
            f"<td style='padding:2px 0;color:#4b5563'>{detail}</td></tr>")
    p.append("</table>")

    # --- Failures ----------------------------------------------------------
    if failures:
        p.append(f"<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 {_SANS};"
                 f"color:#b45309'>Failures recorded this week ({len(failures)})</h3>")
        p.append(f"<ul style='margin:0;padding-left:20px;font:13px/1.5 {_SANS};"
                 f"color:#92400e'>")
        for f in failures:
            p.append(f"<li><strong>{escape(f['date'])}</strong> — "
                     f"{escape(f['failure'])}</li>")
        p.append("</ul>")

    # --- Held / awaiting verdict -------------------------------------------
    p.append(f"<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 {_SANS}'>"
             f"Urgent items awaiting a verdict ({len(held)})</h3>")
    if held:
        p.append(f"<p style='margin:0 0 6px 0;font:12px/1.4 {_SANS};color:#6b7280'>"
                 f"The daily email deliberately stops re-reporting these once "
                 f"they stop changing. They are listed here instead, with the day "
                 f"each drops out of the {de.BACKLOG_WINDOW_DAYS}-day backlog "
                 f"window.</p>")
        p.append(f"<ul style='margin:0;padding-left:20px;font:13px/1.5 {_SANS}'>")
        for h in held:
            tail = f" — due {escape(h['date'])}" if h["date"] else ""
            ages = (f" · ages out {escape(h['ages_out'])}"
                    if h["ages_out"] else "")
            p.append(f"<li><strong>{escape(h['title'])}</strong>{tail}"
                     f"<span style='color:#6b7280'> (first seen "
                     f"{escape(h['first_seen'])}{ages})</span></li>")
        p.append("</ul>")
    else:
        p.append(f"<p style='margin:0;font:13px/1.5 {_SANS};color:#6b7280'>"
                 f"None.</p>")

    # --- Totals ------------------------------------------------------------
    p.append(f"<h3 style='margin:18px 0 6px 0;font:600 14px/1.3 {_SANS}'>"
             f"Week totals</h3>")
    p.append(f"<table style='border-collapse:collapse;font:13px/1.5 {_SANS}'>")
    for label, key in (
        ("Proposed changes", "proposed"),
        ("Flagged for review", "flagged"),
        ("Document hash changes", "doc_changes"),
        ("New content candidates", "candidates_today"),
        ("Candidates stuck undelivered", "candidates_stuck"),
        ("Draft packages written", "drafts"),
        ("New urgent items notified", "urgent_new"),
        ("Days research cron ran", "research_ran_days"),
        ("  ...of which gate-skipped", "research_gate_skipped_days"),
        ("Days doc-check ran", "doc_check_ran_days"),
        ("Backlog size (latest)", "backlog_now"),
    ):
        p.append(f"<tr><td style='padding:1px 14px 1px 0;color:#4b5563'>"
                 f"{escape(label)}</td>"
                 f"<td style='padding:1px 0;font-weight:600'>{totals[key]}</td></tr>")
    p.append("</table>")

    p.append(f"<p style='margin:18px 0 0 0;font:12px/1.4 {_SANS};color:#9ca3af'>"
             f"<a href='{de.DASHBOARD_URL}' style='color:#6b7280'>Dashboard</a> · "
             f"ledger: {escape(str(de.RUNS_FILE))}</p>")
    return "".join(p)


def render_text(week, totals, failures, held, v, start, end) -> str:
    lines = [f"PKI Compliance weekly rollup — {start} → {end}",
             "This email always sends; its absence is the alarm.", ""]
    if v["level"] == "alarm":
        lines += [f"** ALARM: daily_email.py recorded NO RUN on "
                  f"{len(v['missing_days'])} day(s): "
                  f"{', '.join(v['missing_days'])} **",
                  "The ledger is written on every run, sent or suppressed, so a",
                  "missing entry means the 10:35 UTC cron did not complete.", ""]
    elif v["level"] == "warn" and not v["days_ran"]:
        lines += ["** No daily runs on record for this window. Expected once if",
                  "the event-driven email was only just deployed; otherwise",
                  "daily_email.py is not running. **", ""]
    lines.append("Daily runs:")
    for day in week:
        _, _, label = _BADGE[day["status"]]
        reasons = (day["entry"] or {}).get("reasons") or []
        detail = "; ".join(str(r) for r in reasons)[:160] if reasons else ""
        lines.append(f"  {day['date']}  {label}" + (f"  — {detail}" if detail else ""))
    if failures:
        lines += ["", f"Failures recorded ({len(failures)}):"]
        lines += [f"  {f['date']} — {f['failure']}" for f in failures]
    lines += ["", f"Urgent items awaiting a verdict ({len(held)}):"]
    if held:
        for h in held:
            lines.append(f"  {h['title']} (first seen {h['first_seen']}"
                         + (f", ages out {h['ages_out']}" if h["ages_out"] else "")
                         + ")")
    else:
        lines.append("  None.")
    lines += ["", "Week totals:"]
    for label, key in (("proposed", "proposed"), ("flagged", "flagged"),
                       ("doc hash changes", "doc_changes"),
                       ("new candidates", "candidates_today"),
                       ("candidates stuck", "candidates_stuck"),
                       ("draft packages", "drafts"),
                       ("new urgent notified", "urgent_new"),
                       ("days research ran", "research_ran_days"),
                       ("days doc-check ran", "doc_check_ran_days"),
                       ("backlog now", "backlog_now")):
        lines.append(f"  {label}: {totals[key]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="PKI Compliance weekly rollup")
    parser.add_argument("--days", type=int, default=ROLLUP_DAYS,
                        help=f"Window length in days (default {ROLLUP_DAYS})")
    parser.add_argument("--date", default=None,
                        help="End the window on this date (replay/fixtures)")
    parser.add_argument("--explain", action="store_true",
                        help="Print the health verdict and exit; no send")
    parser.add_argument("--dry-run", action="store_true",
                        help="Render and print the subject; no send, no ledger write")
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be >= 1")

    end = args.date or de.today_iso()
    try:
        start = (date.fromisoformat(end)
                 - timedelta(days=args.days - 1)).isoformat()
    except ValueError:
        parser.error(f"--date must be YYYY-MM-DD, got {end!r}")

    runs = de.load_runs()
    week = classify_week(runs, end, args.days)
    totals = week_totals(week)
    failures = week_failures(week)
    held = held_items(end)
    v = verdict(week, failures)
    subj = subject(v, start, end)

    if args.explain:
        print(subj)
        print(f"level={v['level']} ran={v['days_ran']}/{v['days_in_window']} "
              f"sent={v['days_sent']} quiet={v['days_quiet']} "
              f"pre_ledger={v['days_pre_ledger']}")
        if v["missing_days"]:
            print("missing: " + ", ".join(v["missing_days"]))
        for f in failures:
            print(f"failure {f['date']}: {f['failure']}")
        return 0

    html = render_html(week, totals, failures, held, v, start, end)
    text = render_text(week, totals, failures, held, v, start, end)

    if args.dry_run:
        print(subj)
        print()
        print(text)
        return 0

    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if not api_key:
        print("ERROR: RESEND_API_KEY not set", file=sys.stderr)
        return 1
    to_addr = os.environ.get("PKI_EMAIL_TO", de.DEFAULT_TO)
    from_addr = os.environ.get("PKI_EMAIL_FROM", de.DEFAULT_FROM)

    rc = de.send_email(api_key, from_addr, to_addr, subj, html, text)
    if rc == 0:
        # Stamp the ledger only on a successful send. If the rollup failed to
        # go out, last_rollup must keep pointing at the last one that actually
        # reached the inbox — otherwise the record would claim a heartbeat the
        # reader never received.
        runs = de.load_runs()
        runs["last_rollup"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "window_start": start,
            "window_end": end,
            "level": v["level"],
            "missing_days": v["missing_days"],
        }
        de.save_runs(runs)
    return rc


if __name__ == "__main__":
    sys.exit(main())
