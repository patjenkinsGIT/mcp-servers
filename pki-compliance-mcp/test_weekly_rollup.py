#!/usr/bin/env python3
"""Offline tests for weekly_rollup.py — the heartbeat behind the daily email's
silence (2026-08-20).

No network, no Resend, no Anthropic API. The load-bearing properties:

  1. A missing ledger entry on a day the ledger was already running is an
     ALARM. This is the only reason the file exists: it is what distinguishes
     a dead pipeline from a quiet week now that daily_email.py suppresses
     itself on no-op days.
  2. A missing entry BEFORE the ledger's first record is not an alarm. Without
     this the first rollup after deploy reports phantom outages and trains the
     reader to ignore the one alarm that matters.
  3. The rollup never suppresses itself. There is no gate; a heartbeat that
     can go quiet is not a heartbeat.
  4. Totals count only days that ran. A missing day must not contribute a
     reassuring zero.
"""
import tempfile
from datetime import date, timedelta
from pathlib import Path

import daily_email as de
import weekly_rollup as wr

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


def day(entry_sent, proposed=0, failures=None, ran=True):
    """One ledger entry in the shape daily_email.record_run writes."""
    return {
        "ran_at": "2026-08-20T10:35:00+00:00",
        "sent": entry_sent,
        "reasons": ["1 proposed change(s)"] if proposed else [],
        "counts": {"proposed": proposed, "flagged": 0, "doc_changes": 0,
                   "candidates_today": 0, "candidates_stuck": 0, "drafts": 0,
                   "urgent_new": 0, "backlog": 3},
        "backlog_sig": "abc123",
        "backlog_changed": bool(proposed),
        "failures": failures or [],
        "research": {"ran": ran, "started_at": "2026-08-20T10:00:01+00:00",
                     "skipped": False, "skip_reason": ""},
        "doc_check": {"ran": ran, "checked_at": "2026-08-20T10:30:00+00:00",
                      "changes_detected": 0},
    }


def week_of(end="2026-08-21", n=7):
    return [(date.fromisoformat(end) - timedelta(days=i)).isoformat()
            for i in range(n - 1, -1, -1)]


END = "2026-08-21"
DAYS = week_of(END)

print("== classify_week ==")

# Every day present: nothing missing.
full = {"runs": {d: day(False) for d in DAYS}}
w = wr.classify_week(full, END, 7)
check("one record per day", len(w) == 7)
check("oldest first", w[0]["date"] == DAYS[0] and w[-1]["date"] == END)
check("all quiet when nothing sent",
      all(x["status"] == wr.STATUS_QUIET for x in w))
check("sent day classified sent",
      wr.classify_week({"runs": {**full["runs"], END: day(True)}},
                       END, 7)[-1]["status"] == wr.STATUS_SENT)

# THE ALARM: a hole in the middle of a running ledger.
holed = {"runs": {d: day(False) for d in DAYS if d != DAYS[3]}}
w = wr.classify_week(holed, END, 7)
check("gap inside a running ledger is no-run",
      w[3]["status"] == wr.STATUS_NO_RUN)
check("only the gap is no-run",
      sum(1 for x in w if x["status"] == wr.STATUS_NO_RUN) == 1)

# THE NON-ALARM: days before the ledger ever recorded anything.
fresh = {"runs": {DAYS[5]: day(False), DAYS[6]: day(False)}}
w = wr.classify_week(fresh, END, 7)
check("days before first record are pre-ledger, not missing",
      [x["status"] for x in w[:5]] == [wr.STATUS_PRE_LEDGER] * 5)
check("no false alarm on a fresh deploy",
      not any(x["status"] == wr.STATUS_NO_RUN for x in w))

# Empty ledger: everything pre-ledger, still no alarm.
w = wr.classify_week({"runs": {}}, END, 7)
check("empty ledger yields no no-run days",
      all(x["status"] == wr.STATUS_PRE_LEDGER for x in w))

print("== verdict ==")
check("clean week is ok",
      wr.verdict(wr.classify_week(full, END, 7), [])["level"] == "ok")
check("missing day is an alarm",
      wr.verdict(wr.classify_week(holed, END, 7), [])["level"] == "alarm")
check("failures downgrade to warn",
      wr.verdict(wr.classify_week(full, END, 7),
                 [{"date": END, "failure": "x"}])["level"] == "warn")
check("missing day outranks failures",
      wr.verdict(wr.classify_week(holed, END, 7),
                 [{"date": END, "failure": "x"}])["level"] == "alarm")
check("empty ledger is warn, not ok",
      wr.verdict(wr.classify_week({"runs": {}}, END, 7), [])["level"] == "warn")
_v = wr.verdict(wr.classify_week(holed, END, 7), [])
check("verdict names the missing date", _v["missing_days"] == [DAYS[3]])
check("verdict counts days that ran", _v["days_ran"] == 6)

print("== subject ==")
_alarm = wr.subject(wr.verdict(wr.classify_week(holed, END, 7), []), DAYS[0], END)
check("alarm subject leads with the siren", _alarm.startswith("🔴"))
check("alarm subject states the count", "1 DAY(S) WITH NO RUN" in _alarm)
_ok = wr.subject(wr.verdict(wr.classify_week(full, END, 7), []), DAYS[0], END)
check("healthy subject leads with a tick", _ok.startswith("✅"))
check("healthy subject shows ran/total", "7/7 days ran" in _ok)

print("== week_totals ==")
mixed = {"runs": {DAYS[0]: day(True, proposed=2),
                  DAYS[1]: day(True, proposed=3),
                  DAYS[2]: day(False)}}
t = wr.week_totals(wr.classify_week(mixed, END, 7))
check("sums counts across days that ran", t["proposed"] == 5)
check("backlog is a level, not a sum", t["backlog_now"] == 3)
check("counts days research ran", t["research_ran_days"] == 3)
_none = wr.week_totals(wr.classify_week({"runs": {}}, END, 7))
check("missing days contribute nothing", _none["proposed"] == 0)
check("unknown backlog reports zero, not a stale level",
      _none["backlog_now"] == 0)

print("== week_failures ==")
withf = {"runs": {DAYS[0]: day(True, failures=["research cron did not run"]),
                  DAYS[1]: day(False)}}
f = wr.week_failures(wr.classify_week(withf, END, 7))
check("failure surfaced with its date",
      len(f) == 1 and f[0]["date"] == DAYS[0])
check("failure text preserved",
      "research cron did not run" in f[0]["failure"])

print("== held_items degrades ==")
_tmp = Path(tempfile.mkdtemp())
_orig = de.DATA_DIR
try:
    de.DATA_DIR = _tmp  # no pending files, and no network in tests
    held = wr.held_items(END)
    check("held_items returns a list when there is nothing to hold",
          isinstance(held, list))
finally:
    de.DATA_DIR = _orig

print("== rendering ==")
_w = wr.classify_week(holed, END, 7)
_t = wr.week_totals(_w)
_f = wr.week_failures(_w)
_vv = wr.verdict(_w, _f)
html = wr.render_html(_w, _t, _f, [], _vv, DAYS[0], END)
text = wr.render_text(_w, _t, _f, [], _vv, DAYS[0], END)
check("html names the missing day", DAYS[3] in html)
check("html says the absence is the alarm", "absence is itself the alarm" in html)
check("text names the missing day", DAYS[3] in text)
check("text carries the alarm banner", "ALARM" in text)
check("html renders every day row", all(d in html for d in DAYS))
_hheld = [{"title": "LE CP/CPS gap", "date": "", "first_seen": "2026-08-20",
           "ages_out": "2026-09-03", "kind": "review"}]
_h = wr.render_html(_w, _t, _f, _hheld, _vv, DAYS[0], END)
check("held item rendered with its age-out",
      "LE CP/CPS gap" in _h and "2026-09-03" in _h)

print("== no self-suppression ==")
# The rollup must have no gate. If someone adds one, this catches it.
_src = Path(__file__).with_name("weekly_rollup.py").read_text()
check("no decide_send-style gate in the rollup", "def decide_send" not in _src)
check("send is unconditional on the happy path",
      _src.count("de.send_email(") == 1)

print()
print(f"{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
