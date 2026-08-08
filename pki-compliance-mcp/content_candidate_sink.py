#!/usr/bin/env python3
"""Content-candidate sink: emission ledger + news-desk adapter.

The fifth pipeline outcome (see the Part A block in compliance_auto_refresh.py)
needs a place to land — a state machine writing to /dev/null is not a state
machine. This module owns BOTH halves of that:

  1. The ledger (~/.pki-compliance-mcp/content_candidates.json): one row per
     candidate signature, carrying `sink_status`:
       - "pending": recorded but not yet delivered to the news desk. Retried
         on every subsequent run. A candidate is NEVER lost to a sink failure.
       - "posted":  delivered as a news-desk draft; re-emission suppressed.
     Suppression happens ONLY on "posted" — recording a row before delivery
     succeeds must not count as delivered (an unset NEWS_ADMIN_SECRET on the
     first run would otherwise mark every candidate done and silently eat it).

  2. The news adapter: POST /api/news/items with status=draft, the same call
     fixmycert-news-mcp's news_add makes. The two systems share no code, so
     this is the thin adapter over the existing HTTP API — nothing in the
     Replit backend changes.

EMISSION SCOPING: this ledger suppresses content emission only. Nothing on the
deadline path (dedup_changes, auto_approve.classify) reads it, so an item that
later acquires a date certain qualifies as a deadline on its next pass no
matter what this file says about it.

Material change: a candidate whose prose hash changes is re-emitted
(sink_status back to "pending" with the fresh payload) — but at most once per
RE_EMIT_COOLDOWN_DAYS. The research model re-words the same story day to day
(the reason _review_sig hashes anchors, not text), so a bare hash-change
trigger would re-post the same EU procedure item every morning until adoption.
The cooldown caps that at one draft a week per story; hash changes inside the
window update the ledger quietly. Adoption itself needs no re-emission at all —
a date certain routes the item down the deadline path, which never reads this
ledger.

KNOWN QUIRK (news API, confirmed 2026-08-07): item creation ignores
publishedAt — every item gets an ingest date. Real-world dates therefore go in
the excerpt TEXT, and the displayed feed date is expected to be wrong.

Env (read at call time, host cron environment):
  NEWS_API_BASE_URL   e.g. https://fixmycert.com
  NEWS_ADMIN_SECRET   same value as in the FixMyCert Replit Secrets
Missing env is a loud log + rows stay pending — never an exception upward.
"""

import difflib
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import httpx

import compliance_auto_refresh as car

LEDGER_NAME = "content_candidates.json"
HTTP_TIMEOUT = 20.0
RE_EMIT_COOLDOWN_DAYS = 7  # see "Material change" in the module docstring

# newsCluster is required by the news API. Rule-based mapping, `other` as the
# deliberate fallback (decision 2026-08-07): drafts are not public, a wrong
# cluster costs nothing, and gating on a human would rebuild the review queue
# this state exists to bypass.
CLUSTER_RULES = [
    ("pqc", re.compile(r"post[- ]?quantum|\bpqc\b|ml[- ]?kem|ml[- ]?dsa"
                       r"|slh[- ]?dsa|\bhqc\b|\bfalcon\b|quantum[- ]safe", re.IGNORECASE)),
    ("smime", re.compile(r"s[/-]?mime", re.IGNORECASE)),
    ("code-signing", re.compile(r"code[- ]?sign", re.IGNORECASE)),
    ("ca-action", re.compile(r"root (store|program)|distrust|mass revocation", re.IGNORECASE)),
]
DEFAULT_CLUSTER = "other"
NEWS_CATEGORY = "research"
EXCERPT_MAX = 600


def _ledger_path():
    # Derived at call time so tests that redirect car.DATA_DIR are honored.
    return car.DATA_DIR / LEDGER_NAME


def load_ledger() -> dict:
    try:
        data = json.loads(_ledger_path().read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        # An unreadable ledger must not kill the run, but silently starting
        # fresh would re-emit everything — log loudly and emit nothing new.
        car.log(f"WARNING: unreadable {LEDGER_NAME} ({e}) — treating ALL "
                f"candidates as already recorded this run; fix the file")
        return None  # sentinel: caller skips this run's emission entirely


def save_ledger(ledger: dict) -> None:
    car.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ledger_path().write_text(json.dumps(ledger, indent=2, sort_keys=True))


def _content_hash(item: dict) -> str:
    """Material-change detector: hash of the normalized prose fields.

    Deliberately EXCLUDES the id: the research model invents a fresh id for
    the same topic every day (the whole reason _review_sig exists), so hashing
    id would turn every re-encounter into a "material change" and re-emit the
    same story daily — the exact noise failure this state exists to prevent.
    Reuses the pipeline's normalization so cosmetic rewording (case,
    punctuation) does not count as change, while new substance does."""
    text = " ".join(str(item.get(k) or "")
                    for k in ("title", "topic", "description", "reason"))
    return hashlib.sha256(car._norm(text).encode()).hexdigest()[:16]


def _resolve_sig(ledger: dict, item: dict, sig: str) -> str:
    """The ledger key this candidate belongs to.

    _review_sig is stable when an item carries ballot/regulation anchors, but
    most content candidates (EU legislative items especially) carry none, so
    the signature falls back to the first words of the text — which the model
    re-words daily. Without this, every rewording looked like a brand-new
    candidate and re-posted the same story every morning (caught by the
    cooldown tests before first deploy). Text-signature misses therefore fall
    back to fuzzy TITLE matching against existing rows, the same containment +
    0.6-ratio rule the news server uses for deadline coverage matching.
    Anchor signatures stay exact — anchors are the stable case by design.
    """
    if sig in ledger or sig.startswith("anchors:"):
        return sig
    title = car._norm(str(item.get("title") or item.get("topic") or ""))
    if not title:
        return sig
    for key, row in ledger.items():
        if not isinstance(row, dict):
            continue
        row_title = car._norm(str(row.get("title") or ""))
        if not row_title:
            continue
        if (title == row_title or title in row_title or row_title in title
                or difflib.SequenceMatcher(None, title, row_title).ratio() >= 0.6):
            return key
    return sig


def _cooldown_elapsed(row: dict, date_str: str) -> bool:
    """True when the last emission event is at least RE_EMIT_COOLDOWN_DAYS old.
    Unparseable dates fail toward NOT re-emitting — the noisy direction is the
    one that costs trust."""
    last = (row.get("last_material_change") or row.get("first_seen") or "")
    try:
        last_d = datetime.strptime(str(last), "%Y-%m-%d").date()
        today = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return False
    return (today - last_d).days >= RE_EMIT_COOLDOWN_DAYS


def _cluster_for(text: str) -> str:
    for cluster, pat in CLUSTER_RULES:
        if pat.search(text):
            return cluster
    return DEFAULT_CLUSTER


def _payload_for(item: dict, date_str: str) -> dict:
    """The news_add-shaped body for one candidate, stored on the ledger row so
    retries need no pending file."""
    title = (item.get("title") or item.get("topic") or item.get("id")
             or (item.get("description") or "")[:80] or "(untitled)")
    desc = (item.get("description") or item.get("reason") or "").strip()
    rule = item.get("content_rule", "no-date-certain")
    # The news API stamps ingest dates, so the real date context must live in
    # the text itself (see KNOWN QUIRK above).
    excerpt = (f"{desc[:EXCERPT_MAX]}"
               f"{'…' if len(desc) > EXCERPT_MAX else ''} "
               f"[Content candidate, flagged {date_str}; rule: {rule}. "
               f"Dates in text — the feed date is the ingest date.]").strip()
    urls = [u for u in (item.get("provenance_urls") or []) if isinstance(u, str)]
    primary = item.get("primary_url") if isinstance(item.get("primary_url"), str) else None
    body = {
        "title": str(title)[:200],
        "excerpt": excerpt,
        "category": NEWS_CATEGORY,
        "newsCluster": _cluster_for(f"{title} {desc}"),
        "isPriority": False,
        "status": "draft",
    }
    url = primary or (urls[0] if urls else None)
    if url:
        body["url"] = url
    return body


def _news_config() -> tuple[str, str]:
    return (os.environ.get("NEWS_API_BASE_URL", "").rstrip("/"),
            os.environ.get("NEWS_ADMIN_SECRET", ""))


def _post_news(payload: dict, base: str, secret: str) -> str:
    """POST one draft to the news desk. Returns the created item id.
    Raises on any failure; the caller decides what failure means."""
    resp = httpx.post(
        f"{base}/api/news/items",
        json=payload,
        headers={"Authorization": f"Bearer {secret}"},
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
    )
    resp.raise_for_status()
    data = resp.json()
    item = data.get("item", data) if isinstance(data, dict) else {}
    return str(item.get("id", "")) if isinstance(item, dict) else ""


def process(changes: dict, date_str: str) -> dict:
    """Record today's candidates in the ledger, then deliver every pending row.

    Returns a small stats dict (recorded / suppressed / posted / failed) for
    tests and logs. Never raises on sink trouble — the caller additionally
    wraps this in try/except as a belt-and-braces rule that the morning run
    survives anything this module does.
    """
    stats = {"recorded": 0, "material_change": 0, "suppressed": 0,
             "posted": 0, "failed": 0}
    ledger = load_ledger()
    if ledger is None:  # unreadable file — see load_ledger
        return stats

    now = datetime.now(timezone.utc).isoformat()
    for it in changes.get("content_candidates") or []:
        if not isinstance(it, dict):
            continue
        sig = _resolve_sig(ledger, it, car._review_sig(it))
        h = _content_hash(it)
        row = ledger.get(sig)
        label = it.get("title") or it.get("id") or (it.get("description") or "")[:60]
        if row is None:
            ledger[sig] = {
                "first_seen": date_str,
                "content_hash": h,
                "sink_status": "pending",
                "rule": it.get("content_rule", "no-date-certain"),
                "rule_matched": it.get("content_rule_matched", ""),
                "title": str(label)[:200],
                "provenance_urls": [u for u in (it.get("provenance_urls") or [])
                                    if isinstance(u, str)][:3],
                "primary_url": it.get("primary_url")
                               if isinstance(it.get("primary_url"), str) else None,
                "payload": _payload_for(it, date_str),
                "attempts": 0,
                "recorded_at": now,
            }
            stats["recorded"] += 1
            car.log(f"content sink: recorded candidate [{ledger[sig]['rule']}] "
                    f"{label!r} (sig {sig})")
        elif row.get("content_hash") != h:
            row["content_hash"] = h
            if _cooldown_elapsed(row, date_str):
                # Material change outside the cooldown: refresh the payload and
                # re-emit, keeping first_seen so the story stays in one row.
                row["sink_status"] = "pending"
                row["payload"] = _payload_for(it, date_str)
                row["last_material_change"] = date_str
                stats["material_change"] += 1
                car.log(f"content sink: material change on {label!r} — re-emitting")
            else:
                stats["suppressed"] += 1
                car.log(f"content sink: text changed on {label!r} within the "
                        f"{RE_EMIT_COOLDOWN_DAYS}d cooldown — hash updated, not re-emitting")
        else:
            stats["suppressed"] += 1
            car.log(f"content sink: suppressed (already recorded, unchanged): {label!r}")

    pending = {sig: row for sig, row in ledger.items()
               if isinstance(row, dict) and row.get("sink_status") == "pending"}
    if pending:
        base, secret = _news_config()
        if not base or not secret:
            car.log(f"WARNING: content sink cannot deliver {len(pending)} pending "
                    f"candidate(s) — NEWS_API_BASE_URL/NEWS_ADMIN_SECRET not set in "
                    f"this environment. Rows stay pending and retry next run; "
                    f"candidates remain visible in the ledger and the daily email.")
        else:
            for sig, row in pending.items():
                row["attempts"] = int(row.get("attempts") or 0) + 1
                row["last_attempt"] = now
                try:
                    news_id = _post_news(row.get("payload") or {}, base, secret)
                    row["sink_status"] = "posted"
                    row["news_id"] = news_id
                    row["posted_at"] = now
                    stats["posted"] += 1
                    car.log(f"content sink: posted draft {news_id or '(id unknown)'} "
                            f"for {row.get('title')!r}")
                except Exception as e:
                    stats["failed"] += 1
                    car.log(f"WARNING: content sink POST failed for "
                            f"{row.get('title')!r} (attempt {row['attempts']}): "
                            f"{type(e).__name__}: {e} — row stays pending")

    save_ledger(ledger)
    return stats
