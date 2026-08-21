#!/usr/bin/env python3
"""Urgent-event notifier, and content drafter on explicit request.

NOTIFY-FIRST SINCE 2026-08-20. The cron run NOTIFIES and DRAFTS NOTHING.
Drafting is a separate, explicit trigger with an explicit channel set.

Why the shape changed: an item tagged urgent:true used to generate a full
five-channel package before any human had decided the story was worth
telling. On 2026-08-20 the flag was wrong, the package was wrong in all five
channels, and Pat then permanently DECLINED two channels (Kit broadcast and
YouTube) and rewrote from scratch the two he shipped. Most of the generated
output was discarded. Deciding first and drafting second costs nothing when
the answer is "don't", and the ~2.6-minute source-verified package stops
being generated speculatively.

Chained in the 10:00 UTC crontab line after auto_approve.py. When today's
research produced items tagged "urgent": true (announced mass revocation,
distrust of a trusted root/CA, or a compliance obligation landing within ~60
days), the cron run:

  - pushes an iPhone notification for items not notified before (ntfy.sh)
  - writes urgent_notice_<date>.json — what it is, why it was flagged, its
    source URLs, and the exact command that would draft it — which
    daily_email.py renders as a notification block
  - drafts NOTHING

Nothing is dropped: an urgent item still reaches the review queue, the
rolling backlog and the email exactly as before. Only the speculative
drafting is gone.

When drafting IS triggered (--draft --channels ...), it produces the
requested subset of the five-piece FixMyCert content package:

  blog.md          - FixMyCert blog post (markdown, enterprise cert-team reader)
  linkedin.md      - LinkedIn post
  youtube.md       - full YouTube publish package (NotebookLM audio + visual
                     prompts, title, description, pinned comment, thumbnail
                     prompt) following the fixmycert-youtube-publisher skill
                     conventions
  tweet.md         - X post (<=280 chars)
  kit_broadcast.md - Kit email broadcast (subject, preview text, body) with
                     an operator header: audience MUST be scoped to the
                     FixMyCert brand tag (brand:fmc, tag 19302998), never
                     the full list, and never auto-sent

DRAFTS ONLY - nothing is published anywhere. Files land in the repo under
content_drafts/<date>-<slug>/ and are committed + pushed so they show up
locally on git pull; daily_email.py surfaces the short pieces inline.

Dedup: content_drafted.json stores anchor signatures of items already
drafted, so an urgent item lingering in the 14-day review backlog only
drafts once. Rejected items (rejected via the morning review) never draft.

Always writes a line to content_drafts.log - even on no-urgent-items days -
because daily_email.py polls that file as the end-of-chain marker.

Usage:
  python3 content_drafts.py                  # cron run: notify only, draft nothing
  python3 content_drafts.py --date YYYY-MM-DD
  python3 content_drafts.py --list           # show today's urgent items + their
                                             # selectors, no push, no state write

  # Drafting — always explicit, always channel-scoped:
  python3 content_drafts.py --draft --channels blog,tweet
  python3 content_drafts.py --draft --channels all --item letsencrypt
  python3 content_drafts.py --draft --channels linkedin --dry-run

  python3 content_drafts.py --pending-file X # read urgent items from an
                                             # alternate pending file (fire drills)

Channels: blog, linkedin, tweet, kit, youtube — or "all". --channels is
REQUIRED with --draft and has no default: assuming all five is the behaviour
that produced four discarded channels on 2026-08-20.

Env: ANTHROPIC_API_KEY required to draft (a notify-only run does not need it).
PKI_REPO_PATH optional override, same default as the pipeline. NTFY_TOPIC
optional: when set, newly-seen urgent items trigger an iPhone push via
ntfy.sh (skipped on --dry-run and --list).
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

import compliance_auto_refresh as car

DATA_DIR = car.DATA_DIR
LOG_FILE = DATA_DIR / "content_drafts.log"
STATE_FILE = DATA_DIR / "content_drafted.json"
# Separate ledger from STATE_FILE on purpose: "we have told Pat about this"
# and "we have drafted copy for this" are now different facts, and an item can
# sit in the first state for days without ever entering the second.
NOTIFIED_FILE = DATA_DIR / "content_notified.json"
REPO = Path(os.environ.get("PKI_REPO_PATH", car.PKI_REPO_PATH))
DRAFTS_DIRNAME = "content_drafts"
MODEL = car.MODEL
MAX_DRAFTS_PER_RUN = 3  # an event wave should not turn into an API bill

# channel -> (output filename, the JSON key the model returns it under)
CHANNELS = {
    "blog": ("blog.md", "blog_markdown"),
    "linkedin": ("linkedin.md", "linkedin"),
    "tweet": ("tweet.md", "tweet"),
    "kit": ("kit_broadcast.md", "kit_email"),
    "youtube": ("youtube.md", "youtube"),
}


def log(message: str):
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"
    print(line)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Urgent item collection
# ---------------------------------------------------------------------------


def _load_ledger(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("signatures"), dict):
            return data
    except Exception:
        pass
    return {"signatures": {}}


def load_state() -> dict:
    return _load_ledger(STATE_FILE)


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_notified() -> dict:
    return _load_ledger(NOTIFIED_FILE)


def save_notified(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTIFIED_FILE.write_text(json.dumps(state, indent=2))


def collect_urgent(pending: dict, skip_drafted: bool = True) -> list[dict]:
    """Pull urgent-tagged items out of a pending-updates dict.

    Returns [{"kind", "sig", "item"}] minus rejected items, using the
    pipeline's signature machinery so "same item" means exactly what it means
    to the email backlog and dedup pass.

    `urgent` here is the flag the diff pass assigns under the three tests in
    DIFF_SYSTEM_PROMPT, held to the word "announced" by
    car.deescalate_speculative_urgent(). This module deliberately defines no
    second notion of urgency — a parallel definition is how the 2026-08-20
    class of bug comes back.

    skip_drafted=True (the notify path and the old default) also drops items
    already in content_drafted.json. The explicit --draft path passes False:
    when a human asks for copy, "nothing happened because we drafted this
    once before" is a worse failure than regenerating it.
    """
    rejected = car.load_rejected()
    drafted = set(load_state().get("signatures", {})) if skip_drafted else set()
    out = []
    for key, kind in (("new_deadlines", "deadline"),
                      ("updated_deadlines", "update"),
                      ("regulatory_updates", "regulatory")):
        for it in pending.get(key, []):
            if isinstance(it, dict) and it.get("urgent"):
                sig = car._sig(it)
                if (it.get("id") in rejected["ids"]
                        or sig in rejected["signatures"] or sig in drafted):
                    continue
                out.append({"kind": kind, "sig": sig, "item": it})
    for it in pending.get("needs_human_review", []):
        if isinstance(it, dict) and it.get("urgent"):
            sig = car._review_sig(it)
            if (it.get("id") in rejected["ids"]
                    or sig in rejected["signatures"] or sig in drafted):
                continue
            out.append({"kind": "flagged", "sig": sig, "item": it})
    return out


# ---------------------------------------------------------------------------
# Notification (the default path)
# ---------------------------------------------------------------------------


def _item_sources(item: dict) -> list[str]:
    urls = []
    for u in [item.get("primary_url")] + list(item.get("provenance_urls") or []):
        if isinstance(u, str) and u.strip() and u not in urls:
            urls.append(u.strip())
    return urls[:5]


def _selector(entry: dict) -> str:
    """Short, stable, greppable handle for --item. The id when there is one,
    else the signature — both are exact-matchable and neither changes between
    the notice and the draft run."""
    return str(entry["item"].get("id") or entry["sig"])


def build_notice(entry: dict, date_str: str, is_new: bool) -> dict:
    """One notification record: what it is, why it was flagged, its sources,
    and the command that turns it into copy. No drafted content."""
    item = entry["item"]
    why = (item.get("description") or item.get("reason") or item.get("impact")
           or item.get("update") or "(no description on the item)")
    return {
        "signature": entry["sig"],
        "selector": _selector(entry),
        "kind": entry["kind"],
        "id": item.get("id") or "",
        "title": item.get("title") or item.get("id") or item.get("topic") or "",
        "date": item.get("date") or "",
        "why": str(why),
        "source_urls": _item_sources(item),
        "is_new": is_new,
        "first_notified": date_str,
        "draft_command": (
            "cd /opt/mcp-servers/pki-compliance-mcp && python3 content_drafts.py "
            f"--draft --channels <blog|linkedin|tweet|kit|youtube|all> "
            f"--date {date_str} --item {_selector(entry)}"),
    }


def write_notice_file(date_str: str, notices: list[dict]) -> Path:
    """Persist the notifications daily_email.py renders. Written even when the
    list is empty so a reader can tell "checked, none" from "never ran"."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = DATA_DIR / f"urgent_notice_{date_str}.json"
    p.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "notify-only — no content drafted",
        "items": notices,
    }, indent=2))
    return p


# ---------------------------------------------------------------------------
# Push notification (ntfy.sh)
# ---------------------------------------------------------------------------


def notify_urgent(urgent: list[dict]) -> None:
    """Push an iPhone notification for new urgent items via ntfy.sh.

    No-op unless NTFY_TOPIC is set. Best effort: a push failure must never
    block the rest of the run, so errors are logged and swallowed.
    """
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    titles = "\n".join(
        f"• {(e['item'].get('title') or e['item'].get('id') or '(untitled)')[:120]}"
        for e in urgent
    )
    body = (f"{titles}\n\nNothing has been drafted. Review it in the morning "
            "email, then choose channels if it is worth telling.")
    try:
        httpx.post(
            f"https://ntfy.sh/{topic}",
            content=body.encode(),
            headers={
                "Title": f"🚨 {len(urgent)} urgent PKI event(s) detected",
                "Priority": "high",
                "Tags": "rotating_light,lock",
                "Click": "https://fixmycert.com/compliance",
            },
            timeout=15,
        ).raise_for_status()
        log(f"content_drafts: push notification sent ({len(urgent)} item(s))")
    except Exception as e:
        log(f"content_drafts: push notification failed ({e}) - continuing")


# ---------------------------------------------------------------------------
# Draft generation
# ---------------------------------------------------------------------------

CONTENT_SYSTEM_PROMPT = """\
You are the content writer for FixMyCert (https://fixmycert.com), a PKI \
education site. An automated pipeline flagged a PKI ecosystem event as \
urgent and you are drafting the content package. THAT URGENCY FLAG IS \
MACHINE-ASSIGNED AND OFTEN WRONG - treat it as a hint about priority, never \
as a fact about severity, and never let it set the tone of the copy. The \
reader is the enterprise certificate team - the people who own cert \
inventories, renewals, and compliance evidence - NOT certificate \
authorities. Practical, precise, calm-but-direct; explain what happened, \
who is affected, concrete dates, and what to do about it, which is \
sometimes nothing. Never invent facts, dates, or URLs beyond the event \
details and sources provided. If the event affects only CAs, say plainly \
that enterprise teams need no action unless they operate a CA.

FIRST, CLASSIFY THE EVENT, because most of what follows branches on it. \
Decide whether it imposes a CONCRETE ACTION on the enterprise certificate \
team on a FUTURE DATE CERTAIN.

- ACTION event: there is something the reader must actually do, and a \
future date by which to do it.
- NO-ACTION event: there is not. This covers CA-side compliance matters, \
policy-document defects, past-dated obligations, open questions, and \
anything whose only dates have already passed.

A defect in a CA's policy document - its CP/CPS, its disclosures, its audit \
paperwork - IS NOT a defect in the certificates that CA issued, and does \
not by itself require anyone to touch their certificate inventory. \
Revocation obligations attach to mis-issuance, not to paperwork. Do not \
imply otherwise, and do not treat a CA's own incident report as evidence \
that subscriber certificates are at risk.

On a NO-ACTION event: say plainly and EARLY that no action is required. \
Never manufacture urgency the facts do not carry. Never tell readers to \
inventory, rotate, or pre-emptively replace certificates "just in case". \
Never present an unresolved question as if it were a looming obligation. \
The honest no-action piece is the one that teaches the reader to triage the \
next event faster.

Return ONLY a JSON object with these keys:

"slug": short kebab-case topic slug, max 5 words.

"requires_reader_action": true ONLY if the event imposes a concrete action \
on the reader on a future date certain, false otherwise. The tone, the \
"What to do now" section, the pinned comment and the thumbnail all branch \
on this value, so decide it deliberately and be honest about it.

"source_check": object recording what the source-verification pass below \
found, written BEFORE you draft anything else:
  "verified": true if you successfully read at least one of the item's \
sources, false if none could be fetched.
  "sources_read": array of the URLs you actually read.
  "discrepancies": array of short strings, one per point where the item's \
description disagreed with its own sources - empty array if none. Say what \
the description claimed and what the source actually says.
  "drafted_from": "sources" if you corrected the description from what you \
read, "description" if the sources agreed with it, "description-unverified" \
if you could not read any source.

"blog_markdown": complete FixMyCert blog post in markdown. Structure: \
H1 title; hook paragraph (what just happened, why it matters); "What \
happened"; "Who is affected"; "Key dates" (bullet list; on a no-action \
event, state explicitly that none of them is a reader deadline); "What to \
do now" - numbered and actionable when requires_reader_action is true, but \
on a no-action event this section says plainly that nothing needs doing and \
spends its words on the distinction that lets the reader triage the next \
event faster, NEVER on padded busywork; short closing pointing to \
https://fixmycert.com/compliance for the live tracker. 600-900 words.

"linkedin": LinkedIn post, 120-200 words. Strong first line (no clickbait), \
short paragraphs, ends with a question or discussion prompt, then \
"Live tracker: https://fixmycert.com/compliance". 3-5 relevant hashtags.

"tweet": X post, MAX 260 characters including the link \
https://fixmycert.com/compliance. Lead with the concrete fact, not hype. \
1-2 hashtags.

"kit_email": object for a Kit email broadcast to the FixMyCert list \
(enterprise cert practitioners and kit buyers) with:
  "subject": max 60 chars, concrete and direct - the fact, not hype \
(house style: "New from FixMyCert: the 47-Day Readiness Kit").
  "preview_text": max 90 chars, adds a second concrete detail the \
subject didn't cover.
  "body_markdown": 250-400 words. Personal but efficient - written as \
Patrick from FixMyCert emailing practitioners who trust him for exactly \
this kind of alert. Structure: one-line what-happened; who is affected \
(and who can ignore it); key dates as a short list; then 3-4 concrete next \
steps when requires_reader_action is true, or a plain statement that there \
is nothing to do and why when it is false - do not invent steps to fill \
the slot; close with the live tracker link \
https://fixmycert.com/compliance. No hard sell - this email is the \
product. Sign off "- Patrick".

"youtube": object with:
  "title": 50-65 chars, front-loaded primary keyword, pattern \
"[Primary Keyword] - [Hook/Value Prop]", title case, no clickbait.
  "notebooklm_audio_prompt": instructions for NotebookLM audio overview. \
Tone branches on requires_reader_action: when TRUE, "Two senior PKI \
engineers walking through an obligation their audience has to meet, and \
what missing it costs"; when FALSE, "Two senior PKI engineers calmly \
explaining why this one asks nothing of their audience". NEVER alarmed on a \
no-action event, and never imply the listener is at risk when they are not. \
Include 4-6 key points the discussion MUST cover, \
specific terms/dates to mention, what the listener walks away knowing, and \
these literal rules: do NOT use the phrase 'years of experience' or \
reference how long the speaker has been in the industry; keep it under 15 \
minutes; conversational, not a lecture.
  "notebooklm_visual_prompt": "BACKGROUND: Dark navy (#0f172a) - solid, no \
gradients. ACCENT COLOR: Red #ef4444 (compliance). STYLE: Clean \
iconography, minimal clutter, dark mode native. TEXT: White #ffffff \
primary, gray #94a3b8 secondary. DO NOT: busy backgrounds, cartoon \
characters, 3D effects, stock photos." plus 2-3 content-specific visual \
elements. On a no-action event, also rule out alarm iconography entirely - \
no warning triangles, no pulsing or flashing "unresolved" markers, no \
distress glows - and keep the elements explanatory.
  "description": YouTube description: 2-3 sentence summary; "🔑 Key \
Points:" 3-5 bullets; "📚 Full written guide:" with \
https://fixmycert.com/compliance (placeholder until a dedicated guide \
exists); "🔗 More PKI education: https://fixmycert.com"; hashtag line \
"#SSL #TLS #Certificates #PKI #CyberSecurity #DevOps #SRE" plus 2-4 \
topic tags.
  "pinned_comment": branches on requires_reader_action. When TRUE: \
timeline/countdown format starting "📌 Key deadlines from this video:", \
🔴/🟡/🟢 date lines. When FALSE: start "📌 The short version: this one \
needs nothing from you." and give a plain dated timeline with NO deadline \
framing and no 🔴 lines, stating outright that the story carries no reader \
deadline. Never write "Key deadlines" over dates that are not deadlines. \
Either way, ends "Full compliance tracker: \
https://fixmycert.com/compliance". Scannable in 10 seconds, no subscribe \
CTAs.
  "thumbnail_prompt": "BACKGROUND: Solid dark navy (#0f172a). LEFT 60%: \
bold white text 2 lines max (line 1 primary keyword, line 2 hook 3-4 \
words), heavy sans-serif. RIGHT 40%: single icon for the topic. NO faces, \
busy backgrounds, gradients, small text, more than 2 colors + white." with \
the 2 text lines filled in for this event. Add a red #ef4444 glow and a \
small warning indicator to the icon ONLY when requires_reader_action is \
true. On a no-action event use a plain icon with no glow and no warning \
indicator, and make line 2 state the settled answer rather than pose a \
scare question - no question mark anywhere on a no-action thumbnail.
"""


def _extract_json(text: str) -> dict:
    """Parse the model's JSON reply.

    strict=False tolerates literal newlines/control characters inside
    strings — the most common defect in long model-emitted JSON (the blog
    body alone is 600-900 words). Raises ValueError/JSONDecodeError on
    anything worse; generate() treats that as retryable.
    """
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON object in model response")
    try:
        return json.loads(m.group(), strict=False)
    except json.JSONDecodeError:
        pass
    # The greedy span above runs first-brace-to-last-brace, which breaks the
    # moment the model emits prose containing braces of its own BEFORE the
    # package. That became a live risk on 2026-08-20 when generate() gained
    # web_search for source verification: search summaries and citations
    # routinely carry braces. Fall back to scanning for balanced candidates
    # and take the last one that parses — the package is emitted last.
    # Walk TOP-LEVEL objects only, skipping past each one once closed. Scanning
    # every "{" instead would keep matching the package's own nested objects
    # and return the innermost one.
    best, i, n = None, 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc, end = 0, False, False, None
        for j in range(i, n):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            break  # unbalanced tail; nothing further can close
        try:
            best = json.loads(text[i:end + 1], strict=False)
        except json.JSONDecodeError:
            pass
        i = end + 1
    if best is None:
        raise ValueError("no parseable JSON object in model response")
    return best


def _channel_instruction(channels: list[str]) -> str:
    """Tell the model which channels to produce. Requested explicitly on every
    call — there is no all-five default anywhere in this path."""
    wanted = ", ".join(f"`{CHANNELS[c][1]}`" for c in channels)
    skipped = sorted({k for _, k in CHANNELS.values()}
                     - {CHANNELS[c][1] for c in channels})
    lines = [
        "## Channels requested\n",
        f"Draft ONLY these channels: {wanted}.\n",
        'Always include "slug", "requires_reader_action" and "source_check" - '
        "they are not channels, they are the record of what you decided and "
        "what you verified.\n",
    ]
    if skipped:
        lines.append(
            "OMIT these keys from the JSON entirely - do not draft them, do "
            "not emit them empty, do not apologise for their absence: "
            + ", ".join(f"`{k}`" for k in skipped) + ".\n")
    return "".join(lines) + "\n"


def generate(entry: dict, summary: str, channels: list[str],
             max_retries: int = 3) -> dict:
    """One API call -> the requested channels as a dict."""
    if not car.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    prompt = (
        f"## Urgent item ({entry['kind']})\n\n"
        f"{json.dumps(entry['item'], indent=2)}\n\n"
        f"## Research-run summary (context)\n\n{summary or '(none)'}\n\n"
        + _channel_instruction(channels) +
        "## Source verification - DO THIS FIRST\n\n"
        "Before drafting anything, USE web_fetch TO READ THE FULL TEXT of "
        "the item's own sources: `primary_url` first, then each entry in "
        "`provenance_urls`. Fetch them - do not rely on web_search snippets, "
        "which show only the top of a page and routinely miss a decisive "
        "reply partway down a thread. Use web_search only to find material "
        "the item does not link, such as a follow-up thread. That "
        "description was written by an earlier automated research pass, "
        "possibly days ago. THE SOURCES ARE AUTHORITATIVE AND THE "
        "DESCRIPTION IS NOT.\n\n"
        "Confirm in particular:\n"
        "1. Any claim that something is unresolved, disputed, still being "
        "debated, or awaiting a decision. THESE GO STALE FASTEST AND ARE THE "
        "MAIN REASON THIS STEP EXISTS. Read the whole thread, not its first "
        "post: the answer is usually a later reply, often from the "
        "organisation that filed the report, and often in a follow-up thread "
        "at a DIFFERENT URL from the original announcement. If you find such "
        "an answer, THE ANSWER IS THE STORY - lead with it, and do not "
        "describe as open a question that has been closed.\n"
        "2. Every date.\n"
        "3. Every bug, ballot, incident, or version number - a wrong one "
        "sends readers to an unrelated incident.\n\n"
        "If a source contradicts the description or has moved past it, "
        "DRAFT FROM THE SOURCE and record the gap in `source_check."
        "discrepancies`. Do not repeat a claim you could not confirm; if a "
        "question the description calls open has since been answered, the "
        "answer is the story.\n\n"
        "Then draft the content package. Return ONLY the JSON object, with "
        "no preamble and no summary of your searches."
    )
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
                response = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": car.ANTHROPIC_API_KEY,
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": 16000,
                        "system": CONTENT_SYSTEM_PROMPT,
                        # Source verification (2026-08-20). Without this the
                        # drafter was a closed-world text expander over
                        # whatever the research run wrote: it held
                        # primary_url and provenance_urls in the item dict
                        # and never opened them, so any staleness upstream
                        # became five channels of confident publish-ready
                        # copy. That is exactly how the 2026-08-20 Let's
                        # Encrypt package shipped "the community is debating
                        # whether this forces mandated revocation" two days
                        # after the CA had answered that question on the
                        # record.
                        #
                        # BOTH tools, and web_fetch is the one that matters.
                        # web_search alone returns SNIPPETS: the 2026-08-20
                        # drill read all three source URLs and still reported
                        # the revocation question as open, because the answer
                        # sits in a reply partway down a long forum thread
                        # that no snippet surfaced. web_fetch pulls the actual
                        # page text. It only fetches URLs already present in
                        # the conversation, which is exactly the case here --
                        # primary_url and provenance_urls are in the prompt.
                        "tools": [
                            {"type": "web_fetch_20260209", "name": "web_fetch",
                             "max_uses": 8},
                            {"type": "web_search_20260209",
                             "name": "web_search"},
                        ],
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
        except httpx.TimeoutException:
            log(f"  timed out, retrying (attempt {attempt+1}/{max_retries})")
            continue
        if response.status_code == 429:
            wait = 30 * (attempt + 1)
            log(f"  rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        text = "\n".join(
            block["text"] for block in response.json()["content"]
            if block["type"] == "text"
        )
        try:
            return _extract_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            log(f"  unparseable JSON from model ({e}), retrying "
                f"(attempt {attempt+1}/{max_retries})")
            continue
    raise RuntimeError(f"no valid response after {max_retries} attempts "
                       "(rate limit, timeout, or unparseable JSON)")


# ---------------------------------------------------------------------------
# Output + git
# ---------------------------------------------------------------------------


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:60] or "urgent-item"


def write_drafts(base_dir: Path, date_str: str, drafts: dict, entry: dict,
                 channels: list[str]) -> Path:
    """Write ONLY the requested channels. A channel nobody asked for produces
    no file — the point of the 2026-08-20 change is that unwanted copy is not
    generated, so it must not be written either."""
    slug = slugify(drafts.get("slug") or entry["item"].get("title")
                   or entry["item"].get("id") or "urgent-item")
    out_dir = base_dir / f"{date_str}-{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "youtube" in channels:
        yt = drafts.get("youtube") or {}
        yt_md = "\n\n".join(
            f"## {label}\n\n{yt.get(key, '(missing)')}"
            for label, key in (
                ("Title", "title"),
                ("NotebookLM audio prompt", "notebooklm_audio_prompt"),
                ("NotebookLM visual prompt", "notebooklm_visual_prompt"),
                ("Description", "description"),
                ("Pinned comment", "pinned_comment"),
                ("Thumbnail prompt", "thumbnail_prompt"),
            )
        )
        (out_dir / "youtube.md").write_text(
            f"# YouTube publish package - {yt.get('title', slug)}\n\n{yt_md}\n")
    if "kit" in channels:
        kit = drafts.get("kit_email") or {}
        kit_md = (
            "<!-- OPERATOR NOTES - do not send as-is:\n"
            "     audience: FixMyCert brand ONLY (brand:fmc, tag 19302998) - NOT the full list\n"
            "     sender:   Patrick from FixMyCert <patrick@fixmycert.com> (account default)\n"
            "     This is a DRAFT. Create in Kit as a draft broadcast; never auto-send. -->\n\n"
            f"## Subject\n\n{kit.get('subject', '(missing)')}\n\n"
            f"## Preview text\n\n{kit.get('preview_text', '(missing)')}\n\n"
            f"## Body\n\n{kit.get('body_markdown', '(missing)')}"
        )
        (out_dir / "kit_broadcast.md").write_text(kit_md + "\n")
    for name in ("blog", "linkedin", "tweet"):
        if name in channels:
            filename, key = CHANNELS[name]
            (out_dir / filename).write_text(drafts.get(key, "(missing)") + "\n")
    # source_check and requires_reader_action are persisted so a drafter that
    # caught a stale or wrong source_item leaves a visible record instead of
    # silently writing better copy. A non-empty `discrepancies` means the
    # description that came out of the research run disagreed with its own
    # sources — worth reading before publishing, and worth noticing if it
    # keeps happening. Both degrade to a marker when the model omits them.
    (out_dir / "meta.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": entry["kind"],
        "signature": entry["sig"],
        "source_item": entry["item"],
        "channels": channels,
        "requires_reader_action": drafts.get("requires_reader_action"),
        "source_check": drafts.get("source_check") or {"verified": None},
        "status": "draft - not published",
    }, indent=2))
    return out_dir


def commit_push(dirs: list[Path]) -> str | None:
    """Commit and push draft dirs. Returns an error string or None."""
    rels = [str(d.relative_to(REPO)) for d in dirs]
    r = car_run(["git", "push", "--dry-run", "origin", "main"])
    if r.returncode != 0:
        return f"git push unavailable: {(r.stderr or '').strip().splitlines()[-1] if r.stderr else 'unknown'}"
    names = ", ".join(d.name for d in dirs)
    msg = (f"Content drafts: {names}\n\n"
           "Generated by content_drafts.py for urgent-tagged items. "
           "Drafts only - review before publishing anywhere.")
    for cmd in (["git", "add", *rels],
                ["git", "commit", "-m", msg],
                ["git", "push", "origin", "main"]):
        r = car_run(cmd)
        if r.returncode != 0:
            return f"{' '.join(cmd[:2])} failed: {(r.stderr or r.stdout)[-500:]}"
    return None


def car_run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_channels(raw: str) -> list[str]:
    """"blog,tweet" or "all" -> ordered channel list. Raises on anything else."""
    names = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if names == ["all"]:
        return list(CHANNELS)
    unknown = [c for c in names if c not in CHANNELS]
    if unknown:
        raise ValueError(f"unknown channel(s): {', '.join(unknown)}. "
                         f"Valid: {', '.join(CHANNELS)}, or 'all'")
    if not names:
        raise ValueError("no channels given")
    return [c for c in CHANNELS if c in names]


def _load_pending(args) -> tuple[dict | None, Path]:
    pending_path = (Path(args.pending_file) if args.pending_file
                    else DATA_DIR / f"pending_updates_{args.date}.json")
    if not pending_path.exists():
        return None, pending_path
    try:
        return json.loads(pending_path.read_text()), pending_path
    except Exception as e:
        log(f"content_drafts: pending file unreadable ({e})")
        return None, pending_path


def run_notify(args, pending: dict) -> int:
    """The cron path: notify, record, draft nothing."""
    urgent = collect_urgent(pending)
    notified = load_notified()
    seen = set(notified.get("signatures", {}))
    notices = [build_notice(e, args.date, is_new=e["sig"] not in seen)
               for e in urgent]
    fresh = [e for e in urgent if e["sig"] not in seen]

    if not args.dry_run:
        write_notice_file(args.date, notices)

    if not urgent:
        log(f"content_drafts: no urgent items for {args.date} - "
            "nothing to notify, nothing drafted")
        return 0

    if fresh and not args.dry_run:
        notify_urgent(fresh)
        for e in fresh:
            notified.setdefault("signatures", {})[e["sig"]] = args.date
        save_notified(notified)

    for n in notices:
        log(f"content_drafts: URGENT {'(new)' if n['is_new'] else '(already notified)'} "
            f"[{n['kind']}] {n['title'][:80]!r} selector={n['selector']} "
            f"sources={len(n['source_urls'])}")
    log(f"content_drafts: notified {len(fresh)} new of {len(urgent)} urgent "
        f"item(s) for {args.date}; NOTHING DRAFTED - drafting is an explicit "
        "trigger (--draft --channels ...)")
    return 0


def run_draft(args, pending: dict, channels: list[str]) -> int:
    """The explicit path: draft the requested channels for the selected items."""
    # skip_drafted=False: an explicit human request must not silently no-op
    # because the same item was drafted once before.
    urgent = collect_urgent(pending, skip_drafted=False)
    if args.item:
        needle = args.item.lower()
        urgent = [e for e in urgent
                  if needle in _selector(e).lower()
                  or needle in str(e["item"].get("title") or "").lower()
                  or needle in e["sig"].lower()]
        if not urgent:
            log(f"content_drafts: --item {args.item!r} matched no urgent item "
                f"in {args.date}; run --list to see the selectors")
            return 1
    if not urgent:
        log(f"content_drafts: no urgent items for {args.date} - nothing to draft")
        return 1

    already = set(load_state().get("signatures", {}))
    for e in urgent:
        if e["sig"] in already:
            log(f"content_drafts: NOTE - {_selector(e)} was drafted before; "
                "re-drafting on explicit request (files will be overwritten)")

    if len(urgent) > MAX_DRAFTS_PER_RUN:
        log(f"content_drafts: {len(urgent)} urgent items, capping at "
            f"{MAX_DRAFTS_PER_RUN}; re-run with --item to pick the rest")
        urgent = urgent[:MAX_DRAFTS_PER_RUN]

    base_dir = (DATA_DIR / "drafts_preview" if args.dry_run
                else REPO / DRAFTS_DIRNAME)
    summary = str(pending.get("summary", ""))[:2000]
    state = load_state()
    written: list[Path] = []
    for entry in urgent:
        title = (entry["item"].get("title") or entry["item"].get("id")
                 or "(untitled)")[:80]
        log(f"content_drafts: drafting [{','.join(channels)}] for "
            f"[{entry['kind']}] {title}")
        try:
            drafts = generate(entry, summary, channels)
        except Exception as e:
            log(f"content_drafts: GENERATION FAILED for {title} - {e}")
            continue
        out_dir = write_drafts(base_dir, args.date, drafts, entry, channels)
        written.append(out_dir)
        state.setdefault("signatures", {})[entry["sig"]] = args.date
        sc = drafts.get("source_check") or {}
        log(f"content_drafts: wrote {out_dir} "
            f"(requires_reader_action={drafts.get('requires_reader_action')}, "
            f"source_check.verified={sc.get('verified')}, "
            f"discrepancies={len(sc.get('discrepancies') or [])})")

    if not written:
        log("content_drafts: all generations failed - nothing written")
        return 1

    if args.dry_run:
        log(f"content_drafts DRY RUN: {len(written)} package(s) in {base_dir}; "
            "no git, no state update")
        return 0

    err = commit_push(written)
    if err:
        # Drafts stay on disk either way; the email links the directory name.
        log(f"content_drafts: PUSH FAILED ({err}) - drafts remain local on droplet")
    else:
        log(f"content_drafts: {len(written)} package(s) committed and pushed")
    save_state(state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--draft", action="store_true",
                        help="Actually draft content (default is notify only)")
    parser.add_argument("--channels", default=None,
                        help="Comma-separated: blog,linkedin,tweet,kit,youtube "
                             "or 'all'. REQUIRED with --draft; no default.")
    parser.add_argument("--item", default=None,
                        help="Draft only the urgent item whose id/signature/title "
                             "contains this string (see --list)")
    parser.add_argument("--list", action="store_true",
                        help="List today's urgent items and their selectors, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write to DATA_DIR/drafts_preview; no git, no state update")
    parser.add_argument("--pending-file", default=None,
                        help="Alternate pending-updates JSON (fire drills)")
    args = parser.parse_args()

    if args.channels and not args.draft:
        parser.error("--channels only applies to --draft")

    pending, pending_path = _load_pending(args)
    if pending is None:
        # Still an end-of-chain log line: daily_email.py polls this file.
        log(f"content_drafts: no usable pending file at {pending_path} "
            f"for {args.date} - nothing to notify or draft")
        return 0

    if args.list:
        urgent = collect_urgent(pending, skip_drafted=False)
        if not urgent:
            log(f"content_drafts: no urgent items for {args.date}")
            return 0
        for e in urgent:
            n = build_notice(e, args.date, is_new=False)
            print(f"{n['selector']}\t[{n['kind']}]\t{n['title'][:70]}\t"
                  f"{len(n['source_urls'])} source(s)")
        return 0

    if not args.draft:
        return run_notify(args, pending)

    # Drafting is explicit AND channel-scoped. Refusing to default here is the
    # fix: assuming all five is what produced four discarded channels on
    # 2026-08-20.
    if not args.channels:
        parser.error("--draft requires --channels (e.g. --channels blog,tweet, "
                     "or --channels all). There is deliberately no default.")
    try:
        channels = parse_channels(args.channels)
    except ValueError as e:
        parser.error(str(e))
    return run_draft(args, pending, channels)


if __name__ == "__main__":
    sys.exit(main())
