#!/usr/bin/env python3
"""Urgent-event content draft generator.

Chained in the 10:00 UTC crontab line after auto_approve.py. When today's
research produced items tagged "urgent": true (announced mass revocation,
distrust of a trusted root/CA, or a compliance obligation landing within ~60
days), drafts a five-piece FixMyCert content package per item:

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
  python3 content_drafts.py                  # normal cron run (today)
  python3 content_drafts.py --date YYYY-MM-DD
  python3 content_drafts.py --dry-run        # generate to DATA_DIR/drafts_preview,
                                             # no git, no state update
  python3 content_drafts.py --pending-file X # read urgent items from an
                                             # alternate pending file (fire drills)

Env: ANTHROPIC_API_KEY required to draft (a no-urgent-items run exits clean
without it). PKI_REPO_PATH optional override, same default as the pipeline.
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
REPO = Path(os.environ.get("PKI_REPO_PATH", car.PKI_REPO_PATH))
DRAFTS_DIRNAME = "content_drafts"
MODEL = car.MODEL
MAX_DRAFTS_PER_RUN = 3  # an event wave should not turn into an API bill


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


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"signatures": {}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def collect_urgent(pending: dict) -> list[dict]:
    """Pull urgent-tagged items out of a pending-updates dict.

    Returns [{"kind", "sig", "item"}] minus rejected and already-drafted
    items, using the pipeline's signature machinery so "same item" means
    exactly what it means to the email backlog and dedup pass.
    """
    rejected = car.load_rejected()
    drafted = set(load_state().get("signatures", {}))
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
# Draft generation
# ---------------------------------------------------------------------------

CONTENT_SYSTEM_PROMPT = """\
You are the content writer for FixMyCert (https://fixmycert.com), a PKI \
education site. An urgent PKI ecosystem event was just detected and you are \
drafting the first-response content package. The reader is the enterprise \
certificate team - the people who own cert inventories, renewals, and \
compliance evidence - NOT certificate authorities. Practical, precise, \
calm-but-direct; explain what happened, who is affected, concrete dates, \
and what to do this week. Never invent facts, dates, or URLs beyond the \
event details provided. If the event affects only CAs, say plainly that \
enterprise teams need no action unless they operate a CA.

Return ONLY a JSON object with these keys:

"slug": short kebab-case topic slug, max 5 words.

"blog_markdown": complete FixMyCert blog post in markdown. Structure: \
H1 title; hook paragraph (what just happened, why it matters); "What \
happened"; "Who is affected"; "Key dates" (bullet list); "What to do now" \
(numbered, actionable); short closing pointing to \
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
(and who can ignore it); key dates as a short list; 3-4 concrete next \
steps; close with the live tracker link \
https://fixmycert.com/compliance. No hard sell - this email is the \
product. Sign off "- Patrick".

"youtube": object with:
  "title": 50-65 chars, front-loaded primary keyword, pattern \
"[Primary Keyword] - [Hook/Value Prop]", title case, no clickbait.
  "notebooklm_audio_prompt": instructions for NotebookLM audio overview. \
Tone: "Two senior PKI engineers genuinely alarmed that organizations don't \
know what's coming". Include 4-6 key points the discussion MUST cover, \
specific terms/dates to mention, what the listener walks away knowing, and \
these literal rules: do NOT use the phrase 'years of experience' or \
reference how long the speaker has been in the industry; keep it under 15 \
minutes; conversational, not a lecture.
  "notebooklm_visual_prompt": "BACKGROUND: Dark navy (#0f172a) - solid, no \
gradients. ACCENT COLOR: Red #ef4444 (compliance). STYLE: Clean \
iconography, minimal clutter, dark mode native. TEXT: White #ffffff \
primary, gray #94a3b8 secondary. DO NOT: busy backgrounds, cartoon \
characters, 3D effects, stock photos." plus 2-3 content-specific visual \
elements.
  "description": YouTube description: 2-3 sentence summary; "🔑 Key \
Points:" 3-5 bullets; "📚 Full written guide:" with \
https://fixmycert.com/compliance (placeholder until a dedicated guide \
exists); "🔗 More PKI education: https://fixmycert.com"; hashtag line \
"#SSL #TLS #Certificates #PKI #CyberSecurity #DevOps #SRE" plus 2-4 \
topic tags.
  "pinned_comment": timeline/countdown format - starts with "📌 Key \
deadlines from this video:", 🔴/🟡/🟢 date lines, ends "Full compliance \
tracker: https://fixmycert.com/compliance". Scannable in 10 seconds, no \
subscribe CTAs.
  "thumbnail_prompt": "BACKGROUND: Solid dark navy (#0f172a). LEFT 60%: \
bold white text 2 lines max (line 1 primary keyword, line 2 hook 3-4 \
words), heavy sans-serif. RIGHT 40%: single icon for the topic with red \
#ef4444 glow, optional small warning indicator. NO faces, busy \
backgrounds, gradients, small text, more than 2 colors + white." with the \
2 text lines filled in for this event.
"""


def generate(entry: dict, summary: str, max_retries: int = 3) -> dict:
    """One API call -> the full four-piece package as a dict."""
    if not car.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    prompt = (
        f"## Urgent item ({entry['kind']})\n\n"
        f"{json.dumps(entry['item'], indent=2)}\n\n"
        f"## Research-run summary (context)\n\n{summary or '(none)'}\n\n"
        "Draft the content package. Return ONLY the JSON object."
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
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            raise RuntimeError("no JSON object in model response")
        return json.loads(json_match.group())
    raise RuntimeError(f"rate limited or timed out after {max_retries} retries")


# ---------------------------------------------------------------------------
# Output + git
# ---------------------------------------------------------------------------


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:60] or "urgent-item"


def write_drafts(base_dir: Path, date_str: str, drafts: dict, entry: dict) -> Path:
    slug = slugify(drafts.get("slug") or entry["item"].get("title")
                   or entry["item"].get("id") or "urgent-item")
    out_dir = base_dir / f"{date_str}-{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    (out_dir / "blog.md").write_text(drafts.get("blog_markdown", "(missing)") + "\n")
    (out_dir / "linkedin.md").write_text(drafts.get("linkedin", "(missing)") + "\n")
    (out_dir / "tweet.md").write_text(drafts.get("tweet", "(missing)") + "\n")
    (out_dir / "kit_broadcast.md").write_text(kit_md + "\n")
    (out_dir / "youtube.md").write_text(
        f"# YouTube publish package - {yt.get('title', slug)}\n\n{yt_md}\n")
    (out_dir / "meta.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": entry["kind"],
        "signature": entry["sig"],
        "source_item": entry["item"],
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Write to DATA_DIR/drafts_preview; no git, no state update")
    parser.add_argument("--pending-file", default=None,
                        help="Alternate pending-updates JSON (fire drills)")
    args = parser.parse_args()

    pending_path = (Path(args.pending_file) if args.pending_file
                    else DATA_DIR / f"pending_updates_{args.date}.json")
    if not pending_path.exists():
        log(f"content_drafts: no pending file for {args.date} - nothing to draft")
        return 0
    try:
        pending = json.loads(pending_path.read_text())
    except Exception as e:
        log(f"content_drafts: pending file unreadable ({e}) - nothing to draft")
        return 0

    urgent = collect_urgent(pending)
    if not urgent:
        log(f"content_drafts: no new urgent items for {args.date} - nothing to draft")
        return 0

    if len(urgent) > MAX_DRAFTS_PER_RUN:
        log(f"content_drafts: {len(urgent)} urgent items, capping at "
            f"{MAX_DRAFTS_PER_RUN}; the rest draft on later runs")
        urgent = urgent[:MAX_DRAFTS_PER_RUN]

    base_dir = (DATA_DIR / "drafts_preview" if args.dry_run
                else REPO / DRAFTS_DIRNAME)
    summary = str(pending.get("summary", ""))[:2000]
    state = load_state()
    written: list[Path] = []
    for entry in urgent:
        title = (entry["item"].get("title") or entry["item"].get("id")
                 or "(untitled)")[:80]
        log(f"content_drafts: drafting package for [{entry['kind']}] {title}")
        try:
            drafts = generate(entry, summary)
        except Exception as e:
            log(f"content_drafts: GENERATION FAILED for {title} - {e}")
            continue
        out_dir = write_drafts(base_dir, args.date, drafts, entry)
        written.append(out_dir)
        state.setdefault("signatures", {})[entry["sig"]] = args.date
        log(f"content_drafts: wrote {out_dir}")

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


if __name__ == "__main__":
    sys.exit(main())
