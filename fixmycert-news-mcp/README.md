# fixmycert-news-mcp

News desk manager for the fixmycert.com `/news` page. A **thin client** over the
live news API in the FixMyCert Repl — no local database, every tool is an HTTP
call against production.

## Tools (12)

| Tool | Kind | Purpose |
|------|------|---------|
| `news_dashboard` | read | Counts: total, first-party, published, drafts, priority, by cluster, last aggregator fetch, uncovered deadlines |
| `news_list` | read | Admin feed (drafts included) with filters: status, category, newsCluster, isFirstParty, isPriority, source, since; paginated (limit/offset) |
| `news_get` | read | One item by id, all fields |
| `news_get_uncovered` | read | Compliance Hub deadlines with no first-party entry (deadlineId match, else fuzzy title) |
| `news_add` | write | Insert a first-party item — **defaults to draft**; `newsCluster` required |
| `news_update` | write | Patch fields on an item |
| `news_publish` | write | `status=published` → item goes live on public `/api/news` |
| `news_tag_opportunity` | write | Set `opportunityTags[]` (replaces existing) |
| `news_archive` | write (destructive) | `status=archived` — removes from public feed; no hard delete exists |
| `news_generate_digest` | read | Period digest grouped by cluster with Hub deadline context: `{period, entries[], markdown}` |
| `news_trigger_fetch` | write | Trigger RSS aggregator refresh (`/api/cron/fetch-news`, uses `NEWS_CRON_SECRET`) |
| `news_list_sources` | read | Aggregator RSS source list |

## Editorial conventions

- New items stay **draft** until `news_publish` — nothing is public before that.
- `newsCluster` is required on add: `ca-action`, `tls-validity`, `pqc`,
  `code-signing`, `smime`, `client-auth`, `other`.
- `isPriority` floats an item to the top of `/news` — reserve for high-impact changes.
- First-party items should carry a primary-source `url`, and a `symptomString`
  when the change causes a visible failure (e.g. `NET::ERR_CERT_AUTHORITY_INVALID`).
- `internalUrl` is the FixMyCert-internal link shown alongside the item — a
  site-relative path like `/guides/47-day-certificate-timeline`. Settable via
  `news_add` / `news_update`.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEWS_API_BASE_URL` | yes | `https://fixmycert.com` — the server exits at startup if missing |
| `NEWS_ADMIN_SECRET` | yes | Dedicated admin secret; must equal the value in the FixMyCert Replit Secrets. Server exits at startup if missing |
| `NEWS_CRON_SECRET` | no | Only for `news_trigger_fetch`; equals `CRON_SECRET` in Replit Secrets (separate from the admin secret) |

Auth (confirmed against the Repl's `checkNewsKey` helper): the three admin
routes accept `?key=` **or** `Authorization: Bearer` against
`NEWS_ADMIN_SECRET` — this client uses the Bearer header so the secret never
appears in URLs or logs. `/api/cron/fetch-news` is query-only (`?key=`)
against `CRON_SECRET`; httpx request logging is silenced so that URL stays
out of container logs.

## Backend behavior notes (learned the hard way, 2026-07-22)

- **Unique-URL constraint**: the backend rejects (500) any item whose `url` duplicates an
  existing item's. When several items share a primary source (e.g. the CNSA 2.0 FAQ PDF,
  Chrome root policy page), append a distinguishing `#fragment`.
- **`opportunityTags` is an enum**: `content-guide | video | consultant-digest | product | partnership | none`.
  Anything else is a 400.
- **`publishedAt` is stamped at draft creation and NOT updated by publishing.** If a draft
  sits before going live, set `publishedAt` via `news_update` at publish time or the feed
  shows the creation date.
- **Retention**: aggregated RSS items (`isFirstParty=false`) are auto-pruned 30 days after
  `publishedAt` by the cron's cleanup. First-party items are never auto-pruned — they stay
  until manually archived.
- **Public feed**: max 50 items/request (default 20), offset pagination, no total cap.
  `category` is the only public filter — `isFirstParty` filtering is admin-only.

## Editorial state (as of 2026-07-22)

All Compliance Hub deadlines have first-party coverage (54 items, `news_get_uncovered` = 0).
Rollout was tranched to avoid a bot-like publish burst: tranche 1 (9 near-term 2026 items)
published 2026-07-22; tranches 2 (8), 3a (15), 3b (22) auto-publish via one-time cloud
routines on Jul 24 / 27 / 30, each stamping `publishedAt` to its run date.

## Automation

Publishing is done — all 54 approved first-party drafts went out by 2026-07-27 (tranche 1
locally, 2 locally after the cloud run published nothing, 3a+3b locally over Tailscale).
The three one-time publish routines and their local backstop tasks are spent.

The one standing job is the **weekly gap-check**, `news_gap_check.py`, a droplet cron:

```
0 15 * * 1  # Mondays 15:00 UTC — see the script's docstring for the full crontab line
```

Read-only report, emailed via Resend: Hub deadlines with no first-party coverage, deadlines
covered only by an unpublished draft, and feed health (item counts + a stall warning if the
aggregator's `updatedAt` goes >48h without moving). A fetch failure emails too, so a silent
Monday always means "checked and clean", never "never ran".

Two things about it are load-bearing:

- **It runs on the droplet, not in the cloud.** It was a claude.ai routine
  (`trig_01KtjShXBbnXayCAKLgC1LJo`) from 2026-07-22 until 2026-08-05, and never once
  succeeded: the cloud sandbox proxy denies CONNECT to fixmycert hosts (explicit 403 on
  `compliance-api.fixmycert.com`, 2026-07-23). Adding both hosts to the routine allowlist
  did not fix it. Don't move it back.
- **Coverage is judged against the admin feed, not the public one.** 17 first-party items
  are archived, 10 deliberately (the 2026-07-22 editorial pre-screen dropped general
  cyber-regulation items to keep /news PKI-core). Archived means *decided*, not missing.
  Judging against the public feed reports those 10 as fresh gaps every week — the old cloud
  routine did exactly that. This is also why the script agrees with `news_get_uncovered`.

**Firewall note**: all six MCP ports (8081–8086) have been Tailscale-only since 2026-07-23
(`/usr/local/sbin/mcp-firewall.sh`, persisted by `mcp-firewall.service`). The gap-check is
unaffected — it talks to the public HTTPS API, never to port 8086.

## Backing API

- `GET /api/news` — public feed (published only)
- `GET /api/news/admin` — authed feed incl. drafts; filters status/category/newsCluster/isFirstParty/isPriority
- `POST /api/news/items` — authed insert (auto `isFirstParty=true`)
- `PATCH /api/news/items/:id` — authed update / status flips
- `GET /api/news/sources` — aggregator sources
- `GET https://compliance-api.fixmycert.com/api/compliance-data` — Hub deadlines (for uncovered/digest)
- `GET /api/cron/fetch-news` — RSS refresh (CRON_SECRET)

## Deploy

Runs as the `fixmycert-news` service in the repo's `docker-compose.yml`
(port **8086**, shared `Dockerfile.mcp`). Set `NEWS_ADMIN_SECRET` (and
optionally `NEWS_CRON_SECRET`) in `/opt/mcp-servers/.env` on the droplet, then:

```bash
cd /opt/mcp-servers && git pull origin main && docker compose up -d --build fixmycert-news
```

Client registration (Claude Desktop / Claude Code):

```json
"fixmycert_news": {"command": "npx", "args": ["mcp-remote", "http://100.70.144.60:8086/sse", "--allow-http"]}
```

## Smoke test

1. `news_add` a draft (with `internalUrl`) → shows in `news_list` (admin) but **not** on public `/api/news`
2. `internalUrl` reads back via `news_get`, then is changed via `news_update` and reads back again
3. `news_publish` it → appears on public `/api/news`
4. `news_dashboard` and `news_get_uncovered` return sane data
5. (cleanup) `news_archive` the test item
