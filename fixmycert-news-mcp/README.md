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

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEWS_API_BASE_URL` | yes | `https://fixmycert.com` — the server exits at startup if missing |
| `NEWS_ADMIN_SECRET` | yes | Dedicated admin secret; must equal the value in the FixMyCert Replit Secrets. Server exits at startup if missing |
| `NEWS_CRON_SECRET` | no | Only for `news_trigger_fetch`; equals `CRON_SECRET` in Replit Secrets (separate from the admin secret) |

Auth is sent **both** as `?key=` query param (documented form for
`GET /api/news/admin`) and as an `x-admin-key` header (the convention used by
fixmycert.com's other admin routes), so either server-side check works.

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

1. `news_add` a draft → shows in `news_list` (admin) but **not** on public `/api/news`
2. `news_publish` it → appears on public `/api/news`
3. `news_dashboard` and `news_get_uncovered` return sane data
4. (cleanup) `news_archive` the test item
