# MCP Servers

Dockerized MCP servers running on DigitalOcean, accessible from any machine via SSE transport.

## Services

| Service | Port | Description |
|---------|------|-------------|
| fixmycert-content-mcp | 8081 | FixMyCert content tracker |
| fixmycert-yt-mcp | 8082 | FixMyCert YouTube channel manager |
| myrobotictrader-mcp | 8083 | MyRoboticTrader tools |
| myrobotictrader-content-mcp | 8084 | MyRoboticTrader content tracker |
| fixmycert-news-mcp | 8086 | FixMyCert news desk (thin client over fixmycert.com news API) |
| pki-compliance-mcp | 5000 | PKI compliance deadlines API + MCP (backs the fixmycert.com Compliance Hub; systemd, not Docker) |

### fixmycert-content-mcp (27 tools)
Content registry, backlog, keywords, SEO logging, index submissions, and partnerships for FixMyCert. Replaces the fixmycerttracker spreadsheet as the single source of truth.

### fixmycert-yt-mcp (32 tools)
YouTube channel manager for FixMyCert — video registry, cross-link tracking, placeholder debt, update queue, A/B testing, content coverage gaps, and YouTube API sync. Includes `yt_update_description` (full/section mode), `yt_push_all_crosslinks`, `yt_push_pinned_comment`, and `yt_bulk_update_descriptions` for pushing updates directly to YouTube.

### myrobotictrader-mcp (17 tools)
Content generation powered by crypto news feeds, Discord monitoring, CoinMarketCap market data, and live trading performance from Google Sheets.

### myrobotictrader-content-mcp (25 tools)
Content registry, backlog, keywords, SEO logging, index submissions, and partnerships for MyRoboticTrader.

### fixmycert-news-mcp (12 tools)
News desk manager for [fixmycert.com/news](https://fixmycert.com/news) — draft/publish workflow for first-party news items, uncovered-deadline gap analysis against the Compliance Hub, period digests, opportunity tagging, and aggregator source/refresh control. Thin client over the live news API in the FixMyCert Repl; **no local database**.

### pki-compliance-mcp
PKI compliance deadline tracker backing the [FixMyCert Compliance Hub](https://fixmycert.com/compliance) — CA/Browser Forum ballots, browser root programs, NIST/NSA, DORA/NIS2/UK CSR. Includes a cost-gated daily AI research pipeline that proposes updates for human review. Runs as a systemd service (`pki-compliance-api`) behind nginx, not via docker-compose. See [pki-compliance-mcp/README.md](pki-compliance-mcp/README.md) for the full process flow and operations guide.

See each service's own README for full tool listings.

## Setup

1. Clone the repo:
   ```bash
   git clone git@github.com:YOUR_USERNAME/mcp-servers.git
   cd mcp-servers
   ```

2. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

3. Start all services:
   ```bash
   docker compose up -d --build
   ```

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `YOUTUBE_API_KEY` | fixmycert-yt-mcp | YouTube Data API key (read-only operations) |
| `YOUTUBE_OAUTH_TOKEN` | fixmycert-yt-mcp | OAuth2 refresh token (write operations) |
| `YOUTUBE_CLIENT_ID` | fixmycert-yt-mcp | Google OAuth2 client ID |
| `YOUTUBE_CLIENT_SECRET` | fixmycert-yt-mcp | Google OAuth2 client secret |
| `NEWS_API_BASE_URL` | fixmycert-news-mcp | News API base URL (`https://fixmycert.com`) |
| `NEWS_ADMIN_SECRET` | fixmycert-news-mcp | Admin secret for the news API (same value as in FixMyCert Replit Secrets) |
| `NEWS_CRON_SECRET` | fixmycert-news-mcp | Optional — RSS refresh route secret (`CRON_SECRET` in Replit Secrets) |

## Deployment

1. Push changes to GitHub:
   ```bash
   git push origin main
   ```

2. SSH into the DigitalOcean droplet, pull, and rebuild:
   ```bash
   cd /opt/mcp-servers
   git pull origin main
   docker compose up -d --build
   ```
