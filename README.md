# MCP Servers

Dockerized MCP servers running on DigitalOcean, accessible from any machine via SSE transport.

## Services

| Service | Port | Description |
|---------|------|-------------|
| fixmycert-content-mcp | 8081 | FixMyCert content tracker |
| fixmycert-yt-mcp | 8082 | FixMyCert YouTube channel manager |
| myrobotictrader-mcp | 8083 | MyRoboticTrader tools |
| myrobotictrader-content-mcp | 8084 | MyRoboticTrader content tracker |

### fixmycert-content-mcp (27 tools)
Content registry, backlog, keywords, SEO logging, index submissions, and partnerships for FixMyCert. Replaces the fixmycerttracker spreadsheet as the single source of truth.

### fixmycert-yt-mcp (29 tools)
YouTube channel manager for FixMyCert — video registry, cross-link tracking, placeholder debt, update queue, A/B testing, content coverage gaps, and YouTube API sync. Includes `yt_update_description` and `yt_push_all_crosslinks` for pushing description updates directly to YouTube.

### myrobotictrader-mcp (17 tools)
Content generation powered by crypto news feeds, Discord monitoring, CoinMarketCap market data, and live trading performance from Google Sheets.

### myrobotictrader-content-mcp (25 tools)
Content registry, backlog, keywords, SEO logging, index submissions, and partnerships for MyRoboticTrader.

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

## Deployment

1. Push changes to GitHub:
   ```bash
   git push origin main
   ```

2. SSH into the DigitalOcean droplet, pull, and rebuild:
   ```bash
   cd ~/mcp-servers
   git pull origin main
   docker compose up -d --build
   ```
