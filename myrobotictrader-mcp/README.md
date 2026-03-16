# MyRoboticTrader MCP Server

Automated content generation tools powered by crypto news, Discord monitoring, CoinMarketCap data, and live trading performance from Google Sheets.

## Setup

### 1. Install dependencies
```bash
cd myrobotictrader-mcp
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in this directory:
```env
NEWS_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
GOOGLE_SHEETS_CREDS=path-to-creds.json
TRADING_SHEET_ID=your-sheet-id
COINMARKETCAP_API_KEY=your-key
DISCORD_BOT_TOKEN=your-token
```

### 3. Start via Docker Compose (recommended)
```bash
docker compose up -d --build myrobotictrader
```

## Available Tools (17)

### Topic Management
| Tool | Description |
|------|-------------|
| `add_topic` | Add a topic to your watchlist for content generation |
| `remove_topic` | Remove a topic from your watchlist |
| `list_topics` | List tracked topics, filterable by category or priority |
| `search_topic_now` | Search current news about a specific topic |
| `fetch_crypto_news` | Fetch latest crypto/financial/trading news |

### Discord Integration
| Tool | Description |
|------|-------------|
| `discord_list_servers` | List all Discord servers the bot is in |
| `discord_list_channels` | List channels in a Discord server |
| `discord_read_messages` | Read recent messages from a channel |
| `discord_search_keywords` | Search channel messages for specific keywords |
| `discord_monitor_announcements` | Auto-search for TGE, airdrop, launch, mainnet keywords |

### Cryptocurrency Market Data
| Tool | Description |
|------|-------------|
| `get_crypto_price` | Current price and market data for a cryptocurrency |
| `get_top_cryptos` | Top cryptocurrencies by market cap |
| `get_market_movers` | Top gainers and losers |
| `get_fear_greed_index` | Crypto Fear & Greed Index |
| `get_altcoin_season_index` | Altcoin Season Index (0-100) |

### Trading Data & Content Generation
| Tool | Description |
|------|-------------|
| `fetch_trading_data` | Pull live trading metrics from Google Sheets |
| `generate_content` | Generate blog posts, tweets, or social content from news + trading data |
