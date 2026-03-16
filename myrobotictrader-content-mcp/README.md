# MyRoboticTrader Content Manager MCP Server

Single source of truth for all MyRoboticTrader content tracking — blog posts, social media, SEO, keywords, backlog, and affiliates.

## Setup

### 1. Install dependencies
```bash
cd myrobotictrader-content-mcp
pip install -r requirements.txt
```

### 2. Start via Docker Compose (recommended)
```bash
docker compose up -d --build myrobotictrader-content
```

### 3. Seed the database (first run)
Tell Claude: "Run mrt_seed_data to initialize the content tracker"

## Available Tools (25)

### Content Registry (7 tools)
| Tool | Description |
|------|-------------|
| `mrt_register_content` | Register a new content piece (blog, guide, landing page) |
| `mrt_get_content` | Get details by ID, path, or title |
| `mrt_list_content` | List content with filters (type, category, status) |
| `mrt_search_content` | Full-text search across all fields |
| `mrt_update_content` | Update any field on a content record |
| `mrt_content_stats` | Summary statistics for the content library |
| `mrt_get_uncovered` | Content without YouTube videos |

### Backlog (5 tools)
| Tool | Description |
|------|-------------|
| `mrt_add_backlog` | Add a backlog item with priority and rationale |
| `mrt_list_backlog` | List/filter backlog items |
| `mrt_update_backlog` | Update status, priority, or other fields |
| `mrt_complete_backlog` | Mark done + optionally register as content |
| `mrt_next_up` | Top backlog items to work on next |

### Keywords & SEO (6 tools)
| Tool | Description |
|------|-------------|
| `mrt_track_keyword` | Add or update keyword with GSC metrics |
| `mrt_list_keywords` | List tracked keywords, filterable by status |
| `mrt_keyword_wins` | Keywords where you're ranking well (position < 10) |
| `mrt_keyword_snapshot` | Bulk update from GSC data export |
| `mrt_log_seo_change` | Record an SEO optimization |
| `mrt_list_seo_changes` | Review SEO change log |

### Index Submissions (2 tools)
| Tool | Description |
|------|-------------|
| `mrt_log_index_submission` | Record a GSC index submission |
| `mrt_list_index_submissions` | Check submission status |

### Partnerships (3 tools)
| Tool | Description |
|------|-------------|
| `mrt_add_partner` | Add an affiliate or partnership lead |
| `mrt_list_partners` | List all leads |
| `mrt_update_partner` | Update status, next action, or notes |

### Dashboard & Admin (2 tools)
| Tool | Description |
|------|-------------|
| `mrt_dashboard` | Weekly check-in overview |
| `mrt_seed_data` | Initialize empty database (first run) |

## Storage

All data lives in the `/data` volume (Docker) or `~/.myrobotictrader-content/` (local):
```
├── content.json
├── backlog.json
├── keywords.json
├── seo-log.json
├── index-submissions.json
├── partnerships.json
└── backup/
```

Auto-backup before every write. Last 10 backups kept per file.
