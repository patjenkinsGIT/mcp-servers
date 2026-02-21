# FixMyCert Content Manager MCP Server

Replaces the fixmycerttracker spreadsheet completely. Single source of truth for all FixMyCert content tracking.

## Setup

### 1. Install dependencies
```bash
pip install mcp pydantic openpyxl
```

### 2. Place the server file
```bash
# Copy to a permanent location
cp fixmycert_content_mcp.py ~/fixmycert_content_mcp.py
```

### 3. Add to Claude Desktop config

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) or equivalent:

```json
{
  "mcpServers": {
    "fixmycert_ct": {
      "command": "python3",
      "args": ["/Users/YOUR_USERNAME/fixmycert_content_mcp.py"]
    }
  }
}
```

### 4. Seed from spreadsheet (first time only)

Place your `fixmycerttracker_*.xlsx` file in your home directory or Downloads, then tell Claude:

> "Seed the content tracker from the spreadsheet"

Claude will call `ct_seed_from_xlsx` which finds the latest xlsx file and imports everything.

### 5. Retire the spreadsheet 🎉

The spreadsheet is now deprecated. All tracking happens through Claude.

## Tools (27 total)

### Content Registry (8 tools)
- `ct_register_content` — Add new content
- `ct_get_content` — Get details by ID/path/title
- `ct_list_content` — List with filters (type, category, status, has_video)
- `ct_search_content` — Full-text search
- `ct_update_content` — Update any field
- `ct_content_stats` — Summary statistics
- `ct_get_uncovered` — Content without YouTube videos
- `ct_bulk_register` — (redirects to seed)

### Backlog (5 tools)
- `ct_add_backlog` — Add backlog item
- `ct_list_backlog` — List/filter backlog
- `ct_update_backlog` — Update item fields
- `ct_complete_backlog` — Mark done + optionally register content
- `ct_next_up` — What to build next

### Keywords (4 tools)
- `ct_track_keyword` — Add/update keyword metrics
- `ct_list_keywords` — List tracked keywords
- `ct_keyword_wins` — Show winning keywords
- `ct_keyword_snapshot` — Bulk update from GSC

### SEO (2 tools)
- `ct_log_seo_change` — Record an optimization
- `ct_list_seo_changes` — Review change log

### Index Submissions (2 tools)
- `ct_log_index_submission` — Record submission
- `ct_list_index_submissions` — Check status

### Partnerships (3 tools)
- `ct_add_partner` — Add lead
- `ct_list_partners` — List all
- `ct_update_partner` — Update status/notes

### Dashboard & Seed (3 tools)
- `ct_dashboard` — Weekly check-in overview
- `ct_seed_data` — Seed with hardcoded data
- `ct_seed_from_xlsx` — Import from spreadsheet (preferred)

## Storage

```
~/.fixmycert-content/
├── content.json
├── backlog.json
├── keywords.json
├── seo-log.json
├── index-submissions.json
├── partnerships.json
└── backup/
```

Auto-backup before every write. Last 10 backups kept per file.

## Pairing with YT MCP

Both MCPs use content paths as the join key:
- Content tracker stores `youtube_url` on each content piece
- YT tracker stores `guide_url` on each video

Claude bridges them conversationally — no direct coupling between servers.
