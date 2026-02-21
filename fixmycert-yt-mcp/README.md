# FixMyCert YouTube Channel Manager — MCP Server

A local MCP server that tracks videos, cross-references, placeholder debt, update queues, and content coverage gaps for the FixMyCert YouTube channel.

## Setup

### 1. Install dependencies

```bash
cd fixmycert_yt_mcp
pip install -r requirements.txt
```

### 2. Set your YouTube API key

```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc)
export YOUTUBE_API_KEY="your-api-key-here"
```

The API key enables live sync with YouTube (view counts, description verification, auto-discovery of new videos). The server works without it — API features will just return an error message.

### 3. Add to Claude Desktop config

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) or the equivalent config file:

```json
{
  "mcpServers": {
    "fixmycert_yt": {
      "command": "python",
      "args": ["/full/path/to/fixmycert_yt_mcp/server.py"],
      "env": {
        "YOUTUBE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### 4. Restart Claude Desktop

### 5. Seed the database

On first use, tell Claude:
> "Run yt_seed_data to initialize the YouTube tracker"

This loads all 17 known videos with metadata.

## Available Tools

### Registry & Lookup
| Tool | Description |
|------|-------------|
| `yt_register_video` | Add a new video after publishing |
| `yt_get_video` | Get full details for a specific video |
| `yt_list_videos` | List all videos (filter by category/tag) |
| `yt_update_video` | Update any field on a video |
| `yt_search` | Full-text search across all fields |

### Cross-Links & Placeholders
| Tool | Description |
|------|-------------|
| `yt_get_placeholders` | Show all unfilled related video links |
| `yt_suggest_crosslinks` | Suggest which videos should reference each other |
| `yt_resolve_placeholder` | Replace a placeholder with a real URL |
| `yt_get_crosslink_map` | Visualize the relationship graph |

### Update Queue
| Tool | Description |
|------|-------------|
| `yt_flag_update_needed` | Flag videos needing description updates |
| `yt_get_update_queue` | Show all pending updates by priority |
| `yt_resolve_update` | Mark an update as done |

### Content Coverage
| Tool | Description |
|------|-------------|
| `yt_register_content` | Register content that could become a video |
| `yt_get_coverage_gaps` | Show content without videos |

### A/B Testing
| Tool | Description |
|------|-------------|
| `yt_start_ab_test` | Record a new A/B test |
| `yt_end_ab_test` | Record test results |
| `yt_get_ab_tests` | Show active and completed tests |

### YouTube API Sync (requires YOUTUBE_API_KEY)
| Tool | Description |
|------|-------------|
| `yt_sync_from_youtube` | Pull all video data from YouTube — discovers new videos, updates stats |
| `yt_refresh_video` | Refresh a single video's data (views, likes, description) |
| `yt_check_descriptions` | Compare local descriptions vs. what's live on YouTube |

### Dashboard
| Tool | Description |
|------|-------------|
| `yt_dashboard` | Full channel overview with action items |
| `yt_seed_data` | Initialize with all 17 known videos (first run only) |

## Data Storage

All data lives in `~/.fixmycert-yt/`:
```
~/.fixmycert-yt/
├── videos.json          # All video records
├── content-map.json     # Content coverage tracking
├── update-queue.json    # Flagged updates
└── backup/              # Auto-backups before every write
```

## Typical Workflows

### After publishing a new video:
1. "I just published [title] at [URL] for [guide]"
2. Claude registers it → checks placeholders → suggests cross-links
3. You update existing video descriptions in YouTube Studio

### When news breaks:
1. "Root Causes just covered a 47-day update — flag affected videos"
2. Claude flags all compliance-tagged videos
3. Later: "What's in my update queue?"

### Weekly check-in:
1. "Give me the YouTube dashboard"
2. See placeholder debt, pending updates, coverage gaps, A/B tests
