#!/usr/bin/env python3
"""
MyRoboticTrader Content Manager MCP Server

Single source of truth for all MyRoboticTrader content tracking.
Blog posts, social media, SEO, keywords, backlog, affiliates.

Transport: stdio (local) or sse (Docker/remote)
Storage: JSON files in ~/.myrobotictrader-content/ or /data (Docker)
"""

import json
import os
import re
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================
# Constants
# ============================================================

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / ".myrobotictrader-content")))
CONTENT_FILE = DATA_DIR / "content.json"
BACKLOG_FILE = DATA_DIR / "backlog.json"
KEYWORDS_FILE = DATA_DIR / "keywords.json"
SEO_LOG_FILE = DATA_DIR / "seo-log.json"
INDEX_SUBS_FILE = DATA_DIR / "index-submissions.json"
PARTNERSHIPS_FILE = DATA_DIR / "partnerships.json"
BACKUP_DIR = DATA_DIR / "backup"

VALID_CONTENT_TYPES = [
    "blog", "guide", "landing-page", "tool", "comparison", "case-study",
    "social-post", "email", "video", "infographic",
]

VALID_CATEGORIES = [
    "education", "anti-gambling", "market-intelligence", "lifestyle",
    "results", "trading-basics", "crypto-news", "passive-income",
    "automation", "transparency", "affiliate", "depin",
]

VALID_PRIORITIES = ["critical", "high", "medium", "low"]

VALID_BACKLOG_STATUSES = [
    "backlog", "spec-needed", "spec-complete", "ready", "in-progress", "done",
]

VALID_CONTENT_STATUSES = ["live", "draft", "planned", "scheduled"]

VALID_KEYWORD_STATUSES = [
    "winning", "optimized", "needs-time", "new-content", "declining",
]

# ============================================================
# Server
# ============================================================

mcp = FastMCP("myrobotictrader_ct")

# ============================================================
# Storage helpers
# ============================================================


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def _backup(filepath: Path) -> None:
    if filepath.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = BACKUP_DIR / f"{filepath.stem}_{ts}.json"
        shutil.copy2(filepath, dest)
        # Keep only last 10 backups per file
        prefix = filepath.stem
        backups = sorted(BACKUP_DIR.glob(f"{prefix}_*.json"))
        for old in backups[:-10]:
            old.unlink()


def _load(filepath: Path) -> list:
    _ensure_data_dir()
    if not filepath.exists():
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def _save(filepath: Path, data: list) -> None:
    _ensure_data_dir()
    _backup(filepath)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _make_id(path_or_title: str) -> str:
    """Generate a content_id from path or title."""
    text = path_or_title.strip("/").split("/")[-1] if "/" in path_or_title else path_or_title
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _match(record: dict, identifier: str, id_field: str = "content_id") -> bool:
    """Check if a record matches an identifier (ID, path, or title substring)."""
    identifier_lower = identifier.lower().strip()
    if record.get(id_field, "").lower() == identifier_lower:
        return True
    if record.get("path", "").lower() == identifier_lower:
        return True
    if record.get("path", "").lower().endswith(identifier_lower):
        return True
    if identifier_lower in record.get("title", "").lower():
        return True
    return False


def _search_fields(record: dict, query: str) -> bool:
    """Full-text search across all string fields."""
    query_lower = query.lower()
    for val in record.values():
        if isinstance(val, str) and query_lower in val.lower():
            return True
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and query_lower in item.lower():
                    return True
    return False


# ============================================================
# Response formatting
# ============================================================

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


def _fmt_content_md(item: dict) -> str:
    video = "🎬" if item.get("youtube_url") else "—"
    return (
        f"### {item['title']}\n"
        f"- **Type:** {item['content_type']} | **Category:** {item['category']}\n"
        f"- **Path:** `{item['path']}`\n"
        f"- **Video:** {video} {item.get('youtube_url', '')}\n"
        f"- **Status:** {item['status']} | **Added:** {item.get('date_added', '?')}\n"
        f"- **Tags:** {', '.join(item.get('tags', [])) or 'none'}\n"
    )


def _fmt_content_short(item: dict) -> str:
    video = "🎬" if item.get("youtube_url") else "  "
    return f"{video} [{item['content_type']:12}] [{item['category']:20}] {item['title']}"


def _fmt_backlog_md(item: dict) -> str:
    return (
        f"### {item['title']}\n"
        f"- **Priority:** {item['priority']} | **Status:** {item['status']}\n"
        f"- **Type:** {item['content_type']} | **Category:** {item['category']}\n"
        f"- **Why:** {item.get('why', '—')}\n"
    )


def _fmt_keyword_md(item: dict) -> str:
    return (
        f"**{item['keyword']}** → `{item.get('content_path', '?')}`\n"
        f"  Pos: {item.get('position', '?')} | Imp: {item.get('impressions', '?')} | "
        f"Clicks: {item.get('clicks', '?')} | CTR: {item.get('ctr', '?')} | "
        f"Status: {item.get('status', '?')}\n"
    )


# ============================================================
# Pydantic Input Models
# ============================================================


class RegisterContentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., description="Content title", min_length=1, max_length=200)
    content_type: str = Field(..., description="Type: blog, guide, landing-page, tool, comparison, case-study, social-post, email, video, infographic")
    category: str = Field(..., description="Category: education, anti-gambling, market-intelligence, lifestyle, results, trading-basics, crypto-news, passive-income, automation, transparency, affiliate, depin")
    path: str = Field(..., description="MyRoboticTrader URL path (e.g., '/blog/why-systematic-trading-wins')")
    youtube_url: Optional[str] = Field(default=None, description="YouTube video URL if one exists")
    status: Optional[str] = Field(default="live", description="Status: live, draft, planned, scheduled")
    date_added: Optional[str] = Field(default=None, description="Date added (defaults to current month)")
    tags: Optional[List[str]] = Field(default=None, description="Topic tags for cross-referencing")
    notes: Optional[List[str]] = Field(default=None, description="Any notes about this content")


class ContentIdentifierInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identifier: str = Field(..., description="Content ID, URL path, or title substring")


class ListContentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_type: Optional[str] = Field(default=None, description="Filter by type")
    category: Optional[str] = Field(default=None, description="Filter by category")
    status: Optional[str] = Field(default=None, description="Filter by status")
    has_video: Optional[bool] = Field(default=None, description="Filter by video presence")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., description="Search term", min_length=1)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class UpdateContentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identifier: str = Field(..., description="Content ID, path, or title substring")
    title: Optional[str] = Field(default=None)
    content_type: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    path: Optional[str] = Field(default=None)
    youtube_url: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    add_tags: Optional[List[str]] = Field(default=None, description="Add tags without replacing existing")
    tags: Optional[List[str]] = Field(default=None, description="Replace all tags")
    add_note: Optional[str] = Field(default=None, description="Add a note")


class AddBacklogInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., description="What to build", min_length=1, max_length=200)
    content_type: str = Field(..., description="Type: blog, guide, landing-page, tool, comparison, etc.")
    category: str = Field(..., description="Content category")
    priority: str = Field(default="medium", description="Priority: critical, high, medium, low")
    why: Optional[str] = Field(default=None, description="Why this matters / rationale")
    status: Optional[str] = Field(default="backlog", description="Status: backlog, spec-needed, spec-complete, ready, in-progress")


class ListBacklogInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    priority: Optional[str] = Field(default=None, description="Filter by priority")
    category: Optional[str] = Field(default=None, description="Filter by category")
    status: Optional[str] = Field(default=None, description="Filter by status")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class UpdateBacklogInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identifier: str = Field(..., description="Backlog item title substring or ID")
    priority: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    why: Optional[str] = Field(default=None)
    add_note: Optional[str] = Field(default=None)


class CompleteBacklogInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identifier: str = Field(..., description="Backlog item title substring or ID")
    path: Optional[str] = Field(default=None, description="If provided, also registers as content at this path")
    youtube_url: Optional[str] = Field(default=None, description="YouTube URL if video was also created")


class TrackKeywordInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keyword: str = Field(..., description="Search keyword/phrase", min_length=1)
    impressions: Optional[int] = Field(default=None, ge=0)
    clicks: Optional[int] = Field(default=None, ge=0)
    position: Optional[float] = Field(default=None)
    ctr: Optional[str] = Field(default=None, description="CTR as string (e.g., '6%')")
    status: Optional[str] = Field(default=None, description="winning, optimized, needs-time, new-content, declining")
    action: Optional[str] = Field(default=None, description="Action taken or planned")
    content_path: Optional[str] = Field(default=None, description="Path to the content this keyword targets")


class ListKeywordsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Optional[str] = Field(default=None)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class KeywordSnapshotInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keywords: List[Dict[str, Any]] = Field(..., description="List of keyword objects with at minimum 'keyword' field plus any metrics to update")


class LogSeoChangeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: str = Field(..., description="Page path or identifier")
    change_type: str = Field(..., description="Type: title-meta, content-expansion, new-content, schema, internal-links")
    old_value: Optional[str] = Field(default=None)
    new_value: Optional[str] = Field(default=None)
    target_keyword: Optional[str] = Field(default=None)
    before_position: Optional[str] = Field(default=None)
    date: Optional[str] = Field(default=None, description="Date of change (defaults to today)")


class ListSeoChangesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: Optional[str] = Field(default=None, description="Filter by page path")
    change_type: Optional[str] = Field(default=None)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class LogIndexSubmissionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="URL path submitted")
    title: Optional[str] = Field(default=None, description="Page title")
    method: Optional[str] = Field(default="gsc-request", description="Method: gsc-request, sitemap, url-inspection")
    status: Optional[str] = Field(default="submitted", description="Status: submitted, indexed, not-indexed")
    notes: Optional[str] = Field(default=None)
    date: Optional[str] = Field(default=None)


class ListIndexSubmissionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Optional[str] = Field(default=None)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class AddPartnerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Contact or company name", min_length=1)
    company: Optional[str] = Field(default=None)
    relationship: Optional[str] = Field(default=None)
    network: Optional[str] = Field(default=None, description="Their audience/network")
    partnership_type: Optional[str] = Field(default=None, description="affiliate, content-collaboration, referral, software-partner")
    potential_value: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default="to-explore")
    next_action: Optional[str] = Field(default=None)
    notes: Optional[List[str]] = Field(default=None)


class UpdatePartnerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Partner name to find")
    status: Optional[str] = Field(default=None)
    next_action: Optional[str] = Field(default=None)
    add_note: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)
    partnership_type: Optional[str] = Field(default=None)


class DashboardInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


# ============================================================
# CONTENT REGISTRY TOOLS
# ============================================================


@mcp.tool(name="mrt_register_content")
async def register_content(params: RegisterContentInput) -> str:
    """Register a new content piece (blog post, guide, landing page, etc.).

    Use this when the user says they've published new content. Generates
    content_id from the path automatically. Checks for duplicates by path.

    Args:
        params: Content details including title, type, category, path.

    Returns:
        str: Confirmation with content ID.
    """
    items = _load(CONTENT_FILE)

    # Duplicate check
    for item in items:
        if item.get("path", "").lower() == params.path.lower():
            return f"⚠️ Content already exists at `{params.path}`: **{item['title']}**"

    content_id = _make_id(params.path)
    record = {
        "content_id": content_id,
        "title": params.title,
        "content_type": params.content_type.lower(),
        "category": params.category.lower(),
        "path": params.path,
        "youtube_url": params.youtube_url or "",
        "status": (params.status or "live").lower(),
        "date_added": params.date_added or datetime.now().strftime("%b %Y"),
        "tags": params.tags or [],
        "notes": params.notes or [],
    }

    items.append(record)
    _save(CONTENT_FILE, items)

    return f"✅ Registered: **{params.title}** (`{content_id}`)\n- Path: `{params.path}`\n- Type: {record['content_type']} | Category: {record['category']}"


@mcp.tool(name="mrt_get_content")
async def get_content(params: ContentIdentifierInput) -> str:
    """Get full details for a specific content piece by ID, path, or title search.

    Args:
        params: Identifier to search by.

    Returns:
        str: Full content details in markdown format.
    """
    items = _load(CONTENT_FILE)
    for item in items:
        if _match(item, params.identifier):
            return _fmt_content_md(item)

    return f"❌ No content found matching '{params.identifier}'"


@mcp.tool(name="mrt_list_content")
async def list_content(params: ListContentInput) -> str:
    """List all content, optionally filtered by type, category, status, or video presence.

    Use to answer 'what do we have about X' or 'show me all blog posts'.

    Args:
        params: Optional filters and response format.

    Returns:
        str: List of matching content.
    """
    items = _load(CONTENT_FILE)

    if params.content_type:
        items = [i for i in items if i["content_type"] == params.content_type.lower()]
    if params.category:
        items = [i for i in items if i["category"] == params.category.lower()]
    if params.status:
        items = [i for i in items if i["status"] == params.status.lower()]
    if params.has_video is True:
        items = [i for i in items if i.get("youtube_url")]
    elif params.has_video is False:
        items = [i for i in items if not i.get("youtube_url")]

    if not items:
        return "No content matches the given filters."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(items, indent=2)

    lines = [f"## Content ({len(items)} items)\n"]
    for item in sorted(items, key=lambda x: (x["category"], x["title"])):
        lines.append(_fmt_content_short(item))

    return "\n".join(lines)


@mcp.tool(name="mrt_search_content")
async def search_content(params: SearchInput) -> str:
    """Full-text search across all content fields — title, tags, notes, category, path.

    Args:
        params: Search query and response format.

    Returns:
        str: Matching content.
    """
    items = _load(CONTENT_FILE)
    matches = [i for i in items if _search_fields(i, params.query)]

    if not matches:
        return f"No content found matching '{params.query}'"

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(matches, indent=2)

    lines = [f"## Search: '{params.query}' ({len(matches)} results)\n"]
    for item in matches:
        lines.append(_fmt_content_short(item))
    return "\n".join(lines)


@mcp.tool(name="mrt_update_content")
async def update_content(params: UpdateContentInput) -> str:
    """Update any field on a content record.

    Args:
        params: Content identifier and fields to update.

    Returns:
        str: Confirmation of changes.
    """
    items = _load(CONTENT_FILE)
    changes = []

    for item in items:
        if _match(item, params.identifier):
            if params.title:
                item["title"] = params.title
                changes.append("title")
            if params.content_type:
                item["content_type"] = params.content_type.lower()
                changes.append("content_type")
            if params.category:
                item["category"] = params.category.lower()
                changes.append("category")
            if params.path:
                item["path"] = params.path
                item["content_id"] = _make_id(params.path)
                changes.append("path")
            if params.youtube_url:
                item["youtube_url"] = params.youtube_url
                changes.append("youtube_url")
            if params.status:
                item["status"] = params.status.lower()
                changes.append("status")
            if params.add_tags:
                existing = item.get("tags", [])
                item["tags"] = list(set(existing + params.add_tags))
                changes.append("tags (added)")
            if params.tags is not None:
                item["tags"] = params.tags
                changes.append("tags (replaced)")
            if params.add_note:
                notes = item.get("notes", [])
                if isinstance(notes, str):
                    notes = [notes] if notes else []
                notes.append(f"[{_now()}] {params.add_note}")
                item["notes"] = notes
                changes.append("note added")

            if changes:
                _save(CONTENT_FILE, items)
                return f"✅ Updated **{item['title']}**: {', '.join(changes)}"
            return "No changes specified."

    return f"❌ No content found matching '{params.identifier}'"


@mcp.tool(name="mrt_content_stats")
async def content_stats(params: DashboardInput) -> str:
    """Summary statistics for the content library.

    Shows total count, breakdown by type, category, and video coverage.

    Args:
        params: Response format.

    Returns:
        str: Content statistics.
    """
    items = _load(CONTENT_FILE)

    by_type: Dict[str, int] = {}
    by_cat: Dict[str, int] = {}
    with_video = 0

    for item in items:
        t = item["content_type"]
        c = item["category"]
        by_type[t] = by_type.get(t, 0) + 1
        by_cat[c] = by_cat.get(c, 0) + 1
        if item.get("youtube_url"):
            with_video += 1

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "total": len(items),
            "by_type": by_type,
            "by_category": by_cat,
            "with_video": with_video,
            "video_coverage_pct": round(with_video / len(items) * 100, 1) if items else 0,
        }, indent=2)

    lines = [
        f"## Content Stats\n",
        f"**Total:** {len(items)} | **With Video:** {with_video} ({round(with_video / len(items) * 100, 1) if items else 0}%)\n",
        "### By Type",
    ]
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"- {t}: {count}")
    lines.append("\n### By Category")
    for c, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        lines.append(f"- {c}: {count}")

    return "\n".join(lines)


@mcp.tool(name="mrt_get_uncovered")
async def get_uncovered(params: DashboardInput) -> str:
    """Show content pieces that don't have YouTube videos yet.

    Args:
        params: Response format.

    Returns:
        str: Content without videos, sorted by category.
    """
    items = _load(CONTENT_FILE)
    uncovered = [i for i in items if not i.get("youtube_url")]

    if not uncovered:
        return "🎉 All content has YouTube videos!"

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(uncovered, indent=2)

    lines = [f"## Content Without Videos ({len(uncovered)} / {len(items)})\n"]
    current_cat = None
    for item in sorted(uncovered, key=lambda x: (x["category"], x["title"])):
        if item["category"] != current_cat:
            current_cat = item["category"]
            lines.append(f"\n### {current_cat.title()}")
        lines.append(f"- [{item['content_type']}] {item['title']} (`{item['path']}`)")

    return "\n".join(lines)


# ============================================================
# BACKLOG TOOLS
# ============================================================


@mcp.tool(name="mrt_add_backlog")
async def add_backlog(params: AddBacklogInput) -> str:
    """Add a new item to the content backlog.

    Use when the user identifies something to build. Include the 'why'
    to make prioritization decisions easier later.

    Args:
        params: Backlog item details.

    Returns:
        str: Confirmation.
    """
    items = _load(BACKLOG_FILE)

    # Duplicate check
    for item in items:
        if item["title"].lower() == params.title.lower():
            return f"⚠️ Already on backlog: **{item['title']}** [{item['priority']}] [{item['status']}]"

    backlog_id = _make_id(params.title)
    record = {
        "backlog_id": backlog_id,
        "title": params.title,
        "content_type": params.content_type.lower(),
        "category": params.category.lower(),
        "priority": params.priority.lower(),
        "why": params.why or "",
        "status": (params.status or "backlog").lower(),
        "completed_date": None,
        "notes": [],
    }

    items.append(record)
    _save(BACKLOG_FILE, items)

    return f"✅ Added to backlog: **{params.title}** [{params.priority}] [{record['status']}]"


@mcp.tool(name="mrt_list_backlog")
async def list_backlog(params: ListBacklogInput) -> str:
    """List backlog items, filterable by priority, category, or status.

    Args:
        params: Filters and response format.

    Returns:
        str: Matching backlog items.
    """
    items = _load(BACKLOG_FILE)

    # Exclude done items by default unless specifically filtered
    if params.status:
        items = [i for i in items if params.status.lower() in i["status"].lower()]
    else:
        items = [i for i in items if "done" not in i["status"].lower()]

    if params.priority:
        items = [i for i in items if i["priority"] == params.priority.lower()]
    if params.category:
        items = [i for i in items if i["category"] == params.category.lower()]

    if not items:
        return "No backlog items match the given filters."

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    items.sort(key=lambda x: (priority_order.get(x["priority"], 99), x["title"]))

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(items, indent=2)

    lines = [f"## Backlog ({len(items)} items)\n"]
    for item in items:
        lines.append(f"- **[{item['priority'].upper()}]** {item['title']} — {item['status']}")
        if item.get("why"):
            lines.append(f"  _{item['why']}_")

    return "\n".join(lines)


@mcp.tool(name="mrt_update_backlog")
async def update_backlog(params: UpdateBacklogInput) -> str:
    """Update status, priority, or other fields on a backlog item.

    Args:
        params: Item identifier and fields to update.

    Returns:
        str: Confirmation.
    """
    items = _load(BACKLOG_FILE)
    changes = []

    for item in items:
        if _match(item, params.identifier, id_field="backlog_id"):
            if params.priority:
                item["priority"] = params.priority.lower()
                changes.append(f"priority → {params.priority}")
            if params.status:
                item["status"] = params.status.lower()
                changes.append(f"status → {params.status}")
            if params.why:
                item["why"] = params.why
                changes.append("why updated")
            if params.add_note:
                notes = item.get("notes", [])
                notes.append(f"[{_now()}] {params.add_note}")
                item["notes"] = notes
                changes.append("note added")

            if changes:
                _save(BACKLOG_FILE, items)
                return f"✅ Updated **{item['title']}**: {', '.join(changes)}"
            return "No changes specified."

    return f"❌ No backlog item found matching '{params.identifier}'"


@mcp.tool(name="mrt_complete_backlog")
async def complete_backlog(params: CompleteBacklogInput) -> str:
    """Mark a backlog item as done and optionally register it as content.

    This is a two-in-one: close backlog + register content. Use when the
    user says they've finished building something that was on the backlog.

    Args:
        params: Item identifier and optional path for content registration.

    Returns:
        str: Confirmation with next steps.
    """
    items = _load(BACKLOG_FILE)
    result_lines = []

    for item in items:
        if _match(item, params.identifier, id_field="backlog_id"):
            item["status"] = "done"
            item["completed_date"] = _now()
            _save(BACKLOG_FILE, items)
            result_lines.append(f"✅ Completed: **{item['title']}**")

            # Also register as content if path provided
            if params.path:
                reg = RegisterContentInput(
                    title=item["title"],
                    content_type=item["content_type"],
                    category=item["category"],
                    path=params.path,
                    youtube_url=params.youtube_url,
                )
                reg_result = await register_content(reg)
                result_lines.append(reg_result)

            return "\n".join(result_lines)

    return f"❌ No backlog item found matching '{params.identifier}'"


@mcp.tool(name="mrt_next_up")
async def next_up(params: DashboardInput) -> str:
    """Show the top backlog items to work on next.

    Returns high-priority items with 'ready' or 'spec-complete' status first,
    then other high-priority items. Includes the 'why' field for discussion.

    Use when the user asks 'what should I work on?' or 'what's next?'

    Args:
        params: Response format.

    Returns:
        str: Prioritized list of next items to build.
    """
    items = _load(BACKLOG_FILE)
    active = [i for i in items if "done" not in i["status"].lower()]

    # Score: priority + readiness
    priority_score = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    readiness_score = {"ready": 0, "spec-complete": 0, "in-progress": 1, "spec-needed": 2, "backlog": 3}

    active.sort(key=lambda x: (
        priority_score.get(x["priority"], 99),
        readiness_score.get(x["status"].lower().replace(" ", "-"), 99),
        x["title"],
    ))

    top = active[:10]

    if not top:
        return "🎉 Backlog is empty! Time to brainstorm."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(top, indent=2)

    lines = ["## Next Up\n"]
    for item in top:
        emoji = "🔴" if item["priority"] == "critical" else "🟠" if item["priority"] == "high" else "🟡" if item["priority"] == "medium" else "⚪"
        lines.append(f"{emoji} **{item['title']}** [{item['status']}]")
        if item.get("why"):
            lines.append(f"   _{item['why']}_")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# GSC KEYWORD TOOLS
# ============================================================


@mcp.tool(name="mrt_track_keyword")
async def track_keyword(params: TrackKeywordInput) -> str:
    """Add or update a keyword with current GSC metrics.

    If the keyword already exists, updates its metrics and saves the previous
    values to history. If new, creates a new tracking entry.

    Args:
        params: Keyword and its current metrics.

    Returns:
        str: Confirmation.
    """
    items = _load(KEYWORDS_FILE)

    # Find existing
    existing = None
    for item in items:
        if item["keyword"].lower() == params.keyword.lower():
            existing = item
            break

    if existing:
        # Save current to history before updating
        history_entry = {
            "date": existing.get("last_updated", _now()),
            "position": existing.get("position"),
            "impressions": existing.get("impressions"),
            "clicks": existing.get("clicks"),
        }
        history = existing.get("history", [])
        history.append(history_entry)
        existing["history"] = history

        # Update fields
        if params.impressions is not None:
            existing["impressions"] = params.impressions
        if params.clicks is not None:
            existing["clicks"] = params.clicks
        if params.position is not None:
            existing["position"] = params.position
        if params.ctr is not None:
            existing["ctr"] = params.ctr
        if params.status:
            existing["status"] = params.status
        if params.action:
            existing["action"] = params.action
        if params.content_path:
            existing["content_path"] = params.content_path
        existing["last_updated"] = _now()

        _save(KEYWORDS_FILE, items)
        return f"✅ Updated keyword: **{params.keyword}** (pos: {existing.get('position')})"
    else:
        record = {
            "keyword": params.keyword,
            "impressions": params.impressions or 0,
            "clicks": params.clicks or 0,
            "position": params.position,
            "ctr": params.ctr or "0%",
            "status": params.status or "new-content",
            "action": params.action or "",
            "content_path": params.content_path or "",
            "last_updated": _now(),
            "history": [],
        }
        items.append(record)
        _save(KEYWORDS_FILE, items)
        return f"✅ Now tracking: **{params.keyword}**"


@mcp.tool(name="mrt_list_keywords")
async def list_keywords(params: ListKeywordsInput) -> str:
    """List all tracked keywords, optionally filtered by status.

    Args:
        params: Optional status filter and response format.

    Returns:
        str: Keyword list with metrics.
    """
    items = _load(KEYWORDS_FILE)

    if params.status:
        items = [i for i in items if i.get("status") == params.status.lower()]

    if not items:
        return "No keywords tracked yet."

    items.sort(key=lambda x: x.get("position") or 999)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(items, indent=2)

    lines = [f"## Tracked Keywords ({len(items)})\n"]
    for item in items:
        lines.append(_fmt_keyword_md(item))
    return "\n".join(lines)


@mcp.tool(name="mrt_keyword_wins")
async def keyword_wins(params: DashboardInput) -> str:
    """Show keywords where MyRoboticTrader is performing well.

    Returns keywords with status 'winning' or position < 10.

    Args:
        params: Response format.

    Returns:
        str: Winning keywords.
    """
    items = _load(KEYWORDS_FILE)
    wins = [
        i for i in items
        if i.get("status") == "winning" or (i.get("position") and i["position"] < 10)
    ]

    if not wins:
        return "No keyword wins yet — keep building!"

    wins.sort(key=lambda x: x.get("position") or 999)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(wins, indent=2)

    lines = ["## 🏆 Keyword Wins\n"]
    for item in wins:
        lines.append(_fmt_keyword_md(item))
    return "\n".join(lines)


@mcp.tool(name="mrt_keyword_snapshot")
async def keyword_snapshot(params: KeywordSnapshotInput) -> str:
    """Bulk update keyword metrics from a GSC data export.

    Pass a list of keyword objects. Each must have a 'keyword' field;
    other fields (impressions, clicks, position, ctr, status) are optional.

    Args:
        params: List of keyword data to update.

    Returns:
        str: Summary of updates.
    """
    updated = 0
    created = 0

    for kw_data in params.keywords:
        keyword = kw_data.get("keyword")
        if not keyword:
            continue
        track_input = TrackKeywordInput(
            keyword=keyword,
            impressions=kw_data.get("impressions"),
            clicks=kw_data.get("clicks"),
            position=kw_data.get("position"),
            ctr=kw_data.get("ctr"),
            status=kw_data.get("status"),
            action=kw_data.get("action"),
            content_path=kw_data.get("content_path"),
        )
        result = await track_keyword(track_input)
        if "Updated" in result:
            updated += 1
        else:
            created += 1

    return f"✅ Keyword snapshot complete: {updated} updated, {created} new"


# ============================================================
# SEO LOG TOOLS
# ============================================================


@mcp.tool(name="mrt_log_seo_change")
async def log_seo_change(params: LogSeoChangeInput) -> str:
    """Record an SEO optimization you just made.

    Use after changing titles, meta descriptions, content, schema,
    or internal links on any page.

    Args:
        params: Details of the SEO change.

    Returns:
        str: Confirmation.
    """
    items = _load(SEO_LOG_FILE)

    record = {
        "date": params.date or _now(),
        "page": params.page,
        "change_type": params.change_type,
        "old_value": params.old_value or "",
        "new_value": params.new_value or "",
        "target_keyword": params.target_keyword or "",
        "before_position": params.before_position or "",
    }

    items.append(record)
    _save(SEO_LOG_FILE, items)

    return f"✅ Logged SEO change: {params.change_type} on `{params.page}`"


@mcp.tool(name="mrt_list_seo_changes")
async def list_seo_changes(params: ListSeoChangesInput) -> str:
    """List SEO changes, optionally filtered by page or change type.

    Args:
        params: Filters and response format.

    Returns:
        str: SEO change log.
    """
    items = _load(SEO_LOG_FILE)

    if params.page:
        items = [i for i in items if params.page.lower() in i["page"].lower()]
    if params.change_type:
        items = [i for i in items if params.change_type.lower() in i["change_type"].lower()]

    if not items:
        return "No SEO changes logged yet."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(items, indent=2)

    lines = [f"## SEO Log ({len(items)} entries)\n"]
    for item in items:
        lines.append(f"**{item['date']}** | `{item['page']}` | {item['change_type']}")
        if item.get("target_keyword"):
            lines.append(f"  Target: _{item['target_keyword']}_")
        if item.get("new_value"):
            lines.append(f"  → {item['new_value']}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# INDEX SUBMISSION TOOLS
# ============================================================


@mcp.tool(name="mrt_log_index_submission")
async def log_index_submission(params: LogIndexSubmissionInput) -> str:
    """Record a GSC index submission.

    Args:
        params: Submission details.

    Returns:
        str: Confirmation.
    """
    items = _load(INDEX_SUBS_FILE)

    record = {
        "date": params.date or _now(),
        "path": params.path,
        "title": params.title or "",
        "method": params.method or "gsc-request",
        "status": params.status or "submitted",
        "notes": params.notes or "",
    }

    items.append(record)
    _save(INDEX_SUBS_FILE, items)

    return f"✅ Logged index submission: `{params.path}` via {record['method']}"


@mcp.tool(name="mrt_list_index_submissions")
async def list_index_submissions(params: ListIndexSubmissionsInput) -> str:
    """List index submissions, optionally filtered by status.

    Args:
        params: Optional status filter and response format.

    Returns:
        str: Submission list.
    """
    items = _load(INDEX_SUBS_FILE)

    if params.status:
        items = [i for i in items if params.status.lower() in i["status"].lower()]

    if not items:
        return "No index submissions logged."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(items, indent=2)

    lines = [f"## Index Submissions ({len(items)})\n"]
    for item in items:
        status_emoji = "✅" if item["status"] == "indexed" else "⏳" if item["status"] == "submitted" else "❌"
        lines.append(f"{status_emoji} {item['date']} | `{item['path']}` | {item['method']} | {item['status']}")

    return "\n".join(lines)


# ============================================================
# PARTNERSHIP / AFFILIATE TOOLS
# ============================================================


@mcp.tool(name="mrt_add_partner")
async def add_partner(params: AddPartnerInput) -> str:
    """Add a new affiliate or partnership lead.

    Args:
        params: Partner details.

    Returns:
        str: Confirmation.
    """
    items = _load(PARTNERSHIPS_FILE)

    # Duplicate check
    for item in items:
        if item["name"].lower() == params.name.lower():
            return f"⚠️ Partner already exists: **{item['name']}** [{item.get('status', '?')}]"

    record = {
        "name": params.name,
        "company": params.company or "",
        "relationship": params.relationship or "",
        "network": params.network or "",
        "partnership_type": params.partnership_type or "",
        "potential_value": params.potential_value or "",
        "status": params.status or "to-explore",
        "next_action": params.next_action or "",
        "notes": params.notes or [],
    }

    items.append(record)
    _save(PARTNERSHIPS_FILE, items)

    return f"✅ Added partner: **{params.name}** ({params.company or 'Independent'})"


@mcp.tool(name="mrt_list_partners")
async def list_partners(params: DashboardInput) -> str:
    """List all affiliate and partnership leads.

    Args:
        params: Response format.

    Returns:
        str: Partner list.
    """
    items = _load(PARTNERSHIPS_FILE)

    if not items:
        return "No partnership or affiliate leads tracked."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(items, indent=2)

    lines = ["## Affiliates & Partners\n"]
    for item in items:
        lines.append(f"### {item['name']} — {item.get('company', '?')}")
        lines.append(f"- **Type:** {item.get('partnership_type', '?')} | **Status:** {item.get('status', '?')}")
        lines.append(f"- **Value:** {item.get('potential_value', '?')}")
        if item.get("next_action"):
            lines.append(f"- **Next:** {item['next_action']}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(name="mrt_update_partner")
async def update_partner(params: UpdatePartnerInput) -> str:
    """Update a partnership lead's status, next action, or notes.

    Args:
        params: Partner name and fields to update.

    Returns:
        str: Confirmation.
    """
    items = _load(PARTNERSHIPS_FILE)
    changes = []

    for item in items:
        if params.name.lower() in item["name"].lower():
            if params.status:
                item["status"] = params.status
                changes.append(f"status → {params.status}")
            if params.next_action:
                item["next_action"] = params.next_action
                changes.append("next_action updated")
            if params.company:
                item["company"] = params.company
                changes.append("company updated")
            if params.partnership_type:
                item["partnership_type"] = params.partnership_type
                changes.append("partnership_type updated")
            if params.add_note:
                notes = item.get("notes", [])
                if isinstance(notes, str):
                    notes = [notes] if notes else []
                notes.append(f"[{_now()}] {params.add_note}")
                item["notes"] = notes
                changes.append("note added")

            if changes:
                _save(PARTNERSHIPS_FILE, items)
                return f"✅ Updated **{item['name']}**: {', '.join(changes)}"
            return "No changes specified."

    return f"❌ No partner found matching '{params.name}'"


# ============================================================
# DASHBOARD
# ============================================================


@mcp.tool(name="mrt_dashboard")
async def dashboard(params: DashboardInput) -> str:
    """High-level overview of MyRoboticTrader content status.

    Shows content counts, backlog status, keyword wins, pending items.
    Run this as a weekly check-in.

    Args:
        params: Response format.

    Returns:
        str: Dashboard summary.
    """
    content = _load(CONTENT_FILE)
    backlog = _load(BACKLOG_FILE)
    keywords = _load(KEYWORDS_FILE)
    seo_log = _load(SEO_LOG_FILE)
    idx_subs = _load(INDEX_SUBS_FILE)
    partners = _load(PARTNERSHIPS_FILE)

    # Content stats
    by_type: Dict[str, int] = {}
    with_video = 0
    for item in content:
        t = item["content_type"]
        by_type[t] = by_type.get(t, 0) + 1
        if item.get("youtube_url"):
            with_video += 1

    # Backlog stats
    active_backlog = [b for b in backlog if "done" not in b["status"].lower()]
    ready_items = [b for b in active_backlog if b["status"].lower() in ("ready", "spec-complete")]
    critical_items = [b for b in active_backlog if b["priority"] == "critical"]

    # Keyword stats
    winning_kw = [k for k in keywords if k.get("status") == "winning" or (k.get("position") and k["position"] < 10)]

    # Pending submissions
    pending_subs = [s for s in idx_subs if s["status"] == "submitted"]

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "content_total": len(content),
            "content_by_type": by_type,
            "video_coverage": with_video,
            "backlog_active": len(active_backlog),
            "backlog_ready": len(ready_items),
            "backlog_critical": len(critical_items),
            "keywords_tracked": len(keywords),
            "keywords_winning": len(winning_kw),
            "seo_changes": len(seo_log),
            "pending_index_submissions": len(pending_subs),
            "partnership_leads": len(partners),
        }, indent=2)

    lines = [
        "# 📊 MyRoboticTrader Content Dashboard\n",
        f"## Content Library: {len(content)} pieces",
    ]
    if content:
        lines.append(f"- Video coverage: {with_video}/{len(content)} ({round(with_video / len(content) * 100, 1)}%)")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"- {t}: {count}")

    lines.append(f"\n## Backlog: {len(active_backlog)} active items")
    if critical_items:
        lines.append(f"- 🔴 **{len(critical_items)} critical**")
    lines.append(f"- ✅ {len(ready_items)} ready to build")

    lines.append(f"\n## Keywords: {len(keywords)} tracked")
    lines.append(f"- 🏆 {len(winning_kw)} winning (position < 10 or status=winning)")

    if pending_subs:
        lines.append(f"\n## Pending: {len(pending_subs)} index submissions awaiting confirmation")

    lines.append(f"\n## Affiliates & Partners: {len(partners)} leads")

    if seo_log:
        lines.append(f"\n## SEO Log: {len(seo_log)} changes recorded")
        last = seo_log[-1]
        lines.append(f"- Last: {last['date']} — {last['change_type']} on `{last['page']}`")

    return "\n".join(lines)


# ============================================================
# SEED DATA
# ============================================================


@mcp.tool(name="mrt_seed_data")
async def seed_data(params: DashboardInput) -> str:
    """Seed the database with initial MyRoboticTrader content data.

    ⚠️ This will overwrite existing data. Only run on first setup.

    Args:
        params: Response format (ignored).

    Returns:
        str: Confirmation with counts.
    """
    _ensure_data_dir()

    # Start with empty data files — content will be added as we go
    content_data: list = []
    backlog_data: list = []
    keywords_data: list = []
    seo_data: list = []
    idx_data: list = []
    partner_data: list = []

    _save(CONTENT_FILE, content_data)
    _save(BACKLOG_FILE, backlog_data)
    _save(KEYWORDS_FILE, keywords_data)
    _save(SEO_LOG_FILE, seo_data)
    _save(INDEX_SUBS_FILE, idx_data)
    _save(PARTNERSHIPS_FILE, partner_data)

    return (
        "✅ MyRoboticTrader content tracker initialized!\n"
        "All data stores created (empty — ready for content).\n\n"
        "Next steps:\n"
        "- Use `mrt_register_content` to add existing blog posts\n"
        "- Use `mrt_add_backlog` to queue up content ideas\n"
        "- Use `mrt_track_keyword` to start tracking GSC keywords\n"
        "- Use `mrt_add_partner` to track affiliate relationships"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import os as _os
    _transport = _os.environ.get("MCP_TRANSPORT", "stdio")
    if _transport == "sse":
        mcp.settings.host = _os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(_os.environ.get("MCP_PORT", "8084"))
        mcp.settings.transport_security.enable_dns_rebinding_protection = False
        mcp.run(transport="sse")
    else:
        mcp.run()
