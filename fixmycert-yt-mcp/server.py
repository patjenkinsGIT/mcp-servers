"""
FixMyCert YouTube Channel Manager MCP Server

Tracks videos, cross-references, placeholder debt, update queues,
and content coverage gaps for the FixMyCert YouTube channel.

Transport: stdio (local only)
Storage: JSON files in ~/.fixmycert-yt/
"""

import json
import os
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ============================================================
# Constants
# ============================================================

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path.home() / ".fixmycert-yt")))
VIDEOS_FILE = DATA_DIR / "videos.json"
CONTENT_MAP_FILE = DATA_DIR / "content-map.json"
UPDATE_QUEUE_FILE = DATA_DIR / "update-queue.json"
BACKUP_DIR = DATA_DIR / "backup"

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

CATEGORIES = [
    "compliance", "certificates", "troubleshooting", "enterprise",
    "fundamentals", "openssl", "java", "cdn", "email", "web-servers",
    "governance",
]

# Community Posts
POSTS_FILE = DATA_DIR / "community-posts.json"
POST_TYPES = ["text", "poll", "image", "link"]
POST_STATUSES = ["draft", "published", "scheduled"]

# Known compliance deadlines for yt_suggest_posts
COMPLIANCE_DEADLINES = [
    {"date": "2026-03-15", "event": "47-day certificate maximum validity begins (Apple enforcement)"},
    {"date": "2026-03-15", "event": "200-day maximum validity takes effect"},
    {"date": "2026-09-15", "event": "100-day maximum validity takes effect"},
    {"date": "2027-03-15", "event": "Apple 45-day DCV reuse limit begins"},
    {"date": "2027-03-15", "event": "47-day maximum validity takes effect (all browsers)"},
    {"date": "2028-03-01", "event": "Domain validation method sunset complete"},
    {"date": "2029-03-15", "event": "10-day DCV reuse limit takes effect"},
]

# Poll idea bank for yt_suggest_posts
POLL_TEMPLATES = [
    {
        "content": "What's your biggest certificate management headache?",
        "poll_options": ["Manual renewals", "Incomplete chains", "Wildcard sprawl", "\"It works in my browser\""],
        "tags": ["troubleshooting", "certificates"],
    },
    {
        "content": "How are you preparing for 47-day certificates?",
        "poll_options": ["ACME automation", "CLM platform (Venafi, etc.)", "Still figuring it out", "Wait, what?"],
        "tags": ["47-day", "automation"],
    },
    {
        "content": "What certificate automation method do you use most?",
        "poll_options": ["HTTP-01", "DNS-01", "TLS-ALPN-01", "None yet"],
        "tags": ["automation", "ACME"],
    },
    {
        "content": "Where do you manage your certificates?",
        "poll_options": ["Spreadsheet/manual", "Venafi/CLM platform", "Let's Encrypt + cron", "Cloud provider (ACM/Key Vault)"],
        "tags": ["enterprise", "automation"],
    },
    {
        "content": "What's the worst certificate outage you've dealt with?",
        "poll_options": ["Expired cert on production", "Wrong chain served", "Wildcard key compromise", "CA distrust (Symantec/Entrust)"],
        "tags": ["troubleshooting", "pki-disasters"],
    },
    {
        "content": "Does your organization have a Certificate Practice Statement (CPS)?",
        "poll_options": ["Yes, fully documented", "Partial/outdated", "No, but we should", "What's a CPS?"],
        "tags": ["governance", "compliance"],
    },
    {
        "content": "Which cloud provider's certificate management is the most confusing?",
        "poll_options": ["AWS ACM", "Azure Key Vault", "GCP Certificate Manager", "They're all painful"],
        "tags": ["cloud", "cdn"],
    },
    {
        "content": "How do you handle internal CA certificates?",
        "poll_options": ["ADCS (Active Directory)", "HashiCorp Vault", "step-ca / smallstep", "OpenSSL scripts"],
        "tags": ["enterprise", "certificates"],
    },
]

# ============================================================
# Server
# ============================================================

mcp = FastMCP("fixmycert_yt_mcp")

# ============================================================
# Storage helpers
# ============================================================


def _ensure_data_dir() -> None:
    """Create data directory structure if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def _backup(filepath: Path) -> None:
    """Create a timestamped backup before writing."""
    if filepath.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = BACKUP_DIR / f"{filepath.stem}_{stamp}.json"
        shutil.copy2(filepath, dest)
        # Keep only last 20 backups per file type
        prefix = filepath.stem
        backups = sorted(BACKUP_DIR.glob(f"{prefix}_*.json"))
        for old in backups[:-20]:
            old.unlink()


def _read_json(filepath: Path) -> list | dict:
    """Read JSON file, return empty list if missing."""
    _ensure_data_dir()
    if not filepath.exists():
        return []
    with open(filepath, "r") as f:
        return json.load(f)


def _write_json(filepath: Path, data: list | dict) -> None:
    """Write JSON file with backup."""
    _ensure_data_dir()
    _backup(filepath)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _now_iso() -> str:
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# Video helpers
# ============================================================


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    url = url.strip()
    # Handle youtu.be/ID
    if "youtu.be/" in url:
        part = url.split("youtu.be/")[1]
        return part.split("?")[0].split("[")[0].strip()
    # Handle youtube.com/watch?v=ID
    if "v=" in url:
        part = url.split("v=")[1]
        return part.split("&")[0].split("[")[0].strip()
    # Handle youtube.com/embed/ID
    if "/embed/" in url:
        part = url.split("/embed/")[1]
        return part.split("?")[0].split("[")[0].strip()
    return url.strip()


def _find_video(videos: list, identifier: str) -> dict | None:
    """Find a video by ID, URL fragment, or title substring (case-insensitive)."""
    identifier = identifier.strip().lower()
    for v in videos:
        if v["id"].lower() == identifier:
            return v
        if identifier in v.get("url", "").lower():
            return v
        if identifier in v.get("title", "").lower():
            return v
    return None


def _canonical_url(video_id: str) -> str:
    """Return canonical YouTube URL."""
    return f"https://www.youtube.com/watch?v={video_id}"


def _format_video_summary(v: dict) -> str:
    """Format a single video as a markdown summary."""
    lines = [
        f"### {v['title']}",
        f"**URL:** {v['url']}",
        f"**ID:** {v['id']}",
        f"**Category:** {v.get('category', 'unset')}",
        f"**Guide:** {v.get('guide_url', 'none')}",
        f"**Published:** {v.get('published_date', 'unknown')}",
    ]
    tags = v.get("tags", [])
    if tags:
        lines.append(f"**Tags:** {', '.join(tags)}")

    related = v.get("related_videos", [])
    if related:
        lines.append("**Related Videos:**")
        for r in related:
            status_icon = "✅" if r.get("status") == "linked" else "⏳"
            lines.append(f"  {status_icon} {r.get('video_id', 'unknown')} ({r.get('status', 'unknown')})")

    notes = v.get("notes", [])
    if notes:
        lines.append("**Notes:**")
        for n in notes:
            lines.append(f"  - {n}")

    ab = v.get("ab_test", {})
    if ab.get("status") and ab["status"] != "none":
        lines.append(f"**A/B Test:** {ab['status']} — {ab.get('element', '')} (started {ab.get('start_date', '?')})")

    return "\n".join(lines)


# ============================================================
# Input models
# ============================================================


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class RegisterVideoInput(BaseModel):
    """Input for registering a new video."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(..., description="Video title as shown on YouTube", min_length=1, max_length=200)
    url: str = Field(..., description="YouTube URL (any format — youtu.be, watch?v=, etc.)", min_length=5)
    guide_url: Optional[str] = Field(default=None, description="FixMyCert guide/demo/page path (e.g., '/guides/47-day-certificate-timeline')")
    category: Optional[str] = Field(default=None, description="Content category: compliance, certificates, troubleshooting, enterprise, fundamentals, cdn, email, governance, etc.")
    tags: Optional[list[str]] = Field(default_factory=list, description="Topic tags for cross-referencing (e.g., ['47-day', 'automation', 'ACME'])")
    published_date: Optional[str] = Field(default=None, description="Publish date (YYYY-MM-DD). Defaults to today.")
    related_video_ids: Optional[list[str]] = Field(default_factory=list, description="YouTube IDs of related videos (will be marked as 'linked')")
    placeholder_topics: Optional[list[str]] = Field(default_factory=list, description="Topics for related videos not yet published (will be marked as 'placeholder')")
    description: Optional[str] = Field(default=None, description="Current YouTube description text")
    pinned_comment: Optional[str] = Field(default=None, description="Pinned comment text")
    notes: Optional[list[str]] = Field(default_factory=list, description="Any notes about the video")


class UpdateVideoInput(BaseModel):
    """Input for updating a video record."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    identifier: str = Field(..., description="Video ID, URL, or title substring to find the video")
    title: Optional[str] = Field(default=None, description="Updated title")
    guide_url: Optional[str] = Field(default=None, description="Updated guide URL path")
    category: Optional[str] = Field(default=None, description="Updated category")
    tags: Optional[list[str]] = Field(default=None, description="Replace tags (pass full list)")
    add_tags: Optional[list[str]] = Field(default=None, description="Add tags without replacing existing")
    description: Optional[str] = Field(default=None, description="Updated description text")
    pinned_comment: Optional[str] = Field(default=None, description="Updated pinned comment")
    add_note: Optional[str] = Field(default=None, description="Add a note to the video")
    add_related_id: Optional[str] = Field(default=None, description="Add a linked related video by ID")
    add_placeholder: Optional[str] = Field(default=None, description="Add a placeholder related video topic")


class SearchInput(BaseModel):
    """Input for searching videos."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., description="Search term — matches against title, tags, notes, category, guide URL", min_length=1)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class ListVideosInput(BaseModel):
    """Input for listing videos."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    category: Optional[str] = Field(default=None, description="Filter by category")
    tag: Optional[str] = Field(default=None, description="Filter by tag")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class VideoIdentifierInput(BaseModel):
    """Input that identifies a single video."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    identifier: str = Field(..., description="Video ID, URL, or title substring")


class ResolvePlaceholderInput(BaseModel):
    """Input for resolving a placeholder with a real video URL."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    placeholder_topic: str = Field(..., description="The placeholder topic string to resolve")
    video_url: str = Field(..., description="The actual YouTube URL to replace it with")


class SuggestCrosslinksInput(BaseModel):
    """Input for suggesting cross-links for a video."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    identifier: str = Field(..., description="Video ID, URL, or title substring")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class FlagUpdateInput(BaseModel):
    """Input for flagging a video as needing an update."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    identifier: str = Field(..., description="Video ID, URL, or title substring (or 'topic:keyword' to flag all videos matching a tag/category)")
    reason: str = Field(..., description="Why this video needs updating (e.g., 'Root Causes podcast covered new 47-day clock considerations')", min_length=1, max_length=500)
    priority: Optional[str] = Field(default="normal", description="Priority: 'high', 'normal', or 'low'")


class RegisterContentInput(BaseModel):
    """Input for registering content that could have a video."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    path: str = Field(..., description="FixMyCert URL path (e.g., '/guides/mtls-explained')")
    title: str = Field(..., description="Content title")
    content_type: str = Field(default="guide", description="Type: guide, demo, tool, blog, page")
    category: Optional[str] = Field(default=None, description="Content category")
    priority: Optional[str] = Field(default="medium", description="Video priority: high, medium, low")
    notes: Optional[str] = Field(default=None, description="Any notes about why this needs a video")


class ABTestInput(BaseModel):
    """Input for A/B test operations."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    identifier: str = Field(..., description="Video ID, URL, or title substring")
    element: Optional[str] = Field(default=None, description="What's being tested (e.g., 'thumbnail', 'title')")
    result: Optional[str] = Field(default=None, description="Test result/winner description (for ending tests)")


class DashboardInput(BaseModel):
    """Input for dashboard."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


# Community Post input models

class RegisterPostInput(BaseModel):
    """Input for registering a community post."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    post_type: str = Field(..., description="Type: text, poll, image, link")
    content: str = Field(..., description="The post text", min_length=1, max_length=2000)
    poll_options: Optional[list[str]] = Field(
        default=None, description="Poll choices (required if post_type is poll)"
    )
    link_url: Optional[str] = Field(
        default=None, description="External link included in post"
    )
    linked_video_id: Optional[str] = Field(
        default=None, description="YouTube video ID this post promotes"
    )
    linked_content_path: Optional[str] = Field(
        default=None, description="FixMyCert guide/demo path"
    )
    category: Optional[str] = Field(default=None, description="Content category")
    tags: Optional[list[str]] = Field(default=None, description="Topic tags")
    status: Optional[str] = Field(
        default="published", description="draft, published, or scheduled"
    )
    published_date: Optional[str] = Field(
        default=None, description="Publish date (YYYY-MM-DD). Defaults to today."
    )

    @field_validator("post_type")
    @classmethod
    def validate_post_type(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in POST_TYPES:
            raise ValueError(f"post_type must be one of: {POST_TYPES}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str:
        if v is None:
            return "published"
        v = v.lower().strip()
        if v not in POST_STATUSES:
            raise ValueError(f"status must be one of: {POST_STATUSES}")
        return v


class ListPostsInput(BaseModel):
    """Input for listing community posts."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    post_type: Optional[str] = Field(default=None, description="Filter by type")
    status: Optional[str] = Field(default=None, description="Filter by status")
    category: Optional[str] = Field(default=None, description="Filter by category")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class UpdatePostInput(BaseModel):
    """Input for updating a community post."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    identifier: str = Field(..., description="Post ID or content substring")
    status: Optional[str] = Field(default=None, description="Update status")
    engagement_notes: Optional[str] = Field(
        default=None, description="Add engagement results"
    )
    add_note: Optional[str] = Field(default=None, description="Add a note")
    add_tags: Optional[list[str]] = Field(
        default=None, description="Add tags without replacing existing"
    )


class SuggestPostsInput(BaseModel):
    """Input for suggesting community post ideas."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    strategy: Optional[str] = Field(
        default="all",
        description="Strategy: promote_recent, compliance_deadlines, engagement, content_gaps, all"
    )
    count: Optional[int] = Field(
        default=5, description="Number of suggestions", ge=1, le=15
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class PostCalendarInput(BaseModel):
    """Input for post calendar view."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    weeks_back: Optional[int] = Field(
        default=8, description="How many weeks to show", ge=1, le=52
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


# ============================================================
# Tools — Registry & Lookup
# ============================================================


@mcp.tool(
    name="yt_register_video",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def yt_register_video(params: RegisterVideoInput) -> str:
    """Register a new YouTube video in the channel tracker.

    Use this immediately after publishing a new video. After registering,
    always call yt_get_placeholders to check what existing videos need
    their descriptions updated with the new video's URL.

    Args:
        params: Video details including title, URL, guide path, category, tags, etc.

    Returns:
        str: Confirmation with video ID and any immediate action items
    """
    videos = _read_json(VIDEOS_FILE)
    video_id = _extract_video_id(params.url)

    # Check for duplicates
    if _find_video(videos, video_id):
        return f"⚠️ Video already registered: {video_id}. Use yt_update_video to modify."

    # Build related videos list
    related = []
    for rid in (params.related_video_ids or []):
        related.append({"video_id": _extract_video_id(rid), "status": "linked"})
    for topic in (params.placeholder_topics or []):
        related.append({"video_id": f"PLACEHOLDER_{topic.replace(' ', '_').lower()}", "status": "placeholder", "topic": topic})

    video = {
        "id": video_id,
        "title": params.title,
        "url": _canonical_url(video_id),
        "published_date": params.published_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "guide_url": params.guide_url,
        "category": params.category,
        "tags": params.tags or [],
        "related_videos": related,
        "description": params.description,
        "pinned_comment": params.pinned_comment,
        "thumbnail_type": None,
        "ab_test": {"status": "none", "element": None, "start_date": None, "end_date": None, "result": None},
        "notes": params.notes or [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    videos.append(video)
    _write_json(VIDEOS_FILE, videos)

    # Check if this video resolves any placeholders in other videos
    resolved = []
    for v in videos:
        if v["id"] == video_id:
            continue
        for r in v.get("related_videos", []):
            if r.get("status") == "placeholder":
                topic = r.get("topic", "").lower()
                # Check if any tag from the new video matches the placeholder topic
                for tag in (params.tags or []):
                    if tag.lower() in topic or topic in tag.lower():
                        resolved.append({"video_title": v["title"], "placeholder_topic": r.get("topic")})
                        break

    result = f"✅ Registered: **{params.title}** ({video_id})\n\n"
    result += f"Total videos tracked: {len(videos)}\n\n"

    if resolved:
        result += "🔗 **This video may resolve these placeholders:**\n"
        for r in resolved:
            result += f"  - {r['video_title']} → placeholder: \"{r['placeholder_topic']}\"\n"
        result += "\nRun `yt_get_placeholders` and `yt_suggest_crosslinks` to see all pending updates."
    else:
        result += "💡 Run `yt_get_placeholders` to check if existing videos should link to this one."

    return result


@mcp.tool(
    name="yt_get_video",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_get_video(params: VideoIdentifierInput) -> str:
    """Get full details for a specific video by ID, URL, or title search.

    Args:
        params: Identifier to search by (video ID, URL fragment, or title substring)

    Returns:
        str: Full video details in markdown format
    """
    videos = _read_json(VIDEOS_FILE)
    video = _find_video(videos, params.identifier)
    if not video:
        return f"❌ No video found matching: '{params.identifier}'"
    return _format_video_summary(video)


@mcp.tool(
    name="yt_list_videos",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_list_videos(params: ListVideosInput) -> str:
    """List all videos, optionally filtered by category or tag.

    Args:
        params: Optional category and tag filters, response format

    Returns:
        str: List of videos matching filters
    """
    videos = _read_json(VIDEOS_FILE)

    if params.category:
        videos = [v for v in videos if v.get("category", "").lower() == params.category.lower()]
    if params.tag:
        videos = [v for v in videos if params.tag.lower() in [t.lower() for t in v.get("tags", [])]]

    if not videos:
        return "No videos found matching filters."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(videos, indent=2, default=str)

    lines = [f"## FixMyCert YouTube Videos ({len(videos)} total)\n"]
    # Group by category
    by_cat: dict[str, list] = {}
    for v in videos:
        cat = v.get("category", "uncategorized") or "uncategorized"
        by_cat.setdefault(cat, []).append(v)

    for cat in sorted(by_cat.keys()):
        lines.append(f"### {cat.title()} ({len(by_cat[cat])})")
        for v in by_cat[cat]:
            placeholder_count = sum(1 for r in v.get("related_videos", []) if r.get("status") == "placeholder")
            flag = " ⏳" if placeholder_count > 0 else ""
            lines.append(f"- [{v['title']}]({v['url']}){flag}")
            if v.get("guide_url"):
                lines.append(f"  Guide: {v['guide_url']}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="yt_update_video",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def yt_update_video(params: UpdateVideoInput) -> str:
    """Update any field on a video record.

    Use this to update descriptions, add tags, add notes, add related videos, etc.

    Args:
        params: Video identifier and fields to update

    Returns:
        str: Confirmation of changes made
    """
    videos = _read_json(VIDEOS_FILE)
    video = _find_video(videos, params.identifier)
    if not video:
        return f"❌ No video found matching: '{params.identifier}'"

    changes = []

    if params.title is not None:
        video["title"] = params.title
        changes.append(f"title → {params.title}")
    if params.guide_url is not None:
        video["guide_url"] = params.guide_url
        changes.append(f"guide_url → {params.guide_url}")
    if params.category is not None:
        video["category"] = params.category
        changes.append(f"category → {params.category}")
    if params.tags is not None:
        video["tags"] = params.tags
        changes.append(f"tags → {params.tags}")
    if params.add_tags:
        existing = set(video.get("tags", []))
        new_tags = [t for t in params.add_tags if t not in existing]
        video.setdefault("tags", []).extend(new_tags)
        if new_tags:
            changes.append(f"added tags: {new_tags}")
    if params.description is not None:
        video["description"] = params.description
        changes.append("description updated")
    if params.pinned_comment is not None:
        video["pinned_comment"] = params.pinned_comment
        changes.append("pinned_comment updated")
    if params.add_note:
        video.setdefault("notes", []).append(f"[{_now_iso()[:10]}] {params.add_note}")
        changes.append(f"note added: {params.add_note}")
    if params.add_related_id:
        rid = _extract_video_id(params.add_related_id)
        existing_ids = [r["video_id"] for r in video.get("related_videos", [])]
        if rid not in existing_ids:
            video.setdefault("related_videos", []).append({"video_id": rid, "status": "linked"})
            changes.append(f"added related video: {rid}")
        else:
            changes.append(f"related video {rid} already exists")
    if params.add_placeholder:
        video.setdefault("related_videos", []).append({
            "video_id": f"PLACEHOLDER_{params.add_placeholder.replace(' ', '_').lower()}",
            "status": "placeholder",
            "topic": params.add_placeholder,
        })
        changes.append(f"added placeholder: {params.add_placeholder}")

    if not changes:
        return "No changes specified."

    video["updated_at"] = _now_iso()
    _write_json(VIDEOS_FILE, videos)
    return f"✅ Updated **{video['title']}**:\n" + "\n".join(f"  - {c}" for c in changes)


@mcp.tool(
    name="yt_search",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_search(params: SearchInput) -> str:
    """Full-text search across all video fields — title, tags, notes, category, guide URL, description.

    Args:
        params: Search query and response format

    Returns:
        str: Matching videos
    """
    videos = _read_json(VIDEOS_FILE)
    query = params.query.lower()
    matches = []

    for v in videos:
        searchable = " ".join([
            v.get("title") or "",
            v.get("category") or "",
            v.get("guide_url") or "",
            v.get("description") or "",
            " ".join(v.get("tags") or []),
            " ".join(v.get("notes") or []),
        ]).lower()
        if query in searchable:
            matches.append(v)

    if not matches:
        return f"No videos found matching: '{params.query}'"

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(matches, indent=2, default=str)

    lines = [f"## Search Results for '{params.query}' ({len(matches)} found)\n"]
    for v in matches:
        lines.append(f"- **{v['title']}** ({v['id']})")
        lines.append(f"  {v['url']} | Category: {v.get('category', '?')} | Guide: {v.get('guide_url', 'none')}")
    return "\n".join(lines)


# ============================================================
# Tools — Dependency & Cross-Link Management
# ============================================================


@mcp.tool(
    name="yt_get_placeholders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_get_placeholders(params: DashboardInput) -> str:
    """Return all videos with placeholder (unfilled) related video links.

    This is your placeholder debt queue. Run this after publishing any new video
    to check what descriptions need updating.

    Args:
        params: Response format

    Returns:
        str: List of videos with placeholders and what they're waiting for
    """
    videos = _read_json(VIDEOS_FILE)
    debts = []

    for v in videos:
        placeholders = [r for r in v.get("related_videos", []) if r.get("status") == "placeholder"]
        if placeholders:
            debts.append({"video": v, "placeholders": placeholders})

    if not debts:
        return "🎉 No placeholder debt! All related video links are filled."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps([{
            "video_id": d["video"]["id"],
            "title": d["video"]["title"],
            "placeholders": [p.get("topic", p["video_id"]) for p in d["placeholders"]],
        } for d in debts], indent=2)

    lines = [f"## Placeholder Debt ({sum(len(d['placeholders']) for d in debts)} unfilled links across {len(debts)} videos)\n"]
    for d in debts:
        lines.append(f"### {d['video']['title']}")
        lines.append(f"URL: {d['video']['url']}")
        for p in d["placeholders"]:
            topic = p.get("topic", p["video_id"])
            lines.append(f"  ⏳ Waiting for: **{topic}**")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="yt_suggest_crosslinks",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_suggest_crosslinks(params: SuggestCrosslinksInput) -> str:
    """Suggest which existing videos should cross-reference a given video.

    Based on shared tags, categories, and guide relationships. Use after publishing
    a new video to know which existing video descriptions need updating.

    Args:
        params: Video identifier and response format

    Returns:
        str: Suggested cross-links with reasoning
    """
    videos = _read_json(VIDEOS_FILE)
    target = _find_video(videos, params.identifier)
    if not target:
        return f"❌ No video found matching: '{params.identifier}'"

    target_tags = set(t.lower() for t in target.get("tags", []))
    target_cat = (target.get("category") or "").lower()
    target_id = target["id"]

    suggestions = []

    for v in videos:
        if v["id"] == target_id:
            continue

        score = 0
        reasons = []

        # Check tag overlap
        v_tags = set(t.lower() for t in v.get("tags", []))
        shared_tags = target_tags & v_tags
        if shared_tags:
            score += len(shared_tags) * 2
            reasons.append(f"shared tags: {', '.join(shared_tags)}")

        # Check same category
        v_cat = (v.get("category") or "").lower()
        if v_cat and v_cat == target_cat:
            score += 1
            reasons.append(f"same category: {v_cat}")

        # Check if already linked
        existing_related_ids = [r["video_id"] for r in v.get("related_videos", [])]
        already_linked = target_id in existing_related_ids

        if score > 0:
            suggestions.append({
                "video": v,
                "score": score,
                "reasons": reasons,
                "already_linked": already_linked,
            })

    suggestions.sort(key=lambda x: x["score"], reverse=True)

    if not suggestions:
        return f"No cross-link suggestions for '{target['title']}'. Consider adding more tags."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps([{
            "video_id": s["video"]["id"],
            "title": s["video"]["title"],
            "score": s["score"],
            "reasons": s["reasons"],
            "already_linked": s["already_linked"],
        } for s in suggestions], indent=2)

    lines = [f"## Cross-Link Suggestions for: {target['title']}\n"]
    for s in suggestions:
        status = "✅ Already linked" if s["already_linked"] else "🔗 **Should link**"
        lines.append(f"- {status} | **{s['video']['title']}**")
        lines.append(f"  Score: {s['score']} | {', '.join(s['reasons'])}")
        if not s["already_linked"]:
            lines.append(f"  → Add to {s['video']['title']}'s description: `{target['title']}: {target['url']}`")
    return "\n".join(lines)


@mcp.tool(
    name="yt_resolve_placeholder",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_resolve_placeholder(params: ResolvePlaceholderInput) -> str:
    """Replace a placeholder related video with an actual video URL.

    Use when a video that was previously a placeholder gets published.

    Args:
        params: The placeholder topic string and the actual video URL

    Returns:
        str: List of videos updated
    """
    videos = _read_json(VIDEOS_FILE)
    video_id = _extract_video_id(params.video_url)
    topic_lower = params.placeholder_topic.lower()
    updated = []

    for v in videos:
        for i, r in enumerate(v.get("related_videos", [])):
            if r.get("status") == "placeholder":
                r_topic = (r.get("topic") or r.get("video_id", "")).lower()
                if topic_lower in r_topic or r_topic in topic_lower:
                    v["related_videos"][i] = {"video_id": video_id, "status": "linked"}
                    v["updated_at"] = _now_iso()
                    updated.append(v["title"])

    if not updated:
        return f"❌ No placeholders found matching: '{params.placeholder_topic}'"

    _write_json(VIDEOS_FILE, videos)
    return f"✅ Resolved placeholder '{params.placeholder_topic}' → {video_id}\n\nUpdated {len(updated)} videos:\n" + "\n".join(f"  - {t}" for t in updated)


@mcp.tool(
    name="yt_get_crosslink_map",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_get_crosslink_map(params: ListVideosInput) -> str:
    """Show the full relationship graph for all videos or a specific category.

    Helps visualize which videos reference each other and find gaps.

    Args:
        params: Optional category filter and response format

    Returns:
        str: Cross-reference map
    """
    videos = _read_json(VIDEOS_FILE)

    if params.category:
        videos = [v for v in videos if (v.get("category") or "").lower() == params.category.lower()]

    if not videos:
        return "No videos found."

    # Build lookup
    id_to_title = {v["id"]: v["title"] for v in _read_json(VIDEOS_FILE)}

    lines = [f"## Cross-Reference Map ({len(videos)} videos)\n"]
    for v in videos:
        related = v.get("related_videos", [])
        linked = [r for r in related if r.get("status") == "linked"]
        placeholders = [r for r in related if r.get("status") == "placeholder"]

        lines.append(f"### {v['title']}")
        if linked:
            for r in linked:
                title = id_to_title.get(r["video_id"], r["video_id"])
                lines.append(f"  ✅ → {title}")
        if placeholders:
            for r in placeholders:
                topic = r.get("topic", r["video_id"])
                lines.append(f"  ⏳ → {topic} (placeholder)")
        if not linked and not placeholders:
            lines.append("  ⚠️ No related videos set")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="yt_flag_update_needed",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def yt_flag_update_needed(params: FlagUpdateInput) -> str:
    """Flag a video (or videos matching a topic) as needing a description update.

    Use when news breaks, podcasts cover new angles, or content changes that affect
    existing videos. Pass 'topic:keyword' to flag all videos matching that tag or category.

    Args:
        params: Video identifier (or topic:keyword), reason, and priority

    Returns:
        str: Confirmation of flagged videos
    """
    videos = _read_json(VIDEOS_FILE)
    queue = _read_json(UPDATE_QUEUE_FILE)

    flagged = []

    if params.identifier.startswith("topic:"):
        keyword = params.identifier.split("topic:", 1)[1].strip().lower()
        for v in videos:
            searchable = " ".join([
                v.get("category", ""),
                " ".join(v.get("tags", [])),
                v.get("title", ""),
            ]).lower()
            if keyword in searchable:
                flagged.append(v)
    else:
        video = _find_video(videos, params.identifier)
        if video:
            flagged.append(video)

    if not flagged:
        return f"❌ No videos found matching: '{params.identifier}'"

    for v in flagged:
        # Check if already in queue
        existing = [q for q in queue if q["video_id"] == v["id"] and q.get("resolved") is not True]
        if not existing:
            queue.append({
                "video_id": v["id"],
                "video_title": v["title"],
                "reason": params.reason,
                "priority": params.priority or "normal",
                "flagged_at": _now_iso(),
                "resolved": False,
            })

    _write_json(UPDATE_QUEUE_FILE, queue)
    return f"🚩 Flagged {len(flagged)} video(s) for update:\n" + "\n".join(
        f"  - {v['title']} ({params.priority}): {params.reason}" for v in flagged
    )


@mcp.tool(
    name="yt_get_update_queue",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_get_update_queue(params: DashboardInput) -> str:
    """Return all videos flagged for description updates, sorted by priority.

    This is your to-do list for video maintenance.

    Args:
        params: Response format

    Returns:
        str: Pending updates sorted by priority
    """
    queue = _read_json(UPDATE_QUEUE_FILE)
    pending = [q for q in queue if not q.get("resolved")]

    if not pending:
        return "🎉 Update queue is empty! No videos need attention."

    # Sort: high > normal > low
    priority_order = {"high": 0, "normal": 1, "low": 2}
    pending.sort(key=lambda q: priority_order.get(q.get("priority", "normal"), 1))

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(pending, indent=2, default=str)

    lines = [f"## Update Queue ({len(pending)} pending)\n"]
    for q in pending:
        priority_icon = {"high": "🔴", "normal": "🟡", "low": "🟢"}.get(q.get("priority", "normal"), "🟡")
        lines.append(f"{priority_icon} **{q['video_title']}**")
        lines.append(f"  Reason: {q['reason']}")
        lines.append(f"  Flagged: {q['flagged_at'][:10]}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="yt_resolve_update",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_resolve_update(params: VideoIdentifierInput) -> str:
    """Mark an update queue item as resolved after updating the video description.

    Args:
        params: Video identifier to mark as resolved

    Returns:
        str: Confirmation
    """
    queue = _read_json(UPDATE_QUEUE_FILE)
    videos = _read_json(VIDEOS_FILE)
    video = _find_video(videos, params.identifier)

    if not video:
        return f"❌ No video found matching: '{params.identifier}'"

    resolved_count = 0
    for q in queue:
        if q["video_id"] == video["id"] and not q.get("resolved"):
            q["resolved"] = True
            q["resolved_at"] = _now_iso()
            resolved_count += 1

    if resolved_count == 0:
        return f"No pending updates found for: {video['title']}"

    _write_json(UPDATE_QUEUE_FILE, queue)
    return f"✅ Resolved {resolved_count} update(s) for: {video['title']}"


# ============================================================
# Tools — Content Coverage
# ============================================================


@mcp.tool(
    name="yt_register_content",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def yt_register_content(params: RegisterContentInput) -> str:
    """Register a guide, demo, tool, or blog post as potential video content.

    Tracks which FixMyCert content has videos and which doesn't.

    Args:
        params: Content details including path, title, type, category, priority

    Returns:
        str: Confirmation
    """
    content_map = _read_json(CONTENT_MAP_FILE)

    # Check for duplicate
    for c in content_map:
        if c["path"] == params.path:
            return f"⚠️ Content already registered: {params.path}. Use yt_update_video to modify."

    # Check if a video already exists for this path
    videos = _read_json(VIDEOS_FILE)
    matching_video = None
    for v in videos:
        if v.get("guide_url") == params.path:
            matching_video = v
            break

    entry = {
        "path": params.path,
        "title": params.title,
        "type": params.content_type,
        "category": params.category,
        "has_video": matching_video is not None,
        "video_id": matching_video["id"] if matching_video else None,
        "priority": params.priority or "medium",
        "notes": params.notes,
        "created_at": _now_iso(),
    }

    content_map.append(entry)
    _write_json(CONTENT_MAP_FILE, content_map)

    status = f"(already has video: {matching_video['title']})" if matching_video else "(no video yet)"
    return f"✅ Registered: {params.title} — {params.path} {status}"


@mcp.tool(
    name="yt_get_coverage_gaps",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_get_coverage_gaps(params: DashboardInput) -> str:
    """Show content that doesn't have videos yet, sorted by priority.

    Args:
        params: Response format

    Returns:
        str: Content pieces without videos, sorted by priority
    """
    content_map = _read_json(CONTENT_MAP_FILE)
    gaps = [c for c in content_map if not c.get("has_video")]

    if not gaps:
        if not content_map:
            return "📋 No content registered yet. Use yt_register_content to add guides/demos that should have videos."
        return "🎉 All registered content has videos!"

    priority_order = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda c: priority_order.get(c.get("priority", "medium"), 1))

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(gaps, indent=2, default=str)

    lines = [f"## Content Coverage Gaps ({len(gaps)} without videos)\n"]
    for g in gaps:
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(g.get("priority", "medium"), "🟡")
        lines.append(f"{priority_icon} **{g['title']}**")
        lines.append(f"  Path: {g['path']} | Type: {g.get('type', '?')} | Category: {g.get('category', '?')}")
        if g.get("notes"):
            lines.append(f"  Notes: {g['notes']}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Tools — A/B Testing
# ============================================================


@mcp.tool(
    name="yt_start_ab_test",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def yt_start_ab_test(params: ABTestInput) -> str:
    """Record that an A/B test started on a video thumbnail or title.

    Args:
        params: Video identifier and what's being tested

    Returns:
        str: Confirmation
    """
    videos = _read_json(VIDEOS_FILE)
    video = _find_video(videos, params.identifier)
    if not video:
        return f"❌ No video found matching: '{params.identifier}'"

    video["ab_test"] = {
        "status": "active",
        "element": params.element or "thumbnail",
        "start_date": _now_iso()[:10],
        "end_date": None,
        "result": None,
    }
    video["updated_at"] = _now_iso()
    _write_json(VIDEOS_FILE, videos)
    return f"🧪 A/B test started on **{video['title']}** — testing: {params.element or 'thumbnail'}"


@mcp.tool(
    name="yt_end_ab_test",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def yt_end_ab_test(params: ABTestInput) -> str:
    """Record A/B test completion and result.

    Args:
        params: Video identifier and result description

    Returns:
        str: Confirmation
    """
    videos = _read_json(VIDEOS_FILE)
    video = _find_video(videos, params.identifier)
    if not video:
        return f"❌ No video found matching: '{params.identifier}'"

    ab = video.get("ab_test", {})
    if ab.get("status") != "active":
        return f"⚠️ No active A/B test on: {video['title']}"

    video["ab_test"]["status"] = "completed"
    video["ab_test"]["end_date"] = _now_iso()[:10]
    video["ab_test"]["result"] = params.result or "no result recorded"
    video["updated_at"] = _now_iso()
    _write_json(VIDEOS_FILE, videos)
    return f"✅ A/B test completed on **{video['title']}** — result: {params.result}"


@mcp.tool(
    name="yt_get_ab_tests",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_get_ab_tests(params: DashboardInput) -> str:
    """Show all active and completed A/B tests.

    Args:
        params: Response format

    Returns:
        str: A/B test status for all videos with tests
    """
    videos = _read_json(VIDEOS_FILE)
    tests = [v for v in videos if v.get("ab_test", {}).get("status", "none") != "none"]

    if not tests:
        return "No A/B tests recorded."

    active = [v for v in tests if v["ab_test"]["status"] == "active"]
    completed = [v for v in tests if v["ab_test"]["status"] == "completed"]

    lines = []
    if active:
        lines.append(f"## Active A/B Tests ({len(active)})\n")
        for v in active:
            ab = v["ab_test"]
            lines.append(f"🧪 **{v['title']}** — testing: {ab.get('element', '?')} (since {ab.get('start_date', '?')})")
        lines.append("")

    if completed:
        lines.append(f"## Completed A/B Tests ({len(completed)})\n")
        for v in completed:
            ab = v["ab_test"]
            lines.append(f"✅ **{v['title']}** — {ab.get('element', '?')}: {ab.get('result', 'no result')}")
            lines.append(f"  {ab.get('start_date', '?')} → {ab.get('end_date', '?')}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Tools — Dashboard
# ============================================================


@mcp.tool(
    name="yt_dashboard",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_dashboard(params: DashboardInput) -> str:
    """High-level overview of the YouTube channel status.

    Shows: total videos, placeholder debt, update queue, coverage gaps, active A/B tests.
    Run this as a weekly check-in to see what needs attention.

    Args:
        params: Response format

    Returns:
        str: Dashboard summary
    """
    videos = _read_json(VIDEOS_FILE)
    queue = _read_json(UPDATE_QUEUE_FILE)
    content_map = _read_json(CONTENT_MAP_FILE)

    total_videos = len(videos)

    # Placeholder debt
    placeholder_count = 0
    videos_with_placeholders = 0
    for v in videos:
        ph = [r for r in v.get("related_videos", []) if r.get("status") == "placeholder"]
        if ph:
            placeholder_count += len(ph)
            videos_with_placeholders += 1

    # Update queue
    pending_updates = [q for q in queue if not q.get("resolved")]
    high_priority = [q for q in pending_updates if q.get("priority") == "high"]

    # Coverage
    content_without_video = [c for c in content_map if not c.get("has_video")]

    # A/B tests
    active_tests = [v for v in videos if v.get("ab_test", {}).get("status") == "active"]

    # Categories
    categories: dict[str, int] = {}
    for v in videos:
        cat = v.get("category", "uncategorized") or "uncategorized"
        categories[cat] = categories.get(cat, 0) + 1

    # Videos with no related videos at all
    no_related = [v for v in videos if not v.get("related_videos")]

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "total_videos": total_videos,
            "placeholder_debt": {"total_placeholders": placeholder_count, "videos_affected": videos_with_placeholders},
            "update_queue": {"pending": len(pending_updates), "high_priority": len(high_priority)},
            "coverage_gaps": len(content_without_video),
            "active_ab_tests": len(active_tests),
            "categories": categories,
            "videos_with_no_related": len(no_related),
        }, indent=2)

    lines = [
        "# 📊 FixMyCert YouTube Dashboard",
        f"\n**Total Videos:** {total_videos}",
        f"**Categories:** {', '.join(f'{k} ({v})' for k, v in sorted(categories.items()))}",
        "",
    ]

    # Action items
    action_items = []
    if placeholder_count > 0:
        action_items.append(f"⏳ **{placeholder_count} placeholder links** across {videos_with_placeholders} videos need real URLs")
    if pending_updates:
        item = f"🚩 **{len(pending_updates)} videos** need description updates"
        if high_priority:
            item += f" ({len(high_priority)} high priority)"
        action_items.append(item)
    if content_without_video:
        action_items.append(f"📹 **{len(content_without_video)} content pieces** registered without videos")
    if active_tests:
        action_items.append(f"🧪 **{len(active_tests)} A/B test(s)** running")
    if no_related:
        action_items.append(f"⚠️ **{len(no_related)} videos** have no related video links at all")

    if action_items:
        lines.append("## ⚡ Action Items\n")
        for item in action_items:
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines.append("## ✅ All Clear!\nNo immediate action items. Keep creating! 🚀\n")

    # Community Posts summary
    posts = _read_json(POSTS_FILE)
    if posts:
        published_posts = [p for p in posts if p.get("status") == "published"]
        by_type: dict[str, int] = {}
        for p in posts:
            t = p.get("post_type", "text")
            by_type[t] = by_type.get(t, 0) + 1

        published_posts.sort(key=lambda p: p.get("published_date", ""), reverse=True)
        last_date = published_posts[0].get("published_date", "???") if published_posts else "never"

        this_month = datetime.now(timezone.utc).strftime("%Y-%m")
        month_count = sum(1 for p in published_posts if p.get("published_date", "").startswith(this_month))

        type_parts = ", ".join(f"{c} {t}" for t, c in sorted(by_type.items()))
        lines.append("## 📝 Community Posts")
        lines.append(f"- Total: {len(posts)} posts ({type_parts})")
        lines.append(f"- Last post: {last_date}")
        lines.append(f"- This month: {month_count} posts")
        lines.append(f"- Suggested cadence: 2-3 posts/week")
    else:
        lines.append("## 📝 Community Posts")
        lines.append("- No posts tracked yet. Use `yt_register_post` to start.")

    return "\n".join(lines)


# ============================================================
# Seed data tool
# ============================================================


@mcp.tool(
    name="yt_seed_data",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": False},
)
async def yt_seed_data(params: DashboardInput) -> str:
    """Initialize the database with all 17 known FixMyCert videos.

    ⚠️ This will overwrite existing video data. Only run on first setup.

    Args:
        params: Response format (ignored, always returns markdown)

    Returns:
        str: Confirmation with seeded video count
    """
    existing = _read_json(VIDEOS_FILE)
    if existing:
        return f"⚠️ Database already has {len(existing)} videos. Delete {VIDEOS_FILE} first if you want to re-seed."

    seed_videos = [
        {
            "id": "TlGXMTIwU8I",
            "title": "47-Day SSL Certificates Explained (2026-2029 Timeline)",
            "url": "https://www.youtube.com/watch?v=TlGXMTIwU8I",
            "published_date": "2026-01",
            "guide_url": "/guides/47-day-certificate-timeline",
            "category": "compliance",
            "tags": ["47-day", "certificate-validity", "automation", "ACME", "SC-081", "timeline"],
        },
        {
            "id": "wCFnpj4zntU",
            "title": "The DCV Methods Sunset - 11 Validation Methods Gone by 2028",
            "url": "https://www.youtube.com/watch?v=wCFnpj4zntU",
            "published_date": "2026-01",
            "guide_url": "/guides/dcv-methods-sunset",
            "category": "compliance",
            "tags": ["dcv", "domain-validation", "sunset", "SC-090", "email-validation"],
        },
        {
            "id": "M4T7bVcjUjk",
            "title": "Which DCV Method Should You Automate? (Decision Framework)",
            "url": "https://www.youtube.com/watch?v=M4T7bVcjUjk",
            "published_date": "2026-02",
            "guide_url": "/guides/which-dcv-method-to-automate",
            "category": "compliance",
            "tags": ["dcv", "automation", "decision-framework", "ACME", "dns-01", "http-01", "tls-alpn-01"],
        },
        {
            "id": "64JHWgrCHpU",
            "title": "DNS-01 Automation - Secure Your Certs Without Exposing Your Zone",
            "url": "https://www.youtube.com/watch?v=64JHWgrCHpU",
            "published_date": "2026-02",
            "guide_url": "/guides/dns-01-automation",
            "category": "compliance",
            "tags": ["dns-01", "automation", "ACME", "wildcard", "dns-api"],
        },
        {
            "id": "MMTW4CrcYdE",
            "title": "HTTP-01 Automation on Nginx, Apache, and IIS - The Simplest ACME Setup",
            "url": "https://www.youtube.com/watch?v=MMTW4CrcYdE",
            "published_date": "2026-02",
            "guide_url": "/guides/http-01-automation",
            "category": "compliance",
            "tags": ["http-01", "automation", "ACME", "nginx", "apache", "IIS"],
        },
        {
            "id": "8v9veID3A_Q",
            "title": "TLS-ALPN-01 Explained - No Files, No DNS, Just TLS",
            "url": "https://www.youtube.com/watch?v=8v9veID3A_Q",
            "published_date": "2026-02",
            "guide_url": "/guides/tls-alpn-01-automation",
            "category": "compliance",
            "tags": ["tls-alpn-01", "automation", "ACME", "caddy"],
        },
        {
            "id": "hNFrbqWvZzI",
            "title": "AWS ACM Explained - 3 Secrets That Save You Hours",
            "url": "https://www.youtube.com/watch?v=hNFrbqWvZzI",
            "published_date": "2026-01",
            "guide_url": "/guides/aws-acm-certificate-manager",
            "category": "cdn",
            "tags": ["aws", "acm", "cloud", "certificate-manager", "cloudfront", "route53"],
        },
        {
            "id": "m0hpo10ZvF4",
            "title": "Cloudflare Error 526 - Fix Invalid SSL Certificate in Minutes",
            "url": "https://www.youtube.com/watch?v=m0hpo10ZvF4",
            "published_date": "2025-12",
            "guide_url": "/guides/cloudflare-error-526",
            "category": "troubleshooting",
            "tags": ["cloudflare", "error-526", "ssl-error", "origin-certificate"],
        },
        {
            "id": "QHGFifNh3BQ",
            "title": "How TLS Works - Handshake, Key Exchange & Encryption Explained",
            "url": "https://www.youtube.com/watch?v=QHGFifNh3BQ",
            "published_date": "2026-02",
            "guide_url": "/demos/tls-handshake",
            "category": "fundamentals",
            "tags": ["tls", "handshake", "key-exchange", "encryption", "ssl"],
        },
        {
            "id": "OCZG3BbT37g",
            "title": "Certificate Practice Statements - Why Your Internal CA Needs One",
            "url": "https://www.youtube.com/watch?v=OCZG3BbT37g",
            "published_date": "2026-02",
            "guide_url": "/guides/what-is-a-cps",
            "category": "governance",
            "tags": ["cps", "certificate-policy", "rfc-3647", "compliance", "governance", "ca"],
        },
        {
            "id": "vSLzCrJ68q8",
            "title": "PKI Compliance Hub — Every Certificate Deadline Through 2035",
            "url": "https://www.youtube.com/watch?v=vSLzCrJ68q8",
            "published_date": "2026-02",
            "guide_url": "/compliance",
            "category": "compliance",
            "tags": ["compliance", "deadlines", "47-day", "200-day", "pqc", "post-quantum", "CA-browser-forum", "NIST", "NIS2", "DORA"],
        },
        {
            "id": "9s3QfTCi7ow",
            "title": "S/MIME Email Security — How Certificate-Based Email Signing Works",
            "url": "https://www.youtube.com/watch?v=9s3QfTCi7ow",
            "published_date": "2026-02",
            "guide_url": "/guides/smime-email",
            "category": "email",
            "tags": ["smime", "email-security", "digital-signatures", "email-encryption", "phishing"],
        },
        {
            "id": "yNwqUCIWuFw",
            "title": "Email Phishing and PKI — The $250K Security Suite That Lost to Hotmail",
            "url": "https://www.youtube.com/watch?v=yNwqUCIWuFw",
            "published_date": "2026-02",
            "guide_url": "/blog/email-phishing-pki-solutions-already-exist",
            "category": "email",
            "tags": ["email-security", "phishing", "dmarc", "dkim", "spf", "smime", "mta-sts"],
        },
        {
            "id": "5BNGZSy-4fc",
            "title": "Certificate Anatomy - Every X.509 Field Explained",
            "url": "https://www.youtube.com/watch?v=5BNGZSy-4fc",
            "published_date": "2026-02",
            "guide_url": "/demos/certificate-anatomy",
            "category": "fundamentals",
            "tags": ["x509", "certificate-anatomy", "certificate-fields", "san", "key-usage"],
        },
        {
            "id": "yH3N5dhb26w",
            "title": "PKI Compliance Documentation — What Auditors Actually Look For",
            "url": "https://www.youtube.com/watch?v=yH3N5dhb26w",
            "published_date": "2026-02",
            "guide_url": None,
            "category": "governance",
            "tags": ["compliance", "audit", "documentation", "cps", "certificate-policy", "governance"],
        },
        {
            "id": "ap6ylX7yf0U",
            "title": "Azure Key Vault Certificates - 5 Gotchas Nobody Warns You About",
            "url": "https://www.youtube.com/watch?v=ap6ylX7yf0U",
            "published_date": "2026-02",
            "guide_url": "/guides/azure-key-vault-certificates",
            "category": "cdn",
            "tags": ["azure", "key-vault", "cloud", "certificate-management", "gotchas"],
        },
        {
            "id": "ir86n4r1LkQ",
            "title": "Certificate Chain Builder - Fix Incomplete Chain Errors Fast",
            "url": "https://www.youtube.com/watch?v=ir86n4r1LkQ",
            "published_date": "2026-02",
            "guide_url": "/demos/chain-builder",
            "category": "troubleshooting",
            "tags": ["chain-builder", "incomplete-chain", "intermediate-certificate", "ssl-error", "trust-chain"],
        },
    ]

    # Add common fields
    now = _now_iso()
    for v in seed_videos:
        v.setdefault("related_videos", [])
        v.setdefault("description", None)
        v.setdefault("pinned_comment", None)
        v.setdefault("thumbnail_type", None)
        v.setdefault("ab_test", {"status": "none", "element": None, "start_date": None, "end_date": None, "result": None})
        v.setdefault("notes", [])
        v["created_at"] = now
        v["updated_at"] = now

    _write_json(VIDEOS_FILE, seed_videos)

    # Also initialize empty content map and update queue
    if not _read_json(CONTENT_MAP_FILE):
        _write_json(CONTENT_MAP_FILE, [])
    if not _read_json(UPDATE_QUEUE_FILE):
        _write_json(UPDATE_QUEUE_FILE, [])

    return f"✅ Seeded {len(seed_videos)} videos into the database.\n\nRun `yt_dashboard` to see the overview.\nRun `yt_get_placeholders` to see what needs linking."


# ============================================================
# YouTube Data API helpers
# ============================================================


def _check_api_key() -> str | None:
    """Return API key or None if not set."""
    key = os.environ.get("YOUTUBE_API_KEY", "")
    return key if key else None


async def _yt_api_get(endpoint: str, params: dict) -> dict:
    """Make a YouTube Data API GET request."""
    key = _check_api_key()
    if not key:
        raise ValueError("YOUTUBE_API_KEY environment variable not set. Set it and restart the server.")
    params["key"] = key
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{YOUTUBE_API_BASE}/{endpoint}", params=params)
        resp.raise_for_status()
        return resp.json()


async def _fetch_channel_videos(channel_id: str = None) -> list[dict]:
    """Fetch all videos from the channel using the YouTube Data API.

    If no channel_id provided, uses 'mine' which requires OAuth (won't work with API key).
    With API key, we search by channel ID.
    """
    # First, get channel info if we don't have an ID — search for FixMyCert
    if not channel_id:
        result = await _yt_api_get("search", {
            "part": "snippet",
            "q": "FixMyCert",
            "type": "channel",
            "maxResults": 1,
        })
        items = result.get("items", [])
        if not items:
            raise ValueError("Could not find FixMyCert channel. Provide channel_id directly.")
        channel_id = items[0]["snippet"]["channelId"]

    # Fetch all videos from the channel
    all_videos = []
    page_token = None

    while True:
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "order": "date",
            "maxResults": 50,
        }
        if page_token:
            params["pageToken"] = page_token

        result = await _yt_api_get("search", params)
        items = result.get("items", [])
        all_videos.extend(items)

        page_token = result.get("nextPageToken")
        if not page_token:
            break

    return all_videos


async def _fetch_video_details(video_ids: list[str]) -> list[dict]:
    """Fetch detailed video info (description, stats, etc.) for a list of video IDs."""
    details = []
    # API allows up to 50 IDs per request
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        result = await _yt_api_get("videos", {
            "part": "snippet,statistics,status",
            "id": ",".join(chunk),
        })
        details.extend(result.get("items", []))
    return details


# ============================================================
# Input models — YouTube API
# ============================================================


class SyncInput(BaseModel):
    """Input for syncing with YouTube API."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    channel_id: Optional[str] = Field(default=None, description="YouTube channel ID. If not provided, searches for 'FixMyCert'.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class RefreshVideoInput(BaseModel):
    """Input for refreshing a single video from YouTube API."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    identifier: str = Field(..., description="Video ID, URL, or title substring")


class ChannelStatsInput(BaseModel):
    """Input for getting channel stats."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    channel_id: Optional[str] = Field(default=None, description="YouTube channel ID. If not provided, searches for 'FixMyCert'.")


# ============================================================
# Tools — YouTube API Sync
# ============================================================


@mcp.tool(
    name="yt_sync_from_youtube",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def yt_sync_from_youtube(params: SyncInput) -> str:
    """Sync video data from YouTube Data API.

    Pulls all videos from the channel and updates the local database with
    current titles, descriptions, view counts, and publish dates. New videos
    found on YouTube but not in the local DB are flagged for registration.

    Existing local data (guide_url, tags, category, related_videos, notes)
    is preserved — only YouTube-sourced fields are updated.

    Requires YOUTUBE_API_KEY environment variable.

    Args:
        params: Optional channel ID and response format

    Returns:
        str: Summary of sync results — new videos found, fields updated, etc.
    """
    if not _check_api_key():
        return "❌ YOUTUBE_API_KEY environment variable not set. Set it and restart the server."

    try:
        # Fetch videos from YouTube
        yt_videos = await _fetch_channel_videos(params.channel_id)
    except Exception as e:
        return f"❌ YouTube API error: {str(e)}"

    if not yt_videos:
        return "⚠️ No videos found on the channel."

    # Get detailed info for all found videos
    video_ids = [v["id"]["videoId"] for v in yt_videos if v.get("id", {}).get("videoId")]
    try:
        details = await _fetch_video_details(video_ids)
    except Exception as e:
        return f"❌ Error fetching video details: {str(e)}"

    # Build lookup by ID
    detail_map = {d["id"]: d for d in details}

    # Load local database
    videos = _read_json(VIDEOS_FILE)
    local_map = {v["id"]: v for v in videos}

    new_videos = []
    updated_videos = []
    unchanged = 0

    for vid in video_ids:
        detail = detail_map.get(vid)
        if not detail:
            continue

        snippet = detail.get("snippet", {})
        stats = detail.get("statistics", {})

        yt_title = snippet.get("title", "")
        yt_description = snippet.get("description", "")
        yt_published = snippet.get("publishedAt", "")[:10]
        yt_views = int(stats.get("viewCount", 0))
        yt_likes = int(stats.get("likeCount", 0))
        yt_comments = int(stats.get("commentCount", 0))

        if vid in local_map:
            local = local_map[vid]
            changes = []

            # Update YouTube-sourced fields only
            if yt_title and yt_title != local.get("title"):
                changes.append(f"title: '{local.get('title')}' → '{yt_title}'")
                local["title"] = yt_title

            if yt_published and yt_published != local.get("published_date"):
                local["published_date"] = yt_published

            # Store current description from YouTube for comparison
            local["yt_description"] = yt_description

            # Update stats
            local["stats"] = {
                "views": yt_views,
                "likes": yt_likes,
                "comments": yt_comments,
                "last_synced": _now_iso(),
            }

            if changes:
                local["updated_at"] = _now_iso()
                updated_videos.append({"title": yt_title, "changes": changes})
            else:
                unchanged += 1
        else:
            # New video not in local DB
            new_videos.append({
                "id": vid,
                "title": yt_title,
                "url": _canonical_url(vid),
                "published_date": yt_published,
                "guide_url": None,
                "category": None,
                "tags": [],
                "related_videos": [],
                "description": None,
                "yt_description": yt_description,
                "pinned_comment": None,
                "thumbnail_type": None,
                "ab_test": {"status": "none", "element": None, "start_date": None, "end_date": None, "result": None},
                "notes": [f"[{_now_iso()[:10]}] Auto-discovered via YouTube API sync"],
                "stats": {
                    "views": yt_views,
                    "likes": yt_likes,
                    "comments": yt_comments,
                    "last_synced": _now_iso(),
                },
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            })

    # Add new videos to local DB
    for nv in new_videos:
        videos.append(nv)

    _write_json(VIDEOS_FILE, videos)

    # Build report
    lines = [
        "# 🔄 YouTube Sync Complete\n",
        f"**Channel videos found:** {len(video_ids)}",
        f"**Local database:** {len(videos)} videos total\n",
    ]

    if new_videos:
        lines.append(f"## 🆕 New Videos Discovered ({len(new_videos)})\n")
        for nv in new_videos:
            lines.append(f"- **{nv['title']}** ({nv['id']})")
            lines.append(f"  ⚠️ Needs: category, tags, guide_url, related_videos")
        lines.append("\n💡 Use `yt_update_video` to fill in the missing metadata for each new video.")

    if updated_videos:
        lines.append(f"\n## ✏️ Updated ({len(updated_videos)})\n")
        for uv in updated_videos:
            lines.append(f"- **{uv['title']}**: {', '.join(uv['changes'])}")

    if unchanged:
        lines.append(f"\n✅ {unchanged} videos unchanged")

    return "\n".join(lines)


@mcp.tool(
    name="yt_refresh_video",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def yt_refresh_video(params: RefreshVideoInput) -> str:
    """Refresh a single video's data from the YouTube API.

    Pulls current title, description, and stats for one video.
    Use after updating a description in YouTube Studio to verify the change.

    Requires YOUTUBE_API_KEY environment variable.

    Args:
        params: Video identifier (ID, URL, or title substring)

    Returns:
        str: Updated video details with what changed
    """
    if not _check_api_key():
        return "❌ YOUTUBE_API_KEY environment variable not set."

    videos = _read_json(VIDEOS_FILE)
    video = _find_video(videos, params.identifier)
    if not video:
        return f"❌ No video found matching: '{params.identifier}'"

    try:
        details = await _fetch_video_details([video["id"]])
    except Exception as e:
        return f"❌ YouTube API error: {str(e)}"

    if not details:
        return f"⚠️ Video {video['id']} not found on YouTube."

    detail = details[0]
    snippet = detail.get("snippet", {})
    stats = detail.get("statistics", {})

    changes = []

    yt_title = snippet.get("title", "")
    if yt_title and yt_title != video.get("title"):
        changes.append(f"title: '{video.get('title')}' → '{yt_title}'")
        video["title"] = yt_title

    yt_desc = snippet.get("description", "")
    old_yt_desc = video.get("yt_description", "")
    if yt_desc != old_yt_desc:
        changes.append("YouTube description changed")
    video["yt_description"] = yt_desc

    video["stats"] = {
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0)),
        "last_synced": _now_iso(),
    }

    video["updated_at"] = _now_iso()
    _write_json(VIDEOS_FILE, videos)

    lines = [
        f"### {video['title']}",
        f"**Views:** {video['stats']['views']:,} | **Likes:** {video['stats']['likes']:,} | **Comments:** {video['stats']['comments']:,}",
        f"**Synced:** {video['stats']['last_synced'][:16]}",
    ]

    if changes:
        lines.append(f"\n**Changes detected:**")
        for c in changes:
            lines.append(f"  - {c}")
    else:
        lines.append("\n✅ No changes from YouTube")

    return "\n".join(lines)


@mcp.tool(
    name="yt_check_descriptions",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def yt_check_descriptions(params: DashboardInput) -> str:
    """Compare local description records against what's actually on YouTube.

    Pulls current descriptions from YouTube and flags any videos where
    the live description differs from what we have stored locally. Useful
    for catching descriptions you updated in YouTube Studio but forgot
    to update in the tracker, or vice versa.

    Requires YOUTUBE_API_KEY environment variable.

    Args:
        params: Response format

    Returns:
        str: List of description mismatches
    """
    if not _check_api_key():
        return "❌ YOUTUBE_API_KEY environment variable not set."

    videos = _read_json(VIDEOS_FILE)
    if not videos:
        return "No videos in database."

    video_ids = [v["id"] for v in videos]

    try:
        details = await _fetch_video_details(video_ids)
    except Exception as e:
        return f"❌ YouTube API error: {str(e)}"

    detail_map = {d["id"]: d for d in details}

    mismatches = []
    no_local_desc = []
    matched = 0

    for v in videos:
        detail = detail_map.get(v["id"])
        if not detail:
            continue

        yt_desc = detail.get("snippet", {}).get("description", "")
        local_desc = v.get("description") or v.get("yt_description") or ""

        if not local_desc:
            no_local_desc.append(v["title"])
        elif yt_desc.strip() != local_desc.strip():
            mismatches.append({
                "title": v["title"],
                "id": v["id"],
                "local_len": len(local_desc),
                "youtube_len": len(yt_desc),
            })
        else:
            matched += 1

        # Always update the yt_description field
        v["yt_description"] = yt_desc

    _write_json(VIDEOS_FILE, videos)

    lines = [f"## Description Check ({len(videos)} videos)\n"]

    if mismatches:
        lines.append(f"### ⚠️ Mismatches ({len(mismatches)})\n")
        for m in mismatches:
            lines.append(f"- **{m['title']}** — local: {m['local_len']} chars, YouTube: {m['youtube_len']} chars")
        lines.append("\nUse `yt_refresh_video` on each to pull the latest from YouTube.")

    if no_local_desc:
        lines.append(f"\n### 📝 No Local Description Stored ({len(no_local_desc)})\n")
        for t in no_local_desc:
            lines.append(f"- {t}")
        lines.append("\nRun `yt_sync_from_youtube` to pull descriptions from YouTube.")

    if matched:
        lines.append(f"\n✅ {matched} videos have matching descriptions")

    return "\n".join(lines)


# ============================================================
# Community Posts helpers
# ============================================================


def _load_posts() -> list[dict]:
    """Load community posts from storage."""
    return _read_json(POSTS_FILE)


def _save_posts(posts: list[dict]) -> None:
    """Save community posts to storage."""
    _write_json(POSTS_FILE, posts)


def _next_post_id(posts: list[dict]) -> str:
    """Generate the next post ID (date-based, sequential)."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    today_posts = [p for p in posts if p.get("post_id", "").startswith(f"post_{today}")]
    seq = len(today_posts) + 1
    return f"post_{today}_{seq:03d}"


def _find_post(posts: list[dict], identifier: str) -> dict | None:
    """Find a post by ID or content substring."""
    for p in posts:
        if p.get("post_id") == identifier:
            return p
    identifier_lower = identifier.lower()
    for p in posts:
        if identifier_lower in p.get("content", "").lower():
            return p
    return None


def _format_post_md(post: dict) -> str:
    """Format a single post as markdown."""
    lines = []
    pid = post.get("post_id", "???")
    ptype = post.get("post_type", "text")
    status = post.get("status", "published")
    date = post.get("published_date", "???")

    type_icons = {"text": "📝", "poll": "📊", "image": "🖼️", "link": "🔗"}
    icon = type_icons.get(ptype, "📝")

    lines.append(f"### {icon} {pid} ({ptype} | {status} | {date})")
    lines.append(f"{post.get('content', '')}")

    if post.get("poll_options"):
        for opt in post["poll_options"]:
            lines.append(f"  - {opt}")

    if post.get("link_url"):
        lines.append(f"🔗 Link: {post['link_url']}")
    if post.get("linked_video_id"):
        lines.append(f"🎬 Video: https://www.youtube.com/watch?v={post['linked_video_id']}")
    if post.get("linked_content_path"):
        lines.append(f"📚 Guide: https://fixmycert.com{post['linked_content_path']}")
    if post.get("tags"):
        lines.append(f"🏷️ Tags: {', '.join(post['tags'])}")
    if post.get("engagement_notes"):
        lines.append(f"📈 Engagement: {post['engagement_notes']}")
    if post.get("notes"):
        for note in post["notes"]:
            lines.append(f"📌 {note}")

    return "\n".join(lines)


# ============================================================
# Tools — Community Posts
# ============================================================


@mcp.tool(
    name="yt_register_post",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def yt_register_post(params: RegisterPostInput) -> str:
    """Register a new community post (published, draft, or scheduled).

    Use this when the user publishes a community post on YouTube or
    wants to save a draft for later.

    Args:
        params: Post details including type, content, links, and status.

    Returns:
        str: Confirmation with post ID.
    """
    posts = _load_posts()

    if params.post_type == "poll" and not params.poll_options:
        return "❌ Poll posts require poll_options. Provide 2-4 choices."
    if params.poll_options and len(params.poll_options) < 2:
        return "❌ Polls need at least 2 options."

    post_id = _next_post_id(posts)
    post = {
        "post_id": post_id,
        "post_type": params.post_type,
        "content": params.content,
        "poll_options": params.poll_options,
        "link_url": params.link_url,
        "linked_video_id": params.linked_video_id,
        "linked_content_path": params.linked_content_path,
        "category": params.category,
        "tags": params.tags or [],
        "status": params.status,
        "published_date": params.published_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "engagement_notes": None,
        "notes": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    posts.append(post)
    _save_posts(posts)

    return f"✅ Registered community post: **{post_id}** ({params.post_type})\n\n{_format_post_md(post)}"


@mcp.tool(
    name="yt_list_posts",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_list_posts(params: ListPostsInput) -> str:
    """List all community posts, optionally filtered by type, status, or category.

    Args:
        params: Optional filters and response format.

    Returns:
        str: List of matching posts.
    """
    posts = _load_posts()

    if params.post_type:
        posts = [p for p in posts if p.get("post_type") == params.post_type.lower()]
    if params.status:
        posts = [p for p in posts if p.get("status") == params.status.lower()]
    if params.category:
        posts = [p for p in posts if p.get("category") == params.category.lower()]

    if not posts:
        return "No community posts found matching those filters."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(posts, indent=2, default=str)

    lines = [f"## 📝 Community Posts ({len(posts)} total)\n"]
    posts.sort(key=lambda p: p.get("published_date", ""), reverse=True)
    for post in posts:
        lines.append(_format_post_md(post))
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="yt_update_post",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_update_post(params: UpdatePostInput) -> str:
    """Update any field on a community post record.

    Args:
        params: Post identifier and fields to update.

    Returns:
        str: Confirmation of changes.
    """
    posts = _load_posts()
    post = _find_post(posts, params.identifier)

    if not post:
        return f"❌ No post found matching '{params.identifier}'"

    changes = []

    if params.status:
        old = post.get("status")
        post["status"] = params.status.lower()
        changes.append(f"Status: {old} → {params.status}")

    if params.engagement_notes:
        post["engagement_notes"] = params.engagement_notes
        changes.append(f"Engagement notes: {params.engagement_notes}")

    if params.add_note:
        if "notes" not in post:
            post["notes"] = []
        post["notes"].append(params.add_note)
        changes.append(f"Added note: {params.add_note}")

    if params.add_tags:
        existing = set(post.get("tags", []))
        new_tags = set(params.add_tags) - existing
        post["tags"] = list(existing | new_tags)
        if new_tags:
            changes.append(f"Added tags: {', '.join(new_tags)}")

    if not changes:
        return "⚠️ No changes specified."

    post["updated_at"] = _now_iso()
    _save_posts(posts)
    return f"✅ Updated **{post['post_id']}**:\n" + "\n".join(f"- {c}" for c in changes)


@mcp.tool(
    name="yt_suggest_posts",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def yt_suggest_posts(params: SuggestPostsInput) -> str:
    """Generate community post ideas based on videos, deadlines, and content gaps.

    Strategies:
    - promote_recent: Teaser posts for videos published in last 14 days
    - compliance_deadlines: Deadline reminder posts based on known dates
    - engagement: Poll and discussion starter ideas
    - content_gaps: Link posts driving traffic to guides without videos
    - all: Mix of all strategies

    Args:
        params: Strategy type, count, and response format.

    Returns:
        str: Post suggestions with ready-to-paste draft content.
    """
    import random
    from datetime import timedelta

    suggestions = []
    strategy = (params.strategy or "all").lower()
    existing_posts = _load_posts()
    existing_content = {p.get("content", "").lower() for p in existing_posts}

    # --- Strategy: Promote Recent Videos ---
    if strategy in ("promote_recent", "all"):
        videos = _read_json(VIDEOS_FILE)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")

        recent = [v for v in videos if v.get("published_date", "") >= cutoff]

        for v in recent[:3]:
            title = v.get("title", "Untitled")
            vid = v.get("id", "")
            guide = v.get("guide_url", "")
            url = _canonical_url(vid) if vid else ""

            draft = f"🎬 New video: {title}\n\nFull breakdown with interactive demo:\n{url}"
            if guide:
                draft += f"\n\n📚 Written guide: https://fixmycert.com{guide}"

            if title.lower() not in " ".join(existing_content):
                suggestions.append({
                    "strategy": "promote_recent",
                    "post_type": "link",
                    "content": draft,
                    "linked_video_id": vid,
                    "linked_content_path": guide,
                    "tags": v.get("tags", []),
                    "rationale": f"Promote '{title}' — published recently, drive views",
                })

    # --- Strategy: Compliance Deadlines ---
    if strategy in ("compliance_deadlines", "all"):
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        upcoming = [d for d in COMPLIANCE_DEADLINES if d["date"] >= today_str]
        upcoming.sort(key=lambda d: d["date"])

        for dl in upcoming[:2]:
            date_obj = datetime.strptime(dl["date"], "%Y-%m-%d")
            days_until = (date_obj - datetime.now()).days

            if days_until <= 0:
                continue

            draft = (
                f"⏰ {days_until} days until: {dl['event']}\n\n"
                f"Are you ready? Here's everything you need to know:\n"
                f"https://fixmycert.com/compliance"
            )

            suggestions.append({
                "strategy": "compliance_deadlines",
                "post_type": "text",
                "content": draft,
                "linked_content_path": "/compliance",
                "tags": ["compliance", "47-day"],
                "rationale": f"Deadline in {days_until} days — urgency drives engagement",
            })

    # --- Strategy: Engagement (Polls) ---
    if strategy in ("engagement", "all"):
        used_questions = {
            p.get("content", "").lower()
            for p in existing_posts
            if p.get("post_type") == "poll"
        }

        available = [
            t for t in POLL_TEMPLATES
            if t["content"].lower() not in used_questions
        ]

        if available:
            selected = random.sample(available, min(2, len(available)))
            for poll in selected:
                suggestions.append({
                    "strategy": "engagement",
                    "post_type": "poll",
                    "content": poll["content"],
                    "poll_options": poll["poll_options"],
                    "tags": poll["tags"],
                    "rationale": "Polls drive comments and algorithm engagement",
                })

    # --- Strategy: Content Gaps ---
    if strategy in ("content_gaps", "all"):
        content_map = _read_json(CONTENT_MAP_FILE)
        uncovered = [
            c for c in content_map
            if not c.get("has_video") and c.get("priority") in ("high", "medium")
        ]
        for c in uncovered[:2]:
            path = c.get("path", "")
            title = c.get("title", "Untitled")
            draft = (
                f"📚 Did you know we have a full guide on {title}?\n\n"
                f"Step-by-step walkthrough with interactive demos:\n"
                f"https://fixmycert.com{path}"
            )
            suggestions.append({
                "strategy": "content_gaps",
                "post_type": "link",
                "content": draft,
                "linked_content_path": path,
                "tags": [c.get("category", "")],
                "rationale": f"Drive traffic to '{title}' — no video yet, post can fill the gap",
            })

    # Trim to requested count
    suggestions = suggestions[:params.count]

    if not suggestions:
        return "No post suggestions available. Try a different strategy or check that you have recent videos registered."

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(suggestions, indent=2, default=str)

    lines = [f"## 💡 Community Post Suggestions ({len(suggestions)})\n"]
    for i, s in enumerate(suggestions, 1):
        lines.append(f"### Suggestion {i} — {s['strategy']} ({s['post_type']})")
        lines.append(f"**Rationale:** {s['rationale']}\n")
        lines.append("**Draft content:**")
        lines.append(f"```\n{s['content']}\n```")
        if s.get("poll_options"):
            lines.append("**Poll options:**")
            for opt in s["poll_options"]:
                lines.append(f"  - {opt}")
        if s.get("tags"):
            lines.append(f"🏷️ Tags: {', '.join(s['tags'])}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="yt_get_post_stats",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_get_post_stats(params: DashboardInput) -> str:
    """Summary statistics for community posts.

    Shows total count, breakdown by type, posts per month, and most active categories.

    Args:
        params: Response format.

    Returns:
        str: Post statistics.
    """
    posts = _load_posts()

    if not posts:
        return "No community posts registered yet. Use `yt_register_post` to start tracking."

    by_type: dict[str, int] = {}
    for p in posts:
        t = p.get("post_type", "text")
        by_type[t] = by_type.get(t, 0) + 1

    by_status: dict[str, int] = {}
    for p in posts:
        s = p.get("status", "published")
        by_status[s] = by_status.get(s, 0) + 1

    by_month: dict[str, int] = {}
    for p in posts:
        date = p.get("published_date", "")
        if len(date) >= 7:
            month = date[:7]
            by_month[month] = by_month.get(month, 0) + 1

    published = [p for p in posts if p.get("status") == "published"]
    published.sort(key=lambda p: p.get("published_date", ""), reverse=True)
    last_post_date = published[0].get("published_date", "???") if published else "none"

    by_cat: dict[str, int] = {}
    for p in posts:
        cat = p.get("category", "uncategorized") or "uncategorized"
        by_cat[cat] = by_cat.get(cat, 0) + 1

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "total": len(posts),
            "by_type": by_type,
            "by_status": by_status,
            "by_month": by_month,
            "by_category": by_cat,
            "last_post_date": last_post_date,
        }, indent=2)

    lines = ["## 📊 Community Post Stats\n"]
    lines.append(f"**Total posts:** {len(posts)}")
    lines.append(f"**Last post:** {last_post_date}")

    type_icons = {"text": "📝", "poll": "📊", "image": "🖼️", "link": "🔗"}
    type_parts = [f"{type_icons.get(t, '📝')} {t}: {c}" for t, c in sorted(by_type.items())]
    lines.append(f"**By type:** {', '.join(type_parts)}")

    if by_month:
        lines.append("\n**Posts per month:**")
        for month in sorted(by_month.keys(), reverse=True)[:6]:
            lines.append(f"  - {month}: {by_month[month]} posts")

    if by_cat:
        top_cats = sorted(by_cat.items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append(f"\n**Top categories:** {', '.join(f'{c} ({n})' for c, n in top_cats)}")

    return "\n".join(lines)


@mcp.tool(
    name="yt_get_post_calendar",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def yt_get_post_calendar(params: PostCalendarInput) -> str:
    """Show posting frequency week-by-week and identify gaps.

    Helps maintain a consistent posting cadence for channel growth.

    Args:
        params: Number of weeks to show and response format.

    Returns:
        str: Week-by-week post count with streak and cadence info.
    """
    from datetime import timedelta

    posts = _load_posts()
    published = [p for p in posts if p.get("status") == "published"]

    if not published:
        return "No published posts yet. Start posting to build your calendar!"

    today = datetime.now(timezone.utc)
    weeks = []

    for w in range(params.weeks_back):
        if w == 0:
            # Current partial week: Monday through today
            week_start = today - timedelta(days=today.weekday())
            week_end = today
        else:
            # Previous full weeks: Monday to Sunday
            prev_monday = today - timedelta(days=today.weekday(), weeks=w)
            week_start = prev_monday
            week_end = prev_monday + timedelta(days=6)
        start_str = week_start.strftime("%Y-%m-%d")
        end_str = week_end.strftime("%Y-%m-%d")

        count = sum(
            1 for p in published
            if start_str <= p.get("published_date", "") <= end_str
        )
        weeks.append({
            "week_start": start_str,
            "week_end": end_str,
            "count": count,
        })

    weeks.reverse()  # Oldest first

    # Calculate streak (consecutive weeks with at least 1 post)
    streak = 0
    for w in reversed(weeks):
        if w["count"] > 0:
            streak += 1
        else:
            break

    total_posts = sum(w["count"] for w in weeks)
    avg = total_posts / len(weeks) if weeks else 0

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({
            "weeks": weeks,
            "current_streak": streak,
            "average_per_week": round(avg, 1),
            "total_in_period": total_posts,
        }, indent=2)

    lines = [f"## 📅 Post Calendar (last {params.weeks_back} weeks)\n"]

    for w in weeks:
        bars = "█" * w["count"] if w["count"] > 0 else "·"
        lines.append(f"  {w['week_start']} — {w['week_end']}: {bars} ({w['count']})")

    lines.append(f"\n**Current streak:** {streak} {'week' if streak == 1 else 'weeks'}")
    lines.append(f"**Average:** {avg:.1f} posts/week")
    lines.append(f"**Total in period:** {total_posts}")

    if avg < 1:
        lines.append("\n💡 **Suggestion:** Aim for 2-3 posts/week to build channel engagement.")
    elif avg < 2:
        lines.append("\n💡 **Suggestion:** Good start! Try to hit 2-3 posts/week consistently.")
    else:
        lines.append("\n✅ Great posting cadence! Keep it up.")

    return "\n".join(lines)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import os as _os
    _transport = _os.environ.get("MCP_TRANSPORT", "stdio")
    if _transport == "sse":
        mcp.settings.host = _os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(_os.environ.get("MCP_PORT", "8082"))
        mcp.settings.transport_security.enable_dns_rebinding_protection = False
        mcp.run(transport="sse")
    else:
        mcp.run()