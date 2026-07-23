"""
FixMyCert News Desk MCP Server

Thin MCP client over the fixmycert.com news API (the "Part 1" news backend
living in the FixMyCert Repl). No local database — every tool is a live HTTP
call against the backing API.

Transport: stdio locally, SSE in Docker (MCP_TRANSPORT=sse)
Config: env vars only — NEWS_API_BASE_URL, NEWS_ADMIN_SECRET (required),
        NEWS_CRON_SECRET (optional, only for news_trigger_fetch)

Auth (confirmed against the Repl's checkNewsKey helper, 2026-07-22):
the admin routes accept `?key=` OR `Authorization: Bearer` against
NEWS_ADMIN_SECRET; we use the Bearer header so the secret never appears in
URLs (httpx logs full request URLs at INFO). /api/cron/fetch-news is
query-only (`?key=`) against CRON_SECRET — no header support there, so
httpx logging is silenced below to keep it out of container logs.
"""

import difflib
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

# ============================================================
# Constants / config
# ============================================================

NEWS_API_BASE_URL = os.environ.get("NEWS_API_BASE_URL", "").rstrip("/")
NEWS_ADMIN_SECRET = os.environ.get("NEWS_ADMIN_SECRET", "")
NEWS_CRON_SECRET = os.environ.get("NEWS_CRON_SECRET", "")

COMPLIANCE_DATA_URL = "https://compliance-api.fixmycert.com/api/compliance-data"

NEWS_CLUSTERS = [
    "ca-action", "tls-validity", "pqc", "code-signing",
    "smime", "client-auth", "other",
]
STATUSES = ["draft", "published", "archived"]

HTTP_TIMEOUT = httpx.Timeout(30.0)

# httpx logs full request URLs at INFO — would leak ?key= secrets into
# container logs (the cron route only accepts the secret via query string)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ============================================================
# Server
# ============================================================

mcp = FastMCP("fixmycert_news_mcp")

# ============================================================
# Enums / input models
# ============================================================


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class NewsCluster(str, Enum):
    CA_ACTION = "ca-action"
    TLS_VALIDITY = "tls-validity"
    PQC = "pqc"
    CODE_SIGNING = "code-signing"
    SMIME = "smime"
    CLIENT_AUTH = "client-auth"
    OTHER = "other"


class Status(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class DashboardInput(BaseModel):
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format"
    )


class ListNewsInput(BaseModel):
    status: Optional[Status] = Field(default=None, description="Filter by status (draft/published/archived)")
    category: Optional[str] = Field(default=None, description="Filter by category (e.g. ca_news, browser_updates, vendors, research)")
    newsCluster: Optional[NewsCluster] = Field(default=None, description="Filter by news cluster")
    isFirstParty: Optional[bool] = Field(default=None, description="Filter to first-party (true) or aggregated (false) items")
    isPriority: Optional[bool] = Field(default=None, description="Filter to priority items")
    source: Optional[str] = Field(default=None, description="Filter by source name (case-insensitive substring)")
    since: Optional[str] = Field(default=None, description="Only items published/fetched on or after this date (YYYY-MM-DD or ISO timestamp)")
    limit: int = Field(default=20, ge=1, le=200, description="Max items to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class GetNewsInput(BaseModel):
    id: str = Field(description="News item id (UUID from news_list)")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class GetUncoveredInput(BaseModel):
    sinceDays: int = Field(
        default=60, ge=1, le=730,
        description="Look-back window: consider Hub deadlines dated after (today - sinceDays), plus all future deadlines",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class AddNewsInput(BaseModel):
    title: str = Field(min_length=5, description="Headline for the news item")
    excerpt: Optional[str] = Field(default=None, description="1-3 sentence summary shown in the feed")
    url: Optional[str] = Field(default=None, description="Primary-source URL (ballot, root program announcement, CA bulletin). If omitted the backend generates an internal permalink — first-party items should normally carry one")
    internalUrl: Optional[str] = Field(default=None, description="FixMyCert-internal link shown alongside the item, as a site-relative path (e.g. /guides/47-day-certificate-timeline)")
    category: Optional[str] = Field(default=None, description="Feed category (e.g. ca_news, browser_updates, vendors, research)")
    newsCluster: NewsCluster = Field(description="Required cluster: ca-action | tls-validity | pqc | code-signing | smime | client-auth | other")
    symptomString: Optional[str] = Field(default=None, description="If the change causes a visible failure, the error string users will see (e.g. NET::ERR_CERT_AUTHORITY_INVALID)")
    deadlineId: Optional[str] = Field(default=None, description="Compliance Hub deadline id this item covers (see news_get_uncovered)")
    opportunityTags: Optional[list[str]] = Field(default=None, description="Content-opportunity tags (e.g. guide, checklist, video)")
    isPriority: bool = Field(default=False, description="Float to top of /news — reserve for high-impact changes")
    status: Status = Field(default=Status.DRAFT, description="Initial status; leave as draft and use news_publish to go live")
    publishedAt: Optional[str] = Field(default=None, description="Override publish timestamp (ISO 8601); defaults server-side")

    @field_validator("publishedAt")
    @classmethod
    def _validate_published_at(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            _parse_dt(v, field="publishedAt")  # raises ValueError if bad
        return v


class UpdateNewsInput(BaseModel):
    id: str = Field(description="News item id to update")
    title: Optional[str] = Field(default=None, description="New title")
    excerpt: Optional[str] = Field(default=None, description="New excerpt")
    url: Optional[str] = Field(default=None, description="New primary-source URL")
    internalUrl: Optional[str] = Field(default=None, description="New FixMyCert-internal link (site-relative path, e.g. /guides/47-day-certificate-timeline)")
    category: Optional[str] = Field(default=None, description="New category")
    newsCluster: Optional[NewsCluster] = Field(default=None, description="New cluster")
    symptomString: Optional[str] = Field(default=None, description="New symptom string")
    deadlineId: Optional[str] = Field(default=None, description="New Compliance Hub deadline id")
    opportunityTags: Optional[list[str]] = Field(default=None, description="Replacement opportunity tags")
    isPriority: Optional[bool] = Field(default=None, description="New priority flag")
    status: Optional[Status] = Field(default=None, description="New status (prefer news_publish / news_archive)")
    publishedAt: Optional[str] = Field(default=None, description="New publish timestamp (ISO 8601)")


class ItemIdInput(BaseModel):
    id: str = Field(description="News item id")


class TagOpportunityInput(BaseModel):
    id: str = Field(description="News item id")
    opportunityTags: list[str] = Field(min_length=1, description="Opportunity tags to set on the item (replaces existing tags)")


class GenerateDigestInput(BaseModel):
    fromDate: Optional[str] = Field(default=None, description="Period start (YYYY-MM-DD); overrides sinceDays")
    toDate: Optional[str] = Field(default=None, description="Period end (YYYY-MM-DD); defaults to today")
    sinceDays: int = Field(default=30, ge=1, le=365, description="Period length when fromDate is not given")
    clusters: Optional[list[NewsCluster]] = Field(default=None, description="Restrict to these news clusters")
    frameworks: Optional[list[str]] = Field(default=None, description="Restrict to entries whose linked Hub deadline belongs to these frameworks (matched against framework_name, case-insensitive substring)")


class ListSourcesInput(BaseModel):
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


# ============================================================
# HTTP helpers
# ============================================================


class NewsAPIError(Exception):
    """Raised with a user-actionable message; never contains the secret."""


def _redact(text: str) -> str:
    """Strip secrets from any string that might reach tool output."""
    for secret in (NEWS_ADMIN_SECRET, NEWS_CRON_SECRET):
        if secret:
            text = text.replace(secret, "***")
    return re.sub(r"(key=)[^&\s\"']+", r"\1***", text)


def _auth_params_headers() -> tuple[dict, dict]:
    """Bearer header only — confirmed accepted by all news admin routes;
    keeps the secret out of URLs and logs."""
    return {}, {"Authorization": f"Bearer {NEWS_ADMIN_SECRET}"}


async def _request(
    method: str,
    path: str,
    *,
    authed: bool = False,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
    base_url: Optional[str] = None,
) -> Any:
    url = (base_url or NEWS_API_BASE_URL) + path
    params = dict(params or {})
    headers = {}
    if authed:
        auth_params, auth_headers = _auth_params_headers()
        params.update(auth_params)
        headers.update(auth_headers)
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            resp = await client.request(method, url, params=params or None, json=json_body, headers=headers)
    except httpx.HTTPError as e:
        raise NewsAPIError(
            f"Could not reach {_redact(url)}: {_redact(str(e))}. "
            f"Check NEWS_API_BASE_URL and network connectivity from this host."
        )
    if resp.status_code == 401:
        raise NewsAPIError(
            f"401 Unauthorized from {_redact(url)}. "
            f"NEWS_ADMIN_SECRET on this host does not match the secret in the FixMyCert Replit Secrets "
            f"(or the route uses a different secret — the RSS refresh route needs NEWS_CRON_SECRET, not the admin secret)."
        )
    if resp.status_code == 404:
        raise NewsAPIError(
            f"404 Not Found from {_redact(url)}. "
            f"If this is an item route, verify the id with news_list; otherwise the backend may not expose this route yet."
        )
    if resp.status_code >= 400:
        body = _redact(resp.text[:300])
        raise NewsAPIError(f"{resp.status_code} from {_redact(url)}: {body}")
    try:
        return resp.json()
    except ValueError:
        raise NewsAPIError(f"Non-JSON response from {_redact(url)}: {_redact(resp.text[:200])}")


async def _fetch_admin_items(filters: Optional[dict] = None) -> list[dict]:
    """Fetch all matching items from /api/news/admin (drafts included).

    Handles both response shapes: a bare list, or {items,total,hasMore}.
    Paginates defensively if the route honors limit/offset.
    """
    filters = {k: v for k, v in (filters or {}).items() if v is not None}
    # booleans as lowercase strings for query params
    for k, v in list(filters.items()):
        if isinstance(v, bool):
            filters[k] = "true" if v else "false"

    items: list[dict] = []
    offset = 0
    page_size = 500
    for _ in range(20):  # hard cap: 10k items
        data = await _request("GET", "/api/news/admin", authed=True,
                              params={**filters, "limit": page_size, "offset": offset})
        if isinstance(data, list):
            page = data
            has_more = False
        else:
            page = data.get("items", [])
            has_more = bool(data.get("hasMore")) and len(page) > 0
        items.extend(page)
        if not has_more:
            break
        offset += len(page)
    return items


async def _fetch_deadlines() -> list[dict]:
    data = await _request("GET", "", base_url=COMPLIANCE_DATA_URL)
    return data.get("deadlines", []) if isinstance(data, dict) else []


# ============================================================
# Formatting / matching helpers
# ============================================================


def _parse_dt(value: str, field: str = "date") -> datetime:
    v = value.strip().replace("Z", "+00:00")
    for fmt in None, "%Y-%m-%d":
        try:
            dt = datetime.fromisoformat(v) if fmt is None else datetime.strptime(v, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Invalid {field}: '{value}' — use YYYY-MM-DD or ISO 8601")


def _item_dt(item: dict) -> Optional[datetime]:
    for key in ("publishedAt", "fetchedAt"):
        if item.get(key):
            try:
                return _parse_dt(item[key])
            except ValueError:
                continue
    return None


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).strip()


def _titles_match(deadline_title: str, item_title: str) -> bool:
    a, b = _norm_title(deadline_title), _norm_title(item_title)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


def _fmt_item_line(item: dict) -> str:
    flags = []
    if item.get("isPriority"):
        flags.append("⭐ priority")
    if item.get("isFirstParty"):
        flags.append("🏠 first-party")
    status = item.get("status", "?")
    status_icon = {"draft": "📝", "published": "✅", "archived": "🗄️"}.get(status, "•")
    dt = _item_dt(item)
    when = dt.strftime("%Y-%m-%d") if dt else "?"
    parts = [f"{status_icon} **{item.get('title', '(untitled)')}**"]
    meta = [status, when, item.get("source") or "?"]
    if item.get("newsCluster"):
        meta.append(item["newsCluster"])
    if flags:
        meta.extend(flags)
    parts.append(f"  `{item.get('id', '?')}` — {' · '.join(meta)}")
    return "\n".join(parts)


def _fmt_item_full(item: dict) -> str:
    lines = [f"## {item.get('title', '(untitled)')}", ""]
    for key in ("id", "status", "source", "sourceUrl", "url", "internalUrl", "category", "newsCluster",
                "symptomString", "deadlineId", "opportunityTags", "isPriority",
                "isFirstParty", "excerpt", "publishedAt", "fetchedAt"):
        val = item.get(key)
        if val not in (None, "", []):
            lines.append(f"- **{key}**: {val}")
    return "\n".join(lines)


def _require_config() -> None:
    missing = []
    if not NEWS_API_BASE_URL:
        missing.append("NEWS_API_BASE_URL (e.g. https://fixmycert.com)")
    if not NEWS_ADMIN_SECRET:
        missing.append("NEWS_ADMIN_SECRET (same value as in FixMyCert Replit Secrets)")
    if missing:
        sys.stderr.write(
            "fixmycert-news-mcp: missing required environment variables:\n  - "
            + "\n  - ".join(missing)
            + "\nSet them in /opt/mcp-servers/.env (droplet) or your shell and restart.\n"
        )
        sys.exit(1)


# ============================================================
# Tools — read
# ============================================================


@mcp.tool(
    name="news_dashboard",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def news_dashboard(params: DashboardInput) -> str:
    """News desk overview: item counts by status/cluster, first-party coverage,
    priority items, last aggregator fetch, and uncovered Hub deadlines.

    Args:
        params: Output format only

    Returns:
        str: Dashboard in markdown (or JSON)
    """
    try:
        items = await _fetch_admin_items()
        deadlines = await _fetch_deadlines()
    except NewsAPIError as e:
        return f"❌ {e}"

    total = len(items)
    first_party = [i for i in items if i.get("isFirstParty")]
    published = [i for i in items if i.get("status") == "published"]
    drafts = [i for i in items if i.get("status") == "draft"]
    priority = [i for i in items if i.get("isPriority")]
    by_cluster: dict[str, int] = {}
    for i in items:
        cluster = i.get("newsCluster") or "(none)"
        by_cluster[cluster] = by_cluster.get(cluster, 0) + 1
    fetched = [d for d in (_parse_dt(i["fetchedAt"]) for i in items if i.get("fetchedAt")) if d]
    last_fetched = max(fetched).isoformat() if fetched else None

    uncovered = _compute_uncovered(deadlines, first_party, since_days=60)

    stats = {
        "total": total,
        "firstParty": len(first_party),
        "published": len(published),
        "draft": len(drafts),
        "priority": len(priority),
        "byCluster": by_cluster,
        "lastFetchedAt": last_fetched,
        "uncovered": len(uncovered),
    }
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(stats, indent=2)

    lines = ["## 📰 FixMyCert News Dashboard", ""]
    lines.append(f"- **Total items**: {total}")
    lines.append(f"- **First-party**: {len(first_party)}")
    lines.append(f"- **Published**: {len(published)} · **Drafts**: {len(drafts)} · **Priority**: {len(priority)}")
    lines.append(f"- **Last aggregator fetch**: {last_fetched or 'never'}")
    lines.append(f"- **Uncovered Hub deadlines (60d window)**: {len(uncovered)}"
                 + (" — run news_get_uncovered" if uncovered else ""))
    lines.append("\n### By cluster")
    for cluster in sorted(by_cluster, key=by_cluster.get, reverse=True):
        lines.append(f"- {cluster}: {by_cluster[cluster]}")
    if drafts:
        lines.append("\n### Drafts awaiting publish")
        for d in drafts[:10]:
            lines.append(f"- `{d.get('id')}` {d.get('title')}")
    return "\n".join(lines)


@mcp.tool(
    name="news_list",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def news_list(params: ListNewsInput) -> str:
    """List news items via the admin feed (drafts and archived included).

    Args:
        params: Filters (status, category, newsCluster, isFirstParty, isPriority,
                source, since) plus limit/offset pagination and output format

    Returns:
        str: Matching items with ids, paginated
    """
    try:
        items = await _fetch_admin_items({
            "status": params.status.value if params.status else None,
            "category": params.category,
            "newsCluster": params.newsCluster.value if params.newsCluster else None,
            "isFirstParty": params.isFirstParty,
            "isPriority": params.isPriority,
        })
    except NewsAPIError as e:
        return f"❌ {e}"

    # Client-side safety net: apply every filter locally too, so behavior is
    # identical whether or not the backend honors a given query param.
    def keep(i: dict) -> bool:
        if params.status and i.get("status") != params.status.value:
            return False
        if params.category and (i.get("category") or "").lower() != params.category.lower():
            return False
        if params.newsCluster and i.get("newsCluster") != params.newsCluster.value:
            return False
        if params.isFirstParty is not None and bool(i.get("isFirstParty")) != params.isFirstParty:
            return False
        if params.isPriority is not None and bool(i.get("isPriority")) != params.isPriority:
            return False
        if params.source and params.source.lower() not in (i.get("source") or "").lower():
            return False
        return True

    items = [i for i in items if keep(i)]
    if params.since:
        try:
            since_dt = _parse_dt(params.since, field="since")
        except ValueError as e:
            return f"❌ {e}"
        items = [i for i in items if (_item_dt(i) or datetime.min.replace(tzinfo=timezone.utc)) >= since_dt]

    total = len(items)
    page = items[params.offset:params.offset + params.limit]
    has_more = params.offset + params.limit < total

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"items": page, "total": total, "offset": params.offset,
                           "limit": params.limit, "hasMore": has_more}, indent=2, default=str)

    if not page:
        return f"No news items match these filters (total matching: {total})."
    lines = [f"## News items {params.offset + 1}–{params.offset + len(page)} of {total}", ""]
    for i in page:
        lines.append(_fmt_item_line(i))
        lines.append("")
    if has_more:
        lines.append(f"➡️ More available — call again with offset={params.offset + params.limit}")
    return "\n".join(lines)


@mcp.tool(
    name="news_get",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def news_get(params: GetNewsInput) -> str:
    """Get full details of one news item by id (drafts included).

    Args:
        params: Item id and output format

    Returns:
        str: All fields of the item
    """
    try:
        items = await _fetch_admin_items()
    except NewsAPIError as e:
        return f"❌ {e}"
    item = next((i for i in items if i.get("id") == params.id), None)
    if not item:
        prefix = [i for i in items if str(i.get("id", "")).startswith(params.id)]
        if len(prefix) == 1:
            item = prefix[0]
        elif prefix:
            opts = "\n".join(f"- `{i['id']}` {i.get('title')}" for i in prefix[:5])
            return f"❌ Ambiguous id prefix '{params.id}':\n{opts}"
        else:
            return f"❌ No news item with id '{params.id}'. Use news_list to find ids."
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(item, indent=2, default=str)
    return _fmt_item_full(item)


def _compute_uncovered(deadlines: list[dict], first_party_items: list[dict], since_days: int) -> list[dict]:
    """Deadlines (recent past within since_days, plus all future) with no
    first-party item matching by deadlineId, else fuzzy title."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    uncovered = []
    for d in deadlines:
        try:
            d_date = _parse_dt(d.get("date", ""))
        except ValueError:
            continue
        if d_date < cutoff:
            continue
        covered_by = None
        for item in first_party_items:
            if d.get("id") and item.get("deadlineId") == d.get("id"):
                covered_by = item
                break
        if not covered_by:
            for item in first_party_items:
                if _titles_match(d.get("title", ""), item.get("title", "")):
                    covered_by = item
                    break
        if covered_by is None:
            uncovered.append(d)
    return uncovered


@mcp.tool(
    name="news_get_uncovered",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def news_get_uncovered(params: GetUncoveredInput) -> str:
    """Compliance Hub deadlines with no first-party news entry yet — the
    content gap list. Matches by deadlineId first, then fuzzy title.

    Args:
        params: sinceDays look-back window (deadlines older than that are ignored;
                future deadlines always considered) and output format

    Returns:
        str: Uncovered deadlines with id, date, and framework — feed deadlineId
             into news_add to close a gap
    """
    try:
        deadlines = await _fetch_deadlines()
        items = await _fetch_admin_items({"isFirstParty": True})
    except NewsAPIError as e:
        return f"❌ {e}"
    first_party = [i for i in items if i.get("isFirstParty")]
    uncovered = _compute_uncovered(deadlines, first_party, params.sinceDays)
    uncovered.sort(key=lambda d: d.get("date", ""))

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"uncovered": uncovered, "total": len(uncovered),
                           "sinceDays": params.sinceDays,
                           "firstPartyItemsChecked": len(first_party)}, indent=2, default=str)

    if not uncovered:
        return (f"✅ All {len(deadlines)} Hub deadlines in the window (last {params.sinceDays} days + future) "
                f"have first-party coverage ({len(first_party)} first-party items checked).")
    lines = [f"## 🕳️ Uncovered Hub deadlines ({len(uncovered)})",
             f"_Window: last {params.sinceDays} days + all future · {len(first_party)} first-party items checked_", ""]
    for d in uncovered:
        lines.append(f"- **{d.get('date')}** — {d.get('title')}")
        meta = [f"id: `{d.get('id')}`"]
        if d.get("framework_name") or d.get("framework_id"):
            meta.append(f"framework: {d.get('framework_name') or d.get('framework_id')}")
        if d.get("category"):
            meta.append(f"category: {d.get('category')}")
        lines.append(f"  {' · '.join(meta)}")
    lines.append("\nUse news_add with the deadlineId to close a gap.")
    return "\n".join(lines)


@mcp.tool(
    name="news_list_sources",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def news_list_sources(params: ListSourcesInput) -> str:
    """List the RSS aggregator's source feeds.

    Args:
        params: Output format only

    Returns:
        str: Source name, URL, and category for each aggregator feed
    """
    try:
        data = await _request("GET", "/api/news/sources")
    except NewsAPIError as e:
        return f"❌ {e}"
    sources = data.get("sources", data if isinstance(data, list) else [])
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"sources": sources, "total": len(sources)}, indent=2)
    lines = [f"## Aggregator sources ({len(sources)})", ""]
    for s in sources:
        lines.append(f"- **{s.get('name')}** — {s.get('url')} ({s.get('category')})")
    return "\n".join(lines)


# ============================================================
# Tools — write
# ============================================================


@mcp.tool(
    name="news_add",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def news_add(params: AddNewsInput) -> str:
    """Create a first-party news item (defaults to draft — nothing is public
    until news_publish). newsCluster is required; carry a primary-source url,
    and a symptomString when the change causes a visible failure.

    Args:
        params: Item fields; newsCluster required, status defaults to draft

    Returns:
        str: The created item with its id and next-step hint
    """
    body: dict[str, Any] = {
        "title": params.title,
        "newsCluster": params.newsCluster.value,
        "isPriority": params.isPriority,
        "status": params.status.value,
    }
    for key in ("excerpt", "url", "internalUrl", "category", "symptomString", "deadlineId", "opportunityTags", "publishedAt"):
        val = getattr(params, key)
        if val is not None:
            body[key] = val
    try:
        created = await _request("POST", "/api/news/items", authed=True, json_body=body)
    except NewsAPIError as e:
        return f"❌ {e}"
    item = created.get("item", created) if isinstance(created, dict) else created
    lines = ["## ✅ News item created", "", _fmt_item_full(item) if isinstance(item, dict) else str(item)]
    if params.status == Status.DRAFT:
        item_id = item.get("id") if isinstance(item, dict) else "?"
        lines.append(f"\n📝 Draft only — not public until you run news_publish with id `{item_id}`.")
    if not params.url:
        lines.append("\n⚠️ No primary-source url given — backend will use an internal permalink. "
                     "First-party items should normally link the primary source.")
    return "\n".join(lines)


async def _patch_item(item_id: str, fields: dict) -> dict:
    updated = await _request("PATCH", f"/api/news/items/{item_id}", authed=True, json_body=fields)
    return updated.get("item", updated) if isinstance(updated, dict) else updated


@mcp.tool(
    name="news_update",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def news_update(params: UpdateNewsInput) -> str:
    """Update fields on an existing news item. Only the fields you pass are
    changed; omitted fields are untouched.

    Args:
        params: Item id plus any fields to change

    Returns:
        str: The updated item
    """
    fields: dict[str, Any] = {}
    for key in ("title", "excerpt", "url", "internalUrl", "category", "symptomString", "deadlineId",
                "opportunityTags", "isPriority", "publishedAt"):
        val = getattr(params, key)
        if val is not None:
            fields[key] = val
    if params.newsCluster is not None:
        fields["newsCluster"] = params.newsCluster.value
    if params.status is not None:
        fields["status"] = params.status.value
    if not fields:
        return "❌ No fields to update — pass at least one field besides id."
    try:
        item = await _patch_item(params.id, fields)
    except NewsAPIError as e:
        return f"❌ {e}"
    return "## ✅ News item updated\n\n" + (_fmt_item_full(item) if isinstance(item, dict) else str(item))


@mcp.tool(
    name="news_publish",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def news_publish(params: ItemIdInput) -> str:
    """Publish a draft: sets status=published so the item appears on the
    public /api/news feed (and fixmycert.com/news).

    Args:
        params: Item id

    Returns:
        str: Confirmation with the published item
    """
    try:
        item = await _patch_item(params.id, {"status": "published"})
    except NewsAPIError as e:
        return f"❌ {e}"
    return "## 🚀 Published\n\n" + (_fmt_item_full(item) if isinstance(item, dict) else str(item))


@mcp.tool(
    name="news_tag_opportunity",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def news_tag_opportunity(params: TagOpportunityInput) -> str:
    """Set opportunity tags on an item (e.g. guide, checklist, video) to mark
    follow-up content worth producing. Replaces the item's existing tags.

    Args:
        params: Item id and the full replacement tag list

    Returns:
        str: Confirmation with the updated item
    """
    try:
        item = await _patch_item(params.id, {"opportunityTags": params.opportunityTags})
    except NewsAPIError as e:
        return f"❌ {e}"
    return "## 🏷️ Opportunity tags set\n\n" + (_fmt_item_full(item) if isinstance(item, dict) else str(item))


@mcp.tool(
    name="news_archive",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def news_archive(params: ItemIdInput) -> str:
    """Archive an item: sets status=archived, removing it from the public feed.
    No hard delete exists — archived items remain visible via news_list.

    Args:
        params: Item id

    Returns:
        str: Confirmation with the archived item
    """
    try:
        item = await _patch_item(params.id, {"status": "archived"})
    except NewsAPIError as e:
        return f"❌ {e}"
    return "## 🗄️ Archived (removed from public feed)\n\n" + (_fmt_item_full(item) if isinstance(item, dict) else str(item))


# ============================================================
# Tools — digest & aggregator
# ============================================================


@mcp.tool(
    name="news_generate_digest",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def news_generate_digest(params: GenerateDigestInput) -> dict:
    """Build a period digest of published news, grouped by cluster, with Hub
    deadline context — ready to paste into a newsletter or LinkedIn post.

    Args:
        params: Period (fromDate/toDate or sinceDays), optional cluster and
                framework filters

    Returns:
        dict: {period, entries[{title,cluster,summary,impact,sourceUrl,deadline}], markdown}
    """
    try:
        to_dt = _parse_dt(params.toDate, "toDate") if params.toDate else datetime.now(timezone.utc)
        from_dt = (_parse_dt(params.fromDate, "fromDate") if params.fromDate
                   else to_dt - timedelta(days=params.sinceDays))
    except ValueError as e:
        return {"error": str(e)}
    if from_dt > to_dt:
        return {"error": f"fromDate {from_dt.date()} is after toDate {to_dt.date()}"}

    try:
        items = await _fetch_admin_items({"status": "published"})
        deadlines = await _fetch_deadlines()
    except NewsAPIError as e:
        return {"error": str(e)}

    deadline_by_id = {d.get("id"): d for d in deadlines if d.get("id")}
    cluster_filter = {c.value for c in params.clusters} if params.clusters else None

    entries = []
    for i in items:
        if i.get("status") != "published":
            continue
        dt = _item_dt(i)
        if not dt or not (from_dt <= dt <= to_dt):
            continue
        cluster = i.get("newsCluster") or "other"
        if cluster_filter and cluster not in cluster_filter:
            continue
        deadline = deadline_by_id.get(i.get("deadlineId"))
        if params.frameworks:
            fw = (deadline or {}).get("framework_name") or (deadline or {}).get("framework_id") or ""
            if not any(f.lower() in fw.lower() for f in params.frameworks):
                continue
        if i.get("isPriority"):
            impact = "High — flagged priority"
        elif i.get("symptomString"):
            impact = f"Causes visible failure: {i['symptomString']}"
        else:
            impact = "Informational"
        entries.append({
            "title": i.get("title"),
            "cluster": cluster,
            "summary": i.get("excerpt") or "",
            "impact": impact,
            "sourceUrl": i.get("url") or i.get("sourceUrl") or "",
            "deadline": ({"id": deadline.get("id"), "date": deadline.get("date"),
                          "title": deadline.get("title")} if deadline else None),
        })

    entries.sort(key=lambda e: (e["impact"] != "High — flagged priority", e["cluster"], e["title"] or ""))
    period = f"{from_dt.date().isoformat()} to {to_dt.date().isoformat()}"

    md = [f"# FixMyCert News Digest — {period}", ""]
    if not entries:
        md.append("_No published news in this period matches the filters._")
    by_cluster: dict[str, list] = {}
    for e in entries:
        by_cluster.setdefault(e["cluster"], []).append(e)
    for cluster in sorted(by_cluster):
        md.append(f"## {cluster}")
        for e in by_cluster[cluster]:
            md.append(f"### {e['title']}")
            if e["summary"]:
                md.append(e["summary"])
            md.append(f"- **Impact**: {e['impact']}")
            if e["deadline"]:
                md.append(f"- **Deadline**: {e['deadline']['date']} — {e['deadline']['title']}")
            if e["sourceUrl"]:
                md.append(f"- **Source**: {e['sourceUrl']}")
            md.append("")
    return {"period": period, "entries": entries, "markdown": "\n".join(md)}


@mcp.tool(
    name="news_trigger_fetch",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def news_trigger_fetch() -> str:
    """Trigger the RSS aggregator refresh (GET /api/cron/fetch-news). Uses the
    separate NEWS_CRON_SECRET, not the admin secret.

    Returns:
        str: The refresh route's response (fetched/inserted counts)
    """
    if not NEWS_CRON_SECRET:
        return ("❌ NEWS_CRON_SECRET is not set on this host. The RSS refresh route uses the "
                "CRON_SECRET from the FixMyCert Replit Secrets (separate from NEWS_ADMIN_SECRET). "
                "Add NEWS_CRON_SECRET to /opt/mcp-servers/.env and restart this container.")
    url = NEWS_API_BASE_URL + "/api/cron/fetch-news"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), follow_redirects=True) as client:
            # cron route is query-only (?key=) — confirmed no header support
            resp = await client.get(url, params={"key": NEWS_CRON_SECRET})
    except httpx.HTTPError as e:
        return f"❌ Could not reach {_redact(url)}: {_redact(str(e))}"
    if resp.status_code == 401:
        return "❌ 401 from the refresh route — NEWS_CRON_SECRET does not match CRON_SECRET in Replit Secrets."
    if resp.status_code >= 400:
        return f"❌ {resp.status_code} from refresh route: {_redact(resp.text[:300])}"
    return f"## ✅ Aggregator refresh triggered\n\n```json\n{_redact(resp.text[:1000])}\n```"


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    _require_config()
    _transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if _transport == "sse":
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8086"))
        mcp.settings.transport_security.enable_dns_rebinding_protection = False
        mcp.run(transport="sse")
    else:
        mcp.run()
