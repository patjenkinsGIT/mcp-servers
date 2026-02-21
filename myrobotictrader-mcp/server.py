"""
MyRoboticTrader Content Automation MCP Server

This MCP server provides tools for automated content generation based on:
- Current crypto and financial news (via NewsAPI)
- Discord channel monitoring for project updates
- Regulatory updates and gambling news
- Wild financial stories and trends
- Your trading performance data from Google Sheets
- Dynamic topic management for breaking news
- Live cryptocurrency prices and market data from CoinMarketCap

Author: Built for Patrick Jenkins / MyRoboticTrader.com & DailyProfits.net
"""

# Load environment variables from .env file
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the directory where server.py is located
SERVER_DIR = Path(__file__).parent
# Load .env from that directory
load_dotenv(SERVER_DIR / ".env")

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
import json
import httpx
from contextlib import asynccontextmanager
import asyncio

# Google Sheets integration
from google.oauth2 import service_account
from googleapiclient.discovery import build

# CoinMarketCap integration
from coinmarketcap_tools import (
    fetch_crypto_price,
    fetch_multiple_cryptos,
    fetch_top_cryptos,
    fetch_gainers_losers,
    fetch_fear_greed_index,
    fetch_altcoin_season_index,
    format_crypto_data_markdown,
    format_top_cryptos_markdown,
    format_gainers_losers_markdown
)

# Discord integration
from discord_tools import (
    fetch_channel_messages,
    search_messages_for_keywords,
    get_guild_channels,
    get_bot_guilds,
    monitor_for_announcements,
    format_messages_markdown,
    format_channels_markdown,
    format_guilds_markdown,
    DEPIN_KEYWORDS,
    TRADING_KEYWORDS,
    ALERT_KEYWORDS
)

# Initialize the MCP server
mcp = FastMCP("myrobotictrader_mcp")

# ============================================================================
# GOOGLE SHEETS HELPER FUNCTIONS
# ============================================================================

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def _get_sheets_service():
    """Get Google Sheets API service."""
    creds_path = os.getenv("GOOGLE_SHEETS_CREDS")
    if not creds_path:
        return None
    
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=SCOPES
    )
    return build('sheets', 'v4', credentials=creds)

def _read_topics_from_sheet(sheet_id: str, sheet_name: str = "Topics") -> Dict[str, Any]:
    """Read topics from Google Sheet synchronously."""
    try:
        service = _get_sheets_service()
        if not service:
            return {"topics": []}
        
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A2:G"
        ).execute()
        
        rows = result.get('values', [])
        topics = []
        
        for row in rows:
            while len(row) < 7:
                row.append('')
            
            topic = {
                "topic": row[0],
                "context": row[1],
                "category": row[2],
                "priority": int(row[3]) if row[3] else 5,
                "added_date": row[4],
                "posts_generated": int(row[5]) if row[5] else 0,
                "last_searched": row[6] if row[6] else None
            }
            topics.append(topic)
        
        return {"topics": topics}
    except Exception as e:
        print(f"Error reading from Google Sheets: {e}")
        return {"topics": []}

def _write_topics_to_sheet(sheet_id: str, data: Dict[str, Any], sheet_name: str = "Topics") -> bool:
    """Write topics to Google Sheet synchronously."""
    try:
        service = _get_sheets_service()
        if not service:
            return False
        
        topics = data.get("topics", [])
        
        rows = [[
            "Topic", "Context", "Category", "Priority",
            "Added Date", "Posts Generated", "Last Searched"
        ]]
        
        for topic in topics:
            rows.append([
                topic.get("topic", ""),
                topic.get("context", ""),
                topic.get("category", ""),
                str(topic.get("priority", 5)),
                topic.get("added_date", ""),
                str(topic.get("posts_generated", 0)),
                topic.get("last_searched", "")
            ])
        
        service.spreadsheets().values().clear(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A:G"
        ).execute()
        
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A1",
            valueInputOption='RAW',
            body={'values': rows}
        ).execute()
        
        return True
    except Exception as e:
        print(f"Error writing to Google Sheets: {e}")
        return False

def _read_trading_data_from_sheet(sheet_id: str, sheet_name: str = "Calculations") -> Dict[str, Any]:
    """
    Read trading performance data from Google Sheets Calculations tab.

    Dynamically finds the Grand Total row (supports up to 5 years / 60 months).

    Sheet structure:
    - Row 1: Headers (Month, Profit By Month, Trades, Avg Profit/Trade, Monthly Avg, Daily Avg, Best Month)
    - Row 2+: Monthly data
    - Last row: Grand Total
    """
    try:
        service = _get_sheets_service()
        if not service:
            return {
                "error": "Google Sheets credentials not configured",
                "total_trades": 0,
                "win_rate": 1.0,
                "total_profit": 0,
                "avg_monthly_profit": 0,
                "last_updated": datetime.now().isoformat()
            }

        # Read up to 70 rows with UNFORMATTED values (raw numbers)
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=f"{sheet_name}!A1:H70",
            valueRenderOption='UNFORMATTED_VALUE'
        ).execute()

        rows = result.get('values', [])

        if not rows:
            return {
                "error": "No data found in sheet",
                "total_trades": 0,
                "win_rate": 1.0,
                "total_profit": 0,
                "avg_monthly_profit": 0,
                "last_updated": datetime.now().isoformat()
            }

        def safe_float(val):
            """Safely convert value to float."""
            if val is None or val == '':
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                # Try removing currency symbols if it's a string
                if isinstance(val, str):
                    cleaned = val.replace(', ', '').replace(',', '').strip()
                    try:
                        return float(cleaned)
                    except:
                        return 0.0
                return 0.0

        def safe_int(val):
            """Safely convert value to int."""
            if val is None or val == '':
                return 0
            try:
                return int(float(val))
            except (ValueError, TypeError):
                if isinstance(val, str):
                    cleaned = val.replace(',', '').strip()
                    try:
                        return int(float(cleaned))
                    except:
                        return 0
                return 0

        # Find the Grand Total row dynamically
        grand_total_row = None
        grand_total_index = None

        for i, row in enumerate(rows):
            if row and len(row) > 0:
                first_cell = str(row[0]).strip().lower()
                if first_cell == "grand total":
                    grand_total_row = row
                    grand_total_index = i
                    break

        if grand_total_row is None:
            return {
                "error": "Could not find Grand Total row",
                "total_trades": 0,
                "win_rate": 1.0,
                "total_profit": 0,
                "avg_monthly_profit": 0,
                "last_updated": datetime.now().isoformat()
            }

        # Extract values from Grand Total row
        # Column A=Month, B=Profit, C=Trades, D=Avg/Trade, E=Monthly Avg, F=Daily Avg, G=Best Month
        total_profit = safe_float(grand_total_row[1]) if len(grand_total_row) > 1 else 0
        total_trades = safe_int(grand_total_row[2]) if len(grand_total_row) > 2 else 0
        avg_per_trade = safe_float(grand_total_row[3]) if len(grand_total_row) > 3 else 0
        monthly_avg = safe_float(grand_total_row[4]) if len(grand_total_row) > 4 else 0
        daily_avg = safe_float(grand_total_row[5]) if len(grand_total_row) > 5 else 0
        best_month = safe_float(grand_total_row[6]) if len(grand_total_row) > 6 else 0

        # Get monthly breakdown - data starts at row 2 (index 1), ends before Grand Total
        monthly_data = []
        for row in rows[1:grand_total_index]:
            if len(row) >= 3 and row[0]:
                month_name = str(row[0]).strip()
                if month_name and month_name.lower() != "grand total" and month_name.lower() != "month":
                    monthly_data.append({
                        "month": month_name,
                        "profit": round(safe_float(row[1]), 2) if len(row) > 1 else 0,
                        "trades": safe_int(row[2]) if len(row) > 2 else 0
                    })

        return {
            "total_trades": total_trades,
            "win_rate": 1.0,
            "total_profit": round(total_profit, 2),
            "avg_per_trade": round(avg_per_trade, 2),
            "avg_monthly_profit": round(monthly_avg, 2),
            "avg_daily_profit": round(daily_avg, 2),
            "best_month": round(best_month, 2),
            "monthly_breakdown": monthly_data,
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error reading trading data from Google Sheets: {e}")
        return {
            "error": str(e),
            "total_trades": 0,
            "win_rate": 1.0,
            "total_profit": 0,
            "avg_monthly_profit": 0,
            "last_updated": datetime.now().isoformat()
        }

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

API_TIMEOUT = 30.0
MAX_ARTICLES = 10

# API Keys from environment
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

# Google Sheets configuration for topics storage
TOPICS_SHEET_ID = os.getenv("TOPICS_SHEET_ID", "")
TOPICS_SHEET_NAME = "Topics"

# ============================================================================
# ENUMS
# ============================================================================

class ContentType(str, Enum):
    """Types of content that can be generated."""
    BLOG_POST = "blog_post"
    TWEET_THREAD = "tweet_thread"
    SHORT_TWEET = "short_tweet"

class TopicCategory(str, Enum):
    """Categories for news topics."""
    CRYPTO = "crypto"
    REGULATION = "regulation"
    GAMBLING = "gambling"
    FINANCIAL_DISASTER = "financial_disaster"
    PASSIVE_INCOME = "passive_income"
    AI_TRADING = "ai_trading"
    DEFI = "defi"
    DEPIN = "depin"  # Added for DailyProfits
    GENERAL = "general"

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"

class Urgency(str, Enum):
    """Search urgency level for topics."""
    BREAKING = "breaking"
    TRENDING = "trending"
    EVERGREEN = "evergreen"

class Brand(str, Enum):
    """Brand context for content generation."""
    MYROBOTICTRADER = "myrobotictrader"
    DAILYPROFITS = "dailyprofits"

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AddTopicInput(BaseModel):
    """Input for adding a new topic to track."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    topic: str = Field(
        ...,
        description="The topic to track (e.g., '50-year mortgages', 'crypto gambling ban')",
        min_length=3,
        max_length=200
    )
    context: Optional[str] = Field(
        default=None,
        description="Why this topic matters / how it relates to your trading platform",
        max_length=500
    )
    category: TopicCategory = Field(
        default=TopicCategory.GENERAL,
        description="Category for organizing topics"
    )
    priority: int = Field(
        default=5,
        description="Priority level 1-10 (10 = highest priority)",
        ge=1,
        le=10
    )

class RemoveTopicInput(BaseModel):
    """Input for removing a tracked topic."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    topic: str = Field(
        ...,
        description="Exact topic name to remove",
        min_length=1
    )

class ListTopicsInput(BaseModel):
    """Input for listing tracked topics."""
    model_config = ConfigDict(validate_assignment=True)
    
    category: Optional[TopicCategory] = Field(
        default=None,
        description="Filter by category (optional)"
    )
    min_priority: Optional[int] = Field(
        default=None,
        description="Show only topics with priority >= this value",
        ge=1,
        le=10
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class SearchTopicInput(BaseModel):
    """Input for searching news on a specific topic."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    topic: str = Field(
        ...,
        description="Topic to search for (e.g., 'sports betting legalization', 'crypto regulation')",
        min_length=3,
        max_length=200
    )
    urgency: Urgency = Field(
        default=Urgency.TRENDING,
        description="How urgent/current the search should be"
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of articles to return",
        ge=1,
        le=50
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class FetchNewsInput(BaseModel):
    """Input for fetching general crypto/financial news."""
    model_config = ConfigDict(validate_assignment=True)
    
    categories: List[TopicCategory] = Field(
        default_factory=lambda: [TopicCategory.CRYPTO, TopicCategory.AI_TRADING],
        description="News categories to fetch"
    )
    days_back: int = Field(
        default=3,
        description="How many days back to search",
        ge=1,
        le=30
    )
    max_results: int = Field(
        default=10,
        description="Maximum articles per category",
        ge=1,
        le=50
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class FetchTradingDataInput(BaseModel):
    """Input for fetching trading performance data."""
    model_config = ConfigDict(validate_assignment=True)
    
    sheet_id: str = Field(
        ...,
        description="Google Sheet ID containing trading data",
        min_length=10
    )
    sheet_name: Optional[str] = Field(
        default="Sheet1",
        description="Name of the sheet tab to read from"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format"
    )

class GenerateContentInput(BaseModel):
    """Input for generating content."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    content_type: ContentType = Field(
        ...,
        description="Type of content to generate"
    )
    topic: str = Field(
        ...,
        description="Main topic or angle for the content",
        min_length=5,
        max_length=300
    )
    news_context: Optional[str] = Field(
        default=None,
        description="Recent news or stories to reference (from search results)",
        max_length=5000
    )
    trading_data: Optional[str] = Field(
        default=None,
        description="Your trading performance data to include (from fetch_trading_data)",
        max_length=2000
    )
    tone: str = Field(
        default="contrarian",
        description="Tone: contrarian, educational, provocative, inspirational",
        pattern="^(contrarian|educational|provocative|inspirational)$"
    )
    cta_url: Optional[str] = Field(
        default="https://myrobotictrader.com",
        description="Call-to-action URL to include"
    )

# Discord-specific models
class DiscordChannelInput(BaseModel):
    """Input for Discord channel operations."""
    channel_id: str = Field(
        ...,
        description="Discord channel ID to monitor",
        min_length=15,
        max_length=25
    )
    limit: int = Field(
        default=50,
        description="Number of messages to fetch",
        ge=1,
        le=100
    )

class DiscordSearchInput(BaseModel):
    """Input for searching Discord messages."""
    channel_id: str = Field(
        ...,
        description="Discord channel ID to search",
        min_length=15,
        max_length=25
    )
    keywords: List[str] = Field(
        ...,
        description="Keywords to search for",
        min_length=1
    )
    limit: int = Field(
        default=100,
        description="Number of messages to scan",
        ge=1,
        le=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class DiscordGuildInput(BaseModel):
    """Input for Discord guild/server operations."""
    guild_id: str = Field(
        ...,
        description="Discord server/guild ID",
        min_length=15,
        max_length=25
    )

# CoinMarketCap models
class GetCryptoPriceInput(BaseModel):
    """Input for getting cryptocurrency price."""
    symbol: str = Field(
        ...,
        description="Crypto symbol (e.g., 'BTC', 'ETH', 'SOL')",
        min_length=1,
        max_length=10
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class GetTopCryptosInput(BaseModel):
    """Input for getting top cryptocurrencies."""
    limit: int = Field(
        default=10,
        description="Number of top cryptos to return",
        ge=1,
        le=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

class GetMarketMoversInput(BaseModel):
    """Input for getting market gainers and losers."""
    limit: int = Field(
        default=10,
        description="Number of gainers/losers to return",
        ge=1,
        le=50
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def load_topics() -> Dict[str, Any]:
    """Load topics from Google Sheets."""
    sheet_id = TOPICS_SHEET_ID
    
    if not sheet_id:
        print("⚠️  TOPICS_SHEET_ID not set. Using in-memory storage.")
        return {"topics": []}
    
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read_topics_from_sheet, sheet_id, "Topics")
    except Exception as e:
        print(f"⚠️  Error loading topics from Google Sheets: {e}")
        return {"topics": []}

async def save_topics(data: Dict[str, Any]) -> None:
    """Save topics to Google Sheets."""
    sheet_id = TOPICS_SHEET_ID
    
    if not sheet_id:
        print("⚠️  TOPICS_SHEET_ID not set. Topics not persisted.")
        return
    
    try:
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, _write_topics_to_sheet, sheet_id, data, "Topics")
        if success:
            print(f"✓ Saved {len(data.get('topics', []))} topics to Google Sheets")
    except Exception as e:
        print(f"⚠️  Error saving topics to Google Sheets: {e}")

def format_error(error: Exception) -> str:
    """Format errors consistently."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP Error {error.response.status_code}: {error.response.text[:200]}"
    return f"Error: {type(error).__name__}: {str(error)}"

# ============================================================================
# NEWSAPI INTEGRATION
# ============================================================================

async def search_news_api(query: str, max_results: int = 10, days_back: int = 7) -> List[Dict[str, Any]]:
    """
    Search for news articles using NewsAPI.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        days_back: How many days back to search
    
    Returns:
        List of article dictionaries
    """
    if not NEWSAPI_KEY:
        return [{
            "title": "NewsAPI not configured",
            "url": "",
            "source": "System",
            "published_at": datetime.now().isoformat(),
            "description": "Set NEWSAPI_KEY in your .env file to enable real news search.",
            "relevance_score": 0
        }]
    
    # Calculate date range
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWSAPI_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": min(max_results, 100),
        "from": from_date
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "published_at": article.get("publishedAt", ""),
                    "description": article.get("description", "")[:300] if article.get("description") else "",
                    "relevance_score": 0.9  # NewsAPI doesn't provide this
                })
            
            return articles
            
    except httpx.HTTPStatusError as e:
        return [{
            "title": f"NewsAPI Error: {e.response.status_code}",
            "url": "",
            "source": "Error",
            "published_at": datetime.now().isoformat(),
            "description": e.response.text[:200],
            "relevance_score": 0
        }]
    except Exception as e:
        return [{
            "title": f"Error: {str(e)}",
            "url": "",
            "source": "Error",
            "published_at": datetime.now().isoformat(),
            "description": str(e),
            "relevance_score": 0
        }]

async def fetch_google_sheet_data(sheet_id: str, sheet_name: str) -> Dict[str, Any]:
    """Fetch data from Google Sheets using the real integration."""
    return _read_trading_data_from_sheet(sheet_id)

# ============================================================================
# MCP TOOLS - TOPIC MANAGEMENT
# ============================================================================

@mcp.tool(
    name="add_topic",
    annotations={
        "title": "Add Topic to Track",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def add_topic(params: AddTopicInput) -> str:
    """
    Add a new topic to your watchlist for content generation.
    
    Use this when you hear about breaking news or trending topics that you want
    to create content about. The topic will be saved and can be referenced later.
    
    Args:
        params: AddTopicInput containing:
            - topic: The topic name (e.g., "50-year mortgages")
            - context: Why this matters for your audience
            - category: Topic category for organization
            - priority: 1-10 priority level
    
    Returns:
        Success message with topic details in JSON format
    """
    try:
        data = await load_topics()
        
        # Check for duplicates
        for existing in data["topics"]:
            if existing["topic"].lower() == params.topic.lower():
                return json.dumps({
                    "status": "exists",
                    "message": f"Topic already exists: {params.topic}",
                    "topic": existing
                }, indent=2)
        
        # Create new topic
        new_topic = {
            "topic": params.topic,
            "context": params.context or "",
            "category": params.category.value,
            "priority": params.priority,
            "added_date": datetime.now().strftime("%Y-%m-%d"),
            "posts_generated": 0,
            "last_searched": ""
        }
        
        data["topics"].append(new_topic)
        await save_topics(data)
        
        return json.dumps({
            "status": "success",
            "message": f"Added topic: {params.topic}",
            "topic": new_topic
        }, indent=2)
        
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="remove_topic",
    annotations={
        "title": "Remove Topic from Watchlist",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def remove_topic(params: RemoveTopicInput) -> str:
    """
    Remove a topic from your watchlist.
    
    Use this to clean up old or irrelevant topics that you no longer want to track.
    
    Args:
        params: RemoveTopicInput containing:
            - topic: Exact topic name to remove
    
    Returns:
        Success or error message in JSON format
    """
    try:
        data = await load_topics()
        
        original_count = len(data["topics"])
        data["topics"] = [t for t in data["topics"] if t["topic"].lower() != params.topic.lower()]
        
        if len(data["topics"]) == original_count:
            return json.dumps({
                "status": "not_found",
                "message": f"Topic not found: {params.topic}"
            }, indent=2)
        
        await save_topics(data)
        
        return json.dumps({
            "status": "success",
            "message": f"Removed topic: {params.topic}"
        }, indent=2)
        
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="list_topics",
    annotations={
        "title": "List Tracked Topics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def list_topics(params: ListTopicsInput) -> str:
    """
    List all topics you're currently tracking.
    
    Optionally filter by category or minimum priority level.
    Shows topics sorted by priority (highest first).
    
    Args:
        params: ListTopicsInput containing:
            - category: Optional category filter
            - min_priority: Optional minimum priority threshold
            - response_format: "markdown" or "json"
    
    Returns:
        List of topics in requested format
    """
    try:
        data = await load_topics()
        topics = data["topics"]
        
        # Filter by category
        if params.category:
            topics = [t for t in topics if t["category"] == params.category.value]
        
        # Filter by priority
        if params.min_priority:
            topics = [t for t in topics if t["priority"] >= params.min_priority]
        
        # Sort by priority (highest first)
        topics.sort(key=lambda t: t["priority"], reverse=True)
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "total_topics": len(topics),
                "filters": {
                    "category": params.category.value if params.category else None,
                    "min_priority": params.min_priority
                },
                "topics": topics
            }, indent=2)
        
        # Markdown format
        if not topics:
            return "# Tracked Topics\n\nNo topics found matching your filters."
        
        output = f"# Tracked Topics ({len(topics)} total)\n\n"
        
        for topic in topics:
            output += f"## [{topic['priority']}/10] {topic['topic']}\n"
            output += f"- **Category**: {topic['category']}\n"
            if topic['context']:
                output += f"- **Context**: {topic['context']}\n"
            output += f"- **Added**: {topic['added_date']}\n"
            output += f"- **Posts Generated**: {topic['posts_generated']}\n\n"
        
        return output
        
    except Exception as e:
        return format_error(e)

# ============================================================================
# MCP TOOLS - NEWS & SEARCH
# ============================================================================

@mcp.tool(
    name="search_topic_now",
    annotations={
        "title": "Search News for Topic",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def search_topic_now(params: SearchTopicInput) -> str:
    """
    Search for current news and articles about a specific topic.
    
    Use this when you want to research a topic before writing content,
    or when you need to find the latest information on a trending story.
    
    Args:
        params: SearchTopicInput containing:
            - topic: What to search for
            - urgency: breaking/trending/evergreen
            - max_results: Number of articles to return
            - response_format: "markdown" or "json"
    
    Returns:
        News articles and stories about the topic
    """
    try:
        # Determine days_back based on urgency
        days_map = {
            Urgency.BREAKING: 1,
            Urgency.TRENDING: 7,
            Urgency.EVERGREEN: 30
        }
        days_back = days_map.get(params.urgency, 7)
        
        articles = await search_news_api(params.topic, params.max_results, days_back)
        
        # Update topic's last_searched if it exists
        data = await load_topics()
        for topic in data["topics"]:
            if params.topic.lower() in topic["topic"].lower():
                topic["last_searched"] = datetime.now().isoformat()
        await save_topics(data)
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "topic": params.topic,
                "urgency": params.urgency.value,
                "articles_found": len(articles),
                "articles": articles
            }, indent=2)
        
        # Markdown format
        output = f"# News Search: {params.topic}\n\n"
        output += f"*Urgency: {params.urgency.value} | Found {len(articles)} articles*\n\n"
        
        for i, article in enumerate(articles, 1):
            output += f"## {i}. {article['title']}\n"
            output += f"- **Source**: {article['source']}\n"
            output += f"- **Published**: {article['published_at'][:10] if article['published_at'] else 'Unknown'}\n"
            output += f"- **URL**: {article['url']}\n"
            output += f"- **Summary**: {article['description']}\n\n"
        
        return output
        
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="fetch_crypto_news",
    annotations={
        "title": "Fetch General Crypto & Financial News",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def fetch_crypto_news(params: FetchNewsInput) -> str:
    """
    Fetch the latest crypto, financial, and trading news.
    
    Gets news from multiple categories to give you content ideas and
    current events to reference in your posts.
    
    Args:
        params: FetchNewsInput containing:
            - categories: List of news categories to fetch
            - days_back: How far back to search (1-30 days)
            - max_results: Articles per category
            - response_format: "markdown" or "json"
    
    Returns:
        News articles organized by category
    """
    try:
        results = {}
        
        for category in params.categories:
            # Build search query based on category
            query_map = {
                TopicCategory.CRYPTO: "cryptocurrency bitcoin ethereum",
                TopicCategory.REGULATION: "crypto regulation SEC cryptocurrency law",
                TopicCategory.GAMBLING: "online gambling sports betting crypto gambling",
                TopicCategory.FINANCIAL_DISASTER: "trading loss bankruptcy crypto scam",
                TopicCategory.PASSIVE_INCOME: "passive income side hustle cryptocurrency mining",
                TopicCategory.AI_TRADING: "AI trading automated trading algorithmic",
                TopicCategory.DEFI: "DeFi decentralized finance yield farming",
                TopicCategory.DEPIN: "DePIN decentralized infrastructure Web3 node"
            }
            
            query = query_map.get(category, category.value)
            articles = await search_news_api(query, params.max_results, params.days_back)
            results[category.value] = articles
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "categories": list(results.keys()),
                "days_back": params.days_back,
                "total_articles": sum(len(articles) for articles in results.values()),
                "results": results
            }, indent=2)
        
        # Markdown format
        output = "# Crypto & Financial News Digest\n\n"
        
        for category, articles in results.items():
            output += f"## {category.upper().replace('_', ' ')}\n"
            output += f"*{len(articles)} articles*\n\n"
            
            for article in articles:
                output += f"### {article['title']}\n"
                output += f"- {article['source']} | {article['published_at'][:10] if article['published_at'] else 'Unknown'}\n"
                output += f"- {article['description']}\n"
                output += f"- [Read more]({article['url']})\n\n"
        
        return output
        
    except Exception as e:
        return format_error(e)

# ============================================================================
# MCP TOOLS - DISCORD
# ============================================================================

@mcp.tool(
    name="discord_list_servers",
    annotations={
        "title": "List Discord Servers",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def discord_list_servers() -> str:
    """
    List all Discord servers the bot is in.
    
    Use this to see which servers you can monitor for content ideas.
    
    Returns:
        List of servers with their IDs
    """
    try:
        guilds = await get_bot_guilds()
        return format_guilds_markdown(guilds)
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="discord_list_channels",
    annotations={
        "title": "List Discord Channels",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def discord_list_channels(params: DiscordGuildInput) -> str:
    """
    List all channels in a Discord server.
    
    Use this to find channel IDs for monitoring.
    
    Args:
        params: DiscordGuildInput containing:
            - guild_id: Discord server/guild ID
    
    Returns:
        List of channels with their IDs
    """
    try:
        channels = await get_guild_channels(params.guild_id)
        return format_channels_markdown(channels)
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="discord_read_messages",
    annotations={
        "title": "Read Discord Messages",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def discord_read_messages(params: DiscordChannelInput) -> str:
    """
    Read recent messages from a Discord channel.
    
    Use this to see what people are talking about in project channels.
    
    Args:
        params: DiscordChannelInput containing:
            - channel_id: Discord channel ID
            - limit: Number of messages to fetch (max 100)
    
    Returns:
        Recent messages from the channel
    """
    try:
        messages = await fetch_channel_messages(params.channel_id, params.limit)
        return format_messages_markdown(messages, "Recent Messages")
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="discord_search_keywords",
    annotations={
        "title": "Search Discord for Keywords",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def discord_search_keywords(params: DiscordSearchInput) -> str:
    """
    Search Discord channel messages for specific keywords.
    
    Great for finding:
    - TGE announcements
    - Airdrop mentions
    - Trading signals
    - Project updates
    
    Args:
        params: DiscordSearchInput containing:
            - channel_id: Discord channel ID
            - keywords: List of keywords to search for
            - limit: Number of messages to scan
            - response_format: "markdown" or "json"
    
    Returns:
        Messages containing any of the keywords
    """
    try:
        messages = await search_messages_for_keywords(
            params.channel_id,
            params.keywords,
            params.limit
        )
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "channel_id": params.channel_id,
                "keywords": params.keywords,
                "matches_found": len(messages),
                "messages": messages
            }, indent=2)
        
        return format_messages_markdown(
            messages,
            f"Keyword Search: {', '.join(params.keywords)}"
        )
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="discord_monitor_announcements",
    annotations={
        "title": "Monitor Discord for Announcements",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def discord_monitor_announcements(params: DiscordChannelInput) -> str:
    """
    Monitor a Discord channel for important announcements.
    
    Automatically searches for keywords like:
    - TGE, token, airdrop, launch
    - mainnet, testnet, staking
    - urgent, important, breaking
    
    Perfect for DePIN project tracking (Dabba, TIMPI, Aethir, etc.)
    
    Args:
        params: DiscordChannelInput containing:
            - channel_id: Discord channel ID
            - limit: Number of messages to scan
    
    Returns:
        Important announcements found
    """
    try:
        messages = await monitor_for_announcements(params.channel_id)
        return format_messages_markdown(messages, "🔔 Announcements & Alerts")
    except Exception as e:
        return format_error(e)

# ============================================================================
# MCP TOOLS - COINMARKETCAP
# ============================================================================

@mcp.tool(
    name="get_crypto_price",
    annotations={
        "title": "Get Cryptocurrency Price",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def get_crypto_price(params: GetCryptoPriceInput) -> str:
    """
    Get current price and market data for a specific cryptocurrency.
    
    Use this to reference current crypto prices in your content,
    compare performance, or provide market context.
    
    Args:
        params: GetCryptoPriceInput containing:
            - symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'SOL')
            - response_format: "markdown" or "json"
    
    Returns:
        Current price, changes, market cap, and volume data
    """
    try:
        data = await fetch_crypto_price(params.symbol.upper())
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)
        
        return format_crypto_data_markdown(data)
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="get_top_cryptos",
    annotations={
        "title": "Get Top Cryptocurrencies",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def get_top_cryptos(params: GetTopCryptosInput) -> str:
    """
    Get top cryptocurrencies by market cap.
    
    Use this to reference the overall crypto market, discuss
    major coins, or provide market overview in your content.
    
    Args:
        params: GetTopCryptosInput containing:
            - limit: Number of top cryptos (1-100)
            - response_format: "markdown" or "json"
    
    Returns:
        List of top cryptocurrencies with prices and market data
    """
    try:
        data = await fetch_top_cryptos(params.limit)
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)
        
        return format_top_cryptos_markdown(data)
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="get_market_movers",
    annotations={
        "title": "Get Market Gainers & Losers",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def get_market_movers(params: GetMarketMoversInput) -> str:
    """
    Get top gainers and losers in the crypto market.
    
    Use this to discuss market volatility, highlight trading
    opportunities, or provide market sentiment context.
    
    Args:
        params: GetMarketMoversInput containing:
            - limit: Number of gainers/losers (1-50)
            - response_format: "markdown" or "json"
    
    Returns:
        Lists of top gainers and losers with 24h performance
    """
    try:
        data = await fetch_gainers_losers(params.limit)
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)
        
        return format_gainers_losers_markdown(data)
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="get_fear_greed_index",
    annotations={
        "title": "Get Fear & Greed Index",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def get_fear_greed_index() -> str:
    """
    Get the current Crypto Fear & Greed Index from CoinMarketCap.
    
    Use this to discuss market sentiment, provide context for
    trading decisions, or reference emotional market conditions.
    
    Returns:
        Current fear/greed index value and classification
    """
    try:
        data = await fetch_fear_greed_index()
        
        if "error" in data:
            return f"Error fetching Fear & Greed Index: {data['error']}"
        
        value = data.get("value", 0)
        classification = data.get("value_classification", "Unknown")
        
        # Add emoji based on classification
        emoji_map = {
            "Extreme Fear": "😱",
            "Fear": "😰",
            "Neutral": "😐",
            "Greed": "🤑",
            "Extreme Greed": "🚀"
        }
        emoji = emoji_map.get(classification, "📊")
        
        output = f"# {emoji} Crypto Fear & Greed Index\n\n"
        output += f"**Current Value:** {value}/100\n"
        output += f"**Classification:** {classification}\n\n"
        
        if value < 25:
            output += "*Market is in extreme fear - historically a buying opportunity.*"
        elif value < 45:
            output += "*Market is fearful - caution prevails.*"
        elif value < 55:
            output += "*Market is neutral - balanced sentiment.*"
        elif value < 75:
            output += "*Market is greedy - optimism is high.*"
        else:
            output += "*Market is extremely greedy - potential correction ahead.*"
        
        return output
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="get_altcoin_season_index",
    annotations={
        "title": "Get Altcoin Season Index",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def get_altcoin_season_index() -> str:
    """
    Get the current Altcoin Season Index from CoinMarketCap.
    
    This index measures whether the market is in Bitcoin or Altcoin season:
    - 0-25: Bitcoin Season (BTC dominance, alts underperforming)
    - 25-75: Mixed Market (no clear trend)
    - 75-100: Altcoin Season (alts outperforming BTC)
    
    Use this to:
    - Identify market cycle phases
    - Adjust trading strategy (BTC vs alts)
    - Provide market context in content
    - Discuss portfolio allocation
    
    Returns:
        Current altcoin season index value and classification
    """
    try:
        data = await fetch_altcoin_season_index()
        
        if not data or "error" in data:
            return f"Error fetching Altcoin Season Index: {data.get('error', 'Unknown error')}"
        
        value = data.get("value", 0)
        classification = data.get("classification", "Unknown")
        
        # Add emoji based on classification
        emoji_map = {
            "Bitcoin Season": "🔵",
            "Mixed Market": "🟡",
            "Altcoin Season": "🟢"
        }
        emoji = emoji_map.get(classification, "📊")
        
        output = f"# {emoji} Altcoin Season Index\n\n"
        output += f"**Current Value:** {value}/100\n"
        output += f"**Classification:** {classification}\n\n"
        
        if value <= 25:
            output += "*Bitcoin is dominating. Altcoins are underperforming BTC. Consider BTC-heavy portfolio.*"
        elif value <= 50:
            output += "*Mixed market with slight Bitcoin preference. No strong trend established.*"
        elif value <= 75:
            output += "*Mixed market with slight Altcoin preference. Some alts starting to outperform.*"
        else:
            output += "*Altcoin Season! Majority of altcoins are outperforming Bitcoin. Alt-heavy strategies favorable.*"
        
        output += "\n\n**What this means:**\n"
        output += f"- **{value}% of top 100 altcoins** have outperformed Bitcoin in the past 90 days\n"
        output += f"- This indicates {'strong altcoin momentum' if value > 75 else 'Bitcoin dominance' if value < 25 else 'a balanced market'}\n"
        
        return output
    except Exception as e:
        return format_error(e)

# ============================================================================
# MCP TOOLS - TRADING DATA & CONTENT
# ============================================================================

@mcp.tool(
    name="fetch_trading_data",
    annotations={
        "title": "Fetch Your Trading Performance Data",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def fetch_trading_data(params: FetchTradingDataInput) -> str:
    """
    Fetch your current trading performance data from Google Sheets.
    
    Gets your live trading metrics to include in content:
    - Total trades, win rate, profits
    - Recent performance
    - Key statistics
    
    Args:
        params: FetchTradingDataInput containing:
            - sheet_id: Your Google Sheet ID
            - sheet_name: Sheet tab name (default: "Sheet1")
            - response_format: "json" or "markdown"
    
    Returns:
        Your trading performance metrics
    """
    try:
        data = await fetch_google_sheet_data(params.sheet_id, params.sheet_name)
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)
        
        output = "# Trading Performance Summary\n\n"
        output += f"- **Total Trades**: {data['total_trades']}\n"
        output += f"- **Win Rate**: {data['win_rate'] * 100}%\n"
        output += f"- **Total Profit**: ${data['total_profit']:,.2f}\n"
        output += f"- **Avg Monthly Profit**: ${data['avg_monthly_profit']:,.2f}\n"
        output += f"- **Last Updated**: {data['last_updated'][:10]}\n"
        
        return output
        
    except Exception as e:
        return format_error(e)

@mcp.tool(
    name="generate_content",
    annotations={
        "title": "Generate Blog Post or Social Content",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def generate_content(params: GenerateContentInput, ctx: Context) -> str:
    """
    Generate blog posts, tweets, or social media content.
    
    Combines news context and your trading data to create engaging content
    that positions your robotic trader as the solution.
    
    Args:
        params: GenerateContentInput containing:
            - content_type: blog_post, tweet_thread, or short_tweet
            - topic: Main topic or angle
            - news_context: Recent news to reference (optional)
            - trading_data: Your performance data (optional)
            - tone: contrarian, educational, provocative, or inspirational
            - cta_url: Call-to-action link
    
    Returns:
        Generated content ready to publish
    """
    try:
        prompt_parts = []
        
        if params.content_type == ContentType.BLOG_POST:
            prompt_parts.append("Write a blog post (800-1200 words) with:")
            prompt_parts.append("- Attention-grabbing headline")
            prompt_parts.append("- Hook using the news/topic")
            prompt_parts.append("- Problem identification")
            prompt_parts.append("- Solution (systematic AI trading)")
            prompt_parts.append("- Proof (trading data)")
            prompt_parts.append("- Strong CTA")
            
        elif params.content_type == ContentType.TWEET_THREAD:
            prompt_parts.append("Write a Twitter thread (8-12 tweets):")
            prompt_parts.append("- Hook tweet that stops scrolling")
            prompt_parts.append("- Thread with story progression")
            prompt_parts.append("- Include trading data")
            prompt_parts.append("- End with CTA")
            
        else:
            prompt_parts.append("Write a single tweet (280 chars max):")
            prompt_parts.append("- Punchy and provocative")
            prompt_parts.append("- Clear value prop")
            prompt_parts.append("- Include CTA link")
        
        prompt_parts.append(f"\nTopic: {params.topic}")
        prompt_parts.append(f"Tone: {params.tone}")
        
        if params.news_context:
            prompt_parts.append(f"\nNews Context to Reference:\n{params.news_context[:500]}")
        
        if params.trading_data:
            prompt_parts.append(f"\nTrading Performance to Highlight:\n{params.trading_data}")
        
        prompt_parts.append(f"\nCTA URL: {params.cta_url}")
        
        generated_content = "\n\n".join(prompt_parts)
        generated_content += "\n\n# Generated content\n\n"
        generated_content += f"Create a {params.content_type.value} about: {params.topic}\n\n"
        
        if params.news_context:
            generated_content += f"Recent News Context:\n{params.news_context[:300]}...\n\n"
        
        generated_content += f"Tone: {params.tone}\n"
        generated_content += f"CTA URL: {params.cta_url}\n\n"
        generated_content += "Generate engaging content that educates readers about systematic trading and positions AI-enhanced autonomous trading as superior to gambling."
        
        # Update topic post count
        data = await load_topics()
        for topic in data["topics"]:
            if params.topic.lower() in topic["topic"].lower():
                topic["posts_generated"] += 1
        await save_topics(data)
        
        return generated_content
        
    except Exception as e:
        return format_error(e)

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import os as _os
    _transport = _os.environ.get("MCP_TRANSPORT", "stdio")
    if _transport == "sse":
        mcp.settings.host = _os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(_os.environ.get("MCP_PORT", "8083"))
        mcp.settings.transport_security.enable_dns_rebinding_protection = False
        mcp.run(transport="sse")
    else:
        mcp.run()