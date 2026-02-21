"""
Discord Integration Tools for MyRoboticTrader MCP Server

Provides tools to:
- Monitor Discord channels for keywords
- Fetch recent messages from channels (including embeds!)
- Summarize Discord activity
- Track project announcements (Dabba, TIMPI, Aethir, Flux, etc.)

Author: Built for Patrick Jenkins
"""

import os
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

# Discord API base URL
DISCORD_API_BASE = "https://discord.com/api/v10"

# Get bot token from environment
def get_bot_token() -> Optional[str]:
    """Get Discord bot token from environment."""
    return os.getenv("DISCORD_BOT_TOKEN")

def get_headers() -> Dict[str, str]:
    """Get headers for Discord API requests."""
    token = get_bot_token()
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN not set in environment")
    return {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json"
    }

# ============================================================================
# EMBED CONTENT EXTRACTION
# ============================================================================

def extract_embed_content(embeds: List[Dict[str, Any]]) -> str:
    """
    Extract readable text content from Discord embeds.
    
    Embeds contain: title, description, fields, author, footer
    """
    if not embeds:
        return ""
    
    content_parts = []
    
    for embed in embeds:
        # Get title
        if embed.get("title"):
            content_parts.append(f"**{embed['title']}**")
        
        # Get author name
        if embed.get("author", {}).get("name"):
            content_parts.append(f"By: {embed['author']['name']}")
        
        # Get description (main content)
        if embed.get("description"):
            content_parts.append(embed["description"])
        
        # Get fields (often contain key info)
        for field in embed.get("fields", []):
            field_name = field.get("name", "")
            field_value = field.get("value", "")
            if field_name and field_value:
                content_parts.append(f"{field_name}: {field_value}")
        
        # Get footer
        if embed.get("footer", {}).get("text"):
            content_parts.append(f"— {embed['footer']['text']}")
    
    return "\n".join(content_parts)

def extract_full_message_content(message: Dict[str, Any]) -> str:
    """
    Extract all content from a message including embeds.
    
    Returns combined text from message content + embeds.
    """
    parts = []
    
    # Regular message content
    if message.get("content"):
        parts.append(message["content"])
    
    # Embed content
    embed_text = extract_embed_content(message.get("embeds", []))
    if embed_text:
        parts.append(embed_text)
    
    return "\n".join(parts)

# ============================================================================
# DISCORD API FUNCTIONS
# ============================================================================

async def fetch_channel_messages(
    channel_id: str,
    limit: int = 50,
    before: Optional[str] = None,
    after: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch messages from a Discord channel.
    
    Args:
        channel_id: The Discord channel ID
        limit: Number of messages to fetch (max 100)
        before: Get messages before this message ID
        after: Get messages after this message ID
    
    Returns:
        List of message objects with extracted content
    """
    if not get_bot_token():
        return [{"error": "DISCORD_BOT_TOKEN not configured"}]
    
    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    params = {"limit": min(limit, 100)}
    
    if before:
        params["before"] = before
    if after:
        params["after"] = after
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=get_headers(), params=params)
            response.raise_for_status()
            messages = response.json()
            
            # Enrich messages with extracted embed content
            for msg in messages:
                msg["full_content"] = extract_full_message_content(msg)
            
            return messages
    except httpx.HTTPStatusError as e:
        return [{"error": f"Discord API error: {e.response.status_code}", "detail": e.response.text}]
    except Exception as e:
        return [{"error": str(e)}]

async def search_messages_for_keywords(
    channel_id: str,
    keywords: List[str],
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Search channel messages for specific keywords (including embeds).
    
    Args:
        channel_id: The Discord channel ID
        keywords: List of keywords to search for
        limit: Number of messages to scan
    
    Returns:
        List of messages containing any of the keywords
    """
    messages = await fetch_channel_messages(channel_id, limit=limit)
    
    if messages and "error" in messages[0]:
        return messages
    
    matching = []
    keywords_lower = [k.lower() for k in keywords]
    
    for msg in messages:
        # Search in full content (includes embeds)
        full_content = msg.get("full_content", "").lower()
        
        if any(kw in full_content for kw in keywords_lower):
            # Get source info from author or reference
            author = msg.get("author", {}).get("username", "Unknown")
            
            # Check if it's a webhook (forwarded announcement)
            if msg.get("webhook_id"):
                author = msg.get("author", {}).get("username", "Announcement")
            
            matching.append({
                "id": msg.get("id"),
                "content": msg.get("full_content", "")[:500],  # Truncate for readability
                "author": author,
                "timestamp": msg.get("timestamp"),
                "matched_keywords": [kw for kw in keywords if kw.lower() in full_content]
            })
    
    return matching

async def get_guild_channels(guild_id: str) -> List[Dict[str, Any]]:
    """
    Get all channels in a Discord server (guild).
    
    Args:
        guild_id: The Discord server/guild ID
    
    Returns:
        List of channel objects
    """
    if not get_bot_token():
        return [{"error": "DISCORD_BOT_TOKEN not configured"}]
    
    url = f"{DISCORD_API_BASE}/guilds/{guild_id}/channels"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=get_headers())
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return [{"error": f"Discord API error: {e.response.status_code}", "detail": e.response.text}]
    except Exception as e:
        return [{"error": str(e)}]

async def get_bot_guilds() -> List[Dict[str, Any]]:
    """
    Get all servers (guilds) the bot is in.
    
    Returns:
        List of guild objects
    """
    if not get_bot_token():
        return [{"error": "DISCORD_BOT_TOKEN not configured"}]
    
    url = f"{DISCORD_API_BASE}/users/@me/guilds"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=get_headers())
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return [{"error": f"Discord API error: {e.response.status_code}", "detail": e.response.text}]
    except Exception as e:
        return [{"error": str(e)}]

# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_messages_markdown(messages: List[Dict[str, Any]], title: str = "Discord Messages") -> str:
    """Format messages as markdown."""
    if not messages:
        return f"# {title}\n\nNo messages found."
    
    if "error" in messages[0]:
        return f"# {title}\n\n**Error:** {messages[0].get('error')}\n{messages[0].get('detail', '')}"
    
    output = f"# {title}\n\n"
    output += f"*{len(messages)} messages*\n\n"
    
    for msg in messages[:25]:  # Limit output
        author = msg.get("author", "Unknown")
        if isinstance(author, dict):
            author = author.get("username", "Unknown")
        
        timestamp = msg.get("timestamp", "")[:10] if msg.get("timestamp") else ""
        
        # Use full_content which includes embeds
        content = msg.get("full_content") or msg.get("content", "")
        content = content[:300] if content else "[No text content]"
        
        output += f"---\n"
        output += f"**{author}** ({timestamp})\n\n"
        output += f"{content}\n\n"
        
        if msg.get("matched_keywords"):
            output += f"*🔍 Matched: {', '.join(msg['matched_keywords'])}*\n\n"
    
    return output

def format_channels_markdown(channels: List[Dict[str, Any]]) -> str:
    """Format channels list as markdown."""
    if not channels:
        return "# Discord Channels\n\nNo channels found."
    
    if "error" in channels[0]:
        return f"# Discord Channels\n\n**Error:** {channels[0].get('error')}"
    
    output = "# Discord Channels\n\n"
    
    # Group by type
    text_channels = [c for c in channels if c.get("type") == 0]
    voice_channels = [c for c in channels if c.get("type") == 2]
    categories = [c for c in channels if c.get("type") == 4]
    
    if categories:
        output += "## Categories\n"
        for c in categories:
            output += f"- 📁 {c.get('name')} (ID: `{c.get('id')}`)\n"
        output += "\n"
    
    if text_channels:
        output += "## Text Channels\n"
        for c in text_channels:
            output += f"- 💬 #{c.get('name')} (ID: `{c.get('id')}`)\n"
        output += "\n"
    
    if voice_channels:
        output += "## Voice Channels\n"
        for c in voice_channels:
            output += f"- 🔊 {c.get('name')} (ID: `{c.get('id')}`)\n"
    
    return output

def format_guilds_markdown(guilds: List[Dict[str, Any]]) -> str:
    """Format guilds list as markdown."""
    if not guilds:
        return "# Discord Servers\n\nBot is not in any servers."
    
    if "error" in guilds[0]:
        return f"# Discord Servers\n\n**Error:** {guilds[0].get('error')}"
    
    output = "# Discord Servers\n\n"
    output += f"*Bot is in {len(guilds)} servers*\n\n"
    
    for g in guilds:
        output += f"- **{g.get('name')}** (ID: `{g.get('id')}`)\n"
    
    return output

# ============================================================================
# CONTENT-FOCUSED HELPERS
# ============================================================================

# Default keywords for different content types
DEPIN_KEYWORDS = [
    "TGE", "token", "airdrop", "launch", "mainnet", "testnet",
    "staking", "rewards", "announcement", "update", "release",
    "snapshot", "claim", "distribution", "listing", "exchange"
]

TRADING_KEYWORDS = [
    "pump", "dump", "moon", "dip", "buy", "sell", "profit",
    "loss", "whale", "volume", "breakout", "support", "resistance"
]

ALERT_KEYWORDS = [
    "urgent", "important", "breaking", "announcement", "warning",
    "scam", "rug", "hack", "exploit", "attention", "critical"
]

async def monitor_for_announcements(
    channel_id: str,
    keywords: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Monitor a channel for important announcements.
    
    Args:
        channel_id: The Discord channel ID
        keywords: Custom keywords (defaults to DEPIN_KEYWORDS + ALERT_KEYWORDS)
    
    Returns:
        List of relevant messages
    """
    if keywords is None:
        keywords = DEPIN_KEYWORDS + ALERT_KEYWORDS
    
    return await search_messages_for_keywords(channel_id, keywords, limit=100)
