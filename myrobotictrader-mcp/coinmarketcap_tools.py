"""
CoinMarketCap API Integration for MyRoboticTrader MCP Server

This module provides helper functions for fetching cryptocurrency data
from CoinMarketCap API to use in content generation.

Required environment variable:
    COINMARKETCAP_API_KEY - Your CMC API key
"""

import os
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

CMC_BASE_URL = "https://pro-api.coinmarketcap.com/v1"

async def get_cmc_headers() -> Dict[str, str]:
    """Get headers for CoinMarketCap API requests."""
    api_key = os.getenv("COINMARKETCAP_API_KEY")
    if not api_key:
        raise ValueError("COINMARKETCAP_API_KEY not found in environment variables")
    
    return {
        "X-CMC_PRO_API_KEY": api_key,
        "Accept": "application/json"
    }

async def fetch_crypto_price(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch current price and data for a specific cryptocurrency.
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'SOL')
    
    Returns:
        Dictionary with price data or None if error
    """
    try:
        headers = await get_cmc_headers()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{CMC_BASE_URL}/cryptocurrency/quotes/latest",
                headers=headers,
                params={"symbol": symbol.upper(), "convert": "USD"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and symbol.upper() in data["data"]:
                crypto = data["data"][symbol.upper()]
                quote = crypto["quote"]["USD"]
                
                return {
                    "name": crypto["name"],
                    "symbol": crypto["symbol"],
                    "price": round(quote["price"], 2),
                    "change_24h": round(quote["percent_change_24h"], 2),
                    "change_7d": round(quote["percent_change_7d"], 2),
                    "change_30d": round(quote["percent_change_30d"], 2),
                    "market_cap": round(quote["market_cap"], 0),
                    "volume_24h": round(quote["volume_24h"], 0),
                    "last_updated": quote["last_updated"]
                }
            return None
            
    except Exception as e:
        print(f"Error fetching {symbol} price: {e}")
        return None

async def fetch_multiple_cryptos(symbols: List[str]) -> Dict[str, Any]:
    """
    Fetch price data for multiple cryptocurrencies.
    
    Args:
        symbols: List of crypto symbols (e.g., ['BTC', 'ETH', 'SOL'])
    
    Returns:
        Dictionary mapping symbols to price data
    """
    try:
        headers = await get_cmc_headers()
        symbols_str = ",".join([s.upper() for s in symbols])
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{CMC_BASE_URL}/cryptocurrency/quotes/latest",
                headers=headers,
                params={"symbol": symbols_str, "convert": "USD"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            results = {}
            if "data" in data:
                for symbol in symbols:
                    symbol_upper = symbol.upper()
                    if symbol_upper in data["data"]:
                        crypto = data["data"][symbol_upper]
                        quote = crypto["quote"]["USD"]
                        
                        results[symbol_upper] = {
                            "name": crypto["name"],
                            "symbol": crypto["symbol"],
                            "price": round(quote["price"], 2),
                            "change_24h": round(quote["percent_change_24h"], 2),
                            "change_7d": round(quote["percent_change_7d"], 2),
                            "change_30d": round(quote["percent_change_30d"], 2),
                            "market_cap": round(quote["market_cap"], 0),
                            "volume_24h": round(quote["volume_24h"], 0)
                        }
            
            return results
            
    except Exception as e:
        print(f"Error fetching multiple cryptos: {e}")
        return {}

async def fetch_top_cryptos(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch top cryptocurrencies by market cap.
    
    Args:
        limit: Number of cryptos to return (1-100)
    
    Returns:
        List of top cryptocurrencies with price data
    """
    try:
        headers = await get_cmc_headers()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{CMC_BASE_URL}/cryptocurrency/listings/latest",
                headers=headers,
                params={"limit": min(limit, 100), "convert": "USD"},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "data" in data:
                for crypto in data["data"]:
                    quote = crypto["quote"]["USD"]
                    results.append({
                        "rank": crypto["cmc_rank"],
                        "name": crypto["name"],
                        "symbol": crypto["symbol"],
                        "price": round(quote["price"], 2),
                        "change_24h": round(quote["percent_change_24h"], 2),
                        "change_7d": round(quote["percent_change_7d"], 2),
                        "market_cap": round(quote["market_cap"], 0),
                        "volume_24h": round(quote["volume_24h"], 0)
                    })
            
            return results
            
    except Exception as e:
        print(f"Error fetching top cryptos: {e}")
        return []

async def fetch_gainers_losers(limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch top gainers and losers in the last 24 hours.
    
    Args:
        limit: Number of gainers/losers to return
    
    Returns:
        Dictionary with 'gainers' and 'losers' lists
    """
    try:
        # Fetch top 100 to sort by 24h change
        top_cryptos = await fetch_top_cryptos(100)
        
        if not top_cryptos:
            return {"gainers": [], "losers": []}
        
        # Sort by 24h change
        sorted_by_change = sorted(top_cryptos, key=lambda x: x["change_24h"], reverse=True)
        
        gainers = sorted_by_change[:limit]
        losers = sorted_by_change[-limit:]
        losers.reverse()  # Show worst performers first
        
        return {
            "gainers": gainers,
            "losers": losers
        }
        
    except Exception as e:
        print(f"Error fetching gainers/losers: {e}")
        return {"gainers": [], "losers": []}

async def fetch_fear_greed_index() -> Optional[Dict[str, Any]]:
    """
    Fetch the Crypto Fear & Greed Index from Alternative.me (free API).
    
    Since CoinMarketCap doesn't offer this on their free tier, we use
    Alternative.me's free Fear & Greed Index API instead.
    
    Returns:
        Dictionary with current fear/greed value and classification
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.alternative.me/fng/",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and len(data["data"]) > 0:
                fg_data = data["data"][0]
                value = int(fg_data["value"])
                
                # Classify based on value
                if value <= 25:
                    classification = "Extreme Fear"
                elif value <= 45:
                    classification = "Fear"
                elif value <= 55:
                    classification = "Neutral"
                elif value <= 75:
                    classification = "Greed"
                else:
                    classification = "Extreme Greed"
                
                return {
                    "value": value,
                    "value_classification": classification,
                    "timestamp": fg_data.get("timestamp"),
                    "time_until_update": fg_data.get("time_until_update")
                }
            
            return None
            
    except Exception as e:
        print(f"Error fetching fear/greed index: {e}")
        return None

async def fetch_altcoin_season_index() -> Optional[Dict[str, Any]]:
    """
    Fetch the Altcoin Season Index from blockchaincenter.net (free API).
    
    Since CoinMarketCap doesn't offer this on their free tier, we use
    blockchaincenter.net's free Altcoin Season Index API instead.
    
    The Altcoin Season Index measures whether we're in Bitcoin or Altcoin season.
    - 0-25: Bitcoin Season (BTC dominance)
    - 25-75: Mixed/Neutral
    - 75-100: Altcoin Season (alts outperforming BTC)
    
    Returns:
        Dictionary with current altcoin season value and classification
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.blockchaincenter.net/api/altcoin-season-index",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and data["data"]:
                value = int(data["data"])
                
                # Determine classification
                if value <= 25:
                    classification = "Bitcoin Season"
                elif value >= 75:
                    classification = "Altcoin Season"
                else:
                    classification = "Mixed Market"
                
                return {
                    "value": value,
                    "classification": classification,
                    "timestamp": datetime.now().isoformat(),
                    "month_performance": None
                }
            
            return None
            
    except Exception as e:
        print(f"Error fetching altcoin season index: {e}")
        return None

def format_crypto_data_markdown(crypto_data: Dict[str, Any]) -> str:
    """Format cryptocurrency data as markdown."""
    if not crypto_data:
        return "No data available"
    
    output = f"## {crypto_data['name']} ({crypto_data['symbol']})\n\n"
    output += f"**Current Price:** ${crypto_data['price']:,.2f}\n"
    output += f"**24h Change:** {crypto_data['change_24h']:+.2f}%\n"
    output += f"**7d Change:** {crypto_data['change_7d']:+.2f}%\n"
    output += f"**30d Change:** {crypto_data['change_30d']:+.2f}%\n"
    output += f"**Market Cap:** ${crypto_data['market_cap']:,.0f}\n"
    output += f"**24h Volume:** ${crypto_data['volume_24h']:,.0f}\n"
    
    return output

def format_top_cryptos_markdown(cryptos: List[Dict[str, Any]]) -> str:
    """Format list of top cryptos as markdown table."""
    if not cryptos:
        return "No data available"
    
    output = "## Top Cryptocurrencies by Market Cap\n\n"
    output += "| Rank | Name | Price | 24h Change | Market Cap |\n"
    output += "|------|------|-------|------------|------------|\n"
    
    for crypto in cryptos:
        change_emoji = "ðŸŸ¢" if crypto["change_24h"] >= 0 else "ðŸ”´"
        output += f"| {crypto['rank']} | {crypto['name']} ({crypto['symbol']}) | "
        output += f"${crypto['price']:,.2f} | {change_emoji} {crypto['change_24h']:+.2f}% | "
        output += f"${crypto['market_cap']:,.0f} |\n"
    
    return output

def format_gainers_losers_markdown(data: Dict[str, List[Dict[str, Any]]]) -> str:
    """Format gainers and losers as markdown."""
    output = "## ðŸš€ Top Gainers (24h)\n\n"
    
    for crypto in data["gainers"]:
        output += f"- **{crypto['name']} ({crypto['symbol']})**: "
        output += f"${crypto['price']:,.2f} | ðŸŸ¢ {crypto['change_24h']:+.2f}%\n"
    
    output += "\n## ðŸ“‰ Top Losers (24h)\n\n"
    
    for crypto in data["losers"]:
        output += f"- **{crypto['name']} ({crypto['symbol']})**: "
        output += f"${crypto['price']:,.2f} | ðŸ”´ {crypto['change_24h']:+.2f}%\n"
    
    return output
