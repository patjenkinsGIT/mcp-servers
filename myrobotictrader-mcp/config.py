"""
Configuration management for MyRoboticTrader MCP Server

Loads API keys and configuration from environment variables.
"""

import os
from pathlib import Path
from typing import Optional

# Try to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass

class Config:
    """Configuration settings loaded from environment variables."""
    
    # News API
    NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY")
    NEWS_API_BASE_URL: str = "https://newsapi.org/v2"
    
    # Anthropic API for content generation
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Google Sheets
    GOOGLE_SHEETS_CREDS: Optional[str] = os.getenv("GOOGLE_SHEETS_CREDS")
    TRADING_SHEET_ID: Optional[str] = os.getenv("TRADING_SHEET_ID")
    
    # Server settings
    SERVER_NAME: str = "myrobotictrader_mcp"
    TOPICS_FILE: str = "topics.json"
    
    # API timeouts
    HTTP_TIMEOUT: float = 30.0
    
    # Content generation defaults
    DEFAULT_CTA_URL: str = "https://myrobotictrader.com"
    
    @classmethod
    def validate(cls) -> dict[str, bool]:
        """Check which API credentials are configured."""
        return {
            "news_api": cls.NEWS_API_KEY is not None,
            "anthropic_api": cls.ANTHROPIC_API_KEY is not None,
            "google_sheets": cls.GOOGLE_SHEETS_CREDS is not None and cls.TRADING_SHEET_ID is not None,
        }
    
    @classmethod
    def get_missing_credentials(cls) -> list[str]:
        """Return list of missing API credentials."""
        validation = cls.validate()
        missing = []
        
        if not validation["news_api"]:
            missing.append("NEWS_API_KEY (for news fetching)")
        if not validation["anthropic_api"]:
            missing.append("ANTHROPIC_API_KEY (for content generation)")
        if not validation["google_sheets"]:
            missing.append("GOOGLE_SHEETS_CREDS and TRADING_SHEET_ID (for trading data)")
        
        return missing
    
    @classmethod
    def print_status(cls) -> None:
        """Print configuration status to console."""
        print(f"\n{'='*60}")
        print(f"MyRoboticTrader MCP Server - Configuration Status")
        print(f"{'='*60}")
        
        validation = cls.validate()
        
        status_icon = lambda x: "✓" if x else "✗"
        
        print(f"\n{status_icon(validation['news_api'])} News API: {'Configured' if validation['news_api'] else 'Not configured'}")
        print(f"{status_icon(validation['anthropic_api'])} Anthropic API: {'Configured' if validation['anthropic_api'] else 'Not configured'}")
        print(f"{status_icon(validation['google_sheets'])} Google Sheets: {'Configured' if validation['google_sheets'] else 'Not configured'}")
        
        missing = cls.get_missing_credentials()
        if missing:
            print(f"\n⚠️  Missing credentials:")
            for cred in missing:
                print(f"   - {cred}")
            print(f"\nAdd these to your .env file or environment variables.")
            print(f"See .env.example for template.")
        else:
            print(f"\n✓ All APIs configured!")
        
        print(f"\n{'='*60}\n")


# Helper functions for API access

def get_news_api_key() -> Optional[str]:
    """Get NewsAPI key."""
    if not Config.NEWS_API_KEY:
        print("⚠️  NEWS_API_KEY not set. News fetching will use mock data.")
        print("   Get a free key at: https://newsapi.org/")
    return Config.NEWS_API_KEY

def get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key."""
    if not Config.ANTHROPIC_API_KEY:
        print("⚠️  ANTHROPIC_API_KEY not set. Content generation will use templates.")
        print("   Get an API key at: https://console.anthropic.com/")
    return Config.ANTHROPIC_API_KEY

def get_google_sheets_config() -> tuple[Optional[str], Optional[str]]:
    """Get Google Sheets credentials and sheet ID."""
    if not Config.GOOGLE_SHEETS_CREDS or not Config.TRADING_SHEET_ID:
        print("⚠️  Google Sheets not configured. Trading data will use mock data.")
        print("   See README.md for setup instructions.")
    return Config.GOOGLE_SHEETS_CREDS, Config.TRADING_SHEET_ID


# Example usage
if __name__ == "__main__":
    Config.print_status()
