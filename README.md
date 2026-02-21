# MCP Servers

Dockerized MCP servers running on DigitalOcean. Accessible from any machine via SSE transport.

## Services

| Service | Port | Description |
|---------|------|-------------|
| fixmycert-content-mcp | 8081 | FixMyCert content tracker |
| fixmycert-yt-mcp | 8082 | FixMyCert YouTube manager |
| myrobotictrader-mcp | 8083 | MyRoboticTrader tools |

## Quick Start
```bash
git clone git@github.com:YOUR_USERNAME/mcp-servers.git
cd mcp-servers
docker compose up -d --build
```
