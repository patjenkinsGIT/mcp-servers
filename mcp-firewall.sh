#!/bin/bash
# Restrict MCP ports (8081-8086) and the compliance API (5000) to Tailscale and
# loopback: drop anything arriving on the public interface. Tailscale traffic
# arrives on tailscale0 and is unaffected.
# Idempotent — safe to re-run. Installed 2026-07-23. Port 5000 added 2026-08-21.
set -e
PORTS="8081,8082,8083,8084,8085,8086"
API_PORT="5000"
PUB_IF="eth0"

# --- Host-process rules first: these must not depend on Docker ---------------
# The compliance API on :5000 is a HOST process (pki-compliance-api.service),
# not a container, so it is reached via INPUT only — no DOCKER-USER rule applies
# and there is no DNAT path. It listens on 0.0.0.0:5000 with no IPv6 socket, so
# no ip6tables rule is needed. nginx reaches it over loopback (proxy_pass to
# 127.0.0.1:5000 for compliance-api.fixmycert.com) and the tailnet path still
# works, because this rule is scoped to the public interface.
# THIS BLOCK IS DELIBERATELY FIRST: the script is `set -e`, so if Docker is down
# and the DOCKER-USER commands below fail, :5000 is already protected before the
# script aborts. Do not move it back below the Docker-dependent blocks.
iptables -C INPUT -i "$PUB_IF" -p tcp --dport "$API_PORT" -j DROP 2>/dev/null || \
  iptables -I INPUT -i "$PUB_IF" -p tcp --dport "$API_PORT" -j DROP

# --- Docker-dependent rules --------------------------------------------------
# Docker-routed traffic (DNAT path) traverses DOCKER-USER in FORWARD
iptables -C DOCKER-USER -i "$PUB_IF" -p tcp -m multiport --dports "$PORTS" -j DROP 2>/dev/null || \
  iptables -I DOCKER-USER -i "$PUB_IF" -p tcp -m multiport --dports "$PORTS" -j DROP
# docker-proxy userspace listeners are hit via INPUT (v4 and v6)
iptables -C INPUT -i "$PUB_IF" -p tcp -m multiport --dports "$PORTS" -j DROP 2>/dev/null || \
  iptables -I INPUT -i "$PUB_IF" -p tcp -m multiport --dports "$PORTS" -j DROP
ip6tables -C INPUT -i "$PUB_IF" -p tcp -m multiport --dports "$PORTS" -j DROP 2>/dev/null || \
  ip6tables -I INPUT -i "$PUB_IF" -p tcp -m multiport --dports "$PORTS" -j DROP 2>/dev/null || true
