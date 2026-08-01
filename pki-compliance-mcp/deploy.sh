#!/bin/bash
set -euo pipefail

# Deploys BOTH runtimes that serve pki_compliance_mcp.py.
#
# They load the same file in different ways and drift apart silently if only one
# is deployed:
#   - systemd pki-compliance-api  reads the file from disk  -> `git pull` updates it
#   - docker  pki-compliance-mcp  bakes the file into the image at build time
#                                 -> needs a rebuild, a restart is NOT enough
#
# On 2026-07-31 a data commit was deployed with `systemctl restart` alone. The
# container kept serving 2026-07-29 code for two days (73 vs 75 deadlines,
# dataVersion 2.4.10 vs 2.4.12, missing SMC017v2 and Netherlands NIS2) with no
# visible symptom. The verify step at the bottom is what makes that loud.

REPO_DIR=/opt/mcp-servers
APP_DIR="$REPO_DIR/pki-compliance-mcp"
CONTAINER=pki-compliance-mcp
SERVICE=pki-compliance-api
LOG=/var/log/pki-compliance-deploy.log

cd "$APP_DIR"
git pull origin main

# No host-level pip install here, deliberately. requirements.txt is the
# CONTAINER's dependency set and `docker compose build` installs it inside the
# image. The systemd API runs --simple-http, which guards the MCP SDK import
# behind try/except and never needs it; its own deps (httpx, feedparser,
# pydantic, bs4) come from apt. Installing requirements.txt on the host pulled
# mcp[cli] -> PyJWT>=2.13, which collides with apt's python3-jwt 2.7 (no pip
# RECORD, so pip cannot uninstall it). Under `set -e` that aborted the deploy
# before it restarted anything. If a host dep is ever genuinely missing, the
# verify step below catches it — the probe imports the module.

# 1. systemd API (:5000, fixmycert.com frontend) — picks up the file from disk
sudo systemctl restart "$SERVICE"

# 2. Docker MCP (:8085) — must be rebuilt, not just restarted.
# `COPY . .` in Dockerfile.mcp invalidates on content change, so a plain build
# is enough; --no-cache is the fallback below if it somehow isn't.
# NOTE: --force-recreate drops active SSE sessions on :8085. Clients reconnect.
cd "$REPO_DIR"
docker compose build pki-compliance
docker compose up -d --force-recreate pki-compliance

# 3. Verify the two runtimes agree. This is the drift guard: it compares the code
# each side actually loaded, not what we hope it loaded.
PROBE='import pki_compliance_mcp as m; print("%s|%d" % (m.COMPLIANCE_METADATA.get("dataVersion"), len(m.get_all_deadlines_unified())))'

# Give the container a moment to come up before exec'ing into it.
for _ in $(seq 1 10); do
    docker exec "$CONTAINER" python -c "pass" >/dev/null 2>&1 && break
    sleep 1
done

DISK=$(cd "$APP_DIR" && python3 -c "$PROBE" | tail -1)
IMAGE=$(docker exec "$CONTAINER" python -c "$PROBE" | tail -1)

if [ "$DISK" != "$IMAGE" ]; then
    # Plain build did not take. Escalate to a full rebuild and re-check once.
    echo "WARNING: disk ($DISK) != container ($IMAGE) after build. Retrying --no-cache." >&2
    docker compose build --no-cache pki-compliance
    docker compose up -d --force-recreate pki-compliance
    for _ in $(seq 1 10); do
        docker exec "$CONTAINER" python -c "pass" >/dev/null 2>&1 && break
        sleep 1
    done
    IMAGE=$(docker exec "$CONTAINER" python -c "$PROBE" | tail -1)
fi

COMMIT=$(git -C "$APP_DIR" rev-parse --short HEAD)

if [ "$DISK" != "$IMAGE" ]; then
    echo "$(date): DEPLOY FAILED $COMMIT — DRIFT: systemd=$DISK container=$IMAGE" >> "$LOG"
    echo "DEPLOY FAILED: systemd and container disagree after rebuild." >&2
    echo "  systemd (disk):   dataVersion|unified = $DISK" >&2
    echo "  docker (image):   dataVersion|unified = $IMAGE" >&2
    echo "The MCP on :8085 is serving stale code. Do not trust pki_get_deadlines." >&2
    exit 1
fi

echo "$(date): Deployed $COMMIT — systemd + container both at $DISK (dataVersion|unified)" >> "$LOG"
echo "Deployed $COMMIT. Both runtimes at dataVersion|unified = $DISK"
