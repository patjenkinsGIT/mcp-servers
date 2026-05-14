#!/bin/bash
# Daily PKI document hash check
#
# Invokes pki_check_all_documents inside the pki-compliance-mcp container so
# /data/state.json (last_check + document_hashes) advances without anyone
# calling the MCP tool manually. Host cron entry runs this once a day.
#
# Exit codes: 0 ok, 1 container not running, 2 check raised.

set -euo pipefail

CONTAINER=pki-compliance-mcp
LOG_DIR=/root/.pki-compliance-mcp
LOG_FILE="$LOG_DIR/doc_check.log"

mkdir -p "$LOG_DIR"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

{
  echo "===== $(ts) daily_doc_check start ====="

  if ! docker ps --filter "name=^${CONTAINER}$" --filter "status=running" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container $CONTAINER is not running"
    exit 1
  fi

  docker exec "$CONTAINER" python3 -c '
import asyncio, json, sys
from pki_compliance_mcp import check_all_documents, CheckAllDocumentsInput, ResponseFormat

async def main():
    result = await check_all_documents(
        CheckAllDocumentsInput(response_format=ResponseFormat.JSON)
    )
    data = json.loads(result)
    print(f"checked_at: {data.get('checked_at')}")
    print(f"changes_detected: {data.get('changes_detected')}")
    for d in data.get('documents', []):
        print(f"  {d['document_id']:28s} {d['status']:10s} hash={d.get('hash', '-')}")

asyncio.run(main())
'

  echo "===== $(ts) daily_doc_check done ====="
} >> "$LOG_FILE" 2>&1
