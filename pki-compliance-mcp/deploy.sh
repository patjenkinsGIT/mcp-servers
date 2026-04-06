#!/bin/bash
set -e

cd /opt/mcp-servers/pki-compliance-mcp
git pull origin main
pip install -r requirements.txt --break-system-packages -q
sudo systemctl restart pki-compliance-api
echo "$(date): Deployed and restarted" >> /var/log/pki-compliance-deploy.log