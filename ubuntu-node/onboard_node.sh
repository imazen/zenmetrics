#!/usr/bin/env bash
# onboard_node.sh <host> — push a FRESH 7-day R2 cred to a running zen-node and restart its worker.
#
# Run from the DEV box (holds the CF token). The node runs zen-worker as a systemd service reading
# /etc/zen-node/worker.env; this replaces the three AWS_* lines in that file with a freshly-minted
# 7-day scoped cred and restarts the service. The drive is built with an initial cred, so this is
# only needed to top it up — weekly, given the 7-day cap. LAN only (no Tailscale); <host> is the
# node's mDNS name (zen-node-1.local) or its LAN IP.
#
# Hands-off weekly refresh — add to the dev box's crontab:
#   17 4 * * 1  bash /home/lilith/work/zen/zenmetrics/ubuntu-node/onboard_node.sh zen-node-1.local >> ~/tmp/zen-node-refresh.log 2>&1
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE="${1:?usage: onboard_node.sh <host-or-ip>}"

CREDS_B64="$(bash "$HERE/mint_cred.sh" | base64 -w0)"   # AWS_ACCESS_KEY_ID/_SECRET/_SESSION_TOKEN
SSHN="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes zen@$NODE"

$SSHN "set -e
  F=/etc/zen-node/worker.env
  sudo install -d -m755 /etc/zen-node
  sudo sed -i '/^AWS_ACCESS_KEY_ID=/d;/^AWS_SECRET_ACCESS_KEY=/d;/^AWS_SESSION_TOKEN=/d' \$F 2>/dev/null || true
  echo '$CREDS_B64' | base64 -d | sudo tee -a \$F >/dev/null
  sudo chmod 600 \$F
  sudo systemctl restart zen-worker
  echo -n 'zen-worker is now: '; sudo systemctl is-active zen-worker"

echo "pushed fresh 7-day cred to $NODE."
echo "verify: ssh zen@$NODE 'docker top zen720 | grep -c zenmetrics'   (~= core count once scoring)"
