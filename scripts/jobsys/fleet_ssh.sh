#!/usr/bin/env bash
# Mass-control the Hetzner fleet: run a command on EVERY box whose name matches a
# prefix, in parallel, and collect per-box output. The one way to inspect/fix/repurpose
# a running fleet at scale (billing is hourly-rounded — a box that's up is paid for, so
# make it productive rather than tearing it down; see ~/.claude/CLAUDE.md).
#   usage: fleet_ssh.sh <name-prefix> '<remote command>'
#     env: SSH_KEY_FILE (default ~/.ssh/zen-arm-dev), FLEET_SSH_TIMEOUT (default 20),
#          FLEET_SSH_PAR (max parallel, default 40)
# Examples:
#   fleet_ssh.sh hzsf-bf 'ps aux|grep -cE "[z]enmetrics jobexec"'            # concurrency per box
#   fleet_ssh.sh hzsf-bf 'docker logs $(docker ps -q|head -1) 2>&1|tail -1'  # last log line
#   fleet_ssh.sh hzsf-bf 'docker restart $(docker ps -q|head -1)'            # mass restart
set -uo pipefail
PREFIX="${1:?usage: fleet_ssh.sh <name-prefix> '<cmd>'}"; shift
CMD="$*"; [ -n "$CMD" ] || { echo "no command"; exit 2; }
KEY="${SSH_KEY_FILE:-$HOME/.ssh/zen-arm-dev}"
PAR="${FLEET_SSH_PAR:-40}"
TMO="${FLEET_SSH_TIMEOUT:-20}"
export HCLOUD_TOKEN="${HCLOUD_TOKEN:-$(grep -E '^api_token=' ~/.config/hetzner/credentials 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' \r')}"
mapfile -t IPS < <(hcloud server list -o columns=name,ipv4 2>/dev/null | grep "$PREFIX" | awk '{print $2}')
echo "# $PREFIX: ${#IPS[@]} boxes"
run_one() {
  local ip="$1"
  ssh-keygen -R "$ip" >/dev/null 2>&1
  local out
  out=$(timeout "$TMO" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes \
        -o LogLevel=ERROR -i "$KEY" root@"$ip" "$CMD" 2>&1 | tr '\n' '~')
  printf '%s | %s\n' "$ip" "$out"
}
i=0
for ip in "${IPS[@]}"; do
  run_one "$ip" &
  i=$((i + 1))
  if [ "$((i % PAR))" -eq 0 ]; then wait; fi
done
wait
