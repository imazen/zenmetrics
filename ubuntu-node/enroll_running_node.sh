#!/usr/bin/env bash
# enroll_running_node.sh [--start] <host> [ssh-user]
#
# Enroll an ALREADY-RUNNING Ubuntu box as a zenfleet CPU worker — no PXE, no disk wipe. This is the
# in-place counterpart to build_node_drive.sh (bakes a drive) and the PXE flow (render_install.py):
# for a box that is already up and reachable over SSH, it installs Docker (if missing) + the
# zen-worker systemd unit + /etc/zen-node/worker.env + pre-pulls the :exec image.
#
# Default: leaves the worker DISABLED + STOPPED — so it will NOT restart-loop against an empty/drained
# pool. Run WITHOUT --start to "enroll ready", then activate later when a sweep is queued:
#     bash enroll_running_node.sh --start <host>
# --start also mints a fresh 7-day scoped R2 cred (mint_cred.sh) and `systemctl enable --now zen-worker`.
# (Weekly cred top-up after that is onboard_node.sh, or re-run this with --start.)
#
# Run from the DEV box (holds the CF token). LAN only. <host> = mDNS name or LAN IP; ssh-user defaults
# to 'lilith' (the always-Ubuntu boxes) — pass 'zen' for PXE-installed nodes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START=0; [ "${1:-}" = "--start" ] && { START=1; shift; }
NODE="${1:?usage: enroll_running_node.sh [--start] <host> [ssh-user]}"
USER_AT="${2:-lilith}"
IMG="${ZEN_WORKER_IMAGE:-ghcr.io/imazen/zenfleet-worker:exec-zensim944-57b7b9ad}"   # canonical CPU worker (public on ghcr); 2026-08-02 default = the 944-wave tag (zensim-foldapp2 binary, SOTA-944 P1 bigcodec)

# R2 endpoint from the CF creds (same resolution as mint_cred.sh); the account id is not secret.
CREDS="${R2_CREDENTIALS_FILE:-$HOME/.config/cloudflare/r2-credentials}"
[ -r "$CREDS" ] || { echo "enroll: no readable CF creds at $CREDS (set R2_CREDENTIALS_FILE)"; exit 1; }
set -a; . "$CREDS"; set +a
R2EP="https://${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing from creds}.r2.cloudflarestorage.com"

SSHN="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes ${USER_AT}@${NODE}"
HOSTID="$($SSHN hostname | tr -d '\r')"
echo "enrolling $HOSTID ($USER_AT@$NODE) as CPU worker; image=$IMG; start=$START"

# worker.env — config only; the AWS_* cred is appended below iff --start (7-day cap; don't pre-mint a
# cred that will expire while the worker sits stopped). POOL mode is self-describing, so this same
# config works for whatever run lands in the pool.
ENVBODY="$(cat <<EOF
# zen fleet worker config (enroll_running_node.sh). AWS_* cred appended by --start / onboard_node.sh.
AWS_REGION=auto
ZEN_R2_ENDPOINT=$R2EP
ZEN_BUCKET=zentrain
ZEN_POOL_RUNLIST=${ZEN_POOL_RUNLIST:-s3://zentrain/jobs/_pool944/runlist.tsv}
ZEN_CORPUS_PREFIX=refs/clean-picker-corpus-2026-06-26
ZEN_MAX_MIN=700
ZEN_CORE_OVERSUBSCRIBE=3
ZEN_PERSISTENT_EXEC=1
RAYON_NUM_THREADS=1
OMP_NUM_THREADS=1
ZEN_CHUNK_WALL_SEC=20
ZEN_PASS_TIMEOUT=5400
ZEN_PROVIDER=basement
ZEN_WORKER=$HOSTID
EOF
)"

UNIT="$(cat <<EOF
[Unit]
Description=zen fleet worker (CPU)
Wants=network-online.target docker.service
After=network-online.target docker.service
[Service]
ExecStartPre=-/usr/bin/docker rm -f zen720
ExecStart=/usr/bin/docker run --rm --name zen720 --env-file /etc/zen-node/worker.env --entrypoint /usr/local/bin/fleet-entrypoint.sh $IMG
ExecStop=/usr/bin/docker rm -f zen720
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
)"

ENVB64="$(printf '%s\n' "$ENVBODY" | base64 -w0)"
UNITB64="$(printf '%s\n' "$UNIT" | base64 -w0)"
CREDB64=""
[ "$START" = "1" ] && CREDB64="$(bash "$HERE/mint_cred.sh" | base64 -w0)"

$SSHN "set -e
  sudo -n install -d -m755 /etc/zen-node
  if ! command -v docker >/dev/null; then
    sudo -n apt-get -o DPkg::Lock::Timeout=600 -y update
    sudo -n apt-get -o DPkg::Lock::Timeout=600 -y install docker.io
  fi
  sudo -n systemctl enable --now docker || true
  echo '$ENVB64' | base64 -d | sudo -n tee /etc/zen-node/worker.env >/dev/null
  if [ -n '$CREDB64' ]; then echo '$CREDB64' | base64 -d | sudo -n tee -a /etc/zen-node/worker.env >/dev/null; fi
  sudo -n chmod 600 /etc/zen-node/worker.env
  echo '$UNITB64' | base64 -d | sudo -n tee /etc/systemd/system/zen-worker.service >/dev/null
  sudo -n systemctl daemon-reload
  echo 'pulling $IMG ...'; sudo -n docker pull $IMG >/dev/null && echo 'image pulled'
  if [ '$START' = '1' ]; then
    sudo -n systemctl enable --now zen-worker
    echo -n 'zen-worker is now: '; sudo -n systemctl is-active zen-worker
  else
    sudo -n systemctl disable zen-worker 2>/dev/null || true
    sudo -n systemctl stop zen-worker 2>/dev/null || true
    echo 'zen-worker ENROLLED but STOPPED/DISABLED — activate with:  bash enroll_running_node.sh --start $NODE'
  fi
"
echo "done: $HOSTID enrolled (start=$START)."