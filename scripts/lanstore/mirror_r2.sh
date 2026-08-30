#!/usr/bin/env bash
# mirror_r2.sh — copy an R2 prefix onto the LAN store at the IDENTICAL bucket/key,
# and/or verify that it is already there. Written for the 2026-08-30 working-set
# mirror; methodology + measured numbers in benchmarks/r2_lan_mirror_2026-08-30.md.
#
#   scripts/lanstore/mirror_r2.sh copy   <bucket> <prefix> [more prefixes...]
#   scripts/lanstore/mirror_r2.sh verify <bucket> <prefix> [more prefixes...]
#
# Endpoint + credentials come from scripts/lib/s3env.sh (the resolver) for the LAN
# side and ~/.config/cloudflare/r2-credentials for the R2 side. Nothing is echoed.
#
# THREE THINGS THIS ENCODES, all measured 2026-08-30 — read before "simplifying":
#
#  1. `aws s3 ls` for LAN-side counts, never `s5cmd ls`. s5cmd's lister UNDERCOUNTS
#     non-deterministically against SeaweedFS (40450/40466/40467 vs the true 40473
#     on a settled prefix; the omitted keys were verifiably present). A mirror
#     diffed with s5cmd re-transfers phantom-missing objects forever.
#  2. A tower LOAD GATE. Large sequential objects are cheap for this store (load
#     ~10, 81-90% idle). Sustained small-object PUTs are not: ~333 obj/s drove the
#     tower to load 30.7 / 48.9% iowait with the S3 endpoint unresponsive. That box
#     runs the household's media, which outranks any mirror.
#  3. `rclone copyto <exact-key>` for a single object inside a job-run prefix.
#     `rclone copy <prefix> --include <file>` enumerates the run's whole blobs/
#     store first and takes minutes per file instead of seconds.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/s3env.sh
. "$HERE/../lib/s3env.sh"          # -> EP, AWS_* for the selected store (LAN by default)
LAN_EP="$EP"; LAN_AK="$AWS_ACCESS_KEY_ID"; LAN_SK="$AWS_SECRET_ACCESS_KEY"

# NOTE: s3env.sh's LAN arm re-exports the LAN key under the legacy R2_* names for
# back-compat, so the R2 source credentials MUST be read separately, into their own
# variables, AFTER it has run — reading $R2_ACCESS_KEY_ID here would silently hand
# the LAN key to the R2 endpoint and every listing would come back empty.
_r2creds="$HOME/.config/cloudflare/r2-credentials"
[ -f "$_r2creds" ] || { echo "need $_r2creds for the R2 source side" >&2; exit 2; }
SRC_ACCOUNT=$(sed -n 's/^[[:space:]]*\(export[[:space:]]*\)\?R2_ACCOUNT_ID=//p'        "$_r2creds" | tr -d '"'"'"'" ' | head -1)
SRC_AK=$(sed -n 's/^[[:space:]]*\(export[[:space:]]*\)\?R2_ACCESS_KEY_ID=//p'          "$_r2creds" | tr -d '"'"'"'" ' | head -1)
SRC_SK=$(sed -n 's/^[[:space:]]*\(export[[:space:]]*\)\?R2_SECRET_ACCESS_KEY=//p'      "$_r2creds" | tr -d '"'"'"'" ' | head -1)
: "${SRC_ACCOUNT:?no R2_ACCOUNT_ID in $_r2creds}" "${SRC_AK:?}" "${SRC_SK:?}"
R2_EP="https://${SRC_ACCOUNT}.r2.cloudflarestorage.com"

TOWER_HOST="${ZEN_TOWER_SSH:-root@tower}"
LOAD_GATE="${ZEN_LOAD_GATE:-15}"
TRANSFERS="${ZEN_TRANSFERS:-2}"
LOG="${ZEN_MIRROR_LOG:-$HOME/tmp/lanstore_mirror.log}"
mkdir -p "$(dirname "$LOG")"

_aws_lan() { AWS_ACCESS_KEY_ID="$LAN_AK" AWS_SECRET_ACCESS_KEY="$LAN_SK" AWS_REGION=us-east-1 \
             aws --endpoint-url "$LAN_EP" "$@"; }
_aws_r2()  { AWS_ACCESS_KEY_ID="$SRC_AK" AWS_SECRET_ACCESS_KEY="$SRC_SK" AWS_REGION=auto \
             aws --endpoint-url "$R2_EP" "$@"; }
_sum() {  # _sum <aws-fn> <bucket> <prefix>  ->  "<objects> <bytes>"
  "$1" s3 ls "s3://$2/$3" --recursive --summarize 2>/dev/null \
    | tail -3 | awk '/Total Objects/{o=$3}/Total Size/{s=$3}END{print (o==""?0:o)" "(s==""?0:s)}'
}
_gate() {                       # block while the tower is busy; media has priority
  local n=0 l
  while :; do
    l=$(ssh -o ConnectTimeout=8 "$TOWER_HOST" 'cut -d" " -f1 /proc/loadavg' 2>/dev/null)
    if [ -z "$l" ]; then sleep 30; n=$((n+1)); [ $n -gt 20 ] && return 1; continue; fi
    awk -v a="$l" -v b="$LOAD_GATE" 'BEGIN{exit !(a<b)}' && return 0
    echo "[$(date -u +%FT%TZ)] gate: tower load $l >= $LOAD_GATE, waiting" | tee -a "$LOG"
    sleep 120; n=$((n+1)); [ $n -gt 240 ] && return 1
  done
}

cmd="${1:-}"; bucket="${2:-}"; shift 2 2>/dev/null || { sed -n '2,8p' "$0"; exit 2; }
[ -n "$cmd" ] && [ -n "$bucket" ] && [ $# -gt 0 ] || { sed -n '2,8p' "$0"; exit 2; }
rc_all=0

for prefix in "$@"; do
  case "$cmd" in
    copy)
      _gate || { echo "[$(date -u +%FT%TZ)] ABORT $bucket/$prefix — tower stayed busy" | tee -a "$LOG"; rc_all=1; break; }
      echo "[$(date -u +%FT%TZ)] copy s3://$bucket/$prefix" | tee -a "$LOG"
      RCLONE_CONFIG_R2S_TYPE=s3 RCLONE_CONFIG_R2S_PROVIDER=Cloudflare \
      RCLONE_CONFIG_R2S_ENDPOINT="$R2_EP" RCLONE_CONFIG_R2S_ACCESS_KEY_ID="$SRC_AK" \
      RCLONE_CONFIG_R2S_SECRET_ACCESS_KEY="$SRC_SK" RCLONE_CONFIG_R2S_REGION=auto \
      RCLONE_CONFIG_LANS_TYPE=s3 RCLONE_CONFIG_LANS_PROVIDER=Other \
      RCLONE_CONFIG_LANS_ENDPOINT="$LAN_EP" RCLONE_CONFIG_LANS_ACCESS_KEY_ID="$LAN_AK" \
      RCLONE_CONFIG_LANS_SECRET_ACCESS_KEY="$LAN_SK" RCLONE_CONFIG_LANS_REGION=us-east-1 \
      RCLONE_CONFIG_LANS_FORCE_PATH_STYLE=true \
      nice -n19 ionice -c3 rclone copy "r2s:$bucket/$prefix" "lans:$bucket/$prefix" \
        --transfers "$TRANSFERS" --checkers $((TRANSFERS*2)) \
        --s3-upload-concurrency 2 --s3-chunk-size 64M \
        --retries 5 --low-level-retries 20 --timeout 5m \
        --stats 180s --stats-one-line --stats-log-level NOTICE \
        --log-file "$LOG" --log-level NOTICE || rc_all=1
      ;&
    verify)
      read -r rn rb <<<"$(_sum _aws_r2  "$bucket" "$prefix")"
      read -r ln lb <<<"$(_sum _aws_lan "$bucket" "$prefix")"
      if [ "$rn" = "$ln" ] && [ "$rb" = "$lb" ]; then verdict=MATCH; else verdict=MISMATCH; rc_all=1; fi
      printf '%-62s r2=%s/%s lan=%s/%s %s\n' "$bucket/$prefix" "$rn" "$rb" "$ln" "$lb" "$verdict" | tee -a "$LOG"
      ;;
    *) sed -n '2,8p' "$0"; exit 2;;
  esac
done
exit $rc_all
