#!/usr/bin/env bash
# fleet_util_snapshot.sh — print a one-line-per-box utilization snapshot.
#
# Usage:
#   fleet_util_snapshot.sh                              # auto-detect via label
#   fleet_util_snapshot.sh --label cvvdp-v15rc-2026-05-18
#   fleet_util_snapshot.sh 37053754 37053755 ...        # explicit IDs
#
# Fields:
#   gpu_util  — % from vast.ai's polling. A *brief* 0-5% between encodes is normal,
#               but a box SUSTAINED at low gpu+cpu util past warmup is idle = wasted $.
#               Such boxes are flagged ⚠IDLE and summed into a "$X/hr wasted" banner.
#   cpu_util  — % across the box's allocated vCPUs.
#   mem_use   — host RAM in GB.
#   gpu_ram   — GPU VRAM total (MB) — sizing for OOM checks.
#   thru_chk  — sidecars/hr trailing from the fleet's R2 prefix
#               (filled in below if --run-id passed).
set -euo pipefail

# Idle thresholds — mirror zenfleet-core::idle (crates/zenfleet-core/src/idle.rs):
# a box past warmup with GPU AND CPU util at/below these % is idle/underutilized.
GPU_IDLE_PCT="${GPU_IDLE_PCT:-10}"
CPU_IDLE_PCT="${CPU_IDLE_PCT:-10}"
GRACE_MIN="${GRACE_MIN:-2}" # ignore boxes younger than this (still warming up)

LABEL_PREFIX=""
RUN_ID=""
IDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL_PREFIX="$2"; shift 2;;
        --run-id) RUN_ID="$2"; shift 2;;
        -h|--help) sed -n '2,18p' "$0" >&2; exit 0;;
        *) IDS+=("$1"); shift;;
    esac
done

if (( ${#IDS[@]} == 0 )); then
    # Auto-detect by label (default to the cvvdp run-id we're presently sweeping)
    LABEL_PREFIX="${LABEL_PREFIX:-cvvdp-v15rc-2026-05-18}"
    mapfile -t IDS < <(vastai show instances --raw 2>/dev/null \
        | python3 -c "
import json, sys, re
d = json.loads(sys.stdin.read())
for inst in d:
    if (inst.get('label') or '').startswith('${LABEL_PREFIX}'):
        print(inst['id'])
")
fi

if (( ${#IDS[@]} == 0 )); then
    echo "no instances matched label prefix '${LABEL_PREFIX}'" >&2
    exit 1
fi

printf '%-10s %-14s %-9s %-9s %-9s %-9s %-7s %s\n' \
    "id" "gpu" "gpu_util" "cpu_util" "mem_gb" "gpu_mb" "up_min" "status"

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT

for id in "${IDS[@]}"; do
    vastai show instance "$id" --raw 2>&1 | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
gu = d.get('gpu_util', 0) or 0
cu = d.get('cpu_util', 0) or 0
up = int((d.get('duration', 0) or 0) / 60)
dph = d.get('dph_total', 0) or 0
idle = up >= $GRACE_MIN and gu <= $GPU_IDLE_PCT and cu <= $CPU_IDLE_PCT
flag = '⚠ IDLE' if idle else 'ok'
print(f'{d[\"id\"]:<10} {d.get(\"gpu_name\",\"?\")[:13]:<14} {gu:<9.1f} {cu:<9.1f} {d.get(\"mem_usage\",0):<9.2f} {int(d.get(\"gpu_totalram\",0)):<9d} {up:<7d} {flag}')
open('$TMP', 'a').write(f'{int(idle)} {dph} {d[\"id\"]}\n')
" 2>/dev/null || echo "  $id <error>"
done

# Idle summary — the headline 'you are wasting \$X/hr' banner the old snapshot never showed.
python3 -c "
rows = [l.split() for l in open('$TMP') if l.strip()]
idle = [(float(d), i) for f, d, i in rows if f == '1']
print()
if idle:
    waste = sum(d for d, _ in idle)
    ids = ' '.join(i for _, i in idle)
    print(f'⚠  {len(idle)}/{len(rows)} boxes IDLE — burning \${waste:.2f}/hr. Destroy: {ids}')
    print('   echo ' + ids + ' | xargs -n1 -I{} sh -c \"echo y | vastai destroy instance {}\"')
else:
    print(f'✓ no idle boxes ({len(rows)} active)')
"

# Throughput trailing if --run-id provided
if [[ -n "$RUN_ID" ]]; then
    : "${R2_ACCOUNT_ID:?source ~/.config/cloudflare/r2-credentials first}"
    echo
    N=$(s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" --profile r2 \
        ls "s3://zentrain/${RUN_ID}/omni/" 2>/dev/null | grep -c parquet || echo 0)
    echo "sidecars produced: ${N}"
fi
