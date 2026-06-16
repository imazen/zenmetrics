#!/usr/bin/env bash
#
# fleet_status.sh — one-shot status dashboard for a metric backfill
# fleet. Combines `zenfleet-vastai status`, R2 sidecar count, sidecar
# validity sampling, and an ETA in a single command.
#
# Usage:
#   fleet_status.sh <run-id>
#
# Example:
#   fleet_status.sh ssim2-backfill-2026-05-18
#
# Required tools on PATH: zenfleet-vastai, s5cmd, python3 with pyarrow.
# Required env vars (R2 credentials): R2_ACCOUNT_ID  R2_ACCESS_KEY_ID
#                                     R2_SECRET_ACCESS_KEY
#
# Use this AT FLEET LAUNCH and any time you suspect the destroyer
# crashed. Run it before destroying so you have a snapshot of state.

set -uo pipefail
# shellcheck disable=SC1091
source ~/.config/cloudflare/r2-credentials 2>/dev/null || true

RUN_ID="${1:-${RUN_ID:-}}"
if [[ -z "$RUN_ID" ]]; then
    echo "usage: $0 <run-id>" >&2
    exit 1
fi

# Per-run conventions: sidecars under s3://zentrain/<run-id>/. Override
# with R2_SIDECAR_PREFIX if your generator places them elsewhere.
R2_SIDECAR_PREFIX="${R2_SIDECAR_PREFIX:-s3://zentrain/${RUN_ID}/}"
R2_FAILURES_PREFIX="${R2_FAILURES_PREFIX:-s3://zentrain/${RUN_ID}/failures/}"
R2_CHUNKS="${R2_CHUNKS:-s3://coefficient/jobs/${RUN_ID}/chunks.jsonl}"

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2() { s5cmd --endpoint-url "$R2_ENDPOINT" --profile r2 "$@"; }

echo "=== fleet_status: $RUN_ID ==="
echo

echo "--- fleet (via zenfleet-vastai) ---"
if command -v zenfleet-vastai >/dev/null; then
    zenfleet-vastai status --label-prefix "$RUN_ID" 2>&1 || true
else
    echo "WARN: zenfleet-vastai not on PATH; falling back to vastai show instances-v1"
    vastai show instances-v1 --raw -a 2>/dev/null | python3 -c "
import json, sys
d = json.loads(sys.stdin.read() or '{\"instances\":[]}')
insts = d.get('instances', d) if isinstance(d, dict) else d
matched = [i for i in insts if i and isinstance(i.get('label'), str) and '$RUN_ID' in i['label']]
dph = sum(float(i.get('dph_total') or 0) for i in matched)
print(f'instances: {len(matched)}')
print(f'burn rate: \${dph:.3f}/hr')
" || echo "  (vastai parse failed)"
fi
echo

echo "--- sidecars in R2 ---"
SIDECAR_COUNT=$(R2 ls "$R2_SIDECAR_PREFIX" 2>/dev/null | grep -E '\.parquet$' | wc -l)
echo "$R2_SIDECAR_PREFIX  → $SIDECAR_COUNT *.parquet"

# Total target = chunks count (if available).
if R2 ls "$R2_CHUNKS" >/dev/null 2>&1; then
    N_CHUNKS=$(R2 cat "$R2_CHUNKS" 2>/dev/null | wc -l)
    PCT=$(python3 -c "print(f'{${SIDECAR_COUNT}/${N_CHUNKS}*100:.1f}')" 2>/dev/null || echo "?")
    echo "chunks total:  $N_CHUNKS"
    echo "progress:      ${SIDECAR_COUNT}/${N_CHUNKS}  (${PCT}%)"
else
    echo "(chunks.jsonl not found at $R2_CHUNKS — can't compute progress)"
fi
echo

echo "--- failure logs in R2 ---"
FAIL_COUNT=$(R2 ls "$R2_FAILURES_PREFIX" 2>/dev/null | grep -E '\.log$' | wc -l)
echo "$R2_FAILURES_PREFIX  → $FAIL_COUNT *.log"
if (( FAIL_COUNT > 0 )); then
    echo "  (recent 5):"
    R2 ls "$R2_FAILURES_PREFIX" 2>/dev/null | grep -E '\.log$' | tail -5 | sed 's/^/    /'
fi
echo

echo "--- sample sidecar validity (3 random) ---"
if (( SIDECAR_COUNT >= 1 )); then
    # Pick 3 sidecars at random and dump score-column stats. This catches
    # the iwssim-NaN-on-identical mode where every score is 0 or NaN.
    SIDECAR_LIST=$(R2 ls "$R2_SIDECAR_PREFIX" 2>/dev/null | grep -E '\.parquet$' | awk '{print $NF}')
    SAMPLE=$(echo "$SIDECAR_LIST" | shuf -n 3 2>/dev/null || echo "$SIDECAR_LIST" | head -3)
    TMPDIR=$(mktemp -d)
    trap 'rm -rf "$TMPDIR"' EXIT
    for name in $SAMPLE; do
        local_path="$TMPDIR/$name"
        R2 cp "${R2_SIDECAR_PREFIX}${name}" "$local_path" >/dev/null 2>&1 || {
            echo "  [$name] FAILED to download"
            continue
        }
        python3 - "$local_path" "$name" <<'PYEOF'
import sys
try:
    import pyarrow.parquet as pq
except ImportError:
    print("  (pyarrow not installed; cannot validate sidecars)", file=sys.stderr)
    sys.exit(0)

import math
(_, path, name) = sys.argv
t = pq.read_table(path)
cols = t.column_names
score_col = next((c for c in cols if any(k in c.lower() for k in [
    "iwssim", "ssim2", "cvvdp", "dssim", "zensim", "butteraugli",
])), None)
if score_col is None:
    print(f"  [{name}] WARN no recognisable score column; columns={cols}")
    sys.exit(0)
col = t.column(score_col).to_pylist()
n = len(col)
n_nan = sum(1 for v in col if v is None or (isinstance(v, float) and math.isnan(v)))
finite = [v for v in col if isinstance(v, float) and math.isfinite(v)]
if not finite:
    print(f"  [{name}] {score_col} 0/{n} finite — ENTIRELY BOGUS")
    sys.exit(0)
mn = min(finite)
mx = max(finite)
mean = sum(finite) / len(finite)
flag = ""
if n_nan > 0:
    flag = " *** NaN rows!"
if mx - mn < 0.01 and len(finite) >= 4:
    flag += " *** constant column!"
print(f"  [{name}] {score_col} n={n} nan={n_nan} min={mn:.4f} max={mx:.4f} mean={mean:.4f}{flag}")
PYEOF
    done
else
    echo "  (no sidecars yet — skipped validity sample)"
fi
echo

echo "--- summary ---"
echo "  fleet:    see 'zenfleet-vastai status' above"
echo "  sidecars: $SIDECAR_COUNT / ${N_CHUNKS:-?}"
echo "  failures: $FAIL_COUNT"
if (( FAIL_COUNT > 0 )); then
    echo "  ACTION: review failure logs at $R2_FAILURES_PREFIX"
fi
