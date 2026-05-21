#!/usr/bin/env bash
# Watches for all 10 expected sidecars to exist, then runs the
# Mohammadi panel comparison.
set -uo pipefail
OUT_DIR=/mnt/v/output/zensim/synthetic-v2
expected=(
    "${OUT_DIR}/kadid_acumen_modea_2026-05-21.parquet"
    "${OUT_DIR}/tid_acumen_modea_2026-05-21.parquet"
    "${OUT_DIR}/aic3_acumen_modea_2026-05-21.parquet"
    "${OUT_DIR}/cid22_acumen_modea_2026-05-21.parquet"
    "${OUT_DIR}/kadid_baseline_2026-05-21.parquet"
    "${OUT_DIR}/tid_baseline_2026-05-21.parquet"
    "${OUT_DIR}/aic3_baseline_2026-05-21.parquet"
    "${OUT_DIR}/cid22_baseline_2026-05-21.parquet"
)
while true; do
    all_done=1
    for p in "${expected[@]}"; do
        if [ ! -f "$p" ] || [ "$(stat -c %s "$p" 2>/dev/null || echo 0)" -lt 100 ]; then
            all_done=0
            break
        fi
    done
    if [ "$all_done" -eq 1 ]; then
        break
    fi
    sleep 60
done
echo "[$(date -u +%H:%M:%SZ)] All 10 sidecars ready — computing Mohammadi panel"
python3 /tmp/gate_a_compare.py > /tmp/gate_a_panel_output.md 2>&1
cat /tmp/gate_a_panel_output.md
