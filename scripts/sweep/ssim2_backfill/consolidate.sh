#!/usr/bin/env bash
# Sync R2 ssim2 sidecars to local, then build a single consolidated parquet.
# Run once the destroyer reports N=754 sidecars landed.
set -euo pipefail
source ~/.config/cloudflare/r2-credentials

R2_PREFIX="s3://zentrain/ssim2-backfill-2026-05-18/ssim2_imazen/"
LOCAL_DIR="/mnt/v/zen/zensim-training/2026-05-18-ssim2/ssim2_imazen/"
OUT_PARQUET="/mnt/v/zen/zensim-training/2026-05-18-ssim2/ssim2_imazen_consolidated.parquet"

mkdir -p "$LOCAL_DIR" "$(dirname "$OUT_PARQUET")"

echo "[consolidate-ssim2] step 1: sync R2 → local"
s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" --profile r2 \
    sync "$R2_PREFIX" "$LOCAL_DIR" 2>&1 | tail -10

NUM=$(ls "$LOCAL_DIR"*.parquet 2>/dev/null | wc -l)
echo "[consolidate-ssim2] step 1 done: $NUM sidecars in $LOCAL_DIR"

echo "[consolidate-ssim2] step 2: consolidate to single parquet"
python3 - "$LOCAL_DIR" "$OUT_PARQUET" <<'PYEOF'
import os, sys, glob
import pyarrow as pa
import pyarrow.parquet as pq

(_, local_dir, out_parquet) = sys.argv
files = sorted(glob.glob(os.path.join(local_dir, "*.parquet")))
print(f"  reading {len(files)} sidecars")
tables = []
for f in files:
    try:
        t = pq.read_table(f)
        tables.append(t)
    except Exception as e:
        print(f"  WARN: failed to read {os.path.basename(f)}: {e}")
combined = pa.concat_tables(tables)
print(f"  combined rows: {combined.num_rows}")
pq.write_table(combined, out_parquet, compression='zstd', compression_level=9)
print(f"  written: {out_parquet}")
print(f"  size: {os.path.getsize(out_parquet) / 1024 / 1024:.2f} MB")
PYEOF
echo "[consolidate-ssim2] done"
