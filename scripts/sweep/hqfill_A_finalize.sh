#!/usr/bin/env bash
# Finalize hqfill-A: sha256 the canonical parquet, push canonical parquet +
# encoded variants + per-chunk feature sidecars + TSVs to R2 zentrain, mirror to
# Tower. Content-addressed R2 layout under jxl-lossy-hqfill-A/2026-07-01/.
set -euo pipefail

OUT=/mnt/v/output/jxl-hqfill-A-2026-07-01
FINAL=$OUT/zenjxl_lossy_hqfill_A_2026-07-01.parquet
R2_PREFIX=s3://zentrain/jxl-lossy-hqfill-A/2026-07-01
TOWER=/mnt/tower/output/jxl-hqfill-A-2026-07-01
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export S3_ENDPOINT_URL="$ENDPOINT"

echo "=== sha256 canonical parquet ==="
SHA=$(sha256sum "$FINAL" | cut -d' ' -f1)
echo "$SHA  $(basename "$FINAL")" | tee "$OUT/SHA256SUMS"

echo "=== upload canonical parquet + analysis to R2 ($R2_PREFIX) ==="
aws s3 cp --endpoint-url="$ENDPOINT" "$FINAL" "$R2_PREFIX/$(basename "$FINAL")"
aws s3 cp --endpoint-url="$ENDPOINT" "$OUT/SHA256SUMS" "$R2_PREFIX/SHA256SUMS"

echo "=== upload encoded variants (content-addressed) to R2 via s5cmd ==="
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  s5cmd --endpoint-url="$ENDPOINT" cp "$OUT/encoded/*" "$R2_PREFIX/artifacts/zenjxl/"

echo "=== upload per-chunk feature sidecars + TSVs to R2 ==="
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  s5cmd --endpoint-url="$ENDPOINT" cp "$OUT/features/*.parquet" "$R2_PREFIX/features/"
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  s5cmd --endpoint-url="$ENDPOINT" cp "$OUT/tsv/*.tsv" "$R2_PREFIX/tsv/"

echo "=== Tower mirror ==="
mkdir -p "$TOWER"
cp "$FINAL" "$TOWER/"
cp "$OUT/SHA256SUMS" "$TOWER/"
# mirror encoded + features to Tower too
mkdir -p "$TOWER/encoded" "$TOWER/features"
cp "$OUT"/features/*.parquet "$TOWER/features/" 2>/dev/null || true
# encoded is large (~8GB) — rsync
rsync -a "$OUT/encoded/" "$TOWER/encoded/"

echo "=== verify: sha256 3 random encoded variants local vs Tower ==="
for f in $(ls "$OUT/encoded/" | shuf | head -3); do
  a=$(sha256sum "$OUT/encoded/$f" | cut -d' ' -f1)
  b=$(sha256sum "$TOWER/encoded/$f" | cut -d' ' -f1)
  [[ "$a" == "$b" ]] && echo "OK $f" || echo "MISMATCH $f"
done

echo "=== R2 verification listing ==="
aws s3 ls --endpoint-url="$ENDPOINT" "$R2_PREFIX/"
echo "canonical parquet sha256: $SHA"
