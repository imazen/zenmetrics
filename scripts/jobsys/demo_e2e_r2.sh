#!/usr/bin/env bash
# Live end-to-end demo of the zen job system against an ISOLATED R2 prefix.
#
# Proves, live (not just unit tests):
#   A  declare → manifest; gap is idempotent (re-run after work = empty, structural no-op)
#   E  worker converges the gap to done; restartable (2nd pass skips done); R2 ledger is truth
#   I  coverage query (done/poison/gap per codec×metric) derived from the ledger
#   Foundations  content-addressed blobs in R2, columnar Parquet ledger in R2,
#                R2 conditional-write-lease queue (exactly-one claim per job)
#
# Synthetic metric jobs; the executor is /bin/cat (emits the job JSON as the score blob), so the
# demo needs no encoder/GPU and costs a handful of tiny R2 objects, all under one prefix that is
# deleted at the end (pass KEEP=1 to retain).
#
# Requires: R2_* env (R2_ACCOUNT_ID/R2_BUCKET/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY), aws (v1.44+
# supports put-object --if-none-match), s5cmd, and built zen-jobworker + zen-jobctl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WK="$ROOT/target/debug/zen-jobworker"
JC="$ROOT/target/debug/zen-jobctl"
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET="$R2_BUCKET"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_REGION=auto AWS_DEFAULT_REGION=auto
PFX="jobsys-demo-$(date -u +%Y%m%d-%H%M%S)"
W="$(mktemp -d)"
cleanup() {
  rm -rf "$W"
  if [ "${KEEP:-0}" != "1" ]; then
    s5cmd --endpoint-url "$EP" rm "s3://$BUCKET/$PFX/*" >/dev/null 2>&1 || true
    echo "### cleaned up s3://$BUCKET/$PFX/"
  else
    echo "### KEPT s3://$BUCKET/$PFX/"
  fi
}
trap cleanup EXIT

sha() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }
n() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1"; }

echo "### zen job system — live E2E demo @ s3://$BUCKET/$PFX"
echo

# ── A. declare a spec into a DesiredJob manifest ────────────────────────────────
cat > "$W/spec.json" <<JSON
{ "items": [
   {"image_path":"demo/a.png","codec":"zenjpeg","q":80,"encode_sha":"$(sha enc-a)"},
   {"image_path":"demo/b.png","codec":"zenavif","q":50,"encode_sha":"$(sha enc-b)"}
 ], "metrics": ["cvvdp","ssim2"] }
JSON
echo "[A] declare spec (2 items × 2 metrics):"
"$JC" declare --spec "$W/spec.json" --out "$W/manifest.json"

echo "[A] gap with no ledger (nothing done yet):"
"$JC" gap --manifest "$W/manifest.json" --out "$W/gap0.json"
echo "    gap before work = $(n "$W/gap0.json") jobs"
echo

# ── E. worker pass 1: claim (R2 conditional-write) → exec → R2 blobs + R2 Parquet ledger ──
echo "[E] worker pass 1 (claims via R2 lease, exec=/bin/cat, ledger+blobs to R2):"
"$WK" --manifest "$W/gap0.json" \
  --ledger-out "s3://$BUCKET/$PFX/pass1.parquet" \
  --blobs-r2-bucket "$BUCKET" --blobs-r2-prefix "$PFX/blobs" \
  --claims-r2-bucket "$BUCKET" --claims-prefix "$PFX/claims" \
  --r2-endpoint "$EP" --exec /bin/cat --worker demo-w1 --provider local
echo

# ── Foundations: content-addressed blobs + Parquet ledger landed in R2 ──────────
echo "[F] R2 artifacts written by pass 1:"
echo "    blobs (content-addressed):"
s5cmd --endpoint-url "$EP" ls "s3://$BUCKET/$PFX/blobs/" 2>/dev/null | sed 's/^/      /'
echo "    ledger (Parquet) + claims:"
s5cmd --endpoint-url "$EP" ls "s3://$BUCKET/$PFX/pass1.parquet" 2>/dev/null | sed 's/^/      /'
s5cmd --endpoint-url "$EP" ls "s3://$BUCKET/$PFX/claims/" 2>/dev/null | sed 's/^/      /'
echo

# ── I. coverage query, derived from the ledger ──────────────────────────────────
echo "[I] coverage (done/poison/gap per codec×metric) from the R2 ledger:"
"$JC" catalog --manifest "$W/manifest.json" --ledger "s3://$BUCKET/$PFX/pass1.parquet" --r2-endpoint "$EP"
echo

# ── A/I idempotency: gap after work is empty (re-declaring done work is a no-op) ─
echo "[A/I] gap AFTER pass 1 (idempotent — should be 0):"
"$JC" gap --manifest "$W/manifest.json" --ledger "s3://$BUCKET/$PFX/pass1.parquet" --r2-endpoint "$EP" --out "$W/gap1.json"
echo "    gap after work = $(n "$W/gap1.json") jobs"
echo

# ── E restartable: a 2nd worker pass folding in the ledger skips all done jobs ───
echo "[E] worker pass 2 (same manifest, folds in pass1 ledger — should skip all, do 0):"
"$WK" --manifest "$W/manifest.json" --ledger-in "s3://$BUCKET/$PFX/pass1.parquet" \
  --ledger-out "s3://$BUCKET/$PFX/pass2.parquet" \
  --blobs-r2-bucket "$BUCKET" --blobs-r2-prefix "$PFX/blobs" \
  --r2-endpoint "$EP" --exec /bin/cat --worker demo-w2 --provider local
echo
echo "### DEMO COMPLETE — A (declare/idempotent gap), E (converge+restartable+lease), I (coverage) shown live."
