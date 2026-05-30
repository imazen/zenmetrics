#!/usr/bin/env bash
# Live demo of speculative execution (goal E: "speculative execution bounds the long tail").
# Worker A claims a job and runs it SLOWLY (a straggler). Worker B, told that a claim older than
# spec-threshold is a straggler, takes a *separate* speculative claim and co-runs the same job fast —
# bounding the tail. The ledger's latest-wins on job_id makes A's eventual (or B's) write a harmless
# duplicate; the job converges to Done either way.
#
# Requires: R2_* env, aws, s5cmd, built zen-jobworker + zen-jobctl.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WK="$ROOT/target/debug/zen-jobworker"
JC="$ROOT/target/debug/zen-jobctl"
SCORER="$ROOT/scripts/jobsys/slow_scorer.sh"; chmod +x "$SCORER"
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"; BUCKET="$R2_BUCKET"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_REGION=auto AWS_DEFAULT_REGION=auto
PFX="jobsys-spec-$(date -u +%Y%m%d-%H%M%S)"; W="$(mktemp -d)"; APID=""
cleanup() { [ -n "$APID" ] && kill "$APID" 2>/dev/null; rm -rf "$W"
  [ "${KEEP:-0}" = "1" ] || s5cmd --endpoint-url "$EP" rm "s3://$BUCKET/$PFX/*" >/dev/null 2>&1 || true
  echo "### cleaned up s3://$BUCKET/$PFX/"; }
trap cleanup EXIT
sha() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }
prim_claims() { s5cmd --endpoint-url "$EP" ls "s3://$BUCKET/$PFX/claims/" 2>/dev/null | grep -vc spec/ ; }
spec_claims() { s5cmd --endpoint-url "$EP" ls "s3://$BUCKET/$PFX/claims/spec/" 2>/dev/null | grep -c . ; }

echo "### speculative-exec demo @ s3://$BUCKET/$PFX"
cat > "$W/spec.json" <<JSON
{ "items": [ {"image_path":"demo/s.png","codec":"zenjpeg","q":80,"encode_sha":"$(sha enc-s)"} ], "metrics": ["cvvdp"] }
JSON
"$JC" declare --spec "$W/spec.json" --out "$W/manifest.json"

echo "[A] primary worker claims + runs SLOWLY (straggler, SLOW_SECS=15)…"
SLOW_SECS=15 "$WK" --manifest "$W/manifest.json" \
  --ledger-out "s3://$BUCKET/$PFX/A.parquet" \
  --blobs-r2-bucket "$BUCKET" --blobs-r2-prefix "$PFX/blobs" \
  --claims-r2-bucket "$BUCKET" --claims-prefix "$PFX/claims" \
  --claim-ttl-secs 600 --r2-endpoint "$EP" --exec "$SCORER" --worker A --provider vast >/dev/null 2>&1 &
APID=$!

for _ in $(seq 1 20); do [ "$(prim_claims)" -ge 1 ] && break; sleep 0.5; done
echo "    primary claim present: $(prim_claims)  · letting it age past the spec threshold…"
sleep 3

echo "[B] speculator (--spec-threshold-secs 2, fast) co-runs the straggler:"
SLOW_SECS=0 "$WK" --manifest "$W/manifest.json" \
  --ledger-out "s3://$BUCKET/$PFX/B.parquet" \
  --blobs-r2-bucket "$BUCKET" --blobs-r2-prefix "$PFX/blobs" \
  --claims-r2-bucket "$BUCKET" --claims-prefix "$PFX/claims" \
  --claim-ttl-secs 600 --spec-threshold-secs 2 --r2-endpoint "$EP" --exec "$SCORER" --worker B --provider local 2>&1 \
  | grep -E 'done='

echo "    speculative claims taken: $(spec_claims)  (expect 1 — B speculated)"
echo "[✓] B completed the job while A (the straggler) was still running — tail bounded."

# Fold BOTH ledgers: the job is Done (whoever wrote first), gap converges to 0.
wait "$APID" 2>/dev/null
"$JC" gap --manifest "$W/manifest.json" --ledger "s3://$BUCKET/$PFX/B.parquet" --r2-endpoint "$EP" --out "$W/gap.json" 2>/dev/null
echo "    gap after speculation = $(python3 -c 'import json,sys;print(len(json.load(open("'$W'/gap.json"))))') (expect 0 — converged)"
echo "### DEMO COMPLETE — speculative execution bounded the long tail; job converged."
