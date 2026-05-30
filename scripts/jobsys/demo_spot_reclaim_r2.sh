#!/usr/bin/env bash
# Live demo of spot-preemption reclaim (goal F: "spot reclaim is a non-event — SIGTERM releases the
# claim → requeue"). A SIGTERM is a SIGTERM whether from vast.ai/spot or a local `kill`, so this
# reproduces the real preemption path with no cloud box.
#
# Flow: start a worker on a deliberately-slow job (so SIGTERM lands mid-execution) → confirm the R2
# claim exists → `kill -TERM` the worker → confirm the claim was RELEASED (deleted) → confirm the job
# is back in the gap (requeued) → run a normal worker that completes it.
#
# Requires: R2_* env, aws, s5cmd, built zen-jobworker + zen-jobctl.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WK="$ROOT/target/debug/zen-jobworker"
JC="$ROOT/target/debug/zen-jobctl"
SCORER="$ROOT/scripts/jobsys/slow_scorer.sh"
chmod +x "$SCORER"
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET="$R2_BUCKET"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_REGION=auto AWS_DEFAULT_REGION=auto
PFX="jobsys-spot-$(date -u +%Y%m%d-%H%M%S)"
W="$(mktemp -d)"
cleanup() {
  rm -rf "$W"
  [ "${KEEP:-0}" = "1" ] || s5cmd --endpoint-url "$EP" rm "s3://$BUCKET/$PFX/*" >/dev/null 2>&1 || true
  echo "### cleaned up s3://$BUCKET/$PFX/"
}
trap cleanup EXIT

claims_count() { s5cmd --endpoint-url "$EP" ls "s3://$BUCKET/$PFX/claims/" 2>/dev/null | grep -c . ; }
sha() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }
ngap() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1"; }

echo "### spot-reclaim demo @ s3://$BUCKET/$PFX"
cat > "$W/spec.json" <<JSON
{ "items": [ {"image_path":"demo/slow.png","codec":"zenjpeg","q":80,"encode_sha":"$(sha enc-slow)"} ],
  "metrics": ["cvvdp"] }
JSON
"$JC" declare --spec "$W/spec.json" --out "$W/manifest.json"

# 1. start worker on a slow job (claims the job, then sleeps in-exec)
echo "[1] starting worker on a slow job (SLOW_SECS=8) with R2 claims…"
SLOW_SECS=8 "$WK" --manifest "$W/manifest.json" \
  --ledger-out "s3://$BUCKET/$PFX/pass.parquet" \
  --blobs-r2-bucket "$BUCKET" --blobs-r2-prefix "$PFX/blobs" \
  --claims-r2-bucket "$BUCKET" --claims-prefix "$PFX/claims" \
  --claim-ttl-secs 600 --r2-endpoint "$EP" --exec "$SCORER" --worker spot-w1 --provider vast &
WPID=$!

# 2. wait until the claim appears (job is now in-flight)
for _ in $(seq 1 20); do [ "$(claims_count)" -ge 1 ] && break; sleep 0.5; done
echo "[2] claims in R2 while job in-flight: $(claims_count)  (expect 1)"

# 3. simulate spot preemption
echo "[3] sending SIGTERM to worker (pid $WPID) — simulating spot reclaim…"
kill -TERM "$WPID"
wait "$WPID" 2>/dev/null; echo "    worker exit code: $?  (expect 130)"

# 4. the in-flight claim should have been released
sleep 1
echo "[4] claims in R2 after preemption: $(claims_count)  (expect 0 — released for fast requeue)"

# 5. the job is back in the gap (interrupted worker wrote no ledger row)
"$JC" gap --manifest "$W/manifest.json" --out "$W/gap.json" 2>/dev/null
echo "[5] gap after preemption: $(ngap "$W/gap.json") job  (expect 1 — requeued, not lost)"

# 6. a fresh worker completes it (fast exec)
echo "[6] fresh worker re-runs the requeued job (fast):"
SLOW_SECS=0 "$WK" --manifest "$W/gap.json" \
  --ledger-out "s3://$BUCKET/$PFX/pass2.parquet" \
  --blobs-r2-bucket "$BUCKET" --blobs-r2-prefix "$PFX/blobs" \
  --claims-r2-bucket "$BUCKET" --claims-prefix "$PFX/claims" \
  --claim-ttl-secs 600 --r2-endpoint "$EP" --exec "$SCORER" --worker spot-w2 --provider local
echo "### DEMO COMPLETE — SIGTERM released the in-flight claim and the job requeued + completed."
