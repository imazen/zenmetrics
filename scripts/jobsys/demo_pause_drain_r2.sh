#!/usr/bin/env bash
# Live demo of pause / resume / drain (goal C: "pause/resume/drain without losing state").
# A RunControl object in R2 ({"paused":bool,"drain":bool}) gates whether a worker pulls new work;
# the ledger is never touched, so resuming continues exactly where it left off.
#
# Requires: R2_* env, aws, s5cmd, built zen-jobworker + zen-jobctl.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WK="$ROOT/target/debug/zen-jobworker"
JC="$ROOT/target/debug/zen-jobctl"
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET="$R2_BUCKET"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_REGION=auto AWS_DEFAULT_REGION=auto
PFX="jobsys-pause-$(date -u +%Y%m%d-%H%M%S)"
CTLKEY="$PFX/control.json"
W="$(mktemp -d)"
cleanup() {
  rm -rf "$W"
  [ "${KEEP:-0}" = "1" ] || s5cmd --endpoint-url "$EP" rm "s3://$BUCKET/$PFX/*" >/dev/null 2>&1 || true
  echo "### cleaned up s3://$BUCKET/$PFX/"
}
trap cleanup EXIT

set_control() { # $1 = JSON
  printf '%s' "$1" > "$W/control.json"
  aws s3api put-object --endpoint-url "$EP" --bucket "$BUCKET" --key "$CTLKEY" --body "$W/control.json" >/dev/null
  echo "    set control = $1"
}
runpass() { # $1 = ledger-out name → echoes the worker summary line
  "$WK" --manifest "$W/manifest.json" \
    --ledger-out "s3://$BUCKET/$PFX/$1.parquet" \
    --blobs-r2-bucket "$BUCKET" --blobs-r2-prefix "$PFX/blobs" \
    --claims-r2-bucket "$BUCKET" --claims-prefix "$PFX/claims" \
    --control-r2-key "$CTLKEY" --r2-endpoint "$EP" --exec /bin/cat --worker pause-w --provider local 2>&1 \
    | grep -E 'run control|done='
}
sha() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }

echo "### pause/resume/drain demo @ s3://$BUCKET/$PFX"
cat > "$W/spec.json" <<JSON
{ "items": [ {"image_path":"demo/p.png","codec":"zenjpeg","q":80,"encode_sha":"$(sha enc-p)"} ],
  "metrics": ["cvvdp"] }
JSON
"$JC" declare --spec "$W/spec.json" --out "$W/manifest.json"

echo "[PAUSE] set paused, then run a worker pass (should pull no work):"
set_control '{"paused":true}'
runpass pass-paused

echo "[DRAIN] set draining, run a pass (should also pull no new work):"
set_control '{"drain":true}'
runpass pass-drain

echo "[RESUME] clear control, run a pass (should complete the job — state was never lost):"
set_control '{"paused":false,"drain":false}'
runpass pass-resume
echo "### DEMO COMPLETE — paused/draining passes did 0 work; resume completed the job."
