#!/usr/bin/env bash
# Keep-alive fleet-worker entrypoint (goal H). Claims work off the one R2 conditional-write-lease queue
# and runs it until the gap drains (K consecutive passes win nothing) or the box is reclaimed
# (SIGTERM → zen-jobworker releases its in-flight claim → fast requeue, goal F). No boot-time installs:
# every tool is baked into the image; if one is missing we fail loud (the image is broken, rebuild it).
#
# Env (the launcher / cloud-init injects these):
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN  — SCOPED temp R2 creds (never root)
#   ZEN_R2_ENDPOINT   — https://<acct>.r2.cloudflarestorage.com
#   ZEN_BUCKET        — R2 bucket
#   ZEN_RUN           — run prefix under the bucket (queue namespace)
#   ZEN_MANIFEST_URI  — s3:// URI of the DesiredJob manifest to work
#   ZEN_PROVIDER      — tier label for ledger rows (hetzner/vast/oracle/basement/local/…)
#   ZEN_WORKER        — worker id (default: hostname)
#   ZEN_EXEC          — executor program (default /bin/cat; a real box bakes its encoder/scorer)
#   ZEN_SPEC_THRESHOLD_SECS — optional speculative-execution threshold (goal E)
#   ZEN_CONTROL_KEY   — optional s3 key of a RunControl object for pause/drain (goal C)
#   ZEN_IDLE_PASSES (5) / ZEN_PASS_SLEEP (0.2) — drain detection + pacing
set -uo pipefail
: "${ZEN_R2_ENDPOINT:?}" "${ZEN_BUCKET:?}" "${ZEN_RUN:?}" "${ZEN_MANIFEST_URI:?}"
PROVIDER="${ZEN_PROVIDER:-fleet}"
WORKER="${ZEN_WORKER:-$(hostname)}"
EXEC="${ZEN_EXEC:-/bin/cat}"
export AWS_REGION="${AWS_REGION:-auto}" AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"

for tool in aws s5cmd zen-jobworker; do
  command -v "$tool" >/dev/null || { echo "FATAL: baked tool '$tool' missing — image is broken, rebuild"; exit 3; }
done

s5cmd --endpoint-url "$ZEN_R2_ENDPOINT" cp "$ZEN_MANIFEST_URI" /tmp/manifest.json \
  || { echo "FATAL: cannot fetch manifest $ZEN_MANIFEST_URI"; exit 4; }
echo "fleet-entrypoint: $WORKER ($PROVIDER) claiming from s3://$ZEN_BUCKET/$ZEN_RUN/"

idle=0 i=0
while [ "$idle" -lt "${ZEN_IDLE_PASSES:-5}" ]; do
  i=$((i + 1))
  out=$(zen-jobworker --manifest /tmp/manifest.json \
    --ledger-out "s3://$ZEN_BUCKET/$ZEN_RUN/ledger/pass-$WORKER-$i.parquet" \
    --blobs-r2-bucket "$ZEN_BUCKET" --blobs-r2-prefix "$ZEN_RUN/blobs" \
    --claims-r2-bucket "$ZEN_BUCKET" --claims-prefix "$ZEN_RUN/claims" \
    ${ZEN_SPEC_THRESHOLD_SECS:+--spec-threshold-secs "$ZEN_SPEC_THRESHOLD_SECS"} \
    ${ZEN_CONTROL_KEY:+--control-r2-key "$ZEN_CONTROL_KEY"} \
    --r2-endpoint "$ZEN_R2_ENDPOINT" --exec "$EXEC" --worker "$WORKER" --provider "$PROVIDER" 2>&1 | tail -1)
  echo "$(date -u +%H:%M:%S) $out"
  # "won nothing this pass" K times in a row ⇒ the gap is drained or fully claimed by peers ⇒ exit.
  if echo "$out" | grep -qE 'done=0 '; then idle=$((idle + 1)); else idle=0; fi
  sleep "${ZEN_PASS_SLEEP:-0.2}"
done
echo "fleet-entrypoint: $WORKER drained (idle $idle passes) — exiting clean"
