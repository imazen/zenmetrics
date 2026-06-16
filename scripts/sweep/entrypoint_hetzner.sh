#!/usr/bin/env bash
# entrypoint_hetzner.sh — Hetzner deploy-image entrypoint.
#
# Unlike Salad (sidecar + push) or vast.ai (in-image worker + R2 atomic
# claim), Hetzner has NO managed queue and NO sidecar. The cloud-init
# script the launcher's `provision()` synthesizes runs:
#
#   docker run -d --env-file=/etc/zen/worker.env <image> \
#     /usr/local/bin/zenfleet-sweep worker --backend hetzner \
#         --run-id=<sweep_id> \
#         --chunks-r2=s3://<bucket>/runs/<sweep_id>/chunks.jsonl
#
# So in the default Hetzner deploy path this entrypoint is NOT invoked —
# `zenfleet-sweep` is the container's entrypoint, and the env-file
# carries every env var the worker reads.
#
# This script exists for the OPTIONAL case where the operator overrides
# the docker image's entrypoint with this script (e.g. when wrapping a
# debug `bash` shell around the worker). It performs the same env-
# hydration the Salad entrypoint does, then execs the worker.
#
# Run-time env contract (from cloud-init's --env-file):
#   SWEEP_RUN_ID            REQUIRED. Sweep id (R2 prefix scope).
#   WORKER_BACKEND          Set to "hetzner".
#   R2_ACCOUNT_ID           REQUIRED. R2 account for endpoint derivation.
#   R2_ACCESS_KEY_ID        REQUIRED. Scoped R2 cred key id.
#   R2_SECRET_ACCESS_KEY    REQUIRED. Scoped R2 cred secret.
#   AWS_SESSION_TOKEN       REQUIRED. Scoped R2 session token.
#   BUCKET                  REQUIRED. R2 working bucket.
#   CHUNKS_QUEUE_PREFIX     REQUIRED. e.g. `runs/<sweep_id>/queue/`.
#   CHUNKS_R2               REQUIRED. The chunks.jsonl manifest URI.

set -eu
log() { echo "[entrypoint-hetzner] $*" >&2; }

# Required vars (fail loud so the worker container exits with a clear
# message instead of a silent hang).
: "${SWEEP_RUN_ID:?SWEEP_RUN_ID env missing}"
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID env missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID env missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY env missing}"
: "${AWS_SESSION_TOKEN:?AWS_SESSION_TOKEN env missing (scoped R2 cred REQUIRES session token)}"
: "${BUCKET:?BUCKET env missing}"
: "${CHUNKS_QUEUE_PREFIX:?CHUNKS_QUEUE_PREFIX env missing}"
: "${CHUNKS_R2:?CHUNKS_R2 env missing}"

# s5cmd credentials file mirrors entrypoint_salad.sh's shape.
mkdir -p ~/.aws
{
    echo "[r2]"
    echo "aws_access_key_id = ${R2_ACCESS_KEY_ID}"
    echo "aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}"
    echo "aws_session_token = ${AWS_SESSION_TOKEN}"
} > ~/.aws/credentials

log "run_id=${SWEEP_RUN_ID} bucket=${BUCKET} queue=${CHUNKS_QUEUE_PREFIX}"
log "launching zenfleet-sweep --backend hetzner"

exec /usr/local/bin/zenfleet-sweep worker --backend hetzner \
    --run-id "${SWEEP_RUN_ID}" \
    --chunks-r2 "${CHUNKS_R2}"
