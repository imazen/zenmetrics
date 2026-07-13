#!/usr/bin/env bash
# onstart_hdr_pairs.sh — vast.ai onstart for the HDR persisted-pairs metric
# fleet (kadis-hdr-style corpora). Static modulo sharding: box SHARD_IDX of
# SHARD_N owns chunks where line_index % SHARD_N == SHARD_IDX — no claim
# infrastructure needed at this fleet size; per-chunk sidecar existence makes
# reruns idempotent. HDR binary arrives via SWEEP_BIN_OVERRIDE (the
# hetzner_cpu_sweep.sh precedent — the baked image lacks the `hdr` feature
# until the task-8 bake lands).
#
# Env contract (set by launch_backfill.sh --onstart this):
#   R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY  (scoped creds;
#       R2_SESSION_TOKEN or split _0.._7 parts per the launcher)
#   SWEEP_RUN_ID     run id; chunks default s3://zentrain/<run>/chunks.jsonl
#   CHUNKS_R2        optional explicit chunks.jsonl URL
#   SCRIPTS_R2_PREFIX  where hdr_pairs_chunk_worker.sh was uploaded
#   SWEEP_BIN_OVERRIDE s3:// URL of the hdr zenmetrics binary
#   SHARD_IDX / SHARD_N  static shard assignment (launcher per-box env)
#   METRICS / GPU_RUNTIME  forwarded to the worker
set -euo pipefail

if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|METRICS|CHUNKS_*|SCRIPTS_*|SHARD_*|GPU_*) export "$k=$v" ;;
        esac
    done < /proc/1/environ
fi
: "${R2_ACCOUNT_ID:?}" ; : "${R2_ACCESS_KEY_ID:?}" ; : "${R2_SECRET_ACCESS_KEY:?}"
: "${SWEEP_RUN_ID:?}" ; : "${SWEEP_BIN_OVERRIDE:?}" ; : "${SCRIPTS_R2_PREFIX:?}"
SHARD_IDX="${SHARD_IDX:-0}"; SHARD_N="${SHARD_N:-1}"

mkdir -p ~/.aws
{ echo "[r2]"
  echo "aws_access_key_id = ${R2_ACCESS_KEY_ID}"
  echo "aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}"; } > ~/.aws/credentials
if [[ -z "${R2_SESSION_TOKEN:-}" ]]; then
    _st=""
    for _i in 0 1 2 3 4 5 6 7; do _v="R2_SESSION_TOKEN_${_i}"; [[ -n "${!_v:-}" ]] && _st="${_st}${!_v}"; done
    [[ -n "$_st" ]] && export R2_SESSION_TOKEN="$_st"
fi
[[ -n "${R2_SESSION_TOKEN:-}" ]] && echo "aws_session_token = ${R2_SESSION_TOKEN}" >> ~/.aws/credentials

EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2() { s5cmd --endpoint-url "$EP" --profile r2 "$@"; }

echo "[onstart-hdr-pairs] shard ${SHARD_IDX}/${SHARD_N} run=${SWEEP_RUN_ID}" >&2
R2 cp "$SWEEP_BIN_OVERRIDE" /usr/local/bin/zenmetrics && chmod +x /usr/local/bin/zenmetrics
/usr/local/bin/zenmetrics --version >&2 || { echo "binary broken" >&2; exit 1; }
R2 cp "${SCRIPTS_R2_PREFIX}/hdr_pairs_chunk_worker.sh" /usr/local/bin/hdr_pairs_chunk_worker.sh
chmod +x /usr/local/bin/hdr_pairs_chunk_worker.sh

CHUNKS_R2="${CHUNKS_R2:-s3://zentrain/${SWEEP_RUN_ID}/chunks.jsonl}"
R2 cp "$CHUNKS_R2" /tmp/chunks.jsonl

rc_total=0; idx=-1
while IFS= read -r line; do
    [[ "$line" == \{* ]] || continue
    idx=$((idx+1))
    (( idx % SHARD_N == SHARD_IDX )) || continue
    cid=$(echo "$line" | jq -r '.chunk_id'); outp=$(echo "$line" | jq -r '.out_prefix')
    # Idempotency: the worker writes _DONE last, only on full success.
    if R2 ls "${outp}/${cid}/_DONE" >/dev/null 2>&1; then
        echo "[onstart-hdr-pairs] $cid already scored — skip" >&2; continue
    fi
    echo "[onstart-hdr-pairs] processing $cid" >&2
    if ! CHUNK_JSON="$line" METRICS="${METRICS:-}" GPU_RUNTIME="${GPU_RUNTIME:-cuda}" \
         /usr/local/bin/hdr_pairs_chunk_worker.sh; then
        echo "[onstart-hdr-pairs] $cid FAILED (continuing)" >&2; rc_total=1
    fi
done < /tmp/chunks.jsonl

echo "[onstart-hdr-pairs] shard complete rc=$rc_total — self-destroying" >&2
# Billing protection: destroy this instance when the shard is done.
# REQUIRES VAST_API_KEY in the box env (launcher: INJECT_VAST_API_KEY=1).
# Without it the DELETE silently no-ops and the box IDLES AT FULL COST —
# observed 2026-07-13 (4 drained boxes sat "running" until manually reaped).
# Fallback when not injecting the key: run `zenfleet-vastai watch` locally,
# or a log-tail reaper that destroys boxes printing "shard complete".
if [[ -z "${VAST_API_KEY:-}" ]]; then
    echo "[onstart-hdr-pairs] WARNING: VAST_API_KEY not set — cannot self-destroy; operator must reap this box" >&2
fi
if [[ -n "${CONTAINER_ID:-${VAST_CONTAINERLABEL:-}}" && -n "${VAST_API_KEY:-}" ]]; then
    IID="${CONTAINER_ID:-${VAST_CONTAINERLABEL##C.}}"
    curl -s -X DELETE "https://console.vast.ai/api/v0/instances/${IID}/?api_key=${VAST_API_KEY}" >/dev/null 2>&1 || true
fi
exit "$rc_total"
