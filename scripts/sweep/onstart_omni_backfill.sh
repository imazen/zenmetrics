#!/usr/bin/env bash
# onstart_omni_backfill.sh — entry point for the omni (multi-metric +
# encoded-variants) backfill. Wraps the same chunk-claim + heartbeat
# loop as onstart_cvvdp_backfill_imazen.sh but pulls the omni worker
# instead of the cvvdp dual-impl one.
set -uo pipefail

if [[ -r /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        case "$k" in
            R2_*|SWEEP_*|WORKER_*|PARALLEL|GPU_RUNTIME|CONTAINER_*)
                export "$k=$v" ;;
        esac
    done < /proc/1/environ
fi

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing}"
: "${SWEEP_RUN_ID:?SWEEP_RUN_ID missing}"

WORKER_ID="${WORKER_ID:-$(hostname)-$$}"
PARALLEL="${PARALLEL:-0}"
[[ "$PARALLEL" == "auto" ]] && PARALLEL=0
GPU_RUNTIME="${GPU_RUNTIME:-cuda}"
WORKDIR="${WORKDIR:-/workspace/omni-backfill}"
SCRIPTS_R2_PREFIX="${SCRIPTS_R2_PREFIX:-s3://coefficient/jobs/${SWEEP_RUN_ID}}"

mkdir -p "$WORKDIR" ~/.aws
cat > ~/.aws/credentials <<CREDS
[r2]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
CREDS

R2() { s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" --profile r2 "$@"; }

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] [omni-onstart] %s\n' "$(ts)" "$*" >&2; }

for tool in zen-metrics s5cmd jq python3; do
    command -v "$tool" >/dev/null || { log "FAIL: $tool missing"; exit 2; }
done

log "pulling chunks.jsonl + omni worker from $SCRIPTS_R2_PREFIX"
R2 cp "${SCRIPTS_R2_PREFIX}/chunks.jsonl" "$WORKDIR/chunks.jsonl"
# Worker may be in /usr/local/bin (image bake) OR be fetched from R2.
if [[ ! -x /usr/local/bin/omni_backfill_chunk_worker.sh ]]; then
    R2 cp "${SCRIPTS_R2_PREFIX}/omni_backfill_chunk_worker.sh" \
        /usr/local/bin/omni_backfill_chunk_worker.sh
    chmod +x /usr/local/bin/omni_backfill_chunk_worker.sh
fi

N_CHUNKS=$(wc -l < "$WORKDIR/chunks.jsonl")
log "$N_CHUNKS chunks; worker=$WORKER_ID; PARALLEL=$PARALLEL; GPU_RUNTIME=$GPU_RUNTIME"

# Shuffle + claim loop. One chunk at a time; the worker handles
# per-group parallelism inside via --parallel.
shuf "$WORKDIR/chunks.jsonl" > "$WORKDIR/chunks.shuf.jsonl"

process_chunk() {
    local line="$1"
    local cid out_sidecar
    cid=$(jq -r '.chunk_id' <<< "$line")
    # The omni worker uploads its sidecar to
    # s3://zentrain/<run>/omni/<cid>.parquet by default; respect any
    # override the chunk JSON specifies.
    out_sidecar=$(jq -r --arg r "$SWEEP_RUN_ID" --arg c "$cid" \
        '.out_sidecar_omni // ("s3://zentrain/" + $r + "/omni/" + $c + ".parquet")' \
        <<< "$line")

    local CLAIM_KEY="s3://coefficient/claims/${SWEEP_RUN_ID}/${cid}.claim"

    # Idempotency: skip if the sidecar is already uploaded (covers
    # resumes after crashes + dedups across concurrent workers).
    if R2 ls "$out_sidecar" >/dev/null 2>&1; then
        log "[skip-done] $cid sidecar already in R2"
        return 0
    fi

    if [[ "${SKIP_CLAIMS:-0}" != "1" ]]; then
        # Token-based claim with read-back verification. Matches the
        # pattern in onstart_cvvdp_backfill_imazen.sh which has been
        # battle-tested across iwssim + cvvdp fleets:
        #   1. Write a unique token (worker-id + pid + nanos).
        #   2. Sleep briefly so any concurrent writers settle.
        #   3. Read back the claim; if our token survived, we own it.
        #      Otherwise another worker won the race — skip this chunk.
        # A claim older than CLAIM_STALE_SEC is treated as abandoned
        # (worker crashed) and overwritten.
        local claim_body=/tmp/claim-${cid}.txt
        local token="${WORKER_ID}-$$-$(date +%s%N)"
        local now_epoch; now_epoch=$(date +%s)
        local CLAIM_STALE_SEC="${CLAIM_STALE_SEC:-600}"
        printf '%s\t%s\t%s' "$token" "$now_epoch" "$WORKER_ID" > "$claim_body"

        local existing
        existing=$(R2 cat "$CLAIM_KEY" 2>/dev/null) || existing=""
        if [[ -n "$existing" ]]; then
            local existing_epoch existing_worker
            existing_epoch=$(printf '%s' "$existing" | awk -F'\t' '{print $2}')
            existing_worker=$(printf '%s' "$existing" | awk -F'\t' '{print $3}')
            if [[ -n "$existing_epoch" ]] \
                    && (( now_epoch - existing_epoch < CLAIM_STALE_SEC )) \
                    && [[ "$existing_worker" != "$WORKER_ID" ]]; then
                # Fresh claim held by another worker.
                rm -f "$claim_body"
                return 0
            fi
            # Stale claim or own claim — overwrite below.
        fi

        if ! R2 cp "$claim_body" "$CLAIM_KEY" 2>/dev/null; then
            log "WARN: claim upload failed for $cid; skipping"
            rm -f "$claim_body"
            return 0
        fi
        rm -f "$claim_body"

        # Read-back verification — guard against last-writer-wins on
        # near-simultaneous writes.
        sleep 1.5
        local verified
        verified=$(R2 cat "$CLAIM_KEY" 2>/dev/null | awk -F'\t' '{print $1}')
        if [[ "$verified" != "$token" ]]; then
            log "[lost-race] $cid (claim now=$verified)"
            return 0
        fi
    fi
    log "claimed $cid; running worker"
    if PARALLEL="$PARALLEL" GPU_RUNTIME="$GPU_RUNTIME" \
        /usr/local/bin/omni_backfill_chunk_worker.sh --chunk-json "$line" 2>&1 \
        | sed "s/^/  /" >&2; then
        log "$cid done"
    else
        log "$cid FAILED (worker exited non-zero)"
        # Don't bail the whole onstart; move to next chunk. The trap
        # wrapper will self-destroy if the box has a fatal issue.
    fi
}
export -f process_chunk log ts R2
export R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY \
    SWEEP_RUN_ID WORKER_ID PARALLEL GPU_RUNTIME SCRIPTS_R2_PREFIX

while IFS= read -r line; do
    process_chunk "$line"
done < "$WORKDIR/chunks.shuf.jsonl"

log "all chunks processed"
