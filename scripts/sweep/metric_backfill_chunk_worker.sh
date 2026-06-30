#!/usr/bin/env bash
#
# metric_backfill_chunk_worker.sh — unified single-metric backfill chunk
# worker.
#
# Replaces the per-metric files:
#   - iwssim_backfill_chunk_worker.sh
#   - ssim2_backfill_chunk_worker.sh (lives on feat/ex3-ssim2-target-backfill)
#   - the single-metric subset of cvvdp_backfill_chunk_worker.sh
#
# This worker handles ONE metric per invocation (--metric). The
# dual-implementation cvvdp flow (cvvdp-gpu + pycvvdp side-by-side via
# cvvdp_backfill_chunk_worker.sh) was removed 2026-06-25 — cvvdp now scores
# through the unified worker, and cvvdp-gpu<->pycvvdp parity is validated in
# the cvvdp-gpu crate (goldens + CHROMA_DRIFT_INVESTIGATION.md), not the fleet.
#
# The worker:
#   1. Reads one chunk-manifest line (JSON object) via --chunk-json or
#      stdin.
#   2. Downloads the input parquet from R2 into a scratch dir.
#   3. Syncs the chunk's image_basenames from R2's source_dir_r2.
#   4. Reads rows[row_range[0]:row_range[1]] from the parquet, groups by
#      (codec, q, knob_tuple_json), and re-encodes the dist images via
#      `zenmetrics sweep` once per group.
#   5. Runs `zenmetrics score-pairs --metric <METRIC> --fail-on-bogus`
#      to produce the per-metric sidecar.
#   6. If score-pairs exited rc=2 (bogus): uploads a structured failure
#      log to s3://zentrain/<run>/failures/<chunk>.log instead of the
#      sidecar, then exits rc=2.
#      If score-pairs exited rc=0: uploads the sidecar to
#      out_sidecar_<metric> from the chunk manifest.
#
# Chunk-manifest field name for the sidecar output path:
#   - iwssim   → out_sidecar_iwssim
#   - ssim2    → out_sidecar_ssim2
#   - cvvdp    → out_sidecar_imazen   (matches the dual-impl chunk gen)
#   - (other)  → out_sidecar_<metric_short>
# Override with --out-sidecar-field if your generator uses a different
# name.
#
# Required tools on PATH (or docker image with same baked in):
#   - s5cmd (R2 transfers)
#   - jq    (chunk JSON parsing)
#   - python3 with pyarrow (parquet slicing)
#   - docker (if running scorer in a container)
#
# Required env vars (R2 credentials, same as onstart_unified.sh):
#   R2_ACCOUNT_ID  R2_ACCESS_KEY_ID  R2_SECRET_ACCESS_KEY
#
# Usage:
#
#   echo '<one chunk JSON line>' | \
#       metric_backfill_chunk_worker.sh \
#           --metric iwssim-gpu \
#           --zenmetrics-image ghcr.io/imazen/zenmetrics-sweep:0.6.4-iwssim-fixed-6227c1a
#
# OR (host binary, no docker):
#
#   metric_backfill_chunk_worker.sh \
#       --metric ssim2 \
#       --chunk-json "$(head -1 chunks.jsonl)" \
#       --work-dir /tmp/ssim2-chunk \
#       --skip-upload
#
# Env var alternatives (all match flags):
#   METRIC, CHUNK_JSON, WORK_DIR, ZEN_METRICS_IMAGE, GPU_RUNTIME,
#   FAIL_ON_BOGUS (default 1), SKIP_UPLOAD (default 0),
#   KEEP_WORK (default 0)

set -euo pipefail

METRIC="${METRIC:-}"
CHUNK_JSON="${CHUNK_JSON:-}"
WORK_DIR="${WORK_DIR:-}"
ZEN_METRICS_IMAGE="${ZEN_METRICS_IMAGE:-}"
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
DOCKER_GPUS="${DOCKER_GPUS:---gpus all}"
KEEP_WORK="${KEEP_WORK:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
FAIL_ON_BOGUS="${FAIL_ON_BOGUS:-1}"
OUT_SIDECAR_FIELD="${OUT_SIDECAR_FIELD:-}"
SWEEP_METRIC_FOR_ENCODE="${SWEEP_METRIC_FOR_ENCODE:-ssim2}"
EXTRA_SCORE_PAIRS_ARGS="${EXTRA_SCORE_PAIRS_ARGS:-}"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

[[ $# -gt 0 && ("$1" == "-h" || "${1:-}" == "--help") ]] && usage 0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --metric) METRIC="$2"; shift 2;;
        --chunk-json) CHUNK_JSON="$2"; shift 2;;
        --work-dir) WORK_DIR="$2"; shift 2;;
        --zenmetrics-image) ZEN_METRICS_IMAGE="$2"; shift 2;;
        --gpu-runtime) GPU_RUNTIME="$2"; shift 2;;
        --out-sidecar-field) OUT_SIDECAR_FIELD="$2"; shift 2;;
        --sweep-metric-for-encode) SWEEP_METRIC_FOR_ENCODE="$2"; shift 2;;
        --extra-score-pairs-args) EXTRA_SCORE_PAIRS_ARGS="$2"; shift 2;;
        --no-fail-on-bogus) FAIL_ON_BOGUS=0; shift;;
        --keep-work) KEEP_WORK=1; shift;;
        --skip-upload) SKIP_UPLOAD=1; shift;;
        *) echo "unknown arg: $1" >&2; usage 1;;
    esac
done

if [[ -z "$METRIC" ]]; then
    echo "ERROR: --metric is required" >&2
    usage 1
fi

# Default chunk-manifest field name from the metric.
# `cvvdp` is the historical alias for the imazen single-impl sidecar in
# the dual-impl chunks.jsonl.
if [[ -z "$OUT_SIDECAR_FIELD" ]]; then
    case "$METRIC" in
        iwssim|iwssim-gpu) OUT_SIDECAR_FIELD="out_sidecar_iwssim";;
        ssim2|ssim2-gpu)   OUT_SIDECAR_FIELD="out_sidecar_ssim2";;
        cvvdp)             OUT_SIDECAR_FIELD="out_sidecar_imazen";;
        dssim|dssim-gpu)   OUT_SIDECAR_FIELD="out_sidecar_dssim";;
        zensim|zensim-gpu) OUT_SIDECAR_FIELD="out_sidecar_zensim";;
        butteraugli|butteraugli-gpu) OUT_SIDECAR_FIELD="out_sidecar_butteraugli";;
        *)
            short="${METRIC%-gpu}"
            OUT_SIDECAR_FIELD="out_sidecar_${short}"
            ;;
    esac
fi

# Default scratch dir keyed by metric so concurrent workers in the same
# host don't collide.
if [[ -z "$WORK_DIR" ]]; then
    WORK_DIR="/tmp/${METRIC}-chunk-$$"
fi

if [[ -z "$CHUNK_JSON" ]]; then
    CHUNK_JSON="$(cat)"
fi
if [[ -z "$CHUNK_JSON" ]]; then
    echo "ERROR: no chunk JSON (pass --chunk-json or pipe to stdin)" >&2
    exit 1
fi

for tool in jq python3 s5cmd; do
    command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done

CHUNK_ID=$(echo "$CHUNK_JSON" | jq -r '.chunk_id')
INPUT_PARQUET=$(echo "$CHUNK_JSON" | jq -r '.input_parquet')
INPUT_PARQUET_R2=$(echo "$CHUNK_JSON" | jq -r '.input_parquet_r2')
ROW_START=$(echo "$CHUNK_JSON" | jq -r '.row_range[0]')
ROW_END=$(echo "$CHUNK_JSON" | jq -r '.row_range[1]')
SOURCE_DIR_R2=$(echo "$CHUNK_JSON" | jq -r '.source_dir_r2')
OUT_SIDECAR_R2=$(echo "$CHUNK_JSON" | jq -r --arg f "$OUT_SIDECAR_FIELD" '.[$f]')

# RUN_ID for the failure-log path. Prefer the chunk's run_id field if
# present, else derive from the sidecar path's first directory.
RUN_ID=$(echo "$CHUNK_JSON" | jq -r '.run_id // empty')
if [[ -z "$RUN_ID" || "$RUN_ID" == "null" ]]; then
    # s3://zentrain/<run-id>/sidecars/... → strip leading "s3://zentrain/"
    # and take the first path component.
    RUN_ID=$(echo "$OUT_SIDECAR_R2" | sed -E 's#^s3://zentrain/##; s#/.*$##')
fi

if [[ -z "$CHUNK_ID" || "$CHUNK_ID" == "null" ]]; then
    echo "ERROR: chunk_id missing from JSON" >&2
    exit 1
fi
if [[ -z "$OUT_SIDECAR_R2" || "$OUT_SIDECAR_R2" == "null" ]]; then
    echo "ERROR: $OUT_SIDECAR_FIELD missing from JSON" >&2
    exit 1
fi

mkdir -p "$WORK_DIR/sources" "$WORK_DIR/dist" "$WORK_DIR/out"
cd "$WORK_DIR"

cleanup() {
    if [[ "$KEEP_WORK" != "1" ]]; then
        cd /
        rm -rf "$WORK_DIR"
    fi
}
trap cleanup EXIT

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing in environment}"
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2() {
    s5cmd --endpoint-url "$R2_ENDPOINT" --profile r2 "$@"
}

LOG_PREFIX="[${METRIC}-chunk-worker $CHUNK_ID]"

echo "$LOG_PREFIX step 1/6: download input parquet" >&2
R2 cp "$INPUT_PARQUET_R2" "$WORK_DIR/$INPUT_PARQUET" >&2

echo "$LOG_PREFIX step 2/6: sync source basenames" >&2
N_BASENAMES=$(echo "$CHUNK_JSON" | jq -r '.image_basenames | length')
echo "  $N_BASENAMES unique basenames" >&2
echo "$CHUNK_JSON" | jq -r --arg src "$SOURCE_DIR_R2" '
    .image_basenames[] |
    "cp \($src)/\(.) \(.)"
' > "$WORK_DIR/sources/_download.run"
( cd "$WORK_DIR/sources" && s5cmd --endpoint-url "$R2_ENDPOINT" --profile r2 run "$WORK_DIR/sources/_download.run" >&2 ) || {
    echo "ERROR: failed to sync sources" >&2
    exit 2
}
rm -f "$WORK_DIR/sources/_download.run"

echo "$LOG_PREFIX step 3/6: slice + group" >&2
python3 - "$WORK_DIR/$INPUT_PARQUET" "$ROW_START" "$ROW_END" "$WORK_DIR" <<'PYEOF' >&2
import json
import os
import sys
from collections import defaultdict

import pyarrow.parquet as pq

(_, parquet_path, row_start, row_end, work_dir) = sys.argv
row_start = int(row_start)
row_end = int(row_end)

table = pq.read_table(
    parquet_path,
    columns=["image_path", "codec", "q", "knob_tuple_json"],
)
rows = table.to_pylist()[row_start:row_end]

groups = defaultdict(list)
for r in rows:
    key = (r["codec"], r["q"], r["knob_tuple_json"])
    basename = os.path.basename(r["image_path"])
    groups[key].append((r["image_path"], basename))

manifest_path = os.path.join(work_dir, "_groups.tsv")
with open(manifest_path, "w") as f:
    f.write("group_id\tcodec\tq\tknob_tuple_json\tbasename\timage_path\n")
    for gid, ((codec, q, kj), members) in enumerate(sorted(groups.items())):
        for image_path, basename in members:
            f.write(f"g{gid:04d}\t{codec}\t{q}\t{kj}\t{basename}\t{image_path}\n")

print(f"  {len(rows)} rows -> {len(groups)} groups", file=sys.stderr)
PYEOF

echo "$LOG_PREFIX step 4/6: per-group sweep" >&2
GROUP_LINES=$(awk -F'\t' 'NR>1 {print $1"|"$2"|"$3"|"$4}' "$WORK_DIR/_groups.tsv" | sort -u)
> "$WORK_DIR/pairs.tsv"
G_IDX=0
while IFS='|' read -r gid codec q kj; do
    G_IDX=$((G_IDX + 1))
    GROUP_DIR="$WORK_DIR/g${gid}"
    GROUP_SRC="$GROUP_DIR/sources"
    GROUP_DIST="$GROUP_DIR/dist"
    mkdir -p "$GROUP_SRC" "$GROUP_DIST"
    awk -F'\t' -v g="$gid" 'NR>1 && $1==g {print $5}' "$WORK_DIR/_groups.tsv" | sort -u | while read -r b; do
        ln -sf "$WORK_DIR/sources/$b" "$GROUP_SRC/$b" 2>/dev/null || true
    done

    # Encode-only sweep — pick a cheap metric (ssim2 default) just to
    # satisfy `zenmetrics sweep`'s metric arg. The actual scoring
    # happens in step 5 below via score-pairs. Override with
    # --sweep-metric-for-encode if some future metric needs special
    # encode params at the sweep stage.
    SWEEP_ARGS=(
        sweep
        --codec "$codec"
        --sources "$GROUP_SRC"
        --q-grid "$q"
        --output "$GROUP_DIR/sweep.tsv"
        --pairs-tsv "$GROUP_DIR/pairs.tsv"
        --distorted-out-dir "$GROUP_DIST"
        --metric "$SWEEP_METRIC_FOR_ENCODE"
        --gpu-runtime "$GPU_RUNTIME"
    )
    if [[ "$kj" != "{}" && -n "$kj" ]]; then
        KNOB_GRID=$(echo "$kj" | jq -c 'with_entries(.value |= [.])')
        SWEEP_ARGS+=(--knob-grid "$KNOB_GRID")
    fi

    set +e
    if [[ -n "$ZEN_METRICS_IMAGE" ]]; then
        docker run --rm $DOCKER_GPUS \
            --entrypoint /usr/local/bin/zenmetrics \
            -v "$WORK_DIR":"$WORK_DIR":rw \
            -w "$GROUP_DIR" \
            "$ZEN_METRICS_IMAGE" \
            "${SWEEP_ARGS[@]}" > "$GROUP_DIR/sweep.stderr.log" 2>&1
        sweep_rc=$?
    else
        zenmetrics "${SWEEP_ARGS[@]}" > "$GROUP_DIR/sweep.stderr.log" 2>&1
        sweep_rc=$?
    fi
    set -e
    sed "s/^/  [sweep g${gid}] /" < "$GROUP_DIR/sweep.stderr.log" >&2
    if (( sweep_rc != 0 )); then
        echo "$LOG_PREFIX step 4/6: sweep g${gid} FAILED rc=$sweep_rc codec=$codec q=$q" >&2
        echo "  knob_tuple_json: $kj" >&2
        echo "  ls $GROUP_SRC:" >&2
        ls -la "$GROUP_SRC" 2>&1 | sed 's/^/    /' >&2
        exit 1
    fi

    if [[ "$G_IDX" == "1" ]]; then
        cat "$GROUP_DIR/pairs.tsv" >> "$WORK_DIR/pairs.tsv"
    else
        tail -n +2 "$GROUP_DIR/pairs.tsv" >> "$WORK_DIR/pairs.tsv"
    fi
done <<< "$GROUP_LINES"

SIDECAR="$WORK_DIR/out/${METRIC}_sidecar.parquet"

echo "$LOG_PREFIX step 5/6: score-pairs $METRIC" >&2

SCORE_ARGS=(
    score-pairs
    --metric "$METRIC"
    --pairs-tsv "$WORK_DIR/pairs.tsv"
    --out-parquet "$SIDECAR"
    --gpu-runtime "$GPU_RUNTIME"
)
if [[ "$FAIL_ON_BOGUS" == "1" ]]; then
    SCORE_ARGS+=(--fail-on-bogus)
fi
# iwssim's --allow-small-images is a per-metric concern; allow callers
# to inject extra flags via EXTRA_SCORE_PAIRS_ARGS. The launcher / onstart
# script sets this when needed (e.g. IWSSIM_ALLOW_SMALL=1).
if [[ -n "$EXTRA_SCORE_PAIRS_ARGS" ]]; then
    # shellcheck disable=SC2206  # intentional word-split — caller's contract.
    SCORE_ARGS+=( $EXTRA_SCORE_PAIRS_ARGS )
fi

set +e
if [[ -n "$ZEN_METRICS_IMAGE" ]]; then
    docker run --rm $DOCKER_GPUS \
        --entrypoint /usr/local/bin/zenmetrics \
        -v "$WORK_DIR":"$WORK_DIR":rw \
        -w "$WORK_DIR" \
        "$ZEN_METRICS_IMAGE" \
        "${SCORE_ARGS[@]}" 2>&1 | tee "$WORK_DIR/score.log" | sed "s/^/  [${METRIC}] /" >&2
    score_rc=${PIPESTATUS[0]}
else
    zenmetrics "${SCORE_ARGS[@]}" 2>&1 | tee "$WORK_DIR/score.log" | sed "s/^/  [${METRIC}] /" >&2
    score_rc=${PIPESTATUS[0]}
fi
set -e

# rc=2 means --fail-on-bogus tripped the sanity check. Upload the
# stderr/score log to a failure path so the orchestrator can see why
# the chunk was rejected without re-running it.
if (( score_rc == 2 )); then
    FAIL_LOG_R2="s3://zentrain/${RUN_ID}/failures/${CHUNK_ID}.log"
    echo "$LOG_PREFIX step 6/6: BOGUS — uploading failure log to $FAIL_LOG_R2" >&2
    {
        echo "# metric_backfill_chunk_worker failure log"
        echo "# chunk_id: $CHUNK_ID"
        echo "# metric: $METRIC"
        echo "# run_id: $RUN_ID"
        echo "# timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# score-pairs exit code: $score_rc"
        echo "# sidecar path (NOT uploaded): $SIDECAR"
        echo "#"
        echo "# === chunk JSON ==="
        echo "$CHUNK_JSON" | jq -c .
        echo "#"
        echo "# === score-pairs stderr ==="
        cat "$WORK_DIR/score.log"
    } > "$WORK_DIR/failure.log"
    if [[ "$SKIP_UPLOAD" != "1" ]]; then
        R2 cp "$WORK_DIR/failure.log" "$FAIL_LOG_R2" >&2 || {
            echo "WARN: failed to upload failure log; keeping locally at $WORK_DIR/failure.log" >&2
            KEEP_WORK=1
        }
    fi
    exit 2
fi

if (( score_rc != 0 )); then
    echo "$LOG_PREFIX step 5/6: score-pairs FAILED rc=$score_rc" >&2
    exit 1
fi

if [[ "$SKIP_UPLOAD" != "1" ]]; then
    echo "$LOG_PREFIX step 6/6: upload sidecar" >&2
    R2 cp "$SIDECAR" "$OUT_SIDECAR_R2" >&2
else
    echo "$LOG_PREFIX step 6/6: SKIPPED upload" >&2
    echo "  ${METRIC} sidecar at: $SIDECAR" >&2
fi

echo "$LOG_PREFIX done" >&2
