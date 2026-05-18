#!/usr/bin/env bash
#
# iwssim_backfill_chunk_worker.sh — process one chunk of the
# iwssim-backfill fleet (V_22-mix retrain dependency).
#
# Adapted from cvvdp_backfill_chunk_worker.sh — same chunk format,
# same R2 layout, just one scorer (iwssim) instead of two (cvvdp +
# pycvvdp). Output sidecar path: `out_sidecar_iwssim` (NOT
# `out_sidecar_imazen` from the cvvdp chunks — the new chunks.jsonl
# uses a dedicated field).
#
# The worker:
#   1. Reads one chunk-manifest line (JSON object) via --chunk-json or stdin.
#   2. Downloads the input parquet from R2 into a scratch dir.
#   3. Syncs the chunk's image_basenames from R2's source_dir_r2.
#   4. Reads rows[row_range[0]:row_range[1]] from the parquet, groups by
#      (codec, q, knob_tuple_json), and re-encodes the dist images via
#      `zen-metrics sweep` once per group.
#   5. Runs `zen-metrics score-pairs --metric iwssim` to produce the
#      iwssim sidecar.
#   6. Uploads sidecar to out_sidecar_iwssim from the chunk manifest.
#
# Required tools on PATH (or docker image with same baked in):
#   - s5cmd (R2 transfers)
#   - jq    (chunk JSON parsing)
#   - python3 with pyarrow (parquet slicing)
#   - docker (if running scorer in a container)
#
# Required env vars (R2 credentials, same as onstart_v3.sh):
#   R2_ACCOUNT_ID  R2_ACCESS_KEY_ID  R2_SECRET_ACCESS_KEY
#
# Usage:
#
#   echo '<one chunk JSON line>' | \
#       iwssim_backfill_chunk_worker.sh \
#           --zen-metrics-image ghcr.io/imazen/zen-metrics-sweep:0.6.4-iwssim-<sha>
#
# OR:
#
#   iwssim_backfill_chunk_worker.sh \
#       --chunk-json "$(head -1 chunks.jsonl)" \
#       --work-dir /tmp/iwssim-chunk

set -euo pipefail

CHUNK_JSON="${CHUNK_JSON:-}"
WORK_DIR="${WORK_DIR:-/tmp/iwssim-chunk-$$}"
ZEN_METRICS_IMAGE="${ZEN_METRICS_IMAGE:-}"
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
DOCKER_GPUS="${DOCKER_GPUS:---gpus all}"
KEEP_WORK="${KEEP_WORK:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

[[ $# -gt 0 && "$1" == "-h" || "${1:-}" == "--help" ]] && usage 0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunk-json) CHUNK_JSON="$2"; shift 2;;
        --work-dir) WORK_DIR="$2"; shift 2;;
        --zen-metrics-image) ZEN_METRICS_IMAGE="$2"; shift 2;;
        --gpu-runtime) GPU_RUNTIME="$2"; shift 2;;
        --keep-work) KEEP_WORK=1; shift;;
        --skip-upload) SKIP_UPLOAD=1; shift;;
        *) echo "unknown arg: $1" >&2; usage 1;;
    esac
done

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
OUT_IWSSIM_R2=$(echo "$CHUNK_JSON" | jq -r '.out_sidecar_iwssim')

if [[ -z "$CHUNK_ID" || "$CHUNK_ID" == "null" ]]; then
    echo "ERROR: chunk_id missing from JSON" >&2
    exit 1
fi
if [[ -z "$OUT_IWSSIM_R2" || "$OUT_IWSSIM_R2" == "null" ]]; then
    echo "ERROR: out_sidecar_iwssim missing from JSON" >&2
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

echo "[iwssim-chunk-worker $CHUNK_ID] step 1/6: download input parquet" >&2
R2 cp "$INPUT_PARQUET_R2" "$WORK_DIR/$INPUT_PARQUET" >&2

echo "[iwssim-chunk-worker $CHUNK_ID] step 2/6: sync source basenames" >&2
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

echo "[iwssim-chunk-worker $CHUNK_ID] step 3/6: slice + group" >&2
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

echo "[iwssim-chunk-worker $CHUNK_ID] step 4/6: per-group sweep" >&2
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

    SWEEP_ARGS=(
        sweep
        --codec "$codec"
        --sources "$GROUP_SRC"
        --q-grid "$q"
        --output "$GROUP_DIR/sweep.tsv"
        --pairs-tsv "$GROUP_DIR/pairs.tsv"
        --distorted-out-dir "$GROUP_DIST"
        --metric ssim2
        --gpu-runtime "$GPU_RUNTIME"
    )
    if [[ "$kj" != "{}" && -n "$kj" ]]; then
        KNOB_GRID=$(echo "$kj" | jq -c 'with_entries(.value |= [.])')
        SWEEP_ARGS+=(--knob-grid "$KNOB_GRID")
    fi

    set +e
    if [[ -n "$ZEN_METRICS_IMAGE" ]]; then
        docker run --rm $DOCKER_GPUS \
            --entrypoint /usr/local/bin/zen-metrics \
            -v "$WORK_DIR":"$WORK_DIR":rw \
            -w "$GROUP_DIR" \
            "$ZEN_METRICS_IMAGE" \
            "${SWEEP_ARGS[@]}" > "$GROUP_DIR/sweep.stderr.log" 2>&1
        sweep_rc=$?
    else
        zen-metrics "${SWEEP_ARGS[@]}" > "$GROUP_DIR/sweep.stderr.log" 2>&1
        sweep_rc=$?
    fi
    set -e
    sed "s/^/  [sweep g${gid}] /" < "$GROUP_DIR/sweep.stderr.log" >&2
    if (( sweep_rc != 0 )); then
        echo "[iwssim-chunk-worker $CHUNK_ID] step 4/6: sweep g${gid} FAILED rc=$sweep_rc codec=$codec q=$q" >&2
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

SIDECAR_IWSSIM="$WORK_DIR/out/iwssim_imazen_v0_0_1.parquet"

echo "[iwssim-chunk-worker $CHUNK_ID] step 5/6: score-pairs iwssim" >&2
# IWSSIM_ALLOW_SMALL toggles `--allow-small-images` so sub-176-px pairs
# go through the reflect-pad adaptive path instead of returning NaN.
# Default ON for backfill — the 754-chunk filtered set already excludes
# the very-small-image chunks (image_basenames pre-screened), but for
# safety we keep adaptive mode on so future relaxations don't trip on
# the size guard. Stock-size inputs (≥176 on both axes) are unaffected.
IWSSIM_EXTRA_ARGS=()
if [[ "${IWSSIM_ALLOW_SMALL:-1}" == "1" ]]; then
    IWSSIM_EXTRA_ARGS+=(--allow-small-images)
fi
if [[ -n "$ZEN_METRICS_IMAGE" ]]; then
    docker run --rm $DOCKER_GPUS \
        --entrypoint /usr/local/bin/zen-metrics \
        -v "$WORK_DIR":"$WORK_DIR":rw \
        -w "$WORK_DIR" \
        "$ZEN_METRICS_IMAGE" \
        score-pairs \
            --metric iwssim \
            --pairs-tsv "$WORK_DIR/pairs.tsv" \
            --out-parquet "$SIDECAR_IWSSIM" \
            --gpu-runtime "$GPU_RUNTIME" \
            "${IWSSIM_EXTRA_ARGS[@]}" 2>&1 | sed 's/^/  [iwssim] /' >&2
else
    zen-metrics score-pairs \
        --metric iwssim \
        --pairs-tsv "$WORK_DIR/pairs.tsv" \
        --out-parquet "$SIDECAR_IWSSIM" \
        --gpu-runtime "$GPU_RUNTIME" \
        "${IWSSIM_EXTRA_ARGS[@]}" 2>&1 | sed 's/^/  [iwssim] /' >&2
fi

if [[ "$SKIP_UPLOAD" != "1" ]]; then
    echo "[iwssim-chunk-worker $CHUNK_ID] step 6/6: upload sidecar" >&2
    R2 cp "$SIDECAR_IWSSIM" "$OUT_IWSSIM_R2" >&2
else
    echo "[iwssim-chunk-worker $CHUNK_ID] step 6/6: SKIPPED upload" >&2
    echo "  iwssim sidecar at: $SIDECAR_IWSSIM" >&2
fi

echo "[iwssim-chunk-worker $CHUNK_ID] done" >&2
