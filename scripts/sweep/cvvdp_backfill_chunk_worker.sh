#!/usr/bin/env bash
#
# cvvdp_backfill_chunk_worker.sh — process one chunk of the
# cvvdp-backfill fleet (PINNED TASK).
#
# Companion to scripts/sweep/generate_cvvdp_backfill_chunks.py, which
# emits the chunks.jsonl manifest this worker consumes one line at a
# time. The chunk format is documented at the top of that file.
#
# The worker:
#   1. Reads one chunk-manifest line (JSON object) via --chunk-json or stdin.
#   2. Downloads the input parquet from R2 into a scratch dir.
#   3. Syncs the chunk's image_basenames from R2's source_dir_r2.
#   4. Reads rows[row_range[0]:row_range[1]] from the parquet, groups by
#      (codec, q, knob_tuple_json), and re-encodes the dist images via
#      `zen-metrics sweep` once per group.
#   5. Runs `zen-metrics score-pairs --metric cvvdp` to produce the
#      cvvdp_imazen sidecar, then `pycvvdp-worker score-pairs` for the
#      cvvdp_pycvvdp_v054 sidecar. Both wrap the host-installed
#      binaries by default; pass --zen-metrics-image / --pycvvdp-image
#      to run them inside docker (matches dual_impl_chunk_docker.sh).
#   6. Uploads both sidecars to out_sidecar_imazen / out_sidecar_pycvvdp
#      from the chunk manifest.
#
# Required tools on PATH:
#   - s5cmd (R2 transfers)
#   - jq    (chunk JSON parsing)
#   - python3 with pyarrow (parquet slicing)
#   - docker (if running scorers in containers)
#   OR host-installed zen-metrics + pycvvdp venv (if running directly)
#
# Required env vars (R2 credentials, same as onstart_v3.sh):
#   R2_ACCOUNT_ID  R2_ACCESS_KEY_ID  R2_SECRET_ACCESS_KEY
#
# Usage:
#
#   echo '<one chunk JSON line>' | \
#       cvvdp_backfill_chunk_worker.sh \
#           --zen-metrics-image ghcr.io/imazen/zen-metrics-sweep:0.6.4-aba984c \
#           --pycvvdp-image ghcr.io/imazen/pycvvdp-scorer:0.5.4
#
# OR:
#
#   cvvdp_backfill_chunk_worker.sh \
#       --chunk-json "$(head -1 chunks.jsonl)" \
#       --work-dir /tmp/cvvdp-chunk
#
# Defaults to host binaries when no --*-image flag is passed.

set -euo pipefail

CHUNK_JSON="${CHUNK_JSON:-}"
WORK_DIR="${WORK_DIR:-/tmp/cvvdp-chunk-$$}"
ZEN_METRICS_IMAGE="${ZEN_METRICS_IMAGE:-}"
PYCVVDP_IMAGE="${PYCVVDP_IMAGE:-}"
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
DOCKER_GPUS="${DOCKER_GPUS:---gpus all}"
KEEP_WORK="${KEEP_WORK:-0}"
SKIP_IMAZEN="${SKIP_IMAZEN:-0}"
SKIP_PYCVVDP="${SKIP_PYCVVDP:-0}"
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
        --pycvvdp-image) PYCVVDP_IMAGE="$2"; shift 2;;
        --gpu-runtime) GPU_RUNTIME="$2"; shift 2;;
        --keep-work) KEEP_WORK=1; shift;;
        --skip-imazen) SKIP_IMAZEN=1; shift;;
        --skip-pycvvdp) SKIP_PYCVVDP=1; shift;;
        --skip-upload) SKIP_UPLOAD=1; shift;;
        *) echo "unknown arg: $1" >&2; usage 1;;
    esac
done

# Read chunk JSON from stdin if not on the CLI.
if [[ -z "$CHUNK_JSON" ]]; then
    CHUNK_JSON="$(cat)"
fi
if [[ -z "$CHUNK_JSON" ]]; then
    echo "ERROR: no chunk JSON (pass --chunk-json or pipe to stdin)" >&2
    exit 1
fi

# Sanity-check required tools.
for tool in jq python3 s5cmd; do
    command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done

# Parse chunk fields.
CHUNK_ID=$(echo "$CHUNK_JSON" | jq -r '.chunk_id')
INPUT_PARQUET=$(echo "$CHUNK_JSON" | jq -r '.input_parquet')
INPUT_PARQUET_R2=$(echo "$CHUNK_JSON" | jq -r '.input_parquet_r2')
ROW_START=$(echo "$CHUNK_JSON" | jq -r '.row_range[0]')
ROW_END=$(echo "$CHUNK_JSON" | jq -r '.row_range[1]')
SOURCE_DIR_R2=$(echo "$CHUNK_JSON" | jq -r '.source_dir_r2')
OUT_IMAZEN_R2=$(echo "$CHUNK_JSON" | jq -r '.out_sidecar_imazen')
OUT_PYCVVDP_R2=$(echo "$CHUNK_JSON" | jq -r '.out_sidecar_pycvvdp')

if [[ -z "$CHUNK_ID" || "$CHUNK_ID" == "null" ]]; then
    echo "ERROR: chunk_id missing from JSON" >&2
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

# ── Step 1: pull the input parquet ────────────────────────────────────
echo "[chunk-worker $CHUNK_ID] step 1/6: download input parquet" >&2
s5cmd cp "$INPUT_PARQUET_R2" "$WORK_DIR/$INPUT_PARQUET" >&2

# ── Step 2: sync this chunk's source basenames ────────────────────────
echo "[chunk-worker $CHUNK_ID] step 2/6: sync source basenames" >&2
N_BASENAMES=$(echo "$CHUNK_JSON" | jq -r '.image_basenames | length')
echo "  $N_BASENAMES unique basenames" >&2
# s5cmd does batched downloads via a run-file. One basename per line.
echo "$CHUNK_JSON" | jq -r --arg src "$SOURCE_DIR_R2" '
    .image_basenames[] |
    "cp \($src)/\(.) \(.)"
' > "$WORK_DIR/sources/_download.run"
( cd "$WORK_DIR/sources" && s5cmd run "$WORK_DIR/sources/_download.run" >&2 ) || {
    echo "ERROR: failed to sync sources" >&2
    exit 2
}
rm -f "$WORK_DIR/sources/_download.run"

# ── Step 3: slice the parquet, group rows by (codec, q, knob_tuple) ───
echo "[chunk-worker $CHUNK_ID] step 3/6: slice + group" >&2
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

# Group rows by (codec, q, knob_tuple_json) — each group becomes one
# `zen-metrics sweep` invocation. We don't grid-expand q here because
# the parquet's identity tuples are explicit, not Cartesian.
groups = defaultdict(list)
for r in rows:
    key = (r["codec"], r["q"], r["knob_tuple_json"])
    basename = os.path.basename(r["image_path"])
    groups[key].append((r["image_path"], basename))

# Emit one TSV per group: basename, image_path. The worker
# materialises a per-group sources dir (symlinks) and runs sweep
# against it.
manifest_path = os.path.join(work_dir, "_groups.tsv")
with open(manifest_path, "w") as f:
    f.write("group_id\tcodec\tq\tknob_tuple_json\tbasename\timage_path\n")
    for gid, ((codec, q, kj), members) in enumerate(sorted(groups.items())):
        for image_path, basename in members:
            f.write(f"g{gid:04d}\t{codec}\t{q}\t{kj}\t{basename}\t{image_path}\n")

print(f"  {len(rows)} rows -> {len(groups)} groups", file=sys.stderr)
PYEOF

# ── Step 4: per-group sweep to produce dist images + pairs.tsv ────────
echo "[chunk-worker $CHUNK_ID] step 4/6: per-group sweep" >&2
GROUPS=$(awk -F'\t' 'NR>1 {print $1"|"$2"|"$3"|"$4}' "$WORK_DIR/_groups.tsv" | sort -u)
> "$WORK_DIR/pairs.tsv"
G_IDX=0
while IFS='|' read -r gid codec q kj; do
    G_IDX=$((G_IDX + 1))
    GROUP_DIR="$WORK_DIR/g${gid}"
    GROUP_SRC="$GROUP_DIR/sources"
    GROUP_DIST="$GROUP_DIR/dist"
    mkdir -p "$GROUP_SRC" "$GROUP_DIST"
    # Symlink the basenames this group references into a per-group
    # sources dir. zen-metrics sweep walks the dir, scoring every file.
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
        # knob_tuple_json is a single tuple like `{"effort":1,"subsampling":"422"}`.
        # zen-metrics sweep --knob-grid expects `{axis: [values]}` cartesian
        # form. Wrap each value in a single-element list.
        KNOB_GRID=$(echo "$kj" | jq -c 'with_entries(.value |= [.])')
        SWEEP_ARGS+=(--knob-grid "$KNOB_GRID")
    fi

    if [[ -n "$ZEN_METRICS_IMAGE" ]]; then
        docker run --rm $DOCKER_GPUS \
            -v "$WORK_DIR":"$WORK_DIR":rw \
            -w "$GROUP_DIR" \
            "$ZEN_METRICS_IMAGE" \
            zen-metrics "${SWEEP_ARGS[@]}" 2>&1 | sed "s/^/  [sweep g${gid}] /" >&2
    else
        zen-metrics "${SWEEP_ARGS[@]}" 2>&1 | sed "s/^/  [sweep g${gid}] /" >&2
    fi

    # Concatenate each group's pairs.tsv into the chunk-level one.
    # Keep the header from group 0 only.
    if [[ "$G_IDX" == "1" ]]; then
        cat "$GROUP_DIR/pairs.tsv" >> "$WORK_DIR/pairs.tsv"
    else
        tail -n +2 "$GROUP_DIR/pairs.tsv" >> "$WORK_DIR/pairs.tsv"
    fi
done <<< "$GROUPS"

# ── Step 5: score-pairs both implementations ──────────────────────────
SIDECAR_IMAZEN_NAME="${SIDECAR_IMAZEN_NAME:-cvvdp_imazen_v0_0_1}"
SIDECAR_IMAZEN="$WORK_DIR/out/${SIDECAR_IMAZEN_NAME}.parquet"
SIDECAR_PYCVVDP="$WORK_DIR/out/cvvdp_pycvvdp_v054.parquet"

if [[ "$SKIP_IMAZEN" != "1" ]]; then
    echo "[chunk-worker $CHUNK_ID] step 5a/6: score-pairs cvvdp (imazen)" >&2
    if [[ -n "$ZEN_METRICS_IMAGE" ]]; then
        docker run --rm $DOCKER_GPUS \
            -v "$WORK_DIR":"$WORK_DIR":rw \
            -w "$WORK_DIR" \
            "$ZEN_METRICS_IMAGE" \
            zen-metrics score-pairs \
                --metric cvvdp \
                --pairs-tsv "$WORK_DIR/pairs.tsv" \
                --out-parquet "$SIDECAR_IMAZEN" \
                --gpu-runtime "$GPU_RUNTIME" 2>&1 | sed 's/^/  [imazen] /' >&2
    else
        zen-metrics score-pairs \
            --metric cvvdp \
            --pairs-tsv "$WORK_DIR/pairs.tsv" \
            --out-parquet "$SIDECAR_IMAZEN" \
            --gpu-runtime "$GPU_RUNTIME" 2>&1 | sed 's/^/  [imazen] /' >&2
    fi
fi

if [[ "$SKIP_PYCVVDP" != "1" ]]; then
    echo "[chunk-worker $CHUNK_ID] step 5b/6: pycvvdp-worker score-pairs" >&2
    if [[ -n "$PYCVVDP_IMAGE" ]]; then
        docker run --rm $DOCKER_GPUS \
            -v "$WORK_DIR":"$WORK_DIR":rw \
            -w "$WORK_DIR" \
            "$PYCVVDP_IMAGE" \
            pycvvdp-worker score-pairs \
                --pairs-tsv "$WORK_DIR/pairs.tsv" \
                --out-parquet "$SIDECAR_PYCVVDP" 2>&1 | sed 's/^/  [pycvvdp] /' >&2
    else
        : "${PYCVVDP_PYTHON:?need PYCVVDP_PYTHON when not using --pycvvdp-image}"
        : "${PYCVVDP_WORKER:?need PYCVVDP_WORKER when not using --pycvvdp-image}"
        "$PYCVVDP_PYTHON" "$PYCVVDP_WORKER" score-pairs \
            --pairs-tsv "$WORK_DIR/pairs.tsv" \
            --out-parquet "$SIDECAR_PYCVVDP" 2>&1 | sed 's/^/  [pycvvdp] /' >&2
    fi
fi

# ── Step 6: upload sidecars to R2 ─────────────────────────────────────
if [[ "$SKIP_UPLOAD" != "1" ]]; then
    echo "[chunk-worker $CHUNK_ID] step 6/6: upload sidecars" >&2
    [[ "$SKIP_IMAZEN" != "1" ]] && s5cmd cp "$SIDECAR_IMAZEN" "$OUT_IMAZEN_R2" >&2
    [[ "$SKIP_PYCVVDP" != "1" ]] && s5cmd cp "$SIDECAR_PYCVVDP" "$OUT_PYCVVDP_R2" >&2
else
    echo "[chunk-worker $CHUNK_ID] step 6/6: SKIPPED upload" >&2
    [[ "$SKIP_IMAZEN" != "1" ]] && echo "  imazen sidecar at: $SIDECAR_IMAZEN" >&2
    [[ "$SKIP_PYCVVDP" != "1" ]] && echo "  pycvvdp sidecar at: $SIDECAR_PYCVVDP" >&2
fi

echo "[chunk-worker $CHUNK_ID] done" >&2
