#!/usr/bin/env bash
#
# dual_impl_chunk_docker.sh — docker-wrapper variant of
# dual_impl_chunk.sh. Runs the same per-chunk dual-implementation
# sweep using the two pushed docker images instead of host-installed
# binaries, so local repro and remote collaborators get identical
# environments.
#
# Companion to scripts/sweep/dual_impl_chunk.sh, which runs the same
# pipeline against host-installed `zen-metrics` + a python venv. Use
# the host version when iterating on the binaries; use this docker
# wrapper for "I just want to score some pairs" repro or for the
# eventual production fan-out.
#
# Required images (built and pushed by the agent that lands the
# tags into CLAUDE.md's PINNED TASK progress markers):
#   ghcr.io/imazen/zen-metrics-sweep:<tag>    — cvvdp-gpu + sweep
#   ghcr.io/imazen/pycvvdp-scorer:0.5.4       — pycvvdp v0.5.4 reference
#
# Required tools on PATH:
#   - docker (with `--gpus all` support — needs nvidia-container-toolkit)
#
# Required env vars (or pass via flags):
#   CODEC               — zenjpeg | zenwebp | zenavif | zenjxl | zenpng
#   SOURCES_DIR         — directory of reference images (mounted ro)
#   Q_GRID              — comma-separated qualities (e.g. "50,90")
#   KNOB_GRID           — JSON object knob grid, or "" for none
#   OUT_DIR             — output directory (sidecars + pairs land here)
#   ZEN_METRICS_IMAGE   — full ghcr.io/imazen/zen-metrics-sweep:<tag>
#                         (or pull-policy-compatible registry path)
#   PYCVVDP_IMAGE       — full ghcr.io/imazen/pycvvdp-scorer:0.5.4
#                         (default: ghcr.io/imazen/pycvvdp-scorer:0.5.4)
#   SKIP_IMAZEN         — set to 1 to skip the cvvdp-gpu run
#   SKIP_PYCVVDP        — set to 1 to skip the pycvvdp run
#   DOCKER_GPUS         — "--gpus all" (default) or "--gpus '\"device=0\"'"
#
# Outputs (under $OUT_DIR):
#   sweep.tsv                                 — main Pareto TSV
#   pairs.tsv                                 — identity tuple + ref + dist paths
#   dist/                                     — distorted PNGs
#   cvvdp_imazen_<tag>.parquet                — cvvdp-gpu sidecar
#   cvvdp_pycvvdp_v054.parquet                — pycvvdp sidecar
#   parity.tsv                                — joined side-by-side
#                                               (computed when both ran)

set -euo pipefail

CODEC="${CODEC:-}"
SOURCES_DIR="${SOURCES_DIR:-}"
Q_GRID="${Q_GRID:-}"
KNOB_GRID="${KNOB_GRID:-}"
OUT_DIR="${OUT_DIR:-}"
ZEN_METRICS_IMAGE="${ZEN_METRICS_IMAGE:-}"
PYCVVDP_IMAGE="${PYCVVDP_IMAGE:-ghcr.io/imazen/pycvvdp-scorer:0.5.4}"
SKIP_IMAZEN="${SKIP_IMAZEN:-0}"
SKIP_PYCVVDP="${SKIP_PYCVVDP:-0}"
DOCKER_GPUS="${DOCKER_GPUS:---gpus all}"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit "${1:-0}"
}

[[ $# -gt 0 && "$1" == "-h" || "${1:-}" == "--help" ]] && usage 0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --codec) CODEC="$2"; shift 2;;
        --sources) SOURCES_DIR="$2"; shift 2;;
        --q-grid) Q_GRID="$2"; shift 2;;
        --knob-grid) KNOB_GRID="$2"; shift 2;;
        --out-dir) OUT_DIR="$2"; shift 2;;
        --zen-metrics-image) ZEN_METRICS_IMAGE="$2"; shift 2;;
        --pycvvdp-image) PYCVVDP_IMAGE="$2"; shift 2;;
        --skip-imazen) SKIP_IMAZEN=1; shift;;
        --skip-pycvvdp) SKIP_PYCVVDP=1; shift;;
        *) echo "unknown arg: $1" >&2; usage 1;;
    esac
done

: "${CODEC:?CODEC is required}"
: "${SOURCES_DIR:?SOURCES_DIR is required}"
: "${Q_GRID:?Q_GRID is required}"
: "${OUT_DIR:?OUT_DIR is required}"
: "${ZEN_METRICS_IMAGE:?ZEN_METRICS_IMAGE is required (use --zen-metrics-image or env)}"

# Resolve absolute paths up-front so docker volume mounts don't surprise.
SOURCES_ABS="$(cd -- "$SOURCES_DIR" && pwd)"
mkdir -p "$OUT_DIR" "$OUT_DIR/dist"
OUT_ABS="$(cd -- "$OUT_DIR" && pwd)"

# Container-side paths (we mount $SOURCES_ABS at /work/sources:ro and
# $OUT_ABS at /work/out:rw). The sweep + score-pairs commands all run
# with /work as cwd — keeps the pairs.tsv `image_path` column free of
# host-leaking absolute paths.
CTR_SOURCES="/work/sources"
CTR_OUT="/work/out"
CTR_SWEEP_TSV="$CTR_OUT/sweep.tsv"
CTR_PAIRS_TSV="$CTR_OUT/pairs.tsv"
CTR_DIST_DIR="$CTR_OUT/dist"

# Same default name discipline as the host-binary variant. Override
# SIDECAR_IMAZEN_NAME if the in-image zen-metrics has a different
# CVVDP_IMPL_TAG baked in (e.g. CI-tagged images bake the git short
# hash).
SIDECAR_IMAZEN_NAME="${SIDECAR_IMAZEN_NAME:-cvvdp_imazen_v0_0_1}"
SIDECAR_IMAZEN_HOST="$OUT_ABS/${SIDECAR_IMAZEN_NAME}.parquet"
SIDECAR_PYCVVDP_HOST="$OUT_ABS/cvvdp_pycvvdp_v054.parquet"

run_zen_metrics() {
    docker run --rm $DOCKER_GPUS \
        -v "$SOURCES_ABS":"$CTR_SOURCES":ro \
        -v "$OUT_ABS":"$CTR_OUT":rw \
        -w "$CTR_OUT" \
        "$ZEN_METRICS_IMAGE" \
        zen-metrics "$@"
}

run_pycvvdp() {
    docker run --rm $DOCKER_GPUS \
        -v "$SOURCES_ABS":"$CTR_SOURCES":ro \
        -v "$OUT_ABS":"$CTR_OUT":rw \
        -w "$CTR_OUT" \
        "$PYCVVDP_IMAGE" \
        pycvvdp-worker "$@"
}

echo "[dual-impl-docker] step 1/4: sweep $CODEC over $SOURCES_DIR (q=$Q_GRID)" >&2
SWEEP_ARGS=(
    sweep
    --codec "$CODEC"
    --sources "$CTR_SOURCES"
    --q-grid "$Q_GRID"
    --output "$CTR_SWEEP_TSV"
    --pairs-tsv "$CTR_PAIRS_TSV"
    --distorted-out-dir "$CTR_DIST_DIR"
    --metric ssim2
)
[[ -n "$KNOB_GRID" ]] && SWEEP_ARGS+=(--knob-grid "$KNOB_GRID")
run_zen_metrics "${SWEEP_ARGS[@]}" 2>&1 | sed 's/^/  [sweep] /'

if [[ "$SKIP_IMAZEN" != "1" ]]; then
    echo "[dual-impl-docker] step 2/4: score-pairs cvvdp (imazen, GPU)" >&2
    run_zen_metrics score-pairs \
        --metric cvvdp \
        --pairs-tsv "$CTR_PAIRS_TSV" \
        --out-parquet "$CTR_OUT/${SIDECAR_IMAZEN_NAME}.parquet" \
        2>&1 | sed 's/^/  [imazen] /'
else
    echo "[dual-impl-docker] step 2/4: SKIPPED (cvvdp-gpu)" >&2
fi

if [[ "$SKIP_PYCVVDP" != "1" ]]; then
    echo "[dual-impl-docker] step 3/4: pycvvdp-worker score-pairs (v0.5.4)" >&2
    run_pycvvdp score-pairs \
        --pairs-tsv "$CTR_PAIRS_TSV" \
        --out-parquet "$CTR_OUT/cvvdp_pycvvdp_v054.parquet" \
        2>&1 | sed 's/^/  [pycvvdp] /'
else
    echo "[dual-impl-docker] step 3/4: SKIPPED (pycvvdp)" >&2
fi

# Side-by-side parity stats when both ran. Uses python from the
# pycvvdp image (pyarrow is already installed there).
if [[ "$SKIP_IMAZEN" != "1" && "$SKIP_PYCVVDP" != "1" ]]; then
    echo "[dual-impl-docker] step 4/4: parity table" >&2
    docker run --rm \
        -v "$OUT_ABS":"$CTR_OUT":rw \
        -w "$CTR_OUT" \
        --entrypoint python3 \
        "$PYCVVDP_IMAGE" \
        - "$CTR_OUT/${SIDECAR_IMAZEN_NAME}.parquet" \
          "$CTR_OUT/cvvdp_pycvvdp_v054.parquet" \
          "$CTR_OUT/parity.tsv" \
          "$SIDECAR_IMAZEN_NAME" <<'PYEOF' 2>&1 | sed 's/^/  [parity] /'
import csv
import sys
import pyarrow.parquet as pq

(_, imazen_path, pycvvdp_path, parity_path, imazen_col) = sys.argv

imazen = pq.read_table(imazen_path).to_pylist()
pycvvdp = pq.read_table(pycvvdp_path).to_pylist()

key = lambda r: (r["image_path"], r["codec"], r["q"], r["knob_tuple_json"])
imap = {key(r): r[imazen_col] for r in imazen}
pmap = {key(r): r["cvvdp_pycvvdp_v054"] for r in pycvvdp}
joined = []
for k, im in imap.items():
    pv = pmap.get(k)
    if pv is None:
        continue
    joined.append((k, im, pv, im - pv))

joined.sort()
with open(parity_path, "w") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow([
        "image_path", "codec", "q", "knob_tuple_json",
        imazen_col, "cvvdp_pycvvdp_v054", "diff",
    ])
    for k, im, pv, d in joined:
        w.writerow([*k, f"{im:.6f}", f"{pv:.6f}", f"{d:+.6f}"])

diffs = [abs(d) for _, _, _, d in joined]
if not diffs:
    print("no joinable rows — check identity tuples")
    sys.exit(1)
diffs.sort()
n = len(diffs)
print(f"n={n}  |diff|: mean={sum(diffs)/n:.4f}  median={diffs[n//2]:.4f}  max={max(diffs):.4f}")
PYEOF
else
    echo "[dual-impl-docker] step 4/4: SKIPPED (parity table needs both impls)" >&2
fi

echo "[dual-impl-docker] done. Outputs under: $OUT_DIR" >&2
echo "  imazen sidecar: $SIDECAR_IMAZEN_HOST" >&2
echo "  pycvvdp sidecar: $SIDECAR_PYCVVDP_HOST" >&2
