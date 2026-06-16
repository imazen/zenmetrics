#!/usr/bin/env bash
#
# dual_impl_chunk.sh — run a single sweep chunk and score it with
# BOTH cvvdp implementations (cvvdp_imazen + cvvdp_pycvvdp_v054),
# producing two parquet sidecars joinable by the identity tuple.
#
# This is the per-chunk shape that the vast.ai dispatcher orchestrates;
# running it locally on a small sample is the verification gate
# before turning on the production fleet for the PINNED TASK at
# repo-root CLAUDE.md.
#
# Required tools on PATH:
#   - zenmetrics (cargo install or pre-built binary; built with
#     --features sweep,gpu-cvvdp,gpu-cuda for the cvvdp-gpu path)
#   - python3 in a venv with pycvvdp + pyarrow + numpy + pillow
#     (see scripts/cvvdp_goldens/requirements.txt)
#
# Required env vars (or pass via flags):
#   CODEC               — zenjpeg | zenwebp | zenavif | zenjxl | zenpng
#   SOURCES_DIR         — directory of reference images
#   Q_GRID              — comma-separated qualities (e.g. "50,90")
#   KNOB_GRID           — JSON object knob grid, or "" for none
#   OUT_DIR             — output directory (sidecars + pairs land here)
#   PYCVVDP_PYTHON      — path to python3 with pycvvdp installed
#                         (default: scripts/cvvdp_goldens/.venv/bin/python3
#                         relative to repo root)
#   PYCVVDP_WORKER      — path to pycvvdp_worker.py (default:
#                         scripts/sweep/pycvvdp_worker.py)
#   GPU_RUNTIME         — auto | cuda | wgpu | cpu (default: auto)
#   SKIP_IMAZEN         — set to 1 to skip the cvvdp-gpu run
#   SKIP_PYCVVDP        — set to 1 to skip the pycvvdp run
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
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
SKIP_IMAZEN="${SKIP_IMAZEN:-0}"
SKIP_PYCVVDP="${SKIP_PYCVVDP:-0}"

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
        --gpu-runtime) GPU_RUNTIME="$2"; shift 2;;
        --skip-imazen) SKIP_IMAZEN=1; shift;;
        --skip-pycvvdp) SKIP_PYCVVDP=1; shift;;
        *) echo "unknown arg: $1" >&2; usage 1;;
    esac
done

: "${CODEC:?CODEC is required}"
: "${SOURCES_DIR:?SOURCES_DIR is required}"
: "${Q_GRID:?Q_GRID is required}"
: "${OUT_DIR:?OUT_DIR is required}"

# Resolve repo-relative defaults for the pycvvdp side. The script can
# live anywhere; SCRIPT_DIR + REPO_ROOT lets us find sibling tools
# without hard-coding paths.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)"
PYCVVDP_PYTHON="${PYCVVDP_PYTHON:-$REPO_ROOT/scripts/cvvdp_goldens/.venv/bin/python3}"
PYCVVDP_WORKER="${PYCVVDP_WORKER:-$REPO_ROOT/scripts/sweep/pycvvdp_worker.py}"

mkdir -p "$OUT_DIR" "$OUT_DIR/dist"

SWEEP_TSV="$OUT_DIR/sweep.tsv"
PAIRS_TSV="$OUT_DIR/pairs.tsv"
DIST_DIR="$OUT_DIR/dist"

echo "[dual-impl] step 1/4: sweep $CODEC over $SOURCES_DIR (q=$Q_GRID)" >&2
SWEEP_ARGS=(
    sweep
    --codec "$CODEC"
    --sources "$SOURCES_DIR"
    --q-grid "$Q_GRID"
    --output "$SWEEP_TSV"
    --pairs-tsv "$PAIRS_TSV"
    --distorted-out-dir "$DIST_DIR"
    --gpu-runtime "$GPU_RUNTIME"
)
# Sweep needs a metric — use ssim2 as the cheap default since real
# scoring happens in the score-pairs stage below.
SWEEP_ARGS+=(--metric ssim2)
[[ -n "$KNOB_GRID" ]] && SWEEP_ARGS+=(--knob-grid "$KNOB_GRID")
zenmetrics "${SWEEP_ARGS[@]}" 2>&1 | sed 's/^/  [sweep] /'

# Locate the cvvdp_imazen column tag from a dry score-pairs call.
# zenmetrics doesn't expose the column name directly via the CLI, so
# we hard-code the default form (`cvvdp_imazen_v0_0_1`) and let the
# parquet header confirm. If the column name diverges from the
# default, override SIDECAR_IMAZEN_NAME via env.
SIDECAR_IMAZEN_NAME="${SIDECAR_IMAZEN_NAME:-cvvdp_imazen_v0_0_1}"
SIDECAR_IMAZEN="$OUT_DIR/${SIDECAR_IMAZEN_NAME}.parquet"
SIDECAR_PYCVVDP="$OUT_DIR/cvvdp_pycvvdp_v054.parquet"

if [[ "$SKIP_IMAZEN" != "1" ]]; then
    echo "[dual-impl] step 2/4: score-pairs cvvdp (imazen, GPU)" >&2
    zenmetrics score-pairs \
        --metric cvvdp \
        --pairs-tsv "$PAIRS_TSV" \
        --out-parquet "$SIDECAR_IMAZEN" \
        --gpu-runtime "$GPU_RUNTIME" \
        2>&1 | sed 's/^/  [imazen] /'
else
    echo "[dual-impl] step 2/4: SKIPPED (cvvdp-gpu)" >&2
fi

if [[ "$SKIP_PYCVVDP" != "1" ]]; then
    echo "[dual-impl] step 3/4: pycvvdp_worker.py score-pairs (v0.5.4)" >&2
    if [[ ! -x "$PYCVVDP_PYTHON" ]]; then
        echo "  pycvvdp python not found at $PYCVVDP_PYTHON" >&2
        echo "  override via PYCVVDP_PYTHON or run with --skip-pycvvdp" >&2
        exit 2
    fi
    "$PYCVVDP_PYTHON" "$PYCVVDP_WORKER" score-pairs \
        --pairs-tsv "$PAIRS_TSV" \
        --out-parquet "$SIDECAR_PYCVVDP" \
        2>&1 | sed 's/^/  [pycvvdp] /'
else
    echo "[dual-impl] step 3/4: SKIPPED (pycvvdp)" >&2
fi

# Side-by-side parity stats when both ran.
if [[ "$SKIP_IMAZEN" != "1" && "$SKIP_PYCVVDP" != "1" ]]; then
    echo "[dual-impl] step 4/4: parity table" >&2
    "$PYCVVDP_PYTHON" - "$SIDECAR_IMAZEN" "$SIDECAR_PYCVVDP" "$OUT_DIR/parity.tsv" \
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
    echo "[dual-impl] step 4/4: SKIPPED (parity table needs both impls)" >&2
fi

echo "[dual-impl] done. Outputs under: $OUT_DIR" >&2
