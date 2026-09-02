#!/usr/bin/env bash
# avifdoe_declare.sh — declare the AVIF knob-tuning design-of-experiments runs
# through the CANONICAL plan path, with the two-stage gates the plan registers.
#
# Design + budget + gates: benchmarks/avif_doe_plan_2026-09-01.md
# Knob semantics + hazards:  benchmarks/avif_knob_dossier_2026-09-01.md
# Machinery + prior art:     benchmarks/avif_sweep_permutation_retrofit_2026-09-01.md
#
# This script does NOT launch workers. It produces, for each block:
#   <out>/<run>.plan.json   the flatten audit (every merge + drop recorded)
#   <out>/<run>_cells.jsonl the declare items
#   <out>/<run>_manifest.json
# and prints the dedup accounting (G-DEDUP). Declaring is idempotent — the same
# cell yields the same JobId — so re-running is always safe.
#
# Usage:
#   scripts/jobsys/avifdoe_declare.sh [--out DIR] [--sources DIR] [--dry-run-only]
#   scripts/jobsys/avifdoe_declare.sh --smoke        # 2-image G-DEDUP smoke only
#
# Requires: a zenmetrics built with `--features sweep,avif,avif-svt` (the svt
# strata land in `invalid_skipped` without `avif-svt`, loudly, never silently),
# and zenfleet-ctl. Point ZM_BIN / ZFC_BIN at them to skip the default paths.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZM_BIN="${ZM_BIN:-$REPO/target/release/zenmetrics}"
ZFC_BIN="${ZFC_BIN:-$REPO/target/release/zenfleet-ctl}"
# The BUDGET corpus (1024^2 crops + sub-budget natives) is the default for every
# knob block; the native corpus is only used by the cross-size gate.
SOURCES="${SOURCES:-/mnt/v/output/avif-doe-1024-2026-09-01/sources}"
NATIVE_SOURCES="${NATIVE_SOURCES:-/mnt/v/output/avifsvt-subsample-2026-09-01/sources}"
OUT="${OUT:-$HOME/tmp/avifdoe/declare}"
SMOKE=0
DRY_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --sources) SOURCES="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run-only) DRY_ONLY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# The registered quality ladders (plan doc section 2.3). Every point is a
# member of the control arm's 29-point grid, so a knob cell at q=X is directly
# comparable to the default cell at q=X on the same image with NO interpolation
# and no anchor error. Low-q density is deliberately higher than high-q: on this
# very corpus most of the achieved-quality range lives below q 60.
# ---------------------------------------------------------------------------
Q_LADDER9="5,15,25,35,45,60,76,90,96"
Q_LADDER3="15,45,90"
# The control arm's full grid (q100 merges into q98 on both backends, measured).
Q_FULL29="1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98"

# The 16-image class-stratified subset is the pre-registered FIRST STEP of the
# de-scope ladder (plan doc section 7.3), not the default: A2 now runs on all 32
# at the pixel budget. Frozen here so it can never be chosen after seeing
# results. Rule: largest cluster of each of the 12 content classes, then the 2
# smallest-MP and 2 largest-MP remaining picks.
A2_FALLBACK_IMAGES="1008.scale3000x4000.png 1220.scale3000x4000.png 1420.scale3000x4000.png
1432.scale3000x4000.png 1634.scale3000x4000.png 3006.scale3000x2235.png
6038.scale2479x3230.png 6602.scale3302x4844.png 6604.scale3286x4868.png
7076.scale1024x1024.png 8288.scale375x667.png 8434.scale414x896.png
8446.scale2560x1440.png 9032.scale1024x1536.png 9100.scale1024x1536.png
9954.scale1024x1536.png"

mkdir -p "$OUT"
for b in "$ZM_BIN" "$ZFC_BIN"; do
  [ -x "$b" ] || { echo "missing binary: $b (build it, or set ZM_BIN/ZFC_BIN)" >&2; exit 1; }
done
[ -d "$SOURCES" ] || { echo "missing corpus dir: $SOURCES" >&2; exit 1; }

# The de-scope fallback dir is materialised (symlinks only, no pixel copied) so
# firing step 1 of section 7.3 is one flag, not a scramble.
A2_FALLBACK="$OUT/a2_fallback_sources"
mkdir -p "$A2_FALLBACK"
for f in $A2_FALLBACK_IMAGES; do
  [ -e "$SOURCES/$f" ] || { echo "A2 fallback pick missing from corpus: $f" >&2; exit 1; }
  ln -sf "$SOURCES/$f" "$A2_FALLBACK/$f"
done

# ---------------------------------------------------------------------------
# One block. `--emit-cells-image-path basename` is REQUIRED: image_path is half
# the content-addressed CellId, and fleet workers resolve sources against
# ZEN_CORPUS_PREFIX, not against an absolute local path.
# ---------------------------------------------------------------------------
declare_block() {   # $1 run  $2 plan  $3 q-grid  $4 sources  $5 max-deviations
  local run="$1" plan="$2" qgrid="$3" src="$4" devs="$5"
  echo "=== $run  (plan $plan, q $qgrid, max-deviations $devs)"
  "$ZM_BIN" sweep --codec zenavif --plan "$plan" \
    --sources "$src" --q-grid "$qgrid" \
    --max-deviations "$devs" \
    --dry-run --emit-cells "$OUT/${run}_cells.jsonl" \
    --emit-cells-image-path basename \
    --output "$OUT/$run"
  echo "    cells: $(wc -l < "$OUT/${run}_cells.jsonl")"
  # G-DEDUP: the audit manifest records every merge and every drop. A plan
  # whose duplicates_merged is 0 on a knob grid is a red flag, not a clean
  # bill of health — tune 3 alone should absorb the knobs it forces.
  if command -v python3 >/dev/null; then
    python3 - "$OUT/$run.plan.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
keys = ("cells", "duplicates_merged", "invalid_skipped", "compute_tier_skipped",
        "q_coarsenings", "over_budget", "dropped_axes")
for k in keys:
    v = m.get(k)
    if isinstance(v, list):
        print(f"    {k}: {len(v)}")
    elif v is not None:
        print(f"    {k}: {v}")
PY
  fi
  [ "$DRY_ONLY" = 1 ] && return 0
  "$ZFC_BIN" declare-encodes --cells "$OUT/${run}_cells.jsonl" \
    --out "$OUT/${run}_manifest.json"
}

if [ "$SMOKE" = 1 ]; then
  # Two images, two q points: proves the plan resolves, the ids round-trip and
  # the dedup fires, in seconds and without declaring anything.
  SMOKE_SRC="$OUT/smoke_sources"; mkdir -p "$SMOKE_SRC"
  ln -sf "$SOURCES/8288.scale375x667.png" "$SMOKE_SRC/" 2>/dev/null || true
  ln -sf "$SOURCES/9032.scale1024x1536.png" "$SMOKE_SRC/" 2>/dev/null || true
  DRY_ONLY=1
  declare_block avifdoe-smoke-main     svt_doe_main     "45,90" "$SMOKE_SRC" 1
  declare_block avifdoe-smoke-pairwise svt_doe_pairwise "45"    "$SMOKE_SRC" 2
  declare_block avifdoe-smoke-transfer svt_doe_transfer "45"    "$SMOKE_SRC" 1
  exit 0
fi

# ---------------------------------------------------------------------------
# The blocks. $SOURCES is the BUDGET corpus (1024^2 crops + sub-budget natives,
# built by avifdoe_build_budget_corpus.py) for everything except AG, which is
# the cross-size gate and must run on the NATIVE corpus.
# ---------------------------------------------------------------------------
# A0R — the same-size default arm every knob cell is differenced against.
declare_block avifdoe-svt-a0r-20260901 svt_speed_dense  "$Q_FULL29"  "$SOURCES" 9
# A1  — 17 main-effect arms x ALL 7 effective presets x 9 q x 32 images.
declare_block avifdoe-svt-a1-20260901  svt_doe_main     "$Q_LADDER9" "$SOURCES" 1
# A2  — pairwise interactions at preset 7, all 32 images.
declare_block avifdoe-svt-a2-20260901  svt_doe_pairwise "$Q_LADDER9" "$SOURCES" 2
# AG  — the cross-size transfer gate, on the NATIVE corpus (see NATIVE_SOURCES).
declare_block avifdoe-svt-ag-20260901  svt_doe_transfer "$Q_LADDER3" \
    "${NATIVE_SOURCES:-/mnt/v/output/avifsvt-subsample-2026-09-01/sources}" 1

cat <<EOF

Declared into $OUT. NOT launched.

NOT declared here: A3 (the aom knob arms). aom-rs has no EncoderConfig — it is
driven straight from zenmetrics-cli — so its knobs need threading through
encode_avif_aom_rs onto BOTH the port's ToggleKnobs and the C oracle
(EncodeCell::c_encode_ctrls), plus a resolver in sweep/dedup.rs, plus gates
G-AOM-BASE and G-AOM-ARM. That work is registered in the plan doc, not done.

Next (per the plan doc's two-stage mandate):
  1. Build + upload the budget corpus if it is not already there:
       scripts/jobsys/avifdoe_build_budget_corpus.py --out /mnt/v/output/avif-doe-1024-2026-09-01
       aws s3 sync ... s3://codec-corpus/avif-doe-1024-2026-09-01/
     G-CROP: the builder exits non-zero until the crops' own features are
     re-extracted and their cluster assignments checked.
  2. Upload each <run>_manifest.json to s3://zentrain/jobs/<run>/manifest.json
  3. Start ONE worker, let one cell land, then G-FIRSTCELL:
       aws s3 ls --endpoint-url "\$ZEN_S3_ENDPOINT" s3://zentrain/jobs/<run>/blobs/
     (the aws CLI, never s5cmd — it undercounts on the LAN store)
  4. Only then scale out, every worker with
       ZEN_CONTROL_KEY=jobs/<run>/control.json
     so the wave is pausable (G-CONTROL; the A0 wave is not).
  5. Point the score gap-fill loop at the new runs AND start score workers —
     as of 2026-09-02T01:55Z nothing on the fleet was scoring at all.
EOF
