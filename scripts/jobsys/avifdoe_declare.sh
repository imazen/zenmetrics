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
#   scripts/jobsys/avifdoe_declare.sh --stage-b6     # Stage B, trigger B-6 only
#   scripts/jobsys/avifdoe_declare.sh --track-t1     # HBD arm, Track T1 (bd10) only
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
STAGE_B6=0
TRACK_T1=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --sources) SOURCES="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --stage-b6) STAGE_B6=1; shift ;;
    --track-t1) TRACK_T1=1; shift ;;
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
# ---------------------------------------------------------------------------
# STAGE B, trigger B-6 — the ONLY Stage-B block declared (user decision
# 2026-09-02: "go with B-6 first"). The rest of the 60k envelope stays
# undeclared.
#
# B-6 fires when an arm fails T1 of the cross-size transfer gate: its
# effect at the 1024^2 screening budget does not carry to native size, so
# "that knob's Stage-B grid runs at NATIVE size" (plan doc section 7.2).
# Stage A certified only mtx32 and qml1.8.15 for reduced-size screening;
# `acb3` and `shp3` are the two arms it PROVED not screenable
# (avif_doe_stageA_2026-09-02.md section 8.1).
#
# The grid is B-1's registered follow-up shape at native: 5 levels x the
# FULL 29-q ladder x 32 images x speeds {4,6,7}. Both knobs share their
# default level (ac_bias 0.0 == sharpness 0 == the default config), so the
# axis carries 9 levels and the run is 27 strata x 29 q x 32 images =
# 25,056 cells, not the trigger list's per-knob sum of 27,840. The 2,784
# difference IS that shared control, counted once instead of twice.
#
# TWO things here differ from every other block and both are load-bearing:
#   * NATIVE_SOURCES, not SOURCES. The two corpora share filenames by
#     construction (section 12.3), so pointing this at the budget corpus
#     would silently encode the wrong pixels. The declared `source_sha`
#     makes the two disjoint at CellId level -- proven, not assumed.
#   * --max-deviations 2, not 1. `speeds` is itself an axis, so a knob
#     level at speed 6 or 7 spends TWO deviations; at 1 this block would
#     silently emit 11 strata instead of 27 (only the speed-4 leg).
#     It cannot leak interactions: `svt_knobs` is ONE axis, so no cell can
#     spell two knobs at once. `duplicates_merged: 0` is therefore
#     STRUCTURALLY CORRECT here, not the red flag it is on a pairwise plan.
# ---------------------------------------------------------------------------
# HIGH-BIT-DEPTH ARM, TRACK T1 — the `bd10` coverage gaps.
#
# Design + gates: benchmarks/avif_hdr_arm_plan_2026-09-02.md §4.2 and §5.
# This is the registered discharge of trigger B-3 for `bd10`, plus the two
# gaps B-3 does not cover. Stage A measured `bd10` at speed 4 only and it
# WON there (BD-rate -1.02%, CI [-1.23, -0.36], 23/31 images) while being
# "the worst-covered arm in the wave" -- no s6 main effect, no interaction
# coverage, no transfer evidence (avif_doe_stageA_2026-09-02.md §11.8).
#
# THREE runs, and the block-to-run mapping is not one-to-one -- read this
# before comparing counts to the plan doc:
#   * T1-a AND T1-c are ONE run. T1-c ("bd10 @ s6") is literally the s6 leg
#     of T1-a's preset ladder, so declaring it separately would put the same
#     CellIds in two runs' manifests and split one identity space in two.
#     The ladder also KEEPS speed 4 -- already encoded by A1 -- because
#     re-declaring a content-addressed cell is idempotent and free, and it
#     makes the s4 leg a live identity check that this plan reproduces A1's
#     cells (pinned by zenavif's `svt_doe_t1_ladder_s4_cell_ids_match_
#     svt_doe_main`). So T1-a+c declares 7x288 = 2,016, not the plan's 1,728.
#   * T1-b is its own run: 15 live arms x 9 q x 32 = 4,320, exactly as
#     registered.
#   * T1-d is its own run and the ONLY one on the NATIVE corpus.
#
# COUNT RECONCILIATION (both numbers are right for what they count, as with
# B-6's 27,840-vs-25,056): the plan doc's block sum is 6,432 =
# 1,728 (T1-a) + 4,320 (T1-b) + 288 (T1-c) + 96 (T1-d). T1-c's 288 cells are
# counted there BOTH as their own block and inside T1-a's 6-speed ladder, and
# they appear a third time as T1-b's default-knob stratum at s6. What gets
# DECLARED here is 2,016 + 4,320 + 96 = 6,432 job ids. MEASURED distinct
# cells: 6,087 -- two overlaps, both structural:
#   * 288 = the s6 knob-default stratum, shared by the ladder and knob runs.
#   * 57  = T1-d cells that are ALSO t1ac cells. 19 of the 32 corpus images
#     are sub-budget PASSTHROUGHS: their "native" and "1024^2 budget" files
#     are byte-identical, so they share a source_sha and therefore a CellId
#     (Stage A measured this exactly -- "19 byte-identical passthroughs, 13
#     genuinely different crops", and relies on it: A0-native == A0R on
#     3,857/3,857 shared cells). 19 images x 3 probe-q = 57.
#
# ⚠ ANALYSIS CONSEQUENCE FOR T1-d, and it is not cosmetic: the transfer gate
# has **n = 13 images, not 32**. On the 19 passthroughs there is no size
# transfer to measure -- the two sizes are the same pixels and the same
# encode. T1-d's 96 declared cells are the right declaration (the 57 are free,
# already-done work), but any "does -1.02% survive native size" statement must
# be made over the 13 genuinely-cropped images and must report that n.
#
# 10-BIT IS PINNED, NOT CROSSED, and that is the whole mechanism. In
# `svt_doe_main` the bit-depth axis is [Auto, Ten], so at one deviation a
# 10-bit cell has already spent it on depth and can only exist at the default
# speed -- which is exactly why `bd10` lives only at s4 today. The T1 plans
# pin `bit_depths = [Ten]`, making Ten index 0 and therefore ZERO deviations,
# so the speed (or the knob) becomes the isolated deviation. Pinned by
# zenavif's `svt_doe_t1_blocks_pin_ten_bit_as_the_zero_deviation_stratum`.
#
# LAUNCH IN STAGES, per the plan's cheapest-discriminating-first order:
# declare all three (declaring is free), then run T1-a+c and T1-d FIRST --
# their s6 and native legs are the 384 cells that answer "is the effect real
# at a second preset and at native size", which is what gates whether T1-b's
# 4,320 cells are worth buying at all.
#
# GATES that bind at execution, not here: G3 reads the DEPTH BACK OUT OF THE
# STORED BITSTREAM (a request for depth 10 is not evidence of a 10-bit
# stream -- `with_bit_depth` silently coerces, hazard H-BD-3), and G4 forbids
# fitting a BD-rate-vs-speed slope across the preset 8/9 producer seam
# (speeds 1-6 use the full-RD funnel, 7-10 the level-only post-pass -- H-BD-1).
# Every T1 number is scoped by H-BD-4: at presets <= 8 the port is NOT
# byte-identical to C SVT-AV1, so T1 measures THIS PORT's 10-bit encoder and
# no result may be stated as a property of SVT-AV1.
if [ "$TRACK_T1" = 1 ]; then
  # T1-a + T1-c — the preset ladder, budget corpus, 9-q knob ladder.
  declare_block avifdoe-svt-t1ac-20260902 svt_doe_t1_bd10_ladder "$Q_LADDER9" \
      "$SOURCES" 1
  # T1-b — bd10 x the 15 live single-deviation arms at speed 6.
  declare_block avifdoe-svt-t1b-20260902 svt_doe_t1_bd10_knobs "$Q_LADDER9" \
      "$SOURCES" 1
  # T1-d — the cross-size transfer gate. NATIVE corpus, 3-point probe ladder.
  # The two corpora share filenames by construction, so the declared
  # source_sha is what keeps these disjoint from the budget cells at CellId
  # level (proven on 6604: 769b0df4 native vs 4ac38273 crop).
  declare_block avifdoe-svt-t1d-20260902 svt_doe_t1_bd10_transfer "$Q_LADDER3" \
      "${NATIVE_SOURCES:-/mnt/v/output/avifsvt-subsample-2026-09-01/sources}" 1
  cat <<'T1EOF'

Declared Track T1 into the output dir. NOT launched.

  expected: t1ac 2,016  +  t1b 4,320  +  t1d 96  =  6,432 declared
            6,087 DISTINCT — minus 288 (the s6 knob-default stratum, shared
            by the ladder and knob runs) and minus 57 (T1-d cells on the 19
            byte-identical passthrough images, which ARE t1ac cells).
            T1-d's transfer gate therefore has n=13 images, not 32.

  corpus prefix (workers):  s3://codec-corpus/avif-doe-1024-2026-09-01/
                            s3://codec-corpus/avif-subsample-2026-09-01/  (t1d)
  control keys:             jobs/avifdoe-svt-t1{ac,b,d}-20260902/control.json

The worker image MUST know the three `svt_doe_t1_*` plans. An older image
answers `unknown zenavif plan "svt_doe_t1_bd10_ladder"` and every cell fails
as an encoder panic -- the same trap B-6 documented. Rebuild + push the
executor image before launching, and smoke ONE cell first.

  built + verified for this wave:
    ghcr.io/imazen/zenfleet-worker:exec-avifhbd-t1-32e68a8f
    (statically linked musl; `sweep --plan NOPE` inside the image lists all
     three svt_doe_t1_* plans, which is the stale-image check)

RUNBOOK -- STAGE. A declared run needs THREE objects at its prefix, not two.
`declare-encodes` writes the manifest locally; upload it, upload a control.json,
and then MAKE THE LEDGER SNAPSHOT -- `lan_score_launch.sh` sets
ZEN_REQUIRE_SNAPSHOT=1 by default ("strict for single-run queues"), so a run
with no `ledger_snapshot.parquet` at its root is REFUSED BY EVERY WORKER:

  for r in t1ac t1b t1d; do
    RUN=avifdoe-svt-$r-20260902
    aws s3 cp --endpoint-url "$ZEN_S3_ENDPOINT" \
      $OUT/${RUN}_manifest.json s3://zentrain/jobs/$RUN/manifest.json
    echo '{"paused":true,"drain":true}' | \
      aws s3 cp --endpoint-url "$ZEN_S3_ENDPOINT" - s3://zentrain/jobs/$RUN/control.json
    ./target/release/zenfleet-ctl compact --run "$RUN" --bucket zentrain \
      --endpoint "$ZEN_S3_ENDPOINT" --upload      # <-- the ledger_snapshot.parquet
  done

RUNBOOK -- ENCODE. One canonical launcher per run; note the CORPUS PREFIX
differs for t1d and getting it wrong is the silent-garbage hazard that
avifdoe_score_gapfill.sh's header documents (the two corpora share all 32
filenames; 13 of them are genuinely different pixels):

  IMG=ghcr.io/imazen/zenfleet-worker:exec-avifhbd-t1-32e68a8f
  # t1ac + t1b -- the 1024^2 BUDGET corpus
  ZEN_CORPUS_BUCKET=codec-corpus ZEN_CORPUS_PREFIX=avif-doe-1024-2026-09-01 \
    bash scripts/jobsys/lan_score_launch.sh <host> avifdoe-svt-t1ac-20260902 t1ac cpu "$IMG"
  ZEN_CORPUS_BUCKET=codec-corpus ZEN_CORPUS_PREFIX=avif-doe-1024-2026-09-01 \
    bash scripts/jobsys/lan_score_launch.sh <host> avifdoe-svt-t1b-20260902  t1b  cpu "$IMG"
  # t1d -- the NATIVE corpus. NOT the budget one.
  ZEN_CORPUS_BUCKET=codec-corpus ZEN_CORPUS_PREFIX=avif-subsample-2026-09-01 \
    bash scripts/jobsys/lan_score_launch.sh <host> avifdoe-svt-t1d-20260902  t1d  cpu "$IMG"

Each run is staged with control.json {"paused":true,"drain":true}; flip to
{"paused":false,"drain":false} to start it. Launch t1ac + t1d FIRST and leave
t1b paused -- their s6 and native legs are the 384 cells that gate whether
t1b's 4,320 are worth buying.

RUNBOOK -- SCORE. Do NOT fork the gapfill loop; it is parameterised for
exactly this reuse. t1d takes the NATIVE refs prefix:

  ZEN_DOE_SFRUN=avifdoe-svt-t1-sf-cpu-20260902 \
  ZEN_DOE_RUNS="avifdoe-svt-t1ac-20260902=s3://codec-corpus/avif-doe-1024-2026-09-01/ \
                avifdoe-svt-t1b-20260902=s3://codec-corpus/avif-doe-1024-2026-09-01/ \
                avifdoe-svt-t1d-20260902=s3://codec-corpus/avif-subsample-2026-09-01/" \
    bash scripts/jobsys/avifdoe_score_gapfill.sh

TOPOLOGY, measured 2026-09-02 while B-6 was draining (observe before load):
  * r7900x  24 threads, load ~9   -- runs one B-6 encode worker; the natural
            home for t1ac once B-6 clears.
  * tower   32 threads, load ~11  -- ALSO runs a B-6 encode worker, ALONGSIDE
            the household media stack (plex/sonarr/homeassistant/...). Never
            launch uncapped here: ZEN_CPUSET=0-23 ZEN_CPU_SHARES=256
            ZEN_MEMORY=24g leaves the household its 8 cores.
  * dev     32 threads, load ~33  -- SATURATED by the 5 score workers. Do NOT
            add an encode worker here; it is the box that scores.
B-6's encode ran on BOTH tower and r7900x, so both free up together.

GATE G3 before trusting any BD-rate -- a request for depth 10 is not evidence
of a 10-bit stream:

  cargo build --release -p zenmetrics-cli --example avif_depth_verify \
    --no-default-features --features avif
  ./target/release/examples/avif_depth_verify --expect-depth 10 <blobs-dir>/
T1EOF
  exit 0
fi

if [ "$STAGE_B6" = 1 ]; then
  declare_block avifdoe-svt-b6-20260902 svt_doe_b6 "$Q_FULL29" \
      "${NATIVE_SOURCES:-/mnt/v/output/avifsvt-subsample-2026-09-01/sources}" 2
  cat <<'B6EOF'

Declared B-6 into the output dir. NOT launched.

  corpus prefix (workers):  s3://codec-corpus/avif-subsample-2026-09-01/
  fleet image:              ghcr.io/imazen/zenfleet-worker:exec-avifdoe-b6-<sha>
  control key:              jobs/avifdoe-svt-b6-20260902/control.json

The image MUST carry zenavif >= 43423054 (the commit that adds the
`svt_doe_b6` plan). An older image answers `unknown zenavif plan
"svt_doe_b6"` and every cell fails as an encoder panic -- the exact
G-FIRSTCELL failure section 11.5 records. Verify the tag resolves the plan
before scaling past one worker.
B6EOF
  exit 0
fi

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
