#!/usr/bin/env bash
# hdrgrid_drain_endgame.sh — the COMMITTED ENDGAME for the hdrgrid GPU drain
# (registered 2026-08-27; zensim PLAN "BROWSER refresh" + "HDR-944 L1" both
# consume this output). Run it ONCE when the drain completes; every step is
# fail-loud and idempotent-safe (re-running after a partial failure is fine).
#
#   1. GATE: all six GPU score runs report VERDICT COMPLETE (live gap 0).
#   2. HARVEST: writeback_scores.py two-stage per codec -> refreshed
#      scores/features parquets (env template below; bridge parquet =
#      pairs_full.parquet, carries encode_sha per cell).
#   3. ERA: re-run the judge-era deriver so the era table covers any
#      newly-landed zensim rows; consumers use the era-B slice ONLY.
#   4. POOL: regen per coefficient viewer/static/data-hdrgrid/README.md
#      (rollup + viewer-check) — from the REFRESHED base.
#   5. MANIFEST + orientation gate + Tower mirror per the recorded chain
#      (PLAN line ~734).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ZM_ROOT="$(cd "$HERE/../.." && pwd)"
CTL="$ZM_ROOT/target/release/zenfleet-ctl"
HG=/mnt/v/output/hdrgrid-2026-08-06
RUNS=(hdrgrid-sf-gpu-20260807 hdrgrid-sf-gpu-huge-20260807 hdrgrid-sf-gpu-small-20260807 \
      hdrgrid-sf2-gpu-20260807 hdrgrid-sf2-gpu-huge-20260807 hdrgrid-sf2-gpu-small-20260807)
say(){ echo "[$(date -u +%FT%TZ)] $*"; }

# creds (on-box only, never argv)
set -a; . "$HOME/.config/zen/lanstore.env"; set +a
export AWS_ACCESS_KEY_ID="$ZEN_S3_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$ZEN_S3_SECRET_ACCESS_KEY"

say "STEP 1: drain gate"
ARGS=(); for r in "${RUNS[@]}"; do ARGS+=(--run "$r"); done
OUT=$("$CTL" report "${ARGS[@]}" 2>&1) || true
echo "$OUT" | tail -9
echo "$OUT" | grep -q "VERDICT: COMPLETE" || { say "GATE FAIL: drain not complete — aborting (gaps above)"; exit 2; }

say "STEP 2: harvest (writeback per codec, two-stage env)"
[ -s "$HG/pairs_full.parquet" ] || { say "missing bridge parquet $HG/pairs_full.parquet"; exit 3; }
STAMP=$(date -u +%Y-%m-%d)
export ZEN_STORE=tower ZEN_JOBS_BUCKET=zentrain ZEN_PAIRS_PARQUET="$HG/pairs_full.parquet"
export ZEN_WRITEBACK_DIR="$HG/harvest-$STAMP/%s"   # writeback substitutes codec if it %s-formats; else per-codec below
SF_RUNS="hdrgrid-sf-gpu-20260807,hdrgrid-sf-gpu-huge-20260807,hdrgrid-sf-gpu-small-20260807,hdrgrid-sf2-gpu-20260807,hdrgrid-sf2-gpu-huge-20260807,hdrgrid-sf2-gpu-small-20260807,hdrgrid-sf-cpu-20260807,hdrgrid-sf2-cpu-20260807"
# ONE call: writeback's codec arg is a label, and the pairs bridge covers the
# whole corpus — the 2026-08-27 run's per-codec loop produced 3 IDENTICAL
# full-corpus harvests (canonical promoted to the harvest root, dupes bak'd).
export ZEN_WRITEBACK_DIR="$HG/harvest-$STAMP"
say "  writeback (full corpus, single call)"
python3 "$HERE/writeback_scores.py" hdrgrid all "$SF_RUNS" \
  || { say "WRITEBACK FAIL — check the two-stage env contract"; exit 4; }

say "STEP 3: judge-era refresh (era-B is the consumable zensim slice)"
python3 /home/lilith/work/zen/zensim/scripts/canonical_corpus/derive_hdrgrid_zensim_judge_era.py \
  --cells "$HG/zensim_scores_by_judge_era.parquet" || exit 5

say "STEP 4: pool regen (coefficient owner recipe) — run from ~/work/coefficient:"
say "  python3 scripts/rollup_zenmetrics.py --base <refreshed viewer base> --sidecar <refreshed sidecar> --datasets hdrgrid_hdr --out viewer/static/data-hdrgrid && just viewer-check"
say "  (viewer base ETL from harvest-$STAMP per viewer/static/data-hdrgrid/README.md; score_zensim MUST come from the era-B slice)"

say "STEP 5: manifest + orientation gate + Tower mirror (recorded chain)"
say "  check_target_orientation.py + _MANIFEST.json build_commit + scp to tower:/mnt/user/coefficient/output/hdrgrid-2026-08-06/"
say "ENDGAME COMPLETE through step 3; steps 4-5 are printed owner commands (interactive halves)."
