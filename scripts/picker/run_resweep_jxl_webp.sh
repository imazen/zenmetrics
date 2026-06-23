#!/usr/bin/env bash
# Re-sweep jxl + webp to fix coarse ssim2 overhead (rd_core gave too few cells:
# jxl 32.5%, webp 50.3%) AND give the scalar heads real signal. Findings from
# the 2026-06-22 plan preview (docs: benchmarks/picker_smoothness_2026-06-22.md):
#   - webp modes_full (279 cells/img) keeps `method` varying -> ONE combo sweep
#     covers categorical richness + scalar.
#   - jxl  modes_full (228 cells/img) DROPS `effort` (budget collapse) -> rich
#     categorical but degenerate scalar; so ALSO run jxl scalar_dense (70, keeps
#     effort) and MERGE -> train on the union.
# Both metrics in one combo sweep (ssim2-gpu + CPU zensim) now that the
# MetricCache deadlock is fixed (8c373d54); encodes persisted (--encoded-out-dir).
#
# RUN ONLY when the GPU is free (the zensim-gpu repair agent must have finished)
# — ssim2-gpu + this CPU-heavy sweep must not collide with the agent's GPU work.
set -u
SRC=/mnt/v/output/picker-pipeline-2026-06-22
OUT=/home/lilith/picker-pp
# Decoupled per-codec corpora. jxl effort 5-9 encode is slow, so default both to
# 64 imgs (~1.5-2h total) — modes_full's cell richness, not corpus size, is what
# fixes the coarse rd_core overhead. Raise JXL_CORPUS=corpus (154) for a longer
# lower-noise run.
WEBP_CORPUS=${WEBP_CORPUS:-corpus64}
JXL_CORPUS=${JXL_CORPUS:-corpus64}
QG=${QG:-"5,15,30,50,70,85,95"}   # 7 web-weighted q's (low-q dense per CLAUDE.md)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
BIN=/home/lilith/work/zen/zenmetrics/target/release/zenmetrics
combo() { local c=$1 corp=$2 plan=$3 budget=$4 out=$5
  echo "=== $(date -u +%H:%M:%S) $c $plan ($corp, budget $budget) ==="
  timeout 7200 nice -n19 "$BIN" sweep --codec "$c" --sources "$SRC/$corp" --q-grid "$QG" \
    --plan "$plan" --plan-budget "$budget" --metric ssim2-gpu --metric zensim \
    --encoded-out-dir "$OUT/enc/$c" --output "$OUT/sweeps/$out"
  echo "=== $(date -u +%H:%M:%S) $c $plan rc=$? rows=$(wc -l < "$OUT/sweeps/$out" 2>/dev/null) ==="
}
# webp: modes_full covers both
combo zenwebp "$WEBP_CORPUS" modes_full 300 zenwebp.both.tsv
# jxl: modes_full (categorical) + scalar_dense (effort), then merge
combo zenjxl  "$JXL_CORPUS"  modes_full   200 zenjxl.modes.tsv
combo zenjxl  "$JXL_CORPUS"  scalar_dense 120 zenjxl.scalar.tsv
# merge jxl omnis (same header) -> zenjxl.both.tsv
{ head -1 "$OUT/sweeps/zenjxl.modes.tsv"; tail -n +2 -q "$OUT/sweeps/zenjxl.modes.tsv" "$OUT/sweeps/zenjxl.scalar.tsv"; } > "$OUT/sweeps/zenjxl.both.tsv"
echo "merged jxl: $(wc -l < "$OUT/sweeps/zenjxl.both.tsv") rows"
echo "RESWEEP DONE — now adapt+train zenjxl/zenwebp ssim2+zensim_a from *.both.tsv with the improved configs"
