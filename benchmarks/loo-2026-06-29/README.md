# Fleet-LOO single-feature ablation — ssim2 pickers (2026-06-29)

Round-1 single-feature leave-one-out (drop each feature, retrain, measure held-out
ssim2 RD-overhead delta) for jpeg/webp/jxl lossy pickers, fanned across EU Hetzner
ccx33 boxes (avif skipped — first-cut showed its picker is inert). Per-feature
val/test deltas in `loo_<codec>_ssim2.tsv`; ranked summaries in `*_summary.md`.

Baselines + feature usage (must-keep = val delta > 0.05pp when dropped):
- jpeg: val 8.23% / test 7.88% — only **5** must-keep (deltas ≤0.22pp); weak picker.
- webp: val 2.51% / test 2.53% — **26** must-keep (deltas ≤0.25pp); moderate.
- jxl:  val 3.91% / test 4.03% — **74** must-keep (deltas up to **+1.2pp**: aq_map_mean,
  luma_histogram_entropy, quant_survival_*); content-rich, the valuable picker.

ROUND-2 (pair-aware group-drop verification — single-LOO under-estimates redundant
pairs) pending: drop the droppable-looking set jointly; if RD jumps, bisect for the
largest safely-droppable subset. Run locally (no fleet).

## INFRA BUG (fix before next fleet-LOO)
Box-side self-destruct in `scripts/train/loo_box_runner.sh` FAILED: round-1 boxes
finished + uploaded results ~02:56 UTC but did NOT delete themselves — 18 ccx33 boxes
idled ~6h (~€29 wasted) until manually killed. The watchdog backstop also wasn't
independently armed (it was the stopped agent's child). Fix: (a) make the box-side
teardown robust (verify the hcloud delete via metadata-id + token actually fires on
BOTH success and failure paths), and (b) always arm an independent host-side €/age
watchdog that survives the launcher process.
