# K=1 single-encode picker — cross-codec verdict + jpeg/webp lever investigation (2026-06-29)

Clean ssim2-target K=1 pickers on the balanced `clean-picker-corpus-2026-06-26`
data (origin even/odd split), trained through the **zone-aware DATA_STARVED gate**
+ gate-aligned knob-veto deriver. Commits: zenpredict `unachievable_zone` /
`picker_safety` (eb7385d, 896e279), train/bake zone descriptor (10f48478),
veto-deriver gate-alignment (a2e12892); zenmetrics features-fix (ae39463b).

## Cross-codec K=1 verdict

| codec | cells | teacher argmin_acc | student K=1 mean | p99 | max | gate |
|-------|-------|--------------------|------------------|-----|-----|------|
| **jxl-lossy**  | 3  | 86%   | **0.78%** (TEST) | 17% | 72%* | **PASS** ✅ |
| webp-lossy     | 11 | 37%   | 2.51%            | 44–67% | 610–1199% | FAIL (tail) |
| avif-lossy     | 8  | 19%   | 7.95%            | 74% | 342% | FAIL (mean+tail) |
| jpeg-lossy     | 18 | 34%   | 9.34%            | 58% | 237% | FAIL (mean+tail) |

\*jxl after 2 gate-aligned vetoes (train tail 89→72%). Baked HONESTLY (no
`--allow-unsafe`): `zenjxl_ssim2_v0.1.bin`, 2 vetoes + 3 unachievable zones.

**Key result: teacher ≈ student for ALL** → these are **feature ceilings**, not
MLP underfit. K=1 is clean only when one cell dominates (jxl vd_zen wins 94%);
rich multi-cell codecs whose optimal encoder-strategy is content-dependent can't
be picked at K=1 with the current features.

## jpeg lever investigation (per user 2026-06-29: chroma features / sampling / q-range)

- **Subsampling is the weak axis.** Decomposing the jpeg pick error: **strategy
  (jp3/moz/gls/pw4) 76.5% correct, but subsampling (420/422/444/xyb) only 57%.**
  43% of the overhead is sub mis-picks → chroma features are the right lever
  (they ARE present: chroma_complexity, cb/cr_sharpness, chroma_luma_covariance…).
- **libjpeg q 35–94 ≈ ssim2 55–90** (q50→64, q70→72, q85→81, q94→86). Restricting
  the eval to that production range does **NOT** help: mean 9.47% (production) vs
  9.34% (full grid) — the overhead is **pervasive across the production range**,
  not an out-of-range artifact. So the q-range focus alone doesn't rescue jpeg;
  the limit to document is that the picker is ~9% K=1 *throughout* 35–94, not
  just at the extremes.
- **Sub-accuracy degrades exactly with sampling pressure** (`DEFAULT_PIXEL_BUDGET
  = 500_000`; tiny ≤4096 + small ≤65536 are fully sampled, only medium >500k is
  stride-sampled):

  | size | sampled @500k | sub-correct | mean overhead |
  |------|---------------|-------------|---------------|
  | tiny  | full   | 68.3% | 5.4% |
  | small | full   | 60.8% | 8.6% |
  | **medium** | **stride ~½** | **53.1%** | **10.3%** |

  Medium — the only budget-limited class — has the **worst** sub discrimination
  and overhead. This strongly supports the user's hypothesis: **raising the
  pixel budget so medium is fully sampled should sharpen its chroma features →
  better sub discrimination → lower medium overhead.** (Caveat: tiny/small are
  already fully sampled, so the budget lever is medium-only; webp's worst tail is
  on TINY (610–1199%), which is a too-few-blocks/structural issue, not budget.)

## Validated next eval (the user's "eval that")

Re-extract features with full sampling and retrain jpeg, measuring the medium
sub-accuracy delta:
- zenanalyze has the mechanism — `analyze_specialized_raw(full_budgets=true)`
  sets `pixel_budget = usize::MAX` (lib.rs:924); `AnalysisQuery.pixel_budget`
  (lib.rs:660/900) is the lower-level knob. The public `analyze_features_rgb8`
  freezes it as a crate invariant, so the eval needs a small extractor change
  (use the budget-configurable entry at a high/MAX budget) + a zenanalyze
  rebuild + re-extract the clean-picker-corpus renditions + retrain.
- Apply to **both ssim2 and zq (zensim) targets**.
- If medium sub-accuracy jumps (53% → ?), re-extract the full corpus and rebake
  jpeg/webp/avif (and re-verify jxl, already full-budget-insensitive at 3 cells).

## Per-codec path

- **jxl**: ship (baked honestly). 
- **jpeg/webp/avif**: blocked at K=1 by the subsampling feature ceiling. Levers,
  in order: (1) full-budget chroma re-extraction (above, medium-focused);
  (2) additional chroma-crate features for sub discrimination; (3) if the ceiling
  holds, the cell decomposition / accept-overhead decision. The zone-aware gate
  correctly REFUSES these rather than `--allow-unsafe`-ing a feature ceiling.
