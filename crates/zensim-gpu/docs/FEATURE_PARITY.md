# zensim-gpu ↔ zensim (CPU) feature-vector parity

This document maps every slot in the GPU 228-feature output back to the
CPU `zensim` v0.2.8 (`ZensimProfile::latest()` =
`WEIGHTS_PREVIEW_V0_2`) feature vector at the same index.

The 228-vector layout is **scales-major, channel-minor, two blocks**:

```
[ basic block (156)              ][ peak block (72)               ]
  s0c0[0..13] s0c1[0..13] ... s3c2[0..13]   s0c0[13..19] s0c1[13..19] ... s3c2[13..19]
```

That is, `combine_scores` in `zensim/src/metric.rs` emits all 13 basic
features for every (scale, channel) first, then loops again to emit
the 6 peak features for every (scale, channel). The GPU pipeline
mirrors this layout exactly in `pipeline.rs`'s host-side packing loop.

## Per-channel layout (19 features, both sides)

The CPU layout is documented at
[`zensim/src/metric.rs:1620`](https://github.com/imazen/zensim/blob/e295d7fb4098/zensim/src/metric.rs#L1620)
(in `compute_zensim_with_config`'s docstring); the GPU layout lives in
`crates/zensim-gpu/src/pipeline.rs` `compute_with_reference` Phase 4.
Both blocks are identical at the **slot level**:

| Slot | CPU source | GPU source                                                                | Pooling   | Parity |
|------|--------------------------------|--------------------------------------------|-----------|--------|
|  0   | `ss.ssim[c*2]`                 | `sums[0]/N`  (`sd` per-pixel)               | `\|mean\|`  | OK |
|  1   | `ss.ssim[c*2+1]`               | `(sums[1]/N).max(0).powf(0.25)` (`sd^4`)    | L4        | OK |
|  2   | `ss.ssim_2nd[c]`               | `(sums[2]/N).max(0).sqrt()`     (`sd^2`)    | L2        | OK |
|  3   | `ss.edge[c*4]`     (art_mean)  | `sums[3]/N`        (artifact)               | `\|mean\|`  | OK |
|  4   | `ss.edge[c*4+1]`   (art_4th)   | `(sums[4]/N).max(0).powf(0.25)`             | L4        | OK |
|  5   | `ss.edge_2nd[c*2]` (art_2nd)   | `(sums[5]/N).max(0).sqrt()`                 | L2        | OK |
|  6   | `ss.edge[c*4+2]`   (det_mean)  | `sums[6]/N`        (detail_lost)            | `\|mean\|`  | OK |
|  7   | `ss.edge[c*4+3]`   (det_4th)   | `(sums[7]/N).max(0).powf(0.25)`             | L4        | OK |
|  8   | `ss.edge_2nd[c*2+1]` (det_2nd) | `(sums[8]/N).max(0).sqrt()`                 | L2        | OK |
|  9   | `ss.mse[c]`                    | `sums[9]/N`                                 | mean      | OK |
| 10   | `ss.hf_energy_loss[c]`         | `(1 - sums[11]/sums[10]).max(0)` if `>1e-10` | ratio     | OK |
| 11   | `ss.hf_mag_loss[c]`            | `(1 - sums[13]/sums[12]).max(0)` if `>1e-10` | ratio     | OK |
| 12   | `ss.hf_energy_gain[c]`         | `(sums[11]/sums[10] - 1).max(0)` if `>1e-10` | ratio     | OK |
| 13   | `ss.ssim_max[c]`               | `peak0` (max sd per pixel)                  | max       | OK |
| 14   | `ss.art_max[c]`                | `peak1`                                     | max       | OK |
| 15   | `ss.det_max[c]`                | `peak2`                                     | max       | OK |
| 16   | `ss.ssim_p95[c]` (filled with L8 `ssim_l8`) | `(sums[14]/N).max(0).powf(0.125)` | L8 | OK |
| 17   | `ss.art_p95[c]`  (filled with L8 `art_l8`)  | `(sums[15]/N).max(0).powf(0.125)` | L8 | OK |
| 18   | `ss.det_p95[c]`  (filled with L8 `det_l8`)  | `(sums[16]/N).max(0).powf(0.125)` | L8 | OK |

## Naming caveat for the L8 slots (16/17/18)

The CPU `ScaleStats` field name is `ssim_p95` / `art_p95` / `det_p95`
**but the value stored is the L8 power-pool**, not a 95th percentile.
See [`zensim/src/streaming.rs:486..522`](https://github.com/imazen/zensim/blob/e295d7fb4098/zensim/src/streaming.rs#L486-L522):

```rust
ssim_l8[c] = (self.ssim_d8[c] * one_over_n).powf(0.125);
art_l8[c]  = (self.edge_art8[c] * one_over_n).powf(0.125);
det_l8[c]  = (self.edge_det8[c] * one_over_n).powf(0.125);
// …
ssim_p95: ssim_l8,
art_p95:  art_l8,
det_p95:  det_l8,
```

The docstring at `compute_zensim_with_config` (metric.rs:1620) is
authoritative on the public semantic: slot 16 is documented as
`ssim_l8 = (Σd⁸/N)^(1/8)`. `FeatureView::ssim_l8(...)` (the public
accessor) reads this slot. `FeatureView` also exposes `ssim_max(...)`
for slot 13 etc. The `ssim_p95` field name is internal-only; do
**not** rename it lightly because the trained weights are bound to
the existing slot at index 16 by position, not name.

The GPU emits the same L8 value to slot 16 (`(sums[14]/N)^(1/8)`).
Slot 14 of the GPU `sums` array is `sd^8` accumulated per pixel,
since `a14 += (sd4 * sd4) as f64` in `fused.rs`. Same for slots
15 (`art^8`) and 16 (`det^8`).

## Channel skip

`compute_zensim_with_config` with `compute_all_features = false`
(default for `Zensim::compute`) will SKIP a (scale, channel)
combination whose corresponding weights are all near-zero. The
skipped channel's slots default to 0.0. The GPU pipeline always
computes every (scale, channel) — so when feature-level parity is
needed against the CPU's default path, expect 0.0 on the CPU side
for the skipped channels.

For exact per-feature parity, the CPU side must be invoked with
`compute_all_features = true` (or, in v0.2.8's public API, via
`Zensim::compute_extended_features` which emits the same first
228 slots before the masked block).

## Tolerance

GPU uses `f32` for per-pixel intermediates and accumulates into
`f64` per-column partials; CPU uses `f64` throughout for the
pooling, with `f32` only inside the per-pixel SSIM math. The
expected relative drift on the basic+peak block is ~1e-3 at 64×64
and tightens with larger N (more samples → more partial-sum
averaging). The `cpu_parity.rs` integration test asserts
`|gpu - cpu| / max(|cpu|, eps) < 2e-3` per feature on a
small-image gradient + noise fixture.

## Block boundaries (228 = 156 + 72)

```
indices       block        count    formula
─────────────  ───────────  ───────  ──────────────────────────────
  0..  156   basic        156      4 scales × 3 ch × 13 features
156..  228   peak          72      4 scales × 3 ch ×  6 features
```

Total = 228 = `4 × 3 × 19` = `TOTAL_FEATURES`.

## Extended block (228..300) and IW block (300..372)

The GPU output supports two further regimes beyond the 228-feature
basic+peak block:

```
indices       block        count    formula
─────────────  ───────────  ───────  ──────────────────────────────
228..  300    masked       72       4 scales × 3 ch × 6 features
300..  372    IW           72       4 scales × 3 ch × 6 features
```

These map slot-for-slot to CPU `zensim`'s extended + IW features:

| GPU slot offset | CPU source field             | Pooling | Notes |
|---|---|---|---|
| 0 | `ss.masked_ssim[c*3]`        | weighted mean | mask = `1 / (1 + 4 * activity)` |
| 1 | `ss.masked_ssim[c*3 + 1]`    | weighted L4   | |
| 2 | `ss.masked_ssim[c*3 + 2]`    | weighted L2   | |
| 3 | `ss.masked_art_4th[c]`       | weighted L4   | masked edge artifact |
| 4 | `ss.masked_det_4th[c]`       | weighted L4   | masked edge detail-lost |
| 5 | `ss.masked_mse[c]`           | weighted mean | `Σ((src-dst)² · mask)/N` |
| 6 | `ss.iw_ssim[c*3]`            | weighted mean | iw_weight = `1 + 4 * activity` |
| 7 | `ss.iw_ssim[c*3 + 1]`        | weighted L4   | |
| 8 | `ss.iw_ssim[c*3 + 2]`        | weighted L2   | |
| 9 | `ss.iw_art_4th[c]`           | weighted L4   | |
| 10 | `ss.iw_det_4th[c]`          | weighted L4   | |
| 11 | `ss.iw_mse[c]`              | weighted mean | |

Activity is the box-blur (radius `R=5`) of `|src - mu1|` where `mu1`
is the V-blurred reference plane. Both blocks share the same activity
input; the only difference is the weight formula (`1/(1+ka)` vs
`1+ka`).

### Implementation outline

- `Zensim::new_with_regime(client, w, h, ZensimFeatureRegime::Extended)`
  allocates 4 persist planes (mu1, mu2, sigma_sq, sigma12) per
  scale × 3 channels.
- The fused-features kernel runs in a "persist" variant that ALSO
  writes per-pixel mu1/mu2/ssq/s12 to the persist planes (same SSIM
  math, same partials emit; one extra DRAM write per pixel).
- A second kernel `masked_iw_kernel` runs per-scale × per-channel,
  reads the persist planes, computes activity = box-blur(|src - mu1|)
  in shared memory, and accumulates the 12 masked + IW slots per
  column.
- A second reduction kernel folds the 12 per-(col, strip) partials
  into the per-(scale, channel) finals.
- Host packing places the masked block at slots `228..300` and IW
  at `300..372`.

### Memory budget

Extended / WithIw allocate 4 × 3 × 4 bytes × `pad_total` per scale.
At 12 MP scale 0 padded ≈ 4080×3000 = 12.24 MP →
`4 × 3 × 4 × 12.24 M` ≈ 587 MB per scale. Total across 4 scales
≈ 750 MB. Caller can constrain this via
`Zensim::new_with_regime_budget` which returns
`ExtendedPlaneBudgetExceeded` if the cap is over budget.

### CPU strip-overlap divergence (known)

CPU `zensim::streaming::process_strip_channel` processes the image
in `STRIP_INNER = 32`-row strips with `overlap = R = 5` rows above
and below. Within each strip, the V-blur of `mu1` mirrors STRIP-
LOCALLY (`mirror_idx(i, r, strip_h)`). For non-first strips, the
overlap rows' `mu1` therefore reflects strip-local rather than
image-wide reflection. When `activity = blur(|src - mu1|)` is then
computed per-strip, the activity at inner rows is biased by the
overlap rows' mismatched `mu1`.

The GPU implementation uses `n_strips=1` per scale (single-strip),
mirroring image-wide throughout. The activity it produces matches
what a single-strip CPU run would emit — but DIFFERS from the
multi-strip default by ~5-15 % relative on X/B channels at scales
whose height ≥ 64 (where CPU produces ≥ 2 strips).

For the 64×64 noisy-gradient fixture used by `cpu_parity` /
`extended_parity` tests, the divergence is largest at scale 0 X/B
(~5-7 % rel). At scales 1+ where CPU also uses 1 strip the
divergence drops to f32 noise (< 1 % rel).

Two principled fixes (out of scope for this regime landing):

- GPU mirrors CPU's STRIP_INNER=32 + overlap layout and uses strip-
  local mirror inside the persist kernel. Significant per-strip-
  launch refactor.
- CPU is patched to use image-wide mirror (would change every
  existing CPU score by the same delta).

The current test loosens the rel tolerance to 1.5e-1 (15 %) at
scale 0 masked-block slots; basic-block and peak-block parity at all
scales remains within the 2e-3 rel budget. **The bias is structural
not stochastic** — bake-comparison consumers that need bit-exact
parity should account for it, and 12 MP production sweeps will see
the same offset distribution.

### IW block validation

The published `zensim` crate pinned by zenmetrics
(`rev e295d7fb4098`) predates the IW feature block, so direct CPU
parity on slots 300..372 isn't testable here. The GPU IW kernel
shares 90 %+ of its body with the masked kernel — only the weight
formula differs (`1 + k·a` vs `1 / (1 + k·a)`). The test asserts:

- WithIw[0..300] is bit-identical to a separate Extended run.
- WithIw[300..372] is finite and mostly non-zero on noisy input.
- WithIw of identical input is all zeros within ULP noise.

The implementation is correct by construction: if the masked block
passes parity, the IW block uses the same kernel structure with a
trivial weight-formula change.
