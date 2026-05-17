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

Adding the extended (masked) block would push this to 300; adding
IW pushes to 372. Neither is currently a target for zensim-gpu.
