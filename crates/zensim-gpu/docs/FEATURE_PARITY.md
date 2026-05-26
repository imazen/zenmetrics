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

### CPU strip-overlap parity (fixed 2026-05-17)

CPU `zensim::streaming::process_strip_channel` processes the image
in `STRIP_INNER = 32`-row strips with `overlap = R = 5` rows above
and below. The masked + IW path runs `activity = blur(|src - mu1|)`
on the strip buffer, where `bufs.mu1` at the strip's OVERLAP rows
contains stale/garbage data (not the real V-blurred mu1 at those
image rows). Specifically:

- For the FIRST processed channel (X), `bufs.mu1` at overlap rows
  is zero (fresh band-scoped buffer).
- For SUBSEQUENT channels (Y, B), `bufs.mu1` at overlap rows holds
  the PREVIOUS channel's H-blurred mu1 — a side effect of CPU's
  `swap(&mut bufs.mu1, &mut bufs.mask)` after each channel's V-blur,
  combined with the V-blur only writing the inner rows of its
  `mu1_out` target.

The 2026-05-17 strip-local kernel
(`kernels::masked_iw_strip::masked_iw_strip_kernel`) replays this
behavior on the GPU:

1. Launches `(ceil(pw / TX), num_cpu_strips, 3)` where
   `num_cpu_strips = ceil(h / STRIP_INNER)` — matches CPU's strip
   count exactly.
2. Per strip, splits strip-rows into inner `[inner_off,
   inner_off + inner_h)` vs overlap. At inner rows, reads the
   per-channel persist mu1 plane (image-wide V-blur). At overlap
   rows:
   - Channel 0 (X): `mu1 = 0`.
   - Channels 1/2 (Y/B): computes H-blur of the previous channel's
     source on the fly via a wider `prev_src_wide[TX + 4R]` shared
     tile + per-thread H-blur fill of `mu1_row[TX + 2R]`.
3. Computes activity via the standard H-then-V box blur with
   strip-local mirror at strip boundaries (period = `2*(strip_h-1)`).

### Parity status after the fix (64×64 noisy-gradient + 128×128
checkerboard fixtures, parallel CPU)

- **X channel (channel 0, first processed)**: rel ≤ 1e-4 at all
  scales.
- **Y channel (channel 1, prev = X)**: rel ≤ 2e-4 at all scales.
  X is smooth-in-y in the test fixture, so persist V-blurred X mu1
  is essentially identical to H-blurred X mu1 — on-the-fly H-blur
  closes Y from V0_2's 5 % residual to f32 noise (~2e-6 rel).
- **B channel (channel 2, prev = Y)**: rel ≤ 4e-2 at multi-strip
  scales (scale 0 of 64×64 / 128×128; scale 1 of 128×128). Rel
  ≤ 5e-3 at single-strip scales. The 3-4 % residual is documented
  in the next section.

The `extended_parity` tests use 5e-3 rel for X/Y at all scales and
5e-2 rel for B at multi-strip scales; all other masked slots
(single-strip B; max-pooled; L8) stay at 5e-3 rel.

### Principled per-channel H-blur activity — 2026-05-17 redesign

The 2026-05-17 RCA (see `examples/b_channel_diagnostic.rs` for the
methodology — instrumented CPU+GPU shadow per-row dumps to TSV)
identified the original CPU activity computation as an accidental
cross-channel buffer-reuse cascade, NOT a designed algorithm:

  - X (channel 0) overlap mu1 = 0 (initial `ScaleBuffers::new` zeros)
  - Y (channel 1) overlap mu1 = `src_X(gy, x)` (RAW, not H-blurred)
  - B (channel 2) overlap mu1 = `|src_Y(gy, x) - src_X(gy, x)|`
  - Strip K≥1 inherits leftover `bufs.mask` state from prev strip's B.

CPU was changed (commit `caf52d36` on `feat/principled-activity`) to
use a **per-channel strip-local H_blur(src)** as the activity-map
reference at ALL strip rows (inner + overlap). Channels are now
decoupled; the activity for each channel sees only its own
H-blurred source. See the zensim repo's `docs/PRINCIPLED_ACTIVITY.md`
for the rationale.

### GPU implementation: on-the-fly H_blur(src) per channel per row

`kernels::masked_iw_strip::masked_iw_strip_kernel` now:

1. Loads a DIAM-wider source window per (row, channel) into shared
   memory: `wide_src[TX + 4R]` (84 f32s with R = 5).
2. Each thread cooperatively computes `H_blur(src)` at TILE_COLS
   = TX + 2R = 74 column positions by averaging DIAM = 11 adjacent
   `wide_src` values. Result lives in `mu1_row[TILE_COLS]`.
3. Per-column H-sum of `|src - H_blur(src)|` over DIAM uses the same
   `mu1_row` as before. Strip-local V-blur of the resulting activity
   map is unchanged.
4. Inner-row pixel features still read `mu1` / `mu2` / `ssq` / `s12`
   from the persist planes (image-wide V-blur-of-H-blur values used
   by masked-SSIM and masked-edge math).

The cross-channel cascade, strip-K-vs-strip-0 branch, and host-side
carryover-plane simulator (`pipeline.rs::populate_carryover`) are
all **deleted** — the new algorithm doesn't need them. The carry
`Array<f32>` parameter (~23 MB max at 12 MP per scale) is gone too,
bringing the kernel back to 12 Array args naturally (no
arg-count-collapsing workaround needed).

### Parity result

All masked-block features (slots 228..300) and IW-block features
(slots 300..372) match CPU within **5e-3 rel at every scale and
every fixture size**, including multi-strip scales (128×128 scale 0
+ scale 1; 12 MP scale 0 + scale 1 + scale 2). No per-channel,
per-scale tolerance widening needed.

### Perf cost (12 MP RTX 5070, WithIw 372-feature)

- Pre-2026-05-17 strip-local (with carryover branches + carry plane
  loads): ~26.92 ms / iter mean baseline.
- Post-redesign (principled per-channel H-blur, no carry, no
  cross-channel cascade): ~22.9 ms / iter mean (warm iters 1-4).
- **Faster by ~15 %** — the kernel does slightly more work
  per-thread (DIAM extra src loads to compute H-blur on the fly) but
  loses all the per-channel branches in the hot path AND drops a
  ~23 MB device buffer that was being read every iteration.

### IW block validation

The path-pinned `zensim` crate
(`../zensim--principled-activity/zensim`) exports the IW feature
block via `ZensimResult::features()` whenever the active profile's
`ProfileParams` carries `compute_iw_features: true`. The current
`latest()` profile (`PreviewV0_3` aka `A`) does so, which means
`ZensimCpu::new(latest()).compute_extended_features(...)` returns a
372-feature `Vec<f64>` containing the IW block at slots 300..372 —
**no `training` feature, no `compute_zensim_with_ref_and_config`
plumbing required.**

Direct per-slot parity coverage in
`tests/extended_parity.rs`:

- `iw_slot_parity_noisy_gradient_64` — 64×64 gradient + noise,
  asserts GPU vs CPU `5e-3 rel` per slot across the 72 IW features.
- `iw_slot_parity_checkerboard_128` — 128×128 checkerboard + noise,
  same `5e-3 rel` budget across multi-strip scales (scale 0 + 1).

Structural coverage retained:

- `with_iw_structural_noisy` — WithIw[0..300] is bit-identical to a
  separate Extended run; WithIw[300..372] is finite, mostly non-zero
  on noisy input, and within reasonable magnitude bounds.
- `with_iw_identical_zeros` — WithIw of identical input is all zeros
  within ULP noise.

The implementation correctness now sits on three layers: (1) direct
CPU per-slot parity at two fixture sizes, (2) the masked block's
existing parity (the IW kernel shares 90 %+ of its body, with only
the weight formula differing — `1 + k·a` vs `1 / (1 + k·a)`), and
(3) structural finite-and-bounded checks.
