# ssim2-gpu fix assessment — 2026-05-27

Phase 0 gating doc for the launch-fusion work tracked by the
companion `SSIM2_OPTIMIZATION_REVIEW.md` (commit `b1d080a`). The
question to answer here is **easy fixes vs hard fix, in what
order, and what's the expected payoff per dollar of risk**. The
six-fix taxonomy and HtoD/launch counts referenced below are from
that review — see it for the per-line citations.

## Is there an "easy fix"? Yes — but the payoff cap is small.

Two genuine quick wins exist, both subsumed by Fix #1 if it lands
but ship-able standalone:

- **3-channel-fused downscale.** ssim2's `downscale_2x_plane_kernel`
  (`crates/ssim2-gpu/src/kernels/downscale.rs:17`) already exists as
  a 3-channel variant in zensim (`zensim-gpu/src/kernels/downscale.rs:11`,
  `downscale_2x_3ch_kernel`). Port it verbatim, swap the
  3-launches-per-scale-transition loop in
  `pipeline.rs:1903-1931` (`build_linear_pyramid_until`) for a
  single launch. Saves **~5 × (S-1) = 25 launches per `compute()`**
  (5 scales × 3 channels → 5 launches), bit-identical output
  modulo the saturating-vs-bare-subtract clamp difference (which
  is a no-op for the always-positive widths/heights ssim2 uses).
  ~30 LOC change, near-zero risk.

- **3-channel-fused pointwise mul.** `pointwise_mul_kernel`
  (`pipeline.rs:2556`) gets launched once per channel inside
  `pointwise_mul` / `pointwise_mul_masked`. Add a 3-ch variant
  (`pointwise_mul_3ch_kernel`) and a `pointwise_mul_masked_3ch`
  wrapper. Saves 27 launches per `compute()` (3 products × 3 channels
  × ~5 active scales after skip-map). Same shape of port; ~50 LOC.

The cvvdp "constants lift" idiom **does not apply** here. ssim2 has
no per-call uniform uploads to lift — every persistent buffer is
already allocated in `Ssim2::new`/`Scale::new`, every per-call HtoD
is the distorted image bytes themselves (already on the pinned
fast path per T_x.O 2026-05-17). The persistent-transient idiom
from cvvdp's commit a5247a80 also doesn't fit: ssim2's per-scale
buffer set is already persistent (the `Scale` struct), so there's
nothing to convert.

## EASY-FIX expected payoff (order of magnitude)

Combined easy-fix delta if both #2 and #3 ship standalone:

- HtoD: 52 → ~28 per `compute_with_reference_with_mode` (-46%).
- Steady-state wall at 12 MP: estimated -0.6 to -1.0 ms based on
  ~25-50 µs/launch × ~25 launches. The CPU dispatch overhead is
  real but small compared to the per-scale DRAM traffic that
  fixes #1/#4/#5 wipe out.
- **VRAM: zero change.** The 6.2 GB headline is owned entirely by
  the five `_full` plane families + `_t` xyb pair, both of which
  only Fix #1 (and its piggyback #4/#5) touch.

**Bottom line.** Easy fixes alone save ~halve the launch count and
~1 ms wall. They do **not** fix the 6.2 GB "doesn't-fit-on-8GB"
headline that motivates this whole work item.

## HARD-FIX scope (zensim-style fused-features port)

Concretely: write a `ssim2_fused_features_kernel` modelled on
`zensim-gpu/src/kernels/fused.rs:95`. The kernel math is **already
identical** to ssim2's semantics — verified by reading both:

- zensim's V-blur SSIM: `denom_s = ssq - mu1² - mu2² + C2`
  (fused.rs:294); ssim2's `error_maps_kernel`: `(s11 - mu11) +
  (s22 - mu22) + C2` (error_maps.rs:79). Equal under `ssq =
  blur(ref² + dis²)` (which is exactly what zensim's H-blur step
  produces at fused.rs:235).
- zensim's artifact/detail: `ed = (1+|dv-mu2|)/(1+|sv-mu1|) - 1`
  (fused.rs:322); ssim2's: identical (`error_maps.rs:88`).
- zensim's already emits per-channel `(Σssd, Σssd⁴)` (`a0`,`a1`),
  `(Σart, Σart⁴)` (`a3`,`a4`), `(Σdet, Σdet⁴)` (`a6`,`a7`) —
  matches ssim2's 6-stat-per-(scale,ch) need exactly.

**LOC estimate:** ~700 LOC for the kernel (mostly straight copy of
zensim's `fused_features_kernel` trimmed of the 9 features ssim2
doesn't use — `a2/a5/a8/a9..a16` peaks). Plus ~300 LOC of pipeline
wiring (replace `process_scale`'s blur+xform+errormap+reduce stack
with one fused launch + a per-(scale,ch,map) folding reduction).

**Risk: medium.** The hottest single risk is the per-(col,strip,ch)
partials layout — zensim emits 17 f64 + 3 f32 per col, ssim2 needs
only 6 f64 per (scale, ch). Two viable layouts:

1. **Copy zensim's layout** (per-col partials → host fold).
   Bit-identical to current ssim2 for the per-(scale,ch,map)
   reduction (atomic-add tolerance not relevant because no atomics
   are used; the per-thread accumulator → memory store → host fold
   IS the current zensim path). Pros: closest to zensim's working
   code. Cons: ~6× the partials buffer size ssim2 currently uses.

2. **Atomic-add into the existing 54-slot sums buffer.**
   Eliminates the per-col partials buffer entirely but introduces
   atomic-add tolerance (~5e-5 expected) — fine per the task brief.
   Pros: ssim2's `read_and_aggregate` doesn't change at all. Cons:
   atomic contention on 54 slots from millions of threads could
   tank throughput.

Recommended path: **(1) first** because it preserves bit-identical
output and matches zensim's proven-working code; **(2)** only if
profiling shows the partials-buffer DRAM traffic dominates.

**Parity gate:** existing `parity_lock.rs` test compares against
the CPU `ssimulacra2` reference at multiple sizes / quality levels.
Run after every commit. ssim2's score should match to within
2 ULP of f64 (the score itself goes through a 108-term weighted
dot product, so a few ULP of drift in the per-cell stats can
multiply out — the current `parity_lock` tolerance is the gate).

## Recommendation

**Path A (easy fixes first) for one commit, then Path B (hard fix).**

Rationale:

- The 3-channel-downscale port is genuinely ~30 LOC and de-risks the
  3-channel-arg cubecl pattern before the bigger fused kernel
  (where the same idiom expands to 6 channels × 3 ref/dis = 6
  arrays of f32 per channel pair).
- Easy fix #3 (3-ch pointwise) is **not worth shipping standalone**
  because Fix #1 subsumes it AND the pointwise_mul_3ch kernel becomes
  dead code post-Fix #1 (no `sigma11_in`/`sigma22_in`/`sigma12_in`
  buffer to write to — the fused kernel produces blurred sigmas in
  shared memory only). Skip.
- After the downscale port + parity verification, go straight to the
  fused kernel port. That's where the 3-5× speedup + the 3 GB VRAM
  drop live.

If ANY phase of the fused-kernel port breaks parity beyond the
atomic-add tolerance and can't be restored, honest-stop per the
brief: document the failure mode, leave the easy-fix commit pushed,
do not ship a broken fused kernel.

## Commit plan

1. **`perf(ssim2-gpu): port 3-channel-fused downscale from zensim-gpu`**
   - Add `downscale_2x_3ch_kernel` to ssim2's kernels.
   - Rewire `build_linear_pyramid_until` to single-launch per scale.
   - Parity: existing tests pass; report HtoD/iter delta and wall p50.

2. **`perf(ssim2-gpu): port zensim-style fused-features kernel — kernel-only`**
   - Add `fused_features_kernel` to ssim2's kernels, layout matching
     zensim's (per-col partials, host fold).
   - DON'T wire into the pipeline yet — landing as a dead-code
     review-able commit first.

3. **`perf(ssim2-gpu): wire fused-features kernel through compute path`**
   - Replace `run_self_products_masked` + `run_cross_product_masked`
     + `run_blur_full_masked` + `run_transpose_raw_xyb_pair_masked`
     + `run_error_maps_masked` + `run_reductions_masked` with a
     single `launch_blur_and_features`. Drop the 5 `_full` /
     `_v_scratch` / `_t_scratch` / `_t` buffer families from
     `Scale::new`. Adjust `read_and_aggregate` for the new partials
     layout.
   - Parity: bit-identical at multiple sizes; VRAM @ 4096² should
     drop from ~6.2 GB to ~2-3 GB.

4. **(optional) `perf(ssim2-gpu): port to cached-ref / strip paths`**
   - Update `compute_with_reference_*` and strip mode to use the same
     fused kernel.

Each commit pushes immediately and gates on the 6-point checklist
(test pass, score parity, nvsmi delta, steady-state wall p50,
cached-ref smoke, strip mode smoke).

Estimated total scope: 1-1.5 days for a developer familiar with the
zensim-gpu fused kernel pattern.
