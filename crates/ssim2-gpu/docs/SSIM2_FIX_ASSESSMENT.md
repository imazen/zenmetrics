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

## REVISION — 2026-05-27 (post first-attempt deep dive)

After landing the easy-fix #2 (3-channel-fused downscale, commit
3ac28448) the implementation team confirmed bit-identical output and
the assessment's "easy fixes save launches but not memory" prediction.
The hard fix turned out to be **harder than this assessment described**
because of one missed factor:

### ssim2 uses IIR, zensim uses FIR

ssim2's blur kernel is a recursive Charalampidis IIR Gaussian
(`crates/ssim2-gpu/src/kernels/blur.rs:124`) — a column-walker with
6 floats of IIR state per column, walking each column top-to-bottom.
zensim's fused kernel uses an **11-tap mirror-padded FIR** in shared
memory (DIAM=11, `crates/zensim-gpu/src/kernels/fused.rs:230-238`).
These two blurs are mathematically distinct — running the SSIMULACRA2
math on FIR-blurred sigmas produces a **different score** than
running it on IIR-blurred sigmas. SSIMULACRA2 spec mandates the IIR.

So zensim's `fused_features_kernel` is NOT portable verbatim to
ssim2-gpu. A direct port would break the CPU-parity guarantee
(`parity_lock::parity_jpeg_corpus`).

### Revised hard-fix design

The architecture that **does** work for ssim2 (preserves IIR semantics):

1. Eliminate the v-pass + transpose + v-pass shape by introducing an
   **H-pass IIR kernel** (row-walking, one thread per row, 6 floats of
   state per thread). The IIR is separable; v-pass + h-pass produces
   the same result as v + transpose + v.
2. Fuse the h-pass-second with the error_maps math. The fused kernel
   reads 5 v-pass outputs (sigma11/22/12/ref/dis) in untransposed
   orientation, walks horizontally with 30 floats of IIR state per
   thread (5 planes × 6 state floats), at each emit step has the
   fully-blurred sigmas/mus available + reads raw ref/dis untransposed,
   computes ssim/art/det, accumulates per-thread (Σ, Σ⁴).
3. Process channels SERIALLY to share the 5 v-pass buffers across
   channels (-1.8 GB at 4096² scale 0). Per-thread state is 30 floats
   = 120 bytes → 256-thread cube uses 30 KB shared memory, well within
   per-SM limits.

**Storage delta** per scale per channel after fusion:
- Drops: `sigma11_full, sigma22_full, sigma12_full, mu1_full, mu2_full`
  (5 planes), shared `t_scratch` slot, `ref_xyb_t`, `dis_xyb_t`,
  `ssim, artifact, detail` (3 planes) = 11 planes dropped.
- Adds: 5 v_buf planes shared across channels.
- Net: -28 planes per scale (after × 3 channels and accounting for
  shared) = **~1.8 GB at 4096² scale 0**.

**Launch delta** per `compute_with_reference_with_mode`:
- Drops: blur_pass second v-pass × 5 × 3 = 15; transpose × 5 × 3 = 15;
  error_maps × 3; launch_sum_p4 × 9; = **42 launches per scale × 6
  scales = 252 launches dropped per call**.
- Adds: fused h-pass+features × 3 = 3 per scale × 6 = 18 launches.
- Net: -234 launches per call. HtoD/iter: 52 → ~12.

### Revised scope estimate

This is genuinely the original assessment's "medium risk, ~1000 LOC"
shape — just with a different kernel pattern than the zensim fused
verbatim port. The new H-pass IIR kernel is ~120 LOC (mirror of the
existing blur_pass_kernel with X/Y swapped); the fused h-pass+features
is ~250 LOC (the new IIR-state ×5 walk + per-pixel features).

**Decision after revision**: ship the easy fix (downscale; done).
Document this assessment update. Pursue the H-pass IIR kernel + the
fused h-pass+features as the architectural fix when the dedicated
time budget is available — it's a larger change than the original
"copy zensim verbatim" path implied. The user's "1-2 days estimate"
in the original task brief was based on the zensim verbatim port; the
revised estimate is 2-3 days because of the new IIR kernel + 5-plane
fusion. The score-parity gate is more demanding: bit-identical IIR
behavior must be preserved through the new kernel boundary.

### Refined commit plan (REVISED 2 — post-attempt landing analysis)

The original 4-commit plan presumed a zensim-verbatim port. After
attempting the wire-in (this session), here's what actually shipped
and what got blocked.

**Shipped:**

1. **`perf(ssim2-gpu): port 3-channel-fused downscale`** — commit
   `3ac28448`. Bit-identical scores; saves 10 launches/call.
2. **`perf(ssim2-gpu): add H-pass IIR blur kernel`** — commit
   `d3d1c3ec`. ~120 LOC kernel mirroring `blur_pass_kernel` with
   X/Y axes swapped. **Parity gate: bit-identical** (max |diff| = 0)
   vs `v+t+v` at 7 sizes including 4096² — verified by the new
   `examples/blur_h_pass_parity`. The
   `blur_plane_two_pass_iir_untransposed` + `blur_plane_two_pass_untransposed`
   helpers land as `#[allow(dead_code)]` building blocks.

**Blocked at wire-in:**

The h-pass kernel is **NOT** wired into the production compute paths
in this session. The blocker is the shared `_full` plane slots
(`sigma11_full`, `sigma22_full`, `sigma12_full`, `mu1_full`,
`mu2_full` per Scale) between three paths that must agree on
orientation:

- `set_reference` writes the cache (`mu1_full`, `sigma11_full`).
- `compute_with_reference_with_mode` reads the cache + writes dis-side
  blurs.
- `Ssim2Batch::compute_batch` reads the cache + uses its own
  `error_maps_broadcast_batched_kernel` which is committed to
  **transposed-orientation** broadcast indexing.
- `process_scale` (non-cached `compute`) re-writes the cache for the
  ref-side blurs (because the slots are shared with set_reference's
  cache). The test `cached_reference_matches_direct_all_modes`
  guards against the patterns interleaving — if `compute_with_mode`
  is called between `set_reference` and `compute_with_reference_with_mode`,
  the cache is clobbered.

If only some of those paths switch to untransposed while others stay
transposed, the cache is interpreted inconsistently and scores break.
The whole set must move together. The dependency chain to migrate:

1. Port `error_maps_broadcast_batched_kernel` (in
   `crates/ssim2-gpu/src/kernels/error_maps.rs`) to untransposed
   orientation — its broadcast math assumes inner-stride = full_h
   (transposed). The change: inner-stride = full_w (untransposed),
   adjust the `idx % plane_stride` / `idx / plane_stride` math.
   **Estimated ~50 LOC + a fresh batched parity test.**
2. Port `pipeline_batch.rs::Ssim2Batch::compute_batch` to use the
   untransposed dispatcher. **Estimated ~80 LOC.**
3. Flip `set_reference`, `process_scale`, `compute_with_reference_with_mode`
   to use the untransposed dispatcher. **Estimated ~150 LOC.**
4. Drop `s.ref_xyb_t`, `s.dis_xyb_t`, `s.t_scratch` from `Scale`
   (now unused). Drop `cache_s.ref_xyb_t_full` from
   `StripCachedRefScale`. **Estimated -200 LOC.**

Steps 1+2 are independent of the ssim2 perf-fix work — they're
isolated changes to the Ssim2Batch infrastructure. Steps 3+4 are
the actual perf-fix wire-in.

Estimated total: 1-2 more days of work, mostly in step 1's batched
parity test (which exercises the broadcast math at the edges).

**Strip-mode mode-E** stays on the transposed contract for the
foreseeable — its `error_maps_strip_from_full_ref_kernel` indexes
the full ref buffer by `(outer * full_h + inner_full)` which is the
transposed-orientation walk. Migrating that path is a separate
refactor (the `inner_offset` math assumes transposed layout).

### Honest-stop notes for this session

What got shipped this session:
- The 3-channel-fused downscale port (`3ac28448`, perf neutral on VRAM
  but -10 launches/call).
- The H-pass IIR kernel + bit-identical parity gate (`d3d1c3ec`).
- The orientation-aware error_maps + untransposed building-block
  helpers (this commit). All `#[allow(dead_code)]` until the
  Ssim2Batch migration lands.

What did NOT ship:
- The 5-plane fused-features kernel (the original assessment's
  "Fix #1" — the 3 GB VRAM drop). Requires either: an IIR-respecting
  per-thread accumulator that maintains 5×6=30 floats of state +
  reads 7 input planes (5 v-pass outputs + raw ref_xyb + raw dis_xyb),
  OR the much bigger refactor of running v-pass on all 5 planes
  serially per channel and then a fused-h-pass+features kernel.
  Estimated 1-2 days standalone.
- The h-pass production wire-in. Blocked on Ssim2Batch migration
  (~1-2 days first).

The user's "1-2 day estimate" in the task brief was based on the
zensim-verbatim port, which doesn't apply because of the
IIR-vs-FIR algorithmic difference. Realistic total to land the
full perf-fix vision: 4-6 days from where the work currently stands.
