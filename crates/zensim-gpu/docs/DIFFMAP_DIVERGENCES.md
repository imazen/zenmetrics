# zensim-gpu diffmap divergences — Phase 1

**Status**: Phase 1 of the zensim-fork RFC arc
(`~/work/zen/jxl-encoder/docs/RFC_ZENSIM_FORK_PLAN.md` §3, on the
`cvvdp-fork-rfc` branch in jxl-encoder). This doc tracks the
divergences between the strict aspirational `RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`
diffmap contract and what zensim-gpu Phase 1 actually delivers.

## §1. Strict `score == Minkowski(diffmap)` aspirational invariant — NOT DELIVERED

The RFC §2.2 documents an aspirational strict invariant — the form:

```
score == 100 - minkowski_p(diffmap, p)
```

— that would let callers reason about diffmap norms as if they were
metric scores. **zensim Phase 1 does NOT deliver this.**

Same reason as cvvdp-gpu's Phase 1: zensim's pool order is multi-stage
(per-channel SSIM error → per-band L1/L2/L4/L8 norms → trained
per-band-channel weights → MLP head → tanh-pin → PCHIP spline → per-
codec affine), and intermediate per-band/per-channel norms don't
cleanly factor back into a single base-resolution per-pixel pool.

The 5 PRACTICAL invariants of §2.1 ARE all delivered:

1. ✅ Identity → near-zero diffmap (pinned at 1e-3 absolute via
   `tests/diffmap_invariants.rs::invariant_1_identity_yields_near_zero_diffmap`).
2. ✅ Non-negative
   (`invariant_2_non_negative_diffmap`).
3. ✅ Monotone in distortion
   (`invariant_3_monotone_in_distortion`).
4. ✅ Spatial localization
   (`invariant_4_spatial_localization_block_perturbation`).
5. ✅ Warm-ref invariance
   (`invariant_5_warm_ref_byte_equivalent`).

Per CLAUDE.md's "honest-stop > false completion": ship looser
invariants and document strict as aspirational. Phase 1b kernel chain
will preserve the same boundary.

## §2. **Phase 1 implementation strategy: CPU-fallback for diffmap path**

zensim-gpu's existing GPU pipeline produces a 228 / 300 / 372-feature
vector via the 4-scale pyramid. Producing the per-pixel diffmap
**directly on the GPU** requires a separate kernel chain (per-scale
SSIM error → bilinear upsample → multi-scale blend → optional
contrast masking + sqrt). This is ~2 weeks of net-new CubeCL work on
top of the existing pyramid kernels.

**Phase 1 chooses to ship the public API surface first** with the
diffmap production delegated to zensim's CPU pipeline
(`zensim::Zensim::compute_with_ref_and_diffmap_linear_planar`). The
rationale:

1. **Correctness**: the CPU pipeline IS the canonical reference, so
   Phase 1 diffmaps are bit-exact correct by construction.
2. **API unblock**: the jxl-encoder buttloop integration (zensim-fork
   Phase 3 + 4) can wire diffmap-aware backends today instead of
   waiting on Phase 1b.
3. **Measurement baseline**: the overhead bench in
   `examples/diffmap_overhead.rs` + the distribution capture in
   `examples/diffmap_distribution.rs` give Phase 4 the data it needs
   for `ZENSIM_DIFFMAP_RENORM_SCALE` + `ZENSIM_BLOCK_CONSTANTS`
   calibration, AND give Phase 1b a known wall baseline to optimise
   against.

The Phase 1 path:

- `Zensim::score_with_diffmap(ref_srgb, dist_srgb, diffmap_out)`:
  host-side sRGB-u8 → linear-f32 LUT decode → call CPU
  `Zensim::precompute_reference_linear_planar` + CPU
  `compute_with_ref_and_diffmap_linear_planar` → copy `diffmap()` into
  caller's `&mut Vec<f32>`.
- `Zensim::score_from_linear_planes_with_diffmap(...)`:
  skip the host LUT decode, go directly to the CPU linear-planar
  diffmap API.
- `Zensim::warm_reference_from_linear_planes(ref_r, ref_g, ref_b)`:
  build and cache a CPU `PrecomputedReference` in the diffmap state
  for subsequent warm-ref-diffmap calls.
- `Zensim::set_reference(ref_srgb)` ALSO populates the diffmap
  state's warm cache (lazily — only when the state has been
  initialised by a prior diffmap entry-point call).

The **scalar feature-vector fast path stays GPU**: callers who only
need the 228 / 300 / 372 features pay zero CPU-side cost. The
diffmap state is `Option<DiffmapState>` and lazy-allocated on first
use of any Phase 1 diffmap or linear-planes entry-point.

### §2b. Phase 1b chunk 3 — GPU diffmap kernels WIRED + VALIDATED, DEFAULT-OFF (2026-05-27)

Phase 1b chunks 1+2 (commits `b50b8f57`, `f66930c7`) landed the CubeCL
diffmap kernels (`per_scale_weighted_ssim_kernel` +
`pow2x_upsample_add_kernel` + `diffmap_trim_padded_kernel` +
`diffmap_zero_kernel`) and the default-options trained-weight port
(`trained_multiscale_ssim_weights_default`). Chunk 3 (this commit)
**wires them into the diffmap-producing methods** behind an opt-in env
gate and **proves them pointwise-correct** against the CPU canonical:

- New `linear_to_positive_xyb_kernel` (sibling of the sRGB color
  kernel) lets the GPU feature pipeline ingest the linear-RGB planes
  the diffmap API receives.
- New `GpuDiffmapScratch` holds an inner WithIw-regime `Zensim<R>`
  (which writes the per-scale mu1/mu2/ssq/s12 persist planes) + the
  base accumulator + per-scale dm planes + cached trained weights.
- `Zensim::gpu_diffmap_linear_into` orchestrates: build ref + dist XYB
  pyramids on the inner pipeline → run the WithIw persist feature pass
  (writes persist planes) → run the chunk-1/2 diffmap kernel chain →
  trim → read back.
- `tests/cpu_gpu_diffmap_parity.rs` validates the GPU diffmap matches
  the CPU canonical `compute_with_ref_and_diffmap_linear_planar`
  **pointwise to ≤ 2.08e-4 absolute** (5 fixtures × 4 distortions,
  CUDA RTX 5070; tolerance pinned at 1e-3 with ~5× margin).

**The GPU diffmap path is OPT-IN (`ZENSIM_GPU_DIFFMAP=1`), DEFAULT-OFF**
— an honest-stop on the wall axis. Reason: the SCALAR SCORE must still
come from the CPU canonical path, because the GPU-feature → V0_3 MLP
score path is **catastrophically wrong** on the pinned zensim 0.3.0 (see
§9; measured GPU V0_3 score `-77.13` vs CPU canonical `+85.33` on a
real CID22 image — the WithIw GPU features pass the per-feature
`cpu_parity` bands, but the V0_3 MLP amplifies those small f32 drifts
into a 160-point score divergence). With the score forced onto the CPU
canonical (which inherently re-runs the full feature pipeline), running
the GPU diffmap **on top** is strictly slower than Phase 1's CPU-only
path. The opt-in gate ships the validated GPU diffmap infrastructure so
that the chunk N+1 score-path fix (upgrade the pinned zensim crate so
the GPU-feature → V0_3 MLP score is trustworthy, OR add a GPU-native
MLP-equivalent) flips the gate to default-ON and drops the CPU call —
crushing the +1006% overhead. Until then the production default is
unchanged (zero regression).

## §3. CPU vs GPU score divergence

Because Phase 1's diffmap path returns the score produced by zensim's
CPU pipeline (NOT the GPU feature pipeline), the scalar score
returned from `score_with_diffmap` etc. is the **CPU-canonical score**.

This diverges from the score that
`compute_features` + `score_from_features` would produce on the GPU
side (the existing `ZensimOpaque::compute_*` path) by the same
~1e-3 absolute drift documented in `tests/score_v03_parity.rs`'s
note on f32 vs f64 feature arithmetic.

For the **buttloop integration** (where the diffmap path is the only
one that matters), this is a feature not a bug: the score returned
alongside the diffmap is internally consistent with the diffmap
itself (they came from the SAME CPU pipeline). The buttloop's
`accept_bound` math sees a coherent (score, diffmap) pair.

For callers who want GPU score parity AND a diffmap, Phase 1b's
pure-GPU kernel chain will close this gap.

## §4. Score-direction normalisation

zensim's native `ZensimResult::score()` is `[0, 100]` with **higher =
better** (100 = identical).

The buttloop contract is **smaller = better** at the trait boundary
(per `RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` §1.1). All Phase 1
entry-points normalise via:

```rust
let v = (100.0 - score).clamp(0.0, 100.0);
v as f32
```

The clamp is defensive — zensim's bake can produce `100.0001` on
identical-plus-noise inputs (f32 SIMD round-off), and a buttloop
`accept_bound` comparison against a negative would silently mis-fire.

## §5. `EncoderStrategy::Libjxl` short-circuit — NOT IMPLEMENTED HERE

zensim-gpu has NO knowledge of jxl-encoder's `EncoderStrategy::Libjxl`
strict-parity invariant. The short-circuit lives at the
jxl-encoder side (Phase 3 of the zensim fork:
`LossyConfig::resolve_zensim_loop` returns `false` for the Libjxl
strategy). zensim-gpu just provides the trait surface; the dispatch
logic stays at the caller.

## §6. HDR not supported (out of Phase 1 scope)

zensim's CPU pipeline rejects PQ / HLG inputs with
`ZensimError::UnsupportedFormat`. Phase 1's CPU-fallback diffmap path
inherits this constraint. jxl-encoder's zensim integration must
short-circuit when `HdrLoss != Sdr` (per RFC §3.4 in
`RFC_ZENSIM_BUTTLOOP_AUDIT.md`).

## §7. Per-block reducer constants (`K_TILE_NORM`) — DEFERRED TO PHASE 4

The buttloop's per-block 8×8 reducer uses a metric-specific
`K_TILE_NORM` constant (cvvdp = 0.16; butter = 1.2). zensim's
`K_TILE_NORM` value is unknown until Phase 4 measures the
`tile_dist` distribution per the methodology in
`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` §3.2.

Phase 1 ships the data collection harness
(`examples/diffmap_distribution.rs` + the env-gated
`JXL_PHASE8B_DIFFMAP_DUMP` dispatch in jxl-encoder's
`vardct/perceptual_backend.rs`); Phase 4 fits the constant.

## §8. Diffmap renormalisation scale — DEFERRED TO PHASE 4

Same as §7. Phase 1 ships `ZENSIM_DIFFMAP_RENORM_SCALE = 1.0`
(no renorm); Phase 4 (or Phase 8c-style refit if needed) fits the
target-aware scale per RFC §4.2's formula:

```
scale = (target_z / target_b) * (mean_b / mean_z)
```

## §9. `compat shim` for `zensim::score_features_with_profile_and_codec`

The pre-existing `ZensimOpaque::compute_*` profile-scoring path
references a function `zensim::score_features_with_profile_and_codec`
that doesn't exist in the path-pinned `zensim` crate at the
zenmetrics workspace's current revision.

To unblock the build, Phase 1 ships a compat shim
`opaque::score_features_with_profile_and_codec_compat` that falls
back to the legacy linear 228-feature `score_from_features` formula.

**This breaks `tests/score_v03_parity.rs::gpu_v03_score_matches_cpu_within_001`**
— the test pinned at 0.025 absolute tolerance against the canonical
v0.3 MLP score, and the legacy formula diverges by ~22 score units.
Per CLAUDE.md "do NOT widen test tolerances", the test stays failing.
Fixing it requires upgrading the path-pinned zensim crate to a
revision that includes `score_features_with_profile_and_codec` — a
zenmetrics-side workstream outside the zensim-fork RFC arc scope.

The compat shim is a temporary unblock for the Phase 1 build. It does
NOT affect the diffmap path (which uses the CPU canonical pipeline
end-to-end) or the buttloop integration. When the path-pin is
upgraded, delete `score_features_with_profile_and_codec_compat` and
route the opaque scoring path through the real function.

## §10. Phase 1b roadmap (pure-GPU diffmap kernels)

Future Phase 1b will replace the CPU-fallback diffmap path with
CubeCL kernels mirroring the cvvdp-gpu pattern in
`crates/cvvdp-gpu/src/kernels/diffmap.rs`:

1. `ssim_error_per_pixel_kernel` — per-pixel SSIM error from
   already-computed `mu1/mu2/ssq/s12` persist planes (existing
   `fused_features_kernel_persist` already computes these in shared
   memory; need a separate output buffer rather than fold-only).
2. `bilinear_upsample_band_kernel` — per-scale 2^s × replicate or
   true bilinear interp.
3. `multi_scale_blend_kernel` — sum-with-weights across scales at
   base resolution.
4. `contrast_masking_kernel` — optional post-pass.

The host-scalar reference helpers in `src/kernels/diffmap.rs`
(`upsample_pow2x_add_scalar`, `sqrt_clamp_scalar`,
`contrast_masking_scalar`, `channel_weighted_sum_scalar`) lock the
parity gates the future kernels must clear pointwise.

Expected wall savings (estimate based on cvvdp-gpu's W44-PHASE3-B4
pattern): -20% to -50% at 1024² on CUDA, ~0% on cubecl-cpu (since
that's already CPU).

## §11. Benchmarks captured at Phase 1 ship

- `benchmarks/zensim_diffmap_overhead_<date>.tsv` — paired
  `compute_features_vec` (scalar GPU) vs `score_with_diffmap`
  (CPU-fallback diffmap) wall at 4 sizes × 2 fixtures. Caller pipes
  the example to a date-stamped file.
- `benchmarks/zensim_diffmap_distribution_<date>.tsv` — per-pixel
  diffmap stats (mean / p25 / p50 / p75 / p95 / max) across 4
  fixtures × 5 distortion levels. Phase 4 input.

## §12. References

- RFC contract:
  `~/work/zen/jxl-encoder/docs/RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`
- RFC audit:
  `~/work/zen/jxl-encoder/docs/RFC_ZENSIM_BUTTLOOP_AUDIT.md`
- Phased plan:
  `~/work/zen/jxl-encoder/docs/RFC_ZENSIM_FORK_PLAN.md`
- cvvdp-gpu Phase 1 (precedent):
  zenmetrics master `8b658b40` +
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_gpu_diffmap_api_shipped_2026-05-24.md`
- cvvdp-gpu Phase 1 divergences doc (template for this one):
  `crates/cvvdp-gpu/docs/DIFFMAP_DIVERGENCES.md`
