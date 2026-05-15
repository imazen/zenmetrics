# cvvdp-gpu port status

Tracking faithful-port progress against the Python reference
(`gfxdisp/ColorVideoVDP`). One row per pipeline stage.

| Stage              | Module                 | Status                                   | Parity check                              |
|--------------------|------------------------|------------------------------------------|-------------------------------------------|
| sRGB → linear      | `kernels/color`        | host scalar + cubecl kernel body         | host 2e-3 vs pycvvdp; GPU 3e-5 vs scalar  |
| Display model      | `kernels/color`        | fused into host scalar + kernel          | same                                      |
| RGB → DKL          | `kernels/color`        | fused into host scalar + kernel          | same                                      |
| Laplacian pyramid  | `kernels/pyramid`      | host scalar + cubecl kernels             | pycvvdp 3 bands + cuda kernels parity     |
| Weber-contrast pyr | `kernels/pyramid`      | host scalar + fused `subtract_weber_3ch_kernel` (3ch + log_l_bkg one launch) + `baseband_divide_3ch_kernel` (GPU baseband finishing, no host roundtrip) | scalar via shadow_jod; 14-pt + fused-kernel parity + direct `baseband_divide` unit test |
| CSF weighting      | `kernels/csf`          | scalar + fused 3ch + 6ch (REF+DIST one launch) kernels | scalar + per-pixel + 3ch + 6ch parity all green |
| Contrast masking   | `kernels/masking`      | scalar + fused 3ch min_abs + 3ch PU blur with folded scale + mult_mutual_3ch + diff_abs_3ch (baseband) | scalar + 3ch + with-blurred + diff_abs parity |
| Per-band pooling   | `kernels/pool`         | GPU `pool_band_kernel` (atomic f32 partials) consumed by `compute_dkl_jod` | 3 host fixtures + GPU vs lp_norm_mean     |
| Host fold / JOD    | `kernels/pool`         | host scalar `do_pooling_and_jod_still_3ch` + `met2jod` over a ~144-byte partials Vec | 3 fixtures + kink continuity              |
| Composed pipeline  | `Cvvdp::compute_dkl_jod` (GPU) + `Cvvdp::compute_dkl_jod_with_warm_ref` (GPU batch) + `host_scalar::predict_jod_still_3ch` (CPU reference) | full GPU path: color → weber → CSF → masking → pool → host fold; CPU path retained as parity-locked reference. `Cvvdp::score` still routes through CPU per `shadow_jod` v1 manifest anchor. 12 MP CUDA timing (RTX 5070): cold `jod` **36.1 ns/px**; warm-ref **20.6 ns/px**. vs canonical **pycvvdp v0.5.4 CUDA: 14 ns/px** — we are **2.58× slower cold / 1.47× slower warm** than the reference (pycvvdp benefits from cuDNN-optimised separable conv on the downscale/upscale pyramid; see `benchmarks/pycvvdp_12mp_cuda_2026-05-14.md`). We win on portability: WGPU + HIP backends, 50 MB static binary vs ~3 GB PyTorch runtime, ~1 s warm-up vs 1–13 s graph compile. | host: ≤0.01 JOD vs pycvvdp v1 manifest (`shadow_jod`); GPU: matches host within f32 precision at q≥20 (`compute_dkl_jod_matches_host_scalar`), ~0.4 JOD cumulative drift at q=1 through `met2jod`'s steep slope (`shadow_jod_gpu` anchor); warm-ref vs cold-ref parity ≤1e-5 JOD (`compute_dkl_jod_with_warm_ref_matches_unwarm_path`) |

## Reference version pin

`gfxdisp/ColorVideoVDP` **v0.5.4** (latest tag as of 2026-05-14).
Driver script in `scripts/cvvdp_goldens/` runs `pycvvdp==0.5.4` to
produce parity goldens. When bumping: also bump the R2 prefix
(`v1` → `v2`), the `GOLDEN_VERSION` const in `tests/common/mod.rs`,
and the version assertion in `tests/parity.rs`.

The cvvdp parameter JSON gets vendored into
`crates/cvvdp-gpu/data/cvvdp_v0.5.4.json` once the script lands (small
~5 KB file, safe to commit) and loaded through `params::CvvdpParams`.

## Out of scope (v0)

- Video / temporal channels (sustained + transient).
- Foveation / gaze maps.
- HDR display models — sRGB-std only for the initial parity pass.

## Open questions

- **(Resolved tick 21)** Phase-uncertainty Gaussian blur in
  masking. cvvdp's σ=3 separable Gaussian for bands > 6×6 is now
  applied via `mult_mutual_band` + `phase_uncertainty_band`.
  Closed by replicating torchvision's `GaussianBlur(13, 3.0)`
  kernel + reflect padding. Whole-image parity gate via `shadow_jod`
  closed ~0.5-1.5 JOD of the gap.

- **(Resolved tick 24)** cvvdp v0.5.4 uses `weber_contrast_pyr` for
  the `contrast = "weber_g1"` config. Ported as
  `kernels::pyramid::weber_contrast_pyr_dec_scalar`; the shadow JOD
  on the corpus now matches pycvvdp within 0–0.7 JOD across q1–q90
  (was 1.4–1.7 before this tick). The shadow now slightly
  *overshoots* pycvvdp at low q — see `band_mul = 2.0` below.

- **(Resolved tick 25)** `lpyr.get_band` multiplies non-edge
  Laplacian bands by 2.0. Applied at the host_scalar consumption
  site as a `band_mul` scaling — keeps the Weber-pyramid storage
  canonical, mirrors cvvdp's readout pattern.

- **(Resolved tick 25)** Baseband bypass formula
  (`|T_f - R_f| * S`, no masking, no CH_GAIN). Wired in
  host_scalar; the Weber-pyramid magnitudes work cleanly with this
  formula (no 100× blow-up the tick-23 vanilla-Laplacian attempt
  hit).

- **cvvdp bug: column-parity check in `gausspyr_reduce`.** Line 206
  of cvvdp v0.5.4's `lpyr_dec.gausspyr_reduce` checks
  `x.shape[-2] % 2` (row count) when deciding the right-column edge
  fix-up — the variable being patched is `y[...,:,-1]`, the
  rightmost column, so the parity check should clearly use
  `x.shape[-1] % 2` (column count). Doesn't affect the
  zenmetrics-corpus (all 2^k square inputs through the pyramid),
  but will cause a divergence on non-square inputs at odd-height-
  but-even-width levels. To preserve bit-stable parity our port
  reproduces the bug verbatim; document it here and re-evaluate when
  the cvvdp pin moves.

  Status: pure-symmetric-reflection happens to be equivalent to
  cvvdp's `zero-pad + explicit edge patches` for even-input dims, so
  `gausspyr_reduce_scalar` matches cvvdp exactly on the corpus's
  pyramid levels. `gausspyr_expand_scalar` now uses cvvdp's explicit
  edge-replication scheme (`interleave_zeros_and_pad`) so the
  constant-signal test passes across the whole buffer.
- **(Resolved)** Per-band CSF weight precomputation chose the
  flat-upload form. `Cvvdp::new_with_geometry` uploads one
  `Vec<[Handle; N_CHANNELS]>` per pyramid level (the 32-entry
  per-channel logs_row LUT), all `n_levels × 3` handles allocated
  once and reused across calls. A per-band tensor form would offer
  no functional benefit and adds Handle-juggling.

- **(Resolved for cuda + wgpu; open for cubecl-cpu)** Atomic-f32
  pooling. `pool_band_kernel` uses `Atomic<f32>::fetch_add` and is
  parity-tested via `gpu::pool_band_kernel_matches_host_lp_norm_mean`
  + `compute_dkl_jod_matches_host_scalar`. cubecl-cpu still lacks
  `Atomic<f32>::fetch_add` (same gap zensim-gpu hits), so the cpu
  backend can't run the pool path. A per-block partial-tree
  reduction would be the cpu-backend port if/when that runtime
  becomes necessary.

- **(Resolved tick 157)** Wasted-readback discard pattern. Several
  public helpers (`compute_dkl_planes`, `compute_dkl_gauss_pyramid`,
  `compute_dkl_laplacian_pyramid`, `compute_dkl_weber_pyramid`) were
  called internally by downstream stages with `let _ = ...`, paying
  for full host transfers that immediately got dropped. Each gained
  a private `_dispatch_*_gpu` sibling that does the GPU launches
  and leaves the data on `gauss_ref` / `bands_ref` handles.
  Internal callers now use the dispatch-only path; the public
  functions wrap them with the readback for test/external use.
  Net saving per JOD at 12 MP: ~460 MB of GPU→host transfer that
  would otherwise be allocated and discarded.

- **(Resolved tick 170)** Cached-reference fast path. The
  `score_with_reference` doc had promised "the fast path lands
  when GPU composition stops re-running the reference side"
  since v0.0.1. Now real:
  - `Cvvdp::warm_reference(ref_srgb)` dispatches the REF weber
    pyramid once and caches the GPU state (`bands_ref[k]` and
    `weber_scratch[k].log_l_bkg` planes + the
    `log_l_bkg_baseband` scalar in `warm_ref_baseband_log_l_bkg`).
  - `Cvvdp::compute_dkl_jod_with_warm_ref(dist_srgb, ppd)` skips
    the REF half of the JOD pipeline. Same JOD output as
    `compute_dkl_jod(ref, dist, ppd)` within 1e-5 absolute tol.
  - Per-DIST throughput at 12 MP: cold path 36.1 ns/px → warm
    path 20.6 ns/px (42.9% saved, 1.75× faster per call).
  - Warm state is invalidated automatically by any method that
    dispatches REF weber (`compute_dkl_jod`, `compute_dkl_d_bands`,
    `compute_dkl_weber_pyramid`, etc.); `Error::NoWarmReference`
    surfaces clearly when the cache is cold.

  `Cvvdp::score_with_reference` (the public host-scalar path) is
  unchanged — it still routes through `predict_jod_still_3ch`
  for manifest-precise scoring. Users opting into GPU drift for
  speed call `compute_dkl_jod_with_warm_ref` explicitly.

- **(Open, tick 159)** 3-channel `upscale_v_3ch_kernel` /
  `upscale_h_3ch_kernel` fusion regressed ~4% jod at 12 MP on
  RTX CUDA across two runs. Same fusion pattern as the
  `weber_contrast_compute_3ch` and `subtract_weber_3ch` kernels
  that won 3-5%, but those did *math fusion* (shared
  per-pixel arithmetic across channels) while upscale just
  reads from 3 separate arrays at the same indices. The 3ch
  upscale's per-thread register footprint reduced warp-level
  latency hiding more than launch-overhead reduction was
  costing us. Future attempts should change the memory access
  pattern (e.g. shared-memory tiling that loads a coarse tile
  once and serves multiple destination pixels), not just
  rearrange the work across kernels. See the breadcrumb in
  `kernels/pyramid.rs` immediately after
  `subtract_weber_3ch_kernel`.
