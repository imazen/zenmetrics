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
| Composed pipeline  | `Cvvdp::score` → `compute_dkl_jod` (GPU, tick 213); `compute_dkl_jod_with_warm_ref` (GPU batch); `compute_dkl_jod_host_pool` + `..._with_warm_ref` (cpu backend, tick 208/212); `host_scalar::predict_jod_still_3ch` (host-only reference) | full GPU path: color → weber → CSF → masking → pool → host fold. The public `Cvvdp::score` API now routes through this path after tick 207 tightened the manifest parity tolerance to 0.005 JOD and tick 213 made the switch. Host-pool variants run on `cubecl-cpu` where the GPU atomic pool kernel is unsupported. 12 MP CUDA timing (RTX 5070, ceil-div tick 175+): cold `jod` ~62 ns/px; warm-ref ~34 ns/px. vs canonical **pycvvdp v0.5.4 CUDA: 14 ns/px** — we are **4.4× slower cold / 2.4× slower warm** with correct output. pycvvdp benefits from cuDNN-optimised separable conv; cubek path or shared-memory tiling can close the gap. We win on portability: WGPU + HIP backends, 50 MB static binary vs ~3 GB PyTorch runtime. | host + GPU: **≤0.005 JOD vs pycvvdp v1 manifest** (`shadow_jod`, tick 207; was loose 0.05/0.5 schedule before ticks 204/206 closed the chroma_shift and 73×91 drifts). Measured max diff 0.0031 JOD across q=1,5,20,45,70,90; q=1 closed from 0.4 → 0.0000. Synth fixtures: 12 MP 0.000012 JOD, 256² blur3x1 0.000172 / blur1x3 0.000161 / noise 0.000048 / chroma_shift 0.000000 JOD, 73×91 odd-dim 0.000001 JOD (tick 206 replicated pycvvdp's gausspyr_reduce parity-check bug). warm-ref vs cold-ref ≤1e-5 JOD; host_pool vs GPU pool 0.000000 (tick 208); cpu-runtime host_pool vs pycvvdp 0.000001 (tick 223) |

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

- **(Resolved tick 206)** cvvdp bug: column-parity check in
  `gausspyr_reduce`. Line ~206 of cvvdp v0.5.4's
  `lpyr_dec.gausspyr_reduce` checks `x.shape[-2] % 2` (row count)
  when deciding the horizontal-pass right-column edge fix-up — the
  variable being patched is `y[...,:,-1]`, the rightmost column, so
  the parity check should clearly use `x.shape[-1] % 2` (column
  count). Comments say "odd number of columns"; the code tests
  rows.

  Affects mixed-parity inputs at any pyramid level (e.g. 6×5 → 3×3
  at level 4→5 of the 73×91 pyramid; 46×37 → 23×19 at level 1→2);
  doesn't affect same-parity pyramid levels (256² + 4 MP corpus
  hits this exclusively).

  Tick 206 fix: `gausspyr_reduce_scalar` rewritten from pure
  reflection to zero-pad + explicit pycvvdp-bug-compatible patches.
  GPU `downscale_kernel` keeps the reflect-based main path (matches
  pycvvdp on all same-parity inputs) and adds a delta correction at
  the right column when `sw` and `sh` parities differ. Result:
  `compute_dkl_jod_matches_pycvvdp_at_73x91_odd` passes at
  `diff = 0.000000` (was 0.006 before fix). Re-evaluate when the
  cvvdp pin moves.

  `gausspyr_expand_scalar` uses cvvdp's explicit edge-replication
  scheme (`interleave_zeros_and_pad`); constant-signal test passes
  across the whole buffer.
- **(Resolved)** Per-band CSF weight precomputation chose the
  flat-upload form. `Cvvdp::new_with_geometry` uploads one
  `Vec<[Handle; N_CHANNELS]>` per pyramid level (the 32-entry
  per-channel logs_row LUT), all `n_levels × 3` handles allocated
  once and reused across calls. A per-band tensor form would offer
  no functional benefit and adds Handle-juggling.

- **(Resolved tick 204)** Baseband CSF rho override. pycvvdp's
  `process_block_of_frames` overrides `rho_band[-1] = 0.1` cy/deg
  for the CSF lookup at the baseband (`cvvdp_metric.py:628`),
  separately from the geometric `lpyr.band_freqs`. Our pipeline
  was using the geometric value (0.190 at 256² standard_4k) — a
  0.117 JOD drift on the chroma_shift fixture that drove the
  ticks 191-203 investigation. Closed by adding
  `kernels::csf::CSF_BASEBAND_RHO = 0.1` and applying it at the
  baseband in both `host_scalar::predict_jod_still_3ch` and
  `Cvvdp::new`'s `logs_row` pre-upload. After the fix
  `compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift` passes
  at 0.000000 diff. See `docs/CHROMA_DRIFT_INVESTIGATION.md` for
  the full investigation timeline.

- **(Resolved tick 208)** Atomic-f32 pooling on cubecl-cpu.
  `pool_band_kernel` uses `Atomic<f32>::fetch_add` and is parity-
  tested via `gpu::pool_band_kernel_matches_host_lp_norm_mean` +
  `compute_dkl_jod_matches_host_scalar`. cubecl-cpu lacks
  `Atomic<f32>::fetch_add`; instead of porting a per-block partial-
  tree reduction, tick 208 added `Cvvdp::compute_dkl_jod_host_pool`
  which reuses `compute_dkl_d_bands` to read D bands back to host
  then pools with the host-scalar `lp_norm_mean`. Same JOD output
  to f32 precision on GPU backends (parity test
  `compute_dkl_jod_host_pool_matches_compute_dkl_jod` reports
  `diff = 0.000000`). Use it on cubecl-cpu where the atomic path
  fails; GPU backends should keep `compute_dkl_jod` for the small
  partials readback.

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

  Tick 213 update: `Cvvdp::score_with_reference` now routes
  through the GPU `compute_dkl_jod` cold-ref path (matches
  `Cvvdp::score`'s switch). For the dedicated warm-ref fast path
  callers use `Cvvdp::warm_reference` + `compute_dkl_jod_with_warm_ref`
  explicitly. For the all-host scalar reference (no GPU runtime
  needed), call `host_scalar::predict_jod_still_3ch` directly.

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
