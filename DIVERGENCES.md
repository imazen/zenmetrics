# DIVERGENCES — workspace ledger of every known deviation from a reference

One row-set per metric, across every implementation version we track. This is
the **index** — each entry cites the crate-level doc that carries the detail.
When a new port lands or a reference version moves, update the ledger in the
same change. Companion ledger: `docs/METRIC_PROVENANCE.md` records *which*
reference each metric was produced from (paper + impl + version), the
provenance class (wrapper / reimplementation / clean-room / in-house), the
oracle, and the validation tolerances we gate.

Tags (same vocabulary as `crates/cvvdp/docs/UPSTREAM_DIVERGENCES.md`):

- **DIVERGES** — open difference vs the reference, intentional or accepted.
- **RESOLVED** — was a discrepancy during porting; fixed and verified. Kept
  for provenance because these are the subtle traps a re-port would hit again.
- **ORACLE-ARTIFACT** — the discrepancy was in the reference/oracle *side*
  (typing bug, broken detection), not in our port.
- **EXTENSION** — behavior the reference does not define (stride support,
  small-input handling, extra color paths).
- **OUT-OF-SCOPE** — reference feature deliberately not ported.

## Reference-version ledger

| Crate | Reference implementation + version | Oracle for goldens/parity | Worst observed delta |
|---|---|---|---|
| `cvvdp` | pycvvdp **v0.5.7** (gfxdisp/ColorVideoVDP); `PYCVVDP_REFERENCE_VERSION` | pycvvdp itself + cvvdp-conformance cells | 2e-6 JOD (u16 video cells), 1.2e-5 (stills); scalar parity 3e-6 JOD |
| `cvvdp-gpu` | pycvvdp **v0.5.7**; `parity-goldens` test pins **v0.5.4** manifest from R2 | pycvvdp goldens + in-tree `cvvdp` | see crate `docs/PORT_STATUS.md` |
| `hdrvdp` | official **HDR-VDP 2.2.2** MATLAB release (SourceForge) | Octave 11.1 run of official 2.2.2; UPIQ corpus | P_det 1.4e-5 rel, C_max 1e-4 rel, P_map 1.7e-2 abs, `res.Q` 7.8e-4 |
| `vmaf` | Netflix **libvmaf 3.2.1** (vendored via `vmaf-head-sys 0.2.0`, test-only) | libvmaf FFI oracle in `tests/ffi_fusion.rs` | v0 asserted ≤ 1e-4 features (motion2 1e-8), ≤ 0.02 score; integer stat paths exact by construction |
| `gmsd` | **libgmsd** (Ponomarenko group) + `GMSD.m` | libgmsd bit-comparison | bit-identical map on even dims; f64 rounding only in score |
| `iwssim` | **Python-IW-SSIM** @ `f9de37c` (Jack-guo-xy) | committed JSON goldens | identical ≤ 1e-5, distorted ≤ 5e-3; strip-vs-whole ≤ 1e-6 |
| `iwssim-piq` | same algorithm as `iwssim`; the **JPEG AIC-4 `IW-SSIM` column** (jpeg-ai-qaf `IW_SSIM` on unrounded Y) | published AIC-4 `metrics_fullres.tab` | med 3.9e-6, max 2.6e-5 over 54 pairs (2026-09-26) |
| `psnrhvs` | authors' **`psnrhvsm.m`** (metrix MATLAB) | Octave goldens, `validation/` | ≤ ~4e-4 dB (asserted ≤ 1e-3) |
| `psnrhvs` (daala module) | Daala/Xiph **`dump_psnrhvs`** as vendored in libvmaf `psnr_hvs` @ **f85a8536** (`vmaf-head-sys 0.2.0`) | libvmaf FFI oracle in `crates/msssim/tests/ffi_libvmaf.rs` | asserted ≤ 2e-4 dB, YUV420+YUV444; observed ≪ gate |
| `haarpsi` | authors' MIT **`HaarPSI.m`** | Octave goldens, `validation/` | ≤ 5e-5 over 16 rows |
| `fsim` | authors' **`FR_FSIMc.m`** (research license) | Octave goldens, `validation/` | ≤ 5e-8 over 18 rows |
| `vsi` | authors' **`VSI.m`** (author site, research license) | Octave + `pkg load image` goldens | ≤ 1e-4 over 14 rows |
| `msssim` | Wang's **`msssim.m`** (MAD_Competition archive) | Octave goldens, `validation/` | ≤ 8.8e-6 over 15 rows |
| `msssim` (libvmaf module) | libvmaf **`float_ssim`/`float_ms_ssim`** @ **f85a8536** (vendored via `vmaf-head-sys 0.2.0`, test-only) | libvmaf FFI oracle in `tests/ffi_libvmaf.rs` | asserted ≤ 2e-4; observed ≪ gate (8-bit, 10-bit, identical) |
| `vif` | authors' **`vifp_mscale.m`** (pixel-domain release) | Octave goldens, `validation/` | ≤ ~5e-13 over 17 rows |
| `vifvec` | authors' **`vifvec.m`** + matlabPyrTools `sp5Filters`/`buildSpyr`/`vifsub_est_M` (steerable-pyramid vecGSM release — a different metric from `vif` that shares the name) | GNU Octave run of the official `.m` set | ≤ 1e-9 over 10 goldens (incl. odd dims, 72px floor, identity, black-on-flat) |
| `mad-iqa` | Larson & Chandler **`hi_index.m`/`lo_index.m`** + `ical_std.c`/`ical_stat.c` (STMAD_2011, archived in Netflix/vmaf); JEI 2010 combine | Octave `.m` shims of the C-mex + official `.m` drivers | hi 2.4e-7, lo 1.8e-6, mad 1.3e-6 rel over 13 rows |
| `mdctpsnr` | **`thorfdbg/mDCTpsnr`** (Thomas Richter / U. Stuttgart; zlib-style license), compiled GCC `-O3 -ffast-math` + AVX2 + glibc 2.43 libmvec on x86_64 | the built `dctpsnr` binary + published AIC-4 `mDCT-PSNR` column | ~4e-6 dB on synthetic goldens incl. all `(w−13)%8` classes; AIC-4 53-pair max ~1.13e-5 dB (see entry) |
| `ssim2` (CPU) | external **`fast-ssim2`** crate (sibling repo; C++ SSIMULACRA2 parity) | fast-ssim2's own parity suite | owned by fast-ssim2 repo |
| `ssim2-gpu` | published **`ssimulacra2` 0.5** crate | CPU-reference parity tests | FIR path ~5e-5 vs IIR — see entry |
| `butteraugli` (CPU) | external **`butteraugli`** crate 0.9.4 (imazen fork, path dep) | the crate itself | external — not ported here |
| `butteraugli-gpu` | **`butteraugli` v0.9.2** parity target | CPU-reference parity tests | strip reduce order: ≤ 1e-4 rel |
| `dssim` (CPU) | external **`dssim-core`** — workspace pins `^3.4` (crates.io max); README table names 3.5 (unpublished local line) | the crate itself | external — not ported here |
| `dssim-gpu` | **`dssim-core` v3.4** | integration tests vs published crate | see crate README/`PORT_STATUS` |
| `iwssim-gpu` | in-tree **`iwssim`** (this repo) | CPU-vs-GPU parity | see entry — column-name split is deliberate |
| `zensim` (CPU) | in-house metric — **no external reference exists** | self | n/a — the crate *is* the definition |
| `zensim-gpu` | **`zensim` 0.2.8 / pinned 0.3.0** | `tests/cpu_gpu_diffmap_parity.rs` | ≤ 2.08e-4 pointwise; 12 documented items |

## Cross-cutting house conventions (diverge from every MATLAB reference by design)

1. **Unrounded luma.** House `*_rgb8`/`*_rgb` APIs compute unrounded f32
   luma — `0.2989/0.5870/0.1140` (mad-iqa, vif, msssim, psnrhvs BT.601) or
   `0.299/0.587/0.114` YIQ `Y` (fsim, vsi). MATLAB `rgb2gray`/`rgb2ycbcr`
   **round to uint8**; we do not. Goldens reproduce our coefficients
   explicitly in Octave rather than calling `rgb2gray`. **DIVERGES** — the
   house convention is strictly more accurate; scores differ from
   uint8-luma pipelines at ~1e-3 scale.
2. **Precision policy per crate.** f32 planes + f64 block/norm accumulation:
   psnrhvs, haarpsi, fsim, vsi, msssim, mad-iqa. f64 end-to-end: **vif**
   (the reference's 1e-10 degenerate masks presuppose f64's noise floor —
   f32 `E[x²]−μ²` cancellation at 255-scale leaves ~1e-3 residue and flips
   every mask). hdrvdp ran f64, then moved planes/intermediates to f32 for
   fleet throughput — the bit lock was retired and the official-golden
   tolerance test is now the gate. gmsd is bit-identical f32 + f64 score.
   vmaf reproduces libvmaf's integer/fixed-point paths exactly.
3. **SIMD bit-parity.** All archmage tiers (scalar/v3/v4/neon/wasm128) are
   bit-identical by construction in every new port — lane count never enters
   per-element expressions, reductions accumulate in a fixed scalar order.
   vmaf's v3 horizontal stats substitute lo16/hi16 u32 accumulation for
   libvmaf's `vpmuludq` 64-bit widening (unavailable in magetypes) — same
   exactness class, different codegen.
4. **Degenerate-input semantics preserved, not smoothed.** Where the
   reference emits `NaN`/`-Inf`/complex-real we emit the same value:
   fsim NaN on constant input, vsi NaN on flat, msssim real-part of
   `negative^λ`, mad-iqa NaN on `min(w,h) < 34` (edge-trim empties the
   block maps), vif NaN on flat reference / 0.999999999992 on identical,
   haarpsi NaN on zero-weight.
5. **Stride + dimension checks are explicit.** Rust APIs take strided
   planes and return typed `Err`; MATLAB takes dense matrices. **EXTENSION**
   everywhere; stride-invariance is tested.

## Per-metric records

### `cvvdp` (CPU) — vs pycvvdp v0.5.7

Deep doc: `crates/cvvdp/docs/UPSTREAM_DIVERGENCES.md` (7 open items),
`crates/cvvdp/docs/NAN_ON_IDENTICAL_INPUT.md`,
`docs/CVVDP_CONFORMANCE.md`.

- **DIVERGES** — Temporal channel (`Y_t`, `beta_t` pooling, 4th slots of
  `mask_q`/`xcm_weights`/`baseband_weight`): not ported; crate scoped to
  stills at authoring, video path added later without transient channel.
- **DIVERGES** — `cvvdp_ml_saliency` ONNX foveation weighting: not ported
  (torch/ONNX dep incompatible with `forbid(unsafe_code)`+`no_std`).
- **DIVERGES** — Display `exposure` field (multiplies linear light
  post-EOTF): not modeled; no named preset sets `exposure != 1`.
- **DIVERGES** — Alternative CSF LUTs and runtime CSF/masking parameter
  override: pinned to the reference's shipped tables only.
- **DIVERGES** — PU21 perceptual-uniform encoding path: not ported.
- **DIVERGES** — Color spaces beyond {sRGB, BT.2020, P3}: upstream accepts
  wider spaces; we accept the three.
- **RESOLVED** — CSF `log_rho` axis flat-clamp at high PPD (2026-05-26);
  several further items resolved in v0.1.0 — see the crate doc.

### `cvvdp-gpu` — vs pycvvdp + in-tree `cvvdp`

Deep docs: `crates/cvvdp-gpu/docs/DIFFMAP_DIVERGENCES.md`,
`PORT_STATUS.md`, `NAN_ON_IDENTICAL_INPUT.md`.

- **DIVERGES** — `score == Minkowski(diffmap)` does not hold strictly;
  the diffmap satisfies a documented weaker invariant set instead
  (DIFFMAP_DIVERGENCES §"What we implement instead").
- **DIVERGES** — `PerfMode::Strict` (the parity-test calibration target)
  and the `parity-goldens` manifest are pinned to pycvvdp **v0.5.4**,
  while the crate targets **v0.5.7**: measured status is ≤ 0.005 JOD vs
  v0.5.7 on the v2 R2 manifest, and the v1/v0.5.4 manifest agrees with
  v2 to 1.4e-6 JOD — a bounded, documented version delta, not an error.
- **EXTENSION** — v1 column names retained for back-compat with existing
  sidecars while the canonical sRGB parity tests run the current API.

### `hdrvdp` — vs official HDR-VDP 2.2.2

Deep docs: `crates/hdrvdp/docs/VALIDATION.md`,
`benchmarks/hdrvdp_upiq_2026-09-24.md`.

- **ORACLE-ARTIFACT** — upstream `reconSpyr.m`/`reconSpyrLevs.m` `is_mex`
  detection is broken under Octave (a `.m` path can never equal the
  2-char `'.m'`, so `is_mex` is always true and the code tries to call a
  mex that isn't there). Golden generation patches `is_mex = false`;
  this changes nothing numerically — the `.m` path is the reference
  algorithm — but it is required for the oracle to run at all.
- **DIVERGES** — f32 planes/intermediates (fleet-throughput pass). The
  byte-exact lock taken before optimization was deliberately retired;
  correctness gate is now the official-golden tolerance test.
- **DIVERGES** — UPIQ corpus: korshunov scores essentially exact
  (median |Δ| 0.013); the **narwaria** family shows a ~+2 residual in one
  `n-i07` reference family — presumed upstream protocol detail,
  unresolved. Aggregate SROCC 0.8203 vs official column's 0.8117 clears
  the published 0.812 bar.
- **DIVERGES** — `q` is the 2.2.2-published `res.Q` correlate; `q_mos`
  reproduces a logistic variant upstream *removed* — kept under a
  distinct name so columns are unambiguous.
- **OUT-OF-SCOPE** — HDR-VDP-3 (different pipeline, display-adaptive) not
  ported; crate tracks 2.2.2 only. GPU not planned (`parallel` CPU is the
  fleet answer).

### `vmaf` — vs libvmaf 3.2.1

- **DIVERGES** — v3 horizontal statistics use lo16/hi16 u32 accumulation
  (each strictly < 2^32, exact for all u32 inputs) in place of libvmaf's
  `vpmuludq` 64-bit widening — bit-exact results, different emitted code
  (`vmovdqu`/`vpmulld`/`vpaddd` ymm). See README SIMD paragraph.
- **DIVERGES** — v1 feature path remains scalar; only v0 VIF stats (all
  scales, horizontal + vertical) are SIMD. Measured in README.
- **DIVERGES** — `ModelVariant::V1_NEG` names the standard v1 model with
  the no-enhancement-gain limits the official v1.0.16 models already set;
  upstream has no separate v1-NEG family — the name is our disambiguation.
- **EXTENSION** — crate is API-only (not yet in `zenmetrics --metric` or
  the orchestrator); requires planar YUV420 8/10-bit ≥ 34×34 — no packed
  RGB, HDR, or 12/16-bit.

### `gmsd` — vs libgmsd

- **DIVERGES** — odd width/height: half-res grid is `⌊w/2⌋×⌊h/2⌋` with
  the trailing row/column dropped. libgmsd's `downsample_2x2` writes one
  element *past* that allocation on odd input (latent upstream bug);
  `GMSD.m` instead keeps an extra half-zero-padded sample. We follow the
  documented size. Even-dim map is bit-identical.
- **DIVERGES** — score accumulation is one-pass shifted f64 instead of
  libgmsd's two-pass mean/variance — f64-rounding-level score difference
  only (record in `benchmarks/gmsd_parity_2026-09-22.md`).

### `iwssim` — vs Python-IW-SSIM `f9de37c`

- **RESOLVED** (was a real porting bug, fixed 2026-09-26) — the IW
  weighting path fed the GSM eigendecomposition the next **Gaussian**
  level (`g_ref[s+1]`) as the parent band; both references — the
  authors' MATLAB `iwssim_rgb.m` (`pyrBand(pyro, pind, nband+1)`) and
  Python-IW-SSIM (`imgopr[scale+1]` = `pyr_coeffs[(scale,0)]`) — use
  the next **Laplacian** band. Corrected in `pipeline.rs`, `strip.rs`,
  `weights.rs`; strip/full parity re-verified (14 tests). The fix
  removed the systematic +1.2e-3 offset vs the AIC-4 `IW-SSIM` column
  (residual before ingress: ~2e-4).
- **EXTENSION** — `LumaConvention` selects the RGB→gray ingress:
  `Bt601Rounded` (default) reproduces Python-IW-SSIM's
  `utils.rgb2gray` (`round(0.2989R+0.5870G+0.1140B)` u8); `YiqUnrounded`
  is the piq/jpeg-ai-qaf convention (`0.299R+0.587G+0.114B`, no
  rounding — `IwssimParams::piq_luma()`). The AIC-4 `IW-SSIM` column
  used the latter (verified: `IW_SSIM_PyTorch.py` takes a caller-made
  luma plane; the qaf feeds it the unrounded 0–255 Y).
- **EXTENSION** — `allow_small` tiles inputs below the 176px floor up to
  it; the reference hard-requires ≥ 176 min-dim. Canonical `score_gray`
  keeps the reference's requirement.
- **EXTENSION** — `score_strip{,_gray}` bounded-height strip processing;
  strip-vs-whole self-divergence ≤ 1e-6 (`tests/strip_parity.rs`).
- Golden tolerances are looser than the measured agreement (identical
  ≤ 1e-5, distorted ≤ 5e-3 asserted) — the reference itself is a Python
  reimplementation of Wang & Li's MATLAB, so the tolerance reflects
  oracle fidelity, not port sloppiness.

### `iwssim-piq` — vs the JPEG AIC-4 `IW-SSIM` column

- Same code path as `iwssim` with `LumaConvention::YiqUnrounded`;
  exposed as a separate `MetricKind`/`IWSSIM_PIQ_COLUMN_NAME` so a
  results table never mixes ingress conventions in one column.
- **VALIDATED** vs `metrics_fullres.tab` `IW-SSIM` on the 54 available
  pairs (med |Δ| 3.9e-6, max 2.6e-5, 2026-09-26). The column was
  computed by the jpeg-ai-qaf harness calling the Jack-guo `IW_SSIM`
  port on the **unrounded** 0–255 Y plane (`metrics.py:IWSSIM.calc` →
  `convert_range(yuv['Y'], range, [0,255])`, kornia-matrix `rgb_to_yuv`,
  `a = 0.299`) — upstream mirror at
  `/home/lilith/tmp/jpeg-ai-qaf-hdrvdp3` @ `0628a6b`.
- **DIVERGES (deliberate)** from `iwssim`'s default only in luma
  ingress; algorithm, pyramid, and weights are shared code.

### `iwssim-gpu` — vs in-tree `iwssim`

- **DIVERGES** — emits under a **different column name** on purpose
  (`lib.rs` note): a future GPU port differing by ~1e-3 would not be
  wrong, so the distinct name documents that columns are not claimed
  bit-comparable to `iwssim`.

### `psnrhvs` — vs `psnrhvsm.m`

- **DIVERGES** — per-block DCT energies in f32 with a single f64 global
  accumulation → ≤ ~4e-4 dB (asserted ≤ 1e-3).
- **ORACLE-ARTIFACT** — the reference is integer-typed at some call
  paths; `gen_goldens_rgb.m` must `double(plane)` before calling, same
  trap as mad-iqa's gen (see below).
- House luma is unrounded BT.601 (cross-cutting #1).

#### `daala` module (`psnrhvs-daala`) — vs libvmaf `psnr_hvs` (Daala `dump_psnrhvs`)

- **RESOLVED** — `od_bin_fdct8x8` transform order. The C computes the 2-D
  bin-DCT as column-DCT → (implicit transpose of the intermediate) →
  column-DCT; each 1-D butterfly chain rounds to `od_coeff` (i32) between
  stages, so row-first is *not* numerically equivalent (~1.3% higher MSE,
  ~0.06 dB on the FFI plane). Port matches C's column-first sequence;
  parity ≤ 2e-4 dB.
- **EXTENSION** — `DaalaPlane::{Y,Cb,Cr}` selects the reference's three
  CSF tables; `psnr_hvs_daala_yuv420` and `psnr_hvs_daala_yuv444` cover
  both chroma geometries (libvmaf itself only ever runs 420). CLI
  `psnrhvs-daala` uses **YUV444** studio-601 chroma — identified as the
  AIC-4 convention by matching the published `PSNR-HVS-Cb`/`-Cr`
  columns (Δ ≤ 0.13 dB over 53 pairs; 420 ingress diverges ~4 dB).
- **DIVERGES** — u8 plane input only (the reference is an 8-bit
  integer-arithmetic metric; no hbd variant exists upstream). CLI
  ingress is *rounded* studio-601 luma/chroma — deliberately unlike
  cross-cutting #1 (unrounded house luma), because the reference's
  contract is the quantized plane.
- Per-plane scores are `-10·log10(masked_MSE)` with the reference's
  `0.8·Y + 0.1·(Cb+Cr)` MSE-space combine; masking is skipped on the DC
  term, matching `calc_psnrhvs`.

### `haarpsi` — vs `HaarPSI.m`

- **Fidelity note** — `conv2(...,'same')` even-kernel anchor is the
  non-obvious trap: MATLAB's 'same' anchors the *flipped* kernel with a
  forward-looking window (`s = n/2`, 0-based). Verified against Octave
  up-front and documented in `kernel.rs` — not a divergence, recorded
  here because a re-port that assumes a centered window diverges
  silently on every Haar scale.
- NaN on zero-weight inputs preserved exactly (cross-cutting #4).
- `haarpsi-y` (luma-only) is an **EXTENSION** — the reference is the
  full RGB→YIQ path only.

### `fsim` / `fsimc` / `fsim-y` — vs `FR_FSIMc.m`

- **DIVERGES** — vendored self-contained f32 FFT (Bluestein for
  arbitrary lengths) vs MATLAB's f64 FFTW: ≤ 5e-8 across goldens,
  including odd dims.
- **EXTENSION** — `fsim-y` luma variant; the reference ships FSIM
  (gray) + FSIMc (YIQ) only.
- House luma/YIQ `Y` is unrounded (cross-cutting #1); `F` auto-decimation
  is `max(1, round(min/256))` single-shot — the reference's exact rule.
- **RESOLVED (docs)** — this crate's README previously described the
  decimation as `floor(min/256)` "applied recursively", and `vsi`'s
  `lib.rs`/`kernel.rs` docs repeated a phantom "FSIM recursive `floor`"
  rule. Verified against `FR_FSIMc.m` (mirror of the authors' release):
  the decimation is `max(1,round(minDimension/256))`, single-shot —
  identical to VSI.m's rule. Corrected in fsim README, vsi
  `lib.rs`/`kernel.rs`, and the `gen_goldens.m` case labels.

### `vsi` — vs `VSI.m`

- **RESOLVED** — `imresize` bilinear-shrink semantics: Octave applies
  **symmetric whole-point padding** and normalizes by the *full* kernel
  sum — not drop-and-renormalize. `conv_interp_vec` was ported
  line-for-line and verified **bit-identical at all 17 scale combos**.
- **RESOLVED** — `conv2(...,'same')` anchor for even `F`: correct anchor
  is `s = ⌊F/2⌋` (forward window `{j, j+1}` for F=2), not `(f−1)/2`.
  Caught by the 520×400 golden (odd F unaffected).
- **DIVERGES** — deliberate fidelity traps *kept*, not fixed: D50 white
  point in `RGB2Lab` (`Xr=0.9642, Zr=0.8251`, nonstandard — do not
  "correct" it to D65). The `F = max(1, round(min/256))` subsample rule
  is **identical** to FSIM.m's — earlier crate docs claimed a
  difference (see the fsim RESOLVED-docs row).
- NaN on flat inputs preserved (cross-cutting #4).

### `msssim` — vs Wang's `msssim.m`

- **DIVERGES** — auto-level convention: the reference requires an
  explicit `(level, weights)` call; we expose `auto_level(w,h) =
  min(5, ⌊log2(min/11)⌋+1)` which is defined as exactly the reference
  call `(level=L, weight=w(1:L))`. Pinned by goldens at L=1..5.
- **DIVERGES** — negative `mssim`/`cs` factors: reference `x^y` on a
  negative base goes complex and callers see `real(z)` (`|m|^w·cos πw`).
  We compute the complex-product real part directly — same value, no
  complex arithmetic in the public path. (VSI's chroma term had the same
  semantics.)
- **RESOLVED** — `imfilter` 2×2 'same'+'symmetric' decimation: forward
  window `{i,i+1}` and whole-point symmetric pad (`x(N+1)→x(N)`), both
  verified against Octave before porting.
- Constant + identical inputs → exactly 1.0; `min < 11` → `TooSmall`
  (reference returns `-Inf`).

#### `libvmaf` module (`ssim-libvmaf`, `msssim-libvmaf`) — vs libvmaf `float_ssim`/`float_ms_ssim` @ f85a8536

- **RESOLVED** — hbd input scaling: libvmaf's `picture_copy` divides
  bpc>8 planes by `1<<(bpc−8)` *before* the float feature sees them
  (10-bit → ÷4, i.e. the feature runs in the 8-bit range). First port
  passed `data/255` naively → ~5e-3 error on 10-bit FFI cases; now the
  `bpc` parameter mirrors the C scaling exactly.
- **RESOLVED** — accumulation semantics: `float_ssim` averages the
  per-pixel l·c·s map in f64 then truncates the mean to f32
  (`(float)(sum/(double)n)`); `float_ms_ssim` raises those f32-rounded
  means to the scale weights in f64. Both roundings modeled; parity
  ≤ 2e-4 (observed ≪ gate).
- **DIVERGES** — different algorithm family from the crate's Wang
  `msssim` path on purpose: libvmaf uses a fixed 5-level 9/7-LPF pyramid
  with weights `0.0448/0.2856/0.3001/0.2363/0.1333` and its own
  decimate+Gaussian recipe (`scale = round(min_dim/256)` box decimate,
  11×11 σ1.5 window, zli l·c·s, L=255). The Wang path stays `msssim`;
  these land as `-libvmaf`-qualified metrics — do not merge the names.
- **DIVERGES** — CLI ingress is *rounded* studio-601 luma (deliberately
  unlike cross-cutting #1): libvmaf's float features are defined on the
  quantized luma plane, and the AIC-4 `SSIM`/`MS-SSIM` columns
  reproduce at raw med 1.0e-6 / max 1.8e-5 under it.
- **OUT-OF-SCOPE** — libvmaf's `enable_lcs`/`clip_db` option plumbing is
  test-oracle only; the module exposes raw scores (the defaults) only.

### `vif` — vs `vifp_mscale.m`

- **DIVERGES** — f64 end-to-end (cross-cutting #2). Not gratuitous: the
  reference's 1e-10 GSM mask thresholds are calibrated to f64's noise
  floor; an f32 statistics plane leaves ~1e-3 `E[x²]−μ²` residue on flat
  input and flips every mask (constant inputs would score finite instead
  of NaN). With f64 all degenerate semantics reproduce exactly.
- Identical inputs → `0.999999999992`, **not** exactly 1 — the GSM gain
  stays just under one; preserved, not clamped.
- Can exceed 1 legitimately (information ratio); CLI bound `(−0.001, 10)`.

### `vifvec` — vs `vifvec.m` (steerable-pyramid vecGSM release)

- **NOT the same metric as `vif`** — `vifp_mscale.m` is the later
  pixel-domain scalar-GSM simplification; `vifvec` is the original
  4-level steerable-pyramid (SP5) vector-GSM VIF of the 2006 paper.
  Shared name, different algorithm: keep the `vif`/`vifvec` column
  names distinct in any join.
- **RESOLVED** — matlabPyrTools `buildSpyrLevs` stores each 7×7
  orientation filter column-major (`bfilts(:,b)`); the Rust `BFILTS`
  constants are the same values transcribed row-major (machine-extracted
  from `sp5Filters.m`, not hand-copied).
- **RESOLVED** — `buildSpyr` correlation boundary is `reflect1`
  (half-sample symmetric, edge count 3 for a 7-tap filter); the Rust
  `corr_dn` reproduces it exactly (verified bit-equal at 1e-9 on
  96×96…300×260 goldens including 97×95 and 96×80 odd/uneven dims).
- **EXTENSION** — reference errors when `min(W,H) < 72`
  (`maxPyrHt` cannot build the 4th level); we return `Err` with the
  same boundary instead of scoring.
- RGB entry `vifvec_rgb8` uses u8-rounded `round(0.299R+0.587G+0.114B)`
  (cv2/PIL gray) — the ingress the AIC-4 `VIF` column was computed
  under: med |Δ| 1.9e-6 / max 6.3e-6 vs `metrics_fullres.tab` over
  the 54 available pairs (2026-09-26); unrounded luma misses by ~7e-4.
  `vifvec_plane_f64`
  exposes raw-f64 ingress for callers with their own convention.
- f64 end-to-end, like `vif` — the vecGSM log-det / `inv(cu)` terms sit
  at the same noise-floor as `vifp`'s 1e-10 masks.

### `mad-iqa` — vs `hi_index.m`/`lo_index.m` + C-mex + JEI combine

Deep doc: `crates/mad-iqa/validation/README.md`.

- **ORACLE-ARTIFACT** — `gen(2)` returned **uint64** (from `bitxor`);
  `hi_index.m`'s `isinteger` check only tests the *reference*, so the
  non-LUT branch ran `dst.^(2.2/3)` in **integer arithmetic**, collapsing
  luminance to 0/1 and corrupting the first golden set. `gen.m` now
  forces `double(img)`. Any port validated against uint8-typed oracle
  output bakes the same trap — documented so a re-port doesn't.
- **DIVERGES** — the uint8 LUT branch in `hi_index.m` is unreachable
  through our f32 API: we always compute the float-power luminance, which
  is what the LUT approximates anyway.
- **RESOLVED** — `imfilter 'same'` anchor for the 16×16 lmse kernel:
  `{i−7..i+8}` maps to padded `{x+1..x+16}`; first draft was off by one.
- **RESOLVED** — odd-dimension `fftshift` asymmetry: `hi_index`'s
  `ifftshift(fftshift(X)·csf)` shifts by `⌊n/2⌋` while `lo_index`'s
  `X·fftshift(filter)` shifts by `⌈n/2⌉`. Identical on even axes, off by
  one on odd — the 65×63 golden caught it; `gabor_apply` uses `div_ceil`.
- **DIVERGES** — the official combined `MAD_index` binary was
  unreachable (Shizuoka archive down); the geometric blend is
  reconstructed from the JEI paper (`b1=exp(−2.55/3.35)`,
  `b2=1/(ln10·3.35)`, `sig=1/(1+b1·HI^b2)`, `MAD=HI^sig·LO^(1−sig)`) and
  published ports. HI/LO are pinned to the official `.m`+C sources; the
  *combine* is pinned to the paper — noted because a future surfacing of
  `MAD_index` mex could reveal a different constant.
- `min(w,h) < 34` → NaN via reference edge-trim (cross-cutting #4);
  symmetric padding uses edge-doubling `idx(−k)=k−1`, `idx(N+k)=N−1−k`.

### `mdctpsnr` — vs `thorfdbg/mDCTpsnr` (compiled reference)

Deep doc: crate `src/lib.rs` header; goldens in the crate test suite.

The parity target is **the compiled binary's behavior**, not the source's
apparent semantics — under `-O3 -ffast-math` GCC reassociates, contracts
to FMA, vectorizes "scalar" libm calls into libmvec, and replaces division
with `vrcpps` + Newton refinement. Every deviation below was found by
disassembly + bitwise trace, fixed, and is recorded so a re-port doesn't
rediscover them.

- **RESOLVED** — `MeasureInBand` association: compiled order is
  `err = (|r−d|·visbase)·max(mask_r,mask_d)` and
  `p = (err·√err)·err²`, not the source's apparent
  `|r−d|·(vis·max)` / `err³·√err`.
- **RESOLVED** — pooling `error += 1−e` is not sequential: GCC emits an
  8-lane strided f32 accumulator, fold `acc[j]+acc[j+4]`, collapse
  `(x0+x2)+(x1+x3)`, a 4-wide tail block **iff `rem ≥ 4`**, scalar
  remainder after. `w13 % 8 == 3` widths were ~11 dB off before this was
  reproduced exactly (the 4-wide block does *not* run for `rem == 3` —
  no out-of-bounds slack lane is ever read).
- **RESOLVED** — vertical column-sum tree reassociated to
  `((r0+r7)+(r5+r6))+((r1+r2)+(r3+r4))`; AAN DCT passes reassociated and
  FMA-contracted differently in pass 1 vs pass 2 (decoded form in
  `aan_1d`); mask conv contracted to
  `fma(m2,p[i+2], fma(m0,p[i]+p[i+4], m1·(p[i+1]+p[i+3])))`; mask division
  is `rcpps`+NR with a fused denominator; DC fixup is
  `fma(m11,0.25,(m01+m10)·0.5)`; lowpass k-loop is a fused pair-tree;
  `slh = vis·NormLo·NormHi` is shared for both LH and HL blocks.
- **RESOLVED** — libmvec: GCC vectorized the "scalar" `powf`/`cosf`/`expf`
  loops into `_ZGVdN8vv_powf`, `_ZGVbN4v_cosf`, `_ZGVdN8v_expf`. We call the
  real `_ZGVdN8vv_powf`/`_ZGVdN8v_expf` via `asm!` (gated
  `x86_64+linux+gnu`) and pin the 5 kernel taps + post-filter taps to the
  reference's exact bits (vec-`cosf` differs from scalar `cosf` ~1ulp on
  2 of 5 taps). Scalar fallbacks elsewhere drift ~1ulp/step.
- **DIVERGES** — residual ~1.7e-5 in the diagnostic f64 `error64` sum (a
  few hundred 1-ulp errorline values across 3.4M positions); the f32
  `error` used for the score converged to ≤ ~4e-6 dB. Not user-visible.
- **DIVERGES** — platform scope: the parity claim is x86_64 GNU/Linux +
  AVX2 + glibc libmvec. Other platforms run the same algorithm with
  scalar libm (still bit-stable, but not bit-identical to the reference
  binary; test tolerance widens to 1e-2 off-platform).
- **EXTENSION** — stride-agnostic packed-RGB8 API, `Err` on dim mismatch /
  undersize instead of the reference's unchecked UB.
- **OUT-OF-SCOPE** — the reference's ifdef'd-out paths (saliency,
  `WEIGHT_MSE`/`DELTA_E`, `NO_BASE_VISIBILITY`) are not compiled.

### `ssim2` (CPU) — external `fast-ssim2`

The CPU column is the published/sibling `fast-ssim2` crate, not an
in-repo port — divergence tracking lives in that repo (its CLAUDE.md
records the C++ SSIMULACRA2 parity policy and the sub-8px reflect-pad
unification). `hdr-pu` (`compute_ssimulacra2_pu_nits`) is an upstream
**EXTENSION** the `cpu-ssim2` HDR route consumes.

### `ssim2-gpu` — vs `ssimulacra2` 0.5

- **DIVERGES** — the default portable path uses an **FIR** blur while
  the CPU reference uses the recursive **IIR** — per-image scores diverge
  by ~5e-5 in the final score; the two are treated as *distinct metrics*
  (different impulse-response support) and sweep tooling lands the FIR
  variant deliberately (README §"FIR vs IIR").

### `butteraugli` / `butteraugli-gpu`

- CPU column = external `butteraugli` 0.9.4 (imazen fork via path dep;
  crates.io equivalent when the sibling checkout is absent).
- **DIVERGES** — butteraugli-gpu strip mode: per-strip host-side max +
  p3/p6/p12 reduction order differs from the single fused on-device
  reduce → ≤ 1e-4 relative, enforced by `tests/multires_strip.rs`.

### `dssim` / `dssim-gpu`

- **DIVERGES** — version-pin naming: workspace pins `dssim-core ^3.4`
  because crates.io tops out at 3.4.0 while the local sibling holds an
  unpushed 3.5.0; the README table says "3.5". `^3.4` is the honest
  constraint — dssim-gpu's own lib.rs names "published dssim-core v3.4"
  as its parity target. The `[patch]` to the local checkout was removed
  on purpose (#43): parity tests must run against the *published* crate.
- **DIVERGES** — `cubecl-cpu` backend is build-check-only (reduction
  kernels need atomics + `CUBE_COUNT` that cubecl-cpu 0.10 lacks) — the
  CPU fallback is explicitly **not** a runtime parity target.

### `zensim` / `zensim-gpu`

- `zensim` is the in-house metric — **there is no external reference to
  diverge from**; its pinned versions are the contract for zensim-gpu.
- zensim-gpu deep doc: `crates/zensim-gpu/docs/DIFFMAP_DIVERGENCES.md`
  (12 items — strict `score == Minkowski(diffmap)` not delivered,
  CPU-fallback diffmap path, score-direction normalisation, `Libjxl`
  short-circuit unimplemented, HDR out of Phase-1 scope, deferred
  reducer constants and diffmap renormalisation, compat shim for
  `score_features_with_profile_and_codec`, benchmark record).
- **DIVERGES** — diffmap CPU-vs-GPU ≤ 2.08e-4 pointwise
  (`tests/cpu_gpu_diffmap_parity.rs`), larger than the new ports' parity
  because the diffmap path still runs a CPU fallback (§2b/§9).

## Maintenance rule

Every metric port or reference-version bump updates this file **in the
same commit**: a ledger row (or delta refresh) plus a per-metric entry
for anything that isn't pure float noise. Crate-level depth stays in the
crate docs — this file links, it does not duplicate.
