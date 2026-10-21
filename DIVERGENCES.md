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
| `zenmetrics-nn` PieAPP | Official v0.1 `compute_score`, release `0513613d1f575136ea52418e35f2cebc5a49f783` | Explicit stride-six public-entry-point grid | Worst 1.84e-6 operational JND over 24 supported pairs; demo CLI sampling differs (see section below) |
| `cvvdp` | pycvvdp **v0.5.7** (gfxdisp/ColorVideoVDP); `PYCVVDP_REFERENCE_VERSION` | pycvvdp itself + cvvdp-conformance cells | 2e-6 JOD (u16 video cells), 1.2e-5 (stills); scalar parity 3e-6 JOD |
| `cvvdp-gpu` | pycvvdp **v0.5.7**; `parity-goldens` test pins **v0.5.4** manifest from R2 | pycvvdp goldens + in-tree `cvvdp` | see crate `crates/cvvdp-gpu/docs/PORT_STATUS.md` |
| `hdrvdp` | official **HDR-VDP 2.2.2** MATLAB release (SourceForge) | Octave 11.1 run of official 2.2.2; UPIQ corpus | P_det 1.4e-5 rel, C_max 1e-4 rel, P_map 1.7e-2 abs, `res.Q` 7.8e-4 |
| `hdrvdp` `v3` | official **HDR-VDP-3.0.7** MATLAB release (SourceForge) + **jpeg-ai-qaf `VDP3/`** torch port @ `0628a6b` (faithful — verified against the original, see entry) | 21-case golden corpus from the reference port + Octave run of MATLAB 3.0.7 on AIC-4 pairs | ≤ 1.6e-13 JOD worst |Q_JOD| over 21 cases; **bit-equal** to MATLAB 3.0.7 at print precision on 3 AIC-4 pairs |
| `vmaf` | Netflix **libvmaf 3.2.1** (vendored via `vmaf-head-sys 0.2.0`, test-only) | libvmaf FFI oracle in `crates/vmaf/tests/ffi_fusion.rs` | v0 asserted ≤ 1e-4 features (motion2 1e-8), ≤ 0.02 score; integer stat paths exact by construction |
| `gmsd` | **libgmsd** (Tom Clunie's C port of the authors' `GMSD.m`) | libgmsd bit-comparison | bit-identical map on even dims; f64 rounding only in score |
| `iwssim` | **Python-IW-SSIM** @ `f9de37c` (Jack-guo-xy) | committed JSON goldens | identical ≤ 1e-5, distorted ≤ 5e-3; strip-vs-whole ≤ 1e-6 |
| `iwssim-piq` | same algorithm as `iwssim`; the **JPEG AIC-4 `IW-SSIM` column** (jpeg-ai-qaf `IW_SSIM` on unrounded Y) | published AIC-4 `metrics_fullres.tab` | med 3.9e-6, max 2.6e-5 over 54 pairs (2026-09-26) |
| `psnrhvs` | authors' **`psnrhvsm.m`** (metrix MATLAB) | Octave goldens, `validation/` | ≤ ~4e-4 dB (asserted ≤ 1e-3) |
| `psnrhvs` (daala module) | Daala/Xiph **`dump_psnrhvs`** as vendored in libvmaf `psnr_hvs` @ **f85a8536** (`vmaf-head-sys 0.2.0`) | libvmaf FFI oracle in `crates/msssim/tests/ffi_libvmaf.rs` | asserted ≤ 2e-4 dB, YUV420+YUV444; observed ≪ gate |
| `haarpsi` | authors' MIT **`HaarPSI.m`** | Octave goldens, `validation/` | ≤ 5e-5 over 16 rows |
| `fsim` | authors' **`FR_FSIMc.m`** (research license) | Octave goldens, `validation/` | ≤ 5e-8 over 18 rows |
| `vsi` | authors' **`VSI.m`** (author site, research license) | Octave + `pkg load image` goldens | ≤ 1e-4 over 14 rows |
| `msssim` | Wang's **`msssim.m`** (MAD_Competition archive) | Octave goldens, `validation/` | ≤ 8.8e-6 over 15 rows |
| `msssim` (libvmaf module) | libvmaf **`float_ssim`/`float_ms_ssim`** @ **f85a8536** (vendored via `vmaf-head-sys 0.2.0`, test-only) | libvmaf FFI oracle in `crates/msssim/tests/ffi_libvmaf.rs` | asserted ≤ 2e-4; observed ≪ gate (8-bit, 10-bit, identical) |
| `vif` | authors' **`vifp_mscale.m`** (pixel-domain release) | Octave goldens, `validation/` | ≤ ~5e-13 over 17 rows |
| `vifvec` | authors' **`vifvec.m`** + matlabPyrTools `sp5Filters`/`buildSpyr`/`vifsub_est_M` (steerable-pyramid vecGSM release — a different metric from `vif` that shares the name) | GNU Octave run of the official `.m` set | ≤ 1e-9 over 10 goldens (incl. odd dims, 72px floor, identity, black-on-flat) |
| `mad-iqa` | Larson & Chandler **`hi_index.m`/`lo_index.m`** + `ical_std.c`/`ical_stat.c` (STMAD_2011, archived in Netflix/vmaf); JEI 2010 combine | Octave `.m` shims of the C-mex + official `.m` drivers | hi 2.4e-7, lo 1.8e-6, mad 1.3e-6 rel over 13 rows |
| `mdctpsnr` | **`thorfdbg/mDCTpsnr`** (Thomas Richter / U. Stuttgart; zlib-style license), compiled GCC `-O3 -ffast-math` + AVX2 + glibc 2.43 libmvec on x86_64 | the built `dctpsnr` binary + published AIC-4 `mDCT-PSNR` column | ~4e-6 dB on synthetic goldens incl. all `(w−13)%8` classes; AIC-4 53-pair max ~1.13e-5 dB (see entry) |
| `nlpd` | **`Valerolaparra/NLPD_Pytorch`** (authors' PyTorch ref; PyTorch 2.5.1) | torch-run golden, `tests` | within 1e-5 (f64 ref vs f32 ops) |
| `nlpd-iqa` | **`IQA_pytorch` `NLPD(channels=1)`** (dingkeyan93 IQA-optimization — alexhepburn nlpd-tensorflow lineage; a different metric that shares the name) | numpy replica + published AIC-4 `NLPD` column | ≤ 1e-4 on synthetic golden; AIC-4 53-pair med 2.6e-3 / max 6.7e-3 (see entry) |
| `ssim2` (CPU) | external **`fast-ssim2`** crate (sibling repo; C++ SSIMULACRA2 parity) | fast-ssim2's own parity suite | owned by fast-ssim2 repo |
| `ssim2-gpu` | published **`ssimulacra2` 0.5** crate | CPU-reference parity tests | FIR path ~5e-5 vs IIR — see entry |
| `butteraugli` (CPU) | external **`butteraugli`** crate 0.9.4 (imazen fork, path dep) | the crate itself | external — not ported here |
| `butteraugli-gpu` | **`butteraugli` v0.9.2** parity target | CPU-reference parity tests | strip reduce order: ≤ 1e-4 rel |
| `dssim` (CPU) | external **`dssim-core`** — workspace pins `^3.4` (crates.io max); README table names 3.5 (unpublished local line) | the crate itself | external — not ported here |
| `dssim-gpu` | **`dssim-core` v3.4** | integration tests vs published crate | see crate README/`PORT_STATUS` |
| `iwssim-gpu` | in-tree **`iwssim`** (this repo) | CPU-vs-GPU parity | see entry — column-name split is deliberate |
| `zensim` (CPU) | in-house metric — **no external reference exists** | self | n/a — the crate *is* the definition |
| `zensim-gpu` | **`zensim` 0.2.8 / pinned 0.3.0** | `crates/zensim-gpu/tests/it/cpu_gpu_diffmap_parity.rs` | ≤ 2.08e-4 pointwise; 12 documented items |
| `wadiqam` (zenmetrics-nn) | **`dmaniry/deepIQA`** @ `51dece99` (MIT code + weights), `top="weighted"` checkpoints | chainer CPU oracle (`models/wadiqam/oracle.py`) | 4.2e-5 worst export-check score delta; both-engine parity ≤ 7.8e-6 JND — see entry |
| `zenmetrics-nn` `dmm` | **Baoliang93/DMM** `DMM_PyTorch` 0.1.2 (MIT), CPU fp32 `torch.svd` | netlib-`sgesdd` goldens on the engine's own taps (`dmm_goldens_netlib_<engine>.tsv`); MKL goldens recorded, not gated | score is solver-draw-dependent in degenerate patches — see `### dmm` |

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

Deep doc: `crates/cvvdp/docs/UPSTREAM_DIVERGENCES.md` (6 open items),
`crates/cvvdp/docs/NAN_ON_IDENTICAL_INPUT.md`,
`crates/cvvdp/docs/CVVDP_CONFORMANCE.md`.

- **RESOLVED** — Temporal channel (`Y_t`, `beta_t` pooling, 4th slots of
  `mask_q`/`xcm_weights`/`baseband_weight`): ported — `VideoScorer`
  carries all 4 channels including the transient, all three upstream
  `temp_filter` branches (`default` Gaussian band-pass, `hp_trans`,
  `grad_trans`; ≤1e-5 JOD vs pycvvdp on the 44-cell corpus), plus
  `replicate`/`symmetric` padding.
- **OUT-OF-SCOPE** — `temp_padding="valid"` (raises `RuntimeError`
  upstream anyway), `temp_resample`, video heatmap output,
  `dump_channels`, `masking_model` variants other than `mult-mutual`.
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

- **EXTENSION — native map definition.** The CPU port's native diffmap
  bilinearly upsamples masked-error bands, adds them per channel, then folds
  channels with a Minkowski norm (`crates/cvvdp/src/diffmap.rs`). pycvvdp
  0.5.7's `heatmap="raw"` folds channels within each band, reconstructs its
  heatmap pyramid, applies `1 - met2jod(map)/10`, and stores float16 output.
  These are different map definitions and units. Scalar parity does not
  establish pointwise equivalence between those outputs. [Local source inspection and file hashes](https://github.com/imazen/zenmetrics/blob/master/benchmarks/canonical_recheck_2026-10-19/cvvdp-map-semantics.json)
  record that boundary.

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

### Local original HDR numerical diagnostics (2026-10-20)

The [partial 12-of-48-case grid](benchmarks/canonical_recheck_2026-10-19/hdr-original-partial12.json)
compares the base sRGB-display routes with the original official functions.
HDR-VDP v2 differs by 0.009351851113351017 on the 1024×1024 photo at both worker
budgets, above the retained 0.005 golden-check level. That level is an existing
fixture check, not a newly established universal bound. V3 differs by at most
7.105427357601002e-15 in these completed cases; its bound remains unassigned.

These are numerical diagnostics under Octave 6.4/image 2.12 with two reviewed
upstream image 2.20.1 resize M-file overlays. Full original/direct controls,
runtime qualification and the original-file/native-embedded parameter boundary
remain incomplete. Every ratio and native parity qualification is withheld.
This finding does not change the historical validation below; its cause remains
unresolved. The 1024×1024 v2 one-worker wrapper recorded 69 s / 0.57 GiB peak RSS;
v3 recorded 85 s / 0.60 GiB.

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
- **OUT-OF-SCOPE** — crate's top-level `score()` API tracks 2.2.2 only;
  HDR-VDP-3 lives in `hdrvdp::v3` (next entry). GPU not planned
  (`parallel` CPU is the fleet answer).

### `hdrvdp` `v3` — vs official HDR-VDP-3.0.7 + jpeg-ai-qaf `VDP3/` port

Implementation: `crates/hdrvdp/src/v3/` — clean Rust reimplementation of
the `VDP3/` numpy/torch port in jpeg-ai-qaf @ `0628a6b`
(`hdrvdp3.py`/`hdrvdp_visual_pathway.py`/`hdrvdp_spyr.py`/…), f64
throughout, reusing the crate's `resize`/`interp1`/`corrDn`-style
primitives but its own sp0 pyramid (`spyr0`), spectral path, and f64 FFT.

- **VALIDATED** — 21-case golden corpus generated from the reference port
  (`gen_vdp3_goldens.py`, `tests/vdp3_goldens/`): every input encoding
  (rgb-bt.709/rgb-bt.2020/rgb-native/xyz + `luminance`), every task
  (quality/side-by-side/flicker), preset + 2-channel-custom emission,
  `surround=none|mean|<cd/m²>`, `do_pixel_threshold`, `si_gauss`,
  `do_masking` off, `luma-display`, odd dims, identity. Worst |ΔQ_JOD|
  **1.6e-13**, worst |ΔP_map| 6.1e-13. `tests/reference_vdp3.rs` asserts
  all of it plus staged intermediates (L_adapt ≤ 1.4e-12, JND LUTs,
  pyramid bands ≤ 1e-12).
- **VALIDATED** — the port itself is faithful to MATLAB: GNU Octave run
  of the **original HDR-VDP-3.0.7** (`hdrvdp3.m`, SourceForge) on three
  AIC-4 pairs under the stated harness ingress (PQ-decode sRGB codes,
  +`L_refl = 0.005·100/π`, clamp `L_max+L_refl`, `rgb-bt.709`, ppd 64.0634,
  quality, `led-lcd-srgb`, age 24, surround none) gives **9.696845 /
  8.517431 / 9.645288 — identical to our Rust at print precision**, and
  identical to the numpy port. (Octave shims needed: `pkg load image`,
  plus a local `geomean.m` since the statistics package was absent — both
  oracle-side only.)
- **DIVERGES (unresolved, upstream-side)** — the *published* AIC-4
  `HDR_VDP_3` column (`metrics_fullres.tab`: 9.339219 / 7.598649 /
  9.352748 on the same pairs) is **not** reproducible from any recovered
  configuration: ppd sweeps (28–64, incl. image-derived ~59), task
  variants, EOTF/no-EOTF, reflectance removal, ref/test swap, and
  `base_sensitivity_correction` probes all leave distortion-proportional
  residuals. The published `HDR_VDP_2` column likewise misses our
  2.2.2-validated v2 at both ppd 30 and 64.05. Conclusion: the columns
  were generated by a different implementation revision or harness
  configuration than the shipped code + stated defaults; our numbers
  match MATLAB 3.0.7 exactly, so we record the discrepancy rather than
  fit to it.
- **ORACLE-ARTIFACT** — reference `'sRGB-display'` input encoding is
  case-sensitive *inconsistently*: `hdrvdp3.py` checks `'srgb-display'`
  (all-lowercase) for the gamma decode but `.lower()`s for the
  ITU→native matrix, so `'sRGB-display'` silently skips the decode while
  still applying the matrix. We implement the exact-match behaviour;
  pass `srgb-display` (or better, avoid sRGB inputs — see next).
- **ORACLE-ARTIFACT** — `ignore_freqs_lower_than` crashes the reference
  (`UnboundLocalError`: `D` unbound when a band is ignored). Ours
  handles it (zero-fills the band) — deliberately *not* bug-compatible;
  flagged here so the behavioural difference is on record.
- **DIVERGES (API, intentional)** — v3 has **no default viewing
  geometry**: `pix_per_deg` is a required `ViewingConditions` field, and
  the CLI hard-requires `--hdrvdp3-ppd`. v2's `DEFAULT_PIX_PER_DEG = 30`
  is a 2.2.2 convention; v3's task/emission/encoding/surface make a
  silent default produce incomparable scores.
- **DIVERGES (API, intentional)** — v3 input is absolute luminance only
  (interleaved nits, `HdrFeeding::IntegratedPuNits`); the sRGB8 and
  display-relative feedings `Err` loudly rather than reinterpret codes.
- Score scale: `q_jod` on [0,10] (10 = identical) — **not** comparable
  to v2's `res.Q` on [0,100]. Column name `hdrvdp3_jod_imazen_v*`
  keeps them distinct.
- Dead options preserved as accepted-but-inert for parity:
  `do_robust_pdet`, `do_spatial_total_pooling`, `view_dist` are parsed
  upstream but never read; `sensitivity_correction` likewise.

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

A source-bound [October 19 primary C grid](benchmarks/port_spotcheck_2026-10-18/primary-gmsd.md)
now rechecks the distinction below: 18 even-size cases have bit-identical maps,
while all six original 255×257 C cases report an AddressSanitizer buffer overflow
in `downsample_2x2`. The documented score/mean bounds remain unchanged; no odd
input is cropped and no C odd-size numerical or timing pass is claimed.

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
  strip-vs-whole self-divergence ≤ 1e-6 (`crates/iwssim/tests/strip_parity.rs`).
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
- **Observed native-map difference (2026-10-20)** — the current frozen
  `efae6b8fed9e57346157d92dffc3cdb57c44560e` worker produces 41 NaN map
  positions on a 3840×2160 illustration where the original similarity/weight
  arrays yield a finite weighted map, at both 1 and 28 threads. The other
  22 size/content/thread cases have no nonfinite mismatches. The largest
  finite-pixel difference over all 24 cases is 1.1233741833871491e-5;
  no external map bound is assigned. Map and scalar-only APIs return identical
  scalar bits in every case. [Current evidence and raw-plane hashes](
  benchmarks/canonical_recheck_2026-10-19/haarpsi-current-maps/haarpsi-maps.json).

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

- **REFERENCE-RUNTIME DIFFERENCE, measured 2026-10-19** — a 64×64 S67/CCM15 photo crop gives
  Rust `vsi`/`vsi-nodownscale` 0.9567964257437082 versus original author
  `VSI.m` 0.94679144074028154 through Octave 6.4.0/image 2.12.0.
  Absolute difference 0.010004985003426614 exceeds the unchanged 1e-4 gate.
  Independent PPM/raw ingress equality and the unmodified primary source
  reproduce it. A resize-dependency control identifies the local difference:
  unchanged image-2.20.1 resize M-files give 0.95679610895892431 through the
  same author source, for an absolute Rust difference of 3.1678478384922215e-7.
  Three procedural low-contrast cases also pass with that source selection.
  No metric code or gate changed; the larger newly bound panel remains a
  separate experiment. [Initial observations](benchmarks/port_spotcheck_2026-10-18/vsi-photo-finding.md),
  [dependency controls and runtime limits](benchmarks/port_spotcheck_2026-10-18/vsi-resize-runtime.md).
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

### `nlpd` / `nlpd-iqa` — one name, two metrics (NLPD variant split)

`nlpd` (`score_rgb_u8`/`score_f32`) ports the Laparra-author
`NLPD_Pytorch` configuration: **RGB**, six pyramid levels, per-level RMS
pooled by `(Σ rms^0.6)^(1/0.6)`.

`nlpd-iqa` (`crates/nlpd/src/iqa.rs`, CLI `--metric nlpd-iqa`) ports the
configuration the **JPEG AIC-4 `NLPD` column** was actually computed
with — `IQA_pytorch` (`dingkeyan93/IQA-optimization`, the alexhepburn
nlpd-tensorflow lineage; NOT the later pyiqa port):

- **DIVERGES** (by design, vs `nlpd`) — single-channel BT.709 Y in
  `[0,1]` re-quantized to the 8-bit grid (the harness's
  `rgb_to_yuv` + `round_plane(.., 8)`), not RGB.
- **DIVERGES** — pooling is the plain mean of the six per-level RMS
  values, not `0.6`-norm pooling — alone accounts for a
  `6^(1/0.6) ≈ 19.8×` score factor (observed ≈20.5× with chroma).
- **DIVERGES** — downsample is `ReflectionPad2d(2)` + stride-2 convolve
  (fixed pad-2 phase; the Laparra path's parity-dependent pad differs by
  one pixel on even dims), upsample is `bilinear(×2, align_corners=True)`
  onto the `2·lw × 2·lh` grid with a `nearest`-crop on odd dims, and the
  six loop bands include the terminal low-pass as a difference source —
  no separately appended residual band.
- **RESOLVED** — dataset reproduction: vs the published `NLPD` column
  over 53 stratified AIC-4 pairs, `nlpd-iqa` lands at med |Δ| 0.0026 /
  max |Δ| 0.0067 (column range ~0.03–0.22; med-rel 2.63% — f32
  op-order vs torch plus bilinear edge detail), where the RGB `nlpd`
  reads ≈20.5× high. Emits its own `nlpd_iqa` column so the two variants
  never mix in a join.

### `ssim2` (CPU) — external `fast-ssim2`

The CPU column is the published/sibling `fast-ssim2` crate, not an
in-repo port — divergence tracking lives in that repo (its CLAUDE.md
records the C++ SSIMULACRA2 parity policy and the sub-8px reflect-pad
unification). `hdr-pu` (`compute_ssimulacra2_pu_nits`) is an upstream
**EXTENSION** the `cpu-ssim2` HDR route consumes.

### `ssim2-gpu` — vs `ssimulacra2` 0.5

- **Primary C++ small-input observation, 2026-10-20.** The retained 64² photo
  control gives GPU 84.8498333349087 and primary SSIMULACRA2 v2.1
  90.86388011956463 in all three blocks at both 1/28 thread budgets. Source
  inspection finds four GPU scales (64/32/16/8) versus five in the primary
  loop (also 4), because their size checks occur on different sides of the
  downsample. This establishes a scale-selection difference; it does not yet
  isolate the entire score delta to that difference. A fresh ordinary CPU-route
  control on the same inputs returns exactly 90.86388011956463, matching primary.
  No new bound or accepted
  parity result is assigned. See the [source and scalar receipt](benchmarks/canonical_recheck_2026-10-19/ssim2-gpu-small-scalar-finding.json).

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
  reduce → ≤ 1e-4 relative, enforced by `crates/butteraugli-gpu/tests/it/multires_strip.rs`.

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
  (`crates/zensim-gpu/tests/it/cpu_gpu_diffmap_parity.rs`), larger than
  the new ports' parity
  because the diffmap path still runs a CPU fallback (§2b/§9).

### `dmm` (`zenmetrics-nn`) — vs Baoliang93/DMM `DMM_PyTorch` 0.1.2

- **DIVERGES** — `wt` weights each patch's singular-value distance by
  the spread of elementwise products of its two SVD bases. Degenerate
  singular clusters permit different orthonormal bases, and numerical paths
  can change the score. The host head is based on LAPACK stages in f32;
  source correspondence alone does not establish whole-panel parity.
- Native score-revision-1 comparisons against upstream MKL on all 24 ladder
  rows per engine (score units; 1 JND = 0.191420): rten median
  1.673475654e-3 / maximum 5.674371081e-2; ort-cuda median
  7.338384395e-4 / maximum 1.764253188e-1. The full 24-row CUDA repeat
  is bit-identical. Original rounded TSV measurements are retained in
  [the audit receipt](benchmarks/nn_port_2026-10-15/dmm_fix_2026-10-17.md).
  Historical upstream CPU-vs-CUDA (`torch.svd` MKL vs
  cuSolver, same pipeline): up to **1.73 absolute (~9 JND)** on
  `2207_128`, +1.7% on `2207_256`, +3.3% on `2406_128`. These historical
  observations do not establish acceptance for a different solver revision.
- Gate: `dmm` is out of the default `nn-parity` model set;
  `just nn-parity-dmm` asserts the port against per-engine
  netlib-`sgesdd` goldens (`dmm_goldens_netlib_{rten,ort_cuda}.tsv`,
  `models/dmm/gen_netlib_goldens.py` on `examples/dmm_tap_dump`
  same-engine taps, isolating the solver from trunk differences).
  Rows the transliteration tracks at 1e-3 JND gate
  there; rows whose netlib-vs-port draws drift past the gate are pinned
  at their measured residual +5% (`kind=pinned` in the TSV) — a
  regression floor, not a parity claim. The MKL comparison is printed
  per row, not asserted. All 48 frozen row limits pass on the restored solver.
  Current revision-1 classification: rten 6 jnd + 18 pinned
  (worst jnd-row err 1.09e-4), ort-cuda 7 jnd + 17 pinned (worst
  1.81e-4). The combined revision-2 experiment failed nine frozen rten limits and twelve
  CUDA limits, including two/four existing JND rows respectively. Actual
  canonical-JND coverage is 5/24 rten and 3/24 CUDA; that candidate was not
  qualified by the frozen panels. Its proposed row promotion is reverted
  with the candidate. The active arithmetic and classifications are restored
  to the frozen revision-1 baseline. All eight isolated arithmetic changes
  were rejected after complete 48-row comparisons: each worsens at least
  one residual, and five also fail frozen limits. Every 24-row CUDA repeat
  is bit-identical.
  No original limit is increased and no original JND row is demoted. See
  [the solver audit](benchmarks/nn_port_2026-10-15/dmm_fix_2026-10-17.md),
  `models/dmm/card.toml`, and `benchmarks/dmm_aic_probe/`.

## PieAPP sampling and official entry point (2026-10-16)

The port targets official `compute_score` on a regular stride-six patch grid.
It does not reproduce the demo CLI's sub-image sampling: in the 24-pair
panel the worst gap is 1.88 JND (`2207_2048`), median 0.44 JND, and 10/24
pairs exceed 1 JND. See [model provenance](crates/zenmetrics-nn/models/pieapp/PROVENANCE.md)
and the [measured panel](benchmarks/pieapp_review_2026-10-16/cli.json).

## `wd` — Cool-Chic Wasserstein Distortion (2026-10-16)

- **RESOLVED:** RTen and CUDA VGG16 trunk plus Rust statistics pass 25 upstream
  pairs × five columns per engine at 1e-3 KonJND. Worst error is
  3.443705654e-6 JND. The 1008-pair calibration and source hashes are in the
  [receipt](benchmarks/wd_port_2026-10-16/README.md).
- **EXTENSION:** native strided RGB8 and warm-reference feature reuse; explicit
  size/area errors. Spatial maps are unsupported.
- **EXTENSION:** `wd_s3` is Cool-Chic's default `WassersteinLoss` (`log2_sigma=3`);
  `wd_s1`, `wd_s2`, `wd_s4` and `wd_s5` run the same code at `log2_sigma` 1, 2, 4
  and 5, which Cool-Chic does not expose.
- **MATCHES (on purpose):** local variance is `E[x²] − μ²` in f32, then clamped
  to 5e-7, as upstream. It cancels in smooth regions; a centred variance would
  be more stable but would break parity with Cool-Chic.
- **DIVERGES:** the published AIC preprocessing remains unidentified. A separate
  time-boxed search selected raw post-pool taps with feature-rescaled sigma on
  three pairs; all-61 verification gives column median site errors
  0.2310363687–0.3724274401 JND, worst 2.4017891693 JND. No AIC match is claimed;
  the shipped metric retains Cool-Chic's pre-pool raw-input convention.

## `afine` — A-FINE (zenmetrics-nn, 2026-10-16)

- **DIVERGES (unresolved, upstream-side):** the published AIC-4 `A-FINE`
  column is **not** upstream `afine_all` under any convention found. The
  black-box search ([receipt](benchmarks/afine_aic_2026-10-16/README.md))
  dumped every internal sub-score (`n_r`/`n_d`/`fid`/`s_r`/`s_d`/`s_f`/`λ`)
  under the native convention on 60 of 61 pairs and evaluated ~700
  recombinations plus ref/dist order against them; the other five
  preprocessing variants (BGR, no-norm, 224/384/512 resizes) were compared
  on `afine_all` only, over the 39 pairs the pre-fix sweep reached.
  Best raw candidate `s_f + (s_d − s_r)` still misses at 0.106 JND median /
  0.59 max vs the ≤0.01/≤0.05 gate; fitted forms cap SRCC ≈0.92 — the column
  is no monotone transform of any searched candidate. Likely a different
  checkpoint or pipeline step on the site side; the shipped metric stays the
  upstream-verified `afine_all` (rten parity 1.42e-5, ort-cuda 1.94e-5 JND).
- **Resource note:** the rten path's activation demand on the 8.4 MP pair
  (S67 3355×2516) OOM-kills on this ~30 GB box — measured peaks 22.3 GB
  (8 threads) and 27.2 GB (4 threads) before the kill; ort-cuda scores the
  pair's `afine_all` in VRAM instead.

## `wadiqam` — deepIQA WaDIQaM FR+NR (2026-10-16)

- **RESOLVED:** ONNX exports of upstream's own chainer model classes (ported to
  torch for export; mirror verified ≤ 4.2e-5 pooled-score delta against the
  checkpoints) pass 29 ladder pairs — including non-square and 4096² — at
  1e-3 KonJND on rten and ort-cuda, worst 7.8e-6 JND
  (`benchmarks/wadiqam_port_2026-10-16.md`).
- **DIVERGES:** inputs with more than 32,768 patches (≈ 5824² at stride 32,
  above the 4096² evaluation ceiling) are refused. Upstream accepts arbitrary
  sizes and silently batches inference at 2,000 patches per graph call.
- **DIVERGES:** the host-side attention pool `Σ(y·a)/Σa` accumulates in f64;
  upstream's chainer path is f32 end-to-end. Observed effect is below the
  1e-3-JND parity gate (≤ 7.8e-6 JND vs the chainer oracle).
- **EXTENSION:** the public API accepts strided RGB8 input (`Rgb8In`), which
  upstream's `evaluate.py` does not define; extraction is non-overlapping
  stride-32 patches at native resolution, edge partials dropped, same as
  upstream.
- **EXTENSION:** inference chunking at 2,000 patches per graph call matches
  upstream's batch size; chunk-vs-whole equivalence is unit-tested, scores are
  bit-identical by construction (concatenated inputs, per-patch outputs).

## ort-webgpu engine (`Engine::OrtWebgpu`, feature `ort-webgpu`)

A second ORT engine, running the same exported graphs on ONNX Runtime's
WebGPU plugin EP (`libonnxruntime_providers_webgpu.so` from the
version-matched `onnxruntime-ep-webgpu` package, registered from
`ZENMETRICS_NN_WEBGPU_EP_LIB` — like `ORT_DYLIB_PATH`, the variable names
trusted code that is dlopen'd into the process). Sessions set
deterministic compute and forbid CPU fallback: a graph that leaves a
node on the CPU fails to commit. Memory pattern stays ON — the CUDA-side
arena-retention bug it is disabled for does not exist in this EP;
measured identical peaks either way (receipt below).
Full receipt:
[`benchmarks/ort_webgpu_2026-10-17/`](benchmarks/ort_webgpu_2026-10-17/README.md).

- **DIVERGES (capability):** five models cannot commit a session at all
  because their graphs carry CPU-assigned host ops — `Cast`,
  `ConstantOfShape`, `EyeLike`, `Gather`, `SequenceAt` shape/metadata
  housekeeping that ORT will not place on the WebGPU EP and
  `session.disable_cpu_ep_fallback` then rejects: **afine, arniqa, ckdn,
  deepdc, musiq**. `from_dir` fails loudly at load — `Error::Engine` for
  environment/registration problems, `Error::BadModel` when the graph
  cannot commit.
- **DIVERGES (capacity):** the fused pair graphs for **dists, lpips_vgg,
  stlpips_vgg** exhaust a 16 GB RTX 4080's device memory at 2048² under
  the WebGPU EP's allocation model (`vkAllocateMemory` →
  `VK_ERROR_OUT_OF_DEVICE_MEMORY`) — **PARTIAL** support. Every scored
  pair at ≤1024² matches the goldens: dists worst 1.132e-6 abs (1.2e-5
  JND over printed pairs), lpips_vgg 5.96e-8 abs (3.1e-7 JND),
  stlpips_vgg 2.96e-8 abs (4.5e-7 JND). The same graphs complete the full
  64–2048² ladder on ort-cuda, whose allocator pools and reuses device
  memory.
- **PASS at the 1e-3-JND gate with bit-identical repeats (18 columns):**
  lpips_alex 2.28e-6, lpips_squeeze 1.50e-6, stlpips_alex 2.57e-6,
  dreamsim 1.35e-5, maniqa 8.35e-6, liqe 4.42e-5, hyperiqa 5.98e-6,
  paq2piq 3.13e-5, wadiqam_fr_live 2.37e-6, wadiqam_fr_tid 5.09e-6,
  wadiqam_nr_live 2.02e-7, wadiqam_nr_tid 6.81e-6, pieapp 9.19e-7,
  wd_s1..s5 ≤ 2.5e-6 JND (2026-10-17, i265, RTX 4080 via Vulkan).
  Together with the three PARTIAL columns and five LOADFAIL columns that
  is the full 26-column inventory; per-pair receipts:
  [`parity.tsv`](benchmarks/ort_webgpu_2026-10-17/parity.tsv).
- **DIVERGES (selection):** the adapter is `ZENMETRICS_NN_WEBGPU_DEVICE`
  (index into the EP's GPU-device list) else the best vendor-name rank
  (NVIDIA > AMD/ATI > other); more than one equally-ranked GPU without
  the pin is an `Error::Engine` naming every candidate, since a silent
  pick between equals is not reproducible. The safe ort API exposes no
  per-adapter discreteness flag — the rank and the pin are the
  resolution.

## Maintenance rule

Every metric port or reference-version bump updates this file **in the
same commit**: a ledger row (or delta refresh) plus a per-metric entry
for anything that isn't pure float noise. Crate-level depth stays in the
crate docs — this file links, it does not duplicate.


## PyIQA SSIM precision conventions observed 2026-10-19

The [twenty-name raw RGB grid](benchmarks/port_spotcheck_2026-10-18/primary-pyiqa-ssim.md)
retains 480 named scalar observations. The captured PyIQA 0.1.16 default dtype
constructs a float32 Gaussian before promotion. Its original rounded SSIM class
also promotes float32 luma coefficients. Selected-case controls through the
unchanged original functions, passing a float64 Gaussian and (where applicable)
float64 rounded luma, reduce the three observed differences from
1.5453514410235236e-5 / 1.727458853739705e-7 / 3.8032963667333775e-8 to
6.994405055138486e-15 / 3.0309088572266774e-14 / 6.40884012526044e-11.
These are diagnostic parameter controls, not replacement reference outputs or
new acceptance bounds. The original primary scores reproduce exactly and are
retained. On October 19 the user authorized selecting reasonable external
bounds: 1e-6 absolute for eighteen unrounded/direct/fixed routes and 5e-5 for
the two original rounded classes. The separately retained
[policy-applied view](benchmarks/port_spotcheck_2026-10-18/approved-pyiqa-ssim.json)
has 390 candidate scalar passes and 90 matching size refusals; the baseline
has twelve numerical misses and six refusal mismatches. The original boundless
tables and all existing Rust golden expectations remain unchanged. Map and
timing qualification remain open.

## Primary VIFp RGB precision finding, 2026-10-19

The [full frozen VIFp scalar grid](benchmarks/port_spotcheck_2026-10-18/primary-vif.md)
returns scores in all 24 cases, with 22 differences above the existing 1e-9
absolute gate. Largest |delta| is 4.093188010756421e-9 on the 1024-squared photo
pair, at both thread budgets. A private f32 luma coefficient array promoted
into the f64 pipeline is the candidate explanation; the f64 correction is
being evaluated on unmerged `fix/vif-rgb-luma-precision`. It is not yet a
verified fix. The primary source and every original numerical gate remain
unchanged; `vifvec` behavior is a separate required before/after control.

### Supported-runtime followup for the local HDR-v2 diagnostic (2026-10-21)

The [two original-only controls](benchmarks/canonical_recheck_2026-10-19/hdr-octave111-v2-controls.json)
use Octave 11.1.0 with the full upstream image 2.20.1 package. The 1024² photo
original is 61.71098098074218 versus retained native 61.701629129628834, a
0.009351851113343912 difference. The original's legacy-runtime difference is
7.105427357601002e-15. This rules out that legacy runtime as the explanation
for this cell; the cause of the native difference remains unresolved. The
0.005 existing golden-fixture level is unchanged and is not a universal bound.
These controls provide no comparable timing ratio or native parity acceptance.
The earlier partial grid and its runtime qualifications remain retained.

### Local BRISQUE 64² numerical failure (2026-10-21)

The [64² photo comparison](benchmarks/canonical_recheck_2026-10-19/brisque-64-numerical-finding.json)
returns native 75.64959716796875 versus PIQ 0.8.0/torch 2.14.1+cpu
75.73614501953125. Absolute difference 0.0865478515625 exceeds the unchanged
0.015138428 bound in all three blocks at both one and 28 workers. Ratios are
withheld. The 255×257 photo remains within that bound. Existing BRISQUE
validation uses a different torch wheel on CPU; controlled runtime/ISA
diagnostics are pending and do not waive this measured failure. Cause unresolved.
