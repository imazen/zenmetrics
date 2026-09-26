# Metric Provenance, Citations, and Validation Tolerances

One table per concern: (A) what each `--metric` actually is — paper, reference
implementation, and *how* our Rust code was produced; (B) the oracle each
implementation is validated against and the tolerance we gate; (C) the
tolerance policy — what residual we consider OK and why; (D) the JPEG AIC-4
reproduction matrix measured against the published dataset.

Related docs: `DIVERGENCES.md` (known deviations vs each reference version),
`docs/PARITY_TOLERANCES.md` (internal GPU-vs-CPU / strip-vs-whole parity
tolerances — a different axis from the reference-parity tolerances here).

## 1. Provenance classes

We do not call everything a "port". Terms used below:

- **third-party crate** — we wrap a published Rust crate unmodified.
- **sibling crate** — we wrap a locally maintained crate in this workspace
  (`../fast-ssim2`, `../butteraugli`, `../zensim`, …). The implementation
  lives outside this repo; provenance of the *algorithm* is recorded there.
- **reference reimplementation** — rewritten in Rust against an official
  reference implementation's source (MATLAB `.m`, C, C++), preserving its
  algorithm, constants, and edge-case semantics; validated against goldens
  produced by running that reference. Structural differences are
  engineering-only (SIMD, threading, allocation), not algorithmic.
- **clean-room (paper)** — implemented from the published paper's math, not
  from reference code; validated against published numbers or datasets.
- **in-house** — original metric developed in this workspace.

## 2. Implementation ledger

| `--metric` | crate | metric paper | reference impl (validation oracle) | provenance | input convention | output |
|---|---|---|---|---|---|---|
| `ssim2` | sibling `fast-ssim2` (path `../fast-ssim2/fast-ssim2`, `imgref`) — **not** crates.io `ssimulacra2`, which is a third-party crate we do not maintain; that dep exists only inside `ssim2-gpu` for reference checks | SSIMULACRA2, Cloudinary (2022+, no paper) | Cloudinary `ssimulacra2` C++ (v2 lineage) | sibling crate (algorithm reimplemented in that repo, SIMD-native) | sRGB8 | ~0–100, higher better |
| `ssim2-gpu` | `ssim2-gpu` (in-tree GPU twin) | " | `fast-ssim2` (CPU) + third-party crates.io `ssimulacra2` for parity | in-tree GPU reimplementation | sRGB8 | " |
| `dssim` | crates.io `dssim-core` ^3.4 | Wang et al. MS-SSIM 2003 (porneL's variant) | dssim-core itself (the canonical impl) | third-party crate | sRGB8 | 0 best, unbounded |
| `dssim-gpu` | `dssim-gpu` (in-tree twin) | " | dssim-core | in-tree GPU | " | " |
| `butteraugli` | sibling `butteraugli` | Alakuijala et al. 2017, doi:10.1117/12.2272310 | libjxl butteraugli v0.9.2 / 0.9.4 | sibling crate (wraps the libjxl implementation) | sRGB8 | distance; emits `*_max` + `*_pnorm3` |
| `butteraugli-gpu` | `butteraugli-gpu` | " | same | in-tree GPU | " | " |
| `cvvdp` | in-tree `crates/cvvdp` | Mantiuk et al., ColorVideoVDP, ACM SIGGRAPH 2024 | `pycvvdp` **v0.5.7** (PyPI) | reference reimplementation (native SIMD CPU) | sRGB8 + display preset (default `standard_4k`) | JOD 0–10 |
| `cvvdp-gpu` | `cvvdp-gpu` twin | " | pycvvdp v0.5.7 + CPU twin | in-tree GPU | " | " |
| `hdrvdp` | in-tree `crates/hdrvdp` | Mantiuk et al., HDR-VDP-2, ACM TOG 30(4) 2011 | official HDR-VDP **2.2.2** MATLAB release (ISC-licensed) | reference reimplementation | absolute luminance cd/m² + display params — **not** sRGB8 | JOD / P_det / Q |
| `hdrvdp3` | in-tree `crates/hdrvdp` (`v3` module) | same HDR-VDP lineage — Mantiuk et al., ACM TOG 30(4) 2011, doi:10.1145/2010324.1964935 (v3 is the display-adaptive revision; upstream asks to cite the 2011 paper + version number) | official HDR-VDP **3.0.7** MATLAB release + jpeg-ai-qaf `VDP3/` numpy port @ `0628a6b` (verified equal — see DIVERGENCES) | reference reimplementation (f64 end-to-end) | absolute luminance cd/m², **explicit required viewing conditions** — ppd, task, encoding (rgb-bt709/bt2020/native/xyz/luminance), emission, surround, observer age | `q_jod` 0–10 (10 = identical) + P_map |
| `vmaf` (lib only) | in-tree `crates/vmaf` | Li et al., Netflix VMAF, 2016 | libvmaf **3.2.1** via FFI oracle | reference reimplementation | YUV420 planes (studio-swing ingress); v0 models luma-only | VMAF 0–100 + features (adm2, motion2, vif×4) |
| `vif` | in-tree `crates/vif` | Sheikh & Bovik, IEEE TIP 15(2) 2006 | authors' `vifp_mscale.m` (MATLAB) | reference reimplementation (f64 pipeline) | sRGB8 → MATLAB-weights gray | [0,∞), 1 ≈ identical |
| `vifvec` | in-tree `crates/vif` (`vifvec` module) | Sheikh & Bovik, IEEE TIP 15(2) 2006 — **the original vector-GSM formulation, a different algorithm from `vifp`** | authors' `vifvec.m` + matlabPyrTools `sp5Filters`/`buildSpyr`/`corrDn`/`vifsub_est_M`/`ind2wtree` — the implementation behind pyiqa/`IQA_pytorch` `VIFs` and the AIC-4 `VIF` column | reference reimplementation (f64, row-major-transcribed SP5 filters) | sRGB8 → u8-rounded `round(0.299R+0.587G+0.114B)` gray (cv2/MATLAB convention); `vifvec_plane_f64` for raw planes | [0,∞), 1 = identical |
| `msssim` | in-tree `crates/msssim` | Wang, Simoncelli, Bovik 2003 (ACSSC) | authors' `ssim_mscale_new.m` | reference reimplementation | sRGB8 → BT.601 luma | [0,1] |
| `ssim-libvmaf` | in-tree `crates/msssim` (`libvmaf` module) | Wang et al., IEEE TIP 2004 (as implemented by Z. Li / tdistler iqa) | libvmaf `float_ssim` @ f85a8536 — FFI oracle (`vmaf-head-sys` dev-dep) | reference reimplementation (integer-accumulating decimate + f32/f64 rounding points preserved) | sRGB8 → studio-swing BT.601 luma — the YUV `Y` plane libvmaf consumes; NOT the MATLAB `rgb2gray` luma | [0,1] |
| `msssim-libvmaf` | in-tree `crates/msssim` (`libvmaf` module) | Wang/Simoncelli/Bovik 2003 (libvmaf's 5-scale variant) | libvmaf `float_ms_ssim` @ f85a8536 — FFI oracle | reference reimplementation (9/7 LPF pyramid, per-scale f32 means, double `pow`) | " | [0,1] |
| `iwssim` | in-tree `crates/iwssim` | Wang & Li, IEEE TIP 20(5) 2011, doi:10.1109/TIP.2010.2096950 | `Python-IW-SSIM` f9de37c (community reimpl of the lost MATLAB ref) | reference reimplementation | sRGB8 → u8-rounded BT.601 gray (`0.2989/0.5870/0.1140`, `LumaConvention::Bt601Rounded`) | [0,1] |
| `iwssim-piq` | in-tree `crates/iwssim` (`piq_luma` params) | " | jpeg-ai-qaf `IW_SSIM` (torch port of the same Jack-guo ref) on the unrounded 0–255 Y — the AIC-4 `IW-SSIM` column; upstream mirror `/home/lilith/tmp/jpeg-ai-qaf-hdrvdp3` @ `0628a6b` | same code path, `LumaConvention::YiqUnrounded` ingress | sRGB8 → unrounded `0.299R+0.587G+0.114B` | [0,1] |
| `iwssim-gpu` | `iwssim-gpu` twin | " | CPU twin | in-tree GPU | " | " |
| `gmsd` | in-tree `crates/gmsd` | Xue et al., IEEE TIP 23(2) 2014 | `libgmsd` de646c9a (C) | reference reimplementation | sRGB8 → luma | 0 best, unbounded |
| `psnrhvs` / `psnrhvs-y` | in-tree `crates/psnrhvs` | Ponomarenko/Egiazarian PSNR-HVS-M (VPQM 2007) | authors' `psnrhvsm.m` | reference reimplementation | RGB per-channel 8×8 DCT (`-y` = luma plane) | dB, higher better |
| `psnrhvs-daala` | in-tree `crates/psnrhvs` (`daala` module) | Daala/Xiph PSNR-HVS (integer bin-DCT variant adopted by libvmaf; distinct from the Ponomarenko MATLAB `psnrhvs`) | libvmaf `psnr_hvs` feature @ f85a8536 — FFI oracle | reference reimplementation (column-first `od_bin_fdct8x8`, CSF tables `csf_y`/`cb420`/`cr420`, integer-product mask accumulation, f32 `ret`) | sRGB8 → studio-swing BT.601 **YUV444** planes — the AIC-4 convention (`psnr_hvs_daala_yuv420` also exposed for YUV420 pictures) | dB; emits `psnrhvs_daala_{y,cb,cr}` + combined |
| `haarpsi` / `haarpsi-y` | in-tree `crates/haarpsi` | Reisenhofer et al., Sci. Rep. 2018, doi:10.1038/s41598-018-21354-2 | authors' `haarpsi.m` | reference reimplementation | RGB (YIQ chroma) / luma (`-y`) | [0,1] |
| `fsim` / `fsim-y` | in-tree `crates/fsim` | Zhang et al., IEEE TIP 20(8) 2011, doi:10.1109/TIP.2011.2109730 | authors' `FR_FSIMc.m` | reference reimplementation | sRGB8 (FSIMc chroma terms) / luma (`-y`) | [0,1] |
| `vsi` | in-tree `crates/vsi` | Zhang et al., IEEE TIP 23(10) 2014 | authors' `VSI.m` | reference reimplementation | sRGB8 (SDSP colour saliency) | [0,1] |
| `mad` | in-tree `crates/mad-iqa` | Larson & Chandler, JEI 19(1) 2010 | official MATLAB release (larschandler.com) | reference reimplementation | sRGB8 → luma | distance; emits `mad`, `mad_hi`, `mad_lo` |
| `mdctpsnr` | in-tree `crates/mdctpsnr` | Richter, "An Autoregressive Multi-DCT Domain Image Quality Metric", QoMEX 2009 (authored impl) | author's official C++ `thorfdbg/mDCTpsnr` (zlib-style license), built GCC `-O3 -ffast-math` AVX2 + glibc libmvec | reference reimplementation — parity target is the **compiled** binary's op order (reassociations/FMA/rcp-NR/libmvec decoded from disassembly), not the source's apparent semantics | sRGB8 → linear → BT.601 YCbCr | dB, higher better; `+inf` identical |
| `nlpd` | in-tree `crates/nlpd` | Laparra et al., normalized Laplacian pyramid distance (laparra16a-preprint) | Valerolaparra/`NLPD_Pytorch` (the authors' PyTorch reference; PyTorch 2.5.1 oracle) | reference reimplementation | sRGB8 → RGB planes `[0,1]`; six levels, `(Σ rms^0.6)^(1/0.6)` pooling | distance; 0 = identical |
| `nlpd-iqa` | in-tree `crates/nlpd` (`iqa` module) | " — the **dingkeyan93 `IQA_pytorch`** configuration (`NLPD(channels=1)`, alexhepburn nlpd-tensorflow lineage), NOT pyiqa and NOT the Laparra RGB reference | `IQA_pytorch` source mirrored at `/tmp/nlpd_orig.py`; semantics replicated to med \|Δ\| 0.0026 / max 0.0067 vs the AIC-4 `NLPD` column (n=53) | reference reimplementation (f32, pad-2 downsample phase, bilinear `align_corners` upsample, mean-of-RMS pooling) | sRGB8 → BT.709 Y `[0,1]` re-quantized to the 8-bit grid (`iqa::score_rgb_u8`); `iqa::score_y_f32` for caller-supplied planes | distance; 0 = identical |
| `zensim` | sibling `zensim` crate | — | — | **in-house** (ML-trained; not a reproduction target) | sRGB8 | 0–100 |
| `zensim-gpu` | `zensim-gpu` twin | — | — | in-house | " | " |

Notes on tricky provenance:

- `ssim2`: the CLI resolves through `zenmetrics-api` `cpu-ssim2` →
  `fast-ssim2 = { version = "0.8.2", path = "../fast-ssim2/fast-ssim2" }` —
  the **local sibling always wins** (CLI Cargo.toml: "scores use local,
  crates versions banned"). `fast-ssim2` is the Imazen crate; crates.io
  `ssimulacra2` is a *third-party* crate — it appears only inside `ssim2-gpu`
  for reference parity and never reaches the scoring path.
- `iwssim`: the authors' MATLAB source is not publicly distributed; the
  validation oracle is the widely-used Python reimplementation pinned to
  commit f9de37c — recorded honestly as a second-order oracle.
- `vmaf`: model weights are the published libvmaf `vmaf_v0.6.1` /
  `vmaf_v0.6.1neg` / `vmaf_4k_v0.6.1(+neg)` pickles converted at build time;
  feature extraction (adm2, motion2, vif×4) is our reimplementation.

## 3. Validation tolerances — what we gate and what we consider OK

Tolerance classes (keep these distinct — they mean different things):

- **golden parity** — the test-gate delta vs the reference oracle's output
  on our fixtures. This is the number that proves the reimplementation.
- **dataset reproduction** — measured delta vs a *published* score column
  (e.g. AIC-4), which additionally absorbs the publisher's ingress
  convention, rounding, cropping, and implementation-version drift. These
  are always looser than golden parity and identify *conventions*, not
  formula bugs.
- **internal parity** — GPU-vs-CPU, strip-vs-whole, SIMD-vs-scalar of the
  same implementation. Owned by `docs/PARITY_TOLERANCES.md`, not here.
- **ingress-variant shift** — the expected *score change* when the same
  metric is fed a different legitimate luma (house BT.601-full vs
  `yuv601-studio`). A convention difference, not a defect.

| metric | oracle | golden parity gate | observed worst | dataset reproduction (AIC-4, n=53 subset unless noted) |
|---|---|---|---|---|
| cvvdp | pycvvdp 0.5.7 | ≤ 1e-3 JOD | ~1e-4 rel / 1e-3 JOD | med 1.1e-4 / max 1.3e-3 JOD @ standard_fhd (n=53) |
| hdrvdp | HDR-VDP 2.2.2 MATLAB | P_det 1e-4; JOD ~1e-2 cases | 1.4e-5 P_det | not reproduced (needs abs-nits config) |
| hdrvdp3 | HDR-VDP-3.0.7 MATLAB + jpeg-ai-qaf `VDP3/` port | 21-case golden corpus | 1.6e-13 JOD worst |Q_JOD| | published `HDR_VDP_3` col NOT reproduced — we match MATLAB 3.0.7 exactly (3 pairs bit-equal at print precision); the column is an unrecovered run config — see DIVERGENCES |
| vmaf | libvmaf 3.2.1 FFI | feats 1e-4, score 0.02 | ≪ gate | VMAF 0.002, neg 0.001, adm2 4e-6, vif_Σ 3e-5 (S01 pt) |
| vif | `vifp_mscale.m` | goldens f64 | 5e-13 | n/a (`VIF` col = vifvec — different algo, med 0.06) |
| vifvec | `vifvec.m` via Octave | 10 goldens < 1e-9 | within gate | med 1.9e-6 / max 6.3e-6 vs `VIF` col (n=54) |
| msssim | `ssim_mscale_new.m` | 15 Octave goldens | 8.8e-6 | med 4e-4 vs libvmaf col / 1.2e-3 vs pyiqa col (n=53) |
| ssim-libvmaf | libvmaf `float_ssim` FFI | ≤ 2e-4 (vs vendored f85a8536, 5 cases 8b+10b) | ≪ gate | med 1.0e-6 / max 1.8e-5 vs `SSIM` col (n=53) |
| msssim-libvmaf | libvmaf `float_ms_ssim` FFI | ≤ 2e-4 (same harness) | ≪ gate | med 1.0e-6 / max 1.6e-5 vs `MS-SSIM` col (n=53) |
| iwssim | Python-IW-SSIM f9de37c | ~1e-4 | within gate | parent-band bug fixed 2026-09-26 — column reproduction now owned by `iwssim-piq` below |
| iwssim-piq | jpeg-ai-qaf `IW_SSIM` ingress | — (same code path) | — | med 3.9e-6 / max 2.6e-5 vs `IW-SSIM` col (n=54) |
| gmsd | libgmsd de646c9a | 1e-12 rel (map), 1e-14 mean | ~1e-12 | mean 3.5e-7 / max 1.9e-6 (n=53) |
| psnrhvs | `psnrhvsm.m` | 1e-3 dB | ≤ 4e-4 dB | ~7.6dB off (Daala variant — different algo, n=53) |
| psnrhvs-daala | libvmaf `psnr_hvs` FFI | ≤ 2e-4 dB (420 + 444 cases) | ≪ gate | combined med 1.1e-3 / max 4.5e-2 dB; Y med 1.3e-3 / max 8.6e-3; Cb 1.3e-2/8.0e-2; Cr 2.6e-3/1.3e-1 dB (n=53) |
| haarpsi | `haarpsi.m` | 5e-5 | ~1e-5 | med 8.4e-6 / max 5.3e-5 (n=53) |
| fsim/-y | `FR_FSIMc.m` | 5e-5 | within gate | med 2.5e-5 / max 1.7e-3 (n=53) |
| vsi | `VSI.m` | 1e-4 | within gate | med 9.6e-6 / max 1.1e-4 (n=53) |
| mad-iqa | official MATLAB | 13 goldens | rel: hi 2.4e-7, lo 1.8e-6 | not in AIC-4 |
| mdctpsnr | built `dctpsnr` binary (GCC `-O3 -ffast-math`, glibc 2.43, AVX2) | ≤ 2e-4 dB on x86_64+GNU goldens (64×48 + all 8 `(w−13)%8` classes), 1e-2 off-platform | ≤ ~4e-6 dB | med 4.0e-7 / max 1.13e-5 dB vs `mDCT-PSNR` col (n=53) |
| ssim2 | fast-ssim2 ↔ C++ | bit-parity SIMD tiers | — | med 6.3e-3 / max 5.5e-2 — version drift (n=53) |
| dssim | dssim-core | upstream | — | mean 2.5e-7 / max 6.5e-7 (n=53) |
| nlpd | `NLPD_Pytorch` (authors' PyTorch) | 1e-5 golden (97×99 pair) | within gate | n/a — RGB/0.6-norm variant, ≈20.5× the `NLPD` column (different algo) |
| nlpd-iqa | `IQA_pytorch` `NLPD(channels=1)` via numpy replica | 1e-4 golden (f64 replica → f32 ops) | ~4e-6 | med 2.6e-3 / max 6.7e-3 vs `NLPD` col, med-rel 2.63% (n=53) — residual is f32 op-order + torch bilinear edge detail |

**Acceptable-residual rules of thumb:**

- Golden parity: gate at the measured floor × ~3, documented per test;
  identical-input invariants (1.0 / 0 / NaN semantics) are exact.
- AIC-4 reproduction: ≤ ~1e-4 median on a [0,1] similarity, ≤ ~0.01 dB,
  ≤ ~0.005 JOD, ≤ ~0.01 VMAF points counts as *identified*; a systematic
  ~1e-3 offset means the column's implementation is a different *variant*
  (not an ingress problem — see `IW-SSIM`); anything larger is a different
  algorithm entirely.
- `yuv601-studio` vs pyiqa `to_y_channel`: u8 luma rounding alone explains
  ~3e-5 on [0,1] scores — that's the inherent floor of the mode.
- f32-planes/f64-pooling vs an f64 MATLAB reference: ~1e-6–1e-4 rel is the
  normal band; vif is f64 end-to-end precisely because its 1e-10 masks sit
  below the f32 noise floor.

## 4. JPEG AIC-4 reproduction matrix (metrics_fullres.tab)

Measured on `S01_Ref_00.png` (1769×1988) vs `S01_AVIF_01.png` through this
CLI + probes, against the published AIC-4 row, then extended to a 53-pair
stratified subset (S01/S03/S19/S38/S57 × {AVIF,JXL,JPG,J2K,WEBP} × levels
{02,09,16}). Local corpus: `/tmp/v_ro/input/datasets/aic2026/` — the tower
NFS export `tower:/mnt/user/coefficient` mounted read-only — holds the full
zip corpus (`AIC2026-dataset-complete.zip`, 9,620 distorted + 70 sources)
plus `metrics_fullres.csv` / `metrics_cropped.csv` and
`encoding_recipes.md`; individual images stream-extract with
`unzip -p <zip> distorted/<name>.png`. Ingress conventions inferred:
libvmaf-family = studio-swing BT.601 YUV (u8-rounded, even-crop); pyiqa =
same transform unrounded (`to_y_channel`); MATLAB-family = own rgb2gray.

Deltas below are |ours − published| over the 53-pair subset (`mean`/`median`/`max`);
near-transparent rows compress toward 1.0/0 so medians are the honest center.
The single-point S01_AVIF_01 spot check is kept where it drove identification.

| AIC-4 column | published impl | med\|Δ\| | max\|Δ\| | status |
|---|---|---|---|---|
| `CVVDP` (`standard_fhd`) | pycvvdp | 1.1e-4 | 1.3e-3 | ✅ strong at full quality range |
| `VMAF`, `VMAF-neg`, `ADM2`, `VIF_vmaf` | libvmaf v0.6.1 | ≤0.002 (S01 pt) | — | ✅ |
| `PSNR`, `PSNR-Y`, `PSNR-YCbCr611` | RGB-MSE / rounded studio-Y / 6:1:1 | ≤0.006 (S01 pt) | — | ✅ |
| `GMSD` | libgmsd lineage | 2.8e-7 | 1.9e-6 | ✅ essentially exact |
| `DSSIM` | dssim-core | 2.7e-7 | 6.5e-7 | ✅ essentially exact |
| `HaarPSI` | authors' matlab | 8.4e-6 | 5.3e-5 | ✅ |
| `VSI` | authors' matlab | 9.6e-6 | 1.1e-4 | ✅ |
| `FSIM` / `FSIMc` | authors' matlab | 2.5e-5 | 1.7e-3 | ✅ |
| `SSIM` | libvmaf `float_ssim` | 1.0e-6 | 1.8e-5 | ✅ implemented — `ssim-libvmaf` (FFI-verified port) |
| `SSIMc` | MATLAB `ssim` RGB-mean | 1.6e-5 (S01 pt, numpy repro) | — | ✅ identified — no crate |
| `MS-SSIM` | libvmaf `float_ms_ssim` | 1.0e-6 | 1.6e-5 | ✅ implemented — `msssim-libvmaf` (FFI-verified port; was the 2.64-JND gap) |
| `MS-SSIM-pyiqa` | pyiqa `ms_ssim` | 1.2e-3 | 6.4e-3 | ✅ tracks; same impl-family residual |
| `IW-SSIM` | jpeg-ai-qaf `IW_SSIM` on unrounded Y | 3.9e-6 | 2.6e-5 | ✅ implemented — `iwssim-piq` (the +1.2e-3 gap was a Gaussian-vs-Laplacian parent-band bug + ingress, both resolved) |
| `SSIMULACRA2` | "v2.1" | 6.3e-3 | 5.5e-2 | ✅ drift immaterial in JND (≤0.004 — §4.1) |
| `JND_CVVDP` | transform `3.1889·(10−JOD)^1.0129` | — | — | ✅ transform verified |

| column family | published | what it is | status |
|---|---|---|---|
| `PSNR-HVS`, `-Y`, `-Cb`, `-Cr` | 50.05 / 50.86 / 47.49 / 48.10 | **Daala/Xiph `dump_psnrhvs`** — implemented as `psnrhvs-daala` (FFI-verified); on studio-601 **YUV444** chroma: combined med 1.1e-3 / max 4.5e-2 dB, Y med 1.3e-3 / max 8.6e-3 dB | ✅ implemented |
| `VIF` | 0.934422 | `IQA_pytorch.VIFs` / `vifvec.m` family (SP5 steerable pyramid vecGSM) on u8-rounded Y — NOT pixel `vifp` | ✅ implemented — `vifvec` (Octave goldens < 1e-9; published column: med 1.9e-6 / max 6.3e-6 (n=54)) |
| `HDR_VDP_2` | 68.806 | hdrvdp-2.2.x under an unknown sRGB→nits display config | ⬜ config unknown — misses our 2.2.2-validated v2 at ppd 30 *and* 64.05 (±5.7), so the published run used a different config than either default |
| `HDR_VDP_3` | 9.620 | HDR-VDP-3.0.7 lineage; the jpeg-ai-qaf `VDP3/` numpy port (mirror `/home/lilith/tmp/jpeg-ai-qaf-hdrvdp3` @ `0628a6b`) is confirmed **bit-faithful** to MATLAB 3.0.7 | ⚠️ implemented — `hdrvdp3` — but column **unreproduced**: under the stated harness ingress we get 9.6968/8.5174/9.6453 = MATLAB 3.0.7 exactly, while published reads 9.3392/7.5986/9.3527. Full probe sweep (ppd 28–64, task, EOTF, reflectance, swap, bsc) leaves distortion-proportional residuals → different impl revision/config upstream. See DIVERGENCES |
| `mDCT-PSNR` | 71.529 | Richter, QoMEX 2009; author's C++ ref impl `thorfdbg/mDCTpsnr` | ✅ implemented — `mdctpsnr` (compiled-binary parity: med 4.0e-7 / max 1.13e-5 dB over n=53; see DIVERGENCES for the codegen-order details) |
| `NLPD` | ~0.03–0.22 | `IQA_pytorch` `NLPD(channels=1)` (dingkeyan93, not pyiqa) — BT.709 Y `[0,1]` 8-bit-gridded, mean-of-RMS pooling | ✅ implemented — `nlpd-iqa` (med 2.6e-3 / max 6.7e-3, med-rel 2.63%, n=53; `nlpd` is the RGB/Laparra variant — ≈20.5× off, deliberately distinct) |
| `CW-SSIM`, `CIEDE2000`, `FLIP` | — | pyiqa / colour metrics | ⬜ |
| `DISTS`, `LPIPS`×4, `PieAPP`, `WaDIQaM`, `DeepDC`, `DreamSim`, `TOPIQ`×2, `AHIQ`, `STLPIPS`×2 | — | torch models | ⬜ out of scope (no torch) |
| `proposal-*` (Butteraugli, DVIFM, mDCTPSNR) | — | AIC-4 **submitted** metrics — not public impls | ⬜ unreproducible by design |

### 4.1 JND-normalized deltas

The dataset carries fitted `metric → JND` remappings (the 7 `JND_*` display
columns plus the benchmark fitting-tool's 91-metric power-law maps —
`a·(b−x)^c`). `docs/AIC2026_METRICS_AND_FITTING.md` records both mapping
families verbatim, our reconstructed `JND_*` coefficients, and the full
JND-normalized delta table. Headline findings on the same 53 pairs:

- Reproduction-grade in JND (med ≤0.01, max ≤0.05): GMSD, DSSIM, HaarPSI,
  CVVDP@`standard_fhd`, VMAF-neg, PSNR-Y, **SSIMULACRA2** — its 6.3e-3 raw
  drift is immaterial once remapped (≤0.004 JND); the version-drift worry
  was overblown.
- Acceptable (max ≤0.5 JND): VSI, FSIM, FSIMc.
- **Closed since**: `SSIM`/`MS-SSIM` via `ssim-libvmaf`/`msssim-libvmaf`
  (med 1.0e-6, max 1.8e-5 raw — the libvmaf ports are exact modulo u8
  ingress rounding; the 2.64-JND MS-SSIM tail was our MATLAB variant's
  family difference, now bypassed) and `PSNR-HVS` via `psnrhvs-daala`
  (combined med 1.1e-3 dB / max 4.5e-2 ≈ ≤0.004 JND — the previous
  Ponomarenko-vs-Daala 1.9–6.4 JND gap resolved; residual is the chroma
  ingress — studio-601 YUV444 matches their Cb/Cr to ≤0.13 dB).
- VIF + IW-SSIM now reproduce their columns (`vifvec`, `iwssim-piq`);
  remaining big gaps are the HDR/deep columns.

Priority order in §6 is JND-ranked.

## 5. `--luma-ingress yuv601-studio`

Shared ingress added for the libvmaf/pyiqa-luma column family:
`Y = round(16 + (65.481·R + 128.553·G + 24.966·B)/255)`, broadcast gray,
even-cropped. Applies only to `MetricKind::is_luma_only()` — `gmsd`,
`psnrhvs-y`, `haarpsi-y`, `fsim-y`, `msssim`, `vif`, `mad`, `iwssim`.
Colour metrics ignore it; `--hdr` forces `house`. Important: it is a
**reproduction aid, not a universal improvement** — MATLAB-convention
metrics (gmsd, haarpsi, fsim, vsi) match the published columns *better* on
their house luma and should stay on `house` unless a specific libvmaf-style
column is being targeted.

## 6. Gaps & candidate work (JND-ranked priority order)

~~1. **Daala `dump_psnrhvs`**~~ — ✅ **DONE** (`psnrhvs-daala`): integer
   bin-DCT + CSF masking port in `crates/psnrhvs/src/daala.rs`, FFI-verified
   vs vendored libvmaf f85a8536 (≤2e-4 dB, 420+444). Gotcha found in
   parity work: `od_bin_fdct8x8` is column-DCT-then-column-DCT over the
   transposed intermediate — row-first is *not* equivalent once the
   integer rounding is accounted for (~1.3% MSE / 0.059 dB).
   AIC-4 chroma convention: studio-601 **YUV444** (full-res chroma —
   verified via the -Cb/-Cr columns).
~~2. **libvmaf `float_ssim` / `float_ms_ssim`**~~ — ✅ **DONE**
   (`ssim-libvmaf`, `msssim-libvmaf`): ports live in
   `crates/msssim/src/libvmaf.rs` (the SSIM-family crate — `crates/vmaf`
   is a separate active lane). FFI-verified incl. 10-bit
   (`picture_copy_hbd` /4 scaling) and the auto-scale
   `round(min_dim/256)` decimate. Studio-601 luma is built in — the
   AIC-4 columns reproduce at med 1.0e-6 / max 1.8e-5.
~~3. **`vifvec`** (wavelet VIF)~~ — ✅ **DONE** (`vifvec`): clean-room
   port of `vifvec.m` + matlabPyrTools SP5 pyramid in
   `crates/vif/src/vifvec.rs`. Octave goldens < 1e-9 over 10 cases
   (odd dims, 72px floor, identity). RGB ingress = u8-rounded
   `0.299/0.587/0.114` — the convention the `VIF` column was
   computed under (match ~1e-6).
~~4. **pyiqa `iwssim` variant**~~ — ✅ **DONE** (`iwssim-piq`):
   `LumaConvention::YiqUnrounded` + the Laplacian-parent-band fix
   (a real bug vs both oracles — Gaussian had been fed instead).
   `IW-SSIM` column: med 3.9e-6 / max 2.6e-5 over n=54.
~~5. **`mDCT-PSNR`**~~ — ✅ **DONE** (`mdctpsnr`): port of
   `thorfdbg/mDCTpsnr` (zlib-style license). The catch that made it a
   "medium" port: bit-parity required reproducing the **compiled** binary's
   codegen — GCC `-ffast-math` reassociations, FMA contraction, `rcpps`+NR
   instead of division, libmvec `_ZGVdN8vv_powf`/`_ZGVdN8v_expf` vector
   calls, and the 8-lane strided pooling accumulator (whose `rem ≥ 4`
   tail guard cost ~11 dB on `w13 % 8 == 3` widths until decoded). AIC-4
   `mDCT-PSNR` column: med 4.0e-7 / max 1.13e-5 dB over n=53.
~~6. **HDR-VDP-3**~~ — ✅ **DONE** (`hdrvdp3`, `crates/hdrvdp/src/v3/`):
   clean Rust reimplementation of the jpeg-ai-qaf `VDP3/` numpy port,
   verified bit-faithful to **official MATLAB HDR-VDP-3.0.7** (Octave run,
   3 AIC-4 pairs identical at print precision; 21-case golden corpus ≤
   1.6e-13 JOD). Viewing conditions are explicit required parameters —
   `pix_per_deg` has no default (CLI `--hdrvdp3-ppd` is mandatory), plus
   task/input-encoding/emission/surround/observer-age. Absolute-luminance
   interleaved nits only (`IntegratedPuNits`); no sRGB8 path. The
   published `HDR_VDP_3` column itself remains unreproduced — exhaustive
   protocol probes show it came from a different run config than the
   shipped code's stated defaults; recorded in DIVERGENCES rather than
   fitted to. The AIC `HDR_VDP_2` display config is likewise still
   unknown (v2 validated vs official 2.2.2 regardless).
~~7. **`NLPD`**~~ — ✅ **DONE** (`nlpd-iqa`, `crates/nlpd/src/iqa.rs`):
   the column is `IQA_pytorch` `NLPD(channels=1)` (dingkeyan93 —
   alexhepburn nlpd-tensorflow lineage), a *different metric* from the
   crate's Laparra-official RGB `nlpd`: single BT.709 Y plane on the
   8-bit grid, reflect-2 downsample (fixed phase — not the parity-dependent
   pad), bilinear `align_corners` upsample to the 2× grid then
   nearest-crop on odd dims, six loop bands, mean-of-RMS pooling. Column
   residual med 2.6e-3 / max 6.7e-3 (n=53) ≈ f32 op-order. The RGB `nlpd`
   keeps its own score path and `nlpd` column name.
8. `CW-SSIM`, `CIEDE2000`, `FLIP` — later.

Validation probes used for this matrix live at
`crates/vmaf/examples/aic_probe.rs` (studio-Y YUV420 → `VmafV0Scorer`) and
`crates/vif/examples/aic_vif.rs` (candidate luma conventions through
`vif_plane_f32`); both take raw-RGB dumps as argv.
