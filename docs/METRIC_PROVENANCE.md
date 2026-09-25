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
| `ssim2` | sibling `fast-ssim2` (path `../fast-ssim2/fast-ssim2`, `imgref`) — **not** crates.io `ssimulacra2`; the crates.io dep exists only inside `ssim2-gpu` for reference checks | SSIMULACRA2, Cloudinary (2022+, no paper) | Cloudinary `ssimulacra2` C++ (v2 lineage) | sibling crate (algorithm reimplemented in that repo, SIMD-native) | sRGB8 | ~0–100, higher better |
| `ssim2-gpu` | `ssim2-gpu` (in-tree GPU twin) | " | `fast-ssim2` (CPU) + crates.io `ssimulacra2` for parity | in-tree GPU reimplementation | sRGB8 | " |
| `dssim` | crates.io `dssim-core` ^3.4 | Wang et al. MS-SSIM 2003 (porneL's variant) | dssim-core itself (the canonical impl) | third-party crate | sRGB8 | 0 best, unbounded |
| `dssim-gpu` | `dssim-gpu` (in-tree twin) | " | dssim-core | in-tree GPU | " | " |
| `butteraugli` | sibling `butteraugli` | Alakuijala et al. 2017, doi:10.1117/12.2272310 | libjxl butteraugli v0.9.2 / 0.9.4 | sibling crate (wraps the libjxl implementation) | sRGB8 | distance; emits `*_max` + `*_pnorm3` |
| `butteraugli-gpu` | `butteraugli-gpu` | " | same | in-tree GPU | " | " |
| `cvvdp` | in-tree `crates/cvvdp` | Mantiuk et al., ColorVideoVDP, ACM SIGGRAPH 2024 | `pycvvdp` **v0.5.7** (PyPI) | reference reimplementation (native SIMD CPU) | sRGB8 + display preset (default `standard_4k`) | JOD 0–10 |
| `cvvdp-gpu` | `cvvdp-gpu` twin | " | pycvvdp v0.5.7 + CPU twin | in-tree GPU | " | " |
| `hdrvdp` | in-tree `crates/hdrvdp` | Mantiuk et al., HDR-VDP-2, ACM TOG 30(4) 2011 | official HDR-VDP **2.2.2** MATLAB release (ISC-licensed) | reference reimplementation | absolute luminance cd/m² + display params — **not** sRGB8 | JOD / P_det / Q |
| `vmaf` (lib only) | in-tree `crates/vmaf` | Li et al., Netflix VMAF, 2016 | libvmaf **3.2.1** via FFI oracle | reference reimplementation | YUV420 planes (studio-swing ingress); v0 models luma-only | VMAF 0–100 + features (adm2, motion2, vif×4) |
| `vif` | in-tree `crates/vif` | Sheikh & Bovik, IEEE TIP 15(2) 2006 | authors' `vifp_mscale.m` (MATLAB) | reference reimplementation (f64 pipeline) | sRGB8 → MATLAB-weights gray | [0,∞), 1 ≈ identical |
| `msssim` | in-tree `crates/msssim` | Wang, Simoncelli, Bovik 2003 (ACSSC) | authors' `ssim_mscale_new.m` | reference reimplementation | sRGB8 → BT.601 luma | [0,1] |
| `iwssim` | in-tree `crates/iwssim` | Wang & Li, IEEE TIP 20(5) 2011, doi:10.1109/TIP.2010.2096950 | `Python-IW-SSIM` f9de37c (community reimpl of the lost MATLAB ref) | reference reimplementation | sRGB8 → BT.601 gray | [0,1] |
| `iwssim-gpu` | `iwssim-gpu` twin | " | CPU twin | in-tree GPU | " | " |
| `gmsd` | in-tree `crates/gmsd` | Xue et al., IEEE TIP 23(2) 2014 | `libgmsd` de646c9a (C) | reference reimplementation | sRGB8 → luma | 0 best, unbounded |
| `psnrhvs` / `psnrhvs-y` | in-tree `crates/psnrhvs` | Ponomarenko/Egiazarian PSNR-HVS-M (VPQM 2007) | authors' `psnrhvsm.m` | reference reimplementation | RGB per-channel 8×8 DCT (`-y` = luma plane) | dB, higher better |
| `haarpsi` / `haarpsi-y` | in-tree `crates/haarpsi` | Reisenhofer et al., Sci. Rep. 2018, doi:10.1038/s41598-018-21354-2 | authors' `haarpsi.m` | reference reimplementation | RGB (YIQ chroma) / luma (`-y`) | [0,1] |
| `fsim` / `fsim-y` | in-tree `crates/fsim` | Zhang et al., IEEE TIP 20(8) 2011, doi:10.1109/TIP.2011.2109730 | authors' `FR_FSIMc.m` | reference reimplementation | sRGB8 (FSIMc chroma terms) / luma (`-y`) | [0,1] |
| `vsi` | in-tree `crates/vsi` | Zhang et al., IEEE TIP 23(10) 2014 | authors' `VSI.m` | reference reimplementation | sRGB8 (SDSP colour saliency) | [0,1] |
| `mad` | in-tree `crates/mad-iqa` | Larson & Chandler, JEI 19(1) 2010 | official MATLAB release (larschandler.com) | reference reimplementation | sRGB8 → luma | distance; emits `mad`, `mad_hi`, `mad_lo` |
| `zensim` | sibling `zensim` crate | — | — | **in-house** (ML-trained; not a reproduction target) | sRGB8 | 0–100 |
| `zensim-gpu` | `zensim-gpu` twin | — | — | in-house | " | " |

Notes on tricky provenance:

- `ssim2`: the CLI resolves through `zenmetrics-api` `cpu-ssim2` →
  `fast-ssim2 = { version = "0.8.2", path = "../fast-ssim2/fast-ssim2" }` —
  the **local sibling always wins** (CLI Cargo.toml: "scores use local,
  crates versions banned"). `crates.io ssimulacra2` (0.5/0.8.x) appears only
  in `ssim2-gpu` for reference parity — never on the scoring path.
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

| metric | oracle | golden parity gate | observed worst | dataset reproduction (AIC-4, S01_AVIF_01) |
|---|---|---|---|---|
| cvvdp | pycvvdp 0.5.7 | ≤ 1e-3 JOD | ~1e-4 rel / 1e-3 JOD | **2e-5 JOD** vs `CVVDP` (std_fhd) |
| hdrvdp | HDR-VDP 2.2.2 MATLAB | P_det 1e-4; JOD ~1e-2 cases | 1.4e-5 P_det | not reproduced (needs abs-nits config) |
| vmaf | libvmaf 3.2.1 FFI | feats 1e-4, score 0.02 | ≪ gate | VMAF 0.002, neg 0.001, adm2 4e-6, vif_Σ 3e-5 |
| vif | `vifp_mscale.m` | goldens f64 | 5e-13 | n/a (`VIF` col = vifvec — different algo) |
| msssim | `ssim_mscale_new.m` | 15 Octave goldens | 8.8e-6 | MS-SSIM-pyiqa 3e-7; libvmaf col 4.8e-5 via `--luma-ingress` |
| iwssim | Python-IW-SSIM f9de37c | ~1e-4 | within gate | 3e-5 vs pyiqa col via `--luma-ingress` |
| gmsd | libgmsd de646c9a | 1e-12 rel (map), 1e-14 mean | ~1e-12 | 1e-7 |
| psnrhvs | `psnrhvsm.m` | 1e-3 dB | ≤ 4e-4 dB | col is Daala variant — different algo |
| haarpsi | `haarpsi.m` | 5e-5 | ~1e-5 | 8e-6 |
| fsim/-y | `FR_FSIMc.m` | 5e-5 | within gate | 1e-6 / 4e-6 (`-y` vs `FSIM` studio-Y) |
| vsi | `VSI.m` | 1e-4 | within gate | 2e-6 |
| mad-iqa | official MATLAB | 13 goldens | rel: hi 2.4e-7, lo 1.8e-6 | not in AIC-4 |
| ssim2 | fast-ssim2 ↔ C++ | bit-parity SIMD tiers | — | 0.043 vs col — version drift |
| dssim | dssim-core | upstream | — | 5e-7 |

**Acceptable-residual rules of thumb:**

- Golden parity: gate at the measured floor × ~3, documented per test;
  identical-input invariants (1.0 / 0 / NaN semantics) are exact.
- AIC-4 reproduction: ≤ ~1e-4 on a [0,1] similarity, ≤ ~0.01 dB, ≤ ~0.005
  JOD, ≤ ~0.01 VMAF points counts as *identified*; anything larger is a
  convention/algorithm mismatch to investigate, not hand-wave.
- `yuv601-studio` vs pyiqa `to_y_channel`: u8 luma rounding alone explains
  ~3e-5 on [0,1] scores — that's the inherent floor of the mode.
- f32-planes/f64-pooling vs an f64 MATLAB reference: ~1e-6–1e-4 rel is the
  normal band; vif is f64 end-to-end precisely because its 1e-10 masks sit
  below the f32 noise floor.

## 4. JPEG AIC-4 reproduction matrix (metrics_fullres.tab)

Measured on `S01_Ref_00.png` (1769×1988) vs `S01_AVIF_01.png` through this
CLI + probes, against the published AIC-4 row. Ingress conventions inferred:
libvmaf-family = studio-swing BT.601 YUV (u8-rounded, even-crop); pyiqa =
same transform unrounded (`to_y_channel`); MATLAB-family = own rgb2gray.

| AIC-4 column | published | ours | Δ | status |
|---|---|---|---|---|
| `CVVDP` (`standard_fhd`) | 9.9358 | 9.935776 | 2e-5 | ✅ |
| `VMAF`, `VMAF-neg`, `ADM2`, `VIF_vmaf` | 95.746 / 94.847 / 0.994208 / 3.820725 | 95.744 / 94.846 / 0.994204 / 3.8207 | ≤0.002 | ✅ |
| `PSNR` (RGB mean-MSE) | 38.9463 | 38.9466 | 4e-4 | ✅ |
| `PSNR-Y` (rounded studio-Y) | 43.645957 | 43.6459 | ~0 | ✅ |
| `PSNR-YCbCr611` | 43.9036 | 43.9098 | 0.006 | ✅ |
| `SSIM` (libvmaf `float_ssim`, scale=2) | 0.997922 | 0.997991 (numpy repro) | 7e-5 | ✅ identified — not yet ported |
| `MS-SSIM` (libvmaf `float_ms_ssim`) | 0.998818 | 0.998866 (`msssim --luma-ingress yuv601-studio`) | 4.8e-5 | ✅ |
| `MS-SSIM-pyiqa` | 0.9987947 | 0.998795 | 3e-7 | ✅ |
| `SSIMc` (MATLAB `ssim` RGB-mean) | 0.9774163 | 0.977432 (numpy repro) | 1.6e-5 | ✅ identified — no crate |
| `GMSD` | 0.0023320 | 0.002332 | 1e-7 | ✅ |
| `FSIM` / `FSIMc` | 0.9999090 / 0.9998984 | 0.999908 / 0.999897 | ~1e-6 | ✅ |
| `HaarPSI` | 0.9924069 | 0.992399 | 8e-6 | ✅ |
| `VSI` | 0.9999610 | 0.999963 | 2e-6 | ✅ |
| `DSSIM` | 0.00042453 | 0.000425 | 5e-7 | ✅ |
| `IW-SSIM` (pyiqa, unrounded studio-Y) | 0.9989235 | 0.998891 (`iwssim --luma-ingress yuv601-studio`) | 3e-5 | ✅ |
| `SSIMULACRA2` | 87.8839 | 87.927 | 0.043 | ⚠️ version drift |
| `JND_CVVDP` | 0.1976 | 0.1977 | — | ✅ transform verified |

| column family | published | what it is | status |
|---|---|---|---|
| `PSNR-HVS`, `-Y`, `-Cb`, `-Cr` | 50.05 / 50.86 / 47.49 / 48.10 | **Daala/Xiph `dump_psnrhvs`** layout — not our `psnrhvsm` port (Ponomarenko masked-Y gives 52.65) | ⬜ new impl needed — **CTC anchor** |
| `VIF` | 0.934422 | pyiqa **wavelet vifvec** (SP5 steerable pyramid), not pixel `vifp` (ours: 0.8535 studio-Y) | ⬜ new impl needed |
| `HDR_VDP_2` | 68.806 | hdrvdp-2.2.x under an unknown sRGB→nits display config | ⬜ config unknown |
| `HDR_VDP_3` | 9.620 | HDR-VDP-3 — different metric version | ⬜ new impl |
| `mDCT-PSNR` | 71.529 | Richter, QoMEX 2009; author's C++ ref impl `thorfdbg/mDCTpsnr` (~60KB, Highway SIMD) | ⬜ feasible medium port |
| `CW-SSIM`, `NLPD`, `CIEDE2000`, `FLIP` | — | pyiqa / colour / HDR metrics | ⬜ |
| `DISTS`, `LPIPS`×4, `PieAPP`, `WaDIQaM`, `DeepDC`, `DreamSim`, `TOPIQ`×2, `AHIQ`, `STLPIPS`×2 | — | torch models | ⬜ out of scope (no torch) |
| `proposal-*` (Butteraugli, DVIFM, mDCTPSNR) | — | AIC-4 **submitted** metrics — not public impls | ⬜ unreproducible by design |

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

## 6. Gaps & candidate work (priority order)

1. **Daala `dump_psnrhvs`** — the published `PSNR-HVS` family is a CTC
   anchor (drives `JND_PSNR-HVS`); verify against the -Y/-Cb/-Cr triple,
   add alongside (not replacing) `psnrhvs`.
2. **libvmaf `float_ssim` / `float_ms_ssim`** (with `scale`) in `crates/vmaf`
   — anchors `SSIM`/`MS-SSIM`; small (tdistler iqa decimate + l·c·s).
3. **`vifvec`** (wavelet VIF) — new port for the `VIF` column.
4. **`mDCT-PSNR`** — author's official C++ impl exists; DCT-domain masking +
   pooling, moderate scope; license check first.
5. **HDR-VDP-3** + the AIC HDR_VDP_2 display config.
6. `CW-SSIM`, `NLPD`, `CIEDE2000`, `FLIP` — later.

Validation probes used for this matrix live at
`crates/vmaf/examples/aic_probe.rs` (studio-Y YUV420 → `VmafV0Scorer`) and
`crates/vif/examples/aic_vif.rs` (candidate luma conventions through
`vif_plane_f32`); both take raw-RGB dumps as argv.
