# Display specifications

This document covers the display-side configuration surface added
to `cvvdp-gpu` for ColorVideoVDP-aligned scoring across SDR and
HDR pipelines: the EOTFs supported, the RGB primaries supported,
and the named presets shipped via [`DisplayModel::by_name`].

The intent is to mirror upstream pycvvdp v0.5.4's behaviour for
the still-image path. The corresponding upstream files are
vendored under `data/display_models.json` and
`data/color_spaces.json` (MIT-licensed; full attribution in
`data/THIRD_PARTY.md`).

## EOTFs (`Eotf`)

| Variant | Used by | Math |
| --- | --- | --- |
| `Srgb` | sRGB / BT.709 / `BT.709` color spaces; the v1 parity contract | `((V+0.055)/1.055)^2.4` (upper branch), `V/12.92` (V≤0.04045) |
| `Pq` | BT.2020-PQ presets (`standard_hdr_pq`, `65inch_hdr_pq_*`, `lg_oled_2026_hdr_pq`) | SMPTE ST 2084 `pq2lin(V) = 10000 * ((max(V^(1/m2) − c1, 0)) / (c2 − c3·V^(1/m2)))^(1/m1)` with `m1=0.15930175`, `m2=78.84375`, `c1=0.8359375`, `c2=18.8515625`, `c3=18.6875` |
| `Hlg` | BT.2020-HLG presets (`standard_hdr_hlg`, `iphone_14_pro_hdr*`) | BT.2100-1 Table 5: per-channel inverse OETF + system OOTF `Y_s^(γ−1)`. Gamma is 1.2 below 1000 cd/m² peak; above, uses the BBC WHP369 lift `γ = 1.2 + 0.42·log10(Y_peak/1000) − 0.07623·log10(E_ambient/5)` |
| `Linear` | BT.709-linear / luminance presets (`standard_hdr_linear*`) | Identity. Input already in cd/m² (or 0..1 linear-light). Clipped to `[max(0.005, Y_black), Y_peak]`, then offset by `Y_refl`. No `(Y_peak − Y_black)` scaling. |
| `Bt1886` | Reserved for explicit BT.1886 graders | Lifted gamma 2.4: `L = a · max(V + b, 0)^γ` with `a, b` chosen so `L(0)=Y_black`, `L(1)=Y_peak` |
| `Gamma(g)` | Color spaces with numeric EOTF strings (`Adobe RGB → "2.2"`, `Apple RGB → "1.8"`, etc.) | `(Y_peak − Y_black) · V^g + Y_black + Y_refl` |

The `Eotf::forward(v, y_peak, y_black, y_refl)` dispatcher mirrors
pycvvdp's `vvdp_display_photo_eotf.forward` branch-for-branch. PQ
and Linear output absolute cd/m² (no `(Y_peak − Y_black)` scaling
— the input is already absolute); sRGB, HLG, BT.1886, and Gamma
output `(Y_peak − Y_black) · inverse_EOTF(V) + Y_black + Y_refl`
(the input is relative).

Reference values verified by `tests/eotf_primaries_invariants.rs`:

- `pq_eotf_scalar(0.5) ≈ 92.25 cd/m²`
- `pq_eotf_scalar(1.0) ≈ 10000 cd/m²`
- `hlg_inverse_oetf_scalar(0.5) = 1/12`
- `hlg_inverse_oetf_scalar(1.0) = 1.0`
- sRGB seam at `V = 0.04045` continuous to 1e-6

## Primaries (`Primaries`)

| Variant | Used by | Per-stage matrix |
| --- | --- | --- |
| `Bt709` | sRGB, BT.709, BT.709-linear color spaces; default | `SRGB_LINEAR_TO_DKL` — bit-pinned (v1 parity contract) |
| `Bt2020` | BT.2020-PQ, BT.2020-HLG, BT.2020-linear | `BT2020_LINEAR_TO_DKL` |
| `DisplayP3` | "Display P3 Apple" color space | `DISPLAY_P3_LINEAR_TO_DKL` |
| `DciP3` | (alias for `DisplayP3` today — no theatrical DCI preset upstream) | `DISPLAY_P3_LINEAR_TO_DKL` |

Each matrix is computed at f64 precision from upstream's per-stage
matrices:

```
DKL = LMS2006_to_DKLd65 @ XYZ_to_LMS2006 @ RGB_to_XYZ
```

with `XYZ_to_LMS2006` and `LMS2006_to_DKLd65` from cvvdp's
`pycvvdp/display_model.py`, and `RGB_to_XYZ` from
`pycvvdp/vvdp_data/color_spaces.json` (BT.709, BT.2020, Display
P3 Apple). The BT.709 row in the dispatch table is bit-identical
to the existing `SRGB_LINEAR_TO_DKL` const so every v1 parity
test continues to pass without modification.

## Preset registry

`DisplayModel::by_name(name)` and `DisplayGeometry::by_name(name)`
load named displays from two vendored files:

* `data/display_models.json` — the 22 presets from upstream
  pycvvdp v0.5.4, plus 3 `65inch_hdr_pq_*` entries that match
  upstream's `main` style but are not in the 0.5.4 release.
* `data/display_models_imazen.json` — imazen-added reference
  conditions not in upstream (every entry's `source` is
  `"imazen"`).

**Provenance matters for interpretation.** Only `eizo_CG3146` is a
calibration-grounded display — it is the EIZO CG3146 that captured
the XR-DAVID subjective dataset ColorVideoVDP v0.5.4 was fit on
(arXiv 2401.11485). `standard_4k` is the documented metric default
and cross-study comparability anchor. The remaining `standard_*`
entries are canonical viewing conditions (`source: "none"`). Every
device-named entry (`iphone_*`, `ipad_*`, `macbook_*`,
`lg_oled_*`) is a manufacturer/review spec transcription with **no
subjective data behind it** — useful as a prediction target, but
carrying no research weight. The metric's calibration is
display-agnostic: it takes photometry as input and predicts for
whatever display you name.

| Preset | Resolution | Distance | Peak (cd/m²) | EOTF | Primaries | Provenance |
| --- | --- | --- | --- | --- | --- | --- |
| `standard_4k` | 3840×2160 | 0.7472 m | 200 | sRGB | BT.709 | Default / comparability anchor |
| `standard_hdr_pq` | 3840×2160 | 0.7472 m | 1500 | PQ | BT.2020 | |
| `standard_hdr_hlg` | 3840×2160 | 0.7472 m | 1500 | HLG | BT.2020 | |
| `standard_hdr_linear` | 3840×2160 | 0.7472 m | 1500 | Linear | BT.709 | |
| `standard_hdr_linear_dark` | 3840×2160 | 0.7472 m | 1500 | Linear | BT.709 | 0 lux ambient |
| `standard_hdr_linear_zoom` | 3840×2160 | 0.25 m | 10000 | Linear | BT.709 | Close-viewing artefact spotter |
| `standard_fhd` | 1920×1080 | 0.6 m | 200 | sRGB | BT.709 | |
| `standard_hmd` | 1440×1600 | — | 100 | sRGB | BT.709 | FOV-only; no `DisplayGeometry` |
| `standard_phone` | 2400×1080 | 0.4 m | 500 | sRGB | BT.709 | |
| `sdr_4k_30` | 3840×2160 | 0.6 m | 100 | sRGB | BT.709 | |
| `sdr_fhd_24` | 1920×1080 | 0.6 m | 100 | sRGB | BT.709 | |
| `htc_vive_pro` | 1440×1600 | — | 133.3 | sRGB | BT.709 | FOV-only |
| `iphone_12_pro` | 2532×1170 | 20" | 825 | sRGB | BT.709 | |
| `iphone_14_pro` | 2532×1170 | 20" | 1025 | sRGB | BT.709 | |
| `iphone_14_pro_vert` | 1170×2532 | 20" | 1025 | sRGB | BT.709 | Portrait orientation |
| `iphone_14_pro_hdr` | 2532×1170 | 20" | 1590 | HLG | BT.2020 | |
| `iphone_14_pro_hdr_vert` | 1170×2532 | 20" | 1590 | HLG | BT.2020 | |
| `ipad_pro_12_9` | 2732×2048 | 20" | 600 | sRGB | BT.709 | |
| `macbook_pro_16` | 3072×1920 | 25" | 500 | sRGB | BT.709 | |
| `lg_oled_2017_sdr` | 3840×2160 | 101" | 272 | sRGB | BT.709 | TV viewing distance |
| `lg_oled_2017_hdr` | 3840×2160 | 101" | 754 | sRGB | BT.709 | upstream omits `colorspace` |
| `eizo_CG3146` | 4096×2160 | 0.73406 m | 300 | sRGB | BT.709 | **Calibration display (XR-DAVID)** |
| `65inch_hdr_pq_4knit` | 3840×2160 | 1.98 m | 4000 | PQ | BT.2020 | upstream main, not 0.5.4 |
| `65inch_hdr_pq_2Knit` | 3840×2160 | 1.98 m | 2000 | PQ | BT.2020 | upstream main, not 0.5.4 |
| `65inch_hdr_pq_1Knit` | 3840×2160 | 1.98 m | 1000 | PQ | BT.2020 | upstream main, not 0.5.4 |
| `lg_oled_2026_hdr_pq` | 3840×2160 | 86.62" | 3000 | PQ | BT.2020 | local addition (avforums review) |
| `modern_oled_phone_indoor` | 2532×1170 | 0.35 m | 400 | sRGB | BT.709 | **imazen** — indoor SDR auto-brightness |

That's 27 presets total (22 upstream 0.5.4 + 3 upstream-main +
1 local + 1 imazen). All load both `DisplayModel::by_name` and
`DisplayGeometry::by_name`. The two FOV-only entries
(`standard_hmd`, `htc_vive_pro`) are converted via
`DisplayGeometry::from_fov_diagonal` (commit `1280571a`).

`modern_oled_phone_indoor` deserves a note: its `y_peak` of 400
cd/m² is the **SDR auto-brightness setpoint** for indoor viewing,
NOT the panel's HDR/sunlight peak (1000-2000 cd/m²). Modern phones
dim SDR content indoors; the headline peak is reserved for HDR
highlights and sunlight boost. The OLED's near-zero native black
(0.0005 cd/m²) is washed out by 250 lux ambient reflection
(`y_refl ≈ 0.398 cd/m²`), so effective contrast is ~1,000:1 — the
OLED's deep-black advantage only appears in dark-room viewing,
which this indoor preset doesn't model. Use `iphone_14_pro` /
`iphone_14_pro_hdr` only when you genuinely mean the panel peak
(sunlight/HDR), not indoor SDR.

## Scope of this release

| Surface | sRGB / BT.709 | PQ / HLG / Linear / Gamma / BT.1886 | BT.2020 / Display P3 primaries |
| --- | --- | --- | --- |
| `DisplayModel::by_name(...)` registry | ✓ | ✓ | ✓ |
| `DisplayModel::new(...)` constructor | ✓ | ✓ | ✓ |
| `Eotf::forward(...)` scalar dispatcher | ✓ | ✓ | n/a |
| `display_byte_to_dkl_scalar(...)` host-scalar | ✓ | ✓ | ✓ |
| `display_linear_rgb_to_dkl_scalar(...)` host-scalar | ✓ | n/a | ✓ |
| `host_scalar::predict_jod_still_3ch(...)` | ✓ | ✓ | ✓ |
| GPU fast path (`Cvvdp::score`, etc.) | ✓ | ✓ | ✓ |
| GPU linear-planes (`score_from_linear_planes`, etc.) | ✓ | ✓ | ✓ |

GPU dispatch was wired in commit `f8bf2729` (2026-05-25):

* `srgb_to_dkl_kernel` takes `eotf_tag` (u32), `gamma_exp` (f32),
  `hlg_gamma` (f32 — precomputed system gamma), and the 9 RGB→DKL
  matrix entries as runtime scalars. tag=0 (sRGB) keeps the LUT fast
  path; all other tags route through `apply_eotf_branch` and (for HLG)
  a per-pixel `hlg_ootf` step that resolves the Y_s OOTF gamma at the
  RGB-triple level.
* `linear_rgb_planes_to_dkl_kernel` takes the same 9 matrix scalars
  so non-BT.709 linear-RGB inputs route into the right opponent
  space.
* sRGB + BT.709 stays bit-identical to the pre-dispatch
  hardcoded-matrix kernel (pinned by
  `tests/color_kernel::srgb_to_dkl_kernel_matches_host_scalar`).

EOTF coverage on GPU: Srgb, Pq, Hlg, Linear, Bt1886, Gamma(g) — all 6
variants of `params::Eotf`.

Primaries coverage on GPU: Bt709 (default), Bt2020, DisplayP3, DciP3
(currently aliased to DisplayP3 — see `Primaries` docs).

Parity vs pycvvdp v0.5.4 — measured on the real GPU path
(cubecl-cuda / RTX 5070 / CUDA 13.2, native build):

* `standard_4k` over 13 mixed-content pairs: mean abs_diff = 0.037 JOD,
  median = 0.002 JOD, max = 0.391 JOD (single outlier on heavy noise).
* `iphone_14_pro` over the same 13 pairs: mean = 0.030 JOD,
  median = 0.001 JOD, max = 0.331 JOD.

Both displays meet the mean<0.10 gate; the max outlier appears on both
displays at the same pair (`photo_dark_noise_heavy`), indicating a
content-specific divergence rather than a display-dispatch defect.
GPU and host_scalar numbers agree to ~0.0002 JOD, consistent with the
GPU↔scalar pin in `tests/color_kernel_display_dispatch.rs`.

Full breakdown: `benchmarks/cvvdp_iphone14_parity_2026-05-25.tsv`.

Reproducers (both target the same TSV layout):
* GPU:
  `cargo run -p cvvdp-gpu --release --example parity_iphone_eval_gpu --features cuda,cubecl-types --no-default-features`
  (writes `/tmp/cvvdp-display-eval/parity_v2_gpu.tsv`)
* Host scalar fallback (no CUDA required):
  `cargo run -p cvvdp-gpu --release --example parity_iphone_eval`
  (writes `parity_v2_scalar.tsv`)

## Refreshing the vendored JSON

If pycvvdp ships new presets:

```sh
curl -sL -o crates/cvvdp-gpu/data/display_models.json \
  https://raw.githubusercontent.com/gfxdisp/ColorVideoVDP/main/pycvvdp/vvdp_data/display_models.json
curl -sL -o crates/cvvdp-gpu/data/color_spaces.json \
  https://raw.githubusercontent.com/gfxdisp/ColorVideoVDP/main/pycvvdp/vvdp_data/color_spaces.json
cargo test -p cvvdp-gpu --features cubecl-types --lib presets::tests
cargo test -p cvvdp-gpu --features cubecl-types --test it eotf_primaries_invariants
```

If a new preset adds a `colorspace` value we haven't mapped (or a
new EOTF / primaries variant), the `resolve_colorspace` function
in `src/presets.rs` will fall back to `(Eotf::Srgb, Primaries::Bt709)`
for unknown values. The registry tests will still pass, but the
new preset won't score correctly until the variant is added to
`Eotf` / `Primaries`.
