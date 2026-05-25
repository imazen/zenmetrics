# UPSTREAM_DIVERGENCES — cvvdp-cpu v0.1.0 vs gfxdisp/ColorVideoVDP

Items where cvvdp-cpu v0.1.0 intentionally differs from
[gfxdisp/ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP) main.
Each row explains *what diverges*, *why*, and *the path to close it
when / if needed*.

This document is the persistent counterpart to
`UPSTREAM_PARITY_AUDIT.md`: parity rows from the audit that are
closed by code land here as "RESOLVED" rows; rows that stay open by
design land as "DIVERGES" rows.

---

## 1. DIVERGES — Temporal channels (4th channel, "Y_t")

**Upstream**: ColorVideoVDP scores video by adding a transient
luminance channel (`Y_t`). Several parameters carry a 4th slot
(`mask_q`, `xcm_weights`, `baseband_weight`, `BETA_TCH`) for this
channel. The pooling stage adds a `beta_t` term across frames.

**cvvdp-cpu**: 3 channels (A, RG, VY) only. The 4th slot in
`mask_q` / `xcm_weights` / `baseband_weight` is dropped. No
temporal pooling.

**Why**: cvvdp-cpu is targeted at the still-image / web-encoder /
JPEG XL butteraugli-loop use-case. Temporal scoring requires a
multi-frame buffer + the transient pyramid + `beta_t`. The cvvdp-gpu
README explicitly scopes this crate as still-image-only.

**Path to close**: A Phase 2 effort would (a) build a transient
pyramid alongside the per-band weber pyramids, (b) add 4th slots
to `MASK_Q` / `XCM_3X3` (becomes `XCM_4X4`) / `BASEBAND_W`, (c)
add `beta_t` cross-frame pooling. Multi-week. Untracked.

## 2. DIVERGES — Foveation / saliency (`cvvdp_ml_saliency`)

**Upstream**: ships a `cvvdp_ml_saliency` ONNX/torch model that
per-pixel-weights the metric by visual attention. Optional in the
upstream Python API.

**cvvdp-cpu**: not ported. The CSF is applied uniformly across the
image at the central PPD.

**Why**: same reason as temporal channels — out-of-scope for the
still-image-encoder use-case. Saliency models also create a
dependency on a torch / ONNX inference runtime that doesn't fit
the `forbid(unsafe_code) + no_std + alloc` constraints of this
crate.

**Path to close**: Phase 2+ — pluggable saliency-map weighting
where callers supply a per-pixel `f32` map.

## 3. DIVERGES — Display `exposure` field

**Upstream**: `vvdp_display_photo_eotf` has an `exposure` parameter
(`display_model.py:312`) that multiplies the linear-light values
post-EOTF before display scaling. Used to model under/over-exposed
content. None of the 26 named display presets in
`display_models.json` sets `exposure != 1`.

**cvvdp-cpu**: `DisplayModel` has no `exposure` field. Effectively
`exposure = 1.0` everywhere.

**Why**: zero of the canonical upstream presets exercise it; adding
a field is API-breaking and ripples through every constructor /
preset. The audit (`UPSTREAM_PARITY_AUDIT.md` row D9) flags this as
"minor — reserved for future opt-in."

**Path to close**: small additive change — add an `exposure: f32`
field (default 1.0 via `Default`), thread through
`display_byte_to_dkl_scalar` and `display_linear_rgb_to_dkl_scalar`
on the cvvdp-gpu side. Then ship a `DisplayModel::with_exposure`
builder. Estimated 2-3 hours; deferred because the constant lookup
table on `DisplayModel` (chunk 2's 23 presets) all use `exposure =
1.0` and no parity test would shift.

## 4. DIVERGES — Alternative CSF LUTs

**Upstream**: ships 6 CSF LUTs under `pycvvdp/vvdp_data/csf_lut_*.json`:
`weber_fixed_size` (our default), `weber`, `weber_supra`, `weber_old`,
`log`, `dkl_cone`, `none`. Selectable via the `csf` field in
`cvvdp_parameters.json`.

**cvvdp-cpu**: only `weber_fixed_size` is vendored (via cvvdp-gpu's
`kernels/csf_lut/v0_5_4.rs`).

**Why**: `weber_fixed_size` is what pycvvdp v0.5.4 selects in its
shipped `cvvdp_parameters.json`. The other LUTs are for ablation
research; no production caller selects them.

**Path to close**: vendor the additional LUTs as separate `const`s,
add a `CsfLut` enum field on `CvvdpParams`. Significant code-size
increase (each LUT is ~6 KB). Untracked.

## 5. DIVERGES — Runtime CSF / masking parameter override

**Upstream**: `cvvdp_parameters.json` is loaded at runtime; researchers
can swap in a custom JSON for ablation.

**cvvdp-cpu**: the v0.5.4 parameters are inlined as `const`s in
`cvvdp_gpu::kernels::pool::{BETA_SPATIAL, BETA_BAND, …}`,
`cvvdp_gpu::kernels::masking::{MASK_P, MASK_Q, XCM_3X3, …}`. The
`CvvdpParams::{csf, masking, pooling, jod}` sub-bundles are
**scaffolding** — declared in the public API for forward-compat
but not consumed by the hot path.

**Why**: every parity test pins the v0.5.4 constants. Letting
runtime overrides shift them would break the goldens contract for
zero production benefit (no caller has asked for this).

**Path to close**: replace the kernel-local `const`s with reads
from `CvvdpParams`. Requires a re-bake of every parity test against
synthetic overrides + a "default == const" pin. ~1 week of work.
Untracked.

## 6. DIVERGES — PU21 perceptual uniform encoding

**Upstream**: ships PU21 / PU encoding (`pycvvdp/utils.py PU`) for
side-by-side display-encoded-frame comparison.

**cvvdp-cpu**: not ported. The `is_input_display_encoded` path that
gates PU encode in upstream doesn't exist here.

**Why**: PU21 is upstream's choice for the "display_encoded_*"
target colorspaces in `source_2_target_colorspace`. cvvdp-cpu has
exactly one target colorspace — DKLd65 — and routes through the
`linear_2_target_colorspace` path. PU21 isn't reached.

**Path to close**: would require adding a `TargetColorspace` enum +
the `PU` class. ~3-day port. Untracked.

## 7. DIVERGES — Color spaces beyond {sRGB, BT.2020, P3}

**Upstream**: `color_spaces.json` defines 12+ color spaces including
Adobe RGB, Apple RGB, Wide Gamut RGB, BT.601, P3D60 (theatrical),
NTSC, etc.

**cvvdp-cpu**: ships 3 (BT.709 / BT.2020 / DisplayP3 (= DciP3 alias)).

**Why**: the 3 shipped cover every upstream display preset. The
others are reserved for future opt-in if a caller asks. Note
DisplayP3 + DciP3 are currently aliased to the same matrix — the
DCI theatrical white point (DCI-P3, 6300 K, no D65 chromatic
adaptation) would need its own matrix when a 48-nit cinema preset
appears.

**Path to close**: vendor additional matrices via Python @ f64 then
inline. Each new variant is ~10 LOC. Untracked.

---

## RESOLVED in v0.1.0

| # | Audit row | What v0.1.0 ships |
|---|---|---|
| R1 | D7b | `srgb_to_dkl_planar` honors `display.eotf` via dispatch path |
| R2 | D8a | `srgb_to_dkl_planar` honors `display.primaries` via dispatch path |
| R3 | P2-P21 | 23 named DisplayModel constants (HDR PQ, HDR HLG, FHD, phones, iPads, MacBook, LG OLED, EIZO, 65-inch panels) |
| R4 | G3 | `DisplayGeometry::from_inches` constructor (inch-denominated) |
| R5 | G5 | `DisplayGeometry::from_meters_diagonal` constructor |
| R6 | G6 | `DisplayGeometry::from_fov_diagonal` constructor (VR HMD path) |
| R7 | G14 | `DisplayGeometry::display_width_m / display_height_m / display_width_deg / display_height_deg` getters |
| R8 | (test) | 18 new tests pinning every preset's fields + every EOTF / Primaries round-trip |

---

## Out-of-scope (untracked)

- `iphone_14_pro_vert` — present geometry preset only; the photometric
  side is identical to `IPHONE_14_PRO`. Callers wanting the vertical
  config use `DisplayModel::IPHONE_14_PRO` + `DisplayGeometry::IPHONE_14_PRO_VERT`.
- `iphone_14_pro_hdr_vert` — same: pair `DisplayModel::IPHONE_14_PRO_HDR`
  + a custom vertical geometry.
