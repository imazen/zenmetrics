# Mandatory sweep axes — the "never silently drop a first-class mode" guardrail

**Added 2026-06-27** after the zenjpeg-XYB disaster: a budgeted `modes_full`
sweep **silently dropped the XYB color path**, and the jpeg picker trained on
that data could never choose XYB. An audit found the same class of failure in
**every** codec (below). This doc is the contract + enforcement that stops it
recurring.

## The disaster (root causes — three flavors, all silent)

The sweep planner produces a `dropped_axes` manifest ("no-silent-caps report")
— but **nothing downstream ever checked it**, so a picker would happily train on
data missing a first-class mode. The three ways a mode goes missing:

1. **Budget tail-shed of a late-appended mode.** `modes_full()` appends axis
   values in priority order; the budget "ladder" truncates from the tail
   (`collapse_one_axis`, reducing each axis toward its `rd_core` base). A
   high-value mode appended late (jpeg **XYB**, webp **sharp_yuv / filter /
   sns**, the 2026-06-12 dense-sweep additions) is shed *first*.
2. **`rd_core` swept where `modes_full` was needed.** avif's shipped picker data
   and jxl-modular's data are `rd_core`-only — the high-value axes (avif **RGB**,
   trellis, VAQ, 27 probes; jxl-modular efforts e1–e10, predictors) were never
   in the cross at all.
3. **Plan omission (RESOLVED 2026-06-27).** zenpng's `SweepAxes` had **no
   palette/quantize axis** — the single biggest PNG RD lever was never coded
   into the plan. Now landed: `modes_full`/`scalar_dense` carry 8 mandatory
   quantize cells ({imagequant, zenquant} × {256,128,64,32}) as a UNION with
   the lossless compression cells. See "png quantize axis" below.

### Audit map (2026-06-27)

| codec | failure | picker can NEVER choose | severity |
|---|---|---|---|
| jpeg | (1) tail-shed | XYB color, 422, trellis ladder | CRITICAL |
| avif | (2) rd_core-only | RGB (25–40% RD), trellis, VAQ, speed8, 27 probes | CRITICAL |
| webp | (1) tail-shed (budget=300) | sharp_yuv, filter_strength, sns_strength, filter_sharpness | CRITICAL |
| png | (3) plan omission — RESOLVED 2026-06-27 | (was: palette/quantize never swept) — now both backends × {256,128,64,32} | OK |
| jxl-lossy | — clean (full e1–e9, probes present) | — | OK |
| jxl-modular | (2) rd_core-only (e5/7/9), deferred | e1–4/6/8/10, predictors, group-size, internal probes | HIGH |

## The contract: mandatory axes

Per the user (2026-06-27): *"some axis should be defined for mandatory sweeping,
like color mode, subsampling, and sub-30 second effort modes."*

Each codec declares a **mandatory set** of axis values. Rules:

- **Color mode** — every distinct color path is mandatory (jpeg YCbCr+XYB;
  avif YCbCr+RGB; webp lossy+lossless; png truecolor+palette; jxl-lossy XYB is
  native/always-on).
- **Subsampling / chroma mode** — every first-class subsampling is mandatory
  (420, 444, and 422 where the codec ships it; webp sharp_yuv is subsampling-class).
- **Sub-30-second effort/speed tiers** — every effort/speed tier whose *reference
  encode time is < 30 s* is mandatory. Only pathologically-slow tiers (png
  Crush/Maniac/Brag/Minutes; jxl e10 under `butteraugli-loop`) are optional.

Mandatory values are **never tail-shed by the budget**. If the mandatory-only
cross still exceeds budget, the plan reports `over_budget = true` (raise budget /
split the sweep) — it must **never** silently drop a mandatory value.

### Per-codec mandatory coverage (the enforcement spec)

Expressed as config_name tokens that MUST each appear ≥1× in the swept pareto:

- **zenjpeg**: color/sub ∈ must include `_420`, `_444`, `_422` (HalfHorizontal),
  and an XYB token (`xybBq` and/or `xybFull`).
- **zenavif**: an `-rgb` config (RGB), plus `420` and `444`; speeds covering the
  sub-30s tiers.
- **zenwebp**: a `vp8l` (lossless) AND `vp8` (lossy) config; a `-syuv`
  (sharp_yuv) config.
- **zenpng**: BOTH palette backends — an imagequant config (`-iq<N>`) AND a
  zenquant config (`-zq<N>`); effort tiers ≤ the sub-30s cut. (Axis landed
  2026-06-27 — see below.)
- **zenjxl lossy**: full `e1..e9`. **modular**: `e1..e10` + predictor coverage.

## Enforcement points (belt + suspenders)

1. **Source (each codec `SweepAxes`)** — mark color / subsampling / sub-30s-effort
   axes non-collapsible; the budget ladder skips them; `modes_full` MUST contain
   every mandatory value.
   - **zenjpeg** ✅ LANDED (main `7afedf4c`) — `color_modes` removed from
     `collapse_one_axis` (XYB + 4:2:2 survive any budget) + regression test.
   - **zenavif** ✅ LANDED (main `68cd644`) — `color_models` removed from the
     ladder (RGB survives) + test. (subsampling was already never-collapsed.)
   - **zenwebp** ✅ LANDED (main `d5254f6`) — `lossy.sharp_yuv` + `lossy.methods`
     removed from the ladder + test.
   - **zenpng** ✅ LANDED (2026-06-27) — the palette/quantize axis was ADDED to
     `zenpng::sweep` (`SweepVariant.quantize: Option<QuantizeSpec>`,
     `QuantBackend::{Imagequant,Zenquant}`). `modes_full`/`scalar_dense` emit 8
     quantize cells ({256,128,64,32} × both backends) as a UNION with the
     lossless compression cells; the cells are metric-class (changes pixels), so
     png joins the metric-scored sweep. zenmetrics encodes them via
     `SweepVariant::encode_png` (plan path) + a `quantize` knob (`encode_png`
     knob path). Coverage gate updated to require both backends.
   - **zenjxl** — lossy is clean; modular needs a re-sweep with `modes_full`/
     `scalar_dense` (not a source change — it was swept `rd_core`).
2. **Plan-time (zenmetrics `plan.rs`)** — after building, assert no mandatory
   value was dropped; surface in the manifest; ERROR (not silent) on a mandatory drop. *(queued)*
3. **Picker-train-time (zenmetrics `scripts/picker/check_mandatory_coverage.py`)**
   — the universal net: before training, assert the pareto covers every mandatory
   token for the codec; FAIL LOUD with a re-sweep instruction otherwise. Catches
   all three root causes regardless of where they originate. **(landed first — see
   that script.)**

## Re-sweep plan — REUSE existing work, sweep only the delta (user: 2026-06-27)

Don't re-sweep anything already validly swept. Existing data is **omni-TSV**
form (`knob_tuple_json` carries `{cell,fp,plan}`), not a job-ledger — so reuse =
**delta-sweep + merge**, per codec:

1. Emit the CORRECTED `modes_full` cells (now carrying the mandatory modes):
   `zenmetrics sweep --codec X --plan modes_full --plan-budget B' --dry-run --emit-cells corrected.jsonl`.
2. Delta = corrected cells whose `cell` id ∉ the existing omni's swept cells.
3. Sweep ONLY the delta (job system declare / chunk sweep of the delta list) on
   the granular ≤5-min work-stealing system (PR #31).
4. Merge delta-omni ⊎ existing-omni → complete omni → omni_to_pareto → retrain
   (now passes `check_mandatory_coverage.py`).

| codec | reuse (already swept) | delta to sweep |
|---|---|---|
| jpeg | 48 cfgs (8fam×3tr×{420,444}) | XYB (jpegli-fam×tr) + 4:2:2 (~30–48 new) |
| avif | 24 rd_core cfgs | modes_full adds RGB + trellis + vaq + 27 probes |
| webp | 12 cfgs | sharp_yuv + fast method (+ filter/sns probes) |
| jxl-modular | 9 cfgs (e5/7/9) | e1–4/6/8/10 + predictors |
| png | 0 (new) | compression cells + **8 quantize cells** |

**png quantize axis (LANDED 2026-06-27; user spec 2026-06-27 superseded the
earlier "6 cells / {256,64,16}" sketch):** `{imagequant, zenquant} × {256, 128,
64, 32}` max_colors = **8 metric-class cells**, as a UNION with the lossless
compression cells (NOT a cross — the compression cells stay truecolor; each
quantize spec is one cell at the default Balanced compression). `modes_full` is
therefore 17 cells (9 truecolor + 8 palette). Cross-repo feature landed:
- zenpng `SweepVariant` gained `quantize: Option<QuantizeSpec>` +
  `SweepVariant::encode_png` (truecolor via `encode_rgb8`; palette via
  `ZenquantQuantizer`/`ImagequantQuantizer` + `encode_indexed`, NOT
  `EncodeConfig`). Cell ids: `png-<preset>-iq<N>` / `-zq<N>`.
- zenmetrics applies it in the plan path (`PlannedConfig::Zenpng` →
  `encode_png`) and the knob path (`encode_png`'s `quantize` knob; `PNG_KNOBS`
  gained `"quantize"`).
- `check_mandatory_coverage.py` requires BOTH `-iq\d` AND `-zq\d` config_names.

**Before launching paid boxes:** bump the fleet image's codec pins to the fixed
commits (zenjpeg 7afedf4c / zenavif 68cd644 / zenwebp d5254f6 + png once landed)
and post a box/cost estimate. jxl-lossy is clean — no re-sweep.
