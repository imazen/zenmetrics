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
3. **Plan omission.** zenpng's `SweepAxes` has **no palette/quantize axis** —
   the single biggest PNG RD lever was never coded into the plan.

### Audit map (2026-06-27)

| codec | failure | picker can NEVER choose | severity |
|---|---|---|---|
| jpeg | (1) tail-shed | XYB color, 422, trellis ladder | CRITICAL |
| avif | (2) rd_core-only | RGB (25–40% RD), trellis, VAQ, speed8, 27 probes | CRITICAL |
| webp | (1) tail-shed (budget=300) | sharp_yuv, filter_strength, sns_strength, filter_sharpness | CRITICAL |
| png | (3) plan omission | palette/quantize (no png data swept at all) | CRITICAL |
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
- **zenpng**: a palette/quantize config (axis must be ADDED first — see below);
  effort tiers ≤ the sub-30s cut.
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
   - **zenpng** ⏳ — the palette/quantize axis must be ADDED (plan omission, not a
     collapse); it is metric-class (changes pixels), so png joins the metric-scored
     sweep. Bigger than a collapse-line fix; not yet done.
   - **zenjxl** — lossy is clean; modular needs a re-sweep with `modes_full`/
     `scalar_dense` (not a source change — it was swept `rd_core`).
2. **Plan-time (zenmetrics `plan.rs`)** — after building, assert no mandatory
   value was dropped; surface in the manifest; ERROR (not silent) on a mandatory drop. *(queued)*
3. **Picker-train-time (zenmetrics `scripts/picker/check_mandatory_coverage.py`)**
   — the universal net: before training, assert the pareto covers every mandatory
   token for the codec; FAIL LOUD with a re-sweep instruction otherwise. Catches
   all three root causes regardless of where they originate. **(landed first — see
   that script.)**

## Re-sweep needed (all crippled pickers)

jpeg (+XYB/422), avif (+RGB/trellis/probes — the failed avifrd run produced
nothing, re-run on the granular system), webp (+sharp_yuv/filters/sns), png
(+palette, never swept), jxl-modular (full effort ladder + predictors). Run on
the granular ≤5-min work-stealing job system (PR #31). jxl-lossy is clean.
