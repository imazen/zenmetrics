# AVIF autotune bake — consumer contract (v1, 2026-09-04)

The consumer is **`zenavif`'s `auto-tune` feature** (`src/auto_tune.rs`, with
`fast_heads.rs` / `q0_head.rs` / `palette_gate.rs` as siblings). This file is
what the wiring lane needs to load these bakes; the science is in
`zenmetrics/benchmarks/avif_autotune_v1_2026-09-04.md`.

---

## 0. READ THIS FIRST — what is and is not shippable

| head | verdict | evidence |
|---|---|---|
| **bytes / knob pick** | **usable with a stated 13–15 % mean regret**; excellent on screen content (screenshot mean 0.7 %, 92 % exactly-optimal), poor on photo/scan (17–19 %) | §4 |
| **backend pick** | **FAILS — do not ship as a decision.** **54.0 %** agreement with the measured per-image winner against a **67.7 % always-`zenav1-svt` baseline** (161 budget-corpus decisions; the 132 native rows are NOT-COMPARABLE, because the reference itself is budget-only). It never recovers a `zenrav1e` win (0/25 and 3/27 on the two eval images where zenrav1e wins). | §4 |
| **time / `encode_ms`** | **not usable as a budget gate.** Trainer's own safety gate fires `TIME_HEAD_R2` (median per-cell R² **0.316** core / **−1.981** full, bar 0.60); held-out p50 relative error 71 % / 83 %. Use the emitted **LUT** instead, which is what the current runtime already does. | §4 |
| **quality → q** | shipped as the LUT, same schema as today's `rav1e_quality_lut_v0_1_1.json` | §3 |

**Both bakes carry `safety_report.passed = false`** and were baked with
`--allow-unsafe`. That is a deliberate, recorded state, not an oversight: the
corpus is 32 references and the trainer's thresholds are calibrated for corpora
two orders of magnitude larger. Treat v1 as an instrument, not a default-on
product path.

---

## 1. Artifacts

Root: `/mnt/v/zen/avif-autotune-2026-09-04/`

| file | sha256 | bytes |
|---|---|--:|
| `models/zenavif_autotune_v1_full.bin` | `ef91a4b971189a2a9640abcd116af0a0c645e51ab5030476e6020d4b804fd478` | 270,572 |
| `models/zenavif_autotune_v1_core.bin` | `c2bfb016cbf2c02cb948e6e57bf608599716eeb7276f83f814e18ce50de7f9f4` | 161,756 |
| `features_avif_autotune.tsv` | `fb5378250ec89b03f206694e6dcc6a890da8bc92c79437fe65976b86df33c68f` | 83,521 |
| `pareto_avif_autotune.parquet` | `aa7f055a8ff63bf6a7d8de244872502bd3eb352cba87054a4beca25cd7cb57e4` | 3,662,245 |
| `pareto_avif_autotune_core.parquet` | `49bd24701bcfabe00710ab6cd31a9faca66769faa73c552c158041d912c3fed4` | 1,990,175 |

Per-bake companions in `validation/`: `<name>_cellmap.json`,
`<name>_encode_ms_lut.json`, `<name>_quality_lut.json`, `<name>_validation.json`.
Every file's sha256 is in `_MANIFEST.json`.

**`full` vs `core`.** Same view, same features, same split — different cell set.
`core` admits only cells measured on BOTH the 1024² budget corpus and the
native corpus, so every pickable cell has cross-size evidence. That rule
**excludes `zenrav1e` entirely**, because the `zenrav1e` arm (`brsdr`) only ever
ran on the budget corpus — so `core` is svt-only and has no backend decision at
all (it reports `backend_pick.status = NOT-APPLICABLE`, never a zero). `full`
carries both backends and pays for it in regret and in the failed backend head.

---

## 2. Feature contract — `zenanalyze-api` Request terms

**Selector:** `Select::Features(&[NamedFeature])`, one per column of
`zentrain.feature_columns`, each qualified with **this build's** feature code
version — i.e. exactly what `auto_tune::reuse_from_offer` already does:

```rust
let name = c.strip_prefix("feat_").unwrap_or(c);
let full = zenanalyze::versioning::feature_version_hash_by_name(name)?;
NamedFeature::qualified_for(name, NamedFeature::fold_hash(full))
```

- **Column form is BARE `feat_<name>`, 117 of them** — NOT the `name@hex8`
  qualified form some newer feature tables use. This matters: the runtime's
  `strip_prefix("feat_")` returns the whole string for a qualified column, the
  version lookup then misses, and the own-pass fills **0.0**. A qualified-name
  bake would score silently-wrong, not fail. These bakes are bare by
  construction.
- **The 117 names are exactly zenanalyze's current
  `benchmarks/feature_qualified_names.tsv` set** — verified: 117 of 117 present,
  0 extra, 0 missing. So `FeatureSet::SUPPORTED` covers every column and the
  own-pass fills none with the culled-feature 0.0 fallback.
- **Extractor era:** `zenanalyze/target/release/examples/extract_features_imazen26_crops`,
  which calls `analyze_features_rgb8(rgb, w, h, &AnalysisQuery::new(FeatureSet::SUPPORTED))`
  — the SAME entry point the runtime's own-pass uses, so a runtime pass
  reproduces these values.
- **Stamps in the bake:** `zentrain.analyzer_version = "0.2.0"`,
  `zentrain.feature_config_hash = 0`. `reuse_from_offer` compares both against
  the Offer's `Provenance`; a mismatch declines reuse and runs its own pass,
  which is the safe direction.
- **Gate for a pixel-shared cross-check:** 19 of the 32 references are
  byte-identical between the two corpora; their 117 feature values are
  byte-identical in both feature tables (2,223/2,223 cells). Any re-extraction
  that breaks that identity has drifted.

## 2b. Input-vector layout — **THE ONE BREAKING DIFFERENCE FROM TODAY'S RUNTIME**

`auto_tune.rs` step 4 hardcodes a **4-wide** size one-hot and builds `2n + 10`.
These bakes have a **2-wide** one-hot and `n_inputs = 242`, because
`train_hybrid._scope_size_classes` narrows the grid to the size classes the
corpus actually covers, and this corpus is `medium` + `large` only (no image
below 256²). The general rule is:

```
n_inputs = 2 * len(feature_columns) + 5 + len(SIZE_CLASSES) + 1
         = 2 * 117               + 5 + 2                  + 1  = 242
```

The current runtime would build 244 and hit `FeatureLenMismatch` — loud, not
silent, but still a failure. **New metadata key `zentrain.input_layout`**
(added to `bake_picker.py` this pass) makes the layout self-describing: it is
the aux-slot names in order, newline-separated, after the `feat_cols` block —

```
size_medium
size_large
log_pixels
log_pixels_sq
zq_norm
zq_norm_sq
zq_norm_x_log_pixels
zq_x_feat_variance
… (one zq_x_ per feature, same order)
icc_placeholder
```

so the runtime can read `size_*` from the bake instead of hardcoding four.
The trainer's own prescription for an out-of-grid size (`_scope_size_classes`
docstring) is **map to the nearest modelled class**; the validator implements
exactly that and it is the behaviour to copy. Note the coverage consequence:
**neither bake has any training data for `tiny` (<64²) or `small` (<256²)
images** — a runtime that maps a 200×200 image to `medium` is extrapolating,
and should probably decline.

Everything else in step 4 is unchanged: `log_px = ln(w*h)`,
`zq_norm = target/100`, the same five interaction terms in the same order, the
`zq_norm * raw_feats` block, and the trailing `icc_placeholder` 0.0.

---

## 3. Output head layout

`zentrain.hybrid_heads_layout` (and `hybrid_heads_manifest.output_layout` in the
JSON) gives the exact slices. For these bakes:

| bake | n_cells | n_outputs | `bytes_log` | `time_log` | `metric_log` |
|---|--:|--:|---|---|---|
| core | 48 | 144 | `[0, 48)` | `[48, 96)` | `[96, 144)` |
| full | 143 | 429 | `[0, 143)` | `[143, 286)` | `[286, 429)` |

- `bytes_log[c]` = predicted `ln(bytes)` for cell `c` at the requested target.
  The pick is `argmin` over the allowed mask — same as today.
- `time_log[c]` = predicted `ln(encode_ms)`. **Do not gate on this** (§0).
- `metric_log[c]` = predicted `ln(ssim2)` — new relative to the shipped picker;
  `--emit-metric-head` produced it.

**Cell index → knob tuple is NOT `speed = cell + 1` any more.** The shipped
`rav1e_picker_v0_1_1.bin` had one cell per speed; these have a
(backend × speed × knobs × bit-depth) cell axis. Decode via
`<name>_cellmap.json`, whose grammar is `<backend>_s<speed>_<knobs|default>_bd<8|10>`:

```json
{"index": 0, "label": "svt_s1_default_bd8", "backend": "zenav1-svt",
 "speed": 1, "knobs": [], "bit_depth": 8, "chroma": "4:2:0",
 "chroma_is_derived_from_backend": true}
```

`knobs` entries are the DOE's own codes (`qml1.2.10`, `shp7`, `mtx32`, `tl1.0`,
`tn3`, `vbst1.3.5`, `acb1`, `scm3`, …), which reach the encoder through
zenavif's plan path / `SvtParams`, **not** through `EncoderConfig` — the sweep
tool cannot set them today either (`avif_speed_instrument`'s §4 records the same
gap). Wiring a knob cell to a real encode therefore needs a zenavif change.

**`chroma` is DERIVED, never chosen.** Every `svt` cell is 4:2:0 and every
`rav` cell 4:4:4 in this corpus — measured over 1,114 `av1C` boxes with zero
exceptions, because `avif_config_from_knobs` pins `Yuv420` for svt and leaves
zenravif on `EncoderConfig`'s `Yuv444` default. **No chroma knob is wired for
AVIF at all.** So `backend` and `chroma` are perfectly collinear: the model
cannot separate them and neither can any consumer of it.

`zenpicker.knob_vetoes` carries the trainer's derived feature-thresholded cell
vetoes (1 on core, 4 on full) — apply them as an extra mask before the argmin,
same shape as today's `speed_range` / time masks.

---

## 4. LUTs

`<name>_encode_ms_lut.json` matches today's `EncodeMsLut` schema
(`median_ms_per_mpx` → per-cell → `{tiny, small, medium, large}`), keyed
`cell<N>` instead of `speed<N>` because the cell axis is no longer the speed
axis. **The values are modelled, not measured** (see `_note` in the file): the
view's `encode_ms` comes from the 2026-09-03 speed instrument's `α + β·MP`
fits — single-threaded wall time, q45-anchored, per-source fits on 5 of 32
sources, pooled fits flagged `linear_model_failed` on 20/20 arms with β
spreading up to 24.3×. Size classes with no corpus coverage are filled from
the nearest measured class.

`<name>_quality_lut.json` matches today's `QualityLut` schema (`cells`,
`target_zqs`, `median_q`, `-1` = unreachable). **`target_zqs` are `ssim2`
values, not zensim** — `ssim2` is the only corpus-wide scalar response the AVIF
DOE produced (zensim was emitted as a 720-wide feature vector with no scalar).
`QualityTarget::Zensim` in the current API is therefore the wrong name for what
these bakes take; the wiring lane should either add a `QualityTarget::Ssim2`
variant or document the reinterpretation. **Do not silently feed a zensim
target to an ssim2-calibrated LUT.**

---

## 5. Validation summary (held-out `eval8` leg, 6 origins)

Regret = picked bytes ÷ cheapest cell that reaches the target − 1.

| bake | n | mean | p50 | p90 | max | exactly optimal |
|---|--:|--:|--:|--:|--:|--:|
| core | 266 | **13.4 %** | 8.8 % | 31.1 % | 78.7 % | 21.1 % |
| full | 293 | **14.5 %** | 8.3 % | 41.7 % | 78.6 % | 21.8 % |

Per coarse content class (full / core mean regret): screenshot 5.9 % / **0.7 %**,
plot 4.0 % / 4.5 %, scan 11.0 % / 18.6 %, photo 25.5 % / 17.0 %,
ai-gen 26.6 % / 11.5 %. `core` is better overall and dramatically better on
screen content; `full` is better on scan and plot.

**The split is not the canonical one.** All 32 references are EVEN-origin by
construction (the corpus was k-means-selected under `--parity 0`), so the
canonical `{1,3,5}` validate and `{7,9}` test buckets are **structurally
empty** — there is no odd-origin AVIF encode anywhere. The registered even-only
sub-split was used instead (`DATA_SPLITS.md` L158, the `avifgen-2026-08-06`
precedent): `{0,2,4,6}` train (26 origins), `{8}` = `eval8` holdout
(6 origins: `1008` photo, `6018` + `6038` scan, `7058` plot, `8288` screenshot,
`9118` ai-gen). `eval8` is never trained on, but **it is not the canonical
holdout and must not be described as one.**

---

## 6. Reproduce

```sh
# 1. the view (no encodes; reads only harvested parquets)
python3 zenmetrics/scripts/jobsys/avif_autotune_view.py \
    --out /mnt/v/zen/avif-autotune-2026-09-04

# 2. train  (PYTHONPATH needs zentrain/examples + zenmetrics/scripts/picker)
cd zenanalyze/zentrain/tools
PYTHONPATH=../examples:../../../zenmetrics/scripts/picker \
uv run --with scikit-learn --with numpy --with pyarrow --with torch \
  python3 train_hybrid.py --codec-config zenavif_autotune_2026_09_04 \
    --objective size_optimal --emit-metric-head

# 3. bake  (safety gate fails by design — see §0)
uv run --with numpy python3 zenanalyze/tools/bake_picker.py \
    --model  .../models/zenavif_autotune_v1_full.json \
    --out    .../models/zenavif_autotune_v1_full.bin \
    --dtype f16 --allow-unsafe

# 4. validate + emit LUTs
python3 zenmetrics/scripts/jobsys/avif_autotune_validate.py \
    --model .../models/zenavif_autotune_v1_full.json \
    --pareto .../pareto_avif_autotune.parquet \
    --features .../features_avif_autotune.tsv \
    --out .../validation --name zenavif_autotune_v1_full
```

---

## 7. Coverage limits (all measured, none inferred)

1. **`aom-rs` is not a backend here.** No aom arm was ever declared in the DOE
   (Stage-A's A3 block), so the cell axis has two backends, not three.
2. **`zenrav1e` has no native-size leg.** Its whole arm (`brsdr`) ran on the
   1024² budget corpus. Every backend decision at native size is an
   extrapolation — which is the likeliest reason the backend head fails on the
   two 12 MP+ eval images.
3. **Chroma is unswept** (§3). Splitting it from backend needs the rank-1 gap:
   a zenrav1e 4:2:0 arm (~24 CPU-h + one change to `avif_config_from_knobs`).
4. **HDR is a separate regime and is NOT modelled.** `cells_hdr.parquet`
   (3,680 rows, 16 refs, 10-bit PQ) is carried in the view for completeness,
   but Track T2 swept only the speed/preset dial with zero DOE knobs, on a
   corpus where backend, chroma AND matrix all differ at once. It is an RD
   baseline; there is no HDR bake and one should not be trained on it.
5. **`ssim2` is sharpness-blind**, and the sharpness knobs (`shp*`) are exactly
   the family the DOE rejects. "Costs bits at matched ssim2" and "not worth
   enabling" are different claims; only the first is measured.
6. **No `tiny`/`small` size coverage** (§2b).
7. **Dead knobs are excluded by measurement, not by doc.** `tn0` and `scm3` are
   byte-identical to their control on 288/288 cells at speeds 4 and 6 (both
   corpora), 29 pair-arms are byte-identical aliases of a single-knob arm, and
   svt presets 8/9/10 are byte-identical to 7 — 10,614 rows dropped. `scm3` at
   speed 7 is NOT inert and is kept. The census is
   `sidecar_inert_arm_census.tsv` / `sidecar_alias_arms.tsv` /
   `sidecar_speed_alias.tsv`.
