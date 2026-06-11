# Plan-driven sweeps: the cross-codec PlanSpec contract

Status 2026-06-11: all five zen codecs are on the plan-cell bridge and
smoke-validated end-to-end (`benchmarks/planspec_smoke_*_2026-06-11.*`).
This is the contract reference for `--plan` sweeps, the per-cell
identity, and the per-codec axis inventory with scalar axes tagged for
`zenpicker-train --scalar-axes`.

The codec-neutral playbook (16 patterns: knobs-live-on-the-variant,
resolved-state fingerprints, self-describing ids, budget ladder with no
silent caps, …) is `zenjpeg/docs/VARIANT_GENERATION.md`; each codec's
adoption doc / module docs carry its provenance tables. This document
covers the zenmetrics side: how plans execute and what the axes are.

## 1. The shape of the bridge

Each codec owns its sweep space behind an `__expert`-class unstable
feature (promotion to stable surfaces is a separate, user-approved step
— e.g. imazen/zenjxl#8 for `resolve_plan`):

| codec | planner module | plan names | modes | budget ladder |
|---|---|---|---|---|
| zenjpeg | `zenjpeg::encode::sweep` | `rd_core`, `modes_full` | lossy | yes |
| zenavif | `zenavif::sweep` | `rd_core`, `modes_full`, `modes_full_alpha` | lossy (+alpha probes) | yes |
| zenjxl | `zenjxl::sweep` | `rd_core`, `modes_full` | lossy (VarDCT) + lossless (modular) | yes |
| zenwebp | `zenwebp::sweep` | `rd_core`, `modes_full` | lossy (VP8) + lossless (VP8L) | yes |
| zenpng | `zenpng::sweep` | `rd_core`, `modes_full` | all-lossless | no (9 cells max; over-budget reported, never sampled) |

Common planner shape: `SweepAxes` (most-important-first axis vectors) ×
`QualityGrid` (`Step5` 21-point floor / `TrainingDense` 31-point /
`Explicit`; zenjxl also `ExplicitDistance`) → `SweepBuilder::plan()` →
`SweepPlan { cells, dropped, invalid_skipped, duplicates_merged,
q_coarsenings, over_budget }`. Cells are emitted main-effects-first
(deviations 0, then 1, then combos); every reduction is reported, never
silent.

`zen-metrics-cli`'s bridge (`crates/zen-metrics-cli/src/sweep/plan.rs`):

- `build_plan(codec, name, budget, q_grid) -> BuiltPlan` — the single
  dispatch point. Writes the audit manifest to `<output>.plan.json`.
- `PlannedCell { q, knob_json, config: PlannedConfig }` — one encode
  cell; `PlannedConfig` is the per-codec fully-built config enum.
- `resolve_verified(codec, cell_id, q, fp_hex) -> PlannedConfig` — the
  executor half: reconstructs the config from the **self-describing
  cell id** (`config_from_cell_id` / `variant_from_cell_id`), recomputes
  the resolved-state fingerprint, and fails loudly on mismatch (the
  id-grammar-drift tripwire; never a silently wrong encode).

## 2. The durable per-cell identity

One identity rides every path, in the TSV/parquet `knob_tuple_json`
column and in `JobKind::Encode.knobs`:

```json
{"cell":"<stratum-id>","fp":"<16-hex fingerprint>","plan":"<plan-name>"}
```

- `cell` is the stratum id **without** the `_q<q>` token (q has its own
  column). Grammar per codec (additive-only, renderer/parser lockstep
  enforced by each codec's roundtrip test):
  - zenjpeg: `jp3_t0_small_420`-style (see `config_from_cell_id` docs)
  - zenavif: `s<speed>[-flags…]` (`config_from_cell_id(base_id, q)`)
  - zenjxl: `vd-e<eff>_<strategy>_<label>[-flags…]` (lossy) /
    `mod-e<eff>_<label>[-flags…]` (lossless)
  - zenwebp: `vp8-m<m>_<label>[-flags…]` (lossy) / `vp8l-m<m>[-ql<q>]`
  - zenpng: `png-<preset>` / `png-e<effort>`
- `fp` is FNV-64 over **resolved** state (pattern 4: resolution is the
  identity, not the spelling — `png-e13` ≡ `png-balanced`, planner-level
  aliases are merged and reported in the manifest). For lossy cells the
  fingerprint covers the resolved quality/distance, so the same stratum
  at different q has different fp.
- **q sentinel:** lossless cells carry `q = 0` (i64 in declare items /
  input parquets). Their ids have no quality token and the parsers
  ignore q for them; zenpng rides the sentinel on every cell. A lossy
  plan identity presented with the wrong q dies on fingerprint mismatch.

## 3. Three execution paths, one identity

1. **Local / chunk-mode CLI** — `zen-metrics sweep --codec C --plan
   NAME [--plan-budget N] --q-grid … --sources DIR --output OUT.tsv`.
   Mutually exclusive with `--knob-grid`. Builds the plan once per
   sweep, writes `OUT.plan.json`, walks cells × images via rayon.
   Requires `--features sweep` + the codec feature (`sweep` force-pulls
   jpeg/webp/avif; add `jxl`/`png` explicitly).
2. **Job system** — `… --plan NAME --dry-run --emit-cells cells.jsonl`
   emits one declare item per (source × cell): `{image_path, codec,
   q:i64, knob_tuple_json, source_sha}` (q must be integer-valued).
   `zen_jobctl declare_encodes` → content-addressed `DesiredJob`s →
   `zen-metrics jobexec` executes each via `resolve_verified` (encode
   jobs emit bytes; metric jobs re-encode + score). See
   `docs/RUNNING_JOBS.md` §4b.
3. **Vast.ai chunk fleet** — plan-mode input parquets carry the
   identity JSON per row (v26 schema `image_path/codec/q:int64/
   knob_tuple_json`, NO schema change). Generate with:
   ```bash
   zen-metrics sweep --codec C --plan NAME [--plan-budget N] \
     --sources DIR --q-grid … --dry-run --emit-cells cells.jsonl \
     --output /tmp/plan.tsv
   python3 scripts/sweep/generate_sweep_input.py --codec C \
     --run-id RUN --cells-jsonl cells.jsonl \
     --source-dir-r2 s3://…/sources --out-dir OUT
   ```
   The worker groups rows by `(codec, knob_tuple_json)` as always; the
   sweep runner's tuple path detects the `{"cell","fp","plan"}` shape
   and routes through `resolve_verified` (byte-identical to the Planned
   path — enforced by `sweep::run::tests::
   plan_identity_tuple_matches_planned_cell_bytes`). `InlineGroupSpec.
   plan` also accepts a plan NAME for whole-plan-per-group runs, but
   per-cell identity rows are the fleet-native form (retry unit stays
   one chunk; cells slice across chunks).

## 4. Per-codec axis inventory — scalar axes tagged

**SCALAR** = continuous/quantized-numeric knob suitable for
`zenpicker-train --scalar-axes` regression heads (vs categorical /
boolean / ordinal-preset axes). Bounds + curated steps + provenance live
in each codec's sweep module docs; this is the cross-codec index.

### zenjpeg (`zenjpeg/zenjpeg/src/encode/sweep.rs` module docs table)

| axis | kind | bound / steps (modes_full) |
|---|---|---|
| quality | SCALAR (rate axis) | 1–100, grid |
| trellis λ₁ (`lambda_log_scale1`) | **SCALAR** | 12.0–17.0; steps 13.5, 14.0, 14.5, 14.75, 15.5, 16.0 |
| trellis λ₂ (`lambda_log_scale2`) | **SCALAR** | 14.0–18.0; steps 16.0, 16.5, 17.0 at λ₁=14.75 |
| aq_coupling.scale | **SCALAR** | −8..+8 (clamped ±1.0 MANDATORY); steps −8, −4, +4 |
| aq_coupling.exponent | **SCALAR** | 0.5–2.0; probe 2.0 |
| delta_dc_weight | **SCALAR** | 0.0–5.0; probe 1.0 (response-surface only — quality collapses q≤70) |
| chroma_distance_scales [Cb,Cr] | **SCALAR ×2** | each 0.1–5.0; pairs [0.5,0.5],[2,2],[1,2],[2,1] (Cb/Cr independently live) |
| moz chroma_quality Δ | **SCALAR (delta)** | −30..0 vs grid q; steps −10, −20 |
| pre_blur σ | **SCALAR** | 0.0–1.0; step 0.4 |
| families / scans / color_modes / downsampling | categorical | quant-table families, progressive modes, 444/420, … |
| aq / deringing / allow_16bit | boolean | both ways |

### zenavif (`zenavif/src/sweep.rs`; resolved mediators in `encode_plan.rs`)

| axis | kind | bound / steps |
|---|---|---|
| quality | SCALAR (rate axis) | 1–100 grid; **resolved `quantizer` is the trained-on mediator** (`SweepCell::feature_row` emits it; also `alpha_quantizer`) |
| vaq strength | **SCALAR** | 0.0–4.0; axis Some(0.5) + probe 2.0 (1.0 is a structural no-op — byte-identical to off) |
| seg_boost | **SCALAR** | 0.5–4.0; probes 1.5, 2.5 |
| alpha_quality Δ | **SCALAR (delta)** | ±25 vs grid q, clamp 1–100 (alpha corpora only, `modes_full_alpha`) |
| speed | ordinal | 1–10; axes {4, 6, 2} + 8 |
| partition_range | ordinal pair | probes (4,16), (16,64) |
| qm / trellis / tune / cdef / rdotx / sgr / segcx / bup / lrf / fdb | boolean probes | both ways each (preset-equal spelling dedupes) |
| subsampling / bit_depth / color_model / alpha_mode | categorical | 444/420, 8/10-bit, YCbCr/RGB, Clean/Dirty/Premultiplied |

`feature_columns()` / `feature_row()` is the picker-training contract:
one numeric column per knob, resolved state preferred (append-only).

### zenjxl (`zenjxl/src/sweep.rs` provenance table)

| axis | kind | bound / steps |
|---|---|---|
| distance | SCALAR (rate axis) | resolved from generic q via `resolve_distance_for_quality` (plateau q≤20 dedupes); `ExplicitDistance` grid for native-distance sweeps |
| `k_info_loss_mul_base` | **SCALAR** | probe 1.3 (default ~1.0) |
| `entropy_mul_table` | **SCALAR ×12** (preset-spelled) | per-DCT-class multipliers; curated probe = `experimental()` preset only |
| effort | ordinal | 1–10; axes {7,5,9} + {3,10} |
| faster_decoding | ordinal | 0–4; lossy probes 4; lossless 2 |
| strategy / encoder_mode / progressive | categorical | Zenjxl/Libjxl/LeanFaster; Reference/Experimental; Single/2 progressive modes |
| gaborish / noise / ans | boolean / pin | gaborish Some(false), noise on, ans Some(false) |
| epf_level | ordinal | −1 auto, 0, 3 |
| lossless: predictor | categorical | None + Some(6) Weighted + Some(0) Zero (5/15 byte-alias the default) |
| lossless: group_size_shift | ordinal | None + Some(0) + Some(3) |
| lossless probes (rct1/wp5/buckets256/props16/seeds2/lloyd) | ordinal/bool | single-knob internal-params probes |

### zenwebp (`zenwebp/src/sweep.rs`)

| axis | kind | bound / steps |
|---|---|---|
| quality | SCALAR (rate axis) | 1–100 grid (lossy; fingerprint-hashed) |
| sns_strength | **SCALAR** (u8) | 0–100; steps None (encoder-derived), 0, 100 |
| filter_strength | **SCALAR** (u8) | 0–100; steps None, 0 |
| partition_limit | **SCALAR** (u8) | 0–100; probe `plim50` |
| method | ordinal | 0–6; axes {4, 6} + 2 |
| segments | ordinal | 1–4; probe Some(1) |
| sharp_yuv / multi_pass_stats / smooth_segment_map / cost_model | boolean / categorical probes | syuv, mpass (live at m4), smooth, parity |
| lossless method × VP8L dial | ordinal (trial-class) | m{4,6,0} × ql{75,100,25} — byte-only, pixels identical |

### zenpng (`zenpng/src/sweep.rs`)

| axis | kind | bound / steps |
|---|---|---|
| compression | ordinal (trial-class) | 9 named presets (rd_core 3); `Effort(n)` 0..=200 spelling aliases by resolved effort |

No metric-class scalars by design: `near_lossless_bits` (0–4) changes
pixels and is deliberately excluded from the curated trial-class axes
(sweep it in metric-scored runs via the classic knob vocabulary, where
`zen-metrics sweep --knob-grid '{"near_lossless_bits":[1,2]}'` reaches
it).

## 5. Scalar-axis gaps (dense-sweep program backlog)

Scalar knobs that exist (fingerprinted / reachable) but are NOT on any
curated plan axis today — i.e. what a dozen-knob dense sweep or a
`--scalar-axes` training run cannot get from `--plan` yet:

- **zenjpeg**: jpegli AQ strength is bool-only on the axes (no
  continuous strength); `chroma_quality` ABSOLUTE form deliberately
  unswept (delta form covers the idiom); trellis `speed` tiers unswept.
- **zenavif**: `vaq_strength` has 2 curated points (0.5, 2.0) — no
  dense ladder; `seg_boost` 2 points (1.5, 2.5); `alpha_quality` delta
  only ±25; no direct `quantizer` axis (mediated via quality — the
  resolved value is in `feature_row`, but plans can't pin qp directly).
- **zenjxl**: `k_ac_quant` (f32) fingerprinted but on no curated axis;
  `fine_grained_step` (u8) likewise; `entropy_mul_table` has a single
  preset probe — no per-class scalar steps; `lossy_search_seeds` is
  structurally dead in `__expert` builds (needs `butteraugli-loop`);
  lossless `lz77`/`palette`/`patches` + `tree_sample_fraction` blocked
  upstream (jxl-encoder#69) and `chroma_subsampling`/alpha axes are out
  of scope per imazen/zenjxl#8.
- **zenwebp**: `sns_strength` curated at endpoints {0,100} only — no
  mid steps; `filter_strength` {None,0} only; `filter_sharpness` (0–7)
  not an axis; near-lossless preprocessing not in the lossless axes.
- **zenpng**: none (by design; see above).

When adding any of these, follow the playbook: validate liveness with
the codec's `sweep_validate` harness first (inert steps are forbidden),
document bounds + step provenance in the module-docs table, and keep
ids additive-only.

## 6. Manifest + tripwire semantics (operational notes)

- `<output>.plan.json` is the no-silent-caps audit: cells, alias
  merges (`duplicates_merged` + per-cell `aliases`), `invalid_skipped`
  strata, `dropped_axes` from the budget ladder, `q_coarsenings`, and
  `over_budget` (the budget could not be met — the plan is complete,
  nothing was sampled away; the caller decides).
- Fingerprint mismatch at execute time (jobexec, fleet tuple path, or
  `resolve_verified` callers) is a deterministic FAILED row — expected
  causes: id-grammar drift between declaring and executing builds, a
  codec dep bump that changed resolved defaults, or a tampered/corrupt
  row. Re-declare from the current build rather than patching fps.
- Plan cells pin machine-dependent knobs (threads=1, parallel=false)
  per playbook pattern 9 — content addressing stays byte-stable across
  boxes.
- `--emit-cells` requires integer q (CellId.q is i64).
- Feature gating: `--plan` paths need `--features sweep` (which pulls
  jpeg/webp/avif) + `jxl`/`png` for those codecs. The default (no
  `sweep`) build has no plan machinery.
