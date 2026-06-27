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
| zenjpeg | `zenjpeg::encode::sweep` | `rd_core`, `modes_full`, `scalar_dense` | lossy | yes |
| zenavif | `zenavif::sweep` | `rd_core`, `modes_full`, `modes_full_alpha`, `scalar_dense` | lossy (+alpha probes) | yes |
| zenjxl | `zenjxl::sweep` | `rd_core`, `modes_full`, `scalar_dense` | lossy (VarDCT) + lossless (modular) | yes |
| zenwebp | `zenwebp::sweep` | `rd_core`, `modes_full`, `scalar_dense` | lossy (VP8) + lossless (VP8L) | yes |
| zenpng | `zenpng::sweep` | `rd_core`, `modes_full`, `scalar_dense` | all-lossless | no (9 cells max; over-budget reported, never sampled) |
| zengif | `zengif::sweep` | `rd_core`, `modes_full`, `scalar_dense` | lossy (quantizer-driven) | no (over-budget reported, never sampled) |
| zentiff | `zentiff::sweep` | `rd_core`, `modes_full`, `scalar_dense` | all-lossless | no (≤16 cells; over-budget reported, never sampled) |

Common planner shape: `SweepAxes` (most-important-first axis vectors) ×
`QualityGrid` (`Step5` 21-point floor / `TrainingDense` 31-point /
`Explicit`; zenjxl also `ExplicitDistance`) → `SweepBuilder::plan()` (the
`__expert` codecs) / a free `plan_constrained(axes[, grid],
compute_limit, max_deviations)` (the public-API codecs png/gif/tiff) →
`SweepPlan { cells, …, compute_tier_skipped }`. Cells are emitted
main-effects-first (deviations 0, then 1, then combos); every reduction
is reported, never silent.

`scalar_dense` is the third plan mode on every codec: the dense per-knob
ladders a trained **scalar head** fits (continuous knobs laddered finer
than `modes_full`; the codec's compute axis kept dense). Pair it with
`--max-deviations 1` (the isolated main-effects regime — auto-defaulted
when `--plan scalar_dense` and `--max-deviations` is absent) so the head
sees clean per-axis response curves instead of cartesian interactions.

Two cross-codec constraint knobs (both default off → unconstrained):

- **`compute_limit: Option<u8>`** (`--compute-limit`) drops cells whose
  `compute_tier()` exceeds the cap, recording every dropped id in the
  manifest's `compute_tier_skipped` — the no-silent-caps report for a
  CPU-bound fleet or a "fast configs only" picker. `compute_tier()` is
  an ordinal cost proxy each codec exposes (jpeg: mode-cost ladder;
  webp/jxl: method/effort; png: effort; gif: quantizer backend band;
  tiff: compression-method ladder).
- **`max_deviations: Option<u8>`** (`--max-deviations`) keeps only cells
  within N axis deviations of the default stratum (`1` = isolated
  main-effects, `0` = the default stratum alone).

`zenmetrics-cli`'s bridge (`crates/zenmetrics-cli/src/sweep/plan.rs`):

- `build_plan(codec, name, budget, q_grid, compute_limit,
  max_deviations) -> BuiltPlan` — the single dispatch point. Writes the
  audit manifest to `<output>.plan.json` (which now also records
  `compute_limit` / `max_deviations` / `compute_tier_skipped`).
- `PlannedCell { q, knob_json, config: PlannedConfig }` — one encode
  cell; `PlannedConfig` is the per-codec fully-built config enum.
- `resolve_verified(codec, cell_id, q, fp_hex) -> PlannedConfig` — the
  executor half: reconstructs the config from the **self-describing
  cell id** (`config_from_cell_id` / `variant_from_cell_id`), recomputes
  the resolved-state fingerprint, and fails loudly on mismatch (the
  id-grammar-drift tripwire; never a silently wrong encode).

gif/tiff are **plan-only** in the bridge — they have no `--knob-grid`
JSON vocabulary, so a `--knob-grid` sweep on them errors with a
"use --plan" hint. Their encoded variants decode back through
`decode.rs` (gif via `decode_gif`'s first composed frame; tiff via the
PixelBuffer funnel) so the encode→decode→score round-trip closes.

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
  - zengif: `gif-<backend>[-d<dither>]` (lossy — quality lives in the
    `_q<q>` token; re-attached at resolve time)
  - zentiff: `tiff-<method>[-hpred][-big]` (lossless — q=0 sentinel)
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

1. **Local / chunk-mode CLI** — `zenmetrics sweep --codec C --plan
   NAME [--plan-budget N] [--compute-limit N] [--max-deviations N]
   --q-grid … --sources DIR --output OUT.tsv`. Mutually exclusive with
   `--knob-grid`. Builds the plan once per sweep, writes `OUT.plan.json`,
   walks cells × images via rayon. Requires `--features sweep` + the
   codec feature (`sweep` force-pulls jpeg/webp/avif; add
   `jxl`/`png`/`gif`/`tiff` explicitly).
2. **Job system** — `… --plan NAME --dry-run --emit-cells cells.jsonl`
   emits one declare item per (source × cell): `{image_path, codec,
   q:i64, knob_tuple_json, source_sha}` (q must be integer-valued).
   `zenfleet_ctl declare_encodes` → content-addressed `DesiredJob`s →
   `zenmetrics jobexec` executes each via `resolve_verified` (encode
   jobs emit bytes; metric jobs re-encode + score). See
   `docs/RUNNING_JOBS.md` §4b.
3. **Vast.ai chunk fleet** — plan-mode input parquets carry the
   identity JSON per row (v26 schema `image_path/codec/q:int64/
   knob_tuple_json`, NO schema change). Generate with:
   ```bash
   zenmetrics sweep --codec C --plan NAME [--plan-budget N] \
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

### 3a. Granular, work-stealing, resumable chunk sizing (`plan-chunks`)

`generate_sweep_input.py` sizes chunks by a flat `--cells-per-chunk`
heuristic, and the early Hetzner split assigned ONE giant static chunk
per box (≈5 h) — so a dead box stranded its whole multi-hour chunk and a
fast box couldn't claim more. `zenmetrics plan-chunks` (the canonical
Rust sizer; also `fleet plan-chunks`) replaces the heuristic with
**estimation-balanced granularity**: it reads the SAME image-major input
parquet the worker consumes, estimates each cell's encode+score wall time
+ peak RAM from the SAME models `fleet-plan` uses (codec
`estimate_encode_resources` for encode wall_ms/peak_ram; the per-metric
GPU estimators, or a `--cpu-score-mp-ms` rate, for score time), and packs
a **contiguous run of images** into each chunk so the chunk's estimated
`Σ(encode+score) ≤ --target-seconds` (default 300 = 5 min) AND its peak
host RAM ≤ `--mem-budget-mb`. It emits the SAME canonical `chunks.jsonl`
(one record per chunk; `row_range` stays image-major contiguous), just
MANY small chunks instead of N=box-count.

```bash
zenmetrics sweep --codec C --plan NAME [--plan-budget N] \
  --sources DIR --q-grid … --dry-run --emit-cells cells.jsonl \
  --output /tmp/plan.tsv
python3 scripts/sweep/generate_sweep_input.py --codec C \
  --run-id RUN --cells-jsonl cells.jsonl \
  --source-dir-r2 s3://…/sources --out-dir OUT       # writes the input parquet
fleet plan-chunks --codec C --run-id RUN \
  --input-parquet OUT/C_RUN_input.parquet \
  --input-parquet-r2 s3://zentrain/RUN/input/C_RUN_input.parquet \
  --source-dir-r2 s3://…/sources --out OUT/chunks.jsonl \
  --metrics ssim2,zensim --target-seconds 300 --mem-budget-mb 20000
# upload OUT/chunks.jsonl → s3://coefficient/jobs/RUN/chunks.jsonl; launch the
# omni worker normally — it loops, claiming each granular chunk.
```

All five properties hold **simultaneously**, and four of them are the
omni worker's *existing* guarantees (`plan-chunks` only adds the sizing):

- **Granular ≤5-min** — the sizer's `Σ(encode+score) ≤ target` bound.
- **Work-stealing** — the worker's token-race claim
  (`zenfleet-vastai::worker::claim::try_claim`); a STALE claim (dead box,
  past `stale_secs`) is re-stealable, so a sub-5-min chunk completes
  elsewhere. Fast boxes claim more because each claim is one atomic op.
- **Resumable** — completion = the per-chunk omni sidecar; `try_claim`
  returns `AlreadyDone` once it exists, so a re-launch skips done chunks
  and the `gap` reconcile re-runs only the missing ones. A dead box loses
  ≤5 min.
- **Decode-once** — the inline pipeline groups a chunk's rows by
  `(codec, knob_tuple_json)`; each source is decoded exactly once and all
  its q/knob cells reuse that decode (zero re-decode).
- **Corruption-impossible** — the omni parquet is built completely on
  local disk, then uploaded with ONE atomic S3/R2 PUT; an interrupted
  upload leaves NOTHING (never a truncated sidecar), and the idempotent
  skip means a re-run can't double-write.

Live-R2 proof of the work-stealing + resumability claim (exactly-once,
dead-box re-steal, resume-skip) against the real `try_claim`:
`crates/zenfleet-vastai/tests/claim_workstealing_r2.rs` (gated on
`ZEN_R2_SMOKE=1`). Sizer unit tests:
`crates/zenmetrics-cli/src/plan_chunks.rs::tests`.

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
| aq_coupling.scale | **SCALAR** | −8..+8 (clamped ±1.0 MANDATORY); steps −8, −4, −2, +2, +4, +8 (6-pt symmetric, zenjpeg fff81900) |
| aq_coupling.exponent | **SCALAR** | 0.5–2.0; probes 0.5, 2.0 (1.0 default unspelled — would alias) |
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
| vaq strength | **SCALAR** | 0.0–4.0; axis Some(0.5) + probes 0.25, 2.0, 3.0 (1.0 is a structural no-op — byte-identical to off; 0.0 excluded pending semantics) |
| seg_boost | **SCALAR** | 0.5–4.0; probes 0.75, 1.5, 2.5, 4.0. **Still-envelope equivalence (proven by encode, e9de3022)**: `seg_boost(x)` ≡ `vaq_strength(x)` byte-identically on still encodes, so the two ladders interleave into one alias-free joint 8-point effective ladder {0.25, 0.5, 0.75, 1.5, 2.0, 2.5, 3.0, 4.0} |
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
| `k_ac_quant` | **SCALAR** | > 0; steps 0.575, 0.65, 0.88, 1.0 around the 0.765 default (0.65 = the jxl-encoder#25 flip value, sanctioned as a learned-dispatch axis; 4c0d672f) |
| `fine_grained_step` | **SCALAR** (u8) | 1–8; live steps 1, 3. **Multiples of 4 are structurally dead** — the only consumer (non-aligned 32×32-class pass) skips `(cy\|cx) % 4 == 0` positions, so step 4/8 ≡ `non_aligned_eval=false` (4c0d672f) |
| `entropy_mul_table` | **SCALAR ×12** (preset-spelled) | per-DCT-class multipliers; 3 curated presets: `experimental()`, `screenshot_suppressed()`, `high_d_photo_smooth_suppressed()` (4c0d672f) |
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
| sns_strength | **SCALAR** (u8) | 0–100; steps None, 0, 25, 80, 100 → effective ladder {0, 25, 50, 80, 100} (25=Drawing / 80=Photo preset constants; 700aa4a8) |
| filter_strength | **SCALAR** (u8) | 0–100; steps None, 0, 10, 30, 100 → effective {0, 10, 30, 60, 100} (10=Drawing, 30=Photo; 700aa4a8) |
| filter_sharpness | **SCALAR** (u8) | 0–7; steps None, 3, 6, 7 → effective {0, 3, 6, 7} — NEW axis (`-shp<v>` id token; 3=Photo, 6=Drawing, 7=clamp bound; 700aa4a8) |
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
`zenmetrics sweep --knob-grid '{"near_lossless_bits":[1,2]}'` reaches
it).

## 5. Scalar-axis gaps (dense-sweep program backlog)

Scalar knobs that exist (fingerprinted / reachable) but are NOT on any
curated plan axis today — i.e. what a dozen-knob dense sweep or a
`--scalar-axes` training run cannot get from `--plan` yet.

**Closed 2026-06-12** (all four codecs landed scalar-ladder densification
on their mains — zenjpeg `fff81900`, zenavif `e9de3022`, zenjxl
`4c0d672f`, zenwebp `700aa4a8`; §4 reflects the shipped ladders):
zenavif vaq/seg dense ladders (joint 8-point, alias-free, still-envelope
equivalence proven), zenjxl `k_ac_quant` + `fine_grained_step` +
2 extra `entropy_mul_table` presets, zenwebp sns/filter mid-ladders +
the new `filter_sharpness` axis, zenjpeg aq_coupling scale/exponent
densification.

Still open:

- **zenjpeg**: `chroma_quality` ABSOLUTE form deliberately unswept
  (delta form covers the idiom); trellis `speed` tiers unswept.
- **zenavif**: `alpha_quality` delta only ±25.
- **zenjxl**: `entropy_mul_table` has no per-class scalar steps (3
  named presets only); `lossy_search_seeds` is structurally dead in
  `__expert` builds (needs `butteraugli-loop`); lossless
  `lz77`/`palette`/`patches` + `tree_sample_fraction` blocked upstream
  (jxl-encoder#69) and `chroma_subsampling`/alpha axes are out of
  scope per imazen/zenjxl#8.
- **zenwebp**: near-lossless preprocessing not in the lossless axes.
- **zenpng**: none (by design; see above).

### Blocked on encoder knob (not reachable from the sweep layer)

These are NOT sweep-harness gaps — the encoder exposes no setter, so a
curated axis cannot exist until the codec (or its vendored encoder)
grows the knob. Verified against encoder source on 2026-06-12:

- **zenavif direct `quantizer` (qp) axis**: neither zenavif's
  `EncoderConfig` nor zenravif exposes a quantizer setter
  (`quality_to_quantizer` is internal). The resolved `quantizer` /
  `alpha_quantizer` ARE already the picker-training mediators via
  `feature_row` — plans just can't pin qp directly. (e9de3022 doc)
- **zenjpeg scalar AQ strength**: `aq_enabled` is the only direct AQ
  knob; the AQ-field shape has no config-exposed strength scalar
  (`quant/aq/mod.rs` bakes `mul = K_AC_QUANT × dampen`,
  `add = (1−dampen) × base_level`). The aq_coupling scale/exponent
  ladders (§4) are the swept proxy. (fff81900 doc)

When adding any remaining axis, follow the playbook: validate liveness
with the codec's `sweep_validate` harness first (inert steps are
forbidden), document bounds + step provenance in the module-docs table,
and keep ids additive-only.

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
- **Codec-rev pairing (updated 2026-06-12):** the 2026-06-12 scalar-axis
  landings extend each codec's id grammar (e.g. zenwebp's new `-shp<v>`
  token) and enter the fingerprints, so declaring and executing builds
  MUST both sit at (or past) the new codec revs or the tripwire above
  fires on every new-axis cell. Both pin surfaces were moved
  accordingly: (a) CI sibling clones now pin zenjpeg `94fb6ec6` /
  zenavif `e9de3022` / zenjxl `4c0d672f` / zenwebp `700aa4a8` / zenpng
  `2e82aa94` (their mains as of 2026-06-12); (b) fleet images have no
  rev pins of their own — they `COPY` a binary built from the local
  sibling checkouts, which were advanced to the same revs. Rebuild the
  sweep image before declaring plans that use the new axes; mixing a
  pre-axis worker binary with a post-axis plan fails closed (FAILED
  rows, no silent mis-encodes).
- Plan cells pin machine-dependent knobs (threads=1, parallel=false)
  per playbook pattern 9 — content addressing stays byte-stable across
  boxes.
- `--emit-cells` requires integer q (CellId.q is i64).
- Feature gating: `--plan` paths need `--features sweep` (which pulls
  jpeg/webp/avif) + `jxl`/`png` for those codecs. The default (no
  `sweep`) build has no plan machinery.

## 7. HDR sweeps (`--hdr`, added 2026-06-12)

The gate for all HDR training-data collection. SDR sweeps flow
`decode_image_to_rgb8 → codec encode (RGB8) → decode-back (RGB8) → u8
metric kernels`; a 16-bit PQ reference pushed through that silently
quantises absolute-luminance code values to "8-bit sRGB" — scores look
plausible and mean nothing (the imazen/zenmetrics#25 failure class).
`--hdr` replaces every stage (`sweep::hdr` module):

- **References**: 16-bit PQ PNGs (PNG 3.0 cICP, transfer 16) per the
  imazen-26-png-v2 corpus contract — samples are PQ code values of
  absolute light with SDR white = 203 cd/m² (BT.2408), so the PQ EOTF
  alone recovers cd/m². Primaries (1 / 12) pass through. HLG and
  cICP-less PNGs are rejected loudly. The SDR decode path equally
  refuses PQ/HLG-signaled PNGs (`decode.rs` tripwire) instead of
  crushing them.
- **Codec round-trip**: only codecs with a true HDR path run.
  **zenjxl** is wired: 16-bit PQ samples enter the zencodec adapter as
  `Rgb16` with `Metadata::cicp` driving the codestream color encoding
  (PQ + BT.2100/P3); decode-back requires the decoded descriptor to
  come back PQ-tagged and errors otherwise. **SDR-only today (refused,
  never approximated): zenjpeg, zenwebp, zenpng, zenavif.** zenavif
  10-bit PQ and zenpng 16-bit cICP re-encode are the natural next
  candidates; either needs an honest decode-back-to-nits path before
  it may join.
- **Scoring**: `zenmetrics_api::hdr::hdr_feeding` per metric — cvvdp /
  butteraugli linear planes, GPU ssim2 integrated PU21, iwssim float
  PU(luma) gray, SSIM-family PU-rescale u8; dssim is Unsupported by
  design. Scorers live in a process-static cache mirroring
  `MetricCache`'s cubecl-pool discipline. GPU metrics need an explicit
  `--gpu-runtime cuda|wgpu`.
- **Output schema**: the TSV (and therefore the fleet omni parquet,
  whose schema is inferred from it) gains a trailing `hdr_mode` column,
  value `pq1000` (PQ-decoded absolute nits, 1000 cd/m² reference peak).
  SDR sweeps stay byte-identical — no column is added.
- **Not wired in HDR mode (validated at startup, loud errors)**:
  `--plan` (PlannedCell encodes via the RGB8-typed path),
  `--feature-output` (u8 feature extractors), `--distorted-out-dir` /
  `--pairs-tsv` (8-bit PNG writers), `--use-orchestrator`. zenjxl
  expert knobs are also rejected — HDR cells accept
  `{lossless, distance, noise, effort}` only, and unknown knobs error
  rather than being silently dropped.

### v26 → v27 chunk schema note

`ChunkRecord` (chunks.jsonl) and `InlineGroupSpec` gain an `hdr: bool`
(serde-default `false`) — strictly additive; every v26 chunks.jsonl
deserialises unchanged. An HDR chunk on a worker image whose
zenmetrics-cli lacks the `hdr` feature fails loudly at `run_sweep`
validation (it can never silently score SDR). The production vastai
worker (`zenfleet-vastai`) builds with `hdr` enabled as of this
change.
