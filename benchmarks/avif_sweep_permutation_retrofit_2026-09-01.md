# AVIF svt-rs/aom-rs sweep: permutation-builder retrofit, alias audit, zenfleet reconcile (2026-09-01)

Companion to [`avif_sweep_subsample_2026-09-01.md`](avif_sweep_subsample_2026-09-01.md),
which registered the grid and launched it. That sweep was declared as a **naive
Cartesian product** — `32 sources x 10 speeds x 30 q x 2 backends = 19,200 cells`
— through `scripts/jobsys/avifsvt_cells.py`, whose last line is literally
`assert cells == len(names) * len(speeds) * len(qs)`: an assertion that **no
deduplication happened**.

The zen codecs already own a settings-permutation builder that exists to prevent
exactly this. This document (a) names it and its owners, (b) audits the live grid
against it with `output_sha` evidence, and (c) records the retrofit: the same
19,200-cell intent, flattened to **15,776** cells through the canonical
machinery, reconciled into the running zenfleet run with **zero completed cells
orphaned**.

Headline: **3,424 of 19,200 declared cells (17.8 %) are byte-wise duplicate
work**, and **92 of them had already been encoded** when the audit ran.

---

## 1. PRIOR ART — the machinery, its real names, and where it lives

The vocabulary is not "permutation builder / flattener". It is
**`SweepAxes` -> `SweepBuilder::plan()` -> `SweepPlan`**, deduplicated by
**resolved-state byte-identity fingerprints**. Greps that find it:
`SweepAxes`, `SweepBuilder`, `SweepPlan`, `SweepCell`, `cross(`, `fingerprint(`,
`duplicates_merged`, `aliases`, `deviations`, `compute_tier`,
`config_from_cell_id`, `resolve_verified`, `encode_fingerprint`. The words
*permutation*, *flatten*, *cartesian* and *knob_tuple* return essentially
nothing — which is why two prior lanes missed it.

### 1.1 Doctrine

| doc | what it is |
|---|---|
| `~/work/zen/zenjpeg/docs/VARIANT_GENERATION.md` | **The codec-neutral playbook**, 18 numbered patterns. Pattern 4 is the load-bearing one: *"A sweep cell's identity is its resolved state, not its config spelling… Equal fingerprint ⇒ identical bytes for the same input ⇒ one encode serves all aliased spellings. In zenjpeg's `rd_core × Step5` this merges **46 %** of the naive cross product before any encode runs."* It also states the rule this document obeys: *"Every exclusion must be proven by encode, not by reading code."* Each codec repo carries an adoption copy at `<repo>/docs/VARIANT_GENERATION.md`. |
| `zenmetrics/docs/PLAN_SWEEPS.md` | The cross-codec `PlanSpec` contract. §2 defines the durable per-cell identity; §4 is the per-codec axis inventory; §6 the manifest/tripwire semantics. |
| `zenmetrics/docs/JXL_LOSSY_KNOBSPACE_ABLATION_PROGRAM.md` | The worked JXL precedent: the declare -> gap -> reconcile iteration loop. |

### 1.2 Owners (file paths)

**Codec-side planners** — each codec owns *what cells exist*:

| codec | file |
|---|---|
| zenjpeg | `~/work/zen/zenjpeg/zenjpeg/src/encode/sweep.rs` (the reference implementation, `#![cfg(feature = "__expert")]`) |
| zenjxl | `~/work/zen/zenjxl/src/sweep.rs` |
| **zenavif** | **`~/work/zen/zenavif/src/sweep.rs`** |
| zenwebp | `~/work/zen/zenwebp/src/sweep.rs` |
| zenpng / zengif / zentiff | `<repo>/src/sweep.rs` (free `plan_constrained`, no builder) |

Key entry points, zenavif spelling:

```rust
pub struct SweepBuilder { .. }                                  // sweep.rs
pub fn new(axes: SweepAxes, grid: QualityGrid) -> Self
pub fn with_budget(self, max_cells: usize) -> Self
pub fn with_compute_limit(self, max_tier: u8) -> Self
pub fn with_max_deviations(self, max_deviations: u8) -> Self
pub fn plan(&self) -> SweepPlan
pub fn fingerprint(config: &EncoderConfig) -> u64                // THE dedup key
pub fn compute_tier(config: &EncoderConfig) -> u8
pub fn config_from_cell_id(base_id: &str, quality: f32) -> Result<EncoderConfig, String>
fn cross(axes: &SweepAxes, q_points: &[f32]) -> (Vec<SweepCell>, Vec<String>, usize)
```

`cross()` is the flattener: pass 1 enumerates strata and resolves each into a
real `EncoderConfig` through the encoder's own resolution path (validity
failures are **reported** in `invalid_skipped`, never silently dropped); pass 2
expands quality and merges by fingerprint, keep-first, recording every merge in
`SweepCell::aliases` and `SweepPlan::duplicates_merged`.

**Executor-side bridge** — `zenmetrics/crates/zenmetrics-cli/src/sweep/plan.rs`:
`build_plan(codec, name, budget, q_grid, compute_limit, max_deviations)` is the
single cross-codec dispatch point; `resolve_verified(codec, cell_id, q, fp_hex)`
is the executor half, which recomputes the fingerprint and **hard-fails** on a
mismatch (`jobexec.rs:332` and `sweep/run.rs:173`).

**The naive path, for contrast** — `zenmetrics/crates/zenmetrics-cli/src/sweep/grid.rs`:
`KnobGrid::iter_tuples()` is a plain row-major product with zero dedup and zero
validity filtering. `--plan` and `--knob-grid` are `conflicts_with` each other.

**A second dedup tier exists** and is worth knowing about:
`zenjxl::sweep::encode_fingerprint` (encode-COMPUTE dedup, distinct from the
per-knobset row identity) is consumed by
`zenfleet-ctl::declare_encodes` via the `EncodeDeclareItem::encode_fp` field:
when present, ONE encode job is declared per `(codec, source_sha, encode_fp)`
group. **It is not usable yet** — see §5.3.

### 1.3 How the JXL sweeps used it (the precedent)

```
zenmetrics sweep --codec zenjxl --plan lossy_dense --q-grid <…> \
    --dry-run --emit-cells cells.jsonl
  -> 13,320 items {image_path, codec, q, knob_tuple_json{cell,fp,plan}, source_sha}
  -> zenfleet-ctl declare-encodes --cells cells.jsonl --out manifest.json
  -> 13,320 DesiredJobs
```

Plans are **named presets in codec source**, not committed grid files; the only
committed artifacts are the emitted `cells.jsonl` and the `<output>.plan.json`
audit manifest (`cells`, `duplicates_merged`, `invalid_skipped`,
`compute_tier_skipped`, `q_coarsenings`, `over_budget`, `dropped_axes`,
`aliases`). Multi-codec drivers: `scripts/picker/run_sweeps.sh`,
`scripts/picker/run_resweep_jxl_webp.sh`, `scripts/sweep/datagen_encode.sh`,
`scripts/sweep/hetzner_cpu_sweep.sh`.

### 1.4 The canonical `knob_tuple_json` schema (the compatibility target)

All 7 canonical picker datasets (`/mnt/v/output/canonical-picker-2026-06-27/`)
carry ONE shape — compact separators, keys sorted `cell,fp,plan`:

```json
{"cell":"s8-noqm","fp":"8657c728f73693f5","plan":"modes_full"}
```

`fp` varies with `q` (the fingerprint covers the resolved quantizer), `cell`
carries **no** `_q<q>` token. The zenavif_lossy dataset is 48 cells x 7 q = 336
distinct tuples on plan `modes_full`, sweep run `mandfix4-zenavif-1782593621`.

**There are two live schemas, and this sweep is on the other one.** The
`--knob-grid` path emits a raw knob map (`{"backend":"svt-rs","speed":4}`),
which is what `avifsvt_cells.py`, `hdrgrid_cells.py` and the 2026-08-30 wave
all use. Consequences of the raw-map schema, measured against the consumers:

- `assemble/flat_picker.rs:151` yields NULL `cell`/`fp`/`knob_plan` (silent);
- `scripts/picker/omni_to_pareto.py:61` renders `config_name` as
  `backend=svt-rs,speed=4` instead of `s4`;
- `scripts/picker/check_mandatory_coverage.py:40` regexes that `config_name`
  for `-420` / `-rgb` tokens and **hard-fails** for zenavif;
- `scripts/analysis/omni_to_pareto.py:27` does `["cell"]` (KeyError, not `.get`).

**Migration is not free and was deliberately NOT done here** — see §5.2.

---

## 2. AUDIT — where the duplicates are

Two alias classes, one per axis. Both were derived from the encoders' own
resolution code and then **confirmed by encode**.

### 2.1 svt-rs: speeds 7, 8, 9 and 10 are ONE encode

`zenavif/src/encoder_svt_rs.rs::speed_to_svt_preset` maps zenavif's speed dial
1..=10 onto SVT presets. Upstream `imazen/svtav1`'s
`AvifEncoder::speed_to_preset` — which that helper's own doc comment says it
mirrors — clamps `.min(9)`, because **C remaps every all-intra preset above M9
down to M9** (`enc_handle.c:4416-4419`). The resulting map:

| speed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|--|--|--|--|--|--|--|--|--|--|
| raw `((s-1)*13+4)/9` | 0 | 1 | 3 | 4 | 6 | 7 | 9 | 10 | 12 | 13 |
| **effective preset** | 0 | 1 | 3 | 4 | 6 | 7 | **9** | **9** | **9** | **9** |

So the dial has **7 distinct settings, not 10**, and presets 2, 5 and 8 are
unreachable through it at all (a *gap*, not a duplicate — noted for the DOE
lane, not acted on).

**A drifted mirror was found here and repaired** (zenavif `292582fb`):
`speed_to_svt_preset` returned the un-clamped 0..=13 value, and its test
`speed_to_preset_matches_upstream_boundaries` asserted `speed_to_svt_preset(10)
== 13` while the upstream test it names asserts `9`. Byte-neutral (upstream
records presets 9/10/13 as each byte-identical to C's M9 output; §3 confirms it
end to end) and both mono preset floors are unaffected — speeds 1..=6 map to
0/1/3/4/6/7 either way and 7..=10 clear `MONO_HBD_MIN_PRESET` (9) in both — but
it made the API advertise a distinction the encoder does not have, which is
precisely what upstream's clamp exists to stop.

**The launch lane measured this alias and did not recognise it.** Its own smoke
table (`avif_sweep_subsample_2026-09-01.md` §3) reads `speed 7 -> 12 ms /
121 ms` and `speeds 8-10 -> 12 ms / 120 ms`.

### 2.2 Both backends: q 98 and q 100 are ONE encode

Each backend clamps its lossy quality dial away from the lossless quantizer, and
the top grid point falls into the clamp:

| backend | mapping | q=98 | q=100 |
|---|---|--|--|
| svt-rs | `quality_to_qp_gated` = `AvifEncoder::quality_to_qp_static(q).max(1)`, i.e. `round(63 - (q-1)·63/99)` clamped to ≥1 | QP **1** | QP 0 -> clamped **1** |
| aom-rs | `aom_rs_cq_level` = `round((100-q)·63/100).clamp(1,63)` | cq **1** | cq 0 -> clamped **1** |

Every other point of the 30-point grid is distinct on both backends
(29 distinct quantizers of 30 grid points). The clamps themselves are correct
and stay: quantizer 0 is lossless, which is not what "quality 100 of a lossy
ladder" means.

### 2.3 aom-rs speed is injective — no alias there

`--cpu-used` 0..=9 is passed through raw; all 10 values are distinct encodes
(gated by `dedup::tests::aom_rs_speed_dial_is_injective`).

### 2.4 Exact accounting

| | declared | flattened | removed |
|---|--:|--:|--:|
| svt-rs (32 x 10 speeds x 30 q) | 9,600 | **6,496** (32 x 7 x 29) | 3,104 |
| aom-rs (32 x 10 speeds x 30 q) | 9,600 | **9,280** (32 x 10 x 29) | 320 |
| **total** | **19,200** | **15,776** | **3,424 (17.8 %)** |

Breakdown of the removals: 2,880 svt speed-alias cells (speeds 8/9/10), 224 svt
q=100 cells (on the 7 surviving speeds), 320 aom q=100 cells.

The flattener reproduces these numbers independently — `zenmetrics sweep
--dry-run` reports `203 cells/image (from 300 spellings, 97 merged by resolved
state, 0 without a resolver)` for svt-rs and `290 (from 300, 10 merged)` for
aom-rs.

---

## 3. EVIDENCE — content-addressed, with controls

### 3.1 Ledger `output_sha` (production data, no new encodes)

Read from `s3://zentrain/jobs/avifsub-{svt,aom}-enc-20260901/ledger/`,
latest-wins per `(image_path, q, knob_tuple_json)`, at 2026-09-02T01:20Z
(1,410 svt + 403 aom DONE):

| | identical-`output_sha` groups | redundant encodes already burned |
|---|--:|--:|
| svt-rs | 75 (47 q-only, 26 speed-only, 2 mixed) | **79** |
| aom-rs | 13 (all q-only) | **13** |

- **Every q-alias group's q-set is exactly `{98, 100}`.** 47 of 47 on svt-rs,
  13 of 13 on aom-rs. Zero groups with any other q-set on aom-rs.
- **Of the 30 `(image, q)` cells now covered at ≥2 of speeds {7,8,9,10},
  30 have identical `output_sha`.** 30/30, no exceptions.

### 3.2 Direct encode probe, with a discriminating control

`zenmetrics sweep --codec zenavif --knob-grid '{"backend":["svt-rs"],
"speed":[6,7,8,9,10]}' --q-grid 50,90,98,100 --metric ssim2`, 2 corpus images
(`8288.scale375x667`, `9032.scale1024x1536`), 40 cells:

| image | q | speed 6 (preset 7) | speeds 7, 8, 9, 10 (preset 9) |
|---|--:|--:|--:|
| 8288 | 50 | 19,808 B / ssim2 70.958172 | **31,957 B / 67.363667** (all four) |
| 8288 | 90 | 48,153 B / 77.190790 | **76,329 B / 77.297388** (all four) |
| 8288 | 98 **and** 100 | 68,839 B / 77.491890 (both) | **112,166 B / 77.515143** (all eight) |
| 9032 | 50 | 49,294 B / 77.563596 | **54,877 B / 76.825191** (all four) |
| 9032 | 90 | 128,463 B / 89.076772 | **147,723 B / 88.899967** (all four) |
| 9032 | 98 **and** 100 | 216,731 B / 89.770437 (both) | **250,768 B / 89.708746** (all eight) |

Identical encoded size AND identical SSIMULACRA2 to six decimals across the M9
class and across q 98/100, while **speed 6 differs on every single cell** — the
control that proves the probe discriminates.

### 3.3 Where the gates now live

| claim | gate |
|---|---|
| speeds 7..=10 -> preset 9 | `zenavif encoder_svt_rs::tests::speed_to_preset_matches_upstream_boundaries` |
| the planner merges the M9 class (7 cells, 3 merges) | `zenavif sweep::tests::svt_speed_dial_merges_the_m9_class` |
| q98 ≡ q100 on svt-rs | `zenavif sweep::tests::svt_quality_98_and_100_are_one_cell` |
| q98 ≡ q100 on aom-rs, q96 does NOT merge | `zenmetrics sweep::dedup::tests::aom_rs_q98_and_q100_share_one_identity` |
| aom-rs `--cpu-used` is injective | `zenmetrics sweep::dedup::tests::aom_rs_speed_dial_is_injective` |
| svt-rs M9 class merges on the knob-grid path, speed 6 does not | `zenmetrics sweep::dedup::tests::svt_rs_m9_speed_class_shares_one_identity` |
| svt-rs and zenravif never merge | both repos' `*_never_merge` tests |
| **adding the backend axis moved no archived fingerprint** | `zenavif sweep::tests::backend_axis_does_not_move_archived_zenravif_fingerprints` — replays 12 `(cell, q, fp)` triples minted by the pre-axis build and shipped in `/mnt/v/output/canonical-picker-2026-06-27/zenavif_lossy/` |

---

## 4. THE RETROFIT

### 4.1 zenavif — the planner learns about backends (`292582fb`, on `main@origin`)

`SweepAxes` had no backend axis and `Stratum::id()` no backend token, so an
svt-rs cell **could not be expressed as a plan cell at all**. Landed:

- `SweepAxes.backends: Vec<Av1Backend>` (index 0 = default stratum). Every
  pre-existing preset pins `vec![Av1Backend::Zenravif]`, so ids, deviation
  counts and fingerprints are unchanged — pinned by the golden test above.
- Cell-id grammar gains `-svt`:
  `s<speed>[-svt][-noqm][-420][-bd8|-bd10][-rgb][-vaq<f>][-trel][<probe>]`.
  Zenravif renders no token, so archived ids stay valid.
- **`fingerprint()` resolves svt-rs through the mediators that backend actually
  reads** (`speed_to_svt_preset`, `quality_to_qp_gated`) instead of zenravif's
  `quality_to_quantizer` + `speed_derived` search table, which the svt-rs path
  never consults. This is the whole correctness point: hashing zenravif's
  mediators for an svt cell would both **miss** the two real alias classes and
  **risk merging** cells that differ.
- `SweepAxes::svt_speed_dense()` + `by_name("svt_speed_dense")`.
- Without the `encode-svt-rs` feature, `validate()` rejects the backend, so
  every svt stratum lands in `invalid_skipped` — reported, never silent
  (gated by `svt_strata_are_reported_invalid_without_the_feature`).

### 4.2 zenmetrics — the knob-grid path stops being naive

- **`sweep/dedup.rs`** (new): `knob_cell_identity(codec, q, knobs) -> Option<u64>`,
  the resolved-state identity of a `--knob-grid` cell. `None` = no registered
  resolver = **no dedup** (the row-safe baseline). The AVIF arm routes through
  `encode::avif_config_from_knobs` + `zenavif::sweep::fingerprint` for
  zenravif/svt-rs, and `encode::aom_rs_cq_level` + `--cpu-used` for aom-rs.
- **`encode::avif_config_from_knobs`** (extracted from `encode_avif`): the
  dedup resolves a knob tuple through the *same* code the encoder runs, rather
  than a mirror that can drift.
- **`encode::aom_rs_cq_level`** (extracted): one owner for the aom-rs quality
  mediator, documented as non-injective at the top of the dial.
- **`sweep --dry-run [--emit-cells]` now works with `--knob-grid`**, not just
  `--plan`. It writes the same `<output>.plan.json` audit shape
  (`cells`, `cells_before_dedup`, `duplicates_merged`, `cells_without_resolver`,
  `aliases[]`) and the same `zenfleet-ctl declare-encodes` item lines.
- **`--emit-cells-image-path full|basename`** (default `full`, the historical
  plan-path shape). `image_path` is half the content-addressed `CellId`, so this
  decides whether declared jobs join an existing run's identity space or start a
  new one; fleet runs that resolve sources against `ZEN_CORPUS_PREFIX` need
  `basename`.

### 4.3 How the DOE lane declares a plan through this

**Curated plan (preferred — the codec owns the axes):**

```sh
zenmetrics sweep --codec zenavif --plan svt_speed_dense \
  --sources /mnt/v/output/avifsvt-subsample-2026-09-01/sources \
  --q-grid 1,5,10,…,70,72,…,100 \
  --dry-run --emit-cells cells.jsonl --emit-cells-image-path basename \
  --output /path/run            # writes /path/run.plan.json
zenfleet-ctl declare-encodes --cells cells.jsonl --out manifest.json
```

Emits the canonical `{"cell","fp","plan"}` identity, so the run is
picker-dataset compatible. `--plan-budget`, `--compute-limit` and
`--max-deviations` all apply, and every drop is reported in `run.plan.json`.

**Adding an axis** = add a field to `zenavif::sweep::SweepAxes` + a token to
`Stratum::id()`/`config_from_cell_id` + a preset, exactly as the backend axis
did in §4.1. **Do not add a knob to `AVIF_KNOBS` and sweep it with
`--knob-grid`** unless you also register a resolver in `sweep/dedup.rs`;
otherwise the new axis is un-deduplicated by construction. If the new axis
changes bytes only in combination with another (an override that equals what
the preset already derives), the fingerprint will merge it automatically — that
is the point.

**Raw knob-grid (only when joining an existing run's identity space):**

```sh
zenmetrics sweep --codec zenavif --knob-grid '{"backend":["aom-rs"],"speed":[0,…,9]}' \
  --q-grid … --dry-run --emit-cells cells.jsonl --emit-cells-image-path basename \
  --output /path/run
```

`aom-rs` has no `EncoderConfig` representation (it is the zenav1-aom port driven
directly from `zenmetrics-cli`), so it is only declarable this way today.

### 4.4 zenfleet reconcile — measured

`zenfleet-ctl declare-encodes` on the flattened cells, then verified against the
original declaration **before** anything was replaced:

```
svt: old 9600  new 6496  new-is-subset-of-old=True  removed=3104
     jobs differing in ANY field vs the original declaration: 0
aom: old 9600  new 9280  new-is-subset-of-old=True  removed=320
     jobs differing in ANY field vs the original declaration: 0
```

A strict subset with byte-identical retained jobs ⇒ **no `JobId` moved** ⇒ every
completed cell is reused by construction. Joining the flattened cell set against
the live ledger:

| | count |
|---|--:|
| flattened declare items | 15,776 |
| DONE cells at reconcile time | 1,813 |
| **DONE cells reused by the flattened plan** | **1,417** |
| DONE cells outside it (alias-duplicates already burned) | 396 |
| remaining to encode | 14,359 |

The 396 are broken down `svt speed10: 143, svt speed8: 116, svt speed9: 73,
svt q100: 51, aom q100: 13` — i.e. exactly the two alias classes. **Their ledger
rows are untouched and their blobs remain in the store**; they are simply no
longer *declared*, so no worker will re-run or re-verify them.

Nothing was deleted and no worker was killed:

- `manifest.json` in both run prefixes was **copied to
  `manifest.pre-flatten-2026-09-01.json`** before being replaced.
- The six running workers (2 local, 2 r7900x, 2 tower) each had their manifest
  swapped in place, with the original preserved beside it
  (`/tmp/manifest.pre-flatten.json` in the containers,
  `*.pre-flatten-2026-09-01.bak` locally). `zenfleet_worker::run` re-reads the
  manifest **every pass** (`lib.rs:2794`), so each worker picks up the flattened
  set on its next pass, mid-flight, with no restart and no lost chunk.

---

## 5. CALLS MADE, AND WHAT WAS DELIBERATELY NOT DONE

### 5.1 No new knob axes

Per the 2026-09-01 mid-flight directive, the "additions per the jxl precedent"
clause is cancelled: a dedicated design-of-experiments lane owns axis expansion
and will declare through §4.3. Nothing new is registered here. The gaps this
audit *found* are recorded as observations only: SVT presets 2, 5 and 8 are
unreachable through the speed dial; the canonical zenavif `modes_full` plan
sweeps subsampling / bit-depth / colour-model / QM axes that this grid pins.

### 5.2 The sweep stays on the raw knob-map schema

Migrating to the canonical `{"cell","fp","plan"}` identity would change
`knob_tuple_json`, which is half the content-addressed `CellId`, which would
give **every one of the 1,813 completed cells a new `JobId`** and re-run them
all. The brief's constraint (reuse done cells) and the schema goal are in direct
conflict for a *running* sweep, so: the flattening is expressed by *which tuples
are declared*, not by changing the identity encoding, and the canonical-identity
path is built, tested and documented (§4.3) for the next wave, which starts
clean. Downstream consequence, stated plainly: this run's rows will need the
`omni_to_pareto.py` raw-map branch, and
`check_mandatory_coverage.py --codec zenavif` will fail on them.

### 5.3 `encode_fp` compute-dedup was NOT switched on

`EncodeDeclareItem::encode_fp` would have been the elegant lever — one encode
job per `(codec, source_sha, encode_fp)` group while *keeping* every per-knobset
row. It is declared but **populated by no emitter and consumed by no score-side
fan-out** (`grep -rn encode_fp scripts/` returns nothing). Its own doc comment
spells out the prerequisite: *"`build_score_spec.py` skips cells whose encode is
unindexed, so a deduped sweep without that fan-out fix would DROP the
non-representative rows."* Turning it on today would silently lose rows. Left
alone; registered here as the correct future mechanism once the fan-out carries
`encode_fp -> encode_sha`.

### 5.4 Which cells were merged away is a *labelling* judgement, and it is right here

Dropping speeds 8/9/10 loses no information for a **speed model**: their encoded
bytes AND their encode times are identical to speed 7 (§3.2). Keeping them would
actively harm a tuning model — it would teach it that four dial positions differ
when they do not, and triple the sample weight of the M9 point. The mapping fact
("dial 8, 9 and 10 all mean preset 9") belongs in this table, not in 2,880
encodes.

### 5.5 Not mine, but found

5 aom-rs cells are `failed / encoder_panic`, all on `1432.scale3000x4000.png` at
`speed 0` (q 10, 25, 40, and two more). That is a zenav1-aom port issue for the
KB-41 lane, not a sweep-declaration issue; the ledger rows carry the error class
and the cells remain declared, so a fixed port re-runs them additively.

---

## 6. STOPPING THE SWEEP CLEANLY (for the DOE lane)

The ledger is the durable artifact and **nothing below touches it.** Options, in
increasing order of finality:

**(a) Pause / drain — the canonical, reversible stop.** A `RunControl` object
`{"paused":bool,"drain":bool}` at `jobs/<run>/control.json` gates whether a
worker pulls new work; the ledger is never touched, so resuming continues
exactly where it left off (`scripts/jobsys/demo_pause_drain_r2.sh`).

```sh
source ~/tmp/_lan_env.sh
for r in avifsub-svt-enc-20260901 avifsub-aom-enc-20260901; do
  printf '{"paused":true,"drain":false}' > /tmp/control.json
  aws s3api put-object --endpoint-url "$ZEN_S3_ENDPOINT" --bucket zentrain \
      --key "jobs/$r/control.json" --body /tmp/control.json
done
```

⚠ **This only works for workers started with `--control-r2-key` / `ZEN_CONTROL_KEY`.**
The six workers running on 2026-09-01 were **not** — verified from their
command lines — so for *this* run (a) is inert until they are relaunched with it.
Any relaunch should set `ZEN_CONTROL_KEY=jobs/<run>/control.json`.

**(b) Shrink the declaration — what this retrofit did.** Re-declare a smaller
cell set and swap `manifest.json`; workers re-read it every pass and simply stop
claiming what is no longer declared. Fully reversible (the pre-flatten manifest
is preserved beside it), loses no ledger row and no in-flight chunk. To stop the
run entirely this way, declare an empty (or already-complete) cell set.

**(c) Stop the workers.** `docker stop avifsub-{svt,aom}-{r7900x,tower}` and
`kill <pid>` for the two local `zenfleet-worker` processes (**kill by recorded
PID — never `pkill -f`**, which self-matches the invoking shell). A worker
killed mid-pass loses only the cells it had claimed but not yet written; those
claims expire and the cells return to the gap. Restarting later resumes from the
ledger with no duplicate work, because claims are content-addressed leases.

**Never** delete `ledger/`, `blobs/` or `claims/` to "reset" a run — the ledger
IS the result, and the blobs are the encoded bytes the whole sweep exists to
produce.

Progress readout at any time:

```sh
zenfleet-ctl catalog --manifest <manifest.json> \
    --ledger s3://zentrain/jobs/<run>/ledger/ --r2-endpoint "$ZEN_S3_ENDPOINT"
```

---

## 7. WHAT THE ALREADY-ENCODED CELLS ARE GOOD FOR

Read on the flattened grid, the completed work is a **clean preset x q baseline
axis at default settings** — which is exactly the control arm a knob-tuning DOE
needs, and it is already partly paid for.

- **1,417 reused DONE cells**, every one at the backend's default envelope: svt-rs
  4:2:0 / 8-bit-auto / YCbCr, aom-rs ALLINTRA defaults. No QM, VAQ, trellis,
  subsampling or colour-model deviation anywhere in the run. So every cell is a
  *default-stratum* point: the `deviations = 0` row a factorial design measures
  its main effects against.
- **The q axis is dense and complete** (29 distinct quantizers spanning the full
  range, denser at both ends), so per-(image, preset) RD curves are usable as
  soon as a source's q sweep fills in — not only at the end of the run.
- **svt-rs is the more complete arm** (14.7 % vs 4.2 %) and its 7 effective
  presets are the whole compute ladder, so a preset-vs-q response surface at
  default settings is the first thing that becomes readable.
- **Caveat for a speed model**: `encode_ms` on these rows was measured under
  real fleet contention on three heterogeneous hosts (one uncapped, two capped
  and shared), and the ledger carries `worker`/`provider` but not thread count
  or load. Treat encode times as a *ranking* signal per host, not a calibrated
  cost; the launch doc's §5 host-meta join is the intended repair.
- **Do not** treat the 396 alias-duplicate rows as independent samples. They are
  byte-identical repeats of rows that are still in the grid; joined naively they
  will over-weight the M9 preset and the top quantizer.

---

## 8. ARTIFACTS

| what | where |
|---|---|
| flattened declare items | `~/tmp/permretrofit/flat/{svt,aom}_cells.jsonl` (6,496 + 9,280) |
| flatten audit manifests | `~/tmp/permretrofit/flat/{svt,aom}.plan.json` (every merge recorded) |
| flattened job manifests | `s3://zentrain/jobs/avifsub-{svt,aom}-enc-20260901/manifest.json` |
| **pre-flatten manifests (preserved)** | `s3://zentrain/jobs/avifsub-{svt,aom}-enc-20260901/manifest.pre-flatten-2026-09-01.json` |
| alias probe (encode evidence) | `~/tmp/permretrofit/aliasprobe/svt_probe.tsv` |
| zenavif change | `292582fb` on `zenavif main@origin` |

