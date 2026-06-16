# RFC: Unified vast.ai sweep worker for all zen codecs + Coefficient

**Status:** DRAFT — pure audit + architecture, no implementation in this commit.
**Author:** claude-vast-worker-rfc (delegated, 2026-05-26)
**Repos audited:** `jxl-encoder`, `zenmetrics`, `coefficient`, and per-codec
sweep needs across `zenjpeg`, `zenwebp`, `zenavif`, `zenpng`, `zengif`,
`zenflate`, `zenbitmaps`.
**Companion docs:**
- `~/work/zen/jxl-encoder/docs/RFC_DISPLAY_CONFIG_BACKFILL.md` — cross-codec
  multi-display sweep that this unification would enable.
- `~/work/zen/jxl-encoder/docs/RFC_CUSTOM_QUANT_WEIGHTS_RESEARCH.md` —
  decoder-stress sweep that needs cross-codec worker dispatch.
- `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/incident_killed_other_user_vastai_pod_2026-05-22.md`
  — the "tag-then-manage" safety rule any unified worker MUST encode.

---

## §1. Motivation

The vast.ai sweep stack has been re-implemented at least three times across
the imazen workspace:

1. **`jxl-encoder/scripts/zenjxl-tuning-sweep/`** — 5+ per-sweep launcher /
   janitor / finalize / merge / build-chunks Bash + Python tuples
   (W44-215 / W44-216 / W44-219 / W44-229 / W44-PHASE4-S1 / W44-PHASE4-S2),
   plus a Rust per-cell worker crate `zenjxl-tuning-runner` (2.7 KLOC across
   8 modules) and `Dockerfile.zenjxl-tuning-sweep.v2` (141 LOC).
   1,608 LOC across the canonical S2 trio + Dockerfile + worker.sh + onstart.sh.
2. **`zenmetrics/scripts/sweep/`** — 51 files spanning `onstart_*.sh` ×8
   variants, per-metric `*_chunk_worker.sh` ×4, generic `launch_backfill.sh`
   (382 LOC) and a Python `sweep_janitor.py` (178 LOC), plus the
   `crates/zenfleet-vastai/` Rust binary (4.8 KLOC across 15 modules), plus a
   collapsed canonical `Dockerfile.sweep.v26` (300 LOC). Already partially
   unified by the `zenfleet-vastai` binary's `worker` subcommand (omni and
   feature-backfill modes).
3. **`coefficient/`** — its own pattern: `src/bin/vastai_worker.rs` (470 LOC),
   `src/bin/vastai_dispatch.rs` (312 LOC), `src/bin/cloud_worker.rs`
   (879 LOC), `src/bin/do_worker.rs` (589 LOC), `src/cloud/vastai.rs`
   (867 LOC), `src/cloud/do_droplet.rs` (586 LOC), `src/cloud/batch.rs`
   (1092 LOC), `src/worker/mod.rs` (1706 LOC). 6.5 KLOC of cloud
   orchestration. JobSpec-based, dual-backend (vast.ai + DigitalOcean
   Droplets), with `BatchApi` abstraction in `cloud/batch_api.rs`.
   `scripts/vastai_create_workergroup.sh` (114 LOC) for the autoscaler path.
4. **Per-codec sweep needs** — `zenavif` has `examples/encode_sweep.rs`,
   `examples/predictor_sweep.rs`, `examples/phase2_oat.rs` and
   `scripts/cron_lhs_sweep.sh` (70 LOC, single-host cron). `zenjpeg`,
   `zenwebp`, `zenpng`, `zengif`, `zenflate`, `zenbitmaps` have no fleet
   infra at all. They WANT to do parameter sweeps (24-28 example binaries
   per codec, many parametric) but lack the lift to spin up a fleet.

The user's stated goals (verbatim from chunk brief):

> Cross-repo deps OK. jxl-encoder can depend on zenmetrics. Open to
> dependency inversion (worker depends on codecs, not the other way
> around).
>
> Universal codec support: solution must replace Coefficient's sweep infra
> AND support sweeps for every zen codec.
>
> Each codec basically patches itself and does a build of the vast worker.

The 2026-05-22 incident (destroyed another user's pod 37372377 while
cleaning up "idle" pods) made the **tag-then-manage rule** load-bearing for
any unified worker. The current state requires every operator to remember
to set `--label "claude-..."` in every per-sweep `launch_*.sh`. A unified
worker can encode this in the launch path with no per-sweep maintenance.

The current redundancy costs are:

- **Per-new-sweep cost in `jxl-encoder`:** copy 5 scripts (launch/janitor/
  finalize/merge/build-chunks), rename W44-PHASE4-S2 → W44-PHASE4-S3, edit
  ~20 grep/replace points. 4-6 hours of careful work where a single typo
  burns ~$10-$50 of vast.ai.
- **Per-codec onboarding cost (zenjpeg / zenwebp / zenavif sweeps):** today
  this requires either (a) forking the jxl-encoder script trio (4-6 hours
  + Dockerfile rework + the worker crate which currently imports
  `jxl_encoder` directly), or (b) writing a per-codec equivalent from
  scratch (zenavif's `cron_lhs_sweep.sh` chose option b, single-host
  cron only — no fleet). Cost to bring zenjpeg+zenwebp+zenavif+zenpng to
  jxl-encoder parity: ~3-5 days each.
- **Coefficient migration debt:** 6.5 KLOC of cloud orchestration in
  `src/cloud/`, `src/bin/vastai_*`, `src/bin/cloud_worker.rs`, plus a
  competing `Dockerfile` + `Dockerfile.gpu` chain. None of this shares
  code with the zenmetrics `zenfleet-vastai` despite solving the same
  problem.
- **Drift risk:** 3 stacks means 3 versions of the tag-then-manage rule
  enforcement, 3 image-rebuild flows, 3 onstart contracts. The 2026-05-22
  incident root cause was that one of these stacks omitted the label.

---

## §2. Audit results

| Repo / area | Sweep infra LOC | Worker LOC | Janitor | Launcher | Docker | Active sweeps 2026 | Dep direction |
|---|---|---|---|---|---|---|---|
| `jxl-encoder/scripts/zenjxl-tuning-sweep/` | 1,608 (per-sweep trio + worker.sh + onstart + Dockerfile.v2) | 2,716 (`zenjxl-tuning-runner` crate, 8 modules) | per-sweep bash, 214 LOC each | per-sweep bash, ~220 LOC each | `Dockerfile.zenjxl-tuning-sweep.v2` 141 LOC | W44-215..W44-229, W44-PHASE4-S1, W44-PHASE4-S2 (6+ sweeps in 2 weeks) | worker → `jxl-encoder` lib via path dep |
| `zenmetrics/scripts/sweep/` | 51 files: 8× onstart, 4× chunk-worker shells, generic `launch_backfill.sh`, `sweep_janitor.py`, `destroy_all.sh`, per-metric `*_backfill/` dirs | 4,781 (`zenfleet-vastai` binary, 15 modules, with `worker` subcommand handling omni / feature-backfill / source-features modes) | `sweep_janitor.py` 178 LOC + `fleet_status.sh` | `launch_backfill.sh` 382 LOC + `launch_single_instance.sh` | `Dockerfile.sweep.v26` 300 LOC (collapsed single-file, layer-aware) | omni / feature-backfill / source-features (currently running cvvdp-fork Phase 8 + zensim-fork sweeps) | `zenfleet-vastai` → `zenmetrics-cli` (sibling crate), `zenmetrics-cli` → all codec crates |
| `coefficient/src/bin/` + `coefficient/src/cloud/` | ~6,500 (`vastai_worker.rs`, `vastai_dispatch.rs`, `cloud_worker.rs`, `do_worker.rs`, `cloud/{batch,vastai,do_droplet,jobspec,mock_batch,quota,config}.rs`) | `cloud_worker.rs` 879 + `worker/mod.rs` 1,706 | implicit via `BatchApi` trait (`cloud/batch.rs`) | `vastai_dispatch.rs` 312 LOC + `vastai_create_workergroup.sh` 114 LOC | `Dockerfile` + `Dockerfile.gpu` (separate chain) | JobSpec-backed (`RescoreTsv`, `SweepCodec`, `TrainPicker`); not currently in active use vs zenmetrics fleet | `coefficient_*` bins → `coefficient` lib (self-contained) |
| `zenavif/scripts/` | 70 (`cron_lhs_sweep.sh`, single-host cron only) | 0 dedicated; uses `examples/predictor_sweep.rs` direct | 0 | cron | `Dockerfile.references` (NOT a sweep image) | nightly LHS sweep (rav1e tuples) | self-contained example, no fleet |
| `zenjpeg/scripts/` | 0 fleet | 0 | 0 | 0 | 0 | none in 2026 | N/A |
| `zenwebp/examples/` | 16 examples (`heap_strip_pipeline.rs`, `lossless_corpus.rs`, `lossless_rt_check.rs`) but no fleet | 0 | 0 | 0 | 0 | none in 2026 | N/A |
| `zenpng/examples/` | 25 examples, no fleet | 0 | 0 | 0 | 0 | none in 2026 | N/A |
| `zengif/examples/` | 10 examples, no fleet | 0 | 0 | 0 | 0 | none in 2026 | N/A |
| `zenflate/examples/` | 8 examples + 1 minimal `Dockerfile` (profiling, not sweep) | 0 | 0 | 0 | 0 | none in 2026 | N/A |
| `zenbitmaps/` | 0 | 0 | 0 | 0 | 0 | none in 2026 | N/A |

### Common patterns (what every sweep needs)

Every observed sweep, whether jxl-encoder, zenmetrics, or coefficient,
implements these 11 concerns:

1. **Chunk builder** — produces `chunks.jsonl` (NDJSON, one cell per line).
   Cell schema is sweep-specific; all observed cells carry `(image_path,
   codec_params, output_key)`.
2. **Corpus stager** — pre-flight check that every `image_path` in the
   manifest exists in R2 (post-W44-PHASE4-S1h fix; the chunk-builder did
   NOT stage corpus bytes pre-fix).
3. **Worker binary** — reads one cell, encodes, scores, writes a sidecar
   parquet/TSV.
4. **Encoder dispatch** — selects codec + parameters per cell. Cross-codec
   sweeps need a uniform encode interface.
5. **Metric scoring** — usually GPU (cvvdp / butteraugli / ssim2 /
   zensim), sometimes CPU. The "+1006% wall delegation overhead" noted in
   the zensim-gpu Phase 1 memo is real and motivates per-process metric
   init.
6. **R2 upload** — content-addressed `artifacts/<ext>/<sha256>.<ext>` plus
   per-chunk sidecar. The 2026-05-24 SHIP rule (CLAUDE.md §4) makes this
   non-optional for sweeps costing >$0.10.
7. **Chunk claim + rescue** — atomic claim mechanism (rename to
   `in-progress/<id>.json`); rescue stale in-progress claims older than N
   minutes.
8. **Lifecycle: launch** — `vastai create instance` × N boxes with a
   `--label "claude-<sweep>-<chunk>"` prefix (post-2026-05-22 incident).
9. **Lifecycle: janitor** — periodically scan instances by label, destroy
   idle pods (CPU + GPU util < threshold for grace period). The W44-229j
   fix raised this to: "workers are CPU-bound with brief GPU spikes; don't
   destroy on GPU idle alone."
10. **Lifecycle: destroy-all** — at sweep end, destroy every instance
    matching the label prefix.
11. **Merge + finalize** — concatenate sidecars to one merged parquet,
    sha256-verify, mirror to Tower NAS.

### Codec-specific concerns

The variable surface across sweeps is narrower than the boilerplate suggests:

- **Which encoder API to call.** `zenjxl-tuning-runner::main.rs` calls
  `jxl_encoder::LossyConfig::new(distance).encode(&request)`, a 4-line
  call. zenjpeg / zenwebp / zenavif each have analogous 4-10 line entry
  points (`zenjpeg::JpegEncoder::encode_rgba8`,
  `zenwebp::Encoder::with_quality(q).encode`, etc.).
- **Which parameter struct to sweep.** jxl-encoder has `RuntimeTuning`
  (postcard-serialized in the `params_blob` column). Other codecs need
  their own opaque blob.
- **Which metric set is relevant.** All current sweeps use the same
  butteraugli + ssim2 + cvvdp + zensim set via `zenmetrics-cli`. Future
  iwssim / DSSIM / PSNR sweeps fit the same interface.
- **What corpus access pattern.** Most sweeps fetch a single source image
  per cell from `s3://zen-corpus/<image>.png`. Picker-training sweeps
  also fetch a feature parquet sidecar. Cross-codec multi-display sweeps
  (`RFC_DISPLAY_CONFIG_BACKFILL`) need both a source image AND a display
  config blob per cell.

The "patch-then-build" insight: the only thing that varies across
sweeps WITHIN one codec is the in-flight encoder source (we patch with a
`[patch.crates-io]` to a feature branch and rebuild the worker). The
thing that varies BETWEEN codecs is the encode-call dispatch. Both fit
naturally into a single Rust crate with cargo features per codec.

### Where today's stacks already converged

The `zenfleet-vastai` Rust binary in zenmetrics is the closest thing to a
unified worker today. It has:

- `Status` subcommand (one-shot report by label prefix)
- `Destroy` subcommand (defensive, label-prefix matched)
- `Watch` subcommand (auto-destroy at target sidecar count)
- `SelfDestroy` subcommand (worker's EXIT trap calls this)
- `Worker` subcommand (chunk-claim + R2 IO + parquet IO + inline
  CubeCL-init scoring loop)

What it does NOT have today:

- `Launch` subcommand (still `launch_backfill.sh` 382 LOC)
- `Janitor` subcommand (still `sweep_janitor.py` 178 LOC; the
  jxl-encoder janitors are per-sweep bash variants with the W44-229j
  fix grafted in)
- A `SweepTarget` trait that lets non-zenmetrics codecs plug in
- A patch-then-build flow contract for per-sweep encoder pinning

These are the gaps this RFC proposes to close.

---

## §3. Architecture

### §3.1 Crate placement: `zenmetrics/crates/vast-worker/`

**Recommended home: a new `vast-worker` crate inside the existing
`zenmetrics` workspace, alongside `zenfleet-vastai`.** Justification, weighing
the three candidate placements:

- **(a) New repo `imazen/zen-vast-worker`** — clean dependency story
  (worker depends on every codec; no codec depends on the worker). BUT:
  every code-push to a codec triggers a worker rebuild even when the
  codec change doesn't touch sweep infra; CI complexity multiplies
  (worker has to track N codecs' versions); discovery cost for future
  contributors who'll look for vast.ai infra in zenmetrics first.
- **(b) `zenmetrics/crates/vast-worker/` alongside `zenfleet-vastai`** —
  the worker depends on `zenfleet-vastai` directly (already a Rust binary;
  share lifecycle subcommands), depends on `zenmetrics-cli` directly
  (already lives in this workspace; share scoring backend), and adds
  codec crates as path-or-git deps gated behind cargo features. The
  v26 Dockerfile already bakes both `zenfleet-vastai` and `zenmetrics`
  binaries; adding `vast-worker` is one more `COPY --chmod=0755` line.
  Operator muscle memory ("the sweep stack lives in zenmetrics") is
  preserved.
- **(c) `coefficient/crates/vast-worker/`** — coefficient is the
  highest-LOC cloud orchestration stack today, but it's also the most
  divergent (DigitalOcean Droplet path, JobSpec mechanism, BatchApi
  trait). Placing the worker here would require coefficient to absorb
  the zenfleet-vastai binary too, which is a much bigger refactor.

Option (b) is the lowest-friction. The cross-repo dep direction works:
zenmetrics already imports every codec it scores (it's a metric library
that operates on encoded images). Adding `vast-worker` that also imports
every codec for sweep-time encoding does not change the dependency
topology of the workspace.

Concretely:

```
zenmetrics/
  crates/
    zenfleet-vastai/       — lifecycle (launch / janitor / status /
                          destroy / watch / self-destroy)
    vast-worker/        — NEW: per-cell encode + score, dispatched
                          via cargo features per codec
    zenmetrics-cli/    — score backend (already exists)
    zenmetrics-api/     — shared types (already exists)
    zenmetrics-corpus/  — corpus access (already exists)
  Cargo.toml            — workspace member registration
  Dockerfile.sweep.v26  — already bakes zenfleet-vastai + zenmetrics
                          binaries; v27 will also bake vast-worker
```

### §3.2 Codec dispatch: cargo features + trait impls (both)

The `vast-worker` crate uses a hybrid dispatch:

1. **Trait** — codecs implement `SweepTarget`. The trait makes
   per-codec sweep work a fixed-size contract.
2. **Cargo features** — each codec is opt-in via
   `--features <codec_name>` at build time. The patch-then-build flow
   pins each per-sweep image to exactly one codec + version.

This lets sweep ops do:

```bash
# Per-sweep image (only zenjxl baked in):
cargo build --release -p vast-worker --features jxl-encoder --no-default-features

# Cross-codec sweep image (multiple codecs baked):
cargo build --release -p vast-worker --features "jxl-encoder,zenwebp,zenavif" --no-default-features
```

The CLI dispatches on a `--codec <name>` flag and the matching cargo
feature being compiled in. If a sweep asks for a codec not compiled
in, the worker fails loudly at process start (NOT at first cell, NOT
silently — per the v14 Dockerfile lesson in CLAUDE.md).

### §3.3 `SweepTarget` trait sketch

```rust
/// One codec's contract with the unified sweep worker. Every zen
/// codec that wants fleet sweeps implements this in its own crate
/// or in a `vast-worker-<codec>` subcrate that wraps it.
pub trait SweepTarget {
    /// CLI-facing codec identifier, e.g. `"jxl-encoder"`, `"zenwebp"`.
    /// MUST match the cargo feature name 1:1.
    fn codec_id() -> &'static str;

    /// Sweep-specific parameter blob (RuntimeTuning for jxl-encoder,
    /// WebpTuning for zenwebp, etc.). Postcard-serialized in the
    /// per-cell JSON spec under `params_blob` / `params_blob_uri`.
    type Params: serde::de::DeserializeOwned + Send + 'static;

    /// Encode one cell. Receives the source linear-sRGB image, the
    /// per-cell params, and a small context with corpus / stage paths.
    /// Returns encoded bytes + per-cell metadata (encode_ms, peak_rss,
    /// codec-specific counters).
    fn encode_one(
        ctx: &CellContext<'_>,
        params: &Self::Params,
    ) -> Result<EncodeOutcome, SweepError>;

    /// Codec-specific result columns to attach to the per-cell parquet
    /// row (in addition to the standard `image_path / codec / params /
    /// encoded_sha256 / butteraugli / ssim2 / cvvdp / zensim / encode_ms`
    /// columns). Empty vec for "no extras".
    fn extra_parquet_columns(out: &EncodeOutcome) -> Vec<(&'static str, ParquetValue)>;
}
```

### §3.4 Per-sweep config schema

One TOML file per sweep, named `<sweep_id>.toml`, lives in
`zenmetrics/sweeps/<sweep_id>/sweep.toml`. The launcher reads this,
generates the chunks.jsonl, builds the image, and creates the fleet:

```toml
# sweeps/w44-phase4-s3-recon-2026-06-01/sweep.toml
[sweep]
id            = "w44-phase4-s3-recon-2026-06-01"
codec         = "jxl-encoder"                 # selects cargo feature
n_boxes       = 30
max_dph       = 0.30
boxes_min_ram = 16
image_tag     = "ghcr.io/imazen/zen-vast-worker:v0-jxl-encoder-<git-sha>"

[patches]
# Optional [patch.crates-io] overrides applied to the worker build.
# Each entry pins one crate to a git ref or path. Worker image is
# rebuilt per-sweep when this section is non-empty.
"jxl-encoder" = { git = "https://github.com/imazen/jxl-encoder", branch = "w44-phase4-s3-fix" }

[corpus]
source       = "s3://zen-corpus/cid22-512/"
images       = ["1418519.png", "1025469.png", "1531677.png"]
# Pre-flight enforced by launcher (post-W44-PHASE4-S1h fix).

[params]
# Path to a chunk-builder script that emits NDJSON cells.
builder        = "scripts/build_w44_phase4_s3_chunks.py"
builder_args   = ["--target-cells", "30000"]

[metrics]
butteraugli   = "gpu"        # gpu | cpu | skip
ssim2         = "gpu"
cvvdp         = "gpu"
zensim        = "skip"        # not in this sweep
multimetric   = true          # write all 5 norms of butteraugli

[artifacts]
save_encoded   = true         # CLAUDE.md §4 mandate (default on)
save_diffmap   = true
diffmap_format = "png16"

[fleet]
janitor_grace_min       = 8
janitor_cells_min_floor = 100
janitor_idle_cpu_pct    = 5
auto_destroy_at_cells   = 28000   # 95% target → trigger destroy-all
```

### §3.5 Patch-then-build flow

The operator workflow per sweep, from cold start to merged parquet, is
seven commands:

```bash
# 1. Write sweep.toml (TOML editor, or copy a previous sweep's file).
$EDITOR zenmetrics/sweeps/w44-phase4-s3-recon-2026-06-01/sweep.toml

# 2. Build chunk manifest (this calls the python builder named in
#    sweep.toml; result is uploaded to R2 in step 4).
vast-worker chunks build --sweep w44-phase4-s3-recon-2026-06-01

# 3. Build + push the per-sweep image. Applies [patches], cargo
#    builds with the right --features, COPYs binary into v26+1
#    Dockerfile, pushes to ghcr.io with a stable tag.
vast-worker image build --sweep w44-phase4-s3-recon-2026-06-01 --push

# 4. Stage corpus + chunks + params to R2. Pre-flight verifies every
#    image_path is in s3://zen-corpus/ (W44-PHASE4-S1h rule).
vast-worker chunks stage --sweep w44-phase4-s3-recon-2026-06-01

# 5. Launch the fleet (mandatory --label "claude-<sweep>-..." per
#    2026-05-22 incident; baked into vast-worker launch, not optional).
vast-worker launch --sweep w44-phase4-s3-recon-2026-06-01

# 6. Watch (auto-destroys at target_cells; tag-then-manage destroy
#    of idle pods). Idempotent; survives reboots of the operator's
#    laptop.
vast-worker watch --sweep w44-phase4-s3-recon-2026-06-01

# 7. Finalize: merge per-chunk sidecars to one merged.parquet, sync
#    to Tower NAS, sha256-verify, archive.
vast-worker finalize --sweep w44-phase4-s3-recon-2026-06-01
```

Every subcommand is implemented in Rust (currently bash). The Rust
implementation gives:

- Compile-time validation of sweep.toml (today: a typo in a per-sweep
  bash launcher costs $10+ in cold-start time).
- Single-place implementation of the 2026-05-22 label rule.
- Single-place implementation of the W44-229j idle-detection fix
  (CPU + GPU floor, not GPU alone).
- Single-place implementation of the CLAUDE.md §4 artifact-persistence
  mandate (verified at sweep launch, not discovered after $30 burn).

### §3.6 Integration with `zenfleet-vastai`

The `vast-worker` CLI wraps `zenfleet-vastai` for the lifecycle subcommands
(`launch`, `janitor`, `watch`, `destroy-all`). `zenfleet-vastai` already
handles label-scoped destroy and watch; the new code is in `launch`
(today: bash) and `janitor` (today: bash + python). These move into
`zenfleet-vastai` subcommands so `vast-worker` shells to one well-tested
binary instead of three loosely-coupled scripts.

The worker-side chunk loop stays in `zenfleet-vastai worker` (already
exists) — `vast-worker` calls `zenfleet-vastai worker --codec <X>` inside
the docker container's `ENTRYPOINT`. The `--codec` flag dispatches to
the right `SweepTarget` impl at runtime.

This means **`vast-worker` is mostly a build-system + per-sweep
config-loader + lifecycle wrapper**, NOT a re-implementation of the
worker loop. The worker loop stays in `zenfleet-vastai`. The codec-specific
encode dispatch moves into a new `SweepTarget` trait that's implemented
inside each codec crate (or inside `vast-worker` as a thin adapter).

---

## §4. Migration plan

Each phase below has concrete acceptance criteria + an effort estimate.
No phase ships without the prior phase being green.

### Phase M1: Generalize `zenfleet-vastai` with `launch` + `janitor` subcommands

**Scope:** Port `zenmetrics/scripts/sweep/launch_backfill.sh` (382 LOC bash)
and `zenmetrics/scripts/sweep/sweep_janitor.py` (178 LOC python) into new
`zenfleet-vastai launch` and `zenfleet-vastai janitor` subcommands. Bake in the
2026-05-22 tag-then-manage rule (require `--label` prefix). Bake in the
W44-229j idle-detection fix.

**Effort:** XS-S (1-3 days). The existing `Destroy` / `Watch` /
`SelfDestroy` subcommands already establish the pattern. Most of the
work is porting the create-instance loop in `launch_backfill.sh`.

**Acceptance:**
- `zenfleet-vastai launch --sweep <id> --boxes N --image <ghcr-tag>` is
  byte-identical (in resulting vast.ai state) to the bash version on
  a smoke test (1 box, 1 chunk).
- `zenfleet-vastai janitor --sweep <id> --once` matches `sweep_janitor.py`
  reaping decisions on a recorded TSV input (10 worker stats files,
  golden destruction set).
- Rejects launch invocations with missing `--label` prefix (today: bash
  scripts silently launch un-labeled, which caused the 2026-05-22
  incident).

**Blockers:** None. Self-contained inside zenmetrics workspace.

### Phase M2: `vast-worker` crate skeleton + `SweepTarget` impl for `jxl-encoder`

**Scope:** Create `zenmetrics/crates/vast-worker/` with the trait, the
CLI subcommands (`chunks build` / `chunks stage` / `image build` /
`launch` / `watch` / `finalize`), and ONE `SweepTarget` impl
(`jxl-encoder`). Translate `zenjxl-tuning-runner` to the new trait
interface — the existing 2.7 KLOC binary becomes a `vast-worker-jxl`
adapter (~200 LOC after the trait absorbs the common parquet writer +
metrics shell-out).

**Effort:** M (5-8 days). The trait surface design is the hard part.
Once the trait is right, the jxl-encoder impl is mechanical.

**Acceptance:**
- `vast-worker chunks build --sweep w44-PHASE4-S2-replay` produces a
  byte-identical chunks.jsonl to the existing
  `build_w44_phase4_s2_chunks.py` output for a fixed seed.
- `vast-worker image build --sweep w44-PHASE4-S2-replay --push` produces
  a docker image that decodes the same per-cell parquet schema as
  `zenjxl-tuning-runner` does on a 1-cell smoke test.
- `vast-worker launch --sweep ... --boxes 1` + chunk drain matches
  W44-PHASE4-S1g artifact-persistence (encoded bytes + diffmap landing
  in R2 under content-addressed keys per CLAUDE.md §4).
- All `cargo test --workspace` PASS in `zenmetrics`.

**Blockers:** Need to settle the trait surface (this RFC's §5 lists
open questions). Once settled, no blockers.

### Phase M3: Replace `jxl-encoder` per-sweep bash with `vast-worker` invocations

**Scope:** Retire the per-sweep launcher / janitor / finalize trios in
`jxl-encoder/scripts/zenjxl-tuning-sweep/`. Replace with one
`sweeps/<sweep-id>.toml` per sweep, invoked via `vast-worker`. Retain
the chunk-builder Python scripts (they're sweep-specific; they live
under `sweeps/<sweep-id>/build_chunks.py` and are referenced by
`sweep.toml`).

**Effort:** S (2-4 days). Each per-sweep trio becomes one TOML file.
Validate by re-running W44-PHASE4-S1's exact cells via the new path.

**Acceptance:**
- Diff between W44-PHASE4-S1 sidecars (recorded) and a replay via
  `vast-worker` is bit-identical at the parquet column level (encoded
  bytes will differ by content-addressed sha — that's expected and OK).
- Total LOC in `jxl-encoder/scripts/zenjxl-tuning-sweep/` drops from
  current ~3,000 LOC of bash/python to <500 LOC of per-sweep TOML +
  chunk-builders.
- `zenjxl-tuning-runner` crate is retired (its work absorbed into
  `vast-worker-jxl` impl).

**Blockers:** Phase M2 complete. No code dep on coefficient.

### Phase M4: Add per-codec impls (zenjpeg, zenwebp, zenavif, zenpng, etc.)

**Scope:** One `SweepTarget` impl per codec. Each impl is XS (50-200 LOC)
once the trait is right. Choose codec-specific param structs to mirror
each codec's existing parametric API. Each impl ships with a smoke-test
that encodes one cell end-to-end.

**Effort per codec:** XS (1-2 days each). Total for 6 codecs (zenjpeg /
zenwebp / zenavif / zenpng / zengif / zenflate): 1-2 weeks.

**Acceptance per codec:**
- One smoke `vast-worker chunks build --sweep <codec>-smoke` produces a
  3-cell chunks.jsonl that encodes successfully on a 1-box launch.
- Per-cell parquet schema validates (no required column null).
- Codec-specific extra columns appear in the schema per
  `SweepTarget::extra_parquet_columns`.

**Blockers:** Phase M2 complete. Per-codec authors can work in parallel.

### Phase M5: Migrate Coefficient's sweep infra

**Scope:** This is the largest unknown. Coefficient has ~6.5 KLOC of cloud
orchestration spanning two cloud backends (vast.ai + DigitalOcean Droplets)
and a `BatchApi` trait abstraction in `cloud/batch_api.rs`. The right path
depends on whether Coefficient wants to:

- **(M5a) Replace `vastai_worker.rs` + `vastai_dispatch.rs` with
  `vast-worker` invocations** — preserves Coefficient's DO Droplet path,
  preserves the `BatchApi` trait, only swaps vastai backend. Effort: M
  (1 week). Requires `vast-worker` to expose subcommand-callable Rust API
  (today this RFC assumes CLI invocation; M5a needs a library mode).
- **(M5b) Migrate Coefficient to JobSpec-via-vast-worker AND port DO
  Droplet path into `zenfleet-vastai`** — unifies vastai + DO under one
  fleet binary. Effort: L (2-3 weeks). Bigger refactor but eliminates
  the parallel `do_droplet.rs` (586 LOC) + `do_worker.rs` (589 LOC)
  + `cloud_worker.rs` (879 LOC) stacks.
- **(M5c) Coefficient keeps its own JobSpec mechanism; vast-worker
  becomes the codec sweep tool only; Coefficient remains the
  picker-training / TSV-rescore tool** — minimal change. Effort: XS
  (0.5 day, just a doc note). Accepts the duplicated vast.ai surface
  as the cost of Coefficient's pipeline-agnostic JobSpec abstraction.

**Recommendation:** M5c is the lowest-friction; M5a is the right
medium-term goal. M5b is interesting but out of scope for the initial
unification. Defer M5a/b decision to a follow-on RFC after M1-M4 land.

**Acceptance:** Coefficient's `vastai_create_workergroup.sh` either
continues to work unchanged (M5c) or is replaced by `vast-worker launch
--workergroup` (M5a/b).

**Blockers:** Phase M2 complete; M5 decision needs user input.

### Phase M6: Retire old per-sweep bash scripts

**Scope:** Once M3 is shipped and W44-PHASE4-S3+ runs through `vast-worker`,
delete the per-sweep launcher / janitor / finalize / merge trios under
`jxl-encoder/scripts/zenjxl-tuning-sweep/`. Retain Dockerfile.v2 in
git history only (replaced by the v27 zenmetrics-side image).

**Effort:** XS (0.5 day). Pure deletion + CLAUDE.md doc update.

**Acceptance:**
- `jxl-encoder/scripts/zenjxl-tuning-sweep/` contains only chunk-builder
  Python and the worker source crate (retained until M3 absorbs it).
- All references in jxl-encoder CLAUDE.md / docs/ point at zenmetrics
  vast-worker.

**Blockers:** M3 stable for at least one production sweep cycle.

---

## §5. Open questions

These six decisions need user input before implementation starts.

### Q1. Crate placement: confirm `zenmetrics/crates/vast-worker/`?

This RFC recommends (b) — `zenmetrics/crates/vast-worker/` alongside
`zenfleet-vastai`. Alternative: (a) new repo `imazen/zen-vast-worker`.

**Need user signal on:** is the discovery cost of "sweep infra lives
in zenmetrics" preferable to the dependency-cycle cost of "every codec
push triggers a worker rebuild"? Recommended (b) is preferred but
non-blocking.

### Q2. Dependency direction: worker depends on codecs, or codecs depend on worker?

This RFC recommends **worker depends on codecs** via cargo features +
the `SweepTarget` trait implemented inside each codec crate (or in a
thin `vast-worker-<codec>` adapter in the zenmetrics workspace).

Alternative: each codec depends on a tiny `vast-worker-api` crate that
defines the `SweepTarget` trait, and the codec exports its impl;
`vast-worker` itself depends on no codecs and dispatches via dynamic
plugin discovery at boot. More dependency-pure but adds plugin
discovery as a new failure mode.

**Need user signal on:** preference between (a) cargo features
(static, fail-loud at build time) and (b) plugin discovery (dynamic,
fail-loud at boot). Recommended (a).

### Q3. Sweep config format: TOML, YAML, or Rust?

This RFC sketches TOML per-sweep config files. Alternatives:
- **YAML** — matches GitHub Actions / kubernetes conventions; less
  Rust-native.
- **Rust** — a `sweep.rs` file per sweep, compiled with the worker;
  catches typos at build time. Slightly heavier to author.

**Need user signal on:** TOML vs YAML vs Rust. Recommended TOML for
operator-edit friendliness; the launcher validates against a Rust
schema at parse time.

### Q4. Image-build cadence: per-sweep vs daily-batch?

Per-sweep image builds (recommended) mean every new sweep triggers a
`cargo build --release` + `docker push`. This is ~5-15 minutes per
sweep cold, ~30 seconds warm (layer cache). Across the recent W44-215
through W44-PHASE4-S2 arc, that's 6+ image builds.

Alternative: daily-batch image builds, where per-sweep changes apply via
runtime `[patch.crates-io]` overrides without rebuild. Bigger image
(must bake all in-flight branches) and more complex `vast-worker` runtime.

**Need user signal on:** per-sweep image build (recommended; matches
the CLAUDE.md "BAKE EVERYTHING" rule for vast.ai) vs daily-batch.

### Q5. Coefficient migration: M5a, M5b, or M5c?

Per §4 Phase M5:
- (M5a) Replace Coefficient's vastai path with vast-worker, keep DO
  Droplet path. ~1 week.
- (M5b) Migrate everything to vast-worker, port DO into zenfleet-vastai.
  ~2-3 weeks.
- (M5c) Leave Coefficient unchanged; vast-worker is codec sweeps only.
  0.5 day doc note.

**Need user signal on:** which path. Recommended start with M5c
(minimal), revisit after M1-M4 land for a year of operational data.

### Q6. Migration sequencing: opt-in or default-on for jxl-encoder?

Phase M3 retires the W44-PHASE4-S2 launcher trio. Two options:
- **Opt-in:** W44-PHASE4-S3 uses `vast-worker`; older trios stay until
  removed in Phase M6. Slower retirement, parallel paths during the
  transition.
- **Default-on:** W44-PHASE4-S3 is the first sweep that REQUIRES
  `vast-worker` (no fallback to the old bash trio). Faster retirement,
  risk-concentrated.

**Need user signal on:** opt-in (recommended; allows comparing
W44-PHASE4-S2 bash-trio numbers vs W44-PHASE4-S3 vast-worker numbers
on a fresh sweep) vs default-on.

---

## §6. Cross-references

**Existing infra:**

- `jxl-encoder/scripts/zenjxl-tuning-sweep/` — current per-sweep bash
  trios (W44-215 through W44-PHASE4-S2), the `zenjxl-tuning-runner`
  Rust worker crate, and `Dockerfile.zenjxl-tuning-sweep.v2`.
- `zenmetrics/scripts/sweep/` — 51 files: per-metric onstart variants,
  per-metric chunk-worker shells, `launch_backfill.sh` (canonical
  launcher today), `sweep_janitor.py` (canonical janitor today),
  `destroy_all.sh`.
- `zenmetrics/crates/zenfleet-vastai/` — Rust binary with Status / Destroy
  / Watch / SelfDestroy / Worker subcommands. The skeleton this RFC
  extends.
- `zenmetrics/Dockerfile.sweep.v26` — collapsed single-file canonical
  sweep image; layer-aware; bakes `zenmetrics` + `zenfleet-vastai`
  binaries.
- `zenmetrics/scripts/sweep/README.md` — operator guide for the current
  zenmetrics sweep stack; the unified vast-worker docs will subsume it.
- `coefficient/src/bin/vastai_worker.rs` + `coefficient/src/cloud/` —
  Coefficient's parallel cloud orchestration stack. §4 Phase M5
  proposes migration paths.
- `coefficient/scripts/vastai_create_workergroup.sh` — Coefficient's
  workergroup-based launcher (different mechanism from
  `launch_backfill.sh`'s per-instance loop).
- `zenavif/scripts/cron_lhs_sweep.sh` — single-host cron sweep, the
  only non-jxl-encoder codec with active sweep infra today.

**Related RFCs:**

- `~/work/zen/jxl-encoder/docs/RFC_DISPLAY_CONFIG_BACKFILL.md` — the
  cross-codec multi-display sweep that the unified worker enables.
  Today it's blocked because adding "display config" as a per-cell
  dimension to the W44-PHASE4-S2 chunk-builder requires forking
  another script trio.
- `~/work/zen/jxl-encoder/docs/RFC_CUSTOM_QUANT_WEIGHTS_RESEARCH.md`
  — decoder-stress sweep that requires running both encode (jxl-encoder)
  AND decode (jxl-rs / zenjxl-decoder) per cell, in the same worker
  process. The unified worker's `SweepTarget` trait accommodates
  multi-pass cells naturally.
- `~/work/zen/jxl-encoder/docs/RFC_CVVDP_FORK.md` — the cvvdp-fork
  sweep arc demonstrated the cost of per-sweep bash trios across
  Phase 6 (4-backend), Phase 8c+8d (cumulative stack), Phase 8f
  (full-corpus validation). Each phase required script-trio surgery;
  the unified worker would have been one new `sweeps/<phase>.toml`.

**Safety rules the unified worker MUST encode:**

- `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/incident_killed_other_user_vastai_pod_2026-05-22.md`
  — tag-then-manage rule (`--label "claude-<sweep>-<chunk>"` MANDATORY).
- W44-229j fix in `jxl-encoder/scripts/zenjxl-tuning-sweep/janitor_w44_229.sh`
  — idle detection must use CPU + GPU floor, NOT GPU floor alone
  (workers are CPU-bound with brief GPU spikes).
- CLAUDE.md §4 (ML Data Pipeline Discipline) — encoded bytes + diffmaps
  + multi-metric variants MUST persist on any sweep >$0.10. The
  unified worker encodes this as a config-default-on with explicit
  opt-out per sweep.
- W44-PHASE4-S1h fix in
  `jxl-encoder/scripts/zenjxl-tuning-sweep/launch_w44_phase4_s1_fleet.sh`
  — pre-flight check that every `image_path` in chunks.jsonl exists in
  R2 corpus bucket BEFORE launching the fleet. Cost: ~1s for 30 images;
  prevented the $30 / 4-hour S1 coverage loss.

**Audit reference (read-only, all cited LOC counts):**

- `jxl-encoder/scripts/zenjxl-tuning-sweep/worker.sh` (229 LOC)
- `jxl-encoder/scripts/zenjxl-tuning-sweep/onstart.sh` (215 LOC)
- `jxl-encoder/scripts/zenjxl-tuning-sweep/launch_w44_phase4_s2_fleet.sh` (218 LOC)
- `jxl-encoder/scripts/zenjxl-tuning-sweep/janitor_w44_phase4_s2.sh` (214 LOC)
- `jxl-encoder/scripts/zenjxl-tuning-sweep/finalize_w44_phase4_s2.sh` (98 LOC)
- `jxl-encoder/scripts/zenjxl-tuning-sweep/build_w44_phase4_s2_chunks.py` (493 LOC)
- `jxl-encoder/scripts/zenjxl-tuning-sweep/Dockerfile.zenjxl-tuning-sweep.v2` (141 LOC)
- `jxl-encoder/zenjxl-tuning-runner/src/` (2,716 LOC across 8 modules)
- `zenmetrics/scripts/sweep/launch_backfill.sh` (382 LOC)
- `zenmetrics/scripts/sweep/sweep_janitor.py` (178 LOC)
- `zenmetrics/scripts/sweep/onstart_v3.sh` (401 LOC, deprecated but present)
- `zenmetrics/scripts/sweep/dispatch.sh` (78 LOC)
- `zenmetrics/scripts/sweep/destroy_all.sh` (22 LOC)
- `zenmetrics/Dockerfile.sweep.v26` (300 LOC)
- `zenmetrics/crates/zenfleet-vastai/src/` (4,781 LOC across 15 modules)
- `coefficient/src/bin/vastai_worker.rs` (470 LOC)
- `coefficient/src/bin/vastai_dispatch.rs` (312 LOC)
- `coefficient/src/bin/cloud_worker.rs` (879 LOC)
- `coefficient/src/bin/do_worker.rs` (589 LOC)
- `coefficient/src/bin/do_cli.rs` (460 LOC)
- `coefficient/src/bin/batch_cli.rs` (1,239 LOC)
- `coefficient/src/cloud/batch.rs` (1,092 LOC)
- `coefficient/src/cloud/do_droplet.rs` (586 LOC)
- `coefficient/src/cloud/vastai.rs` (867 LOC)
- `coefficient/src/cloud/jobspec.rs` (221 LOC)
- `coefficient/src/cloud/mock_batch.rs` (650 LOC)
- `coefficient/src/cloud/quota.rs` (313 LOC)
- `coefficient/src/cloud/config.rs` (546 LOC)
- `coefficient/src/worker/mod.rs` (1,706 LOC)
- `coefficient/scripts/vastai_create_workergroup.sh` (114 LOC)
- `zenavif/scripts/cron_lhs_sweep.sh` (70 LOC)

Total observable LOC of sweep / fleet / cloud-orchestration code today:
~30,000 LOC across three stacks. The unified `vast-worker` proposal
targets reducing this to ~6,000-8,000 LOC in one Rust crate plus per-codec
adapter shims of 50-200 LOC each.
