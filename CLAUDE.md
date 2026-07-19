# zenmetrics CLAUDE.md

See global ~/.claude/CLAUDE.md for general instructions.

## Canonical branch is `master` — NEVER push `main` (enforced)

This repo's one true branch is **`master`** (the GitHub default; the only branch
CI triggers on; where all history lives). There is **no `main` branch** — and a
GitHub ruleset (`no-main-branch`, id 18099751) **blocks creating `refs/heads/main`
server-side**, so a stray push to `main` is rejected, not silently merged.

Why this rule exists: the global `~/.claude/CLAUDE.md` examples say `main`
(`jj new main@origin`, `jj bookmark set main`, `jj git push --bookmark main`). For
THIS repo that creates a stray `main` that diverges from `master` and orphans work
off the default branch. On 2026-06-25 `main` had accrued 7 commits that had to be
rebased back onto `master` and the branch killed. **Substitute `master` for `main`
in every jj/git command here:**

```
jj new master@origin -m "<task>"                       # start
jj bookmark set master -r @ && jj git push --bookmark master   # push
jj git fetch && jj rebase -d master@origin              # if push rejected
```

If `jj git push --bookmark main` is rejected by the ruleset, you followed the
global `main` example by reflex — re-point to `master`. Do not "fix" it by
disabling the ruleset.

## ghcr package names — ONE per artifact (enforced)

Before referencing or pushing any `ghcr.io/imazen/<name>` image: the canonical
package set is **`zenmetrics-sweep`, `zenfleet-worker`, `pycvvdp-scorer`,
`zen-train`** — and that's it. Variants (GPU build, provider flavor, generation,
commit pin, the shared base) are **TAGS** (`:exec-gpu`, `:hetzner`, `:v27`,
`:base-x86-cuda`, `:<sha>`), never new package names. The bake-everything base is
`zenfleet-worker:base-{x86,arm,x86-cuda}`, not a separate package. The source of truth is [`ghcr-packages.json`](ghcr-packages.json);
`just ghcr-check` (CI: `.github/workflows/ghcr-guard.yml`) fails if any infra file
uses a non-canonical name. To add a real new artifact, add it to the manifest in
the same change. Policy + the migration playbook for the existing splinters:
[`docs/GHCR_PACKAGES.md`](docs/GHCR_PACKAGES.md). `just ghcr-audit` diffs the live
org packages against the manifest.

## Fleet monitoring — actively flag idle/wasted infrastructure (standing rule)

Whenever a fleet is up (vast.ai / Hetzner / RunPod / Salad / basement), every box
costs money per hour. **Actively watch for idle/underutilized infrastructure the
whole time it runs — do not launch-and-forget, and report waste without being asked.**

- **Canonical detector: `zenfleet-core::idle`** (`crates/zenfleet-core/src/idle.rs`).
  A box past warmup is idle if: no heartbeat in 180s (frozen/dead) OR GPU ≤10% on a
  GPU box OR ≤1 job/hr (from `jobs_done/uptime`). A paid idle box burns
  `wasted_usd_per_hr`. **Every tool uses these same thresholds — do not invent new ones.**
- **There is ONE monitoring command: `scripts/jobsys/fleet`.** It replaced the old
  6-script sprawl (fleet_util_snapshot / fleet_status / watch_fleet / fleet_startup_watch /
  vast_cost_watch — all deleted). `fleet watch <run>` shows EVERYTHING in one place —
  boxes, $/hr burn, per-box GPU/CPU util, IDLE boxes, boxes that FAILED TO START within
  ~2 min (image-pull hang / onstart crash / fast-crash), and ledger/sidecar progress —
  and alerts (with `--destroy`, tears down) on idle / startup-failure / `--max-burn`.
  `fleet status <run>` = one-shot; `fleet top` = live ledger top; `fleet launch` / `fleet
  kill` wrap the launcher / teardown. `launch_fleet.sh` auto-spawns `fleet watch` in the bg.
- **Do NOT add another monitoring/launch/onstart script.** The guard `just fleet-check`
  (CI: `.github/workflows/fleet-guard.yml`) fails if a new `fleet_*` / `*_watch` /
  `launch_*` / `onstart_*` script appears outside the canonical set in `fleet-tools.json`.
  Add a subcommand to `fleet`, not a new script.
- **Canonical idle detector: `zenfleet-core::idle`** (`crates/zenfleet-core/src/idle.rs`) —
  past warmup: no heartbeat 180s, GPU ≤10%, or ≤1 job/hr. `fleet` mirrors these thresholds;
  the dashboard (`zenfleet-dash`) fires `FleetStalled` / `Underutilized` + shows util per worker.
- **On an idle / failed-to-start paid box: tear it down** and tell the user the $/hr saved.

## Data provenance — READ BEFORE TRAINING

**[`~/work/zen/DATA_PROVENANCE.md`](../DATA_PROVENANCE.md)** is the
canonical record of which R2 sidecars came from which codec commits.
Consult before training any picker / metric / regression on the
backfilled data — codecs like `jxl-encoder` shift RD curves between
commits, so mixing v22-produced and v23-produced JXL rows poisons the
fit. The doc records:

- R2 paths (input parquets, sidecars, encoded variants)
- Codec HEAD commit SHAs per backfill image (v22 / v23)
- Sidecar schema (column types + meanings)
- Reading recipes (pyarrow + s3fs)

Append a new section to that doc when you start a new backfill.

## CVVDP scoring on zensim training datasets

Historical notes (the 2026-05-1x cvvdp sidecar/backfill program — NOT a binding
pin per the user's 2026-06-25 correction) were moved to
[`docs/CVVDP_HISTORY.md`](docs/CVVDP_HISTORY.md) on 2026-07-19. cvvdp now scores
via `zenmetrics score-pairs --metric cvvdp` + the unified worker
(`onstart_unified.sh` → `zenfleet-sweep worker`).

## CANONICAL picker corpus + train/val/test split (read before ANY picker/sweep work)

**Full guide: [`docs/CLEAN_PICKER_PROGRAM.md`](docs/CLEAN_PICKER_PROGRAM.md).** Blind/forgetful
sessions: read it; do NOT invent a split or pick a corpus ad-hoc.

- **Split rule (one source of truth: [`scripts/picker/origin_split.py`](scripts/picker/origin_split.py)):**
  by ORIGIN image, last digit of the origin id — **{0,2,4,6,8}=train, {1,3,5}=validation, {7,9}=test**;
  every sizing/crop/encode derivative inherits the origin's bucket (nothing leaks). Deterministic, no
  seed. Train only ever sees even-origin content. Call `origin_split.split_of()` — never re-implement
  parity or use a seeded/random shuffle (the old `train_hybrid` per-rendition 20% shuffle was WRONG:
  per-rendition → scale leakage). `train_hybrid` now hard-errors if `origin_split` isn't on PYTHONPATH
  (add `scripts/picker`) — refuses a leaky fallback — and reports held-out **test** (7/9) alongside val.
- **Canonical corpus = imazen-26** (`/mnt/v/output/imazen-26-features/imazen26_manifest.tsv`, sha256-
  provenanced, 2157 origins → 1082 train / 657 val / 418 test, balanced across all 12 content classes).
  Segmented: `scripts/picker/segment_imazen26.py` → `imazen26_split_evenodd.tsv` + `imazen-26-split/{train,validate,test}/`.
  **dense-r6 is SUPERSEDED for clean training** (built from `K500_even` reps → train-biased, only 64 val
  + 48 test origins; `o_`=imazen-26, `v2_src`=imazen-26-png-v2).
- **Deliverables: clean even/odd pickers for jxl lossy + lossless, zenjpeg, zenavif** — sweep on
  segmented imazen-26 → train (origin split) → bake ZNPR → **commit the `.bin` into the codec crate**.
  Status table lives in `docs/CLEAN_PICKER_PROGRAM.md`. Verified on dense-r6: clean split holds the
  ≤1% top-3-verify (val 0.52% / TEST 0.42%, val→test +0.08pp — generalizes).

## PINNED PROGRAM — JXL lossy knob-space ablation (iterate to the picker shape)

**Status: active, multi-cycle. Survives compaction. Full plan:
[`docs/JXL_LOSSY_KNOBSPACE_ABLATION_PROGRAM.md`](docs/JXL_LOSSY_KNOBSPACE_ABLATION_PROGRAM.md).**

Goal: discover the **minimal knob shape** a JXL lossy picker should explore — which knobs +
*crosses* carry **content-dependent** RD value worth picking — and push everything else into
**code** (fixed default or feature-derived rule). Loop: design grid → fast Hetzner fleet sweep
(job system, per-cell, persist-everything to zentrain) → analyze (Pareto win-rate /
content-dependence / interaction, GBDT importance) → prune+pivot → **edit jxl crates to code the
settled knobs** → repeat until the grid stabilizes and the picker's achieved RD ≈ oracle. A knob
graduating swept→coded is a SUCCESS (shrinks codec + picker).

Decision rule per knob/cross: inert or universal → CODE; feature-deterministic → CODE RULE;
content-dependent + moves RD → PICKER axis; joint≠main-effects → keep the CROSS, else code the
main effects. Sweep ALL efforts **e1–e9 first** (each adds a real gate — incl. e9's lz77 +
enhanced_clustering — so we don't wrongly bury a knob that only pays off at e9; e10–12 only under
`--features butteraugli-loop`). Honor the byte-inert skip-list + the content-gate pinning gotcha
(see the doc). codec-corpus RO / zentrain RW. Carry CVVDP (cost-model never re-fit).

Every /loop tick: re-read the doc's "Current state / next action" and advance the next phase
(P0 main-effects → P1 crosses → P2 code-the-settled → P3 picker+oracle-gap) rather than drifting.

## burn: GPU-metric kernels ABANDONED ≠ training (separate binary, NOT a graph conflict)

Two *different* questions about burn live in this repo; don't conflate them:

1. **burn/cubek for GPU metric KERNELS** — ABANDONED (`burn-conv-spike`,
   `crates/cvvdp-gpu/docs/BURN_PORT_PLAN.md` "Status: ABANDONED", 4.32× slower
   than the hand-written separable stencil). The `cvvdp_burn_*` column namespace
   stays reserved but unused. Keep hand-written `#[cube]` kernels.

2. **burn for model TRAINING** — VIABLE and the chosen path. `burn-ranknet-spike`
   trains a RankNet/picker MLP via autodiff (custom pairwise + monotonicity loss
   → 0.998 pair-acc) — replacing `zensim-train-core`'s hand-rolled backprop.

**Architecture (decided 2026-06-09):** run metric scoring as separate binaries
that emit **parquet** sidecars; run training as a **separate standalone binary**
(burn + its own cubecl) that consumes those parquets and bakes ZNPR. They hand
off **data, not tensors** — so burn and the published **`zenforks-cubecl`** fork
**never share one cargo graph.** That coexistence problem is sidestepped by
construction. Do **NOT** add `burn` to this workspace's (zenforks-cubecl) member
graph: the rename `cubecl = { package = "zenforks-cubecl" }` can't reach burn's
own `cubecl-core` dep, `[patch]` can re-source but not rename, and the rename
exists precisely so our GPU crates can be *published* (patch is build-local). The
only thing that would force one graph is **differentiable metrics** (autodiff
*through* a zenforks kernel) — not on the table; revisit per
`crates/burn-ranknet-spike/README.md` if it ever is.

**Full ML-strategy write-up:** [`docs/ML_FRAMEWORK_AND_PICKER_ABLATION_2026-06-09.md`](docs/ML_FRAMEWORK_AND_PICKER_ABLATION_2026-06-09.md)
— the candle/burn/linfa 3-layer verdict, the GBDT teacher/GD-MLP-student framing +
measured model sizes (GBDT 975 KB / 109 KB gz vs ~27 KB ZNPR MLP), and the **picker
feature/knob ablation design** (conditional features×knobs×zq matrix; ablate inputs
by redundancy cluster, ablate outputs by RD-spread + content-dependence; GBDT as the
feature-selection instrument). Read it before scoping any picker.

## Local CUDA toolkit (for building/running GPU metrics)

The water-cooled 7950X workstation has CUDA 13.2.1 SDK installed at the
default location, but **nvcc is not on PATH by default**. CUDA layout:

    /usr/local/cuda            → /usr/local/cuda-13.2  (current symlink)
    /usr/local/cuda-13.2/bin/nvcc
    /usr/local/cuda-13.2/lib64/  (libcudart.so etc.)

Other versions also installed: 12.6, 13. Use `/usr/local/cuda` (the
symlink) unless you have a reason for a specific version.

To compile a `cargo` invocation that needs nvcc, prepend:

    PATH=/usr/local/cuda/bin:$PATH
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

But note: **cubecl-cuda dynamically loads CUDA at runtime** via dlopen,
so building `--features sweep,gpu,gpu-cuda` succeeds even with nvcc off
PATH. The runtime fallback is sufficient for `zenmetrics` builds. Set
PATH explicitly only when shelling out to nvcc directly.

GPU info: `nvidia-smi` driver 596.21 / CUDA capability runtime 13.2.

## Sweep scheduling models — read BEFORE touching sweep features (CRITICAL)

This repo has TWO ways to execute sweep work; new sweep capabilities must land in BOTH
or explicitly document why not (2026-06-11: the --plan integration initially landed only
in chunk mode and had to be retrofitted):

1. **Chunk mode** — `zenmetrics sweep` (sweep/run.rs) + the vastai worker
   (`InlineGroupSpec`). Unit of retry = (image × grid-or-plan). For one-pass GPU-metric
   fleet runs.
2. **Job system** — zenfleet-core ledger + `zenmetrics jobexec` (the ZEN_EXEC executor).
   Per-cell content-addressed `DesiredJob`s; completion = declare → gap → re-reconcile.
   Built precisely because big sweeps (100k-cell AVIF) never finish in one pass. Entry:
   `--plan … --dry-run --emit-cells` → `zenfleet_ctl::declare_encodes`.

Plan-driven cells (ALL FIVE codecs: zenjpeg/zenavif/zenjxl/zenwebp/zenpng, verified
end-to-end 2026-06-11) flow through both with ONE identity (`{"cell","fp","plan"}` in
`knob_tuple_json` / `Encode.knobs`); the stratum id is self-describing
(`config_from_cell_id` / `variant_from_cell_id` per codec) and the fp is verified at
execute time. The vastai chunk fleet consumes plan cells as identity rows in plan-mode
input parquets (`generate_sweep_input.py --cells-jsonl`; the sweep runner's tuple path
routes them through `resolve_verified` — byte-identical to the Planned path, tested).
Contract + per-codec scalar-axis inventory: `docs/PLAN_SWEEPS.md`; job-system flow:
`docs/RUNNING_JOBS.md` §4b. Local-build note: the `zenjxl-decoder` workspace patch is
now a pinned git rev (0bd33d21, decoder main with `reject_progressive`) — zenjxl main
(b04ca75 onward; sibling checkout + CI pin now at 4c0d672f, the 2026-06-12 scalar-axes
landing) consumes that unreleased API; drop the patch when zenjxl-decoder 0.3.11
publishes AND zenjxl bumps its `jxl` dep (Cargo.toml patch comment). The 2026-06-12
scalar-axis landings (zenjpeg fff81900 / zenavif e9de3022 / zenjxl 4c0d672f / zenwebp
700aa4a8) extend the id grammars + fingerprints — declare/execute builds must pair at
those revs or newer (PLAN_SWEEPS.md §6 "Codec-rev pairing").

## Sweep build cheat sheet

- **Default CPU+GPU build (development)**:
  `cargo build --release -p zenmetrics-cli`
  → includes both `cpu-metrics` (default) and `sweep` codecs. ~2 min cold,
  seconds incremental.

- **GPU sweep build (production worker)**:
  `cargo build --release -p zenmetrics-cli --no-default-features --features sweep,png,gpu,gpu-cuda`
  → builds the GPU metric backends. **CORRECTION (audit 2026-06-25): this does NOT exclude
  cpu-metrics and is NOT a forced-GPU-only build.** `gpu` enables `gpu-zensim`, which pulls
  `cpu-metrics` transitively (`crates/zenmetrics-cli/Cargo.toml`: `gpu` → `gpu-zensim` →
  `cpu-metrics`), so the CPU butteraugli/zensim/ssim2 paths ARE compiled in and a chunk CAN fall
  back to CPU — the old "fail loudly / can't silently fall back" guarantee was false. To force-fail
  on CPU metrics you must first break the `gpu-zensim → cpu-metrics` dep in Cargo.toml; not possible
  via feature selection alone today. ~4 min cold.

- **WGPU variant (broader GPU compatibility, no CUDA SDK required)**:
  `cargo build --release -p zenmetrics-cli --no-default-features --features sweep,png,gpu,gpu-wgpu`
  → uses Vulkan/Metal/DX12 via wgpu. Use when targeting AMD/Intel GPUs
  on vast.ai. CUDA NVIDIA GPUs work but CUDA backend is faster.

- **CPU metric coverage — `cpu-metrics` is 4 of 6, NOT all six (the trap that cost a session 2026-06-26):**
  the default `cpu-metrics` bundle pulls CPU **butteraugli / zensim / ssim2 / dssim** only.
  **cvvdp and iwssim have in-tree SIMD CPU crates (`crates/cvvdp`, `crates/iwssim`) but are NOT in
  `cpu-metrics`** — reach them via `--features orchestrator,orchestrator-cpu-cvvdp` (resp.
  `orchestrator-cpu-iwssim`), which turn on `zenmetrics-api/cpu-cvvdp` so `zenmetrics-api::cpu_dispatch`
  (`Backend::Cpu`) holds a `cvvdp::Cvvdp`. So cvvdp/iwssim are **NOT "GPU-only"** — the README (lines
  16–36) states all six expose a CPU backend and `zenmetrics-orchestrator` has a *tested* no-GPU fallback
  ladder (`tests/no_gpu_fallback.rs`, `gpu.rs:32`) that selects `Backend::Cpu`. But that failover only
  reaches cvvdp/iwssim **if the build enabled their `cpu-*` feature.** A build with neither `gpu-cvvdp`
  nor `cpu-cvvdp` errors on `score-pairs --metric cvvdp` (`orchestrator_glue.rs:200`: "CPU variant of
  'cvvdp' is not available in this build; rebuild with --features …") — that error is a **build-config**
  message, not an architecture limit.
- **`score-pairs` bypasses the umbrella/failover for cvvdp when built with `gpu-cvvdp`:** it constructs a
  typed `cvvdp_gpu::CvvdpBatchScorer` (caches one `Cvvdp<R>` GPU instance across pairs to dodge the
  ~200 MB/NVRTC per-pair compile that OOMs fleet chunks — `Cargo.toml:91-93`) and calls it directly
  (`main.rs:2134`, `scored_via_cvvdp`), short-circuiting `run_metric()` — the umbrella `Metric::new` /
  `compute_srgb_u8` path that the orchestrator's `Backend` selection + CPU failover live behind. So on a
  `gpu-cvvdp` build cvvdp never consults the failover. To force CPU cvvdp: build WITHOUT `gpu-cvvdp`,
  WITH `orchestrator,orchestrator-cpu-cvvdp`.
- **Before claiming any metric is "GPU-only": `ls crates/` first.** `crates/cvvdp` + `crates/iwssim` are
  SIMD CPU ports; never infer architecture from a feature-gated build error (memory:
  `enumerate-repo-before-capability-claims`).

## Sweep runner discipline

- **GPU metrics only on production workers.** Mixing CPU/GPU scores
  across a sweep produces inconsistent training data — pickers/trainers
  expect a single metric backend. NOTE (corrected 2026-06-26): the GPU
  build does NOT force-fail to GPU (see the cheat-sheet correction above) —
  a chunk CAN fall back to CPU for metrics whose `cpu-*` feature is compiled.
  Keep a sweep on one backend by **selecting metrics whose backend you
  control** and verifying the score column's impl tag, not by assuming the
  build forbids CPU.
- **Pre-uploaded binary lives at**
  `s3://coefficient/binaries/zenmetrics-<version>-linux-x86_64`
  (R2 endpoint: `${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`). Workers
  fetch via `SWEEP_BIN_OVERRIDE` env var.
- **Onstart script**: `scripts/sweep/onstart_unified.sh` — the ONE worker entry;
  execs `zenfleet-sweep worker --backend vastai --mode omni` (claim loop, adaptive
  concurrency, in-process scoring, arrow parquet IO — one process, all metrics). The
  legacy per-metric bash onstarts (onstart_v3/omni/cvvdp/iwssim/…) were deleted
  2026-06-25; `--mode feature-backfill` and `onstart_orchestrator.sh` cover the variants.
- **Every onstart MUST self-destroy on failure** — upload tail log to
  R2 + issue `vastai destroy instance ${CONTAINER_ID}`. See
  `scripts/sweep/CLAUDE.md#critical-every-onstart-must-self-destroy-on-failure`
  for the two acceptable patterns (image-level
  `run_with_error_trap.sh` wrapper — what `onstart_unified.sh` uses). Workers that exit without
  destroying burn \$/hr until externally cleaned up — that's the
  cost-leak the 2026-05-18 EXP-LARGER-LARGE incident chased.

## Heterogeneous SPLIT — encode-once (CPU) / score-many (GPU)

For multi-GPU-metric passes (butteraugli + cvvdp + ssim2-gpu + zensim-gpu),
encode once on cheap CPU and persist the variants, then score every GPU metric
over those persisted variants — never re-encode per metric.

- **CPU half**: `scripts/sweep/hetzner_cpu_sweep.sh` — sweeps with
  `--encoded-out-dir`, tars variants to R2 (the master record: 372 zensim
  features / diffmaps / future metrics re-derivable with no re-encode), and
  emits `pairs.tsv` (`image_path codec q knob_tuple_json ref_path dist_path`,
  in-container `/data/` paths).
- **GPU half**: `scripts/sweep/split_score_worker.sh` in
  `ghcr.io/imazen/zenmetrics-sweep:v29-split` (FROM the v29 GPU binary). Pulls
  variants+ref+pairs.tsv, runs `zenmetrics score-pairs --metric <m>` per GPU
  metric → one parquet sidecar each. Self-uploads its log to
  `sidecars/worker.log` and self-destroys on success.
- **vast quirk**: vast runs `--onstart-cmd`, NOT the image ENTRYPOINT — launch
  via `--onstart-cmd "bash /usr/local/bin/split_score_worker.sh > /var/log/split.log 2>&1"`.
  Pick a fast-net (`inet_down>300`) CUDA-matched (`cuda_max_good>=12.6`) offer;
  cheapest offers are slow-pull duds. Snap-docker here can't read `/tmp` — build
  SPLIT images from a `$HOME` context.
- Doc: `benchmarks/picker_fleet_2026-06-23.md`; memory `heterogeneous-fleet-split.md`.

## Known Bugs

- **zenmetrics-api consolidated `it` suite self-poisons when run as ONE
  process** (observed 2026-06-10, pre-existing — A/B-identical 26-test failure
  set on master 7158c443 with and without the PuLumaGrayF32 change):
  `session_cap::allocator_cap_recycle_leak` caps the shared cubecl session
  allocator and later GPU tests in the same process inherit the poisoned
  client (panics at zenforks-cubecl-runtime client.rs:905). Same family as the
  ssim2-gpu one-process OOM below; workaround: run per-module/per-test
  processes. Lib + hdr unit tests and per-test runs are green.

- **ssim2-gpu consolidated `it` suite OOMs the 12 GB RTX 5070 when run as
  ONE process** (observed 2026-06-10, pre-existing at 704b19dd — NOT from
  the PU21 commit de2ced69; identical 61-test failure set on both). The
  42a107b1 test consolidation put all 98 GPU tests in one binary; cubecl's
  CUDA memory pool grows across tests and never returns pages, so
  `cargo test -p ssim2-gpu --features cuda,cubecl-types --release -- --test-threads=1`
  hits `CUDA_ERROR_OUT_OF_MEMORY` (PTX load) partway through, the server
  goes `ServerUnhealthy`, and every later kernel test cascade-fails. Onset
  point varies with ambient GPU pressure (54 vs 61 failures across runs).
  Every individual test passes in a fresh process (verified exhaustively,
  101/101 at e0995ae7 via per-module + batch-of-3 runs). Workarounds:
  filter to module groups (`--test it strip_parity::` etc.) in separate
  invocations. Proper fix candidates: per-module process isolation in CI
  invocations, a cubecl pool flush/shrink hook between tests, or capping
  concurrent pipeline allocations in the heavy 4096² tests. CI's
  macos-Metal job (8 GB unified) may hit the same wall.

### Resolved

- **`sweep --metric ssim2` failed with "not enabled in this build" on
  `cpu-metrics` builds — CPU dispatch checked the wrong feature flag (found
  2026-07-02/03) — FIXED 2026-07-03 by `9f93e56b`** ("fix(metrics):
  ssim2/dssim/butter/zensim CPU-only builds cannot construct MetricParams").
  Root cause: the sweep's CPU scoring path was gated behind the GPU-typed
  `MetricParams::Ssim2` variant — i.e. `zenmetrics-api`'s PLAIN `ssim2`
  feature (`["dep:ssim2-gpu", "dep:zenmetrics-gpu-core"]`, GPU-only) —
  instead of `cpu-ssim2` / the umbrella `Backend::Cpu` dispatch, so a
  CPU-only build (`cpu-metrics` without `gpu`/`gpu-ssim2`) could never
  satisfy the cfg even though `list-metrics` correctly reported ssim2 as
  CPU-available. Historical workaround in pre-fix sweeps: `--metric zensim`
  only, with `score_ssim2` backfilled as 100.0 for lossless corpora (a
  measured constant — every lossless cell has identical pixels).

- **jxl `modes_full` memory — RESOLVED 2026-06-25; the "BufferPool leak" was a
  MISDIAGNOSIS.** There is NO per-cell / within-process leak. Measured on current
  HEAD (agent replication; `/tmp/repro_jxl_VERDICT.md`, `/tmp/repro_jxl_rss.tsv`
  — evidence was in /tmp — wiped; re-measure to ~/tmp if needed):
  serial jxl `modes_full` RSS is a **sawtooth that returns to baseline between
  images** (per-image peaks ~11 GB @1.77 MP, ~22 GB @3.15 MP; valleys 1.5–2.7 GB),
  `--jobs 1` runs to completion with NO OOM, and heaptrack leaked **3.62 MB over
  55 cells** (a 60 MB/cell leak would be ~3.3 GB). jpeg stays flat <200 MB.
  `butteraugli::image::BufferPool` is a plain struct capped at 8 buffers
  (`image.rs:16,141`), owned inside a per-encode `ButteraugliReference`,
  constructed fresh in `butteraugli_refine_quant_field` and dropped on return —
  it does NOT persist across encode calls. The per-encode pool fix already landed
  in jxl-encoder `26a8d9cd` (#93) + a `MemoryBudget` guard.
  **A single jxl encode is SMALL (measured 2026-06-25, 3.15 MP, isolated via
  `jobexec`):** lossy VarDCT **0.20 GB**, lossless modular **1.50 GB**
  (thread-independent — same at 1 and 28 threads). NOTE on cell mix: raw
  `modes_full` is ~99% LOSSY (77,760 lossy strata across 10 axes + 630 lossless;
  156k cells/image unbudgeted) — it is the full Cartesian product, meant to be
  paired with `--plan-budget`/`--max-deviations`. A `--plan-budget` collapses the
  lossy cross hard: `--plan-budget 400` → 6 `_def` lossy modes + 315 modular (the
  earlier "96% modular" figure was this budgeted artifact, NOT raw modes_full).
  Neither raw nor budgeted modes_full is a good lossy-picker plan — see
  PLAN_SWEEPS / the lossy_dense recommendation (cross the high-value perceptual
  knobs: epf/gaborish/k_ac_quant/try_dct*/entropy_mul, ~360 strata × dense-q).
  **The OOM is the MONOLITHIC `zenmetrics sweep` accumulating across cells within
  ONE process** — `modes_full` on a single 3.15 MP image ramps RSS to ~13–24 GB
  across its 315 modular cells with NO per-cell release (allocator high-water, not
  a true leak), × parallel images on the box → the 31 GB OOM. The old
  "NOT thread-bound" datum fits: per-cell memory is fixed (1.5 GB); cells-in-one-
  process is the driver, not threads. **The job system bounds it by construction:
  one encode per FRESH process = ≤1.5 GB, freed on exit** — so `modes_full` runs
  fine via the job system, per-box concurrency ≈ box_RAM ÷ 1.5 GB (a 32 GB box ≈
  ~18 concurrent modular encodes). Do NOT use `rd_core` to dodge this — it's the
  crippled pre-ablated set (RD_ABLATION_2026-06-24.md); use `modes_full` via the
  **job system** (not the monolithic sweep). If the monolithic sweep must be used
  for jxl, bound image concurrency AND add a per-cell free / `malloc_trim` (the
  cross-cell within-process growth is a sweep-mode artifact, not a single-encode
  cost). Measurements: `/tmp/repro_jxl_VERDICT.md` (evidence was in /tmp — wiped;
  re-measure to ~/tmp if needed), single-encode `/usr/bin/time -v`.

## CHANGELOG.md

Maintained in repo root.

## KADIS-700k dataset (zensim 2026-06-30; GPU-metrics 2026-07-01)

700,000 distorted-image cells — 140k KADIS pristine references × 1 `dist_type_1` × 5 severity
levels, each with a 372-D zensim feature vector. **THIS crate ran both sweeps** (chunk-mode on a
vast.ai fleet). Two canonical variants (same 700k cells, same `source_id` split key):

- **★ GPU-metrics canonical (2026-07-01) — current, richest.**
  `s3://zentrain/kadis-700k-gpu/canonical/kadis700k_canonical_gpu_2026-07-01.parquet`
  (700k×387, ~936 MB zstd, 0 nulls; sha256 `c9a6fd56…`). **7 perceptual scores** —
  `score_{zensim,ssim2,butteraugli_max,butteraugli_pnorm3,iwssim,dssim}_gpu` + `score_cvvdp_cpu_imazen_v0_1_0`
  — plus `distorted_url` (a persisted distorted PNG per cell → rescore-from-links, via
  `ZEN_PERSIST_DISTORTED=1`), on top of the 372-D `feat_*` + shared keys. Config
  `METRICS=zensim-gpu,ssim2-gpu,butteraugli-gpu,cvvdp,iwssim-gpu,dssim-gpu` + `ZENMETRICS_SWEEP_LEGACY=1`
  + `with-iw`. Sidecars `s3://zentrain/kadis-700k-gpu/{omni,zensim_features,pairs}/` + `distorted/<chunk>/*.png`.
- **zensim-only canonical (2026-06-30) — earlier variant.**
  `s3://zentrain/kadis-700k/canonical/kadis700k_canonical_2026-06-30.parquet` (700k×380, ~906 MB
  zstd, 0 nulls; sha256 `b57e4b3f…`). Pure-CPU config `METRICS=zensim` + `ZENMETRICS_SWEEP_LEGACY=1`
  + `with-iw` + `MAX_CHUNKS_PER_PROCESS=50`. ~91 cells/s/box, ~$0.7 total. `score_zensim` +
  `feat_0..feat_371`. Sidecars `s3://zentrain/kadis-700k/{omni,zensim_features,source_features}/` (350 each).
- **Both runs used `ZENMETRICS_SWEEP_LEGACY=1`** to disable the orchestrator cubecl warm-bench —
  the descriptor race at `cubecl-runtime memory_manage.rs:418` is why the full-orchestrator GPU path
  races on fresh boxes (removing that need is tracked separately: `sweep_runner.rs:76`). Three
  upstream bugs noted in `~/work/kadis-distort/benchmarks/pipeline_full_700k_2026-06-30.md`: hardcoded
  `coefficient` claim bucket (`chunk.rs:63`); omni-skip gated on `!skip_claims` (`chunk.rs:30`);
  orchestrator/cubecl init even when metrics don't need it (`sweep_runner.rs:76`).
- **Update 2026-07-01 (score-many opt): the Legacy=1 need was RE-TESTED and does NOT reproduce on a
  real card.** A real-Linux repro (vast RTX 3060, 12 GB, driver 570) ran the MODERN orchestrator GPU
  path with NO Legacy under two concurrency forms — `zenmetrics sweep --jobs 8` (80/80 cells,
  score-fail=0) AND 8 concurrent independent `score-pairs` processes (all 8 wrote 600/600, 0 NaN) —
  with ZERO panics / `memory_manage` / `CUDA_ERROR` / `ServerUnhealthy`. The `memory_manage.rs:418`
  race did NOT fire. **Fleet-default recommendation: modern orchestrator + `--bench-on-start no`**
  (skips the warm-bench the bullet above blames, keeps the OOM ladder + capability cache) **+ a
  per-box GPU self-test** (score one known pair at onstart, `exit 1` on failure so
  `run_with_error_trap` self-destroys — this ALSO catches a runtime-image missing `cuda_runtime.h`,
  which makes cubecl's NVRTC JIT fail to compile cvvdp/dssim/butteraugli; GPU fleet images MUST bake
  CUDA dev headers or set the NVRTC include path). Keep `ZENMETRICS_SWEEP_LEGACY=1` as an ESCAPE HATCH
  only; do NOT chase a deep cubecl fix without a reproducer. Caveat: one card/driver/workload tested.
  Repro + the score-many warm-ref opt (TAR-SHARD + `Orchestrator::run_all` warm-ref, 1.60× measured):
  `docs/SCOREMANY_OPT.md`.
- **Shared keys (both):** `source_id` (stable split key 0..139999 — split on this, never on row),
  `source_filename`, `dist_type`, `dist_name`, `severity_level`, `dist_param` (signed for 7/18/25).
- **Mirrors:** `/mnt/v/datasets/kadis700k/canonical/`, `/mnt/tower/output/kadis700k/canonical/`.
- **Full README + schema:** `s3://zentrain/kadis-700k-gpu/README.md` + `s3://zentrain/kadis-700k/README.md`
  (and `~/work/kadis-distort/docs/DATASET.md`).
- **Credit:** reference images + distortion design © VQA Group, Universität Konstanz (Lin, Hosu,
  Saupe) — KADID-10k / KADIS-700k, https://database.mmsp-kn.de/kadid-10k-database.html ("freely
  available to the research community"). Cite KADID-10k (QoMEX 2019) + DeepFL-IQA (arXiv:2001.08113).
