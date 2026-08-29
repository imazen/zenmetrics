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
- **Canonical imazen-26 = the dedicated `imazen/imazen-26` REPO** (user-directed move 2026-08-23;
  local `~/work/imazen-26`; predecessor record = codec-corpus PR #12). Source of truth: the repo
  root's `CORPUS-MANIFEST.tsv` (membership oracle, 2160 images) + `manifests/{train,validate,test,split_map}.tsv`
  (last-digit rule, 1084/658/418; sha256 per image) + the **variant-sets/ registry** (per-set
  files.tsv sha manifests + per-project consumer map; new sets ONLY via `scripts/make_variant_set.py`). **Anything derived from
  any OTHER imazen-26 copy is broken unless sha-verified against those manifests** — the `/mnt/v`
  copies below are caches, and the historical feature-era manifest has 2157 of the 2160 images.
- Historical cache (feature era): `/mnt/v/output/imazen-26-features/imazen26_manifest.tsv` (sha256-
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
a pinned git rev (f1faec7 = decoder origin/main, version 0.4.0, since the 2026-08-27
zencodec 0.1.26 migration; earlier 0bd33d21 → 0599dcf) — zenjxl main (9226d3a, the CI
pin too) REQUIRES `zenjxl-decoder = "0.4.0"`, which is unpublished (crates.io: 0.3.10);
drop the patch when 0.4.0 publishes AND zenjxl bumps its `jxl` dep (Cargo.toml patch
comment). Sibling `jxl-encoder` still wants `^0.3`, so the lock carries BOTH decoders
(registry 0.3.10 + git 0.4.0) until it moves. The 2026-06-12
scalar-axis landings (zenjpeg fff81900 / zenavif e9de3022 / zenjxl 4c0d672f / zenwebp
700aa4a8) extend the id grammars + fingerprints — declare/execute builds must pair at
those revs or newer (PLAN_SWEEPS.md §6 "Codec-rev pairing").

## CUDA 12 vs 13 — selected at RUNTIME, never baked (2026-08-20)

**CUDA Toolkit 13.0 dropped offline compilation for Maxwell, Pascal and Volta; its floor is
sm_75 (Turing).** This fleet straddles that line, so no single baked toolkit serves it:

| card | sm | CUDA 12 | CUDA 13 |
|---|---|---|---|
| GTX 1050, GTX 1060 | 6.1 | ✅ | ❌ cannot target |
| GTX 1660 Ti, RTX 2080 | 7.5 | ✅ | ✅ |
| RTX 3070 | 8.6 | ✅ | ✅ |
| RTX 5070 (Blackwell) | 12.0 | ❌ | ✅ |

The `:x86-cuda` base image therefore bakes **both**, and
[`scripts/cuda/cuda-select.sh`](scripts/cuda/cuda-select.sh) picks per box at container start.
`fleet-entrypoint.sh` sources it before any scoring; it is a no-op on CPU boxes.

**How selection works without a rebuild:** cudarc loads NVRTC via `dlopen`, and its FIRST
candidate is the **unversioned** `libnvrtc.so` (`get_lib_name_candidates`, cudarc src/lib.rs).
`dlopen` resolves that through `LD_LIBRARY_PATH`, so exporting `CUDA_PATH` +
`LD_LIBRARY_PATH` chooses the toolkit. The binary stays built with
`CUDARC_CUDA_VERSION=12000` — that governs the **driver** API, which is backward compatible;
only NVRTC is version-sensitive.

**The `-dev` packages are load-bearing.** They provide the unversioned `libnvrtc.so` symlink.
Without it in BOTH toolkit dirs, `dlopen` falls through to whatever `ldconfig` cached and
selection silently becomes a no-op. The Dockerfile asserts both exist at build time.

Decision order: no GPU → no-op · compute cap < 7.5 → **12** · driver advertises CUDA < 13 →
**12** · else **13**. Override with `ZEN_CUDA_MAJOR=12|13`. Driver capability is read from
`nvidia-smi`'s reported CUDA version rather than a hardcoded driver-number table.

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

- **`zenmetrics-orchestrator --lib`: 2 tests fail on ANY macOS host — environment, not a
  regression (confirmed 2026-08-28).** `tests::detect_cpu_returns_nonempty_brand` (lib.rs:938)
  asserts `detect_cpu().brand` is non-empty, and `detect_cpu` has no macOS/`sysctl` arm, so the
  brand comes back `""`. `tests::fresh_profile_is_not_stale_by_time` (lib.rs:893) builds a
  `fake_profile()` claiming an `NVIDIA GeForce RTX 5070` with `gpu.present = true`;
  `is_profile_stale` then calls the REAL `detect_gpu()`, which finds no NVIDIA card on a Mac and
  returns `true`. Both are pure host-capability assertions with no cross-platform arm — neither
  touches the executor / chooser / adapter code. The other 78 pass. Do NOT chase these as a
  regression from an orchestrator change; the fix (if anyone wants one) is a macOS branch in
  `detect_cpu` plus making `fresh_profile_is_not_stale_by_time` use a GPU-absent fake profile.
  Command: `cargo test -p zenmetrics-orchestrator --no-default-features
  --features bench,cuda,cpu-butter --lib`.

- **zensim-gpu integrated-PU (HDR) score drifts from CPU `compute_pu_linear` by up to ~0.5 points
  on textured content — an f32 peak-feature precision limit at HDR magnitudes, NOT the PU kernel
  or the routing (measured 2026-08-28 on Metal/wgpu; zenmetrics#25).** `ZensimOpaque::
  compute_pu_linear_nits_interleaved` vs CPU on the same nits pair: |Δ| 5.0e-3 at 128×96, 0.47 at
  200×150, 0.36 at 256×256. Attribution (per-feature diff + block-swap rescoring): the scale-0
  PU-XYB planes match < 5e-5 and the mean-pooled basic features < 2e-3; the gap is the
  `ssim_max` / `ssim_p95` peak features (idx 156 + 18·scale + 6·ch + {0,3}) — GPU 4.5e-3 vs CPU
  1.5e-3 on Y at 256×256. Both sides form the local variance as `E[x²] − μ²` in f32; PU-XYB Y is
  up to ≈ 2.5 (4000 cd/m²) vs ≲ 1 on SDR planes, so the cancellation noise in smooth regions is
  ~6× larger, the two blur orderings round it differently, and a `max` pool picks the worst pixel.
  Scaling the same pair ×0.002 collapses the drift to 5e-3 (magnitude-pinned). The SDR peaks show
  the same class at ~4% relative (score Δ ~1e-3), so the SDR parity locks are unaffected. Fix
  direction if it matters: a cancellation-stable local variance (blur of `(x−μ)²`, or centred
  planes) in BOTH the CPU and GPU kernels — a feature-definition change that re-bakes BHdr.
  CUDA/Vulkan envelope not yet recorded (the SDR peak drift is smaller there than on Metal).
  Gate + numbers: `crates/zensim-gpu/tests/it/pu_xyb_parity.rs::PU_SCORE_ABS_TOL`.
  **Corollary for test authors:** a pure linear-gradient pair (the umbrella tests' `hdr_pair`
  ramp) is NOT a valid GPU↔CPU parity input on EITHER path — `hf_mag_loss` (basic feature 11) is
  ill-conditioned there (a box mean reproduces a ramp exactly; Σ|x−μ| is rounding noise on both
  backends): CPU 0.22 vs GPU 0.07 on PU, CPU 0.19 vs GPU 0.0 on SDR at the same feature, 4.3
  points apart through the BHdr bake (1e-2 through B). Use textured content for cross-backend
  checks.

- **iwssim-gpu `it::opaque::opaque_gray_f32_identity_and_typed_parity` — ROOT-CAUSED
  2026-08-28: the bug is the `compute_gray_f32` INGRESS, and it is NOT Metal-specific.
  Still not FIXED.** The distorted gray-f32 pair scores `0.9999679561389043` (the test wants
  `< 0.999`), deterministic, with or without `RUST_TEST_THREADS=1`; the other 91 tests pass.
  The earlier note guessed "smells like the gray-f32 path on Metal collapsing the distortion".
  **Half right — it IS the gray-f32 path collapsing the distortion, but Metal is exonerated.**
  A probe fed the SAME content (the test's 256×256 ramp, every 5th pixel +14.0) through three
  paths on this Metal box:

  | path | identical pair | distorted pair |
  |---|---|---|
  | GPU `compute_gray_f32` | 0.9999999468992812 | **0.9999679561389043** |
  | GPU `compute_srgb_u8` | 1.0000001836245906 | **0.9844796071748093** |
  | CPU `iwssim::score` (u8) | 0.9999999973515667 | **0.9857888858819658** |

  The GPU's u8 path detects the distortion and agrees with the CPU reference to 1.3e-3. Only
  the f32 ingress loses it — ~470× less response to the identical distortion. So the pyramid,
  the reduction and the Metal backend are all fine; the defect is in how `compute_gray_f32`
  feeds scale 0. **Prime suspect (read, not yet proven):** `compute_rgb` fills the EXISTING
  `scales[0].g_ref`/`g_dis` buffers via the `rgb_u32_to_gray_kernel` dispatch
  (`pipeline.rs:1152`, `rgb_u32_to_gray_from_packed`), whereas `compute_gray` *replaces the
  handles themselves* with fresh `create_from_slice` allocations (`pipeline.rs:1224-1225`,
  "Swap handles into scale-0"). Anything in the pipeline still bound to the original scale-0
  buffers would then read stale contents. That would explain a distorted score pinned near the
  identity score but not exactly equal to it. Next step: check what else aliases
  `scales[0].g_ref`/`g_dis` across `run_pipeline()`, and make `compute_gray` write INTO the
  existing buffers (as the RGB path does) instead of swapping handles.
  Command: `cargo test -p iwssim-gpu --no-default-features --features wgpu --test it
  opaque::opaque_gray_f32_identity_and_typed_parity`.

- **iwssim-gpu has ZERO CPU↔GPU parity coverage on the wgpu/Metal backend — found
  2026-08-28.** `tests/it/parity_cpu.rs` is the only GPU↔CPU parity module and it is
  `#![cfg(feature = "cuda")]` AND `#[ignore]`d, so `cargo test -p iwssim-gpu
  --no-default-features --features wgpu --test it parity_cpu::` runs **0 tests** (92 filtered
  out). Every wgpu/Metal iwssim result is currently unchecked against the canonical CPU port.
  This is why the gray-f32 ingress defect above could sit undetected behind a test that only
  asserts a threshold. Incidental datum from the probe above: on this Metal box the u8 path
  drifts 1.3e-3 from CPU, which is **above** `parity_cpu.rs`'s own `TOL = 1e-3` — so simply
  un-gating that module would need its cross-backend tolerance re-derived on evidence (a
  tolerance change needs user sign-off; do NOT just widen it).

- **cvvdp-gpu multi-strip Mode B (`StripPair`) walker on macOS/Metal — ROOT-CAUSED
  2026-08-28. The cause is an UNALIGNED STORAGE-BUFFER BINDING OFFSET, and the fix is
  caller-side in cvvdp-gpu (not yet made).** The walker slices row-strips out of its planes
  with `Handle::offset_start(byte_off)` (`pipeline.rs:2926-2929, 3469, 3612-3613, 3662, 3698,
  3795-3803`), where `byte_off = strip_row × level_width × 4`. Metal requires every storage
  binding to start on a multiple of `min_storage_buffer_offset_alignment`, which is **256** on
  this device (measured — `min_uniform_buffer_offset_alignment` is also 256). An image
  pyramid's narrow levels have a row stride BELOW 256 bytes, so those strip offsets are not
  256-aligned: the observed failing offsets are 8-, 48- and 64-aligned (e.g. 3138368 = 256·12259
  + 64, 3139632 = 256·12264 + 48, 5109256 = 256·19958 + 8). CUDA and HIP have no such
  requirement, which is exactly why this is Metal-only — **the walker's offsets are wrong on
  every backend; only Metal enforces it.**

  `wgpu` raises this in `Device::create_bind_group`, which runs on a `DSD-*` device-service
  thread, so it arrived as a panic no `Result` on the caller's thread could see — surfacing as
  the misleading `zenforks-cubecl-runtime/src/client.rs: called Result::unwrap() on an Err
  value: CallError`. **Fixed on the cubecl side** in imazen/zenforks-cubecl `f528c4b5`: the
  wgpu backend now validates each binding offset and reports the offset, the required
  alignment, and the fact that cubecl's own allocations are always aligned so an unaligned one
  came from a caller-built `offset_start` sub-view. Same commit also corrects the pool
  alignment to `max(min_uniform, min_storage)` — it previously used the uniform limit alone
  for pools created with `BufferUsages::STORAGE`. That makes the failure honest and
  actionable; **it does not make the walker work.**

  **The remaining fix is in cvvdp-gpu**: round `byte_off` DOWN to a multiple of 256 and pass
  the leftover element offset to the kernel as a scalar (or pad each pyramid level's row
  stride up to a multiple of 64 elements). Both are real design changes across the strip
  kernels and need the Mode B parity tests as the gate — do NOT weaken those tolerances.

  Original symptom record: `cargo test -p cvvdp-gpu --no-default-features --features wgpu
  --test it mode_b_walker_parity::` fails `mode_b_walker_jod_matches_full_at_128`,
  `_at_1024`, `_at_1024_h_body_256` and `mode_b_walker_dispatches_n_strips_at_1024` with a
  `DSD-*` device-thread panic at `wgpu-29.0.3/src/backend/wgpu_core.rs:1277` surfacing as
  `zenforks-cubecl-runtime-0.10.1/src/client.rs:105: called Result::unwrap() on an Err value:
  CallError`. Any geometry that yields ≥ 2 strips at level 0 (e.g. 256×256 / h_body 64,
  128×128 / h_body 32) hits it; the single-strip 64×64 / h_body 512 case
  (`strip_mode_b_parity::mode_b_score_matches_full_64x64`) passes. Full mode and Mode E
  (warm ref) are fine on Metal. cvvdp-gpu is already omitted from CI's Metal matrix (the
  `Atomic<f32>` pool-kernel note in ci.yml), so CI never sees this either way; the #30
  `tests/it/cancel.rs::strip_pair_mode_polls_per_strip` test is `cuda`-gated for this reason.

- **`zenfleet-worker::tests::exec_command_deadline_kills_and_classifies_timeout` FAILS on
  the ubuntu-latest CI runner — the per-cell watchdog does not stop a `sh -c 'sleep 30'`
  child at its 1 s deadline there; the call returns after the full 30.00 s
  (`returned in 30.002861621s — watchdog must not hang`, lib.rs:3163). First seen on the
  first CI run that could reach the step at all (run 33091197770, commit 60ab7452, 2026-08-27);
  the watchdog (`exec_command_deadline`, `3a5e94ed`, 2026-08-24) has never been CI-tested on
  Linux before because every job died at manifest load from `9093cc23` to `9796743f`.
  Passes on macOS (this suite: 40/40 locally on aarch64 darwin). The zencodec 0.1.26
  migration commit touched that function only via `cargo fmt` reflow (`git diff -w` is empty
  there), so this is pre-existing Linux behavior — likely the `kill -- -<pgid>` / `child.kill()`
  pair not reaching the grandchild `sleep` (so the stdout/stderr reader threads block until it
  exits) — NOT root-caused, NOT fixed. This is the only red job in `Compile (ubuntu-latest)`;
  the same step's other 39 tests pass there.

- **zensim-gpu `it` suite on macOS/Metal (wgpu): 2 deterministic failures, pre-existing
  (verified 2026-08-27 — identical values on baseline b07a0485 before that day's commits, in
  isolation and serially):** `diffmap_invariants::invariant_1_identity_yields_near_zero_diffmap`
  (identity score 2.3330367, expected ~0 — the default CPU-diffmap path, 16×16) and
  `cached_ref_slot_rebuild::gpu_score_correct_after_foreign_size_rebuild` (one-shot AND cached-ref
  GPU 51.2412 vs CPU 48.8323, 2.41 JOD off after the 1024→1448 rebuild + `reclaim_pooled_vram`).
  Not root-caused; the other 103 tests pass. Command:
  `cargo test -p zensim-gpu --no-default-features --features wgpu --tests`. The CI macos-latest
  Metal job runs this same suite — check whether it is red there before blaming a local change.

- **A batch job's gap can have a permanently-unresolved handful of jobs that
  never get a terminal ledger row — NOT root-caused (the restart-loop half of
  this same incident IS resolved, see "Resolved" below).** The
  `fleetbench-gpuscore` (warm-exec) job's every pass reported `skipped=3995` —
  the same 5/4,000 declared jobs that never got a terminal ledger row in the
  original warm-exec run (a permanently-unclaimable/unresolvable chunk, cause
  unknown). Open question if anyone picks this up: why do exactly 5 specific
  declared jobs never resolve to a terminal ledger row. Full context:
  `benchmarks/fleetbench_2026-08-24.md`'s "Fleet-waste finding" section.

- **`zenjxl` encoder panics on specific inputs — root-caused 2026-08-25, fix landed
  upstream + independently re-verified, but NOT YET ACTIVE in this repo's production
  path (needs a human decision before it changes fleet behavior) — see
  `benchmarks/autonomous_fix_run_2026-08-25.md` Track 1 for the full writeup.**
  Originally observed via fleetbench's GPU-score leg (found 2026-08-24): 69/4,000
  GPU-score jobs (1.7%) failed with `error_class: encoder_panic` during `jobexec`'s
  metric-job re-encode step (see `benchmarks/fleetbench_2026-08-24.md`'s CPU/GPU-score
  section). 4 distinct source images (3 of 4 flat-color line-art/illustration
  content — plausible, not proven, common factor), `zenjxl` 68/69 (`zenavif` 1/69),
  spread across effort 5/7/9 (both VarDCT and modular cells) and evenly across all 5
  GPU metrics scored (confirms the panic is in the shared re-encode, not
  metric-specific code). **Root cause (confirmed 2026-08-25 by reproducing the exact
  failing image bit-for-bit under `ulimit -v`):** `count_zero_coefficients` allocates
  small fixed-size (≤16 KiB) scratch buckets infallibly; under genuine allocator
  starvation this hits Rust's default `handle_alloc_error`, which **aborts the whole
  process** (SIGABRT) — not a catchable panic, a hard process abort, hence the message
  never reaching `nomad alloc logs` via normal stderr capture. **Fix: jxl-encoder
  commit `cf50d7cf99de11dbe943b831317bbee49c3abe36`** converts these allocations to
  fallible (`Result`-returning, opt-in via `Limits::with_fallible_alloc(true)`,
  respecting the crate's existing caller-controls-fallibility design) +
  **zenjxl commit `e79179ecce51b4250d9106584b3ce9d68d994ea3`**. Independently
  re-verified: both commits are current `origin/main` HEAD in their repos; the new
  tests pass on a fresh build; the exact real failing image reproduces the original
  SIGABRT without the flag and a graceful `Result::Err` with it, under identical
  `ulimit -v`. **Confirmed NOT yet active in production**: `zenmetrics-cli`'s
  `sweep/plan.rs` zenjxl re-encode call site uses bare `.with_threads(1).encode(...)`
  with no `Limits` object, so the fallible path is never engaged today — fleet
  behavior for this crash class is unchanged until a deliberate zenmetrics-side change
  opts in. **Decision pending (yours):** should `plan.rs` opt into
  `with_fallible_alloc(true)`, or should jxl-encoder's default policy for small bounded
  scratch buffers change unconditionally (overriding the crate's per-caller-fallibility
  convention)? Neither was decided for you. Also still open: zenmetrics'
  `classify_msg()` labels this failure `encoder_panic`, which is accurate-enough but
  imprecise (it's an allocator abort, not a Rust `panic!()`) — flagged, not fixed.

- **fleetbench GPU-score OOM under concurrency — found 2026-08-25, separate class from
  the encoder_panic above, NOT fixed (this repo's problem, unlike the panic above).**
  A single jxl encode is small (~0.8–1.1 GB measured), but the zenfleet-worker
  GPU-score executor admits multiple concurrent single-threaded encodes into one
  memory-capped container — each fits alone, the aggregate exceeds the cap. This is a
  `BoxBudget`/admission-control accounting gap for concurrent re-encode processes, not
  a jxl-encoder or zenjxl issue. Not root-caused further or fixed; needs
  `zenfleet-worker` admission control to account for concurrent re-encode memory, not
  just per-job VRAM (see the existing VRAM-admission work in `### Resolved` below for
  the analogous GPU-side mechanism). Full context:
  `benchmarks/autonomous_fix_run_2026-08-25.md` Track 1.

- **Chunk-mode's per-chunk lease claim provides ZERO cross-worker dedup on a
  heterogeneous-core-count fleet — measured 100% duplicate-execution rate on a real
  3-box run (found 2026-08-24, fleetbench-2026-08-24 G-P0 baseline,
  `benchmarks/fleetbench_2026-08-24.md` has the full writeup).** `chunk-id =
  sha256(member job-ids)` and chunk MEMBERSHIP comes from
  `BoxBudget::pack_chunks_lpt(&self, jobs, target_wall_sec)`, where `&self` is the
  CALLING BOX'S OWN core/RAM budget (`crates/zenfleet-worker/src/lib.rs:2141-2148`).
  Boxes with different core counts (r7900x 24C / i265 20C / r3500 6C, in the
  measured run) partition the identical job gap into differently-shaped chunks, so
  their chunk-ids never collide and the lease has nothing to catch — confirmed
  directly: r7900x logged `done=8415 … skipped=0` for its own pass 1 (all 8,415
  cells, alone), i265 logged the identical tally for its own concurrent pass 1.
  Two fast boxes each independently executed the ENTIRE manifest. **This is
  DIFFERENT and more severe than the documented lease-mode race in
  `docs/RUNNING_JOBS.md` §5b** (the "avifgen … 3.6x" figure — likely measured on a
  homogeneous fleet where same-shaped chunking WOULD collide) — this is a
  structural non-collision, not a milder stale-view race. **Every household LAN
  fleet is heterogeneous by construction**, so this bug is maximally relevant to
  exactly the fleet the Nomad-migration mission targets. NOT fixed here — likely
  fix directions: (a) derive chunk boundaries from a canonical box-independent
  partition so chunk-ids are comparable across boxes, or (b) epoch-sharded
  claiming (`ZEN_CLAIM_MODE=epoch-sharded`, shards by rendezvous-hashing
  individual cells, independent of any box's `BoxBudget`) — untested against this
  exact failure mode as of this writing; see the benchmarks doc for the rerun
  once done. Secondary finding along the way: `LedgerRow.ts` is `ctx.now`, set
  ONCE per `run()` call (one pass) and stamped on every row that pass writes
  (`lib.rs:970,1178`) — NOT a per-cell completion timestamp; don't derive
  per-cell timing from raw ledger `ts` values.

- **`zenfleet-worker`'s `tests::exec_command_reads_stderr_class_marker` flakes under
  concurrent/high system load — 100% stable in isolation (found 2026-08-24, during the
  Nomad-migration P0 precondition landing).** Observed exactly twice, both times in a
  full `cargo test -p zenfleet-worker --lib` run executed back-to-back with several other
  heavy `cargo check`/`cargo test`/`cargo clippy` invocations from the same session (i.e.
  under real ambient box load), never once across ~10 other full-suite runs on the same
  commits, and never in isolation (`cargo test ... tests::exec_command_reads_stderr_
  class_marker`, 3/3 clean). The test writes a small shell script to a per-test `tmp()`
  dir (uniquely named via an atomic counter + pid, so it is not a naming collision) and
  asserts a specific `ErrorClass::DiskFull` classification from its stderr marker. Not
  investigated further — unrelated to the JobId/VRAM/SIGTERM/claim-CAS changes landed
  the same session (verified: the same flake pattern would need to implicate shared
  process-spawn timing under load, not any of that logic). Re-run the full suite a few
  times before trusting a single red run from this test specifically.

- **master CI was fully red from at least 97acd176 (2026-07-29) through d5dacbd2
  (2026-08-27), independent of code pushes.** The 2026-07-29 causes (a stale
  `zenavif-parse` sibling path + two unformatted `kernel_tiers.rs` benches) were
  fixed in between; from `9093cc23` on, every job died at manifest load instead
  with `failed to read …/zenflate/Cargo.toml` (the `hdr` feature's new zenflate
  path-dep had no CI sibling clone). The zencodec 0.1.26 migration commit
  (2026-08-27) adds that clone, bumps every CI sibling pin to the 0.1.26-era
  sibling HEADs, and clears the stable-1.98 clippy debt the month-dead Lint job
  had been hiding — see CHANGELOG `[Unreleased]`. **Rule this teaches: every
  new `path = "../../../<sibling>"` dep in a workspace manifest needs a matching
  `git clone` in ALL FIVE copies of ci.yml's "Clone sibling-repo path-dep
  targets" step in the same commit**, or CI dies at `cargo metadata` before any
  job runs.

- **zenmetrics-api `it::backend_resolve::resolve_auto_host_and_force_no_gpu`
  fails on DEFAULT-features runs — test const out of lockstep with the
  library** (confirmed 2026-07-31, pre-existing: identical failure at master
  b5ec5a9c and after the #37 fix, and fails even serially). The test's
  `EXPECTED_NO_GPU` const is `Backend::CubeclCpu` when no `cpu-*` feature is
  compiled (tests/it/backend_resolve.rs), but `capability::cpu_fallback_backend`
  was later changed to return native `Backend::Cpu` in BOTH cfg branches (the
  "cubecl-cpu is NEVER auto-dispatched" PROJECT RULE), so the forced-no-GPU
  assertion `resolved_forced == EXPECTED_NO_GPU` compares `Cpu != CubeclCpu`
  and fails on any `cargo test -p zenmetrics-api --test it` with default
  features. Green on `cpu-metrics`-enabled runs (both sides = `Cpu`), which is
  presumably why CI never sees it. Fix = update the test's no-cpu-features
  const to `Backend::Cpu` (a test-expectation change — get user confirmation).

- **zenmetrics-api `it` suite: `session_cap::allocator_cap_recycle_leak` +
  `session_owned_cap::owned_into_metric_respects_cap_and_recycles` fail under
  PARALLEL one-process runs** (confirmed 2026-07-31 — identical failure pair at
  master b5ec5a9c and after the #37 fix; both PASS with `--test-threads=1`).
  Their global-ALLOCATOR slot-count assertions race sibling session tests'
  live slots (observed counts drift run-to-run: 91/108 vs expected 128, 5/8 vs
  expected 1). Same one-process family as the self-poisoning entry below.

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

- **r3500's Nomad client registration failure (`Node.Register` → `Permission
  denied`) — root-caused and FIXED 2026-08-25, homefleet commit `f6546ce355af`
  (`NODES.md`), independently re-verified live in this session.** Cause:
  `state.db` retains a cached node-identity JWT tied to the OLD `node_id` when
  only `client-id`/`secret-id` are wiped; the client then authenticates with
  that stale cached identity instead of the freshly-minted introduction token,
  so `Node.Register` fails with an expired-token permission denial regardless
  of how fresh the new intro token actually is — this is why i265's identical
  config eventually worked (no stale `state.db`) while r3500's repeated wipes
  never touched the one file that mattered. Fix: wipe `client-id`, `secret-id`,
  `state.db`, AND `intro_token.jwt` together, then mint + deploy a fresh intro
  token and start. Verified stable there across 2 restart cycles + 60s idle
  heartbeating. **Re-confirmed independently this session** (not just trusting
  the fix commit): `nomad node status` shows r3500 (`d74a82bb`) `ready`/
  `eligible`, and a fresh SSH check moments later shows its `nomad` systemd
  unit `active (running)`. r3500 is back in the fleet for Nomad-scheduled work.

- **Root cause found for the "batch job restarts forever after a clean idle-drain"
  half of the `fleetbench-gpuscore` (warm-exec) incident above — FIXED 2026-08-25
  in `homefleet` (commit `a176613482b1`), NOT in this repo.** The working theory
  in this doc ("the Nomad `restart` stanza treating a clean-but-nonzero exit as a
  failure") is **DISPROVEN by live testing**: `fleet-entrypoint.sh`'s own exit
  code on the `idle >= ZEN_IDLE_PASSES` drain-and-`break` path is verified **0**
  — reproduced twice by running the actual, unmodified script (mocked `aws`/
  `s5cmd`/`zenfleet-worker` so it drains fast) as PID 1 of a real Docker
  container under a real Nomad batch job with the exact `restart{attempts=2,
  mode="fail"}` stanza these jobspecs use: both times `Client Status=complete`,
  `Exit Code: 0`, `Total Restarts: 0`. The **actual** mechanism, also confirmed
  live: every one of these jobspecs' `template` blocks reads a Nomad Variable
  shared across ~20 jobs (`nomad/jobs/zenfleet-worker-pilot`, the R2 creds) and
  none of them set `change_mode`, so Nomad's **default `change_mode = "restart"`**
  applies — whenever that variable's rendered content changes (a credential
  rotation via `nomad var put`, per `homefleet/zenmetrics/ORCHESTRATION-2026-08.md`),
  Nomad SIGKILLs the task and starts a fresh attempt, **counted against the same
  `restart{}` policy**, regardless of whether the task was mid-work or sitting
  idle-draining. Verified live twice: default `change_mode` → updating the
  shared var mid-run produced `"Restart Signaled: Template with change_mode
  restart re-rendered"` → `Terminated Exit Code: 137` → a brand-new task attempt
  (fresh PID 1, `Total Restarts` incremented) — the exact "fresh pass 1, idle
  reset to 0" symptom: with `change_mode = "noop"` instead, the identical
  rotation re-rendered the secrets file (confirmed via `nomad alloc exec`) but
  left the running task untouched, and a full end-to-end run of the real
  `fleet-entrypoint.sh` (8 configured idle passes, credential rotated mid-run)
  completed every pass and reached `Client Status=complete / Exit Code=0 /
  Total Restarts=0`. **Fix (homefleet, not this repo):** `change_mode = "noop"`
  added to the shared template block in all 19 `type="batch"` jobspecs under
  `zenmetrics/ubuntu-node/nomad/jobs/`; the one `type="service"` job sharing the
  pattern (`zenfleet-worker-pilot.nomad.hcl`) intentionally keeps the default
  restart-on-rotation behavior since a long-lived worker legitimately should
  pick up rotated creds. This does not touch anything in this repo — no code
  or jobspec here needed to change; `fleet-entrypoint.sh`'s exit-code handling
  was already correct. The other half of the incident (5/4,000 jobs never
  resolving to a terminal ledger row) remains open, see "Known Bugs" above.

- **`dev` (one of the 3 Nomad servers, `node_class=always_on` + docker-driver
  client per its own config) had no Docker installed — found and FIXED
  2026-08-25.** `systemctl is-active docker` → `inactive`; `docker` → `command
  not found`; every docker-driver job constrained to `dev` failed to place
  ("missing drivers" filter). Fixed via the official `get.docker.com` install
  script (Docker Engine 29.7.2) + `systemctl restart nomad` to pick up the
  newly-available driver. Verified live: a `jxl-lossy-dense-dev` job that had
  previously failed to place now placed and ran successfully
  (`Client Status = running`) after the fix.

- **SIGTERM chunk-claim release did NOT actually release the claim in a real
  end-to-end test; the P0 precondition reported "DONE, verified" had only ever
  been verified against a synthetic/mocked harness — found AND resolved
  2026-08-24 (G-P2 gate testing — `benchmarks/fleetbench_2026-08-24.md` has the
  full investigation arc).** A real `nomad node drain -force` against a live
  `zenfleet-worker` allocation with a confirmed live chunk claim (read back
  directly from the LAN store before draining) exited the container promptly
  and cleanly (exit 143 = SIGTERM, ~1.8s) and the bash entrypoint's trap
  correctly fired and forwarded the signal — but the S3 claim object was read
  back UNCHANGED, and neither the release-attempt log line nor `run_chunked`'s
  own unconditional first print appeared anywhere in the logs. Narrowed via a
  raw (no Docker/Nomad) reproduction: the release LOGIC itself worked correctly
  in isolation (confirmed claim deletion after a direct `kill -TERM <pid>`),
  proving the bug was specific to the Docker+Nomad+entrypoint layering, not the
  core algorithm. **Fix: added an immediate marker + explicit `stderr` flushes
  to `spawn_spot_reclaim_chunk` (lib.rs) and sub-second timestamps around
  `fleet-entrypoint.sh`'s `wait "$PASS_PID"`, rebuilt the executor image (musl
  target) and re-tested — 2/2 real repros then showed the claim genuinely
  deleted** (bash-level pre-wait→post-wait timing: 3-5ms). **Honest residual
  puzzle, not fully explained:** even in the 2 successful runs, the NEW
  Rust-level diagnostic lines still never appeared in `nomad alloc logs`,
  despite the claim's deletion proving that exact code path executed — so the
  underlying mechanism (did the extra flush calls fix a genuine scheduling
  race, or was something else going on) is not conclusively understood, only
  that the fix is repeatable. **A separate exactly-once check (does the
  released chunk's work get redone by exactly one other worker) was attempted
  but confounded by the already-documented lease-mode heterogeneous-chunking
  bug** (this test ran in lease mode, not epoch-sharded) — not cleanly
  isolated; needs a dedicated epoch-sharded rerun. Treat the release mechanism
  as re-verified-but-not-fully-understood: safe to build on for now given the
  repeatable positive evidence, but re-test before leaning on it harder (e.g.
  before G-P3's full autoscale cycle). `crates/zenfleet-worker/src/lib.rs:246-279`
  (`spawn_spot_reclaim_chunk`), `fleet-entrypoint.sh`'s `run_pass`/`on_term`.
  **UPDATE (found later same day, during the G-T1 VRAM-admission investigation
  below): the "residual puzzle" above is now fully explained, not just newly
  true elsewhere.** `run_pass()` (`fleet-entrypoint.sh:114-126`) redirects the
  whole pass's stdout+stderr to an `mktemp` file, `rm -f`'d right after `wait`
  returns; the caller only forwards specific extracted lines to the container's
  real stdout (the done/failed summary always; the FULL output only on pass 1
  or a non-zero exit). A custom `eprintln!` that doesn't match one of those
  forwarding rules is genuinely invisible in `nomad alloc logs`/`docker logs`
  WHILE the pass is running, but surfaces once the pass ends if it's pass 1 or
  fails. To read a still-running pass's raw output: `sudo cat
  /proc/<zenfleet-worker-pid>/fd/2` on the host (the file is unlinked but the
  open fd still reads). This isn't a fix, just the explanation — anyone hitting
  "my eprintln never shows up" should check `/proc/PID/fd/2` before concluding
  the output is lost.

- **VRAM admission (`BoxBudget::can_admit`) was WRONGLY believed broken for
  part of one session (2026-08-24, G-T1 ladder pass) — a live GPU-score run's
  `nvidia-smi --query-compute-apps` reading (16 concurrent PIDs against a
  predicted ~2-3 ceiling) was reported as "P0 precondition re-opened,
  `can_admit` does not bind" and briefly pushed to master/main before being
  corrected in the same session.** Rather than leave "needs dedicated tracing"
  as follow-up, added `ZEN_DEBUG_ADMIT=1`-gated `eprintln!`s logging
  `can_admit`'s actual `cand_vram`/`running.count`/`running.vram_bytes`/
  `budget.vram_budget_bytes` at its call site, rebuilt the GPU exec image
  (`:exec-gpu-vram-debug` — reusing the already-proven `zenmetrics` binary
  extracted via `docker cp` from the deployed image, since a fresh local
  build had drifted to a newer glibc than the GPU_BASE rootfs supports), and
  re-ran the EXACT same 500-job scenario plus a smaller 250-job confirmation.
  **Result: 750/750 real admissions across both runs never exceeded
  `count=2` — `can_admit` was correct the whole time.** `nvidia-smi
  --query-compute-apps`'s PID count (independently re-polled on the same
  reruns: 0-26, no correlation with the ground-truth count) is not a
  reliable proxy for a scheduler's admitted concurrency on this GPU-metrics
  workload (5 metrics scored per cell per process — plausibly transient
  multi-context registration or driver-level teardown lag per process, not
  confirmed further since it's irrelevant to the conclusion). **Correction
  applied same session**: the benchmarks doc, this file, and the ADR were all
  updated to remove the "re-opened" framing — do NOT treat any earlier
  "VRAM admission re-opened" statement (in git history, in a stale local
  checkout, or in a prior context-window summary) as current; the corrected
  state is "measured correct." Full writeup:
  `benchmarks/fleetbench_2026-08-24.md`'s G-T1 VRAM-admission section (kept
  deliberately in first-conclusion-then-correction form, not rewritten to
  hide the wrong first pass).

- **Two real bugs found and fixed in `scripts/jobsys/fleet_power.py` while
  chasing the above (2026-08-24, same G-P2 session):** (1) `suspend()`'s
  `nomad node drain -enable -deadline 2m` (no `-force`) does NOT promptly
  interrupt a running allocation — confirmed live, an in-flight allocation kept
  running (and even claimed a SECOND chunk after the drain command was issued)
  for ~70s until it happened to finish its own work naturally; `-force` is now
  added, since without it "claim released promptly...seconds, not claim-TTL"
  cannot hold. (2) `cmd_apply` called `suspend(box["ssh"], args.nomad_addr,
  None)` — the `node_id` param was hardcoded `None`, so the drain branch
  inside `suspend()` never ran in the real code path at all; added
  `resolve_node_id()` and wired it through. (3) related, found by direct
  observation: a drained node stays `Eligibility = ineligible` FOREVER after
  the drain completes (separate flag from `Drain`) — a box that went through
  one sleep cycle would wake up, rejoin the cluster, and then sit idle forever
  as far as Nomad scheduling is concerned; added `re_enable_eligibility()`,
  called every `cmd_apply` tick for any reachable box (idempotent,
  self-healing). Gotcha hit while fixing (1): Nomad's CLI rejects `-force`
  combined with `-deadline` outright ("can't be combined") — the naive first
  fix (add `-force` next to the existing `-deadline 2m`) fails at runtime, and
  with `check=False` on the subprocess call this would have been COMPLETELY
  SILENT in production. **Caught twice, the second time for real**: validated
  the correct shape (drop `-deadline`) via raw CLI testing and wrote it up as
  fixed — but the actual code edit that day only ADDED `-force` next to
  `-deadline`, never removed it, so the bug shipped anyway; it resurfaced
  live during the first real G-P3 test (`suspend()` silently failed on both
  target boxes). Fixed for real this time (`1cf5003a`) — a docs/commit-message
  claim of a fix is not the same fact as the code containing it; re-read the
  file (or re-run the exact call) before writing "fixed".

- **JobId was `serde_json/preserve_order`-SENSITIVE — the core golden test failed in any
  build that co-compiled zenmetrics-cli (found 2026-08-06) — FIXED 2026-08-24
  (`1b2a1452`).** `JobId::of` serialized `serde_json::json!({"kind":…,"inputs":…})` — a
  `Value`, whose map ordering flips from alphabetical (BTreeMap) to insertion-order
  (IndexMap) when ANY crate in the build enables `serde_json/preserve_order`.
  zenmetrics-cli enables it deliberately (compare-output ordering), and both
  `zenfleet-sweep` (hard dep) and `zenfleet-vastai` (default `inline-sweep` feature)
  pull zenmetrics-cli in. Consequence: `cargo test -p zenfleet-core -p zenfleet-sweep`
  (or `-p zenfleet-vastai`) failed `job::tests::scorefile_sdr_serialization_and_job_id_
  are_golden_stable` (`797c1300…` vs golden `6e79bec2…`) — the latent hazard was real: a
  binary built in a combined invocation that includes zenmetrics-cli computed DIFFERENT
  JobIds than a solo-built `zenfleet-worker` — cross-binary dedup/claims/ledger joins
  would have silently missed. Fix: `JobId::of` now hashes a plain `#[derive(Serialize)]`
  struct instead of a `Value` — struct field order is fixed at compile time by
  declaration order and is not sensitive to `preserve_order` at all. Field order chosen
  to reproduce the historical alphabetical byte output exactly, so no existing JobId,
  ledger row, or manifest changed — verified golden-stable under both a plain build and
  one co-compiled with zenmetrics-cli. **Combined worker/ctl + zenmetrics-cli builds are
  no longer required to avoid this** (the old workaround — never build them together —
  is no longer necessary, though harmless if still followed).

- **POOL-mode sweep RE-SCORED already-done cells on RESUMED runs — the pool pass passed no
  `--ledger-in` — FIXED 2026-07-21 (`1ce1fd7a`).** `fleet-entrypoint.sh` POOL mode passed only
  `--ledger-out` + claims, so the worker's reconcile *view* was empty and its gap was *every cell
  each pass*; dedup fell entirely to the 10-min claim lease. On a run whose earlier claims had
  expired (e.g. a prior fleet's work being resumed) it re-scored done cells — a **measured 2.0–3.4×
  waste** on the big runs (jpeg run: ~79k wasted re-score rows). Fix: the pass fetches a per-run
  done-set snapshot (`jobs/<run>/ledger_snapshot.parquet`) and passes it as `--ledger-in` so the gap
  = only-undone (guarded — a run with no snapshot runs the old way). Snapshots are compacted from the
  worker's own ledger rows (schema matches `read_ledger`) by
  `scripts/jobsys/{compact_ledgers.py,refresh_snapshots.sh}` on an **hourly cron on the cred-holding
  box** (core-pinned to cores 24+ so it can't steal the co-located worker's cores). `pool_progress.py`
  = fast distinct-done readout from snapshot footers. Validated 1:1 row growth on the zensim-720
  backfill. **If you launch a pool sweep that RESUMES prior work, seed snapshots first** (run
  `refresh_snapshots.sh` once) or it re-scores the existing progress.

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
