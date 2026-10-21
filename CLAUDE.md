# zenmetrics CLAUDE.md

## September 15: explicit common-primary native HDR scoring

[Contract and validation](docs/HDR_COMMON_PRIMARIES_2026-09-15.md). Use
`score-pairs --hdr --hdr-common-primaries` for the corrected declared-PQ
BT.709/P3/BT.2020 metric ingress. Legacy flags retain their historical era.
CPU CVVDP now has an explicit native route; the display is still BT.709.
Do not relabel old 8-bit-shell outputs or feature sidecars as this contract.


See global ~/.claude/CLAUDE.md for general instructions.

### Known Bugs: HDR-VDP v2 original-call diagnostic difference (2026-10-20)

The partial [original HDR numerical grid](benchmarks/canonical_recheck_2026-10-19/hdr-original-partial12.json)
records `hdrvdp` versus the unchanged official sRGB-display call differing by
0.009351851113351017 on the 1024×1024 photo at both one and 28 workers. This
exceeds the retained 0.005 golden-check level; it does not establish a universal
bound violation or a new Rust regression. The local reference uses Octave 6.4 /
image 2.12 with two reviewed image 2.20.1 resize M-file overlays. Full original
versus direct-call controls, runtime qualification and parameter-storage scope
are incomplete. Root cause is unresolved; native parity and all timing ratios
remain unqualified. No metric code or numerical expectation was changed.

## Crate naming (owner decision 2026-10-11)

- **Image quality metrics we publish first and canonically keep bare names**:
  `zensim`, `butteraugli`, `fast-ssim2`, and the metric ports here (`cvvdp`,
  `iwssim`, `vmaf`, `gmsd`, …). A GPU twin is `<metric>-gpu`.
- **Shared infrastructure is `zenmetrics-*`** (`-api`, `-cli`, `-orchestrator`,
  `-gpu-core`, `-math`, `-corpus`), and so is the evaluation crate:
  `zenmetrics-stats` (renamed from `zenstats`).
- **`dvifmish` may never be published**: it is an independent reimplementation
  reconstructed from a public talk (the author has not released code), and
  publishing it needs the original author's consent. Its manifest says
  `publish = false` (dvifmish `0fa38910`).
- `crates/margarine` is package `margarine` (the `-lab` suffix added when it
  moved here on 2026-09-27 had no reason and was dropped on 2026-10-11).
- `zen-iqa` is a third-party project; do not use `zeniqa`-like names.

## VMAF CPU optimization discipline

For future pure-Rust VMAF performance work, inspect generated code with
`cargo-show-asm` (or equivalent disassembly) before claiming SIMD or
autovectorization. Shape hot loops around contiguous data and predictable
bounds; separate interior work from border handling where useful, without
changing fixed-point rounding or overflow behavior. Use representative PGO
evidence to justify inlining, cold-path, branch, or layout annotations rather
than applying them speculatively. Measure scratch allocations and reuse,
cache locality, access strides, memory traffic, tiling/chunk sizes, and
threading overhead. Reprofile before choosing a bottleneck, then gate changes
on libvmaf feature/frame/NEG parity and scalar, SIMD, and threaded
end-to-end benchmarks; retain only measured improvements.

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

### Push through `scripts/safe_push.sh` — a bare `jj git push` clobbers other lanes

`jj bookmark set master -r @ && jj git push --bookmark master` is a
**non-fast-forward push with no prompt and no warning** whenever `@` does not
descend from `master@origin`: the bookmark moves SIDEWAYS and every commit only
reachable from the old tip becomes unreachable. Nothing errors. MEASURED in the
sibling repo on 2026-09-04 — nine commits from six lanes dropped in one
afternoon (zensim's `benchmarks/push_clobber_2026-09-05.md`); the mechanism is
jj's, not zensim's, so it applies here identically.

`scripts/safe_push.sh` is a MIRROR of zensim's, defaulting to `master`:

    fetch -> assert <bookmark>@origin is an ANCESTOR of the target
          -> hygiene: address/identifier check on the outgoing diff
          -> bookmark set -> push -> verify it landed

**Exit 3** = not a fast-forward; it prints every commit the push would drop and
does not touch the bookmark. **Exit 9** = the dashboard guard: a push that
changes `benchmarks/` or `scripts/dashboard/` must still build the dashboard from the
outgoing tree (register or classify new benchmark files first). **Exit 7** = the second gate, `hygiene:
address/identifier check`, which scans the lines the push would ADD for the
classes that must not enter a public repo (private-range addresses, hardware
addresses, household framing). Neither gate has a bypass flag — the 2026-09-04
incident is what an escape hatch on a guard is worth.

Patterns have ONE owner, [`scripts/lib/hygiene_patterns.txt`](scripts/lib/hygiene_patterns.txt),
shared with `scripts/ci/check_hygiene.py`, which runs the same classes over
every tracked file in CI (`just hygiene-check`) so an INHERITED hit is found and
not only a newly added one. Site-specific patterns (names, labels) are NOT in
this repo: they load from `hygiene_patterns` in the private homefleet config
(`HOMEFLEET_NODES`, default `~/work/zen/homefleet/zenmetrics/fleet/nodes.toml`),
because a public pattern list that spelled out the values it protects would leak
exactly what it exists to keep out.

Self-tests, both with negative controls: `scripts/safe_push.sh --self-test`
(6 cases — a sideways target REFUSED with the remote provably unmoved, and the
canonical form of the same line ACCEPTED) and
`python3 scripts/ci/check_hygiene.py --self-test` (16 assertions).

## Cargo.lock — never regenerate it in a dev tree; use `scripts/ci/lock.sh` (enforced)

`Cargo.lock` records every sibling PATH package (zenavif, jxl-encoder, zenjpeg, …) with its
version and dependency list. A lock regenerated by running cargo in a dev tree resolves against
whatever siblings are checked out there and cannot be reproduced by CI's clones: on 2026-09-25/26
the lock flipped between the zenavif 0.2.0 / jxl-encoder 0.4.0 state and the 0.1.7 / 0.3.2 state
and every CI job died at dependency resolution. The CI siblings are pinned in ONE file,
`ci/sibling-pins.tsv` (read by `scripts/ci/clone-siblings.sh`, which every ci.yml job runs).

    scripts/ci/lock.sh --check     # does Cargo.lock resolve unchanged against the pinned siblings?
    scripts/ci/lock.sh --regen     # re-resolve it there (isolated snapshot), write it back, re-check

To build or test on a dev box whose sibling checkouts have drifted from the pins (a plain
`cargo` there can fail to resolve; on 2026-10-11 dev's `ultrahdr` asked for an unpublished
registry `zenjpeg ^0.9.0`, now redirected by the root `zenjpeg` patch), run cargo in the pinned
snapshot: `just snap-cargo @ test -p <crate>`. A plain `cargo` in a dev tree rewrites
`Cargo.lock` to match the local siblings: never commit that; `jj restore Cargo.lock`.

`snap-cargo` runs in a snapshot copy, so `snap-cargo @ fmt` would format the copy and leave
the tree unformatted (2026-10-15: that is how 48e768ed reached master unformatted). The recipe
refuses `fmt` without `--check`; format the tree with `just fmt`.

`scripts/safe_push.sh` refuses (exit 8) any push that changes `Cargo.lock` or `ci/sibling-pins.tsv`
unless `lock.sh --check` passes on the outgoing commit. If a pin must move, edit
`ci/sibling-pins.tsv`, run `--regen`, and commit both together.

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

Whenever a fleet is up (vast.ai / Hetzner / RunPod / Salad / on-prem), every box
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

## burn: GPU-metric kernels ABANDONED ≠ training ≠ NN metrics (one graph is fine; no shared types)

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
off **data, not tensors**.

**Coexistence — MEASURED 2026-10-14; replaces the earlier "never share one cargo
graph / do NOT add burn to this workspace" rule.** burn 0.22 (upstream cubecl
0.11, wgpu 30) and the zenforks-cubecl crates (0.10.2, wgpu 29) build, link and
run in ONE binary: `benchmarks/nn_runtime_shootout_2026-10-14/coexist-probe/`
interleaves burn matmuls with `ssim2-gpu` on CUDA and on wgpu against the same
RTX 4080, and the ssim2 score was bit-identical across rounds and runs. The
rename only means burn cannot *use* the fork: no tensors, handles or clients
cross between them, only host data. Costs: two GPU runtimes, memory pools and
kernel caches per process (VRAM budgeting must count both), two wgpu versions,
longer builds. Both sides resolve to ONE `cudarc` (0.19.10, same feature set
with or without burn as of the probe): re-check
`cargo tree -i cudarc -f '{p} [{f}]'` whenever either side bumps. burn's deps
are crates.io, so burn-based crates stay publishable; anything depending on the
git-pinned fork does not. Autodiff *through* a zenforks kernel (differentiable
metrics) is still impossible in this arrangement. Owner direction 2026-10-14:
the NN metric crates live in this workspace — `docs/PYIQA_PORT_PLAN.md`.

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
- **Feature-combination gate**: `just cli-feature-matrix` (CI twin: the `cli-feature-matrix` job in
  `.github/workflows/ci.yml`) check-builds all 12 zenmetrics-cli shapes (10 `--features` combos +
  bare + default) and prints a PASS/FAIL table — run it after touching CLI cfg/feature gates; keep
  the justfile and CI lists in sync.

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

## Metrics in an ENCODER SEARCH LOOP — CPU warm-ref, never GPU (MEASURED 2026-08-30)

A quality-search loop scores ONE reference against N candidates. That is a
different problem from fleet scoring and it has the opposite answer. Full
numbers, provenance and the decision function: `benchmarks/metric_loop_2026-08-30.md`
(+ `.meta`, `metric_loop_{st,mt}_2026-08-30.tsv`). Harness:
`benchmarks/heaptrack/drivers/cpu_profile/src/bin/loop_wall.rs` (bin `loop-wall`)
— it times the **planar linear f32** entry points, which is what jxl-encoder's
`vardct/{perceptual,zensim,ssim2}_loop.rs` actually use; `cpu-wall` times the
packed-sRGB-u8 sweep shape and answers a different question.

- **All three metrics already have a CPU warm-reference API** — `butteraugli::
  ButteraugliReference`, `zensim::Zensim::precompute_reference{,_linear_planar}`,
  `fast_ssim2::Ssimulacra2Reference`. Nothing is missing; do not go build one.
- **Warm reference pays back at N ≥ 2 for every metric at every size** (64²–4096²).
  Use it unconditionally. `T(N) = precompute + N × per_candidate` is accurate to
  −2.9…+1.8 % against a measured 5-candidate loop.
- **Warm/cold ratio is FLAT across three orders of magnitude of pixel count** —
  butteraugli 0.61, SSIMULACRA2 0.58, zensim 0.92 (ST) / 0.78 (MT). One number
  per metric predicts the saving at any size.
- **zensim's warm saving is structurally capped ~10–22 %**, not a bug: its warm
  reference hoists only the ref XYB + pyramid, because `fused_blur_h_ssim`
  (`zensim/src/blur.rs:2289`) deliberately fuses `blur(src)`/`blur(dst)`/
  `blur(src·dst)` into one pass. Don't chase it in the shipped v1 path.
- **Whether GPU pays off depends on the GPU's MEMORY ARCHITECTURE, not on image
  size or candidate count.** Candidates needed on ONE image before GPU beats
  this host's 12-core CPU:
  **Metal / Apple integrated (measured here): N\* = 2–4 at >=512^2**, because
  setup is only 45–60 ms at 1 MP (no NVRTC, no discrete upload).
  **Discrete CUDA (RTX-5070, quoted from `gpu_coldstart_2026-05-29.tsv`):
  N\* = 14–299.** Either way N\* = **1** if the instance is reused across
  images. jxl-encoder runs **3 compares at e8, 5 at e9/e10**, 18 at e11 — so a
  unified-memory GPU already wins at e8 above 1 MP, while a discrete one does
  not until e12. Below 0.25 MP no GPU wins for any metric. Driver for the Metal
  numbers: `crates/zenmetrics-api/examples/loop_gpu_probe.rs`.
- **Threading (aarch64 numbers — x86 NOT MEASURED)**: on this M4 Pro,
  butteraugli 2.83× on 12 cores, zensim 4.49×, **SSIMULACRA2 1.02× (it does not
  parallelise as shipped)**. `fast-ssim2`'s non-default `rayon` feature measured
  1.00–1.04× at ≥256² and **0.31× (3.2× SLOWER) at 64²** — so leaving it off is
  right *here*, which is what every consumer already does.
  **Do NOT generalise this to x86.** The owner reports fast-ssim2 used to be much
  faster with more threads on x86; different core count, memory bandwidth and
  SIMD width, and extrapolating across architectures is banned. Re-run the A/B on
  the 7950X before concluding anything for x86.
- **zensim below ~128²: force single-threaded.** Its multi-core dispatch
  intercept (0.332 ms) exceeds an entire 64² compare (0.249 ms ST vs 0.315 MT).
- **Stride: ALL THREE take it correctly and for free.** butteraugli and zensim
  take an arbitrary `stride` on both sides; fast-ssim2 takes it through `ImgRef`
  (`impl ToLinearRgb for ImgRef<'_, [f32; 3]>` walks `self.pixels()`,
  `fast-ssim2/src/input.rs:261-266`). All bit-identical to the tight path (parity
  delta exactly 0 at all 7 sizes). Measured strided÷packed for fast-ssim2:
  **0.984–1.008** across 64²–4096², both thread configs.
  **CORRECTION 2026-08-30: an earlier version of this section said fast-ssim2
  violated the stride rule. That was WRONG** — it came from reading
  `ToLinearRgb` and never running a strided input (the harness pre-flattened to
  a packed `Img`). Reading an implementation is not measuring it.
  What fast-ssim2 actually lacks is a **planar** entry point, so a *planar*
  caller must interleave: **4.1–8.8 %** of the per-candidate wall (0.8.2). An
  *interleaved* caller pays zero. `Ssimulacra2Reference::new_linear_planar` is
  the ONE missing API.
- **fast-ssim2 version trap — the graph carries TWO.** `[patch.crates-io]` pins
  local **0.8.2** for anything requiring `^0.8` (all zenmetrics crates), but
  **jxl-encoder requires `^0.7.1` and silently gets the REGISTRY 0.7.1** — a
  patch only applies to requirements its version satisfies, and because the
  patch IS used elsewhere it never appears in `[[patch.unused]]`. Consequence:
  `CompareContext`/`compare_with` landed in **0.8.1**, so **jxl-encoder cannot
  call them today**; bumping its dep to `^0.8.1` is a prerequisite for the
  zero-alloc recommendation. **Anything benchmarking against this workspace MUST
  record the RESOLVED version** (`cargo tree -p <pkg> -i <crate>`), not the
  declared one — `scripts/loopbench_sweep.sh` writes a `.versions` sidecar per
  run. Detail: `benchmarks/metric_loop_2026-08-30.md` §6.
- **Batching (`ssim2-gpu::Ssim2Batch`) beats sequential warm-reference calls from
  N=2 and scales further on Metal than CUDA: 1.95x/2.86x/5.65x/**7.24x** at
  N=2/4/8/16 (256^2) vs CUDA's 2.00/2.84/3.30/3.58. At N=1 it is a 5% LOSS —
  gate on N>=2. `butteraugli-gpu` has `ButteraugliBatch`; **`zensim-gpu` has no
  batch API**; neither batch type is reachable from the opaque layer.
  Inapplicable to jxl-encoder's loops as written — they are sequential by
  construction (candidate i+1 depends on candidate i's score).
- Report **α and β separately** for anything in this space: fitting
  `α + β·MP` over the whole 64²–4096² ladder returns a *negative* intercept for
  every metric (unweighted OLS tilted by the 16 MP point). Fit α on 64²–512² and
  β on 1024²–4096².

## Known Bugs

- **SSIMULACRA2 GPU small-input scalar discrepancy (2026-10-20).** Three
  retained 64×64 photo blocks each return GPU 84.8498333349087 versus primary
  C++ v2.1 90.86388011956463, absolute delta 6.014046784655932. The same
  observations occur at thread budget 28. Source inspection finds different
  scale selection: the GPU allocates 64/32/16/8, while the primary checks its
  previous size before downsampling and also evaluates 4. This structural
  difference is verified; the complete numerical attribution is not. The
  discrepancy is absent in the ordinary CPU route: its fresh current40-worker
  control returns exactly 90.86388011956463 on the same input pair.
  GPU pyramid source is unchanged from the measured worker. No case bound was
  assigned, no tolerance changed and no metric fix landed. Source hashes, raw
  scores and resource receipts: `benchmarks/canonical_recheck_2026-10-19/ssim2-gpu-small-scalar-finding.json`.

- **Linux durable recovery fixture selected a different cell (2026-10-20).**
  The pending-launch fixture selected the first filesystem directory but
  asserted against the first reported cell. Directory iteration order can
  differ from manifest order. Selecting the reported cell identity fixes the
  fixture; a forced reverse-order control fails before and passes after with
  the original two-attempt assertion unchanged. Production recovery logic
  and its acceptance checks are unchanged.

- **Direct classical benchmark module resolution (2026-10-20).** The
  MS-SSIM benchmark includes the CLI classical module by path; rustfmt then
  sought its PSNR child beside the parent instead of in `classical/`.
  An explicit child path selects the existing kernel source for both the
  normal module and direct benchmark include. The formatter reproduces the
  missing-file failure before this annotation and passes after it. No metric
  arithmetic or numerical expectation changed; no local binary rebuild was
  performed for this source-path correction.

- **NR AVX-512 feature build exposed missing generated arms (2026-10-19).**
  Enabling NIQE/PIQE `avx512` called three `_v4x_t` functions omitted by the
  default `attune(_*_t)` expansion. The integration explicitly generates those
  arms while preserving their existing lane layout and arithmetic. The combined
  all-feature crate tests are the build regression gate; final verification is
  recorded with the snapshot evidence.

- **Checkpoint portability gates fail on the frozen 53f6d26d source (2026-10-19).**
  [Run 39275363269](https://github.com/imazen/zenmetrics/actions/runs/39275363269)
  fails the common BMP alpha tests on all seven native configurations; the
  owner-approved test correction is included in the integration. macOS ARM64
  additionally fails SSIMULACRA1 and FLIP numerical pins; Windows x64 fails the
  FLIP-v17 pin (observed 0.07714098691940308, expected 0.07714099436998367 with
  unchanged 1e-9 allowance). The native math observation workflow records the
  same fixture outputs without accepting new values or bounds.
  The browser job refuses `ms-ssim-pyiqa-f32-k6` at the 1024 fixture: its fifth
  scoring view is 10×10, below the unchanged 11×11 requirement. The owner-approved test
  patch retains that refusal and adds a 1152 success fixture.
  These are checkpoint-publication blockers, independent of the 23-name rerun
  scope in `benchmarks/rebenchmark_since_f7668c7/README.md`.

- **MS-SSIM libvmaf threading gate needs a 4K regression recheck (2026-10-19).**
  The unmerged gate experiment records 227.459 ms before and 282.117 ms after
  at 3840×2160 / 32 threads, with identical score bits. Small-image improvements
  do not establish a win at this size. The integration retains the experiment
  and includes this name in the minimal rerun list; the current combined tree
  has not yet been timed. See `benchmarks/opt_msssim-libvmaf_2026-10-18/summary.tsv`
  and `benchmarks/rebenchmark_since_f7668c7/README.md`.

- **Checkpoint at `53f6d26d` fails BMP alpha-policy tests; macOS also fails two numerical pins (2026-10-19).** The full CLI checkpoint profile exposes two old BMP tests that assume implicit alpha dropping, while the current decoder defaults to `RequireOpaque`. Linux x64 WGPU/CUDA, Linux ARM64 and macOS ARM64 reproduce those failures. On macOS ARM64, `ssimulacra1::oracle_tests::original_author_synthetic_pin` also exceeds its unchanged `1e-12` allowance, and `flip_and_cw_ssim_match_pinned_reference_values` reports FLIP-v17 `0.07714100182056427` versus pinned `0.07714099436998367` under the unchanged `1e-9` bound. Those numerical differences are not yet attributed. Checkpoint publication remains blocked; no tag or gate adjustment has been made. Run: https://github.com/imazen/zenmetrics/actions/runs/39275363269

- **HaarPSI native map has nonfinite mismatches on a retained illustration pair (2026-10-19).** At 3840×2160, 41 positions at each of 1/28 threads return Rust NaN against a finite weighted mean of the original primary similarity/weight arrays. The primary's weight sums at those positions are between `2.1316282072803006e-14` and `3.410605131648481e-13`; the cause of Rust's NaNs is not established. Scalar results retain their earlier values and the existing scalar gate. Maps have no assigned external bound. Evidence and exact input/source bindings: [map observations](benchmarks/port_spotcheck_2026-10-18/haarpsi-maps.md).

  Rechecked 2026-10-20 against frozen master source `efae6b8fed9e57346157d92dffc3cdb57c44560e`: all 24 cases execute; the same 41 nonfinite mismatches persist at each thread count on the 4K illustration, with none in the other 22 cases. Scalar map/score APIs agree bit-for-bit in every case. [Current map evidence](benchmarks/canonical_recheck_2026-10-19/haarpsi-current-maps/haarpsi-maps.json).

- **VIFp RGB luma precision, found 2026-10-19:** the frozen e2d414c3
  `vif` route misses the unchanged 1e-9 author-reference gate in 22/24 raw
  size/content/thread cases (largest difference 4.093188010756421e-9).
  Its private luma coefficient array is f32 before promotion into the f64
  pipeline. A correction is under independent evaluation on unmerged
  `fix/vif-rgb-luma-precision`; do not treat it as qualified or merged.
  `vifvec` is separate and must retain its scores/refusals.
  [Original measurements and limits](benchmarks/port_spotcheck_2026-10-18/primary-vif.md).

- **VSI reference resize mismatch (2026-10-19), locally explained.** At
  source `e2d414c3`, `vsi` and `vsi-nodownscale` return 0.9567964257437082
  for a retained 64×64 S67/CCM15 crop; original author `VSI.m` returns
  0.94679144074028154 with image 2.12.0. Its resize behavior differs from
  the claimed antialiased convention. Selecting unchanged upstream image
  2.20.1 `imresize.m` and its private helper gives 0.95679610895892431:
  absolute difference 3.1678478384922215e-7 passes the unchanged 1e-4 bound.
  This is a two-M-file overlay on Octave 6.4, not a full newer package install.
  No metric code changed. Retain the initial observations and bind resize
  sources in new panels. Details: `benchmarks/port_spotcheck_2026-10-18/vsi-resize-runtime.md`.

- **2026-10-18 — Linux durable benchmark scope placement and ORT budgets.**
  A user scope's `AllowedCPUs` property can be present while the kernel
  cpuset remains unrestricted when the controller is not delegated. Each
  resource wrapper also creates a fresh scope. Durable Linux attempts now
  set their actual scope boundary and retain zenbench's checked placement
  proof (`9954fadd`). The first native CUDA control also exposed default ORT
  session pools despite thread environment variables; those timings stay
  diagnostic. The durable build uses the existing session-budget snapshot
  patch and the real control requires the emitted one/eight-worker budget
  (`aafa96b8`). All five corrected CPU/NN controls pass. See the
  [launch receipt](benchmarks/iqa_i265_durable_2026-10-18/README.md).

- **2026-10-18 — Mac warm IQA admission after interruption.** An interrupted
  NLPD warm attempt left a running zenbench reservation whose owner had exited.
  Later cold cells continued, while warm attempts refused admission and remained
  pending retry, so the exhausted-cell failure count alone concealed the blockage.
  The durable worker now records its PID/group before exec and archives only its
  matching dead reservation after group cleanup and ticket-guard acquisition.
  Live-owner/group and changed-owner controls remain fail-closed on both Unix
  process-inventory paths. See [the recovery contract](docs/IQA_DURABLE_MAC.md).

- **2026-10-18 ORT WebGPU on mac/Metal:** official plugin 0.4.0 and ORT
  core 1.31.0 load. LPIPS-Alex passes 18/18 golden pairs plus warm/cold parity.
  DISTS reaches 2/18 pairs, then the resource wrapper kills both combined and
  isolated attempts at 2048² when its available-page proxy falls below 4 GiB;
  full DISTS parity and its warm/cold check remain unvalidated under the 12 GiB
  guard. An isolated one-job retry on 2026-10-18 again completed 2/18 pairs
  and stopped at 2048² (exit 137, min-avail proxy 3696 MiB); largest passing
  size is 1024². Warm/cold parity was not reached. No further retry attempted.
  [Commands and measurements](benchmarks/mac_validation_2026-10-18/README.md).

- **2026-10-18 Mac/Metal validation at a3bc3a98:** CVVDP Mode B walkers
  8/8, CSF halos 5/5, Mode E 12/12, dispatch limits 2/2 and cancellation
  4/4 pass in separate processes, with no device-thread error messages.
  Zensim `it` passes 107/113: the two documented failures persist, plus
  pointwise diffmap parity and three odd-dimension pyramid tests fail.
  No implementation or threshold changes made.
  [Commands, values and retained evidence](benchmarks/mac_validation_2026-10-18/README.md).

- **2026-10-16 — PieAPP weight-head reconstruction:** the initial absolute-value
  hypothesis matched the positive observed weights, but opaque bias/projection
  controls identify signed `weight+1e-6`. Score revision 3 removes the
  unsupported absolute value and accepts finite signed weights in pooling.
  Both native 24-pair panels and exact warm/cold identity pass. All 1008
  corrected KonJND scores are bit-identical and all input hashes match.
  Original goldens, calibration scale and JND gates are unchanged.
  Evidence: `benchmarks/pieapp_review_2026-10-16/weight-identify.json`.

- **2026-10-16 — ORT CUDA arena/memory-pattern accumulation across changing
  input shapes. FIXED (`f48b71a0`).** The afine fused pair graph (`OrtCudaPair`) OOM'd the GPU
  arena at pair 40/61 (S46) of the shape-varying AIC sweep (a ~2.4 GB `Softmax`
  intermediate). WD's trunk had the same class fixed in `ae582a3b`
  (memory-pattern off, CPU arena disabled, outputs consumed by move). The fix
  is now shared by every ort-cuda session in `engine::ort_engine::cuda_session`
  — `OrtGpuPair` (afine, ckdn, deepdc, dists, dreamsim, lpips/stlpips),
  `OrtGpuImage` (arniqa, hyperiqa, maniqa, liqe, musiq, paq2piq) and
  `OrtGpuTrunk` (wd) — plus `take_output` move-consume on the pair/image
  paths. afine sweep scores are bit-identical before/after (357/357
  shared rows) and ort-cuda ladder parity re-verified on afine,
  lpips_alex, stlpips_alex, stlpips_vgg, dists, deepdc, ckdn, dreamsim,
  arniqa, maniqa, liqe, musiq, hyperiqa, paq2piq, wadiqam and wd;
  lpips_vgg covered by the fixed-size nn_bench A/B in the same receipt
  (lpips_squeeze shares lpips_alex's `OrtCudaPair` graph shape, weights
  only — not separately probed). RSS numbers are in
  `benchmarks/afine_aic_2026-10-16/README.md`.

- **2026-10-16 — WD CUDA full-resolution host retention, fixed (`ae582a3b`).**
  A varying-size AIC control sequence reached its 16-GiB cap at 3355×2516.
  The WD trunk now disables its CPU arena and consumes output tensors individually.
  The four-case recovery panel completes at 11,210,496 KiB maximum RSS, with
  unchanged pilot bits and both-engine ladder parity. See
  [the measured receipt](benchmarks/wd_port_2026-10-16/README.md).


- **`cvvdp-gpu` fails to compile when built in one cargo invocation with `dssim-gpu`,
  `ssim2-gpu` or `iwssim-gpu` — found 2026-10-15, reproduced on master `c95dbc6d`, NOT
  fixed.** `zenmetrics-gpu-core::Backend` gates its `Wgpu` and `Cpu` variants on
  gpu-core's own `wgpu`/`cpu` features. When a sibling crate in the same build turns
  those on, feature unification adds the variants, and `cvvdp-gpu`'s exhaustive matches
  (`src/memory_mode.rs:131`, `src/opaque.rs:376`, `src/session.rs:70`) fail with
  E0004 "`Backend::Wgpu` and `Backend::Cpu` not covered". Each crate builds on its own,
  and `-p zenmetrics-api -p cvvdp-conformance` builds. Repro:
  `just snap-cargo @ check -p cvvdp-conformance -p dssim-gpu`. Fix direction: give
  those matches arms for variants cvvdp-gpu was not built for (an error return), as
  the other `*-gpu` crates already compile under the same unification.

- **i265's RTX 4080 flips a bit every few minutes (owner report, 2026-10-14).**
  Any GPU result on i265 that disagrees with a reference, a previous run, or
  another backend must be rerun before it counts as a finding; a mismatch is
  real only if it reproduces. Parity harnesses run on that box must rerun
  mismatching cases automatically and report both runs. Do not use a single
  i265 GPU run as a golden. Timing numbers are unaffected.

- **2026-10-14 — VSI diffmaps were mislabeled at min(w,h) ≥ 384. FIXED (`c9248347`).** VSI.m decimates by `F = round(min(w,h)/256)`. The kernel wrote `⌈w/F⌉ × ⌈h/F⌉` cells into a `w × h` buffer and `VsiDiffmap` reported `w × h`, so every map with `F ≥ 2` was scrambled with a zero tail. The fleet diffmap executor (`jobexec.rs`, `"vsi"` arm, wired in `c110edaf` on 2026-10-13) persisted those maps as `w × h` PFMs. `JobKind::Diffmap` carries no metric revision, so any VSI diffmap blob written between `c110edaf` and the fix for an image with min(w,h) ≥ 384 is wrong and must be deleted and regenerated by hand; re-declaring will not replace it. Regression: `vsi::tests::vsi_diffmap_is_decimated_at_large_sizes`.

- **2026-10-14 — VMAF v1 `speed_chroma_uv` disagrees with libvmaf v3.2.1 on aarch64 macOS. NOT root-caused.** `cargo test -p vmaf --test ffi_fusion full_v1_pixel_scores_match_v321` (default features, so the scalar path) fails on master `a4d0b8fc`: Standard1080p, 8-bit, distorted, frame 0, ours 0.5156307220458984 vs libvmaf 0.2641477584838867 (tolerance 1e-3); cambi, adm3 and motion3 agree. The other 25 FFI parity tests pass. x86 not checked. The tolerance must not be widened; the SPEED chroma port needs fixing.

- **2026-10-13 — cached browser glue/WASM generation mismatch.** An older loader's 25-page shared memory failed against a module requiring 42 pages. Pages assembly now versions the app, coordinator, generated glue, WASM and nested Rayon helpers together (`e05d1c91`). Entry-page/source/compatibility-worker cache rules are disjoint from immutable asset rules (`dee64105`). Browser checks block historical package URLs, exercise an undersized-memory LinkError control, and retain demo/scalar/map/cancellation assertions. WebGPU readiness is a separate compute/readback diagnostic; the selectable browser metrics remain CPU routes.


- **2026-10-13 — browser no-default-features compile failure.** The native CVVDP spatial arm used display methods even when browser cpu-cvvdp was disabled, making the display type uninhabited. `8290373f` gates that arm; the pinned no-default-features compile check passes. Default-feature browser behavior is unchanged.


- **2026-10-13 — browser large-image zensim cleanup trap.** The live 768×512 demo decoded and prepared its reference, then trapped in `std::thread::spawn` while deallocating streaming planes. Zensim fix `96c3c9bb` releases those buffers synchronously on wasm32 while keeping Rayon compute kernels. The browser regression now covers the real demo, full-HD score/map completion, shared atomic cancellation, subsequent scoring and fresh-worker UI recovery. Small fixture success did not establish this path.


- **2026-10-12 — the internal IW-SSIM pyramid's old reflect1 kernels cannot repeatedly downsample a one/two-sample axis.** The retained materialized oracle panics on a 1×1 depth-two build; public scoring pads small inputs to its native dimensions. Exact rolling-pyramid tests exercise tiny axes at residual-only depth and deeper supported shapes explicitly. This is an existing internal kernel restriction, not score drift from the rolling schedule.

- **2026-10-12 — native adapter CI could hit resource admission before testing its requested child outcome.** The macOS runner in [run 38317895817](https://github.com/imazen/zenmetrics/actions/runs/38317895817) refused the timeout fixture as `busy`: CPU load 1.0, six heavy processes, a two-second admission budget. Controls now allow 30 seconds for resource admission, with a matching bounded parent wait; the two-second child timeout and every expected outcome remain unchanged. Linux controls pass complete, OS-peak, wrong-output, timeout, cancellation, routing refusal and the native worker/ledger path. The admission gate remains enabled.

- **2026-10-12 — IW-SSIM SIMD tests lacked `alloc::vec`/`Vec` imports with default features disabled. FIXED.** The test module now imports both explicitly. The existing 41 tests pass with and without default features; assertions are unchanged. Reproduce with `just snap-cargo @ test -p iwssim --no-default-features`.

- **2026-10-12 — Windows checkpoint manifests used host path separators.**
  The archive converted backslashes to forward slashes while the manifest retained
  backslashes, so strict release verification refused the otherwise green Windows
  ARM64 package. Native/browser manifest writers and archive naming now use POSIX
  relative paths. Archive controls check Windows conversion, nested-file roundtrip
  and hash-tamper refusal; existing verification gates remain unchanged.

- The CLI no-GPU fixture hid CUDA and Vulkan but left Metal reachable. macOS ARM64 therefore correctly returned a GPU score while the fixture expected refusal. Selecting an unavailable WGPU device index also excludes Metal and DX12 without changing either refusal assertion or the separate GPU-present numeric parity test.

- GPU memory-mode defaults must fit the target address width. The 8 GiB fallback budget overflowed `usize` when the i686 CLI test graph compiled GPU dependencies. Saturating multiplication retains exactly 8 GiB on 64-bit targets and limits 32-bit budgeting to representable bytes. The complete i686 CLI all-targets run passes; score arithmetic and GPU benchmark acceptance are unchanged.
- The distortion-worker protocol fixture decodes its UTF-8 JSON frame before parsing, preserving its byte payload and assertions on Python versions that require a string for `json.loads`. Both existing RGB8/RGB16 round-trip fixtures pass inside the Python-enabled i686 builder. No test was skipped or relaxed.

- VMAF ADM/CAMBI/SPEED portable dispatch generation must import `attune` on every SIMD-enabled target, including i686. Architecture-specific tokens remain gated. The i686 cross compile and aarch64 compile pass with the portable macro imports; numerical execution on i686 was not measured.

- The pinned fast-ssim2 revision d989014a predates the PixelSlice/config API already used by zenmetrics-api. Native CPU CI fails resolving those symbols. Pin 96ad1ab0 supplies that surface and the hdr-pu cancellation fix; regenerate the lock against pinned siblings when changing it.
- Adding Benchmark to the fleet enum requires the dashboard kind label as well as core/controller/worker matches. The broad CI compile caught the missing dashboard arm after the smaller contract suite passed.

- **2026-10-15 — `just snap-cargo <older-rev>` could return the previous
  revision's binary. FIXED.** `git archive` stamps exported files with the
  commit time, so after a build of a newer tree an older revision's sources
  looked older than cargo's dep-info and cargo skipped the rebuild (found by
  the nn-opt lane while A/B-benchmarking revisions). `export-snapshot.py` now
  keeps `<tree>.mtimes.json` (sha256, mode, mtime per file): byte-identical
  files keep their previous mtime, changed files get one newer than any
  earlier stamp. Regression: `just lock-snapshot-tests`
  (`test_revision_switch_restamps_only_changed_files`; fails on the old
  exporter). Benchmarks taken with `snap-cargo --rev` before this fix may
  have measured the wrong revision.

- **2026-10-07 — CI lock snapshots copied ignored custom Cargo target
  directories. FIXED (`8cbf9cbb`).** Export the jj-tracked tree, excluding
  marked `target*` directories even when accidentally tracked, plus repository
  metadata. Working source additions/edits/deletions remain included; explicit
  revisions stay immutable. `--check` still resolves against the unchanged
  23 sibling pins, without rewriting Cargo.lock. Regression commands:
  `just lock-snapshot-tests` and `just lock-check`. The old exporter fails
  the cache-copy tripwires; the new exporter passes them.

- **2026-10-07 — same-owner timer/progress CAS contention disarmed an active
  worker chunk; a matching-owner malformed timestamp authorized release.
  FIXED locally (`962f5e47`).** Failed renewal CAS now re-reads ownership once:
  a matching owner retains held/in-flight state and future timer renewal; a
  missing/foreign owner disarms it without another PUT. The colliding progress
  update is not persisted. Release requires a u64 timestamp and still accepts
  fresh two-token claims. Four new regressions fail before the fix and pass
  after with unchanged assertions; all 11 ownership regressions and the full
  65-test worker suite pass, as do clippy and fmt in the CI-pinned snapshot.
  The documented best-effort GET→DELETE race remains unchanged.

- **2026-10-07 — stale worker shutdown could delete a later owner's claim, and
  progress renewal could overwrite that owner's lease. FIXED locally (`d3ca1108`).**
  Shutdown release now reads the claim and checks its second token against the
  acquiring worker identity. Progress writes check ownership and use the read
  ETag with `put_update`; confirmed ownership loss clears the matching
  held/in-flight chunk. Late callbacks preserve a newer in-flight chunk.
  Seven ownership regressions pass, including the two failures reproduced with
  old unconditional behavior; the full worker crate tests and clippy pass in
  the CI-pinned snapshot. Release uses separate GET and DELETE operations;
  running execution is not cancelled on lease loss. See
  [RUNNING_JOBS.md](docs/RUNNING_JOBS.md) for the lease contract.

- **`ButteraugliOpaque`'s cached reference is a NO-OP for any image taller than
  224 px — it silently replays the full one-shot. Found 2026-08-30 (task #163),
  by reading; not a regression, a design consequence.** `resolve_auto` tries
  Strip *first* — "even if Full fits, butter is strip-preferred"
  (`crates/butteraugli-gpu/src/memory_mode.rs:190-194`) — whenever
  `height > MIN_STRIP_BODY + 2 × HALO_ROWS` = 64 + 2·80 = **224**. In strip mode
  `set_reference_srgb_u8` only does `self.cached_ref_strip = Some(ref_rgb.to_vec())`
  (`opaque.rs:703`), and `compute_with_reference_srgb_u8` calls
  `self.inner.compute_srgb_u8(&held, dis_rgb, …)` (`opaque.rs:726`) — the whole
  pair compute, **plus a host `Vec` clone of the reference on every call**. So a
  warm-reference loop on an `Auto`-built instance is the cold path with extra
  memcpy. The crate's own doc claim that "the cached-ref score is **identical**
  to the one-shot score — no parity risk" (`opaque.rs:317-322`, `:676-680`,
  `:712-714`) is TRUE and VACUOUS: identical because it is the same computation.
  The umbrella hits this too — `zenmetrics-api/src/memory_mode.rs:107` maps
  `Auto → butteraugli_gpu::MemoryMode::Auto`.
  **MEASURED on Metal (task #163, `benchmarks/metric_loop_gpu_metal_2026-08-30.tsv`,
  driver `crates/zenmetrics-api/examples/loop_gpu_probe.rs`): `warm/cold` under
  `Auto` = 1.002 / 1.010 / 1.011 / 1.003 at 256²/512²/1024²/2048² — dead flat at
  1.0. Under `Full`: 0.95 / 1.17 / 0.79 / 0.66.** `Full` is also the *cheaper*
  constructor (166 vs 201 ms at 2048²), so there is no reason to take `Auto` for
  a warm-reference workload. Scores are identical in every cell, both modes.
  **Workaround: construct with `MemoryMode::Full`** to get the real device-cached
  reference (`pipeline.rs:920-933` caches opsin + frequency separation + ref mask
  at both resolutions). The typed multires-strip pipeline rejects `set_reference`
  outright (`pipeline.rs:912-914`, `StripModeUnsupported`) — the opaque layer is
  what papers over it.
  **Corollary — `benchmarks/gpu_coldstart_2026-05-29.tsv`'s butteraugli
  `warm_per_call_ms` (1.5 / 3.6 / 12.9 / 50.2 ms) is STALE for the opaque API.**
  That file predates the strip held-reference, which landed 2026-05-31 in
  `302ff4fc` ("#160"); before it, strip-mode `set_reference` errored. Those are
  whole-image numbers. Do not quote them for an `Auto`-constructed instance
  without re-measuring under `MemoryMode::Full`.

- **`zenmetrics-orchestrator --lib`: host-capability tests fail on any host lacking the
  hardware they assume — environment, not a regression (confirmed 2026-08-28 on macOS;
  scope corrected 2026-09-09 after hitting it on Linux).** `tests::detect_cpu_returns_nonempty_brand` (lib.rs:938)
  asserts `detect_cpu().brand` is non-empty, and `detect_cpu` has no macOS/`sysctl` arm, so the
  brand comes back `""`. `tests::fresh_profile_is_not_stale_by_time` (lib.rs:893) builds a
  `fake_profile()` claiming an `NVIDIA GeForce RTX 5070` with `gpu.present = true`;
  `is_profile_stale` then calls the REAL `detect_gpu()`, which finds no NVIDIA card on a Mac and
  returns `true`. Both are pure host-capability assertions with no cross-platform arm — neither
  touches the executor / chooser / adapter code. The other 78 pass.
  **CORRECTION 2026-09-09: `fresh_profile_is_not_stale_by_time` is NOT macOS-specific.** It
  fails on ANY host without the NVIDIA card `fake_profile()` claims — observed on `i265`
  (Linux, Intel iGPU only): 50 passed / 1 failed, same assertion at `lib.rs:893`. The real
  condition is "no NVIDIA GPU present", which covers most non-fleet dev boxes, not just Macs.
  `detect_cpu_returns_nonempty_brand` DOES pass on Linux — that one is genuinely macOS-only,
  since `detect_cpu` lacks a `sysctl` arm. So Linux shows 1 failure, macOS 2. Do NOT chase these as a
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

- **cvvdp-gpu on wgpu: a dispatch that fails on cubecl's device thread can still yield a
  score — FOUND 2026-09-26, NOT fixed; the fix belongs in imazen/zenforks-cubecl.** Kernel
  launches are fire-and-forget tasks on the `DSD-*` device-service thread, and the task runner
  runs each one under `catch_unwind` and only `log::warn!`s a panic
  (`cubecl-common/src/device/handle/channel.rs:465` and `:479` at the pinned rev 9084240). So
  any dispatch that panics there is dropped with nothing reaching the caller: the fork's own
  binding-alignment `assert!` (`cubecl-wgpu/src/compute/stream.rs:564`), or a wgpu validation
  error, which wgpu's default uncaptured-error handler turns into a panic (cubecl-wgpu installs
  no handler). cubecl-wgpu does have a stream error sink (`stream.errors`, reported by
  `flush`/`sync`), but the panic never writes it, and `read_resources` flushes with
  `ignore: true` anyway. Whether a CVVDP call then errors is luck: it does only if a LATER
  blocking task fails too. Pre-fix multi-strip Mode B got `ReadbackFailed(... CallError)`; a
  single dispatch with a 4-byte-offset binding in `compute_dkl_jod_from_handles` returns
  `Ok(9.263138)` where the valid call scores 1.9945726 (the regression test
  `tests/it/failed_dispatch.rs`, `wgpu_failed_dispatch_returns_an_error_not_a_score`, is held
  off master until the fork fix lands so the wgpu suite stays green). Not checked on CUDA,
  whose launches cannot hit these two wgpu-only rejections. Proposed fork fix: (i) the
  alignment check and a new per-axis workgroup-count check push a `ServerError` into the
  stream's error sink and skip the dispatch instead of panicking; (ii) `read_resources`
  returns and drains the pending sink errors, so the readback that ends every CVVDP call
  fails (every cvvdp-gpu `read_one` already maps its error to `Error::ReadbackFailed`);
  (iii) an uncaptured-error handler that records wgpu validation errors into the same sink;
  (iv) backstop: the task runner counts caught panics and the next read/flush/sync on that
  device returns an error. **Where it lands needs a decision:** the fork's `main` (b7e9ab99)
  was rebased onto upstream 5b37ec40 and no longer contains 9084240, so it is either a
  branch off 9084240 or `main` plus a zenmetrics pin bump across that rebase. On `main` the
  same swallow is at `channel.rs:692`/`:706` and `check_binding_alignment` still panics
  (`stream.rs:122`). Until then: **on wgpu, a `cubecl-wgpu:` message or a `panicked at` line
  from a `DSD-*` thread in stderr means that process's scores are suspect**, even when every
  call returned Ok.

- **zensim-gpu `it` suite on macOS/Metal (wgpu): 2 deterministic failures, pre-existing
  (verified 2026-08-27 — identical values on baseline b07a0485 before that day's commits, in
  isolation and serially):** `diffmap_invariants::invariant_1_identity_yields_near_zero_diffmap`
  (identity score 2.3330367, expected ~0 — the default CPU-diffmap path, 16×16) and
  `cached_ref_slot_rebuild::gpu_score_correct_after_foreign_size_rebuild` (one-shot AND cached-ref
  GPU 51.2412 vs CPU 48.8323, 2.41 JOD off after the 1024→1448 rebuild + `reclaim_pooled_vram`).
  Rechecked on Metal 2026-10-18 at a3bc3a98: identity remains 2.3330367; rebuild
  GPU=51.24123255864118 vs CPU=48.827715234845805. The current suite has 107/113
  passing, with four additional failures detailed in the dated validation report.
  Not root-caused. Historical command:
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
  (2026-08-27), independent of code pushes.** (Epilogue: the last remaining red
  job, `Compile (ubuntu-latest)`, went green on 2026-08-28 — see the
  `exec_command_deadline` entry under Resolved.) The 2026-07-29 causes (a stale
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

- **cvvdp-gpu `it` suite (`--features parity-goldens`) exhausts an 8 GB RTX 2080
  when run as ONE process** (observed 2026-09-23 during the pycvvdp v0.5.7 pin
  bump, 4bf7bdf8): 554 pass / 18 fail. One was the intended `version_matches` pin
  (updated in that commit); the other 17 failed with `ReadbackFailed(ServerUnhealthy {
  Io(can't allocate buffer of size: 31888384), Io(couldn't find resource for that
  handle: Memory page 0 doesn't exist) … })`, cascading through `pipeline_score`,
  `pyramid_kernel`, `state_machine_independence`, `strip_kernel_parity`,
  `strip_mode_e_phase3`, `mode_b_walker_parity` and friends. The GPU was otherwise
  idle (1 MiB used). **Every affected module passes in its own process** (123/123:
  `cargo test -p cvvdp-gpu --release --features parity-goldens --test it --
  <module>::`). Same family as the ssim2-gpu entry below. The bump touched no
  cvvdp-gpu runtime code; that it is pre-existing is inferred from this, not
  A/B-measured against the parent commit. Workaround: per-module processes.

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

- **Shared-`TMPDIR` staging collision lost ledger Done rows — found and FIXED 2026-10-03 (`7a268033`, #63).** Every
  `s3://` transfer in zenfleet-ledger staged through `$TMPDIR/zenledger_{ul,dl,bi,wb,rb}_{pid}_{n}`. The tower LAN fleet binds
  ONE host directory as every container's `TMPDIR`, and each container has its own PID namespace, so workers share pids
  and counters. Measured: Tower-q103's sidecar `pass-Tower-q103-1.chunk-0f08e08e.parquet` was byte-identical (sha256) to
  Tower-q106's `pass-Tower-q106-1.chunk-7cf28b06.parquet`; q103's Done row was gone, its later passes met their own
  completed claim and logged "every chunk is claimed elsewhere" until the 7200 s TTL expired, then re-ran the cell.
  Result blobs were never affected (`R2BlobStore::put` is an in-process conditional PUT, no temp file). Fix:
  `zenfleet_core::tmp::unique_temp_path` (`{tag}_{pid}_{counter}_{nanos}`, `create_new`). **Rule: never build a scratch
  name from `process::id()` alone in code that runs in containers** — pids repeat across PID namespaces.

- **cvvdp-gpu multi-strip Mode B (`StripPair`) failed on wgpu (Metal and Vulkan) — FIXED
  2026-09-26 (`c4bfe0fd`).** Root-caused on Metal 2026-08-28: the DKL, Gaussian-pyramid, Weber
  and CSF strip walkers sliced row-strips with `Handle::offset_start(byte_off)`,
  `byte_off = strip_row × level_width × 4`, which is not a multiple of
  `min_storage_buffer_offset_alignment` (256 on Metal, 32 on the RTX 2080's Vulkan) once a
  pyramid level's row stride drops below it. CUDA and HIP have no such requirement, so the
  offsets were wrong everywhere and only wgpu enforced it. cubecl-wgpu (`f528c4b5`) rejects
  such a binding with a panic on the device thread; the call failed with
  `ReadbackFailed(... CallError)`. Fix: every walker binds at `aligned_split`'s 256-byte base
  and launches a `*_shifted_kernel` twin that adds the leftover elements to its indices (the
  masking walker got the same treatment in `f29a8371`). After, on Vulkan:
  `mode_b_walker_parity` 8/8, `strip_mode_b_csf_halo_parity` 5/5, 0 device-thread panics,
  Strip == Full bit for bit at 256², 1024² and 1000×750; CUDA probe output byte-identical.
  `cancel::strip_pair_mode_polls_per_strip` is no longer cuda-gated (`86449118`). **Run on Metal 2026-10-18 at a3bc3a98:** walkers 8/8 and CSF halos 5/5
  pass, with zero device-thread error messages. The existing 4096² test remains
  ignored. cvvdp-gpu is still omitted from CI's Metal matrix.

- **cvvdp-gpu Mode E (`MemoryMode::Strip`) on wgpu scored 1000×750 far from Full — FIXED
  2026-09-26 (`f29a8371`, test `445aa175`).** Root cause: the masking strip walker bound row-window
  sub-views (`offset_start(top_global · bw · 4)`) at offsets that are not multiples of
  `min_storage_buffer_offset_alignment` for the odd level widths of 1000×750 (250, 125, 63).
  cubecl-wgpu's check **does** fire, but it panics on the device-service thread; the call still
  returns, with those masking strips never dispatched (30 device-thread panics per probe
  process). The earlier note here said the path did not trip the check; that was wrong. Mode E
  scored the noise pair 4.603752 against Full 5.008305 (mild 9.990167 vs 9.961642); CUDA
  matched exactly. After: wgpu Mode E = Full bit for bit, 0 device panics, CUDA byte-identical.
  Rechecked on Metal 2026-10-18 at a3bc3a98: Mode E 12/12 pass; 1000x750
  noise Full=Strip=5.008305 and offset Full=Strip=9.458392, printed difference 0
  in both cases, with zero device-thread error messages.
  Gate: `strip_mode_e_parity::mode_e_matches_full_1000x750_h_body_128`. **Lesson: on wgpu a
  `cubecl-wgpu: storage buffer binding … not a multiple of …` line in stderr means the score is
  wrong, even when the call returns Ok.**

- **cvvdp-gpu on wgpu returned JOD 10.000 for every image of 2048×2048 or larger — found
  and FIXED 2026-09-26 (cvvdpdet lane).** Every per-pixel kernel was launched with a 1-D
  grid `(n / 64, 1, 1)`; above 65 535 workgroups on one axis (n > 4 194 240 px) wgpu rejects
  the dispatch on its device thread, no error reaches the caller, the D planes stay unwritten
  and the pool yields JOD 10. Measured on Vulkan with master 9e918021, noise pair, Full mode:
  2000×2000 → 4.178, **2048×2048 → 10.000, 3000×2000 → 10.000**; 4096² failed loudly only
  because the old pool's own dispatch also overflowed. Fix: `cube_count_1d` in
  `crates/cvvdp-gpu/src/pipeline.rs` folds large 1-D grids into 2-D (all kernels index with the
  linearized `ABSOLUTE_POS`). After: wgpu 2048² / 3000×2000 / 4096² = 1.854582 / 3.804538 /
  1.843300, CUDA 1.854579 / 3.804536 / 1.843298. **Any CVVDP score computed on a wgpu backend
  for an image ≥ 2048×2048 before this fix is invalid (it is 10.0).** CUDA was never affected
  (2³¹ − 1 workgroups per axis). The sibling GPU metric crates already fold their grids
  (their own `cube_count_1d`); not re-measured here.
  Gate: `tests/it/dispatch_limit.rs` (wgpu Full at 2048² / 3000×2000, Mode E and Mode B with
  window-sized strips, and wgpu vs CUDA); with the fold removed both wgpu tests fail with JOD
  10.000000.

- **cvvdp-gpu GPU scores varied run to run (`Atomic<f32>` pool) — FIXED 2026-09-26.** The
  spatial pool summed with `fetch_add`, so the order was the scheduler's: band scores
  differed on every call and the JOD's low bits wandered (~1e-6 relative), failing
  zenmetrics-api's bit-identical `cancel::` assertions (3 of 16 runs of one test on this host). Now a
  fixed-order two-pass reduction (`pool_rows_3ch_kernel` + `pool_rows_finalize_kernel`):
  bit-identical across calls and processes, and Full / Mode E / Mode B pool identical D
  planes identically. Record: `benchmarks/cvvdp_pool_determinism_2026-09-26.md`.

- **`exec_command_deadline` could hang for the child's FULL runtime instead of
  its deadline — root-caused and FIXED 2026-08-28 (`031f5aa6`), confirmed green
  on the ubuntu runner.** This is the failure that held `Compile
  (ubuntu-latest)` red from 2026-08-27
  (`exec_command_deadline_kills_and_classifies_timeout`, "returned in
  30.002861621s", passing on macOS). **Cause:** the deadline kill targets the
  child's process *group*, which normally reaches grandchildren — but the
  stdout/stderr reader threads were collected with `join()`, and `read_to_end`
  returns only when EVERY write-end of the pipe is closed, including a
  grandchild the group kill missed. `join()` was therefore an unbounded wait on
  a process the watchdog does not control, which is exactly what the deadline
  exists to prevent. The old note's guess ("the kill not reaching the
  grandchild, so the reader threads block") named the right mechanism but drew
  the wrong conclusion: whether the kill lands is not something the watchdog can
  guarantee, so the **wait** is what has to be bounded. **Fix:** the readers
  deliver over `mpsc` channels and are collected with `recv_timeout` (2 s on the
  timeout path, 60 s on the normal path); the group kill also gained the `--`
  separator, without which a `kill` implementation may read `-<pgid>` as an
  option and refuse the invocation silently (the status is discarded).
  **Verified, not assumed:** the new test
  `exec_command_deadline_returns_even_when_a_grandchild_escapes_the_kill`
  reproduces the class on any Unix (the child backgrounds a `perl -e 'setpgrp;
  …'` grandchild that leaves the group and holds the pipes), and reverting the
  collect to an unbounded wait makes it fail with "returned in 30.178194625s" —
  matching the CI symptom to the second. **Lesson for anyone writing a
  watchdog:** killing a process does not close pipes that a grandchild still
  holds; bound the collection, do not trust the kill.

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

## Margarine research

`crates/margarine` is an unpublished standalone workspace, excluded from the
umbrella dependency graph. Read its `DATA_PROVENANCE.md` before evaluation.
Its source snapshot under `vendor/butteraugli` is pinned and hash-verified by
`just margarine-vendor-check`; do not silently replace it with current upstream
or a sibling checkout. The selected row-Malta command and newer native-UHF
control are distinct. Migration does not establish quality/resource acceptance,
and scalar ranks do not qualify spatial steering. Build/test recipes start
with `margarine-`; the dedicated workflow carries the original feature checks.

## Quality hygiene (advisory, standing)

Run `just quality` before pushing (or after any multi-file change): fmt
check, clippy warning census, unused deps, `cargo deny`, typos, stale-doc
scan, API-surface exposure report. Nothing here gates CI — the contract is
that warning counts shrink over time, never grow. If your change adds
rustc/clippy warnings, fix them in the same commit; don't leave them for
"later". New `pub` items need a concrete current caller — default to
`pub(crate)` (`unreachable_pub` is warned workspace-wide); the api-guard PR
bot flags new surface plus pub fields, unsealed traits, missing docs, and
re-export growth. `unsafe_code` is `forbid` workspace-wide — unsafe belongs
in magetypes/archmage/disjointmut. Crates that still carry unsafe (the
GPU-dispatch / FFI-oracle / SIMD debt: *-gpu, mdctpsnr, zenfleet-vastai,
zenmetrics-cli, zenmetrics-api, zenmetrics-orchestrator, cvvdp, vmaf,
msssim) keep it compiling via a mirrored lint block with an explicit
`unsafe_code = "allow"` — that allow is debt to remove, not a license, and
new crates get the forbid automatically. `just coverage` (llvm-cov)
measures test coverage — check
it before refactoring anything without tests. Heavy stages honor
`~/work/zen/scripts/run-heavy` conventions: wrap `just quality` runs in it
when the box is busy.

## Known Bugs — SHIPPATH11 (2026-10-07)

The strict training-only harvester previously trusted reported epoch/pair budgets and nonempty admission fields. The qualified program contract now binds exact expected budget, receipt/freeze/decision and seven-table admission to the manifest program/data hashes; the canonical checkpoint decoder must agree. Local smoke argv identities are verification-only and cannot install. The separately deployed E28 harvest driver still delegates to its existing verifier without these checks; it was inspected but left unchanged pending coordinator sign-off.

### Known Bugs: benchmark evidence retention (2026-10-12)

Benchmark receipts contain nested content-store artifacts, while the original
GC only marked each Done row's output receipt. The benchmark GC expansion now
marks bundle files, runner and all receipt artifacts before latest-wins folding.
Changed or unknown-schema metadata refuses GC. Preserve the original attempt
ledgers as inputs to reachability; discarded history cannot be reconstructed.

### Known Bugs: Windows benchmark parent lifetime (2026-10-12)

Windows preserves a dead parent's PID in child metadata, so comparing getppid
alone does not detect worker exit. The adapter now holds a synchronization-only
parent process handle and polls it without spawning monitoring processes. Native
contract tests exit a real parent and require its surviving child's watcher to
observe termination. The handle also prevents PID reuse from changing identity.

### Known Bugs: checkpoint feature combinations (2026-10-12)

The full CLI feature probe exposed HDR-SVT imports through expert-only module
paths, the TIFF fleet planner without `zentiff/zencodec`, and an AVIF dependency
pin predating `SvtParams::chroma_q`. Use SVT's public `pipeline` facade and
compile the TIFF codec wrapper. The AVIF pin must include both the chroma
control and its fingerprint fix (`4d999037`): the intervening control-only
revision merges distinct chroma cells. The existing dedup assertion reproduced
that collision; all 98 CLI library tests pass after the paired fix, with no
assertion changes (two existing ignored tests remain).

### Known Bugs: portable VMAF SIMD generation imports (2026-10-12)

VMAF's `attune` macro import was restricted to x86 in the VIF and motion
modules while their portable kernels and dispatch calls were enabled by `simd`
on every architecture. ARM checkpoint builds failed before metric execution.
Import `attune` under `simd` independently of the x86 token/trait imports.
Pinned-source `cargo check` with `simd,avx512` passes on both aarch64 Linux and
x86_64 Linux; this establishes compilation, not ARM numerical execution.

### Known Bugs: browser cached reference after rejected input (2026-10-12)

The checkpoint worker freed its cached reference before validating a replacement.
An alpha rejection then left the previous identity attached to the freed object;
restoring the original input could reuse it. Prepare the new reference before
freeing or replacing the current one. Chromium recovery controls fail with the
old worker and pass with the corrected worker, retaining exact native scalar
and raw-map checks.

### Known Bugs: CUDA initialization after driver replacement (2026-10-12)

i134's GTX 1080 Ti requires the proprietary NVIDIA kernel module. After
replacing its incompatible open module with 580.178.04, direct controls passed
all six CUDA metric variants, but native worker attempts still intermittently
fail `CudaContext::new(0)` with `CUDA_ERROR_UNKNOWN`. Udev repeatedly
unloaded/reloaded modules; targeted child-module unload actions reduced that
loop without qualifying the worker deployment. All failed packets and logs
are retained in the [CUDA deployment receipt](benchmarks/iqa_cuda_deployment_2026-10-12.meta).
A Done ledger row with `packet.status=child_failed` is not a usable measurement.

After an approved reboot, all 24 direct controls and all 24 native worker
observations pass. The [post-reboot receipt](benchmarks/iqa_cuda_deployment_i134_reboot_2026-10-12.meta)
records the actual adapter, runtime hashes and resource boundaries. Earlier
initialization failures remain preserved as historical attempts.

Removing the temporary udev override and restoring unchanged vendor-rule
behavior also passes a fresh 24/24 worker panel. Its [receipt](benchmarks/iqa_cuda_deployment_i134_vendor_2026-10-12.meta)
records the cleanup and interval with no kernel unload/Xid errors.

### Known Bugs: browser map visualization scale (2026-10-13)

The shared fixed map maximum of 1 saturated Butteraugli and CVVDP maps and made
small Zensim maps nearly invisible on the real JPEG demo. The viewer now starts
with per-image 99.5th-percentile square-root scaling and labels its displayed
range; fixed calibration remains explicit. Raw metric maps and scalar scores
are unchanged. Pages qualification renders all three demo maps and checks
visible color variation, in addition to existing native-array comparisons.

### Known Bugs: browser CI lock resolution (2026-10-13)

Website run 38362633510 failed before compilation because the checked-in lock
no longer matched the pinned sibling manifests. The pinned snapshot reproduced
that failure. Regeneration through `scripts/ci/lock.sh --regen` updated required
codec/zensim transitive entries; the 23-sibling locked check then passed.

### Known Bugs: candidate C browser spatial maps (2026-10-13)

The pinned Zensim C native diffmap route fails with `ModelForwardFailed` because
its bake reads beyond that route's feature vector. Canonical C scalar scoring
works without a live stop token. The browser labels C scalar-only, disposes its
worker for cancellation, and refuses C spatial maps explicitly. It never serves
a B map as C. Native-oracle controls cover all six offered SDR models and the
browser verifies cache invalidation when changing models.

### Known Bugs: legacy browser profile exact parity on CI (2026-10-13)

Website run 38386150704 failed the exact native/WASM profile assertion for
Preview v0.1. The local native oracle and browser fixture matched exactly.
The assertion now reports both scalar values and maximum map delta so CI can
identify the difference. The cause remains unverified; numeric expectations
and benchmark acceptance are unchanged.

Recurred 2026-10-15 on website run 38729760597 (`dc9b8c32`), attempt 1:
browser 91.21909967783705, native 91.21909967392351 (difference 3.9e-9),
maximum map delta 0.0. Attempt 2 of the same commit passed and deployed, so
the browser or the native oracle scalar varies between runs (not measured
which); the maps matched exactly both times.

Recurred again 2026-10-16 on runs 38756697855 (`48e768ed`) and 38884395886
(`71560bd8`, attempt 1) with the SAME two values (browser 91.21909967783705,
native 91.21909967392351, map delta 0.0). Identical values across commits and
days point to two deterministic paths rather than run-to-run noise; one
untested hypothesis is that the native oracle's runtime SIMD tier follows the
CI runner's CPU. Next step: have `browser_oracle` record the dispatched tier
and the runner CPU model, then compare a failing and a passing run.

Measured 2026-10-16 on i265 (Core Ultra 7 265K, no AVX-512, so archmage
dispatches the AVX2 tier): the native `browser_oracle` gives preview-v0.1
91.21909967783705, exactly the browser's value, while the failing CI runs'
native oracle gave 91.21909967392351 (other profiles on i265: a 83.85812862461522,
b 91.2532491069443, c 76.20876739427798, d 90.20924821186748, preview-v0.2
90.79464912498275). So the AVX2 native path and wasm agree, and the mismatch
appears only when the native oracle runs on a different tier — most likely
AVX-512 runners (not yet confirmed on an AVX-512 host). That is two native SIMD
tiers disagreeing in zensim's preview-v0.1 path: per the zero-tolerance rule it
needs a zensim tier-parity test and fix, not a looser assertion. Next: run the
oracle on an AVX-512 host (dev) and bisect the preview-v0.1 kernels by tier.

### Browser native map rendering (2026-10-14)

The viewer now consumes native cell geometry for FSIM/FSIMc, HaarPSI, VSI,
MS-SSIM and GMSD instead of stretching reduced or cropped maps. MS-SSIM's
five-pixel border has no native samples. HaarPSI's zero-weight windows yield
NaN local similarity; the browser preserves raw values and marks those cells
explicitly while retaining the valid pooled scalar. VSI's saliency-weighted
quality contribution is displayed directly. Regression coverage lives in
`scripts/checkpoint/check-pages.py`; details and limits in
[the browser viewer contract](docs/BROWSER_VIEWER_2026-10-13.md).

### Known Bugs: learned NR export qualification (2026-10-15)

HyperIQA's traced HyperNet supports a fixed batch of 25, despite the original
ONNX input's dynamic-axis label. The model card and exporter now state the fixed
contract; fresh exports omit the misleading axis label. Its linear head is
unbounded; parity qualifies the documented deterministic grid, distinct from
upstream random-crop sampling. MUSIQ and HyperIQA exports require complete
goldens and image pairs and fail above the canonical 1e-3 JND gate. Run
`just nn-export-checks` for the rejection controls. MUSIQ reports patch-row
limits and degenerate aspect ratios through distinct public error variants.

### Known Bugs: BRISQUE integer powers (2026-10-15)

BRISQUE's initial scalar power path differs from torch's multiplication for
integer squares/cubes. The correction uses explicit multiplication and bumps
the score revision. A 400,000-case GGD/AGGD ratio-and-argmin sweep and disjoint
KonJND images supplement the existing 170 goldens; the fixed argmin case is
pinned in the Rust unit tests. Measurement details live in
`benchmarks/brisque_review_2026-10-15/`.

### Known Bugs: NN preprocessing worker caps and empty rows (2026-10-16)

A-FINE's pixel paths previously requested three helpers even with
RTEN_NUM_THREADS=1. They now use the configured work-scaled cap; a child-process
regression records zero helper launches at a cap of one. Planar conversion
visits all three channels in one packed-row pass. Resize work counts scalar
MACs without an extra output-axis multiplier, and sequential rows avoid a
row-descriptor allocation. Empty preprocessing/resizing inputs return empty
buffers, while DreamSim refuses zero dimensions before resize/inference.
Review measurements live in benchmarks/nn_opt2_2026-10-15.md and its linked TSVs.

### Known Bugs: PaQ-2-PiQ routing and input allocation (2026-10-16)

The sweep range match lacked PaQ-2-PiQ and the default scheduler sent it to
an FR-only orchestrator route. Add its range and direct NR exclusion together;
check both sweep/GPU and sweep/orchestrator/NN builds. The native-resolution
port now refuses either axis above 4096 before allocating its input tensor.
Export must write both loader metadata and a provenance receipt; the oracle
TSV must have a header so parity does not drop its first pair. Review commands
and measurements live in `benchmarks/paq2piq_review_2026-10-16.md`.

### Known Bugs: CAMBI bogus-output checks (2026-10-15)

CAMBI scores only the distorted image; zero is normal for unbanded photos.
Identity and constant-output checks previously rejected valid zero batches.
CAMBI now has no identity range and is exempt from the constant-output check;
empty/nonfinite checks remain active. The sweep-feature binary regression
`cambi_zero_distribution_is_not_bogus` covers the all-zero batch.

### Known Bugs: CAMBI active-band rows and nested frame work (2026-10-16)

A direct nine-column row control found AVX2 bits `402aaaab` for an active
pixel below the histogram band, while `c_value_pixel` returned zero. Gather
index clamping now has a separate in-band output mask (`3b3e81de`). The row
control covers vector groups, scalar tails and masked pixels; only the low
band edge has a predicate-true out-of-band pixel.
A two-frame 512×512 control observed two column splits inside parallel frame
workers. Those workers now keep CAMBI serial; the single-frame control still
uses columns, and all feature/score bits match the serial scorer. Both new
controls fail before the fix and pass after it without assertion changes.
Column regions also borrow disjoint output-row slices instead of allocating
and copying a second c-value plane (`24fe54f8`). Review measurements belong in
[the perf receipt](benchmarks/aic_perf_big_2026-10-16/README.md).

### Known Bugs: DMM solver revision qualification (2026-10-17)

The restored DMM solver passes all 48 frozen same-tap SGESDD row limits.
Canonical JND coverage is 6/24 rten and 7/24 CUDA; the remaining rows have
owner-approved residual floors. The full CUDA repeat is bit-identical.
These floors do not establish canonical parity on the pinned rows.

Reference-source correspondence does not qualify an arithmetic change.
The rejected combined candidate fails nine rten and twelve CUDA limits,
including two/four existing JND rows. Every above-limit CUDA score reproduces.
Its proposed floor increases were rejected and its temporary promotion reverted.

Owner direction 2026-10-17: retain every frozen limit and classification.
Measure each arithmetic change independently on all 48 rows; retain it only
if every limit and native control passes and no row residual increases.
Epsilon and ORG2R-order changes pass all limits but worsen 18 and 24 of the
48 residuals, respectively; removing small-N scaling fails eight frozen limits. Those
three changes are rejected. SLARTG passes every limit but worsens 26 of
the 48 residuals and is also rejected. Direct SORM2R fails eight limits
and worsens 25 residuals; it is rejected. Right-SGER order fails twenty
limits, including five original JND rows, and worsens 26 residuals. It is
also rejected. F64 norm fails nineteen limits, including six original JND
rows, and worsens 22 residuals; it is rejected. Reference-f32 norm fails
seventeen limits, including three original JND rows, and worsens 24
residuals; it is also rejected. All eight isolated changes are rejected.
Every complete CUDA repeat is bit-identical on all 24 rows; the original
numerical section, SVD assertions and frozen golden TSVs remain unchanged.
Source identities, complete rows, resource receipts and historical experiments
are linked from [the solver audit](benchmarks/nn_port_2026-10-15/dmm_fix_2026-10-17.md)
and [the per-change table](benchmarks/nn_port_2026-10-15/dmm_bisect_results_2026-10-17.md).

### E36 fit qualification (2026-10-17)

The private E36 mask changes base table identities. Do not qualify these with an
unchanged E34 package. Use the native E36 profile and exact metadata derivation
before the existing checkpoint gate; see [transport](docs/E36_FIT_TRANSPORT.md).
`just e36-fit-contract-check` uses invented metadata and never trains.

### Known Bugs: Linux IQA benchmark timeout admission (2026-10-17)

The competition harness's killed Rust scopes could leave zenbench running
reservations, causing later cells to refuse admission. The Linux controller now
records each child's PID/process group and recovers only its matching dead
reservation after verifying the whole group has exited and acquiring the guard.
It archives the original bytes and retains a backup in the admission directory.
The [real interruption control](benchmarks/iqa_compete_2026-10-16/timeout-recovery-control.json)
keeps interrupted samples unqualified; OS-backed controls protect live groups,
other owners and locked guards. This is benchmark scheduling state, not a metric
score discrepancy.

### Known Bugs: IQA benchmark ORT worker budgets (2026-10-17)

The initial compete harness set thread environment variables but left ORT session
pools at their default. The installed runtime's default pool used twenty native
threads and expanded worker affinities beyond the requested one-CPU mask. The
benchmark snapshot now sets every session's intra-op budget explicitly, with one
inter-op thread. ORT's full telemetry opt-out also removes its import-time helper
and background I/O from this study. The original exact one/twenty-worker controls
pass without assertion changes. Earlier ORT rows remain diagnostics; corrected
runs have their own ledger and configured-session receipts. See the
[measured control](benchmarks/iqa_compete_2026-10-16/ort-budget-control.json).

### Known Bugs: IQA competition native-map adapter omissions (2026-10-17)

The initial competition harness recorded `fsim-y`, `haarpsi-y` and
`iwssim-piq` maps as unsupported despite existing native-map kernels. Those
historical rows remain preserved and do not establish map unavailability. The
coverage gate now treats unsupported rows for known native CPU map routes as
pending adapter work. Snapshot adapters and replacement attempts are pending
runtime qualification; the [audit receipt](benchmarks/iqa_compete_2026-10-16/coverage-known-map-omissions.json)
records the affected first-panel cells. No metric arithmetic or shipped API is
changed by this benchmark work.

### Known Bugs: benchmark GPU scorer residency at 8K (2026-10-17)

The grouped IQA benchmark keeps independent scalar and map GPU scorers alive.
At 7680×4320, Butteraugli CUDA completes its initial scalar warmup, then fails
a 1,045,012,480-byte device allocation during map warmup. Both requested thread
budgets exit 101 without qualified timing rows. The isolated-mode benchmark
path keeps one scorer per process. All four missing mode/thread cells then
complete with one retained sample each and clean contention receipts. The
[measured receipt](benchmarks/iqa_compete_2026-10-16/butter8k-isolated-v17-control.json)
records 8–9-second scopes and 9.22–9.23 GiB peak host RSS; these are not VRAM measurements.
The original attempt logs remain under `~/tmp/iqa-compete/coverage-rust-v16/cuda/logs/`.
No metric algorithm or allocation limit is changed.

### Known Bugs: dashboard derived-copy staleness (2026-10-17)

The Pages workflow's committed-copy checks failed after a source-document
change without refreshed markdown extracts. Deployment now regenerates the
catalog and extracts and reports copy drift in its build summary. Local
`--check` recipes remain strict. Malformed sources and unclassified benchmark
uploads still fail. Regeneration handles added/obsolete owned shards without
removing unrelated files; controls cover drift, idempotence, malformed-source
refusal and the retained inventory gate.

### Known Bugs: alpha policy in lean CLI builds (2026-10-17)

The decoder's unconditional alpha-policy signatures require `zenmetrics-api`
even when scoring and default features are disabled. Making that dependency
unconditional fixes the bare, jobexec and nn-webgpu feature builds; `alpha`
remains a compatibility flag for its existing re-export. The same decoder and
manifest were present on master; a default-feature check enabled alpha and
did not exercise these lean shapes. Workspace patch, lock and sibling pins
are unchanged. Strict clippy also requires the merged alpha/folder-scoring
lint fixes. The complete final gate and seven existing decoder tests pass.

### Known Bugs: macOS IQA timeout process containment (2026-10-17)

A timed A-FINE scope survived its resource wrapper's pressure-triggered exit
because gtimeout created a separate process group. The owned group was
terminated and its attempt logs retained. The benchmark launcher now uses
`gtimeout --foreground`; a real-child guard control confirms the child exits.
The [receipt](benchmarks/iqa_compete_2026-10-16/task4-mac-supervision-control-20261017.json)
binds that control and the earlier interruption record. This does not claim a
macOS hard memory cap or establish a metric's peak memory usage.

### Known Bugs: portable dashboard projection notes (2026-10-17)

The portable coordinator projects `notes`, but the registered reproduction
competition schemas omitted that supported field. A generated smoke shard
therefore failed strict dashboard ingestion. Both reproduction schemas now
accept `notes`; unknown columns still fail. This preserves per-row workload and
agreement qualifications in versioned observations.

### Known Bugs: browser catalog drift after CLI additions (2026-10-18)

Website run 39103825394 failed because seven new CLI variants were absent from
`web/metrics/coverage.json`. The catalog now includes them in declaration order
and resolves the specific pyiqa model cards before deriving restriction reasons.
CW-SSIM explicitly reports its non-commercial code restriction; DMM retains its
host-SVD blocker. `just browser-coverage-check` runs exact-order, per-variant
reason and missing/misclassified negative controls before the website build.
The catalog fix does not qualify these methods for browser scoring.

### Known Bugs: NN 4K benchmark resume selection (2026-10-18)

The GPU phase of the local NN 4K driver skipped nonempty missing-metric lists,
so waiting longer could not schedule those cells. The tracked
`scripts/complexity/drive_nn_4k.sh` tests for an empty list before skipping.
Probe timeouts now clean up only the process group created by that probe,
replacing broad process-name cleanup. Header repair retains the original bytes
in a sibling backup. Failed attempts remain unqualified; scheduler controls
do not establish metric throughput or complete the missing 4K panel.

### Browser local-weight qualification (2026-10-18)

CKDN and HyperIQA now have conditional `nn-local` CPU-WASM routes. Both
fixtures pass the original 0.001 JND gate against native RTEN, along with
wrong-file/hash refusal, per-model consent isolation, repeat identity and
replacement-worker recovery. The full existing Pages regression passes.
Selected exports stay in memory and are not hosted or persisted by the viewer.
The [qualification receipt](web/metrics/local-weights-validation.json) binds
graph/source hashes and observed controls; it does not qualify WebGPU or all
input sizes. Run `just browser-local-weights-check` with caller-supplied
site, model directory, native baseline and report paths.

### Known Bugs: NN complexity tally multiplicity and dynamic shapes (2026-10-18)

Seven pair-graph tallies multiplied operations already covering both inputs
by two. LIQE CPU runs its trunk once but visits 15 repeated embeddings in
its f64 host head. MUSIQ counts now include the pooled stem and L+1
transformer tokens. Frozen graph receipts and schedule controls qualify
these count corrections. Pair-density pointwise estimates, MUSIQ resampling
and hashing, traffic, halo and span remain partial. The standard 3840x2160
MUSIQ evaluator point exceeds the scorer patch limit and is diagnostic only.
Metric arithmetic and numerical gates are unchanged.

### dvifmish benchmark coverage (2026-10-18)

Six benchmark-only dvifmish routes cover four port presets and the default
talk-inspired float/integer paths. Native maps retain per-plane/per-level
block contributions in raw distortion units; they are not JND pixel maps.
The dependency stays pinned in an explicit private benchmark snapshot overlay,
with no normal workspace dependency or public API addition. Separate 512-first
supplements preserve original manifests and share existing host resource locks.
The i265 real control passes 96/96 cells and 240 clean warm rounds.
Build, scope, hashes and scheduling commands belong in
[the supplement contract](docs/IQA_DVIFMISH_SUPPLEMENT.md). Publishing remains prohibited.

### Known Bugs: IQA controller pre-attempt hash admission (2026-10-18)

An original frozen Mac NN controller hashes model assets before acquiring the
shared heavy lock. A passive admitted observation resolved that controller as
foreign CPU work. Maintained durable execution now acquires admission before
per-attempt asset/manifest/binary/pair rehashing; all hashes still precede the
measurement clocks. The OS-backed control fails on the historical controller.
Existing frozen controllers/results remain unchanged; startup manifest checks
are separate. Matched raw observations also identify genuine Spotlight work
in a diagnostic run. See [the supplement contract](docs/IQA_DVIFMISH_SUPPLEMENT.md).

### Known Bugs: Mac IQA owner CPU percentage undercount (2026-10-18)

Protected internal-only observations capture sysinfo 0.36.1 reporting 10.355%
while the same owner's cumulative CPU advances 863 ms across 202.086 ms.
The false residual exactly matches warm-round flags. Zenbench `3b1d0f71`
provides a private cumulative-counter correction with PID/start-time identity,
conservative missing owner history and retained foreign-work controls. The
benchmark snapshot overlay applies only that correction to the frozen
`ac675d3` source. It changes no resource gates, public API or metric arithmetic.
Corrected worker hashes and new frozen controls must qualify separately;
old failures remain preserved. Details belong in the
[dvifmish supplement contract](docs/IQA_DVIFMISH_SUPPLEMENT.md).

### dvifmish native supplement qualification (2026-10-18)

The corrected Mac worker and clean i265 worker each pass 96/96 control cells
and 240/240 clean warm rounds; all six fixture score bits agree exactly.
i265 production 512 also completes 96/96 with 240 clean rounds. Mac's two
corrected LaunchAgents are installed and waiting on original NN Metal stages;
both hosts retain separate 576-cell larger supplements. Original manifests,
failed attempts and input-contract refusals remain unchanged. No crate/weights
publication or public API addition occurred. Source/build hashes, private Mac
counter correction, true native-map units, dependency waits and resolving
commands are in [the owning contract](docs/IQA_DVIFMISH_SUPPLEMENT.md).

### Known Bugs: 8K scalar FSIM no-downscale warm resource boundary (2026-10-18)

The frozen corrected Mac scalar worker completes the 7680×4320/eight-worker
`fsim-nodownscale` cold cell at 11.79 GiB peak host RSS, but its warm cell
hits the unchanged monitored 12-GiB guard on both allowed attempts (exit 137,
49s/48s, cleanup verified; peak RSS unavailable). The warm cell remains
unqualified and exhausted. Smaller retained qualified resolutions are not
relabeled as 8K coverage. No downsampling, ingress, guard or metric arithmetic
changed. See [the scalar deadline receipt](docs/IQA_SCALAR_PRIORITY.md#measured-8k-fsim-no-downscale-resource-refusal).

### Known Bugs: VIF-vector thread-budget score identity (2026-10-19)

The AIC execution inventory reproduces different VIF-vector scores at one and
eight workers. At 1024x1024: 0.6175632083942326 versus 0.6175632083957185;
at 3840x2160: 0.6184347550659302 versus 0.6184347550346455. Two repeats per
budget reproduce each value. The parallel covariance uses striped partial
sums while the one-worker path uses one raster-order accumulation. A correction
must remain on an unmerged fix bookmark under the owner rule; no metric path
was changed by the inventory. See
[the retained repeat panel](benchmarks/aic_refusals_2026-10-18/vifvec-thread-repeat.tsv).


The VIF correction is now qualified on unmerged `fix/vifvec-thread-order`
(source 0281c13e). Its exact regression fails before/passes after; 22 tests,
doctests, strict clippy and no-default-feature compilation pass. The measured
1024-square/4K one/eight-worker scores now match the original one-worker bits.
Master remains unchanged. Before/after timings, RSS and the affected-loop
retally are in `benchmarks/aic_refusals_2026-10-18/vif-thread-order.meta`.

### Known Bugs: STLPIPS-Alex Metal warm repeatability (2026-10-19)

Two retained 2048×1080 warm first attempts (one/eight workers) report
`variant_mismatch`, each with three differing call results and ten clean
resource-gate observations. Baseline bits `3f82881020000000`, observed alternate
`3f857d5180000000`. Both remain unqualified with verified cleanup and bounded
retries. The counter includes warmup and calibration; the varying timed rounds
and cause are unproven. No gate, worker or service changes were made for the
[read-only capture](benchmarks/mac_validation_2026-10-18/STLPIPS_VARIATION_2026-10-19.md).

The owner subsequently authorized finite-only Mac GPU timing acceptance with
logged error deltas and instability flags. A separate worker/bundle records
all nonfinite observations and finite extrema after the measurement clock.
Its six native controls pass all 40 warm resource gates; both original
STLPIPS-Alex 2K cases then pass their remaining production attempt as flagged
timings. Original failed packets and native-oracle checks remain unchanged.
This does not establish a precision-related cause. See the
[finite Metal policy receipt](benchmarks/mac_validation_2026-10-18/finite-metal-policy-2026-10-19.json)
and [timing contract](docs/IQA_DURABLE_MAC.md).

### Known Bugs: fixed-factor MS-SSIM pyramid ordering (2026-10-19)

The fixed-factor PyIQA MS-SSIM variants previously pooled the initial image
and built their pyramid from that pooled image. The corrected route pools
each scoring view independently while the twofold pyramid advances from the
unpooled level. It checks the 11×11 scoring window at every level, including
odd dimensions. Unsuffixed SSIM/MS-SSIM routes retain their existing path.
Tests compare against independent materialized scoring views and direct
window statistics; historical fixed-factor results must not be relabeled
as results of this correction. Source change: `970ea199`.

### Known Bugs: lean VMAF CLI feature gating (2026-10-19)

Enabling `cpu-vmaf` without `cpu-metrics` compiled the scoring dispatch but
excluded its local VMAF adapter module, resolving calls against the external
crate instead. The module now compiles for either feature.
`just metric-dispatch-check` covers the lean native consumer combination.

### Known Bugs: IQA due time and consent propagation (2026-10-19)

The 15-hour due time was incorrectly interpreted as permission to stop;
the Mac guard stopped unfinished CPU work at 13:01:10Z. A new immutable
no-stop successor resumed the existing remaining budgets at 14:44:56Z.
Maintained plans require explicit `stop_at_deadline=true` for a cutoff.

The i265 preparer passed literal `all` into the native named-consent parser.
Seven gated NN models therefore refused with missing `WeightsConsent` despite
explicit benchmark acceptance. The preparation path now expands explicitly
requested acceptance to named terms and records both forms; default empty
consent and the native gate stay strict. A separately owner-authorized retry
campaign preserves all old attempts and grants two new attempts only to
proven exhausted consent refusals. Other exhausted bounds/allocation/incomplete-round and process-exit
failures remain separately blocked. See [the owning contract](docs/IQA_DURABLE_MAC.md).

### Known Bugs: AIC execution refusal outcomes (2026-10-18)

The complexity repetition driver converted returned metric errors into panic
exit 101. The historical HDR-VDP 2/3 SDR requests and NR-disabled build therefore
looked like kernel crashes. It now preserves the refusal message and exits 2.
The 64-square inventory also found an explicit panic in the disabled
`zensim-gpu` dispatch arm; that arm now returns its existing diagnostic as an
error. A direct dispatch regression covers that outcome without changing scores
or enabling the disabled route. Evidence belongs in
`benchmarks/aic_refusals_2026-10-18/`.

### Known Bugs: portable Mac AIC derivative hashes (2026-10-19)

The ARM64 portable worker built from `fa8194c9` plus coordinator options from
`519a6ecb` refuses its newly prepared S67 inputs against the unchanged manifest:
reference and distorted bytes differ at cid22, aic-small, aic-medium and 2k.
The full-size and tiled 4k/8k files match. The encoded PNG hashes match;
the cause within preparation has not been established. No timing rows were
produced by that refused panel. Original generated bytes and hashes are retained.
The new Mac benchmark output uses copies of the previously frozen canonical
RGB files, verifying every original manifest hash before scoring. This does not
qualify the new native preparation path or relax its checksum gate. See
[the remeasurement record](benchmarks/iqa_changed_2026-10-19/README.md).

### Known Bugs: result-publication CI setup (2026-10-20)

A fresh dashboard runner lacked the disk-backed scratch parent used by the
comparative-shard tamper control. The fixture now creates that parent before
its temporary directory; its hash-mismatch expectation is unchanged. The
existing complexity-brief publication watcher is registered under its distinct
report-publication concern in the tooling inventory; no fleet launcher or
worker is added.

### Resolved: local canonical neural plan timeout (2026-10-20)

The first local original-network plans requested 1200 seconds per arm, above the
existing supervisor's 1100-second maximum. Both arms refused before execution;
these attempts contain no neural scores or timings. Original-network preparers
now validate their 1100-second budget against the actual supervisor before
writing plans. Replacements use new plan and result paths, retaining the refused
attempts. Numerical bounds and native warm-round gates are unchanged.

### Known Bugs: local canonical performance flags (2026-10-20)

The Ryzen 9 7950X / WSL2 resident-RGB comparisons flag FLIP and FLIP-v17 at
eight threads, and RTEN LPIPS-Alex at one and eight threads. Fixed-stage
Rust/canonical medians and descriptive 95% intervals are:

| Case (512×512) | Blocks | Rust/canonical | 95% interval |
| --- | ---: | ---: | --- |
| FLIP, 8 threads | 60 | 1.957927961720583 | 1.917466421937279–1.973160062353411 |
| FLIP-v17, 8 threads | 20 | 1.9073448386822038 | 1.873063545529841–1.9386291310864587 |
| LPIPS-Alex RTEN, 1 thread | 20 | 1.162649413650009 | 1.1455797183886958–1.1843805843799495 |
| LPIPS-Alex RTEN, 8 threads | 20 | 1.3014597690629857 | 1.2544969542532667–1.3354180677802032 |

The last interval misses ±3% precision; its fixed additional stage remains in
progress. SSIMULACRA1 and LPIPS-VGG one-thread intervals straddle 1.08 and are
borderline. These are comparison flags, with native resource flags retained,
not independently qualified regression acceptance. No optimization is merged.
The [local ledger](benchmarks/canonical_recheck_2026-10-19/README.md) owns the
full intervals, per-case process RSS, resource-wrapper receipts and profile
hashes. Its raw cohorts remain separate from the multi-host benchmark series.

### Resolved: original WaDIQaM benchmark CPU fixture (2026-10-20)

The first forward control caught the legacy reference's global
`chainer.cuda.cupy = numpy` alias confusing Chainer's own device dispatch.
Constructor-only readiness did not establish forward execution. The new
reference wrapper applies CPU aliases in the original author modules, with
gradient recording disabled. All six scalar outputs across the four models
on one 512×512 pair match bit-for-bit between gradient policies. Four
unexecuted plans were replaced with new identities and retained backups;
native settings, budgets and numerical bounds are preserved. See the
[reference-fixture control](benchmarks/canonical_recheck_2026-10-19/README.md#original-wadiqam-reference-setup).

### Benchmark integration finding, 2026-10-20

The new FLIP SIMD module is included directly by `cpu-profile`, so its
archmage and zenmetrics-math references must also be declared in that
benchmark package. A fresh build of `7d27d22b` exposed missing direct
dependencies and failed before scoring. The harness manifest repair passes a clean CPU benchmark build with the
AVX-512 dependency fingerprints verified. The build reports 144s, peak-RSS
0.88 GiB, minimum available 10210 MiB and peak load 6.51. Production metric
arithmetic is unchanged.

### DSSIM reference provenance, 2026-10-20

The local canonical recheck initially built its DSSIM reference from the CI
sibling checkout (`lilith/dssim` `eec33f55`). It is an older fork baseline,
not the author-repository checkout. Retain those observations as alternative
comparisons; author DSSIM 3.5.1 at `0e44c9b7` is being acquired separately.
See `benchmarks/canonical_recheck_2026-10-19/dssim-reference-lineage.json`.
No native code, numerical bound or expected value changes for this correction.

### Known Bugs: supported-runtime HDR-v2 followup (2026-10-21)

The [Octave 11.1/full image 2.20.1 controls](benchmarks/canonical_recheck_2026-10-19/hdr-octave111-v2-controls.json)
reproduce the retained native-v2 1 MP photo gap, 0.009351851113343912. The
original's difference from the legacy runtime is only 7.105427357601002e-15.
The legacy runtime does not explain this case. Cause unresolved; no numerical
bound, metric path or existing result changed. HDR timing remains withheld.

### Known Bugs: local BRISQUE 64² reference difference (2026-10-21)

The [retained photo finding](benchmarks/canonical_recheck_2026-10-19/brisque-64-numerical-finding.json)
has an observed absolute difference of 0.0865478515625, exceeding the unchanged
0.015138428 bound against PIQ0.8.0/torch2.14.1+cpu, repeating at both worker
settings. Timing ratios remain withheld and the bound is unchanged. Original-only
wheel/dispatch controls are pending; no root cause or exception is accepted.

### BRISQUE runtime controls completed (2026-10-21)

[Eight original-only controls](benchmarks/canonical_recheck_2026-10-19/brisque-runtime-controls-2026-10-21.json) use both installed torch 2.14.1 wheels
(+cpu and +cu130 running on CPU), with default AVX-512 and effective ATen AVX2.
Every configuration gives the same 64² score 75.73614501953125 and 255×257 score
89.107421875. Those tested wheel/dispatch choices do not explain the 64² native
difference. The computed R_GGD and R_AGGD grids differ from the frozen native
tables by one ULP in 329 and 327 entries respectively; this is not established
as the cause. The numerical failure and unchanged 0.015138428 bound remain
retained. No timing ratio or exception is accepted.

The CPU-wheel wrapper recorded 7 s / 0.33 GiB peak RSS / minimum available 14,227 MiB
/ peak load 2.79. The CUDA-wheel-on-CPU wrapper recorded 8 s / 0.80 GiB / 14,055 MiB
/ 2.52. Per-process GNU time receipts and exact source snapshots are retained.

### Backend and NIQE numerical follow-up (2026-10-21)

The [oneDNN controls](benchmarks/canonical_recheck_2026-10-19/brisque-backend-controls-2026-10-21.json) narrow
BRISQUE's 64² photo miss. With oneDNN's effective ISA limited to AVX2, the
unchanged PIQ source returns 75.65032958984375, versus Rust 75.64959716796875
(difference 0.000732421875). The retained default original is 75.73614501953125.
ATen-only AVX2 and the separate MKL AVX2 request do not change that default
score; the latter request does not prove MKL's effective ISA. Disabling oneDNN
gives 75.47998046875. These counterfactual diagnostics do not replace the
default canonical failure or its 0.015138428 bound. The wrapper recorded 7 s,
0.33 GiB peak RSS, minimum available 13,998 MiB and peak load 2.28.

The [native stage controls](benchmarks/canonical_recheck_2026-10-19/brisque-native-original-stages.json)
reproduce both retained native scalars exactly. Actual native and original
grayscale planes are bit-identical on both photos. The default original differs
in 20 of 36 statistical features at 64² and 26 of 36 at 255×257. The oneDNN
AVX2 control changes two 64² quantized shape parameters into exact agreement
with Rust. This narrows numerical attribution to statistical feature computation;
it does not clear the default-backend failure. The native diagnostic wrapper
recorded 1 s, 0.22 GiB peak RSS, minimum available 14,262 MiB and peak load 1.87.

The [NIQE odd-photo finding](benchmarks/canonical_recheck_2026-10-19/niqe-odd-numerical-finding.json) retains
three failing blocks at both 1 and 28 workers: Rust 18.807175290717275 versus
BasicSR 18.799147370663427, difference 0.008027920053848447 above the unchanged
0.000602329 bound. This is separate from the documented 64² size refusal.
Root cause and a code correction remain unverified. Earlier golden checks and
all original observations remain retained.

### NIQE feature attribution (2026-10-21)

The [stage cross-rescore](benchmarks/canonical_recheck_2026-10-19/niqe-native-original-stages.json) reproduces
both retained scalars before comparing intermediates. All 65,535 native grayscale
samples equal the original on the 255×257 photo. BasicSR's final MVG operations
on native block features yield 18.80717529071772, only 4.440892098500626e-13
from native's 18.807175290717275. Original features reproduce BasicSR's
18.799147370663427 exactly. The observed score gap is therefore attributable
to block-feature differences in this control, rather than RGB ingress or the
final native covariance calculation. Of 144 feature entries, 61 differ: 7 at
scale one and 54 at scale two. Two second-scale shape parameters differ by
one 0.001 grid step. The precise operation responsible remains unverified;
the numerical failure and original 0.000602329 bound remain unchanged.

Resource receipts: native diagnostic build/control 11 s, peak RSS 0.42 GiB,
minimum available 13,658 MiB, peak load 1.26; original feature control 2 s,
0.27 GiB, 13,840 MiB, 1.22; cross-rescore under one second, 0.04 GiB,
14,108 MiB, 1.18. These are diagnostic-process resources, not metric timings.
