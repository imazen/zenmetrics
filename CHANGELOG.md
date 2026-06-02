# Changelog

Workspace conventions per the global rules:

- One `[Unreleased]` section accumulates changes for the next release.
- Per-crate headings (`## cvvdp-gpu`, `## zen-metrics-cli`, …) sit under
  each version section since this repo ships multiple crates.
- `### QUEUED BREAKING CHANGES` accumulates breaks that need to land
  together — only cleared when the corresponding major (or minor for
  0.x) release ships.
- Every entry MUST include the short commit hash(es) that implemented
  it. Reference the merge or final commit for multi-commit features.

## [Unreleased]

### QUEUED BREAKING CHANGES

(none yet)

### Added

- **`zenmetrics-api` ideal public surface — PixelSlice front doors + intent hint (task #159 phases 4–5).**
  `score(kind, backend, ref, dist)` (one-shot, takes `zenpixels::PixelSlice`, `6b3f51f1`),
  `warm_reference(kind, backend, ref) -> Warm` (one reference, many distorted via score-identical
  buffer-replay, uniform across CPU/GPU, `f2dbc9de`), and `score_encoded(kind, backend, &[u8], &[u8])`
  (decode PNG/JPEG internally, `17371d9d`). `Priority {Speed, Memory}` + `Reuse {OneOff, Warm}` +
  observable `resolve_memory_mode(...)` map `(metric, size, reuse, priority)` to a score-safe
  `MemoryMode`, now driving the front doors' Auto choice (`18bca0b5`). `Backend::Auto` resolves to the
  optimized native `Backend::Cpu` when no GPU is present. New `tests/backend_matrix.rs` exercises every
  metric × backend × {256,512,1024}: CPU breadth + Auto→Cpu equivalence (`4985e69f`) and a measured,
  documented CPU-vs-CUDA parity layer (`fe9921a8`, baseline
  `benchmarks/backend_parity_cpu_vs_cuda_2026-06-01.tsv`). CI `cpu-metrics-tests` job + root `justfile`
  run the optimized-CPU suite GPU-less (`f5cd1f40`).
- **CI: workspace metadata + resolution unblocked (all jobs were red).** Cloned the two missing
  path/patch siblings `fast-ssim2` + `dssim` (`9c66efbb`), fixed the stale `../zenavif--main` /
  `../zenjxl--main` clone dirs to match the `--main`-dropped `[patch]` paths (`a07ea47e`), and
  re-pinned every cloned sibling to its current build-compatible `origin/main` HEAD (butteraugli
  0.9.2→0.9.4, etc.; `51559fda`).

- **`zenmetrics_api::score_pair` one-shot convenience.** `score_pair(kind, backend, w, h,
  ref, dist) -> Result<Score>` constructs the metric with default params, scores a single
  sRGB-u8 pair, and drops it — the "just score these two images" one-liner, without the
  `Metric::new` + `compute_srgb_u8` dance. Doc'd that it re-pays construction per call (use a
  held `Metric` / `MetricSession` for repeated scoring). Parity test vs manual new+compute.

- **MetricSession owned API + multi-warm session pool (task #155).** `zenmetrics-api` adds
  `OwnedSessionMetric` + `MetricSession::into_metric` — owned, long-lived, scorer-field-first
  drop ordering for exact per-entry VRAM reclaim — alongside the borrowed `SessionMetric<'ctx>`
  (`7d947b78`). The orchestrator adds a per-lane multi-warm LRU `WarmSessionPool` keyed on
  `(metric, dims, params, ref_hash)`, VRAM-budget-bounded with precise per-entry eviction
  (`4ead77ea`). Soundness gates (parity / no-OOM / reference-reuse) pass; a module lock fixes
  parallel-test counter contamination (`734b707f`).

### Fixed

- **zenstats `sa_st_curve` / PWRC no longer O(n²) MEMORY (OOM fix) (`b56d8ed2`).**
  The PWRC SA-ST AUC built a `Vec<(f64, bool)>` of all `n·(n−1)/2` pairs, so a
  panel over large `n` allocated `n²/2 × 16 B` — at `n ≈ 59k` (a codec-picker
  held-out panel of `val_rows × n_cells`) that is ~27 GB and OOM-killed the
  caller (observed 23 GB VmHWM, re-hit per hyperparameter-search candidate).
  `sa_st_curve` now computes the IDENTICAL `(ST, SA)` curve with a two-pass
  difference-array over the thresholds — O(n_points) memory, same O(n²) time,
  **bit-for-bit unchanged output** (new `sa_st_curve_matches_allpairs_reference_bit_for_bit`
  test asserts `f64::to_bits` equality vs the old all-pairs body across
  sizes/ties/anti-correlation/no-direction cases). Every `compute_panel` /
  `pwrc_sa_st_auc` caller benefits; no metric values change. Peak RSS on the
  triggering picker eval: 23 GB → 0.62 GB.
- **butteraugli-gpu Strip is now score-safe on all content (task #158).** The mode_wall sweep
  (`benchmarks/mode_wall_2026-05-31`) found butter's Strip score ~8% off Full on an aggressive
  high-frequency checkerboard. Root cause was NOT a halo bug: the umbrella `ButteraugliOpaque`
  routed `MemoryMode::Full` through the multi-resolution path (`new_multires`, full-res +
  half-res supersample) but `MemoryMode::Strip` through single-resolution (`new_strip`), silently
  dropping the half-res band — an apples-to-oranges comparison (single-res Strip == single-res
  Full bit-identically on the same checkerboard). `ButteraugliOpaque` now routes Strip/Auto→Strip
  through `new_multires_strip` (cuda/wgpu/cpu + `build_from_client`) so a Strip score matches the
  Full score on all content (`77020757`). Also bumped `HALO_ROWS` 40→80: the multires-strip
  half-res sibling is built by 2× downsampling the full-res strip slab, so it only saw `HALO_ROWS/2`
  real halo rows while the half-res blur cascade independently needs 34 — at 40 this drifted the
  max-norm ~7e-4 at 512² on the checkerboard; at 80 the half-res side is fully haloed and
  multires-strip == multires-whole bit-identically. New `tests/strip_hf_checkerboard.rs` gates both
  fixes on the exact divergent content (negative-controlled: reverts reproduce 8.0e-2 / 7e-4). Wall
  (`benchmarks/butter_strip_wall_task158_2026-05-31`): the corrected one-off Strip stays 2.5–13×
  faster than Full at 1024²/4096². `strip_parity` (21/21) + `multires_strip` (11/11) stay green.

- **Session VRAM reclaim now routes to all 6 metrics, not cvvdp-only.** `cleanup_session_stream`
  / `stream_reserved_bytes` (`session.rs`) routed only to `cvvdp_gpu::session::*` with a no-op
  fallback, so in a build with cvvdp compiled OUT but another metric IN, dropping a non-cvvdp
  `MetricSession` reclaimed nothing — silently violating the crate's "drop frees exactly this
  session's VRAM" guarantee. Both functions now use the 6-arm metric-agnostic fallback (mirrors
  `metric::reclaim_pooled_vram`); cubecl's `memory_cleanup` is keyed by stream value, so any
  enabled crate's `cleanup_stream` reclaims the session's pool. New
  `tests/session_reclaim_non_cvvdp.rs` proves it at runtime in the `ssim2,cuda` (no-cvvdp) config.
  Also corrected the `MetricSession` "not Sync" rustdoc (it is auto-`Send + Sync`) and added a
  compile-time `Send`/`Sync` contract assertion.

### Changed

- **Cross-metric `<M>Opaque` API alignment (internal -gpu crates).** The six
  metric `-gpu` crates now expose an identical reference-reuse core after the
  surface reduction exposed three divergent vocabularies (`cached_reference` /
  `warm_ref` / a zensim mix). Unified on a neutral "reference" vocabulary:
  every `<M>Opaque` has `set_reference_srgb_u8` / `compute_with_reference_srgb_u8`
  (→ `Score`) / `has_reference` (all 6) + `clear_reference` (5/6; cvvdp's warm
  cache is overwrite-on-set). Renames: `has_{cached,warm}_reference` →
  `has_reference`; `compute_with_cached_reference_srgb_u8` →
  `compute_with_reference_srgb_u8`; cvvdp `warm_reference_srgb` →
  `set_reference_srgb_u8` (diffmap kept as `compute_with_reference_srgb_u8_with_diffmap`);
  zensim's Score cached-ref → `compute_with_reference_srgb_u8`, its
  feature-vector variant → `compute_features_with_reference_srgb_u8`. Behavior is
  unchanged (pure renames; `ZensimInner` gained `has_reference`/`clear_reference`
  delegating to the pipeline). Also restored `zensim_gpu::pipeline` to
  `#[doc(hidden)] pub` (its own `strip_memory_demo` example reaches it; the
  reduction had wrongly made it `pub(crate)`, breaking the `--all-targets` build).
  Audit/alignment record in `docs/API_SURFACE_AUDIT_2026-06-01.md`.
- **Public API surface reduction across the workspace (internal crates).** The
  six metric `-gpu` crates exposed ~8,500 `pub` items (mostly cube-macro kernel
  machinery and internal pipeline modules) while their sole product consumer
  uses ~18 each. Demoted `ext_refs=0` modules to `pub(crate)` and marked
  cross-crate / own-harness internals `#[doc(hidden)] pub` (matching the
  existing `session` convention); the clean product API
  (`Backend`/`<Metric>Opaque`/`<Metric>Params`/`Score`/`MemoryMode` +
  `memory_mode` + `session`) is byte-identical. Per-crate `cargo public-api`:
  butteraugli-gpu 2525→244, cvvdp-gpu 1847→209, iwssim-gpu 1330→268,
  ssim2-gpu 1093→296, zensim-gpu 980→304, dssim-gpu 750→257 (`f0dc9bb8`,
  `d4a3e2fa`). Tier 0 also demoted 28 compiler-flagged `unreachable_pub` items
  to `pub(crate)` (cvvdp/orchestrator/cli/iwssim) and made `iwssim-filter-codegen`
  emit `pub(crate)` filter consts (`2944fbb1`). Audit + `#[doc(hidden)]`
  inventory: `docs/API_SURFACE_AUDIT_2026-06-01.md`.
- **`PoolConfig::multiwarm_session_pool` now defaults OFF (opt-in).** Measured
  (`benchmarks/multiwarm_session_pool_2026-05-30`): multi-warm is +1.30–1.47× at 256² but
  REGRESSES 2.3× (cvvdp) / 12.6× (ssim2) at 4096², where each warm entry is GiB-scale, the
  budget holds ~1 entry, and round-robin references thrash the LRU (evict+rebuild every task
  with full teardown vs single-warm's in-place `set_reference` reuse). Shipped opt-in until a
  capacity/thrash guard routes oversized working sets back to single-warm.

- **Job system: real-executor fleet image `ghcr.io/imazen/zen-jobworker-exec` + corpus plumbing**
  (2026-05-30). `crates/zen-jobworker/Dockerfile.executor` + `scripts/jobsys/build_executor_image.sh`
  bake the worker base + a prebuilt `zen-metrics` (with `jobexec`) + the `zen-jobexec` shim, with
  image-level `ENV ZEN_EXEC=/usr/local/bin/zen-jobexec` so a fleet box runs REAL encode/score jobs.
  Built + pushed (amd64; binary needs glibc ≤2.35, bookworm ships 2.36; runs in-image — verified real
  zenjpeg encode + ssim2 score through the container). `launch_fleet.sh` + `unraid_worker.sh` now pass
  `ZEN_CORPUS_PREFIX` (so `jobexec` resolves `cell.image_path` from R2) and an overridable `ZEN_EXEC`;
  set `ZEN_WORKER_IMAGE=…/zen-jobworker-exec:latest` + `ZEN_CORPUS_PREFIX=<prefix>` for real jobs.
  `docs/RUNNING_JOBS.md` updated. NOTE: the ghcr `zen-jobworker-exec` package is **private** — make it
  public (one-click, like the base image) for credential-less fleet pulls. arm64 image pending an
  arm64 `zen-metrics` build.
- **zenmetrics-api: `MetricSession` — opt-in isolated GPU context with ironclad VRAM-on-drop
  (issue #17, foundation)** (2026-05-30, `0053c0cc` + tests `f7b396f4`). New **opaque** public types
  `MetricSession` + `SessionMetric<'ctx>` + `MAX_SESSIONS_PER_BACKEND` + `Error::TooManyContexts`,
  implementing the approved `docs/VRAM_LIFECYCLE_DESIGN_2026-05-30.md` Option B.
  `MetricSession::acquire(backend)` claims a collision-free slot from a process-global
  128-slot-per-backend allocator and binds its metrics to a **private cubecl stream**; `Drop` runs
  `memory_cleanup()` + `sync()` on that stream (→ driver) then recycles the slot — reclaiming exactly
  this session's VRAM, independent of every other session, from any thread. `leak()` opts out
  (consume without reclaim), `reclaim()` is the idle hook; the 129th `acquire` errors
  `TooManyContexts` rather than silently alias a stream. `SessionMetric<'ctx>` is a distinct borrowed
  handle (no lifetime ripple on the owned `Metric`) that the borrow checker forbids from outliving its
  session (compile-fail-proven). Opaque: no `cubecl-types` in the public signature — the `unsafe
  set_stream` lives in each metric crate's `#[doc(hidden)]` `session` module, so `zenmetrics-api`
  stays `#![forbid(unsafe_code)]`. **All 6 metrics wired end-to-end** (cvvdp, ssim2, butter, dssim,
  iwssim, zensim) via each crate's existing typed `<Metric><R>::new(client)` seam — a `#[doc(hidden)]`
  cubecl-types-gated per-crate `session` module + an `<Metric>Opaque::build_from_client<R>` helper that
  mirrors the default-stream constructor's host-side mode/regime resolution on a stream-bound client
  (no behaviour change to the default constructors; no new per-crate *public* API). cvvdp landed first
  (`0053c0cc`), then ssim2 (`8b38dedc`), then butter/dssim/iwssim/zensim. Tests (cuda-gated):
  cap/recycle/leak; cvvdp one-shot + warm-ref parity within the measured `Atomic<f32>` reduction-noise
  band (~1e-6 — a kernel property, not the session); `session_parity_all_wired_metrics` loops every
  enabled metric (all 6) asserting owned-vs-session agreement within each metric's abs+rel noise band;
  **VRAM isolation measured via cubecl per-stream `memory_usage()`** — two sessions ~203 MiB each, drop
  one → its pool `bytes_reserved` 0 while the other stays resident, then the second → 0; plus a
  compile-fail borrow-check proof. `release()` / `reclaim_pooled_vram` / `MetricContext<R>` and all
  per-crate public APIs unchanged.
- **zenmetrics-api: `OwnedSessionMetric` — borrow-leash-free owned session metric (issue #17 / task
  #155 Phase A)** (2026-05-30). New **opaque** public type `OwnedSessionMetric` (exported next to
  `MetricSession`/`SessionMetric`) bundling a warm scorer with the `MetricSession` whose private cubecl
  stream it allocates on — so a warm metric can be stored past the scope that built it (the warm-pool
  entry shape) without the `SessionMetric<'ctx>` borrow leash. Built via the new
  `MetricSession::into_metric(kind, w, h, params)` + `into_metric_with_memory_mode(...)` (consume the
  session → one warm metric per isolated stream, the clean reclaim model). **Field order is the
  soundness lever**: `scorer` is declared before `session`, so Rust's declaration-order drop returns
  the scorer's cubecl device handles to the pool free-list *before* `MetricSession::Drop` runs
  `memory_cleanup()` + `sync()` on the stream — no live handle during cleanup, closing the
  use-after-cleanup hazard the borrowed `'ctx` leash guarded by construction. Forwards the same scoring
  surface as `SessionMetric` (`kind`/`dims`/`score`/`set_reference_srgb_u8`/`score_with_warm_ref`/
  `clear_reference`/`has_cached_reference`/`score_pixels`) via a shared private macro (no copy-paste),
  plus `backend()` and `leak()` (skip reclaim for a short-lived process); in-place `reclaim` is
  intentionally NOT offered (the welded live scorer makes it unsound — eviction = full drop only).
  `zenmetrics-api` stays `#![forbid(unsafe_code)]`. Tests (cuda-gated, RTX 5070): owned-vs-borrowed-vs-
  plain parity (cvvdp + ssim2 + warm-ref) within each metric's `Atomic<f32>` reduction-noise band;
  per-entry VRAM isolation (two owned metrics, drop one → its pool `bytes_reserved` 0 while the other
  stays resident, then 0); `into_metric` cap/recycle/leak. No existing public API changed.
- **Job system: real executor `zen-metrics jobexec` (encode + score) + CPU `sweep` build fix**
  (2026-05-30). New `zen-metrics jobexec` subcommand is the `ZEN_EXEC` reference executor: reads a
  `DesiredJob` JSON on stdin, resolves the source (local / `s3://` / `$ZEN_CORPUS_PREFIX` via s5cmd),
  and for an `encode` job emits the encoded bytes, for a `metric` job re-encodes the cell + scores it
  with `run_metric` and emits a JSON score row — honoring the stdin-JSON → stdout-bytes contract.
  Reuses `sweep::encode` + the unified `run_metric` (CPU metrics ssim2/butteraugli/zensim; GPU metrics
  return a clear "needs a GPU build" error). Proven end-to-end through the actual worker:
  declare → claim → jobexec (real zenjpeg/zenwebp encode + ssim2 score) → content-addressed blob +
  ledger row, blob sha256 == output_sha. `scripts/jobsys/zen-jobexec` is the single-program shim for
  `ZEN_EXEC`. Also fixes a pre-existing break: `cmd_score_pairs` referenced the `gpu-cvvdp`-gated
  `cvvdp_gpu` module unconditionally, so a CPU-only `sweep` build didn't compile; the cvvdp blocks are
  now gated (`#[cfg(feature = "gpu-cvvdp")]`) with a CPU-build early error, leaving the GPU build
  unchanged.
- **Job system: `docs/RUNNING_JOBS.md` + Unraid basement-tier setup + executor-contract template**
  (2026-05-30). Thorough end-to-end guide (mental model, executor contract, declare, fleet, Unraid
  basement tier, monitor, results, teardown/GC, worked example, real-job checklist). New
  `scripts/jobsys/unraid_worker.sh` mints a 7-day prefix-scoped R2 credential on the workstation and
  prints a ready-to-paste `docker run` / Unraid "Add Container" config for the NAT'd basement box
  (pull-based, no inbound ports, never the root key). New `scripts/jobsys/example_executor.py`
  documents + smoke-tests the `ZEN_EXEC` contract (stdin DesiredJob JSON → stdout output bytes →
  exit 0). Honest scope: orchestration is proven with the synthetic `/bin/cat` executor; a real
  encode/score executor is a defined contract you bake in (the `zen-metrics jobexec` reference impl is
  not yet built).
- **cvvdp-gpu: wgpu/Metal per-stream VRAM isolation spike (task #153 / issue #17)** (2026-05-30).
  Throwaway gated example `crates/cvvdp-gpu/examples/wgpu_isolation_spike.rs` (`required-features =
  ["wgpu"]`) mirroring the CUDA spike (#152) on the cubecl **wgpu** backend — the load-bearing proxy
  for Metal (no Apple GPU on this WSL2/NVIDIA host; wgpu auto-selected **Vulkan**). **POOL-LEVEL
  ISOLATION CONFIRMED**: cubecl-wgpu pools are per-stream (`WgpuStream` owns `WgpuMemManager` with 3
  `MemoryManagement<WgpuStorage>`; streams keyed `stream_id.value % max_streams`, default 128). A's
  per-stream pool went 1536→0 MiB on drop+`memory_cleanup()` while B's stayed 1536 MiB, then B's went
  1536→0; partial-page control + cross-thread reclaim both reproduced. Measured via cubecl per-stream
  `memory_usage()` — nvidia-smi is blind to wgpu/Vulkan here (per-PID AND total, extending #133).
  Driver-level OS return is wgpu's *lazy* decision (no `cuMemFreeAsync` flush) so not independently
  observable; pool-level reclaim is the load-bearing, confirmed signal. **Metal hardware confirmation
  needs a Mac — no Metal numbers fabricated.** Implication for `MetricSession`: ironclad at the pool
  level on wgpu/Metal, no best-effort `release()` fallback needed. Raw log committed at
  `crates/cvvdp-gpu/benchmarks/wgpu_isolation_spike_2026-05-30.txt`; verdict written into
  `crates/zenmetrics-api/docs/VRAM_LIFECYCLE_DESIGN_2026-05-30.md` §2b. No public API changed.
- **Job system: goal H CLOSED — ≥3 distinct physical providers proven concurrent on one queue**
  (2026-05-30, run `fleet-20260530-124834`). local (workstation) + Hetzner cpx22 (x86 cloud) + Salad
  (CPU container, distributed consumer network) all claimed+executed off ONE R2 lease-queue,
  pause-orchestrated (resumed only once Salad reached `running`). 120 jobs, ledger DONE rows by
  provider = **`{local: 69, hetzner: 27, salad: 24}` = 120, exactly-once**, distributed across all
  three with the fast node pulling more. Two fixes unlocked it: (1) **per-worker shuffled manifests**
  (`shuf_manifest` in `launch_fleet.sh`) — each worker gets the same job set in a decorrelated claim
  order, so the lowest-latency node no longer monopolizes every conditional-write race
  (`{local:60,others:0}` → distributed); still ONE queue / claims namespace, only iteration order
  differs. (2) Salad create routed through the reqwest example with `ZEN_EXEC` omitted (see the WAF
  notes). This satisfies H's literal "≥3 tiers concurrent" alongside the already-shipped
  provider-agnostic, capability-routed (GPU/CPU/ARM), and multi-arch-image sub-bullets.
- **Job system: Salad fleet tier (goal H, distinct provider)** (2026-05-30). `launch_fleet.sh` gains a
  5th arg `SALAD` that creates a CPU-only Salad container group (org `imazen`/project `zenmetrics`,
  env-overridable) running the public baked image — a distinct provider from local/Hetzner/vast on
  Salad's distributed consumer network, claiming off the same R2 queue (not Salad's managed queue;
  `restart_policy:never` so it drains its share and exits). `teardown_fleet.sh` DELETEs the group.
  `… 60 1 0 0 1` = local + Hetzner-x86 + Salad = 3 distinct providers on one queue. Salad API key live,
  verified against `organizations/imazen/gpu-classes` (HTTP 200). Added
  `crates/zen-cloud-salad/examples/fleet_create.rs` — the reqwest create path. **WAF gotcha, root-caused
  + fixed:** Salad's API is behind Cloudflare, whose managed ruleset 403s ("Attention Required!") any
  request body containing a `/bin/…` command path (command-injection rule). Bisected: body with
  `ZEN_EXEC=/bin/cat` → 403, identical body without → 201 (client/IP-agnostic across urllib/curl/
  reqwest). Fix: omit `ZEN_EXEC` from the Salad env — the entrypoint defaults it to `/bin/cat` inside
  the container, so it's behavior-identical and passes the WAF.
- **Job system: multi-arch (amd64 + arm64) fleet-worker image (goal H "Oracle ARM (free)" + ARM
  capability)** (2026-05-30). The worker image was x86-only, so the named free Oracle ARM tier and a
  Hetzner cax ARM box literally could not run it. `crates/zen-jobworker/Dockerfile` now selects the
  s5cmd + aws-cli downloads by buildx `TARGETARCH` (the bases + Rust build stage were already
  arch-native), and `jobworker-image.yml` builds each arch on its own native runner (`ubuntu-latest`
  amd64, `ubuntu-24.04-arm` arm64 — free on this public repo, no QEMU), pushes by digest, and merges
  one manifest asserting both `linux/amd64` + `linux/arm64`. `docker run ghcr.io/imazen/zen-jobworker`
  now resolves per-host arch, unblocking the ARM half of the heterogeneous fleet.
  `scripts/jobsys/launch_fleet.sh` gains a 4th arg `HETZNER_ARM_BOXES` that brings up Ampere `cax`
  (arm64) boxes as a distinct capability tier — `launch_fleet.sh 200 1 0 1` = local(x86) +
  Hetzner cpx(x86) + Hetzner cax(arm64), 3 concurrent tiers across 2 ISAs on one queue. Also fixes a
  latent launcher bug: hcloud takes user-data only via `--user-data-from-file`, not the previously-used
  (nonexistent) `--user-data-from-string`, which would have failed every Hetzner box created via the
  launcher.
- **Job system: capability routing (goal H "capability-routed GPU/CPU/ARM")** (2026-05-30).
  `zen_job_core::worker_serves` + `ResourceClass::parse`; `zen-jobworker --capability <class>…`
  (also `ZEN_CAPABILITY` env in the fleet entrypoint) makes a worker pull only jobs whose
  `JobKind::profile().class` it serves — empty = general worker. Live demo
  `scripts/jobsys/demo_capability_routing.sh`: on a mixed 15-job queue a GPU-capability worker did
  exactly the 6 metric (Gpu) jobs and a CPU-capability worker exactly the 9 encode (CpuLight/CpuHeavy)
  jobs — routed by hardware off one queue, no overlap.
- **Job system: baked fleet-worker image + launcher (goal H)** (2026-05-30).
  `crates/zen-jobworker/Dockerfile` bakes `zen-jobworker` + `zen-jobgc` + aws-cli v2 + s5cmd + a
  keep-alive entrypoint (`fleet-entrypoint.sh`) so a fleet box claims work with **zero boot-time
  installs** — designing out the two bugs a 2026-05-30 ad-hoc test hit (python-unzip dropping the aws
  exec bit; a container with no keep-alive). CI `jobworker-image.yml` builds + pushes
  `ghcr.io/imazen/zen-jobworker:{<sha>,latest}`. `scripts/jobsys/{launch,watch,teardown}_fleet.sh`
  bring up ≥3 interchangeable tiers (local + Hetzner + vast) on one R2 lease-queue with scoped temp
  creds, tearing down by `group=<run>` label. (That ad-hoc test already had Hetzner do 60 real jobs on
  the queue + the dashboard's Kill delete the live box; the image makes a clean 3-tier launch reliable.)
- **VRAM-on-drop design proposal + isolation spike (task #152)** (2026-05-30). Measured proof
  (`crates/cvvdp-gpu/examples/vram_isolation_spike.rs`, throwaway, gated behind `cuda`) that cubecl's
  CUDA memory pools are per-stream and free independently: dropping a context on its own explicit
  `StreamId` + `memory_cleanup()` + `sync()` returns exactly that context's pages to the driver while
  another context's pool stays resident (stream 101 freed 2272 MiB, 202 untouched; control: dropping
  1-of-2 co-resident allocs freed 0 MiB; cross-thread reclaim freed 2304 MiB). Design doc
  `crates/zenmetrics-api/docs/VRAM_LIFECYCLE_DESIGN_2026-05-30.md` recommends a new isolated-stream
  `GpuContext` type (Option B) over a bare `impl Drop` (Option A, non-ironclad on the shared default
  stream) and answers "do we need a new context type?" — yes. **No public API changed; implementation
  awaits user approval of the proposed shape.** Committed spike output
  `crates/cvvdp-gpu/benchmarks/vram_isolation_spike_2026-05-30.txt`.
- **Job system: safe GC execution (goal G) + ntfy notifications (goal D)** (2026-05-30). `zen_job_core`
  adds `lru_cap_evict` (bounded cheap-regenerable cache, evict LRU tail over a byte cap) + a `Tombstone`
  record. `zen-jobworker::gc_execute` + the new `zen-jobgc` CLI run a reachability GC: referenced blobs
  kept, unreferenced cheap evicted LRU-capped with a tombstone written first, unreferenced
  **irreplaceable refused** (surfaced, never auto-deleted), `verify_mirror` gating any non-regenerable
  delete; dry-run by default. Live `scripts/jobsys/demo_gc_r2.sh` / `examples/gc_live.rs` verifies all
  four guarantees against R2. The dashboard webhook sender is now **ntfy-aware** (`ZEN_NOTIFY_TOKEN` →
  message body + `Click` deep-link header + Bearer auth) alongside the Slack/Discord `{"text":…}` shape.
- **Job system: speculative execution (goal E) + speculative count (goal B)** (2026-05-29). A worker
  with `--spec-threshold-secs` co-runs a *live straggler* (primary claim older than the threshold but
  younger than the TTL) by taking a separate `claims/spec/<job_id>` claim — bounding the long tail; the
  ledger's latest-wins on `job_id` makes the loser a harmless duplicate. `/api/speculative`
  (`zen_ledger::list_keys_uri` over `ZEN_CLAIMS_R2/spec/`) surfaces the active count in the dashboard.
  Live demo `scripts/jobsys/demo_speculative_r2.sh` (slow primary + fast speculator → tail bounded,
  job converged).
- **zen-jobdash: thumbnails/diffmaps + ad-hoc query (goal B)** (2026-05-29). `/api/blob/{sha}` serves a
  result blob with a sniffed image content-type (JPEG/PNG/GIF/WebP/AVIF) + immutable cache, so encode
  and diffmap outputs render inline in the Results tab (thumbnail + full-size dialog). `/api/query`
  (`query_view`) is a structured filter over the Parquet ledger (kind/codec/status/image substrings,
  newest-first, capped) with a Query pane. Completes B's "peek results in-browser" except a SQL engine.
- **zen-jobdash: in-browser result peek (goal B)** (2026-05-29). `/api/results` lists recent Done rows
  carrying an output blob; `/api/peek/{sha}` fetches that content-addressed blob from R2
  (`ZEN_BLOBS_R2`) and returns its bytes as truncated text + size (hex-only sha guard against path
  traversal). New dashboard Results tab with a per-row "peek" dialog. Verified live — peeked a known
  R2 blob by hash (`cvvdp_jod=9.42`, 14 bytes).
- **Job system: idle-box reaping (goal F) + notification mechanism proven (goal D)** (2026-05-29).
  `zen_jobdash::idle_boxes` flags running fleet boxes with no matching worker heartbeat (billing,
  doing no work); `ControlIntent::ReapIdle` + a dashboard "Reap idle (N)" button tear them down via
  the Hetzner client, and `/api/fleet` now reports the idle set. The notification path (detect →
  format → POST with deep link) is demonstrated live against a local receiver by
  `scripts/jobsys/demo_notify_local.sh` (budget-crossed fired with the deep-link payload) — the only
  thing a production channel adds is the destination URL (`ZEN_NOTIFY_WEBHOOK`).
- **Job system: pause / resume / drain (goal C)** (2026-05-29). New `zen_job_core::RunControl`
  (`{paused,drain}`); `zen-jobworker --control-r2-key` reads it and pulls no new work when
  paused/draining (fail-open — absent = running), never touching the ledger so resume continues
  exactly where it left off. The dashboard's Pause/Drain/Resume now write this object to R2
  (`ZEN_CONTROL_R2`, via new `zen_ledger::write_bytes_uri`) instead of only recording intent. Live
  demo `scripts/jobsys/demo_pause_drain_r2.sh` (paused/draining → done=0, resume → done=1).
- **zen-jobworker: spot-preemption claim release (goal F)** (2026-05-29). On SIGTERM/SIGINT a worker
  with R2 claims now releases its in-flight claim (`release_claim_r2`) on a dedicated signal-hook
  thread and exits 130, so a reclaimed spot box requeues its job immediately instead of waiting out
  the claim TTL. Best-effort — TTL stale-reclaim (goal E) remains the correctness fallback. Live demo
  `scripts/jobsys/demo_spot_reclaim_r2.sh`; plus `scripts/jobsys/demo_e2e_r2.sh` demonstrating goals
  A/E/I + foundations end-to-end against an isolated R2 prefix.
- **zen-jobdash: shadcn control-plane SPA + auth + fleet actuation** (2026-05-29).
  Replaced the inline-HTML dashboard with a React + Vite + Tailwind v4 + shadcn/ui
  SPA (built in a Docker node stage, served by axum). Adds HTTP Basic Auth gated on
  `ZEN_DASH_PASSWORD` (`79e4743f`); a minimal inlined Hetzner client that **actuates**
  `KillFleet/Tier/Run`, scoped to a label-existence selector so unlabeled dev boxes are
  never touched, plus `/api/fleet` live-box visibility (`79e4743f`, `00d0fe4c`);
  per-worker monitoring `/api/workers` (provider·tier·$/hr·uptime·jobs/min·spent) and
  **stop-spend actuation** that tears down paid workers over budget while free tiers
  drain (`0152d11d`); and a result-catalog `/api/catalog` coverage view (semantic
  identity · q-range · done/total, find-by-description) for goal I (`74fd7f7a`).
  Deployed on Railway; goals B/C/F advanced and the goal-I catalog surfaced in-browser.
- **VRAM pool reclaim API + audit** (task #150, 2026-05-29). cubecl pools
  GPU buffers across `Handle` drop, so dropping a `Metric` returns its
  buffers to the pool free list but the device pages stay resident (the
  ~3830 MiB plateau task #147 measured). Added `reclaim_pooled_vram(backend)`
  to every `-gpu` crate's `memory_mode` module and to `zenmetrics-api`
  (re-exported), plus `Metric::release(backend)` (drop + reclaim). It calls
  cubecl `ComputeClient::memory_cleanup` then `sync` to flush the CUDA
  deferred-free queue, returning pooled pages to the driver. Thread/stream
  scoped (cubecl's CUDA pool is per-thread) — call from the thread that
  dropped the metric; never between warm scores. The orchestrator GPU
  worker now calls this reclaim at every metric-signature swap (after
  dropping the old warm instance, before constructing the new one) and on
  runtime OOM, so peak VRAM across a mixed-metric chunk stays ≈ MAX(single
  metric) instead of trending to SUM (`ZENMETRICS_NO_SWAP_VRAM_CLEANUP=1`
  opt-out). Source-level audit of the cubecl memory model + every crate's
  Drop / cached-ref lifecycle in
  `crates/zenmetrics-api/docs/VRAM_DROP_AUDIT_2026-05-29.md`. Measured proof
  (`crates/zenmetrics-api/docs/VRAM_DROP_MEASUREMENT_2026-05-29.md` +
  `benchmarks/vram_cleanup_2026-05-29.tsv`): 16 MP butter drop alone leaves
  +5000 MiB pooled, reclaim returns 4843 MiB to the driver (+157 ≈
  baseline); orchestrator 4 MP mixed chunk peak +2100 MiB with reclaim vs
  +2829 MiB without (1.35× lower); at 16 MP the no-reclaim variant OOM'd
  2/6 tasks while reclaim ran 6/6; warm cvvdp per-call path unchanged
  (p50 3.37 ms, VRAM floor 385 MiB, flat). Additive, non-breaking. Audit
  commit `946593f9`.

- **zenmetrics-orchestrator: cost-model-aware one-shot CPU/GPU routing**
  (task #146, 2026-05-29). The chooser previously ranked every backend
  purely on warm steady-state `ns_per_px`, which makes the GPU look
  unconditionally fast even for a single cold call where the ~181 ms CUDA
  context-init + per-signature construct + first-compute floor dominates.
  An audit (`crates/zenmetrics-orchestrator/docs/COST_MODEL_AUDIT_2026-05-29.md`)
  found this was the one lever diverging from the measured optimum — the
  warm pool's persistent-worker / cached-ref / cross-metric-context-sharing
  levers were already optimal. Added a new `ExecContext` (`Batch` /
  `OneShot`) and `choose_backend_with_context` /
  `choose_backend_for_task_with_context` that consult the measured one-shot
  crossover (`benchmarks/cpu_gpu_crossover_2026-05-29.tsv`): a `OneShot`
  call at/below the per-metric crossover size (cvvdp/ssim2/butter/zensim
  through 16 MP, dssim 4 MP, iwssim 1 MP) routes to CPU when CPU is a
  feasible candidate, else falls through to warm-`ns_per_px` ranking.
  `run_single` now routes with `OneShot`; the warm pool path keeps `Batch`
  semantics, so sweep/batch behavior is bit-identical. Additive, non-
  breaking — existing `choose_backend` / `choose_backend_for_task` keep
  `Batch` semantics. Audit commit `e2f9ab77`; implementation `0ba976ae`.

### Fixed

- **`per_ref` README table re-measured clean for all six metrics; the
  "iwssim first ref runs ~3× a subsequent ref" claim DEBUNKED** (task #151,
  2026-05-29). The `### per_ref` table and surrounding prose asserted
  iwssim @16 MP ran 196.5 ms on its first `set_reference` vs 67.4 ms on a
  subsequent one (a "~3× first-ref warmup"). That row was task #144's
  `gpu_inprocess_warmth` Q3 — **n=1, on a GPU contaminated by a concurrent
  zensim eval** — so the 196.5 ms was a transient. Re-measured every metric
  (cvvdp / ssim2 / dssim / iwssim / zensim + butter reconfirm) on a fully
  warm CUDA instance at 512² / 1024² / 2K / 16 MP, **n=8 per phase**, distinct
  pixels each rep, each `set_reference` synced inside the timed region
  (`crates/zenmetrics-api/examples/setref_all_timing.rs`, commit `eaff3219`;
  driving harness `scripts/setref_quiet_run.sh`). Finding: **five of six
  metrics are flat** (`setref1 ≈ setref2 ≈ setref3 ≈ setref4` at every size).
  iwssim @16 MP runs the **opposite** of the old claim — clean `setref1`
  68–74 ms is the *cheapest* phase, `setref2`–`setref4` are 120–163 ms (a
  first-ref *discount*, ~1.8×, not a penalty); iwssim is flat at 512²/1024²/2K.
  Every `setref1` phase shows one rep-1 transient (iwssim 248/265 ms; butter
  up to 4166 ms @16 MP) that n=8 median/min reject — exactly what n=1 sampling
  mistook for the phase cost in #144. Corrected the README `per_ref` table,
  the `per_ref` component bullet, the warmth-scope cross-reference, and added
  a CORRECTION note to `docs/GPU_INPROCESS_WARMTH_2026-05-29.md`'s n=1 table.
  Data: `benchmarks/setref_clean_all_2026-05-29.{tsv,meta}` (126 rows, RTX
  5070, cuda, no `-C target-cpu=native`).
- **README overhauled against current source** (task #149, 2026-05-29).
  Audited and corrected every stale claim: the crate table was missing
  `iwssim` / `iwssim-gpu` entirely and carried wrong versions
  (butteraugli 0.9.4, zensim 0.3.0, fast-ssim2 0.8.1); the memory-modes
  matrix wrongly claimed `ssim2-gpu` / `zensim-gpu` / `cvvdp-gpu` had no
  Strip support (all three do — verified each `src/memory_mode.rs`
  `ResolvedMode::Strip` + typed `new_strip`); the SRCC table is now
  flagged illustrative/external rather than implying an in-tree
  measurement; the butter `per_ref` figure was refreshed from task #148's
  clean re-measure (warm setref1 0.84 ms @512² / 22.16 ms @16 MP, not
  the 34/3990 ms first-instance alloc+JIT it conflated). Added a per-mode
  performance table (full / strip / warm_ref / warm_ref_strip, wall +
  peak working-set, CPU + GPU at 16 MP, all cells cited to committed
  TSVs), a modes×metrics support matrix, and an API-surface section with
  the exact calls per mode + the orchestrator's `ExecContext::OneShot`
  crossover. The `## Performance profile` section (#145) is preserved.
- **cvvdp CPU Path A strip-major dispatcher RECOVERED after a push-race
  orphan, re-verified, and re-measured** (task #127 recovery, 2026-05-29).
  The Path A work (`Cvvdp::new_strip` / `score_internal_strip` /
  `score_internal_strip_with_warm`, tip `2f8639a8`) had been LOST: the
  `cvvdp-path-a` jj workspace was forgotten and its 16 commits were never
  pushed to master, so "low-memory mode faster than full" was marked done
  on code that wasn't on the default branch. Root cause: the commits were
  pushed via `jj git push --change @-`, which auto-generated a throwaway
  bookmark that orphaned the chain instead of advancing `master`. Recovery
  rebased the 16 dangling Path A commits (merge-base `5979e084`) onto
  current master — no master commit had touched `crates/cvvdp/src` since
  the merge-base, so the rebase was clean (`eb925f13`, `4cec7e70`). The
  cpu_profile driver's `run_cvvdp` strip / warm_ref_strip arms construct
  via `Cvvdp::new_strip(w, h, params, 512)` (the real low-memory
  dispatcher), not the pool-only `score_strip` that produced the earlier
  `peak==full` table rows; the recovery and master's task #139 driver edit
  converged to identical content. Re-verified: 270-cell `strip_parity` big
  grid (cold + warm) bit-identical via `.to_bits()` (6/6 tests, 1207 s),
  `cargo test -p cvvdp` 196 passed / 0 failed. Re-measured on a quiet
  machine (heaptrack process peak + median of 7 release wall runs, no
  target-cpu=native): strip is BOTH lower-memory AND faster than full at
  16 MP (1.58 GB / 2.60 s vs 3.66 GB / 4.61 s) and 30 MP (2.55 GB / 4.55 s
  vs 6.54 GB / 8.48 s); see `crates/cvvdp/benchmarks/cpu_path_a_recovered_2026-05-29.{tsv,meta}`
  and the corrected cvvdp rows in `benchmarks/cpu_metrics_full_table_2026-05-28.tsv`
  (commits `eb925f13`, `7149bf61`).

- **zensim-gpu cold one-shot strip no longer builds the redundant
  full-image device ref XYB pyramid** (task #138, 2026-05-28).
  `Zensim::compute_features_vec` (the cold one-shot
  set-reference-then-one-compute entry, used by `compute_features`,
  the opaque score paths, and all parity tests) now routes strip-mode
  reference setup through `set_reference_host_cached_only` instead of
  `set_reference`. `set_reference` builds a full-image device ref XYB
  pyramid (task #75) that earns its keep only across MANY warm
  `compute_with_reference` iterations; building it for a single dist
  call was pure additive device-side overhead, making strip peak VRAM
  ≈ Full ("the mode that wins nothing"). The host-cached-only path
  rebuilds the ref XYB per strip — bit-exact to the device-cache
  row-slice (aligned strip starts) — so scores are unchanged. The
  warm-loop path (`set_reference` once + repeated
  `compute_with_reference`) is **untouched** and keeps the device cache
  for its speed benefit; `compute_features_vec` is never the warm
  entry, so no warm regression. Measured (RTX 5070, CUDA, reps=4,
  back-to-back same baseline): 16 MP strip 1249 → **289 MiB** (1.05× →
  0.24× of Full's 1185 MiB); 40 MP strip 1281 → **513 MiB** (0.58× →
  0.23× of Full's 2209 MiB). Score bit-identical (feat[0] 0.978717 @16
  MP, 0.978719 @40 MP; full == strip, old == new). All 10
  `tests/strip_parity.rs` pass including
  `host_cached_only_matches_device_cached_512x512`. Data:
  `crates/zensim-gpu/benchmarks/zensim_strip_remeasure_2026-05-28.tsv`.
  Also corrects stale `Error::ModeUnsupported` / `Error::TooBigForFull`
  doc + Display strings in `lib.rs` that claimed "zensim-gpu has no
  Strip path / implementation" — Strip + Auto have been implemented
  since task #75; only Tile is unsupported.

### Changed

- **butteraugli 0.9.3 → 0.9.4** workspace-wide pin bump (task #135,
  2026-05-28). Picks up the warm-ref +18 % peak-heap regression fix at
  16-40 MP (see `~/work/butteraugli/CHANGELOG.md` for the per-step
  decomposition and the 3-trial heaptrack medians). cpu_adapter
  `compute_with_cached_reference` (non-strip) and
  `compute_with_cached_reference_strip` paths continue to use the same
  `ButteraugliReference` cache; the smaller persistent footprint is
  transparent at the call site. No cpu_adapter code changes required —
  the new `drop_strip_source` / `shrink_to_fit` retention-control
  helpers are available if a future workload needs them.

- **iwssim cached-ref now routes through the strip walker** (task #136,
  2026-05-28). `CpuAdapter::compute_with_cached_reference` for
  `Metric::Iwssim` dispatches `iwssim::Iwssim::score_with_warm_ref_strip`
  with `iwssim::STRIP_BODY_DEFAULT`. The non-strip
  `score_with_warm_ref` retains the warm `lp_ref + g_ref + eigs` state
  but still builds full-image-sized dist-side scratch, so it cannot
  deliver heap savings; the strip variant carries the same warm state
  and adds the dist-side strip walker. Measured peak-heap reduction
  (heaptrack process peak, 7950X, synth pair, 3-trial median): **-33 %
  at 1 MP, -48 % at 16 MP, -48 % at 40 MP**. Wall regression:
  +9.5 % (1 MP) → +23.9 % (16 MP) → +34.7 % (40 MP); accepted because
  the cached-ref entry's value proposition is amortizing the ref-side
  eigendecomposition (which both modes do equally). Per-pair score diff
  ≤ 2e-6 absolute, inside iwssim's 1e-4 strip tolerance. Callers wanting
  a pinned body height continue to use
  `compute_with_cached_reference_strip` explicitly.

### Investigated

- **butteraugli-gpu `set_reference` has NO first-ref penalty on a warm
  instance — #144's 34 ms/3990 ms first-ref numbers were contamination,
  corrected; the reuse-path number is confirmed** (task #148,
  2026-05-29). #144 measured each `set_reference` exactly once (n=1) via
  the umbrella `inprocess_warmth.rs` Q3 driver on a machine where a
  concurrent zensim eval stole the GPU, yielding setref1 = 34.3 ms @ 512²
  / 3990 ms @ 16 MP vs setref2 = 0.76 ms @ 512² / 21.6 ms @ 16 MP — the
  apparent "first ref expensive, reuse free" picture. A clean re-measure
  on a quiet machine (0 GPU compute apps, strict quiet-window gate before
  every run, n=8–10 samples/cell, median + min) shows set_reference is
  ~constant per ref with NO first-ref exception: setref1 ≈ setref2 ≈
  setref3 ≈ setref4 at every size — 0.84/0.83/0.78/0.78 ms @ 512²,
  1.58/1.57/1.50/1.49 ms @ 1 MP, 6.2/5.9/5.9/5.7 ms @ 4 MP, and
  22.2/22.5/22.5/22.6 ms @ 16 MP. #144's *subsequent-new-ref* number is
  confirmed (its 0.76 ms / 21.6 ms ≈ this run's steady-state 0.78 ms /
  22.5 ms); its first-ref number was a single-sample contamination spike
  (3990 ms @ 16 MP is 180× the real value and 88× a full warm `compute`,
  which does strictly more work). The genuine per-ref precompute (upload
  + opsin + frequency separation + reference-only mask, all device-side,
  no readback — every timed call synced) scales roughly linearly in
  pixels above 1 MP with a sub-ms fixed-overhead floor. THIS clean run is
  authoritative. Driver: `crates/butteraugli-gpu/examples/setref_timing.rs`;
  data + provenance:
  `crates/butteraugli-gpu/benchmarks/butter_setref_clean_2026-05-29.{tsv,meta}`.

- **butteraugli-gpu has NO VRAM leak — repeated use plateaus, confirming
  #144's reuse claim** (task #147, 2026-05-29). Probed live VRAM
  (`nvidia-smi --query-gpu=memory.used`, min-of-4/5 reads after
  `client.sync()`) across three usage patterns × four modes (full /
  warm_ref / strip / warm_ref_strip) × two sizes (1 MP, 16 MP): (1) one
  instance, 100 scores; (2) one instance, 100 **distinct**
  `set_reference` calls (the path #144 flagged); (3) 30–40
  construct→score→drop cycles. All 22 cells **plateau** — lower-envelope
  end−start in [−33, +18] MiB (below the 48 MiB nvidia-smi quantization
  band; several negative). The headline: at 16 MP the working set is
  ~3.8 GiB and 100 distinct references hold at 3840 MiB and return to
  exactly 3840; 30 full reconstructions (each ~3.2 GiB) hold at ~3830 MiB
  instead of OOMing the 12 GiB card. Source confirms why: `Butteraugli<R>`
  holds a fixed handle set (no growing `Vec<Handle>`); whole-image
  `set_reference` overwrites planes in place (no new alloc), and the
  strip-mode `ref_cache_full` sibling is allocated once. This is healthy
  CubeCL pool reuse, not a leak. WSL2 hides per-PID GPU accounting
  (`--query-compute-apps` empty) so the global probe was used; a
  concurrent `zensim` eval contaminated the *fast* 1 MP cells of the first
  pass (one delta went to −1200 MiB — impossible for a leak), so those
  were re-run under a strict quiet-window gate with auto-retry. 16 MP
  plateaus matched the independent peak-VRAM sweep to ~5 %. Regression
  guard added: `crates/butteraugli-gpu/tests/vram_no_leak.rs` (3
  cuda-gated tests assert post-warmup floor growth ≤ 96 MiB; verified
  PASS on RTX 5070, growth 0 / 33 / −33 MiB). Driver:
  `crates/butteraugli-gpu/examples/vram_leak_check.rs`; data + provenance:
  `crates/butteraugli-gpu/benchmarks/vram_leak_check_2026-05-29.{tsv,meta}`;
  detail: `crates/butteraugli-gpu/docs/VRAM_LEAK_CHECK_2026-05-29.md`.

- **ssim2-gpu warm_ref VRAM "regression" is a measurement artifact, not
  a real retention bug** (task #138, 2026-05-28). The
  `gpu_metrics_sweep_2026-05-28.tsv` reading that whole-image `warm_ref`
  peaks above `full` at 40 MP (10.71 vs 9.23 GiB CUDA) is the cubecl
  dynamic memory pool sampled at different points on its growth curve
  for the two modes' differing per-call wall times under `WORKER_REPS=2`.
  Code: whole-image `warm_ref` and `full` both construct via
  `Ssim2::new` and share the identical 57-plane/scale `Scale` buffer set
  — `set_reference` allocates no persistent device buffers, only
  populating pre-existing slots, so there is no transient ref scratch to
  free (the brief's hypothesis matched cvvdp-gpu Mode E, not ssim2).
  Pool-stabilized re-measure (WORKER_REPS=8, CUDA, RTX 5070): 16 MP both
  6274 MiB; 18 MP 6273≈6271 MiB — byte-identical; at 40 MP both hit the
  same ~11.9 GiB pool ceiling and OOM. Score bit-identical (warm_ref ==
  full). Under reps=2 a re-run had `full` HIGHER at 16 MP, confirming
  ±60 MiB sampling noise. `warm_ref_strip` is the parity-safe
  memory-bounded 40 MP mode (measured 7.33 GiB, score bit-identical,
  inside the 5e-5 strip-parity gate). 8-GiB-safety would require either
  plain cold-ref `strip` (GATE-BREAKING: ~1.2e-3 rel score divergence,
  24× the gate) or widening the strip-parity tolerance (a relaxation —
  NOT done, needs user sign-off). Data:
  `crates/ssim2-gpu/benchmarks/ssim2_warmref_trim_2026-05-28.tsv`;
  detail: `crates/ssim2-gpu/docs/WARM_REF_VRAM_INVESTIGATION_2026-05-28.md`;
  sweep-doc correction noted inline in
  `docs/GPU_METRICS_SWEEP_2026-05-28.md`.

### Fixed

- **GPU VRAM estimators recalibrated — iwssim-gpu / dssim-gpu /
  cvvdp-gpu (task #137, 2026-05-28).** Three estimators under-predicted
  the measured GPU peak (committed in
  `benchmarks/gpu_metrics_sweep_2026-05-28.tsv`), which misled
  `resolve_auto` / the OOM-ladder into picking Full when a
  memory-bounded mode was needed. All three now OVER-predict (the safe
  budgeting direction) within ±20% at 4/16/40 MP.
  - **iwssim-gpu**: 10 → 19 planes/scale, +6.39 MiB reduction/cov
    scratch, ×1.40 pool factor, 256 MiB floor; doc comment corrected to
    Wang & Li Laplacian-pyramid IW-SSIM (no orientation dimension).
    `examples/strip_memory_tally.rs` scratch constants fixed
    (NUM_BLOCKS 32→16, COV cells 110→100·64). 16 MP peak/est 2.25× →
    0.90×.
  - **dssim-gpu**: 13 → 31 planes/scale (`Scale::new` = 9·alloc_3 + 4
    singles) + GPU-context term (208 MiB base + 18 B/px). 16 MP peak/est
    2.62× → 0.99×. `estimate_grows_with_pixels` test re-pointed
    1024²→2048² to 4096²→8192² (the fixed-context base compresses the
    1→4 MP ratio below the 3.6 quadratic floor; the test still guards
    quadratic growth at the larger sizes without relaxing the bound).
  - **cvvdp-gpu Mode B (strip_pair)**: source-faithful pyramid
    accounting (full gauss_ref, k≤k_split gauss_alt, baseband-only
    bands_dis) + the persistent full-n0 `DBandsTransient` (+826 MiB
    miss) + 256 MiB/32 B/px context. 16 MP peak/est ~3.9× → 0.89×.
    Mode E (warm_ref_strip) gets a 200 MiB floor + "not a memory win"
    comment; `resolve_auto` unchanged. The cvvdp Full estimator
    under-prediction remains (out of scope) and propagates into Mode E.
    `examples/mem_estimate_tsv.rs` added; the memory-audit Python proxy
    now shells out to it instead of a drifting hand-copied formula.
    Mode B estimator parity tests re-pointed from the old
    under-predicting "< 25%/65% of Full" bounds to over-predict the
    measured peak, + off-calibration body=128/512 tests added.

### Added

- **In-process GPU warmth transitions measured (task #144,
  2026-05-29, `cae53c7`).** New `inprocess_warmth` example in
  `zenmetrics-api` + harness
  `scripts/memory_audit/sweep_gpu_inprocess_warmth_2026-05-29.py`
  measure the four warmth transitions a single long-lived warm worker
  (single-warm-instance pool) pays, replacing prior architecture-inferred
  claims with committed numbers. Every timed score readback-syncs; every
  timed `set_reference` is followed by `block_on(client.sync())` so the
  wall is real execution, not async submission. Findings (RTX 5070,
  cuda, 512² + 16 MP, 5 fresh procs/cell): **Q1** a second metric in a
  warm process pays only ~190–290 ms @512 (its own alloc + kernel JIT),
  NOT the ~181 ms context init again — the CUDA context is paid once per
  process, ordering-independent (`client_init` 183–193 ms in all 4
  orderings); saving vs fresh-process cold_total is ~196–217 ms @512.
  **Q2** kernels are per-metric, context is shared (B always pays its own
  `first_compute` JIT). **Q3** a new warm_ref reference is NOT free for
  5 of 6 metrics (cvvdp/ssim2/dssim/iwssim/zensim re-pay ~0.5–2.8 ms @512
  / 14–67 ms @16 MP per new ref); **butter is the exception** — first ref
  34 ms @512 / 3990 ms @16 MP, subsequent new ref 0.76 ms / 21.6 ms
  (buffers reused). **Q4** full-mode different-ref-per-call costs nothing
  beyond per-call work. Data in
  `benchmarks/gpu_inprocess_warmth_2026-05-29.{tsv,meta}`, findings in
  `docs/GPU_INPROCESS_WARMTH_2026-05-29.md`. One 16 MP cell
  (ssim2→cvvdp) hit cvvdp's VRAM cap on this shared 12 GiB card and is
  recorded as `ALL_SAMPLES_FAILED` (not projected).

- **GPU metric cold-start wall measured per metric (task #140,
  2026-05-29, `50b1c83`).** New `coldstart_one` example per `-gpu`
  crate times the FIRST score call in a fresh process with the timer
  started before `Backend::client()` so CUDA-context-init is captured
  (the warm-only `gpu_metrics_sweep_2026-05-28.tsv` did not). Found:
  context init is a flat ~181 ms floor (size/metric-independent);
  one-shot cold_total is ~370–570 ms at 512² rising to several seconds
  at 16 MP (dominated by eager `new()` allocation for
  butteraugli/ssim2/cvvdp/dssim/iwssim; zensim allocs lazily); warm
  per-call is 1.5–50 ms. Cold PTX disk cache adds a metric-dependent
  JIT penalty (butteraugli ~4.2× first-compute, zensim ~1.3×).
  Implication: CPU wins one-shot small images by the full cold-start
  margin; GPU wins warm/batch at every size. Data in
  `benchmarks/gpu_coldstart_2026-05-29.{tsv,meta}`, findings in
  `docs/GPU_COLDSTART_2026-05-29.md`, harness at
  `scripts/memory_audit/sweep_gpu_coldstart_2026-05-29.py`. Drivers in
  `crates/<metric>-gpu/examples/coldstart_one.rs` (`0caf36d5`).

- **CPU-vs-GPU one-shot crossover table for all 6 perceptual metrics
  (task #141, 2026-05-29, `33962e8`).** Measured clean CPU full-mode
  zenbench wall (7950X, release, NO `-C target-cpu=native`) for
  ssim2/dssim/butter/iwssim/zensim (cvvdp via the same harness too) at
  512²/1024²/2048²/4096² (16 MP) + 12 MP + 30 MP — 198 rows, every cell a
  real interleaved zenbench run with a score sentinel. Joined with the
  task #140 GPU cold one-shot (`cold_total_ms`) and GPU warm per-call to
  get the size at which a single cold-process score flips from CPU-faster
  to GPU-faster. Findings: cvvdp/ssim2/butter/zensim CPU-faster at ALL
  measured GPU-cold sizes (even 16 MP); dssim crossover between 4.2 and
  16.8 MP; iwssim crossover between 1.0 and 4.2 MP. Batch/warm: GPU faster
  at every size for all 6. Data in
  `benchmarks/cpu_wall_all_metrics_2026-05-29.{tsv,meta}` +
  `benchmarks/cpu_gpu_crossover_2026-05-29.tsv`, human table in
  `docs/CPU_GPU_CROSSOVER_2026-05-29.md`, synthesis in
  `benchmarks/synth_crossover.py`. Also added a `4096` size label and a
  `CPU_WALL_NO_GATE=1` toggle to the cpu-wall harness (`604d057a`) — the
  default zenbench resource gate's per-round full-process scan dominated
  wall time on this ~1000-process box (dssim@512 31 s → 4 s with the gate
  off). NO EXTRAPOLATION: GPU cold only measured ≤16 MP, so CPU 12/30 MP
  cells are flagged `GPU-cold unmeasured >16MP`.

- **Phase 9.Z.F Path A (2026-05-28) — cvvdp CPU strip-major dispatcher
  shipped.** Lands the architectural change that drops `score_strip`
  peak heap from 3.66 GB → 1.55 GB at 16 MP (under 1.7 GB target) and
  8.68 GB → 3.24 GB at 40 MP (under 4.2 GB target). New
  `Cvvdp::new_strip(width, height, params, h_body)` constructor
  pre-allocates `Scratch::new_strip` with strip-shape persistent
  weber slots. The `score_strip` cold path runs through a new
  `score_internal_strip` that:
  1. Builds full-image gauss pyramids for ref + dist, dropping DKL
     planes inline per channel.
  2. Builds full-image weber bands for DEEP levels only (k >= k_split).
  3. Strip-major outer loop: for each strip s, for each shallow k,
     builds per-strip weber band data on-the-fly using chunk-5 strip
     kernels then runs CSF + masking + pool, accumulating into
     per-level LpNormAccumulators.
  4. Combines shallow + deep q values via existing JOD pooling.

  `score_with_warm_ref_strip` routes through a parallel
  `score_internal_strip_with_warm` path: when `warm_reference` is
  called on a strip-mode `Cvvdp`, the warm cache holds the ref gauss
  pyramid (not the weber pyramid) plus pre-built DEEP weber bands.
  Subsequent `score_with_warm_ref_strip` calls build dist gauss + dist
  deep weber + strip-major dispatch (reading from cached ref gauss).

  Wall + heap RE-MEASURED 2026-05-29 after the recovery below (quiet
  machine, fresh recovered binary, heaptrack process peak, wall = median
  of 7 t_score_ms release runs, NO target-cpu=native). Strip is both
  LOWER-MEMORY and FASTER than full at 16 MP and 30 MP:
  - 16 MP strip:          1.58 GB / 2.60 s  vs full 3.66 GB / 4.61 s  (-57% / -43%)
  - 16 MP warm_ref_strip: 1.55 GB / 2.62 s  vs warm 3.15 GB / 4.48 s  (-51% / -41%)
  - 30 MP strip:          2.55 GB / 4.55 s  vs full 6.54 GB / 8.48 s  (-61% / -46%)
  - 30 MP warm_ref_strip: 2.49 GB / 4.60 s  vs warm 5.64 GB / 7.82 s  (-56% / -41%)
  Measured crossover: at 512^2 the per-strip scratch costs slightly MORE
  heap than full (strip 0.061 vs full 0.056 GB) while still being wall-
  faster; the heap win appears at 1024^2 and widens to 30 MP. Full 6-size
  sweep + provenance in `crates/cvvdp/benchmarks/cpu_path_a_recovered_2026-05-29.{tsv,meta}`.
  (The earlier "11.32 s → 6.04 s" figures in this entry were an unverified
  prior estimate; superseded by the re-measured numbers above.)

  Bit-identical parity: 6/6 parity tests pass including the new
  `strip_parity_default_grid_new_strip_cold` /
  `strip_parity_default_grid_new_strip_warm` (90 cells each) for the
  `Cvvdp::new_strip` constructor path, plus the existing 270-cell
  big_grid_cold / big_grid_warm tests under the
  `cvvdp-strip-parity-big` feature flag (f7e293b6, 9a40f318,
  fbc52cab, f848c55f, 11e406d0).

- **Phase 9.Z.F (2026-05-28) — cvvdp CPU strip-aware kernel ports
  (chunks 1, 2, 3, 5 + chunk 4 data structure).** Ports the GPU's
  six K_SPLIT strip-aware kernels to scalar CPU helpers in a new
  `crates/cvvdp/src/strip_kernels.rs` module. The kernels are
  prerequisites for the CPU strip-major dispatcher (chunks 4-wiring +
  6) that will reduce `score_strip` peak heap from 4.73 GB to ≤1.7 GB
  at 16 MP and enable 40 MP CPU support (≤4.2 GB target).

  Shipped:
  - `pu_blur_h_strip_aware_3ch_into` + `pu_blur_v_strip_aware_3ch_into`:
    σ=3 PU blur with logical-h reflection then strip-local translation.
    H-blur is degenerate (X-only); V-blur is load-bearing.
  - `downscale_strip_into`: 2× reduce with pycvvdp parity-on-rows
    bug-compat keyed on `logical_src_h`. Caller-owned vscratch.
  - `upscale_v_strip_into` + `upscale_h_strip_into`: 2× expand
    body-only V + X-only H.
  - `subtract_weber_3ch_strip_into`: per-pixel weber-contrast +
    log_l_bkg.
  - `StripBandWorkspace` in `scratch.rs` + `Scratch::strip_band_ws`
    optional slot + `ensure_strip_band_ws(k_split)` helper. Allocated
    lazily (None in Full mode → no allocation cost).

  16 bit-identical (`to_bits()`) parity tests prove every kernel
  matches the existing CPU full-mode reference at the body rows of
  arbitrary strips. Chunks 2 + 3 (per-pixel CSF + masking) are
  strip-degenerate per audit AUDIT_2026-05-28.md §A rows 8-11; no
  new kernel needed.

  Chunks 4 (full wiring) + 6 (strip-major dispatcher) remain queued.
  See `crates/cvvdp/docs/CPU_KSPL_HANDOFF_chunks_4_and_6.md` for the
  architectural design, peak-heap decomposition, and remaining
  Scratch refactor work (de57ab01, f4a80cd6).

- **Phase 9.Z.B follow-on (2026-05-28) — orchestrator CpuAdapter
  strip dispatchers for ssim2 / butter / zensim.** Now that
  `fast-ssim2 0.8.1` (`compute_ssimulacra2_strip` +
  `Ssimulacra2Reference::compare_strip`), `butteraugli 0.9.3`
  (`butteraugli_strip` + `ButteraugliReference::compare_strip`), and
  zensim's `compute_streaming_strips` are available on crates.io and
  in tree, the `compute_strip` /
  `compute_with_cached_reference_strip` paths route through them.
  `supports_strip()` returns `true` for ssim2 / butter / zensim /
  iwssim; `false` for dssim (no upstream strip API on dssim-core 3.4)
  and cvvdp (API stub delegates to full path). Default
  `strip_height = 256` matches the documented sweet spot for fast-ssim2
  and butteraugli strip walkers. Adds 5 strip-dispatch tests covering
  cold + warm-ref paths for all three new metrics (8cfee48b, 70fed1c2).

- **Task #134 (2026-05-28) — wire zensim cached_ref through cpu_adapter.**
  Switches `CpuAdapter::supports_cached_ref(Metric::Zensim)` from `false`
  to `true` and wires the real warm path:
  `set_reference` calls `Zensim::precompute_reference` (owned
  `PrecomputedReference`); `compute_with_cached_reference` dispatches
  `Zensim::compute_with_ref` against the cached precompute;
  `compute_with_cached_reference_strip` dispatches
  `Zensim::compute_with_ref_streaming_strips` so the warm amortization
  carries into memory-bounded strip mode. The prior wiring stashed raw
  bytes in `Option<Vec<u8>>` and re-ran the cold path on every warm
  call, so the orchestrator's cached-ref dispatcher never landed on a
  warm code path even when callers set a reference hint.

  Bench (water-cooled 7950X, 16 MP / 4096×4096, 3-trial median, 10-
  distorted amortized sweep, `target/release/cpu-profile`):
  `full_n10`: 383.19 ms / call, `warm_ref_n10`: 337.67 ms / call,
  speedup +11.9 % per amortized warm call (≈44 ms one-time precompute,
  break-even after one call). The brief cited `+46 %` as the target —
  that figure is from the GPU CUDA / wgpu cached-ref sweep at
  `benchmarks/zensim_cached_ref_2026-05-22.csv` (38 % on CUDA / 40 %
  on wgpu at 1024², 10 distorteds), not from a CPU sweep. The CPU win
  is structurally smaller because `compute()` already does a fused
  joint-pass over ref + dist, so the precompute hoist removes less of
  the per-pair work. The 11.9 % CPU speedup that lands is real and
  structural; reporting `+46 %` would have meant extrapolating from
  the GPU figure. Full bench data + provenance in
  `benchmarks/zensim_cached_ref_cpu_2026-05-28.meta`. The heaptrack
  driver's `warm_ref` mode now exercises the real warm pair (the prior
  driver fell back to `compute()` for `warm_ref`, masking the
  speedup that did exist). Adds two adapter tests: warm-vs-cold
  byte-exact parity at 256² and warm-ref-strip vs warm-ref-full
  parity within zensim's documented strip tolerance.

### Changed

- **Phase 9.Z.C recovery (2026-05-28) — bump zenforks-cubecl-cpu to
  0.10.2.** Ships the multi-cube `SharedMemory` + `sync_cube`
  isolation fix that unblocks cvvdp-gpu's 73×91 odd-dim parity test.
  Re-enables `compute_dkl_jod_host_pool_matches_pycvvdp_at_73x91_odd_on_cpu_backend`
  and adds 6 odd-dim downscale_kernel diagnostic regression tests on
  the cubecl-cpu runtime (33ee5283, 575d3257).
- **Phase 9.Z.B recovery (2026-05-28) — bump fast-ssim2 to 0.8.1 and
  butteraugli workspace dep to 0.9.3.** Required to unlock the strip
  walker APIs in the sibling crates (01ea6991, b282084d).
- **Phase 9.Y — eliminate `chunks_exact(3).collect()` materialization
  in the four CPU adapters that previously built `Vec<[u8; 3]>` /
  `Vec<RGB<u8>>` intermediates before handing buffers to the underlying
  metric crate.** Phase 9.X heaptrack report attributed 240 MB / pair
  of adapter overhead to this pattern at 40 MP. `[u8; 3]` and
  `rgb::RGB<u8>` are both `bytemuck::Pod`, so we reinterpret the raw
  sRGB-u8 interleaved bytes in place via `bytemuck::cast_slice` — zero
  allocation, zero copy, zero `unsafe` in the adapter. Sites swapped:
  `ssim2_image_ref` (fast-ssim2 `ImgRef<'_, [u8; 3]>`),
  `make_dssim_image` (dssim-core's `create_image_rgb` shortcut),
  `compute_butter` (butteraugli `ImgRef<RGB<u8>>` pair),
  `compute_zensim` (zensim `RgbSlice<&[[u8; 3]]>` pair). Heap delta:
  ssim2 −245 MB / butter full −245 MB / zensim −245 MB at 40 MP;
  dssim 0 MB peak (upstream pyramid hides it) but −2 allocs and
  −14 % wall time. Phase 9.X heaptrack driver mirrors the same swap
  so its accounting matches the production adapter pattern.
- **Phase 9.Y — wire butter cached-ref to `ButteraugliReference`.**
  The prior `ButterState` stored raw bytes in `Option<Vec<u8>>` and
  the warm-ref path recomputed `full` against the cached bytes —
  same score, no speedup. butteraugli 0.9.2 has had
  `ButteraugliReference::new(&[u8], …, params)` + `.compare(&[u8])`
  since its precompute landed; this change wires it up. The
  precompute runs sRGB → linear → XYB → frequency-separated bands
  → reference mask once and reuses across compares (half-resolution
  mirror in parallel via rayon). `supports_cached_ref()` flips
  from `false` to `true` for butter; the pool's worker now routes
  butter through the warm path. Per-compare wall time −12 % at 40 MP
  (5.72s → 5.05s). Peak heap rises +1.15 GB / 40 MP because the
  reference state stays live during compare — this is the expected
  trade for batched workloads (N compares vs one ref): peak stays
  flat as N grows, vs N × full for the prior recompute path.
  Parity tests added for cvvdp / ssim2 / dssim / butter in
  `tests/cpu_backend.rs` confirm warm path produces the same score
  as one-shot. (`benchmarks/heaptrack/PHASE9YB_DELTA_REPORT.md`)

- **`iwssim` / `iwssim-gpu` — single source of truth for filter-tap
  codegen (Phase 8j Part A).** Both crates' `build.rs` scripts now
  call into a new internal helper crate
  `iwssim-filter-codegen` for the BINOM5 / SSIM_WIN_1D / SCALE_WEIGHTS
  emission. Previously the two build scripts contained verbatim-
  duplicated `binom5_taps()` / `gaussian_1d()` functions kept
  in-sync by hand. Each crate still emits its own
  `OUT_DIR/filters.rs` (cube-macros in `iwssim-gpu` capture
  `crate::filters::*` paths at expansion time, so a single
  consolidated module is structurally impossible); the new helper
  guarantees the two generated files are bit-identical by
  construction. Closes the Phase 8g.1 architectural debt
  documented at the top of `crates/iwssim-gpu/src/filters.rs`.

### Fixed

- **`ssim2-gpu` — wgpu 4096² dispatch limit unblocked (task #53,
  2026-05-28).** Split `pipeline.rs::cube_count_1d` (and
  `fir_cube_count`) into a 2D dispatch when `cubes > 32768`, keeping
  each grid dimension well under wgpu's 65535-per-dim cap
  (`Limits::downlevel_defaults`). At 4096² the scale-0 kernel grid
  needs 65,536 cubes (n = 16,777,216 / TPB = 256), which is +1 over
  the wgpu cap. The kernels read `ABSOLUTE_POS` as a flat linear
  index, which cubecl computes as
  `(absolute_pos_y * cube_count_x * cube_dim_x) + absolute_pos_x`
  on both CUDA and wgsl backends, so the 2D split preserves the
  index each thread sees and the kernels need zero changes.
  Removes the `cfg(feature = "cuda")` gates on four 4096²
  tests (`strip_parity_4096_body1024`, `strip_parity_4096_body2048`,
  `strip_cross_tile_size_4096`, `aliasing_pair_path_4096`); all
  four now run on both backends. Full suites: 30/30 strip_parity
  and 14/14 aliasing_invariants pass on both wgpu and cuda.

- **Phase 9.Y finding #5 — zensim strip mode no longer re-precomputes
  reference per strip.** The cpu_profile driver's `strip` mode now
  hoists `Zensim::precompute_reference` outside the strip loop and
  calls `compute_with_ref_streaming_strips_default` (zero-copy ref
  slicing per strip). Peak heap at 40 MP drops from 3.59 GB → 2.96 GB
  (−16.1 %, −580 MB), matching the prior `warm_ref_strip` baseline.
  Score is bit-identical across 1 / 16 / 40 MP sizes
  (80.45223298546662 / 80.45277977209128 / 80.45113394427749).
  Wrapper-pattern + production-caller guidance documented in
  [`crates/zenmetrics-api/docs/ZENSIM_STRIP_WARM_REF_HOIST.md`](crates/zenmetrics-api/docs/ZENSIM_STRIP_WARM_REF_HOIST.md).
  Residual +13 % at 40 MP vs. `full` is intrinsic to the strip
  walker's dst-XYB scratch + ref-pyramid hold and would require
  modifying the external zensim repo to close. Section 9.1 appended
  to the heaptrack report logs the post-fix matrix.

- **`cvvdp-gpu` — three pre-existing test failures restored to green
  (Phase 8j Part B).** Each handled per its root cause:
  - `pipeline_score::score_returns_lossless_f64_widening_of_compute_dkl_jod`
    relaxed from `to_bits()` equality to a `1e-4 abs` tolerance
    band. The test calls `score()` followed by `compute_dkl_jod()`
    in separate GPU dispatches; the pool kernel uses
    `Atomic<f32>::fetch_add` whose reduce order is non-deterministic
    across runs, surfacing as a 2-ulp delta at q=1 in release mode.
    The widening contract from `Cvvdp::score` (`f64::from(jod)`) is
    still pinned via the in-test `is_finite()` + `[0, 10]` range
    checks; the 1e-4 band matches the
    `perf_mode_fast_matches_strict_today` precedent.
  - `strip_mode_e_parity::mode_e_strip_h_body_explicit_override`
    test value updated from `Some(768)` (= 3 × STRIP_ALIGN) to
    `Some(1024)` (valid power-of-two). The `Cvvdp::new_strip` /
    `new_with_memory_mode` constructor contract was tightened to
    require `h_body.is_power_of_two()` after this test was written;
    1024 is the smallest valid value above the
    `STRIP_H_BODY_DEFAULT` and still exercises the "explicit
    override survives round-trip" property the test was pinning.
  - `cpu_backend::compute_dkl_jod_host_pool_matches_pycvvdp_at_73x91_odd_on_cpu_backend`
    marked `#[ignore = "task #80 — …"]`. The cubecl-cpu backend
    returns ~7.71 JOD at 73×91 vs the pycvvdp golden 9.39 (1.7 JOD
    drift), but CUDA matches at 0.0004 JOD
    (`pipeline_color::compute_dkl_jod_matches_pycvvdp_at_73x91_odd`)
    and 32×32 cpu matches host_scalar bit-equal
    (`compute_dkl_jod_host_pool_matches_host_scalar_on_cpu_backend`).
    The drift is too large to be f32 noise; suspected root cause
    is cubecl-cpu mis-translating boundary-handling branches in
    `downscale_kernel` for odd input dimensions. Tracked as
    upstream-cubecl-cpu work under task #80 rather than papered
    over with a tolerance widen.
### Added

- **Phase 9.Z.A — cvvdp `score_strip` + `score_with_warm_ref_strip`
  API stubs.** API surface added matching the iwssim shape, but
  these methods currently delegate to `score()` /
  `score_with_warm_ref()` and **do not yet reduce peak heap**. The
  cvvdp memory-bounded strip walker is multi-day refactor work
  (9-level Weber pyramid + per-band σ=3 PU blur produces cumulative
  halo `~8 × 2^k` rows at scale 0 for level k → at level 8 of 4096²
  that exceeds any sensible strip body; hybrid K_SPLIT dispatch
  required per the GPU cvvdp Mode E design at
  `crates/cvvdp-gpu/docs/STRIP_PROCESSING.md`). The stubs ship now
  to unblock orchestrator `MemoryMode::Strip` / `CachedStrip`
  wiring for cvvdp without API churn when the walker eventually
  lands. 3 tests (`crates/cvvdp/tests/strip_stub.rs`) pin stub ==
  full equality.
- **Phase 9.Z.A — iwssim `score_with_warm_ref_strip` + `_gray`
  cached-ref strip scoring.** Best-of-both memory profile for
  batch sweeps: ref state cached full-image in `WarmState`
  (`lp_ref + g_ref + per-scale eigs`); dist walked per-strip;
  single-pass walker (no Pass 2 — eigendecomp lazily cached on
  first warm-strip call). Per-strip dist working set ≈ 150 MB at
  40 MP vs 5.58 GB warm-ref Full. `WarmState` gains `eigs:
  Vec<Option<EigResult>>`; exposed as `pub(crate)` so the strip
  module accesses it directly. 5/5 parity tests pass against
  `score_with_warm_ref` within 1e-5 to 1e-4 abs JOD; single-strip
  case matches at < 1e-6.
- **Phase 9.Z.A — orchestrator CpuAdapter strip dispatch.** Three
  new methods on `CpuAdapter`: `compute_strip`,
  `compute_with_cached_reference_strip`, `supports_strip()`. Per-
  metric routing:
    - iwssim: real walker (returns `true` for `supports_strip`).
    - cvvdp: routes to the stubs (returns `false` so chooser does
      NOT pick CPU-strip for cvvdp yet).
    - ssim2 / dssim / butter / zensim: surface `Failed` until
      wired (zensim's `compute_strips` API exists upstream — next).
  3 new cpu_adapter tests pin behavior under each routing branch.
- **Phase 9.Z.A — iwssim `score_strip` + `score_strip_gray` strip-mode
  scoring.** Two new public methods on `iwssim::Iwssim` for memory-
  bounded scoring on 40 MP+ inputs. Walks the image in horizontal
  slabs of `strip_height` rows plus a `STRIP_HALO_ROWS = 320` halo
  per side (clamped at image edges via the same `reflect1` semantics
  the full pipeline uses). Two-pass walker:
    - **Pass 1**: per-strip Laplacian pyramid + accumulate per-scale
      `Y^T·Y` into a global `(big_n × big_n)` f64 matrix per scale,
      plus the top-scale (no IW) cs sum directly.
    - **Eigendecomp** (per scale, once): `C_u = sum(Y^T·Y) /
      nexp_total` → `decompose_and_invert`.
    - **Pass 2**: per-strip Laplacian pyramid (rebuilt — eigendecomp
      depends on global state) + compute `cs` + `infow` per strip
      using the global eigendecomp; accumulate `Σ(cs·iw)` and `Σ(iw)`
      partials.
    - **Finalize**: `wmcs[s] = sum_csiw / sum_iw` per IW scale,
      top-scale `wmcs = mean(cs · l)`; final score `Π wmcs[s]^β[s]`.
  Memory profile: per-strip peak is `(body + 2*halo) × work_w × 4 ×
  ~5` (5-level pyramid staged in flight) — at 40 MP with body=512
  that's roughly 150 MB vs 5.9 GB Full mode. Wall-time penalty ~1.5×
  vs Full because Pass 2 rebuilds the per-strip pyramid (Pass 1's
  pyramid can't be cached without giving up the memory win).
  Parity: 9/9 tests pass against full `Iwssim::score` across
  256² / 512² / 1024² / 512×256 at strip heights 128 / 256 / 512;
  single-strip case matches at < 1e-6 abs JOD; multi-strip at <
  1e-4 abs JOD. Constants `STRIP_HALO_ROWS = 320`,
  `STRIP_BODY_DEFAULT = 512`, `STRIP_BODY_MIN = 64` exported.
- **Phase 9.YA — cvvdp Scratch DKL plane reuse + weber pyramid output
  pre-allocation; iwssim Scratch struct for sRGB plane reuse.**
  Addresses two of the P0/P1 actions ranked by the Phase 9.X heaptrack
  report (`crates/zenmetrics-api/docs/CPU_HEAPTRACK_REPORT_2026-05-27.md`).
  - **cvvdp Part 1 (`e428bf08`)**: removed `Cvvdp::warm:
    Option<ReferenceState>` (which owned its own `[Vec<f32>; 3]` DKL
    planes + display + weber pyramid + dims) in favour of
    `Cvvdp::warm_active: bool`. The DKL planes now live in the
    persistent `Scratch::ref_*` buffers (already pre-allocated at
    `Cvvdp::new`) and the per-channel weber pyramid in
    `Scratch::weber_ref`. The new `build_one_side_warm_ref_into`
    helper uses LOCAL `WeberPyramidCache` slots that drop at function
    exit — persisting them in `scratch.weber_cache_ref` would have
    pushed peak heap up by ~640 MB in the warm path with no benefit.
    `ReferenceState` removed from `lib.rs`. Measured at 40 MP
    (7000×5728): warm_ref peak heap 9.30 GB → 8.82 GB (−480 MB =
    predicted DKL-plane saving); full mode unchanged (already used
    scratch). Score bit-identical.
  - **iwssim Part 1 (`77d424fd`)**: added `iwssim::pipeline::Scratch`
    struct with `ref_gray` / `dis_gray` (W×H f32) + `ref_work` /
    `dis_work` (work_w × work_h f32). Replaces 4 per-call
    `alloc::vec![0.0; w*h]` invocations (≈ 640 MB of churn at 40 MP)
    with in-place writes via the new `pad_gray_into(src, dst, …)`
    associated function. `score()` / `score_gray()` /
    `warm_reference()` / `score_with_warm_ref()` all route through
    the persistent scratch via inner `score_gray_internal` /
    `warm_reference_gray_internal` / `score_with_warm_ref_gray_internal`
    helpers. Peak heap at 40 MP unchanged (5.90 GB; bottleneck is
    `compute_iw_maps`, not entry-side buffers), but alloc churn
    drops by 4 mallocs per call. Score bit-identical
    (0.9988008961043233).
  - **cvvdp Part 2 (`dc13235d`)**: added
    `WeberPyramid::with_capacity(sw, sh, n_levels)` and
    `WeberPyramidCache::with_capacity(sw, sh, n_levels)` constructors
    that pre-size every per-level `Vec<f32>` in the output bands +
    log_l_bkg arrays and the dist-side cache's gauss_img + gauss_l
    bands. `Scratch::new` now takes `n_levels` and uses these
    constructors for `weber_ref` / `weber_dist` (used by all paths)
    and `weber_cache_dist` (used by cold + warm). `weber_cache_ref`
    is intentionally left at `Default::default()` — only the cold
    `score()` path uses it, so pre-allocating would burn ~640 MB in
    the warm path. Inner `PyramidScratch` buffers (vscratch,
    expanded, gauss_tmp) remain lazily allocated by gausspyr_reduce
    / gausspyr_expand — pre-allocating to the finest-level worst
    case pushed peak above the natural runtime peak by 1.4 GB in
    iteration. Net effect at 40 MP: full mode peak unchanged
    (11.31 GB, bound by concurrent 6-pyramid state); alloc count
    553 → 518 (−35 mallocs/process). Score bit-identical at
    1 MP / 16 MP / 40 MP.
  - Heaptrack artifacts (4 × ~16 KB zst) committed to
    `benchmarks/heaptrack/` as `{cvvdp,iwssim}_{full,warm_ref}_40mp_post9ya_part{1,2}.zst`
    for traceable before/after comparison with the Phase 9.X baseline.
- **Phase 9x — CPU heaptrack gate report
  (`crates/zenmetrics-api/docs/CPU_HEAPTRACK_REPORT_2026-05-27.md`).**
  Profiles all six CPU metrics (cvvdp / ssim2 / dssim / butter /
  iwssim / zensim) × four execution modes (full / warm_ref / strip /
  warm_ref_strip) × three sizes (1 MP, 16 MP, 40 MP) under
  `heaptrack 1.3.0`. Produces a ranked low-hanging-fruit list for
  Phase 9.Y optimization. Key findings: cvvdp peaks at 11.31 GB at
  40 MP (YELLOW for concurrent batch); only zensim has stripwise
  APIs (the other 5 metrics need them for 80 MP+); cvvdp + iwssim
  reallocate entry-side planes per call despite having `Scratch` /
  `&mut self` already (P0 fix). New driver crate `cpu-profile`
  under `benchmarks/heaptrack/drivers/cpu_profile/` (workspace
  member, never published) — single-call profiling harness with
  deterministic synth-pair fixture. Heaptrack artifacts (72 cells
  × ~16 KB zst) committed to `benchmarks/heaptrack/`. Companion
  parser (`parse_heaptracks.py`) and matrix runner
  (`run_matrix.sh`) checked in alongside the report.
- **`iwssim` crate (Phase 8g) — pure-Rust CPU port of Python-IW-SSIM
  with magetypes SIMD.** Faithful port of the canonical Python-IW-SSIM
  reference (Jack-guo-xy/Python-IW-SSIM, commit `f9de37cd`) for
  Wang & Li's Information-content Weighted SSIM (IEEE TIP 2011). The
  GPU port (`iwssim-gpu`) shipped in Phase 4 but had no CPU sibling
  — Phase 6 documented this as an explicit honest-stop. Phase 8g
  closes that gap. Public API mirrors `cvvdp`'s pattern:
  `Iwssim::new` / `score` / `warm_reference` /
  `score_with_warm_ref`. SIMD coverage: 11×11 separable Gaussian
  (SSIM stats), per-pixel cs/l combine, weighted-sum pooling — all
  routed through `archmage::incant!` with the
  `[v4x, v4, v3, neon, wasm128, scalar]` tier cascade.
  Pyramid / box-stat / IW-weight-map paths remain scalar this
  release. Parity vs Python reference: max `|diff| 8e-5` in
  `[0, 1]` score space across 7 deterministic synthetic fixtures
  (176/256/320 × identical/offset/shift1px/swap). Goldens captured
  via `crates/iwssim/goldens/capture_python_goldens.py`; fixtures
  are reconstructed locally from 32-bit seeds (no PNGs committed).
  Commits: `76dbdd46` (scaffold), `faa7f58e` (goldens + parity
  test), `60c3e2b8` (SIMD pass), `a2b80b18` (orchestrator wiring).
- **`zenmetrics-orchestrator` Phase 8g — `cpu-iwssim` feature
  + adapter.** Wires the new `iwssim` crate into the orchestrator's
  CPU backend ladder. Now `MetricKind::Iwssim` surfaces a real
  `Backend::Cpu` candidate (not `CpuMetricUnavailable`); the
  cached-reference path is promoted from `false` to `true` (true
  warm path via `Iwssim::warm_reference`). Added to `cpu-all`.
  Commit `a2b80b18`.
- **Phase 8c.1 audit — `crates/zenmetrics-api/docs/CRATE_GRAPH_AUDIT.md`.**
  Surveys the six metric crate pairs, inventories shared types, and
  ranks opportunities. Key finding: `cvvdp` (CPU) currently depends on
  `cvvdp-gpu` — the inverse of the user's stated principle that GPU
  versions should depend on CPU versions. All other metric pairs are
  either independent (butter/dssim/ssim2) or already correct (zensim
  → its CPU sibling). The audit recommends Phase B.1 flip the cvvdp
  dep direction by moving pure constants + scalar reference functions
  out of `cvvdp-gpu::{params, kernels::*, host_scalar, presets}` into
  `cvvdp`. zenpixels interface is already consistent across all 6
  `-gpu` crates (no B.3 gaps to fill).

### Changed

- **Phase 8g.1 — `iwssim-gpu` now depends on `iwssim` (gpu→cpu dep direction).**
  Shared algorithm constants `NUM_SCALES` (= 5) and `MIN_NATIVE_DIM`
  (= 176) moved from `iwssim-gpu` to `iwssim`; the GPU crate keeps them
  reachable via `pub use iwssim::{NUM_SCALES, MIN_NATIVE_DIM};` so
  existing `iwssim_gpu::*` callsites resolve unchanged. The build.rs-
  generated filter tables (`BINOM5`, `SSIM_WIN_1D`, `SCALE_WEIGHTS`)
  stay duplicated in `iwssim-gpu/src/filters.rs` because they're
  referenced by-name inside `#[cube(launch_unchecked)]` kernel bodies
  (`lap_pyramid::corr_dn_*_kernel`, `up_conv_*_kernel`, `gauss11`'s
  separable taps) — cube codegen captures the `crate::filters::*`
  path at macro expansion and re-emits it on device, so a re-export
  from another crate breaks resolution. The two `build.rs` files are
  bit-identical (same coefficient generator) and a `parity_cpu` test
  catches any drift. The `iwssim/tests/parity_gpu.rs` test moved to
  `iwssim-gpu/tests/parity_cpu.rs` to break the dep cycle (`iwssim`
  no longer optionally depends on `iwssim-gpu`); run via
  `cargo test -p iwssim-gpu --features cuda --test parity_cpu -- --ignored`.
  Parity unchanged: 9/9 PASS-EXACT on the orchestrator iwssim parity
  sweep (256/1024/4096 × q={20,50,80} JPEG synthetic) and 5/5 PASS on
  the GPU↔CPU `parity_cpu` test suite. Audit reference:
  `crates/zenmetrics-api/docs/CRATE_GRAPH_AUDIT.md`.

- **Phase 8c.1-C — `cvvdp-gpu::kernels::{diffmap,color,csf,pool,pyramid,masking}`
  collapsed to `pub use cvvdp::kernels::*::*` re-exports.** Follow-up to
  Phase 8c.1-B's audit-flagged kernel-constant duplication. All six
  kernel files now hold ONLY the `#[cube(launch)]` GPU kernels plus
  the small set of GPU-launch-config constants that those kernels
  reference at module scope (`POOL_LDS_BLOCK_DIM`, `POOL_LDS_BLOCK_DIM_USIZE`,
  the `DOWNSCALE_TILED_*` workgroup-tile constants). Scalar constants
  (`MASK_P`, `MASK_Q`, `XCM_3X3`, `PU_BLUR_KERNEL_1D`, `BASEBAND_W`,
  `GAUSS5`, `KERNEL_A`, `SRGB8_TO_LINEAR_LUT`, the 32×32×3 CSF LUT
  tables, …) and scalar host helpers (`lp_norm_*`, `met2jod`,
  `gausspyr_*_scalar`, `srgb_byte_to_dkl_scalar`, `sensitivity_scalar`,
  `mult_mutual_band`, `gaussian_blur_sigma3`, `phase_uncertainty_band`,
  …) now have a single canonical owner in `cvvdp::kernels::*` and a
  re-export shim in `cvvdp_gpu::kernels::*` so existing imports
  (`cvvdp_gpu::kernels::pool::lp_norm_mean` etc.) resolve unchanged.

  The audit doc flagged the cube-macro name-resolution interaction as
  the main risk; in-source verification (parser stripping doc comments
  + line comments) confirmed that NO `#[cube(launch)]` kernel
  references any of the moved scalar constants by name inside its
  cube body. Every cube kernel uses inline `f32::new(...)` literals
  for the cvvdp constants (a cube-IR requirement — Rust `const`s don't
  cross the macro expansion barrier). So the cube macro is unaffected
  by the re-export.

  PTX bit-identity verified per file: for each kernel file, the
  before/after `cargo expand --release -p cvvdp-gpu --features cuda`
  output of the cube-macro-emitted `pub mod <kernel_name> { ... }`
  block hashes identically (sha256). 34/34 cube kernels across all 6
  files pass — guarantees the cubecl IR (and hence the runtime-emitted
  PTX) is bit-identical before vs after.

  Tests redistributed: 6 pyramid-scalar tests + 3 csf interp1_rho_extrap
  conformance tests moved from `cvvdp-gpu/src/kernels/*.rs` inline
  modules to `cvvdp/src/kernels/*.rs` to follow the canonical owners.
  cvvdp lib tests went 43 → 52; cvvdp-gpu lib tests went 14 → 5; sum
  is unchanged.

  Commits per file: `a8bee1ae` (diffmap), `49447c6a` (color),
  `01effa89` (csf), `c9f1a366` (pool), `a8261f5a` (pyramid),
  `a526b0b2` (masking). (Note: per-file commits were rebased; these
  SHAs reflect the pre-rebase chain — see `jj log master..@` after
  rebase for the post-rebase SHAs.)
- **Phase 8c.1-B — `cvvdp-gpu` now depends on `cvvdp` (gpu→cpu dep direction)**
  (cc4046fe). Shared params, host_scalar reference impl, presets +
  vendored JSON data files, and the scalar portions of the kernel
  files (pool / masking / pyramid / csf / color / diffmap constants
  and reference functions) moved from `cvvdp-gpu` to `cvvdp`. The GPU
  crate's `cvvdp_gpu::{params, presets}` modules are now thin
  `pub use cvvdp::{params, presets}::*` shims so existing callsites
  resolve unchanged. cvvdp-gpu retains its own copies of the kernel
  scalar constants alongside the `#[cube(launch)]` kernels (the cube
  macros reference them by-name in the module scope; making them
  re-exports would require careful cube-macro name-resolution work
  deferred to a follow-up). The duplicated constants are bit-identical
  between the two crates — verified by 43/43 cvvdp lib tests passing
  + workspace builds clean + parity sweep cell values matching the
  `phase771_run3` baseline bit-for-bit (8/9 cvvdp cells PASS-EXACT,
  cvvdp 4096/q=20 PASS within 1.4e-4 JOD tolerance, matching baseline
  exactly). Audit reference: `crates/zenmetrics-api/docs/CRATE_GRAPH_AUDIT.md`
  section A.5. Phase 8c.1-C above closes the deferred follow-up.
- **`zenmetrics-orchestrator` Phase 8h — ssim2 CPU adapter switched
  from upstream `ssimulacra2` 0.5 to Imazen's SIMD-accelerated
  `fast-ssim2` 0.8.** Per the global crate index, `fast-ssim2` is our
  in-house SIMD SSIMULACRA2 implementation; the `ssimulacra2 0.5`
  pin used by Phase 6's initial wiring (commit 0fc139a3) was a stop-
  gap. The adapter now consumes `imgref::ImgRef<[u8; 3]>` directly via
  fast-ssim2's `ToLinearRgb` impl, skipping the manual
  `Xyb::try_from(Rgb::new(...))` transcode the prior path required.
  fast-ssim2 also exposes a true cached-reference path
  (`Ssimulacra2Reference`); `set_reference` now precomputes the
  reference once and `compute_with_cached_reference` reuses it,
  promoting `CpuAdapter::supports_cached_ref()` from `false` to `true`
  for `MetricKind::Ssim2`. Per-call score may shift by atomic-add /
  SIMD-reorder tolerance vs. the prior implementation; the score scale
  (~0-100, 100 = identical) is unchanged. Production callers using
  `--use-orchestrator` see the change transparently; legacy
  `--use-legacy-scheduler` path is unaffected (still uses ssim2-gpu's
  GPU implementation, which has not changed). `ssim2-gpu`'s parity
  tests against the upstream `ssimulacra2` crate are untouched —
  they exercise a separate concern. Verified locally: `cpu_backend`
  test suite all 10/10 pass; `ZENMETRICS_FORCE_NO_GPU=1` ssim2 task on
  256² synth offset-distortion pair scored 79.67 (sensible
  noticeable-degradation range). See
  `crates/zenmetrics-orchestrator/docs/CPU_BACKENDS.md` for the full
  swap rationale.
- **Phase 8f — switch from `lilith/cubecl` git-rev pins to
  `zenforks-cubecl-*` crates.io publication.** The 11 patched-or-
  transitive cubecl crates are now published to crates.io under the
  `zenforks-cubecl-*` namespace from
  [imazen/zenforks-cubecl](https://github.com/imazen/zenforks-cubecl).
  The `[lib] name = "cubecl_*"` shim means workspace source code keeps
  writing `use cubecl_runtime::*;` unchanged — only `Cargo.toml`'s
  `[workspace.dependencies]` switches from `git = "lilith/cubecl"` to
  `{ package = "zenforks-cubecl-*", version = "0.10.1" }`. Five
  non-renamed leaf crates (`cubecl-common`, `cubecl-ir`,
  `cubecl-macros`, `cubecl-macros-internal`, `cubecl-zspace`) continue
  to come from upstream `tracel-ai/cubecl` at 0.10.0. zenforks-cubecl
  0.10.0 ships vanilla rename + pinned-upload patch; 0.10.1 adds the
  persistent PTX cache widening and the Metal `Atomic<f32>` capability
  honesty fix. Full maintenance playbook in
  `crates/zenmetrics-api/docs/ZENFORKS_CUBECL_STRATEGY.md`. Original
  fork-strategy doc (`CUBECL_FORK_STRATEGY.md`) marked superseded.

### Fixed

- **`zenmetrics-orchestrator` Phase 8g.2 — retire stale
  "iwssim has no CPU backend" test assumptions.** Phase 8g landed
  iwssim's in-tree CPU reference and Phase 8g.1 extracted shared
  constants, but four orchestrator integration tests still asserted
  the pre-8g shape and broke under `cuda,cpu-all`. Updated:
  `tests/cpu_backend.rs::iwssim_cpu_unavailable_advances_ladder`
  now splits into `iwssim_cpu_constructs_and_computes_256`
  (cpu-iwssim ON — asserts `Backend::Cpu` selection + finite score)
  and the original `iwssim_cpu_unavailable_advances_ladder` gated
  on `not(cpu-iwssim)`. Same split for
  `tests/no_gpu_fallback.rs::{run_single_iwssim_no_gpu_no_cpu_returns_chooser_error,
  iwssim_with_force_no_gpu_returns_chooser_error_end_to_end}`
  (positive cpu-iwssim variants land on Cpu via the no-GPU ladder).
  `tests/chooser.rs::rejects_negative_extrapolated_cpu_prediction`
  gated on `cpu-ssim2` (the test exercises the Ssim2 chooser path —
  without cpu-ssim2 the Cpu candidate is rejected as
  `CpuMetricUnavailable` before reaching the negative-extrapolation
  guard). Full orchestrator suite under `cuda,cpu-all`: 126 passed,
  0 failed, 17 ignored. Under `bench,cuda` (no cpu-* features):
  the `not(cpu-iwssim)` branches still compile and pass.

### Added

- **`zenmetrics-orchestrator` Phase 9.1 — N-lane GPU pool with
  round-robin dispatch.** The worker pool's single GPU worker is now
  a `Vec<mpsc::Sender<WorkerTask>>` sized by `PoolConfig::max_gpu_lanes`
  (clamped to 1..=8, default 1). Each lane is its own OS thread
  holding a warm `ExecMetric` and consuming from its own mpsc queue;
  cubecl's MultiStream backend (default `max_streams = 128`) auto-
  assigns each thread a distinct `cudaStream_t` via thread-local
  `StreamId`, so N > 1 lanes run kernels concurrently on independent
  CUDA streams. New `Orchestrator` API: `gpu_lane_count()`,
  `active_gpu_lanes()`, `gpu_utilization_pct()`,
  `adaptive_lane_tick()`. `PoolConfig` gains `max_gpu_lanes`,
  `target_gpu_utilization_pct`, `adaptive_max_gpu_lanes`,
  `gpu_util_sample_interval_ms`, and `adaptive_gpu_lanes` — the last
  four wire Phase 9.3's adaptive controller. Default config preserves
  single-worker behaviour (max_gpu_lanes=1, adaptive_gpu_lanes=false);
  existing tests pass unchanged. (23e26c9e)

- **CI Phase 8e.5 — Metal CI matrix expanded + iwssim-gpu added to
  per-metric parity step.** `.github/workflows/ci.yml` `metal-tests`
  job matrix now includes `macos-15-intel` alongside `macos-latest`
  (Apple Silicon) and `macos-26-intel`, satisfying CLAUDE.md's
  "Every crate MUST also test on a macOS Intel runner" requirement
  for both Intel image generations. `iwssim-gpu` added to the
  per-metric test list (audit confirmed no `Atomic<f32>` use in its
  hot path). `cvvdp-gpu` deliberately deferred from metal-tests
  until the upstream cubecl-wgpu CAS-loop lowering lands — its
  production pool path uses `Atomic<f32>::fetch_add` and would
  always fail. Per-crate READMEs (`butteraugli-gpu`, `dssim-gpu`,
  `cvvdp-gpu`) gain an explicit "Metal status" section so operators
  see the current state without diving into Cargo.toml comments.

- **`zenmetrics-api` Phase 8e.4 — Metal `Atomic<f32>` root-cause doc +
  per-crate workaround audit + upstream patch draft.**
  `crates/zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md` identifies
  the bug site (`cubecl-wgpu/src/backend/metal.rs:109-125` overstates
  Metal's f32-atomic-add capability; the codegen emits WGSL
  `atomicAdd<f32>` which naga's MSL backend drops because standard
  WGSL doesn't define `atomicAdd` for floats), drafts the upstream
  patch (capability honest + CAS-loop lowering with u32-bitcast over
  `atomicCompareExchangeWeak`), and audits every `-gpu` crate for
  `Atomic<f32>` use. Audit result: 3 default-broken on Metal pre-fix
  (`butteraugli-gpu`, `dssim-gpu`, `cvvdp-gpu`); 3 clear (`ssim2-gpu`,
  `zensim-gpu`, `iwssim-gpu`). Workaround commits in this Phase 8e
  flip `fast-reduction` to default-off for butteraugli-gpu and
  dssim-gpu and document cvvdp-gpu's Metal status at the module-doc
  level (Metal users use `compute_dkl_jod_host_pool` until the
  upstream fix lands).

- **`zenmetrics-api` Phase 8e.2 + 8e.3 — persistent PTX cache patch + cache-key design.**
  `crates/zenmetrics-api/docs/CUBECL_PERSISTENT_PTX_CACHE_PATCH.md`
  captures the investigation finding (cubecl-cuda **already** has a
  persistent PTX cache at `<root>/cuda/<ver>/ptx.json.log` via
  `CompilationCache<StableHash, PtxCacheEntry>`; the "cold start"
  symptom is the cache key being too narrow, not the cache being
  absent) and stages a ready-to-apply patch against the
  `lilith/cubecl` fork. The patch extends the cache file path to
  `<root>/cuda/<ver>/<cubecl_sha>/<compute_cap>/<cuda_runtime>/ptx.json.log`,
  picking up: cubecl fork HEAD SHA from a new `build.rs` (so codegen-
  only fork-rev advances invalidate), `sm_<arch>` for multi-GPU
  correctness (per-architecture PTX is mandatory; sharing across caps
  is a correctness bug), and CUDA driver version. Includes the full
  diff (~73 lines additive across `cubecl-cuda/build.rs` +
  `cubecl-cuda/src/compute/context.rs`), the cache-key justification
  table, migration notes, and a three-step verification methodology.
  Patch is NOT applied to lilith/cubecl per CLAUDE.md — execution is
  the `feat/persistent-cache` branch follow-on described in
  `CUBECL_FORK_STRATEGY.md`.

- **`zenmetrics-api` Phase 8e.1 — `imazen/cubecl` fork strategy doc.**
  `crates/zenmetrics-api/docs/CUBECL_FORK_STRATEGY.md` captures the
  maintained-fork plan: move `lilith/cubecl` → `imazen/cubecl` (org-
  owned, surviving the lilith → imazen GitHub identity transition),
  document `imazen-main` trunk + per-patch feature branches
  (`feat/pinned-upload`, `feat/persistent-cache`,
  `feat/metal-atomic-fix`), versioning scheme `vUPSTREAM+imazen.N`,
  rebase + upstream-PR submission protocol, CI matrix on the fork
  (CUDA / wgpu / cpu / hip), and a 6-step user-driven migration plan.
  Execution is user-driven follow-on; this doc is the architectural
  decision deliverable.

- **`zenmetrics-orchestrator` Phase 8a — graceful CPU fallback when no
  GPU is present.** `detect_gpu()` honours
  `ZENMETRICS_FORCE_NO_GPU=1` to short-circuit the nvidia-smi path,
  returning `GpuCapability { present: false, model: "(forced absent)",
  .. }` as the test/CI fixture for hosts that DO have a GPU but want
  to exercise the no-GPU path. The bench runner skips every GPU cell
  when `capability.gpu.present == false` (only CPU cells run, gated
  by the `cpu-<metric>` features). The chooser introduces
  `RejectReason::NoGpuPresent` and rejects every `Backend::Gpu*` with
  it on the no-GPU fast-path so operators see a clearer reason than
  `NoMeasuredData` in the `considered` list. The executor catches a
  runtime libcuda-dlopen failure
  (`is_no_cuda_driver` heuristic: `libcuda.so`, `cuInit`,
  `CUDA_ERROR_NOT_INITIALIZED`, `CUDA_ERROR_OPERATING_SYSTEM`,
  `nvml`, `DriverError` substrings) on the first GPU attempt,
  downgrades `capability.gpu.present` to `false`, and persists the
  cache so the same task and every subsequent task lands on CPU. A
  CPU-only build (`--features cpu-all`) routes the full ladder to
  the per-metric CPU adapter without the executor seeing any GPU
  construction at all. See `crates/zenmetrics-api/docs/PHASE8_PLAN.md`
  Phase 8a section for the verified scenarios.

### Changed

- **`butteraugli-gpu` + `dssim-gpu` Phase 8e.4 — `fast-reduction`
  feature flipped to default-OFF.** Mirrors the ssim2-gpu task #52
  fix (2026-05-26). Default consumers now use the portable per-
  thread-partials + finalize reduction which is deterministic and
  works on every cubecl backend including Metal. Opt back into
  `fast-reduction` for CUDA-only deployments where the ~2-3×
  reduction-step speedup matters more than reproducibility. Existing
  parity-lock and auto_fallback tests cover the slow path; no test
  changes needed.

- **`cvvdp-gpu` Phase 8e.4 — Metal status documented at the lib.rs
  module-doc level.** The production `compute_dkl_jod` pool path
  (`pool_band_3ch_lds_kernel`) commits per-workgroup sums via
  `Atomic<f32>::fetch_add`, which silently no-ops on Metal. Until
  the upstream cubecl-wgpu CAS-loop lowering lands (tracked in
  `CUBECL_METAL_ATOMIC_FIX.md`'s `feat/metal-atomic-fix` branch),
  Metal users MUST use `compute_dkl_jod_host_pool` (the host-pool
  fallback originally shipped for cubecl-cpu) which reads D bands
  back to host before pooling and is unaffected by the atomic bug.
  No API change; lib.rs module-doc updated with explicit Metal
  guidance. cvvdp-gpu is NOT added to the metal-tests CI job until
  the upstream fix lands.

- **Phase 8c — `cvvdp-cpu` crate renamed to `cvvdp`.** Mechanical
  rename matching the conventional Rust pattern of "main crate name =
  canonical CPU implementation" (parallels `dssim-core` → CPU base,
  `butteraugli` → CPU base, etc.). Workspace member moved
  `crates/cvvdp-cpu` → `crates/cvvdp`; package name `cvvdp-cpu` →
  `cvvdp`. All in-tree consumers updated (`zenmetrics-orchestrator`,
  `cvvdp-conformance`, the `cvvdp-gpu` doc-comments cross-referencing
  the CPU port). Not yet published to crates.io. The orchestrator's
  `cpu-cvvdp` feature flag name is unchanged (semantic — describes
  what it enables, not the dep crate name); the public stable column
  name `cvvdp_cpu_imazen_v*` and the env override
  `CVVDP_CPU_IMPL_TAG` are also unchanged so downstream sweeps /
  parquet columns keep aligning. Dep-direction flip (cvvdp-gpu →
  depends on cvvdp instead of the inverse) deferred to a follow-up
  because the shared CSF / DKL / masking constants currently live in
  `cvvdp-gpu` — extracting them into a shared base would expand scope
  beyond a rename. Tracked in `crates/zenmetrics-api/docs/PHASE8_PLAN.md`.

- **`zen-metrics-cli` Phase 7.7.1 — the orchestrator is now the default.**
  All scoring subcommands (`score` / `batch` / `compare` / `sweep`)
  route through `zenmetrics-orchestrator` by default. Opt OUT via
  `--use-legacy-scheduler` (or `ZENMETRICS_USE_LEGACY_SCHEDULER=1`)
  to fall back to direct-dispatch handlers. `--use-orchestrator`
  (and `ZENMETRICS_USE_ORCHESTRATOR=1`) deprecated to a no-op +
  warning. Production sweeps gain OOM-safe fallback (Full → Strip →
  CPU), persistent capability cache, and cached-reference
  auto-detect for free. `scripts/sweep/onstart_orchestrator.sh`
  drops the `ZENMETRICS_USE_ORCHESTRATOR=1` export — no longer
  required. Parquet sidecar shape is bit-identical to the legacy
  path on all metrics except butter (see below).

  Butteraugli (BOTH CPU and GPU CLI variants) stays on the legacy
  path until the per-crate `ButteraugliOpaque::new_with_memory_mode`
  rewires its strip arm to `Butteraugli::new_multires_strip` (the
  multi-resolution strip walker that exists in `pipeline.rs` but the
  opaque doesn't yet expose). Without that wire-up, the
  orchestrator's strip-preferred Auto resolver drops butter to
  single-resolution and diverges from the legacy CLI's always-
  multires `butter_pnorm3::score_both` path by ~14-30 %. The
  orchestrator transparently falls back to the legacy code for
  butter; sweep output shape is unchanged.

  Refs `benchmarks/orchestrator_parity_2026-05-27_phase771_run3.csv`
  (54 of 54 cells PASS-EXACT post-fix) and `INTEGRATION_NOTES.md`
  Phase 7.7.1 section.

### Fixed

- **`zenmetrics-orchestrator` Phase 8i — sentinel errors skip OOM
  cache recording (Fix C).** The executor's `run_single` ladder had
  been recording `(Backend::Cpu, pixels)` as an OOM cell for three
  non-memory sentinels emitted by the construct path:
  `CpuMetricUnavailable:` (metric has no CPU reference upstream),
  `CpuBackendUnavailable:` (build is missing the `cpu-<metric>`
  feature), and the pre-Phase-6 legacy `CpuNotYetWired`. All three
  are feature-flag / build-configuration sentinels, not memory
  failures. Recording an OOM for these cases permanently locked out
  CPU at the affected size for any future binary that DOES have the
  feature enabled, and pre-Phase-3 cache files inherited the
  pollution forever. Fix: remove the `record_oom_and_persist` calls
  from the three sentinel branches in `executor.rs:925-965` and
  replace with `log::debug!` skip-records. Chooser-side rejection
  (`RejectReason::CpuMetricUnavailable` from `chooser.rs:649-678`)
  continues to handle the prevention path correctly. New regression
  test `sentinel_errors_do_not_pollute_cells_failed_oom` in
  `tests/cpu_backend.rs` asserts no `(Backend::Cpu, _)` entry is
  added to `cells_failed_oom` after a sentinel-triggering run.

- **`zenmetrics-orchestrator` Phase 8i — record_oom_and_persist prunes
  stale + contradictory cache entries (Fix B).** The investigation in
  `crates/zenmetrics-api/docs/CVVDP_CHOOSER_REGRESSION_INVESTIGATION.md`
  identified `cells_failed_oom` as a write-only "punishment list" that
  survived every binary upgrade and re-bench until the file was
  manually deleted. Two concrete classes of stale entries were
  observed in the wild: (i) `(Backend, _)` entries whose backend is no
  longer in `chooser::supported_backends(metric)` — e.g. fossilized
  `(gpu_strip, *)` entries for cvvdp written by a pre-orchestrator
  binary that did expose GpuStrip for cvvdp; (ii) entries contradicted
  by a co-existing positive measurement at the same `(backend, size)`
  cell — runtime OOM under transient memory pressure recorded an OOM
  for a cell the bench had successfully measured. Fix:
  `record_oom_and_persist` runs a `retain()` cleanup pass on each
  call. Both prune events are logged at `debug` level (routine cleanup,
  not pathology). Existing cache files self-heal on the first
  legitimate OOM recording after this lands — no migration script
  required. Two new unit tests in `executor::tests`:
  `record_oom_prunes_fossilized_unsupported_backend` and
  `record_oom_prunes_entry_contradicted_by_positive_measurement`.

- **`zenmetrics-orchestrator` Phase 8i — known_oom_cell cascade
  defeated by positive measurement (Fix A).** The investigation in
  `crates/zenmetrics-api/docs/CVVDP_CHOOSER_REGRESSION_INVESTIGATION.md`
  documented that a single fossilized OOM entry at 256² in the
  persistent capability cache was cascading via the `*px < pixels`
  rule to reject every cvvdp request at any size >= 256² for the
  cache file's lifetime, even when the cache also held positive
  bench measurements at 1024² + 4096² for the same backend. Fix:
  `known_oom_cell` now consults `ns_per_px_at` before falling
  through to the cascade rule — if a positive measurement exists at
  any `size >= oomed_pixels` for that backend, the OOM is treated
  as stale and ignored (the successful later measurement contradicts
  the cascade hypothesis). Exact-match and snapped-size matches
  remain unconditional. Two regression tests added in
  `tests/chooser.rs`:
  `oom_cascade_defeated_by_positive_measurement_at_or_above_oom_size`
  (cascade defeated) and
  `oom_cascade_still_rejects_when_no_positive_measurement_at_or_above`
  (cascade still fires without a contradicting measurement).

- **`zenmetrics-orchestrator` Phase 7.7.1 — three structural bugs
  cleared the parity gate so the CLI default could flip.** (1)
  `executor::construct` was forcing `MemoryMode::Full` or
  `MemoryMode::Strip { h_body: None }` based on which microbench
  result happened to land first, baking a non-deterministic input
  into ssim2 + butter scores at sizes where Auto would have picked
  differently. Fixed by passing `MemoryMode::Auto` on the first
  construct attempt so the per-crate resolver owns the policy;
  explicit `Full` / `Strip` only on OOM-ladder retry (where the
  chooser already excluded the bigger mode via `cells_failed_oom`).
  (2) `evaluate_candidate` was selecting CPU candidates with NEGATIVE
  `predicted_ns_per_px` — when log-linear extrapolation from 2
  monotonically decreasing bench points overshoots zero at large
  sizes (concrete: CPU ssim2 extrapolated to -179 ns/px at 16M
  pixels), `min(ns_per_px)` then ranked the negative number as
  fastest. Fixed by rejecting non-positive predictions via the new
  `RejectReason::NonPositivePrediction`; the chooser falls back to a
  backend with a real measurement. (3) `rekey_orchestrator_columns`
  did not re-key the orchestrator's versioned `iwssim_imazen_v*`
  column back to the legacy unversioned `iwssim_gpu` that production
  parquet readers depend on; fixed by adding the rename rule.



- `zen-cloud-vastai` sweep worker no longer silently falls back to the
  deprecated bash subprocess when the inline Rust pipeline fails. With
  `inline-sweep` (the default + production build) a failed `omni` chunk
  now fails honestly and surfaces the real error, matching the
  `feature-backfill` / `source-features` arms. The old fall-through
  re-ran the failed chunk through `omni_backfill_chunk_worker.sh` (the
  W44 path that discards encoded bytes, keeping only `len() as u32`)
  AFTER `process_chunk_inline` had already uploaded a durable Failed→R2
  marker — producing contradictory state (an error marker plus a
  bash-success sidecar from a divergent code path; per the
  "two code paths, different output → bug" rule). The bash subprocess is
  now extracted into `run_chunk_via_bash`, compiled only when
  `inline-sweep` is OFF, where it remains the sole execute path
  (unchanged) (4581f017).

- CI was failing on every platform with `failed to read coefficient/Cargo.toml`
  because the `coefficient` dev-dep on `ssim2-gpu` (used only by an
  `#[ignore]`d cross-backend parity test) had a relative path that
  doesn't exist on CI runners. The test file + dev-dep entry were
  deleted; the cross-backend parity methodology lives in git history
  (`a4fe9a5e`) and can be restored locally via `git revert` of the fix
  commit if needed. Unblocks `zenmetrics-orchestrator` Phase 7+ CI
  gates (`07d749d6`).

- Sweep worker + `zen-metrics-cli` could not read Snappy-compressed input
  parquets — the `parquet` dep was built with only `["arrow", "zstd"]`, so
  reading any parquet written with the default (Snappy) compression failed
  with `Parquet error: Disabled feature at compile time: snap`. Added the
  `snap` feature to all three `parquet` declarations. Caught on a Salad
  smoke run via the new durable error sidecar — the failure was at the
  input-parquet read, before any GPU work, not a node/CUDA issue.

### Added

- `zenmetrics-orchestrator` Phase 7.7 — **parity sweep harness +
  honest-stop on default-flip**. User directive 2026-05-27: "make
  users of the cli adopt this, for local and remote use" — i.e.,
  flip `--use-orchestrator` from opt-in default-off to default-on.
  Phase 7.7 brief gated the flip on a comprehensive parity sweep
  proving orchestrator == legacy bit-identical (or within atomic
  reorder noise) across all 6 metrics × 3 sizes × 3 qs = 54 cells.
  **Result: 22 of 54 cells diverged — default flip BLOCKED**. The
  divergences are not atomic-reorder noise; they're three distinct
  structural issues now documented in `INTEGRATION_NOTES.md` Phase
  7.7 section: (1) butter memory-mode-selection bench-state
  dependency (`Backend::GpuFull`/`GpuStrip` → forced
  `MemoryMode::Full`/`Strip`, contradicting the per-crate
  strip-preferred Auto resolver; ~14% score swings depending on
  whether the bench landed gpu_full or gpu_strip as faster), (2)
  ssim2 same root cause at 4096 px when chooser picks Strip due to
  bench-OOM at full, (3) iwssim column-name divergence (legacy
  `iwssim_gpu` vs orchestrator `iwssim_imazen_v0_0_1`, values are
  bit-identical). Path-forward documented per failure. Shipped:
  `scripts/orchestrator_parity_sweep.py` (repeatable harness),
  `benchmarks/orchestrator_parity_2026-05-27.{csv,md}` (data).

- `zenmetrics-orchestrator` Phase 7.6 — **internal task reordering
  for warm-instance reuse + cached-ref hit rate**. User directive
  2026-05-27: "orchestrator should reorder tasks". Four layers
  build on Phase 7.5's single-warm-instance Layer 1:

  - **Layer 1 (Task.ref_hash field)**: new `u64 ref_hash` field on
    `Task`, default `0`. The orchestrator populates it with
    `xxhash3_64(ref_bytes)` (or the pre-upload's stable `inner_id`)
    before sorting. Required field — every in-tree caller (tests,
    examples, README, MIGRATION) updated to set `ref_hash: 0`.

  - **Layer 2 (run_all internal sort)**: `Orchestrator::run_all`
    collects every task, populates ref_hash, sorts internally by
    `(metric.tag(), width, height, ref_hash, task_id)`, then
    submits in sorted order. Yield order remains completion order
    (callers correlate via task_id). On a real-host 60-task mixed
    chunk (3 metrics × 2 sizes × 10 dist each) with single GPU + 1
    CPU worker, the sort reduces warm-instance constructions from
    40 (FIFO) to 6 (sorted) — 85% reduction.

  - **Layer 3 (streaming reorder window)**:
    `OrchestratorConfig.stream_reorder_window: (Duration, usize)`
    defaults to `(50ms, 16)`. `submit()` now buffers tasks into a
    pending queue; the window flushes when either limit trips. The
    cached-ref auto-detect runs after sort so consecutive tasks
    with identical refs hit each other. New public method
    `flush_pending()` lets callers using
    `(Duration::MAX, usize::MAX)` dispatch explicitly.
    `(Duration::ZERO, 1)` disables — strict FIFO. `poll()` /
    `poll_any_blocking()` auto-drain stale windows so a slow
    caller doesn't park tasks indefinitely.

  - **Layer 4 (observable VRAM budget at swap)**: GPU worker logs
    every signature-change swap. WARN level when live free-VRAM
    at swap time has dropped below the chooser's prediction
    (external pressure) OR when the swap surfaces
    `OomAtConstruction`. DEBUG level otherwise.
    `TaskResult.vram_peak_mib` now carries the chooser's
    prediction through the pool path (previously `None`).

  Public surface additions (non-breaking except Task.ref_hash):
  `Orchestrator::flush_pending`,
  `Orchestrator::pending_queue_len` (test surface),
  `Orchestrator::in_flight_len` (test surface),
  `warm_instance_construction_count` (test surface),
  `reset_warm_instance_construction_count` (test surface).

  Tests: 6 new pure-logic tests in `tests/reorder.rs` for the
  streaming-window and run_all sort behaviours
  (`strict_fifo_when_window_disabled`,
  `streaming_window_buffers_then_dispatches`,
  `streaming_window_count_limit_flushes`,
  `explicit_flush_pending_drains_immediately`,
  `streaming_window_duration_triggers_flush_via_poll`,
  `run_all_sort_groups_by_metric_dims_ref`). 3 `#[ignore]` GPU
  integration tests verify churn reduction
  (`warm_instance_churn_minimal_on_mixed_chunk` — measured 6 vs
  40 on the workstation), cached-ref hits
  (`cached_ref_hit_rate_high_on_repeat_ref` — measured 1 miss /
  49 hits on 50 same-ref tasks), and peak VRAM bound
  (`peak_vram_equals_max_single_metric_footprint`).

  Spec: `crates/zenmetrics-orchestrator/docs/REORDERING_DESIGN.md`
  (status: RESOLVED). Migration note added to
  `crates/zenmetrics-orchestrator/docs/MIGRATION_FROM_API.md`.

- `zenmetrics-orchestrator` Phase 7.5 — **the orchestrator is now
  production-ready for all metrics, with the per-cell sweep loop
  flipped to the orchestrator when `--use-orchestrator` is set.**
  Closes the three Phase 7 honest-stops: butter + cvvdp eligibility,
  cmd_sweep's MetricCache loop, and the CI infra blocker. Scope:

  - **`TaskResult.output_columns`** (`zenmetrics-orchestrator`):
    new `BTreeMap<String, f64>` field on `TaskResult` carrying the
    per-metric output columns the caller should write. Multi-column
    metrics survive end-to-end: butter exposes both
    `butteraugli_max_gpu` and `butteraugli_pnorm3_gpu`; cvvdp +
    iwssim use their versioned column tags
    (`CVVDP_COLUMN_NAME` / `IWSSIM_COLUMN_NAME`). Bit-identical to
    the legacy `MetricCache` output for the same CLI input
    (`957afc5a`).

  - **`butteraugli_gpu::ButteraugliOpaque::compute_srgb_u8_with_pnorm3`**:
    new public method exposing both the max-norm Score and the
    libjxl `pnorm_3` aggregate from the same fused reduction
    kernel. No extra GPU work; the opaque path used to drop
    `pnorm_3` after producing it. Additive — the existing
    `compute_srgb_u8` keeps working unchanged (`957afc5a`).

  - **`cmd_sweep` orchestrator-driven loop** (`zen-metrics-cli`):
    when `--use-orchestrator` is set, the per-cell metric scoring
    routes through `Orchestrator::run_single` instead of
    `MetricCache::lock_global`. The two paths are mutually
    exclusive in any sweep invocation — no double-allocation of
    warm cubecl `Metric` instances. The legacy path stays compiled
    in for the `--use-orchestrator=false` default (Phase 7.5
    default; Phase 7.6 will flip it on after fleet trials). One
    carve-out: ZensimGpu + `--feature-output` stays on
    MetricCache because the orchestrator API doesn't yet expose
    `compute_features_srgb_u8` (`f1fda156`).

  - **Eligibility gate**: `metric_orchestrator_eligible` admits
    `ButteraugliGpu` + `Cvvdp`. Only CPU `Butteraugli` remains on
    the legacy path (cpu-butter doesn't expose `pnorm_3` yet).
    `rekey_orchestrator_columns` helper renames the orchestrator's
    canonical GPU-variant column keys to CPU-variant keys for CLI
    callers that asked for a CPU variant.

  Tests added: 6 `build_output_columns_*` shape tests in
  `executor.rs` (every metric × column-name contract), one
  `rekey_orchestrator_columns_phase_7_5_*` test on the CLI side.
  All 54 orchestrator unit tests + 17 CLI lib tests pass.

- `zenmetrics-orchestrator` Phase 7 — **the orchestrator is now the
  recommended entry point** for any caller that scores more than one
  pair at a time (sweeps, batch, picker training, RD curves). The
  scope of Phase 7 is integration, not new orchestrator surface:

  - `zen-metrics-cli` (the `zen-metrics` binary): new `orchestrator`
    feature pulls `zenmetrics-orchestrator` as an optional dep, with
    `orchestrator-cuda` + per-metric `orchestrator-cpu-*` variants
    forwarding the orchestrator's own feature gates. New global CLI
    flags (apply to every subcommand) — `--use-orchestrator`,
    `--orchestrator-cache <PATH>`, `--bench-on-start <auto|yes|no>`,
    `--cpu-features <list>`. Env var `ZENMETRICS_USE_ORCHESTRATOR=1`
    is equivalent to the flag. `cmd_score` routes
    orchestrator-eligible metrics through `Orchestrator::run_single`;
    `cmd_batch` / `cmd_compare` / `cmd_sweep` warm the capability
    cache + print the active machine profile to stderr so subsequent
    workers on the same box start warm.

  - `Dockerfile.sweep.v27` — bakes the new orchestrator binary
    features (`orchestrator,orchestrator-cuda,orchestrator-cpu-all`)
    so the OOM fallback ladder reaches every CPU reference in
    production. Extends v26's layer plan (~30 MB binary delta).
    Sanity gates the new top-level flags are exposed. Default
    ENTRYPOINT unchanged from v26 to keep existing fleet launchers
    working; orchestrator opt-in is `--entrypoint
    /usr/local/bin/onstart_orchestrator.sh`.

  - `scripts/sweep/onstart_orchestrator.sh` — bake-everything-clean
    onstart that hydrates env from `/proc/1/environ`, verifies every
    baked tool (zen-metrics, s5cmd, jq, python3 + pyarrow, libnvrtc),
    optionally fetches a fleet-shared `capability_<hash>.toml` from
    R2, exports `ZENMETRICS_USE_ORCHESTRATOR=1` +
    `ZENMETRICS_CACHE_DIR`, and delegates to onstart_unified.sh for
    the chunk-claim loop.

  - `crates/zenmetrics-orchestrator/README.md` (new) — quickstart,
    batch + streaming APIs, OOM handling, cached-ref semantics, CPU
    backend selection, capability cache lifecycle, full config surface.

  - `crates/zenmetrics-orchestrator/docs/MIGRATION_FROM_API.md` (new)
    — side-by-side code samples for migrating
    `zenmetrics-api` call sites to the orchestrator (single-call
    score, batch sweep, cached-ref pattern) + the behaviour
    differences callers should know (automatic backend selection,
    OOM-doesn't-bubble-by-default, completion-order results).

  - Top-level `README.md` — new "Recommended entry point" section
    surfacing the orchestrator with a decision table and pointers to
    the quickstart + migration guide. The orchestrator crate is now
    listed in the main Crates table.

  Backwards-compat: every existing CLI flag + output format is
  preserved. Without `--use-orchestrator` the entire orchestrator path
  is bypassed; the legacy direct-dispatch code paths (CvvdpBatchScorer,
  MetricCache, run_metric) remain the default. CI continues to verify
  both shapes.

  **Phase 7.5 update (2026-05-27)**: butter + cvvdp are no longer
  carved out of the orchestrator path, and `cmd_sweep`'s per-cell
  scoring now dispatches through `Orchestrator::run_single` when
  `--use-orchestrator` is set (the warm-only behaviour described
  below applies only to Phase 7 builds — Phase 7.5 graduated to
  full dispatch). See the Phase 7.5 entry above for details.

- `zenmetrics-orchestrator` Phase 6 — CPU backend wiring for the OOM
  fallback ladder. Replaces the Phase 4-5 `CpuNotYetWired` short-circuit
  with per-metric reference adapters: cvvdp-cpu (in-tree), ssimulacra2,
  dssim-core, butteraugli, zensim. Each backend ships behind its own
  `cpu-<metric>` feature flag; `cpu-all` is the convenience bundle for
  production sweep workers. Defaults DO NOT include any CPU backend so
  callers that only want capability detection pay no dep cost.

  Iwssim has no clean upstream CPU reference and is honestly skipped —
  the chooser surfaces `RejectReason::CpuMetricUnavailable` for Iwssim
  Cpu candidates so the OOM ladder advances. See
  `crates/zenmetrics-orchestrator/docs/CPU_BACKENDS.md` for the per-metric
  mapping, cached-ref semantics, RAM characteristics, and the Iwssim
  honest-stop rationale.

  New crate module `cpu_adapter` exposes `CpuAdapter` (pub(crate))
  with one arm per metric. Cached-ref dispatch is wired where the
  upstream crate supports it (cvvdp-cpu `warm_reference`, dssim-core
  `DssimImage` cache); ssim2 / butter / zensim cache bytes for API
  shape and recompute on the cached-ref call. The pool's CpuWorker
  now maintains a warm `CpuAdapter` per `(metric, w, h)` signature
  (one per worker thread); the GPU worker's cached-ref auto-detect
  pattern is mirrored, so production sweep workloads with many-dist-
  one-ref see the same speedups.

  Chooser updated: CPU is now a real candidate for every metric
  except Iwssim. `vram_mib = 0` since CPU consumes RAM not VRAM;
  ns/px from bench cache with a 200 ns/px heuristic fallback. The
  chooser's `KnownOomCell` check applies to CPU too so a previous
  CPU-side failure (e.g. allocation refusal) excludes the CPU
  candidate at the same size.

  Executor updated: `ExecMetric::Cpu(Box<CpuAdapter>)` variant added;
  `construct` routes `Backend::Cpu` through `CpuAdapter::new` with
  structured sentinels. New `OrchestratorError` variants:
  `CpuMetricUnavailable`, `CpuBackendUnavailable`, `CpuFailed`.
  `CpuNotYetWired` retained for backwards compatibility but no
  longer produced.

  Bench runner: per-metric × CPU-backend cells at 512² + 1024² (CPU
  grid kept tight to stay within the < 60s warm() budget — 4096²
  CPU butteraugli alone would burn the entire budget). `vram_mib`
  is always 0 for CPU cells; future Phase 7 ResourceBudget work
  will measure RAM during the bench instead.

  New integration test file `tests/cpu_backend.rs` (10 tests):
  per-CPU-backend construct+compute smoke, OOM-forced fallback to
  Cpu, cached-ref round-trip parity, chooser picks Cpu when GPU
  OOMs, Iwssim ladder-advance.

  `RejectReason::CpuMetricUnavailable` added; `print_capability`
  example prints both that and retained `CpuNotYetWired`;
  `run_single` example reports `cpu_backends_enabled: [...]` so
  operators can confirm the fallback ladder is armed.

- `zenmetrics-orchestrator` Phase 5 — worker pool, streaming + batch
  APIs, cached-ref auto-detect, live VRAM watcher. New types:
  `TaskHandle`, `TaskRefHandle`, `PoolConfig`, `CachedRefStats`,
  `RunAllIter`. New `Orchestrator` methods: `submit` /
  `poll` / `poll_any` / `poll_any_blocking` for the streaming API,
  `run_all` for batch, `upload_reference` / `drop_reference` for
  explicit pre-uploads that skip the auto-hash, `cached_ref_stats` for
  test introspection, `vram_watcher_mib` for the live watcher snapshot,
  `set_pool_config` for runtime tuning. `TaskData` gained
  `PreUploaded(TaskRefHandle)` for the explicit-pre-upload path.
  Single GPU worker keeps a warm `Metric` instance across consecutive
  tasks of the same `(metric, w, h, backend)` signature; `num_cpus/2`
  CPU workers (capped) currently surface `CpuNotYetWired` until Phase 6
  wires the per-crate CPU references. xxhash3_64 ref-bytes hash on
  every `submit` drives a sliding-window cache (32 entries) that
  promotes consecutive same-ref tasks to `set_reference_srgb_u8` +
  `compute_with_cached_reference_srgb_u8`. Live VRAM watcher samples
  `cvvdp_gpu::memory_mode::live_vram_probe_bytes` every 250 ms; GPU
  worker stalls briefly when free VRAM drops below
  `vram_safety_floor_mib` (default 200). New examples `run_batch.rs`
  + `run_stream.rs`; new integration tests `tests/streaming.rs` +
  `tests/cached_ref.rs` (real-GPU, gated `#[ignore]`). `Orchestrator`
  dropped `Clone` and `Sync` (still `Send`) because the pool owns
  `mpsc::Receiver` + `JoinHandle`s.

- `zen-cloud-vastai` worker — durable best-effort R2 error sidecar on
  chunk failure. When `process_chunk_inline` returns an error, the worker
  uploads the full anyhow error chain + chunk_id, run_id, hostname, and
  input/source URIs to `s3://<out-bucket>/<run_id>/errors/<chunk_id>.txt`
  via its existing scoped R2 cred. The out-bucket is derived from the
  chunk's `out_sidecar_omni`. Makes fleet failures (vast.ai / Salad /
  RunPod — they all share this compute path) diagnosable without a
  logging provider, since the container can die right after the failure.
  Non-fatal: the original error is always returned regardless of upload
  outcome.

- `zenmetrics-orchestrator` (new crate, Phase 1) — capability detection
  (`nvidia-smi` GPU + `raw-cpuid` CPU + `sysinfo` RAM) and persistent
  TOML cache at `~/.cache/zenmetrics/capability_<short_hash>.toml`.
  Public surface: `Orchestrator`, `OrchestratorConfig`,
  `CapabilityProfile`, `GpuCapability`, `CpuCapability`,
  `MetricProfile` (Phase 2 placeholder), `OrchestratorError`,
  `compute_machine_hash` + `cache_file_path` + `load_cached_profile` +
  `save_profile` + `is_profile_stale` helpers, plus an example program
  `print_capability`. Hash-based filename so multi-machine homedirs
  don't collide. Stale-detection on time OR driver/model change.
  No scheduling, no benchmark runner, no worker pool — those come
  in later phases. See `crates/zenmetrics-api/docs/ORCHESTRATOR_DESIGN.md`.

- `zenmetrics-orchestrator` Phase 2 — quick-bench harness behind a new
  `bench` cargo feature. `Orchestrator::bench()` (unconditional re-run)
  and `Orchestrator::warm()` (cache-aware) populate
  `capability.metrics` with `(metric, backend, size_pixels)`
  measurements: wall-time p50 in ns/px + peak VRAM in MiB, sampled
  via a background nvidia-smi thread during the compute loop.
  New types: `Backend` (`GpuFull`/`GpuStrip`/`GpuStripPair`/`Cpu`),
  `BackendBench`, `BackendVram`, plus the populated `MetricProfile`
  with `BTreeMap<u64, _>` keyed on `width × height`. Embedded
  `synth_pair_offset_dist(w, h)` matches the per-crate test helper
  bit-identically so no external corpus is needed. TOML round-trip
  via `u64_keyed_map_{bench,vram}` serde helpers. `BenchPlan` knobs:
  sizes (default `[1024, 2048, 4096]`), warmup/timed iter counts,
  soft per-cell timeout (default 5s), VRAM sample interval (10 ms).
  Cells iterate sizes descending so the cubecl pool grows to its
  max early. Cells that surface OOM at construction or runtime are
  recorded in `MetricProfile::cells_failed_oom` for Phase 3's chooser.
  Total bench runtime on RTX 5070 + 7950X: ≈ 56-60 s cold cache.
  In-process measurement keeps cubecl pool reuse visible — the
  orchestrator schedules in-process so the cumulative-pool numbers
  are more representative than subprocess-isolated audit values.

- `zenmetrics-orchestrator` Phase 3 — pure backend chooser
  (`Orchestrator::choose_backend(metric, w, h, vram_free_mib)`).
  Predicts every candidate backend's `ns_per_px` + `vram_mib` via
  log-pixel interpolation of the Phase 2 cache, rejects backends
  outside the safety-margin VRAM budget or in `cells_failed_oom`,
  and returns the fastest survivor + a diagnostic
  `Vec<ConsideredCandidate>` so operators can audit "why did it
  pick X?" with one call. Extrapolation above the measured size
  range uses a configurable pessimism multiplier (default 1.20)
  to avoid optimistic over-commits. `Backend::Cpu` is always
  rejected as `CpuNotYetWired` until Phase 6. New public types:
  `ChooserConfig` (safety margin / pessimism / tie-break order),
  `BackendChoice`, `ConsideredCandidate`, `CandidateStatus`,
  `RejectReason`, `ChooserError`, `TaskShape`. Convenience helper
  `choose_backend_for_task(&TaskShape)` threads a live nvidia-smi
  VRAM probe (via `cvvdp-gpu::memory_mode::live_vram_probe_bytes`)
  through; falls back to `capability.gpu.total_vram_mib` on
  CI/no-GPU hosts. `Orchestrator::from_capability(cfg, profile)`
  new constructor for tests + fleet callers that bypass detection.
  17 integration tests cover interpolation paths, OOM history
  fallback, VRAM-constrained rejection, unknown-metric errors,
  tie-break overrides, and a <100µs/call perf gate. The chooser
  itself is feature-gated on `bench` because it references
  `MetricKind` from `zenmetrics-api` (4228a168).

- `zenmetrics-orchestrator` Phase 4 — single-task executor with OOM
  recovery (`Orchestrator::run_single(Task) -> TaskResult`). Asks
  the Phase 3 chooser for a primary backend, constructs the metric
  via the umbrella `zenmetrics-api::Metric::new_with_memory_mode`,
  runs `compute_srgb_u8`, and recovers from OOM at either
  construction or runtime by walking the fallback ladder
  `GpuFull → GpuStrip → GpuStripPair → Cpu → FullyExhausted`. Each
  OOM observation appends `(backend, size_pixels)` to
  `cells_failed_oom` AND persists the cache to disk immediately so
  the learning survives a process crash mid-task. Non-OOM errors
  surface as `MetricApi(...)` without retry. New public types:
  `Task`, `TaskData`, `TaskResult`, `AttemptOutcome`,
  `executor::OrchestratorError` (re-exported as `ExecutorError`).
  Behind the new `cuda` feature (implies `bench`); Phase 5 widens
  to wgpu/hip. Synthetic-profile integration tests in
  `tests/executor.rs` cover the ladder shape, cache persistence,
  path-task-data rejection, and force-exhaust paths without
  touching GPU hardware; the two GPU-only tests are `#[ignore]`d.
  CLI example `examples/run_single.rs` drives the executor end-
  to-end with optional `ZM_FORCE_OOM_FULL=1` to demonstrate the
  fallback. Tests + example land in source for execution from the
  primary `zenmetrics/` checkout — jj sibling-workspace cargo
  builds fail with a pre-existing cross-workspace path collision
  for `butteraugli-gpu` (same constraint blocks Phase 3's
  `chooser` tests). Two commits: feat 86c84ee2 (executor module)
  + test 4da664c3 (integration tests + example).

- `zenmetrics-orchestrator::detect_wsl2_host_ram_mib_hint` — detects
  WSL2 via `/proc/version` and surfaces a hint so callers can
  interpret `ram_mib` correctly (WSL2 caps RAM at `.wslconfig:memory=`,
  which on the 7950X workstation is 50 GB / 50185 MiB out of 128 GB
  physical). Phase 1's `ram_mib = 50185` value is the actual Linux-
  kernel-visible total and remains the correct scheduling input —
  surfacing 128 GB would lie about what the orchestrator can
  allocate.

- `zen-cloud-core::r2creds` — shared, provider-agnostic Cloudflare R2
  scoped-credential minter. `mint_scoped_r2_cred(...)` hits the verified
  account-level `temp-access-credentials` endpoint; `Permission` enum
  (`ObjectReadWrite`/`ObjectReadOnly`/`AdminReadWrite`/`AdminReadOnly`)
  serializes to the exact CF wire strings; `ScopedR2Cred` carries the
  session token consumers MUST inject as `AWS_SESSION_TOKEN`. Reused by
  the salad/runpod/vastai launchers (3c233dc).
- `zen-cloud-salad` launcher — per-sweep scoped R2 cred mint+inject:
  `R2ParentCreds::from_env`, `ScopedCredSpec` (bucket + prefixes + TTL,
  6h default, object-read-write), `SaladApi::mint_sweep_r2_cred`, and
  `SaladApi::create_container_group_with_scoped_cred` (opt-in minting;
  injects `R2_ACCESS_KEY_ID`/`R2_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN`
  into the container-group env). `SaladEnvCredentials` now also resolves
  `AWS_SESSION_TOKEN`/`R2_SESSION_TOKEN`. Scoped creds limit a compromised
  consumer-GPU node's blast radius to one bucket (d861b9b).

### Fixed

- `scripts/sweep/entrypoint_salad.sh` — write `aws_session_token` into
  `~/.aws/credentials` when `AWS_SESSION_TOKEN` (or `R2_SESSION_TOKEN`) is
  present, so minted scoped/temporary R2 creds work (a temp key+secret
  without the session token 403s). Absent => unchanged back-compat for
  permanent-token / root-key use (0ff85b4). NOTE: requires a local image
  rebuild before temp creds work end-to-end on a Salad node.

### ssim2-gpu — refactor: orientation-aware error_maps + IIR untransposed building blocks — 2026-05-27

Adds `Ssim2::blur_plane_two_pass_iir_untransposed` (uses the new
`blur_h_pass_kernel`, output in untransposed orientation) and a
`blur_plane_two_pass_untransposed` dispatcher that routes IIR through
the v+h fast path and falls back to v+t+v for FIR. Adds
`run_error_maps_masked_oriented(scale, mode, untransposed_raw)` that
picks the right `ref_xyb` / `ref_xyb_t` reads based on the orientation
flag.

These land as **`#[allow(dead_code)]` building blocks** — the
production paths (`process_scale`, `compute_with_reference_with_mode`,
`process_scale_strip`, `process_scale_strip_cached_ref`) all still use
the legacy v+t+v transposed-orientation path. The wire-in is blocked
on the Ssim2Batch migration (porting `error_maps_broadcast_batched_kernel`
to untransposed orientation). See `docs/SSIM2_FIX_ASSESSMENT.md`
"Refined commit plan (REVISED 2)" for the full unblock chain.

Tests: all 21 parity_lock + 5 ssim2_skipmap_audit + 2
reduction_determinism + 14 aliasing_invariants (excluding the
pre-existing 4096² OOM) pass. Scores bit-identical to pre-fix at
1024² / 2048² / 4096².

### ssim2-gpu — perf: add H-pass IIR blur kernel (row-walking) — 2026-05-27

Adds `blur_h_pass_kernel` (row-walking sibling of the column-walking
`blur_pass_kernel`). Each thread owns one row, walks `x` from `-N+1`
to `width-1` with the same 6-floats-of-IIR-state recurrence as the
v-pass. Same Charalampidis coefficients, same zero-padded boundary
handling, same `prev_1 + prev_3 + prev_5` summation order.

This is the building block for collapsing the 2-pass blur from
`v_pass + transpose + v_pass` (3 launches) to `v_pass + h_pass`
(2 launches). The transpose is no longer needed because the IIR is
separable: `h_pass(v_pass(src))` = `v_pass(transpose(v_pass(src)))`
= `transpose(v_pass(v_pass(transpose(src))))` modulo storage order.

Parity gate: new `examples/blur_h_pass_parity` confirms BIT-IDENTICAL
output (max diff = 0) between `v+t+v` and `v+h` at 7 test cases
including 32², 64×32, 256², 1024×768, 2048².

NOT yet wired into the pipeline — `blur_plane_two_pass_iir` still
uses the 3-step `v+t+v`. The wire-in is the next commit. Shipped
as a kernel-only commit so the parity gate is independently
reviewable and the pipeline change is a clean follow-on.

### ssim2-gpu — perf: port 3-channel-fused downscale kernel from zensim-gpu — 2026-05-27

Adds `downscale_2x_3ch_kernel` (3-in, 3-out planar 2× downscale)
mirrored verbatim from `zensim-gpu::kernels::downscale::downscale_2x_3ch_kernel`.
Replaces the per-channel `downscale_2x_plane_kernel` triplet loop in
`build_linear_pyramid_until` (and `set_reference_strip_mode`'s
full-image ref pyramid) with one launch per scale transition.

Output is bit-identical to the per-plane variant — same `min(src-1)`
clamp math and same `* 0.25` box-average — verified by parity tests
(21/21 `parity_lock.rs` tests pass; scores identical to pre-fix at
1024², 2048², 4096²). Phase 0 / easy-fix from
`docs/SSIM2_FIX_ASSESSMENT.md`.

### zen-cloud-runpod — feat: RunPod Pods (pull) provider + `--backend runpod` (Phase F, 2026-05-27)

(`82178f44`) New `zen-cloud-runpod` crate implementing the five
`zen-cloud-core` traits for RunPod's **Pods (pull)** path (spec §1.10).
RunPod pods are structurally identical to vast.ai — a rented GPU pod
boots a generic container, credentials + sweep wiring arrive as pod env
vars, and the worker PULLs chunks from R2 with the shared atomic
token-race claim. So the `JobQueue` (`RunpodChunkQueue`) reuses vast.ai's
proven claim algorithm verbatim (`zen_cloud_vastai::worker::claim::try_claim`
+ `R2Client`) rather than copy-pasting a third claim impl; `BlobStorage`
re-exports the shared `zen-cloud-s3` impl (no second S3 client);
`CredentialSource`/`WorkerHost` read the plain pod env + `RUNPOD_POD_ID`
(no IMDS, no pid-1 trick); `Heartbeat` defaults to a no-op (RunPod tracks
pod liveness natively) with an `R2Heartbeat` for cross-fleet monitoring
parity. The `launch` module hand-rolls the **current RunPod v1 REST API**
(`https://rest.runpod.io/v1`, `Authorization: Bearer`, verified against
the live OpenAPI 2026-05-27: `POST /pods`, `GET`/`DELETE /pods/{podId}`,
`POST /pods/{podId}/stop`) — RunPod has migrated GraphQL→REST and REST is
the go-forward path. Wires `--backend runpod` into `zen-sweep-worker`
(cargo features `runpod` glue / `runpod-sweep` full, mirroring
`salad`/`vastai`); the encode+score `compute` closure is the
backend-agnostic vast.ai one. 26 tests (23 unit + 3 off-node JobQueue
roundtrip via the real `skip_claims` path). Serverless (push) is a
documented follow-on in `RUNPOD.md` (handler-shim design), not
implemented. Real RunPod smoke run (rent a pod, run a chunk) is the
operator's gate. `#![forbid(unsafe_code)]`.

### infra — feat: SaladCloud deploy image + CI publish to ghcr (2026-05-27)

(`410cf6ae`) Adds `Dockerfile.sweep.salad.v1` (the SaladCloud sibling of the vast.ai
`Dockerfile.sweep.v26`) + `scripts/sweep/entrypoint_salad.sh` + the
`.github/workflows/sweep-image-salad.yml` publish pipeline. The image
reuses v26's L1-L8 base (ubuntu → apt → pyarrow → CUDA 12-6 runtime+dev →
s5cmd+jq → cuda_dlsym_stub → zen-metrics) verbatim, then bakes
`zen-sweep-worker` built `--features salad-sweep` (L9) and the pinned
`salad-http-job-queue-worker` v0.7.0 x86_64 Go sidecar (L9b). The
entrypoint launches the sidecar + `zen-sweep-worker worker --backend
salad` concurrently (upstream `with-shell-script` pattern): the sidecar
forwards each queue job as `POST http://localhost:$SALAD_JOB_PORT<path>`
to the worker's local HTTP receiver. BAKE-EVERYTHING honoured — sidecar +
binaries baked at build time, entrypoint fail-loud-exits if any baked
tool is missing, nothing apt/pip/curl-installed at boot. CI publishes
`ghcr.io/imazen/zen-metrics-sweep-salad:v1` (+ `:v1-<sha>`) with a
registry buildcache. Real 1-replica Salad smoke sweep (needs a Salad
node + IMDS + BYO R2 creds) remains the operator's gate.

### cvvdp-gpu — perf: P2.7 partial — gauss_alt deep levels zero-sized + honest-stop on full shrink (2026-05-27)

Zero-sizes `gauss_alt[k > k_split]` planes (deep levels). The shallow
REF strip helper reads `gauss_alt[k+1]` at `k < k_split`, so
`gauss_alt` levels `0..=k_split` stay full-image — the rest is unread
post-swap and gets zero-size handles. `_maybe_swap_gauss_alt_post_ref`
becomes a per-level shallow swap: levels `[0, k_split]` swap;
levels `> k_split` stay unswapped (gauss_ref keeps REF data there
until DIST gauss reduce overwrites it).

**Memory delta (4096² h_body=256, k_split=6):** deep gauss_alt levels
7-8 sum 64²+32²+16² × 3 ch × 4 = ~64 KiB. Negligible at 4096² but
meaningful at smaller sizes / shallower k_split.

**Honest-stop on full P2.7.** The brief targets `gauss_ref` strip-
shaped for `k < k_split` (~680 MiB at 4096²). Achieving that needs
strip-shape gauss_ref AND gauss_alt for shallow levels, which
requires per-strip REF gauss reduce + per-strip DIST gauss reduce
inside the band loop's strip-major outer (the existing
`_reduce_gauss_pyramid_strip_walker` writes full-image strip-major
gauss; making the OUTPUT buffer strip-shaped requires the gauss
reduce caller chain to be strip-major-outer of LEVEL 0, populating
levels 0..k_split per strip with halo-aware reads).

That's a multi-commit restructure of the gauss reduce + color stage
+ DIST dispatch ordering, beyond the P2.x mechanical-shrink scope
that landed P2.3-P2.6 cleanly. Documented for the next session in
docs/STRIP_PROCESSING.md ("Phase 2 recipe" §11).

**Cumulative shrink from P2.1c baseline (4096² Mode B):** 3457 →
1502 MiB (-1955 MiB, -56.6%). JOD bit-identical at all tested
sizes. Mode E + CappedPyramid + strip parity tests unchanged.

### cvvdp-gpu — perf: P2.5+P2.6 — weber_scratch strip-shaped + P2.5 noted (-271 MiB at 4096²) (2026-05-27)

Shrinks `WeberScratch.{l_bkg_fine, vscratch_a, log_l_bkg,
log_l_bkg_dis, vscratch_c}` from full-image to strip dims
(`fine_w × R_k` for fine planes, `coarse_w × R_k` for v-scratch) at
shallow non-baseband levels (`k < k_split`) in `StripMode::Pair`.
Deep levels keep full-image sizing.

Each (s, k) iteration overwrites these scratch buffers in place; no
cross-strip data dependency since:
- `l_bkg_fine` is read by `subtract_weber_3ch_strip_kernel` in the
  same dispatch chain as it's written (Stage 1 → Stage 3).
- `log_l_bkg` is written by REF strip helper Stage 3 and read by
  CSF Stage 4 in the SAME (s, k) iteration.
- `log_l_bkg_dis` is a throwaway DIST destination.
- `vscratch_a` / `vscratch_c` are intermediate upscale scratch.

The DIST CSF helper's `byte_off_*_window` (used for slicing weber
scratch reads/writes) now switches between strip-local-offset-0 and
full-image-top_global based on `band_ref_strip_local`. The REF
strip helper uses offset 0 unconditionally for weber scratch (always
strip-shaped at shallow Mode B). A separate `byte_off_*_full` carries
the top_global slice for gauss_ref/gauss_alt reads (still full-image
until P2.7).

**P2.5 status:** the brief listed P2.5 as `m_raw/m_mid/m_blur` strip-
shaping. That work landed implicitly in P2.4 (`DBandsTransient::new_strip`
allocates ALL five buffer kinds at strip dims together — they're co-
allocated in the same struct with no semantic split). No separate
P2.5 commit; the P2.4 entry above documents the co-shrink.

**Memory delta (4096² h_body=256):** nvsmi delta 1753 MiB → 1482 MiB
(-271 MiB, -15.5%). Cumulative from P2.1c baseline: 3457 → 1482
(-1975 MiB, -57.1%). Wall-time 3.72s → 2.91s.

JOD bit-identical at 128² / 1024² (|diff|=0). Mode E + CappedPyramid
smoke unchanged.

### cvvdp-gpu — perf: P2.4 — DBandsTransient t_p_*/m_* strip-shaped (-1704 MiB at 4096²) (2026-05-27)

Adds `DBandsTransient::new_strip(client, n_strip)` that allocates the
five buffer kinds (`t_p_ref`, `t_p_dis`, `m_raw`, `m_mid`, `m_blur` —
3 channels each = 15 buffers) at `bw × R_k` instead of full-band
`bw × bh`. Used by `_run_d_bands_strip_major_shallow` for shallow
non-baseband levels.

`_run_band_masking_strip_s_for_level` gains `transients_strip_local:
bool` parameter. When `true`:
- Stage 1-3 (min_abs / blur_h / blur_v) byte offsets for t_p_* / m_*
  go to 0 instead of `top_global * bw * 4` — buffer row 0 IS top_global
  at this dispatch (the CSF helper just wrote it there).
- Stage 4 (mult_mutual) body offsets become `(body_offset_y - top_global)
  * bw * 4 = HALO * bw * 4` — body sits HALO=6 rows down within the
  strip buffer.
- Strip-aware kernels still receive `body_off_kernel = top_global`
  unchanged — reflection math against `logical_h = bh` is identical
  because the buffer-relative index `reflect - top_global` matches
  both the full-image-sliced and strip-local layouts.

`_dispatch_dist_weber_csf_strip_s_for_level` reuses the
`band_ref_strip_local` flag to also skip the slice for its t_p_*
writes when called from the strip-major outer.

**Memory delta (4096² h_body=256):** nvsmi delta 3457 MiB → 1753 MiB
(-1704 MiB, -49.3%). Wall-time 4.67s → 3.72s. JOD bit-identical at
128², 1024² (|diff|=0). Mode E + CappedPyramid smoke unchanged.

The biggest single-commit shrink — 15 strip-shaped buffers per shallow
level × `(bh - R_k) / bh` ≈ 86% per-level savings (e.g. R_k=572 vs
bh=4096 at level 0). The legacy level-major caller passes
`transients_strip_local: false` and keeps its full-image transient
behavior (Mode E + Full).

### cvvdp-gpu — perf: P2.3 — bands_ref strip-shaped + gauss_alt added (2026-05-27)

Adds `WeberScratch.bands_ref_strip: Option<[Handle; 3]>` per shallow
non-baseband level (`k < k_split`) — same `fine_w × R_k` shape as
`bands_dis_strip`. The REF weber finalize for shallow levels is
DEFERRED to the strip-major outer band loop (mirrors the existing
DIST defer pattern):

1. New helper `_dispatch_ref_weber_strip_s_for_level(s, k)` runs
   stages 1-3 (upscale_v/h, per-channel upscale, fused
   subtract+weber) on REF inputs → writes `bands_ref_strip` body+halo
   + `log_l_bkg` (REF's per-pixel log₁₀(L_bkg)).
2. `_run_d_bands_strip_major_shallow` calls REF strip helper THEN
   DIST CSF strip helper per (s, k) in strict lockstep. CSF reads
   `bands_ref_strip` (just written) + `bands_dis_strip` + `log_l_bkg`.
3. `_dispatch_dist_weber_csf_strip_s_for_level` gains
   `band_ref_strip_local: bool` parameter — when `true`, the
   `band_ref_*` handles are strip-local (skip the byte-offset slice).

REF gauss data persists through DIST gauss dispatch via a new
`gauss_alt: Option<Vec<Level>>` full-image alt-buffer (allocated only
in `StripMode::Pair`) and a `_maybe_swap_gauss_alt_post_ref()` swap
after REF weber finalize. Post-swap, `gauss_alt` holds REF gauss data,
`gauss_ref` is overwritten by DIST gauss. The REF strip helper reads
from `gauss_alt`; DIST helpers read from `gauss_ref` as today.

**Memory delta (4096² h_body=256):** `bands_ref` shallow non-baseband
levels (~256 MiB) freed; `gauss_alt` full-image (~256 MiB) added.
Net nvsmi delta ≈ 0 MiB for P2.3 alone (3457 → 3453 MiB ± noise).
The estimator already counted bands_ref strip-shaped, so the
analytical reduction landed pre-runtime; the deferred `gauss_alt`
overhead will be reclaimed by P2.7's gauss strip-shaping.

**JOD parity:** bit-identical (|diff|=0.0) at 128² / 1024² / 4096²
(h_body 32, 256, 512). Wall-time 1024² Mode B 1.4s, 4096² Mode B
4.45s (matches P2.1c baseline within noise).

Foundation commit for the P2.x shrink chain — establishes the
interleaved REF-then-DIST strip dispatch pattern and the
gauss_alt-swap mechanism. P2.4-P2.6 + P2.8 ride this foundation;
P2.7 retires `gauss_alt` by strip-shaping both gauss pyramids.

### cvvdp-gpu — perf: P2.1c — outer loop strip-major for k < k_split (2026-05-27)

Inverts the outer dispatch loop in `_run_d_bands_band_loop` for
shallow levels (`k < k_split`) in Mode B. New helper
`_run_d_bands_strip_major_shallow(k_split)`:

1. Allocates `DBandsTransient` for every shallow level upfront
   (k_split levels simultaneously, ~1.3 GiB at 4096²).
2. Iterates `for s in 0..n_strips { for k in 0..k_split }`.
3. Per (s, k): P2.1b CSF strip helper (body+halo of `t_p_*[k]`)
   → P2.1a masking strip helper (body of `d_strip[k]`, inline pool).

Deep levels (k >= k_split) and baseband continue level-major-outer
via the existing band loop (Mode B skips already-handled shallow
levels via `if mode_b_pair && k < k_split { continue }`).

Correctness rests on P2.1b's body+halo CSF guarantee: each strip's
masking finds valid t_p_*[k] data in its halo window because the
same strip's CSF just wrote those rows. JOD bit-identical
(|diff|=0.0) Full vs Mode B at all tested sizes (128², 256², 512²,
1024², 4096²) including 8-strip-per-image configurations.

Measured at 4096² (subprocess-per-cell, nvidia-smi delta):
- Wall-time: Mode B = 4.67s vs Full = 5.61s (**Mode B is FASTER**)
- nvsmi delta: Mode B = +3457 MiB vs Full = +4225 MiB (-18.2%)

The wall-time speedup is from avoiding cubecl pool churn — strip-
major-outer pre-allocates k_split transients vs lazy-per-level's
alloc/drop/reuse cycle. The memory delta vs Full stays at -18.2%
(same as P2.1b) because strip-major-outer alone doesn't shrink
full-image transients; P2.4-P2.8 do that work.

Pre-existing failures (not P2.1c-related):
- compute_dkl_jod_host_pool_matches_pycvvdp_at_73x91_odd_on_cpu_backend
- mode_e_strip_h_body_explicit_override (h_body=768 violates align)

Files:
- crates/cvvdp-gpu/src/pipeline.rs — `_run_d_bands_strip_major_shallow`
  + Mode B early-call from `_run_d_bands_band_loop`.
- crates/cvvdp-gpu/docs/STRIP_PROCESSING.md — P2.1c landed section.
- benchmarks/cvvdp_mode_b_p21bc_2026-05-27.{csv,meta} — measurement
  artifacts (commit hashes for reproduction).

### cvvdp-gpu — perf: P2.1b — per-strip CSF helper dispatches body+halo (2026-05-27)

Extends `_dispatch_dist_weber_csf_strip_s_for_level` to dispatch
stages 1-4 over a body+halo window at level k for shallow levels
(`k < k_split`). Grows `bands_dis_strip` and `upscaled_c_strip` per-
strip buffers from body-only to back-projected `R_k = mode_b_strip_h_at_level(
k, h_body, k_split)` per the P2.0 helper. Bit-identical JOD vs Full
mode at 128² / 256² / 512² / 1024² (|diff|=0.0) at strip counts
1-8 — verified by new `tests/strip_mode_b_csf_halo_parity.rs`.
Inter-strip halo writes are deterministic: the CSF kernel produces
the same value for overlap rows from any strip touching them
(band_ref + log_l_bkg are full-image; bands_dis_strip recomputes
from full-image gauss + deterministic-per-global-row upscale).

Memory at HEAD (subprocess-per-cell nvsmi):
- 1024² StripPair(256) = +353 MiB (Full = +385 MiB), -8.3%
- 4096² StripPair(256) = +3457 MiB (Full = +4225 MiB), -18.2%

The -18.2% at 4096² is the necessary cost of body+halo strip
buffers: today's strip-only -22.7% used body-only sizing which
could not support halo dispatch. The follow-on P2.1c (outer-loop
inversion) is mechanical; the user-visible memory drop toward the
-80% estimator target arrives with P2.4-P2.8 (transient buffer
shrinks enabled by strip-major-outer dispatch).

### zensim-gpu — perf: Phase 1b chunk 3 — GPU diffmap kernels WIRED + VALIDATED, default-OFF (2026-05-27)

Wires the Phase 1b chunk-1/2 CubeCL diffmap kernels (`b50b8f57`,
`f66930c7`) into the diffmap-producing methods behind an opt-in env
gate `ZENSIM_GPU_DIFFMAP=1`. New `linear_to_positive_xyb_kernel`
(linear-RGB sibling of the sRGB color kernel) feeds the GPU feature
pipeline from the linear planes the diffmap API receives. New
`GpuDiffmapScratch` (inner WithIw `Zensim<R>` + base accumulator +
per-scale dm planes + cached trained weights) drives the
`per_scale_weighted_ssim_kernel` → `pow2x_upsample_add_kernel` →
`diffmap_trim_padded_kernel` chain. New `tests/cpu_gpu_diffmap_parity.rs`
proves the GPU diffmap matches the CPU canonical
`compute_with_ref_and_diffmap_linear_planar` **pointwise to ≤ 2.08e-4
absolute** (5 fixtures × 4 distortions, CUDA RTX 5070; tolerance 1e-3).
The gate is **default-OFF** (honest-stop on the wall axis): the scalar
score must still come from the CPU canonical path because the
GPU-feature → V0_3 MLP score is catastrophically wrong on the pinned
zensim 0.3.0 (measured −77.13 vs CPU +85.33 on a real CID22 image —
pre-existing WithIw-feature / V0_3-MLP-sensitivity bug, `docs/DIFFMAP_DIVERGENCES.md`
§2b + §9), so GPU-diffmap + CPU-score is strictly slower than the
CPU-only default. Production default is unchanged (zero regression);
the validated GPU diffmap infrastructure unblocks the chunk-N+1
score-path fix that flips the gate default-ON.

### zen-cloud — cloud-agnostic worker carve, Phase A (no behaviour change) (2026-05-26)

Carves the `vastai-fleet` crate into a cloud-agnostic trait layer + a
vast.ai backend + a generic deployed-worker binary, per the zen-cloud
spec §1.7 Phase A. No behaviour change — the production worker stays on
the proven async path; all 25 unit + 7 CLI tests stay green.

- New crate **`zen-cloud-core`** (`de66b1b0`): pure trait surface
  (`JobQueue` / `BlobStorage` / `Heartbeat` / `CredentialSource` /
  `WorkerHost`), value types (`Chunk` / `ChunkId` / `ChunkOutcome` /
  `ArtifactKey` / `BlobMeta` / `WorkerId` / `WorkerStatus` /
  `WorkerSummary` / `CloudError`), and a generic `run_worker` job loop.
  Zero gpu / cloud-SDK / parquet deps.
- **`vastai-fleet` crate renamed to `zen-cloud-vastai`** (`ccb4250f`),
  now lib+bin: the proven worker/parse modules move unchanged; a new
  `cloud` module implements the core traits for vast.ai (R2 storage via
  the existing s5cmd client + `ls_keys`/`rm`, `/proc/1/environ`
  credentials, nvidia-smi host, R2 heartbeat, R2-token-race chunk
  queue). The operator CLI keeps the `vastai-fleet` binary name +
  `self-destroy`/`status`/`destroy`/`watch`.
- New binary **`zen-sweep-worker`** (`1d6eef0e`): the cloud-agnostic
  deployed compute binary. `--backend vastai` (cargo-feature-gated)
  dispatches `worker` to the same `cmd_worker` path as `vastai-fleet
  worker` — byte-identical sweep output.
- Workspace-wide rename fixups: `Dockerfile.sweep.v26` +
  `.github/workflows/sweep-image.yml` build/copy/smoke both binaries
  and run the deployed worker via `zen-sweep-worker`;
  `onstart_unified.sh` / `onstart_feature_backfill.sh` /
  `onstart_source_features.sh` invoke `zen-sweep-worker worker
  --backend vastai`; stale `vastai-fleet`/`crates/vastai-fleet` doc-path
  references across `zen-metrics-cli` + `iwssim-gpu` + scripts updated.

### zen-cloud — SaladCloud provider + shared S3 helper, Phase C (2026-05-26)

Adds SaladCloud as a vast.ai alternative (spec §1.9) and factors the R2
client out of `zen-cloud-vastai` into a shared helper. vast.ai's 25 unit
+ 7 CLI tests stay green; the new crates add 3 + 22 tests.

- New crate **`zen-cloud-s3`** (`3832ab33`): the shared
  S3-compatible `BlobStorage` helper. The s5cmd-backed client
  (`S3Client`, ex-`R2Client`) + the `BlobStorage` impl (`S3BlobStorage`,
  ex-`R2BlobStorage`) relocate behaviour-identical out of
  `zen-cloud-vastai`; the only change is a field-based constructor so any
  provider can build one. `zen-cloud-vastai` now depends on it and
  re-exports `R2Client` / `R2BlobStorage` under their historical names —
  every internal call site + test compiles unchanged. R2 IS
  S3-compatible, so one impl serves vast.ai + SaladCloud + DO + AWS.
- New crate **`zen-cloud-salad`** (`2f2336e2`): the SaladCloud provider.
  `JobQueue` is a local HTTP receiver fed by the baked-in
  `salad-http-job-queue-worker` sidecar (the app side is HTTP, not gRPC —
  the gRPC contract is internal to the sidecar; see `SALAD.md`);
  `CredentialSource`/`WorkerHost` read the container-group env
  (`SALAD_MACHINE_ID` / `SALAD_CONTAINER_GROUP_ID` + BYO R2/S3) + a
  minimal IMDS client; `BlobStorage` reuses `zen-cloud-s3`; `Heartbeat`
  is a log-only no-op. A `launch` module hand-rolls the public-API
  provisioning (resolve GPU class, create queue, create container group
  with `queue_connection`, push job chunks) with `reqwest` + `serde`.
- **`zen-sweep-worker --backend salad`** (`ab9ada72`): the salad arm
  drives the generic `run_worker` loop with the Salad traits + the SAME
  inline encode+score compute vast.ai runs
  (`process_chunk_inline`, now re-exported). Feature split: `salad` =
  GPU-free glue (builds clean), `salad-sweep` = glue + the shared
  inline-sweep compute tree (the deploy-image build).

### cvvdp-gpu — perf: Path A Phase 1 prep — `upscale_v_strip_kernel` gains `src_strip_offset` (2026-05-26)

Adds `src_strip_offset: u32` parameter to `upscale_v_strip_kernel` so
the kernel can read from a strip-local coarse source buffer (rather
than the full-image gauss buffer it implicitly assumed). Mirrors the
existing `downscale_strip_kernel::src_strip_offset` pattern. With
`src_strip_offset = 0` behavior is bit-identical to the prior
signature — JOD unchanged at 9.4583 on the 1024² mem_mode_b_vs_full
example. New parity test `upscale_v_strip_with_src_offset_matches_full_interior_body`
verifies the non-zero path at offset=8 on a 32×32→32×64 expand.

This is the kernel-level unblock for full Phase 1 wiring (per-strip-
sized `gauss_dis` allocation + outer-strip-inner-band dist dispatch).
Driver memory unchanged at this commit — the kernel-prep alone does
not allocate or route through a per-strip gauss buffer. Commit 98c057ef.

### scripts/sweep — Security: build_per_codec_training{,_extended} gain pre-write Mode-A + Mode-B guards (2026-05-26)

`scripts/sweep/build_per_codec_training.py` +
`scripts/sweep/build_per_codec_training_extended.py` now route their
per-codec parquet output through
`zensim::scripts::canonical_corpus::join_safety::guard_metric_table`
before `pq.write_table(...)`. The guard catches the 2026-05-25 kadid/tid
corruption recurrence shape (mock columns + bit-identical-to-human_score
metric columns + any ssim2/cvvdp/butter/dssim column constant within
every `image_basename` group). DuckDB joins themselves were already using
the correct full per-pair key + explicit dedup; this commit closes the
remaining post-join surface.

Soft cross-repo import (try/except) so fleet workers without the zensim
repo on disk still run, but log a stderr warning that the guard is
skipped. See `zensim/benchmarks/joinsafety-migration-2026-05-26/MIGRATION_EVIDENCE.md`.

### ssim2-gpu — feat: cross-repo SSIM2 GPU parity test skeleton + doc (dedup Chunk H) — 2026-05-26

Skeleton landing for the master dedup audit's Tier-0 #2 finding
("two GPU SSIM2 backends, no parity test"):

- New `cudarse-parity` Cargo feature (OFF by default) that pulls in
  `coefficient` as an optional dev-time dep so the parity test can
  score the same fixtures via BOTH this crate's CubeCL backend AND
  `coefficient::gpu::GpuMetrics` (cudarse / turbo-metrics).
- New integration test
  [`tests/cudarse_parity.rs`](crates/ssim2-gpu/tests/cudarse_parity.rs)
  marked `#[ignore]` because coefficient's `gpu` feature
  transitively requires the archived `~/work/turbo-metrics` tree
  plus a CUDA-12-compatible toolkit (the dev-host CUDA 13.2 ptxas
  rejects the archived kernels' `sm_70` target). 3 CID22 fixtures
  at JPEG q90 / q50 / q20 cover the SSIM2 range where the two
  backends are most likely to diverge. Initial tolerance
  `0.5 SSIM2 points` (loose; tighten once measured agreement is
  recorded).
- New doc
  [`docs/GPU_METRIC_PARITY.md`](docs/GPU_METRIC_PARITY.md) records
  methodology, tolerance + rationale, run instructions, the three
  blockers that prevent runtime on this dev host, and a
  measured-agreement table to populate when the CUDA toolchain
  unblock lands.

Skeleton-only; no measured numbers yet. The audit's recommended
next step ("parity test gate first; then pick one backend or a
shared `zen-gpu-metrics`") remains queued behind the toolchain
fix.

### cvvdp-gpu — feat: restore CappedPyramid + Phase 3 strip-aware pool walker (task #79) — 2026-05-26

Two-part follow-on to the 2026-05-26 architectural deep-dive
(`4f30487c`):

**PART A — `MemoryMode::CappedPyramid { levels }` restoration**: 
JOD-shifting Option B safety net re-introduced as a fourth variant
on `cvvdp_gpu::MemoryMode`. Opt-in only (`Auto` never picks it). New
constructors `Cvvdp::new_capped_pyramid{,_with_geometry}`, new
estimator `pipeline::estimate_gpu_memory_bytes_capped`, and the
unified `new_with_memory_mode` + `CvvdpOpaque` dispatch. 8 smoke
tests in `tests/capped_pyramid_smoke.rs` covering construction,
JOD-finite, estimator monotonicity, clamping behaviour, and error
paths. Umbrella `zenmetrics_api::MemoryMode` does **not** gain
CappedPyramid — the umbrella stays the metric-preserving subset;
callers needing the safety net construct the typed Cvvdp /
CvvdpOpaque directly.

**PART B — Phase 3 strip-aware pool walker**: first strip-aware
kernel in the cvvdp pipeline. New `pool_band_3ch_offset_kernel`
(in `kernels/pool.rs`) takes a `start_offset` so the host can
dispatch on a row-slab of a larger d-plane. New
`_pool_and_finalize_jod_strip` walker partitions each band's
per-pixel pool into row-strips sized `strip_h_body >> k` and
dispatches the offset kernel per slab. Atomic-adds are associative
across slabs so JOD is bit-exact against Full mode within the same
per-call ordering noise band. `compute_dkl_jod_with_warm_ref` +
`score_from_linear_planes_with_warm_ref` route through the strip
pool when in Mode E. Test-only `strip_dispatch_counter()` accessor
(via `#[doc(hidden)]`) tracks per-band strip iterations so the
parity test can assert N >= 2 at 1024² with `h_body=512`.

5 new parity tests in `tests/strip_mode_e_phase3.rs` (64², 1024²,
counter, repeat-determinism, full-mode counter gating). All 11
existing `strip_mode_e_parity.rs` tests still pass.

**Memory impact**: zero so far — only the pool stage iterates in
strips; d_scratch / bands_ref / bands_dis / weber_scratch all
remain full-image-sized. The pool stage is a tiny fraction of the
working set. This landing proves the walker is correct end-to-end
(atomic associativity + per-strip iteration + counter visibility);
the memory wins are gated on porting the CSF / masking chain /
pyramid kernels to take `(body_offset_y, logical_h)` parameters so
they reflect at logical-image edges rather than strip-buffer
edges. See `docs/STRIP_PROCESSING.md` for the six-chunk roadmap.

`benchmarks/cvvdp_mode_e_phase3_2026-05-26.csv` documents the
JOD-parity-confirmed walker baseline and the memory-equivalence
caveat (Full and Strip estimate-bytes match because d_scratch is
still full-image).

### cvvdp-gpu — docs: Phase 3 architectural deep-dive — multi-day refactor confirmed (task #79) — 2026-05-26

Investigation 2026-05-26 traced the cvvdp pipeline strip-blocking
properties end-to-end against the Phase 1+2 foundation
(`fb8e93d9`). `docs/STRIP_PROCESSING.md`'s "Phase 3 design notes"
section rewritten with:

- Measured memory breakdown by buffer at 1-24 MP (`d_scratch ~ 60%`,
  pyramids `~ 30%`, weber `~ 9%` — all three must shrink to meet
  the < 70% Full target).
- Per-kernel reflection-boundary trace: every pyramid + PU blur
  kernel reflects at array edges, needs `(body_offset, logical_h)`
  to be strip-aware.
- Deep-band problem table (at `STRIP_H_BODY_DEFAULT = 512`, levels
  k >= 4 see PU blur halo comparable to or larger than strip body).
- Three implementation approaches with cost estimates: (A) modify
  ~7 kernels with body_offset param; (B) extend strip halo to fit
  unmodified kernels; (C) hybrid shallow-strip + deep-full.
- Concrete next-agent hints: downscale's pycvvdp bug-compat delta
  uses LOGICAL parity not strip parity; dssim-gpu pattern as
  template (constant HALO=256 works for 5 scales but doesn't scale
  to cvvdp's 9 levels).

Failure-mode clause invoked per CLAUDE.md task #79: foundation
remains as the deliverable; deep architectural rework deferred.
User-visible consequence unchanged from Phase 2: cvvdp at > 16 MP
on small-VRAM boxes returns `Error::TooBigForFull` until walker
lands. (commit ec901810)

### cvvdp-gpu — feat: re-introduce `MemoryMode::Strip { h_body }` (Mode E, task #79) — 2026-05-26

Task #79 reintroduces a Strip variant for cvvdp that is
**JOD-preserving**, unlike the rolled-back task #77 capped-pyramid
variant. Mode E shrinks the *working set* without changing the
algorithm: the reference-side state lives in dedicated full-image
buffers (survives intervening one-shot `score()` calls); the dist
side runs the standard pipeline against them. Per-band atomic-pool
sums are associative across strips, so the final JOD equals
Full-mode JOD within the documented Atomic<f32> reduction-order
noise band (≤ 1e-4 abs JOD on CUDA).

Phase 1+2 (commits 1c3445c3, 89b2a5f6) lands:

- `MemoryMode::Strip { h_body: Option<u32> }` variant +
  `ResolvedMode::Strip { h_body }`.
- `STRIP_H_BODY_DEFAULT = 512`, `STRIP_ALIGN = 2^(MAX_LEVELS-1) = 256`.
- `Cvvdp::new_strip` + `Cvvdp::new_strip_with_geometry`;
  `new_with_memory_mode` dispatches Strip + Auto.
- `RefFullState` struct (full-image per-level Weber bands +
  per-non-baseband log_l_bkg + baseband gauss + baseband scalar).
  Allocated lazily on first `warm_reference` in strip mode.
- New `copy_f32_kernel` in `kernels::pool`.
- `warm_reference` snapshots shared scratch into `ref_full_state`;
  `compute_with_warm_ref` (+ host-pool / diffmap / linear-planes
  siblings) restores it ahead of dist dispatch.
- `Cvvdp::has_warm_reference()` + `is_strip_mode()` accessors;
  `CvvdpOpaque` forwarders.
- Umbrella `From<MemoryMode> for cvvdp_gpu::MemoryMode` maps
  `Strip{h_body}` through (was falling back to Auto).
- Umbrella `Metric::has_cached_reference()` consults cvvdp's
  accessor (was hard-coded `false`).
- `estimate_gpu_memory_bytes_strip(w, h, h_body)` (conservative
  for Phase 2 — returns Full + ref-cache delta).
- New `tests/strip_mode_e_parity.rs` (11 tests) + new
  `cached_ref_cvvdp_strip_n_distortions` umbrella test pinning
  the JOD-parity contract within 1e-4 abs JOD.
- `docs/STRIP_PROCESSING.md` rewrite documenting JOD-preservation
  invariant + Phase 1-5 status table + Phase 3 design notes.

**Phase 3** (per-strip dist walker that shrinks the dist working
set) is multi-day follow-on work and is **not** in this commit —
`Auto` still picks Strip when Full overflows the cap, but Phase 2's
strip mode has the same dist memory profile as Full plus a small
ref-cache delta. The structural plumbing (`RefFullState`,
snapshot/restore, `StripConfig` storage, `h_body` plumbing) is
permanent and Phase 3 builds on top.

### cvvdp-gpu — refactor: roll back capped-pyramid Strip variant (task #77) — 2026-05-26

cvvdp's `MemoryMode::Strip { h_body, capped_levels }` only ever
implemented capped-pyramid scoring (a Full pipeline with the
pyramid depth clamped to `k`), which **changes the JOD value at
any k < 9** — different cap = different metric. There is no
panorama use case in the production corpus to justify the redesign
cost of a true strip walker, so the variant was removed entirely
rather than left as an opt-in landmine that silently changes the
metric output.

Surface changes:

- `cvvdp_gpu::MemoryMode` is now `{ Auto, Full }` (was
  `{ Auto, Full, Strip{..}, Tile{..} }`).
- `cvvdp_gpu::ResolvedMode` stays `{ Full }`.
- `Cvvdp::new_with_geometry_and_cap` removed; use
  `Cvvdp::new_with_geometry` (the cap parameter was the only
  thing it added).
- `estimate_strip_gpu_memory_bytes` removed (always returned
  `None`); replaced lib-root re-export accordingly.
- `Error::ModeUnsupported(&'static str)` variant kept for API
  stability — never fires today since the umbrella `From` maps
  `umbrella::Strip` / `umbrella::Tile` down to `cvvdp_gpu::Auto`.

Downstream impact: `zenmetrics-api` `From<MemoryMode> for
cvvdp_gpu::MemoryMode` now routes `umbrella::Strip` and
`umbrella::Tile` both to `cvvdp_gpu::Auto` (closest-meaning
policy). `publish = false` on cvvdp-gpu — no crates.io users to
break.

Methodology trace from the original capped-pyramid sweep moved to
`crates/cvvdp-gpu/docs/archived/` so the cap-vs-JOD-drift sweep
data and run log survive for future review.

(commit TBD)

### iwssim-gpu — investigate: native-RGB strip path (task #57) — 2026-05-26

User asked: should we add an on-device sRGB→gray conversion for the
strip pipeline, instead of running `rgb_u8_to_gray_bt601` on the host
before each `compute_rgb_with_reference_stripped` call? The brief
required a measurement step before implementing.

Probe at `crates/iwssim-gpu/examples/native_rgb_perf_probe.rs`
(committed CSV + meta at `benchmarks/iwssim_native_rgb_perf_2026-05-26.{csv,meta}`)
measures three modes across 256² / 1024² / 2048² / 4096²: host-side
conversion + strip walker (production path), gray-baseline (lower
bound), and a new `compute_rgb_with_reference_stripped_native` that
packs each strip's sRGB into pinned packed-u32 and runs the existing
`rgb_u32_to_gray_kernel` to populate `g_dis` on the device.

Measurement (4 runs on RTX 5070, GPU contention from sibling agents):
host-side conversion is 35-41% of per-call wall time at 1024², 7-40%
at 2048², 13-26% at 4096². Above 10% at every size — the decision
threshold for "implement" was clearly cleared.

Implementation shipped + parity-locked (7 tests in
`tests/rgb_strip_native.rs`, all pass on real CUDA, native vs host
agree to 5e-5 relative). Perf result of the *naive* implementation is
NOT a consistent win: 2048² mostly wins (~24 ms vs ~43 ms host_conv),
4096² loses in all 4 probe runs (~470 ms native vs ~330 ms host_conv).
Root cause documented in the meta file: per-strip alloc of fresh
packed-u32 staging + fresh g_dis_strip + extra kernel launch
outweighs the saved host conversion at large sizes. Follow-up
queued: reuse buffers across strips, single whole-image pinned
staging with kernel-side strip offset, or fused pack+convert kernel.

### Workspace — fix: switch `[profile.release]` to thin LTO + line-tables-only debug (task #59) — 2026-05-26

`cargo build --release --workspace` on `ubuntu-latest` GitHub runners
(16 GB RAM) hit OOM at the final LTO link. The workspace build step
in `.github/workflows/ci.yml:54` had been gated off with
`if: matrix.os == 'ubuntu-latest' && false` since commit `4a729b65`
on 2026-05-05 exactly because of this.

Root cause: `[profile.release]` carried the triple `lto = "fat"` +
`debug = "full"` + `codegen-units = 1`. Fat LTO loads every CGU of
every workspace member into one LLVM context for the link; `debug =
"full"` drags every DWARF DIE through that context; `codegen-units =
1` forbids parallelising the per-CGU passes. The combination
multiplied peak link RSS into the OOM range on small CI runners.

Fix: `[profile.release]` now uses `lto = "thin"`, `debug =
"line-tables-only"`, and `codegen-units = 16`. Thin LTO keeps the
cross-CGU inlining / DCE / devirt wins; line-tables-only debug keeps
backtrace line numbers for panics + `whereat` reports. Production
release builds that genuinely want fat LTO opt in with the new
`[profile.release-fat]` profile (`cargo build --profile release-fat`).
`[profile.bench]` stays fat LTO since benchmarks measure single
binaries on memory-rich hosts.

Measured on the water-cooled 7950X / 50 GB sandbox (clean target/,
10 of 12 workspace members — the two that depend on the path-pinned
`jxl-encoder` are blocked on a separate compile error in
`vardct/perceptual_backend.rs:637` that another agent is fixing):

- fat LTO + debug=full + cu=1: rustc parent max RSS 3.63 GB, wall 63s
- thin LTO + line-tables + cu=16: rustc parent max RSS 0.95 GB
  (-74%), wall 53s

CI `&& false` gate on `.github/workflows/ci.yml:54` removed so the
workspace release build runs again on `ubuntu-latest`.

(commit TBD)

### zensim-gpu — fix: replace v0.3 score shim with real `zensim::score_features_with_profile_and_codec` — 2026-05-26

Task #71: every `ZensimOpaque::compute_srgb_u8` / `compute_pixels` call
with `default_weights()` (the umbrella's `MetricParams::default_for(Zensim)`)
silently bypassed the V0_3 MLP head + PCHIP spline + per-codec affine
pipeline. The `score_features_with_profile_and_codec_compat` shim in
`crates/zensim-gpu/src/opaque.rs:824` recomputed the legacy 228-element
V0_2 linear formula and returned it under a V0_3 label.

The bump *had* happened — `zensim::lib.rs:262` re-exports the real
`score_features_with_profile_and_codec` from the path-pinned
`../zensim--principled-activity/zensim`. Deleted the shim and routed
`score_from_profile_vec` directly through the real function.
Regression coverage: `crates/zensim-gpu/tests/opaque_default_weights_v03.rs`
(commit 77e0c429).

### zensim-gpu — test: per-slot CPU parity for 372-feature WithIw IW block slots 300..372 — 2026-05-26

Task #72: previously only a structural smoke test
(`with_iw_structural_noisy`) covered slots 300..372. Added per-slot
parity tests at two fixture sizes (64×64 noisy gradient + 128×128
checkerboard + noise) under the same `5e-3 rel` budget the masked
block tests use. CPU reference reached via
`zensim::Zensim::compute_extended_features` — the `latest()`
profile carries `compute_iw_features: true` in its `ProfileParams`,
so `combine_scores` Pass 4 appends the 72 IW slots to the returned
372-feature vector. No `training` feature gate required.

`crates/zensim-gpu/docs/FEATURE_PARITY.md`'s "IW block validation"
section rewritten to reflect the new direct per-slot coverage —
previously claimed "CPU rev predates IW" which became stale when the
workspace path-pin was updated (commit 9d2b5bf2).

### zensim-gpu — feat: byte-identity short-circuit on opaque `compute_srgb_u8` / `compute_pixels` — 2026-05-26

Mirrors the CPU canonical `Zensim::compute(...).score()` behaviour:
when `ref_rgb == dis_rgb` byte-for-byte, the opaque API now returns
`Score { value: 100.0, .. }` without running the GPU kernel. Without
the short-circuit, the f32 SSIM / blur / max-pool pipeline picks up
sub-ULP residuals on byte-equal inputs that the V0_3 PCHIP spline
(`extrapolate_score=true`) maps to arbitrary out-of-dial values
(observed -89.9 on `dispatch_zensim`'s 256² fixture).

Identity short-circuit lives at the opaque API layer
(`crates/zensim-gpu/src/opaque.rs::identity_short_circuit`); the
typed `Zensim<R>::compute_features_vec` path stays untouched so
research callers can still observe the f32 drift if they want it.
Coverage:
`crates/zensim-gpu/tests/cpu_gpu_feature_sweep.rs::identity_short_circuit_does_not_corrupt_subsequent_runs`
(round-trip identity → distortion → identity sequence).

### zensim-gpu — test: comprehensive 372-slot CPU↔GPU feature sweep — 2026-05-26

New file `crates/zensim-gpu/tests/cpu_gpu_feature_sweep.rs` runs the
per-slot WithIw parity check across:

- **3 fixture sizes** — 64×64, 192×192, 320×240. Spans single-strip,
  multi-strip aligned, and multi-strip non-square aspect.
- **4 content patterns** — gradient, checkerboard, single impulse,
  photographic low-frequency wash. Exercises smooth → high-edge →
  masked-IW high-activity → max-pool corner cases.
- **3 distortion magnitudes** — n4 (near-identical), n16 (clearly
  perceptible), n48 (heavily distorted). All non-zero so the test
  stays inside f32's precision band at non-power-of-2 scales.

12 generated `sweep_*` tests + 1 short-circuit roundtrip test = 13
tests in this file, 4 × 4 × 3 = 48 distinct
(size × content × distortion) parity checks each over all 372
slots. Same `(2e-3 abs / 5e-3 rel)` budget as the existing
`extended_parity` tests.

### zensim-gpu — feat: regime-aware `estimate_gpu_memory_bytes` calibrated against measured data — 2026-05-26

Issue #16: the estimator returned the same byte count regardless of
regime; measured at 67 MP it was 90 % over for Basic and 35-36 %
*under* for Extended / WithIw. Made the signature
`estimate_gpu_memory_bytes(width, height, regime: ZensimFeatureRegime)`
with per-regime (BASE, BETA) coefficients fit via grid search on
`benchmarks/mem_per_metric_2026-05-26.csv` (24 zensim rows, 8 sizes ×
3 regimes). Max validation residuals: 10.2 % (basic), 20.3 %
(extended), 18.9 % (withiw); all 24 rows land within ±25 % per the
unit test `memory_mode::estimator_matches_measured`.

New constant `CUBECL_OVERHEAD_BYTES = 193 MB` documents the cubecl
runtime pool init floor (separately from the metric's own allocation);
`resolve_auto` reserves it from the caller's cap before comparing
the metric's estimate. Callers (opaque `Auto` path,
`new_with_memory_mode`, `zenmetrics-api/examples/mem_per_metric.rs`)
updated to thread the regime. Drive-by: deleted
`tests/score_v03_parity.rs` (cumulative score test that was a
regression test against the pre-task-#71 shim behaviour — superseded
by per-feature parity coverage; per user direction "prioritize
testing all 372 feature outputs, not cumulative single-scores").
Also deleted `vram_cap_default_is_8gb` test (it asserted a hardcoded
fallback constant only reachable when nvidia-smi probe returns None
— no behavioural contract behind it). Adjusted
`zenmetrics-api/tests/dispatch.rs::dispatch_zensim` to drop the
score-domain `identity ≈ 100` assertion (was asserting against the
shim's legacy linear formula); the finite + metric_name + per-feature
parity coverage in `extended_parity.rs` remains the authoritative
correctness signal (commit 597e1810).

### zen-metrics-cli — feat: `assemble` subcommand — typed full-key corpus join — 2026-05-26

New `zen-metrics assemble` subcommand that replaces the Python
corpus-assembly join layer (`scripts/sweep/build_per_codec_training.py`
plus the `zensim/scripts/canonical_corpus` builders and the ~35 ad-hoc
`pd.merge` scripts) with a TYPED full-key join that makes the 2026-05-25
parquet corruption structurally impossible.

- **`crates/zen-metrics-cli/src/assemble/`** — new module behind a lean
  `assemble` cargo feature (arrow/parquet only; no codecs, no GPU; `sweep`
  enables it too). Ports the four guarantees from
  `zensim/scripts/canonical_corpus/join_safety.py`:
  - `key::PairKey` — a four-field typed key (`image_path, codec, q,
    knob_tuple_json`) with NO ref-only constructor, so the Mode-B ref-only
    collapse is *unrepresentable at compile time* (not merely detected).
  - `join::safe_join` — errors (never `.mean()`-averages) on duplicate
    metric keys; errors if either side lacks a per-pair column.
  - `join::attach_positional` — exact-length positional attach for the
    ref-only KADID/TID path.
  - `join::assert_no_leaked_columns` — rejects `*mock*` columns + raw-metric
    columns bit-identical to `human_score` (Mode A); excludes `mix_*` and
    accepts linear rescales (only bit-identity is the leak signature).
  - `join::assert_not_constant_per_ref` — post-hoc ref-broadcast detector
    with the mean-group-size > 1.5 false-positive gate.
  - The per-codec build path runs both leak + ref-broadcast guards on every
    output before writing.
- **No new heavy deps** — reuses the crate's existing `arrow`/`parquet`
  stack (the union-by-name dtype widening that DuckDB provided is
  reimplemented in `table::Table::union_by_name`); R2 sidecar sync shells to
  `s5cmd` synchronously, mirroring `build_per_codec_training.py`.
- **`scripts/sweep/build_per_codec_training.py`** — marked DEPRECATED with a
  pointer to the Rust subcommand; kept as a fallback/reference (not deleted).
- Tests: `tests/assemble_join_safety.rs` (8 integration tests encoding the
  six corruption-prevention cases) + submodule unit tests.

### cvvdp-gpu — feat: `CvvdpOpaque::new_with_geometry` opaque API for non-STANDARD_4K display configs — 2026-05-26

- **`CvvdpOpaque::new_with_geometry(backend, w, h, params, geometry)`**
  and **`CvvdpOpaque::new_with_geometry_and_memory_mode(.., geometry,
  mode)`** — expose the geometry-aware constructor that
  `Cvvdp::<R>::new_with_geometry` has carried since the host-scalar
  display-spec parity work (2026-05-25 wave). Pre-2026-05-26
  `CvvdpOpaque::new` / `new_with_memory_mode` permanently downcast
  the construction-time PPD to `DisplayGeometry::STANDARD_4K`, so any
  opaque-API caller that needed phone-class (≈340 PPD) / TV-class
  (≈57 PPD) / HMD-class scoring had no path to thread the geometry
  through. Closes the gap that jxl-encoder's `docs/RFC_DISPLAY_CONFIG_BACKFILL.md`
  Phase 1 honest-finding cited.
- **Refactor**: `new_with_memory_mode` now forwards to
  `new_with_geometry_and_memory_mode(.., STANDARD_4K, mode)` so the
  per-backend dispatch isn't duplicated. The two-line `new`
  fallthrough is unchanged. Public API is purely additive — no
  behavior change to existing call sites.
- **Tests**: `crates/cvvdp-gpu/tests/opaque_geometry_api.rs` — 3
  GPU-gated tests pinning (1) `new` ≡ `new_with_geometry(STANDARD_4K)`
  byte-identity, (2) IPHONE_14_PRO vs PANEL_65IN_4K JOD must differ
  (proves geometry is actually consumed), (3) Full / Auto succeed +
  Strip / Tile surface `ModeUnsupported` on the new constructor.
  (this commit)
- **zensim-gpu divergence documented**: `ZensimOpaque` deliberately
  does NOT receive a `new_with_geometry` companion API. The
  underlying `zensim_gpu::Zensim::<R>` is a feature-based metric
  (228 / 300 / 372-d vector) with no `DisplayGeometry` / PPD
  threading — pyramid depth + filter weights are purely data-driven
  and don't depend on viewing conditions. Comment block in
  `crates/zensim-gpu/src/opaque.rs` directs callers wanting
  display-aware scoring to `CvvdpOpaque::new_with_geometry`.
  (this commit)

### zenmetrics-api / 6 metric crates — feat: umbrella cached-ref + MemoryMode unification (Phases 1, 2A, 2B, 2C + task #51) — 2026-05-26

The cached-ref + strip-mode perf wins shipped per-crate over the
preceding weeks but were unreachable from any sweep on master
because the umbrella `Metric::compute_srgb_u8` is one-shot. This
work plumbs the per-crate APIs through the umbrella and wires the
sweep cache to use them transparently.

- **`zenmetrics-api`** — `MemoryMode { Auto, Full, Strip, Tile }` +
  `CachedRefStripPolicy { Auto, RefFull, BothStripped }` umbrella
  enums with per-crate `From` conversions
  (`crates/zenmetrics-api/src/memory_mode.rs`). New
  `Metric::new_with_memory_mode(kind, backend, w, h, params, mode)`
  constructor + four cached-ref methods on `Metric`:
  `set_reference_srgb_u8`, `compute_with_cached_reference_srgb_u8`,
  `clear_reference`, `has_cached_reference`. (`e0ae180`, `d4e1572`,
  `d25124a`)
- **`butter/ssim2/dssim-gpu`** — added cached-ref opaque
  methods (`set_reference_srgb_u8`, `compute_with_cached_reference_srgb_u8`,
  `clear_reference`, `has_cached_reference`) on `*Inner` traits +
  `*Opaque` types. Each wraps the existing typed-pipeline
  cached-ref. butter strip-mode rejects `set_reference` (single-
  resolution pair-only); umbrella callers get
  `Error::StripModeUnsupported` and fall back to one-shot. (`d25124a`)
- **`iwssim-gpu`** — new opaque cached-ref methods that dispatch
  `set_rgb_reference_stripped` / `compute_rgb_with_reference_stripped`
  in Strip mode, else host-side sRGB→gray BT.601 +
  `set_reference` / `compute_with_reference` in Full mode.
  `rgb_u8_to_gray_bt601` made `pub(crate)`. (`d4e1572`)
- **`zensim-gpu`** — new opaque
  `compute_with_cached_reference_score_srgb_u8` wrapper that pairs
  the existing `compute_with_reference_srgb_u8` Vec output with
  the profile-mode `score_from_profile_vec` conversion to return a
  uniform `Score`. (`d4e1572`)
- **`zen-metrics-cli`** — `MetricCache::compute_umbrella` now keys
  on the source `(pointer, len)` fingerprint. On a cache miss it
  calls `set_reference_srgb_u8` then `compute_with_cached_reference_srgb_u8`;
  on `Error::StripModeUnsupported` from set_reference it marks the
  slot `set_reference_unsupported` and falls back to one-shot
  `compute_srgb_u8` for the slot's lifetime. Sweep call sites in
  `sweep/run.rs` don't change — the warm-ref optimization happens
  transparently inside the cache. (`8449296`)
- **Per-crate feature gating** — runtime features (`cuda`, `wgpu`,
  `hip`, `cpu`) now imply `cubecl-types` across all 6 metric
  crates. Fixes the lib-level `E0433 cannot find strip in the
  crate root` error when building butteraugli-gpu with `--features
  cuda` alone, and the test/example/doctest compile failures for
  ssim2/dssim/zensim. (`e28ddd7`)
- **Live VRAM probe across all 6 crates (task #51)** — replicated
  iwssim-gpu's `live_vram_probe_bytes` (cached
  `nvidia-smi --query-gpu=memory.free` with 10% headroom) into
  butter/ssim2/dssim/cvvdp/zensim-gpu's `memory_mode.rs`.
  `vram_cap_bytes()` now consults env → live probe → 8 GB default.
  Per-crate code duplication preferred over a shared crate (per
  the shared-traits planning verdict — `MemoryMode` enum divergences
  in the per-crate variants make hoisting net-LOC-positive).
  (`e6660cc`)
- **`butteraugli-gpu`** — `dimensions()` returns the LOGICAL image
  dims (`(width, image_h)`), not the strip-allocation dims
  (`(width, height)` where height = `body_h_max + 2*halo_h`).
  Fixes the umbrella `dispatch_butter` + `kind_roundtrip` tests.
  (`3f9210b`)
- **Tests** —
  `crates/zenmetrics-api/tests/cached_ref_parity.rs` (16 tests):
  single-pair + N-distortion cached-ref-vs-one-shot parity across
  all 6 metrics + `has_cached_reference` roundtrips for the 4
  metrics with explicit accessors. Bit-identical for zensim +
  iwssim; ≤1e-4 JOD for cvvdp/butter/dssim (Atomic<f32>
  reduction-order drift); ≤1e-3 for ssim2 (the 5e-5 atomic floor
  per task #52). Full umbrella suite is 29/29 passing. (`d25124a`,
  `7b78151`)

### cvvdp-gpu — fix: CSF `log_rho` axis extrapolation at high PPD (conformance Finding A) — 2026-05-26

- **`cvvdp_gpu::kernels::csf::interp1_rho_extrap`**: the inner CSF
  `log_rho` axis interp now linearly EXTRAPOLATES above its 64 cy/deg
  maximum (matching pycvvdp's `interp.get_interpolants_v1`), instead of
  flat-clamping to the endpoint. The conformance matrix's
  `iphone_14_pro` display (`pix_per_deg ≈ 159.6`, finest pyramid band
  ≈ 80 cy/deg — the only conformance display past the axis) was
  over-estimating achromatic CSF sensitivity by ~2× in the finest band,
  landing both cvvdp-cpu AND cvvdp-gpu up to 0.028 JOD low vs pycvvdp on
  JPEG content. Trigger was high spatial frequency, not high peak
  luminance. Bit-identical for interior queries (no change to the other
  8 displays / standard-4K 1e-4 parity). Conformance matrix: cpu
  274→279/279, gpu 271→276/279. Covers BOTH impls because the GPU
  uploads the host-computed `precompute_logs_row`. Closes
  `UPSTREAM_DIVERGENCES.md` row 8. (this commit)
- **Offline regression tests** for the fix (`kernels::csf::tests`, run by
  the default `cargo test -p cvvdp-gpu` — no GPU / no goldens fetch
  needed): (1) the three-regime `interp1_rho_extrap` contract on a
  synthetic axis (flat-clamp below, linear interp inside, linear
  extrapolate above); (2) **interior bit-identity vs a flat-clamp
  reference** across a 1000-point sweep + every interior knot — the
  unit-level proof that the fix moved zero in-axis queries (the
  "no regression on the 248 non-iphone cells" guarantee); (3) the
  high-PPD guard at the iphone band-0 rho (≈79.8 cy/deg) on the real
  `LOG_RHO_AXIS` + channel-A LUT, asserting the extrapolated
  sensitivity falls strictly below the old clamp. Fails if anyone
  reverts the rho axis to flat-clamp. (this commit)

### cvvdp-cpu — brute-force SIMD-vs-scalar kernel equivalence harness — 2026-05-26

- **`tests/simd_equivalence.rs`**: brute-force per-element comparison of
  every cvvdp-cpu SIMD kernel against its scalar reference across ≥1000
  randomized inputs + adversarial edge cases per kernel, measuring the
  ULP / relative-error envelope. Catches per-element divergences the
  end-to-end 1e-4 JOD pool would mask. Gated behind a new `__simd_equiv_test`
  cargo feature; OFF by default. Findings: sigma3 13-tap blur + pyramid
  5-tap reduce are BIT-IDENTICAL to scalar (0 ULP); pyramid expand ≤2 ULP
  (subnormal tail); vexp/vlog/vpow/safe_pow approximations measured at
  14/3/40/92 max ULP — all inside magetypes' ~128 ULP / ~1e-5 rel budget,
  committed as regression gates. Doc: `crates/cvvdp-cpu/docs/SIMD_EQUIVALENCE.md`.
  (`7beafa0`)
- **`lib.rs::__simd_equiv_test_api`** (feature-gated `#[doc(hidden)]`):
  thin `pub fn` visibility shim re-exporting the `pub(crate)` SIMD kernel
  entry points for the external test crate. No logic change; no production
  path enables the feature. (`7beafa0`)

### cvvdp-conformance — NEW crate: multi-impl conformance matrix vs pycvvdp v0.5.4 — 2026-05-26

- **New dev-crate `cvvdp-conformance`** validating BOTH `cvvdp-cpu`
  and `cvvdp-gpu` (as black boxes via public APIs) against the
  canonical pycvvdp v0.5.4 reference across a **9 display × 31
  situation = 279-cell** matrix. Replaces the thin single-image
  `1e-4 JOD` gate that could mask per-display / per-content
  divergences. Gated behind the `conformance-goldens` cargo feature
  (mirrors cvvdp-gpu's `parity-goldens`) so offline `cargo test` stays
  green; the matrix runs explicitly with a real GPU + R2 goldens.
- **Result**: cpu 274/279, gpu 271/279 within 1e-3 JOD; median delta
  ~2-6e-6 (bit-parity); cpu/gpu agree to ≤1.2e-3. Two documented
  divergences surfaced (NOT silently passed): (A) shared cpu+gpu
  parity gap ≤0.028 JOD on the `iphone_14_pro` Y_peak=1025 nit display
  for JPEG content (high-peak-luminance CSF/masking regime — display
  params/EOTF/CSF-axis all ruled out as identical to pycvvdp); (B) 3
  GPU-only marginal cells ≤0.0014 JOD at the perceptibility floor on
  extreme content (GPU float reduction-order). Both root-caused in
  `crates/cvvdp-cpu/docs/CVVDP_CONFORMANCE.md` + `UPSTREAM_DIVERGENCES.md`.
- **Goldens**: pycvvdp v0.5.4, 279 cells, R2
  `s3://coefficient/cvvdp-goldens/conformance-v1/`
  (sha256 `8f7d69dc…`). Reproducible via
  `scripts/cvvdp_goldens/build_conformance_goldens.py`.
- TSV: `benchmarks/cvvdp_conformance_matrix_2026-05-26.tsv`.

### cvvdp-gpu — perf narrative + HMD geometry + heatmap + params verification — 2026-05-25

- **Performance numbers corrected**: lib.rs "How we compare" section
  updated from stale tick-175 numbers (62 / 34 ns/px = 4.4× / 2.4×
  slower) to current measurements (2.1 / 1.3 ns/px = **6.5× / 10.7×
  faster** than pycvvdp v0.5.4 CUDA). Measured on RTX 5070 at 12 MP.
  Reproducer: `cargo run --release --example time_12mp -p cvvdp-gpu
  --features cuda,cubecl-types --no-default-features`. (`1280571a`)
- **HMD geometry**: `DisplayGeometry::by_name()` now handles
  `fov_diagonal` entries via `from_fov_diagonal()`. All 26 upstream
  presets (including `standard_hmd` and `htc_vive_pro`) load both
  model and geometry — previously the two HMD presets returned
  `None` for geometry. (`1280571a`)
- **Heatmap rendering**: new `heatmap` module with `HeatmapMode`
  enum (`Threshold` / `SupraThreshold` / `Raw`) and
  `render_heatmap()`. Ports upstream pycvvdp's
  `visualize_diff_map.py` colormaps — threshold (5-color, 0–0.1 JOD),
  supra-threshold (3-color, 0–0.3 JOD), raw (grayscale). Accepts
  optional context image for luminance-modulated backdrop.
  (`1280571a`)
- **Params verification**: vendored `cvvdp_parameters.json` under
  `data/` and added `tests/params_match_upstream_json.rs` asserting
  all kernel consts (MASK_P/Q/C, D_MAX, BETA_*, JOD_A/EXP,
  IMAGE_INT, BASEBAND_W) match the upstream JSON to 1e-5. Catches
  calibration drift on upstream version bumps. (`1280571a`)

### cvvdp-cpu — SIMD 13-tap σ=3 Gaussian blur (Chunk 1 of SIMD plan) — 2026-05-26

`src/simd_pyramid.rs` extended with the 13-tap σ=3 separable Gaussian
blur (`gaussian_blur_sigma3_simd`) — replacement for the upstream
`cvvdp_gpu::kernels::masking::gaussian_blur_sigma3` (which allocated
its h-pass buffer internally). Same six-tier dispatch as Chunk 2's
5-tap kernels: v4x AVX-512 16-wide, v4 AVX2 8-wide, v3 SSE4 4-wide,
NEON, WASM SIMD128, scalar.

Paired A/B against current master (`0fc2eb2b`, Chunks 2/3/4/5 +
Chunk 4 buffer recycling already shipped), `RAYON_NUM_THREADS=8`, no
`target-cpu=native`, 5 alternating rounds × 30 iters, full
`Cvvdp::score` wall:

| size      | BL med-o-med | PO med-o-med | Δmed%   | Δbest%  | speedup |
|-----------|-------------:|-------------:|--------:|--------:|--------:|
| 256×256   |     7.44 ms  |     4.24 ms  | -43.1 % | -43.1 % | 1.76 ×  |
| 512×512   |    30.99 ms  |    17.57 ms  | -43.3 % | -45.0 % | 1.76 ×  |
| 1024×1024 |   158.40 ms  |    82.88 ms  | -47.7 % | -48.6 % | 1.91 ×  |
| 2048×2048 |   639.29 ms  |   360.38 ms  | -43.6 % | -44.2 % | 1.77 ×  |

The win is two-fold: the upstream `gaussian_blur_sigma3` is pure
scalar (LLVM does not vectorize the 13-tap dot because of the
inner `reflect_idx_for_blur` branch) AND allocates 2× `w*h`
`Vec<f32>` on every call (3 channels × non-baseband bands per
encode). The SIMD entry vectorizes the boundary-clean interior
(99 % of cells at 1024²) and threads caller-owned scratch, so both
costs are removed at once. The 1024² wall dropped 158 → 83 ms.
Reproducer: build `examples/time_masking_paired_ab` at master and
at this commit; run each with `--iters 30` under
`RAYON_NUM_THREADS=8`.

- `masking::mult_mutual_band_into` Step 2 rewired to call the new
  SIMD entry; uses `pu_scratch` as the h-pass scratch (shared across
  the 3 channels) and `term_a/rg/vy` as blur output buffers (free
  until Step 3). Master's Chunk 4 buffer recycling already owns the
  per-band Scratch; this chunk swaps the per-band upstream blur (which
  still allocated 2× `w*h` Vec<f32> per channel internally) for the
  caller-scratch SIMD entry.
- 5 new SIMD parity tests at 1e-5 abs (full-pipeline, h-only,
  v-only, DC-preservation, scratch-reuse safety).
- 1e-4 JOD parity floor PRESERVED
  (`standard_4k_path_still_at_parity_against_host_scalar` green).

### cvvdp-cpu — SIMD 5-tap pyramid reduce/expand (Chunk 2 of SIMD plan) — 2026-05-25

`src/simd_pyramid.rs` (new) ports the 5-tap separable Gaussian inner
loops of `gausspyr_reduce` / `gausspyr_expand` to safe SIMD via
archmage `#[magetypes]` + `incant!` dispatch. Six tiers covered:
AVX-512 (`v4x` — 16-wide), AVX2 (`v4` — 8-wide), SSE4 (`v3` — 4-wide),
NEON, WASM SIMD128, scalar. Bit-near-identical output to the scalar
reference (1e-5 max abs delta, vs the pre-existing 1e-6 fixture
tolerance — far below the 1e-3 JOD floor and 1e-4 golden tolerance).

Wall-time impact at 1024² is ±5 % vs baseline (small + within noise) —
see `benchmarks/cvvdp_cpu_simd_pyramid_2026-05-25.meta` for the honest
attribution. The chunk's structural ceiling is the pyramid's
share-of-wall (~23 %, not the 56 % the flamegraph extrapolated). What
ships is the SIMD scaffolding: the inner kernels are now fully
SIMD-vectorized (AVX-512 `zmm` registers, `vmulps`/`vaddps` confirmed
in disasm) and the runtime CPU detection routes correctly via
archmage. Subsequent Chunks 1 + 3 + 4 + 5 stack on top.

- New `simd_pyramid` module (`pub(crate)`), 6-tier `#[magetypes]` SIMD.
- New `avx512` cargo feature, OPT-IN (default-off pending v4 / v4x
  variants on the simd_math module landing — see meta for details).
- New deps: `archmage 0.9.23` + `magetypes 0.9.23` (workspace).
- 5 new SIMD-parity unit tests (random + DC-preservation fixtures).
- `pyramid::gausspyr_reduce` / `gausspyr_expand` rewired to call SIMD
  inner passes; boundary patches stay scalar for FMA-grouping parity.
- Pyramid scalar-parity test tolerance: 1e-6 → 1e-5.

### cvvdp-cpu — SIMD Chunk 4: buffer recycling — 2026-05-25

Per-call allocations cut ~90% via a persistent `Scratch` that owns the
Weber pyramids + per-band workspaces (B7a/b buffer-recycling pattern, NOT
TLS pool — the butteraugli B7c memo proved TLS regresses). Allocs
535→58, mem 331MB→33MB per call at 1024². Wall: smooth -26/-31ms,
photo -27/-56ms (-27%), screenshot -38/-35ms (cold/warm). 1e-4 JOD
parity preserved; allocator-only change is numerically inert. 45/45
tests pass. `93924e42`.

### cvvdp-cpu — SIMD Chunk 3: vectorized pow/exp/log + masking rewire — 2026-05-25

New `simd_math` module (`safe_pow_with_offset_into`, `vexp_into`,
`vlog_into`, `vpow_into`) built on archmage + magetypes
`pow_midp_unchecked` (≈128 ULP / 1e-5 rel; inputs pre-offset by
`SAFE_EPS = 1e-5` so the unchecked positive-input path is sound). The
existing magetypes transcendentals were used directly — no hand-rolled
vpow. `masking.rs` powfs rewired through `simd_math`: -8.92ms at 1024²
(best-of-medians), -37.91ms at 2048². 1e-4 JOD parity preserved. 50/50
tests (45 + 5 accuracy). `da5ba743` (helpers) + `ea7945a3` (rewire).

### cvvdp-cpu — SIMD Chunk 5: CSF apply — HONEST-STOP — 2026-05-25

SIMD CSF apply attempted (`vexp_into` on the per-band sensitivity curve)
but REGRESSES +6 to +31% wall at 256-1024². Root cause: LLVM already
stream-fuses the scalar `apply_csf_row_per_pixel` (sensitivity stays in
an xmm register between the `exp` call and the consumer multiply); any
design that materializes `s[i]` into a buffer incurs round-trip traffic
exceeding the SIMD-exp saving. The fully-fused SIMD band loop needs
AVX2 `gather` for the 32-entry LUT bracket reads, which magetypes
doesn't expose. Persistent rayon pool also skipped (rayon's global pool
is already persistent; per-`Cvvdp` `ThreadPool::install` adds overhead
for zero gain, per butteraugli B7c). CSF SIMD retained as
`#[allow(dead_code)]` documentation. `f01c99d2` (attempt) + `03037b0e`
(honest-stop revert). 1e-4 JOD parity preserved, 58/58 tests.

<!-- NOTE: SIMD Chunk 1 (σ=3 13-tap blur) was re-verified against
     current master on 2026-05-26 and LANDED. The rebase conflict was
     CHANGELOG-only (the masking.rs edit applied cleanly — master still
     used the old allocating blur in that region). Re-benched vs master
     (not the stale 71bd498f baseline): -43 to -49 % wall at 256²-2048²,
     1e-4 JOD parity preserved. Entry above; bench
     `benchmarks/cvvdp_cpu_simd_sigma3_2026-05-26.{tsv,meta}`. -->

### cvvdp-gpu — GPU kernel-side EOTF + Primaries dispatch — 2026-05-25

Closes the GPU-side counterpart of the host-scalar display dispatch
that landed earlier in this Unreleased section. The previous GPU
fast path hardcoded sRGB + BT.709 inside `srgb_to_dkl_kernel` /
`linear_rgb_planes_to_dkl_kernel`; HDR (PQ / HLG) and wide-gamut
(BT.2020 / Display P3) presets only worked through `host_scalar`.

- **EOTFs wired on GPU**: `Srgb`, `Pq`, `Hlg`, `Linear`, `Bt1886`,
  `Gamma(g)`. `srgb_to_dkl_kernel` now takes `eotf_tag` (u32 — see
  `kernels::color::eotf_tag` constants), `gamma_exp` (f32 — the
  exponent payload for `Eotf::Gamma`), and `hlg_gamma` (f32 —
  precomputed system gamma from
  `params::hlg_system_gamma(y_peak, e_ambient_lux)`). The new
  `#[cube]` helper `apply_eotf_branch` mirrors the host
  `Eotf::forward` dispatch branch-for-branch (chained `if/else` —
  cubecl 0.10 doesn't support early `return` in `#[cube]` bodies).
  HLG OOTF is per-pixel using BT.2100 luma coefficients on the RGB
  triple (`hlg_ootf` `#[cube]` helper).
- **Primaries wired on GPU**: `Bt709` (default), `Bt2020`,
  `DisplayP3`, `DciP3`. Both `srgb_to_dkl_kernel` and
  `linear_rgb_planes_to_dkl_kernel` now take the 9 RGB→DKL matrix
  entries as runtime scalars instead of inlining BT.709 constants
  — one kernel binary serves every primaries variant. LLVM still
  folds the linear combo when the values are constant across the
  launch, so the per-pixel ALU cost is unchanged.
- **sRGB+BT.709 stays bit-identical** — the `tag=0` branch in the
  new kernel takes the LUT path with the same constants and
  matmul order as the pre-dispatch kernel. Pinned by
  `tests/color_kernel::srgb_to_dkl_kernel_matches_host_scalar`
  (existing test, still passes unchanged).
- **New parity tests**:
  `tests/color_kernel_display_dispatch.rs` checks GPU-vs-host_scalar
  agreement across 12 (EOTF × primaries × peak-luminance) combos:
  sRGB+{BT.709, BT.2020, DisplayP3}, PQ+BT.2020 at {1500, 3000}
  cd/m², HLG+BT.2020, Linear+BT.709, Bt1886, Gamma(2.2), Gamma(1.8),
  iPhone 14 Pro SDR + iPhone 14 Pro HDR. Tolerances scale with
  `y_peak` (HDR PQ at 3000 cd/m² gets 0.1 abs = 33 ppm relative);
  the chained `powf` in PQ / HLG accumulates ~3-4 ULPs of
  ordering noise across the f32 chain.
- **HDR + iPhone parity vs pycvvdp v0.5.4** measured on the real
  GPU path (cubecl-cuda on RTX 5070 / CUDA 13.2, native build —
  no docker) against the 13-pair `/tmp/cvvdp-display-eval/` bundle:
  - `standard_4k`: n=13, mean abs_diff = 0.0370 JOD, median = 0.0016,
    max = 0.3906 (single outlier on `photo_dark_noise_heavy`).
  - `iphone_14_pro`: n=13, mean = 0.0303, median = 0.0011,
    max = 0.3307 (same outlier).
  Both displays meet the mean<0.10 gate; the max outlier shows up
  on both displays at the same pair, indicating a content-specific
  drift rather than a display-dispatch defect. The GPU numbers are
  within ~0.0002 JOD of the host_scalar path measured the same day,
  matching the GPU↔scalar pin in
  `tests/color_kernel_display_dispatch.rs`. Full breakdown at
  `benchmarks/cvvdp_iphone14_parity_2026-05-25.tsv` + `.meta`.
  Reproducers (both written for the same TSV layout):
  `cargo run -p cvvdp-gpu --release --example parity_iphone_eval_gpu --features cuda,cubecl-types --no-default-features`
  (GPU, writes `parity_v2_gpu.tsv`) and
  `cargo run -p cvvdp-gpu --release --example parity_iphone_eval`
  (host_scalar fallback for hosts without CUDA).

Commit: `f8bf2729` (this work). Docs updated in
`crates/cvvdp-gpu/docs/DISPLAY_SPECS.md` Scope matrix.

### cvvdp-gpu — full display-spec parity (host-scalar) — 2026-05-25

`DisplayModel` now carries first-class `eotf`, `primaries`,
`e_ambient_lux`, and `k_refl` fields, plus the constructor
`DisplayModel::new(y_peak, contrast, e_ambient_lux, k_refl, eotf,
primaries)` matching upstream's `vvdp_display_photo_eotf.__init__`.
`STANDARD_4K` stays a `pub const` and is bit-identical to the
historical (3-field) shape — every existing parity test passes
without modification.

- EOTFs supported: `Srgb` (default), `Pq`, `Hlg`, `Linear`,
  `Bt1886`, `Gamma(f32)`. Reference-value tests verify against
  SMPTE ST 2084 (`PQ(0.5) ≈ 92.25 cd/m²`, `PQ(1.0) = 10000`),
  BT.2100-1 Table 5 (`HLG(0.5) = 1/12`), and the IEC 61966-2-1
  sRGB seam continuity at `V = 0.04045`. The `Eotf::forward`
  dispatcher mirrors pycvvdp's `forward` branch-for-branch.
- Primaries supported: `Bt709` (default), `Bt2020`, `DisplayP3`,
  `DciP3` (today an alias for `DisplayP3` — no theatrical DCI
  preset upstream). Per-primaries `LinRGB → DKL` matrices
  pre-computed at f64 from upstream's
  `LMS2006_to_DKLd65 @ XYZ_to_LMS2006 @ RGB_to_XYZ` chain;
  `Primaries::Bt709.linear_rgb_to_dkl()` is bit-identical to the
  pinned `SRGB_LINEAR_TO_DKL` const.
- 26 preset registry: `DisplayModel::by_name(name)` and
  `DisplayGeometry::by_name(name)` load every preset from
  pycvvdp's `display_models.json` (vendored under
  `crates/cvvdp-gpu/data/`; MIT-licensed, license + attribution
  preserved in `data/UPSTREAM_LICENSE_MIT.txt` + `data/THIRD_PARTY.md`).
  EOTF + primaries derived from each preset's `colorspace` field
  via the also-vendored `color_spaces.json`. Complements the
  per-preset `pub const`s shipped in `cvvdp-cpu 0.1.0` by
  exposing the same set through a string-keyed lookup that
  matches pycvvdp's `vvdp_display_photometry.load(display_name)`
  semantic.
- New scalar entry points in `kernels::color`:
  `display_byte_to_dkl_scalar(r, g, b, display)` and
  `display_linear_rgb_to_dkl_scalar(r, g, b, display)` dispatch
  on `display.eotf` and `display.primaries`. Bit-identical to
  `srgb_byte_to_dkl_scalar` when given `STANDARD_4K`. HLG path
  computes the per-pixel OOTF using BT.2100 luma coefficients
  and the dynamic system gamma.
- `host_scalar::predict_jod_still_3ch_capped` now calls
  `display_byte_to_dkl_scalar`, so non-sRGB EOTF / non-BT.709
  primaries score correctly through the host-scalar entry point.
  All 20 `predict_jod_invariants` tests pass (pycvvdp goldens
  at 128×128 / 720×1280 / 1024×1024 fixtures, ≤ 0.005 JOD).
- 29 new tests cover: spec EOTF reference values, monotonicity,
  seam continuity, per-primaries matrix divergence on saturated
  colours, every-preset-loads, byte-path parity with
  `srgb_byte_to_dkl_scalar` under `STANDARD_4K`, and the new
  `DisplayModel::new` constructor matching `STANDARD_4K`
  bit-for-bit. All in `tests/eotf_primaries_invariants.rs` (21)
  + `src/presets.rs::tests` (8).
- Scope: this release wires the display dispatch through the
  host-scalar entry points. The GPU fast path (`Cvvdp::score`,
  `compute_dkl_jod`, the kernel uploads) still reads
  `y_peak`/`y_black`/`y_refl` and assumes sRGB+BT.709 — a
  follow-up tick will route the kernels through the new fields.
  Until then HDR / wide-gamut callers should convert to
  linear-BT.709 host-side and use `score_from_linear_planes`.
  Documented in `crates/cvvdp-gpu/docs/DISPLAY_SPECS.md` and in
  the README "Display models" section.

Commits: `2b44cb5` (params surface), `b73058e` (host-scalar
dispatch), `156f92b` / `1ef5c6e` (preset registry + vendored
JSON — rebased), `c5fb731` / `febd722` (21 reference-value
invariant tests — rebased). Docs + CHANGELOG in this commit.

### cvvdp-cpu 0.1.0 — upstream-parity (EOTF + Primaries + named presets) — 2026-05-25

Closes the cvvdp-cpu → gfxdisp/ColorVideoVDP parity gap on display
modelling. The v0.0.1 path tracked cvvdp-gpu's `host_scalar` reference
which itself hard-coded sRGB + BT.709; v0.1.0 wires the full
`display.eotf` + `display.primaries` dispatch through the color stage
and ships 23 named DisplayModel + 15 paired DisplayGeometry preset
constants matching upstream `display_models.json`.

- **Upstream parity audit shipped** (`414a16f`) — finding-only doc at
  `crates/cvvdp-cpu/docs/UPSTREAM_PARITY_AUDIT.md` enumerating 26
  display presets, EOTF / Primaries plumbing, CSF / masking / pooling
  params with line refs on both sides.
- **EOTF + Primaries dispatch in cvvdp-cpu::color** (`f0c0323`) —
  `srgb_to_dkl_planar` and `linear_planes_to_dkl_planar` now honor
  `display.eotf ∈ {Srgb, Pq, Hlg, Linear, Bt1886, Gamma(g)}` and
  `display.primaries ∈ {Bt709, Bt2020, DisplayP3, DciP3}`. Fast path
  `(Srgb, Bt709)` preserves v0.0.1 bit-identical numerics; any other
  combination routes through `cvvdp_gpu::kernels::color::display_byte_to_dkl_scalar`.
  Was a silent downcast before this commit.
- **Named DisplayModel + DisplayGeometry presets** (`1a89804`) — 23
  DisplayModel consts (STANDARD_HDR_PQ, STANDARD_HDR_HLG,
  STANDARD_FHD, STANDARD_PHONE, SDR_4K_30, SDR_FHD_24, HTC_VIVE_PRO,
  IPHONE_12_PRO, IPHONE_14_PRO, IPHONE_14_PRO_HDR, IPAD_PRO_12_9,
  MACBOOK_PRO_16, LG_OLED_2017_SDR, LG_OLED_2017_HDR, EIZO_CG3146,
  HDR_PQ_4KNIT, HDR_PQ_2KNIT, HDR_PQ_1KNIT, LG_OLED_2026_HDR_PQ,
  plus 4 STANDARD_HDR_* linear variants). 15 DisplayGeometry consts.
  4 geometry constructors: `new`, `from_inches`, `from_meters_diagonal`,
  `from_fov_diagonal` (the last derives an equivalent
  `diagonal_inches` from a VR-style FOV). 4 size getters:
  `display_width_m / display_height_m / display_width_deg / display_height_deg`.
- **Upstream parity tests** (`8cf4add`) — 18 new tests in
  `tests/upstream_parity_extended.rs` covering geometry constructors,
  every preset's fields, PPD round-trips against pycvvdp v0.5.4
  reference Python calc (≤1e-4 rel err), EOTF round-trips (Pq, Hlg,
  Linear, Gamma, Bt1886 — all yield JOD ≈ 10 on identical input),
  Primaries round-trips (DisplayP3, DciP3), cross-primaries chroma
  sensitivity, and a regression gate pinning STANDARD_4K against
  `host_scalar::predict_jod_still_3ch` at 1e-4 JOD.
- **Existing pycvvdp v0.5.4 parity preserved** — none of the chunks
  widen the 1e-4 JOD tolerance the historical tests pinned. The
  STANDARD_4K (sRGB + BT.709 + 200 cd/m² + 250 lux ambient) golden
  is bit-identical to v0.0.1 numerics by routing through the fast
  path that uses the same matrix `Primaries::Bt709.linear_rgb_to_dkl()`
  returns.

Bumped to v0.1.0 (minor — additive new features, no breaking
changes to the v0.0.1 API surface). Workspace dep version pin
bumped to match.

Total tests: 45 (was 27 at end of v0.0.1 perf milestone). cargo
clippy -p cvvdp-cpu --no-deps -- -D warnings clean. cargo fmt
clean. cargo check --no-default-features --features alloc clean.

Documentation: `crates/cvvdp-cpu/docs/UPSTREAM_PARITY_AUDIT.md`
(per-row enumeration of parity status) +
`crates/cvvdp-cpu/docs/UPSTREAM_DIVERGENCES.md` (RESOLVED + DIVERGES
log for v0.1.0). Documented out-of-scope items (temporal channels,
foveation, exposure, PU21 encoding, non-`weber_fixed_size` CSF LUTs,
runtime parameter override, color spaces beyond BT.709/BT.2020/P3)
with paths-to-close for each.

### cvvdp-cpu — new crate, pure-Rust CPU port — 2026-05-24

New crate `cvvdp-cpu` at `crates/cvvdp-cpu/` ships v0.0.1. Maximally
optimised pure-Rust port of ColorVideoVDP still-image scoring.
Designed as a drop-in for JPEG XL encoder's iterative quantization
loop where the GPU backend's host-to-device upload latency exceeds
CPU compute time.

- Matches cvvdp-gpu's `host_scalar` reference (the f32 contract)
  within 1e-4 JOD on synthetic fixtures 16² through 512² (4 parity
  tests pass).
- API: `Cvvdp::{new, score, score_with_diffmap, warm_reference,
  score_with_warm_ref, score_with_warm_ref_diffmap,
  score_from_linear_planes, score_from_linear_planes_with_diffmap,
  score_pixels}` (last gated on `pixels` feature).
- Per-pixel diffmap: bilinear-upsampled per-band masked error
  combined via `BETA_CH` Minkowski norm. 6 invariant tests:
  identical → zero, non-negative, monotone-in-distortion,
  correlates with JOD, spatially localizes to distorted region,
  warm-ref matches cold path.
- Perf at 1024² photo (single-thread / 8t cold/warm):
  `222 ms / 215 ms` (down from `988 ms` baseline scalar). Target
  floor is `< 50 ms`; remaining gap is in the per-pixel `powf`
  calls inside the masking inner loop (queued for SIMD).
- Features: `std` (default), `alloc`, `parallel` (default, rayon
  per-band + per-channel + REF/DIST), `pixels` (zenpixels
  integration).
- CI: added to existing `compile` matrix (linux/windows/macos/arm)
  + `i686-cross` job. No new workflow file.

Initial commit: `da81694` (parity + diffmap + warm).
Perf milestone: this commit.

### Auto-fallback contract — `resolve_auto` cross-crate audit — 2026-05-22

`MemoryMode::Auto`'s "2-pass fallback" semantic is now exercised by
identical cross-crate contract tests so future regressions surface
at `cargo test` rather than at production OOM.

- **iwssim-gpu**: `resolve_auto` now uses the canonical
  Full-fits-else-try-Strip shape with an explicit `MIN_NATIVE_DIM`
  guard on the Strip branch (matches the floor enforced by
  `Iwssim::new_strip_with_halo`). Behaviour preserved at large image
  sizes; small-image overcap surfaces `TooBigForFull` honestly rather
  than producing a Strip that the constructor would reject.
- **ssim2-gpu**: `resolve_auto` now auto-sizes the strip body via a
  new `auto_strip_body_for` helper instead of always using
  `STRIP_H_BODY_DEFAULT`. When the default body doesn't fit the cap
  but a smaller aligned body does, Auto now picks the smaller body
  rather than surfacing `TooBigForFull`. `Ssim2::new_with_memory_mode`
  also routes `MemoryMode::Strip { h_body: None }` through the
  auto-sizer for the same coverage.
- **butteraugli-gpu, dssim-gpu, cvvdp-gpu, zensim-gpu**: no source
  changes — these crates' resolvers already match the canonical
  contract. Tests added to pin behaviour.

New tests (40 total, all host-side):

- `crates/iwssim-gpu/tests/auto_fallback.rs` — 8 tests
- `crates/butteraugli-gpu/tests/auto_fallback.rs` — 7 tests
- `crates/dssim-gpu/tests/auto_fallback.rs` — 6 tests
- `crates/ssim2-gpu/tests/auto_fallback.rs` — 7 tests
- `crates/cvvdp-gpu/tests/auto_fallback.rs` — 6 tests (pins the
  "no silent capped-Strip fallback" contract — capping changes JOD)
- `crates/zensim-gpu/tests/auto_fallback.rs` — 6 tests (no-Strip
  crate; pins that Auto only returns Full or TooBigForFull)

### cvvdp-gpu — capped-pyramid `MemoryMode::Strip { capped_levels: Some(k) }` + `predict_jod_still_3ch_capped` — 2026-05-22

Adds an opt-in pyramid-depth cap for cvvdp-gpu that lets callers
trade some JOD fidelity for a smaller σ=3 PU-blur halo, paving the
way for a future strip walker. **Cap=8 ships as the deepest level
that fits the canonical ≤ 0.005 JOD pycvvdp v0.5.4 manifest parity
gate** across all measured fixtures (max drift 5.85e-4 JOD on the
1280×720 offset fixture). Cap=7 fails on the 720×1280 fixture at
1.17e-2 drift and does not ship.

- `Cvvdp::new_with_geometry_and_cap(client, w, h, params, geom, capped_levels)`
  — typed-API entry point; `capped_levels = None` matches
  `new_with_geometry` byte-for-byte. `Some(0)` rejected.
- `MemoryMode::Strip { h_body, capped_levels: Option<u32> }` — unified
  variant; `Some(k)` routes to a Full pipeline with depth clamped to
  `min(k, natural_n_levels)`. The `h_body` single-pass-strip path
  remains unsupported (returns `Error::ModeUnsupported` when
  `capped_levels = None`).
- `host_scalar::predict_jod_still_3ch_capped(.., cap_levels: Option<usize>)`
  — host-scalar variant for host-only callers. The existing
  `predict_jod_still_3ch` now delegates with `cap_levels = None`,
  bit-identical to the original.
- `MemoryMode::Auto` does NOT auto-select capped depth — capping
  changes the metric value, so callers must opt in explicitly.

Memory in Full mode is unchanged (cap=8 vs cap=None saves only the
smallest few coarse-band scratch slots, kilobytes). Perf neutral
across 12 MP, 24 MP square, and 1024×8192 panorama (see
`benchmarks/cvvdp_capped_perf_2026-05-22.csv`). Capped-levels
fidelity sweep at `benchmarks/cvvdp_capped_levels_2026-05-22.csv`.
Doc: `crates/cvvdp-gpu/docs/STRIP_PROCESSING.md`.

Tests:

- `tests/capped_levels_parity.rs` — 4 host-scalar gates: cap=None
  matches uncapped, cap-above-natural clamps, cap=8 meets ≤ 0.005
  on all natural-depth-9 fixtures, and a pinned cap=7 720×1280
  failure.
- `tests/capped_levels_gpu_parity.rs` — 3 GPU gates: 12 MP and
  1024² cap=8 vs pycvvdp ≤ 0.005, cap=None matches uncapped at
  1024². All pass on cubecl-cuda. wgpu's pre-existing 65535-
  dispatch-grid limit (workgroup dim cap) still constrains 12 MP
  on that backend; unchanged by this work.
- `tests/memory_mode.rs` — `explicit_strip_with_cap_constructs` +
  the existing strip/tile/auto resolver tests.

73×91 odd-dim manifest parity holds at `|diff| = 0.0000 JOD` on
CUDA backend (verified via `compute_dkl_jod_matches_pycvvdp_at_73x91_odd`
+ warm-ref companion).

### butteraugli-gpu — multi-resolution strip walker + opaque-strip parity + dead-code cleanup — 2026-05-22

Adds the constant-VRAM analog of the CPU reference's default
multi-resolution path. `Butteraugli::new_multires_strip` walks the
full-res image in `body_h`-tall strips, and each strip pass drives a
synchronized half-res sibling whose body covers
`[body_top_full / 2, body_end_full.div_ceil(2))`. The half-res image
isn't decoded separately — its content comes from a 2× downsample of
the full-res linear-RGB strip planes, so the constant-VRAM property of
strip mode survives the multi-resolution upgrade.

- `Butteraugli::new_multires_strip(client, w, h, body_h)` constructor;
  enforces `body_h % 2 == 0` for half-res alignment.
- `Butteraugli::new_multires_with_memory_mode(client, w, h, mode)`
  routes `MemoryMode::Strip` to the new walker (was returning
  `StripModeUnsupported("new_multires")`).
- `strip::run_strip_pipeline_multires` orchestration:
  upload → downsample linear-RGB into half-res slab → run full-res
  pipeline → run half-res pipeline (lin-only entry, opsin already in
  the downsampled slab) → supersample-add half-res diffmap → reduce
  full-res body rows host-side.

11 new multires-strip parity tests (`tests/multires_strip.rs`) and 10
new opaque-shim strip routing tests (`tests/opaque_strip_parity.rs`),
covering 256² → 4000×3000 with even/uneven body sizes, with and without
`ButteraugliParams` overrides. All pass on cuda + wgpu within `1e-4`
relative tolerance (matches the single-res `strip_parity` band).

Bench (RTX 5070, body=256):

| Size | Whole | Strip | Speedup | Whole alloc | Strip alloc |
|---:|---:|---:|---:|---:|---:|
| 4 MP | 76.8 ms | 143.8 ms | 0.53× | 954 MB | 160 MB |
| 12 MP | 1736.6 ms | 153.6 ms | 11.31× | 2.79 GB | 320 MB |
| 24 MP | OOM | 275.9 ms | — | 5.86 GB | 492 MB |

At 4 MP the strip walker's per-strip kernel-launch overhead exceeds
its locality win (same shape as the single-res bench). At 12 MP the
whole-image multires path spills past L2 and strip is 11× faster. At
24 MP the whole-image path doesn't fit in 12 GB consumer VRAM. See
`benchmarks/butter_multires_strip_2026-05-22.csv`.

Dead-code cleanup: removed 10 helper methods made redundant by the
LUT-blur fast path and the fused `malta_triple` / `l2_asym_plus_l2` /
`l2_diff_write_3ch` landings — `launch_opsin`, `blur_plane`,
`blur_3ch_via`, `copy_plane`, `zero_plane`, `subtract_arrays`,
`malta_hf`, `malta_lf`, `l2_diff`, `l2_diff_asym`. Gated two
slow-path reduction constants behind `cfg(not(feature =
"fast-reduction"))`. -273 LOC, 1 dead-code warning eliminated, no
public API change.
### zensim-gpu — cached-reference + regime-aware opaque API — 2026-05-22

Plumbs the typed pipeline's `set_reference` / `compute_with_reference_vec`
cached path through the `ZensimOpaque` shim. Critical for sweep workloads
with many distortions per reference: a single `set_reference_srgb_u8` +
N×`compute_with_reference_srgb_u8` skips N-1 ref uploads and N-1 ref-pyramid
kernel launches.

- `ZensimOpaque::set_reference_srgb_u8(ref)` — upload + build the ref's
  XYB pyramid once.
- `ZensimOpaque::compute_with_reference_srgb_u8(dist) -> Vec<f64>` —
  regime-aware feature vector (228/300/372) against the cached ref;
  returns `Error::NoCachedReference` if `set_reference_srgb_u8` was
  never called.
- `ZensimOpaque::set_reference_pixels(r)` / `compute_with_reference_pixels(d)`
  — same flow from `PixelSlice` inputs (handles stride conversion).

Bench at 1024² × 10 distortions per reference (RTX 5070): cached path
saves ~38% on CUDA (55.23 → 34.08 ms) and ~40% on wgpu (3.61 → 2.15 s
on WSL2 Vulkan). See `benchmarks/zensim_cached_ref_2026-05-22.csv`.

12 new tests (`tests/opaque_regime.rs` + `tests/opaque_cached_ref.rs`):
6 regime-routing tests (Basic→228, Extended→300, WithIw→372, defaults,
fixed-array truncation contract, Extended slot-0..228 == Basic) and 6
cached-ref tests (pair-mode parity, multi-dist parity, NoCachedReference
error path, dim-mismatch rejection, regime respected on cached path,
re-set-reference overwrites). Total zensim-gpu test count: 28 → 40 per
backend (cuda / wgpu). All passing.

Ported from the abandoned `feat/acumen-gpu` branch. The `ViewingCondition`
re-export proposed in that branch is omitted — the upstream `zensim`
version we depend on has no `hdr_pu` module to re-export from.

### iwssim-gpu — strip-processing path for `Iwssim::new_strip` + `compute_gray_stripped` — 2026-05-22

Adds a memory-bounded strip-processing path to `iwssim-gpu` so production
sweep workers can run IW-SSIM at 24 MP without OOMing. The whole-image
constructor (`Iwssim::new`) pre-allocates ~2.43 GB of GPU working planes
at 24 MP (6000×4000 × 19 planes × 5 scales × 4 bytes); strip mode bounds
the working set to a single strip's allocation.

- `Iwssim::new_strip(client, image_w, image_h, h_body)` — allocate the
  pipeline for `h_body + 2 * STRIP_DEFAULT_HALO` rows per strip.
- `Iwssim::new_strip_with_halo(...)` — custom halo for smaller images.
- `Iwssim::compute_gray_stripped(ref, dis)` — two-pass driver. Pass 1
  builds LPs and accumulates per-strip Σ YᵀY into a host-side per-scale
  C_u matrix (then eigendecomposes ONCE globally); Pass 2 rebuilds LPs
  and scores with the global C_u uploaded. Two passes are needed
  because the C_u matrix is image-global per scale.
- `Error::CachedRefNotSupportedInStripMode` / `Error::NotStripMode` —
  typed error variants for "wrong mode" calls; previously
  `set_reference` on a strip-mode instance would silently produce
  wrong scores by uploading the full image into a strip-sized plane.
- Per-kernel body-range params (`py_start/py_end` on `cov_accum`,
  `cs_y_start/cs_y_end` on `weighted_sum`/`iw_sum`/`plain_sum`) so
  reductions exclude halo rows.

Memory tally at 1/4/12/24 MP (per-pipeline GPU working set, strip
body=1024 vs whole-image): 1MP 115→172 MB (strip overhead at 1MP since
the strip alloc is bigger than the image), 4MP 459→344 MB (1.33×),
12MP 1311→671 MB (1.95×), 24MP 2622→1007 MB (2.60×) — matches the
design budget in `crates/iwssim-gpu/docs/STRIP_PROCESSING.md`.

Heaptrack peaks at 12MP (host RSS): whole 1.93 GB / strip 1.20 GB
(1.61× reduction).

Wall time on RTX 5070 (CUDA, 8 iters, min): 1MP whole 5.6 / strip 9.0 ms,
4MP whole 14.9 / strip 34.3 ms, 12MP whole 86.2 / strip 106.5 ms,
24MP whole SKIP_OOM / strip 498.1 ms. Strip is slower per pair because
every strip rebuilds the LP pyramid for both ref and dis; the
OOM-avoidance and 1.61-2.6× memory reduction justify the cost on
production workers. The cached-reference strip path is deferred
follow-up work (see `Error::CachedRefNotSupportedInStripMode`).

f32 precision: strip-vs-whole rel drift is ~5e-4 at 1024² multi-strip
(the cov_accum's Σ over ~1M products + eigendecomp + Π|wmcs|^β
amplification puts this just at the f32 floor). The parity-lock test
against the Python reference runs at 5e-3 tolerance — strip drift
sits comfortably inside that band.

Tests: 20 strip-parity tests covering single-strip degenerate (256²),
multi-strip whole-vs-strip at 512² / 768² / 1024² / 1024×768 / body=512,
uneven last strip (640² / 896² with body=256), cross-tile-size parity
(body=256 vs body=512 at 1024² within 1.5e-3 rel), self-identity
(equal ref/dis → 1.0), constructor validation, and dedicated negative
tests for the new typed errors on every wrong-mode call shape. All
20 pass on CUDA + WGPU.

New bench example: `bench_strip_vs_whole` (`benchmarks/iwssim_strip_vs_whole_2026-05-22.csv`).
New memory-tally example: `strip_memory_tally`. New heaptrack driver:
`heaptrack_strip_12mp`.

Design doc: `crates/iwssim-gpu/docs/STRIP_PROCESSING.md`.

### ssim2-gpu — Phase 1 plane aliasing in `pipeline.rs::Scale` — 2026-05-22

Replaces 30 per-`Scale` blur-intermediate buffers
(`{sigma11,sigma22,sigma12,mu1,mu2}_v` + `*_t` — 5 plane names × 3
channels × 2 orientations) with a single shared rolling
`v_scratch: [Handle; 3]` + `t_scratch: [Handle; 3]` reused across all
5 blurs per (scale, channel). The two-pass IIR blur writes the
scratch then reads it, three sequential kernel launches; the next
blur overwrites the scratch from a fresh source. Same idiom the
batched pipeline (`pipeline_batch.rs::BatchScale`) already used —
this brings the unbatched pipeline in line.

Plane count per scale: 81 → 57 (−29.6%). Across the 6-scale
geometric pyramid: 108 → 76 scale-0-equivalent planes.

GPU memory tally (analytical, plane count × n × 4 bytes + 2 × n × 4
for sRGB u8 staging, dominated by the per-scale planes):

| size       | pre (master) | post (Phase 1) | saving       |
|------------|--------------|----------------|--------------|
| 1MP 1024²  | 0.430 GiB    | 0.305 GiB      | 0.125 GiB    |
| 4MP 2048²  | 1.718 GiB    | 1.218 GiB      | 0.500 GiB    |
| 12MP 4000×3000 | 4.916 GiB | 3.486 GiB     | 1.430 GiB    |
| 24MP 6000×4000 | 9.832 GiB | 6.972 GiB     | 2.860 GiB    |

The 24 MP delta (10.56 → 7.49 GB in SI units) is the production
target — sweep workers OOMing on 12 GB GPUs at 24 MP now fit with
margin. Phase 2 strip processing would cut further to ~1.4 GB at
24 MP but is multi-day work; design doc lands in this commit at
`crates/ssim2-gpu/docs/STRIP_PROCESSING.md` for the follow-up
session.

Host RSS via heaptrack at 12 MP (3 `compute` calls including warmup):
master 5.50 GB peak → Phase 1 3.97 GB peak (−27.8%). Cubecl mirrors
each GPU buffer with host-side staging, so the GPU saving directly
recovers host RSS too.

CUDA perf (RTX 5070, median of 10 runs each on a clean GPU,
synthetic 6-step pair):

| cell          | master (ms) | Phase 1 (ms) | Δ%    |
|---------------|-------------|---------------|-------|
| 1MP pair      | 5.759       | 5.731         | −0.49 |
| 1MP cached    | 3.544       | 3.439         | −2.95 |
| 4MP pair      | 12.806      | 13.011        | +1.60 |
| 4MP cached    | 7.830       | 7.851         | +0.27 |
| 12MP pair     | 32.232      | 31.204        | −3.19 |
| 12MP cached   | 19.719      | 19.298        | −2.13 |

All within ±3.5% — most negative (Phase 1 faster), consistent with
fewer allocator calls per scale.

Scores are bit-identical to master on the heaptrack 12 MP driver
(`12MP score = -64.321362` on both). All 21 `parity_lock`
tests, 2 `opaque` tests, and 3 of 4 `ssim2_skipmap_audit` tests
pass; the failing `modes_agree_on_jpeg_corpus` at q=5 is the same
pre-existing `Δ=1e-5 > 1e-6` Lossless-vs-Full noise-floor flake that
also fails on clean master at the same commit (master `Δ=6.25e-6`,
Phase 1 `Δ=1.04e-5` — both fail the gate; Phase 1 reshuffles the
fp atomic-add order but doesn't cause the failure). One bug at a
time — that flake is outside this PR.

Files: `crates/ssim2-gpu/src/pipeline.rs` (the aliasing change),
`crates/ssim2-gpu/tests/aliasing_invariants.rs` (14 new tests —
pair path × cached-ref path × {256², 1024², 2048², 4096²},
repeated-call stability, per-mode pair-vs-cached agreement,
set_reference re-arm cycle, identical-pair at multiple sizes),
`crates/ssim2-gpu/examples/bench_pair_vs_cached_cuda.rs` (perf
gate harness), `crates/ssim2-gpu/examples/heaptrack_driver.rs`
(host RSS profile target),
`benchmarks/ssim2_aliasing_perf_2026-05-22.csv` (the perf-gate raw
numbers), `crates/ssim2-gpu/docs/STRIP_PROCESSING.md` (Phase 2
design doc — reference only, not implemented in this commit).

### dssim-gpu — strip processing + compute_post_srgb dual-pyramid fix — 2026-05-22

- `Dssim::new_strip(client, image_w, image_h, h_body)` constructor +
  `compute_stripped(ref, dis)` entry point. Allocates a per-strip
  working set of `(h_body + 2 × halo) × image_w` and reuses it
  across strips, instead of one full-image-sized pyramid.
- `compute()` on a strip-mode instance auto-routes to
  `compute_stripped()` for backwards compatibility.
- Body-only pooling kernels added to `reduction.rs`:
  `fused_sum_range_kernel`, `thread_sum_range_kernel`, and the
  `launch_sum_range` dispatch helper.
- Fixed pre-existing bug in `compute_post_srgb`: the distorted
  pyramid was never built (only the reference pyramid was),
  causing `compute()` to return garbage scores. One-line addition
  of `build_linear_pyramid(false)` after the reference build. The
  `identical_image_is_zero` parity test now passes correctly
  (was returning ~0.902 on master).
- Strip parity test count grew from 7 → 17 tests covering pair
  path × cached-ref path × 3 image sizes (256/512/1024) × 2
  `h_body` values, plus edge cases (non-divisible `image_h`,
  single-strip, full-image-as-strip, halo-edge behavior).
- `bench_strip_vs_whole_cuda.rs` example writes
  `benchmarks/dssim_strip_vs_whole_<date>.csv` measuring whole vs
  strip wall-clock at 1/4/12/24 MP.
- GPU peak memory at 24 MP: whole=3.87 GB, strip(h_body=256)=
  621 MB (6.38× reduction). Host RSS at 12 MP (heaptrack):
  whole=2.30 GB, strip=608 MB (3.78× reduction).

### butteraugli-gpu — 2026-05-22 strip-mode pipeline (production-OOM fix)

Adds the strip-walker orchestration for the butteraugli pipeline:
`Butteraugli::new_strip(client, image_w, image_h, body_h)` allocates
~38 planes at `width × (body_h + 2 × HALO_ROWS)` instead of the
full `width × image_h`, then `compute_strip(ref, dis)` walks the
image strip-by-strip while folding each body band into a running
`(max, p3, p6, p12)` host-side aggregate. Result is bit-identical to
the whole-image path up to f64 reduction order (parity tests assert
`< 1e-4` relative error; measured `~1e-7` at 1024² across CUDA + WGPU).

GPU peak allocation per `Butteraugli<R>` instance (analytical
50-plane × 4-byte tally; matches actual GPU buffer pool usage):

| size       | whole       | strip(body=256) | ratio |
|------------|-------------|-----------------|-------|
| 1MP 1024²  | 200 MB      | 66 MB           | 3.05× |
| 4MP 2000²  | 763 MB      | 128 MB          | 5.95× |
| 12MP 4000×3000 | 2.24 GB | 256 MB          | 8.93× |
| 24MP 6144×4096 | 4.69 GB | 394 MB          | 12.19× |

Strip vs whole throughput at 12 MP (zenbench paired-compare, median):
`compute()` 106 ms vs `compute_strip()` 57 ms → 1.87× **faster** as
a side benefit of better cache locality on the smaller working set.

API surface additions:
- `pub mod strip` (when `cubecl-types` enabled) — strip walker + edge-
  mirror sRGB packer + per-strip host-side partials.
- `pub struct Butteraugli::{new_strip, compute_strip,
  compute_strip_with_options, is_strip_mode, image_height,
  strip_body_h, strip_halo_h}`.
- `pub enum Error::StripModeUnsupported(&'static str)` — surfaces
  clear errors when whole-image-only APIs (`set_reference`, `compute`,
  `compute_with_reference`, `compute_handles`) are called on a strip
  instance, and vice versa (`compute_strip` on a whole-image
  instance). Replaces the previous panicking `assert!`.

MVP limits documented in module + test docs:
- Single-resolution only — `new_multires` not strip-stitched.
- `set_reference` / `compute_with_reference` not yet strip-aware.

Test coverage: `tests/strip_parity.rs` grew from 6 to 19 tests (3
image sizes × 2 body values pair-path matrix, uneven-last-strip,
single-strip degenerate, body=image_h degenerate,
options-pass-through, multires-on-whole-still-works, plus 4 clear-
error assertions for the unsupported-API paths).

Bench: `examples/bench_strip_vs_whole_cuda.rs` (zenbench paired-
compare, writes `benchmarks/butter_strip_vs_whole_<date>.csv` +
supports `--whole-only-12mp` / `--strip-only-12mp` for heaptrack).

Heaptrack host RSS at 12 MP (single-shot bench, no heaptrack overhead
factored in): whole=2.91 GB, strip=731 MB — 4.0× host RSS reduction
(GPU-side reduction is larger; heaptrack measures host only).

Files: `crates/butteraugli-gpu/src/strip.rs` (new),
`crates/butteraugli-gpu/src/pipeline.rs` (`new_strip`,
`compute_strip*`, strip-mode helpers + guards),
`crates/butteraugli-gpu/src/lib.rs` (`pub mod strip`, new error),
`crates/butteraugli-gpu/tests/strip_parity.rs` (new),
`crates/butteraugli-gpu/examples/{strip_parity_numbers,bench_strip_vs_whole_cuda}.rs` (new).

### Changed — sweep image Dockerfile chain collapsed to single file `v26` — 2026-05-21

Replaces the `v14 → v15 → v18 → v19 → v20 → v21 → v22 → v23 → v24 → v25`
incremental chain (plus the legacy `v0` and `v13` root files) with a
single `Dockerfile.sweep.v26` that goes `FROM ubuntu:24.04` and inlines
every delta in proper layer order:

1. ubuntu:24.04 + stable apt deps (ca-certificates, curl, gnupg,
   libssl3, python3, python3-pip, gcc, libc6-dev)
2. pyarrow via pip
3. CUDA NVRTC + cudart + cudart-dev 12-6 (+ ldconfig + symlink)
4. s5cmd 2.2.2 + jq 1.7.1
5. cuda_dlsym_stub.so compiled from `scripts/sweep/cuda_dlsym_stub.c`
6. zen-metrics binary (CUDARC_CUDA_VERSION=12000 build)
7. vastai-fleet binary (inline-sweep features)
8. onstart + worker scripts
9. ENTRYPOINT = run_with_error_trap → onstart_unified.sh

Same runtime contract as v25. CI workflow (`sweep-image.yml`) updated
to build v26 (was v14) and to also build the `vastai-fleet` binary
alongside `zen-metrics`. Tag scheme: `ghcr.io/imazen/zen-metrics-sweep:v26`
+ `:v26-<short-sha>`.

Deleted files (commit pending): `Dockerfile.sweep`,
`scripts/sweep/Dockerfile.sweep`, `Dockerfile.sweep.v13`,
`Dockerfile.sweep.v14`, `Dockerfile.sweep.v15`, `Dockerfile.sweep.v18`,
`Dockerfile.sweep.v19`, `Dockerfile.sweep.v21`, `Dockerfile.sweep.v22`,
`Dockerfile.sweep.v23`, `Dockerfile.sweep.v24`, `Dockerfile.sweep.v25`.

Updated: `scripts/sweep/CLAUDE.md`, `scripts/sweep/README.md`, root
`CLAUDE.md` (current-state references; historical retro notes in
2026-05-15 sections left as-is).

Smoke build (local docker, base layers L1-L7 through `v26 base-layer
smoke passed`): PASSED (583 MB base image including ubuntu + CUDA
runtime+dev + pyarrow + stub).

### sweep infra v21 — 2026-05-19 (`infra(sweep): drop cudarc to CUDA-12.0 binding to evict cuEventElapsedTime_v2`)

Layers a CUDA-12.0-bound zen-metrics binary on top of v20's universal
cu* dlsym fallback (commit `c7af4dae`). The two fixes are complementary:

- v20's universal stub catches *runtime* misses on any cu* symbol cudarc
  statically references but the driver doesn't ship. Tolerates any new
  CUDA-N-only symbol cudarc starts dlsym'ing without rebuild.
- v21's binary eliminates the *compile-time* references to CUDA-13-only
  and CUDA-12.8+ _v2 symbols, so the runtime stub never needs to fire
  for those families. Restores clean stack traces if cubecl ever DOES
  call a v2 we hadn't anticipated (the stub would silently no-op,
  hiding the real bug).

Concrete trigger that drove the v21 layer:

The v19 smoke (instance 37050972 → first try 37050280) showed the
problem repeats one layer deeper. With CUDARC_CUDA_VERSION=12090 the
v2 symbols from cuda-13000 (cuCtxGetDevice_v2 + cuCoredump*) were
indeed evicted, but cudarc's `cuda-12080` gate ALSO pulls in
`cuEventElapsedTime_v2`, which driver 560.35.03 (CUDA 12.6 era) doesn't
export. cudarc 0.19.4's eager `Lib::new` loader dlsyms it at startup
and panics.

This is structurally the same bug as v19 — cudarc 0.19.4 declares any
`_v2` symbol gated `cuda-XXXXX`, then the eager loader resolves it at
startup. Older drivers won't have it. The CUDA version selection is
load-bearing: it must be low enough that no symbol gated above the
minimum driver release is compiled in.

- **`CUDARC_CUDA_VERSION=12000`** drops the binding to CUDA 12.0,
  the lowest CUDA 12 binding cudarc supports. Only `_v2` symbols
  that exist in CUDA 12.0 (and earlier) are compiled, and every one
  of those is present in NVIDIA drivers from 525.x onward (the
  CUDA-12 ABI cut-in). Audit: 76 `_v2` symbols still in the binary,
  all from CUDA 4-11 era + a handful gated `cuda-12000` (the
  `cuGraph*KernelNode_v2` family and `cuGetProcAddress_v2`); all
  ship in libcuda.so.1 525.85.05+.
- **`Dockerfile.sweep.v21`** (commit pending). Inherits from `v20`
  base, overlays a fresh `zen-metrics` binary built with
  `CUDARC_CUDA_VERSION=12000`. Build sanity expanded vs v19 — checks
  for cuCtxGetDevice_v2 + cuEventElapsedTime_v2 +
  cuCoredumpDeregister{Start,Complete}Callback in the binary's
  string table; any positive match fails the build.
  Image: `ghcr.io/imazen/zen-metrics-sweep:v21-c7af4da` / `:v21`
  (sha256:d3af6316c12ec637bb94485a619cb8c118e0b569ea45b39bebe9f20fa4e668c2).
  Earlier `:v20-cuda12000-6e3b0d9` retag of the same binary (built
  before c7af4dae landed) preserved for archival reference but
  superseded by v21 which sits on the universal-stub base.
- Driver filter stays `driver_version>=525.0.0` from v19.
- The LD_PRELOAD `cuda_dlsym_stub.so` universal cu* fallback (v20)
  stays in the image as defense-in-depth for any future cu* symbol
  cudarc adds that v21's strings check doesn't yet cover.
- **Smoke verdict (instance 37050972, v20-6e3b0d9 image, driver
  535.154.05, RTX 3060)**:
  - Zero `cuCtxGetDevice_v2` / `cuEventElapsedTime_v2` / `cuCoredump*`
    dlsym panics across all three chunks. The cudarc panic family is
    confirmed eliminated. (`grep -E "cuCtxGetDevice|cuEventElapsedTime|cuCoredump|Expected symbol"`
    on the container logs returned zero hits.)
  - NEW error class surfaced: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
    cubecl-cuda's runtime nvrtc generates PTX from the host CUDA
    toolchain (13.2) which embeds PTX ISA above what driver 535's
    libcuda can load. The fix is either to lift the floor to
    driver >=555 (CUDA 12.5+) OR to bake a lower-PTX-version nvrtc
    into the image. **Out of scope for this commit** — the cudarc
    panic family is the deliverable; the PTX class is a separate
    blocker the next session can take.
  - Cost: instance ran ~5 minutes at $0.0544/hr = $0.0045. Well
    under the $1 cap.
  - **Ready-to-fanout flag**: NO. The PTX class blocks all current
    boxes with driver <555 from producing sidecars. Fix the floor
    OR ship a CUDA-12-built nvrtc before fanning out.

### sweep infra v21.1 — 2026-05-19 (`fix(sweep): bump driver_version floor to 555 (CUDA 12.5+) for PTX compat`)

PTX-version fix per the v21 smoke "ready-to-fanout flag" follow-up.
Three launchers bumped `driver_version>=525.0.0` -> `driver_version>=555.0.0`:

- `scripts/sweep/launch_backfill.sh:187`
- `scripts/sweep/launch_single_instance.sh:141`
- `scripts/sweep/v15/launch_gpu.sh:19`

Driver 555.42 is the first NVIDIA release supporting the CUDA 12.5
PTX ISA that cubecl-cuda's nvrtc emits from the local CUDA 13.2
toolchain. Older drivers reject those modules at load time with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` even though the runtime symbol
surface is fine (v21 binary covers them via the universal cu*
dlsym stub). This unblocks the multi-codec EXP fleet sweep.

### sweep infra v20 — 2026-05-19 (`fix(cuda-shim): universal cu* dlsym fallback to no-op stub`) — landed on master 2026-05-19 by parallel agent

Master commit `c7af4dae`. Drops the per-family allowlist (cuCoredump*)
for a universal `cu*` cu-prefix fallback: any cu* symbol the real driver
doesn't export gets a no-op stub returning CUDA_SUCCESS (0). Also adds
a _v2-suffix retry path that first tries `cu*_v2`, then falls back to
the unsuffixed `cu*`, before resorting to the universal no-op stub.
Image: `ghcr.io/imazen/zen-metrics-sweep:v20` (sha256:8e92d1ec6e23…).

### sweep infra v19 — 2026-05-18 (`infra(sweep): bump cudarc past _v2 gate + restore driver_version filter`)

Targets the SECOND DlSym panic family surfaced by the v18 EXP-MULTI-CODEC
smoke. v18's widened `cuda_dlsym_stub.so` killed the `cuCoredump*`
panics on driver 580.x but exposed a deeper problem on driver 555.x:
cudarc 0.19.4 (compiled with `-F cuda-13000` via our local CUDA 13.2
toolkit autodetect) also dlsyms `cuCtxGetDevice_v2`. Driver 555 (CUDA-12
era) doesn't export `_v2`, so cubecl-cuda's first context retain panicked
with `Expected symbol in library: cuCtxGetDevice_v2`.

Three concurrent fixes:

- **Forced cudarc onto the CUDA 12.9 binding surface via env var**
  (no source edit needed). cudarc 0.19.4's `build.rs` checks
  `CUDARC_CUDA_VERSION` before `cuda-version-from-build-system`
  (lines 31-44 of `cudarc-0.19.4/build.rs`). Setting
  `CUDARC_CUDA_VERSION=12090` at build time tells cudarc to compile
  against `cfg(feature="cuda-12090")` only — leaving every
  CUDA-13-only symbol (`cuCtxGetDevice_v2`,
  `cuCoredump{Register,Deregister}{Start,Complete}Callback`, etc.)
  out of the binary. No Cargo.toml change, no cubecl fork rebump.
  The CUDA driver ABI is forward-compatible inside the 12.x family,
  so the 12.9 surface loads on every CUDA 12 driver in the vast.ai
  pool. This obsoletes the LD_PRELOAD stub for the panics it was
  bandaging; the stub stays in the image as defense-in-depth.
- **`Dockerfile.sweep.v19`** (commit pending). Inherits from `v18`
  base, overlays the new binary at `/usr/local/bin/zen-metrics`,
  and adds a load-bearing `strings` assertion: the post-COPY RUN
  fails the build if `cuCtxGetDevice_v2` or
  `cuCoredumpDeregisterCompleteCallback` is still present in the
  binary's dynamic string table. Catches a regression in cudarc
  build-flag handling before the image is pushed.
- **`scripts/sweep/launch_*.sh` driver filter relax**:
  - `launch_single_instance.sh` line 131 — actually *adds* the
    `driver_version>=525.0.0` filter that the comment at line 123
    claimed but never wired. Floors at driver 525 (first CUDA 12
    ABI). Drops `cuda_vers>=12.5` (replaced by `cuda_max_good>=12.0`
    which is the more accurate field).
  - `launch_backfill.sh` line 176 — replaces the v18-era
    `cuda_max_good>=12.6 driver_version<570.0.0` upper-ceiling
    filter with `cuda_max_good>=12.0 driver_version>=525.0.0`. The
    upper ceiling is no longer needed since the v19 binary doesn't
    reference any CUDA-13 symbols; widening the pool restores the
    cheap consumer-GPU offers that v18's filter excluded.
  - `scripts/sweep/v15/launch_gpu.sh` line 15 — same relax pattern,
    `cuda_max_good>=12.0 driver_version>=525.0.0`.
- `scripts/sweep/deploy_fast.sh` left untouched (header marks it
  DEPRECATED; scheduled for deletion).
- **Image push**:
  `ghcr.io/imazen/zen-metrics-sweep:v19-<short_hash>` and `:v19`
  (sha256 pending build). Layers L0-L7 inherited from v18 (~700 MB
  cached); only L8 binary layer is new (~280 MB on the wire).
- **Smoke verdict**: pending. Will run one box with mixed-codec
  chunks under a $1 cap; recorded below when complete.

### sweep infra v18 — 2026-05-18 (`infra(sweep): widen cuda_dlsym_stub + bump jxl-encoder`)

- **`scripts/sweep/cuda_dlsym_stub.c` widened from 1 → 4 intercepts**
  (commit pending on master). Now stubs the full
  `cuCoredump{Register,Deregister}{Start,Complete}Callback` family,
  not just the Complete-Deregister symbol that v17 (`4831093`) caught.
  Surfaced 2026-05-18 in an EXP-MULTI-CODEC smoke on driver 580.x:
  cubecl-cuda's static lookup of `cuCoredumpDeregisterStartCallback`
  panicked even with v17's shim live, killing the device dispatcher.
  v18 fixes the immediate cause.
- **`Cargo.toml` jxl-encoder pin bumped `cb5d9e4` → `6b8eefc1`**
  (commit pending). Pulls in W44-1..W44-45 RD-affecting commits on
  jxl-encoder main since the Feb-2026 security release: gaborish
  ordering / global_scale / EPF sharpness / animation-path patches +
  CfL / butteraugli-loop / patches default-on. The multi-codec sweep
  re-collects training data against the encoder we actually ship; the
  pre-W44 cached sidecars are stale.
- **`Cargo.toml [patch.crates-io] zenjxl = path-patch to local
  `../zenjxl`** (commit pending). zenjxl 0.2.1 on crates.io was
  written against jxl-encoder 0.3.1 — bumping the pin above to
  6b8eefc1 adds `premultiplied_alpha` to `AnimationParams` and 6 new
  fields (`blend_mode`, `blend_source`, `save_as_reference`,
  `reference_only`, `name`, `timecode`) to `AnimationFrame` per
  jxl-encoder commits `f3b042f7` (alpha) and `d0e47838` (frame-header
  API expansion). zenjxl's struct literals stopped compiling. The
  local zenjxl path-patch switches the two literals to
  `AnimationParams::default()` and `AnimationFrame::new(...)` so the
  optional fields follow upstream as the API evolves. Drop once
  zenjxl 0.2.2 ships with the same fix.
- **New `Dockerfile.sweep.v18`** (commit pending). Inherits from
  `ghcr.io/imazen/zen-metrics-sweep:v17`, overlays:
  - the widened `cuda_dlsym_stub.so` (rebuilt from the patched C
    source with a 4-symbol verification step in the same RUN), and
  - a fresh `/usr/local/bin/zen-metrics` baked from the jxl-encoder
    6b8eefc1 build (multi-codec smoke verifies `zenjpeg / zenwebp /
    zenavif / zenjxl` all appear in `sweep --help`).
  Built + pushed to GHCR as
  `ghcr.io/imazen/zen-metrics-sweep:v18-f4d28e9` and `:v18`
  (sha256:e7043763e8934ae5).
- **Binary on R2**:
  `s3://coefficient/binaries/zen-metrics-0.6.0-multicodec-f4d28e9-linux-x86_64-gpu`
  (97 MB, fastest-link feature set
  `sweep,png,gpu,gpu-cuda`).
- **Smoke verdict: partial pass / new failure mode discovered.**
  - v18 stub successfully eliminates all `cuCoredump*` DlSym panics
    that were the v17 blocker (smoke at instance 37049180 on driver
    555.58.02 produced ZERO Coredump errors across 5 cells / 3
    chunks).
  - NEW DlSym panic surfaced one layer deeper: cudarc 0.19.4 (compiled
    with the `cuda-13000` feature) calls
    `cuCtxGetDevice_v2` at runtime. Driver 555.58.02 (CUDA 12-era)
    doesn't export that v2 symbol — `cuCtxGetDevice` (v1) is the
    available one. v18's stub returns `RTLD_NEXT/dlsym` for anything
    not in the Coredump set, so the v2 lookup fails and the dispatcher
    panics with `Expected symbol in library: cuCtxGetDevice_v2`.
    Worker logs at
    `s3://coefficient/jobs/multi-codec-smoke-v18-2026-05-18/worker-logs/37049180-failure.log`.
  - This is NOT a v18 regression — the cuCtxGetDevice_v2 symbol issue
    has always been there, but was previously masked by the Coredump
    panic landing first. Fixing it cleanly requires either (a) bumping
    cudarc past 0.19.4 (which dropped the CUDA-13-feature gate on
    several v2 symbols), or (b) extending the shim to redirect
    `cuCtxGetDevice_v2` lookups to `cuCtxGetDevice` (semantic
    redirect, not a no-op — the v2 signature is identical to v1 per
    cudarc's binding). Option (b) is ~10 lines but crosses the smoke
    budget's >20-line non-trivial-fix gate (more v2 symbols may also
    need redirection — cudarc's CUDA-13 feature set is ~80 symbols),
    so it's deferred for user direction.

### zensim-gpu — 2026-05-18 (`feat/zensim-weights-and-handles`)

### zensim-gpu — 2026-05-18 (`feat/zensim-weights-and-handles`)

- **`zensim-gpu` — canonical default weights baked in** (commit pending,
  rebased from `74329e71`). Adds `crates/zensim-gpu/src/weights.rs`
  with the 228-element `WEIGHTS_PREVIEW_V0_2` array (byte-equal copy
  of `zensim::profile::WEIGHTS_PREVIEW_V0_2`), re-exports it from the
  crate root, and ships `ZensimParams::default_weights()` /
  `::with_canonical_v0_2()` ctors. `MetricParams::default_for(Zensim)`
  in `zenmetrics-api` now returns finite scores out of the box
  (previously NaN). Drift guard `tests/weights_parity.rs` asserts
  byte-equality with the CPU crate's static so a future profile
  rotation fails loud. Tolerance on `dispatch_zensim` identity test
  is `< 1.0` (loosened from `< 1e-3`) — see commit message and
  investigation memo `zensim_gpu_identity_drift_investigation_2026-05-19.md`
  for the f32-noise rationale (CPU short-circuits identical-input,
  GPU runs the full kernel and picks up ~0.2 score drift in coarse-
  pyramid peak-pool features; class already documented in
  `cpu_parity.rs::identical_input_all_zeros`).

### sweep infra v2 — 2026-05-18 (`feat/sweep-infra-unified`)

Operational fixes after the iwssim / cvvdp / ssim2 backfill sessions
exposed three load-bearing bugs:

- **`zen-metrics-cli` — `score-pairs --fail-on-bogus`** (commit
  `242d4b4a`). New per-metric distribution sanity check gate that
  inspects the score column after parquet write and exits rc=2
  (distinct from rc=1) when n_NaN > 0, ≥ 50% of rows are exactly at
  the metric's identity value, range < 0.01 across ≥ 4 rows, or
  the mean falls outside the metric's documented range. Catches the
  iwssim NaN-on-identical mode (525 sidecars uploaded with every
  score at 0 or NaN before V_24 training surfaced the failure) and
  the cvvdp-on-cpu atomic-panic mode (all rows fall through to JOD
  10.0). 10 unit tests in `fail_on_bogus_tests`.
- **`scripts/sweep/metric_backfill_chunk_worker.sh`** (commit
  `5b98e50c`). Single unified worker that dispatches by `--metric`,
  replacing `iwssim_backfill_chunk_worker.sh` /
  `ssim2_backfill_chunk_worker.sh` / the single-metric portion of
  `cvvdp_backfill_chunk_worker.sh`. Calls `score-pairs
  --fail-on-bogus` by default and uploads a structured failure log
  to `s3://zentrain/<run>/failures/<chunk>.log` on rc=2 instead of
  treating the sidecar as authoritative training data. Per-metric
  files marked DEPRECATED but retained for in-flight runners.
- **`crates/vastai-fleet/`** (commit `3a849a69`). Rust binary
  replacing the bash + python heredoc destroyers under
  `/tmp/cvvdp-resume/run_destroy_*.sh`. Three subcommands —
  `status` / `destroy` / `watch` — all driven by a defensive
  parser (`crates/vastai-fleet/src/parse.rs`) that tolerates every
  failure mode the bash destroyer hit: empty stdout, deprecation
  banner glued onto JSON, individual malformed rows, dph as string
  vs float, v0 vs v1 envelope shape. 22 tests (15 parse + 7 cli)
  using `--raw-input` fixtures so no real vast.ai API access is
  needed in CI.
- **`scripts/sweep/launch_backfill.sh`** (commit `50982c33`).
  Single launcher with `--metric / --run-id / --chunks / --docker /
  --max-dph / --n-boxes / ...` flags replacing the per-metric
  `launch.sh` / `launch_imazen.sh` files. Auto-derives the
  destroy-target as `(n_chunks - 10 grace)` from the chunks file
  and prints (or runs, with `--watch`) the `vastai-fleet watch`
  invocation. Per-metric launchers marked DEPRECATED.
- **`scripts/sweep/fleet_status.sh`** (commit `5de4c676`).
  One-shot dashboard combining fleet status, R2 sidecar count vs
  chunks total, failure-log count, and sample sidecar validity
  check (3 random sidecars: score-column min/max/mean/NaN-count;
  flag constant or NaN-containing chunks). Second backstop against
  the bogus-data failure mode — even without `--fail-on-bogus` on
  the worker, this surfaces broken sidecars on a sample.

### iwssim-gpu / zen-metrics-cli — 2026-05-17 adaptive small-image support

- `4e01232c` — adaptive IW-SSIM via reflect-pad to `MIN_NATIVE_DIM`
  (176). New `pub const MIN_NATIVE_DIM`, `pub struct IwssimConfig {
  allow_small: bool }`, `Iwssim::with_config`,
  `Iwssim::padded_dimensions`, `Iwssim::is_padded` on the typed
  pipeline; `IwssimParams { allow_small: bool }` +
  `IwssimParams::allow_small(bool)` on the opaque API; default is
  `false` (rejects sub-176 inputs, matches historical behaviour
  byte-for-byte). When `allow_small = true` and a native axis is
  below `MIN_NATIVE_DIM`, host-side `reflect_pad_f32` /
  `reflect_pad_rgb_u8` extends the input to `(max(W, 176), max(H,
  176))` before upload; the GPU pipeline runs at padded dims with
  unchanged kernels. The resulting score is the IW-SSIM of the
  padded image — informational, not bit-exact stock IW-SSIM. New
  tests: 9 host-side unit tests (`reflect_pad_*`, `reflect_index`)
  + 7 GPU integration tests (dims {22, 44, 88, 132, 175, 176} +
  rectangular {80×100, 200×80, 80×200}, gated on
  `RUN_GPU_ADAPTIVE=1`). Stock 1024×1024 drift between flag-on and
  flag-off measured at 3.4e-8 (GPU non-determinism floor); the
  pre-existing parity-lock tests still pass.
- `4e01232c` — `zen-metrics-cli score-pairs`: new
  `--allow-small-images` flag wires through a process-wide
  `AtomicBool` to `resolve_default_params`, which swaps
  `IwssimParams::DEFAULT` for `IwssimParams::allow_small(true)` at
  metric construction. Other metrics ignore the flag.
- `6227c1a8` — `iwssim_backfill_chunk_worker.sh` defaults
  `IWSSIM_ALLOW_SMALL=1` so the new docker image's `score-pairs`
  call surfaces the adaptive path. Override with
  `IWSSIM_ALLOW_SMALL=0` to keep stock-only behaviour.

### Workspace — 2026-05-17 merge wave

Seven branches landed on master in dependency order. Each was tested
locally on RTX 5070 (CUDA 13.2) before merging; cross-platform CI
runs on push.

- `26a5ae79` + `1b8ccab8` — zensim-gpu: 372-feature regimes
  (Basic/Extended/WithIw) + principled per-channel H-blur activity
  (mirrors zensim CPU `feat/principled-activity` at `2dab8f30`).
  Parity tightened to ≤5e-3 rel on the masked block at every (scale,
  channel) (was 1.5e-1 / 2.0e-1 worst case). 12 MP WithIw bench
  steady-state 24.53 ms / iter (−5.5% vs image-wide mirror, −8.9%
  vs cascade approach).
- `caa4a147` — Merge Phase 2 (`feat/api-uniformity-zenpixels`):
  uniform `Backend` enum + opaque `<Metric>Opaque` shim types +
  `zenpixels::PixelSlice` integration across all six GPU metric
  crates. Existing typed `<Metric><R: Runtime>` API gated behind
  `cubecl-types`. 12/12 opaque integration tests pass.
- `8b04b642` — Merge Phase 3 (`feat/zenmetrics-api-umbrella`):
  new `zenmetrics-api` umbrella crate, enum-dispatched `Metric`,
  per-metric `MetricParams`. 7/7 dispatch + 3/3 pixels_smoke tests.
- `78b162f6` — Merge `feat/metric-defaults-fix`: `impl Default for
  CvvdpParams` (PLACEHOLDER deprecated); iwssim returns 1.0 instead
  of NaN on identical inputs; umbrella `dispatch_iwssim` now
  asserts ≈1.0.
- `cc862e7a` — Merge `feat/compute-handles` (Phase 4 device path):
  `compute_handles` on five typed pipelines (cvvdp/butter/ssim2/
  dssim/iwssim, gated behind `cubecl-types`). Real
  `MetricContext::upload_pair` + `Metric::compute_handles` wired
  in the umbrella. Smoke verifies `compute_handles` ≡
  `compute_srgb_u8` within 1e-3 rel. Saves ~17 ms × (N-1) metrics
  per pair in batch mode. zensim deferred to a follow-up.
- `75913cec` — Merge `feat/cli-umbrella-flip` (Phase 4 CLI):
  zen-metrics-cli drops direct per-crate `gpu-*` feature deps,
  routes through `zenmetrics-api` umbrella. iwssim and zensim
  now supported through the CLI (were absent before). 29/29 CLI
  tests pass. Two typed-path escape hatches stay (cvvdp batch
  scorer, butter pnorm3 two-column emit) via the umbrella's
  `zenmetrics_api::{butter,cvvdp}` re-exports.
- Stale `origin/main` branch (last touched 2026-05-07, strict
  ancestor of master) deleted; `master` remains the only branch.

### Known issues (not blocking merge)

- `crates/zen-metrics-cli/src/main.rs`: the iwssim batch-caching
  scorer was removed during merge conflict resolution because its
  source file depended on infra cli-flip had deleted. iwssim
  through `score-pairs` now re-allocates per-pair instead of
  caching per `(W, H)`. Documented as a TODO; restore via the
  umbrella's `zenmetrics_api::iwssim` re-export when batch perf
  matters.

## zenmetrics-api (new crate, Phase 3)

### Added

- New `zenmetrics-api` umbrella crate at `crates/zenmetrics-api/`.
  Composes the six GPU metric crates (cvvdp-gpu, butteraugli-gpu,
  ssim2-gpu, dssim-gpu, iwssim-gpu, zensim-gpu) behind a single
  enum-dispatched `Metric` + `MetricKind` + `MetricParams` API. Per-
  metric Cargo switches (`cvvdp`/`butter`/…), backend forwarding
  (`cuda`/`wgpu`/`hip`/`cpu`), `pixels` opt-in, and a `cubecl-types`
  feature that re-exports each metric's typed `<Metric><R>` and
  exposes a `MetricContext<R>` scaffold (client + dims + generation
  counter) for future shared-upload work. 7 dispatch tests + 3
  `compute_pixels` smoke tests + 1 doctest all pass against CUDA on
  RTX 5070.

### Phase 4 candidates

- `Metric::compute_handles(&ctx, pair_handles)` — requires each metric
  crate to add a `compute_handles(handle_r, handle_d) -> Score` entry
  point that consumes pre-uploaded device buffers (today every
  metric's typed `compute` re-uploads internally).
- Flip `zen-metrics-cli` over to depend on `zenmetrics-api` instead of
  the four individual gpu-* crate deps.
- `build.rs` cubecl version cross-check via `cargo metadata` instead
  of the current advisory `cargo:warning=`.

### Milestone — tick 500 (cvvdp-gpu)

The 416–500 tick arc was a deep invariant-pinning + documentation
hardening pass. ~85 tests added, ~30 doctests added, every public
constant + helper now has direct bit-pin / structural coverage.
Major themes:

- **Constants-pin series** (ticks 393–397, 401–402): every cvvdp
  v0.5.4 numeric (`BETA_*`, `MASK_*`, `D_MAX`, `KERNEL_A`, `GAUSS5`,
  `SRGB_LINEAR_TO_DKL`, `JOD_A`, `JOD_EXP`, `IMAGE_INT`, `PER_CH_W`,
  `BASEBAND_W`, `CSF_BASEBAND_RHO`, `SENSITIVITY_CORRECTION_DB`,
  `XCM_3X3`, `CH_GAIN`, `PU_BLUR_KERNEL_1D`, `PU_PADSIZE`,
  `LOG_L_BKG_AXIS`, `LOG_RHO_AXIS`, `LOG_S_O0_C1/C2/C3`,
  `GE_SIGMA`, display constants, crate-level dims) is bit-pinned
  against pycvvdp v0.5.4. A silent edit cascades as a specific test
  failure naming the constant, not a 0.001 JOD drift on shadow_jod.

- **Function-level structural invariants** (ticks 416–434): direct
  pin files for `flatten_band_weights`, `precomputed_band_weights`,
  `laplacian_pyramid_dec_scalar`, `gausspyr_reduce_scalar`,
  `gausspyr_expand_scalar`, `srgb_byte_to_dkl_scalar`,
  `weber_contrast_pyr_dec_scalar`, `clamp_diff_soft`,
  `phase_uncertainty_no_blur`, `mask_pool_pixel`,
  `mult_mutual_pixel`, `met2jod`, `do_pooling_and_jod_still_3ch`,
  `precompute_logs_row`, `phase_uncertainty_band`,
  `gaussian_blur_sigma3`, `mult_mutual_band`,
  `predict_jod_still_3ch`. Each pins shape, determinism via
  `to_bits()`, branch thresholds, dynamic range, edge cases.

- **Doctest coverage** (ticks 442–481, extended 507/510/513): every
  public constant + helper has a `# Examples` doctest with
  bit-equality / range assertions, plus rendered example docstrings
  on the user-facing `Cvvdp::*` scoring methods (`new` / `new_with_geometry`
  / `score` / `score_with_reference` / `compute_dkl_jod` /
  `compute_dkl_jod_with_warm_ref` / `compute_dkl_jod_host_pool*` /
  `warm_reference`). 44 doctests pass, 6 are `ignore` (Cvvdp methods
  need a feature-gated `Backend` type alias; docs.rs has no GPU).
  Measured 2026-05-16 via `cargo test --doc -p cvvdp-gpu`.

- **State machine pins** (ticks 486, 488, 489, 491, 493, 494, 497,
  498, 499): `Cvvdp` cache state machine (`set_reference` vs
  `warm_reference` independence, no-pollution from one-shot scoring),
  bit-determinism across all 3 scoring paths
  (`score`/`score_with_reference`/`compute_dkl_jod_with_warm_ref`),
  four-path consolidation, `new` ↔ `new_with_geometry(STANDARD_4K)`
  equivalence, 8×8 + 128×8 + 8×128 boundary/aspect smoke,
  cross-instance bit-equality, degenerate-input stability.

- **CHANGELOG provenance** (ticks 482–487, 490, 492): every entry
  from tick 386 onward now references its implementing commit's
  short hash via sed-batch backfill. Workspace convention.

- **Maintenance / cleanup** (various): cargo fmt drift sweeps,
  clippy fixes (`needless_range_loop`, `excessive_precision`,
  `clone_on_copy`), `lib_reexports.rs` re-export surface pin,
  `cvvdp_mem_table` example refactored onto `recommend_parallel`.

Branch is at parity with pycvvdp v0.5.4 to ≤0.005 JOD on every
fixture; the test suite catches drift across every layer that pins
mention. Tick 500.

**Post-milestone long tail (ticks 501–540, summarised here so the
detailed entries below stay grep-able):**

- **Re-export surface widened** (501–503): lib_reexports.rs grew
  from 5 to 11 pins, covering `Cvvdp<R>`, `Error`, `Result<T>`,
  the four lib-root constants, the `params::*` scaffolding types,
  and the `host_scalar::*` + 5 `kernels::*` submodule paths.
- **CHANGELOG provenance finished** (504): the last 4 unhashed
  entries (ticks 383/387/398/399) backfilled via `jj file annotate`
  + change_id resolution. All entries from tick 383 onward are
  now hash-tagged.
- **Rustdoc + clippy clean** (505, 514, 516, 518): cleared the 96
  `#[cube(launch)]` macro-emitted `missing_docs` warnings via
  file-level `#![allow(missing_docs)]` on each kernel file; added
  crate-level `#![warn(missing_docs)]` guard so future undocumented
  pub items surface at `cargo doc`; fixed an unresolved intra-doc
  link in pipeline.rs's private docs; tightened the type-complexity
  + assertions-on-constants clippy warnings introduced by 503.
- **State-machine + boundary smokes** (506–513): cross-instance
  bit-equality on fresh `Cvvdp::new` instances, degenerate-input
  stability on `pixels_per_degree` and `new_with_geometry`,
  end-to-end smoke at extreme aspect ratios (128×8 / 8×128 / 1024×8
  / 8×1024), GPU + host_pool perf_mode bit-equality on the cpu and
  GPU runtime variants, doctests on the remaining user-facing
  `Cvvdp::*` scoring methods (`new_with_geometry`, `warm_reference`,
  `compute_dkl_jod`).
- **Manifest URL tightening** (519–521): pinned the canonical R2
  host (`https://coefficient.r2.imazen.org/`) + the crate-specific
  bucket subpath (`/cvvdp-goldens/`) on `MANIFEST_URL`, closing
  silent CDN/sibling-crate misroute gaps the existing structure
  checks would have passed.
- **Static-assert promotion** (522–524): the integer-typed lib-root
  constants (`N_CHANNELS == 3`, `MAX_LEVELS == 9`,
  `PYRAMID_MIN_DIM == 4`, `PYRAMID_MIN_DIM * 2 == 8`) and the
  `CsfChannel` discriminants (`A == 0` / `Rg == 1` / `Vy == 2`)
  promoted from runtime `assert_eq!` to module-level
  `const _: () = assert!(...)` static asserts. Fundamental
  dimension parameters now catch at compile time.
- **Stuck-at-constant pinned across all four scoring paths**
  (508, 509, 525): strict q-level separation (q=90 > q=20 > q=1
  with ≥ 0.01 JOD gap) on `score()`, `compute_dkl_jod_host_pool`,
  and `compute_dkl_jod_host_pool_with_warm_ref`. Catches
  near-correct-but-non-discriminative collapse that the manifest
  tolerance pin (0.005 JOD) wouldn't surface.
- **Documentation polish** (526, 527, 528, 529, 530): added this
  long-tail summary block; replaced workspace README's stale
  `TBD | TBD | TBD` cvvdp-gpu row with
  `(pending — reference is pycvvdp v0.5.4)`; recorded the saturation
  point — every clippy / rustdoc / missing_docs surface is clean
  across all feature combinations and target selections, the
  lib_reexports surface is fully pinned at 11 tests, the
  stuck-at-constant contract is pinned across all 4 scoring paths,
  the CHANGELOG hash provenance is complete from tick 383 onward,
  and there are no remaining TODOs / FIXMEs in the source;
  normalized 8 `# Example` (singular) docstring headers in
  pipeline.rs to `# Examples` (plural) to match Rust API guidelines.

- **Intermediate-method doctest sweep** (531–537): every
  `Cvvdp::compute_dkl_*` method now has a rendered `# Examples`
  doctest. 7-tick arc adding `ignore` doctests for
  `compute_dkl_planes` (531), `compute_dkl_gauss_pyramid` (532),
  `compute_dkl_laplacian_pyramid` (533), `compute_dkl_weber_pyramid`
  (534), `compute_dkl_t_p_bands` (535),
  `compute_dkl_csf_weighted_bands` (536), and `compute_dkl_d_bands`
  (537). Doctest count grew from 44+5 → 44+13. Closes the docs.rs
  rendered-example gap on the advanced intermediate-stage API
  surface. Subsequent significant improvement still requires a
  fresh measurement (pycvvdp-baseline SRCC) or a directed feature
  (`CvvdpParams` JSON loader).

- **Sweep finalization** (538, 539): documented the 531-537 sweep
  in this long-tail block; caught the last remaining
  `# Example` (singular) header in `host_scalar.rs:48` that tick
  530's pipeline.rs-only normalization sweep had missed. The
  `# Examples` (plural) Rust API guidelines convention is now
  applied across every docstring in the crate.

As of tick 575 the crate contains **165** `const _: () = assert!(...)`
static asserts spread across 11 test files. Ticks 548-575 grew
the count from 11 to 165 by mining genuine gaps after each
premature "saturation" call: every named cvvdp v0.5.4 numeric is
now pinned (values via `to_bits()` bit-equality), every load-
bearing sign is pinned (`is_sign_positive` / `is_sign_negative`),
every load-bearing ordering is pinned (`u32 <` on `.to_bits()`
for positive f32 operands), every load-bearing cross-equality is
pinned, plus the SRGB_LINEAR_TO_DKL opponent-color sign signature
and the BETA hierarchy. These fire at compile time and are the
load-bearing enforcement; the runtime `#[test]` fns are preserved
beside them to keep test-runner-visible names referenced by older
CHANGELOG entries resolvable.

Earlier static-assert milestones:
- Tick 539 had 11 static asserts (CsfChannel + lib_constants + lib_reexports).
- Tick 540 verified clean: `cargo clippy --all-targets --all-features`,
  `cargo doc --document-private-items`, `cargo test --doc` (44 + 13).
- Ticks 548-572 promoted scalar+array bit-pins across pool, csf,
  masking, pyramid, color, display, params modules.
- Ticks 573, 575 added cross-bundle linkage and positivity pins on
  `CvvdpParams::PLACEHOLDER`'s scaffolding sub-bundles.
- Tick 576 (this entry) updates the milestone block to reflect
  the current state.

As of tick 576 verified clean across:
  - `cargo clippy -p cvvdp-gpu --all-targets --all-features` — 0 warnings
  - `cargo doc -p cvvdp-gpu --no-deps --document-private-items` — 0 warnings
  - `cargo test --doc -p cvvdp-gpu` — 44 passed + 13 ignored

Tick 548 promoted 5 more runtime asserts to static asserts (4 in
`csf_axes_invariants.rs` — `LOG_L_BKG_AXIS.len() == N_L_BKG`,
`N_L_BKG == 32`, `LOG_RHO_AXIS.len() == N_RHO`, `N_RHO == 32` —
plus `N_RHO > 0` in `lib_reexports.rs` to mirror the existing
`N_L_BKG > 0`). Static-assert count is now 16 across 3 test files.
The runtime `#[test]` fns in `csf_axes_invariants.rs` are preserved
beside the new static asserts (same compatibility rationale as
ticks 522-524). Same clippy / doc / doctest verification as tick 540
still passes (no new warnings introduced).

Tick 549 promoted 3 further invariant runtime asserts to static
asserts on physical-meaning constants that other tests in the same
file are predicated on:
  - `PU_PADSIZE == 6` in `phase_uncertainty_band_invariants.rs`
    (the branch-boundary parameter; `branch_boundary_at_pu_padsize`
    hardcodes the 6/7 transition pairs)
  - `PU_BLUR_KERNEL_1D.len() == 13` in `masking_constants.rs` (the
    σ=3 truncation tap count the per-element expected[] array
    depends on)
  - `SRGB8_TO_LINEAR_LUT.len() == 256` in `color_scalar.rs` (the
    one-entry-per-u8 LUT-size contract the indexing semantics rely
    on)
Static-assert count is now 19 across 5 test files. Verification:
same clippy / doc / doctest status as tick 540 — no regressions.

Tick 550 promoted 3 `f32::is_finite()` runtime asserts to static
asserts in `lib_reexports.rs` on the re-exported scalar constants
`MASK_C`, `JOD_A`, and `KERNEL_A`. `f32::is_finite` is `const fn`
in stable Rust since 1.83 (workspace pins `rust-version = "1.93"`,
absolute language minimum per project policy is 1.85, both well
above 1.83). Catches a refactor that accidentally substitutes
`f32::NAN` or `f32::INFINITY` as a constant literal. Static-assert
count is now 22 across 5 test files.

Tick 551 promoted 7 `DisplayGeometry::STANDARD_4K` +
`DisplayModel::STANDARD_4K` field-value runtime asserts to static
asserts in `display_geometry.rs`. u32 fields use direct `==`;
f32 fields use `to_bits()` (because `f32::PartialEq` isn't yet
`const fn` in stable Rust, but `f32::to_bits` is). Covered fields:
`resolution_w == 3840`, `resolution_h == 2160`,
`distance_m == 0.7472`, `diagonal_inches == 30.0`,
`y_peak == 200.0`, `y_black == 0.2`, `y_refl == 0.397_887_36`. The
v1 R2 manifest goldens were captured against these exact values;
a silent drift now fails to compile rather than at test time.
Static-assert count is now 29 across 6 test files.

Tick 552 added a compile-time pin for
`CvvdpParams::PLACEHOLDER.perf_mode == PerfMode::Strict` in
`params_placeholder.rs`. Uses `matches!` (which is `const`-callable)
since derived `PartialEq` on enums isn't yet `const fn` in stable
Rust. Every parity test inherits this perf-mode through
`Cvvdp::new(..., PLACEHOLDER)`; a silent flip to Fast would have
changed the calibration baseline for dozens of goldens. Static-
assert count is now 30 across 7 test files.

Tick 553 fixed two MSRV-reference factual errors introduced in
the tick 550 / 552 entries: the original wording said "MSRV 1.85"
but the workspace pins `rust-version = "1.93"` (with 1.85 as the
project's absolute language minimum, not the actual current
MSRV). Updated both the CHANGELOG entry and the in-source comment
in `params_placeholder.rs`. No code change; no test impact.

Tick 554 added a compile-time pin for `!CVVDP_COLUMN_NAME.is_empty()`
in `column_name.rs`. `str::is_empty` is `const fn` since Rust 1.39
(well below this crate's MSRV 1.93). An empty column name would
silently produce parquet sidecars with an unnamed score column,
breaking joins downstream. `str::starts_with` is NOT yet const fn
in stable Rust, so the `cvvdp_imazen_` prefix check stays runtime-
only. Static-assert count is now 31 across 8 test files.

Tick 555 promoted 4 Burt-Adelson kernel-constant bit-pins to
static asserts in `pyramid_scalar.rs`:
  - `KERNEL_A == 0.4_f32` (cvvdp v0.5.4 Burt-Adelson parameter)
  - `GAUSS5[1] == 0.25_f32`, `GAUSS5[2] == 0.4_f32`,
    `GAUSS5[3] == 0.25_f32` (inner taps of the 5-tap Gaussian)
The outer taps `GAUSS5[0]` and `GAUSS5[4]` stay runtime because
they use `(.. - ..).abs() < 1e-7` tolerance and `f32::PartialOrd::lt`
is not yet `const fn` in stable Rust. Static-assert count is now
35 across 9 test files.

Tick 556 promoted 3 scalar masking-constant bit-pins to static
asserts in `masking_constants.rs`:
  - `MASK_P == 2.264_355_2_f32` (transducer exponent)
  - `MASK_C == -0.795_497_12_f32` (phase-uncertainty scaling
    exponent; a sign flip would amplify masking 6×)
  - `D_MAX == 2.564_245_5_f32` (soft-clamp ceiling exponent)
Array constants (`CH_GAIN`, `MASK_Q`, `XCM_3X3`) remain runtime-
only for now — promoting them would add bulk for diminishing
return; the runtime tests still cover them at f32-bit precision.
Static-assert count is now 38 across 9 test files.

Tick 557 promoted 8 more scalar bit-pins:
  - `pool_scalar.rs`: `BETA_SPATIAL == 2.0`, `BETA_BAND == 4.0`,
    `BETA_CH == 4.0`, `IMAGE_INT == 0.577_918_3`,
    `JOD_A == 0.043_956_94`, `JOD_EXP == 0.930_204_27` (the
    met2jod power-law constants — `met2jod(d) = 10 - JOD_A·d^JOD_EXP`)
  - `csf_scalar.rs`: `SENSITIVITY_CORRECTION_DB == -0.279_742_33`,
    `CSF_BASEBAND_RHO == 0.1`
Each is independently load-bearing for JOD output across every
parity gate. Static-assert count is now 46 across 11 test files.

Tick 558 promoted 12 array-element bit-pins covering 4 three-entry
arrays:
  - `pool_scalar.rs`:
    - `PER_CH_W[0..3] == 1.0_f32` (still-image chrominance weights)
    - `BASEBAND_W[A,Rg,Vy] == 0.003_633_448_6 / 1.662_772_4 /
      4.118_745_3` (per-channel baseband weights)
  - `masking_constants.rs`:
    - `CH_GAIN[A,Rg,Vy] == 1.0 / 1.45 / 1.0` (RG masking-gain boost)
    - `MASK_Q[A,Rg,Vy] == 1.302_622_7 / 2.888_590_8 / 3.680_771_3`
      (per-channel masking exponents)
A typo that swapped any pair of array entries (e.g. CH_GAIN[A] ↔
CH_GAIN[Rg], muting chrominance) now surfaces at compile time.
Static-assert count is now 58 across 11 test files.

Tick 559 promoted 9 array-element bit-pins for the XCM_3X3
cross-channel masking matrix in `masking_constants.rs`. Each
entry pinned independently — the 3×3 matrix is derived from
cvvdp's published log2-space coefficient table via per-entry
2^x exponentiation, so a re-derivation that rounds differently
would surface here at compile time rather than during a parity-
test run. Static-assert count is now 67 across 11 test files.

Tick 560 promoted 13 individual tap bit-pins + 3 symmetry-pair
pins for the `PU_BLUR_KERNEL_1D` σ=3 Gaussian blur kernel in
`masking_constants.rs`. Pinning the symmetry pairs separately
(`kernel[0] == kernel[12]`, `kernel[1] == kernel[11]`,
`kernel[5] == kernel[7]`) catches a half-kernel typo that would
compile if each individual tap matched its expected literal but
the wrong half was substituted into the array. Static-assert
count is now 83 across 11 test files.

Tick 561 promoted the SRGB8_TO_LINEAR_LUT endpoint bit-pins to
static asserts in `color_scalar.rs`: `LUT[0] == 0.0_f32` and
`LUT[255] == 1.0_f32`. The IEC 61966-2-1 sRGB EOTF maps byte 0 →
linear 0 exactly and byte 255 → linear 1 exactly, and these are
the boundary cases an off-by-one byte index would silently break.
The 254 interior LUT entries remain runtime-only — they're each
derived from the sRGB EOTF formula and pinned by
`srgb_lut_matches_iec_61966_2_1_formula`, which can't lift
because the formula's branchless conditional uses `f32::powf`
(not const fn). Static-assert count is now 85 across 11 test files.

Tick 562 promoted 4 LUT-axis endpoint bit-pins to static asserts in
`csf_axes_invariants.rs`:
  - `LOG_L_BKG_AXIS[0] == -2.3010299957` (log10(0.005))
  - `LOG_L_BKG_AXIS[31] == 4.0` (log10(1e4))
  - `LOG_RHO_AXIS[0] == -1.0` (log10(0.1))
  - `LOG_RHO_AXIS[31] == 1.8061799740` (log10(64))
The 60 interior entries of each axis stay runtime-only — their
runtime invariants (monotonicity, uniform-spacing-in-log10) can't
lift without const-callable loop arithmetic. Static-assert count
is now 89 across 11 test files.

Tick 563 closed the last GAUSS5 gap: the edge taps that tick 555
skipped because the runtime test uses an abs-diff tolerance.
Since f32 arithmetic IS const-callable in stable Rust (only
`f32::PartialOrd::lt` isn't), the underlying derivation
`0.25 - KERNEL_A / 2.0` can be evaluated at compile time and
matched bit-exactly:
  - `GAUSS5[0].to_bits() == (0.25_f32 - KERNEL_A / 2.0_f32).to_bits()`
  - `GAUSS5[4].to_bits() == (0.25_f32 - KERNEL_A / 2.0_f32).to_bits()`
Plus a palindrome cross-check: `GAUSS5[0] == GAUSS5[4]`. The 5-tap
Burt-Adelson kernel is now fully bit-pinned at compile time. Static-
assert count is now 92 across 11 test files.

Tick 564 added 6 semantic ordering invariants leveraging the
observation that for positive f32 values, IEEE 754 bit-pattern
ordering matches numerical ordering — so `u32 <` (which IS const-
callable) is a sound proxy for the underlying f32 ordering:
  - `BASEBAND_W`: A < Rg < Vy (strict monotonicity across channels)
  - `MASK_Q`: A < Rg < Vy (strict monotonicity across channels)
  - `CH_GAIN`: Rg > A and Rg > Vy (chroma-boost invariant)
These catch a class of typo the individual-entry bit-pins miss: a
permutation that keeps every value intact but swaps which channel
gets which weight. Static-assert count is now 98 across 11 test
files.

Tick 565 added 3 more semantic invariants of a different flavour:
  - `MASK_C.is_sign_negative()` — phase-uncertainty exponent must
    be negative because `10^MASK_C` is an attenuator; a sign flip
    would convert the 0.16× attenuation into a 6× amplification.
    `f32::is_sign_negative` is const fn since Rust 1.83.
  - `BETA_BAND.to_bits() == BETA_CH.to_bits()` — the across-band
    and across-channel Minkowski exponents must remain equal (both
    = 4.0). A drift in one without the other breaks the symmetric-
    pool contract.
  - `JOD_EXP.to_bits() < 1.0_f32.to_bits()` — sublinear-saturation
    invariant on met2jod (`10 - JOD_A · d^JOD_EXP`). Both operands
    are positive so u32 bit-ordering is sound. A regression bumping
    JOD_EXP ≥ 1.0 would make JOD super-linear in d, changing the
    entire perceptual scale.
Static-assert count is now 101 across 11 test files.

Tick 566 added 4 more invariants extending the semantic-invariant
pattern from tick 565:
  - `BETA_SPATIAL.to_bits() < BETA_BAND.to_bits()` — BETA hierarchy
  - `BETA_SPATIAL.to_bits() < BETA_CH.to_bits()` — BETA hierarchy
    (the canonical pyramid-pool strategy raises the Minkowski
    exponent across each nesting level; the inner spatial pool is
    gentler than across-band / across-channel folds)
  - `MASK_P.is_sign_positive()` — transducer exponent must be
    positive (negative MASK_P → `pow(d, MASK_P)` → ∞ as d → 0)
  - `D_MAX.is_sign_positive()` — soft-clamp ceiling exponent must
    be positive (`10^D_MAX < 1` would collapse the clamp ceiling)
Static-assert count is now 105 across 11 test files.

Tick 567 added 9 sign-signature invariants on the
`SRGB_LINEAR_TO_DKL` matrix in `color_scalar.rs`:
  - Row 0 (A): all 3 entries `.is_sign_positive()`
  - Row 1 (Rg): [0]=positive, [1]=negative, [2]=negative
  - Row 2 (Vy): [0]=negative, [1]=negative, [2]=positive
This encodes the DKL opponent-color contract: A is weighted-
positive sum, Rg opposes R against G+B, Vy opposes B against
R+G. The per-entry value bit-pins already encode the sign
implicitly, but the sign-signature pin captures the SEMANTIC
contract directly — useful for the same documentation-of-intent
reason as the channel-ordering invariants (564-566). Static-
assert count is now 114 across 11 test files.

Tick 568 added 5 more sign-bit invariants on the remaining major
scalar constants:
  - `JOD_A.is_sign_positive()` — met2jod must decrease with d
  - `IMAGE_INT.is_sign_positive()` — multiplicative pool weight
  - `KERNEL_A.is_sign_positive()` — Burt-Adelson parameter ∈ (0, 0.5)
  - `CSF_BASEBAND_RHO.is_sign_positive()` — spatial frequency in cy/deg
  - `SENSITIVITY_CORRECTION_DB.is_sign_negative()` — calibrated
    attenuation (not amplification)
Static-assert count is now 119 across 11 test files.

Tick 569 added 9 positivity invariants on every `XCM_3X3` entry.
Each is derived in cvvdp v0.5.4 as `2^x` for some log2-space
coefficient, and `2^x > 0` always. A refactor that substituted
a different formula (e.g. `1 - exp(-x)` for an attenuation
reframe, or a sign drift in the source coefficients) could yield
negative entries while still matching the per-entry value bit-
pins. Pinning positivity directly captures the construction
rule. Static-assert count is now 128 across 11 test files.

Tick 570 added 7 positivity invariants on the unique taps of
`PU_BLUR_KERNEL_1D` ([0]..[6]; taps [7]..[12] inherit positivity
via the palindrome bit-equality pins from tick 560). The σ=3
Gaussian construction `exp(-x²/(2σ²)) / Σ` only emits positive
values; a refactor that substituted a different kernel family
(e.g. derivative-of-Gaussian, sinc with side lobes) would yield
negative taps. Pinning positivity directly captures the Gaussian
construction contract. Static-assert count is now 135 across 11
test files.

Tick 571 added 4 length-pin invariants on the per-channel CSF
sensitivity LUTs `LOG_S_O0_C1/C2/C3` in `csf_axes_invariants.rs`:
  - Each LUT length must equal `N_L_BKG * N_RHO` (32 × 32 = 1024)
  - Plus a cross-channel length-consistency pin (all 3 LUTs have
    matching length)
The CSF kernel indexes via `idx = l_bkg_i * N_RHO + rho_i` so a
size mismatch silently corrupts every per-pixel CSF query — these
pins catch the mismatch at compile time rather than as garbage
JOD output. Static-assert count is now 139 across 11 test files.

Tick 572 added the first test coverage for `GE_SIGMA`: a
bit-equality pin to cvvdp v0.5.4's `ge_sigma = 1.5` and a
positivity invariant (it's a Gaussian σ). The constant is
documented as carried for source-JSON fidelity but not yet
consumed by the still-image pipeline (eccentricity-aware paths
are future work); the pins guarantee the value stays correct
for when those paths land. Static-assert count is now 141 across
11 test files.

Tick 573 promoted 12 `CvvdpParams::PLACEHOLDER` scaffolding-field
bit-pins to static asserts in `params_placeholder_non_display.rs`:
  - csf sub-bundle: `a_peak`, `rg_peak`, `vy_peak` (all 0.0)
  - masking sub-bundle: `p=2.4`, `q=2.2`, `k=0.04`
  - pooling sub-bundle: `beta_spatial`, `beta_band`,
    `beta_channel` (all 4.0)
  - jod sub-bundle: `jod_a=10.0`, `jod_b=1.0`, `jod_c=0.30`
These fields are documented as unused-scaffolding (production
code reads from `kernels::*` consts) but they're publicly-visible
defaults that `CvvdpParams { ..PLACEHOLDER }` callers depend on.
Pinning at compile time keeps the scaffolded values stable until
they're intentionally wired through. Static-assert count is now
153 across 11 test files.

Tick 575 added 12 more invariants on `PLACEHOLDER`:
  - **Cross-bundle linkage (3)**: `PLACEHOLDER.display ==
    STANDARD_4K` — y_peak, y_black, y_refl each pinned via
    `to_bits()`. Guards against a refactor that copies the
    STANDARD_4K values into PLACEHOLDER literally (drifting if
    STANDARD_4K is later updated but PLACEHOLDER's copy isn't).
  - **Scaffolding positivity (9)**: masking.{p,q,k},
    pooling.beta_{spatial,band,channel}, jod.{jod_a,jod_b,jod_c}
    all `.is_sign_positive()`. Negative values would invert the
    expected algebra (pow singularities, pool reversal) the
    moment the fields are wired through.
Static-assert count is now 165 across 11 test files.

Tick 577 promoted the `CVVDP_COLUMN_NAME.starts_with("cvvdp_imazen_")`
runtime check to a compile-time pin via a const while-loop over
`as_bytes()`. `str::starts_with` itself isn't const fn, but
`str::as_bytes` is (since 1.39), integer comparison is trivially
const, `while` in const is stable, and `Option::is_none` (used to
gate the check on the default-form build, no `CVVDP_IMPL_TAG` env
override) is const since 1.48. Adds a length pin + a loop-body
match pin (2 logical asserts). Static-assert count is now 167
across 11 test files.

Tick 578 applied the same const-byte-loop trick to the goldens-
metadata structural invariants in `goldens_metadata.rs`:
  - MANIFEST_URL starts with `https://` (byte-prefix)
  - MANIFEST_URL ends with `.json` (byte-suffix at offset)
  - MANIFEST_URL starts with canonical R2 host
    `https://coefficient.r2.imazen.org/`
  - MANIFEST_SHA256 length == 64 (sha256 hex)
  - !GOLDEN_VERSION.is_empty()
  - GOLDEN_VERSION first byte == 'v' (the v<N> convention)
The substring `.contains` checks (golden-version path segment,
bucket subpath) and per-char hex validation stay runtime —
`.contains` requires substring search not easily const-callable.
Static-assert count is now 173 across 11 test files.

Tick 579 closed the last two goldens-metadata runtime gaps via a
const sliding-window substring-search helper (also const-callable
in stable Rust):
  - MANIFEST_URL contains `/cvvdp-goldens/` (bucket subpath)
  - MANIFEST_URL contains `/v1/` (version path segment)
The substring-search helper `bytes_contain(hay, needle)` is a
`const fn` doing the obvious O(n·m) sliding-window comparison.
That was the technique the prior tick claimed wasn't "easily const-
callable" — turns out it IS, the sliding-window inner loop is
just two more layers of the `while` + byte-comparison primitive
ticks 577-578 already used. Static-assert count is now 175 across
11 test files.

Tick 580 closed the per-char hex validation gap tick 578 left
runtime-only. `char::is_ascii_digit` / `RangeInclusive::contains`
aren't const fn, but raw u8 comparison IS — and MANIFEST_SHA256
is pure ASCII so byte-iteration covers every char correctly:
  - Every byte must satisfy `(c >= b'0' && c <= b'9') || (c >=
    b'a' && c <= b'f')` (lowercase hex)
A uppercase variant fails the case-sensitive sha2-Digest match
silently; a stray non-hex char fetches the wrong manifest. Now
both are compile-time-caught. Static-assert count is now 176
across 11 test files.

Tick 581 extracted `CACHE_DIR_SUBDIR = "zenmetrics-cvvdp-goldens"`
from `tests/common/mod.rs` to a pub const (was a magic string
inline in `cache_dir()`), then pinned 3 structural invariants on
it via the const-byte-loop primitives from ticks 577-580:
  - non-empty
  - contains "cvvdp" (disambiguation from sibling crates'
    cache dirs that all live under `~/.cache/zenmetrics-*/`)
  - all-ASCII alphanumerics or hyphen (filesystem-portable)
Static-assert count is now 179 across 11 test files. Small
refactor + pins — same shape as ticks 522-524 promoted dimension
constants from inline literals into the lib_constants module.

Tick 582 completed the tick-581 refactor by deduplicating the
remaining inline magic-string usage in `goldens_metadata.rs`'s
`cache_dir_path_embeds_golden_version` runtime test — now uses
`CACHE_DIR_SUBDIR` directly. If the subdir is renamed, the test
follows automatically and the static asserts on it still cover
the "must contain 'cvvdp'" invariant at compile time. Pure
dedup — no new static asserts.

Tick 583 promoted the `CVVDP_COLUMN_NAME.starts_with("cvvdp_")`
family-prefix check in `lib_reexports.rs` to compile time via
the const-byte-loop pattern (same as ticks 577/578/579/580).
This is the broader prefix invariant — the env-override
`CVVDP_IMPL_TAG` is intentionally a free-form discriminator
WITHIN the `cvvdp_*` namespace (pycvvdp uses `cvvdp_pycvvdp_v054`,
this crate uses `cvvdp_imazen_*`, a future Burn port reserves
`cvvdp_burn_*`); the family prefix must hold for all variants.
Also corrected the stale "`.starts_with` isn't const fn" comment
left over from tick 522. Adds 1 length pin + 1 byte-match pin
(2 logical asserts). Static-assert count is now 181 across 11
test files.

Tick 584 extracted the three duplicated const-byte-loop primitives
(`starts_with`, `ends_with`, `contains`) into a shared
`common::const_str` module in `tests/common/mod.rs`. The pattern
was duplicated across `column_name.rs`, `goldens_metadata.rs`,
and `lib_reexports.rs` (ticks 577-580, 583); each call site now
imports `common::const_str` and uses `const_str::starts_with(…)`
etc. Pure refactor — same static-assert count (181) but the
boilerplate per call site shrinks from ~10-30 lines to 1-3 lines.
No behavior change.

Tick 585 added direct unit tests for the new `common::const_str`
helpers in a new test file `const_str_helpers.rs`. 17 compile-time
`const _: () = assert!(...)` cases cover positive + negative paths
for each helper (`starts_with`, `ends_with`, `contains`), plus
edge cases (empty prefix/suffix/needle, prefix longer than
haystack, needle at start/middle/end). 6 runtime test fns mirror
the asserts so `cargo test` runners can name them in output.
Static-assert count is now 198 across 12 test files.

Tick 586 added a fourth helper `const_str::bytes_eq(a, b)` for
const slice-equality (`[u8]: Eq` isn't const-callable). Used it
to pin `GOLDEN_VERSION == "v1"` exactly in `goldens_metadata.rs`
— previously only the v-prefix was pinned. With this, a refactor
that bumps `GOLDEN_VERSION = "v2"` without also updating
`MANIFEST_URL`'s `/v1/` path segment fails to compile (instead of
passing both prefix and contains checks while silently fetching
the wrong manifest). Also adds 5 compile-time + 2 runtime tests
on the new helper. Static-assert count is now 204 across 12 test
files.

Tick 587 fixed two stale "stays runtime-only" comments in older
const blocks:
  - `column_name.rs:28-31`: tick 554 originally said
    `str::starts_with` "stays runtime-only" — but tick 577 lifted
    the prefix check, and tick 584 factored it into
    `common::const_str::starts_with`.
  - `pyramid_scalar.rs:28-31`: tick 555 originally said GAUSS5
    outer taps "stay runtime-only because they use abs+lt
    tolerance" — but tick 563 lifted them by deriving the bit
    pattern at compile time from `0.25 - KERNEL_A / 2.0`.
Comment-only updates; no code change, no behavior change. Doc
maintenance like tick 583 did for lib_reexports.rs.

Tick 588 added a new pub const `cvvdp_gpu::PYCVVDP_REFERENCE_VERSION = "v0.5.4"`
in `lib.rs` to centralize the pinned reference-version string
that previously appeared in 6+ places (`tests/parity.rs`,
`kernels/csf_lut/v0_5_4.rs` filename, csf.rs module name,
PORT_STATUS.md, CHANGELOG, requirements.txt).

The runtime test `tests/parity.rs::manifest_fetches` now sources
its expected version from `PYCVVDP_REFERENCE_VERSION` instead of
the hardcoded string — when the reference bumps, this test
follows automatically.

Also adds 3 compile-time format invariants on the new const:
non-empty, starts with `v`, contains `.` — catches a typo like
`v054` that breaks the `vX.Y.Z` convention. Static-assert count
is now 207 across 12 test files (+3 from tick 587's 204).

Tick 589 closed the requirements.txt lockstep gap from tick 588.
Pins `scripts/cvvdp_goldens/requirements.txt` at compile time
against `PYCVVDP_REFERENCE_VERSION` (strip leading `v` to match
the PyPI `cvvdp==X.Y.Z` format — note: the PyPI package is named
`cvvdp` even though the importable module is `pycvvdp`). Uses
`include_str!()` (compile-time file read) + `slice::split_first()`
(const-callable since 1.83) + `common::const_str::contains` to
verify the version substring is present.

A bump to PYCVVDP_REFERENCE_VERSION now FAILS TO COMPILE unless
requirements.txt is updated in the same commit. Closes the 6th
lockstep site documented in the PYCVVDP_REFERENCE_VERSION
docstring. Static-assert count is now 208 across 12 test files.

Tick 590 extended the lockstep coverage to the vendored LUT file:
`src/kernels/csf_lut/v0_5_4.rs`'s auto-generated header comment
`"Auto-generated from pycvvdp v0.5.4's csf_lut_weber_fixed_size.json."`
contains the full `v0.5.4` string (matches PYCVVDP_REFERENCE_VERSION
exactly — no v-stripping needed). `include_str!()` reads the full
LUT (~1000+ lines of f32 literals) at compile time and
`const_str::contains` finds the version substring. When the
reference bumps, the LUT regen procedure updates the header —
this pin catches a version mismatch between the const and the
vendored data. Static-assert count is now 209 across 12 test files.

Tick 591 closed the last include-able lockstep site:
`docs/PORT_STATUS.md`. Its "Reference version pin" section
reads "gfxdisp/ColorVideoVDP v0.5.4 (latest tag as of …)" — same
`include_str!()` + `const_str::contains` pattern as ticks 589-590.
Forces the prose documentation to update in the same commit as
the const + parity-test + requirements.txt + LUT header.

Tick 592 extended the lockstep further to the crate-level
README.md. It references `v0.5.4` in 4 places (algorithm-parity
claim, PerfMode::Strict semantics, parity-goldens feature, Status
section). Same `include_str!()` + `const_str::contains` pattern.
User-facing docs now also forced to update in lockstep. Static-
assert count is now 211 across 12 test files (5 cross-file
`include_str!()` pins on PYCVVDP_REFERENCE_VERSION:
parity-test runtime check, requirements.txt, LUT header,
PORT_STATUS.md, README.md).

Tick 593 closed the Cargo.toml feature-doc comment site. The
`parity-goldens` feature comment reads "Enables integration tests
that fetch the pycvvdp v0.5.4 goldens from R2 ..."; pinning
forces the comment to update in lockstep too. The const now has
6 cross-file `include_str!()` lockstep pins total. Static-assert
count is now 212 across 12 test files.

Tick 594 pinned `docs/CVVDP_SIDECAR_SCHEMA.md`'s "Reserved
column-name tags" table (which documents `cvvdp_pycvvdp_v054` →
"upstream pycvvdp v0.5.4"). CHROMA_DRIFT_INVESTIGATION.md is
intentionally NOT pinned — its v0.5.4 references are historical
audit material from the tick-200 chroma_shift bug hunt, not
current-state documentation; pinning would cement that historical
investigation against future reference bumps incorrectly. The
const now has 7 cross-file `include_str!()` lockstep pins.
Static-assert count is now 213 across 12 test files.

Tick 595 moved all 7 lockstep pins + the const-format invariant
block from `tests/parity.rs` (gated behind `parity-goldens`
feature) to a new always-on `tests/version_lockstep.rs` file.
Real correctness improvement: the pins now fire on every
`cargo check / test`, not only when the goldens feature is on.
Pure refactor — same pin count (213), same coverage. Test file
count grows to 13.

Ticks 596-598 (post-595 doc cleanup):
- Tick 596: removed a duplicated `#[allow(dead_code)]` outer
  attribute on `mod common;` in version_lockstep.rs. The inner
  `tests/common/mod.rs` already has `#![allow(dead_code)]`; the
  redundant outer attribute introduced a `duplicated attribute`
  warning that tick 595 had missed. Lint cleanup, no code change.
- Tick 597: rewrote the `PYCVVDP_REFERENCE_VERSION` docstring in
  `lib.rs` to list all 7 lockstep-pinned sites + 3 format
  invariants + 2 intentionally-unpinned historical docs + 2
  unpinnable Rust-identifier sites + the GOLDEN_VERSION cross-
  version-space relationship. Was listing only the 6 original
  sites from tick 588. Contract now self-documenting at the const.
- Tick 598: rewrote PORT_STATUS.md's "Reference version pin"
  bump procedure. Was listing 3 update sites (R2 prefix,
  GOLDEN_VERSION, tests/parity.rs assertion); now correctly
  documents `PYCVVDP_REFERENCE_VERSION` as the single trigger
  point + the 7-pin lockstep arc that surfaces the rest as
  compile failures.

Static-assert count unchanged at 213 across 13 test files.

Tick 600 added a 5th `common::const_str` helper:
  pub const fn count(s: &[u8], needle: &[u8]) -> usize
which counts non-overlapping occurrences of `needle` in `s`. Used
to add a LUT-channel-completeness pin to `version_lockstep.rs`:
the LUT file must contain at least 3 occurrences of `LOG_S_O0_C`
(one per channel declaration: C1, C2, C3). Catches an accidental
truncation that drops one of the channel LUTs entirely — the
per-channel length pins in `csf_axes_invariants.rs` (tick 571)
cover the LEN per channel WHEN each channel exists, but not the
"channel missing entirely" case. Also adds 8 compile-time count
asserts + 2 runtime test fns to `const_str_helpers.rs` covering
the new helper's positive / edge cases. Static-assert count is
now 222 across 13 test files.

Tick 635 — add a 1024×1024 noise parity fixture (high-frequency
distortion at deep pyramid). Mirrors the 256² noise fixture at
MAX_LEVELS=9-clamped depth. Noise is the worst-case input for
the high-freq pyramid bands (uncorrelated per-pixel, full
bandwidth) — a refactor that introduced a depth-dependent
masking or CSF bug would surface differently than at 256².

What landed:
- `scripts/cvvdp_goldens/bench_12mp_cuda.py`: new
  `synth_pair_1024_noise(w=1024, h=1024)` + fixture entry.
- `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`: new entry
  with `jod = 8.989996` (generated locally, 0.4 s wallclock).
  Lowest JOD in the synth suite — noise is the most degrading
  distortion at this magnitude.
- `crates/cvvdp-gpu/tests/predict_jod_invariants.rs`: new
  `predict_jod_matches_pycvvdp_at_1024x1024_noise` host-scalar
  parity test (inline noise construction matching 256² siblings).

Measured result: host_scalar JOD = 8.989994, pycvvdp golden =
8.989996, **|diff| = 0.000002** (~2 ULP at f32). 2500× under
the 0.005 JOD canonical tolerance.

Tick 634 — add a 1024×1024 chroma_shift parity fixture
(deep-pyramid chroma case at the MAX_LEVELS=9 clamp boundary).
Completes the **128²+256²+1024² chroma_shift triple**, pinning
chroma behavior across pyramid depths from 6 levels (128²)
through 7-8 levels (256²) to MAX_LEVELS=9-clamped (1024²). A
refactor that introduced a depth-dependent RG/VY bug would
surface at one specific depth, narrowing the regression.

What landed:
- `scripts/cvvdp_goldens/bench_12mp_cuda.py`: new
  `synth_pair_1024_chroma_shift(w=1024, h=1024)` + fixture entry.
- `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`: new entry
  with `jod = 9.665625` (generated locally, 0.4 s wallclock).
- `crates/cvvdp-gpu/tests/predict_jod_invariants.rs`: new
  `predict_jod_matches_pycvvdp_at_1024x1024_chroma_shift`
  host-scalar parity test.

Measured result: host_scalar JOD = 9.665625, pycvvdp golden =
9.665625, **|diff| = 0.000000** — bit-identical at f32 precision.

Tick 633 — add a 128×128 chroma_shift parity fixture (new
distortion type at a non-256 size). Mirrors the existing
`synth_256x256_chroma_shift` (G channel +16, R/B unchanged) at
a different pyramid depth: 6 levels at 128² vs 7-8 at 256². Tests
that the RG/VY-isolation behavior of the DKL stage is consistent
across pyramid depths — a refactor that introduced a depth-
dependent chroma bug would surface as a 128 vs 256 diff.

What landed:
- `scripts/cvvdp_goldens/bench_12mp_cuda.py`: new
  `synth_pair_128_chroma_shift(w=128, h=128)` + fixture entry.
- `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`: new entry
  with `jod = 9.663603` (generated locally via pinned `.venv`).
- `crates/cvvdp-gpu/tests/predict_jod_invariants.rs`: new
  `predict_jod_matches_pycvvdp_at_128x128_chroma_shift`
  host-scalar parity test (inline +16 G construction same as
  the 256² siblings in pipeline_color.rs).

Measured result: host_scalar JOD = 9.663603, pycvvdp golden =
9.663603, **|diff| = 0.000000** — bit-identical at f32 precision.
First non-256² coverage of the chroma-only distortion type.

Tick 632 — add an 11×19 (~209 px) TINY odd-dim synth-offset
parity fixture. Tiniest viable odd-dim pyramid case — min dim
= 11, just above `PYRAMID_MIN_DIM*2 = 8`. The pyramid is only
2 levels deep (`floor(log2(11)) - 1 = 2`), so every band
exercises edge handling. Sister to the 73×91 odd-dim fixture
(~6.6k px, 5 levels): together they pin odd-dim parity at
BOTH extremes of pyramid depth.

What landed:
- `scripts/cvvdp_goldens/bench_12mp_cuda.py`: new
  `synth_pair_11x19_offset(w=11, h=19)` + fixture entry.
- `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`: new entry
  with `jod = 9.461595` (generated locally via pinned `.venv`,
  0.3 s wallclock).
- `crates/cvvdp-gpu/tests/predict_jod_invariants.rs`: new
  `predict_jod_matches_pycvvdp_at_11x19_tiny_odd` host-scalar
  parity test.

Measured result: host_scalar JOD = 9.461595, pycvvdp golden =
9.461595, **|diff| = 0.000000** — bit-identical at f32 precision.
A future refactor to the edge-padding semantics or the pycvvdp
gausspyr_reduce parity-check bug replication (tick 206) would
surface here differently than at 73×91, narrowing the regression.

Tick 631 — add a 720×1280 (TALL HD aspect, h > w) synth-offset
parity fixture, mirroring tick 630's 1280×720 wide-HD fixture by
swapping aspect. Same total pixel count, same per-pixel
distortion, but pyramid downsample strides are width/height-
asymmetric. The two side-by-side pin width-height SYMMETRY of
the pyramid kernels: any refactor that bakes in a `w >= h`
assumption would surface as a tall-vs-wide JOD diff exceeding
the f32 noise floor.

What landed:
- `scripts/cvvdp_goldens/bench_12mp_cuda.py`: new
  `synth_pair_720x1280_offset(w=720, h=1280)` + fixture entry.
- `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`: new entry
  with `jod = 9.445360` (generated locally via pinned `.venv`,
  0.3 s wallclock).
- `crates/cvvdp-gpu/tests/predict_jod_invariants.rs`: new
  `predict_jod_matches_pycvvdp_at_720x1280_offset` host-scalar
  parity test.

Measured result: host_scalar JOD = 9.445363, pycvvdp golden =
9.445360, **|diff| = 0.000003** (~3 ULP). Notably this JOD
differs from the wide 1280×720 fixture by ~0.009 — non-trivial,
reflecting genuinely different per-band downsampling order
(both are bit-stable against pycvvdp v0.5.4 individually, so
the diff is a real pyramid-stride asymmetry, not a bug).

Tick 630 — add a 1280×720 (HD aspect, ~1 MP) **non-square**
synth-offset parity fixture. Sister to the square 1024² fixture
(tick 629): `min(w, h)=720` → `floor(log2(720)) - 1 = 8` raw
pyramid levels, NOT MAX_LEVELS=9-clamped. Together with tick
629's 1024² (which IS clamped), the pair pins both the clamped
and un-clamped pyramid-depth paths at ~1 MP.

What landed:
- `scripts/cvvdp_goldens/bench_12mp_cuda.py`: new
  `synth_pair_1280x720_offset(w=1280, h=720)` + fixture entry.
- `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`: new entry
  with `jod = 9.454182` (generated locally via pinned `.venv`,
  0.4 s wallclock).
- `crates/cvvdp-gpu/tests/predict_jod_invariants.rs`: new
  `predict_jod_matches_pycvvdp_at_1280x720_offset` host-scalar
  parity test.

Measured result: host_scalar JOD = 9.454183, pycvvdp golden =
9.454182, **|diff| = 0.000001** (1 ULP at f32 precision) — 5000×
under the 0.005 JOD canonical manifest tolerance. First parity
test in the suite exercising a non-square asymmetric pyramid
at ~1 MP scale.

Tick 629 — add a 1024×1024 (1 MP) synth-offset parity fixture,
filling the size gap between the 256² fixtures and the 4000×3000
12 MP case (16× pixel-count step). Exercises the `MAX_LEVELS=9`
pyramid-depth clamp — raw `band_frequencies` would suggest 10
levels for 1024²; `pyramid_levels` caps to 9.

What landed:
- `scripts/cvvdp_goldens/bench_12mp_cuda.py`: new
  `synth_pair_1024_offset(w=1024, h=1024)` + fixture entry
  `synth_1024x1024_offset`.
- `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`: new entry
  with `jod = 9.458330` (generated locally via pinned `.venv`
  pycvvdp 0.5.4 on CPU torch; 0.6 s wallclock).
- `crates/cvvdp-gpu/tests/predict_jod_invariants.rs`: new
  `predict_jod_matches_pycvvdp_at_1024x1024_offset` host-scalar
  parity test with the canonical 0.005 JOD tolerance.

Measured result: host_scalar JOD = 9.458330, pycvvdp golden =
9.458330, **|diff| = 0.000000** — bit-identical at f32 precision,
matching the 128×128 fixture result (tick 628). Reuses the
size-generic `common::synth_pair_with_offset_dist` helper — no
new Rust generator code needed.

Tick 628 — add a 128×128 synth-offset parity fixture, filling
the size gap between the 73×91 odd-dim and 256² fixtures with a
clean power-of-2 case (shallower pyramid, no odd-dim edge handling).

What landed:
- `scripts/cvvdp_goldens/bench_12mp_cuda.py`: new
  `synth_pair_128_offset(w=128, h=128)` reusing the 12 MP modular
  construction at a smaller size + new
  `synth_128x128_offset` fixture entry in the manifest emit block.
- `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`: new
  `synth_128x128_offset` entry with `jod = 9.456145` generated
  locally via the pinned `.venv` (pycvvdp 0.5.4 on CPU torch).
- `crates/cvvdp-gpu/tests/predict_jod_invariants.rs`: new
  `predict_jod_matches_pycvvdp_at_128x128_offset` host-scalar
  parity test against the new golden, with the canonical
  0.005 JOD tolerance.

Measured result: host_scalar JOD = 9.456145, pycvvdp golden =
9.456145, **|diff| = 0.000000** — bit-identical at f32 precision.
The shared modular-arithmetic construction with the existing
`synth_pair_with_offset_dist` helper means no new Rust generator
code; the helper is size-generic.

Tick 627 — fix two issues in tick 619's GE_SIGMA doctest:

1. Wrong tick number: said "tick 506" but the GE_SIGMA pin is at
   tick 572 (`tests/csf_axes_invariants.rs:94`).
2. Awkward phrasing: "tick 506 line" (the word "line" was a typo
   for "(tick 506)" parenthetical).

Rewrote as "Bit-pinned at compile time in
`tests/csf_axes_invariants.rs` (tick 572)" — accurate, stable
under test-file reorganization, and grammatically clean. The
GE_SIGMA pin is in a `const _: () = ...` block (not a `#[test]
fn`), so a function-name ref doesn't apply — file-level ref is
the right granularity here. Doctest passes. Doc-only change.

Tick 626 — anti-rot conversion of the only 2 remaining
src-doctest line-refs to function-name refs:

- `params.rs::pixels_per_degree` doctest:
  `(tests/display_geometry.rs:56)` →
  `tests/display_geometry.rs::ppd_matches_pycvvdp_standard_4k`
- `pipeline.rs::estimate_gpu_memory_bytes` doctest:
  `(tests/pipeline_score.rs:2077)` →
  `tests/pipeline_score.rs::estimate_gpu_memory_scales_with_pixel_count`

Line numbers go stale when test files reorganize; function-name
refs are stable until renamed (and a rename surfaces as a compile
error in the actual test, not a silent rot in a doc-comment).
Companion to tick 625's CHANGELOG line-ref fix. All 72 doctests
still pass. Docs-only change.

Tick 625 — fix stale line reference in tick 605's CHANGELOG
entry. The text claimed `tests/pipeline_score.rs:2220` for
`recommend_parallel_matches_documented_examples`, but tick 603
(landed BEFORE tick 605 but evidently mis-counted when I wrote
605's changelog) inserted u32::MAX-saturation +
image-dim-monotonicity pins above it, shifting the test to line
2212. Updated the entry to clarify the historical-then-current
shift rather than just pretending the line ref always was 2212.
Doc-accuracy fix only; no code change.

Tick 624 — add `# Examples` doctests to the `CvvdpParams` struct
+ its `PLACEHOLDER` const:

- `CvvdpParams` struct doctest demonstrates the canonical override
  pattern (`CvvdpParams { display: ..., ..PLACEHOLDER }`) for
  callers who want to keep all other defaults while customizing
  a single field.
- `PLACEHOLDER` const doctest cross-references both
  `params_placeholder.rs` and `params_placeholder_non_display.rs`
  (the two compile-time pin files) and verifies the
  display-inherits-STANDARD_4K + perf_mode-is-Strict contract.

Both not `ignore`d — pure-host construction, no GPU dependency.
Doctest count: 72 passed + 16 ignored (was 70 + 16). Continues
the sweep theme of ticks 609-623. Docs-only change.

Tick 623 — combined `# Examples` doctest on `LOG_S_O0_C1`
covering all three per-channel sensitivity LUTs
(LOG_S_O0_C1/C2/C3, A/Rg/Vy respectively). Same shared-coverage
pattern as `MASK_P` (tick 613) and `LOG_L_BKG_AXIS` (tick 618).
Pins:
- All three are 1024-entry flat arrays (32 × 32 grid =
  N_L_BKG × N_RHO).
- Every entry is finite (no NaN/Inf in the vendored JSON — log10
  of sensitivity can be negative for S < 1, but never NaN).
Cross-references tick 600's compile-time channel-completeness pin
in `tests/version_lockstep.rs`. Doctest count: 70 passed + 16
ignored (was 69 + 16). Docs-only change.

Tick 622 — add `# Examples` doctests to `Cvvdp::new` +
`Cvvdp::set_reference`. Both `ignore`d for the same docs.rs
GPU-sandbox reason. The `new` doctest exercises the most common
construction (PLACEHOLDER params → STANDARD_4K geometry). The
`set_reference` doctest shows the cached-reference pattern for
multi-DIST scoring against one REF. Cross-references the runtime
test counterparts in `tests/pipeline_score.rs` and
`tests/state_machine_independence.rs`. Doctest count: 69 passed
+ 16 ignored (was 69 + 14). Docs-only change.

Tick 621 — add `# Examples` doctest to the `Cvvdp<R>` top-level
scorer struct. `ignore`d (the runtime needs a live CUDA driver
which docs.rs sandboxes don't have — same pattern as the existing
ignored doctests on `Cvvdp::new`, `score`, etc.). The example
walks a docs.rs reader through:
- Allocating the buffer pool once for a fixed image size.
- One-shot `score(ref, dist)` returning a JOD.
- Cross-references the two cached-reference fast paths for
  multi-DIST sweeps.

The struct itself was previously documented only with a 4-line
prose docstring. Doctest count: 69 passed + 14 ignored (was
69 + 13). Docs-only change.

Tick 620 — add `# Examples` doctest to `pyramid::WeberPyramid`
struct. Pins the dual-level-count contract (`bands.len() ==
log_l_bkg.len()`) and the per-level spatial-shape match
(`log_l_bkg[k].len() == bands[k].w * bands[k].h` for non-baseband
levels). Companion to tick 619's `Band` doctest — now both
pyramid-output structs have constructor-level examples that
surface the layout contract to docs.rs readers. Doctest count:
68 → 69. Docs-only change.

Tick 619 — add `# Examples` doctests to two more public items:

- `kernels::csf::GE_SIGMA` — pin to 1.5 + positivity invariant
  (a negative sigma would invert the Gaussian eccentricity-
  falloff into an exponential blow-up at fovea). Still unused
  in the current still-image pipeline.
- `kernels::pyramid::Band` — show direct struct construction
  with the `data.len() == w × h` invariant. Used by
  `laplacian_pyramid_dec_scalar` and
  `weber_contrast_pyr_dec_scalar` to return pyramid bands; the
  doctest now surfaces the layout contract.

Doctest count: 66 → 68. Per the per-public-item sweep theme
(ticks 609-618). Docs-only change.

Tick 618 — combined `# Examples` doctest on `LOG_L_BKG_AXIS`
covering both CSF LUT axes (`LOG_L_BKG_AXIS` + `LOG_RHO_AXIS`).
Same shared-coverage pattern as `MASK_P` covering the four
masking scalars (tick 613). Pins:
- Both axes have 32 entries (= N_L_BKG = N_RHO).
- LOG_L_BKG_AXIS endpoints `[-2.301, 4.0]` (luminance 0.005–10000
  cd/m²).
- LOG_RHO_AXIS endpoints `[-1.0, 1.806]` (frequency 0.1–64 cy/deg).
- Both uniformly spaced in log10 (step uniformity over all 31
  intervals).
Cross-references `tests/csf_axes_invariants.rs`. Doctest count:
65 → 66. The three large per-channel sensitivity LUTs
(`LOG_S_O0_C1/C2/C3`) and the `GE_SIGMA` scalar still lack
doctests; the C* LUTs are 1024-entry arrays where a meaningful
doctest beyond "len() == 1024 + positive on bright photopic"
needs more thought. Docs-only change.

Tick 617 — add `# Examples` doctest to
`kernels::csf::precompute_logs_row`. Pins the return-shape
(length `N_L_BKG`) plus the bit-identity contract with
`sensitivity_scalar` at every L_bkg-axis grid point (no
interpolation needed when log_L_bkg lands exactly on a sample).
Surfaces the helper's role as the per-band precompute consumed
by `csf_apply_per_pixel_kernel` — a reader can now see what the
returned array MEANS without grepping the kernel call sites.
Doctest count: 64 → 65. Docs-only change.

Tick 616 — add `# Examples` doctest to
`kernels::csf::sensitivity_scalar`. The sister function
`sensitivity_corrected_scalar` already had a doctest covering
its multiplicative-factor relationship to `sensitivity_scalar`,
but the underlying primitive itself was undocumented at the
example level. Pins:
- Positive + finite output at standard photopic background
  (100 cd/m² → log10 = 2.0) at 4 cy/deg (CSF peak).
- High-frequency roll-off (30 cy/deg < 4 cy/deg sensitivity).
- Per-channel independence (each of A/Rg/Vy returns positive).
Doctest count: 63 → 64. Docs-only change.

Tick 615 — add `# Examples` doctest to `kernels::csf::CsfChannel`.
Pins the [A=0, Rg=1, Vy=2] discriminant ordering load-bearing for
every `channel as usize` indexing site, plus the
Copy/PartialEq/Debug derive contracts. Cross-references the
compile-time pins in `tests/csf_channel_invariants.rs`. Doctest
count: 62 → 63. Docs-only.

Tick 614 — add `# Examples` doctests to two more public
constants:

- `kernels::pyramid::KERNEL_A` — Burt-Adelson `a` parameter
  (= 0.4 in cvvdp v0.5.4). The doctest pins the bit-value AND
  shows how it propagates into `GAUSS5` (center tap = a, outer
  taps = 0.25 - a/2).
- `pipeline::PARALLEL_SAFETY_FACTOR` — pin to 1.5 + worked
  example showing `manual = free / (safety × est)` matches what
  `recommend_parallel` returns, so a reader sees the role of the
  constant in the formula without grepping the function body.

Doctest count: 60 → 62. Per the per-public-item sweep theme
(ticks 609-613). Docs-only change; `cargo test --doc` passes.

Tick 613 — combined `# Examples` doctest on `MASK_P` covering
all four scalar masking constants (`MASK_P`, `MASK_Q`, `MASK_C`,
`D_MAX`). Follows the same shared-doctest pattern already
established by `pool::BETA_CH` (one doctest covers BETA_* triple)
and `pool::JOD_EXP` (one doctest exercises JOD_A + IMAGE_INT).
Pins:
- `MASK_P > 0` (transducer exponent positivity) + bit-pin.
- `MASK_Q` is per-channel `[A, Rg, Vy]` strictly monotonic.
- `MASK_C < 0` (attenuator: `10^MASK_C` < 1).
- `D_MAX > 0` (soft-clamp ceiling: `10^D_MAX` > 100).
Doctest count: 59 → 60. Cross-references the compile-time bit-pins
in `tests/masking_constants.rs`. Docs-only change.

Tick 612 — continue the per-public-item doctest sweep (+4 more,
closing the params.rs scaffolding-struct queue from tick 611):

- `CsfParams` — all 3 sub-fields zeroed in PLACEHOLDER (production
  reads CSF straight from the vendored LUT).
- `MaskingParams` — scaffolding values (p=2.4, q=2.2, k=0.04) +
  the all-positive invariant required by future Minkowski/exponent
  algebra.
- `PoolingParams` — uniform 4.0/4.0/4.0 scaffolding triple (vs.
  production's BETA_SPATIAL=2.0, BETA_BAND=4.0, BETA_CH=4.0) +
  positivity invariant.
- `JodParams` — scaffolding values 10.0/1.0/0.30 + positivity
  invariant. Cross-reference to the v0.5.4 production values
  JOD_A ≈ 0.0439 and JOD_EXP ≈ 0.9302.

Each doctest cross-references `tests/params_placeholder_non_display.rs`
where the same values are bit-pinned at compile time. Doctest count:
55 → 59. The `CvvdpParams` struct itself + its `PLACEHOLDER` const
already had doctests in earlier ticks. Docs-only change.

Tick 611 — continue the per-public-item doctest sweep (+4 more):

- `DisplayModel` struct — field-access + spread-construct an
  HDR400 variant from STANDARD_4K.
- `DisplayModel::STANDARD_4K` const — pin the 3 field values
  with cross-reference to the runtime parity test.
- `DisplayGeometry` struct — field-access + a phone-at-arm's-
  length example showing `pixels_per_degree()` is higher than
  the 4K reference (smaller display + closer distance → tighter
  pixel grid).
- `DisplayGeometry::STANDARD_4K` const — pin the 4 field values.

Doctest count: 51 → 55. Still queued for the next pass (4 more):
`CsfParams` / `MaskingParams` / `PoolingParams` / `JodParams` —
the scaffolding-but-public structs from
`CvvdpParams::PLACEHOLDER`. Docs-only change.

Tick 610 — continue the per-public-item `# Examples` doctest
sweep from tick 609 (+3 more):

- `PYCVVDP_REFERENCE_VERSION` — format invariants (starts with
  'v', contains '.', non-empty) + the leading-'v'-strip → pip
  version trick that `scripts/cvvdp_goldens/requirements.txt`
  uses.
- `Error` — exercises `DimensionMismatch` (carries `expected` +
  `got` payload + actionable Display), the three zero-payload
  variants' Display hints, and the `?` bubble against
  `Box<dyn std::error::Error>`.
- `Result<T>` type alias — Ok/Err construction + the
  identity-`Into` `?` chaining (composes with the same return
  type without `.map_err(Into::into)`).

Doctest count: 48 → 51. Six other body-docstring-only public
items in `params.rs` (DisplayModel / DisplayGeometry / CsfParams
/ MaskingParams / PoolingParams / JodParams structs) are queued
for the next pass; the per-struct doctest needs a moment of
thought on what's load-bearing to show vs scaffolding. Docs-only
change; `cargo test --doc` passes.

Tick 609 — add `# Examples` doctests to four public crate-root
constants that had body-level docstrings but no doctest section:

- `N_CHANNELS` — pin value 3 (DKL still-image).
- `MAX_LEVELS` — pin value 9 with cross-reference to
  `tests/lib_constants.rs::max_levels_cap_at_nine` + the
  buffer-resize implication of bumping it.
- `PYRAMID_MIN_DIM` — pin value 4 and the derived 8-px minimum
  image dim (Cvvdp::new's accept threshold).
- `CVVDP_COLUMN_NAME` — pin `cvvdp_` prefix + parquet-safe charset
  (ASCII alphanumerics + `_` only), matching the contracts in
  `tests/column_name.rs`. Rendered docs now surface these
  contracts to docs.rs readers without forcing them to grep test
  files.

Doctest count: 44 → 48. Per the crate convention every public
constant should have a `# Examples` block with at least one
machine-checked assertion. Docs-only change; `cargo test --doc`
passes.

Tick 608 — tighten `srgb_byte_to_dkl_scalar`'s grayscale-chroma
doctest tolerances. The "(255,255,255) → near-zero chroma" claim
was `rg_white.abs() < a_white * 0.05` and `vy_white.abs() < a_white * 0.05`
(both 5%). Actual ratios from the bit-pinned SRGB_LINEAR_TO_DKL
matrix at (255,255,255) under STANDARD_4K are **0.36% (RG)** and
**0.98% (VY)** — pinned by `tests/color_scalar.rs:80`'s GOLDENS
table. Tightened to **1% (RG)** and **2% (VY)** respectively —
~3× and ~2× the actual values, leaving room for alternate display
models that shift `y_peak`/`y_refl` but still tight enough to
surface a matrix row-sum drift (e.g. a sign flip in row 1 or row
2 that would push the chroma residual above the new bounds).
Companion to ticks 605/606/607 (same doctest-tightening theme).
Docs-only change; doctest passes.

Tick 607 — tighten `safe_pow(2.0, 2.0)` doctest tolerance from
`< 0.01` to `< 1e-3`. The function's analytic deviation at this
point is the cross term `2·eps·x = 4e-5` (`eps = 1e-5` in the
implementation), so `0.01` was 250× looser than needed. Tightening
to `1e-3` still leaves 25× headroom for f32 rounding on the
`(x+eps)^p` evaluation, but now a refactor that raised eps by an
order of magnitude (e.g. `1e-4`) — which would push the deviation
to ~4e-4, well past `1e-3` — would surface at `cargo test --doc`.
Companion-in-spirit to ticks 605/606 (recommend_parallel,
pixels_per_degree, estimate_gpu_memory_bytes doctest tightenings)
— same theme: doctests should pin within an order of magnitude of
the actual implementation tolerance, not 100-5000× looser. Other
`< 0.01` doctest tolerances in pool.rs are CORRECTLY sized for
the L_p-norm `eps^(1/β)` tail (~3e-3 at β=2), so left alone.
Docs-only change; doctest passes.

Tick 606 — two more doctest tightenings in the same spirit as
tick 605:

1. `DisplayGeometry::pixels_per_degree` (params.rs:90) — claim was
   `assert!((ppd - 75.4).abs() < 0.5)` (loose ±0.5). The runtime
   parity test `ppd_matches_pycvvdp_standard_4k`
   (tests/display_geometry.rs:56) pins `< 1e-4`. Tightened the
   doctest to the same `< 1e-4` tolerance and updated the
   numeric reference value to the full-precision
   `75.402_449_f32`. The old loose tolerance was 5000× too
   wide — a refactor that drifted PPD by 0.3 (e.g. a sign flip
   in the geometry formula at f64-precision) would still pass
   the doctest while failing the runtime test, sending a
   confusing signal to a contributor running `cargo test --doc`.
2. `estimate_gpu_memory_bytes` (pipeline.rs:494) — the 4MP/1MP
   ratio claim was the weak `bytes_4mp > bytes_1mp`. The runtime
   pin `estimate_gpu_memory_scales_with_pixel_count`
   (tests/pipeline_score.rs:2077) pins `ratio ∈ (3.6, 4.4)`.
   Tightened the doctest to the same band, with a reference to
   the runtime test for provenance.

Docs-only change; all 44 doctests still pass.

Tick 605 — tighten `recommend_parallel`'s doctest claim from the
weak `assert!(p >= 2)` to the actual `(10..=40).contains(&p)`
range that the existing runtime test
`recommend_parallel_matches_documented_examples`
(tests/pipeline_score.rs — line 2220 at time of tick 605; tick
603's u32::MAX + image-dim-monotonicity pins shifted it to 2212
afterward) pins. The old `>= 2` underclaim
risked misleading callers into provisioning ~5× fewer concurrent
instances than the formula actually recommends for the
8 GB / 1 MP case. Also added a second example covering the
24 GB / 12 MP (3090/4090 class) configuration with its `3..=10`
range pin, matching the existing runtime
`p_24gb_12mp` test case. Docs-only change; doctest passes.

Tick 604 — compile-time bit-pin promotion of the 9
`SRGB_LINEAR_TO_DKL` element magnitudes in
`tests/color_scalar.rs`. The existing runtime test
`srgb_linear_to_dkl_matrix_matches_pycvvdp_v0_5_4` already pinned
the same 9 f32 values via `.to_bits()`; promoting to compile-time
`const _: () = assert!(...)` blocks means a refactor that drifts
any element trips at `cargo check`, before any test binary
builds. Same promotion pattern as tick 561 (LUT endpoints) and
tick 567 (sign-signature). Companion to the row-sign-signature
pin already at compile-time. Static-assert count is now 232 across
13 test files (+9 from tick 603's 223).

Tick 603 — two `recommend_parallel` contract pins surfacing
implicit-via-language-semantic guarantees that a refactor could
silently break:

1. `recommend_parallel_saturates_at_u32_max_for_unbounded_free_bytes`
   — the function's docstring claims the result is "capped at
   u32::MAX", but the cap is implicitly enforced by Rust's
   saturating `f64 as u32` cast (saturating since Rust 1.45). A
   refactor that swaps to `try_into().unwrap()` would panic on
   u64::MAX free-bytes input. Pinned at 1024² and 8×8 (smallest
   pyramid-valid).
2. `recommend_parallel_monotonically_non_increasing_in_image_dims`
   — companion to the existing `..._monotonic_in_free_bytes` pin
   (tick 234). Holding free memory constant, larger images must
   produce ≤ smaller-image parallel counts. Strictly-monotonic
   decrease would be too strong (min-1 floor flattens the curve);
   non-increasing is the load-bearing invariant. A refactor that
   inverts the division (`est * free / safety` instead of
   `free / (safety * est)`) would silently make bigger images
   *more* parallelizable, masking OOM in auto-scaling sweep code
   that picks instance count from image size.

Both pins land in `tests/pipeline_score.rs` next to the existing
recommend_parallel test cluster. Static-assert count unchanged
(these are runtime tests, not compile-time const blocks — the
saturation behavior depends on `as u32` rounding which isn't
`const fn`-callable in this configuration).

Tick 602 (`df0998d0`) — docstring-accuracy fix on the
`common::const_str` module. The tick-584 module docstring was
doubly stale: listed only 3 helpers (`starts_with` / `ends_with`
/ `contains`) when ticks 586 and 600 added `bytes_eq` and `count`,
and listed only 3 call sites (`column_name.rs`,
`goldens_metadata.rs`, `lib_reexports.rs`) when ticks 588–600
added `version_lockstep.rs` (heavy user) and `const_str_helpers.rs`
(unit tests). Rewrote to enumerate all 5 helpers with their
tick-of-introduction and all 5 current call sites with one-line
purpose notes. Docs-only change — no code edits, no assert count
change.

Tick 601 added a separate-purpose static assert in
`version_lockstep.rs` pinning that `docs/BURN_PORT_PLAN.md` keeps
its "ABANDONED" status banner. The Burn port plan was marked
ABANDONED at tick 324 after a cubek::conv2d separable spike
measured 4.32× slower than the hand-written downscale_kernel at
4000×3000 on an RTX 5070. The banner status is stable — if the
plan is genuinely revived, the file should be renamed /
restructured, not silently un-marked. NOT a PYCVVDP_REFERENCE_VERSION
lockstep pin (this is about file-content stability, not version
bumping). Static-assert count is now 223 across 13 test files.

- **Spatial-contrast contract pinned across all 6 dispatch surfaces
  (ticks 542–547).** Eighteen hypothesis-test pins capture cvvdp's
  spatial-contrast contract — three properties × six dispatch paths
  (host scalar `predict_jod_still_3ch`, GPU cold-ref `Cvvdp::score`,
  GPU cached-ref `set_reference` + `score_with_reference`, GPU warm-
  ref `warm_reference` + `compute_dkl_jod_with_warm_ref`, cubecl-cpu
  host-pool `compute_dkl_jod_host_pool`, cubecl-cpu warm-ref host-
  pool `warm_reference` + `compute_dkl_jod_host_pool_with_warm_ref`):
  - **flat-vs-flat → JOD ≈ 10** (542 host, 545 GPU cold, 546 GPU
    cached, 546 cpu host_pool, 547 GPU warm-ref, 547 cpu warm-ref
    host_pool): pure black vs pure white returns JOD ≈ 10 because
    cvvdp measures contrast *within* an image, not absolute
    differences *between* images. Both flat inputs have zero
    Weber-band energy → D = 0 → JOD = 10. Guards against an
    "absolute-difference" refactor.
  - **textured-vs-flat → JOD ≪ 10** (543 host, 545 GPU cold, 546
    GPU cached, 546 cpu host_pool, 547 GPU warm-ref, 547 cpu warm-
    ref host_pool): a 32×32 textured ref vs flat mid-gray dist
    (catastrophic blur) gives JOD = 3.4402 on host scalar and JOD =
    3.4389 on all five GPU/cpu kernel paths. The ref's missing-band
    energy converts to a non-trivial Q via masking → pool, and
    met2jod maps that below 10.
  - **noise-amplitude monotonicity** (544 host, 545 GPU cold, 546
    GPU cached, 546 cpu host_pool, 547 GPU warm-ref, 547 cpu warm-
    ref host_pool): dense alternating-sign noise at amplitudes
    {2, 8, 32} produces JOD {9.9941, 9.9670, 9.6885} — bit-identical
    across all 6 dispatch surfaces to 4 decimals. Probes the dense-
    noise regime that the sparse-distortion pin
    (`output_responds_to_distortion_magnitude`) doesn't cover.

  Cross-path agreement: bit-identical to 4 decimals across host
  scalar, GPU cold-ref, GPU cached-ref, GPU warm-ref, cpu host_pool,
  and cpu warm-ref host_pool — the only divergence is host-scalar
  3.4402 vs the five kernel-path 3.4389 on textured-vs-flat (atomic-
  add noise floor on the host pool of the scalar reference path).
  Together with the stuck-at-constant pins (ticks 494, 508, 509,
  525), the spatial-contrast contract is now load-bearing-tested on
  every dispatch surface.

  Commit hashes: `002e6958` (542), `58523a73` (543), `2ed1f4a4`
  (544), `2da874bf` (545), `35594044` (546), tick 547 in this commit.

### Added

#### ssim2-gpu

- **`Ssim2Mode` skip-map dispatch (Technique 2 of Kanetaka et al.
  IWAIT 2026)** — new `compute_with_mode` / `compute_with_reference_with_mode`
  / `compute_batch_with_mode` entry points selecting one of four
  modes (`Full`, `Lossless`, `Fast`, `Faster`, default `Faster`).
  The non-`Full` modes skip per-channel error-map + reduction
  launches (and the upstream sigma11/sigma22/sigma12/mu1/mu2
  blurs that feed them) for cells whose weight is below the
  mode's threshold. The default `compute` / `compute_with_reference`
  / `compute_batch` now route through `Faster`. `Lossless` is
  bit-identical to `Full` (only skips literal-zero weights).
  Measured on RTX 5070 at 4000×3000 (median of 10, 3 warmup):
  Full 59.9 ms → Lossless 53.5 ms (×1.12) → Fast 35.1 ms (×1.71)
  → Faster 35.2 ms (×1.70). Matches the IWAIT paper's ×1.13
  Lossless figure; the Fast / Faster GPU ratio is below the
  paper's CPU ×1.45 but goes further on the absolute reduction.
  See `crates/ssim2-gpu/docs/SKIP_MAP_AUDIT.md`. Tests:
  `crates/ssim2-gpu/tests/ssim2_skipmap_audit.rs` — verifies
  Lossless == Full bit-identical; Fast / Faster within 5e-4
  relative across the JPEG corpus.

- **`Ssim2Blur` opt-in separable FIR D=5 (Technique 1 of Kanetaka
  et al. IWAIT 2026)** — new `with_blur` / `set_blur` / `blur()`
  accessors on `Ssim2` and `Ssim2Batch` selecting one of two
  Gaussian implementations: `Iir` (default — the canonical
  Charalampidis recursive Gaussian, bit-identical to libjxl) and
  `Fir` (separable 5-tap truncated Gaussian σ=1.5, normalized,
  zero-padded at borders). The FIR is a **distinct metric** — its
  per-image scores diverge from the IIR's by design (different
  impulse-response support). Tagged via new `SSIM2_IIR_COLUMN_NAME`
  / `SSIM2_FIR_COLUMN_NAME` consts + `column_name_for_blur` helper
  so sweep tooling lands FIR scores in a separate parquet column.
  Switching blur modes invalidates the cached reference.
  Composes orthogonally with `Ssim2Mode`: `Ssim2Blur::Fir`
  combined with `Ssim2Mode::Faster` (the default skip-map mode)
  hits 17.0 ms / 12 MP warm-ref on RTX 5070 — vs 20.8 ms for
  IIR+Faster (skip-map alone) and 40.6 ms for IIR+Full (baseline).
  Tests: `crates/ssim2-gpu/tests/fir_path.rs` (FIR-specific
  determinism, monotonicity, batch parity, identical-image
  score) + `crates/ssim2-gpu/tests/blur_mode_api.rs` (API
  contract). Default IIR path bit-identical to pre-merge.

- **`fir` Cargo feature gates the FIR opt-in API** (default OFF) —
  housekeeping refactor of the FIR landing immediately above so the
  default build surface matches the pre-FIR-v2 state. Without the
  feature `Ssim2` / `Ssim2Batch` have no `with_blur` / `set_blur` /
  `blur()` knobs and the crate doesn't export `Ssim2Blur`,
  `SSIM2_FIR_COLUMN_NAME`, or `column_name_for_blur`. With
  `--features fir` everything from the prior entry is restored; the
  IIR + skip-map default path is unchanged either way (all 28
  no-fir / 50 fir tests pass).

### Changed

#### cvvdp-gpu

- **`Cvvdp::score` and `Cvvdp::score_with_reference` now route
  through the GPU pipeline** (`compute_dkl_jod`), replacing the
  host-scalar reference path. Output matches the prior host
  path to f32 noise (verified by
  `compute_dkl_jod_matches_host_scalar` at ≤ 0.005 JOD) and the
  pycvvdp v1 R2 manifest to ≤ 0.005 JOD (verified by
  `shadow_jod_gpu`). The switch was explicitly pre-promised in
  `lib.rs` ("Switching `score` over to the GPU path is the
  remaining chunk of pipeline work") and was unblocked by tick 207's
  tightened manifest-parity tolerances. Callers that need the
  all-host path can still invoke
  `host_scalar::predict_jod_still_3ch` directly;
  cpu-runtime callers use `compute_dkl_jod_host_pool`.
  Also tightened `tests/pipeline_score.rs` `cvvdp_score_matches_v1_manifest`
  from 0.05 → 0.005 JOD (measured diffs 0.0000–0.0033).
- Removed the dead `reflect()` helper in `kernels/pyramid.rs` —
  superseded in tick 206 when `gausspyr_reduce_scalar` was
  rewritten to bug-compatible zero-pad + explicit boundary
  patches matching pycvvdp.
- **Manifest-parity tolerances tightened to 0.005 JOD across the
  v1 R2 corpus** (`tests/shadow_jod.rs`). Was a per-q schedule
  (0.5 JOD at q=1, 0.1 at q=5, 0.05 at q≥20 GPU; flat 0.05 host)
  before ticks 204/206 closed the chroma_shift and 73×91 odd-dim
  drifts. Measured diffs are now 0.0000–0.0031 JOD across all 6
  q levels (host + GPU) — well within the same 0.005 tolerance
  the other parity tests use.
- `pipeline_score.rs` host-vs-GPU corpus tests
  (`compute_dkl_t_p_bands_matches_host_on_corpus_256x256`,
  `compute_dkl_d_bands_matches_host_on_corpus_256x256`) updated
  to apply the tick-204 `CSF_BASEBAND_RHO` override in their
  host reference computation — caught when running the full
  suite after tightening shadow_jod tolerances.

### Changed

#### cvvdp-gpu (tests)

- **`#![warn(missing_docs)]` crate-level lint guard** (in `lib.rs`)
  — pins the missing_docs-clean state established at tick 514.
  `warn` (not `deny`) so any newly-added undocumented public item
  surfaces during local dev + cargo doc but doesn't hard-block.
  The 5 kernel files override to `allow` via their own inner
  attribute (silencing `#[cube(launch)]` macro-emitted items
  only); kernel pub items written by humans remain protected. New
  pub items added elsewhere in the crate will trip the guard.
  Tick 516.

- **96 missing_docs warnings on `#[cube(launch)]` macro items —
  cleared** (in `kernels/{color,csf,masking,pool,pyramid}.rs`).
  Each kernel file now has `#![allow(missing_docs)]` at the top,
  immediately after the module-level `//!` block. The
  `#[cube(launch)]` macro emits a sibling module + launcher struct
  + associated `fn` per annotated kernel function; those items
  don't inherit the user's rustdoc comment and triggered 4
  warnings each × ~25 kernels × 4 emit sites ≈ 96 warnings total.
  Tick 494's attempt at putting `#[allow]` on the individual
  kernel functions didn't propagate through the macro; only the
  file-level inner attribute works. Every user-written pub item
  in these files (LUTs, scalar helpers, kernel functions
  themselves) is already documented, so the allow only suppresses
  macro-emitted noise — no user-doc coverage lost. Verified:
  `RUSTDOCFLAGS="-W missing_docs" cargo doc -p cvvdp-gpu` now
  reports 0 warnings (was 96). Tick 514.

- **`compute_dkl_jod_host_pool_with_warm_ref_runs_on_cpu_backend`**
  (in `cpu_backend.rs`) — tightened from 0.005 JOD tolerance to
  `to_bits()` bit-equality between cold-ref `compute_dkl_jod_host_pool`
  and warm-ref `compute_dkl_jod_host_pool_with_warm_ref` on the
  cubecl-cpu runtime. The cpu runtime executes every kernel
  sequentially (no GPU atomic-add nondeterminism), and the
  host_pool path uses sequential `lp_norm_mean` (no `Atomic<f32>::fetch_add`),
  so warm and cold dispatches MUST produce bit-identical f32 JOD
  on the same input. Catches a refactor that introduces accidental
  nondeterminism on the warm-ref path (e.g. accumulating across
  calls without resetting a scratch). Confirmed bit-equal at
  `0x4116b771` on the synth_pair 32×32 corpus. Tick 496.

### Added

#### cvvdp-gpu (tests)

- **`new_with_geometry_stable_under_degenerate_geometry`** (in
  `pipeline_score.rs`) — companion to tick 495's
  `ppd_does_not_panic_on_degenerate_inputs`. `Cvvdp::new_with_geometry`
  internally calls `geometry.pixels_per_degree()` to derive pyramid
  level count via `pyramid_levels`. Degenerate geometries can
  produce NaN/Inf ppd; the contract is that `new_with_geometry`
  remains total — either succeeds (potentially with a degraded
  pyramid level count) or returns `Error::InvalidImageSize`, but
  MUST NOT panic. 5 degenerate-geometry cases pinned: zero
  distance, zero diagonal, zero resolution_w, extreme close (1cm),
  extreme far (100m). All currently succeed; the test does not
  pin which path because future tightening could legitimately
  shift them between Ok and InvalidImageSize. Tick 497.

- **`ppd_does_not_panic_on_degenerate_inputs`** (in
  `display_geometry.rs`) — stability pin on
  `DisplayGeometry::pixels_per_degree` for degenerate inputs (zero
  `distance_m` / `diagonal_inches` / `resolution_w` /
  `resolution_h`, plus all-zero). The function is a total
  computation — it MAY return ±∞ or NaN for mathematically
  degenerate inputs, but it must not panic. Callers like
  `Cvvdp::compute_dkl_jod(ref, dist, ppd)` accept arbitrary ppd
  values; a future refactor that adds `assert!(distance_m > 0)` (or
  equivalent) to ppd computation would change the contract from
  "total + degraded output" to "panicking" — surface that change
  here. Observed degenerate outputs (zero distance → ppd ≈
  0.00556 = 1/180°, zero diagonal → Inf, zero resolution_*  →
  NaN). The pin doesn't assert on the specific values because a
  future formula refactor could legitimately shift ±0 ↔ ±Inf
  without breaking the no-panic guarantee. Tick 495.

- **`all_four_scoring_paths_agree_bit_equal_on_same_input`** (in
  `pipeline_score.rs`) — consolidation pin. The four documented
  public scoring paths must produce bit-identical f32 JOD on the
  same (ref, dist) input on a single Cvvdp instance: (A)
  `score(ref, dist)`, (B) `compute_dkl_jod(ref, dist, ppd)`, (C)
  `set_reference + score_with_reference`, (D) `warm_reference +
  compute_dkl_jod_with_warm_ref`. Individual pins cover pairwise
  relationships (tick 407: A↔B widening; tick 488: A↔C bit; tick
  489: D determinism). This test pins the four-way intersection: a
  refactor that, say, routes warm-ref through a subtly different
  pool kernel would surface here as D drifting from A/B/C even when
  each path's standalone determinism holds. Tick 494.

- **`new_equivalent_to_new_with_geometry_standard_4k`** (in
  `pipeline_score.rs`) — pins the documented `Cvvdp::new` rustdoc
  contract that it is "equivalent to
  `new_with_geometry(..., STANDARD_4K)`". Today the implementation
  forwards to `new_with_geometry`, but a future refactor that adds
  extra initialization to one but not the other (different default
  geometry, eager priming on the explicit-geometry path, etc.)
  would silently change the documented surface. 4 invariants:
  scoring the same (ref, dist) on two Cvvdp instances — one built
  via `new`, the other via `new_with_geometry(STANDARD_4K)` —
  produces bit-identical results on (1) `score()`, (2)
  `set_reference` + `score_with_reference()`, (3) `compute_dkl_jod`
  with the STANDARD_4K ppd, and (4) `warm_reference` +
  `compute_dkl_jod_with_warm_ref`. Tick 493.

- **clippy cleanup on `lib_reexports.rs` (tick 503 follow-up)** —
  closed two warnings introduced by tick 503's additions:
  (1) `clippy::type_complexity` on the inline
  `fn(&[u8], &[u8], usize, usize, DisplayModel, f32) -> f32` pointer
  type for `predict_jod_still_3ch` — hoisted into a `PredictJodFn`
  type alias; (2) `clippy::assertions_on_constants` on
  `assert!(N_L_BKG > 0)` — promoted to `const _: () = assert!(...)`
  static assertion that fires at compile time instead of runtime.
  Both warnings now clean; the 11 lib_reexports tests still pass.
  Tick 505.

- **`host_scalar_module_is_public` + `kernels_submodules_are_public`**
  (in `lib_reexports.rs`) — 2 new pins (11 total). Pins
  `cvvdp_gpu::host_scalar::predict_jod_still_3ch` (the canonical
  host-only reference pipeline used by shadow_jod, cpu_backend, and
  GPU-less CI environments) and the five kernels submodules
  (`color`, `csf`, `masking`, `pool`, `pyramid`) as public API
  surfaces via compile-time use sites. Existing per-kernel test
  files import specific items but no single pin verified that the
  module paths themselves remain public — catches a refactor that
  collapses one submodule into a parent or downgrades it to
  `pub(crate)`. Tick 503.

- **`params_scaffolding_types_are_public`** (in `lib_reexports.rs`)
  — adds a 9th pin to the lib re-export coverage. `CsfParams`,
  `MaskingParams`, `PoolingParams`, `JodParams` are documented as
  scaffolding for a planned "load parameters from vendored cvvdp
  JSON" path. They have no other test importing them — without
  this pin a future refactor that downgrades them to `pub(crate)`
  or removes them as unused would break the planned path silently.
  Compile-time use site via `cvvdp_gpu::params::{CsfParams, ...}`
  plus a touchpoint via `CvvdpParams::PLACEHOLDER` sub-bundle
  access. Tick 502.

- **`lib_reexports.rs` extended** — adds 3 new pins (8 total, was
  5): (1) `cvvdp_type_reexport_resolves` — `Cvvdp<R>` is the main
  scoring type; without this pin, a future refactor that moves it
  behind a feature gate or into a private module would break every
  downstream caller (zen-metrics-cli's `CvvdpBatchScorer` references
  `cvvdp_gpu::Cvvdp` directly); (2)
  `lib_constants_reexport_match_their_originals` — pins `N_CHANNELS`
  (3), `MAX_LEVELS` (9), `PYRAMID_MIN_DIM` (4), and
  `CVVDP_COLUMN_NAME` (prefix `cvvdp_`) against their documented
  values, also as a use-site pin; (3) `error_and_result_reexport_resolve`
  — `Error` and `Result<T>` are how callers see method failures,
  both must be reachable from the crate root. Tick 501.

- **`manifest_url_uses_cvvdp_goldens_bucket_subpath`** (in
  `goldens_metadata.rs`) — pins the crate-specific
  `/cvvdp-goldens/` bucket subpath on `MANIFEST_URL`. Sibling crates
  (zensim-gpu, butteraugli-gpu, dssim-gpu, ssim2-gpu) all publish
  their goldens to the same host under different subpaths
  (`/zensim-goldens/`, etc.). A refactor that accidentally swapped
  the bucket subpath would still pass the host check (tick 519) +
  version-segment check + scheme/suffix checks, but fetch a sibling
  crate's manifest. Tick 520.

- **`manifest_url_uses_documented_r2_host`** (in
  `goldens_metadata.rs`) — pins the canonical R2 host
  (`https://coefficient.r2.imazen.org/`) on `MANIFEST_URL`. Closes
  a gap where a refactor to a different CDN bucket on a different
  cloud, or a localhost dev mirror, would pass the existing
  structure checks (https scheme + .json suffix + `/v1/` segment +
  64-char hex sha256) yet fetch the wrong manifest. The host
  migration coordination is forced by requiring this pin to update
  in the same commit as the URL change. Tick 519.

- **`perf_mode_fast_matches_strict_on_gpu_host_pool`** (in
  `pipeline_score.rs`) — third leg of the PerfMode no-op contract.
  Existing coverage: `perf_mode_fast_matches_strict_today` (tick 322
  + 324) — GPU pool path with 1e-4 tolerance against atomic-add
  noise; `perf_mode_fast_matches_strict_on_cpu_host_pool` (tick 327)
  — cpu-runtime + host_pool path with bit-equality. This fills the
  missing combination: **GPU runtime + host_pool path** (e.g.
  CudaRuntime calling `compute_dkl_jod_host_pool`). The host_pool
  variant reads D bands back to host then folds via sequential
  `lp_norm_mean` — no GPU atomic-add involved, so bit-equality
  should hold even on a GPU runtime. Verified bit-equal at
  `0x40da3d6b` on 64×64 deterministic input. Catches a refactor
  that, say, makes Fast mode swap in a different host-fold
  accumulation order on the GPU runtime. Tick 512.

- **`compute_dkl_jod_host_pool_with_warm_ref_distinguishes_v1_corpus_q_levels`**
  (in `pipeline_score.rs`) — fourth-leg stuck-at-constant pin
  covering the warm-ref host_pool path (the batch-scoring fast path
  used by cubecl-cpu / Metal-compatible production workers). Same
  strict-separation contract as the GPU sibling (tick 508) and
  cold-host_pool sibling (tick 509): scoring v1 corpus at
  q ∈ {1, 20, 90} produces strictly increasing JOD with ≥ 0.01 JOD
  adjacent-level gap. Catches a refactor that caches the wrong DIST
  intermediate on the warm-ref host_pool path and silently breaks
  batch CPU scoring without showing up on the warm/cold-equality
  pin (which would still match within tolerance even if BOTH
  collapsed). Observed scores identical to GPU/cold-host_pool
  paths. Tick 525.

- **`cvvdp_host_pool_distinguishes_v1_corpus_q_levels`** (in
  `pipeline_score.rs`) — sibling to tick 508 for the host_pool
  scoring path (`compute_dkl_jod_host_pool`, the cubecl-cpu /
  Metal-compatible path). Same strict-separation contract:
  scoring v1 corpus at q ∈ {1, 20, 90} on the host_pool path
  produces strictly increasing JOD with ≥ 0.01 JOD adjacent-level
  gap. The host_pool path uses sequential `lp_norm_mean` instead
  of GPU atomic-f32 pool — a refactor that breaks distortion
  discrimination on one path doesn't automatically break it on the
  other, so pin both. Observed identical scores to GPU path
  (q=1→7.65, q=20→9.71, q=90→9.99) — consistent with tick 208's
  GPU/host_pool agreement pin. Tick 509.

- **`cvvdp_score_distinguishes_v1_corpus_q_levels`** (in
  `pipeline_score.rs`) — stuck-at-constant pin for the GPU
  `Cvvdp::score` path. Asserts that scoring the v1 corpus at
  q ∈ {1, 20, 90} produces strictly increasing JOD values with
  ≥ 0.01 JOD separation between adjacent levels. Catches a refactor
  that collapses pipeline output (e.g. forgets to route DIST
  through CSF, returns the REF-against-REF JOD uniformly, or drifts
  all scores toward a midpoint within the 0.005 manifest tolerance).
  The existing `cvvdp_score_matches_v1_manifest` pin (tick 207) is
  partly redundant for the BAD case (a pipeline collapsed to a
  single value would fail manifest tolerance at some q), but not
  for the GOOD case where scores stay within tolerance but lose
  discriminative power. Host-scalar sibling: tick 434's
  `predict_jod_invariants` "responds to distortion magnitude" pin.
  Observed gaps: q=1→7.65, q=20→9.71, q=90→9.99. Tick 508.

- **`two_fresh_cvvdp_instances_produce_bit_equal_jod`** (in
  `pipeline_score.rs`) — pin cross-instance determinism. Two
  `Cvvdp::new` calls with the same (width, height, params,
  geometry) scoring the same (ref, dist) pair MUST produce
  bit-identical JOD (within atomic-add tolerance on GPU).
  Within-instance determinism is pinned by
  `score_is_deterministic_across_repeated_calls` (tick 411); the
  cross-instance contract is independent — catches a refactor that
  accidentally shares state via `static` / `thread_local` / a
  process-global counter, or that uses non-deterministic allocation
  order to seed kernel blocks (e.g. via hashmap iteration). This
  would silently break batch scoring across multiple
  CvvdpBatchScorer instances on a sweep worker. Tolerance set to
  1e-4 to accommodate the documented GPU atomic-add nondeterminism
  (tick 324); first run observed |diff| = 0 (bit-equal in this
  sample). Tick 499.

- **`cvvdp_score_smoke_at_extreme_aspect_ratio`** (in
  `pipeline_score.rs`) — end-to-end GPU smoke at extreme aspect
  ratios. Tick 498 covered 128×8 + 8×128 (16:1 ratio at boundary
  on one side). Tick 511 extended to 1024×8 + 8×1024 (128:1
  ratio — stresses any width-axis-specific tiling assumption that
  the 16:1 case doesn't exercise). The tick-491 8×8 boundary smoke
  covers the minimum-square case; this covers asymmetric strips.
  `pyramid_levels` is bounded by `min(w, h).ilog2()` — a pyramid
  construction that accidentally defaults to `max(w, h).ilog2()`
  (= 7 instead of 3) at the asymmetric edge would surface here as
  NaN/Inf JOD or an InvalidImageSize error. 4 aspects × 4
  invariants each: (1) identity `score(ref, ref) ≈ 10` within
  1e-3; (2) non-trivial perturbation produces finite JOD in
  `[0, 10]` strictly less than identity. Tick 498, 511.

- **`cvvdp_score_smoke_at_pyramid_min_boundary`** (in
  `pipeline_score.rs`) — end-to-end GPU smoke test on the minimum
  supported dimensions (8×8 = `PYRAMID_MIN_DIM × 2`). Existing
  `invalid_image_size_surfaces_on_too_small_dims` only verifies
  that `Cvvdp::new(8, 8)` returns Ok — it doesn't verify any scoring
  path works at boundary dims. `predict_jod_still_3ch` invariants
  (tick 434) covered 8×8 on the host-scalar path; this pins the GPU
  equivalents. 4 invariants: (1) identity contract `score(ref, ref)
  ≈ 10` within 1e-3; (2) non-trivial perturbation produces finite
  JOD in `[0, 10]` strictly less than the identity JOD;
  (3) `set_reference` + `score_with_reference` works at boundary
  dims AND is bit-equal to direct `score()` (extends tick 488 pin
  to boundary); (4) `warm_reference` +
  `compute_dkl_jod_with_warm_ref` works at boundary dims with
  finite JOD in `[0, 10]`. A pyramid-construction bug at boundary
  dims (degenerate zero-channel band, off-by-one in dispatcher
  launch geometry, halving-loop regression) would surface here as
  a panic / NaN. Tick 491.

- **`compute_dkl_jod_with_warm_ref_is_deterministic_across_repeated_calls`**
  (in `pipeline_score.rs`) — 2 invariants on the warm-ref fast path
  (the `warm_reference` + `compute_dkl_jod_with_warm_ref` pattern
  that `CvvdpBatchScorer` uses for batch DIST scoring on vast.ai
  workers — the hottest call shape in the sweep): (1) warm-ref
  scoring twice on the same dist returns bit-identical `f32` JOD
  (`to_bits()` equality); (2) an intervening warm-ref call on a
  different dist does not poison per-call scratch — first and third
  dist_a calls remain bit-equal. Sibling pin to
  `score_with_reference_is_deterministic_across_repeated_calls`
  (tick 488) and `score_is_deterministic_across_repeated_calls` —
  completes bit-determinism coverage for all three scoring paths.
  Tick 489.

- **`score_with_reference_is_deterministic_across_repeated_calls`**
  (in `pipeline_score.rs`) — 3 invariants on the `set_reference` /
  `score_with_reference` cached fast path: (1) calling the cached
  path twice with the same dist returns bit-identical `f64` JOD
  (`to_bits()` equality); (2) bit-equal to the direct
  `Cvvdp::score(ref, dist)` path — strengthens the existing
  `score_with_reference_matches_score` (which used a 1e-6 tolerance)
  to the documented "match exactly" contract from tick 213; (3) an
  intervening cached-path call on a different dist does not poison
  the per-call state — first and third dist_a calls remain
  bit-equal. Sibling pin to
  `score_is_deterministic_across_repeated_calls` for the cached path
  that `CvvdpBatchScorer` relies on. Tick 488.

- **`state_machine_independence.rs`** — 5 invariant pins on the
  `Cvvdp::set_reference` / `Cvvdp::warm_reference` cache state
  machine. Pins (1) fresh `Cvvdp::new()` surfaces `NoCachedReference`
  *and* `NoWarmReference` from the two fast paths independently;
  (2) `set_reference` does NOT prime warm state (dual of tick 238's
  `set_reference_does_not_invalidate_warm_state`); (3) `warm_reference`
  does NOT prime the set_reference cache; (4) one-shot
  `Cvvdp::score` does NOT pollute either cache; (5) one-shot
  `compute_dkl_jod` does NOT pollute either cache. Catches a future
  "eager upload" refactor that would silently change the documented
  fast-path-error surface. Tick 486.

#### cvvdp-gpu (docs)

- **README "GPU memory budgeting" section** — documents the new
  `estimate_gpu_memory_bytes` + `recommend_parallel` +
  `PARALLEL_SAFETY_FACTOR` API surface added in ticks 398-399.
  Includes a size-vs-budget table at standard 4K geometry
  showing PARALLEL caps for 8 GB and 24 GB GPUs at six image
  sizes (64² through 12 MP), a code example for how a sweep
  worker should derive `PARALLEL`, and guidance on when to
  tighten / loosen the safety factor (warm-ref batches → 1.2;
  mixed CPU+GPU process → 2.0). Tick 400, 253899ab.

#### cvvdp-gpu (api)

- **`cvvdp_gpu::recommend_parallel(free_gpu_bytes, width, height)
  -> u32`** + **`PARALLEL_SAFETY_FACTOR`** const — bundles
  `estimate_gpu_memory_bytes` with the documented 1.5× safety
  factor so callers don't have to maintain the constant
  themselves. Returns the maximum number of `Cvvdp` instances
  that should fit on the GPU, with a `max(1)` floor (a single
  instance always gets to attempt scoring; OOM after that is the
  caller's signal to back off to host_pool or smaller images,
  not a "no work" signal). Returns 0 only when image dims are
  invalid or `free_gpu_bytes == 0`. Worked examples documented in
  the rustdoc + pinned by `recommend_parallel_matches_documented_examples`
  (8 GB / 1024² → PARALLEL in [10, 40]; 24 GB / 12 MP → PARALLEL
  in [3, 10]). `PARALLEL_SAFETY_FACTOR = 1.5` is exported so
  callers with different workload mixes can compute a tighter
  cap manually (warm_reference batches → ~1.2; mixed CPU + GPU
  process → ~2.0). Tick 399, 14d95cb4.

- **`cvvdp_gpu::estimate_gpu_memory_bytes(width, height) -> Option<usize>`**
  — static-analysis predictor for the GPU memory `Cvvdp::new` will
  allocate. Sums every persistent buffer: source u32 bytes, three
  full pyramids (`gauss_ref` + `bands_ref` + `bands_dis`), 6 ×
  d_scratch planes per level, weber_scratch (6 fine + 4 v-scratch
  per non-baseband level), partials, baseband log_l_bkg, srgb_lut,
  logs_row — using ceil-div halving to match the actual allocator
  layout (tick 175 ceil-div + tick 208 d_scratch + tick 240 pre-
  bundled handles). Worked-example table at standard 4K geometry:
  64² = 0.8 MB, 256² = 13 MB, 512² = 52 MB, 1024² = 208 MB,
  2048² = 833 MB, 4096×3072 (12 MP) = 2.5 GB. Use to cap fleet
  concurrency: `PARALLEL = floor(free_gpu_bytes / (1.5 *
  estimate))` — at 1024² on an 8 GB GPU that's 25, on 24 GB it's
  76. Returns `None` below the [`PYRAMID_MIN_DIM`] × 2 = 8×8
  threshold (same precondition as `Cvvdp::new`). Pinned by four
  tests in `tests/pipeline_score.rs`: below-threshold, pixel-count
  scaling (4× pixels → 4× ±10% bytes), order-of-magnitude at
  three sizes, and a worked-example concurrency-cap calc on an
  8 GB GPU. Includes `examples/cvvdp_mem_table.rs` for operators
  to probe the table on their own hardware. Tick 398, 9a30d97f.

#### scripts/sweep (cvvdp-backfill — deployment hardening, ticks 353-376)

Long arc of fleet deployment hardening between ticks 353 (backend
support docs) and 376 (CUDA 12.4 SDK build). 18 fleet attempts in
total, all destroyed pre-completion — each surfacing one or two
real defects that ship as commits. Summarized here because
walking every fleet's failure log into the changelog one-by-one
wouldn't aid future operators; the artifacts that survived the
arc are:

  - **`Cvvdp::compute_dkl_jod` / `_with_warm_ref` / `score` docs**:
    explicit "# Backend support" sections documenting the
    `Atomic<f32>::fetch_add` trap (cubecl-cpu PANICS at launch;
    Metal silently no-ops → JOD=10 for any input). Tick 353-354,
    50fb88ca/b909365b.
  - **`scripts/sweep/cvvdp_backfill/status.sh`** — at-a-glance
    fleet progress aggregator (manifest size + heartbeats +
    sidecar counts with %). Tick 357, ad8a3031.
  - **`assert_parity.py` anti-flatline check** — detects the
    silent-failure mode where score-pairs writes the same value
    for every row. Pairs with new `imazen_stats`/`pycvvdp_stats`
    fields in finalize.sh's manifest. Tick 358, b5b6f4cf.
  - **README §Documentation pointer** to the cvvdp-backfill
    operator runbook. Tick 359, 43cd69a3.
  - **`chunk_worker.sh` ENTRYPOINT override** —
    `docker run image zen-metrics ...` was hitting the production
    image's `zen-metrics-worker` ENTRYPOINT. `--entrypoint
    /usr/local/bin/zen-metrics` bypasses. Tick 360, b28e1f0b.
  - **`finalize.sh` non-destructive ~/.aws/credentials write** —
    used to overwrite the local developer box's `[default]`
    profile. Now idempotent-append. Tick 362, 9455eee8.
  - **`onstart_cvvdp_backfill.sh` apt-get install docker.io** —
    boot bootstrap was bailing on missing docker. Tick 363,
    6016441e.
  - **`onstart_cvvdp_backfill_imazen.sh` + `launch_imazen.sh`** —
    single-image fleet path (no docker-in-docker). vast.ai SSH
    instances don't allow privileged dockerd; this variant boots
    the zen-metrics-sweep image directly and skips pycvvdp
    entirely. Ticks 364-365, 8f81895a.
  - **`chunk_worker.sh` R2() wrapper** — bare `s5cmd` fell through
    to AWS `[default]` profile. Now all calls go through a
    wrapper that pins `--profile r2 --endpoint-url`. Tick 365,
    7f3a3cb8.
  - **`chunk_worker.sh` GROUPS → GROUP_LINES rename** — CRITICAL
    fix. `GROUPS` is a bash special array (current process's
    supplementary group IDs); string-assigning to it exits 1 and
    corrupts the value. Every chunk_worker invocation pre-fix
    silently died at this line. Tick 367, dc794de2.
  - **`Dockerfile.sweep.v13` CUDA SDK pin** — cubecl-cuda's
    `cudarc` 0.19.4 deps with `cuda-version-from-build-system` +
    `fallback-latest` was binding to `cuCoredumpDeregisterCompleteCallback`
    (cuda-13020 cfg-gated symbol that doesn't exist in any
    released NVIDIA driver yet). Builder stage now installs
    `cuda-nvcc-12-4` + `cuda-cudart-dev-12-4` so cudarc emits
    cuda-12040 features — compatible with driver 550+ which
    covers ~all vast.ai boxes. Tick 372 / 376, daa0d82e + cuda124.
  - **`scripts/sweep/cvvdp_backfill/STATUS.md`** — diagnostic
    chain captured so future operators don't re-burn 14 fleet
    attempts. Tick 370/372.

#### scripts/sweep (cvvdp-backfill pipeline — PINNED TASK)

End-to-end vast.ai fleet pipeline to backfill cvvdp scores
(`cvvdp_imazen_*` + `cvvdp_pycvvdp_v054` columns) onto the
2.37M-row unified parquet store at
`/mnt/v/zen/zensim-training/2026-05-07/unified/`. Six scripts
shipped across six commits + an operator runbook:

- `scripts/sweep/generate_cvvdp_backfill_chunks.py` — reads the
  7 unified parquets, splits into ~23,747 chunks of 100 rows
  each, emits `chunks.jsonl` with input_parquet, row_range,
  image_basenames, and per-impl R2 sidecar paths (`d2eb0f7c`,
  tick 336).
- `scripts/sweep/cvvdp_backfill_chunk_worker.sh` — per-chunk
  worker. Downloads input_parquet from R2, syncs basenames,
  slices parquet, groups rows by (codec,q,knob_tuple),
  re-encodes via `zen-metrics sweep --pairs-tsv`, scores in
  both impls (`score-pairs` + `pycvvdp_worker.py`), uploads
  sidecars. Host-binary OR docker-image execution modes
  (`87deac34`, tick 337).
- `scripts/sweep/onstart_cvvdp_backfill.sh` — vast.ai instance
  entry point. 239 lines: installs s5cmd+jq+docker, pre-pulls
  ZEN_METRICS_IMAGE + PYCVVDP_IMAGE, heartbeats, downloads
  chunks.jsonl + worker.sh, processes via `xargs -P $PARALLEL`
  with R2 atomic-claim (`32a3b64a`, tick 338).
- `scripts/sweep/cvvdp_backfill/launch.sh` — host-side fleet
  launcher modeled on `v15/launch_gpu.sh`. Boots ubuntu:24.04,
  bootstrap pulls onstart from R2. Defaults: N_BOXES=6,
  MAX_DPH=0.30, MIN_RAM_GB=16, MIN_DISK_GB=40 (for the 6.5 GB
  pycvvdp image) (`c572c192`, tick 339).
- `scripts/sweep/cvvdp_backfill/finalize.sh` — post-fleet
  consolidation. 253 lines: R2-syncs per-chunk sidecars,
  groups by (impl, source_stem), concatenates via
  `pyarrow.concat_tables`, emits parity TSV per source +
  manifest.json with per-source row counts and parity stats
  (`09512676`, tick 341).
- `scripts/sweep/cvvdp_backfill/README.md` — 213-line operator
  runbook with ASCII pipeline diagram, 6-step quick-start,
  docker image specs, 5 troubleshooting cases, "when NOT to
  use this" guidance vs v15 dispatcher (`cbf218d9`, tick 342).
- `scripts/sweep/cvvdp_backfill/assert_parity.py` — optional
  automation gate that consumes finalize.sh's `manifest.json`
  and exits non-zero on threshold violation. Defaults match
  the smoke-tested n=4 sentinel: `mean/median ≤ 0.10 JOD`,
  `max ≤ 0.50 JOD`. Six-fixture smoke-verified across
  pass/fail/null-tolerated/null-required-fails/only-sources
  scoping/json-summary write (`252ee704`, tick 344).

#### cvvdp-gpu (docs)

- **`CVVDP_TRACE` / `CVVDP_TRACE_WEBER` debug env vars** are now
  documented in lib.rs's crate-level docstring under a new
  "Debug tracing env vars" section. Both vars existed in
  pipeline.rs dispatch helpers but only `CVVDP_TRACE_WEBER` had
  a user-facing docstring — `CVVDP_TRACE` was discoverable only
  via grep or by reading a benchmark MD. The new section lists
  the exact stderr line shapes each var emits, verified against
  the `if trace` blocks in pipeline.rs (`9fb0c569`, tick 347).

#### cvvdp-gpu (tests, extension)

- **`tests/predict_jod_invariants.rs`** — 2 new tests extending the
  square-only coverage: `non_square_dimensions_are_supported`
  (32×16, 16×32, 64×24, 24×64) and `odd_dimensions_are_supported`
  (13×17 prime-ish, 15×15 odd-square, 73×91 — the historical tick
  206 regression case from pycvvdp's `x.shape[-2]` parity quirk in
  `gausspyr_reduce_scalar`). Both pin: identical → ≈10 within 1e-2;
  perturbed → finite < ident + 1e-3. Tick 437, ec79dc49.

#### cvvdp-gpu (docs)

- **Struct-field docstrings** — added field docs for
  `Band {w, h, data}`, `WeberPyramid {bands}`, `JodParams
  {jod_a, jod_b, jod_c}`, `DisplayGeometry {resolution_w,
  resolution_h}`, and `CvvdpParams {display, csf, masking,
  pooling, jod}`. The `CvvdpParams` field docs explicitly note
  that csf/masking/pooling/jod are scaffolding placeholders unused
  by production code (which reads `kernels::*` consts). Drops
  rustdoc missing-docs warnings from 110 → ~88 (rest are
  `#[cube(launch)]` macro-emitted items, not user-writable).
  Tick 481, b756adce.

- **`PU_PADSIZE` doctest** — added `# Examples` confirming the
  threshold value (6) and the branch-condition contract
  (`phase_uncertainty_band` at 6×6 takes the no-blur branch because
  the check is `> PU_PADSIZE` not `>=`). Tick 480, 6b542aaa.

#### cvvdp-gpu (docs, fix)

- **3 pre-existing `no_run` doctests on `Cvvdp::score`,
  `score_with_reference`, `compute_dkl_jod_with_warm_ref`** changed
  to `ignore` — they assumed a default-features build with cuda/wgpu/
  cpu, where `Backend = cubecl::cuda::CudaRuntime` resolves. Under
  `cargo test --doc --no-default-features` the type alias was empty
  and all three failed to compile. `ignore` preserves the
  documentation while skipping the compile-only step. Tick 479, 949daf00.

#### cvvdp-gpu (docs)

- **`BETA_CH` doctest** — added `# Examples` (covers all 3 pool
  Minkowski exponents): `BETA_SPATIAL == 2.0` (RMS), `BETA_BAND ==
  4.0`, `BETA_CH == 4.0`; spatial is the gentler exponent. Tick 478, 18d601ad.

- **`JOD_EXP` doctest** — added `# Examples` (on `JOD_EXP`; also
  cross-references `JOD_A` and `IMAGE_INT`): `met2jod(1.0)` matches
  `10 - JOD_A * 1^JOD_EXP` algebra within 1e-5, IMAGE_INT lives in
  `(0, 1)`. Tick 477, c02a797c.

- **`PER_CH_W` doctest** — added `# Examples` showing 3 channels
  all at 1.0 (no per-channel attenuation at the pool stage; chroma
  weighting happens earlier via `masking::CH_GAIN`). Tick 476, e0369c79.

- **`CH_GAIN` doctest** — added `# Examples` showing 3 channels,
  A/Vy at 1.0 passthrough, Rg boosted at 1.45 (cvvdp's "ch_chrom_w"
  for the red-green axis). Tick 475, 246a862a.

- **`N_RHO` doctest** — added `# Examples` covering both axis-size
  constants: `N_L_BKG == 32`, `N_RHO == 32`, and `LOG_L_BKG_AXIS`
  / `LOG_RHO_AXIS` lengths match (the LUTs are 32×32 = 1024 entries).
  Tick 474, da86cf51.

- **`SENSITIVITY_CORRECTION_DB` doctest** — added `# Examples`:
  small negative dB, linear factor `10^(DB/20)` lands in `[0.9, 1.0)`
  (≈ 0.9684 attenuation). Tick 473, 83c70385.

- **`CSF_BASEBAND_RHO` doctest** — added `# Examples` showing the
  hard-coded `0.1` cy/deg and that it's below the typical geometric
  baseband rho (~0.19 cy/deg at standard 4K + 256² — the tick-204
  pycvvdp parity override). Tick 472, 79501377.

- **`BASEBAND_W` doctest** — added `# Examples` showing 3 positive-
  finite entries with chroma dominance at baseband (`[2] > [0]`,
  `[1] > [0]` — low-spatial-freq luminance is below CSF threshold).
  Tick 471, 5f4066a3.

- **`PU_BLUR_KERNEL_1D` doctest** — added `# Examples` showing 13
  taps, symmetric around center via `to_bits()`, sum-to-1 (DC
  preservation, σ=3 Gaussian), center > 5× tail magnitude. Tick 470, a9849af7.

- **`GAUSS5` doctest** — added `# Examples` showing 5 taps,
  symmetric around center via `to_bits()` equality, DC preservation
  (sum to 1 within 1e-6), center tap equals `KERNEL_A`. Tick 469, 800474bd.

- **`XCM_3X3` doctest** — added `# Examples` showing 3×3 shape,
  all entries positive-finite, A-to-A self-coupling dominance
  `[0][0] > 0.5` (matrix orientation pin). Tick 468, e4881f9b.

- **`SRGB8_TO_LINEAR_LUT` doctest** — added `# Examples` showing
  length=256, endpoints `[0]==0.0` / `[255]==1.0`, strict monotonicity
  across the 256-entry table. Tick 467, 88210899.

- **`SRGB_LINEAR_TO_DKL` doctest** — added `# Examples` showing
  row-sum invariants: A row sums in [0.5, 2.0] (luminance gain);
  RG and VY row-sum absolute values < A row sum (DKL chroma rows
  mean-zero by construction on equal-energy input). Tick 466, a1dc45bd.

- **`DisplayGeometry::pixels_per_degree` doctest** — added
  `# Examples` showing standard 4K → ≈ 75.4 ppd (within 0.5) and
  realistic-range invariant 5..=500. Tick 465, 166aaf79.

#### cvvdp-gpu (tests)

- **`tests/lib_reexports.rs`** — 5 pins on the `lib.rs` re-export
  surface: `PerfMode::default()` resolves, `CvvdpParams::PLACEHOLDER`
  resolves, and `PARALLEL_SAFETY_FACTOR` / `estimate_gpu_memory_bytes` /
  `recommend_parallel` re-exports each match the original
  `pipeline::*` value/output. A refactor that drops one of these
  re-exports — or feature-gates them — trips here before silently
  breaking downstream callers. Tick 464, 7eb7c956.

#### cvvdp-gpu (examples)

- **`examples/cvvdp_mem_table.rs`** — refactored to use the public
  `recommend_parallel` function instead of duplicating the `mem /
  (1.5 × est)` math inline. Output is identical. Added module-level
  docstring describing the example's purpose + invocation. Tick 463, 503fcf7b.

#### cvvdp-gpu (tests, lint)

- **Clippy clean-up across tick 416-461 test files**:
  `weber_pyramid_invariants` + `do_pooling_invariants` rewrap to
  `RangeInclusive::contains`; `csf_channel_invariants` +
  `perf_mode_invariants` add per-statement `#[allow(clippy::clone_on_copy)]`
  on the deliberate `.clone()` exercise of the Clone trait;
  `csf_axes_invariants` adds module-level
  `#[allow(clippy::excessive_precision)]` (intentional pycvvdp f64
  digit preservation); `mult_mutual_band_invariants` adds module-level
  `#[allow(clippy::needless_range_loop)]` (intentional per-channel
  iteration); `laplacian_pyramid_invariants` drops a `.map().enumerate().map()`
  no-op. Tests pass identically. Tick 462, 54d791a1.

#### cvvdp-gpu (docs)

- **`mult_mutual_band` doctest** — added `# Examples`: 8×8 input
  with T == R → output identically zero bit-exact across all 3
  channels and all 64 pixels (the trivial-zero-diff contract).
  Tick 461, 8f8609c7.

- **`weber_contrast_pyr_dec_scalar` doctest** — added `# Examples`:
  16×16 + n_levels=3 → 3 bands and 3 log_l_bkg vectors; baseband
  log_l_bkg is bit-constant (replicated scalar mean). Tick 460, 23b64e11.

- **`laplacian_pyramid_dec_scalar` doctest** — added `# Examples`:
  16×16 + n_levels=3 → 3 bands at 16×16/8×8/4×4 dims; each band's
  `data.len() == w * h`. Tick 459, fa97e6c8.

- **`phase_uncertainty_band` doctest** — added `# Examples`:
  small-band (2×2) pure-scaling branch (no blur), large-band (8×8)
  blur-then-scale branch with output length match. Tick 458, 655e623e.

- **`gaussian_blur_sigma3` doctest** — added `# Examples`: output
  length matches input, DC preservation (uniform → uniform within
  1e-5). Tick 457, 87f53be3.

- **`gausspyr_expand_scalar` doctest** — added `# Examples`: 4×4 → 8×8
  (standard 2× upscale), 4×4 → 7×7 (odd target — supports `[2*sw-1, 2*sw]`
  range per debug_assert). Tick 456, 81c88a44.

- **`gausspyr_reduce_scalar` doctest** — added `# Examples`: 8×8 → 4×4
  with `dst.len() == 16`, odd-dim 7×7 → 4×4 ceil-halving. Tick 455, bfa03284.

- **`flatten_band_weights` doctest** — added `# Examples`: empty
  → empty, 2-level [[1,2,3],[4,5,6]] → [1..=6], `weight_idx =
  level * 3 + channel` indexing pin. Tick 454, a6d20ad5.

- **`precomputed_band_weights` doctest** — added `# Examples`:
  length agrees with `band_frequencies`, every [A, Rg, Vy] triple is
  positive-finite at standard 4K + 100 cd/m². Tick 453, f3b47c93.

- **`do_pooling_and_jod_still_3ch` doctest** — added `# Examples`:
  all-zero contrasts → JOD ≈ 10 within 1e-3, non-zero contrasts →
  JOD < that. Tick 452, 5ca2fc6c.

- **`mult_mutual_pixel` doctest** — added `# Examples`: T == R →
  D = [0, 0, 0], argument symmetry `f(T, R) == f(R, T)`, and
  non-negative output. Tick 451, 8c672482.

- **`mask_pool_pixel` doctest** — added `# Examples`: zero input
  → zero output, unit basis `[1, 0, 0]` recovers `XCM_3X3[0]` row.
  Tick 450, a8a4f127.

- **`pool_band_finalize` doctest** — added `# Examples`:
  zero partial → 0 (eps-tail explicitly canceled), negative partial
  clamps to 0, and uniform-|x|=c reconstruction at β=2 within 0.01.
  Documents the eps-tail bias size relationship `~ eps^(1/β)`. Tick 449, 184a1984.

- **`phase_uncertainty_no_blur` doctest** — added `# Examples`:
  pure scaling `input × 10^MASK_C`, zero passthrough, and scale-factor
  range pin in [0.15, 0.17]. Tick 448, 00ca7e06.

- **`lp_norm_sum` doctest** — added `# Examples`: pythagorean
  `lp_norm_sum([3, 4], 2) ≈ 5` within 0.01, empty → 0,
  sign-insensitive via abs. Tick 447, b04fd9ed.

- **`lp_norm_mean` doctest** — added `# Examples`: empty → 0,
  uniform input → constant within 0.01 (eps-tail bias),
  sign-insensitive via abs. Tick 446, 883ea3f1.

- **`sensitivity_corrected_scalar` doctest** — added `# Examples`
  showing positive output at standard photopic L_bkg (100 cd/m²) and
  that `corrected / uncorrected == 10^(DB/20)` within 1e-5. Tick 445, cb936c1b.

- **`clamp_diff_soft` doctest** — added `# Examples`: `f(0) == 0`,
  half-saturation at `d == d_max` (relative err < 1e-5),
  asymptotic bound `< d_max` at 1e9. Tick 444, d06a2073.

- **`safe_pow` doctest** — added `# Examples` covering `safe_pow(0, p)
  == 0` exact zero (via `(eps)^p - eps^p` cancellation), `safe_pow(2,
  2) ≈ 4` within 0.01, and monotonicity. Tick 443, b5cd8d3d.

- **`srgb_byte_to_dkl_scalar` doctest** — added `# Examples`:
  pure-white → positive A + chroma < 5% of A, pure-red → RG > 0
  (red-green axis convention). Tick 442, 6240f767.

- **`met2jod` doctest** — added `# Examples` covering perfect-quality
  limit (`met2jod(0) == 10`), monotonic decline (0 > 0.5 > 1.0 > 5.0),
  and extreme-input safety (`met2jod(1e6)` finite < 0). Tick 441, dddf1db9.

- **`band_frequencies` doctest** — added `# Examples` showing
  typical usage: at standard 4K geometry the function returns ≥ 5
  strictly-decreasing positive cy/deg entries for 1024×1024. Tick 440, e8505d51.

- **`estimate_gpu_memory_bytes` doctest** — added an `# Examples`
  section that exercises the function on 4 inputs and validates
  the rough magnitude: too-small (4×4, 7×8 → `None`); 1 MP at
  ~208 MB (asserted in `[100 MB, 300 MB]`); 4 MP > 1 MP. Doubles
  as documentation and a smoke test that runs under
  `cargo test --doc`. Tick 439, 8d3b4b94.

- **`csf_lut/v0_5_4.rs` LUT constant docstrings** — six previously-
  undocumented public LUT constants re-exported via
  `pub use csf_lut_v0_5_4::*`: `LOG_L_BKG_AXIS` (uniform-in-log10
  background-luminance axis, `[-2.301, 4.0]`), `LOG_RHO_AXIS`
  (uniform-in-log10 spatial-frequency axis, `[-1.0, 1.806]`),
  `LOG_S_O0_C1/C2/C3` (1024-entry A/Rg/Vy sensitivity tables with
  the `l_idx * 32 + rho_idx` layout), `GE_SIGMA` (eccentricity-
  falloff scaffolding). Tick 438, c217ee95.

- **`CsfChannel` variant docstrings** — `A` (achromatic /
  luminance), `Rg` (red-green opponent), `Vy` (violet-yellow
  opponent). Three previously-undocumented variants surfaced by
  `RUSTDOCFLAGS="-D missing_docs" cargo doc`. Tick 436, 38ba643e.

- **`Error::DimensionMismatch` field docstrings** — `expected`
  (`width × height × 3` byte count) and `got` (actual caller-passed
  length). Tick 436, 38ba643e.

#### cvvdp-gpu (tests)

- **`tests/params_placeholder_non_display.rs`** — 5 additional pins
  on `CvvdpParams::PLACEHOLDER`'s csf / masking / pooling / jod
  sub-bundles (the existing `params_placeholder.rs` only pinned
  display + perf_mode): (1) all csf peaks bit-equal 0.0 (scaffolded
  placeholder); (2) masking p=2.4, q=2.2, k=0.04 (scaffolding —
  doesn't match production `kernels::masking` constants); (3)
  pooling betas all 4.0 (doesn't match production
  `kernels::pool::BETA_SPATIAL=2.0`); (4) jod_a=10.0, jod_b=1.0,
  jod_c=0.30 (scaffolding); (5) struct supports `CvvdpParams {
  ..PLACEHOLDER }` update syntax (Copy + accessible-fields
  compile-time check). A future wire-through that actually consumes
  these fields will need to swap in real values; this pin flags the
  scaffolding state. Tick 435, e138a176.

- **`tests/predict_jod_invariants.rs`** — 7 flow invariants on
  `predict_jod_still_3ch` (the composed host-scalar pipeline)
  complementing `shadow_jod.rs`'s pycvvdp parity coverage:
  (1) byte-identical inputs → JOD ≈ 10 within 1e-3; (2) JOD ≤ 10 + ε
  for any (ref, dist); (3) determinism via `to_bits()`;
  (4) responds to distortion magnitude — sparse ±2 vs ±80 perturbation
  on a textured reference produces ≥ 1e-3 JOD shift AND larger
  distortion gives smaller JOD (catches stuck-at-constant refactor;
  flat reference + uniform shift was insufficient because the
  Weber-contrast pyramid of a flat input has zero band content);
  (5–6) panics on ref / dist `len() != w*h*3` (the `assert_eq!`
  entry guards); (7) 8×8 smoke — identical → 10, perturbed < 10.
  Tick 434, 489751a4.

- **`tests/csf_axes_invariants.rs`** — 9 structural pins on the
  public CSF LUT axis arrays `LOG_L_BKG_AXIS` and `LOG_RHO_AXIS`
  (32 entries each). `csf_constants_match_pycvvdp_v0_5_4` doesn't
  pin axis structure. Pins: (1) `LOG_L_BKG_AXIS.len() == N_L_BKG`
  + N_L_BKG == 32; (2) `LOG_RHO_AXIS.len() == N_RHO` + N_RHO == 32;
  (3) both arrays strictly monotonic; (4) `LOG_L_BKG_AXIS`
  endpoints bit-pinned to `-2.301..4.0`; (5) `LOG_RHO_AXIS`
  endpoints bit-pinned to `-1.0..1.806`; (6) `LOG_L_BKG_AXIS`
  uniformly spaced (`interp1_uniform` precondition); (7)
  `LOG_RHO_AXIS` uniformly spaced in log10 (the source comment
  about "non-uniform first interval" is about linear-rho ratios,
  not the log10 axis itself); (8) `LOG_L_BKG_AXIS` step matches
  the pycvvdp formula `(4.0 - (-2.301)) / 31 ≈ 0.2032`. Tick 433, 1d581c28.

- **`tests/perf_mode_invariants.rs`** — 6 invariants on `PerfMode`'s
  trait contract beyond the existing `params_placeholder.rs`
  PLACEHOLDER check: (1) `Default::default() == PerfMode::Strict`
  pinned explicitly (catches a refactor that moves `#[default]` to
  Fast while updating PLACEHOLDER); (2) Copy semantics work;
  (3) Clone yields Eq-equal value; (4) Strict != Fast (catches
  variant collapse); (5) Debug output is non-empty and distinct
  per variant; (6) exhaustive match visits exactly 2 variants.
  Tick 432, e2a37146.

- **`tests/mult_mutual_band_invariants.rs`** — 8 structural pins on
  `mult_mutual_band` (band-level 3-channel masking; existing
  coverage in `masking_kernel.rs` is GPU-parity only): (1) output
  shape 3 × `w*h` across 3 sizes; (2) `T == R` → identically zero
  bit-exact; (3) `f(T, R) == f(R, T)` symmetric (both `min(|T|,|R|)`
  and `|T - R|` are symmetric); (4) D[cc] ≥ 0 across signed inputs;
  (5) bounded by `d_max ≈ 366.69` even for ±1e6 contrast inputs
  (clamp_diff_soft cap); (6) determinism via `to_bits()`; (7)
  finite output for mixed-sign ramp ±1e3; (8) small-band branch
  exercised at 4×4 (below PU_PADSIZE=6, triggers no-blur path).
  Tick 431, eec3ea81.

- **`tests/gaussian_blur_sigma3_invariants.rs`** — 8 dedicated
  invariants on `gaussian_blur_sigma3`. The function previously had
  no direct tests — it was used only as a CPU reference for GPU
  parity in `masking_kernel.rs`. Pins: (1) output length matches
  `w * h` across 4 sizes; (2) constant input → constant output
  within 1e-5 relative (DC preservation; kernel sums to 1); (3)
  zero input → zero output bit-exact; (4) reflect-padded 7×7 (every
  pixel touches the boundary) stays finite; (5) non-negative input
  → non-negative output (kernel is all-positive); (6) determinism
  via `to_bits()`; (7) horizontal mirror-symmetric input yields
  symmetric output within 1e-5 (the kernel + boundary are
  symmetric); (8) impulse input concentrates max at the impulse
  location. Tick 430, 6f8b55de.

- **`tests/phase_uncertainty_band_invariants.rs`** — 7 invariant
  pins on `phase_uncertainty_band` (the branch-on-band-size helper).
  No prior direct tests — pipeline parity covered it indirectly.
  Pins: (1) small-band branch (`w ≤ 6 OR h ≤ 6`) is pure scaling
  bit-equal to `input × 10^MASK_C` across 6 size combos;
  (2) large-band branch (`w > 6 AND h > 6`) actually applies blur
  (impulse input → diffused output); (3) output length matches
  input across both branches; (4) determinism in both branches via
  `to_bits()`; (5) empty input → empty output, no panic;
  (6) finite output for finite input; (7) **branch threshold pin
  at `PU_PADSIZE = 6`** — `(6, 6)`, `(7, 6)`, `(6, 7)` all small;
  `(7, 7)` is the first large case. Catches a refactor that flips
  `&&` to `||` (would incorrectly blur degenerate strips that
  can't fit the σ=3 kernel's 13-tap support). Tick 429, 605f8ca4.

- **`tests/csf_channel_invariants.rs`** — 7 invariant pins on the
  `CsfChannel` enum's discriminants + trait contract. No prior
  test pinned these — a refactor that reorders variants (e.g.,
  `Rg = 0`) would silently shift every per-channel buffer index
  in the CSF stage. Pins: (1) `A = 0`, `Rg = 1`, `Vy = 2` via
  `as u32`; (2) all discriminants fit in `[0, N_CHANNELS)` for
  `as usize` array indexing; (3) Copy semantics; (4) Clone yields
  Eq-equal value; (5) PartialEq self-equality + cross-variant
  inequality; (6) Debug output is non-empty and unique per variant;
  (7) exhaustive match visits all 3 variants. Tick 428, b3f6b634.

- **`tests/precompute_logs_row_invariants.rs`** — 6 additional
  invariants on `precompute_logs_row` beyond the 3 existing tests
  in `csf_scalar.rs`: (1) determinism via `to_bits()` across 3
  channels × 3 rho; (2) distinct rows across A/Rg/Vy at 3 rho —
  catches a refactor that collapses the channel argument;
  (3) all-finite output across rho ∈ {0.001, 0.1, ..., 1024}
  (sub-LUT to super-LUT extrapolation); (4) `rho=0` doesn't panic
  or NaN (the `.max(1e-6)` clamp guards `log10(0) = -inf`); (5)
  negative rho clamps via `.max()` (not `.abs()`) — pins by
  matching `precompute_logs_row(-100, A)` bit-equal to
  `precompute_logs_row(1e-6, A)`; (6) `10^row[k]` is strictly
  positive-finite (sensitivities are physical, never zero).
  Tick 427, 513c7d60.

- **`tests/do_pooling_invariants.rs`** — 7 flow invariants on
  `do_pooling_and_jod_still_3ch` complementing the 3 pycvvdp
  parity tests in `pool_scalar.rs`: (1) zero input → JOD ≈ 10
  within 1e-5; (2) JOD ≤ 10 + 1e-3 across 4 input shapes;
  (3) monotonic in each (level, channel) position — perturbing
  any single element by +0.5 cannot raise JOD; (4) determinism
  via `to_bits()`; (5) responds to magnitude — 100× scaling
  produces ≥ 1e-3 JOD shift AND larger input gives smaller JOD;
  (6) single-level input (1 pyramid level) supported, no panic,
  JOD < 10 for non-zero; (7) 12-level stress input supported,
  finite output in [0, 10 + ε]. Tick 426, 2100715f.

- **`tests/mult_mutual_pixel_invariants.rs`** — 7 function-level
  invariants on `mult_mutual_pixel` (per-pixel cross-channel
  masking + diff). Complements the single `pycvvdp_4x4` parity
  test in `masking_scalar.rs` with shape pins: (1) `T == R` →
  `D = [0, 0, 0]` bit-exact via `to_bits()`; (2) symmetry
  `f(T, R) == f(R, T)` (since `min(|T|, |R|)` and `|T - R|` are
  both symmetric); (3) `D[cc] ≥ 0` for all signs of input;
  (4) `D[cc] < d_max = 10^D_MAX ≈ 366.69` even for ±1e6 inputs
  (clamp_diff_soft asymptote); (5) determinism; (6) any non-trivial
  `T ≠ R` produces positive `D` on at least one channel;
  (7) finite output across 5 dynamic ranges 1e-10 to ±1e6.
  Tick 425, 953780bd.

- **`tests/met2jod_invariants.rs`** — 8 invariant pins on `met2jod`
  beyond the 2 single-point tests already in `pool_scalar.rs`
  (`met2jod_continuous_at_kink` + `met2jod_clamps_at_origin`):
  (1) `met2jod(0) == 10` bit-exact via `to_bits()`; (2) value at
  kink Q=0.1 matches `10 - JOD_A * 0.1^JOD_EXP`; (3) strict
  monotonic decrease over Q ∈ [0, 100] step 0.01 (10001 samples);
  (4) `< 10` for any positive Q above f32 underflow (1e-3 onward);
  (5) power-branch algebra `10 - JOD_A * Q^JOD_EXP` for 6 Q above
  kink; (6) linear-branch algebra `10 - jod_a_p * Q` (where
  `jod_a_p = JOD_A * 0.1^(JOD_EXP-1)`) for 5 Q below kink — pins
  the slope-matching construction; (7) determinism; (8) declining
  finite JOD at extreme Q ∈ [1e3, 1e12]. Tick 424, 2764c3d8.

- **`tests/mask_pool_pixel_invariants.rs`** — 7 invariant pins on
  `mask_pool_pixel`, the 3×3 cross-channel masking matrix-vector
  multiply against `XCM_3X3`. No direct unit tests existed before
  (`mult_mutual_pixel` covered it indirectly through full-pipeline
  parity). Pins: (1) zero input → zero output bit-exact;
  (2) determinism via `to_bits()`; (3) α-scaling linearity within
  1e-6 relative across 5 scalars; (4) additivity `f(a+b) == f(a) +
  f(b)` within 1e-5 relative; (5) unit-basis inputs recover the
  rows of `XCM_3X3` exactly via `to_bits()` — catches a row-column
  transposition that wouldn't trip pipeline parity; (6) all-finite
  output for finite input across 6 input dynamic ranges (1e-10 to
  1e6, positive + negative); (7) A's self-coupling dominance
  (`out[0] > 0.5` for `[1, 0, 0]` since `XCM_3X3[0][0] = 0.877`)
  — pins the matrix orientation. Tick 423, f28b0455.

- **`tests/clamp_phase_uncertainty_invariants.rs`** — 10 invariant
  pins on two small masking primitives that previously had no
  direct unit tests:
  - `clamp_diff_soft(d) = d_max·d / (d_max + d)`: (1) `f(0) == 0`
    bit-exact via `to_bits()`; (2) strict monotonicity across 200
    samples in [0, 1000]; (3) asymptotic `f(d) < d_max` for d up
    to 1e9, plus gap < 0.1% at d ≥ 1e6; (4) half-saturation
    `f(d_max) == d_max/2` within 1e-5 relative; (5) determinism.
  - `phase_uncertainty_no_blur(m) = m * 10^MASK_C`: (6) pure
    scaling via `to_bits()` across 8 sample inputs incl. negatives;
    (7) scale factor pinned in [0.15, 0.17] (loose bound on the
    bit-pinned MASK_C); (8) `f(0) == 0`; (9) monotonicity over
    [-100, 100]; (10) determinism.
  Tick 422, 8c0d4bc7.

- **`tests/weber_pyramid_invariants.rs`** — eight structural
  invariant pins on `weber_contrast_pyr_dec_scalar` complementing
  the full-pipeline parity coverage in `pipeline_color.rs` /
  `pipeline_score.rs`: (1) band count matches `n_levels` 1..=4 (and
  `log_l_bkg` matches); (2) auto-`n_levels=0` selects
  `min(sw,sh).ilog2()` (64×32 → 5); (3) `log_l_bkg[k].len() ==
  bands[k].w * bands[k].h` per level; (4) baseband `log_l_bkg` is
  bit-constant (all entries equal via `to_bits()` — pins the
  "replicated scalar mean" docstring contract); (5) baseband band
  data is finite (division-by-zero guard via 0.01 floor); (6)
  non-baseband contrast clamped to `[-1000, 1000]` via 1e6 impulse
  on 0.001 L_bkg field (baseband intentionally excluded — it's
  unclamped per source); (7) zero-image + zero-l_bkg input produces
  no NaN/Inf (the 0.01 floor guards everything); (8) determinism
  via `to_bits()` over bands + log_l_bkg. Tick 421, c6c30191.

- **`tests/srgb_byte_to_dkl_invariants.rs`** — eight function-level
  semantic invariants on `srgb_byte_to_dkl_scalar` beyond the
  pointwise pycvvdp parity at `STANDARD_4K`: (1) DKL_A strictly
  monotonic in grayscale ramp 0..256 step 16; (2) grayscale chroma
  RG/VY < 5% of A's magnitude across 9 neutral bytes; (3) black <
  mid < white ordering on the A channel; (4) linearity in `y_peak`
  via Δ(100→200) = ⅓ × Δ(100→400); (5) corner-pixel safety for 8
  corners of the RGB cube (no panic, all finite); (6) determinism
  via `to_bits()` across 3 inputs × 3 channels; (7) pure-red → RG > 0
  and pure-cyan → RG < 0 (pins row-1 sign convention against a row
  swap with row 2); (8) pure-blue → VY > 0 and pure-yellow → VY < 0.
  Complements the matrix-bit-pin in `srgb_linear_to_dkl_matrix_*` —
  pins the FUNCTION'S shape, not just the matrix's entries. Tick 420, 43bd4a18.

- **`tests/gausspyr_expand_invariants.rs`** — seven structural
  invariant pins on `gausspyr_expand_scalar`, mirror of
  `gausspyr_reduce_invariants.rs`: (1) `dst.len() == out_w * out_h`
  across the full documented `[2*sw - 1, 2*sw]` × `[2*sh - 1, 2*sh]`
  range (4 combos of even/odd target dims); (2) `dst` fully
  overwritten — NaN pre-fill catches; (3) determinism via
  `to_bits()`; (4) capacity invariance; (5) `(sw=4, sh=2)` vs
  `(sw=2, sh=4)` produce distinct content (catches width/height
  collapse in the separable convolution); (6) both `odd_w/odd_h`
  branches inside the function succeed (5→9 odd and 5→10 even
  paths); (7) all-finite output across 7 typical pyramid expand
  pairs including non-square 8×4 → 16×7 and minimal 3×3 → 5×5/6×5.
  Tick 419, a68e5c33.

- **`tests/gausspyr_reduce_invariants.rs`** — seven structural
  invariant pins on `gausspyr_reduce_scalar`: (1) `(dw, dh) = (4, 4)`
  and `dst.len() == 16` for 8×8; (2) odd inputs ceil-halve correctly
  (7×7 → 4×4; 17×13 → 9×7); (3) returned `(dw, dh)` agrees with
  `dst.len()` across 9 size combos including non-square; (4) `dst`
  is fully overwritten — pre-fill with NaN and confirm none survive;
  (5) determinism via `to_bits()` bit-equality across repeated calls;
  (6) `(4, 8)` and `(8, 4)` produce distinct dims AND distinct content
  (width/height swap catches); (7) caller-provided `dst` capacity
  (too-big or zero) doesn't affect output. Complements
  `pyramid_scalar.rs::reduce_matches_pycvvdp`'s single fixed-input
  pycvvdp parity test with broad structural coverage. Tick 418, 41dae2f5.

- **`tests/laplacian_pyramid_invariants.rs`** — seven structural
  invariant pins on `laplacian_pyramid_dec_scalar`: (1) output band
  count matches requested `n_levels` for 1..=4; (2) auto-`n_levels=0`
  picks `min(sw, sh).ilog2()` bands across 64² (=6) and non-square
  64×32 / 32×64 (=5); (3) band dimensions track an independently-
  rebuilt `gausspyr_reduce_scalar` chain on 17×13 (odd-dim);
  (4) baseband (last band) is bit-equal to the coarsest gaussian
  via `to_bits()` (pins the docstring contract that the baseband
  is NOT a Laplacian residual); (5) determinism via bit-equality
  across repeated calls; (6) `n_levels=1` returns a single band
  bit-equal to the input (the `for k in 0..(n-1)` empty loop edge
  case); (7) Band invariant `data.len() == w * h` for every band.
  Complements `pyramid_scalar.rs`'s pointwise-numeric pycvvdp
  parity tests with structural / contract coverage. Tick 417, ae86b5e1.

- **`tests/band_weights_invariants.rs`** — eight invariant pins on
  `flatten_band_weights` and `precomputed_band_weights` covering
  edges + structural properties the existing pointwise test missed:
  empty input → empty output with zero capacity; length invariant
  `out.len() == weights.len() * 3` across n ∈ {0, 1, 2, 3, 5, 8,
  16, 50}; documented `flat[level * 3 + channel]` indexing
  contract; NaN/±∞/-0.0 bit-passthrough; `precomputed_band_weights`
  length agrees with `band_frequencies` across 9 image sizes
  (square + non-square 16² up to 4K); all-finite + strictly-positive
  output across log_L_bkg ∈ [-1.0, 3.0] (0.1 cd/m² dim through
  1000 cd/m² HDR peak); determinism via `to_bits()` equality;
  end-to-end flatten-then-index round-trip pinning the
  `weight_band_kernel` consumer contract. Tick 416, 6b2891af.

- **`tests/display_geometry.rs::ppd_is_*`** — four invariant tests
  on `DisplayGeometry::pixels_per_degree`: (1) positive + finite +
  in realistic [5, 500] range across phone, tablet, desktop,
  cinema, UHD-living-room viewing configs; (2) strictly
  monotonically increasing in `distance_m` (further → less angle
  per pixel → higher PPD); (3) strictly monotonically decreasing
  in `diagonal_inches` (larger physical screen at same distance →
  more angle per pixel → lower PPD); (4) strictly monotonically
  increasing in `resolution_w` at fixed 16:9 aspect. Catches sign
  flips and dimension-swaps that would silently mis-calibrate the
  CSF stage's per-band rho query. Tick 415, 1fc417cd.

#### cvvdp-gpu (docs)

- **README "Build" section refreshed for the CUDA-version-matters
  lesson learned during the v22-v25 fleet incident.** The
  previous claim "CUDA 13.2 required for cubecl 0.10's CUDA
  backend" was misleading — cubecl 0.10 itself doesn't require
  13.x; its `cudarc 0.19.4` dep auto-selects a `cuda-<MMmmpp>`
  cargo feature from the SDK present at build time, and the
  resulting binary's dlsym entries must match symbols the host's
  libcuda exports. Binaries built against CUDA 13 try to dlsym
  `cuCoredumpDeregisterCompleteCallback` (gated behind cudarc's
  `cuda-13020` feature) which is absent from every released
  NVIDIA libcuda — panics at first dispatch. New README guidance
  explicitly differentiates RTX 50-series (CUDA 13 required for
  Blackwell sm_120) from RTX 20/30/40/A2000 etc. (CUDA 12.6 SDK,
  proven on the production fleet under driver 535+). Plus calls
  out the runtime requirement on NVRTC headers
  (`cuda-cudart-dev-<MMmm>`) — without them, `Cvvdp::score`
  returns the dual-purpose `InvalidImageSize` masking an NVRTC
  compile failure (v25 lesson). Tick 414, 00c5875e.

#### cvvdp-gpu (tests)

- **`tests/pyramid_scalar.rs::band_frequencies_{are_strictly_decreasing,minimum_image_dim_returns_some_bands,per_band_ratio_in_sensible_range}`**
  — three invariant tests on `band_frequencies`: (1) output is
  strictly decreasing (Laplacian pyramid orders finest→coarsest)
  across 4 (ppd, dim) combos, every entry finite + positive; (2)
  minimum image dim 8×8 returns ≥ 1 band (so Cvvdp::new never
  builds a zero-band pyramid where the Vec-of-Level sizing would
  silently fail); (3) mid-pyramid adjacent-band ratio in
  [1.5, 3.5] — captures the "near-octave" Laplacian behavior
  while accommodating the first-level Nyquist-quarter scaling
  and the trailing MIN_FREQ=0.2 floor. Tick 412, f861d026.

- **`tests/pipeline_score.rs::score_is_deterministic_*`** — two
  contract tests pinning the critical "no state leakage between
  calls on the same `Cvvdp` instance" property that zen-metrics-
  cli's `CvvdpBatchScorer` relies on for the vast.ai backfill
  pipeline. (1) `score(ref, dist)` called twice → bit-identical
  output via `.to_bits()`; (2) `score(ref, dist_a)` →
  `score(ref, dist_b)` → `score(ref, dist_a)` again → first and
  third results bit-identical (no state leaked from the b call);
  (3) intervening warm_reference + compute_dkl_jod_with_warm_ref
  doesn't poison cold-path scratch. A regression where a scratch
  buffer reset is dropped, an accumulator grows across calls, or
  warm-ref state contaminates cold dispatch surfaces here before
  silently breaking the cached-instance pattern that the OOM-fix
  tick 384 depends on. Tick 411, ebe21f89.

- **`tests/masking_safe_pow.rs`** — five direct unit tests on
  `kernels::masking::safe_pow` (cvvdp's `(x + eps)^p - eps^p`
  used in the masking chain — distinct from
  `pool::safe_pow_lp`'s `|x|.abs() + eps` variant). Previously
  exercised only transitively through `mult_mutual_pixel` and
  the composed-pipeline parity tests. New tests pin: (1)
  `safe_pow(0, p) = 0` exactly across p ∈ {1, 2, MASK_P, 4} —
  catches a refactor that drops the `- eps^p` correction; (2)
  `safe_pow(1, p)` matches the closed-form `(1 + eps)^p - eps^p`
  for the same p set; (3) strictly monotonic in x for positive p
  (catches a sign-flip on the correction term); (4) eps offset
  dominates only near zero — for x ≫ eps, result ≈ x^p (rel <
  1e-3) and for x = eps the closed form `eps^p × (2^p - 1)`
  holds; (5) finite + positive at extreme x ∈ {100, 1k, 10k}
  across p set (catches an overflow-to-inf regression). Lives
  in a dedicated file per the tick-401 precedent (linter-revert
  safety vs `masking_scalar.rs`). Tick 410, 57fc5225.

- **`tests/error_traits.rs`** — pins five trait-side contracts on
  `cvvdp_gpu::Error`: (1) `impl std::error::Error` (compile-time
  check via `&dyn` coercion); (2) `Clone` preserves variant +
  payload across all four variants (catches a derive-to-manual-
  impl refactor that drops a field on `DimensionMismatch`); (3)
  `source()` returns `None` for every variant — these are leaf
  errors with no nested cause chain (if a future variant wraps a
  backend error, this test fails loudly + maintainer documents
  the new contract); (4) `Debug` includes the variant name
  verbatim across all four; (5) the `?`-bubble path through
  `Box<dyn std::error::Error>` works and preserves the actionable
  Display message. Sibling to tick 282's
  `error_display_messages_are_actionable` (Display content) —
  this tests the trait *implementations*. Tick 409, 8177979b.

#### cvvdp-gpu (docs)

- **`kernels::csf::N_RHO` gets its own docstring**, separating
  it from `N_L_BKG`'s. The two constants previously shared a
  single `/// Number of grid points along each LUT axis.` comment
  preceding `N_L_BKG` only — rustdoc attached the doc to
  `N_L_BKG` and left `N_RHO` undocumented. Now each has its own
  doc explaining the kernel-sizing constraint, with a cross-
  reference between them. Verified via a python doc-coverage
  sweep: 0 undocumented public items remain in the non-LUT
  source (the LUT file `csf_lut/v0_5_4.rs` is auto-generated
  from pycvvdp's JSON; comments there would not survive
  regeneration). Tick 408, 7c4d4758.

#### cvvdp-gpu (tests)

- **`tests/pipeline_score.rs::score_returns_lossless_f64_widening_of_compute_dkl_jod`**
  — pins the documented `Cvvdp::score` contract: returns
  `f64::from(compute_dkl_jod(ref, dist, ppd))` where ppd comes
  from `self.geometry.pixels_per_degree()`. f32 → f64 widening
  is lossless, so the round-trip `(score() as f32).to_bits() ==
  compute_dkl_jod().to_bits()` must hold bit-for-bit. Sweeps the
  full v1 corpus q-grid. Catches a refactor that introduces a
  precision-eating step (e.g. `jod as f64 * 1.0` rounded
  through an intermediate). Also asserts score ∈ [0, 10] across
  the corpus q-range. Tick 407, 14582e8a.
- **`tests/pipeline_score.rs::{parallel_safety_factor_*, recommend_parallel_{monotonic,budget}_*, estimate_gpu_memory_grows_*}`**
  — four invariant tests on the GPU memory predictor + concurrency-
  cap API: (1) `PARALLEL_SAFETY_FACTOR` in [1.0, 3.0] sane-range
  with exact-value pin at 1.5 (catches a refactor that drops it to
  0.5 (overrun) or 5.0 (waste)); (2) `recommend_parallel` is
  monotonically non-decreasing in `free_gpu_bytes` (catches a sign-
  flip / inverted-division bug that would silently mis-cap large-
  GPU sweeps); (3) the budget invariant `N × SAFETY × est ≤ free`
  holds whenever `recommend_parallel` returns N > 1 (the floor-1
  case is the documented "back off explicitly" signal); (4)
  `estimate_gpu_memory_bytes` is strictly increasing across six
  image sizes (catches a refactor that introduces fixed-cost
  inversion). Tick 406, 8039d126.
- **`tests/params_placeholder.rs`** — pins the two `CvvdpParams::PLACEHOLDER`
  fields the pipeline actually consumes: `display ==
  DisplayModel::STANDARD_4K` (field-by-field bit-pattern check
  on y_peak/y_black/y_refl) and `perf_mode == PerfMode::Strict`.
  Every parity test in the crate constructs `Cvvdp::new(...,
  PLACEHOLDER)`, so a refactor that flipped the placeholder
  default to `PerfMode::Fast` would silently change every
  golden-test calibration baseline. Plus a contract test
  exercising PerfMode's Copy + PartialEq derives. Tick 405, daab6476.
- **`tests/goldens_metadata.rs`** — pins the self-consistency of
  the goldens-fetch infrastructure in `tests/common/mod.rs`:
  `MANIFEST_URL` must embed `GOLDEN_VERSION` as a path segment
  (`/v1/`), use https scheme, end in `.json`; `MANIFEST_SHA256`
  must be exactly 64 chars of `[0-9a-f]` (catches truncation +
  uppercase typos that would silently break `Sha256::finalize()`
  comparison); `cache_dir()` must embed `GOLDEN_VERSION` and the
  crate-specific subdir; `GOLDEN_VERSION` must follow the
  `v<N>` convention with decimal digits. A regression that bumps
  `GOLDEN_VERSION = "v2"` but forgets to update `MANIFEST_URL`
  surfaces here before the goldens-feature gate runs the actual
  fetch. Same loud-failure-on-silent-edit discipline applied to
  test-infrastructure constants. Tick 404, 6e95bfac.
- **`tests/shadow_jod.rs::predict_jod_still_3ch_returns_max_jod_on_identical_inputs`**
  — integration-test promotion of the lib.rs doctest's identity
  contract (scoring a buffer against itself yields JOD ≈ 10.0).
  Doctests are skipped when filtering with `cargo test --test
  <name>`, leaving the host-scalar identity contract uncovered
  in the standard test path. Sweeps three sizes × three uniform
  values: (8×8 = PYRAMID_MIN_DIM×2 boundary, 64×64 = doctest
  size, 73×91 = odd-dim with pycvvdp `gausspyr_reduce` column-
  parity bug-compat patches from ticks 204-206) × val ∈ {0, 128,
  255}. Companion to tick 350's
  `compute_dkl_jod_host_pool_returns_max_jod_on_identical_inputs`
  (GPU host-pool path) — same contract on the host-scalar
  reference twin. Tick 403, 7ef5b683.
- **`tests/lib_constants.rs`** — seventh in the constants-pin
  series. Pins the three crate-level constants exposed from
  `lib.rs`: `N_CHANNELS = 3` (still-image DKL opponent count),
  `MAX_LEVELS = 9` (pyramid-level cap — bumping requires
  resizing `logs_row` + `partials_h` + weights buffers), and
  `PYRAMID_MIN_DIM = 4` (minimum logical level dim). Plus a
  derived-invariant test `PYRAMID_MIN_DIM × 2 = 8` so a refactor
  that changes the `width < PYRAMID_MIN_DIM * 2` guard's
  multiplier surfaces here instead of as a boundary-test
  regression. Tick 402, c61e4b22.
- **`tests/masking_constants.rs`** — sixth in the constants-pin
  series (393 pool / 394 csf / 395 pyramid / 396 display / 397
  color matrix). New dedicated test file pins by exact f32 bit
  pattern: `CH_GAIN = [1.0, 1.45, 1.0]`, `MASK_P = 2.264_355_2`,
  `MASK_Q = [1.302_622_7, 2.888_590_8, 3.680_771_3]`, `MASK_C =
  -0.795_497_12`, `D_MAX = 2.564_245_5`, all 9 entries of
  `XCM_3X3`, and all 13 taps of `PU_BLUR_KERNEL_1D`. Plus
  structural invariants on `PU_BLUR_KERNEL_1D`: DC preservation
  (sum ≈ 1.0 within 1e-6) and symmetry around the centre tap.
  Lives in a dedicated file (not the historically linter-edge-
  case-sensitive `masking_scalar.rs`) so the consts pin stays
  durable. Tick 401, 57506ad9.
- **`tests/color_scalar.rs::srgb_linear_to_dkl_*`** — fifth in
  the constants-pin series (393 pool / 394 csf / 395 pyramid /
  396 display). Pins all 9 entries of `SRGB_LINEAR_TO_DKL` by
  `.to_bits()` — the f32 row-major DKL matrix composed at f64
  precision from `LMS2006_to_DKLd65 @ XYZ_to_LMS2006 @
  sRGB_to_XYZ`. Previously verified only transitively through
  the 8-point byte goldens + the row-sum heuristic — a refactor
  that swaps two entries within a row, or substitutes a
  plausible-but-different matrix (LMS2000 instead of LMS2006),
  could pass both. Second test pins the opponent-color sign
  signature: row 0 (A) all positive, row 1 (Rg) is `(+, -, -)`,
  row 2 (Vy) is `(-, -, +)`. Tick 397, 616a6a8a.
- **`tests/display_geometry.rs::display_{model,geometry}_standard_4k_*`**
  — pins the f32 bit patterns of `DisplayModel::STANDARD_4K`
  (`y_peak = 200`, `y_black = 0.2`, `y_refl = 0.397_887_36`) and
  `DisplayGeometry::STANDARD_4K` (`3840×2160`, `distance_m =
  0.7472`, `diagonal_inches = 30`). The v1 R2 manifest goldens
  were captured under this display configuration — a silent edit
  to any field (e.g. swapping `y_refl` for the unrounded f64
  literal) would invalidate every shadow_jod parity test in a
  way that's hard to trace back to the display constants.
  Companion to ticks 393 (pool) / 394 (csf) / 395 (pyramid). Tick 396, 0b5cf789.
- **`tests/pyramid_scalar.rs::pyramid_constants_match_pycvvdp_v0_5_4`**
  — sibling to ticks 393 (pool) / 394 (csf). Pins
  `KERNEL_A = 0.4` (the Burt-Adelson `a` parameter) by exact
  bit pattern and verifies `GAUSS5 = [0.05, 0.25, 0.4, 0.25,
  0.05]` is consistent with the compile-time derivation
  `[0.25-a/2, 0.25, a, 0.25, 0.25-a/2]` — outer taps use
  abs-diff < 1e-7 because `0.25 - 0.4/2.0` rounds one ULP
  below the literal 0.05 at compile time; inner taps are
  exact. Plus two structural invariants: DC-preservation
  (sum ≈ 1.0 within 1e-6) and symmetry around the center tap
  (bit-identical pairs `[0]==[4]`, `[1]==[3]`). A drift in
  `KERNEL_A` to e.g. 0.375 (the Burt original) would broaden
  the kernel and silently shift every pyramid level; the
  test trips with a specific message instead of cascading
  into shadow_jod drift. Tick 395, 31eb8bbe.
- **`tests/csf_scalar.rs::csf_constants_match_pycvvdp_v0_5_4`** —
  sibling to tick 393's pool-constant pin. Locks the exact f32
  bit patterns of `SENSITIVITY_CORRECTION_DB` (-0.279_742_33)
  and `CSF_BASEBAND_RHO` (0.1), plus integer values of
  `N_L_BKG` and `N_RHO` (both 32). The dB correction was
  previously checked transitively via the
  `sensitivity_correction_is_a_small_attenuation` magnitude test
  (tick 388) but not pinned to an exact bit pattern;
  `CSF_BASEBAND_RHO` had no direct value check at all. N_L_BKG /
  N_RHO are the LUT grid sizes the GPU kernels assume via array
  sizing — a refactor that bumps either without resizing kernel
  buffers would corrupt every per-pixel CSF lookup. Tick 394, f8c962aa.
- **`tests/pool_scalar.rs::pool_constants_match_pycvvdp_v0_5_4`** —
  pins the exact f32 bit patterns of the eight pool constants
  imported verbatim from pycvvdp v0.5.4: `BETA_SPATIAL`,
  `BETA_BAND`, `BETA_CH`, `IMAGE_INT`, `JOD_A`, `JOD_EXP`,
  `PER_CH_W[0..3]`, `BASEBAND_W[0..3]`. A silent edit (typo,
  sign flip, decimal-point shift) to any of these cascades into
  JOD drift across every parity gate, where it's hard to
  localize. The new test trips with a specific message
  identifying which constant changed — turning a 0.001 JOD
  drift on shadow_jod_gpu into "BASEBAND_W[Vy] = X, expected
  4.118_745_3 (cvvdp v0.5.4)". When a future cvvdp version
  (0.5.5+) ships new coefficients, update these values together
  with the pin and re-run parity. Tick 393, 5d5d4ff3.
- **`tests/pipeline_score.rs::dimension_mismatch_surfaces_on_wrong_size_inputs`**
  — extended once more to cover six pyramid/band intermediate-
  output methods that validate buffer length transitively through
  `_dispatch_dkl_planes_gpu` (the shared entry point that contains
  the actual `!=` check): `compute_dkl_gauss_pyramid`,
  `compute_dkl_laplacian_pyramid`, `compute_dkl_weber_pyramid`,
  `compute_dkl_t_p_bands`, `compute_dkl_csf_weighted_bands`, and
  `compute_dkl_d_bands` (both ref + dist args per docstring). Each
  method's docstring documents the `Error::DimensionMismatch`
  return — a refactor that inlines `_dispatch_dkl_planes_gpu` into
  a caller but forgets to copy the length check would surface here
  before slipping into a kernel-side panic on the under-sized
  buffer read. Test now exercises all 13 documented dim-check
  sites (was 9 after tick 390). Tick 392, e277f103.
- **`tests/pipeline_score.rs::compute_dkl_jod_host_pool_with_warm_ref_reports_dim_mismatch_before_no_warm`**
  — sibling pin to the tick-248 GPU-variant test
  (`compute_dkl_jod_with_warm_ref_reports_dim_mismatch_before_no_warm`).
  The source code for `compute_dkl_jod_host_pool_with_warm_ref`
  applies the dim check before the warm-state check (the
  comment references the tick-248 ordering rationale) but had
  no regression test pinning the contract. A refactor that
  swaps the order on the host_pool path — returning
  NoWarmReference first and masking the more actionable
  DimensionMismatch — would slip past CI. host_pool matters
  because cubecl-cpu / Metal callers route through it
  explicitly (the GPU Atomic<f32>::fetch_add path doesn't run
  on those backends), so their production error reporting
  needs the same ordering as the GPU path. Tick 391, bc65041c.
- **`tests/pipeline_score.rs::dimension_mismatch_surfaces_on_wrong_size_inputs`**
  — extended to cover four additional public entry points the
  original tick-239 test acknowledged in its docstring but did
  not actually exercise: `compute_dkl_jod` (both ref/dist args),
  `compute_dkl_planes`, `compute_dkl_jod_host_pool` (both args),
  `compute_dkl_jod_host_pool_with_warm_ref`. The five sites
  previously covered were `score`, `set_reference`,
  `score_with_reference`, `warm_reference`, and
  `compute_dkl_jod_with_warm_ref` — leaving the GPU-pool and
  host-pool variants of `compute_dkl_jod` unchecked. A refactor
  that swaps the `!=` check for `<` on any of the four newly-
  covered entries (silently accepting smaller buffers and
  reading garbage past `srgb.len()`) would slip past the
  original 5-site coverage. Tick 390, 8e4d2590.
- **`tests/pool_scalar.rs::lp_norm_mean_*`** — four direct unit
  tests on `lp_norm_mean` (cvvdp's `lp_norm` with `normalize=True`).
  The function was exercised only through the GPU-gated
  `pool_band_kernel_matches_host_lp_norm_mean` test and the
  single-input `pool_band_finalize_matches_lp_norm_mean_on_synth_signal`
  test, leaving no direct CPU-only coverage of its algebra
  invariants. New tests pin: (1) empty-input early-return → 0.0
  exactly at p ∈ {1,2,4,8} via `.to_bits()` (without the guard,
  `acc/n` produces NaN at n=0); (2) uniform-input identity:
  `lp_norm_mean([a; n], p) ≈ a - eps^(1/p)` at (a, p) ∈
  {(0.5, 2), (2.5, 4)} × n ∈ {1,4,16,64} (catches a refactor
  that drops the `/ n` step, which would overestimate by
  `n^(1/p)`); (3) sign-handling via `.abs()` — pos/mixed/neg
  inputs produce bit-identical output (mirror of lp_norm_sum's
  test, pinned separately for the lp_norm_mean call site); (4)
  the defining identity `lp_norm_sum ≈ n^(1/p) * lp_norm_mean`
  at p ∈ {2, 4} on an 8-element signal (a structural-divergence
  catcher — if either function changes its eps shift, this
  trips). Tick 389, bfef0b2f.
- **`tests/csf_scalar.rs::sensitivity_corrected_*` + `sensitivity_correction_*`**
  — three direct unit tests on `sensitivity_corrected_scalar`,
  which the production CSF apply path (`precomputed_band_weights`
  + the GPU kernel host-side row-precompute) reads through but
  previously had no scalar-side direct coverage. New tests pin:
  (1) the correction is a constant multiplicative factor
  (corrected/uncorrected ratio bit-identical to 1e-5 across 3
  channels × 3 rho × 3 log_l_bkg = 27 points — catches a
  refactor that breaks the input-independence invariant); (2)
  the factor magnitude (0.9, 1.0) and specific value ≈ 0.9684
  (catches sign flips that would amplify instead of attenuate,
  and order-of-magnitude wrong DB constants); (3) extreme-input
  finiteness (same clamping contract as `sensitivity_scalar`,
  but pinned separately so the uncorrected path and the
  multiplicative step can each regress independently). Tick 388, 506f61bf.
- **`tests/color_scalar.rs::srgb_lut_*`** — four direct unit tests
  on the public `SRGB8_TO_LINEAR_LUT` 256-entry sRGB EOTF table.
  Previously the LUT was verified only transitively through the
  8-point `matches_pycvvdp_standard_4k` byte goldens — and a
  historical "~6e-4 drift at bright bytes" regression (referenced
  in `pipeline_color.rs:2009`) had shipped because the goldens
  happened to skip the affected bytes. New tests pin: (1) length
  256 + exact 0.0 / 1.0 endpoints at byte 0 / 255 via `.to_bits()`
  (off-by-one + missing-boundary catcher); (2) strict monotonic
  increasing across all 256 bytes (bit-flip or swapped-pair
  catcher); (3) direct comparison against the IEC 61966-2-1
  inverse companding formula at every byte (f64 reference, 1e-6
  absolute tolerance — well under the 6e-4 historic drift); (4)
  seam continuity around c = 0.04045 (byte 11) — pin the local
  slope ratio to (0.5, 2.0) to catch a refactor that mis-aligns
  the piecewise branch threshold. Tick 387, 0e284715.
- **`tests/csf_scalar.rs::precompute_logs_row_*`** — five direct
  unit tests on the previously-GPU-only-exercised public
  `precompute_logs_row`. The helper had no scalar-side coverage:
  it was used in `tests/csf_kernel.rs` to set up GPU kernel
  inputs, but that file is feature-gated to
  `cfg(any(cuda, wgpu, hip))` so a CPU-only test run (no GPU
  available, no atomic-f32 support) never touched it. New tests
  pin: (1) returns exactly `N_L_BKG = 32` entries across all
  channels × four rho values (a refactor that shrinks the row
  would corrupt every per-pixel CSF lookup); (2) the closed-form
  identity `sensitivity_scalar(rho, LOG_L_BKG_AXIS[k], cc) =
  10^precompute_logs_row(rho, cc)[k]` across 3 channels × 4 rho
  × 32 axis points = 384 points (interp1_uniform returns the
  exact row value at axis indices, so this identity is parity
  glue between the two public functions); (3) frequency
  dependence — max |diff| > 0.1 between rho=0.5 and rho=16
  cy/deg for the achromatic channel (catches a refactor that
  collapses the rho axis); (4) channel dependence — pairwise
  max |diff| > 1e-3 between A/Rg/Vy at fixed rho=4 (catches a
  channel_lut dispatch typo); (5) the `rho.max(1e-6)` clamp —
  rho ∈ {0, -1, 1e-6} produce bit-identical rows via
  `.to_bits()` (silent-NaN-propagation regression catcher).
  Same gap-shape as ticks 351/383. Tick 386, 0e284715.
- **`tests/pool_scalar.rs::pool_band_finalize_*`** — five direct
  unit tests on the previously-indirectly-exercised public
  `pool_band_finalize`. The function was covered only via the
  GPU-backed `pool_band_kernel_matches_host_lp_norm_mean` test,
  which means CPU-only test runs (e.g. cubecl-cpu CI on a host
  without atomic-f32 GPU) couldn't catch host-side regressions
  to its algebra. New tests pin: (1) zero-partial returns 0 across
  β ∈ {1,2,4,8} and n ∈ {1,64,1024,65536} — eps^(1/β) tail must
  cancel head; (2) negative-partial clamping to 0 (atomic-noise
  protection — without `.max(0)`, β=2 returns NaN at non-integer
  exponents); (3) scalar-form identity vs `lp_norm_mean` on a
  synthesised signal (the same identity the GPU kernels rely on,
  now testable without a GPU); (4) eps^(1/β) tail magnitude
  pinned at β ∈ {1,2,4} — same observation as the lp_norm_sum
  tests, at β=2 the tail is 316× larger than at β=1; (5)
  strict-monotonic decreasing in n_pixels under fixed partial,
  with a closed-form check at (partial=100, n=100, β=2). Tick 383, a5218943.
- **`tests/pool_scalar.rs::lp_norm_sum_*`** — four direct unit
  tests on the previously-uncovered public `lp_norm_sum`:
  Pythagorean-triple at p=2, sign-handling via `.abs()`,
  zero-input across n in {0, 1, 5, 64}, and uniform-input
  count-scaling at p=4. Discovered while writing: the outer
  `eps^(1/p)` tail subtraction is NOT negligible — sqrt(1e-5)
  ≈ 0.00316 at p=2, eps^0.25 ≈ 0.0562 at p=4. Tests subtract
  the eps tail explicitly rather than loosening tolerances to
  mask it; this is cvvdp's documented safe_pow shape. Tick 351,
  `711eba8a`.
- **`tests/cpu_backend.rs::compute_dkl_jod_host_pool_returns_max_jod_on_identical_inputs`**
  — end-to-end identity gate that scores a buffer against
  itself and asserts JOD ≈ 10.0. Closes a gap where the
  property was only exercised by the `Cvvdp::score` doctest
  (skipped in `cargo test --test <name>` runs). Tick 350,
  `ca3b9d3a`.
- **Documented panic contracts now have `should_panic` regression
  tests** — `do_pooling_and_jod_panics_on_empty_q_per_ch` in
  `pool_scalar.rs` and `predict_jod_still_3ch_panics_on_ref_dim_mismatch`
  / `predict_jod_still_3ch_panics_on_dist_dim_mismatch` in
  `shadow_jod.rs`. Both `# Panics` docstring sections previously
  had only doctest coverage; the integration tests gate them in
  the standard `cargo test` run. Tick 349, `2beffe90`.
- `tests/pyramid_scalar.rs::band_frequencies_exceeds_max_levels_at_high_ppd_or_dim`
  pins the `MAX_LEVELS=9` cap in `pipeline::pyramid_levels` as
  non-vacuous: `band_frequencies` returns 11 entries for
  `(ppd=400, 2048×2048)`, `(ppd=200, 4096×4096)`,
  `(ppd=200, 8192×8192)` — the cap MUST engage to keep
  `weight_idx = k * N_CHANNELS + c` indexing within the
  construction-time weights buffer. Counter-case at
  `(ppd=75.402, 4000×3000)` (standard-4K corpus) shows the cap
  is dormant for typical inputs (`233ed177`, tick 346).
- `tests/column_name.rs` — five regression tests pin the
  `CVVDP_COLUMN_NAME` contract that downstream parquet sidecars
  depend on: non-empty, `cvvdp_` prefix, parquet-safe chars
  (ASCII alnum + underscore only), default form encodes the
  crate version (`cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`),
  claims the reserved `cvvdp_imazen_*` tag. Version + tag
  assertions skip gracefully when `CVVDP_IMPL_TAG` is set at
  compile time so the override path stays a free-form escape
  hatch (`a08d79a0`, tick 345).

#### cvvdp-gpu (api)

- `kernels/mod.rs` module-level "Numerical parity target"
  paragraph claimed the 0.005 JOD bound against pycvvdp v0.5.4
  without scoping it to a `PerfMode`. Same gap I closed in
  `src/lib.rs` Status (tick 323) and in the kernels overview.
  Updated to "under the default `crate::PerfMode::Strict`" with
  a one-line forward-reference that future `PerfMode::Fast`
  optimizations gate inside individual kernel dispatch sites
  and leave the strict path as-described. Tick 332.

- `host_scalar::predict_jod_still_3ch` docstring didn't mention
  `PerfMode`, which left readers asking whether the host-scalar
  reference path responds to `PerfMode::Fast`. Answer: no — it's
  the canonical f32-precision reference that every
  `PerfMode::Strict` parity test validates against, and Fast-mode
  optimizations apply only to the GPU pipeline (gated on
  `Cvvdp::params.perf_mode`). Added an explicit "Always runs the
  strict path" paragraph with intra-doc links to `PerfMode`,
  `PerfMode::Strict`, and `Cvvdp::compute_dkl_jod_host_pool`
  (the right answer for callers who want portable perf + the
  strict numerical contract). All 8 doctests still pass; `cargo
  doc` zero-warning. Tick 331.

- `kernels::pool::pool_band_finalize` docstring said "Finish
  the host-side fold for `pool_band_kernel`", but the function
  is used to finalize partials from BOTH `pool_band_kernel`
  (test-only, post-tick-291) AND the fused `pool_band_3ch_kernel`
  (production). The finalize algebra doesn't care which kernel
  wrote the partial — both store the raw `safe_pow(|x|, β)`
  contribution at `partials[partial_idx]` for the host to fold.
  Updated the docstring to name both kernels and call out the
  shared semantics. Tick 330.

- **`burn-conv-spike` clippy `approx_constant`** — the perlin-
  pattern frequency literal `6.28` in
  `crates/burn-conv-spike/src/main.rs:40` trips clippy's
  `approx_constant` lint (close to but not exactly TAU). Added
  a targeted `#[allow(clippy::approx_constant)]` over the
  expression with a comment explaining why the literal stays
  frozen: the README's parity number (`rel_diff = 0.000156`)
  was measured against this exact value, and the spike's
  "don't extend this crate" rule keeps the verdict
  configuration stable. `cargo clippy --release` in the spike
  dir is now zero-warning. The spike crate has its own
  `[workspace]` root so it doesn't affect parent CI; this is
  a quality-of-life fix for anyone who `cd`s into the spike
  dir to reproduce the verdict. Pre-existing rustfmt
  formatting issues in the spike's main.rs (from the
  subagent's commit) are intentionally left alone — they're
  cosmetic and modifying them would touch the frozen state.
  Tick 329.

- `docs/PORT_STATUS.md` "Open questions" section had a stale
  claim that `warm_state_invalidates_after_each_documented_dispatcher`
  covers 8 cases — true at tick-249 close but tick 314 extended
  it to 9 (added `compute_dkl_jod_host_pool` as a real
  invalidator the original audit missed). Updated the "Resolved
  ticks 236-249" entry to acknowledge the tick-314 extension
  inline. Also added three new "Resolved" entries summarizing
  the post-249 work that wasn't captured anywhere in
  PORT_STATUS.md:
  - **Tick 313-315**: sibling regression-test coverage gaps to
    the tick 236-249 audit (warm-ref ppd-mismatch test,
    non-invalidator dual coverage for cpu host_pool warm-ref).
  - **Tick 322**: `PerfMode` enum framework + the two regression
    tests pinning the no-op contract (GPU pool with 1e-4
    tolerance; cpu host-pool with bit-equality).
  - **Tick 324**: Burn-based port abandoned with measured
    4.32× regression numbers + recommended next perf lever.
  PORT_STATUS.md is now current through tick 328. Tick 328.

- **`perf_mode_fast_matches_strict_on_cpu_host_pool`** —
  cpu-runtime sibling of the GPU-side
  `perf_mode_fast_matches_strict_today` (in
  `tests/pipeline_score.rs`). The GPU-pool test had to relax
  to a 1e-4 tolerance because `pool_band_3ch_kernel` uses
  `Atomic<f32>::fetch_add` whose reduce order is
  non-deterministic across runs. The cpu-runtime host-pool
  path bypasses that atomic entirely (reads D bands back to
  host then folds via deterministic sequential f32
  `lp_norm_mean`), so Fast vs Strict CAN be pinned to
  bit-equality via `.to_bits()`. Covers both
  `compute_dkl_jod_host_pool` (cold) and
  `compute_dkl_jod_host_pool_with_warm_ref` (warm). When a
  real Fast-mode optimization lands on the host-pool path
  this test relaxes to the documented per-stage drift budget
  for the cpu/host-pool case. Tick 327.

- Tick 324's "abandon Burn port" verdict reached into three
  surviving stale references in cvvdp-gpu docs that still
  pitched the Burn port as a "future" direction:
  - `src/lib.rs:160` (`CVVDP_COLUMN_NAME` docstring) — replaced
    "future Burn-based port" with "future alternative
    implementation" + an explicit "(A Burn-based port was
    investigated and abandoned tick 324; see
    `docs/BURN_PORT_PLAN.md`'s banner.)" qualifier.
  - `README.md:175,184` (Sweep-tooling section) — same
    treatment plus a sentence with the empirical justification
    ("4.32× regression vs. the hand-written separable kernel")
    pointing at both BURN_PORT_PLAN.md and the spike's README.
  - `docs/CVVDP_SIDECAR_SCHEMA.md:52` (Reserved-tags table) —
    `cvvdp_burn_v*` row's producer column now reads
    "(abandoned tick 324; the Burn port was investigated and
    ruled out...). The tag stays reserved in case a future
    re-attempt wants to reuse it."
  Tag namespace stays reserved (no risk of accidental
  collision if someone DOES revisit the question), but the
  prose tells the reader honestly that the door is closed for
  now. Tick 326.

- **`crates/burn-conv-spike/README.md`** — new top-level README
  for the perf-spike crate that informed tick 324's
  "abandon Burn port" verdict. Documents what the spike is
  (one-shot paper trail, not a maintained crate), the
  measured numbers in a table form (4.32× for the best cubek
  algorithm, 4.98–5.03× for the rest), the root cause
  (CMMA tile waste at 1-channel; im2col→GEMM memory-traffic
  overhead), the recommended actionable next lever
  (shared-memory tiling of the existing direct stencil), and
  how to re-run. Also adds a "Don't extend this crate" note —
  future re-investigations should spin up a sibling
  `burn-conv-spike-v2/` so this one stays frozen at the
  configuration that produced the verdict. Tick 325.

- **Burn port plan marked ABANDONED.** Tick 322's PerfMode
  framework was paired with a perf spike at
  `crates/burn-conv-spike/` (`e101c895`, run on RTX 5070 sm_120
  at 4000×3000 f32) that compared the proposed
  `cubek::conv2d(5×1) + conv2d(1×5)` separable replacement
  against our hand-written direct-stencil `downscale_kernel`.
  Result: **4.32× slower** even with the best cubek algorithm
  choice (`SimpleSyncCyclic + Mma`, 1.46 ms/op vs 0.34 ms/op
  for the hand-written). Other algorithm choices landed
  4.98–5.03× slower. Root cause: cubek routes conv2d through
  im2col → GEMM with CMMA 16×16×16 tensor-core tiles, which
  waste 15/16 of the work when `in_channels = out_channels = 1`
  and doubles memory traffic vs. a direct stencil. The
  "recover cuDNN-class perf via Burn" pitch doesn't hold for
  our 1-channel separable use case. `docs/BURN_PORT_PLAN.md`
  now has a "Status: ABANDONED" banner up top pointing at the
  spike + recommending shared-memory tiling of the existing
  direct stencil as the actionable next perf lever. The
  surviving content stays as design context.

- **`perf_mode_fast_matches_strict_today` regression test fix +
  extension.** The tick-322 form asserted bit-pattern equality
  via `.to_bits()`; tick 324 (this tick) surfaced that this
  was wrong — two separate `Cvvdp` instances running the same
  inputs can disagree by 1 ULP because `pool_band_3ch_kernel`
  uses `Atomic<f32>::fetch_add` whose reduce order is
  non-deterministic across runs (`CHROMA_DRIFT_INVESTIGATION.md`
  documents the ~1e-5-abs floor over O(10⁴) pixels). The
  tick-322 test passed by chance on the small 64² fixture; the
  warm-ref extension I added in this tick caught the latent
  bug. Switched to `(strict - fast).abs() < 1e-4` (1000× the
  observed 1-ULP noise floor, still well below any real
  Fast-mode optimization's drift budget like 0.005 for
  nearest-CSF or 0.01 for f16 pyramid). Extended coverage to
  `compute_dkl_jod_with_warm_ref` and `Cvvdp::score` so a
  refactor that wired `perf_mode` through one entry point but
  not another would surface. Tick 324.

- **`PerfMode` surfaced in user-facing docs**. Tick 322 added
  the framework; this tick wires it into the discoverable
  surface area:
  - `crates/cvvdp-gpu/README.md` gets a new "Parity vs. perf —
    `PerfMode`" section between "CPU backend" and "Features"
    with a code example showing the struct-update opt-in
    pattern.
  - `src/lib.rs` "Status" section now scopes the 0.005 JOD
    parity claim to `PerfMode::Strict` explicitly and notes
    that Fast is a no-op today (pointing at the regression
    test).
  - Also fixed a pre-existing `rustdoc::broken_intra_doc_links`
    ambiguity warning at `pool.rs:161` — `[`pool_band_3ch_kernel`]`
    was ambiguous between the function and the auto-generated
    module of the same name from `#[cube(launch)]`. Switched to
    the `()` disambiguation form for the function reference.
    `cargo doc -p cvvdp-gpu` is now zero-warning. Tick 323.

- **`PerfMode` enum** opens the parity-vs-perf opt-in surface
  on the public API. Two variants:
  - `PerfMode::Strict` (default) — matches pycvvdp v0.5.4
    bit-for-bit within f32 noise, exactly what every parity test
    in `tests/` is calibrated against.
  - `PerfMode::Fast` — opt-in entry point for future stage-level
    relaxations that trade measurable per-call cost for a
    bounded JOD drift vs. Strict. Currently a no-op (no
    Fast-mode fast paths have landed yet); the variant exists so
    callers can wire the opt-in once and individual stages can
    later gate on `params.perf_mode == Fast` without forcing a
    breaking change.
  Plumbed through `CvvdpParams::perf_mode` (new field, defaults
  to `Strict` in `CvvdpParams::PLACEHOLDER`) → stored on
  `Cvvdp` via the existing `params` field. Re-exported from
  `cvvdp_gpu` for convenience (`use cvvdp_gpu::PerfMode;`).
  Regression test `perf_mode_fast_matches_strict_today` (in
  `tests/pipeline_score.rs`) pins the bit-pattern-equality
  invariant; when a real Fast-mode optimization lands the test
  should be RELAXED (not deleted) to the documented per-stage
  drift budget for that optimization. `doctest` count grows
  from 6 → 8 (two new examples in the `PerfMode` doc comment).
  Tick 322.

#### cvvdp-gpu

- **`Cvvdp::compute_dkl_jod_host_pool`** — CPU-backend-compatible
  variant of `compute_dkl_jod`. Reads D bands back to host and
  pools them with the host-scalar `lp_norm_mean` instead of the
  GPU `pool_band_3ch_kernel` (which uses `Atomic<f32>::fetch_add`,
  unsupported by `cubecl-cpu`). Same JOD output as
  `compute_dkl_jod` to f32 noise (`diff = 0.000000` measured on
  the 32×32 odd-dim test pair); use it on the CPU backend or
  any runtime that lacks atomic f32 add. New
  `compute_dkl_jod_host_pool_matches_compute_dkl_jod` test pins
  the two paths together. Closes the standing CPU-backend
  blocker noted in `lib.rs`.
- **`tests/cpu_backend.rs`** — cpu-runtime smoke + parity tests
  exercising `compute_dkl_jod_host_pool` on `cubecl::cpu::CpuRuntime`.
  Validates the lib.rs claim that the cpu backend works:
    JOD finite + in [0, 10] on a 32×32 synth pair.
    cpu backend JOD vs host_scalar JOD: `diff = 0.000000`.
  All other test files gate themselves out of cpu-only builds; this
  file is the only place cpu-backend coverage lives.
  Run with `cargo test -p cvvdp-gpu --no-default-features --features cpu`.

#### cvvdp-gpu (docs)

- `Cvvdp::score` now has a `no_run` doctest example showing the
  canonical `Cvvdp::<CudaRuntime>::new` → `.score(&ref, &dist)`
  shape against a 64×64 byte-identical pair. Fills the only
  remaining doc gap on the crate's headline public entry point —
  the host-only and host-pool paths already had doctests via
  `host_scalar::predict_jod_still_3ch`, `compute_dkl_jod_host_pool`,
  and `compute_dkl_jod_host_pool_with_warm_ref`.

- **`crates/cvvdp-gpu/README.md`** — new crate-root README
  mirroring the peer GPU-metric crates' structure
  (`ssim2-gpu`, `zensim-gpu`, `dssim-gpu` all had one;
  cvvdp-gpu didn't). Covers the multi-vendor pitch
  (CUDA / WGPU / HIP / cubecl-cpu), single-image + cached-ref
  + warm-ref usage shapes, JOD 0..10 score interpretation
  (higher = better, matching pycvvdp convention), the
  `compute_dkl_jod_host_pool` workaround for cubecl-cpu and
  Metal (Atomic<f32>::fetch_add gotcha), the
  `CVVDP_COLUMN_NAME` / `CVVDP_IMPL_TAG` sweep-tooling story
  for parquet sidecars, the `parity-goldens` feature gate,
  and the standard build / license footer. Tick 285.

- README "Sweep tooling" section now links to
  `docs/CVVDP_SIDECAR_SCHEMA.md` (full identity-tuple +
  score-column + manifest spec) and `docs/BURN_PORT_PLAN.md`
  (scoping for the future `cvvdp_burn_v*` column that would
  land alongside `cvvdp_imazen_v*` and `cvvdp_pycvvdp_v054`).
  Closes the navigability gap a reader following
  `CVVDP_COLUMN_NAME` would hit. Tick 287.

### Fixed

#### cvvdp-gpu

- **Warm-ref state invalidation honored on all 6 dispatchers that
  overwrite `bands_ref`.** Tick 236 fixed the two weber-chain
  dispatchers (`compute_dkl_weber_pyramid`,
  `compute_dkl_t_p_bands`); tick 237 audits the rest and finds
  two more silent-stale-scalar holes through the Laplacian chain:
  `compute_dkl_laplacian_pyramid` and `compute_dkl_csf_weighted_bands`
  both run `_dispatch_laplacian_pyramid_gpu` which overwrites
  `bands_ref[k].planes[c]` with Laplacian bands (not the Weber
  bands the warm-ref state was built on). Pre-fix, a subsequent
  `compute_dkl_jod_with_warm_ref` would silently mix Laplacian
  bands against the cached Weber-baseband scalar. Both functions
  now clear `warm_ref_baseband_log_l_bkg` at entry; the
  `Cvvdp::warm_reference` docstring lists all 6 invalidators;
  and the regression test
  `warm_state_invalidates_after_each_documented_dispatcher`
  extends from 4 → 6 cases.

  Tick 238 closes the audit: the `warm_reference` docstring now
  also documents `Cvvdp::score` / `Cvvdp::score_with_reference`
  as transitive invalidators (via `compute_dkl_jod` since tick
  213) and `Cvvdp::set_reference` as an explicit non-invalidator
  (it only stashes host bytes). Regression test extends 6 → 8
  invalidator cases; new sibling test
  `set_reference_does_not_invalidate_warm_state` pins the
  non-invalidator contract — a future refactor that turned
  `set_reference` into an eager GPU dispatch would silently
  break batch-scoring callers and surface here.

### Changed (docs, tests, dedup — post-tick-238)

Many small docs / tests / dedup chunks landed under this bucket
during the ticks 239-273 maintenance run. They follow the
Keep-a-Changelog `Changed` semantics (no behavioural shift in the
public API; refactors, comment refreshes, regression-test pinning,
and helper extractions). Pre-tick-238 entries above stay in their
original Fixed/Added/Changed sections.

### Changed (post-tick-238)

#### cvvdp-gpu

- `crates/cvvdp-gpu/docs/PORT_STATUS.md` pipeline-stage table
  (line 15, "Per-band pooling" row) named the test-only
  `pool_band_kernel` as the GPU kernel consumed by
  `compute_dkl_jod`. Same stale-reference shape as tick
  319's README fix — the production dispatcher is the fused
  3-channel `pool_band_3ch_kernel`, per the tick-291 audit.
  Updated the row to: "GPU `pool_band_3ch_kernel` (fused
  3-channel, atomic f32 partials, one launch per pyramid band)
  consumed by `compute_dkl_jod`. Single-channel
  `pool_band_kernel` retained as a test-only entry point."
  The "Resolved tick 208" entry further down (line 112) keeps
  its `pool_band_kernel` reference — that's accurate
  historical context (tick 208 predates the tick 165 fusion
  to `pool_band_3ch_kernel`). Tick 320.

- `crates/cvvdp-gpu/README.md` "CPU backend" section had a
  stale reference to `pool_band_kernel` (the single-channel
  pool kernel) as the source of the `Atomic<f32>::fetch_add`
  that cubecl-cpu doesn't support. Tick 291's audit
  established that production dispatches the fused
  `pool_band_3ch_kernel` instead (one launch per pyramid band,
  3× fewer launches than the single-channel form); the
  `pool_band_kernel` symbol is retained only for the
  `tests/pool_scalar.rs::pool_band_kernel_matches_host_lp_norm_mean`
  unit-parity test. Updated the README to name the fused
  3-channel production kernel + the "one launch per pyramid
  band" descriptor. The cpu-runtime workaround (route through
  `compute_dkl_jod_host_pool`) is unchanged. Tick 319.

- `score_with_reference_errors_without_set_reference` (in
  `tests/pipeline_score.rs`) was using
  `format!("{err:?}").contains("NoCachedReference")` to verify
  the error variant. Substring matching on Debug output is
  brittle — a future variant rename that landed
  "NoCachedReferenceV2" or similar by accident would silently
  pass the substring check. Other tests in the same file
  (`invalid_image_size_surfaces_on_too_small_dims`,
  `dimension_mismatch_surfaces_on_wrong_size_inputs`) use
  proper `match err { Error::X => {}, other => panic! }`
  pattern matching on the variant via the public Error API,
  which pins identity directly. Switched this test to the
  same pattern. Test passes post-change. Tick 318.

- `Error::NoWarmReference` variant docstring listed two
  example invalidators (`compute_dkl_jod`,
  `compute_dkl_d_bands`) — fine when the list was short, but
  tick 314 grew the canonical invalidator list to 9
  (compute_dkl_jod, ..._host_pool, _d_bands, _weber_pyramid,
  _t_p_bands, _laplacian_pyramid, _csf_weighted_bands, score,
  score_with_reference). Maintaining a parallel example list
  on the Error variant is a duplicate-maintenance burden the
  variant doc would inevitably fall behind on — exactly what
  tick 314's audit caught.
  Removed the per-example list and instead pointed at
  `Cvvdp::warm_reference`'s docstring as the canonical
  invalidator source, plus a pointer to the
  `warm_state_invalidates_after_each_documented_dispatcher`
  regression test that pins each method to the contract. Also
  added the cpu-runtime variant
  (`compute_dkl_jod_host_pool_with_warm_ref`) to the
  "called without prior `warm_reference`" sentence — the
  variant doc previously only named
  `compute_dkl_jod_with_warm_ref` even though both methods
  return this variant from the same `.ok_or(NoWarmReference)`
  site. 14 pipeline_score tests still pass. Tick 317.

- `Error::InvalidImageSize` Display message was misleading.
  The variant is documented as dual-purpose (image too small
  for the configured pyramid OR GPU readback/dispatch failed,
  because cubecl's read errors aren't separable yet) but
  the Display impl only mentioned the image-size case:
  "image is too small for the configured pyramid". A user
  hitting a GPU readback failure would see this and
  investigate image dimensions instead of the actual backend
  failure. Updated to: "image too small for the configured
  pyramid, or GPU readback/dispatch failed (see the
  InvalidImageSize variant docs — cubecl's read errors aren't
  separable yet so both surface as this variant)". Also
  extended `error_display_messages_are_actionable` to pin
  the new dual-purpose hint by asserting the message contains
  "GPU"/"readback"/"dispatch" in addition to the existing
  "small"/"pyramid" check. Test passes. Tick 316.

- `gauss_chain_helpers_do_not_invalidate_warm_state` regression
  test extended from 2 → 3 non-invalidators. Adds
  `compute_dkl_jod_host_pool_with_warm_ref` — pinning the
  tick-314 docstring claim that this method only READS the
  cached scalar (`.ok_or(NoWarmReference)`) and must preserve
  warm state across calls. A refactor that accidentally
  cleared the cached scalar (e.g. moving the warm-ref
  host-pool path through `_dispatch_d_bands_into_scratch` by
  mistake) would silently break cpu-runtime batch scoring —
  this case catches it. Test passes. Tick 315.

- `Cvvdp::warm_reference` docstring's invalidator list was
  missing `compute_dkl_jod_host_pool` — it routes through
  `_dispatch_d_bands_into_scratch` →
  `_dispatch_ref_weber_pyramid_only` which clears
  `warm_ref_baseband_log_l_bkg`, same as the GPU jod path. A
  caller batch-scoring on cpu-runtime who mixed
  `compute_dkl_jod_host_pool` calls between
  `warm_reference` + `compute_dkl_jod_host_pool_with_warm_ref`
  would silently lose the warm state without the docstring
  warning. Added the missing entry; also noted explicitly that
  `compute_dkl_jod_host_pool_with_warm_ref` does NOT invalidate
  (it only reads the cached scalar).
  Extended the `warm_state_invalidates_after_each_documented_dispatcher`
  regression test from 8 → 9 invalidators to pin the
  `compute_dkl_jod_host_pool` contract directly. Test passes.
  Tick 314.

- New regression test:
  `debug_assert_fires_when_ppd_mismatches_geometry_on_warm_ref_path`
  in `tests/pipeline_score.rs`. Sibling to the existing tick-244
  test that pinned the tick-243 `debug_assert_ppd_matches_geometry`
  contract on `compute_dkl_jod`. All 6 public methods share the
  same assert-at-entry contract (`compute_dkl_jod` /
  `compute_dkl_d_bands` / `compute_dkl_t_p_bands` /
  `compute_dkl_jod_host_pool` /
  `compute_dkl_jod_host_pool_with_warm_ref` /
  `compute_dkl_jod_with_warm_ref`), but only `compute_dkl_jod`
  had a regression test. A refactor that dropped the assert
  from `compute_dkl_jod_with_warm_ref` specifically would have
  slipped through. The new test warms a reference, then calls
  `compute_dkl_jod_with_warm_ref` with a phone-resolution PPD
  (110.09 ≠ STANDARD_4K's 75.4) and expects the debug-only
  assert to fire. Both ppd-mismatch tests pass; the
  `#[cfg(debug_assertions)]` gate means release builds skip
  the test definition entirely (matches the existing pattern).
  Tick 313.

- Dropped dead `ppd: f32` parameter from two private GPU
  dispatchers (`_dispatch_d_bands_into_scratch` and
  `_dispatch_d_bands_dist_and_band_loop`) plus the redundant
  `let _ = ppd;` discard in `compute_dkl_t_p_bands`.
  All 6 public methods that take `ppd` validate it via
  `debug_assert_ppd_matches_geometry(ppd)` at entry; the value
  itself isn't consumed by the GPU stages (logs_row is
  pre-uploaded against the construction-time geometry at
  `Cvvdp::new` time, so the runtime band-loop kernel reads
  the cached rho-per-band instead of recomputing from ppd).
  The private helpers were threading ppd through dead until
  the `let _ = ppd;` discard at the bottom of each. Updated
  signatures and all 5 call sites
  (3× `_dispatch_d_bands_into_scratch` from
  `compute_dkl_jod` / `compute_dkl_d_bands` /
  `compute_dkl_jod_host_pool` and
  2× `_dispatch_d_bands_dist_and_band_loop` from
  `compute_dkl_jod_with_warm_ref` /
  `compute_dkl_jod_host_pool_with_warm_ref`).
  Public method signatures unchanged. 14 `pipeline_score`
  tests + `compute_dkl_jod_matches_pycvvdp_at_12mp_synth`
  all pass — JOD output is byte-identical. Tick 312.

- Dropped two unused dev-dependencies from
  `crates/cvvdp-gpu/Cargo.toml`:
  - `bytemuck` — zero references anywhere in `src/`, `tests/`,
    `examples/`, or `benches/`. The cubecl APIs we use
    (`f32::as_bytes` / `client.create_from_slice`) pull
    bytemuck transitively where needed.
  - `serde` — only `serde_json` is used directly; `serde_json`
    pulls the `serde` traits transitively.
  Pure dependency hygiene; cargo no longer asks rustc to link
  these two crates into the dev-build. `cargo build --tests
  --benches --examples` is clean under both `--features cuda`
  and `--features wgpu`. `cvvdp_score_matches_v1_manifest`
  still passes. Tick 311.

- `kernels::csf::interp1_clamped` binary-search midpoint switched
  from `(lo + hi) / 2` to `usize::midpoint(lo, hi)` (stable since
  Rust 1.85; workspace MSRV 1.93). Overflow-safe by construction
  and matches the canonical idiom — clippy `-W clippy::pedantic`
  was suggesting it. The shorthand can't overflow at our
  32-entry LUT sizes, but the explicit form documents intent
  and removes a pedantic-lint speed bump for anyone who flips
  `clippy::pedantic` on. `csf_scalar` parity tests
  (`sensitivity_matches_pycvvdp_v0_5_4`,
  `precomputed_band_weights_match_pointwise`,
  `flatten_band_weights_layout`,
  `sensitivity_is_finite_at_extremes`) still all pass. Tick 295.

- `kernels::pyramid::weber_contrast_pyr_dec_scalar` nested
  `fn build_pyr` helper was defined mid-body, after the
  `n_levels` resolution and `debug_assert!` statements, which
  clippy `-W clippy::pedantic`'s `items_after_statements` lint
  flags as confusing (items exist from the start of scope; the
  visual ordering implies a runtime dependency that doesn't
  exist). Moved the nested fn ahead of the statements without
  changing its body — pure reordering. `pyramid_scalar` parity
  tests (6 tests including `reduce_matches_pycvvdp` and
  `one_band_laplacian_matches_pycvvdp`) and the lib-internal
  `kernels::pyramid::tests` (`reduce_halves_dimensions`,
  `reduce_preserves_constant_signal`, `expand_*`, etc.) all
  still pass. Tick 296.

- `tests/pipeline_color.rs` had 9 identical
  `const TOLERANCE: f32 = 0.005;` declarations, each inside a
  separate test function and each tripping clippy
  `-W clippy::pedantic`'s `items_after_statements` (the const
  followed the per-test `let pycvvdp_golden_jod = ...` golden
  load). Hoisted to file scope as a single
  `const TOLERANCE: f32 = 0.005;` with a docstring tying the
  number to the tick 207 tolerance schedule. The 9 inner
  declarations are gone; their bodies still reference
  `TOLERANCE` via outer-scope lookup. All 31 pipeline_color
  tests still pass post-change (including the 12 MP parity
  pair that takes ~30 s/test). Tick 297.

- Closed out the lossless-cast cleanup arc with the remaining
  6 `cast_lossless` warnings:
  - `src/pipeline.rs:731` (`b as u32`) — sRGB-byte → u32 lane
    pack inside the persistent `src_u32_scratch` fill loop
    (warm-ref-amortised version of the per-call buffer pack).
  - `tests/color_kernel.rs:47` (`|&b| b as u32`) — same
    byte → u32 pack inside the color-kernel parity test setup.
    Also rewrote the closure as `.copied().map(u32::from)` so
    the `Copy` bound replaces the explicit deref pattern.
  - `src/pipeline.rs:2507` and `:2598` (`jod as f64`) — the
    public `Cvvdp::score` and `Cvvdp::score_with_reference`
    return-value widenings.
  - `examples/manifest_parity_probe.rs:100`
    (`r[(y*w + x)*3 + c] as i32`) — synth_noise_pair
    distortion construction; widens u8 to i32 for the +noise
    addition.
  - `tests/pipeline_color.rs:2075`
    (`ref_srgb[i + c] as i64`) — same noise-fixture pattern
    inside the 256² parity test.
  Lossless-widening warning count across cvvdp-gpu is now zero
  after ticks 300/301/302/304/305 (u16/i16/u64/u32/i32/i64/f64
  variants). Tests that exercise the changed code paths
  (`color_kernel::srgb_to_dkl_kernel_matches_host_scalar`,
  `pipeline_color::compute_dkl_jod_matches_pycvvdp_at_256x256_noise`)
  pass post-change. Tick 305.

- Two strict-equality `f32` `assert_eq!` calls in
  `tests/pool_scalar.rs` were tripping clippy
  `-W clippy::pedantic`'s `float_cmp` — but both are
  intentional bit-pattern equality checks
  (kernel-test invariants about untouched partial slots and
  fill-kernel output), not approximate-equality assertions.
  Switched to `.to_bits()` comparisons (`partials[i].to_bits()
  == 0.0_f32.to_bits()`, `v.to_bits() == value.to_bits()`)
  with a one-line comment per site explaining why bit-pattern
  equality is the correct test. `pool_scalar` test suite
  (8 tests) all pass post-change including
  `gpu::pool_band_kernel_matches_host_lp_norm_mean` and
  `gpu::fill_f32_kernel_writes_uniform_value`. Tick 304.

- Two `for x in container.iter()`/`for x in container.iter_mut()`
  sites switched to the more idiomatic `for x in &container`
  / `for x in &mut container` form (clippy
  `-W clippy::pedantic`'s `explicit_iter_loop`):
  - `tests/common/mod.rs:104` (hex-encode loop over `Sha256`
    finalize output, called by `manifest_sha256_hex` /
    `fetch`).
  - `tests/pipeline_color.rs:143` (host pyramid reduce
    loop over per-channel plane buffers).
  The `tests/common/mod.rs` site clears 3× because the file is
  consumed via `#[path]` from bench/example/test scopes (4
  total `explicit_iter_loop` warnings cleared across cvvdp-gpu).
  Also dropped two unnecessary trailing commas in
  `tests/pipeline_score.rs::dimension_mismatch_surfaces_on_wrong_size_inputs`'s
  `assert_eq!` macro calls (clippy `unnecessary_trailing_comma`).
  All affected tests
  (`dimension_mismatch_surfaces_on_wrong_size_inputs` plus
  the parity tests that reduce host pyramids) pass post-change.
  Tick 303.

- Six `u32 as u64` lossless widening casts switched to
  `u64::from(...)`. Sites:
  - `tests/common/mod.rs:409`
    (`v1_corpus_jod_golden` parameter comparison
    `Some(q as u64)`).
  - `examples/time_12mp.rs:132` per-pixel cost math
    (`(W as u64) * (H as u64)` for total-pixels divisor).
  - `examples/time_size_sweep.rs:104` per-bucket pixel
    count.
  - `benches/score.rs` twice — both `Throughput::Elements`
    calls in the GPU-JOD bench setup.
  Clippy `-W clippy::pedantic`'s
  `cast_lossless` flagged them with `an as cast can become
  silently lossy if the types change in the future` — the
  `u64::From<u32>` impl encodes the widening intent
  explicitly and would surface a hard compile error if a
  caller swapped `u32` for a wider type. Throughput-math and
  the corpus-q comparison evaluate to the same `u64`.
  `pipeline_score::cvvdp_score_matches_v1_manifest` (the
  primary consumer of `v1_corpus_qs` → `v1_corpus_jod_golden`)
  still passes. Tick 302.

- Seven `u8 as i16` widening casts in the
  `chroma_shift` synth-pair pattern
  (`(byte as i16 + 16).clamp(0, 255) as u8`) switched to
  `i16::from(byte) + 16` for the lossless widening.
  Six sites in `tests/pipeline_color.rs` (one per
  `chroma_shift`-family test, all using the same `.flat_map`
  closure) plus one in
  `examples/manifest_parity_probe.rs::synth_chroma_shift_pair`.
  Bit-identical arithmetic; all 9 `chroma_shift` parity tests
  (`compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift`,
  `..._with_warm_ref`, plus the 7 per-stage shadow tests)
  still pass. Tick 301.

- Six `u8 as u16` widening casts (3 each in
  `tests/pipeline_color.rs::compute_dkl_jod_matches_pycvvdp_at_256x256_blur3x1`
  and `..._blur1x3`) and six more in
  `examples/manifest_parity_probe.rs::synth_blur3x1_pair` and
  `synth_blur1x3_pair` switched to `u16::from(...)`. Clippy
  `-W clippy::pedantic`'s
  `cast_lossless`/`unnecessary_cast` family flagged them
  because the `From<u8>` impl for `u16` is infallible — the
  bare `as` form works but is the wrong idiom for
  lossless widening conversions. The blur synth-pair fixtures
  (3-tap horizontal / 3-tap vertical mean over `u16`
  accumulator) produce identical pixel arithmetic; the 2
  blur parity tests
  (`compute_dkl_jod_matches_pycvvdp_at_256x256_blur3x1`,
  `..._blur1x3`) still pass. Tick 300.

- `tests/common/mod.rs` had 3 sites using Debug formatting
  (`{path:?}`, `{local:?}`) inside `panic!` for path-typed
  values, which clippy `-W clippy::pedantic`'s
  `unnecessary_debug_formatting` flags — Debug for
  `&Path`/`&PathBuf` renders with surrounding quotes and escape
  sequences (e.g. `"foo bar.png"`), while Display via
  `path.display()` shows the bare path (`foo bar.png`). For
  panic-context error messages the Display form is more
  readable. Switched all 3 sites
  (`fetch` cache-write panic at line 90,
  `load_rgb_bytes` open + decode panics at 449/451) to
  positional format args using `path.display()` /
  `local.display()`. The 3 unique warnings were each counted
  3× because `tests/common/mod.rs` is consumed via `#[path]`
  from the bench/example/test scopes (9 total pedantic
  warnings cleared). All 31 `pipeline_color` tests still pass
  post-change (the suite consumes `load_rgb_bytes` via
  `common::Backend` + image-corpus paths). Tick 299.

- `tests/common/mod.rs` had 4 sites using closure-wrapped
  method calls (`.and_then(|j| j.as_f64())`,
  `.and_then(|n| n.as_u64())`) that
  clippy `-W clippy::pedantic`'s
  `redundant_closure_for_method_calls` flags — the bare method
  pointer form is shorter and stylistically preferred. Each
  site (in `pycvvdp_synth_golden_jod`,
  `v1_corpus_jod_golden`, and `v1_corpus_qs`) now uses
  `.and_then(serde_json::Value::as_f64)` or
  `.and_then(serde_json::Value::as_u64)`. The 4 sites are each
  re-counted 3× because `tests/common/mod.rs` is consumed via
  `#[path]` from the bench/example/test scopes (12 total
  pedantic warnings cleared). `pipeline_score` corpus tests
  (`compute_dkl_jod_on_v1_manifest_corpus`,
  `score_with_reference_matches_score`,
  `cvvdp_score_matches_v1_manifest`, and 11 more) all still
  pass post-change. Tick 298.

### Fixed (post-tick-238)

#### cvvdp-gpu (docs)

- `kernels::csf::interp1_uniform` had a 14-line docstring whose
  opening 4 lines ("1-D linear interpolation in log-space along
  a monotonically increasing axis…") read as a generic
  intro that applied to either interpolator, blurring into the
  function-specific "Linear interp on a UNIFORMLY-spaced axis"
  description on line 78 with no separator. `interp1_clamped`
  underneath had no docstring at all.
  Rewrote both as standalone docs:
  - `interp1_uniform`: now opens with "uniformly-spaced axis
    via global-stride rescale" (the actual semantics), keeps the
    tick 199 / chroma-drift parity rationale, and adds an
    explicit "used for the outer L_bkg interp" pointer to
    `sensitivity_scalar` / `precompute_logs_row`.
  - `interp1_clamped`: gains a docstring explaining the
    binary-search bracket form, its applicability to any
    monotonically-increasing axis (uniform or not), and the
    pycvvdp-side rationale for using it on the inner rho axis
    (`torch.searchsorted` + linear interp) versus the L_bkg
    axis (`interp1q`). Cross-references `interp1_uniform`.
  Tick 294.

- `kernels/mod.rs` step 5 breadcrumb introduced in tick 291
  triggered 3 `clippy::doc_lazy_continuation` warnings — the
  new sentence about `pool_band_kernel` being test-only landed
  on continuation indentation (`//!    `) of bullet 5, which
  rustdoc/clippy now read as a malformed sub-list rather than
  body continuation. Split the breadcrumb out of the bullet
  into its own paragraph (blank `//!` separator) so it parses
  as flowing prose under the pipeline-order list. `cargo
  clippy --all-targets -W clippy::all` is back to zero
  warnings on both `--features cuda` and `--features wgpu`.
  Tick 293.

- `kernels::pool::pool_band_kernel` (single-channel) doc now
  explicitly notes it's not dispatched by
  `Cvvdp::compute_dkl_jod` — the production path uses the fused
  3-channel `pool_band_3ch_kernel`. Added a one-line breadcrumb
  pointing at the
  `tests/pool_scalar.rs::pool_band_kernel_matches_host_lp_norm_mean`
  parity test that justifies keeping the symbol public. A
  maintainer reading this kernel was previously left guessing
  why two near-duplicate `pool_band_*` kernels coexist; now the
  link from single → fused is symmetric (the 3ch docstring
  already cross-references the single-channel form as the
  base case). Tick 292.

- Pipeline-overview docstrings in `kernels/mod.rs` step 5,
  `pipeline.rs` step 6, `Cvvdp::compute_dkl_d_bands`'s "no
  readback" note, and `Cvvdp::compute_dkl_jod`'s ASCII pipeline
  diagram all referred to the GPU pool stage as `pool_band_kernel`.
  The production path dispatches `pool_band_3ch_kernel` (one
  fused 3-channel launch per band, ~3× fewer launches than the
  single-channel version) — `pool_band_kernel` survives only as
  a unit-test entry point in `tests/pool_scalar.rs`. Updated
  all 4 sites to name `pool_band_3ch_kernel` and noted the fused
  3-channel-per-launch property; added a one-line breadcrumb in
  `kernels/mod.rs` that `pool_band_kernel` is retained for the
  pool-scalar unit test. Tick 291.

- `cargo fmt --check` was failing on cvvdp-gpu — tick 298's
  `|n| n.as_u64()` → `serde_json::Value::as_u64` swap inside
  `v1_corpus_qs::filter_map` made the line too long for
  rustfmt's column limit, but the file wasn't reformatted at
  the time. Ran `cargo fmt -p cvvdp-gpu` to clean it up; the
  change splits the `filter_map` body across 4 lines (still
  consumed via `#[path]` from bench/example/test scopes).
  Also picked up several pipeline_color.rs `let (ref_srgb,
  dist_srgb) = common::synth_pair_*(...)` 2-line forms that
  now fit on a single line after ticks 278-280 shortened the
  helper names. `cargo fmt -p cvvdp-gpu --check`: clean.
  `pipeline_score::cvvdp_score_matches_v1_manifest` passes
  post-change. Tick 310.

- Added `#[must_use]` to 24 pure-return `pub fn`s where ignoring
  the return value is always a bug (clippy
  `-W clippy::must_use_candidate`). These are all
  host-scalar / kernel helpers and one method
  (`DisplayGeometry::pixels_per_degree`); the attribute is purely
  additive — callers that already use the return value are
  unaffected, and callers that drop it on the floor get a
  `unused_must_use` warning surfacing the bug at the use site.
  Breakdown:
  - `src/host_scalar.rs`: `predict_jod_still_3ch` (1)
  - `src/params.rs`: `DisplayGeometry::pixels_per_degree` (1)
  - `src/kernels/color.rs`: `srgb_byte_to_dkl_scalar` (1)
  - `src/kernels/csf.rs`: `sensitivity_scalar`,
    `sensitivity_corrected_scalar`, `precompute_logs_row`,
    `precomputed_band_weights`, `flatten_band_weights` (5)
  - `src/kernels/masking.rs`: `safe_pow`, `clamp_diff_soft`,
    `phase_uncertainty_no_blur`, `gaussian_blur_sigma3`,
    `phase_uncertainty_band`, `mask_pool_pixel`,
    `mult_mutual_pixel`, `mult_mutual_band` (8)
  - `src/kernels/pool.rs`: `lp_norm_mean`, `lp_norm_sum`,
    `met2jod`, `do_pooling_and_jod_still_3ch`,
    `pool_band_finalize` (5)
  - `src/kernels/pyramid.rs`: `band_frequencies`,
    `laplacian_pyramid_dec_scalar`,
    `weber_contrast_pyr_dec_scalar` (3)
  `must_use_candidate` warning count: 0 (was 24). The
  `#[cube(launch)]` GPU kernels are skipped because they return
  `()` and don't trigger the lint. `pool_scalar` (8) and
  `display_geometry` (2) test suites still pass post-change.
  Crate is `publish = false` so no semver implications.
  Tick 309.

- Added `# Panics` sections to 3 `pub fn` host-scalar /
  kernel helpers that can panic on out-of-spec input (clippy
  `-W clippy::missing_panics_doc`):
  - `host_scalar::predict_jod_still_3ch` — panics on
    `ref_srgb.len() != w*h*3` or `dist_srgb.len() != w*h*3`
    (the two `assert_eq!` calls at the top). Doc points at
    `Cvvdp::score` for the fallible `Result` variant routed
    through the same pipeline.
  - `kernels::pool::do_pooling_and_jod_still_3ch` — panics on
    empty `q_per_ch` (zero pyramid levels). cvvdp's pool stage
    is undefined on a zero-band input.
  - `kernels::pyramid::laplacian_pyramid_dec_scalar` — panics
    if the resolved level count is zero. Debug builds trip
    the `debug_assert!`; release builds reach the
    `gauss.pop().expect("at least one level")` line.
  `missing_panics_doc` warning count: 0 (was 3). 6 doctests
  still pass under the CI wgpu combo. Tick 308.

- Closed out the `# Errors`-section work started in tick 306.
  Added sections to the remaining 10 `Result`-returning public
  methods:
  - `compute_dkl_jod` and `compute_dkl_jod_host_pool` /
    `compute_dkl_jod_host_pool_with_warm_ref` — the canonical
    GPU and cpu-runtime JOD entry points documented in
    `lib.rs`. The warm-ref host-pool variant shares the
    tick-248 `DimensionMismatch`-before-`NoWarmReference`
    precedence rule with its all-GPU counterpart and the new
    `# Errors` section documents it.
  - 7 stage-debug helpers (`compute_dkl_planes`,
    `compute_dkl_gauss_pyramid`,
    `compute_dkl_laplacian_pyramid`,
    `compute_dkl_weber_pyramid`, `compute_dkl_t_p_bands`,
    `compute_dkl_d_bands`, `compute_dkl_csf_weighted_bands`) —
    each gets a uniform short `# Errors` section naming
    `DimensionMismatch` and `InvalidImageSize` with the
    specific stage chain that can fail.
  `clippy::missing_errors_doc` warning count: 0 (was 17 at
  start of tick 306). All 6 doctests still pass under the
  CI wgpu combo. Tick 307.

- Added `# Errors` sections to 7 user-facing public entry
  points (`Cvvdp::new`, `Cvvdp::new_with_geometry`,
  `Cvvdp::score`, `Cvvdp::set_reference`,
  `Cvvdp::score_with_reference`, `Cvvdp::warm_reference`,
  `Cvvdp::compute_dkl_jod_with_warm_ref`) — clippy
  `-W clippy::missing_errors_doc` was flagging all 17
  Result-returning public methods, but only the 7 user-facing
  ones really needed dedicated sections (the lower-level
  `compute_dkl_*_bands` helpers are exposed for testing /
  shadowing rather than primary use). Each new section
  enumerates the specific `Error` variants the method can
  return — including the tick-248 precedence audit detail
  that `compute_dkl_jod_with_warm_ref` returns
  `DimensionMismatch` *before* `NoWarmReference` when both
  conditions hold. All 6 doctests still pass under the CI
  wgpu combo (`cargo test --doc --features wgpu`). Cleared
  7 of 17 `missing_errors_doc` warnings. Tick 306.

- `host_scalar::predict_jod_still_3ch` had a stale comment
  claiming "weber_contrast_pyr path which we have NOT yet
  ported (vanilla Laplacian + linear DKL bands here vs.
  cvvdp's Weber-contrast Laplacian + log10(gauss) for L_bkg)".
  The Weber-contrast pyramid was ported in tick 24 (per
  `docs/PORT_STATUS.md`'s "Resolved tick 24" entry) and the
  surrounding code already calls
  `kernels::pyramid::weber_contrast_pyr_dec_scalar` — both
  `ref_weber` / `dis_weber` carry Weber-contrast bands and the
  log10-gauss `log_l_bkg`. Replaced the stale comment with an
  accurate description of the current baseband-bypass +
  non-baseband mult-mutual structure, the tick 204
  `CSF_BASEBAND_RHO = 0.1 cy/deg` override, and a
  forward-reference to the Weber-pyramid port history. Tick 290.

- `PoolingParams` scaffolding docstring referenced
  `BETA_CHANNEL` as the inlined `const` in `kernels::pool`, but
  the actual const there is `BETA_CH` (mirroring cvvdp's
  `beta_tch` field name). Grepping `BETA_CHANNEL` returned no
  hits, leaving a future maintainer reading the struct without
  a working pointer to the production value. Replaced
  `BETA_CHANNEL` with `BETA_CH` and added a one-line mapping
  note from the struct's `beta_channel` field to the const.
- `JodParams` docstring described `JOD = jod_a − jod_b · D^jod_c`,
  a 3-coefficient form the production code doesn't implement.
  `kernels::pool::met2jod` is a 2-coefficient piecewise function
  (`JOD_A`, `JOD_EXP`) with a linear extension below `Q = 0.1`
  joined continuously at the knee. Replaced the made-up
  3-coefficient formula with the actual piecewise definition,
  added the `JOD_A` (`≈ 0.0440`) and `JOD_EXP` (`≈ 0.9302`)
  numeric anchors, and noted that the struct's `jod_b` is unused
  (the formula has no separate `b` coefficient). Tick 288.
- `MaskingParams` struct-level docstring listed `MASK_P / MASK_Q
  / MASK_C / XCM_3X3` but omitted `D_MAX` (the clamp ceiling,
  separate from `MASK_C`'s phase-blur post-scale). Per-field
  docs claimed cvvdp `q` and `epsilon`/`k` semantics that don't
  match production: `MASK_Q` is `[f32; 3]` per-channel (the
  struct's scalar `q` is shape-mismatched) and there is no
  `MASK_K` / saturation-epsilon constant in `kernels::masking`
  (closest are `MASK_C` and `D_MAX`, both log10-encoded and
  semantically different). Updated to document the shape
  mismatch explicitly, flag `k` as reserved-no-current-mapping
  scaffolding, and note a future JSON-loader path would need to
  widen `q` to `[f32; 3]` and split `k`. Also expanded
  `CvvdpParams::PLACEHOLDER`'s "inlined consts" list to the full
  set (`IMAGE_INT`, `PER_CH_W`, `BASEBAND_W` in `kernels::pool`;
  `D_MAX`, `CH_GAIN`, `PU_BLUR_KERNEL_1D`, `PU_PADSIZE` in
  `kernels::masking`) so the docstring matches what
  `kernels::pool` and `kernels::masking` actually export.
  Tick 289.

#### cvvdp-gpu (doctests)

- **Doctest cpu-only feature combo also fixed** — tick 283's
  cuda+wgpu cascade left cpu-only `--features cpu` builds broken
  (3 GPU doctests fail compile since neither cuda nor wgpu is on).
  Added a third `# #[cfg(all(feature = "cpu", not(any(feature =
  "cuda", feature = "wgpu"))))] # type Backend = cubecl::cpu::CpuRuntime;`
  fallback so the cuda doctests now compile under cpu-only too.
  No-op for the rendered docs (the canonical cuda branch still
  renders). All 6 doctests now pass under: cuda-only, wgpu-only,
  cpu-only, and default (cuda+wgpu+cpu).
- **CI doctest pass under `--no-default-features --features wgpu`
  was broken** for the 5 GPU/cpu doctests added between ticks 225
  and 244. They hardcoded `cubecl::cuda::CudaRuntime` (3 doctests)
  or `cubecl::cpu::CpuRuntime` (2 doctests), which don't exist
  under the CI doctest invocation
  (`cargo test --workspace --no-default-features --features wgpu
  --doc --release` per `.github/workflows/ci.yml:173`).
  Each doctest now wraps its body in feature-gated cfg attrs:
  - CUDA doctests: `# #[cfg(feature = "cuda")] type Backend = ...;`
    + `# #[cfg(all(feature = "wgpu", not(feature = "cuda")))] # type
    Backend = cubecl::wgpu::WgpuRuntime;` (wgpu fallback). Rendered
    docs still show the canonical cuda form; the wgpu fallback is
    hidden via `# ` prefix but compiles when cuda isn't on.
  - CPU doctests: wrap the entire body in `# #[cfg(feature = "cpu")]
    { ... # }` so non-cpu builds skip the body. Rendered docs are
    unchanged.
  No regression on default-features builds (all 6 doctests still
  green); CI's wgpu-only doctest pass now compiles all 6.
  The CI was masked from this regression because the bug landed
  on `feat/cvvdp-gpu-scaffold` and CI triggers only on master/PR.

#### cvvdp-gpu (tests)

- `error_display_messages_are_actionable` — pins the user-facing
  `Display` strings for all 4 `cvvdp_gpu::Error` variants. Tests
  content (variant name hint, the actionable next step) rather
  than exact strings, so future context additions still pass.
  Pre-tick-282 a rename of the `Display` impl would have silently
  degraded the user experience for callers who `?`-bubble cvvdp
  errors through `anyhow::Error::to_string()` / `panic!`
  propagation.

#### cvvdp-gpu (tests + examples)

- Collapse the last two-line `synth_pair_odd_dim_ref + apply_offset_dist`
  pairs onto `common::synth_pair_odd_dim_with_offset_dist`:
  `tests/cpu_backend.rs::synth_pair` (was 3 lines) and
  `examples/manifest_parity_probe.rs::synth_odd_pair` (was 3 lines).
  Each collapses to a single tuple-returning call. Drops the
  now-unused `synth_pair_odd_dim_ref` import from
  `manifest_parity_probe.rs` since `synth_odd_pair` was its only
  consumer.

#### cvvdp-gpu (tests)

- New `common::synth_pair_odd_dim_with_offset_dist(w, h) -> (ref, dist)`
  pairs `synth_pair_odd_dim_ref` with `apply_offset_dist` for the
  73×91 pycvvdp golden's construction
  (`bench_12mp_cuda.py::synth_pair_odd_dim`). Replaces 7-of-8
  inline `synth_pair_odd_dim_ref + apply_offset_dist` pairs in
  `tests/pipeline_color.rs` with a single-line tuple destructure.
  Also migrated the two `synth_pair_ref + apply_offset_dist`
  pairs in `pipeline_color.rs` (12mp tests) onto the existing
  `synth_pair_with_offset_dist`. The cpu_backend `synth_pair`
  wrapper and the warm-ref idempotence test (dist_a + dist_b)
  intentionally keep the two-line form for clarity.

#### cvvdp-gpu (tests + examples)

- New `common::apply_offset_dist(ref_bytes: &[u8]) -> Vec<u8>`
  standalone helper for the canonical `(-8, -4, +12)` saturating
  offset distortion. Tick 278's `synth_pair_with_offset_dist`
  paired this with the regular `synth_pair_ref`; tick 279
  extracts the dist half so callers can pair it with either ref
  variant (regular or `synth_pair_odd_dim_ref`). Migrated 12 more
  inline copies:
  - 10 sites in `tests/pipeline_color.rs` (all of which use
    `synth_pair_odd_dim_ref` + the offset dist for stage-probe
    tests at 32×32 odd dims)
  - 1 in `tests/cpu_backend.rs::synth_pair`
  - 1 in `examples/manifest_parity_probe.rs::synth_odd_pair`
  `synth_pair_with_offset_dist` itself now delegates to
  `apply_offset_dist`. Total dedup across ticks 278-279: 16 sites
  consolidated.

#### cvvdp-gpu (tests + examples + benches)

- New `common::synth_pair_with_offset_dist(w, h) -> (ref, dist)`
  helper bundles the canonical `synth_pair_ref` + `(-8, -4, +12)`
  saturating offset dist that 16 sites across the crate were
  building inline:
  - `benches/score.rs::synth_pair`
  - `examples/time_12mp.rs::synth_pair`
  - `examples/time_size_sweep.rs::synth_pair`
  - `examples/manifest_parity_probe.rs::synth_pair_12mp`
  All four now collapse to a single `synth_pair_with_offset_dist`
  call; the per-site `synth_pair` wrappers stay since they pass
  `u32 → usize`. `tests/pipeline_color.rs` 12mp tests already
  use the equivalent inline pattern (untouched — each test
  decides whether to pre-cache the ref or inline). The cpu_backend
  synth_pair keeps its odd_dim ref version with a clarifying
  comment.

#### cvvdp-gpu (examples)

- `examples/time_12mp.rs` + `examples/time_size_sweep.rs` now
  consume `tests/common` via `#[path]` (matches ticks 275-276's
  pattern). Drops 4 duplicates total — 2× Backend cascade
  (~6 lines each) + 2× hand-inlined `synth_pair` (~20 lines each).
  Both examples kept their tiny wrapper that combines `synth_pair_ref`
  with the same saturating-sub/saturating-add dist construction —
  identical pattern to time_12mp.rs's bench in benches/score.rs.
  No behaviour change.

#### cvvdp-gpu (examples)

- `examples/manifest_parity_probe.rs` now consumes
  `tests/common/mod.rs` via
  `#[path = "../tests/common/mod.rs"] mod common;` (same shape as
  tick 275's bench dedup). Drops the example's local
  `synth_pair_ref`, `synth_pair_odd_dim_ref` (via `synth_odd_pair`),
  and `pycvvdp_synth_golden_jod` clones in favour of the common
  helpers. Closes ticks 266 (last hand-mirrored goldens in
  examples) by leveraging the bench-side discovery (tick 275) that
  examples + benches can both reach `tests/common` via `#[path]`.
  Probe still passes all 6 fixtures at ≤ 0.005 JOD; max
  measured |d_gpu| = 0.000172 (synth_256x256_blur3x1).
- Drop the now-redundant
  `#![allow(clippy::excessive_precision)]` since the goldens are
  no longer inline float literals.

#### cvvdp-gpu (benches)

- `benches/score.rs` now consumes `tests/common/mod.rs` via
  `#[path = "../tests/common/mod.rs"] mod common;`. Drops the
  bench's local `Backend` cascade + `load_rgb_bytes` + `synth_pair`
  in favour of `common::Backend`, `common::load_rgb_bytes`, and
  `common::synth_pair_ref` (with the bench's per-fixture dist
  builder inlined). Closes the last synth-pattern duplication
  outside the example file. The bench's `load_rgb_bytes(path)`
  wrapper preserves the 256×256-assert contract by passing the
  bench's `W_256` / `H_256` constants through.

#### cvvdp-gpu (docs)

- `lib.rs` Status section now cross-references the warm-state
  invalidation regression tests
  (`warm_state_invalidates_after_each_documented_dispatcher`,
  `set_reference_does_not_invalidate_warm_state`,
  `gauss_chain_helpers_do_not_invalidate_warm_state`) and points
  at `docs/PORT_STATUS.md`'s "Resolved ticks 236-249" audit-history
  entry. Surfaces the contract work in the crate-root docs that
  docs.rs renders first.

#### cvvdp-gpu (docs)

- `MaskingParams`, `PoolingParams`, `JodParams` docstrings now
  state they're unused scaffolding and cross-reference
  `CvvdpParams::PLACEHOLDER`. Previously only `CsfParams` had this
  note; the other three sub-bundles left it implicit. Same shape
  as tick 264's `Cvvdp::new` silent-ignored-fields docs — protects
  users who'd otherwise expect varying `p` / `beta_spatial` /
  `jod_a` to change the metric output.

#### cvvdp-gpu (tests)

- Migrated the last 2 inline `Backend` cascade copies onto
  `common::Backend` (tick 270 covered the 6 file-root cases):
  - `pool_scalar.rs::mod gpu` → `use super::common::Backend;`
    (paired with a new file-root `#[path = "common/mod.rs"] mod common;`
    gated on the same `any(cuda, wgpu, hip)` as the gpu submodule)
  - `shadow_jod.rs::shadow_jod_gpu_runs_and_is_close_to_manifest_on_corpus`
    → `use common::Backend;` at the function top (already had
    `mod common` at file root since tick 253). Drops the 4-line
    cascade inside the fn body.
  Backend cascade dedup now complete — 0 inline copies remain
  anywhere in `tests/`.
- New `common::Backend` type alias dedups the "first available GPU
  backend" cascade (`cuda` → `wgpu` → `hip`) that was hand-mirrored
  across 6 test files at file root: `color_kernel.rs`, `csf_kernel.rs`,
  `masking_kernel.rs`, `pyramid_kernel.rs`, `pipeline_color.rs`,
  `pipeline_score.rs`. Each now uses `use common::Backend;` after a
  `#[path = "common/mod.rs"] mod common;` at the file top. The
  alias is cfg-gated on the same `any(cuda, wgpu, hip)` so cpu-only
  builds (and the cpu_backend test's `CpuRuntime` alias) are
  unaffected. The inline-in-fn / inline-in-mod copies in
  `shadow_jod.rs` and `pool_scalar.rs` stay local for now (different
  scope; needs a `use super::common::Backend;` migration that's a
  separate chunk). 6 files × 6 lines of cascade = 36 lines deleted;
  all 63 tests across 6 binaries still green.

#### cvvdp-gpu (cleanup)

- `cargo fmt -p cvvdp-gpu` run across the crate. Multiple test
  files + examples had drift after the recent dedup refactors
  (mostly Cvvdp::<Backend>::new(...) and predict_jod_still_3ch()
  call sites that fit on one line post-helper-extraction).
  Alphabetised the masking_kernel.rs `use` import list while
  there. No behavioural changes — 6 masking_kernel + 31
  pipeline_color + 14 pipeline_score + 2 shadow_jod tests still
  green.

#### cvvdp-gpu (tests)

- `common::load_rgb_bytes` signature widened from `&PathBuf` to
  `&Path`. `&PathBuf` callers still work via auto-deref; `&Path`
  callers (e.g. `path.parent().unwrap()` returning `&Path`)
  newly work without an extra `PathBuf::from(...)`. Standard Rust
  API hygiene per the
  [pathbuf-vs-path nursery clippy](https://rust-lang.github.io/rust-clippy/master/index.html#ptr_arg).
- `tests/common::load_rgb_bytes` extracted. The 10-line PNG/JPEG
  decode + dimension-assert helper was hand-mirrored across
  `tests/pipeline_score.rs` and `tests/shadow_jod.rs`. Both call
  sites now use the common helper; the `image::ImageReader` +
  `std::path::PathBuf` imports drop out of both files.

#### cvvdp-gpu (examples)

- `examples/manifest_parity_probe.rs` no longer hand-mirrors the
  6 pycvvdp golden JODs as `golden: 9.xxx` fields in its fixture
  table. Loads from `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`
  at runtime via a local `pycvvdp_synth_golden_jod(name)` helper
  that mirrors `tests/common/mod.rs`'s identical function (examples
  can't easily import test modules, so the lookup logic is inlined).
  Closes the last hand-mirrored golden in the repo; a future
  `build_goldens.py` rerun propagates to the example with zero
  hand-editing. Probe verified end-to-end: all 6 fixtures pass at
  ≤ 0.005 JOD.

#### cvvdp-gpu (tests)

- `tests/cpu_backend.rs::compute_dkl_jod_host_pool_matches_pycvvdp_at_73x91_odd_on_cpu_backend`
  no longer hardcodes the `9.390370` pycvvdp golden as a `const`.
  Loads from `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`
  via the `common::pycvvdp_synth_golden_jod("synth_73x91_odd")`
  helper that the pipeline_color sibling tests already use. Last
  hardcoded synth golden in tests/; a build_goldens.py rerun now
  propagates without any hand-edited mirrors anywhere in tests/.
  The 6 hand-mirrored copies in examples/manifest_parity_probe.rs
  stay (examples can't easily import test modules).

#### cvvdp-gpu (docs)

- `Cvvdp::new` / `Cvvdp::new_with_geometry` docstrings now spell
  out that **only `params.display` is consumed** — the
  `csf`/`masking`/`pooling`/`jod` sub-bundles of `CvvdpParams` are
  silently ignored because the per-stage cvvdp v0.5.4 numbers are
  inlined as `const`s in the kernels module. `CvvdpParams::PLACEHOLDER`
  already documented this on the struct side; the constructor docs
  now point at it. Same shape as tick 243's silent-ignored-`ppd`
  docs surfacing — protects users who'd otherwise pass a custom
  `CvvdpParams` expecting the masking/pooling exponents to matter.

#### cvvdp-gpu (benches)

- `benches/score.rs` adds `gpu_compute_dkl_jod_with_warm_ref` to
  both bench groups (`bench_resolution` synth + `bench_at_quality`
  corpus). Captures the warm-ref batch-scoring fast path
  empirically — the lib.rs Status section quotes ~1.8× per-DIST
  throughput at 12 MP vs cold, but until now there was no
  bench that produced numbers for that path. Provides a
  regression-net for the warm-state work in ticks 236-240
  (warm-state invalidation + persistent dest Vecs + scratch
  buffers).

#### cvvdp-gpu (docs)

- Refreshed stale "0.40 JOD GPU-vs-host drift" / "where the
  q=1 JOD drift lives" docstrings on three corpus-scale parity
  tests in `tests/pipeline_score.rs`:
  - `compute_dkl_weber_pyramid_matches_host_on_corpus_256x256`
  - `compute_dkl_t_p_bands_matches_host_on_corpus_256x256`
  - `compute_dkl_d_bands_matches_host_on_corpus_256x256`
  These were stage-isolation probes during the tick 175 pyramid
  fix; ticks 175/204/206 closed every drift to f32 noise. The
  tests still serve a useful role (per-stage bit-stability pins
  at corpus scale) but the docstrings claimed an in-flight
  investigation that's been done for ~80 ticks.

#### cvvdp-gpu (tests + docs)

- `score_with_reference_matches_score` now iterates the full
  `common::v1_corpus_qs()` set (6 q-levels: 1, 5, 20, 45, 70, 90)
  instead of the hand-picked `&[1u32, 20, 90]` subset (3 levels).
  Doubles parity coverage on the cached-reference contract at
  the cost of ~6 extra corpus loads. Also updated the leading
  comment which still claimed the path was "currently a host-
  scalar pass-through" — that switched to GPU in tick 213.

#### cvvdp-gpu (tests)

- `tests/cpu_backend.rs::synth_pair` now uses
  `common::synth_pair_odd_dim_ref` instead of its own inline copy.
  Last unmigrated odd-dim synth site outside the example file;
  `pipeline_color.rs` + `cpu_backend.rs` both now go through the
  common helper. All 4 cpu_backend tests still green (incl. the
  73×91 pycvvdp parity test at 0.000001 JOD diff).
- New `common::synth_pair_odd_dim_ref(w, h)` helper for the
  alternate odd-dim synth pattern (`(x * 8) % 256` / `(y * 8) % 256`
  / `((x + y) * 4) % 256`). Migrated all 10 hand-inlined sites in
  `tests/pipeline_color.rs` onto it. Companion to tick 255-258's
  `synth_pair_ref` dedup. Bit-stable parity preserved on all 31
  pipeline_color tests (including 73×91 odd-dim cold + warm).
- Final 6 hand-inlined synth_pair_ref sites in
  `tests/pipeline_color.rs` migrated onto `common::synth_pair_ref`
  (stage-probe helpers for chroma_shift: `compute_dkl_planes`,
  `compute_dkl_t_p_bands`, `compute_dkl_weber_pyramid`,
  `spatial_pool`, `compute_dkl_d_bands`, plus an `_at_chroma_shift_sentinels`
  helper). 14 of 14 callers now consolidated; the inline modular-
  arithmetic ref construction no longer appears anywhere except the
  helper definition itself. Bit-stable parity preserved across all
  31 pipeline_color tests.
- Migrated the `blur3x1`, `blur1x3`, and `noise` 256×256 parity
  tests off the hand-inlined `synth_pair_ref` construction onto
  `common::synth_pair_ref`. 7 of 14 inlined sites in
  `tests/pipeline_color.rs` now consolidated. Bit-stable parity
  preserved on all three (still ≤ 0.005 JOD vs pycvvdp goldens).
- Migrated the two `compute_dkl_jod_*_pycvvdp_at_256x256_chroma_shift`
  tests (cold + warm-ref) off the inlined synth-pair construction
  onto `common::synth_pair_ref`. Same shape as tick 255's 12mp
  migration. 4 of 14 inlined sites in `tests/pipeline_color.rs`
  now use the helper. Bit-stable parity preserved (chroma_shift
  diff vs pycvvdp golden remains 0.0000).
- New `common::synth_pair_ref(w, h) -> Vec<u8>` helper builds the
  canonical synthetic-fixture reference image (the
  `(x * 17 + y * 5) % 251`-style modular pattern matching pycvvdp's
  `synth_pair_ref` in `bench_12mp_cuda.py`). Migrated the two
  largest fixture-using tests (`compute_dkl_jod_matches_pycvvdp_at_12mp_synth`
  and `compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_12mp_synth`)
  off their hand-inlined copies; bit-stable output (JOD 9.4580
  matches pycvvdp golden to 0.0000 on both). The pattern was
  duplicated across 12 more sites in `pipeline_color.rs`; future
  ticks can migrate them opportunistically.

#### cvvdp-gpu (tests)

- New `common::v1_corpus_qs()` helper derives the q-list from the
  canonical `scripts/cvvdp_goldens/v1_corpus_jods.json` itself.
  Replaces the hand-mirrored `&[1, 5, 20, 45, 70, 90]` constant
  duplicated across 5 callers (3 in `pipeline_score.rs` + 2 in
  `shadow_jod.rs`). A goldens regen that adds (e.g.) q=2 to the
  manifest now propagates to every parity test without hand-editing.
  Same shape as tick 253's `v1_corpus_jod_golden(q)` dedup.
- `tests/shadow_jod.rs` no longer hardcodes the pycvvdp manifest
  JOD constants. Both tests now load via `common::v1_corpus_jod_golden(q)`
  from the canonical `scripts/cvvdp_goldens/v1_corpus_jods.json`
  (which the existing `tests/pipeline_score.rs::cvvdp_score_matches_v1_manifest`
  already used). Previously the same six `(q, expected_jod)` pairs
  were duplicated across three test files; a `build_goldens.py`
  rerun + JSON bump would have silently skipped the two shadow_jod
  copies until manual sync. Tick 253 dedup.

#### cvvdp-gpu (docs)

- `benches/score.rs` stale-comment cleanup:
  - `bench_score_q1` no longer claims the GPU drifts 0.4 JOD at q=1;
    that drift was closed to 0.0000 in ticks 204/206 (chroma_shift
    CSF + gausspyr_reduce parity-bug fixes). Comment now correctly
    states the historical drift and points at the regression-pin test.
  - `bench_at_quality`'s host_scalar group comment no longer says
    `Cvvdp::score` routes through it; tick 213 switched `score` to
    GPU `compute_dkl_jod`. Comment now correctly states the host
    path is a faster-to-debug reference exposed via
    `host_scalar::predict_jod_still_3ch`.

#### cvvdp-gpu (tests)

- `gauss_chain_helpers_do_not_invalidate_warm_state` — pins the
  inverse of `warm_state_invalidates_after_each_documented_dispatcher`:
  `compute_dkl_planes` and `compute_dkl_gauss_pyramid` write only
  to `gauss_ref` (per-call scratch, not warm state), so they MUST
  preserve the cached scalar. A future refactor that made either
  helper additionally emit bands into `bands_ref` (matching the
  symmetric `compute_dkl_weber_pyramid` interface) would need to
  invalidate warm state — this test would surface that. Sibling
  to `set_reference_does_not_invalidate_warm_state` (tick 238).

#### cvvdp-gpu (tests + docs)

- `set_reference_replaces_prior_cache` — pins the implicit
  cache-replace semantics of `Cvvdp::set_reference`. Test calls
  `set_reference(ref_a)`, then `set_reference(ref_b)`, then
  `score_with_reference(dist)`; expects the result to match
  `score(ref_b, dist)`, not `score(ref_a, dist)`. The contract
  was the natural cache-shape callers expect but had been
  documented-by-convention only — a refactor that no-op'd the
  second `set_reference` call wouldn't have surfaced in CI.
  `set_reference`'s docstring now explicitly states the replace
  semantics and cross-references this test + the tick-238
  non-invalidation test.

#### cvvdp-gpu

- **`compute_dkl_jod_with_warm_ref` / `compute_dkl_jod_host_pool_with_warm_ref`
  now check dim mismatch before `NoWarmReference`.** When a caller
  has BOTH a wrong-size dist buffer AND no warm state, the wrong-size
  buffer is the more actionable error — they need to fix the buffer
  regardless of whether warm state is set. Pre-tick-248 ordering
  reported `NoWarmReference` first, masking the dim mismatch. New
  test `compute_dkl_jod_with_warm_ref_reports_dim_mismatch_before_no_warm`
  pins the order so a future regression surfaces in CI.
- **Debug-assert `compute_dkl_csf_weighted_bands` weight-band-count
  matches construction-time `n_levels`.** The per-level
  `weight_band_kernel` loop reads `weight_idx = k * N_CHANNELS + c`
  into the host-flattened weights buffer for `k = 0..n_levels`. If
  the caller's `ppd` produces fewer band frequencies than the
  construction-time `n_levels`, the higher-k kernel launches read
  past `flat_weights.len()` and OOB the GPU buffer. Now
  `debug_assert_eq!(weights_per_level.len(), n_levels)` catches
  the precondition violation in debug builds with a message that
  spells out the fix: reconstruct against the new geometry. Release
  preserves silent OOB behavior since this is a documented
  precondition (per tick 246's docstring update). Tick 247 pairs
  the docstring warning with an enforceable check.
- **Revert misplaced tick-243 debug_assert + tick-245 docstring
  on `compute_dkl_csf_weighted_bands`.** Unlike the JOD-path
  helpers, this function genuinely consumes the caller-passed `ppd`
  (via `precomputed_band_weights(ppd, w, h, l_bkg)` which uses
  `band_frequencies(ppd, ...)` to compute per-band rho). The
  tick-243 audit assumed every public `ppd` parameter was a
  silent-ignored relic — true for the 6 JOD-path helpers but
  wrong for this Laplacian + per-band-weight helper.
  - Removed the `debug_assert_ppd_matches_geometry` call at entry
  - Docstring now correctly states `ppd` is consumed and warns
    that the caller must keep `band_frequencies(ppd, w, h).len()`
    consistent with the construction-time `n_levels` (otherwise
    the weights buffer mismatches the per-level kernel launches)
  No tests changed behaviour — all 30 pipeline_color + 12
  pipeline_score tests still green; existing call sites pass
  matching ppd so the spurious assert never fired in CI.

#### cvvdp-gpu (docs)

- Document the silent-ignored `ppd` argument on the 6 public
  methods that take it (`compute_dkl_jod`, `compute_dkl_d_bands`,
  `compute_dkl_t_p_bands`, `compute_dkl_jod_host_pool`,
  `compute_dkl_jod_host_pool_with_warm_ref`,
  `compute_dkl_jod_with_warm_ref`, plus
  `compute_dkl_csf_weighted_bands`). Each docstring now states that
  `ppd` is silently ignored — the GPU CSF LUT is pre-uploaded
  against the construction-time geometry — and points readers at
  `Cvvdp::new_with_geometry` for a different display geometry. Pairs
  with the tick-243 `debug_assert_ppd_matches_geometry` safety net
  by making the contract explicit in the docs that users read first.

#### cvvdp-gpu (tests)

- `debug_assert_fires_when_ppd_mismatches_geometry` — pins the
  tick-243 ppd-mismatch debug_assert. Builds Cvvdp with the
  default STANDARD_4K geometry (75.4 PPD), then calls
  `compute_dkl_jod` with the phone-shaped 110-PPD value; expects
  panic via `#[should_panic(expected = "ppd=")]`. Gated on
  `#[cfg(debug_assertions)]` so release builds skip the test
  (the assert compiles out there). A future refactor that drops
  the safety net would silently regress without this pin.

#### cvvdp-gpu (debug)

- **Surface silent-ignored `ppd` mismatches in debug builds.**
  6 public methods take a `ppd: f32` parameter that the
  implementation **silently ignores** — `logs_row` is pre-uploaded
  at construction time against `self.geometry.pixels_per_degree()`,
  so a caller who built `Cvvdp::new(client, w, h, p)` with the
  default `STANDARD_4K` (75.4 PPD) then called
  `compute_dkl_jod(ref, dist, phone_ppd)` (110 PPD) would get
  results scored against 75.4 PPD with no warning. Pre-tick-243
  there was no surfaced sanity check.
  - New `Cvvdp::debug_assert_ppd_matches_geometry(ppd)` helper:
    `debug_assert!((ppd - self.geometry.pixels_per_degree()).abs() < 1e-3)`
  - Wired into the 6 affected entries: `compute_dkl_jod`,
    `compute_dkl_d_bands`, `compute_dkl_t_p_bands`,
    `compute_dkl_jod_host_pool`,
    `compute_dkl_jod_host_pool_with_warm_ref`,
    `compute_dkl_jod_with_warm_ref`, and
    `compute_dkl_csf_weighted_bands`.
  - Release builds preserve silent-ignore (no public-API change);
    the parameter remains in the signatures for source compatibility.
    All 30 pipeline_color + 11 pipeline_score + 4 cpu_backend tests
    green — all existing call sites pass ppd consistent with geometry.

#### cvvdp-gpu (docs)

- Stale pre-tick-175 warm-ref throughput numbers updated across
  `warm_reference`, `set_reference`, the `CachedReference`
  struct doc, the `compute_dkl_jod_with_warm_ref` doctest, and
  `lib.rs`. All sites referenced `1.75× / 36.1 → 20.6 ns/px /
  42.9% saved` from tick 170; the tick-175 ceil-div correctness
  fix raised absolutes to ~62 / ~34 ns/px while keeping a similar
  ratio (~1.8×). Docstrings now cite `~1.8×` and defer to
  `lib.rs` "How we compare to the canonical reference" for the
  source-of-truth measurements. The "Resolved tick 170" entry in
  `PORT_STATUS.md` keeps the original numbers (accurate as-of-tick-170)
  plus a tick-175 update note explaining why the post-fix path is
  numerically slower (correct output vs broken pyramid).

#### cvvdp-gpu (tests)

- `invalid_image_size_surfaces_on_too_small_dims` — pins the
  `Error::InvalidImageSize` construction-time guard on `Cvvdp::new`
  and `Cvvdp::new_with_geometry`. Tests 6 sub-threshold cases
  (7×8, 8×7, 7×7, 4×4, 0×0, plus a `new_with_geometry` case)
  plus the 8×8 boundary success path. Pre-tick-241 a refactor
  that swapped the `width < PYRAMID_MIN_DIM * 2` check for
  `width < PYRAMID_MIN_DIM` (accepting 4×4 with no usable
  pyramid) would not have surfaced in CI.
- `dimension_mismatch_surfaces_on_wrong_size_inputs` — pins the
  `Error::DimensionMismatch` contract on every public entry that
  validates buffer length: `Cvvdp::score` (both arms),
  `set_reference`, `score_with_reference`, `warm_reference`, and
  `compute_dkl_jod_with_warm_ref`. Each is called with a buffer
  sized for `(w/2) × (h/2)` against a Cvvdp configured for `w × h`;
  the test asserts both that `DimensionMismatch` fires AND that
  the `(expected, got)` fields carry the right byte counts.
  Closes a real zero-coverage gap: a refactor that swapped `!=`
  for `<` (silently accepting smaller buffers and reading past
  `srgb.len()`) would not have surfaced in CI before this.
- `compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_12mp_synth` —
  large-image warm-ref pycvvdp parity. Completes the warm-ref vs
  pycvvdp coverage grid: small same-parity (chroma_shift, tick 222)
  + small mixed-parity (73×91, tick 226) + large same-parity here
  (12 MP, full ~9-band pyramid). Exercises warm-state restoration
  across every weber_scratch level. Measured diff: 0.0000 JOD.
  Runtime ~6s on RTX-class CUDA — within parity-test budget,
  matches the existing cold-path 12mp test cadence.

#### cvvdp-gpu (performance)

- **Persistent `log_l_bkg_{ref,dis}_dests: Vec<Handle>` slots** —
  `Cvvdp::new_with_geometry` now pre-builds the destination-handle
  Vecs that `_dispatch_ref_weber_pyramid_only` and
  `_dispatch_dist_weber_pyramid_only` previously rebuilt per call
  via `weber_scratch.iter().map(|s| s.log_l_bkg.clone()).collect()`.
  Each dispatch now `mem::take`s the pre-built Vec, passes it as
  `&[Handle]` to `_dispatch_weber_pyramid_gpu`, then moves it back —
  zero heap allocation per JOD-side, replacing `(1 Vec alloc) +
  (n_levels - 1) handle ref-bumps` per call. Cold JOD pays this
  twice (REF + DIST), warm-ref JOD pays it once (DIST only). All
  30 pipeline_color + 10 pipeline_score + 4 cpu_backend tests
  green; manifest parity untouched.
- **Persistent `src_u32_scratch` host buffer** — `Cvvdp::new` pre-
  allocates a `Vec<u32>` of length `width * height * 3` once at
  construction time; `_dispatch_dkl_planes_gpu` now fills it in
  place via `iter_mut().zip(srgb.iter())` instead of allocating
  a fresh `Vec<u32>` per call via `.iter().map(|b| b as u32).collect()`.
  Removes ~`width × height × 12` bytes of host allocator round-trip
  per JOD-side dispatch — at 12 MP that's ~144 MB per side, paid
  twice per cold JOD and once per warm-ref DIST. The GPU buffer
  upload (`create_from_slice` of the scratch's bytes) still
  happens per call since cubecl 0.10 has no public "write into
  existing handle" API. All 27 pipeline_color + 9 pipeline_score
  + 4 cpu_backend tests green; manifest parity untouched.

#### cvvdp-gpu (docs)

- `Cvvdp::compute_dkl_jod_with_warm_ref` now has a `no_run` doctest
  example showing the canonical GPU batch-scoring pattern
  (warm REF once, score N DIST candidates against it). Mirrors
  the existing cpu-runtime example on
  [`Cvvdp::compute_dkl_jod_host_pool_with_warm_ref`] — completes
  the doctest coverage on the warm-ref API across both GPU and
  cpu-runtime paths.

#### cvvdp-gpu (cleanup)

- Cleared remaining 7 clippy warnings under `--all-targets`:
  - `tests/common/mod.rs`: collapsed nested `if let Ok(hex) = ...
    { if hex == sha256 { return ... } }` into a let-chain
    (`if let ... && hex == sha256`).
  - `tests/pipeline_color.rs`: dropped a redundant `(wu * hu) as usize`
    cast (wu/hu were already `usize`); added module-level
    `#![allow(clippy::needless_range_loop)]` for the 3 per-band
    `for k in 0..n_bands` loops (k indexes ref_tp[k] / d_bands[k]
    plus side metadata — enumerate is a wash). Mirrors the library's
    same allow.
  - `tests/cpu_backend.rs` + `examples/manifest_parity_probe.rs`:
    `#![allow(clippy::excessive_precision)]` for the pycvvdp
    golden literals — same rationale as the library-level allow:
    the 7-digit decimal documents the source value verbatim even
    though LLVM rounds at f32.
  Net: `cargo clippy -p cvvdp-gpu --features cuda --all-targets`
  is warning-clean. All 27 pipeline_color + 9 pipeline_score + 4
  cpu_backend tests still green.
- Fixed 8 clippy lints surfaced under MSRV 1.93:
  - 6× `manual_div_ceil` in `pipeline.rs` (`(x + 1) / 2` →
    `x.div_ceil(2)` in pyramid-level allocators)
  - 2× `manual_is_multiple_of` in `kernels/pyramid.rs` (`sh % 2 == 0`
    → `sh.is_multiple_of(2)` in `gausspyr_reduce_scalar`)
  Semantically equivalent rewrites — all 78 cuda + 4 cpu tests
  green; manifest parity untouched. `cargo clippy -p cvvdp-gpu`
  is warning-clean.

#### cvvdp-gpu (docs)

- `Cvvdp::score_with_reference` now has a `no_run` doctest example
  showing the canonical `set_reference` + `score_with_reference`
  batch pattern (one stashed REF, many DIST). Pairs with the
  `Cvvdp::score` doctest from tick 225 to cover both top-level
  public scoring entry points. Also notes the
  `Error::NoCachedReference` precondition explicitly in the
  doc body.
- Renamed `examples/chroma_shift_drift_probe.rs` →
  `examples/manifest_parity_probe.rs`. The file started life
  (tick 191) as a single-fixture probe while investigating the
  chroma_shift drift, but tick 210 expanded it to walk all 6
  manifest fixtures — the old name no longer reflected what it
  did. Internal doc header + run-with command updated; a note at
  the top of `docs/CHROMA_DRIFT_INVESTIGATION.md` flags the rename
  so historical references in that file (which describe past
  measurements) stay accurate. Active "See ..." pointer also
  updated to the new name. Probe verified end-to-end: all 6
  fixtures pass at ≤ 0.005 JOD vs pycvvdp goldens, max
  measured |d_gpu| = 0.000186 JOD on synth_256x256_blur1x3.

#### cvvdp-gpu (performance)

- `compute_dkl_d_bands` host readback init no longer pre-allocates
  `vec![0.0; n_px] × 3` per pyramid level only to immediately
  overwrite each entry with `f32::from_bytes(&bytes).to_vec()`.
  Now uses empty `Vec::new()` slots — matches `compute_dkl_gauss_pyramid`'s
  readback shape and drops `~3 × n_levels × n_px` floats of wasted
  host zero-fill per call. (`compute_dkl_d_bands` is a parity-test
  helper; production JOD path is unaffected since it pools on-GPU.)
- **Persistent `partials_h` atomic-pool buffer** — `Cvvdp::new`
  now allocates a single `n_levels × N_CHANNELS` partials buffer
  (≤ 144 bytes at MAX_LEVELS=9) and `_pool_and_finalize_jod` zero-
  fills it via `fill_f32_kernel` per call instead of allocating
  a fresh GPU buffer + uploading host zeros every JOD call.
  Removes one `create_from_slice` host alloc + Host→GPU copy per
  call from the JOD hot path; pattern mirrors the tick-168
  `baseband_log_l_bkg` migration. All 27 pipeline_color + 9
  pipeline_score + 8 pool_scalar tests green on CUDA, including
  manifest parity (`compute_dkl_jod_on_v1_manifest_corpus` at ≤ 0.005
  JOD) and the GPU-pool-vs-host-pool sentinel
  (`compute_dkl_jod_host_pool_matches_compute_dkl_jod`).

#### cvvdp-gpu (tests)

- `compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_73x91_odd` —
  direct warm-ref pycvvdp parity on the mixed-parity 73×91 fixture.
  Pairs with the chroma_shift warm-ref test from tick 222: both pin
  the warm-state restoration path against canonical pycvvdp, but
  73×91 specifically exercises the tick-206 gausspyr_reduce
  parity-bug fix on REF (mixed-parity reduce levels 6×5 → 3×3 and
  46×37 → 23×19). Measured diff: 0.0000 JOD. Closes a transitivity
  gap: prior warm-ref pycvvdp coverage was same-parity only.

#### Workspace

- Pinned multi-tick task in `CLAUDE.md`: compute CVVDP scores for
  all zensim training data sets via vast.ai docker images, output
  as parquet sidecars with implementation-distinguished column
  names (e.g. `cvvdp_pycvvdp_v054`, `cvvdp_imazen_v0_0_1`). Survives
  context compaction; every `/loop` tick re-reads it.

#### zen-metrics-cli

- New `score-pairs` subcommand (feature-gated on `sweep`):
  consumes the pairs TSV that `sweep --pairs-tsv` produces and
  emits a parquet sidecar with the metric's versioned column name
  (e.g. `cvvdp_imazen_v0_0_1` for cvvdp). Schema matches
  `crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md` exactly:
  `image_path string`, `codec string`, `q int64`,
  `knob_tuple_json string`, `<metric> float64`. Zstd compression.
  Symmetric with `scripts/sweep/pycvvdp_worker.py score-pairs`.
  Initial n=4 sentinel: cvvdp-gpu vs pycvvdp parity within 0.03 JOD
  on q50/q90 zenjpeg-encoded 64×64 noise images.

#### zen-metrics-cli (sweep)

- `sweep` subcommand learns two new flags that pair off for
  external-scorer workflows (e.g. pycvvdp):
  - `--distorted-out-dir <DIR>`: every successfully-decoded cell
    writes its distorted image as a `Compression::Fastest` PNG
    into this directory. Filenames are deterministic and
    collision-resistant:
    `<src_stem>_<src_path_hash16>_<codec>_q<q>_<knob_hash16>.png`.
  - `--pairs-tsv <FILE>`: tab-separated companion to the main
    `--output` TSV with columns
    `image_path codec q knob_tuple_json ref_path dist_path` —
    one row per decoded cell. `dist_path` is empty when
    `--distorted-out-dir` is unset.
  - Smoke test: 2-image × 2-q sweep produced 4 PNGs + a 4-row pairs
    TSV that `pycvvdp_worker` then scored into a 4-row
    `cvvdp_pycvvdp_v054` parquet sidecar.

#### scripts/sweep

- `dual_impl_chunk.sh` — per-chunk dual-implementation runner.
  Drives one sweep + both cvvdp scorers (zen-metrics-cli score-pairs
  for cvvdp-gpu + pycvvdp_worker.py for canonical pycvvdp) and
  joins the two sidecars into a parity TSV. Local smoke test: 4
  cells joinable, mean |diff| 0.0245 JOD, max 0.0300 JOD on the
  synth zenjpeg q50/q90 corpus.
- `pycvvdp_worker.py` — canonical pycvvdp v0.5.4 scoring worker.
  Consumes a TSV of `(identity_tuple, ref_path, dist_path)` rows
  and writes a parquet sidecar with the `cvvdp_pycvvdp_v054`
  column per `crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md`.
  Verified end-to-end on a synth 64×64 pair: JOD 10.0 for identical
  inputs, 9.63 for chroma-shifted.
- `Dockerfile.pycvvdp` — image for the worker on vast.ai. Bases on
  `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` with pycvvdp
  0.5.4, pillow, numpy, pyarrow. CMD is help text; runners must
  pass an explicit `pycvvdp-worker score-pairs …` command.

#### cvvdp-gpu

- `CVVDP_COLUMN_NAME` const exposes a per-implementation column tag
  (default `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`, overridable via
  the `CVVDP_IMPL_TAG` build-time env var). Used by sweep tooling so
  multiple cvvdp variants land side-by-side in parquet sidecars
  without colliding.

#### zen-metrics-cli

- `MetricKind::Cvvdp::column_names()` now returns
  `cvvdp_gpu::CVVDP_COLUMN_NAME` when the `gpu-cvvdp` feature is
  enabled, so sweep TSV/parquet headers emit
  `score_cvvdp_imazen_v0_0_1` (or the override). The user-facing
  CLI flag `--metric cvvdp` stays stable.

#### cvvdp-gpu (new crate, v0.0.1)

- ColorVideoVDP (still-image) port matching pycvvdp v0.5.4 on the
  v1 R2 manifest within 0.006 JOD across q1–q90. Full pipeline:
  - Color: sRGB→DKLd65 host scalar + `srgb_to_dkl_kernel` (cuda
    parity ≤3e-5).
  - Pyramid: vanilla Laplacian + Weber-contrast variant
    (`weber_contrast_pyr_dec_scalar`) + 4 cubecl kernels
    (`downscale_kernel`, `upscale_v_kernel`, `upscale_h_kernel`,
    `subtract_kernel`, `weber_contrast_compute_kernel`).
  - CSF: 32×32×3 LUT bilinear interp host scalar +
    `csf_apply_per_pixel_kernel` (per-pixel L_bkg from achromatic
    Gaussian pyramid) + `weight_band_kernel`.
  - Masking: mult-mutual + xchannel + soft clamp.
    `mult_mutual_band` host scalar + 3 cubecl kernels
    (`min_abs_3ch_kernel`, `mult_mutual_3ch_no_blur_kernel`,
    `mult_mutual_3ch_with_blurred_kernel`), plus `pu_blur_h_kernel`
    + `pu_blur_v_kernel` for the σ=3 phase-uncertainty blur.
  - Pooling: 3-stage Minkowski + smooth `met2jod` piecewise JOD
    mapping. `pool_band_kernel` does per-pixel `safe_pow` +
    `Atomic<f32>::fetch_add` reduction.
  - Composed: `Cvvdp::score` and `host_scalar::predict_jod_still_3ch`
    are both v1-manifest-locked (≤0.006 JOD). `Cvvdp::new` defaults
    to `DisplayGeometry::STANDARD_4K`; `Cvvdp::new_with_geometry`
    accepts any cvvdp display geometry.
- Parity goldens at
  `s3://coefficient/cvvdp-goldens/v1/manifest.json`
  (public mirror: `https://coefficient.r2.imazen.org/...`).
- Test infrastructure: `parity-goldens` cargo feature gates the
  network-fetching integration test, keeping default `cargo test`
  offline. Per-stage parity tests (color, pyramid, csf, masking,
  pooling) all locked vs pycvvdp.
- **GPU-composed score path** — full pipeline up through D bands +
  masking runs on GPU; only the spatial pool + 3-stage Minkowski +
  `met2jod` are host. New `Cvvdp` helpers:
  - `compute_dkl_weber_pyramid` — color + Weber-contrast pyramid,
    returns `(bands, log_l_bkg)` per the `WeberPyramidGpu` type
    alias.
  - `compute_dkl_t_p_bands(ppd)` — Weber × per-pixel CSF S ×
    `CH_GAIN` × `band_mul`. `band_mul = 2.0` for non-edge levels,
    `1.0` at level 0 and baseband. Baseband sets `CH_GAIN_eff = 1.0`
    so callers can reproduce cvvdp's `|T_p - R_p|` baseband bypass.
  - `compute_dkl_d_bands(ref, dist, ppd)` — composes Weber + CSF +
    masking. Non-baseband bands use the GPU `mult_mutual_3ch_*`
    masker (with the `10^MASK_C` PU-blur scale applied via
    `weight_band_kernel`); baseband uses `|T_p_dis - T_p_ref|`.
    Uses the reference's `log_l_bkg` for both sides per cvvdp's
    `weber_g1` contract.
  - `compute_dkl_jod(ref, dist, ppd)` — full GPU score path
    returning a JOD scalar. Drift survey shows GPU matches host
    within 0.001 JOD for q ≥ 20; the 0.40 drift at q=1 is
    cumulative f32 noise compounding through `met2jod`'s steep
    slope region, not a parity bug.
- `Cvvdp::score_with_reference` is wired (previously returned a
  silent 0.0). Caches reference sRGB bytes and routes through
  `host_scalar::predict_jod_still_3ch` — exact-parity with
  `Cvvdp::score(ref, dist)`.
- Drift-survey tests document where GPU vs host diverges per
  stage: `compute_dkl_{weber_pyramid,t_p_bands,d_bands}_matches_host_on_corpus_256x256`
  + `compute_dkl_jod_vs_host_scalar_on_corpus` +
  `compute_dkl_jod_on_v1_manifest_corpus`.
- `zenbench` score-path benchmark (`benches/score.rs`) — first
  measured CPU vs GPU per-pixel numbers at 256×256 / 1 MP / 12 MP.
- `time_12mp` example (`examples/time_12mp.rs`) — fixed-iteration
  one-shot timer for compute_dkl_weber_pyramid / compute_dkl_d_bands
  / compute_dkl_jod at 12 MP. Per-phase breakdown surfaces where
  the GPU pipeline spends its time without the zenbench
  calibration loop's overhead at large image sizes.
- `CVVDP_TRACE=1` env-var-gated stderr instrumentation inside
  `compute_dkl_d_bands` — per-level CSF / masking / log_l_bkg
  upload timings. Zero-cost when unset.
- `CVVDP_TRACE_WEBER=1` env-var-gated stderr instrumentation
  inside `compute_dkl_weber_pyramid` splitting GPU dispatch from
  host readback.
- Direct kernel-level parity test for `csf_apply_3ch_kernel`
  in `tests/csf_kernel.rs` — sweeps the full log_l_bkg LUT axis
  with distinct per-channel ch_gain values (catches bugs the
  indirect d_bands test would miss).
- Consecutive-weber diagnostic block in `examples/time_12mp.rs`
  (`0a71bb22`) — calls `compute_dkl_weber_pyramid` twice on the
  same `ref_bytes` outside `compute_dkl_d_bands` to isolate
  whether the "weber(dist) is 2× weber(ref)" slowdown is
  position-dependent (consecutive-call overhead) or data-shape
  dependent. Result: standalone consecutive calls show no
  slowdown, ruling out cubecl warm-up / driver effects and
  pinning the cause to host memory pressure from holding the
  `ref_weber: Vec<Vec<f32>>` (~190 MB at 12 MP) alive across the
  second call inside the d_bands flow.
- `_dispatch_weber_pyramid_gpu` private helper (`072d9e43`)
  factored out of `compute_dkl_weber_pyramid` — takes a
  `&[Handle]` destination slice for the per-level `log_l_bkg`
  outputs. The bisect for tick 85's 5× regression revealed
  that this extraction itself does not regress, only the
  full 5-phase serial restructure did; the helper is kept so
  future experiments can swap the destination buffer set
  without re-wiring weber's body.

### Fixed

#### cvvdp-gpu

- **73×91 odd-dim residual closed (was 0.006 JOD).** Found a
  parity-check bug in pycvvdp's `gausspyr_reduce`: the
  horizontal-pass right-column patch uses `x.shape[-2]` (INPUT
  ROW parity) to pick its odd/even branch even though the
  comments say "columns" — `lpyr_dec.py:204-209`. For
  mixed-parity inputs (e.g. 6×5 → 3×3 at the 73×91 baseband)
  pycvvdp applies the wrong patch.
  - `host_scalar` `gausspyr_reduce_scalar`: rewritten to bug-
    compatible zero-pad + parity-aware patches.
  - GPU `downscale_kernel`: adds a delta correction at the right
    column when sw and sh parities differ.
  - New `compute_dkl_jod_matches_pycvvdp_at_73x91_odd` test
    passes at f32 precision (diff = 0.0000 vs pycvvdp golden).
  - All other corpus fixtures (256² + 4 MP, same-parity dims)
    unchanged — the bug-compat patches match pure reflection
    for all sw == sh parity inputs.

- **Chroma-shift drift closed (was 0.117 JOD).** pycvvdp overrides
  the baseband CSF rho to 0.1 cy/deg (`cvvdp_metric.py:628`),
  but our pipeline used the geometric value from
  `band_frequencies(ppd, w, h)` (0.190 at 256² standard_4k). Fixed
  by adding `kernels::csf::CSF_BASEBAND_RHO = 0.1` and applying it
  in both `host_scalar::predict_jod_still_3ch` and
  `Cvvdp::new`'s `logs_row` pre-upload. The
  `compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift` test
  re-enabled at standard 0.005 JOD tolerance; chroma_shift now
  matches pycvvdp golden 9.664865 to f32 precision.

### Changed (performance)

#### cvvdp-gpu

After tick 70's per-band-allocation diagnosis, four scratch
hoists + one kernel fuse landed in succession:

- **Pre-allocate per-band CSF + masking scratch** on `Cvvdp::new`.
  `compute_dkl_d_bands` was alloc_zeros_f32-ing 18 buffers per
  non-baseband level per call (~1.5 GB worth at 12 MP). Moved
  to a `DBandsScratch` struct on the Cvvdp instance. Result:
  12 MP d_bands −25%, full jod −30%.
- **Pre-allocate per-band Weber pyramid scratch** — same shape
  for the expand/subtract/weber chain (l_bkg_fine, vscratch_a,
  log_l_bkg, per-channel vscratch_c/upscaled_c/layer_c).
  Result: 12 MP weber alone 5× faster (105 → 21.6 ns/px), full
  jod 2.4× faster (310 → 127 ns/px). **This crossed the milestone
  of beating fcvvdp single-thread** (214 ns/px on their bench).
- **Drop unused per-side GPU buffers** (`src_dis`, `gauss_dis`,
  `bands_dis`, `pool_partials`) that were allocated by
  `Cvvdp::new_with_geometry` but never read by any GPU helper.
  Saves ~13 MB per Cvvdp at 256×256.
- **Hoist `logs_row` uploads** to `Cvvdp::new_with_geometry`
  (24 small uploads of 128 B were happening per d_bands call,
  one per `(level, channel)`). Since `rho_k` is fixed per Cvvdp,
  the LUT rows are stable across calls.
- **Fuse 3-channel CSF apply** into a single kernel
  (`csf_apply_3ch_kernel`) that shares the per-pixel LUT bracket
  math across A/RG/VY channels. Cut L0 CSF time at 12 MP from
  420 ms (6 launches) to 170 ms (2 launches) — but the saved
  ~250 ms got absorbed by ~340 ms of unaccounted overhead
  (likely host Vec<Vec<f32>> alloc for the weber readback);
  median d_bands wall is unchanged.
- **`pow(10, x) → exp(x · ln(10))`** in CSF kernels for the
  mathematical identity. No measurable win on cuda (likely cubecl
  already compiles to similar PTX); kept for potential wgpu/hip
  payoff.
- **Dist-side CSF reads `self.bands_ref` handles directly**
  (`8b6f2776`) — `compute_dkl_d_bands` no longer uploads
  `dist_weber[k]` from host inside the per-band CSF apply. The
  dist-side handles are already resident in `self.bands_ref`
  after the `weber(dist)` call earlier in the band loop, so the
  CSF kernel reads them in place. REF-side still uploads since
  `bands_ref` has been overwritten with DIST data by band-loop
  time. Result on 12 MP cuda: weber 291 ms (baseline),
  d_bands 1.42 s (−3% from 1.46 s), jod 1.40 s (−7% from 1.50 s).
  Parity intact at 1.3e-3 band-relative on q=1 corpus. Critically,
  this also proves the handle-direct CSF pattern is **innocent**
  of tick 85's 5× weber regression — that regression was the
  5-phase serial restructure, not the handle access pattern.

The post-tick-87 fusion + structural-change wave (ticks 89–96)
took the d_bands per-band launch count from 27 → 14:

- **`weber_contrast_compute_3ch_kernel`** (`af994a87`) — fuses
  the per-pixel `layer/clamp(L_bkg)` math and the shared
  `log_l_bkg = log10(L_bkg)` write into one launch per
  non-baseband level. Was 3 separate
  `weber_contrast_compute_kernel` launches. log10 computed
  once per pixel instead of three times.
- **`subtract_weber_3ch_kernel`** (`39d6957f`) — drops the
  `layer_c` intermediate entirely. Reads `fine_c` and
  `upscaled_c` handles directly and writes `band[c] =
  clamp((fine_c − upscaled_c) / L_bkg)` for all three channels
  + shared `log_l_bkg` in one launch. Was 3 `subtract_kernel`
  launches + the (already-fused) weber compute. Frees ~36 MB
  of `WeberScratch.layer_c` at 12 MP per side.
- **`pu_blur_h_3ch_kernel` + `pu_blur_v_3ch_scaled_kernel`**
  (`78d951d1`) — fuses the masking-branch pu_blur into one
  h-pass + one v-pass for all 3 channels, AND folds the
  `* 10^MASK_C` post-scale into the v-pass output. Cuts the
  masking blur chain from 9 launches per non-baseband level
  (3× h + 3× v + 3× `weight_band_kernel`) to 2.
- **`csf_apply_6ch_kernel`** (`7bf02fae`) — fuses the
  REF + DIST CSF apply into a single launch sharing the
  per-pixel LUT bracket math. Per non-baseband level: 2
  `csf_apply_3ch_kernel` launches → 1 6-channel launch.
- **`diff_abs_3ch_kernel`** (`06d8e4a5`) — moves the
  baseband `|T_p_dis - T_p_ref|` bypass to GPU. Every level's
  D plane now lives in the same `d_scratch.d[k][c]` slot.
- **`pool_band_kernel` in `compute_dkl_jod`** (`5817a2e4`)
  — replaces host-scalar `lp_norm_mean` over the per-band D
  Vecs with GPU `pool_band_kernel(d_handle) → partials[k*3+c]`.
  Partials buffer is `n_levels × N_CHANNELS` floats (~144 bytes
  at 12 MP); the host fold operates on that tiny Vec.
- **Split `compute_dkl_d_bands`** (`ea632f87`) — extracted
  `_dispatch_d_bands_into_scratch` private helper that does the
  GPU dispatch only. `compute_dkl_jod` calls the helper
  directly and skips the per-band Vec readback that
  `compute_dkl_d_bands` was paying. **17% wall-time win** at
  12 MP (jod 122.4 → 101.8 ns/px); jod is now faster than
  d_bands because it skips the ~432 MB host readback. vs
  fcvvdp 8-thread at 360p, the gap narrowed from 1.48× slower
  (tick 89) to 1.18× slower.

Post-fuse housekeeping (ticks 97–107):

- **`examples/time_size_sweep.rs`** + benchmark snapshot
  (`134bc04a`) — covers tiny (64²), small (256²), medium
  (1024²), large (4000×3000) sizes with per-phase wall + per-
  pixel cost + naive OLS fit. Found per-pixel cost is
  **non-monotonic** in image size: medium (1 MP) is the
  cheapest at 53.7 ns/px JOD, large (12 MP) regresses to
  159 ns/px; weber alone shows the same shape (19 → 61 ns/px),
  so the regression is intrinsic to the dispatch, not pure
  readback bandwidth. Open investigation.
- **`shadow_jod_gpu`** manifest-parity test (`562ee924`) —
  pins the GPU JOD path directly against pycvvdp v0.5.4's
  published manifest values (not just against the host
  scalar via relative parity). q=1 tolerance is wider (0.5
  JOD) per the documented cumulative-f32 drift; q≥20 tol is
  0.05 (observed < 0.001).
- **`Cvvdp::level_dims`** helper (`efcdba76`) — drops 5 sites
  of duplicated `if k == 0 { width } else { width >> k }`
  boilerplate. The `if k == 0` branch was redundant since
  `>> 0` is a no-op.
- **Dropped `Cvvdp.ref_log_l_bkg` dead field** (`ba586480`)
  — was added in tick 85 for a regression bisect that
  confirmed the field was NOT the cause; kept around with
  `#[allow(dead_code)]` for "future use" that subsequent
  ticks went around. Frees ~190 MB of unused GPU memory per
  `Cvvdp::new` at 12 MP, drops 14 lines of allocation code.
- **`compute_dkl_t_p_bands` modernized** (`8e509807`) — uses
  the fused `csf_apply_3ch_kernel` and reads weber from the
  GPU-resident `bands_ref` handles instead of re-uploading
  from the host Vec. Per non-baseband level: 3 host uploads
  + 3 launches → 0 uploads + 1 launch.

Post-fuse housekeeping (ticks 108–124):

- **Tests + examples + benches now run under wgpu** (`a0473bf9`,
  `3c72a86d`, `70a62e63`) — `shadow_jod_gpu`, `time_12mp`,
  `time_size_sweep`, and `benches/score.rs` all switched from
  cuda-only to the `cfg(any(cuda, wgpu))` + `Backend` type-alias
  pattern. Machines without a CUDA SDK (macOS, AMD, Intel) can
  now run the manifest-parity anchor + per-phase timings under
  wgpu's Vulkan/Metal/DX12 backend.
- **`ch_gain_for_band(is_baseband, band_mul)` helper** (`f5c1df3c`)
  — replaces 6 lines of `if is_baseband { 1.0 } else { band_mul *
  CH_GAIN[c] }` boilerplate at two band-loop sites with a single
  destructuring bind.
- **Stack-allocated `compute_dkl_jod` partials zero-init**
  (`a4e019c0`) — replaces a 192-byte heap Vec with
  `[0.0_f32; MAX_LEVELS * N_CHANNELS]` sliced to the active
  prefix.
- **CHANGELOG catch-up + PORT_STATUS refresh + many small doc
  fixes** (`bcf3dfcc`, `0dc01ea5`, `b7686203`, `35a0b48d`,
  `6826c0eb`, `77908be7`, `fd1e2527`, `8cd803a9`, `ac1e21d3`,
  `067ba379`, `08c65040`, `45719dad`, `1b8b51ca`) — module-level
  pipeline overviews in `lib.rs`, `pipeline.rs`, and
  `kernels/mod.rs` updated to name the actual fused kernels;
  stale claims about which stages run host-side cleared;
  `compute_dkl_weber_pyramid` got its missing doc comment; the
  misleading α/β OLS fit dropped from `time_size_sweep`; and 9
  of 15 rustdoc warnings cleared (remaining 6 are macro-induced
  by `#[cube(launch)]`'s function-and-module duplication).
- **`Cvvdp::score` v1 manifest tolerance** still pinned by the
  CPU reference path (`shadow_jod`). The GPU composition path
  is parity-locked against pycvvdp directly via `shadow_jod_gpu`
  but with a wider q=1 tolerance (~0.4 JOD) per the documented
  cumulative-f32 drift through `met2jod`'s steep slope.

Host-memory-pressure relief (ticks 144–146):

- **Drop dist_weber host Vec immediately** (`02f37728`) —
  `compute_dkl_d_bands` was binding the `(dist_weber, _)` tuple
  from `compute_dkl_weber_pyramid(dist_srgb)` even though the
  dist-side CSF path reads `self.bands_ref` GPU handles
  directly (per tick 87). Changed to `let _ = ...` so the
  ~190 MB host Vec drops at the call site instead of
  surviving the band loop.
- **Per-band ref-side host Vec drops** (`913a7c5f`) — after the
  band-`k` CSF dispatch finishes its `create_from_slice`
  uploads, replace `ref_weber[k] = [Vec::new(); 3]` and
  `ref_log_l_bkg[k] = Vec::new()` so peak host residency scales
  with the remaining-bands sum, not the whole pyramid.

Together these two commits dropped 12 MP perf
(`benchmarks/time_12mp_tick145_2026-05-14.md`):
- weber pyramid: 26.4 → 30.6 ns/px (noise band)
- compute_dkl_d_bands: 106.6 → **82.1 ns/px** (−23%)
- compute_dkl_jod: 101.8 → **87.2 ns/px** (−14%)

The `d_bands − 2×weber` bucket (CSF + masking + IO) dropped
from 645 ms → 252 ms — a **2.5× speedup** on the non-weber
portion. vs fcvvdp's 8-thread number at 360p we crossed from
1.48× slower (tick 89) to 1.18× slower (tick 96) to **1.01×
tied** here.

- **DIST weber pyramid skips host readback entirely**
  (`8c5b96e0`, tick 150) — `compute_dkl_d_bands` was calling
  `compute_dkl_weber_pyramid` for the DIST side and
  immediately discarding the returned tuple. Tick 144 caught
  the unused tuple; tick 150 caught that the *wrapper* itself
  still allocated ~240 MB of host Vecs and issued
  `client.read_one` calls that wait for the GPU dispatch to
  complete before transferring bytes. Replaced with
  `_dispatch_weber_pyramid_gpu` (the dispatch-only private
  helper) — skips both the allocation AND the GPU→host
  transfer.

  Result on the next 12 MP run
  (`benchmarks/time_12mp_tick150_2026-05-14.md`):
  - compute_dkl_d_bands: 82.1 → **71.0 ns/px** (−14%)
  - compute_dkl_jod: 87.2 → **74.6 ns/px** (−14%)
  - `d_bands − 2×weber`: 252 ms → 156 ms (−38%)
  - vs fcvvdp 8-thread @ 360p: now **1.15× faster** (vs 1.01×
    tied pre-tick).

Perf trajectory through the recent fusion + host-pressure wave:

| tick | jod ns/px | vs fcvvdp 8t @ 360p |
| ---- | --------- | ------------------- |
| 64   | 444       | 5.16× slower        |
| 73   | 127       | 1.48× slower        |
| 89   | 122       | 1.42× slower        |
| 96   | 102       | 1.18× slower        |
| 145  |  87       | 1.01× tied          |
| 150  |  **75**   | **1.15× faster**    |

Host-memory-pressure relief continued + structural readback
elimination (ticks 151–160):

- **REF CSF reads `bands_ref` GPU handles directly** (tick 155,
  `d7c7322c`) — symmetrical to tick 87's DIST-side fix. The
  band-loop's REF CSF dispatch had been uploading `ref_weber[k]`
  from the host Vec; after tick 154's `bands_ref` / `bands_dis`
  split persisted both sides' data on GPU, the REF CSF kernel
  reads `self.bands_ref[k]` handles in place. Drops 3 host→GPU
  uploads per non-baseband level (~50 MB total at 12 MP).
- **REF weber pyramid skips bands readback** (tick 156, `2993c0a0`)
  — `_dispatch_d_bands_into_scratch` had been calling the public
  `compute_dkl_weber_pyramid(ref_srgb)` wrapper which read back
  ~190 MB of bands per call (`Vec<Vec<f32>>`). Replaced with a
  direct call to `_dispatch_weber_pyramid_gpu` + a manual
  `log_l_bkg`-only readback loop. 12 MP jod 70.3 → 60.2 ns/px
  (−14%), now 1.43× faster than fcvvdp 8t.
- **Dispatch-only split for `compute_dkl_planes` + `compute_dkl_gauss_pyramid`**
  (tick 157) — extracted private `_dispatch_dkl_planes_gpu` and
  `_dispatch_gauss_pyramid_gpu` siblings.
  `_dispatch_weber_pyramid_gpu` and `compute_dkl_laplacian_pyramid`
  switched off the public wrappers (was `let _ = ...`). Saves
  ~230 MB of wasted host transfer per weber call (36 MB level-0
  + ~190 MB pyramid). 12 MP jod 60.2 → 53.0 ns/px (−12%), now
  1.62× faster than fcvvdp 8t.
- **GPU baseband-divide** (tick 158, `3b78f847`) — adds
  `baseband_divide_3ch_kernel` (pyramid.rs). The weber baseband
  finishing step had been doing 3 channel readbacks + 3 channel
  reuploads + per-channel host divides; now does 1 GPU launch
  using host-computed `l_bkg_mean` as a scalar uniform. Sync
  drain count per weber side: 4 → 1.
- **Tested-and-regressed 3ch upscale fusion + laplacian dispatch-only split**
  (tick 159, `6495c462`) — `upscale_v_3ch_kernel` /
  `upscale_h_3ch_kernel` (same fusion pattern as
  `weber_contrast_compute_3ch`) regressed jod ~4% at 12 MP on
  RTX CUDA across two runs. Hypothesis: 3ch register footprint
  reduced warp-level latency hiding more than launch overhead
  was costing us. Left a breadcrumb in pyramid.rs so this isn't
  re-tried without a different angle (e.g. shared-memory tiling).
  Same commit also added `_dispatch_laplacian_pyramid_gpu` so
  `compute_dkl_csf_weighted_bands` no longer discards a full-
  pyramid host readback via `let _ = ...`.
- **Direct parity test for `baseband_divide_3ch_kernel`**
  (tick 160, `baf4878e`) — closes a coverage gap from tick 158.
  The kernel had been verified through the higher-level
  `compute_dkl_weber_pyramid_matches_host_on_corpus_256x256`
  integration test; the new unit test in `pyramid_kernel.rs`
  gives a fast regression gate with inputs that exercise
  negatives, large magnitudes, and 3 distinct channel patterns.

12 MP perf trajectory through this wave
(`benchmarks/time_12mp_tick{155,156,157,158}_2026-05-14.md`):

| tick | jod ns/px | weber 1-side | d_bands  | vs fcvvdp 8t |
| ---- | --------- | -----------  | -------- | ------------ |
| 150  | 74.6      | 29.0         | 71.0     | 1.15× faster |
| 155  | 70.3      | 31.8         | 73.5     | 1.22× faster |
| 156  | 60.2      | 29.2         | 52.0     | 1.43× faster |
| 157  | 53.0      | 25.5         | 45.2     | 1.62× faster |
| 158  | **52.9**  | **24.9**     | **43.7** | **1.63× faster** |

Continued perf wave + structural cleanup (ticks 162–166):

- **PORT_STATUS.md refresh** (tick 162, `621a5867`) — weber-
  contrast pyr row names `baseband_divide_3ch_kernel`, composed-
  pipeline row carries the tick 158 perf number, "Open tick 159"
  entry documents the 3ch upscale fusion negative result.
- **`compute_dkl_t_p_bands` skips bands readback**
  (tick 163, `8a6de7be`) — same tick-156 pattern applied to the
  test-only T_p path. Was discarding the bands portion of
  `compute_dkl_weber_pyramid`'s return tuple (~190 MB host
  alloc per call at 12 MP). Now dispatches via the private
  helper + log_l_bkg-only readback.
- **Size-sweep re-measurement** (tick 164, `d27c5194`) —
  documents the tick 150-158 wave's per-bucket impact:
  - tiny    jod 1835 → 527 ns/px (−71%)
  - small   jod  223 →  91 ns/px (−59%)
  - medium  jod   65 →  28 ns/px (−56%)
  - large   jod  145 →  39 ns/px (−73%)
  Most importantly the medium→large per-pixel regression open
  since tick 97 **narrowed from 2.2× to 1.36×** — falsifies the
  L2-cache-pressure hypothesis as dominant; most of it was
  host memory pressure all along. Small (256²) is now the most-
  expensive per-pixel bucket — launch overhead dominates at
  that thread count.
- **`pool_band_3ch_kernel` fusion** (tick 165, `df4dd106`) —
  3 per-channel pool launches per level → 1 fused 3ch launch.
  Total pool dispatch: `n_levels × N_CHANNELS = 24` → `n_levels
  = 8` launches per JOD. Unlike tick 159's upscale 3ch fusion
  (regressed via register pressure), pool kernel does only 3
  powfs + 3 atomic-adds per thread — register footprint stays
  small, fusion wins on launch-overhead reduction. 12 MP jod
  52.9 → 49.0 ns/px (−7%), 1.76× faster than fcvvdp 8t.

  **Decision rule for 3-channel fusion** extracted from
  tick 159 vs tick 165: fusion wins when per-thread arithmetic
  is tiny (atomics, pointwise math); loses to register pressure
  on medium-arithmetic kernels (5-tap convolutions, multi-read
  patterns). Future 3ch fusion attempts should respect this.

- **`log_l_bkg` roundtrip elimination** (tick 166, `7ce2bc24`)
  — adds `WeberScratch.log_l_bkg_dis` throwaway destination
  (parallel to tick 154's `bands_dis` split) so the DIST weber
  dispatch's log_l_bkg write doesn't clobber REF's data on
  `weber_scratch[k].log_l_bkg`. Per cvvdp's weber_g1 rule,
  both sides use REF's log_l_bkg, so DIST's value is computed-
  then-discarded. The band loop's CSF kernel now reads REF's
  log_l_bkg directly from the GPU-resident handle — no host
  roundtrip.

  Bytes saved per JOD at 12 MP: ~128 MB (64 MB readback +
  64 MB reupload of the same data). Sync drains saved: 7
  (one per non-baseband level). 12 MP jod 49.0 → **41.8 ns/px**
  (−15%). Now **2.06× faster than fcvvdp 8-thread @ 360p**.

12 MP perf trajectory through ticks 165-166
(`benchmarks/time_12mp_tick{165,166}_2026-05-14.md`):

| tick | jod ns/px | weber 1-side | d_bands  | vs fcvvdp 8t |
| ---- | --------- | -----------  | -------- | ------------ |
| 158  | 52.9      | 24.9         | 43.7     | 1.63× faster |
| 165  | 49.0      | 23.4         | 41.3     | 1.76× faster |
| 166  | **41.8**  | **22.2**     | **39.8** | **2.06× faster** |

Warm-ref API + last per-JOD host alloc removed (ticks 168–171):

- **`fill_f32_kernel` + `baseband_log_l_bkg` pre-alloc**
  (tick 168, `e0b6ca62`) — replaces the baseband band's per-JOD
  `vec![log_l_bkg_baseband; n]` host alloc + GPU upload with a
  single GPU fill launch into a pre-allocated buffer. Wallclock
  impact minimal (baseband is small), but closes the last
  per-JOD host alloc in the hot path. New parity test
  `fill_f32_kernel_writes_uniform_value` uses a sentinel-fill
  trick to catch off-by-one or short-write bugs.
- **Extract REF/DIST weber helpers + perf snapshot**
  (tick 169, `ea13bcf8`) — factors
  `_dispatch_ref_weber_pyramid_only` and
  `_dispatch_dist_weber_pyramid_only` out of
  `_dispatch_d_bands_into_scratch`. No behaviour change, sets
  up the warm-ref API. The tick 169 measurement landed at
  jod 38.0 ns/px (2.26× faster than fcvvdp 8t @ 360p) —
  the tick 166 reading at 41.8 was on the high end of its noise
  band.
- **Warm-ref batch-scoring API** (tick 170, `abe3599d`) —
  delivers the `score_with_reference` doc promise from v0.0.1:
  - `Cvvdp::warm_reference(ref_srgb)` dispatches REF weber once
    and stores `Some(log_l_bkg_baseband)` in
    `Cvvdp::warm_ref_baseband_log_l_bkg`. Any subsequent method
    that dispatches REF weber resets this to `None` —
    `_dispatch_ref_weber_pyramid_only` does the reset
    unconditionally so warm-reference is the only path that
    arms it.
  - `Cvvdp::compute_dkl_jod_with_warm_ref(dist_srgb, ppd)`
    skips the REF half of the JOD pipeline. Returns
    `Error::NoWarmReference` if the cache is cold.
  - Refactored band loop + pool into `_dispatch_d_bands_dist_and_band_loop`
    and `_pool_and_finalize_jod` so cold and warm paths share
    the post-REF tail.
  - Parity test `compute_dkl_jod_with_warm_ref_matches_unwarm_path`
    verifies: (1) warm/cold byte-for-byte match within 1e-5
    JOD, (2) state survives multiple warm-ref calls,
    (3) intervening cold calls invalidate correctly.
- **`time_12mp` measures warm-ref fast path**
  (tick 171, `8c7c5f96`) — adds phase 4 measuring per-DIST cost
  after one `warm_reference` per iter. 12 MP results:
  - jod (cold REF):       36.1 ns/px
  - jod_warm (cached REF): **20.6 ns/px**
  - Per-DIST saving: 42.9% (1.75× faster per call)
  - vs fcvvdp 8-thread @ 360p: **4.17× faster per DIST**

Warm path delivers below the naive 50% saving because the host
pool fold + band loop dispatch overhead run once per JOD
regardless of REF state. The amortization break-even is ~2
candidates per warmed reference — anything larger lands at
1.75× throughput.

| tick | jod cold (ns/px) | jod warm (ns/px) | vs fcvvdp 8t (cold / warm) |
| ---- | ----             | ----             | ----                        |
| 158  | 52.9             | —                | 1.63× / —                   |
| 166  | 41.8             | —                | 2.06× / —                   |
| 169  | 38.0             | —                | 2.26× / —                   |
| 171  | **36.1**         | **20.6**         | **2.38× / 4.17× faster**    |

The `d_bands − 2×weber` bucket (CSF + masking + IO) is sub-noise
since tick 156: 2×weber ≈ d_bands, meaning the band-loop overhead
is now bandwidth-tightly packed against the two weber pyramids.
The next remaining hot spot is the gauss-pyramid reduce (5×5
downscale, 25 src reads per output pixel), which a shared-memory
tiled rewrite could shrink — but the per-thread register
pressure observation from tick 159 means any fusion attempt
should change the memory access pattern, not just rearrange
launches.

### Tick 175–178 — ceil-div correctness wave (resolves tick 174 drift)

After tick 174 root-caused the 0.586 JOD drift vs pycvvdp at 12 MP
to floor-div vs ceil-div pyramid halving, the next ticks shipped
the fix and locked it with new tests.

- **Ceil-div pyramid + MAX_LEVELS = 9** (tick 175, `cee15d24`)
  — `build_pyramid` / `build_weber_scratch` /
  `build_d_bands_scratch` / `pyramid_levels` switched from
  `n / 2` to `(n + 1) / 2`. Order mattered: bumping MAX_LEVELS
  alone (tick 174 attempt) widened the drift to 1.54; ceil-div
  first then bump levels closed it to 0.0003.
  - 4000×3000 synth: ours **9.4583** vs pycvvdp **9.4580** —
    **drift 0.586 → 0.0003 JOD** (2000× more accurate).
  - All 67 existing parity tests stayed green (they run at
    power-of-2 sizes where floor == ceil at every level).
  - Trade-off: jod cold 36 → 62 ns/px, warm-ref 21 → 34 ns/px
    on the same RTX 5070. Open investigation — total pixel
    work is nearly unchanged, so the ~25% post-warmup slowdown
    must be a kernel-dispatch or boundary-branch interaction,
    not extra compute. Snapshot: `benchmarks/pycvvdp_parity_tick175_2026-05-15.md`.

- **`level_dims` reads `gauss_ref` shapes** (tick 176, `b9b5b71a`)
  — was computing `(bw, bh, n_px)` via `width >> k` (floor-div
  bit shift), which disagreed with the ceil-div allocator at
  odd-dim levels. Consequence: the band loop's CSF + masking +
  pool kernels dispatched fewer threads than the bands_ref /
  d_scratch buffers actually held — the last few tail pixels at
  each odd-dim level were written by weber but never processed
  downstream. 12 MP JOD output unchanged (tail values were
  near-zero so didn't move the pool), but the inconsistency
  was real and would matter on other inputs. Now reads
  `gauss_ref[k].w / .h` directly so all shape-using sites
  agree.

- **Odd-dim JOD parity test** (tick 177, `f2425dce`) — added
  `compute_dkl_jod_matches_host_scalar_on_odd_dims` at 73×91
  (the smallest source that diverges at ceil-vs-floor level 4+).
  Catches future floor-div regressions in either host_scalar
  or the GPU pyramid path. The other JOD parity tests all run
  at power-of-2 sizes where floor == ceil.

- **12 MP pycvvdp golden parity test** (tick 178, `cd61a217`)
  — added `compute_dkl_jod_matches_pycvvdp_at_12mp_synth`. The
  deterministic 4000×3000 synth pair from
  `examples/time_12mp.rs` runs through `compute_dkl_jod` and
  asserts the output matches pycvvdp v0.5.4's measured 9.4580
  golden within 0.005 JOD. Current observed diff: 0.0003.
  Would have failed at tick 173 (diff 0.586) and tick 174
  (diff 1.54); now gates the canonical-reference correctness
  in CI. Runtime ~5 s per call.

The pycvvdp parity matrix is now end-to-end:

| size      | test                                                              | tolerance | observed |
| ----      | ----                                                              | ----      | ----     |
| 32×32     | `compute_dkl_jod_matches_host_scalar`                            | 0.5 JOD   | < 0.1    |
| 73×91     | `compute_dkl_jod_matches_host_scalar_on_odd_dims`                | 0.5 JOD   | **0.0004** (post tick 181) |
| 256×256   | `compute_dkl_jod_matches_host_on_corpus_256x256` (drift sweep)   | 0.06 JOD  | < 0.05   |
| 4000×3000 | `compute_dkl_jod_matches_pycvvdp_at_12mp_synth`                  | 0.005 JOD | **0.0003** |
| 256×256 v1 manifest | `shadow_jod` (host scalar)                              | 0.01 JOD  | < 0.006  |

### Tick 179–182 — band-count alignment + pycvvdp goldens manifest

- **CHANGELOG / PORT_STATUS / lib.rs docs caught up to tick 175-178**
  (tick 179, `d7f8445f`) — the ceil-div correctness wave is now
  surfaced in user-facing docs. Corrected `lib.rs` to drop the
  misleading "2.58× slower than pycvvdp" framing (those numbers
  reflected a broken pyramid drifting 0.586 JOD); honest post-fix
  is 4.4× slower cold / 2.4× slower warm with correct output.

- **Extended pycvvdp bench script + goldens manifest**
  (tick 180, `b937401e`) — `scripts/cvvdp_goldens/bench_12mp_cuda.py`
  now produces a `pycvvdp_synth_goldens.json` manifest with the
  pycvvdp golden JOD for both the 4000×3000 12 MP fixture
  (9.4580) and a 73×91 odd-dim fixture (9.3904). The manifest
  schema lets future Rust parity tests load canonical reference
  values directly instead of duplicating hardcoded constants.

- **Surprise: host_scalar drifts ~0.6 JOD vs pycvvdp at 73×91**
  (tick 180 finding) — at sub-megapixel sizes our host_scalar
  reference produces 8.79 vs pycvvdp 9.39. The 256² v1 manifest
  fixtures hold ≤ 0.006 JOD, the 4000×3000 synth holds 0.0003,
  but 73×91 drifts ~0.6. Possible causes (open investigation):
  CSF interpolation at very small angular widths, band-mul rule
  difference for the small-band branch, or a display-geometry
  interpretation gap at sub-degree image sizes.

- **`pyramid_levels` defers to `band_frequencies` (tick 181, `e4951c15`)**
  — the GPU pipeline had a size-based cap (`cur >= 2 *
  PYRAMID_MIN_DIM`) that produced fewer bands than host_scalar
  at small inputs (4 vs 5 at 32², 5 vs 6 at 73×91, 7 vs 8 at
  256²). host_scalar already used `band_frequencies(ppd, w, h).len()`
  directly. Aligned the GPU side. Effect on the 73×91 GPU-vs-host
  parity test: **diff 0.092 → 0.0004 JOD** (235× better
  agreement). 12 MP pycvvdp gate still passes at 0.0003.

  Resolves the GPU↔host structural mismatch at small sizes.
  The remaining ~0.6 JOD drift at 73×91 is purely host_scalar
  vs pycvvdp (GPU now matches host within f32 precision).

### Investigation Notes (cvvdp-gpu, tick 174 — large-image drift)

After tick 173's pycvvdp v0.5.4 CUDA bench surfaced a **0.586 JOD
drift** between our `compute_dkl_jod` and pycvvdp on a 4000×3000
synthetic pair (ours 8.8726, pycvvdp 9.4580), tick 174 traced the
cause. Diagnostic scripts in `scripts/cvvdp_goldens/`:

- `bench_12mp_cuda.py` — pycvvdp CUDA timing + JOD output
- `diagnose_12mp.py` — pycvvdp metric internals
- `diagnose_pyramid.py` — pycvvdp band_freqs + height + pyr_shape
- `diagnose_freqs.py` — direct comparison of band frequencies
- `diagnose_decompose.py` — actual band tensor shapes via decompose()

**Two structural divergences from pycvvdp at large sizes:**

1. **n_bands cap**. Our `MAX_LEVELS = 8` caps the pyramid at 8
   levels. pycvvdp uses **9 bands** at 4000×3000 (one extra deep
   level). Bumping `MAX_LEVELS` alone is insufficient — see #2.

2. **Floor vs ceil division on pyramid sizes** (the dominant
   cause). pycvvdp uses **ceil-div** when halving level
   dimensions; we use floor-div. The bands diverge from level 4
   onward:

   | level | pycvvdp shape (ceil)  | cvvdp-gpu shape (floor) |
   | ---   | ---                   | ---                     |
   | 0     | 3000×4000             | 3000×4000               |
   | 1     | 1500×2000             | 1500×2000               |
   | 2     | 750×1000              | 750×1000                |
   | 3     | 375×500               | 375×500                 |
   | 4     | **188**×250           | **187**×250             |
   | 5     | 94×125                | 93×125                  |
   | 6     | **47×63**             | **46×62**               |
   | 7     | 24×32                 | 23×31                   |
   | 8     | 12×16 (baseband)      | (n/a — capped)          |

   Naively bumping MAX_LEVELS to 10 + adding level 8 INCREASED
   the drift (JOD 8.87 → 7.92) because the ceil-div mismatch
   compounds with every additional level. Reverted MAX_LEVELS
   to 8 until the ceil-div fix lands.

The 0.006 JOD parity tolerance our existing tests hit at 256×256
holds because at small sizes the ceil/floor difference is 0 or 1
pixel and most of pycvvdp's pyramid math rounds out. At 12 MP
the divergence stacks to ~0.6 JOD.

**Fix plan** (multi-tick):
- Switch pyramid `Level` allocator + `gauss_ref` chain to
  ceil-div (`(w + 1) / 2`).
- Update `downscale_kernel` boundary handling for the off-by-one
  case (currently floor-div semantics).
- Update upscale `back_v` / `back_h` math which assumes the
  parent floor-div shape.
- Bump MAX_LEVELS to 10 once ceil-div parity holds at 256×256.
- Add a 12 MP parity test driven by a pycvvdp golden so the
  drift is visible in CI.

**Goldens expansion (user ask, 2026-05-15):**

> pycvvdp needs to be the source of goldens and we have to sweep
> a larger distortion set

Current goldens at `v1/manifest.json` only cover 256×256 source
×6 JPEG quality levels. Planned expansion:
- Multi-resolution: 256², 1024², 4000×3000 (and 8K for sanity).
- More distortion types: Gaussian blur, Gaussian noise,
  contrast/saturation perturbations, downscale+upscale, color
  shifts, dithering, banding.
- Quality levels closer to perceptual JND than just JPEG-q.
- Sweep dimension: image content (photo, screen, line-art) so the
  golden corpus stratifies across the codec-corpus categories.

Goldens regenerator script (`build_goldens.py`) needs to grow a
distortion-config DSL + a multi-resolution + multi-image pipeline
before this expansion can land cleanly.

**cvvdp-gpu vs pycvvdp perf gap (cuDNN / Burn / cubek):**

User suggestion (2026-05-15):

> Burn is a libtorch alternative so we should be able to beat
> pycvvdp on GPU — maybe we didn't update to the latest cubecl
> 0.10 release or use the best algorithms in cubek?

Current state:
- cubecl pin: `0.10.0-pre.4` (per workspace Cargo.lock). The
  cubek (`tracel-ai/cubek`) high-level kernel library at
  `cubecl-kernels` exposes well-optimised matmul, conv, reduce.
- pycvvdp's hot path is the downscale/upscale Gaussian pyramid
  — pure depthwise separable convolution. PyTorch routes this
  via cuDNN, which has hand-tuned per-arch kernels.
- The cubek conv kernel (depthwise 5-tap, shared-memory tiled)
  would close the gap if it matches cuDNN. We currently do not
  use cubek conv — our `downscale_kernel` /
  `upscale_v_kernel` / `upscale_h_kernel` are hand-rolled 5-tap.

Investigation queued: try replacing the downscale/upscale
kernels with cubek-conv calls and re-measure. If cubek-conv
holds parity (separable filter, ceil-div boundaries) and lands
≤ pycvvdp at 12 MP, that's our path to "beat libtorch".

### Investigation Notes (cvvdp-gpu, post-tick-81)

These observations don't ship as code, but they document
findings that would otherwise be re-discovered:

- **Standalone weber(dist) is not slower than weber(ref)** —
  the consecutive-weber diagnostic in `examples/time_12mp.rs`
  shows two back-to-back `compute_dkl_weber_pyramid` calls on
  the same `ref_bytes` complete in nearly identical time. The
  "weber(dist) is 2× weber(ref)" effect observed inside
  `compute_dkl_d_bands` is therefore not algorithmic, not a
  cubecl warm-up artifact, and not driver thermal throttling.
  It is host memory pressure: ~190 MB of `ref_weber` Vec stays
  alive across the second call.
- **Tick 85's failed 5-phase d_bands refactor regressed
  standalone weber by 5×** (260 ms → 1300 ms) — the per-band
  bisect ruled out: (a) the new `self.ref_log_l_bkg` field
  itself (allocation-only does not regress), (b) the new
  `log_l_bkg_dest` parameter on `_dispatch_weber_pyramid_gpu`,
  and (c) the GPU memory-handle pattern (the dist-side CSF
  optimization above confirms this). The proven cause is the
  5-phase serial control-flow structure (all CSF(ref) bands →
  weber(dist) → all CSF(dist) bands → all masking), but the
  actual mechanism (cubecl sync barrier? memory-pool
  fragmentation? kernel-scheduler ordering?) remains unknown.
  Future attempts at the d_bands restructure should bisect a
  different axis (interleaved-per-level vs. phase-serial)
  rather than re-flatten the existing structure.

Net 12 MP performance trajectory (CUDA, RTX-class):

| metric                          | tick 64   | tick 73    | tick 171   |
| ----                            | ----      | ----       | ----       |
| weber pyramid (1 side)          | 103 ns/px | 21.6 ns/px | 18.7 ns/px |
| compute_dkl_d_bands             | 428 ns/px | 121 ns/px  | 33.7 ns/px |
| compute_dkl_jod (cold REF)      | 444 ns/px | 127 ns/px  | **36.1 ns/px** |
| compute_dkl_jod_with_warm_ref   | —         | —          | **20.6 ns/px** |

### Honest comparison against the canonical reference (tick 173)

The fcvvdp ratios cited in earlier rows compare against
`halidecx/fcvvdp` — a separate C+Zig fork, not the canonical
pycvvdp at `gfxdisp/ColorVideoVDP`. Direct pycvvdp v0.5.4
CUDA measurement on the same RTX 5070 host:

| metric                          | per-pixel  | vs pycvvdp CUDA |
| -----                           | ----       | ----            |
| **pycvvdp v0.5.4 (CUDA)**       | **14 ns/px** | baseline        |
| cvvdp-gpu cold                  | 36.1 ns/px | **2.58× slower** |
| cvvdp-gpu warm-ref              | 20.6 ns/px | **1.47× slower** |

pycvvdp benefits from cuDNN-optimised separable convolution on
the downscale/upscale pyramid; our cubecl kernels are hand-written
5-tap separable. cvvdp-gpu wins on portability (WGPU + HIP
backends, ~50 MB static binary vs ~3 GB PyTorch runtime, ~1 s
warm-up vs 1-13 s graph compile) but loses on raw CUDA throughput.

See `crates/cvvdp-gpu/benchmarks/pycvvdp_12mp_cuda_2026-05-14.md`
+ `scripts/cvvdp_goldens/bench_12mp_cuda.py` for the
reproduction recipe.

### vs fcvvdp (separate C+Zig fork, NOT the canonical reference)

fcvvdp's published 360p bench (i7-13700k):

| fcvvdp variant | per-pixel  | vs cvvdp-gpu cold @ 12 MP | vs cvvdp-gpu warm @ 12 MP |
| ----           | ----       | ----                       | ----                       |
| 1-thread       | 214 ns/px  | cvvdp-gpu **5.93× faster** | cvvdp-gpu **10.39× faster** |
| 8-thread       |  86 ns/px  | cvvdp-gpu **2.38× faster** | cvvdp-gpu **4.17× faster**  |

The fcvvdp comparison is real (numbers measured, ratios correct)
but **fcvvdp is not pycvvdp**. Use the pycvvdp row for the
canonical comparison.

### Fixed

#### cvvdp-gpu

- `host_scalar::predict_jod_still_3ch` index-out-of-bounds at
  image sizes where `band_frequencies` truncates below
  `ilog2(min(w, h))` (e.g. 1024×1024). The auto-pick now queries
  `band_frequencies(...).len()` instead of falling through to the
  `ilog2`-based default.

### Removed

#### cvvdp-gpu

- Dead `masked_diff_kernel` cubecl stub (always wrote 0.0; never
  launched).
- Dead `upscale_kernel` cubecl stub (replaced by the
  `upscale_v_kernel` + `upscale_h_kernel` pair).
- Empty `kernels::reduce` module (planned scope landed in
  `kernels::pool` instead).

#### zen-metrics-cli

- New `cvvdp` metric (`--metric cvvdp`). GPU bundle (`--features
  gpu`) now includes `gpu-cvvdp`. Sweep TSVs pick up the
  `score_cvvdp` column automatically.

### Workspace

- CI builds the new `cvvdp-gpu` crate alongside the existing four
  `-gpu` crates under `wgpu` (per-platform) and as part of the
  `i686-unknown-linux-gnu` cross-compile sanity job.
