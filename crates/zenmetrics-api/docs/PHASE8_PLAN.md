# Phase 8 + 9 plan — post-/goal hardening (2026-05-27)

After the orchestrator /goal landed (Phase 7.7.1 default-flip), user
requested five expansions:

1. Graceful no-GPU fallback verification + fix
2. Persistent kernel cache (varied by deps commit + CUDA runtime)
3. Rename `cvvdp-cpu` → `cvvdp`, flip dependency direction
4. Upstream the pinned-upload patch to cubecl
5. **NEW**: establish maintained cubecl forks (multiple if needed)
6. **NEW**: support Metal as a first-class backend
7. **NEW**: orchestrator improves GPU % utilization via concurrency
   (CUDA streams, multi-worker per device, pipelined transfer + compute)

## Tasks

### Phase 8a — no-GPU fallback verification + fix
Small, independent. Add a `ZENMETRICS_FORCE_NO_GPU=1` test fixture
that bypasses GPU detection. Assert end-to-end CPU scoring works
without crashes when no GPU is present (and when libcuda dlopen
fails at runtime — cubecl-cuda is lazy). 1-day task.

### Phase 8c — rename `cvvdp-cpu` → `cvvdp`  [LANDED — rename only]
Move `crates/cvvdp-cpu` → `crates/cvvdp`. Update package name. Flip
dep direction: `cvvdp-gpu` now `[dependencies] cvvdp = { path = "../cvvdp" }`
to share the `Score` type + CPU fallback. Update all callers
(workspace members, zenmetrics-orchestrator's `cpu-cvvdp` feature,
tests). Extend to `iwssim` consistency check (no upstream iwssim
crate exists, so iwssim-gpu stays singular — document). 1-day task.

**Status (2026-05-27): rename landed; dep flip deferred.** The
mechanical rename of `crates/cvvdp-cpu` → `crates/cvvdp` shipped,
with the workspace member, package name, all callers
(`zenmetrics-orchestrator` / `cvvdp-conformance` / docs / examples
/ tests) updated. The `cpu-cvvdp` feature flag name is preserved
(semantic — describes what it enables, not the dep crate name); the
public column-name namespace `cvvdp_cpu_imazen_v*` and env override
`CVVDP_CPU_IMPL_TAG` are also preserved so downstream sweeps keep
aligning.

The dep-direction flip (`cvvdp-gpu` depending on `cvvdp` rather than
the inverse) is **deferred to Phase 8c.1** — the shared CSF / DKL /
masking / params constants currently live in `cvvdp-gpu::kernels::*`
and `cvvdp-gpu::params`, and extracting a shared base crate (or
moving the constants into `cvvdp` for `cvvdp-gpu` to consume) is a
non-trivial restructuring beyond the rename. The current direction
(`cvvdp` depends on `cvvdp-gpu` for those constants) compiles
cleanly and matches Phase 7.7.1 runtime behaviour, so the flip is
not on the Phase 8 critical path.

Phase 8c.1 (follow-up) — extract shared constants into `cvvdp`,
make `cvvdp-gpu` depend on `cvvdp`, share `Score` / `Jod` / display
geometry types. ~1-day task once scheduled.

iwssim consistency: there is no upstream iwssim CPU crate to mirror,
so iwssim-gpu stays singular; no rename required. The naming asymmetry
(`cvvdp` for CPU base, `cvvdp-gpu` for GPU variant, vs `iwssim-gpu`
with no CPU sibling) is intentional and documented here.

### Phase 8d — upstream pinned-upload PR to tracel-ai/cubecl
**SCRAPPED 2026-05-27 per user directive "dont upstream coordinate,
fork to new crate names - zenforks-cubecl-".** The PR draft work
landed at commit 4355dce0 (`PINNED_UPLOAD_UPSTREAM_PR.md` + README
fork-dep section) but the upstream submission path is abandoned.
The pinned-upload patch (and any future patches) ship through our
own published forks instead — see Phase 8e revised below.

The draft PR doc + bench numbers are retained as **internal
methodology reference** — if a future maintainer wants to
re-attempt upstreaming after we've validated patches in production,
the diff + bench harness are ready.

### Phase 8e — published `zenforks-cubecl-*` fork crates + persistent cache + Metal

**REVISED 2026-05-27 per user directive "dont upstream coordinate,
fork to new crate names - zenforks-cubecl-".** Instead of maintaining
a git-fork dep on `lilith/cubecl` and chasing upstream PRs, we
publish renamed forked crates to crates.io. Downstream library
users get stable `Cargo.toml`-pinned deps without git-deps.

#### 8e.1 — establish `zenforks-cubecl-*` publication strategy
- Fork `lilith/cubecl` rev de2f9857 (or current pin) → rename ALL
  crates under the `zenforks-cubecl-` prefix:
  - `cubecl` → `zenforks-cubecl`
  - `cubecl-runtime` → `zenforks-cubecl-runtime`
  - `cubecl-core` → `zenforks-cubecl-core`
  - `cubecl-common` → `zenforks-cubecl-common`
  - `cubecl-ir` → `zenforks-cubecl-ir`
  - `cubecl-cuda` → `zenforks-cubecl-cuda`
  - `cubecl-cpu` → `zenforks-cubecl-cpu`
  - `cubecl-wgpu` → `zenforks-cubecl-wgpu`
  - `cubecl-hip` → `zenforks-cubecl-hip`
  - `cubecl-cpp` → `zenforks-cubecl-cpp`
  - (any others encountered during the rename)
- Update all internal cross-crate refs (each `zenforks-cubecl-*`
  crate references siblings by the renamed names)
- Versioning: `0.10.X` series tracking upstream's `0.10.X` with our
  patches baked in. Bump on cubecl upstream changes we cherry-pick
  OR on our own patch additions.
- Publish to crates.io as `zenforks-cubecl-*` (one publish per
  crate, in dep-order)
- Document the fork-rename strategy + upgrade workflow in
  `crates/zenmetrics-api/docs/ZENFORKS_CUBECL_STRATEGY.md`
- Workspace `Cargo.toml`: replace all `cubecl = { git = ... }` deps
  with `zenforks-cubecl-* = "0.10.x"` from crates.io
- All zenmetrics `-gpu` crates: rename `use cubecl::*` → `use
  zenforks_cubecl::*` (mechanical refactor)

This is substantially more work than the original "maintained git
fork" plan but unblocks downstream library users completely. They
can `cargo add zenforks-cubecl` without git deps.

#### 8e.2 — persistent PTX cache in our cubecl fork
- Hash kernel source + compile flags → cache key
- Write PTX to `~/.cache/cubecl/<cache_key>.ptx` after compile
- On startup: source-hash hit → skip source→PTX, hand PTX to driver
  (driver still does its PTX→SASS JIT, but that's already fast +
  cached at `~/.nv/ComputeCache/`)
- Expected effect: cold-start CLI from ~18s → ~500ms on a fresh
  process where source hasn't changed

#### 8e.3 — cache invalidation by deps commit + CUDA runtime
The cache key must include:
- `cubecl_source_hash` — the kernel source bytes (already)
- `cubecl_crate_commit` — our fork's HEAD SHA (different cubecl version
  may produce different PTX even from the same source)
- `cuda_runtime_version` — driver may JIT differently across versions
- `gpu_compute_capability` — different cap may need different PTX features
- `gpu_model` (optional — usually compute_capability captures this)

When any of those change, the cache entry is stale and re-compiled.
Mechanism: embed all 4 in the cache key directory structure:
`~/.cache/cubecl/<cubecl_sha>/<cuda_runtime>/<compute_cap>/<source_hash>.ptx`

#### 8e.4 — Metal `Atomic<f32>` fix (per CLAUDE.md note)
ssim2-gpu's `fast-reduction` feature is OFF by default because cubecl-wgpu's
Metal backend reports `Atomic<f32> = LoadStore|Add` as supported but the
codegen silently no-ops — every reduction returns zero, every score
collapses to ~100.

Two fix paths:
- **Upstream fix**: contribute to cubecl-wgpu's Metal codegen so atomics
  actually emit the right Metal Shading Language intrinsics
- **Workaround**: ensure all 6 metrics use the non-atomic partial-sum +
  finalize reduction path universally (matches what ssim2 already does
  by default since task #52)

Both worth doing — upstream is the right fix; workaround unblocks Metal
shipping in the meantime.

#### 8e.5 — Metal CI + verification
- Add `macos-latest` runners to `.github/workflows/ci.yml` (Apple Silicon
  by default; `macos-15-intel` for Intel coverage per CLAUDE.md)
- Run a per-metric Metal parity test asserting scores match CUDA within
  documented tolerance
- Document Metal-specific perf characteristics in each `-gpu` crate's
  README

### Phase 9 — orchestrator GPU concurrency

Currently the orchestrator pool has ONE GPU worker per device (Phase 5
design). Tasks serialize. GPU % utilization is bounded by single-task
kernel launches.

Improvements:

#### 9.1 — CUDA streams for true concurrency
Each "worker" within the pool gets its own CUDA stream. Multiple
streams can have concurrent kernels in flight on the same GPU
(provided memory budget allows). Implementation:
- cubecl 0.10 already exposes per-handle streams (verify)
- Pool maintains N "lanes" per GPU device, each with its own stream
- N = `min(recommend_parallel(median_size), 4)` — bounded so we
  don't oversaturate the GPU's command queue

#### 9.2 — pipelined transfer + compute overlap
For batched workloads (sweep), pipeline:
- Lane 0: uploading task K+1's bytes
- Lane 1: computing task K
- Lane 2: downloading task K-1's result
- Lane 3: about to upload task K+2

The cubecl pinned-staging (already shipped) helps the upload latency;
this pipeline overlaps it with compute.

#### 9.3 — adaptive worker count based on observed GPU utilization
Watch `nvidia-smi --query-gpu=utilization.gpu` periodically. If
utilization stays below ~80% with 1 worker, spin up a 2nd. If
contention (VRAM, watchdog timeouts) appears, drop back.

Constants: `target_gpu_utilization = 80%`, `max_workers_per_device = 4`,
`adjust_interval = 5s`.

#### 9.4 — bench-driven worker-count selection
Phase 2's bench runner already measures per-cell wall + VRAM. Extend
to measure with N=1, 2, 4 concurrent workers. Cache the optimal N
per (metric, size) in the capability profile. The chooser then uses
N from cache.

## Dispatch order

Wave 1 (parallel, small): 8a, 8c, 8d, the cubecl-fork-design doc

Wave 2 (sequential, big — after Wave 1 lands):
- 8e (umbrella: 8e.1 → 8e.2 → 8e.3 → 8e.4 → 8e.5)
- 9 (sub-phases sequential: 9.1 → 9.2 → 9.3 → 9.4)

8e and 9 can run in parallel waves once their respective designs are
pinned because they touch different parts of the stack (cubecl
internals vs orchestrator scheduling).

Total estimated effort: ~4-6 weeks for Phase 8 + 9 combined.
