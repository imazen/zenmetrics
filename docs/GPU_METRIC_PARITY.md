# Cross-repo GPU perceptual-metric parity

**Status (2026-05-26): SKELETON LANDED, RUNTIME PENDING.**

## Why this document exists

The master dedup audit
(`zensim/benchmarks/dedup_inventory_master_2026-05-26.md`, Tier-0 #2 /
§A.6 #1 / Class 7) found that we ship **two independent GPU
implementations of the same SSIMULACRA2 metric**:

| Backend                       | Crate                          | Runtime                            | Coverage                          |
|---                            |---                             |---                                 |---                                |
| **CubeCL** (this repo)        | `ssim2-gpu`                    | lilith/cubecl fork                 | CUDA / wgpu (Vulkan/DX12/Metal) / HIP / CPU |
| **cudarse / turbo-metrics**   | `coefficient::gpu::GpuMetrics` | `cudarse-driver` + `cudarse-npp`   | NVIDIA CUDA only                  |

Same metric, two backends, no cross-impl parity test → silent
risk that they score differently on the same input and no one notices
until a downstream join (a picker bake, a Pareto comparison, a
selector training set) mixes scores from both.  The audit ranks this
as one of the Top-5 risks in Cluster A; converting it from "could be
different and nobody knows" to "measured agreement within X SSIM2
points" is the cheapest first step before deciding whether to migrate
coefficient onto CubeCL or extract a shared `zen-gpu-metrics`.

This document records the methodology + tolerance + measured deltas.

## Methodology

* **Same input shape** for both backends: packed sRGB-RGB8 `&[u8]` of
  length `width * height * 3`, row-major, no stride padding.
* **Same output convention**: scalar SSIM2 score where higher = better,
  100 = identical.  Both backends return `f64`.
* **Fixtures**: 3 CID22 photos at descending JPEG qualities, chosen to
  cover the SSIM2 range where the two backends are most likely to
  disagree (near-identical, mid-range, heavy distortion).  Distorted
  variants are synthesized inside the test (round-trip through the
  `image` crate's JPEG codec) so no pre-staged fixtures are required.
* **Initial tolerance**: `abs(zenmetrics - coefficient) < 0.5` SSIM2
  points.  Two independent multi-scale perceptual metric pipelines
  with different blur kernels, reduction orders, and linearization
  sites will never agree to 1e-6; start loose, tighten in a follow-on
  once the measured agreement is in the table below.

The test lives at
[`crates/ssim2-gpu/tests/cudarse_parity.rs`](../crates/ssim2-gpu/tests/cudarse_parity.rs)
and is gated behind the `cudarse-parity` Cargo feature.  It is marked
`#[ignore]` so that `cargo check -p ssim2-gpu --features
cudarse-parity` validates the signature surface without requiring CUDA
+ coefficient's gpu deps to build everywhere.

## Run instructions

```bash
# Skeleton-only: validates the test compiles and the optional cudarse
# dep resolves.  Runs on any machine where the `coefficient` crate's
# `gpu` feature builds (currently requires a live `../turbo-metrics`).
cargo check -p ssim2-gpu --features cudarse-parity

# Full run on a CUDA-equipped host with coefficient's gpu deps
# resolvable (NVIDIA driver + toolkit + path to turbo-metrics).
cargo test -p ssim2-gpu --features cudarse-parity \
    --test cudarse_parity -- --ignored --nocapture
```

When the test runs, the `--nocapture` output prints one line per
fixture:

```text
fixture=<label> dims=<W>x<H> cubecl=<score> cudarse=<score> delta=<delta>
max_delta across fixtures = <value>
```

Paste those numbers into the **Measured agreement** table below.

## Why `#[ignore]` (not always-runnable)

Three independent blockers measured on this dev host on 2026-05-26
during the skeleton-landing chunk:

1. **coefficient's `gpu` feature has dangling path-deps on
   `../turbo-metrics`.**  `coefficient/Cargo.toml:115-120` references
   `../turbo-metrics/crates/{ssimulacra2-cuda, butteraugli-cuda,
   dssim-cuda, cudarse/cudarse-driver, cudarse/cudarse-npp}` — but
   `~/work/turbo-metrics` was archived to
   `~/work/turbo-metrics--archived-2026-05-06`, leaving the path-dep
   dangling.  **Workaround validated on this host**: symlink
   `~/work/turbo-metrics -> turbo-metrics--archived-2026-05-06`.
   With the symlink, `cargo check -p ssim2-gpu --features
   cudarse-parity` proceeds past Cargo resolution.
2. **Cudarse build needs `CUDA_PATH` env var** even when nvcc is on
   `PATH`.  `nvptx-builder/src/lib.rs:111` reads `CUDA_PATH`
   explicitly; failure mode is a panic `CUDA_PATH must be set to the
   path of your CUDA installation: NotPresent`.  Workaround: invoke
   with `CUDA_PATH=/usr/local/cuda cargo check --features
   cudarse-parity`.
3. **Archived turbo-metrics targets `sm_70` which CUDA 13.2's ptxas
   no longer accepts.**  Build fails with `ptxas fatal: Value
   'sm_70' is not defined for option 'gpu-name'`.  This is the
   blocker that prevents the test from actually running on this host
   as of 2026-05-26 — fixing it requires either pinning to a CUDA
   12.x toolkit OR porting the archived turbo-metrics kernels to a
   currently-supported SM target.  Neither is in scope for the
   parity-test chunk.

`#[ignore]` lets the test skeleton land on master without making CI
or `cargo test -p ssim2-gpu` red on hosts where any of the three
blockers applies.  When the runtime blockers are fixed (likely by
the broader audit chunk that picks the canonical GPU backend), drop
the `#[ignore]` attribute and re-record the measured-agreement table
below.

## Measured agreement

| Date       | Fixture           | dims    | CubeCL score | cudarse score | abs delta | within tolerance? |
|---         |---                |---      |---           |---            |---        |---                |
| TBD        | 1001682_q90       | 512×512 | TBD          | TBD           | TBD       | TBD               |
| TBD        | 1028637_q50       | 512×512 | TBD          | TBD           | TBD       | TBD               |
| TBD        | 1029604_q20       | 512×512 | TBD          | TBD           | TBD       | TBD               |

Once at least one row has real numbers, document the tolerance
disposition: tighten the constant in the test, OR open a finding
issue if the delta exceeds the loose initial 0.5 SSIM2 gate.

## What this test does NOT cover (follow-on candidates)

* **Butteraugli parity** (`butteraugli-gpu` vs
  `butteraugli-cuda`) — same shape, ~30 LOC to add.
* **DSSIM parity** (`dssim-gpu` vs `dssim-cuda`) — same shape.
* **Per-octave intermediate parity** — single-backend run-to-run
  determinism is covered by
  [`tests/reduction_determinism.rs`](../crates/ssim2-gpu/tests/reduction_determinism.rs).
  Cross-backend agreement on the per-octave intermediates is a
  tighter gate worth landing after the scalar score gate stabilizes
  and any large discrepancies have been localized.
* **Cross-platform parity** — same backend (CubeCL) across CUDA vs
  wgpu vs HIP.  Different problem; the existing `parity_lock` /
  `aliasing_invariants` tests already exercise this on a single
  backend.
* **Migration plan** — once the measured agreement is known, the
  audit's recommended next step is either "delete coefficient's
  `src/gpu.rs` and depend on `ssim2-gpu`" or "extract a shared
  `zen-gpu-metrics` interface and let both backends implement it".
  Both are out of scope for the parity-test chunk; this doc is the
  measurement that enables the decision.

## CI integration

Not yet wired.  When the measured-agreement table has 3 real rows
and the tolerance is documented, propose a follow-on chunk to:

* Run `cargo test --features cudarse-parity --test cudarse_parity --
  --ignored` on a CUDA-equipped GitHub Actions runner (one of the
  `T4` / `L4` self-hosted runners zenmetrics' sweep fleet has
  available).
* Fail the run if `max_delta` exceeds the tolerance constant.
* Publish the delta table as a build artifact for trend tracking.
