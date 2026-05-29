# zenmetrics

Multi-vendor GPU implementations of the perceptual image quality
metrics Imazen runs in production, plus a unified CLI.

Built on [CubeCL](https://github.com/tracel-ai/cubecl) — a single
`#[cube]`-annotated Rust kernel source dispatches across CUDA (NVIDIA),
WGPU (Vulkan / Metal / DX12 / WebGPU), HIP (AMD ROCm), and a
build-time CPU fallback.

## Crates

| Crate | Metric | Range / shape | Parity reference |
|---|---|---|---|
| [`butteraugli-gpu`](crates/butteraugli-gpu/) | Butteraugli | distance, max-norm + 3-norm | [`butteraugli`](https://crates.io/crates/butteraugli) v0.9 |
| [`ssim2-gpu`](crates/ssim2-gpu/) | SSIMULACRA2 | 0–100, higher better | [`ssimulacra2`](https://crates.io/crates/ssimulacra2) v0.5 |
| [`dssim-gpu`](crates/dssim-gpu/) | DSSIM | distance, 0 = identical | [`dssim-core`](https://crates.io/crates/dssim-core) v3.4 |
| [`zensim-gpu`](crates/zensim-gpu/) | zensim feature extractor | 228-feature vector + scalar score 0–100 | [`zensim`](https://github.com/imazen/zensim) v0.2.8 |
| [`cvvdp-gpu`](crates/cvvdp-gpu/) | ColorVideoVDP (still-image, GPU) | JOD 0–10, higher better | [`pycvvdp`](https://github.com/gfxdisp/ColorVideoVDP) v0.5.4 |
| [`cvvdp`](crates/cvvdp/) | ColorVideoVDP (still-image, CPU, JXL buttloop) | JOD 0–10 + per-pixel diffmap | [`pycvvdp`](https://github.com/gfxdisp/ColorVideoVDP) v0.5.4 |
| [`zen-metrics-cli`](crates/zen-metrics-cli/) | CLI front-end | — | uses the five metrics above |
| [`zenmetrics-corpus`](crates/zenmetrics-corpus/) | shared test images | — | (test infra) |
| [`zenmetrics-orchestrator`](crates/zenmetrics-orchestrator/) | Capability-aware scheduler + persistent benchmark cache + OOM fallback ladder | — | wraps the umbrella `zenmetrics-api` |

## Recommended entry point: `zenmetrics-orchestrator`

For any caller that scores **more than one (ref, dist) pair** —
sweeps, picker training, RD curves, batch comparison, anything with
multiple tasks — use [`zenmetrics-orchestrator`](crates/zenmetrics-orchestrator/).
It adds three things every previous in-tree caller had to hand-roll:

1. **Backend selection.** Persistent per-machine benchmark cache picks
   the fastest backend that fits available VRAM for each task. Knows
   which `(metric, size)` combinations OOM on this machine and avoids
   them on subsequent runs.
2. **OOM-safe fallback ladder.** `GpuFull → GpuStrip → (Cvvdp:
   GpuStripPair) → Cpu`. Each downgrade is recorded in the cache so the
   same machine never tries the failing combination twice.
3. **Cached-reference auto-detect.** xxhash3 hashes ref bytes per task,
   promotes consecutive same-ref tasks to the `set_reference` +
   `compute_with_cached_reference` API for the 1.5–3× speedup that
   sweeps benefit from.

**Quick decision table:**

| Caller shape | Use |
| --- | --- |
| One `(ref, dist)` per process, no fallback needed | `zenmetrics-api` directly |
| Batch / sweep / picker training / RD curve | **`zenmetrics-orchestrator`** |
| Streaming workload | **`zenmetrics-orchestrator`** |
| OOM-tolerant scoring | **`zenmetrics-orchestrator`** |
| One-ref / many-dist workloads | **`zenmetrics-orchestrator`** |

See [`crates/zenmetrics-orchestrator/README.md`](crates/zenmetrics-orchestrator/README.md)
for quickstart, the streaming + batch APIs, OOM handling details,
cached-ref semantics, CPU backend selection, capability cache lifecycle,
and the full configuration surface. Migration code samples in
[`crates/zenmetrics-orchestrator/docs/MIGRATION_FROM_API.md`](crates/zenmetrics-orchestrator/docs/MIGRATION_FROM_API.md).

The `zen-metrics` CLI routes scoring through the orchestrator by
default (since Phase 7.7.1, 2026-05-27). The legacy direct-dispatch
path remains available via `zen-metrics --use-legacy-scheduler …` (or
`ZENMETRICS_USE_LEGACY_SCHEDULER=1`) — useful when an archived parquet
sidecar needs bit-identical regeneration, or when comparing the two
paths for parity. The orchestrator path itself was validated as
bit-identical to legacy across all 54 cells (6 metrics × 3 sizes × 3
qs) on RTX 5070 + 7950X — see
[`benchmarks/orchestrator_parity_2026-05-27_phase771_run3.csv`](benchmarks/orchestrator_parity_2026-05-27_phase771_run3.csv)
for the per-cell data. The `--use-orchestrator` flag and
`ZENMETRICS_USE_ORCHESTRATOR` env var are accepted for
backwards-compat with pre-Phase-7.7.1 scripts / Docker images but
emit a deprecation warning.

The new sweep image
[`Dockerfile.sweep.v27`](Dockerfile.sweep.v27) bakes the orchestrator
features in and ships
[`scripts/sweep/onstart_orchestrator.sh`](scripts/sweep/onstart_orchestrator.sh)
as an entrypoint that drives the per-cell scoring through the
orchestrator's worker pool.

One per-metric carve-out remains: butteraugli stays on the legacy
direct-dispatch path because `ButteraugliOpaque::new_with_memory_mode`
resolves Auto to strip-mode (butter is strip-preferred), which drops
to single-resolution scoring and diverges from the legacy CLI's
always-multires output by ~14-30 %. The orchestrator transparently
falls back to legacy for butter; sweeps emit the same column shape
in both paths.

## SRCC sanity table

Spearman rank correlation coefficient against published still-image
MOS datasets (numbers from Cloudinary's SSIMULACRA2 benchmark, sign
normalized so higher = better):

| Metric | TID2013 | KADID-10k | CID22 |
|---|---|---|---|
| `dssim-gpu` (= DSSIM) | 0.871 | 0.856 | 0.872 |
| `ssim2-gpu` (= SSIMULACRA2) | 0.819 | 0.785 | 0.885 |
| `zensim-gpu` (= zensim) | (Imazen-internal benchmark) | | |
| `cvvdp-gpu` (= ColorVideoVDP) | (pending — reference is pycvvdp v0.5.4) | | |
| `butteraugli-gpu` (3-norm) | 0.664 | 0.543 | 0.794 |

## Memory modes

Every metric crate exposes a `MemoryMode` enum + `new_with_memory_mode`
constructor so callers can choose how the GPU working set is laid
out. The shape is uniform across the six metric crates:

```rust
pub enum MemoryMode {
    /// Pick Full or Strip based on a VRAM cap. Default.
    Auto,
    /// Allocate one working set for the whole image.
    Full,
    /// Allocate one working set for a strip of `h_body` body rows
    /// plus the crate's halo per side. `h_body == None` lets the
    /// resolver pick the largest body that fits the cap.
    Strip { h_body: Option<u32> },
    /// 2-D tile mode. Reserved — returns `Error::ModeUnsupported`.
    Tile { h: u32, w: u32 },
}
```

### Per-crate support matrix

| Crate | Strip available | Strip-preferred when Full fits | Auto picks Full when cap is generous |
|---|---|---|---|
| `butteraugli-gpu` | yes | **yes** (Strip is 1.9-4.9× faster) | no — Strip first |
| `dssim-gpu` | yes | no (Strip is 2-5× slower) | yes |
| `iwssim-gpu` | yes | no (Strip is ~1.7× slower; cached-ref strip path deferred) | yes |
| `ssim2-gpu` | no — `MemoryMode::Strip` → `Error::ModeUnsupported` | n/a | yes (always Full) |
| `zensim-gpu` | no — `MemoryMode::Strip` → `Error::ModeUnsupported` | n/a | yes (always Full) |
| `cvvdp-gpu` | no — architecturally blocked at 24 MP square | n/a | yes (always Full) |

### Auto policy

`MemoryMode::Auto` resolves by:

1. Reading `ZENMETRICS_VRAM_CAP_BYTES` (decimal usize). When unset,
   defaults to 8 GB.
2. Estimating the whole-image working-set bytes via the per-crate
   `estimate_gpu_memory_bytes` helper.
3. Picking Full when it fits AND the crate is not strip-preferred;
   else picking Strip with an auto-sized `h_body` that fits the cap.
4. Returning `Error::TooBigForFull { needed, cap }` when neither
   mode fits.

`butteraugli-gpu` is **strip-preferred** — Auto picks Strip even when
Full would fit, because the strip walker is the faster path on this
crate.

### Backwards compatibility

The historical `Metric::new(client, w, h, ...)` constructor is
preserved and now delegates through `new_with_memory_mode(..,
MemoryMode::Auto)`. Existing call sites compile and behave the same
unless `ZENMETRICS_VRAM_CAP_BYTES` is set tight enough to force a
mode change.

### Explicit override

To force a specific mode, use the per-crate
`new_with_memory_mode` (typed) or the opaque shim's same-named
constructor:

```rust
use butteraugli_gpu::{ButteraugliOpaque, ButteraugliParams, MemoryMode};

// Force whole-image even when the Auto cap would pick Strip.
let scorer = ButteraugliOpaque::new_with_memory_mode(
    backend,
    width,
    height,
    ButteraugliParams::default(),
    MemoryMode::Full,
)?;
```

For per-row body size control:

```rust
use dssim_gpu::{DssimOpaque, MemoryMode};

let scorer = DssimOpaque::new_with_memory_mode(
    backend, width, height, params,
    MemoryMode::Strip { h_body: Some(256) },
)?;
```

## Performance profile

GPU scoring cost splits into three components. Modelling a workload as

```
total ≈ process_start + Σ_refs(per_ref) + Σ_dists(per_dist)
```

is accurate because each piece is paid in a different scope and each was
measured separately:

- **`process_start`** — paid **once per process**: the CUDA context init
  (`Backend::client()`, a flat ~181 ms floor that is independent of metric
  and image size) plus the first-kernel PTX/JIT load for each metric the
  first time it runs. On the CPU backend this term is ≈ 0 (no device
  handshake — it starts computing immediately).
- **`per_ref`** — paid **once per distinct reference image** you cache via
  `set_reference` / `warm_reference`: the metric's reference-side
  precompute. For 5 of 6 metrics a new reference re-pays this cost; butter
  is the exception (its first reference is dear, subsequent ones reuse the
  buffers and are nearly free).
- **`per_dist`** — paid **once per scored distorted image** against a warm
  cached reference: `score_with_warm_ref(dist)`, the steady-state per-call
  wall.

The consequence is a ~181 ms one-time GPU floor (plus per-metric JIT). For a
**single small image on a freshly-launched process the CPU wins** — it has
no floor to amortize. As the image grows or the batch lengthens, the GPU's
throughput outruns the CPU even after paying the floor: for **batch / server
use (warm context, reference cached) the GPU is faster at every measured
size** (warm per-call is 10–100× below the CPU wall). The per-process floor
is paid once and shared across every metric and every pair scored in that
process — which is exactly why
[`zenmetrics-orchestrator`](crates/zenmetrics-orchestrator/) keeps one
long-lived warm worker. The full warmth-scope analysis (which transitions
re-pay which component, with the butter / iwssim allocation exceptions) is in
[`docs/GPU_INPROCESS_WARMTH_2026-05-29.md`](docs/GPU_INPROCESS_WARMTH_2026-05-29.md).

All numbers below are measured medians; no value is interpolated or
extrapolated. Sizes are 512² (0.262 MP), 1024² (1.049 MP), 2048² / "2K"
(4.194 MP), and 4096² / "16 MP" (16.777 MP).

### `process_start` — CUDA context + first-kernel JIT (once per process)

API: `Backend::client()` then the first `compute_*` on each metric.
Source: [`benchmarks/gpu_coldstart_2026-05-29.tsv`](benchmarks/gpu_coldstart_2026-05-29.tsv)
(`client_init_ms` / `first_compute_ms` / `cold_total_ms`, warm-disk,
7-process medians). Host: RTX 5070 (12 GiB), cuda backend, no
`-C target-cpu=native`.

`cold_total = client_init + metric_new + first_compute`. `client_init`
(the CUDA context) is the shared ~181 ms floor; the rest is per-metric and,
at large sizes, allocation-dominated.

| Metric | `client_init` (ms) | first-kernel JIT `first_compute` 512² (ms) | `cold_total` 512² (ms) | `cold_total` 16 MP (ms) |
|---|---|---|---|---|
| `butteraugli-gpu` | 166.8 | 286.7 | 498.7 | 4923.9 |
| `cvvdp-gpu` | 172.5 | 272.4 | 504.5 | 4282.7 |
| `ssim2-gpu` | 187.1 | 129.4 | 396.2 | 6740.5 |
| `dssim-gpu` | 185.0 | 136.5 | 376.1 | 3949.4 |
| `iwssim-gpu` | 182.5 | 265.1 | 491.4 | 2512.5 |
| `zensim-gpu` | 182.2 | 385.0 | 570.3 | 914.2 |

The `client_init` column is flat across metrics and sizes (measured range
166.8–191.2 ms over all 24 warm rows) — this is the once-per-process floor.
First-ever JIT on an empty PTX disk cache inflates `first_compute` further
(butter 1024² 303 → 1288 ms, +~1050 ms one-shot; zensim 1024² 382 → 506 ms,
+~175 ms — rows 26–27); the figures above are the warm-disk case (process
N>1 after the box has run any GPU job).

### `per_ref` — cache a reference once

API: `set_reference` / `warm_reference` /
`Ssimulacra2Reference::new` / `Zensim::precompute_reference` /
`ButteraugliReference::new`. Source:
[`benchmarks/gpu_inprocess_warmth_2026-05-29.tsv`](benchmarks/gpu_inprocess_warmth_2026-05-29.tsv)
Q3 rows (`setref1` = first reference on a warm instance, `setref2` = a
*different-pixel* second reference, each followed by `block_on(client.sync())`).
Host: RTX 5070, cuda, no `-C target-cpu=native`.

| Metric | first ref `setref1` 512² (ms) | new ref `setref2` 512² (ms) | first ref `setref1` 16 MP (ms) | new ref `setref2` 16 MP (ms) |
|---|---|---|---|---|
| `cvvdp-gpu` | 2.35 | 1.47 | 16.86 | 16.94 |
| `ssim2-gpu` | 2.54 | 2.26 | 28.45 | 29.36 |
| `dssim-gpu` | 2.16 | 2.02 | 19.91 | 20.31 |
| `iwssim-gpu` | 2.81 | 1.89 | 196.51 | 67.40 |
| `zensim-gpu` | 0.58 | 0.48 | 14.73 | 13.95 |
| `butteraugli-gpu` | **34.28** | **0.76** | **3990.41** | **21.64** |

For 5 of 6 metrics `setref2 ≈ setref1` — caching a new reference is **not**
free; budget one `per_ref` per distinct reference. **butter is the
exception**: its first reference eagerly allocates the full reference working
set (34 ms @512², 3990 ms @16 MP), but a subsequent new reference reuses
those buffers and costs only 0.76 ms @512² / 21.64 ms @16 MP — a 45× / 184×
drop. iwssim shows a milder version at 16 MP (196 → 67 ms).

### `per_dist` — warm per-call score against a cached reference

API: `score_with_warm_ref(dist)`. Source:
[`benchmarks/gpu_coldstart_2026-05-29.tsv`](benchmarks/gpu_coldstart_2026-05-29.tsv)
(`warm_per_call_ms`, intra-process warm repeats, every call ends in a host
readback so the wall is real GPU execution). Cross-confirmed by the
`warm_ref` cuda rows in
[`benchmarks/gpu_metrics_sweep_2026-05-28.tsv`](benchmarks/gpu_metrics_sweep_2026-05-28.tsv).
Host: RTX 5070, cuda, no `-C target-cpu=native`.

| Metric | 512² (ms) | 1024² (ms) | 2K / 4.2 MP (ms) | 16 MP (ms) |
|---|---|---|---|---|
| `butteraugli-gpu` | 1.54 | 3.61 | 12.93 | 50.20 |
| `cvvdp-gpu` | 4.23 | 6.00 | 11.80 | 41.33 |
| `ssim2-gpu` | 3.96 | 6.50 | 14.16 | 47.70 |
| `dssim-gpu` | 4.14 | 5.21 | 12.17 | 46.81 |
| `iwssim-gpu` | 6.53 | 9.47 | 12.78 | 39.44 |
| `zensim-gpu` | 1.66 | 3.27 | 9.67 | 37.80 |

So scoring a batch of N distorted images against one cached reference at
16 MP on cvvdp is `~504.5 + 16.86 + N×41.33 ms` (process_start512 floor is
size-independent; per_ref and per_dist scale with image size). The
`gpu_metrics_sweep` `warm_ref` cuda column gives the same per-call shape
measured by the independent sweep harness (e.g. cvvdp 4 MP 11.80 ms here vs
7.60 ms there, ssim2 16 MP 47.70 vs 43.98 — same order, different warm-up
counts).

### CPU full-mode wall (`score(ref, dist)`)

API: `score(ref, dist)` (umbrella `zenmetrics-api`, full mode — build +
one cold score per call). Source:
[`benchmarks/cpu_wall_all_metrics_2026-05-29.tsv`](benchmarks/cpu_wall_all_metrics_2026-05-29.tsv)
(`mode=full`, `cold_or_warm=cold`, `mean_ms`). Harness: zenbench 0.1.8
interleaved round-robin (paired stats, loop-overhead compensated — not
criterion). Host: AMD Ryzen 9 7950X, release, no `-C target-cpu=native`
(runtime archmage SIMD dispatch only).

| Metric | 512² (ms) | 1024² (ms) | 2K / 4.2 MP (ms) | 16 MP (ms) |
|---|---|---|---|---|
| `cvvdp` | 32.48 | 128.35 | 607.28 | 3812.26 |
| `ssim2` | 16.67 | 70.05 | 297.76 | 2591.03 |
| `dssim` | 30.53 | 123.48 | 546.16 | 4114.34 |
| `butter` | 12.69 | 62.69 | 347.53 | 1690.87 |
| `iwssim` | 59.81 | 261.88 | 1169.06 | 6665.18 |
| `zensim` | 6.92 | 13.92 | 78.86 | 369.66 |

### CPU vs GPU one-shot crossover

The size below which a **single** image on a **cold process** is faster on
CPU than GPU. `gpu_cold_total_ms` is the one-shot GPU floor (context-init +
metric_new + first_compute). Source:
[`benchmarks/cpu_gpu_crossover_2026-05-29.tsv`](benchmarks/cpu_gpu_crossover_2026-05-29.tsv)
+ [`docs/CPU_GPU_CROSSOVER_2026-05-29.md`](docs/CPU_GPU_CROSSOVER_2026-05-29.md).
Hosts: CPU 7950X, GPU RTX 5070, cuda, no `-C target-cpu=native`.

| Metric | one-shot: CPU wins up to | one-shot: GPU wins from | batch (warm) winner |
|---|---|---|---|
| `cvvdp` | 16.8 MP (all measured) | — | GPU at all sizes |
| `ssim2` | 16.8 MP (all measured) | — | GPU at all sizes |
| `butter` | 16.8 MP (all measured) | — | GPU at all sizes |
| `zensim` | 16.8 MP (all measured) | — | GPU at all sizes |
| `dssim` | 4.2 MP (2048²) | 16.8 MP (4096²) | GPU at all sizes |
| `iwssim` | 1.0 MP (1024²) | 4.2 MP (2048²) | GPU at all sizes |

Crossovers stated as a bracket between two measured sizes are interpolated,
never a fabricated MP. GPU-cold was measured only at 512² / 1024² / 2048² /
4096²; the 12 MP and 30 MP CPU rows in the source TSV have no GPU-cold
counterpart and are not given a one-shot winner. For **batch / warm** use
there is no crossover in range — GPU wins everywhere.

### Reproduce these numbers

One runner drives all four measurement harnesses:

```sh
# full grid (512² / 1024² / 2K / 16 MP) — matches the committed TSVs
scripts/perf/reproduce_perf_profile.sh

# quick smoke (512² + 16 MP only)
scripts/perf/reproduce_perf_profile.sh --quick
```

It invokes the existing drivers — no new measurement code:

- **`process_start` + `per_dist`** —
  [`scripts/memory_audit/sweep_gpu_coldstart_2026-05-29.py`](scripts/memory_audit/sweep_gpu_coldstart_2026-05-29.py)
  (builds each crate's `examples/coldstart_one`,
  e.g. [`crates/cvvdp-gpu/examples/coldstart_one.rs`](crates/cvvdp-gpu/examples/coldstart_one.rs)).
- **`per_ref`** —
  [`scripts/memory_audit/sweep_gpu_inprocess_warmth_2026-05-29.py`](scripts/memory_audit/sweep_gpu_inprocess_warmth_2026-05-29.py)
  (builds [`crates/zenmetrics-api/examples/inprocess_warmth.rs`](crates/zenmetrics-api/examples/inprocess_warmth.rs)).
- **CPU full wall** — the `cpu-wall` zenbench binary
  (`cargo build --release -p cpu-profile --bin cpu-wall`).

The GPU harnesses require a CUDA-capable host; the CPU wall runs anywhere.
Outputs land in a timestamped scratch dir and are diffed against the
committed TSVs. See the script header for per-harness flags.

## Documentation

- [`docs/CUBECL_PORTING_GUIDE.md`](docs/CUBECL_PORTING_GUIDE.md) — patterns
  for porting more CUDA / scalar metrics to multi-vendor CubeCL.
- [`docs/CUBECL_GOTCHAS.md`](docs/CUBECL_GOTCHAS.md) — 30-entry catalogue
  of cubecl-0.10-era traps with symptoms / fixes / examples.
- [`docs/SSIMULACRA2_PORTING_PLAN.md`](docs/SSIMULACRA2_PORTING_PLAN.md),
  [`docs/SSIM2_GPU_HANDOFF.md`](docs/SSIM2_GPU_HANDOFF.md) — the per-crate
  porting playbooks.
- [`crates/cvvdp-gpu/docs/PORT_STATUS.md`](crates/cvvdp-gpu/docs/PORT_STATUS.md)
  — ColorVideoVDP per-stage port status against pycvvdp v0.5.4
  (host scalar reference path + GPU composition + parity test
  matrix).
- [`scripts/sweep/cvvdp_backfill/README.md`](scripts/sweep/cvvdp_backfill/README.md)
  — operator runbook for the vast.ai pipeline that backfills cvvdp
  JOD scores onto the zensim training parquet store. Produces side-
  by-side `cvvdp_imazen_*` + `cvvdp_pycvvdp_v054` sidecars with a
  parity gate (`assert_parity.py`) that catches both threshold
  violations and silent-failure flatlines.

## License

Dual-licensed: AGPL-3.0-only (see [`LICENSE-AGPL3`](LICENSE-AGPL3)) or
Imazen commercial (see [`COMMERCIAL.md`](COMMERCIAL.md)). `dssim-gpu`'s
commercial track requires Kornel's upstream DSSIM licensing —
see `COMMERCIAL.md`, but this crate is neither maintained nor warrantied by him.
