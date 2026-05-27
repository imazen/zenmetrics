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
| [`cvvdp-cpu`](crates/cvvdp-cpu/) | ColorVideoVDP (still-image, CPU, JXL buttloop) | JOD 0–10 + per-pixel diffmap | [`pycvvdp`](https://github.com/gfxdisp/ColorVideoVDP) v0.5.4 |
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
