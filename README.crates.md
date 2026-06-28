<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenmetrics

zenmetrics is the Imazen workspace for perceptual image-quality metrics:
multi-vendor **GPU** implementations of the metrics we run in production, the
**CPU** reference crates they are checked against, a unified `zenmetrics` CLI, and
**zenfleet** — the content-addressed job system that drives codec sweeps and
metric backfills across a heterogeneous fleet. Pure Rust, `#![forbid(unsafe_code)]`.

The GPU metrics are built on CubeCL via the
[`zenforks-cubecl`](https://crates.io/crates/zenforks-cubecl) publication of
[tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) (0.10.x — carries
pinned-upload, PTX-cache-widening, and Metal `Atomic<f32>` capability patches for
our use case). A single `#[cube]`-annotated kernel source dispatches across CUDA
(NVIDIA), WGPU (Vulkan / Metal / DX12 / WebGPU), HIP (AMD ROCm), and a build-time
CPU fallback.

> Every crate in this workspace is `publish = false` — nothing ships to
> crates.io. Build the CLI and libraries from source (see Quick start), which is
> why the only badges above are CI and license.

## Quick start

The `zenmetrics` CLI scores one or many `(reference, distorted)` pairs on CPU or
GPU. Build it from the workspace:

```sh
git clone https://github.com/imazen/zenmetrics && cd zenmetrics
cargo build --release -p zenmetrics-cli       # binary: target/release/zenmetrics

# or install the binary directly
cargo install --git https://github.com/imazen/zenmetrics zenmetrics-cli
```

Score a single pair — CPU SSIMULACRA2, no GPU required:

```sh
zenmetrics score --metric ssim2 --reference ref.png --distorted out.jpg
```

Score one reference against several encoded variants across several metrics —
each unique image decoded once — as a TSV:

```sh
zenmetrics compare \
  --reference ref.png \
  --variant out-q60.jpg --variant out-q80.jpg --variant out.avif \
  --metric ssim2 --metric butteraugli --metric dssim \
  --output tsv
```

In the default build, `--metric` accepts the CPU metrics `ssim2`, `dssim`,
`butteraugli`, and `zensim`; `cvvdp` and `iwssim` need their CPU features
(`--features orchestrator,orchestrator-cpu-cvvdp` / `orchestrator-cpu-iwssim`),
and the GPU variants (`ssim2-gpu`, `dssim-gpu`, `butteraugli-gpu`, `iwssim-gpu`,
`zensim-gpu`, `cvvdp` via `gpu-cvvdp`) need `--features gpu-<metric>`. Run
`zenmetrics list-metrics` to print exactly what your build enabled and which
require a GPU. Other subcommands: `batch` (a TSV of pairs), `sweep` (drive a
codec across a quality × knob grid and score every variant into a Pareto TSV),
`score-pairs` / `assemble` (parquet sidecars + training corpora), `fleet-plan`
(size a sweep's fleet), and `jobexec` (the job-system executor — see below).

For scoring **many** pairs in one process (sweeps, picker training, RD curves),
call [`zenmetrics-orchestrator`](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-orchestrator/README.md)
rather than the CLI per pair. For scoring across a **fleet of machines**, use the
zenfleet job system. Both are covered below.

## Metric crates

Six GPU metric crates plus the two in-tree CPU reference crates the orchestrator's
CPU ladder dispatches to:

| Crate | Metric | Range / shape | Parity reference |
|---|---|---|---|
| [`butteraugli-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/butteraugli-gpu) | Butteraugli | distance, max-norm (default) + libjxl 3-norm | [`butteraugli`](https://crates.io/crates/butteraugli) 0.9.4 |
| [`ssim2-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/ssim2-gpu) | SSIMULACRA2 | 0–100, higher better | [`ssimulacra2`](https://crates.io/crates/ssimulacra2) 0.5 |
| [`dssim-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/dssim-gpu) | DSSIM | distance, 0 = identical | [`dssim-core`](https://crates.io/crates/dssim-core) 3.5 |
| [`iwssim-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/iwssim-gpu) | IW-SSIM (Wang & Li 2011) | `[0, 1]`, 1.0 = identical | [`iwssim`](https://github.com/imazen/zenmetrics/tree/master/crates/iwssim) (in-tree CPU port) |
| [`zensim-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/zensim-gpu) | zensim feature extractor | 228-feature vector + scalar score 0–100 | [`zensim`](https://github.com/imazen/zensim) 0.3.0 |
| [`cvvdp-gpu`](https://github.com/imazen/zenmetrics/tree/master/crates/cvvdp-gpu) | ColorVideoVDP (still-image, GPU) | JOD ~3–10, higher better | [`pycvvdp`](https://github.com/gfxdisp/ColorVideoVDP) 0.5.4 |
| [`iwssim`](https://github.com/imazen/zenmetrics/tree/master/crates/iwssim) | IW-SSIM (CPU reference + SIMD) | `[0, 1]`, 1.0 = identical | self (pure-Rust port) |
| [`cvvdp`](https://github.com/imazen/zenmetrics/tree/master/crates/cvvdp) | ColorVideoVDP (still-image, CPU) | JOD ~3–10 + per-pixel diffmap | [`pycvvdp`](https://github.com/gfxdisp/ColorVideoVDP) 0.5.4 |

The metric each GPU crate computes is bit-comparable to its cited reference. The
CPU side of each metric comes from an external reference crate
([`fast-ssim2`](https://crates.io/crates/fast-ssim2) 0.8.1,
[`dssim-core`](https://crates.io/crates/dssim-core) 3.5,
[`butteraugli`](https://crates.io/crates/butteraugli) 0.9.4,
[`zensim`](https://github.com/imazen/zensim) 0.3.0) or an in-tree crate
([`cvvdp`](https://github.com/imazen/zenmetrics/tree/master/crates/cvvdp),
[`iwssim`](https://github.com/imazen/zenmetrics/tree/master/crates/iwssim)).

**Feature gating (important):** the four external-crate CPU backends (ssim2 /
dssim / butteraugli / zensim) ship in the default `cpu-metrics` bundle, but the
two in-tree CPU ports — **`cvvdp` and `iwssim` are NOT in `cpu-metrics`.** Enable
them explicitly (`--features orchestrator,orchestrator-cpu-cvvdp`, resp.
`orchestrator-cpu-iwssim`). A build with neither `gpu-cvvdp` nor `cpu-cvvdp`
reports cvvdp as unavailable — that is a build-config message, not a "cvvdp is
GPU-only" limitation.

### Supporting crates

| Crate | Role |
|---|---|
| [`zenmetrics-api`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-api) | Umbrella: one `MetricKind` enum + one `Metric` type dispatching to every per-crate opaque scorer |
| [`zenmetrics-gpu-core`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-gpu-core) | Shared backend / score / sRGB / stream plumbing for the `*-gpu` crates (CubeCL) |
| [`zenmetrics-orchestrator`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-orchestrator) | Capability-aware backend chooser + persistent benchmark cache + OOM fallback ladder + warm worker pool |
| [`zenmetrics-cli`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-cli) | the `zenmetrics` CLI (`score` / `batch` / `compare` / `sweep` / `score-pairs` / `jobexec` / `assemble` / `fleet-plan`) |
| [`zenstats`](https://github.com/imazen/zenmetrics/tree/master/crates/zenstats) | Paper-correct IQA statistical panel (SROCC / PLCC / KROCC / OR / PWRC + bootstrap-CI A-vs-B) |
| [`zenmetrics-corpus`](https://github.com/imazen/zenmetrics/tree/master/crates/zenmetrics-corpus) / [`zenhdr-corpus`](https://github.com/imazen/zenmetrics/tree/master/crates/zenhdr-corpus) | Shared SDR / HDR test-image corpora (test infra) |
| [`cvvdp-conformance`](https://github.com/imazen/zenmetrics/tree/master/crates/cvvdp-conformance) | pycvvdp conformance fixtures + parity harness for the cvvdp crates |

## In-process scoring entry point: `zenmetrics-orchestrator`

For any caller that scores **more than one** `(ref, dist)` pair — sweeps, picker
training, RD curves, batch comparison — reach for
[`zenmetrics-orchestrator`](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-orchestrator/README.md)
instead of constructing metrics by hand. It adds three things every in-tree
caller used to hand-roll:

1. **Backend selection** — a persistent per-machine benchmark cache picks the
   fastest backend that fits available VRAM, and remembers which `(metric, size)`
   combinations OOM on this machine so it never retries them.
2. **OOM-safe fallback ladder** — `GpuFull → GpuStrip → (cvvdp: GpuStripPair) →
   Cpu`, each downgrade recorded in the cache.
3. **Cached-reference auto-detect** — hashes each task's reference bytes and
   promotes consecutive same-reference tasks to the warm-reference fast path for
   the 1.5–3× speedup sweeps benefit from.

The `zenmetrics` CLI routes scoring through the orchestrator by default. The
legacy direct-dispatch path stays available via `--use-legacy-scheduler` (or
`ZENMETRICS_USE_LEGACY_SCHEDULER=1`) for bit-identical regeneration of archived
parquet sidecars; butteraugli always flows through legacy because its `Auto`
resolves to strip-mode (single-resolution) and diverges from the legacy
always-multires output. The orchestrator path was validated bit-identical to
legacy across all 54 cells (6 metrics × 3 sizes × 3 qs) on RTX 5070 + 7950X. See
the [orchestrator README](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-orchestrator/README.md)
for the streaming + batch APIs, OOM handling, and cached-reference semantics.

## Distributed sweeps: the zenfleet job system

zenfleet is the canonical orchestrator for encode / score / sweep work that spans
many machines — the in-tree system that replaced hand-rolled chunk launchers. It
is content-addressed end to end:

- **Jobs are content-addressed.** A `JobId` is `sha256(kind + sorted inputs)`, so
  declaring the same work twice is a structural no-op.
- **The ledger is the truth, not the queue.** Every finished job writes a row to a
  columnar Parquet ledger in R2 (latest-wins on `(job_id, ts)`); coverage, the
  dashboard, and the reconciler all read the ledger, so a run converges after any
  partial pass or crash.
- **The queue is an R2 conditional-write lease** — a worker claims a job by
  `PutObject` with `If-None-Match: *` on `claims/<job_id>`, so exactly one worker
  wins each job and there is no double execution.
- **Workers are interchangeable and pull-based** (outbound HTTPS to R2 only), so a
  NAT'd basement box is a first-class tier alongside vast.ai / Hetzner / cloud.

Job kinds (`zenfleet_core::JobKind`): `Encode` · `Metric` · `Feature` ·
`Diffmap` · `Resample` · `Bake`, each carrying a resource class for capability
routing and a GC regenerability policy (expensive encodes are kept; cheap
re-scores are LRU-cached).

| Crate | Role |
|---|---|
| [`zenfleet-core`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-core) | Content-addressed job taxonomy, identity, status, blob addressing, and the idle / waste detector |
| [`zenfleet-ledger`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-ledger) | Columnar Parquet ledger + blob index with latest-wins compaction |
| [`zenfleet-ctl`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-ctl) | Declare desired jobs + query coverage / gap from the ledger |
| [`zenfleet-worker`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-worker) | Claims the reconciler's gap, runs a handler via the `ZEN_EXEC` executor, content-addresses outputs, emits ledger rows |
| [`zenfleet-dash`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-dash) | Railway-deployable dashboard + control API (reads the ledger; never runs workers) |
| [`zenfleet-sweep`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-sweep) | Cloud-agnostic sweep worker binary (selects a backend via `--backend`) |
| [`zenfleet-cloud`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-cloud) / [`-local`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-local) / [`-vastai`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-vastai) / [`-hetzner`](https://github.com/imazen/zenmetrics/tree/master/crates/zenfleet-hetzner) | Provider backends behind one common trait |

The thing that does the actual encode/score is `zenmetrics jobexec` — the
`ZEN_EXEC` reference executor: it reads one `DesiredJob` as JSON on stdin and
writes the output bytes (encode) or a JSON score row (metric) to stdout. Drive a
run with the one consolidated command —
[`scripts/jobsys/fleet`](https://github.com/imazen/zenmetrics/blob/master/scripts/jobsys/fleet)
`launch | status | watch | top | kill` (there is no other monitor; `fleet watch`
shows boxes, $/hr burn, per-box GPU/CPU util, idle / failed-to-start boxes, and
ledger progress in one place). Worker images bake every dependency at build time
([`scripts/jobsys/build_executor_image.sh`](https://github.com/imazen/zenmetrics/blob/master/scripts/jobsys/build_executor_image.sh)
copies a precompiled binary in; nothing is apt/pip-installed at boot). Full
runbook: [`docs/RUNNING_JOBS.md`](https://github.com/imazen/zenmetrics/blob/master/docs/RUNNING_JOBS.md);
sweep-plan contract: [`docs/PLAN_SWEEPS.md`](https://github.com/imazen/zenmetrics/blob/master/docs/PLAN_SWEEPS.md).


## License

Dual-licensed: AGPL-3.0-only (see [`LICENSE-AGPL3`](https://github.com/imazen/zenmetrics/blob/master/LICENSE-AGPL3))
or Imazen commercial (see [`COMMERCIAL.md`](https://github.com/imazen/zenmetrics/blob/master/COMMERCIAL.md)).
`dssim-gpu`'s commercial track requires Kornel's upstream DSSIM licensing — see
[`COMMERCIAL.md`](https://github.com/imazen/zenmetrics/blob/master/COMMERCIAL.md); this crate is
neither maintained nor warrantied by him.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · **zenmetrics** · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
