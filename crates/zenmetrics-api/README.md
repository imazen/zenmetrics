# zenmetrics-api

Umbrella crate that composes the six zen GPU image-quality metrics —
[`cvvdp-gpu`](../cvvdp-gpu/), [`butteraugli-gpu`](../butteraugli-gpu/),
[`ssim2-gpu`](../ssim2-gpu/), [`dssim-gpu`](../dssim-gpu/),
[`iwssim-gpu`](../iwssim-gpu/), and [`zensim-gpu`](../zensim-gpu/) —
behind a single enum-dispatched API. Pull in one dependency, pick a
metric via `MetricKind`, get a `Score` back.

Phase 3 of the zenmetrics API-uniformity rollout: Phase 2 landed
uniform `*Opaque` shims on every metric crate (so they all expose the
same `Backend` enum + `Score` struct shape); this umbrella unifies
those shims behind a single `Metric` enum and adds opt-in cubecl-
typed shared-context scaffolding for Phase 4.

## Quickstart

```rust,no_run
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};

let r = vec![128u8; 256 * 256 * 3];
let d = vec![100u8; 256 * 256 * 3];

// Score one pair with DSSIM.
let mut m = Metric::new(
    MetricKind::Dssim,
    Backend::Cuda,
    256,
    256,
    MetricParams::default_for(MetricKind::Dssim),
)?;
let score = m.compute_srgb_u8(&r, &d)?;
println!("{} = {:.6} (impl {})", score.metric_name, score.value, score.metric_version);

// Same shape for the other five metrics.
let mut s = Metric::new(
    MetricKind::Ssim2,
    Backend::Cuda,
    256, 256,
    MetricParams::default_for(MetricKind::Ssim2),
)?;
let ssim_score = s.compute_srgb_u8(&r, &d)?;
# Ok::<(), zenmetrics_api::Error>(())
```

Pass [`zenpixels::PixelSlice`] inputs via `Metric::compute_pixels` —
non-sRGB and strided buffers convert per-call.

## Three usage regimes

| Regime | Feature flags | What's available | When to use |
|---|---|---|---|
| Opaque (default) | `default` (= `all-metrics,cuda,pixels`) | `Metric`, `MetricKind`, `MetricParams`, `Score` — pure value types, no cubecl in your public API | Default for downstream apps. Cubecl version doesn't leak through your dependency graph; you can bump `zenmetrics-api` without forcing every crate that depends on you to recompile against a new cubecl. |
| Opaque + pixels | `default` | Adds `Metric::compute_pixels(PixelSlice, PixelSlice)` | Have `zenpixels::PixelBuffer` already (e.g. from a zen decoder) and want to skip the manual `to_srgb_rgb8` step. |
| cubecl-types (advanced) | `cubecl-types` (on top of any combination above) | Re-exports each metric crate's typed `<Metric><R: Runtime>` + exposes `MetricContext<R>` for sharing a runtime client across multiple metric instances | Advanced: building a metric scheduler that wants to keep one `ComputeClient<R>` alive across many `Metric::new` calls, or pinning to a specific cubecl version because you also call into cubecl directly. |

## Per-metric switches

By default every metric is enabled. Pass `--no-default-features
--features <list>` to compile out the ones you don't need:

```toml
# Only ssim2 and dssim, both via CUDA, no pixels integration.
zenmetrics-api = { version = "0.0.1", default-features = false, features = ["ssim2", "dssim", "cuda"] }
```

Per-metric switches:

- `cvvdp` — pulls in `cvvdp-gpu`
- `butter` — pulls in `butteraugli-gpu`
- `ssim2` — pulls in `ssim2-gpu`
- `dssim` — pulls in `dssim-gpu`
- `iwssim` — pulls in `iwssim-gpu`
- `zensim` — pulls in `zensim-gpu`
- `all-metrics` — alias for the six above

Backend switches (forwarded to every enabled metric crate):

- `cuda` (default), `wgpu`, `hip`, `cpu`

Other:

- `pixels` (default) — enables the `compute_pixels` paths
- `cubecl-types` — see "Advanced regime" below

A `MetricKind` whose Cargo feature is disabled in a given build
returns `Err(Error::MetricNotEnabled)` at `Metric::new`. A `Backend`
not enabled in this build returns `Err(Error::BackendNotEnabled)`. The
enums themselves stay exhaustive across builds so caller `match` arms
keep compiling.

## Advanced regime: cubecl-types + MetricContext

```rust,ignore
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams, MetricContext};

// `MetricContext` is only available with the `cubecl-types` feature.
// It bundles a cubecl `ComputeClient<R>` + image dims so a future
// scheduler can share GPU resources across multiple metric instances.
let client = cubecl::cuda::CudaRuntime::client(&Default::default());
let mut ctx = MetricContext::<cubecl::cuda::CudaRuntime>::new(client, 1024, 1024);

// Phase 3 scope: shared client + generation counter, no shared upload yet.
// Phase 4 will add `Metric::compute_handles(&ctx, pair_handles)` for
// upload-once / score-many across all 6 metrics.
let _handles = ctx.upload_pair(&ref_bytes, &dis_bytes);
```

### Phase 4 tracking note

The full upload-once optimisation (saves ~17 ms × N metrics on the
host-to-device transfer in batch mode) requires each metric crate to
add an internal `<Metric>::compute_handles(handle_r, handle_d)`
method that consumes pre-uploaded device buffers. Today every
metric's `compute` path takes `&[u8]` and uploads internally.

Phase 3 (this crate) ships:

- `MetricContext<R>` with the shared `ComputeClient<R>`, `(w, h)`, and
  a generation counter that bumps on every `upload_pair`.
- A stable surface (`MetricContext::upload_pair → PairHandles`) so
  caller code can be written today against the upload-once shape and
  pick up the perf win when Phase 4 lands without re-plumbing.

Phase 4 ships:

- Per-metric `Cvvdp<R>::compute_handles` etc.
- Their opaque-shim equivalents.
- `Metric::compute_handles(&MetricContext<R>, PairHandles) -> Score`
  here.

### cubecl version guard

`MetricContext` assumes every enabled metric crate resolves to the
same `cubecl` version. Cargo's workspace inheritance handles this
in-tree (every metric crate's Cargo.toml uses `cubecl = { workspace
= true }`, so they all pick up the workspace pin).

The umbrella's `build.rs` currently emits a `cargo:warning=` reminder
when `cubecl-types` is enabled rather than a hard version check. A
proper `cargo metadata`-driven cross-check is a Phase 4 follow-up
candidate. If you've added a `[patch.crates-io] cubecl = ...` entry
that drives one metric crate to a different cubecl version than the
others, the typed `<Metric><R>` types will be distinct in the
trait-bound sense and `MetricContext` won't compile against them
— that's the design.

## Per-metric quirks the umbrella papers over

| Metric | Quirk | Umbrella behaviour |
|---|---|---|
| cvvdp | `CvvdpParams` has no `Default` impl (it uses `PLACEHOLDER` as the conventional default) | `MetricParams::default_for(MetricKind::Cvvdp)` returns `CvvdpParams::PLACEHOLDER` |
| butteraugli / ssim2 / dssim / iwssim / zensim | All have either `Default` derive or a `DEFAULT` const | `MetricParams::default_for` calls the right one |
| `iwssim` | Returns NaN on truly identical inputs (per-scale information weighting collapses in the log domain) | Documented in tests; the umbrella surfaces the NaN as-is |
| `zensim` | Is a feature extractor, not a scalar metric — `Score::value` is NaN when no trained weights are wired into `ZensimParams::weights` | Surface as-is. Caller must opt in by setting `ZensimParams::weights` to a `zensim::profile::WEIGHTS_*` table |
| `cvvdp` | Only `Cuda` is fully supported for the `score` path — Wgpu and Cpu are accepted by the constructor but the kernels rely on `Atomic<f32>` reductions that those runtimes don't support | Documented per-metric; the constructor succeeds, kernels panic at first dispatch on unsupported backends |
| `butter` | Opaque score is the libjxl max-norm; the 3-norm aggregation is dropped (only available via the typed `Butteraugli<R>` surface) | Documented in `butteraugli-gpu::opaque` |
| All metric crates | Each ships its own non-exhaustive `Score` struct | Umbrella converts to a single `zenmetrics_api::Score` at the boundary |

## Tests

| Test | What it covers |
|---|---|
| `dispatch::dispatch_<metric>` (6 tests) | Construct via `Metric::new(kind, Cuda, 256, 256, default_params)`, score one pair, verify metric_name + value range. Uses identity inputs for everything except iwssim which also uses a non-identical pair (identity inputs degenerate IW-SSIM's information-weighting). |
| `dispatch::kind_roundtrip` | `Metric::new(kind, ...).kind() == kind` for all 6 metrics |
| `pixels_smoke::pixels_<metric>` (3 tests: cvvdp, ssim2, dssim) | Construct a `zenpixels::PixelSlice<sRGB-RGB8>`, score via `compute_pixels`, verify the dispatch lands the right metric |
| `lib.rs` doctest | The Quickstart example compiles |

Run with:

```bash
PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  cargo test --release -p zenmetrics-api --features cuda
```

All 11 tests pass on a CUDA-capable host (verified 2026-05-17 on
RTX 5070 + cubecl 0.10).

## What this crate is NOT

- **Not a benchmark harness.** Use `zen-metrics-cli sweep` or the
  `zenbench` integration in each metric crate for that.
- **Not a CPU-only metrics shim.** Every variant requires one of the
  GPU backends to be enabled at compile time. The `cpu` feature
  selects cubecl-cpu (which several metrics' kernels don't fully
  support — see the per-metric quirks table).
- **Not a place for per-metric pickers / picker training data.** Per
  CLAUDE.md's "no zenpredict in codec crates" rule, the umbrella
  stays a pure dispatch layer.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. See `LICENSE-AGPL3`
and `COMMERCIAL.md` at the workspace root.
