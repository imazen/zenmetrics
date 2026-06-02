# Public API surface audit — 2026-06-01

Tooling: `cargo public-api 0.50.2` (nightly rustdoc) for the surface export,
`cargo rustc -p <crate> --lib -- --force-warn unreachable_pub` for the
compiler-proven "this `pub` should be `pub(crate)`" list, and a per-symbol
cross-crate usage grep for the reachable-but-unused tier.

Full per-crate surface dumps: `/mnt/v/output/zenmetrics/api-surface/<crate>.txt`
(+ `_SUMMARY.tsv`, `_UNREACHABLE_PUB.tsv`, `<crate>.unreachable.txt.locs`).
Block storage, not committed (per-crate dumps are 4–2525 lines each).

## Surface sizes (cargo public-api --simplified, line count)

| crate | pub items | publish | in-workspace consumers |
|---|---:|---|---|
| butteraugli-gpu | 2525 | internal | zenmetrics-api |
| cvvdp-gpu | 1847 | internal | cvvdp-conformance, zenmetrics-api, zenmetrics-orchestrator, gpu-vram-profile |
| iwssim-gpu | 1330 | internal | zenmetrics-api |
| ssim2-gpu | 1093 | internal | zenmetrics-api |
| zensim-gpu | 980 | internal | zenmetrics-api |
| dssim-gpu | 750 | internal | zenmetrics-api |
| cvvdp | 586 | internal | cvvdp-conformance, cvvdp-gpu, cpu-profile |
| zenmetrics-api | 328 | internal | zen-metrics-cli, zenmetrics-orchestrator |
| zenstats | 284 | **publishable** | (none in-workspace — external consumer) |
| iwssim | 99 | internal | iwssim-gpu, cpu-profile |
| zen-job-core | 1460 | **publishable** | zen-jobctl/dash/worker, zen-ledger |
| zen-jobdash | 866 | **publishable** | (none — LEAF) |
| zenfleet-orchestrator | 457 | internal | zen-cloud-salad, zencloud-hetzner |
| zen-cloud-salad | 481 | internal | zencloud-hetzner |
| zencloud-hetzner | 376 | internal | (none — LEAF) |
| zen-cloud-core | 329 | internal | 6 cloud crates |
| zenmetrics-orchestrator | 299 | internal | zen-metrics-cli |
| (others) | <250 | mixed | — |

The 6 metric `-gpu` crates total ~8,500 pub items; their sole product consumer
(zenmetrics-api) uses ~18 each. The bulk is cube-macro-generated kernel
machinery (`pub mod kernels::*` → per-kernel `Kernel`/`KernelInfo` structs +
`new`/`define`/`id` + derived/auto-trait impls) and internal pipeline modules.

## Tier 0 — compiler-proven `unreachable_pub` (28 items, zero-risk)

`pub` items inside non-`pub` modules — already unreachable externally, so
`pub(crate)` is a zero-observable-change fix (no consumer, no integration test
can reach them). Force-warn `unreachable_pub` flags exactly these:

- **cvvdp (13):** `strip.rs` {STRIP_H_BODY_DEFAULT, is_valid_strip_h_body,
  accumulate_slab, finalize, mode_b_halo_at_level, mode_b_k_split,
  mode_b_strip_h_at_level}, `scratch.rs` {new_strip, new, ensure_band_ws,
  ensure_strip_band_ws}, `pyramid.rs` {WeberPyramid}, `diffmap.rs` {new}
- **zenmetrics-orchestrator (9):** `bench.rs:310`, `lib.rs` ×8 (struct fields)
- **zen-metrics-cli (4):** `metrics/{butteraugli,dssim,ssim2,zensim}.rs` top-level item
- **iwssim (2):** `eig.rs` {lambdas, c_u_inv_slice} (eig is a private `mod`)

zenstats, zen-job-core, and all 6 -gpu crates: **0** unreachable_pub — their
surface is all *reachable* items (the -gpu over-exposure is Tier 1, not this).

## Tier 1 — reachable-but-path-unused `pub mod` → `pub(crate) mod` (keep re-exports)

Crate-root `pub mod`s that **no external code reaches by path** (`ext_refs=0`).
Making them `pub(crate) mod` while keeping the crate-root `pub use mod::{...}`
re-exports hides the module path with zero API impact:

| crate | hide (ext_refs=0) | keep (path-used by consumers) |
|---|---|---|
| butteraugli-gpu | `opaque`, `pipeline_batch`, `strip` | memory_mode, pipeline, session |
| ssim2-gpu | `pipeline`, `pipeline_batch`, `skipmap` | memory_mode, opaque, session |
| dssim-gpu | `kernels`, `opaque`, `pipeline`, `pipeline_batch` | memory_mode, session |
| zensim-gpu | `opaque`, `weights` | kernels*, memory_mode, pipeline*, session |
| cvvdp-gpu | `heatmap`, `opaque`, `presets` | host_scalar, kernels, memory_mode, params, pipeline, session |
| iwssim-gpu | `eig`, `filters`, `kernels`, `opaque` | memory_mode, pipeline, session |

\* zensim-gpu `kernels`/`pipeline` show 1 external ref (a bench/test driver) — verify before hiding.

`dssim-gpu::kernels` and `iwssim-gpu::kernels` have **zero** external refs — hiding
them collapses the largest chunk of those crates' surface immediately.

## Tier 2 — `kernels` used only by the crate's own parity examples/tests

`butteraugli-gpu::kernels` (2 examples) and `ssim2-gpu::kernels` (4 examples)
are reached only by in-crate parity examples (GPU-kernel-vs-CPU-reference
checks). `cvvdp-gpu::kernels` is genuinely shared (113 refs, 10 consumers incl.
the cvvdp CPU crate + conformance) — **keep pub**.

Mechanism options for butteraugli/ssim2 kernels:
- **(a) `#[doc(hidden)] pub mod kernels`** — keeps examples compiling, drops it
  from rendered docs and (with `--omit doc-hidden`) the official surface. Zero
  refactor. Doesn't make it `pub(crate)`.
- **(b) `pub(crate) mod kernels` + convert the parity examples to `#[cfg(test)]`
  in-crate unit tests** — fully privatizes the kernel tree AND the parity checks
  start running under `cargo test`. ~6 example files to relocate per crate.

## Publishable-crate caveat

zenstats (0.1.0, external consumer), zen-job-core (1460 items), zen-jobdash,
zen-ledger, zen-jobctl are `publish`-able. zenstats is already minimal
(`unreachable_pub` = 0). zen-job-core's 1460-item surface is the largest
*publishable* over-exposure and worth a dedicated pass (separate from the
metric-crate work).

## What landed (2026-06-01)

| commit | change | surface |
|---|---|---|
| `2944fbb1` | Tier 0: 28 `unreachable_pub` → `pub(crate)` (cvvdp 13, orchestrator 9, cli 4, iwssim 2) + `iwssim-filter-codegen` emits `pub(crate)` consts (clears 7 generated) | — |
| `f0dc9bb8` | dssim-gpu: kernels/opaque/pipeline/pipeline_batch → `pub(crate)` | 750 → 257 |
| `d4a3e2fa` | iwssim/zensim/cvvdp/butteraugli/ssim2-gpu Tier 1+2 | see below |

Per-crate `cargo public-api --simplified --features cuda` line counts:

| crate | before | after |
|---|---:|---:|
| butteraugli-gpu | 2525 | 244 |
| cvvdp-gpu | 1847 | 209 |
| iwssim-gpu | 1330 | 268 |
| ssim2-gpu | 1093 | 296 |
| zensim-gpu | 980 | 304 |
| dssim-gpu | 750 | 257 |

The clean product API is byte-identical — only module *paths* and
unreachable items were demoted; every crate-root re-export
(`Backend` / `<Metric>Opaque` / `<Metric>Params` / `Score` /
`MemoryMode` + `memory_mode` + `session`) is unchanged.

## `#[doc(hidden)] pub` inventory — reachable internals that are NOT public API

These items stay `pub` (so a separate compilation unit — another crate,
or this crate's own integration tests / examples / benches — can reach
them by path) but are marked `#[doc(hidden)]`: they are workspace-internal
machinery, not a supported per-crate API. Use `zenmetrics_api` for the
supported surface.

### Added by this work

| item | kind | why it must stay reachable |
|---|---|---|
| `cvvdp_gpu::kernels` | cross-crate | the cvvdp **CPU** crate re-exports the scalar kernels (`crates/cvvdp/src/kernels/mod.rs`); `cvvdp-conformance` asserts against `csf`/`masking`/`color`/`pool`/`pyramid` constants |
| `cvvdp_gpu::pipeline` | cross-crate | the cvvdp CPU crate's strip walker (`crates/cvvdp/src/strip.rs`) calls into it |
| `cvvdp_gpu::host_scalar` | own harness | cvvdp-gpu's own parity benches/examples/tests (the CPU scalar reference) — no external crate |
| `zensim_gpu::kernels` | cross-crate | `cvvdp-gpu/src/kernels/color.rs` shares the scalar color reference |
| `zensim_gpu::STRIP_ALIGN` | own test (re-export) | `zensim-gpu/tests/memory_mode.rs` asserts `h_body` is a multiple of it (added `#[doc(hidden)] pub use pipeline::STRIP_ALIGN` so `pipeline` could go `pub(crate)`) |
| `iwssim_gpu::pipeline` | cross-crate | `zenmetrics-api/tests/dispatch.rs` (umbrella dispatch test) + iwssim-gpu's own `native_rgb_perf_probe` example |
| `butteraugli_gpu::pipeline` | cross-crate | `zen-metrics-cli/src/orchestrator_runner.rs` |
| `butteraugli_gpu::kernels` | own GPU parity examples | `examples/{blur,colors}_parity.rs` — execute kernels via a GPU runtime (built under `--all-targets`); not convertible to CI unit tests without a GPU-availability gate |
| `ssim2_gpu::kernels` | own GPU parity examples | `examples/{blur,blur_h_pass,srgb,xyb}_parity.rs` — same GPU-runtime reason |

### Pre-existing (unchanged by this work — listed for completeness)

| item | why |
|---|---|
| `{butteraugli,ssim2,dssim,zensim,cvvdp,iwssim}_gpu::session` | stream-bound `MetricSession` plumbing (issue #17), gated `cubecl-types`; reached by `zenmetrics-api`'s session layer |

### Kept fully `pub` (genuine per-crate API, not hidden)

`memory_mode` (holds `MemoryMode` + `reclaim_pooled_vram`, path-accessed
by the umbrella) in all six; `cvvdp_gpu::params` (holds `CvvdpParams` +
`DisplayGeometry`, the latter path-accessed by `zenmetrics-api`).

### Not done — deliberately

butteraugli/ssim2 `kernels` were left `#[doc(hidden)] pub` rather than
`pub(crate)` + example→unit-test conversion: their parity examples
*execute* GPU kernels and have never run under wgpu/Metal in CI, so
converting them to CI unit tests would introduce never-validated parity
tolerances (a real flaky-red-CI risk). `#[doc(hidden)]` removes them from
the documented API with zero risk. zen-job-core (1460-item publishable
surface) is a separate future pass.

## Justification of the remaining public API

After the reduction the metric `-gpu` crates still report 244–304
`cargo public-api` items. That count is **not** maintained surface — it
is dominated by compiler-generated impls. Breakdown of dssim-gpu's 257
(representative; the other five have the same shape):

| category | count | maintained? |
|---|---:|---|
| auto-trait impls (`Send`/`Sync`/`Unpin`/`Freeze`/`UnwindSafe`/`RefUnwindSafe`) | 84 | no — compiler-emitted, 6 per pub type |
| derive impls (`Clone`/`Debug`/`PartialEq`/`Eq`/`Default`/…) | 46 | no — `#[derive]`, scale with type count |
| `pub fn` (methods + module fns) | 69 | **yes** |
| `pub struct` / `pub enum` | 12 | **yes** |
| `pub const` / `pub mod` | 5 | **yes** |

So ~51% (130/257) is auto-trait + derive enumeration that costs nothing
to maintain and shrinks only if the *number of public types* shrinks.
The genuinely-maintained surface is the **~12 types + their methods +
the `memory_mode`/`session` module fns**, every one of which is load-bearing:

| kept-`pub` item | why it cannot be hidden |
|---|---|
| `Backend` {`Cpu`,`Cuda`,`Wgpu`} | the caller picks the compute backend; `zenmetrics-api` maps its own `Backend` onto this per crate. Removing it removes backend selection. |
| `<Metric>Opaque` + `new` / `new_with_memory_mode` / `compute_srgb_u8` / `*_cached_ref` / `dims` / `set_reference` … | THE scorer. The whole point of the `Opaque` shim is to hide the cubecl `Runtime` generic so consumers don't pin a cubecl version through their public types — it must stay public to do that job. Used by `zenmetrics-api` for every metric. |
| `<Metric>Params` (+ `default`/`DEFAULT`, public fields) | scoring configuration the caller tunes; passed into `new`. |
| `Score` (+ `value` / `metric_name` / `metric_version`) | the return type of `compute_*`. |
| `MemoryMode` {`Auto`,`Full`,`Strip`,…} + `ResolvedMode` | memory-vs-speed selector; `zenmetrics-api`'s `resolve_memory_mode` produces these and feeds them to `new_with_memory_mode`. |
| `memory_mode::{reclaim_pooled_vram, estimate_gpu_memory_bytes*, vram_cap_bytes}` | path-accessed by the umbrella's VRAM accounting + proactive pool reclaim. Not reachable via a crate-root re-export, so the module stays `pub`. |
| `session::{new_opaque_on_stream, cleanup_stream, stream_reserved_bytes}` | `#[doc(hidden)]` already — `MetricSession` stream plumbing (issue #17). |
| cvvdp-gpu `params::DisplayGeometry` (+ `STANDARD_4K`) | cvvdp needs display geometry; `zenmetrics-api` sets it by path. |

Nothing in the kept set is gratuitous: each is consumed either by
`zenmetrics-api` (the product umbrella) or is the irreducible
type/method a direct caller of the crate must name. The only lever to
shrink the *count* further is reducing the number of public types
(which would proportionally drop the 130 auto/derive impls) — e.g.
folding `ResolvedMode` into `MemoryMode` — but those are real types with
real consumers, so they stay.

## Cross-metric API alignment (2026-06-01)

The reduction exposed that the six metric `-gpu` crates shared only a
partial core; the reference-reuse API in particular used three different
vocabularies (`cached_reference` / `warm_ref` / a zensim mix). Unified on
one neutral **"reference"** vocabulary. Every `<M>Opaque` now exposes the
identical core:

| method | signature | all 6? |
|---|---|---|
| `new` | `(Backend, w, h, <M>Params) -> Result<Self>` | ✓ |
| `new_with_memory_mode` | `(Backend, w, h, <M>Params, MemoryMode) -> Result<Self>` | ✓ |
| `compute_srgb_u8` | `(&mut, ref, dis) -> Result<Score>` | ✓ |
| `compute_pixels` | `(&mut, PixelSlice, PixelSlice) -> Result<Score>` | ✓ |
| `dims` | `(&self) -> (u32, u32)` | ✓ |
| `set_reference_srgb_u8` | `(&mut, ref) -> Result<()>` | ✓ |
| `compute_with_reference_srgb_u8` | `(&mut, dis) -> Result<Score>` | ✓ |
| `has_reference` | `(&self) -> bool` | ✓ |
| `clear_reference` | `(&mut)` | 5/6 (cvvdp omits — its warm cache is overwrite-on-set, no separate clear) |

Renames applied (workspace-wide, callers + tests + umbrella adapters):
- `has_cached_reference` / `has_warm_reference` → `has_reference`
- `compute_with_cached_reference_srgb_u8` → `compute_with_reference_srgb_u8`
- cvvdp `warm_reference_srgb` → `set_reference_srgb_u8`; `compute_with_warm_ref_srgb(dis, None)` → `compute_with_reference_srgb_u8(dis)` (diffmap kept as `compute_with_reference_srgb_u8_with_diffmap`)
- zensim: `compute_with_cached_reference_score_srgb_u8` → `compute_with_reference_srgb_u8` (Score); the feature-returning `compute_with_reference_srgb_u8 -> Vec<f64>` → `compute_features_with_reference_srgb_u8`; `ZensimInner` extended with `has_reference`/`clear_reference`.

Legitimately **not** uniformized (domain-specific, kept):
- cvvdp: `*_with_diffmap`, `new_with_geometry*` + `DisplayGeometry`, `*_from_linear_planes`
- zensim: `compute_features_*` (learned-metric feature output), `*_from_linear_planes`, `ZensimFeatureRegime`
- batch types (`<M>Batch`), `Gpu<M>Result`, the typed `<M>` pipeline, ssim2 `Ssim2Mode`, iwssim `<M>Config`/`<M>Strategy`, and the strip-mode reference variants (`*_stripped`) — these vary by metric capability.

Also fixed a reduction miss: `zensim_gpu::pipeline` was demoted to
`pub(crate)` but the crate's own `strip_memory_demo` example reaches it
by path; restored to `#[doc(hidden)] pub` (own-example consumer, like
`kernels`). This had been failing the `--all-targets` Compile job since
the reduction landed.
