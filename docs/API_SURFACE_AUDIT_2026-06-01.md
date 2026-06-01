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
