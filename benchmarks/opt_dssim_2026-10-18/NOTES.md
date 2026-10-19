# opt_dssim_2026-10-18 — overlap ref/dist image construction (rayon::join)

Branch: `opt/dssim-join-create`. Change: `compute_dssim` in
`crates/zenmetrics-api/src/cpu_dispatch.rs` builds the reference and distorted
`DssimImage`s under `rayon::join` instead of sequentially. Adds
`dep:rayon` to the `cpu-dssim` feature (rayon was already in the graph via
dssim-core's `threads` feature).

## Why this is the whole honest lever

The portcheck flag for dssim is a **timing-scope asymmetry**, not an
algorithmic gap:

- Canonical shim (`port_spotcheck_2026-10-18/dssim_canonical.c`, lilith/dssim
  C API, rev eec33f5): constructs both `DssimImage`s ONCE before the timed
  loop; `SAMPLE` lines time `dssim_compare` alone.
- Ours (`iqa-compete` score path): every rep is a fresh `Metric` →
  `compute_srgb_u8` → `compute_dssim` = `create_image_rgb`×2 **inside** the
  timed region + `compare`. No session/cached-ref reuse exists in the driver.

Measured split at 1 MP (pinned dssim-core 331b56cf, same pair):

| | t1 | t32 |
|---|---|---|
| canonical compare-only (eec33f5 build) | 14.1 ms | 9.7 ms |
| our create×1 | ~9-11 ms | ~4.4-6.4 ms |
| our compare (pin 331b56cf) | ~11.7-12.1 ms | ~5.8-6.4 ms |
| our full call (before) | 28.2-33.3 ms | 13.9-14.7 ms |

At like-for-like scope our `compare` is already **faster** than canonical's
(11.7 vs 14.1 ms t1; the canonical binary uses the older eec33f5 branch —
the zenmetrics pin's streamed-moment compare is ~3x faster than that branch's).
The flagged gap is entirely the two image constructions the canonical shim
doesn't time.

`cache_for_reuse` (pin feature `cache_for_reuse(true)`) was measured and
rejected: it eagerly stores moments during create, so the one-shot full call
gets *worse* (1 MP t1: 44.8 vs 29.0 ms; t32: 36.2 vs 20.3 ms).

## Results (iqa-compete score/warm, A/B parent-vs-branch, same host)

Score bits identical on every cell (e.g. 1MP `3f2d8346e62e2000`).

- 64^2 t32: 5.10 -> 3.11 ms (−39%)
- 255^2 t32: 4.37 -> 3.88 ms (−11%)
- 1 MP t32: 13.9 -> 13.6 ms (−2.4%)
- 4K t32: 91.2 -> 81.8 ms (−10%)
- t1: neutral within noise (rayon::join is sequential on a 1-thread pool;
  255^2 2.19->1.63, 4K 308->274 include contention drift — micro-bench
  shows ±0 at t1)

## Residual flag status

Cannot reach 1.08x while the canonical timed scope excludes image
construction: our floor is create×2+compare ≈ 30 ms t1 vs canonical's
compare-only 14.1 ms. Even a zero-cost create would leave ~11.7/14.1 = 0.83x
(i.e. our compare alone is already below canonical). The flag at t1 is a
protocol artifact; recommend adjudicating the dssim cell as compare-only vs
create+compare scope, or timing canonical's create+compare like ssim2's
iqa-jxl shim does.

## Verify

- `cargo check`/`test -p zenmetrics-api --features cpu-metrics`: all dssim CPU
  tests pass (`dssim_cpu_is_finite_and_discriminates`, `cached_ref_dssim_*`,
  dispatch/pixels/hdr gates). The only failure is the pre-existing
  `cached_ref_dssim_strip_n_distortions_24mp` GPU test — CUDA OOM on this
  8 GB box, identical on parent 1358268a (environmental).
- `scripts/ci/lock.sh --check`/`--regen`: PASS; Cargo.lock gains only the
  rayon dep edge on zenmetrics-api.
