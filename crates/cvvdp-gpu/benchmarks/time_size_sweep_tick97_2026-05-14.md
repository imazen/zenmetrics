# cvvdp-gpu — size-sweep snapshot (tick 97, 2026-05-14)

First measurement across all 4 size buckets after the tick-91–96
fusion chain. Shape captured by `examples/time_size_sweep.rs`.

## Environment

- Host: lilith's water-cooled 7950X / 128 GB / RTX-class CUDA
- Commit: just before this benchmarks commit (post `ddeeddde`)
- Build: `cargo run --release --example time_size_sweep -p cvvdp-gpu --features cuda`
- Synthetic 8/4/12-step shift pair, fresh `Cvvdp::new` per size
  bucket, 1 warm-up iter + 5 timed iters per size.

## Per-phase medians

| bucket | pixels      | weber       | d_bands      | jod          |
| ----   | ----        | ----        | ----         | ----         |
| tiny   | 4 096       | 3.95 ms     | 10.81 ms     | 7.13 ms      |
| small  | 65 536      | 5.57 ms     | 14.58 ms     | 12.30 ms     |
| medium | 1 048 576   | 20.05 ms    | 62.09 ms     | 56.26 ms     |
| large  | 12 000 000  | 728.8 ms    | 1961.0 ms    | 1912.4 ms    |

## Per-pixel cost (ns/px)

| bucket | weber  | d_bands | jod    |
| ----   | ----   | ----    | ----   |
| tiny   | 964.78 | 2639.23 | 1740.50 |
| small  |  85.01 |  222.42 |  187.67 |
| medium |  19.12 |   59.21 |   53.65 |
| large  |  60.73 |  163.41 |  159.37 |

## Observations

### Per-pixel cost is **not** monotonic in image size

The minimum per-pixel cost is at **medium (1 MP)**, not large
(12 MP). Going small→medium per-pixel cost drops by ~3.5× as
expected (launch overhead amortizes). But going medium→large the
per-pixel cost **rises** ~3×.

That medium→large regression is the more interesting finding.
Hypotheses (not yet investigated):
- Memory bandwidth saturation — 12 MP D-band readback alone is
  ~432 MB; the d_bands path hits PCIe / GPU DRAM ceilings the
  medium path doesn't.
- Per-band scratch allocation — `d_scratch` for 12 MP holds
  multiple full-image f32 buffers (~144 MB each); cache eviction
  could be material.
- Kernel occupancy — large-image grids might saturate SMs in a
  pattern that hurts the masking-chain serialization.

The "weber" phase shows the same medium→large rise (19 → 61
ns/px). Since the weber pyramid does no readback, that rules out
the readback-bandwidth hypothesis as the *sole* cause.

The medium→large per-pixel regression is the next perf chunk to
investigate; tick 97 just surfaces it.

### Tiny-size fixed cost ≈ 7 ms

A single `compute_dkl_jod` call at 64×64 takes ~7 ms. That's
the launch-overhead floor — the cost any call pays regardless
of image content. With ~14 kernel launches per non-baseband
level × 7 non-baseband levels at 64×64 ≈ 100 launches, at
~70 µs each that's plausible.

### The α / β linear fit is **misleading** here

The 4-point OLS fit prints negative α intercepts (−12 ms /
−30 ms / −33 ms). That's a numerical artifact — the large
datapoint dominates, and the relationship isn't a clean
straight line across 4 orders of magnitude in size. Don't use
the fit numbers; the per-pixel table tells the real story.

A meaningful α fit would need either (a) a denser sweep
(20+ log-spaced sizes per CLAUDE.md training-data rules) or
(b) restricting the fit to the linear-regime range
(small + medium only).

## Next chunks

- Investigate the medium→large per-pixel regression. Start with
  `CVVDP_TRACE=1` + larger images to see which kernel grows
  super-linearly.
- Add a denser sweep (`time_size_dense.rs`) if the medium→large
  hypothesis turns out to need careful α + β fits per phase.
- The tiny / small launch-count win from the tick-91–93 fusions
  is plausibly already-realized in these numbers but unverifiable
  without a pre-tick-91 baseline. Documenting the post-fuse
  baseline here so a future regression check has anchor numbers.
