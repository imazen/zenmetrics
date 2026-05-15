# cvvdp-gpu — 12 MP warm-ref benchmark (tick 171, 2026-05-14)

Measures the new `Cvvdp::warm_reference` + `compute_dkl_jod_with_warm_ref`
fast path (tick 170, `abe3599d`) against the cold path
`compute_dkl_jod(ref, dist, ppd)` on the same synthetic 4000×3000
pair. Each iter calls `warm_reference(&ref)` before the timed
warm-path call — so the measurement captures the per-DIST cost
after a one-time REF dispatch (realistic batch-scoring shape).

## Environment

- Host: lilith's water-cooled 7950X / 128 GB / RTX-class CUDA
- Build: `cargo run --release --example time_12mp -p cvvdp-gpu --features cuda`
- Synthetic 4000×3000 RGB8 pair

## Per-phase medians (5 iters)

| phase                          | wall      | per-pixel  |
| ------                         | ----      | ----       |
| weber pyramid (1 side)         | 224.3 ms  | 18.7 ns/px |
| `compute_dkl_d_bands` (full)   | 404.3 ms  | 33.7 ns/px |
| `compute_dkl_jod` (cold REF)   | 432.8 ms  | 36.1 ns/px |
| `compute_dkl_jod_with_warm_ref` | **247.3 ms** | **20.6 ns/px** |

**Per-DIST saving from warm REF: 185.5 ms = 42.9%**, i.e.
**1.75× faster per call** in the batch-scoring workflow.

## vs fcvvdp at 360p

| variant                          | per-pixel   | vs cvvdp-gpu cold | vs cvvdp-gpu warm |
| ----                             | ----        | ----              | ----              |
| fcvvdp 1-thread                  | 214 ns/px   | 5.93× slower      | 10.39× slower     |
| fcvvdp 8-thread                  |  86 ns/px   | 2.38× slower      | **4.17× slower**  |
| cvvdp-gpu (cold REF)             | 36.1 ns/px  | —                 | 1.75× slower      |
| cvvdp-gpu (warm REF, per DIST)   | **20.6 ns/px** | 1.75× faster   | —                 |

## What the warm path skips

In `_dispatch_d_bands_into_scratch` the cold path runs:

```text
REF: srgb_to_dkl → 21 downscales → 6 upscales/level × 7 + L_bkg expand × 7 + weber_compute × 7
DIST: same as REF
band loop (CSF + masking + pool input)
pool (24 → 8 atomic launches after tick 165)
```

The warm path skips the entire REF block — that's color +
gauss reduce + weber expand + subtract_weber_3ch + baseband_divide
all for the REF side. Roughly half the GPU compute per JOD.

The measurement (42.9% saving) is slightly below the naive 50%
because:
- The pool + host fold isn't halved (runs once per JOD regardless).
- Band loop dispatch overhead is the same.

## Use cases

- **Codec quality sweeps** — one REF compared against N quality
  levels × M codecs. Warm-ref turns this from `N×M× ~36 ns/px`
  into `~36 ns/px + N×M× ~21 ns/px`, i.e. 1.75× throughput once
  N×M is large enough to amortize the initial REF dispatch.
- **Fixture-based testing** — golden reference + many candidate
  encoders. Same shape as above.
- **Live-tuning UIs** — user adjusts a parameter, we score the
  result against a fixed reference. Each candidate evaluation
  is ~250 ms instead of ~435 ms at 12 MP.

## Trajectory

| tick | jod cold | jod warm | vs fcvvdp 8t (cold / warm) |
| ---- | -------- | -------- | --------------------------- |
| 64   | 444      | —        | 5.16× slower / —            |
| 156  | 60       | —        | 1.43× faster / —            |
| 158  | 53       | —        | 1.63× faster / —            |
| 166  | 42       | —        | 2.06× faster / —            |
| 169  | 38       | —        | 2.26× faster / —            |
| 171  | **36**   | **21**   | **2.38× / 4.17× faster**    |

## Open observations

- Cold-path jod 36.1 ns/px here is consistent with tick 169's
  38 ns/px (run-to-run variance ~5%). Still the load-bearing
  baseline number.
- Warm-path at 20.6 ns/px is ~2× weber per side, matching the
  prediction: the warm path is dominated by DIST weber (one
  side) + band loop + pool, where the band loop fits inside
  DIST weber's variance.
- **Next big lever**: shared-memory tiled downscale or
  upscale (genuinely-different memory access pattern, not just
  fusion). Current bottleneck is DRAM bandwidth in the weber
  pyramid.
