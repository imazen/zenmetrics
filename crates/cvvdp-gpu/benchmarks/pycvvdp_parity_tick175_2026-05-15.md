# cvvdp-gpu — 12 MP pycvvdp parity (tick 175, 2026-05-15)

Headline: **0.586 JOD drift → 0.0003 JOD drift** at 12 MP. Our
compute_dkl_jod now matches pycvvdp v0.5.4 within f32 precision
on the synthetic 4000×3000 pair that tick 173 first surfaced
the drift on.

## Fix

Two structural changes drove the 2000× accuracy improvement:

1. **Ceil-div pyramid halving** (`build_pyramid`,
   `build_weber_scratch`, `build_d_bands_scratch`, `pyramid_levels`).
   pycvvdp's `gausspyr_reduce` uses `(n + 1) // 2`; our allocators
   were using `n / 2` (floor-div). At even dimensions they
   agree; at odd dimensions (375 → 188 vs 187) they diverge by
   one pixel, and the mismatch compounds at every subsequent
   level. At 4000×3000 the cumulative effect on the JOD output
   was 0.586.

2. **MAX_LEVELS 8 → 9** (`src/lib.rs`). pycvvdp uses 9 bands at
   4000×3000 (the deepest is a 12×16 baseband). We were
   capping at 8. With the ceil-div fix in place, allowing the
   9th band closes the residual ~0.19 JOD drift to
   essentially zero.

Tick 174's failed attempt (MAX_LEVELS=10 alone, no ceil-div)
went the wrong direction — drift widened from 0.586 to 1.54 —
because the structural mismatch compounded across the new
extra level. Order matters: ceil-div first, then bump levels.

## Results

| variant                            | JOD output | drift vs pycvvdp |
| ----                               | ----       | ----             |
| pycvvdp v0.5.4 CUDA                | 9.4580     | 0 (reference)    |
| cvvdp-gpu tick 173 (broken)        | 8.8726     | 0.586            |
| cvvdp-gpu tick 174 (MAX_LEVELS=10) | 7.9200     | 1.538            |
| cvvdp-gpu tick 175 (ceil-div only) | 9.2692     | 0.189            |
| **cvvdp-gpu tick 175 (full fix)**  | **9.4583** | **0.0003**       |

## Perf trade-off (open issue)

| metric                          | tick 169 (floor) | tick 175 (ceil) | Δ      |
| ----                            | ----             | ----            | ----   |
| weber pyramid (1 side)          | 18.7 ns/px       | 41.3 ns/px      | +121%  |
| compute_dkl_d_bands             | 33.7 ns/px       | 66.7 ns/px      | +98%   |
| compute_dkl_jod (cold)          | 36.1 ns/px       | 61.8 ns/px      | +71%   |
| compute_dkl_jod_with_warm_ref   | 20.6 ns/px       | 33.8 ns/px      | +64%   |

The ceil-div pyramid shapes themselves only change by a few
pixels per level — the total pixel work is nearly identical.
The 2× weber regression is therefore NOT from extra compute.
Hypothesis: CUDA dispatch shapes (cube_count.div_ceil(64)) now
round to slightly different block counts at the deeper levels,
and the kernel paths through the boundary reflection branches
are taken more often with odd-dim source data. To investigate
in a follow-up tick.

## vs pycvvdp at 12 MP

| variant                            | per-pixel  | ratio          |
| ----                               | ----       | ----           |
| pycvvdp v0.5.4 CUDA                | 14 ns/px   | baseline       |
| cvvdp-gpu tick 175 cold            | 61.8 ns/px | **4.4× slower** |
| cvvdp-gpu tick 175 warm-ref        | 33.8 ns/px | **2.4× slower** |

The pre-fix tick 173 perf (cold 36, warm 21 ns/px) was actually
**wrong** — those numbers reflected the broken pyramid that
gave 0.586 JOD drift. The honest tick 175 numbers above are
slower but produce correct output.

Closing the perf gap is the next priority. Two paths:
- Diagnose the per-level dispatch regression (probably small
  fix, recoups a chunk).
- Replace hand-rolled downscale/upscale with cubek depthwise
  separable conv (matches pycvvdp's cuDNN advantage).

## Parity tests

All 67 existing parity tests still pass on CUDA. They run at
32×32 / 256×256 / 256×256-corpus where floor and ceil agree
exactly, so they didn't catch the drift in the first place.
A new 12 MP parity test driven by a pycvvdp golden is queued
for a follow-up tick — for now `examples/time_12mp` prints
the JOD output so the drift is visible in any run.
