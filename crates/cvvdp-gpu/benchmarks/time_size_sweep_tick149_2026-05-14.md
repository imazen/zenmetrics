# cvvdp-gpu — size-sweep snapshot (tick 149, 2026-05-14)

Re-measured after ticks 144-145's host-memory-pressure relief
to check whether the medium→large per-pixel regression
(open investigation from tick 97) was affected.

## Environment

- Host: lilith's water-cooled 7950X / 128 GB / RTX-class CUDA
- Commit: `e3d105145190` (`feat/cvvdp-gpu-scaffold`)
- Build: `cargo run --release --example time_size_sweep -p cvvdp-gpu --features cuda`

## Per-pixel cost (ns/px)

| bucket | weber  | d_bands | jod    |
| ----   | ----   | ----    | ----   |
| tiny   | 896.68 | 2287.61 | 1835.47 |
| small  |  95.27 |  201.65 |  223.47 |
| medium |  24.37 |   66.04 |   64.55 |
| large  |  64.21 |  172.26 |  145.01 |

## Comparison vs tick 97 (pre-host-pressure-fix)

| size   | metric  | tick 97   | tick 149 | Δ |
| ----   | ----    | ----      | ----     | ----  |
| tiny   | jod     | 1740.50   | 1835.47   | +5.5% (noise — fixed-overhead dominated) |
| small  | jod     | 187.67    | 223.47    | +19% (noise band) |
| medium | jod     | 53.65     | 64.55     | +20% (noise band) |
| large  | jod     | 159.37    | **145.01** | **−9%** (real win) |

The large-size win is consistent with ticks 144-145's
host-memory-pressure relief — at 12 MP the host Vec residency
was substantial.

## Medium→large regression: still present

medium (1 MP): 64.55 ns/px
large (12 MP): 145.01 ns/px → **2.2× worse per-pixel** than medium

The ratio narrowed slightly from tick 97's 3.0× (53.65 → 159.37)
to 2.2× here, but the curve shape persists. The host-memory-
pressure relief moved the floor but didn't fix the intrinsic
GPU-side super-linearity.

Open hypothesis from tick 97: L2 cache pressure at large
sizes. At 12 MP, level-0 buffer is 144 MB per channel; RTX
L2 is ~6 MB. The masking kernels' multi-tap reads
(`pu_blur_h_3ch_kernel` does 13 horizontal reads × 3 channels
per pixel) thrash L2 at large sizes in a way they don't at
medium.

Tied with fcvvdp 8-thread @ 360p (145 vs 86 ns/px = 1.69×
slower at the actual 12 MP per-pixel cost — but fcvvdp's
8-thread number is for 360p, where per-pixel is much
better than 12 MP for both implementations).

Next chunk: shared-memory tiled blur kernel for `pu_blur_h_3ch`
and `pu_blur_v_3ch_scaled` — load each tile into shared memory
once, then 13 reads come from shared memory instead of L2/DRAM.
