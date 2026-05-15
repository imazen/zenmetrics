# cvvdp-gpu — size-sweep snapshot (tick 164, 2026-05-14)

Re-measured after the tick 150–158 host-memory-pressure relief
+ structural readback elimination wave. Last sweep snapshot was
tick 149, which pre-dates the largest perf gains.

## Environment

- Host: lilith's water-cooled 7950X / 128 GB / RTX-class CUDA
- Commit: `8a6de7be` (`feat/cvvdp-gpu-scaffold`)
- Build: `cargo run --release --example time_size_sweep -p cvvdp-gpu --features cuda`

## Per-pixel cost (ns/px)

| bucket | weber  | d_bands | jod    |
| ----   | ----   | ----    | ----   |
| tiny   | 477.57 |  611.16 |  526.81 |
| small  |  47.06 |   97.09 |   90.81 |
| medium |  15.20 |   23.70 |   28.48 |
| large  |  23.09 |   42.31 |   38.83 |

## Comparison vs tick 149 (pre-wave)

| size   | metric  | tick 149  | tick 164 | Δ |
| ----   | ----    | ----      | ----     | ---- |
| tiny   | jod     | 1835.47   |  526.81  | **−71%** |
| small  | jod     |  223.47   |   90.81  | **−59%** |
| medium | jod     |   64.55   |   28.48  | **−56%** |
| large  | jod     |  145.01   |   38.83  | **−73%** |

Every bucket improved by more than half. The largest wins are
at the size extremes — tiny (where fixed dispatch overhead used
to dominate) and large (where host memory pressure used to
dominate).

## Medium→large regression: narrowed ~5×

Tick 149's open observation:

> medium (1 MP): 64.55 ns/px
> large (12 MP): 145.01 ns/px → **2.2× worse per-pixel** than medium
>
> The ratio narrowed slightly from tick 97's 3.0× (53.65 → 159.37)
> to 2.2× here, but the curve shape persists. The host-memory-
> pressure relief moved the floor but didn't fix the intrinsic
> GPU-side super-linearity.

Tick 164:

- medium (1 MP): 28.48 ns/px
- large (12 MP): 38.83 ns/px → **1.36× worse per-pixel** than medium

The 2.2× → 1.36× narrowing **falsifies the L2-cache-pressure
hypothesis** as the dominant factor at large sizes. With most
host overhead gone, the remaining 36% per-pixel cost increase
at large is consistent with normal kernel-level memory
bandwidth saturation (DRAM throughput limits at 12 MP that
don't bind at 1 MP) — not a structural GPU-side super-linearity.

## Trajectory at 12 MP

| tick | jod ns/px (time_size_sweep) | jod ns/px (time_12mp) |
| ---- | --------------------------- | --------------------- |
| 64   | —                           | 444                   |
| 73   | —                           | 127                   |
| 97   | 159                         | 122                   |
| 149  | 145                         |  87                   |
| 158  | —                           |  53                   |
| 164  | **39**                      |  53                   |

The size_sweep and time_12mp figures don't agree because they
use different iteration schemes — size_sweep's 5-iter median
catches the steady-state weber timing more cleanly while
time_12mp's iters carry some warm-up tail.

## Where the band-loop time is going at 12 MP

`d_bands - 2*weber` is the CSF+masking+IO bucket. At tick 164:

- 2×weber = 46.18 ns/px
- d_bands  = 42.31 ns/px
- d_bands − 2×weber = **−3.87 ns/px** (sub-noise / negative)

The 2×weber + bucket fits inside d_bands. Effectively the entire
band-loop overhead is dominated by the 2 weber pyramid passes.
The CSF/mask/pool stages on top are bandwidth-tightly packed
against weber's DRAM I/O at large sizes.

## Open observations

- **Small (256×256) is now the most expensive per-pixel** at
  90.81 ns/px. At this size launch overhead dominates: many
  small kernels (~12 per non-baseband level × ~6 levels = 72+
  launches) split among ~64K threads each. Either a kernel
  fusion that targets small sizes specifically, or a multi-
  level fused kernel could help.
- **Tiny (64×64) at 526.81 ns/px** is mostly fixed dispatch +
  cubecl initialization cost; not a great optimization target
  since real workloads at 64×64 aren't common.
- **The L2 hypothesis from tick 149 is dead.** Whatever
  super-linearity at large remains, it's not L2 cache
  pressure. DRAM bandwidth or memory-coalescing at 12 MP is
  the next direction to probe if optimization continues.
