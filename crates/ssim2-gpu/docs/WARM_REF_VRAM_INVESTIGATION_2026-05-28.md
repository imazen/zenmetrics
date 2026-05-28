# ssim2-gpu warm_ref VRAM investigation (task #138, 2026-05-28)

## TL;DR

The `gpu_metrics_sweep_2026-05-28.tsv` reading that ssim2-gpu
`warm_ref` peaks **higher** than `full` at 40 MP (10.71 GiB vs
9.23 GiB CUDA; 11.12 vs 10.51 wgpu) is **a measurement artifact, not
a real retention regression.** Whole-image `warm_ref` and `full`
share the *identical* per-instance device buffer set, so their
steady-state working set is the same byte-for-byte. The published
gap is the cubecl dynamic memory pool being sampled at different
points on its growth curve for the two modes (which have different
per-call wall times under the `WORKER_REPS=2` + 400 ms hold window).

No code change reduces `warm_ref` below `full`, because there is no
`warm_ref`-specific allocation to free. The brief's hypothesis
("retains transient per-scale gaussian/mu/sigma/blur scratch that
should be freed after ref precompute") describes the **cvvdp-gpu
Mode E** path (see `docs/GPU_METRICS_SWEEP_2026-05-28.md` lines
102-106), not ssim2-gpu's whole-image warm-ref path.

## Why warm_ref == full structurally

Both `full` and `warm_ref` construct the instance via `Ssim2::new`
(`pipeline.rs::new`), which pre-allocates every per-scale plane up
front in `Scale::new`: **57 planes per scale** (19 fields × 3
channels) — `ref_lin, dis_lin, ref_xyb, dis_xyb, sigma11_in,
sigma22_in, sigma12_in, v_scratch, t_scratch, sigma11_full,
sigma22_full, sigma12_full, mu1_full, mu2_full, ref_xyb_t,
dis_xyb_t, ssim, artifact, detail`. This matches
`memory_mode::estimate_gpu_memory_bytes`'s `PLANES_PER_SCALE = 57`.

- `full`: `compute` → `process_scale` runs ref + dist work
  interleaved per scale, reusing the shared `v_scratch`/`t_scratch`
  pair for all 5 blurs.
- `warm_ref`: `set_reference` runs the *ref-side* pipeline (2 blurs:
  sigma11 + mu1) writing into the **same pre-allocated** `sigma11_full`
  / `mu1_full` / `ref_xyb_t` slots, then `compute_with_reference`
  runs the *dist-side* pipeline (3 blurs) into `sigma22_full` /
  `mu2_full` / `sigma12_full`, reusing the **same** `v_scratch` /
  `t_scratch`.

Neither path performs any `client.create` of persistent device
buffers beyond the one transient sRGB staging upload per call (same
size for both). `set_reference` allocates nothing new; it only
populates pre-existing slots. There is no transient ref scratch to
free in whole-image mode.

(The whole-image `set_reference` is distinct from strip-mode
`set_reference_strip_mode`, which *does* allocate a
`StripCachedRefScale` cache + per-scale temp scratch — but that
temp scratch is already correctly dropped at scope end,
`pipeline.rs:1082`, and that path is `warm_ref_strip`, not
`warm_ref`.)

## Measurements (RTX 5070, CUDA, this WSL2 host)

Protocol: subprocess-per-cell, `nvidia-smi memory.used` polled
tightly during run, peak − baseline delta. Data in
`benchmarks/ssim2_warmref_trim_2026-05-28.tsv`.

### Pool-stabilized (WORKER_REPS=8) — the true steady-state

| size  | full delta | warm_ref delta | verdict |
|-------|-----------:|---------------:|---------|
| 16 MP | 6274 MiB   | 6274 MiB       | identical |
| 18 MP | 6273 MiB   | 6271 MiB       | identical (±2 MiB noise) |

### The artifact, reproduced (WORKER_REPS=2, published protocol), 16 MP

| trial | full delta | warm_ref delta |
|-------|-----------:|---------------:|
| 1     | 6274 MiB   | 6274 MiB       |
| 2     | **6332 MiB** | 6273 MiB     | ← full HIGHER (opposite of "regression") |
| 3     | 6273 MiB   | —              |

Which mode reports higher under `reps=2` is pure ±~60 MiB sampling
noise. The published 16 MP gap (warm_ref 6.19 vs full 6.15 GiB) sits
inside this band. The 40 MP gap is the same phenomenon amplified by
the steeper pool-growth ramp at that size.

### 40 MP (7680×5184)

With the desktop near-idle (baseline ~0.9-1.2 GiB), **both** `full`
and `warm_ref` grow the cubecl pool past ~11.9 GiB used and OOM
(`CUDA_ERROR_OUT_OF_MEMORY`) on the 12.2 GiB card — i.e. they hit the
**same** ceiling. The published 9.23 / 10.71 GiB figures were the
pool caught mid-growth, not the stable peak. `warm_ref_strip`
succeeds at **7504 MiB** delta (7.33 GiB) with score `-394.688187`
(bit-identical to `full`'s published `-394.680354`±strip-tol).

## 8-GiB-safety at 40 MP — what it would require

Neither `full` nor `warm_ref` is 8-GiB-safe at 40 MP (both exceed
the card and OOM here). The parity-safe memory-bounded mode is
**`warm_ref_strip`** (`new_strip` + `set_reference` + repeated
`compute_with_reference`): measured 7.33 GiB at 40 MP, score within
the 5e-5 strip-parity gate (`tests/strip_parity.rs`, `STRIP_REL_TOL`).

A path tighter than warm_ref_strip would need EITHER:
- plain `strip` (cold-ref, ~2.14 GiB at 40 MP per the sweep) — but
  its score diverges ~1.2e-3 relative from `full`, which is ~24× the
  5e-5 strip-parity gate. **GATE-BREAKING — do not route there.**
- widening the strip-parity gate at 40 MP — a **tolerance
  relaxation** that requires explicit user sign-off (per CLAUDE.md).
  Not done here.

So tight-card (≤8 GiB) 40 MP scoring should use `warm_ref_strip`.
`full`/`warm_ref` are 12-GiB-class modes and even there 40 MP is at
the edge of the pool ceiling.

## Score parity (warm_ref == full, before any change)

`warm_ref` and `full` produce **bit-identical** scores at every
measured size (e.g. 16 MP −382.580852, 18 MP −389.846324, 1 MP
−323.533831, 4 MP −366.493664). This is expected: identical kernels,
identical buffers, only the launch *ordering* differs (ref-then-dist
vs interleaved), and the IIR/error-map math is order-independent at
the reduction level.

## Test-suite note (host environment)

`tests/strip_parity.rs` run as a single concurrent process on this
WSL2 box surfaces `ServerUnhealthy` / `CallError` cascades: once one
test trips a transient cubecl runtime error (GPU memory pressure from
~30 large-image tests sharing one 12 GiB card with a live Windows
desktop), the cubecl client is poisoned and every later test in the
process fails. Run in isolated processes (`--exact <name>` per
invocation) all 15 parity tests pass (15/15). The failures are
environmental, not parity breaks.
