# cvvdp-gpu baseline on RTX 5070 (master @ 2610ae9c)

Date: 2026-05-16, commit `2610ae9c8b8a` (style: cargo fmt after cvvdp-gpu merge)

Bench command: `cargo bench -p cvvdp-gpu --bench score --features cuda --no-default-features -- '12mp'`

| Path | mean ms ±mad | ns/pixel | vs vship 2.40 ns/px |
|------|--------------|----------|---------------------|
| 12 MP cold (compute_dkl_jod) | 655.9 ±43.1 | 54.66 | 22.8× slower |
| 12 MP warm-ref | 351.7 ±22.6 | 29.31 | 12.2× slower |
| 1 MP cold | 30.5 ±7.3 | 29.07 | 12.1× slower |
| 1 MP warm-ref | 17.5 ±4.0 | 16.67 | 6.9× slower |
| 256² cold | 4.7 ±1.4 | 71.78 | 29.9× slower |
| 256² warm-ref | 2.6 ±0.8 | 39.67 | 16.5× slower |
| 256² d_bands | 6.1 ±1.3 | 93.08 | (extra readback) |
| 256² host_scalar | 46.0 ±3.2 | 701.83 | reference scalar |

Improvement target after T1.B (LDS-tiled downscale) + T1.C (LDS pool reduce):
- 12 MP warm-ref: 29.3 → ~10-15 ns/px (3× speedup, roughly pycvvdp parity)
- 12 MP cold: 54.7 → ~25-30 ns/px

After all-tier port: target ~3-5 ns/px (vship territory). Multi-month.
