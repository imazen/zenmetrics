# Sweep `--jobs` throughput vs core count (2026-06-25, 7950X / 28 logical cores)

Q: optimal simultaneous jxl encodes? Measured: 12 mixed-size renditions (64×48→1024×768),
`zenmetrics sweep --plan lossy_dense --plan-budget 300 --q-grid 15,50,90 --metric zensim`,
`nice` (no run-heavy, so `--jobs` solely sets the rayon pool). %CPU = effective cores.

| --jobs | throughput (cells/s) | effective cores |
|---|---|---|
| 1  | 14.3 | 2.9 |
| 2  | 34.5 | 9.0 |
| 4  | 52.6 | 14.6 |
| 8  | 59.3 | 18.2 |
| 12 | 63.8 | 19.4 |
| 16 | 63.2 | 19.4 |
| 20 | 64.3 | 19.9 |
| 28 | 62.8 | 19.7 |

Findings:
- A SINGLE encode ≈ 2.9 effective cores on mixed content (size-dependent: tiny ~1 group/core,
  1024px ~8-12). So one small encode does NOT fill the box; parallel encodes help up front (1→4 = +3.7×).
- The sweep SATURATES at ~19-20 of 28 cores and won't exceed it at any --jobs (memory bandwidth +
  serial encoder portions). Throughput plateaus from --jobs 12; flat through 28 (no thrash regression —
  rayon oversubscribes gracefully — but load inflates with parked threads; load≈195 at jobs-8 was
  parked threads, ~18 cores actually working).
- **Optimal --jobs 8-12.** jobs-8 = 93% throughput at HALF the memory (31.8 GB high-q round); jobs-12 =
  peak (~48 GB). The binding constraint at high --jobs is RAM (per-encode working set × N), not speed.
- Rule: optimal_jobs ≈ min(cores / per_encode_cores, RAM / per_encode_peak). Large-heavy corpus →
  --jobs 4-6 (each encode uses more cores); tiny-heavy → --jobs 12+; mixed → 8-12.
