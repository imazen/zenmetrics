# GMSD peak memory (heaptrack) — 2026-09-22

Instrument: `drivers/cpu_profile` (`cpu-profile <metric> <mode> <w> <h>`, one metric call
per process), `gmsd` arm added this day; matrix driver `gmsd_matrix.sh`. Raw rows:
[`gmsd_peak_2026-09-22.tsv`](gmsd_peak_2026-09-22.tsv) (heaptrack's "peak heap memory
consumption" and "peak RSS (including heaptrack overhead)"). Build: `cpu-profile` release at
zenmetrics `9ecfb9d4` + this driver arm (sha256 `4e1cc19c…`), gmsd `parallel` feature;
`RAYON_NUM_THREADS` = the threads column. Peers run in the same session, `full` mode, 8 threads.

The driver allocates the two sRGB8 inputs itself (2·w·h·3 bytes, the `inputs_bytes`
column); the metric's own share is `peak_heap − inputs`.

| size | GMSD `full` 1T | GMSD `full` 8T | GMSD `map` 1T | fast-ssim2 | butteraugli | zensim (latest) | IW-SSIM |
|---|---|---|---|---|---|---|---|
| 64² | 0.08 MB | 0.13 MB | 0.12 MB | — | — | — | — |
| 256² | 0.10 MB | 0.13 MB | 0.69 MB | — | — | — | — |
| 1 MP | 0.13 MB | 0.35 MB | 9.5 MB | 159 MB | 226 MB | 42 MB | 148 MB |
| 16 MP | 0.25 MB | 1.2 MB | 151 MB | 2.55 GB | 3.57 GB | 465 MB | 2.37 GB |
| 40 MP (7000×5728) | 0.34 MB | 1.9 MB | 361 MB | 6.74 GB | 8.67 GB | 1.06 GB | 5.66 GB |

(metric share = peak heap − inputs; 1 MB = 10⁶ B as heaptrack prints; peers at 8 threads.)

- **`full`** = `gmsd_rgb8`: sRGB8 rows are converted to gray inside each 64-row band
  (`c59cb81e`), so the only allocations are per-band ring rows and scratch rows plus one
  `(f64, f64)` per half-resolution row. Its footprint is O(width × concurrent bands) + O(height),
  not O(pixels): 0.34 MB at 40 MP on one thread, 1.9 MB with eight bands in flight.
- **`map`** = the heaviest caller shape (two packed f32 gray planes from `rgb8_to_gray`, then
  `gmsd_with_map` writing the full quarter-size f32 map): 8 bytes/pixel for the planes + 1 byte/
  pixel for the map, measured 151 MB at 16 MP (= 134 MB planes + 16.8 MB map) — the caller's
  buffers, not the kernel's.
- Scores are identical across modes and thread counts in every cell (the `score_line` column).

Not measured: peers at 64²/256² and at 1 thread; strip / warm-reference modes (GMSD has
neither — its full mode already streams).

## Re-measured 2026-09-23 after the integer + AVX-512 path (`051b37af`, `8fc65d91`, `ce18b182`)

Raw rows: [`gmsd_peak_v2_2026-09-23.tsv`](gmsd_peak_v2_2026-09-23.tsv); same driver, `gmsd` built with
`avx512` (v4 tier live on this host). The fused path no longer keeps gray scratch rows at all
(sRGB8 → half-resolution in one integer pass):

| size | GMSD `full` 1T | GMSD `full` 8T | GMSD `map` 1T |
|---|---|---|---|
| 1 MP | 0.11 MB | 0.24 MB | 9.5 MB |
| 16 MP | 0.18 MB | 0.64 MB | 151 MB |
| 40 MP | 0.23 MB | 0.97 MB | 361 MB |

(Peers were not re-run; their rows above are unchanged code.)
