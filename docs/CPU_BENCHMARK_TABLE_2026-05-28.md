# CPU Metric Benchmark Table — 2026-05-28

Full CPU wall-time + peak-heap table for the 6 CPU metrics across
5 sizes x every mode each metric offers x cold/warm. Supersedes the
stale `benchmarks/heaptrack/stats.tsv` (which had GAP strip stubs,
pre-0.9.4 butteraugli, and pre-Path-A cvvdp strip).

## Provenance

- **Date:** 2026-05-28
- **Git commit:** `884c2722ab102367a5634de0987caa2b4f99cde4`
- **Host:** lilith (AMD Ryzen 9 7950X 16-core, 32-thread; WSL2, ~49 GB usable)
- **Build:** `cargo build --release` (workspace profile: opt-level=3, thin-LTO, codegen-units=16). **NO `-C target-cpu=native`** — runtime SIMD dispatch is what users get (per CLAUDE.md).
- **Wall times:** zenbench 0.1.8 (interleaved round-robin, paired stats; NOT criterion, NOT the heaptrack-instrumented runtime). 16 rounds/cell at 512/1024, 14 at 2K, 12 at 12MP, 10 at 30MP; all 32 cells per size interleave in one group.
- **Peak heap:** heaptrack 1.3.0, PROCESS peak (`peak heap memory consumption`), NOT top-callstack (top-callstack misreports per task #130). One process per (metric, mode, size).
- **rustc:** 1.95.0
- **butteraugli:** 0.9.4 (via `[patch.crates-io]` local sibling). **fast-ssim2:** 0.8.1. **dssim-core:** 3.4.0. **cvvdp/iwssim/zensim:** local workspace/sibling crates at the commit above.
- **Synthetic input:** deterministic per-pixel pattern (ref) + fixed channel offset (dist), identical between the wall and heaptrack drivers.
- **Strip body height:** 512 rows for all strip modes (cvvdp/iwssim STRIP_H_BODY_DEFAULT; ssim2/butter take an explicit height).

### Commands

```bash
# Driver (real crate APIs for every (metric,mode)):
cargo build --release   # in benchmarks/heaptrack/drivers/cpu_profile
# Peak heap:
bash benchmarks/heaptrack/refresh_2026-05-28/run_heaptrack_sweep.sh \
     benchmarks/heaptrack/refresh_2026-05-28
# Wall (per size):
target/release/cpu-wall <512|1024|2K|12MP|30MP> benchmarks/cpu_wall_2026-05-28.tsv
# Assemble:
python3 benchmarks/heaptrack/refresh_2026-05-28/assemble_table.py
```

### cold vs warm

- `full` / `strip` are inherently **cold**: each call builds the reference from scratch. One row, `cold_or_warm=cold`, `wall_ms` = per-call wall.
- `warm_ref` / `warm_ref_strip` are **cached-ref**: two rows.
  - **cold** = first scored pair, including the one-time reference precompute.
  - **warm** = amortized per-call cost reusing the cached reference (zenbench loops the score call with the reference warmed once outside the loop). The cold/warm wall delta is the precompute one-time cost.
  - Peak heap: heaptrack measures the mode's full single-call execution once; that process peak is the **cold-path** peak and is listed on both rows (it is an upper bound on the warm steady-state working set).
- `dssim` strip / warm_ref_strip: **NOT_SUPPORTED** — dssim-core 3.4 has no strip walker (honest gap, not a stub).

## Headline — Full mode across metrics

> Note: the task brief mentions a "16 MP" headline, but the measured size grid uses 12 MP (4000x3000) and 30 MP (6000x5000) per the task's size list. No 16 MP cell was measured; nothing is extrapolated.

### 12 MP (4000x3000) — `full` mode

| metric | wall (ms) | peak heap |
|--------|----------:|----------:|
| cvvdp | NA | 2.62G |
| ssim2 | NA | 2.01G |
| dssim | NA | 2.60G |
| butter | NA | 2.37G |
| iwssim | NA | 1.77G |
| zensim | NA | 742.93M |

### 30 MP (6000x5000) — `full` mode

| metric | wall (ms) | peak heap |
|--------|----------:|----------:|
| cvvdp | NA | 6.54G |
| ssim2 | NA | 4.83G |
| dssim | NA | 6.02G |
| butter | NA | 5.82G |
| iwssim | NA | 4.42G |
| zensim | NA | 1.81G |

## Per-metric breakdown

### cvvdp

**512^2** (0.26 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 51.47 | 57.25M |
| strip | cold | 56.18 | 57.25M |
| warm_ref | cold | 47.30 | 49.38M |
| warm_ref | warm | 28.29 | 49.38M |
| warm_ref_strip | cold | 49.53 | 49.38M |
| warm_ref_strip | warm | 29.83 | 49.38M |

**1024^2** (1.05 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 213.82 | 228.72M |
| strip | cold | 253.62 | 228.72M |
| warm_ref | cold | 176.85 | 197.25M |
| warm_ref | warm | 113.92 | 197.25M |
| warm_ref_strip | cold | 242.98 | 197.25M |
| warm_ref_strip | warm | 152.93 | 197.25M |

**2K** (4.19 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 914.53M |
| strip | cold | NA | 914.53M |
| warm_ref | cold | NA | 788.67M |
| warm_ref | warm | NA | 788.67M |
| warm_ref_strip | cold | NA | 788.67M |
| warm_ref_strip | warm | NA | 788.67M |

**12 MP** (12.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 2.62G |
| strip | cold | NA | 2.62G |
| warm_ref | cold | NA | 2.26G |
| warm_ref | warm | NA | 2.26G |
| warm_ref_strip | cold | NA | 2.26G |
| warm_ref_strip | warm | NA | 2.26G |

**30 MP** (30.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 6.54G |
| strip | cold | NA | 6.54G |
| warm_ref | cold | NA | 5.64G |
| warm_ref | warm | NA | 5.64G |
| warm_ref_strip | cold | NA | 5.64G |
| warm_ref_strip | warm | NA | 5.64G |

### ssim2

**512^2** (0.26 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 24.90 | 41.51M |
| strip | cold | 31.33 | 47.80M |
| warm_ref | cold | 22.16 | 38.36M |
| warm_ref | warm | 14.75 | 38.36M |
| warm_ref_strip | cold | 34.95 | 50.94M |
| warm_ref_strip | warm | 23.37 | 50.94M |

**1024^2** (1.05 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 92.35 | 165.77M |
| strip | cold | 112.21 | 126.19M |
| warm_ref | cold | 85.91 | 153.18M |
| warm_ref | warm | 56.96 | 153.18M |
| warm_ref_strip | cold | 122.90 | 148.99M |
| warm_ref_strip | warm | 93.87 | 148.99M |

**2K** (4.19 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 662.82M |
| strip | cold | NA | 345.11M |
| warm_ref | cold | NA | 612.44M |
| warm_ref | warm | NA | 612.44M |
| warm_ref_strip | cold | NA | 461.45M |
| warm_ref_strip | warm | NA | 461.45M |

**12 MP** (12.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 2.01G |
| strip | cold | NA | 902.86M |
| warm_ref | cold | NA | 1.81G |
| warm_ref | warm | NA | 1.81G |
| warm_ref_strip | cold | NA | 1.21G |
| warm_ref_strip | warm | NA | 1.21G |

**30 MP** (30.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 4.83G |
| strip | cold | NA | 1.63G |
| warm_ref | cold | NA | 4.42G |
| warm_ref | warm | NA | 4.42G |
| warm_ref_strip | cold | NA | 2.62G |
| warm_ref_strip | warm | NA | 2.62G |

### dssim

**512^2** (0.26 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 38.95 | 50.93M |
| warm_ref | cold | 39.42 | 50.93M |
| warm_ref | warm | 29.85 | 50.93M |
| strip | - | NOT_SUPPORTED | NOT_SUPPORTED |
| warm_ref_strip | - | NOT_SUPPORTED | NOT_SUPPORTED |

**1024^2** (1.05 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 166.73 | 203.48M |
| warm_ref | cold | 169.76 | 203.48M |
| warm_ref | warm | 116.91 | 203.48M |
| strip | - | NOT_SUPPORTED | NOT_SUPPORTED |
| warm_ref_strip | - | NOT_SUPPORTED | NOT_SUPPORTED |

**2K** (4.19 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 813.68M |
| warm_ref | cold | NA | 813.68M |
| warm_ref | warm | NA | 813.68M |
| strip | - | NOT_SUPPORTED | NOT_SUPPORTED |
| warm_ref_strip | - | NOT_SUPPORTED | NOT_SUPPORTED |

**12 MP** (12.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 2.60G |
| warm_ref | cold | NA | 2.60G |
| warm_ref | warm | NA | 2.60G |
| strip | - | NOT_SUPPORTED | NOT_SUPPORTED |
| warm_ref_strip | - | NOT_SUPPORTED | NOT_SUPPORTED |

**30 MP** (30.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 6.02G |
| warm_ref | cold | NA | 6.02G |
| warm_ref | warm | NA | 6.02G |
| strip | - | NOT_SUPPORTED | NOT_SUPPORTED |
| warm_ref_strip | - | NOT_SUPPORTED | NOT_SUPPORTED |

### butter

**512^2** (0.26 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 16.48 | 50.02M |
| strip | cold | 16.33 | 48.95M |
| warm_ref | cold | 34.51 | 51.86M |
| warm_ref | warm | 19.54 | 51.86M |
| warm_ref_strip | cold | 29.62 | 68.08M |
| warm_ref_strip | warm | 15.88 | 68.08M |

**1024^2** (1.05 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 78.90 | 228.11M |
| strip | cold | 70.92 | 144.65M |
| warm_ref | cold | 135.22 | 205.69M |
| warm_ref | warm | 82.46 | 205.69M |
| warm_ref_strip | cold | 131.44 | 242.17M |
| warm_ref_strip | warm | 69.34 | 242.17M |

**2K** (4.19 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 814.61M |
| strip | cold | NA | 376.82M |
| warm_ref | cold | NA | 799.02M |
| warm_ref | warm | NA | 799.02M |
| warm_ref_strip | cold | NA | 742.72M |
| warm_ref_strip | warm | NA | 742.72M |

**12 MP** (12.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 2.37G |
| strip | cold | NA | 801.15M |
| warm_ref | cold | NA | 2.31G |
| warm_ref | warm | NA | 2.31G |
| warm_ref_strip | cold | NA | 1.93G |
| warm_ref_strip | warm | NA | 1.93G |

**30 MP** (30.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 5.82G |
| strip | cold | NA | 1.50G |
| warm_ref | cold | NA | 5.86G |
| warm_ref | warm | NA | 5.86G |
| warm_ref_strip | cold | NA | 4.22G |
| warm_ref_strip | warm | NA | 4.22G |

### iwssim

**512^2** (0.26 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 65.13 | 38.34M |
| strip | cold | 70.02 | 39.46M |
| warm_ref | cold | 67.13 | 38.34M |
| warm_ref | warm | 63.44 | 38.34M |
| warm_ref_strip | cold | 62.55 | 36.66M |
| warm_ref_strip | warm | 39.78 | 36.66M |

**1024^2** (1.05 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 342.70 | 153.83M |
| strip | cold | 355.45 | 110.62M |
| warm_ref | cold | 347.36 | 153.83M |
| warm_ref | warm | 312.20 | 153.83M |
| warm_ref_strip | cold | 308.22 | 103.64M |
| warm_ref_strip | warm | 199.58 | 103.64M |

**2K** (4.19 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 616.48M |
| strip | cold | NA | 316.09M |
| warm_ref | cold | NA | 616.48M |
| warm_ref | warm | NA | 616.48M |
| warm_ref_strip | cold | NA | 321.27M |
| warm_ref_strip | warm | NA | 321.27M |

**12 MP** (12.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 1.77G |
| strip | cold | NA | 701.25M |
| warm_ref | cold | NA | 1.77G |
| warm_ref | warm | NA | 1.77G |
| warm_ref_strip | cold | NA | 919.39M |
| warm_ref_strip | warm | NA | 919.39M |

**30 MP** (30.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 4.42G |
| strip | cold | NA | 1.32G |
| warm_ref | cold | NA | 4.42G |
| warm_ref | warm | NA | 4.42G |
| warm_ref_strip | cold | NA | 2.30G |
| warm_ref_strip | warm | NA | 2.30G |

### zensim

**512^2** (0.26 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 8.32 | 16.91M |
| strip | cold | 7.64 | 18.51M |
| warm_ref | cold | 8.00 | 17.71M |
| warm_ref | warm | 6.01 | 17.71M |
| warm_ref_strip | cold | 7.80 | 16.65M |
| warm_ref_strip | warm | 6.28 | 16.65M |

**1024^2** (1.05 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | 16.38 | 58.93M |
| strip | cold | 26.41 | 50.96M |
| warm_ref | cold | 17.62 | 60.33M |
| warm_ref | warm | 13.87 | 60.33M |
| warm_ref_strip | cold | 21.93 | 50.96M |
| warm_ref_strip | warm | 23.08 | 50.96M |

**2K** (4.19 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 263.82M |
| strip | cold | NA | 222.40M |
| warm_ref | cold | NA | 280.46M |
| warm_ref | warm | NA | 280.46M |
| warm_ref_strip | cold | NA | 222.40M |
| warm_ref_strip | warm | NA | 222.40M |

**12 MP** (12.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 742.93M |
| strip | cold | NA | 687.99M |
| warm_ref | cold | NA | 790.37M |
| warm_ref | warm | NA | 790.37M |
| warm_ref_strip | cold | NA | 687.99M |
| warm_ref_strip | warm | NA | 687.99M |

**30 MP** (30.00 MP)

| mode | cold/warm | wall (ms) | peak heap |
|------|-----------|----------:|----------:|
| full | cold | NA | 1.81G |
| strip | cold | NA | 1.94G |
| warm_ref | cold | NA | 1.93G |
| warm_ref | warm | NA | 1.93G |
| warm_ref_strip | cold | NA | 1.94G |
| warm_ref_strip | warm | NA | 1.94G |

---

Raw data: `benchmarks/cpu_metrics_full_table_2026-05-28.tsv` (unified), `benchmarks/cpu_wall_2026-05-28.tsv` (wall), `benchmarks/heaptrack/refresh_2026-05-28/heaptrack_peaks.tsv` (heap) + raw `.zst` traces in that dir.
