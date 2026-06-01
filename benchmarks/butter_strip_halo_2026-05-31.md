# butteraugli-gpu — score-safe Strip on HF content (task #158) — 2026-05-31

Root-cause + fix verification for the mode_wall finding that butter's
Strip score diverged ~8% from Full on an aggressive high-frequency
checkerboard (`benchmarks/mode_wall_2026-05-31.{csv,md}`). The other five
-gpu crates' strip modes were score-safe; this documents why butter's
wasn't and how it was fixed.

Hardware: water-cooled AMD Ryzen 9 7950X + NVIDIA RTX 5070 (12 GB), CUDA
13.2.1, driver 596.21, cubecl-cuda (dlopen). GPU shared with another
process (~7 GB resident during the run). Fix commits: `77020757`
(opaque→multires-strip + HALO_ROWS 40→80), test `strip_hf_checkerboard.rs`.

Content: the EXACT `structured_pair` from `mode_wall.rs` — smooth RGB
gradient reference + a period-8 (8×8 block) ±12 checkerboard perturbation
on the distorted image. This is the adversarial HF input that exposed the
bug; the smooth+moderate-HF `make_image` content the other strip tests use
does NOT reproduce it.

---

## Root cause (two independent defects, both in the multi-resolution path)

butteraugli-gpu is the only -gpu crate with a multi-resolution mode (CPU
butteraugli's default: full-res diffmap + a half-res sibling whose diffmap
is supersample-added before reduction). Both defects live there.

### 1. Umbrella mode mismatch (the ~8% headline)

`ButteraugliOpaque` (what `zenmetrics-api`'s `Metric::Butter` wraps) routed
`MemoryMode::Full` → `new_multires` (full-res + half-res) but
`MemoryMode::Strip` → `new_strip` (**single-resolution**), silently
dropping the half-res band. The half-res supersample raises the score, so
single-res Strip was ~8% *below* multires Full. **This is not a halo bug:**
single-res Strip == single-res Full bit-identically on the same
checkerboard.

| size | single-res whole | MULTIRES whole | Δ |
|---|---:|---:|---:|
| 512² | 14.914985 | 16.287182 | +9.2% |
| 1024² | 15.007467 | 16.317377 | +8.7% |

The mode_wall sweep reported butter `full=16.317377` (multires) vs
`strip=15.007467` (single-res) → the 8.0e-2 "divergence" is exactly the
dropped half-res band.

Fix: `ButteraugliOpaque` routes Strip/Auto→Strip → `new_multires_strip`
(cuda/wgpu/cpu + `build_from_client`).

### 2. Under-haloed half-res sibling (a residual ~7e-4 max-norm)

The multires-strip half-res sibling is built by 2× downsampling the
full-res strip slab, so it only sees `HALO_ROWS / 2` real halo rows. The
half-res blur cascade uses the SAME sigmas (σ=7.16 / 3.22 / 1.2 …) in
half-res *pixel* space and independently needs 34 halo rows. At
`HALO_ROWS = 40` the half-res side got only 20 < 34 → the single worst
boundary pixel drifted the max-norm `score` up to ~7e-4 rel at 512².

`HALO_ROWS = 80` gives the half-res side 40 ≥ 34 → exact. (The single-res
path only ever needed 34; the bump is borne by strip mode as extra halo
rows — body-256 slab 336→416, +24%.)

---

## Parity verification — multires Strip vs multires Whole (HF checkerboard)

`crates/butteraugli-gpu/examples/strip_hf_diag.rs`, `rel = |strip−whole| /
|whole|`. Both the max-norm `score` and the `pnorm_3` aggregate.

### Before the HALO bump (HALO_ROWS = 40) — multires strip drifts at 512²

512², mag=12, multires strip vs multires whole:

| body | nstrips | rel_score | rel_p3 |
|---:|---:|---:|---:|
| 64 | 8 | **1.13e-4** | 2.65e-5 |
| 96 | 6 | 0.00e0 | 1.81e-5 |
| 128 | 4 | **1.81e-4** | 1.05e-5 |
| 192 | 3 | 0.00e0 | 1.42e-5 |
| 256 | 2 | **6.90e-4** | 7.01e-6 |
| 384 | 2 | 0.00e0 | 8.02e-6 |
| 512 | 1 | 0.00e0 | 1.56e-7 |

The max-norm drift is erratic (not monotonic in nstrips) — it spikes only
when the single hottest pixel lands on an under-haloed half-res boundary
row. `pnorm_3` is immune (it averages). The single-res strip is
bit-identical here (0.00e0 at every body) — confirming the defect is
specific to the half-res sibling.

### After the fix (HALO_ROWS = 80) — bit-identical

512² and 1024², mag=12 (representative; full grid in `strip_hf_diag`):

| size | body | rel_score | rel_p3 |
|---:|---:|---:|---:|
| 512² | 64 | 0.00e0 | 1.56e-7 |
| 512² | 128 | 0.00e0 | 1.56e-7 |
| 512² | 256 | 0.00e0 | 1.56e-7 |
| 1024² | 64 | 0.00e0 | 0.00e0 |
| 1024² | 128 | 0.00e0 | 0.00e0 |
| 1024² | 256 | 0.00e0 | 0.00e0 |

### Umbrella opaque path — Full vs Strip (the EXACT surface the sweep measured)

`ButteraugliOpaque` `MemoryMode::Full` vs `MemoryMode::Strip`, `.value`
(max-norm), after the fix:

| size | body | opaque_full | opaque_strip | rel |
|---:|---:|---:|---:|---:|
| 512² | 64 | 16.287182 | 16.287182 | 0.00e0 |
| 512² | 256 | 16.287182 | 16.287182 | 0.00e0 |
| 1024² | 128 | 16.317377 | 16.317377 | 0.00e0 |
| 1024² | 256 | 16.317377 | 16.317377 | 0.00e0 |

Before the fix this was `full=16.317 strip=15.007 rel=8.0e-2`.

---

## Gate test (negative-controlled)

`crates/butteraugli-gpu/tests/strip_hf_checkerboard.rs` — 8 cases (5 typed
multires + 3 opaque), tol 1e-4 rel on both score and pnorm_3, on the period-8
±12 checkerboard. All green with the fix. Negative controls:

- Reverting the opaque Strip→multires fix: opaque cases fail at the measured
  **8.028e-2** (`full=16.317 strip=15.007`).
- Reverting `HALO_ROWS` 80→40: the 512² typed cases fail at **6.905e-4** and
  **1.132e-4** (the under-haloed half-res).

`strip_parity` (21/21) and `multires_strip` (11/11) stay green.

---

## Wall delta — multires Full vs the fixed multires Strip

See `benchmarks/butter_strip_wall_task158_2026-05-31.csv`. Measured with
zenbench (interleaved, paired) at the task's 256²/1024²/4096² grid, in both
ONE-OFF (construct+compute+drop in the timed region — what `score_pair`
pays) and WARM (pre-built instance reused) contexts. GPU was shared, so
some cells report fewer clean rounds + wider CV; the one-off deltas at
1024²/4096² are large enough (>2×) to be unambiguous.

(Table inserted from the CSV below.)
