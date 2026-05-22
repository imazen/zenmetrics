# Strip processing in cvvdp-gpu

> **Status (2026-05-22): capped-depth Full mode shipped (cap=8 is the
> deepest level that fits the ≤ 0.005 JOD pycvvdp parity gate); true
> strip walker not yet.** The
> [`MemoryMode::Strip { capped_levels: Some(k) }`](../src/memory_mode.rs)
> constructor routes to a Full pipeline with the pyramid depth
> clamped at `k`. `Strip { capped_levels: None, .. }` and `Tile`
> still return `Error::ModeUnsupported`. `Auto` resolves to `Full`
> when it fits the VRAM cap and never auto-selects capped depth
> (capping changes the JOD value — caller must opt in explicitly).
> 73×91 odd-dim manifest parity holds at `|diff| = 0.0000 JOD`
> (tested 2026-05-22, CUDA backend).

## What's actually shipped vs not

| Path | Status | Constructor | Memory effect at 24 MP square |
|---|---|---|---|
| Full (uncapped) | shipped | `Cvvdp::new` | ~4.5 GB raw — fits 8 GB cap |
| Capped Full (`cap=8`) | shipped | `Strip { capped_levels: Some(8) }` | ~4.5 GB raw (same: cap drops only the smallest scratch) |
| Capped Full (`cap=6`) | shipped | `Strip { capped_levels: Some(6) }` | ~4.5 GB raw (same) |
| True strip walker | **not implemented** | n/a | would enable < 1 GB peak, but requires multi-pass design |
| Panorama strip walker (tall/wide) | **not implemented** | n/a | would enable ~4× memory savings at 1024×8192 |
| Tile (2-D) | **not implemented** | n/a | reserved |

**Important honest take**: the capped-levels feature shipped here does
**not by itself reduce memory in Full mode** — the cvvdp scratch
buffers are dominated by the finest-band sizes (`l_bkg_fine`,
`upscaled_c`, `d_scratch[0]`) which scale with the full image, not
with the number of pyramid levels. Capping the depth from 9 to 6
saves only the smallest few coarse-band scratch slots — kilobytes,
not gigabytes. The cap was implemented as a **prerequisite for a
future strip walker**: the σ=3 PU-blur halo at the dominant
non-baseband level scales as `6 × 2^(n_levels - 2)` rows per side,
and a strip walker is only useful when that halo is much smaller
than the body rows. Shipping capped-mode now means the strip-walker
patch lands later without changing the public API.

## Why cvvdp-gpu doesn't strip the way the other metrics do

zensim-gpu, dssim-gpu, butteraugli-gpu, iwssim-gpu, and ssim2-gpu all
implement `MemoryMode::Strip` by walking a fixed-halo window down the
image. Each strip is processed independently against the corresponding
reference strip; per-strip outputs are reduced into a global score.
Their pyramids are 3-5 levels deep with halo ≤ 64 rows, so strips of
~512 rows give clear memory savings.

cvvdp-gpu has two reasons the same template doesn't apply directly:

1. **Pyramid depth.** cvvdp's pyramid runs `min(log2(min(w, h)) - 1,
   MAX_LEVELS=9)` levels. At 4900×4900 (≈ 24 MP square) that's 9
   levels. The σ=3 phase-uncertainty Gaussian blur at non-baseband
   bands has kernel radius 6 in band coordinates. Mapped back to the
   finest level, the dominant non-baseband level contributes
   `6 × 2^(n_levels - 2)` rows of dependency per side — at `n=9`,
   that's `6 × 2^7 = 768` rows, so a **1536-row halo** for a 4900-row
   image. Strip body shrinks to ~3400 rows; the per-strip overhead
   (re-uploading the halo + rebuilding the partial pyramid chain)
   makes the strip walker slower than Full in addition to barely
   saving memory.

2. **Global statistics.** The Weber-contrast baseband finishing step
   (`baseband_divide_3ch_kernel` in `src/kernels/pyramid.rs`) scales
   the three coarsest planes by `inv_l_bkg_mean`, computed as the
   mean of `max(gauss_a, 0.01)` across the **entire coarsest level**.
   With a 9-level pyramid the coarsest level is ~10×10 pixels at
   24 MP — small enough that we already read it back to host and
   compute the mean there. Strip processing would either need to
   accumulate the mean across strips (cheap — one extra pass over the
   coarsest level data, which is tiny) or pre-compute the coarsest
   gauss_a in a first pass before the strip walker starts.

The first reason is the hard one; the second is straightforward
plumbing.

## How the cap shrinks halo (the math)

`pyramid_levels(ppd, w, h)` returns `n_levels = band_frequencies(...).len()`,
capped at `MAX_LEVELS = 9`. The dominant halo term per side is the
PU-blur radius at the deepest non-baseband level mapped back to level
0:

```
halo_per_side(n_levels) = 6 × 2^(n_levels - 2)
```

| n_levels | halo per side (rows at level 0) | 4900-row strip body |
|---|---|---|
| 9 | 768 | 3364 (×0.7 of full) |
| 8 | 384 | 4132 (×0.84) |
| 7 | 192 | 4516 (×0.92) |
| 6 | 96 | 4708 (×0.96) |
| 5 | 48 | 4804 (×0.98) |

The halo dominates the strip body at `n=9`; at `n=6`, a strip walker
becomes truly viable. **The cap is the lever that turns "strip" from
unusable-and-slower-than-Full into worth-implementing.**

## Capped-depth parity vs pycvvdp v0.5.4 (the fidelity tradeoff)

Capping CHANGES the JOD score. Sweep data at
`benchmarks/cvvdp_capped_levels_2026-05-22.csv` measures
`|jod_capped(k) − jod_pycvvdp_golden|` across 6 fixtures and
caps 5..=9. Worst-case per cap:

| cap | worst-case |diff| | fits ≤ 0.005 gate? |
|---|---|---|
| 9 (no cap) | 0.000012 (12 MP) | yes |
| 8 | 0.000585 (1280×720) | **yes** |
| 7 | 0.011690 (720×1280) | no |
| 6 | 0.013057 (720×1280) | no |
| 5 | 0.013108 (720×1280) | no |

**Cap=8 is the deepest cap that ships safely.** All 4 measured
fixtures with `natural_n_levels = 9` (synth_4000x3000,
synth_1024x1024, synth_1280x720, synth_720x1280) stay under 0.001 JOD
drift at cap=8. The 720×1280 fixture fails at cap=7 because the
720-pixel short axis already constrains `band_frequencies` at the
upper end of its detectable range — dropping one more band exposes
chrominance contrast that the un-capped pool would have averaged out.

The 73×91 odd-dim fixture has `natural_n = 6` so caps 7..9 are
no-ops there; cap=5 shifts it by 0.001 JOD which still fits the
gate. (Tested via `cap_5_drift_exceeds_gate_on_720x1280`-style
gates per fixture.)

**Test coverage**:

- `tests/capped_levels_parity.rs` — host-scalar parity, cap=8 vs
  pycvvdp ≤ 0.005 JOD across 5 fixtures.
- `tests/capped_levels_gpu_parity.rs` — GPU parity for the same.
- `tests/capped_levels_parity.rs::cap_7_drift_exceeds_gate_on_720x1280`
  — pins the known cap=7 failure so a future implementation that
  closes the drift can revisit the cap-7 ship decision.

## Decision matrix (updated)

| Image shape       | Path A (`Strip { capped_levels: Some(8) }`) | Path B (panorama strip) | Current   |
|-------------------|---------------------------------------------|-------------------------|-----------|
| 24 MP square      | **OK** (cap=8 saves nothing in Full mode but is forward-compatible with the strip walker design) | not yet impl | Full (4.5 GB raw) — fits 8 GB cap natively |
| 24 MP tall (1×24) | n/a (no benefit without strip walker) | required for < 1 GB | Unsupported |
| 12 MP square      | n/a (fits Full) | n/a | Full (2.3 GB raw) |
| 12 MP tall        | n/a | optional | Full (1.6 GB) |

**Honest blocker**: shipping a true strip walker requires the
multi-pass design sketched in "Path B" below, plus a separate
`MemoryMode` resolution policy for panorama detection. That work is
not in scope for this commit.

## Cap=8 is the deepest-level-that-fits — final finding

**Cap depth shipped: 8 (from natural depth 9 at STANDARD_4K
geometry).** The capped-levels sweep at
`benchmarks/cvvdp_capped_levels_2026-05-22.csv` measured |JOD drift|
vs pycvvdp v0.5.4 across caps 5..=9 on 6 fixtures. Worst-case per
cap, rounded for the table; full data in the CSV:

| cap | worst |diff| | meets ≤ 0.005 gate? | dominant failure |
|---|---|---|---|
| 9 (none) | 1.2e-5 (12 MP) | yes | n/a |
| **8** | **5.85e-4 (1280×720)** | **yes (shipped)** | n/a |
| 7 | 1.17e-2 (720×1280) | no | drops a band whose 720-pixel short axis is at the upper end of `band_frequencies`'s detectable range |
| 6 | 1.31e-2 (720×1280) | no | same failure deeper |
| 5 | 1.31e-2 (720×1280) | no | same |

**Cap=8 ships in production** via `MemoryMode::Strip { capped_levels:
Some(8) }`. Going deeper (cap ≤ 7) gives no further memory benefit
under Full mode (where cvvdp-gpu lives today — the coarsest band's
scratch is sub-megabyte) and breaks parity on the 720×1280 fixture,
so there's no production case for it. The cap-7 failure is pinned
by `tests/capped_levels_parity.rs::cap_7_drift_exceeds_gate_on_720x1280`
to surface any future change that closes that drift; if it does
close, the gate test re-opens the cap-7 ship decision.

Perf data at `benchmarks/cvvdp_capped_perf_2026-05-22.csv` confirms
the cap mechanism is **perf-neutral** in Full mode (cap=8 and
cap=6 match uncapped within run-to-run noise across 12 MP, 24 MP
square, and 1024×8192 panorama). The cap exists for the future strip
walker, not for today's Full pipeline.

## 73×91 odd-dim — closed (no further work)

The 73×91 odd-dim manifest parity is at `|diff| = 0.0000 JOD` per
`tests/pipeline_color.rs::compute_dkl_jod_matches_pycvvdp_at_73x91_odd`
and the warm-ref companion (verified 2026-05-22 on CUDA backend at
`d51a20d1`). The fix in `gausspyr_reduce` for mixed-parity reduce
levels (tick 206) carries over to the capped-levels code path
unchanged. No residual drift requires further attention.

## Path B — Panorama strip (the future work, not shipped)

**Idea**: for images where `min(w, h) << max(w, h)`, the constraining
axis already limits `natural_n_levels` via `band_frequencies`'s
`max_levels = floor(log2(min_dim)) - 1` cutoff. At 1024×8192 the
pyramid depth is bounded by `log2(1024) - 1 = 9` (same MAX_LEVELS),
but the halo expressed in rows of the **long axis** dominates only if
the strip is along the short axis.

If the strip is taken along the LONG axis (height ≫ width), the strip
body holds the full 1024 width and walks down the 8192-row direction.
Per-strip body = 1024 × strip_h pixels; total working set vs Full
shrinks proportionally.

This path keeps the pyramid depth and the JOD score intact — the
strip is invisible to the math because each strip's pyramid is built
from a copy of the full-width input rows. **The challenge**: the
weber baseband still wants global `inv_l_bkg_mean`. Two-pass
implementation:

1. Pass 1 — walk strips, accumulate per-strip
   `sum(max(gauss_a_strip[k_last], 0.01))` and `count`; combine to a
   single global `inv_l_bkg_mean` scalar.
2. Pass 2 — walk strips again, run the full color → weber → CSF →
   masking → pool chain per strip with the global scalar threaded
   through. Pool partials accumulate across strips into a single
   `n_levels × N_CHANNELS` Vec; host fold at the end.

**Expected memory savings** at 1024×8192:

- Full-mode raw (`estimate_gpu_memory_bytes`): 1589 MB (measured via
  `examples/cap_memory_estimate.rs`).
- 1024-row strip body + 192-row halo per side (at `cap=6`): strip
  `= 1024 × (1024 + 2×192) = 1024 × 1408` = 1.44 MP per strip.
- 8 strip iterations × ~360 MB peak ≈ ~360 MB peak (one strip live
  at a time).
- **~4.4× peak memory reduction** for the panorama case (pencil-math
  pending the panorama walker landing — there is no walker yet, so
  this isn't measurable end-to-end).

For comparison, **Full-mode panorama 1024×8192 wall-clock latency
runs at ~18 ms/iter steady-state** (3 iters, `benchmarks/cvvdp_capped_perf_2026-05-22.csv`),
identical between uncapped, cap=8, and cap=6 — the cap mechanism is
perf-neutral in Full mode as expected. The strip walker's value
proposition is memory, not throughput.

## References

- `src/memory_mode.rs` — `MemoryMode` enum + `capped_levels`
  variant + auto-resolver.
- `src/pipeline.rs::Cvvdp::new_with_geometry_and_cap` — cap-accepting
  constructor (the implementation `Strip { capped_levels: Some(_)
  }` routes to).
- `src/pipeline.rs::pyramid_levels` — natural-depth function the
  cap clamps against.
- `src/kernels/pyramid.rs::band_frequencies` — pycvvdp v0.5.4's
  cutoff at `MIN_FREQ = 0.2 cy/deg`; clipped further by
  `MAX_LEVELS = 9`.
- `src/kernels/masking.rs::pu_blur_h_kernel` — 13-tap σ=3
  Gaussian for non-baseband bands above `PU_PADSIZE = 6` pixels per
  axis. This is the halo-driver.
- `examples/capped_levels_sweep.rs` — host-scalar capped-depth parity
  sweep (CSV output at `benchmarks/cvvdp_capped_levels_2026-05-22.csv`).
- `examples/cap_memory_estimate.rs` — Full-mode memory at canonical
  sizes (data in the matrix above).
- `examples/cap_perf_compare.rs` — GPU wall-clock perf at 12 MP,
  24 MP square, and 1024×8192 panorama for uncapped / cap=8 / cap=6.
  CSV output at `benchmarks/cvvdp_capped_perf_2026-05-22.csv`.
- `tests/capped_levels_parity.rs` + `tests/capped_levels_gpu_parity.rs`
  — host + GPU parity gates.
- `crates/butteraugli-gpu/src/strip.rs` and
  `crates/dssim-gpu/src/strip.rs` — reference implementations of the
  strip-walker pattern in sibling crates (not directly portable to
  cvvdp without the global-statistic two-pass design).
