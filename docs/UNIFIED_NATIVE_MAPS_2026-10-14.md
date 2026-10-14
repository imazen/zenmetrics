# Native maps on the unified surface, 2026-10-14

These metrics now return native spatial outputs through
`Metric::evaluate(.., EvalOpts::with_native_map())` →
`Evaluation::map()` / `Evaluation::maps()`. Each map is the metric's own
intermediate, at its own resolution, with polarity and `MapGeometry`. None is
a scalar broadcast or another metric's map. Where a metric has no per-pixel
headline map, the maps are labeled as the components or contributions they
are.

Maps are opt-in: a score-only request runs the unchanged score path, and a
map request returns bit-identical scores (checked per metric below and by
`conformance::assert_conforms`). Strided `PixelSlice` inputs are accepted;
the adapters repack them exactly as before.

## NLPD (`nlpd`)

| Map | Values | Polarity | Grid |
| --- | --- | --- | --- |
| `level_0` (primary) … `level_5` | `Σ_{R,G,B} (N_ref − N_dist)²`, the squared difference of the normalized Laplacian responses at pyramid level `l` | Difference (`0` = identical) | `⌈w/2^l⌉ × ⌈h/2^l⌉` cells of `2^l` pixels |

The score is `(Σ_l rms_l^0.6)^(1/0.6)` with `rms_l = sqrt(mean(level_l) / 3)`,
so each map is one level's component, not a per-pixel NLPD.

Geometry. The stride-2 5-tap downsample centres output `k` on parent sample
`2k + 1` when the parent extent is even (pad 1) and on `2k` when it is odd
(pad 2). Level `l` sample `k` therefore sits at input `a_l + 2^l·k`, with
`a_l = a_{l−1} + 2^{l−1}` exactly when the level-`(l−1)` extent is even; each
cell is the `2^l`-wide block centred on its sample. The local halo
`10·2^l − 2` bounds the footprint: eight level samples (blur, align-corners
upsample, downsample, divisive-normalization neighbour) plus a level sample's
own `2^{l+1} − 2` input pixels.

Validation (`crates/nlpd`):

- Level maps equal the materialized reference path (`downsample` →
  `upsample_bilinear` → `blur5` → subtract → `normalize_band`) bit for bit.
- Folding the maps reproduces the score within 1e-6 relative.
- A linear ramp pushed through the downsample reads back each level's sample
  centres, which match the geometry, for all four parity combinations.
- With a changed patch at the interior, the top-left corner and the
  bottom-right corner, every cell whose declared support misses the patch is
  exactly 0, and every level responds.
- Map-on and map-off scores are bit-identical for textured, identical, flat and
  patched pairs at 112×100, 97×101, 96×98 and 131×96. Strided input gives the
  packed maps. Subsampled requests and cancellation behave as specified. Tests
  pass with default, `parallel` and `avx512,parallel` features.

## PSNR-HVS (`psnrhvs`)

| Map | Values | Polarity | Grid |
| --- | --- | --- | --- |
| `csf_error` (primary) | mean over R, G, B of `csf_error_{r,g,b}` | Difference | complete 8×8 blocks |
| `masked_error` | mean over R, G, B of `masked_error_{r,g,b}` | Difference | same |
| `csf_error_r`, `_g`, `_b` | block mean per-coefficient CSF-weighted squared DCT error, `S_block / 64` (PSNR-HVS) | Difference | same |
| `masked_error_r`, `_g`, `_b` | same after between-coefficient contrast masking (PSNR-HVS-M) | Difference | same |

The grid is the score's own: `⌊(w−8)/8⌋+1 × ⌊(h−8)/8⌋+1` blocks from the
origin. The reference drops partial right/bottom strips, so they have no
cells. Each value depends only on its block (support halo 0). A channel's
score is `10·log10(255² / mean(map))`; the headline averages the three
channels' decibels, so no single map folds to it.

Validation (`crates/psnrhvs`):

- Per-channel maps fold to `psnr_hvs` and `psnr_hvs_m` bit for bit using
  the kernel's band split (64×48, 37×29, 8×8, 130×75). The block energies
  are written after each band's own fold.
- An independent f64 transcription of one `psnrhvsm.m` block (DCT by
  definition, `maskeff`, both energies) agrees with every block to 2.0e-6
  relative.
- Block maps are bit-identical across SIMD tiers. A change inside one block
  moves only that block; a change in the unscored strip moves none.
- Map-on and map-off scores are bit-identical (textured, identical, flat,
  odd sizes); strided input gives the packed maps. Subsampled and stopped
  requests behave as specified. Tests pass with default, `parallel`,
  `avx512,parallel` and no-std builds.

## mDCT-PSNR (`mdctpsnr`)

| Map | Values | Polarity | Grid |
| --- | --- | --- | --- |
| `detection` | `1 − Π exp(−(|Δc|·vis·mask)^3.5)` over the 195 band/component terms at one sliding-window position | DetectionProbability (`0` = identical) | `W−13` × `H−12` cells of 1 pixel |

These are the terms the score sums: `score = −20·log10(Σ / ((W−13)(H−13)·3))`
(the reference's `H−13` denominator over `H−12` measured lines is kept).
Map row `j` is measured image line `j + 12`, which scores block row `j + 3`
(the warm-up skips block rows 0–2). Column `i` scores block column `i + 2`,
masked by block positions `i … i+4`. The value therefore reads input pixels
`[i, i+12) × [j+1, j+13)`. The cell owns pixel `(i+5, j+6)`, the upper-left
of the scored block's four centre pixels, and the halo is 6.

Validation (`crates/mdctpsnr`):

- Folding the map rows in the reference's vectorized f32 order reproduces the
  score bits at 64×48, 37×29, 14×14, 100×23 and 45×61. The existing
  `dctpsnr` goldens are unchanged.
- Cells whose window misses a changed patch are exactly 0. For an 8×8 patch,
  the responding cells fill exactly the predicted window box on both axes.
  When a window holds only one column or row of a few changed pixels,
  `exp(−p)` can round to 1 in f32 and that edge cell stays 0.
- Identical pairs detect nothing. Map-on and map-off scores are
  bit-identical. Strided input gives the packed map. Subsampled and stopped
  requests behave as specified. Tests pass with default, `parallel` and
  no-std builds.
