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

## MAD (`mad-iqa`)

| Map | Values | Polarity | Grid |
| --- | --- | --- | --- |
| `hi` (primary) | `hi_index`'s map: contrast-threshold mask × 16×16 local MSE | Difference | 1-pixel cells over the interior |
| `lo` | `lo_index`'s map: `Σ_s w_s·(|Δσ| + 2|Δskew| + |Δkurt|)` over the log-Gabor bank | Difference | 4×4 tiles over the interior |

Both indices pool only the interior `[16, w−17) × [16, h−17)`, so the maps
cover just that region. `HI = 10·RMS(hi)`. `LO` is the RMS of the tile values
over interior pixels; the last tile row and column reach up to 3 pixels past
the interior. Each tile value covers the 4×4 pixels at its tile origin, as
the reference writes it. Support is global (DFT-domain CSF and log-Gabor
filters). These are the two strategy components: `MAD = HI^σ·LO^(1−σ)`
combines the indices, so there is no per-pixel MAD, and MAD is 0 whenever
HI is. Below 34 pixels there is no interior and a map request fails with
`NoMap`.

Validation (`crates/mad-iqa`):

- `HI` and `LO` fold from the maps bit for bit at 96×96, 77×61, 34×40 and
  130×70. The existing Octave goldens are unchanged.
- A bright patch on a flat field lights `hi` only inside the 16×16
  local-MSE box `{x−7 … x+8}` that holds it, which pins the pixel alignment.
  A faint change in texture can stay below the contrast threshold, and then
  `hi` is legitimately 0.
- The `hi` and `lo` peaks sit at a localized patch. Identical pairs give zero
  maps. Map-on and map-off scores are bit-identical. Strided input gives the
  packed maps. Subsampled, stopped and too-small requests behave as
  specified. Tests pass with default, `parallel`, `_dev,avx512` and no-std
  builds.

## IW-SSIM (`iwssim`)

| Map | Values | Polarity | Grid |
| --- | --- | --- | --- |
| `cs_0` (primary), `cs_1`, `cs_2`, `cs_3` | contrast-structure `(2σ₁₂+C₂)/(σ₁²+σ₂²+C₂)` of the Laplacian band at scale `s` | Similarity (`1` = identical) | `(w_s−10) × (h_s−10)` cells of `2^s` pixels |
| `csl_4` | `cs·l` at the coarsest scale (luminance included) | Similarity | same at scale 4 |

Scales 0–3 pool their map with the information-content weights, the top
scale by its plain mean, and the score is `Π wmcs_s^β_s`. These maps are
per-scale components, not a per-pixel IW-SSIM. The weights need a
non-quality map polarity, which the abstractions do not define yet; they are
not exposed here.

Geometry. The binom5 `corr_dn` keeps even samples, so scale-`s` sample `k`
sits at input `2^s·k`. Cell `i` of the valid 11×11-window map is centred on
scale sample `i + 5`, and each cell is the `2^s` block around that centre.
The halo is `11·2^s − 2` (window 5 samples, Laplacian band `6·2^s − 2`) and
`7·2^s − 2` at the residual top scale. When `allow_small` tiles a small
input up to 176, values also read the wrapped copies, so support is global.

Validation (`crates/iwssim`):

- With `iw_flag` off, every map's mean reproduces `wmcs_s` bit for bit; with
  weighting on, `csl_4`'s does.
- Warm candidates return the cold maps and scores bit for bit, repeatedly,
  and still score without maps afterwards.
- Cells whose declared support misses a changed patch are bit-identical to
  the unchanged pair's. A single changed pixel dips `cs_0` at the cell that
  owns it, within one pixel.
- Map-on and map-off scores are bit-identical (textured, identical, flat and
  patched pairs at 176×176, 200×183, 256×192 and tiled 64×48). Strided input
  gives the packed maps, and stopped requests fail as specified. Tests pass
  with default, `parallel`, no-std and `_dev,avx512` builds; the Python parity
  and strip goldens are unchanged.

## VIF (`vif`: `Vif` = VIFp, `Vifvec`)

VIF is `Σnum / Σden`: the information the distorted channel carries over
the information the reference channel carries. The maps are per-position
information loss, `den − num`. Summed over all maps and divided by `Σden`,
they give `1 − VIF`. They are contributions, not a local VIF ratio. They
are Difference polarity: `0` means no loss, and values are negative where
the distortion adds information (VIF > 1). The raw `num` and `den` maps
would need a non-quality map polarity, which the abstractions do not
define yet.

| Metric | Maps | Values | Grid |
| --- | --- | --- | --- |
| `Vif` | `info_loss_0` (primary) … `info_loss_3`, one per scale with a non-empty statistics grid | `log10(1 + σ₁²/σₙ²) − log10(1 + g²σ₁²/(σᵥ² + σₙ²))` | scale `s` samples input `8 + 4s + 2^s·i`; `2^s` cells, halo `8 + 4s` |
| `Vifvec` | `info_loss_l1_b1` (primary), `info_loss_l1_b4`, … `info_loss_l4_b4` (level, MATLAB band) | per 3×3 block, `Σ_λ [log2(1 + s·λ/σₙ²) − log2(1 + g²·s·λ/(σᵥ² + σₙ²))]` | cropped block grid; `3·2^(L−1)` pixels; global support |

VIFp geometry. Scale `s > 0` filters the previous plane 'valid' with this
scale's own `N = 2^(4−s)+1`-tap window and decimates `1:2:end`, as
`vifp_mscale.m` does. The statistics grid is that window's 'valid' output,
so the sample centres are 8, 12, 16 and 20 plus `2^s·i`, and a value reads
the input within `8 + 4s` pixels of its centre. In vifvec, `corrDn` (start 0,
stride 2 between levels) puts level-`L` sample `k` at input `2^(L−1)·k`. Cell
`i` is block `i + offset` of the cropped grid. Its S-field and eigenvalues
come from whole-subband statistics.

Validation (`crates/vif`):

- The captured VIFp terms fold to the score bit for bit in one running
  accumulator (64×48, 97×61, 17×17, 130×129, 40×18). The vifvec block terms
  regroup the same `log2` terms and fold within 1e-12.
- VIFp cells whose declared support misses a changed patch match the
  unchanged pair bit for bit. The support is a bound: the 'valid' decimation
  chain need not reach a corner at coarse scales. A single changed pixel
  peaks `info_loss_0` at the cell that owns it, and vifvec's finest-level
  peak sits at a localized patch.
- Map-on and map-off scores are bit-identical. Strided input gives the
  packed maps. Subsampled and stopped requests behave as specified. Tests
  pass with default, `parallel`, no-std and `_dev,avx512` builds; the Octave
  goldens are unchanged.
