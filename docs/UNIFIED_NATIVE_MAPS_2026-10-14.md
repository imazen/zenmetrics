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
per-scale components, not a per-pixel IW-SSIM. With IW pooling on, `iw_0` …
`iw_3` follow: the information-content weights, cropped to the same grids as
`cs_s` (`wmcs_s = Σ cs·iw / Σ iw`). They are mutual-information pooling
weights with no match direction (`MapPolarity::Feature`), and their support is
global because they come from the whole band's covariance eigendecomposition.
The fold holds to f32 precision: the kernel narrows both f64 sums to f32
before dividing.

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
the distortion adds information (VIF > 1). After the loss maps come the two information terms themselves on the same
grids: `num_s`/`den_s` (VIFp) and `num_lL_bB`/`den_lL_bB` (vifvec). They use
`MapPolarity::Feature`, and `den` depends on the reference alone.

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

## VMAF (`vmaf`, v1.0.16 models)

| Map | Values | Polarity | Grid |
| --- | --- | --- | --- |
| `speed_u` (primary), `speed_v` | SPEED's per-block `|S_ref − S_dist|`, the entropy-weighted local difference of one chroma plane | Difference | 5×5 operating-sample blocks; global support |
| `aim_{s}_{h,v,d}`, `s` = 0…3 | ADM3's AIM terms: per band sample, the cube of the additive impairment left visible after masking by the restored signal | Difference (`0` = no visible impairment) | `2^(s+1)`-pixel blocks over ADM's pooled inner region; global support |
| `adm_num_{s}_{h,v,d}`, `adm_den_{s}_{h,v,d}` | ADM3's detail-loss terms: the cube of the restored detail left visible after masking by the additive impairment, and the cube of the reference's CSF-weighted detail | Feature | the same grids as AIM |
| `cambi_0` … `cambi_4` | CAMBI's per-scale c-values of the distorted frame | Feature (higher = more visible banding) | point samples at `2^s·k`; local support |

Coverage: 3 of the model's 4 elementary features (SPEED chroma, ADM3,
CAMBI). VMAF has no per-pixel score; it is an SVM over pooled
features. SPEED chroma's per-block terms have an f32 mean equal to the
channel score (before the U/V combine and the 45 clamp). ADM3 blends
`1 − AIM` (weight 0.3) with the detail-loss ratio `Σnum/Σden` (0.7); AIM is
`Σ_s aim_num_s / Σ_s den_s`, and `aim_num_s` adds, per band, the cube root
of the band's summed AIM terms; `num_s` and `den_s` are built the same way
from their terms, plus a noise floor per band. The maps are those terms, so
each band map sums (before the cube root) to its band's argument, not to a
share of the score. The detail-loss terms have no reference-comparison
reading of their own (more restored detail is not, by itself, worse), so
they are Feature maps; their per-scale ratio after the cube roots is ADM's
detail-loss measure.
CAMBI's c-values are what its top-k pooling reads, and libvmaf dumps the
same values with `heatmaps_path`. CAMBI measures banding in the distorted
frame alone, so it has no reference comparison. `motion3` is temporal and
reference-only.

SPEED geometry. An operating sample is every 16th sample of the
antialiased, prescaled chroma plane, and a block is 5×5 of them. A block is
therefore 160 luma pixels at prescale 1 (origin −15) and 320 at prescale 0.5
(origin −30), in continuous luma coordinates. Support is global: the block
entropies use frame-wide covariance eigenvalues. The phone models' 0.6
prescale has no whole-pixel grid, so they serve no SPEED maps rather than an
approximated grid.

AIM geometry. ADM's DWT output `k` reads inputs `2k − 1 … 2k + 2`, so a
scale-`s` band sample owns the `2^(s+1)`-pixel block at `k·2^(s+1)`; the
maps cover the region ADM pools, the band minus `⌊0.1·w_s − 0.5⌋` samples on
each side (rows likewise). ADM has no prescale, so every model, phone
included, serves AIM maps; they are the phone models' primary maps. The
masking and decoupling chain is declared global rather than bounded.

CAMBI geometry. `decimate` point-samples even positions per scale, so
scale-`s` sample `k` is input pixel `2^s·k` (doubled under the
high-resolution speedup). Cells are centred on those samples. The halo
`(window/2 + 2)·factor + 5` covers the histogram window, the per-scale mode
filters, and the full-resolution spatial mask and anti-dithering.

Map requests are served for single pairs (`Metric::evaluate`); sequences
refuse them.

Validation (`crates/vmaf`):

- SPEED: map requests leave all five outputs bit-identical, and the maps fold
  to `speed_chroma_uv` bit for bit (Standard1080p and Consumer4k; distinct,
  identical and chroma-shifted pairs). That feature is pinned to libvmaf by
  the existing FFI suite. A chroma change confined to one block peaks in that
  block's cell. Identical frames leave per-block terms up to 3.05e-5 at
  prescale 1, from `log2(1 + σ²)` taken in f32 for the reference and f64 for
  the distorted side, as libvmaf computes it; they are 0 at prescale 0.5.
- ADM: the per-position integer terms, folded with ADM's per-row rounding
  shifts, reproduce each scale's detail-loss numerator and denominator and
  AIM numerator from the score path bit for bit (five sizes from 33×40 to
  321×183, 8- and 10-bit, four models), with default, `simd`,
  `simd,parallel` and `avx512,parallel` features; debug builds assert it on
  every map request. The host
  has no AVX-512, so the `avx512` build ran its AVX2 kernels. The f32 maps
  sum to each band's argument within the reference's per-row rounding (half
  a unit per row) plus f32 storage. Identical frames give exact AIM zeros.
  A luma change confined to one region puts every scale-0 AIM band's peak
  cell over it.
- CAMBI: the served maps, quantized as libvmaf quantizes its heatmaps, match
  libvmaf's `heatmaps_path` dump exactly (0 samples differ; 91,133 nonzero;
  640×360 10-bit banded fixture). Folding them through the same top-k
  selection reproduces the CAMBI score bit for bit. Changing the distorted
  frame outside a cell's declared support leaves the cell bit-identical, and
  changing the reference changes nothing.
- Sequence and stopped requests refuse. Tests pass with default (43),
  `simd` (64), `simd,parallel` (66) and `avx512,parallel` (66)
  features, libvmaf parity included.

## SSIMULACRA2 (`fast-ssim2`, adapter `zenmetrics-browser::ssim2_map`)

fast-ssim2 `compute_ssimulacra2_maps` (PR #12) returns, per pyramid scale and
positive-XYB channel, the three per-pixel terms the score pools:
`max(0, 1 − SSIM)`, edge artifact and edge detail lost. The score takes the
mean and the 4-norm of each term per scale, then a weighted sum of the 108
aggregates and a nonlinear transform, so there is no per-pixel SSIMULACRA2.
All maps are Difference polarity:

- `diffmap` (primary): `Ssimulacra2Maps::diffmap`, the terms fused at input
  resolution with the score's own weights. Each L1 and L4 aggregate splits
  into per-sample shares (`w₁·x/N + w₄·x⁴/(N·L4³)`), and each scale's shares
  spread over the pixels it covers, so the map's mean is the weighted sum the
  score transforms. This is the shape encoder quantization loops steer by:
  jxl-encoder's `PerceptualBackend` takes one `width × height` diffmap per
  candidate against a cached reference and reduces it to 8×8-block 16-norms
  (`tile_distmap.rs`), as zensim's `compute_with_ref_and_diffmap` supplies.
  Support is global (the L4 normalization).
- The terms themselves, `{ssim,artifact,detail_lost}_{x,y,b}_{scale}`, then
  `ssim_y_0` first among them.

Geometry. Scale `s` samples `2^s × 2^s` pixel blocks from the origin, and a
value reads its scale within the blur's 5-sample radius (halo `5·2^s`).
fast-ssim2 measured recursive-blur rounding residue past that radius of up to
1.7e-4 against a 0.66 peak. Inputs under 8×8 are reflect-padded, so their
cells beyond the image clip and their support is global.

The adapter lives in the browser crate because the CLI does not depend on
the abstractions. It accepts packed RGB8/RGBA8 sRGB and linear RGBF32
(strided rows natively), and its warm reference reuses
`Ssimulacra2Reference` for scores and maps (`compare_with_maps`, bit-identical
to the cold maps).

Validation:

- fast-ssim2: the map-path score is bit-identical to the score path (14
  input shapes and formats, both kernels, every cancellation poll). Scale-0
  L1/L4 from the maps match the reference-order helpers to 1e-6 relative, and
  the score rebuilt from the maps is within 4.7e-7.
- Adapter: the headline equals `fast_ssim2::compute_ssimulacra2` bit for bit;
  it passes conformance at 64×48 and 5×7; warm candidates match cold scores and
  maps; strided input; cancellation.
- Pin: moving zenmetrics from fast-ssim2 96ad1ab0 to the PR head leaves
  SSIMULACRA2 score bits unchanged on seven fixtures (64×48 to 1024×768 with
  rayon, sub-8, the tank pair). Browser native tests passed (44, 2 ignored, as
  before) and the zenmetrics-api CPU suite passed (22) at the PR head; after
  merging master's eight newer browser routes the browser tests pass (46, 2
  ignored).
- fast-ssim2 (`c91ca7ed`): cached-reference maps equal the cold maps bit for
  bit (RGB8, identical, RGBA8, RGBF32, RGBAF32 at four sizes, both kernels);
  the diffmap's mean matches the weighted sum of the stored terms to 5.8e-10
  relative and transforms back to the score within 8.3e-8.
- Adapter: `diffmap` equals `fast_ssim2::compute_ssimulacra2_maps(..).diffmap()`
  bit for bit; warm candidates serve every cold map bit for bit. Browser
  native tests pass (46, 2 ignored).
- Chromium (`check-pages.py`): the `ssim2` route's diffmap matches
  fast-ssim2's own diffmap from `browser_oracle` exactly on the 64×64 fixture
  (4,096 samples), identical 257×253 inputs give exact zeros, and the demo UI
  renders it.

## Fused maps (`diffmap`), 2026-10-15

Metrics that pool several scales also serve `diffmap`: the metric's own
score broken down at input resolution, so one map shows where all of its
scales find error, not just the finest. Each is built from the metric's
native terms with its own weights; shares from a scale spread evenly over the
input pixels each cell covers, the map is cropped to the pixels some cell
covers, and it is scaled so **its mean is the quantity named below**. All
are Difference polarity with global support (the pooling normalizes over the
whole image).

| Metric | Mean of `diffmap` | Allocation |
| --- | --- | --- |
| SSIMULACRA2 | the weighted sum the score transforms | exact: `w₁·x/N + w₄·x⁴/(N·L4³)` per term |
| NLPD | NLPD | exact: NLPD is homogeneous of degree ½ in the level maps, so `2·x·∂NLPD/∂x` per sample |
| VIF (VIFp, vifvec) | `1 − VIF` | exact: `(den − num)/Σden` per element |
| MS-SSIM | `−ln MS-SSIM` | each level's `w_l·(−ln m_l)` split in proportion to `1 − x` |
| IW-SSIM | `−ln IW-SSIM` | each scale's `β_s·(−ln wmcs_s)` split in proportion to `iw·(1 − cs)` |

The MS-SSIM and IW-SSIM splits are a choice (the log of a mean has no unique
per-sample decomposition); the totals are exact while every level mean lies in
`(0, 1)`. `diffmap` is the primary map for NLPD and VIF; MS-SSIM keeps `ssim`
and IW-SSIM `cs_0` as primary, with `diffmap` among their maps. DSSIM is not
fused: its scale weights are not exposed by `dssim-core`.

Validation: NLPD's mean equals the score within 1e-6 relative; VIF's equals
`1 − VIF` within 1e-6 absolute (about 4e-8 measured), and a flat reference
(VIF undefined) gives zeros; MS-SSIM's and IW-SSIM's equal `−ln` of the score
within 1e-6 and 1e-5 relative; identical inputs give exact zeros (IW-SSIM:
at most 1.3e-7, cs within f32 rounding of 1), and a local change peaks over
it, for each metric. In headless Chromium the SSIMULACRA2, NLPD, VIF, vifvec
and MS-SSIM fused maps equal the native oracle exactly; IW-SSIM's differs by at
most 1.8e-7, the same cross-SIMD-tier rounding its `cs` maps show.

## Browser viewer

The WASM viewer serves each metric's `diffmap` when it has one, else its
primary map, through the same call
(`feaafabd`): `nlpd`, `psnrhvs`, `mdctpsnr`, `mad`, `iwssim`, `vif`, `vifvec`
and `vmaf-v1`, and `ssim2` serves the fused `diffmap` through the adapter
above. The
other maps stay available to native callers through `Evaluation::maps()`.
Routes and browser parity: [browser viewer](BROWSER_VIEWER_2026-10-13.md).
