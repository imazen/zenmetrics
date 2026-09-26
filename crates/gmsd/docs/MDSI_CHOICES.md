# Interpretation choices (clean-room MDSI)

Source: `MDSI_paper_IEEEAccess2016.md` only. The paper text in that file is OCR-scrambled in places (equations 5, 7,
12 and the section III-H paragraph); the readings below are the ones consistent with the surrounding prose.
"Validated" means the choice was checked against the 116 black-box `author_mdsi` values; a sweep of the alternatives
in that column is recorded under "Rejected".

Legend: **P** = stated by the paper, **C** = a choice where the paper is silent or ambiguous.

## Stated by the paper (no choice)

| Item | Value | Paper sentence |
|---|---|---|
| Luminance | `0.2989 R + 0.5870 G + 0.1140 B` | "conversion to luminance is done through the following formula: L = 0.2989 R + 0.5870 G + 0.1140 B" |
| H, M channels | `H = .30R + .04G − .35B`, `M = .34R − .60G + .17B` | eq. (1) |
| Gradient operator | Prewitt on the luminance of R and D | "Through this paper, Prewitt operator is used to compute gradient magnitudes of luminance L channels of reference and distorted images" |
| GS, GS_RF, GS_DF, GS^c | eqs. (2)–(5) | "GS^c(x) = GS(x) + \|GS_DF(x) − GS_RF(x)\|"-form of eq. (5), see C6 |
| Fused image | `F = 0.5 (R + D)` on luminance | "We fuse the luminance L channels of the R and D by a simple averaging: F = 0.5 × (R + D)" |
| CS | eq. (7) | see C5 |
| Combination | summation, eq. (9) | "The proposed metric MDSI uses equation (9)" |
| Constants | `α = 0.6, C1 = 140, C2 = 55, C3 = 550` | "we set α = 0.6, C1 = 140, C2 = 55, and C3 = 550" |
| Pooling | `ρ = 1, q = 1/4, o = 1/4` (eq. 13) | "for the proposed index MDSI, we set ρ = 1, q = 1/4 and o = 1/4" |
| Perfect quality | score 0 | "an image with perfect quality is assessed by a quality score of zero since there is no variation in its similarity map" |
| Pre-processing | M×M mean filter, downsample by M, then colour conversion | "It first applies average filtering of size M × M on each channel of the R and D images, downsample them by a factor M and convert the results to a luminance and two chromaticity channels" |

## Choices

**C1. Pixel scale.** Constants (`C1 = 140` etc.) are used on the 0..255 scale, RGB8 values unnormalised.
Rests on: III-F gives constants of order 10²; on a 0..1 scale they would be meaningless (also the paper's GMSD
discussion of `c = 170` on 0..255). Validated (all 116 pairs).

**C2. Downsampling factor.** `M = max(1, round(min(h, w) / 256))`, `round` half away from zero.
Rests on: "The value of M is set to min[(h, w)/256] [48] … and [.] is the round operator" (the extracted text is
garbled; the only sensible reading is `round(min(h, w)/256)`). The paper does not say what happens when this is 0
(images below 128 px): the `max(1, ·)` clamp is mine and means "no downsampling". Validated: the target set has
M = 1 (17×13, 63×47, 3×5, 33×31, 1×1), M = 2 (512×384, 511×384, 512×512, 769×513), M = 3 (1022×818) and M = 4
(1024×1024).

**C3. How the M×M average filter + downsample is done.** Zero-padded `'same'` box filter (window always divided by
M², missing samples count as 0), evaluated at samples `0, M, 2M, …`; output size `⌈w/M⌉ × ⌈h/M⌉`. For the window
of output `i` the taps are `i·M − ⌊(M−1)/2⌋ … + M − 1` (the usual centred window; for even M the extra tap is on the
far side). Rests on: "applies average filtering of size M × M on each channel … downsample them by a factor M" —
the paper gives no boundary rule, so this is the reading that fits the numbers. Validated.
*Rejected* (swept against the targets, none reaches 116/116; best of each family with everything else at the
chosen reading): replicate or symmetric padding (90/116 pass, worst rel error 3.0e-3 — every pair with an odd
size at M ≥ 2 or with M ≥ 3 fails, M = 1 and even-size M = 2 pairs pass), "valid" non-overlapping block mean
with `⌊w/M⌋` outputs (89/116, worst 6.2e-2), and sampling at the window centre instead of the window start.

**C4. Gradient boundary.** Prewitt is a 3×3 `'same'` correlation with **zero padding**, kernel scale `1/3`
(`[1 0 −1; 1 0 −1; 1 0 −1]/3` and its transpose), magnitude `sqrt(gx² + gy²)`, the map keeping the size of the
downsampled image. Rests on: "Prewitt operator" only; neither scale nor boundary is stated. The `1/3` scale is
the conventional normalised Prewitt and is what the constants `C1, C2` (tuned on 0..255 data) need. Validated.
*Rejected* (swept; none reaches 116/116): scale 1 (unnormalised), replicate/symmetric padding, and "valid" (border
dropped, `(w−2)×(h−2)`).
Note that zero padding makes every image border a strong edge in `L`; the paper does not comment on it.

**C5. Reading of the scrambled eq. (7).** `CS = (2 (H_R H_D + M_R M_D) + C3) / (H_R² + H_D² + M_R² + M_D² + C3)`.
Rests on: "the proposed formulation calculates a color similarity map using both chromaticity channels at once"
and "the above joint color similarity (CS) formulation gives equal weight to both chromaticity channels H and M"
(so the numerator is the sum of both channels' products, doubled, as in eq. 6 per channel). Validated (the other
plausible reading, `2 H_R H_D + M_R M_D`, gives unequal weights and was not tried).

**C6. Reading of the scrambled eq. (5).** `GS^c = GS_RD + (GS_DF − GS_RF)` with **no absolute value** (the
extracted text prints "GS(x) + [GS_DF(x) − GS_RF(x)]"). Rests on: "The added term GS_DF(x) − GS_RF(x) will put more
emphasize on removed edges from R than added edges to D", and on "GS^c … might have values greater than −1 and/or
positive values smaller than +2": a term that can be negative or positive gives `GS^c ∈ [0−δ1, 1+δ2]`, as the
paper says of GCS in II-E. Validated.

**C7. Negative GCS under the fourth root (eq. 13).** The paper states "possible interval for GCS is [0 − δ1,
1 + δ2]" and "GCS < 0 are highly distorted pixels", yet evaluates `GCS^{1/4}`. A negative real number has no real
fourth root, and the paper does not say what to do. I take the **principal complex root**
(`|v|^{1/4}·e^{iπ/4}`), then the mean is a complex mean and `|x_i − mean|` is the complex modulus, exactly what the
formula means when read over ℂ. Validated: this is the only reading that matches; **this choice was found by
iteration against the black-box targets** (see history). *Rejected*: `NaN` (propagates, fails), `|v|^{1/4}`
(worst rel error 8.6e-2), clamping negative values to 0 (5.9e-2).

**C8. Deviation pooling detail.** `MDSI = ( (1/N) Σ |x_i − (1/N) Σ x_j| )^{1/4}` with `x_i = GCS_i^{1/4}`, mean
(not median/mode) as the measure of central tendency, `N` (not `N − 1`) as normaliser. Rests on: eqs. (10)–(13)
and "The only MCT used in this paper is mean". Validated.

**C9. Precision and order.** Everything in `f64`; sums accumulated in row-major order; window sums of the
box filter are exact integer sums divided by M² once. The paper says nothing on this. An f32 emulation of the
luma/chroma planes was tried and is *worse* (worst rel error 4e-6), so the reference is evidently double precision.

**C10. Identical images / degenerate sizes.** Identity gives exactly 0 (all similarity maps are exactly 1 or the
same constant, so every deviation is 0). 1×1 gives 0 (validated by target `stress0`). Zero width/height is an error
(`Error::TooSmall`). API: `mdsi_rgb8` panics on bad input, `mdsi_rgb8_strided` returns `Result`.

**C11. Build.** `crates/gmsd/Cargo.toml` made standalone: `workspace = true` dependencies replaced by the versions in
`workspace-dependencies.toml` (`archmage`/`magetypes` `0.9.23` requirement — crates.io resolved 0.9.29), the
`edition`/`rust-version` workspace inheritances replaced by `edition = "2024"`, `rust-version = "1.89"` (the
workspace root is not in this tree, so these two values are my guess, not read from it), and an empty `[workspace]`
table added so cargo does not look for a parent. The module is gated on `feature = "std"`.

## Iteration history against the targets (each change, and why)

1. I wrote a scratch explorer (kept outside the deliverables) that scores all 116 pairs for a grid of readings:
   box-filter padding (zero / replicate / symmetric / valid-block) × window-sample offset (start / centre) ×
   Prewitt padding (same four) × Prewitt scale (1/3, 1) × handling of negative GCS under the fourth root (NaN,
   absolute value, clamp to 0, complex principal root) × rounding of the downsampled RGB to integers (no / yes) —
   512 variants, paper constants throughout.
2. Result: exactly one variant reaches 116/116 within 1e-9 (worst 4.8e-10): zero-padded box filter sampled at the
   window start, zero-padded Prewitt/3, complex principal root, no intermediate rounding. Next best is 90/116
   (replicate/symmetric box padding, otherwise the same), then 89/116; without the complex root the best is
   87/116 (clamp 5.9e-2, abs 8.6e-2, NaN infinite worst error). The complex root (C7) and the zero paddings
   (C3, C4) were therefore found by matching the black-box numbers, not by reading the paper.
3. No parameter (C1, C2, C3, α) was fitted: the paper's values were used throughout.
4. An f32 emulation of the luma/chroma planes was also tried after the fact (C9) and made things worse.

The negative control (`--c3 5500`, also `--c1 280`, `--alpha 0.5`, `--c2 110`) fails 109 of 116 pairs; the other
7 are the exact-zero pairs (`000`–`003_identity`, `011_odd`, `011_tiny`, and the 1×1 `stress0`), which no constant can move.

## Integration notes (zenmetrics `crates/gmsd`)

- The delivered clean-room module is `src/mdsi.rs::reference`, unchanged in its arithmetic (the fourth root uses
  `powf`/`libm::pow`, the complex-root angle comes from `sin_cos(π/4)`); only its plumbing changed (`Error`,
  `check_rgb8`, a `Params` struct for the negative control, integer `M = max(1, round(min/256))`, exhaustively checked
  against the float `round`).
- The public `gmsd::mdsi_rgb8(reference, distorted, width, height, stride_bytes) -> Result<f64>` is the optimized
  path. It is pinned bit for bit to `reference` by the unit tests (score and GCS map, 14 sizes × packed/strided), at
  every tier and thread count. The integer window sums are exact, so C3's box filter is unchanged.
- C10's "`mdsi_rgb8` panics on bad input" describes the clean-room delivery; the crate's entry returns `Result`.
- The 116-pair author-score gate is `mdsi::tests::author_score_gate`; the caller selects it with
  `GMSD_MDSI_GATE=require|skip` and `GMSD_MDSI_TARGETS=<target table>` (see `justfile`).
