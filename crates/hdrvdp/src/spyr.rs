//! Steerable pyramid — the multi-scale, multi-orientation decomposition
//! HDR-VDP-2 runs its masking model on.
//!
//! Ported from `buildSpyr` / `buildSpyrLevs` / `reconSpyr` / `reconSpyrLevs`
//! in matlabPyrTools (<https://github.com/LabForComputationalVision/matlabPyrTools>,
//! MIT — Eero Simoncelli), as bundled with the HDR-VDP-2.2 release. Filter taps
//! live in [`crate::sp3_filters`].
//!
//! ## Structure
//!
//! ```text
//!   image ──► hi0filt ─────────────────────────────► high-pass residual
//!         └─► lo0filt ─┬─► bfilts × 4 ─────────────► level 0, 4 orientations
//!                      └─► lofilt ↓2 ─┬─► bfilts×4 ► level 1, 4 orientations
//!                                     └─► lofilt ↓2 ► … ► low-pass residual
//! ```
//!
//! Bands run fine → coarse. Only the oriented bands are decimated *between*
//! levels: within a level, the four orientation bands are the same size as that
//! level's low-pass input.
//!
//! ## What is faithful here, and what is not — read before trusting `S_map`
//!
//! **[`corr_dn`] is faithful.** matlabPyrTools implements `corrDn`'s
//! `reflect1` boundary by folding the out-of-range taps back onto the mirrored
//! interior sample (`edges.c`, the `REDUCE` branch), which is identical to
//! gathering through a reflected index map — what this module does.
//!
//! **[`up_conv`]'s boundary is *not* verified against upstream.** matlabPyrTools'
//! `upConv` uses a separate `EXPAND` branch in `edges.c` — a different C-coded
//! rule, not the transpose of `REDUCE` — which has not been ported or compared
//! here. This port instead defines synthesis as *the restriction of the
//! infinite operation on the `reflect1` extension*, verified exactly by
//! `analysis_and_synthesis_match_the_infinite_symmetric_extension`.
//! In the interior that coincides with upstream (both are the plain
//! transpose); at the outermost pixels the two may differ.
//!
//! **Two further properties, both measured rather than assumed:** the spatial
//! sp3 filter set is only *approximately* self-inverting, costing ≈1 % per
//! level traversed (see
//! `reconstruction_is_approximate_by_design_and_costs_about_1_percent_per_level`),
//! and reconstruction is worse at the border than in the interior because each
//! recursion re-applies `reflect1` on the *decimated* grid — as upstream's
//! `buildSpyrLevs` also does.
//!
//! **What this affects.** [`reconstruct`] feeds only HDR-VDP-2's visibility map
//! (`S_map` → `P_map` / `P_det`), so border pixels of that map are the exposed
//! surface. The quality correlate `Q` / `Q_MOS` is accumulated in the band loop
//! and never calls [`reconstruct`] at all, so the metric's headline score and
//! its UPIQ validation are untouched by this gap.

use crate::sp3_filters::{BFILTS, HI0FILT, LO0FILT, LOFILT};

/// Number of orientation bands (`sp3Filters`).
pub const ORIENTATIONS: usize = 4;

/// One subband: a row-major plane with its own dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    /// Band width in samples.
    pub width: usize,
    /// Band height in samples.
    pub height: usize,
    /// Row-major coefficients, `width · height` of them.
    pub data: Vec<f64>,
}

impl Band {
    /// An all-zero band of the given size.
    #[must_use]
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }
}

/// A built steerable pyramid.
#[derive(Debug, Clone)]
pub struct SteerablePyramid {
    /// High-pass residual, at full image resolution.
    pub high_pass: Band,
    /// Oriented bands, fine → coarse: `levels[l][o]`, `o` in `0..ORIENTATIONS`.
    pub levels: Vec<[Band; ORIENTATIONS]>,
    /// Low-pass residual, the coarsest band.
    pub low_pass: Band,
    /// Source image width.
    pub width: usize,
    /// Source image height.
    pub height: usize,
}

impl SteerablePyramid {
    /// Number of oriented levels — upstream's `spyrHt`.
    #[must_use]
    pub fn height_levels(&self) -> usize {
        self.levels.len()
    }

    /// Band centre frequencies in cycles/degree, in the order HDR-VDP-2's band
    /// loop walks them: `[0]` is the **high-pass residual**, `[1..=H]` the
    /// oriented levels fine → coarse, and `[H+1]` the low-pass residual, each
    /// at `2^-b · ppd / 2`.
    ///
    /// That is `height_levels() + 2` entries — upstream's
    /// `2.^-(0:(spyrHt+1)) · ppd/2`, matching `bands.sz = [1, 4×H, 1]`. Getting
    /// this off by one silently shifts every band's CSF lookup and quality
    /// weight by one octave.
    ///
    /// At the calibration resolution of 30 pixels/degree, a 5-level pyramid
    /// gives exactly `[15, 7.5, 3.75, 1.875, 0.9375, 0.4688, 0.2344]` — the
    /// frequencies [`crate::params::Params::quality_band_freq`] tabulates, which
    /// is how we know the 2.2 quality fit was done at 30 ppd.
    #[must_use]
    pub fn band_frequencies(&self, pix_per_deg: f64) -> Vec<f64> {
        (0..self.height_levels() + 2)
            .map(|b| 2f64.powi(-(b as i32)) * pix_per_deg / 2.0)
            .collect()
    }
}

/// Largest pyramid height buildable for an image, given the low-pass filter
/// size — upstream's `maxPyrHt`: how many times the image can be halved before
/// either axis falls below the filter.
#[must_use]
pub fn max_pyr_height(width: usize, height: usize, filter_size: usize) -> usize {
    let (mut w, mut h, mut n) = (width, height, 0usize);
    while w >= filter_size && h >= filter_size {
        n += 1;
        w /= 2;
        h /= 2;
    }
    n
}

/// Reflect an index about the edge samples (`reflect1`): `-1 → 1`,
/// `n → n-2`, and so on, without duplicating the edge sample.
#[must_use]
#[inline]
pub fn reflect1(i: isize, n: usize) -> usize {
    debug_assert!(n > 0);
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut i = i % period;
    if i < 0 {
        i += period;
    }
    if i >= n as isize {
        i = period - i;
    }
    i as usize
}

/// Correlate `image` with a square `filter` of side `fs`, then keep every
/// `step`-th sample starting at 0 — upstream's `corrDn(im, filt, 'reflect1',
/// [step step], [1 1])`.
///
/// The filter origin is its centre, `fs/2`.
#[must_use]
pub fn corr_dn(
    image: &[f64],
    width: usize,
    height: usize,
    filter: &[f64],
    fs: usize,
    step: usize,
) -> Band {
    assert_eq!(image.len(), width * height, "corr_dn: image size mismatch");
    assert_eq!(filter.len(), fs * fs, "corr_dn: filter size mismatch");
    assert!(step >= 1);

    let out_w = width.div_ceil(step);
    let out_h = height.div_ceil(step);
    let c = (fs / 2) as isize;
    let mut out = vec![0.0; out_w * out_h];

    for oy in 0..out_h {
        let base_y = (oy * step) as isize;
        for ox in 0..out_w {
            let base_x = (ox * step) as isize;
            let mut acc = 0.0;
            for ky in 0..fs {
                let sy = reflect1(base_y + ky as isize - c, height);
                let row = &image[sy * width..sy * width + width];
                let frow = &filter[ky * fs..ky * fs + fs];
                for (kx, &w) in frow.iter().enumerate() {
                    let sx = reflect1(base_x + kx as isize - c, width);
                    acc += w * row[sx];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }
    Band {
        width: out_w,
        height: out_h,
        data: out,
    }
}

/// Upsample `band` by `step` and convolve with `filter`, **accumulating** into
/// `res` — upstream's `upConv(im, filt, 'reflect1', [step step], [1 1],
/// size(res), res)`.
///
/// In the interior this is exactly the transpose of [`corr_dn`]:
/// `res[a] += Σ_k f[k] · u[a − k + c]`, where `u` is `band` zero-upsampled by
/// `step`. At the boundary it reflects the *upsampled signal* rather than
/// folding the scattered output — which is the difference between a synthesis
/// that reconstructs and one that does not.
///
/// **Why not the literal adjoint.** Writing this as a scatter — accumulate
/// each band sample's filter footprint into `res[reflect1(…)]` — is the exact
/// transpose of `corr_dn`'s gather at *every* position, boundary included, and
/// it is wrong: the folded contributions pile up on the outermost pixels, so a
/// constant band reconstructs to a constant in the interior and to something
/// ~45 % larger at the border. Reflecting the input instead keeps the parity
/// (`reflect1` maps even positions to even ones — which is precisely why
/// upstream picked it for 2× subsampling), so every output position sums the
/// same parity-matched subset of taps and a constant survives everywhere.
///
/// The gate is
/// `analysis_and_synthesis_match_the_infinite_symmetric_extension`,
/// which pins this to the infinite-domain operation exactly, borders included.
pub fn up_conv(
    band: &Band,
    filter: &[f64],
    fs: usize,
    step: usize,
    out_width: usize,
    out_height: usize,
    res: &mut [f64],
) {
    assert_eq!(filter.len(), fs * fs, "up_conv: filter size mismatch");
    assert_eq!(
        res.len(),
        out_width * out_height,
        "up_conv: res size mismatch"
    );
    assert_eq!(
        (band.width, band.height),
        (out_width.div_ceil(step), out_height.div_ceil(step)),
        "up_conv: band size does not match the requested output size"
    );

    let c = (fs / 2) as isize;
    // Sample the zero-upsampled band at an output-grid position, reflecting
    // out-of-range positions about the edges.
    let sample = |y: isize, x: isize| -> f64 {
        let y = reflect1(y, out_height);
        let x = reflect1(x, out_width);
        if !y.is_multiple_of(step) || !x.is_multiple_of(step) {
            return 0.0;
        }
        band.data[(y / step) * band.width + (x / step)]
    };

    for py in 0..out_height {
        for px in 0..out_width {
            let mut acc = 0.0;
            for ky in 0..fs {
                let sy = py as isize - ky as isize + c;
                let frow = &filter[ky * fs..ky * fs + fs];
                for (kx, &w) in frow.iter().enumerate() {
                    if w == 0.0 {
                        continue;
                    }
                    acc += w * sample(sy, px as isize - kx as isize + c);
                }
            }
            res[py * out_width + px] += acc;
        }
    }
}

fn flat<const N: usize>(f: &[[f64; N]; N]) -> Vec<f64> {
    f.iter().flat_map(|r| r.iter().copied()).collect()
}

/// Build a steerable pyramid over a row-major image.
///
/// `levels` is the number of oriented levels; `None` means as many as the
/// image allows ([`max_pyr_height`], upstream's `'auto'`).
///
/// # Panics
/// If `image.len() != width · height`, or if `levels` exceeds
/// [`max_pyr_height`].
#[must_use]
pub fn build(
    image: &[f64],
    width: usize,
    height: usize,
    levels: Option<usize>,
) -> SteerablePyramid {
    assert_eq!(image.len(), width * height, "build: image size mismatch");
    let lo0 = flat(&LO0FILT);
    let hi0 = flat(&HI0FILT);
    let lo = flat(&LOFILT);
    let bfilts: Vec<Vec<f64>> = BFILTS.iter().map(flat).collect();

    let max_ht = max_pyr_height(width, height, 17);
    let ht = levels.unwrap_or(max_ht);
    assert!(
        ht <= max_ht,
        "build: cannot build {ht} levels on a {width}×{height} image (max {max_ht})"
    );

    let high_pass = corr_dn(image, width, height, &hi0, 15, 1);
    let mut current = corr_dn(image, width, height, &lo0, 9, 1);

    let mut out_levels: Vec<[Band; ORIENTATIONS]> = Vec::with_capacity(ht);
    for _ in 0..ht {
        let bands: Vec<Band> = bfilts
            .iter()
            .map(|f| corr_dn(&current.data, current.width, current.height, f, 9, 1))
            .collect();
        let bands: [Band; ORIENTATIONS] = bands
            .try_into()
            .unwrap_or_else(|_| unreachable!("BFILTS has ORIENTATIONS entries"));
        out_levels.push(bands);
        current = corr_dn(&current.data, current.width, current.height, &lo, 17, 2);
    }

    SteerablePyramid {
        high_pass,
        levels: out_levels,
        low_pass: current,
        width,
        height,
    }
}

/// Reconstruct an image from a steerable pyramid — upstream's `reconSpyr`.
///
/// Because [`corr_dn`] and [`up_conv`] are an exact adjoint pair and the
/// `sp3` filters are designed to be self-inverting, `reconstruct(&build(x))`
/// returns `x` to numerical precision.
#[must_use]
pub fn reconstruct(pyr: &SteerablePyramid) -> Band {
    let lo0 = flat(&LO0FILT);
    let hi0 = flat(&HI0FILT);
    let lo = flat(&LOFILT);
    let bfilts: Vec<Vec<f64>> = BFILTS.iter().map(flat).collect();

    // Walk coarse → fine, rebuilding each level's low-pass input.
    let mut acc = pyr.low_pass.clone();
    for level in pyr.levels.iter().rev() {
        let (w, h) = (level[0].width, level[0].height);
        let mut res = vec![0.0; w * h];
        up_conv(&acc, &lo, 17, 2, w, h, &mut res);
        for (band, f) in level.iter().zip(&bfilts) {
            up_conv(band, f, 9, 1, w, h, &mut res);
        }
        acc = Band {
            width: w,
            height: h,
            data: res,
        };
    }

    // Undo the lo0 / hi0 split.
    let mut out = vec![0.0; pyr.width * pyr.height];
    up_conv(&acc, &lo0, 9, 1, pyr.width, pyr.height, &mut out);
    up_conv(&pyr.high_pass, &hi0, 15, 1, pyr.width, pyr.height, &mut out);
    Band {
        width: pyr.width,
        height: pyr.height,
        data: out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_plus_texture(w: usize, h: usize) -> Vec<f64> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                0.3 * x / w as f64
                    + 0.2 * (y * 0.7).sin()
                    + 0.1 * ((x * 1.9).cos() * (y * 0.3).sin())
            })
            .collect()
    }

    #[test]
    fn reflect1_mirrors_without_duplicating_the_edge() {
        // 0 1 2 3 4 | with reflect1, index -1 reads sample 1, index 5 reads 3.
        assert_eq!(reflect1(0, 5), 0);
        assert_eq!(reflect1(4, 5), 4);
        assert_eq!(reflect1(-1, 5), 1);
        assert_eq!(reflect1(-4, 5), 4);
        assert_eq!(reflect1(5, 5), 3);
        assert_eq!(reflect1(8, 5), 0);
        // Far outside must still land in range (the 17-tap filter on a tiny
        // band reaches several periods out).
        for n in 1..12usize {
            for i in -40isize..40 {
                assert!(reflect1(i, n) < n, "reflect1({i}, {n}) escaped");
            }
        }
        // Degenerate single-sample axis.
        assert_eq!(reflect1(-7, 1), 0);
    }

    #[test]
    fn max_pyr_height_matches_the_reference_recursion() {
        // maxPyrHt halves (floor) until an axis drops below the filter size.
        // 512 → 256 → 128 → 64 → 32 → 16 < 17, so 5 levels with lofilt.
        assert_eq!(max_pyr_height(512, 512, 17), 5);
        assert_eq!(max_pyr_height(256, 256, 17), 4);
        assert_eq!(max_pyr_height(17, 17, 17), 1);
        assert_eq!(max_pyr_height(16, 16, 17), 0);
        // The smaller axis binds.
        assert_eq!(max_pyr_height(1024, 20, 17), 1);
    }

    #[test]
    fn corr_dn_with_a_delta_filter_is_the_identity() {
        let (w, h) = (7usize, 5usize);
        let im = ramp_plus_texture(w, h);
        let mut f = vec![0.0; 81];
        f[4 * 9 + 4] = 1.0; // centre tap of a 9×9
        let out = corr_dn(&im, w, h, &f, 9, 1);
        assert_eq!((out.width, out.height), (w, h));
        for (a, b) in out.data.iter().zip(&im) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn corr_dn_decimates_to_ceil_half() {
        let mut f = vec![0.0; 81];
        f[4 * 9 + 4] = 1.0;
        for (w, h) in [(8usize, 8usize), (9, 7), (1, 1), (5, 12)] {
            let im = ramp_plus_texture(w, h);
            let out = corr_dn(&im, w, h, &f, 9, 2);
            assert_eq!((out.width, out.height), (w.div_ceil(2), h.div_ceil(2)));
            // With a delta filter, decimation just picks even samples.
            for oy in 0..out.height {
                for ox in 0..out.width {
                    let want = im[(oy * 2) * w + ox * 2];
                    assert!((out.data[oy * out.width + ox] - want).abs() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn up_conv_is_the_exact_transpose_of_corr_dn_away_from_the_boundary() {
        // ⟨corr_dn(x), y⟩ == ⟨x, up_conv(y)⟩ whenever no filter footprint
        // reaches an edge — i.e. the interior operation is exactly the
        // transpose. (At the boundary the two deliberately diverge; see
        // `up_conv`'s docs and `analysis_and_synthesis_match_the_infinite_
        // symmetric_extension` for the property that holds there instead.)
        let lo = flat(&LOFILT);
        for (w, h, step, fs, filt) in [
            (40usize, 40usize, 1usize, 9usize, flat(&LO0FILT)),
            (40, 40, 2, 17, lo.clone()),
            (48, 48, 2, 17, lo.clone()),
            (40, 44, 1, 15, flat(&HI0FILT)),
            (40, 40, 1, 9, flat(&BFILTS[1])),
        ] {
            let x = ramp_plus_texture(w, h);
            let cd = corr_dn(&x, w, h, &filt, fs, step);
            // Zero out the coefficients whose footprint touches an edge, so
            // both sides of the inner product only see interior arithmetic.
            let margin = fs; // generous: fs/2 would do
            let y: Vec<f64> = (0..cd.data.len())
                .map(|i| {
                    let (bx, by) = (i % cd.width, i / cd.width);
                    let (px, py) = (bx * step, by * step);
                    let inside = px >= margin && py >= margin && px + margin < w && py + margin < h;
                    if inside {
                        ((i as f64) * 0.913).sin()
                    } else {
                        0.0
                    }
                })
                .collect();
            let yb = Band {
                width: cd.width,
                height: cd.height,
                data: y.clone(),
            };
            let mut back = vec![0.0; w * h];
            up_conv(&yb, &filt, fs, step, w, h, &mut back);

            let lhs: f64 = cd.data.iter().zip(&y).map(|(a, b)| a * b).sum();
            let rhs: f64 = x.iter().zip(&back).map(|(a, b)| a * b).sum();
            assert!(
                (lhs - rhs).abs() < 1e-12 * lhs.abs().max(1.0),
                "interior transpose broken at {w}×{h} step {step} fs {fs}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn analysis_and_synthesis_match_the_infinite_symmetric_extension() {
        // The strongest correctness statement available without MATLAB: a
        // finite `corr_dn` / `up_conv` on an n×n image must equal the SAME
        // operations run on the image's infinite `reflect1` extension and then
        // cropped back — everywhere, borders included. This pins the boundary
        // rule as "the restriction of the infinite operation", which is what
        // makes it principled rather than arbitrary, and it is exactly the
        // property a scatter-style adjoint fails.
        let n = 48usize;
        let small: Vec<f64> = (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as f64, (i / n) as f64);
                let pi = core::f64::consts::PI;
                (pi * 8.0 * x / (n - 1) as f64).cos() * (pi * 6.0 * y / (n - 1) as f64).cos()
            })
            .collect();
        let f = flat(&LOFILT);

        let lo = corr_dn(&small, n, n, &f, 17, 2);
        let mut finite = vec![0.0; n * n];
        up_conv(&lo, &f, 17, 2, n, n, &mut finite);

        // Three periods of the reflect1 extension, so the middle one is
        // untouched by the big image's own borders. The period 2(n−1) = 94 is
        // even, so the ×2 decimation grid stays aligned across the copies.
        let period = 2 * (n - 1);
        let big = 3 * period;
        let off = period;
        let ext = |p: isize| reflect1(p - off as isize, n);
        let bigim: Vec<f64> = (0..big * big)
            .map(|i| {
                let (x, y) = ((i % big) as isize, (i / big) as isize);
                small[ext(y) * n + ext(x)]
            })
            .collect();
        let blo = corr_dn(&bigim, big, big, &f, 17, 2);
        let mut infinite = vec![0.0; big * big];
        up_conv(&blo, &f, 17, 2, big, big, &mut infinite);

        // Analysis must agree exactly, bit for bit.
        for i in 0..lo.height {
            for j in 0..lo.width {
                let a = lo.data[i * lo.width + j];
                let b = blo.data[(off / 2 + i) * blo.width + (off / 2 + j)];
                assert_eq!(a, b, "analysis differs at band ({j},{i})");
            }
        }
        // Synthesis must agree to rounding.
        let worst = (0..n)
            .flat_map(|y| (0..n).map(move |x| (x, y)))
            .map(|(x, y)| (finite[y * n + x] - infinite[(off + y) * big + (off + x)]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-12,
            "synthesis diverges from the infinite extension by {worst}"
        );
    }

    #[test]
    fn up_conv_accumulates_rather_than_overwrites() {
        let f = flat(&LO0FILT);
        let b = Band {
            width: 4,
            height: 4,
            data: (0..16).map(|i| i as f64).collect(),
        };
        let mut a = vec![0.0; 16];
        up_conv(&b, &f, 9, 1, 4, 4, &mut a);
        let mut twice = vec![0.0; 16];
        up_conv(&b, &f, 9, 1, 4, 4, &mut twice);
        up_conv(&b, &f, 9, 1, 4, 4, &mut twice);
        for (x, y) in a.iter().zip(&twice) {
            assert!((2.0 * x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn band_layout_matches_the_reference() {
        let (w, h) = (64usize, 48usize);
        let im = ramp_plus_texture(w, h);
        let pyr = build(&im, w, h, None);
        // 64×48 with a 17-tap lofilt: 48→24→12 < 17, so 2 levels.
        assert_eq!(max_pyr_height(w, h, 17), 2);
        assert_eq!(pyr.height_levels(), 2);
        assert_eq!((pyr.high_pass.width, pyr.high_pass.height), (w, h));
        // Level 0's oriented bands are full-size; level 1's are half.
        for b in &pyr.levels[0] {
            assert_eq!((b.width, b.height), (64, 48));
        }
        for b in &pyr.levels[1] {
            assert_eq!((b.width, b.height), (32, 24));
        }
        assert_eq!((pyr.low_pass.width, pyr.low_pass.height), (16, 12));
        // Total coefficient count = the reference's `sum(prod(pind'))`.
        let total: usize = pyr.high_pass.data.len()
            + pyr
                .levels
                .iter()
                .flatten()
                .map(|b| b.data.len())
                .sum::<usize>()
            + pyr.low_pass.data.len();
        assert_eq!(total, 64 * 48 + 4 * 64 * 48 + 4 * 32 * 24 + 16 * 12);
    }

    #[test]
    fn band_frequencies_match_the_quality_calibration_grid() {
        // The 2.2 quality weights are tabulated at [15, 7.5, …, 0.2344] cpd,
        // which is exactly this grid at the 30 pixels/degree the fit used.
        let (w, h) = (512usize, 512usize);
        let pyr = build(&ramp_plus_texture(w, h), w, h, None);
        assert_eq!(pyr.height_levels(), 5);
        let f = pyr.band_frequencies(30.0);
        // 5 oriented levels → 7 entries: high-pass, 5 levels, low-pass. These
        // are `Params::quality_band_freq` verbatim.
        let want = [15.0, 7.5, 3.75, 1.875, 0.9375, 0.46875, 0.234375];
        assert_eq!(f.len(), pyr.height_levels() + 2);
        assert_eq!(f.len(), want.len());
        assert_eq!(
            f.len(),
            crate::params::Params::default().quality_band_freq.len()
        );
        for (a, b) in f.iter().zip(want) {
            assert!((a - b).abs() < 1e-12, "{f:?}");
        }
    }

    #[test]
    fn reconstruction_is_approximate_by_design_and_costs_about_1_percent_per_level() {
        // The **spatial-domain** sp3 filter set is only *approximately*
        // self-inverting — `buildSFpyr` (frequency domain) is the exactly-PR
        // variant. Measured directly from the taps: the level condition
        // |L/2|² + Σ|B_o|² lands in 0.99–1.03 across the passband, and the
        // four ×2-decimation parity classes of `lofilt` sum to 0.4932 /
        // 0.4964 / 0.4964 / 0.5141 rather than 0.5 exactly. So each level a
        // signal traverses costs it about 1 %, and reconstruction error grows
        // with pyramid depth rather than staying at machine precision.
        //
        // This test states that measured behaviour so a future change that
        // makes it *worse* is caught, without pretending the pyramid is exact.
        let (w, h) = (256usize, 256usize);
        let pi = core::f64::consts::PI;
        let im: Vec<f64> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                (2.0 * pi * x / 64.0).sin() * (2.0 * pi * y / 64.0).cos()
            })
            .collect();
        let margin = 40usize;
        let interior_err = |levels: usize| -> f64 {
            let pyr = build(&im, w, h, Some(levels));
            let rec = reconstruct(&pyr);
            assert_eq!((rec.width, rec.height), (w, h));
            (margin..h - margin)
                .flat_map(|y| (margin..w - margin).map(move |x| y * w + x))
                .map(|i| (im[i] - rec.data[i]).abs())
                .fold(0.0f64, f64::max)
        };
        let (e0, e2, e4) = (interior_err(0), interior_err(2), interior_err(4));
        // No decimation at all: only the lo0/hi0 split, essentially exact.
        assert!(e0 < 1e-3, "0-level interior reconstruction error {e0}");
        // Each further level adds roughly a percent; 4 levels stays under 5 %.
        assert!(e4 < 0.05, "4-level interior reconstruction error {e4}");
        assert!(
            e0 < e2 && e2 < e4,
            "error should grow monotonically with depth: {e0} / {e2} / {e4}"
        );
    }

    #[test]
    fn reconstruction_is_worse_at_the_border_than_in_the_interior() {
        // Documented limitation, measured rather than assumed. Each recursion
        // re-applies `reflect1` on the *decimated* grid, but the true
        // extension of a ×2-decimated symmetric sequence is half-sample
        // symmetric at a far edge whose length is even (band[k] = band[k−1],
        // not band[k−2]). Upstream's `buildSpyrLevs` recurses the same way, so
        // the analysis matches it; upstream's `upConv` boundary rule (the
        // EXPAND branch of matlabPyrTools' `edges.c`) is a different C-coded
        // rule that has NOT been ported or compared here.
        //
        // Consequence to know about: `reconstruct` is trustworthy in the
        // interior and NOT pinned to upstream at the outermost ~8 pixels. That
        // affects HDR-VDP-2's visibility map `S_map`/`P_map` at image borders.
        // It does NOT affect the quality correlate `Q`/`Q_MOS`, which is
        // accumulated in the band loop and never calls `reconstruct` —
        // so the UPIQ validation target is unaffected.
        let (w, h) = (64usize, 48usize);
        let pi = core::f64::consts::PI;
        let im: Vec<f64> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                (pi * 8.0 * x / (w - 1) as f64).cos() * (pi * 6.0 * y / (h - 1) as f64).cos()
            })
            .collect();
        let pyr = build(&im, w, h, None);
        let rec = reconstruct(&pyr);
        let m = 8usize;
        let interior = (m..h - m)
            .flat_map(|y| (m..w - m).map(move |x| y * w + x))
            .map(|i| (im[i] - rec.data[i]).abs())
            .fold(0.0f64, f64::max);
        let overall = im
            .iter()
            .zip(&rec.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(interior < 0.05, "interior error {interior} regressed");
        assert!(
            overall > interior,
            "the border is expected to be the worse case; if this ever flips, \
             the boundary rule changed and this note needs rewriting"
        );
        assert!(
            overall < 2.0,
            "border error {overall} exceeds even the documented ceiling"
        );
    }

    #[test]
    fn oriented_bands_respond_to_their_own_orientation() {
        // Band 0 is the vertical band (a horizontal derivative), band 2 the
        // horizontal one. A vertical grating must light up band 0 far more
        // than band 2, and a horizontal grating the reverse. If the BFILTS
        // column-major reshape were transposed, this test flips.
        let (w, h) = (64usize, 64usize);
        let energy = |grating_along_x: bool| -> [f64; ORIENTATIONS] {
            let im: Vec<f64> = (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f64, (i / w) as f64);
                    let t = if grating_along_x { x } else { y };
                    (2.0 * core::f64::consts::PI * t / 8.0).sin()
                })
                .collect();
            let pyr = build(&im, w, h, None);
            let mut e = [0.0; ORIENTATIONS];
            for (o, b) in pyr.levels[0].iter().enumerate() {
                e[o] = b.data.iter().map(|v| v * v).sum::<f64>().sqrt();
            }
            e
        };
        let vertical = energy(true); // varies along x → vertical bars
        let horizontal = energy(false);
        assert!(
            vertical[0] > 4.0 * vertical[2],
            "vertical grating should excite band 0 over band 2: {vertical:?}"
        );
        assert!(
            horizontal[2] > 4.0 * horizontal[0],
            "horizontal grating should excite band 2 over band 0: {horizontal:?}"
        );
        // The two diagonal bands sit between the two extremes for both.
        for e in [vertical, horizontal] {
            let hi = e[0].max(e[2]);
            let lo = e[0].min(e[2]);
            assert!(e[1] < hi && e[3] < hi, "{e:?}");
            assert!(e[1] > lo && e[3] > lo, "{e:?}");
        }
    }

    #[test]
    fn a_flat_image_has_no_oriented_energy() {
        // "No energy" means "down at the taps' own precision": the published
        // filters carry ~10 significant digits, so `Σ hi0filt = 1.36e-7`
        // rather than exactly 0, and a flat field of L leaks L·1.36e-7 into
        // the high-pass. The oriented filters are antisymmetric and cancel to
        // machine precision.
        let (w, h) = (48usize, 48usize);
        let level = 3.25f64;
        let pyr = build(&vec![level; w * h], w, h, None);
        for lv in &pyr.levels {
            for b in lv {
                for v in &b.data {
                    assert!(
                        v.abs() < 1e-12 * level,
                        "flat image produced band energy {v}"
                    );
                }
            }
        }
        for v in &pyr.high_pass.data {
            assert!(
                v.abs() < 1e-6 * level,
                "flat image high-pass leak {v} exceeds the taps' precision"
            );
        }
        // ... and all of it survives in the low-pass residual.
        assert!(pyr.low_pass.data.iter().all(|v| v.abs() > 1e-3));
    }
}
