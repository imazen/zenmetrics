//! The FSIM/FSIMc pipeline, ported pass-for-pass from the authors'
//! `FR_FSIMc.m` (the "FSIM Index with automatic downsampling, Version
//! 1.0" reference — phasecong2 and lowpassfilter are Peter Kovesi's code
//! embedded in that file, MIT-style licensed: notice in `LICENSE-kovesi`).
//!
//! Precision layout: f32 planes end-to-end (the vendored FFT is f32),
//! f64 for every scalar reduction (medians, noise sums, the pooling
//! numerator/denominator) in fixed raster order. The element-wise map
//! math goes through `*_body!` macro instantiations so every SIMD tier
//! computes the identical per-pixel expression; everything outside a
//! body (FFT, filter construction, conv borders, reductions) is shared
//! scalar code and therefore tier-independent by construction.

use alloc::vec;
use alloc::vec::Vec;

use crate::fft::{Complex, Plan, fft2_planned, ifft2_planned};
use crate::{I_COEF, Q_COEF, Y_COEF, plane_from_rgb8};

// ------------------------------------------------------------------
// Constants (verbatim from FR_FSIMc.m / phasecong2).
// ------------------------------------------------------------------

/// `nscale` — number of wavelet scales in the filterbank.
const NSCALE: usize = 4;
/// `norient` — number of filter orientations.
const NORIENT: usize = 4;
/// `minWaveLength` — wavelength of the smallest-scale filter.
const MIN_WAVELENGTH: f64 = 6.0;
/// `mult` — scaling factor between successive filters.
const MULT: f64 = 2.0;
/// `sigmaOnf` — log-Gabor bandwidth (sigma / centre frequency).
const SIGMA_ONF: f64 = 0.55;
/// `dThetaOnSigma` — angular interval / angular sigma ratio.
const DTHETA_ON_SIGMA: f64 = 1.2;
/// `k` — noise threshold in sigma units beyond the Rayleigh mean.
const K_NOISE: f64 = 2.0;
/// `epsilon` — added to `XEnergy` to prevent division by zero.
const EPSILON: f32 = 0.0001;
/// Low-pass pre-filter: Butterworth `1/(1+(r/cutoff)^(2n))`,
/// `cutoff = .45`, `n = 15`.
const LP_CUTOFF: f64 = 0.45;
/// Butterworth order of the low-pass pre-filter.
const LP_ORDER: f64 = 15.0;
/// `T1` — phase-congruency similarity constant.
const T1: f32 = 0.85;
/// `T2` — gradient similarity constant.
const T2: f32 = 160.0;
/// `T3` — I-channel similarity constant.
const T3: f32 = 200.0;
/// `T4` — Q-channel similarity constant.
const T4: f32 = 200.0;
/// `lambda` — FSIMc chroma exponent.
const LAMBDA: f64 = 0.03;
/// `cos(π·λ)` — the real part of `(-1)^λ`, used by `real(z^λ)` for
/// negative chroma products exactly as MATLAB's element-wise `.^`
/// complex power produces.
const COS_PI_LAMBDA: f64 = 0.99556196460308; // cos(0.03·π)

/// Scharr-style gradient kernels from the reference:
/// `dx = [3 0 -3; 10 0 -10; 3 0 -3]/16`, `dy` its transpose.
/// `conv2 'same'` applies the kernel flipped — `out(i,j) =
/// Σ K[a,b]·Y(i+1−a, j+1−b)` with zero outside.
const DX: [[f32; 3]; 3] = [
    [3.0 / 16.0, 0.0, -3.0 / 16.0],
    [10.0 / 16.0, 0.0, -10.0 / 16.0],
    [3.0 / 16.0, 0.0, -3.0 / 16.0],
];
const DY: [[f32; 3]; 3] = [
    [3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0],
    [0.0, 0.0, 0.0],
    [-3.0 / 16.0, -10.0 / 16.0, -3.0 / 16.0],
];

// ------------------------------------------------------------------
// Vector leaf bodies — instantiated per tier inside `score_body!`.
// Planes here are unpadded `Vec<f32>`/`Vec<Complex>` of `n = w·h`.
// ------------------------------------------------------------------

/// `se[i] += e[i].re; so[i] += e[i].im; sa[i] += |e[i]|` — the
/// per-scale even/odd/amplitude accumulation of one orientation.
macro_rules! anacc_body {
    ($token:ident, $F32:ident, $LANES:literal, $e:ident, $se:ident, $so:ident, $sa:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let re = $F32::from_array_t($token, core::array::from_fn(|k| $e[x + k].re));
            let im = $F32::from_array_t($token, core::array::from_fn(|k| $e[x + k].im));
            let vse = $F32::from_array_t($token, core::array::from_fn(|k| $se[x + k]));
            let vso = $F32::from_array_t($token, core::array::from_fn(|k| $so[x + k]));
            let vsa = $F32::from_array_t($token, core::array::from_fn(|k| $sa[x + k]));
            (vse + re).store((&mut $se[x..x + $LANES]).try_into().unwrap());
            (vso + im).store((&mut $so[x..x + $LANES]).try_into().unwrap());
            let an = (re * re + im * im).sqrt();
            (vsa + an).store((&mut $sa[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let (re, im) = ($e[x].re, $e[x].im);
            $se[x] += re;
            $so[x] += im;
            $sa[x] += (re * re + im * im).sqrt();
            x += 1;
        }
    }};
}

/// `xe[i] = sqrt(se²+so²)+eps; me[i] = se/xe; mo[i] = so/xe` — the
/// weighted mean phase-angle unit vector of one orientation.
macro_rules! xenergy_body {
    ($token:ident, $F32:ident, $LANES:literal, $se:ident, $so:ident, $me:ident, $mo:ident, $n:ident) => {{
        let n = $n;
        let eps = $F32::splat_t($token, EPSILON);
        let mut x = 0usize;
        while x + $LANES <= n {
            let vse = $F32::from_array_t($token, core::array::from_fn(|k| $se[x + k]));
            let vso = $F32::from_array_t($token, core::array::from_fn(|k| $so[x + k]));
            let xe = (vse * vse + vso * vso).sqrt() + eps;
            (vse / xe).store((&mut $me[x..x + $LANES]).try_into().unwrap());
            (vso / xe).store((&mut $mo[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let xe = ($se[x] * $se[x] + $so[x] * $so[x]).sqrt() + EPSILON;
            $me[x] = $se[x] / xe;
            $mo[x] = $so[x] / xe;
            x += 1;
        }
    }};
}

/// `energy[i] += e.re·me + e.im·mo − |e.re·mo − e.im·me|` — the
/// `An(cos Δφ − |sin Δφ|)` accumulation of one scale.
macro_rules! energy_body {
    ($token:ident, $F32:ident, $LANES:literal, $e:ident, $me:ident, $mo:ident, $energy:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let re = $F32::from_array_t($token, core::array::from_fn(|k| $e[x + k].re));
            let im = $F32::from_array_t($token, core::array::from_fn(|k| $e[x + k].im));
            let vme = $F32::from_array_t($token, core::array::from_fn(|k| $me[x + k]));
            let vmo = $F32::from_array_t($token, core::array::from_fn(|k| $mo[x + k]));
            let prev = $F32::from_array_t($token, core::array::from_fn(|k| $energy[x + k]));
            let term = (re * vme + im * vmo) - (re * vmo - im * vme).abs();
            (prev + term).store((&mut $energy[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let (re, im) = ($e[x].re, $e[x].im);
            $energy[x] += re * $me[x] + im * $mo[x] - (re * $mo[x] - im * $me[x]).abs();
            x += 1;
        }
    }};
}

/// `eall[i] += max(energy[i] − t, 0); aall[i] += sa[i]` — noise
/// thresholding and the cross-orientation fold.
macro_rules! ethr_body {
    ($token:ident, $F32:ident, $LANES:literal, $energy:ident, $sa:ident, $eall:ident, $aall:ident, $t:ident, $n:ident) => {{
        let n = $n;
        let tv = $F32::splat_t($token, $t);
        let zv = $F32::splat_t($token, 0.0);
        let mut x = 0usize;
        while x + $LANES <= n {
            let e = $F32::from_array_t($token, core::array::from_fn(|k| $energy[x + k]));
            let sa = $F32::from_array_t($token, core::array::from_fn(|k| $sa[x + k]));
            let pe = $F32::from_array_t($token, core::array::from_fn(|k| $eall[x + k]));
            let pa = $F32::from_array_t($token, core::array::from_fn(|k| $aall[x + k]));
            (pe + (e - tv).max(zv)).store((&mut $eall[x..x + $LANES]).try_into().unwrap());
            (pa + sa).store((&mut $aall[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            $eall[x] += ($energy[x] - $t).max(0.0);
            $aall[x] += $sa[x];
            x += 1;
        }
    }};
}

/// `pc[i] = eall[i] / aall[i]` — the finished phase-congruency map.
macro_rules! pcdiv_body {
    ($token:ident, $F32:ident, $LANES:literal, $eall:ident, $aall:ident, $pc:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let e = $F32::from_array_t($token, core::array::from_fn(|k| $eall[x + k]));
            let a = $F32::from_array_t($token, core::array::from_fn(|k| $aall[x + k]));
            (e / a).store((&mut $pc[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            $pc[x] = $eall[x] / $aall[x];
            x += 1;
        }
    }};
}

/// `g[i] = sqrt(ix[i]² + iy[i]²)` — gradient magnitude.
macro_rules! grad_body {
    ($token:ident, $F32:ident, $LANES:literal, $ix:ident, $iy:ident, $g:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let vx = $F32::from_array_t($token, core::array::from_fn(|k| $ix[x + k]));
            let vy = $F32::from_array_t($token, core::array::from_fn(|k| $iy[x + k]));
            (vx * vx + vy * vy)
                .sqrt()
                .store((&mut $g[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            $g[x] = ($ix[x] * $ix[x] + $iy[x] * $iy[x]).sqrt();
            x += 1;
        }
    }};
}

// ------------------------------------------------------------------
// Shared scalar pieces (identical on every tier — no token needed).
// ------------------------------------------------------------------

/// `F = max(1, round(min(w,h)/256))` — the reference's automatic
/// downsampling factor. Integer-exact: `round` on a positive real is
/// `(2m + 256) / 512` floored, i.e. `(m + 128) / 256` for `m ≥ 0`.
pub(crate) fn downsample_factor(w: usize, h: usize) -> usize {
    ((w.min(h) + 128) / 256).max(1)
}

/// `conv2(x, ones(F,F)/F², 'same')` then `x(1:F:end, 1:F:end)` — the
/// reference's averaging downsample. The box kernel is separable and
/// symmetric, so the conv is a horizontal then vertical running sum
/// with the `same` anchor `s = F/2` (forward-looking for even F,
/// centred for odd F), zero outside.
fn decimate_plane(x: &[f32], w: usize, h: usize, f: usize) -> (usize, usize, Vec<f32>) {
    if f == 1 {
        return (w, h, x.to_vec());
    }
    let s = f / 2;
    let w2 = w.div_ceil(f);
    let h2 = h.div_ceil(f);
    // Keep only the horizontal sums consumed by vertical decimation.
    // Each retained window still accumulates left to right, un-normalised.
    let mut tmp = vec![0.0f32; w2 * h];
    for i in 0..h {
        let row = &x[i * w..(i + 1) * w];
        let trow = &mut tmp[i * w2..(i + 1) * w2];
        for (oj, t) in trow.iter_mut().enumerate() {
            let j = oj * f;
            let mut acc = 0.0f32;
            // out[i,j] = Σ_b x[i, j+s−b] over b ∈ [0, f)
            let lo = (j + s).saturating_sub(f - 1);
            let hi = (j + s).min(w - 1);
            for v in row[lo..=hi].iter() {
                acc += v;
            }
            *t = acc;
        }
    }
    // Vertical box sums at the decimated positions only, normalised.
    let mut out = vec![0.0f32; w2 * h2];
    let inv = 1.0 / (f as u64 * f as u64) as f32;
    for oi in 0..h2 {
        let i = oi * f;
        let lo = (i + s).saturating_sub(f - 1);
        let hi = (i + s).min(h - 1);
        for oj in 0..w2 {
            let mut acc = 0.0f32;
            for r in lo..=hi {
                acc += tmp[r * w2 + oj];
            }
            out[oi * w2 + oj] = acc * inv;
        }
    }
    (w2, h2, out)
}

fn for_each_compact_row<T: Send>(
    out: &mut [T],
    width: usize,
    op: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    if out.len() >= 1 << 16 && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        out.par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| op(y, row));
        return;
    }
    for (y, row) in out.chunks_mut(width).enumerate() {
        op(y, row);
    }
}

/// One horizontal window for a single decimated output — the reference's
/// per-pixel `sums[c] += coef·rgb` accumulation order. Used for edge and
/// tail outputs; the interior runs through [`decimate_hwin_rgb8_inner`].
#[inline]
fn decimate_hwin_one<const N: usize>(
    pixels: &[u8],
    w: usize,
    f: usize,
    oj: usize,
    coefficients: &[[f32; 3]; N],
    out: &mut [f32; N],
) {
    let s = f / 2;
    let j = oj * f;
    let lo = (j + s).saturating_sub(f - 1);
    let hi = (j + s).min(w - 1);
    for px in pixels[lo * 3..(hi + 1) * 3].as_chunks::<3>().0 {
        let [r, g, b] = px.map(f32::from);
        for c in 0..N {
            let coef = coefficients[c];
            out[c] += coef[0] * r + coef[1] * g + coef[2] * b;
        }
    }
}

/// Eight interior (full-width) horizontal windows at once — lanes are
/// consecutive decimated outputs. Lane `j` accumulates its window's
/// pixels in the same left-to-right `acc += coef·rgb` order as
/// [`decimate_hwin_one`], so results are bit-identical. `row` is the
/// flat `[f32]` view of the interior output run (stride `n_ch`).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn decimate_hwin_rgb8_inner(
    token: Token,
    pixels: &[u8],
    row: &mut [f32],
    n_ch: usize,
    first_oj: usize,
    f: usize,
    coefficients: &[[f32; 3]],
) {
    let s = f / 2;
    let zero = f32x8::splat_t(token, 0.0);
    let ngroups = row.len() / n_ch / 8;
    let cs: [[f32x8; 3]; 3] = core::array::from_fn(|c| {
        let k = coefficients.get(c).copied().unwrap_or([0.0; 3]);
        [
            f32x8::splat_t(token, k[0]),
            f32x8::splat_t(token, k[1]),
            f32x8::splat_t(token, k[2]),
        ]
    });
    for g in 0..ngroups {
        let oj0 = first_oj + g * 8;
        // Interior windows tile exactly: lane j's t-th source pixel is
        // at x = (oj0+j)·f + s − (f−1) + t, i.e. stride f across lanes.
        let x0 = oj0 * f + s + 1 - f;
        let mut acc: [f32x8; 3] = core::array::from_fn(|_| zero);
        for t in 0..f {
            let ra: [f32; 8] = core::array::from_fn(|j| pixels[(x0 + j * f + t) * 3] as f32);
            let ga: [f32; 8] = core::array::from_fn(|j| pixels[(x0 + j * f + t) * 3 + 1] as f32);
            let ba: [f32; 8] = core::array::from_fn(|j| pixels[(x0 + j * f + t) * 3 + 2] as f32);
            let rv = f32x8::load_t(token, &ra);
            let gv = f32x8::load_t(token, &ga);
            let bv = f32x8::load_t(token, &ba);
            for c in 0..n_ch {
                acc[c] = acc[c] + (cs[c][0] * rv + cs[c][1] * gv + cs[c][2] * bv);
            }
        }
        for c in 0..n_ch {
            let mut tmp = [0.0f32; 8];
            acc[c].store(&mut tmp);
            for j in 0..8 {
                row[(g * 8 + j) * n_ch + c] = tmp[j];
            }
        }
    }
}

/// Horizontal decimation pass over one source row: interior outputs go
/// through the batched kernel, edge and tail outputs stay scalar.
fn decimate_hwin_rgb8<const N: usize>(
    pixels: &[u8],
    w: usize,
    f: usize,
    coefficients: &[[f32; 3]; N],
    row: &mut [[f32; N]],
) {
    let s = f / 2;
    let w2 = row.len();
    // Interior outputs have an unclamped f-pixel window:
    // `oj·f + s ≥ f−1` on the left and `oj·f + s ≤ w−1` on the right.
    let first = f.saturating_sub(1 + s).div_ceil(f).min(w2);
    let last = w
        .checked_sub(1 + s)
        .map_or(0, |v| v / f + 1)
        .min(w2)
        .max(first);
    for oj in 0..first {
        decimate_hwin_one(pixels, w, f, oj, coefficients, &mut row[oj]);
    }
    let flat = row[first..last].as_flattened_mut();
    let nfull = (flat.len() / N / 8) * 8;
    archmage::attuned!(
        decimate_hwin_rgb8_inner(
            pixels,
            &mut flat[..nfull * N],
            N,
            first,
            f,
            coefficients.as_slice()
        ),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for k in nfull..(last - first) {
        decimate_hwin_one(pixels, w, f, first + k, coefficients, &mut row[first + k]);
    }
    for oj in last..w2 {
        decimate_hwin_one(pixels, w, f, oj, coefficients, &mut row[oj]);
    }
}

// Convert each RGB pixel inside its retained horizontal window. This keeps
// the same rounded color values and sum order without full-size YIQ planes.
fn decimate_rgb8<const N: usize>(
    data: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    f: usize,
    coefficients: [[f32; 3]; N],
) -> (usize, usize, [Vec<f32>; N]) {
    if f == 1 {
        return (
            w,
            h,
            core::array::from_fn(|c| plane_from_rgb8(data, w, h, stride, coefficients[c])),
        );
    }
    let s = f / 2;
    let w2 = w.div_ceil(f);
    let h2 = h.div_ceil(f);
    let mut horizontal = vec![[0.0f32; N]; w2 * h];
    for_each_compact_row(&mut horizontal, w2, |y, row| {
        let pixels = &data[y * stride..y * stride + w * 3];
        decimate_hwin_rgb8(pixels, w, f, &coefficients, row);
    });
    let inv = 1.0 / (f as u64 * f as u64) as f32;
    let mut compact = vec![[0.0f32; N]; w2 * h2];
    for_each_compact_row(&mut compact, w2, |oi, row| {
        let i = oi * f;
        let lo = (i + s).saturating_sub(f - 1);
        let hi = (i + s).min(h - 1);
        for (oj, values) in row.iter_mut().enumerate() {
            for r in lo..=hi {
                for c in 0..N {
                    values[c] += horizontal[r * w2 + oj][c];
                }
            }
            for value in values {
                *value *= inv;
            }
        }
    });
    let planes = core::array::from_fn(|c| compact.iter().map(|pixel| pixel[c]).collect());
    (w2, h2, planes)
}

#[cfg(test)]
mod decimation_tests {
    use super::*;

    #[test]
    fn fused_rgb_windows_match_materialized_yiq_bits() {
        for (w, h) in [
            (1, 1),
            (1, 17),
            (19, 1),
            (7, 11),
            (32, 29),
            (65, 63),
            (1009, 171),
        ] {
            let stride = w * 3 + 13;
            let mut rgb = vec![0xa5; stride * (h - 1) + w * 3];
            for y in 0..h {
                for x in 0..w * 3 {
                    rgb[y * stride + x] = ((x * 37 + y * 19) ^ (x * y % 251)) as u8;
                }
            }
            for f in 1..=12 {
                let (fw, fh, got) = decimate_rgb8(&rgb, w, h, stride, f, [Y_COEF, I_COEF, Q_COEF]);
                let (lw, lh, [luma]) = decimate_rgb8(&rgb, w, h, stride, f, [Y_COEF]);
                assert_eq!((lw, lh), (fw, fh));
                for (c, coef) in [Y_COEF, I_COEF, Q_COEF].into_iter().enumerate() {
                    let source = plane_from_rgb8(&rgb, w, h, stride, coef);
                    let (ew, eh, expected) = decimate_plane(&source, w, h, f);
                    assert_eq!((fw, fh), (ew, eh));
                    for (i, (&actual, &old)) in got[c].iter().zip(&expected).enumerate() {
                        assert_eq!(
                            actual.to_bits(),
                            old.to_bits(),
                            "{w}x{h} f={f} channel={c} pixel={i}"
                        );
                        if c == 0 {
                            assert_eq!(
                                luma[i].to_bits(),
                                old.to_bits(),
                                "luma {w}x{h} f={f} pixel={i}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn retained_columns_match_full_box_filter_bits() {
        for (w, h) in [(1, 1), (1, 17), (19, 1), (7, 11), (32, 29), (65, 63)] {
            let x: Vec<f32> = (0..w * h)
                .map(|i| ((i * 37 % 251) as f32 - 125.0) / ((i % 7 + 1) as f32))
                .collect();
            for f in 1..=12 {
                let (dw, dh, actual) = decimate_plane(&x, w, h, f);
                assert_eq!((dw, dh), (w.div_ceil(f), h.div_ceil(f)));
                for oy in 0..dh {
                    for ox in 0..dw {
                        let mut expected = 0.0f32;
                        for dy in 0..f {
                            let y = (oy * f + f / 2) as isize - (f - 1) as isize + dy as isize;
                            if !(0..h as isize).contains(&y) {
                                continue;
                            }
                            let mut row = 0.0f32;
                            for dx in 0..f {
                                let xx = (ox * f + f / 2) as isize - (f - 1) as isize + dx as isize;
                                if (0..w as isize).contains(&xx) {
                                    row += x[y as usize * w + xx as usize];
                                }
                            }
                            expected += row;
                        }
                        expected *= 1.0 / (f as u64 * f as u64) as f32;
                        assert_eq!(
                            actual[oy * dw + ox].to_bits(),
                            expected.to_bits(),
                            "{w}x{h}, F={f}, ({ox},{oy})"
                        );
                    }
                }
            }
        }
    }
}

/// `conv2(Y, K, 'same')` for a 3×3 kernel — `out(i,j) =
/// Σ_{a,b} K[a,b]·Y(i+1−a, j+1−b)`, zero outside (the kernel is
/// applied flipped, as MATLAB's `conv2` does).
/// `rayon::join` under `parallel`, sequential otherwise — each side
/// produces its own `Vec`, so results are identical at any thread count.
fn maybe_join<A, B>(fa: impl FnOnce() -> A + Send, fb: impl FnOnce() -> B + Send) -> (A, B)
where
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    {
        if rayon::current_num_threads() <= 1 {
            return (fa(), fb());
        }
        rayon::join(fa, fb)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (fa(), fb())
    }
}

/// One output element of `conv2_same_3x3` — the per-tap `acc += k·y`
/// accumulation with edge skips, identical for every driver.
#[inline]
#[allow(clippy::too_many_arguments)]
fn conv2_same_3x3_one(
    o: &mut f32,
    y: &[f32],
    w: usize,
    h: usize,
    i: usize,
    j: usize,
    k: &[[f32; 3]; 3],
) {
    let mut acc = 0.0f32;
    for a in 0..3isize {
        let r = i as isize + 1 - a;
        if r < 0 || r >= h as isize {
            continue;
        }
        for b in 0..3isize {
            let c = j as isize + 1 - b;
            if c < 0 || c >= w as isize {
                continue;
            }
            acc += k[a as usize][b as usize] * y[r as usize * w + c as usize];
        }
    }
    *o = acc;
}

/// Eight interior outputs of one `conv2_same_3x3` row — lanes are
/// consecutive columns; every tap is an unaligned contiguous load and
/// `acc += k·y` keeps the same (a,b) order per lane, so results are
/// bit-identical to [`conv2_same_3x3_one`]. `out[t]` is column `j0+t`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn conv2_same_3x3_row_inner(
    token: Token,
    out: &mut [f32],
    y: &[f32],
    w: usize,
    i: usize,
    j0: usize,
    k: &[[f32; 3]; 3],
) {
    let ks: [[f32x8; 3]; 3] =
        core::array::from_fn(|a| core::array::from_fn(|b| f32x8::splat_t(token, k[a][b])));
    let zero = f32x8::splat_t(token, 0.0);
    let nfull = out.len() & !7;
    let mut t = 0;
    while t < nfull {
        let j = j0 + t;
        let mut acc = zero;
        for a in 0..3 {
            let r = i + 1 - a;
            for b in 0..3 {
                let base = r * w + j + 1 - b;
                let v = f32x8::load_t(token, (&y[base..base + 8]).try_into().unwrap());
                acc = acc + ks[a][b] * v;
            }
        }
        acc.store((&mut out[t..t + 8]).try_into().unwrap());
        t += 8;
    }
}

/// One output row of `conv2_same_3x3` — per-element tap order identical
/// under either driver (reads at most the two neighbouring source rows).
fn conv2_same_3x3_row(out: &mut [f32], y: &[f32], w: usize, h: usize, i: usize, k: &[[f32; 3]; 3]) {
    // Edge rows (and narrow planes) keep the per-tap skip loop scalar.
    if i == 0 || i + 1 == h || w < 3 {
        for (j, o) in out.iter_mut().enumerate() {
            conv2_same_3x3_one(o, y, w, h, i, j, k);
        }
        return;
    }
    conv2_same_3x3_one(&mut out[0], y, w, h, i, 0, k);
    conv2_same_3x3_one(&mut out[w - 1], y, w, h, i, w - 1, k);
    let interior = &mut out[1..w - 1];
    let nfull = interior.len() & !7;
    archmage::attuned!(
        conv2_same_3x3_row_inner(&mut interior[..nfull], y, w, i, 1, k),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for j in 1 + nfull..w - 1 {
        conv2_same_3x3_one(&mut out[j], y, w, h, i, j, k);
    }
}

fn conv2_same_3x3(y: &[f32], w: usize, h: usize, k: &[[f32; 3]; 3]) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        out.par_chunks_mut(w)
            .enumerate()
            .for_each(|(i, row)| conv2_same_3x3_row(row, y, w, h, i, k));
        return out;
    }
    for (i, row) in out.chunks_mut(w).enumerate() {
        conv2_same_3x3_row(row, y, w, h, i, k);
    }
    out
}

/// The frequency grids `phasecong2` builds once per size:
/// `(radius, sin θ, cos θ)` in `ifftshift`ed DFT-bin order with the
/// `radius(1,1) = 1` DC fudge applied. `x,y` are the ±0.5-normalised
/// centred coordinates; `theta = atan2(−y, x)`.
fn freq_grids(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Centred normalised coordinates, then ifftshift: the value at
    // index j is `cen[(j + n/2) % n]` — matches MATLAB's
    // `ifftshift` for both parities.
    let xcoord: Vec<f64> = (0..w)
        .map(|j| {
            let c = (j + w / 2) % w;
            if w.is_multiple_of(2) {
                (c as f64 - w as f64 / 2.0) / w as f64
            } else {
                (c as f64 - (w as f64 - 1.0) / 2.0) / (w as f64 - 1.0)
            }
        })
        .collect();
    let ycoord: Vec<f64> = (0..h)
        .map(|i| {
            let c = (i + h / 2) % h;
            if h.is_multiple_of(2) {
                (c as f64 - h as f64 / 2.0) / h as f64
            } else {
                (c as f64 - (h as f64 - 1.0) / 2.0) / (h as f64 - 1.0)
            }
        })
        .collect();
    let mut radius = vec![0.0f32; w * h];
    let mut sintheta = vec![0.0f32; w * h];
    let mut costheta = vec![0.0f32; w * h];
    for (i, &yc) in ycoord.iter().enumerate() {
        for (j, &xc) in xcoord.iter().enumerate() {
            let r = libm::sqrt(xc * xc + yc * yc);
            let t = libm::atan2(-yc, xc);
            let p = i * w + j;
            radius[p] = r as f32;
            sintheta[p] = libm::sin(t) as f32;
            costheta[p] = libm::cos(t) as f32;
        }
    }
    radius[0] = 1.0; // DC fudge so log(radius) never diverges.
    (radius, sintheta, costheta)
}

/// The radial filter components: `lp` and `logGabor[s]` for
/// `s ∈ [0, nscale)` — `lg = exp(−log(r/fo)² / (2·log σ²))·lp`,
/// `fo = 1/(minWaveLength·mult^s)`, DC reset to 0.
fn radial_filters(w: usize, h: usize, radius: &[f32]) -> [Vec<f32>; NSCALE] {
    let n = w * h;
    let mut lg: [Vec<f32>; NSCALE] = core::array::from_fn(|_| vec![0.0; n]);
    let log_sigma = libm::log(SIGMA_ONF);
    let mut a64 = vec![0.0f64; n];
    let mut l64 = vec![0.0f64; n];
    let mut p64 = vec![0.0f64; n];
    {
        let mut pw_in = vec![0.0f64; n];
        for i in 0..n {
            pw_in[i] = radius[i] as f64 / LP_CUTOFF;
        }
        // lp = 1/(1 + pow(r/LP_CUTOFF, 2·LP_ORDER)) — the pow is
        // scale-invariant, so it runs on lanes once (bit-identical to
        // `libm::pow` per element).
        zenmetrics_math::vpow64_libm_into(&pw_in, &mut p64, 2.0 * LP_ORDER);
    }
    for (s, lg_s) in lg.iter_mut().enumerate() {
        let fo = 1.0 / (MIN_WAVELENGTH * libm::pow(MULT, s as f64));
        for i in 0..n {
            a64[i] = radius[i] as f64 / fo;
        }
        // l = log(r / fo) on lanes — bit-identical to `libm::log`.
        zenmetrics_math::vlog_libm_into(&a64, &mut l64);
        for i in 0..n {
            let l = l64[i];
            let lp = 1.0 / (1.0 + p64[i]);
            lg_s[i] = (libm::exp(-(l * l) / (2.0 * log_sigma * log_sigma)) * lp) as f32;
        }
        lg_s[0] = 0.0; // undo the radius fudge at DC.
    }
    lg
}

/// The angular filter components: `spread[o]` for `o ∈ [0, norient)`,
/// `spread = exp(−dθ² / (2·θσ²))` with `θσ = π/norient/dThetaOnSigma`
/// and `dθ = |atan2(sinΔ, cosΔ)|` the wrapped angular distance to
/// `angl = o·π/norient`.
fn angular_filters(w: usize, h: usize, sintheta: &[f32], costheta: &[f32]) -> [Vec<f32>; NORIENT] {
    let n = w * h;
    let theta_sigma = core::f64::consts::PI / NORIENT as f64 / DTHETA_ON_SIGMA;
    let denom = 2.0 * theta_sigma * theta_sigma;
    let mut spread: [Vec<f32>; NORIENT] = core::array::from_fn(|_| vec![0.0; n]);
    for (o, sp) in spread.iter_mut().enumerate() {
        let angl = o as f64 * core::f64::consts::PI / NORIENT as f64;
        let (sa, ca) = (libm::sin(angl), libm::cos(angl));
        for i in 0..n {
            let (st, ct) = (sintheta[i] as f64, costheta[i] as f64);
            let ds = st * ca - ct * sa;
            let dc = ct * ca + st * sa;
            let dtheta = libm::fabs(libm::atan2(ds, dc));
            sp[i] = libm::exp(-(dtheta * dtheta) / denom) as f32;
        }
    }
    spread
}

/// MATLAB `median` of a f32 vector: mean of the two middle elements
/// for even counts, the middle for odd — computed in f64. Uses O(n)
/// selection instead of a full sort; the selected values are identical
/// to the sorted-order medians, so results are bit-identical.
fn median_f32(v: &mut [f32]) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        *v.select_nth_unstable_by(n / 2, f32::total_cmp).1 as f64
    } else {
        let hi = *v.select_nth_unstable_by(n / 2, f32::total_cmp).1;
        // Everything at index ≥ n/2 is ≥ hi; the lower-middle element
        // is the max of the strict lower half under total_cmp order.
        let lo = *v[..n / 2]
            .select_nth_unstable_by(n / 2 - 1, f32::total_cmp)
            .1;
        (lo as f64 + hi as f64) / 2.0
    }
}

// ------------------------------------------------------------------
// `real(z^λ)` for the FSIMc chroma product, computed as
// `pow(|z|,λ)·(z<0 ? cos(πλ) : 1)` — MATLAB's `.^` complex power — with
// the f64 `pow` batched four lanes wide by `pow64_x4` in the pooling
// fold (the tail runs scalar `libm::pow`, same bits per lane). The lane
// port lives in `zenmetrics-math`'s `fdlibm64` (a full lane-parallel
// transcription of `libm`'s `e_pow` onto `f64x4`/`i64x4`, bit-identical
// per lane on every input).
// ------------------------------------------------------------------

/// `pow(xa[i], ya[i])` for four lanes, bit-identical to `libm::pow` —
/// wraps `zenmetrics_math::fdlibm64::pow64_libm_4x`.
#[inline(always)]
fn pow64_x4<T: magetypes::simd::backends::I64x4Bitcast>(
    token: T,
    xa: &[f64; 4],
    ya: &[f64; 4],
) -> [f64; 4] {
    use magetypes::simd::generic::f64x4;
    zenmetrics_math::fdlibm64::pow64_libm_4x(
        token,
        f64x4::<T>::from_array_t(token, *xa),
        f64x4::<T>::from_array_t(token, *ya),
    )
    .to_array()
}

/// Everything in the phase-congruency pipeline that is a pure function of
/// `(w, h)`: the log-Gabor×spread filter planes, their `real(ifft2·√n)`
/// planes (`iff`), and the per-orientation `em_n`/`sum_sa2`/`sum_saij`
/// aggregates. Building this per call dominated FSIM's setup — ~5M libm
/// calls plus `NSCALE·NORIENT` iffts — and is now cached per image size
/// (single-entry; repeated same-size calls reuse it).
struct FilterBank {
    /// `logg[s] · spread[o]` — the per-(o,s) filter, `[o][s]`.
    filters: [[Vec<f32>; NSCALE]; NORIENT],
    /// `Σ filters[o][0]²` (the s=0 filter energy per orientation).
    em_n: [f64; NORIENT],
    /// `Σ_s Σ_i iff[o][s][i]²` per orientation.
    sum_sa2: [f64; NORIENT],
    /// `Σ_{si<sj} Σ_i iff[o][si][i]·iff[o][sj][i]` per orientation.
    sum_saij: [f64; NORIENT],
    wplan: Plan,
    hplan: Plan,
}

impl FilterBank {
    fn build(w: usize, h: usize) -> Self {
        let n = w * h;
        let sqrt_n = libm::sqrt(n as f64) as f32;
        let (radius, sintheta, costheta) = freq_grids(w, h);
        let logg = radial_filters(w, h, &radius);
        let spread = angular_filters(w, h, &sintheta, &costheta);
        let (wplan, hplan) = (Plan::new(w), Plan::new(h));
        let mut filters: [[Vec<f32>; NSCALE]; NORIENT] =
            core::array::from_fn(|_| core::array::from_fn(|_| Vec::new()));
        let mut iff: [[Vec<f32>; NSCALE]; NORIENT] =
            core::array::from_fn(|_| core::array::from_fn(|_| Vec::new()));
        let mut em_n = [0.0f64; NORIENT];
        let mut sum_sa2 = [0.0f64; NORIENT];
        let mut sum_saij = [0.0f64; NORIENT];
        for (o, (fo, io)) in filters.iter_mut().zip(iff.iter_mut()).enumerate() {
            for (s, (f, is)) in fo.iter_mut().zip(io.iter_mut()).enumerate() {
                f.extend(logg[s].iter().zip(&spread[o]).map(|(a, b)| a * b));
                if s == 0 {
                    for &v in f.iter() {
                        em_n[o] += (v as f64) * (v as f64);
                    }
                }
                let mut ff: Vec<Complex> = f.iter().map(|&v| Complex::new(v, 0.0)).collect();
                ifft2_planned(&mut ff, w, h, &wplan, &hplan);
                is.extend(ff.iter().map(|z| z.re * sqrt_n));
            }
            for s in 0..NSCALE {
                for i in 0..n {
                    let v = io[s][i] as f64;
                    sum_sa2[o] += v * v;
                }
            }
            for si in 0..NSCALE - 1 {
                for sj in si + 1..NSCALE {
                    for i in 0..n {
                        sum_saij[o] += io[si][i] as f64 * io[sj][i] as f64;
                    }
                }
            }
        }
        FilterBank {
            filters,
            em_n,
            sum_sa2,
            sum_saij,
            wplan,
            hplan,
        }
    }
}

/// Single-slot size-keyed bank — video IQA calls repeat one size.
#[cfg(feature = "std")]
fn filter_bank(w: usize, h: usize) -> std::sync::Arc<FilterBank> {
    static CACHE: std::sync::Mutex<Option<(usize, usize, std::sync::Arc<FilterBank>)>> =
        std::sync::Mutex::new(None);
    let mut g = CACHE.lock().unwrap();
    match &*g {
        Some((bw, bh, b)) if *bw == w && *bh == h => b.clone(),
        _ => {
            let b = std::sync::Arc::new(FilterBank::build(w, h));
            *g = Some((w, h, b.clone()));
            b
        }
    }
}

#[cfg(not(feature = "std"))]
fn filter_bank(w: usize, h: usize) -> FilterBank {
    FilterBank::build(w, h)
}

// ------------------------------------------------------------------
// Whole-pipeline body — instantiated once per tier.
// ------------------------------------------------------------------

macro_rules! score_body {
    ($token:ident, $F32:ident, $LANES:literal, $yr:ident, $yd:ident, $w:ident, $h:ident, $iq:ident, $map:ident) => {{
        let (w, h) = ($w, $h);
        // Rebind `mut` so the pool loop can `as_deref_mut` per pixel.
        #[allow(unused_mut)]
        let mut $map = $map;
        let n = w * h;
        let bank = filter_bank(w, h);
        let (wplan, hplan) = (&bank.wplan, &bank.hplan);

        // One orientation's accumulators: `(Energy_o, An_o, T)` —
        // Energy_o is *not* yet thresholded (the fold applies `T`).
        let orient = |yf: &[Complex], o: usize| -> (Vec<f32>, Vec<f32>, f32) {
            let mut sum_e = vec![0.0f32; n];
            let mut sum_o = vec![0.0f32; n];
            let mut sum_an = vec![0.0f32; n];
            let mut eo: [Vec<Complex>; NSCALE] =
                core::array::from_fn(|_| vec![Complex::new(0.0, 0.0); n]);
            let em_n = bank.em_n[o];
            for s in 0..NSCALE {
                // filter = logGabor[s] · spread[o] — from the size-keyed
                // bank; only the image-dependent ifft remains here.
                let filter = &bank.filters[o][s];
                let mut e: Vec<Complex> = yf
                    .iter()
                    .zip(filter)
                    .map(|(c, f)| Complex::new(c.re * f, c.im * f))
                    .collect();
                ifft2_planned(&mut e, w, h, &wplan, &hplan);
                anacc_body!($token, $F32, $LANES, e, sum_e, sum_o, sum_an, n);
                eo[s] = e;
            }
            // Weighted mean phase angle.
            let mut me = vec![0.0f32; n];
            let mut mo = vec![0.0f32; n];
            xenergy_body!($token, $F32, $LANES, sum_e, sum_o, me, mo, n);
            // Energy = Σ_s An·(cos Δφ − |sin Δφ|)
            let mut energy = vec![0.0f32; n];
            for e in eo.iter() {
                energy_body!($token, $F32, $LANES, e, me, mo, energy, n);
            }
            // Noise threshold: Rayleigh estimate from the smallest
            // scale's |EO|² median.
            let mut e2n: Vec<f32> = eo[0].iter().map(|c| c.re * c.re + c.im * c.im).collect();
            let median_e2n = median_f32(&mut e2n);
            let mean_e2n = -median_e2n / libm::log(0.5f64);
            let noise_power = mean_e2n / em_n;
            let sum_sa2 = bank.sum_sa2[o];
            let sum_saij = bank.sum_saij[o];
            let est_ne2 = 2.0 * noise_power * sum_sa2 + 4.0 * noise_power * sum_saij;
            let tau = libm::sqrt(est_ne2 / 2.0);
            let est_noise_energy = tau * libm::sqrt(core::f64::consts::PI / 2.0);
            let est_noise_sigma = libm::sqrt((2.0 - core::f64::consts::PI / 2.0) * tau * tau);
            // PC_2's empirical 1.7 rescaling of the PC_1 noise estimate.
            let t = ((est_noise_energy + K_NOISE * est_noise_sigma) / 1.7) as f32;
            (energy, sum_an, t)
        };

        // Phase-congruency maps of both planes.
        let pc = |y: &[f32]| -> Vec<f32> {
            let mut yf: Vec<Complex> = y.iter().map(|&v| Complex::new(v, 0.0)).collect();
            fft2_planned(&mut yf, w, h, &wplan, &hplan);
            let mut eall = vec![0.0f32; n];
            let mut aall = vec![0.0f32; n];
            #[cfg(feature = "parallel")]
            let parts: Vec<(Vec<f32>, Vec<f32>, f32)> = {
                if rayon::current_num_threads() > 1 {
                    use rayon::prelude::*;
                    (0..NORIENT)
                        .into_par_iter()
                        .map(|o| orient(&yf, o))
                        .collect()
                } else {
                    (0..NORIENT).map(|o| orient(&yf, o)).collect()
                }
            };
            #[cfg(not(feature = "parallel"))]
            let parts: Vec<(Vec<f32>, Vec<f32>, f32)> =
                (0..NORIENT).map(|o| orient(&yf, o)).collect();
            // Fold the four orientations in fixed order.
            for (energy, sum_an, t) in parts {
                ethr_body!($token, $F32, $LANES, energy, sum_an, eall, aall, t, n);
            }
            let mut pc = vec![0.0f32; n];
            pcdiv_body!($token, $F32, $LANES, eall, aall, pc, n);
            pc
        };
        // Phase-congruency maps of both planes — independent pipelines,
        // joined under `parallel` (the serial path runs them in order).
        let (pc1, pc2) = maybe_join(|| pc($yr), || pc($yd));

        // Gradient magnitudes of both planes (Scharr pair).
        let ((ix1, iy1), (ix2, iy2)) = maybe_join(
            || {
                (
                    conv2_same_3x3($yr, w, h, &DX),
                    conv2_same_3x3($yr, w, h, &DY),
                )
            },
            || {
                (
                    conv2_same_3x3($yd, w, h, &DX),
                    conv2_same_3x3($yd, w, h, &DY),
                )
            },
        );
        let mut g1 = vec![0.0f32; n];
        grad_body!($token, $F32, $LANES, ix1, iy1, g1, n);
        let mut g2 = vec![0.0f32; n];
        grad_body!($token, $F32, $LANES, ix2, iy2, g2, n);

        // Pooling: Σ GSim·PCSim·PCm(·chroma) / Σ PCm — f64 raster
        // order, identical on every tier. Per-block the similarity
        // terms run at $F32 width (per-lane identical to the scalar
        // expressions); the `+=` accumulation stays scalar in raster
        // order.
        let mut num = 0.0f64;
        let mut numc = 0.0f64;
        let mut den = 0.0f64;
        // FSIMc chroma `pow` batched per f64x4 block: `real_pow_lambda(z)`
        // ≡ `pow(|z|,λ)·(z<0 ? cos(πλ) : 1)` — for `z = −0`, `|z| = +0`
        // and `pow(+0,λ) = pow(−0,λ) = +0`.
        let mut zq = [0.0f64; 4];
        let mut bs = [0.0f64; 4];
        let mut sg = [0.0f64; 4];
        let mut fill = 0usize;
        let two = $F32::splat_t($token, 2.0);
        let (t1v, t2v) = ($F32::splat_t($token, T1), $F32::splat_t($token, T2));
        let (t3v, t4v) = ($F32::splat_t($token, T3), $F32::splat_t($token, T4));
        let nanv = $F32::splat_t($token, f32::NAN);
        let mut i = 0usize;
        while i + $LANES <= n {
            let av = $F32::from_array_t($token, core::array::from_fn(|k| pc1[i + k]));
            let bv = $F32::from_array_t($token, core::array::from_fn(|k| pc2[i + k]));
            let xv = $F32::from_array_t($token, core::array::from_fn(|k| g1[i + k]));
            let zv = $F32::from_array_t($token, core::array::from_fn(|k| g2[i + k]));
            let pcsim = (two * av * bv + t1v) / (av * av + bv * bv + t1v);
            let gsim = (two * xv * zv + t2v) / (xv * xv + zv * zv + t2v);
            // FSIMmap = GSim·PCsim — a pure store; accumulation unchanged.
            let gp = gsim * pcsim;
            if let Some(m) = $map.as_deref_mut() {
                gp.store((&mut m[i..i + $LANES]).try_into().unwrap());
            }
            // MATLAB `max` propagates NaN; `max()` under the not-NaN mask
            // is operand-order-insensitive for non-NaN lanes.
            let nanm = av.simd_ne(av) | bv.simd_ne(bv);
            let pcm = $F32::blend(nanm, nanv, av.max(bv));
            let ba = (gp * pcm).to_array(); // base = gsim·pcsim·pcm per lane
            let pa = pcm.to_array();
            for k in 0..$LANES {
                num += ba[k] as f64;
                den += pa[k] as f64;
            }
            if let Some((i1, i2, q1, q2)) = $iq {
                let iav = $F32::from_array_t($token, core::array::from_fn(|k| i1[i + k]));
                let ibv = $F32::from_array_t($token, core::array::from_fn(|k| i2[i + k]));
                let isim = (two * iav * ibv + t3v) / (iav * iav + ibv * ibv + t3v);
                let qav = $F32::from_array_t($token, core::array::from_fn(|k| q1[i + k]));
                let qbv = $F32::from_array_t($token, core::array::from_fn(|k| q2[i + k]));
                let qsim = (two * qav * qbv + t4v) / (qav * qav + qbv * qbv + t4v);
                for (k, zf) in (isim * qsim).to_array().into_iter().enumerate() {
                    let z = zf as f64;
                    zq[fill] = z.abs();
                    sg[fill] = if z < 0.0 { COS_PI_LAMBDA } else { 1.0 };
                    bs[fill] = ba[k] as f64;
                    fill += 1;
                    if fill == 4 {
                        let pr = pow64_x4($token, &zq, &[LAMBDA; 4]);
                        for j in 0..4 {
                            numc += bs[j] * (pr[j] * sg[j]);
                        }
                        fill = 0;
                    }
                }
            }
            i += $LANES;
        }
        while i < n {
            let (a, b) = (pc1[i], pc2[i]);
            let pcsim = (2.0 * a * b + T1) / (a * a + b * b + T1);
            let (x, z) = (g1[i], g2[i]);
            let gsim = (2.0 * x * z + T2) / (x * x + z * z + T2);
            if let Some(m) = $map.as_deref_mut() {
                m[i] = gsim * pcsim;
            }
            let pcm = if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                a.max(b)
            };
            let base = (gsim * pcsim * pcm) as f64;
            num += base;
            if let Some((i1, i2, q1, q2)) = $iq {
                let (ia, ib) = (i1[i], i2[i]);
                let isim = (2.0 * ia * ib + T3) / (ia * ia + ib * ib + T3);
                let (qa, qb) = (q1[i], q2[i]);
                let qsim = (2.0 * qa * qb + T4) / (qa * qa + qb * qb + T4);
                let z = (isim * qsim) as f64;
                zq[fill] = z.abs();
                sg[fill] = if z < 0.0 { COS_PI_LAMBDA } else { 1.0 };
                bs[fill] = base;
                fill += 1;
                if fill == 4 {
                    let pr = pow64_x4($token, &zq, &[LAMBDA; 4]);
                    for k in 0..4 {
                        numc += bs[k] * (pr[k] * sg[k]);
                    }
                    fill = 0;
                }
            }
            den += pcm as f64;
            i += 1;
        }
        for k in 0..fill {
            numc += bs[k] * (libm::pow(zq[k], LAMBDA) * sg[k]);
        }
        (
            num / den,
            if $iq.is_some() { numc / den } else { num / den },
        )
    }};
}

// ------------------------------------------------------------------
// Per-tier entry points + dispatch.
// ------------------------------------------------------------------

/// Arguments to a tier entry: the two decimated luma planes plus an
/// optional `(I1, I2, Q1, Q2)` decimated chroma quartet for FSIMc.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Planes<'a> {
    /// Decimated luma planes, `w2·h2` each.
    pub yr: &'a [f32],
    /// Decimated distorted luma.
    pub yd: &'a [f32],
    /// Decimated width.
    pub w: usize,
    /// Decimated height.
    pub h: usize,
    /// Optional decimated chroma `(I_ref, I_dis, Q_ref, Q_dis)`.
    pub iq: Option<IqPlanes<'a>>,
}

/// Decimated I/Q plane quad for the FSIMc chroma term:
/// `(I_ref, I_dis, Q_ref, Q_dis)`.
pub(crate) type IqPlanes<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32]);

/// The `v3` tier (AVX2/FMA, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::attune]
pub fn fsim_core_tier_v3_t(
    token: archmage::X64V3Token,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 8, yr, yd, w, h, iq, map)
}

/// The `v4x` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
pub fn fsim_core_tier_v4x_t(
    token: archmage::X64V4xToken,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 16, yr, yd, w, h, iq, map)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::attune]
pub fn fsim_core_tier_neon_t(
    token: archmage::NeonToken,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 8, yr, yd, w, h, iq, map)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::attune]
pub fn fsim_core_tier_wasm128_t(
    token: archmage::Wasm128Token,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 8, yr, yd, w, h, iq, map)
}

/// The `scalar` tier (8-wide f32, scalar-emulated lanes).
pub fn fsim_core_tier_scalar_t(
    token: archmage::ScalarToken,
    p: Planes<'_>,
    map: Option<&mut [f32]>,
) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 8, yr, yd, w, h, iq, map)
}

/// Runtime-dispatched score: the best tier this CPU supports.
fn fsim_core(p: Planes<'_>, map: Option<&mut [f32]>) -> (f64, f64) {
    archmage::attuned!(
        fsim_core_tier(p, map),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

// ------------------------------------------------------------------
// Entry points called by lib.rs.
// ------------------------------------------------------------------

/// FSIM of two strided f32 luma planes (`w`×`h` inputs, the reference's
/// automatic `F` decimation applied inside).
pub(crate) fn fsim_planes(
    r: &[f32],
    d: &[f32],
    w: usize,
    h: usize,
    stride: usize,
    map: Option<&mut [f32]>,
) -> (f64, usize, usize) {
    let f = downsample_factor(w, h);
    let (w2, h2, yr) = decimate_plane(&crate::plane_from_f32(r, w, h, stride), w, h, f);
    let (_, _, yd) = decimate_plane(&crate::plane_from_f32(d, w, h, stride), w, h, f);
    debug_assert!(map.as_deref().is_none_or(|m| m.len() == w2 * h2));
    let (s, _) = fsim_core(
        Planes {
            yr: &yr,
            yd: &yd,
            w: w2,
            h: h2,
            iq: None,
        },
        map,
    );
    (s, w2, h2)
}

/// FSIM + FSIMc of two packed sRGB8 images.
pub(crate) fn fsim_rgb8_planes(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    map: Option<&mut [f32]>,
    downscale: crate::Downscale,
) -> ((f64, f64), usize, usize) {
    let f = downscale.factor(w, h);
    let (w2, h2, [yr, ir, qr]) = decimate_rgb8(r, w, h, stride, f, [Y_COEF, I_COEF, Q_COEF]);
    let (_, _, [yd, id, qd]) = decimate_rgb8(d, w, h, stride, f, [Y_COEF, I_COEF, Q_COEF]);
    debug_assert!(map.as_deref().is_none_or(|m| m.len() == w2 * h2));
    let (fsim, fsimc) = fsim_core(
        Planes {
            yr: &yr,
            yd: &yd,
            w: w2,
            h: h2,
            iq: Some((&ir, &id, &qr, &qd)),
        },
        map,
    );
    ((fsim, fsimc), w2, h2)
}

/// FSIM of the unrounded BT.601 luma of two packed sRGB8 images.
pub(crate) fn fsim_luma8_planes(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    map: Option<&mut [f32]>,
) -> (f64, usize, usize) {
    let f = downsample_factor(w, h);
    let (w2, h2, [yr]) = decimate_rgb8(r, w, h, stride, f, [Y_COEF]);
    let (_, _, [yd]) = decimate_rgb8(d, w, h, stride, f, [Y_COEF]);
    debug_assert!(map.as_deref().is_none_or(|m| m.len() == w2 * h2));
    let (s, _) = fsim_core(
        Planes {
            yr: &yr,
            yd: &yd,
            w: w2,
            h: h2,
            iq: None,
        },
        map,
    );
    (s, w2, h2)
}

#[cfg(test)]
mod pow_tests {
    use super::pow64_x4;
    use archmage::{ScalarToken, SimdToken};

    fn corpus_x() -> alloc::vec::Vec<f64> {
        let mut xs = alloc::vec::Vec::new();
        // FSIMc operating range + dense log-space sweep.
        for i in -1000..=1000 {
            xs.push(f64::exp(i as f64 / 100.0 * 6.0)); // 1e-13..1e13
        }
        for k in -60..=60 {
            xs.push(f64::powi(2.0, k));
            xs.push(f64::powi(2.0, k) * (1.0 + f64::EPSILON));
            xs.push(f64::powi(2.0, k) * (1.0 - f64::EPSILON / 2.0));
        }
        // interval boundaries: j == 0x3988E / 0xBB67A in 1+f mantissa.
        for jw in [0x3988Eu32, 0xbb67au32] {
            for delta in -2i64..=2 {
                let hi = (0x3ff00000u64 + (jw as u64).wrapping_add_signed(delta)) << 32;
                xs.push(f64::from_bits(hi | 0x5a5a5a5a));
                xs.push(f64::from_bits(hi));
            }
        }
        // near-1 (the huge-|y| log-series path region) and edges.
        for k in 20..=24 {
            xs.push(1.0 - f64::powi(2.0, -k));
            xs.push(1.0 + f64::powi(2.0, -k));
        }
        xs.extend([
            0.0,
            f64::MIN_POSITIVE,
            f64::from_bits(1),                  // smallest subnormal
            f64::from_bits(0x000fffffffffffff), // largest subnormal
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
            1.0,
            2.0,
            0.5,
            1.5,
            -1.0,
            -0.0,
            -2.5,
            f64::NEG_INFINITY,
            f64::MIN,
        ]);
        xs
    }

    fn corpus_y() -> alloc::vec::Vec<f64> {
        let ys: alloc::vec::Vec<f64> = alloc::vec::Vec::from([
            0.03, // LAMBDA
            0.4,
            2.4,
            1.0 / 3.0,
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            -0.5,
            3.0,
            -3.0,
            100.0,
            -100.0,
            1074.9,
            1075.0,
            1075.19,
            1075.2,
            -1074.9,
            -1075.0,
            -1075.19,
            1024.0,
            -1024.0,
            f64::powi(2.0, 31) - 1.0,
            f64::powi(2.0, 31),
            f64::powi(2.0, 31) + 1.0,
            -f64::powi(2.0, 31),
            f64::powi(2.0, 63) - 1e15,
            f64::powi(2.0, 64),
            f64::powi(2.0, 64) * 2.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1e-300,
            -1e-300,
            f64::MIN_POSITIVE,
            f64::EPSILON,
        ]);
        ys
    }

    #[test]
    fn pow64_x4_matches_libm_bitwise() {
        let token = ScalarToken::summon().unwrap();
        let xs = corpus_x();
        let ys = corpus_y();
        let mut checked = 0usize;
        // Batch every (x, y) pair through 4-lane groups.
        let mut quad: alloc::vec::Vec<(f64, f64)> = alloc::vec::Vec::new();
        for &x in &xs {
            for &y in &ys {
                quad.push((x, y));
                if quad.len() == 4 {
                    let xa = [quad[0].0, quad[1].0, quad[2].0, quad[3].0];
                    let ya = [quad[0].1, quad[1].1, quad[2].1, quad[3].1];
                    let got = pow64_x4(token, &xa, &ya);
                    for k in 0..4 {
                        let want = libm::pow(xa[k], ya[k]);
                        assert_eq!(
                            got[k].to_bits(),
                            want.to_bits(),
                            "pow({:e}, {:e}): got {:e} ({:#x}) want {:e} ({:#x})",
                            xa[k],
                            ya[k],
                            got[k],
                            got[k].to_bits(),
                            want,
                            want.to_bits()
                        );
                        checked += 1;
                    }
                    quad.clear();
                }
            }
        }
        // Tail: pad the partial group with ordinary lanes.
        if !quad.is_empty() {
            while quad.len() < 4 {
                quad.push((2.0, 0.03));
            }
            let xa = [quad[0].0, quad[1].0, quad[2].0, quad[3].0];
            let ya = [quad[0].1, quad[1].1, quad[2].1, quad[3].1];
            let got = pow64_x4(token, &xa, &ya);
            for k in 0..4 {
                let want = libm::pow(xa[k], ya[k]);
                assert_eq!(
                    got[k].to_bits(),
                    want.to_bits(),
                    "pow({:e}, {:e}) tail mismatch",
                    xa[k],
                    ya[k]
                );
                checked += 1;
            }
        }
        assert!(checked > 50_000);
    }
}
