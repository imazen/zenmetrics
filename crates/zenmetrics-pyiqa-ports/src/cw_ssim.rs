//! CW-SSIM as pyiqa 0.1.16 computes it (`pyiqa.create_metric('cw_ssim')`:
//! `CW_SSIM(channels=1, level=4, ori=8, guardb=0, K=0, test_y_channel=True,
//! color_space='yiq')`, `pyiqa/archs/ssim_arch.py` with
//! `pyiqa/matlab_utils/{scfpyr_util,math_util,functions}.py`). The AIC2026
//! `CW-SSIM` column is this function (pyiqa vs the published column: max
//! |Δ| 2.2e-16 over its 61 crop pairs). zenmetrics' own `cw-ssim` follows
//! pyrtools instead.
//!
//! Pipeline, in pyiqa's float order:
//! 1. `ToTensor` (`u8 / 255` in f32), widened to f64; YIQ luma with the f32
//!    weights `0.299, 0.587, 0.114` widened to f64, `× 255`, rounded half to
//!    even.
//! 2. `SCFpyr_PyTorch(height=4, nbands=8)`: f64 `fft2`, `fftshift`, the
//!    raised-cosine low-pass `lo0mask`, then three rounds of {crop the
//!    centred spectrum to `⌈(d − ½)/2⌉`, multiply by `lomask`}; at the fourth
//!    level the eight oriented bands `ifft2(ifftshift(i · lodft · anglemask_b ·
//!    himask))`. Every mask is computed in f64 by `np.interp` over pyiqa's
//!    lookup tables and **rounded to f32** (`.float()`), as pyiqa does.
//!    Only the coarsest oriented level (`bandind = level = 4`) is scored, so
//!    the finer levels' bands, the high-pass and the low-pass residual, which
//!    pyiqa also computes, are skipped here.
//! 3. Per band: `corr = b1 · conj(b2)`, 7×7 valid box sums (weight
//!    `f32(1/49)`) of `corr` and of `|b1|² + |b2|²` (with
//!    `|z| = sqrt(re² + im² + 1e-12)`), `cssim = 2·|Σcorr| / Σvarr`, weighted
//!    by a centred Gaussian of size `(h₄ − 6, w₄ − 6)` and σ = `h₄ / 4`
//!    (`fspecial`, normalised in f64, rounded to f32); the score is the mean
//!    over the eight bands.
//!
//! Higher is better; identical images score 1. Needs both sides ≥ 64
//! (pyiqa refuses fewer than four levels' worth of octaves).
//!
//! The warm reference caches the size plan and the reference's bands.

use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use zenmetrics_abstractions as za;
use zenmetrics_nn::Error;

use crate::port::{Port, check_buffer, impl_metric};

const COLUMN: &str = "cw_ssim_pyiqa";
const LEVELS: usize = 4;
const ORI: usize = 8;
/// Smallest side: `4 > ⌊log2 min⌋ − 2` is refused.
const MIN_SIZE: usize = 64;
/// Largest side accepted (bounds the f64 spectrum at 16 B per pixel).
const MAX_SIDE: usize = 16_384;

/// `np.interp(x, xp, fp)` with numpy's branch order (left/right clamps,
/// exact-knot hit, `slope · (x − xp[j]) + fp[j]`).
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    // Largest j with xp[j] <= x.
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// `np.linspace(start, stop, num)`: `i · step + start`, last element `stop`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    (0..num)
        .map(|i| {
            if i == num - 1 {
                stop
            } else {
                i as f64 * step + start
            }
        })
        .collect()
}

/// numpy's pairwise summation (`pairwise_sum`, block 8, cut-over 128).
fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        a.iter().fold(0.0, |s, &v| s + v)
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let full = n - n % 8;
        for c in a[8..full].as_chunks::<8>().0 {
            for (rj, &v) in r.iter_mut().zip(c) {
                *rj += v;
            }
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[full..] {
            res += v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// `ndarray.sum()` of a contiguous array: the first element seeds the
/// reduction, the rest is summed pairwise.
fn np_sum(a: &[f64]) -> f64 {
    a[0] + pairwise(&a[1..])
}

/// pyiqa's `rcosFn(1, -0.5)` → `(Xrcos, Yrcos = sqrt(Y), YIrcos)`.
fn rcos() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    const N: i32 = 256;
    let x: Vec<f64> = (-N - 1..2)
        .map(|k| PI * f64::from(k) / 2.0 / f64::from(N))
        .collect();
    let mut y: Vec<f64> = x
        .iter()
        .map(|&v| {
            let c = v.cos();
            c * c
        })
        .collect();
    y[0] = y[1];
    y[N as usize + 2] = y[N as usize + 1];
    let xr = x
        .iter()
        .map(|&v| -0.5 + 2.0 / PI * (v + PI / 4.0))
        .collect();
    let yr: Vec<f64> = y.iter().map(|v| v.sqrt()).collect();
    let yi = yr.iter().map(|&v| (1.0 - v * v).sqrt()).collect();
    (xr, yr, yi)
}

/// `Xcosn` and `Ycosn` of `SCFpyr_PyTorch` (lutsize 1024, order 7).
fn cosn() -> (Vec<f64>, Vec<f64>) {
    const LUT: i32 = 1024;
    let xc: Vec<f64> = (-(2 * LUT + 1)..LUT + 2)
        .map(|k| PI * f64::from(k) / f64::from(LUT))
        .collect();
    let order = (ORI - 1) as i32;
    let fact = |n: i32| (1..=n).map(f64::from).product::<f64>();
    let f = fact(order);
    let konst = 2f64.powi(2 * order) * (f * f) / (ORI as f64 * fact(2 * order));
    let lead = 2.0 * konst.sqrt();
    let yc = xc
        .iter()
        .map(|&x| {
            let alpha = (x + PI).rem_euclid(2.0 * PI) - PI;
            let gate = if alpha.abs() < PI / 2.0 { 1.0 } else { 0.0 };
            lead * x.cos().powf(f64::from(order)) * gate
        })
        .collect();
    (xc, yc)
}

/// Round an f64 mask value to f32 and back, like `torch.from_numpy(m).float()`
/// multiplied into an f64 tensor.
fn f32r(v: f64) -> f64 {
    f64::from(v as f32)
}

/// `(start, len)` of pyiqa's low-pass crop of a centred spectrum axis `d`.
fn crop_axis(d: usize) -> (usize, usize) {
    let nd = d.div_ceil(2);
    (d / 2 - nd / 2, nd)
}

/// A centred crop `(y0, h, x0, w)` of a spectrum.
type Crop = (usize, usize, usize, usize);

/// Everything that depends only on the image size.
struct Plan {
    w: usize,
    h: usize,
    fwd_row: Arc<dyn Fft<f64>>,
    fwd_col: Arc<dyn Fft<f64>>,
    /// Full-size f32 low-pass mask (fftshifted layout).
    lo0: Vec<f64>,
    /// Per downsampling round: the crop `(y0, h, x0, w)` of the previous
    /// level's spectrum and the f32 `lomask` at the cropped size.
    rounds: Vec<(Crop, Vec<f64>)>,
    /// Level-4 band masks `anglemask_b · himask` factors (f32 values).
    himask: Vec<f64>,
    angle: [Vec<f64>; ORI],
    /// Level-4 size.
    h4: usize,
    w4: usize,
    inv_row: Arc<dyn Fft<f64>>,
    inv_col: Arc<dyn Fft<f64>>,
    /// Gaussian weight `(h4 − 6) × (w4 − 6)`, f32 values.
    gauss: Vec<f64>,
}

impl Plan {
    fn new(w: usize, h: usize) -> Self {
        // prepare_grid(m = h, n = w).
        let axis = |m: usize| {
            let half = m as f64 / 2.0;
            let lo = -((m / 2) as f64) / half;
            let hi = (m / 2) as f64 / half - ((1 - m % 2) * 2) as f64 / m as f64;
            linspace(lo, hi, m)
        };
        let (gy, gx) = (axis(h), axis(w));
        let mut log_rad = vec![0.0; w * h];
        let mut angle = vec![0.0; w * h];
        for i in 0..h {
            for j in 0..w {
                angle[i * w + j] = gy[i].atan2(gx[j]);
                log_rad[i * w + j] = (gx[j] * gx[j] + gy[i] * gy[i]).sqrt();
            }
        }
        log_rad[(h / 2) * w + w / 2] = log_rad[(h / 2) * w + w / 2 - 1];
        for v in &mut log_rad {
            *v = v.log2();
        }
        let (mut xr, yr, yi) = rcos();
        let lo0 = log_rad.iter().map(|&v| f32r(interp(v, &xr, &yi))).collect();
        let (mut ch, mut cw) = (h, w);
        let mut rounds = Vec::with_capacity(LEVELS - 1);
        for _ in 0..LEVELS - 1 {
            for v in &mut xr {
                *v -= 1.0;
            }
            let ((y0, nh), (x0, nw)) = (crop_axis(ch), crop_axis(cw));
            let crop = |a: &[f64]| -> Vec<f64> {
                (0..nh)
                    .flat_map(|i| a[(y0 + i) * cw + x0..][..nw].iter().copied())
                    .collect()
            };
            log_rad = crop(&log_rad);
            angle = crop(&angle);
            let lomask = log_rad.iter().map(|&v| f32r(interp(v, &xr, &yi))).collect();
            rounds.push(((y0, nh, x0, nw), lomask));
            (ch, cw) = (nh, nw);
        }
        for v in &mut xr {
            *v -= 1.0;
        }
        let himask = log_rad.iter().map(|&v| f32r(interp(v, &xr, &yr))).collect();
        let (xc, yc) = cosn();
        let angle = std::array::from_fn(|b| {
            let xs: Vec<f64> = xc.iter().map(|&x| x + PI * b as f64 / ORI as f64).collect();
            angle.iter().map(|&a| f32r(interp(a, &xs, &yc))).collect()
        });
        // fspecial((h4 - 6, w4 - 6), h4 / 4).
        let (gh, gw) = (ch - 6, cw - 6);
        let sigma = ch as f64 / 4.0;
        let (m, n) = ((gh as f64 - 1.0) / 2.0, (gw as f64 - 1.0) / 2.0);
        let mut g: Vec<f64> = (0..gh)
            .flat_map(|i| {
                let y = -m + i as f64;
                (0..gw).map(move |j| {
                    let x = -n + j as f64;
                    (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()
                })
            })
            .collect();
        let gmax = g.iter().copied().fold(f64::MIN, f64::max);
        for v in &mut g {
            if *v < f64::EPSILON * gmax {
                *v = 0.0;
            }
        }
        let s = np_sum(&g);
        if s != 0.0 {
            for v in &mut g {
                *v /= s;
            }
        }
        let gauss = g.into_iter().map(f32r).collect();
        let mut planner = FftPlanner::new();
        Self {
            w,
            h,
            fwd_row: planner.plan_fft_forward(w),
            fwd_col: planner.plan_fft_forward(h),
            lo0,
            rounds,
            himask,
            angle,
            h4: ch,
            w4: cw,
            inv_row: planner.plan_fft_inverse(cw),
            inv_col: planner.plan_fft_inverse(ch),
            gauss,
        }
    }
}

/// In-place 2-D FFT of a row-major `h × w` buffer: rows, then columns.
fn fft2(a: &mut [Complex<f64>], w: usize, h: usize, row: &dyn Fft<f64>, col: &dyn Fft<f64>) {
    let mut scratch = vec![
        Complex::default();
        row.get_inplace_scratch_len()
            .max(col.get_inplace_scratch_len())
    ];
    for r in a.chunks_exact_mut(w) {
        row.process_with_scratch(r, &mut scratch);
    }
    let mut c = vec![Complex::default(); h];
    for x in 0..w {
        for (y, v) in c.iter_mut().enumerate() {
            *v = a[y * w + x];
        }
        col.process_with_scratch(&mut c, &mut scratch);
        for (y, v) in c.iter().enumerate() {
            a[y * w + x] = *v;
        }
    }
}

/// `out[i][j] = a[(i + dy) % h][(j + dx) % w]` (pyiqa's `roll_n` shifts).
fn roll(a: &[Complex<f64>], w: usize, h: usize, dx: usize, dy: usize) -> Vec<Complex<f64>> {
    let mut out = Vec::with_capacity(w * h);
    for i in 0..h {
        let src = &a[((i + dy) % h) * w..][..w];
        out.extend_from_slice(&src[dx..]);
        out.extend_from_slice(&src[..dx]);
    }
    out
}

/// The eight level-4 oriented bands, each `h4 · w4` complex values.
type Bands = Vec<Vec<Complex<f64>>>;

/// YIQ luma rounded to integers, as pyiqa's `to_y_channel(x, 255, 'yiq')`.
fn luma(rgb: &[u8], w: usize, h: usize, stride: usize) -> Vec<Complex<f64>> {
    let k = [
        f64::from(0.299f32),
        f64::from(0.587f32),
        f64::from(0.114f32),
    ];
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for p in rgb[y * stride..][..3 * w].as_chunks::<3>().0 {
            let c = |i: usize| f64::from(f32::from(p[i]) / 255.0);
            let v = c(0) * k[0] + c(1) * k[1] + c(2) * k[2];
            out.push(Complex::new((v * 255.0).round_ties_even(), 0.0));
        }
    }
    out
}

fn bands(plan: &Plan, rgb: &[u8], stride: usize) -> Bands {
    let (w, h) = (plan.w, plan.h);
    let mut a = luma(rgb, w, h, stride);
    fft2(&mut a, w, h, &*plan.fwd_row, &*plan.fwd_col);
    let mut lo = roll(&a, w, h, w.div_ceil(2), h.div_ceil(2));
    drop(a);
    for (v, &m) in lo.iter_mut().zip(&plan.lo0) {
        *v = Complex::new(v.re * m, v.im * m);
    }
    let mut cw = w;
    for &((y0, nh, x0, nw), ref lomask) in &plan.rounds {
        let mut next = Vec::with_capacity(nh * nw);
        for i in 0..nh {
            next.extend_from_slice(&lo[(y0 + i) * cw + x0..][..nw]);
        }
        for (v, &m) in next.iter_mut().zip(lomask) {
            *v = Complex::new(m * v.re, m * v.im);
        }
        lo = next;
        cw = nw;
    }
    let (h4, w4) = (plan.h4, plan.w4);
    let scale = 1.0 / (h4 * w4) as f64;
    plan.angle
        .iter()
        .map(|am| {
            let band: Vec<Complex<f64>> = lo
                .iter()
                .zip(am)
                .zip(&plan.himask)
                .map(|((v, &a), &hm)| {
                    let (re, im) = (v.re * a * hm, v.im * a * hm);
                    // × (-0 + 1i): (re, im) → (−im, re).
                    Complex::new(-im, re)
                })
                .collect();
            let mut b = roll(&band, w4, h4, w4 / 2, h4 / 2);
            fft2(&mut b, w4, h4, &*plan.inv_row, &*plan.inv_col);
            for v in &mut b {
                *v = Complex::new(v.re * scale, v.im * scale);
            }
            b
        })
        .collect()
}

/// 7×7 valid box sum with weight `f32(1/49)`, row-major kernel order.
fn box7(a: &[f64], w: usize, h: usize) -> Vec<f64> {
    let k = f64::from(1.0f32 / 49.0);
    let (ow, oh) = (w - 6, h - 6);
    let mut out = Vec::with_capacity(ow * oh);
    for y in 0..oh {
        for x in 0..ow {
            let mut s = 0.0;
            for dy in 0..7 {
                for &v in &a[(y + dy) * w + x..][..7] {
                    s += k * v;
                }
            }
            out.push(s);
        }
    }
    out
}

fn mag(re: f64, im: f64) -> f64 {
    (re * re + im * im + 1e-12).sqrt()
}

/// A reference image's size plan and level-4 bands.
pub(crate) struct WarmRef {
    plan: Plan,
    bands: Bands,
}

/// CW-SSIM, pyiqa convention (see the module docs). No model files.
#[derive(Debug, Default, Clone, Copy)]
pub struct CwSsimPyiqa;

fn check(img: &za::Rgb8In<'_>) -> Result<(), Error> {
    check_buffer(img)?;
    if img.width < MIN_SIZE || img.height < MIN_SIZE {
        return Err(Error::TooSmall { min: MIN_SIZE });
    }
    if img.width > MAX_SIDE || img.height > MAX_SIDE {
        return Err(Error::TooLarge {
            limit: MAX_SIDE * MAX_SIDE,
        });
    }
    Ok(())
}

impl CwSsimPyiqa {
    /// The metric (it has no weights or model files).
    pub fn new() -> Self {
        Self
    }
}

impl Port for CwSsimPyiqa {
    type Warm = WarmRef;
    const COLUMN: &'static str = COLUMN;
    const KIND: za::ScoreKind = za::ScoreKind::Similarity;
    const REVISION: u32 = 1;

    fn reference(&self) -> String {
        "pyiqa 0.1.16 cw_ssim (level 4, 8 orientations, K 0, YIQ luma)".into()
    }

    fn warm(&self, r: &za::Rgb8In<'_>) -> Result<WarmRef, Error> {
        check(r)?;
        let plan = Plan::new(r.width, r.height);
        let bands = bands(&plan, r.pixels, r.stride);
        Ok(WarmRef { plan, bands })
    }

    fn warm_size(w: &WarmRef) -> (usize, usize) {
        (w.plan.w, w.plan.h)
    }

    fn score_warm(&self, r: &WarmRef, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        check(d)?;
        let d = bands(&r.plan, d.pixels, d.stride);
        let (w4, h4) = (r.plan.w4, r.plan.h4);
        let mut total = 0.0;
        // pyiqa calls cw_ssim(X = distorted, Y = reference): band1 is the
        // distorted image's band, band2 the reference's.
        for (b1, b2) in d.iter().zip(&r.bands) {
            let n = w4 * h4;
            let (mut cre, mut cim, mut varr) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            for i in 0..n {
                let (a, b) = (b1[i].re, b1[i].im);
                let (c, dd) = (b2[i].re, -b2[i].im);
                cre[i] = a * c - b * dd;
                cim[i] = b * c + a * dd;
                let (m1, m2) = (mag(a, b), mag(c, b2[i].im));
                varr[i] = m1 * m1 + m2 * m2;
            }
            let (sre, sim, sv) = (box7(&cre, w4, h4), box7(&cim, w4, h4), box7(&varr, w4, h4));
            let mut band = 0.0;
            for i in 0..sre.len() {
                let cssim = (2.0 * mag(sre[i], sim[i]) + 0.0) / (sv[i] + 0.0);
                band += cssim * r.plan.gauss[i];
            }
            total += band;
        }
        Ok(total / ORI as f64)
    }
}

impl_metric!(CwSsimPyiqa);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_matches_pyiqa_formula() {
        for d in 1..200usize {
            let df = d as f64;
            // pyiqa: start = ceil((d+.5)/2) - ceil((ceil((d-.5)/2)+.5)/2),
            // len = ceil((d-.5)/2).
            let len = ((df - 0.5) / 2.0).ceil();
            let start = ((df + 0.5) / 2.0).ceil() - ((len + 0.5) / 2.0).ceil();
            assert_eq!(crop_axis(d), (start as usize, len as usize), "d={d}");
        }
    }

    #[test]
    fn identical_images_score_just_below_one() {
        // pyiqa adds 1e-12 under every |z| sqrt, so identical images score
        // slightly below 1 (1 - 1.6e-9 on KonJND photos; more on bands with
        // little energy, like this pattern's).
        let (w, h) = (80, 72);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 37) % 251) as u8).collect();
        let img = za::Rgb8In {
            pixels: &rgb,
            width: w,
            height: h,
            stride: 3 * w,
        };
        let m = CwSsimPyiqa;
        let s = m.score_warm(&m.warm(&img).unwrap(), &img).unwrap();
        assert!(s <= 1.0 && 1.0 - s < 1e-6, "{s}");
    }

    #[test]
    fn interp_matches_numpy_edges() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [10.0, 20.0, 40.0];
        assert_eq!(interp(-1.0, &xp, &fp), 10.0);
        assert_eq!(interp(3.0, &xp, &fp), 40.0);
        assert_eq!(interp(2.0, &xp, &fp), 40.0);
        assert_eq!(interp(1.0, &xp, &fp), 20.0);
        assert_eq!(interp(1.5, &xp, &fp), 30.0);
    }
}
