//! LDR-FLIP, ported from NVlabs/flip FLIP.h at
//! d3ab02a4425eb167d8a4021b382a2bafc6e6c838 (BSD-3-Clause; LICENSE-flip).
//! Uses the reference default 0.7 m / 3840 px / 0.7 m viewing conditions.
// These are the reference f32 coefficients; shortening them changes rounding.
#![allow(clippy::excessive_precision)]
use std::f32::consts::PI;
type Pixel = [f32; 3];
const WHITE: Pixel = [0.950428545, 1.0, 1.088900371];
const INV_WHITE: Pixel = [1.052156925, 1.0, 0.918357670];

fn xyz(p: Pixel) -> Pixel {
    [
        10135552.0 / 24577794.0 * p[0]
            + 8788810.0 / 24577794.0 * p[1]
            + 4435075.0 / 24577794.0 * p[2],
        2613072.0 / 12288897.0 * p[0]
            + 8788810.0 / 12288897.0 * p[1]
            + 887015.0 / 12288897.0 * p[2],
        1425312.0 / 73733382.0 * p[0]
            + 8788810.0 / 73733382.0 * p[1]
            + 70074185.0 / 73733382.0 * p[2],
    ]
}
fn hunt_lab<const A: bool>(p: Pixel) -> Pixel {
    let q = xyz(p);
    let delta = 6.0f32 / 29.0;
    let f: Pixel = std::array::from_fn(|i| {
        let t = q[i] * INV_WHITE[i];
        if t > delta * delta * delta {
            cbrt_t::<A>(t)
        } else {
            1.0 / (3.0 * delta * delta) * t + 4.0 / 29.0
        }
    });
    let l = 116.0 * f[1] - 16.0;
    [
        l,
        0.01 * l * (500.0 * (f[0] - f[1])),
        0.01 * l * (200.0 * (f[1] - f[2])),
    ]
}
/// Fast f32 exp2/log2/pow approximations for the `APPROX=true` variant of
/// [`error_map_impl`]. ~1-2 ulp class error vs libm; they exist only to
/// measure the transcendental share of FLIP's runtime — the bit-exact
/// contract uses `APPROX=false`.
#[inline(always)]
fn log2_fast(x: f32) -> f32 {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xff) as i32 - 127;
    let m = f32::from_bits((b & 0x7fffff) | 0x3f800000);
    // log2(1+f) on f in [0,1): degree-5 minimax-ish poly
    let f = m - 1.0;
    let p = ((((-0.0783 * f + 0.2838) * f - 0.4972) * f + 0.8045) * f - 0.7212) * f + 1.4427;
    e as f32 + p * f
}
#[inline(always)]
fn exp2_fast(x: f32) -> f32 {
    let i = x.floor();
    let f = x - i;
    // 2^f on f in [0,1): degree-5 poly
    let p = ((((0.0019 * f - 0.0083) * f + 0.0524) * f - 0.2409) * f + 0.6932) * f + 1.0;
    f32::from_bits((((i as i32) + 127) << 23) as u32) * p
}
#[inline(always)]
fn cbrt_fast(x: f32) -> f32 {
    // Bit seed + two Newton refinements (~1 ulp class).
    let s = f32::from_bits(x.to_bits() / 3 + 709921077);
    let s = (2.0 * s + x / (s * s)) / 3.0;
    (2.0 * s + x / (s * s)) / 3.0
}
#[inline(always)]
fn powf_t<const A: bool>(x: f32, y: f32) -> f32 {
    if A {
        exp2_fast(y * log2_fast(x))
    } else {
        x.powf(y)
    }
}
/// zenmetrics-math's bit-exact glibc `powf` lanes for batched sweeps —
/// identical bits to `x.powf(y)` on every input (verified exhaustively in
/// `zenmetrics-math::glibc` tests).
fn vpowf_glibc(xs: &[f32], ys: &[f32], out: &mut [f32]) {
    zenmetrics_math::glibc::vpowf_glibc_pairs_into(xs, ys, out)
}
#[inline(always)]
fn cbrt_t<const A: bool>(x: f32) -> f32 {
    if A { cbrt_fast(x) } else { x.powf(1.0 / 3.0) }
}

fn hyab(a: Pixel, b: Pixel) -> f32 {
    (a[0] - b[0]).abs() + ((a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
/// The three cube-root inputs of `filtered_lab` — the shared linear chain
/// between the batched (A=false) and per-pixel (A=true) paths. Returns
/// `t[i] = q[i] * INV_WHITE[i]`; the caller applies the cbrt-or-linear
/// branch per element.
fn filtered_lab_root_inputs(p: [f32; 4]) -> Pixel {
    let y = (p[0] + 16.0) / 116.0;
    let q = [
        (y + p[1] / 500.0) * WHITE[0],
        y,
        (y - (p[2] + p[3]) / 200.0) * WHITE[2],
    ];
    let r = [
        (3.241003275 * q[0] - 1.537398934 * q[1] - 0.498615861 * q[2]).clamp(0.0, 1.0),
        (-0.969224334 * q[0] + 1.875930071 * q[1] + 0.041554224 * q[2]).clamp(0.0, 1.0),
        (0.055639423 * q[0] - 0.204011202 * q[1] + 1.057148933 * q[2]).clamp(0.0, 1.0),
    ];
    let q = xyz(r);
    std::array::from_fn(|i| q[i] * INV_WHITE[i])
}
fn filtered_lab<const A: bool>(p: [f32; 4]) -> Pixel {
    let y = (p[0] + 16.0) / 116.0;
    let q = [
        (y + p[1] / 500.0) * WHITE[0],
        y,
        (y - (p[2] + p[3]) / 200.0) * WHITE[2],
    ];
    hunt_lab::<A>([
        (3.241003275 * q[0] - 1.537398934 * q[1] - 0.498615861 * q[2]).clamp(0.0, 1.0),
        (-0.969224334 * q[0] + 1.875930071 * q[1] + 0.041554224 * q[2]).clamp(0.0, 1.0),
        (0.055639423 * q[0] - 0.204011202 * q[1] + 1.057148933 * q[2]).clamp(0.0, 1.0),
    ])
}
// sRGB -> linear is a function of the 8-bit input alone; the table holds the
// exact f32 results the per-pixel expression produced.
fn linear_lut() -> [f32; 256] {
    std::array::from_fn(|i| {
        let s = i as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    })
}
/// Parallel engage point (pixels) for per-pixel stages. Each output element
/// is a pure function of its inputs, so parallel indexed output preserves
/// both order and bits.
const PAR_MIN_PX: usize = 1 << 16;

fn par_px<R: Send, F: Fn(usize) -> R + Sync>(n: usize, f: F) -> Vec<R> {
    if n >= PAR_MIN_PX && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(&f).collect()
    } else {
        (0..n).map(&f).collect()
    }
}

fn ingress(data: &[u8], w: usize, h: usize, stride: usize) -> Result<Vec<Pixel>, &'static str> {
    let need = h
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(w.checked_mul(3)?));
    if w == 0 || h == 0 || stride < w.saturating_mul(3) || need.is_none_or(|n| data.len() < n) {
        return Err("FLIP: invalid RGB geometry");
    }
    let linear = linear_lut();
    Ok(par_px(w * h, |i| {
        let y = i / w;
        let p: [u8; 3] = data[y * stride + (i % w) * 3..y * stride + (i % w) * 3 + 3]
            .try_into()
            .unwrap();
        let q = xyz(p.map(|v| linear[usize::from(v)]));
        let q: Pixel = std::array::from_fn(|i| q[i] * INV_WHITE[i]);
        [
            116.0 * q[1] - 16.0,
            500.0 * (q[0] - q[1]),
            200.0 * (q[1] - q[2]),
        ]
    }))
}
fn spatial_filter() -> Vec<[f32; 4]> {
    let ppd = 0.7 * (3840.0 / 0.7) * (PI / 180.0);
    let radius = (3.0 * (0.04 / (2.0 * PI * PI)).sqrt() * ppd).ceil() as i32;
    let mut f = Vec::new();
    let mut sum = [0.0; 4];
    for x in -radius..=radius {
        let ix = x as f32 * (1.0 / ppd);
        let x2 = ix * ix;
        let mut v = [0.0; 4];
        for (c, (a, b)) in [(1.0, 0.0047), (1.0, 0.0053), (34.1, 0.04), (13.5, 0.025)]
            .into_iter()
            .enumerate()
        {
            let t: f32 = a * (PI / b).sqrt();
            v[c] = if c < 2 { t } else { t.sqrt() } * (-PI * PI * x2 / b).exp();
            sum[c] += v[c];
        }
        f.push(v);
    }
    let norm = [
        1.0 / sum[0],
        1.0 / sum[1],
        1.0 / (sum[2] * sum[2] + sum[3] * sum[3]).sqrt(),
        1.0 / (sum[2] * sum[2] + sum[3] * sum[3]).sqrt(),
    ];
    for v in &mut f {
        for i in 0..4 {
            v[i] *= norm[i];
        }
    }
    f
}
fn feature_filter() -> Vec<Pixel> {
    let sd = 0.5 * 0.082 * (0.7 * (3840.0 / 0.7) * (PI / 180.0));
    let radius = (3.0 * sd).ceil() as i32;
    let mut f = Vec::new();
    let mut pos = [0.0; 3];
    let mut neg = [0.0; 3];
    for x in -radius..=radius {
        let x = x as f32;
        let g = (-(x * x) / (2.0 * sd * sd)).exp();
        let v = [g, -x * g, (x * x / (sd * sd) - 1.0) * g];
        for i in 0..3 {
            if v[i] > 0.0 {
                pos[i] += v[i];
            } else {
                neg[i] -= v[i];
            }
        }
        f.push(v);
    }
    for v in &mut f {
        for i in 0..3 {
            v[i] /= if v[i] > 0.0 { pos[i] } else { neg[i] };
        }
    }
    f
}
/// Planar (per-channel) 1-D convolution on one row or column band.
/// `src`/`dst` are one channel deinterleaved; `dir=true` means vertical.
/// Each output sample keeps the scalar tap order (`acc += t*v` per tap),
/// so lanes stay bit-identical to the scalar convolve.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x16))]
fn conv_band_t(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    w: usize,
    h: usize,
    y0: usize,
    y1: usize,
    taps: &[f32],
    dir: bool,
) {
    let r = taps.len() / 2;
    if dir {
        // Vertical: the clamped source row is uniform across the lane, so
        // every output row is fully vectorizable — no edge cases.
        for y in y0..y1 {
            let mut x = 0;
            while x + 16 <= w {
                let mut acc = f32x16::splat_t(token, 0.0);
                for (k, &t) in taps.iter().enumerate() {
                    let yy = (y + k).saturating_sub(r).min(h - 1);
                    acc += f32x16::splat_t(token, t)
                        * f32x16::load_t(
                            token,
                            src[yy * w + x..yy * w + x + 16].try_into().unwrap(),
                        );
                }
                dst[(y - y0) * w + x..(y - y0) * w + x + 16].copy_from_slice(&acc.to_array());
                x += 16;
            }
            while x < w {
                let mut acc = 0.0f32;
                for (k, &t) in taps.iter().enumerate() {
                    let yy = (y + k).saturating_sub(r).min(h - 1);
                    acc += t * src[yy * w + x];
                }
                dst[(y - y0) * w + x] = acc;
                x += 1;
            }
        }
    } else {
        // Horizontal: interior columns take unaligned shifted loads; the
        // first/last r columns clamp and stay scalar.
        let lo = r.min(w);
        let hi = w.saturating_sub(r);
        for y in y0..y1 {
            let srow = &src[y * w..y * w + w];
            let drow = &mut dst[(y - y0) * w..(y - y0) * w + w];
            for x in 0..lo {
                let mut acc = 0.0f32;
                for (k, &t) in taps.iter().enumerate() {
                    let xx = (x + k).saturating_sub(r).min(w - 1);
                    acc += t * srow[xx];
                }
                drow[x] = acc;
            }
            let mut x = lo;
            while x + 16 <= hi {
                let mut acc = f32x16::splat_t(token, 0.0);
                for (k, &t) in taps.iter().enumerate() {
                    acc += f32x16::splat_t(token, t)
                        * f32x16::load_t(
                            token,
                            srow[x + k - r..x + k - r + 16].try_into().unwrap(),
                        );
                }
                drow[x..x + 16].copy_from_slice(&acc.to_array());
                x += 16;
            }
            while x < w {
                let mut acc = 0.0f32;
                for (k, &t) in taps.iter().enumerate() {
                    let xx = (x + k).saturating_sub(r).min(w - 1);
                    acc += t * srow[xx];
                }
                drow[x] = acc;
                x += 1;
            }
        }
    }
}

/// Saturating-sub/ clamp helper parity note: `(y+k).saturating_sub(r)`
/// equals `(y as isize + k as isize - r).clamp(0, ..)` for in-range indices
/// (y,k < 2^31), and `.min(h-1)` is the upper clamp — identical to the
/// scalar reference for every reachable index.
fn conv_band(
    src: &[f32],
    dst: &mut [f32],
    w: usize,
    h: usize,
    y0: usize,
    y1: usize,
    taps: &[f32],
    dir: bool,
) {
    archmage::attuned!(
        conv_band_t(src, dst, w, h, y0, y1, taps, dir),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Row-band size for the parallel convolution: big enough that a band is
/// several cache lines of work, small enough for load balance.
const CONV_BAND_ROWS: usize = 64;

fn conv_dir(src: &[f32], dst: &mut [f32], w: usize, h: usize, taps: &[f32], dir: bool) {
    if h * w >= (1 << 16) && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        dst.par_chunks_mut(w * CONV_BAND_ROWS)
            .enumerate()
            .for_each(|(b, band)| {
                let y0 = b * CONV_BAND_ROWS;
                let y1 = (y0 + CONV_BAND_ROWS).min(h);
                conv_band(src, band, w, h, y0, y1, taps, dir);
            });
        return;
    }
    conv_band(src, dst, w, h, 0, h, taps, dir);
}

/// One 1-D separable convolution over N AoS channels, on planar views.
/// Bit-exact vs the original interleaved `convolve`: the planar layout
/// changes no arithmetic; each output element accumulates the same taps in
/// the same order.
fn conv_dir_n<const N: usize>(
    planes: &[Vec<f32>; N],
    f: &[[f32; N]],
    w: usize,
    h: usize,
    dir: bool,
) -> [Vec<f32>; N] {
    let mut out: [Vec<f32>; N] = std::array::from_fn(|_| vec![0.0; w * h]);
    for (c, (d, s)) in out.iter_mut().zip(planes.iter()).enumerate() {
        let t: Vec<f32> = f.iter().map(|t| t[c]).collect();
        conv_dir(s, d, w, h, &t, dir);
    }
    out
}

fn to_planes<const N: usize>(a: &[[f32; N]]) -> [Vec<f32>; N] {
    let mut p: [Vec<f32>; N] = std::array::from_fn(|_| Vec::with_capacity(a.len()));
    for v in a {
        for c in 0..N {
            p[c].push(v[c]);
        }
    }
    p
}

fn from_planes<const N: usize>(p: &[Vec<f32>; N]) -> Vec<[f32; N]> {
    (0..p[0].len())
        .map(|i| std::array::from_fn(|c| p[c][i]))
        .collect()
}

// Every convolution extends the nearest border sample, as in the C++ reference.
fn convolve<const N: usize>(
    a: &[[f32; N]],
    w: usize,
    h: usize,
    f: &[[f32; N]],
    vertical: bool,
) -> Vec<[f32; N]> {
    let planes = to_planes(a);
    from_planes(&conv_dir_n(&planes, f, w, h, vertical))
}
fn colors<const A: bool>(a: &[Pixel], w: usize, h: usize, f: &[[f32; 4]]) -> Vec<Pixel> {
    let a: Vec<_> = a.iter().map(|p| [p[0], p[1], p[2], p[2]]).collect();
    let a = convolve(&a, w, h, f, false);
    let a = convolve(&a, w, h, f, true);
    if A {
        return par_px(a.len(), move |i| filtered_lab::<true>(a[i]));
    }
    // Bit-exact path: identical per-pixel arithmetic; only the conditional
    // cube roots batch through the glibc-exact SIMD powf.
    let n = a.len();
    let mut t = vec![0.0f32; 3 * n];
    if n >= PAR_MIN_PX && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        t.par_chunks_mut(3)
            .zip(a.par_iter())
            .for_each(|(o, p)| o.copy_from_slice(&filtered_lab_root_inputs(*p)));
    } else {
        for (o, p) in t.chunks_mut(3).zip(a.iter()) {
            o.copy_from_slice(&filtered_lab_root_inputs(*p));
        }
    }
    let mut cb = vec![0.0f32; 3 * n];
    zenmetrics_math::glibc::vpowf_glibc_into(&t, &mut cb, 1.0 / 3.0);
    par_px(n, |i| {
        let delta = 6.0f32 / 29.0;
        let f: Pixel = std::array::from_fn(|c| {
            let t = t[3 * i + c];
            if t > delta * delta * delta {
                cb[3 * i + c]
            } else {
                1.0 / (3.0 * delta * delta) * t + 4.0 / 29.0
            }
        });
        let l = 116.0 * f[1] - 16.0;
        [
            l,
            0.01 * l * (500.0 * (f[0] - f[1])),
            0.01 * l * (200.0 * (f[1] - f[2])),
        ]
    })
}
fn features(a: &[Pixel], w: usize, h: usize, f: &[Pixel]) -> Vec<[f32; 2]> {
    let a: Vec<_> = a
        .iter()
        .map(|p| {
            let y = p[0] * (1.0 / 116.0) + 16.0 / 116.0;
            [y; 3]
        })
        .collect();
    // Horizontal channels are dx, ddx, Gaussian. Vertical channels yield
    // dx, ddx, dy and ddy independently (no derivatives are applied twice).
    let hf: Vec<_> = f.iter().map(|p| [p[1], p[2], p[0]]).collect();
    let a = convolve(&a, w, h, &hf, false);
    let a: Vec<_> = a.iter().map(|p| [p[0], p[1], p[2], p[2]]).collect();
    let vf: Vec<_> = f.iter().map(|p| [p[0], p[0], p[1], p[2]]).collect();
    let a = convolve(&a, w, h, &vf, true);
    par_px(a.len(), move |i| {
        let p = a[i];
        [
            (p[0] * p[0] + p[2] * p[2]).sqrt(),
            (p[1] * p[1] + p[3] * p[3]).sqrt(),
        ]
    })
}
/// Raw error map; arithmetic deliberately follows the f32 C++ reference.
pub(super) fn error_map_impl<const A: bool>(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    rs: usize,
    ds: usize,
) -> Result<Vec<f32>, &'static str> {
    let r = ingress(r, w, h, rs)?;
    let d = ingress(d, w, h, ds)?;
    let sf = spatial_filter();
    let ff = feature_filter();
    let cr = colors::<A>(&r, w, h, &sf);
    let cd = colors::<A>(&d, w, h, &sf);
    let fr = features(&r, w, h, &ff);
    let fd = features(&d, w, h, &ff);
    let cmax = powf_t::<A>(
        hyab(
            hunt_lab::<A>([0.0, 1.0, 0.0]),
            hunt_lab::<A>([0.0, 0.0, 1.0]),
        ),
        0.7,
    );
    let pc = 0.4 * cmax;
    if A {
        return Ok(par_px(w * h, |i| {
            let c = powf_t::<true>(hyab(cr[i], cd[i]), 0.7);
            let c = if c < pc {
                c * (0.95 / pc)
            } else {
                0.95 + ((c - pc) / (cmax - pc)) * (1.0 - 0.95)
            };
            let f = ((1.0 / 2.0f32.sqrt())
                * (fr[i][0] - fd[i][0]).abs().max((fr[i][1] - fd[i][1]).abs()))
            .powf(0.5);
            powf_t::<true>(c, 1.0 - f)
        }));
    }
    // Bit-exact path: identical scalar stages; the three transcendental
    // sweeps batch through glibc-exact SIMD powf.
    let n = w * h;
    let c_raw: Vec<f32> = par_px(n, |i| hyab(cr[i], cd[i]));
    let mut c = vec![0.0f32; n];
    zenmetrics_math::glibc::vpowf_glibc_into(&c_raw, &mut c, 0.7);
    for v in &mut c {
        *v = if *v < pc {
            *v * (0.95 / pc)
        } else {
            0.95 + ((*v - pc) / (cmax - pc)) * (1.0 - 0.95)
        };
    }
    let fbase: Vec<f32> = par_px(n, |i| {
        (1.0 / 2.0f32.sqrt()) * (fr[i][0] - fd[i][0]).abs().max((fr[i][1] - fd[i][1]).abs())
    });
    let mut e = vec![0.0f32; n];
    zenmetrics_math::glibc::vpowf_glibc_into(&fbase, &mut e, 0.5);
    for v in &mut e {
        *v = 1.0 - *v;
    }
    let mut out = vec![0.0f32; n];
    vpowf_glibc(&c, &e, &mut out);
    Ok(out)
}

pub(super) fn error_map(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    rs: usize,
    ds: usize,
) -> Result<Vec<f32>, &'static str> {
    error_map_impl::<false>(r, d, w, h, rs, ds)
}

/// Mean of the error map, summed row-major in f64 as NVlabs FLIP.h does at
/// the pinned revision (`flip` column).
pub(super) fn mean(map: &[f32]) -> f64 {
    map.iter().fold(0.0f64, |s, &v| s + f64::from(v)) / map.len() as f64
}

/// Mean as flip-evaluator 1.7 computes it, and as the JPEG AIC-4 `FLIP`
/// column was produced: a row-major f32 running sum (`flip-v17` column).
/// The f32 sum loses precision as the map grows: on a uniform 0.075 map it
/// reads 1.0e-3 (2048²) and 1.7e-2 (4096²) low, relative to [`mean`].
pub(super) fn mean_v17(map: &[f32]) -> f64 {
    f64::from(map.iter().fold(0.0f32, |s, &v| s + v) / map.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Diagnostic (2026-10-20, on the 64x64 pinned-test inputs):
    /// `libm::powf` (zenmetrics-math's bit-exact target) vs `f32::powf`
    /// (glibc): cbrt 1776/12288, powf(0.7) 367/4096, powf(0.5) 0/4096,
    /// powf(c,1-f) 373/4096 input mismatches, ~1e-7 max rel — the math
    /// crate's ports replicate `libm` the crate, not glibc, so they cannot
    /// keep this file's pinned and published-column bit-matches.

    /// Wall-clock + score-delta comparison of the bit-exact (A=false) vs
    /// approximated-transcendental (A=true) error-map pipelines.
    #[test]
    #[ignore]
    fn approx_vs_exact_bench() {
        let (w, h) = (1024usize, 1024usize);
        let r: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 11 + i / 7919) % 256) as u8)
            .collect();
        let mut d = r.clone();
        for p in d.iter_mut().step_by(7) {
            *p = p.wrapping_add(9);
        }
        for p in d.iter_mut().step_by(131) {
            *p ^= 127;
        }
        let reps = 3;
        let t0 = std::time::Instant::now();
        let mut me = 0.0;
        for _ in 0..reps {
            me = mean(&error_map_impl::<false>(&r, &d, w, h, w * 3, w * 3).unwrap());
        }
        let te = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let t0 = std::time::Instant::now();
        let mut ma = 0.0;
        for _ in 0..reps {
            ma = mean(&error_map_impl::<true>(&r, &d, w, h, w * 3, w * 3).unwrap());
        }
        let ta = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let ea = error_map_impl::<false>(&r, &d, w, h, w * 3, w * 3).unwrap();
        let aa = error_map_impl::<true>(&r, &d, w, h, w * 3, w * 3).unwrap();
        let mut maxd = 0.0f32;
        let mut n_mismatch = 0u32;
        for (x, y) in ea.iter().zip(&aa) {
            if x != y {
                n_mismatch += 1;
                maxd = maxd.max((x - y).abs());
            }
        }
        eprintln!(
            "exact {te:.2}ms  approx {ta:.2}ms  ({:.2}x)  mean {me:.8e} vs {ma:.8e}  d={:.3e}  map mismatches {n_mismatch}/{}  max {maxd:e}",
            te / ta,
            (me - ma).abs(),
            ea.len()
        );
    }

    /// The register-accumulating `convolve` is bit-identical to the original
    /// in-place accumulation, on odd sizes, both directions, clamped borders.
    #[test]
    fn convolve_matches_in_place_reference() {
        fn reference<const N: usize>(
            a: &[[f32; N]],
            w: usize,
            h: usize,
            f: &[[f32; N]],
            vertical: bool,
        ) -> Vec<[f32; N]> {
            let mut out = vec![[0.0; N]; w * h];
            let r = (f.len() / 2) as isize;
            for y in 0..h {
                for x in 0..w {
                    let v = &mut out[y * w + x];
                    for (k, t) in f.iter().enumerate() {
                        let xx = if vertical {
                            x
                        } else {
                            (x as isize + k as isize - r).clamp(0, w as isize - 1) as usize
                        };
                        let yy = if vertical {
                            (y as isize + k as isize - r).clamp(0, h as isize - 1) as usize
                        } else {
                            y
                        };
                        for c in 0..N {
                            v[c] += t[c] * a[yy * w + xx][c];
                        }
                    }
                }
            }
            out
        }
        let bits = |v: &[[f32; 3]]| v.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
        for &(w, h) in &[(1, 1), (7, 5), (33, 17), (64, 3)] {
            let a: Vec<[f32; 3]> = (0..w * h)
                .map(|i| std::array::from_fn(|c| ((i * 31 + c * 7) % 97) as f32 * 0.013 - 0.4))
                .collect();
            let f: Vec<[f32; 3]> = (0..9)
                .map(|k| std::array::from_fn(|c| ((k * 5 + c) % 11) as f32 * 0.021 + 0.003))
                .collect();
            for vertical in [false, true] {
                assert_eq!(
                    bits(&convolve(&a, w, h, &f, vertical)),
                    bits(&reference(&a, w, h, &f, vertical)),
                    "{w}x{h} vertical={vertical}"
                );
            }
        }
    }

    #[test]
    fn srgb_linear_lut_matches_per_pixel_expression() {
        let lut = linear_lut();
        for v in 0u16..=255 {
            let s = v as u8 as f32 / 255.0;
            let direct = if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
            assert_eq!(lut[usize::from(v as u8)], direct, "v={v}");
        }
    }
    #[test]
    fn identity_stride_and_border_distortion() {
        let (w, h) = (19, 17);
        let r: Vec<_> = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 19) % 256) as u8)
            .collect();
        let mut d = r.clone();
        d[0] ^= 127;
        let map = error_map(&r, &d, w, h, w * 3, w * 3).unwrap();
        assert!(mean(&map) > 0.0);
        let mut rs = vec![211; (w * 3 + 5) * h];
        let mut ds = rs.clone();
        for y in 0..h {
            rs[y * (w * 3 + 5)..y * (w * 3 + 5) + w * 3]
                .copy_from_slice(&r[y * w * 3..(y + 1) * w * 3]);
            ds[y * (w * 3 + 5)..y * (w * 3 + 5) + w * 3]
                .copy_from_slice(&d[y * w * 3..(y + 1) * w * 3]);
        }
        assert_eq!(
            map,
            error_map(&rs, &ds, w, h, w * 3 + 5, w * 3 + 5).unwrap()
        );
        assert!(
            error_map(&r, &r, w, h, w * 3, w * 3)
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(error_map(&r, &d, w, h, w * 3 - 1, w * 3).is_err());
    }
}
