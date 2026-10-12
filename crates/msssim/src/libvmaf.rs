//! Pure-Rust port of libvmaf's **`float_ssim`** and **`float_ms_ssim`**
//! feature extractors (`libvmaf/src/feature/float_ssim.c`,
//! `float_ms_ssim.c`, `ssim.c`, `ms_ssim.c`, `iqa/{ssim_tools,decimate,
//! convolve}.c` — Netflix vmaf @ `f85a8536` / v3.2.1).
//!
//! This is the zli-nflx l·c·s SSIM variant, **not** the Wang
//! MATLAB-family implementation in [`crate::kernel`]: per-pixel
//! luminance × contrast × structure with the structure term computed
//! against `sqrt(σr·σd)` plus the zli flat-region clamp, means taken
//! over the *valid* convolution region (the outer 5 px ring drops out).
//!
//! - `float_ssim`: `scale = max(1, round(min(w,h)/256))`, decimate both
//!   planes by `scale` with a normalized `scale×scale` box kernel
//!   (symmetric boundary), then a single `_iqa_ssim` pass.
//! - `float_ms_ssim`: Wang's 5-scale variant — successive decimate-by-2
//!   through the 9/7-biorthogonal `g_lpf` (9-tap, symmetric boundary),
//!   `_iqa_ssim` per level, `msssim = Π l^α·c^β·s^γ` with
//!   α = {0,0,0,0,0.1333}, β = γ = {0.0448,0.2856,0.3001,0.2363,0.1333}.
//!
//! Constants `C1 = (0.01·255)²`, `C2 = (0.03·255)²`, `C3 = C2/2` are
//! hard-coded to L = 255 regardless of input bit depth, matching the
//! reference (`int L = 255` inside `_iqa_ssim`).
//!
//! Arithmetic mirrors the C exactly: convolution sums in f64, plane
//! storage in f32, the l/c numerators in f64 (the `2.0` literal
//! promotes), the c/denominator and s term in f32, `sqrt` calls through
//! f64, score means divided in f64 and rounded to f32 (the `_iqa_ssim`
//! float returns), the MS-SSIM product in f64.

use alloc::vec;
use alloc::vec::Vec;

use crate::Error;

/// 11-tap Gaussian (σ = 1.5) — `g_gaussian_window_{h,v}` (sums to 1).
const GAUSS: [f32; 11] = [
    0.001028, 0.007599, 0.036001, 0.109361, 0.213006, 0.266012, 0.213006, 0.109361, 0.036001,
    0.007599, 0.001028,
];
const GAUSS_LEN: usize = 11;

/// 9×9 9/7-biorthogonal low-pass for MS-SSIM decimation — `g_lpf`.
#[rustfmt::skip]
const LPF9: [[f32; 9]; 9] = [
    [ 0.000714,-0.000450,-0.002090, 0.007132, 0.016114, 0.007132,-0.002090,-0.000450, 0.000714],
    [-0.000450, 0.000283, 0.001316,-0.004490,-0.010146,-0.004490, 0.001316, 0.000283,-0.000450],
    [-0.002090, 0.001316, 0.006115,-0.020867,-0.047149,-0.020867, 0.006115, 0.001316,-0.002090],
    [ 0.007132,-0.004490,-0.020867, 0.071207, 0.160885, 0.071207,-0.020867,-0.004490, 0.007132],
    [ 0.016114,-0.010146,-0.047149, 0.160885, 0.363505, 0.160885,-0.047149,-0.010146, 0.016114],
    [ 0.007132,-0.004490,-0.020867, 0.071207, 0.160885, 0.071207,-0.020867,-0.004490, 0.007132],
    [-0.002090, 0.001316, 0.006115,-0.020867,-0.047149,-0.020867, 0.006115, 0.001316,-0.002090],
    [-0.000450, 0.000283, 0.001316,-0.004490,-0.010146,-0.004490, 0.001316, 0.000283,-0.000450],
    [ 0.000714,-0.000450,-0.002090, 0.007132, 0.016114, 0.007132,-0.002090,-0.000450, 0.000714],
];

/// MS-SSIM per-level weights — `g_alphas`/`g_betas`/`g_gammas` (β = γ).
const MS_ALPHAS: [f32; 5] = [0.0, 0.0, 0.0, 0.0, 0.1333];
const MS_BETAS: [f32; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

/// `KBND_SYMMETRIC` — whole-sample mirror (`-1-x` / `2n-1-x`), no edge
/// repetition. Only ever evaluated a few taps past the border, so no
/// fixpoint folding is needed (matches the reference, which mirrors once).
#[inline]
fn sym(i: isize, n: usize) -> usize {
    let n = n as isize;
    if i < 0 {
        (-1 - i) as usize
    } else if i >= n {
        (n - (i - n) - 1) as usize
    } else {
        i as usize
    }
}

/// `_iqa_filter_pixel` — kernel centered at (`cx`,`cy`), symmetric
/// boundary, f64 accumulation, `kscale` = 1 for normalized kernels.
fn filter_pixel(
    img: &[f32],
    w: usize,
    h: usize,
    cx: usize,
    cy: usize,
    k: &[f32],
    kw: usize,
) -> f32 {
    let uc = (kw / 2) as isize;
    let kw_even = (kw & 1) == 0;
    let mut sum = 0.0f64;
    let mut off = 0usize;
    for v in -uc..=(if kw_even { uc - 1 } else { uc }) {
        for u in -uc..=(if kw_even { uc - 1 } else { uc }) {
            let yi = sym(cy as isize + v, h);
            let xi = sym(cx as isize + u, w);
            sum += img[yi * w + xi] as f64 * k[off] as f64;
            off += 1;
        }
    }
    sum as f32
}

/// `_iqa_decimate` — output `(w/f + (w&1)) × (h/f + (h&1))`, positions
/// `(x·f, y·f)`, f64 accumulation. The `w&1` (not `w%f`) is the C
/// original: for even `w` the trailing partial kernel column is skipped
/// even when `f` doesn't divide `w`.
#[archmage::attune(_*_t, define(f64x4))]
fn decimate_inner(
    token: Token,
    img: &[f32],
    w: usize,
    h: usize,
    factor: usize,
    k: &[f32],
    kw: usize,
) -> (Vec<f32>, usize, usize) {
    let sw = w / factor + (w & 1);
    let sh = h / factor + (h & 1);
    let mut out = vec![0.0f32; sw * sh];
    let half = kw / 2;
    let right = kw - half - 1;
    crate::kernel::for_each_row(&mut out, sw, |y, orow| {
        let cy = y * factor;
        let mut x = 0;
        // Resolve reflected rows once, rather than for every kernel column.
        let row_offsets: Vec<usize> = (0..kw)
            .map(|v| sym(cy as isize + v as isize - half as isize, h) * w)
            .collect();
        while x < sw && x * factor < half {
            orow[x] = filter_pixel(img, w, h, x * factor, cy, k, kw);
            x += 1;
        }
        if kw == 9 && factor == 2 {
            let taps: &[f32; 81] = k.try_into().unwrap();
            let rows: [&[f32]; 9] =
                core::array::from_fn(|v| &img[row_offsets[v]..row_offsets[v] + w]);
            while x + 4 <= sw && (x + 3) * 2 + 4 < w {
                let mut sum = f64x4::splat_t(token, 0.0);
                for v in 0..9 {
                    let off = x * 2 - 4;
                    let pixels: &[f32; 15] = rows[v][off..off + 15].try_into().unwrap();
                    for u in 0..9 {
                        let values = f64x4::from_array_t(
                            token,
                            [
                                pixels[u] as f64,
                                pixels[u + 2] as f64,
                                pixels[u + 4] as f64,
                                pixels[u + 6] as f64,
                            ],
                        );
                        sum += values * f64x4::splat_t(token, taps[v * 9 + u] as f64);
                    }
                }
                let values = sum.to_array();
                for j in 0..4 {
                    orow[x + j] = values[j] as f32;
                }
                x += 4;
            }
        }
        while x + 4 <= sw && (x + 3) * factor + right < w {
            let mut sum = f64x4::splat_t(token, 0.0);
            for v in 0..kw {
                let row = &img[row_offsets[v]..row_offsets[v] + w];
                for u in 0..kw {
                    let off = x * factor + u - half;
                    let pixels = f64x4::from_array_t(
                        token,
                        core::array::from_fn(|j| row[off + j * factor] as f64),
                    );
                    sum += pixels * f64x4::splat_t(token, k[v * kw + u] as f64);
                }
            }
            let values = sum.to_array();
            for j in 0..4 {
                orow[x + j] = values[j] as f32;
            }
            x += 4;
        }
        while x < sw {
            orow[x] = filter_pixel(img, w, h, x * factor, cy, k, kw);
            x += 1;
        }
    });
    (out, sw, sh)
}

fn decimate(
    img: &[f32],
    w: usize,
    h: usize,
    factor: usize,
    k: &[f32],
    kw: usize,
) -> (Vec<f32>, usize, usize) {
    archmage::attuned!(
        decimate_inner(img, w, h, factor, k, kw),
        [_v3, _neon, _wasm128, _scalar]
    )
}

/// `_iqa_convolve` (IQA_CONVOLVE_1D separable path) — valid region only:
/// output `(w-10) × (h-10)`, top-left anchored; f64 accumulation, f32 out.
#[archmage::attune(_*_t, define(f64x4))]
fn conv_gauss_valid_inner(
    token: Token,
    img: &[f32],
    w: usize,
    h: usize,
) -> (Vec<f32>, usize, usize) {
    let dw = w - (GAUSS_LEN - 1);
    let dh = h - (GAUSS_LEN - 1);
    let mut tmp = vec![0.0f32; h * dw];
    crate::kernel::for_each_row(&mut tmp, dw, |y, trow| {
        let row = &img[y * w..(y + 1) * w];
        let mut x = 0;
        while x + 4 <= dw {
            let mut sum = f64x4::splat_t(token, 0.0);
            for u in 0..GAUSS_LEN {
                let v = f64x4::from_array_t(token, core::array::from_fn(|j| row[x + u + j] as f64));
                sum += v * f64x4::splat_t(token, GAUSS[u] as f64);
            }
            let values = sum.to_array();
            for j in 0..4 {
                trow[x + j] = values[j] as f32;
            }
            x += 4;
        }
        while x < dw {
            let mut sum = 0.0f64;
            for u in 0..GAUSS_LEN {
                sum += row[x + u] as f64 * GAUSS[u] as f64;
            }
            trow[x] = sum as f32;
            x += 1;
        }
    });
    let mut out = vec![0.0f32; dh * dw];
    crate::kernel::for_each_row(&mut out, dw, |y, orow| {
        let rows: [&[f32]; GAUSS_LEN] =
            core::array::from_fn(|u| &tmp[(y + u) * dw..(y + u + 1) * dw]);
        let mut x = 0;
        while x + 4 <= dw {
            let mut sum = f64x4::splat_t(token, 0.0);
            for u in 0..GAUSS_LEN {
                let v = f64x4::from_array_t(token, core::array::from_fn(|j| rows[u][x + j] as f64));
                sum += v * f64x4::splat_t(token, GAUSS[u] as f64);
            }
            let values = sum.to_array();
            for j in 0..4 {
                orow[x + j] = values[j] as f32;
            }
            x += 4;
        }
        while x < dw {
            let mut sum = 0.0f64;
            for u in 0..GAUSS_LEN {
                sum += rows[u][x] as f64 * GAUSS[u] as f64;
            }
            orow[x] = sum as f32;
            x += 1;
        }
    });
    (out, dw, dh)
}

fn conv_gauss_valid(img: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    archmage::attuned!(
        conv_gauss_valid_inner(img, w, h),
        [_v3, _neon, _wasm128, _scalar]
    )
}

/// Elementwise `out[i] = f(src[i])` — par under `parallel` when the
/// pool has >1 thread, serial otherwise (per-element math identical).
fn map2(out: &mut [f32], src: &[f32], f: impl Fn(&f32) -> f32 + Sync) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        out.par_iter_mut()
            .zip(src.par_iter())
            .for_each(|(o, &v)| *o = f(&v));
        return;
    }
    for (o, &v) in out.iter_mut().zip(src.iter()) {
        *o = f(&v);
    }
}

/// Fold `f(i)` over `0..n` into `(ssim, l, c, s)` sums. The fold uses
/// a fixed 32-band striped order in every build — the parallel path
/// distributes the same bands across threads, so serial and parallel
/// results are bit-identical (both differ from the C raster fold only
/// within f64 tolerance; the FFI parity tests gate that).
fn fold_sums(n: usize, f: impl Fn(usize) -> (f64, f64, f64, f64) + Sync) -> (f64, f64, f64, f64) {
    let body = |r: core::ops::Range<usize>| -> (f64, f64, f64, f64) {
        r.fold((0.0, 0.0, 0.0, 0.0), |(a, b, c, s), i| {
            let (pa, pb, pc, ps) = f(i);
            (a + pa, b + pb, c + pc, s + ps)
        })
    };
    const BANDS: usize = 32;
    let bs = n.div_ceil(BANDS);
    let merge = |parts: Vec<(f64, f64, f64, f64)>| {
        parts
            .into_iter()
            .fold((0.0, 0.0, 0.0, 0.0), |(a, b, c, s), (pa, pb, pc, ps)| {
                (a + pa, b + pb, c + pc, s + ps)
            })
    };
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let parts: Vec<(f64, f64, f64, f64)> = (0..BANDS)
            .into_par_iter()
            .map(|b| {
                let lo = b * bs;
                let hi = (lo + bs).min(n);
                if lo >= hi {
                    (0.0, 0.0, 0.0, 0.0)
                } else {
                    body(lo..hi)
                }
            })
            .collect();
        return merge(parts);
    }
    merge(
        (0..BANDS)
            .map(|b| {
                let lo = b * bs;
                let hi = (lo + bs).min(n);
                if lo >= hi {
                    (0.0, 0.0, 0.0, 0.0)
                } else {
                    body(lo..hi)
                }
            })
            .collect(),
    )
}

/// `_iqa_ssim` default branch — returns `(ssim, l, c, s)` means over the
/// valid region, each rounded to f32 exactly as the C returns them.
/// `w`,`h` are the (possibly decimated) plane dims; both must be ≥ 11.
fn iqa_ssim(r: &[f32], d: &[f32], w: usize, h: usize) -> (f64, f64, f64, f64) {
    // C1 = (K1·L)², C2 = (K2·L)², C3 = C2/2 — computed in f32 exactly
    // as the reference does (K1/K2 are float locals; L stays 255
    // regardless of input bit depth).
    let k1 = 0.01f32;
    let k2 = 0.03f32;
    let c1 = (k1 * 255.0f32) * (k1 * 255.0f32);
    let c2 = (k2 * 255.0f32) * (k2 * 255.0f32);
    let c3 = c2 / 2.0f32;

    let ((mu_r, vw, vh), (mu_d, _, _)) =
        crate::kernel::maybe_join(|| conv_gauss_valid(r, w, h), || conv_gauss_valid(d, w, h));
    let sq = |img: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; img.len()];
        map2(&mut out, img, |&v| v * v);
        out
    };
    let mut rd = vec![0.0f32; r.len()];
    #[cfg(feature = "parallel")]
    let mut ri = 0usize;
    #[cfg(not(feature = "parallel"))]
    let ri = 0usize;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        rd.par_iter_mut()
            .zip(r.par_iter())
            .zip(d.par_iter())
            .for_each(|((o, &a), &b)| *o = a * b);
        ri = rd.len();
    }
    for k in ri..rd.len() {
        rd[k] = r[k] * d[k];
    }
    let (r2, d2) = crate::kernel::maybe_join(|| sq(r), || sq(d));
    let ((sr_c, _, _), (sd_c, _, _)) = crate::kernel::maybe_join(
        || conv_gauss_valid(&r2, w, h),
        || conv_gauss_valid(&d2, w, h),
    );
    let (sb_c, _, _) = conv_gauss_valid(&rd, w, h);

    let n = vw * vh;
    // Striped partial sums: per-element l/c/s math is unchanged; the
    // f64 fold order differs from the serial raster fold within float
    // tolerance (bands merge in raster order).
    let partial = |i: usize| -> (f64, f64, f64, f64) {
        // float-domain moments, mirroring the C expression arithmetic.
        let sr = (sr_c[i] - mu_r[i] * mu_r[i]).max(0.0);
        let sd = (sd_c[i] - mu_d[i] * mu_d[i]).max(0.0);
        let sb = sb_c[i] - mu_r[i] * mu_d[i];
        // C: `float s_rc = sqrt(sr*sd)` — the float product promotes to
        // double, `sqrt(double)`, result assigned back to float.
        let s_rc = libm::sqrt((sr * sd) as f64) as f32;
        // l: double numerator / float denominator (C promotes via `2.0`).
        let l = (2.0f64 * mu_r[i] as f64 * mu_d[i] as f64 + c1 as f64)
            / (mu_r[i] * mu_r[i] + mu_d[i] * mu_d[i] + c1) as f64;
        let c = (2.0f64 * s_rc as f64 + c2 as f64) / (sr + sd + c2) as f64;
        let csb = if sb < 0.0 && s_rc <= 0.0 { 0.0 } else { sb };
        // s computed fully in float then promoted (matches C `float` locals).
        let s = ((csb + c3) / (s_rc + c3)) as f64;
        (l * c * s, l, c, s)
    };
    let (ssim_sum, l_sum, c_sum, s_sum) = fold_sums(n, partial);
    // C: `(float)(ssim_sum / (double)(w*h))` — a real f64 division, then
    // the mean is rounded to f32 (the `_iqa_ssim` return type and the
    // l/c/s mean outputs are all float). Widened to f64 for our API.
    let n = n as f64;
    (
        (ssim_sum / n) as f32 as f64,
        (l_sum / n) as f32 as f64,
        (c_sum / n) as f32 as f64,
        (s_sum / n) as f32 as f64,
    )
}

/// `picture_copy`'s hbd→float rule: 8-bit casts straight to float;
/// bpc 10/12/16 divide by `1<<(bpc−8)` (4 / 16 / 256), offset 0 for
/// the SSIM features.
fn hbd_scaler(bpc: u32) -> f32 {
    match bpc {
        10 => 4.0,
        12 => 16.0,
        16 => 256.0,
        _ => 1.0,
    }
}

fn plane_to_f32(y: &[u16], w: usize, h: usize, stride: usize, scaler: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for row in 0..h {
        for x in 0..w {
            out.push(y[row * stride + x] as f32 / scaler);
        }
    }
    out
}

fn check_plane(buf: &[u16], w: usize, h: usize, stride: usize) -> Result<(), Error> {
    let needed = if h == 0 { 0 } else { stride * (h - 1) + w };
    if buf.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: buf.len(),
        });
    }
    Ok(())
}

/// libvmaf **`float_ssim`** on a luma plane (`compute_ssim`, gaussian
/// window, `scale = max(1, round(min(w,h)/256))` decimation).
/// `stride` is the row pitch in elements (`≥ width`); `bpc` is the
/// input bit depth (8-bit values in `u16` lanes for `bpc = 8`; the
/// `C1/C2` constants stay at L = 255 for every `bpc`, exactly like the
/// reference — bpc only drives the hbd→float `1<<(bpc−8)` downscale).
pub fn float_ssim(
    reference_y: &[u16],
    distorted_y: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    bpc: u32,
) -> Result<f64, Error> {
    let (s, _, _, _) = float_ssim_lcs(reference_y, distorted_y, width, height, stride, bpc)?;
    Ok(s)
}

/// Hidden companion returning `(ssim, l, c, s)` — the same tuple the C
/// `compute_ssim` writes to `*l_score/*c_score/*s_score` (each an
/// f32-rounded mean). Used by the libvmaf parity tests.
#[doc(hidden)]
pub fn float_ssim_lcs(
    reference_y: &[u16],
    distorted_y: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    bpc: u32,
) -> Result<(f64, f64, f64, f64), Error> {
    check_plane(reference_y, width, height, stride)?;
    check_plane(distorted_y, width, height, stride)?;
    if width < GAUSS_LEN || height < GAUSS_LEN {
        return Err(Error::TooSmall);
    }
    let scaler = hbd_scaler(bpc);
    let mut r = plane_to_f32(reference_y, width, height, stride, scaler);
    let mut d = plane_to_f32(distorted_y, width, height, stride, scaler);
    let mut w = width;
    let mut h = height;
    // C `_round(x)` = round-half-away-from-zero; all inputs positive here.
    let scale = ((w.min(h) as f32) / 256.0).round().max(1.0) as usize;
    if scale > 1 {
        // normalized scale×scale box kernel, symmetric boundary
        let kv = 1.0f32 / (scale * scale) as f32;
        let k = vec![kv; scale * scale];
        let (nr, nw, nh) = decimate(&r, w, h, scale, &k, scale);
        r = nr;
        let (nd, _, _) = decimate(&d, w, h, scale, &k, scale);
        d = nd;
        w = nw;
        h = nh;
    }
    if w < GAUSS_LEN || h < GAUSS_LEN {
        return Err(Error::TooSmall);
    }
    Ok(iqa_ssim(&r, &d, w, h))
}

/// libvmaf **`float_ms_ssim`** on a luma plane — 5-level 9/7-lpf
/// pyramid, Wang variant, Gaussian window. Requires the reference's
/// minimum dims (each of the 5 scale levels must keep min(w,h) ≥ 11
/// before its halving).
pub fn float_ms_ssim(
    reference_y: &[u16],
    distorted_y: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    bpc: u32,
) -> Result<f64, Error> {
    let (_, _, _, msssim) =
        float_ms_ssim_scales(reference_y, distorted_y, width, height, stride, bpc)?;
    Ok(msssim)
}

/// Per-scale `(l, c, s)` means plus the `float_ms_ssim` product, as returned by
/// [`float_ms_ssim_scales`].
type ScaleScores = ([f64; 5], [f64; 5], [f64; 5], f64);

/// Hidden companion returning the per-scale `(l, c, s)` f32-rounded
/// means (the C `l_scores/c_scores/s_scores` feature arrays) plus the
/// `float_ms_ssim` product. Used by the libvmaf parity tests.
#[doc(hidden)]
pub fn float_ms_ssim_scales(
    reference_y: &[u16],
    distorted_y: &[u16],
    width: usize,
    height: usize,
    stride: usize,
    bpc: u32,
) -> Result<ScaleScores, Error> {
    check_plane(reference_y, width, height, stride)?;
    check_plane(distorted_y, width, height, stride)?;
    // Reference guard: each of the 5 halvings must stay ≥ GAUSSIAN_LEN.
    let (mut cw, mut ch) = (width, height);
    for _ in 0..5 {
        if cw < GAUSS_LEN || ch < GAUSS_LEN {
            return Err(Error::TooSmall);
        }
        cw /= 2;
        ch /= 2;
    }

    let scaler = hbd_scaler(bpc);
    let mut r_imgs: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(5);
    let mut d_imgs: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(5);
    r_imgs.push((
        plane_to_f32(reference_y, width, height, stride, scaler),
        width,
        height,
    ));
    d_imgs.push((
        plane_to_f32(distorted_y, width, height, stride, scaler),
        width,
        height,
    ));
    let lpf: Vec<f32> = LPF9.iter().flatten().copied().collect();
    for idx in 1..5 {
        let (img, w, h) = &r_imgs[idx - 1];
        r_imgs.push(decimate(img, *w, *h, 2, &lpf, 9));
        let (img, w, h) = &d_imgs[idx - 1];
        d_imgs.push(decimate(img, *w, *h, 2, &lpf, 9));
    }

    let mut msssim = 1.0f64;
    let mut l_arr = [0.0f64; 5];
    let mut c_arr = [0.0f64; 5];
    let mut s_arr = [0.0f64; 5];
    for idx in 0..5 {
        let (r, w, h) = &r_imgs[idx];
        let (d, _, _) = &d_imgs[idx];
        let (_, l, c, s) = iqa_ssim(r, d, *w, *h);
        l_arr[idx] = l;
        c_arr[idx] = c;
        s_arr[idx] = s;
        msssim *= libm::pow(l, MS_ALPHAS[idx] as f64)
            * libm::pow(c, MS_BETAS[idx] as f64)
            * libm::pow(s, MS_BETAS[idx] as f64);
    }
    Ok((l_arr, c_arr, s_arr, msssim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatched_filters_match_ordered_scalar() {
        for (w, h) in [(11, 11), (17, 23), (31, 32), (177, 176), (257, 193)] {
            let img: Vec<f32> = (0..w * h)
                .map(|i| (((i * 17) ^ (i / w * 29)) % 1021) as f32 * 0.25)
                .collect();
            let (actual, dw, dh) = conv_gauss_valid(&img, w, h);
            let mut tmp = vec![0.0f32; dw * h];
            for y in 0..h {
                for x in 0..dw {
                    let mut sum = 0.0f64;
                    for u in 0..GAUSS_LEN {
                        sum += img[y * w + x + u] as f64 * GAUSS[u] as f64;
                    }
                    tmp[y * dw + x] = sum as f32;
                }
            }
            for y in 0..dh {
                for x in 0..dw {
                    let mut sum = 0.0f64;
                    for u in 0..GAUSS_LEN {
                        sum += tmp[(y + u) * dw + x] as f64 * GAUSS[u] as f64;
                    }
                    assert_eq!(actual[y * dw + x].to_bits(), (sum as f32).to_bits());
                }
            }
            for (factor, kw) in [(2, 9), (2, 2), (3, 3), (4, 4)] {
                let kernel: Vec<f32> = if kw == 9 {
                    LPF9.into_iter().flatten().collect()
                } else {
                    vec![1.0 / (kw * kw) as f32; kw * kw]
                };
                let (actual, sw, sh) = decimate(&img, w, h, factor, &kernel, kw);
                for y in 0..sh {
                    for x in 0..sw {
                        let expected =
                            filter_pixel(&img, w, h, x * factor, y * factor, &kernel, kw);
                        assert_eq!(
                            actual[y * sw + x].to_bits(),
                            expected.to_bits(),
                            "{w}x{h}, factor {factor}, {x},{y}"
                        );
                    }
                }
            }
        }
    }
}
