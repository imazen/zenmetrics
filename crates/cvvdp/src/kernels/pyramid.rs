//! Scalar pyramid helpers for still-image cvvdp (Burt-Adelson 5-tap
//! separable Gaussian; Laplacian + Weber-contrast variants).
//!
//! Phase 8c.1-B moved these out of `cvvdp-gpu::kernels::pyramid` so
//! the CPU crate owns the canonical scalar implementation; cvvdp-gpu
//! continues to re-export the same paths. GPU-side `#[cube(launch)]`
//! kernels remain in `cvvdp-gpu::kernels::pyramid`.

use alloc::vec;
use alloc::vec::Vec;

/// Burt-Adelson kernel parameter `a` used by cvvdp v0.5.4.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pyramid::{GAUSS5, KERNEL_A};
///
/// assert_eq!(KERNEL_A, 0.4);
/// assert_eq!(GAUSS5[2], KERNEL_A);
/// assert_eq!(GAUSS5[0], 0.25 - KERNEL_A / 2.0);
/// ```
pub const KERNEL_A: f32 = 0.4;

/// 5-tap separable Gaussian, evaluated from [`KERNEL_A`].
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pyramid::{GAUSS5, KERNEL_A};
///
/// assert_eq!(GAUSS5.len(), 5);
/// assert_eq!(GAUSS5[0].to_bits(), GAUSS5[4].to_bits());
/// assert_eq!(GAUSS5[1].to_bits(), GAUSS5[3].to_bits());
/// let sum: f32 = GAUSS5.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
///
/// assert_eq!(GAUSS5[2], KERNEL_A);
/// ```
pub const GAUSS5: [f32; 5] = [
    0.25 - KERNEL_A / 2.0,
    0.25,
    KERNEL_A,
    0.25,
    0.25 - KERNEL_A / 2.0,
];

/// 2D separable 5-tap Gaussian + 2× decimation in each axis.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pyramid::gausspyr_reduce_scalar;
///
/// let src = vec![1.0_f32; 64];
/// let mut dst = Vec::new();
/// let (dw, dh) = gausspyr_reduce_scalar(&src, 8, 8, &mut dst);
/// assert_eq!((dw, dh), (4, 4));
/// assert_eq!(dst.len(), 16);
///
/// let src7 = vec![1.0_f32; 49];
/// let mut dst7 = Vec::new();
/// let (dw7, dh7) = gausspyr_reduce_scalar(&src7, 7, 7, &mut dst7);
/// assert_eq!((dw7, dh7), (4, 4));
/// ```
pub fn gausspyr_reduce_scalar(
    src: &[f32],
    sw: usize,
    sh: usize,
    dst: &mut Vec<f32>,
) -> (usize, usize) {
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    dst.clear();
    dst.resize(dw * dh, 0.0);
    let k = GAUSS5;

    let mut vscratch = vec![0.0_f32; sw * dh];
    for dy in 0..dh {
        let cy = 2 * dy as isize;
        for x in 0..sw {
            let read = |off: isize| -> f32 {
                let r = cy + off;
                if r < 0 || r >= sh as isize {
                    0.0
                } else {
                    src[r as usize * sw + x]
                }
            };
            vscratch[dy * sw + x] = k[0] * read(-2)
                + k[1] * read(-1)
                + k[2] * read(0)
                + k[3] * read(1)
                + k[4] * read(2);
        }
    }
    if dh > 0 && sh >= 2 {
        for x in 0..sw {
            vscratch[x] += src[x] * k[1] + src[sw + x] * k[0];
        }
    }
    if dh > 0 {
        let last_dy = dh - 1;
        if sh % 2 == 1 && sh >= 2 {
            for x in 0..sw {
                vscratch[last_dy * sw + x] +=
                    src[(sh - 1) * sw + x] * k[3] + src[(sh - 2) * sw + x] * k[4];
            }
        } else if sh.is_multiple_of(2) {
            for x in 0..sw {
                vscratch[last_dy * sw + x] += src[(sh - 1) * sw + x] * k[4];
            }
        }
    }

    for dy in 0..dh {
        for dx in 0..dw {
            let cx = 2 * dx as isize;
            let read = |off: isize| -> f32 {
                let c = cx + off;
                if c < 0 || c >= sw as isize {
                    0.0
                } else {
                    vscratch[dy * sw + c as usize]
                }
            };
            dst[dy * dw + dx] = k[0] * read(-2)
                + k[1] * read(-1)
                + k[2] * read(0)
                + k[3] * read(1)
                + k[4] * read(2);
        }
    }
    if dw > 0 && sw >= 2 {
        for dy in 0..dh {
            dst[dy * dw] += vscratch[dy * sw] * k[1] + vscratch[dy * sw + 1] * k[0];
        }
    }
    if dw > 0 {
        let last_dx = dw - 1;
        if sh % 2 == 1 && sw >= 2 {
            for dy in 0..dh {
                dst[dy * dw + last_dx] +=
                    vscratch[dy * sw + sw - 1] * k[3] + vscratch[dy * sw + sw - 2] * k[4];
            }
        } else if sh.is_multiple_of(2) {
            for dy in 0..dh {
                dst[dy * dw + last_dx] += vscratch[dy * sw + sw - 1] * k[4];
            }
        }
    }
    (dw, dh)
}

/// 2× upscale: zero-insert at stride 2 + 5-tap separable Gaussian.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pyramid::gausspyr_expand_scalar;
///
/// let src = vec![1.0_f32; 16];
/// let mut dst = Vec::new();
/// gausspyr_expand_scalar(&src, 4, 4, 8, 8, &mut dst);
/// assert_eq!(dst.len(), 64);
///
/// let mut dst_odd = Vec::new();
/// gausspyr_expand_scalar(&src, 4, 4, 7, 7, &mut dst_odd);
/// assert_eq!(dst_odd.len(), 49);
/// ```
pub fn gausspyr_expand_scalar(
    src: &[f32],
    sw: usize,
    sh: usize,
    out_w: usize,
    out_h: usize,
    dst: &mut Vec<f32>,
) {
    debug_assert!(out_w >= 2 * sw - 1 && out_w <= 2 * sw);
    debug_assert!(out_h >= 2 * sh - 1 && out_h <= 2 * sh);

    let k = GAUSS5;

    let mut vscratch = vec![0.0_f32; sw * out_h];
    let z_len_v = out_h + 4;
    let mut z_v = vec![0.0_f32; z_len_v];
    let odd_h = out_h & 1;
    let back_idx_v = out_h + 2 + odd_h;
    for x in 0..sw {
        z_v.fill(0.0);
        z_v[0] = src[x];
        for ky in 0..sh {
            z_v[2 + 2 * ky] = src[ky * sw + x];
        }
        z_v[back_idx_v] = src[(sh - 1) * sw + x];
        for y in 0..out_h {
            let sum = k[0] * z_v[y]
                + k[1] * z_v[y + 1]
                + k[2] * z_v[y + 2]
                + k[3] * z_v[y + 3]
                + k[4] * z_v[y + 4];
            vscratch[y * sw + x] = 2.0 * sum;
        }
    }

    dst.clear();
    dst.resize(out_w * out_h, 0.0);
    let z_len_h = out_w + 4;
    let mut z_h = vec![0.0_f32; z_len_h];
    let odd_w = out_w & 1;
    let back_idx_h = out_w + 2 + odd_w;
    for y in 0..out_h {
        z_h.fill(0.0);
        z_h[0] = vscratch[y * sw];
        for kx in 0..sw {
            z_h[2 + 2 * kx] = vscratch[y * sw + kx];
        }
        z_h[back_idx_h] = vscratch[y * sw + sw - 1];
        for x in 0..out_w {
            let sum = k[0] * z_h[x]
                + k[1] * z_h[x + 1]
                + k[2] * z_h[x + 2]
                + k[3] * z_h[x + 3]
                + k[4] * z_h[x + 4];
            dst[y * out_w + x] = 2.0 * sum;
        }
    }
}

/// One band of a Laplacian pyramid — a flat plane plus its dimensions.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pyramid::Band;
///
/// let b = Band {
///     w: 8,
///     h: 4,
///     data: vec![0.5_f32; 8 * 4],
/// };
/// assert_eq!(b.data.len(), b.w * b.h);
/// assert!(b.w >= 1 && b.h >= 1);
/// ```
pub struct Band {
    /// Width in pixels.
    pub w: usize,
    /// Height in pixels.
    pub h: usize,
    /// Row-major pixel data; `data.len() == w * h`.
    pub data: Vec<f32>,
}

/// Compute the per-band spatial frequencies (cy/deg) for a cvvdp pyramid.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pyramid::band_frequencies;
/// use cvvdp::params::DisplayGeometry;
///
/// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
/// let freqs = band_frequencies(ppd, 1024, 1024);
///
/// for i in 1..freqs.len() {
///     assert!(freqs[i] < freqs[i - 1]);
/// }
/// assert!(freqs.iter().all(|&f| f > 0.0));
/// assert!(freqs.len() >= 5);
/// ```
#[must_use]
pub fn band_frequencies(ppd: f32, width: usize, height: usize) -> Vec<f32> {
    const MIN_FREQ: f32 = 0.2;
    let min_dim = width.min(height);
    debug_assert!(min_dim >= 2, "pyramid needs at least 2px shortest side");
    let max_levels = (min_dim as f32).log2().floor() as usize - 1;
    let half_ppd = ppd / 2.0;

    let mut candidate = Vec::with_capacity(15);
    candidate.push(half_ppd);
    for f in 0..14 {
        candidate.push(0.3228_f32 * 2.0_f32.powi(-f) * half_ppd);
    }
    let max_band = candidate
        .iter()
        .position(|&b| b <= MIN_FREQ)
        .unwrap_or(max_levels);
    let n_levels = (max_band + 1).min(max_levels);

    let mut freqs = Vec::with_capacity(n_levels + 1);
    freqs.push(half_ppd);
    for f in 0..n_levels {
        freqs.push(0.3228_f32 * 2.0_f32.powi(-(f as i32)) * half_ppd);
    }
    freqs
}

/// Multi-level Laplacian pyramid decomposition (host scalar).
///
/// # Panics
///
/// Panics if the resolved level count is zero.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pyramid::laplacian_pyramid_dec_scalar;
///
/// let src: Vec<f32> = (0..16 * 16).map(|i| i as f32).collect();
/// let bands = laplacian_pyramid_dec_scalar(&src, 16, 16, 3);
/// assert_eq!(bands.len(), 3);
///
/// assert_eq!((bands[0].w, bands[0].h), (16, 16));
/// assert_eq!((bands[1].w, bands[1].h), (8, 8));
/// assert_eq!((bands[2].w, bands[2].h), (4, 4));
///
/// for b in &bands {
///     assert_eq!(b.data.len(), b.w * b.h);
/// }
/// ```
#[must_use]
pub fn laplacian_pyramid_dec_scalar(
    src: &[f32],
    sw: usize,
    sh: usize,
    n_levels: usize,
) -> Vec<Band> {
    let n = if n_levels == 0 {
        sw.min(sh).ilog2() as usize
    } else {
        n_levels
    };
    debug_assert!(n >= 1, "pyramid needs at least 1 level");

    let mut gauss: Vec<Band> = Vec::with_capacity(n);
    gauss.push(Band {
        w: sw,
        h: sh,
        data: src.to_vec(),
    });
    for k in 1..n {
        let prev = &gauss[k - 1];
        let mut next_data = Vec::new();
        let (nw, nh) = gausspyr_reduce_scalar(&prev.data, prev.w, prev.h, &mut next_data);
        gauss.push(Band {
            w: nw,
            h: nh,
            data: next_data,
        });
    }

    let mut bands: Vec<Band> = Vec::with_capacity(n);
    let mut expanded = Vec::new();
    for k in 0..(n - 1) {
        let fine = &gauss[k];
        let coarse = &gauss[k + 1];
        gausspyr_expand_scalar(
            &coarse.data,
            coarse.w,
            coarse.h,
            fine.w,
            fine.h,
            &mut expanded,
        );
        let mut band_data = vec![0.0_f32; fine.w * fine.h];
        for (i, dst) in band_data.iter_mut().enumerate() {
            *dst = fine.data[i] - expanded[i];
        }
        bands.push(Band {
            w: fine.w,
            h: fine.h,
            data: band_data,
        });
    }
    let coarsest = gauss.pop().expect("at least one level");
    bands.push(coarsest);
    bands
}

/// Output of `weber_contrast_pyr_dec_scalar`.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pyramid::{Band, WeberPyramid};
///
/// let pyr = WeberPyramid {
///     bands: vec![
///         Band { w: 16, h: 16, data: vec![0.0; 16 * 16] },
///         Band { w: 8, h: 8, data: vec![0.0; 8 * 8] },
///     ],
///     log_l_bkg: vec![
///         vec![1.0; 16 * 16],
///         vec![1.0; 8 * 8],
///     ],
/// };
/// assert_eq!(pyr.bands.len(), 2);
/// assert_eq!(pyr.log_l_bkg.len(), 2);
/// for k in 0..pyr.log_l_bkg.len() {
///     assert_eq!(pyr.log_l_bkg[k].len(), pyr.bands[k].w * pyr.bands[k].h);
/// }
/// ```
pub struct WeberPyramid {
    /// One band per pyramid level.
    pub bands: Vec<Band>,
    /// `log10(L_bkg)` per band.
    pub log_l_bkg: Vec<Vec<f32>>,
}

/// Single-channel Weber-contrast pyramid for cvvdp v0.5.4's
/// `contrast = "weber_g1"` path.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pyramid::weber_contrast_pyr_dec_scalar;
///
/// let src: Vec<f32> = (0..16 * 16).map(|i| (i as f32) * 0.1 + 1.0).collect();
/// let pyr = weber_contrast_pyr_dec_scalar(&src, &src, 16, 16, 3);
/// assert_eq!(pyr.bands.len(), 3);
/// assert_eq!(pyr.log_l_bkg.len(), 3);
///
/// let baseband_log = pyr.log_l_bkg.last().unwrap();
/// let first = baseband_log[0];
/// assert!(baseband_log.iter().all(|&v| v.to_bits() == first.to_bits()));
/// ```
#[must_use]
pub fn weber_contrast_pyr_dec_scalar(
    image_plane: &[f32],
    l_bkg_plane: &[f32],
    sw: usize,
    sh: usize,
    n_levels: usize,
) -> WeberPyramid {
    fn build_pyr(src: &[f32], sw: usize, sh: usize, n: usize) -> Vec<Band> {
        let mut p = Vec::with_capacity(n);
        p.push(Band {
            w: sw,
            h: sh,
            data: src.to_vec(),
        });
        let mut w = sw;
        let mut h = sh;
        for _ in 1..n {
            let mut next = Vec::new();
            let (nw, nh) = gausspyr_reduce_scalar(&p.last().unwrap().data, w, h, &mut next);
            p.push(Band {
                w: nw,
                h: nh,
                data: next,
            });
            w = nw;
            h = nh;
        }
        p
    }

    let n = if n_levels == 0 {
        sw.min(sh).ilog2() as usize
    } else {
        n_levels
    };
    debug_assert!(n >= 1);

    let gauss_img = build_pyr(image_plane, sw, sh, n);
    let gauss_l = build_pyr(l_bkg_plane, sw, sh, n);

    let mut bands: Vec<Band> = Vec::with_capacity(n);
    let mut log_l_bkg: Vec<Vec<f32>> = Vec::with_capacity(n);

    let mut expanded_buf = Vec::new();
    for k in 0..n {
        let is_baseband = k == n - 1;
        let fine = &gauss_img[k];
        let l_fine = &gauss_l[k];
        let n_px = fine.w * fine.h;

        if is_baseband {
            let sum: f32 = l_fine.data.iter().map(|v| v.max(0.01)).sum();
            let l_bkg_mean = sum / l_fine.data.len() as f32;
            let log_l = l_bkg_mean.log10();
            let mut contrast = vec![0.0_f32; n_px];
            for i in 0..n_px {
                contrast[i] = fine.data[i] / l_bkg_mean;
            }
            bands.push(Band {
                w: fine.w,
                h: fine.h,
                data: contrast,
            });
            log_l_bkg.push(vec![log_l; n_px]);
        } else {
            let coarse = &gauss_l[k + 1];
            gausspyr_expand_scalar(
                &coarse.data,
                coarse.w,
                coarse.h,
                fine.w,
                fine.h,
                &mut expanded_buf,
            );
            let img_coarse = &gauss_img[k + 1];
            let mut img_expanded = Vec::new();
            gausspyr_expand_scalar(
                &img_coarse.data,
                img_coarse.w,
                img_coarse.h,
                fine.w,
                fine.h,
                &mut img_expanded,
            );

            let mut contrast = vec![0.0_f32; n_px];
            let mut log_l = vec![0.0_f32; n_px];
            for i in 0..n_px {
                let l_bkg = expanded_buf[i].max(0.01);
                let layer = fine.data[i] - img_expanded[i];
                let c = (layer / l_bkg).clamp(-1000.0, 1000.0);
                contrast[i] = c;
                log_l[i] = l_bkg.log10();
            }
            bands.push(Band {
                w: fine.w,
                h: fine.h,
                data: contrast,
            });
            log_l_bkg.push(log_l);
        }
    }
    WeberPyramid { bands, log_l_bkg }
}
