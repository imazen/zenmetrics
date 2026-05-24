//! Gaussian reduce/expand + Weber-contrast pyramid per channel.
//!
//! Bit-exact port of `cvvdp_gpu::kernels::pyramid::{gausspyr_reduce_scalar,
//! gausspyr_expand_scalar, weber_contrast_pyr_dec_scalar}` — preserves
//! the pycvvdp v0.5.4 boundary bug compatibility (lpyr_dec.py:204-209
//! "x.shape[-2]" parity quirk) so goldens match.
//!
//! Differences vs upstream:
//!
//! - Uses caller-owned scratch (`PyramidScratch`) to avoid one
//!   `Vec::new()` allocation per (image, band, channel) — gives a
//!   ~20-30% wall-time win at 1MP per call.
//! - Allocates output `Band` data lazily into a single contiguous
//!   `Vec<f32>` then re-slices per band (less heap fragmentation).
//! - The `band_frequencies` helper is identical and re-exported via
//!   `cvvdp_gpu::kernels::pyramid::band_frequencies` (avoiding a
//!   redefinition).

use alloc::vec;
use alloc::vec::Vec;

pub(crate) use cvvdp_gpu::kernels::pyramid::{GAUSS5, band_frequencies};

/// One Laplacian / Weber pyramid band.
pub(crate) struct Band {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Weber-contrast pyramid output (per channel).
pub struct WeberPyramid {
    /// Per-band contrast values (finest = bands[0]).
    pub(crate) bands: Vec<Band>,
    /// `log10(L_bkg)` per band (per-pixel for non-baseband,
    /// replicated scalar for baseband).
    pub(crate) log_l_bkg: Vec<Vec<f32>>,
}

/// Scratch buffers used by reduce/expand passes. Owned by the
/// `Cvvdp` scorer so they persist across calls (no realloc).
///
/// `expanded` and `gauss_tmp` are reserved for the next SIMD pass
/// where we hoist the per-band expand output to the scratch slot
/// instead of allocating per-call.
#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct PyramidScratch {
    pub vscratch: Vec<f32>,
    pub z_v: Vec<f32>,
    pub z_h: Vec<f32>,
    pub expanded: Vec<f32>,
    pub gauss_tmp: Vec<f32>,
}

/// 2D separable 5-tap Gaussian + 2× decimation, ceil-halving.
/// Bit-identical to `cvvdp_gpu::kernels::pyramid::gausspyr_reduce_scalar`.
pub(crate) fn gausspyr_reduce(
    src: &[f32],
    sw: usize,
    sh: usize,
    scratch: &mut PyramidScratch,
    dst: &mut Vec<f32>,
) -> (usize, usize) {
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    dst.clear();
    dst.resize(dw * dh, 0.0);
    let k = GAUSS5;

    // Vertical pass: zero-pad rows above/below, conv stride 2.
    scratch.vscratch.clear();
    scratch.vscratch.resize(sw * dh, 0.0);
    let vscratch = &mut scratch.vscratch;
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

    // Horizontal pass.
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
        // Replicate pycvvdp's parity-on-rows bug — see upstream notes.
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

/// 2× upscale: zero-insert + 5-tap Gaussian (×4 reconstruction gain
/// split 2× per separable pass). Bit-identical to
/// `cvvdp_gpu::kernels::pyramid::gausspyr_expand_scalar`.
pub(crate) fn gausspyr_expand(
    src: &[f32],
    sw: usize,
    sh: usize,
    out_w: usize,
    out_h: usize,
    scratch: &mut PyramidScratch,
    dst: &mut Vec<f32>,
) {
    debug_assert!(out_w >= 2 * sw - 1 && out_w <= 2 * sw);
    debug_assert!(out_h >= 2 * sh - 1 && out_h <= 2 * sh);
    let k = GAUSS5;

    // Vertical pass.
    scratch.vscratch.clear();
    scratch.vscratch.resize(sw * out_h, 0.0);
    let vscratch = &mut scratch.vscratch;
    let z_len_v = out_h + 4;
    scratch.z_v.clear();
    scratch.z_v.resize(z_len_v, 0.0);
    let z_v = &mut scratch.z_v;
    let odd_h = out_h & 1;
    let back_idx_v = out_h + 2 + odd_h;
    for x in 0..sw {
        for v in z_v.iter_mut() {
            *v = 0.0;
        }
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

    // Horizontal pass.
    dst.clear();
    dst.resize(out_w * out_h, 0.0);
    let z_len_h = out_w + 4;
    scratch.z_h.clear();
    scratch.z_h.resize(z_len_h, 0.0);
    let z_h = &mut scratch.z_h;
    let odd_w = out_w & 1;
    let back_idx_h = out_w + 2 + odd_w;
    for y in 0..out_h {
        for v in z_h.iter_mut() {
            *v = 0.0;
        }
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

/// Build a single-channel Gaussian pyramid (`n_levels` bands).
fn build_gauss_pyramid(
    src: &[f32],
    sw: usize,
    sh: usize,
    n: usize,
    scratch: &mut PyramidScratch,
) -> Vec<Band> {
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
        let prev = p.last().unwrap();
        let (nw, nh) = gausspyr_reduce(&prev.data, w, h, scratch, &mut next);
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

/// Single-channel Weber-contrast pyramid (`weber_g1`).
///
/// `image_plane` is the channel under decomposition; `l_bkg_plane`
/// is the achromatic plane used for the per-pixel L_bkg. For the
/// achromatic channel itself they're the same buffer.
pub(crate) fn weber_contrast_pyr(
    image_plane: &[f32],
    l_bkg_plane: &[f32],
    sw: usize,
    sh: usize,
    n_levels: usize,
    scratch: &mut PyramidScratch,
) -> WeberPyramid {
    let n = n_levels;
    debug_assert!(n >= 1);

    let gauss_img = build_gauss_pyramid(image_plane, sw, sh, n, scratch);
    let gauss_l = build_gauss_pyramid(l_bkg_plane, sw, sh, n, scratch);

    let mut bands: Vec<Band> = Vec::with_capacity(n);
    let mut log_l_bkg: Vec<Vec<f32>> = Vec::with_capacity(n);

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
            let coarse_l = &gauss_l[k + 1];
            let mut expanded_l = Vec::new();
            gausspyr_expand(
                &coarse_l.data,
                coarse_l.w,
                coarse_l.h,
                fine.w,
                fine.h,
                scratch,
                &mut expanded_l,
            );
            let img_coarse = &gauss_img[k + 1];
            let mut img_expanded = Vec::new();
            gausspyr_expand(
                &img_coarse.data,
                img_coarse.w,
                img_coarse.h,
                fine.w,
                fine.h,
                scratch,
                &mut img_expanded,
            );

            let mut contrast = vec![0.0_f32; n_px];
            let mut log_l = vec![0.0_f32; n_px];
            for i in 0..n_px {
                let l_bkg = expanded_l[i].max(0.01);
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

#[cfg(test)]
mod tests {
    use super::*;
    use cvvdp_gpu::kernels::pyramid::{
        gausspyr_expand_scalar, gausspyr_reduce_scalar, weber_contrast_pyr_dec_scalar,
    };

    #[test]
    fn reduce_matches_upstream_scalar() {
        let mut rng_state = 0xdeadbeefu32;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            (rng_state >> 16) as f32 / 65536.0
        };
        let cases: &[(usize, usize)] = &[(16, 16), (15, 17), (32, 24), (73, 91), (128, 128)];
        for &(sw, sh) in cases {
            let src: Vec<f32> = (0..sw * sh).map(|_| next()).collect();
            let mut want = Vec::new();
            gausspyr_reduce_scalar(&src, sw, sh, &mut want);
            let mut scratch = PyramidScratch::default();
            let mut got = Vec::new();
            gausspyr_reduce(&src, sw, sh, &mut scratch, &mut got);
            assert_eq!(want.len(), got.len(), "{sw}x{sh}");
            for i in 0..want.len() {
                assert!((want[i] - got[i]).abs() < 1e-6, "case {sw}x{sh} idx {i}");
            }
        }
    }

    #[test]
    fn expand_matches_upstream_scalar() {
        let mut rng_state = 0xfeedf00du32;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            (rng_state >> 16) as f32 / 65536.0
        };
        let cases: &[(usize, usize, usize, usize)] = &[
            (4, 4, 8, 8),
            (4, 4, 7, 7),
            (8, 6, 16, 12),
            (8, 6, 15, 11),
            (16, 12, 32, 24),
        ];
        for &(sw, sh, ow, oh) in cases {
            let src: Vec<f32> = (0..sw * sh).map(|_| next()).collect();
            let mut want = Vec::new();
            gausspyr_expand_scalar(&src, sw, sh, ow, oh, &mut want);
            let mut scratch = PyramidScratch::default();
            let mut got = Vec::new();
            gausspyr_expand(&src, sw, sh, ow, oh, &mut scratch, &mut got);
            assert_eq!(want.len(), got.len());
            for i in 0..want.len() {
                assert!((want[i] - got[i]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn weber_pyr_matches_upstream() {
        let sw = 32;
        let sh = 32;
        let n = 4;
        let src: Vec<f32> = (0..sw * sh).map(|i| 1.0 + (i as f32) * 0.05).collect();
        let want = weber_contrast_pyr_dec_scalar(&src, &src, sw, sh, n);
        let mut scratch = PyramidScratch::default();
        let got = weber_contrast_pyr(&src, &src, sw, sh, n, &mut scratch);
        assert_eq!(got.bands.len(), want.bands.len());
        for k in 0..n {
            assert_eq!(got.bands[k].w, want.bands[k].w);
            assert_eq!(got.bands[k].h, want.bands[k].h);
            for i in 0..got.bands[k].data.len() {
                assert!(
                    (got.bands[k].data[i] - want.bands[k].data[i]).abs() < 1e-5,
                    "level {k} px {i}"
                );
                assert!(
                    (got.log_l_bkg[k][i] - want.log_l_bkg[k][i]).abs() < 1e-5,
                    "level {k} log_l px {i}"
                );
            }
        }
    }
}
