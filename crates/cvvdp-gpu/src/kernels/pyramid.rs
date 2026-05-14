//! Laplacian pyramid analysis (still-image cvvdp).
//!
//! For each of the 3 DKL channels, produces `n_levels` band buffers:
//!
//! - `band[k]` for `k < n_levels - 1` = `gauss[k] - upscale(gauss[k+1])`
//! - `band[n_levels - 1]` = the coarsest gaussian (residual)
//!
//! cvvdp v0.5.4 uses a 5-tap separable Gaussian (the "Burt-Adelson
//! kernel" with `a = 0.4`):
//!
//! ```text
//! K[a] = [0.25 - a/2, 0.25, a, 0.25, 0.25 - a/2]
//! ```
//!
//! At `a = 0.4` that's `[0.05, 0.25, 0.40, 0.25, 0.05]`. Applied
//! separably (vertical then horizontal) at stride 2 in each direction
//! for the reduce step; zero-interleaved at stride 2 + filtered for
//! the expand step.
//!
//! Edge handling: **symmetric padding**. cvvdp's `gausspyr_reduce`
//! uses `F.conv2d` with `padding=2` (zero-pad) and then patches the
//! first/last rows/cols with explicit reflection terms. For the
//! scalar reference here we collapse those patches into a single
//! reflect-index helper; numerical equivalence is verified against
//! pycvvdp goldens (per-band f32 dumps) in `tests/pyramid_scalar.rs`
//! once the per-stage tap lands in `build_goldens.py`.
//!
//! Kernels in this module:
//! - `downscale_kernel` — 5-tap separable Gaussian + 2× decimation
//!   (gauss-pyramid reduce step).
//! - `upscale_kernel`   — 2× zero-insertion + 5-tap separable Gaussian
//!   (gauss-pyramid expand step), with reconstruction gain ×4.
//! - `subtract_kernel`  — `band = fine - upscaled_coarse`.
//!
//! Kernel bodies are still stubs; the host scalar functions below
//! lock the numerical contract first so the GPU kernels can be
//! validated against them in a later round.

use cubecl::prelude::*;

/// Burt-Adelson kernel parameter `a` used by cvvdp v0.5.4.
pub const KERNEL_A: f32 = 0.4;

/// 5-tap separable Gaussian, evaluated from [`KERNEL_A`].
pub const GAUSS5: [f32; 5] = [
    0.25 - KERNEL_A / 2.0,
    0.25,
    KERNEL_A,
    0.25,
    0.25 - KERNEL_A / 2.0,
];

/// Symmetric reflection at boundaries `[0, n)`. Matches cvvdp's
/// effective access pattern (sympad inside `gausspyr_reduce`).
///
/// For `i = -1` returns `0`; `i = -2` returns `1`; ...
/// For `i = n` returns `n-1`; `i = n+1` returns `n-2`; ...
fn reflect(i: isize, n: usize) -> usize {
    let n_i = n as isize;
    let mut j = i;
    // Up to two folds cover the kernel-radius-2 range we use here.
    for _ in 0..3 {
        if j < 0 {
            j = -j - 1;
        } else if j >= n_i {
            j = 2 * n_i - j - 1;
        } else {
            break;
        }
    }
    debug_assert!(j >= 0 && j < n_i);
    j as usize
}

/// 2D separable 5-tap Gaussian + 2× decimation in each axis. Output
/// dimensions are `((sw + 1) / 2, (sh + 1) / 2)` — cvvdp rounds odd
/// dims up. Edge handling = symmetric reflection.
///
/// Two-pass: vertical pass decimates h by 2 into `sw × dh` scratch,
/// horizontal pass decimates w by 2 into the final `dw × dh` output.
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

    let mut vscratch = vec![0.0_f32; sw * dh];
    let k = GAUSS5;
    for dy in 0..dh {
        let cy = 2 * dy;
        for x in 0..sw {
            let s = |off: isize| -> f32 {
                let r = reflect(cy as isize + off, sh);
                src[r * sw + x]
            };
            vscratch[dy * sw + x] =
                k[0] * s(-2) + k[1] * s(-1) + k[2] * s(0) + k[3] * s(1) + k[4] * s(2);
        }
    }
    for dy in 0..dh {
        for dx in 0..dw {
            let cx = 2 * dx;
            let s = |off: isize| -> f32 {
                let r = reflect(cx as isize + off, sw);
                vscratch[dy * sw + r]
            };
            dst[dy * dw + dx] =
                k[0] * s(-2) + k[1] * s(-1) + k[2] * s(0) + k[3] * s(1) + k[4] * s(2);
        }
    }
    (dw, dh)
}

/// 2× upscale: zero-insert at stride 2 + 5-tap separable Gaussian.
///
/// Faithful to cvvdp's `gausspyr_expand` /
/// `interleave_zeros_and_pad`: each axis is expanded by building a
/// length-`(m+4)` buffer with the source values at even positions
/// starting at index 2, the input's first sample replicated at
/// index 0, and the input's last sample replicated at index
/// `m + 2 + (m & 1)`. A 5-tap conv with no padding then yields a
/// length-`m` row. Each axis multiplies output by 2; total
/// reconstruction gain is therefore ×4 across the separable pass.
///
/// `out_w`, `out_h` may be `2*sw`, `2*sh-1`, etc. depending on the
/// parity rule used by the matching reduce — pass the target size
/// explicitly.
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

    // Vertical pass: each column of `src` is expanded to `out_h`
    // samples via the zero-interleave + edge-replicate scheme.
    let mut vscratch = vec![0.0_f32; sw * out_h];
    let z_len_v = out_h + 4;
    let mut z_v = vec![0.0_f32; z_len_v];
    let odd_h = out_h & 1;
    let back_idx_v = out_h + 2 + odd_h;
    for x in 0..sw {
        for slot in &mut z_v {
            *slot = 0.0;
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

    // Horizontal pass: each row of vscratch is expanded to `out_w`
    // samples via the same scheme.
    dst.clear();
    dst.resize(out_w * out_h, 0.0);
    let z_len_h = out_w + 4;
    let mut z_h = vec![0.0_f32; z_len_h];
    let odd_w = out_w & 1;
    let back_idx_h = out_w + 2 + odd_w;
    for y in 0..out_h {
        for slot in &mut z_h {
            *slot = 0.0;
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

/// 2× downscale with the cvvdp 5-tap Gaussian. Stub kernel.
#[cube(launch)]
#[allow(unused_variables)]
pub fn downscale_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    dst[idx] = 0.0;
}

/// 2× upscale with the cvvdp 5-tap Gaussian. Stub kernel.
#[cube(launch)]
#[allow(unused_variables)]
pub fn upscale_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    dst[idx] = 0.0;
}

/// `band = fine - upscaled_coarse`.
#[cube(launch)]
#[allow(unused_variables)]
pub fn subtract_kernel(
    fine: &Array<f32>,
    upscaled_coarse: &Array<f32>,
    band: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    band[idx] = 0.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss5_sums_to_one() {
        let s: f32 = GAUSS5.iter().sum();
        assert!((s - 1.0).abs() < 1e-7, "GAUSS5 sums to {s}, not 1.0");
    }

    #[test]
    fn reduce_halves_dimensions() {
        let src = vec![1.0_f32; 16 * 16];
        let mut dst = Vec::new();
        let (dw, dh) = gausspyr_reduce_scalar(&src, 16, 16, &mut dst);
        assert_eq!((dw, dh), (8, 8));
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn reduce_preserves_constant_signal() {
        // GAUSS5 sums to 1; on a constant input every output pixel
        // must equal the constant. Catches coefficient typos and
        // off-by-one edge errors simultaneously.
        let src = vec![3.14_f32; 16 * 16];
        let mut dst = Vec::new();
        gausspyr_reduce_scalar(&src, 16, 16, &mut dst);
        for &v in &dst {
            assert!(
                (v - 3.14).abs() < 1e-6,
                "constant-signal reduce produced {v} ≠ 3.14"
            );
        }
    }

    #[test]
    fn expand_preserves_constant_signal() {
        // With the cvvdp-style explicit edge extension (z[0] =
        // src[0], z[back] = src[-1]), every output sample's kernel
        // hits either the K[0]+K[2]+K[4] subset or the K[1]+K[3]
        // subset of the 5-tap, each summing to 0.5; the ×2 gain per
        // axis recovers full unity. So a constant input must produce
        // a constant output across the entire buffer — boundaries
        // included.
        let src = vec![7.5_f32; 8 * 8];
        let mut dst = Vec::new();
        gausspyr_expand_scalar(&src, 8, 8, 16, 16, &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert!(
                (v - 7.5).abs() < 1e-5,
                "constant-signal expand produced {v} ≠ 7.5 at index {i}"
            );
        }
    }

    #[test]
    fn reduce_then_expand_round_trips_constant() {
        let src = vec![2.0_f32; 16 * 16];
        let mut reduced = Vec::new();
        let (dw, dh) = gausspyr_reduce_scalar(&src, 16, 16, &mut reduced);
        let mut expanded = Vec::new();
        gausspyr_expand_scalar(&reduced, dw, dh, 16, 16, &mut expanded);
        for (i, &v) in expanded.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-5, "round-trip {v} ≠ 2.0 at index {i}");
        }
    }

    #[test]
    fn expand_preserves_constant_odd_target() {
        // Odd target dimension exercises the `out_h & 1` parity branch
        // in the edge-replication index. cvvdp uses div_ceil on
        // reduce, so the inverse target can be one less than 2*sh.
        let src = vec![4.0_f32; 4 * 4];
        let mut dst = Vec::new();
        gausspyr_expand_scalar(&src, 4, 4, 7, 7, &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert!((v - 4.0).abs() < 1e-5, "odd-target expand {v} ≠ 4.0 at {i}");
        }
    }
}
