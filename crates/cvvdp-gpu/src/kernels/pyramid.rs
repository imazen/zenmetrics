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

/// 2× upscale: zero-insert at stride 2 in each axis, then 5-tap
/// separable Gaussian with reconstruction gain (×4) — half-gain
/// folded into the kernel by applying GAUSS5 with the standard
/// coefficients and compensating by ×4 on the output, matching
/// cvvdp's `gausspyr_expand` convention.
///
/// `out_w`, `out_h` may be one less than `2*sw`, `2*sh` depending on
/// the original parity (the inverse of `div_ceil` used in reduce).
pub fn gausspyr_expand_scalar(
    src: &[f32],
    sw: usize,
    sh: usize,
    out_w: usize,
    out_h: usize,
    dst: &mut Vec<f32>,
) {
    debug_assert!(out_w == 2 * sw || out_w == 2 * sw - 1);
    debug_assert!(out_h == 2 * sh || out_h == 2 * sh - 1);

    dst.clear();
    dst.resize(out_w * out_h, 0.0);

    // Zero-insertion: build a sparse `out_w × out_h` grid where
    // src[y][x] lands at (2*y, 2*x). Then run separable conv at the
    // full output resolution. The reconstruction gain ×4 cancels the
    // factor-of-4 of zeros introduced.
    let mut sparse = vec![0.0_f32; out_w * out_h];
    for y in 0..sh {
        for x in 0..sw {
            let dy = 2 * y;
            let dx = 2 * x;
            if dy < out_h && dx < out_w {
                sparse[dy * out_w + dx] = src[y * sw + x];
            }
        }
    }

    let k = GAUSS5;
    let mut vscratch = vec![0.0_f32; out_w * out_h];
    for y in 0..out_h {
        for x in 0..out_w {
            let s = |off: isize| -> f32 {
                let r = reflect(y as isize + off, out_h);
                sparse[r * out_w + x]
            };
            vscratch[y * out_w + x] =
                k[0] * s(-2) + k[1] * s(-1) + k[2] * s(0) + k[3] * s(1) + k[4] * s(2);
        }
    }
    for y in 0..out_h {
        for x in 0..out_w {
            let s = |off: isize| -> f32 {
                let r = reflect(x as isize + off, out_w);
                vscratch[y * out_w + r]
            };
            let v = k[0] * s(-2) + k[1] * s(-1) + k[2] * s(0) + k[3] * s(1) + k[4] * s(2);
            dst[y * out_w + x] = 4.0 * v;
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
    fn expand_preserves_constant_signal_interior() {
        // Boundary cells deviate because zero-insertion + reflection
        // breaks the 2-zeros-then-2-source-values cadence at the edge
        // (a reflected pixel re-uses the source row, doubling its
        // contribution). cvvdp's `gausspyr_expand` has explicit edge
        // fix-ups; until those land in the Rust port, this test only
        // checks the kernel-radius interior of a 16×16 expand.
        let src = vec![7.5_f32; 8 * 8];
        let mut dst = Vec::new();
        gausspyr_expand_scalar(&src, 8, 8, 16, 16, &mut dst);
        for y in 4..12 {
            for x in 4..12 {
                let v = dst[y * 16 + x];
                assert!(
                    (v - 7.5).abs() < 1e-5,
                    "interior constant-signal expand produced {v} ≠ 7.5 at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn reduce_then_expand_round_trips_constant_interior() {
        let src = vec![2.0_f32; 16 * 16];
        let mut reduced = Vec::new();
        let (dw, dh) = gausspyr_reduce_scalar(&src, 16, 16, &mut reduced);
        let mut expanded = Vec::new();
        gausspyr_expand_scalar(&reduced, dw, dh, 16, 16, &mut expanded);
        for y in 4..12 {
            for x in 4..12 {
                let v = expanded[y * 16 + x];
                assert!((v - 2.0).abs() < 1e-5, "interior {v} ≠ 2.0 at ({x},{y})");
            }
        }
    }
}
