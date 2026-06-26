//! Filter coefficients for the CPU IW-SSIM port. Bit-identical to
//! `iwssim-gpu/src/filters.rs` by construction (the same literal values).
//!
//! These were previously emitted at build time by the
//! `iwssim-filter-codegen` helper crate. They are pure mathematical
//! constants that never change — pyrtools `binom5`, `fspecial('gaussian',
//! 11, 1.5)` reduced to a separable 1D kernel, and the Wang & Li 2011
//! MS-SSIM scale weights — so on 2026-06-26 the build-time codegen was
//! flattened to committed literals (one fewer workspace member + two
//! fewer build scripts). The literals below are the exact bytes the
//! codegen emitted (`{:.20e}` f32); the `tests` module guards their
//! invariants so accidental edits can't silently drift the taps.

// SSIM_WIN_RADIUS is currently read only in tests; suppress the
// dead-code warning rather than gating it behind cfg(test) (we may
// need it in production paths once boundary handling moves to SIMD).
#[allow(dead_code)]
mod consts {
    // BINOM5 — pyrtools `binom5` taps: `sqrt(2) * [1,4,6,4,1] / 16`.
    // The `sqrt(2)` factor is pyrtools' L2-norm-preserving convention;
    // it matters for byte-for-byte parity with the Python reference.
    pub(crate) const BINOM5_LEN: usize = 5;
    pub(crate) const BINOM5_RADIUS: i32 = 2;
    pub(crate) const BINOM5: [f32; 5] = [
        8.83883476483184465922e-2_f32,
        3.53553390593273786369e-1_f32,
        5.30330085889910707309e-1_f32,
        3.53553390593273786369e-1_f32,
        8.83883476483184465922e-2_f32,
    ];

    // SSIM_WIN_1D — `fspecial('gaussian', 11, 1.5)` applied separably
    // as a 1D 11-tap kernel (outer product reproduces the 2D window).
    pub(crate) const SSIM_WIN_LEN: usize = 11;
    pub(crate) const SSIM_WIN_RADIUS: i32 = 5;
    pub(crate) const SSIM_WIN_1D: [f32; 11] = [
        1.02838008447911008481e-3_f32,
        7.59875813523918502979e-3_f32,
        3.60007721284308288001e-2_f32,
        1.09360689509700015343e-1_f32,
        2.13005537711253689626e-1_f32,
        2.66011724861794363051e-1_f32,
        2.13005537711253689626e-1_f32,
        1.09360689509700015343e-1_f32,
        3.60007721284308288001e-2_f32,
        7.59875813523918502979e-3_f32,
        1.02838008447911008481e-3_f32,
    ];

    // SCALE_WEIGHTS — per-scale MS-SSIM combination weights (β in eq 47
    // of Wang & Li 2011) verbatim from `iwssim.m` / `IW_SSIM_PyTorch.py`.
    pub(crate) const SCALE_WEIGHTS: [f32; 5] =
        [4.48e-2_f32, 2.856e-1_f32, 3.001e-1_f32, 2.363e-1_f32, 1.333e-1_f32];
}

pub(crate) use consts::*;

#[cfg(test)]
mod tests {
    use super::consts::*;

    // Mirrors the invariants the old `iwssim-filter-codegen` crate
    // tested, so flattening the codegen didn't drop the guard.
    #[test]
    fn binom5_sums_to_sqrt2() {
        let s: f64 = BINOM5.iter().map(|&v| v as f64).sum();
        assert!((s - 2.0_f64.sqrt()).abs() < 1e-6, "BINOM5 sum {s} != sqrt(2)");
    }

    #[test]
    fn ssim_win_normalised() {
        let s: f64 = SSIM_WIN_1D.iter().map(|&v| v as f64).sum();
        assert!((s - 1.0).abs() < 1e-6, "SSIM_WIN_1D sum {s} != 1.0");
        assert_eq!(SSIM_WIN_1D.len(), SSIM_WIN_LEN);
    }

    #[test]
    fn scale_weights_normalised() {
        let s: f64 = SCALE_WEIGHTS.iter().map(|&v| v as f64).sum();
        assert!((s - 1.0).abs() < 1e-3, "SCALE_WEIGHTS sum {s} != 1.0");
    }
}
