//! Fast `mult_mutual_band` with caller-owned scratch.
//!
//! Functionally identical to `crate::kernels::masking::mult_mutual_band`
//! but:
//!
//! - Precomputes `D_MAX_LINEAR = 10^D_MAX`, `MASK_C_LINEAR = 10^MASK_C`
//!   once instead of per-pixel.
//! - Reuses scratch buffers `m_mm_*`, `term_*`, `pu_h` from the parent
//!   `Scratch` struct — no per-band allocation of intermediate
//!   `Vec<f32>` buffers.
//! - Emits `d_a / d_rg / d_vy` into caller-owned `&mut Vec<f32>`.
//! - The per-pixel `safe_pow` calls are vectorized via
//!   [`crate::simd_math::safe_pow_with_offset_into`] (archmage SIMD).
//!   Inputs are pre-offset by `SAFE_EPS > 0` so the unchecked
//!   `pow_midp_unchecked` path is sound. The loop-invariant
//!   `SAFE_EPS.powf(*)` constants are still hoisted once per band.

use alloc::vec::Vec;

use crate::kernels::masking::{D_MAX, MASK_C, MASK_P, MASK_Q, PU_PADSIZE, XCM_3X3};

use crate::simd_math::safe_pow_with_offset_into;
use crate::simd_pyramid::gaussian_blur_sigma3_simd;

const SAFE_EPS: f32 = 1e-5;

/// `mult_mutual_band` writing into caller-owned scratch.
///
/// Arguments:
/// - `t_p_per_ch` / `r_p_per_ch` — CSF-weighted contrasts (already
///   include `CH_GAIN`).
/// - `bw`, `bh` — band dimensions.
/// - `d_a`, `d_rg`, `d_vy` — output buffers (resized to `bw*bh`).
/// - `m_mm_a` / `m_mm_rg` / `m_mm_vy` — scratch buffers used for the
///   per-channel `min(|T|, |R|)` intermediates.
/// - `term_a`, `term_rg`, `term_vy` — scratch buffers used for
///   `safe_pow(|M_mm|, q[ch])` intermediates.
/// - `pu_scratch` — scratch buffer used by `gaussian_blur_sigma3` for
///   large bands (resized internally when needed).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mult_mutual_band_into(
    t_p_per_ch: &[Vec<f32>; 3],
    r_p_per_ch: &[Vec<f32>; 3],
    bw: usize,
    bh: usize,
    d_a: &mut Vec<f32>,
    d_rg: &mut Vec<f32>,
    d_vy: &mut Vec<f32>,
    m_mm_a: &mut Vec<f32>,
    m_mm_rg: &mut Vec<f32>,
    m_mm_vy: &mut Vec<f32>,
    term_a: &mut Vec<f32>,
    term_rg: &mut Vec<f32>,
    term_vy: &mut Vec<f32>,
    pu_scratch: &mut Vec<f32>,
) {
    let n = bw * bh;
    debug_assert_eq!(t_p_per_ch[0].len(), n);

    d_a.clear();
    d_a.resize(n, 0.0);
    d_rg.clear();
    d_rg.resize(n, 0.0);
    d_vy.clear();
    d_vy.resize(n, 0.0);
    m_mm_a.clear();
    m_mm_a.resize(n, 0.0);
    m_mm_rg.clear();
    m_mm_rg.resize(n, 0.0);
    m_mm_vy.clear();
    m_mm_vy.resize(n, 0.0);
    term_a.clear();
    term_a.resize(n, 0.0);
    term_rg.clear();
    term_rg.resize(n, 0.0);
    term_vy.clear();
    term_vy.resize(n, 0.0);

    // Step 1: M_mm_raw = min(|T|, |R|).
    for i in 0..n {
        let ta = t_p_per_ch[0][i].abs();
        let ra = r_p_per_ch[0][i].abs();
        m_mm_a[i] = ta.min(ra);
        let trg = t_p_per_ch[1][i].abs();
        let rrg = r_p_per_ch[1][i].abs();
        m_mm_rg[i] = trg.min(rrg);
        let tvy = t_p_per_ch[2][i].abs();
        let rvy = r_p_per_ch[2][i].abs();
        m_mm_vy[i] = tvy.min(rvy);
    }

    // Step 2: phase_uncertainty per channel.
    let mask_c_lin: f32 = 10.0_f32.powf(MASK_C);
    if bw > PU_PADSIZE && bh > PU_PADSIZE {
        // SIMD σ=3 13-tap separable Gaussian (Chunk 1 of the SIMD
        // optimization plan, replacing the upstream
        // `gaussian_blur_sigma3` which allocated its h-pass buffer
        // internally). We thread `pu_scratch` as the h-pass scratch
        // (shared across the 3 channels — resized to n inside the
        // SIMD entry) and write blur output into the free `term_*`
        // buffers, which aren't consumed until Step 3 below.
        gaussian_blur_sigma3_simd(m_mm_a, bw, bh, pu_scratch, term_a);
        for i in 0..n {
            m_mm_a[i] = term_a[i] * mask_c_lin;
        }
        gaussian_blur_sigma3_simd(m_mm_rg, bw, bh, pu_scratch, term_rg);
        for i in 0..n {
            m_mm_rg[i] = term_rg[i] * mask_c_lin;
        }
        gaussian_blur_sigma3_simd(m_mm_vy, bw, bh, pu_scratch, term_vy);
        for i in 0..n {
            m_mm_vy[i] = term_vy[i] * mask_c_lin;
        }
    } else {
        for i in 0..n {
            m_mm_a[i] *= mask_c_lin;
            m_mm_rg[i] *= mask_c_lin;
            m_mm_vy[i] *= mask_c_lin;
        }
    }

    // Step 3: term[ch] = safe_pow(|M_mm[ch]|, q[ch])
    //                 = (|M_mm[ch]| + SAFE_EPS)^q[ch] - SAFE_EPS^q[ch].
    //
    // After Step 2 the m_mm_* buffers hold `min(|T|, |R|) * mask_c_lin`,
    // which is non-negative — no per-element `abs()` needed. We feed the
    // slices straight into the SIMD pow kernel (one chunked
    // pow_midp_unchecked per channel), which is the bulk of the per-pixel
    // wall time for this stage.
    let q_a = MASK_Q[0];
    let q_rg = MASK_Q[1];
    let q_vy = MASK_Q[2];
    let eps_qa = SAFE_EPS.powf(q_a);
    let eps_qrg = SAFE_EPS.powf(q_rg);
    let eps_qvy = SAFE_EPS.powf(q_vy);
    safe_pow_with_offset_into(m_mm_a, term_a.as_mut_slice(), SAFE_EPS, q_a, eps_qa);
    safe_pow_with_offset_into(m_mm_rg, term_rg.as_mut_slice(), SAFE_EPS, q_rg, eps_qrg);
    safe_pow_with_offset_into(m_mm_vy, term_vy.as_mut_slice(), SAFE_EPS, q_vy, eps_qvy);

    // Step 4: cross-channel pool + masked diff.
    //
    // The per-pixel work splits into:
    //   (a) diff[c]   = |T[c] - R[c]|                     (3 reads + abs + sub)
    //   (b) pow[c]    = (diff[c] + eps)^p - eps^p          (the hot transcendental)
    //   (c) m[c]      = Σ_in XCM[in][c] * term[in]        (3×3 fused-multiply-add)
    //   (d) du[c]     = pow[c] / (1.0 + m[c])
    //   (e) d[c]      = D_MAX_LIN * du[c] / (D_MAX_LIN + du[c])
    //
    // We split this into THREE passes so the heaviest stage (b) can run
    // through the vectorized `safe_pow_with_offset_into`:
    //   pass 1 (scalar, LLVM auto-vectorizes): write |T-R| into the now-free
    //          m_mm_* buffers — these were last touched in Step 3 as the pow
    //          input, are free as scratch from here on.
    //   pass 2 (archmage SIMD): pow into d_* (we don't need d_* until
    //          pass 3 anyway, and it's already the right size).
    //   pass 3 (scalar): the cheap per-pixel pool + clamp; reads pow from
    //          d_* and overwrites d_* with the final diff value.
    let xcm00 = XCM_3X3[0][0];
    let xcm10 = XCM_3X3[1][0];
    let xcm20 = XCM_3X3[2][0];
    let xcm01 = XCM_3X3[0][1];
    let xcm11 = XCM_3X3[1][1];
    let xcm21 = XCM_3X3[2][1];
    let xcm02 = XCM_3X3[0][2];
    let xcm12 = XCM_3X3[1][2];
    let xcm22 = XCM_3X3[2][2];
    let p = MASK_P;
    let eps_p = SAFE_EPS.powf(p);
    let d_max_lin: f32 = 10.0_f32.powf(D_MAX);

    let ta = &t_p_per_ch[0];
    let trg = &t_p_per_ch[1];
    let tvy = &t_p_per_ch[2];
    let ra = &r_p_per_ch[0];
    let rrg = &r_p_per_ch[1];
    let rvy = &r_p_per_ch[2];

    // Pass 1: diff[c] = |T[c] - R[c]| → m_mm_* (free scratch).
    for i in 0..n {
        m_mm_a[i] = (ta[i] - ra[i]).abs();
        m_mm_rg[i] = (trg[i] - rrg[i]).abs();
        m_mm_vy[i] = (tvy[i] - rvy[i]).abs();
    }

    // Pass 2: pow[c] = (diff[c] + eps)^p - eps^p → d_* (vectorized).
    safe_pow_with_offset_into(m_mm_a, d_a.as_mut_slice(), SAFE_EPS, p, eps_p);
    safe_pow_with_offset_into(m_mm_rg, d_rg.as_mut_slice(), SAFE_EPS, p, eps_p);
    safe_pow_with_offset_into(m_mm_vy, d_vy.as_mut_slice(), SAFE_EPS, p, eps_p);

    // Pass 3: cross-channel pool + soft clamp, scalar (auto-vectorizes —
    // no transcendentals left).
    for i in 0..n {
        let t0 = term_a[i];
        let t1 = term_rg[i];
        let t2 = term_vy[i];
        let m0 = xcm00 * t0 + xcm10 * t1 + xcm20 * t2;
        let m1 = xcm01 * t0 + xcm11 * t1 + xcm21 * t2;
        let m2 = xcm02 * t0 + xcm12 * t1 + xcm22 * t2;

        let pow0 = d_a[i];
        let pow1 = d_rg[i];
        let pow2 = d_vy[i];

        let du0 = pow0 / (1.0 + m0);
        let du1 = pow1 / (1.0 + m1);
        let du2 = pow2 / (1.0 + m2);

        // clamp_diff_soft: D_MAX_LIN * d / (D_MAX_LIN + d).
        d_a[i] = d_max_lin * du0 / (d_max_lin + du0);
        d_rg[i] = d_max_lin * du1 / (d_max_lin + du1);
        d_vy[i] = d_max_lin * du2 / (d_max_lin + du2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::masking::mult_mutual_band;

    #[test]
    fn matches_upstream_on_random() {
        // Sweep several sizes spanning the PU_PADSIZE branch boundary.
        let cases: &[(usize, usize)] =
            &[(4, 4), (6, 6), (7, 7), (8, 8), (12, 16), (32, 32), (64, 64)];
        for &(bw, bh) in cases {
            let n = bw * bh;
            let mut s = 0xabcdef01_u32;
            let mut prng = || {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 16) as f32 / 65536.0 - 0.5
            };
            let mut t: [Vec<f32>; 3] = [
                (0..n).map(|_| prng()).collect(),
                (0..n).map(|_| prng()).collect(),
                (0..n).map(|_| prng()).collect(),
            ];
            let mut r: [Vec<f32>; 3] = [
                (0..n).map(|_| prng()).collect(),
                (0..n).map(|_| prng()).collect(),
                (0..n).map(|_| prng()).collect(),
            ];
            // Multiply by a typical CSF×CH_GAIN factor (~10..100).
            for v in &mut t[0] {
                *v *= 30.0;
            }
            for v in &mut t[1] {
                *v *= 50.0;
            }
            for v in &mut t[2] {
                *v *= 20.0;
            }
            for v in &mut r[0] {
                *v *= 30.0;
            }
            for v in &mut r[1] {
                *v *= 50.0;
            }
            for v in &mut r[2] {
                *v *= 20.0;
            }
            let want = mult_mutual_band(&t, &r, bw, bh);

            let mut d_a = Vec::new();
            let mut d_rg = Vec::new();
            let mut d_vy = Vec::new();
            let mut m_mm_a = Vec::new();
            let mut m_mm_rg = Vec::new();
            let mut m_mm_vy = Vec::new();
            let mut term_a = Vec::new();
            let mut term_rg = Vec::new();
            let mut term_vy = Vec::new();
            let mut pu_scratch = Vec::new();
            mult_mutual_band_into(
                &t,
                &r,
                bw,
                bh,
                &mut d_a,
                &mut d_rg,
                &mut d_vy,
                &mut m_mm_a,
                &mut m_mm_rg,
                &mut m_mm_vy,
                &mut term_a,
                &mut term_rg,
                &mut term_vy,
                &mut pu_scratch,
            );
            for i in 0..n {
                let da = (d_a[i] - want[0][i]).abs();
                let drg = (d_rg[i] - want[1][i]).abs();
                let dvy = (d_vy[i] - want[2][i]).abs();
                // f32 noise dominated by the safe_pow arithmetic
                // reassociation — 1e-3 relative is comfortable.
                let s = want[0][i].abs().max(want[1][i].abs()).max(want[2][i].abs());
                let tol = 1e-3_f32 * s.max(1e-6);
                assert!(
                    da < tol && drg < tol && dvy < tol,
                    "case {bw}x{bh} idx {i}: |Δa|={da} |Δrg|={drg} |Δvy|={dvy} tol={tol}"
                );
            }
        }
    }
}
