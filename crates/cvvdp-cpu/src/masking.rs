//! Fast `mult_mutual_band` with caller-owned scratch.
//!
//! Functionally identical to `cvvdp_gpu::kernels::masking::mult_mutual_band`
//! but:
//!
//! - Precomputes `D_MAX_LINEAR = 10^D_MAX`, `MASK_C_LINEAR = 10^MASK_C`
//!   once instead of per-pixel.
//! - Reuses scratch buffers `m_mm_*`, `term_*`, `pu_h` from the parent
//!   `Scratch` struct — no per-band allocation of intermediate
//!   `Vec<f32>` buffers.
//! - Emits `d_a / d_rg / d_vy` into caller-owned `&mut Vec<f32>`.
//! - The `safe_pow` calls remain (correctness-critical) but the
//!   sentinel `eps^p` is precomputed.

use alloc::vec::Vec;

use cvvdp_gpu::kernels::masking::{
    D_MAX, MASK_C, MASK_P, MASK_Q, PU_PADSIZE, XCM_3X3, gaussian_blur_sigma3,
};

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
        // Blur each channel then scale. We re-use `gaussian_blur_sigma3`'s
        // upstream impl which allocates internally. (Optimizing the blur
        // itself is the next perf chunk; for now we wrap.)
        let blur_a = gaussian_blur_sigma3(m_mm_a, bw, bh);
        let blur_rg = gaussian_blur_sigma3(m_mm_rg, bw, bh);
        let blur_vy = gaussian_blur_sigma3(m_mm_vy, bw, bh);
        for i in 0..n {
            m_mm_a[i] = blur_a[i] * mask_c_lin;
            m_mm_rg[i] = blur_rg[i] * mask_c_lin;
            m_mm_vy[i] = blur_vy[i] * mask_c_lin;
        }
        // Drop the upstream-allocated blur buffers — we can wire them
        // into `pu_scratch` in a future chunk via a custom in-place
        // blur impl.
        let _ = pu_scratch;
    } else {
        for i in 0..n {
            m_mm_a[i] *= mask_c_lin;
            m_mm_rg[i] *= mask_c_lin;
            m_mm_vy[i] *= mask_c_lin;
        }
    }

    // Step 3: term[ch] = safe_pow(|M_mm[ch]|, q[ch]).
    let q_a = MASK_Q[0];
    let q_rg = MASK_Q[1];
    let q_vy = MASK_Q[2];
    let eps_qa = SAFE_EPS.powf(q_a);
    let eps_qrg = SAFE_EPS.powf(q_rg);
    let eps_qvy = SAFE_EPS.powf(q_vy);
    for i in 0..n {
        let va = m_mm_a[i].abs();
        let vrg = m_mm_rg[i].abs();
        let vvy = m_mm_vy[i].abs();
        term_a[i] = (va + SAFE_EPS).powf(q_a) - eps_qa;
        term_rg[i] = (vrg + SAFE_EPS).powf(q_rg) - eps_qrg;
        term_vy[i] = (vvy + SAFE_EPS).powf(q_vy) - eps_qvy;
    }

    // Step 4: cross-channel pool + masked diff.
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

    for i in 0..n {
        let t0 = term_a[i];
        let t1 = term_rg[i];
        let t2 = term_vy[i];
        // M[c] = sum_in XCM[in][c] * term[in]
        let m0 = xcm00 * t0 + xcm10 * t1 + xcm20 * t2;
        let m1 = xcm01 * t0 + xcm11 * t1 + xcm21 * t2;
        let m2 = xcm02 * t0 + xcm12 * t1 + xcm22 * t2;

        let diff0 = (ta[i] - ra[i]).abs();
        let diff1 = (trg[i] - rrg[i]).abs();
        let diff2 = (tvy[i] - rvy[i]).abs();

        let pow0 = (diff0 + SAFE_EPS).powf(p) - eps_p;
        let pow1 = (diff1 + SAFE_EPS).powf(p) - eps_p;
        let pow2 = (diff2 + SAFE_EPS).powf(p) - eps_p;

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
    use cvvdp_gpu::kernels::masking::mult_mutual_band;

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
