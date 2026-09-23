//! Temporal channel filters for the cvvdp *video* path — a port of
//! pycvvdp `cvvdp_metric.py::get_temporal_filters` (v0.5.7, default
//! `temp_filter` Gaussian band-pass branch; `hp_trans`/`grad_trans`
//! are not ported).
//!
//! Upstream math (torch, f32):
//!
//! 1. `N = 2*ceil(0.125*fps) + 1` — always odd.
//! 2. `N_omega = N/2 + 1` response bins;
//!    `omega[k] = k * (fps/2) / (N_omega-1)` (≡ `linspace(0, fps/2,
//!    N_omega)`). NOTE the response grid is *not* the DFT bin grid for
//!    odd `N` — this is upstream behaviour, replicated verbatim.
//! 3. Sustained channels 0..=2: `R[k] = exp(-omega[k]^beta_tf[c] /
//!    sigma_tf[c])` (low-pass).
//! 4. Transient channel 3:
//!    `R[k] = exp(-(omega[k]^beta_tf[3] - 5^beta_tf[3])^2 / sigma_tf[3])`
//!    (Gaussian band-pass centred on 5 Hz).
//! 5. `F[c] = fftshift(Re(irfft(R[c], norm="backward", n=N)))` — a
//!    `1/N`-normalised real IDFT of a real spectrum, which for odd `N`
//!    is the cosine sum
//!    `x[m] = (R[0] + 2*sum_{k=1}^{N_omega-1} R[k]*cos(2*pi*k*m/N)) / N`.
//!
//! The resulting kernel is symmetric about its centre tap. pycvvdp
//! applies it as a *causal* FIR over the sliding window
//! (`R_f = sum_f sw_buf[f] * flipud(F)[f]`, window ending at the
//! current frame), so the emitted filter carries a `(N-1)/2`-frame
//! group delay: output frame `t` reflects content centred at
//! `t - (N-1)/2`. The tap array returned here is in the same causal
//! order as upstream's `F[c]` — `taps[k]` multiplies frame `n - k`,
//! `taps[0]` is the kernel edge, `taps[N/2]` the kernel centre.
//!
//! Verified against the pycvvdp v0.5.7 taps to ≤ 7e-8 absolute
//! (computed in f64, cast to f32 — the residual is torch's own f32
//! FFT rounding).

use alloc::vec::Vec;

/// `sigma_tf` from `vvdp_data/cvvdp_parameters.json` (pycvvdp v0.5.7).
/// Per-channel Gaussian width of the temporal response.
pub(crate) const SIGMA_TF: [f64; 4] = [5.79336, 14.1255, 6.63661, 0.12314];

/// `beta_tf` from `vvdp_data/cvvdp_parameters.json` (pycvvdp v0.5.7).
/// Per-channel frequency exponent of the temporal response.
pub(crate) const BETA_TF: [f64; 4] = [1.3314, 1.1196, 0.947901, 0.1898];

/// Transient-channel band-pass centre frequency in Hz — the `5.0`
/// literal in `get_temporal_filters`.
pub(crate) const TRANSIENT_CENTER_HZ: f64 = 5.0;

/// Filter length `N = 2*ceil(0.125*fps) + 1` (always odd).
///
/// `frames_per_s` must be finite and > 0 — callers validate.
pub(crate) fn temporal_filter_len(frames_per_s: f32) -> usize {
    debug_assert!(frames_per_s.is_finite() && frames_per_s > 0.0);
    let n = (0.125 * frames_per_s as f64).ceil() as usize * 2 + 1;
    debug_assert!(n % 2 == 1);
    n
}

/// FIR taps for the 4 temporal channels at `frames_per_s` Hz, causal
/// order (`taps[c][k]` weights the frame `k` positions back).
///
/// Returns `[4][N]` with `N = temporal_filter_len(fps)`. Channel
/// order: sustained A, sustained RG, sustained VY, transient A.
pub(crate) fn temporal_filters(frames_per_s: f32) -> [Vec<f32>; 4] {
    let n = temporal_filter_len(frames_per_s);
    let n_omega = n / 2 + 1;
    let fps = frames_per_s as f64;
    let n_f = n as f64;
    let shift = n.div_ceil(2);

    let mut out: [Vec<f32>; 4] = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for (c, taps) in out.iter_mut().enumerate() {
        // Response spectrum R[k], k = 0..n_omega.
        let mut resp = Vec::with_capacity(n_omega);
        for k in 0..n_omega {
            let omega = k as f64 * (fps / 2.0) / (n_omega - 1) as f64;
            let w = omega.powf(BETA_TF[c]);
            let r = if c < 3 {
                (-w / SIGMA_TF[c]).exp()
            } else {
                let dw = w - TRANSIENT_CENTER_HZ.powf(BETA_TF[3]);
                (-dw * dw / SIGMA_TF[3]).exp()
            };
            resp.push(r);
        }
        // Odd-N real IDFT + fftshift for odd N:
        // taps[j] = x[(j + (N+1)/2) mod N],
        // x[m] = (R[0] + 2*Σ_{k≥1} R[k]·cos(2πkm/N)) / N.
        for j in 0..n {
            let m = (j + shift) % n;
            let mut acc = resp[0];
            for (k, &rk) in resp.iter().enumerate().skip(1) {
                acc += 2.0 * rk * (2.0 * core::f64::consts::PI * k as f64 * m as f64 / n_f).cos();
            }
            taps.push((acc / n_f) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_len_matches_upstream_formula() {
        // N = 2*ceil(0.125*fps)+1 — checked against the pycvvdp dump.
        assert_eq!(temporal_filter_len(24.0), 7);
        assert_eq!(temporal_filter_len(30.0), 9);
        assert_eq!(temporal_filter_len(60.0), 17);
    }

    #[test]
    fn sustained_dc_near_one_transient_dc_near_zero() {
        for fps in [24.0f32, 30.0, 60.0] {
            let taps = temporal_filters(fps);
            for (c, t) in taps.iter().enumerate().take(3) {
                let s: f32 = t.iter().sum();
                assert!(
                    (s - 1.0).abs() < 1e-5,
                    "fps={fps} ch={c} sustained DC sum {s}"
                );
            }
            let s3: f32 = taps[3].iter().sum();
            assert!(s3.abs() < 1e-4, "fps={fps} transient DC sum {s3}");
        }
    }

    #[test]
    fn taps_are_symmetric() {
        for fps in [24.0f32, 30.0, 60.0] {
            let taps = temporal_filters(fps);
            let n = taps[0].len();
            for t in &taps {
                for k in 0..n / 2 {
                    assert_eq!(t[k].to_bits(), t[n - 1 - k].to_bits());
                }
            }
        }
    }

    // pycvvdp v0.5.7 `get_temporal_filters` taps, dumped by
    // `scripts/cvvdp_goldens/build_video_goldens.py` into
    // `scripts/cvvdp_goldens/video_goldens.json` (`temporal_filters`
    // table, rounded to 10 decimals). Embedded as consts so the unit
    // test needs no filesystem access.
    #[rustfmt::skip]
    const T24: [[f32; 7]; 4] = [
        [6.7387618100e-2, 1.0669139030e-1, 1.9620880480e-1, 2.5942444800e-1, 1.9620880480e-1, 1.0669139030e-1, 6.7387618100e-2],
        [2.4482384300e-2, 2.9598236100e-2, 1.5759317580e-1, 5.7665240760e-1, 1.5759317580e-1, 2.9598236100e-2, 2.4482384300e-2],
        [4.3334052000e-2, 5.5673081400e-2, 1.7041555050e-1, 4.6115469930e-1, 1.7041555050e-1, 5.5673081400e-2, 4.3334052000e-2],
        [-1.3349041340e-1, -1.7868687210e-1, -4.0082901700e-2, 7.0452070240e-1, -4.0082901700e-2, -1.7868687210e-1, -1.3349041340e-1],
    ];
    #[rustfmt::skip]
    const T30: [[f32; 9]; 4] = [
        [4.6771094200e-2, 6.4188815700e-2, 1.0736732930e-1, 1.7480950060e-1, 2.1372655030e-1, 1.7480950060e-1, 1.0736733680e-1, 6.4188815700e-2, 4.6771086800e-2],
        [1.5257254200e-2, 2.4257481100e-2, 3.4000299900e-2, 1.6903038320e-1, 5.1490914820e-1, 1.6903039810e-1, 3.4000299900e-2, 2.4257481100e-2, 1.5257244900e-2],
        [2.9910691100e-2, 3.9635770000e-2, 5.7371247600e-2, 1.7125067110e-1, 4.0366327760e-1, 1.7125067110e-1, 5.7371247600e-2, 3.9635770000e-2, 2.9910694800e-2],
        [-1.0165669020e-1, -1.1033840480e-1, -1.5002247690e-1, 3.1385585700e-2, 6.6126430030e-1, 3.1385600600e-2, -1.5002247690e-1, -1.1033840480e-1, -1.0165669770e-1],
    ];
    #[rustfmt::skip]
    const T60: [[f32; 17]; 4] = [
        [2.4023566400e-2, 2.6379292800e-2, 3.1514603600e-2, 4.0214844000e-2, 5.3314931700e-2, 7.0816613700e-2, 9.0425014500e-2, 1.0672388230e-1, 1.1317450550e-1, 1.0672388230e-1, 9.0425014500e-2, 7.0816613700e-2, 5.3314931700e-2, 4.0214844000e-2, 3.1514603600e-2, 2.6379292800e-2, 2.4023566400e-2],
        [8.2237441000e-3, 9.0745213000e-3, 1.0319180800e-2, 1.4051385200e-2, 1.9178895300e-2, 3.4498147700e-2, 6.6441744600e-2, 1.8095543980e-1, 3.1451383230e-1, 1.8095543980e-1, 6.6441744600e-2, 3.4498147700e-2, 1.9178895300e-2, 1.4051385200e-2, 1.0319180800e-2, 9.0745213000e-3, 8.2237441000e-3],
        [1.5770252800e-2, 1.6824966300e-2, 1.8817558900e-2, 2.3212464500e-2, 3.0096935100e-2, 4.5616760800e-2, 7.5601048800e-2, 1.5528406200e-1, 2.3755194250e-1, 1.5528406200e-1, 7.5601048800e-2, 4.5616760800e-2, 3.0096935100e-2, 2.3212464500e-2, 1.8817558900e-2, 1.6824966300e-2, 1.5770252800e-2],
        [-5.2364628800e-2, -5.4407488600e-2, -5.9709508000e-2, -6.4904145900e-2, -7.2556927800e-2, -6.4987704200e-2, -2.5122856700e-2, 1.7744171620e-1, 4.3322339650e-1, 1.7744171620e-1, -2.5122856700e-2, -6.4987704200e-2, -7.2556927800e-2, -6.4904145900e-2, -5.9709508000e-2, -5.4407488600e-2, -5.2364628800e-2],
    ];

    /// V1 parity pin: every tap within 1e-6 of pycvvdp v0.5.7.
    /// (Measured closed-form-vs-torch deviation is ≤ 7e-8; the test
    /// keeps the 1e-6 gate from the work order.)
    #[test]
    fn taps_match_pycvvdp_0_5_7() {
        let cases: [(f32, &[&[f32]]); 3] = [
            (24.0, &[&T24[0], &T24[1], &T24[2], &T24[3]]),
            (30.0, &[&T30[0], &T30[1], &T30[2], &T30[3]]),
            (60.0, &[&T60[0], &T60[1], &T60[2], &T60[3]]),
        ];
        for (fps, expected) in cases {
            let taps = temporal_filters(fps);
            for (c, exp) in expected.iter().enumerate() {
                assert_eq!(taps[c].len(), exp.len(), "fps={fps} ch={c} len");
                for (k, &e) in exp.iter().enumerate() {
                    assert!(
                        (taps[c][k] - e).abs() < 1e-6,
                        "fps={fps} ch={c} tap[{k}]: {} vs {e}",
                        taps[c][k]
                    );
                }
            }
        }
    }
}
