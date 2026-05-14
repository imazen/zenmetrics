//! Parity test for `kernels::masking::mult_mutual_pixel` against
//! pycvvdp v0.5.4's `apply_masking_model` in `mult-mutual` mode with
//! xchannel masking on.
//!
//! The test inputs are a deterministic 4×4×3 pair of CSF-weighted
//! contrasts. 4×4 is below cvvdp's `pu_padsize = 6` threshold, so
//! `phase_uncertainty` skips the Gaussian blur and the simple
//! `M * 10^mask_c` form applies — which is what the Rust port
//! currently implements. The blur path lands when whole-image
//! parity is wired (coarse-level bands > 6 px).

use cvvdp_gpu::kernels::masking::mult_mutual_pixel;

#[rustfmt::skip]
const T_P: [f32; 48] = [
    -1.49736404e-02,  1.07288718e+00, -1.64609027e+00, -1.47187805e+00,
    -7.70308733e-01,  5.36314726e-01, -3.96263599e-02,  1.58577895e+00,
    -1.77488089e-01,  5.29225111e-01, -6.04426146e-01, -3.93130779e-01,
    -1.91069698e+00, -1.32456422e+00, -8.24446201e-01,  7.40871429e-02,
     7.90670395e-01,  1.20004559e+00, -1.35588217e+00, -8.70925665e-01,
     7.26434231e-01,  1.66077590e+00, -4.11600351e-01,  1.49662352e+00,
    -3.22366714e-01,  2.11628199e-01,  1.81095243e+00, -1.85534072e+00,
    -1.25907588e+00, -5.06330490e-01, -7.79599905e-01,  1.72800159e+00,
    -1.29635930e+00, -9.20665741e-01, -1.39728093e+00, -1.87312198e+00,
    -1.16748095e+00,  1.71919608e+00,  8.92436743e-01,  9.69345093e-01,
     1.05183125e-01, -1.02536702e+00,  3.38369370e-01, -1.86738944e+00,
    -1.44513249e+00, -1.03105998e+00,  1.26187587e+00,  1.17264247e+00,
];

#[rustfmt::skip]
const R_P: [f32; 48] = [
    -8.86990070e-01, -7.21647739e-02,  1.27912140e+00,  1.98826623e+00,
     7.93764353e-01,  2.70185709e-01,  1.34097266e+00, -1.17760468e+00,
     3.72688055e-01, -1.55061102e+00, -1.38617229e+00, -1.03316712e+00,
     9.04946089e-01,  8.04320812e-01, -1.18470502e+00,  6.04214191e-01,
     1.09794402e+00, -2.52434731e-01,  7.63630867e-02,  4.63409424e-01,
     1.24075317e+00,  1.92038822e+00, -1.54124713e+00, -7.32939482e-01,
     7.86019802e-01,  1.65709877e+00,  1.74041462e+00,  1.76471353e+00,
     3.98029089e-01, -1.73916531e+00,  1.83984995e-01, -1.25121069e+00,
    -1.86390829e+00,  1.77698493e+00,  1.52071953e+00, -1.99505591e+00,
     3.74344110e-01, -3.36920023e-01, -3.29122305e-01, -9.15513754e-01,
     7.69112349e-01, -1.18460703e+00,  7.33182669e-01,  1.01141620e+00,
     1.43174314e+00,  7.47822285e-01, -1.97947049e+00, -1.29739380e+00,
];

#[rustfmt::skip]
const D_OUT: [f32; 48] = [
     6.91511929e-01,  1.33346593e+00,  9.47845364e+00,  1.23217335e+01,
     2.55601931e+00,  4.36255075e-02,  2.05631208e+00,  8.67819405e+00,
     2.55833626e-01,  4.91521358e+00,  4.80585009e-01,  3.04408103e-01,
     8.95886803e+00,  5.09732580e+00,  8.99003148e-02,  2.17990384e-01,
     6.87788278e-02,  2.31245279e+00,  2.23293686e+00,  1.89445734e+00,
     2.20964149e-01,  4.59021628e-02,  1.31251597e+00,  6.01464033e+00,
     1.25759399e+00,  2.28569317e+00,  2.39155861e-03,  1.69824619e+01,
     3.09952235e+00,  1.59586537e+00,  9.14802670e-01,  1.13281679e+01,
     2.76414394e-01,  9.21547604e+00,  1.08664722e+01,  8.39139707e-03,
     2.63685131e+00,  5.02410698e+00,  1.56636310e+00,  4.12625885e+00,
     3.94951612e-01,  1.55574214e-02,  1.21139348e-01,  1.05788078e+01,
     1.05528469e+01,  3.63435221e+00,  1.37246819e+01,  7.56529856e+00,
];

#[test]
fn mult_mutual_pixel_matches_pycvvdp_4x4() {
    // Layout: each plane is 16 elements row-major; 3 planes back to
    // back (channels A, RG, VY).
    let plane = 16;
    let mut max_rel = 0.0_f32;
    let mut worst_idx = 0usize;
    let mut worst_got = [0.0_f32; 3];
    let mut worst_exp = [0.0_f32; 3];
    for i in 0..plane {
        let t = [T_P[i], T_P[plane + i], T_P[2 * plane + i]];
        let r = [R_P[i], R_P[plane + i], R_P[2 * plane + i]];
        let got = mult_mutual_pixel(t, r);
        let exp = [D_OUT[i], D_OUT[plane + i], D_OUT[2 * plane + i]];
        for c in 0..3 {
            let rel = ((got[c] - exp[c]) / exp[c].abs().max(1e-6)).abs();
            if rel > max_rel {
                max_rel = rel;
                worst_idx = i;
                worst_got = got;
                worst_exp = exp;
            }
        }
    }
    // Loose tol — the masking math chains safe_pow + cross-channel
    // sums + soft-clamp, each amplifying f32 noise. 1e-3 catches
    // structural mistakes (wrong matrix, wrong exponent) while
    // absorbing accumulated rounding.
    assert!(
        max_rel < 1e-3,
        "mult_mutual_pixel max relative error vs pycvvdp = {max_rel}; \
         worst pixel idx {worst_idx}, got {worst_got:?}, expected {worst_exp:?}"
    );
}
