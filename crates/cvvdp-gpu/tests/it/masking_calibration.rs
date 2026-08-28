//! zenmetrics#14 §3 "runtime parameter loading" — the masking stage's
//! calibration (`mask_p`, `mask_q`, `10^mask_c`, `10^d_max`) is a runtime
//! kernel argument now: `MaskingCalibration::V0_5_4` reproduces the
//! former baked literals bit-for-bit (the pycvvdp-golden parity suite in
//! `pipeline_score` is the regression gate), `from_upstream_json` loads the
//! vendored `cvvdp_parameters.json` to the same numbers, and a perturbed
//! calibration visibly changes the JOD (proof the kernels consume the
//! runtime values rather than a stale literal).

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cvvdp_gpu::kernels::masking::{D_MAX, MASK_C, MASK_P, MASK_Q};
use cvvdp_gpu::{Cvvdp, CvvdpParams, MaskingCalibration};

use crate::common;
use common::{Backend, load_rgb_bytes};

/// The baked linear-unit literals sit 2.4e-4 / 3.0e-4 relative above the
/// true `10^const` (see `kernels::masking::D_MAX_LIN`); this is the bound
/// on that documented provenance offset.
const LIN_LITERAL_REL: f32 = 5e-4;

const UPSTREAM_JSON: &str = include_str!("../../data/cvvdp_parameters.json");

/// Atomic-reorder noise band for a repeated JOD on the same instance.
const JOD_TOL: f64 = 1e-4;

#[test]
fn default_is_v0_5_4_and_reproduces_the_kernel_literals() {
    let c = MaskingCalibration::default();
    assert_eq!(c, MaskingCalibration::V0_5_4);
    assert_eq!(c.mask_p, MASK_P);
    assert_eq!(c.mask_q, MASK_Q);
    // The linear-unit consts are the exact literals the kernels baked;
    // they sit within the documented provenance offset of 10^const.
    let d_rel = (c.d_max_lin - 10.0_f32.powf(D_MAX)).abs() / c.d_max_lin;
    let p_rel = (c.pu_scale_lin - 10.0_f32.powf(MASK_C)).abs() / c.pu_scale_lin;
    assert!(
        d_rel < LIN_LITERAL_REL,
        "d_max_lin off 10^D_MAX by {d_rel:.2e}"
    );
    assert!(
        p_rel < LIN_LITERAL_REL,
        "pu_scale_lin off 10^MASK_C by {p_rel:.2e}"
    );
}

#[test]
fn vendored_upstream_json_loads_to_v0_5_4() {
    let j = MaskingCalibration::from_upstream_json(UPSTREAM_JSON).expect("vendored json parses");
    let v = MaskingCalibration::V0_5_4;
    assert!(
        (j.mask_p - v.mask_p).abs() < 1e-5,
        "{} vs {}",
        j.mask_p,
        v.mask_p
    );
    for c in 0..3 {
        assert!((j.mask_q[c] - v.mask_q[c]).abs() < 1e-5, "q[{c}]");
    }
    // The JSON path computes the true powers; the const carries the baked
    // literals — they differ by the documented provenance offset only.
    assert!((j.d_max_lin - v.d_max_lin).abs() / v.d_max_lin < LIN_LITERAL_REL);
    assert!((j.pu_scale_lin - v.pu_scale_lin).abs() / v.pu_scale_lin < LIN_LITERAL_REL);
    assert!((j.d_max_lin - 10.0_f32.powf(D_MAX)).abs() / j.d_max_lin < 1e-5);
    assert!((j.pu_scale_lin - 10.0_f32.powf(MASK_C)).abs() / j.pu_scale_lin < 1e-5);
}

#[test]
fn malformed_json_is_an_error() {
    assert!(MaskingCalibration::from_upstream_json("{}").is_err());
    assert!(MaskingCalibration::from_upstream_json("not json").is_err());
    let neg = r#"{"mask_p": -1.0, "mask_q": [1.0, 2.0, 3.0], "mask_c": -0.8, "d_max": 2.5}"#;
    assert!(MaskingCalibration::from_upstream_json(neg).is_err());
    let short = r#"{"mask_p": 2.0, "mask_q": [1.0, 2.0], "mask_c": -0.8, "d_max": 2.5}"#;
    assert!(MaskingCalibration::from_upstream_json(short).is_err());
}

/// Perturbation probes run on corpus JPEG pairs, NOT the synthetic offset
/// pair: a uniform DC offset only lights up the baseband, whose D comes
/// from `diff_abs_3ch_kernel` (no masking), so masking calibration
/// changes are below f32 resolution there (measured 2026-08-28: mask_p ×
/// 1.25 moved it 3e-6 JOD, d_max × 0.5 moved it 0 bits). The q=5 pair has
/// strong mid-band differences (soft clip engaged → `d_max` bites); the
/// q=45 pair is mild (below the clip → `mask_p` bites).
#[test]
fn runtime_calibration_is_consumed_by_the_masking_kernels() {
    let client = Backend::client(&Default::default());
    let mut c = Cvvdp::<Backend>::new(client, 256, 256, CvvdpParams::PLACEHOLDER).expect("new");
    assert_eq!(c.masking_calibration(), MaskingCalibration::V0_5_4);
    let cr = load_rgb_bytes(&zenmetrics_corpus::source_png(), 256, 256);
    let strong = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(5), 256, 256);
    let mild = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(45), 256, 256);

    // Probe 1: strong pair — halve the soft-clip ceiling.
    let base_s = c.score(&cr, &strong).expect("strong baseline");
    let mut low_clip = MaskingCalibration::V0_5_4;
    low_clip.d_max_lin *= 0.5;
    c.set_masking_calibration(low_clip);
    assert_eq!(c.masking_calibration(), low_clip);
    let moved_s = c.score(&cr, &strong).expect("perturbed d_max");
    assert!(
        (moved_s - base_s).abs() > 1e-3,
        "d_max_lin × 0.5 left the q=5 JOD unchanged ({base_s} → {moved_s}); the kernels are not \
         reading the runtime calibration"
    );
    // Restoring V0_5_4 restores the score (to atomic-reorder noise).
    c.set_masking_calibration(MaskingCalibration::V0_5_4);
    let back_s = c.score(&cr, &strong).expect("restored");
    assert!((back_s - base_s).abs() <= JOD_TOL, "{base_s} → {back_s}");

    // Probe 2: mild pair — raise the excitation exponent.
    let base_m = c.score(&cr, &mild).expect("mild baseline");
    let mut hot = MaskingCalibration::V0_5_4;
    hot.mask_p *= 1.25;
    c.set_masking_calibration(hot);
    let moved_m = c.score(&cr, &mild).expect("perturbed mask_p");
    assert!(
        (moved_m - base_m).abs() > 1e-3,
        "mask_p × 1.25 left the q=45 JOD unchanged ({base_m} → {moved_m})"
    );
    c.set_masking_calibration(MaskingCalibration::V0_5_4);
    let back_m = c.score(&cr, &mild).expect("mild restored");
    assert!((back_m - base_m).abs() <= JOD_TOL, "{base_m} → {back_m}");

    // The loaded vendored JSON (true 10^const powers, 2–3e-4 rel off the
    // baked literals) scores like the const default within calibration
    // noise on both pairs.
    let json = MaskingCalibration::from_upstream_json(UPSTREAM_JSON).expect("json");
    c.set_masking_calibration(json);
    let json_s = c.score(&cr, &strong).expect("json calib strong");
    assert!((json_s - base_s).abs() <= 1e-3, "{base_s} → {json_s}");
    let json_m = c.score(&cr, &mild).expect("json calib mild");
    assert!((json_m - base_m).abs() <= 1e-3, "{base_m} → {json_m}");
    eprintln!(
        "runtime calib: q5 base {base_s:.6} d_max×0.5 {moved_s:.6} json {json_s:.6}; \
         q45 base {base_m:.6} mask_p×1.25 {moved_m:.6} json {json_m:.6}"
    );
}
