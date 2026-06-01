//! Verifies that the inline kernel constants match the vendored
//! upstream `cvvdp_parameters.json`. If upstream bumps a calibration
//! value, this test catches the drift.

use cvvdp_gpu::kernels::masking;
use cvvdp_gpu::kernels::pool;

const UPSTREAM_JSON: &str = include_str!("../data/cvvdp_parameters.json");

fn json_f32(obj: &serde_json::Value, key: &str) -> f32 {
    obj[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing key: {key}")) as f32
}

fn json_f32_arr(obj: &serde_json::Value, key: &str) -> Vec<f32> {
    obj[key]
        .as_array()
        .unwrap_or_else(|| panic!("missing array: {key}"))
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

#[test]
fn masking_constants_match_upstream() {
    let v: serde_json::Value = serde_json::from_str(UPSTREAM_JSON).unwrap();
    assert!(
        (masking::MASK_P - json_f32(&v, "mask_p")).abs() < 1e-5,
        "MASK_P drift: ours={} upstream={}",
        masking::MASK_P,
        json_f32(&v, "mask_p")
    );
    assert!(
        (masking::MASK_C - json_f32(&v, "mask_c")).abs() < 1e-5,
        "MASK_C drift: ours={} upstream={}",
        masking::MASK_C,
        json_f32(&v, "mask_c")
    );
    assert!(
        (masking::D_MAX - json_f32(&v, "d_max")).abs() < 1e-5,
        "D_MAX drift: ours={} upstream={}",
        masking::D_MAX,
        json_f32(&v, "d_max")
    );
    let upstream_q = json_f32_arr(&v, "mask_q");
    for (i, &our_q) in masking::MASK_Q.iter().enumerate() {
        assert!(
            (our_q - upstream_q[i]).abs() < 1e-5,
            "MASK_Q[{i}] drift: ours={our_q} upstream={}",
            upstream_q[i]
        );
    }
}

#[test]
fn pool_constants_match_upstream() {
    let v: serde_json::Value = serde_json::from_str(UPSTREAM_JSON).unwrap();
    assert!(
        (pool::BETA_SPATIAL - json_f32(&v, "beta")).abs() < 1e-5,
        "BETA_SPATIAL drift"
    );
    assert!(
        (pool::BETA_BAND - json_f32(&v, "beta_sch")).abs() < 1e-5,
        "BETA_BAND drift"
    );
    assert!(
        (pool::IMAGE_INT - json_f32(&v, "image_int")).abs() < 1e-5,
        "IMAGE_INT drift"
    );
    assert!(
        (pool::JOD_A - json_f32(&v, "jod_a")).abs() < 1e-5,
        "JOD_A drift"
    );
    assert!(
        (pool::JOD_EXP - json_f32(&v, "jod_exp")).abs() < 1e-5,
        "JOD_EXP drift"
    );
    let upstream_bw = json_f32_arr(&v, "baseband_weight");
    for (i, &our_bw) in pool::BASEBAND_W.iter().enumerate() {
        assert!(
            (our_bw - upstream_bw[i]).abs() < 1e-5,
            "BASEBAND_W[{i}] drift: ours={our_bw} upstream={}",
            upstream_bw[i]
        );
    }
}

#[test]
fn version_matches() {
    let v: serde_json::Value = serde_json::from_str(UPSTREAM_JSON).unwrap();
    let upstream_ver = v["version"].as_str().unwrap();
    assert_eq!(
        upstream_ver, "0.5.4",
        "vendored cvvdp_parameters.json version changed — update kernel consts"
    );
}
