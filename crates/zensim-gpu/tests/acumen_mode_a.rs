//! Acumen Mode A wiring test.
//!
//! Verifies that enabling `ZensimParams::acumen_mode_a` causes the
//! 12 HF band-energy feature slots (basic indices 10, 11, 12 per
//! channel per scale) to be scaled by per-(scale, channel)
//! castleCSF weights, while the other 192 features remain
//! byte-identical to the default V_22-shipped path.
//!
//! Runs on the CPU CubeCL backend so it doesn't require a GPU on
//! CI. See `cpu_parity.rs` for the GPU↔CPU per-slot parity test.

//! Requires `cuda` or `wgpu` feature: the `cubecl-cpu` runtime
//! triggers `atomic<f32>` codegen failures on zensim's fused
//! kernels (documented in zenmetrics CLAUDE.md), so we run this
//! test against a real GPU backend.

#![cfg(any(feature = "cuda", feature = "wgpu"))]
#![allow(unused_imports)]

use zensim_gpu::{
    Backend, FEATURES_PER_CHANNEL_BASIC, SCALES, TOTAL_FEATURES, ZensimOpaque, ZensimParams,
};

#[cfg(feature = "cuda")]
const BACKEND: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND: Backend = Backend::Wgpu;

fn gradient_pair(w: usize, h: usize, noise: i16) -> (Vec<u8>, Vec<u8>) {
    let mut r = Vec::with_capacity(w * h * 3);
    let mut d = Vec::with_capacity(w * h * 3);
    let mut seed = 0xCAFE_BABE_u32;
    for y in 0..h {
        for x in 0..w {
            let rr = ((x * 255) / w.max(1)) as u8;
            let gg = ((y * 255) / h.max(1)) as u8;
            let bb = (((x + y) * 255) / (w + h).max(1)) as u8;
            r.push(rr);
            r.push(gg);
            r.push(bb);
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let noise_amt = ((seed & 0xFF) as i16 % (noise * 2 + 1)) - noise;
            d.push((rr as i16 + noise_amt).clamp(0, 255) as u8);
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let noise_amt = ((seed & 0xFF) as i16 % (noise * 2 + 1)) - noise;
            d.push((gg as i16 + noise_amt).clamp(0, 255) as u8);
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let noise_amt = ((seed & 0xFF) as i16 % (noise * 2 + 1)) - noise;
            d.push((bb as i16 + noise_amt).clamp(0, 255) as u8);
        }
    }
    (r, d)
}

#[test]
fn legacy_path_is_bit_stable_when_acumen_off() {
    // Two identical Zensim runs with no acumen should produce
    // identical features.
    let (r, d) = gradient_pair(64, 64, 30);

    let mut z1 = ZensimOpaque::new(BACKEND, 64, 64, ZensimParams::new())
        .expect("construct zensim 1");
    let mut z2 = ZensimOpaque::new(BACKEND, 64, 64, ZensimParams::new())
        .expect("construct zensim 2");

    let f1 = z1.compute_features_srgb_u8(&r, &d).expect("compute 1");
    let f2 = z2.compute_features_srgb_u8(&r, &d).expect("compute 2");
    assert_eq!(f1, f2);
}

#[test]
fn acumen_mode_a_scales_only_hf_band_features() {
    let (r, d) = gradient_pair(64, 64, 30);

    let mut z_off = ZensimOpaque::new(BACKEND, 64, 64, ZensimParams::new())
        .expect("construct off");
    let mut z_on = ZensimOpaque::new(
        BACKEND,
        64,
        64,
        ZensimParams::new().with_acumen_mode_a(
            zensim::acumen::viewing::ViewingCondition::LAB_REFERENCE,
        ),
    )
    .expect("construct on");

    let f_off = z_off.compute_features_srgb_u8(&r, &d).expect("compute off");
    let f_on = z_on.compute_features_srgb_u8(&r, &d).expect("compute on");

    // Features 0-9 + the peaks block + masked/iw blocks: byte-
    // identical. Only basic slots 10/11/12 per (scale, channel)
    // can differ.
    let basic_total = SCALES * FEATURES_PER_CHANNEL_BASIC * 3;
    let mut any_basic_10_11_12_diff = false;
    for s in 0..SCALES {
        for ch in 0..3 {
            let bb = s * 3 * FEATURES_PER_CHANNEL_BASIC + ch * FEATURES_PER_CHANNEL_BASIC;
            // Slots 0..=9 must be identical.
            for slot in 0..10 {
                let a = f_off[bb + slot];
                let b = f_on[bb + slot];
                assert_eq!(
                    a, b,
                    "scale={s} ch={ch} basic slot={slot} differs: off={a} on={b}"
                );
            }
            // Slots 10..=12 may differ. They MUST scale by exactly
            // the same factor (the per-(scale, ch) castleCSF
            // weight). When the legacy values are ~0, the acumen
            // path is also ~0 — that's fine.
            let off10 = f_off[bb + 10];
            let on10 = f_on[bb + 10];
            let off11 = f_off[bb + 11];
            let on11 = f_on[bb + 11];
            let off12 = f_off[bb + 12];
            let on12 = f_on[bb + 12];
            // If off-value is non-trivial, then on-value must scale
            // by the same scalar.
            let mut scale_seen: Option<f64> = None;
            for (off, on) in [(off10, on10), (off11, on11), (off12, on12)] {
                if off.abs() > 1e-8 {
                    let s_obs = on / off;
                    if let Some(prev) = scale_seen {
                        assert!(
                            (s_obs - prev).abs() < 1e-4,
                            "scale factor inconsistent within (s={s},ch={ch}): {} vs {}",
                            prev,
                            s_obs
                        );
                    } else {
                        scale_seen = Some(s_obs);
                    }
                    if (off - on).abs() > 1e-6 {
                        any_basic_10_11_12_diff = true;
                    }
                }
            }
        }
    }
    // After basic block (228 features for SCALES=4): all remaining
    // feature slots must be byte-identical.
    for i in basic_total..TOTAL_FEATURES {
        assert_eq!(
            f_off[i], f_on[i],
            "feature {i} (after basic block) should be identical"
        );
    }
    assert!(
        any_basic_10_11_12_diff,
        "expected at least one HF band-energy feature to differ; acumen weighting may not be applied"
    );
}

#[test]
fn acumen_mode_a_with_different_viewings_produces_different_features() {
    let (r, d) = gradient_pair(64, 64, 30);
    let mut z_lab = ZensimOpaque::new(
        BACKEND,
        64,
        64,
        ZensimParams::new().with_acumen_mode_a(
            zensim::acumen::viewing::ViewingCondition::LAB_REFERENCE,
        ),
    )
    .expect("lab");
    let mut z_mobile = ZensimOpaque::new(
        BACKEND,
        64,
        64,
        ZensimParams::new().with_acumen_mode_a(
            zensim::acumen::viewing::ViewingCondition::MOBILE_RETINA,
        ),
    )
    .expect("mobile");

    let f_lab = z_lab
        .compute_features_srgb_u8(&r, &d)
        .expect("compute lab");
    let f_mobile = z_mobile
        .compute_features_srgb_u8(&r, &d)
        .expect("compute mobile");
    assert_ne!(
        f_lab, f_mobile,
        "lab-reference and mobile-retina viewings should produce different features"
    );
}
