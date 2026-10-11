//! Each available tier is compared directly; unsupported hardware is not invoked.
use super::*;
use archmage::SimdToken;

macro_rules! check_tier {
    ($token:expr, $width:ident) => {{
        let token = $token;
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 257] {
            let edge = [
                0.0,
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                1e-5,
                0.01,
                1.0,
                200.0,
            ];
            let xs: Vec<f32> = (0..n).map(|i| edge[i % edge.len()]).collect();
            let mut scalar = vec![0.0; n];
            let mut simd = vec![0.0; n];
            for p in [0.5_f32, 1.3, 2.26, 3.7] {
                let ep = 1e-5_f32.powf(p);
                width8::safe_pow_with_offset_kernel::<_, Midp>(
                    ScalarToken,
                    &xs,
                    &mut scalar,
                    1e-5,
                    p,
                    ep,
                );
                $width::safe_pow_with_offset_kernel::<_, Midp>(token, &xs, &mut simd, 1e-5, p, ep);
                for i in 0..n {
                    assert!(
                        (scalar[i] - simd[i]).abs() <= 5e-5 * scalar[i].abs().max(1e-6),
                        "safe_pow n={n} i={i} p={p} scalar={} simd={} delta={}",
                        scalar[i],
                        simd[i],
                        (scalar[i] - simd[i]).abs()
                    );
                }
                let zeros = vec![0.0; n];
                $width::vabs_diff_pow_kernel::<_, Midp>(token, &mut simd, &xs, &zeros, 1e-5, p, ep);
                for i in 0..n {
                    assert!(
                        (scalar[i] - simd[i]).abs() <= 5e-5 * scalar[i].abs().max(1e-6),
                        "fused offset power n={n} i={i} p={p}"
                    );
                    if xs[i] + 1e-5 == 1e-5 {
                        assert_eq!(simd[i].to_bits(), 0.0_f32.to_bits());
                    }
                }
            }
            let xs: Vec<f32> = (0..n).map(|i| -20.0 + (i % 81) as f32 * 0.5).collect();
            width8::vexp_kernel::<_, Midp>(ScalarToken, &xs, &mut scalar);
            $width::vexp_kernel::<_, Midp>(token, &xs, &mut simd);
            for i in 0..n {
                assert!((scalar[i] - simd[i]).abs() <= 5e-5 * scalar[i].abs());
            }
            let xs: Vec<f32> = (0..n)
                .map(|i| [1e-5, 0.01, 0.999, 1.0, 1.001, 200.0][i % 6])
                .collect();
            width8::vlog_kernel::<_, Midp>(ScalarToken, &xs, &mut scalar);
            $width::vlog_kernel::<_, Midp>(token, &xs, &mut simd);
            for i in 0..n {
                assert!((scalar[i] - simd[i]).abs() <= 5e-4 * scalar[i].abs().max(1e-4));
            }
            // Arithmetic has an IEEE domain, including NaNs and subnormals.
            let xs: Vec<f32> = (0..n)
                .map(|i| [f32::NAN, -0.0, f32::from_bits(1), -1.0, f32::INFINITY][i % 5])
                .collect();
            width8::vabs_diff_kernel::<_, Midp>(ScalarToken, &mut scalar, &xs, &xs);
            $width::vabs_diff_kernel::<_, Midp>(token, &mut simd, &xs, &xs);
            for i in 0..n {
                assert!(
                    scalar[i].is_nan() && simd[i].is_nan()
                        || scalar[i].to_bits() == simd[i].to_bits()
                );
            }
            // Dense values on both sides of every LUT boundary, including clamp endpoints.
            let xs: Vec<f32> = (0..n).map(|i| (i as f32 - 20.0) * 0.125).collect();
            let lut: Vec<f32> = (0..32)
                .map(|i| ((i * 17) % 13) as f32 * 0.1 - 0.5)
                .collect();
            width8::gather_lerp_kernel::<_, Midp, true>(
                ScalarToken,
                &xs,
                &lut,
                -2.30103,
                4.9198306,
                30.999999,
                -0.1,
                core::f32::consts::LN_10,
                &mut scalar,
            );
            $width::gather_lerp_kernel::<_, Midp, true>(
                token,
                &xs,
                &lut,
                -2.30103,
                4.9198306,
                30.999999,
                -0.1,
                core::f32::consts::LN_10,
                &mut simd,
            );
            for i in 0..n {
                assert!((scalar[i] - simd[i]).abs() <= 5e-5 * scalar[i].abs());
            }
        }
    }};
}
#[test]
fn scalar_contract() {
    check_tier!(ScalarToken, width8);
}
#[test]
#[cfg(target_arch = "x86_64")]
fn v3_contract() {
    if let Some(t) = archmage::X64V3Token::summon() {
        check_tier!(t, width8);
    }
}
#[test]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn v4_contract() {
    if let Some(t) = archmage::X64V4Token::summon() {
        check_tier!(t, width16);
    }
}
#[test]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn v4x_contract() {
    if let Some(t) = archmage::X64V4xToken::summon() {
        check_tier!(t, width16);
    }
}
#[test]
#[cfg(target_arch = "aarch64")]
fn neon_contract() {
    if let Some(t) = archmage::NeonToken::summon() {
        check_tier!(t, width8);
    }
}
#[test]
#[cfg(target_arch = "wasm32")]
fn wasm_contract() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        check_tier!(t, width8);
    }
}

#[test]
fn gather_matches_bracket_reference() {
    let lut: Vec<f32> = (0..32).map(|i| ((i * 7) % 19) as f32).collect();
    let xs: Vec<f32> = (-17..1033).map(|i| i as f32 / 32.0).collect();
    let mut out = vec![0.0; xs.len()];
    gather_lerp_into(&xs, &lut, 0.0, 1.0, 30.999999, &mut out);
    for (&x, &got) in xs.iter().zip(&out) {
        let idx = x.clamp(0.0, 30.999999);
        let lo = idx.floor() as usize;
        let want = lut[lo] + (idx - lo as f32) * (lut[lo + 1] - lut[lo]);
        assert_eq!(got.to_bits(), want.to_bits());
    }
}

#[test]
fn par_order_and_rows() {
    assert_eq!(
        par::collect_indexed(257, |i| i),
        (0..257).collect::<Vec<_>>()
    );
    let mut rows = vec![0.0; 513 * 257];
    par::map_rows(&mut rows, 257, |y, band| {
        for (j, row) in band.chunks_mut(257).enumerate() {
            row.fill((y + j) as f32);
        }
    });
    for (y, row) in rows.chunks(257).enumerate() {
        assert!(row.iter().all(|&x| x == y as f32));
    }
    assert_eq!(par::join2(|| 17, || 31), (17, 31));
}

#[test]
#[cfg(feature = "avx512")]
fn scalar_native_width_contract() {
    check_tier!(ScalarToken, width16);
}
