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
            // Three-channel fused apply: one shared bracket, per-channel LUTs,
            // and `(v * (band_mul*sens)) * gain` against both inputs.
            let luts_v: [Vec<f32>; 3] = core::array::from_fn(|k| {
                (0..32)
                    .map(|i| ((i * (7 + 2 * k)) % 13) as f32 * 0.1 - 0.6)
                    .collect()
            });
            let luts = [&luts_v[0][..], &luts_v[1][..], &luts_v[2][..]];
            let mk = |k: usize, salt: f32| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i * 31 + k * 7) % 97) as f32 * salt - 0.5)
                    .collect()
            };
            let ins_a = [mk(0, 0.03), mk(1, 0.07), mk(2, 0.05)];
            let ins_b = [mk(3, 0.06), mk(4, 0.04), mk(5, 0.08)];
            let ins_a_r = [&ins_a[0][..], &ins_a[1][..], &ins_a[2][..]];
            let ins_b_r = [&ins_b[0][..], &ins_b[1][..], &ins_b[2][..]];
            let mut sa: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; n]);
            let mut sb: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; n]);
            let mut ta: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; n]);
            let mut tb: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; n]);
            let [sa0, sa1, sa2] = &mut sa;
            let [sb0, sb1, sb2] = &mut sb;
            width8::gather_lerp_exp3_apply_kernel::<_, Midp>(
                ScalarToken,
                &xs,
                luts,
                -2.30103,
                4.9198306,
                30.999999,
                -0.1,
                core::f32::consts::LN_10,
                1.3,
                [1.0, 1.1, 0.9],
                ins_a_r,
                ins_b_r,
                [&mut sa0[..], &mut sa1[..], &mut sa2[..]],
                [&mut sb0[..], &mut sb1[..], &mut sb2[..]],
            );
            let [ta0, ta1, ta2] = &mut ta;
            let [tb0, tb1, tb2] = &mut tb;
            $width::gather_lerp_exp3_apply_kernel::<_, Midp>(
                token,
                &xs,
                luts,
                -2.30103,
                4.9198306,
                30.999999,
                -0.1,
                core::f32::consts::LN_10,
                1.3,
                [1.0, 1.1, 0.9],
                ins_a_r,
                ins_b_r,
                [&mut ta0[..], &mut ta1[..], &mut ta2[..]],
                [&mut tb0[..], &mut tb1[..], &mut tb2[..]],
            );
            for c in 0..3 {
                for i in 0..n {
                    assert!(
                        (sa[c][i] - ta[c][i]).abs() <= 5e-5 * sa[c][i].abs().max(1e-6),
                        "exp3_apply out_a c={c} n={n} i={i} scalar={} simd={}",
                        sa[c][i],
                        ta[c][i]
                    );
                    assert!(
                        (sb[c][i] - tb[c][i]).abs() <= 5e-5 * sb[c][i].abs().max(1e-6),
                        "exp3_apply out_b c={c} n={n} i={i} scalar={} simd={}",
                        sb[c][i],
                        tb[c][i]
                    );
                }
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
    if let Some(t) = archmage::X64V4xToken::summon() {
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

/// The v4x hardware-gather specialization must produce bit-identical
/// output to the generic width16 tile kernel: gathers return the same
/// lanes and every downstream op is the same vector arithmetic.
#[test]
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
fn v4_exp3_apply_matches_generic_width16() {
    let Some(token) = archmage::X64V4xToken::summon() else {
        return;
    };
    for n in [0, 1, 7, 8, 15, 16, 17, 33, 256, 257, 1029] {
        let xs: Vec<f32> = (0..n).map(|i| (i as f32 - 40.0) * 0.061).collect();
        let luts_v: [Vec<f32>; 3] = core::array::from_fn(|k| {
            (0..32)
                .map(|i| ((i * (5 + 3 * k)) % 11) as f32 * 0.1 - 0.55)
                .collect()
        });
        let luts = [&luts_v[0][..], &luts_v[1][..], &luts_v[2][..]];
        let mk = |k: usize, salt: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 37 + k * 13) % 89) as f32 * salt - 0.4)
                .collect()
        };
        let ins_a = [mk(0, 0.05), mk(1, 0.03), mk(2, 0.07)];
        let ins_b = [mk(3, 0.04), mk(4, 0.08), mk(5, 0.06)];
        let ins_a_r = [&ins_a[0][..], &ins_a[1][..], &ins_a[2][..]];
        let ins_b_r = [&ins_b[0][..], &ins_b[1][..], &ins_b[2][..]];
        let mut ga: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; n]);
        let mut gb: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; n]);
        let mut ha: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; n]);
        let mut hb: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; n]);
        let [ga0, ga1, ga2] = &mut ga;
        let [gb0, gb1, gb2] = &mut gb;
        width16::gather_lerp_exp3_apply_kernel::<_, Midp>(
            token,
            &xs,
            luts,
            -2.30103,
            4.9198306,
            30.999999,
            -0.1,
            core::f32::consts::LN_10,
            1.7,
            [1.0, 0.95, 1.05],
            ins_a_r,
            ins_b_r,
            [&mut ga0[..], &mut ga1[..], &mut ga2[..]],
            [&mut gb0[..], &mut gb1[..], &mut gb2[..]],
        );
        let [ha0, ha1, ha2] = &mut ha;
        let [hb0, hb1, hb2] = &mut hb;
        width16::gather_lerp_exp3_apply_kernel_v4::<Midp>(
            token,
            &xs,
            luts,
            -2.30103,
            4.9198306,
            30.999999,
            -0.1,
            core::f32::consts::LN_10,
            1.7,
            [1.0, 0.95, 1.05],
            ins_a_r,
            ins_b_r,
            [&mut ha0[..], &mut ha1[..], &mut ha2[..]],
            [&mut hb0[..], &mut hb1[..], &mut hb2[..]],
        );
        for c in 0..3 {
            for i in 0..n {
                assert_eq!(
                    ga[c][i].to_bits(),
                    ha[c][i].to_bits(),
                    "exp3_apply v4-vs-generic out_a c={c} n={n} i={i}"
                );
                assert_eq!(
                    gb[c][i].to_bits(),
                    hb[c][i].to_bits(),
                    "exp3_apply v4-vs-generic out_b c={c} n={n} i={i}"
                );
            }
        }
    }
}
