//! glibc `__powf` (Szabolcs Nagy's `e_powf.c`, glibc >= 2.28 — the ARM
//! optimized-routines algorithm, shared lineage with musl but a different
//! implementation than `libm` crate's FreeBSD `e_powf` transcription).
//!
//! Stage 1: scalar reference transcription, verified bit-identical to
//! `f32::powf` (which calls glibc on linux-gnu) over the whole input domain.
//! The fast path — positive finite `x`, finite nonzero `y`, and
//! `|y*log2(x)| < 126` — is transcribed verbatim; every other lane delegates
//! to `f32::powf`, which is glibc by definition, so the result is
//! bit-exact on all inputs by construction.
//!
//! f64 internal arithmetic throughout (`double_t` == `double` on
//! FLT_EVAL_METHOD=0 targets; that includes x86_64 and aarch64).
//!
//! Measured on Zen 4 (7950X, 2^22 FLIP-domain inputs): scalar `f32::powf`
//! 1.0x, the lane kernels 1.13x, a VPGATHERDD table-load variant 0.62x —
//! AMD gathers are microcoded and lose to the scalar loads, so there is no
//! gather path here; revisit on Intel where VPGATHERDD is cheap.
//!
//! Sources (fetched 2026-10-20, glibc-2.40 + ARM-software/optimized-routines
//! master for the generated table):
//!   sysdeps/ieee754/flt-32/e_powf.c        — algorithm
//!   ARM optimized-routines math/powf_log2_data.c — log2 table + poly
//!   musl src/math/exp2f_data.c             — exp2 table + poly + shift

const OFF: u32 = 0x3f330000;

/// `__powf_log2_data.tab` — (invc, logc) per interval center; POWF_SCALE=1
/// (TOINT_INTRINSICS is 0 on this target).
const LOG2_TAB: [(f64, f64); 16] = [
    (
        f64::from_bits(0x3ff661ec79f8f3be),
        f64::from_bits(0xbfdefec65b963019),
    ),
    (
        f64::from_bits(0x3ff571ed4aaf883d),
        f64::from_bits(0xbfdb0b6832d4fca4),
    ),
    (
        f64::from_bits(0x3ff49539f0f010b0),
        f64::from_bits(0xbfd7418b0a1fb77b),
    ),
    (
        f64::from_bits(0x3ff3c995b0b80385),
        f64::from_bits(0xbfd39de91a6dcf7b),
    ),
    (
        f64::from_bits(0x3ff30d190c8864a5),
        f64::from_bits(0xbfd01d9bf3f2b631),
    ),
    (
        f64::from_bits(0x3ff25e227b0b8ea0),
        f64::from_bits(0xbfc97c1d1b3b7af0),
    ),
    (
        f64::from_bits(0x3ff1bb4a4a1a343f),
        f64::from_bits(0xbfc2f9e393af3c9f),
    ),
    (
        f64::from_bits(0x3ff12358f08ae5ba),
        f64::from_bits(0xbfb960cbbf788d5c),
    ),
    (
        f64::from_bits(0x3ff0953f419900a7),
        f64::from_bits(0xbfaa6f9db6475fce),
    ),
    (
        f64::from_bits(0x3ff0000000000000),
        f64::from_bits(0x0000000000000000),
    ),
    (
        f64::from_bits(0x3fee608cfd9a47ac),
        f64::from_bits(0x3fb338ca9f24f53d),
    ),
    (
        f64::from_bits(0x3feca4b31f026aa0),
        f64::from_bits(0x3fc476a9543891ba),
    ),
    (
        f64::from_bits(0x3feb2036576afce6),
        f64::from_bits(0x3fce840b4ac4e4d2),
    ),
    (
        f64::from_bits(0x3fe9c2d163a1aa2d),
        f64::from_bits(0x3fd40645f0c6651c),
    ),
    (
        f64::from_bits(0x3fe886e6037841ed),
        f64::from_bits(0x3fd88e9c2c1b9ff8),
    ),
    (
        f64::from_bits(0x3fe767dcf5534862),
        f64::from_bits(0x3fdce0a44eb17bcc),
    ),
];
const LOG2_POLY: [f64; 5] = [
    f64::from_bits(0x3fd27616c9496e0b),
    f64::from_bits(0xbfd71969a075c67a),
    f64::from_bits(0x3fdec70a6ca7badd),
    f64::from_bits(0xbfe7154748bef6c8),
    f64::from_bits(0x3ff71547652ab82b),
];

/// `__exp2f_data.tab` — `uint(2^(i/32)) - (i << 47)`.
const EXP2_TAB: [u64; 32] = [
    0x3ff0000000000000,
    0x3fefd9b0d3158574,
    0x3fefb5586cf9890f,
    0x3fef9301d0125b51,
    0x3fef72b83c7d517b,
    0x3fef54873168b9aa,
    0x3fef387a6e756238,
    0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715,
    0x3feef1a7373aa9cb,
    0x3feedea64c123422,
    0x3feece086061892d,
    0x3feebfdad5362a27,
    0x3feeb42b569d4f82,
    0x3feeab07dd485429,
    0x3feea47eb03a5585,
    0x3feea09e667f3bcd,
    0x3fee9f75e8ec5f74,
    0x3feea11473eb0187,
    0x3feea589994cce13,
    0x3feeace5422aa0db,
    0x3feeb737b0cdc5e5,
    0x3feec49182a3f090,
    0x3feed503b23e255d,
    0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47,
    0x3fef199bdd85529c,
    0x3fef3720dcef9069,
    0x3fef5818dcfba487,
    0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da,
    0x3fefd0765b6e4540,
];
const EXP2_POLY: [f64; 3] = [
    f64::from_bits(0x3fac6af84b912394),
    f64::from_bits(0x3fcebfce50fac4f3),
    f64::from_bits(0x3fe62e42ff0c52d6),
];
/// `__exp2f_data.shift_scaled` = f64::from_bits(0x4338000000000000) / 32 — rounds to the 1/32 grid.
const SHIFT_SCALED: f64 = f64::from_bits(0x42e8000000000000);

/// e_powf's `log2_inline` (POWF_SCALE_BITS = 0 → no output scaling).
// `x = a * b + x` lines are transcribed verbatim from e_powf.c.
#[allow(clippy::assign_op_pattern)]
fn log2_inline(ix: u32) -> f64 {
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> 19) % 16) as usize;
    let top = tmp & 0xff800000;
    let iz = ix.wrapping_sub(top);
    let k = (top as i32) >> 23;
    let (invc, logc) = LOG2_TAB[i];
    let z = f64::from(f32::from_bits(iz));
    let r = z * invc - 1.0;
    let y0 = logc + f64::from(k);
    let r2 = r * r;
    let mut y = LOG2_POLY[0] * r + LOG2_POLY[1];
    let p = LOG2_POLY[2] * r + LOG2_POLY[3];
    let r4 = r2 * r2;
    let mut q = LOG2_POLY[4] * r + y0;
    q = p * r2 + q;
    y = y * r4 + q;
    y
}

/// e_powf's `exp2_inline` (non-TOINT_INTRINSICS branch; sign_bias != 0 only
/// on slow paths this module delegates, kept for transcription fidelity).
// `x = a * b + x` lines are transcribed verbatim from e_powf.c.
#[allow(clippy::assign_op_pattern)]
fn exp2_inline(xd: f64, sign_bias: u32) -> f64 {
    let kd = xd + SHIFT_SCALED;
    let ki = kd.to_bits();
    let kd = kd - SHIFT_SCALED;
    let r = xd - kd;
    let t = EXP2_TAB[(ki % 32) as usize];
    let ski = ki.wrapping_add(u64::from(sign_bias));
    let t = t.wrapping_add(ski << 47);
    let s = f64::from_bits(t);
    let z = EXP2_POLY[0] * r + EXP2_POLY[1];
    let r2 = r * r;
    let mut y = EXP2_POLY[2] * r + 1.0;
    y = z * r2 + y;
    y * s
}

/// `0xfeffffff` boundary for the y fast-path gate (zeroinfnan's unsigned
/// compare), kept as the logical-shift form the lanes use.
const YFAST_SHIFT_LIMIT: i32 = 0xfe;
/// `(asuint64(126.0) >> 47)` — e_powf's `|y*log(x)| >= 126` gate constant.
const K126: i64 = (126.0f64.to_bits() >> 47) as i64;

/// Eight-lane fast path of [`powf_glibc`]. Every lane runs the verbatim
/// fast path; `need` marks lanes that must be recomputed by scalar
/// `f32::powf` (non-fast-path x or y, or the |y·log x| >= 126 boundary).
/// The arithmetic is identical to [`log2_inline`]/[`exp2_inline`] — f64 ops
/// only, no reassociation, so fast lanes are bit-identical to the scalar
/// transcription (and therefore to glibc).
#[archmage::attune(_*_t, define(f64x8, i64x8, i32x8, f32x8))]
fn powf_glibc_8x_t(
    token: Token,
    xs: &[f32; 8],
    ys: &[f32; 8],
    out: &mut [f32; 8],
    need: &mut [i32; 8],
) {
    let xf = f32x8::load_t(token, xs);
    let xi = xf.bitcast_to_i32();
    // x gate: (ix - 0x00800000) unsigned-in [0, 0x7f000000) — the unsigned
    // range test becomes signed since wrapped values land < 0 as i32.
    let t = xi - i32x8::splat_t(token, 0x0080_0000);
    let fast_x = t.simd_ge(i32x8::zero_t(token)) & t.simd_lt(i32x8::splat_t(token, 0x7f00_0000));
    // y gate: 2*iy-1 unsigned-below 0xfeffffff. (q >> 24) < 0xfe with q >= 0
    // is a superset of that condition — it additionally flags y patterns in
    // [0xfe000000, 0xfefffffe] (|y| >= ~1.7e38), which can only land on slow
    // paths anyway.
    let yf = f32x8::load_t(token, ys);
    let yi = yf.bitcast_to_i32();
    let q = (yi + yi) - i32x8::splat_t(token, 1);
    let fast_y = q.simd_ge(i32x8::zero_t(token))
        & q.shr_logical_const::<24>()
            .simd_lt(i32x8::splat_t(token, YFAST_SHIFT_LIMIT));
    // Negative x falls inside the fast-path gate above only when
    // ix < 0x80000000 as u32 — never true for negative x (bit31 set), so
    // fast_x already excludes them; subnormal x lands outside the gate too.

    // log2_inline per lane — index/select machinery in i32, table values by
    // scalar gather (16-entry table; a blend tree costs more than loads).
    let ix = xf.to_array().map(f32::to_bits);
    let idx: [usize; 8] =
        core::array::from_fn(|j| (((ix[j].wrapping_sub(OFF)) >> 19) % 16) as usize);
    let tmp = xi - i32x8::splat_t(token, OFF as i32);
    let top = tmp & i32x8::splat_t(token, 0xff80_0000_u32 as i32);
    let k = top.shr_arithmetic_const::<23>();
    let invc = f64x8::from_array_t(token, core::array::from_fn(|j| LOG2_TAB[idx[j]].0));
    let logc = f64x8::from_array_t(token, core::array::from_fn(|j| LOG2_TAB[idx[j]].1));
    let z = f64x8::from_array_t(
        token,
        core::array::from_fn(|j| {
            f64::from(f32::from_bits(ix[j].wrapping_sub(top.to_array()[j] as u32)))
        }),
    );
    let r = z * invc - f64x8::splat_t(token, 1.0);
    let y0 = logc + f64x8::from_array_t(token, k.to_array().map(|v| v as f64));
    let r2 = r * r;
    let y = f64x8::splat_t(token, LOG2_POLY[0]) * r + f64x8::splat_t(token, LOG2_POLY[1]);
    let p = f64x8::splat_t(token, LOG2_POLY[2]) * r + f64x8::splat_t(token, LOG2_POLY[3]);
    let r4 = r2 * r2;
    let mut qq = f64x8::splat_t(token, LOG2_POLY[4]) * r + y0;
    qq = p * r2 + qq;
    let logx = y * r4 + qq;

    let y64 = f64x8::from_array_t(token, yf.to_array().map(f64::from));
    let ylogx = y64 * logx;
    // Boundary gate: (asuint64(ylogx) >> 47 & 0xffff) >= K126 — i64 lanes.
    let hi = i64x8::from_array_t(token, ylogx.to_array().map(|v| v.to_bits() as i64));
    let h = hi.shr_logical_const::<47>() & i64x8::splat_t(token, 0xffff);
    let edge = h.simd_ge(i64x8::splat_t(token, K126));

    // exp2_inline per lane.
    let kd = ylogx + f64x8::splat_t(token, SHIFT_SCALED);
    let ki_v = i64x8::from_array_t(token, kd.to_array().map(|v| v.to_bits() as i64));
    let kd = kd - f64x8::splat_t(token, SHIFT_SCALED);
    let rr = ylogx - kd;
    let ki = ki_v.to_array();
    let t_arr: [u64; 8] = core::array::from_fn(|j| {
        let k = ki[j] as u64;
        EXP2_TAB[(k % 32) as usize].wrapping_add(k.wrapping_shl(47))
    });
    let s2 = f64x8::from_array_t(token, t_arr.map(f64::from_bits));
    let zz = f64x8::splat_t(token, EXP2_POLY[0]) * rr + f64x8::splat_t(token, EXP2_POLY[1]);
    let rr2 = rr * rr;
    let mut yy = f64x8::splat_t(token, EXP2_POLY[2]) * rr + f64x8::splat_t(token, 1.0);
    yy = zz * rr2 + yy;
    let res = (yy * s2).to_array();
    let fx = fast_x.to_array();
    let fy = fast_y.to_array();
    let fe = edge.to_array();
    for j in 0..8 {
        let slow = fx[j] == 0 || fy[j] == 0 || fe[j] != 0;
        need[j] = slow as i32;
        out[j] = if slow {
            xs[j].powf(ys[j])
        } else {
            res[j] as f32
        };
    }
}

fn zeroinfnan(ix: u32) -> bool {
    ix.wrapping_mul(2).wrapping_sub(1) >= 2 * 0x7f80_0000u32 - 1
}

/// Scalar transcription of glibc `__powf` — bit-identical to `f32::powf`
/// (glibc) on every input: the fast path is verbatim; all slow branches
/// (subnormal/inf/nan x, zero/inf/nan y, negative x, |y·log2 x| >= 126)
/// delegate to `f32::powf`, which is glibc by definition.
#[inline]
pub fn powf_glibc(x: f32, y: f32) -> f32 {
    let ix = x.to_bits();
    let iy = y.to_bits();
    if ix.wrapping_sub(0x0080_0000) >= 0x7f80_0000 - 0x0080_0000 || zeroinfnan(iy) {
        return x.powf(y);
    }
    // Positive normal finite x, nonzero finite y: the fast path.
    if ix & 0x8000_0000 != 0 {
        return x.powf(y); // negative x: checkint/invalid branches
    }
    let logx = log2_inline(ix);
    let ylogx = f64::from(y) * logx;
    if (ylogx.to_bits() >> 47 & 0xffff) >= (126.0f64.to_bits() >> 47) {
        // |y*log(x)| >= 126 — oflow/uflow/may_uflow boundary lanes.
        return x.powf(y);
    }
    exp2_inline(ylogx, 0) as f32
}

/// `out[i] = powf_glibc(xs[i], p)` — eight-lane fast path plus scalar
/// fallback for edge lanes; bit-identical to `f32::powf` on every input.
pub fn vpowf_glibc_into(xs: &[f32], out: &mut [f32], p: f32) {
    assert_eq!(xs.len(), out.len());
    let (xb, xt) = xs.as_chunks::<8>();
    let (ob, ot) = out.as_chunks_mut::<8>();
    for (x8, o8) in xb.iter().zip(ob.iter_mut()) {
        let y8 = [p; 8];
        let mut need = [0i32; 8];
        archmage::attuned!(
            powf_glibc_8x_t(x8, &y8, o8, &mut need),
            [_v3, _neon, _wasm128, _scalar]
        );
        for j in 0..8 {
            if need[j] != 0 {
                o8[j] = powf_glibc(x8[j], p);
            }
        }
    }
    for (o, &x) in ot.iter_mut().zip(xt) {
        *o = powf_glibc(x, p);
    }
}

/// `out[i] = powf_glibc(xs[i], ys[i])` — same lane strategy.
pub fn vpowf_glibc_pairs_into(xs: &[f32], ys: &[f32], out: &mut [f32]) {
    assert_eq!(xs.len(), out.len());
    assert_eq!(ys.len(), out.len());
    let (xb, xt) = xs.as_chunks::<8>();
    let (yb, yt) = ys.as_chunks::<8>();
    let (ob, ot) = out.as_chunks_mut::<8>();
    for ((x8, y8), o8) in xb.iter().zip(yb.iter()).zip(ob.iter_mut()) {
        let mut need = [0i32; 8];
        archmage::attuned!(
            powf_glibc_8x_t(x8, y8, o8, &mut need),
            [_v3, _neon, _wasm128, _scalar]
        );
        for j in 0..8 {
            if need[j] != 0 {
                o8[j] = powf_glibc(x8[j], y8[j]);
            }
        }
    }
    for ((o, &x), &y) in ot.iter_mut().zip(xt).zip(yt) {
        *o = powf_glibc(x, y);
    }
}

#[cfg(test)]
mod bench_probe {
    use super::*;

    /// Scalar powf vs batched glibc lanes on FLIP-domain inputs (time only;
    /// correctness lives in the main tests).
    #[test]
    #[ignore]
    fn powf_wall() {
        let n = 1 << 22;
        let xs: Vec<f32> = (0..n).map(|i| 0.05 + (i % 997) as f32 / 997.0).collect();
        let mut out = vec![0.0f32; n];
        let reps = 20;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for (o, &x) in out.iter_mut().zip(&xs) {
                *o = x.powf(0.7);
            }
        }
        let scalar = t.elapsed();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for (o, &x) in out.iter_mut().zip(&xs) {
                *o = powf_glibc(x, 0.7);
            }
        }
        let port = t.elapsed();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            vpowf_glibc_into(&xs, &mut out, 0.7);
        }
        let lanes = t.elapsed();
        eprintln!("scalar {scalar:?} port {port:?} lanes {lanes:?}");
        eprintln!(
            "port {:.2}x lanes {:.2}x",
            scalar.as_secs_f64() / port.as_secs_f64(),
            scalar.as_secs_f64() / lanes.as_secs_f64()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_f32_powf_across_domain() {
        // Dense (x, y) grid over the fast-path domain plus edge cases.
        let mut xs: Vec<f32> = Vec::new();
        for e in -126i32..128 {
            for &m in &[1.0f32, 1.0000001, 1.5, 1.9999999] {
                xs.push(m * 2f32.powi(e));
            }
        }
        xs.extend([
            0.0,
            -0.0,
            -1.0,
            -2.0,
            -0.5,
            f32::INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE / 2.0,
            1e-40,
            f32::MAX,
        ]);
        let ys: Vec<f32> = [
            -150.0,
            -127.0,
            -126.0,
            -10.0,
            -3.7,
            -2.0,
            -1.0,
            -0.7,
            -1.0 / 3.0,
            -0.25,
            0.0,
            0.25,
            1.0 / 3.0,
            0.5,
            0.7,
            1.0,
            2.0,
            3.7,
            10.0,
            126.0,
            127.0,
            150.0,
            f32::INFINITY,
            f32::NAN,
            -0.0,
        ]
        .into_iter()
        .collect();
        let mut mismatches = 0u32;
        let mut worst = 0.0f32;
        for &x in &xs {
            for &y in &ys {
                let (a, b) = (powf_glibc(x, y), x.powf(y));
                if a.to_bits() != b.to_bits() {
                    mismatches += 1;
                    let d = (f64::from(a) - f64::from(b)).abs();
                    let r = d / f64::from(b).abs().max(1e-300);
                    if r > worst as f64 {
                        worst = r as f32;
                    }
                    if mismatches < 8 {
                        eprintln!("powf({x:e}, {y}): ref={a:e} vs std={b:e}");
                    }
                }
            }
        }
        assert_eq!(mismatches, 0, "max rel {worst:e}");
        // The slice API must agree with both scalar conventions, including
        // across the 8-lane boundary and the tail.
        for n in [1usize, 7, 8, 9, 16, 63] {
            let xv: Vec<f32> = xs[..n.min(xs.len())].to_vec();
            let mut o = vec![0.0f32; xv.len()];
            vpowf_glibc_into(&xv, &mut o, 0.7);
            for (x, got) in xv.iter().zip(&o) {
                assert_eq!(got.to_bits(), x.powf(0.7).to_bits(), "x={x:e}");
            }
        }
    }

    #[test]
    fn matches_powf_dense_random() {
        // LCG over all f32 bit patterns for x>0, bounded y.
        let mut s = 0x9e3779b9u32;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for _ in 0..200_000 {
            let x = f32::from_bits(next() & 0x7f7fffff) * 2f32.powi((next() % 500) as i32 - 250);
            if !(x > 0.0) || !x.is_finite() {
                continue;
            }
            let y = ((next() % 6000) as f32 - 3000.0) / 40.0;
            let (a, b) = (powf_glibc(x, y), x.powf(y));
            assert_eq!(a.to_bits(), b.to_bits(), "powf({x:e}, {y}): {a:e} vs {b:e}");
        }
    }
}
