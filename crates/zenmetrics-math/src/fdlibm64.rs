//! Lane transcription of `libm::pow` (FreeBSD/SunPro `e_pow.c`, libm 0.2.16).
//!
//! Every lane is bit-identical to `libm::pow(x, y)` on every input. The
//! common domain — positive finite normal `x` and moderate `y` — runs
//! lane-parallel with e_pow's constants and operation order: separate
//! `*`/`+` (never fused), correctly rounded vector division, and the
//! `& 0xffffffff00000000` truncations as bitwise ANDs on the lane pattern.
//! Integer decode, interval select, the `[z+0.5]` extraction and the `2**n`
//! reconstruction run on `i64x4` lanes with e_pow's two per-lane variable
//! shifts as masked binary select trees (magetypes offers only constant
//! and uniform shifts). The early-outs that return `1.0` for any partner
//! argument (`y = ±0`, `x = 1`) are answered in vector; every other
//! early-out or edge branch is flagged per lane and recomputed by scalar
//! `libm::pow`:
//!
//! - `x <= 0` (including `±0` and every negative `x`, which would need the
//!   `y`-integer decode), `x` subnormal, infinite or NaN;
//! - `y = ±1`, `y = 2`, `y = +0.5` (the `sqrt` path), `|y| > 2^31`
//!   (including infinite and NaN `y`);
//! - `z = y·log2(x)` at or beyond the overflow (`≥ 1024`) or underflow
//!   (`≤ -1075`) thresholds;
//! - results whose exponent field would leave the normal range
//!   (e_pow's `scalbn` branch).

use magetypes::simd::backends::I64x4Bitcast;
use magetypes::simd::generic::{f64x4, i64x4};

const BP: [f64; 2] = [1.0, 1.5];
const DP_H: [f64; 2] = [0.0, 5.84962487220764160156e-01]; /* 0x3fe2b80340000000 */
const DP_L: [f64; 2] = [0.0, 1.35003920212974897128e-08]; /* 0x3e4cfdeb43cfd006 */
const L1: f64 = 5.99999999999994648725e-01; /* 0x3fe3333333333303 */
const L2: f64 = 4.28571428578550184252e-01; /* 0x3fdb6db6db6fabff */
const L3: f64 = 3.33333329818377432918e-01; /* 0x3fd55555518f264d */
const L4: f64 = 2.72728123808534006489e-01; /* 0x3fd17460a91d4101 */
const L5: f64 = 2.30660745775561754067e-01; /* 0x3fcd864a93c9db65 */
const L6: f64 = 2.06975017800338417784e-01; /* 0x3fca7e284a454eef */
const P1: f64 = 1.66666666666666019037e-01; /* 0x3fc555555555553e */
const P2: f64 = -2.77777777770155933842e-03; /* 0xbf66c16c16bebd93 */
const P3: f64 = 6.61375632143793436117e-05; /* 0x3f11566aaf25de2c */
const P4: f64 = -1.65339022054652515390e-06; /* 0xbebbbd41c5d26bf1 */
const P5: f64 = 4.13813679705723846039e-08; /* 0x3e66376972bea4d0 */
const LG2: f64 = 6.93147180559945286227e-01; /* 0x3fe62e42fefa39ef */
const LG2_H: f64 = 6.93147182464599609375e-01; /* 0x3fe62e43_00000000 */
const LG2_L: f64 = -1.90465429995776804525e-09; /* 0xbe205c610ca86c39 */
const CP: f64 = 9.61796693925975554329e-01; /* 0x3feec709dc3a03fd =2/(3ln2) */
const CP_H: f64 = 9.61796700954437255859e-01; /* 0x3feec709e0000000 */
const CP_L: f64 = -7.02846165095275826516e-09; /* 0xbe3e2fe0145b01f5 */

/// Recompute the flagged lanes with scalar `libm::pow`.
#[cold]
#[inline(never)]
fn scalar_lanes<const N: usize>(x: &[f64; N], y: &[f64; N], need: &[i64; N], out: &mut [f64; N]) {
    for i in 0..N {
        if need[i] != 0 {
            out[i] = libm::pow(x[i], y[i]);
        }
    }
}

/// `i64x4::<T>::splat_t(token, v)`.
macro_rules! ci {
    ($token:ident, $v:expr) => {
        i64x4::<T>::splat_t($token, $v)
    };
}

/// `f64x4::<T>::splat_t(token, v)`.
macro_rules! cf {
    ($token:ident, $v:expr) => {
        f64x4::<T>::splat_t($token, $v)
    };
}

/// `f64::from_bits(v.to_bits() & 0xffffffff00000000)` — e_pow's
/// `with_set_low_word(v, 0)`.
macro_rules! trunc {
    ($token:ident, $v:expr) => {
        $v & cf!($token, f64::from_bits(0xffff_ffff_0000_0000))
    };
}

/// Per-lane `v >> s` for non-negative `v` and `s` in [0, 63]: one constant
/// logical shift per bit of `s`, selected by a lane mask. Lanes whose `s`
/// is out of range are never consumed.
macro_rules! shr_var {
    ($token:ident, $v:expr, $s:expr) => {{
        let s = $s;
        let zero = ci!($token, 0);
        let v = $v;
        let v = i64x4::<T>::blend(
            (s & ci!($token, 32)).simd_ne(zero),
            v.shr_logical_const::<32>(),
            v,
        );
        let v = i64x4::<T>::blend(
            (s & ci!($token, 16)).simd_ne(zero),
            v.shr_logical_const::<16>(),
            v,
        );
        let v = i64x4::<T>::blend(
            (s & ci!($token, 8)).simd_ne(zero),
            v.shr_logical_const::<8>(),
            v,
        );
        let v = i64x4::<T>::blend(
            (s & ci!($token, 4)).simd_ne(zero),
            v.shr_logical_const::<4>(),
            v,
        );
        let v = i64x4::<T>::blend(
            (s & ci!($token, 2)).simd_ne(zero),
            v.shr_logical_const::<2>(),
            v,
        );
        i64x4::<T>::blend(
            (s & ci!($token, 1)).simd_ne(zero),
            v.shr_logical_const::<1>(),
            v,
        )
    }};
}

/// Lane-vectorized `libm::pow` — bit-identical per lane on all inputs.
///
/// Positive finite normal `x` with `|y| <= 2^31` and an in-range result
/// runs in vector; any other lane is recomputed by scalar `libm::pow`.
/// Coverage-gated variant of [`pow64_libm_4x`]; `G` is a
/// [`crate::fdlibm::coverage`] bitmask. `SUBN` is a no-op here: subnormal
/// `x` is routed to scalar by EDGE rather than normalized in-vector.
#[inline(always)]
pub fn pow64_libm_4x_gated<const G: u8, T: I64x4Bitcast>(
    token: T,
    x: f64x4<T>,
    y: f64x4<T>,
) -> f64x4<T> {
    let edge = G & crate::fdlibm::coverage::EDGE != 0;
    let subn = G & crate::fdlibm::coverage::SUBN != 0;
    let range = G & crate::fdlibm::coverage::RANGE != 0;
    let tail = G & crate::fdlibm::coverage::TAIL != 0;
    let zero = ci!(token, 0);
    let xbits = i64x4::<T>::from_array_t(token, x.to_array().map(|v| v.to_bits() as i64));
    let ybits = i64x4::<T>::from_array_t(token, y.to_array().map(|v| v.to_bits() as i64));
    let xabs = xbits & ci!(token, 0x7fff_ffff_ffff_ffff);
    let yabs = ybits & ci!(token, 0x7fff_ffff_ffff_ffff);

    // x**±0 = 1 and 1**y = 1, even if the other argument is NaN.
    // `one_out`/`need` are dead unless their gates are compiled in.
    let _one_out = if edge {
        yabs.simd_eq(zero) | xbits.simd_eq(ci!(token, 0x3ff0_0000_0000_0000))
    } else {
        zero
    };
    // Every other early-out: x <= 0 (all negatives incl. -0), x subnormal,
    // x inf/NaN, y inf/NaN, |y| > 2^31, y = ±1, y = 2, y = +0.5.
    let mut _need = if edge {
        xbits.simd_lt(zero)
            | xbits.simd_le(ci!(token, 0x000f_ffff_ffff_ffff))
            | xabs.simd_ge(ci!(token, 0x7ff0_0000_0000_0000))
            | yabs.simd_ge(ci!(token, 0x7ff0_0000_0000_0000))
            | yabs.simd_gt(ci!(token, 0x41e0_0000_0000_0000))
            | yabs.simd_eq(ci!(token, 0x3ff0_0000_0000_0000))
            | ybits.simd_eq(ci!(token, 0x4000_0000_0000_0000))
            | ybits.simd_eq(ci!(token, 0x3fe0_0000_0000_0000))
    } else {
        zero
    };
    // SUBN owns the subnormal-x routing term when EDGE is off: under
    // `POSITIVE_FINITE` those lanes still reach scalar `libm::pow` (the
    // f32 port normalizes in-vector; this port routes instead).
    if subn && !edge {
        _need |= xbits.simd_le(ci!(token, 0x000f_ffff_ffff_ffff));
    }

    // Live lanes: x normal and positive. hx = the high 32 bits.
    let hx = xbits.shr_arithmetic_const::<32>();
    // n = (hx >> 20) - 0x3ff
    let mut n = hx.shr_arithmetic_const::<20>() - ci!(token, 0x3ff);
    let j = hx & ci!(token, 0x000f_ffff);
    // determine interval
    let ixn = j | ci!(token, 0x3ff0_0000); /* normalize ix */
    let below_a = j.simd_le(ci!(token, 0x3988e)); /* j <= 0x3988E → k = 0 */
    let below_b = j.simd_lt(ci!(token, 0xbb67a)); /* j < 0xBB67A → k = 1 */
    let k1 = below_b & below_a.not(); /* k = 1 */
    let wrap = below_b.not(); /* k = 0; n += 1; ix -= 0x00100000 */
    n -= wrap;
    let ixn = ixn - (wrap & ci!(token, 0x0010_0000));
    // ax = with_high_word(ax, ixn): keep x's low word.
    let ax = (ixn.shl_const::<32>() | (xbits & ci!(token, 0xffff_ffff))).bitcast_to_f64();
    let k1f = k1.bitcast_to_f64();
    let bp = f64x4::<T>::blend(k1f, cf!(token, BP[1]), cf!(token, BP[0]));
    let dp_h = f64x4::<T>::blend(k1f, cf!(token, DP_H[1]), cf!(token, DP_H[0]));
    let dp_l = f64x4::<T>::blend(k1f, cf!(token, DP_L[1]), cf!(token, DP_L[0]));
    // t_h = ax+bp[k] High: ((ix>>1 | 0x20000000) + 0x80000 + (k<<18)) << 32
    let th_bits = ((ixn.shr_logical_const::<1>() | ci!(token, 0x2000_0000))
        + ci!(token, 0x0008_0000)
        + (k1 & ci!(token, 1 << 18)))
    .shl_const::<32>();
    let t_h = th_bits.bitcast_to_f64();
    let nf = f64x4::<T>::from_array_t(token, n.to_array().map(|v| v as f64));

    /* compute s = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
    let mut u = ax - bp;
    let mut v = cf!(token, 1.0) / (ax + bp);
    let s = u * v;
    let s_h = trunc!(token, s);
    // TAIL: sub-ulp corrections; cleared terms keep the same tree shape.
    let s_l = if tail {
        let t_l0 = ax - (t_h - bp);
        v * ((u - s_h * t_h) - s_h * t_l0)
    } else {
        cf!(token, 0.0)
    };
    /* compute log(ax) */
    let mut s2 = s * s;
    let mut r = s2
        * s2
        * (cf!(token, L1)
            + s2 * (cf!(token, L2)
                + s2 * (cf!(token, L3)
                    + s2 * (cf!(token, L4) + s2 * (cf!(token, L5) + s2 * cf!(token, L6))))));
    if tail {
        r += s_l * (s_h + s);
    }
    s2 = s_h * s_h;
    let t_h = trunc!(token, cf!(token, 3.0) + s2 + r);
    let t_l = if tail {
        r - ((t_h - cf!(token, 3.0)) - s2)
    } else {
        cf!(token, 0.0)
    };
    /* u+v = s*(1+...) */
    u = s_h * t_h;
    v = s_l * t_h + t_l * s;
    /* 2/(3log2)*(s+...) */
    let mut p_h = trunc!(token, u + v);
    let mut p_l = if tail { v - (p_h - u) } else { cf!(token, 0.0) };
    let z_h = cf!(token, CP_H) * p_h; /* cp_h+cp_l = 2/(3*log2) */
    let z_l = cf!(token, CP_L) * p_h + p_l * cf!(token, CP) + dp_l;
    /* log2(ax) = (s+..)*2/(3*log2) = n + dp_h + z_h + z_l */
    let t1 = trunc!(token, ((z_h + z_l) + dp_h) + nf);
    let t2 = if tail {
        z_l - (((t1 - nf) - dp_h) - z_h)
    } else {
        cf!(token, 0.0)
    };

    /* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
    let y1 = trunc!(token, y);
    p_l = (y - y1) * t1 + y * t2;
    p_h = y1 * t1;
    let z = p_l + p_h;

    // z's bits feed integer decode; extract once.
    let zbits = i64x4::<T>::from_array_t(token, z.to_array().map(|v| v.to_bits() as i64));
    let hz = zbits.shr_arithmetic_const::<32>();
    // Overflow (z >= 1024) and underflow (|z| >= 1075.39) branches.
    if range {
        _need |= hz.simd_ge(ci!(token, 0x4090_0000))
            | (hz & ci!(token, 0x7fff_ffff)).simd_ge(ci!(token, 0x4090_cc00));
    }

    /* if |z| > 0.5, set n = [z+0.5] via the bit trick */
    let iiz = hz & ci!(token, 0x7fff_ffff);
    let kz = iiz.shr_logical_const::<20>() - ci!(token, 0x3ff);
    let nn = hz + shr_var!(token, ci!(token, 0x0010_0000), kz + ci!(token, 1));
    let kz2 = (nn & ci!(token, 0x7fff_ffff)).shr_logical_const::<20>() - ci!(token, 0x3ff);
    let tmask = shr_var!(token, ci!(token, 0x000f_ffff), kz2);
    let tv = ((nn & tmask.not()).shl_const::<32>()).bitcast_to_f64();
    let npos = shr_var!(
        token,
        (nn & ci!(token, 0x000f_ffff)) | ci!(token, 0x0010_0000),
        ci!(token, 20) - kz2
    );
    let nsgn = i64x4::<T>::blend(hz.simd_lt(zero), -npos, npos);
    let half = iiz.simd_gt(ci!(token, 0x3fe0_0000));
    let nn_f = half.bitcast_to_f64();
    let nv = half & nsgn;
    let p_h = p_h - f64x4::<T>::blend(nn_f, tv, cf!(token, 0.0));

    /* compute 2**(p_h+p_l) */
    let t3 = trunc!(token, p_l + p_h);
    let u3 = t3 * cf!(token, LG2_H);
    let v3 = (p_l - (t3 - p_h)) * cf!(token, LG2) + t3 * cf!(token, LG2_L);
    let z3 = u3 + v3;
    let w = if tail {
        v3 - (z3 - u3)
    } else {
        cf!(token, 0.0)
    };
    let tt = z3 * z3;
    let t1p = z3
        - tt * (cf!(token, P1)
            + tt * (cf!(token, P2)
                + tt * (cf!(token, P3) + tt * (cf!(token, P4) + tt * cf!(token, P5)))));
    let r2 = (z3 * t1p) / (t1p - cf!(token, 2.0)) - (w + z3 * w);
    let zf = cf!(token, 1.0) - (r2 - z3);

    /* j += n << 20 reconstruction; scalbn output → scalar. */
    let zfbits = i64x4::<T>::from_array_t(token, zf.to_array().map(|v| v.to_bits() as i64));
    let j2 = zfbits.shr_arithmetic_const::<32>() + nv.shl_const::<20>();
    if range {
        _need |= j2.shr_arithmetic_const::<20>().simd_le(zero);
    }
    let res = (j2.shl_const::<32>() | (zfbits & ci!(token, 0xffff_ffff))).bitcast_to_f64();
    let res = if edge {
        f64x4::<T>::blend(_one_out.bitcast_to_f64(), cf!(token, 1.0), res)
    } else {
        res
    };
    if edge || range || subn {
        let need = if edge { _need & _one_out.not() } else { _need };
        if need.any_true() {
            let mut out = res.to_array();
            scalar_lanes::<4>(&x.to_array(), &y.to_array(), &need.to_array(), &mut out);
            return f64x4::<T>::from_array_t(token, out);
        }
    }
    res
}

/// Lane-vectorized `libm::pow` on f64x4 — bit-identical per lane on all
/// inputs (full `coverage::EXACT` gate).
#[inline(always)]
pub fn pow64_libm_4x<T: I64x4Bitcast>(token: T, x: f64x4<T>, y: f64x4<T>) -> f64x4<T> {
    pow64_libm_4x_gated::<{ crate::fdlibm::coverage::EXACT }, _>(token, x, y)
}

// ------------------------------------------------------------------
// `e_log.c` / `e_log2.c` — identical argument reduction and Lg1..Lg7
// polynomial; only the final reconstruction differs. `LOG2` selects
// e_log2's `hi/lo` split + `IVLN2HI/LO` scale; otherwise e_log's flat
// `s*(hfsq+r) + dk*LN2_LO - hfsq + f + dk*LN2_HI` tail.
// ------------------------------------------------------------------

const LG1: f64 = 6.666666666666735130e-01; /* 3FE55555 55555593 */
const LG2C: f64 = 3.999999999940941908e-01; /* 3FD99999 9997FA04 */
const LG3: f64 = 2.857142874366239149e-01; /* 3FD24924 94229359 */
const LG4: f64 = 2.222219843214978396e-01; /* 3FCC71C5 1D8E78AF */
const LG5: f64 = 1.818357216161805012e-01; /* 3FC74664 96CB03DE */
const LG6C: f64 = 1.531383769920937332e-01; /* 3FC39A09 D078C69F */
const LG7: f64 = 1.479819860511658591e-01; /* 3FC2F112 DF3E5244 */
const IVLN2HI: f64 = 1.44269504072144627571e+00; /* 0x3ff71547, 0x65200000 */
const IVLN2LO: f64 = 1.67517131648865118353e-10; /* 0x3de705fc, 0x2eefa200 */
const LN2_HI: f64 = 6.93147180369123816490e-01; /* 3fe62e42 fee00000 */
const LN2_LO: f64 = 1.90821492927058770002e-10; /* 3dea39ef 35793c76 */

/// Recompute the flagged lanes with scalar `libm::log`/`log2`.
#[cold]
#[inline(never)]
fn scalar_log_lanes<const N: usize, const LOG2: bool>(
    x: &[f64; N],
    need: &[i64; N],
    out: &mut [f64; N],
) {
    for i in 0..N {
        if need[i] != 0 {
            out[i] = if LOG2 {
                libm::log2(x[i])
            } else {
                libm::log(x[i])
            };
        }
    }
}

/// Lane-vectorized `libm::log`/`libm::log2` — bit-identical per lane.
///
/// Positive finite `x` (including subnormal inputs, via the masked
/// `x *= 2^54` normalize) runs in vector. Lanes that are `x <= 0`
/// (any sign bit or `±0`) take the scalar early-outs verbatim —
/// `-inf`, `NaN`, `-NaN` — since e_log's `(x-x)/0.0` NaN construction
/// is left to the scalar oracle. `x = ±inf`/`+NaN` (exponent `0x7ff`)
/// return `x` and `x = 1` returns `+0` as in-vector blends.
#[inline(always)]
fn log_libm_4x_impl<T: I64x4Bitcast, const LOG2: bool>(token: T, x: f64x4<T>) -> f64x4<T> {
    let zero = ci!(token, 0);
    let xbits = i64x4::<T>::from_array_t(token, x.to_array().map(|v| v.to_bits() as i64));
    // need = sign bit set or x == ±0 (ui << 1 == 0).
    let need = xbits.simd_lt(zero) | (xbits.shl_const::<1>()).simd_eq(zero);
    // inf / NaN (|x| >= 0x7ff00000 after abs): return x.
    let xabs = xbits & ci!(token, 0x7fff_ffff_ffff_ffff);
    let ret_x = xabs.simd_ge(ci!(token, 0x7ff0_0000_0000_0000));
    // x == 1.0: return +0.
    let ret_zero = xbits.simd_eq(ci!(token, 0x3ff0_0000_0000_0000));

    // Subnormal positive x (exponent field 0): k -= 54; x *= 2^54.
    // `hx < 0x00100000` catches exp==0 for the high word — lanes that are
    // negative or ±0 are in `need` anyway, so including them here is safe.
    let hx = xbits.shr_logical_const::<32>();
    let subm = hx.simd_lt(ci!(token, 0x0010_0000));
    let xn = f64x4::<T>::blend(
        subm.bitcast_to_f64(),
        x * cf!(token, f64::from_bits(0x4350_0000_0000_0000)),
        x,
    );
    let k = subm & ci!(token, -54);
    let xnbits = i64x4::<T>::from_array_t(token, xn.to_array().map(|v| v.to_bits() as i64));
    let mut hxr = xnbits.shr_logical_const::<32>();

    /* reduce x into [sqrt(2)/2, sqrt(2)] */
    hxr += ci!(token, 0x3ff0_0000 - 0x3fe6_a09e);
    let k = k + (hxr.shr_logical_const::<20>() - ci!(token, 0x3ff));
    hxr = (hxr & ci!(token, 0x000f_ffff)) + ci!(token, 0x3fe6_a09e);
    let xru = (hxr.shl_const::<32>() | (xnbits & ci!(token, 0xffff_ffff))).bitcast_to_f64();

    let f = xru - cf!(token, 1.0);
    let hfsq = cf!(token, 0.5) * f * f;
    let s = f / (cf!(token, 2.0) + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (cf!(token, LG2C) + w * (cf!(token, LG4) + w * cf!(token, LG6C)));
    let t2 =
        z * (cf!(token, LG1) + w * (cf!(token, LG3) + w * (cf!(token, LG5) + w * cf!(token, LG7))));
    let r = t2 + t1;
    let dk = f64x4::<T>::from_array_t(token, k.to_array().map(|v| v as f64));

    let res = if LOG2 {
        /* hi+lo = f - hfsq + s*(hfsq+R) ~ log(1+f) */
        let hi = trunc!(token, f - hfsq);
        let lo = f - hi - hfsq + s * (hfsq + r);
        let val_hi = hi * cf!(token, IVLN2HI);
        let mut val_lo = (lo + hi) * cf!(token, IVLN2LO) + lo * cf!(token, IVLN2HI);
        /* spadd(val_hi, val_lo, dk) */
        let wv = dk + val_hi;
        val_lo += (dk - wv) + val_hi;
        val_lo + wv
    } else {
        s * (hfsq + r) + dk * cf!(token, LN2_LO) - hfsq + f + dk * cf!(token, LN2_HI)
    };

    // inf/NaN lanes return x; x==1 lanes return +0.
    let res = f64x4::<T>::blend(ret_x.bitcast_to_f64(), x, res);
    let res = f64x4::<T>::blend(ret_zero.bitcast_to_f64(), cf!(token, 0.0), res);
    if need.any_true() {
        let mut out = res.to_array();
        scalar_log_lanes::<4, LOG2>(&x.to_array(), &need.to_array(), &mut out);
        f64x4::<T>::from_array_t(token, out)
    } else {
        res
    }
}

/// Lane `libm::log` — bit-identical per lane on all inputs.
#[inline(always)]
pub fn log_libm_4x<T: I64x4Bitcast>(token: T, x: f64x4<T>) -> f64x4<T> {
    log_libm_4x_impl::<T, false>(token, x)
}

/// Lane `libm::log2` — bit-identical per lane on all inputs.
#[inline(always)]
pub fn log2_libm_4x<T: I64x4Bitcast>(token: T, x: f64x4<T>) -> f64x4<T> {
    log_libm_4x_impl::<T, true>(token, x)
}

/// `out[i] = libm::log(xs[i])`, bit-identical to the scalar call.
#[inline(always)]
pub fn vlog_libm_kernel_4<T: I64x4Bitcast>(token: T, xs: &[f64], out: &mut [f64]) {
    assert_eq!(xs.len(), out.len());
    let (in_chunks, in_tail) = f64x4::<T>::partition_slice_t(token, xs);
    let (out_chunks, out_tail) = f64x4::<T>::partition_slice_mut_t(token, out);
    for (x, o) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        log_libm_4x(token, f64x4::<T>::load_t(token, x)).store(o);
    }
    for (x, o) in in_tail.iter().zip(out_tail.iter_mut()) {
        *o = libm::log(*x);
    }
}

/// `out[i] = libm::log2(xs[i])`, bit-identical to the scalar call.
#[inline(always)]
pub fn vlog2_libm_kernel_4<T: I64x4Bitcast>(token: T, xs: &[f64], out: &mut [f64]) {
    assert_eq!(xs.len(), out.len());
    let (in_chunks, in_tail) = f64x4::<T>::partition_slice_t(token, xs);
    let (out_chunks, out_tail) = f64x4::<T>::partition_slice_mut_t(token, out);
    for (x, o) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        log2_libm_4x(token, f64x4::<T>::load_t(token, x)).store(o);
    }
    for (x, o) in in_tail.iter().zip(out_tail.iter_mut()) {
        *o = libm::log2(*x);
    }
}

/// `out[i] = libm::pow(xs[i], p)`, bit-identical to the scalar call.
#[inline(always)]
pub fn vpow64_libm_kernel_4<const G: u8, T: I64x4Bitcast>(
    token: T,
    xs: &[f64],
    out: &mut [f64],
    p: f64,
) {
    assert_eq!(xs.len(), out.len());
    let pv = f64x4::<T>::splat_t(token, p);
    let (in_chunks, in_tail) = f64x4::<T>::partition_slice_t(token, xs);
    let (out_chunks, out_tail) = f64x4::<T>::partition_slice_mut_t(token, out);
    for (x, o) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        pow64_libm_4x_gated::<G, _>(token, f64x4::<T>::load_t(token, x), pv).store(o);
    }
    for (x, o) in in_tail.iter().zip(out_tail.iter_mut()) {
        *o = libm::pow(*x, p);
    }
}

/// `out[i] = libm::pow(xs[i], ys[i])`, bit-identical to the scalar call.
#[inline(always)]
pub fn vpow64_libm_pairs_kernel_4<const G: u8, T: I64x4Bitcast>(
    token: T,
    xs: &[f64],
    ys: &[f64],
    out: &mut [f64],
) {
    assert_eq!(xs.len(), out.len());
    assert_eq!(ys.len(), out.len());
    let (x_chunks, x_tail) = f64x4::<T>::partition_slice_t(token, xs);
    let (y_chunks, y_tail) = f64x4::<T>::partition_slice_t(token, ys);
    let (out_chunks, out_tail) = f64x4::<T>::partition_slice_mut_t(token, out);
    for ((x, y), o) in x_chunks.iter().zip(y_chunks).zip(out_chunks.iter_mut()) {
        pow64_libm_4x_gated::<G, _>(
            token,
            f64x4::<T>::load_t(token, x),
            f64x4::<T>::load_t(token, y),
        )
        .store(o);
    }
    for ((x, y), o) in x_tail.iter().zip(y_tail).zip(out_tail.iter_mut()) {
        *o = libm::pow(*x, *y);
    }
}
