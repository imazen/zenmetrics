//! Lane transcription of `libm::powf` (FreeBSD/SunPro `e_powf.c`, libm 0.2.16).
//!
//! Every lane is bit-identical to `libm::powf(x, y)` on every input. The
//! common domain — positive finite `x` (normal or subnormal) and moderate
//! `y` — runs lane-parallel with e_powf's constants, operation order and bit
//! tricks: separate `*`/`+` (never fused), correctly rounded vector division,
//! and the `& 0xfffff000` truncations as integer masks on the bit pattern.
//! The two early-outs that return `1.0` for any partner argument
//! (`y = ±0`, `x = 1`) are answered in vector, because metric callers hit
//! `x = 1` on identical pixels. Every other early-out or edge branch is
//! flagged per lane and recomputed by scalar `libm::powf`:
//!
//! - `x <= 0` (including `±0` and every negative `x`), `x` infinite or NaN;
//! - `y = ±1`, `y = 2`, `y = 0.5` (the `sqrtf` path), `|y| > 2^27`
//!   (including infinite and NaN `y`);
//! - `z = y·log2(x)` at or beyond the overflow (`≥ 128`) or underflow
//!   (`≤ -150`) thresholds;
//! - results whose exponent field would leave the normal range
//!   (e_powf's `scalbnf` branch).
//!
//! e_powf's per-lane variable shifts (`>> (k + 1)`, `>> k`, `>> (23 - k)`)
//! run as a masked binary select tree over the shift count's bits; magetypes
//! offers only constant and uniform shifts. `round()` is deliberately not
//! used for `n = [z + 0.5]`: the bit trick rounds half away from zero and
//! `round()` rounds half to even.

use magetypes::simd::backends::F32x8Convert;
#[cfg(feature = "avx512")]
use magetypes::simd::backends::F32x16Convert;
use magetypes::simd::generic::{f32x8, i32x8};
#[cfg(feature = "avx512")]
use magetypes::simd::generic::{f32x16, i32x16};

const BP: [f32; 2] = [1.0, 1.5];
const DP_H: [f32; 2] = [0.0, 5.84960938e-01]; /* 0x3f15c000 */
const DP_L: [f32; 2] = [0.0, 1.56322085e-06]; /* 0x35d1cfdc */
const TWO24: f32 = 16777216.0; /* 0x4b800000 */
const L1: f32 = 6.0000002384e-01; /* 0x3f19999a */
const L2: f32 = 4.2857143283e-01; /* 0x3edb6db7 */
const L3: f32 = 3.3333334327e-01; /* 0x3eaaaaab */
const L4: f32 = 2.7272811532e-01; /* 0x3e8ba305 */
const L5: f32 = 2.3066075146e-01; /* 0x3e6c3255 */
const L6: f32 = 2.0697501302e-01; /* 0x3e53f142 */
const P1: f32 = 1.6666667163e-01; /* 0x3e2aaaab */
const P2: f32 = -2.7777778450e-03; /* 0xbb360b61 */
const P3: f32 = 6.6137559770e-05; /* 0x388ab355 */
const P4: f32 = -1.6533901999e-06; /* 0xb5ddea0e */
const P5: f32 = 4.1381369442e-08; /* 0x3331bb4c */
const LG2: f32 = 6.9314718246e-01; /* 0x3f317218 */
const LG2_H: f32 = 6.93145752e-01; /* 0x3f317200 */
const LG2_L: f32 = 1.42860654e-06; /* 0x35bfbe8c */
const CP: f32 = 9.6179670095e-01; /* 0x3f76384f =2/(3ln2) */
const CP_H: f32 = 9.6191406250e-01; /* 0x3f764000 =12b cp */
const CP_L: f32 = -1.1736857402e-04; /* 0xb8f623c6 =tail of cp_h */

/// `0xfffff000` / `0xffff8000` as the `i32` lane masks e_powf applies to
/// `u32` bit patterns.
const HI12: i32 = 0xffff_f000_u32 as i32;
const HI15: i32 = 0xffff_8000_u32 as i32;

/// Recompute the flagged lanes with scalar `libm::powf`.
#[cold]
#[inline(never)]
fn scalar_lanes<const N: usize>(x: &[f32; N], y: &[f32; N], need: &[i32; N], out: &mut [f32; N]) {
    for i in 0..N {
        if need[i] != 0 {
            out[i] = libm::powf(x[i], y[i]);
        }
    }
}

/// `I::splat_t(token, v)`.
macro_rules! ci {
    ($token:ident, $I:ident, $v:expr) => {
        $I::<T>::splat_t($token, $v)
    };
}

/// `F::splat_t(token, v)`.
macro_rules! cf {
    ($token:ident, $F:ident, $v:expr) => {
        $F::<T>::splat_t($token, $v)
    };
}

/// `f32::from_bits(v.to_bits() & mask)`.
macro_rules! trunc {
    ($token:ident, $F:ident, $I:ident, $v:expr, $mask:expr) => {
        $F::<T>::from_i32_bitcast_t($token, $v.bitcast_to_i32() & ci!($token, $I, $mask))
    };
}

/// Per-lane `v >> s` for non-negative `v` and `s` in [0, 31]: one constant
/// shift per bit of `s`, selected by a lane mask. Lanes whose `s` is out of
/// range are never consumed. A macro, not a closure, so the select tree
/// inlines into the caller's target-feature context.
macro_rules! shr_var {
    ($token:ident, $I:ident, $v:expr, $s:expr) => {{
        let s = $s;
        let zero = ci!($token, $I, 0);
        let v = $v;
        let v = $I::<T>::blend(
            (s & ci!($token, $I, 16)).simd_ne(zero),
            v.shr_logical_const::<16>(),
            v,
        );
        let v = $I::<T>::blend(
            (s & ci!($token, $I, 8)).simd_ne(zero),
            v.shr_logical_const::<8>(),
            v,
        );
        let v = $I::<T>::blend(
            (s & ci!($token, $I, 4)).simd_ne(zero),
            v.shr_logical_const::<4>(),
            v,
        );
        let v = $I::<T>::blend(
            (s & ci!($token, $I, 2)).simd_ne(zero),
            v.shr_logical_const::<2>(),
            v,
        );
        $I::<T>::blend(
            (s & ci!($token, $I, 1)).simd_ne(zero),
            v.shr_logical_const::<1>(),
            v,
        )
    }};
}

macro_rules! powf_libm_lanes {
    (
        $(#[$doc:meta])*
        $name:ident, $kernel:ident, $pairs:ident, $F:ident, $I:ident, $Convert:ident, $lanes:literal
    ) => {
        $(#[$doc])*
        #[inline(always)]
        pub fn $name<T: $Convert>(token: T, x: $F<T>, y: $F<T>) -> $F<T> {
            type F<T> = $F<T>;
            type I<T> = $I<T>;
            let zero_i = I::<T>::splat_t(token, 0);
            let hx = x.bitcast_to_i32();
            let hy = y.bitcast_to_i32();
            let ix0 = hx & ci!(token, I, 0x7fff_ffff);
            let iy = hy & ci!(token, I, 0x7fff_ffff);

            // x**0 = 1 and 1**y = 1, even if the other argument is NaN.
            let one_out = iy.simd_eq(zero_i) | hx.simd_eq(ci!(token, I, 0x3f80_0000));
            // Every other early-out: x <= 0, x inf/NaN, |y| > 2^27 (incl.
            // inf/NaN), y = ±1, y = 2, y = 0.5.
            let mut need = hx.simd_le(zero_i)
                | ix0.simd_ge(ci!(token, I, 0x7f80_0000))
                | iy.simd_gt(ci!(token, I, 0x4d00_0000))
                | iy.simd_eq(ci!(token, I, 0x3f80_0000))
                | hy.simd_eq(ci!(token, I, 0x4000_0000))
                | hy.simd_eq(ci!(token, I, 0x3f00_0000));

            // ax = fabsf(x)
            let ax = F::<T>::from_i32_bitcast_t(token, ix0);
            let mut ix = ix0;
            // take care subnormal number (ax *= TWO24 only feeds ix)
            let sub = ix.simd_lt(ci!(token, I, 0x0080_0000));
            let ax_s = ax * cf!(token, F, TWO24);
            let mut n = sub & ci!(token, I, -24);
            ix = I::<T>::blend(sub, ax_s.bitcast_to_i32(), ix);
            n += ix.shr_arithmetic_const::<23>() - ci!(token, I, 0x7f);
            let j = ix & ci!(token, I, 0x007f_ffff);
            // determine interval
            ix = j | ci!(token, I, 0x3f80_0000); /* normalize ix */
            let below_sqrt3_2 = j.simd_le(ci!(token, I, 0x1c_c471));
            let below_sqrt3 = j.simd_lt(ci!(token, I, 0x5d_b3d7));
            let k1 = below_sqrt3 & below_sqrt3_2.not(); /* k = 1 */
            let wrap = below_sqrt3.not(); /* k = 0; n += 1; ix -= 0x00800000 */
            n -= wrap;
            ix -= wrap & ci!(token, I, 0x0080_0000);
            let ax = F::<T>::from_i32_bitcast_t(token, ix);
            let k1f = k1.bitcast_to_f32();
            let bp = F::<T>::blend(k1f, cf!(token, F, BP[1]), cf!(token, F, BP[0]));
            let dp_h = F::<T>::blend(k1f, cf!(token, F, DP_H[1]), cf!(token, F, DP_H[0]));
            let dp_l = F::<T>::blend(k1f, cf!(token, F, DP_L[1]), cf!(token, F, DP_L[0]));

            /* compute s = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
            let mut u = ax - bp;
            let mut v = cf!(token, F, 1.0) / (ax + bp);
            let s = u * v;
            let s_h = trunc!(token, F, I, s, HI12);
            /* t_h=ax+bp[k] High */
            let is = (ix.shr_logical_const::<1>() & ci!(token, I, HI12)) | ci!(token, I, 0x2000_0000);
            let mut t_h = F::<T>::from_i32_bitcast_t(
                token,
                is + ci!(token, I, 0x0040_0000) + (k1 & ci!(token, I, 1 << 21)),
            );
            let mut t_l = ax - (t_h - bp);
            let s_l = v * ((u - s_h * t_h) - s_h * t_l);
            /* compute log(ax) */
            let mut s2 = s * s;
            let mut r = s2
                * s2
                * (cf!(token, F, L1)
                    + s2 * (cf!(token, F, L2) + s2 * (cf!(token, F, L3) + s2 * (cf!(token, F, L4) + s2 * (cf!(token, F, L5) + s2 * cf!(token, F, L6))))));
            r += s_l * (s_h + s);
            s2 = s_h * s_h;
            t_h = cf!(token, F, 3.0) + s2 + r;
            t_h = trunc!(token, F, I, t_h, HI12);
            t_l = r - ((t_h - cf!(token, F, 3.0)) - s2);
            /* u+v = s*(1+...) */
            u = s_h * t_h;
            v = s_l * t_h + t_l * s;
            /* 2/(3log2)*(s+...) */
            let mut p_h = trunc!(token, F, I, u + v, HI12);
            let mut p_l = v - (p_h - u);
            let z_h = cf!(token, F, CP_H) * p_h; /* cp_h+cp_l = 2/(3*log2) */
            let z_l = cf!(token, F, CP_L) * p_h + p_l * cf!(token, F, CP) + dp_l;
            /* log2(ax) = (s+..)*2/(3*log2) = n + dp_h + z_h + z_l */
            let t = F::<T>::from_i32_t(token, n);
            let t1 = trunc!(token, F, I, ((z_h + z_l) + dp_h) + t, HI12);
            let t2 = z_l - (((t1 - t) - dp_h) - z_h);

            /* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
            let y1 = trunc!(token, F, I, y, HI12);
            p_l = (y - y1) * t1 + y * t2;
            p_h = y1 * t1;
            let z = p_l + p_h;
            let j = z.bitcast_to_i32();
            // Overflow (z >= 128) and underflow (z <= -150) branches.
            need |= j.simd_ge(ci!(token, I, 0x4300_0000)) | (j & ci!(token, I, 0x7fff_ffff)).simd_ge(ci!(token, I, 0x4316_0000));

            /* compute 2**(p_h+p_l) */
            let i = j & ci!(token, I, 0x7fff_ffff);
            let k = i.shr_arithmetic_const::<23>() - ci!(token, I, 0x7f);
            /* if |z| > 0.5, set n = [z+0.5] */
            let half_plus = i.simd_gt(ci!(token, I, 0x3f00_0000));
            let nn = j + shr_var!(token, I, ci!(token, I, 0x0080_0000), k + ci!(token, I, 1));
            let k = (nn & ci!(token, I, 0x7fff_ffff)).shr_arithmetic_const::<23>() - ci!(token, I, 0x7f); /* new k for n */
            let t = F::<T>::from_i32_bitcast_t(token, nn & shr_var!(token, I, ci!(token, I, 0x007f_ffff), k).not());
            let nv = shr_var!(token, I, (nn & ci!(token, I, 0x007f_ffff)) | ci!(token, I, 0x0080_0000), ci!(token, I, 23) - k);
            let nv = I::<T>::blend(j.simd_lt(zero_i), zero_i - nv, nv);
            let n = half_plus & nv;
            p_h = F::<T>::blend(half_plus.bitcast_to_f32(), p_h - t, p_h);
            let t = trunc!(token, F, I, p_l + p_h, HI15);
            let u = t * cf!(token, F, LG2_H);
            let v = (p_l - (t - p_h)) * cf!(token, F, LG2) + t * cf!(token, F, LG2_L);
            let z = u + v;
            let w = v - (z - u);
            let t = z * z;
            let t1 = z - t * (cf!(token, F, P1) + t * (cf!(token, F, P2) + t * (cf!(token, F, P3) + t * (cf!(token, F, P4) + t * cf!(token, F, P5)))));
            let r = (z * t1) / (t1 - cf!(token, F, 2.0)) - (w + z * w);
            let z = cf!(token, F, 1.0) - (r - z);
            let j = z.bitcast_to_i32() + n.shl_const::<23>();
            // Subnormal output: e_powf's scalbnf branch.
            need |= j.shr_arithmetic_const::<23>().simd_le(zero_i);
            // `sn * z` with `sn = 1.0` (x > 0) is the identity on these
            // finite lanes.
            let res = F::<T>::from_i32_bitcast_t(token, j);
            let res = F::<T>::blend(one_out.bitcast_to_f32(), cf!(token, F, 1.0), res);
            let need = need & one_out.not();
            if need.any_true() {
                let mut out = res.to_array();
                scalar_lanes::<$lanes>(&x.to_array(), &y.to_array(), &need.to_array(), &mut out);
                F::<T>::from_array_t(token, out)
            } else {
                res
            }
        }

        /// `out[i] = libm::powf(xs[i], p)`, bit-identical to the scalar call.
        #[inline(always)]
        pub fn $kernel<T: $Convert>(token: T, xs: &[f32], out: &mut [f32], p: f32) {
            assert_eq!(xs.len(), out.len());
            let pv = $F::<T>::splat_t(token, p);
            let (in_chunks, in_tail) = $F::<T>::partition_slice_t(token, xs);
            let (out_chunks, out_tail) = $F::<T>::partition_slice_mut_t(token, out);
            for (x, o) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                $name(token, $F::<T>::load_t(token, x), pv).store(o);
            }
            for (x, o) in in_tail.iter().zip(out_tail.iter_mut()) {
                *o = libm::powf(*x, p);
            }
        }

        /// `out[i] = libm::powf(xs[i], ys[i])`, bit-identical to the scalar call.
        #[inline(always)]
        pub fn $pairs<T: $Convert>(token: T, xs: &[f32], ys: &[f32], out: &mut [f32]) {
            assert_eq!(xs.len(), out.len());
            assert_eq!(ys.len(), out.len());
            let (x_chunks, x_tail) = $F::<T>::partition_slice_t(token, xs);
            let (y_chunks, y_tail) = $F::<T>::partition_slice_t(token, ys);
            let (out_chunks, out_tail) = $F::<T>::partition_slice_mut_t(token, out);
            for ((x, y), o) in x_chunks.iter().zip(y_chunks).zip(out_chunks.iter_mut()) {
                $name(token, $F::<T>::load_t(token, x), $F::<T>::load_t(token, y)).store(o);
            }
            for ((x, y), o) in x_tail.iter().zip(y_tail).zip(out_tail.iter_mut()) {
                *o = libm::powf(*x, *y);
            }
        }
    };
}

powf_libm_lanes!(
    /// Lane-vectorized `libm::powf` — bit-identical per lane on all inputs.
    ///
    /// Positive finite `x` with `|y| <= 2^27` and an in-range result runs in
    /// vector; any other lane is recomputed by scalar `libm::powf`.
    powf_libm_8x, vpow_libm_kernel_8, vpowf_libm_pairs_kernel_8, f32x8, i32x8, F32x8Convert, 8
);

#[cfg(feature = "avx512")]
powf_libm_lanes!(
    /// Sixteen-lane [`powf_libm_8x`]: bit-identical per lane to `libm::powf`.
    powf_libm_16x, vpow_libm_kernel_16, vpowf_libm_pairs_kernel_16, f32x16, i32x16, F32x16Convert, 16
);
