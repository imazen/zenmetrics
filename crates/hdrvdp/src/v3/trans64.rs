//! Vectorised `f64` transcendentals for the **relaxed** floating-point
//! path (`FpMode::Relaxed`): `exp`, `ln`, `|x|^q`, `sign·|x|^q` and the
//! fused per-pixel maps that consume them — eight lanes per op.
//!
//! Accuracy: ~2 ulp on `exp`/`ln`, ~6 ulp on the composed `pow` — orders
//! of magnitude inside the documented relaxed-mode score tolerance. The
//! strict path never calls these kernels; scalar tails use libm.
//!
//! `f64x8` has no integer-conversion op, so exponent extraction and the
//! `2^k` reconstruction round-trip through `[f64; 8]`/`u64` arrays —
//! a handful of scalar bit ops per lane, still an order of magnitude
//! cheaper than a ~50-cycle libm call.

use magetypes::simd::backends::F64x8Backend;
use magetypes::simd::generic::f64x8;

/// `2^k` for integer-valued `k` per lane via the exponent-field bit
/// trick. `k < −1021` yields 0, `k > 1023` yields ∞, non-finite `k`
/// yields 1 (the caller blends edge lanes anyway).
#[inline(always)]
fn two_pow_k<T: F64x8Backend>(token: T, k: f64x8<T>) -> f64x8<T> {
    let ka = k.to_array();
    let mut s = [0.0f64; 8];
    for (s, &ki) in s.iter_mut().zip(&ka) {
        *s = if ki < -1021.0 {
            0.0
        } else if ki > 1023.0 {
            f64::INFINITY
        } else if ki.is_finite() {
            f64::from_bits(((ki as i64 + 1023) as u64) << 52)
        } else {
            1.0
        };
    }
    f64x8::from_array_t(token, s)
}

/// `exp(x)` per lane, ~2 ulp. `k = rint(x·log2e)`, `r = x − k·ln2` split
/// hi/lo; `exp(r)` is a degree-13 Horner on `r ∈ [−ln2/2, ln2/2]`.
/// `x = ±inf` leaves `r` NaN, so the result is blended on `k`: underflow
/// → 0, overflow → ∞, matching libm.
#[inline(always)]
fn exp8<T: F64x8Backend>(token: T, x: f64x8<T>) -> f64x8<T> {
    let k = (x * f64x8::splat_t(token, std::f64::consts::LOG2_E)).round();
    // ln2 = LN2_HI + LN2_LO to 26 decimal digits.
    let r = k.mul_add(f64x8::splat_t(token, -0.693_147_180_369_123_816), x);
    let r = k.mul_add(f64x8::splat_t(token, -0.000_000_000_190_821_492_927), r);
    // Taylor for exp(r), r ∈ [−0.3466, 0.3466]: the r¹⁵/15! tail is
    // ≈ 1e-19 — well under f64 epsilon.
    let mut p = f64x8::splat_t(token, 1.147_074_559_772_972_5e-11); // 1/14!
    p = p.mul_add(r, f64x8::splat_t(token, 1.605_904_383_682_161_3e-10)); // 1/13!
    p = p.mul_add(r, f64x8::splat_t(token, 2.087_675_698_786_81e-9)); // 1/12!
    p = p.mul_add(r, f64x8::splat_t(token, 2.505_210_838_544_172e-8)); // 1/11!
    p = p.mul_add(r, f64x8::splat_t(token, 2.755_731_922_398_589e-7)); // 1/10!
    p = p.mul_add(r, f64x8::splat_t(token, 2.755_731_922_398_589_3e-6)); // 1/9!
    p = p.mul_add(r, f64x8::splat_t(token, 2.480_158_730_158_73e-5)); // 1/8!
    p = p.mul_add(r, f64x8::splat_t(token, 1.984_126_984_126_984e-4)); // 1/7!
    p = p.mul_add(r, f64x8::splat_t(token, 1.388_888_888_888_889e-3)); // 1/6!
    p = p.mul_add(r, f64x8::splat_t(token, 8.333_333_333_333_333e-3)); // 1/5!
    p = p.mul_add(r, f64x8::splat_t(token, 4.166_666_666_666_666_4e-2)); // 1/4!
    p = p.mul_add(r, f64x8::splat_t(token, 1.666_666_666_666_666_7e-1)); // 1/3!
    p = p.mul_add(r, f64x8::splat_t(token, 5.0e-1)); // 1/2!
    p = p.mul_add(r, f64x8::splat_t(token, 1.0)); // 1/1!
    p = p.mul_add(r, f64x8::splat_t(token, 1.0)); // 1/0!
    let res = p * two_pow_k(token, k);
    let under = k.simd_lt(f64x8::splat_t(token, -1021.5));
    let over = k.simd_gt(f64x8::splat_t(token, 1023.5));
    f64x8::blend(
        under,
        f64x8::zero_t(token),
        f64x8::blend(over, f64x8::splat_t(token, f64::INFINITY), res),
    )
}

/// `ln(x)` per lane, ~2 ulp. `x = m·2^e` with `m` folded into
/// `[√2/2, √2)`; `ln m = 2·atanh(z)`, `z = (m−1)/(m+1) ∈ [−0.172, 0.172]`
/// — a degree-19 odd polynomial. Denormals are pre-scaled by `2^52`.
/// Edges match libm: `x = 0 → −∞`, `x < 0` or `NaN → NaN`,
/// `x = +∞ → +∞`.
#[inline(always)]
fn ln8<T: F64x8Backend>(token: T, x: f64x8<T>) -> f64x8<T> {
    // Pre-scale denormals into the normal range so the exponent field
    // extraction below works; the lane's `e` is corrected by −52.
    let tiny = x.simd_lt(f64x8::splat_t(token, 2.225_073_858_507_201_4e-308));
    let xs = f64x8::blend(
        tiny,
        x * f64x8::splat_t(token, 4.503_599_627_370_496e15),
        x,
    );

    let xa = xs.to_array();
    let mut ef = [0.0f64; 8];
    let mut mb = [0.0f64; 8];
    for i in 0..8 {
        let b = xa[i].to_bits();
        ef[i] = (((b >> 52) & 0x7ff) as i64 - 1023) as f64;
        mb[i] = f64::from_bits((b & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    }
    let m = f64x8::from_array_t(token, mb);
    let e = f64x8::from_array_t(token, ef);
    let e = e - f64x8::blend(tiny, f64x8::splat_t(token, 52.0), f64x8::zero_t(token));

    // m ∈ [1, 2); fold m ≥ √2 down so m ∈ [√2/2, √2), keeping
    // z = (m−1)/(m+1) inside [−0.172, 0.172].
    let sqrt2 = f64x8::splat_t(token, std::f64::consts::SQRT_2);
    let hi = m.simd_ge(sqrt2);
    let m = f64x8::blend(hi, m * f64x8::splat_t(token, 0.5), m);
    let e = e + f64x8::blend(hi, f64x8::splat_t(token, 1.0), f64x8::zero_t(token));

    let one = f64x8::splat_t(token, 1.0);
    let z = (m - one) / (m + one);
    let zz = z * z;
    // 2·atanh(z) = 2z·(1 + z²/3 + z⁴/5 + … + z¹⁶/17).
    let mut p = f64x8::splat_t(token, 1.0 / 17.0);
    p = p.mul_add(zz, f64x8::splat_t(token, 1.0 / 15.0));
    p = p.mul_add(zz, f64x8::splat_t(token, 1.0 / 13.0));
    p = p.mul_add(zz, f64x8::splat_t(token, 1.0 / 11.0));
    p = p.mul_add(zz, f64x8::splat_t(token, 1.0 / 9.0));
    p = p.mul_add(zz, f64x8::splat_t(token, 1.0 / 7.0));
    p = p.mul_add(zz, f64x8::splat_t(token, 1.0 / 5.0));
    p = p.mul_add(zz, f64x8::splat_t(token, 1.0 / 3.0));
    p = p.mul_add(zz, one);
    let res = (z + z).mul_add(p, e * f64x8::splat_t(token, std::f64::consts::LN_2));

    // Edge lanes: the mantissa/exponent trick computes finite garbage on
    // them, so libm results are blended in — `0 → −∞`, `x < 0 → NaN`,
    // `NaN → NaN`, `+∞ → +∞`.
    let zero = f64x8::zero_t(token);
    let nan = f64x8::splat_t(token, f64::NAN);
    let inf = f64x8::splat_t(token, f64::INFINITY);
    let res = f64x8::blend(x.simd_eq(zero), f64x8::splat_t(token, f64::NEG_INFINITY), res);
    let res = f64x8::blend(x.simd_eq(inf), inf, res);
    let res = f64x8::blend(x.simd_lt(zero), nan, res);
    // Ordered compares make `x == x` false exactly for NaN — portable
    // `is_nan` (some backends lack an unordered `ne`).
    f64x8::blend(x.simd_eq(x).not(), nan, res)
}

/// `|x|^q` per lane: `exp(q·ln|x|)`, with `0^q = 0` (`q > 0` for all
/// callers). `±∞` and `NaN` propagate through `exp8`'s edge blends.
#[inline(always)]
fn abs_pow8<T: F64x8Backend>(token: T, x: f64x8<T>, q: f64x8<T>) -> f64x8<T> {
    let zero = f64x8::zero_t(token);
    let r = exp8(token, q * ln8(token, x.abs()));
    f64x8::blend(x.simd_eq(zero), zero, r)
}

/// `x^q` per lane without the abs — matches scalar `powf`: `x < 0` → NaN
/// via `ln8`, `x = 0` → 0.
#[inline(always)]
fn raw_pow8<T: F64x8Backend>(token: T, x: f64x8<T>, q: f64x8<T>) -> f64x8<T> {
    let zero = f64x8::zero_t(token);
    let r = exp8(token, q * ln8(token, x));
    f64x8::blend(x.simd_eq(zero), zero, r)
}

/// `sign(x)·|x|^q` per lane. `−0` yields `+0` (scalar `signum(−0)·0`
/// gives `−0`) — a signed-zero difference under the relaxed contract.
#[inline(always)]
fn sign_pow8<T: F64x8Backend>(token: T, x: f64x8<T>, q: f64x8<T>) -> f64x8<T> {
    let r = abs_pow8(token, x, q);
    let neg = x.simd_lt(f64x8::zero_t(token));
    f64x8::blend(neg, -r, r)
}

/// Full-`Vec` elementwise helpers — the relaxed path runs the block
/// kernel on all full 8-lane chunks and libm on the tail.
macro_rules! vec_map {
    ($doc:literal, $fn_:ident, $inner:ident, |$t:ident, $v:ident, $q:ident| $body:expr, $tail:expr) => {
        #[doc = $doc]
        #[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
        fn $inner(token: Token, x: &[f64], q: f64, out: &mut [f64]) {
            let qv = f64x8::splat_t(token, q);
            for (c, o) in x.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
                let $t = token;
                let $v = f64x8::load_t(token, c.try_into().unwrap());
                let $q = qv;
                $body.store((&mut o[..8]).try_into().unwrap());
            }
        }

        #[doc = $doc]
        ///
        /// `q` is the exponent parameter for the pow maps (unused by the
        /// `exp`/`ln` variants); the non-multiple-of-8 tail uses libm.
        #[allow(dead_code)] // some variants are exercised only via tests
        pub(super) fn $fn_(x: &[f64], q: f64) -> Vec<f64> {
            let full = x.len() & !7;
            let mut out = vec![0.0; x.len()];
            archmage::attuned!(
                $inner(&x[..full], q, &mut out[..full]),
                [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
            );
            for (o, &v) in out[full..].iter_mut().zip(&x[full..]) {
                *o = $tail(v, q);
            }
            out
        }
    };
}

vec_map!(
    "`exp(x)` elementwise.",
    exp_vec,
    exp_map_inner,
    |t, v, _qv| exp8(t, v),
    |v: f64, _q| v.exp()
);
vec_map!(
    "`ln(x)` elementwise.",
    ln_vec,
    ln_map_inner,
    |t, v, _qv| ln8(t, v),
    |v: f64, _q| v.ln()
);
vec_map!(
    "`|x|^q` elementwise.",
    abs_pow_vec,
    abs_pow_map_inner,
    |t, v, qv| abs_pow8(t, v, qv),
    |v: f64, q| v.abs().powf(q)
);
vec_map!(
    "`sign(x)·|x|^q` elementwise.",
    sign_pow_vec,
    sign_pow_map_inner,
    |t, v, qv| sign_pow8(t, v, qv),
    |v: f64, q| v.signum() * v.abs().powf(q)
);

/// `out = 1 − exp(scale·x)` block kernel — `P_map` generation and the
/// `do_sprob_sum` post-pass (`scale = 1.0`).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn one_mexp_scale_inner(token: Token, x: &[f64], scale: f64, out: &mut [f64]) {
    let sv = f64x8::splat_t(token, scale);
    let one = f64x8::splat_t(token, 1.0);
    for (c, o) in x.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        let v = f64x8::load_t(token, c.try_into().unwrap());
        (one - exp8(token, v * sv)).store((&mut o[..8]).try_into().unwrap());
    }
}

/// `out = 1 − exp(scale·x)` elementwise.
pub(super) fn one_mexp_scale(x: &[f64], scale: f64) -> Vec<f64> {
    let full = x.len() & !7;
    let mut out = vec![0.0; x.len()];
    archmage::attuned!(
        one_mexp_scale_inner(&x[..full], scale, &mut out[..full]),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for (o, &v) in out[full..].iter_mut().zip(&x[full..]) {
        *o = 1.0 - (scale * v).exp();
    }
    out
}

/// `out = ln(1.0 − x + 1e-4)` block kernel — `log1m` for `do_sprob_sum`.
/// The argument is formed with the scalar op order (`(1 − x) + 1e-4`).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn ln_1mx_inner(token: Token, x: &[f64], out: &mut [f64]) {
    let one = f64x8::splat_t(token, 1.0);
    let eps = f64x8::splat_t(token, 1e-4);
    for (c, o) in x.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        let v = f64x8::load_t(token, c.try_into().unwrap());
        ln8(token, (one - v) + eps).store((&mut o[..8]).try_into().unwrap());
    }
}

/// `out = ln(1.0 − x + 1e-4)` elementwise.
pub(super) fn ln_1mx(x: &[f64]) -> Vec<f64> {
    let full = x.len() & !7;
    let mut out = vec![0.0; x.len()];
    archmage::attuned!(
        ln_1mx_inner(&x[..full], &mut out[..full]),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for (o, &v) in out[full..].iter_mut().zip(&x[full..]) {
        *o = (1.0 - v + 1e-4).ln();
    }
    out
}

/// Masked-band denominator — `e / (n^(2p) + n_mask²)^0.5` with
/// `n_mask = k_self·|sm|^q + kxo0 + k_xn·|xn|^q` — one fused pass over
/// three pow evaluations (the masking arm of `apply_band`). `n` uses
/// `raw_pow8` (scalar `powf` has no abs); `sm`/`xn` match the scalar's
/// explicit `.abs()`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
#[allow(clippy::too_many_arguments)]
fn masked_e_inner(
    token: Token,
    e: &[f64],
    n: &[f64],
    sm: &[f64],
    xn: &[f64],
    k_self: f64,
    kxo0: f64,
    k_xn: f64,
    q: f64,
    p2: f64,
    out: &mut [f64],
) {
    let ks = f64x8::splat_t(token, k_self);
    let kxo = f64x8::splat_t(token, kxo0);
    let kxn = f64x8::splat_t(token, k_xn);
    let qv = f64x8::splat_t(token, q);
    let p2v = f64x8::splat_t(token, p2);
    for i in 0..e.len() / 8 {
        let (a, b) = (8 * i, 8 * i + 8);
        let ev = f64x8::load_t(token, e[a..b].try_into().unwrap());
        let nv = f64x8::load_t(token, n[a..b].try_into().unwrap());
        let smv = f64x8::load_t(token, sm[a..b].try_into().unwrap());
        let xnv = f64x8::load_t(token, xn[a..b].try_into().unwrap());
        let n_mask = kxn.mul_add(
            abs_pow8(token, xnv, qv),
            ks.mul_add(abs_pow8(token, smv, qv), kxo),
        );
        let den = n_mask
            .mul_add(n_mask, raw_pow8(token, nv, p2v))
            .sqrt();
        (ev / den).store((&mut out[a..b]).try_into().unwrap());
    }
}

/// Masked-band denominator elementwise.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(super) fn masked_e_vec(
    e: &[f64],
    n: &[f64],
    sm: &[f64],
    xn: &[f64],
    k_self: f64,
    kxo0: f64,
    k_xn: f64,
    q: f64,
    p2: f64,
) -> Vec<f64> {
    let full = e.len() & !7;
    let mut out = vec![0.0; e.len()];
    archmage::attuned!(
        masked_e_inner(
            &e[..full],
            &n[..full],
            &sm[..full],
            &xn[..full],
            k_self,
            kxo0,
            k_xn,
            q,
            p2,
            &mut out[..full]
        ),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for i in full..e.len() {
        let n_mask = k_self * sm[i].abs().powf(q) + kxo0 + k_xn * xn[i].abs().powf(q);
        out[i] = e[i] / (n[i].powf(p2) + n_mask.powi(2)).sqrt();
    }
    out
}

/// `out = e / n^p` block kernel (unmasked arm of `apply_band`); `raw_pow8`
/// matches scalar `powf`'s `n < 0 → NaN`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn e_div_np_inner(token: Token, e: &[f64], n: &[f64], p: f64, out: &mut [f64]) {
    let pv = f64x8::splat_t(token, p);
    for i in 0..e.len() / 8 {
        let (a, b) = (8 * i, 8 * i + 8);
        let ev = f64x8::load_t(token, e[a..b].try_into().unwrap());
        let nv = f64x8::load_t(token, n[a..b].try_into().unwrap());
        (ev / raw_pow8(token, nv, pv)).store((&mut out[a..b]).try_into().unwrap());
    }
}

/// `out = e / n^p` elementwise.
#[must_use]
pub(super) fn e_div_np_vec(e: &[f64], n: &[f64], p: f64) -> Vec<f64> {
    let full = e.len() & !7;
    let mut out = vec![0.0; e.len()];
    archmage::attuned!(
        e_div_np_inner(&e[..full], &n[..full], p, &mut out[..full]),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for i in full..e.len() {
        out[i] = e[i] / n[i].powf(p);
    }
    out
}

/// `Σ|d·m|^q` block kernel — the per-band error accumulator. Vector
/// accumulation reassociates the fold (relaxed contract); the scalar
/// tail keeps libm `powf`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn acc_abspow_inner(token: Token, d: &[f64], m: &[f64], q: f64) -> f64 {
    let qv = f64x8::splat_t(token, q);
    let mut acc = f64x8::zero_t(token);
    for i in 0..d.len() / 8 {
        let (a, b) = (8 * i, 8 * i + 8);
        let dv = f64x8::load_t(token, d[a..b].try_into().unwrap());
        let mv = f64x8::load_t(token, m[a..b].try_into().unwrap());
        acc = acc + abs_pow8(token, dv * mv, qv);
    }
    acc.reduce_add()
}

/// `Σ|d·m|^q` over the full slices (tail lanes use libm).
#[must_use]
pub(super) fn acc_abspow(d: &[f64], m: &[f64], q: f64) -> f64 {
    let full = d.len() & !7;
    let mut acc = archmage::attuned!(
        acc_abspow_inner(&d[..full], &m[..full], q),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for i in full..d.len() {
        acc += (d[i] * m[i]).abs().powf(q);
    }
    acc
}

/// `out = (a·inv)^(1/norm)` block kernel — `mutual_masking`'s output map;
/// `raw_pow8` matches scalar `powf`'s negative-base NaN.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn pow_scale_inner(token: Token, a: &[f64], inv: f64, inv_norm: f64, out: &mut [f64]) {
    let iv = f64x8::splat_t(token, inv);
    let nv = f64x8::splat_t(token, inv_norm);
    for (c, o) in a.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        let v = f64x8::load_t(token, c.try_into().unwrap());
        raw_pow8(token, v * iv, nv).store((&mut o[..8]).try_into().unwrap());
    }
}

/// `out = (a·inv)^(1/norm)` elementwise.
#[must_use]
pub(super) fn pow_scale_vec(a: &[f64], inv: f64, inv_norm: f64) -> Vec<f64> {
    let full = a.len() & !7;
    let mut out = vec![0.0; a.len()];
    archmage::attuned!(
        pow_scale_inner(&a[..full], inv, inv_norm, &mut out[..full]),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for (o, &v) in out[full..].iter_mut().zip(&a[full..]) {
        *o = (v * inv).powf(inv_norm);
    }
    out
}

/// `out = ln(max(x, floor))` block kernel — `local_adapt`'s log pass and
/// `geomean`'s `max(0)` variant.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn ln_max_inner(token: Token, x: &[f64], floor: f64, out: &mut [f64]) {
    let fv = f64x8::splat_t(token, floor);
    for (c, o) in x.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        let v = f64x8::load_t(token, c.try_into().unwrap());
        ln8(token, v.max(fv)).store(o.try_into().unwrap());
    }
}

/// `out = ln(max(x, floor))` elementwise.
#[must_use]
pub(super) fn ln_max_vec(x: &[f64], floor: f64) -> Vec<f64> {
    let full = x.len() & !7;
    let mut out = vec![0.0; x.len()];
    archmage::attuned!(
        ln_max_inner(&x[..full], floor, &mut out[..full]),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for (o, &v) in out[full..].iter_mut().zip(&x[full..]) {
        *o = v.max(floor).ln();
    }
    out
}

/// `out = Σ_k a_k·exp(−b_k·ρ)` block kernel — the four-term MTF shape,
/// one fused pass per eight grid points.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn mtf_inner(token: Token, rho: &[f64], a: &[f64; 4], b: &[f64; 4], out: &mut [f64]) {
    let av: [f64x8; 4] = std::array::from_fn(|k| f64x8::splat_t(token, a[k]));
    let bv: [f64x8; 4] = std::array::from_fn(|k| f64x8::splat_t(token, b[k]));
    for (c, o) in rho.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        let r = f64x8::load_t(token, c.try_into().unwrap());
        let mut acc = f64x8::zero_t(token);
        for k in 0..4 {
            acc = acc + av[k] * exp8(token, -(bv[k] * r));
        }
        acc.store(o.try_into().unwrap());
    }
}

/// `out = Σ_k a_k·exp(−b_k·ρ)` elementwise (tail lanes use libm).
#[must_use]
pub(super) fn mtf_vec(rho: &[f64], a: &[f64; 4], b: &[f64; 4]) -> Vec<f64> {
    let full = rho.len() & !7;
    let mut out = vec![0.0; rho.len()];
    archmage::attuned!(
        mtf_inner(&rho[..full], a, b, &mut out[..full]),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for (i, o) in out[full..].iter_mut().enumerate() {
        let r = rho[full + i];
        *o = (0..4).map(|k| a[k] * (-b[k] * r).exp()).sum::<f64>();
    }
    out
}

/// One axis of the separable FFT-domain Gaussian: `kx²` for each lattice
/// position. `(0.5 + xx/n) % 1.0 − 0.5` wraps around `1.0`; for the
/// `xx/n ∈ [0,1)` domain the `%` is exactly `u − 1` for `u ≥ 1` — the
/// conditional subtract is bit-identical to `fmod` here.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn kern_axis_inner(token: Token, n: usize, out: &mut [f64]) {
    let np = f64x8::splat_t(token, n as f64);
    let half = f64x8::splat_t(token, 0.5);
    let one = f64x8::splat_t(token, 1.0);
    let tau = f64x8::splat_t(token, core::f64::consts::TAU);
    let mut xx = f64x8::from_array_t(token, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let step = f64x8::splat_t(token, 8.0);
    for o in out.chunks_exact_mut(8) {
        let u = half + xx / np;
        let w = f64x8::blend(u.simd_ge(one), u - one, u) - half;
        let kx = w * tau;
        (kx * kx).store(o.try_into().unwrap());
        xx = xx + step;
    }
}

/// `out = exp(−0.5·(kx2 + ky2)·σ·σ)` for one kernel row — `kx2`/`ky2`
/// are precomputed squares (`ky2` scalar for the row).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn kern_row_inner(token: Token, kx2: &[f64], ky2: f64, sigma: f64, out: &mut [f64]) {
    let ky = f64x8::splat_t(token, ky2);
    let nsm5 = f64x8::splat_t(token, -0.5);
    let sv = f64x8::splat_t(token, sigma);
    for (c, o) in kx2.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        let v = f64x8::load_t(token, c.try_into().unwrap());
        let arg = nsm5 * (v + ky) * sv * sv;
        exp8(token, arg).store(o.try_into().unwrap());
    }
}

/// The full FFT-domain Gaussian lattice `exp(−0.5·(kx²+ky²)·σ²)` —
/// `pw·ph` elements, row-major — replacing the scalar `fmod`+`exp` loop
/// in `fast_gauss`. `pw`/`ph` are even (the doubled lattice), so each
/// row is a whole number of 8-lane blocks when `pw % 8 == 0`; otherwise
/// rows take a scalar tail.
#[must_use]
pub(super) fn gauss_kern(pw: usize, ph: usize, sigma: f64) -> Vec<f64> {
    let mut kx2 = vec![0.0; pw];
    let full_w = pw & !7;
    archmage::attuned!(
        kern_axis_inner(pw, &mut kx2[..full_w]),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for (xx, v) in kx2[full_w..].iter_mut().enumerate() {
        let u = 0.5 + (full_w + xx) as f64 / pw as f64;
        let kx = (u % 1.0 - 0.5) * core::f64::consts::TAU;
        *v = kx * kx;
    }
    let mut k = vec![0.0; pw * ph];
    for y in 0..ph {
        let u = 0.5 + y as f64 / ph as f64;
        let w = (u % 1.0 - 0.5) * core::f64::consts::TAU;
        let row = &mut k[y * pw..(y + 1) * pw];
        archmage::attuned!(
            kern_row_inner(&kx2[..full_w], w * w, sigma, &mut row[..full_w]),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
        for xx in full_w..pw {
            row[xx] = (-0.5 * (kx2[xx] + w * w) * sigma * sigma).exp();
        }
    }
    k
}

/// `out = sqrt(kx2 + ky2)` row kernel for `cycdeg_image` — pure IEEE
/// ops in the scalar element order, so results are bit-identical under
/// both FP modes.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn cycdeg_row_inner(token: Token, kx2: &[f64], ky2: f64, out: &mut [f64]) {
    let ky = f64x8::splat_t(token, ky2);
    for (c, o) in kx2.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        let v = f64x8::load_t(token, c.try_into().unwrap());
        (v + ky).sqrt().store(o.try_into().unwrap());
    }
}

/// `create_cycdeg_image` — per-coefficient spatial frequency in cpd,
/// `sqrt(kx²+ky²)·ppd` on the w×h lattice. Bit-identical to the scalar
/// nested-loop version (same per-element op order; the `%` wraps are
/// exact conditional subtracts on the `[0.5, 1.5)` domain).
#[must_use]
pub(super) fn cycdeg(width: usize, height: usize, ppd: f64) -> Vec<f64> {
    let mut kx2 = vec![0.0; width];
    for (xx, v) in kx2.iter_mut().enumerate() {
        let kx = ((0.5 + xx as f64 / width as f64) % 1.0 - 0.5) * ppd;
        *v = kx * kx;
    }
    let full_w = width & !7;
    let mut out = vec![0.0; width * height];
    for y in 0..height {
        let ky = ((0.5 + y as f64 / height as f64) % 1.0 - 0.5) * ppd;
        let row = &mut out[y * width..(y + 1) * width];
        archmage::attuned!(
            cycdeg_row_inner(&kx2[..full_w], ky * ky, &mut row[..full_w]),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
        for xx in full_w..width {
            row[xx] = (kx2[xx] + ky * ky).sqrt();
        }
    }
    out
}

/// `pm *= 1 − exp(ln(0.5)·(|pr−pt| / pr / m_thr)^3.5)` — the
/// `do_pixel_threshold` Weber-fraction pass. `raw_pow8` matches scalar
/// `powf`: `w < 0` (negative `pr`) → NaN, propagating like libm.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn thresh_inner(token: Token, pr: &[f64], pt: &[f64], m_thr: f64, pm: &mut [f64]) {
    let one = f64x8::splat_t(token, 1.0);
    let ln05 = f64x8::splat_t(token, 0.5f64.ln());
    let e35 = f64x8::splat_t(token, 3.5);
    let thr = f64x8::splat_t(token, m_thr);
    for i in 0..pm.len() / 8 {
        let (a, b) = (8 * i, 8 * i + 8);
        let prv = f64x8::load_t(token, pr[a..b].try_into().unwrap());
        let ptv = f64x8::load_t(token, pt[a..b].try_into().unwrap());
        let w = (prv - ptv).abs() / prv / thr;
        let is_diff = one - exp8(token, ln05 * raw_pow8(token, w, e35));
        let pmv = f64x8::load_t(token, pm[a..b].try_into().unwrap());
        (pmv * is_diff).store((&mut pm[a..b]).try_into().unwrap());
    }
}

/// `pm *= 1 − exp(ln(0.5)·(|pr−pt| / pr / m_thr)^3.5)` elementwise
/// (tail lanes use libm).
pub(super) fn thresh_mul(pm: &mut [f64], pr: &[f64], pt: &[f64], m_thr: f64) {
    let full = pm.len() & !7;
    let (pm_full, pm_tail) = pm.split_at_mut(full);
    archmage::attuned!(
        thresh_inner(&pr[..full], &pt[..full], m_thr, pm_full),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    for (i, pmv) in pm_tail.iter_mut().enumerate() {
        let i = full + i;
        let is_diff = 1.0
            - (0.5f64.ln() * ((pr[i] - pt[i]).abs() / pr[i] / m_thr).powf(3.5)).exp();
        *pmv *= is_diff;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(a: f64, b: f64) -> f64 {
        if a == b {
            0.0
        } else {
            (a - b).abs() / a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
        }
    }

    // All sweeps feed ≥8 lanes so the vector path is exercised — a
    // single-element input would silently fall through to the libm tail.

    #[test]
    fn exp_accuracy() {
        let xs: Vec<f64> = (-1000..=1000).map(|i| i as f64 * 0.025).collect();
        let got = exp_vec(&xs, 0.0);
        for (i, (&x, &g)) in xs.iter().zip(&got).enumerate() {
            let want = x.exp();
            if want.is_finite() && want > 0.0 {
                assert!(rel_err(g, want) < 1e-13, "exp({x}) lane {i}: {g:e} vs {want:e}");
            }
        }
        // Edges — padded to 8 lanes to stay on the vector path.
        let got = exp_vec(
            &[f64::NEG_INFINITY, f64::INFINITY, f64::NAN, -1000.0, 800.0, 0.0, -0.0, 1.0],
            0.0,
        );
        assert_eq!(got[0], 0.0);
        assert_eq!(got[1], f64::INFINITY);
        assert!(got[2].is_nan());
        assert_eq!(got[3], 0.0);
        assert!(got[4].is_finite() || got[4] == f64::INFINITY);
        assert_eq!(got[5], 1.0);
    }

    #[test]
    fn ln_accuracy_and_edges() {
        let xs: Vec<f64> = (1..=4000)
            .map(|i| (i as f64 * 0.005).exp())
            .collect();
        let got = ln_vec(&xs, 0.0);
        for (i, (&x, &g)) in xs.iter().zip(&got).enumerate() {
            let want = x.ln();
            assert!(rel_err(g, want) < 1e-13, "ln({x}) lane {i}: {g:e} vs {want:e}");
        }
        // Sub-unity, denormal and edge inputs.
        let xs = [1e-320, 5e-324, 2.3e-300, 1e-100, 0.5, 0.999, 1.0, 1.001];
        let got = ln_vec(&xs, 0.0);
        for (i, (&x, &g)) in xs.iter().zip(&got).enumerate() {
            assert!(rel_err(g, x.ln()) < 1e-13, "ln({x:e}) lane {i}");
        }
        let got = ln_vec(&[0.0, -1.0, f64::NAN, f64::INFINITY, 2.0, 3.0, 4.0, 5.0], 0.0);
        assert_eq!(got[0], f64::NEG_INFINITY);
        assert!(got[1].is_nan());
        eprintln!("ln got = {got:?}");
        assert!(got[2].is_nan(), "got[2] = {:e}", got[2]);
        assert_eq!(got[3], f64::INFINITY);
    }

    #[test]
    fn pow_accuracy_and_edges() {
        for &q in &[0.8, 1.25, 2.4, 0.5, 3.0] {
            // Signed sweep including |x| < 1 — the domain the masking
            // bands actually feed.
            let xs: Vec<f64> = (0..1200).map(|i| (i as f64 + 0.5) * 0.02 - 12.0).collect();
            let got = abs_pow_vec(&xs, q);
            for (i, (&x, &g)) in xs.iter().zip(&got).enumerate() {
                let want = x.abs().powf(q);
                if want.is_finite() && want != 0.0 {
                    assert!(rel_err(g, want) < 5e-12, "|{x}|^{q} lane {i}: {g:e} vs {want:e}");
                }
            }
        }
        let got = abs_pow_vec(&[0.0, f64::INFINITY, f64::NAN, 2.0, 4.0, 8.0, 16.0, 0.5], 2.0);
        assert_eq!(got[0], 0.0);
        assert_eq!(got[1], f64::INFINITY);
        assert!(got[2].is_nan());
        assert!(rel_err(got[3], 4.0) < 5e-12);
        // Sign propagation
        let r = sign_pow_vec(&[-4.0, 0.0, 9.0, -9.0, 2.0, -2.0, 1.0, -1.0], 2.0);
        assert!(rel_err(r[0], -16.0) < 5e-12);
        assert_eq!(r[1], 0.0);
        assert!(rel_err(r[2], 81.0) < 5e-12);
        assert!(rel_err(r[3], -81.0) < 5e-12);
        // Tail lanes (non-multiple-of-8) use libm
        let t = sign_pow_vec(&[2.0; 11], 3.0);
        assert_eq!(t.len(), 11);
        for (i, &v) in t.iter().enumerate() {
            assert!(rel_err(v, 8.0) < 5e-12, "tail lane {i}: {v:e}");
        }
    }

    #[test]
    fn fused_kernels_match_scalar() {
        let e: Vec<f64> = (0..40).map(|i| 0.5 + i as f64 * 0.11).collect();
        let n: Vec<f64> = (0..40).map(|i| 0.3 + i as f64 * 0.07).collect();
        let sm: Vec<f64> = (0..40).map(|i| 0.1 + i as f64 * 0.03).collect();
        let xn: Vec<f64> = (0..40).map(|i| 0.2 + i as f64 * 0.02).collect();
        let (ks, kxo0, kxn, q, p2) = (1.5, 0.0, 0.7, 0.9, 2.4);
        let got = masked_e_vec(&e, &n, &sm, &xn, ks, kxo0, kxn, q, p2);
        for (i, &g) in got.iter().enumerate() {
            let n_mask = ks * sm[i].abs().powf(q) + kxo0 + kxn * xn[i].abs().powf(q);
            let want = e[i] / (n[i].powf(p2) + n_mask.powi(2)).sqrt();
            assert!(rel_err(g, want) < 5e-12, "masked_e[{i}]");
        }
        let got = e_div_np_vec(&e, &n, 1.2);
        for (i, &g) in got.iter().enumerate() {
            assert!(rel_err(g, e[i] / n[i].powf(1.2)) < 5e-12, "e_div_np[{i}]");
        }
        let dm: Vec<f64> = (0..40).map(|i| 0.8 + i as f64 * 0.01).collect();
        let got = acc_abspow(&e, &dm, 0.8);
        let want: f64 = e.iter().zip(&dm).map(|(&a, &b)| (a * b).abs().powf(0.8)).sum();
        assert!(rel_err(got, want) < 5e-12, "acc_abspow");
        let got = one_mexp_scale(&e, 0.5f64.ln());
        for (i, &g) in got.iter().enumerate() {
            assert!(rel_err(g, 1.0 - (0.5f64.ln() * e[i]).exp()) < 5e-12, "one_mexp[{i}]");
        }
        let got = ln_1mx(&e.iter().map(|v| v.min(0.99)).collect::<Vec<_>>());
        let pm: Vec<f64> = e.iter().map(|v| v.min(0.99)).collect();
        for (i, &g) in got.iter().enumerate() {
            assert!(rel_err(g, (1.0 - pm[i] + 1e-4).ln()) < 5e-12, "ln_1mx[{i}]");
        }
    }
}
