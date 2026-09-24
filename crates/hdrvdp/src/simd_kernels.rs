//! magetypes-dispatched SIMD kernels for the HDR-VDP-2 hot paths.
//!
//! Coverage:
//!
//! - `transducer_plane`    — the masking transducer: `ex_diff = sign_pow(Δ/n, p)·n`
//!   plus the three masking terms and the `sqrt(n^(2p) + n_mask²)` denominator.
//!   Replaces ~6 scalar `powf` calls per pixel with `pow_midp` lanes.
//! - `sign_pow_reshape`    — `out = sign_pow(v/n, pf)·n`, the psychometric
//!   reshape that feeds the visibility pyramid (full-result path only).
//! - `masked_sq_sum`       — `Σ (v·m)²` in chunked f32x8 lanes, widened to f64
//!   per block — the per-plane `msre` reduction.
//!
//! All entry points route through `archmage::incant!`. Kernels needing
//! `pow`/`log2`/`exp2` use the `F32x8Convert` tier cascade
//! `[v3, neon, wasm128, scalar]` (AVX-512 tokens lack that backend);
//! pure-arithmetic kernels use the full `[v4x, v4, v3, neon, wasm128,
//! scalar]` cascade. Tail elements run the same scalar arithmetic the
//! lane code implements.
//!
//! Precision: `pow_midp` is ~3 ULP — far inside the golden tolerances the
//! f32 pipeline is gated against (worst case `res.Q` 5e-3, `P_map` 5e-2).
//! `sign_pow(x, e)` is evaluated as `x·|x|^(e−1)`, which is the identical
//! expression for `e > 1` (all model exponents here are) and avoids any
//! sign-lane handling: `x·0 = 0` when `x = 0`, sign preserved otherwise.

/// Per-pixel transducer + masking terms for one band plane.
///
/// Writes `out[i] = d[i]` where, with `n = n_ncsf[i]` (`1` for the base band
/// — pass `csf = None`), `bn = band_norm`:
///
/// ```text
/// ex_diff = sign_pow((t[i] − r[i])/bn, p)·bn
/// n_mask  = bn·(k_self·(sm[i]/(n·bn))^q + k_xo·(xo[i]/(n·bn))^q + k_xn·(xn[i]/n)^q)
///   xo[i] = max(xo_tot[i] − sm[i], 0)
/// d[i]    = ex_diff / sqrt(n^(2p) + n_mask²)      (do_masking)
///         = ex_diff / n^p                          (otherwise)
/// ```
///
/// All input slices and `out` have the same length.
#[allow(clippy::too_many_arguments)]
pub fn transducer_plane(
    t: &[f32],
    r: &[f32],
    csf: Option<&[f32]>,
    sm: &[f32],
    xo_tot: &[f32],
    xn: &[f32],
    band_norm: f32,
    p: f32,
    q: f32,
    k_self: f32,
    k_xo: f32,
    k_xn: f32,
    do_masking: bool,
    out: &mut [f32],
) {
    debug_assert_eq!(t.len(), r.len());
    debug_assert_eq!(t.len(), sm.len());
    debug_assert_eq!(t.len(), xo_tot.len());
    debug_assert_eq!(t.len(), xn.len());
    debug_assert_eq!(t.len(), out.len());
    if let Some(c) = csf {
        debug_assert_eq!(t.len(), c.len());
    }
    archmage::incant!(
        transducer_inner(
            t, r, csf, sm, xo_tot, xn, band_norm, p, q, k_self, k_xo, k_xn, do_masking, out
        ),
        [v3, neon, wasm128, scalar]
    );
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn transducer_inner(
    token: Token,
    t: &[f32],
    r: &[f32],
    csf: Option<&[f32]>,
    sm: &[f32],
    xo_tot: &[f32],
    xn: &[f32],
    band_norm: f32,
    p: f32,
    q: f32,
    k_self: f32,
    k_xo: f32,
    k_xn: f32,
    do_masking: bool,
    out: &mut [f32],
) {
    let n = t.len();
    let bn = f32x8::splat(token, band_norm);
    let inv_bn = f32x8::splat(token, 1.0 / band_norm);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let pm1 = p - 1.0;
    let p2 = 2.0 * p;
    let ks = f32x8::splat(token, k_self);
    let kxo = f32x8::splat(token, k_xo);
    let kxn = f32x8::splat(token, k_xn);

    let mut i = 0usize;
    while i + 8 <= n {
        let load = |s: &[f32]| f32x8::load(token, (&s[i..i + 8]).try_into().unwrap());
        let tv = load(t);
        let rv = load(r);
        let bd = (tv - rv) * inv_bn;
        let ex = bd * bd.abs().pow_midp(pm1) * bn;
        let nv = match csf {
            Some(c) => one / f32x8::load(token, (&c[i..i + 8]).try_into().unwrap()),
            None => one,
        };
        let d = if do_masking {
            let smv = load(sm);
            let xo = (load(xo_tot) - smv).max(zero);
            let xnv = load(xn);
            let nb = nv * bn;
            let nm = bn
                * (ks * (smv / nb).pow_midp(q)
                    + kxo * (xo / nb).pow_midp(q)
                    + kxn * (xnv / nv).pow_midp(q));
            ex / (nv.pow_midp(p2) + nm * nm).sqrt()
        } else {
            ex / nv.pow_midp(p)
        };
        d.store((&mut out[i..i + 8]).try_into().unwrap());
        i += 8;
    }
    while i < n {
        let bd = (t[i] - r[i]) / band_norm;
        let ex = bd * bd.abs().powf(p - 1.0) * band_norm;
        let nv = csf.map_or(1.0, |c| 1.0 / c[i]);
        out[i] = if do_masking {
            let smv = sm[i];
            let xo = (xo_tot[i] - smv).max(0.0);
            let nb = nv * band_norm;
            let nm = band_norm
                * (k_self * (smv / nb).powf(q)
                    + k_xo * (xo / nb).powf(q)
                    + k_xn * (xn[i] / nv).powf(q));
            ex / (nv.powf(2.0 * p) + nm * nm).sqrt()
        } else {
            ex / nv.powf(p)
        };
        i += 1;
    }
}

/// `out[i] = sign_pow(v[i]/n, pf)·n` — the psychometric reshape between the
/// transducer output and the visibility pyramid's `D` bands.
pub fn sign_pow_reshape(v: &[f32], band_norm: f32, pf: f32, out: &mut [f32]) {
    debug_assert_eq!(v.len(), out.len());
    archmage::incant!(
        sign_pow_reshape_inner(v, band_norm, pf, out),
        [v3, neon, wasm128, scalar]
    );
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn sign_pow_reshape_inner(token: Token, v: &[f32], band_norm: f32, pf: f32, out: &mut [f32]) {
    let n = v.len();
    let bn = f32x8::splat(token, band_norm);
    let inv_bn = f32x8::splat(token, 1.0 / band_norm);
    let pfm1 = pf - 1.0;
    let mut i = 0usize;
    while i + 8 <= n {
        let x = f32x8::load(token, (&v[i..i + 8]).try_into().unwrap()) * inv_bn;
        let y = x * x.abs().pow_midp(pfm1) * bn;
        y.store((&mut out[i..i + 8]).try_into().unwrap());
        i += 8;
    }
    while i < n {
        let x = v[i] / band_norm;
        out[i] = x * x.abs().powf(pf - 1.0) * band_norm;
        i += 1;
    }
}

/// Uniform-grid LUT lookup with linear interpolation, `f32` version of
/// [`crate::interp::point_op`] / the photoreceptor's `point_op32`:
/// `pos = (x − x0)·inv_dx` clamped to `[0, len−1]`, then `lut[floor] +
/// t·(lut[floor+1] − lut[floor])`. Out-of-range and NaN inputs follow the
/// scalar edge rules exactly (NaN → `lut[0]` via the pre-clamp to `lo`,
/// matching `clamp32`'s NaN→lo rule before the log-domain lookup).
///
/// The table index is a per-lane gather — 8 scalar loads per chunk — which is
/// still several× cheaper than a scalar `log10` + lerp per pixel.
#[allow(clippy::too_many_arguments)]
pub fn lut_plane(
    x: &[f32],
    lut: &[f32],
    x0: f32,
    inv_dx: f32,
    lo: f32,
    hi: f32,
    log10_input: bool,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), out.len());
    debug_assert!(lut.len() >= 2);
    archmage::incant!(
        lut_plane_inner(x, lut, x0, inv_dx, lo, hi, log10_input, out),
        [v3, neon, wasm128, scalar]
    );
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn lut_plane_inner(
    token: Token,
    x: &[f32],
    lut: &[f32],
    x0: f32,
    inv_dx: f32,
    lo: f32,
    hi: f32,
    log10_input: bool,
    out: &mut [f32],
) {
    let last = lut.len() - 1;
    let lo_v = f32x8::splat(token, lo);
    let hi_v = f32x8::splat(token, hi);
    let x0v = f32x8::splat(token, x0);
    let idxv = f32x8::splat(token, inv_dx);
    let lastv = f32x8::splat(token, last as f32);
    let lastm1 = f32x8::splat(token, (last - 1) as f32);
    let zero = f32x8::zero(token);
    let mut i = 0usize;
    while i + 8 <= x.len() {
        let mut v = f32x8::load(token, (&x[i..i + 8]).try_into().unwrap());
        // clamp32 semantics: NaN lands on `lo` before the log10.
        v = f32x8::blend(v.simd_ne(v), lo_v, v).clamp(lo_v, hi_v);
        let v = if log10_input { v.log10_midp() } else { v };
        let pos = ((v - x0v) * idxv).clamp(zero, lastv);
        let i_f = pos.floor().min(lastm1);
        let t = pos - i_f;
        let idx = i_f.to_i32().to_array();
        let mut a = [0.0f32; 8];
        let mut b = [0.0f32; 8];
        for l in 0..8 {
            let j = idx[l] as usize;
            a[l] = lut[j];
            b[l] = lut[j + 1];
        }
        let av = f32x8::from_array(token, a);
        let bv = f32x8::from_array(token, b);
        (bv - av)
            .mul_add(t, av)
            .store((&mut out[i..i + 8]).try_into().unwrap());
        i += 8;
    }
    while i < x.len() {
        let mut v = x[i];
        if v.is_nan() {
            v = lo;
        }
        let v = v.clamp(lo, hi);
        let v = if log10_input { v.log10() } else { v };
        let pos = ((v - x0) * inv_dx).clamp(0.0, last as f32);
        let j = (pos.floor() as usize).min(last - 1);
        let t = pos - j as f32;
        out[i] = lut[j] + t * (lut[j + 1] - lut[j]);
        i += 1;
    }
}

/// `Σ (v[i]·m[i])²` accumulated in `f32x8` lanes over 2048-element blocks and
/// widened to `f64` per block — the per-plane `msre` numerator. The block
/// bound keeps f32 accumulation error at ~1e-5 relative worst case, far inside
/// the `res.Q` tolerance, while the f64 outer sum matches the scalar order
/// closely enough that this is a pure speed change.
pub fn masked_sq_sum(v: &[f32], m: &[f32]) -> f64 {
    debug_assert_eq!(v.len(), m.len());
    archmage::incant!(
        masked_sq_sum_inner(v, m),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
fn masked_sq_sum_inner(token: Token, v: &[f32], m: &[f32]) -> f64 {
    const BLOCK: usize = 2048;
    let mut total = 0.0f64;
    let mut i = 0usize;
    while i < v.len() {
        let end = (i + BLOCK).min(v.len());
        let mut acc = f32x8::zero(token);
        while i + 8 <= end {
            let a = f32x8::load(token, (&v[i..i + 8]).try_into().unwrap());
            let b = f32x8::load(token, (&m[i..i + 8]).try_into().unwrap());
            let vm = a * b;
            acc += vm * vm;
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        acc.store(&mut lanes);
        total += lanes.iter().map(|x| *x as f64).sum::<f64>();
        while i < end {
            let vm = v[i] * m[i];
            total += (vm as f64) * (vm as f64);
            i += 1;
        }
    }
    total
}
