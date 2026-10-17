//! 16×16 SVD for the DMM head: a LAPACK-routine port in f32 (bidiagonal
//! reduction `dgebd2`, `dorg2r`, `dbdsqr` via `dlasdq`, `dbdsdc`'s scaling
//! and descending sort).
//!
//! Copied verbatim from zenmetrics `crates/zenmetrics-nn/src/dmm.rs` lines
//! 515–1350 at commit e9cd38d2, unchanged at e7e4aff3 (that module is crate-private there).
//! Adapted from Reference-LAPACK (BSD-3-Clause); the notice is in
//! `LICENSE-lapack` next to this crate's `Cargo.toml`.
#![allow(clippy::all)]

// and its `u` is *this* patch's V factor. The head therefore contracts
// `|v_r ∘ v_d|`, not `|u_r ∘ u_d|`.
//
// The score's `wt` term consumes the basis inside degenerate singular
// clusters. Solver implementations may choose different bases. This head
// retains the arithmetic qualified by the frozen same-tap netlib panels.
// Reference-source changes are accepted only after per-row measurement.
// See models/dmm/card.toml for the measured JND/pinned-row counts.
// Only the left (U) factor of the transposed patch is accumulated.
//
// `svd16(m)` therefore transposes `m` first (matching upstream's
// `transpose(2,4)`) and returns `u` of the transposed patch.

// Frozen solver constant; changing it requires passing the full panels.
const D_EPS: f32 = f32::EPSILON;
const D_SAFMIN: f32 = f32::MIN_POSITIVE;

fn dlapy2(x: f32, y: f32) -> f32 {
    let (x, y) = (x.abs(), y.abs());
    let (w, z) = if x < y { (y, x) } else { (x, y) };
    if w == 0.0 {
        0.0
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

fn dnrm2(x: &[f32]) -> f32 {
    let (mut scale, mut ssq) = (0.0f32, 1.0f32);
    for &xi in x {
        if xi != 0.0 {
            let a = xi.abs();
            if scale < a {
                ssq = 1.0 + ssq * (scale / a) * (scale / a);
                scale = a;
            } else {
                ssq += (a / scale) * (a / scale);
            }
        }
    }
    scale * ssq.sqrt()
}

/// DLARFG: householder reflector on (alpha; x). Returns (beta, tau); the
/// normalized tail is written back into `x`.
fn dlarfg(alpha: f32, x: &mut [f32]) -> (f32, f32) {
    if x.is_empty() {
        return (alpha, 0.0);
    }
    let mut alpha = alpha;
    let xnorm = dnrm2(x);
    if xnorm == 0.0 {
        return (alpha, 0.0);
    }
    let mut beta = -dlapy2(alpha, xnorm).copysign(alpha);
    let safmin = D_SAFMIN / D_EPS;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        while beta.abs() < safmin && knt < 20 {
            knt += 1;
            for xi in x.iter_mut() {
                *xi *= rsafmn;
            }
            beta *= rsafmn;
            alpha *= rsafmn;
        }
        let xnorm = dnrm2(x);
        beta = -dlapy2(alpha, xnorm).copysign(alpha);
    }
    let tau = (beta - alpha) / beta;
    let inv = 1.0 / (alpha - beta);
    for xi in x.iter_mut() {
        *xi *= inv;
    }
    for _ in 0..knt {
        beta *= safmin;
    }
    (beta, tau)
}

/// DLARTGP: scaled safe Givens rotation. Returns (cs, sn, r).
fn dlartgp(f: f32, g: f32) -> (f32, f32, f32) {
    let safmn2 = 2f32.powi(((D_SAFMIN / D_EPS).ln() / 2f32.ln() / 2.0) as i32);
    let safmx2 = 1.0 / safmn2;
    if g == 0.0 {
        return (1.0f32.copysign(f), 0.0, f.abs());
    }
    if f == 0.0 {
        return (0.0, 1.0f32.copysign(g), g.abs());
    }
    let (mut f1, mut g1) = (f, g);
    let scale = f1.abs().max(g1.abs());
    let (cs, sn, mut r);
    if scale >= safmx2 {
        let mut count = 0;
        let mut scale = scale;
        while scale >= safmx2 && count < 20 {
            count += 1;
            f1 *= safmn2;
            g1 *= safmn2;
            scale = f1.abs().max(g1.abs());
        }
        r = (f1 * f1 + g1 * g1).sqrt();
        cs = f1 / r;
        sn = g1 / r;
        for _ in 0..count {
            r *= safmx2;
        }
    } else if scale <= safmn2 {
        let mut count = 0;
        let mut scale = scale;
        while scale <= safmn2 {
            count += 1;
            f1 *= safmx2;
            g1 *= safmx2;
            scale = f1.abs().max(g1.abs());
        }
        r = (f1 * f1 + g1 * g1).sqrt();
        cs = f1 / r;
        sn = g1 / r;
        for _ in 0..count {
            r *= safmn2;
        }
    } else {
        r = (f1 * f1 + g1 * g1).sqrt();
        cs = f1 / r;
        sn = g1 / r;
    }
    if r < 0.0 {
        return (-cs, -sn, -r);
    }
    (cs, sn, r)
}

/// DLAS2: singular values of the 2x2 lower-triangular [f 0; g h].
/// Returns (ssmin, ssmax).
fn dlas2(f: f32, g: f32, h: f32) -> (f32, f32) {
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    let (ssmin, ssmax);
    if fhmn == 0.0 {
        ssmin = 0.0;
        ssmax = if fhmx == 0.0 {
            ga
        } else {
            fhmx.max(ga) * (1.0 + (fhmx.min(ga) / fhmx.max(ga)).powi(2)).sqrt()
        };
    } else if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = (ga / fhmx) * (ga / fhmx);
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        ssmin = fhmn * c;
        ssmax = fhmx / c;
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ssmin = (fhmn * fhmx) / ga;
            ssmax = ga;
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let c = 1.0 / ((1.0 + (as_ * au).powi(2)).sqrt() + (1.0 + (at * au).powi(2)).sqrt());
            let mut s = (fhmn * c) * au;
            s += s;
            ssmin = s;
            ssmax = ga / (c + c);
        }
    }
    (ssmin, ssmax)
}

/// DLASV2: SVD of the 2x2 [f g; 0 h]. Returns (ssmin, ssmax, snr, csr, snl, csl).
fn dlasv2(f: f32, g: f32, h: f32) -> (f32, f32, f32, f32, f32, f32) {
    let (mut ft, mut fa) = (f, f.abs());
    let (mut ht, mut ha) = (h, h.abs());
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = g.abs();
    let (clt, slt, crt, srt, ssmin, ssmax);
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        slt = 0.0;
        crt = 1.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        let mut clt_ = 1.0;
        let mut slt_ = 0.0;
        let mut crt_ = 1.0;
        let mut srt_ = 0.0;
        let mut ssmin_ = 0.0;
        let mut ssmax_ = 0.0;
        if ga > fa {
            pmax = 2;
            if fa / ga < D_EPS {
                gasmal = false;
                ssmax_ = ga;
                ssmin_ = if ha > 1.0 {
                    fa / (ga / ha)
                } else {
                    (fa / ga) * ha
                };
                clt_ = 1.0;
                slt_ = ht / gt;
                srt_ = 1.0;
                crt_ = ft / gt;
            }
        }
        if gasmal {
            let d = fa - ha;
            let l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 {
                m.abs()
            } else {
                (l * l + mm).sqrt()
            };
            let a = 0.5 * (s + r);
            ssmin_ = ha / a;
            ssmax_ = fa * a;
            let t = if mm == 0.0 {
                if l == 0.0 {
                    2.0f32.copysign(ft) * 1.0f32.copysign(gt)
                } else {
                    gt / d.copysign(ft) + m / t
                }
            } else {
                (m / (s + t) + m / (r + l)) * (1.0 + a)
            };
            let l = (t * t + 4.0).sqrt();
            crt_ = 2.0 / l;
            srt_ = t / l;
            clt_ = (crt_ + srt_ * m) / a;
            slt_ = (ht / ft) * srt_ / a;
        }
        clt = clt_;
        slt = slt_;
        crt = crt_;
        srt = srt_;
        ssmin = ssmin_;
        ssmax = ssmax_;
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };
    let tsign = match pmax {
        1 => 1.0f32.copysign(csr) * 1.0f32.copysign(csl) * 1.0f32.copysign(f),
        2 => 1.0f32.copysign(snr) * 1.0f32.copysign(csl) * 1.0f32.copysign(g),
        _ => 1.0f32.copysign(snr) * 1.0f32.copysign(snl) * 1.0f32.copysign(h),
    };
    let ssmax = ssmax.copysign(tsign);
    let ssmin = ssmin.copysign(tsign * 1.0f32.copysign(f) * 1.0f32.copysign(h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

/// DLARF 'Left': apply H = I - tau·v·vᵀ (v = [1, x]) to `a[r..16, c0..]`,
/// reference-BLAS accumulation order (row r0 first, then tail rows in order).
fn dlarf_left(r0: usize, x: &[f32], tau: f32, a: &mut [f32; 256], c0: usize) {
    if tau == 0.0 {
        return;
    }
    for c in c0..16 {
        let mut w = a[r0 * 16 + c]; // v[0] = 1 implicit
        for (i, &xi) in x.iter().enumerate() {
            w += xi * a[(r0 + 1 + i) * 16 + c];
        }
        let tw = tau * w;
        a[r0 * 16 + c] -= tw;
        for (i, &xi) in x.iter().enumerate() {
            a[(r0 + 1 + i) * 16 + c] -= xi * tw;
        }
    }
}

/// DLARF 'Right': apply H on the right to `a[r0.., c0..16]`; v = [1, x] sits
/// on row r0 starting at c0 (tail `x` is already stored there by the caller).
fn dlarf_right(r0: usize, c0: usize, x: &[f32], tau: f32, a: &mut [f32; 256]) {
    if tau == 0.0 {
        return;
    }
    for r in (r0 + 1)..16 {
        let mut w = a[r * 16 + c0];
        for (i, &xi) in x.iter().enumerate() {
            w += a[r * 16 + c0 + 1 + i] * xi;
        }
        let tw = tau * w;
        a[r * 16 + c0] -= tw;
        for (i, &xi) in x.iter().enumerate() {
            a[r * 16 + c0 + 1 + i] -= tw * xi;
        }
    }
}

/// DGEBD2 on a 16x16 row-major matrix. On return `a` carries the left
/// reflector tails below the diagonal (LAPACK storage), `d`/`e` the
/// bidiagonal and `tq` the left taus. Right reflectors are applied (they
/// shape the bidiagonal) but not accumulated: the head never reads V.
fn dgebd2(a: &mut [f32; 256], d: &mut [f32; 16], e: &mut [f32; 15], tq: &mut [f32; 16]) {
    for j in 0..16 {
        let mut x = [0f32; 15];
        for r in 0..(15usize.saturating_sub(j)) {
            x[r] = a[(j + 1 + r) * 16 + j];
        }
        let (beta, tau) = dlarfg(a[j * 16 + j], &mut x[..15usize.saturating_sub(j)]);
        for r in 0..(15usize.saturating_sub(j)) {
            a[(j + 1 + r) * 16 + j] = x[r];
        }
        a[j * 16 + j] = beta;
        tq[j] = tau;
        if j < 15 {
            dlarf_left(j, &x[..15 - j], tau, a, j + 1);
        }
        d[j] = beta;
        if j < 15 {
            let mut xr = [0f32; 14];
            for c in 0..(14usize.saturating_sub(j)) {
                xr[c] = a[j * 16 + j + 2 + c];
            }
            let (beta2, tau2) = dlarfg(a[j * 16 + j + 1], &mut xr[..14usize.saturating_sub(j)]);
            for c in 0..(14usize.saturating_sub(j)) {
                a[j * 16 + j + 2 + c] = xr[c];
            }
            a[j * 16 + j + 1] = beta2;
            dlarf_right(j, j + 1, &xr[..14 - j], tau2, a);
            e[j] = beta2;
        }
    }
}

/// DORG2R/dorgbr 'Q': build Q1 = H_0·...·H_15 in place, column-wise.
/// `a` holds the dgebd2 storage (reflector tails below the diagonal).
fn dorg2r_q(a: &mut [f32; 256], tq: &[f32; 16]) {
    for i in (0..16).rev() {
        let tau = tq[i];
        if i < 15 {
            a[i * 16 + i] = 1.0;
            for c in (i + 1)..16 {
                let mut w = a[i * 16 + c];
                for r in (i + 1)..16 {
                    w += a[r * 16 + i] * a[r * 16 + c];
                }
                a[i * 16 + c] -= tau * w;
                for r in (i + 1)..16 {
                    a[r * 16 + c] -= tau * a[r * 16 + i] * w;
                }
            }
        }
        for r in (i + 1)..16 {
            a[r * 16 + i] *= -tau;
        }
        a[i * 16 + i] = 1.0 - tau;
        for r in 0..i {
            a[r * 16 + i] = 0.0;
        }
    }
}

/// Rotate U column pair (i, j): LAPACK DROT convention.
fn drot_cols(u: &mut [f32; 256], i: usize, j: usize, c: f32, s: f32) {
    for r in 0..16 {
        let (a, b) = (u[r * 16 + i], u[r * 16 + j]);
        u[r * 16 + i] = c * a + s * b;
        u[r * 16 + j] = c * b - s * a;
    }
}

/// DLASR 'R','V': apply a sequence of plane rotations to U column pairs
/// (c0+i, c0+i+1). forward: i = 0..k; backward: i = k-1..0.
fn dlasr_cols(u: &mut [f32; 256], c0: usize, cs: &[f32], sn: &[f32], forward: bool) {
    if forward {
        for i in 0..cs.len() {
            drot_cols(u, c0 + i, c0 + i + 1, cs[i], sn[i]);
        }
    } else {
        for i in (0..cs.len()).rev() {
            drot_cols(u, c0 + i, c0 + i + 1, cs[i], sn[i]);
        }
    }
}

/// DBDSQR, uplo='U', nru=16 (U accumulated), ncvt=0 (no Vt). Faithful port
/// of the reference control flow; `tol < 0` branches are dropped because
/// the caller always runs relative-accuracy (tol > 0).
fn dbdsqr(d: &mut [f32; 16], e: &mut [f32; 15], u: &mut [f32; 256]) {
    const N: usize = 16;
    let nm1 = N - 1;
    let nm12 = 2 * nm1;
    let nm13 = 3 * nm1;
    let mut idir = 0;
    let tolmult = (10.0f32).max((100.0f32).min(D_EPS.powf(-0.125)));
    let tol = tolmult * D_EPS;
    let mut smax = 0.0f32;
    for i in 0..N {
        smax = smax.max(d[i].abs());
    }
    for i in 0..N - 1 {
        smax = smax.max(e[i].abs());
    }
    // LAPACK inits SMIN=0; every path that reads it assigns smin_c first.
    #[allow(unused_assignments)]
    let mut smin = 0.0f32;
    // relative-accuracy threshold (tol > 0 path)
    let mut sminoa = d[0].abs();
    let thresh = if sminoa == 0.0 {
        // LAPACK goes through the absacc path here; threshold is unused
        // below eps-size anyway for our data
        6.0 * (N as f32 * (N as f32 * D_SAFMIN))
    } else {
        let mut mu = sminoa;
        for i in 1..N {
            mu = d[i].abs() * (mu / (mu + e[i - 1].abs()));
            sminoa = sminoa.min(mu);
            if sminoa == 0.0 {
                break;
            }
        }
        sminoa /= (N as f32).sqrt();
        (tol * sminoa).max(6.0 * (N as f32 * (N as f32 * D_SAFMIN)))
    };
    let maxitdivn = 6 * N;
    let mut iterdivn = 0;
    let mut iter = -1i64;
    let (mut oldll, mut oldm) = (-1i64, -1i64);
    let mut m = N; // 1-based position: active block is ll..m
    let mut work = [0f32; 4 * N];

    loop {
        // label 60
        if m <= 1 {
            break;
        }
        if iter >= N as i64 {
            iter -= N as i64;
            iterdivn += 1;
            if iterdivn >= maxitdivn {
                // LAPACK returns INFO>0 here: the caller (DMM head)
                // treats nonconvergence like convergence — the score
                // uses whatever basis was accumulated.
                return;
            }
        }
        // find block
        let mut smax2 = d[m - 1].abs();
        let mut ll = 0usize;
        for lll in 1..m {
            let llx = m - lll; // 1-based LL -> e index LL-1
            let abss = d[llx - 1].abs();
            let abse = if llx - 1 < N - 1 {
                e[llx - 1].abs()
            } else {
                0.0
            };
            if abse <= thresh {
                ll = llx;
                e[llx - 1] = 0.0;
                break;
            }
            smax2 = smax2.max(abss).max(abse);
        }
        if ll == m - 1 {
            m -= 1;
            continue;
        }
        ll += 1; // active block rows ll..m (1-based); 0-based ll-1..m-1
        if ll == m - 1 {
            // 2x2 block at (m-2, m-1) 0-based
            let (m1, m2) = (m - 2, m - 1);
            let (ssmin, ssmax, _sinr, _cosr, sinl, cosl) = dlasv2(d[m1], e[m1], d[m2]);
            d[m1] = ssmax;
            e[m1] = 0.0;
            d[m2] = ssmin;
            drot_cols(u, m1, m2, cosl, sinl);
            m -= 2;
            continue;
        }
        // shift direction
        if ll as i64 > oldm || (m as i64) < oldll {
            idir = if d[ll - 1].abs() >= d[m - 1].abs() {
                1
            } else {
                2
            };
        }
        // convergence tests
        if idir == 1 {
            if e[m - 2].abs() <= tol.abs() * d[m - 1].abs() {
                e[m - 2] = 0.0;
                continue;
            }
            let mut mu = d[ll - 1].abs();
            let mut smin_c = mu;
            let mut done = false;
            for lll in ll..m {
                if e[lll - 1].abs() <= tol * mu {
                    e[lll - 1] = 0.0;
                    done = true;
                    break;
                }
                mu = d[lll].abs() * (mu / (mu + e[lll - 1].abs()));
                smin_c = smin_c.min(mu);
            }
            if done {
                continue;
            }
            smin = smin_c;
        } else {
            if e[ll - 1].abs() <= tol.abs() * d[ll - 1].abs() {
                e[ll - 1] = 0.0;
                continue;
            }
            let mut mu = d[m - 1].abs();
            let mut smin_c = mu;
            let mut done = false;
            // LAPACK: LLL = M-1 downto LL over e index lll-1
            for lll in (ll..m).rev() {
                if e[lll - 1].abs() <= tol * mu {
                    e[lll - 1] = 0.0;
                    done = true;
                    break;
                }
                mu = d[lll - 1].abs() * (mu / (mu + e[lll - 1].abs()));
                smin_c = smin_c.min(mu);
            }
            if done {
                continue;
            }
            smin = smin_c;
        }
        oldll = ll as i64;
        oldm = m as i64;
        // shift
        let mut shift = 0.0;
        if tol * (N as f32) * (smin / smax2) <= D_EPS.max(0.01 * tol) {
            // tiny cluster: shift stays 0 (LAPACK's zero-shift path)
        } else {
            if idir == 1 {
                let sll = d[ll - 1].abs();
                let (s2, _r) = dlas2(d[m - 2], e[m - 2], d[m - 1]);
                shift = s2;
                if sll > 0.0 && (shift / sll).powi(2) < D_EPS {
                    shift = 0.0;
                }
            } else {
                let sll = d[m - 1].abs();
                let (s2, _r) = dlas2(d[ll - 1], e[ll - 1], d[ll]);
                shift = s2;
                if sll > 0.0 && (shift / sll).powi(2) < D_EPS {
                    shift = 0.0;
                }
            }
        }
        iter += (m - ll) as i64;
        if shift == 0.0 {
            if idir == 1 {
                let mut cs = 1.0;
                let mut oldcs = 1.0;
                let mut oldsn = 1.0;
                for i in (ll - 1)..(m - 1) {
                    let (csn, sn, r) = dlartgp(d[i] * cs, e[i]);
                    cs = csn;
                    if i > ll - 1 {
                        e[i - 1] = oldsn * r;
                    }
                    let (ocs, osn, dd) = dlartgp(oldcs * r, d[i + 1] * sn);
                    oldcs = ocs;
                    oldsn = osn;
                    d[i] = dd;
                    work[i - (ll - 1)] = cs;
                    work[i - (ll - 1) + nm1] = sn;
                    work[i - (ll - 1) + nm12] = oldcs;
                    work[i - (ll - 1) + nm13] = oldsn;
                }
                let hh = d[m - 1] * cs;
                d[m - 1] = hh * oldcs;
                e[m - 2] = hh * oldsn;
                dlasr_cols(
                    u,
                    ll - 1,
                    &work[nm12..nm12 + (m - ll)],
                    &work[nm13..nm13 + (m - ll)],
                    true,
                );
                if e[m - 2].abs() <= thresh {
                    e[m - 2] = 0.0;
                }
            } else {
                let mut cs = 1.0;
                let mut oldcs = 1.0;
                let mut oldsn = 1.0;
                for i in (ll..m).rev() {
                    // i 0-based from m-1 down to ll
                    let (csn, sn, r) = dlartgp(d[i] * cs, e[i - 1]);
                    cs = csn;
                    if i < m - 1 {
                        e[i] = oldsn * r;
                    }
                    let (ocs, osn, dd) = dlartgp(oldcs * r, d[i - 1] * sn);
                    oldcs = ocs;
                    oldsn = osn;
                    d[i] = dd;
                    work[i - ll] = cs;
                    work[i - ll + nm1] = -sn;
                    work[i - ll + nm12] = oldcs;
                    work[i - ll + nm13] = -oldsn;
                }
                let hh = d[ll - 1] * cs;
                d[ll - 1] = hh * oldcs;
                e[ll - 1] = hh * oldsn;
                dlasr_cols(
                    u,
                    ll - 1,
                    &work[..m - ll],
                    &work[nm1..nm1 + (m - ll)],
                    false,
                );
                if e[ll - 1].abs() <= thresh {
                    e[ll - 1] = 0.0;
                }
            }
        } else if idir == 1 {
            let mut f =
                (d[ll - 1].abs() - shift) * (1.0f32.copysign(d[ll - 1]) + shift / d[ll - 1]);
            let mut g = e[ll - 1];
            for i in (ll - 1)..(m - 1) {
                let (cosr, sinr, r) = dlartgp(f, g);
                if i > ll - 1 {
                    e[i - 1] = r;
                }
                f = cosr * d[i] + sinr * e[i];
                e[i] = cosr * e[i] - sinr * d[i];
                g = sinr * d[i + 1];
                d[i + 1] *= cosr;
                let (cosl, sinl, r) = dlartgp(f, g);
                d[i] = r;
                f = cosl * e[i] + sinl * d[i + 1];
                d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                if i < m - 2 {
                    g = sinl * e[i + 1];
                    e[i + 1] *= cosl;
                }
                work[i - (ll - 1)] = cosr;
                work[i - (ll - 1) + nm1] = sinr;
                work[i - (ll - 1) + nm12] = cosl;
                work[i - (ll - 1) + nm13] = sinl;
            }
            e[m - 2] = f;
            dlasr_cols(
                u,
                ll - 1,
                &work[nm12..nm12 + (m - ll)],
                &work[nm13..nm13 + (m - ll)],
                true,
            );
            if e[m - 2].abs() <= thresh {
                e[m - 2] = 0.0;
            }
        } else {
            let mut f = (d[m - 1].abs() - shift) * (1.0f32.copysign(d[m - 1]) + shift / d[m - 1]);
            let mut g = e[m - 2];
            for i in (ll..m).rev() {
                let (cosr, sinr, r) = dlartgp(f, g);
                if i < m - 1 {
                    e[i] = r;
                }
                f = cosr * d[i] + sinr * e[i - 1];
                e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                g = sinr * d[i - 1];
                d[i - 1] *= cosr;
                let (cosl, sinl, r) = dlartgp(f, g);
                d[i] = r;
                f = cosl * e[i - 1] + sinl * d[i - 1];
                d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                if i > ll {
                    g = sinl * e[i - 2];
                    e[i - 2] *= cosl;
                }
                work[i - ll] = cosr;
                work[i - ll + nm1] = -sinr;
                work[i - ll + nm12] = cosl;
                work[i - ll + nm13] = -sinl;
            }
            e[ll - 1] = f;
            if e[ll - 1].abs() <= thresh {
                e[ll - 1] = 0.0;
            }
            dlasr_cols(
                u,
                ll - 1,
                &work[..m - ll],
                &work[nm1..nm1 + (m - ll)],
                false,
            );
        }
    }
    // label 160: make positive (Vt sign-flip dropped: no Vt)
    for i in 0..N {
        if d[i] < 0.0 {
            d[i] = -d[i];
        }
    }
    // dbdsqr's own sort: move min to end, LE ties -> last occurrence
    for i in 1..N {
        let tgt = N - i;
        let mut isub = 0usize;
        let mut smin_ = d[0];
        for j in 1..=tgt {
            if d[j] <= smin_ {
                isub = j;
                smin_ = d[j];
            }
        }
        if isub != tgt {
            d[isub] = d[tgt];
            d[tgt] = smin_;
            for r in 0..16 {
                u.swap(r * 16 + isub, r * 16 + tgt);
            }
        }
    }
}

/// DLASDQ: dbdsqr + ascending selection sort on (d, U cols).
fn dlasdq(d: &mut [f32; 16], e: &mut [f32; 15], u: &mut [f32; 256]) {
    dbdsqr(d, e, u);
    for i in 0..16 {
        let mut isub = i;
        let mut smin_ = d[i];
        for j in (i + 1)..16 {
            if d[j] < smin_ {
                isub = j;
                smin_ = d[j];
            }
        }
        if isub != i {
            d[isub] = d[i];
            d[i] = smin_;
            for r in 0..16 {
                u.swap(r * 16 + isub, r * 16 + i);
            }
        }
    }
}

/// SVD of a 16×16 patch matrix via the LAPACK dgebd2/dorg2r/dbdsqr
/// pipeline (the `gesdd` path for n=16). `m` is the patch in natural
/// (y,x) layout; upstream feeds `Mᵀ` to `torch.svd`, so this transposes
/// first and returns `u` of the transposed matrix — the patch's own V.
pub(crate) fn svd16(m: &[f32; 256]) -> ([f32; 256], [f32; 16]) {
    let mut a = [0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            a[i * 16 + j] = m[j * 16 + i]; // P = Mᵀ
        }
    }
    let mut d = [0f32; 16];
    let mut e = [0f32; 15];
    let mut tq = [0f32; 16];
    dgebd2(&mut a, &mut d, &mut e, &mut tq);
    // a now holds reflector tails below diag; build Q1 in place
    dorg2r_q(&mut a, &tq); // a <- Q1
    // dbdsdc scale
    let orgnrm = d
        .iter()
        .fold(0.0f32, |m, &x| m.max(x.abs()))
        .max(e.iter().fold(0.0f32, |m, &x| m.max(x.abs())));
    if orgnrm == 0.0 {
        let mut s = [0f32; 16];
        let u = a; // Q1 (identity for zero matrix)
        s.copy_from_slice(&d);
        return (u, s);
    }
    let mut ds = [0f32; 16];
    let mut es = [0f32; 15];
    for i in 0..16 {
        ds[i] = d[i] / orgnrm;
    }
    for i in 0..15 {
        es[i] = e[i] / orgnrm;
    }
    let mut ub = [0f32; 256];
    for i in 0..16 {
        ub[i * 16 + i] = 1.0;
    }
    dlasdq(&mut ds, &mut es, &mut ub);
    // dbdsdc's descending selection sort with U column swaps
    for i in 0..16 {
        let mut kk = i;
        let mut p = ds[i];
        for j in (i + 1)..16 {
            if ds[j] > p {
                kk = j;
                p = ds[j];
            }
        }
        if kk != i {
            ds[kk] = ds[i];
            ds[i] = p;
            for r in 0..16 {
                ub.swap(r * 16 + kk, r * 16 + i);
            }
        }
    }
    // U = Q1 @ Ub ; s = ds * orgnrm
    let mut u = [0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = 0.0;
            for r in 0..16 {
                acc += a[i * 16 + r] * ub[r * 16 + j];
            }
            u[i * 16 + j] = acc;
        }
    }
    let mut s = [0f32; 16];
    for i in 0..16 {
        s[i] = ds[i] * orgnrm;
    }
    (u, s)
}
