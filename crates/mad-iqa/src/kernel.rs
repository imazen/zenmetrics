//! MAD pipeline internals — `hi_index` (CSF-weighted luminance,
//! blocky `ical_std` masking statistics, contrast-threshold mask ×
//! local MSE) and `lo_index` (5×4 log-Gabor bank, blocky `ical_stat`
//! moment maps, weighted accumulation), then the JEI geometric
//! combine.
//!
//! Provenance: `hi_index.m` / `lo_index.m` / `ical_std.c` /
//! `ical_stat.c` from the authors' STMAD_2011 release (archived in
//! `Netflix/vmaf`); the single-image combine is the JEI 2010 paper
//! formula (`validation/README.md` documents the provenance — the
//! original `MAD_index` bundle was unreachable, so `hi_index` and
//! `lo_index` are each golden-verified independently and the combine
//! follows the paper and every published port).
//!
//! Structure: `f32` planes; one-time grids and all block-stat
//! numerators / final norms accumulate `f64`. The bulk elementwise
//! stages (error plane, spectrum diff, `mp_lo` accumulation, the
//! separable lmse passes) go through `score_body!`/`$V` so every
//! tier produces bit-identical output; the blocky `ical_*` maps,
//! CSF/Gabor application, and `|EO|`/`msk` scalar stages are shared
//! code with fixed per-element order.

use alloc::vec;
use alloc::vec::Vec;

use crate::fft::{self, Complex};

/// `k = 0.02874` — luminance transform gain.
const K_LUM: f64 = 0.02874;
/// `G = 0.5` — luminance threshold on `m1_1` (dark blocks are masked
/// out of the contrast map entirely).
const G_LUM: f64 = 0.5;
/// `Ci_thrsh = -5` — contrast where the detection slope starts.
const CI_THR: f64 = -5.0;
/// `Cd_thrsh = -5` — saturated detection threshold.
const CD_THR: f64 = -5.0;
/// `BSIZE = 16` — block-stat window and lmse kernel size.
const B: usize = 16;
/// Scale weights `s = [0.5 0.75 1 5 6]/13.25` indexed by `gb_i`.
const LO_WEIGHTS: [f32; 5] = [
    0.5 / 13.25,
    0.75 / 13.25,
    1.0 / 13.25,
    5.0 / 13.25,
    6.0 / 13.25,
];

/// `rayon::join` under `parallel`, sequential otherwise.
fn maybe_join<A, B>(fa: impl FnOnce() -> A + Send, fb: impl FnOnce() -> B + Send) -> (A, B)
where
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    {
        rayon::join(fa, fb)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (fa(), fb())
    }
}

/// Arguments to one MAD run, bundled for `attuned!`.
#[doc(hidden)]
pub struct Args<'a> {
    /// Reference plane, packed `w·h`.
    pub im1: &'a [f32],
    /// Distorted plane, packed `w·h`.
    pub im2: &'a [f32],
    /// Width × height.
    pub w: usize,
    /// Height.
    pub h: usize,
}

// ============================ shared helpers ============================

/// `make_csf(M, N, 32)'` — the centered CSF grid *after* the
/// reference's transpose: element (r,c) uses real comp
/// `(r − M/2 + 0.5)·64/N`, imag comp `(c − N/2 + 0.5)·64/N`.
/// Computed in f64, stored f32.
fn make_csf(m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    let csf_px = |r: usize, c: usize| {
        let x = (r as f64 - m as f64 / 2.0 + 0.5) * 64.0 / n as f64;
        let y = (c as f64 - n as f64 / 2.0 + 0.5) * 64.0 / n as f64;
        let mut rf = libm::hypot(x, y);
        let s = 0.15 * libm::cos(4.0 * libm::atan2(y, x)) + 0.85;
        rf /= s;
        if rf < 7.8909 {
            0.9809
        } else {
            (2.6 * (0.0192 + 0.114 * rf) * libm::exp(-libm::pow(0.114 * rf, 1.1))) as f32
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
            for (c, v) in row.iter_mut().enumerate() {
                *v = csf_px(r, c);
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    for r in 0..m {
        for c in 0..n {
            out[r * n + c] = csf_px(r, c);
        }
    }
    out
}

/// `real(ifft2(ifftshift(fftshift(fft2(src)) .* csf')))` — the csf
/// grid is on centered coordinates; the shifted-position remap
/// `(u+M/2)%M` applies it directly to the unshifted spectrum.
fn csf_apply(
    src: &[f32],
    csf: &[f32],
    m: usize,
    n: usize,
    wplan: &fft::Plan,
    hplan: &fft::Plan,
) -> Vec<f32> {
    let mut buf: Vec<Complex> = src.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft::fft2_planned(&mut buf, n, m, wplan, hplan);
    // The shifted-position remap materialises as a weight plane; the
    // multiply then rides the inverse transform's transpose-in
    // (`conj(v·w)` — identical arithmetic to multiply-then-conjugate),
    // so no separate pass over `buf` is needed.
    let mut wmap = vec![0.0f32; m * n];
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        wmap.par_chunks_mut(n).enumerate().for_each(|(u, row)| {
            let su = (u + m / 2) % m;
            for (v, w) in row.iter_mut().enumerate() {
                *w = csf[su * n + (v + n / 2) % n];
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    for u in 0..m {
        let su = (u + m / 2) % m;
        for v in 0..n {
            wmap[u * n + v] = csf[su * n + (v + n / 2) % n];
        }
    }
    fft::ifft2_planned_mul(&mut buf, n, m, wplan, hplan, &wmap);
    buf.iter().map(|c| c.re).collect()
}

/// j-block strip size for the `ical` passes under `parallel`: each strip
/// is `g` block-row steps (4·g contiguous output rows), targeting ~4
/// strips per worker for load balance.
#[cfg(feature = "parallel")]
fn jstrip_g(m: usize) -> usize {
    let iters = m.saturating_sub(B - 1).div_ceil(4).max(1);
    iters.div_ceil(rayon::current_num_threads() * 4).max(1)
}

/// Tile-grid dims: the stride-4 block starts of B-sized windows, as
/// `(tile_rows, tile_cols)`. `(0, 0)` when no block fits.
fn tile_dims(m: usize, n: usize) -> (usize, usize) {
    if m >= B && n >= B {
        ((m - B) / 4 + 1, (n - B) / 4 + 1)
    } else {
        (0, 0)
    }
}

/// `ical_std(dst−ref, ref)` → `(std_2, std_1, m1_1)` — verbatim port
/// of the C mex: stride-4 16×16 windows, blocky 4×4 tiles; `std_1` is
/// the 8×8 sample-std of ref min-pooled over `{0,5}` offsets.
/// Numerators accumulate f64 (the C code is `double`).
///
/// The C mex stores each statistic into a whole 4×4 tile of a full-res
/// map; every consumer only needs the per-tile value, so these are
/// `mt × nt` tile-resolution maps (`tile[jt * nt + it]` = pixel block
/// starting at `(4jt, 4it)`). Positions outside the tile grid would
/// have read 0 from the full-res maps, and callers honour that.
fn ical_std(diff: &[f32], reff: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mt, nt) = tile_dims(m, n);
    let (mut std2, mut std1, mut m1) = (
        vec![0.0f32; mt * nt],
        vec![0.0f32; mt * nt],
        vec![0.0f32; mt * nt],
    );
    let mut tmp = vec![0.0f32; mt * nt];
    if mt == 0 {
        return (std2, std1, m1);
    }
    // Pass 1: 16×16 blocks at stride 4. (it,jt) blocks are independent —
    // each writes only its own tile, so jt-strips are safe to run on
    // threads (identical per-element values either way).
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let g = jstrip_g(m);
        std2.par_chunks_mut(g * nt)
            .zip(m1.par_chunks_mut(g * nt))
            .enumerate()
            .for_each(|(ci, (s2s, m1s))| {
                let ts = ci * g;
                for jt in ts..(ts + g).min(mt) {
                    let j = 4 * jt;
                    for it in 0..nt {
                        let i = 4 * it;
                        let (mut sx, mut sy) = (0.0f64, 0.0f64);
                        for r in j..j + B {
                            for c in i..i + B {
                                sx += diff[r * n + c] as f64;
                                sy += reff[r * n + c] as f64;
                            }
                        }
                        let (mean, mean2) = (sx / 256.0, sy / 256.0);
                        let mut s2 = 0.0f64;
                        for r in j..j + B {
                            for c in i..i + B {
                                let d = diff[r * n + c] as f64 - mean;
                                s2 += d * d;
                            }
                        }
                        let o = (jt - ts) * nt + it;
                        s2s[o] = libm::sqrt(s2 / 255.0) as f32;
                        m1s[o] = mean2 as f32;
                    }
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for it in 0..nt {
        let i = 4 * it;
        for jt in 0..mt {
            let j = 4 * jt;
            let (mut sx, mut sy) = (0.0f64, 0.0f64);
            for r in j..j + B {
                for c in i..i + B {
                    sx += diff[r * n + c] as f64;
                    sy += reff[r * n + c] as f64;
                }
            }
            let (mean, mean2) = (sx / 256.0, sy / 256.0);
            let mut s2 = 0.0f64;
            for r in j..j + B {
                for c in i..i + B {
                    let d = diff[r * n + c] as f64 - mean;
                    s2 += d * d;
                }
            }
            let o = jt * nt + it;
            std2[o] = libm::sqrt(s2 / 255.0) as f32;
            m1[o] = mean2 as f32;
        }
    }
    // Pass 2: 8×8 blocks at the same stride-4 starts.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let g = jstrip_g(m);
        tmp.par_chunks_mut(g * nt)
            .zip(std1.par_chunks_mut(g * nt))
            .enumerate()
            .for_each(|(ci, (tps, s1s))| {
                let ts = ci * g;
                for jt in ts..(ts + g).min(mt) {
                    let j = 4 * jt;
                    for it in 0..nt {
                        let i = 4 * it;
                        let mut sy = 0.0f64;
                        for r in j..j + 8 {
                            for c in i..i + 8 {
                                sy += reff[r * n + c] as f64;
                            }
                        }
                        let mean = sy / 64.0;
                        let mut s2 = 0.0f64;
                        for r in j..j + 8 {
                            for c in i..i + 8 {
                                let d = reff[r * n + c] as f64 - mean;
                                s2 += d * d;
                            }
                        }
                        let o = (jt - ts) * nt + it;
                        let stdev = libm::sqrt(s2 / 63.0) as f32;
                        tps[o] = stdev;
                        s1s[o] = stdev;
                    }
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for it in 0..nt {
        let i = 4 * it;
        for jt in 0..mt {
            let j = 4 * jt;
            let mut sy = 0.0f64;
            for r in j..j + 8 {
                for c in i..i + 8 {
                    sy += reff[r * n + c] as f64;
                }
            }
            let mean = sy / 64.0;
            let mut s2 = 0.0f64;
            for r in j..j + 8 {
                for c in i..i + 8 {
                    let d = reff[r * n + c] as f64 - mean;
                    s2 += d * d;
                }
            }
            let stdev = libm::sqrt(s2 / 63.0) as f32;
            tmp[jt * nt + it] = stdev;
            std1[jt * nt + it] = stdev;
        }
    }
    // Pass 3: min-pool over {0,5} pixel offsets = {0,+1} tile offsets
    // (block starts are ≡0 mod 4, so pixel j+5 lands in tile jt+1).
    // The bounds checks stay pixel-level, exactly as the reference
    // evaluates them. Tile reads reach one tile row past a strip's
    // edge but are immutable by then (pass 2 joined first).
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let g = jstrip_g(m);
        let tmp_ref: &[f32] = &tmp;
        std1.par_chunks_mut(g * nt)
            .enumerate()
            .for_each(|(ci, s1s)| {
                let ts = ci * g;
                for jt in ts..(ts + g).min(mt) {
                    let j = 4 * jt;
                    for it in 0..nt {
                        let i = 4 * it;
                        let mut mn = tmp_ref[jt * nt + it];
                        if i + 5 < n - 15 && j < m - 15 && mn > tmp_ref[jt * nt + it + 1] {
                            mn = tmp_ref[jt * nt + it + 1];
                        }
                        if i < n - 15 && j + 5 < m - 15 && mn > tmp_ref[(jt + 1) * nt + it] {
                            mn = tmp_ref[(jt + 1) * nt + it];
                        }
                        if i + 5 < n - 15 && j + 5 < m - 15 && mn > tmp_ref[(jt + 1) * nt + it + 1]
                        {
                            mn = tmp_ref[(jt + 1) * nt + it + 1];
                        }
                        s1s[(jt - ts) * nt + it] = mn;
                    }
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for it in 0..nt {
        let i = 4 * it;
        for jt in 0..mt {
            let j = 4 * jt;
            let mut mn = tmp[jt * nt + it];
            if i + 5 < n - 15 && j < m - 15 && mn > tmp[jt * nt + it + 1] {
                mn = tmp[jt * nt + it + 1];
            }
            if i < n - 15 && j + 5 < m - 15 && mn > tmp[(jt + 1) * nt + it] {
                mn = tmp[(jt + 1) * nt + it];
            }
            if i + 5 < n - 15 && j + 5 < m - 15 && mn > tmp[(jt + 1) * nt + it + 1] {
                mn = tmp[(jt + 1) * nt + it + 1];
            }
            std1[jt * nt + it] = mn;
        }
    }
    (std2, std1, m1)
}

/// `ical_stat(x)` → `(std, skew, kurt)` — verbatim port of the C
/// mex: stride-4 16×16 windows, blocky 4×4 tiles; skew/kurtosis
/// normalised by population moments (`stmp`); flat → 0.
fn ical_stat(x: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mt, nt) = tile_dims(m, n);
    let (mut std_o, mut skw, mut krt) = (
        vec![0.0f32; mt * nt],
        vec![0.0f32; mt * nt],
        vec![0.0f32; mt * nt],
    );
    if mt == 0 {
        return (std_o, skw, krt);
    }
    let block = |j: usize, i: usize, so: &mut [f32], sk: &mut [f32], kr: &mut [f32], o: usize| {
        let mut sx = 0.0f64;
        for r in j..j + B {
            for c in i..i + B {
                sx += x[r * n + c] as f64;
            }
        }
        let mean = sx / 256.0;
        let (mut s2, mut s3, mut s4) = (0.0f64, 0.0f64, 0.0f64);
        for r in j..j + B {
            for c in i..i + B {
                let d = x[r * n + c] as f64 - mean;
                let d2 = d * d;
                s2 += d2;
                s3 += d2 * d;
                s4 += d2 * d2;
            }
        }
        so[o] = libm::sqrt(s2 / 255.0) as f32;
        let stmp = libm::sqrt(s2 / 256.0);
        if stmp != 0.0 {
            sk[o] = ((s3 / 256.0) / (stmp * stmp * stmp)) as f32;
            kr[o] = ((s4 / 256.0) / (stmp * stmp * stmp * stmp)) as f32;
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let g = jstrip_g(m);
        std_o
            .par_chunks_mut(g * nt)
            .zip(skw.par_chunks_mut(g * nt))
            .zip(krt.par_chunks_mut(g * nt))
            .enumerate()
            .for_each(|(ci, ((so, sk), kr))| {
                let ts = ci * g;
                for jt in ts..(ts + g).min(mt) {
                    for it in 0..nt {
                        block(4 * jt, 4 * it, so, sk, kr, (jt - ts) * nt + it);
                    }
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for it in 0..nt {
        for jt in 0..mt {
            block(4 * jt, 4 * it, &mut std_o, &mut skw, &mut krt, jt * nt + it);
        }
    }
    (std_o, skw, krt)
}

/// Whole-point symmetric index map (`idx(−k)=k−1`, `idx(n+k)=n−1−k`),
/// periodic for arbitrarily large excursions.
fn sym(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    let period = 2 * n;
    i = i.rem_euclid(period);
    usize::try_from(if i >= n { period - 1 - i } else { i }).unwrap()
}

/// The centered log-Gabor radial components (`logGabors{1..5}`) and
/// angular spreads (`spread{1..4}`), computed in f64, stored f32.
/// `logGabors[s]` has the center element zeroed at the reference's
/// `round(rows/2+1), round(cols/2+1)` fudge index.
fn gabor_grids(m: usize, n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    // round(x/2+1) 1-based → 0-based
    let (rc, cc) = (m.div_ceil(2), n.div_ceil(2));
    let mut radius = vec![0.0f64; m * n];
    let (mut sinth, mut costh) = (vec![0.0f64; m * n], vec![0.0f64; m * n]);
    for r in 0..m {
        for c in 0..n {
            let x = (c as f64 - n as f64 / 2.0) / (n as f64 / 2.0);
            let y = (r as f64 - m as f64 / 2.0) / (m as f64 / 2.0);
            let mut rad = libm::hypot(x, y);
            if r == rc && c == cc {
                rad = 1.0; // radius fudge
            }
            radius[r * n + c] = libm::log(rad);
            let th = libm::atan2(-y, x);
            sinth[r * n + c] = libm::sin(th);
            costh[r * n + c] = libm::cos(th);
        }
    }
    let sigma_onf = 0.55f64;
    let theta_sigma = core::f64::consts::PI / 4.0 / 1.5;
    let mut logs = Vec::with_capacity(5);
    for s in 0..5 {
        let rfo = (1.0 / (3.0 * libm::pow(3.0, s as f64))) / 0.5;
        let lrfo = libm::log(rfo);
        let den = -2.0 * libm::log(sigma_onf) * libm::log(sigma_onf);
        let mut lg = vec![0.0f32; m * n];
        for i in 0..m * n {
            let d = radius[i] - lrfo;
            lg[i] = libm::exp(d * d / den) as f32;
        }
        lg[rc * n + cc] = 0.0; // undo the radius fudge
        logs.push(lg);
    }
    let mut spreads = Vec::with_capacity(4);
    for o in 0..4 {
        let angl = o as f64 * core::f64::consts::PI / 4.0;
        let (ca, sa) = (libm::cos(angl), libm::sin(angl));
        let mut sp = vec![0.0f32; m * n];
        for i in 0..m * n {
            let ds = sinth[i] * ca - costh[i] * sa;
            let dc = costh[i] * ca + sinth[i] * sa;
            let dth = libm::atan2(ds, dc).abs();
            sp[i] = libm::exp(-dth * dth / (2.0 * theta_sigma * theta_sigma)) as f32;
        }
        spreads.push(sp);
    }
    (logs, spreads)
}

/// `ifft2(imagefft .* fftshift(logGabor·spread))` — returns the
/// complex EO plane. `fftshift` maps centered index `i` to unshifted
/// `(i + ceil(d/2)) % d` — for odd dims this is *not* the same offset
/// as `hi_index`'s `ifftshift(fftshift(X)·csf)` (floor), so the two
/// remaps differ by one element on odd axes.
fn gabor_apply(
    imagefft: &[Complex],
    lg: &[f32],
    sp: &[f32],
    m: usize,
    n: usize,
    wplan: &fft::Plan,
    hplan: &fft::Plan,
) -> Vec<Complex> {
    let mut buf = imagefft.to_vec();
    // Same fusion as `csf_apply`: the `fftshift`ed `lg·sp` product is a
    // real weight plane that rides the inverse transform's transpose-in.
    let mut wmap = vec![0.0f32; m * n];
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        wmap.par_chunks_mut(n).enumerate().for_each(|(u, row)| {
            let su = (u + m.div_ceil(2)) % m;
            for (v, w) in row.iter_mut().enumerate() {
                let sv = (v + n.div_ceil(2)) % n;
                *w = lg[su * n + sv] * sp[su * n + sv];
            }
        });
    }
    #[cfg(not(feature = "parallel"))]
    for u in 0..m {
        let su = (u + m.div_ceil(2)) % m;
        for v in 0..n {
            let sv = (v + n.div_ceil(2)) % n;
            wmap[u * n + v] = lg[su * n + sv] * sp[su * n + sv];
        }
    }
    fft::ifft2_planned_mul(&mut buf, n, m, wplan, hplan, &wmap);
    buf
}

/// `|EO|` element magnitudes — scalar `hypotf`, matching Octave
/// `abs()` to the ulp on every tier.
fn eo_magnitude(eo: &[Complex]) -> Vec<f32> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        eo.par_iter().map(|c| libm::hypotf(c.re, c.im)).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        eo.iter().map(|c| libm::hypotf(c.re, c.im)).collect()
    }
}

/// The forward spectrum of a plane.
fn spectrum(src: &[f32], m: usize, n: usize, wplan: &fft::Plan, hplan: &fft::Plan) -> Vec<Complex> {
    let mut buf: Vec<Complex> = src.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft::fft2_planned(&mut buf, n, m, wplan, hplan);
    buf
}

/// `norm(mp(17:end-17, 17:end-17)(:), 2) / sqrt(numel)` — the
/// reference's edge kill (0-based rows/cols `16 ..= dim−18`).
fn interior_norm(mp: &[f32], m: usize, n: usize) -> f64 {
    if m < 34 || n < 34 {
        return f64::NAN;
    }
    let mut acc = 0.0f64;
    let mut count = 0usize;
    for r in 16..m - 17 {
        for c in 16..n - 17 {
            let v = mp[r * n + c] as f64;
            acc += v * v;
            count += 1;
        }
    }
    libm::sqrt(acc) / libm::sqrt(count as f64)
}

// ============================ tiered core ================================

/// The full pipeline for one `(im1, im2, w, h)` → `(hi, lo)`.
/// `$V` vectorizes the bulk elementwise/stencil stages with fixed
/// per-element order; the `ical_*` block stats, `msk` chain, FFT
/// wrappers, and `|EO|` magnitudes are shared scalar code, so every
/// tier produces bit-identical output.
macro_rules! score_body {
    ($token:ident, $V:ident, $LANES:literal, $a:ident) => {{
        let (m, n) = ($a.h, $a.w); // MATLAB [M N] = size(im) → rows, cols
        let (wplan, hplan) = (fft::Plan::new(n), fft::Plan::new(m));

        // ---------- hi_index ----------
        // Luminance transform `k·x^(2.2/3)` — scalar pow per element
        // (integer-input LUT in the reference is the same math).
        let mut lum_r = vec![0.0f32; m * n];
        let mut lum_d = vec![0.0f32; m * n];
        // Integral inputs (every u8-derived plane — the `mad_rgb8` path and
        // the C reference's own LUT) hit a 256-entry table built with the
        // same `libm::pow` — identical bits, ~100x cheaper per element.
        let lum_lut: [f32; 256] =
            core::array::from_fn(|i| (K_LUM * libm::pow(i as f64, 2.2 / 3.0)) as f32);
        let lum_px = |v: f32| -> f32 {
            if (0.0..256.0).contains(&v) && v == v.trunc() {
                lum_lut[v as usize]
            } else {
                (K_LUM * libm::pow(v as f64, 2.2 / 3.0)) as f32
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let n_px = m * n;
            lum_r
                .par_chunks_mut(1 << 14)
                .zip($a.im1[..n_px].par_chunks(1 << 14))
                .for_each(|(lc, ic)| {
                    for (o, &v) in lc.iter_mut().zip(ic) {
                        *o = lum_px(v);
                    }
                });
            lum_d
                .par_chunks_mut(1 << 14)
                .zip($a.im2[..n_px].par_chunks(1 << 14))
                .for_each(|(lc, ic)| {
                    for (o, &v) in lc.iter_mut().zip(ic) {
                        *o = lum_px(v);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        for i in 0..m * n {
            lum_r[i] = lum_px($a.im1[i]);
            lum_d[i] = lum_px($a.im2[i]);
        }
        let csf = make_csf(m, n);
        let (ref_l, dst_l) = maybe_join(
            || csf_apply(&lum_r, &csf, m, n, &wplan, &hplan),
            || csf_apply(&lum_d, &csf, m, n, &wplan, &hplan),
        );
        // diff = dst_l − ref_l (elementwise).
        let mut diff = vec![0.0f32; m * n];
        {
            let mut i = 0usize;
            while i + $LANES <= m * n {
                let dv = <$V>::from_array_t($token, core::array::from_fn(|k| dst_l[i + k]))
                    - <$V>::from_array_t($token, core::array::from_fn(|k| ref_l[i + k]));
                diff[i..i + $LANES].copy_from_slice(&dv.to_array());
                i += $LANES;
            }
            while i < m * n {
                diff[i] = dst_l[i] - ref_l[i];
                i += 1;
            }
        }
        let (std2, std1, m1) = ical_std(&diff, &ref_l, m, n);

        // lmse = imfilter(err², box16/256, 'symmetric','same','conv')
        // — separable passes over a whole-point-symmetric padded copy
        // (pad = 8: the {i−7..i+8} window needs 7 left / 8 right).
        let lm = {
            let mut e2 = vec![0.0f32; m * n];
            {
                let mut i = 0usize;
                while i + $LANES <= m * n {
                    let dv = <$V>::from_array_t($token, core::array::from_fn(|k| $a.im1[i + k]))
                        - <$V>::from_array_t($token, core::array::from_fn(|k| $a.im2[i + k]));
                    let sq = dv * dv;
                    e2[i..i + $LANES].copy_from_slice(&sq.to_array());
                    i += $LANES;
                }
                while i < m * n {
                    let d = $a.im1[i] - $a.im2[i];
                    e2[i] = d * d;
                    i += 1;
                }
            }
            // Padded copy: (m+16)×(n+16), sym-indexed.
            let (pw, ph) = (n + 2 * 8, m + 2 * 8);
            let mut pad = vec![0.0f32; pw * ph];
            for y in 0..ph {
                let sy = sym(y as isize - 8, m);
                for x in 0..pw {
                    pad[y * pw + x] = e2[sy * n + sym(x as isize - 8, n)];
                }
            }
            // Horizontal box over padded rows: tmp (ph)×n. The
            // 'same' window {x−7..x+8} maps to padded {x+1..x+16}.
            let mut tmp = vec![0.0f32; ph * n];
            for y in 0..ph {
                let srow = &pad[y * pw..y * pw + pw];
                let drow = &mut tmp[y * n..y * n + n];
                let mut x = 0usize;
                while x + $LANES <= n {
                    let mut acc = <$V>::splat_t($token, 0.0);
                    for u in 0..B {
                        let xv = <$V>::from_array_t(
                            $token,
                            core::array::from_fn(|k| srow[x + 1 + u + k]),
                        );
                        acc = acc + xv;
                    }
                    let outv = acc * <$V>::splat_t($token, 1.0 / 16.0);
                    drow[x..x + $LANES].copy_from_slice(&outv.to_array());
                    x += $LANES;
                }
                while x < n {
                    let mut acc = 0.0f64;
                    for u in 0..B {
                        acc += srow[x + 1 + u] as f64;
                    }
                    drow[x] = (acc / 16.0) as f32;
                    x += 1;
                }
            }
            // Vertical box: out[y][x] = Σ tmp[y+1+u][x]/16.
            let mut out = vec![0.0f32; m * n];
            for y in 0..m {
                let drow = &mut out[y * n..y * n + n];
                let mut x = 0usize;
                while x + $LANES <= n {
                    let mut acc = <$V>::splat_t($token, 0.0);
                    for u in 0..B {
                        let xv = <$V>::from_array_t(
                            $token,
                            core::array::from_fn(|k| tmp[(y + 1 + u) * n + x + k]),
                        );
                        acc = acc + xv;
                    }
                    let outv = acc * <$V>::splat_t($token, 1.0 / 16.0);
                    drow[x..x + $LANES].copy_from_slice(&outv.to_array());
                    x += $LANES;
                }
                while x < n {
                    let mut acc = 0.0f64;
                    for u in 0..B {
                        acc += tmp[(y + 1 + u) * n + x] as f64;
                    }
                    drow[x] = (acc / 16.0) as f32;
                    x += 1;
                }
            }
            out
        };

        // msk (scalar — per-element logs and thresholds) × lmse over
        // covered tiles; outside coverage m1 = std_1 = 0, which the
        // reference's own NaN/−inf path maps to msk = 0. All three stat
        // maps are tile-uniform, so the mask math runs once per tile —
        // identical f64 inputs, identical result.
        let mut mp_hi = vec![0.0f32; m * n];
        let (mt, nt) = tile_dims(m, n);
        for jt in 0..mt {
            for it in 0..nt {
                let t = jt * nt + it;
                let mu = m1[t] as f64;
                let s1 = std1[t] as f64;
                let s2 = std2[t] as f64;
                let ci_ref = libm::log(s1 / mu);
                let ci_dst = if m1[t] < G_LUM as f32 {
                    f64::NEG_INFINITY
                } else {
                    libm::log(s2 / mu)
                };
                let msk = if ci_ref > CI_THR && ci_dst > (ci_ref - CI_THR + CD_THR) {
                    ci_dst - (ci_ref - CI_THR + CD_THR)
                } else if ci_ref <= CI_THR && ci_dst > CD_THR {
                    ci_dst - CD_THR
                } else {
                    0.0
                } as f32;
                for r in 4 * jt..4 * jt + 4 {
                    for c in 4 * it..4 * it + 4 {
                        let idx = r * n + c;
                        mp_hi[idx] = msk * lm[idx];
                    }
                }
            }
        }

        // ---------- lo_index ----------
        let (logs, spreads) = gabor_grids(m, n);
        let (xfft_r, xfft_d) = maybe_join(
            || spectrum(&$a.im1, m, n, &wplan, &hplan),
            || spectrum(&$a.im2, m, n, &wplan, &hplan),
        );
        // mp += s·(|Δstd| + 2|Δskw| + |Δkrt|) over the whole map, but the
        // stat maps are tile-uniform so the 20-iteration accumulation
        // runs once per tile — the same f32 adds in the same order —
        // and the result expands to pixels in a single pass.
        let mut mp_lo_t = vec![0.0f32; mt * nt];
        let two = <$V>::splat_t($token, 2.0);
        for s in 0..5 {
            for o in 0..4 {
                let (eo_r, eo_d) = maybe_join(
                    || gabor_apply(&xfft_r, &logs[s], &spreads[o], m, n, &wplan, &hplan),
                    || gabor_apply(&xfft_d, &logs[s], &spreads[o], m, n, &wplan, &hplan),
                );
                let (mag_r, mag_d) = (eo_magnitude(&eo_r), eo_magnitude(&eo_d));
                let ((sr, kwr, krr), (sd, kwd, krd)) =
                    maybe_join(|| ical_stat(&mag_r, m, n), || ical_stat(&mag_d, m, n));
                let swv = <$V>::splat_t($token, LO_WEIGHTS[s]);
                let mut i = 0usize;
                while i + $LANES <= mt * nt {
                    let ds = (<$V>::from_array_t($token, core::array::from_fn(|k| sr[i + k]))
                        - <$V>::from_array_t($token, core::array::from_fn(|k| sd[i + k])))
                    .abs();
                    let dk = (<$V>::from_array_t($token, core::array::from_fn(|k| kwr[i + k]))
                        - <$V>::from_array_t($token, core::array::from_fn(|k| kwd[i + k])))
                    .abs();
                    let dt = (<$V>::from_array_t($token, core::array::from_fn(|k| krr[i + k]))
                        - <$V>::from_array_t($token, core::array::from_fn(|k| krd[i + k])))
                    .abs();
                    let acc = <$V>::from_array_t($token, core::array::from_fn(|k| mp_lo_t[i + k]))
                        + swv * (ds + two * dk + dt);
                    mp_lo_t[i..i + $LANES].copy_from_slice(&acc.to_array());
                    i += $LANES;
                }
                while i < mt * nt {
                    let ds = (sr[i] - sd[i]).abs();
                    let dk = (kwr[i] - kwd[i]).abs();
                    let dt = (krr[i] - krd[i]).abs();
                    mp_lo_t[i] += LO_WEIGHTS[s] * (ds + 2.0 * dk + dt);
                    i += 1;
                }
            }
        }
        let mut mp_lo = vec![0.0f32; m * n];
        for jt in 0..mt {
            for it in 0..nt {
                let v = mp_lo_t[jt * nt + it];
                for r in 4 * jt..4 * jt + 4 {
                    mp_lo[r * n + 4 * it..r * n + 4 * it + 4].fill(v);
                }
            }
        }

        // ---------- trim + norms ----------
        let hi = interior_norm(&mp_hi, m, n) * 10.0;
        let lo = interior_norm(&mp_lo, m, n);
        (hi, lo)
    }};
}

/// The `v3` tier (AVX2/FMA, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::attune]
pub fn mad_core_tier_v3_t(token: archmage::X64V3Token, a: Args<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    score_body!(token, V, 8, a)
}

/// The `v4x` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
pub fn mad_core_tier_v4x_t(token: archmage::X64V4xToken, a: Args<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
    score_body!(token, V, 16, a)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::attune]
pub fn mad_core_tier_neon_t(token: archmage::NeonToken, a: Args<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    score_body!(token, V, 8, a)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::attune]
pub fn mad_core_tier_wasm128_t(token: archmage::Wasm128Token, a: Args<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    score_body!(token, V, 8, a)
}

/// The `scalar` tier (8-wide f32, scalar-emulated lanes).
#[archmage::attune]
pub fn mad_core_tier_scalar_t(token: archmage::ScalarToken, a: Args<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    score_body!(token, V, 8, a)
}

/// Runtime-dispatched MAD score → [`crate::MadScore`].
pub(crate) fn mad_score(inputs: &crate::Inputs, w: usize, h: usize) -> crate::MadScore {
    let a = Args {
        im1: &inputs.r,
        im2: &inputs.d,
        w,
        h,
    };
    let (hi, lo) = archmage::attuned!(
        mad_core_tier(a),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    // JEI 2010 combine: sig = 1/(1+b1·HI^b2), MAD = HI^sig·LO^(1−sig).
    let b1 = libm::exp(-2.55 / 3.35);
    let b2 = 1.0 / (libm::log(10.0) * 3.35);
    let sig = 1.0 / (1.0 + b1 * libm::pow(hi, b2));
    crate::MadScore {
        hi,
        lo,
        mad: libm::pow(hi, sig) * libm::pow(lo, 1.0 - sig),
    }
}
