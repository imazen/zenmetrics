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

/// Arguments to one MAD run, bundled for `incant!`.
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
    for r in 0..m {
        for c in 0..n {
            let x = (r as f64 - m as f64 / 2.0 + 0.5) * 64.0 / n as f64;
            let y = (c as f64 - n as f64 / 2.0 + 0.5) * 64.0 / n as f64;
            let mut rf = libm::hypot(x, y);
            let s = 0.15 * libm::cos(4.0 * libm::atan2(y, x)) + 0.85;
            rf /= s;
            out[r * n + c] = if rf < 7.8909 {
                0.9809
            } else {
                (2.6 * (0.0192 + 0.114 * rf) * libm::exp(-libm::pow(0.114 * rf, 1.1))) as f32
            };
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
    for u in 0..m {
        let su = (u + m / 2) % m;
        for v in 0..n {
            let sv = (v + n / 2) % n;
            let w = csf[su * n + sv];
            let e = &mut buf[u * n + v];
            *e = Complex::new(e.re * w, e.im * w);
        }
    }
    fft::ifft2_planned(&mut buf, n, m, wplan, hplan);
    buf.iter().map(|c| c.re).collect()
}

/// `ical_std(dst−ref, ref)` → `(std_2, std_1, m1_1)` — verbatim port
/// of the C mex: stride-4 16×16 windows, blocky 4×4 tiles; `std_1` is
/// the 8×8 sample-std of ref min-pooled over `{0,5}` offsets.
/// Numerators accumulate f64 (the C code is `double`); maps store
/// f32.
fn ical_std(diff: &[f32], reff: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut std2, mut std1, mut m1) = (
        vec![0.0f32; m * n],
        vec![0.0f32; m * n],
        vec![0.0f32; m * n],
    );
    let mut tmp = vec![0.0f32; m * n];
    if m < B || n < B {
        return (std2, std1, m1);
    }
    // Pass 1: 16×16 blocks at stride 4.
    for i in (0..=n - B).step_by(4) {
        for j in (0..=m - B).step_by(4) {
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
            let stdev = libm::sqrt(s2 / 255.0) as f32;
            for r in j..j + 4 {
                for c in i..i + 4 {
                    std2[r * n + c] = stdev;
                    m1[r * n + c] = mean2 as f32;
                }
            }
        }
    }
    // Pass 2: 8×8 blocks at the same stride-4 starts.
    for i in (0..=n - B).step_by(4) {
        for j in (0..=m - B).step_by(4) {
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
            for r in j..j + 4 {
                for c in i..i + 4 {
                    tmp[r * n + c] = stdev;
                    std1[r * n + c] = stdev;
                }
            }
        }
    }
    // Pass 3: min-pool over {0,5} offsets, position bounds-checked.
    for i in (0..=n - B).step_by(4) {
        for j in (0..=m - B).step_by(4) {
            let mut mn = tmp[j * n + i];
            for ib in (i..=i + 7).step_by(5) {
                for jb in (j..=j + 7).step_by(5) {
                    if ib < n - 15 && jb < m - 15 && mn > tmp[jb * n + ib] {
                        mn = tmp[jb * n + ib];
                    }
                }
            }
            for r in j..j + 4 {
                for c in i..i + 4 {
                    std1[r * n + c] = mn;
                }
            }
        }
    }
    (std2, std1, m1)
}

/// `ical_stat(x)` → `(std, skew, kurt)` — verbatim port of the C
/// mex: stride-4 16×16 windows, blocky 4×4 tiles; skew/kurtosis
/// normalised by population moments (`stmp`); flat → 0.
fn ical_stat(x: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut std_o, mut skw, mut krt) = (
        vec![0.0f32; m * n],
        vec![0.0f32; m * n],
        vec![0.0f32; m * n],
    );
    if m < B || n < B {
        return (std_o, skw, krt);
    }
    for i in (0..=n - B).step_by(4) {
        for j in (0..=m - B).step_by(4) {
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
            let stdev = libm::sqrt(s2 / 255.0) as f32;
            let stmp = libm::sqrt(s2 / 256.0);
            let (sk, ku) = if stmp != 0.0 {
                (
                    ((s3 / 256.0) / (stmp * stmp * stmp)) as f32,
                    ((s4 / 256.0) / (stmp * stmp * stmp * stmp)) as f32,
                )
            } else {
                (0.0, 0.0)
            };
            for r in j..j + 4 {
                for c in i..i + 4 {
                    std_o[r * n + c] = stdev;
                    skw[r * n + c] = sk;
                    krt[r * n + c] = ku;
                }
            }
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
    for u in 0..m {
        let su = (u + m.div_ceil(2)) % m;
        for v in 0..n {
            let sv = (v + n.div_ceil(2)) % n;
            let w = lg[su * n + sv] * sp[su * n + sv];
            let e = &mut buf[u * n + v];
            *e = Complex::new(e.re * w, e.im * w);
        }
    }
    fft::ifft2_planned(&mut buf, n, m, wplan, hplan);
    buf
}

/// `|EO|` element magnitudes — scalar `hypotf`, matching Octave
/// `abs()` to the ulp on every tier.
fn eo_magnitude(eo: &[Complex]) -> Vec<f32> {
    eo.iter().map(|c| libm::hypotf(c.re, c.im)).collect()
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
        for i in 0..m * n {
            lum_r[i] = (K_LUM * libm::pow($a.im1[i] as f64, 2.2 / 3.0)) as f32;
            lum_d[i] = (K_LUM * libm::pow($a.im2[i] as f64, 2.2 / 3.0)) as f32;
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
                let dv = <$V>::from_array($token, core::array::from_fn(|k| dst_l[i + k]))
                    - <$V>::from_array($token, core::array::from_fn(|k| ref_l[i + k]));
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
                    let dv = <$V>::from_array($token, core::array::from_fn(|k| $a.im1[i + k]))
                        - <$V>::from_array($token, core::array::from_fn(|k| $a.im2[i + k]));
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
                    let mut acc = <$V>::splat($token, 0.0);
                    for u in 0..B {
                        let xv =
                            <$V>::from_array($token, core::array::from_fn(|k| srow[x + 1 + u + k]));
                        acc = acc + xv;
                    }
                    let outv = acc * <$V>::splat($token, 1.0 / 16.0);
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
                    let mut acc = <$V>::splat($token, 0.0);
                    for u in 0..B {
                        let xv = <$V>::from_array(
                            $token,
                            core::array::from_fn(|k| tmp[(y + 1 + u) * n + x + k]),
                        );
                        acc = acc + xv;
                    }
                    let outv = acc * <$V>::splat($token, 1.0 / 16.0);
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
        // reference's own NaN/−inf path maps to msk = 0.
        let mut mp_hi = vec![0.0f32; m * n];
        if m >= B && n >= B {
            for by in (0..=m - B).step_by(4) {
                for bx in (0..=n - B).step_by(4) {
                    for r in by..by + 4 {
                        for c in bx..bx + 4 {
                            let idx = r * n + c;
                            let s1 = std1[idx] as f64;
                            let mu = m1[idx] as f64;
                            let s2 = std2[idx] as f64;
                            let ci_ref = libm::log(s1 / mu);
                            let ci_dst = if m1[idx] < G_LUM as f32 {
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
                            };
                            mp_hi[idx] = msk as f32 * lm[idx];
                        }
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
        let mut mp_lo = vec![0.0f32; m * n];
        let two = <$V>::splat($token, 2.0);
        for s in 0..5 {
            for o in 0..4 {
                let (eo_r, eo_d) = maybe_join(
                    || gabor_apply(&xfft_r, &logs[s], &spreads[o], m, n, &wplan, &hplan),
                    || gabor_apply(&xfft_d, &logs[s], &spreads[o], m, n, &wplan, &hplan),
                );
                let (mag_r, mag_d) = (eo_magnitude(&eo_r), eo_magnitude(&eo_d));
                let ((sr, kwr, krr), (sd, kwd, krd)) =
                    maybe_join(|| ical_stat(&mag_r, m, n), || ical_stat(&mag_d, m, n));
                // mp += s·(|Δstd| + 2|Δskw| + |Δkrt|) over the whole
                // map — outside coverage every map is 0 so the add is
                // a no-op there.
                let swv = <$V>::splat($token, LO_WEIGHTS[s]);
                let mut i = 0usize;
                while i + $LANES <= m * n {
                    let ds = (<$V>::from_array($token, core::array::from_fn(|k| sr[i + k]))
                        - <$V>::from_array($token, core::array::from_fn(|k| sd[i + k])))
                    .abs();
                    let dk = (<$V>::from_array($token, core::array::from_fn(|k| kwr[i + k]))
                        - <$V>::from_array($token, core::array::from_fn(|k| kwd[i + k])))
                    .abs();
                    let dt = (<$V>::from_array($token, core::array::from_fn(|k| krr[i + k]))
                        - <$V>::from_array($token, core::array::from_fn(|k| krd[i + k])))
                    .abs();
                    let acc = <$V>::from_array($token, core::array::from_fn(|k| mp_lo[i + k]))
                        + swv * (ds + two * dk + dt);
                    mp_lo[i..i + $LANES].copy_from_slice(&acc.to_array());
                    i += $LANES;
                }
                while i < m * n {
                    let ds = (sr[i] - sd[i]).abs();
                    let dk = (kwr[i] - kwd[i]).abs();
                    let dt = (krr[i] - krd[i]).abs();
                    mp_lo[i] += LO_WEIGHTS[s] * (ds + 2.0 * dk + dt);
                    i += 1;
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
#[archmage::arcane]
pub fn mad_core_tier_v3(token: archmage::X64V3Token, a: Args<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    score_body!(token, V, 8, a)
}

/// The `v4` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub fn mad_core_tier_v4(token: archmage::X64V4Token, a: Args<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4Token>;
    score_body!(token, V, 16, a)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn mad_core_tier_neon(token: archmage::NeonToken, a: Args<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    score_body!(token, V, 8, a)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn mad_core_tier_wasm128(token: archmage::Wasm128Token, a: Args<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    score_body!(token, V, 8, a)
}

/// The `scalar` tier (8-wide f32, scalar-emulated lanes).
pub fn mad_core_tier_scalar(token: archmage::ScalarToken, a: Args<'_>) -> (f64, f64) {
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
    let (hi, lo) = archmage::incant!(mad_core_tier(a), [v4, v3, neon, wasm128, scalar]);
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
