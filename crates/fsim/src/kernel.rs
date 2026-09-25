//! The FSIM/FSIMc pipeline, ported pass-for-pass from the authors'
//! `FR_FSIMc.m` (the "FSIM Index with automatic downsampling, Version
//! 1.0" reference — phasecong2 is Kovesi's public-domain code embedded
//! in that file).
//!
//! Precision layout: f32 planes end-to-end (the vendored FFT is f32),
//! f64 for every scalar reduction (medians, noise sums, the pooling
//! numerator/denominator) in fixed raster order. The element-wise map
//! math goes through `*_body!` macro instantiations so every SIMD tier
//! computes the identical per-pixel expression; everything outside a
//! body (FFT, filter construction, conv borders, reductions) is shared
//! scalar code and therefore tier-independent by construction.

use alloc::vec;
use alloc::vec::Vec;

use crate::fft::{Complex, Plan, fft2_planned, ifft2_planned};
use crate::{I_COEF, Q_COEF, Y_COEF, plane_from_rgb8};

// ------------------------------------------------------------------
// Constants (verbatim from FR_FSIMc.m / phasecong2).
// ------------------------------------------------------------------

/// `nscale` — number of wavelet scales in the filterbank.
const NSCALE: usize = 4;
/// `norient` — number of filter orientations.
const NORIENT: usize = 4;
/// `minWaveLength` — wavelength of the smallest-scale filter.
const MIN_WAVELENGTH: f64 = 6.0;
/// `mult` — scaling factor between successive filters.
const MULT: f64 = 2.0;
/// `sigmaOnf` — log-Gabor bandwidth (sigma / centre frequency).
const SIGMA_ONF: f64 = 0.55;
/// `dThetaOnSigma` — angular interval / angular sigma ratio.
const DTHETA_ON_SIGMA: f64 = 1.2;
/// `k` — noise threshold in sigma units beyond the Rayleigh mean.
const K_NOISE: f64 = 2.0;
/// `epsilon` — added to `XEnergy` to prevent division by zero.
const EPSILON: f32 = 0.0001;
/// Low-pass pre-filter: Butterworth `1/(1+(r/cutoff)^(2n))`,
/// `cutoff = .45`, `n = 15`.
const LP_CUTOFF: f64 = 0.45;
/// Butterworth order of the low-pass pre-filter.
const LP_ORDER: f64 = 15.0;
/// `T1` — phase-congruency similarity constant.
const T1: f32 = 0.85;
/// `T2` — gradient similarity constant.
const T2: f32 = 160.0;
/// `T3` — I-channel similarity constant.
const T3: f32 = 200.0;
/// `T4` — Q-channel similarity constant.
const T4: f32 = 200.0;
/// `lambda` — FSIMc chroma exponent.
const LAMBDA: f64 = 0.03;
/// `cos(π·λ)` — the real part of `(-1)^λ`, used by `real(z^λ)` for
/// negative chroma products exactly as MATLAB's element-wise `.^`
/// complex power produces.
const COS_PI_LAMBDA: f64 = 0.99556196460308; // cos(0.03·π)

/// Scharr-style gradient kernels from the reference:
/// `dx = [3 0 -3; 10 0 -10; 3 0 -3]/16`, `dy` its transpose.
/// `conv2 'same'` applies the kernel flipped — `out(i,j) =
/// Σ K[a,b]·Y(i+1−a, j+1−b)` with zero outside.
const DX: [[f32; 3]; 3] = [
    [3.0 / 16.0, 0.0, -3.0 / 16.0],
    [10.0 / 16.0, 0.0, -10.0 / 16.0],
    [3.0 / 16.0, 0.0, -3.0 / 16.0],
];
const DY: [[f32; 3]; 3] = [
    [3.0 / 16.0, 10.0 / 16.0, 3.0 / 16.0],
    [0.0, 0.0, 0.0],
    [-3.0 / 16.0, -10.0 / 16.0, -3.0 / 16.0],
];

// ------------------------------------------------------------------
// Vector leaf bodies — instantiated per tier inside `score_body!`.
// Planes here are unpadded `Vec<f32>`/`Vec<Complex>` of `n = w·h`.
// ------------------------------------------------------------------

/// `se[i] += e[i].re; so[i] += e[i].im; sa[i] += |e[i]|` — the
/// per-scale even/odd/amplitude accumulation of one orientation.
macro_rules! anacc_body {
    ($token:ident, $F32:ident, $LANES:literal, $e:ident, $se:ident, $so:ident, $sa:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let re = $F32::from_array($token, core::array::from_fn(|k| $e[x + k].re));
            let im = $F32::from_array($token, core::array::from_fn(|k| $e[x + k].im));
            let vse = $F32::from_array($token, core::array::from_fn(|k| $se[x + k]));
            let vso = $F32::from_array($token, core::array::from_fn(|k| $so[x + k]));
            let vsa = $F32::from_array($token, core::array::from_fn(|k| $sa[x + k]));
            (vse + re).store((&mut $se[x..x + $LANES]).try_into().unwrap());
            (vso + im).store((&mut $so[x..x + $LANES]).try_into().unwrap());
            let an = (re * re + im * im).sqrt();
            (vsa + an).store((&mut $sa[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let (re, im) = ($e[x].re, $e[x].im);
            $se[x] += re;
            $so[x] += im;
            $sa[x] += (re * re + im * im).sqrt();
            x += 1;
        }
    }};
}

/// `xe[i] = sqrt(se²+so²)+eps; me[i] = se/xe; mo[i] = so/xe` — the
/// weighted mean phase-angle unit vector of one orientation.
macro_rules! xenergy_body {
    ($token:ident, $F32:ident, $LANES:literal, $se:ident, $so:ident, $me:ident, $mo:ident, $n:ident) => {{
        let n = $n;
        let eps = $F32::splat($token, EPSILON);
        let mut x = 0usize;
        while x + $LANES <= n {
            let vse = $F32::from_array($token, core::array::from_fn(|k| $se[x + k]));
            let vso = $F32::from_array($token, core::array::from_fn(|k| $so[x + k]));
            let xe = (vse * vse + vso * vso).sqrt() + eps;
            (vse / xe).store((&mut $me[x..x + $LANES]).try_into().unwrap());
            (vso / xe).store((&mut $mo[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let xe = ($se[x] * $se[x] + $so[x] * $so[x]).sqrt() + EPSILON;
            $me[x] = $se[x] / xe;
            $mo[x] = $so[x] / xe;
            x += 1;
        }
    }};
}

/// `energy[i] += e.re·me + e.im·mo − |e.re·mo − e.im·me|` — the
/// `An(cos Δφ − |sin Δφ|)` accumulation of one scale.
macro_rules! energy_body {
    ($token:ident, $F32:ident, $LANES:literal, $e:ident, $me:ident, $mo:ident, $energy:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let re = $F32::from_array($token, core::array::from_fn(|k| $e[x + k].re));
            let im = $F32::from_array($token, core::array::from_fn(|k| $e[x + k].im));
            let vme = $F32::from_array($token, core::array::from_fn(|k| $me[x + k]));
            let vmo = $F32::from_array($token, core::array::from_fn(|k| $mo[x + k]));
            let prev = $F32::from_array($token, core::array::from_fn(|k| $energy[x + k]));
            let term = (re * vme + im * vmo) - (re * vmo - im * vme).abs();
            (prev + term).store((&mut $energy[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            let (re, im) = ($e[x].re, $e[x].im);
            $energy[x] += re * $me[x] + im * $mo[x] - (re * $mo[x] - im * $me[x]).abs();
            x += 1;
        }
    }};
}

/// `eall[i] += max(energy[i] − t, 0); aall[i] += sa[i]` — noise
/// thresholding and the cross-orientation fold.
macro_rules! ethr_body {
    ($token:ident, $F32:ident, $LANES:literal, $energy:ident, $sa:ident, $eall:ident, $aall:ident, $t:ident, $n:ident) => {{
        let n = $n;
        let tv = $F32::splat($token, $t);
        let zv = $F32::splat($token, 0.0);
        let mut x = 0usize;
        while x + $LANES <= n {
            let e = $F32::from_array($token, core::array::from_fn(|k| $energy[x + k]));
            let sa = $F32::from_array($token, core::array::from_fn(|k| $sa[x + k]));
            let pe = $F32::from_array($token, core::array::from_fn(|k| $eall[x + k]));
            let pa = $F32::from_array($token, core::array::from_fn(|k| $aall[x + k]));
            (pe + (e - tv).max(zv)).store((&mut $eall[x..x + $LANES]).try_into().unwrap());
            (pa + sa).store((&mut $aall[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            $eall[x] += ($energy[x] - $t).max(0.0);
            $aall[x] += $sa[x];
            x += 1;
        }
    }};
}

/// `pc[i] = eall[i] / aall[i]` — the finished phase-congruency map.
macro_rules! pcdiv_body {
    ($token:ident, $F32:ident, $LANES:literal, $eall:ident, $aall:ident, $pc:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let e = $F32::from_array($token, core::array::from_fn(|k| $eall[x + k]));
            let a = $F32::from_array($token, core::array::from_fn(|k| $aall[x + k]));
            (e / a).store((&mut $pc[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            $pc[x] = $eall[x] / $aall[x];
            x += 1;
        }
    }};
}

/// `g[i] = sqrt(ix[i]² + iy[i]²)` — gradient magnitude.
macro_rules! grad_body {
    ($token:ident, $F32:ident, $LANES:literal, $ix:ident, $iy:ident, $g:ident, $n:ident) => {{
        let n = $n;
        let mut x = 0usize;
        while x + $LANES <= n {
            let vx = $F32::from_array($token, core::array::from_fn(|k| $ix[x + k]));
            let vy = $F32::from_array($token, core::array::from_fn(|k| $iy[x + k]));
            (vx * vx + vy * vy)
                .sqrt()
                .store((&mut $g[x..x + $LANES]).try_into().unwrap());
            x += $LANES;
        }
        while x < n {
            $g[x] = ($ix[x] * $ix[x] + $iy[x] * $iy[x]).sqrt();
            x += 1;
        }
    }};
}

// ------------------------------------------------------------------
// Shared scalar pieces (identical on every tier — no token needed).
// ------------------------------------------------------------------

/// `F = max(1, round(min(w,h)/256))` — the reference's automatic
/// downsampling factor. Integer-exact: `round` on a positive real is
/// `(2m + 256) / 512` floored, i.e. `(m + 128) / 256` for `m ≥ 0`.
fn downsample_factor(w: usize, h: usize) -> usize {
    ((w.min(h) + 128) / 256).max(1)
}

/// `conv2(x, ones(F,F)/F², 'same')` then `x(1:F:end, 1:F:end)` — the
/// reference's averaging downsample. The box kernel is separable and
/// symmetric, so the conv is a horizontal then vertical running sum
/// with the `same` anchor `s = F/2` (forward-looking for even F,
/// centred for odd F), zero outside.
fn decimate_plane(x: &[f32], w: usize, h: usize, f: usize) -> (usize, usize, Vec<f32>) {
    if f == 1 {
        return (w, h, x.to_vec());
    }
    let s = f / 2;
    // Horizontal box sums into `tmp` (un-normalised).
    let mut tmp = vec![0.0f32; w * h];
    for i in 0..h {
        let row = &x[i * w..(i + 1) * w];
        let trow = &mut tmp[i * w..(i + 1) * w];
        for (j, t) in trow.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            // out[i,j] = Σ_b x[i, j+s−b] over b ∈ [0, f)
            let lo = (j + s).saturating_sub(f - 1);
            let hi = (j + s).min(w - 1);
            for v in row[lo..=hi].iter() {
                acc += v;
            }
            *t = acc;
        }
    }
    // Vertical box sums at the decimated positions only, normalised.
    let w2 = w.div_ceil(f);
    let h2 = h.div_ceil(f);
    let mut out = vec![0.0f32; w2 * h2];
    let inv = 1.0 / (f * f) as f32;
    for oi in 0..h2 {
        let i = oi * f;
        let lo = (i + s).saturating_sub(f - 1);
        let hi = (i + s).min(h - 1);
        for oj in 0..w2 {
            let j = oj * f;
            let mut acc = 0.0f32;
            for r in lo..=hi {
                acc += tmp[r * w + j];
            }
            out[oi * w2 + oj] = acc * inv;
        }
    }
    (w2, h2, out)
}

/// `conv2(Y, K, 'same')` for a 3×3 kernel — `out(i,j) =
/// Σ_{a,b} K[a,b]·Y(i+1−a, j+1−b)`, zero outside (the kernel is
/// applied flipped, as MATLAB's `conv2` does).
fn conv2_same_3x3(y: &[f32], w: usize, h: usize, k: &[[f32; 3]; 3]) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for i in 0..h {
        for j in 0..w {
            let mut acc = 0.0f32;
            for a in 0..3isize {
                let r = i as isize + 1 - a;
                if r < 0 || r >= h as isize {
                    continue;
                }
                for b in 0..3isize {
                    let c = j as isize + 1 - b;
                    if c < 0 || c >= w as isize {
                        continue;
                    }
                    acc += k[a as usize][b as usize] * y[r as usize * w + c as usize];
                }
            }
            out[i * w + j] = acc;
        }
    }
    out
}

/// The frequency grids `phasecong2` builds once per size:
/// `(radius, sin θ, cos θ)` in `ifftshift`ed DFT-bin order with the
/// `radius(1,1) = 1` DC fudge applied. `x,y` are the ±0.5-normalised
/// centred coordinates; `theta = atan2(−y, x)`.
fn freq_grids(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Centred normalised coordinates, then ifftshift: the value at
    // index j is `cen[(j + n/2) % n]` — matches MATLAB's
    // `ifftshift` for both parities.
    let xcoord: Vec<f64> = (0..w)
        .map(|j| {
            let c = (j + w / 2) % w;
            if w.is_multiple_of(2) {
                (c as f64 - w as f64 / 2.0) / w as f64
            } else {
                (c as f64 - (w as f64 - 1.0) / 2.0) / (w as f64 - 1.0)
            }
        })
        .collect();
    let ycoord: Vec<f64> = (0..h)
        .map(|i| {
            let c = (i + h / 2) % h;
            if h.is_multiple_of(2) {
                (c as f64 - h as f64 / 2.0) / h as f64
            } else {
                (c as f64 - (h as f64 - 1.0) / 2.0) / (h as f64 - 1.0)
            }
        })
        .collect();
    let mut radius = vec![0.0f32; w * h];
    let mut sintheta = vec![0.0f32; w * h];
    let mut costheta = vec![0.0f32; w * h];
    for (i, &yc) in ycoord.iter().enumerate() {
        for (j, &xc) in xcoord.iter().enumerate() {
            let r = libm::sqrt(xc * xc + yc * yc);
            let t = libm::atan2(-yc, xc);
            let p = i * w + j;
            radius[p] = r as f32;
            sintheta[p] = libm::sin(t) as f32;
            costheta[p] = libm::cos(t) as f32;
        }
    }
    radius[0] = 1.0; // DC fudge so log(radius) never diverges.
    (radius, sintheta, costheta)
}

/// The radial filter components: `lp` and `logGabor[s]` for
/// `s ∈ [0, nscale)` — `lg = exp(−log(r/fo)² / (2·log σ²))·lp`,
/// `fo = 1/(minWaveLength·mult^s)`, DC reset to 0.
fn radial_filters(w: usize, h: usize, radius: &[f32]) -> [Vec<f32>; NSCALE] {
    let n = w * h;
    let mut lg: [Vec<f32>; NSCALE] = core::array::from_fn(|_| vec![0.0; n]);
    let log_sigma = libm::log(SIGMA_ONF);
    for (s, lg_s) in lg.iter_mut().enumerate() {
        let fo = 1.0 / (MIN_WAVELENGTH * libm::pow(MULT, s as f64));
        for i in 0..n {
            let r = radius[i] as f64;
            let lp = 1.0 / (1.0 + libm::pow(r / LP_CUTOFF, 2.0 * LP_ORDER));
            let l = libm::log(r / fo);
            lg_s[i] = (libm::exp(-(l * l) / (2.0 * log_sigma * log_sigma)) * lp) as f32;
        }
        lg_s[0] = 0.0; // undo the radius fudge at DC.
    }
    lg
}

/// The angular filter components: `spread[o]` for `o ∈ [0, norient)`,
/// `spread = exp(−dθ² / (2·θσ²))` with `θσ = π/norient/dThetaOnSigma`
/// and `dθ = |atan2(sinΔ, cosΔ)|` the wrapped angular distance to
/// `angl = o·π/norient`.
fn angular_filters(w: usize, h: usize, sintheta: &[f32], costheta: &[f32]) -> [Vec<f32>; NORIENT] {
    let n = w * h;
    let theta_sigma = core::f64::consts::PI / NORIENT as f64 / DTHETA_ON_SIGMA;
    let denom = 2.0 * theta_sigma * theta_sigma;
    let mut spread: [Vec<f32>; NORIENT] = core::array::from_fn(|_| vec![0.0; n]);
    for (o, sp) in spread.iter_mut().enumerate() {
        let angl = o as f64 * core::f64::consts::PI / NORIENT as f64;
        let (sa, ca) = (libm::sin(angl), libm::cos(angl));
        for i in 0..n {
            let (st, ct) = (sintheta[i] as f64, costheta[i] as f64);
            let ds = st * ca - ct * sa;
            let dc = ct * ca + st * sa;
            let dtheta = libm::fabs(libm::atan2(ds, dc));
            sp[i] = libm::exp(-(dtheta * dtheta) / denom) as f32;
        }
    }
    spread
}

/// MATLAB `median` of a f32 vector: mean of the two middle elements
/// for even counts, the middle for odd — computed in f64.
fn median_f32(v: &mut [f32]) -> f64 {
    v.sort_unstable_by(f32::total_cmp);
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2] as f64
    } else {
        (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.0
    }
}

/// `real(z^λ)` for the FSIMc chroma product, exactly as MATLAB's `.^`
/// complex power: `z^λ` for `z ≥ 0`, `|z|^λ·cos(πλ)` for `z < 0`.
fn real_pow_lambda(z: f32) -> f64 {
    let z64 = z as f64;
    if z64 >= 0.0 {
        libm::pow(z64, LAMBDA)
    } else {
        libm::pow(-z64, LAMBDA) * COS_PI_LAMBDA
    }
}

// ------------------------------------------------------------------
// Whole-pipeline body — instantiated once per tier.
// ------------------------------------------------------------------

macro_rules! score_body {
    ($token:ident, $F32:ident, $LANES:literal, $yr:ident, $yd:ident, $w:ident, $h:ident, $iq:ident) => {{
        let (w, h) = ($w, $h);
        let n = w * h;
        let wplan = Plan::new(w);
        let hplan = Plan::new(h);
        let (radius, sintheta, costheta) = freq_grids(w, h);
        let logg = radial_filters(w, h, &radius);
        let spread = angular_filters(w, h, &sintheta, &costheta);

        // One orientation's accumulators: `(Energy_o, An_o, T)` —
        // Energy_o is *not* yet thresholded (the fold applies `T`).
        let orient = |yf: &[Complex], o: usize| -> (Vec<f32>, Vec<f32>, f32) {
            let mut sum_e = vec![0.0f32; n];
            let mut sum_o = vec![0.0f32; n];
            let mut sum_an = vec![0.0f32; n];
            let mut eo: [Vec<Complex>; NSCALE] =
                core::array::from_fn(|_| vec![Complex::new(0.0, 0.0); n]);
            let mut iff: [Vec<f32>; NSCALE] = core::array::from_fn(|_| vec![0.0; n]);
            let mut em_n = 0.0f64;
            let sqrt_n = libm::sqrt(n as f64) as f32;
            for s in 0..NSCALE {
                // filter = logGabor[s] · spread[o]
                let filter: Vec<f32> = logg[s].iter().zip(&spread[o]).map(|(a, b)| a * b).collect();
                // EO{s,o} = ifft2(imagefft · filter)
                let mut e: Vec<Complex> = yf
                    .iter()
                    .zip(&filter)
                    .map(|(c, f)| Complex::new(c.re * f, c.im * f))
                    .collect();
                ifft2_planned(&mut e, w, h, &wplan, &hplan);
                if s == 0 {
                    for &f in &filter {
                        em_n += (f as f64) * (f as f64);
                    }
                }
                // ifftFilterArray{s} = real(ifft2(filter))·sqrt(n)
                let mut ff: Vec<Complex> = filter.iter().map(|&v| Complex::new(v, 0.0)).collect();
                ifft2_planned(&mut ff, w, h, &wplan, &hplan);
                for i in 0..n {
                    iff[s][i] = ff[i].re * sqrt_n;
                }
                anacc_body!($token, $F32, $LANES, e, sum_e, sum_o, sum_an, n);
                eo[s] = e;
            }
            // Weighted mean phase angle.
            let mut me = vec![0.0f32; n];
            let mut mo = vec![0.0f32; n];
            xenergy_body!($token, $F32, $LANES, sum_e, sum_o, me, mo, n);
            // Energy = Σ_s An·(cos Δφ − |sin Δφ|)
            let mut energy = vec![0.0f32; n];
            for e in eo.iter() {
                energy_body!($token, $F32, $LANES, e, me, mo, energy, n);
            }
            // Noise threshold: Rayleigh estimate from the smallest
            // scale's |EO|² median.
            let mut e2n: Vec<f32> = eo[0].iter().map(|c| c.re * c.re + c.im * c.im).collect();
            let median_e2n = median_f32(&mut e2n);
            let mean_e2n = -median_e2n / libm::log(0.5f64);
            let noise_power = mean_e2n / em_n;
            let mut sum_sa2 = 0.0f64;
            for s in 0..NSCALE {
                for i in 0..n {
                    let f = iff[s][i] as f64;
                    sum_sa2 += f * f;
                }
            }
            let mut sum_saij = 0.0f64;
            for si in 0..NSCALE - 1 {
                for sj in si + 1..NSCALE {
                    for i in 0..n {
                        sum_saij += iff[si][i] as f64 * iff[sj][i] as f64;
                    }
                }
            }
            let est_ne2 = 2.0 * noise_power * sum_sa2 + 4.0 * noise_power * sum_saij;
            let tau = libm::sqrt(est_ne2 / 2.0);
            let est_noise_energy = tau * libm::sqrt(core::f64::consts::PI / 2.0);
            let est_noise_sigma = libm::sqrt((2.0 - core::f64::consts::PI / 2.0) * tau * tau);
            // PC_2's empirical 1.7 rescaling of the PC_1 noise estimate.
            let t = ((est_noise_energy + K_NOISE * est_noise_sigma) / 1.7) as f32;
            (energy, sum_an, t)
        };

        // Phase-congruency maps of both planes.
        let pc = |y: &[f32]| -> Vec<f32> {
            let mut yf: Vec<Complex> = y.iter().map(|&v| Complex::new(v, 0.0)).collect();
            fft2_planned(&mut yf, w, h, &wplan, &hplan);
            let mut eall = vec![0.0f32; n];
            let mut aall = vec![0.0f32; n];
            #[cfg(feature = "parallel")]
            let parts: Vec<(Vec<f32>, Vec<f32>, f32)> = {
                use rayon::prelude::*;
                (0..NORIENT)
                    .into_par_iter()
                    .map(|o| orient(&yf, o))
                    .collect()
            };
            #[cfg(not(feature = "parallel"))]
            let parts: Vec<(Vec<f32>, Vec<f32>, f32)> =
                (0..NORIENT).map(|o| orient(&yf, o)).collect();
            // Fold the four orientations in fixed order.
            for (energy, sum_an, t) in parts {
                ethr_body!($token, $F32, $LANES, energy, sum_an, eall, aall, t, n);
            }
            let mut pc = vec![0.0f32; n];
            pcdiv_body!($token, $F32, $LANES, eall, aall, pc, n);
            pc
        };
        let pc1 = pc($yr);
        let pc2 = pc($yd);

        // Gradient magnitudes of both planes (Scharr pair).
        let ix1 = conv2_same_3x3($yr, w, h, &DX);
        let iy1 = conv2_same_3x3($yr, w, h, &DY);
        let mut g1 = vec![0.0f32; n];
        grad_body!($token, $F32, $LANES, ix1, iy1, g1, n);
        let ix2 = conv2_same_3x3($yd, w, h, &DX);
        let iy2 = conv2_same_3x3($yd, w, h, &DY);
        let mut g2 = vec![0.0f32; n];
        grad_body!($token, $F32, $LANES, ix2, iy2, g2, n);

        // Pooling: Σ GSim·PCSim·PCm(·chroma) / Σ PCm — f64 raster
        // order, identical on every tier.
        let mut num = 0.0f64;
        let mut numc = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..n {
            let (a, b) = (pc1[i], pc2[i]);
            let pcsim = (2.0 * a * b + T1) / (a * a + b * b + T1);
            let (x, z) = (g1[i], g2[i]);
            let gsim = (2.0 * x * z + T2) / (x * x + z * z + T2);
            // MATLAB `max(PC1,PC2)` propagates NaN; `f32::max` would
            // swallow it. Constant inputs produce NaN PC maps — the
            // reference's NaN FSIM depends on this.
            let pcm = if a.is_nan() || b.is_nan() {
                f32::NAN
            } else {
                a.max(b)
            };
            let base = (gsim * pcsim * pcm) as f64;
            num += base;
            if let Some((i1, i2, q1, q2)) = $iq {
                let (ia, ib) = (i1[i], i2[i]);
                let isim = (2.0 * ia * ib + T3) / (ia * ia + ib * ib + T3);
                let (qa, qb) = (q1[i], q2[i]);
                let qsim = (2.0 * qa * qb + T4) / (qa * qa + qb * qb + T4);
                numc += base * real_pow_lambda(isim * qsim);
            }
            den += pcm as f64;
        }
        (
            num / den,
            if $iq.is_some() { numc / den } else { num / den },
        )
    }};
}

// ------------------------------------------------------------------
// Per-tier entry points + dispatch.
// ------------------------------------------------------------------

/// Arguments to a tier entry: the two decimated luma planes plus an
/// optional `(I1, I2, Q1, Q2)` decimated chroma quartet for FSIMc.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Planes<'a> {
    /// Decimated luma planes, `w2·h2` each.
    pub yr: &'a [f32],
    /// Decimated distorted luma.
    pub yd: &'a [f32],
    /// Decimated width.
    pub w: usize,
    /// Decimated height.
    pub h: usize,
    /// Optional decimated chroma `(I_ref, I_dis, Q_ref, Q_dis)`.
    pub iq: Option<IqPlanes<'a>>,
}

/// Decimated I/Q plane quad for the FSIMc chroma term:
/// `(I_ref, I_dis, Q_ref, Q_dis)`.
pub type IqPlanes<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [f32]);

/// The `v3` tier (AVX2/FMA, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn fsim_core_tier_v3(token: archmage::X64V3Token, p: Planes<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 8, yr, yd, w, h, iq)
}

/// The `v4` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub fn fsim_core_tier_v4(token: archmage::X64V4Token, p: Planes<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4Token>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 16, yr, yd, w, h, iq)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn fsim_core_tier_neon(token: archmage::NeonToken, p: Planes<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 8, yr, yd, w, h, iq)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn fsim_core_tier_wasm128(token: archmage::Wasm128Token, p: Planes<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 8, yr, yd, w, h, iq)
}

/// The `scalar` tier (8-wide f32, scalar-emulated lanes).
pub fn fsim_core_tier_scalar(token: archmage::ScalarToken, p: Planes<'_>) -> (f64, f64) {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    let (yr, yd, w, h, iq) = (p.yr, p.yd, p.w, p.h, p.iq);
    score_body!(token, V, 8, yr, yd, w, h, iq)
}

/// Runtime-dispatched score: the best tier this CPU supports.
fn fsim_core(p: Planes<'_>) -> (f64, f64) {
    archmage::incant!(fsim_core_tier(p), [v4, v3, neon, wasm128, scalar])
}

// ------------------------------------------------------------------
// Entry points called by lib.rs.
// ------------------------------------------------------------------

/// FSIM of two strided f32 luma planes (`w`×`h` inputs, the reference's
/// automatic `F` decimation applied inside).
pub(crate) fn fsim_planes(r: &[f32], d: &[f32], w: usize, h: usize, stride: usize) -> f64 {
    let f = downsample_factor(w, h);
    let (w2, h2, yr) = decimate_plane(&crate::plane_from_f32(r, w, h, stride), w, h, f);
    let (_, _, yd) = decimate_plane(&crate::plane_from_f32(d, w, h, stride), w, h, f);
    fsim_core(Planes {
        yr: &yr,
        yd: &yd,
        w: w2,
        h: h2,
        iq: None,
    })
    .0
}

/// FSIM + FSIMc of two packed sRGB8 images.
pub(crate) fn fsim_rgb8_planes(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    stride: usize,
) -> (f64, f64) {
    let f = downsample_factor(w, h);
    let (w2, h2, yr) = decimate_plane(&plane_from_rgb8(r, w, h, stride, Y_COEF), w, h, f);
    let (_, _, yd) = decimate_plane(&plane_from_rgb8(d, w, h, stride, Y_COEF), w, h, f);
    let (_, _, ir) = decimate_plane(&plane_from_rgb8(r, w, h, stride, I_COEF), w, h, f);
    let (_, _, id) = decimate_plane(&plane_from_rgb8(d, w, h, stride, I_COEF), w, h, f);
    let (_, _, qr) = decimate_plane(&plane_from_rgb8(r, w, h, stride, Q_COEF), w, h, f);
    let (_, _, qd) = decimate_plane(&plane_from_rgb8(d, w, h, stride, Q_COEF), w, h, f);
    fsim_core(Planes {
        yr: &yr,
        yd: &yd,
        w: w2,
        h: h2,
        iq: Some((&ir, &id, &qr, &qd)),
    })
}

/// FSIM of the unrounded BT.601 luma of two packed sRGB8 images.
pub(crate) fn fsim_luma8_planes(r: &[u8], d: &[u8], w: usize, h: usize, stride: usize) -> f64 {
    let f = downsample_factor(w, h);
    let (w2, h2, yr) = decimate_plane(&plane_from_rgb8(r, w, h, stride, Y_COEF), w, h, f);
    let (_, _, yd) = decimate_plane(&plane_from_rgb8(d, w, h, stride, Y_COEF), w, h, f);
    fsim_core(Planes {
        yr: &yr,
        yd: &yd,
        w: w2,
        h: h2,
        iq: None,
    })
    .0
}
