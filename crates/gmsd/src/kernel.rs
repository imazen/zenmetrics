//! GMSD per-row kernels and per-tier band entry points.
//!
//! Ported from libgmsd <https://github.com/clunietp/libgmsd> (commit
//! `de646c9a957a892e4b9d78ff94601b83b363c158`, 2018-12-17):
//!
//! > MIT License
//! >
//! > Copyright (c) 2018 Tom Clunie
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a copy
//! > of this software and associated documentation files (the "Software"), to deal
//! > in the Software without restriction, including without limitation the rights
//! > to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! > copies of the Software, and to permit persons to whom the Software is
//! > furnished to do so, subject to the following conditions:
//! >
//! > The above copyright notice and this permission notice shall be included in all
//! > copies or substantial portions of the Software.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! > FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! > LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! > SOFTWARE.
//!
//! (Also in `LICENSE-libgmsd` next to this crate's manifest.)
//!
//! # Floating-point order
//!
//! libgmsd computes in `float` with no FMA (its CMake build targets baseline
//! x86-64). Every expression here reproduces its evaluation order term for
//! term, with plain `*` and `+` (Rust never contracts them into FMA), so the
//! GMS map is bit-identical to libgmsd's on finite input:
//!
//! - 2×2 average (`conv2` 'same', kernel rotated 180°, all taps 0.25):
//!   `((0.25·a + 0.25·b) + 0.25·c) + 0.25·e`, `a,b` = row `2y` cols `2x,2x+1`,
//!   `c,e` = row `2y+1`. Then keep every second sample (`downsample_2x2`).
//! - Prewitt, zero-padded 'same', `t = (float)(1/3)`: libgmsd's
//!   `fspecial('prewitt')/3` = `[t t t; 0 0 0; -t -t -t]` and its transpose,
//!   both rotated 180° before the multiply-accumulate. Row `u` = y−1, `m` = y,
//!   `d` = y+1; columns 0,1,2 = x−1, x, x+1:
//!   `hx = ((((−t·u0 + −t·u1) + −t·u2) + t·d0) + t·d1) + t·d2`
//!   `hy = ((((−t·u0 + t·u2) + −t·m0) + t·m2) + −t·d0) + t·d2`
//!   libgmsd also adds the `0·s` products of the zero kernel taps; adding a
//!   signed zero can change only the sign of an exactly-zero partial sum,
//!   which the squaring below erases, so they are skipped.
//! - `g = sqrt(hx·hx + hy·hy)`,
//!   `q = ((2·g1)·g2 + C) / ((g1·g1 + g2·g2) + C)`, `C = 170`.
//!
//! The pooling (standard deviation of `q`) accumulates `1 − q` in f64; see
//! `lib.rs`.

use archmage::magetypes;

/// libgmsd's Prewitt scale, `(matrix_data_t)(1. / 3.)`.
pub(crate) const PREWITT_T: f32 = (1.0f64 / 3.0f64) as f32;

/// The GMS stability constant `c` for 0..255 intensities (libgmsd and the
/// authors' `GMSD.m` both use 170; the paper's 0.0026 is the same constant on
/// a 0..1 scale: `0.0026 · 255² = 169.065`).
pub(crate) const GMS_C: f32 = 170.0;

/// Scalar 2×2 average of one output sample, libgmsd order.
#[inline(always)]
pub(crate) fn ds_px(a: f32, b: f32, c: f32, e: f32) -> f32 {
    ((0.25 * a + 0.25 * b) + 0.25 * c) + 0.25 * e
}

/// Scalar Prewitt gradient magnitude at one sample, libgmsd order.
/// `u`, `m`, `d` are the three padded rows starting at column `x−1`.
#[inline(always)]
pub(crate) fn grad_px(u: [f32; 3], m: [f32; 3], d: [f32; 3]) -> f32 {
    let t = PREWITT_T;
    let tn = -PREWITT_T;
    let hx = ((((tn * u[0] + tn * u[1]) + tn * u[2]) + t * d[0]) + t * d[1]) + t * d[2];
    let hy = ((((tn * u[0] + t * u[2]) + tn * m[0]) + t * m[2]) + tn * d[0]) + t * d[2];
    (hx * hx + hy * hy).sqrt()
}

/// Scalar gradient-magnitude similarity, libgmsd order.
#[inline(always)]
pub(crate) fn gms_px(g1: f32, g2: f32) -> f32 {
    ((g1 * 2.0) * g2 + GMS_C) / ((g1 * g1 + g2 * g2) + GMS_C)
}

/// 2×2 average + decimation of one output row.
///
/// `r0`/`r1` are input rows `2y` and `2y+1` (at least `2 · out.len()`
/// samples each); `out[x]` receives the average of the block at column `2x`.
#[magetypes(rite, define(f32x8), v3, neon, wasm128, scalar)]
fn downsample_row(token: Token, r0: &[f32], r1: &[f32], out: &mut [f32]) {
    let n = out.len();
    let r0 = &r0[..2 * n];
    let r1 = &r1[..2 * n];
    let q = f32x8::splat(token, 0.25);
    let chunks = n / 8;
    for i in 0..chunks {
        // Fixed-size windows: one range check each, none inside.
        let a: &[f32; 16] = r0[16 * i..16 * i + 16].try_into().unwrap();
        let b: &[f32; 16] = r1[16 * i..16 * i + 16].try_into().unwrap();
        let ae = f32x8::from_array(token, core::array::from_fn(|k| a[2 * k]));
        let ao = f32x8::from_array(token, core::array::from_fn(|k| a[2 * k + 1]));
        let be = f32x8::from_array(token, core::array::from_fn(|k| b[2 * k]));
        let bo = f32x8::from_array(token, core::array::from_fn(|k| b[2 * k + 1]));
        let v = ((q * ae + q * ao) + q * be) + q * bo;
        let o: &mut [f32; 8] = (&mut out[8 * i..8 * i + 8]).try_into().unwrap();
        v.store(o);
    }
    for x in chunks * 8..n {
        out[x] = ds_px(r0[2 * x], r0[2 * x + 1], r1[2 * x], r1[2 * x + 1]);
    }
}

/// Gradient magnitude + GMS for one output row, plus its f64 pooling sums.
///
/// `ru/rm/rd` (reference) and `du/dm/dd` (distorted) are the three padded
/// half-resolution rows y−1, y, y+1: sample x lives at index x+1, indices 0
/// and `w+1` are zero, and each slice has at least `w + 10` entries so the
/// last vector's shifted loads stay in bounds. Writes `q` to `map[..w]` and
/// returns `(Σ(1−q), Σ(1−q)²)` over the row.
#[magetypes(rite, define(f32x8, f64x4), v3, neon, wasm128, scalar)]
fn gms_row(
    token: Token,
    ru: &[f32],
    rm: &[f32],
    rd: &[f32],
    du: &[f32],
    dm: &[f32],
    dd: &[f32],
    map: &mut [f32],
) -> (f64, f64) {
    let w = map.len();
    // Exact-length re-slices: with `len == w + 10` and `8·i + 8 <= w`, LLVM
    // proves every 10-wide window below in bounds, so the hot loop carries
    // no per-iteration checks (measured in the disassembly).
    let (ru, rm, rd) = (&ru[..w + 10], &rm[..w + 10], &rd[..w + 10]);
    let (du, dm, dd) = (&du[..w + 10], &dm[..w + 10], &dd[..w + 10]);
    let t = f32x8::splat(token, PREWITT_T);
    let tn = f32x8::splat(token, -PREWITT_T);
    let two = f32x8::splat(token, 2.0);
    let c = f32x8::splat(token, GMS_C);
    let one64 = f64x4::splat(token, 1.0);
    let mut s1a = f64x4::zero(token);
    let mut s1b = f64x4::zero(token);
    let mut s2a = f64x4::zero(token);
    let mut s2b = f64x4::zero(token);
    let chunks = w / 8;
    for i in 0..chunks {
        let x = 8 * i;
        // Ten-wide windows: shifts 0, 1, 2 of eight lanes each.
        let wru: &[f32; 10] = ru[x..x + 10].try_into().unwrap();
        let wrm: &[f32; 10] = rm[x..x + 10].try_into().unwrap();
        let wrd: &[f32; 10] = rd[x..x + 10].try_into().unwrap();
        let wdu: &[f32; 10] = du[x..x + 10].try_into().unwrap();
        let wdm: &[f32; 10] = dm[x..x + 10].try_into().unwrap();
        let wdd: &[f32; 10] = dd[x..x + 10].try_into().unwrap();
        // A macro, not a closure: closures do not reliably inherit the
        // enclosing `#[target_feature]` region.
        macro_rules! ld {
            ($win:ident, $s:literal) => {
                f32x8::from_array(token, core::array::from_fn(|k| $win[k + $s]))
            };
        }

        let (u0, u1, u2) = (ld!(wru, 0), ld!(wru, 1), ld!(wru, 2));
        let (m0, m2) = (ld!(wrm, 0), ld!(wrm, 2));
        let (d0, d1, d2) = (ld!(wrd, 0), ld!(wrd, 1), ld!(wrd, 2));
        let hx = ((((tn * u0 + tn * u1) + tn * u2) + t * d0) + t * d1) + t * d2;
        let hy = ((((tn * u0 + t * u2) + tn * m0) + t * m2) + tn * d0) + t * d2;
        let g1 = (hx * hx + hy * hy).sqrt();

        let (u0, u1, u2) = (ld!(wdu, 0), ld!(wdu, 1), ld!(wdu, 2));
        let (m0, m2) = (ld!(wdm, 0), ld!(wdm, 2));
        let (d0, d1, d2) = (ld!(wdd, 0), ld!(wdd, 1), ld!(wdd, 2));
        let hx = ((((tn * u0 + tn * u1) + tn * u2) + t * d0) + t * d1) + t * d2;
        let hy = ((((tn * u0 + t * u2) + tn * m0) + t * m2) + tn * d0) + t * d2;
        let g2 = (hx * hx + hy * hy).sqrt();

        // Real division, not `recip()`: libgmsd divides.
        let q = ((g1 * two) * g2 + c) / ((g1 * g1 + g2 * g2) + c);
        let o: &mut [f32; 8] = (&mut map[x..x + 8]).try_into().unwrap();
        q.store(o);

        let qa = q.to_array();
        let lo = one64
            - f64x4::from_array(
                token,
                [qa[0] as f64, qa[1] as f64, qa[2] as f64, qa[3] as f64],
            );
        let hi = one64
            - f64x4::from_array(
                token,
                [qa[4] as f64, qa[5] as f64, qa[6] as f64, qa[7] as f64],
            );
        s1a += lo;
        s1b += hi;
        s2a += lo * lo;
        s2b += hi * hi;
    }
    let mut s1 = (s1a + s1b).reduce_add();
    let mut s2 = (s2a + s2b).reduce_add();
    for x in chunks * 8..w {
        let g1 = grad_px(
            [ru[x], ru[x + 1], ru[x + 2]],
            [rm[x], rm[x + 1], rm[x + 2]],
            [rd[x], rd[x + 1], rd[x + 2]],
        );
        let g2 = grad_px(
            [du[x], du[x + 1], du[x + 2]],
            [dm[x], dm[x + 1], dm[x + 2]],
            [dd[x], dd[x + 1], dd[x + 2]],
        );
        let q = gms_px(g1, g2);
        map[x] = q;
        let e = 1.0 - q as f64;
        s1 += e;
        s2 += e * e;
    }
    (s1, s2)
}

/// One plane view: `data[y * stride + x]`, `x < width`, `y < height`.
#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [f32],
    pub stride: usize,
}

impl<'a> Plane<'a> {
    #[inline(always)]
    fn row(&self, y: usize, len: usize) -> &'a [f32] {
        &self.data[y * self.stride..y * self.stride + len]
    }
}

/// One band of output rows `[y0, y1)` of the half-resolution GMS grid.
pub struct Band<'a> {
    pub reference: Plane<'a>,
    pub distorted: Plane<'a>,
    /// Half-resolution width and height (`width / 2`, `height / 2`).
    pub w2: usize,
    pub h2: usize,
    pub y0: usize,
    pub y1: usize,
}

/// Padded row pitch: sample x at index x+1, zeros at 0 and w2+1, and slack so
/// the last vector's 10-wide window stays in bounds.
#[inline(always)]
pub(crate) fn padded_pitch(w2: usize) -> usize {
    w2 + 10
}

// One `#[arcane]` entry per tier; the per-row helpers above are `#[rite]`
// variants of the same tier, so they inline into this one target-feature
// region. The body is shared through a macro because it names the
// tier-suffixed helpers.
macro_rules! band_body {
    ($token:ident, $band:ident, $map:ident, $step:ident, $sums:ident, $ds:ident, $gms:ident) => {{
        let w2 = $band.w2;
        let h2 = $band.h2;
        let pitch = padded_pitch(w2);
        let in_w = 2 * w2;
        // ring[slot] holds padded half-res row `yd` at slot (yd + 1) % 3.
        let mut ring_r = [
            alloc::vec![0.0f32; pitch],
            alloc::vec![0.0f32; pitch],
            alloc::vec![0.0f32; pitch],
        ];
        let mut ring_d = [
            alloc::vec![0.0f32; pitch],
            alloc::vec![0.0f32; pitch],
            alloc::vec![0.0f32; pitch],
        ];
        // A macro, not a closure, for the same target-feature reason as `ld!`.
        macro_rules! fill {
            ($yd:expr) => {{
            let yd: isize = $yd;
            let slot = ((yd + 1) as usize) % 3;
            let rr = &mut ring_r[slot][1..1 + w2];
            let dr = &mut ring_d[slot][1..1 + w2];
            if yd < 0 || yd as usize >= h2 {
                rr.fill(0.0);
                dr.fill(0.0);
            } else {
                let y = yd as usize;
                $ds(
                    $token,
                    $band.reference.row(2 * y, in_w),
                    $band.reference.row(2 * y + 1, in_w),
                    rr,
                );
                $ds(
                    $token,
                    $band.distorted.row(2 * y, in_w),
                    $band.distorted.row(2 * y + 1, in_w),
                    dr,
                );
            }
            }};
        }
        let y0 = $band.y0 as isize;
        fill!(y0 - 1);
        fill!(y0);
        for (i, y) in ($band.y0..$band.y1).enumerate() {
            let yi = y as isize;
            fill!(yi + 1);
            let su = (y) % 3; // slot of y-1 = (y-1+1)%3
            let sm = (y + 1) % 3;
            let sd = (y + 2) % 3;
            // `step` = w2 writes the full map; 0 reuses one scratch row.
            let map_row = &mut $map[i * $step..i * $step + w2];
            $sums[i] = $gms(
                $token,
                &ring_r[su],
                &ring_r[sm],
                &ring_r[sd],
                &ring_d[su],
                &ring_d[sm],
                &ring_d[sd],
                map_row,
            );
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn gmsd_band_v3(
    token: archmage::X64V3Token,
    band: &Band<'_>,
    map: &mut [f32],
    step: usize,
    sums: &mut [(f64, f64)],
) {
    band_body!(token, band, map, step, sums, downsample_row_v3, gms_row_v3)
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn gmsd_band_neon(
    token: archmage::NeonToken,
    band: &Band<'_>,
    map: &mut [f32],
    step: usize,
    sums: &mut [(f64, f64)],
) {
    band_body!(
        token,
        band,
        map,
        step,
        sums,
        downsample_row_neon,
        gms_row_neon
    )
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn gmsd_band_wasm128(
    token: archmage::Wasm128Token,
    band: &Band<'_>,
    map: &mut [f32],
    step: usize,
    sums: &mut [(f64, f64)],
) {
    band_body!(
        token,
        band,
        map,
        step,
        sums,
        downsample_row_wasm128,
        gms_row_wasm128
    )
}

pub fn gmsd_band_scalar(
    token: archmage::ScalarToken,
    band: &Band<'_>,
    map: &mut [f32],
    step: usize,
    sums: &mut [(f64, f64)],
) {
    band_body!(
        token,
        band,
        map,
        step,
        sums,
        downsample_row_scalar,
        gms_row_scalar
    )
}

/// Runtime-dispatched band: the best tier this CPU supports.
pub(crate) fn gmsd_band(band: &Band<'_>, map: &mut [f32], step: usize, sums: &mut [(f64, f64)]) {
    archmage::incant!(
        gmsd_band(band, map, step, sums),
        [v3, neon, wasm128, scalar]
    );
}

/// Straight-line scalar reference of the whole pipeline, full-resolution
/// intermediate planes and libgmsd's exact structure (pad → convolve → crop
/// → pool two-pass). Only used by tests to pin the fast path.
#[cfg(test)]
pub(crate) fn reference_map(
    r: Plane<'_>,
    d: Plane<'_>,
    width: usize,
    height: usize,
) -> alloc::vec::Vec<f32> {
    use alloc::vec::Vec;
    let w2 = width / 2;
    let h2 = height / 2;
    let down = |p: Plane<'_>| -> Vec<f32> {
        let mut o = alloc::vec![0.0f32; w2 * h2];
        for y in 0..h2 {
            for x in 0..w2 {
                o[y * w2 + x] = ds_px(
                    p.data[2 * y * p.stride + 2 * x],
                    p.data[2 * y * p.stride + 2 * x + 1],
                    p.data[(2 * y + 1) * p.stride + 2 * x],
                    p.data[(2 * y + 1) * p.stride + 2 * x + 1],
                );
            }
        }
        o
    };
    let grad = |p: &[f32]| -> Vec<f32> {
        let at = |y: isize, x: isize| -> f32 {
            if y < 0 || x < 0 || y >= h2 as isize || x >= w2 as isize {
                0.0
            } else {
                p[y as usize * w2 + x as usize]
            }
        };
        let mut o = alloc::vec![0.0f32; w2 * h2];
        for y in 0..h2 as isize {
            for x in 0..w2 as isize {
                let row = |yy: isize| [at(yy, x - 1), at(yy, x), at(yy, x + 1)];
                o[y as usize * w2 + x as usize] = grad_px(row(y - 1), row(y), row(y + 1));
            }
        }
        o
    };
    let g1 = grad(&down(r));
    let g2 = grad(&down(d));
    g1.iter()
        .zip(g2.iter())
        .map(|(&a, &b)| gms_px(a, b))
        .collect()
}
