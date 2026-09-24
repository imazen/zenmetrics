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

/// Scalar sRGB8 → gray of one pixel exactly as libgmsd's command-line tool:
/// `((0.299·R + 0.587·G) + 0.114·B)` in f64, then `floor(v + 0.5)` (C
/// `round`, half away from zero, for these non-negative values).
#[inline(always)]
pub(crate) fn gray_px(r: u8, g: u8, b: u8) -> f32 {
    let v = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;
    floor_nonneg(v + 0.5) as f32
}

/// `floor` for non-negative finite values without `std` (truncation).
#[inline(always)]
pub(crate) fn floor_nonneg(v: f64) -> f64 {
    v as i64 as f64
}

// ---------------------------------------------------------------------
// sRGB8 → gray, integer path.
//
// `gray_px` rounds `(0.299·R + 0.587·G) + 0.114·B` in f64. With the integer
// `S = 299·R + 587·G + 114·B` (≤ 255000), `(S + 499) / 1000` equals
// `gray_px` for every triplet EXCEPT the boundary ones `S ≡ 500 (mod 1000)`,
// where the f64 sum lands a few ulps either side of the half-integer and
// either result occurs — proved exhaustively over all 2^24 triplets
// (`tests::integer_luma_exact_off_boundary_exhaustive`). The vectors flag
// exactly those pixels (`S − 1000·q == 500`); flagged lanes are recomputed
// with the f64 formula, so the path is exact by construction + proof.
//
// The quotient uses an f32 detour because SIMD has no integer divide:
// `trunc((S + 499)·0.001f32) == (S + 499)/1000` for every reachable S —
// `S + 499 ≤ 255499` is exact in f32 and the product's error stays far
// below the 1/1000 distance to the nearest integer boundary (also proved
// exhaustively in the test).

/// One group of packed-pixel u32 words (low byte R, then G, B) → base gray
/// `q` and the boundary `flag` as integer vectors of the tier's width.
macro_rules! gray_int_vec {
    ($token:ident, $I32:ident, $F32:ident, $w:expr) => {{
        let w = $w;
        let m255 = $I32::splat($token, 0xFF);
        let r = w & m255;
        let g = w.shr_logical_const::<8>() & m255;
        let b = w.shr_logical_const::<16>() & m255;
        let s = r * 299 + g * 587 + b * 114;
        let q = ((s + 499).to_f32() * $F32::splat($token, 0.001)).to_i32();
        let flag = (s - q * 1000).simd_eq($I32::splat($token, 500));
        (q, flag)
    }};
}

/// Whole-image sRGB8 → gray body: `$LANES` pixels per iteration, integer
/// luma + boundary fixup. `$rgb`/`$out` are one row (out.len() pixels).
macro_rules! gray_row_int_body {
    ($token:ident, $I32:ident, $F32:ident, $LANES:literal, $rgb:ident, $out:ident) => {{
        let n = $out.len();
        let rgb = &$rgb[..3 * n];
        // A pixel's unaligned u32 word covers bytes 3x..3x+4; that stays
        // inside the row slice for x <= n − 2, so the last pixel goes to the
        // scalar tail (`gray_px` — the same result by definition).
        let nvec = n.saturating_sub(1);
        let chunks = nvec / $LANES;
        for i in 0..chunks {
            let x0 = $LANES * i;
            let w = $I32::from_array(
                $token,
                core::array::from_fn(|k| {
                    u32::from_le_bytes(rgb[3 * (x0 + k)..3 * (x0 + k) + 4].try_into().unwrap())
                        as i32
                }),
            );
            let (q, flag) = gray_int_vec!($token, $I32, $F32, w);
            q.to_f32()
                .store((&mut $out[x0..x0 + $LANES]).try_into().unwrap());
            if flag.any_true() {
                let mut m = flag.bitmask();
                while m != 0 {
                    let k = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let p = 3 * (x0 + k);
                    $out[x0 + k] = gray_px(rgb[p], rgb[p + 1], rgb[p + 2]);
                }
            }
        }
        for x in chunks * $LANES..n {
            $out[x] = gray_px(rgb[3 * x], rgb[3 * x + 1], rgb[3 * x + 2]);
        }
    }};
}

/// Fused body: two sRGB8 rows → one half-resolution f32 row. The four luma
/// values of each 2×2 block are integers 0..=255, so the i32 block sum
/// (≤ 1020) is exact, its f32 convert is exact, and `(a+b+c+e)·0.25` equals
/// the libgmsd-order `((0.25·a + 0.25·b) + 0.25·c) + 0.25·e` bit-for-bit:
/// every partial sum there is a multiple of 0.25 ≤ 255, exactly
/// representable (proved in `integer_quad_average_equals_float_order`).
macro_rules! rgb8_pair_half_row_body {
    ($token:ident, $I32:ident, $F32:ident, $LANES:literal, $rgb0:ident, $rgb1:ident, $out:ident) => {{
        let w2 = $out.len();
        let rgb0 = &$rgb0[..6 * w2];
        let rgb1 = &$rgb1[..6 * w2];
        // Output x reads u32 words at byte offsets 6x and 6x+3 — in-slice
        // for x <= w2 − 2; the last output goes to the scalar tail.
        let nvec = w2.saturating_sub(1);
        let chunks = nvec / $LANES;
        let quarter = $F32::splat($token, 0.25);
        for i in 0..chunks {
            let x0 = $LANES * i;
            macro_rules! wv {
                ($row:ident, $off:literal) => {
                    $I32::from_array(
                        $token,
                        core::array::from_fn(|k| {
                            u32::from_le_bytes(
                                $row[6 * (x0 + k) + $off..6 * (x0 + k) + $off + 4]
                                    .try_into()
                                    .unwrap(),
                            ) as i32
                        }),
                    )
                };
            }
            let (qe0, fe0) = gray_int_vec!($token, $I32, $F32, wv!(rgb0, 0));
            let (qo0, fo0) = gray_int_vec!($token, $I32, $F32, wv!(rgb0, 3));
            let (qe1, fe1) = gray_int_vec!($token, $I32, $F32, wv!(rgb1, 0));
            let (qo1, fo1) = gray_int_vec!($token, $I32, $F32, wv!(rgb1, 3));
            let flag = (fe0 | fo0) | (fe1 | fo1);
            let sum = ((qe0 + qo0) + qe1) + qo1; // integer: order irrelevant
            (sum.to_f32() * quarter).store((&mut $out[x0..x0 + $LANES]).try_into().unwrap());
            if flag.any_true() {
                let mut m = flag.bitmask();
                while m != 0 {
                    let k = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let p = 6 * (x0 + k);
                    $out[x0 + k] = ds_px(
                        gray_px(rgb0[p], rgb0[p + 1], rgb0[p + 2]),
                        gray_px(rgb0[p + 3], rgb0[p + 4], rgb0[p + 5]),
                        gray_px(rgb1[p], rgb1[p + 1], rgb1[p + 2]),
                        gray_px(rgb1[p + 3], rgb1[p + 4], rgb1[p + 5]),
                    );
                }
            }
        }
        for x in chunks * $LANES..w2 {
            let p = 6 * x;
            $out[x] = ds_px(
                gray_px(rgb0[p], rgb0[p + 1], rgb0[p + 2]),
                gray_px(rgb0[p + 3], rgb0[p + 4], rgb0[p + 5]),
                gray_px(rgb1[p], rgb1[p + 1], rgb1[p + 2]),
                gray_px(rgb1[p + 3], rgb1[p + 4], rgb1[p + 5]),
            );
        }
    }};
}

/// sRGB8 → gray for a whole strided image into a packed plane (integer path).
#[magetypes(define(i32x8, f32x8), v3, neon, wasm128, scalar)]
fn gray_plane(
    token: Token,
    rgb: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    out: &mut [f32],
) {
    for y in 0..height {
        let row = &rgb[y * stride..y * stride + 3 * width];
        let dst = &mut out[y * width..(y + 1) * width];
        gray_row_int_body!(token, i32x8, f32x8, 8, row, dst)
    }
}

/// 16-wide `gray_plane` for the AVX-512 `v4` tier (crate feature `avx512`).
/// Emits `gray_plane_v4`, which `incant!` picks up by name.
#[magetypes(define(i32x16, f32x16), v4, -scalar)]
fn gray_plane(
    token: Token,
    rgb: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    out: &mut [f32],
) {
    for y in 0..height {
        let row = &rgb[y * stride..y * stride + 3 * width];
        let dst = &mut out[y * width..(y + 1) * width];
        gray_row_int_body!(token, i32x16, f32x16, 16, row, dst)
    }
}

/// Runtime-dispatched whole-image sRGB8 → gray.
pub(crate) fn rgb8_to_gray_plane(
    rgb: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    out: &mut [f32],
) {
    archmage::incant!(
        gray_plane(rgb, width, height, stride, out),
        [v4, v3, neon, wasm128, scalar]
    );
}

/// 2×2 average + decimation of one output row, `$LANES` lanes per step.
///
/// `r0`/`r1` are input rows `2y` and `2y+1` (at least `2 · out.len()`
/// samples each); `out[x]` receives the average of the block at column `2x`.
/// Every output is an independent per-lane f32 chain, so all widths are
/// bit-identical.
macro_rules! downsample_row_body {
    ($token:ident, $F32:ident, $LANES:literal, $r0:ident, $r1:ident, $out:ident) => {{
        let n = $out.len();
        let r0 = &$r0[..2 * n];
        let r1 = &$r1[..2 * n];
        let q = $F32::splat($token, 0.25);
        let chunks = n / $LANES;
        for i in 0..chunks {
            // Fixed-size windows: one range check each, none inside.
            let a: &[f32; 2 * $LANES] = r0[2 * $LANES * i..2 * $LANES * i + 2 * $LANES]
                .try_into()
                .unwrap();
            let b: &[f32; 2 * $LANES] = r1[2 * $LANES * i..2 * $LANES * i + 2 * $LANES]
                .try_into()
                .unwrap();
            let ae = $F32::from_array($token, core::array::from_fn(|k| a[2 * k]));
            let ao = $F32::from_array($token, core::array::from_fn(|k| a[2 * k + 1]));
            let be = $F32::from_array($token, core::array::from_fn(|k| b[2 * k]));
            let bo = $F32::from_array($token, core::array::from_fn(|k| b[2 * k + 1]));
            let v = ((q * ae + q * ao) + q * be) + q * bo;
            v.store(
                (&mut $out[$LANES * i..$LANES * i + $LANES])
                    .try_into()
                    .unwrap(),
            );
        }
        for x in chunks * $LANES..n {
            $out[x] = ds_px(r0[2 * x], r0[2 * x + 1], r1[2 * x], r1[2 * x + 1]);
        }
    }};
}

/// Prewitt gradient magnitude + GMS for `$W` lanes starting at `$x` —
/// the `q` vector, not yet stored. Shared by every width: each lane runs
/// the same independent f32 chain, so all widths are bit-identical.
macro_rules! gms_chunk {
    ($token:ident, $FV:ident, $W:literal, $x:expr, $ru:ident, $rm:ident, $rd:ident, $du:ident, $dm:ident, $dd:ident) => {{
        let x: usize = $x;
        let t = $FV::splat($token, PREWITT_T);
        let tn = $FV::splat($token, -PREWITT_T);
        let two = $FV::splat($token, 2.0);
        let c = $FV::splat($token, GMS_C);
        // A macro, not a closure: closures do not reliably inherit the
        // enclosing `#[target_feature]` region.
        macro_rules! ld {
            ($win:ident, $s:literal) => {
                $FV::from_array($token, core::array::from_fn(|k| $win[k + $s]))
            };
        }
        // ($W + 2)-wide windows: shifts 0, 1, 2 of $W lanes each.
        let wru: &[f32; $W + 2] = $ru[x..x + $W + 2].try_into().unwrap();
        let wrm: &[f32; $W + 2] = $rm[x..x + $W + 2].try_into().unwrap();
        let wrd: &[f32; $W + 2] = $rd[x..x + $W + 2].try_into().unwrap();
        let wdu: &[f32; $W + 2] = $du[x..x + $W + 2].try_into().unwrap();
        let wdm: &[f32; $W + 2] = $dm[x..x + $W + 2].try_into().unwrap();
        let wdd: &[f32; $W + 2] = $dd[x..x + $W + 2].try_into().unwrap();

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
        ((g1 * two) * g2 + c) / ((g1 * g1 + g2 * g2) + c)
    }};
}

/// Gradient magnitude + GMS for one output row, plus its f64 pooling sums.
///
/// `ru/rm/rd` (reference) and `du/dm/dd` (distorted) are the three padded
/// half-resolution rows y−1, y, y+1: sample x lives at index x+1, indices 0
/// and `w+1` are zero, and each slice has at least `w + 10` entries so the
/// last vector's shifted loads stay in bounds. Writes `q` to `map[..w]` and
/// returns `(Σ(1−q), Σ(1−q)²)` over the row.
///
/// `$LANES` is the main-loop width (8 for the established tiers, 16 for
/// `v4`). The f64 pooling keeps a fixed lane grouping on plain scalar
/// arrays — independent of compute width and of any vector reduction
/// order — so every tier's row sums are bit-identical (a backend
/// `reduce_add` is not order-stable across tiers: v3 halves then
/// horizontal-adds, scalar accumulates sequentially).
macro_rules! gms_row_body {
    ($token:ident, $F32:ident, $LANES:literal, $ru:ident, $rm:ident, $rd:ident, $du:ident, $dm:ident, $dd:ident, $map:ident) => {{
        let w = $map.len();
        // Exact-length re-slices: with `len == w + 10` and `x + $LANES <= w`,
        // LLVM proves every ($LANES + 2)-wide window in bounds, so the hot
        // loop carries no per-iteration checks (measured in the disassembly).
        let (ru, rm, rd) = (&$ru[..w + 10], &$rm[..w + 10], &$rd[..w + 10]);
        let (du, dm, dd) = (&$du[..w + 10], &$dm[..w + 10], &$dd[..w + 10]);
        // Pooling accumulators: acc[0..4] / acc[4..8] are the old s1a/s1b
        // (or s2a/s2b) lanes. Per-lane scalar adds are the same ops the
        // f64x4 grouping did elementwise, so the scalar tier's results are
        // unchanged bit-for-bit.
        let mut acc1 = [0.0f64; 8];
        let mut acc2 = [0.0f64; 8];
        // One 8-lane pooling step, fixed order.
        macro_rules! acc8 {
            ($qa:ident, $g:expr) => {{
                let g: usize = $g;
                for j in 0..4 {
                    let lo = 1.0 - $qa[g + j] as f64;
                    let hi = 1.0 - $qa[g + 4 + j] as f64;
                    acc1[j] += lo;
                    acc1[4 + j] += hi;
                    acc2[j] += lo * lo;
                    acc2[4 + j] += hi * hi;
                }
            }};
        }
        let mut x = 0usize;
        while x + $LANES <= w {
            let q = gms_chunk!($token, $F32, $LANES, x, ru, rm, rd, du, dm, dd);
            q.store((&mut $map[x..x + $LANES]).try_into().unwrap());
            let qa = q.to_array();
            for g in (0..$LANES).step_by(8) {
                acc8!(qa, g);
            }
            x += $LANES;
        }
        // A 16-wide main loop can leave a full 8-block unprocessed; 8-wide
        // tiers never enter this loop (their leftover is < 8).
        while x + 8 <= w {
            let q = gms_chunk!($token, f32x8, 8, x, ru, rm, rd, du, dm, dd);
            q.store((&mut $map[x..x + 8]).try_into().unwrap());
            let qa = q.to_array();
            acc8!(qa, 0);
            x += 8;
        }
        // Same order as the old `(s1a + s1b).reduce_add()` on the scalar
        // backend: pairwise lane add, then sequential sum.
        let t1 = [
            acc1[0] + acc1[4],
            acc1[1] + acc1[5],
            acc1[2] + acc1[6],
            acc1[3] + acc1[7],
        ];
        let t2 = [
            acc2[0] + acc2[4],
            acc2[1] + acc2[5],
            acc2[2] + acc2[6],
            acc2[3] + acc2[7],
        ];
        let mut s1 = t1[0] + t1[1] + t1[2] + t1[3];
        let mut s2 = t2[0] + t2[1] + t2[2] + t2[3];
        while x < w {
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
            $map[x] = q;
            let e = 1.0 - q as f64;
            s1 += e;
            s2 += e * e;
            x += 1;
        }
        (s1, s2)
    }};
}

/// 8-wide GMS row for `v3`/`neon`/`wasm128`/`scalar`.
#[magetypes(rite, define(f32x8), v3, neon, wasm128, scalar)]
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
    gms_row_body!(token, f32x8, 8, ru, rm, rd, du, dm, dd, map)
}

/// 16-wide GMS row for the AVX-512 `v4` tier (crate feature `avx512`).
#[magetypes(rite, define(f32x16, f32x8), v4, -scalar)]
fn gms_row16(
    token: Token,
    ru: &[f32],
    rm: &[f32],
    rd: &[f32],
    du: &[f32],
    dm: &[f32],
    dd: &[f32],
    map: &mut [f32],
) -> (f64, f64) {
    gms_row_body!(token, f32x16, 16, ru, rm, rd, du, dm, dd, map)
}

/// One plane view: `data[y * stride + x]`, `x < width`, `y < height`.
#[derive(Clone, Copy)]
pub struct Plane<'a> {
    /// Samples, rows `stride` apart.
    pub data: &'a [f32],
    /// Row pitch in samples.
    pub stride: usize,
}

impl<'a> Plane<'a> {
    #[inline(always)]
    fn row(&self, y: usize, len: usize) -> &'a [f32] {
        &self.data[y * self.stride..y * self.stride + len]
    }
}

/// Where a band reads its input rows from: an f32 gray plane, or sRGB8
/// pixels converted to gray row by row inside the band (so the RGB entry
/// never materialises full-size gray planes and converts in parallel).
#[derive(Clone, Copy)]
pub enum Source<'a> {
    /// Gray f32 plane on the 0..255 scale.
    Gray(Plane<'a>),
    /// Packed RGB triplets, rows `stride` bytes apart.
    Rgb8 {
        /// RGB bytes.
        data: &'a [u8],
        /// Row pitch in bytes.
        stride: usize,
    },
}

/// One band of output rows `[y0, y1)` of the half-resolution GMS grid.
pub struct Band<'a> {
    /// Reference image rows.
    pub reference: Source<'a>,
    /// Distorted image rows.
    pub distorted: Source<'a>,
    /// Half-resolution width and height (`width / 2`, `height / 2`).
    pub w2: usize,
    /// Half-resolution height.
    pub h2: usize,
    /// First output row of the band.
    pub y0: usize,
    /// One past the last output row of the band.
    pub y1: usize,
}

/// Padded row pitch: sample x at index x+1, zeros at 0 and w2+1, and slack so
/// the last vector's 10-wide window stays in bounds.
#[inline(always)]
pub(crate) fn padded_pitch(w2: usize) -> usize {
    w2 + 10
}

// One `#[arcane]` entry per tier; the per-row work is the shared body
// macros expanded inside this target-feature region, so no out-of-line
// call remains per row. `$FV`/`$IV` are the tier's f32/i32 vector type
// names, `$LANES` its width, `$gms` the tier-suffixed GMS row helper.
macro_rules! band_body {
    ($token:ident, $band:ident, $map:ident, $step:ident, $sums:ident, $FV:ty, $IV:ty, $LANES:literal, $gms:ident) => {{
        type V = $FV;
        type I = $IV;
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
                macro_rules! fill_one {
                    ($src:expr, $dst:ident) => {
                        match $src {
                            Source::Gray(p) => {
                                let r0 = p.row(2 * y, in_w);
                                let r1 = p.row(2 * y + 1, in_w);
                                downsample_row_body!($token, V, $LANES, r0, r1, $dst);
                            }
                            // sRGB8 → half-res in one fused integer pass:
                            // no full-size gray scratch rows at all.
                            Source::Rgb8 { data, stride } => {
                                let a = 2 * y * stride;
                                let rgb0 = &data[a..a + 3 * in_w];
                                let rgb1 = &data[a + stride..a + stride + 3 * in_w];
                                rgb8_pair_half_row_body!(
                                    $token, I, V, $LANES, rgb0, rgb1, $dst
                                );
                            }
                        }
                    };
                }
                fill_one!($band.reference, rr);
                fill_one!($band.distorted, dr);
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
/// The `v3` tier of one band (map rows `step` apart, per-row pooling sums).
pub fn gmsd_band_v3(
    token: archmage::X64V3Token,
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
        magetypes::simd::generic::f32x8<archmage::X64V3Token>,
        magetypes::simd::generic::i32x8<archmage::X64V3Token>,
        8,
        gms_row_v3
    )
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
/// The AVX-512 `v4` tier of one band (16-wide; map rows `step` apart,
/// per-row pooling sums).
pub fn gmsd_band_v4(
    token: archmage::X64V4Token,
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
        magetypes::simd::generic::f32x16<archmage::X64V4Token>,
        magetypes::simd::generic::i32x16<archmage::X64V4Token>,
        16,
        gms_row16_v4
    )
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
/// The `neon` tier of one band (map rows `step` apart, per-row pooling sums).
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
        magetypes::simd::generic::f32x8<archmage::NeonToken>,
        magetypes::simd::generic::i32x8<archmage::NeonToken>,
        8,
        gms_row_neon
    )
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
/// The `wasm128` tier of one band (map rows `step` apart, per-row pooling sums).
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
        magetypes::simd::generic::f32x8<archmage::Wasm128Token>,
        magetypes::simd::generic::i32x8<archmage::Wasm128Token>,
        8,
        gms_row_wasm128
    )
}

/// The `scalar` tier of one band (map rows `step` apart, per-row pooling sums).
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
        magetypes::simd::generic::f32x8<archmage::ScalarToken>,
        magetypes::simd::generic::i32x8<archmage::ScalarToken>,
        8,
        gms_row_scalar
    )
}

/// Runtime-dispatched band: the best tier this CPU supports.
pub(crate) fn gmsd_band(band: &Band<'_>, map: &mut [f32], step: usize, sums: &mut [(f64, f64)]) {
    archmage::incant!(
        gmsd_band(band, map, step, sums),
        [v4, v3, neon, wasm128, scalar]
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
