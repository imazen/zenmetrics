//! Attune-mode SIMD kernels for the classical PSNR routes.
//!
//! Exact-score contract: every kernel reproduces the scalar
//! `classical.rs` expressions bit-for-bit. Two mechanisms:
//!
//! - After a rounded-code conversion the error is integer-valued, so the
//!   squared-error sum is an integer. Integer (or integral-valued f64)
//!   addition is order-free while every partial sum stays below 2^53 —
//!   the caller guards that bound, so lane-wise accumulation is bit-exact.
//! - The unrounded f64 luma routes can't reorder the sum: they vectorise
//!   only the per-pixel conversion and `delta²` stages and keep the
//!   scalar left-to-right accumulation.
//!
//! Rounding: `f32::round`/`f64::round` is half-away-from-zero; magetypes
//! `.round()` is round-half-even, so kernels use `floor(x + 0.5)` on the
//! always-positive conversion outputs instead. In the reachable ranges
//! (f32 codes in [16, 240], f64 in [16, 235]) the operand and `x + 0.5`
//! are both exactly representable, so `floor(x + 0.5)` equals half-away
//! rounding bit-for-bit.
//!
//! Kernels are written at 512-bit logical width (u8x64/f64x8/f32x16);
//! magetypes polyfills v3/NEON/WASM128 down to their native widths.

/// Parallel engage point: inputs at or above this many pixels split into
/// `4·workers` chunks on the caller's rayon global pool. Chosen so a
/// 1024² image (~1MP, sub-millisecond even serial) stays serial while the
/// heavy 4K path parallelizes; measured small-input overhead is in
/// `benchmarks/psnr_opt_2026-10-20/grid.tsv`.
const PAR_MIN_PX: usize = 1 << 21;

/// Scalar per-byte squared error — the reference expression and the
/// kernel's remainder step.
#[inline]
fn sqdiff_u8(a: u8, b: u8) -> u64 {
    let d = a as i64 - b as i64;
    (d * d) as u64
}

/// SSE over packed u8 planes (any layout — it's a flat byte sum).
///
/// Returns the exact integer SSE. Caller guarantees
/// `a.len() == b.len()` and `a.len() * 65025 <= 2^53` so the result is
/// identical to the scalar `f64` sum.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(u8x64, i16x32, i32x16))]
fn rgb_sse_inner(token: Token, a: &[u8], b: &[u8]) -> u64 {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let mut total = 0u64;
    let mut i = 0usize;
    // Bounded i32 accumulator: each madd lane adds at most 2·255² = 130050,
    // so 16384 iterations stay under i32::MAX (≈2.13e9 / 130050 ≈ 16512).
    let mut block = i32x16::zero_t(token);
    let mut iters = 0usize;
    while i + 64 <= n {
        let va = u8x64::load_t(token, a[i..i + 64].try_into().unwrap());
        let vb = u8x64::load_t(token, b[i..i + 64].try_into().unwrap());
        let dlo = va.widen_low().bitcast_i16x32() - vb.widen_low().bitcast_i16x32();
        let dhi = va.widen_high().bitcast_i16x32() - vb.widen_high().bitcast_i16x32();
        // madd_adjacent(d, d): lane k = d[2k]² + d[2k+1]² (pmaddwd), exact.
        block = block + dlo.madd_adjacent(dlo) + dhi.madd_adjacent(dhi);
        iters += 1;
        i += 64;
        if iters == 16384 {
            for v in block.to_array() {
                total += v as u64;
            }
            block = i32x16::zero_t(token);
            iters = 0;
        }
    }
    for v in block.to_array() {
        total += v as u64;
    }
    while i < n {
        total += sqdiff_u8(a[i], b[i]);
        i += 1;
    }
    total
}

/// [`rgb_sse_inner`] dispatch. Exact integer SSE over equal-length slices.
/// Above [`PAR_MIN_PX`] pixels and with a multi-threaded caller pool, the
/// input is split at chunk boundaries and partial sums are combined in
/// order — integer addition is order-free, so this is still bit-exact.
pub(crate) fn rgb_sse(a: &[u8], b: &[u8]) -> u64 {
    if a.len() / 3 >= PAR_MIN_PX && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        // Byte-aligned chunks (any boundary is valid for a flat byte SSE).
        let chunk = ((a.len() / (rayon::current_num_threads() * 4)) + 63) & !63;
        return a
            .par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .map(|(x, y)| {
                archmage::attuned!(
                    rgb_sse_inner(x, y),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                )
            })
            .sum();
    }
    archmage::attuned!(
        rgb_sse_inner(a, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Scalar per-pixel squared error of the unrounded f64 luma — the
/// reference expression shared by the kernel remainder step.
#[inline]
fn luma_sqdiff_f64(a: &[u8], b: &[u8], cr: f64, cg: f64, cb: f64) -> f64 {
    let ya = cr * f64::from(a[0]) + cg * f64::from(a[1]) + cb * f64::from(a[2]);
    let yb = cr * f64::from(b[0]) + cg * f64::from(b[1]) + cb * f64::from(b[2]);
    let d = ya - yb;
    d * d
}

/// Ordered f64 SSE of unrounded luma for packed RGB8 (`3·n` bytes each).
///
/// The conversion and `delta²` are vectorised (each lane runs the scalar
/// f64 expression — element-wise ops are bit-identical); the sum keeps
/// the scalar left-to-right order, so the result is bit-exact vs the
/// reference `luma_mse` numerator.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn luma_sqerr_inner(token: Token, a: &[u8], b: &[u8], cr: f64, cg: f64, cb: f64) -> f64 {
    let n = a.len() / 3;
    let vcr = f64x8::splat_t(token, cr);
    let vcg = f64x8::splat_t(token, cg);
    let vcb = f64x8::splat_t(token, cb);
    let mut sum = 0.0f64;
    let mut px = 0usize;
    while px + 8 <= n {
        let base = px * 3;
        let ya = vcr
            * f64x8::from_array_t(token, core::array::from_fn(|j| f64::from(a[base + 3 * j])))
            + vcg
                * f64x8::from_array_t(
                    token,
                    core::array::from_fn(|j| f64::from(a[base + 3 * j + 1])),
                )
            + vcb
                * f64x8::from_array_t(
                    token,
                    core::array::from_fn(|j| f64::from(a[base + 3 * j + 2])),
                );
        let yb = vcr
            * f64x8::from_array_t(token, core::array::from_fn(|j| f64::from(b[base + 3 * j])))
            + vcg
                * f64x8::from_array_t(
                    token,
                    core::array::from_fn(|j| f64::from(b[base + 3 * j + 1])),
                )
            + vcb
                * f64x8::from_array_t(
                    token,
                    core::array::from_fn(|j| f64::from(b[base + 3 * j + 2])),
                );
        let d = ya - yb;
        // Ordered accumulation: same bits as the scalar iterator sum.
        for v in (d * d).to_array() {
            sum += v;
        }
        px += 8;
    }
    while px < n {
        sum += luma_sqdiff_f64(&a[3 * px..], &b[3 * px..], cr, cg, cb);
        px += 1;
    }
    sum
}

/// [`luma_sqerr_inner`] dispatch. `a`/`b` are packed RGB8 (len = 3·n).
/// Returns the numerator sum; caller divides by `n`.
pub(crate) fn luma_sqerr(a: &[u8], b: &[u8], cr: f64, cg: f64, cb: f64) -> f64 {
    archmage::attuned!(
        luma_sqerr_inner(a, b, cr, cg, cb),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Scalar studio-swing BT.601 Y code — the `PsnrYStudio601` reference
/// (f32 arithmetic, `round`, `clamp`, `as u8`).
#[inline]
fn studio601_y_code(x: &[u8]) -> u8 {
    (16.0 + (65.481 * x[0] as f32 + 128.553 * x[1] as f32 + 24.966 * x[2] as f32) / 255.0)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Integer SSE of the f32 studio-601 Y plane (the f32 variant kernel).
///
/// The conversion runs lane-wise with the scalar f32 op order; the code
/// error is integer, so the i32 lane accumulation drained into u64 is
/// bit-exact vs the scalar f64 sum (caller guards the 2^53 bound).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x16, i32x16))]
fn studio601_y_sse_inner(token: Token, a: &[u8], b: &[u8]) -> u64 {
    let n = a.len() / 3;
    let c0 = f32x16::splat_t(token, 65.481);
    let c1 = f32x16::splat_t(token, 128.553);
    let c2 = f32x16::splat_t(token, 24.966);
    let inv255 = f32x16::splat_t(token, 255.0);
    let off = f32x16::splat_t(token, 16.0);
    let half = f32x16::splat_t(token, 0.5);
    let lo = f32x16::splat_t(token, 0.0);
    let hi = f32x16::splat_t(token, 255.0);
    let mut total = 0u64;
    let mut block = i32x16::zero_t(token);
    let mut iters = 0usize;
    let mut px = 0usize;
    while px + 16 <= n {
        let base = px * 3;
        let ra = f32x16::from_array_t(token, core::array::from_fn(|j| a[base + 3 * j] as f32));
        let ga = f32x16::from_array_t(token, core::array::from_fn(|j| a[base + 3 * j + 1] as f32));
        let ba = f32x16::from_array_t(token, core::array::from_fn(|j| a[base + 3 * j + 2] as f32));
        let rb = f32x16::from_array_t(token, core::array::from_fn(|j| b[base + 3 * j] as f32));
        let gb = f32x16::from_array_t(token, core::array::from_fn(|j| b[base + 3 * j + 1] as f32));
        let bb = f32x16::from_array_t(token, core::array::from_fn(|j| b[base + 3 * j + 2] as f32));
        // floor(t + 0.5) == round-half-away for t in [16, 235] (see module
        // docs); the result is already inside the clamp range.
        let ca = ((off + (c0 * ra + c1 * ga + c2 * ba) / inv255) + half)
            .floor()
            .clamp(lo, hi)
            .to_i32();
        let cbv = ((off + (c0 * rb + c1 * gb + c2 * bb) / inv255) + half)
            .floor()
            .clamp(lo, hi)
            .to_i32();
        let d = ca - cbv;
        block += d * d;
        iters += 1;
        px += 16;
        if iters == 16384 {
            for v in block.to_array() {
                total += v as u64;
            }
            block = i32x16::zero_t(token);
            iters = 0;
        }
    }
    for v in block.to_array() {
        total += v as u64;
    }
    while px < n {
        let d =
            i64::from(studio601_y_code(&a[3 * px..])) - i64::from(studio601_y_code(&b[3 * px..]));
        total += (d * d) as u64;
        px += 1;
    }
    total
}

/// [`studio601_y_sse_inner`] dispatch. Exact integer SSE of the
/// u8-rounded studio-601 Y planes of two packed RGB8 images.
/// Parallelizes in pixel-aligned chunks on the caller's rayon pool (see
/// [`rgb_sse`]).
pub(crate) fn studio601_y_sse(a: &[u8], b: &[u8]) -> u64 {
    if a.len() / 3 >= PAR_MIN_PX && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let chunk = (a.len() / (rayon::current_num_threads() * 4)).div_ceil(48) * 48;
        return a
            .par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .map(|(x, y)| {
                archmage::attuned!(
                    studio601_y_sse_inner(x, y),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                )
            })
            .sum();
    }
    archmage::attuned!(
        studio601_y_sse_inner(a, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Scalar libvmaf Y code — the `PsnrYLibvmaf` reference (f64 normalised
/// BT.709, `round`, clamp to [16, 235], `as u8`).
#[inline]
fn libvmaf_y_code(x: &[u8]) -> u8 {
    let lum = 0.2126 * (f64::from(x[0]) / 255.0)
        + 0.7152 * (f64::from(x[1]) / 255.0)
        + 0.0722 * (f64::from(x[2]) / 255.0);
    (16.0 + 219.0 * lum).round().clamp(16.0, 235.0) as u8
}

/// Integer SSE of the f64 libvmaf Y plane.
///
/// Conversion lanes run the scalar f64 expression (incl. the `/255`
/// division — no reciprocal multiply). Codes are integral, so `diff`,
/// `diff²` and the lane-wise f64 sums stay exact; `reduce_add`'s pairing
/// is immaterial for integer values below 2^53 (caller-guarded).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn libvmaf_y_sse_inner(token: Token, a: &[u8], b: &[u8]) -> f64 {
    let n = a.len() / 3;
    let c0 = f64x8::splat_t(token, 0.2126);
    let c1 = f64x8::splat_t(token, 0.7152);
    let c2 = f64x8::splat_t(token, 0.0722);
    let inv255 = f64x8::splat_t(token, 255.0);
    let scale = f64x8::splat_t(token, 219.0);
    let off = f64x8::splat_t(token, 16.0);
    let half = f64x8::splat_t(token, 0.5);
    let lo = f64x8::splat_t(token, 16.0);
    let hi = f64x8::splat_t(token, 235.0);
    let mut acc = f64x8::zero_t(token);
    let mut px = 0usize;
    while px + 8 <= n {
        let base = px * 3;
        let luma = |p: &[u8]| {
            c0 * (f64x8::from_array_t(token, core::array::from_fn(|j| f64::from(p[base + 3 * j])))
                / inv255)
                + c1 * (f64x8::from_array_t(
                    token,
                    core::array::from_fn(|j| f64::from(p[base + 3 * j + 1])),
                ) / inv255)
                + c2 * (f64x8::from_array_t(
                    token,
                    core::array::from_fn(|j| f64::from(p[base + 3 * j + 2])),
                ) / inv255)
        };
        let code = |l: f64x8| (off + scale * l + half).floor().clamp(lo, hi);
        let d = code(luma(a)) - code(luma(b));
        acc += d * d;
        px += 8;
    }
    let mut sum = acc.reduce_add();
    while px < n {
        let d = f64::from(libvmaf_y_code(&a[3 * px..])) - f64::from(libvmaf_y_code(&b[3 * px..]));
        sum += d * d;
        px += 1;
    }
    sum
}

/// [`libvmaf_y_sse_inner`] dispatch. Returns the exact integer SSE as an
/// integral f64 (guaranteed while the total stays below 2^53). Parallel
/// chunks keep exactness: every partial is integral, so the ordered
/// combination below 2^53 is bit-identical (see [`rgb_sse`]).
pub(crate) fn libvmaf_y_sse(a: &[u8], b: &[u8]) -> f64 {
    if a.len() / 3 >= PAR_MIN_PX && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let chunk = (a.len() / (rayon::current_num_threads() * 4)).div_ceil(48) * 48;
        return a
            .par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .map(|(x, y)| {
                archmage::attuned!(
                    libvmaf_y_sse_inner(x, y),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                )
            })
            .sum();
    }
    archmage::attuned!(
        libvmaf_y_sse_inner(a, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Scalar studio-601 plane code: `plane` 0 = Y, 1 = Cb, 2 = Cr — the
/// `studio601_plane_mse` per-pixel conversion (f64).
#[inline]
fn studio601_plane_code(x: &[u8], plane: usize) -> f64 {
    let r = f64::from(x[0]);
    let g = f64::from(x[1]);
    let b = f64::from(x[2]);
    let lin = match plane {
        0 => (65.481 * r + 128.553 * g + 24.966 * b) / 255.0,
        1 => (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0,
        _ => (112.0 * r - 93.786 * g - 18.214 * b) / 255.0,
    };
    let offset = if plane == 0 { 16.0 } else { 128.0 };
    (offset + lin).round().clamp(0.0, 255.0)
}

/// Integer SSE of one f64 studio-601 plane (`plane`: 0/1/2 = Y/Cb/Cr).
///
/// Coefficients are baked per plane by the caller (`pick` in the scalar
/// version also discarded the other two linear parts); lanes run the
/// scalar expression and the integral accumulation is exact.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn studio601_plane_sse_inner(token: Token, a: &[u8], b: &[u8], plane: usize) -> f64 {
    let n = a.len() / 3;
    let (c0, c1, c2) = match plane {
        0 => (65.481, 128.553, 24.966),
        1 => (-37.797, -74.203, 112.0),
        _ => (112.0, -93.786, -18.214),
    };
    let offset = if plane == 0 { 16.0 } else { 128.0 };
    let vc0 = f64x8::splat_t(token, c0);
    let vc1 = f64x8::splat_t(token, c1);
    let vc2 = f64x8::splat_t(token, c2);
    let inv255 = f64x8::splat_t(token, 255.0);
    let voff = f64x8::splat_t(token, offset);
    let half = f64x8::splat_t(token, 0.5);
    let lo = f64x8::splat_t(token, 0.0);
    let hi = f64x8::splat_t(token, 255.0);
    let mut acc = f64x8::zero_t(token);
    let mut px = 0usize;
    while px + 8 <= n {
        let base = px * 3;
        let lin = |p: &[u8]| {
            (vc0 * f64x8::from_array_t(token, core::array::from_fn(|j| f64::from(p[base + 3 * j])))
                + vc1
                    * f64x8::from_array_t(
                        token,
                        core::array::from_fn(|j| f64::from(p[base + 3 * j + 1])),
                    )
                + vc2
                    * f64x8::from_array_t(
                        token,
                        core::array::from_fn(|j| f64::from(p[base + 3 * j + 2])),
                    ))
                / inv255
        };
        let code = |l: f64x8| (voff + l + half).floor().clamp(lo, hi);
        let d = code(lin(a)) - code(lin(b));
        acc += d * d;
        px += 8;
    }
    let mut sum = acc.reduce_add();
    while px < n {
        let d =
            studio601_plane_code(&a[3 * px..], plane) - studio601_plane_code(&b[3 * px..], plane);
        sum += d * d;
        px += 1;
    }
    sum
}

/// [`studio601_plane_sse_inner`] dispatch for the Cb/Cr routes.
/// `plane` is 1 (Cb) or 2 (Cr). Returns the exact integer SSE as f64.
/// Parallel chunks combine integrally — still bit-exact (see [`rgb_sse`]).
pub(crate) fn studio601_chroma_sse(a: &[u8], b: &[u8], plane: usize) -> f64 {
    if a.len() / 3 >= PAR_MIN_PX && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let chunk = (a.len() / (rayon::current_num_threads() * 4)).div_ceil(48) * 48;
        return a
            .par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .map(|(x, y)| {
                archmage::attuned!(
                    studio601_plane_sse_inner(x, y, plane),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                )
            })
            .sum();
    }
    archmage::attuned!(
        studio601_plane_sse_inner(a, b, plane),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// All three studio-601 plane SSEs in one pass — the `PsnrYCbCr611`
/// ingress (one deinterleave feeds the Y, Cb and Cr chains).
///
/// Returns the three exact integer sums as integral f64s; the caller
/// keeps the scalar per-plane division and 6:1:1 weighting.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f64x8))]
fn studio601_yuv_sse_inner(token: Token, a: &[u8], b: &[u8]) -> (f64, f64, f64) {
    let n = a.len() / 3;
    let inv255 = f64x8::splat_t(token, 255.0);
    let half = f64x8::splat_t(token, 0.5);
    let lo = f64x8::splat_t(token, 0.0);
    let hi = f64x8::splat_t(token, 255.0);
    let cy = [
        f64x8::splat_t(token, 65.481),
        f64x8::splat_t(token, 128.553),
        f64x8::splat_t(token, 24.966),
    ];
    let ccb = [
        f64x8::splat_t(token, -37.797),
        f64x8::splat_t(token, -74.203),
        f64x8::splat_t(token, 112.0),
    ];
    let ccr = [
        f64x8::splat_t(token, 112.0),
        f64x8::splat_t(token, -93.786),
        f64x8::splat_t(token, -18.214),
    ];
    let off_y = f64x8::splat_t(token, 16.0);
    let off_c = f64x8::splat_t(token, 128.0);
    let mut acc_y = f64x8::zero_t(token);
    let mut acc_cb = f64x8::zero_t(token);
    let mut acc_cr = f64x8::zero_t(token);
    let mut px = 0usize;
    while px + 8 <= n {
        let base = px * 3;
        let rgb = |p: &[u8]| {
            (
                f64x8::from_array_t(token, core::array::from_fn(|j| f64::from(p[base + 3 * j]))),
                f64x8::from_array_t(
                    token,
                    core::array::from_fn(|j| f64::from(p[base + 3 * j + 1])),
                ),
                f64x8::from_array_t(
                    token,
                    core::array::from_fn(|j| f64::from(p[base + 3 * j + 2])),
                ),
            )
        };
        let (ra, ga, ba) = rgb(a);
        let (rb, gb, bb) = rgb(b);
        let code = |off: f64x8, l: f64x8| (off + l + half).floor().clamp(lo, hi);
        let dy = code(off_y, (cy[0] * ra + cy[1] * ga + cy[2] * ba) / inv255)
            - code(off_y, (cy[0] * rb + cy[1] * gb + cy[2] * bb) / inv255);
        let dcb = code(off_c, (ccb[0] * ra + ccb[1] * ga + ccb[2] * ba) / inv255)
            - code(off_c, (ccb[0] * rb + ccb[1] * gb + ccb[2] * bb) / inv255);
        let dcr = code(off_c, (ccr[0] * ra + ccr[1] * ga + ccr[2] * ba) / inv255)
            - code(off_c, (ccr[0] * rb + ccr[1] * gb + ccr[2] * bb) / inv255);
        acc_y += dy * dy;
        acc_cb += dcb * dcb;
        acc_cr += dcr * dcr;
        px += 8;
    }
    let mut sy = acc_y.reduce_add();
    let mut scb = acc_cb.reduce_add();
    let mut scr = acc_cr.reduce_add();
    while px < n {
        // Same per-pixel scalar conversion as `yuv444_studio601_mse`.
        let (r1, g1, b1) = (
            f64::from(a[3 * px]),
            f64::from(a[3 * px + 1]),
            f64::from(a[3 * px + 2]),
        );
        let (r2, g2, b2) = (
            f64::from(b[3 * px]),
            f64::from(b[3 * px + 1]),
            f64::from(b[3 * px + 2]),
        );
        let y1 = (16.0 + (65.481 * r1 + 128.553 * g1 + 24.966 * b1) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let y2 = (16.0 + (65.481 * r2 + 128.553 * g2 + 24.966 * b2) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let cb1 = (128.0 + (-37.797 * r1 - 74.203 * g1 + 112.0 * b1) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let cb2 = (128.0 + (-37.797 * r2 - 74.203 * g2 + 112.0 * b2) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let cr1 = (128.0 + (112.0 * r1 - 93.786 * g1 - 18.214 * b1) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let cr2 = (128.0 + (112.0 * r2 - 93.786 * g2 - 18.214 * b2) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let (dy, dcb, dcr) = (y1 - y2, cb1 - cb2, cr1 - cr2);
        sy += dy * dy;
        scb += dcb * dcb;
        scr += dcr * dcr;
        px += 1;
    }
    (sy, scb, scr)
}

/// [`studio601_yuv_sse_inner`] dispatch. Returns `(sse_y, sse_cb,
/// sse_cr)` as exact integral f64s (caller guards the 2^53 bound).
/// Parallel chunks combine integrally per plane — bit-exact (see
/// [`rgb_sse`]).
pub(crate) fn studio601_yuv_sse(a: &[u8], b: &[u8]) -> (f64, f64, f64) {
    if a.len() / 3 >= PAR_MIN_PX && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let chunk = (a.len() / (rayon::current_num_threads() * 4)).div_ceil(48) * 48;
        return a
            .par_chunks(chunk)
            .zip(b.par_chunks(chunk))
            .map(|(x, y)| {
                archmage::attuned!(
                    studio601_yuv_sse_inner(x, y),
                    [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
                )
            })
            .reduce(|| (0.0, 0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2));
    }
    archmage::attuned!(
        studio601_yuv_sse_inner(a, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Scalar references — the frozen oracle expressions the SIMD kernels
/// must reproduce bit-for-bit. Kept for tests and A/B measurement.
#[cfg(test)]
pub(crate) mod reference {
    /// Sequential f64 SSE over packed bytes (`Kind::Psnr` numerator).
    pub(crate) fn rgb_sse(a: &[u8], b: &[u8]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = f64::from(x) - f64::from(y);
                d * d
            })
            .sum::<f64>()
    }

    /// Sequential f64 luma squared-error sum (`luma_mse` numerator).
    pub(crate) fn luma_sqerr(a: &[u8], b: &[u8], cr: f64, cg: f64, cb: f64) -> f64 {
        a.as_chunks::<3>()
            .0
            .iter()
            .zip(b.as_chunks::<3>().0)
            .map(|(x, y)| {
                let ya = cr * f64::from(x[0]) + cg * f64::from(x[1]) + cb * f64::from(x[2]);
                let yb = cr * f64::from(y[0]) + cg * f64::from(y[1]) + cb * f64::from(y[2]);
                let d = ya - yb;
                d * d
            })
            .sum()
    }

    /// Studio-601 Y SSE via the f32 u8 code (`PsnrYStudio601` numerator).
    pub(crate) fn studio601_y_sse(a: &[u8], b: &[u8]) -> f64 {
        a.as_chunks::<3>()
            .0
            .iter()
            .zip(b.as_chunks::<3>().0)
            .map(|(x, y)| {
                let code = |p: &[u8]| {
                    f64::from(
                        (16.0
                            + (65.481 * p[0] as f32 + 128.553 * p[1] as f32 + 24.966 * p[2] as f32)
                                / 255.0)
                            .round()
                            .clamp(0.0, 255.0) as u8,
                    )
                };
                let d = code(x) - code(y);
                d * d
            })
            .sum()
    }

    /// libvmaf Y SSE (`PsnrYLibvmaf` numerator).
    pub(crate) fn libvmaf_y_sse(a: &[u8], b: &[u8]) -> f64 {
        a.as_chunks::<3>()
            .0
            .iter()
            .zip(b.as_chunks::<3>().0)
            .map(|(x, y)| {
                let code = |p: &[u8]| {
                    let lum = 0.2126 * (f64::from(p[0]) / 255.0)
                        + 0.7152 * (f64::from(p[1]) / 255.0)
                        + 0.0722 * (f64::from(p[2]) / 255.0);
                    f64::from((16.0 + 219.0 * lum).round().clamp(16.0, 235.0) as u8)
                };
                let d = code(x) - code(y);
                d * d
            })
            .sum()
    }

    /// Single studio-601 plane SSE (`studio601_plane_mse` numerator).
    pub(crate) fn studio601_plane_sse(a: &[u8], b: &[u8], plane: usize) -> f64 {
        a.as_chunks::<3>()
            .0
            .iter()
            .zip(b.as_chunks::<3>().0)
            .map(|(x, y)| {
                let lin = |p: &[u8]| {
                    let (r, g, bl) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
                    match plane {
                        0 => (65.481 * r + 128.553 * g + 24.966 * bl) / 255.0,
                        1 => (-37.797 * r - 74.203 * g + 112.0 * bl) / 255.0,
                        _ => (112.0 * r - 93.786 * g - 18.214 * bl) / 255.0,
                    }
                };
                let off = if plane == 0 { 16.0 } else { 128.0 };
                let cx = (off + lin(x)).round().clamp(0.0, 255.0);
                let cy = (off + lin(y)).round().clamp(0.0, 255.0);
                let d = cx - cy;
                d * d
            })
            .sum()
    }

    /// Three-plane studio-601 SSE (`yuv444_studio601_mse` numerators).
    pub(crate) fn studio601_yuv_sse(a: &[u8], b: &[u8]) -> (f64, f64, f64) {
        let (mut sy, mut scb, mut scr) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.as_chunks::<3>().0.iter().zip(b.as_chunks::<3>().0) {
            let (r1, g1, b1) = (f64::from(x[0]), f64::from(x[1]), f64::from(x[2]));
            let (r2, g2, b2) = (f64::from(y[0]), f64::from(y[1]), f64::from(y[2]));
            let y1 = (16.0 + (65.481 * r1 + 128.553 * g1 + 24.966 * b1) / 255.0)
                .round()
                .clamp(0.0, 255.0);
            let y2 = (16.0 + (65.481 * r2 + 128.553 * g2 + 24.966 * b2) / 255.0)
                .round()
                .clamp(0.0, 255.0);
            let cb1 = (128.0 + (-37.797 * r1 - 74.203 * g1 + 112.0 * b1) / 255.0)
                .round()
                .clamp(0.0, 255.0);
            let cb2 = (128.0 + (-37.797 * r2 - 74.203 * g2 + 112.0 * b2) / 255.0)
                .round()
                .clamp(0.0, 255.0);
            let cr1 = (128.0 + (112.0 * r1 - 93.786 * g1 - 18.214 * b1) / 255.0)
                .round()
                .clamp(0.0, 255.0);
            let cr2 = (128.0 + (112.0 * r2 - 93.786 * g2 - 18.214 * b2) / 255.0)
                .round()
                .clamp(0.0, 255.0);
            let (dy, dcb, dcr) = (y1 - y2, cb1 - cb2, cr1 - cr2);
            sy += dy * dy;
            scb += dcb * dcb;
            scr += dcr * dcr;
        }
        (sy, scb, scr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random pair covering the tail and interior.
    fn pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
        let n = w * h * 3;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut s = 0x9e3779b9u32;
        for i in 0..n {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let v = (s ^ i as u32) as u8;
            a.push(v);
            b.push(v.wrapping_add((s >> 8) as u8));
        }
        (a, b)
    }

    fn assert_bits(got: f64, want: f64, what: &str) {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{what}: {got:.17e} != {want:.17e}"
        );
    }

    /// Every kernel against its scalar reference across sizes that
    /// exercise the vector body and every remainder length.
    #[test]
    fn simd_matches_scalar_reference_all_tails() {
        // widths chosen so 3·w straddles the 64-byte and 16/8-pixel chunks.
        for &(w, h) in &[
            (1, 1),
            (5, 1),
            (7, 3),
            (16, 2),
            (21, 1),
            (22, 1),
            (23, 2),
            (31, 1),
            (32, 1),
            (43, 5),
            (64, 1),
            (65, 3),
            (128, 2),
            (137, 3),
        ] {
            let (a, b) = pair(w, h);
            assert_bits(
                rgb_sse(&a, &b) as f64,
                reference::rgb_sse(&a, &b),
                &format!("rgb_sse {w}x{h}"),
            );
            assert_bits(
                luma_sqerr(&a, &b, 0.2126, 0.7152, 0.0722),
                reference::luma_sqerr(&a, &b, 0.2126, 0.7152, 0.0722),
                &format!("luma709 {w}x{h}"),
            );
            assert_bits(
                luma_sqerr(&a, &b, 0.299, 0.587, 0.114),
                reference::luma_sqerr(&a, &b, 0.299, 0.587, 0.114),
                &format!("luma601 {w}x{h}"),
            );
            assert_bits(
                studio601_y_sse(&a, &b) as f64,
                reference::studio601_y_sse(&a, &b),
                &format!("studio601y {w}x{h}"),
            );
            assert_bits(
                libvmaf_y_sse(&a, &b),
                reference::libvmaf_y_sse(&a, &b),
                &format!("libvmaf_y {w}x{h}"),
            );
            for plane in [1, 2] {
                assert_bits(
                    studio601_chroma_sse(&a, &b, plane),
                    reference::studio601_plane_sse(&a, &b, plane),
                    &format!("chroma{plane} {w}x{h}"),
                );
            }
            let (gy, gcb, gcr) = studio601_yuv_sse(&a, &b);
            let (ry, rcb, rcr) = reference::studio601_yuv_sse(&a, &b);
            assert_bits(gy, ry, &format!("yuv-y {w}x{h}"));
            assert_bits(gcb, rcb, &format!("yuv-cb {w}x{h}"));
            assert_bits(gcr, rcr, &format!("yuv-cr {w}x{h}"));
        }
    }

    /// Above `PAR_MIN_PX` the integer kernels chunk across the rayon
    /// global pool (32 threads here). Partial integer sums combine in
    /// chunk order — still bit-exact vs the scalar reference.
    #[test]
    fn parallel_chunking_matches_scalar() {
        let (a, b) = pair(2048, 1152); // 2.36M px > PAR_MIN_PX
        assert_eq!(rayon::current_num_threads() > 1, true);
        assert_bits(
            rgb_sse(&a, &b) as f64,
            reference::rgb_sse(&a, &b),
            "rgb_sse par",
        );
        assert_bits(
            studio601_y_sse(&a, &b) as f64,
            reference::studio601_y_sse(&a, &b),
            "studio601y par",
        );
        assert_bits(
            libvmaf_y_sse(&a, &b),
            reference::libvmaf_y_sse(&a, &b),
            "libvmaf par",
        );
        for plane in [1, 2] {
            assert_bits(
                studio601_chroma_sse(&a, &b, plane),
                reference::studio601_plane_sse(&a, &b, plane),
                "chroma par",
            );
        }
        let (gy, gcb, gcr) = studio601_yuv_sse(&a, &b);
        let (ry, rcb, rcr) = reference::studio601_yuv_sse(&a, &b);
        assert_bits(gy, ry, "yuv-y par");
        assert_bits(gcb, rcb, "yuv-cb par");
        assert_bits(gcr, rcr, "yuv-cr par");
    }

    /// Identity and full-extremes inputs (infinite PSNR boundary and the
    /// largest possible per-pixel error).
    #[test]
    fn extremes_and_identity() {
        for &(w, h) in &[(1, 1), (21, 1), (64, 4), (333, 7)] {
            let (a, _) = pair(w, h);
            let n = w * h * 3;
            assert_eq!(rgb_sse(&a, &a), 0);
            assert_bits(luma_sqerr(&a, &a, 0.2126, 0.7152, 0.0722), 0.0, "luma id");
            assert_eq!(studio601_y_sse(&a, &a), 0);
            assert_bits(libvmaf_y_sse(&a, &a), 0.0, "libvmaf id");
            let inv: Vec<u8> = a.iter().map(|&v| 255 - v).collect();
            assert_eq!(rgb_sse(&a, &inv), reference::rgb_sse(&a, &inv) as u64);
            let black = vec![0u8; n];
            let white = vec![255u8; n];
            assert_eq!(rgb_sse(&black, &white), 65025 * n as u64);
        }
    }
}
