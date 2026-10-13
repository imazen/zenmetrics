//! SIMD reduction for the scale-wise mean squared distance.

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn squared_difference_inner(token: Token, a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let (ac, at) = f32x8::partition_slice_t(token, a);
    let (bc, bt) = f32x8::partition_slice_t(token, b);
    let mut sum = f32x8::zero_t(token);
    for (aa, bb) in ac.iter().zip(bc) {
        let d = f32x8::load_t(token, aa) - f32x8::load_t(token, bb);
        sum = d.mul_add(d, sum);
    }
    let mut result = sum.to_array().into_iter().map(f64::from).sum::<f64>();
    for (&a, &b) in at.iter().zip(bt) {
        let d = a - b;
        result += f64::from(d * d);
    }
    result
}

pub(crate) fn squared_difference(a: &[f32], b: &[f32]) -> f64 {
    archmage::attuned!(
        squared_difference_inner(a, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn filter5_vertical_inner(
    token: Token,
    src: &[f32],
    width: usize,
    height: usize,
    dst: &mut [f32],
    _out_height: usize,
    stride: usize,
    pad: usize,
) {
    let weights = [0.05, 0.25, 0.4, 0.25, 0.05].map(|v| f32x8::splat_t(token, v));
    super::for_each_row(dst, width, |oy, dst_row| {
        let rows = core::array::from_fn::<_, 5, _>(|ky| {
            let sy = super::reflect((oy * stride + ky) as isize - pad as isize, height);
            &src[sy * width..(sy + 1) * width]
        });
        let (dst_chunks, dst_tail) = f32x8::partition_slice_mut_t(token, dst_row);
        let row_parts = rows.map(|r| f32x8::partition_slice_t(token, r));
        for (i, chunk) in dst_chunks.iter_mut().enumerate() {
            let mut sum = f32x8::zero_t(token);
            for k in 0..5 {
                sum = f32x8::load_t(token, &row_parts[k].0[i]).mul_add(weights[k], sum);
            }
            sum.store(chunk);
        }
        for (x, d) in dst_tail.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for k in 0..5 {
                sum += row_parts[k].1[x] * [0.05, 0.25, 0.4, 0.25, 0.05][k];
            }
            *d = sum;
        }
    });
}

pub(crate) fn filter5_vertical(
    src: &[f32],
    width: usize,
    height: usize,
    dst: &mut [f32],
    out_height: usize,
    stride: usize,
    pad: usize,
) {
    // Callers pass level-0-sized scratch slabs for `dst` — iterate only
    // this level's out_height×width subregion (row chunking keys off it).
    let dst = &mut dst[..out_height * width];
    archmage::attuned!(
        filter5_vertical_inner(src, width, height, dst, out_height, stride, pad),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Fused single-pass 5-tap horizontal conv: `out[i] = Σ_k src[i+k]·w_k`.
/// Bit-identical to the 5-pass accumulate form — each product is rounded
/// separately, then added in k order (no FMA). `ZINIT` replicates the
/// decimate path's `0.0`-initialised accumulator (the `0 + v0·w0` add
/// flushes a negative-zero first product to +0, matching the scalar `+=`
/// onto a cleared slot). Caller guarantees `src.len() >= out.len() + 4`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn h5_fused_inner<const ZINIT: bool>(token: Token, src: &[f32], out: &mut [f32]) {
    debug_assert!(src.len() >= out.len() + 4);
    let w = super::LOWPASS.map(|v| f32x8::splat_t(token, v));
    let span = out.len();
    let full = span & !7;
    for i in (0..full).step_by(8) {
        let mut acc = f32x8::load_t(token, src[i..i + 8].try_into().unwrap()) * w[0];
        if ZINIT {
            acc = acc + f32x8::zero_t(token);
        }
        for k in 1..5 {
            acc = acc + f32x8::load_t(token, src[i + k..i + k + 8].try_into().unwrap()) * w[k];
        }
        acc.store((&mut out[i..i + 8]).try_into().unwrap());
    }
    for (i, o) in out.iter_mut().enumerate().skip(full) {
        let mut acc = src[i] * super::LOWPASS[0];
        if ZINIT {
            acc += 0.0;
        }
        for k in 1..5 {
            acc += src[i + k] * super::LOWPASS[k];
        }
        *o = acc;
    }
}

/// `h5_fused_inner` dispatch: `out[i] = Σ_k src[i+k]·LOWPASS[k]`,
/// `src.len() >= out.len() + 4`.
pub(crate) fn h5_fused<const ZINIT: bool>(src: &[f32], out: &mut [f32]) {
    archmage::attuned!(
        h5_fused_inner::<ZINIT>(src, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Horizontal lerp of one source row: `out[x] = row[x0]·(1−tx) +
/// row[x1]·tx`. Two lane-gathers per output; per-lane arithmetic
/// identical to the scalar formula → bit-exact.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn hlerp_row_inner(
    token: Token,
    row: &[f32],
    x0v: &[usize],
    x1v: &[usize],
    txv: &[f32],
    out: &mut [f32],
) {
    let one = f32x8::splat_t(token, 1.0);
    let span = out.len();
    let full = span & !7;
    for x in (0..full).step_by(8) {
        let g = |ix: &[usize]| {
            f32x8::from_array_t(token, core::array::from_fn(|j| row[ix[j]]))
        };
        let tx = f32x8::load_t(token, txv[x..x + 8].try_into().unwrap());
        (g(&x0v[x..x + 8]) * (one - tx) + g(&x1v[x..x + 8]) * tx)
            .store((&mut out[x..x + 8]).try_into().unwrap());
    }
    for x in full..span {
        let tx = txv[x];
        out[x] = row[x0v[x]] * (1.0 - tx) + row[x1v[x]] * tx;
    }
}

pub(crate) fn hlerp_row(row: &[f32], x0v: &[usize], x1v: &[usize], txv: &[f32], out: &mut [f32]) {
    archmage::attuned!(
        hlerp_row_inner(row, x0v, x1v, txv, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// `out[x] = a[x]·(1−ty) + b[x]·ty` — the vertical blend of two
/// pre-lerped rows. Same per-lane arithmetic as the scalar loop.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn blend_rows_inner(token: Token, a: &[f32], b: &[f32], ty: f32, out: &mut [f32]) {
    let tyv = f32x8::splat_t(token, ty);
    let omty = f32x8::splat_t(token, 1.0 - ty);
    let span = out.len();
    let full = span & !7;
    for x in (0..full).step_by(8) {
        let av = f32x8::load_t(token, a[x..x + 8].try_into().unwrap());
        let bv = f32x8::load_t(token, b[x..x + 8].try_into().unwrap());
        (av * omty + bv * tyv).store((&mut out[x..x + 8]).try_into().unwrap());
    }
    for x in full..span {
        out[x] = a[x] * (1.0 - ty) + b[x] * ty;
    }
}

pub(crate) fn blend_rows(a: &[f32], b: &[f32], ty: f32, out: &mut [f32]) {
    archmage::attuned!(
        blend_rows_inner(a, b, ty, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Interleaved→planar u8 channel extract: `dst[i] = src[3i+c] / 255`.
/// i32-widened lanes keep `u8 → /255` arithmetic identical to scalar.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8, i32x8))]
fn deinterleave_u8_ch_inner(token: Token, src: &[u8], c: usize, dst: &mut [f32]) {
    let inv = f32x8::splat_t(token, 255.0);
    let n = dst.len();
    let full = n & !7;
    for i in (0..full).step_by(8) {
        let arr = core::array::from_fn(|j| src[3 * (i + j) + c] as i32);
        let v = i32x8::from_array_t(token, arr).to_f32() / inv;
        v.store((&mut dst[i..i + 8]).try_into().unwrap());
    }
    for (i, o) in dst.iter_mut().enumerate().skip(full) {
        *o = src[3 * i + c] as f32 / 255.0;
    }
}

pub(crate) fn deinterleave_u8_ch(src: &[u8], c: usize, dst: &mut [f32]) {
    archmage::attuned!(
        deinterleave_u8_ch_inner(src, c, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Interleaved→planar f32 channel extract: `dst[i] = src[3i+c]`.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn deinterleave_f32_ch_inner(token: Token, src: &[f32], c: usize, dst: &mut [f32]) {
    let n = dst.len();
    let full = n & !7;
    for i in (0..full).step_by(8) {
        let v = f32x8::from_array_t(token, core::array::from_fn(|j| src[3 * (i + j) + c]));
        v.store((&mut dst[i..i + 8]).try_into().unwrap());
    }
    for (i, o) in dst.iter_mut().enumerate().skip(full) {
        *o = src[3 * i + c];
    }
}

pub(crate) fn deinterleave_f32_ch(src: &[f32], c: usize, dst: &mut [f32]) {
    archmage::attuned!(
        deinterleave_f32_ch_inner(src, c, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// Each output band retains a 5-row ring of H-blurred `up` rows (slot
// row%5) feeding a 3-row ring of Laplacian rows (slot row%3) — the H
// conv runs inside this pass so the `horizontal` plane is never
// materialized. The vertical blur uses the same five FMAs and scalar
// tail as filter5_vertical; normalization uses ordered non-fused
// additions and an exact divide. Bit-identical to filtering into a
// scratch plane first: the ring rows are produced by the same ops.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn normalize_filtered_inner(
    token: Token,
    original: &[f32],
    up: &[f32],
    w: usize,
    h: usize,
    level: usize,
    dst: &mut [f32],
) {
    let weights = super::LOWPASS.map(|v| f32x8::splat_t(token, v));
    let [top, left, right, bottom] = super::DN[level];
    let sigma = super::SIGMAS[level];
    let tv = f32x8::splat_t(token, top);
    let lv = f32x8::splat_t(token, left);
    let rv = f32x8::splat_t(token, right);
    let bv = f32x8::splat_t(token, bottom);
    let sv = f32x8::splat_t(token, sigma);
    struct Rings {
        hrows: [Vec<f32>; 5],
        hidx: [usize; 5],
        lap: [Vec<f32>; 3],
        lidx: [usize; 3],
    }
    super::for_each_row_init_min(
        dst,
        w,
        16,
        || Rings {
            hrows: core::array::from_fn(|_| vec![0.0; w]),
            hidx: [usize::MAX; 5],
            lap: core::array::from_fn(|_| vec![0.0; w]),
            lidx: [usize::MAX; 3],
        },
        |y, out, rings| {
            // Materialise the three lap rows this output row needs.
            for r in [
                super::reflect(y as isize - 1, h),
                y,
                super::reflect(y as isize + 1, h),
            ] {
                let slot = r % 3;
                if rings.lidx[slot] == r {
                    continue;
                }
                // Lap row r needs H rows reflect(r-2..r+2) — 5
                // consecutive indices, one full residency of the ring.
                for k in 0..5usize {
                    let hr = super::reflect(r as isize + k as isize - 2, h);
                    let hslot = hr % 5;
                    if rings.hidx[hslot] != hr {
                        super::filter5_row(
                            &up[hr * w..(hr + 1) * w],
                            w,
                            &mut rings.hrows[hslot],
                            2,
                            w - 2,
                        );
                        rings.hidx[hslot] = hr;
                    }
                }
                let rows: [&[f32]; 5] = core::array::from_fn(|k| {
                    &rings.hrows[super::reflect(r as isize + k as isize - 2, h) % 5][..]
                });
                let out_lap = &mut rings.lap[slot];
                let (chunks, tail) = f32x8::partition_slice_mut_t(token, out_lap);
                let row_parts = rows.map(|r| f32x8::partition_slice_t(token, r));
                let original_parts =
                    f32x8::partition_slice_t(token, &original[r * w..(r + 1) * w]);
                for (i, chunk) in chunks.iter_mut().enumerate() {
                    let mut sum = f32x8::zero_t(token);
                    for k in 0..5 {
                        let values = &row_parts[k].0[i];
                        sum = f32x8::load_t(token, values).mul_add(weights[k], sum);
                    }
                    let values = &original_parts.0[i];
                    (f32x8::load_t(token, values) - sum).store(chunk);
                }
                let base = w / 8 * 8;
                for (i, o) in tail.iter_mut().enumerate() {
                    let x = base + i;
                    let mut sum = 0.0;
                    for k in 0..5 {
                        sum += rows[k][x] * super::LOWPASS[k];
                    }
                    *o = original[r * w + x] - sum;
                }
                rings.lidx[slot] = r;
            }
            let a = &rings.lap[super::reflect(y as isize - 1, h) % 3];
            let mid = &rings.lap[y % 3];
            let b = &rings.lap[super::reflect(y as isize + 1, h) % 3];
            let scalar = |x: usize| {
                let xm = super::reflect(x as isize - 1, w);
                let xp = super::reflect(x as isize + 1, w);
                let denom = sigma
                    + top * a[x].abs()
                    + left * mid[xm].abs()
                    + right * mid[xp].abs()
                    + bottom * b[x].abs();
                mid[x] / denom
            };
            out[0] = scalar(0);
            out[w - 1] = scalar(w - 1);
            let (chunks, tail) = f32x8::partition_slice_mut_t(token, &mut out[1..w - 1]);
            for (i, chunk) in chunks.iter_mut().enumerate() {
                let x = 1 + i * 8;
                let load =
                    |r: &[f32], x: usize| f32x8::load_t(token, r[x..x + 8].try_into().unwrap());
                let denom = sv
                    + tv * load(a, x).abs()
                    + lv * load(mid, x - 1).abs()
                    + rv * load(mid, x + 1).abs()
                    + bv * load(b, x).abs();
                (load(mid, x) / denom).store(chunk);
            }
            let base = 1 + (w - 2) / 8 * 8;
            for (i, out) in tail.iter_mut().enumerate() {
                *out = scalar(base + i);
            }
        },
    );
}

pub(crate) fn normalize_filtered(
    original: &[f32],
    up: &[f32],
    w: usize,
    h: usize,
    level: usize,
    dst: &mut [f32],
) {
    archmage::attuned!(
        normalize_filtered_inner(original, up, w, h, level, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}
