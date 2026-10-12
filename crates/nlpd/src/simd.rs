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

// Each output band retains three Laplacian rows. The vertical blur uses
// the same five FMAs and scalar tail as filter5_vertical; normalization
// uses ordered non-fused additions and an exact divide.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn normalize_filtered_inner(
    token: Token,
    original: &[f32],
    horizontal: &[f32],
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
    let lap_row = |y: usize, out: &mut [f32]| {
        let rows: [&[f32]; 5] = core::array::from_fn(|k| {
            let sy = super::reflect(y as isize + k as isize - 2, h);
            &horizontal[sy * w..(sy + 1) * w]
        });
        let (chunks, tail) = f32x8::partition_slice_mut_t(token, out);
        let row_parts = rows.map(|r| f32x8::partition_slice_t(token, r));
        let original_parts = f32x8::partition_slice_t(token, &original[y * w..(y + 1) * w]);
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
        for (i, out) in tail.iter_mut().enumerate() {
            let x = base + i;
            let mut sum = 0.0;
            for k in 0..5 {
                sum += rows[k][x] * super::LOWPASS[k];
            }
            *out = original[y * w + x] - sum;
        }
    };
    super::for_each_row_init_min(
        dst,
        w,
        32,
        || (None, core::array::from_fn::<_, 3, _>(|_| vec![0.0; w])),
        |y, out, (last, rows)| {
            if last.is_none() {
                for (row, offset) in rows.iter_mut().zip([-1, 0, 1]) {
                    lap_row(super::reflect(y as isize + offset, h), row);
                }
            } else {
                rows.rotate_left(1);
                lap_row(super::reflect(y as isize + 1, h), &mut rows[2]);
            }
            *last = Some(y);
            let [a, mid, b] = rows;
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
    horizontal: &[f32],
    w: usize,
    h: usize,
    level: usize,
    dst: &mut [f32],
) {
    archmage::attuned!(
        normalize_filtered_inner(original, horizontal, w, h, level, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}
