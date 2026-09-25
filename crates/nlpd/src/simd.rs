//! SIMD reduction for the scale-wise mean squared distance.

#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
fn squared_difference_inner(token: Token, a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let (ac, at) = f32x8::partition_slice(token, a);
    let (bc, bt) = f32x8::partition_slice(token, b);
    let mut sum = f32x8::zero(token);
    for (aa, bb) in ac.iter().zip(bc) {
        let d = f32x8::load(token, aa) - f32x8::load(token, bb);
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
    archmage::incant!(
        squared_difference_inner(a, b),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
fn filter5_vertical_inner(
    token: Token,
    src: &[f32],
    width: usize,
    height: usize,
    dst: &mut [f32],
    out_height: usize,
    stride: usize,
    pad: usize,
) {
    let weights = [0.05, 0.25, 0.4, 0.25, 0.05].map(|v| f32x8::splat(token, v));
    for oy in 0..out_height {
        let rows = core::array::from_fn::<_, 5, _>(|ky| {
            let sy = super::reflect((oy * stride + ky) as isize - pad as isize, height);
            &src[sy * width..(sy + 1) * width]
        });
        let dst_row = &mut dst[oy * width..(oy + 1) * width];
        let (dst_chunks, dst_tail) = f32x8::partition_slice_mut(token, dst_row);
        let row_parts = rows.map(|r| f32x8::partition_slice(token, r));
        for (i, chunk) in dst_chunks.iter_mut().enumerate() {
            let mut sum = f32x8::zero(token);
            for k in 0..5 {
                sum = f32x8::load(token, &row_parts[k].0[i]).mul_add(weights[k], sum);
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
    }
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
    archmage::incant!(
        filter5_vertical_inner(src, width, height, dst, out_height, stride, pad),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}
