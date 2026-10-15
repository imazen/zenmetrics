//! TensorFlow `scale_and_translate` resampling for `tf.image.resize`'s
//! GAUSSIAN method, as MUSIQ's preprocessor uses it
//! (`tf2.image.resize(.., GAUSSIAN)` reaches `gen_image_ops.scale_and_translate`
//! with `kernel_type="gaussian"` and `antialias=False`, so
//! `kernel_scale = 1` — a fixed σ = 0.5, radius 1.5 kernel, no antialiasing).
//!
//! Per `tensorflow/core/kernels/image/scale_and_translate_op.cc`
//! (`ComputeSpansCore`) and `sampling_kernels.h` (`GaussianKernelFunc`),
//! reproduced here in f32:
//! - `inv_scale = 1 / (out/in)`, sample centre `(i + 0.5)·inv_scale`;
//! - span `[⌈sf − 2⌉, ⌊sf + 1⌋]` clamped to `[0, in)`;
//! - tap `w(src) = exp(−2·(src + 0.5 − sf)²)` when `|src + 0.5 − sf| < 1.5`,
//!   else 0 (the `x >= radius` cutoff; TF's `std::exp` runs in f64 —
//!   `2.0 * sigma * sigma` promotes — so the exp is f64, cast to f32);
//! - sequential f32 weight sum, then each weight `*= 1/total` (reciprocal
//!   multiply, not division);
//! - vertical pass before horizontal (the op's process-rows order); each
//!   output accumulates `Σ w·in` sequentially in f32.
//!
//! Verified against `tf.image.resize(.., GAUSSIAN)` (tensorflow 2.15 CPU):
//! bitwise-equal at 8→4, ≤ 4e-7 absolute at 256→224/512→384 (the op's own
//! vectorised accumulation order differs in the last ulp).
//!
//! TEMPORARY: this is a local port of TensorFlow's gaussian kernel until
//! zenresize gains a `gaussian` mode; it then moves there (standing rule:
//! resampling goes through imazen's zenresize). TensorFlow's Apache-2.0
//! licence is in ../LICENSE-tensorflow.

/// One output pixel's span on an axis: the first source index, the tap count
/// (zero for an out-of-range sample centre), and its normalised weights.
struct Span {
    first: usize,
    count: usize,
    weights: [f32; 5],
}

/// `ComputeSpansCore` for the gaussian kernel with `antialias = false`:
/// `kernel_scale` is 1, so the kernel support is a fixed radius of 1.5 input
/// pixels; the `span_size` cap is `min(2·⌈radius·kernel_scale⌉ + 1, in)` =
/// `min(5, in)` (the |d| < 1.5 cutoff trims each span to at most 4 taps).
fn gauss_spans(n_in: usize, n_out: usize) -> Vec<Span> {
    assert!(n_in >= 1 && n_out >= 1, "tf gaussian resize: empty axis");
    let inv = 1.0f32 / (n_out as f32 / n_in as f32);
    let mut out = Vec::with_capacity(n_out);
    for ox in 0..n_out {
        let sf = (ox as f32 + 0.5) * inv;
        let mut span = Span {
            first: 0,
            count: 0,
            weights: [0.0; 5],
        };
        if !(sf < 0.0 || sf > n_in as f32) {
            let lo = ((sf - 1.5) - 0.5).ceil().clamp(0.0, (n_in - 1) as f32) as usize;
            let hi = ((sf + 1.5) - 0.5).floor().clamp(0.0, (n_in - 1) as f32) as usize;
            let mut total = 0.0f32;
            for (i, src) in (lo..=hi).enumerate() {
                let kp = (src as f32 + 0.5) - sf;
                let w = if kp.abs() >= 1.5 {
                    0.0
                } else {
                    // `std::exp(-x*x / (2.0*sigma*sigma))`: the f32 square
                    // is promoted by the f64 denominator — f64 exp, f32 out.
                    (((-(kp * kp) as f64) / (2.0 * 0.5 * 0.5)).exp()) as f32
                };
                total += w;
                span.weights[i] = w;
            }
            if total.abs() >= 1000.0 * f32::MIN_POSITIVE {
                let oot = 1.0f32 / total;
                for w in &mut span.weights[..=hi - lo] {
                    *w *= oot;
                }
            }
            span.first = lo;
            span.count = hi + 1 - lo;
        }
        out.push(span);
    }
    out
}

/// `scale_and_translate` on a planar `[c, h, w]` f32 image with the gaussian
/// kernel: the vertical spans run first, then the horizontal pass reads the
/// intermediate — the op's `ProcessRow`/column order, measured bitwise-equal
/// to TF on the impulse probes.
pub(crate) fn resize_gaussian_tf_f32(
    src: &[f32],
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    assert_eq!(src.len(), c * h * w);
    let vy = gauss_spans(h, oh);
    let mut tmp = vec![0.0f32; c * oh * w];
    for ch in 0..c {
        let plane = &src[ch * h * w..(ch + 1) * h * w];
        let dst = &mut tmp[ch * oh * w..(ch + 1) * oh * w];
        for (oy, span) in vy.iter().enumerate() {
            for i in 0..span.count {
                let row = &plane[(span.first + i) * w..(span.first + i + 1) * w];
                for (d, &s) in dst[oy * w..oy * w + w].iter_mut().zip(row) {
                    *d += span.weights[i] * s;
                }
            }
        }
    }
    let hx = gauss_spans(w, ow);
    let mut out = vec![0.0f32; c * oh * ow];
    for ch in 0..c {
        let plane = &tmp[ch * oh * w..(ch + 1) * oh * w];
        let dst = &mut out[ch * oh * ow..(ch + 1) * oh * ow];
        for oy in 0..oh {
            let row = &plane[oy * w..(oy + 1) * w];
            let drow = &mut dst[oy * ow..(oy + 1) * ow];
            for (ox, span) in hx.iter().enumerate() {
                let mut acc = 0.0f32;
                for i in 0..span.count {
                    acc += span.weights[i] * row[span.first + i];
                }
                drow[ox] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 8→4 impulse responses measured against `tf.image.resize`
    /// (gaussian, tensorflow 2.15 CPU): a source impulse lands only in the
    /// output whose span centre sits half a source pixel away (the radius-1.5
    /// taps one step out sit exactly on the `>= radius` cutoff and read 0).
    #[test]
    fn gaussian_8_to_4_impulses() {
        // tf: src0 -> [0.5,0,0,0]; src2 -> [0,0.5,0,0]; src3 -> [0,0.5,0,0].
        for (k, want) in [
            (0usize, [0.5, 0.0, 0.0, 0.0]),
            (2, [0.0, 0.5, 0.0, 0.0]),
            (3, [0.0, 0.5, 0.0, 0.0]),
        ] {
            let mut src = vec![0.0f32; 8];
            src[k] = 1.0;
            assert_eq!(resize_gaussian_tf_f32(&src, 1, 1, 8, 1, 4), want.to_vec());
        }
    }

    /// Weights normalise to the unit mass: a constant field stays constant.
    #[test]
    fn gaussian_preserves_constant() {
        let src = vec![0.37f32; 3 * 13 * 9];
        let out = resize_gaussian_tf_f32(&src, 3, 13, 9, 7, 5);
        for &v in &out {
            assert!((v - 0.37).abs() < 1e-6, "{v}");
        }
    }

    /// 256→224 matches the weights measured from TF on the impulse probe:
    /// output 1 reads sources 0..2 at [0.04173, 0.72657, 0.23171].
    #[test]
    fn gaussian_256_to_224_weights() {
        let spans = gauss_spans(256, 224);
        let s = &spans[1];
        assert_eq!(s.first, 0);
        assert_eq!(s.count, 3);
        let want = [0.04173f32, 0.72657, 0.23171];
        for (g, w) in s.weights[..3].iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
    }
}
