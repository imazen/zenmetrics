//! sRGB packed u8 RGB → grayscale f32, with BT.601 luma + rounding.
//!
//! ```python
//! gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
//! return np.round(gray)
//! ```
//!
//! `np.round` is banker's rounding (round-half-to-even) but the ties
//! at exactly 0.5 occur only for a measure-zero set of `(r, g, b)`
//! triples on the f32 line — the dominant error source for downstream
//! parity is f32-vs-f64 promotion, not the tie-breaking rule. We use
//! `floor(x + 0.5)` (round-half-up) which agrees with `np.round` on
//! all non-tie inputs and differs by ≤ 1 LSB on the rare tie. This is
//! identical to what `pyrtools` itself observes when it later casts
//! to f32, so we end up with the same LP coefficients at level 0.

use cubecl::prelude::*;

/// `src_rgb_u32` is laid out as `[r0, g0, b0, r1, g1, b1, ...]`, each
/// entry one widened sRGB byte (WGSL has no `u8` storage type).
#[cube(launch_unchecked)]
pub fn rgb_u32_to_gray_kernel(src_rgb_u32: &Array<u32>, dst_gray: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    let n = dst_gray.len();
    if idx >= n {
        terminate!();
    }
    let r = src_rgb_u32[idx * 3] as f32;
    let g = src_rgb_u32[idx * 3 + 1] as f32;
    let b = src_rgb_u32[idx * 3 + 2] as f32;
    let y = 0.2989_f32 * r + 0.5870_f32 * g + 0.1140_f32 * b;
    dst_gray[idx] = f32::floor(y + 0.5_f32);
}
