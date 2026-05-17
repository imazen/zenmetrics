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

/// `src_rgb_u32` is one packed-RGBA u32 per pixel
/// (R | G<<8 | B<<16; alpha unused).
///
/// T4.L (2026-05-16): packed layout cuts host→device upload 3× vs the
/// prior one-byte-per-u32 widening (12 B/pixel → 4 B/pixel). See
/// `docs/CUBECL_GOTCHAS.md` G6.6.
#[cube(launch_unchecked)]
pub fn rgb_u32_to_gray_kernel(src_rgb_u32: &Array<u32>, dst_gray: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    let n = dst_gray.len();
    if idx >= n {
        terminate!();
    }
    let packed = src_rgb_u32[idx];
    let r = (packed & 0xffu32) as f32;
    let g = ((packed >> 8u32) & 0xffu32) as f32;
    let b = ((packed >> 16u32) & 0xffu32) as f32;
    let y = 0.2989_f32 * r + 0.5870_f32 * g + 0.1140_f32 * b;
    dst_gray[idx] = f32::floor(y + 0.5_f32);
}
