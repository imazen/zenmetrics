#![forbid(unsafe_code)]

//! Alpha channel handling, deterministic noise compositing, and transparency fidelity tracking.
//!
//! Provides the fundamental Rust API for evaluating transparent/translucent images
//! with perceptual quality metrics:
//!
//! 1. **Perceptual Compositing:** Perceptual metrics ($SSIM2$, $Butteraugli$, $PSNR$, $CVVDP$)
//!    operate on opaque 3-channel surfaces. Compositing over [`AlphaPolicy::DeterministicNoise`]
//!    (a precomputed 4096-entry L1 linear noise table in $[0.2, 0.8]$) avoids both the color-masking
//!    blind spots of flat white/black backgrounds and the artificial geometric edge penalties
//!    of periodic checkerboards.
//! 2. **Alpha Channel Fidelity Tracking:** [`AlphaStats`] measures channel-level degradation
//!    ($max\_delta$, $MAE$, $PSNR$) independently of the visual composited score.
//! 3. **Strict Opacity Gating:** [`AlphaPolicy::RequireOpaque`] safely strips alpha only when
//!    every pixel is fully opaque ($255$), returning [`AlphaError::AlphaNotOpaque`] otherwise.

use std::fmt;

/// Precomputed deterministic linear noise table in $[0.2, 0.8]$.
///
/// 4096 values fitting in 16 KiB (L1 cache), generated via integer-only hashing
/// (wrapping multiply + xorshift) for bit-identical results across all architectures.
pub const ALPHA_BG_TABLE: [f32; 4096] = {
    let mut table = [0.0f32; 4096];
    let mut i = 0u32;
    while i < 4096 {
        let mut h = i.wrapping_mul(2654435761);
        h ^= h >> 16;
        h = h.wrapping_mul(0x45d9f3b);
        h ^= h >> 16;
        table[i as usize] = 0.2 + (h & 0xFFFF) as f32 * (0.6 / 65535.0);
        i += 1;
    }
    table
};

/// Deterministic noise background value in linear light for pixel coordinate `(x, y)`.
#[inline(always)]
pub fn alpha_background_linear(x: usize, y: usize) -> f32 {
    let yh = (y as u32).wrapping_mul(2654435761);
    ALPHA_BG_TABLE[((x as u32) ^ yh) as usize & 0xFFF]
}

/// Convert sRGB u8 `[0, 255]` to linear float `[0.0, 1.0]`.
#[inline]
pub fn srgb_u8_to_linear(v: u8) -> f32 {
    let c = v as f32 * (1.0 / 255.0);
    if c <= 0.04045 {
        c * (1.0 / 12.92)
    } else {
        ((c + 0.055) * (1.0 / 1.055)).powf(2.4)
    }
}

/// Convert linear float `[0.0, 1.0]` to sRGB u8 `[0, 255]`.
#[inline]
pub fn linear_to_srgb_u8(v: f32) -> u8 {
    let clamped = v.clamp(0.0, 1.0);
    let c = if clamped <= 0.0031308 {
        clamped * 12.92
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Policy for handling images with an alpha channel during perceptual metric evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AlphaPolicy {
    /// Composite over a deterministic pseudo-random linear noise background in $[0.2, 0.8]$.
    ///
    /// Recommended for perceptual image quality metrics. Eliminates masking blind spots
    /// (unlike flat white or black) and avoids triggering high-frequency pyramid filter penalties
    /// (unlike structured checkerboards).
    DeterministicNoise,
    /// Composite over a solid white background `(255, 255, 255)`.
    MatteWhite,
    /// Composite over a solid black background `(0, 0, 0)`.
    MatteBlack,
    /// Composite over a specified solid sRGB color `(r, g, b)`.
    CompositeOnto {
        /// Red channel value in 0..=255.
        r: u8,
        /// Green channel value in 0..=255.
        g: u8,
        /// Blue channel value in 0..=255.
        b: u8,
    },
    /// Evaluate against both white and black backgrounds, returning the lower (worst-case) score.
    MinWhiteBlack,
    /// Unconditionally discard the alpha channel and evaluate the raw RGB channels.
    DiscardUnchecked,
    /// Require all pixels in the alpha channel to be fully opaque ($255$).
    /// Returns [`AlphaError::AlphaNotOpaque`] if any non-opaque pixels exist.
    RequireOpaque,
}

impl Default for AlphaPolicy {
    fn default() -> Self {
        Self::RequireOpaque
    }
}

impl AlphaPolicy {
    /// Parse policy name string from CLI flags or Python parameters.
    pub fn from_str_loose(s: &str) -> Option<Self> {
        let clean = s.trim().to_ascii_lowercase().replace(['-', ' '], "_");
        match clean.as_str() {
            "noise" | "deterministic_noise" | "random_noise" => Some(Self::DeterministicNoise),
            "white" | "matte_white" => Some(Self::MatteWhite),
            "black" | "matte_black" => Some(Self::MatteBlack),
            "min_wb" | "min_white_black" | "dual" => Some(Self::MinWhiteBlack),
            "discard" | "discard_unchecked" | "ignore" => Some(Self::DiscardUnchecked),
            "opaque" | "require_opaque" | "discard_if_opaque" => Some(Self::RequireOpaque),
            _ => None,
        }
    }
}

/// Error type encountered during alpha channel handling and compositing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlphaError {
    /// Image contains transparent or translucent pixels when [`AlphaPolicy::RequireOpaque`] was requested.
    AlphaNotOpaque {
        /// Number of non-opaque pixels detected.
        non_opaque_count: usize,
        /// Maximum transparency delta ($255 - \min(\alpha)$).
        max_transparency: u8,
    },
    /// Buffer length did not match `width * height * channels`.
    BufferLengthMismatch {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        actual: usize,
    },
    /// Dimensions mismatch between reference and distorted images.
    DimensionMismatch {
        /// Reference dimensions `(width, height)`.
        ref_dims: (u32, u32),
        /// Distorted dimensions `(width, height)`.
        dist_dims: (u32, u32),
    },
}

impl fmt::Display for AlphaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlphaNotOpaque { non_opaque_count, max_transparency } => {
                write!(
                    f,
                    "Image contains non-opaque alpha: {non_opaque_count} non-opaque pixels (max transparency: {max_transparency}/255). Image quality metrics operate on opaque RGB. Specify an explicit alpha_policy (e.g. 'noise', 'matte_white', 'matte_black', 'discard_unchecked') to evaluate."
                )
            }
            Self::BufferLengthMismatch { expected, actual } => {
                write!(f, "buffer length mismatch: expected {expected} bytes, got {actual}")
            }
            Self::DimensionMismatch { ref_dims, dist_dims } => {
                write!(
                    f,
                    "dimension mismatch: reference is {}x{}, distorted is {}x{}",
                    ref_dims.0, ref_dims.1, dist_dims.0, dist_dims.1
                )
            }
        }
    }
}

impl std::error::Error for AlphaError {}

/// Quantitative fidelity statistics between reference and distorted alpha channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaStats {
    /// Maximum absolute difference between reference and distorted alpha channels in $0..=255$.
    pub max_delta: u8,
    /// Number of pixels where the alpha channel differs.
    pub pixels_differing: u64,
    /// Mean Absolute Error of alpha channels across all pixels ($0.0..=255.0$).
    pub mae: f64,
    /// Peak Signal-to-Noise Ratio (dB) of the alpha channel. Returns $100.0$ when bit-identical.
    pub psnr: f64,
    /// Total number of pixels compared.
    pub total_pixels: u64,
}

/// Check if an RGBA buffer is fully opaque (every alpha byte is 255).
pub fn is_fully_opaque_rgba(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).all(|px| px[3] == 255)
}

/// Extract alpha channel bytes into a flat vector.
pub fn extract_alpha_channel(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4).map(|px| px[3]).collect()
}

/// Compute fidelity statistics between two flat 1-channel alpha buffers.
pub fn compute_alpha_stats(ref_alpha: &[u8], dist_alpha: &[u8]) -> Result<AlphaStats, AlphaError> {
    if ref_alpha.len() != dist_alpha.len() {
        return Err(AlphaError::BufferLengthMismatch {
            expected: ref_alpha.len(),
            actual: dist_alpha.len(),
        });
    }

    let mut max_delta = 0u8;
    let mut pixels_differing = 0u64;
    let mut sum_abs_diff = 0u64;
    let mut sum_sq_diff = 0u64;
    let total_pixels = ref_alpha.len() as u64;

    for (&r, &d) in ref_alpha.iter().zip(dist_alpha.iter()) {
        let delta = r.abs_diff(d);
        if delta > max_delta {
            max_delta = delta;
        }
        if delta > 0 {
            pixels_differing += 1;
        }
        sum_abs_diff += delta as u64;
        sum_sq_diff += (delta as u64) * (delta as u64);
    }

    if total_pixels == 0 {
        return Ok(AlphaStats {
            max_delta: 0,
            pixels_differing: 0,
            mae: 0.0,
            psnr: 100.0,
            total_pixels: 0,
        });
    }

    let mae = sum_abs_diff as f64 / total_pixels as f64;
    let mse = sum_sq_diff as f64 / total_pixels as f64;
    let psnr = if mse <= 1e-12 {
        100.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    };

    Ok(AlphaStats {
        max_delta,
        pixels_differing,
        mae,
        psnr,
        total_pixels,
    })
}

/// Compute fidelity statistics between two flat 4-channel RGBA buffers.
pub fn compute_rgba_alpha_stats(ref_rgba: &[u8], dist_rgba: &[u8]) -> Result<AlphaStats, AlphaError> {
    if ref_rgba.len() != dist_rgba.len() {
        return Err(AlphaError::BufferLengthMismatch {
            expected: ref_rgba.len(),
            actual: dist_rgba.len(),
        });
    }
    if ref_rgba.len() % 4 != 0 {
        return Err(AlphaError::BufferLengthMismatch {
            expected: (ref_rgba.len() / 4) * 4,
            actual: ref_rgba.len(),
        });
    }

    let mut max_delta = 0u8;
    let mut pixels_differing = 0u64;
    let mut sum_abs_diff = 0u64;
    let mut sum_sq_diff = 0u64;
    let total_pixels = (ref_rgba.len() / 4) as u64;

    for (r_px, d_px) in ref_rgba.chunks_exact(4).zip(dist_rgba.chunks_exact(4)) {
        let r = r_px[3];
        let d = d_px[3];
        let delta = r.abs_diff(d);
        if delta > max_delta {
            max_delta = delta;
        }
        if delta > 0 {
            pixels_differing += 1;
        }
        sum_abs_diff += delta as u64;
        sum_sq_diff += (delta as u64) * (delta as u64);
    }

    if total_pixels == 0 {
        return Ok(AlphaStats {
            max_delta: 0,
            pixels_differing: 0,
            mae: 0.0,
            psnr: 100.0,
            total_pixels: 0,
        });
    }

    let mae = sum_abs_diff as f64 / total_pixels as f64;
    let mse = sum_sq_diff as f64 / total_pixels as f64;
    let psnr = if mse <= 1e-12 {
        100.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    };

    Ok(AlphaStats {
        max_delta,
        pixels_differing,
        mae,
        psnr,
        total_pixels,
    })
}

/// Composite an RGBA sRGB buffer onto an opaque RGB8 sRGB buffer according to the given policy.
///
/// Input `rgba` must be `width * height * 4` bytes.
/// Returns a flat RGB8 vector of `width * height * 3` bytes.
pub fn composite_rgba_srgb8(
    rgba: &[u8],
    width: u32,
    height: u32,
    policy: AlphaPolicy,
) -> Result<Vec<u8>, AlphaError> {
    let expected_len = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected_len {
        return Err(AlphaError::BufferLengthMismatch {
            expected: expected_len,
            actual: rgba.len(),
        });
    }

    let pixel_count = (width as usize) * (height as usize);
    let mut out = vec![0u8; pixel_count * 3];

    // Fast path: if the image is 100% opaque, strip alpha directly without math
    if is_fully_opaque_rgba(rgba) {
        for (i, px) in rgba.chunks_exact(4).enumerate() {
            out[i * 3] = px[0];
            out[i * 3 + 1] = px[1];
            out[i * 3 + 2] = px[2];
        }
        return Ok(out);
    }

    // Image has non-opaque pixels: dispatch based on policy
    match policy {
        AlphaPolicy::RequireOpaque => {
            let mut non_opaque_count = 0usize;
            let mut max_transparency = 0u8;
            for px in rgba.chunks_exact(4) {
                let a = px[3];
                if a < 255 {
                    non_opaque_count += 1;
                    let trans = 255 - a;
                    if trans > max_transparency {
                        max_transparency = trans;
                    }
                }
            }
            Err(AlphaError::AlphaNotOpaque {
                non_opaque_count,
                max_transparency,
            })
        }
        AlphaPolicy::DiscardUnchecked => {
            for (i, px) in rgba.chunks_exact(4).enumerate() {
                out[i * 3] = px[0];
                out[i * 3 + 1] = px[1];
                out[i * 3 + 2] = px[2];
            }
            Ok(out)
        }
        AlphaPolicy::MatteWhite => {
            composite_over_solid_linear(rgba, width, height, [1.0, 1.0, 1.0], &mut out);
            Ok(out)
        }
        AlphaPolicy::MatteBlack => {
            composite_over_solid_linear(rgba, width, height, [0.0, 0.0, 0.0], &mut out);
            Ok(out)
        }
        AlphaPolicy::CompositeOnto { r, g, b } => {
            let bg_lin = [
                srgb_u8_to_linear(r),
                srgb_u8_to_linear(g),
                srgb_u8_to_linear(b),
            ];
            composite_over_solid_linear(rgba, width, height, bg_lin, &mut out);
            Ok(out)
        }
        AlphaPolicy::DeterministicNoise => {
            composite_over_noise_linear(rgba, width, height, &mut out);
            Ok(out)
        }
        AlphaPolicy::MinWhiteBlack => {
            // MinWhiteBlack requires two separate metric passes; the single compositing
            // function defaults to MatteWhite as primary surface.
            composite_over_solid_linear(rgba, width, height, [1.0, 1.0, 1.0], &mut out);
            Ok(out)
        }
    }
}

/// Composite RGBA over a constant background color in linear light.
fn composite_over_solid_linear(
    rgba: &[u8],
    _width: u32,
    _height: u32,
    bg_linear: [f32; 3],
    out: &mut [u8],
) {
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        let a = px[3];
        let out_idx = i * 3;
        if a == 255 {
            out[out_idx] = px[0];
            out[out_idx + 1] = px[1];
            out[out_idx + 2] = px[2];
        } else if a == 0 {
            out[out_idx] = linear_to_srgb_u8(bg_linear[0]);
            out[out_idx + 1] = linear_to_srgb_u8(bg_linear[1]);
            out[out_idx + 2] = linear_to_srgb_u8(bg_linear[2]);
        } else {
            let alpha = a as f32 * (1.0 / 255.0);
            let inv = 1.0 - alpha;
            let rl = srgb_u8_to_linear(px[0]);
            let gl = srgb_u8_to_linear(px[1]);
            let bl = srgb_u8_to_linear(px[2]);

            out[out_idx] = linear_to_srgb_u8(rl * alpha + bg_linear[0] * inv);
            out[out_idx + 1] = linear_to_srgb_u8(gl * alpha + bg_linear[1] * inv);
            out[out_idx + 2] = linear_to_srgb_u8(bl * alpha + bg_linear[2] * inv);
        }
    }
}

/// Composite RGBA over the deterministic linear noise background table.
fn composite_over_noise_linear(
    rgba: &[u8],
    width: u32,
    _height: u32,
    out: &mut [u8],
) {
    let w = width as usize;
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        let x = i % w;
        let y = i / w;
        let a = px[3];
        let out_idx = i * 3;

        if a == 255 {
            out[out_idx] = px[0];
            out[out_idx + 1] = px[1];
            out[out_idx + 2] = px[2];
        } else if a == 0 {
            let bg = alpha_background_linear(x, y);
            let bg_srgb = linear_to_srgb_u8(bg);
            out[out_idx] = bg_srgb;
            out[out_idx + 1] = bg_srgb;
            out[out_idx + 2] = bg_srgb;
        } else {
            let alpha = a as f32 * (1.0 / 255.0);
            let inv = 1.0 - alpha;
            let bg = alpha_background_linear(x, y);
            let rl = srgb_u8_to_linear(px[0]);
            let gl = srgb_u8_to_linear(px[1]);
            let bl = srgb_u8_to_linear(px[2]);

            out[out_idx] = linear_to_srgb_u8(rl * alpha + bg * inv);
            out[out_idx + 1] = linear_to_srgb_u8(gl * alpha + bg * inv);
            out[out_idx + 2] = linear_to_srgb_u8(bl * alpha + bg * inv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_table_bounds_and_determinism() {
        for &v in &ALPHA_BG_TABLE {
            assert!(v >= 0.2 && v <= 0.8, "noise out of [0.2, 0.8]: {v}");
        }
        // Bit-exact across repeated calls
        let bg1 = alpha_background_linear(42, 107);
        let bg2 = alpha_background_linear(42, 107);
        assert_eq!(bg1, bg2);
    }

    #[test]
    fn test_srgb_linear_roundtrip() {
        for b in 0..=255u8 {
            let lin = srgb_u8_to_linear(b);
            let roundtrip = linear_to_srgb_u8(lin);
            assert_eq!(b, roundtrip, "failed roundtrip for {b}");
        }
    }

    #[test]
    fn test_opaque_passes_require_opaque() {
        let rgba = vec![128, 64, 32, 255, 200, 100, 50, 255];
        let rgb = composite_rgba_srgb8(&rgba, 2, 1, AlphaPolicy::RequireOpaque).unwrap();
        assert_eq!(rgb, vec![128, 64, 32, 200, 100, 50]);
    }

    #[test]
    fn test_non_opaque_errors_under_require_opaque() {
        let rgba = vec![128, 64, 32, 255, 200, 100, 50, 200];
        let err = composite_rgba_srgb8(&rgba, 2, 1, AlphaPolicy::RequireOpaque).unwrap_err();
        assert!(matches!(err, AlphaError::AlphaNotOpaque { non_opaque_count: 1, .. }));
    }

    #[test]
    fn test_noise_compositing() {
        let rgba = vec![128, 64, 32, 128, 200, 100, 50, 0];
        let rgb = composite_rgba_srgb8(&rgba, 2, 1, AlphaPolicy::DeterministicNoise).unwrap();
        assert_eq!(rgb.len(), 6);
        // Pixel 1: alpha=0, so it must be pure background noise
        let bg_lin = alpha_background_linear(1, 0);
        let expected_bg = linear_to_srgb_u8(bg_lin);
        assert_eq!(rgb[3], expected_bg);
        assert_eq!(rgb[4], expected_bg);
        assert_eq!(rgb[5], expected_bg);
    }

    #[test]
    fn test_alpha_stats_computation() {
        // 1-channel alpha
        let ref_a = vec![255, 200];
        let dist_a = vec![255, 190];
        let stats_a = compute_alpha_stats(&ref_a, &dist_a).unwrap();
        assert_eq!(stats_a.max_delta, 10);
        assert_eq!(stats_a.pixels_differing, 1);
        assert_eq!(stats_a.total_pixels, 2);
        assert_eq!(stats_a.mae, 5.0);
        assert!(stats_a.psnr > 30.0);

        // 4-channel RGBA
        let ref_rgba = vec![0, 0, 0, 255, 0, 0, 0, 200];
        let dist_rgba = vec![0, 0, 0, 255, 0, 0, 0, 190];
        let stats = compute_rgba_alpha_stats(&ref_rgba, &dist_rgba).unwrap();
        assert_eq!(stats.max_delta, 10);
        assert_eq!(stats.pixels_differing, 1);
        assert_eq!(stats.total_pixels, 2);
        assert_eq!(stats.mae, 5.0);
        assert!(stats.psnr > 30.0);
    }
}
