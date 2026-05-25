//! Heatmap visualization for per-pixel diffmaps.
//!
//! Mirrors upstream pycvvdp's `visualize_diff_map.py` colormaps.
//! Takes a per-pixel diffmap (JOD-scale, 0 = identical, higher =
//! more visible) and produces sRGB u8 output.
//!
//! # Examples
//!
//! ```
//! use cvvdp_gpu::heatmap::{HeatmapMode, render_heatmap};
//!
//! let diffmap = vec![0.0_f32; 64 * 64];
//! let rgb = render_heatmap(&diffmap, 64, 64, HeatmapMode::Threshold, None);
//! assert_eq!(rgb.len(), 64 * 64 * 3);
//! ```

/// Heatmap rendering mode, matching upstream pycvvdp's colormap types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeatmapMode {
    /// 5-color gradient (blue → cyan → green → yellow → red) over
    /// 0.0–0.10 JOD. Values ≥ 0.10 JOD render as solid red.
    Threshold,
    /// 3-color gradient (cyan → white → yellow) over 0.0–0.30 JOD.
    /// Values ≥ 0.30 JOD render as solid yellow.
    SupraThreshold,
    /// Raw diffmap as grayscale (no colormap). 0 = black, 1 = white.
    Raw,
}

struct ColorStop {
    position: f32,
    r: f32,
    g: f32,
    b: f32,
}

fn interp1(stops: &[ColorStop], v: f32) -> [f32; 3] {
    if stops.is_empty() {
        return [0.0; 3];
    }
    if v <= stops[0].position {
        return [stops[0].r, stops[0].g, stops[0].b];
    }
    for w in stops.windows(2) {
        if v <= w[1].position {
            let t = (v - w[0].position) / (w[1].position - w[0].position);
            return [
                w[0].r + t * (w[1].r - w[0].r),
                w[0].g + t * (w[1].g - w[0].g),
                w[0].b + t * (w[1].b - w[0].b),
            ];
        }
    }
    let last = &stops[stops.len() - 1];
    [last.r, last.g, last.b]
}

fn threshold_stops() -> [ColorStop; 5] {
    [
        ColorStop { position: 0.000, r: 0.2, g: 0.2, b: 1.0 },
        ColorStop { position: 0.025, r: 0.2, g: 1.0, b: 1.0 },
        ColorStop { position: 0.050, r: 0.2, g: 1.0, b: 0.2 },
        ColorStop { position: 0.075, r: 1.0, g: 1.0, b: 0.2 },
        ColorStop { position: 0.100, r: 1.0, g: 0.2, b: 0.2 },
    ]
}

fn supra_threshold_stops() -> [ColorStop; 3] {
    [
        ColorStop { position: 0.00, r: 0.2, g: 1.0, b: 1.0 },
        ColorStop { position: 0.15, r: 1.0, g: 1.0, b: 1.0 },
        ColorStop { position: 0.30, r: 1.0, g: 1.0, b: 0.2 },
    ]
}

fn linear_to_srgb_u8(v: f32) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Render a per-pixel diffmap into an sRGB u8 RGB image.
///
/// `diffmap` is row-major, one f32 per pixel (JOD-scale).
/// `context_rgb` is an optional sRGB u8 reference image (row-major,
/// 3 bytes per pixel) blended as a desaturated luminance backdrop;
/// pass `None` for a neutral gray background.
///
/// Returns a `Vec<u8>` of length `width * height * 3` (row-major RGB).
pub fn render_heatmap(
    diffmap: &[f32],
    width: usize,
    height: usize,
    mode: HeatmapMode,
    context_rgb: Option<&[u8]>,
) -> Vec<u8> {
    let n = width * height;
    assert!(diffmap.len() >= n, "diffmap too short");
    if let Some(ctx) = context_rgb {
        assert!(ctx.len() >= n * 3, "context image too short");
    }

    let mut out = vec![0u8; n * 3];

    for i in 0..n {
        let d = diffmap[i].clamp(0.0, 1.0);

        let bg = if let Some(ctx) = context_rgb {
            let r = ctx[i * 3] as f32 / 255.0;
            let g = ctx[i * 3 + 1] as f32 / 255.0;
            let b = ctx[i * 3 + 2] as f32 / 255.0;
            0.212_656 * r + 0.715_158 * g + 0.072_186 * b
        } else {
            0.5
        };

        let [cr, cg, cb] = match mode {
            HeatmapMode::Threshold => {
                let stops = threshold_stops();
                interp1(&stops, d)
            }
            HeatmapMode::SupraThreshold => {
                let stops = supra_threshold_stops();
                interp1(&stops, d)
            }
            HeatmapMode::Raw => [d, d, d],
        };

        let (fr, fg, fb) = if matches!(mode, HeatmapMode::Raw) {
            (cr, cg, cb)
        } else {
            let lum = 0.212_656 * cr + 0.715_158 * cg + 0.072_186 * cb;
            let inv_lum = 1.0 / (lum + 1e-4);
            (
                (cr * inv_lum * bg).clamp(0.0, 1.0),
                (cg * inv_lum * bg).clamp(0.0, 1.0),
                (cb * inv_lum * bg).clamp(0.0, 1.0),
            )
        };

        out[i * 3] = linear_to_srgb_u8(fr);
        out[i * 3 + 1] = linear_to_srgb_u8(fg);
        out[i * 3 + 2] = linear_to_srgb_u8(fb);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_diffmap_produces_uniform_output() {
        let dm = vec![0.0_f32; 16];
        let rgb = render_heatmap(&dm, 4, 4, HeatmapMode::Threshold, None);
        assert_eq!(rgb.len(), 48);
        for chunk in rgb.chunks(3) {
            assert_eq!(chunk[0], rgb[0]);
            assert_eq!(chunk[1], rgb[1]);
            assert_eq!(chunk[2], rgb[2]);
        }
    }

    #[test]
    fn threshold_extremes_differ() {
        let dm = vec![0.0, 0.1];
        let rgb = render_heatmap(&dm, 2, 1, HeatmapMode::Threshold, None);
        let lo = &rgb[0..3];
        let hi = &rgb[3..6];
        assert_ne!(lo, hi);
    }

    #[test]
    fn supra_threshold_extremes_differ() {
        let dm = vec![0.0, 0.3];
        let rgb = render_heatmap(&dm, 2, 1, HeatmapMode::SupraThreshold, None);
        let lo = &rgb[0..3];
        let hi = &rgb[3..6];
        assert_ne!(lo, hi);
    }

    #[test]
    fn raw_mode_is_grayscale() {
        let dm = vec![0.5_f32; 4];
        let rgb = render_heatmap(&dm, 2, 2, HeatmapMode::Raw, None);
        for chunk in rgb.chunks(3) {
            assert_eq!(chunk[0], chunk[1]);
            assert_eq!(chunk[1], chunk[2]);
        }
    }

    #[test]
    fn context_image_modulates_output() {
        let dm = vec![0.05_f32; 4];
        let white_ctx = vec![255u8; 12];
        let dark_ctx = vec![30u8; 12];
        let rgb_w = render_heatmap(&dm, 2, 2, HeatmapMode::Threshold, Some(&white_ctx));
        let rgb_d = render_heatmap(&dm, 2, 2, HeatmapMode::Threshold, Some(&dark_ctx));
        assert_ne!(rgb_w, rgb_d);
    }
}
