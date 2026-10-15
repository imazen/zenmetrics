//! Steering maps for jxl-encoder's perceptual quantization loop.
//!
//! The loop steers each 8×8 block by the 16-norm of a per-pixel diffmap
//! (times 1.2, libjxl's `TileDistMap`) against the target distance, and
//! picks seeds by the map's maximum. Those constants are butteraugli's, so
//! another metric's map is first put on butteraugli's scale: the error
//! reading of its native map (`1 − v` for a similarity map), resampled to
//! one value per pixel, then `scale · e^exponent`. The default `scale` and
//! `exponent` per metric come from `examples/loop_norm_calibrate.rs`
//! (`benchmarks/loop_norm_calibration_2026-10-15.tsv`), which fits them so
//! the metric's block distances match butteraugli's on the same encodes.

use crate::SpatialMap;

/// One metric map's default placement on butteraugli's scale.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LoopNorm {
    /// The metric (`spatial_map` route name).
    pub(crate) metric: &'static str,
    /// The map within the metric.
    pub(crate) map: &'static str,
    /// Multiplier after the exponent.
    pub(crate) scale: f64,
    /// Power applied to the error reading.
    pub(crate) exponent: f64,
    /// Pearson correlation of the fitted block distances with butteraugli's,
    /// in log space, on the held-out images.
    pub(crate) r: f64,
}

/// Fitted defaults (see the module docs). A metric map missing here has no
/// fitted default; the viewer then starts from `scale = 1, exponent = 1`
/// and says so.
#[rustfmt::skip]
pub(crate) const DEFAULTS: &[LoopNorm] = &[
    LoopNorm { metric: "butteraugli", map: "diffmap", scale: 1.000000e0, exponent: 1.0, r: 1.0000 },
    LoopNorm { metric: "cvvdp", map: "diffmap", scale: 5.619538e-1, exponent: 0.35, r: 0.7928 },
    LoopNorm { metric: "dssim", map: "diffmap", scale: 1.668279e1, exponent: 0.45, r: 0.8917 },
    LoopNorm { metric: "dssim", map: "ssim_0", scale: 4.362224e0, exponent: 0.4, r: 0.8841 },
    LoopNorm { metric: "dssim", map: "ssim_1", scale: 7.356318e0, exponent: 0.4, r: 0.9170 },
    LoopNorm { metric: "dssim", map: "ssim_2", scale: 9.729418e0, exponent: 0.35, r: 0.8727 },
    LoopNorm { metric: "dssim", map: "ssim_3", scale: 1.176531e1, exponent: 0.3, r: 0.6957 },
    LoopNorm { metric: "dssim", map: "ssim_4", scale: 9.945249e0, exponent: 0.25, r: 0.4781 },
    LoopNorm { metric: "fsim", map: "similarity", scale: 4.115202e0, exponent: 0.35, r: 0.8513 },
    LoopNorm { metric: "gmsd", map: "gms", scale: 4.530165e0, exponent: 0.35, r: 0.8418 },
    LoopNorm { metric: "haarpsi", map: "similarity", scale: 3.783167e0, exponent: 0.4, r: 0.8486 },
    LoopNorm { metric: "iwssim", map: "diffmap", scale: 5.883567e0, exponent: 0.35, r: 0.7842 },
    LoopNorm { metric: "iwssim", map: "cs_0", scale: 3.969337e0, exponent: 0.45, r: 0.8426 },
    LoopNorm { metric: "iwssim", map: "cs_1", scale: 4.895437e0, exponent: 0.35, r: 0.8636 },
    LoopNorm { metric: "iwssim", map: "cs_2", scale: 4.266364e0, exponent: 0.25, r: 0.7764 },
    LoopNorm { metric: "iwssim", map: "cs_3", scale: 2.890741e0, exponent: 0.15, r: 0.5865 },
    LoopNorm { metric: "iwssim", map: "csl_4", scale: 2.465823e0, exponent: 0.1, r: 0.0872 },
    LoopNorm { metric: "mad", map: "hi", scale: 1.168545e0, exponent: 0.1, r: 0.4483 },
    LoopNorm { metric: "mad", map: "lo", scale: 8.540901e-1, exponent: 0.35, r: 0.3000 },
    LoopNorm { metric: "mdctpsnr", map: "detection", scale: 3.180072e0, exponent: 0.25, r: 0.8673 },
    LoopNorm { metric: "msssim", map: "diffmap", scale: 8.360980e0, exponent: 0.4, r: 0.8959 },
    LoopNorm { metric: "msssim", map: "ssim", scale: 4.291026e0, exponent: 0.45, r: 0.8943 },
    LoopNorm { metric: "nlpd", map: "diffmap", scale: 5.148889e-1, exponent: 0.45, r: 0.7652 },
    LoopNorm { metric: "nlpd", map: "level_0", scale: 9.500104e-1, exponent: 0.3, r: 0.8006 },
    LoopNorm { metric: "nlpd", map: "level_1", scale: 1.522514e0, exponent: 0.3, r: 0.8445 },
    LoopNorm { metric: "nlpd", map: "level_2", scale: 1.913106e0, exponent: 0.25, r: 0.7940 },
    LoopNorm { metric: "nlpd", map: "level_3", scale: 2.985232e0, exponent: 0.25, r: 0.7295 },
    LoopNorm { metric: "nlpd", map: "level_4", scale: 2.395344e0, exponent: 0.15, r: 0.5886 },
    LoopNorm { metric: "nlpd", map: "level_5", scale: 2.571190e0, exponent: 0.1, r: 0.4460 },
    LoopNorm { metric: "psnr", map: "squared_error", scale: 5.755334e0, exponent: 0.25, r: 0.7636 },
    LoopNorm { metric: "psnrhvs", map: "csf_error", scale: 3.889637e-1, exponent: 0.25, r: 0.8246 },
    LoopNorm { metric: "psnrhvs", map: "masked_error", scale: 4.898035e-1, exponent: 0.25, r: 0.8419 },
    LoopNorm { metric: "psnrhvs", map: "csf_error_r", scale: 4.016585e-1, exponent: 0.25, r: 0.8171 },
    LoopNorm { metric: "psnrhvs", map: "csf_error_g", scale: 4.424210e-1, exponent: 0.3, r: 0.8490 },
    LoopNorm { metric: "psnrhvs", map: "csf_error_b", scale: 3.732352e-1, exponent: 0.25, r: 0.7804 },
    LoopNorm { metric: "psnrhvs", map: "masked_error_r", scale: 5.960703e-1, exponent: 0.2, r: 0.7706 },
    LoopNorm { metric: "psnrhvs", map: "masked_error_g", scale: 7.591154e-1, exponent: 0.2, r: 0.7546 },
    LoopNorm { metric: "psnrhvs", map: "masked_error_b", scale: 5.160702e-1, exponent: 0.2, r: 0.7732 },
    LoopNorm { metric: "ssim2", map: "diffmap", scale: 7.351109e-1, exponent: 0.25, r: 0.6707 },
    LoopNorm { metric: "ssim2", map: "ssim_y_0", scale: 5.425148e0, exponent: 0.5, r: 0.8533 },
    LoopNorm { metric: "ssim2", map: "ssim_x_0", scale: 2.175183e0, exponent: 0.25, r: 0.7586 },
    LoopNorm { metric: "ssim2", map: "ssim_b_0", scale: 2.767908e0, exponent: 0.3, r: 0.6471 },
    LoopNorm { metric: "ssim2", map: "artifact_x_0", scale: 2.062433e0, exponent: 0.15, r: 0.3538 },
    LoopNorm { metric: "ssim2", map: "artifact_y_0", scale: 4.932174e0, exponent: 0.35, r: 0.6272 },
    LoopNorm { metric: "ssim2", map: "artifact_b_0", scale: 3.825468e0, exponent: 0.25, r: 0.2457 },
    LoopNorm { metric: "ssim2", map: "detail_lost_x_0", scale: 2.092662e0, exponent: 0.15, r: 0.6197 },
    LoopNorm { metric: "ssim2", map: "detail_lost_y_0", scale: 4.550219e0, exponent: 0.35, r: 0.6566 },
    LoopNorm { metric: "ssim2", map: "detail_lost_b_0", scale: 2.901744e0, exponent: 0.2, r: 0.5767 },
    LoopNorm { metric: "ssim2", map: "ssim_x_1", scale: 2.610287e0, exponent: 0.25, r: 0.7323 },
    LoopNorm { metric: "ssim2", map: "ssim_y_1", scale: 5.186015e0, exponent: 0.35, r: 0.8020 },
    LoopNorm { metric: "ssim2", map: "ssim_b_1", scale: 2.630504e0, exponent: 0.25, r: 0.6390 },
    LoopNorm { metric: "ssim2", map: "artifact_x_1", scale: 2.973358e0, exponent: 0.2, r: 0.5330 },
    LoopNorm { metric: "ssim2", map: "artifact_y_1", scale: 1.054759e1, exponent: 0.45, r: 0.7621 },
    LoopNorm { metric: "ssim2", map: "artifact_b_1", scale: 4.182367e0, exponent: 0.25, r: 0.4576 },
    LoopNorm { metric: "ssim2", map: "detail_lost_x_1", scale: 2.893368e0, exponent: 0.2, r: 0.6312 },
    LoopNorm { metric: "ssim2", map: "detail_lost_y_1", scale: 9.629289e0, exponent: 0.45, r: 0.7436 },
    LoopNorm { metric: "ssim2", map: "detail_lost_b_1", scale: 2.990627e0, exponent: 0.2, r: 0.5823 },
    LoopNorm { metric: "ssim2", map: "ssim_x_2", scale: 2.522480e0, exponent: 0.2, r: 0.6537 },
    LoopNorm { metric: "ssim2", map: "ssim_y_2", scale: 4.811665e0, exponent: 0.25, r: 0.7086 },
    LoopNorm { metric: "ssim2", map: "ssim_b_2", scale: 2.507425e0, exponent: 0.2, r: 0.5528 },
    LoopNorm { metric: "ssim2", map: "artifact_x_2", scale: 4.362275e0, exponent: 0.25, r: 0.6716 },
    LoopNorm { metric: "ssim2", map: "artifact_y_2", scale: 1.180203e1, exponent: 0.4, r: 0.7417 },
    LoopNorm { metric: "ssim2", map: "artifact_b_2", scale: 6.328998e0, exponent: 0.3, r: 0.6212 },
    LoopNorm { metric: "ssim2", map: "detail_lost_x_2", scale: 4.429002e0, exponent: 0.25, r: 0.6588 },
    LoopNorm { metric: "ssim2", map: "detail_lost_y_2", scale: 1.138617e1, exponent: 0.4, r: 0.7304 },
    LoopNorm { metric: "ssim2", map: "detail_lost_b_2", scale: 4.532456e0, exponent: 0.25, r: 0.6109 },
    LoopNorm { metric: "ssim2", map: "ssim_x_3", scale: 2.557516e0, exponent: 0.15, r: 0.4493 },
    LoopNorm { metric: "ssim2", map: "ssim_y_3", scale: 3.381349e0, exponent: 0.15, r: 0.5126 },
    LoopNorm { metric: "ssim2", map: "ssim_b_3", scale: 2.467121e0, exponent: 0.15, r: 0.3631 },
    LoopNorm { metric: "ssim2", map: "artifact_x_3", scale: 7.682740e0, exponent: 0.3, r: 0.6285 },
    LoopNorm { metric: "ssim2", map: "artifact_y_3", scale: 8.174643e0, exponent: 0.3, r: 0.5659 },
    LoopNorm { metric: "ssim2", map: "artifact_b_3", scale: 7.526971e0, exponent: 0.3, r: 0.5924 },
    LoopNorm { metric: "ssim2", map: "detail_lost_x_3", scale: 5.342855e0, exponent: 0.25, r: 0.5618 },
    LoopNorm { metric: "ssim2", map: "detail_lost_y_3", scale: 8.226421e0, exponent: 0.3, r: 0.5698 },
    LoopNorm { metric: "ssim2", map: "detail_lost_b_3", scale: 5.237069e0, exponent: 0.25, r: 0.5273 },
    LoopNorm { metric: "ssim2", map: "ssim_x_4", scale: 3.222154e0, exponent: 0.15, r: 0.2899 },
    LoopNorm { metric: "ssim2", map: "ssim_y_4", scale: 4.279799e0, exponent: 0.15, r: 0.3330 },
    LoopNorm { metric: "ssim2", map: "ssim_b_4", scale: 2.001209e0, exponent: 0.1, r: 0.2137 },
    LoopNorm { metric: "ssim2", map: "artifact_x_4", scale: 5.873805e0, exponent: 0.25, r: 0.5744 },
    LoopNorm { metric: "ssim2", map: "artifact_y_4", scale: 6.457288e0, exponent: 0.25, r: 0.4373 },
    LoopNorm { metric: "ssim2", map: "artifact_b_4", scale: 5.805678e0, exponent: 0.25, r: 0.5286 },
    LoopNorm { metric: "ssim2", map: "detail_lost_x_4", scale: 5.949861e0, exponent: 0.25, r: 0.5095 },
    LoopNorm { metric: "ssim2", map: "detail_lost_y_4", scale: 6.369632e0, exponent: 0.25, r: 0.4482 },
    LoopNorm { metric: "ssim2", map: "detail_lost_b_4", scale: 3.877952e0, exponent: 0.2, r: 0.4683 },
    LoopNorm { metric: "ssim2", map: "ssim_x_5", scale: 2.367506e0, exponent: 0.1, r: 0.2243 },
    LoopNorm { metric: "ssim2", map: "ssim_y_5", scale: 5.039669e0, exponent: 0.15, r: 0.2303 },
    LoopNorm { metric: "ssim2", map: "ssim_b_5", scale: 2.414040e0, exponent: 0.1, r: 0.1343 },
    LoopNorm { metric: "ssim2", map: "artifact_x_5", scale: 4.228155e0, exponent: 0.2, r: 0.4755 },
    LoopNorm { metric: "ssim2", map: "artifact_y_5", scale: 4.623599e0, exponent: 0.2, r: 0.2965 },
    LoopNorm { metric: "ssim2", map: "artifact_b_5", scale: 6.681790e0, exponent: 0.25, r: 0.4166 },
    LoopNorm { metric: "ssim2", map: "detail_lost_x_5", scale: 4.243691e0, exponent: 0.2, r: 0.4698 },
    LoopNorm { metric: "ssim2", map: "detail_lost_y_5", scale: 7.246298e0, exponent: 0.25, r: 0.3550 },
    LoopNorm { metric: "ssim2", map: "detail_lost_b_5", scale: 4.137799e0, exponent: 0.2, r: 0.4108 },
    LoopNorm { metric: "vif", map: "diffmap", scale: 1.268239e0, exponent: 0.3, r: 0.6441 },
    LoopNorm { metric: "vif", map: "info_loss_0", scale: 1.151864e0, exponent: 0.45, r: 0.8089 },
    LoopNorm { metric: "vif", map: "info_loss_1", scale: 2.592072e0, exponent: 0.35, r: 0.7842 },
    LoopNorm { metric: "vif", map: "info_loss_2", scale: 2.819511e0, exponent: 0.3, r: 0.6906 },
    LoopNorm { metric: "vif", map: "info_loss_3", scale: 2.981250e0, exponent: 0.25, r: 0.5486 },
    LoopNorm { metric: "vifvec", map: "diffmap", scale: 1.491903e0, exponent: 0.35, r: 0.6955 },
    LoopNorm { metric: "vifvec", map: "info_loss_l1_b1", scale: 6.059480e-1, exponent: 0.25, r: 0.7414 },
    LoopNorm { metric: "vifvec", map: "info_loss_l1_b4", scale: 5.613156e-1, exponent: 0.3, r: 0.7547 },
    LoopNorm { metric: "vifvec", map: "info_loss_l2_b1", scale: 6.281540e-1, exponent: 0.2, r: 0.6652 },
    LoopNorm { metric: "vifvec", map: "info_loss_l2_b4", scale: 5.841065e-1, exponent: 0.25, r: 0.6797 },
    LoopNorm { metric: "vifvec", map: "info_loss_l3_b1", scale: 6.018770e-1, exponent: 0.25, r: 0.5632 },
    LoopNorm { metric: "vifvec", map: "info_loss_l3_b4", scale: 5.585321e-1, exponent: 0.3, r: 0.6265 },
    LoopNorm { metric: "vifvec", map: "info_loss_l4_b1", scale: 7.181083e-1, exponent: 0.2, r: 0.4626 },
    LoopNorm { metric: "vifvec", map: "info_loss_l4_b4", scale: 6.936516e-1, exponent: 0.25, r: 0.5351 },
    LoopNorm { metric: "vmaf-v1", map: "speed_u", scale: 7.671500e-1, exponent: 0.1, r: 0.1949 },
    LoopNorm { metric: "vmaf-v1", map: "speed_v", scale: 7.943392e-1, exponent: 0.15, r: 0.1589 },
    LoopNorm { metric: "vmaf-v1", map: "aim_0_h", scale: 3.546939e0, exponent: 0.1, r: 0.4811 },
    LoopNorm { metric: "vmaf-v1", map: "aim_0_v", scale: 3.587815e0, exponent: 0.1, r: 0.4691 },
    LoopNorm { metric: "vmaf-v1", map: "aim_0_d", scale: 5.472297e0, exponent: 0.1, r: -0.1730 },
    LoopNorm { metric: "vmaf-v1", map: "aim_1_h", scale: 2.899675e0, exponent: 0.1, r: 0.5305 },
    LoopNorm { metric: "vmaf-v1", map: "aim_1_v", scale: 2.926101e0, exponent: 0.1, r: 0.4834 },
    LoopNorm { metric: "vmaf-v1", map: "aim_1_d", scale: 2.761357e0, exponent: 0.1, r: 0.1234 },
    LoopNorm { metric: "vmaf-v1", map: "aim_2_h", scale: 3.455321e0, exponent: 0.15, r: 0.5948 },
    LoopNorm { metric: "vmaf-v1", map: "aim_2_v", scale: 3.292563e0, exponent: 0.15, r: 0.5963 },
    LoopNorm { metric: "vmaf-v1", map: "aim_2_d", scale: 1.717178e1, exponent: 0.25, r: 0.4265 },
    LoopNorm { metric: "vmaf-v1", map: "aim_3_h", scale: 1.582460e0, exponent: 0.1, r: 0.5600 },
    LoopNorm { metric: "vmaf-v1", map: "aim_3_v", scale: 1.549933e0, exponent: 0.1, r: 0.7257 },
    LoopNorm { metric: "vmaf-v1", map: "aim_3_d", scale: 1.834066e0, exponent: 0.1, r: 0.4246 },
    LoopNorm { metric: "vsi", map: "quality", scale: 7.457903e-1, exponent: 0.1, r: -0.1191 },
    LoopNorm { metric: "zensim", map: "diffmap", scale: 8.402298e0, exponent: 0.35, r: 0.8494 },
];

/// libjxl's `TileDistMap` constant, which jxl-encoder's butteraugli loop
/// multiplies each block's 16-norm by.
pub(crate) const K_TILE_NORM: f32 = 1.2;

/// The error reading of `map` at full resolution: `1 − v` for similarity,
/// the value itself for difference and detection-probability maps. Each
/// pixel takes the native cell that owns it; pixels no cell owns, and
/// undefined native values, read as zero error. Feature maps (pooling
/// weights, information terms) have no error direction and are refused.
pub(crate) fn error_pixels(m: &SpatialMap, width: u32, height: u32) -> Result<Vec<f32>, String> {
    let read: fn(f32) -> f32 = match m.polarity {
        "similarity" => |v| 1.0 - v,
        "difference" | "probability" => |v| v,
        _ => {
            return Err(format!(
                "{} is a feature map with no error direction; choose an error or similarity map",
                m.name
            ));
        }
    };
    let (mw, mh) = (m.width as usize, m.height as usize);
    if mw == 0 || mh == 0 || m.values.len() != mw * mh {
        return Err(format!("{} has an empty or malformed native grid", m.name));
    }
    let (w, h) = (width as usize, height as usize);
    let mut out = Vec::new();
    out.try_reserve_exact(w * h).map_err(|e| e.to_string())?;
    for y in 0..h {
        for x in 0..w {
            let cell = match m.geometry {
                Some(g) => g
                    .cell_at(x as u32, y as u32)
                    .map(|(i, j)| j as usize * mw + i as usize),
                // Proportional ownership for grids without geometry
                // (DSSIM's coarser scales).
                None => Some((y * mh / h).min(mh - 1) * mw + (x * mw / w).min(mw - 1)),
            };
            let e = cell.map_or(0.0, |c| read(m.values[c]));
            out.push(if e.is_finite() { e.max(0.0) } else { 0.0 });
        }
    }
    Ok(out)
}

/// `scale · v^exponent` in place.
pub(crate) fn normalize(values: &mut [f32], scale: f64, exponent: f64) {
    for v in values {
        *v = (scale * f64::from(*v).powf(exponent)) as f32;
    }
}

/// Each 8×8 block's `K_TILE_NORM · (Σ v¹⁶ / n)^(1/16)`, row-major over
/// `ceil(w/8) × ceil(h/8)` blocks; edge blocks average their covered pixels.
pub(crate) fn tile_norms(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    let (bx, by) = (width.div_ceil(8), height.div_ceil(8));
    let mut sum = vec![0f64; bx * by];
    let mut count = vec![0u32; bx * by];
    for y in 0..height {
        for x in 0..width {
            let b = (y / 8) * bx + x / 8;
            sum[b] += f64::from(values[y * width + x]).powi(16);
            count[b] += 1;
        }
    }
    sum.iter()
        .zip(&count)
        .map(|(&s, &n)| K_TILE_NORM * (s / f64::from(n)).powf(1.0 / 16.0) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(polarity: &'static str, values: Vec<f32>, w: u32, h: u32) -> SpatialMap {
        SpatialMap {
            polarity,
            similarity: polarity == "similarity",
            ..SpatialMap::difference("m", w, h, values)
        }
    }

    #[test]
    fn similarity_reads_as_error_and_feature_maps_are_refused() {
        let m = map("similarity", vec![1.0, 0.75, f32::NAN, 0.5], 2, 2);
        assert_eq!(error_pixels(&m, 2, 2).unwrap(), [0.0, 0.25, 0.0, 0.5]);
        assert!(error_pixels(&map("feature", vec![1.0; 4], 2, 2), 2, 2).is_err());
    }

    #[test]
    fn coarse_grids_spread_proportionally() {
        let m = map("difference", vec![1.0, 2.0], 2, 1);
        assert_eq!(error_pixels(&m, 4, 1).unwrap(), [1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn tile_norm_of_a_constant_block_is_the_constant() {
        let v = vec![0.5f32; 16 * 9];
        let t = tile_norms(&v, 16, 9);
        assert_eq!(t.len(), 4);
        for x in t {
            assert!((x - K_TILE_NORM * 0.5).abs() < 1e-6, "{x}");
        }
        let mut n = vec![2.0f32, 4.0];
        normalize(&mut n, 0.5, 2.0);
        assert_eq!(n, [2.0, 8.0]);
    }
}
