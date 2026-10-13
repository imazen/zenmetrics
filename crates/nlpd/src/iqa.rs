//! `NLPD` as the JPEG AIC-4 evaluation harness computes it — the
//! `IQA_pytorch` package (`dingkeyan93/IQA-optimization`, the Alex
//! Hepburn `nlpd-tensorflow` lineage — NOT the later `pyiqa` port and
//! NOT the Laparra-author `NLPD_Pytorch` that [`crate::score_rgb_u8`]
//! mirrors).
//!
//! Differences vs the crate's default `nlpd` entry:
//!
//! - **Luma only.** The harness (`jpeg-ai-qaf` `metrics.py` `NLPD_IQA`)
//!   instantiates `NLPD(channels=1)` and feeds the BT.709 Y plane in
//!   `[0, 1]`, quantized back onto the 8-bit grid
//!   (`round(clamp01(Y)·255)/255`).
//! - **Downsample phase.** Reflect-pad of 2 (PyTorch `ReflectionPad2d`)
//!   before the stride-2 `filt` convolve at every level; the crate's
//!   Laparra-faithful path uses a parity-dependent pad that shifts
//!   phase by one pixel on even dims.
//! - **Upsample.** `bilinear(×2, align_corners=True)` on the low band to
//!   exactly `2·lw × 2·lh`, then the 5×5 low-pass; an odd upstream dim
//!   leaves a one-pixel overshoot which `F.interpolate` (default
//!   `nearest`) crops.
//! - **Six loop bands, no residual band.** The loop runs `k = 6` times
//!   and the terminal low-pass is consumed as the 6th Laplacian
//!   difference source — never appended raw.
//! - **Pooling.** Plain mean of the per-level RMS — not the
//!   `(Σ rms^0.6)^(1/0.6)` aggregation of the author implementation.
//!
//! Validation: numpy replica of this module vs `metrics_fullres.csv`
//! `NLPD` on 53 AIC-4 pairs (2026-09-26): med |Δ| 0.0026, max |Δ| 0.0067
//! on a column range of ~[0.03, 0.22] (≈3% relative; residual is f32
//! op-order + torch bilinear edge detail). The crate's RGB `nlpd` reads
//! ≈20.5× the column (~`6^(1/0.6)` pooling factor plus chroma energy).

use super::{Error, Scratch, buf_give, buf_take, filter5_horizontal, simd};

/// Score two 8-bit sRGB images with the harness ingress: BT.709 luma
/// `Y = 0.2126R + 0.7152G + 0.0722B` on `[0, 1]`, clamped and quantized
/// back to the 8-bit grid — `DataClass.rgb_to_yuv` + `round_plane(.., 8)`.
pub fn score_rgb_u8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<f64, Error> {
    if width < 96 || height < 96 {
        return Err(Error::InvalidDimensions);
    }
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or(Error::InvalidDimensions)?;
    if reference.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: reference.len(),
        });
    }
    if distorted.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: distorted.len(),
        });
    }
    let n = width * height;
    let ry = y709_u8_grid(reference, n);
    let dy = y709_u8_grid(distorted, n);
    Ok(score_planes(ry, dy, width, height))
}

/// Score two single-channel luma planes already in `[0, 1]` — no
/// quantization is applied here; callers reproducing the AIC-4 column
/// should pass the `round(Y·255)/255` grid themselves.
pub fn score_y_f32(
    reference_y: &[f32],
    distorted_y: &[f32],
    width: usize,
    height: usize,
) -> Result<f64, Error> {
    if width < 96 || height < 96 {
        return Err(Error::InvalidDimensions);
    }
    let expected = width.checked_mul(height).ok_or(Error::InvalidDimensions)?;
    if reference_y.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: reference_y.len(),
        });
    }
    if distorted_y.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: distorted_y.len(),
        });
    }
    if reference_y
        .iter()
        .chain(distorted_y.iter())
        .any(|v| !v.is_finite())
    {
        return Err(Error::NonFiniteInput);
    }
    let mut ry = buf_take(expected);
    ry.copy_from_slice(reference_y);
    let mut dy = buf_take(expected);
    dy.copy_from_slice(distorted_y);
    Ok(score_planes(ry, dy, width, height))
}

/// Harness BT.709 luma: f32 weighted sum of the u8 codes normalized to
/// `[0, 1]`, then `round(Y·255)/255` (the `round_plane(yuv, 8)` step).
fn y709_u8_grid(rgb: &[u8], n: usize) -> Vec<f32> {
    // SIMD path — see `simd::y709_grid` for the half-away `roundf`
    // emulation; per-lane arithmetic is identical to the scalar form.
    let mut y = buf_take(n);
    simd::y709_grid(rgb, &mut y[..n]);
    y
}

/// Six loop-levels, mean of per-level RMS — the dingkeyan93 pooling.
fn score_planes(mut ref_p: Vec<f32>, mut dis_p: Vec<f32>, w: usize, h: usize) -> f64 {
    let (mut w, mut h) = (w, h);
    let mut rms_sum = 0.0_f64;
    // The 2× upsampled grid can overshoot the level-0 extent by one row
    // and column on odd dims; the constant covers that slack.
    let mut ref_scratch = Scratch::new(w * h + 2 * (w + h) + 4);
    let mut dis_scratch = Scratch::new(w * h + 2 * (w + h) + 4);
    for level in 0..6 {
        let (low_r, lw, lh) = iqa_level(&ref_p, w, h, level, &mut ref_scratch);
        let (low_d, _, _) = iqa_level(&dis_p, w, h, level, &mut dis_scratch);
        let sq = simd::squared_difference(
            &ref_scratch.normalized[..w * h],
            &dis_scratch.normalized[..w * h],
        );
        rms_sum += (sq / (w * h) as f64).sqrt();
        buf_give(core::mem::replace(&mut ref_p, low_r));
        buf_give(core::mem::replace(&mut dis_p, low_d));
        w = lw;
        h = lh;
    }
    buf_give(ref_p);
    buf_give(dis_p);
    for v in ref_scratch.into_bufs().chain(dis_scratch.into_bufs()) {
        buf_give(v);
    }
    rms_sum / 6.0
}

/// One dingkeyan93 pyramid level: reflect-2 downsample →
/// bilinear-align-corners ×2 upsample → 5×5 low-pass → nearest-crop →
/// subtract → divisive-normalize into `scratch.normalized`. The whole
/// post-downsample chain runs as one fused row-ring pass in
/// `simd::normalize_filtered` — no `up`/`smooth`/`recon` plane is
/// materialized. Returns the low band.
fn iqa_level(
    src: &[f32],
    w: usize,
    h: usize,
    level: usize,
    scratch: &mut Scratch,
) -> (Vec<f32>, usize, usize) {
    // Downsample: ReflectionPad2d(2) + stride-2 valid conv → out = ceil/2,
    // the fixed pad-2 phase (not the Laparra path's parity-dependent pad).
    let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
    filter5_horizontal(src, w, h, &mut scratch.tmp_h[..ow * h], ow, 2, 2);
    let mut low = buf_take(ow * oh);
    simd::filter5_vertical(&scratch.tmp_h[..ow * h], ow, h, &mut low, oh, 2, 2);
    // Upsample targets the ×2 grid (2ow×2oh ≥ w×h); the crop back to w×h
    // is the same `F.interpolate`-nearest gather the ring applies per row.
    simd::normalize_filtered(
        src,
        &low,
        ow,
        oh,
        2 * ow,
        2 * oh,
        w,
        h,
        level,
        &mut scratch.normalized[..w * h],
    );
    (low, ow, oh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_asymmetry() {
        let w = 97;
        let h = 99;
        let reference = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 97) % 256) as u8)
            .collect::<Vec<_>>();
        let mut changed = reference.clone();
        for p in changed.as_chunks_mut::<3>().0.iter_mut().step_by(19) {
            p[0] = p[0].saturating_add(20);
        }
        assert_eq!(score_rgb_u8(&reference, &reference, w, h).unwrap(), 0.0);
        let a = score_rgb_u8(&reference, &changed, w, h).unwrap();
        // numpy replica of IQA_pytorch NLPD(channels=1) on the quantized
        // BT.709 planes of this same pair (dingkeyan93 semantics, f64).
        assert!((a - 0.013732910242701013).abs() < 1e-4, "{a}");
        let b = score_rgb_u8(&changed, &reference, w, h).unwrap();
        assert!(a > 0.0 && (a - b).abs() < 1e-9);
    }

    #[test]
    fn y_f32_entry() {
        let w = 97;
        let h = 99;
        let ry = (0..w * h)
            .map(|i| ((i * 13 + i / 53) % 251) as f32 / 251.0)
            .collect::<Vec<_>>();
        let mut dy = ry.clone();
        for p in dy.iter_mut().step_by(23) {
            *p = (*p + 0.15).min(1.0);
        }
        let a = score_y_f32(&ry, &dy, w, h).unwrap();
        // Same replica on the unquantized f32 planes.
        assert!(a > 0.0 && a < 0.5, "{a}");
        assert_eq!(score_y_f32(&ry, &ry, w, h).unwrap(), 0.0);
        let bad = score_y_f32(&ry[..ry.len() - 1], &dy, w, h);
        assert!(matches!(bad, Err(Error::InvalidLength { .. })));
        let small = score_y_f32(&ry[..95 * 96], &ry[..95 * 96], 95, 96);
        assert!(matches!(small, Err(Error::InvalidDimensions)));
    }
}
