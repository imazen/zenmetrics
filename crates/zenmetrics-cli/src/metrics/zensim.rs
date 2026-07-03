#![forbid(unsafe_code)]

//! CPU zensim score via the `zensim` crate.

use crate::decode::Rgb8Image;

pub(crate) fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    use zensim::{PixelFormat, StridedBytes, Zensim};

    let z = Zensim::new(zensim::ZensimProfile::latest_preview());
    let w = reference.width as usize;
    let h = reference.height as usize;
    let stride = w * 3;

    let src = StridedBytes::try_new(&reference.pixels, w, h, stride, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid reference RGB slice: {e:?}"))?;
    let dst = StridedBytes::try_new(&distorted.pixels, w, h, stride, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid distorted RGB slice: {e:?}"))?;
    let result = z
        .compute(&src, &dst)
        .map_err(|e| format!("zensim: {e:?}"))?;
    Ok(result.score())
}

/// A reference image whose sRGB→XYB multi-scale pyramid has already been built,
/// so many distorted images can be scored against it without rebuilding the
/// reference side each time.
///
/// This is the CPU-side equivalent of the GPU `MetricCache` cached-reference
/// slot. It pays off in the **metric-only** phase (`score-pairs` over persisted
/// variants), where every distorted variant of one source shares the same
/// reference and the encode is no longer in the loop to mask the saving. (In the
/// monolithic `sweep` the per-cell encode dominates, so it buys nothing there.)
/// The amortized score is bit-identical to [`score`] — asserted by the
/// `precomputed_matches_score` test — so it is a pure cost reduction.
pub(crate) struct PrecomputedRef {
    inner: zensim::PrecomputedReference,
}

/// Build a [`PrecomputedRef`] from `reference`: convert to XYB and build the
/// downscale pyramid once. Reuse across many [`score_with_precomputed`] calls.
pub(crate) fn precompute_ref(
    reference: &Rgb8Image,
) -> Result<PrecomputedRef, Box<dyn std::error::Error>> {
    use zensim::{PixelFormat, StridedBytes, Zensim};
    let z = Zensim::new(zensim::ZensimProfile::latest_preview());
    let w = reference.width as usize;
    let h = reference.height as usize;
    let src = StridedBytes::try_new(&reference.pixels, w, h, w * 3, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid reference RGB slice: {e:?}"))?;
    let inner = z
        .precompute_reference(&src)
        .map_err(|e| format!("zensim: precompute_reference: {e:?}"))?;
    Ok(PrecomputedRef { inner })
}

/// Score `distorted` against a [`PrecomputedRef`]. Bit-identical to [`score`]
/// called with the original reference (see `precomputed_matches_score`) but
/// skips rebuilding the reference's XYB pyramid.
pub(crate) fn score_with_precomputed(
    precomputed: &PrecomputedRef,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    use zensim::{PixelFormat, StridedBytes, Zensim};
    let z = Zensim::new(zensim::ZensimProfile::latest_preview());
    let w = distorted.width as usize;
    let h = distorted.height as usize;
    let dst = StridedBytes::try_new(&distorted.pixels, w, h, w * 3, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid distorted RGB slice: {e:?}"))?;
    let result = z
        .compute_with_ref(&precomputed.inner, &dst)
        .map_err(|e| format!("zensim: compute_with_ref: {e:?}"))?;
    Ok(result.score())
}

/// Score plus the full extended 300-feature vector (4 scales × 3 channels ×
/// 25 features/channel at the default profile).
///
/// The `score` returned here is identical to [`score`] — the extra 72
/// masked features have zero weight in the trained profile, so adding
/// them changes neither the weighted distance nor the score. The extra
/// cost on top of [`score`] is the masking pass over the same multi-scale
/// stats; the score-relevant work is shared.
///
/// Only used by the `sweep` subcommand when the user passes
/// `--feature-output <path.parquet>`.
#[cfg(feature = "sweep")]
pub fn score_with_features(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    use zensim::{PixelFormat, StridedBytes, Zensim};

    let z = Zensim::new(zensim::ZensimProfile::latest_preview());
    let w = reference.width as usize;
    let h = reference.height as usize;
    let stride = w * 3;

    let src = StridedBytes::try_new(&reference.pixels, w, h, stride, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid reference RGB slice: {e:?}"))?;
    let dst = StridedBytes::try_new(&distorted.pixels, w, h, stride, PixelFormat::Srgb8Rgb)
        .map_err(|e| format!("zensim: invalid distorted RGB slice: {e:?}"))?;
    let result = z
        .compute_extended_features(&src, &dst)
        .map_err(|e| format!("zensim: {e:?}"))?;
    let score = result.score();
    let features = result.into_features();
    Ok((score, features))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(seed: u32, w: u32, h: u32) -> Rgb8Image {
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        let mut s = seed.wrapping_add(1);
        for y in 0..h {
            for x in 0..w {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                pixels.push(((x.wrapping_mul(3).wrapping_add(s)) & 0xff) as u8);
                pixels.push(((y.wrapping_mul(5).wrapping_add(s >> 8)) & 0xff) as u8);
                pixels.push((((x ^ y).wrapping_mul(7).wrapping_add(s >> 16)) & 0xff) as u8);
            }
        }
        Rgb8Image {
            pixels,
            width: w,
            height: h,
        }
    }

    /// The amortized precompute-ref path MUST be bit-identical to the plain
    /// `score()` path — a perceptual metric the picker trains on is pixel-sacred.
    /// Build the ref ONCE, score many distinct distorted images (the score-pairs
    /// usage pattern), each must equal `score()` to the bit.
    #[test]
    fn precomputed_matches_score() {
        let reference = synth(1, 96, 72);
        let pre = precompute_ref(&reference).unwrap();
        for d in 0..5u32 {
            let distorted = synth(100 + d, 96, 72);
            let direct = score(&reference, &distorted).unwrap();
            let amortized = score_with_precomputed(&pre, &distorted).unwrap();
            assert_eq!(
                direct.to_bits(),
                amortized.to_bits(),
                "precomputed-ref zensim score must be bit-identical to score() \
                 (d={d}): direct={direct} amortized={amortized}"
            );
        }
    }
}

/// PU-linear integrated feature extraction (Profile B v3 regime): absolute
/// nits → `Zensim::compute_pu_linear_extended_features` — no u8 shell, no
/// display-peak anchor. Score is the integrated-PU score (the validated
/// zensim HDR feeding); features share the sRGB extraction layout.
#[cfg(feature = "cpu-metrics")]
pub fn score_with_features_pu_linear(
    ref_nits: &[f32],
    dist_nits: &[f32],
    width: usize,
    height: usize,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    use zensim::Zensim;
    let z = Zensim::new(zensim::ZensimProfile::latest_preview());
    let stride = width * 3;
    let result = z
        .compute_pu_linear_extended_features(ref_nits, dist_nits, width, height, stride, stride)
        .map_err(|e| format!("zensim pu-linear: {e:?}"))?;
    let score = result.score();
    let features = result.into_features();
    Ok((score, features))
}
