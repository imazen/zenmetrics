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
/// Used by the `sweep` subcommand's `--feature-output <path.parquet>` and by
/// the jobexec executor's zensim feature-row emission (not `sweep`-gated —
/// the jobexec-only build needs it too; it has no sweep dependency).
#[allow(dead_code)] // not every feature shape calls it
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

/// Reflect-pad an RGB8 image up to a `min_dim` floor on each axis (mirror /
/// OpenCV `BORDER_REFLECT` semantics `fedcba|abcdef|fedcba`). Returns the pixels
/// borrowed unchanged when already ≥ `min_dim` on both axes, else an owned padded
/// buffer + its new dims.
///
/// Why: the two CPU extractors in [`score_with_features_v2ab`] disagree on
/// sub-64px handling — `compute_v2_features` reflect-pads to the 64px pyramid
/// floor internally, while `compute_zensim_with_config` runs UNPADDED (degenerate
/// coarse scales at 8–63px, hard-errors < 8px). Feeding both the SAME padded ≥64px
/// input makes the 372 and 348 blocks features of one consistent image, and
/// matches the GPU path's own reflect-to-64 (see the `ZensimGpu` arm in cache.rs).
#[cfg(feature = "cpu-metrics")]
fn reflect_pad_to_min(img: &Rgb8Image, min_dim: u32) -> std::borrow::Cow<'_, [u8]> {
    use std::borrow::Cow;
    let (w, h) = (img.width, img.height);
    if w >= min_dim && h >= min_dim {
        return Cow::Borrowed(&img.pixels);
    }
    let (nw, nh) = (w.max(min_dim), h.max(min_dim));
    // BORDER_REFLECT index map (period 2n): t -> source column/row in [0, n).
    let mirror = |t: u32, n: u32| -> usize {
        if n <= 1 {
            return 0;
        }
        let p = 2 * n;
        let m = t % p;
        (if m < n { m } else { p - 1 - m }) as usize
    };
    let (wu, _hu) = (w as usize, h as usize);
    let mut px = Vec::with_capacity((nw as usize) * (nh as usize) * 3);
    for ty in 0..nh {
        let sy = mirror(ty, h);
        for tx in 0..nw {
            let sx = mirror(tx, w);
            let o = (sy * wu + sx) * 3;
            px.extend_from_slice(&img.pixels[o..o + 3]);
        }
    }
    Cow::Owned(px)
}

/// v2 "append-only" CPU zensim extraction: the v1 (PreviewV0_2-weighted) score
/// plus the concatenated **720**-feature vector `[v1-372 ++ v2-348]`, computed
/// entirely on the CPU `zensim` crate — NO GPU kernel.
///
/// This is the path the fleet uses while the GPU zensim kernel is disabled
/// pending v2 GPU-port validation (2026-07-19): the GPU crate only implements
/// the v1 372 regime, and `zensim` was rewritten with an additive v2 extractor
/// (`feature_v2`, 348 bounded features). Per the zensim crate's own append-only
/// design (there is no single-pass 720 API), we compute the two blocks and
/// concatenate:
///   - v1-372 "with-iw / V_22": `compute_zensim_with_config` with
///     `extended_features + compute_iw_features` (needs the `training` feature).
///     Its `.score()` uses the `PreviewV0_2` linear weights (the extra 228..372
///     features carry no weight), matching the canonical `score_zensim` column.
///   - v2-348 bounded: `Zensim::compute_v2_features` (needs `feature-regime-v2`).
///     v2 has NO score/profile yet — features only, for future trainer work.
///
/// Both inputs are reflect-padded to ≥64px first so the two blocks agree on tiny
/// cells (see [`reflect_pad_to_min`]). v2 is an iteration-1 scalar reference
/// extractor — measurably slower than v1's SIMD path; that cost is expected.
#[cfg(feature = "cpu-metrics")]
pub fn score_with_features_v2ab(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    score_with_features_regime(reference, distorted, crate::metrics::ZensimFeatureRegime::V2Ab)
}

/// CPU zensim score + the regime-appropriate feature vector, replacing the GPU
/// `run_gpu_via_umbrella(Zensim, …)` path while the GPU kernel is disabled
/// (2026-07-19). All four regimes compute on the `zensim` crate:
///   - `Basic` (228)    — `ZensimConfig { extended: false, iw: false }`
///   - `Extended` (300) — `{ extended: true, iw: false }`
///   - `WithIw` (372)   — `{ extended: true, iw: true }` (matches the old GPU
///                         `IwWeighted` layout the sidecars used)
///   - `V2Ab` (720)     — `WithIw` (372) ++ `compute_v2_features` (348)
///
/// The score is ALWAYS the canonical `PreviewV0_2.compute()` value (bit-exact,
/// regime-independent) — NOT the config path's own `.score()`, which diverges
/// ~5e-4. Both inputs are reflect-padded to ≥64px first so every block sees one
/// consistent image (see [`reflect_pad_to_min`]).
#[cfg(feature = "cpu-metrics")]
pub fn score_with_features_regime(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    regime: crate::metrics::ZensimFeatureRegime,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    use crate::metrics::ZensimFeatureRegime as R;
    use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "zensim: reference ({}×{}) and distorted ({}×{}) differ in size",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    // Pad both identically to the 64px pyramid floor (no-op when already ≥64).
    let rp = reflect_pad_to_min(reference, 64);
    let dp = reflect_pad_to_min(distorted, 64);
    let w = reference.width.max(64) as usize;
    let h = reference.height.max(64) as usize;
    // Zero-copy &[u8] -> &[[u8;3]]; len is w*h*3 by construction so this is exact.
    // Bind Cow -> &[u8] first so cast_slice's generic `&[A]` infers A = u8.
    let rp: &[u8] = &rp;
    let dp: &[u8] = &dp;
    let src3: &[[u8; 3]] = bytemuck::cast_slice(rp);
    let dst3: &[[u8; 3]] = bytemuck::cast_slice(dp);

    // Canonical v1 score: the standard PreviewV0_2 scoring path (NOT the config
    // path's `.score()`, which diverges ~5e-4). Same padded pixels as the
    // features. This is the `score_zensim` column value, regime-independent.
    let score = Zensim::new(ZensimProfile::PreviewV0_2)
        .compute(&RgbSlice::new(src3, w, h), &RgbSlice::new(dst3, w, h))
        .map_err(|e| format!("zensim: PreviewV0_2 score: {e:?}"))?
        .score();

    // v1 feature block (228 / 300 / 372 by regime flags).
    let (extended, iw) = match regime {
        R::Basic => (false, false),
        R::Extended => (true, false),
        R::WithIw | R::V2Ab => (true, true),
    };
    let mut cfg = ZensimConfig::default();
    cfg.extended_features = extended;
    cfg.compute_iw_features = iw;
    let v1 = compute_zensim_with_config(src3, dst3, w, h, cfg)
        .map_err(|e| format!("zensim: compute_zensim_with_config: {e:?}"))?;
    let mut features = v1.into_features();

    // V2Ab appends the v2 bounded 348 block (profile irrelevant to v2).
    if regime == R::V2Ab {
        let v2 = Zensim::new(ZensimProfile::PreviewV0_2)
            .compute_v2_features(&RgbSlice::new(src3, w, h), &RgbSlice::new(dst3, w, h))
            .map_err(|e| format!("zensim: v2-348 compute_v2_features: {e:?}"))?;
        features.extend_from_slice(v2.features());
    }

    let want = regime.total_features();
    if features.len() != want {
        return Err(format!(
            "zensim: regime {regime:?} expected {want} features, got {}",
            features.len()
        )
        .into());
    }
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

    /// v2ab must emit exactly 720 = v1-372 ++ v2-348, and its score must be
    /// BIT-IDENTICAL to the canonical `PreviewV0_2.compute()` score on the same
    /// (≥64px, so unpadded) pixels — the `score_zensim` column is exact. (This is
    /// why the function scores on the dedicated V0_2 path, not the config path's
    /// own `.score()`, which diverges ~5e-4.)
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn v2ab_score_is_exact_preview_v0_2() {
        use zensim::{RgbSlice, Zensim, ZensimProfile};
        let (w, h) = (96, 72);
        let reference = synth(1, w, h);
        let distorted = synth(101, w, h);
        let (score, feats) = score_with_features_v2ab(&reference, &distorted).unwrap();
        assert_eq!(feats.len(), 720, "v2ab must emit v1-372 ++ v2-348 = 720");
        assert!(score.is_finite(), "v2ab score must be finite");
        assert!(feats.iter().all(|v| v.is_finite()), "all 720 features finite");

        let r3: Vec<[u8; 3]> = reference.pixels.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let d3: Vec<[u8; 3]> = distorted.pixels.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let pv02 = Zensim::new(ZensimProfile::PreviewV0_2)
            .compute(&RgbSlice::new(&r3, w as usize, h as usize), &RgbSlice::new(&d3, w as usize, h as usize))
            .unwrap()
            .score();
        assert_eq!(
            score.to_bits(),
            pv02.to_bits(),
            "v2ab score {score} must be bit-identical to PreviewV0_2.compute() {pv02}"
        );
    }

    /// A sub-64px pair must reflect-pad and still emit exactly 720 (both blocks
    /// see the same ≥64 image — no sub-64 disagreement, no error).
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn v2ab_tiny_image_pads_to_720() {
        let reference = synth(2, 20, 12);
        let distorted = synth(202, 20, 12);
        let (score, feats) = score_with_features_v2ab(&reference, &distorted).unwrap();
        assert_eq!(feats.len(), 720);
        assert!(score.is_finite() && feats.iter().all(|v| v.is_finite()));
    }

    /// Throughput cost of the v2 append regime vs v1 with-iw, on a realistic
    /// image. v2 is an iteration-1 SCALAR extractor, so this quantifies the
    /// per-cell fleet slowdown. Run: `cargo test -p zenmetrics-cli --lib --release
    /// v2ab_throughput -- --ignored --nocapture`.
    #[cfg(feature = "cpu-metrics")]
    #[test]
    #[ignore = "timing measurement, not a correctness gate"]
    fn v2ab_throughput_vs_withiw() {
        use crate::metrics::ZensimFeatureRegime as R;
        use std::time::Instant;
        let (w, h) = (1024, 1024);
        let reference = synth(1, w, h);
        let distorted = synth(101, w, h);
        // warm up
        let _ = score_with_features_regime(&reference, &distorted, R::WithIw).unwrap();
        let _ = score_with_features_regime(&reference, &distorted, R::V2Ab).unwrap();
        let n = 5;
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = score_with_features_regime(&reference, &distorted, R::WithIw).unwrap();
        }
        let iw = t0.elapsed().as_secs_f64() / n as f64;
        let t1 = Instant::now();
        for _ in 0..n {
            let _ = score_with_features_regime(&reference, &distorted, R::V2Ab).unwrap();
        }
        let v2 = t1.elapsed().as_secs_f64() / n as f64;
        eprintln!(
            "ZENSIM {w}x{h}: with-iw(372) {:.1} ms/pair | v2-ab(720) {:.1} ms/pair | v2/v1 = {:.2}x",
            iw * 1e3,
            v2 * 1e3,
            v2 / iw
        );
    }

    /// Same inputs → bit-identical output across runs (guards the v2 path against
    /// nondeterminism, e.g. from internal parallelism).
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn v2ab_deterministic() {
        let reference = synth(3, 80, 64);
        let distorted = synth(303, 80, 64);
        let a = score_with_features_v2ab(&reference, &distorted).unwrap();
        let b = score_with_features_v2ab(&reference, &distorted).unwrap();
        assert_eq!(a.0.to_bits(), b.0.to_bits(), "v2ab score must be deterministic");
        assert_eq!(a.1.len(), b.1.len());
        for (i, (x, y)) in a.1.iter().zip(b.1.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "v2ab feature[{i}] must be deterministic");
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
