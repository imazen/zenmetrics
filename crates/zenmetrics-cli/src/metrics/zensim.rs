#![forbid(unsafe_code)]

//! CPU zensim score via the `zensim` crate.

use crate::decode::Rgb8Image;

// ─── Runtime profile selection (zenmetrics D14) ───────────────────────────────
//
// The CLI's zensim SCORE path never touches `MetricParams` — `metrics::run_metric`
// calls `zensim::score()` here directly rather than going through the umbrella —
// so the `--zensim-profile` / `--zensim-bake` selection has to be read here too.
// The selector itself lives in ONE place, `zenmetrics_api::zensim_profile`; this
// is a reader, not a second implementation.
//
// Only the SCORE-bearing sites below consult it. The feature-extraction entry
// points are deliberately pinned (`PreviewV0_2` for the v2-348 block,
// `codec_target()` for the folded-streaming driver) so their regimes stay
// byte-identical; overriding those would silently change extracted features,
// which is not what a profile flag means.

/// Env fallback for the profile selection, so a fleet worker (or any
/// consumer of this crate as a library, e.g. `zenfleet-vastai`, which never
/// runs `main()`) can select without argv. Same grammar as `--zensim-profile`:
/// a built-in name OR a path to a ZNPR bake.
pub const ZENSIM_PROFILE_ENV: &str = "ZENMETRICS_ZENSIM_PROFILE";

/// Resolved-once env selection. `Ok(None)` = nothing requested.
static ENV_PROFILE: std::sync::OnceLock<Result<Option<zensim::ZensimProfile>, String>> =
    std::sync::OnceLock::new();

/// The zensim profile the CLI's score paths use.
///
/// Precedence: an explicit override installed by `--zensim-profile` /
/// `--zensim-bake` (via `zenmetrics_api::zensim_profile::set_default`) →
/// `ZENMETRICS_ZENSIM_PROFILE` → `ZensimProfile::latest_preview()`. Unset,
/// this is byte-identical to the literal each site used to hard-code.
///
/// A malformed env value is an ERROR, never a silent fall back to the
/// default — scoring a whole fleet with the wrong model because a path
/// typo was swallowed is exactly the failure this must not have.
pub(crate) fn selected_profile() -> Result<zensim::ZensimProfile, Box<dyn std::error::Error>> {
    if let Some(p) = zenmetrics_api::zensim_profile::override_profile() {
        return Ok(p);
    }
    let from_env = ENV_PROFILE.get_or_init(|| match std::env::var(ZENSIM_PROFILE_ENV) {
        Ok(spec) if !spec.trim().is_empty() => zenmetrics_api::zensim_profile::resolve(spec.trim())
            .map(Some)
            .map_err(|e| format!("{ZENSIM_PROFILE_ENV}={spec:?}: {e}")),
        _ => Ok(None),
    });
    match from_env {
        Ok(Some(p)) => Ok(*p),
        Ok(None) => Ok(zensim::ZensimProfile::latest_preview()),
        Err(msg) => Err(msg.clone().into()),
    }
}


pub(crate) fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    use zensim::{PixelFormat, StridedBytes, Zensim};

    let z = Zensim::new(selected_profile()?);
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
    let z = Zensim::new(selected_profile()?);
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
    let z = Zensim::new(selected_profile()?);
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

    let z = Zensim::new(selected_profile()?);
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
///   - v2-348 bounded: `Zensim::compute_v2_features` (needs `feature-regime-v2`).
///
/// FEATURES ONLY — no score. Both inputs are reflect-padded to ≥64px first so the
/// two blocks agree on tiny cells (see [`reflect_pad_to_min`]). NOTE: v2's
/// extractor IS SIMD-fused (magetypes; the zensim `metric.rs` "iteration-1 scalar"
/// doc is STALE — see the crate's own `feature_v2` module doc). The ~1.5× cost vs
/// v1-372 is the extra 348-feature compute PLUS a redundant XYB pyramid (v1 and v2
/// each build it independently). The real speedup is a combined 372+348
/// single-pyramid API in the zensim crate, not a further fuse.
#[cfg(feature = "cpu-metrics")]
#[allow(dead_code)] // convenience wrapper + test entry; the fleet calls extract_features_regime
pub fn extract_features_v2ab(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    extract_features_regime(
        reference,
        distorted,
        crate::metrics::ZensimFeatureRegime::V2Ab,
    )
}

/// CPU zensim FEATURE extraction (NO score) for the requested regime, replacing
/// the GPU `run_gpu_via_umbrella(Zensim, …)` path — the GPU kernel is disabled
/// (2026-07-19) and this is features-only (a v2 score head is trained later; the
/// v1 score column is not wanted). All four regimes compute on the `zensim` crate:
///   - `Basic` (228)    — `ZensimConfig { extended: false, iw: false }`
///   - `Extended` (300) — `{ extended: true, iw: false }`
///   - `WithIw` (372)   — `{ extended: true, iw: true }` (the old GPU IwWeighted layout)
///   - `V2Ab` (720)     — `WithIw` (372) ++ `compute_v2_features` (348)
///
/// Both inputs are reflect-padded to ≥64px first so every block sees one
/// consistent image (see [`reflect_pad_to_min`]).
#[cfg(feature = "cpu-metrics")]
pub fn extract_features_regime(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    regime: crate::metrics::ZensimFeatureRegime,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    use crate::metrics::ZensimFeatureRegime as R;
    use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "zensim: reference ({}×{}) and distorted ({}×{}) differ in size",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    // Folded+append 924/944: the streaming walk does its OWN sub-64 reflect-pad,
    // so it must see the RAW image — this fn's external padding would diverge
    // from the `v2_ab_extract` foldapp/foldapp2 driver on tiny cells (regime
    // purity).
    if regime == R::Folded720Append
        || regime == R::Folded720Append2
        || regime == R::Folded720Append2Carriers
        || regime == R::Folded720Append2Pools
    {
        return extract_features_folded_streaming(reference, distorted, regime);
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

    // FEATURES ONLY — no score. The GPU zensim kernel is disabled (2026-07-19) and
    // the v2 backfill wants only the feature vector; dropping the separate
    // PreviewV0_2 score pass is the biggest cheap saving vs score+features.
    // v1 feature block (228 / 300 / 372 by regime flags).
    let (extended, iw) = match regime {
        R::Basic => (false, false),
        R::Extended => (true, false),
        R::WithIw | R::V2Ab => (true, true),
        // Handled by the early return above (streaming-only, no v1 block).
        R::Folded720Append
        | R::Folded720Append2
        | R::Folded720Append2Carriers
        | R::Folded720Append2Pools => {
            unreachable!("folded streaming early-returns above")
        }
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
    Ok(features)
}

/// Folded+append streaming extraction (regimes `Folded720Append` = 924,
/// `Folded720Append2` = 944, and the two live-pool 944 variants
/// `Folded720Append2Carriers` / `Folded720Append2Pools`) — the STREAMING-ONLY walk (zensim C5,
/// 2026-07-26): O(width) rolling planes, no prepared reference (the cached-ref
/// machinery was deleted in C5), per-thread `V2Scratch` reuse across calls.
/// Every argument is pinned to the `v2_ab_extract` foldapp/foldapp2 driver —
/// `ZensimProfile::codec_target()`, default toggles (944 adds ONLY
/// `append2_block: true`, so `append2_dst_activity` stays OFF per the
/// SOTA-944 P1.5 adjudication), RAW unpadded slices (the walk does its own
/// sub-64 reflect-pad) — so fleet vectors are byte-identical to driver
/// vectors (W1 local legs run the driver; W2/W3/bf944 run this path).
/// append2 is additive-only: the 944 vector's f0..f923 are bitwise-identical
/// to the 924 vector (gated by `folded944_matches_driver_args`).
#[cfg(feature = "cpu-metrics")]
fn extract_features_folded_streaming(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    regime: crate::metrics::ZensimFeatureRegime,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    use std::cell::RefCell;
    use zensim::feature_v2::{V2NewFeatureToggles, V2Scratch};
    use zensim::{RgbSlice, Zensim, ZensimProfile};
    let (append2, v1_pools) = match regime {
        crate::metrics::ZensimFeatureRegime::Folded720Append => {
            (false, zensim::feature_v2::V1PoolsMode::Off)
        }
        crate::metrics::ZensimFeatureRegime::Folded720Append2 => {
            (true, zensim::feature_v2::V1PoolsMode::Off)
        }
        crate::metrics::ZensimFeatureRegime::Folded720Append2Carriers => {
            (true, zensim::feature_v2::V1PoolsMode::Carriers)
        }
        crate::metrics::ZensimFeatureRegime::Folded720Append2Pools => {
            (true, zensim::feature_v2::V1PoolsMode::Full)
        }
        other => return Err(format!("zensim: not a folded streaming regime: {other:?}").into()),
    };
    thread_local! {
        // The scratch's wide buffers grow to the widest image this thread has
        // seen and are reused (the producer buffer pool lives here — a fresh
        // scratch re-faults ~1.7k pages/pair).
        static SCRATCH: RefCell<V2Scratch> = RefCell::new(V2Scratch::new());
    }
    let r3: &[[u8; 3]] = bytemuck::cast_slice(&reference.pixels);
    let d3: &[[u8; 3]] = bytemuck::cast_slice(&distorted.pixels);
    let src = RgbSlice::new(r3, reference.width as usize, reference.height as usize);
    let dst = RgbSlice::new(d3, distorted.width as usize, distorted.height as usize);
    let feats = SCRATCH
        .with(|s| {
            Zensim::new(ZensimProfile::codec_target())
                .with_parallel(false)
                .compute_folded720_append_features_streaming(
                    &src,
                    &dst,
                    V2NewFeatureToggles {
                        append2_block: append2,
                        v1_pools,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut s.borrow_mut(),
                )
                .map(|r| r.features().to_vec())
        })
        .map_err(|e| format!("zensim: folded720append streaming: {e:?}"))?;
    let want = regime.total_features();
    if feats.len() != want {
        return Err(format!(
            "zensim: {regime:?} expected {want} features, got {}",
            feats.len()
        )
        .into());
    }
    Ok(feats)
}

/// Per-source (per-chunk) reference context: the reference reflect-padded to ≥64px
/// ONCE, plus its v1 XYB pyramid precomputed ONCE. A ScoreFile job scores many
/// variants of the SAME source; building this once and reusing it across the
/// variants avoids rebuilding the v1 ref pyramid per variant (the ref is decoded
/// once in `run_score_file`, but without this the ref FEATURE work was redone every
/// variant). Bit-identical to the per-call path — see `v1_precomputed_ref_matches_percall`.
///
/// BOTH the v1 XYB pyramid AND the v2-348 reference (+cached blur moments) are precomputed once
/// and reused across the source's variants — v2 via zensim's `compute_v2_features_with_ref`
/// (landed 2026-07). Previously the v2 block rebuilt the ref pyramid per variant (~40% of the
/// per-cell cost on a many-variant ScoreFile job); reusing it is a pure cost reduction — features
/// are bit-identical (zensim guarantees it — see `v2_precomputed_ref_matches_percall`).
#[cfg(feature = "cpu-metrics")]
pub struct ZensimRefCtx {
    padded: Vec<u8>, // reference reflect-padded to (w,h); w,h ≥ 64
    w: usize,
    h: usize,
    v1_pre: zensim::PrecomputedReference, // v1 XYB pyramid, 4 scales
    v2_pre: zensim::feature_v2::V2PreparedReference, // v2-348 ref (+moments), reused per variant
}

/// Build a [`ZensimRefCtx`] from a decoded reference — pad once, precompute the v1
/// pyramid once. Reuse across all variants of this source via
/// [`extract_features_regime_with_ctx`].
#[cfg(feature = "cpu-metrics")]
pub fn precompute_ref_ctx(
    reference: &Rgb8Image,
) -> Result<ZensimRefCtx, Box<dyn std::error::Error>> {
    let padded = reflect_pad_to_min(reference, 64).into_owned();
    let w = reference.width.max(64) as usize;
    let h = reference.height.max(64) as usize;
    let r3: &[[u8; 3]] = bytemuck::cast_slice(&padded);
    // 4 scales = zensim's NUM_SCALES / the ZensimConfig default; covers every regime.
    let v1_pre = zensim::precompute_reference_with_scales(r3, w, h, 4)
        .map_err(|e| format!("zensim: precompute_reference_with_scales: {e:?}"))?;
    // v2-348 reference prepared ONCE (+cached blur moments so each variant skips the ref V-blur and
    // activity chain). Reused via compute_v2_features_with_ref — bit-identical to the per-variant
    // rebuild, ~40% cheaper on a many-variant ScoreFile job.
    let v2_pre = zensim::Zensim::new(zensim::ZensimProfile::PreviewV0_2)
        .prepare_v2_reference_with_moments(&zensim::RgbSlice::new(r3, w, h))
        .map_err(|e| format!("zensim: prepare_v2_reference_with_moments: {e:?}"))?;
    Ok(ZensimRefCtx {
        padded,
        w,
        h,
        v1_pre,
        v2_pre,
    })
}

/// Extract the regime-appropriate feature vector for `distorted` against a
/// precomputed [`ZensimRefCtx`] — the v1 block reuses the ctx's pyramid
/// (`compute_zensim_with_ref_and_config`); the v2 block reprocesses the ref (no
/// precomputed-ref API). Bit-identical to [`extract_features_regime`].
#[cfg(feature = "cpu-metrics")]
pub fn extract_features_regime_with_ctx(
    ctx: &ZensimRefCtx,
    distorted: &Rgb8Image,
    regime: crate::metrics::ZensimFeatureRegime,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    use crate::metrics::ZensimFeatureRegime as R;
    use zensim::{RgbSlice, ZensimConfig, compute_zensim_with_ref_and_config};

    // No ctx path for the folded regimes: C5 deleted the cached-reference
    // machinery, and ctx.padded diverges from the streaming walk's own sub-64
    // pad. Callers use extract_features_regime (per-pair, thread-local scratch).
    if regime == R::Folded720Append
        || regime == R::Folded720Append2
        || regime == R::Folded720Append2Carriers
        || regime == R::Folded720Append2Pools
    {
        return Err(
            "zensim: folded streaming regimes have no ref-ctx path — call extract_features_regime"
                .into(),
        );
    }
    if distorted.width.max(64) as usize != ctx.w || distorted.height.max(64) as usize != ctx.h {
        return Err(format!(
            "zensim: distorted ({}×{}) does not match ref ctx ({}×{})",
            distorted.width, distorted.height, ctx.w, ctx.h
        )
        .into());
    }
    let dp = reflect_pad_to_min(distorted, 64);
    let dp: &[u8] = &dp;
    let dst3: &[[u8; 3]] = bytemuck::cast_slice(dp);

    // v1 block: reuse the precomputed ref pyramid (bit-identical to per-call).
    let (extended, iw) = match regime {
        R::Basic => (false, false),
        R::Extended => (true, false),
        R::WithIw | R::V2Ab => (true, true),
        // Rejected by the early return above (no ctx path for the folded regimes).
        R::Folded720Append
        | R::Folded720Append2
        | R::Folded720Append2Carriers
        | R::Folded720Append2Pools => {
            unreachable!("folded streaming rejected above")
        }
    };
    let mut cfg = ZensimConfig::default();
    cfg.extended_features = extended;
    cfg.compute_iw_features = iw;
    let v1 = compute_zensim_with_ref_and_config(&ctx.v1_pre, dst3, ctx.w, ctx.h, cfg)
        .map_err(|e| format!("zensim: compute_zensim_with_ref_and_config: {e:?}"))?;
    let mut features = v1.into_features();

    // v2 block: reuse the precomputed v2 ref (bit-identical to the per-variant rebuild).
    if regime == R::V2Ab {
        let v2 = zensim::Zensim::new(zensim::ZensimProfile::PreviewV0_2)
            .compute_v2_features_with_ref(&ctx.v2_pre, &RgbSlice::new(dst3, ctx.w, ctx.h))
            .map_err(|e| format!("zensim: v2-348 compute_v2_features_with_ref: {e:?}"))?;
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
    Ok(features)
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

    /// The fleet 924 path must be byte-identical to a DIRECT zensim streaming
    /// call with the `v2_ab_extract` foldapp driver's exact arguments
    /// (codec_target profile, default toggles, RAW unpadded slices). Pins the
    /// wiring — a profile/toggle/padding drift here silently forks the regime.
    /// Includes a sub-64 case (48×40) so any reintroduction of external padding
    /// fails loudly, and an odd-dims case (127×93, a gates-doc fixture shape).
    #[test]
    fn folded924_matches_driver_args() {
        use zensim::feature_v2::{V2NewFeatureToggles, V2Scratch};
        use zensim::{RgbSlice, Zensim, ZensimProfile};
        for (w, h) in [(96u32, 80u32), (127, 93), (48, 40)] {
            let reference = synth(11, w, h);
            let distorted = synth(12, w, h);
            let ours = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append,
            )
            .unwrap();
            let r3: &[[u8; 3]] = bytemuck::cast_slice(&reference.pixels);
            let d3: &[[u8; 3]] = bytemuck::cast_slice(&distorted.pixels);
            let mut scratch = V2Scratch::new();
            let direct = Zensim::new(ZensimProfile::codec_target())
                .with_parallel(false)
                .compute_folded720_append_features_streaming(
                    &RgbSlice::new(r3, w as usize, h as usize),
                    &RgbSlice::new(d3, w as usize, h as usize),
                    V2NewFeatureToggles::default(),
                    &mut scratch,
                )
                .unwrap();
            assert_eq!(ours.len(), 924, "{w}×{h}");
            assert_eq!(direct.features().len(), 924, "{w}×{h}");
            for (i, (a, b)) in ours.iter().zip(direct.features()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "slot {i} differs ({w}×{h})");
            }
            // Thread-local scratch reuse must not perturb results.
            let again = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append,
            )
            .unwrap();
            assert_eq!(
                ours.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "scratch-reuse divergence ({w}×{h})"
            );
        }
    }

    /// The fleet 944 path (`Folded720Append2`, jobexec metric
    /// "zensim-foldapp2") must be byte-identical to a DIRECT zensim streaming
    /// call with the `v2_ab_extract` foldapp2 driver's exact arguments
    /// (codec_target profile, `append2_block: true` + otherwise-default
    /// toggles — `append2_dst_activity` OFF per the SOTA-944 P1.5
    /// adjudication — RAW unpadded slices). Also pins the G-BF1 premise at
    /// unit level: append2 is additive-only, so slots f0..f923 of the 944
    /// vector must be BITWISE-IDENTICAL to the 924 extraction of the same
    /// pair, and the append2 tail must be finite and bounded [0,1] (the SDR
    /// HL bins' exact 0 lies inside that range). Covers sub-64 + odd dims
    /// like the 924 test.
    /// The carriers regime = the same streaming call with
    /// `V1PoolsMode::Carriers`: bit-identical to a direct call, 944 wide,
    /// exactly the ten carrier slots differ from the plain 944 regime.
    /// The all-944-LIVE regime (`Folded720Append2Pools`, jobexec metric
    /// "zensim-foldapp2pools"): ONE streaming pass emitting f0..155 basic ++
    /// f156..371 v1 pools LIVE (all 216) ++ v2-348 ++ append2-20. Three
    /// things are pinned here, and the third is the point of the regime:
    ///   1. bit-identical to a DIRECT zensim call with `V1PoolsMode::Full`;
    ///   2. every NON-pool slot bit-identical to the plain 944 regime (the
    ///      pool block is purely additive — the fold arithmetic is untouched),
    ///      and the ten carrier slots agree with the carriers regime;
    ///   3. **f156..372 bit-identical to the v1 372-wide `WithIw` regime** at
    ///      exact widths (`simd_padded_width(w) == w`), i.e. the live block
    ///      really is v1's pool block and not a look-alike. Widths 96×64 and
    ///      208×144 are exact; zensim's own `folded720_v1_pools_match_v1_path`
    ///      makes the same assertion one layer down.
    #[test]
    fn folded944pools_matches_driver_args_and_v1_block() {
        use zensim::feature_v2::{V1PoolsMode, V2NewFeatureToggles, V2Scratch};
        use zensim::{RgbSlice, Zensim, ZensimProfile};
        // Both widths are SIMD-exact (`(w+15)&!15 == w`, < 512), so the
        // folded pool block is bit-comparable to the buffered v1 path.
        for (w, h) in [(96u32, 64u32), (208, 144)] {
            let reference = synth(31, w, h);
            let distorted = synth(32, w, h);
            let ours = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append2Pools,
            )
            .unwrap();
            let plain = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append2,
            )
            .unwrap();
            let carriers = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append2Carriers,
            )
            .unwrap();
            let v1 = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::WithIw,
            )
            .unwrap();
            let r3: &[[u8; 3]] = bytemuck::cast_slice(&reference.pixels);
            let d3: &[[u8; 3]] = bytemuck::cast_slice(&distorted.pixels);
            let mut scratch = V2Scratch::new();
            let direct = Zensim::new(ZensimProfile::codec_target())
                .with_parallel(false)
                .compute_folded720_append_features_streaming(
                    &RgbSlice::new(r3, w as usize, h as usize),
                    &RgbSlice::new(d3, w as usize, h as usize),
                    V2NewFeatureToggles {
                        append2_block: true,
                        v1_pools: V1PoolsMode::Full,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut scratch,
                )
                .unwrap();
            assert_eq!(ours.len(), 944, "{w}×{h}");
            assert_eq!(v1.len(), 372, "{w}×{h}");
            // 1. driver-args parity.
            for (i, (a, b)) in ours.iter().zip(direct.features()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "slot {i} differs ({w}×{h})");
            }
            // 2. additive-only vs the plain 944 regime + carrier agreement.
            let mut live = 0;
            for i in 0..944 {
                if (156..372).contains(&i) {
                    assert_eq!(plain[i], 0.0, "plain 944 slot {i} must be the structural 0");
                    if ours[i] != 0.0 {
                        live += 1;
                    }
                    if V1PoolsMode::CARRIER_SLOTS.contains(&i) {
                        assert_eq!(
                            ours[i].to_bits(),
                            carriers[i].to_bits(),
                            "carrier slot {i} disagrees with the carriers regime ({w}×{h})"
                        );
                    }
                } else {
                    assert_eq!(
                        ours[i].to_bits(),
                        plain[i].to_bits(),
                        "non-pool slot {i} differs from the plain 944 regime ({w}×{h})"
                    );
                }
            }
            assert!(live >= 200, "{w}×{h}: only {live}/216 pool slots live");
            // 3. THE POINT: the live block IS v1's, bit-for-bit.
            for i in 156..372 {
                assert_eq!(
                    ours[i].to_bits(),
                    v1[i].to_bits(),
                    "pool slot {i} ({:e}) != v1 372-regime ({:e}) at {w}×{h}",
                    ours[i],
                    v1[i]
                );
            }
        }
    }

    #[test]
    fn folded944carriers_matches_driver_args() {
        use zensim::feature_v2::{V1PoolsMode, V2NewFeatureToggles, V2Scratch};
        use zensim::{RgbSlice, Zensim, ZensimProfile};
        for (w, h) in [(96u32, 80u32), (127, 93)] {
            let reference = synth(21, w, h);
            let distorted = synth(22, w, h);
            let ours = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append2Carriers,
            )
            .unwrap();
            let plain = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append2,
            )
            .unwrap();
            let r3: &[[u8; 3]] = bytemuck::cast_slice(&reference.pixels);
            let d3: &[[u8; 3]] = bytemuck::cast_slice(&distorted.pixels);
            let mut scratch = V2Scratch::new();
            let direct = Zensim::new(ZensimProfile::codec_target())
                .with_parallel(false)
                .compute_folded720_append_features_streaming(
                    &RgbSlice::new(r3, w as usize, h as usize),
                    &RgbSlice::new(d3, w as usize, h as usize),
                    V2NewFeatureToggles {
                        append2_block: true,
                        v1_pools: V1PoolsMode::Carriers,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut scratch,
                )
                .unwrap();
            assert_eq!(ours.len(), 944, "{w}×{h}");
            for (i, (a, b)) in ours.iter().zip(direct.features()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "slot {i} differs ({w}×{h})");
            }
            let mut live = 0;
            for (i, (a, b)) in ours.iter().zip(plain.iter()).enumerate() {
                if V1PoolsMode::CARRIER_SLOTS.contains(&i) {
                    assert_eq!(*b, 0.0, "plain 944 slot {i} must be the structural 0");
                    if *a != 0.0 {
                        live += 1;
                    }
                } else {
                    assert_eq!(a.to_bits(), b.to_bits(), "non-carrier slot {i} differs ({w}×{h})");
                }
            }
            assert!(live >= 8, "{w}×{h}: only {live} carrier slots live on a distorted pair");
        }
    }

    #[test]
    fn folded944_matches_driver_args() {
        use zensim::feature_v2::{V2NewFeatureToggles, V2Scratch};
        use zensim::{RgbSlice, Zensim, ZensimProfile};
        for (w, h) in [(96u32, 80u32), (127, 93), (48, 40)] {
            let reference = synth(21, w, h);
            let distorted = synth(22, w, h);
            let ours = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append2,
            )
            .unwrap();
            let r3: &[[u8; 3]] = bytemuck::cast_slice(&reference.pixels);
            let d3: &[[u8; 3]] = bytemuck::cast_slice(&distorted.pixels);
            let mut scratch = V2Scratch::new();
            let direct = Zensim::new(ZensimProfile::codec_target())
                .with_parallel(false)
                .compute_folded720_append_features_streaming(
                    &RgbSlice::new(r3, w as usize, h as usize),
                    &RgbSlice::new(d3, w as usize, h as usize),
                    V2NewFeatureToggles {
                        append2_block: true,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut scratch,
                )
                .unwrap();
            assert_eq!(ours.len(), 944, "{w}×{h}");
            assert_eq!(direct.features().len(), 944, "{w}×{h}");
            for (i, (a, b)) in ours.iter().zip(direct.features()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "slot {i} differs ({w}×{h})");
            }
            // G-BF1 premise: f0..f923 bitwise-identical to the 924 regime.
            let base924 = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append,
            )
            .unwrap();
            assert_eq!(base924.len(), 924, "{w}×{h}");
            for (i, (a, b)) in ours[..924].iter().zip(base924.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "f{i} of 944 diverges from the 924 regime ({w}×{h}) — append2 must be additive-only"
                );
            }
            // append2 tail: finite, bounded [0,1].
            for (i, v) in ours[924..].iter().enumerate() {
                assert!(
                    v.is_finite() && (0.0..=1.0).contains(v),
                    "append2 slot f{} out of [0,1] or non-finite ({w}×{h}): {v}",
                    924 + i
                );
            }
            // Thread-local scratch reuse must not perturb results.
            let again = extract_features_regime(
                &reference,
                &distorted,
                crate::metrics::ZensimFeatureRegime::Folded720Append2,
            )
            .unwrap();
            assert_eq!(
                ours.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "scratch-reuse divergence ({w}×{h})"
            );
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

    /// v2ab must emit exactly 720 = v1-372 ++ v2-348, all finite, with both blocks
    /// populated. FEATURES ONLY — no score is computed or returned.
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn v2ab_emits_720_finite() {
        let reference = synth(1, 96, 72);
        let distorted = synth(101, 96, 72);
        let feats = extract_features_v2ab(&reference, &distorted).unwrap();
        assert_eq!(feats.len(), 720, "v2ab must emit v1-372 ++ v2-348 = 720");
        assert!(
            feats.iter().all(|v| v.is_finite()),
            "all 720 features finite"
        );
        // both blocks populated (not all-zero)
        assert!(feats[..372].iter().any(|&v| v != 0.0), "v1 block populated");
        assert!(feats[372..].iter().any(|&v| v != 0.0), "v2 block populated");
    }

    /// A sub-64px pair must reflect-pad and still emit exactly 720 (both blocks
    /// see the same ≥64 image — no sub-64 disagreement, no error).
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn v2ab_tiny_image_pads_to_720() {
        let reference = synth(2, 20, 12);
        let distorted = synth(202, 20, 12);
        let feats = extract_features_v2ab(&reference, &distorted).unwrap();
        assert_eq!(feats.len(), 720);
        assert!(feats.iter().all(|v| v.is_finite()));
    }

    /// GATE for the v2 ref-reuse optimization: the precomputed-ctx path (v1 via ctx.v1_pre, v2 via
    /// ctx.v2_pre + `compute_v2_features_with_ref`) MUST be bit-identical to the per-call path
    /// (`extract_features_regime`, which rebuilds both refs per call). This covers the whole 720 —
    /// if the v2 ref reuse shifts ANY feature bit, the backfill data diverges from prior rows and
    /// this fails. The picker's features are pixel-sacred.
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn v2_precomputed_ref_matches_percall() {
        use crate::metrics::ZensimFeatureRegime as R;
        let (w, h) = (96u32, 72u32);
        let reference = synth(1, w, h);
        let ctx = precompute_ref_ctx(&reference).unwrap();
        for d in 0..4u32 {
            let distorted = synth(100 + d, w, h);
            let percall = extract_features_regime(&reference, &distorted, R::V2Ab).unwrap();
            let viactx = extract_features_regime_with_ctx(&ctx, &distorted, R::V2Ab).unwrap();
            assert_eq!(percall.len(), 720);
            assert_eq!(viactx.len(), 720);
            for i in 0..720 {
                assert_eq!(
                    percall[i].to_bits(),
                    viactx[i].to_bits(),
                    "v2ab feature {i} must be bit-identical (per-call vs precomputed-ctx), d={d}: \
                     percall={} viactx={}",
                    percall[i],
                    viactx[i]
                );
            }
        }
    }

    /// Throughput cost of the v2 append regime vs v1 with-iw, on a realistic
    /// image. v2 is SIMD-fused but rebuilds its own pyramid, so this quantifies the
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
        let _ = extract_features_regime(&reference, &distorted, R::WithIw).unwrap();
        let _ = extract_features_regime(&reference, &distorted, R::V2Ab).unwrap();
        let n = 5;
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = extract_features_regime(&reference, &distorted, R::WithIw).unwrap();
        }
        let iw = t0.elapsed().as_secs_f64() / n as f64;
        let t1 = Instant::now();
        for _ in 0..n {
            let _ = extract_features_regime(&reference, &distorted, R::V2Ab).unwrap();
        }
        let v2 = t1.elapsed().as_secs_f64() / n as f64;
        eprintln!(
            "ZENSIM {w}x{h}: with-iw(372) {:.1} ms/pair | v2-ab(720) {:.1} ms/pair | v2/v1 = {:.2}x",
            iw * 1e3,
            v2 * 1e3,
            v2 / iw
        );
    }

    /// GATE for the ref-reuse optimization: does the PRECOMPUTED-ref v1 path
    /// (`precompute_reference_with_scales` + `compute_zensim_with_ref_and_config`)
    /// produce BIT-IDENTICAL 372 features to the per-call `compute_zensim_with_config`?
    /// If yes, the ScoreFile chunk can precompute the ref ONCE and reuse it across
    /// all its variants (the ref pyramid is built once, not per variant). If this
    /// ever fails, the precompute path is NOT usable (precision rule).
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn v1_precomputed_ref_matches_percall() {
        use zensim::{
            ZensimConfig, compute_zensim_with_config, compute_zensim_with_ref_and_config,
            precompute_reference_with_scales,
        };
        let (w, h) = (96usize, 72usize);
        let n_scales = 4; // zensim NUM_SCALES (pub(crate)); the ZensimConfig default
        let reference = synth(1, w as u32, h as u32);
        let r3: Vec<[u8; 3]> = reference
            .pixels
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let mut cfg = ZensimConfig::default();
        cfg.extended_features = true;
        cfg.compute_iw_features = true;
        let pre = precompute_reference_with_scales(&r3, w, h, n_scales).unwrap();
        for d in 0..4u32 {
            let dist = synth(200 + d, w as u32, h as u32);
            let d3: Vec<[u8; 3]> = dist
                .pixels
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            let per_call = compute_zensim_with_config(&r3, &d3, w, h, cfg)
                .unwrap()
                .into_features();
            let via_pre = compute_zensim_with_ref_and_config(&pre, &d3, w, h, cfg)
                .unwrap()
                .into_features();
            assert_eq!(per_call.len(), 372);
            assert_eq!(via_pre.len(), 372);
            for (i, (a, b)) in per_call.iter().zip(via_pre.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "v1 feat[{i}] differs precomputed-ref vs per-call (d={d}): {a} vs {b}"
                );
            }
        }
    }

    /// The precomputed-ref-CTX path must produce BIT-IDENTICAL 720 features to the
    /// per-call `extract_features_regime` — the ScoreFile chunk optimization must
    /// not shift a single feature. Covers ≥64px (no pad) and sub-64px (padded).
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn v2ab_ctx_matches_percall() {
        use crate::metrics::ZensimFeatureRegime as R;
        for (w, h) in [(96u32, 72u32), (40, 24)] {
            let reference = synth(7, w, h);
            let ctx = precompute_ref_ctx(&reference).unwrap();
            for d in 0..3u32 {
                let dist = synth(300 + d, w, h);
                let per_call = extract_features_regime(&reference, &dist, R::V2Ab).unwrap();
                let via_ctx = extract_features_regime_with_ctx(&ctx, &dist, R::V2Ab).unwrap();
                assert_eq!(per_call.len(), 720);
                assert_eq!(via_ctx.len(), 720);
                for (i, (a, b)) in per_call.iter().zip(via_ctx.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "ctx feat[{i}] differs @ {w}x{h} d={d}: {a} vs {b}"
                    );
                }
            }
        }
    }

    /// Measures the ScoreFile ref-reuse win: 12 variants of ONE source (the chunk
    /// shape), per-call vs precomputed-ref ctx. Run: `cargo test -p zenmetrics-cli
    /// --lib --release v2ab_ctx_speedup -- --ignored --nocapture`.
    #[cfg(feature = "cpu-metrics")]
    #[test]
    #[ignore = "timing measurement, not a correctness gate"]
    fn v2ab_ctx_speedup_12var() {
        use crate::metrics::ZensimFeatureRegime as R;
        use std::time::Instant;
        let (w, h) = (512u32, 512u32);
        let reference = synth(1, w, h);
        let dists: Vec<_> = (0..12).map(|d| synth(100 + d, w, h)).collect();
        // per-call: reprocess ref every variant
        let _ = extract_features_regime(&reference, &dists[0], R::V2Ab).unwrap();
        let t0 = Instant::now();
        for d in &dists {
            let _ = extract_features_regime(&reference, d, R::V2Ab).unwrap();
        }
        let per_call = t0.elapsed().as_secs_f64();
        // ctx: precompute ref once, reuse across the 12
        let t1 = Instant::now();
        let ctx = precompute_ref_ctx(&reference).unwrap();
        for d in &dists {
            let _ = extract_features_regime_with_ctx(&ctx, d, R::V2Ab).unwrap();
        }
        let via_ctx = t1.elapsed().as_secs_f64();
        eprintln!(
            "ZENSIM chunk {w}x{h} ×12: per-call {:.0} ms | ctx(ref-reuse) {:.0} ms | speedup {:.2}× ({:.0}% off)",
            per_call * 1e3,
            via_ctx * 1e3,
            per_call / via_ctx,
            (1.0 - via_ctx / per_call) * 100.0
        );
    }

    /// Same inputs → bit-identical features across runs (guards the v2 path against
    /// nondeterminism, e.g. from internal parallelism).
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn v2ab_deterministic() {
        let reference = synth(3, 80, 64);
        let distorted = synth(303, 80, 64);
        let a = extract_features_v2ab(&reference, &distorted).unwrap();
        let b = extract_features_v2ab(&reference, &distorted).unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "v2ab feature[{i}] must be deterministic"
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
    let z = Zensim::new(selected_profile()?);
    let stride = width * 3;
    let result = z
        .compute_pu_linear_extended_features(ref_nits, dist_nits, width, height, stride, stride)
        .map_err(|e| format!("zensim pu-linear: {e:?}"))?;
    let score = result.score();
    let features = result.into_features();
    Ok((score, features))
}
