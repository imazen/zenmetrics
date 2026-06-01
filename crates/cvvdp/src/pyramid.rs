//! Gaussian reduce/expand + Weber-contrast pyramid per channel.
//!
//! Bit-exact port of `crate::kernels::pyramid::{gausspyr_reduce_scalar,
//! gausspyr_expand_scalar, weber_contrast_pyr_dec_scalar}` — preserves
//! the pycvvdp v0.5.4 boundary bug compatibility (lpyr_dec.py:204-209
//! "x.shape[-2]" parity quirk) so goldens match.
//!
//! Differences vs upstream:
//!
//! - Uses caller-owned scratch (`PyramidScratch`) to avoid one
//!   `Vec::new()` allocation per (image, band, channel) — gives a
//!   ~20-30% wall-time win at 1MP per call.
//! - Allocates output `Band` data lazily into a single contiguous
//!   `Vec<f32>` then re-slices per band (less heap fragmentation).
//! - The `band_frequencies` helper is identical and re-exported via
//!   `crate::kernels::pyramid::band_frequencies` (avoiding a
//!   redefinition).

use alloc::vec;
use alloc::vec::Vec;

pub(crate) use crate::kernels::pyramid::{GAUSS5, band_frequencies};

/// One Laplacian / Weber pyramid band.
pub(crate) struct Band {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Weber-contrast pyramid output (per channel).
pub struct WeberPyramid {
    /// Per-band contrast values (finest = bands[0]).
    pub(crate) bands: Vec<Band>,
    /// `log10(L_bkg)` per band (per-pixel for non-baseband,
    /// replicated scalar for baseband).
    pub(crate) log_l_bkg: Vec<Vec<f32>>,
}

impl WeberPyramid {
    /// Create an empty WeberPyramid with reserved capacity hints. Used
    /// by the Scratch struct to preallocate slots whose band buffers
    /// can be reused across scoring calls.
    pub(crate) fn empty() -> Self {
        Self {
            bands: Vec::new(),
            log_l_bkg: Vec::new(),
        }
    }

    /// Pre-allocate all per-level band + log_l_bkg buffers for an
    /// image of `sw × sh` decomposed into `n_levels` bands. Each
    /// level's Vec<f32> is sized correctly so the first
    /// `weber_contrast_pyr_into` call doesn't take an alloc hit.
    /// Phase 9.YA Part 2: amortizes the 213 MB × 9 first-call
    /// allocations measured at 40 MP into a single `Scratch::new`
    /// upfront cost.
    pub(crate) fn with_capacity(sw: usize, sh: usize, n_levels: usize) -> Self {
        let mut bands = Vec::with_capacity(n_levels);
        let mut log_l_bkg = Vec::with_capacity(n_levels);
        let (mut w, mut h) = (sw, sh);
        for _ in 0..n_levels {
            bands.push(Band {
                w,
                h,
                data: vec![0.0_f32; w * h],
            });
            log_l_bkg.push(vec![0.0_f32; w * h]);
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        Self { bands, log_l_bkg }
    }

    /// Strip-shape pre-allocation for the K_SPLIT walker.
    ///
    /// At shallow levels (`k < k_split`) the per-level `bands[k].data`
    /// and `log_l_bkg[k]` are sized at `bw × R_k` where
    /// `R_k = mode_b_strip_h_at_level(k, h_body, k_split)` — the
    /// strip's back-projected buffer height. At deep levels (`k >=
    /// k_split`) the full `bw × bh` allocation is used (these levels
    /// are small in absolute terms; level 8 at 4096² is 16×16 = ~1 KB).
    ///
    /// Saves ~80% of the 1 GB WeberPyramid footprint at 16 MP. Mirrors
    /// the GPU's strip-shape allocator policy.
    ///
    /// The per-level `Band::h` value is set to `bh` (the FULL band
    /// height) — the band's logical shape is preserved for downstream
    /// callers that read `bh`. Only the underlying `data` Vec capacity
    /// is strip-shaped at shallow levels. Strip-major dispatchers must
    /// ensure they write at most `bw × R_k` entries before consuming
    /// the band's data; bands are NOT cumulative-full across strips.
    ///
    /// **Phase 9.Z.F chunk 6 of the CPU K_SPLIT walker port.**
    #[allow(dead_code)]
    pub(crate) fn with_capacity_strip(sw: usize, sh: usize, n_levels: usize, h_body: u32) -> Self {
        let k_split = crate::strip::mode_b_k_split(h_body, n_levels as u32);
        let mut bands = Vec::with_capacity(n_levels);
        let mut log_l_bkg = Vec::with_capacity(n_levels);
        let (mut w, mut h) = (sw, sh);
        for k in 0..n_levels {
            let alloc_h = if (k as u32) < k_split {
                let r_back = crate::strip::mode_b_strip_h_at_level(k as u32, h_body, k_split);
                if r_back == 0 {
                    h
                } else {
                    (r_back as usize).min(h)
                }
            } else {
                h
            };
            bands.push(Band {
                w,
                h,
                data: vec![0.0_f32; w * alloc_h],
            });
            log_l_bkg.push(vec![0.0_f32; w * alloc_h]);
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        Self { bands, log_l_bkg }
    }
}

/// Scratch buffers used by reduce/expand passes. Owned by the
/// `Cvvdp` scorer so they persist across calls (no realloc).
///
/// `expanded` and `gauss_tmp` are reserved for the next SIMD pass
/// where we hoist the per-band expand output to the scratch slot
/// instead of allocating per-call.
#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct PyramidScratch {
    pub vscratch: Vec<f32>,
    pub z_v: Vec<f32>,
    pub z_h: Vec<f32>,
    pub expanded: Vec<f32>,
    pub gauss_tmp: Vec<f32>,
}

/// 2D separable 5-tap Gaussian + 2× decimation, ceil-halving.
/// Bit-identical to `crate::kernels::pyramid::gausspyr_reduce_scalar`
/// for FMA-grouping-equivalent values, and within `< 1e-5 abs` for the
/// SIMD inner-loop chunks (the 5-tap dot accumulator may schedule FMAs
/// differently than the scalar `+` chain — the resulting numeric delta
/// is far below the 1e-4 JOD parity floor).
pub(crate) fn gausspyr_reduce(
    src: &[f32],
    sw: usize,
    sh: usize,
    scratch: &mut PyramidScratch,
    dst: &mut Vec<f32>,
) -> (usize, usize) {
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    dst.clear();
    dst.resize(dw * dh, 0.0);
    let k = GAUSS5;

    // Vertical pass: zero-pad rows above/below, conv stride 2.
    scratch.vscratch.clear();
    scratch.vscratch.resize(sw * dh, 0.0);
    let vscratch = &mut scratch.vscratch;

    // SIMD inner pass — covers all rows uniformly. Note: the SIMD pass
    // overwrites every entry of `vscratch` so the zero-fill above is
    // strictly redundant (kept for parity with the prior scalar code
    // path and for easy debugging — `Vec::resize` on a warm Vec is
    // ~~free since capacity matches; on cold first-call the alloc cost
    // dominates the zero-fill cost regardless).
    crate::simd_pyramid::reduce_vertical_pass(src, sw, sh, dh, vscratch);

    // First-row patch: pycvvdp adds reflected-row contribution. Scalar
    // because it's a one-row scan and preserves the historical FMA
    // ordering (`+= a*k[1] + b*k[0]`) for bit-identical golden parity
    // with the patches alone.
    if dh > 0 && sh >= 2 {
        for x in 0..sw {
            vscratch[x] += src[x] * k[1] + src[sw + x] * k[0];
        }
    }
    if dh > 0 {
        let last_dy = dh - 1;
        if sh % 2 == 1 && sh >= 2 {
            for x in 0..sw {
                vscratch[last_dy * sw + x] +=
                    src[(sh - 1) * sw + x] * k[3] + src[(sh - 2) * sw + x] * k[4];
            }
        } else if sh.is_multiple_of(2) {
            for x in 0..sw {
                vscratch[last_dy * sw + x] += src[(sh - 1) * sw + x] * k[4];
            }
        }
    }

    // Horizontal pass — SIMD inner pass over rows, then scalar boundary
    // patches replicating the upstream parity-on-rows bug.
    crate::simd_pyramid::reduce_horizontal_pass(vscratch, sw, dw, dh, dst);

    if dw > 0 && sw >= 2 {
        for dy in 0..dh {
            dst[dy * dw] += vscratch[dy * sw] * k[1] + vscratch[dy * sw + 1] * k[0];
        }
    }
    if dw > 0 {
        let last_dx = dw - 1;
        // Replicate pycvvdp's parity-on-rows bug — see upstream notes.
        if sh % 2 == 1 && sw >= 2 {
            for dy in 0..dh {
                dst[dy * dw + last_dx] +=
                    vscratch[dy * sw + sw - 1] * k[3] + vscratch[dy * sw + sw - 2] * k[4];
            }
        } else if sh.is_multiple_of(2) {
            for dy in 0..dh {
                dst[dy * dw + last_dx] += vscratch[dy * sw + sw - 1] * k[4];
            }
        }
    }
    (dw, dh)
}

/// 2× upscale: zero-insert + 5-tap Gaussian (×4 reconstruction gain
/// split 2× per separable pass). Bit-identical to
/// `crate::kernels::pyramid::gausspyr_expand_scalar` for
/// FMA-grouping-equivalent values, and within `< 1e-5 abs` for the
/// SIMD inner-loop chunks (see [`gausspyr_reduce`] for the FMA grouping
/// note).
pub(crate) fn gausspyr_expand(
    src: &[f32],
    sw: usize,
    sh: usize,
    out_w: usize,
    out_h: usize,
    scratch: &mut PyramidScratch,
    dst: &mut Vec<f32>,
) {
    debug_assert!(out_w >= 2 * sw - 1 && out_w <= 2 * sw);
    debug_assert!(out_h >= 2 * sh - 1 && out_h <= 2 * sh);

    // Vertical pass: SIMD inner sweep, builds per-column zero-inserted
    // buffer in-flight (no separate `z_v` scratch from the caller).
    scratch.vscratch.clear();
    scratch.vscratch.resize(sw * out_h, 0.0);
    crate::simd_pyramid::expand_vertical_pass(src, sw, sh, out_h, &mut scratch.vscratch);

    // Horizontal pass: SIMD inner sweep, re-uses caller's `z_h` scratch
    // (resized inside).
    dst.clear();
    dst.resize(out_w * out_h, 0.0);
    crate::simd_pyramid::expand_horizontal_pass(
        &scratch.vscratch,
        sw,
        out_w,
        out_h,
        dst,
        &mut scratch.z_h,
    );
}

/// Build a single-channel Gaussian pyramid (`n_levels` bands) into
/// `out`. `out` is grown / shrunk to `n` bands; existing band Vec
/// allocations are reused (clear + resize) when present, eliminating
/// per-call heap churn.
pub(crate) fn build_gauss_pyramid_into(
    src: &[f32],
    sw: usize,
    sh: usize,
    n: usize,
    scratch: &mut PyramidScratch,
    out: &mut Vec<Band>,
) {
    // Grow to required band count, reusing existing Vec<f32> capacity.
    while out.len() < n {
        out.push(Band {
            w: 0,
            h: 0,
            data: Vec::new(),
        });
    }
    out.truncate(n);

    // Band 0: copy src into out[0].data (reusing capacity).
    out[0].w = sw;
    out[0].h = sh;
    out[0].data.clear();
    out[0].data.extend_from_slice(src);

    let mut w = sw;
    let mut h = sh;
    // Split the vec so we can borrow [k] and [k+1] separately.
    for k in 0..n - 1 {
        let (lhs, rhs) = out.split_at_mut(k + 1);
        let prev = &lhs[k];
        let next_band = &mut rhs[0];
        let (nw, nh) = gausspyr_reduce(&prev.data, w, h, scratch, &mut next_band.data);
        next_band.w = nw;
        next_band.h = nh;
        w = nw;
        h = nh;
    }
}

/// Per-pyramid recycling cache: holds the two intermediate Gaussian
/// pyramids (`gauss_img` + `gauss_l`) so successive calls reuse band
/// Vec<f32> capacity. Owned by `Scratch`, one slot per channel
/// pyramid (6 total: 3 REF + 3 DIST). All fields are `pub(crate)` so
/// `Scratch::new` can construct empties.
#[derive(Default)]
pub(crate) struct WeberPyramidCache {
    pub gauss_img: Vec<Band>,
    pub gauss_l: Vec<Band>,
    pub scratch: PyramidScratch,
}

impl WeberPyramidCache {
    /// Pre-allocate all per-level Vec<f32> buffers in `gauss_img` and
    /// `gauss_l` for an image of `sw × sh` decomposed into `n_levels`
    /// bands. The inner `PyramidScratch` is left in `Default` state
    /// and lazily resized inside `gausspyr_reduce` / `gausspyr_expand`
    /// at the actual high-water mark — pre-allocating it at the
    /// finest-level worst case pushed peak heap above the natural
    /// runtime peak by ~640 MB × 6 caches during Phase 9.YA Part 2
    /// iteration.
    ///
    /// Phase 9.YA Part 2: removes the first-call growth cost of the
    /// gauss_img / gauss_l band Vecs (the dominant 213 MB × 6 sites
    /// at 40 MP).
    pub(crate) fn with_capacity(sw: usize, sh: usize, n_levels: usize) -> Self {
        let mut gauss_img = Vec::with_capacity(n_levels);
        let mut gauss_l = Vec::with_capacity(n_levels);
        let (mut w, mut h) = (sw, sh);
        for _ in 0..n_levels {
            gauss_img.push(Band {
                w,
                h,
                data: vec![0.0_f32; w * h],
            });
            gauss_l.push(Band {
                w,
                h,
                data: vec![0.0_f32; w * h],
            });
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        Self {
            gauss_img,
            gauss_l,
            scratch: PyramidScratch::default(),
        }
    }

    /// Strip-shape pre-allocation matching
    /// [`WeberPyramid::with_capacity_strip`]'s policy.
    ///
    /// At shallow levels (`k < k_split`) the per-level `gauss_img[k]`
    /// and `gauss_l[k]` are sized at `bw × R_k` where
    /// `R_k = mode_b_strip_h_at_level(k, h_body, k_split)`. Deep levels
    /// use the full `bw × bh` allocation.
    ///
    /// The per-level `Band::h` value is set to `bh` (the FULL band
    /// height) — the band's logical shape is preserved for downstream
    /// callers that read `bh`. Only the underlying `data` Vec capacity
    /// is strip-shaped at shallow levels.
    ///
    /// Saves ~80% of the 510 MB cache footprint at 16 MP. Mirrors the
    /// GPU's strip-shape cache allocator. Caller must use the
    /// strip-major dispatcher (chunk 6's `_run_d_bands_strip_major_shallow`
    /// CPU port) to consume the strip-shaped cache safely — the
    /// upstream `weber_contrast_pyr_into` path writes full-image
    /// shape and will panic on `data.len() < w*h` accesses against a
    /// strip-shape cache.
    ///
    /// **Phase 9.Z.F chunk 6 of the CPU K_SPLIT walker port.**
    #[allow(dead_code)]
    pub(crate) fn with_capacity_strip(sw: usize, sh: usize, n_levels: usize, h_body: u32) -> Self {
        let k_split = crate::strip::mode_b_k_split(h_body, n_levels as u32);
        let mut gauss_img = Vec::with_capacity(n_levels);
        let mut gauss_l = Vec::with_capacity(n_levels);
        let (mut w, mut h) = (sw, sh);
        for k in 0..n_levels {
            let alloc_h = if (k as u32) < k_split {
                let r_back = crate::strip::mode_b_strip_h_at_level(k as u32, h_body, k_split);
                if r_back == 0 {
                    h
                } else {
                    (r_back as usize).min(h)
                }
            } else {
                h
            };
            gauss_img.push(Band {
                w,
                h,
                data: vec![0.0_f32; w * alloc_h],
            });
            gauss_l.push(Band {
                w,
                h,
                data: vec![0.0_f32; w * alloc_h],
            });
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        Self {
            gauss_img,
            gauss_l,
            scratch: PyramidScratch::default(),
        }
    }
}

/// Single-channel Weber-contrast pyramid (`weber_g1`).
///
/// Writes the result into `out`, reusing existing band/log_l_bkg
/// `Vec<f32>` capacity. `cache` holds two intermediate Gaussian
/// pyramids whose buffers persist across calls.
///
/// `image_plane` is the channel under decomposition; `l_bkg_plane`
/// is the achromatic plane used for the per-pixel L_bkg. For the
/// achromatic channel itself they're the same buffer.
pub(crate) fn weber_contrast_pyr_into(
    image_plane: &[f32],
    l_bkg_plane: &[f32],
    sw: usize,
    sh: usize,
    n_levels: usize,
    cache: &mut WeberPyramidCache,
    out: &mut WeberPyramid,
) {
    let n = n_levels;
    debug_assert!(n >= 1);

    build_gauss_pyramid_into(
        image_plane,
        sw,
        sh,
        n,
        &mut cache.scratch,
        &mut cache.gauss_img,
    );
    build_gauss_pyramid_into(
        l_bkg_plane,
        sw,
        sh,
        n,
        &mut cache.scratch,
        &mut cache.gauss_l,
    );

    // Grow / shrink `out` to exactly `n` bands; reuse existing Vec<f32>.
    while out.bands.len() < n {
        out.bands.push(Band {
            w: 0,
            h: 0,
            data: Vec::new(),
        });
        out.log_l_bkg.push(Vec::new());
    }
    out.bands.truncate(n);
    out.log_l_bkg.truncate(n);

    for k in 0..n {
        let is_baseband = k == n - 1;
        let fine = &cache.gauss_img[k];
        let l_fine = &cache.gauss_l[k];
        let n_px = fine.w * fine.h;

        out.bands[k].w = fine.w;
        out.bands[k].h = fine.h;
        out.bands[k].data.clear();
        out.bands[k].data.resize(n_px, 0.0);
        out.log_l_bkg[k].clear();
        out.log_l_bkg[k].resize(n_px, 0.0);

        if is_baseband {
            let sum: f32 = l_fine.data.iter().map(|v| v.max(0.01)).sum();
            let l_bkg_mean = sum / l_fine.data.len() as f32;
            let log_l = l_bkg_mean.log10();
            let band_data = &mut out.bands[k].data;
            for i in 0..n_px {
                band_data[i] = fine.data[i] / l_bkg_mean;
            }
            let log_band = &mut out.log_l_bkg[k];
            for v in log_band.iter_mut() {
                *v = log_l;
            }
        } else {
            // expanded_l + img_expanded into per-band scratch.
            // Reuse `cache.scratch.expanded` for `expanded_l`, plus a
            // local Vec for `img_expanded` (still better than the
            // pre-fix path because gausspyr_expand uses cache.scratch
            // internally for its own intermediates).
            cache.scratch.expanded.clear();
            // expanded_l is the L_bkg expansion; img_expanded is the
            // image-channel expansion. We need both simultaneously,
            // so we use `cache.scratch.expanded` for one and
            // `cache.scratch.gauss_tmp` for the other.
            let coarse_l = &cache.gauss_l[k + 1];
            let img_coarse = &cache.gauss_img[k + 1];
            // Pre-extract coarse data so we can borrow cache.scratch mutably below
            // without aliasing.
            let coarse_l_data: &[f32] = &coarse_l.data;
            let coarse_l_w = coarse_l.w;
            let coarse_l_h = coarse_l.h;
            let img_coarse_data: &[f32] = &img_coarse.data;
            let img_coarse_w = img_coarse.w;
            let img_coarse_h = img_coarse.h;
            // We can't simultaneously call gausspyr_expand with two
            // different `dst` slots on the same `cache.scratch` —
            // gausspyr_expand writes vscratch/z_v/z_h inside scratch.
            // So we call sequentially and stash one result in
            // `cache.scratch.expanded` and the other in
            // `cache.scratch.gauss_tmp`.
            // Trick: temporarily swap out the gauss_tmp + expanded.
            let mut expanded_l = core::mem::take(&mut cache.scratch.expanded);
            gausspyr_expand(
                coarse_l_data,
                coarse_l_w,
                coarse_l_h,
                fine.w,
                fine.h,
                &mut cache.scratch,
                &mut expanded_l,
            );
            let mut img_expanded = core::mem::take(&mut cache.scratch.gauss_tmp);
            gausspyr_expand(
                img_coarse_data,
                img_coarse_w,
                img_coarse_h,
                fine.w,
                fine.h,
                &mut cache.scratch,
                &mut img_expanded,
            );
            let fine_data: &[f32] = &fine.data;
            let band_data = &mut out.bands[k].data;
            let log_band = &mut out.log_l_bkg[k];
            for i in 0..n_px {
                let l_bkg = expanded_l[i].max(0.01);
                let layer = fine_data[i] - img_expanded[i];
                let c = (layer / l_bkg).clamp(-1000.0, 1000.0);
                band_data[i] = c;
                log_band[i] = l_bkg.log10();
            }
            // Return scratch.
            cache.scratch.expanded = expanded_l;
            cache.scratch.gauss_tmp = img_expanded;
        }
    }
}

/// Owning variant kept for tests that don't have a caller-supplied
/// output buffer. Tests don't sit in the hot path so the allocation
/// here is fine.
#[cfg(test)]
pub(crate) fn weber_contrast_pyr(
    image_plane: &[f32],
    l_bkg_plane: &[f32],
    sw: usize,
    sh: usize,
    n_levels: usize,
    scratch: &mut PyramidScratch,
) -> WeberPyramid {
    let mut cache = WeberPyramidCache {
        gauss_img: Vec::new(),
        gauss_l: Vec::new(),
        scratch: core::mem::take(scratch),
    };
    let mut out = WeberPyramid::empty();
    weber_contrast_pyr_into(
        image_plane,
        l_bkg_plane,
        sw,
        sh,
        n_levels,
        &mut cache,
        &mut out,
    );
    *scratch = cache.scratch;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::pyramid::{
        gausspyr_expand_scalar, gausspyr_reduce_scalar, weber_contrast_pyr_dec_scalar,
    };

    #[test]
    fn reduce_matches_upstream_scalar() {
        let mut rng_state = 0xdeadbeefu32;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            (rng_state >> 16) as f32 / 65536.0
        };
        let cases: &[(usize, usize)] = &[(16, 16), (15, 17), (32, 24), (73, 91), (128, 128)];
        for &(sw, sh) in cases {
            let src: Vec<f32> = (0..sw * sh).map(|_| next()).collect();
            let mut want = Vec::new();
            gausspyr_reduce_scalar(&src, sw, sh, &mut want);
            let mut scratch = PyramidScratch::default();
            let mut got = Vec::new();
            gausspyr_reduce(&src, sw, sh, &mut scratch, &mut got);
            assert_eq!(want.len(), got.len(), "{sw}x{sh}");
            for i in 0..want.len() {
                // 1e-5 tolerance (was 1e-6) per SIMD plan Chunk 2 — the
                // SIMD inner sweep accumulates the 5-tap dot product in
                // a different order than the scalar `+` chain, producing
                // ULP-scale FMA-grouping deltas. Still ~3 orders below
                // the 1e-3 JOD tolerance / 1e-4 golden tolerance.
                assert!(
                    (want[i] - got[i]).abs() < 1e-5,
                    "case {sw}x{sh} idx {i}: want={}, got={}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn expand_matches_upstream_scalar() {
        let mut rng_state = 0xfeedf00du32;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            (rng_state >> 16) as f32 / 65536.0
        };
        let cases: &[(usize, usize, usize, usize)] = &[
            (4, 4, 8, 8),
            (4, 4, 7, 7),
            (8, 6, 16, 12),
            (8, 6, 15, 11),
            (16, 12, 32, 24),
        ];
        for &(sw, sh, ow, oh) in cases {
            let src: Vec<f32> = (0..sw * sh).map(|_| next()).collect();
            let mut want = Vec::new();
            gausspyr_expand_scalar(&src, sw, sh, ow, oh, &mut want);
            let mut scratch = PyramidScratch::default();
            let mut got = Vec::new();
            gausspyr_expand(&src, sw, sh, ow, oh, &mut scratch, &mut got);
            assert_eq!(want.len(), got.len());
            for i in 0..want.len() {
                // 1e-5 tolerance (was 1e-6) per SIMD plan Chunk 2 — see
                // `reduce_matches_upstream_scalar` for FMA grouping note.
                assert!(
                    (want[i] - got[i]).abs() < 1e-5,
                    "case {sw}x{sh}/{ow}x{oh} idx {i}: want={}, got={}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn weber_pyr_matches_upstream() {
        let sw = 32;
        let sh = 32;
        let n = 4;
        let src: Vec<f32> = (0..sw * sh).map(|i| 1.0 + (i as f32) * 0.05).collect();
        let want = weber_contrast_pyr_dec_scalar(&src, &src, sw, sh, n);
        let mut scratch = PyramidScratch::default();
        let got = weber_contrast_pyr(&src, &src, sw, sh, n, &mut scratch);
        assert_eq!(got.bands.len(), want.bands.len());
        for k in 0..n {
            assert_eq!(got.bands[k].w, want.bands[k].w);
            assert_eq!(got.bands[k].h, want.bands[k].h);
            for i in 0..got.bands[k].data.len() {
                assert!(
                    (got.bands[k].data[i] - want.bands[k].data[i]).abs() < 1e-5,
                    "level {k} px {i}"
                );
                assert!(
                    (got.log_l_bkg[k][i] - want.log_l_bkg[k][i]).abs() < 1e-5,
                    "level {k} log_l px {i}"
                );
            }
        }
    }

    /// Chunk 6 step 1 regression: `with_capacity_strip` sizes shallow
    /// levels at `bw × R_k` (strip-shape) and deep levels at `bw × bh`
    /// (full). The per-level `Band::h` field stays at the full band
    /// height — only the underlying `data` Vec capacity is reduced.
    #[test]
    fn weber_pyramid_with_capacity_strip_sizes_shallow_levels() {
        let w = 4096_usize;
        let h = 4096_usize;
        let n_levels = 9_usize;
        let h_body = 512_u32;
        let pyr = WeberPyramid::with_capacity_strip(w, h, n_levels, h_body);
        let k_split = crate::strip::mode_b_k_split(h_body, n_levels as u32) as usize;
        assert_eq!(k_split, 6, "h_body=512, n_levels=9 → k_split=6");

        // Shallow levels (k < k_split): data Vec sized at bw × R_k.
        // Per the canonical table in strip.rs:k_split_table_matches_gpu_doc:
        //   R_0 = 1148, R_1 = 572, R_2 = 284, R_3 = 140, R_4 = 68, R_5 = 32
        let expected_r_k = [1148_usize, 572, 284, 140, 68, 32];
        let mut bw = w;
        let mut bh = h;
        for k in 0..k_split {
            assert_eq!(pyr.bands[k].w, bw, "level {k} bw");
            // The h field stays at the FULL band height (logical shape preserved).
            assert_eq!(pyr.bands[k].h, bh, "level {k} bh (logical)");
            // The data Vec is strip-sized: bw × R_k.
            assert_eq!(
                pyr.bands[k].data.len(),
                bw * expected_r_k[k],
                "level {k} data.len() should be bw={bw} × R_{k}={}",
                expected_r_k[k]
            );
            assert_eq!(
                pyr.log_l_bkg[k].len(),
                bw * expected_r_k[k],
                "level {k} log_l_bkg.len() should be bw={bw} × R_{k}={}",
                expected_r_k[k]
            );
            bw = bw.div_ceil(2);
            bh = bh.div_ceil(2);
        }

        // Deep levels (k >= k_split): data Vec sized at bw × bh (full).
        for k in k_split..n_levels {
            assert_eq!(pyr.bands[k].w, bw, "deep level {k} bw");
            assert_eq!(pyr.bands[k].h, bh, "deep level {k} bh");
            assert_eq!(
                pyr.bands[k].data.len(),
                bw * bh,
                "deep level {k} data.len() should be bw × bh = {} × {}",
                bw,
                bh
            );
            bw = bw.div_ceil(2);
            bh = bh.div_ceil(2);
        }
    }

    /// Chunk 6 step 1 regression: `WeberPyramidCache::with_capacity_strip`
    /// applies the same strip-shape policy to gauss_img + gauss_l.
    #[test]
    fn weber_pyramid_cache_with_capacity_strip_sizes_shallow_levels() {
        let w = 4096_usize;
        let h = 4096_usize;
        let n_levels = 9_usize;
        let h_body = 512_u32;
        let cache = WeberPyramidCache::with_capacity_strip(w, h, n_levels, h_body);
        let k_split = crate::strip::mode_b_k_split(h_body, n_levels as u32) as usize;

        let expected_r_k = [1148_usize, 572, 284, 140, 68, 32];
        let mut bw = w;
        let mut bh = h;
        for k in 0..k_split {
            assert_eq!(cache.gauss_img[k].data.len(), bw * expected_r_k[k]);
            assert_eq!(cache.gauss_l[k].data.len(), bw * expected_r_k[k]);
            // Logical h field stays at full band height.
            assert_eq!(cache.gauss_img[k].h, bh);
            assert_eq!(cache.gauss_l[k].h, bh);
            bw = bw.div_ceil(2);
            bh = bh.div_ceil(2);
        }
        for k in k_split..n_levels {
            assert_eq!(cache.gauss_img[k].data.len(), bw * bh);
            assert_eq!(cache.gauss_l[k].data.len(), bw * bh);
            bw = bw.div_ceil(2);
            bh = bh.div_ceil(2);
        }
    }

    /// Total persistent allocation accounting for chunk 6 step 1.
    ///
    /// At 16 MP / h_body=512 / 9 levels:
    /// - Full-image: 6 WeberPyramid + 3 WeberPyramidCache ≈ 1.6 GB worth of pyramid mass
    /// - Strip-shape: same structure ≈ 460 MB (~71% reduction)
    ///
    /// This test computes the byte counts at construction to lock in the
    /// allocation policy. If the strip-shape formula drifts, this test
    /// catches it.
    #[test]
    fn weber_pyramid_strip_vs_full_byte_accounting_16mp() {
        let w = 4096_usize;
        let h = 4096_usize;
        let n_levels = 9_usize;
        let h_body = 512_u32;

        let full = WeberPyramid::with_capacity(w, h, n_levels);
        let strip = WeberPyramid::with_capacity_strip(w, h, n_levels, h_body);

        let full_bytes: usize = full
            .bands
            .iter()
            .map(|b| b.data.len() * 4 + b.data.len() * 4) // data + log_l_bkg (same size)
            .sum::<usize>()
            / 2  // we counted twice; just one band side
            + full.log_l_bkg.iter().map(|v| v.len() * 4).sum::<usize>();
        let strip_bytes: usize = strip.bands.iter().map(|b| b.data.len() * 4).sum::<usize>()
            + strip.log_l_bkg.iter().map(|v| v.len() * 4).sum::<usize>();

        // Strip should be substantially smaller than full at 16 MP.
        // Allow 30% bound — exact ratio depends on R_k computation.
        assert!(
            strip_bytes < full_bytes / 2,
            "strip allocation should be < half of full: strip={strip_bytes}, full={full_bytes}"
        );

        // Strip should be > 0 (we still allocate deep levels and at least
        // some shallow rows).
        assert!(strip_bytes > 0);
    }
}
