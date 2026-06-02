//! Strip-based orchestration for the butteraugli pipeline — reduces
//! peak GPU memory at large image sizes (24 MP and up).
//!
//! Whole-image mode allocates ~38 planes of `width × height × f32` (3
//! linear-RGB plane sets × 2 images = 6, plus 24 frequency-band planes,
//! plus 8 misc accumulators / scratch). At 24 MP that's roughly 4.8 GB
//! — well over the budget for a 16 GB consumer GPU shared with the
//! encoder front-end.
//!
//! The strip walker allocates the same 38 planes at a smaller
//! `width × strip_h_total` and re-uses them across N body-aligned
//! strips. Per-strip output is reduced to (max, Σd³, Σd⁶, Σd¹²) on the
//! host; the final aggregated score is bit-identical to the
//! whole-image path up to f64 reduction order.
//!
//! ## Halo derivation
//!
//! The pipeline composes blurs and small-window kernels in sequence.
//! For the body rows of an interior strip to match the whole-image
//! output, the halo rows above and below the body must hold real
//! image-aligned content for every intermediate plane that the body
//! reads from. Tracing the dependency chain:
//!
//! - `compute_diffmap` (pointwise) — body+0
//! - `compute_dc_diff` (pointwise) — body+0
//! - mask: `combine + diff_precompute` (pointwise) → blur σ=2.7 (r=6) →
//!   `fuzzy_erosion` (r=3) → mask needs body+0; HF/UHF input must be
//!   valid at body+6+3 = **body+9**.
//! - `malta_triple` over UHF/HF/MF — radius 4 → UHF/HF/MF must be valid
//!   at **body+4** (subsumed by the +9 above for UHF/HF).
//! - HF, UHF after `HF/UHF separation` — V-blur on freq[2] (MF
//!   intermediate) reads ±r(σ=3.22)=±7. → MF intermediate must be
//!   valid at body+9+7 = **body+16**.
//! - MF intermediate (= freq[2] after `LF separation`) — V-blur on
//!   opsin output reads ±r(σ=7.16)=±16. → opsin output must be valid
//!   at body+16+16 = **body+32**.
//! - opsin output — V-blur on lin-RGB reads ±r(σ=1.2)=±2. → lin-RGB
//!   must be valid at body+32+2 = **body+34**.
//! - lin-RGB ← `srgb_to_linear` (pointwise) — u8 must be valid at
//!   body+34.
//!
//! So the *single-resolution* strip path needs **34** real halo rows
//! per side. `new_strip` / `run_strip_pipeline` are bit-identical to
//! whole-image at that depth (verified on aggressive HF checkerboards,
//! not just smooth content — see `tests/strip_hf_checkerboard.rs`).
//!
//! ## Why HALO_ROWS = 80 (the multires-strip requirement)
//!
//! The 34-row depth above is the single-resolution requirement. The
//! **multi-resolution** strip walker ([`run_strip_pipeline_multires`])
//! drives a half-resolution sibling whose slab is built by 2×
//! downsampling the full-res strip's linear-RGB planes. The half-res
//! blur cascade uses the SAME sigmas (σ=7.16 / 3.22 / 1.2 …) operating
//! in half-res *pixel* space, so it independently needs **34 half-res
//! halo rows** per side to be exact.
//!
//! The half-res strip cannot fabricate halo content out of nothing — it
//! only sees what the full-res strip slab provides, downsampled 2×. So
//! the half-res strip's available halo is `full_halo / 2`. For the
//! half-res side to have its 34 rows, the full-res halo must be
//! `2 × 34 = 68`. `HALO_ROWS = 80` clears that (40 half-res rows ≥ 34)
//! with slack for any future stage adding ±a-few rows of work.
//!
//! This was measured: with `HALO_ROWS = 40` the half-res strip got only
//! `40 / 2 = 20` real halo rows (< 34), so on an aggressive period-8
//! ±12 checkerboard the multires-strip max-norm `score` drifted from
//! multires-whole by up to ~7e-4 rel at 512² (the single worst
//! boundary pixel landed on an under-haloed half-res row). At
//! `HALO_ROWS = 80` the half-res side is fully haloed and multires-strip
//! matches multires-whole bit-identically on the same content (max-norm
//! 0.0e0, pnorm_3 ≤ 2e-7). See `benchmarks/butter_strip_halo_2026-05-31`.
//!
//! Each strip plane is sized `width × (body_h + 2 * HALO_ROWS)`. The
//! cost of the 40→80 bump is borne only by strip mode and only as extra
//! halo rows (e.g. body-256 slab grows 336→416 rows, +24%); the strip
//! memory/wall wins at 4096² (≈6–10× leaner/faster than Full) retain
//! ample margin. Correctness — Strip score == Full score — strictly
//! dominates the small memory delta.
//!
//! ## Edge handling
//!
//! Strips at the top of the image have zero "above" content — the
//! top halo is populated by edge-mirror of image row 0. Strips at the
//! bottom mirror image row `image_h - 1` for the bottom halo. Interior
//! strips populate halo rows from the real image rows immediately above
//! and below the body.
//!
//! This matches the blur kernels' built-in `saturating_sub` / `min(h-1)`
//! clamping: a body row sees the same window in the strip plane as it
//! would see in the whole-image plane.
//!
//! ## Multi-resolution strip walker
//!
//! [`run_strip_pipeline_multires`] handles the multires-strip case:
//! it iterates the full-res strips just like the single-resolution
//! walker, but each full-res strip pass also drives the half-res
//! sibling. The half-res strip's body covers half-res image rows
//! `[body_top_full / 2, body_end_full.div_ceil(2))` and its slab
//! is built by 2× downsampling the full-res strip's linear-RGB
//! planes — no separate half-res sRGB buffer is needed (and
//! constructing one would defeat the strip memory savings).
//!
//! The constructor [`Butteraugli::new_multires_strip`] enforces an
//! even `body_h`, which keeps every full-res strip's body_top even
//! and lets the half-res strip mirror it exactly. For images whose
//! `image_h` isn't a multiple of `body_h`, the last full-res strip
//! has a smaller body whose half-res counterpart uses
//! `body_end_full.div_ceil(2) - body_top_full/2` rows so the
//! half-res image's last row is covered.
//!
//! ## Mode E strip walker (cached ref + strip dist)
//!
//! [`run_strip_pipeline_cached_ref`] implements task #45 / issue #15
//! "Mode E": the reference image is held whole-image in a sibling
//! `Butteraugli<R>` cache (allocated lazily on the first
//! `set_reference` call against a strip-mode instance), and each
//! distorted strip's reference-side intermediates are blitted in
//! per-strip via a row-offset slab-copy kernel. Skips the ref-side
//! opsin → freq separation → mask-reference pipeline once per strip.
//!
//! ## What this MVP does NOT do
//!
//! - Mode E multires-strip: the multires-strip path
//!   ([`run_strip_pipeline_multires`]) does not yet wire through a
//!   half-res cached-ref blit. `set_reference` on a multires-strip
//!   instance returns `Error::StripModeUnsupported` so callers fall
//!   back to one-shot `compute_strip`.

use crate::pipeline::Butteraugli;
use crate::{ButteraugliParams, Error, GpuButteraugliResult, Result};

use cubecl::prelude::*;

/// Safe halo size above and below each strip's body in rows. See module
/// docs for the derivation. Body + 2 × HALO_ROWS rows are allocated for
/// every plane in the strip-mode `Butteraugli` instance.
///
/// `80`, not the single-resolution requirement of `34`/`40`: the
/// multi-resolution strip walker's half-res sibling only sees
/// `HALO_ROWS / 2` real halo rows (it is downsampled 2× from the
/// full-res strip slab), and the half-res blur cascade needs its own
/// 34 rows. `80 / 2 = 40 ≥ 34` makes both resolutions exact. See the
/// "Why HALO_ROWS = 80" section in the module docs.
pub const HALO_ROWS: u32 = 80;

/// Per-strip host-side partials (max + p3/p6/p12 sums over the body
/// rows only — halo rows are discarded). Folded into the final score
/// after all strips have been processed.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct StripPartials {
    pub max: f32,
    pub sum_p3: f64,
    pub sum_p6: f64,
    pub sum_p12: f64,
}

impl StripPartials {
    pub(crate) fn merge(&mut self, other: &Self) {
        if other.max > self.max {
            self.max = other.max;
        }
        self.sum_p3 += other.sum_p3;
        self.sum_p6 += other.sum_p6;
        self.sum_p12 += other.sum_p12;
    }

    pub(crate) fn finalize(&self, n_image_pixels: usize) -> GpuButteraugliResult {
        let n_inv = 1.0_f64 / (n_image_pixels as f64);
        let v0 = (n_inv * self.sum_p3).powf(1.0 / 3.0);
        let v1 = (n_inv * self.sum_p6).powf(1.0 / 6.0);
        let v2 = (n_inv * self.sum_p12).powf(1.0 / 12.0);
        let pnorm_3 = ((v0 + v1 + v2) / 3.0) as f32;
        GpuButteraugliResult {
            score: self.max,
            pnorm_3,
        }
    }
}

/// Reduce one strip's `width × strip_h_total` diffmap, considering only
/// the `[body_top, body_top + body_h)` row band, into running partials.
///
/// Body rows are read host-side; this avoids a new on-device kernel
/// for the body-bounded reduce. At 24 MP with body_h = 256 and
/// strip_h_total = 336, one strip's diffmap is ~8 MB — `read_one` plus
/// the host pass is ~3-5 ms, swamped by the GPU compute time.
pub(crate) fn reduce_strip_body<R: Runtime>(
    client: &ComputeClient<R>,
    diffmap_handle: cubecl::server::Handle,
    width: u32,
    _strip_h_total: u32,
    body_top: u32,
    body_h: u32,
) -> StripPartials {
    let bytes = client
        .read_one(diffmap_handle)
        .expect("read_one strip diffmap");
    let plane = f32::from_bytes(&bytes);
    let w = width as usize;
    let body_start = (body_top as usize) * w;
    let body_end = body_start + (body_h as usize) * w;
    debug_assert!(
        body_end <= plane.len(),
        "strip body rows out of range: body_end={body_end}, plane_len={}",
        plane.len()
    );

    let mut partials = StripPartials::default();
    for &v in &plane[body_start..body_end] {
        if v > partials.max {
            partials.max = v;
        }
        let d = v as f64;
        let d3 = d * d * d;
        partials.sum_p3 += d3;
        let d6 = d3 * d3;
        partials.sum_p6 += d6;
        partials.sum_p12 += d6 * d6;
    }
    partials
}

/// Build the packed-u32 sRGB strip buffer for the (k=strip_index)-th
/// strip of an `image_w × image_h` sRGB-u8 source.
///
/// `body_top_img` is the image row where the body starts (inclusive);
/// `body_h_img` is the body's row count; `strip_h_total` is the
/// allocated strip height (body + 2 × halo, may be less near the top
/// or bottom edge — the caller computes the actual usable rows).
///
/// Halo rows above the body, if they sit above the image's row 0, are
/// edge-clamped (copy of image row 0). Likewise below row image_h - 1.
/// This matches the blur kernels' edge-clamp `saturating_sub` /
/// `min(end, h - 1)` behavior.
///
/// Output is `width × strip_h_total` packed `u32`s (R | G<<8 | B<<16,
/// alpha unused), matching the layout
/// [`Butteraugli::pack_srgb_into_packed_u32_handle`] produces.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_strip_srgb_into(
    dst: &mut [u8],
    src: &[u8],
    image_w: u32,
    image_h: u32,
    body_top_img: u32,
    strip_h_total: u32,
    halo_top: u32,
) {
    let w = image_w as usize;
    let pinned_len = (w * strip_h_total as usize) * 4;
    debug_assert_eq!(dst.len(), pinned_len);

    for sy in 0..strip_h_total as usize {
        // Image row this strip row corresponds to (with edge clamp at
        // image top and bottom — matches blur edge-clamp).
        let img_row_i = (body_top_img as i64) + (sy as i64) - (halo_top as i64);
        let img_row = img_row_i.max(0).min(image_h as i64 - 1) as usize;
        let src_off = img_row * w * 3;
        let dst_off = sy * w * 4;
        let src_row = &src[src_off..src_off + w * 3];
        let dst_row = &mut dst[dst_off..dst_off + w * 4];
        for (chunk_out, triple) in dst_row.chunks_exact_mut(4).zip(src_row.chunks_exact(3)) {
            chunk_out[0] = triple[0];
            chunk_out[1] = triple[1];
            chunk_out[2] = triple[2];
            chunk_out[3] = 0;
        }
    }
}

/// Drive the strip walker for `compute_strip` / `compute_strip_with_options`.
/// Lives in `strip.rs` so the public `Butteraugli` impl in `pipeline.rs`
/// stays focused on whole-image flow.
pub(crate) fn run_strip_pipeline<R: Runtime>(
    state: &mut Butteraugli<R>,
    ref_srgb: &[u8],
    dist_srgb: &[u8],
    image_w: u32,
    image_h: u32,
    body_h: u32,
    halo_h: u32,
    params: &ButteraugliParams,
) -> Result<GpuButteraugliResult> {
    crate::pipeline::validate_params(params)?;
    let expected = (image_w as usize) * (image_h as usize) * 3;
    if ref_srgb.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: ref_srgb.len(),
        });
    }
    if dist_srgb.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: dist_srgb.len(),
        });
    }
    state.set_params(*params);

    let n_pixels_image = (image_w as usize) * (image_h as usize);
    let mut combined = StripPartials::default();

    // Walk strips: each strip's body is a row range [body_top, body_end)
    // in image coordinates. Strips are non-overlapping in their bodies.
    let mut body_top: u32 = 0;
    while body_top < image_h {
        let body_end = (body_top + body_h).min(image_h);
        let this_body_h = body_end - body_top;

        // Halo sizing per strip — KEY EDGE-HANDLING RULE.
        //
        // At image edges (top or bottom), the strip plane MUST have its
        // edge coincide with the image edge — zero halo on that side.
        // This lets each kernel's built-in `saturating_sub(y, r)` and
        // `min(end, h - 1)` edge-clamp fire exactly where it would in
        // the whole-image pass, producing the same partial-window output
        // (truncated taps + normalised by partial weight-sum). If we
        // padded the top with replicated row-0 instead, the kernel would
        // see a full-window sum normalised by the FULL weight-sum, which
        // averages in extra "image row 0" content and breaks parity.
        //
        // For interior strips the halo holds real image content above
        // and below the body, so the kernel sees the same neighbourhood
        // it would have seen in the whole-image pass.
        let halo_top = body_top.min(halo_h);
        let halo_bot = (image_h - body_end).min(halo_h);
        let strip_h_total = halo_top + this_body_h + halo_bot;

        // Upload ref / dist strip planes into the pre-allocated src_u8
        // handles. We reuse Butteraugli::set_strip_srgb (added in
        // pipeline.rs).
        state.upload_strip_srgb(
            true,
            ref_srgb,
            image_w,
            image_h,
            body_top,
            strip_h_total,
            halo_top,
        );
        state.upload_strip_srgb(
            false,
            dist_srgb,
            image_w,
            image_h,
            body_top,
            strip_h_total,
            halo_top,
        );

        // Drive the existing whole-image pipeline using the strip plane
        // as if it were the full image. The kernels see height =
        // strip_h_total and clamp at strip top / bottom, but the halo
        // rows hold real image content, so body rows compute the same
        // outputs they would in the whole-image pass.
        state.run_strip_pipeline_compute(strip_h_total);

        // Reduce the body band of this strip's diffmap into running
        // partials.
        let diffmap_handle = state.diffmap_buf_handle();
        let strip_partials = reduce_strip_body::<R>(
            state.client_ref(),
            diffmap_handle,
            image_w,
            strip_h_total,
            halo_top,
            this_body_h,
        );
        combined.merge(&strip_partials);

        body_top = body_end;
    }

    Ok(combined.finalize(n_pixels_image))
}

/// Drive the multires strip walker for `compute_strip` /
/// `compute_strip_with_options` when the instance has a half-res
/// sibling.
///
/// Iterates the full-res image in `body_h`-tall bands. For each strip:
///
/// 1. Upload `(ref, dist)` sRGB strip planes to the full-res
///    instance.
/// 2. Run the full-res strip pipeline up to and including the
///    diffmap.
/// 3. Downsample the full-res strip's linear-RGB slab into the
///    half-res sibling's linear-RGB slab.
/// 4. Run the half-res strip pipeline.
/// 5. Supersample-add the half-res strip diffmap into the full-res
///    strip diffmap.
/// 6. Reduce the full-res strip's body rows into the running
///    partials.
///
/// Halo alignment: the constructor guarantees `body_h_full` is even,
/// so every full-res `body_top` is even and `halo_h_full` is the
/// same `HALO_ROWS`. The half-res instance gets `body_h_full / 2`
/// for its body and the same `HALO_ROWS` for its halo. Within a
/// strip pass, the half-res strip slab's height is
/// `halo_top_half + body_h_half + halo_bot_half` where each value
/// is the floor / ceil of the corresponding full-res value as
/// appropriate so the half-res strip covers the half-res rows
/// `[body_top_full / 2, body_end_full.div_ceil(2))`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_strip_pipeline_multires<R: Runtime>(
    state: &mut Butteraugli<R>,
    ref_srgb: &[u8],
    dist_srgb: &[u8],
    image_w: u32,
    image_h: u32,
    body_h: u32,
    halo_h: u32,
    params: &ButteraugliParams,
) -> Result<GpuButteraugliResult> {
    crate::pipeline::validate_params(params)?;
    let expected = (image_w as usize) * (image_h as usize) * 3;
    if ref_srgb.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: ref_srgb.len(),
        });
    }
    if dist_srgb.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: dist_srgb.len(),
        });
    }
    // body_h MUST be even for the half-res alignment to be exact.
    // The constructor enforces this; if a caller bypassed the
    // constructor we want to fail loudly rather than silently
    // mis-align.
    debug_assert_eq!(body_h % 2, 0, "multires strip requires even body_h");
    state.set_params(*params);
    if let Some(half) = state.half_res_mut() {
        half.set_params(*params);
    }

    let n_pixels_image = (image_w as usize) * (image_h as usize);
    let mut combined = StripPartials::default();

    let half_image_h = image_h.div_ceil(2);

    let mut body_top: u32 = 0;
    while body_top < image_h {
        let body_end = (body_top + body_h).min(image_h);
        let this_body_h = body_end - body_top;

        // Full-res halo sizing — same edge-clamp rule as the single-
        // res walker (see `run_strip_pipeline` for the derivation).
        // body_top is guaranteed even by the constructor + the
        // body_h-even invariant; halo_h (HALO_ROWS) is also even, so
        // halo_top_full is min(body_top, HALO_ROWS) which is even iff
        // body_top is. Since body_top is even and halo_h is even,
        // halo_top_full is always even.
        let halo_top = body_top.min(halo_h);
        let halo_bot = (image_h - body_end).min(halo_h);
        let strip_h_total = halo_top + this_body_h + halo_bot;

        // Half-res strip rows: cover half-res image rows
        // [body_top_half, body_end_half) where:
        let body_top_half = body_top / 2;
        let body_end_half = body_end.div_ceil(2).min(half_image_h);
        let this_body_h_half = body_end_half - body_top_half;
        // halo_top_full is even, so halo_top_half = halo_top_full / 2.
        // halo_bot_half is the smaller of (HALO_ROWS, remaining
        // half-res image rows below the body).
        let halo_top_half = halo_top / 2;
        let halo_bot_half = (half_image_h - body_end_half).min(halo_h);
        let strip_h_total_half = halo_top_half + this_body_h_half + halo_bot_half;

        // ── Full-res strip pass ──
        state.upload_strip_srgb(
            true,
            ref_srgb,
            image_w,
            image_h,
            body_top,
            strip_h_total,
            halo_top,
        );
        state.upload_strip_srgb(
            false,
            dist_srgb,
            image_w,
            image_h,
            body_top,
            strip_h_total,
            halo_top,
        );

        // Run the full-res pipeline UP TO opsin: we want the linear-RGB
        // planes populated so we can downsample them into the half-res
        // sibling BEFORE opsin overwrites them. Since
        // `apply_opsin` writes XYB back into `lin`, we must do the
        // downsample first.
        //
        // The full-res pipeline-up-to-diffmap chain is the same as the
        // single-resolution strip walker, so we can call
        // `run_strip_pipeline_compute` AFTER downsampling. Order:
        //   1. upload (already done above)
        //   2. downsample linear-RGB into half-res slab
        //   3. run full-res pipeline (apply_opsin, freq, mask, diff)
        //   4. run half-res pipeline (apply_opsin, freq, mask, diff)
        //   5. supersample-add half-res diffmap into full-res diffmap
        //   6. reduce full-res body rows

        // Step 2: downsample full-res lin → half-res lin (slab to
        // slab). Temporarily clamp both heights so the downsample
        // kernel covers the populated rows only.
        let mut half = state
            .take_half_res()
            .expect("multires strip walker invoked without half_res sibling");
        // Both still have their slab geometry as `height`. We pass
        // explicit strip_h_total values to the helper, which clamps
        // internally without needing to touch self.height. (The
        // downsample kernel takes src/dst dims explicitly.)
        state.downsample_full_strip_into_half(&half, strip_h_total, strip_h_total_half);

        // Step 3 + 4: drive the full-res and half-res strip
        // pipelines on the now-populated linear-RGB planes.
        state.run_strip_pipeline_compute(strip_h_total);
        half.run_strip_pipeline_compute_lin_only(strip_h_total_half);

        // Step 5: supersample-add half → full.
        state.add_supersampled_from_half_strip(&half, strip_h_total, strip_h_total_half);

        // Step 6: reduce body rows of full-res strip diffmap.
        let diffmap_handle = state.diffmap_buf_handle();
        let strip_partials = reduce_strip_body::<R>(
            state.client_ref(),
            diffmap_handle,
            image_w,
            strip_h_total,
            halo_top,
            this_body_h,
        );
        combined.merge(&strip_partials);

        // Restore half_res for the next iteration / for the caller's
        // accessors.
        state.restore_half_res(half);

        body_top = body_end;
    }

    Ok(combined.finalize(n_pixels_image))
}

/// Mode E strip walker (task #45 / issue #15): walk the distorted
/// image in strips and skip the reference-side recompute per strip
/// by blitting reference planes from a whole-image
/// `Butteraugli<R>::ref_cache_full` sibling.
///
/// The constructor invariant: `state` is a strip-mode instance
/// (`halo_h > 0`) WITHOUT a half-res sibling. `state.ref_cache_full`
/// MUST be `Some` and have `has_reference == true` (the strip-
/// mode `set_reference_with_options` populates this).
///
/// Per-strip pipeline:
/// 1. Upload distorted sRGB strip to `state.src_u8_b` and run sRGB→linear.
/// 2. Blit row slabs of reference planes (`lin_a`, `freq_a[*][*]`,
///    `cached_blurred_a`, `mask`) from `ref_cache_full` into
///    `state`'s strip planes.
/// 3. Run the distorted-side pipeline + mask combine + diffmap +
///    body-band reduction.
///
/// Identical body-row diffmap values to whole-image
/// `compute_with_reference`, modulo the Atomic<f32>::fetch_add
/// scheduling drift the cached-ref tests already tolerate.
pub(crate) fn run_strip_pipeline_cached_ref<R: Runtime>(
    state: &mut Butteraugli<R>,
    dist_srgb: &[u8],
    image_w: u32,
    image_h: u32,
    body_h: u32,
    halo_h: u32,
    params: &ButteraugliParams,
) -> Result<GpuButteraugliResult> {
    crate::pipeline::validate_params(params)?;
    let expected = (image_w as usize) * (image_h as usize) * 3;
    if dist_srgb.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            got: dist_srgb.len(),
        });
    }
    state.set_params(*params);

    // Borrow the ref cache out so we can pass &mut state and &cache
    // into kernel-launch helpers without splitting the borrow.
    let cache = state
        .take_ref_cache_full()
        .expect("compute_with_reference_strip_mode: ref_cache_full must be Some");

    let n_pixels_image = (image_w as usize) * (image_h as usize);
    let mut combined = StripPartials::default();

    let mut body_top: u32 = 0;
    while body_top < image_h {
        let body_end = (body_top + body_h).min(image_h);
        let this_body_h = body_end - body_top;

        // Halo sizing (same edge-clamp rule as `run_strip_pipeline`).
        let halo_top = body_top.min(halo_h);
        let halo_bot = (image_h - body_end).min(halo_h);
        let strip_h_total = halo_top + this_body_h + halo_bot;

        // Source row in the FULL cache where the strip's slab starts.
        let src_row_offset = body_top - halo_top;
        debug_assert!(src_row_offset + strip_h_total <= image_h);

        // 1) Dist sRGB upload + sRGB→linear for this strip.
        state.upload_strip_srgb(
            false,
            dist_srgb,
            image_w,
            image_h,
            body_top,
            strip_h_total,
            halo_top,
        );

        // 2) Blit ref-side intermediates from the full cache.
        state.blit_ref_slab_from_cache(&cache, src_row_offset, strip_h_total);

        // 3) Distorted-side compute + mask + diffmap.
        state.run_strip_pipeline_compute_dist_only(strip_h_total);

        // 4) Reduce body band into running partials.
        let diffmap_handle = state.diffmap_buf_handle();
        let strip_partials = reduce_strip_body::<R>(
            state.client_ref(),
            diffmap_handle,
            image_w,
            strip_h_total,
            halo_top,
            this_body_h,
        );
        combined.merge(&strip_partials);

        body_top = body_end;
    }

    state.restore_ref_cache_full(cache);
    Ok(combined.finalize(n_pixels_image))
}
