//! zensim pipeline orchestration.
//!
//! Wires the kernels in `kernels::*` into the 4-scale zensim feature
//! extractor.
//!
//! Public entry points:
//! - [`Zensim::new`] + [`Zensim::compute_features`] — extract the
//!   228-feature vector from one (ref, dist) pair.
//! - [`Zensim::set_reference`] + [`Zensim::compute_with_reference`] —
//!   cache the reference-side pyramid and score many distorted images
//!   against it (encoder-loop friendly).
//!
//! ## Algorithm (per scale, faithful to `zensim-cuda`'s `compute_features`)
//!
//! 1. sRGB packed-u8 → planar positive-XYB (Halley `cbrtf_fast`).
//! 2. Mirror-fill SIMD-padding columns `[logical_w..padded_w)`.
//! 3. 2× planar downscale to build the pyramid (4 levels).
//! 4. Per scale, per channel:
//!    a. Fused H-blur (mu1, mu2, sigma_sq, sigma12).
//!    b. Fused V-blur + per-pixel features → 17 f64 + 3 f32 per column.
//!    c. Host-side fold across columns → per-channel feature row.
//! 5. Pack into the 228-entry vector matching CPU layout (basic block
//!    of 156 + peaks block of 72).
//!
//! ## Buffer layout
//!
//! Buffers are flat `padded_w × height` planar f32 arrays — no
//! pitched-2D padding. CPU zensim doesn't depend on alignment within
//! the row beyond the SIMD-pad columns we explicitly emit, so flat
//! storage matches its math without translation.

use cubecl::prelude::*;

use crate::kernels::{self, color, downscale, fused, masked_iw_strip, reduce};
// `masked_iw` retained for back-compat in tests; not used in the
// production extended path.
#[allow(unused_imports)]
use crate::kernels::masked_iw;
use crate::{
    Error, FEATURES_PER_CHANNEL_BASIC, FEATURES_PER_CHANNEL_IW, FEATURES_PER_CHANNEL_MASKED,
    FEATURES_PER_CHANNEL_PEAKS, Result, SCALES, TOTAL_FEATURES, ZensimFeatureRegime,
    simd_padded_width,
};

// `logical_w` and `partials_*_per_scale` are bookkeeping kept for
// future debug tooling (per-channel intermediate dump). The
// pre-fused-kernel `h_mu1..h_sigma12` H-blur scratch planes were
// removed in T_z.B (2026-05-16): the tile-fused `fused_features_kernel`
// allocates its working set in shared memory, so 12 padded-f32 planes
// per scale (~576 MB of zero-fill traffic at 12 MP) were dead weight.
#[allow(dead_code)]
struct Scale {
    logical_w: u32,
    padded_w: u32,
    h: u32,
    n_padded: usize,
    n_strips: u32,
    /// Strip count for the **extended (masked + IW) kernel**. Differs
    /// from `n_strips` (which is tuned for GPU occupancy of the basic
    /// kernel) because the masked path reproduces CPU's CPU-shaped
    /// strip layout for parity. Equals `ceil(h / STRIP_INNER)` from
    /// [`kernels::masked_iw_strip::cpu_strip_count`].
    n_strips_ext: u32,

    /// Three planar XYB planes per side at `padded_w × h`. Allocated
    /// `empty()` — `srgb_to_positive_xyb_kernel` writes every pixel
    /// in `[0, padded_w) × [0, h)` (including the mirror-padded
    /// columns) so zero-fill on the host side is unnecessary.
    ref_xyb: [cubecl::server::Handle; 3],
    dis_xyb: [cubecl::server::Handle; 3],

    /// Mirror-offset table (one u32 per padding column). `None` when
    /// `padded_w == logical_w`.
    mirror_offsets: Option<cubecl::server::Handle>,
    pad_count: u32,

    /// Offset (in f64 / f32 units) of this scale's partials within the
    /// big shared `partials_*` buffers. Layout per scale:
    /// `[ch0 strip0 col0 .. col(pw-1) | ch0 strip1 ... | ch1 strip0 ... | ...]`
    /// with 17 f64 (or 3 f32) per slot.
    partials_f64_off: usize,
    partials_max_off: usize,
    partials_f64_per_scale: usize, // = pw × n_strips × 3 channels × 17
    partials_max_per_scale: usize, // = pw × n_strips × 3 channels × 3

    /// Offset of this scale's masked + IW partials within the
    /// `partials_ext_f64` buffer. `0` (unused) when regime == Basic.
    partials_ext_off: usize,
    partials_ext_per_scale: usize, // = pw × n_strips_ext × 3 channels × 12
}

/// Allocate an uninitialised f32 plane on-device. Use only when the
/// caller writes every element before the next kernel reads any —
/// the fused features pipeline matches that contract for every plane
/// (xyb produced by `srgb_to_positive_xyb_kernel`, downscale outputs
/// produced by `downscale_2x_3ch_kernel`, partials overwritten by
/// `fused_features_kernel`'s per-thread store).
fn alloc_empty_f32<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.empty(n * core::mem::size_of::<f32>())
}
fn alloc_empty_f64<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.empty(n * core::mem::size_of::<f64>())
}
/// Choose a per-scale strip count to keep V-blur GPU-occupied at all
/// resolutions. The kernel's parallelism is `padded_w × n_strips × 3
/// channels`. RTX-5070-class GPUs want ≥ 16 K resident threads to
/// hide latency.
fn pick_n_strips(padded_w: u32, height: u32) -> u32 {
    if height <= 64 {
        1
    } else if height >= 1024 {
        8
    } else if padded_w >= 256 {
        4
    } else {
        2
    }
}

impl Scale {
    fn new<R: Runtime>(
        client: &ComputeClient<R>,
        logical_w: u32,
        padded_w: u32,
        h: u32,
        partials_f64_off: usize,
        partials_max_off: usize,
        partials_ext_off: usize,
    ) -> Self {
        let n = (padded_w as usize) * (h as usize);
        let alloc3_empty = || -> [cubecl::server::Handle; 3] {
            [
                alloc_empty_f32(client, n),
                alloc_empty_f32(client, n),
                alloc_empty_f32(client, n),
            ]
        };
        let pad_count = padded_w - logical_w;
        let n_strips = pick_n_strips(padded_w, h);
        // Extended kernel uses CPU's strip layout for parity.
        let n_strips_ext = kernels::masked_iw_strip::cpu_strip_count(h);

        // Mirror-offset table matching CPU zensim
        // (streaming.rs:591-601):
        //   period = 2 * (logical_w - 1)
        //   for i in 0..pad_count:
        //     m = (logical_w + i) % period
        //     offset = if m < logical_w { m } else { period - m }
        let mirror_offsets = if pad_count > 0 {
            let lw = logical_w as usize;
            let pc = pad_count as usize;
            let period = 2 * (lw - 1);
            let host: Vec<u32> = (0..pc)
                .map(|i| {
                    let m = (lw + i) % period;
                    let off = if m < lw { m } else { period - m };
                    off as u32
                })
                .collect();
            Some(client.create_from_slice(u32::as_bytes(&host)))
        } else {
            None
        };

        Self {
            logical_w,
            padded_w,
            h,
            n_padded: n,
            ref_xyb: alloc3_empty(),
            dis_xyb: alloc3_empty(),
            mirror_offsets,
            pad_count,
            partials_f64_off,
            partials_max_off,
            partials_f64_per_scale: (padded_w as usize) * (n_strips as usize) * 3 * 17,
            partials_max_per_scale: (padded_w as usize) * (n_strips as usize) * 3 * 3,
            partials_ext_off,
            partials_ext_per_scale: (padded_w as usize) * (n_strips_ext as usize) * 3 * 12,
            n_strips,
            n_strips_ext,
        }
    }
}

/// One per-resolution zensim pipeline. Allocate once with
/// [`Zensim::new`] for a (width, height); reuse across many image pairs
/// of that resolution.
pub struct Zensim<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    pixels: usize,

    /// Persistent host-side packing scratch (one u32 per pixel = R |
    /// G<<8 | B<<16). Reused across uploads to avoid the alloc + iter
    /// per `compute_with_reference`.
    pack_scratch: Vec<u32>,

    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    /// Device copy of `color::SRGB8_TO_LINEARF32_LUT`.
    srgb_lut: cubecl::server::Handle,

    scales: Vec<Scale>,

    /// Persistent partials buffers. Sized to fit all (scale, channel)
    /// per-column slots. Avoids the per-channel alloc-then-read churn
    /// that dominated the warm-path cost in the original pipeline.
    partials_f64: cubecl::server::Handle,
    partials_max: cubecl::server::Handle,
    partials_f64_len: usize,
    partials_max_len: usize,

    /// Final per-(scale, channel, slot) sums after the on-device
    /// reduction pass — small enough that host read-back is sub-µs.
    finals_f64: cubecl::server::Handle,
    finals_max: cubecl::server::Handle,

    has_cached_reference: bool,

    /// Cached acumen Mode A weights: `[channel][scale] -> f32 castleCSF
    /// sensitivity`. Computed inside [`Zensim::set_reference`] when
    /// the caller-supplied viewing condition is set via
    /// [`Zensim::set_acumen_viewing`]. Phase 4 multiplies the HF band-
    /// energy features (basic slots 10–12) by the matching scalar
    /// when this field is `Some`. `None` => bit-stable V_22 path.
    acumen_band_weights: Option<[[f32; SCALES]; 3]>,

    /// Sticky viewing condition for [`Self::acumen_band_weights`]. When
    /// `Some`, the next [`Self::set_reference`] call computes new
    /// per-image weights using this viewing.
    acumen_viewing: Option<zensim::acumen::viewing::ViewingCondition>,

    /// Loaded once per process for acumen Mode A lookups. `None` until
    /// the first time the caller enables Mode A via
    /// [`Zensim::set_acumen_viewing`].
    acumen_lut_bytes: &'static [u8],

    // ───────── Extended / WithIw regime support ─────────
    regime: ZensimFeatureRegime,
    /// Per-scale per-channel mu1/mu2/ssq/s12 persist planes — laid out
    /// `[ch0_pixels | ch1_pixels | ch2_pixels]` per scale per side. One
    /// pair (ref + dist) per scale; only the ref-side mu1 is consumed
    /// by the masked-IW activity blur, but we persist all four for
    /// matching CPU's masked SSIM math (mu1 + mu2 + ssq + s12 at the
    /// SAME pixel). Each entry is `pad_total × 3` f32. Empty Vec on
    /// Basic regime — zero memory cost on the fast path.
    ///
    /// Indexed by `scales[s].partials_ext_off` is misleading — we
    /// instead keep per-scale per-channel handles directly because the
    /// pad_total varies per scale.
    persist_planes_ref: Vec<[cubecl::server::Handle; 4]>,
    /// Reserved for a future symmetric mask path that also runs the
    /// blur over `|dst - mu2|`. CPU zensim uses the ref-side only, so
    /// these are allocated but never written today.
    #[allow(dead_code)]
    persist_planes_dis: Vec<[cubecl::server::Handle; 4]>,

    /// Masked + IW per-(col, strip, ch) partials buffer. Length =
    /// Σ per-scale (pw × n_strips × 3 × 12). Empty handle on Basic.
    partials_ext_f64: cubecl::server::Handle,
    partials_ext_f64_len: usize,

    /// Reduced masked + IW finals: per-(scale, channel, slot in [0,12)).
    finals_ext_f64: cubecl::server::Handle,
}

impl<R: Runtime> Zensim<R> {
    /// Allocate every per-resolution buffer up front. `width` and
    /// `height` must each be ≥ 8 — zensim's pyramid collapses below
    /// that. Default regime is [`ZensimFeatureRegime::Basic`] (228
    /// features) — backwards-compatible with the pre-372 GPU output.
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        Self::new_with_regime(client, width, height, ZensimFeatureRegime::Basic)
    }

    /// Construct with an explicit feature regime.
    ///
    /// `regime == Extended` adds the 72 masked features (`228..300`).
    /// `regime == WithIw` adds the 72 IW features on top (`300..372`).
    /// See [`ZensimFeatureRegime`] for the slot map.
    ///
    /// **Memory cost**: Extended / WithIw both allocate 4 persist
    /// planes × 3 channels × 2 sides (ref/dis is needed for the IW
    /// activity that uses `src - mu1` so we hold mu1 on the *ref* side
    /// only — see `launch_masked_iw`). The plane footprint is dominated
    /// by scale 0; at 12 MP that's ~600 MB. Use
    /// [`Zensim::new_with_regime_budget`] to fail fast when the budget
    /// is unacceptable.
    pub fn new_with_regime(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        regime: ZensimFeatureRegime,
    ) -> Result<Self> {
        // usize::MAX disables the budget gate; if you care, use
        // `new_with_regime_budget` directly.
        Self::new_with_regime_budget(client, width, height, regime, usize::MAX)
    }

    /// Construct with an explicit feature regime AND an explicit cap on
    /// the extended-regime persist-plane memory footprint (in bytes).
    /// Returns [`Error::ExtendedPlaneBudgetExceeded`] if the regime
    /// requires more than `max_extended_plane_bytes`. The cap is
    /// ignored on `Basic`.
    pub fn new_with_regime_budget(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        regime: ZensimFeatureRegime,
        max_extended_plane_bytes: usize,
    ) -> Result<Self> {
        if width < 8 || height < 8 {
            return Err(Error::InvalidImageSize);
        }
        let pixels = (width as usize) * (height as usize);

        let mut scales = Vec::with_capacity(SCALES);
        let mut logical_w = width;
        let mut padded_w = simd_padded_width(width as usize) as u32;
        let mut h = height;
        let mut plan: Vec<(u32, u32, u32)> = Vec::with_capacity(SCALES);
        for _ in 0..SCALES {
            if logical_w < 8 || h < 8 {
                break;
            }
            plan.push((logical_w, padded_w, h));
            logical_w = logical_w.div_ceil(2);
            padded_w /= 2;
            h = h.div_ceil(2);
        }
        let mut partials_f64_total: usize = 0;
        let mut partials_max_total: usize = 0;
        let mut partials_ext_total: usize = 0;
        for &(_, pw, ph) in &plan {
            let ns = pick_n_strips(pw, ph) as usize;
            let ns_ext = kernels::masked_iw_strip::cpu_strip_count(ph) as usize;
            partials_f64_total += (pw as usize) * ns * 3 * 17;
            partials_max_total += (pw as usize) * ns * 3 * 3;
            partials_ext_total += (pw as usize) * ns_ext * 3 * 12;
        }
        let mut f64_off: usize = 0;
        let mut max_off: usize = 0;
        let mut ext_off: usize = 0;
        for &(lw, pw, ph) in &plan {
            let ns = pick_n_strips(pw, ph) as usize;
            let ns_ext = kernels::masked_iw_strip::cpu_strip_count(ph) as usize;
            scales.push(Scale::new(&client, lw, pw, ph, f64_off, max_off, ext_off));
            f64_off += (pw as usize) * ns * 3 * 17;
            max_off += (pw as usize) * ns * 3 * 3;
            ext_off += (pw as usize) * ns_ext * 3 * 12;
        }

        // Budget check: 4 planes × 3 channels × 2 sides (ref + dis) ×
        // padded_pixels × 4 bytes per scale.
        let needs_planes = regime.needs_extended_kernel();
        let extended_plane_bytes: usize = if needs_planes {
            plan.iter()
                .map(|&(_, pw, ph)| (pw as usize) * (ph as usize) * 3 * 4 * 2 * 4)
                .sum()
        } else {
            0
        };
        if needs_planes && extended_plane_bytes > max_extended_plane_bytes {
            return Err(Error::ExtendedPlaneBudgetExceeded {
                needed_bytes: extended_plane_bytes,
                max_bytes: max_extended_plane_bytes,
            });
        }

        let src_u8_a = client.empty(pixels * core::mem::size_of::<u32>());
        let src_u8_b = client.empty(pixels * core::mem::size_of::<u32>());

        let srgb_lut = client.create_from_slice(f32::as_bytes(
            &crate::kernels::color::SRGB8_TO_LINEARF32_LUT,
        ));

        let partials_f64 = alloc_empty_f64(&client, partials_f64_total);
        let partials_max = alloc_empty_f32(&client, partials_max_total);
        let n_finals_f64 = scales.len() * 3 * 17;
        let n_finals_max = scales.len() * 3 * 3;
        let finals_f64 = alloc_empty_f64(&client, n_finals_f64);
        let finals_max = alloc_empty_f32(&client, n_finals_max);

        // Extended regime allocations — only when needed. The persist
        // planes layout per scale is `[ch0 | ch1 | ch2]` flat, with
        // `pad_total` f32s per channel, one allocation per
        // (scale, side, plane). The masked-IW kernel needs to read
        // ref-side mu1/mu2/ssq/s12, so we currently only fill the
        // ref-side planes (the dist-side is reserved for a future
        // CPU-style symmetric path).
        let mut persist_planes_ref: Vec<[cubecl::server::Handle; 4]> = Vec::new();
        let mut persist_planes_dis: Vec<[cubecl::server::Handle; 4]> = Vec::new();
        let partials_ext_f64: cubecl::server::Handle;
        let finals_ext_f64: cubecl::server::Handle;
        if needs_planes {
            for sc in scales.iter() {
                let plane_len = (sc.padded_w as usize) * (sc.h as usize) * 3;
                let alloc_planes = || -> [cubecl::server::Handle; 4] {
                    [
                        alloc_empty_f32(&client, plane_len),
                        alloc_empty_f32(&client, plane_len),
                        alloc_empty_f32(&client, plane_len),
                        alloc_empty_f32(&client, plane_len),
                    ]
                };
                persist_planes_ref.push(alloc_planes());
                persist_planes_dis.push(alloc_planes());
            }
            partials_ext_f64 = alloc_empty_f64(&client, partials_ext_total);
            finals_ext_f64 = alloc_empty_f64(&client, scales.len() * 3 * 12);
        } else {
            // Basic regime: tiny no-op placeholders. cubecl needs every
            // ArrayArg handle to be valid even if the kernel using it
            // is never launched; placeholders are 1-element rather
            // than 0-element to dodge any backend-side zero-len checks.
            partials_ext_f64 = alloc_empty_f64(&client, 1);
            finals_ext_f64 = alloc_empty_f64(&client, 1);
            // Empty placeholder vecs — code paths that read these check
            // `regime.needs_extended_kernel()` before indexing.
        }

        Ok(Self {
            client,
            width,
            height,
            pixels,
            pack_scratch: vec![0_u32; pixels],
            src_u8_a,
            src_u8_b,
            srgb_lut,
            scales,
            partials_f64,
            partials_max,
            partials_f64_len: partials_f64_total,
            partials_max_len: partials_max_total,
            finals_f64,
            finals_max,
            has_cached_reference: false,
            acumen_band_weights: None,
            acumen_viewing: None,
            acumen_lut_bytes: include_bytes!("../data/castle_csf_v0_5_4_cvvdp.lut"),
            regime,
            persist_planes_ref,
            persist_planes_dis,
            partials_ext_f64,
            partials_ext_f64_len: partials_ext_total.max(1),
            finals_ext_f64,
        })
    }

    /// Which regime this pipeline was constructed for.
    pub fn regime(&self) -> ZensimFeatureRegime {
        self.regime
    }

    /// Debug-only: read back the persist-plane `mu1` at the given scale
    /// and channel. Returns a `Vec<f32>` of length `padded_w * height`
    /// (the plane's stride). `plane_idx` selects between mu1 (0), mu2
    /// (1), ssq (2), s12 (3). Returns an empty vec if the regime
    /// doesn't allocate persist planes.
    pub fn debug_read_persist_plane(
        &self,
        scale: usize,
        channel: usize,
        plane_idx: usize,
    ) -> Vec<f32> {
        if self.persist_planes_ref.is_empty() {
            return Vec::new();
        }
        let s = &self.scales[scale];
        let plane = &self.persist_planes_ref[scale][plane_idx];
        let bytes = self
            .client
            .read_one(plane.clone())
            .expect("read persist plane");
        let all = f32::from_bytes(&bytes);
        // Extract this channel's slice.
        let pt = s.n_padded;
        let ch_start = channel * pt;
        all[ch_start..ch_start + pt].to_vec()
    }

    /// Debug-only: read back the per-scale, per-channel `ref_xyb` (or
    /// `dis_xyb`) plane after `set_reference` / `compute_with_reference`
    /// has been called. Returns a `Vec<f32>` of length `padded_w * height`.
    pub fn debug_read_xyb(&self, scale: usize, channel: usize, ref_side: bool) -> Vec<f32> {
        let s = &self.scales[scale];
        let plane = if ref_side {
            &s.ref_xyb[channel]
        } else {
            &s.dis_xyb[channel]
        };
        let bytes = self.client.read_one(plane.clone()).expect("read xyb plane");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Debug-only: get the padded width and image height for a scale.
    pub fn debug_scale_dims(&self, scale: usize) -> (u32, u32) {
        let s = &self.scales[scale];
        (s.padded_w, s.h)
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }

    /// Compute the 228-feature vector for one (reference, distorted)
    /// pair. **Only valid for `regime == Basic`.** When the pipeline was
    /// constructed with a wider regime, this returns the first 228 slots
    /// (`compute_features_vec` returns the full vector).
    pub fn compute_features(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]> {
        let v = self.compute_features_vec(ref_srgb, dist_srgb)?;
        let mut out = [0.0_f64; TOTAL_FEATURES];
        out.copy_from_slice(&v[..TOTAL_FEATURES]);
        Ok(out)
    }

    /// Compute the regime-appropriate feature vector for one (reference,
    /// distorted) pair. Length matches `self.regime().total_features()`:
    /// 228 / 300 / 372.
    pub fn compute_features_vec(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<Vec<f64>> {
        self.set_reference(ref_srgb)?;
        self.compute_with_reference_vec(dist_srgb)
    }

    /// Cache the reference pyramid; subsequent
    /// [`Zensim::compute_with_reference`] calls reuse it. When an
    /// acumen viewing condition has been configured via
    /// [`Self::set_acumen_viewing`], this call also recomputes the
    /// per-image castleCSF band weights from the reference's mean
    /// luminance.
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.check_dims(ref_srgb)?;
        // Recompute acumen band weights from the reference image if
        // mode A is enabled. Cheap (a single mean-luma pass + 12 LUT
        // lookups, ~µs at 1080p).
        if let Some(viewing) = self.acumen_viewing {
            self.acumen_band_weights =
                Some(Self::compute_acumen_weights(self.acumen_lut_bytes, viewing, ref_srgb));
        }
        self.upload_u8(true, ref_srgb);
        self.run_xyb_pyramid(true);
        self.has_cached_reference = true;
        Ok(())
    }

    /// Configure acumen Mode A: subsequent [`Self::set_reference`]
    /// calls will compute per-image castleCSF weights at this viewing
    /// condition and Phase 4 will apply them to the HF band-energy
    /// features. Pass `None` to disable and return to the legacy
    /// V_22-shipped path.
    pub fn set_acumen_viewing(
        &mut self,
        viewing: Option<zensim::acumen::viewing::ViewingCondition>,
    ) {
        self.acumen_viewing = viewing;
        if viewing.is_none() {
            self.acumen_band_weights = None;
        } else {
            // If a reference is already cached, recompute weights now
            // so the next compute_with_reference call sees them.
            self.acumen_band_weights = None;
            // Caller is expected to re-call set_reference() after
            // enabling acumen so we have ref bytes to compute mean L
            // from. Document this in the rustdoc above.
        }
    }

    /// Compute the per-(channel, scale) castleCSF weights for a given
    /// reference image and viewing condition. Returns weights
    /// normalised so the achromatic peak is 1.0 — keeps absolute
    /// scale on the same order as the legacy `CSF_BAND_WEIGHTS`
    /// prior so downstream MLPs trained with the legacy weights
    /// don't see scale shock on the first eval pass.
    ///
    /// Layout: `[channel][scale]` where channel 0=A, 1=RG, 2=YV and
    /// scale 0=finest (highest rho), N-1=coarsest (lowest rho).
    fn compute_acumen_weights(
        lut_bytes: &[u8],
        viewing: zensim::acumen::viewing::ViewingCondition,
        ref_srgb: &[u8],
    ) -> [[f32; SCALES]; 3] {
        let lut = zensim::acumen::castle_csf::CastleCsfLut::from_bytes(lut_bytes)
            .expect("vendored castleCSF LUT must parse");
        let mean_l = zensim::acumen::band_weights::image_mean_luminance_nits(
            ref_srgb,
            viewing.peak_luminance_nits,
        );
        let bw = zensim::acumen::band_weights::compute_csf_band_weights(&lut, viewing, mean_l)
            .normalized_to_achromatic_peak();
        // Project to [channel][scale] — N_BANDS matches SCALES per
        // the band-rho convention. Both are 4 by design.
        let mut out = [[0.0_f32; SCALES]; 3];
        for ch in 0..3 {
            for s in 0..SCALES {
                out[ch][s] = bw.weights[ch][s];
            }
        }
        out
    }

    pub fn clear_reference(&mut self) {
        self.has_cached_reference = false;
    }

    pub fn has_cached_reference(&self) -> bool {
        self.has_cached_reference
    }

    /// Compute the 228-feature vector for one distorted image against
    /// the cached reference. **Only valid for `regime == Basic`.**
    pub fn compute_with_reference(&mut self, dist_srgb: &[u8]) -> Result<[f64; TOTAL_FEATURES]> {
        let v = self.compute_with_reference_vec(dist_srgb)?;
        let mut out = [0.0_f64; TOTAL_FEATURES];
        out.copy_from_slice(&v[..TOTAL_FEATURES]);
        Ok(out)
    }

    /// Compute the regime-appropriate feature vector for one distorted
    /// image against the cached reference. Returns
    /// [`Error::NoCachedReference`] if [`Zensim::set_reference`] hasn't
    /// been called.
    pub fn compute_with_reference_vec(&mut self, dist_srgb: &[u8]) -> Result<Vec<f64>> {
        if !self.has_cached_reference {
            return Err(Error::NoCachedReference);
        }
        self.check_dims(dist_srgb)?;
        self.upload_u8(false, dist_srgb);
        self.run_xyb_pyramid(false);

        let n_scales = self.scales.len();
        let needs_ext = self.regime.needs_extended_kernel();

        // Phase 1: launch the per-scale fused features kernel. Two
        // variants: persist (writes mu1/mu2/ssq/s12 planes) on the
        // Extended/WithIw regime, plain on Basic. Both produce the
        // same per-column 17 f64 + 3 f32 partials.
        for s in 0..n_scales {
            if needs_ext {
                self.launch_blur_and_features_persist(s);
            } else {
                self.launch_blur_and_features(s);
            }
        }

        // Phase 1b: masked + IW pooling pass. Needs the persist planes
        // from Phase 1.
        if needs_ext {
            for s in 0..n_scales {
                self.launch_masked_iw(s);
            }
        }

        // Phase 2: on-device reduction. Basic partials reduce as before;
        // masked + IW partials reduce via `reduce_ext_kernel`.
        self.launch_reduction();
        if needs_ext {
            self.launch_reduction_ext();
        }

        // Phase 3: ONE read for the basic finals; one more for masked +
        // IW finals when needed. cubecl serialises both reads behind
        // the reduction launches.
        let f64_bytes = self
            .client
            .read_one(self.finals_f64.clone())
            .expect("read finals_f64");
        let max_bytes = self
            .client
            .read_one(self.finals_max.clone())
            .expect("read finals_max");
        let finals_f64 = f64::from_bytes(&f64_bytes);
        let finals_max = f32::from_bytes(&max_bytes);

        let ext_bytes_storage;
        let finals_ext_f64: &[f64] = if needs_ext {
            ext_bytes_storage = self
                .client
                .read_one(self.finals_ext_f64.clone())
                .expect("read finals_ext_f64");
            f64::from_bytes(&ext_bytes_storage)
        } else {
            &[]
        };

        // Phase 4: host packs the regime-appropriate feature vector.
        let total = self.regime.total_features();
        let mut out = vec![0.0_f64; total];
        let basic_total = n_scales * FEATURES_PER_CHANNEL_BASIC * 3;
        let peaks_total = n_scales * FEATURES_PER_CHANNEL_PEAKS * 3;
        let masked_block_off = basic_total + peaks_total;
        let iw_block_off = masked_block_off + n_scales * FEATURES_PER_CHANNEL_MASKED * 3;

        for s in 0..n_scales {
            for ch in 0..3 {
                let final_f64_base = (s * 3 + ch) * 17;
                let final_max_base = (s * 3 + ch) * 3;
                let mut sums = [0.0_f64; 17];
                sums.copy_from_slice(&finals_f64[final_f64_base..final_f64_base + 17]);
                let mut peaks = [0.0_f32; 3];
                peaks.copy_from_slice(&finals_max[final_max_base..final_max_base + 3]);

                let pad_w = self.scales[s].padded_w as usize;
                let h_dim = self.scales[s].h as usize;
                let inv_n = 1.0_f64 / (pad_w as f64 * h_dim as f64);
                let var_src = sums[10] * inv_n;
                let mad_src = sums[12] * inv_n;
                let (hf_energy_loss, hf_energy_gain) = if var_src > 1e-10 {
                    let r = sums[11] / sums[10];
                    ((1.0 - r).max(0.0), (r - 1.0).max(0.0))
                } else {
                    (0.0, 0.0)
                };
                let hf_mag_loss = if mad_src > 1e-10 {
                    (1.0 - sums[13] / sums[12]).max(0.0)
                } else {
                    0.0
                };

                // Basic block: 13 features per channel.
                let bb = s * 3 * FEATURES_PER_CHANNEL_BASIC + ch * FEATURES_PER_CHANNEL_BASIC;
                out[bb] = (sums[0] * inv_n).abs();
                out[bb + 1] = (sums[1] * inv_n).max(0.0).powf(0.25);
                out[bb + 2] = (sums[2] * inv_n).max(0.0).sqrt();
                out[bb + 3] = (sums[3] * inv_n).abs();
                out[bb + 4] = (sums[4] * inv_n).max(0.0).powf(0.25);
                out[bb + 5] = (sums[5] * inv_n).max(0.0).sqrt();
                out[bb + 6] = (sums[6] * inv_n).abs();
                out[bb + 7] = (sums[7] * inv_n).max(0.0).powf(0.25);
                out[bb + 8] = (sums[8] * inv_n).max(0.0).sqrt();
                out[bb + 9] = sums[9] * inv_n;
                // Slots 10/11/12 are HF band-energy losses/gains —
                // the closest analog to the CPU CVVDP-features'
                // CSF-weighted band-ratios. Acumen Mode A weights
                // them by per-(channel, scale) castleCSF
                // sensitivity. When acumen is disabled (default),
                // the multiplier is 1.0 and the V_22-shipped path
                // is bit-stable.
                let acumen_w = self
                    .acumen_band_weights
                    .as_ref()
                    .map(|w| w[ch][s] as f64)
                    .unwrap_or(1.0);
                out[bb + 10] = hf_energy_loss * acumen_w;
                out[bb + 11] = hf_mag_loss * acumen_w;
                out[bb + 12] = hf_energy_gain * acumen_w;

                // Peaks block.
                let pb = basic_total
                    + s * 3 * FEATURES_PER_CHANNEL_PEAKS
                    + ch * FEATURES_PER_CHANNEL_PEAKS;
                out[pb] = peaks[0] as f64;
                out[pb + 1] = peaks[1] as f64;
                out[pb + 2] = peaks[2] as f64;
                out[pb + 3] = (sums[14] * inv_n).max(0.0).powf(0.125);
                out[pb + 4] = (sums[15] * inv_n).max(0.0).powf(0.125);
                out[pb + 5] = (sums[16] * inv_n).max(0.0).powf(0.125);

                // Masked + IW blocks.
                if needs_ext {
                    let ext_base = (s * 3 + ch) * 12;
                    let mut ext_sums = [0.0_f64; 12];
                    ext_sums.copy_from_slice(&finals_ext_f64[ext_base..ext_base + 12]);

                    if self.regime.needs_masked() {
                        // CPU layout: masked_ssim_mean / masked_ssim_4th /
                        // masked_ssim_2nd / masked_art_4th /
                        // masked_det_4th / masked_mse.
                        // GPU slot map: [s0..s5] mirror this order.
                        let mb = masked_block_off
                            + s * 3 * FEATURES_PER_CHANNEL_MASKED
                            + ch * FEATURES_PER_CHANNEL_MASKED;
                        out[mb] = (ext_sums[0] * inv_n).abs();
                        out[mb + 1] = (ext_sums[1] * inv_n).max(0.0).powf(0.25);
                        out[mb + 2] = (ext_sums[2] * inv_n).max(0.0).sqrt();
                        out[mb + 3] = (ext_sums[3] * inv_n).max(0.0).powf(0.25);
                        out[mb + 4] = (ext_sums[4] * inv_n).max(0.0).powf(0.25);
                        out[mb + 5] = ext_sums[5] * inv_n;
                    }
                    if self.regime.needs_iw() {
                        let ib = iw_block_off
                            + s * 3 * FEATURES_PER_CHANNEL_IW
                            + ch * FEATURES_PER_CHANNEL_IW;
                        out[ib] = (ext_sums[6] * inv_n).abs();
                        out[ib + 1] = (ext_sums[7] * inv_n).max(0.0).powf(0.25);
                        out[ib + 2] = (ext_sums[8] * inv_n).max(0.0).sqrt();
                        out[ib + 3] = (ext_sums[9] * inv_n).max(0.0).powf(0.25);
                        out[ib + 4] = (ext_sums[10] * inv_n).max(0.0).powf(0.25);
                        out[ib + 5] = ext_sums[11] * inv_n;
                    }
                }
            }
        }

        Ok(out)
    }

    // ───────────────────────── helpers ─────────────────────────

    fn check_dims(&self, srgb: &[u8]) -> Result<()> {
        let expected = self.pixels * 3;
        if srgb.len() != expected {
            Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            })
        } else {
            Ok(())
        }
    }

    fn cube_count_1d(n: usize) -> CubeCount {
        const TPB: u32 = 256;
        let cubes = (n as u32).div_ceil(TPB);
        CubeCount::Static(cubes.max(1), 1, 1)
    }
    fn cube_dim_1d() -> CubeDim {
        CubeDim::new_1d(256)
    }

    fn upload_u8(&mut self, is_a: bool, srgb: &[u8]) {
        // T4.L (pre-dates this session): pack 3 u8 bytes into one u32
        // per pixel: R | G<<8 | B<<16. Kernel masks the bytes back
        // out; on-device math is unchanged. 3× H2D bandwidth saving
        // vs the older "widen each u8 to its own u32" layout —
        // significant on WSL2 where PCIe is virtualised to ~3 GB/s.
        for (dst, chunk) in self.pack_scratch.iter_mut().zip(srgb.chunks_exact(3)) {
            *dst = (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
        }
        // T4.M (2026-05-16): pinned-host upload via the lilith/cubecl
        // feat/pinned-upload fork — DMAs at 12-25 GB/s on PCIe 4.0 vs
        // 5-6 GB/s pageable. See docs/CUBECL_GOTCHAS.md G6.5.
        let bytes = u32::as_bytes(&self.pack_scratch);
        if is_a {
            self.src_u8_a = self.client.create_from_slice_pinned(bytes);
        } else {
            self.src_u8_b = self.client.create_from_slice_pinned(bytes);
        }
    }

    /// sRGB → positive XYB at scale 0, mirror-fill padding, then
    /// downscale through the pyramid. Operates on either the reference
    /// or distorted side based on `is_a`.
    fn run_xyb_pyramid(&self, is_a: bool) {
        let s0 = &self.scales[0];
        let src = if is_a { &self.src_u8_a } else { &self.src_u8_b };
        let xyb = if is_a { &s0.ref_xyb } else { &s0.dis_xyb };
        // sRGB → XYB at scale 0 (with integrated mirror-pad).
        // `absorbance_bias_neg = -cbrtf_fast(K_B0)` is precomputed
        // host-side using the same `cbrtf_fast` algorithm CPU zensim
        // uses; the kernel takes it as a scalar so the bit-cast inside
        // cbrtf_fast is never asked to operate on a literal.
        let absorbance_bias_neg = -color::cbrtf_fast_host(color::K_B0);
        // The kernel always indexes `mirror_offsets` (so we always
        // bind a non-empty handle); when `pad_count == 0` the kernel
        // never reads it, so we can re-bind any small placeholder.
        // We bind `srgb_lut` itself (always allocated, length 256
        // u32-equivalent bytes) when no mirror is needed — its u32
        // bit pattern doesn't matter because the index path is
        // never taken.
        let mirror_arg = match s0.mirror_offsets.as_ref() {
            Some(mo) => (mo.clone(), s0.pad_count as usize),
            None => (self.srgb_lut.clone(), 1),
        };
        unsafe {
            color::srgb_to_positive_xyb_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s0.n_padded),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), self.pixels),
                ArrayArg::from_raw_parts(self.srgb_lut.clone(), 256),
                ArrayArg::from_raw_parts(mirror_arg.0, mirror_arg.1),
                ArrayArg::from_raw_parts(xyb[0].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[1].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[2].clone(), s0.n_padded),
                self.width,
                self.height,
                s0.padded_w,
                absorbance_bias_neg,
            );
        }
        // Build pyramid via 2× planar downscale, all 3 channels per launch.
        for s in 1..self.scales.len() {
            let prev = &self.scales[s - 1];
            let curr = &self.scales[s];
            let prev_xyb = if is_a { &prev.ref_xyb } else { &prev.dis_xyb };
            let curr_xyb = if is_a { &curr.ref_xyb } else { &curr.dis_xyb };
            unsafe {
                downscale::downscale_2x_3ch_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(curr.n_padded),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(prev_xyb[0].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(prev_xyb[1].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(prev_xyb[2].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[0].clone(), curr.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[1].clone(), curr.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[2].clone(), curr.n_padded),
                    prev.padded_w,
                    prev.h,
                    curr.padded_w,
                    curr.h,
                );
            }
        }
    }

    /// Launch the **tile-fused H-blur + V-blur + features** kernel for
    /// one scale. Grid `(ceil(pw/64), n_strips, 3)`; block dim 64.
    /// One launch per scale (was 2 with the separate H-blur path).
    /// Eliminates the 12 H-blur scratch planes from DRAM — H-blur
    /// outputs live in shared memory across the V-blur slide.
    fn launch_blur_and_features(&self, scale: usize) {
        // Keep in sync with `kernels::fused::TX`.
        const TX: u32 = 64;
        let s = &self.scales[scale];
        let pad_total = s.n_padded;

        let cube_x = s.padded_w.div_ceil(TX).max(1);
        let cube_count = CubeCount::Static(cube_x, s.n_strips, 3);
        let cube_dim = CubeDim::new_3d(TX, 1, 1);
        unsafe {
            fused::fused_features_kernel::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(s.ref_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(self.partials_f64.clone(), self.partials_f64_len),
                ArrayArg::from_raw_parts(self.partials_max.clone(), self.partials_max_len),
                s.padded_w,
                s.h,
                s.n_strips,
                s.partials_f64_off as u32,
                s.partials_max_off as u32,
            );
        }
    }

    /// Persist-planes variant of `launch_blur_and_features`. Same math
    /// + same per-column partials, plus per-pixel writes of mu1/mu2/
    /// ssq/s12 into the appropriate side's persist planes. Run once for
    /// `is_a == true` then once for `false` is NOT what happens — the
    /// existing pipeline already runs the basic kernel once per pair
    /// (combining ref + dist via the (src_a/dst_a) channel arrays).
    /// The persist variant takes the same combined inputs but ALSO
    /// writes the persist planes. We persist to the `ref` side's planes
    /// (the masked-IW kernel needs them for `activity = blur(|src - mu1|)`
    /// which uses the REFERENCE side per CPU).
    fn launch_blur_and_features_persist(&self, scale: usize) {
        const TX: u32 = 64;
        let s = &self.scales[scale];
        let pad_total = s.n_padded;
        let plane_len = pad_total * 3;
        let planes = &self.persist_planes_ref[scale];

        let cube_x = s.padded_w.div_ceil(TX).max(1);
        let cube_count = CubeCount::Static(cube_x, s.n_strips, 3);
        let cube_dim = CubeDim::new_3d(TX, 1, 1);
        unsafe {
            fused::fused_features_kernel_persist::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(s.ref_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(self.partials_f64.clone(), self.partials_f64_len),
                ArrayArg::from_raw_parts(self.partials_max.clone(), self.partials_max_len),
                ArrayArg::from_raw_parts(planes[0].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[1].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[2].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[3].clone(), plane_len),
                s.padded_w,
                s.h,
                s.n_strips,
                s.partials_f64_off as u32,
                s.partials_max_off as u32,
                pad_total as u32,
            );
        }
    }

    /// Launch the masked + IW pooling kernel for one scale. Requires
    /// the persist planes to already be filled by
    /// `launch_blur_and_features_persist`.
    ///
    /// Uses the **strip-local** variant
    /// [`masked_iw_strip::masked_iw_strip_kernel`] which reproduces
    /// CPU's per-strip semantics. Per the 2026-05-17 principled
    /// per-channel H-blur redesign, the activity computation uses
    /// on-the-fly `H_blur(src[channel])` instead of any cross-channel
    /// cascade — see the kernel's docstring and zensim's
    /// `docs/PRINCIPLED_ACTIVITY.md`. No carryover plane needed.
    fn launch_masked_iw(&self, scale: usize) {
        const TX: u32 = 64;

        let s = &self.scales[scale];
        let pad_total = s.n_padded;
        let plane_len = pad_total * 3;
        let planes = &self.persist_planes_ref[scale];

        let cube_x = s.padded_w.div_ceil(TX).max(1);
        // Grid Y dimension matches CPU's strip count (n_strips_ext),
        // NOT the basic kernel's GPU-occupancy strip count.
        let cube_count = CubeCount::Static(cube_x, s.n_strips_ext, 3);
        let cube_dim = CubeDim::new_3d(TX, 1, 1);
        let do_ext = if self.regime.needs_masked() {
            1u32
        } else {
            0u32
        };
        let do_iw = if self.regime.needs_iw() { 1u32 } else { 0u32 };
        unsafe {
            masked_iw_strip::masked_iw_strip_kernel::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(s.ref_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(planes[0].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[1].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[2].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[3].clone(), plane_len),
                ArrayArg::from_raw_parts(self.partials_ext_f64.clone(), self.partials_ext_f64_len),
                s.padded_w,
                s.h,
                s.n_strips_ext,
                pad_total as u32,
                s.partials_ext_off as u32,
                do_ext,
                do_iw,
            );
        }
    }

    /// On-device reduction of the masked + IW per-(col, strip, ch)
    /// partials into per-(scale, ch, slot) finals. One launch per
    /// scale, grid `(36, 1, 1)` (= 3 channels × 12 slots).
    fn launch_reduction_ext(&self) {
        let n_scales = self.scales.len();
        let cube_dim = CubeDim::new_1d(256);
        let n_finals = n_scales * 3 * 12;
        for s in 0..n_scales {
            let sc = &self.scales[s];
            let pw = sc.padded_w as usize;
            // Use the **extended kernel's** strip count, NOT the basic
            // kernel's GPU-occupancy strip count. The strip-local
            // masked + IW kernel writes `pw × n_strips_ext` partials
            // per channel.
            let ns = sc.n_strips_ext as usize;
            let n_partials_per_ch = (pw * ns) as u32;
            let cube_count = CubeCount::Static(36, 1, 1);
            unsafe {
                reduce::reduce_ext_kernel::launch_unchecked::<R>(
                    &self.client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts(
                        self.partials_ext_f64.clone(),
                        self.partials_ext_f64_len,
                    ),
                    ArrayArg::from_raw_parts(self.finals_ext_f64.clone(), n_finals),
                    sc.partials_ext_off as u32,
                    n_partials_per_ch,
                    (s * 3 * 12) as u32,
                );
            }
        }
    }

    /// On-device reduction of per-(col, strip, channel) partials into
    /// per-(scale, channel, slot) finals. One launch per scale (4
    /// total at SCALES = 4); each launch fires a 60-cube grid (3
    /// channels × 20 slot kinds) so the entire pyramid's reduction
    /// costs ~4 launches plus 240 fast cube-level tree reduces.
    fn launch_reduction(&self) {
        let n_scales = self.scales.len();
        let n_finals_f64 = n_scales * 3 * 17;
        let n_finals_max = n_scales * 3 * 3;
        let cube_dim = CubeDim::new_1d(256);
        for s in 0..n_scales {
            let sc = &self.scales[s];
            let pw = sc.padded_w as usize;
            let ns = sc.n_strips as usize;
            let n_partials_per_ch = (pw * ns) as u32;
            let cube_count = CubeCount::Static(60, 1, 1);
            unsafe {
                reduce::reduce_scale_kernel::launch_unchecked::<R>(
                    &self.client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts(self.partials_f64.clone(), self.partials_f64_len),
                    ArrayArg::from_raw_parts(self.partials_max.clone(), self.partials_max_len),
                    ArrayArg::from_raw_parts(self.finals_f64.clone(), n_finals_f64),
                    ArrayArg::from_raw_parts(self.finals_max.clone(), n_finals_max),
                    sc.partials_f64_off as u32,
                    sc.partials_max_off as u32,
                    n_partials_per_ch,
                    (s * 3 * 17) as u32,
                    (s * 3 * 3) as u32,
                );
            }
        }
    }

    /// Host-side fold of one (scale, channel)'s partials. The kernel
    /// laid out per (col, strip, channel) slots; we sum across cols ×
    /// strips for this channel.
    #[allow(dead_code)]
    fn fold_partials(
        &self,
        scale: usize,
        channel: usize,
        parts_all: &[f64],
        maxs_all: &[f32],
    ) -> ([f64; 17], [f32; 3]) {
        let s = &self.scales[scale];
        let pw = s.padded_w as usize;
        let ns = s.n_strips as usize;
        // Slot index: ch × ns × pw + strip × pw + col.
        let f64_ch_base = s.partials_f64_off + channel * ns * pw * 17;
        let max_ch_base = s.partials_max_off + channel * ns * pw * 3;

        let mut sums = [0.0_f64; 17];
        let mut peaks = [0.0_f32; 3];
        for strip in 0..ns {
            let f64_strip_base = f64_ch_base + strip * pw * 17;
            let max_strip_base = max_ch_base + strip * pw * 3;
            for col in 0..pw {
                for i in 0..17 {
                    sums[i] += parts_all[f64_strip_base + col * 17 + i];
                }
                for i in 0..3 {
                    let v = maxs_all[max_strip_base + col * 3 + i];
                    if v > peaks[i] {
                        peaks[i] = v;
                    }
                }
            }
        }
        (sums, peaks)
    }
}
