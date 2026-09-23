//! Streaming video scoring — the pycvvdp v0.5.7 *video* path.
//!
//! Extends the still-image pipeline with a temporal stage:
//!
//! 1. sRGB-8 frame → DKLd65 `(A, RG, VY)` planes per side.
//! 2. Causal temporal FIR over the frame window
//!    (`kernels::temporal`): 3 sustained low-pass channels + a
//!    transient achromatic channel (the sustained-A buffer convolved
//!    with the 5 Hz band-pass kernel — `sw_ch = 0 if cc == 3` in
//!    pycvvdp's `read_block_of_frames`).
//! 3. Per side, per channel: `weber_g1` Weber-contrast pyramid with
//!    the *same side's* filtered sustained-A plane as `L_bkg`
//!    (reproduces upstream's per-side `0::2`/`1::2` division of the
//!    interleaved 8-channel tensor).
//! 4. Per band: sustained `o0` CSF for channels 0..3, transient
//!    `o5_c1` CSF for channel 3, sensitivity always taken at the
//!    *reference* background (`logL_bkg[...,1:2]` upstream).
//! 5. 4-channel `mult-mutual` masking (v0.5.7 `xcm_weights` 4×4,
//!    `mask_q[4]`, `ch_gain=[1,1.45,1,1]`); baseband uses the direct
//!    `|T−R|·S` difference.
//! 6. `do_pooling_and_jod_video_4ch`: band pool (β=4, per-channel
//!    baseband weights), channel pool (β=4, `ch_trans_w` on the
//!    transient channel), normalised frame pool (β=2), `met2jod` —
//!    **no** `image_int` for video.
//!
//! `N_frames == 1` routes to [`crate::Cvvdp::score`] — the same
//! still-image path pycvvdp uses for `is_image`, bit-identical to
//! scoring the frame pair as a still.
//!
//! ## Argument order
//!
//! pycvvdp's `predict` takes the DISTORTED video first; this crate
//! keeps its still-image convention — **`push_frame(ref, dist)` is
//! REFERENCE first**, matching `Cvvdp::score(ref, dist)`.
//!
//! ## Memory
//!
//! The scorer holds at most `filter_len` frames per side (the causal
//! FIR window) plus the `Q_per_ch` accumulator
//! (`frames × bands × 4` f32) and, while the clip could still turn
//! out to be a still image, the first frame's raw bytes. Memory is
//! bounded by the temporal filter length, not the clip length.
//!
//! Padding: `temp_padding="replicate"` only (the pycvvdp default).

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use crate::color::{srgb_planar_to_dkl_planar, srgb_to_dkl_planar};
use crate::csf::compute_sensitivities_into;
use crate::kernels::csf::{
    CSF_BASEBAND_RHO, CsfChannel, precompute_logs_row, precompute_logs_row_o5,
};
use crate::kernels::masking::CH_GAIN_4;
use crate::kernels::pool::{BETA_SPATIAL, do_pooling_and_jod_video_4ch};
use crate::kernels::pyramid::band_frequencies;
use crate::kernels::temporal::{temporal_filter_len, temporal_filters};
use crate::masking::mult_mutual_band_4ch_into;
use crate::params::DisplayGeometry;
use crate::pyramid::{
    Band, WeberPyramid, WeberPyramidCache, build_gauss_pyramid_into, gauss_bands_with_capacity,
    weber_bands_from_gauss,
};
use crate::simd_math::{
    vabs_diff_mul_lp2, vaxpy_into, vaxpy2_into, vmul2_scale2_pair_into, vscale_into, vscale2_into,
};
use crate::{CvvdpParams, Error, Result};

/// One side's DKL planes for one frame: `[A, RG, VY]`.
type FramePlanes = [Vec<f32>; 3];

/// Memory layout of the sRGB-8 frames accepted by [`VideoScorer`] —
/// the Rust analog of pycvvdp's `dim_order` argument, restricted to
/// the two layouts a byte-slice API can express (`"HWC"` /
/// `"CHW"` per frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameLayout {
    /// `width × height × 3` interleaved bytes (`RGBRGB…`) per frame.
    /// This is the layout `Cvvdp::score` and `score_video` use.
    #[default]
    Interleaved,
    /// Three concatenated `width × height` planes per frame: all R
    /// bytes, then all G, then all B. Natural for surfaces decoded
    /// to planar RGB.
    Planar,
}

/// How the temporal FIR resolves frame indices below 0 — pycvvdp's
/// `temp_padding` constructor argument (v0.5.7 supports exactly these
/// two; `"valid"` exists in the docstring but raises at runtime
/// upstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TempPadding {
    /// Frames before index 0 read frame 0 (upstream default).
    #[default]
    Replicate,
    /// Frames before index 0 are mirrored: `frame[-k] = frame[k]`,
    /// ping-ponging when the clip is shorter than the filter
    /// (`_get_symmetric_frame_index` upstream). Output frame `t`
    /// needs input frames up to index `fl−1−t`, so the first `fl−1`
    /// outputs are deferred until enough frames have been pushed —
    /// `push_frame` may emit zero or several rows per call. The frame
    /// ring stays bounded by `fl` regardless.
    Symmetric,
}

/// Construction knobs for [`VideoScorer`] beyond the pycvvdp surface
/// — bundled so [`VideoScorer::with_options`] stays readable instead
/// of growing a fourth positional-argument variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VideoScorerOptions {
    /// Frame byte layout `push_frame` expects (pycvvdp `dim_order`
    /// analog).
    pub layout: FrameLayout,
    /// Temporal padding (pycvvdp `temp_padding` analog).
    pub temp_padding: TempPadding,
    /// Store the temporal-filter window as raw sRGB-8 bytes instead of
    /// f32 DKL planes — a 4× smaller ring (at 1080p/30 fps ~450 MB →
    /// ~113 MB), paid for by re-running the sRGB→DKL conversion per
    /// window slot per emitted frame (~+10–20 % CPU depending on the
    /// clip). Lossless: the conversion is a deterministic LUT+matrix
    /// of the stored bytes, so scores are **bit-identical** to the
    /// f32-window path. No upstream analog — an implementation knob,
    /// not a scoring semantic.
    pub low_memory: bool,
}

/// Positive frame index for a negative `fi` under
/// [`TempPadding::Symmetric`] — verbatim port of pycvvdp
/// `_get_symmetric_frame_index` (`frame[-1] → frame[1]`, ping-pong
/// for `|fi| ≥ frame_count`).
///
/// `frame_count` must be the TOTAL clip length (upstream passes
/// `N_frames`). Callers only invoke this when the resolution is
/// already determined: during streaming, emission gating guarantees
/// `|fi| ≤ frames_pushed − 1` (pure mirror, `frame_count`-
/// independent); the deferred finish-time drain runs with
/// `frame_count == N`.
fn symmetric_frame_index(fi: isize, frame_count: usize) -> usize {
    debug_assert!(fi < 0);
    debug_assert!(frame_count >= 2);
    let fc = frame_count as isize;
    let a = fi.unsigned_abs() as isize;
    let m = fc - 1;
    // floor((a−1)/m) is exact on non-negative operands.
    let is_even = ((a - 1) / m) % 2 == 0;
    if is_even {
        (((a - 1) % m) + 1) as usize
    } else {
        fi.rem_euclid(m) as usize
    }
}

/// Result bundle for a scored clip — the Rust analog of the
/// `(Q_jod, stats)` pair pycvvdp's `predict`/`predict_video_source`
/// returns. `stats` carries `Q_per_ch`, `rho_band`,
/// `frames_per_second`, `width`, `height`, `N_frames`; pycvvdp's
/// optional `heatmap` entry is not ported (see `docs/VIDEO.md`).
#[derive(Debug, Clone)]
pub struct VideoStats {
    /// Final pooled quality in JOD — identical to
    /// [`VideoScorer::finish`]'s return value.
    pub jod: f32,
    /// Per-frame, per-band, per-channel spatially-pooled masked
    /// differences, layout `[frame][band][channel]`, channel order
    /// sustained A, RG, VY, transient A; last band = baseband. For a
    /// single-frame (still-path) score the transient channel entry
    /// is `f32::NAN` — upstream image mode has no transient channel.
    pub q_per_ch: Vec<Vec<[f32; 4]>>,
    /// Spatial frequency of each pyramid band in cycles/degree
    /// (`rho_band` upstream). Length = number of pyramid bands.
    pub rho_band: Vec<f32>,
    /// Frame rate the clip was scored at.
    pub frames_per_second: f32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Number of frames scored.
    pub n_frames: usize,
}

impl VideoStats {
    /// pycvvdp's `loss()` — `10 - JOD`, a minimisable distortion
    /// objective.
    #[must_use]
    pub fn loss(&self) -> f32 {
        10.0 - self.jod
    }
}

/// Per-frame reusable scratch — allocated once in
/// [`VideoScorer::new`] and grown lazily to the largest band, so
/// `push_frame` is allocation-free in steady state. Mirrors the
/// still path's `Scratch` approach.
struct VideoScratch {
    /// Filtered test/ref planes `[sust-A, RG, VY, trans-A]` for the
    /// frame currently being emitted.
    filt_t: [Vec<f32>; 4],
    filt_r: [Vec<f32>; 4],
    /// Shared sustained-A Gaussian pyramid per side — every channel
    /// divides by the same `L_bkg` (`filt_*[0]`), so upstream's
    /// per-channel `gauss_l` builds are four identical reductions;
    /// channel 0's image pyramid is the same input again. One build
    /// per side replaces 8.
    gauss_l_t: Vec<Band>,
    gauss_l_r: Vec<Band>,
    /// Pyramid caches (gauss_img planes + filter scratch) per channel
    /// per side — reused across frames. Channel 0 builds no
    /// `gauss_img` (it reuses the shared pyramid); no cache builds a
    /// `gauss_l` (the shared pyramids above replace all 8).
    cache_t: [WeberPyramidCache; 4],
    cache_r: [WeberPyramidCache; 4],
    /// Output pyramids per channel per side. Only `pyr_r[0]` keeps
    /// `log_l_bkg` planes — the only ones read downstream (CSF
    /// sensitivity is evaluated at the reference sustained-A
    /// background).
    pyr_t: [WeberPyramid; 4],
    pyr_r: [WeberPyramid; 4],
    /// CSF-weighted contrasts for the current band — consumed in
    /// place by `mult_mutual_band_4ch_into` (the clamped-diff planes
    /// are never materialised).
    t_p: [Vec<f32>; 4],
    r_p: [Vec<f32>; 4],
    /// Per-pixel sensitivity maps for the current band — doubles as
    /// the masking `m_mm` scratch (the maps are dead once `t_p`/`r_p`
    /// are computed).
    m_mm: [Vec<f32>; 4],
    /// Masking intermediates (`safe_pow(|M_mm|, q)`).
    term: [Vec<f32>; 4],
    /// PU-blur horizontal-pass scratch.
    pu_scratch: Vec<f32>,
    /// `low_memory` emit scratch — the current window slot's DKL
    /// planes. One slot converts at a time (test side's taps, then
    /// reference side's), so three planes suffice for both.
    win_dkl: FramePlanes,
}

impl VideoScratch {
    fn new(w: usize, h: usize, n_levels: usize, low_memory: bool) -> Self {
        // Channel 0's image pyramid IS the shared sustained-A pyramid
        // (same input plane) — its cache never builds `gauss_img`;
        // no cache ever builds `gauss_l`.
        let video_cache = |img: bool| WeberPyramidCache {
            gauss_img: if img {
                gauss_bands_with_capacity(w, h, n_levels)
            } else {
                Vec::new()
            },
            gauss_l: Vec::new(),
            scratch: crate::pyramid::PyramidScratch::default(),
        };
        Self {
            filt_t: core::array::from_fn(|_| vec![0.0; w * h]),
            filt_r: core::array::from_fn(|_| vec![0.0; w * h]),
            gauss_l_t: gauss_bands_with_capacity(w, h, n_levels),
            gauss_l_r: gauss_bands_with_capacity(w, h, n_levels),
            cache_t: core::array::from_fn(|c| video_cache(c != 0)),
            cache_r: core::array::from_fn(|c| video_cache(c != 0)),
            pyr_t: core::array::from_fn(|_| WeberPyramid::with_capacity_nolog(w, h, n_levels)),
            pyr_r: core::array::from_fn(|c| {
                if c == 0 {
                    WeberPyramid::with_capacity(w, h, n_levels)
                } else {
                    WeberPyramid::with_capacity_nolog(w, h, n_levels)
                }
            }),
            t_p: core::array::from_fn(|_| Vec::new()),
            r_p: core::array::from_fn(|_| Vec::new()),
            m_mm: core::array::from_fn(|_| Vec::new()),
            term: core::array::from_fn(|_| Vec::new()),
            pu_scratch: Vec::new(),
            win_dkl: if low_memory {
                core::array::from_fn(|_| vec![0.0; w * h])
            } else {
                core::array::from_fn(|_| Vec::new())
            },
        }
    }
}

/// Streaming cvvdp video scorer (sRGB-8 frames in, JOD out).
///
/// Construct once per clip with [`VideoScorer::new`] (or
/// [`crate::Cvvdp::video`]), feed frames with
/// [`push_frame`](Self::push_frame), then [`finish`](Self::finish).
/// See `docs/VIDEO.md` for the porting notes.
///
/// # Examples
///
/// ```
/// use cvvdp::{CvvdpParams, DisplayGeometry, VideoScorer};
///
/// let mut v = VideoScorer::new(
///     64,
///     64,
///     30.0,
///     CvvdpParams::default(),
///     DisplayGeometry::STANDARD_4K,
/// )?;
/// let frame = vec![128u8; 64 * 64 * 3];
/// v.push_frame(&frame, &frame)?;
/// v.push_frame(&frame, &frame)?;
/// let jod = v.finish()?;
/// assert!((jod - 10.0).abs() < 1e-3);
/// # Ok::<(), cvvdp::Error>(())
/// ```
pub struct VideoScorer {
    width: usize,
    height: usize,
    /// Full parameter bundle — `params.display` drives the sRGB→DKL
    /// conversion; retained so a 1-frame clip can construct a
    /// [`crate::Cvvdp`] and route through the still path verbatim.
    params: CvvdpParams,
    /// Display geometry — retained for the same 1-frame routing.
    geometry: DisplayGeometry,
    /// `[channel][tap]` causal FIR taps; `taps[c][j]` weights the
    /// frame `j` positions back. Channel 3 (transient) is applied to
    /// the sustained-A plane.
    taps: [Vec<f32>; 4],
    /// Frames pushed so far.
    n_pushed: usize,
    /// Sliding window of the last `taps[0].len()` frames (DKL planes),
    /// test side. Oldest at front. Empty when `low_memory` is on —
    /// `win8_t` holds the raw sRGB bytes instead.
    win_t: VecDeque<FramePlanes>,
    /// Same for the reference side.
    win_r: VecDeque<FramePlanes>,
    /// `low_memory` window — raw sRGB bytes in `layout` order (4×
    /// smaller than DKL planes), re-converted at emit.
    win8_t: VecDeque<Vec<u8>>,
    /// Same for the reference side.
    win8_r: VecDeque<Vec<u8>>,
    /// Whether the window stores u8 sRGB (`win8_*`) or f32 DKL
    /// (`win_*`) — [`VideoScorerOptions::low_memory`].
    low_memory: bool,
    /// Raw bytes of frame 0, kept only until a second frame is pushed
    /// so a one-frame clip can take the still path untouched.
    first_frame: Option<(Vec<u8>, Vec<u8>)>,
    /// `q_per_ch[frame][band][channel]` — spatially-pooled masked
    /// differences, accumulated per emitted output frame.
    q_per_ch: Vec<Vec<[f32; 4]>>,
    /// Per-band spatial frequencies (cy/deg); last entry's band uses
    /// `CSF_BASEBAND_RHO` instead.
    freqs: Vec<f32>,
    /// Byte layout `push_frame` expects (`dim_order` analog).
    layout: FrameLayout,
    /// Temporal padding mode (`temp_padding` analog).
    padding: TempPadding,
    /// Next output frame index to emit — [`TempPadding::Symmetric`]
    /// defers the first `fl−1` outputs until their lookahead frames
    /// have been pushed; replicate emits on every push.
    next_emit: usize,
    /// Evicted ring slots' plane buffers, kept for reuse by the next
    /// `push_frame` — avoids a fresh 3-plane alloc + zero-fill per
    /// frame (the planes are `w*h` f32, the same size every frame).
    spare_planes: Vec<FramePlanes>,
    /// Same recycling for evicted `low_memory` u8 buffers.
    spare8: Vec<Vec<u8>>,
    /// Frame rate the clip is scored at (kept for `VideoStats`).
    fps: f32,
    /// Reusable per-frame scratch (filtered planes, pyramid caches,
    /// band buffers) — allocated in `new`, so `push_frame` does no
    /// large allocations in steady state.
    scratch: VideoScratch,
}

impl VideoScorer {
    /// Create a video scorer for `width × height` sRGB-8 clips at
    /// `frames_per_second` Hz. `params.display` drives the
    /// sRGB→DKLd65 conversion; `geometry` sets pixels-per-degree.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidImageSize`] if `min(width, height) < 8`.
    /// - [`Error::InvalidFps`] if `frames_per_second` is not finite
    ///   and > 0.
    pub fn new(
        width: u32,
        height: u32,
        frames_per_second: f32,
        params: CvvdpParams,
        geometry: DisplayGeometry,
    ) -> Result<Self> {
        Self::with_layout(
            width,
            height,
            frames_per_second,
            params,
            geometry,
            FrameLayout::Interleaved,
        )
    }

    /// [`new`](Self::new) with an explicit frame byte layout — the
    /// analog of pycvvdp's `dim_order` argument on `predict`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidImageSize`] if `min(width, height) < 8`.
    /// - [`Error::InvalidFps`] if `frames_per_second` is not finite
    ///   and > 0.
    pub fn with_layout(
        width: u32,
        height: u32,
        frames_per_second: f32,
        params: CvvdpParams,
        geometry: DisplayGeometry,
        layout: FrameLayout,
    ) -> Result<Self> {
        Self::with_layout_and_padding(
            width,
            height,
            frames_per_second,
            params,
            geometry,
            layout,
            TempPadding::Replicate,
        )
    }

    /// [`with_layout`](Self::with_layout) with an explicit
    /// [`TempPadding`] — the analog of pycvvdp's `temp_padding`
    /// constructor argument. [`TempPadding::Replicate`] matches
    /// upstream's default; [`TempPadding::Symmetric`] mirrors frames
    /// before index 0 (`frame[-k] → frame[k]`, ping-pong for clips
    /// shorter than the filter).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    #[allow(clippy::too_many_arguments)]
    pub fn with_layout_and_padding(
        width: u32,
        height: u32,
        frames_per_second: f32,
        params: CvvdpParams,
        geometry: DisplayGeometry,
        layout: FrameLayout,
        padding: TempPadding,
    ) -> Result<Self> {
        Self::with_options(
            width,
            height,
            frames_per_second,
            params,
            geometry,
            VideoScorerOptions {
                layout,
                temp_padding: padding,
                low_memory: false,
            },
        )
    }

    /// [`with_layout_and_padding`](Self::with_layout_and_padding)
    /// with the full [`VideoScorerOptions`] bundle — currently the
    /// only added knob is `low_memory` (u8 window).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn with_options(
        width: u32,
        height: u32,
        frames_per_second: f32,
        params: CvvdpParams,
        geometry: DisplayGeometry,
        options: VideoScorerOptions,
    ) -> Result<Self> {
        if width < 8 || height < 8 {
            return Err(Error::InvalidImageSize { width, height });
        }
        if !frames_per_second.is_finite() || frames_per_second <= 0.0 {
            return Err(Error::InvalidFps);
        }
        let w = width as usize;
        let h = height as usize;
        let ppd = geometry.pixels_per_degree();
        let freqs = band_frequencies(ppd, w, h);
        let n_levels = freqs.len();
        let layout = options.layout;
        let padding = options.temp_padding;
        let low_memory = options.low_memory;
        Ok(Self {
            width: w,
            height: h,
            params,
            geometry,
            taps: temporal_filters(frames_per_second),
            n_pushed: 0,
            win_t: VecDeque::new(),
            win_r: VecDeque::new(),
            win8_t: VecDeque::new(),
            win8_r: VecDeque::new(),
            low_memory,
            first_frame: None,
            q_per_ch: Vec::new(),
            freqs,
            layout,
            padding,
            next_emit: 0,
            spare_planes: Vec::new(),
            spare8: Vec::new(),
            fps: frames_per_second,
            scratch: VideoScratch::new(w, h, n_levels, low_memory),
        })
    }

    /// Push one `(reference, distorted)` sRGB-8 frame pair.
    /// **Reference first** — matching `Cvvdp::score(ref, dist)`,
    /// opposite of pycvvdp's `predict(test, reference)`.
    ///
    /// Each frame must be `width × height × 3` bytes in the
    /// [`FrameLayout`] the scorer was built with.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] on a wrong-size frame.
    pub fn push_frame(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<()> {
        let n_bytes = self.width * self.height * 3;
        if ref_srgb.len() != n_bytes || dist_srgb.len() != n_bytes {
            return Err(Error::DimensionMismatch {
                expected: n_bytes,
                got: ref_srgb.len().max(dist_srgb.len()),
            });
        }

        if self.n_pushed == 0 {
            // Might still be a 1-frame clip → still path; keep the
            // raw bytes so `finish` can score them exactly like
            // `predict_jod_still_3ch` (no video processing at all).
            self.first_frame = Some((ref_srgb.to_vec(), dist_srgb.to_vec()));
        }

        let fl = self.taps[0].len();
        if self.low_memory {
            // u8 window — keep the raw sRGB bytes; the sRGB→DKL
            // conversion runs at emit time instead of now.
            let mut t8 = self.spare8.pop().unwrap_or_default();
            let mut r8 = self.spare8.pop().unwrap_or_default();
            t8.clear();
            t8.extend_from_slice(dist_srgb);
            r8.clear();
            r8.extend_from_slice(ref_srgb);
            self.win8_t.push_back(t8);
            self.win8_r.push_back(r8);
            if self.win8_t.len() > fl {
                if let Some(old) = self.win8_t.pop_front() {
                    self.spare8.push(old);
                }
                if let Some(old) = self.win8_r.pop_front() {
                    self.spare8.push(old);
                }
            }
        } else {
            let mut t = self.spare_planes.pop().unwrap_or_default();
            let mut r = self.spare_planes.pop().unwrap_or_default();
            let display = self.params.display;
            let (w, h) = (self.width, self.height);
            let [r0, r1, r2] = &mut r;
            let [t0, t1, t2] = &mut t;
            match self.layout {
                FrameLayout::Interleaved => {
                    srgb_to_dkl_planar(ref_srgb, w, h, display, r0, r1, r2);
                    srgb_to_dkl_planar(dist_srgb, w, h, display, t0, t1, t2);
                }
                FrameLayout::Planar => {
                    srgb_planar_to_dkl_planar(ref_srgb, w, h, display, r0, r1, r2);
                    srgb_planar_to_dkl_planar(dist_srgb, w, h, display, t0, t1, t2);
                }
            }
            self.win_t.push_back(t);
            self.win_r.push_back(r);
            if self.win_t.len() > fl {
                if let Some(old) = self.win_t.pop_front() {
                    self.spare_planes.push(old);
                }
                if let Some(old) = self.win_r.pop_front() {
                    self.spare_planes.push(old);
                }
            }
        }

        self.n_pushed += 1;
        if self.n_pushed == 2 {
            // Confirmed a real video — release the still-path bytes.
            self.first_frame = None;
        }
        if self.n_pushed < 2 {
            // One frame could still be a still — emit nothing yet.
            return Ok(());
        }
        let fl = self.taps[0].len();
        while self.next_emit < self.n_pushed {
            let t = self.next_emit;
            // Symmetric taps reach fl−1−t frames ahead of t; replicate
            // reaches only backwards (emit as soon as t is pushed).
            let need = match self.padding {
                TempPadding::Replicate => t + 1,
                TempPadding::Symmetric => (t + 1).max(fl.saturating_sub(t)),
            };
            if self.n_pushed < need {
                break;
            }
            self.emit_output_frame(t);
            self.next_emit += 1;
        }
        Ok(())
    }

    /// Number of frames pushed so far.
    #[must_use]
    pub fn frames_pushed(&self) -> usize {
        self.n_pushed
    }

    /// The accumulated `Q_per_ch` table — pycvvdp's per-frame
    /// spatially-pooled masked differences, layout
    /// `[frame][band][channel]` (channel order: sustained A, RG, VY,
    /// transient A; last band = baseband). Empty until the second
    /// frame is pushed (a single-frame clip takes the still path and
    /// never populates this).
    ///
    /// Diagnostic surface for conformance tests; the pooling that
    /// turns this into JOD is [`do_pooling_and_jod_video_4ch`].
    #[doc(hidden)]
    #[must_use]
    pub fn q_per_ch_table(&self) -> &[Vec<[f32; 4]>] {
        &self.q_per_ch
    }

    /// Per-band spatial frequencies in cycles/degree — pycvvdp's
    /// `stats['rho_band']`. Length = number of pyramid bands.
    #[must_use]
    pub fn band_frequencies(&self) -> &[f32] {
        &self.freqs
    }

    /// Finish the clip and return the JOD score.
    ///
    /// A one-frame clip is scored by the still-image path
    /// (`predict_jod_still_3ch`), bit-identical to
    /// [`crate::Cvvdp::score`].
    ///
    /// # Errors
    ///
    /// [`Error::NoFrames`] if no frames were pushed.
    pub fn finish(self) -> Result<f32> {
        Ok(self.finish_with_stats()?.jod)
    }

    /// Finish the clip and return the JOD plus the stats bundle —
    /// the analog of pycvvdp `predict`'s `(Q_jod, stats)` return.
    ///
    /// For a one-frame clip `jod` comes from [`crate::Cvvdp::score`]
    /// (bit-identical); `q_per_ch` is the scalar reference path's
    /// per-band table (a diagnostic — it may differ from the strip
    /// pipeline's internals by ~1 ulp).
    ///
    /// # Errors
    ///
    /// [`Error::NoFrames`] if no frames were pushed.
    pub fn finish_with_stats(mut self) -> Result<VideoStats> {
        if self.n_pushed == 0 {
            return Err(Error::NoFrames);
        }
        if self.n_pushed == 1 {
            let (r, d) = self.first_frame.expect("frame 0 bytes retained");
            // Still path expects interleaved bytes — shuffle planar
            // input back to interleaved first.
            let (r_i, d_i) = match self.layout {
                FrameLayout::Interleaved => (r, d),
                FrameLayout::Planar => (planar_to_interleaved(&r), planar_to_interleaved(&d)),
            };
            let ppd = self.geometry.pixels_per_degree();
            let (q3, freqs) = crate::host_scalar::still_3ch_q_per_ch(
                &r_i,
                &d_i,
                self.width,
                self.height,
                self.params.display,
                ppd,
                None,
            );
            // Route through the public still path so a 1-frame clip
            // is bit-identical to `Cvvdp::score` on the same pair
            // (pycvvdp does the same: is_image skips temporal
            // filtering entirely).
            let mut still = crate::Cvvdp::with_geometry(
                self.width as u32,
                self.height as u32,
                self.params,
                self.geometry,
            )?;
            let jod = still.score(&r_i, &d_i)?;
            return Ok(VideoStats {
                jod,
                q_per_ch: vec![
                    q3.iter()
                        .map(|&[a, rg, vy]| [a, rg, vy, f32::NAN])
                        .collect(),
                ],
                rho_band: freqs,
                frames_per_second: self.fps,
                width: self.width as u32,
                height: self.height as u32,
                n_frames: 1,
            });
        }
        // Symmetric padding can leave outputs pending at end-of-clip
        // (clips shorter than the filter never reach the lookahead
        // threshold mid-stream — the ring holds all `N < fl` frames,
        // so nothing was dropped).
        while self.next_emit < self.n_pushed {
            let t = self.next_emit;
            self.emit_output_frame(t);
            self.next_emit += 1;
        }
        let jod = do_pooling_and_jod_video_4ch(&self.q_per_ch);
        Ok(VideoStats {
            jod,
            q_per_ch: self.q_per_ch,
            rho_band: self.freqs,
            frames_per_second: self.fps,
            width: self.width as u32,
            height: self.height as u32,
            n_frames: self.n_pushed,
        })
    }

    /// Absolute index of `win_*[0]` — frames before it have scrolled
    /// out of the ring buffer.
    fn win_base(&self) -> usize {
        let len = if self.low_memory {
            self.win8_t.len()
        } else {
            self.win_t.len()
        };
        self.n_pushed - len
    }

    /// Compute output frame `t`'s filtered planes + per-band pooled
    /// differences and append to `q_per_ch`. Output `t` reads frames
    /// `t−fl+1 ..= t` (replicate-extended below 0), all already in the
    /// window.
    fn emit_output_frame(&mut self, t: usize) {
        // `vlp_norm_mean_p2` is specialised to p=2 — BETA_SPATIAL is
        // the only exponent the video path uses.
        debug_assert_eq!(BETA_SPATIAL, 2.0);
        let fl = self.taps[0].len();
        let w0 = self.win_base();
        let (w, h) = (self.width, self.height);
        let n_levels = self.freqs.len();
        let sc = &mut self.scratch;

        // FIR per channel: out[c] = Σ_j taps[c][j] · in[src_c][t−j].
        // Tap-major loop in window-slot order k = 0..fl (oldest →
        // newest) — each output element sees the identical sequence
        // of adds as the reference scalar loop; `vaxpy_into` keeps the
        // non-fused mul+add order per element. The first tap is an
        // overwrite (`vscale_into`) instead of a `fill(0)` + add —
        // identical result (`0 + a·b == a·b`, only a −0/+0 sign flip
        // which is numerically identical downstream), and it removes
        // the 8 full-plane memsets that used to precede the loop.
        {
            // Channels 0 and 3 both FIR-filter the sustained-A plane
            // (different taps) — dual-accumulate so the plane is read
            // once per tap instead of twice.
            let [ft0, ft1, ft2, ft3] = &mut sc.filt_t;
            let [fr0, fr1, fr2, fr3] = &mut sc.filt_r;
            // `win_*[widx][c]` for the current tap k — a plain slice
            // read in the f32 path; in `low_memory` mode the u8 slot
            // is converted into `sc.win_dkl` first (deterministic
            // LUT+matrix → identical values, identical score).
            macro_rules! fir_tap {
                ($k:expr) => {{
                    let s = t as isize - (fl as isize - 1) + $k as isize;
                    let fi = if s >= 0 {
                        s as usize
                    } else {
                        match self.padding {
                            TempPadding::Replicate => 0,
                            TempPadding::Symmetric => {
                                symmetric_frame_index(s, self.n_pushed.max(2))
                            }
                        }
                    };
                    fi - w0
                }};
            }
            if self.low_memory {
                let display = self.params.display;
                let [d0, d1, d2] = &mut sc.win_dkl;
                // Test side, then reference — each side's accumulation
                // order is the same k-ascending sequence as the f32
                // path, so `filt_*` come out bit-identical.
                for k in 0..fl {
                    let widx = fir_tap!(k);
                    let src = &self.win8_t[widx];
                    match self.layout {
                        FrameLayout::Interleaved => {
                            srgb_to_dkl_planar(src, w, h, display, d0, d1, d2)
                        }
                        FrameLayout::Planar => {
                            srgb_planar_to_dkl_planar(src, w, h, display, d0, d1, d2)
                        }
                    }
                    let j = fl - 1 - k;
                    if k == 0 {
                        vscale2_into(ft0, ft3, d0, self.taps[0][j], self.taps[3][j]);
                        vscale_into(ft1, d1, self.taps[1][j]);
                        vscale_into(ft2, d2, self.taps[2][j]);
                    } else {
                        vaxpy2_into(ft0, ft3, d0, self.taps[0][j], self.taps[3][j]);
                        vaxpy_into(ft1, d1, self.taps[1][j]);
                        vaxpy_into(ft2, d2, self.taps[2][j]);
                    }
                }
                for k in 0..fl {
                    let widx = fir_tap!(k);
                    let src = &self.win8_r[widx];
                    match self.layout {
                        FrameLayout::Interleaved => {
                            srgb_to_dkl_planar(src, w, h, display, d0, d1, d2)
                        }
                        FrameLayout::Planar => {
                            srgb_planar_to_dkl_planar(src, w, h, display, d0, d1, d2)
                        }
                    }
                    let j = fl - 1 - k;
                    if k == 0 {
                        vscale2_into(fr0, fr3, d0, self.taps[0][j], self.taps[3][j]);
                        vscale_into(fr1, d1, self.taps[1][j]);
                        vscale_into(fr2, d2, self.taps[2][j]);
                    } else {
                        vaxpy2_into(fr0, fr3, d0, self.taps[0][j], self.taps[3][j]);
                        vaxpy_into(fr1, d1, self.taps[1][j]);
                        vaxpy_into(fr2, d2, self.taps[2][j]);
                    }
                }
            } else {
                for k in 0..fl {
                    // Frame index at window slot k. s<0 resolves per
                    // `temp_padding`: replicate → frame 0 (always win[0]
                    // when s≤0); symmetric → mirrored/ping-pong index.
                    // The emission gate guarantees every resolved index
                    // is inside the ring.
                    let widx = fir_tap!(k);
                    let j = fl - 1 - k;
                    if k == 0 {
                        vscale2_into(
                            ft0,
                            ft3,
                            &self.win_t[widx][0],
                            self.taps[0][j],
                            self.taps[3][j],
                        );
                        vscale2_into(
                            fr0,
                            fr3,
                            &self.win_r[widx][0],
                            self.taps[0][j],
                            self.taps[3][j],
                        );
                        vscale_into(ft1, &self.win_t[widx][1], self.taps[1][j]);
                        vscale_into(ft2, &self.win_t[widx][2], self.taps[2][j]);
                        vscale_into(fr1, &self.win_r[widx][1], self.taps[1][j]);
                        vscale_into(fr2, &self.win_r[widx][2], self.taps[2][j]);
                    } else {
                        vaxpy2_into(
                            ft0,
                            ft3,
                            &self.win_t[widx][0],
                            self.taps[0][j],
                            self.taps[3][j],
                        );
                        vaxpy2_into(
                            fr0,
                            fr3,
                            &self.win_r[widx][0],
                            self.taps[0][j],
                            self.taps[3][j],
                        );
                        vaxpy_into(ft1, &self.win_t[widx][1], self.taps[1][j]);
                        vaxpy_into(ft2, &self.win_t[widx][2], self.taps[2][j]);
                        vaxpy_into(fr1, &self.win_r[widx][1], self.taps[1][j]);
                        vaxpy_into(fr2, &self.win_r[widx][2], self.taps[2][j]);
                    }
                }
            }
        }

        // Shared sustained-A Gaussian pyramids — every channel's
        // l_bkg is the side's own filt[0], so upstream's per-channel
        // gauss_l builds are four identical reductions; channel 0's
        // gauss_img is the same input again. One build per side
        // replaces 8 (identical math — same input, same kernel).
        build_gauss_pyramid_into(
            &sc.filt_t[0],
            w,
            h,
            n_levels,
            &mut sc.cache_t[0].scratch,
            &mut sc.gauss_l_t,
        );
        build_gauss_pyramid_into(
            &sc.filt_r[0],
            w,
            h,
            n_levels,
            &mut sc.cache_r[0].scratch,
            &mut sc.gauss_l_r,
        );

        // Per-side weber band construction via the SIMD/scratch path
        // (`weber_bands_from_gauss` — same math as
        // `weber_contrast_pyr_dec_scalar` to ~1e-5 FMA-order noise).
        // Channel 0 consumes the shared pyramid as both gauss_img and
        // gauss_l; channels 1–3 build only their own gauss_img against
        // the shared l. Only the reference achromatic pyramid writes
        // `log_l_bkg` — the only one read downstream. The 8 band
        // stages are fully independent — each owns a disjoint
        // cache+output slot — so under `parallel` they run on rayon's
        // pool.
        let gl_t: &[Band] = &sc.gauss_l_t;
        let gl_r: &[Band] = &sc.gauss_l_r;
        #[cfg(feature = "parallel")]
        {
            rayon::scope(|s| {
                for (c, ((cache_t, pyr_t), (cache_r, pyr_r))) in sc
                    .cache_t
                    .iter_mut()
                    .zip(sc.pyr_t.iter_mut())
                    .zip(sc.cache_r.iter_mut().zip(sc.pyr_r.iter_mut()))
                    .enumerate()
                {
                    let ft = &sc.filt_t[c];
                    let fr = &sc.filt_r[c];
                    s.spawn(move |_| {
                        if c == 0 {
                            weber_bands_from_gauss(gl_t, gl_t, &mut cache_t.scratch, pyr_t, false);
                        } else {
                            build_gauss_pyramid_into(
                                ft,
                                w,
                                h,
                                n_levels,
                                &mut cache_t.scratch,
                                &mut cache_t.gauss_img,
                            );
                            weber_bands_from_gauss(
                                &cache_t.gauss_img,
                                gl_t,
                                &mut cache_t.scratch,
                                pyr_t,
                                false,
                            );
                        }
                    });
                    s.spawn(move |_| {
                        if c == 0 {
                            weber_bands_from_gauss(gl_r, gl_r, &mut cache_r.scratch, pyr_r, true);
                        } else {
                            build_gauss_pyramid_into(
                                fr,
                                w,
                                h,
                                n_levels,
                                &mut cache_r.scratch,
                                &mut cache_r.gauss_img,
                            );
                            weber_bands_from_gauss(
                                &cache_r.gauss_img,
                                gl_r,
                                &mut cache_r.scratch,
                                pyr_r,
                                false,
                            );
                        }
                    });
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for c in 0..4 {
                if c == 0 {
                    weber_bands_from_gauss(
                        gl_t,
                        gl_t,
                        &mut sc.cache_t[0].scratch,
                        &mut sc.pyr_t[0],
                        false,
                    );
                    weber_bands_from_gauss(
                        gl_r,
                        gl_r,
                        &mut sc.cache_r[0].scratch,
                        &mut sc.pyr_r[0],
                        true,
                    );
                } else {
                    build_gauss_pyramid_into(
                        &sc.filt_t[c],
                        w,
                        h,
                        n_levels,
                        &mut sc.cache_t[c].scratch,
                        &mut sc.cache_t[c].gauss_img,
                    );
                    weber_bands_from_gauss(
                        &sc.cache_t[c].gauss_img,
                        gl_t,
                        &mut sc.cache_t[c].scratch,
                        &mut sc.pyr_t[c],
                        false,
                    );
                    build_gauss_pyramid_into(
                        &sc.filt_r[c],
                        w,
                        h,
                        n_levels,
                        &mut sc.cache_r[c].scratch,
                        &mut sc.cache_r[c].gauss_img,
                    );
                    weber_bands_from_gauss(
                        &sc.cache_r[c].gauss_img,
                        gl_r,
                        &mut sc.cache_r[c].scratch,
                        &mut sc.pyr_r[c],
                        false,
                    );
                }
            }
        }

        let mut q_frame: Vec<[f32; 4]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let is_first = k == 0;
            let is_baseband = k == n_levels - 1;
            let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };

            let bw = sc.pyr_t[0].bands[k].w;
            let bh = sc.pyr_t[0].bands[k].h;
            let n_px_b = bw * bh;
            let rho = if is_baseband {
                CSF_BASEBAND_RHO
            } else {
                self.freqs[k]
            };
            // Sensitivity is always evaluated at the REFERENCE
            // sustained-A background (`logL_bkg[...,1:2]` upstream).
            // Four per-pixel maps: o0_c1/c2/c3 sustained + o5_c1
            // transient — vectorised `compute_sensitivities_into`
            // (exp of LUT-interp + folded correction).
            let rows = [
                precompute_logs_row(rho, CsfChannel::A),
                precompute_logs_row(rho, CsfChannel::Rg),
                precompute_logs_row(rho, CsfChannel::Vy),
                precompute_logs_row_o5(rho),
            ];
            // Sensitivity maps land in `m_mm` — the masking scratch
            // isn't live until after `t_p`/`r_p` are computed, so the
            // two share storage (saves 4 full-band planes).
            for c in 0..4 {
                compute_sensitivities_into(&sc.pyr_r[0].log_l_bkg[k], &rows[c], &mut sc.m_mm[c]);
            }

            if is_baseband {
                // D = |T_f − R_f| · S pooled directly — the fused
                // kernel never materialises the diff plane.
                let mut q_band = [0.0_f32; 4];
                for c in 0..4 {
                    q_band[c] = vabs_diff_mul_lp2(
                        &sc.pyr_t[c].bands[k].data,
                        &sc.pyr_r[c].bands[k].data,
                        &sc.m_mm[c][..n_px_b],
                    );
                }
                q_frame.push(q_band);
            } else {
                for c in 0..4 {
                    let gain = CH_GAIN_4[c];
                    let t_data = &sc.pyr_t[c].bands[k].data;
                    let r_data = &sc.pyr_r[c].bands[k].data;
                    let s = &sc.m_mm[c][..n_px_b];
                    let tp = &mut sc.t_p[c];
                    let rp = &mut sc.r_p[c];
                    if tp.len() < n_px_b {
                        tp.resize(n_px_b, 0.0);
                    }
                    if rp.len() < n_px_b {
                        rp.resize(n_px_b, 0.0);
                    }
                    // `((x·a)·y)·b` — the scalar `band_mul * t * s *
                    // gain` op order, both sides in one pass.
                    vmul2_scale2_pair_into(
                        &mut tp[..n_px_b],
                        &mut rp[..n_px_b],
                        t_data,
                        r_data,
                        s,
                        band_mul,
                        gain,
                    );
                }
                let q_band = mult_mutual_band_4ch_into(
                    &mut sc.t_p,
                    &sc.r_p,
                    bw,
                    bh,
                    &mut sc.m_mm,
                    &mut sc.term,
                    &mut sc.pu_scratch,
                );
                q_frame.push(q_band);
            }
        }
        self.q_per_ch.push(q_frame);
    }
}

/// Score a whole clip in one call — the streaming path internally,
/// so the result is bit-identical to pushing the same frames through
/// [`VideoScorer`].
///
/// `ref_frames` / `dist_frames` are equal-length sequences of
/// `width × height × 3` sRGB-8 frames.
///
/// # Errors
///
/// - [`Error::InvalidImageSize`], [`Error::InvalidFps`],
///   [`Error::DimensionMismatch`] — as in [`VideoScorer`].
/// - [`Error::NoFrames`] if either sequence is empty; also if the two
///   sequences differ in length the longer tail is treated as a
///   mismatch (`DimensionMismatch` is not raised — the shorter
///   sequence's length governs).
///
/// # Examples
///
/// ```
/// use cvvdp::{CvvdpParams, DisplayGeometry, score_video};
///
/// let frames = vec![vec![128u8; 64 * 64 * 3]; 4];
/// let jod = score_video(
///     &frames,
///     &frames,
///     64,
///     64,
///     30.0,
///     CvvdpParams::default(),
///     DisplayGeometry::STANDARD_4K,
/// )?;
/// assert!((jod - 10.0).abs() < 1e-3);
/// # Ok::<(), cvvdp::Error>(())
/// ```
pub fn score_video<F: AsRef<[u8]>>(
    ref_frames: &[F],
    dist_frames: &[F],
    width: u32,
    height: u32,
    frames_per_second: f32,
    params: CvvdpParams,
    geometry: DisplayGeometry,
) -> Result<f32> {
    let mut v = VideoScorer::new(width, height, frames_per_second, params, geometry)?;
    let n = ref_frames.len().min(dist_frames.len());
    for i in 0..n {
        v.push_frame(ref_frames[i].as_ref(), dist_frames[i].as_ref())?;
    }
    v.finish()
}

/// [`score_video`] with an explicit frame [`FrameLayout`] and
/// [`TempPadding`] — returns the JOD plus [`VideoStats`] (`q_per_ch`,
/// `rho_band`, dimensions). This is the full analog of pycvvdp's
/// `predict(test, reference, dim_order, frames_per_second)` →
/// `(Q_jod, stats)`; call [`VideoStats::loss`] for `10 − JOD`.
///
/// # Errors
///
/// As [`score_video`].
///
/// # Examples
///
/// ```
/// use cvvdp::{CvvdpParams, DisplayGeometry, FrameLayout, TempPadding, score_video_with_stats};
///
/// let frames = vec![vec![128u8; 64 * 64 * 3]; 4];
/// let stats = score_video_with_stats(
///     &frames,
///     &frames,
///     64,
///     64,
///     30.0,
///     CvvdpParams::default(),
///     DisplayGeometry::STANDARD_4K,
///     FrameLayout::Interleaved,
///     TempPadding::Replicate,
/// )?;
/// assert!((stats.jod - 10.0).abs() < 1e-3);
/// assert_eq!(stats.n_frames, 4);
/// assert_eq!(stats.q_per_ch.len(), 4);
/// # Ok::<(), cvvdp::Error>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn score_video_with_stats<F: AsRef<[u8]>>(
    ref_frames: &[F],
    dist_frames: &[F],
    width: u32,
    height: u32,
    frames_per_second: f32,
    params: CvvdpParams,
    geometry: DisplayGeometry,
    layout: FrameLayout,
    padding: TempPadding,
) -> Result<VideoStats> {
    let mut v = VideoScorer::with_layout_and_padding(
        width,
        height,
        frames_per_second,
        params,
        geometry,
        layout,
        padding,
    )?;
    let n = ref_frames.len().min(dist_frames.len());
    for i in 0..n {
        v.push_frame(ref_frames[i].as_ref(), dist_frames[i].as_ref())?;
    }
    v.finish_with_stats()
}

/// `R‖G‖B` planar bytes → `RGBRGB…` interleaved (single frame).
fn planar_to_interleaved(planar: &[u8]) -> Vec<u8> {
    let n = planar.len() / 3;
    let mut out = Vec::with_capacity(planar.len());
    for i in 0..n {
        out.push(planar[i]);
        out.push(planar[n + i]);
        out.push(planar[2 * n + i]);
    }
    out
}

/// Temporal filter length at `fps` — exposed for capacity planning
/// (the ring window held by [`VideoScorer`] is this many frames per
/// side).
#[must_use]
pub fn video_filter_len(frames_per_second: f32) -> usize {
    temporal_filter_len(frames_per_second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::DisplayGeometry;

    fn synth_frame(w: usize, h: usize, t: usize, gain: f32) -> Vec<u8> {
        let mut f = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 3 + y * 5 + t * 7) % 200 + 30) as f32 * gain;
                let i = (y * w + x) * 3;
                let b = (v.clamp(0.0, 255.0)) as u8;
                f[i] = b;
                f[i + 1] = b;
                f[i + 2] = (b as i32 + (x as i32 % 17) - 8).clamp(0, 255) as u8;
            }
        }
        f
    }

    fn synth_clip(w: usize, h: usize, n: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let refs: Vec<Vec<u8>> = (0..n).map(|t| synth_frame(w, h, t, 1.0)).collect();
        // Distorted: moving edge flicker — deterministic.
        let dists: Vec<Vec<u8>> = (0..n)
            .map(|t| synth_frame(w, h, t, 1.0 + 0.15 * ((t % 3) as f32 - 1.0)))
            .collect();
        (refs, dists)
    }

    #[test]
    fn streaming_equals_whole_clip_bit_for_bit() {
        let (w, h) = (64usize, 64usize);
        let (refs, dists) = synth_clip(w, h, 12);
        let params = CvvdpParams::default();
        let geo = DisplayGeometry::STANDARD_4K;

        let whole = score_video(&refs, &dists, w as u32, h as u32, 30.0, params, geo).unwrap();

        let mut v = VideoScorer::new(w as u32, h as u32, 30.0, params, geo).unwrap();
        for (rf, df) in refs.iter().zip(dists.iter()) {
            v.push_frame(rf, df).unwrap();
        }
        let streamed = v.finish().unwrap();

        assert_eq!(whole.to_bits(), streamed.to_bits());
    }

    #[test]
    fn low_memory_matches_f32_window_bit_for_bit() {
        let (w, h) = (64usize, 64usize);
        // 16 frames > fl at 30 fps (9) so the ring wraps and slots
        // get recycled; symmetric adds the mirrored-index path.
        let (refs, dists) = synth_clip(w, h, 16);
        let params = CvvdpParams::default();
        let geo = DisplayGeometry::STANDARD_4K;

        for padding in [TempPadding::Replicate, TempPadding::Symmetric] {
            let mut hi = VideoScorer::with_layout_and_padding(
                w as u32,
                h as u32,
                30.0,
                params,
                geo,
                FrameLayout::Interleaved,
                padding,
            )
            .unwrap();
            let mut lo = VideoScorer::with_options(
                w as u32,
                h as u32,
                30.0,
                params,
                geo,
                VideoScorerOptions {
                    layout: FrameLayout::Interleaved,
                    temp_padding: padding,
                    low_memory: true,
                },
            )
            .unwrap();
            for (rf, df) in refs.iter().zip(dists.iter()) {
                hi.push_frame(rf, df).unwrap();
                lo.push_frame(rf, df).unwrap();
            }
            let hi_stats = hi.finish_with_stats().unwrap();
            let lo_stats = lo.finish_with_stats().unwrap();
            assert_eq!(
                hi_stats.jod.to_bits(),
                lo_stats.jod.to_bits(),
                "low_memory changed the score ({padding:?})"
            );
            for (band_hi, band_lo) in hi_stats.q_per_ch.iter().zip(lo_stats.q_per_ch.iter()) {
                for (ch_hi, ch_lo) in band_hi.iter().zip(band_lo.iter()) {
                    for (v_hi, v_lo) in ch_hi.iter().zip(ch_lo.iter()) {
                        assert_eq!(v_hi.to_bits(), v_lo.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn window_is_bounded_by_filter_length() {
        let (w, h) = (32usize, 32usize);
        let fl = video_filter_len(60.0); // 17
        let (refs, dists) = synth_clip(w, h, 2 * fl + 5);

        let mut v = VideoScorer::new(
            w as u32,
            h as u32,
            60.0,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
        )
        .unwrap();
        for (i, (rf, df)) in refs.iter().zip(dists.iter()).enumerate() {
            v.push_frame(rf, df).unwrap();
            assert!(
                v.win_t.len() <= fl && v.win_r.len() <= fl,
                "after push {i}: window {} > filter len {fl}",
                v.win_t.len()
            );
        }
        assert_eq!(v.win_t.len(), fl, "window should saturate at fl={fl}");
        assert!(v.finish().unwrap().is_finite());
    }

    #[test]
    fn single_frame_routes_to_still_path_bit_identical() {
        let (w, h) = (48usize, 40usize);
        let r = synth_frame(w, h, 0, 1.0);
        let d = synth_frame(w, h, 0, 1.2);

        let mut v = VideoScorer::new(
            w as u32,
            h as u32,
            30.0,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
        )
        .unwrap();
        v.push_frame(&r, &d).unwrap();
        let jod_video = v.finish().unwrap();

        // Bit-identical to `Cvvdp::score` on the same pair — the
        // "existing image path" the work order pins. (The host_scalar
        // `predict_jod_still_3ch` reference can differ from
        // `Cvvdp::score` by ~1 ULP — they are distinct implementations
        // — so the contract is against the public still API.)
        let mut still = crate::Cvvdp::with_geometry(
            w as u32,
            h as u32,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
        )
        .unwrap();
        let jod_cvvdp = still.score(&r, &d).unwrap();
        assert_eq!(jod_video.to_bits(), jod_cvvdp.to_bits());
    }

    #[test]
    fn identical_clip_scores_jod_10() {
        let (w, h) = (64usize, 64usize);
        let frames: Vec<Vec<u8>> = (0..8).map(|t| synth_frame(w, h, t, 1.0)).collect();
        let jod = score_video(
            &frames,
            &frames,
            w as u32,
            h as u32,
            30.0,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
        )
        .unwrap();
        assert!((jod - 10.0).abs() < 1e-3, "identical clip: {jod}");
    }

    #[test]
    fn empty_and_bad_inputs_error() {
        let geo = DisplayGeometry::STANDARD_4K;
        let params = CvvdpParams::default();

        // No frames.
        let v = VideoScorer::new(64, 64, 30.0, params, geo).unwrap();
        assert!(matches!(v.finish(), Err(Error::NoFrames)));

        // Bad fps.
        assert!(matches!(
            VideoScorer::new(64, 64, 0.0, params, geo),
            Err(Error::InvalidFps)
        ));
        assert!(matches!(
            VideoScorer::new(64, 64, f32::NAN, params, geo),
            Err(Error::InvalidFps)
        ));
        assert!(matches!(
            VideoScorer::new(64, 64, -30.0, params, geo),
            Err(Error::InvalidFps)
        ));

        // Bad frame size.
        let mut v = VideoScorer::new(64, 64, 30.0, params, geo).unwrap();
        assert!(matches!(
            v.push_frame(&[0u8; 10], &[0u8; 10]),
            Err(Error::DimensionMismatch { .. })
        ));

        // Too-small image.
        assert!(matches!(
            VideoScorer::new(4, 64, 30.0, params, geo),
            Err(Error::InvalidImageSize { .. })
        ));
    }

    fn interleaved_to_planar(frame: &[u8]) -> Vec<u8> {
        let n = frame.len() / 3;
        let mut out = vec![0u8; frame.len()];
        for i in 0..n {
            out[i] = frame[3 * i];
            out[n + i] = frame[3 * i + 1];
            out[2 * n + i] = frame[3 * i + 2];
        }
        out
    }

    #[test]
    fn planar_layout_matches_interleaved() {
        let (w, h) = (64usize, 64usize);
        let (refs, dists) = synth_clip(w, h, 6);
        let params = CvvdpParams::default();
        let geo = DisplayGeometry::STANDARD_4K;

        let jod_i = score_video(&refs, &dists, w as u32, h as u32, 30.0, params, geo).unwrap();

        let refs_p: Vec<Vec<u8>> = refs.iter().map(|f| interleaved_to_planar(f)).collect();
        let dists_p: Vec<Vec<u8>> = dists.iter().map(|f| interleaved_to_planar(f)).collect();
        let stats_p = score_video_with_stats(
            &refs_p,
            &dists_p,
            w as u32,
            h as u32,
            30.0,
            params,
            geo,
            FrameLayout::Planar,
            TempPadding::Replicate,
        )
        .unwrap();

        // Same per-pixel DKL values → bit-identical downstream math.
        assert_eq!(jod_i.to_bits(), stats_p.jod.to_bits());
    }

    #[test]
    fn stats_bundle_reports_pycvvdp_fields() {
        let (w, h) = (64usize, 64usize);
        let (refs, dists) = synth_clip(w, h, 4);
        let params = CvvdpParams::default();
        let geo = DisplayGeometry::STANDARD_4K;

        let mut v = VideoScorer::new(w as u32, h as u32, 30.0, params, geo).unwrap();
        for (rf, df) in refs.iter().zip(dists.iter()) {
            v.push_frame(rf, df).unwrap();
        }
        let n_bands = v.band_frequencies().len();
        let stats = v.finish_with_stats().unwrap();

        assert!(stats.jod.is_finite());
        assert_eq!(stats.n_frames, 4);
        assert_eq!(stats.width, w as u32);
        assert_eq!(stats.height, h as u32);
        assert_eq!(stats.frames_per_second, 30.0);
        assert_eq!(stats.q_per_ch.len(), 4);
        assert!(stats.q_per_ch.iter().all(|row| row.len() == n_bands));
        assert_eq!(stats.rho_band.len(), n_bands);
        assert_eq!(stats.loss(), 10.0 - stats.jod);
    }

    #[test]
    fn single_frame_stats_mark_transient_nan() {
        let (w, h) = (48usize, 40usize);
        let r = synth_frame(w, h, 0, 1.0);
        let d = synth_frame(w, h, 0, 1.2);

        let mut v = VideoScorer::new(
            w as u32,
            h as u32,
            30.0,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
        )
        .unwrap();
        v.push_frame(&r, &d).unwrap();
        let stats = v.finish_with_stats().unwrap();

        assert_eq!(stats.n_frames, 1);
        assert_eq!(stats.q_per_ch.len(), 1);
        // Still-path table: 3 real channels + NaN transient slot.
        for band in &stats.q_per_ch[0] {
            assert!(band[0].is_finite() && band[1].is_finite() && band[2].is_finite());
            assert!(band[3].is_nan());
        }
        // JOD remains bit-identical to `Cvvdp::score`.
        let mut still = crate::Cvvdp::with_geometry(
            w as u32,
            h as u32,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
        )
        .unwrap();
        assert_eq!(stats.jod.to_bits(), still.score(&r, &d).unwrap().to_bits());
    }

    #[test]
    fn planar_single_frame_scores_like_interleaved() {
        let (w, h) = (48usize, 40usize);
        let r = synth_frame(w, h, 0, 1.0);
        let d = synth_frame(w, h, 0, 1.2);
        let rp = interleaved_to_planar(&r);
        let dp = interleaved_to_planar(&d);
        let params = CvvdpParams::default();
        let geo = DisplayGeometry::STANDARD_4K;

        let jod_i = score_video(&[r], &[d], w as u32, h as u32, 30.0, params, geo).unwrap();
        let jod_p = score_video_with_stats(
            &[rp],
            &[dp],
            w as u32,
            h as u32,
            30.0,
            params,
            geo,
            FrameLayout::Planar,
            TempPadding::Replicate,
        )
        .unwrap()
        .jod;
        assert_eq!(jod_i.to_bits(), jod_p.to_bits());
    }

    /// `symmetric_frame_index` vs the table generated from pycvvdp
    /// v0.5.7 `_get_symmetric_frame_index(fi, fc)` for fi in −16..0.
    #[test]
    fn symmetric_index_matches_upstream_table() {
        let cases: &[(usize, [usize; 16])] = &[
            (5, [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]),
            (8, [2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1]),
            (9, [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]),
            (10, [2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
            (20, [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
        ];
        for &(fc, ref want) in cases {
            for (i, &w) in want.iter().enumerate() {
                let fi = (i as isize) - 16;
                assert_eq!(symmetric_frame_index(fi, fc), w, "fc={fc} fi={fi}");
            }
        }
    }

    /// On a static clip every padded index reads an identical frame,
    /// so padding mode cannot change the result — bit-identical.
    #[test]
    fn symmetric_equals_replicate_on_static_clip() {
        let (w, h) = (64usize, 64usize);
        let frames: Vec<Vec<u8>> = (0..8).map(|_| synth_frame(w, h, 0, 1.0)).collect();
        let params = CvvdpParams::default();
        let geo = DisplayGeometry::STANDARD_4K;

        let rep = score_video(&frames, &frames, w as u32, h as u32, 30.0, params, geo).unwrap();
        let sym = score_video_with_stats(
            &frames,
            &frames,
            w as u32,
            h as u32,
            30.0,
            params,
            geo,
            FrameLayout::Interleaved,
            TempPadding::Symmetric,
        )
        .unwrap()
        .jod;
        assert_eq!(rep.to_bits(), sym.to_bits());
    }

    /// Symmetric must emit exactly N output rows even when N < fl
    /// (ping-pong path) — and score differently from replicate on a
    /// genuinely moving clip.
    #[test]
    fn symmetric_emits_all_frames_and_moves_the_score() {
        let (w, h) = (48usize, 40usize);
        let (refs, dists) = synth_clip(w, h, 5); // 5 < fl(30fps)=9 → ping-pong
        let params = CvvdpParams::default();
        let geo = DisplayGeometry::STANDARD_4K;

        let stats_sym = score_video_with_stats(
            &refs,
            &dists,
            w as u32,
            h as u32,
            30.0,
            params,
            geo,
            FrameLayout::Interleaved,
            TempPadding::Symmetric,
        )
        .unwrap();
        assert_eq!(stats_sym.q_per_ch.len(), 5, "all 5 outputs emitted");
        assert_eq!(stats_sym.n_frames, 5);

        let stats_rep = score_video_with_stats(
            &refs,
            &dists,
            w as u32,
            h as u32,
            30.0,
            params,
            geo,
            FrameLayout::Interleaved,
            TempPadding::Replicate,
        )
        .unwrap();
        // Padding changes the filtered planes on a moving clip —
        // the JODs differ (not asserting direction, just that the
        // mode is live).
        assert_ne!(stats_sym.jod.to_bits(), stats_rep.jod.to_bits());
    }

    /// Streaming emission under symmetric defers the first `fl−1`
    /// outputs until the lookahead exists — ring stays ≤ fl.
    #[test]
    fn symmetric_defers_then_catches_up() {
        let (w, h) = (32usize, 32usize);
        let fl = video_filter_len(30.0); // 9
        let (refs, dists) = synth_clip(w, h, fl + 4);

        let mut v = VideoScorer::with_layout_and_padding(
            w as u32,
            h as u32,
            30.0,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
            FrameLayout::Interleaved,
            TempPadding::Symmetric,
        )
        .unwrap();
        for (i, (rf, df)) in refs.iter().zip(dists.iter()).enumerate() {
            v.push_frame(rf, df).unwrap();
            assert!(
                v.q_per_ch.len() <= i + 1,
                "emitted {} rows after {} pushes",
                v.q_per_ch.len(),
                i + 1
            );
            assert!(v.win_t.len() <= fl, "ring bounded by fl");
        }
        // After push fl−1 (0-based idx fl−1, n_pushed=fl) the backlog
        // drains: all fl outputs emitted.
        let stats = v.finish_with_stats().unwrap();
        assert_eq!(stats.q_per_ch.len(), fl + 4);
    }

    #[test]
    fn bad_planar_length_errors() {
        let mut v = VideoScorer::with_layout(
            64,
            64,
            30.0,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
            FrameLayout::Planar,
        )
        .unwrap();
        assert!(matches!(
            v.push_frame(&[0u8; 100], &[0u8; 100]),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    /// Replicate padding: the first output frame must be the pure
    /// frame-0 response (all taps see frame 0), so a clip that is
    /// constant for the first `fl` frames produces identical Q at t=0
    /// as the all-static clip — verified indirectly via JOD on a
    /// freeze-then-change clip vs pycvvdp goldens; here we pin the
    /// window mechanics directly.
    #[test]
    fn replicate_padding_uses_frame_zero() {
        let (w, h) = (32usize, 32usize);
        let fl = video_filter_len(30.0);
        // Two clips identical for frame 0 but diverging after: the
        // *first* output frame must be identical for both (it only
        // sees frame 0 replicated).
        let (ra, da) = synth_clip(w, h, fl + 2);
        let mut rb = ra.clone();
        let mut db = da.clone();
        for k in 1..rb.len() {
            rb[k] = synth_frame(w, h, k + 100, 1.0);
            db[k] = synth_frame(w, h, k + 100, 1.1);
        }

        let mut va = VideoScorer::new(
            w as u32,
            h as u32,
            30.0,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
        )
        .unwrap();
        let mut vb = VideoScorer::new(
            w as u32,
            h as u32,
            30.0,
            CvvdpParams::default(),
            DisplayGeometry::STANDARD_4K,
        )
        .unwrap();
        va.push_frame(&ra[0], &da[0]).unwrap();
        vb.push_frame(&rb[0], &db[0]).unwrap();
        va.push_frame(&ra[1], &da[1]).unwrap();
        vb.push_frame(&rb[1], &db[1]).unwrap();

        let qa = va.q_per_ch_table()[0].clone();
        let qb = vb.q_per_ch_table()[0].clone();
        for (ba, bb) in qa.iter().zip(qb.iter()) {
            for c in 0..4 {
                assert_eq!(
                    ba[c].to_bits(),
                    bb[c].to_bits(),
                    "frame-0 Q must be identical under replicate padding"
                );
            }
        }
    }
}
