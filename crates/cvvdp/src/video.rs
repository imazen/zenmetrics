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

use crate::color::srgb_to_dkl_planar;
use crate::csf::compute_sensitivities_into;
use crate::kernels::csf::{
    CSF_BASEBAND_RHO, CsfChannel, precompute_logs_row, precompute_logs_row_o5,
};
use crate::kernels::masking::CH_GAIN_4;
use crate::kernels::pool::{BETA_SPATIAL, do_pooling_and_jod_video_4ch, lp_norm_mean};
use crate::kernels::pyramid::band_frequencies;
use crate::kernels::temporal::{temporal_filter_len, temporal_filters};
use crate::masking::mult_mutual_band_4ch_into;
use crate::params::DisplayGeometry;
use crate::pyramid::{WeberPyramid, WeberPyramidCache, weber_contrast_pyr_into};
use crate::{CvvdpParams, Error, Result};

/// One side's DKL planes for one frame: `[A, RG, VY]`.
type FramePlanes = [Vec<f32>; 3];

/// Per-frame reusable scratch — allocated once in
/// [`VideoScorer::new`] and grown lazily to the largest band, so
/// `push_frame` is allocation-free in steady state. Mirrors the
/// still path's `Scratch` approach.
struct VideoScratch {
    /// Filtered test/ref planes `[sust-A, RG, VY, trans-A]` for the
    /// frame currently being emitted.
    filt_t: [Vec<f32>; 4],
    filt_r: [Vec<f32>; 4],
    /// Pyramid caches (gauss planes + filter scratch) per channel
    /// per side — reused across frames.
    cache_t: [WeberPyramidCache; 4],
    cache_r: [WeberPyramidCache; 4],
    /// Output pyramids per channel per side.
    pyr_t: [WeberPyramid; 4],
    pyr_r: [WeberPyramid; 4],
    /// CSF-weighted contrasts for the current band.
    t_p: [Vec<f32>; 4],
    r_p: [Vec<f32>; 4],
    /// Per-pixel sensitivity maps for the current band.
    s_map: [Vec<f32>; 4],
    /// Masked diffs / baseband diffs for the current band.
    d: [Vec<f32>; 4],
    /// Masking intermediates (`min(|T|,|R|)` → blurred → pow input,
    /// then reused for `|T−R|`).
    m_mm: [Vec<f32>; 4],
    /// Masking intermediates (`safe_pow(|M_mm|, q)`).
    term: [Vec<f32>; 4],
    /// PU-blur horizontal-pass scratch.
    pu_scratch: Vec<f32>,
}

impl VideoScratch {
    fn new(w: usize, h: usize, n_levels: usize) -> Self {
        Self {
            filt_t: core::array::from_fn(|_| vec![0.0; w * h]),
            filt_r: core::array::from_fn(|_| vec![0.0; w * h]),
            cache_t: core::array::from_fn(|_| WeberPyramidCache::with_capacity(w, h, n_levels)),
            cache_r: core::array::from_fn(|_| WeberPyramidCache::with_capacity(w, h, n_levels)),
            pyr_t: core::array::from_fn(|_| WeberPyramid::with_capacity(w, h, n_levels)),
            pyr_r: core::array::from_fn(|_| WeberPyramid::with_capacity(w, h, n_levels)),
            t_p: core::array::from_fn(|_| Vec::new()),
            r_p: core::array::from_fn(|_| Vec::new()),
            s_map: core::array::from_fn(|_| Vec::new()),
            d: core::array::from_fn(|_| Vec::new()),
            m_mm: core::array::from_fn(|_| Vec::new()),
            term: core::array::from_fn(|_| Vec::new()),
            pu_scratch: Vec::new(),
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
    /// test side. Oldest at front.
    win_t: VecDeque<FramePlanes>,
    /// Same for the reference side.
    win_r: VecDeque<FramePlanes>,
    /// Raw bytes of frame 0, kept only until a second frame is pushed
    /// so a one-frame clip can take the still path untouched.
    first_frame: Option<(Vec<u8>, Vec<u8>)>,
    /// `q_per_ch[frame][band][channel]` — spatially-pooled masked
    /// differences, accumulated per emitted output frame.
    q_per_ch: Vec<Vec<[f32; 4]>>,
    /// Per-band spatial frequencies (cy/deg); last entry's band uses
    /// `CSF_BASEBAND_RHO` instead.
    freqs: Vec<f32>,
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
        Ok(Self {
            width: w,
            height: h,
            params,
            geometry,
            taps: temporal_filters(frames_per_second),
            n_pushed: 0,
            win_t: VecDeque::new(),
            win_r: VecDeque::new(),
            first_frame: None,
            q_per_ch: Vec::new(),
            freqs,
            scratch: VideoScratch::new(w, h, n_levels),
        })
    }

    /// Push one `(reference, distorted)` sRGB-8 frame pair.
    /// **Reference first** — matching `Cvvdp::score(ref, dist)`,
    /// opposite of pycvvdp's `predict(test, reference)`.
    ///
    /// Each frame must be `width × height × 3` bytes row-major.
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

        let mut t: FramePlanes = [Vec::new(), Vec::new(), Vec::new()];
        let mut r: FramePlanes = [Vec::new(), Vec::new(), Vec::new()];
        let display = self.params.display;
        let [r0, r1, r2] = &mut r;
        srgb_to_dkl_planar(ref_srgb, self.width, self.height, display, r0, r1, r2);
        let [t0, t1, t2] = &mut t;
        srgb_to_dkl_planar(dist_srgb, self.width, self.height, display, t0, t1, t2);
        let fl = self.taps[0].len();
        self.win_t.push_back(t);
        self.win_r.push_back(r);
        if self.win_t.len() > fl {
            self.win_t.pop_front();
            self.win_r.pop_front();
        }

        self.n_pushed += 1;
        if self.n_pushed == 2 {
            // Confirmed a real video — release the still-path bytes
            // and emit output frame 0 (deferred until now).
            self.first_frame = None;
            self.emit_output_frame(0);
        }
        if self.n_pushed >= 2 {
            self.emit_output_frame(self.n_pushed - 1);
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
        if self.n_pushed == 0 {
            return Err(Error::NoFrames);
        }
        if self.n_pushed == 1 {
            let (r, d) = self.first_frame.expect("frame 0 bytes retained");
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
            return still.score(&r, &d);
        }
        Ok(do_pooling_and_jod_video_4ch(&self.q_per_ch))
    }

    /// Absolute index of `win_*[0]` — frames before it have scrolled
    /// out of the ring buffer.
    fn win_base(&self) -> usize {
        self.n_pushed - self.win_t.len()
    }

    /// Compute output frame `t`'s filtered planes + per-band pooled
    /// differences and append to `q_per_ch`. Output `t` reads frames
    /// `t−fl+1 ..= t` (replicate-extended below 0), all already in the
    /// window.
    fn emit_output_frame(&mut self, t: usize) {
        let fl = self.taps[0].len();
        let w0 = self.win_base();
        let (w, h) = (self.width, self.height);
        let n_levels = self.freqs.len();
        let sc = &mut self.scratch;

        // FIR per channel: out[c] = Σ_j taps[c][j] · in[src_c][t−j].
        // Tap-major loop in window-slot order k = 0..fl (oldest →
        // newest) — each output element sees the identical sequence
        // of adds as the reference scalar loop, so results are
        // bit-identical; the `zip` form vectorises cleanly.
        for c in 0..4 {
            sc.filt_t[c].fill(0.0);
            sc.filt_r[c].fill(0.0);
        }
        for k in 0..fl {
            // Frame index at window slot k; s<0 → replicate frame 0
            // (which is win[0] whenever s≤0, since the window only
            // starts dropping frames once n_pushed > fl).
            let s = t as isize - (fl as isize - 1) + k as isize;
            let widx = (s.max(0) as usize) - w0;
            for c in 0..4 {
                let src_c = if c == 3 { 0 } else { c };
                let tap = self.taps[c][fl - 1 - k];
                let wt = &self.win_t[widx][src_c];
                let wr = &self.win_r[widx][src_c];
                for (a, &v) in sc.filt_t[c].iter_mut().zip(wt.iter()) {
                    *a += tap * v;
                }
                for (a, &v) in sc.filt_r[c].iter_mut().zip(wr.iter()) {
                    *a += tap * v;
                }
            }
        }

        // Per-side weber pyramids via the SIMD/scratch path
        // (`weber_contrast_pyr_into` — same math as
        // `weber_contrast_pyr_dec_scalar` to ~1e-5 FMA-order noise).
        // L_bkg is always the side's own sustained-A filtered plane
        // (upstream divides the interleaved tensor's even channels by
        // L_bkg[0] = test sustained, odd by L_bkg[1] = ref sustained).
        // The 8 builds (4 channels × 2 sides) are fully independent —
        // each owns a disjoint cache+output slot — so under `parallel`
        // they run on rayon's pool.
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
                    let f0t = &sc.filt_t[0];
                    let fr = &sc.filt_r[c];
                    let f0r = &sc.filt_r[0];
                    s.spawn(move |_| {
                        weber_contrast_pyr_into(ft, f0t, w, h, n_levels, cache_t, pyr_t);
                    });
                    s.spawn(move |_| {
                        weber_contrast_pyr_into(fr, f0r, w, h, n_levels, cache_r, pyr_r);
                    });
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for c in 0..4 {
                weber_contrast_pyr_into(
                    &sc.filt_t[c],
                    &sc.filt_t[0],
                    w,
                    h,
                    n_levels,
                    &mut sc.cache_t[c],
                    &mut sc.pyr_t[c],
                );
                weber_contrast_pyr_into(
                    &sc.filt_r[c],
                    &sc.filt_r[0],
                    w,
                    h,
                    n_levels,
                    &mut sc.cache_r[c],
                    &mut sc.pyr_r[c],
                );
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
            for c in 0..4 {
                compute_sensitivities_into(&sc.pyr_r[0].log_l_bkg[k], &rows[c], &mut sc.s_map[c]);
            }

            if is_baseband {
                // D = |T_f − R_f| · S — direct absolute difference on
                // the contrast bands (no masking, no ch_gain).
                let mut q_band = [0.0_f32; 4];
                for c in 0..4 {
                    let t_data = &sc.pyr_t[c].bands[k].data;
                    let r_data = &sc.pyr_r[c].bands[k].data;
                    let s = &sc.s_map[c];
                    let d = &mut sc.d[c];
                    d.clear();
                    d.reserve(n_px_b);
                    for i in 0..n_px_b {
                        d.push((t_data[i] - r_data[i]).abs() * s[i]);
                    }
                    q_band[c] = lp_norm_mean(d, BETA_SPATIAL);
                }
                q_frame.push(q_band);
            } else {
                for c in 0..4 {
                    let gain = CH_GAIN_4[c];
                    let t_data = &sc.pyr_t[c].bands[k].data;
                    let r_data = &sc.pyr_r[c].bands[k].data;
                    let s = &sc.s_map[c];
                    let tp = &mut sc.t_p[c];
                    let rp = &mut sc.r_p[c];
                    tp.clear();
                    tp.resize(n_px_b, 0.0);
                    rp.clear();
                    rp.resize(n_px_b, 0.0);
                    for i in 0..n_px_b {
                        tp[i] = band_mul * t_data[i] * s[i] * gain;
                        rp[i] = band_mul * r_data[i] * s[i] * gain;
                    }
                }
                mult_mutual_band_4ch_into(
                    &sc.t_p,
                    &sc.r_p,
                    bw,
                    bh,
                    &mut sc.d,
                    &mut sc.m_mm,
                    &mut sc.term,
                    &mut sc.pu_scratch,
                );
                let mut q_band = [0.0_f32; 4];
                for c in 0..4 {
                    q_band[c] = lp_norm_mean(&sc.d[c], BETA_SPATIAL);
                }
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
