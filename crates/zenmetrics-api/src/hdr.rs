//! HDR front-end for the SDR metrics: pixel code values → absolute display
//! luminance (cd/m²) → **PU21** perceptually-uniform encoding, so `ssim2-gpu`,
//! `iwssim-gpu`, `butteraugli-gpu`, and `dssim-gpu` (all `sRGB8`-contract) can
//! score HDR pairs without retraining. `cvvdp` has its own native
//! absolute-luminance path (`cvvdp::params::Eotf`/`DisplayModel`) and does NOT
//! use this module.
//!
//! Transfer functions + display model are reimplemented from the ITU/SMPTE/IEC
//! specs (ITU-R BT.2100, SMPTE ST 2084, IEC 61966-2-1). The PU21 coefficients
//! are the published `gfxdisp/pu21` (BSD-3) values. This mirrors
//! `zensim::{transfer, pu21}` (see zensim `docs/HDR_PLAN.md`); kept here so the
//! metric umbrella has the front-end without a zensim dependency.
//!
//! Gated behind the non-default `hdr` feature.
//!
//! **Per-metric feeding** ([`hdr_feeding`] is the single source of truth):
//! - cvvdp / butteraugli → display-relative **linear planes** (own
//!   absolute-luminance models; no u8).
//! - ssim2 (every backend) → **integrated PU21**
//!   ([`HdrFeeding::IntegratedPuNits`]; PU replaces the cube-root inside the
//!   pipeline — UPIQ 0.704 vs ~0.61 for any input-side shell. GPU class:
//!   `ssim2_gpu::XybFlavor::Pu21`; native CPU:
//!   `fast_ssim2::compute_ssimulacra2_pu_nits`, `hdr-pu` feature).
//! - CPU zensim → **integrated PU21** too ([`HdrFeeding::IntegratedPuNits`]
//!   via `zensim::Zensim::compute_pu_linear`, zensim PR #44 — absolute-nits
//!   f32 in, PU21 banding_glare in place of the cube-root, no u8 round-trip).
//! - iwssim (every backend) → **float PU(luma) gray**
//!   ([`HdrFeeding::PuLumaGrayF32`]; UPIQ 0.808 vs 0.628 through u8 — the
//!   quantization round-trip was the loss).
//! - remaining SSIM-family (dssim, externally capped; GPU zensim pending a
//!   PU kernel in the opaque) → the **u8 shell**: [`to_sdr_u8`] with
//!   [`HdrTransfer::PuRescale`] (best *u8 transfer* per
//!   `benchmarks/hdr_feeding_validation_2026-06-03.md`; the old PU-clamp
//!   collapses highlights and is kept only for back-compat).
//! Measurements: `benchmarks/pu_integrated_upiq_2026-06-09.md` + #25.

// ─── Transfer functions: code value → light ──────────────────────────────────

/// IEC 61966-2-1 sRGB EOTF: sRGB-encoded `v ∈ [0,1]` → relative linear `[0,1]`.
#[inline]
pub fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.040_449_936 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// SMPTE ST 2084 (PQ) EOTF: PQ-encoded `v ∈ [0,1]` → **absolute** luminance in
/// cd/m² over `[0, 10000]`.
#[inline]
pub fn pq_eotf(v: f32) -> f32 {
    const L_MAX: f32 = 10000.0;
    const M1: f32 = 0.159_301_75;
    const M2: f32 = 78.843_75;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_562;
    const C3: f32 = 18.687_5;
    let im = v.powf(1.0 / M2);
    let num = (im - C1).max(0.0);
    let den = C2 - C3 * im;
    L_MAX * (num / den).powf(1.0 / M1)
}

/// ITU-R BT.2100 HLG inverse-OETF: `v ∈ [0,1]` → scene-relative linear `[0,12]`
/// per channel. The OOTF (system gamma) depends on the triple's luminance and
/// is applied at the color stage; see [`hlg_system_gamma`].
#[inline]
pub fn hlg_inverse_oetf(v: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 1.0 - 4.0 * A;
    const C: f32 = 0.559_910_7;
    if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (((v - C) / A).exp() + B) / 12.0
    }
}

/// HLG system gamma (ITU-R BT.2100 / BBC WHP 369): `1.2` at a 1000 cd/m² peak,
/// with a luminance + ambient correction above that.
#[inline]
pub fn hlg_system_gamma(y_peak: f32, e_ambient_lux: f32) -> f32 {
    if y_peak <= 1000.0 {
        1.2
    } else {
        let amb = if e_ambient_lux > 0.0 {
            e_ambient_lux
        } else {
            5.0
        };
        1.2 + 0.42 * (y_peak / 1000.0).log10() - 0.076_23 * (amb / 5.0).log10()
    }
}

// ─── Display model: light → absolute emitted luminance ───────────────────────

/// The physical display the metric assumes: peak/black emitted luminance plus
/// ambient screen reflection, all in cd/m². Matches `cvvdp`'s display presets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayModel {
    /// Peak emitted luminance (white), cd/m².
    pub y_peak: f32,
    /// Black emitted luminance (leakage), cd/m².
    pub y_black: f32,
    /// Ambient light reflected off the screen, cd/m².
    pub y_refl: f32,
}

impl DisplayModel {
    /// SDR reference display (cvvdp `standard_4k`): 200 / 0.2 / 0.3979.
    pub const STANDARD_4K: Self = Self {
        y_peak: 200.0,
        y_black: 0.2,
        y_refl: 0.397_887_36,
    };
    /// 1000 cd/m² PQ HDR reference display (BT.2100 grade-1000).
    pub const STANDARD_HDR_PQ_1000: Self = Self {
        y_peak: 1000.0,
        y_black: 0.005,
        y_refl: 0.397_887_36,
    };

    /// Relative linear light `[0,1]` → absolute emitted luminance:
    /// `L = (peak − black)·lin + black + reflection`.
    #[inline]
    pub fn sdr_linear_to_luminance(&self, lin: f32) -> f32 {
        (self.y_peak - self.y_black) * lin + self.y_black + self.y_refl
    }

    /// PQ code value → absolute emitted luminance, clamped to the display's
    /// reproducible range and lifted by black + reflected ambient.
    #[inline]
    pub fn pq_to_luminance(&self, v: f32) -> f32 {
        pq_eotf(v).min(self.y_peak) + self.y_black + self.y_refl
    }
}

// ─── PU21 (Mantiuk & Azimi 2021) ─────────────────────────────────────────────

/// Minimum luminance PU21 is defined over (cd/m²).
pub const PU21_L_MIN: f32 = 0.005;
/// Maximum luminance PU21 is defined over (cd/m²).
pub const PU21_L_MAX: f32 = 10000.0;

/// The 7 fitted `banding_glare` PU21 parameters `[p1..p7]` (gfxdisp/pu21,
/// 2020-02-06) — the published-recommended and paper-measured-best set, and
/// the only one any zen scoring path uses.
const PU21_P: [f32; 7] = [
    0.353_487_9,
    0.373_465_86,
    8.277_049e-5,
    0.906_256_26,
    0.091_503_03,
    0.909_951_7,
    596.314_8,
];

/// Encode absolute luminance `y` (cd/m², clamped to `[PU21_L_MIN, PU21_L_MAX]`)
/// to the PU21 (`banding_glare`) perceptually-uniform value. `100 cd/m² → ~256`.
#[inline]
pub fn pu21_encode(y: f32) -> f32 {
    let y = y.clamp(PU21_L_MIN, PU21_L_MAX);
    let yp = y.powf(PU21_P[3]);
    let inner = (PU21_P[0] + PU21_P[1] * yp) / (1.0 + PU21_P[2] * yp);
    (PU21_P[6] * (inner.powf(PU21_P[4]) - PU21_P[5])).max(0.0)
}

/// Inverse of [`pu21_encode`].
#[inline]
pub fn pu21_decode(v: f32) -> f32 {
    let v_p = (v / PU21_P[6] + PU21_P[5]).max(0.0).powf(1.0 / PU21_P[4]);
    let num = (v_p - PU21_P[0]).max(0.0);
    let den = PU21_P[1] - PU21_P[2] * v_p;
    (num / den).powf(1.0 / PU21_P[3])
}

// ─── Planar encode: absolute-luminance RGB → PU-encoded ──────────────────────

// ─── PQ inverse-EOTF (encode) + the validated HDR→u8 feedings ─────────────────

/// PQ (SMPTE ST.2084) **inverse-EOTF**: absolute luminance (cd/m²) → coded
/// `[0,1]` (the encode direction; [`pq_eotf`] is the decode). PQ maps the full
/// 0..10000 cd/m² range into `[0,1]` by design, so a u8 quantization of it has
/// NO highlight clamp.
#[inline]
pub fn pq_inverse_eotf(nits: f32) -> f32 {
    const M1: f32 = 0.159_301_76;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_562;
    const C3: f32 = 18.6875;
    let y = (nits / 10000.0).clamp(0.0, 1.0);
    let yp = y.powf(M1);
    ((C1 + C2 * yp) / (1.0 + C3 * yp)).powf(M2)
}

/// HDR→u8 transfer for the SDR-family metric kernels (which take 8-bit input).
/// Which transfer is used decides whether the HDR highlight range survives the
/// u8 quantization. Validated on UPIQ (`benchmarks/hdr_feeding_validation_2026-06-03.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HdrTransfer {
    /// PQ (ST.2084) → u8: the full HDR range fits `[0,1]` by design — no clamp.
    /// Close second (ssim2 0.62).
    Pq,
    /// PU21 rescaled so the display peak maps to 255 — full PU range, no clamp.
    /// Best **u8 transfer** (ssim2 0.65 / dssim 0.66 SRCC, 2026-06-03 feeding
    /// eval). As an overall feeding the u8 shell itself is superseded where an
    /// integrated/float path exists: iwssim float-PU 0.808 vs 0.628, GPU ssim2
    /// integrated 0.704 vs ~0.61 (UPIQ, benchmarks addenda).
    #[default]
    PuRescale,
}

/// Default HDR display peak (cd/m²) — content above it clips in `PuRescale`.
pub const HDR_PEAK_NITS: f32 = 1000.0;

/// Encode interleaved absolute-luminance RGB (cd/m²) to sRGB8 for the SDR-family
/// metric kernels via the chosen [`HdrTransfer`]. `peak_nits` is the display
/// peak used by `PuRescale` (`HDR_PEAK_NITS` is the standard choice).
pub fn to_sdr_u8(rgb_nits: &[f32], transfer: HdrTransfer, peak_nits: f32) -> Vec<u8> {
    let pu_max = pu21_encode(peak_nits).max(1.0);
    rgb_nits
        .iter()
        .map(|&y| {
            let v = match transfer {
                HdrTransfer::Pq => pq_inverse_eotf(y) * 255.0,
                HdrTransfer::PuRescale => pu21_encode(y) * (255.0 / pu_max),
            };
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Absolute-luminance interleaved RGB (cd/m²) → PU21(`BandingGlare`)-encoded
/// BT.709-luma gray plane in 0..255 scale (`255 / PU21(peak_nits)`), kept as
/// **f32** — the float PU(luma) feeding for [`HdrFeeding::PuLumaGrayF32`].
/// Identical math to the `PuRescale` u8 shell on luminance, minus the
/// round-to-u8 step that costs IW-SSIM ~0.18 SROCC on UPIQ HDR.
pub fn nits_interleaved_to_pu_luma_gray(rgb_nits: &[f32], peak_nits: f32) -> Vec<f32> {
    let pu_max = pu21_encode(peak_nits).max(1.0);
    let scale = 255.0 / pu_max;
    rgb_nits
        .chunks_exact(3)
        .map(|c| {
            let y = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            pu21_encode(y) * scale
        })
        .collect()
}

/// How a metric should ingest HDR — the **single source of truth** for the
/// per-metric feeding recipe, validated against HDR MOS on UPIQ + AIC-HDR2025
/// (`benchmarks/hdr_feeding_validation_2026-06-03.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrFeeding {
    /// No HDR path **by design**: dssim's internal transform lives in the
    /// external `dssim-core` crate (no seam for integrated PU), and the u8
    /// shell measured ~0.6 SROCC on UPIQ HDR — omitted rather than shipped
    /// degraded. Scoring attempts error loudly.
    Unsupported,
    /// SSIM-family — no absolute-luminance model; encode to u8 via the transfer.
    SdrU8(HdrTransfer),
    /// Luminance-aware (own opsin / CSF on absolute light) — native
    /// display-relative `[0,1]` linear planes, no u8 round-trip.
    LinearPlanes,
    /// **Integrated PU21** (ssim2 on every backend + CPU zensim): the metric
    /// ingests **absolute-luminance interleaved f32 (cd/m²)** and applies
    /// PU21 *inside* its pipeline at the perceptual-encoding layer, replacing
    /// the cube-root — no u8 round-trip, no input-side PU shell. ssim2:
    /// `ssim2_gpu::XybFlavor::Pu21` on the GPU class /
    /// `fast_ssim2::compute_ssimulacra2_pu_nits` (`hdr-pu`) on the native-CPU
    /// dispatch; zensim CPU: `zensim::Zensim::compute_pu_linear`. UPIQ SRCC
    /// **0.7040** GPU / **0.7044** CPU (n=380, fast-ssim2 35f198af) vs ~0.61
    /// (UPIQ) / 0.65 (2026-06-03 eval) for the `PuRescale` u8 shell; the
    /// fed-PU-as-input shell variant caps at ~0.61 (imazen/zenmetrics#25).
    IntegratedPuNits,
    /// **Float PU(luma) gray** (iwssim): `PU21(bt709-luma(nits)) · 255 /
    /// PU21(peak)` fed as f32 gray planes into the metric's gray-native
    /// entry (`score_gray` / `compute_gray`) — same 0..255 scale as the u8
    /// shell, **no quantization round-trip**. UPIQ HDR (n=380): SROCC
    /// **0.8076** (and 0.8123 with `iw_flag` off) vs **0.628** through the
    /// `PuRescale` u8 shell — the u8 round-trip alone cost ~0.18
    /// (`benchmarks/pu_integrated_upiq_2026-06-09.md` addendum 2, #25).
    PuLumaGrayF32,
}

/// The validated HDR feeding for each metric **on the given backend**.
/// Auditable in one place:
/// - **cvvdp** (display model + CSF) → linear planes — UPIQ SRCC **0.758** (our best).
/// - **butteraugli** (opsin + intensity_target) → linear planes — 0.628, beats its u8 path.
/// - **ssim2 (every backend)** → [`HdrFeeding::IntegratedPuNits`] — PU21
///   replaces the cube-root inside the pipeline. GPU class:
///   `ssim2_gpu::XybFlavor::Pu21`, UPIQ SRCC **0.7040** (n=380). Native CPU:
///   `fast_ssim2::compute_ssimulacra2_pu_nits` (`hdr-pu` feature), UPIQ SRCC
///   **0.7044** measured on fast-ssim2 git `35f198af`; consumed through the
///   workspace `[patch.crates-io]` pin to fast-ssim2 main (crates.io 0.8.1
///   predates `hdr-pu` — swap the api/orchestrator deps back to the registry
///   at the next fast-ssim2 publish). Both beat the u8 shell's ~0.61
///   (imazen/zenmetrics#25).
/// - **dssim** → `PuRescale` u8 — 0.66. dssim-core is external and applies its
///   own internal sRGB→LAB transform on whatever bytes it gets; there is no
///   seam to substitute PU21 internally, so the u8 PU shell is the **known
///   cap** for this metric (not a TODO that can be fixed from this crate).
/// - **iwssim** → `PuRescale` u8 — the structurally right layer (IW-SSIM has
///   no internal perceptual nonlinearity; PU-encoding the input is the
///   PU-IW-SSIM construction), but **measured deficient** on UPIQ HDR
///   (2026-06-09, n=380): this shell scores SROCC 0.628 vs published PU-SSIM
///   0.740. Decomposition: ~0.06 is the u8 shell itself (plain SSIM on the
///   identical PU-u8 grays = 0.682 — quantization + 255/PU(peak) rescale vs
///   float PU values), ~0.02 per-channel PU vs PU-of-luma (luma variant
///   0.648), ~0.03 IW weighting on PU-HDR statistics; highlight clip is nil
///   (peak 10000 → 0.631). Open fix: feed PU(luma) as **float** planes into
///   the iwssim core (it consumes f32 gray internally), skipping the u8
///   round-trip. See `benchmarks/pu_integrated_upiq_2026-06-09.md`.
/// - **zensim on [`Backend::Cpu`](crate::Backend::Cpu)** (XYB SSIM-family) →
///   [`HdrFeeding::IntegratedPuNits`] — `zensim::Zensim::compute_pu_linear`
///   (zensim PR #44, squash 3f0334de) replaces the SDR cube-root with the
///   PU21 banding_glare front-end *inside* the metric: absolute-nits f32 in,
///   no u8 round-trip, no input-side PU shell (the fed-PU-as-input shell
///   capped at ~0.61 UPIQ, #25). SDR scoring is untouched (PR #44's
///   identity-100 + SDR-funnel parity gates). HDR-MOS SROCC for the
///   integrated path is not yet measured (zensim#38 tracks absolute-score
///   calibration); the structural u8-shell losses this removes were ~0.18
///   SROCC where decomposed (iwssim, addendum 2).
/// - **zensim on a GPU-class backend** → `PuRescale` u8 — the zensim-gpu
///   opaque has no PU kernel yet; flip when one lands.
pub fn hdr_feeding(metric: crate::MetricKind, backend: crate::Backend) -> HdrFeeding {
    use crate::MetricKind as M;
    match metric {
        M::Cvvdp | M::Butter => HdrFeeding::LinearPlanes,
        // ssim2 — integrated PU21 on EVERY backend: the GPU-class opaques
        // swap the cube-root XYB stage in-kernel (the cubecl runtimes all
        // run the same kernels, so the integrated path follows the opaque),
        // and the native-CPU dispatch routes
        // `fast_ssim2::compute_ssimulacra2_pu_nits` (`hdr-pu`, workspace
        // [patch] pin until a fast-ssim2 release ships the feature).
        M::Ssim2 => HdrFeeding::IntegratedPuNits,
        // CPU zensim — the integrated PU front-end lives in the CPU crate
        // (zensim::compute_pu_linear, PR #44); the GPU opaque has no PU
        // kernel yet and keeps the u8 shell below.
        M::Zensim if backend.resolve() == crate::Backend::Cpu => HdrFeeding::IntegratedPuNits,
        // iwssim, BOTH classes — the CPU pipeline (`score_gray`) and the GPU
        // pipeline (`compute_gray`) are gray-f32-native, so float PU(luma)
        // routes everywhere. The u8 shell measured 0.628 vs 0.808 float on
        // UPIQ HDR (benchmarks addendum 2) — the quantization was the loss.
        M::Iwssim => HdrFeeding::PuLumaGrayF32,
        M::Dssim => HdrFeeding::Unsupported,
        // GPU zensim — the last u8-shell row; flip when the opaque grows a
        // PU kernel.
        M::Zensim => HdrFeeding::SdrU8(HdrTransfer::PuRescale),
    }
}

// ─── HdrScorer: HDR-aware multi-score over the umbrella ───────────────────────

/// HDR-aware multi-score scorer over the umbrella `Metric`. This is the
/// **default** way to score HDR: it configures the underlying
/// [`Metric`](crate::Metric) for an HDR display at construction (cvvdp's
/// `DisplayModel` peak / butteraugli's `intensity_target` + whole-image mode),
/// and [`compute_multi`](Self::compute_multi) then applies the validated
/// per-metric feeding ([`hdr_feeding`]) automatically — the caller hands over
/// **absolute-luminance linear-RGB (cd/m²)** and gets lossless
/// [`Scores`](crate::Scores) back, with no hand-wired feeding:
///
/// - SSIM-family → per [`hdr_feeding`]: integrated PU (GPU ssim2 / CPU
///   zensim), float PU(luma) gray (iwssim), pu-rescale → u8 (the rest).
/// - `cvvdp` / `butteraugli` → display-relative linear planes (`nits / peak`).
///
/// Returns lossless [`Scores`]: butteraugli → `[max, pnorm_3]`, zensim →
/// score + feature vector, the rest → one scalar.
///
/// ```ignore
/// use zenmetrics_api::{Backend, MetricKind};
/// use zenmetrics_api::hdr::{HdrScorer, HDR_PEAK_NITS};
/// let mut s = HdrScorer::new(MetricKind::Butter, Backend::Cuda, w, h, HDR_PEAK_NITS)?;
/// let scores = s.compute_multi(&ref_nits, &dis_nits)?; // [max, pnorm_3]
/// ```
pub struct HdrScorer {
    metric: crate::Metric,
    kind: crate::MetricKind,
    feeding: HdrFeeding,
    peak_nits: f32,
}

impl HdrScorer {
    /// Build an HDR scorer for `kind` targeting a `peak_nits` display (e.g.
    /// [`HDR_PEAK_NITS`]). cvvdp gets a `STANDARD_HDR_LINEAR` display at that
    /// peak; butteraugli gets `intensity_target = peak_nits` + whole-image
    /// [`MemoryMode::Full`](crate::MemoryMode) (its linear path is whole-image
    /// only); the SSIM-family use default params (fed via pu-rescale at score
    /// time). Errors with [`Error::MetricNotEnabled`](crate::Error) if `kind`'s
    /// Cargo feature is off.
    pub fn new(
        kind: crate::MetricKind,
        backend: crate::Backend,
        width: u32,
        height: u32,
        peak_nits: f32,
    ) -> crate::Result<Self> {
        let feeding = hdr_feeding(kind, backend);
        // Bake the peak into the metric's display model (cvvdp/butter) AND record
        // it as the metric's `display_peak`, so `Metric::compute_pixels` feeds HDR
        // at the same peak this scorer targets — the two stay in sync.
        let metric =
            build_hdr_metric(kind, backend, width, height, peak_nits)?.with_display_peak(peak_nits);
        Ok(Self {
            metric,
            kind,
            feeding,
            peak_nits,
        })
    }

    /// The HDR display peak (cd/m²) this scorer targets.
    pub fn peak_nits(&self) -> f32 {
        self.peak_nits
    }

    /// The metric kind being scored.
    pub fn kind(&self) -> crate::MetricKind {
        self.kind
    }

    /// Override the SDR-family feeding transfer (default [`HdrTransfer::PuRescale`]
    /// from [`hdr_feeding`]) — e.g. to expose [`HdrTransfer::Pq`] for
    /// comparison. Only metrics whose feeding is [`HdrFeeding::SdrU8`] are
    /// affected (today: GPU zensim, the last u8-shell row) — no effect on
    /// cvvdp/butteraugli (linear path) or on any integrated/float feeding
    /// (ssim2 every backend, CPU zensim, iwssim: the PU encode happens inside
    /// the pipeline or at full f32 — there is no u8 transfer shell to
    /// override). To score the legacy u8 shell on such a metric, feed
    /// [`to_sdr_u8`] output through [`crate::Metric::compute_srgb_u8`]
    /// directly.
    pub fn with_transfer(mut self, transfer: HdrTransfer) -> Self {
        if let HdrFeeding::SdrU8(_) = self.feeding {
            self.feeding = HdrFeeding::SdrU8(transfer);
        }
        self
    }

    /// Score one HDR pair given as **interleaved absolute-luminance linear-RGB**
    /// (cd/m², `[R,G,B, R,G,B, …]`, length `width·height·3`). Applies the
    /// per-metric feeding automatically; returns lossless [`Scores`](crate::Scores).
    pub fn compute_multi(
        &mut self,
        ref_nits: &[f32],
        dis_nits: &[f32],
    ) -> crate::Result<crate::Scores> {
        match self.feeding {
            HdrFeeding::Unsupported => Err(crate::Error::Metric {
                kind: "dssim",
                message: "no HDR path by design (external dssim-core transform; \
                          u8 shell measured ~0.6 on UPIQ) — score SDR or pick \
                          another metric"
                    .into(),
            }),
            HdrFeeding::SdrU8(transfer) => {
                let r = to_sdr_u8(ref_nits, transfer, self.peak_nits);
                let d = to_sdr_u8(dis_nits, transfer, self.peak_nits);
                self.metric.compute_srgb_u8_multi(&r, &d)
            }
            HdrFeeding::LinearPlanes => {
                // Display-relative [0,1] linear = nits / peak, CLAMPED to [0,1]:
                // content brighter than the display peak clips (a real `peak`-nit
                // display can't show more), matching the baked cvvdp display peak
                // / butter intensity_target. The clamp is load-bearing — without
                // it, super-peak highlights pass through as > 1.0 and the score
                // diverges from the validated display-referred behaviour.
                let inv = 1.0 / self.peak_nits;
                let r: Vec<f32> = ref_nits
                    .iter()
                    .map(|&v| (v * inv).clamp(0.0, 1.0))
                    .collect();
                let d: Vec<f32> = dis_nits
                    .iter()
                    .map(|&v| (v * inv).clamp(0.0, 1.0))
                    .collect();
                self.metric.compute_from_linear_interleaved_multi(&r, &d)
            }
            // Integrated PU21 (GPU ssim2): the input is already the absolute
            // nits this entry takes — straight through, unclipped (the PU21
            // encode clamps to its [0.005, 10000] operating range in-kernel,
            // matching the UPIQ-validated example feeding).
            HdrFeeding::IntegratedPuNits => self
                .metric
                .compute_pu_nits_interleaved_multi(ref_nits, dis_nits),
            // Float PU(luma) gray (iwssim): PU-encode luminance host-side at
            // full precision and hand the metric f32 gray planes directly.
            HdrFeeding::PuLumaGrayF32 => {
                let r = nits_interleaved_to_pu_luma_gray(ref_nits, self.peak_nits);
                let d = nits_interleaved_to_pu_luma_gray(dis_nits, self.peak_nits);
                self.metric.compute_pu_luma_gray_multi(&r, &d)
            }
        }
    }

    /// Convenience single-score variant — the [`Scores::primary`] scalar.
    pub fn compute(&mut self, ref_nits: &[f32], dis_nits: &[f32]) -> crate::Result<crate::Score> {
        self.compute_multi(ref_nits, dis_nits)
            .map(|s| s.primary_score())
    }

    /// **Descriptor-driven unified entry** — score a pair of
    /// [`PixelSlice`](zenpixels::PixelSlice)s. The pixel **descriptor** decides
    /// the path, so SDR and HDR are one call:
    ///
    /// - An `RGB8_SRGB` slice → the metric's **native SDR** path (the same u8
    ///   the metric kernels take — bit-identical to [`crate::Metric::compute_srgb_u8`],
    ///   so validated SDR scores are preserved exactly).
    /// - Any other descriptor (linear / PQ / HLG HDR) → the **faithful HDR**
    ///   feeding at this scorer's [`peak_nits`](Self::peak_nits): luminance-aware
    ///   metrics get display-relative linear planes; the SSIM-family get the
    ///   pu-rescale u8. zenpixels-convert applies the transfer.
    ///
    /// This is the collapse of the HDR/SDR API split: one descriptor-driven call
    /// over the same warm metric instance. `HdrScorer` is then "a `Metric` + a
    /// display, taking `PixelSlice`s".
    #[cfg(feature = "pixels")]
    pub fn compute_pixels_multi(
        &mut self,
        r: zenpixels::PixelSlice<'_>,
        d: zenpixels::PixelSlice<'_>,
    ) -> crate::Result<crate::Scores> {
        // Thin wrapper: the metric's `display_peak` was set to `self.peak_nits`
        // at construction, so `Metric::compute_pixels_multi` applies the same
        // descriptor-driven per-metric feeding this scorer used to inline (SDR
        // native for `RGB8_SRGB`, pu-rescale / linear-planes for HDR). Single
        // source of truth, so the two paths can't drift.
        self.metric.compute_pixels_multi(r, d)
    }

    /// Single-score variant of [`Self::compute_pixels_multi`].
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(
        &mut self,
        r: zenpixels::PixelSlice<'_>,
        d: zenpixels::PixelSlice<'_>,
    ) -> crate::Result<crate::Score> {
        self.compute_pixels_multi(r, d).map(|s| s.primary_score())
    }
}

/// Convert a slice to packed sRGB8 (validating dims) for the native SDR path.
#[cfg(feature = "pixels")]
pub(crate) fn slice_to_srgb8(
    s: &zenpixels::PixelSlice<'_>,
    w: u32,
    h: u32,
) -> crate::Result<Vec<u8>> {
    if s.width() != w || s.rows() != h {
        return Err(crate::Error::Metric {
            kind: "pixels",
            message: format!("slice {}x{} != metric {}x{}", s.width(), s.rows(), w, h),
        });
    }
    zenmetrics_gpu_core::convert_to_srgb_rgb8(s, zenpixels::PixelDescriptor::RGB8_SRGB).map_err(
        |e| crate::Error::Metric {
            kind: "pixels",
            message: format!("sRGB8 conversion failed: {e:?}"),
        },
    )
}

/// The multiplier turning [`zenmetrics_gpu_core::convert_to_linear_f32`] output
/// into **absolute display luminance (cd/m²)** — the one place the
/// descriptor-vs-nits scale convention is encoded.
///
/// `convert_to_linear_f32` decodes each descriptor to *its own* linear scale:
/// - **PQ** (ST.2084, absolute): `[0,1]` where **1.0 = 10000 cd/m²** (libjxl
///   normalization), so the scale to nits is a fixed `10000`.
/// - **everything relative/scene-referred** (sRGB, BT.709, linear, HLG;
///   `reference_white_nits == 1.0`): `[0,1]` display-relative, so `1.0` is the
///   display's `peak_nits` — the scale is `peak_nits`.
///
/// HLG is treated as relative here (no OOTF) — consistent with its `1.0`
/// reference white; faithful HLG display-light would need the peak-dependent
/// OOTF, which no current input path exercises (planar HDR comes from
/// jxl-encoder as relative linear, per the HDR design).
#[cfg(feature = "pixels")]
fn linear_to_nits_scale(transfer: zenpixels::TransferFunction, peak_nits: f32) -> f32 {
    match transfer {
        zenpixels::TransferFunction::Pq => 10_000.0,
        _ => peak_nits,
    }
}

/// Convert a slice to **interleaved** display-relative `[0,1]` linear
/// (`[R,G,B, …]`) at `peak_nits` — zenpixels-convert applies the descriptor's
/// transfer, then `linear_to_nits_scale` maps to absolute nits and `÷ peak`
/// brings it display-relative (content above the display peak clips, matching
/// [`HdrScorer::compute_multi`]'s LinearPlanes clamp). Interleaved is the
/// canonical linear transport: native CPU takes it as-is, GPU deinterleaves
/// inside the interleaved entry.
#[cfg(feature = "pixels")]
pub(crate) fn slice_to_display_relative_linear_interleaved(
    s: &zenpixels::PixelSlice<'_>,
    peak_nits: f32,
) -> crate::Result<Vec<f32>> {
    let lin = zenmetrics_gpu_core::convert_to_linear_f32(s).map_err(|e| crate::Error::Metric {
        kind: "pixels",
        message: format!("linear conversion failed: {e:?}"),
    })?;
    let rel = peak_nits.recip() * linear_to_nits_scale(s.descriptor().transfer(), peak_nits);
    Ok(lin.iter().map(|&v| (v * rel).clamp(0.0, 1.0)).collect())
}

/// Convert a slice to **interleaved absolute-luminance** linear RGB (cd/m²)
/// for the [`HdrFeeding::IntegratedPuNits`] path: decode to linear via
/// zenpixels-convert, scale to nits via [`linear_to_nits_scale`] — and stop.
/// No `÷peak`, no `[0,1]` clamp: the PU21 encode clamps to its
/// `[0.005, 10000]` cd/m² operating range in-kernel, and the UPIQ-validated
/// feeding (imazen/zenmetrics#25) was raw unclipped nits.
#[cfg(feature = "pixels")]
pub(crate) fn slice_to_absolute_nits_interleaved(
    s: &zenpixels::PixelSlice<'_>,
    peak_nits: f32,
) -> crate::Result<Vec<f32>> {
    let lin = zenmetrics_gpu_core::convert_to_linear_f32(s).map_err(|e| crate::Error::Metric {
        kind: "pixels",
        message: format!("linear conversion failed: {e:?}"),
    })?;
    let scale = linear_to_nits_scale(s.descriptor().transfer(), peak_nits);
    Ok(lin.iter().map(|&v| v * scale).collect())
}

/// Convert a slice to the SSIM-family pu-rescale u8 feeding: decode to linear,
/// scale to absolute nits via [`linear_to_nits_scale`], then [`to_sdr_u8`].
/// (Relative descriptors scale by `peak`; PQ by its fixed `10000`.)
#[cfg(feature = "pixels")]
pub(crate) fn slice_to_pu_rescaled_u8(
    s: &zenpixels::PixelSlice<'_>,
    transfer: HdrTransfer,
    peak_nits: f32,
) -> crate::Result<Vec<u8>> {
    let lin = zenmetrics_gpu_core::convert_to_linear_f32(s).map_err(|e| crate::Error::Metric {
        kind: "pixels",
        message: format!("linear conversion failed: {e:?}"),
    })?;
    let scale = linear_to_nits_scale(s.descriptor().transfer(), peak_nits);
    let nits: Vec<f32> = lin.iter().map(|&v| v * scale).collect();
    Ok(to_sdr_u8(&nits, transfer, peak_nits))
}

/// Construct a [`Metric`](crate::Metric) configured for HDR scoring at
/// `peak_nits` — the feeding-specific construction behind [`HdrScorer::new`].
fn build_hdr_metric(
    kind: crate::MetricKind,
    backend: crate::Backend,
    width: u32,
    height: u32,
    peak_nits: f32,
) -> crate::Result<crate::Metric> {
    // Native-CPU HDR scoring: build straight from the `cpu-*` crates (no GPU
    // `MetricParams`, so no `cvvdp`/`butter`/... GPU feature — and NEVER
    // cubecl-cpu). The CPU dispatch implements every HDR feeding path, so this
    // is a fully-wired score path, not a fallback. Intercept before the
    // GPU-param construction below so a pure-CPU build can reach it.
    #[cfg(any(
        feature = "cpu-ssim2",
        feature = "cpu-cvvdp",
        feature = "cpu-dssim",
        feature = "cpu-butter",
        feature = "cpu-zensim",
        feature = "cpu-iwssim"
    ))]
    if backend.resolve() == crate::Backend::Cpu {
        return crate::Metric::new_cpu_hdr(kind, width, height, peak_nits);
    }
    match kind {
        #[cfg(feature = "cvvdp")]
        crate::MetricKind::Cvvdp => {
            use crate::cvvdp::params::DisplayModel;
            let display = DisplayModel {
                y_peak: peak_nits,
                ..DisplayModel::STANDARD_HDR_LINEAR
            };
            crate::Metric::new(
                kind,
                backend,
                width,
                height,
                crate::MetricParams::cvvdp_with_display(display),
            )
        }
        #[cfg(feature = "butter")]
        crate::MetricKind::Butter => {
            use crate::butter::ButteraugliParams;
            let params = crate::MetricParams::Butter(
                ButteraugliParams::default().with_intensity_target(peak_nits),
            );
            crate::Metric::new_with_memory_mode(
                kind,
                backend,
                width,
                height,
                params,
                crate::MemoryMode::Full,
            )
        }
        // SSIM-family (and any metric whose feature is off → loud error from
        // `Metric::new`): default params; pu-rescale feeding is applied to the
        // u8 input at score time.
        _ => crate::Metric::new(
            kind,
            backend,
            width,
            height,
            crate::MetricParams::default_for(kind),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn pq_eotf_reference_values() {
        assert_eq!(pq_eotf(0.0), 0.0);
        assert!(
            close(pq_eotf(0.5), 92.2466, 0.05),
            "pq(0.5)={}",
            pq_eotf(0.5)
        );
        assert!(
            close(pq_eotf(1.0), 10000.0, 1.0),
            "pq(1.0)={}",
            pq_eotf(1.0)
        );
    }

    #[test]
    fn srgb_and_display_model_nits() {
        let d = DisplayModel::STANDARD_4K;
        // mid-gray 128/255 → 43.73 cd/m² (matches the cvvdp standard_4k path).
        let mid = d.sdr_linear_to_luminance(srgb_eotf(128.0 / 255.0));
        assert!(close(mid, 43.73, 0.05), "mid={mid}");
        assert!(close(
            d.sdr_linear_to_luminance(srgb_eotf(1.0)),
            200.398,
            0.05
        ));
    }

    #[test]
    fn hlg_values() {
        assert!(close(hlg_inverse_oetf(0.5), 1.0 / 12.0, 1e-5));
        assert_eq!(hlg_system_gamma(1000.0, 200.0), 1.2);
    }

    #[test]
    fn pu21_100_nits_near_256_and_default() {
        let v = pu21_encode(100.0);
        assert!((v - 256.0).abs() < 1.5, "encode(100)={v}");
    }

    #[test]
    fn pu21_monotone_and_round_trip() {
        let mut prev = pu21_encode(PU21_L_MIN);
        for i in 1..=200 {
            let y = PU21_L_MIN * (PU21_L_MAX / PU21_L_MIN).powf(i as f32 / 200.0);
            let v = pu21_encode(y);
            assert!(v >= prev, "not monotone at {y}");
            // round-trip ~1% (f32 through two power chains)
            if (0.01..=5000.0).contains(&y) {
                let y2 = pu21_decode(v);
                assert!((y2 - y).abs() <= 0.02 * y.max(1e-6), "rt {y}→{v}→{y2}");
            }
            prev = v;
        }
    }

    #[test]
    fn pq_inverse_round_trips() {
        for nits in [0.1, 1.0, 100.0, 1000.0, 4000.0, 10000.0] {
            let code = pq_inverse_eotf(nits);
            assert!((0.0..=1.0).contains(&code), "{nits} → {code} out of [0,1]");
            assert!(close(pq_eotf(code), nits, nits * 0.01 + 0.01)); // decode⁻¹ ≈ identity
        }
    }

    #[test]
    fn to_sdr_u8_transfers_keep_highlights_distinct() {
        // Two distinct highlights, 600 vs 4000 cd/m². pu-clamp pins BOTH at 255
        // (the bug); pq + pu-rescale keep them distinct and below the ceiling.
        let lo = [600.0, 600.0, 600.0];
        let hi = [4000.0, 4000.0, 4000.0];
        for t in [HdrTransfer::Pq, HdrTransfer::PuRescale] {
            let l = to_sdr_u8(&lo, t, HDR_PEAK_NITS)[0];
            let h = to_sdr_u8(&hi, t, HDR_PEAK_NITS)[0];
            assert!(
                l < h && l < 255,
                "{t:?}: 600({l}) should be < 4000({h}) < 255"
            );
        }
    }

    #[test]
    fn hdr_feeding_table_matches_validation() {
        use crate::Backend as B;
        use crate::MetricKind as M;
        use HdrFeeding::*;
        // Backend-independent rows: every concrete backend yields the same
        // feeding for the luminance-aware pair and the u8-shell family.
        for b in [B::Cuda, B::Wgpu, B::Hip, B::CubeclCpu, B::Cpu] {
            assert_eq!(hdr_feeding(M::Cvvdp, b), LinearPlanes);
            assert_eq!(hdr_feeding(M::Butter, b), LinearPlanes);
            // dssim is omitted from HDR by design.
            assert_eq!(hdr_feeding(M::Dssim, b), Unsupported);
            // iwssim: float PU(luma) gray on EVERY class — both the CPU
            // (`score_gray`) and GPU (`compute_gray`) pipelines are
            // f32-gray-native, and the u8 shell measured 0.628 vs 0.808
            // float on UPIQ HDR (benchmarks addendum 2).
            assert_eq!(hdr_feeding(M::Iwssim, b), PuLumaGrayF32);
        }
    }

    /// zensim is the inverse of ssim2: the integrated PU front-end lives in
    /// the **CPU** crate (`zensim::Zensim::compute_pu_linear`, zensim PR
    /// #44), while the GPU opaque has no PU kernel yet and keeps the u8
    /// PU shell.
    #[test]
    fn hdr_feeding_zensim_cpu_routes_integrated_pu() {
        use crate::Backend as B;
        use crate::MetricKind as M;
        assert_eq!(hdr_feeding(M::Zensim, B::Cpu), HdrFeeding::IntegratedPuNits);
        for b in [B::Cuda, B::Wgpu, B::Hip, B::CubeclCpu] {
            assert_eq!(
                hdr_feeding(M::Zensim, b),
                HdrFeeding::SdrU8(HdrTransfer::PuRescale),
                "{b:?}"
            );
        }
    }

    #[test]
    fn pu_luma_gray_matches_u8_shell_math_minus_quantization() {
        // Gray pixels: the float plane must equal the u8 shell's value
        // BEFORE rounding (same PU21 + same 255/PU(peak) rescale).
        let nits = [
            0.5f32, 0.5, 0.5, 100.0, 100.0, 100.0, 4000.0, 4000.0, 4000.0,
        ];
        let g = nits_interleaved_to_pu_luma_gray(&nits, HDR_PEAK_NITS);
        assert_eq!(g.len(), 3);
        let pu_max = pu21_encode(HDR_PEAK_NITS).max(1.0);
        for (i, &y) in [0.5f32, 100.0, 4000.0].iter().enumerate() {
            let want = pu21_encode(y) * 255.0 / pu_max;
            assert!(
                (g[i] - want).abs() < 1e-4,
                "gray[{i}] = {} want {want}",
                g[i]
            );
        }
        // Strictly increasing in luminance; super-peak content is NOT
        // clamped to 255 (float feed keeps highlight separation).
        assert!(g[0] < g[1] && g[1] < g[2]);
        assert!(
            g[2] > 255.0,
            "4000-nit at 1000-nit peak exceeds 255: {}",
            g[2]
        );
    }

    /// ssim2 routes the integrated PU21 path on EVERY backend: the GPU-class
    /// opaques swap the cube-root XYB stage in-kernel, and the native-CPU
    /// dispatch routes `fast_ssim2::compute_ssimulacra2_pu_nits` (`hdr-pu`,
    /// consumed via the workspace `[patch.crates-io]` pin until a fast-ssim2
    /// release ships the feature). No u8 shell remains for ssim2.
    #[test]
    fn hdr_feeding_ssim2_routes_integrated_pu_every_backend() {
        use crate::Backend as B;
        use crate::MetricKind as M;
        for b in [B::Cuda, B::Wgpu, B::Hip, B::CubeclCpu, B::Cpu] {
            assert_eq!(
                hdr_feeding(M::Ssim2, b),
                HdrFeeding::IntegratedPuNits,
                "{b:?}"
            );
        }
    }

    /// The descriptor→nits scale convention behind the unified `compute_pixels`
    /// helpers: PQ decodes to `[0,1]`=10000 cd/m² (fixed 10000 scale); every
    /// relative/scene-referred transfer decodes to display-relative `[0,1]`
    /// (scale = the display peak). Relative collapsing to `peak` is what makes
    /// the `LinearPlanes` `÷peak` a pass-through (the GPU tests' baseline).
    #[cfg(feature = "pixels")]
    #[test]
    fn linear_to_nits_scale_pq_is_absolute_rest_relative() {
        use zenpixels::TransferFunction as TF;
        let peak = 600.0;
        assert_eq!(linear_to_nits_scale(TF::Pq, peak), 10_000.0);
        for tf in [
            TF::Srgb,
            TF::Bt709,
            TF::Linear,
            TF::Hlg,
            TF::Gamma22,
            TF::Unknown,
        ] {
            assert_eq!(
                linear_to_nits_scale(tf, peak),
                peak,
                "{tf:?} is relative — scale must be the display peak"
            );
        }
    }
}
