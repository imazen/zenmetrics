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
//! **HDR→u8 feeding (validated):** the SDR-metric kernels take 8-bit, so the
//! HDR signal must be encoded to u8 — and *how* decides whether the highlight
//! range survives. [`to_sdr_u8`] with the default [`HdrTransfer::PuRescale`]
//! (PU21 rescaled to fit u8, no clamp) is the **validated best** vs HDR MOS;
//! the old PU-clamp ([`pu_encode_rgb_to_srgb8`]) collapses highlights and is
//! kept only for back-compat. [`hdr_feeding`] is the single source of truth for
//! which feeding each metric gets. See `benchmarks/hdr_feeding_validation_2026-06-03.md`.

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

/// PU21 parameter sets. `BandingGlare` is the recommended default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Pu21Variant {
    /// Minimizes visible banding; no glare model.
    Banding,
    /// Banding + display glare. **Recommended default** (gfxdisp/pu21 default).
    #[default]
    BandingGlare,
    /// Peak-sensitivity tuned; no glare model.
    Peaks,
    /// Peaks + display glare.
    PeaksGlare,
}

impl Pu21Variant {
    /// The 7 fitted parameters `[p1..p7]` (gfxdisp/pu21, updated 2020-02-06).
    // Published exact coefficients — the underscore grouping is verbatim
    // transcription from gfxdisp/pu21, so the cosmetic grouping lint doesn't apply.
    #[allow(clippy::inconsistent_digit_grouping)]
    #[inline]
    const fn params(self) -> [f32; 7] {
        match self {
            Self::Banding => [
                1.070_275_3,
                0.408_827_4,
                0.153_224_3,
                0.252_032_6,
                1.063_512_9,
                1.141_150_5,
                521.452_75,
            ],
            Self::BandingGlare => [
                0.353_487_9,
                0.373_465_86,
                8.277_049e-5,
                0.906_256_26,
                0.091_503_03,
                0.909_951_7,
                596.314_8,
            ],
            Self::Peaks => [
                1.043_882_8,
                0.645_949_55,
                0.319_458_42,
                0.374_025_25,
                1.114_783_4,
                1.095_360_4,
                384.921_76,
            ],
            Self::PeaksGlare => [
                816.885_03,
                1479.464_0,
                0.001_253_215_6,
                0.932_963_7,
                0.067_466_44,
                1.573_435_4,
                419.600_64,
            ],
        }
    }
}

/// Encode absolute luminance `y` (cd/m², clamped to `[PU21_L_MIN, PU21_L_MAX]`)
/// to the PU21 perceptually-uniform value. `100 cd/m² → ~256`.
#[inline]
pub fn pu21_encode(y: f32, variant: Pu21Variant) -> f32 {
    let p = variant.params();
    let y = y.clamp(PU21_L_MIN, PU21_L_MAX);
    let yp = y.powf(p[3]);
    let inner = (p[0] + p[1] * yp) / (1.0 + p[2] * yp);
    (p[6] * (inner.powf(p[4]) - p[5])).max(0.0)
}

/// Inverse of [`pu21_encode`].
#[inline]
pub fn pu21_decode(v: f32, variant: Pu21Variant) -> f32 {
    let p = variant.params();
    let v_p = (v / p[6] + p[5]).max(0.0).powf(1.0 / p[4]);
    let num = (v_p - p[0]).max(0.0);
    let den = p[1] - p[2] * v_p;
    (num / den).powf(1.0 / p[3])
}

// ─── Planar encode: absolute-luminance RGB → PU-encoded ──────────────────────

/// PU-encode each channel of an interleaved **absolute-luminance** (cd/m²) RGB
/// `f32` buffer. Output is PU21 `f32` (range ~`[0, 600]`; 100 cd/m² → ~256) —
/// the **faithful** HDR form for an `f32`-capable metric.
pub fn pu_encode_rgb_planar(rgb_nits: &[f32], variant: Pu21Variant) -> Vec<f32> {
    rgb_nits.iter().map(|&y| pu21_encode(y, variant)).collect()
}

/// **Legacy / degraded** — PU-encode then **clamp** to u8, collapsing the
/// >~100 cd/m² highlight range. Kept for back-compat + the clamp regression
/// test; **prefer [`to_sdr_u8`] with [`HdrTransfer::PuRescale`]**, which is the
/// validated best feeding (UPIQ: 0.55 clamp → 0.65 rescale). See
/// `benchmarks/hdr_feeding_validation_2026-06-03.md`.
pub fn pu_encode_rgb_to_srgb8(rgb_nits: &[f32], variant: Pu21Variant) -> Vec<u8> {
    rgb_nits
        .iter()
        .map(|&y| pu21_encode(y, variant).round().clamp(0.0, 255.0) as u8)
        .collect()
}

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
    /// PU21 (`BandingGlare`) **clamped** to u8 — collapses everything above
    /// ~100 cd/m² to 255 (PU21 ranges to ~600). DEGRADES highlights; the
    /// pre-validation legacy path (worst: ssim2 0.55 SRCC).
    PuClamp,
    /// PQ (ST.2084) → u8: the full HDR range fits `[0,1]` by design — no clamp.
    /// Close second (ssim2 0.62).
    Pq,
    /// PU21 rescaled so the display peak maps to 255 — full PU range, no clamp.
    /// **Validated best** (ssim2 0.65 / dssim 0.66 SRCC vs HDR MOS).
    #[default]
    PuRescale,
}

/// Default HDR display peak (cd/m²) — content above it clips in `PuRescale`.
pub const HDR_PEAK_NITS: f32 = 1000.0;

/// Encode interleaved absolute-luminance RGB (cd/m²) to sRGB8 for the SDR-family
/// metric kernels via the chosen [`HdrTransfer`]. `peak_nits` is the display
/// peak used by `PuRescale` (`HDR_PEAK_NITS` is the standard choice).
pub fn to_sdr_u8(rgb_nits: &[f32], transfer: HdrTransfer, peak_nits: f32) -> Vec<u8> {
    let pu_max = pu21_encode(peak_nits, Pu21Variant::BandingGlare).max(1.0);
    rgb_nits
        .iter()
        .map(|&y| {
            let v = match transfer {
                HdrTransfer::PuClamp => pu21_encode(y, Pu21Variant::BandingGlare),
                HdrTransfer::Pq => pq_inverse_eotf(y) * 255.0,
                HdrTransfer::PuRescale => {
                    pu21_encode(y, Pu21Variant::BandingGlare) * (255.0 / pu_max)
                }
            };
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// How a metric should ingest HDR — the **single source of truth** for the
/// per-metric feeding recipe, validated against HDR MOS on UPIQ + AIC-HDR2025
/// (`benchmarks/hdr_feeding_validation_2026-06-03.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrFeeding {
    /// SSIM-family — no absolute-luminance model; encode to u8 via the transfer.
    SdrU8(HdrTransfer),
    /// Luminance-aware (own opsin / CSF on absolute light) — native
    /// display-relative `[0,1]` linear planes, no u8 round-trip.
    LinearPlanes,
    /// **Integrated PU21** (GPU ssim2): the metric ingests **absolute-luminance
    /// interleaved f32 (cd/m²)** and applies PU21 *inside* its pipeline at the
    /// perceptual-encoding layer (`ssim2_gpu::XybFlavor::Pu21`), replacing the
    /// cube-root — no u8 round-trip, no input-side PU shell. UPIQ SRCC
    /// **0.7040** (n=380) vs 0.65 for the `PuRescale` u8 shell; the
    /// fed-PU-as-input shell variant caps at ~0.61 (imazen/zenmetrics#25).
    IntegratedPuNits,
}

/// The validated HDR feeding for each metric **on the given backend**.
/// Auditable in one place:
/// - **cvvdp** (display model + CSF) → linear planes — UPIQ SRCC **0.758** (our best).
/// - **butteraugli** (opsin + intensity_target) → linear planes — 0.628, beats its u8 path.
/// - **ssim2 on a GPU-class backend** → [`HdrFeeding::IntegratedPuNits`] —
///   PU21 replaces the cube-root inside the GPU pipeline; UPIQ SRCC **0.7040**
///   vs 0.65 for the u8 shell (imazen/zenmetrics#25).
/// - **ssim2 on [`Backend::Cpu`](crate::Backend::Cpu)** → `PuRescale` u8 —
///   fast-ssim2 0.8.1 on crates.io predates the `hdr-pu` feature (it exists
///   only on fast-ssim2 git main, 35f198af; CPU SROCC 0.7044 measured there).
///   Route the CPU path to its integrated PU entry once a fast-ssim2 release
///   carries `hdr-pu`.
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
/// - **zensim** (XYB SSIM-family) → `PuRescale` u8 — its PU front-end is
///   pending zensim PR #44 (`feat/hdr-pu-frontend`); revisit this routing
///   when that merges and a validated SROCC lands.
pub fn hdr_feeding(metric: crate::MetricKind, backend: crate::Backend) -> HdrFeeding {
    use crate::MetricKind as M;
    match metric {
        M::Cvvdp | M::Butter => HdrFeeding::LinearPlanes,
        // GPU-class ssim2 (every backend except the native-CPU dispatch —
        // `Auto` resolves first, so a GPU-less box that resolves to the
        // native CPU path gets the u8 shell). The cubecl runtimes all run
        // the same kernels, so the integrated path follows the opaque.
        M::Ssim2 if backend.resolve() != crate::Backend::Cpu => HdrFeeding::IntegratedPuNits,
        M::Ssim2 | M::Dssim | M::Iwssim | M::Zensim => HdrFeeding::SdrU8(HdrTransfer::PuRescale),
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
/// - SSIM-family (`ssim2`/`dssim`/`iwssim`/`zensim`) → pu-rescale → u8.
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
    /// from [`hdr_feeding`]) — e.g. to expose the legacy `PuClamp` or `Pq` for
    /// comparison. No effect on cvvdp/butteraugli (linear path) **or on GPU
    /// ssim2** ([`HdrFeeding::IntegratedPuNits`] applies PU21 inside the
    /// pipeline — there is no u8 transfer shell to override). To score the
    /// legacy u8 shell on GPU ssim2, feed [`to_sdr_u8`] output through
    /// [`crate::Metric::compute_srgb_u8`] directly.
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
        assert_eq!(Pu21Variant::default(), Pu21Variant::BandingGlare);
        let v = pu21_encode(100.0, Pu21Variant::BandingGlare);
        assert!((v - 256.0).abs() < 1.5, "encode(100)={v}");
    }

    #[test]
    fn pu21_monotone_and_round_trip() {
        let mut prev = pu21_encode(PU21_L_MIN, Pu21Variant::BandingGlare);
        for i in 1..=200 {
            let y = PU21_L_MIN * (PU21_L_MAX / PU21_L_MIN).powf(i as f32 / 200.0);
            let v = pu21_encode(y, Pu21Variant::BandingGlare);
            assert!(v >= prev, "not monotone at {y}");
            // round-trip ~1% (f32 through two power chains)
            if (0.01..=5000.0).contains(&y) {
                let y2 = pu21_decode(v, Pu21Variant::BandingGlare);
                assert!((y2 - y).abs() <= 0.02 * y.max(1e-6), "rt {y}→{v}→{y2}");
            }
            prev = v;
        }
    }

    #[test]
    fn planar_encode_shapes() {
        // SDR-range nits stay within u8; an HDR highlight clamps.
        let nits = [43.7, 100.0, 200.0, 600.0];
        let f = pu_encode_rgb_planar(&nits, Pu21Variant::BandingGlare);
        assert_eq!(f.len(), 4);
        assert!(f[1] > 250.0 && f[1] < 262.0); // 100 nit ≈ 256
        let u = pu_encode_rgb_to_srgb8(&nits, Pu21Variant::BandingGlare);
        assert_eq!(u, [f[0].round() as u8, 255, 255, 255]); // highlights clamp
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
    fn to_sdr_u8_clamp_collapses_highlights_others_dont() {
        // Two distinct highlights, 600 vs 4000 cd/m². pu-clamp pins BOTH at 255
        // (the bug); pq + pu-rescale keep them distinct and below the ceiling.
        let lo = [600.0, 600.0, 600.0];
        let hi = [4000.0, 4000.0, 4000.0];
        assert_eq!(to_sdr_u8(&lo, HdrTransfer::PuClamp, HDR_PEAK_NITS)[0], 255);
        assert_eq!(to_sdr_u8(&hi, HdrTransfer::PuClamp, HDR_PEAK_NITS)[0], 255);
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
            for m in [M::Dssim, M::Iwssim, M::Zensim] {
                assert_eq!(hdr_feeding(m, b), SdrU8(HdrTransfer::PuRescale));
            }
        }
    }

    /// GPU-class ssim2 routes to the integrated PU21 path; the native-CPU
    /// dispatch stays on the u8 PU shell until a fast-ssim2 release ships
    /// the `hdr-pu` feature (it's git-main-only as of 0.8.1).
    #[test]
    fn hdr_feeding_ssim2_gpu_routes_integrated_pu() {
        use crate::Backend as B;
        use crate::MetricKind as M;
        for b in [B::Cuda, B::Wgpu, B::Hip, B::CubeclCpu] {
            assert_eq!(
                hdr_feeding(M::Ssim2, b),
                HdrFeeding::IntegratedPuNits,
                "{b:?}"
            );
        }
        assert_eq!(
            hdr_feeding(M::Ssim2, B::Cpu),
            HdrFeeding::SdrU8(HdrTransfer::PuRescale)
        );
    }

    /// Drift guard: [`pu21_encode`]'s four parameter sets must keep matching
    /// the gfxdisp/pu21 reference. Goldens are float64 values computed by the
    /// pinned reference (generator: zensim `scripts/pu21_golden.py`);
    /// tolerance `0.1 + 5e-3 · |want|` absorbs the f32 power-chain error.
    /// The same goldens are pinned in zensim's `pu21.rs` and (banding_glare
    /// row) in ssim2-gpu's `kernels/xyb.rs`, so all PU21 copies across the
    /// workspace drift-lock to one float64 source.
    #[test]
    fn reference_parity_gfxdisp_goldens() {
        let y = [0.01f32, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0];
        let rows: [(Pu21Variant, [f64; 7]); 4] = [
            (
                Pu21Variant::Banding,
                [
                    6.3053, 36.0057, 84.4045, 158.5061, 261.7517, 388.1423, 520.4673,
                ],
            ),
            (
                Pu21Variant::BandingGlare,
                [
                    0.3722, 5.7171, 36.5439, 123.6475, 256.3839, 420.0969, 595.3939,
                ],
            ),
            (
                Pu21Variant::Peaks,
                [
                    5.0060, 32.6568, 85.5420, 167.5246, 260.7250, 335.6947, 380.9853,
                ],
            ),
            (
                Pu21Variant::PeaksGlare,
                [
                    0.5133, 8.0104, 47.0090, 136.2603, 252.2985, 359.6225, 407.5066,
                ],
            ),
        ];
        for (variant, want_row) in rows {
            for (&yi, &wi) in y.iter().zip(want_row.iter()) {
                let got = pu21_encode(yi, variant) as f64;
                let tol = 0.1 + 5e-3 * wi;
                assert!(
                    (got - wi).abs() <= tol,
                    "{variant:?} PU21({yi}) = {got}, want {wi} ± {tol}"
                );
            }
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
