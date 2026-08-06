#![forbid(unsafe_code)]

//! HDR sweep mode (`zenmetrics sweep --hdr`): the gate for **all HDR
//! training-data collection**.
//!
//! SDR sweeps flow `decode_image_to_rgb8 → codec encode (RGB8) → decode-back
//! (RGB8) → u8 metric kernels`. Pushing a 16-bit PQ reference through that
//! pipeline silently quantises absolute-luminance code values to "8-bit
//! sRGB" — the imazen/zenmetrics#25 failure class (scores look plausible,
//! mean nothing). HDR mode replaces every stage:
//!
//! 1. **Reference decode** → [`HdrRef`]: 16-bit PQ-PNG (cICP transfer 16)
//!    via [`crate::hdr::png_to_rgb16_pq`] — raw PQ code values + cICP for
//!    re-encode, absolute nits (cd/m²) for scoring.
//! 2. **Encode** ([`encode_hdr`]): only codecs with a *true* HDR path are
//!    allowed ([`HdrCodec`]). Today that is **zenjxl** (16-bit PQ input +
//!    CICP signaling through the zencodec adapter → jxl-encoder's HDR
//!    input path), **zenavif** (16-bit PQ input → 10-bit identity-matrix
//!    AV1; the zencodec adapter maps the source CICP onto the container
//!    `nclx` via `apply_cicp_to_config → build_ravif_encoder`; added
//!    2026-07-12), **zenav1-svt** (10-bit BT.2020nc limited 4:2:0 still
//!    CQP via the byte-gated SVT-AV1 port, AVIF-muxed by
//!    zenavif-serialize; `hdr-svt` feature, added 2026-08-06) and
//!    **jpeg-gainmap** (Ultra HDR via `ultrahdr-rs`, HDR-only input with
//!    internal tonemap; `hdr-gainmap` feature, added 2026-08-06). The two
//!    new arms are fleet-path-only (no [`CodecKind`] variant); the sweep
//!    CLI still admits zenjxl/zenavif via [`validate_hdr_sweep`], and
//!    every other codec errors loudly at sweep start — an SDR 8-bit
//!    round-trip is never silently substituted.
//! 3. **Decode-back** ([`decode_encoded_to_nits`]): the encoded variant is
//!    decoded and must carry PQ signaling — the codestream CICP surfaced
//!    on `info.cicp` (transfer 16), or a PQ-tagged descriptor; samples →
//!    PQ EOTF → nits. A variant with neither errors — that means the
//!    codec dropped the HDR signaling and the cell is not an HDR cell.
//! 4. **Scoring** ([`score_hdr_cached`]): `zenmetrics_api::hdr::HdrScorer`
//!    applies the validated per-metric feeding (`hdr_feeding`: cvvdp /
//!    butteraugli linear planes, GPU ssim2 integrated PU21, iwssim float
//!    PU(luma), SSIM-family PU-rescale u8; dssim is Unsupported by
//!    design). Scorers are cached process-static, mirroring
//!    `metrics::cache::MetricCache`'s cubecl-pool discipline.
//!
//! The output TSV gains a trailing `hdr_mode` column (value `pq1000`:
//! PQ-decoded absolute nits scored at the 1000 cd/m² reference peak) so
//! downstream parquet/training joins can never confuse HDR rows with SDR
//! rows. SDR sweeps are byte-identical to before (no column added).

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value};

use crate::hdr::{HDR_DISPLAY_PEAK_NITS, NitsImage};
use crate::metrics::{GpuRuntime, MetricKind};
use crate::sweep::encode::{CodecKind, EncodedCell};

type Err = Box<dyn std::error::Error>;

/// The `hdr_mode` TSV column value for this mode: PQ-decoded absolute
/// nits, scored at the [`HDR_DISPLAY_PEAK_NITS`] (1000 cd/m²) reference
/// peak via the validated per-metric feedings.
pub const HDR_MODE_PQ1000: &str = "pq1000";

/// An HDR reference: raw 16-bit PQ code values (for codec HDR input) +
/// cICP (color authority for re-encode) + absolute-luminance nits (for
/// scoring). Decoded once per source image, shared across cells.
pub struct HdrRef {
    /// Tight interleaved RGB u16 PQ code values (`w*h*3`).
    pub rgb16: Vec<u16>,
    pub width: u32,
    pub height: u32,
    /// The source's cICP (transfer is always 16 = PQ here; primaries pass
    /// through — 1 and 12 both occur in the imazen-26-png-v2 corpus).
    pub cicp: zenpixels::Cicp,
    /// Absolute-luminance interleaved RGB (cd/m²), derived from `rgb16`
    /// via the PQ EOTF.
    pub nits: NitsImage,
}

/// Decode an HDR sweep reference. PQ-PNG (16-bit + cICP transfer 16) is
/// the only wired source format — it is what the HDR corpus
/// (`/mnt/v/output/imazen-26-png-v2/**/*.hdr.png`) contains.
pub fn decode_hdr_ref(path: &Path) -> Result<HdrRef, Err> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext != "png" {
        return Err(format!(
            "HDR sweep references must be PQ PNGs (.png with cICP transfer 16); \
             got .{ext}. EXR / gain-map sources are score-path-only today"
        )
        .into());
    }
    #[cfg(feature = "png")]
    {
        let data = std::fs::read(path)?;
        let (rgb16, width, height, cicp) = crate::hdr::png_to_rgb16_pq(&data)?;
        let nits = crate::hdr::rgb16_pq_to_nits(&rgb16, width, height);
        Ok(HdrRef {
            rgb16,
            width,
            height,
            cicp,
            nits,
        })
    }
    #[cfg(not(feature = "png"))]
    {
        Err("HDR sweep needs the `png` build feature (zenpng) for PQ-PNG references".into())
    }
}

/// Validate an HDR sweep configuration up front — every unsupported
/// combination errors **before** any encode runs, so a fleet chunk can
/// never silently degrade to SDR semantics.
pub fn validate_hdr_sweep(cfg: &crate::sweep::SweepConfig) -> Result<(), Err> {
    match cfg.codec {
        CodecKind::Zenjxl | CodecKind::Zenavif => {}
        other => {
            return Err(format!(
                "HDR sweep: codec {} has no HDR encode+decode path wired; \
                 supported today: zenjxl (16-bit PQ + CICP through the \
                 zencodec adapter) and zenavif (10-bit identity-matrix PQ, \
                 CICP → container nclx). Routing HDR refs through the SDR \
                 8-bit encode would fake the scores (imazen/zenmetrics#25 \
                 class), so it is refused rather than approximated. 16-bit \
                 zenpng is a candidate — see PLAN_SWEEPS.md 'HDR sweeps'",
                other.name()
            )
            .into());
        }
    }
    if cfg.plan.is_some() {
        return Err(
            "HDR sweep: --plan is not wired yet (plan cells encode via the \
                    RGB8-typed PlannedCell path); use --knob-grid"
                .into(),
        );
    }
    if cfg.feature_output.is_some() {
        return Err(
            "HDR sweep: --feature-output (zensim feature sidecar) is not \
                    wired — the feature extractors take u8 sRGB input"
                .into(),
        );
    }
    if cfg.distorted_out_dir.is_some() || cfg.pairs_tsv.is_some() {
        return Err(
            "HDR sweep: --distorted-out-dir / --pairs-tsv write 8-bit PNGs, \
                    which would crush nits output; not supported in HDR mode"
                .into(),
        );
    }
    if cfg.distort_cmd.is_some() {
        // The HDR distortion arm (2026-07-13): variants are 16-bit PQ PNGs
        // (protocol v2, u16 frames), persisted via --encoded-out-dir and
        // decode-backed through the SAME bytes. It needs an explicit row
        // label — falling back to the codec name would collide distortion
        // rows with real codec-encode rows for the same (ref, q) in
        // downstream joins (the corpus builder keys on basename+codec+q).
        if cfg.distort_label.as_deref().unwrap_or("").is_empty() {
            return Err("HDR sweep: --distort-cmd requires --distort-label <name> \
                 (e.g. kadis-hdr) — the omni/pairs codec column for \
                 distortion rows; a codec-name fallback would collide with \
                 real codec rows in downstream joins"
                .into());
        }
        #[cfg(not(feature = "png"))]
        return Err("HDR sweep: --distort-cmd needs the `png` build feature \
                    (zenpng) — distorted variants persist as 16-bit PQ PNGs"
            .into());
    }
    for &m in &cfg.metrics {
        // Resolve the umbrella mapping now so a metric with no HDR path
        // fails the whole sweep at startup instead of blanking every cell.
        umbrella_kind_and_backend(m, cfg.gpu_runtime)?;
        if matches!(m, MetricKind::Zensim) && cfg.feature_output.is_some() {
            return Err("HDR sweep: zensim feature emission is SDR-only".into());
        }
    }
    Ok(())
}

/// The codecs with a true HDR encode+decode-back story. Deliberately a
/// SEPARATE enum from [`CodecKind`]: the HDR arms include codecs that have
/// no SDR sweep path at all (`zenav1-svt`, `jpeg-gainmap`), and extending
/// `CodecKind` would force every SDR match site to carry variants it can
/// never encode. The fleet path (`jobexec` HDR encode jobs) parses these
/// by name; the `sweep --hdr` CLI reaches the [`CodecKind`]-backed subset
/// via [`HdrCodec::from_codec_kind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrCodec {
    /// zenjxl 16-bit PQ + CICP through the zencodec adapter.
    Zenjxl,
    /// zenavif 10-bit identity-matrix (MC=0 GBR 4:4:4) PQ via zenrav1e.
    Zenavif,
    /// zenav1-svt: the byte-gated pure-Rust SVT-AV1 port — 10-bit BT.2020nc
    /// limited-range 4:2:0 still CQP, muxed into AVIF by zenavif-serialize.
    Zenav1Svt,
    /// JPEG + gain map (Ultra HDR) via `ultrahdr-rs`: PQ → linear 203-nit
    /// white → internal tonemap → base JPEG + gain-map JPEG.
    JpegGainmap,
}

impl HdrCodec {
    /// Parse the cell-identity codec name (the `codec` column every join
    /// keys on). Unknown names error loudly — a typo'd arm must never
    /// silently encode as a different codec.
    pub fn from_name(name: &str) -> Result<Self, Err> {
        Ok(match name {
            "zenjxl" => HdrCodec::Zenjxl,
            "zenavif" => HdrCodec::Zenavif,
            "zenav1-svt" => HdrCodec::Zenav1Svt,
            "jpeg-gainmap" => HdrCodec::JpegGainmap,
            other => {
                return Err(format!(
                    "unknown HDR codec {other:?} (supported: zenjxl, zenavif, \
                     zenav1-svt, jpeg-gainmap)"
                )
                .into());
            }
        })
    }

    /// The cell-identity name (inverse of [`Self::from_name`]).
    pub fn name(self) -> &'static str {
        match self {
            HdrCodec::Zenjxl => "zenjxl",
            HdrCodec::Zenavif => "zenavif",
            HdrCodec::Zenav1Svt => "zenav1-svt",
            HdrCodec::JpegGainmap => "jpeg-gainmap",
        }
    }

    /// Map the sweep CLI's [`CodecKind`] onto the HDR arm. Only the codecs
    /// [`validate_hdr_sweep`] admits have a mapping; the svt/gainmap arms
    /// are fleet-path-only (no `CodecKind` variant exists for them).
    pub fn from_codec_kind(codec: CodecKind) -> Result<Self, Err> {
        match codec {
            CodecKind::Zenjxl => Ok(HdrCodec::Zenjxl),
            CodecKind::Zenavif => Ok(HdrCodec::Zenavif),
            other => Err(format!(
                "HDR sweep: codec {} has no HDR encode path (validate_hdr_sweep \
                 should have rejected this sweep)",
                other.name()
            )
            .into()),
        }
    }
}

/// Encode one HDR cell. The match stays exhaustive so a future codec
/// addition must consciously pick its HDR story.
pub fn encode_hdr(
    codec: HdrCodec,
    source: &HdrRef,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Err> {
    match codec {
        HdrCodec::Zenjxl => encode_jxl_hdr(source, q, knobs),
        HdrCodec::Zenavif => encode_avif_hdr(source, q, knobs),
        HdrCodec::Zenav1Svt => encode_svt_hdr(source, q, knobs),
        HdrCodec::JpegGainmap => encode_gainmap_hdr(source, q, knobs),
    }
}

/// Knobs the HDR JXL path consumes. Anything else errors loudly — a knob
/// silently ignored in HDR mode but honored in SDR mode would poison
/// cross-mode training joins.
#[cfg(all(feature = "sweep", feature = "jxl"))]
const JXL_HDR_KNOBS: &[&str] = &["lossless", "distance", "noise", "effort"];

/// zenjxl HDR encode: 16-bit PQ code values as a `RGB16` slice + the
/// source cICP as `Metadata` — the zencodec adapter maps CICP {16, 9|12}
/// to the JXL codestream color encoding (PQ + BT.2100/P3) and hands
/// jxl-encoder the u16 samples unconverted (`PixelLayout::Rgb16`).
/// The descriptor below is a layout carrier; `Metadata::cicp` is the
/// color authority (`resolve_jxl_color` reads only the metadata).
#[cfg(all(feature = "sweep", feature = "jxl"))]
fn encode_jxl_hdr(source: &HdrRef, q: f64, knobs: &Map<String, Value>) -> Result<EncodedCell, Err> {
    use std::time::Instant;
    use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
    use zenjxl::JxlEncoderConfig;
    use zenpixels::{PixelDescriptor, PixelSlice};

    if let Some(unknown) = knobs.keys().find(|k| !JXL_HDR_KNOBS.contains(&k.as_str())) {
        return Err(format!(
            "HDR sweep: zenjxl knob '{unknown}' is not wired in HDR mode \
             (supported: {JXL_HDR_KNOBS:?}); refusing to silently ignore it"
        )
        .into());
    }

    let lossless = knobs
        .get("lossless")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut cfg = if lossless {
        JxlEncoderConfig::new().with_lossless(true)
    } else {
        let mut c = JxlEncoderConfig::new().with_generic_quality(q as f32);
        if let Some(d) = knobs.get("distance").and_then(Value::as_f64) {
            c = c.with_distance(d as f32);
        }
        if let Some(b) = knobs.get("noise").and_then(Value::as_bool) {
            c = c.with_noise(b);
        }
        c
    };
    if let Some(e) = knobs.get("effort").and_then(Value::as_u64) {
        cfg = cfg.with_generic_effort(e.clamp(1, 10) as i32);
    }

    // Native-endian u16 → bytes without bytemuck: PixelSlice wants &[u8].
    let mut bytes = Vec::with_capacity(source.rgb16.len() * 2);
    for &v in &source.rgb16 {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    let stride = (source.width as usize) * 3 * 2;
    let slice = PixelSlice::new(
        &bytes,
        source.width,
        source.height,
        stride,
        PixelDescriptor::RGB16_BT2100_PQ,
    )
    .map_err(|e| format!("zenjxl hdr: pixel slice construction failed: {e}"))?;

    let meta = zencodec::Metadata::none().with_cicp(source.cicp);

    let start = Instant::now();
    // PreserveExact: the metadata here is only the source CICP (the HDR
    // color authority the adapter maps to the JXL codestream color
    // encoding) — nothing privacy-relevant to strip.
    let encoder = cfg
        .job()
        .with_metadata_policy(meta, zencodec::MetadataPolicy::PreserveExact)
        .encoder()
        .map_err(|e| format!("zenjxl hdr encoder construction failed: {e}"))?;
    let output = encoder
        .encode(slice)
        .map_err(|e| format!("zenjxl hdr encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell {
        bytes: output.into_vec(),
        encode_ms,
    })
}

#[cfg(not(all(feature = "sweep", feature = "jxl")))]
fn encode_jxl_hdr(
    _source: &HdrRef,
    _q: f64,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Err> {
    Err("HDR sweep: zenjxl requires building with --features sweep,jxl".into())
}

/// Knobs the HDR AVIF path consumes. `speed` mirrors the SDR arm's knob
/// (rav1e speed 1..=10); anything else errors loudly — a knob silently
/// ignored in HDR mode but honored in SDR mode would poison cross-mode
/// training joins.
#[cfg(all(feature = "sweep", feature = "avif"))]
const AVIF_HDR_KNOBS: &[&str] = &["lossless", "speed"];

/// zenavif HDR encode: 16-bit PQ code values as a `RGB16` slice + the
/// source cICP as `Metadata` — zenavif's zencodec adapter mirrors the
/// CICP triple onto its `EncoderConfig` (`apply_cicp_to_config`), which
/// `build_ravif_encoder` forwards to the container `nclx` (transfer 16 =
/// PQ; primaries pass through). The 16-bit slice routes to
/// `zenavif::encode_rgb16` → 10-bit **identity-matrix** (MC=0, GBR plane
/// order) AV1 — no YUV conversion, no chroma subsampling, matching the
/// PQ-PNG corpus's own MC=0 signaling.
#[cfg(all(feature = "sweep", feature = "avif"))]
fn encode_avif_hdr(
    source: &HdrRef,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Err> {
    use std::time::Instant;
    use zenavif::AvifEncoderConfig;
    use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
    use zenpixels::{PixelDescriptor, PixelSlice};

    if let Some(unknown) = knobs.keys().find(|k| !AVIF_HDR_KNOBS.contains(&k.as_str())) {
        return Err(format!(
            "HDR sweep: zenavif knob '{unknown}' is not wired in HDR mode \
             (supported: {AVIF_HDR_KNOBS:?}); refusing to silently ignore it"
        )
        .into());
    }

    let lossless = knobs
        .get("lossless")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut cfg = if lossless {
        AvifEncoderConfig::new().with_lossless(true)
    } else {
        AvifEncoderConfig::new().with_generic_quality(q as f32)
    };
    if let Some(s) = knobs.get("speed").and_then(Value::as_u64) {
        // Adapter effort is inverted rav1e speed: effort = 10 − speed.
        let speed = s.clamp(1, 10) as i32;
        cfg = cfg.with_generic_effort(10 - speed);
    }

    // Native-endian u16 → bytes without bytemuck: PixelSlice wants &[u8].
    let mut bytes = Vec::with_capacity(source.rgb16.len() * 2);
    for &v in &source.rgb16 {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    let stride = (source.width as usize) * 3 * 2;
    let slice = PixelSlice::new(
        &bytes,
        source.width,
        source.height,
        stride,
        PixelDescriptor::RGB16_BT2100_PQ,
    )
    .map_err(|e| format!("zenavif hdr: pixel slice construction failed: {e}"))?;

    let meta = zencodec::Metadata::none().with_cicp(source.cicp);

    let start = Instant::now();
    // PreserveExact: the metadata here is only the source CICP (the HDR
    // color authority the adapter maps onto the container nclx) —
    // nothing privacy-relevant to strip.
    let encoder = cfg
        .job()
        .with_metadata_policy(meta, zencodec::MetadataPolicy::PreserveExact)
        .encoder()
        .map_err(|e| format!("zenavif hdr encoder construction failed: {e}"))?;
    let output = encoder
        .encode(slice)
        .map_err(|e| format!("zenavif hdr encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell {
        bytes: output.into_vec(),
        encode_ms,
    })
}

#[cfg(not(all(feature = "sweep", feature = "avif")))]
fn encode_avif_hdr(
    _source: &HdrRef,
    _q: f64,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Err> {
    Err("HDR sweep: zenavif requires building with --features sweep,avif".into())
}

/// Knobs the zenav1-svt HDR path consumes. `preset` is the SVT preset
/// (0..=13, default 6 = the budget doc's quality tier); `qp` overrides the
/// q→QP mapping with an explicit CQP value. Anything else errors loudly.
#[cfg(feature = "hdr-svt")]
const SVT_HDR_KNOBS: &[&str] = &["preset", "qp"];

/// Map the generic cell quality (0..=100) onto the SVT CLI-domain QP
/// (1..=63, CQP). Linear and monotone (q0 → 63 worst, q100 → 1 best);
/// QP 0 (AV1 coded-lossless) is outside the port's envelope, so the top
/// end clamps to 1 rather than refusing q100 cells.
#[cfg(feature = "hdr-svt")]
fn svt_q_to_qp(q: f64) -> u8 {
    let q = q.clamp(0.0, 100.0);
    ((63.0 - q * 63.0 / 100.0).round() as i64).clamp(1, 63) as u8
}

/// PQ-coded RGB16 → BT.2020 NCL limited-range 10-bit YCbCr 4:2:0.
/// Standard non-constant-luminance matrix applied to the PQ-encoded
/// components (Kr=0.2627, Kb=0.0593), 2×2 chroma average in the C' domain,
/// then 10-bit limited quantization (Y: 64+876·Y', C: 512+896·C').
/// Ported verbatim from the 2026-08-05 HDR sweep-budget harness
/// (`hdrbudget-2026-08-05/harness/hdrbench/common/src/lib.rs`), which ran
/// this conversion at every ladder size — the shape an SVT-based AVIF HDR
/// pipeline uses (SVT-AV1 v4.2.0 ships 4:2:0 only; identity-matrix RGB
/// would need 4:4:4). The matrix choice is signaled (nclx MC=9), so the
/// decode side inverts exactly what was applied regardless of primaries.
#[cfg(feature = "hdr-svt")]
fn to_yuv420_bd10(rgb16: &[u16], w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    const KR: f64 = 0.2627;
    const KB: f64 = 0.0593;
    const KG: f64 = 1.0 - KR - KB;
    let n = w * h;
    let mut ynorm = vec![0f64; n];
    let mut y = vec![0u16; n];
    for i in 0..n {
        let r = rgb16[3 * i] as f64 / 65535.0;
        let g = rgb16[3 * i + 1] as f64 / 65535.0;
        let b = rgb16[3 * i + 2] as f64 / 65535.0;
        let yv = KR * r + KG * g + KB * b;
        ynorm[i] = yv;
        y[i] = ((876.0 * yv + 64.0).round() as i64).clamp(0, 1023) as u16;
    }
    let (cw, chd) = (w.div_ceil(2), h.div_ceil(2));
    let mut u = vec![0u16; cw * chd];
    let mut v = vec![0u16; cw * chd];
    for cy in 0..chd {
        for cx in 0..cw {
            let (mut sb, mut sr, mut cnt) = (0f64, 0f64, 0f64);
            for dy in 0..2 {
                for dx in 0..2 {
                    let (py, px) = (cy * 2 + dy, cx * 2 + dx);
                    if py >= h || px >= w {
                        continue;
                    }
                    let i = py * w + px;
                    let r = rgb16[3 * i] as f64 / 65535.0;
                    let b = rgb16[3 * i + 2] as f64 / 65535.0;
                    sb += (b - ynorm[i]) / 1.8814;
                    sr += (r - ynorm[i]) / 1.4746;
                    cnt += 1.0;
                }
            }
            u[cy * cw + cx] = ((896.0 * (sb / cnt) + 512.0).round() as i64).clamp(0, 1023) as u16;
            v[cy * cw + cx] = ((896.0 * (sr / cnt) + 512.0).round() as i64).clamp(0, 1023) as u16;
        }
    }
    (y, u, v)
}

/// zenav1-svt HDR encode: PQ code values → BT.2020nc limited 10-bit 4:2:0
/// → `EncodePipeline::try_encode_frame_420_hbd` (the byte-gated native
/// 10-bit still path, CQP) → AVIF container via zenavif-serialize with
/// nclx {source primaries, transfer 16 (PQ), matrix 9 (BT.2020 NCL),
/// limited range}. The OBU-level sequence header carries the same CICP,
/// so the signaling survives even if a reader ignores the container colr.
/// Decode-back goes through the ordinary AVIF path
/// ([`decode_avif_to_nits`]) — the output is a conformant AVIF.
#[cfg(feature = "hdr-svt")]
fn encode_svt_hdr(source: &HdrRef, q: f64, knobs: &Map<String, Value>) -> Result<EncodedCell, Err> {
    use std::time::Instant;
    use svtav1::encoder::pipeline::EncodePipeline;
    use svtav1::encoder::rate_control::{RcConfig, RcMode};

    if let Some(unknown) = knobs.keys().find(|k| !SVT_HDR_KNOBS.contains(&k.as_str())) {
        return Err(format!(
            "HDR sweep: zenav1-svt knob '{unknown}' is not wired in HDR mode \
             (supported: {SVT_HDR_KNOBS:?}); refusing to silently ignore it"
        )
        .into());
    }

    let preset = knobs
        .get("preset")
        .and_then(Value::as_u64)
        .unwrap_or(6)
        .clamp(0, 13) as u8;
    let qp = match knobs.get("qp").and_then(Value::as_u64) {
        Some(v) => v.clamp(1, 63) as u8,
        None => svt_q_to_qp(q),
    };

    let (w, h) = (source.width as usize, source.height as usize);
    let start = Instant::now();
    let (y, u, v) = to_yuv420_bd10(&source.rgb16, w, h);

    let rc = RcConfig {
        mode: RcMode::Cqp,
        qp,
        ..RcConfig::default()
    };
    let mut pipeline = EncodePipeline::new(w as u32, h as u32, preset, rc, 0, 1)
        .with_bit_depth(10)
        .with_tile_rows_log2(0)
        .with_tile_cols_log2(0)
        .with_sb_size(None)
        .with_chroma_420(true);
    // OBU-level CICP: primaries pass through from the source; matrix 9 is
    // what to_yuv420_bd10 applied; limited range matches its quantization.
    pipeline.color_description = svtav1::entropy::obu::ColorDescription {
        color_primaries: source.cicp.color_primaries,
        transfer_characteristics: 16,
        matrix_coefficients: 9,
        full_range: false,
    };
    let obu = pipeline
        .try_encode_frame_420_hbd(&y, &u, &v, w)
        .map_err(|e| format!("zenav1-svt hdr encode failed ({w}x{h} qp{qp} p{preset}): {e}"))?;

    let primaries = match source.cicp.color_primaries {
        1 => zenavif_serialize::constants::ColorPrimaries::Bt709,
        9 => zenavif_serialize::constants::ColorPrimaries::Bt2020,
        12 => zenavif_serialize::constants::ColorPrimaries::DisplayP3,
        other => {
            return Err(format!(
                "zenav1-svt hdr: source cICP primaries {other} has no nclx mapping \
                 (supported: 1=BT.709, 9=BT.2020, 12=Display P3)"
            )
            .into());
        }
    };
    let mut aviffy = zenavif_serialize::Aviffy::new();
    aviffy
        .set_color_primaries(primaries)
        .set_transfer_characteristics(
            zenavif_serialize::constants::TransferCharacteristics::Smpte2084,
        )
        .set_matrix_coefficients(zenavif_serialize::constants::MatrixCoefficients::Bt2020Ncl)
        .set_full_color_range(false);
    let bytes = aviffy.to_vec(&obu, None, source.width, source.height, 10);
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell { bytes, encode_ms })
}

#[cfg(not(feature = "hdr-svt"))]
fn encode_svt_hdr(
    _source: &HdrRef,
    _q: f64,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Err> {
    Err("HDR sweep: the zenav1-svt arm requires building with --features hdr-svt".into())
}

/// Knobs the JPEG-gainmap HDR path consumes. `gm_quality` is the gain-map
/// JPEG quality (default 85 = the ultrahdr-rs crate default the budget
/// harness swept with); `gm_scale` the gain-map downscale divisor (crate
/// default 4). The cell `q` drives the BASE JPEG quality. Anything else
/// errors loudly.
#[cfg(feature = "hdr-gainmap")]
const GAINMAP_HDR_KNOBS: &[&str] = &["gm_quality", "gm_scale"];

/// The gain-map quantization range, linear domain: `[1.0, 10000/203]`.
/// Mirrors the ultrahdr-rs HDR-only defaults (`gain_map_min` 1.0,
/// `target_display_peak` 10000 nits) so the full PQ range is expressible;
/// 8-bit granularity over log2(49.26) ≈ 5.62 stops is 0.022 stops/step.
#[cfg(feature = "hdr-gainmap")]
const GAINMAP_MAX_BOOST: f32 = 10000.0 / 203.0;

/// JPEG + gain map (Ultra HDR) HDR encode: PQ code values → PQ EOTF →
/// linear RGBA f32 at SDR-white 203 cd/m² = 1.0 (BT.2408, the crate's
/// LinearFloat convention) → filmic tonemap (the crate's own
/// `tonemap_image_to_srgb8`, the same derivation its HDR-only path uses)
/// → `ultrahdr_core::compute_gainmap` → base JPEG at the cell q +
/// gain-map JPEG, assembled by `ultrahdr_rs::Encoder`.
///
/// **Why this composes primitives instead of calling the crate's built-in
/// HDR-only path (`set_hdr_image` + `encode()` alone), measured
/// 2026-08-06:** that path quantizes gain-map bytes against the CONFIG
/// boost range (`compute_and_encode_gain`: `config.min_boost.ln()` /
/// `config.max_boost.ln()`) but stores metadata declaring the
/// content-derived ACTUAL range (`compute_gainmap`:
/// `log2(actual_min/max_boost)`), so every conformant reader — including
/// the crate's own `decode_hdr` at full weight — dequantizes on the wrong
/// grid and reconstructs under-boosted (a 2000-nit ramp came back at
/// 732 nits; the byte math reproduces the measurement exactly). Until
/// that is fixed upstream, this arm runs the SAME kernel and then
/// **rewrites the per-channel metadata min/max to the quantization grid
/// it was actually encoded on** (the config range — the libultrahdr
/// convention), which makes the stored bytes and the declared mapping
/// agree. The round-trip test below is the gate.
#[cfg(feature = "hdr-gainmap")]
fn encode_gainmap_hdr(
    source: &HdrRef,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Err> {
    use std::time::Instant;
    use ultrahdr_core::color::tonemap::tonemap_image_to_srgb8;
    use ultrahdr_core::gainmap::compute::{GainMapConfig, compute_gainmap};
    use ultrahdr_rs::{ColorPrimaries, PixelFormat, TransferFunction, pixel_buffer_from_vec};

    if let Some(unknown) = knobs
        .keys()
        .find(|k| !GAINMAP_HDR_KNOBS.contains(&k.as_str()))
    {
        return Err(format!(
            "HDR sweep: jpeg-gainmap knob '{unknown}' is not wired in HDR mode \
             (supported: {GAINMAP_HDR_KNOBS:?}); refusing to silently ignore it"
        )
        .into());
    }

    let base_q = (q.clamp(1.0, 100.0).round() as i64).clamp(1, 100) as u8;
    let gm_q = knobs
        .get("gm_quality")
        .and_then(Value::as_u64)
        .unwrap_or(85)
        .clamp(1, 100) as u8;
    let gm_scale = knobs
        .get("gm_scale")
        .and_then(Value::as_u64)
        .unwrap_or(4)
        .clamp(1, 8) as u8;

    let primaries = match source.cicp.color_primaries {
        1 => ColorPrimaries::Bt709,
        9 => ColorPrimaries::Bt2020,
        12 => ColorPrimaries::DisplayP3,
        other => {
            return Err(format!(
                "jpeg-gainmap hdr: source cICP primaries {other} has no \
                 ultrahdr mapping (supported: 1=BT.709, 9=BT.2020, 12=Display P3)"
            )
            .into());
        }
    };

    let (w, h) = (source.width as usize, source.height as usize);
    let start = Instant::now();
    // PQ code values → linear RGBA f32 with 1.0 = SDR white (203 nits).
    // Reuses the SAME nits the scoring path derived (`source.nits`) so the
    // encoder sees exactly the image the metrics see, divided by 203.
    let mut lin = vec![0u8; w * h * 16];
    for i in 0..w * h {
        let o = i * 16;
        lin[o..o + 4]
            .copy_from_slice(&(source.nits.rgb[3 * i] / crate::hdr::SDR_WHITE_NITS).to_le_bytes());
        lin[o + 4..o + 8].copy_from_slice(
            &(source.nits.rgb[3 * i + 1] / crate::hdr::SDR_WHITE_NITS).to_le_bytes(),
        );
        lin[o + 8..o + 12].copy_from_slice(
            &(source.nits.rgb[3 * i + 2] / crate::hdr::SDR_WHITE_NITS).to_le_bytes(),
        );
        lin[o + 12..o + 16].copy_from_slice(&1.0f32.to_le_bytes());
    }
    let hdr_buf = pixel_buffer_from_vec(
        lin,
        source.width,
        source.height,
        PixelFormat::RgbaF32,
        primaries,
        TransferFunction::Linear,
    )
    .map_err(|e| format!("jpeg-gainmap hdr: pixel buffer: {e:?}"))?;

    // SDR base: the crate's own tonemap derivation (filmic, BT.709 sRGB) —
    // byte-identical to what its HDR-only path would build internally.
    let sdr_pixels = tonemap_image_to_srgb8(&hdr_buf, ColorPrimaries::Bt709)
        .map_err(|e| format!("jpeg-gainmap hdr: tonemap: {e:?}"))?;
    let sdr_buf = pixel_buffer_from_vec(
        sdr_pixels,
        source.width,
        source.height,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )
    .map_err(|e| format!("jpeg-gainmap hdr: sdr buffer: {e:?}"))?;

    // Gain map on the crate's own kernel, with the crate-default HDR-only
    // config (single-channel luminance, gamma 1, 1/64 offsets).
    let config = GainMapConfig {
        scale_factor: gm_scale,
        gamma: 1.0,
        multi_channel: false,
        min_boost: 1.0,
        max_boost: GAINMAP_MAX_BOOST,
        base_offset: 1.0 / 64.0,
        alternate_offset: 1.0 / 64.0,
        base_hdr_headroom: 1.0,
        alternate_hdr_headroom: GAINMAP_MAX_BOOST,
    };
    let (gainmap, mut metadata) =
        compute_gainmap(&hdr_buf, &sdr_buf, &config, ultrahdr_core::Unstoppable)
            .map_err(|e| format!("jpeg-gainmap hdr: compute_gainmap: {e:?}"))?;
    // THE METADATA CORRECTION (see the function doc): declare the range the
    // bytes were actually quantized on. Headroom fields (weight policy) are
    // left as computed.
    for ch in metadata.channels.iter_mut() {
        ch.min = (config.min_boost as f64).log2();
        ch.max = (config.max_boost as f64).log2();
    }

    let mut enc = ultrahdr_rs::Encoder::new();
    enc.set_hdr_image(hdr_buf)
        .set_sdr_image(sdr_buf)
        .set_existing_gainmap(gainmap, metadata)
        .set_gainmap_scale(gm_scale)
        .set_quality(base_q, gm_q);
    let bytes = enc
        .encode()
        .map_err(|e| format!("jpeg-gainmap hdr encode failed ({w}x{h} q{base_q}): {e:?}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell { bytes, encode_ms })
}

#[cfg(not(feature = "hdr-gainmap"))]
fn encode_gainmap_hdr(
    _source: &HdrRef,
    _q: f64,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Err> {
    Err("HDR sweep: the jpeg-gainmap arm requires building with --features hdr-gainmap".into())
}

/// Decode an encoded HDR variant back to absolute nits. The decoded
/// descriptor must be PQ-tagged (the decoder enriches it from the
/// codestream CICP); anything else means the codec did not round-trip
/// the HDR signaling and the cell errors rather than crushing.
pub fn decode_encoded_to_nits(bytes: &[u8], codec: HdrCodec) -> Result<NitsImage, Err> {
    match codec {
        HdrCodec::Zenjxl => decode_jxl_to_nits(bytes),
        // The zenav1-svt arm emits a conformant AVIF (nclx PQ, 10-bit
        // 4:2:0 BT.2020nc limited) — the ordinary AVIF decode-back path
        // handles the matrix/range inversion and the PQ gate.
        HdrCodec::Zenavif | HdrCodec::Zenav1Svt => decode_avif_to_nits(bytes),
        HdrCodec::JpegGainmap => decode_gainmap_jpeg_to_nits(bytes),
    }
}

/// Decode an Ultra HDR JPEG variant back to nits: parse the gain map,
/// reconstruct HDR at FULL content boost (display boost = the PQ ceiling
/// 10000/203 — reconstruction weight saturates at the gain map's own
/// max-content-boost, so this recovers everything the codec stored rather
/// than simulating a limited display; the jxl/avif decode-backs impose no
/// display clamp either), then scale the 203-nit-white linear output to
/// absolute nits. A JPEG with no gain map errors — that means the encoder
/// dropped the HDR half and the cell is not an HDR cell.
#[cfg(feature = "hdr-gainmap")]
fn decode_gainmap_jpeg_to_nits(bytes: &[u8]) -> Result<NitsImage, Err> {
    crate::hdr::ultrahdr_jpeg_bytes_to_nits(bytes, crate::hdr::PQ_PEAK_BOOST)
}

#[cfg(not(feature = "hdr-gainmap"))]
fn decode_gainmap_jpeg_to_nits(_bytes: &[u8]) -> Result<NitsImage, Err> {
    Err("HDR sweep: jpeg-gainmap decode-back requires the `hdr-gainmap` build feature".into())
}

#[cfg(feature = "jxl")]
fn decode_jxl_to_nits(bytes: &[u8]) -> Result<NitsImage, Err> {
    let output = zenjxl::decode(bytes, None, &[]).map_err(|e| format!("zenjxl: {e}"))?;
    // The standalone `zenjxl::decode` surfaces the codestream's CICP on
    // `info.cicp` (the zencodec-adapter path additionally enriches the
    // pixel descriptor, but this path does not) — gate on either signal.
    let cicp_is_pq = matches!(output.info.cicp, Some((_, 16, _, _)));
    let desc_is_pq =
        output.pixels.as_slice().descriptor().transfer() == zenpixels::TransferFunction::Pq;
    if !cicp_is_pq && !desc_is_pq {
        return Err(format!(
            "HDR decode-back: decoded variant carries no PQ signaling \
             (info.cicp={:?}, descriptor transfer={:?}) — the codec did not \
             round-trip the HDR color encoding, so this is not an HDR variant \
             (refusing to guess a nits scale)",
            output.info.cicp,
            output.pixels.as_slice().descriptor().transfer(),
        )
        .into());
    }
    pq_slice_to_nits(&output.pixels.as_slice())
}

#[cfg(not(feature = "jxl"))]
fn decode_jxl_to_nits(_bytes: &[u8]) -> Result<NitsImage, Err> {
    Err("HDR sweep: zenjxl decode-back requires the `jxl` build feature".into())
}

/// Decode an AVIF variant back to nits. 10-bit samples are LSB-replicated
/// to u16 by the decoder (`scale_pixels_to_u16` — exact endpoint mapping,
/// so `v/65535` recovers the PQ code value). The container `nclx` must
/// say transfer 16 (PQ) — a variant that lost the signaling errors.
#[cfg(feature = "avif")]
fn decode_avif_to_nits(bytes: &[u8]) -> Result<NitsImage, Err> {
    let config = zenavif::DecoderConfig::default();
    // ManagedAvifDecoder = the safe (rav1d-safe) decoder, always available;
    // `AvifDecoder` is the unsafe-asm-gated FFI sibling.
    let mut dec = zenavif::ManagedAvifDecoder::new(bytes, &config)
        .map_err(|e| format!("zenavif decode-back: {e}"))?;
    let (buffer, info) = dec
        .decode_full(&enough::Unstoppable)
        .map_err(|e| format!("zenavif decode-back: {e}"))?;
    let tc = info.transfer_characteristics.0;
    if tc != 16 {
        return Err(format!(
            "HDR decode-back: AVIF variant carries transfer_characteristics \
             {tc} (want 16 = PQ) — the codec did not round-trip the HDR \
             color encoding (refusing to guess a nits scale)"
        )
        .into());
    }
    pq_slice_to_nits(&buffer.as_slice())
}

#[cfg(not(feature = "avif"))]
fn decode_avif_to_nits(_bytes: &[u8]) -> Result<NitsImage, Err> {
    Err("HDR sweep: zenavif decode-back requires the `avif` build feature".into())
}

/// Encode 16-bit PQ code values as a PNG-3 carrying `cicp` — the persisted
/// artifact shape for HDR **distortion** cells (`--distort-cmd`). The cell's
/// decode-back goes through [`decode_png_to_nits`] on these SAME bytes, so
/// what lands in the artifact store is exactly what got scored.
#[cfg(all(feature = "sweep", feature = "png"))]
pub(crate) fn encode_pq_png(
    rgb16: &[u16],
    width: u32,
    height: u32,
    cicp: zenpixels::Cicp,
) -> Result<Vec<u8>, Err> {
    use imgref::ImgRef;
    use rgb::Rgb;
    let px: &[Rgb<u16>] = bytemuck::cast_slice(rgb16);
    let img = ImgRef::new(px, width as usize, height as usize);
    let config = zenpng::EncodeConfig::default().with_cicp(Some(cicp));
    zenpng::encode_rgb16(
        img,
        None,
        &config,
        &enough::Unstoppable,
        &enough::Unstoppable,
    )
    .map_err(|e| format!("pq-png encode: {e}").into())
}

/// Decode a 16-bit PQ PNG variant back to absolute nits — the distortion
/// cells' decode-back. Same contract as every other decode-back: the PNG
/// must carry cICP (transfer 16 = PQ, or 18 = HLG which routes through the
/// HLG OOTF), else the cell errors rather than guessing a nits scale.
#[cfg(feature = "png")]
pub(crate) fn decode_png_to_nits(bytes: &[u8]) -> Result<NitsImage, Err> {
    let (rgb16, width, height, cicp) = crate::hdr::png_to_rgb16_pq(bytes)?;
    Ok(match cicp.transfer_characteristics {
        18 => crate::hdr::rgb16_hlg_to_nits(&rgb16, width, height),
        _ => crate::hdr::rgb16_pq_to_nits(&rgb16, width, height),
    })
}

#[cfg(not(feature = "png"))]
pub(crate) fn decode_png_to_nits(_bytes: &[u8]) -> Result<NitsImage, Err> {
    Err("HDR sweep: PQ-PNG decode-back requires the `png` build feature".into())
}

/// Strided **PQ-coded** `PixelSlice` (validated by the caller via the
/// codestream CICP / descriptor) → absolute nits via the PQ EOTF. u8 /
/// u16 samples normalise to `[0,1]` code values first; f32 samples ARE
/// the code values (the JXL decoder's f32 output is PQ-coded, not
/// linear, when the codestream carries CICP PQ). Alpha drops, gray
/// broadcasts.
#[cfg(any(feature = "jxl", feature = "avif"))]
fn pq_slice_to_nits(s: &zenpixels::PixelSlice<'_>) -> Result<NitsImage, Err> {
    use zenmetrics_api::hdr::pq_eotf;
    use zenpixels::{ChannelLayout, ChannelType};

    let desc = s.descriptor();
    let (w, h) = (s.width() as usize, s.rows() as usize);
    let channels: usize = match desc.layout() {
        ChannelLayout::Rgb => 3,
        ChannelLayout::Rgba => 4,
        ChannelLayout::Gray => 1,
        ChannelLayout::GrayAlpha => 2,
        other => {
            return Err(format!("HDR decode-back: unsupported channel layout {other:?}").into());
        }
    };
    let color_channels = if channels >= 3 { 3 } else { 1 };
    let bytes = s.as_strided_bytes();
    let stride = s.stride();
    let mut rgb = Vec::with_capacity(w * h * 3);
    let push_px = |rgb: &mut Vec<f32>, px: &[f32]| {
        if color_channels == 1 {
            let v = pq_eotf(px[0]);
            rgb.extend_from_slice(&[v, v, v]);
        } else {
            rgb.extend_from_slice(&[pq_eotf(px[0]), pq_eotf(px[1]), pq_eotf(px[2])]);
        }
    };
    match desc.channel_type() {
        ChannelType::U16 => {
            let row_bytes = w * channels * 2;
            for y in 0..h {
                let row = &bytes[y * stride..y * stride + row_bytes];
                let mut px = [0f32; 4];
                for (x, sample) in row.chunks_exact(2).enumerate() {
                    px[x % channels] =
                        f32::from(u16::from_ne_bytes([sample[0], sample[1]])) / 65535.0;
                    if x % channels == channels - 1 {
                        push_px(&mut rgb, &px);
                    }
                }
            }
        }
        ChannelType::U8 => {
            let row_bytes = w * channels;
            for y in 0..h {
                let row = &bytes[y * stride..y * stride + row_bytes];
                let mut px = [0f32; 4];
                for (x, &sample) in row.iter().enumerate() {
                    px[x % channels] = f32::from(sample) / 255.0;
                    if x % channels == channels - 1 {
                        push_px(&mut rgb, &px);
                    }
                }
            }
        }
        ChannelType::F32 => {
            let row_bytes = w * channels * 4;
            for y in 0..h {
                let row = &bytes[y * stride..y * stride + row_bytes];
                let mut px = [0f32; 4];
                for (x, sample) in row.chunks_exact(4).enumerate() {
                    px[x % channels] =
                        f32::from_ne_bytes([sample[0], sample[1], sample[2], sample[3]])
                            .clamp(0.0, 1.0);
                    if x % channels == channels - 1 {
                        push_px(&mut rgb, &px);
                    }
                }
            }
        }
        other => {
            return Err(format!("HDR decode-back: unsupported channel type {other:?}").into());
        }
    }
    Ok(NitsImage {
        rgb,
        width: w as u32,
        height: h as u32,
    })
}

// ─── Process-static HdrScorer cache ──────────────────────────────────────────

/// Map a CLI metric to its umbrella kind + backend for HDR sweep scoring.
/// Strict: metrics with no umbrella HDR mapping (feature off / hip / the
/// orchestrator-only kinds) error — the sweep never silently falls back
/// to a u8 path in HDR mode.
fn umbrella_kind_and_backend(
    metric: MetricKind,
    runtime: GpuRuntime,
) -> Result<(zenmetrics_api::MetricKind, zenmetrics_api::Backend), Err> {
    let kind = crate::hdr::to_umbrella_kind(metric).ok_or_else(|| {
        format!(
            "HDR sweep: metric {} has no umbrella HDR path in this build \
             (its gpu-* feature may be off)",
            metric.name()
        )
    })?;
    // dssim is Unsupported by design — fail at validation, not per cell.
    if matches!(kind, zenmetrics_api::MetricKind::Dssim) {
        return Err("HDR sweep: dssim has no HDR path by design (external \
                    dssim-core transform; u8 shell measured ~0.6 on UPIQ) — \
                    pick another metric"
            .into());
    }
    // The unsuffixed `cvvdp` / `iwssim` are now the native-CPU ports
    // (`requires_gpu() == false`), so they fall into the `else` (CPU) branch
    // below automatically — `HdrScorer::new` → `build_hdr_metric` routes
    // `Backend::Cpu` to `Metric::new_cpu_hdr` (native `cvvdp`/`iwssim` crate
    // via `cpu_dispatch`, NEVER cubecl-cpu), exactly like butter/ssim2/zensim.
    // The earlier `requires_gpu()`-lies workaround for cvvdp is gone now that
    // the backend label is honest. `cvvdp-gpu` / `iwssim-gpu` take the GPU
    // branch.
    let backend = if metric.requires_gpu() {
        if matches!(runtime, GpuRuntime::Auto) {
            return Err(format!(
                "HDR sweep: --gpu-runtime auto cannot be expanded for the HDR \
                 scorer cache — pass --gpu-runtime cuda or wgpu explicitly \
                 (metric {})",
                metric.name()
            )
            .into());
        }
        #[cfg(any(
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        ))]
        {
            match crate::metrics::gpu_runtime_to_backend(runtime) {
                Ok(b @ (zenmetrics_api::Backend::Cuda | zenmetrics_api::Backend::Wgpu)) => b,
                Ok(other) => {
                    return Err(format!(
                        "HDR sweep: GPU runtime {other:?} has no umbrella HDR path \
                         (cuda / wgpu only)"
                    )
                    .into());
                }
                Err(e) => return Err(format!("HDR sweep: {e}").into()),
            }
        }
        #[cfg(not(any(
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        )))]
        {
            // Unreachable in practice: GPU MetricKind variants are
            // cfg-gated on their gpu-* feature, so `requires_gpu()`
            // can't return true in a no-GPU build. Keep it compiling.
            return Err(format!(
                "HDR sweep: metric {} needs a gpu-* feature this build lacks",
                metric.name()
            )
            .into());
        }
    } else {
        zenmetrics_api::Backend::Cpu
    };
    Ok((kind, backend))
}

struct ScorerSlot {
    width: u32,
    height: u32,
    scorer: zenmetrics_api::hdr::HdrScorer,
}

/// Process-static HDR scorer cache, mirroring `MetricCache`'s discipline:
/// one warm instance per metric kind, rebuilt when the source dimensions
/// change. Keeping instances process-static bounds the cubecl pool
/// footprint across groups/chunks exactly like the SDR cache does.
static HDR_SCORERS: OnceLock<Mutex<HashMap<MetricKind, ScorerSlot>>> = OnceLock::new();

/// Score one HDR pair (absolute nits) with the validated per-metric
/// feeding, through the process-static scorer cache. Returns the same
/// `(column, value)` row shape the SDR scoring paths produce.
pub fn score_hdr_cached(
    metric: MetricKind,
    reference: &NitsImage,
    distorted: &NitsImage,
    runtime: GpuRuntime,
) -> Result<Vec<(&'static str, f64)>, Box<dyn Error>> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "{}: reference ({}×{}) and distorted ({}×{}) differ in size",
            metric.name(),
            reference.width,
            reference.height,
            distorted.width,
            distorted.height
        )
        .into());
    }
    let (kind, backend) = umbrella_kind_and_backend(metric, runtime)?;
    let mut cache = HDR_SCORERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poison| {
            // Same recovery posture as MetricCache::lock_global: a panic
            // inside one cell's scoring must not poison every later cell.
            poison.into_inner()
        });
    let needs_build = match cache.get(&metric) {
        Some(slot) => slot.width != reference.width || slot.height != reference.height,
        None => true,
    };
    if needs_build {
        // Drop the stale instance BEFORE constructing the replacement so
        // its pool slot is reusable (MetricCache does the same dance).
        cache.remove(&metric);
        let scorer = zenmetrics_api::hdr::HdrScorer::new(
            kind,
            backend,
            reference.width,
            reference.height,
            HDR_DISPLAY_PEAK_NITS,
        )?;
        cache.insert(
            metric,
            ScorerSlot {
                width: reference.width,
                height: reference.height,
                scorer,
            },
        );
    }
    let slot = cache.get_mut(&metric).expect("just inserted");
    let scores = slot.scorer.compute_multi(&reference.rgb, &distorted.rgb)?;
    let cols = metric.column_names();
    let rows: Vec<(&'static str, f64)> = if cols.len() >= 2 {
        // butteraugli: max + libjxl pnorm_3.
        vec![
            (
                cols[0],
                scores.get("max").unwrap_or_else(|| scores.primary()),
            ),
            (cols[1], scores.get("pnorm_3").unwrap_or(f64::NAN)),
        ]
    } else {
        vec![(cols[0], scores.primary())]
    };
    Ok(rows)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Non-JXL codecs and plan/sidecar options are rejected up front.
    #[test]
    fn validate_rejects_sdr_only_codecs_and_unwired_options() {
        let base = crate::sweep::SweepConfig {
            codec: CodecKind::Zenjpeg,
            sources: vec![],
            q_grid: vec![80.0],
            knob_grid: crate::sweep::parse_knob_grid("").expect("empty grid parses"),
            plan: None,
            distort_cmd: None,
            // Pre-existing test breakage on master e6e58bfd: SweepConfig gained
            // `distort_jobs` but this literal was never updated, so the
            // zenmetrics-cli test target did not compile. 1 = pipeline inactive
            // (matches `jobs: 1` / `distort_cmd: None` below). Unrelated to the
            // cvvdp/iwssim metric-registry change; added only to unblock tests.
            distort_jobs: 1,
            distort_label: None,
            metrics: vec![],
            gpu_runtime: GpuRuntime::Auto,
            output: std::path::PathBuf::from("/tmp/x.tsv"),
            feature_output: None,
            feature_regime: crate::metrics::ZensimFeatureRegime::WithIw,
            distorted_out_dir: None,
            encoded_out_dir: None,
            pairs_tsv: None,
            jobs: 1,
            hdr: true,
        };
        let err = validate_hdr_sweep(&base).unwrap_err().to_string();
        assert!(err.contains("no HDR encode+decode path"), "{err}");

        let mut jxl = base;
        jxl.codec = CodecKind::Zenjxl;
        jxl.feature_output = Some(std::path::PathBuf::from("/tmp/f.parquet"));
        let err = validate_hdr_sweep(&jxl).unwrap_err().to_string();
        assert!(err.contains("feature-output"), "{err}");
        jxl.feature_output = None;
        jxl.pairs_tsv = Some(std::path::PathBuf::from("/tmp/p.tsv"));
        let err = validate_hdr_sweep(&jxl).unwrap_err().to_string();
        assert!(err.contains("8-bit PNGs"), "{err}");
        jxl.pairs_tsv = None;
        assert!(validate_hdr_sweep(&jxl).is_ok());
    }

    /// Unknown knobs error instead of being silently dropped.
    #[cfg(feature = "jxl")]
    #[test]
    fn hdr_jxl_encode_rejects_unknown_knobs() {
        let src = HdrRef {
            rgb16: vec![0u16; 16 * 16 * 3],
            width: 16,
            height: 16,
            cicp: zenpixels::Cicp::new(9, 16, 0, true),
            nits: NitsImage {
                rgb: vec![0.0; 16 * 16 * 3],
                width: 16,
                height: 16,
            },
        };
        let mut knobs = Map::new();
        knobs.insert("progressive".into(), Value::Bool(true));
        let err = encode_hdr(HdrCodec::Zenjxl, &src, 80.0, &knobs)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not wired in HDR mode"), "{err}");
    }

    /// Codec-name mapping is exact and total over the four arms.
    #[test]
    fn hdr_codec_names_round_trip() {
        for c in [
            HdrCodec::Zenjxl,
            HdrCodec::Zenavif,
            HdrCodec::Zenav1Svt,
            HdrCodec::JpegGainmap,
        ] {
            assert_eq!(HdrCodec::from_name(c.name()).unwrap(), c);
        }
        assert!(
            HdrCodec::from_name("zenav1_svt").is_err(),
            "underscore typo must not parse"
        );
        assert!(HdrCodec::from_name("zenjpeg").is_err());
    }

    #[cfg(feature = "hdr-svt")]
    #[test]
    fn svt_q_to_qp_is_monotone_and_never_lossless() {
        assert_eq!(svt_q_to_qp(0.0), 63);
        assert_eq!(
            svt_q_to_qp(100.0),
            1,
            "q100 clamps to qp1 (qp0 = lossless is refused by the port)"
        );
        let mut prev = 64u8;
        for q in 0..=100 {
            let qp = svt_q_to_qp(q as f64);
            assert!(
                qp <= prev && (1..=63).contains(&qp),
                "q{q} -> qp{qp} prev {prev}"
            );
            prev = qp;
        }
    }

    /// Synthetic PQ source: a horizontal luminance ramp 0 → `peak_nits` with a
    /// mild vertical color gradient, PQ-inverse-EOTF'd to 16-bit code values.
    /// Even, non-64-aligned dims exercise the svt partial-SB bd10 path.
    #[cfg(any(feature = "hdr-svt", feature = "hdr-gainmap"))]
    pub(crate) fn synthetic_pq_ref(w: u32, h: u32, peak_nits: f64) -> HdrRef {
        fn pq_oetf(nits: f64) -> f64 {
            const M1: f64 = 2610.0 / 16384.0;
            const M2: f64 = 2523.0 / 4096.0 * 128.0;
            const C1: f64 = 3424.0 / 4096.0;
            const C2: f64 = 2413.0 / 4096.0 * 32.0;
            const C3: f64 = 2392.0 / 4096.0 * 32.0;
            let y = (nits / 10000.0).clamp(0.0, 1.0);
            let ym = y.powf(M1);
            ((C1 + C2 * ym) / (1.0 + C3 * ym)).powf(M2)
        }
        let (wu, hu) = (w as usize, h as usize);
        let mut rgb16 = vec![0u16; wu * hu * 3];
        for y in 0..hu {
            for x in 0..wu {
                let ramp = x as f64 / (wu - 1) as f64;
                let tint = 0.85 + 0.15 * (y as f64 / (hu - 1) as f64);
                let n_g = peak_nits * ramp;
                let (n_r, n_b) = (n_g * tint, n_g * (2.0 - tint) * 0.5);
                let i = (y * wu + x) * 3;
                rgb16[i] = (pq_oetf(n_r) * 65535.0).round() as u16;
                rgb16[i + 1] = (pq_oetf(n_g) * 65535.0).round() as u16;
                rgb16[i + 2] = (pq_oetf(n_b) * 65535.0).round() as u16;
            }
        }
        let nits = crate::hdr::rgb16_pq_to_nits(&rgb16, w, h);
        HdrRef {
            rgb16,
            width: w,
            height: h,
            cicp: zenpixels::Cicp::new(1, 16, 0, true),
            nits,
        }
    }

    /// Mean relative error of decoded vs source luma (BT.2020 Y in nits),
    /// over pixels whose source luma is above a small floor (relative error
    /// is meaningless at ~0 nits).
    #[cfg(any(feature = "hdr-svt", feature = "hdr-gainmap"))]
    fn mean_rel_luma_err(src: &NitsImage, dec: &NitsImage) -> f64 {
        assert_eq!((src.width, src.height), (dec.width, dec.height));
        let luma = |p: &[f32]| 0.2627 * p[0] as f64 + 0.6780 * p[1] as f64 + 0.0593 * p[2] as f64;
        let (mut acc, mut n) = (0.0f64, 0u64);
        for (s, d) in src.rgb.chunks_exact(3).zip(dec.rgb.chunks_exact(3)) {
            let (ls, ld) = (luma(s), luma(d));
            if ls > 5.0 {
                acc += (ld - ls).abs() / ls;
                n += 1;
            }
        }
        acc / n as f64
    }

    /// The zenav1-svt arm end-to-end: PQ ref → 10-bit BT.2020nc 4:2:0 CQP
    /// AVIF → the ORDINARY AVIF decode-back → nits. Gates: conformant
    /// container, PQ signaling survives, luma faithful at high q, and the
    /// rate responds to q. This is the pixels-are-sacred gate for the arm —
    /// it proves mux + matrix/range signaling + decode inversion agree.
    #[cfg(all(feature = "hdr-svt", feature = "avif"))]
    #[test]
    fn svt_arm_roundtrips_pq_avif_to_nits() {
        let src = synthetic_pq_ref(64, 48, 2000.0);
        let knobs = Map::new();
        let hi = encode_hdr(HdrCodec::Zenav1Svt, &src, 85.0, &knobs).expect("svt q85 encode");
        assert_eq!(&hi.bytes[4..8], b"ftyp", "must be an ISOBMFF container");
        let dec = decode_encoded_to_nits(&hi.bytes, HdrCodec::Zenav1Svt).expect("decode-back");
        assert_eq!((dec.width, dec.height), (64, 48));
        let err = mean_rel_luma_err(&src.nits, &dec);
        assert!(
            err < 0.10,
            "svt q85 mean relative luma error {err:.4} (expect < 10% on a smooth ramp)"
        );
        // Rate responds to quality (same source, much lower q).
        let lo = encode_hdr(HdrCodec::Zenav1Svt, &src, 10.0, &knobs).expect("svt q10 encode");
        assert!(
            lo.bytes.len() <= hi.bytes.len(),
            "q10 ({}) must not out-size q85 ({})",
            lo.bytes.len(),
            hi.bytes.len()
        );
        // Unknown knobs error instead of being silently dropped.
        let mut bad = Map::new();
        bad.insert("effort".into(), Value::from(7));
        let err = encode_hdr(HdrCodec::Zenav1Svt, &src, 50.0, &bad)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not wired in HDR mode"), "{err}");
    }

    /// The jpeg-gainmap arm end-to-end: PQ ref → Ultra HDR JPEG (internal
    /// tonemap) → gain-map reconstruction → nits. The >812-nit assertion is
    /// the anti-vacuity gate for [`crate::hdr::PQ_PEAK_BOOST`]: decoding the
    /// variant at the gain-map-source 4× display boost would clip the 2000-nit
    /// top of this ramp at ~812 and fail it.
    #[cfg(feature = "hdr-gainmap")]
    #[test]
    fn gainmap_arm_roundtrips_and_preserves_above_812_nits() {
        let src = synthetic_pq_ref(64, 48, 2000.0);
        let knobs = Map::new();
        let cell = encode_hdr(HdrCodec::JpegGainmap, &src, 90.0, &knobs).expect("uhdr q90 encode");
        assert_eq!(&cell.bytes[..2], &[0xFF, 0xD8], "must be a JPEG");
        let dec = decode_encoded_to_nits(&cell.bytes, HdrCodec::JpegGainmap).expect("decode-back");
        assert_eq!((dec.width, dec.height), (64, 48));
        let peak = dec
            .rgb
            .chunks_exact(3)
            .map(|p| 0.2627 * p[0] + 0.6780 * p[1] + 0.0593 * p[2])
            .fold(0.0f32, f32::max);
        assert!(
            peak > 900.0,
            "decoded peak luma {peak:.1} nits — a 4x-display-boost decode would clip at ~812"
        );
        let err = mean_rel_luma_err(&src.nits, &dec);
        assert!(
            err < 0.25,
            "uhdr q90 mean relative luma error {err:.4} (expect < 25% through tonemap+gainmap)"
        );
        // Unknown knobs error instead of being silently dropped.
        let mut bad = Map::new();
        bad.insert("distance".into(), Value::from(1.0));
        let err = encode_hdr(HdrCodec::JpegGainmap, &src, 50.0, &bad)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not wired in HDR mode"), "{err}");
    }
}
