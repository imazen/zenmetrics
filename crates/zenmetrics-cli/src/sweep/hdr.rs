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
//!    allowed. Today that is **zenjxl** (16-bit PQ input + CICP signaling
//!    through the zencodec adapter → jxl-encoder's HDR input path). Every
//!    other codec errors loudly at sweep start ([`validate_hdr_sweep`]) —
//!    an SDR 8-bit round-trip is never silently substituted.
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
        CodecKind::Zenjxl => {}
        other => {
            return Err(format!(
                "HDR sweep: codec {} has no HDR encode+decode path wired; \
                 supported today: zenjxl (16-bit PQ + CICP through the \
                 zencodec adapter). Routing HDR refs through the SDR 8-bit \
                 encode would fake the scores (imazen/zenmetrics#25 class), \
                 so it is refused rather than approximated. zenavif 10-bit \
                 PQ and 16-bit zenpng are candidates — see PLAN_SWEEPS.md \
                 'HDR sweeps'",
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

/// Encode one HDR cell. Only codecs validated by [`validate_hdr_sweep`]
/// arrive here; the match stays exhaustive so a future codec addition
/// must consciously pick its HDR story.
pub fn encode_hdr(
    codec: CodecKind,
    source: &HdrRef,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Err> {
    match codec {
        CodecKind::Zenjxl => encode_jxl_hdr(source, q, knobs),
        other => Err(format!(
            "HDR sweep: codec {} has no HDR encode path (validate_hdr_sweep \
             should have rejected this sweep)",
            other.name()
        )
        .into()),
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

/// Decode an encoded HDR variant back to absolute nits. The decoded
/// descriptor must be PQ-tagged (the decoder enriches it from the
/// codestream CICP); anything else means the codec did not round-trip
/// the HDR signaling and the cell errors rather than crushing.
pub fn decode_encoded_to_nits(bytes: &[u8], codec: CodecKind) -> Result<NitsImage, Err> {
    match codec {
        CodecKind::Zenjxl => decode_jxl_to_nits(bytes),
        other => Err(format!(
            "HDR sweep: codec {} has no HDR decode-back path",
            other.name()
        )
        .into()),
    }
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

/// Strided **PQ-coded** `PixelSlice` (validated by the caller via the
/// codestream CICP / descriptor) → absolute nits via the PQ EOTF. u8 /
/// u16 samples normalise to `[0,1]` code values first; f32 samples ARE
/// the code values (the JXL decoder's f32 output is PQ-coded, not
/// linear, when the codestream carries CICP PQ). Alpha drops, gray
/// broadcasts.
#[cfg(feature = "jxl")]
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
    // cvvdp is flagged `requires_gpu`, but it has a native CPU port. When its
    // GPU backend (`gpu-cvvdp`) is NOT compiled but the CPU port (`cpu-cvvdp`)
    // is, score it on `Backend::Cpu` — `HdrScorer::new` → `build_hdr_metric`
    // routes Cpu to `Metric::new_cpu_hdr` (native `cvvdp` crate via
    // `cpu_dispatch`, NEVER cubecl-cpu), exactly like butter/ssim2/zensim. This
    // makes `--metric cvvdp` work in a no-local-GPU `cpu-metrics` HDR sweep.
    #[cfg(all(not(feature = "gpu-cvvdp"), feature = "cpu-cvvdp"))]
    if matches!(metric, MetricKind::Cvvdp) {
        return Ok((kind, zenmetrics_api::Backend::Cpu));
    }
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
mod tests {
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
        let err = encode_hdr(CodecKind::Zenjxl, &src, 80.0, &knobs)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not wired in HDR mode"), "{err}");
    }
}
