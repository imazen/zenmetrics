#![forbid(unsafe_code)]

//! Codec encode dispatch for the sweep driver.
//!
//! Each codec gets its own `encode_*` function that takes the source RGB8
//! image, the integer quality, and the per-cell knob tuple JSON, and
//! returns the encoded bytes. The functions exist as a thin orchestration
//! layer — they translate JSON into typed builder calls and never modify
//! codec source.
//!
//! Knob coverage per codec (zenmetrics-cli 0.3.0):
//!
//! ## zenpng 0.1.4 (lossless; no `__expert` gate — all knobs are public)
//! `q` is ignored (PNG is lossless; for lossy-ish behaviour use
//! `near_lossless_bits`). Public builders: `compression` (u32 effort 0..=200,
//! or named preset string `"none"|"fastest"|"turbo"|"fast"|"balanced"|
//! "thorough"|"high"|"aggressive"|"intense"|"crush"|"maniac"|"brag"|
//! "minutes"`), `near_lossless_bits` (u8 0..=4 — rounds N LSBs per sample),
//! `parallel` (bool — multi-thread screening + refinement), `max_threads`
//! (u64; `0` = no limit, `1` = fully single-threaded). `filter` only has
//! one variant (`Auto`) so it isn't exposed.
//!
//! ## zenjpeg 0.8.4 (`__expert` + `trellis` enabled, path dep on sibling repo)
//! Public builders: `subsampling` (`"444"` / `"422"` / `"420"` / `"440"`),
//! `progressive` (bool), `sharp_yuv` (bool), `effort` (u8, clamped to
//! 0..=2 by zenjpeg's generic-effort API). Quality is fed through
//! `with_generic_quality(q as f32)` so cross-codec sweeps see calibrated
//! quality on the same scale as the other zen codecs.
//!
//! Expert (via `EncoderConfig::with_internal_params`):
//! `optimize_huffman` (bool), `aq_enabled` (bool), `deringing` (bool),
//! `auto_optimize` (bool, trellis-feature), `chroma_distance_scale` (f32,
//! clamped 0.1..=5.0 by builder), `pre_blur` (f32 sigma),
//! `quant_source` (string: `"jpegli"` / `"mozjpeg_default"`),
//! `progressive_mode` (string: `"baseline"` / `"progressive"` /
//! `"progressive_mozjpeg"` / `"progressive_search"` — richer alternative
//! to the bool `progressive` knob, kept for back-compat),
//! `huffman` (string: `"optimize"` / `"fixed"` / `"fixed_annex_k"` —
//! richer than `optimize_huffman`),
//! `tiny_file_mode` (string: `"auto"` / `"off"` / `"force"`),
//! `downsampling_method` (string: `"box"` / `"gamma_aware"` /
//! `"gamma_aware_iterative"`),
//! `restart_mcu_rows` (u16, 0 disables),
//! `chroma_quality` (u64 → `Some(Some(q as u8))`, null clears),
//! `optimization` (string preset: `"jpegli_baseline"` /
//! `"jpegli_progressive"` / `"mozjpeg_baseline"` / `"mozjpeg_progressive"` /
//! `"mozjpeg_max_compression"` / `"hybrid_baseline"` /
//! `"hybrid_progressive"` / `"hybrid_max_compression"`),
//! `trellis` (bool or object — `TrellisConfig` fields `lambda1` /
//! `lambda2` / `dc` / `delta_dc_weight` / `speed` plus the AQ-coupling
//! group `coupling_*`; replaces the removed `hybrid` knob, which now
//! errors with a migration hint),
//! `progressive_mode` also accepts `"smallest"` / `"smallest_search"`
//! (zenjpeg's exact entropy-stage minimizers).
//!
//! **Plan-driven zenjpeg sweeps** (`--plan rd_core|modes_full`
//! [`--plan-budget N`]) bypass the JSON knob grid entirely: cells come
//! from `zenjpeg::encode::sweep` (curated provenance-stamped axes,
//! fingerprint dedup, validity filtering, main-effects-first ordering,
//! budget ladder). See `super::plan`.
//!
//! ## zenwebp 0.4.5 (`__expert` enabled)
//! Public builders: `method`, `segments`, `sns_strength`, `filter_strength`,
//! `lossless`. Expert (via `LossyConfig::with_internal_params`):
//! `partition_limit`, `multi_pass_stats`, `smooth_segment_map`, `sharp_yuv`
//! (`"off"` / `"on"`).
//!
//! ## zenavif 0.1.7 (`__expert` enabled)
//! Public builders: `speed` (0..=10), `lossless`. Expert (via
//! `EncoderConfig::with_internal_params`): `partition_range` (`[min, max]`
//! pair, members in {4, 8, 16, 32, 64}), `complex_prediction_modes`,
//! `lrf`, `fast_deblock`.
//!
//! ## zenjxl 0.2.1 (`__expert` enabled, but `JxlEncoderConfig` does not
//! currently expose `with_internal_params` for the lossy path — internal
//! knobs live on `jxl_encoder::LossyConfig` and are not reachable from
//! the wrapper). The wrapper handles: `effort` (1..=10), `lossless`,
//! `noise`, `distance`. For lossy, the sweep also drops down to
//! `jxl_encoder::LossyConfig` directly when any expert knob is present:
//! `butteraugli_iters`, `zensim_iters`, `ssim2_iters`, `pixel_domain_loss`,
//! `patches`, `gaborish`, `error_diffusion`, `denoise`, `lf_frame`,
//! `force_strategy` (u8 or null), `progressive` ("off"/"dc-only"/"two-pass").
//! These bypass the macro-knob `effort` bundling so the picker can compose
//! independent decisions (e.g. effort=5 DCT search + 2 butteraugli iters
//! without LZ77).
//!
//! Adding a codec or knob is a local change to this file: extend the
//! match arms, document it in the module header, and add a test under
//! `tests/cli.rs` if the encoder behaviour is observable.

use crate::decode::Rgb8Image;
use serde_json::{Map, Value};
use std::error::Error;
use std::time::Instant;

/// Codec selector for the sweep CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CodecKind {
    /// `zenpng` lossless PNG encoder.
    Zenpng,
    /// `zenjpeg` JPEG encoder (jpegli-style).
    Zenjpeg,
    /// `zenwebp` lossy/lossless WebP encoder.
    Zenwebp,
    /// `zenavif` AV1-still encoder via ravif.
    Zenavif,
    /// `zenjxl` JPEG XL encoder via jxl-encoder.
    Zenjxl,
    /// `zengif` quantizer-driven GIF encoder (plan-only sweeps).
    Zengif,
    /// `zentiff` lossless TIFF encoder (plan-only sweeps).
    Zentiff,
}

impl CodecKind {
    pub fn name(self) -> &'static str {
        match self {
            CodecKind::Zenpng => "zenpng",
            CodecKind::Zenjpeg => "zenjpeg",
            CodecKind::Zenwebp => "zenwebp",
            CodecKind::Zenavif => "zenavif",
            CodecKind::Zenjxl => "zenjxl",
            CodecKind::Zengif => "zengif",
            CodecKind::Zentiff => "zentiff",
        }
    }
}

/// Encoded output bundle.
#[derive(Debug)]
pub struct EncodedCell {
    pub bytes: Vec<u8>,
    pub encode_ms: f64,
}

// ── Recognized knob names per codec ──────────────────────────────────────
//
// Every name a codec's `encode_*` reads MUST appear in its list, and any
// knob NOT in the list is a hard `encode()` error (see
// [`reject_unknown_knobs`]). A silently-ignored knob is the worst failure
// mode for a source-informing sweep: a cell labelled `{"xyb":true}` whose
// encoder quietly drops the unknown key and emits the YCbCr default writes
// thousands of mislabelled rows into the training parquet that look correct.
// When you add a `knobs.get("foo")` to an `encode_*`, add `"foo"` here.
const PNG_KNOBS: &[&str] = &[
    "compression",
    "near_lossless_bits",
    "parallel",
    "max_threads",
    "quantize",
];
const JPEG_KNOBS: &[&str] = &[
    // public builders
    "subsampling",
    "progressive",
    "sharp_yuv",
    "effort",
    // XYB color-mode axes (EncoderConfig::xyb)
    "xyb",
    "xyb_subsampling",
    // expert / InternalParams
    "optimize_huffman",
    "aq_enabled",
    "deringing",
    "auto_optimize",
    "chroma_distance_scale",
    "pre_blur",
    "quant_source",
    "progressive_mode",
    "huffman",
    "tiny_file_mode",
    "downsampling_method",
    "restart_mcu_rows",
    "chroma_quality",
    "optimization",
    "trellis",
    // Tombstone: recognized so encode_jpeg can emit the migration error
    // pointing at "trellis" (HybridConfig was removed from zenjpeg).
    "hybrid",
];
const WEBP_KNOBS: &[&str] = &[
    "method",
    "segments",
    "sns_strength",
    "filter_strength",
    "partition_limit",
    "multi_pass_stats",
    "smooth_segment_map",
    "sharp_yuv",
];
const AVIF_KNOBS: &[&str] = &[
    "speed",
    "lossless",
    "partition_range",
    "lrf",
    "fast_deblock",
];
const JXL_KNOBS: &[&str] = &[
    "distance",
    "noise",
    "effort",
    "denoise",
    "gaborish",
    "patches",
    "pixel_domain_loss",
    "error_diffusion",
    "lf_frame",
    "lz77",
    "butteraugli_iters",
    "zensim_iters",
    "ssim2_iters",
    "force_strategy",
    "max_strategy_size",
    "progressive",
];

// zengif / zentiff are wired for plan-driven sweeps only (`--plan`),
// where cells come from the codec's own planner; they have no JSON
// `--knob-grid` vocabulary, so their recognized-knob set is empty and
// the JSON `encode()` dispatch routes them to a plan-only error.
const GIF_KNOBS: &[&str] = &[];
const TIFF_KNOBS: &[&str] = &[];

impl CodecKind {
    /// The set of knob names this codec recognizes.
    fn recognized_knobs(self) -> &'static [&'static str] {
        match self {
            CodecKind::Zenpng => PNG_KNOBS,
            CodecKind::Zenjpeg => JPEG_KNOBS,
            CodecKind::Zenwebp => WEBP_KNOBS,
            CodecKind::Zenavif => AVIF_KNOBS,
            CodecKind::Zenjxl => JXL_KNOBS,
            CodecKind::Zengif => GIF_KNOBS,
            CodecKind::Zentiff => TIFF_KNOBS,
        }
    }
}

/// Reject any knob the codec does not recognize. An unknown knob is a hard
/// error rather than a silent no-op so a typo or an unsupported axis can
/// never masquerade as the encoder default in the output parquet.
fn reject_unknown_knobs(
    codec: CodecKind,
    knobs: &Map<String, Value>,
) -> Result<(), Box<dyn Error>> {
    let allowed = codec.recognized_knobs();
    for key in knobs.keys() {
        if !allowed.contains(&key.as_str()) {
            let mut names: Vec<&str> = allowed.to_vec();
            names.sort_unstable();
            return Err(format!(
                "unknown {} knob {key:?}; recognized knobs: {}",
                codec.name(),
                names.join(", ")
            )
            .into());
        }
    }
    Ok(())
}

/// Encode `source` with `codec`, quality `q`, and knob assignment `knobs`.
/// Errors propagate as `Box<dyn Error>` so the sweep runner can record a
/// per-cell failure without halting the rest of the grid.
pub fn encode(
    codec: CodecKind,
    source: &Rgb8Image,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    reject_unknown_knobs(codec, knobs)?;
    match codec {
        CodecKind::Zenpng => encode_png(source, q, knobs),
        CodecKind::Zenjpeg => encode_jpeg(source, q, knobs),
        CodecKind::Zenwebp => encode_webp(source, q, knobs),
        CodecKind::Zenavif => encode_avif(source, q, knobs),
        CodecKind::Zenjxl => encode_jxl(source, q, knobs),
        // gif/tiff are plan-only: their cells come from the codec's own
        // sweep planner (`--plan`), never from a `--knob-grid` product.
        CodecKind::Zengif | CodecKind::Zentiff => Err(format!(
            "{} sweeps are plan-only: use --plan rd_core|modes_full|scalar_dense \
             (no --knob-grid vocabulary is defined for {})",
            codec.name(),
            codec.name()
        )
        .into()),
    }
}

// ── zenpng ──────────────────────────────────────────────────────────────

#[cfg(all(feature = "sweep", feature = "png"))]
fn encode_png(
    source: &Rgb8Image,
    _q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    use enough::Unstoppable;
    use imgref::ImgRef;
    use zenpng::{Compression, EncodeConfig};

    // PNG is lossless — `q` is ignored. For lossy-ish behaviour, use the
    // `near_lossless_bits` knob (rounds N LSBs per sample, 0..=4).

    let mut cfg = EncodeConfig::default();

    // Compression: accepts a numeric effort (0..=200) or one of the named
    // presets. Numeric form is preferred for sweep grids since it's
    // continuous; the preset strings are accepted for grid readability.
    if let Some(v) = knobs.get("compression") {
        let comp = match v {
            Value::Number(n) => Compression::Effort(n.as_u64().unwrap_or(13).min(200) as u32),
            Value::String(s) => match s.as_str() {
                "none" => Compression::None,
                "fastest" => Compression::Fastest,
                "turbo" => Compression::Turbo,
                "fast" => Compression::Fast,
                "balanced" => Compression::Balanced,
                "thorough" => Compression::Thorough,
                "high" => Compression::High,
                "aggressive" => Compression::Aggressive,
                "intense" => Compression::Intense,
                "crush" => Compression::Crush,
                "maniac" => Compression::Maniac,
                "brag" => Compression::Brag,
                "minutes" => Compression::Minutes,
                other => {
                    return Err(format!(
                        "zenpng compression must be a 0..=200 effort or one of \
                         none|fastest|turbo|fast|balanced|thorough|high|aggressive|\
                         intense|crush|maniac|brag|minutes; got {other:?}"
                    )
                    .into());
                }
            },
            _ => {
                return Err(
                    "zenpng compression must be a number (0..=200) or preset string".into(),
                );
            }
        };
        cfg = cfg.with_compression(comp);
    }

    if let Some(b) = knobs.get("near_lossless_bits").and_then(Value::as_u64) {
        cfg = cfg.with_near_lossless_bits(b.min(4) as u8);
    }
    if let Some(b) = knobs.get("parallel").and_then(Value::as_bool) {
        cfg = cfg.with_parallel(b);
    }
    if let Some(t) = knobs.get("max_threads").and_then(Value::as_u64) {
        cfg.max_threads = t as usize;
    }

    // Build an ImgRef<Rgb<u8>> over the source buffer without copying.
    let pixels: &[rgb::Rgb<u8>] = bytemuck_cast_rgb(&source.pixels);
    let img = ImgRef::new(pixels, source.width as usize, source.height as usize);

    // Optional palette/quantize knob: "iq{N}" = imagequant, "zq{N}" =
    // zenquant, N = max colors (2..=256). Present → indexed (palette) PNG;
    // absent → truecolor lossless. This is the one lossy PNG axis.
    let quantize = match knobs.get("quantize") {
        None => None,
        Some(Value::String(s)) => Some(parse_png_quantize(s)?),
        Some(_) => {
            return Err("zenpng quantize must be a string like \"iq256\" or \"zq64\"".into());
        }
    };

    let start = Instant::now();
    let bytes = match quantize {
        None => zenpng::encode_rgb8(img, None, &cfg, &Unstoppable, &Unstoppable)
            .map_err(|e| format!("zenpng encode failed: {e}"))?,
        Some((backend, max_colors)) => {
            use rgb::Rgba;
            // Widen RGB → RGBA (opaque) for the indexed encoder.
            let rgba: Vec<Rgba<u8>> = img
                .pixels()
                .map(|p| Rgba::new(p.r, p.g, p.b, 255u8))
                .collect();
            let rgba_img = ImgRef::new(&rgba, img.width(), img.height());
            let quantizer: Box<dyn zenpng::Quantizer> = match backend {
                PngQuantBackend::Imagequant => {
                    Box::new(zenpng::ImagequantQuantizer::default().with_max_colors(max_colors))
                }
                PngQuantBackend::Zenquant => {
                    Box::new(zenpng::ZenquantQuantizer::new().with_max_colors(max_colors))
                }
            };
            zenpng::encode_indexed(
                rgba_img,
                &cfg,
                &*quantizer,
                None,
                &Unstoppable,
                &Unstoppable,
            )
            .map_err(|e| format!("zenpng indexed encode failed: {e}"))?
        }
    };
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell { bytes, encode_ms })
}

/// Palette backend selected by the `quantize` knob.
#[cfg(all(feature = "sweep", feature = "png"))]
#[derive(Clone, Copy)]
enum PngQuantBackend {
    Imagequant,
    Zenquant,
}

/// Parse a `quantize` knob value (`"iq{N}"` / `"zq{N}"`, N = 2..=256) into
/// a (backend, max_colors) pair. Mirrors `zenpng::sweep`'s cell-id grammar.
#[cfg(all(feature = "sweep", feature = "png"))]
fn parse_png_quantize(s: &str) -> Result<(PngQuantBackend, u16), Box<dyn Error>> {
    let (backend, n) = if let Some(n) = s.strip_prefix("iq") {
        (PngQuantBackend::Imagequant, n)
    } else if let Some(n) = s.strip_prefix("zq") {
        (PngQuantBackend::Zenquant, n)
    } else {
        return Err(format!(
            "zenpng quantize must start with `iq` (imagequant) or `zq` (zenquant); got {s:?}"
        )
        .into());
    };
    let max_colors: u16 = n
        .parse()
        .map_err(|e| format!("zenpng quantize color count in {s:?}: {e}"))?;
    if !(2..=256).contains(&max_colors) {
        return Err(format!("zenpng quantize colors {max_colors} out of range 2..=256").into());
    }
    Ok((backend, max_colors))
}

#[cfg(not(all(feature = "sweep", feature = "png")))]
fn encode_png(
    _source: &Rgb8Image,
    _q: f64,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    Err("zenpng encode is disabled (rebuild with `--features sweep,png`)".into())
}

// ── zenjpeg ─────────────────────────────────────────────────────────────

/// Parse the `"trellis"` knob: `true` → `TrellisConfig::default()`;
/// an object overrides individual fields on top of the default
/// (`lambda1`, `lambda2`, `dc`, `delta_dc_weight`, `speed`
/// ("thorough" | "adaptive" | integer level 0..=10), and the AQ-coupling
/// group `coupling_scale` / `coupling_exponent` / `coupling_threshold` /
/// `coupling_max_adjustment` / `coupling_chroma_mul`). A nonzero
/// `coupling_scale` is the old "hybrid" coupling; zenjpeg's curated
/// sweep steps clamp it via `coupling_max_adjustment` (unclamped
/// coupling is a validated quality-destruction mode on high-AQ content).
#[cfg(all(feature = "sweep", feature = "jpeg"))]
fn parse_trellis_knob(
    v: &Value,
) -> Result<zenjpeg::encode::trellis::TrellisConfig, Box<dyn Error>> {
    use zenjpeg::encode::trellis::{AqCoupling, TrellisConfig, TrellisSpeedMode};
    let mut t = TrellisConfig::default();
    match v {
        Value::Bool(true) => Ok(t),
        Value::Bool(false) => {
            t.enabled = false;
            Ok(t)
        }
        Value::Object(o) => {
            if let Some(x) = o.get("lambda1").and_then(Value::as_f64) {
                t.lambda_log_scale1 = x as f32;
            }
            if let Some(x) = o.get("lambda2").and_then(Value::as_f64) {
                t.lambda_log_scale2 = x as f32;
            }
            if let Some(b) = o.get("dc").and_then(Value::as_bool) {
                t.dc_enabled = b;
            }
            if let Some(x) = o.get("delta_dc_weight").and_then(Value::as_f64) {
                t.delta_dc_weight = x as f32;
            }
            match o.get("speed") {
                None => {}
                Some(Value::String(s)) => {
                    t.speed_mode = match s.as_str() {
                        "thorough" => TrellisSpeedMode::Thorough,
                        "adaptive" => TrellisSpeedMode::Adaptive,
                        other => {
                            return Err(format!(
                                "zenjpeg trellis.speed must be thorough|adaptive|0..=10; \
                                 got {other:?}"
                            )
                            .into());
                        }
                    };
                }
                Some(Value::Number(n)) => {
                    let level = n
                        .as_u64()
                        .ok_or("zenjpeg trellis.speed level must be an integer 0..=10")?;
                    t.speed_mode = TrellisSpeedMode::Level(level.min(10) as u8);
                }
                Some(other) => {
                    return Err(format!(
                        "zenjpeg trellis.speed must be thorough|adaptive|0..=10; got {other}"
                    )
                    .into());
                }
            }
            let mut coupling = AqCoupling::OFF;
            if let Some(x) = o.get("coupling_scale").and_then(Value::as_f64) {
                coupling.scale = x as f32;
            }
            if let Some(x) = o.get("coupling_exponent").and_then(Value::as_f64) {
                coupling.exponent = x as f32;
            }
            if let Some(x) = o.get("coupling_threshold").and_then(Value::as_f64) {
                coupling.threshold = x as f32;
            }
            if let Some(x) = o.get("coupling_max_adjustment").and_then(Value::as_f64) {
                coupling.max_adjustment = x as f32;
            }
            if let Some(x) = o.get("coupling_chroma_mul").and_then(Value::as_f64) {
                coupling.chroma_mul = x as f32;
            }
            t.aq_coupling = coupling;
            Ok(t)
        }
        other => Err(format!("zenjpeg trellis must be a bool or an object; got {other}").into()),
    }
}

#[cfg(all(feature = "sweep", feature = "jpeg"))]
fn encode_jpeg(
    source: &Rgb8Image,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
    use zenjpeg::JpegEncoderConfig;
    use zenjpeg::encode::encoder_types::QuantTableSource;
    use zenjpeg::encoder::{
        ChromaSubsampling, ColorMode, DownsamplingMethod, EncoderConfig as ZenEncoderConfig,
        HuffmanStrategy, InternalParams, OptimizationPreset, PixelLayout as ZenPixelLayout,
        ProgressiveScanMode, TinyFileMode, XybSubsampling,
    };
    use zenpixels::{PixelDescriptor, PixelSlice};

    // Quality flows through the cross-codec generic scale so the sweep
    // grid produces comparable-looking output across zen codecs.
    let mut cfg = JpegEncoderConfig::new().with_generic_quality(q as f32);

    // ── Public builders (applied first; internal params can override). ──

    // Subsampling — accept the four ratio strings zenjpeg supports.
    if let Some(s) = knobs.get("subsampling").and_then(Value::as_str) {
        let sub = match s {
            "444" | "4:4:4" => ChromaSubsampling::None,
            "422" | "4:2:2" => ChromaSubsampling::HalfHorizontal,
            "420" | "4:2:0" => ChromaSubsampling::Quarter,
            "440" | "4:4:0" => ChromaSubsampling::HalfVertical,
            other => {
                return Err(format!(
                    "zenjpeg subsampling must be one of 444|422|420|440; got {other:?}"
                )
                .into());
            }
        };
        cfg = cfg.with_subsampling(sub);
    }

    if let Some(b) = knobs.get("progressive").and_then(Value::as_bool) {
        cfg = cfg.with_progressive(b);
    }
    if let Some(b) = knobs.get("sharp_yuv").and_then(Value::as_bool) {
        cfg = cfg.with_sharp_yuv(b);
    }
    if let Some(e) = knobs.get("effort").and_then(Value::as_u64) {
        // zenjpeg's `with_effort_range(0, 2)` clamps internally, but we
        // mirror the pattern used by the other codecs and apply the
        // clamp here for clarity in the sweep grid.
        cfg = cfg.with_generic_effort(e.min(2) as i32);
    }

    // ── Expert knobs (InternalParams). Build only when at least one is
    // present, so the default codepath is exercised as-is when absent.

    let mut params = InternalParams::default();
    let mut any_internal = false;

    if let Some(b) = knobs.get("optimize_huffman").and_then(Value::as_bool) {
        params.optimize_huffman = Some(b);
        any_internal = true;
    }
    if let Some(b) = knobs.get("aq_enabled").and_then(Value::as_bool) {
        params.aq_enabled = Some(b);
        any_internal = true;
    }
    if let Some(b) = knobs.get("deringing").and_then(Value::as_bool) {
        params.deringing = Some(b);
        any_internal = true;
    }
    #[cfg(feature = "sweep")]
    if let Some(b) = knobs.get("auto_optimize").and_then(Value::as_bool) {
        // Trellis feature is a hard transitive enable in
        // zenmetrics-cli's sweep feature, so this field exists.
        params.auto_optimize = Some(b);
        any_internal = true;
    }
    if let Some(s) = knobs.get("chroma_distance_scale").and_then(Value::as_f64) {
        params.chroma_distance_scale = Some(s as f32);
        any_internal = true;
    }
    if let Some(s) = knobs.get("pre_blur").and_then(Value::as_f64) {
        params.pre_blur = Some(s as f32);
        any_internal = true;
    }
    if let Some(s) = knobs.get("quant_source").and_then(Value::as_str) {
        let qs = match s {
            "jpegli" => QuantTableSource::Jpegli,
            "mozjpeg_default" | "mozjpeg" => QuantTableSource::MozjpegDefault,
            other => {
                return Err(format!(
                    "zenjpeg quant_source must be \"jpegli\" or \"mozjpeg_default\"; got {other:?}"
                )
                .into());
            }
        };
        params.quant_source = Some(qs);
        any_internal = true;
    }
    if let Some(s) = knobs.get("progressive_mode").and_then(Value::as_str) {
        let mode = match s {
            "baseline" => ProgressiveScanMode::Baseline,
            "progressive" => ProgressiveScanMode::Progressive,
            "progressive_mozjpeg" | "two_scan" | "mozjpeg" => {
                ProgressiveScanMode::ProgressiveMozjpeg
            }
            "progressive_search" | "search" => ProgressiveScanMode::ProgressiveSearch,
            "smallest" => ProgressiveScanMode::Smallest,
            "smallest_search" => ProgressiveScanMode::SmallestSearch,
            other => {
                return Err(format!(
                    "zenjpeg progressive_mode must be one of \
                     baseline|progressive|progressive_mozjpeg|progressive_search|\
                     smallest|smallest_search; got {other:?}"
                )
                .into());
            }
        };
        params.progressive = Some(mode);
        any_internal = true;
    }
    if let Some(s) = knobs.get("huffman").and_then(Value::as_str) {
        let h = match s {
            "optimize" => HuffmanStrategy::Optimize,
            "fixed" => HuffmanStrategy::Fixed,
            "fixed_annex_k" | "annex_k" => HuffmanStrategy::FixedAnnexK,
            other => {
                return Err(format!(
                    "zenjpeg huffman must be one of optimize|fixed|fixed_annex_k; \
                     got {other:?}"
                )
                .into());
            }
        };
        params.huffman = Some(h);
        any_internal = true;
    }
    if let Some(s) = knobs.get("tiny_file_mode").and_then(Value::as_str) {
        let m = match s {
            "auto" => TinyFileMode::Auto,
            "off" => TinyFileMode::Off,
            "force" => TinyFileMode::Force,
            other => {
                return Err(format!(
                    "zenjpeg tiny_file_mode must be one of auto|off|force; got {other:?}"
                )
                .into());
            }
        };
        params.tiny_file_mode = Some(m);
        any_internal = true;
    }
    if let Some(s) = knobs.get("downsampling_method").and_then(Value::as_str) {
        let m = match s {
            "box" => DownsamplingMethod::Box,
            "gamma_aware" => DownsamplingMethod::GammaAware,
            "gamma_aware_iterative" | "iterative" => DownsamplingMethod::GammaAwareIterative,
            other => {
                return Err(format!(
                    "zenjpeg downsampling_method must be one of \
                     box|gamma_aware|gamma_aware_iterative; got {other:?}"
                )
                .into());
            }
        };
        params.downsampling_method = Some(m);
        any_internal = true;
    }
    if let Some(n) = knobs.get("restart_mcu_rows").and_then(Value::as_u64) {
        params.restart_mcu_rows = Some(n.min(u16::MAX as u64) as u16);
        any_internal = true;
    }
    if let Some(v) = knobs.get("chroma_quality") {
        match v {
            Value::Null => {
                // explicit null → clear override (revert to luma quality)
                params.chroma_quality = Some(None);
                any_internal = true;
            }
            Value::Number(n) => {
                if let Some(q) = n.as_u64() {
                    params.chroma_quality = Some(Some(q.min(100) as u8));
                    any_internal = true;
                }
            }
            _ => {}
        }
    }
    if let Some(s) = knobs.get("optimization").and_then(Value::as_str) {
        let preset = match s {
            "jpegli_baseline" => OptimizationPreset::JpegliBaseline,
            "jpegli_progressive" => OptimizationPreset::JpegliProgressive,
            #[cfg(feature = "sweep")]
            "mozjpeg_baseline" => OptimizationPreset::MozjpegBaseline,
            #[cfg(feature = "sweep")]
            "mozjpeg_progressive" => OptimizationPreset::MozjpegProgressive,
            #[cfg(feature = "sweep")]
            "mozjpeg_max_compression" => OptimizationPreset::MozjpegMaxCompression,
            #[cfg(feature = "sweep")]
            "hybrid_baseline" => OptimizationPreset::HybridBaseline,
            #[cfg(feature = "sweep")]
            "hybrid_progressive" => OptimizationPreset::HybridProgressive,
            #[cfg(feature = "sweep")]
            "hybrid_max_compression" => OptimizationPreset::HybridMaxCompression,
            other => {
                return Err(format!(
                    "zenjpeg optimization must be one of \
                     jpegli_baseline|jpegli_progressive|mozjpeg_baseline|\
                     mozjpeg_progressive|mozjpeg_max_compression|\
                     hybrid_baseline|hybrid_progressive|hybrid_max_compression; \
                     got {other:?}"
                )
                .into());
            }
        };
        params.optimization = Some(preset);
        any_internal = true;
    }
    // `"trellis"` replaces the removed `"hybrid"` knob: zenjpeg merged
    // hybrid into `TrellisConfig` (AQ-coupled lambda lives in
    // `aq_coupling`; `scale == 0` is the classic standalone trellis).
    if let Some(v) = knobs.get("trellis") {
        params.trellis = Some(parse_trellis_knob(v)?);
        any_internal = true;
    }
    if knobs.contains_key("hybrid") {
        return Err("zenjpeg knob \"hybrid\" was removed along with zenjpeg's \
             HybridConfig; use \"trellis\" (true, or an object with lambda1/\
             lambda2/dc/delta_dc_weight/speed/coupling_scale/coupling_exponent/\
             coupling_threshold/coupling_max_adjustment/coupling_chroma_mul — \
             coupling_scale != 0 is the old hybrid coupling). Plan-driven \
             sweeps (--plan) cover the curated trellis/coupling steps."
            .into());
    }

    // ── XYB color mode ──────────────────────────────────────────────────
    // XYB is a distinct color path. We do NOT flip a flag on a fresh config:
    // we reuse `cfg` (which already carries the wrapper's generic-quality
    // mapping — the SAME `Quality::ApproxJpegli(calibrated_jpeg_quality(q))`
    // / `Quality::Zq(q)` the YCbCr cells get — plus progressive / sharp_yuv /
    // effort) and switch only the color mode to XYB via the public builder.
    // That keeps an XYB cell on the exact same quality scale as the YCbCr
    // cells in this sweep. `allow_16bit_quant_tables(false)` matches
    // `EncoderConfig::xyb`'s default (XYB forces SOF1 separately for the DC
    // categories; 16-bit DQT buys nothing). The luma `subsampling` knob does
    // not apply — XYB subsampling is the B-channel `xyb_subsampling` — and
    // `effort` is already folded into `cfg` by the wrapper builder above.
    if knobs.get("xyb").and_then(Value::as_bool) == Some(true) {
        let b_sub = match knobs.get("xyb_subsampling").and_then(Value::as_str) {
            None | Some("bquarter") | Some("b_quarter") | Some("quarter") => {
                XybSubsampling::BQuarter
            }
            Some("full") => XybSubsampling::Full,
            Some(other) => {
                return Err(format!(
                    "zenjpeg xyb_subsampling must be bquarter|full; got {other:?}"
                )
                .into());
            }
        };
        let mut inner: ZenEncoderConfig = cfg.inner().clone();
        inner = inner
            .color_mode(ColorMode::Xyb { subsampling: b_sub })
            .allow_16bit_quant_tables(false);
        if any_internal {
            inner = inner.with_internal_params(params);
        }
        let start = Instant::now();
        let bytes = inner
            .encode_bytes(
                &source.pixels,
                source.width,
                source.height,
                ZenPixelLayout::Rgb8Srgb,
            )
            .map_err(|e| format!("zenjpeg xyb encode failed: {e}"))?;
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
        return Ok(EncodedCell { bytes, encode_ms });
    }

    if any_internal {
        // `JpegEncoderConfig::encode` applies an effort-derived
        // `OptimizationPreset` at encode time, which clobbers fields like
        // `optimization`, `progressive`, `huffman`, `aq_enabled`,
        // `deringing`, and `quant_table_config`. To preserve every
        // expert knob the caller set, drop down to the inner
        // `EncoderConfig` and call `encode_bytes` directly. The 4 public
        // wrapper knobs (subsampling / progressive / sharp_yuv / effort)
        // were already mirrored into `cfg.inner()` by the wrapper's
        // builder methods, so we clone that as the starting point.
        let mut inner: ZenEncoderConfig = cfg.inner().clone();
        inner = inner.with_internal_params(params);

        let start = Instant::now();
        let bytes = inner
            .encode_bytes(
                &source.pixels,
                source.width,
                source.height,
                ZenPixelLayout::Rgb8Srgb,
            )
            .map_err(|e| format!("zenjpeg expert encode failed: {e}"))?;
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

        return Ok(EncodedCell { bytes, encode_ms });
    }

    let stride = (source.width as usize) * 3;
    let slice = PixelSlice::new(
        &source.pixels,
        source.width,
        source.height,
        stride,
        PixelDescriptor::RGB8_SRGB,
    )
    .map_err(|e| format!("zenjpeg: pixel slice construction failed: {e}"))?;

    let start = Instant::now();
    let encoder = cfg
        .job()
        .encoder()
        .map_err(|e| format!("zenjpeg encoder construction failed: {e}"))?;
    let output = encoder
        .encode(slice)
        .map_err(|e| format!("zenjpeg encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell {
        bytes: output.into_vec(),
        encode_ms,
    })
}

#[cfg(not(all(feature = "sweep", feature = "jpeg")))]
fn encode_jpeg(
    _source: &Rgb8Image,
    _q: f64,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    Err("zenjpeg encode is disabled (rebuild with `--features sweep,jpeg`)".into())
}

// ── zenwebp ─────────────────────────────────────────────────────────────

#[cfg(all(feature = "sweep", feature = "webp"))]
fn encode_webp(
    source: &Rgb8Image,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    use zenwebp::{EncodeRequest, EncoderConfig, LossyConfig, PixelLayout};

    let lossless = knobs
        .get("lossless")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let cfg = if lossless {
        EncoderConfig::new_lossless().with_quality(q as f32)
    } else {
        let mut lossy = LossyConfig::new().with_quality(q as f32);
        if let Some(m) = knobs.get("method").and_then(Value::as_u64) {
            lossy = lossy.with_method(m.min(6) as u8);
        }
        if let Some(s) = knobs.get("segments").and_then(Value::as_u64) {
            lossy = lossy.with_segments(s.clamp(1, 4) as u8);
        }
        if let Some(s) = knobs.get("sns_strength").and_then(Value::as_u64) {
            lossy = lossy.with_sns_strength(s.min(100) as u8);
        }
        if let Some(s) = knobs.get("filter_strength").and_then(Value::as_u64) {
            lossy = lossy.with_filter_strength(s.min(100) as u8);
        }
        // Expert knobs — only built when at least one is present, so the
        // default codepath is exercised as-is when all knobs are absent.
        #[cfg(feature = "sweep")]
        {
            let mut params = zenwebp::InternalParams::default();
            let mut any = false;
            if let Some(v) = knobs.get("partition_limit").and_then(Value::as_u64) {
                params.partition_limit = Some(v.min(100) as u8);
                any = true;
            }
            if let Some(v) = knobs.get("multi_pass_stats").and_then(Value::as_bool) {
                params.multi_pass_stats = Some(v);
                any = true;
            }
            if let Some(v) = knobs.get("smooth_segment_map").and_then(Value::as_bool) {
                params.smooth_segment_map = Some(v);
                any = true;
            }
            if let Some(v) = knobs.get("sharp_yuv").and_then(Value::as_str) {
                params.sharp_yuv = Some(match v {
                    "off" => zenwebp::SharpYuvSetting::Off,
                    "on" => zenwebp::SharpYuvSetting::On,
                    other => {
                        return Err(format!(
                            "zenwebp sharp_yuv must be \"off\" or \"on\"; got {other:?}"
                        )
                        .into());
                    }
                });
                any = true;
            }
            if any {
                lossy = lossy.with_internal_params(params);
            }
        }
        EncoderConfig::Lossy(lossy)
    };

    let start = Instant::now();
    let bytes = EncodeRequest::new(
        &cfg,
        &source.pixels,
        PixelLayout::Rgb8,
        source.width,
        source.height,
    )
    .encode()
    .map_err(|e| format!("zenwebp encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell { bytes, encode_ms })
}

#[cfg(not(all(feature = "sweep", feature = "webp")))]
fn encode_webp(
    _source: &Rgb8Image,
    _q: f64,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    Err("zenwebp encode is disabled (rebuild with `--features sweep`)".into())
}

// ── zenavif ─────────────────────────────────────────────────────────────

#[cfg(all(feature = "sweep", feature = "avif"))]
fn encode_avif(
    source: &Rgb8Image,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    use imgref::ImgRef;
    use zenavif::EncoderConfig;

    // Build an ImgRef<Rgb<u8>> over the source buffer without copying.
    let pixels: &[rgb::Rgb<u8>] = bytemuck_cast_rgb(&source.pixels);
    let img = ImgRef::new(pixels, source.width as usize, source.height as usize);

    let mut cfg = EncoderConfig::new().quality(q as f32);
    if let Some(s) = knobs.get("speed").and_then(Value::as_u64) {
        cfg = cfg.speed(s.min(10) as u8);
    }
    if let Some(b) = knobs.get("lossless").and_then(Value::as_bool) {
        cfg = cfg.with_lossless(b);
    }
    // Expert knobs.
    #[cfg(feature = "sweep")]
    {
        let mut params = zenavif::expert::InternalParams::default();
        let mut any = false;
        if let Some(arr) = knobs.get("partition_range").and_then(Value::as_array)
            && arr.len() == 2
        {
            let min = arr[0].as_u64().unwrap_or(0) as u8;
            let max = arr[1].as_u64().unwrap_or(0) as u8;
            params.partition_range = Some((min, max));
            any = true;
        }
        if let Some(v) = knobs
            .get("complex_prediction_modes")
            .and_then(Value::as_bool)
        {
            params.complex_prediction_modes = Some(v);
            any = true;
        }
        if let Some(v) = knobs.get("lrf").and_then(Value::as_bool) {
            params.lrf = Some(v);
            any = true;
        }
        if let Some(v) = knobs.get("fast_deblock").and_then(Value::as_bool) {
            params.fast_deblock = Some(v);
            any = true;
        }
        if any {
            cfg = cfg.with_internal_params(params);
        }
    }

    let start = Instant::now();
    let encoded = zenavif::encode_rgb8(
        img,
        &cfg,
        almost_enough::StopToken::new(enough::Unstoppable),
    )
    .map_err(|e| format!("zenavif encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell {
        bytes: encoded.avif_file,
        encode_ms,
    })
}

#[cfg(not(all(feature = "sweep", feature = "avif")))]
fn encode_avif(
    _source: &Rgb8Image,
    _q: f64,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    Err("zenavif encode is disabled (rebuild with `--features sweep`)".into())
}

// ── zenjxl ──────────────────────────────────────────────────────────────

/// Expert lossy knobs that, if any are present, route encode through
/// `jxl_encoder::LossyConfig` directly instead of the zenjxl wrapper.
#[cfg(all(feature = "sweep", feature = "jxl"))]
const JXL_EXPERT_KNOBS: &[&str] = &[
    "butteraugli_iters",
    "zensim_iters",
    "ssim2_iters",
    "pixel_domain_loss",
    "patches",
    "gaborish",
    "error_diffusion",
    "denoise",
    "lf_frame",
    "force_strategy",
    "max_strategy_size",
    "progressive",
    "lz77",
];

#[cfg(all(feature = "sweep", feature = "jxl"))]
fn encode_jxl(
    source: &Rgb8Image,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    let lossless = knobs
        .get("lossless")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // Expert lossy path: any of the new knobs trigger a direct LossyConfig
    // build that bypasses the wrapper's effort macro-knob. Lossless stays
    // on the wrapper; the new knobs don't apply to modular mode.
    if !lossless && knobs.keys().any(|k| JXL_EXPERT_KNOBS.contains(&k.as_str())) {
        return encode_jxl_expert(source, q, knobs);
    }

    encode_jxl_wrapper(source, q, knobs, lossless)
}

#[cfg(all(feature = "sweep", feature = "jxl"))]
fn encode_jxl_wrapper(
    source: &Rgb8Image,
    q: f64,
    knobs: &Map<String, Value>,
    lossless: bool,
) -> Result<EncodedCell, Box<dyn Error>> {
    use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
    use zenjxl::JxlEncoderConfig;
    use zenpixels::{PixelDescriptor, PixelSlice};

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

    let stride = (source.width as usize) * 3;
    let slice = PixelSlice::new(
        &source.pixels,
        source.width,
        source.height,
        stride,
        PixelDescriptor::RGB8_SRGB,
    )
    .map_err(|e| format!("zenjxl: pixel slice construction failed: {e}"))?;

    let start = Instant::now();
    let encoder = cfg
        .job()
        .encoder()
        .map_err(|e| format!("zenjxl encoder construction failed: {e}"))?;
    let output = encoder
        .encode(slice)
        .map_err(|e| format!("zenjxl encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell {
        bytes: output.into_vec(),
        encode_ms,
    })
}

/// Direct `LossyConfig` path for expert knobs the wrapper doesn't expose.
/// Decouples the macro-knob `effort` from independent decisions like
/// `butteraugli_iters`, `pixel_domain_loss`, `patches`, etc.
#[cfg(all(feature = "sweep", feature = "jxl"))]
fn encode_jxl_expert(
    source: &Rgb8Image,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    use jxl_encoder::{LossyConfig, PixelLayout, ProgressiveMode};

    // Resolve distance: explicit `distance` knob wins, else fall back to
    // the generic q→distance mapping zenjxl uses.
    let distance = if let Some(d) = knobs.get("distance").and_then(Value::as_f64) {
        d as f32
    } else {
        jxl_encoder::quality_to_distance(jxl_encoder::calibrated_jxl_quality(q as f32))
    };

    let mut cfg = LossyConfig::new(distance);

    if let Some(e) = knobs.get("effort").and_then(Value::as_u64) {
        cfg = cfg.with_effort(e.clamp(1, 10) as u8);
    }
    if let Some(b) = knobs.get("noise").and_then(Value::as_bool) {
        cfg = cfg.with_noise(b);
    }
    if let Some(b) = knobs.get("denoise").and_then(Value::as_bool) {
        cfg = cfg.with_denoise(b);
    }
    if let Some(b) = knobs.get("gaborish").and_then(Value::as_bool) {
        cfg = cfg.with_gaborish(b);
    }
    if let Some(b) = knobs.get("patches").and_then(Value::as_bool) {
        cfg = cfg.with_patches(b);
    }
    if let Some(b) = knobs.get("pixel_domain_loss").and_then(Value::as_bool) {
        cfg = cfg.with_pixel_domain_loss(b);
    }
    if let Some(b) = knobs.get("error_diffusion").and_then(Value::as_bool) {
        cfg = cfg.with_error_diffusion(b);
    }
    if let Some(b) = knobs.get("lf_frame").and_then(Value::as_bool) {
        cfg = cfg.with_lf_frame(b);
    }
    if let Some(b) = knobs.get("lz77").and_then(Value::as_bool) {
        cfg = cfg.with_lz77(b);
    }
    if let Some(n) = knobs.get("butteraugli_iters").and_then(Value::as_u64) {
        cfg = cfg.with_butteraugli_iters(n.min(16) as u32);
    }
    if let Some(n) = knobs.get("zensim_iters").and_then(Value::as_u64) {
        cfg = cfg.with_zensim_iters(n.min(16) as u32);
    }
    if let Some(n) = knobs.get("ssim2_iters").and_then(Value::as_u64) {
        cfg = cfg.with_ssim2_iters(n.min(16) as u32);
    }
    // `force_strategy` accepts null (unset) or u8 in 0..=18 (DCT strategy id).
    if let Some(v) = knobs.get("force_strategy") {
        match v {
            Value::Null => cfg = cfg.with_force_strategy(None),
            Value::Number(n) => {
                if let Some(s) = n.as_u64() {
                    cfg = cfg.with_force_strategy(Some(s.min(18) as u8));
                }
            }
            _ => {}
        }
    }
    if let Some(v) = knobs.get("max_strategy_size") {
        match v {
            Value::Null => cfg = cfg.with_max_strategy_size(None),
            Value::Number(n) => {
                if let Some(s) = n.as_u64() {
                    cfg = cfg.with_max_strategy_size(Some(s.min(255) as u8));
                }
            }
            _ => {}
        }
    }
    if let Some(s) = knobs.get("progressive").and_then(Value::as_str) {
        let mode = match s {
            "off" | "single" => ProgressiveMode::Single,
            "two-pass" | "two_pass" | "twopass" | "qac-fac" => ProgressiveMode::QuantizedAcFullAc,
            "three-pass" | "three_pass" | "threepass" | "dc-vlf-lf-ac" => {
                ProgressiveMode::DcVlfLfAc
            }
            other => {
                return Err(format!(
                    "zenjxl: unknown progressive mode '{other}' (want single|two-pass|three-pass)"
                )
                .into());
            }
        };
        cfg = cfg.with_progressive(mode);
    }

    let stride = (source.width as usize) * 3;
    if source.pixels.len() < stride * source.height as usize {
        return Err("zenjxl expert: pixel buffer shorter than width*height*3".into());
    }

    let start = Instant::now();
    let bytes = cfg
        .encode(
            &source.pixels,
            source.width,
            source.height,
            PixelLayout::Rgb8,
        )
        .map_err(|e| format!("zenjxl expert encode failed: {e}"))?;
    let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(EncodedCell { bytes, encode_ms })
}

#[cfg(not(all(feature = "sweep", feature = "jxl")))]
fn encode_jxl(
    _source: &Rgb8Image,
    _q: f64,
    _knobs: &Map<String, Value>,
) -> Result<EncodedCell, Box<dyn Error>> {
    Err("zenjxl encode is disabled (rebuild with `--features sweep`)".into())
}

// ── helpers ─────────────────────────────────────────────────────────────

#[cfg(all(feature = "sweep", feature = "avif"))]
pub(crate) fn bytemuck_cast_rgb(bytes: &[u8]) -> &[rgb::Rgb<u8>] {
    // `rgb::Rgb<u8>` is `repr(C)` over three `u8` fields with no padding,
    // so a flat RGB byte buffer with length divisible by 3 maps 1:1 onto
    // a slice of `Rgb<u8>`. We use the rgb crate's own
    // `FromSlice::as_rgb` which is a safe shim that performs the cast
    // via its own `bytemuck`-style guard internally — keeps our crate
    // `#![forbid(unsafe_code)]` clean.
    use rgb::FromSlice;
    bytes.as_rgb()
}

#[cfg(all(test, feature = "sweep", feature = "jpeg"))]
mod jpeg_knob_tests {
    use super::*;

    fn tiny_image() -> Rgb8Image {
        // Deterministic noise: a solid color has no AC coefficients, so
        // every trellis config would emit identical bytes and the
        // distinctness assertion below would be vacuous.
        let mut state = 0x9e37_79b9_u32;
        let pixels = (0..64 * 64 * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect();
        Rgb8Image {
            pixels,
            width: 64,
            height: 64,
        }
    }

    #[test]
    fn hybrid_knob_errors_with_migration_hint() {
        let mut knobs = Map::new();
        knobs.insert("hybrid".into(), Value::Bool(true));
        let msg = match encode_jpeg(&tiny_image(), 75.0, &knobs) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("hybrid knob must be rejected"),
        };
        assert!(msg.contains("removed"), "got {msg}");
        assert!(msg.contains("trellis"), "got {msg}");
    }

    #[test]
    fn trellis_knob_object_parses_and_encodes() {
        let mut knobs = Map::new();
        knobs.insert(
            "trellis".into(),
            serde_json::json!({
                "lambda1": 13.5,
                "dc": false,
                "coupling_scale": -4.0,
                "coupling_max_adjustment": 1.0,
                "speed": "thorough"
            }),
        );
        let cell = encode_jpeg(&tiny_image(), 75.0, &knobs).unwrap();
        assert!(!cell.bytes.is_empty());
        // Distinct from the default-trellis spelling (lambda differs).
        let mut default_knobs = Map::new();
        default_knobs.insert("trellis".into(), Value::Bool(true));
        let default_cell = encode_jpeg(&tiny_image(), 75.0, &default_knobs).unwrap();
        assert_ne!(cell.bytes, default_cell.bytes);
    }

    #[test]
    fn trellis_speed_rejects_unknown_string() {
        let mut knobs = Map::new();
        knobs.insert("trellis".into(), serde_json::json!({"speed": "warp"}));
        assert!(encode_jpeg(&tiny_image(), 75.0, &knobs).is_err());
    }
}
