#![forbid(unsafe_code)]

//! Plan-driven cells via the codecs' own sweep planners.
//!
//! The classic sweep crosses `--q-grid × --knob-grid` at face value: no
//! validity filtering, no alias dedup, nested-loop ordering, and a knob
//! vocabulary that has to be maintained in parallel with the encoder.
//! The codecs own that machinery (`SweepAxes` / `SweepBuilder` /
//! resolved-state fingerprints / main-effects-first queue ordering /
//! budget ladder with no silent caps), so `--plan <name>` asks the
//! codec for its cells instead of spelling them here. Wired codecs
//! (each behind its cargo feature): zenjpeg + zenjxl + zenwebp
//! (`rd_core` / `modes_full`), zenavif (those + `modes_full_alpha`),
//! and zenpng (all-lossless; every cell on the q=0 sentinel).
//! Cross-codec contract + scalar-axis inventory: `docs/PLAN_SWEEPS.md`.
//!
//! Each planned cell carries a fully-built codec config
//! ([`PlannedConfig`]); the row identity that lands in the TSV /
//! feature parquet `knob_tuple_json` column is the canonical JSON
//! `{"cell":"<stratum-id>","fp":"<fingerprint>","plan":"<name>"}` —
//! stable, sorted keys, unique per `(cell, q)` by construction. The
//! plan's no-silent-caps report (alias merges, invalid strata, budget
//! drops, q-coarsenings) is written once per sweep to
//! `<output>.plan.json` so downstream tooling can see what was *not*
//! encoded and why.

use std::error::Error;

use serde_json::{Map, Value, json};

use crate::decode::Rgb8Image;
use crate::sweep::encode::{CodecKind, EncodedCell};

/// A fully-resolved per-codec encoder config for one planned cell.
///
/// The enum (rather than a trait object) keeps configs `Send + Sync`
/// for the rayon walk and lets each arm call its codec's native encode
/// entry point.
#[derive(Debug)]
pub enum PlannedConfig {
    /// zenjpeg cell (quality already applied).
    #[cfg(feature = "jpeg")]
    Zenjpeg(Box<zenjpeg::encoder::EncoderConfig>),
    /// zenavif cell (quality applied; `threads` pinned by the planner).
    #[cfg(feature = "avif")]
    Zenavif(Box<zenavif::EncoderConfig>),
    /// zenjxl cell (a `SweepVariant`: lossy carries resolved distance;
    /// lossless cells ride the q=0 sentinel in the identity tuple).
    #[cfg(feature = "jxl")]
    Zenjxl(Box<zenjxl::sweep::SweepVariant>),
    /// zenwebp cell (lossless cells ride the q=0 sentinel).
    #[cfg(feature = "webp")]
    Zenwebp(Box<zenwebp::sweep::SweepVariant>),
    /// zenpng cell (all lossless; every cell rides the q=0 sentinel).
    #[cfg(feature = "png")]
    Zenpng(Box<zenpng::sweep::SweepVariant>),
    /// zengif cell (quantizer-driven; the variant carries the resolved
    /// quality + dithering + backend).
    #[cfg(feature = "gif")]
    Zengif(Box<zengif::sweep::SweepVariant>),
    /// zentiff cell (all lossless; every cell rides the q=0 sentinel).
    #[cfg(feature = "tiff")]
    Zentiff(Box<zentiff::sweep::SweepVariant>),
}

impl PlannedConfig {
    /// Encode the cell against an RGB8 source. Timing is measured here
    /// so chunk mode and jobexec report the same `encode_ms` semantics.
    pub fn encode_bytes(&self, source: &Rgb8Image) -> Result<EncodedCell, String> {
        let start = std::time::Instant::now();
        let bytes = match self {
            #[cfg(feature = "jpeg")]
            Self::Zenjpeg(cfg) => cfg
                .encode_bytes(
                    &source.pixels,
                    source.width,
                    source.height,
                    zenjpeg::encoder::PixelLayout::Rgb8Srgb,
                )
                .map_err(|e| format!("zenjpeg plan-cell encode failed: {e}"))?,
            #[cfg(feature = "avif")]
            Self::Zenavif(cfg) => {
                let pixels: &[rgb::Rgb<u8>] =
                    crate::sweep::encode::bytemuck_cast_rgb(&source.pixels);
                let img =
                    imgref::ImgRef::new(pixels, source.width as usize, source.height as usize);
                zenavif::encode_rgb8(img, cfg, almost_enough::StopToken::new(enough::Unstoppable))
                    .map_err(|e| format!("zenavif plan-cell encode failed: {e}"))?
                    .avif_file
            }
            #[cfg(feature = "jxl")]
            Self::Zenjxl(variant) => {
                // Threads pinned in every cell (playbook pattern 9 —
                // ambient-machine defaults poison content addressing).
                match variant.build() {
                    zenjxl::sweep::BuiltConfig::Lossy(c) => c
                        .with_threads(1)
                        .encode(
                            &source.pixels,
                            source.width,
                            source.height,
                            zenjxl::PixelLayout::Rgb8,
                        )
                        .map_err(|e| format!("zenjxl plan-cell encode failed: {e:?}"))?,
                    zenjxl::sweep::BuiltConfig::Lossless(c) => c
                        .with_threads(1)
                        .encode(
                            &source.pixels,
                            source.width,
                            source.height,
                            zenjxl::PixelLayout::Rgb8,
                        )
                        .map_err(|e| format!("zenjxl plan-cell encode failed: {e:?}"))?,
                }
            }
            #[cfg(feature = "webp")]
            Self::Zenwebp(variant) => {
                let cfg = variant.build();
                zenwebp::EncodeRequest::new(
                    &cfg,
                    &source.pixels,
                    zenwebp::PixelLayout::Rgb8,
                    source.width,
                    source.height,
                )
                .encode()
                .map_err(|e| format!("zenwebp plan-cell encode failed: {e:?}"))?
            }
            #[cfg(feature = "png")]
            Self::Zenpng(variant) => {
                // encode_png picks truecolor (quantize: None) or indexed
                // (quantize: Some) per the variant; parallel is pinned off
                // inside build() (pattern 9).
                let pixels: &[rgb::Rgb<u8>] =
                    crate::sweep::encode::bytemuck_cast_rgb(&source.pixels);
                let img =
                    imgref::ImgRef::new(pixels, source.width as usize, source.height as usize);
                variant
                    .encode_png(img, &enough::Unstoppable, &enough::Unstoppable)
                    .map_err(|e| format!("zenpng plan-cell encode failed: {e:?}"))?
            }
            #[cfg(feature = "gif")]
            Self::Zengif(variant) => {
                // GIF dims are u16; a sweep source above 65535 in either
                // axis can't be a single GIF frame.
                if source.width > u32::from(u16::MAX) || source.height > u32::from(u16::MAX) {
                    return Err(format!(
                        "zengif plan-cell: source {}x{} exceeds GIF's 65535x65535 frame limit",
                        source.width, source.height
                    ));
                }
                let cfg = variant.build();
                // RGB8 → opaque RGBA frame (GIF sweep sources are opaque
                // stills; the still-image variant pins use_transparency off).
                let pixels: Vec<zengif::Rgba> = source
                    .pixels
                    .chunks_exact(3)
                    .map(|p| zengif::Rgba {
                        r: p[0],
                        g: p[1],
                        b: p[2],
                        a: 255,
                    })
                    .collect();
                // FrameInput::new pins palette = None (the encoder
                // quantizes per the variant's backend, which is the point).
                let frame =
                    zengif::FrameInput::new(source.width as u16, source.height as u16, 0, pixels);
                let limits = zengif::Limits::none();
                zengif::EncodeRequest::new(&cfg, source.width as u16, source.height as u16)
                    .limits(&limits)
                    .stop(&enough::Unstoppable)
                    .encode(vec![frame])
                    .map_err(|e| format!("zengif plan-cell encode failed: {e:?}"))?
            }
            #[cfg(feature = "tiff")]
            Self::Zentiff(variant) => {
                use zenpixels::{PixelDescriptor, PixelSlice};
                let cfg = variant.build();
                let stride = (source.width as usize) * 3;
                let slice = PixelSlice::new(
                    &source.pixels,
                    source.width,
                    source.height,
                    stride,
                    PixelDescriptor::RGB8_SRGB,
                )
                .map_err(|e| format!("zentiff plan-cell: pixel slice construction failed: {e}"))?;
                zentiff::encode(&slice, &cfg, &enough::Unstoppable)
                    .map_err(|e| format!("zentiff plan-cell encode failed: {e:?}"))?
            }
        };
        Ok(EncodedCell {
            bytes,
            encode_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

/// One plan-generated encode cell.
pub struct PlannedCell {
    /// Quality point (mirrors the TSV `q` column).
    pub q: f64,
    /// Canonical knob-identity JSON for the TSV / parquet join key.
    pub knob_json: String,
    /// Fully-resolved encoder config (quality already applied).
    pub config: PlannedConfig,
}

/// Build plan cells for `codec`'s named plan over the sweep's quality
/// grid — the single codec-dispatch point for `--plan`.
///
/// `compute_limit` / `max_deviations` are the cross-codec constraint
/// knobs (`--compute-limit` / `--max-deviations`): `compute_limit`
/// drops cells whose `compute_tier()` exceeds the cap (reported, never
/// silently sampled); `max_deviations` keeps only cells within N axis
/// deviations of the default stratum (`1` = isolated main-effects — the
/// regime the `scalar_dense` heads want; `0` = the default stratum
/// alone). `None`/`None` reproduces the unconstrained curated space, so
/// existing `--plan rd_core|modes_full` invocations are unchanged. The
/// `__expert` codecs (jpeg/avif/jxl/webp) apply these via
/// `SweepBuilder::with_compute_limit` / `with_max_deviations`; the
/// public-API codecs (png/gif/tiff) via their free `plan_constrained`.
pub fn build_plan(
    codec: CodecKind,
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
    compute_limit: Option<u8>,
    max_deviations: Option<u8>,
) -> Result<BuiltPlan, Box<dyn Error>> {
    match codec {
        #[cfg(feature = "jpeg")]
        CodecKind::Zenjpeg => {
            build_zenjpeg_plan(name, budget, q_grid, compute_limit, max_deviations)
        }
        #[cfg(feature = "avif")]
        CodecKind::Zenavif => {
            build_zenavif_plan(name, budget, q_grid, compute_limit, max_deviations)
        }
        #[cfg(feature = "jxl")]
        CodecKind::Zenjxl => build_zenjxl_plan(name, budget, q_grid, compute_limit, max_deviations),
        #[cfg(feature = "webp")]
        CodecKind::Zenwebp => {
            build_zenwebp_plan(name, budget, q_grid, compute_limit, max_deviations)
        }
        #[cfg(feature = "png")]
        CodecKind::Zenpng => build_zenpng_plan(name, budget, q_grid, compute_limit, max_deviations),
        #[cfg(feature = "gif")]
        CodecKind::Zengif => build_zengif_plan(name, budget, q_grid, compute_limit, max_deviations),
        #[cfg(feature = "tiff")]
        CodecKind::Zentiff => {
            build_zentiff_plan(name, budget, q_grid, compute_limit, max_deviations)
        }
        // Unreachable only when ALL seven codec features are on (the
        // full build covers every CodecKind variant above); reachable —
        // and required — in partial builds like `sweep,png` without jxl.
        #[allow(unreachable_patterns)]
        other => Err(format!(
            "plan-driven sweeps are not wired for codec {:?} in this build \
             (zenjpeg needs --features jpeg, zenavif --features avif, \
             zenjxl --features jxl, zenwebp --features webp, \
             zenpng --features png, zengif --features gif, \
             zentiff --features tiff)",
            other.name()
        )
        .into()),
    }
}

/// A built plan: the cells plus the audit manifest.
pub struct BuiltPlan {
    pub cells: Vec<PlannedCell>,
    /// JSON document for `<output>.plan.json`: cell counts, alias
    /// merges, invalid strata, budget drops, q-coarsenings.
    pub manifest_json: String,
}

/// Build zenjpeg plan cells for the given plan name over the sweep's
/// quality grid. `rd_core` = the RD-front axes; `modes_full` = every
/// user-disableable mode axis (pair with a budget).
#[cfg(feature = "jpeg")]
pub fn build_zenjpeg_plan(
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
    compute_limit: Option<u8>,
    max_deviations: Option<u8>,
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenjpeg::encode::sweep::{QualityGrid, SweepAxes, SweepBuilder};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        "scalar_dense" => SweepAxes::scalar_dense(),
        other => {
            return Err(format!(
                "unknown zenjpeg plan {other:?}; expected rd_core, modes_full, or scalar_dense"
            )
            .into());
        }
    };
    let grid = QualityGrid::Explicit(q_grid.iter().map(|&q| q as f32).collect());
    let mut builder = SweepBuilder::new(axes, grid);
    if let Some(n) = budget {
        builder = builder.with_budget(n);
    }
    if let Some(n) = compute_limit {
        builder = builder.with_compute_limit(n);
    }
    if let Some(n) = max_deviations {
        builder = builder.with_max_deviations(n);
    }
    let plan = builder.plan();

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "compute_limit": compute_limit,
        "max_deviations": max_deviations,
        "q_grid": q_grid,
        "cells": plan.cells.len(),
        "duplicates_merged": plan.duplicates_merged,
        "invalid_skipped": plan.invalid_skipped,
        "compute_tier_skipped": plan.compute_tier_skipped,
        "q_coarsenings": plan.q_coarsenings,
        "over_budget": plan.over_budget,
        "dropped_axes": plan
            .dropped
            .iter()
            .map(|d| json!({"axis": d.axis, "kept": d.kept, "dropped": d.dropped}))
            .collect::<Vec<_>>(),
        "aliases": plan
            .cells
            .iter()
            .filter(|c| !c.aliases.is_empty())
            .map(|c| json!({"cell": c.id, "merged": c.aliases}))
            .collect::<Vec<_>>(),
    });

    let cells = plan
        .cells
        .into_iter()
        .map(|c| {
            // Cell ids end in `_q<q>`; the q lives in its own TSV column,
            // so the identity JSON carries the stratum id without it.
            let base =
                c.id.rfind("_q")
                    .map(|at| &c.id[..at])
                    .unwrap_or(c.id.as_str());
            let mut m = Map::new();
            m.insert("cell".into(), Value::String(base.to_string()));
            m.insert(
                "fp".into(),
                Value::String(format!("{:016x}", c.fingerprint)),
            );
            m.insert("plan".into(), Value::String(name.to_string()));
            PlannedCell {
                q: f64::from(c.quality),
                knob_json: Value::Object(m).to_string(),
                config: PlannedConfig::Zenjpeg(Box::new(c.config)),
            }
        })
        .collect();

    Ok(BuiltPlan {
        cells,
        manifest_json: serde_json::to_string_pretty(&manifest)
            .expect("plan manifest serialization cannot fail"),
    })
}

/// Build zenavif plan cells (`rd_core` / `modes_full` /
/// `modes_full_alpha` — the latter for RGBA corpora). Mirrors the
/// zenjpeg builder; the planner pins `threads(Some(1))` per cell, so
/// chunk-mode parallelism stays at the rayon walk.
#[cfg(feature = "avif")]
pub fn build_zenavif_plan(
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
    compute_limit: Option<u8>,
    max_deviations: Option<u8>,
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenavif::sweep::{QualityGrid, SweepAxes, SweepBuilder};
    let axes = SweepAxes::by_name(name).ok_or_else(|| {
        format!(
            "unknown zenavif plan {name:?}; expected rd_core, modes_full, modes_full_alpha, or scalar_dense"
        )
    })?;
    let grid = QualityGrid::Explicit(q_grid.iter().map(|&q| q as f32).collect());
    let mut builder = SweepBuilder::new(axes, grid);
    if let Some(n) = budget {
        builder = builder.with_budget(n);
    }
    if let Some(n) = compute_limit {
        builder = builder.with_compute_limit(n);
    }
    if let Some(n) = max_deviations {
        builder = builder.with_max_deviations(n);
    }
    let plan = builder.plan();

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "compute_limit": compute_limit,
        "max_deviations": max_deviations,
        "q_grid": q_grid,
        "cells": plan.cells.len(),
        "duplicates_merged": plan.duplicates_merged,
        "invalid_skipped": plan.invalid_skipped,
        "compute_tier_skipped": plan.compute_tier_skipped,
        "q_coarsenings": plan.q_coarsenings,
        "over_budget": plan.over_budget,
        "dropped_axes": plan
            .dropped
            .iter()
            .map(|d| json!({"axis": d.axis, "kept": d.kept, "dropped": d.dropped}))
            .collect::<Vec<_>>(),
        "aliases": plan
            .cells
            .iter()
            .filter(|c| !c.aliases.is_empty())
            .map(|c| json!({"cell": c.id, "merged": c.aliases}))
            .collect::<Vec<_>>(),
    });

    let cells = plan
        .cells
        .into_iter()
        .map(|c| {
            let base =
                c.id.rfind("_q")
                    .map(|at| &c.id[..at])
                    .unwrap_or(c.id.as_str());
            let mut m = Map::new();
            m.insert("cell".into(), Value::String(base.to_string()));
            m.insert(
                "fp".into(),
                Value::String(format!("{:016x}", c.fingerprint)),
            );
            m.insert("plan".into(), Value::String(name.to_string()));
            PlannedCell {
                q: f64::from(c.quality),
                knob_json: Value::Object(m).to_string(),
                config: PlannedConfig::Zenavif(Box::new(c.config)),
            }
        })
        .collect();

    Ok(BuiltPlan {
        cells,
        manifest_json: serde_json::to_string_pretty(&manifest)
            .expect("plan manifest serialization cannot fail"),
    })
}

/// Build zenjxl plan cells. Lossy cells multiply by the (generic
/// quality) grid; lossless cells emit one cell each and ride the q=0
/// sentinel in the identity tuple (`CellId.q` is i64 and a lossless id
/// carries no quality token — the parser ignores q for `mod-` ids).
#[cfg(feature = "jxl")]
pub fn build_zenjxl_plan(
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
    compute_limit: Option<u8>,
    max_deviations: Option<u8>,
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenjxl::sweep::{QualityGrid, SweepAxes, SweepBuilder};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        "scalar_dense" => SweepAxes::scalar_dense(),
        // P0 main-effects mode for the JXL lossy knob-space ablation
        // program: full lossy knob set over e1..=e9, lossy-only. Pair with
        // `--max-deviations 1` (defaulted in main.rs).
        "lossy_dense" => SweepAxes::lossy_dense(),
        other => {
            return Err(format!(
                "unknown zenjxl plan {other:?}; expected rd_core, modes_full, scalar_dense, or lossy_dense"
            )
            .into());
        }
    };
    let grid = QualityGrid::ExplicitQuality(q_grid.iter().map(|&q| q as f32).collect());
    let mut builder = SweepBuilder::new(axes, grid);
    if let Some(n) = budget {
        builder = builder.with_budget(n);
    }
    if let Some(n) = compute_limit {
        builder = builder.with_compute_limit(n);
    }
    if let Some(n) = max_deviations {
        builder = builder.with_max_deviations(n);
    }
    let plan = builder.plan();

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "compute_limit": compute_limit,
        "max_deviations": max_deviations,
        "q_grid": q_grid,
        "cells": plan.cells.len(),
        "duplicates_merged": plan.duplicates_merged,
        "invalid_skipped": plan.invalid_skipped,
        "compute_tier_skipped": plan.compute_tier_skipped,
        "q_coarsenings": plan.q_coarsenings,
        "over_budget": plan.over_budget,
        "dropped_axes": plan
            .dropped
            .iter()
            .map(|d| json!({"axis": d.axis, "kept": d.kept, "dropped": d.dropped}))
            .collect::<Vec<_>>(),
        "aliases": plan
            .cells
            .iter()
            .filter(|c| !c.aliases.is_empty())
            .map(|c| json!({"cell": c.id, "merged": c.aliases}))
            .collect::<Vec<_>>(),
    });

    let cells = plan
        .cells
        .into_iter()
        .map(|c| {
            // Lossy ids end `_q<q>`; lossless ids have no quality token
            // (labels cannot contain '_', so the rfind is unambiguous).
            let base =
                c.id.rfind("_q")
                    .map(|at| &c.id[..at])
                    .unwrap_or(c.id.as_str());
            let mut m = Map::new();
            m.insert("cell".into(), Value::String(base.to_string()));
            m.insert(
                "fp".into(),
                Value::String(format!("{:016x}", c.fingerprint)),
            );
            m.insert("plan".into(), Value::String(name.to_string()));
            PlannedCell {
                q: c.quality.map(f64::from).unwrap_or(0.0),
                knob_json: Value::Object(m).to_string(),
                config: PlannedConfig::Zenjxl(Box::new(c.variant)),
            }
        })
        .collect();

    Ok(BuiltPlan {
        cells,
        manifest_json: serde_json::to_string_pretty(&manifest)
            .expect("plan manifest serialization cannot fail"),
    })
}

/// Build zenwebp plan cells. Lossy cells multiply by the grid;
/// lossless (VP8L) cells emit one cell each on the q=0 sentinel
/// (`vp8l-` ids carry no quality token; the parser ignores q there).
#[cfg(feature = "webp")]
pub fn build_zenwebp_plan(
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
    compute_limit: Option<u8>,
    max_deviations: Option<u8>,
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenwebp::sweep::{QualityGrid, SweepAxes, SweepBuilder};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        "scalar_dense" => SweepAxes::scalar_dense(),
        other => {
            return Err(format!(
                "unknown zenwebp plan {other:?}; expected rd_core, modes_full, or scalar_dense"
            )
            .into());
        }
    };
    let grid = QualityGrid::Explicit(q_grid.iter().map(|&q| q as f32).collect());
    let mut builder = SweepBuilder::new(axes, grid);
    if let Some(n) = budget {
        builder = builder.with_budget(n);
    }
    if let Some(n) = compute_limit {
        builder = builder.with_compute_limit(n);
    }
    if let Some(n) = max_deviations {
        builder = builder.with_max_deviations(n);
    }
    let plan = builder.plan();

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "compute_limit": compute_limit,
        "max_deviations": max_deviations,
        "q_grid": q_grid,
        "cells": plan.cells.len(),
        "duplicates_merged": plan.duplicates_merged,
        "invalid_skipped": plan.invalid_skipped,
        "compute_tier_skipped": plan.compute_tier_skipped,
        "q_coarsenings": plan.q_coarsenings,
        "over_budget": plan.over_budget,
        "dropped_axes": plan
            .dropped
            .iter()
            .map(|d| json!({"axis": d.axis, "kept": d.kept, "dropped": d.dropped}))
            .collect::<Vec<_>>(),
        "aliases": plan
            .cells
            .iter()
            .filter(|c| !c.aliases.is_empty())
            .map(|c| json!({"cell": c.id, "merged": c.aliases}))
            .collect::<Vec<_>>(),
    });

    let cells = plan
        .cells
        .into_iter()
        .map(|c| {
            let base =
                c.id.rfind("_q")
                    .map(|at| &c.id[..at])
                    .unwrap_or(c.id.as_str());
            let mut m = Map::new();
            m.insert("cell".into(), Value::String(base.to_string()));
            m.insert(
                "fp".into(),
                Value::String(format!("{:016x}", c.fingerprint)),
            );
            m.insert("plan".into(), Value::String(name.to_string()));
            PlannedCell {
                q: c.quality.map(f64::from).unwrap_or(0.0),
                knob_json: Value::Object(m).to_string(),
                config: PlannedConfig::Zenwebp(Box::new(c.variant)),
            }
        })
        .collect();

    Ok(BuiltPlan {
        cells,
        manifest_json: serde_json::to_string_pretty(&manifest)
            .expect("plan manifest serialization cannot fail"),
    })
}

/// Build zenpng plan cells. PNG is all-lossless: no quality grid (the
/// passed grid is recorded as ignored in the manifest), every cell on
/// the q=0 sentinel, and the whole modes_full is 9 cells (no budget
/// ladder exists; a budget below the cell count is reported, never
/// silently honored by sampling).
#[cfg(feature = "png")]
pub fn build_zenpng_plan(
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
    compute_limit: Option<u8>,
    max_deviations: Option<u8>,
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenpng::sweep::{SweepAxes, plan_constrained};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        "scalar_dense" => SweepAxes::scalar_dense(),
        other => {
            return Err(format!(
                "unknown zenpng plan {other:?}; expected rd_core, modes_full, or scalar_dense"
            )
            .into());
        }
    };
    // PNG is lossless: no quality grid (the passed grid is recorded as
    // ignored). The free `plan_constrained` applies the compute/deviation
    // constraints in-codec and reports every compute-tier drop.
    let p = plan_constrained(&axes, compute_limit, max_deviations);
    let over_budget = budget.is_some_and(|b| p.cells.len() > b);

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "compute_limit": compute_limit,
        "max_deviations": max_deviations,
        "q_grid": "ignored (lossless; all cells on the q=0 sentinel)",
        "q_grid_passed": q_grid,
        "cells": p.cells.len(),
        "duplicates_merged": p.duplicates_merged,
        "invalid_skipped": [],
        "compute_tier_skipped": p.compute_tier_skipped,
        "q_coarsenings": 0,
        "over_budget": over_budget,
        "dropped_axes": [],
        "aliases": p
            .cells
            .iter()
            .filter(|c| !c.aliases.is_empty())
            .map(|c| json!({"cell": c.id, "merged": c.aliases}))
            .collect::<Vec<_>>(),
    });

    let cells = p
        .cells
        .into_iter()
        .map(|c| {
            let mut m = Map::new();
            m.insert("cell".into(), Value::String(c.id));
            m.insert(
                "fp".into(),
                Value::String(format!("{:016x}", c.fingerprint)),
            );
            m.insert("plan".into(), Value::String(name.to_string()));
            PlannedCell {
                q: 0.0,
                knob_json: Value::Object(m).to_string(),
                config: PlannedConfig::Zenpng(Box::new(c.variant)),
            }
        })
        .collect();

    Ok(BuiltPlan {
        cells,
        manifest_json: serde_json::to_string_pretty(&manifest)
            .expect("plan manifest serialization cannot fail"),
    })
}

/// Build zengif plan cells. GIF stills are quantizer-driven (quality,
/// dithering, and the quantizer backend all change pixels), so cells
/// multiply by the quality grid and carry their resolved quality (the
/// `gif-<backend>[-d<dither>]_q<q>` id grammar). There is no budget
/// ladder; a budget below the cell count is reported, never sampled.
#[cfg(feature = "gif")]
pub fn build_zengif_plan(
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
    compute_limit: Option<u8>,
    max_deviations: Option<u8>,
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zengif::sweep::{QualityGrid, SweepAxes, plan_constrained};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        "scalar_dense" => SweepAxes::scalar_dense(),
        other => {
            return Err(format!(
                "unknown zengif plan {other:?}; expected rd_core, modes_full, or scalar_dense"
            )
            .into());
        }
    };
    // q must be integer-valued for the `gif-..._q<q>` id grammar (CellId
    // quality is u8); fractional grids are rejected, never truncated.
    let mut q_u8: Vec<u8> = Vec::with_capacity(q_grid.len());
    for &q in q_grid {
        if q.fract() != 0.0 || !(0.0..=100.0).contains(&q) {
            return Err(format!(
                "zengif plan q values must be integers in 0..=100 (GIF quality is u8); got {q}"
            )
            .into());
        }
        q_u8.push(q as u8);
    }
    let grid = QualityGrid::Explicit(q_u8);
    let p = plan_constrained(&axes, &grid, compute_limit, max_deviations);
    let over_budget = budget.is_some_and(|b| p.cells.len() > b);

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "compute_limit": compute_limit,
        "max_deviations": max_deviations,
        "q_grid": q_grid,
        "cells": p.cells.len(),
        "duplicates_merged": p.duplicates_merged,
        "invalid_skipped": [],
        "compute_tier_skipped": p.compute_tier_skipped,
        "q_coarsenings": 0,
        "over_budget": over_budget,
        "dropped_axes": [],
        "aliases": p
            .cells
            .iter()
            .filter(|c| !c.aliases.is_empty())
            .map(|c| json!({"cell": c.id, "merged": c.aliases}))
            .collect::<Vec<_>>(),
    });

    let cells = p
        .cells
        .into_iter()
        .map(|c| {
            // Ids end `_q<q>`; q lives in its own TSV column (labels
            // cannot contain '_', so the rfind is unambiguous).
            let base =
                c.id.rfind("_q")
                    .map(|at| &c.id[..at])
                    .unwrap_or(c.id.as_str());
            let mut m = Map::new();
            m.insert("cell".into(), Value::String(base.to_string()));
            m.insert(
                "fp".into(),
                Value::String(format!("{:016x}", c.fingerprint)),
            );
            m.insert("plan".into(), Value::String(name.to_string()));
            PlannedCell {
                q: f64::from(c.variant.quality),
                knob_json: Value::Object(m).to_string(),
                config: PlannedConfig::Zengif(Box::new(c.variant)),
            }
        })
        .collect();

    Ok(BuiltPlan {
        cells,
        manifest_json: serde_json::to_string_pretty(&manifest)
            .expect("plan manifest serialization cannot fail"),
    })
}

/// Build zentiff plan cells. TIFF is all-lossless: no quality grid (the
/// passed grid is recorded as ignored), every cell on the q=0 sentinel.
/// The curated space is ≤ 16 cells (no budget ladder; a budget below the
/// cell count is reported, never sampled). TIFF cells carry no aliases —
/// the curated space is deduplicated by construction.
#[cfg(feature = "tiff")]
pub fn build_zentiff_plan(
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
    compute_limit: Option<u8>,
    max_deviations: Option<u8>,
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zentiff::sweep::{SweepAxes, plan_constrained};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        "scalar_dense" => SweepAxes::scalar_dense(),
        other => {
            return Err(format!(
                "unknown zentiff plan {other:?}; expected rd_core, modes_full, or scalar_dense"
            )
            .into());
        }
    };
    let p = plan_constrained(&axes, compute_limit, max_deviations);
    let over_budget = budget.is_some_and(|b| p.cells.len() > b);

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "compute_limit": compute_limit,
        "max_deviations": max_deviations,
        "q_grid": "ignored (lossless; all cells on the q=0 sentinel)",
        "q_grid_passed": q_grid,
        "cells": p.cells.len(),
        "duplicates_merged": 0,
        "invalid_skipped": [],
        "compute_tier_skipped": p.compute_tier_skipped,
        "q_coarsenings": 0,
        "over_budget": over_budget,
        "dropped_axes": [],
        "aliases": [],
    });

    let cells = p
        .cells
        .into_iter()
        .map(|c| {
            let mut m = Map::new();
            m.insert("cell".into(), Value::String(c.id));
            m.insert(
                "fp".into(),
                Value::String(format!("{:016x}", c.fingerprint)),
            );
            m.insert("plan".into(), Value::String(name.to_string()));
            PlannedCell {
                q: 0.0,
                knob_json: Value::Object(m).to_string(),
                config: PlannedConfig::Zentiff(Box::new(c.variant)),
            }
        })
        .collect();

    Ok(BuiltPlan {
        cells,
        manifest_json: serde_json::to_string_pretty(&manifest)
            .expect("plan manifest serialization cannot fail"),
    })
}

/// Resolve a plan-cell identity to its codec config, verifying the
/// carried resolved-state fingerprint.
///
/// This is the executor-side half of the durable-identity contract: a
/// ledger job stores only `{"cell": <stratum-id>, "fp": <hex>, "plan":
/// …}`, and the codec's id grammar (`config_from_cell_id`) is
/// self-describing — but builds drift, so the fingerprint is recomputed
/// from the resolved config and any mismatch is a loud deterministic
/// failure instead of a silently wrong encode.
pub fn resolve_verified(
    codec: CodecKind,
    cell_id: &str,
    q: f32,
    fp_hex: &str,
) -> Result<PlannedConfig, String> {
    let mismatch = |actual: &str| {
        format!(
            "plan-cell fingerprint mismatch for {cell_id:?} q{q}: declared {fp_hex}, \
             resolved {actual} — id-grammar drift between the declaring and executing builds?"
        )
    };
    match codec {
        #[cfg(feature = "jpeg")]
        CodecKind::Zenjpeg => {
            let cfg = zenjpeg::encode::sweep::config_from_cell_id(cell_id, q)?;
            let actual = format!("{:016x}", zenjpeg::encode::sweep::fingerprint(&cfg));
            if actual != fp_hex {
                return Err(mismatch(&actual));
            }
            Ok(PlannedConfig::Zenjpeg(Box::new(cfg)))
        }
        #[cfg(feature = "avif")]
        CodecKind::Zenavif => {
            let cfg = zenavif::sweep::config_from_cell_id(cell_id, q)?;
            let actual = format!("{:016x}", zenavif::sweep::fingerprint(&cfg));
            if actual != fp_hex {
                return Err(mismatch(&actual));
            }
            Ok(PlannedConfig::Zenavif(Box::new(cfg)))
        }
        #[cfg(feature = "jxl")]
        CodecKind::Zenjxl => {
            // Lossy variants carry their resolved distance, so the
            // parser consumes the full id including the quality token;
            // lossless ids have none (q is the i64 sentinel 0).
            let full_id = if cell_id.starts_with("vd-") {
                format!("{cell_id}_q{q}")
            } else {
                cell_id.to_string()
            };
            let variant = zenjxl::sweep::variant_from_cell_id(&full_id)?;
            let actual = format!("{:016x}", zenjxl::sweep::fingerprint(&variant));
            if actual != fp_hex {
                return Err(mismatch(&actual));
            }
            Ok(PlannedConfig::Zenjxl(Box::new(variant)))
        }
        #[cfg(feature = "webp")]
        CodecKind::Zenwebp => {
            let full_id = if cell_id.starts_with("vp8-") {
                format!("{cell_id}_q{q}")
            } else {
                cell_id.to_string()
            };
            let variant = zenwebp::sweep::variant_from_cell_id(&full_id)?;
            let actual = format!("{:016x}", zenwebp::sweep::fingerprint(&variant));
            if actual != fp_hex {
                return Err(mismatch(&actual));
            }
            Ok(PlannedConfig::Zenwebp(Box::new(variant)))
        }
        #[cfg(feature = "png")]
        CodecKind::Zenpng => {
            let variant = zenpng::sweep::variant_from_cell_id(cell_id)?;
            let actual = format!("{:016x}", zenpng::sweep::fingerprint(&variant));
            if actual != fp_hex {
                return Err(mismatch(&actual));
            }
            Ok(PlannedConfig::Zenpng(Box::new(variant)))
        }
        #[cfg(feature = "gif")]
        CodecKind::Zengif => {
            // GIF cells carry their quality in the id grammar
            // (`gif-..._q<q>`); the stored stratum id drops the `_q<q>`
            // token (it lives in the TSV `q` column), so re-attach it.
            let q_int = q as i64;
            let full_id = format!("{cell_id}_q{q_int}");
            let variant = zengif::sweep::variant_from_cell_id(&full_id)?;
            let actual = format!("{:016x}", zengif::sweep::fingerprint(&variant));
            if actual != fp_hex {
                return Err(mismatch(&actual));
            }
            Ok(PlannedConfig::Zengif(Box::new(variant)))
        }
        #[cfg(feature = "tiff")]
        CodecKind::Zentiff => {
            let variant = zentiff::sweep::variant_from_cell_id(cell_id)?;
            let actual = format!("{:016x}", zentiff::sweep::fingerprint(&variant));
            if actual != fp_hex {
                return Err(mismatch(&actual));
            }
            Ok(PlannedConfig::Zentiff(Box::new(variant)))
        }
        // Unreachable only in the all-seven-codec build; reachable —
        // and required — in partial builds (see build_plan's arm).
        #[allow(unreachable_patterns)]
        other => Err(format!(
            "plan-cell identity on codec {:?} which is not plan-wired in this build",
            other.name()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "jpeg")]
    #[test]
    fn rd_core_plan_builds_cells_with_stable_identity() {
        let plan = build_zenjpeg_plan("rd_core", None, &[50.0, 85.0], None, None).unwrap();
        assert!(!plan.cells.is_empty());
        // First cell is the production-default stratum (main-effects-first).
        let first = &plan.cells[0];
        assert_eq!(first.q, 50.0);
        assert!(
            first.knob_json.contains("\"cell\":\"jp3_t0_small_420\""),
            "got {}",
            first.knob_json
        );
        assert!(first.knob_json.contains("\"plan\":\"rd_core\""));
        // Identity is unique per (cell, q).
        let mut seen = std::collections::HashSet::new();
        for c in &plan.cells {
            assert!(seen.insert((c.knob_json.clone(), c.q.to_bits())));
        }
        assert!(plan.manifest_json.contains("\"plan\": \"rd_core\""));
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn resolve_verified_roundtrips_and_rejects_tampered_fp() {
        let plan = build_zenjpeg_plan("rd_core", None, &[85.0], None, None).unwrap();
        let cell = &plan.cells[0];
        let v: serde_json::Value = serde_json::from_str(&cell.knob_json).unwrap();
        let id = v["cell"].as_str().unwrap();
        let fp = v["fp"].as_str().unwrap();
        let cfg = resolve_verified(CodecKind::Zenjpeg, id, cell.q as f32, fp).unwrap();
        match cfg {
            PlannedConfig::Zenjpeg(cfg) => assert_eq!(
                format!("{:016x}", zenjpeg::encode::sweep::fingerprint(&cfg)),
                fp
            ),
            #[cfg(feature = "avif")]
            PlannedConfig::Zenavif(_) => panic!("zenjpeg identity resolved to zenavif"),
            #[cfg(feature = "jxl")]
            PlannedConfig::Zenjxl(_) => panic!("zenjpeg identity resolved to zenjxl"),
            #[cfg(feature = "webp")]
            PlannedConfig::Zenwebp(_) => panic!("zenjpeg identity resolved to zenwebp"),
            #[cfg(feature = "png")]
            PlannedConfig::Zenpng(_) => panic!("zenjpeg identity resolved to zenpng"),
            #[cfg(feature = "gif")]
            PlannedConfig::Zengif(_) => panic!("zenjpeg identity resolved to zengif"),
            #[cfg(feature = "tiff")]
            PlannedConfig::Zentiff(_) => panic!("zenjpeg identity resolved to zentiff"),
        }
        let err = resolve_verified(CodecKind::Zenjpeg, id, cell.q as f32, "0000000000000000")
            .unwrap_err();
        assert!(err.contains("fingerprint mismatch"), "got {err}");
    }

    #[cfg(feature = "jpeg")]
    #[cfg(feature = "png")]
    #[test]
    fn zenpng_plan_builds_and_cells_resolve_verified() {
        let plan = build_plan(CodecKind::Zenpng, "modes_full", None, &[50.0], None, None).unwrap();
        assert!(plan.cells.len() >= 8);
        for cell in &plan.cells {
            assert_eq!(cell.q, 0.0, "all PNG cells ride the q=0 sentinel");
            let v: serde_json::Value = serde_json::from_str(&cell.knob_json).unwrap();
            let id = v["cell"].as_str().unwrap();
            let fp = v["fp"].as_str().unwrap();
            let resolved = resolve_verified(CodecKind::Zenpng, id, 0.0, fp)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            match resolved {
                PlannedConfig::Zenpng(_) => {}
                #[allow(unreachable_patterns)]
                _ => panic!("zenpng identity resolved to another codec"),
            }
        }
        let first: serde_json::Value = serde_json::from_str(&plan.cells[0].knob_json).unwrap();
        assert!(
            resolve_verified(
                CodecKind::Zenpng,
                first["cell"].as_str().unwrap(),
                0.0,
                "0000000000000000"
            )
            .is_err()
        );
    }

    #[cfg(feature = "webp")]
    #[test]
    fn zenwebp_plan_builds_and_cells_resolve_verified() {
        let plan = build_plan(
            CodecKind::Zenwebp,
            "rd_core",
            None,
            &[50.0, 85.0],
            None,
            None,
        )
        .unwrap();
        assert!(!plan.cells.is_empty());
        let mut lossy = false;
        let mut lossless = false;
        for cell in &plan.cells {
            let v: serde_json::Value = serde_json::from_str(&cell.knob_json).unwrap();
            let id = v["cell"].as_str().unwrap();
            let fp = v["fp"].as_str().unwrap();
            let resolved = resolve_verified(CodecKind::Zenwebp, id, cell.q as f32, fp)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            match resolved {
                PlannedConfig::Zenwebp(variant) => match *variant {
                    zenwebp::sweep::SweepVariant::Lossy(_) => lossy = true,
                    zenwebp::sweep::SweepVariant::Lossless(_) => {
                        assert_eq!(cell.q, 0.0, "lossless cells ride the q=0 sentinel");
                        lossless = true;
                    }
                },
                #[allow(unreachable_patterns)]
                _ => panic!("zenwebp identity resolved to another codec"),
            }
        }
        assert!(lossy && lossless, "both modes must appear");
        let first: serde_json::Value = serde_json::from_str(&plan.cells[0].knob_json).unwrap();
        assert!(
            resolve_verified(
                CodecKind::Zenwebp,
                first["cell"].as_str().unwrap(),
                plan.cells[0].q as f32,
                "0000000000000000"
            )
            .is_err()
        );
    }

    #[cfg(feature = "jxl")]
    #[test]
    fn zenjxl_plan_builds_and_cells_resolve_verified() {
        let plan = build_plan(
            CodecKind::Zenjxl,
            "rd_core",
            None,
            &[50.0, 85.0],
            None,
            None,
        )
        .unwrap();
        assert!(!plan.cells.is_empty());
        let mut checked_lossy = false;
        let mut checked_lossless = false;
        for cell in &plan.cells {
            let v: serde_json::Value = serde_json::from_str(&cell.knob_json).unwrap();
            let id = v["cell"].as_str().unwrap();
            let fp = v["fp"].as_str().unwrap();
            let resolved = resolve_verified(CodecKind::Zenjxl, id, cell.q as f32, fp)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            match resolved {
                PlannedConfig::Zenjxl(variant) => match *variant {
                    zenjxl::sweep::SweepVariant::Lossy(_) => checked_lossy = true,
                    zenjxl::sweep::SweepVariant::Lossless(_) => {
                        assert_eq!(cell.q, 0.0, "lossless cells ride the q=0 sentinel");
                        checked_lossless = true;
                    }
                },
                #[allow(unreachable_patterns)]
                _ => panic!("zenjxl identity resolved to another codec"),
            }
        }
        assert!(checked_lossy && checked_lossless, "both modes must appear");
        // Tampered fp = loud failure.
        let first: serde_json::Value = serde_json::from_str(&plan.cells[0].knob_json).unwrap();
        assert!(
            resolve_verified(
                CodecKind::Zenjxl,
                first["cell"].as_str().unwrap(),
                plan.cells[0].q as f32,
                "0000000000000000"
            )
            .is_err()
        );
    }

    #[test]
    fn unknown_plan_is_an_error() {
        assert!(build_zenjpeg_plan("nope", None, &[50.0], None, None).is_err());
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn budget_is_honored_and_reported() {
        let plan = build_zenjpeg_plan("modes_full", Some(500), &[30.0, 70.0], None, None).unwrap();
        assert!(plan.cells.len() <= 500);
        assert!(plan.manifest_json.contains("dropped_axes"));
    }

    #[cfg(feature = "avif")]
    #[test]
    fn zenavif_plan_builds_and_resolves_with_fp_verification() {
        let plan = build_plan(
            CodecKind::Zenavif,
            "rd_core",
            None,
            &[50.0, 85.0],
            None,
            None,
        )
        .unwrap();
        assert!(!plan.cells.is_empty());
        let first = &plan.cells[0];
        assert_eq!(first.q, 50.0);
        assert!(
            first.knob_json.contains("\"cell\":\"s4\""),
            "all-defaults stratum first, got {}",
            first.knob_json
        );
        assert!(first.knob_json.contains("\"plan\":\"rd_core\""));

        // Identity unique per (cell, q).
        let mut seen = std::collections::HashSet::new();
        for c in &plan.cells {
            assert!(seen.insert((c.knob_json.clone(), c.q.to_bits())));
        }

        // Executor-side roundtrip: id + fp → config, fingerprint-exact.
        let v: serde_json::Value = serde_json::from_str(&first.knob_json).unwrap();
        let id = v["cell"].as_str().unwrap();
        let fp = v["fp"].as_str().unwrap();
        let cfg = resolve_verified(CodecKind::Zenavif, id, first.q as f32, fp).unwrap();
        match cfg {
            PlannedConfig::Zenavif(cfg) => {
                assert_eq!(format!("{:016x}", zenavif::sweep::fingerprint(&cfg)), fp);
            }
            #[cfg(feature = "jpeg")]
            PlannedConfig::Zenjpeg(_) => panic!("zenavif identity resolved to zenjpeg"),
            #[cfg(feature = "jxl")]
            PlannedConfig::Zenjxl(_) => panic!("zenavif identity resolved to zenjxl"),
            #[cfg(feature = "webp")]
            PlannedConfig::Zenwebp(_) => panic!("zenavif identity resolved to zenwebp"),
            #[cfg(feature = "png")]
            PlannedConfig::Zenpng(_) => panic!("zenavif identity resolved to zenpng"),
            #[cfg(feature = "gif")]
            PlannedConfig::Zengif(_) => panic!("zenavif identity resolved to zengif"),
            #[cfg(feature = "tiff")]
            PlannedConfig::Zentiff(_) => panic!("zenavif identity resolved to zentiff"),
        }
        let err = resolve_verified(CodecKind::Zenavif, id, first.q as f32, "0000000000000000")
            .unwrap_err();
        assert!(err.contains("fingerprint mismatch"), "got {err}");

        // The alpha preset resolves too (the RGBA-corpora plan).
        assert!(
            build_plan(
                CodecKind::Zenavif,
                "modes_full_alpha",
                Some(200),
                &[60.0],
                None,
                None
            )
            .is_ok()
        );
        assert!(build_plan(CodecKind::Zenavif, "nope", None, &[50.0], None, None).is_err());
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn scalar_dense_plan_builds_and_constraints_apply() {
        // scalar_dense is a recognized plan and produces cells.
        let full = build_plan(
            CodecKind::Zenjpeg,
            "scalar_dense",
            None,
            &[50.0, 85.0],
            None,
            None,
        )
        .unwrap();
        assert!(!full.cells.is_empty());
        assert!(full.manifest_json.contains("\"plan\": \"scalar_dense\""));

        // max_deviations = 1 keeps the isolated main-effects regime: strictly
        // fewer cells than the unconstrained scalar_dense space (which crosses
        // axes), and the manifest records the constraint.
        let mains = build_plan(
            CodecKind::Zenjpeg,
            "scalar_dense",
            None,
            &[50.0],
            None,
            Some(1),
        )
        .unwrap();
        assert!(!mains.cells.is_empty());
        assert!(mains.manifest_json.contains("\"max_deviations\": 1"));

        // compute_limit drops the expensive tier; cells resolve fingerprint-
        // exact through the executor-side path regardless.
        let limited = build_plan(
            CodecKind::Zenjpeg,
            "scalar_dense",
            None,
            &[85.0],
            Some(0),
            None,
        )
        .unwrap();
        assert!(limited.manifest_json.contains("compute_tier_skipped"));
        for cell in &limited.cells {
            let v: serde_json::Value = serde_json::from_str(&cell.knob_json).unwrap();
            let id = v["cell"].as_str().unwrap();
            let fp = v["fp"].as_str().unwrap();
            resolve_verified(CodecKind::Zenjpeg, id, cell.q as f32, fp)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
        }
    }

    #[cfg(feature = "gif")]
    #[test]
    fn zengif_plan_builds_and_cells_resolve_verified() {
        let plan = build_plan(
            CodecKind::Zengif,
            "modes_full",
            None,
            &[50.0, 85.0],
            None,
            None,
        )
        .unwrap();
        assert!(!plan.cells.is_empty());
        assert!(plan.manifest_json.contains("\"plan\": \"modes_full\""));
        for cell in &plan.cells {
            // GIF cells carry their quality (lossy-style id grammar).
            assert!(cell.q == 50.0 || cell.q == 85.0, "unexpected q {}", cell.q);
            let v: serde_json::Value = serde_json::from_str(&cell.knob_json).unwrap();
            let id = v["cell"].as_str().unwrap();
            let fp = v["fp"].as_str().unwrap();
            let resolved = resolve_verified(CodecKind::Zengif, id, cell.q as f32, fp)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            match resolved {
                PlannedConfig::Zengif(_) => {}
                #[allow(unreachable_patterns)]
                _ => panic!("zengif identity resolved to another codec"),
            }
        }
        // Tampered fp = loud failure.
        let first: serde_json::Value = serde_json::from_str(&plan.cells[0].knob_json).unwrap();
        assert!(
            resolve_verified(
                CodecKind::Zengif,
                first["cell"].as_str().unwrap(),
                plan.cells[0].q as f32,
                "0000000000000000"
            )
            .is_err()
        );
        // scalar_dense is a recognized plan; fractional q is rejected (u8 grid).
        assert!(
            build_plan(
                CodecKind::Zengif,
                "scalar_dense",
                None,
                &[60.0],
                None,
                Some(1)
            )
            .is_ok()
        );
        assert!(build_plan(CodecKind::Zengif, "modes_full", None, &[50.5], None, None).is_err());
        assert!(build_plan(CodecKind::Zengif, "nope", None, &[50.0], None, None).is_err());
    }

    #[cfg(feature = "tiff")]
    #[test]
    fn zentiff_plan_builds_and_cells_resolve_verified() {
        // TIFF is lossless: every cell on the q=0 sentinel.
        let plan = build_plan(CodecKind::Zentiff, "modes_full", None, &[50.0], None, None).unwrap();
        assert!(!plan.cells.is_empty());
        assert!(plan.manifest_json.contains("\"plan\": \"modes_full\""));
        for cell in &plan.cells {
            assert_eq!(cell.q, 0.0, "all TIFF cells ride the q=0 sentinel");
            let v: serde_json::Value = serde_json::from_str(&cell.knob_json).unwrap();
            let id = v["cell"].as_str().unwrap();
            let fp = v["fp"].as_str().unwrap();
            let resolved = resolve_verified(CodecKind::Zentiff, id, 0.0, fp)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            match resolved {
                PlannedConfig::Zentiff(_) => {}
                #[allow(unreachable_patterns)]
                _ => panic!("zentiff identity resolved to another codec"),
            }
        }
        // Tampered fp = loud failure.
        let first: serde_json::Value = serde_json::from_str(&plan.cells[0].knob_json).unwrap();
        assert!(
            resolve_verified(
                CodecKind::Zentiff,
                first["cell"].as_str().unwrap(),
                0.0,
                "0000000000000000"
            )
            .is_err()
        );
        // compute_limit=0 keeps only the cheapest (Uncompressed) method.
        let cheap = build_plan(
            CodecKind::Zentiff,
            "scalar_dense",
            None,
            &[0.0],
            Some(0),
            None,
        )
        .unwrap();
        assert!(!cheap.cells.is_empty());
        assert!(cheap.manifest_json.contains("compute_tier_skipped"));
        assert!(build_plan(CodecKind::Zentiff, "nope", None, &[0.0], None, None).is_err());
    }
}
