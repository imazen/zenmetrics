#![forbid(unsafe_code)]

//! Plan-driven cells via the codecs' own sweep planners.
//!
//! The classic sweep crosses `--q-grid × --knob-grid` at face value: no
//! validity filtering, no alias dedup, nested-loop ordering, and a knob
//! vocabulary that has to be maintained in parallel with the encoder.
//! The codecs own that machinery (`SweepAxes` / `SweepBuilder` /
//! resolved-state fingerprints / main-effects-first queue ordering /
//! budget ladder with no silent caps), so `--plan <name>` asks the
//! codec for its cells instead of spelling them here. Wired codecs:
//! zenjpeg (`rd_core` / `modes_full`) and zenavif (`rd_core` /
//! `modes_full` / `modes_full_alpha`), each behind its cargo feature.
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
                // parallel is pinned off inside build() (pattern 9).
                let cfg = variant.build();
                let pixels: &[rgb::Rgb<u8>] =
                    crate::sweep::encode::bytemuck_cast_rgb(&source.pixels);
                let img =
                    imgref::ImgRef::new(pixels, source.width as usize, source.height as usize);
                zenpng::encode_rgb8(img, None, &cfg, &enough::Unstoppable, &enough::Unstoppable)
                    .map_err(|e| format!("zenpng plan-cell encode failed: {e:?}"))?
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
pub fn build_plan(
    codec: CodecKind,
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
) -> Result<BuiltPlan, Box<dyn Error>> {
    match codec {
        #[cfg(feature = "jpeg")]
        CodecKind::Zenjpeg => build_zenjpeg_plan(name, budget, q_grid),
        #[cfg(feature = "avif")]
        CodecKind::Zenavif => build_zenavif_plan(name, budget, q_grid),
        #[cfg(feature = "jxl")]
        CodecKind::Zenjxl => build_zenjxl_plan(name, budget, q_grid),
        #[cfg(feature = "webp")]
        CodecKind::Zenwebp => build_zenwebp_plan(name, budget, q_grid),
        #[cfg(feature = "png")]
        CodecKind::Zenpng => build_zenpng_plan(name, budget, q_grid),
        // Unreachable only when ALL five codec features are on (the
        // full build covers every CodecKind variant above); reachable —
        // and required — in partial builds like `sweep,png` without jxl.
        #[allow(unreachable_patterns)]
        other => Err(format!(
            "plan-driven sweeps are not wired for codec {:?} in this build \
             (zenjpeg needs --features jpeg, zenavif --features avif, \
             zenjxl --features jxl, zenwebp --features webp, \
             zenpng --features png)",
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
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenjpeg::encode::sweep::{QualityGrid, SweepAxes, SweepBuilder};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        other => {
            return Err(
                format!("unknown zenjpeg plan {other:?}; expected rd_core or modes_full").into(),
            );
        }
    };
    let grid = QualityGrid::Explicit(q_grid.iter().map(|&q| q as f32).collect());
    let mut builder = SweepBuilder::new(axes, grid);
    if let Some(n) = budget {
        builder = builder.with_budget(n);
    }
    let plan = builder.plan();

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "q_grid": q_grid,
        "cells": plan.cells.len(),
        "duplicates_merged": plan.duplicates_merged,
        "invalid_skipped": plan.invalid_skipped,
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
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenavif::sweep::{QualityGrid, SweepAxes, SweepBuilder};
    let axes = SweepAxes::by_name(name).ok_or_else(|| {
        format!("unknown zenavif plan {name:?}; expected rd_core, modes_full, or modes_full_alpha")
    })?;
    let grid = QualityGrid::Explicit(q_grid.iter().map(|&q| q as f32).collect());
    let mut builder = SweepBuilder::new(axes, grid);
    if let Some(n) = budget {
        builder = builder.with_budget(n);
    }
    let plan = builder.plan();

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "q_grid": q_grid,
        "cells": plan.cells.len(),
        "duplicates_merged": plan.duplicates_merged,
        "invalid_skipped": plan.invalid_skipped,
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
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenjxl::sweep::{QualityGrid, SweepAxes, SweepBuilder};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        other => {
            return Err(
                format!("unknown zenjxl plan {other:?}; expected rd_core or modes_full").into(),
            );
        }
    };
    let grid = QualityGrid::ExplicitQuality(q_grid.iter().map(|&q| q as f32).collect());
    let mut builder = SweepBuilder::new(axes, grid);
    if let Some(n) = budget {
        builder = builder.with_budget(n);
    }
    let plan = builder.plan();

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "q_grid": q_grid,
        "cells": plan.cells.len(),
        "duplicates_merged": plan.duplicates_merged,
        "invalid_skipped": plan.invalid_skipped,
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
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenwebp::sweep::{QualityGrid, SweepAxes, SweepBuilder};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        other => {
            return Err(
                format!("unknown zenwebp plan {other:?}; expected rd_core or modes_full").into(),
            );
        }
    };
    let grid = QualityGrid::Explicit(q_grid.iter().map(|&q| q as f32).collect());
    let mut builder = SweepBuilder::new(axes, grid);
    if let Some(n) = budget {
        builder = builder.with_budget(n);
    }
    let plan = builder.plan();

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "q_grid": q_grid,
        "cells": plan.cells.len(),
        "duplicates_merged": plan.duplicates_merged,
        "invalid_skipped": plan.invalid_skipped,
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
) -> Result<BuiltPlan, Box<dyn Error>> {
    use zenpng::sweep::{SweepAxes, plan};
    let axes = match name {
        "rd_core" => SweepAxes::rd_core(),
        "modes_full" => SweepAxes::modes_full(),
        other => {
            return Err(
                format!("unknown zenpng plan {other:?}; expected rd_core or modes_full").into(),
            );
        }
    };
    let p = plan(&axes);
    let over_budget = budget.is_some_and(|b| p.cells.len() > b);

    let manifest = json!({
        "plan": name,
        "budget": budget,
        "q_grid": "ignored (lossless; all cells on the q=0 sentinel)",
        "q_grid_passed": q_grid,
        "cells": p.cells.len(),
        "duplicates_merged": p.duplicates_merged,
        "invalid_skipped": [],
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
        // Unreachable only in the all-five-codec build; reachable —
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
        let plan = build_zenjpeg_plan("rd_core", None, &[50.0, 85.0]).unwrap();
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
        let plan = build_zenjpeg_plan("rd_core", None, &[85.0]).unwrap();
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
        }
        let err = resolve_verified(CodecKind::Zenjpeg, id, cell.q as f32, "0000000000000000")
            .unwrap_err();
        assert!(err.contains("fingerprint mismatch"), "got {err}");
    }

    #[cfg(feature = "jpeg")]
    #[cfg(feature = "png")]
    #[test]
    fn zenpng_plan_builds_and_cells_resolve_verified() {
        let plan = build_plan(CodecKind::Zenpng, "modes_full", None, &[50.0]).unwrap();
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
        let plan = build_plan(CodecKind::Zenwebp, "rd_core", None, &[50.0, 85.0]).unwrap();
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
        let plan = build_plan(CodecKind::Zenjxl, "rd_core", None, &[50.0, 85.0]).unwrap();
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
        assert!(build_zenjpeg_plan("nope", None, &[50.0]).is_err());
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn budget_is_honored_and_reported() {
        let plan = build_zenjpeg_plan("modes_full", Some(500), &[30.0, 70.0]).unwrap();
        assert!(plan.cells.len() <= 500);
        assert!(plan.manifest_json.contains("dropped_axes"));
    }

    #[cfg(feature = "avif")]
    #[test]
    fn zenavif_plan_builds_and_resolves_with_fp_verification() {
        let plan = build_plan(CodecKind::Zenavif, "rd_core", None, &[50.0, 85.0]).unwrap();
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
        }
        let err = resolve_verified(CodecKind::Zenavif, id, first.q as f32, "0000000000000000")
            .unwrap_err();
        assert!(err.contains("fingerprint mismatch"), "got {err}");

        // The alpha preset resolves too (the RGBA-corpora plan).
        assert!(build_plan(CodecKind::Zenavif, "modes_full_alpha", Some(200), &[60.0]).is_ok());
        assert!(build_plan(CodecKind::Zenavif, "nope", None, &[50.0]).is_err());
    }
}
