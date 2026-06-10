#![forbid(unsafe_code)]

//! Plan-driven zenjpeg cells via `zenjpeg::encode::sweep`.
//!
//! The classic sweep crosses `--q-grid × --knob-grid` at face value: no
//! validity filtering, no alias dedup, nested-loop ordering, and a knob
//! vocabulary that has to be maintained in parallel with the encoder.
//! zenjpeg now owns that machinery (`SweepAxes` / `SweepBuilder` /
//! resolved-state fingerprints / main-effects-first queue ordering /
//! budget ladder with no silent caps), so `--plan rd_core|modes_full`
//! asks the codec for its cells instead of spelling them here.
//!
//! Each planned cell carries a fully-built `EncoderConfig`; the row
//! identity that lands in the TSV / feature parquet `knob_tuple_json`
//! column is the canonical JSON
//! `{"cell":"<stratum-id>","fp":"<fingerprint>","plan":"<name>"}` —
//! stable, sorted keys, unique per `(cell, q)` by construction. The
//! plan's no-silent-caps report (alias merges, invalid strata, budget
//! drops, q-coarsenings) is written once per sweep to
//! `<output>.plan.json` so downstream tooling can see what was *not*
//! encoded and why.

use std::error::Error;

use serde_json::{Map, Value, json};
use zenjpeg::encode::sweep::{QualityGrid, SweepAxes, SweepBuilder};
use zenjpeg::encoder::EncoderConfig;

/// One plan-generated encode cell.
pub struct PlannedCell {
    /// Quality point (mirrors the TSV `q` column).
    pub q: f64,
    /// Canonical knob-identity JSON for the TSV / parquet join key.
    pub knob_json: String,
    /// Fully-resolved encoder config (quality already applied).
    pub config: EncoderConfig,
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
pub fn build_zenjpeg_plan(
    name: &str,
    budget: Option<usize>,
    q_grid: &[f64],
) -> Result<BuiltPlan, Box<dyn Error>> {
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
                config: c.config,
            }
        })
        .collect();

    Ok(BuiltPlan {
        cells,
        manifest_json: serde_json::to_string_pretty(&manifest)
            .expect("plan manifest serialization cannot fail"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn unknown_plan_is_an_error() {
        assert!(build_zenjpeg_plan("nope", None, &[50.0]).is_err());
    }

    #[test]
    fn budget_is_honored_and_reported() {
        let plan = build_zenjpeg_plan("modes_full", Some(500), &[30.0, 70.0]).unwrap();
        assert!(plan.cells.len() <= 500);
        assert!(plan.manifest_json.contains("dropped_axes"));
    }
}
