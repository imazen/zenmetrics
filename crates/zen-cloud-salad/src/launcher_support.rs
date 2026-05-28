//! Salad-launcher support helpers: GPU class resolution, the
//! prior-fleet-summary class filter, R2 parent-cred loading, and
//! chunk-list synthesis.
//!
//! These are operator-side helpers extracted from the launcher binary
//! so the binary itself stays at glue-only size. They depend on the
//! `launcher` feature (clap-free at this layer, but they share the
//! sha2/hmac signing via `r2_ops`).

#![cfg(feature = "launcher")]

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use zenfleet_orchestrator::{PriorClassStats, R2Operator};

use crate::launch::{R2ParentCreds, SaladApi};
use crate::r2_ops::R2OperatorImpl;

/// A `chunks.jsonl` row matching the worker's inline pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    /// Stable chunk id (used as the omni sidecar filename).
    pub chunk_id: String,
    /// Worker-local input parquet basename.
    pub input_parquet: String,
    /// `s3://...` URI the worker downloads from.
    pub input_parquet_r2: String,
    /// `[row_start, row_end)` half-open slice the chunk processes.
    pub row_range: [usize; 2],
    /// `s3://...` directory of source images.
    pub source_dir_r2: String,
    /// Image basenames the chunk uses.
    pub image_basenames: Vec<String>,
    /// Sweep id (used by the worker for downstream paths).
    pub run_id: String,
    /// Pre-computed omni sidecar `s3://` URI.
    pub out_sidecar_omni: String,
    /// Pre-computed encoded-prefix `s3://` URI.
    pub out_encoded_prefix: String,
}

/// Knobs the launcher needs for chunk generation.
pub struct ChunkLayout {
    /// Number of chunks to produce.
    pub n: u32,
    /// Cells per chunk (row range upper bound).
    pub cells_per_chunk: u32,
    /// Bucket workers + driver share.
    pub bucket: String,
    /// Sweep id.
    pub sweep_id: String,
    /// `s3://...` URI of the shared input parquet.
    pub input_parquet_r2: String,
    /// `s3://...` directory of source images.
    pub source_dir_r2: String,
    /// Image basenames each chunk references.
    pub image_basenames: Vec<String>,
}

/// Synthesize `n` chunks with stable IDs.
pub fn generate_chunks(layout: &ChunkLayout) -> Vec<ChunkRecord> {
    let row_end = layout.cells_per_chunk.max(1) as usize;
    (0..layout.n)
        .map(|i| {
            let chunk_id = format!("scaleup-{i:03}");
            ChunkRecord {
                chunk_id: chunk_id.clone(),
                input_parquet: "smoke.parquet".into(),
                input_parquet_r2: layout.input_parquet_r2.clone(),
                row_range: [0, row_end],
                source_dir_r2: layout.source_dir_r2.clone(),
                image_basenames: layout.image_basenames.clone(),
                run_id: layout.sweep_id.clone(),
                out_sidecar_omni: format!(
                    "s3://{}/runs/{}/omni/{}.parquet",
                    layout.bucket, layout.sweep_id, chunk_id
                ),
                out_encoded_prefix: format!(
                    "s3://{}/runs/{}/encoded/{}/",
                    layout.bucket, layout.sweep_id, chunk_id
                ),
            }
        })
        .collect()
}

/// What `resolve_gpu_classes` returns.
pub struct GpuClassSelection {
    /// Human-readable names (priority order).
    pub names: Vec<String>,
    /// Resolved Salad class ids (same length, same order).
    pub ids: Vec<String>,
}

/// Resolve the GPU class list from the three accepted inputs.
///
/// Precedence: manual list > auto-by-price > single name.
pub async fn resolve_gpu_classes(
    api: &SaladApi,
    gpu_classes: &[String],
    max_price_per_hour: f64,
    price_priority: &str,
    fallback_single: &str,
) -> Result<GpuClassSelection> {
    if !gpu_classes.is_empty() {
        let names: Vec<String> = gpu_classes
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let mut ids = Vec::with_capacity(names.len());
        for name in &names {
            let id = api
                .resolve_gpu_class(name)
                .await
                .with_context(|| format!("resolve GPU class {name:?}"))?;
            ids.push(id);
        }
        Ok(GpuClassSelection { names, ids })
    } else if max_price_per_hour > 0.0 {
        let classes = api
            .gpu_classes_under_price(max_price_per_hour, price_priority)
            .await
            .context("auto-enumerate GPU classes by price")?;
        Ok(GpuClassSelection {
            names: classes.iter().map(|c| c.name.clone()).collect(),
            ids: classes.iter().map(|c| c.id.clone()).collect(),
        })
    } else {
        let id = api
            .resolve_gpu_class(fallback_single)
            .await
            .with_context(|| format!("resolve GPU class {fallback_single:?}"))?;
        Ok(GpuClassSelection {
            names: vec![fallback_single.to_string()],
            ids: vec![id],
        })
    }
}

/// Outcome of [`apply_class_filter`].
pub struct ClassFilterApplied {
    /// Names kept (after dedup vs prior signal).
    pub names: Vec<String>,
    /// IDs parallel to `names`.
    pub ids: Vec<String>,
    /// Names dropped by the filter.
    pub dropped: Vec<String>,
}

/// Apply the prior-fleet-summary class filter. When
/// `prior_path_or_uri` is None OR the load fails the input list is
/// returned unchanged with `dropped = []`.
pub async fn apply_class_filter(
    r2: &R2OperatorImpl,
    prior_path_or_uri: Option<&str>,
    names: Vec<String>,
    ids: Vec<String>,
    max_warmup_secs: u32,
    min_productive_chunks: f32,
) -> ClassFilterApplied {
    let Some(path) = prior_path_or_uri else {
        return ClassFilterApplied { names, ids, dropped: Vec::new() };
    };
    let bytes = match load_prior_fleet_summary_bytes(r2, path).await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[class-filter] WARN failed to load prior fleet_summary: {e:#}");
            return ClassFilterApplied { names, ids, dropped: Vec::new() };
        }
    };
    let observed = match parse_prior_class_stats(&bytes) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[class-filter] WARN failed to parse prior fleet_summary: {e:#}");
            return ClassFilterApplied { names, ids, dropped: Vec::new() };
        }
    };
    let cfg = zenfleet_orchestrator::SweepConfig {
        max_warmup_secs,
        min_productive_chunks,
        ..zenfleet_orchestrator::SweepConfig::default()
    };
    let outcome = zenfleet_orchestrator::filter_classes(&names, &observed, &cfg);
    if outcome.keep.is_empty() {
        return ClassFilterApplied { names, ids, dropped: Vec::new() };
    }
    let kept_ids: Vec<String> = outcome
        .keep
        .iter()
        .filter_map(|n| names.iter().position(|x| x == n).map(|i| ids[i].clone()))
        .collect();
    let dropped: Vec<String> = outcome.dropped.iter().map(|(n, _)| n.clone()).collect();
    eprintln!(
        "[class-filter] kept={} dropped={}",
        outcome.keep.len(),
        dropped.len()
    );
    ClassFilterApplied {
        names: outcome.keep,
        ids: kept_ids,
        dropped,
    }
}

async fn load_prior_fleet_summary_bytes(r2: &R2OperatorImpl, path_or_uri: &str) -> Result<Vec<u8>> {
    if let Some(rest) = path_or_uri.strip_prefix("s3://") {
        let (bucket, key) = rest
            .split_once('/')
            .with_context(|| format!("malformed s3 URI {path_or_uri:?}"))?;
        r2.get_bytes(bucket, key)
            .await
            .with_context(|| format!("GET {path_or_uri}"))
    } else {
        std::fs::read(path_or_uri).with_context(|| format!("read local file {path_or_uri}"))
    }
}

fn parse_prior_class_stats(bytes: &[u8]) -> Result<HashMap<String, PriorClassStats>> {
    let v: serde_json::Value = serde_json::from_slice(bytes).context("parse prior summary JSON")?;
    let replicas = v
        .get("replicas")
        .and_then(|x| x.as_array())
        .context("missing `replicas` array")?;
    let mut by_class: BTreeMap<String, (Vec<f64>, Vec<u64>)> = BTreeMap::new();
    for r in replicas {
        let class = r
            .get("gpu_class")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        if class.is_empty() {
            continue;
        }
        let warmup = r.get("warmup_seconds").and_then(|x| x.as_f64());
        let chunks = r
            .get("chunks_processed")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let entry = by_class.entry(class).or_default();
        if let Some(w) = warmup {
            entry.0.push(w);
        }
        entry.1.push(chunks);
    }
    let mut out: HashMap<String, PriorClassStats> = HashMap::new();
    for (class, (mut warmups, chunks)) in by_class {
        warmups.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if warmups.is_empty() {
            None
        } else {
            let n = warmups.len();
            Some(if n % 2 == 1 {
                warmups[n / 2] as u32
            } else {
                (0.5 * (warmups[n / 2 - 1] + warmups[n / 2])).round() as u32
            })
        };
        let mean = if chunks.is_empty() {
            0.0
        } else {
            chunks.iter().copied().sum::<u64>() as f32 / chunks.len() as f32
        };
        out.insert(
            class.clone(),
            PriorClassStats {
                name: class,
                median_warmup_secs: median,
                mean_chunks_processed: mean,
            },
        );
    }
    Ok(out)
}

/// Load the operator's R2 parent cred from env or
/// `~/.config/cloudflare/r2-credentials`.
pub fn load_r2_parent_creds_or_env() -> Result<R2ParentCreds> {
    if let Ok(c) = R2ParentCreds::from_env() {
        return Ok(c);
    }
    let home = std::env::var("HOME").context("HOME not set")?;
    let path = PathBuf::from(home).join(".config/cloudflare/r2-credentials");
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("read {} (or set env vars)", path.display()))?;
    let mut map = HashMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            map.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    let token = map
        .get("CF_API_TOKEN")
        .or_else(|| map.get("R2_API_TOKEN"))
        .cloned()
        .context("CF_API_TOKEN/R2_API_TOKEN missing from creds file")?;
    Ok(R2ParentCreds {
        cf_api_token: token,
        account_id: map.get("R2_ACCOUNT_ID").cloned().context("R2_ACCOUNT_ID missing")?,
        parent_access_key_id: map
            .get("R2_ACCESS_KEY_ID")
            .cloned()
            .context("R2_ACCESS_KEY_ID missing")?,
        parent_secret_access_key: map
            .get("R2_SECRET_ACCESS_KEY")
            .cloned()
            .context("R2_SECRET_ACCESS_KEY missing")?,
    })
}

/// Produce a DNS-style label (lowercase alphanumeric + hyphens,
/// ≤50 chars) from an arbitrary sweep id.
pub fn short_id_for_name(id: &str) -> String {
    let mut out = String::new();
    for c in id.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if c == '-' || c == '_' {
            out.push('-');
        }
    }
    if out.len() > 50 {
        out.truncate(50);
    }
    out
}
