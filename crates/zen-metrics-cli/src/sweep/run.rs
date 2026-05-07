#![forbid(unsafe_code)]

//! Sweep execution: walk the (image × q × knobs) Cartesian grid, encode
//! each cell, decode-back, score against the source for every selected
//! metric, and write a Pareto TSV.
//!
//! ## Concurrency model
//!
//! - **Outer loop** (over source images) is **serial** by design. Each
//!   image is decoded once into an `Rgb8Image` shared across all of its
//!   cells via `&Rgb8Image`. Holding only one source's pixels in memory
//!   at a time keeps peak RAM at `1× source + N_threads × decoded_cell`,
//!   which is the bound that lets us run on 12-vCPU vast.ai boxes with
//!   modest memory.
//! - **Inner loop** (over `q × knob_tuple` for the current source) is
//!   **parallel** via rayon. Each rayon task encodes the cell, decodes
//!   it back, scores every metric, and emits a row through a `Mutex<csv
//!   Writer>` plus a `Mutex<FeatureParquetWriter>`. Rows land out-of-
//!   order; downstream tools group by `(image_path, q, knob_tuple)` and
//!   don't depend on order.
//! - **Thread budget** is set by `cfg.jobs` (or rayon's default = num
//!   cpus when `jobs = 0`). The setter is `try_init_thread_pool`, called
//!   exactly once per process from `cmd_sweep`.
//!
//! ## RAM
//!
//! At any moment in flight we hold:
//!   - 1 × source `Rgb8Image` (decoded once per image)
//!   - up to `N_threads` × `(encoded_bytes + decoded_cell + metric scratch)`
//!
//! The encoded bytes are short-lived (KB), the decoded cell is a `Vec<u8>`
//! the same size as the source. We deliberately **do not** pre-collect
//! per-cell results into a `Vec<CellOutcome>` before writing — the rayon
//! `for_each` walks one cell at a time per thread and emits immediately.
//!
//! ## Failure isolation
//!
//! A panic or error in one cell only invalidates that row; surrounding
//! cells continue. Stat counters use `AtomicU64`, so multiple parallel
//! failures don't lose increments to torn writes.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::decode::{Rgb8Image, decode_image_to_rgb8};
use crate::metrics::{GpuRuntime, MetricKind, run_metric, run_zensim_with_features};
use crate::sweep::encode::{CodecKind, encode};
use crate::sweep::feature_writer::FeatureParquetWriter;
use crate::sweep::grid::{KnobGrid, KnobTuple};

/// Runtime parameters for a sweep invocation.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    pub codec: CodecKind,
    pub sources: Vec<PathBuf>,
    pub q_grid: Vec<u32>,
    pub knob_grid: KnobGrid,
    pub metrics: Vec<MetricKind>,
    pub gpu_runtime: GpuRuntime,
    pub output: PathBuf,
    /// When set, every cell that runs the [`MetricKind::Zensim`] metric
    /// also persists its 300-feature extended vector to a parquet sidecar
    /// at this path. Joins back to `output` (TSV) by
    /// `(image_path, codec, q, knob_tuple_json)`.
    ///
    /// Cells that do not run zensim emit nothing to the parquet. If the
    /// metric list does not include `MetricKind::Zensim`, the parquet file
    /// is created but receives no rows; we don't auto-add zensim because
    /// callers may have explicit reasons for the metric set they passed.
    pub feature_output: Option<PathBuf>,
    /// Number of CPU threads for the per-image inner cell loop. `0`
    /// defers to rayon's default (one per logical core). `1` runs cells
    /// serially, useful for debugging.
    pub jobs: usize,
}

/// Initialise the global rayon thread pool. Safe to call multiple times
/// — the first call wins. Returns `Ok` regardless because subsequent
/// initialisations from the same process are a no-op for rayon.
pub fn try_init_thread_pool(jobs: usize) -> Result<(), Box<dyn Error>> {
    if jobs == 0 {
        return Ok(()); // let rayon pick `num_cpus`
    }
    // `build_global` errors if already-initialised; we silently swallow
    // because the harness is run as a one-shot binary — nobody else has
    // initialised the global pool.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global();
    Ok(())
}

/// Drive the sweep end-to-end. Outer loop over sources is serial (one
/// decoded image in memory at a time); inner cell loop is parallel via
/// rayon, with row writes funnelled through a `Mutex`.
pub fn run_sweep(cfg: &SweepConfig) -> Result<SweepStats, Box<dyn Error>> {
    // Honour the configured thread budget. No-op if the rayon global
    // pool is already initialised; first call wins.
    try_init_thread_pool(cfg.jobs)?;

    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(&cfg.output)?;
    write_header(&mut wtr, &cfg.metrics)?;

    let zensim_in_metrics = cfg.metrics.iter().any(|m| *m == MetricKind::Zensim);
    let feature_writer_inner = match &cfg.feature_output {
        Some(path) => Some(FeatureParquetWriter::create(path)?),
        None => None,
    };

    let cells_total =
        (cfg.sources.len() * cfg.q_grid.len() * cfg.knob_grid.cell_count()) as u64;
    let stats = AtomicSweepStats::new(cells_total);

    // Wrap writers in Mutex so rayon tasks can flush rows under a lock.
    // Lock contention is dominated by encode/decode/score work — order
    // of milliseconds — so the critical section is tiny in comparison.
    let wtr = Mutex::new(wtr);
    let feature_writer = Mutex::new(feature_writer_inner);

    for src_path in &cfg.sources {
        // Decode the source once per image so we don't re-PNG-decode for
        // every cell. The bytes are freed when we move to the next image
        // (drops at the end of this loop iteration). This is the entire
        // RAM-discipline knob: one source resident at a time.
        let source = match decode_image_to_rgb8(src_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!(
                    "[sweep] skipping {} (decode failed: {e})",
                    src_path.display()
                );
                stats.add_failed_decode(
                    (cfg.q_grid.len() * cfg.knob_grid.cell_count()) as u64,
                );
                continue;
            }
        };

        // Build a flat list of (q, knob_tuple) pairs for rayon to walk.
        // Cell count is bounded (≤ a few thousand per image typically),
        // so the Vec is cheap. The tuples are small (`Vec<KnobValue>`
        // owned per tuple); we don't clone the source.
        let cells: Vec<(u32, KnobTuple)> = cfg
            .q_grid
            .iter()
            .flat_map(|&q| cfg.knob_grid.iter_tuples().map(move |t| (q, t)))
            .collect();

        cells.par_iter().for_each(|(q, tuple)| {
            let outcome = compute_cell(cfg, src_path, &source, *q, tuple, zensim_in_metrics);
            // Emit row + feature row + update stats. The `wtr` lock is
            // held for the duration of one TSV record; `feature_writer`
            // for one parquet push.
            match outcome {
                CellOutcome::Ok {
                    row,
                    feature,
                    score_failed,
                } => {
                    if let Ok(mut w) = wtr.lock() {
                        if w.write_record(&row).is_ok() {
                            stats.add_emitted();
                        } else {
                            eprintln!("[sweep] write_record failed");
                        }
                    }
                    if let Some((image, codec, q_, knob_json, score, features)) = feature {
                        if let Ok(mut fw_guard) = feature_writer.lock() {
                            if let Some(fw) = fw_guard.as_mut() {
                                if let Err(e) = fw.push_row(
                                    &image, codec, q_, &knob_json, score, &features,
                                ) {
                                    eprintln!(
                                        "[sweep] feature_writer push failed: {} q={q_}: {e}",
                                        image,
                                    );
                                }
                            }
                        }
                    }
                    if score_failed {
                        stats.add_failed_score();
                    }
                }
                CellOutcome::EncodeFailed { row } => {
                    if let Ok(mut w) = wtr.lock() {
                        let _ = w.write_record(&row);
                    }
                    stats.add_failed_encode();
                }
                CellOutcome::DecodeFailed { row } => {
                    if let Ok(mut w) = wtr.lock() {
                        let _ = w.write_record(&row);
                    }
                    stats.add_failed_decode(1);
                }
            }
        });
    }

    // Drop locks and finalize.
    let mut wtr = wtr.into_inner().map_err(|e| format!("wtr lock poisoned: {e}"))?;
    wtr.flush()?;
    if let Some(fw) = feature_writer
        .into_inner()
        .map_err(|e| format!("feature_writer lock poisoned: {e}"))?
    {
        fw.finish()?;
    }
    Ok(stats.snapshot())
}

/// Per-cell outcome — pure result type, written to disk by the caller.
enum CellOutcome {
    Ok {
        row: Vec<String>,
        /// `(image_path, codec, q, knob_json, zensim_score, features)`
        feature: Option<(String, &'static str, u32, String, f32, Vec<f64>)>,
        score_failed: bool,
    },
    EncodeFailed {
        row: Vec<String>,
    },
    DecodeFailed {
        row: Vec<String>,
    },
}

/// Pure per-cell compute — no shared mutable state. Allocations:
///   - encoded bytes (small, dropped before scoring returns)
///   - decoded `Rgb8Image` (≈ source dimensions × 3 bytes; held during
///     metric scoring then dropped)
///   - row `Vec<String>` (small)
///   - optional feature `Vec<f64>` (300 entries when zensim_features_wanted)
fn compute_cell(
    cfg: &SweepConfig,
    src_path: &Path,
    source: &Rgb8Image,
    q: u32,
    tuple: &KnobTuple,
    zensim_in_metrics: bool,
) -> CellOutcome {
    let knob_json = tuple.to_canonical_json();
    let mut row: Vec<String> = vec![
        src_path.display().to_string(),
        cfg.codec.name().to_string(),
        q.to_string(),
        knob_json.clone(),
    ];

    // Encode.
    let cell = match encode(cfg.codec, source, q, &tuple.0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "[sweep] encode failed: {} q={q} knobs={knob_json}: {e}",
                src_path.display()
            );
            row.push("".to_string()); // encoded_bytes
            row.push("".to_string()); // encode_ms
            row.push("".to_string()); // decode_ms
            for m in &cfg.metrics {
                for _ in m.column_names() {
                    row.push("".to_string());
                }
            }
            return CellOutcome::EncodeFailed { row };
        }
    };

    row.push(cell.bytes.len().to_string());
    row.push(format!("{:.3}", cell.encode_ms));

    // Decode-back through the path-based decoder for format-sniff parity
    // with production. Tempfile lifetime ends when this function returns.
    let decode_start = Instant::now();
    let decoded = match decode_encoded_bytes(&cell.bytes, cfg.codec) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "[sweep] decode-back failed: {} q={q} knobs={knob_json}: {e}",
                src_path.display()
            );
            row.push("".to_string()); // decode_ms
            for m in &cfg.metrics {
                for _ in m.column_names() {
                    row.push("".to_string());
                }
            }
            return CellOutcome::DecodeFailed { row };
        }
    };
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    row.push(format!("{decode_ms:.3}"));

    // Dimension check — skip scoring on size mismatch (chroma upsampling
    // bug or wrong pixel-format conversion).
    if decoded.width != source.width || decoded.height != source.height {
        eprintln!(
            "[sweep] dimension mismatch: {} q={q} src={}x{} decoded={}x{}",
            src_path.display(),
            source.width,
            source.height,
            decoded.width,
            decoded.height,
        );
        for m in &cfg.metrics {
            for _ in m.column_names() {
                row.push("".to_string());
            }
        }
        return CellOutcome::DecodeFailed { row };
    }

    // Score every selected metric.
    let mut any_score_failed = false;
    let zensim_features_wanted = cfg.feature_output.is_some() && zensim_in_metrics;
    let mut zensim_features: Option<(f32, Vec<f64>)> = None;
    for &metric in &cfg.metrics {
        let result: Result<Vec<(&'static str, f64)>, Box<dyn Error>> =
            if metric == MetricKind::Zensim && zensim_features_wanted {
                match run_zensim_with_features(source, &decoded) {
                    Ok((score, features)) => {
                        zensim_features = Some((score as f32, features));
                        Ok(vec![("zensim", score)])
                    }
                    Err(e) => Err(e),
                }
            } else {
                run_metric(metric, source, &decoded, cfg.gpu_runtime)
            };
        match result {
            Ok(values) => {
                for (_, v) in &values {
                    row.push(format!("{v:.6}"));
                }
            }
            Err(e) => {
                eprintln!(
                    "[sweep] metric {} failed on {} q={q}: {e}",
                    metric.name(),
                    src_path.display()
                );
                for _ in metric.column_names() {
                    row.push("".to_string());
                }
                any_score_failed = true;
            }
        }
    }

    let feature = zensim_features.map(|(score, features)| {
        (
            src_path.display().to_string(),
            cfg.codec.name(),
            q,
            knob_json.clone(),
            score,
            features,
        )
    });

    CellOutcome::Ok {
        row,
        feature,
        score_failed: any_score_failed,
    }
}

fn decode_encoded_bytes(bytes: &[u8], codec: CodecKind) -> Result<Rgb8Image, Box<dyn Error>> {
    // Path-based decode_image_to_rgb8 sniffs format and dispatches through
    // the per-codec decoder. We write to a tempfile to reuse it unchanged.
    // Performance: write+read is on the order of microseconds for
    // typical encoded sizes (10 KB - 1 MB) and dominates neither encode
    // nor decode wall time, so we don't optimize this away.
    let suffix = match codec {
        CodecKind::Zenpng => ".png",
        CodecKind::Zenjpeg => ".jpg",
        CodecKind::Zenwebp => ".webp",
        CodecKind::Zenavif => ".avif",
        CodecKind::Zenjxl => ".jxl",
    };
    let tmp = tempfile::Builder::new()
        .prefix("zen-metrics-sweep-")
        .suffix(suffix)
        .tempfile()?;
    std::fs::write(tmp.path(), bytes)?;
    decode_image_to_rgb8(tmp.path())
}

fn write_header(
    wtr: &mut csv::Writer<std::fs::File>,
    metrics: &[MetricKind],
) -> Result<(), Box<dyn Error>> {
    let mut headers: Vec<String> = vec![
        "image_path".to_string(),
        "codec".to_string(),
        "q".to_string(),
        "knob_tuple_json".to_string(),
        "encoded_bytes".to_string(),
        "encode_ms".to_string(),
        "decode_ms".to_string(),
    ];
    // Each metric expands to one column per name in `column_names()`. For
    // most metrics that's a single column; butteraugli (CPU and GPU) emits
    // two — `butteraugli_max{,_gpu}` + `butteraugli_pnorm3{,_gpu}`. The
    // sweep TSV prefixes every score column with `score_` to disambiguate
    // from later columns the harness may add (per-cell timings, etc.).
    for m in metrics {
        for col in m.column_names() {
            headers.push(format!("score_{col}"));
        }
    }
    wtr.write_record(&headers)?;
    Ok(())
}

/// Aggregate counters from a sweep run. Useful for the EOM report.
#[derive(Debug, Clone, Copy, Default)]
pub struct SweepStats {
    pub cells_total: usize,
    pub cells_emitted: usize,
    pub cells_failed_encode: usize,
    pub cells_failed_decode: usize,
    pub cells_failed_score: usize,
}

/// Atomic counters used during the parallel sweep. Snapshotted into a
/// plain `SweepStats` once the sweep finishes.
struct AtomicSweepStats {
    cells_total: u64,
    cells_emitted: AtomicU64,
    cells_failed_encode: AtomicU64,
    cells_failed_decode: AtomicU64,
    cells_failed_score: AtomicU64,
}

impl AtomicSweepStats {
    fn new(cells_total: u64) -> Self {
        Self {
            cells_total,
            cells_emitted: AtomicU64::new(0),
            cells_failed_encode: AtomicU64::new(0),
            cells_failed_decode: AtomicU64::new(0),
            cells_failed_score: AtomicU64::new(0),
        }
    }

    fn add_emitted(&self) {
        self.cells_emitted.fetch_add(1, Ordering::Relaxed);
    }
    fn add_failed_encode(&self) {
        self.cells_failed_encode.fetch_add(1, Ordering::Relaxed);
    }
    fn add_failed_decode(&self, n: u64) {
        self.cells_failed_decode.fetch_add(n, Ordering::Relaxed);
    }
    fn add_failed_score(&self) {
        self.cells_failed_score.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> SweepStats {
        SweepStats {
            cells_total: self.cells_total as usize,
            cells_emitted: self.cells_emitted.load(Ordering::Relaxed) as usize,
            cells_failed_encode: self.cells_failed_encode.load(Ordering::Relaxed) as usize,
            cells_failed_decode: self.cells_failed_decode.load(Ordering::Relaxed) as usize,
            cells_failed_score: self.cells_failed_score.load(Ordering::Relaxed) as usize,
        }
    }
}
