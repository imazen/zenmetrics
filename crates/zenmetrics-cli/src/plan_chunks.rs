//! `plan-chunks` — slice a sweep into MANY granular, work-stealable chunks each
//! sized to ≤ a target wall-clock (default 5 minutes) and under a host-RAM
//! budget, using the SAME per-cell cost model `fleet-plan` uses.
//!
//! ## Why this exists
//!
//! The omni-chunk model the fleet runs (`zenfleet-sweep worker --mode omni`)
//! already gives work-stealing (token-race claim + stale-claim recovery in
//! `crates/zenfleet-vastai/src/worker/claim.rs`), perfect resumability (a chunk
//! whose omni sidecar already exists is skipped — `claim.rs` step 1), decode-once
//! (the inline pipeline groups a chunk's rows by `(codec, knob_tuple_json)` and
//! shares the source decode across every q/knob cell), and corruption-impossible
//! durability (the omni parquet is built completely on local disk, then uploaded
//! with ONE atomic S3/R2 PUT — an interrupted upload leaves nothing, never a
//! truncated sidecar). The ONE thing it lacked was *granular sizing*: the legacy
//! chunker used a flat `--cells-per-chunk` heuristic, and the Hetzner split
//! assigned ONE giant static chunk per box (≈5.4 h), so a dead box stranded its
//! whole multi-hour chunk and a fast box couldn't do more.
//!
//! This subcommand closes that gap. It reads the SAME input parquet the worker
//! consumes (`image_path / codec / q / knob_tuple_json`, image-major as
//! `generate_sweep_input.py` writes it) plus a per-image size source, then
//! greedily packs a *contiguous run* of images into each chunk so that the
//! chunk's estimated `Σ(encode + score)` wall time ≤ `--target-seconds` and its
//! peak host RAM ≤ `--mem-budget-mb`. Variable-size images naturally yield
//! variable images-per-chunk, balancing the queue. It emits the canonical
//! `chunks.jsonl` (one record per chunk) the worker already loops over.
//!
//! ## How the five requirements hold (granular + work-stealing + decode-once +
//! corruption-proof + resumable, simultaneously)
//!
//! 1. **Granular ≤5-min** — this sizer: `chunk_est_seconds(images) ≤ target`.
//! 2. **Work-stealing** — the worker's claim loop; a stale (dead-box) claim is
//!    re-stealable, so a sub-5-min chunk completes elsewhere.
//! 3. **Resumable** — completion = the per-chunk omni sidecar; a re-run skips
//!    chunks whose sidecar exists; the `gap` reconcile re-runs only the missing.
//! 4. **Decode-once** — the inline pipeline groups by source within a chunk;
//!    every image is decoded exactly once per chunk it appears in (and a small
//!    chunk only ever holds an image once).
//! 5. **Corruption-impossible** — build-local-then-single-PUT + idempotent skip.
//!
//! The estimators are the same ones `crate::fleet_plan` exposes:
//! - encode: [`crate::fleet_plan::codec_encode_estimate`] →
//!   `zencodec::estimate_encode_resources` (`wall_ms`, `peak_ram`);
//! - score: [`crate::fleet_plan::metric_score_estimate`] → per-metric GPU
//!   estimators (`time_ms`) when a `gpu-<metric>` feature is compiled in, else a
//!   configurable CPU per-megapixel rate (`--cpu-score-mp-ms`) so the sizer is
//!   usable in the CPU-encode (`sweep`) build the Hetzner half runs.

use std::error::Error;
use std::io::Write as _;
use std::path::PathBuf;

use clap::Parser;

use crate::fleet_plan::{codec_encode_estimate, metric_score_estimate, parse_codec};
use crate::metrics::MetricKind;
use crate::sweep::encode::CodecKind;

/// A source-image dimension (`WIDTH × HEIGHT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dim {
    pub(crate) w: u32,
    pub(crate) h: u32,
}

impl Dim {
    fn megapixels(self) -> f64 {
        (self.w as f64 * self.h as f64) / 1_000_000.0
    }
}

#[derive(Parser, Debug)]
pub struct PlanChunksArgs {
    /// Codec the sweep encodes with (`jpeg`/`webp`/`avif`/`png`/`gif`/`tiff`/
    /// `jxl`; the `zen`-prefixed names are also accepted). Drives the per-cell
    /// encode-cost estimate (`estimate_encode_resources`) used to size chunks.
    #[arg(long, value_parser = parse_codec)]
    codec: CodecKind,

    /// Sweep run id (used in the emitted chunk_id, sidecar, and encoded prefixes
    /// — must match what the launcher passes the worker as `SWEEP_RUN_ID`).
    #[arg(long)]
    run_id: String,

    /// The input parquet the worker consumes: rows of
    /// `image_path / codec / q / knob_tuple_json`, image-major (the layout
    /// `generate_sweep_input.py` writes). Each image must carry the SAME number
    /// of cells (the worker's row-range math requires uniform cells/image).
    #[arg(long)]
    input_parquet: PathBuf,

    /// R2 URI the worker fetches the input parquet from
    /// (`s3://bucket/prefix/<name>.parquet`). Written verbatim into each chunk.
    #[arg(long)]
    input_parquet_r2: String,

    /// R2 prefix holding the source images (written into each chunk as
    /// `source_dir_r2`).
    #[arg(long)]
    source_dir_r2: String,

    /// R2 prefix under which omni sidecars + encoded variants land. Each chunk
    /// gets `<prefix>/<run_id>/omni/<chunk_id>.parquet` and
    /// `<prefix>/<run_id>/encoded/<chunk_id>/`. Defaults to `s3://zentrain`,
    /// matching the worker's synthesized default.
    #[arg(long, default_value = "s3://zentrain")]
    out_prefix: String,

    /// Where to write the emitted `chunks.jsonl` (one record per chunk).
    #[arg(long)]
    out: PathBuf,

    /// Per-image sizes, TSV `basename<TAB>WIDTHxHEIGHT` (one per line, `#`
    /// comments allowed). When omitted, sizes are parsed from `scaleWxH` in each
    /// image basename (the picker-corpus convention), falling back to
    /// `--default-size` for any name without it.
    #[arg(long)]
    sizes_tsv: Option<PathBuf>,

    /// Fallback size for any image whose dimensions aren't otherwise known
    /// (`WIDTHxHEIGHT`). Defaults to 1024x1024.
    #[arg(long, default_value = "1024x1024")]
    default_size: String,

    /// Target wall-clock per chunk, seconds. The sizer packs images until adding
    /// the next would exceed this. Default 300 (5 minutes) per the task: a dead
    /// box then loses ≤5 min, and a fast box keeps claiming.
    #[arg(long, default_value = "300")]
    target_seconds: f64,

    /// Per-cell metric set the chunk is scored with (repeat or comma-separate).
    /// Each cell's score time is the sum over these metrics. GPU metrics use the
    /// per-crate estimator (`gpu-<metric>` feature); CPU metrics (or any metric
    /// in a CPU-only build) use the `--cpu-score-mp-ms` rate.
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1..)]
    metrics: Vec<MetricKind>,

    /// CPU score cost per megapixel, milliseconds — used for any metric whose
    /// GPU estimator isn't compiled in (so the sizer works in the CPU-encode
    /// `sweep` build the Hetzner half runs). A conservative default; override
    /// from a measured per-metric rate when you have one.
    #[arg(long, default_value = "120")]
    cpu_score_mp_ms: f64,

    /// CPU cores per box, used to scale the codec's encode `wall_ms` (the codec
    /// estimate is `at_cores`-scaled internally). Match the worker's box class.
    #[arg(long, default_value = "16")]
    cores_per_box: u32,

    /// Per-chunk host-RAM budget, MiB. The sizer also caps a chunk so its
    /// estimated peak RAM (max single-cell encode peak, since the worker bounds
    /// concurrent encodes) stays under this — the memory-aware bound that keeps
    /// jxl-modular chunks from ramping the per-process allocator high-water into
    /// an OOM. A chunk is never empty: at least one image always lands even if a
    /// single image already exceeds the budget (it's flagged on stderr).
    #[arg(long, default_value = "20000")]
    mem_budget_mb: u64,

    /// Encode threads to assume per cell when computing the per-cell encode time
    /// (overrides the codec's thread estimate). Rarely needed; the codec
    /// estimate is `cores`-scaled already.
    #[arg(long)]
    encode_threads: Option<u32>,
}

/// Parse a `WIDTHxHEIGHT` size token.
fn parse_dim(s: &str) -> Result<Dim, String> {
    let (w, h) = s
        .split_once(['x', 'X', '*'])
        .ok_or_else(|| format!("size '{s}' must be WIDTHxHEIGHT (e.g. 1024x768)"))?;
    let w: u32 = w
        .trim()
        .parse()
        .map_err(|_| format!("bad width in '{s}'"))?;
    let h: u32 = h
        .trim()
        .parse()
        .map_err(|_| format!("bad height in '{s}'"))?;
    if w == 0 || h == 0 {
        return Err(format!("size '{s}' has a zero dimension"));
    }
    Ok(Dim { w, h })
}

/// Pull `scaleWxH` out of a picker-corpus-style basename (e.g.
/// `…scale1280x960…png`). `None` when the name carries no such token.
fn dim_from_name(name: &str) -> Option<Dim> {
    let i = name.find("scale")?;
    let rest = &name[i + "scale".len()..];
    // Take the leading `<digits>x<digits>` run.
    let mut chars = rest.char_indices();
    let mut wend = 0;
    for (idx, c) in chars.by_ref() {
        if c.is_ascii_digit() {
            wend = idx + c.len_utf8();
        } else if c == 'x' || c == 'X' {
            break;
        } else {
            return None;
        }
    }
    let w: u32 = rest[..wend].parse().ok()?;
    let after_x = &rest[wend + 1..];
    let mut hend = 0;
    for (idx, c) in after_x.char_indices() {
        if c.is_ascii_digit() {
            hend = idx + c.len_utf8();
        } else {
            break;
        }
    }
    let h: u32 = after_x[..hend].parse().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    Some(Dim { w, h })
}

/// Resolve the dimension for `basename`: the sizes TSV first, then a `scaleWxH`
/// token in the name, then the `--default-size` fallback.
fn resolve_dim(
    basename: &str,
    sizes: &std::collections::HashMap<String, Dim>,
    default: Dim,
) -> Dim {
    if let Some(d) = sizes.get(basename) {
        return *d;
    }
    dim_from_name(basename).unwrap_or(default)
}

/// One chunk record — the exact `chunks.jsonl` shape the worker
/// (`zenfleet-vastai::worker::chunk::ChunkRecord` + the inline pipeline) reads.
#[derive(serde::Serialize)]
struct ChunkSpec {
    chunk_id: String,
    input_parquet: String,
    input_parquet_r2: String,
    /// Half-open `[start, end)` row slice — contiguous in image space.
    row_range: [usize; 2],
    source_dir_r2: String,
    image_basenames: Vec<String>,
    run_id: String,
    out_sidecar_omni: String,
    out_encoded_prefix: String,
}

/// Per-cell cost (encode + score) the sizer accumulates.
struct CellCost {
    /// Encode wall time for one cell, seconds (`cores`-scaled).
    encode_s: f64,
    /// Score wall time for one cell, seconds (Σ over metrics).
    score_s: f64,
    /// Worst-case single-encode peak RAM at this size, bytes.
    peak_ram_bytes: u64,
}

/// Read the input parquet's `image_path` column (image-major) into the ordered
/// list of distinct basenames + the uniform cells-per-image count.
///
/// The worker requires every image to carry the SAME number of cells so a
/// chunk's `row_range = [img_start * cpi, img_end * cpi]` is exact. We verify
/// that here and error loudly otherwise — the same invariant
/// `generate_sweep_input.py::rows_from_cells_jsonl` enforces.
fn read_images_and_cpi(path: &std::path::Path) -> Result<(Vec<String>, usize), Box<dyn Error>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let file = std::fs::File::open(path)
        .map_err(|e| format!("open input parquet {}: {e}", path.display()))?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| format!("read input parquet {}: {e}", path.display()))?;

    // Stream the `image_path` column row by row, preserving the file's row order
    // (which is image-major). Collect distinct basenames in first-seen order and
    // count per-image occurrences (= cells/image).
    let mut order: Vec<String> = Vec::new();
    let mut per_image: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for row in reader.get_row_iter(None)? {
        let row = row?;
        // The column may be named "image_path"; pull it by name, tolerating the
        // parquet `Row` API's column accessor.
        let mut image_path: Option<String> = None;
        for (name, field) in row.get_column_iter() {
            if name == "image_path" {
                image_path = Some(field.to_string().trim_matches('"').to_string());
                break;
            }
        }
        let p = image_path.ok_or("input parquet has no `image_path` column")?;
        let base = std::path::Path::new(&p)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(p);
        if !per_image.contains_key(&base) {
            order.push(base.clone());
        }
        *per_image.entry(base).or_insert(0) += 1;
    }
    if order.is_empty() {
        return Err("input parquet produced no rows".into());
    }
    let counts: std::collections::BTreeSet<usize> = per_image.values().copied().collect();
    if counts.len() != 1 {
        return Err(format!(
            "non-uniform cells/image {:?} — the chunk row-range math requires every \
             image to carry the same number of cells (image-major input parquet from \
             generate_sweep_input.py / --emit-cells guarantees this)",
            counts
        )
        .into());
    }
    let cpi = *counts.iter().next().expect("one count");
    Ok((order, cpi))
}

/// Estimate the per-cell cost at one image size. Encode from the codec model;
/// score from the per-metric GPU estimators (when compiled in) or the CPU rate.
fn cell_cost(
    codec: CodecKind,
    dim: Dim,
    cores: u32,
    metrics: &[MetricKind],
    cpu_score_mp_ms: f64,
    encode_threads_override: Option<u32>,
) -> Result<CellCost, Box<dyn Error>> {
    let enc = codec_encode_estimate(codec, dim.w, dim.h, cores)?;
    // The codec's `wall_ms` is already `cores`-scaled. A thread override only
    // re-scales linearly off the codec's modelled thread count (best-effort; the
    // codec estimate is the authority).
    let encode_ms = match (encode_threads_override, enc.threads) {
        (Some(t), modelled) if t > 0 && modelled > 0 => {
            enc.ms as f64 * (modelled as f64 / t as f64)
        }
        _ => enc.ms as f64,
    };

    let mut score_ms: f64 = 0.0;
    for &m in metrics {
        match metric_score_estimate(m, dim.w, dim.h) {
            // GPU estimator compiled in → use its modelled time.
            Ok((_vram, time_ms)) => score_ms += time_ms as f64,
            // No GPU estimator for this metric in this build (CPU metric, or a
            // GPU metric whose feature is off) → CPU per-megapixel rate.
            Err(_) => score_ms += dim.megapixels() * cpu_score_mp_ms,
        }
    }

    Ok(CellCost {
        encode_s: encode_ms / 1000.0,
        score_s: score_ms / 1000.0,
        peak_ram_bytes: enc.peak_ram_bytes,
    })
}

/// A summary of the planned chunking, returned so tests + the human report can
/// assert on it without re-parsing the emitted file.
pub(crate) struct PlanSummary {
    pub(crate) n_images: usize,
    pub(crate) cells_per_image: usize,
    pub(crate) n_chunks: usize,
    pub(crate) total_cells: usize,
    /// Estimated wall seconds for the largest chunk.
    pub(crate) max_chunk_seconds: f64,
    /// Images that alone exceed `target_seconds` (still emitted as a 1-image
    /// chunk — flagged so the operator knows that chunk runs long).
    pub(crate) oversize_images: usize,
}

/// Core sizing: greedily pack a contiguous run of images into each chunk so the
/// chunk's `Σ(encode + score)` ≤ `target_seconds` and its peak RAM ≤
/// `mem_budget_bytes`. Pure over the (ordered images → per-image cost) input so
/// it is unit-testable without any parquet / R2.
///
/// Returns the chunk image-index ranges `[(start, end_exclusive), …]` and a
/// summary. A single image that alone exceeds the time or RAM budget still
/// becomes its own 1-image chunk (never silently dropped or merged away).
#[allow(clippy::too_many_arguments)]
fn pack_chunks(
    per_image_cost: &[CellCost],
    cells_per_image: usize,
    target_seconds: f64,
    mem_budget_bytes: u64,
) -> (Vec<(usize, usize)>, PlanSummary) {
    let n = per_image_cost.len();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut max_chunk_seconds = 0.0_f64;
    let mut oversize_images = 0usize;

    // One image's full cost = (encode + score) × cells_per_image.
    let img_seconds = |c: &CellCost| (c.encode_s + c.score_s) * cells_per_image as f64;

    let mut start = 0usize;
    while start < n {
        let mut end = start;
        let mut acc_seconds = 0.0_f64;
        while end < n {
            let c = &per_image_cost[end];
            let this = img_seconds(c);
            let would_be = acc_seconds + this;
            // Peak RAM is per-encode (the worker bounds concurrent encodes), so a
            // chunk's RAM ceiling is the max single-cell encode peak among its
            // images — not the sum. Adding an image only raises the ceiling if it
            // is larger.
            let ram_ok = c.peak_ram_bytes <= mem_budget_bytes;
            if end == start {
                // The chunk must contain at least one image. Take this one even
                // if it alone busts a budget, and flag it.
                if would_be > target_seconds || !ram_ok {
                    oversize_images += 1;
                }
                acc_seconds = would_be;
                end += 1;
                continue;
            }
            // Adding this image: stop if it would exceed the time budget or its
            // own encode peak exceeds the RAM budget.
            if would_be > target_seconds || !ram_ok {
                break;
            }
            acc_seconds = would_be;
            end += 1;
        }
        ranges.push((start, end));
        max_chunk_seconds = max_chunk_seconds.max(acc_seconds);
        start = end;
    }

    let summary = PlanSummary {
        n_images: n,
        cells_per_image,
        n_chunks: ranges.len(),
        total_cells: n * cells_per_image,
        max_chunk_seconds,
        oversize_images,
    };
    (ranges, summary)
}

/// Run `plan-chunks`: read the input parquet + sizes, estimate per-image cost,
/// pack contiguous-image chunks under the time + RAM budgets, write
/// `chunks.jsonl`, print a summary.
pub fn run(args: PlanChunksArgs) -> Result<(), Box<dyn Error>> {
    if args.metrics.is_empty() {
        return Err("at least one --metrics value is required".into());
    }
    if args.target_seconds <= 0.0 {
        return Err("--target-seconds must be positive".into());
    }
    let default_dim = parse_dim(&args.default_size)?;
    let mem_budget_bytes = args.mem_budget_mb.saturating_mul(1024 * 1024);

    // Optional sizes TSV: basename<TAB>WIDTHxHEIGHT.
    let mut sizes: std::collections::HashMap<String, Dim> = std::collections::HashMap::new();
    if let Some(p) = &args.sizes_tsv {
        let text = std::fs::read_to_string(p)
            .map_err(|e| format!("read sizes TSV {}: {e}", p.display()))?;
        for (ln, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split('\t');
            let (Some(name), Some(dim_s)) = (it.next(), it.next()) else {
                return Err(
                    format!("{}:{}: expected `basename<TAB>WxH`", p.display(), ln + 1).into(),
                );
            };
            let dim =
                parse_dim(dim_s.trim()).map_err(|e| format!("{}:{}: {e}", p.display(), ln + 1))?;
            let base = std::path::Path::new(name.trim())
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| name.trim().to_string());
            sizes.insert(base, dim);
        }
    }

    // Read the ordered image list + uniform cells/image from the input parquet.
    let (images, cells_per_image) = read_images_and_cpi(&args.input_parquet)?;

    // Per-image cost from the codec + metric estimators.
    let mut per_image_cost: Vec<CellCost> = Vec::with_capacity(images.len());
    for base in &images {
        let dim = resolve_dim(base, &sizes, default_dim);
        per_image_cost.push(cell_cost(
            args.codec,
            dim,
            args.cores_per_box,
            &args.metrics,
            args.cpu_score_mp_ms,
            args.encode_threads,
        )?);
    }

    let (ranges, summary) = pack_chunks(
        &per_image_cost,
        cells_per_image,
        args.target_seconds,
        mem_budget_bytes,
    );

    // Emit chunks.jsonl — the canonical record shape the worker loops over.
    let out_prefix = args.out_prefix.trim_end_matches('/');
    let input_name = args
        .input_parquet
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "input.parquet".to_string());
    let mut f = std::fs::File::create(&args.out)
        .map_err(|e| format!("create {}: {e}", args.out.display()))?;
    for (img_start, img_end) in &ranges {
        // chunk_id is image-index-keyed (zero-padded), stable across re-runs, and
        // self-describing — so a re-launch re-derives the SAME chunk_id, the
        // SAME sidecar URI, and the resumability skip just works.
        let chunk_id = format!("{}-{:06}", args.codec.name(), img_start);
        let spec = ChunkSpec {
            chunk_id: chunk_id.clone(),
            input_parquet: input_name.clone(),
            input_parquet_r2: args.input_parquet_r2.clone(),
            row_range: [img_start * cells_per_image, img_end * cells_per_image],
            source_dir_r2: args.source_dir_r2.trim_end_matches('/').to_string(),
            image_basenames: images[*img_start..*img_end].to_vec(),
            run_id: args.run_id.clone(),
            out_sidecar_omni: format!("{out_prefix}/{}/omni/{chunk_id}.parquet", args.run_id),
            out_encoded_prefix: format!("{out_prefix}/{}/encoded/{chunk_id}/", args.run_id),
        };
        let line = serde_json::to_string(&spec).map_err(|e| format!("serialize chunk: {e}"))?;
        writeln!(f, "{line}").map_err(|e| format!("write {}: {e}", args.out.display()))?;
    }
    f.flush()
        .map_err(|e| format!("flush {}: {e}", args.out.display()))?;

    // Human summary on stderr (stdout stays clean for piping).
    eprintln!(
        "plan-chunks: codec={} run={} | {} images × {} cells/image = {} cells",
        args.codec.name(),
        args.run_id,
        summary.n_images,
        summary.cells_per_image,
        summary.total_cells
    );
    eprintln!(
        "  → {} chunks (target ≤{:.0}s, mem ≤{} MiB); largest chunk est {:.1}s; \
         mean {:.1} images/chunk",
        summary.n_chunks,
        args.target_seconds,
        args.mem_budget_mb,
        summary.max_chunk_seconds,
        summary.n_images as f64 / summary.n_chunks.max(1) as f64,
    );
    if summary.oversize_images > 0 {
        eprintln!(
            "  ⚠ {} image(s) alone exceed the time/RAM budget — each is a 1-image \
             chunk that runs long; lower the q/knob grid or split those sources",
            summary.oversize_images
        );
    }
    eprintln!(
        "  wrote {} ({} chunks)",
        args.out.display(),
        summary.n_chunks
    );
    eprintln!(
        "  upload: aws s3 cp {} s3://coefficient/jobs/{}/chunks.jsonl",
        args.out.display(),
        args.run_id
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(encode_s: f64, score_s: f64, peak_ram_mb: u64) -> CellCost {
        CellCost {
            encode_s,
            score_s,
            peak_ram_bytes: peak_ram_mb * 1024 * 1024,
        }
    }

    #[test]
    fn parse_dim_basics() {
        assert_eq!(parse_dim("1024x768").unwrap(), Dim { w: 1024, h: 768 });
        assert_eq!(parse_dim("64X64").unwrap(), Dim { w: 64, h: 64 });
        assert!(parse_dim("1024").is_err());
        assert!(parse_dim("0x10").is_err());
    }

    #[test]
    fn dim_from_name_picker_corpus() {
        assert_eq!(
            dim_from_name("img-00001.scale1280x960.q80.png"),
            Some(Dim { w: 1280, h: 960 })
        );
        assert_eq!(
            dim_from_name("foo_scale64x64_bar"),
            Some(Dim { w: 64, h: 64 })
        );
        // No scale token → None (falls back to default).
        assert_eq!(dim_from_name("plain.png"), None);
        // Malformed → None.
        assert_eq!(dim_from_name("scaleXx10"), None);
    }

    #[test]
    fn megapixels_math() {
        assert!((Dim { w: 1000, h: 1000 }.megapixels() - 1.0).abs() < 1e-9);
        assert!((Dim { w: 2000, h: 1000 }.megapixels() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn packs_many_small_chunks_under_target() {
        // 10 images, each cell = 1s encode + 0.5s score, 1 cell/image → 1.5s/img.
        // target 5s → ~3 images/chunk.
        let costs: Vec<CellCost> = (0..10).map(|_| cc(1.0, 0.5, 100)).collect();
        let (ranges, summary) = pack_chunks(&costs, 1, 5.0, 32 * 1024);
        assert_eq!(summary.n_images, 10);
        assert_eq!(summary.cells_per_image, 1);
        // Every chunk's est time must be ≤ target (granularity guarantee).
        for (s, e) in &ranges {
            let secs: f64 = costs[*s..*e].iter().map(|c| c.encode_s + c.score_s).sum();
            assert!(
                secs <= 5.0 + 1e-9,
                "chunk [{s},{e}) est {secs}s exceeds 5s target"
            );
        }
        // Granular: MANY chunks, not 1.
        assert!(
            summary.n_chunks >= 3,
            "expected ≥3 chunks, got {}",
            summary.n_chunks
        );
        // Every image is covered exactly once, contiguously.
        let covered: usize = ranges.iter().map(|(s, e)| e - s).sum();
        assert_eq!(covered, 10);
        let mut prev_end = 0;
        for (s, e) in &ranges {
            assert_eq!(
                *s, prev_end,
                "ranges must be contiguous (decode-once layout)"
            );
            prev_end = *e;
        }
    }

    #[test]
    fn cells_per_image_multiplies_chunk_time() {
        // Same per-cell cost but 10 cells/image → each image is 15s → one image
        // already exceeds a 5s target, so every chunk is a single image and is
        // flagged oversize.
        let costs: Vec<CellCost> = (0..4).map(|_| cc(1.0, 0.5, 100)).collect();
        let (ranges, summary) = pack_chunks(&costs, 10, 5.0, 32 * 1024);
        assert_eq!(summary.n_chunks, 4, "each image its own chunk");
        assert_eq!(summary.oversize_images, 4);
        for (s, e) in &ranges {
            assert_eq!(e - s, 1);
        }
    }

    #[test]
    fn memory_budget_caps_chunk_even_when_time_allows() {
        // Cheap in time (0.1s/img) so time never bounds, but each encode peaks at
        // 25 GiB and the budget is 20 GiB → every image must be its own chunk and
        // be flagged oversize (the jxl-modular OOM guard).
        let costs: Vec<CellCost> = (0..5).map(|_| cc(0.05, 0.05, 25 * 1024)).collect();
        let (ranges, summary) = pack_chunks(&costs, 1, 300.0, 20 * 1024 * 1024 * 1024);
        assert_eq!(summary.n_chunks, 5, "RAM budget forces 1 image/chunk");
        assert_eq!(summary.oversize_images, 5);
        for (s, e) in &ranges {
            assert_eq!(e - s, 1);
        }
    }

    #[test]
    fn under_budget_ram_lets_images_pack() {
        // Small encodes (well under budget) pack by time only.
        let costs: Vec<CellCost> = (0..6).map(|_| cc(0.5, 0.0, 200)).collect();
        let (ranges, summary) = pack_chunks(&costs, 1, 2.0, 20 * 1024 * 1024 * 1024);
        // 0.5s/img, 2s target → 4 images/chunk → 2 chunks (4 + 2).
        assert_eq!(summary.n_chunks, 2);
        assert_eq!(ranges[0], (0, 4));
        assert_eq!(ranges[1], (4, 6));
        assert_eq!(summary.oversize_images, 0);
    }

    #[test]
    fn single_image_always_lands_even_if_oversize() {
        let costs = vec![cc(999.0, 0.0, 100)];
        let (ranges, summary) = pack_chunks(&costs, 1, 5.0, 32 * 1024);
        assert_eq!(summary.n_chunks, 1);
        assert_eq!(ranges[0], (0, 1));
        assert_eq!(summary.oversize_images, 1);
    }

    #[test]
    fn row_range_is_image_major_contiguous() {
        // Verify the row_range math a chunk would emit: image range × cpi.
        let costs: Vec<CellCost> = (0..5).map(|_| cc(1.0, 0.0, 100)).collect();
        let cpi = 7;
        let (ranges, _) = pack_chunks(&costs, cpi, 3.0, 32 * 1024);
        // 1s/cell × 7 cells = 7s/image > 3s → each image its own chunk.
        let mut expected_start = 0;
        for (s, e) in &ranges {
            let row_start = s * cpi;
            let row_end = e * cpi;
            assert_eq!(row_start, expected_start, "row ranges must abut");
            expected_start = row_end;
        }
        // Total rows = 5 images × 7 cells.
        assert_eq!(expected_start, 5 * cpi);
    }
}
