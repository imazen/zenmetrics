#![forbid(unsafe_code)]
//! `zenmetrics jobexec` — the reference executor for the zen job system's `ZEN_EXEC` contract.
//!
//! The worker (`zenfleet-worker::exec_command`) pipes ONE `DesiredJob` as JSON to this process's stdin
//! and content-addresses whatever we write to stdout. We do the real encode/score for that one cell:
//!
//!   stdin  <- {"kind":{"kind":"encode"|"metric",...}, "inputs":[...], "cell":{image_path,codec,q,knob_tuple_json}}
//!   stdout -> Encode: the encoded image bytes;  Metric: a one-line JSON score row
//!   exit 0  = success; non-zero = deterministic FAILED row.
//!
//! Source resolution (the source image named by `cell.image_path`):
//!
//! - `s3://…` path -> fetched with s5cmd
//! - else if `$ZEN_CORPUS_PREFIX` is set ->
//!   s3://$ZEN_CORPUS_BUCKET/$ZEN_CORPUS_PREFIX/<image_path> (falls back to $ZEN_BUCKET)
//! - else if the local file exists -> used directly
//!
//! The corpus is READ-ONLY and lives in its own bucket (`$ZEN_CORPUS_BUCKET`, e.g. codec-corpus),
//! separate from the run-write bucket (`$ZEN_BUCKET`, e.g. coefficient) the worker fills with
//! blobs/ledger/claims. When set, `$ZEN_CORPUS_AWS_*` supplies a read-only credential for the corpus
//! fetch so the run-write cred is never used to read the corpus (R2 temp creds are single-bucket, so
//! reading one bucket while writing another genuinely needs two creds).
//!
//! A `metric` job is self-contained: it re-encodes the cell (deterministic) and scores
//! (reference=source, distorted=encode). CPU metrics only (ssim2/butteraugli/zensim) — GPU metrics
//! need a GPU build/tier. This keeps the basement + CPU burst tiers fully useful.
use crate::decode::{Rgb8Image, decode_image_to_rgb8};
use crate::metrics::{GpuRuntime, MetricKind, run_metric};
use crate::sweep::encode::{CodecKind, encode};
use clap::{Parser, ValueEnum};
use serde_json::{Map, Value};
use std::error::Error;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Parser, Debug)]
pub struct JobexecArgs {
    /// R2 prefix under $ZEN_CORPUS_BUCKET (or $ZEN_BUCKET) where source images live (overrides
    /// $ZEN_CORPUS_PREFIX). The job's `cell.image_path` is appended to it. Omit if image_path is
    /// already an `s3://…` or local path.
    #[arg(long)]
    pub corpus_prefix: Option<String>,

    /// Persistent "warm" mode for the GPU job system. Instead of one job on stdin → exit, loop
    /// reading length-framed requests (`[u32 LE len][DesiredJob JSON]`) and writing length-framed
    /// responses (`[u8 status][u32 LE len][payload]`; status 0=ok/output-bytes, 1=job error, 2=panic
    /// with payload=message). The GPU client + compiled kernels stay warm across jobs (cubecl caches
    /// them per process), so CUDA init + kernel compilation are paid ONCE per box, not per job — the
    /// fix for the ~20s/job cold-spawn overhead. The worker's persistent handler drives this; clean
    /// EOF on stdin ends the loop (exit 0).
    #[arg(long)]
    pub serve: bool,
}

fn codec_from_name(name: &str) -> Result<CodecKind, Box<dyn Error>> {
    Ok(match name {
        "zenpng" => CodecKind::Zenpng,
        "zenjpeg" => CodecKind::Zenjpeg,
        "zenwebp" => CodecKind::Zenwebp,
        "zenavif" => CodecKind::Zenavif,
        "zenjxl" => CodecKind::Zenjxl,
        other => return Err(format!("unknown codec {other:?}").into()),
    })
}

fn ext_for(name: &str) -> &'static str {
    match name {
        "zenpng" => "png",
        "zenjpeg" => "jpg",
        "zenwebp" => "webp",
        "zenavif" => "avif",
        "zenjxl" => "jxl",
        _ => "bin",
    }
}

/// Map the job's metric string to a `MetricKind` (clap value names: ssim2, butteraugli, zensim,
/// cvvdp, …), tolerating the `ssimulacra2` alias. Then `run_metric` dispatches to the right backend —
/// CPU metrics work in a default build; GPU metrics return a clear "needs a GPU build" error there.
fn metric_kind(metric: &str) -> Result<MetricKind, Box<dyn Error>> {
    let canon = if metric == "ssimulacra2" {
        "ssim2"
    } else {
        metric
    };
    MetricKind::from_str(canon, true).map_err(|e| format!("unknown metric {metric:?}: {e}").into())
}

/// Score `(reference, distorted)` with `metric`, returning all `(column, value)` pairs run_metric
/// yields (butteraugli yields max-norm + 3-norm; most yield one).
fn score(
    metric: &str,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn Error>> {
    run_metric(metric_kind(metric)?, reference, distorted, GpuRuntime::Auto)
}

/// Point an s5cmd `Command` at the read-only corpus credential (`ZEN_CORPUS_AWS_*`) when one is set,
/// so corpus reads don't reuse the run-write cred. No-op when `ZEN_CORPUS_AWS_ACCESS_KEY_ID` is unset —
/// the command then inherits the ambient `AWS_*` (single-cred / single-bucket setups). When the corpus
/// cred is permanent (no session token), the ambient `AWS_SESSION_TOKEN` is removed so the run cred's
/// session can't leak onto the corpus access key.
fn apply_corpus_creds(cmd: &mut Command) {
    let Ok(ak) = std::env::var("ZEN_CORPUS_AWS_ACCESS_KEY_ID") else {
        return;
    };
    cmd.env("AWS_ACCESS_KEY_ID", ak);
    if let Ok(sk) = std::env::var("ZEN_CORPUS_AWS_SECRET_ACCESS_KEY") {
        cmd.env("AWS_SECRET_ACCESS_KEY", sk);
    }
    match std::env::var("ZEN_CORPUS_AWS_SESSION_TOKEN") {
        Ok(st) => {
            cmd.env("AWS_SESSION_TOKEN", st);
        }
        Err(_) => {
            cmd.env_remove("AWS_SESSION_TOKEN");
        }
    }
}

/// Resolve `cell.image_path` to a readable local file, fetching from R2 if needed.
fn resolve_source(
    image_path: &str,
    corpus_prefix: Option<&str>,
) -> Result<PathBuf, Box<dyn Error>> {
    let local = PathBuf::from(image_path);
    if corpus_prefix.is_none() && !image_path.starts_with("s3://") && local.exists() {
        return Ok(local);
    }
    let endpoint = std::env::var("ZEN_R2_ENDPOINT")
        .map_err(|_| "ZEN_R2_ENDPOINT unset — cannot fetch source from R2")?;
    let uri = if image_path.starts_with("s3://") {
        image_path.to_string()
    } else {
        // The corpus (read-only source images) lives in its own bucket, distinct from the
        // run-write bucket the worker fills with blobs/ledger/claims. Read it from
        // ZEN_CORPUS_BUCKET when set, falling back to ZEN_BUCKET for single-bucket setups.
        let bucket = std::env::var("ZEN_CORPUS_BUCKET")
            .or_else(|_| std::env::var("ZEN_BUCKET"))
            .map_err(|_| "ZEN_CORPUS_BUCKET/ZEN_BUCKET unset")?;
        match corpus_prefix {
            Some(p) if !p.is_empty() => {
                format!("s3://{bucket}/{}/{image_path}", p.trim_end_matches('/'))
            }
            _ => format!("s3://{bucket}/{image_path}"),
        }
    };
    let dst = std::env::temp_dir().join(format!(
        "jobexec_src_{}_{}",
        std::process::id(),
        image_path.rsplit('/').next().unwrap_or("src")
    ));
    // Warm-process source cache: in --serve mode one process scores many cells/metrics of the SAME
    // source image (the manifest is image-major), and after the executor is kept warm the per-job R2
    // download is the dominant cost. So reuse an already-fetched image instead of re-downloading. `dst`
    // exists ONLY after a verified-complete download (we fetch to a sibling `.part` and rename on
    // success), so a cache hit is always a whole file — never a truncated/partial one.
    if dst.exists() {
        return Ok(dst);
    }
    let part = std::path::PathBuf::from(format!("{}.part", dst.display()));
    let mut cmd = Command::new("s5cmd");
    cmd.arg("--endpoint-url")
        .arg(&endpoint)
        .arg("cp")
        .arg(&uri)
        .arg(&part)
        // s5cmd prints a "cp …" line to stdout. In --serve mode stdout is the length-framed response
        // channel, so that line would corrupt a frame and deadlock the worker; in single-shot mode it
        // prefixes the content-addressed blob with noise. Silence stdout — real errors stay on s5cmd's
        // stderr (inherited → the worker log).
        .stdout(Stdio::null());
    // Use the read-only corpus credential (ZEN_CORPUS_AWS_*) for the corpus fetch when provided, so a
    // worker reads codec-corpus read-only while writing the run to a different bucket with the ambient
    // AWS_* run cred. No-op (inherits ambient AWS_*) when unset — single-cred / single-bucket setups.
    apply_corpus_creds(&mut cmd);
    let st = cmd.status().map_err(|e| format!("spawn s5cmd: {e}"))?;
    if !st.success() {
        let _ = std::fs::remove_file(&part);
        return Err(format!("s5cmd cp {uri} failed").into());
    }
    std::fs::rename(&part, &dst).map_err(|e| format!("rename {part:?} -> {dst:?}: {e}"))?;
    Ok(dst)
}

/// Encode the job's cell — used by BOTH job kinds (`metric` re-encodes
/// deterministically before scoring).
///
/// Plan-driven cells carry `{"cell": <stratum-id>, "fp": <hex>,
/// "plan": <name>}` in `knob_tuple_json` (what `--plan` sweeps and
/// `--emit-cells` declare manifests write). Those are self-describing:
/// the config is reconstructed from the stratum id and verified against
/// the carried resolved-state fingerprint
/// (`sweep::plan::resolve_verified`), so id-grammar drift between the
/// declaring and executing builds is a loud deterministic failure —
/// never a silently wrong encode. Everything else goes through the
/// per-codec knob vocabulary as before.
fn encode_cell_for_job(
    codec: CodecKind,
    reference: &Rgb8Image,
    q: f64,
    knob_json: &str,
    knobs: &Map<String, Value>,
) -> Result<crate::sweep::encode::EncodedCell, Box<dyn Error>> {
    let plan_identity = (knobs.contains_key("plan"))
        .then(|| {
            let cell = knobs.get("cell").and_then(Value::as_str)?;
            let fp = knobs.get("fp").and_then(Value::as_str)?;
            Some((cell, fp))
        })
        .flatten();
    if let Some((cell_id, fp_hex)) = plan_identity {
        let _ = knob_json;
        #[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
        {
            // Codec dispatch + fingerprint verification live in
            // sweep::plan::resolve_verified; unsupported codecs error
            // there with the feature that would enable them.
            let cfg = crate::sweep::plan::resolve_verified(codec, cell_id, q as f32, fp_hex)?;
            return Ok(cfg.encode_bytes(reference)?);
        }
        #[cfg(not(all(feature = "sweep", any(feature = "jpeg", feature = "avif"))))]
        return Err(format!(
            "plan cell {cell_id:?} (fp {fp_hex}) needs a build with --features sweep \
             and the codec feature (jpeg/avif)"
        )
        .into());
    }
    encode(codec, reference, q, knobs)
}

/// Monotonic per-job sequence so the warm serve loop (which reuses one pid) never collides on the
/// distorted temp file.
static DIST_SEQ: AtomicU64 = AtomicU64::new(0);

/// One variant's location in `variants.tar`: byte `offset`, `size`, and the tar member basename
/// (`name`, empty for legacy 3-column indices). `name` powers the TAR-SHARD path
/// (`$ZEN_VARIANTS_LOCAL_DIR`): when a worker has already pulled + extracted the tar locally, it
/// reads `<local_dir>/<name>` off disk instead of a per-variant byte-range GET.
#[derive(Clone, Debug)]
struct VariantLoc {
    offset: u64,
    size: u64,
    name: String,
}

/// Variant index: `sha -> VariantLoc` into `variants.tar`. Loaded once per process (the warm
/// `--serve` loop reuses it). TSV `sha\toffset\tsize[\tname]` at `$ZEN_VARIANT_INDEX_URI`, built by
/// the manifest builder from the tar's member headers. Two fetch modes share this index:
///   * byte-range GET from R2 (`$ZEN_VARIANTS_TAR_URI`) — no whole-tar download, one R2 request/variant;
///   * TAR-SHARD local read (`$ZEN_VARIANTS_LOCAL_DIR`) — the worker pulled + extracted the tar ONCE
///     and reads `<dir>/<name>` off disk, ZERO per-variant R2 requests (the I/O-bound fix).
/// The optional 4th `name` column is what enables the local path; legacy 3-column indices still parse
/// (name = "") and fall back to the byte-range GET.
static VARIANT_INDEX: std::sync::OnceLock<std::collections::HashMap<String, VariantLoc>> =
    std::sync::OnceLock::new();

fn variant_index() -> Result<&'static std::collections::HashMap<String, VariantLoc>, Box<dyn Error>>
{
    if let Some(i) = VARIANT_INDEX.get() {
        return Ok(i);
    }
    let uri = std::env::var("ZEN_VARIANT_INDEX_URI").map_err(|_| "ZEN_VARIANT_INDEX_URI unset")?;
    let endpoint = std::env::var("ZEN_R2_ENDPOINT").map_err(|_| "ZEN_R2_ENDPOINT unset")?;
    let dst = std::env::temp_dir().join("zen_variant_index.tsv");
    let st = Command::new("s5cmd")
        .arg("--endpoint-url")
        .arg(&endpoint)
        .arg("cp")
        .arg(&uri)
        .arg(&dst)
        .stdout(Stdio::null())
        .status()
        .map_err(|e| format!("spawn s5cmd (index): {e}"))?;
    if !st.success() {
        return Err(format!("fetch variant index {uri} failed").into());
    }
    Ok(VARIANT_INDEX.get_or_init(|| parse_variant_index(&std::fs::read_to_string(&dst).unwrap_or_default())))
}

/// Parse a `sha\toffset\tsize[\tname]` variant-index TSV into `sha -> VariantLoc`. The 4th `name`
/// column is optional (legacy 3-column indices parse with `name = ""`). Factored out so it is unit
/// testable without R2.
fn parse_variant_index(tsv: &str) -> std::collections::HashMap<String, VariantLoc> {
    let mut m = std::collections::HashMap::new();
    for line in tsv.lines() {
        let mut it = line.split('\t');
        if let (Some(s), Some(o), Some(z)) = (it.next(), it.next(), it.next())
            && let (Ok(offset), Ok(size)) = (o.parse::<u64>(), z.parse::<u64>())
        {
            let name = it.next().unwrap_or("").to_string();
            m.insert(s.to_string(), VariantLoc { offset, size, name });
        }
    }
    m
}

/// Resolve one pre-encoded variant to a local path, WITHOUT re-encoding. Two modes, picked by env:
///
///   * **TAR-SHARD (`$ZEN_VARIANTS_LOCAL_DIR`)** — the worker already pulled + extracted the per-box
///     tar ONCE, so the variant is a file at `<local_dir>/<name>`. Returns that path directly (no
///     copy, no R2). This is the I/O-bound fix: a box that scores every variant in a tar issues ZERO
///     per-variant R2 requests (one tar GET at onstart instead of N range-GETs). Needs the index's
///     4th `name` column; falls through to the byte-range GET if the name is unknown or the local
///     file is missing (e.g. a partial extract), so a stale local dir degrades gracefully rather than
///     failing the cell.
///   * **byte-range GET (`$ZEN_VARIANTS_TAR_URI`)** — fetch just this variant's bytes out of the
///     remote tar via the index — no whole-tar download, one R2 request/variant. The pre-shard path.
///
/// Returns the local path plus whether the caller OWNS it (must delete after decode). TAR-SHARD reads
/// are borrowed (do NOT delete — they belong to the shared extract dir); range-GETs are owned temps.
fn fetch_variant(sha: &str, ext: &str) -> Result<(PathBuf, bool), Box<dyn Error>> {
    let loc = variant_index()?
        .get(sha)
        .ok_or_else(|| format!("sha {sha} not in variant index"))?;
    let (off, sz) = (loc.offset, loc.size);
    // TAR-SHARD: prefer the locally-extracted member when the worker pre-pulled the tar.
    if let Ok(dir) = std::env::var("ZEN_VARIANTS_LOCAL_DIR")
        && !loc.name.is_empty()
    {
        let p = std::path::Path::new(&dir).join(&loc.name);
        if p.is_file() {
            return Ok((p, false)); // borrowed — shared extract dir, do not delete
        }
    }
    let tar_uri =
        std::env::var("ZEN_VARIANTS_TAR_URI").map_err(|_| "ZEN_VARIANTS_TAR_URI unset")?;
    let endpoint = std::env::var("ZEN_R2_ENDPOINT").map_err(|_| "ZEN_R2_ENDPOINT unset")?;
    let bucket = std::env::var("ZEN_BUCKET").map_err(|_| "ZEN_BUCKET unset")?;
    let key = tar_uri
        .strip_prefix(&format!("s3://{bucket}/"))
        .unwrap_or(&tar_uri)
        .to_string();
    let seq = DIST_SEQ.fetch_add(1, Ordering::Relaxed);
    let dst = std::env::temp_dir().join(format!(
        "jobexec_var_{}_{}.{}",
        std::process::id(),
        seq,
        ext
    ));
    let end = off + sz - 1;
    let st = Command::new("aws")
        .arg("s3api")
        .arg("get-object")
        .arg("--endpoint-url")
        .arg(&endpoint)
        .arg("--bucket")
        .arg(&bucket)
        .arg("--key")
        .arg(&key)
        .arg("--range")
        .arg(format!("bytes={off}-{end}"))
        .arg(&dst)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("spawn aws (range-get): {e}"))?;
    if !st.success() {
        return Err(format!("range-get variant {sha} from {tar_uri} failed").into());
    }
    Ok((dst, true)) // owned temp — caller deletes after decode
}

/// PART 2 — warm-reference batch scoring for orchestrator-eligible metrics.
///
/// All variants in a ScoreFile job share ONE reference. The one-shot path (`run_metric` per
/// (variant, metric)) re-uploads that reference to the GPU on EVERY call — nsys shows
/// `cuMemcpyHtoDAsync` at 54% of CUDA API time and GPU sm util ~10% (upload-bound). This routes the
/// orchestrator-eligible metrics (everything except butteraugli, which is
/// `metric_orchestrator_eligible == false`) through `Orchestrator::run_all`, which groups tasks by
/// `ref_hash` and warm-holds the reference precompute device-resident across the group — so the ref
/// uploads ONCE per source and only the distorted side uploads per variant.
///
/// Returns the emitted JSONL rows for the eligible metrics. Butteraugli + zensim-with-features stay
/// on the caller's inline decode-reuse loop (butter is orchestrator-ineligible; zensim needs its
/// 372-feature sidecar which the umbrella score path doesn't emit). Decode is shared: the caller
/// passes the already-decoded `(sha, Rgb8Image)` pairs so no variant is decoded twice.
///
/// Gated by `ZEN_SCOREFILE_WARMREF=1` at the call site; default OFF = byte-identical one-shot path.
#[cfg(all(feature = "orchestrator", feature = "orchestrator-cuda"))]
#[allow(clippy::type_complexity)]
fn warmref_score_eligible(
    image_path: &str,
    codec_name: &str,
    reference: &Rgb8Image,
    decoded: &[(String, Rgb8Image)],
    metrics: &[&str],
) -> Result<Vec<String>, Box<dyn Error>> {
    use crate::orchestrator_runner::{
        build_orchestrator, metric_orchestrator_eligible, rekey_orchestrator_columns,
    };
    use zenmetrics_orchestrator::{Task, TaskData};

    // Which requested metrics are orchestrator-eligible? (butteraugli/-gpu are not.) zensim variants
    // are handled by the inline feature path, NOT here, so exclude them too.
    let eligible: Vec<&str> = metrics
        .iter()
        .copied()
        .filter(|m| {
            *m != "zensim" && *m != "zensim-gpu" && {
                match metric_kind(m).ok().and_then(cli_kind_from_metric_kind) {
                    Some(k) => metric_orchestrator_eligible(k),
                    None => false,
                }
            }
        })
        .collect();
    if eligible.is_empty() {
        return Ok(Vec::new());
    }

    let opts = crate::orchestrator_glue::OrchestratorRuntimeOpts::default();
    let mut orch = build_orchestrator(&opts)?;

    // Build the (variant × eligible-metric) task matrix. task_id encodes (variant_idx, metric_idx)
    // so we can correlate the completion-ordered results back to the right row. All tasks carry the
    // SAME reference bytes -> run_all groups them by ref_hash -> warm-ref hits.
    let w = reference.width;
    let h = reference.height;
    let mut tasks: Vec<Task> = Vec::with_capacity(decoded.len() * eligible.len());
    for (vi, (_sha, dist)) in decoded.iter().enumerate() {
        // Skip variants whose dims differ from the reference — the umbrella requires equal dims and
        // would error per-task anyway; emitting a clean error row keeps parity with the inline path.
        for (mi, m) in eligible.iter().enumerate() {
            let Some(cli_kind) = metric_kind(m).ok().and_then(cli_kind_from_metric_kind) else {
                continue;
            };
            let _ = cli_kind; // kind resolved again below via spec; kept for the eligibility gate
            tasks.push(Task {
                task_id: (vi as u64) << 20 | (mi as u64),
                ref_data: TaskData::Srgb8(reference.pixels.clone()),
                dist_data: TaskData::Srgb8(dist.pixels.clone()),
                width: w.max(dist.width),
                height: h.max(dist.height),
                metric: crate::orchestrator_glue::OrchestratorMetricSpec::from_cli(cli_kind).kind,
                params: None,
                ref_hash: 0,
            });
        }
    }

    // Accumulate results per (variant_idx) -> merged column map across its eligible metrics, so we
    // emit ONE row per (variant, metric) matching the inline path's shape.
    let mut rows: Vec<String> = Vec::with_capacity(tasks.len());
    for res in orch.run_all(tasks) {
        let vi = (res.task_id >> 20) as usize;
        let mi = (res.task_id & 0xFFFFF) as usize;
        let sha = &decoded[vi].0;
        let m = eligible[mi];
        let cli_kind = metric_kind(m).ok().and_then(cli_kind_from_metric_kind);
        let mut o = Map::new();
        o.insert("kind".into(), serde_json::json!("metric"));
        o.insert("image_path".into(), serde_json::json!(image_path));
        o.insert("codec".into(), serde_json::json!(codec_name));
        o.insert("encode_sha".into(), serde_json::json!(sha));
        o.insert("metric".into(), serde_json::json!(m));
        match res.outcome {
            Ok(score) => {
                let mut scores = Map::new();
                if res.output_columns.is_empty() {
                    scores.insert(m.replace('-', "_"), serde_json::json!(score.value));
                } else if let Some(k) = cli_kind {
                    for (col, val) in rekey_orchestrator_columns(k, &res.output_columns) {
                        scores.insert(col, serde_json::json!(val));
                    }
                } else {
                    for (col, val) in &res.output_columns {
                        scores.insert(col.clone(), serde_json::json!(*val));
                    }
                }
                o.insert("score".into(), serde_json::json!(score.value));
                o.insert("scores".into(), Value::Object(scores));
            }
            Err(e) => {
                o.insert("error".into(), serde_json::json!(e.to_string()));
            }
        }
        rows.push(serde_json::to_string(&Value::Object(o))?);
    }
    Ok(rows)
}

/// Map a `zenmetrics_api::MetricKind` (jobexec's metric enum) to the CLI's `CliMetricKind` used by
/// the orchestrator eligibility + column-rekey helpers. Returns `None` for kinds the orchestrator
/// runner doesn't model.
#[cfg(all(feature = "orchestrator", feature = "orchestrator-cuda"))]
fn cli_kind_from_metric_kind(k: MetricKind) -> Option<crate::metrics::MetricKind> {
    // jobexec's `MetricKind` (re-exported from crate::metrics) IS the CLI metric kind, so this is
    // identity — kept as a named seam in case the two enums ever diverge.
    Some(k)
}

/// Whole-file scoring (`JobKind::ScoreFile`): the efficient path. Decode the reference ONCE, then for
/// each input variant sha byte-range-fetch the pre-encoded bytes out of `variants.tar`, decode it
/// ONCE, and score EVERY metric against the shared reference via the same `run_metric` core
/// `score-pairs` uses — so a 24 MP source's decode and each variant's decode happen once, never
/// re-encoded or re-decoded per metric. Emits one JSONL row per (variant, metric); the write-back
/// rejoins q/knobs by `encode_sha`.
///
/// PART 2: with `ZEN_SCOREFILE_WARMREF=1` (and an orchestrator-cuda build), the orchestrator-eligible
/// metrics are scored via one warm-reference `run_all` batch (ref uploaded once per source, not per
/// variant — the fix for the 54% H2D / ~10% GPU util); butteraugli + zensim-with-features stay on the
/// inline path. Default OFF = byte-identical one-shot behaviour.
fn run_score_file(job: &Value, corpus_prefix: Option<&str>) -> Result<Vec<u8>, Box<dyn Error>> {
    let cell = &job["cell"];
    let image_path = cell["image_path"]
        .as_str()
        .ok_or("score_file: cell.image_path missing")?;
    let codec_name = cell["codec"]
        .as_str()
        .ok_or("score_file: cell.codec missing")?;
    let ext = ext_for(codec_name);
    let metrics: Vec<&str> = job["kind"]["metrics"]
        .as_array()
        .ok_or("score_file: kind.metrics missing")?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let shas: Vec<&str> = job["inputs"]
        .as_array()
        .ok_or("score_file: inputs missing")?
        .iter()
        .filter_map(Value::as_str)
        .collect();

    // Decode the reference ONCE for all variants × metrics.
    let src_path = resolve_source(image_path, corpus_prefix)?;
    let reference = decode_image_to_rgb8(&src_path)?;

    let mut rows: Vec<String> = Vec::with_capacity(shas.len() * metrics.len().max(1));
    let mk_row = |sha: &str, extra: Value| -> Result<String, Box<dyn Error>> {
        let mut o = Map::new();
        o.insert("kind".into(), serde_json::json!("metric"));
        o.insert("image_path".into(), serde_json::json!(image_path));
        o.insert("codec".into(), serde_json::json!(codec_name));
        o.insert("encode_sha".into(), serde_json::json!(sha));
        if let Value::Object(m) = extra {
            o.extend(m);
        }
        Ok(serde_json::to_string(&Value::Object(o))?)
    };

    // PART 2 — warm-reference batch path (opt-in via ZEN_SCOREFILE_WARMREF=1, orchestrator-cuda build).
    // Decode every variant ONCE up front, score the orchestrator-eligible metrics via a single
    // `run_all` (ref uploaded once per source, not per variant), then score butteraugli +
    // zensim-with-features inline over the SAME decoded buffers. Default OFF -> the byte-identical
    // one-shot loop below.
    #[cfg(all(feature = "orchestrator", feature = "orchestrator-cuda"))]
    if std::env::var("ZEN_SCOREFILE_WARMREF").map(|v| v == "1").unwrap_or(false) {
        // Decode all variants once (tar-shard local read via fetch_variant).
        let mut decoded: Vec<(String, Rgb8Image)> = Vec::with_capacity(shas.len());
        for sha in &shas {
            match fetch_variant(sha, ext) {
                Ok((p, owned)) => {
                    let d = decode_image_to_rgb8(&p);
                    if owned {
                        let _ = std::fs::remove_file(&p);
                    }
                    match d {
                        Ok(img) => decoded.push(((*sha).to_string(), img)),
                        Err(e) => rows.push(mk_row(
                            sha,
                            serde_json::json!({ "error": format!("decode: {e}") }),
                        )?),
                    }
                }
                Err(e) => rows.push(mk_row(
                    sha,
                    serde_json::json!({ "error": format!("fetch: {e}") }),
                )?),
            }
        }
        // Eligible metrics -> one warm-ref batch.
        rows.extend(warmref_score_eligible(
            image_path, codec_name, &reference, &decoded, &metrics,
        )?);
        // Butteraugli (orchestrator-ineligible) + zensim (needs its feature sidecar) inline, reusing
        // the decoded buffers so no variant is decoded twice.
        for (sha, distorted) in &decoded {
            for metric in &metrics {
                #[cfg(feature = "gpu-zensim")]
                if *metric == "zensim-gpu" || *metric == "zensim" {
                    match crate::metrics::run_zensim_gpu_with_features(
                        &reference,
                        distorted,
                        crate::metrics::GpuRuntime::Auto,
                        crate::metrics::ZensimFeatureRegime::WithIw,
                    ) {
                        Ok((sc, feats)) => {
                            rows.push(mk_row(
                                sha,
                                serde_json::json!({ "metric": metric, "score": sc, "scores": { "zensim_score": sc } }),
                            )?);
                            let mut fo = Map::new();
                            fo.insert("kind".into(), serde_json::json!("feature"));
                            fo.insert("image_path".into(), serde_json::json!(image_path));
                            fo.insert("codec".into(), serde_json::json!(codec_name));
                            fo.insert("encode_sha".into(), serde_json::json!(sha));
                            fo.insert("regime".into(), serde_json::json!("with-iw"));
                            fo.insert("zensim_score".into(), serde_json::json!(sc));
                            fo.insert("features".into(), serde_json::json!(feats));
                            rows.push(serde_json::to_string(&Value::Object(fo))?);
                        }
                        Err(e) => rows.push(mk_row(
                            sha,
                            serde_json::json!({ "metric": metric, "error": e.to_string() }),
                        )?),
                    }
                    continue;
                }
                // butteraugli / butteraugli-gpu (and any non-eligible, non-zensim metric): one-shot.
                if *metric == "butteraugli" || *metric == "butteraugli-gpu" {
                    match score(metric, &reference, distorted) {
                        Ok(pairs) => {
                            let mut scores = Map::new();
                            for (n, v) in &pairs {
                                scores.insert((*n).to_string(), serde_json::json!(v));
                            }
                            rows.push(mk_row(
                                sha,
                                serde_json::json!({
                                    "metric": metric,
                                    "score": pairs.first().map(|(_, v)| *v),
                                    "scores": scores,
                                }),
                            )?);
                        }
                        Err(e) => rows.push(mk_row(
                            sha,
                            serde_json::json!({ "metric": metric, "error": e.to_string() }),
                        )?),
                    }
                }
            }
        }
        return Ok(rows.join("\n").into_bytes());
    }

    for sha in &shas {
        // Fetch + decode the variant ONCE; score every metric on it. `owned` is true for a byte-range
        // temp (delete after decode) and false for a TAR-SHARD local read (borrowed from the shared
        // extract dir — must NOT delete, other cells/metrics reference the same files).
        let (var_path, owned) = match fetch_variant(sha, ext) {
            Ok(p) => p,
            Err(e) => {
                rows.push(mk_row(
                    sha,
                    serde_json::json!({ "error": format!("fetch: {e}") }),
                )?);
                continue;
            }
        };
        let distorted = match decode_image_to_rgb8(&var_path) {
            Ok(d) => d,
            Err(e) => {
                if owned {
                    let _ = std::fs::remove_file(&var_path);
                }
                rows.push(mk_row(
                    sha,
                    serde_json::json!({ "error": format!("decode: {e}") }),
                )?);
                continue;
            }
        };
        if owned {
            let _ = std::fs::remove_file(&var_path);
        }
        for metric in &metrics {
            // zensim-gpu additionally yields the 372-feature vector from the SAME decode — the exact
            // score-pairs --feature-output path (run_zensim_gpu_with_features). Emit a feature row too.
            #[cfg(feature = "gpu-zensim")]
            if *metric == "zensim-gpu" || *metric == "zensim" {
                match crate::metrics::run_zensim_gpu_with_features(
                    &reference,
                    &distorted,
                    crate::metrics::GpuRuntime::Auto,
                    crate::metrics::ZensimFeatureRegime::WithIw,
                ) {
                    Ok((sc, feats)) => {
                        rows.push(mk_row(
                            sha,
                            serde_json::json!({ "metric": metric, "score": sc, "scores": { "zensim_score": sc } }),
                        )?);
                        let mut fo = Map::new();
                        fo.insert("kind".into(), serde_json::json!("feature"));
                        fo.insert("image_path".into(), serde_json::json!(image_path));
                        fo.insert("codec".into(), serde_json::json!(codec_name));
                        fo.insert("encode_sha".into(), serde_json::json!(sha));
                        fo.insert("regime".into(), serde_json::json!("with-iw"));
                        fo.insert("zensim_score".into(), serde_json::json!(sc));
                        fo.insert("features".into(), serde_json::json!(feats));
                        rows.push(serde_json::to_string(&Value::Object(fo))?);
                    }
                    Err(e) => rows.push(mk_row(
                        sha,
                        serde_json::json!({ "metric": metric, "error": e.to_string() }),
                    )?),
                }
                continue;
            }
            match score(metric, &reference, &distorted) {
                Ok(pairs) => {
                    let mut scores = Map::new();
                    for (n, v) in &pairs {
                        scores.insert((*n).to_string(), serde_json::json!(v));
                    }
                    rows.push(mk_row(
                        sha,
                        serde_json::json!({
                            "metric": metric,
                            "score": pairs.first().map(|(_, v)| *v),
                            "scores": scores,
                        }),
                    )?);
                }
                Err(e) => rows.push(mk_row(
                    sha,
                    serde_json::json!({ "metric": metric, "error": e.to_string() }),
                )?),
            }
        }
    }
    Ok(rows.join("\n").into_bytes())
}

/// Do one job end-to-end: resolve+decode the source, encode the cell, and (for a metric job) score
/// it. Returns the output BYTES — encode: the encoded image; metric: the one-line JSON score row.
/// Shared by single-shot `run` and the warm `run_serve` loop, so both paths are byte-identical.
fn run_one_job(job: &Value, corpus_prefix: Option<&str>) -> Result<Vec<u8>, Box<dyn Error>> {
    let kind = job["kind"]["kind"]
        .as_str()
        .ok_or("job.kind.kind missing")?;
    // Whole-file scoring (JobKind::ScoreFile): a different shape — many variants, no single q/knobs,
    // NO re-encode. Decode the reference once, then fetch + score each pre-encoded variant. The
    // efficient path that replaces per-(cell,metric) re-encoding. Handled separately and returns early.
    if kind == "score_file" {
        return run_score_file(job, corpus_prefix);
    }
    let cell = &job["cell"];
    let image_path = cell["image_path"]
        .as_str()
        .ok_or("cell.image_path missing")?;
    let codec_name = cell["codec"].as_str().ok_or("cell.codec missing")?;
    let q = cell["q"].as_i64().ok_or("cell.q missing")? as f64;
    let knob_json = cell["knob_tuple_json"].as_str().unwrap_or("{}");
    let knobs: Map<String, Value> =
        serde_json::from_str(knob_json).map_err(|e| format!("parse knob_tuple_json: {e}"))?;

    let src_path = resolve_source(image_path, corpus_prefix)?;
    let reference = decode_image_to_rgb8(&src_path)?;

    let codec = codec_from_name(codec_name)?;
    let encoded = encode_cell_for_job(codec, &reference, q, knob_json, &knobs)?;

    match kind {
        // The encoded bytes ARE the output → content-addressed to blobs/<sha256> (goal G).
        "encode" => Ok(encoded.bytes),
        "metric" => {
            let metric = job["kind"]["metric"]
                .as_str()
                .ok_or("metric job missing kind.metric")?;
            let seq = DIST_SEQ.fetch_add(1, Ordering::Relaxed);
            let dist_path = std::env::temp_dir().join(format!(
                "jobexec_dist_{}_{}.{}",
                std::process::id(),
                seq,
                ext_for(codec_name)
            ));
            std::fs::write(&dist_path, &encoded.bytes)?;
            let distorted = decode_image_to_rgb8(&dist_path)?;
            let _ = std::fs::remove_file(&dist_path);
            // zensim(-gpu) yields the 372-feature vector from the SAME decode for free — emit a
            // `feature` row ALONGSIDE the metric row so the 372-D vector is ALWAYS captured, exactly
            // like `run_score_file`. Without this, a `declare metric jobs` flow silently drops the
            // features (the gap that broke the jxl HQ re-do). Non-zensim metrics have no native
            // feature vector, so they keep the scalar-only path below.
            #[cfg(feature = "gpu-zensim")]
            if metric == "zensim-gpu" || metric == "zensim" {
                match crate::metrics::run_zensim_gpu_with_features(
                    &reference,
                    &distorted,
                    crate::metrics::GpuRuntime::Auto,
                    crate::metrics::ZensimFeatureRegime::WithIw,
                ) {
                    Ok((sc, feats)) => {
                        let metric_row = serde_json::json!({
                            "kind": "metric",
                            "metric": metric,
                            "image_path": image_path,
                            "codec": codec_name,
                            "q": cell["q"],
                            "knob_tuple_json": knob_json,
                            "score": sc,
                            "scores": { "zensim_score": sc },
                            "encoded_bytes": encoded.bytes.len(),
                            "encode_ms": encoded.encode_ms,
                        });
                        let feat_row = serde_json::json!({
                            "kind": "feature",
                            "image_path": image_path,
                            "codec": codec_name,
                            "q": cell["q"],
                            "knob_tuple_json": knob_json,
                            "regime": "with-iw",
                            "zensim_score": sc,
                            "features": feats,
                        });
                        return Ok(format!(
                            "{}\n{}",
                            serde_json::to_string(&metric_row)?,
                            serde_json::to_string(&feat_row)?
                        )
                        .into_bytes());
                    }
                    Err(e) => return Err(format!("zensim feature extraction: {e}").into()),
                }
            }
            let pairs = score(metric, &reference, &distorted)?;
            let mut scores = Map::new();
            for (name, value) in &pairs {
                scores.insert((*name).to_string(), serde_json::json!(value));
            }
            let row = serde_json::json!({
                "kind": "metric",
                "metric": metric,
                "image_path": image_path,
                "codec": codec_name,
                "q": cell["q"],
                "knob_tuple_json": knob_json,
                "score": pairs.first().map(|(_, v)| *v),
                "scores": scores,
                "encoded_bytes": encoded.bytes.len(),
                "encode_ms": encoded.encode_ms,
            });
            Ok(serde_json::to_string(&row)?.into_bytes())
        }
        other => Err(format!("jobexec: unhandled job kind {other:?}").into()),
    }
}

/// Read exactly `buf.len()` bytes. `Ok(false)` = clean EOF *before any byte* of the frame (the loop's
/// normal termination); EOF mid-frame is a hard error.
fn read_frame_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => {
                return if filled == 0 {
                    Ok(false)
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "truncated frame",
                    ))
                };
            }
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

/// Warm executor loop: keep the process — and cubecl's per-process cached GPU client + compiled
/// kernels — alive across many jobs, so CUDA init and kernel compilation are paid ONCE, not per job.
/// Protocol: request `[u32 LE len][DesiredJob JSON]` → response `[u8 status][u32 LE len][payload]`,
/// status 0=ok (payload=output bytes), 1=job error, 2=panic (payload=message). A per-job panic is
/// caught and returned as a frame, NOT a process exit, so one bad cell never kills the warm worker.
/// Clean EOF on the request stream ends the loop (exit 0).
fn run_serve(corpus_prefix: Option<&str>) -> Result<(), Box<dyn Error>> {
    let stdin = std::io::stdin();
    let mut r = stdin.lock();
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    let mut lenb = [0u8; 4];
    loop {
        if !read_frame_exact(&mut r, &mut lenb)? {
            break; // clean EOF — the worker closed stdin
        }
        let len = u32::from_le_bytes(lenb) as usize;
        let mut jbuf = vec![0u8; len];
        read_frame_exact(&mut r, &mut jbuf)?;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let job: Value = serde_json::from_slice(&jbuf)
                .map_err(|e| -> Box<dyn Error> { format!("parse DesiredJob: {e}").into() })?;
            run_one_job(&job, corpus_prefix)
        }));
        let (status, payload): (u8, Vec<u8>) = match outcome {
            Ok(Ok(bytes)) => (0, bytes),
            Ok(Err(e)) => (1, e.to_string().into_bytes()),
            Err(p) => {
                let msg = p
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| p.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic".to_string());
                (2, format!("jobexec panic: {msg}").into_bytes())
            }
        };
        w.write_all(&[status])?;
        w.write_all(&(payload.len() as u32).to_le_bytes())?;
        w.write_all(&payload)?;
        w.flush()?;
    }
    Ok(())
}

pub fn run(args: JobexecArgs) -> Result<(), Box<dyn Error>> {
    let corpus_prefix = args
        .corpus_prefix
        .or_else(|| std::env::var("ZEN_CORPUS_PREFIX").ok());
    if args.serve {
        return run_serve(corpus_prefix.as_deref());
    }
    // Single-shot: one DesiredJob on stdin → output bytes on stdout (the original ZEN_EXEC contract).
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let job: Value = serde_json::from_str(&buf).map_err(|e| format!("parse DesiredJob: {e}"))?;
    let bytes = run_one_job(&job, corpus_prefix.as_deref())?;
    let mut out = std::io::stdout().lock();
    out.write_all(&bytes)?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_variant_index_4col_carries_name() {
        // 4-column rows (sha, offset, size, name) power the TAR-SHARD local read.
        let tsv = "aaaa\t0\t100\to_1.png.scale64x64_h_zenpng_q0_x.png\n\
                   bbbb\t100\t250\to_2.png.scale128x96_h_zenpng_q0_y.png\n";
        let m = parse_variant_index(tsv);
        assert_eq!(m.len(), 2);
        let a = m.get("aaaa").expect("aaaa present");
        assert_eq!((a.offset, a.size), (0, 100));
        assert_eq!(a.name, "o_1.png.scale64x64_h_zenpng_q0_x.png");
        let b = m.get("bbbb").expect("bbbb present");
        assert_eq!((b.offset, b.size), (100, 250));
        assert_eq!(b.name, "o_2.png.scale128x96_h_zenpng_q0_y.png");
    }

    #[test]
    fn parse_variant_index_3col_legacy_empty_name() {
        // Legacy 3-column indices (no name) must still parse — name defaults empty, so the executor
        // falls back to the byte-range GET path (no TAR-SHARD local read possible without a name).
        let tsv = "cccc\t42\t7\n";
        let m = parse_variant_index(tsv);
        let c = m.get("cccc").expect("cccc present");
        assert_eq!((c.offset, c.size), (42, 7));
        assert_eq!(c.name, "");
    }

    #[test]
    fn parse_variant_index_skips_malformed() {
        let tsv = "good\t1\t2\tn.png\nbadline_no_tabs\nalso\tbad\nok\t3\t4\n";
        let m = parse_variant_index(tsv);
        assert!(m.contains_key("good"));
        assert!(m.contains_key("ok"));
        assert!(!m.contains_key("badline_no_tabs"));
        assert!(!m.contains_key("also")); // "also\tbad" — size col unparseable
        assert_eq!(m.len(), 2);
    }
}
