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

/// Variant index: `sha -> (byte offset, size)` into `variants.tar`. Loaded once per process (the warm
/// `--serve` loop reuses it). TSV `sha\toffset\tsize` at `$ZEN_VARIANT_INDEX_URI`, built by the
/// manifest builder from the tar's member headers — so the executor fetches a pre-encoded variant by
/// a single byte-range GET, never re-encoding and never downloading the whole 4 GB tar.
static VARIANT_INDEX: std::sync::OnceLock<std::collections::HashMap<String, (u64, u64)>> =
    std::sync::OnceLock::new();

fn variant_index() -> Result<&'static std::collections::HashMap<String, (u64, u64)>, Box<dyn Error>>
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
    let mut m = std::collections::HashMap::new();
    for line in std::fs::read_to_string(&dst)?.lines() {
        let mut it = line.split('\t');
        if let (Some(s), Some(o), Some(z)) = (it.next(), it.next(), it.next())
            && let (Ok(o), Ok(z)) = (o.parse::<u64>(), z.parse::<u64>())
        {
            m.insert(s.to_string(), (o, z));
        }
    }
    Ok(VARIANT_INDEX.get_or_init(|| m))
}

/// Byte-range-fetch one pre-encoded variant out of `variants.tar` (`$ZEN_VARIANTS_TAR_URI`) using the
/// index — no per-variant R2 object, no whole-tar download, no re-encode. Returns a temp path.
fn fetch_variant(sha: &str, ext: &str) -> Result<PathBuf, Box<dyn Error>> {
    let &(off, sz) = variant_index()?
        .get(sha)
        .ok_or_else(|| format!("sha {sha} not in variant index"))?;
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
    Ok(dst)
}

/// Whole-file scoring (`JobKind::ScoreFile`): the efficient path. Decode the reference ONCE, then for
/// each input variant sha byte-range-fetch the pre-encoded bytes out of `variants.tar`, decode it
/// ONCE, and score EVERY metric against the shared reference via the same `run_metric` core
/// `score-pairs` uses — so a 24 MP source's decode and each variant's decode happen once, never
/// re-encoded or re-decoded per metric. Emits one JSONL row per (variant, metric); the write-back
/// rejoins q/knobs by `encode_sha`.
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
    for sha in &shas {
        // Fetch + decode the variant ONCE; score every metric on it.
        let var_path = match fetch_variant(sha, ext) {
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
                let _ = std::fs::remove_file(&var_path);
                rows.push(mk_row(
                    sha,
                    serde_json::json!({ "error": format!("decode: {e}") }),
                )?);
                continue;
            }
        };
        let _ = std::fs::remove_file(&var_path);
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
            let pairs = score(metric, &reference, &distorted)?;
            let _ = std::fs::remove_file(&dist_path);
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
