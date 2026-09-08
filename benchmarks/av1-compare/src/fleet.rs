//! Adapters to zenfleet's canonical declaration and executor contracts. No
//! scheduling, claiming, retry, or ledger implementation is duplicated here.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    path::{Component, Path},
};
use zenfleet_core::{DesiredJob, JobKind};
type Error = Box<dyn std::error::Error>;
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn binary_sha() -> Result<String, Error> {
    Ok(sha(&fs::read(std::env::current_exe()?)?))
}
/// Input: the same EncodeDeclareItem JSONL used by zenfleet-ctl. The job is
/// one source/size comparison bundle, retaining every arm and timing repeat.
pub fn declare(text: &str) -> Result<(), Error> {
    let mut items = zenfleet_ctl::parse_emit_cells(text)?;
    let build = binary_sha()?;
    for item in &mut items {
        if item.codec != "av1-compare" || item.hdr || item.q != 0 || item.encode_fp.is_some() {
            return Err(
                "comparison declaration requires codec=av1-compare, q=0, SDR, no encode dedup"
                    .into(),
            );
        }
        let mut knobs: Value = serde_json::from_str(&item.knob_tuple_json)?;
        let object = knobs.as_object_mut().ok_or("knobs must be an object")?;
        object.insert("protocol".into(), Value::String("av1-api-planar-v2".into()));
        object.insert("binary_sha256".into(), Value::String(build.clone()));
        item.knob_tuple_json = serde_json::to_string(&knobs)?;
    }
    let mut jobs = zenfleet_ctl::declare_encodes(&items)?;
    for job in &mut jobs {
        job.requires.push("av1-api-planar-v2".into());
    }
    serde_json::to_writer(io::stdout().lock(), &jobs)?;
    Ok(())
}
/// A successful stdout artifact is a tar containing rows.jsonl, reference PNGs
/// and every content-addressed OBU. The existing worker persists it and its
/// hash in the normal blob store and Parquet ledger.
pub fn execute(text: &str) -> Result<(), Error> {
    let job: DesiredJob = serde_json::from_str(text)?;
    let JobKind::Encode {
        codec,
        q,
        knobs,
        hdr,
    } = job.kind
    else {
        return Err("not a comparison encode job".into());
    };
    if codec != "av1-compare" || q != 0 || hdr || job.inputs.len() != 1 {
        return Err("invalid comparison job".into());
    }
    let mut settings: Value = serde_json::from_str(&knobs)?;
    if settings["protocol"] != "av1-api-planar-v2" || settings["binary_sha256"] != binary_sha()? {
        return Err("executor protocol/build mismatch".into());
    }
    let path = Path::new(&job.cell.image_path);
    if path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err("source must be a corpus basename".into());
    }
    let source = Path::new(&std::env::var("ZEN_CORPUS_DIR")?).join(path);
    if sha(&fs::read(&source)?) != job.inputs[0].as_str() {
        return Err("source hash mismatch".into());
    }
    let root = std::env::var("TMPDIR")?;
    fs::create_dir_all(&root)?;
    let scratch = tempfile::Builder::new()
        .prefix("av1-compare-")
        .tempdir_in(root)?;
    let result = scratch.path().join("result");
    let object = settings
        .as_object_mut()
        .ok_or("settings must be an object")?;
    object.remove("protocol");
    object.remove("binary_sha256");
    object.insert("inputs".into(), serde_json::json!([source]));
    object.insert("output_dir".into(), serde_json::json!(result));
    crate::measure::run(serde_json::from_value(settings)?, false)?;
    let mut archive = tar::Builder::new(io::stdout().lock());
    archive.append_dir_all("comparison", result)?;
    archive.finish()?;
    Ok(())
}
