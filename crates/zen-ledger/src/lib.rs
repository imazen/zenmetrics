#![forbid(unsafe_code)]
//! # zen-ledger
//!
//! Columnar Parquet persistence for the job ledger and blob index — the **Foundations** bullet
//! "ledger = columnar Parquet (latest-wins on (key,ts)), never millions of JSON objects".
//!
//! - [`write_ledger`] / [`read_ledger`]: per-chunk `LedgerRow` sidecars (zstd, matching the existing
//!   sweep writer's `ZstdLevel(3)`). A FAILED row is a column-row, never a gap (goal B).
//! - [`write_blob_index`] / [`read_blob_index`]: the inventory the GC scans instead of LISTing R2.
//! - [`compact_ledger`]: fold many per-chunk sidecars into one consolidated file via the
//!   latest-wins [`LedgerView`] — the small-files-then-compact pattern, so the ledger never becomes
//!   a tiny-file swamp.
//!
//! Built on the same `arrow`/`parquet` 58.x stack the sweep already vendors. No DuckDB, no I/O to R2
//! here — callers move the bytes; this is the columnar shape + round-trip.

use std::fs::File;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{Array, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use zen_job_core::{
    BlobIndexEntry, CellId, ErrorClass, JobId, JobKind, JobStatus, LedgerRow, LedgerView,
    Regenerability, Sha256Hex,
};

#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    #[error("io {0}")]
    Io(String),
    #[error("parquet {0}")]
    Parquet(String),
    #[error("arrow {0}")]
    Arrow(String),
    #[error("decode column {col:?}: {msg}")]
    Decode { col: &'static str, msg: String },
}

// ---- small serde-enum <-> string helpers (avoid hand-maintaining a second mapping) ----

fn enum_str<T: serde::Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(serde_json::Value::String(s)) => s,
        other => unreachable!("snake_case unit enum must serialize to a JSON string, got {other:?}"),
    }
}

fn enum_parse<T: serde::de::DeserializeOwned>(s: &str, col: &'static str) -> Result<T, LedgerError> {
    serde_json::from_value(serde_json::Value::String(s.to_string()))
        .map_err(|e| LedgerError::Decode { col, msg: format!("bad value {s:?}: {e}") })
}

fn props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).expect("zstd level 3 is valid")))
        .build()
}

fn write_batch(path: &Path, schema: SchemaRef, batch: RecordBatch) -> Result<(), LedgerError> {
    let file = File::create(path).map_err(|e| LedgerError::Io(format!("create {}: {e}", path.display())))?;
    let mut w = ArrowWriter::try_new(file, schema, Some(props()))
        .map_err(|e| LedgerError::Parquet(e.to_string()))?;
    w.write(&batch).map_err(|e| LedgerError::Parquet(e.to_string()))?;
    w.close().map_err(|e| LedgerError::Parquet(e.to_string()))?;
    Ok(())
}

// ---- ledger ----

fn ledger_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("job_id", DataType::Utf8, false),
        Field::new("image_path", DataType::Utf8, false),
        Field::new("codec", DataType::Utf8, false),
        Field::new("q", DataType::Int64, false),
        Field::new("knob_tuple_json", DataType::Utf8, false),
        Field::new("output_sha", DataType::Utf8, true),
        Field::new("status", DataType::Utf8, false),
        Field::new("error_class", DataType::Utf8, true),
        Field::new("attempts", DataType::UInt32, false),
        Field::new("ts", DataType::UInt64, false),
        Field::new("worker", DataType::Utf8, false),
        Field::new("provider", DataType::Utf8, false),
        Field::new("kind_json", DataType::Utf8, false),
    ]))
}

/// Write ledger rows to a zstd parquet sidecar.
pub fn write_ledger(path: &Path, rows: &[LedgerRow]) -> Result<(), LedgerError> {
    let schema = ledger_schema();
    let job_id = StringArray::from_iter_values(rows.iter().map(|r| r.job_id.as_str()));
    let image_path = StringArray::from_iter_values(rows.iter().map(|r| r.cell.image_path.as_str()));
    let codec = StringArray::from_iter_values(rows.iter().map(|r| r.cell.codec.as_str()));
    let q = Int64Array::from_iter_values(rows.iter().map(|r| r.cell.q));
    let knobs = StringArray::from_iter_values(rows.iter().map(|r| r.cell.knob_tuple_json.as_str()));
    // nullable → collect via FromIterator<Option<Ptr>>
    let output_sha: StringArray =
        rows.iter().map(|r| r.output_sha.as_ref().map(Sha256Hex::as_str)).collect();
    let status = StringArray::from_iter_values(rows.iter().map(|r| enum_str(&r.status)));
    let error_class: StringArray =
        rows.iter().map(|r| r.error_class.map(|e| enum_str(&e))).collect();
    let attempts = UInt32Array::from_iter_values(rows.iter().map(|r| r.attempts));
    let ts = UInt64Array::from_iter_values(rows.iter().map(|r| r.ts));
    let worker = StringArray::from_iter_values(rows.iter().map(|r| r.worker.as_str()));
    let provider = StringArray::from_iter_values(rows.iter().map(|r| r.provider.as_str()));
    let kind_json = StringArray::from_iter_values(
        rows.iter().map(|r| serde_json::to_string(&r.kind).expect("JobKind is serializable")),
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(job_id),
            Arc::new(image_path),
            Arc::new(codec),
            Arc::new(q),
            Arc::new(knobs),
            Arc::new(output_sha),
            Arc::new(status),
            Arc::new(error_class),
            Arc::new(attempts),
            Arc::new(ts),
            Arc::new(worker),
            Arc::new(provider),
            Arc::new(kind_json),
        ],
    )
    .map_err(|e| LedgerError::Arrow(e.to_string()))?;
    write_batch(path, schema, batch)
}

fn col_str<'a>(b: &'a RecordBatch, idx: usize, name: &'static str) -> Result<&'a StringArray, LedgerError> {
    b.column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or(LedgerError::Decode { col: name, msg: "not Utf8".into() })
}

/// Read ledger rows back from a parquet sidecar.
pub fn read_ledger(path: &Path) -> Result<Vec<LedgerRow>, LedgerError> {
    let file = File::open(path).map_err(|e| LedgerError::Io(format!("open {}: {e}", path.display())))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LedgerError::Parquet(e.to_string()))?
        .build()
        .map_err(|e| LedgerError::Parquet(e.to_string()))?;

    let mut out = Vec::new();
    for batch in reader {
        let b = batch.map_err(|e| LedgerError::Arrow(e.to_string()))?;
        let job_id = col_str(&b, 0, "job_id")?;
        let image_path = col_str(&b, 1, "image_path")?;
        let codec = col_str(&b, 2, "codec")?;
        let q = b.column(3).as_any().downcast_ref::<Int64Array>()
            .ok_or(LedgerError::Decode { col: "q", msg: "not Int64".into() })?;
        let knobs = col_str(&b, 4, "knob_tuple_json")?;
        let output_sha = col_str(&b, 5, "output_sha")?;
        let status = col_str(&b, 6, "status")?;
        let error_class = col_str(&b, 7, "error_class")?;
        let attempts = b.column(8).as_any().downcast_ref::<UInt32Array>()
            .ok_or(LedgerError::Decode { col: "attempts", msg: "not UInt32".into() })?;
        let ts = b.column(9).as_any().downcast_ref::<UInt64Array>()
            .ok_or(LedgerError::Decode { col: "ts", msg: "not UInt64".into() })?;
        let worker = col_str(&b, 10, "worker")?;
        let provider = col_str(&b, 11, "provider")?;
        let kind_json = col_str(&b, 12, "kind_json")?;

        for i in 0..b.num_rows() {
            out.push(LedgerRow {
                job_id: JobId(Sha256Hex::parse(job_id.value(i)).map_err(|e| LedgerError::Decode {
                    col: "job_id",
                    msg: e.to_string(),
                })?),
                kind: serde_json::from_str::<JobKind>(kind_json.value(i)).map_err(|e| {
                    LedgerError::Decode { col: "kind_json", msg: e.to_string() }
                })?,
                cell: CellId {
                    image_path: image_path.value(i).to_string(),
                    codec: codec.value(i).to_string(),
                    q: q.value(i),
                    knob_tuple_json: knobs.value(i).to_string(),
                },
                output_sha: if output_sha.is_null(i) {
                    None
                } else {
                    Some(Sha256Hex::parse(output_sha.value(i)).map_err(|e| LedgerError::Decode {
                        col: "output_sha",
                        msg: e.to_string(),
                    })?)
                },
                status: enum_parse::<JobStatus>(status.value(i), "status")?,
                error_class: if error_class.is_null(i) {
                    None
                } else {
                    Some(enum_parse::<ErrorClass>(error_class.value(i), "error_class")?)
                },
                attempts: attempts.value(i),
                ts: ts.value(i),
                worker: worker.value(i).to_string(),
                provider: provider.value(i).to_string(),
            });
        }
    }
    Ok(out)
}

/// Fold many per-chunk ledger sidecars into one consolidated file via latest-wins on `(job_id, ts)`.
/// Returns the consolidated row count. This is the compaction step that keeps the ledger from
/// becoming a tiny-file swamp.
pub fn compact_ledger(inputs: &[&Path], out: &Path) -> Result<usize, LedgerError> {
    let mut view = LedgerView::new();
    for p in inputs {
        for row in read_ledger(p)? {
            view.apply(row);
        }
    }
    let rows: Vec<LedgerRow> = view.rows().cloned().collect();
    write_ledger(out, &rows)?;
    Ok(rows.len())
}

// ---- s3:// (R2) ledger I/O: same Parquet, staged through a temp file via the s5cmd CLI ----

static URI_TMP_N: AtomicU64 = AtomicU64::new(0);

fn s5cmd_cp(endpoint: &str, src: &str, dst: &str) -> Result<(), LedgerError> {
    let st = Command::new("s5cmd")
        .arg("--endpoint-url")
        .arg(endpoint)
        .arg("cp")
        .arg(src)
        .arg(dst)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .map_err(|e| LedgerError::Io(format!("s5cmd spawn: {e}")))?;
    if st.success() {
        Ok(())
    } else {
        Err(LedgerError::Io(format!("s5cmd cp {src} -> {dst} exited {:?}", st.code())))
    }
}

/// Read a ledger from a local path **or** an `s3://` URI. For `s3://`, downloads via s5cmd (needs the
/// R2 `endpoint` and `AWS_*` creds in the environment), then reads the Parquet.
pub fn read_ledger_uri(uri: &str, endpoint: Option<&str>) -> Result<Vec<LedgerRow>, LedgerError> {
    if uri.starts_with("s3://") {
        let ep = endpoint
            .ok_or_else(|| LedgerError::Io("s3:// ledger requires an R2 endpoint".into()))?;
        let n = URI_TMP_N.fetch_add(1, Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!("zenledger_dl_{}_{}.parquet", std::process::id(), n));
        s5cmd_cp(ep, uri, &tmp.to_string_lossy())?;
        let rows = read_ledger(&tmp);
        let _ = std::fs::remove_file(&tmp);
        rows
    } else {
        read_ledger(Path::new(uri))
    }
}

/// Write a ledger to a local path **or** an `s3://` URI (uploads via s5cmd for `s3://`).
pub fn write_ledger_uri(uri: &str, rows: &[LedgerRow], endpoint: Option<&str>) -> Result<(), LedgerError> {
    if uri.starts_with("s3://") {
        let ep = endpoint
            .ok_or_else(|| LedgerError::Io("s3:// ledger requires an R2 endpoint".into()))?;
        let n = URI_TMP_N.fetch_add(1, Ordering::Relaxed);
        let tmp = std::env::temp_dir().join(format!("zenledger_ul_{}_{}.parquet", std::process::id(), n));
        write_ledger(&tmp, rows)?;
        let r = s5cmd_cp(ep, &tmp.to_string_lossy(), uri);
        let _ = std::fs::remove_file(&tmp);
        r
    } else {
        write_ledger(Path::new(uri), rows)
    }
}

// ---- blob index ----

fn blob_index_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("sha", DataType::Utf8, false),
        Field::new("size", DataType::UInt64, false),
        Field::new("regenerability", DataType::Utf8, false),
        Field::new("last_ref_secs", DataType::UInt64, false),
    ]))
}

pub fn write_blob_index(path: &Path, entries: &[BlobIndexEntry]) -> Result<(), LedgerError> {
    let schema = blob_index_schema();
    let sha = StringArray::from_iter_values(entries.iter().map(|e| e.sha.as_str()));
    let size = UInt64Array::from_iter_values(entries.iter().map(|e| e.size));
    let regen = StringArray::from_iter_values(entries.iter().map(|e| enum_str(&e.regenerability)));
    let last_ref = UInt64Array::from_iter_values(entries.iter().map(|e| e.last_ref_secs));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(sha), Arc::new(size), Arc::new(regen), Arc::new(last_ref)],
    )
    .map_err(|e| LedgerError::Arrow(e.to_string()))?;
    write_batch(path, schema, batch)
}

pub fn read_blob_index(path: &Path) -> Result<Vec<BlobIndexEntry>, LedgerError> {
    let file = File::open(path).map_err(|e| LedgerError::Io(format!("open {}: {e}", path.display())))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LedgerError::Parquet(e.to_string()))?
        .build()
        .map_err(|e| LedgerError::Parquet(e.to_string()))?;
    let mut out = Vec::new();
    for batch in reader {
        let b = batch.map_err(|e| LedgerError::Arrow(e.to_string()))?;
        let sha = col_str(&b, 0, "sha")?;
        let size = b.column(1).as_any().downcast_ref::<UInt64Array>()
            .ok_or(LedgerError::Decode { col: "size", msg: "not UInt64".into() })?;
        let regen = col_str(&b, 2, "regenerability")?;
        let last_ref = b.column(3).as_any().downcast_ref::<UInt64Array>()
            .ok_or(LedgerError::Decode { col: "last_ref_secs", msg: "not UInt64".into() })?;
        for i in 0..b.num_rows() {
            out.push(BlobIndexEntry {
                sha: Sha256Hex::parse(sha.value(i)).map_err(|e| LedgerError::Decode {
                    col: "sha",
                    msg: e.to_string(),
                })?,
                size: size.value(i),
                regenerability: enum_parse::<Regenerability>(regen.value(i), "regenerability")?,
                last_ref_secs: last_ref.value(i),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use zen_job_core::sha256;

    static N: AtomicU64 = AtomicU64::new(0);

    fn tmp(tag: &str) -> std::path::PathBuf {
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("zenledger_{}_{}_{tag}.parquet", std::process::id(), n))
    }

    fn row(metric: &str, enc: &[u8], status: JobStatus, ts: u64, err: Option<ErrorClass>) -> LedgerRow {
        let kind = zen_job_core::JobKind::Metric { metric: metric.into() };
        let input = sha256(enc);
        LedgerRow {
            job_id: JobId::of(&kind, std::slice::from_ref(&input)),
            kind: kind.clone(),
            cell: CellId {
                image_path: "img/x.png".into(),
                codec: "zenjpeg".into(),
                q: 80,
                knob_tuple_json: "{}".into(),
            },
            output_sha: if status == JobStatus::Done { Some(sha256(b"score")) } else { None },
            status,
            error_class: err,
            attempts: 1,
            ts,
            worker: "w1".into(),
            provider: "oracle".into(),
        }
    }

    #[test]
    fn ledger_round_trips() {
        let p = tmp("ledger");
        let rows = vec![
            row("cvvdp", b"a", JobStatus::Done, 100, None),
            row("ssim2", b"a", JobStatus::Failed, 100, Some(ErrorClass::Timeout)),
        ];
        write_ledger(&p, &rows).unwrap();
        let back = read_ledger(&p).unwrap();
        assert_eq!(back.len(), 2);
        // order is preserved within a single write
        assert_eq!(back, rows);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn blob_index_round_trips() {
        let p = tmp("blobidx");
        let entries = vec![
            BlobIndexEntry { sha: sha256(b"jpeg"), size: 123, regenerability: Regenerability::CheapRegenerable, last_ref_secs: 10 },
            BlobIndexEntry { sha: sha256(b"avif"), size: 999_999, regenerability: Regenerability::ExpensiveRegenerable, last_ref_secs: 20 },
            BlobIndexEntry { sha: sha256(b"src"), size: 50_000_000, regenerability: Regenerability::NotRegenerable, last_ref_secs: 30 },
        ];
        write_blob_index(&p, &entries).unwrap();
        let back = read_blob_index(&p).unwrap();
        assert_eq!(back, entries);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn compaction_is_latest_wins() {
        // Two sidecars for the same job_id: a Failed@100 then a Done@200. Compaction keeps Done.
        let p1 = tmp("c1");
        let p2 = tmp("c2");
        let out = tmp("compacted");
        write_ledger(&p1, &[row("cvvdp", b"a", JobStatus::Failed, 100, Some(ErrorClass::Oom))]).unwrap();
        write_ledger(&p2, &[row("cvvdp", b"a", JobStatus::Done, 200, None)]).unwrap();
        let n = compact_ledger(&[&p1, &p2], &out).unwrap();
        assert_eq!(n, 1, "same job_id collapses to one row");
        let back = read_ledger(&out).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].status, JobStatus::Done, "latest ts wins");
        for f in [&p1, &p2, &out] {
            std::fs::remove_file(f).ok();
        }
    }
}
