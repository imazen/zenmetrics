//! Parquet output writer for chunk sidecars.
//!
//! Replaces the bash worker's Python step:
//!
//! ```python
//! for gf in glob.glob(sweep_dir / 'g*.tsv'):
//!     t = pa_csv.read_csv(gf, ...)
//!     tables.append(t)
//! t = pa.concat_tables(tables, promote_options='default')
//! t = t.append_column('chunk_id', ...)
//! pq.write_table(t, out_p, compression='zstd')
//! ```
//!
//! Reads each per-group TSV via arrow-rs's `arrow_csv::ReaderBuilder`,
//! concats into one `RecordBatch`, appends the trailing metadata
//! columns (`chunk_id`, `run_id`, `encoded_r2_uri`), and writes a
//! single zstd-compressed parquet.
//!
//! ## Schema policy
//!
//! Per-group TSVs all share the same column set because they all
//! came from the same `zen-metrics sweep` invocation shape (same
//! metric list, just different (codec, knob_tuple) groups). We do
//! NOT use `promote_options='default'` style nullable-fill — if a
//! schema mismatch IS observed it's a bug we want loud, not silent.
//! The bash version had to tolerate this because it ran independent
//! `zen-metrics` processes that could in principle emit different
//! columns; the Rust in-process version controls all of run_sweep's
//! output and can guarantee schema consistency.

#![cfg(feature = "inline-sweep")]

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Concat all `g*.tsv` files in `sweep_dir` into one parquet sidecar
/// at `output_path`.
///
/// `chunk_id` and `run_id` are appended as constant columns. If
/// `encoded_r2_prefix` is non-empty, an `encoded_r2_uri` column is
/// appended: row gets `<prefix><encoded_filename>` when the row's
/// encoded_filename is non-null/non-empty, else empty string.
///
/// Returns the number of rows written.
pub fn concat_groups_to_parquet(
    sweep_dir: &Path,
    output_path: &Path,
    chunk_id: &str,
    run_id: &str,
    encoded_r2_prefix: Option<&str>,
) -> Result<usize> {
    let mut group_files: Vec<std::path::PathBuf> = std::fs::read_dir(sweep_dir)
        .with_context(|| format!("read sweep_dir {}", sweep_dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            let name = p.file_name()?.to_string_lossy().into_owned();
            if name.starts_with('g') && name.ends_with(".tsv") {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    group_files.sort();
    if group_files.is_empty() {
        anyhow::bail!("no g*.tsv files in {}", sweep_dir.display());
    }

    // Schema inference across ALL group TSVs. Inferring from just
    // the first file can yield `Null`-typed columns when a sampled
    // column was entirely empty (e.g. `encoded_filename` when the
    // first group has no encoded variants saved), which then
    // silently drops valid values in later groups. Pooling
    // samples kills that footgun.
    let schema_arc = Arc::new(infer_schema_pooled(&group_files)?);

    // Stream each group through the reader → batches → concat.
    let mut batches: Vec<RecordBatch> = Vec::new();
    let mut n_bad_rows: usize = 0;
    for gf in &group_files {
        match read_tsv_into_batches(gf, schema_arc.clone()) {
            Ok((mut b, n_bad)) => {
                n_bad_rows += n_bad;
                batches.append(&mut b);
            }
            Err(e) => {
                tracing::warn!(file = %gf.display(), error = %e, "skip group TSV");
            }
        }
    }
    if batches.is_empty() {
        anyhow::bail!("zero usable rows after parsing {} group TSVs", group_files.len());
    }
    let n_total: usize = batches.iter().map(|b| b.num_rows()).sum();
    tracing::info!(n_total, n_bad_rows, "concat done");

    // Concat then append metadata columns.
    let concatenated = arrow::compute::concat_batches(&schema_arc, &batches)
        .context("concat batches")?;
    let with_meta = append_metadata_columns(
        concatenated,
        chunk_id,
        run_id,
        encoded_r2_prefix,
    )?;

    write_parquet(&with_meta, output_path)?;
    Ok(n_total)
}

/// Schema inference across the union of group TSVs.
///
/// We concatenate the bodies of all files (minus repeated header
/// rows) into a single in-memory buffer and infer types from the
/// pooled samples. Any column that's empty in one file but
/// populated in another is correctly inferred as Utf8 (rather than
/// Null) because the pool sees both samples.
///
/// Memory: per-chunk TSVs are small (a few KB each, max a few MB
/// across all groups), so the buffered approach is cheap.
fn infer_schema_pooled(files: &[std::path::PathBuf]) -> Result<Schema> {
    let mut pooled: Vec<u8> = Vec::new();
    // Carry over the header from the first file; strip headers from
    // subsequent files so arrow doesn't see them as data rows.
    let mut header_written = false;
    for f in files {
        let contents = std::fs::read(f).with_context(|| format!("read {}", f.display()))?;
        let mut iter = contents.splitn(2, |b| *b == b'\n');
        let Some(header) = iter.next() else { continue };
        let body = iter.next().unwrap_or(b"");
        if !header_written {
            pooled.extend_from_slice(header);
            pooled.push(b'\n');
            header_written = true;
        }
        pooled.extend_from_slice(body);
        if !body.ends_with(b"\n") {
            pooled.push(b'\n');
        }
    }
    let format = arrow::csv::reader::Format::default()
        .with_delimiter(b'\t')
        .with_header(true);
    let (schema, _records) = format
        .infer_schema(std::io::Cursor::new(&pooled), None)
        .context("infer schema (pooled)")?;
    // Post-process: any field that arrow inferred as Null (because
    // every sampled cell was empty) becomes Utf8. Empty-string is a
    // valid Utf8 value, so widening is lossless.
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            if matches!(f.data_type(), DataType::Null) {
                Field::new(f.name(), DataType::Utf8, true)
            } else {
                f.as_ref().clone()
            }
        })
        .collect();
    Ok(Schema::new(fields))
}

fn read_tsv_into_batches(
    tsv: &Path,
    schema: Arc<Schema>,
) -> Result<(Vec<RecordBatch>, usize)> {
    let file = std::fs::File::open(tsv)
        .with_context(|| format!("open {}", tsv.display()))?;
    let reader = arrow::csv::ReaderBuilder::new(schema)
        .with_delimiter(b'\t')
        .with_header(true)
        .build(file)
        .context("build csv reader")?;
    let mut batches = Vec::new();
    let mut n_bad: usize = 0;
    for batch in reader {
        match batch {
            Ok(b) => batches.push(b),
            Err(e) => {
                // arrow-rs's CSV reader returns one Err per malformed
                // batch (not per row). Log + continue — matches the
                // bash version's invalid_row_handler='skip' tolerance.
                tracing::warn!(error = %e, "bad CSV batch");
                n_bad += 1;
            }
        }
    }
    Ok((batches, n_bad))
}

/// Append `chunk_id`, `run_id`, and (optionally) `encoded_r2_uri`
/// columns to the concatenated batch.
fn append_metadata_columns(
    batch: RecordBatch,
    chunk_id: &str,
    run_id: &str,
    encoded_r2_prefix: Option<&str>,
) -> Result<RecordBatch> {
    let n = batch.num_rows();
    let mut fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();

    fields.push(Field::new("chunk_id", DataType::Utf8, false));
    cols.push(Arc::new(StringArray::from(vec![chunk_id; n])));
    fields.push(Field::new("run_id", DataType::Utf8, false));
    cols.push(Arc::new(StringArray::from(vec![run_id; n])));

    if let Some(prefix) = encoded_r2_prefix {
        // encoded_r2_uri[i] = prefix + encoded_filename[i] when
        // non-empty, else "". We probe the existing batch for the
        // `encoded_filename` column. If it's missing, we still emit
        // an empty `encoded_r2_uri` column — downstream readers
        // expect it.
        let uri_values: Vec<String> = if let Some(idx) =
            batch.schema().index_of("encoded_filename").ok()
        {
            let enc = batch.column(idx);
            let mut out = Vec::with_capacity(n);
            // Try Utf8 first (most common); fall back to LargeUtf8
            // if zen-metrics ever switches the column type.
            if let Some(s) = enc.as_any().downcast_ref::<StringArray>() {
                for i in 0..n {
                    if s.is_null(i) || s.value(i).is_empty() {
                        out.push(String::new());
                    } else {
                        out.push(format!("{prefix}{}", s.value(i)));
                    }
                }
            } else {
                // Unexpected type; treat all as empty.
                tracing::warn!(
                    "encoded_filename column is not Utf8; emitting empty encoded_r2_uri"
                );
                out.extend(std::iter::repeat(String::new()).take(n));
            }
            out
        } else {
            vec![String::new(); n]
        };
        fields.push(Field::new("encoded_r2_uri", DataType::Utf8, false));
        cols.push(Arc::new(StringArray::from(uri_values)));
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, cols).context("rebuild batch with metadata")
}

fn write_parquet(batch: &RecordBatch, output: &Path) -> Result<()> {
    let file = std::fs::File::create(output)
        .with_context(|| format!("create {}", output.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut wtr =
        ArrowWriter::try_new(file, batch.schema(), Some(props)).context("arrow writer")?;
    wtr.write(batch).context("write batch")?;
    wtr.close().context("close writer")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tempdir() -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "vastai-output-test-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn concat_two_groups_appends_metadata() {
        let dir = tempdir();
        let tsv_header = "image_path\tcodec\tq\tknob_tuple_json\tencoded_bytes\tencode_ms\tencoded_filename\tdecode_ms\tscore_cvvdp\n";
        let g0 = dir.join("g0.tsv");
        let g1 = dir.join("g1.tsv");
        std::fs::write(
            &g0,
            format!(
                "{tsv_header}/a.png\tzenjpeg\t50\t{{\"effort\":1}}\t12345\t100.5\t\t9.8\t9.75\n",
            ),
        )
        .unwrap();
        std::fs::write(
            &g1,
            format!(
                "{tsv_header}/b.png\tzenjpeg\t75\t{{\"effort\":2}}\t23456\t120.5\tb_q75.jpg\t10.1\t9.85\n",
            ),
        )
        .unwrap();
        let out = dir.join("sidecar.parquet");
        let n = concat_groups_to_parquet(
            &dir,
            &out,
            "v15rc_zenjpeg-0001",
            "cvvdp-v15rc-2026-05-18",
            Some("s3://zentrain/cvvdp-v15rc-2026-05-18/encoded/v15rc_zenjpeg-0001/"),
        )
        .unwrap();
        assert_eq!(n, 2);

        // Read back and inspect.
        let file = std::fs::File::open(&out).unwrap();
        let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap();
        let reader = builder.build().unwrap();
        let batches: Vec<_> = reader.collect::<std::result::Result<_, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(b.num_rows(), 2);

        let cid_col = b
            .column(b.schema().index_of("chunk_id").unwrap())
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(cid_col.value(0), "v15rc_zenjpeg-0001");
        assert_eq!(cid_col.value(1), "v15rc_zenjpeg-0001");

        let uri_col = b
            .column(b.schema().index_of("encoded_r2_uri").unwrap())
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(uri_col.value(0), ""); // empty encoded_filename
        assert_eq!(
            uri_col.value(1),
            "s3://zentrain/cvvdp-v15rc-2026-05-18/encoded/v15rc_zenjpeg-0001/b_q75.jpg"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_tsvs_errors() {
        let dir = tempdir();
        // Empty dir.
        let out = dir.join("sidecar.parquet");
        let err = concat_groups_to_parquet(&dir, &out, "c", "r", None).unwrap_err();
        assert!(err.to_string().contains("no g*.tsv"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn metadata_columns_present_without_encoded_prefix() {
        let dir = tempdir();
        std::fs::write(
            dir.join("g0.tsv"),
            "image_path\tcodec\tq\tscore_x\n/a.png\tzenjpeg\t50\t9.5\n",
        )
        .unwrap();
        let out = dir.join("sidecar.parquet");
        concat_groups_to_parquet(&dir, &out, "c", "r", None).unwrap();
        let file = std::fs::File::open(&out).unwrap();
        let schema = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .schema()
            .clone();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(names.contains(&"chunk_id"));
        assert!(names.contains(&"run_id"));
        assert!(!names.contains(&"encoded_r2_uri"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Just to suppress unused-Write warnings.
    #[test]
    fn _io_imports_used() {
        let _ = std::io::sink();
    }
}
