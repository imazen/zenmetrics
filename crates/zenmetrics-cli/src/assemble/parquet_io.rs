#![forbid(unsafe_code)]

//! Parquet ⇄ [`Table`] bridge, built on the `arrow`/`parquet` crates already
//! vendored by the `sweep` feature (see `Cargo.toml`). No DuckDB.
//!
//! Read path mirrors the established `ParquetRecordBatchReaderBuilder` pattern
//! used in `crates/zenfleet-vastai/src/worker/feature_backfill.rs`. We accept the
//! handful of arrow types that appear in the corpus parquets (Utf8 / Float64 /
//! Float32 / Int64 / UInt32 / Null) and coerce each to one of the three
//! logical [`Column`] variants. Anything else is an error rather than a silent
//! drop — a surprising dtype in a training corpus is exactly the kind of thing
//! that should fail loudly.
//!
//! Write path emits a zstd-compressed parquet preserving column order, the
//! same writer configuration `feature_writer.rs` uses (`ZstdLevel(3)`), so
//! round-trips are byte-stable across the two writers for identical data.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::{Float32Type, Float64Type, Int64Type, UInt32Type};
// `Array` brings the `is_null` inherent-trait method into scope for the
// downcast primitive arrays read back from parquet.
use arrow_array::{Array as _, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use super::table::{AssembleError, Column, Table};

/// Read a parquet file at `path` into a [`Table`]. Coerces each arrow column
/// to a logical [`Column`] variant:
/// - `Utf8`/`LargeUtf8` → [`Column::Str`]
/// - `Int64`/`UInt32` → [`Column::I64`]
/// - `Float64`/`Float32` → [`Column::F64`]
/// - `Null` → [`Column::F64`] of all-`NaN` (the all-null arrow type that
///   appears when a metric went null for an entire chunk — same widening the
///   union step performs)
pub fn read_parquet(path: &Path) -> Result<Table, AssembleError> {
    let file =
        File::open(path).map_err(|e| AssembleError::Io(format!("open {}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        AssembleError::Io(format!("parquet reader builder {}: {e}", path.display()))
    })?;
    let schema = builder.schema().clone();
    let reader = builder
        .build()
        .map_err(|e| AssembleError::Io(format!("build reader {}: {e}", path.display())))?;

    // Accumulate per-column vectors in schema order.
    let n_cols = schema.fields().len();
    let mut names: Vec<String> = Vec::with_capacity(n_cols);
    let mut str_cols: Vec<Option<Vec<Option<String>>>> = Vec::with_capacity(n_cols);
    let mut i64_cols: Vec<Option<Vec<i64>>> = Vec::with_capacity(n_cols);
    let mut f64_cols: Vec<Option<Vec<f64>>> = Vec::with_capacity(n_cols);

    for field in schema.fields() {
        names.push(field.name().clone());
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => {
                str_cols.push(Some(Vec::new()));
                i64_cols.push(None);
                f64_cols.push(None);
            }
            DataType::Int64 | DataType::UInt32 => {
                str_cols.push(None);
                i64_cols.push(Some(Vec::new()));
                f64_cols.push(None);
            }
            DataType::Float64 | DataType::Float32 | DataType::Null => {
                str_cols.push(None);
                i64_cols.push(None);
                f64_cols.push(Some(Vec::new()));
            }
            other => {
                return Err(AssembleError::Schema(format!(
                    "{}: column {:?} has unsupported parquet type {other:?}",
                    path.display(),
                    field.name()
                )));
            }
        }
    }

    for batch in reader {
        let batch =
            batch.map_err(|e| AssembleError::Io(format!("read batch {}: {e}", path.display())))?;
        for (ci, field) in schema.fields().iter().enumerate() {
            let arr = batch.column(ci);
            let n = batch.num_rows();
            append_arrow_column(
                field.data_type(),
                arr,
                n,
                ci,
                &mut str_cols,
                &mut i64_cols,
                &mut f64_cols,
            )
            .map_err(|e| {
                AssembleError::Schema(format!(
                    "{}: column {:?}: {e}",
                    path.display(),
                    field.name()
                ))
            })?;
        }
    }

    let mut cols: Vec<(String, Column)> = Vec::with_capacity(n_cols);
    for ci in 0..n_cols {
        let col = if let Some(v) = str_cols[ci].take() {
            Column::Str(v)
        } else if let Some(v) = i64_cols[ci].take() {
            Column::I64(v)
        } else {
            Column::F64(f64_cols[ci].take().unwrap_or_default())
        };
        cols.push((names[ci].clone(), col));
    }
    Table::from_columns(cols)
}

#[allow(clippy::too_many_arguments)]
fn append_arrow_column(
    dt: &DataType,
    arr: &ArrayRef,
    n: usize,
    ci: usize,
    str_cols: &mut [Option<Vec<Option<String>>>],
    i64_cols: &mut [Option<Vec<i64>>],
    f64_cols: &mut [Option<Vec<f64>>],
) -> Result<(), String> {
    match dt {
        DataType::Utf8 => {
            let a = arr.as_string::<i32>();
            let dst = str_cols[ci].as_mut().unwrap();
            for i in 0..n {
                dst.push(if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                });
            }
        }
        DataType::LargeUtf8 => {
            let a = arr.as_string::<i64>();
            let dst = str_cols[ci].as_mut().unwrap();
            for i in 0..n {
                dst.push(if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                });
            }
        }
        DataType::Int64 => {
            let a = arr.as_primitive::<Int64Type>();
            let dst = i64_cols[ci].as_mut().unwrap();
            for i in 0..n {
                dst.push(if a.is_null(i) { 0 } else { a.value(i) });
            }
        }
        DataType::UInt32 => {
            let a = arr.as_primitive::<UInt32Type>();
            let dst = i64_cols[ci].as_mut().unwrap();
            for i in 0..n {
                dst.push(if a.is_null(i) { 0 } else { a.value(i) as i64 });
            }
        }
        DataType::Float64 => {
            let a = arr.as_primitive::<Float64Type>();
            let dst = f64_cols[ci].as_mut().unwrap();
            for i in 0..n {
                dst.push(if a.is_null(i) { f64::NAN } else { a.value(i) });
            }
        }
        DataType::Float32 => {
            let a = arr.as_primitive::<Float32Type>();
            let dst = f64_cols[ci].as_mut().unwrap();
            for i in 0..n {
                dst.push(if a.is_null(i) {
                    f64::NAN
                } else {
                    a.value(i) as f64
                });
            }
        }
        DataType::Null => {
            let dst = f64_cols[ci].as_mut().unwrap();
            for _ in 0..n {
                dst.push(f64::NAN);
            }
        }
        other => return Err(format!("unsupported parquet type {other:?}")),
    }
    Ok(())
}

/// Write `table` to `path` as a zstd-level-3 parquet, preserving column order
/// and logical types. Same writer config as `feature_writer.rs`, so two
/// writers emit byte-identical files for identical data.
pub fn write_parquet(table: &Table, path: &Path) -> Result<(), AssembleError> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| AssembleError::Io(format!("mkdir {}: {e}", parent.display())))?;
    }

    let mut fields: Vec<Field> = Vec::with_capacity(table.num_columns());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(table.num_columns());
    for name in table.column_names() {
        let col = table.column(name).expect("name from column_names");
        match col {
            Column::Str(v) => {
                fields.push(Field::new(name, DataType::Utf8, true));
                arrays.push(Arc::new(StringArray::from(
                    v.iter().map(|o| o.as_deref()).collect::<Vec<_>>(),
                )) as ArrayRef);
            }
            Column::I64(v) => {
                fields.push(Field::new(name, DataType::Int64, false));
                arrays.push(Arc::new(Int64Array::from(v.clone())) as ArrayRef);
            }
            Column::F64(v) => {
                // NaN is preserved as a real float value (not a parquet null)
                // so the round-trip is exact; downstream readers treat NaN as
                // null per the union rule.
                fields.push(Field::new(name, DataType::Float64, false));
                arrays.push(Arc::new(Float64Array::from(v.clone())) as ArrayRef);
            }
        }
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| AssembleError::Io(format!("build record batch: {e}")))?;

    let file = File::create(path)
        .map_err(|e| AssembleError::Io(format!("create {}: {e}", path.display())))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(3).map_err(|e| AssembleError::Io(format!("zstd level: {e}")))?,
        ))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| AssembleError::Io(format!("arrow writer {}: {e}", path.display())))?;
    writer
        .write(&batch)
        .map_err(|e| AssembleError::Io(format!("write batch {}: {e}", path.display())))?;
    writer
        .close()
        .map_err(|e| AssembleError::Io(format!("close writer {}: {e}", path.display())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parquet_round_trip_is_byte_stable() {
        let t = Table::from_columns(vec![
            (
                "image_path".into(),
                Column::Str(vec![Some("a.png".into()), Some("b.png".into())]),
            ),
            ("q".into(), Column::I64(vec![50, 60])),
            ("score".into(), Column::F64(vec![1.5, 2.5])),
        ])
        .unwrap();
        let dir = tempdir().unwrap();
        let p1 = dir.path().join("a.parquet");
        let p2 = dir.path().join("b.parquet");
        write_parquet(&t, &p1).unwrap();
        // Read back, re-write, and confirm the bytes match (deterministic
        // writer config → byte-stable for identical data).
        let t2 = read_parquet(&p1).unwrap();
        write_parquet(&t2, &p2).unwrap();
        let b1 = std::fs::read(&p1).unwrap();
        let b2 = std::fs::read(&p2).unwrap();
        assert_eq!(b1, b2, "round-trip parquet must be byte-stable");
        // And the logical content survived.
        assert_eq!(t2.num_rows(), 2);
        assert_eq!(t2.column("q").unwrap().key_at(1), "60");
        assert_eq!(t2.column("score").unwrap().f64_at(0), 1.5);
    }
}
