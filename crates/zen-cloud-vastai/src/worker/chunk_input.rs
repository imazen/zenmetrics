//! Parquet input reader for chunk processing.
//!
//! Replaces the bash worker's Python step:
//!
//! ```python
//! t = pq.read_table(p, columns=['image_path','codec','q','knob_tuple_json']).slice(rs, re_)
//! for i in range(t.num_rows):
//!     ip = t['image_path'][i].as_py()
//!     ...
//!     groups[(codec, knob_tuple_json)]['qs'].add(q)
//! ```
//!
//! Reads the projected columns via arrow-rs, slices the row range
//! the chunk specifies, then groups by `(codec, knob_tuple_json)`.
//! Returns a `Vec<ChunkGroup>` ready for the per-group sweep
//! dispatcher.
//!
//! ## Schema assumptions
//!
//! Input parquets are produced by the `zentrain/tools` pipeline and
//! always have these columns, all required, no nulls:
//!
//! - `image_path` (Utf8) — source image path (any URI flavour).
//! - `codec` (Utf8) — codec name, e.g. `"zenjpeg"`, `"zenwebp"`.
//! - `q` (Int64) — quality.
//! - `knob_tuple_json` (Utf8) — JSON object describing the knob
//!   point for this row, e.g.
//!   `{"effort":1,"subsampling":"444",...}`.
//!
//! Reader is tolerant of additional columns (they're projected away)
//! but strict about the four required columns being present.

#![cfg(feature = "inline-sweep")]

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use arrow_array::cast::AsArray as _;
use arrow_array::types::Int64Type;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// One (codec, knob_tuple_json) group's worth of cells, ready to be
/// turned into an `InlineGroupSpec` and handed to `run_group_inline`.
#[derive(Debug, Clone)]
pub struct ChunkGroup {
    /// Codec name from the input parquet. The dispatcher maps this
    /// string to [`zen_metrics_cli::sweep::CodecKind`] via FromStr —
    /// keeping the raw string here keeps this module decoupled from
    /// the codec enum (testable without features).
    pub codec: String,
    /// The original `knob_tuple_json` cell — one specific point in
    /// knob-space. The caller will lift this to a single-element
    /// Cartesian grid via [`super::sweep_runner::knob_tuple_to_grid_json`].
    pub knob_tuple_json: String,
    /// Unique q values that need to run for this group, in sorted
    /// order. Sorted for determinism and so the comma-list passed
    /// to `--q-grid` is human-readable.
    pub q_values: Vec<u32>,
    /// Unique image basenames in this group. Used to symlink only
    /// the needed sources into the group's scratch directory (the
    /// bash worker does the same dance with awk + ln).
    pub image_basenames: Vec<String>,
}

/// Read a chunk's input parquet, slice the row range, and group.
///
/// `parquet_path` is on local disk (caller already downloaded it
/// from R2). `row_start`/`row_end` are the half-open `[start, end)`
/// row range — same shape as Python's `slice(start, end)`.
///
/// Returns groups in arbitrary order; the dispatcher sorts them
/// before iterating so logs are deterministic across runs.
pub fn read_and_group(
    parquet_path: &Path,
    row_start: usize,
    row_end: usize,
) -> Result<Vec<ChunkGroup>> {
    let file = std::fs::File::open(parquet_path)
        .with_context(|| format!("open parquet {}", parquet_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).context("init parquet reader")?;

    // Project to the four columns we need. Some chunk parquets
    // carry encoded_bytes / encode_ms / etc. — irrelevant here.
    let schema = builder.schema().clone();
    let want = ["image_path", "codec", "q", "knob_tuple_json"];
    let mut indices = Vec::with_capacity(4);
    for name in &want {
        let idx = schema
            .index_of(name)
            .with_context(|| format!("input parquet missing required column `{name}`"))?;
        indices.push(idx);
    }
    let projection = parquet::arrow::ProjectionMask::leaves(builder.parquet_schema(), indices);

    let reader = builder
        .with_projection(projection)
        .build()
        .context("build parquet reader")?;

    // Streaming aggregation. We don't materialise the slice into a
    // single big Arrow table — incremental fold into the BTreeMap
    // keeps peak memory bounded by the number of unique groups (a
    // few dozen) rather than the full chunk (200 rows × 4 cols).
    let mut by_group: BTreeMap<(String, String), GroupAccum> = BTreeMap::new();
    let mut consumed: usize = 0;

    for batch in reader {
        let batch = batch.context("read parquet batch")?;
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }

        // Per-batch row-range intersection with [row_start, row_end).
        let batch_start = consumed;
        let batch_end = consumed + n;
        consumed = batch_end;
        if batch_end <= row_start {
            continue;
        }
        if batch_start >= row_end {
            break;
        }
        let lo = row_start.saturating_sub(batch_start);
        let hi = (row_end - batch_start).min(n);

        // Locate the column arrays. Indices match the projection
        // order [image_path, codec, q, knob_tuple_json].
        let image_paths = batch
            .column(batch.schema().index_of("image_path")?)
            .as_string::<i32>();
        let codecs = batch
            .column(batch.schema().index_of("codec")?)
            .as_string::<i32>();
        let qs = batch
            .column(batch.schema().index_of("q")?)
            .as_primitive::<Int64Type>();
        let knob_tuples = batch
            .column(batch.schema().index_of("knob_tuple_json")?)
            .as_string::<i32>();

        for i in lo..hi {
            let image_path = image_paths.value(i);
            let codec = codecs.value(i).to_string();
            let q_i64 = qs.value(i);
            let knob_tuple = knob_tuples.value(i).to_string();
            if !(0..=100).contains(&q_i64) {
                return Err(anyhow!(
                    "q={q_i64} out of [0,100] at row {}",
                    batch_start + i
                ));
            }
            let q = q_i64 as u32;
            let basename = image_path
                .rsplit('/')
                .next()
                .unwrap_or(image_path)
                .to_string();
            by_group
                .entry((codec, knob_tuple))
                .or_default()
                .add(q, basename);
        }
    }

    Ok(by_group
        .into_iter()
        .map(|((codec, knob_tuple_json), accum)| ChunkGroup {
            codec,
            knob_tuple_json,
            q_values: accum.into_sorted_qs(),
            image_basenames: accum.into_sorted_basenames(),
        })
        .collect())
}

/// Per-group accumulator. Builds the unique q set + unique basename
/// set, with the actual sort-and-dedup done lazily in
/// `into_sorted_*` so the hot path stays branchless.
#[derive(Default)]
struct GroupAccum {
    qs: Vec<u32>,
    basenames: Vec<String>,
}

impl GroupAccum {
    fn add(&mut self, q: u32, basename: String) {
        self.qs.push(q);
        self.basenames.push(basename);
    }

    fn into_sorted_qs(&self) -> Vec<u32> {
        let mut v = self.qs.clone();
        v.sort_unstable();
        v.dedup();
        v
    }

    fn into_sorted_basenames(&self) -> Vec<String> {
        let mut v = self.basenames.clone();
        v.sort();
        v.dedup();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    fn write_test_parquet(path: &Path, rows: &[(&str, &str, i64, &str)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("image_path", DataType::Utf8, false),
            Field::new("codec", DataType::Utf8, false),
            Field::new("q", DataType::Int64, false),
            Field::new("knob_tuple_json", DataType::Utf8, false),
        ]));
        let ip: StringArray = rows.iter().map(|r| Some(r.0)).collect();
        let cd: StringArray = rows.iter().map(|r| Some(r.1)).collect();
        let q: Int64Array = rows.iter().map(|r| Some(r.2)).collect();
        let kt: StringArray = rows.iter().map(|r| Some(r.3)).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ip), Arc::new(cd), Arc::new(q), Arc::new(kt)],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut wtr = ArrowWriter::try_new(file, schema, None).unwrap();
        wtr.write(&batch).unwrap();
        wtr.close().unwrap();
    }

    #[test]
    fn group_by_codec_and_knobs() {
        let tmp = tempdir();
        let p = tmp.path().join("input.parquet");
        write_test_parquet(
            &p,
            &[
                ("/a/img1.png", "zenjpeg", 50, r#"{"effort":1}"#),
                ("/a/img1.png", "zenjpeg", 75, r#"{"effort":1}"#),
                ("/a/img2.png", "zenjpeg", 50, r#"{"effort":2}"#),
                ("/a/img1.png", "zenwebp", 75, r#"{"effort":1}"#),
            ],
        );
        let mut groups = read_and_group(&p, 0, 4).unwrap();
        groups.sort_by(|a, b| (&a.codec, &a.knob_tuple_json).cmp(&(&b.codec, &b.knob_tuple_json)));
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].codec, "zenjpeg");
        assert_eq!(groups[0].q_values, vec![50, 75]);
        assert_eq!(groups[0].image_basenames, vec!["img1.png".to_string()]);
        assert_eq!(groups[2].codec, "zenwebp");
        assert_eq!(groups[2].q_values, vec![75]);
    }

    #[test]
    fn row_range_slicing() {
        let tmp = tempdir();
        let p = tmp.path().join("input.parquet");
        let rows: Vec<(&str, &str, i64, &str)> = (0..10)
            .map(|i| {
                (
                    if i % 2 == 0 { "/x/a.png" } else { "/x/b.png" },
                    "zenjpeg",
                    50 + i,
                    r#"{"effort":1}"#,
                )
            })
            .collect();
        write_test_parquet(&p, &rows);
        // Slice [3, 7) — 4 rows. Q values 53, 54, 55, 56 → unique.
        let groups = read_and_group(&p, 3, 7).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].q_values, vec![53, 54, 55, 56]);
    }

    #[test]
    fn missing_required_column_errors() {
        let tmp = tempdir();
        let p = tmp.path().join("input.parquet");
        // Schema missing `knob_tuple_json`.
        let schema = Arc::new(Schema::new(vec![
            Field::new("image_path", DataType::Utf8, false),
            Field::new("codec", DataType::Utf8, false),
            Field::new("q", DataType::Int64, false),
        ]));
        let ip = Arc::new(StringArray::from(vec!["x"]));
        let cd = Arc::new(StringArray::from(vec!["zenjpeg"]));
        let q = Arc::new(Int64Array::from(vec![50_i64]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ip, cd, q]).unwrap();
        let file = std::fs::File::create(&p).unwrap();
        let mut wtr = ArrowWriter::try_new(file, schema, None).unwrap();
        wtr.write(&batch).unwrap();
        wtr.close().unwrap();

        let err = read_and_group(&p, 0, 1).unwrap_err().to_string();
        assert!(err.contains("knob_tuple_json"), "got: {err}");
    }

    /// Trivial tempdir helper — we don't want a `tempfile` dep just
    /// for tests. Lives under target/test-tmp/<pid>-<nanos>.
    fn tempdir() -> TempDir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::path::PathBuf::from(format!(
            "{}/vastai-fleet-test-{}-{}",
            std::env::temp_dir().display(),
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&path).unwrap();
        TempDir { path }
    }
    struct TempDir {
        path: std::path::PathBuf,
    }
    impl TempDir {
        fn path(&self) -> &Path {
            &self.path
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
