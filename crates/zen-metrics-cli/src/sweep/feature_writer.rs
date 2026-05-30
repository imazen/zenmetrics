#![forbid(unsafe_code)]

//! Per-cell zensim feature vector → Parquet sidecar.
//!
//! When the sweep subcommand is invoked with `--feature-output <path.parquet>`,
//! every cell that runs the zensim metric also persists its 228 / 300 / 372
//! feature vector here. The TSV row continues to carry the human-readable
//! `score_*` columns (ssim2, butteraugli, zensim, etc.) and the parquet sidecar
//! is joined back to the TSV by `(image_path, codec, q, knob_tuple_json)`.
//!
//! Schema (one row per encoded cell):
//! ```text
//! image_path           : utf8
//! codec                : utf8
//! q                    : uint32
//! knob_tuple_json      : utf8
//! zensim_score         : float32
//! feat_0..feat_<N-1>   : float32   (N = num_features, 228 / 300 / 372)
//! ```
//!
//! `N` is set at writer construction via [`FeatureParquetWriter::create_with_n`]
//! (older `create(...)` keeps the legacy 300 default for back-compat). Sweep
//! callers pass the regime they configured zensim with:
//!
//! - `Basic`   → 228 columns (legacy CPU zensim, no extended block)
//! - `Extended`→ 300 columns (legacy CPU `compute_extended_features`)
//! - `WithIw`  → 372 columns (v26+ default; adds IW block on top of Extended)
//!
//! Each `run_sweep` invocation owns one writer and produces one parquet file.
//! For chunked / distributed sweeps each worker writes its own file
//! (`features-<chunk_id>.parquet`); the upstream finalize step concatenates
//! them. We don't try to "append" to an existing parquet — the format isn't
//! row-appendable without a row-group rewrite, and per-chunk files are simpler
//! to reason about.
//!
//! Rows are buffered into Arrow batches and flushed at a fixed batch size
//! (`FLUSH_EVERY`) so the in-memory footprint stays bounded even for
//! million-cell sweeps.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

/// Default number of features when the caller uses the legacy
/// [`FeatureParquetWriter::create`] constructor. Matches CPU zensim's
/// `compute_extended_features` 300-D block (4 scales × 3 channels × 25
/// features/channel). New callers should use
/// [`FeatureParquetWriter::create_with_n`] and pass the regime's
/// `total_features()` (228 / 300 / 372) explicitly.
#[allow(dead_code)]
pub const NUM_FEATURES_DEFAULT: usize = 300;

/// Legacy alias for [`NUM_FEATURES_DEFAULT`]. Kept so downstream
/// references (`zen_cloud_vastai::worker::feature_backfill::NUM_FEATURES`) continue
/// to compile; new code should not depend on a single global feature
/// count.
#[allow(dead_code)]
pub const NUM_FEATURES: usize = NUM_FEATURES_DEFAULT;

/// Common zensim feature counts. Pass one of these to
/// [`FeatureParquetWriter::create_with_n`].
#[allow(dead_code)]
pub mod num_features {
    /// 228 — `Basic` regime (no extended / IW block).
    pub const BASIC: usize = 228;
    /// 300 — `Extended` regime (Basic + 72 masked features).
    pub const EXTENDED: usize = 300;
    /// 372 — `WithIw` regime (Extended + 72 information-weighted features).
    pub const WITH_IW: usize = 372;
}

/// Flush an Arrow record batch to disk every `FLUSH_EVERY` rows. 256 keeps
/// memory bounded (≈ 256 × 304 floats × 4 B = 311 KiB per batch) while
/// amortising the parquet column-encoding fixed cost.
const FLUSH_EVERY: usize = 256;

/// Buffered Parquet writer for the per-cell feature sidecar.
///
/// Append rows with [`FeatureParquetWriter::push_row`]; call [`finish`] (or
/// drop) to flush and close the file. Dropping without `finish()` does its
/// best to flush — but a panic between the last `push_row` and the implicit
/// drop will leave a truncated file. Prefer the explicit `finish()` at the
/// end of a sweep.
pub struct FeatureParquetWriter {
    schema: Arc<Schema>,
    writer: ArrowWriter<File>,
    buf: RowBuffer,
    /// Configured feature count (228 / 300 / 372). Set once at
    /// construction; every `push_row` must pass exactly this many
    /// features or the call returns an error.
    n_features: usize,
}

struct RowBuffer {
    image_path: Vec<String>,
    codec: Vec<String>,
    q: Vec<f64>,
    knob_tuple_json: Vec<String>,
    zensim_score: Vec<f32>,
    /// `feature_columns[i]` collects the values for `feat_i` across rows.
    feature_columns: Vec<Vec<f32>>,
    rows: usize,
}

impl RowBuffer {
    fn new(n_features: usize) -> Self {
        Self {
            image_path: Vec::with_capacity(FLUSH_EVERY),
            codec: Vec::with_capacity(FLUSH_EVERY),
            q: Vec::with_capacity(FLUSH_EVERY),
            knob_tuple_json: Vec::with_capacity(FLUSH_EVERY),
            zensim_score: Vec::with_capacity(FLUSH_EVERY),
            feature_columns: (0..n_features)
                .map(|_| Vec::with_capacity(FLUSH_EVERY))
                .collect(),
            rows: 0,
        }
    }

    fn clear(&mut self) {
        self.image_path.clear();
        self.codec.clear();
        self.q.clear();
        self.knob_tuple_json.clear();
        self.zensim_score.clear();
        for col in &mut self.feature_columns {
            col.clear();
        }
        self.rows = 0;
    }
}

impl FeatureParquetWriter {
    /// Create a new parquet writer at `path` with the legacy 300-feature
    /// schema. Overwrites if the file exists. Prefer
    /// [`Self::create_with_n`] in new code so the schema matches the
    /// regime the GPU/CPU zensim is actually running.
    #[allow(dead_code)]
    pub fn create(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Self::create_with_n(path, NUM_FEATURES_DEFAULT)
    }

    /// Create a new parquet writer at `path` with `n` feature columns
    /// (`feat_0..feat_<n-1>`). Pass `num_features::WITH_IW` (372) for
    /// the v26+ default, `num_features::EXTENDED` (300) for the legacy
    /// CPU zensim block, or `num_features::BASIC` (228) for the
    /// no-extended-block fast path. Overwrites if the file exists.
    pub fn create_with_n(
        path: &Path,
        n: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let schema = Arc::new(build_schema(n));
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            // zstd is a compromise: smaller than snappy, slower than lz4 but
            // not as slow as gzip. We're producing GB-class sweeps where disk
            // and bandwidth dominate; the writer cost per cell is a rounding
            // error next to the zensim compute itself.
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        Ok(Self {
            schema,
            writer,
            buf: RowBuffer::new(n),
            n_features: n,
        })
    }

    /// Configured feature count for this writer (228 / 300 / 372).
    #[allow(dead_code)]
    pub fn num_features(&self) -> usize {
        self.n_features
    }

    /// Append one cell to the buffer. `features` must have length matching
    /// [`Self::num_features`]; mismatch returns an error rather than
    /// silently truncating or padding.
    pub fn push_row(
        &mut self,
        image_path: &str,
        codec: &str,
        q: f64,
        knob_tuple_json: &str,
        zensim_score: f32,
        features: &[f64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_features;
        if features.len() != n {
            return Err(format!(
                "feature_writer: expected {n} features, got {}",
                features.len()
            )
            .into());
        }
        self.buf.image_path.push(image_path.to_string());
        self.buf.codec.push(codec.to_string());
        self.buf.q.push(q);
        self.buf.knob_tuple_json.push(knob_tuple_json.to_string());
        self.buf.zensim_score.push(zensim_score);
        for (i, &v) in features.iter().enumerate() {
            self.buf.feature_columns[i].push(v as f32);
        }
        self.buf.rows += 1;
        if self.buf.rows >= FLUSH_EVERY {
            self.flush_buffer()?;
        }
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.buf.rows == 0 {
            return Ok(());
        }
        let batch = build_record_batch(&self.schema, &self.buf)?;
        self.writer.write(&batch)?;
        self.buf.clear();
        Ok(())
    }

    /// Flush any pending rows and close the writer. Always call this at the
    /// end of a sweep; dropping without `finish()` may leave a partial file.
    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.flush_buffer()?;
        self.writer.close()?;
        Ok(())
    }
}

fn build_schema(n: usize) -> Schema {
    let mut fields: Vec<Field> = Vec::with_capacity(5 + n);
    fields.push(Field::new("image_path", DataType::Utf8, false));
    fields.push(Field::new("codec", DataType::Utf8, false));
    fields.push(Field::new("q", DataType::Float64, false));
    fields.push(Field::new("knob_tuple_json", DataType::Utf8, false));
    fields.push(Field::new("zensim_score", DataType::Float32, false));
    for i in 0..n {
        fields.push(Field::new(format!("feat_{i}"), DataType::Float32, false));
    }
    Schema::new(fields)
}

fn build_record_batch(
    schema: &Arc<Schema>,
    buf: &RowBuffer,
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let n = buf.feature_columns.len();
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(5 + n);
    cols.push(Arc::new(StringArray::from(buf.image_path.clone())));
    cols.push(Arc::new(StringArray::from(buf.codec.clone())));
    cols.push(Arc::new(Float64Array::from(buf.q.clone())));
    cols.push(Arc::new(StringArray::from(buf.knob_tuple_json.clone())));
    cols.push(Arc::new(Float32Array::from(buf.zensim_score.clone())));
    for col in &buf.feature_columns {
        cols.push(Arc::new(Float32Array::from(col.clone())));
    }
    Ok(RecordBatch::try_new(schema.clone(), cols)?)
}
