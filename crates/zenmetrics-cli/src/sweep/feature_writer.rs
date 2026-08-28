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
//! That is **sidecar schema `1`** (SDR — byte-identical to every sidecar ever
//! written; nothing about it changed). **Sidecar schema `2.0-hdr`**
//! (zenmetrics#13 §5, [`FeatureParquetWriter::create_with_n_hdr`]) is the HDR
//! shape: the same columns, then four trailing per-row HDR columns so
//! concatenated chunk files stay self-describing:
//! ```text
//! hdr_mode             : utf8      `nits-mcll` — absolute cd/m² from the source's
//!                                  own signaling, luminance-aware paths at the
//!                                  REFERENCE's measured MaxCLL peak (appendix AA)
//! feature_regime       : utf8      `pu21-u8-shell` (v1 HDR regime: PU21-rescale
//!                                  → u8 → sRGB zensim features) or `pu-linear`
//!                                  (v3 / Profile B-HDR regime: absolute nits →
//!                                  zensim's integrated PU-XYB features)
//! hdr_source           : utf8      the source's signaling class — `linear-exr`,
//!                                  `cicp-png` (PQ 16 / HLG 18), `pq-jxl`,
//!                                  `pq-avif`, `gainmap-jpeg`, `gainmap-heic`
//! ref_peak_nits        : float32   the reference's measured MaxCLL peak (cd/m²)
//! ```
//! Both shapes stamp parquet key-value metadata `zenmetrics.sidecar_schema`
//! (`"1"` / `"2.0-hdr"`) and `zenmetrics.n_features`, so a reader can sniff
//! the version from the footer; a schema-1 reader that selects columns by
//! name (every reader in `scripts/` does — `feat_*` by prefix, identity by
//! name) reads a `2.0-hdr` file unchanged and simply never sees the trailing
//! columns. SDR sidecars are NOT given the HDR columns (mirrors the sweep
//! TSV's `hdr_mode` discipline: SDR output byte-identical, HDR output
//! self-describing).
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
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

/// Parquet key-value metadata key carrying the sidecar schema version
/// ([`SIDECAR_SCHEMA_V1`] / [`SIDECAR_SCHEMA_V2_HDR`]).
pub const META_SIDECAR_SCHEMA: &str = "zenmetrics.sidecar_schema";
/// Parquet key-value metadata key carrying the configured feature count.
pub const META_N_FEATURES: &str = "zenmetrics.n_features";
/// Schema `1`: the SDR shape (identity + `zensim_score` + `feat_*`).
pub const SIDECAR_SCHEMA_V1: &str = "1";
/// Schema `2.0-hdr`: schema 1 + the four trailing HDR columns.
pub const SIDECAR_SCHEMA_V2_HDR: &str = "2.0-hdr";
/// The `hdr_mode` value `score-pairs --hdr` writes: absolute nits from the
/// source's own signaling, luminance-aware paths at the reference's measured
/// MaxCLL peak. (The `sweep --hdr` TSV's `pq-mcll` is the PQ-PNG-only special
/// case of the same discipline.)
pub const HDR_MODE_NITS_MCLL: &str = "nits-mcll";

/// Which zensim feature space an HDR sidecar row was extracted in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureRegimeTag {
    /// v1 HDR regime: PU21-rescale → u8 → the sRGB zensim feature path.
    Pu21U8Shell,
    /// v3 / Profile B-HDR regime: absolute nits → zensim's integrated
    /// PU-XYB features (`--hdr-features-pu-linear`).
    PuLinear,
}

impl FeatureRegimeTag {
    /// The column value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pu21U8Shell => "pu21-u8-shell",
            Self::PuLinear => "pu-linear",
        }
    }
}

/// Per-row HDR provenance for [`FeatureParquetWriter::push_row_hdr`].
#[derive(Clone, Copy, Debug)]
pub struct HdrRowMeta<'a> {
    /// See [`HDR_MODE_NITS_MCLL`].
    pub hdr_mode: &'a str,
    /// Feature space the row's `feat_*` were extracted in.
    pub feature_regime: FeatureRegimeTag,
    /// Source signaling class (see the module docs' `hdr_source`).
    pub hdr_source: &'a str,
    /// The reference's measured MaxCLL peak, cd/m².
    pub ref_peak_nits: f32,
}

/// Default number of features when the caller uses the legacy
/// [`FeatureParquetWriter::create`] constructor. Matches CPU zensim's
/// `compute_extended_features` 300-D block (4 scales × 3 channels × 25
/// features/channel). New callers should use
/// [`FeatureParquetWriter::create_with_n`] and pass the regime's
/// `total_features()` (228 / 300 / 372) explicitly.
#[allow(dead_code)]
pub const NUM_FEATURES_DEFAULT: usize = 300;

/// Legacy alias for [`NUM_FEATURES_DEFAULT`]. Kept so downstream
/// references (`zenfleet_vastai::worker::feature_backfill::NUM_FEATURES`) continue
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

/// The four trailing HDR columns of schema `2.0-hdr`, buffered per batch.
struct HdrColumns {
    hdr_mode: Vec<String>,
    feature_regime: Vec<String>,
    hdr_source: Vec<String>,
    ref_peak_nits: Vec<f32>,
}

impl HdrColumns {
    fn new() -> Self {
        Self {
            hdr_mode: Vec::with_capacity(FLUSH_EVERY),
            feature_regime: Vec::with_capacity(FLUSH_EVERY),
            hdr_source: Vec::with_capacity(FLUSH_EVERY),
            ref_peak_nits: Vec::with_capacity(FLUSH_EVERY),
        }
    }

    fn clear(&mut self) {
        self.hdr_mode.clear();
        self.feature_regime.clear();
        self.hdr_source.clear();
        self.ref_peak_nits.clear();
    }
}

struct RowBuffer {
    image_path: Vec<String>,
    codec: Vec<String>,
    q: Vec<f64>,
    knob_tuple_json: Vec<String>,
    zensim_score: Vec<f32>,
    /// `feature_columns[i]` collects the values for `feat_i` across rows.
    feature_columns: Vec<Vec<f32>>,
    /// `Some` on a `2.0-hdr` writer, `None` on schema 1.
    hdr: Option<HdrColumns>,
    rows: usize,
}

impl RowBuffer {
    fn new(n_features: usize, hdr: bool) -> Self {
        Self {
            image_path: Vec::with_capacity(FLUSH_EVERY),
            codec: Vec::with_capacity(FLUSH_EVERY),
            q: Vec::with_capacity(FLUSH_EVERY),
            knob_tuple_json: Vec::with_capacity(FLUSH_EVERY),
            zensim_score: Vec::with_capacity(FLUSH_EVERY),
            feature_columns: (0..n_features)
                .map(|_| Vec::with_capacity(FLUSH_EVERY))
                .collect(),
            hdr: hdr.then(HdrColumns::new),
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
        if let Some(h) = &mut self.hdr {
            h.clear();
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
    pub fn create_with_n(path: &Path, n: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::create_inner(path, n, false)
    }

    /// Create a **schema `2.0-hdr`** writer (zenmetrics#13 §5): schema 1's
    /// columns plus the four trailing per-row HDR columns (`hdr_mode`,
    /// `feature_regime`, `hdr_source`, `ref_peak_nits`) — see the module
    /// docs. Rows go in through [`Self::push_row_hdr`]; [`Self::push_row`]
    /// errors on this writer so an HDR sidecar can never carry rows with
    /// missing provenance. Overwrites if the file exists.
    pub fn create_with_n_hdr(path: &Path, n: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::create_inner(path, n, true)
    }

    fn create_inner(path: &Path, n: usize, hdr: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let schema = Arc::new(build_schema(n, hdr));
        let file = File::create(path)?;
        let version = if hdr {
            SIDECAR_SCHEMA_V2_HDR
        } else {
            SIDECAR_SCHEMA_V1
        };
        let props = WriterProperties::builder()
            // zstd is a compromise: smaller than snappy, slower than lz4 but
            // not as slow as gzip. We're producing GB-class sweeps where disk
            // and bandwidth dominate; the writer cost per cell is a rounding
            // error next to the zensim compute itself.
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            // Footer metadata so readers can sniff the sidecar version
            // without inspecting column names.
            .set_key_value_metadata(Some(vec![
                KeyValue::new(META_SIDECAR_SCHEMA.to_string(), version.to_string()),
                KeyValue::new(META_N_FEATURES.to_string(), n.to_string()),
            ]))
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        Ok(Self {
            schema,
            writer,
            buf: RowBuffer::new(n, hdr),
            n_features: n,
        })
    }

    /// Configured feature count for this writer (228 / 300 / 372).
    #[allow(dead_code)]
    pub fn num_features(&self) -> usize {
        self.n_features
    }

    /// `true` for a schema `2.0-hdr` writer ([`Self::create_with_n_hdr`]).
    pub fn is_hdr(&self) -> bool {
        self.buf.hdr.is_some()
    }

    /// Append one cell to the buffer. `features` must have length matching
    /// [`Self::num_features`]; mismatch returns an error rather than
    /// silently truncating or padding. Errors on a schema `2.0-hdr` writer
    /// (use [`Self::push_row_hdr`]).
    pub fn push_row(
        &mut self,
        image_path: &str,
        codec: &str,
        q: f64,
        knob_tuple_json: &str,
        zensim_score: f32,
        features: &[f64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.is_hdr() {
            return Err(
                "feature_writer: schema 2.0-hdr writer needs push_row_hdr (HDR provenance \
                        columns are mandatory per row)"
                    .into(),
            );
        }
        self.push_row_inner(
            image_path,
            codec,
            q,
            knob_tuple_json,
            zensim_score,
            features,
            None,
        )
    }

    /// [`Self::push_row`] for a schema `2.0-hdr` writer: the same cell plus
    /// its per-row HDR provenance. Errors on a schema-1 writer.
    pub fn push_row_hdr(
        &mut self,
        image_path: &str,
        codec: &str,
        q: f64,
        knob_tuple_json: &str,
        zensim_score: f32,
        features: &[f64],
        hdr: HdrRowMeta<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_hdr() {
            return Err(
                "feature_writer: push_row_hdr on a schema-1 (SDR) writer — create the \
                        sidecar with create_with_n_hdr for HDR rows"
                    .into(),
            );
        }
        self.push_row_inner(
            image_path,
            codec,
            q,
            knob_tuple_json,
            zensim_score,
            features,
            Some(hdr),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn push_row_inner(
        &mut self,
        image_path: &str,
        codec: &str,
        q: f64,
        knob_tuple_json: &str,
        zensim_score: f32,
        features: &[f64],
        hdr: Option<HdrRowMeta<'_>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.n_features;
        if features.len() != n {
            return Err(format!(
                "feature_writer: expected {n} features, got {}",
                features.len()
            )
            .into());
        }
        if let (Some(cols), Some(meta)) = (&mut self.buf.hdr, hdr) {
            cols.hdr_mode.push(meta.hdr_mode.to_string());
            cols.feature_regime
                .push(meta.feature_regime.as_str().to_string());
            cols.hdr_source.push(meta.hdr_source.to_string());
            cols.ref_peak_nits.push(meta.ref_peak_nits);
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

fn build_schema(n: usize, hdr: bool) -> Schema {
    let mut fields: Vec<Field> = Vec::with_capacity(9 + n);
    fields.push(Field::new("image_path", DataType::Utf8, false));
    fields.push(Field::new("codec", DataType::Utf8, false));
    fields.push(Field::new("q", DataType::Float64, false));
    fields.push(Field::new("knob_tuple_json", DataType::Utf8, false));
    fields.push(Field::new("zensim_score", DataType::Float32, false));
    for i in 0..n {
        fields.push(Field::new(format!("feat_{i}"), DataType::Float32, false));
    }
    if hdr {
        // Trailing, so `feat_*` keep their positions for any positional
        // reader of schema 1.
        fields.push(Field::new("hdr_mode", DataType::Utf8, false));
        fields.push(Field::new("feature_regime", DataType::Utf8, false));
        fields.push(Field::new("hdr_source", DataType::Utf8, false));
        fields.push(Field::new("ref_peak_nits", DataType::Float32, false));
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
    if let Some(h) = &buf.hdr {
        cols.push(Arc::new(StringArray::from(h.hdr_mode.clone())));
        cols.push(Arc::new(StringArray::from(h.feature_regime.clone())));
        cols.push(Arc::new(StringArray::from(h.hdr_source.clone())));
        cols.push(Arc::new(Float32Array::from(h.ref_peak_nits.clone())));
    }
    Ok(RecordBatch::try_new(schema.clone(), cols)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::file::reader::FileReader;

    /// Column names + footer key-value metadata + row count of a written file.
    fn read_back(path: &Path) -> (Vec<String>, Vec<(String, String)>, i64) {
        let file = File::open(path).expect("open");
        let reader = parquet::file::reader::SerializedFileReader::new(file).expect("reader");
        let meta = reader.metadata();
        let names = meta
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        let kv = meta
            .file_metadata()
            .key_value_metadata()
            .map(|kv| {
                kv.iter()
                    .map(|k| (k.key.clone(), k.value.clone().unwrap_or_default()))
                    .collect()
            })
            .unwrap_or_default();
        (names, kv, meta.file_metadata().num_rows())
    }

    fn feats(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * 0.5).collect()
    }

    /// Schema 1 is byte-for-byte the historical shape (identity + score +
    /// feat_*) — no HDR columns — and now carries the version footer.
    #[test]
    fn schema_v1_shape_and_footer_unchanged_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1.parquet");
        let mut w = FeatureParquetWriter::create_with_n(&path, 228).unwrap();
        assert!(!w.is_hdr());
        w.push_row("a.png", "zenjxl", 50.0, "{}", 80.0, &feats(228))
            .unwrap();
        w.finish().unwrap();
        let (names, kv, rows) = read_back(&path);
        assert_eq!(rows, 1);
        assert_eq!(names.len(), 5 + 228);
        assert_eq!(
            &names[..5],
            &[
                "image_path",
                "codec",
                "q",
                "knob_tuple_json",
                "zensim_score"
            ]
        );
        assert_eq!(names[5], "feat_0");
        assert_eq!(names[5 + 227], "feat_227");
        assert!(!names.iter().any(|n| n == "hdr_mode"));
        assert!(kv.contains(&(
            META_SIDECAR_SCHEMA.to_string(),
            SIDECAR_SCHEMA_V1.to_string()
        )));
        assert!(kv.contains(&(META_N_FEATURES.to_string(), "228".to_string())));
    }

    /// Schema 2.0-hdr = schema 1 + the four trailing HDR columns, with the
    /// `feat_*` positions untouched, and the `2.0-hdr` footer.
    #[test]
    fn schema_v2_hdr_trailing_columns_and_footer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v2.parquet");
        let mut w = FeatureParquetWriter::create_with_n_hdr(&path, 372).unwrap();
        assert!(w.is_hdr());
        let meta = HdrRowMeta {
            hdr_mode: HDR_MODE_NITS_MCLL,
            feature_regime: FeatureRegimeTag::PuLinear,
            hdr_source: "linear-exr",
            ref_peak_nits: 1234.5,
        };
        w.push_row_hdr("a.exr", "zenjxl", 50.0, "{}", 70.0, &feats(372), meta)
            .unwrap();
        w.push_row_hdr("a.exr", "zenjxl", 90.0, "{}", 90.0, &feats(372), meta)
            .unwrap();
        w.finish().unwrap();
        let (names, kv, rows) = read_back(&path);
        assert_eq!(rows, 2);
        assert_eq!(names.len(), 5 + 372 + 4);
        assert_eq!(names[5], "feat_0");
        assert_eq!(names[5 + 371], "feat_371");
        assert_eq!(
            &names[5 + 372..],
            &["hdr_mode", "feature_regime", "hdr_source", "ref_peak_nits"]
        );
        assert!(kv.contains(&(
            META_SIDECAR_SCHEMA.to_string(),
            SIDECAR_SCHEMA_V2_HDR.to_string()
        )));
        assert!(kv.contains(&(META_N_FEATURES.to_string(), "372".to_string())));

        // Values round-trip (read the first row group's HDR columns back).
        let file = File::open(&path).unwrap();
        let reader = parquet::file::reader::SerializedFileReader::new(file).unwrap();
        let mut iter = reader.get_row_iter(None).unwrap();
        let row = iter.next().unwrap().unwrap();
        let cols: Vec<(String, String)> = row
            .get_column_iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect();
        let get = |k: &str| {
            cols.iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.clone())
                .unwrap()
        };
        assert_eq!(get("hdr_mode"), format!("\"{HDR_MODE_NITS_MCLL}\""));
        assert_eq!(get("feature_regime"), "\"pu-linear\"");
        assert_eq!(get("hdr_source"), "\"linear-exr\"");
        assert_eq!(get("ref_peak_nits"), "1234.5");
    }

    /// The two row entries are shape-locked to their writer: an HDR writer
    /// refuses provenance-less rows and an SDR writer refuses HDR rows, so a
    /// mis-shaped sidecar cannot be written by calling the wrong entry.
    #[test]
    fn push_row_entries_are_shape_locked() {
        let dir = tempfile::tempdir().unwrap();
        let meta = HdrRowMeta {
            hdr_mode: HDR_MODE_NITS_MCLL,
            feature_regime: FeatureRegimeTag::Pu21U8Shell,
            hdr_source: "cicp-png",
            ref_peak_nits: 1000.0,
        };
        let mut hdr =
            FeatureParquetWriter::create_with_n_hdr(&dir.path().join("h.parquet"), 228).unwrap();
        let err = hdr
            .push_row("a", "c", 1.0, "{}", 1.0, &feats(228))
            .expect_err("push_row on an HDR writer must fail");
        assert!(err.to_string().contains("push_row_hdr"), "{err}");
        let mut sdr =
            FeatureParquetWriter::create_with_n(&dir.path().join("s.parquet"), 228).unwrap();
        let err = sdr
            .push_row_hdr("a", "c", 1.0, "{}", 1.0, &feats(228), meta)
            .expect_err("push_row_hdr on an SDR writer must fail");
        assert!(err.to_string().contains("create_with_n_hdr"), "{err}");
        // Feature-count mismatch is still rejected on the HDR entry.
        let err = hdr
            .push_row_hdr("a", "c", 1.0, "{}", 1.0, &feats(227), meta)
            .expect_err("wrong feature count");
        assert!(err.to_string().contains("expected 228"), "{err}");
    }
}
