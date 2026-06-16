#![forbid(unsafe_code)]

//! `zenmetrics assemble` — typed, full-key corpus join.
//!
//! # What this replaces and why
//!
//! This subcommand replaces the Python corpus-assembly join layer:
//! `zenmetrics/scripts/sweep/build_per_codec_training.py`, the
//! `zensim/scripts/canonical_corpus/build_*` builders, and the ~35 ad-hoc
//! `pd.merge` scripts that joined metric scores onto feature tables. Those
//! scripts produced the 2026-05-25 parquet corruption documented in
//! `zensim/benchmarks/DATA_INTEGRITY_root_cause_2026-05-25.md`. Two
//! independent bugs:
//!
//! - **Mode A** — a validation-only `iwssim := human_score` MOCK leaked into
//!   training parquets after the "mock" filename qualifier was renamed away.
//! - **Mode B** — a per-pair `ssim2_gpu` metric was joined onto a features
//!   table carrying only `ref_basename`; the join key silently collapsed to
//!   `["ref_basename"]` and a `groupby(...).mean()` broadcast one mean score
//!   onto all ~125 distortions of each reference, destroying the signal.
//!
//! # How `assemble` makes both modes impossible
//!
//! The fix is structural, not a downstream check:
//!
//! 1. **[`key::PairKey`] — compile-time defense.** The join API takes a typed
//!    four-field key (`image_path, codec, q, knob_tuple_json`). There is no
//!    constructor with fewer fields, so a caller *cannot express* the
//!    ref-only collapse. This is the type-level equivalent of
//!    `join_safety.REF_ONLY_KEYS`.
//! 2. **[`join::safe_join`] — runtime defense.** Errors (never averages) on
//!    duplicate metric keys; errors if either side lacks a per-pair column.
//! 3. **[`join::attach_positional`] — the correct KADID/TID path.** Exact
//!    length assert for ref-only tables whose metric was computed in row
//!    order.
//! 4. **[`join::assert_no_leaked_columns`] / [`join::assert_not_constant_per_ref`]**
//!    — last-line detectors for the mock leak + the ref-broadcast signature.
//!
//! Every output parquet passes the leak + constant-per-ref guards before it
//! is written. There is no codepath that emits a corrupt corpus.
//!
//! # Modes
//!
//! - `per-codec` (default) — ports `build_per_codec_training.py`: sync omni /
//!   zensim_features / source_features sidecars from R2, union-by-name within
//!   each kind, 3-way inner join on the per-pair key then per-source
//!   `image_basename`, namespace colliding `feat_<N>` columns, write one
//!   parquet per codec.

pub mod join;
pub mod key;
pub mod parquet_io;
pub mod r2_sync;
pub mod table;

use std::path::PathBuf;

use clap::Parser;

use join::{assert_no_leaked_columns, assert_not_constant_per_ref};
use table::{AssembleError, Column, Table};

/// Which assembly to run.
#[derive(Debug, Clone, Copy, clap::ValueEnum, Default)]
pub enum AssembleMode {
    /// Port of `build_per_codec_training.py`: 3-way join of omni scores ×
    /// zensim features × source features, one output parquet per codec.
    #[default]
    PerCodec,
}

/// `zenmetrics assemble` arguments. Mirrors `build_per_codec_training.py`'s
/// CLI (`--runs`, `--cache-dir`, `--out-dir`, `--codecs`) plus the R2
/// connection flags the Python script read from env.
#[derive(Parser, Debug)]
pub struct AssembleArgs {
    /// Which assembly to run. Currently `per-codec` (the
    /// build_per_codec_training.py port).
    #[arg(long, value_enum, default_value = "per-codec")]
    pub mode: AssembleMode,

    /// R2 run ids to ingest. For each run, the assembler syncs the
    /// `s3://<bucket>/<run>/{omni,zensim_features,source_features}/` sidecar
    /// prefixes. Pass once per run: `--runs a --runs b`.
    #[arg(long = "runs", action = clap::ArgAction::Append, required = true)]
    pub runs: Vec<String>,

    /// R2 bucket the run prefixes live under. Default `zentrain` matches the
    /// Python builder's hard-coded `s3://zentrain/<run>/<kind>/`.
    #[arg(long, default_value = "zentrain")]
    pub bucket: String,

    /// Local cache for downloaded sidecars. Re-used (resumable) across runs.
    #[arg(long)]
    pub cache_dir: PathBuf,

    /// Output directory for the per-codec parquets
    /// (`<out_dir>/<codec>_training.parquet`).
    #[arg(long)]
    pub out_dir: PathBuf,

    /// Filter to these codec column values. Default: every codec found in the
    /// joined data.
    #[arg(long = "codecs", action = clap::ArgAction::Append)]
    pub codecs: Vec<String>,

    /// R2 endpoint URL. Falls back to `$R2_ENDPOINT`, then
    /// `https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com`.
    #[arg(long)]
    pub r2_endpoint: Option<String>,

    /// `s5cmd` AWS profile (default `r2`, matching the Python builder).
    #[arg(long, default_value = "r2")]
    pub s5cmd_profile: String,

    /// `s5cmd` binary name / path.
    #[arg(long, default_value = "s5cmd")]
    pub s5cmd_bin: String,

    /// Skip the R2 sync and assume the sidecars are already present under
    /// `--cache-dir/<run>/<kind>/`. Useful for re-runs and offline tests.
    #[arg(long, default_value_t = false)]
    pub no_sync: bool,
}

/// Per-pair key the cell-level join uses. Defined here (not in `key`) because
/// it is the *string* names list; `PairKey::COLUMNS` is the authoritative set.
const CELL_JOIN_KEYS: [&str; 4] = key::PairKey::COLUMNS;

/// The omni score columns we project (matches build_per_codec_training.py's
/// explicit SELECT list — these are the per-cell metric scores + encode
/// stats). Columns absent in the actual sidecar are simply skipped (the
/// Python SELECT would have errored; we are lenient because run sidecars
/// evolve their score-column set over time).
const OMNI_PASSTHROUGH: &[&str] = &[
    "encoded_bytes",
    "encode_ms",
    "decode_ms",
    "encoded_filename",
    "score_zensim_gpu",
    "score_ssim2_gpu",
    "score_butteraugli_max_gpu",
    "score_butteraugli_pnorm3_gpu",
    "score_cvvdp_imazen_v0_0_1",
    "score_dssim_gpu",
    "score_iwssim_gpu",
    "run_id",
    "chunk_id",
];

/// Entry point for `zenmetrics assemble`.
pub fn run_assemble(args: &AssembleArgs) -> Result<(), AssembleError> {
    match args.mode {
        AssembleMode::PerCodec => run_per_codec(args),
    }
}

/// Port of `build_per_codec_training.py`. See module docs for the corruption
/// it prevents. file:mod.rs:run_per_codec is the orchestration; the
/// load-bearing safety lives in file:join.rs:safe_join.
fn run_per_codec(args: &AssembleArgs) -> Result<(), AssembleError> {
    let sync = if args.no_sync {
        None
    } else {
        Some(r2_sync::R2Sync::new(
            args.r2_endpoint.as_deref(),
            &args.s5cmd_profile,
            &args.s5cmd_bin,
        )?)
    };

    // === STEP A+B: sync + load + union-by-name per kind ===
    eprintln!(
        "=== assemble per-codec: load sidecars ({} runs) ===",
        args.runs.len()
    );
    let omni = load_kind(args, sync.as_ref(), "omni")?;
    eprintln!(
        "  omni: {} rows × {} cols",
        omni.num_rows(),
        omni.num_columns()
    );
    let mut zsm = load_kind(args, sync.as_ref(), "zensim_features")?;
    eprintln!(
        "  zsm:  {} rows × {} cols",
        zsm.num_rows(),
        zsm.num_columns()
    );
    let mut src = load_kind(args, sync.as_ref(), "source_features")?;
    eprintln!(
        "  src:  {} rows × {} cols",
        src.num_rows(),
        src.num_columns()
    );

    // === STEP C: namespace feat_<N> collisions ===
    // zensim features → zsm_feat_<N> (keep join keys + zensim_score).
    zsm.prefix_columns_where(
        "zsm_",
        &[
            CELL_JOIN_KEYS[0],
            CELL_JOIN_KEYS[1],
            CELL_JOIN_KEYS[2],
            CELL_JOIN_KEYS[3],
            "zensim_score",
        ],
        |n| n.starts_with("feat_"),
    );
    // source features → src_feat_<N> (keep join keys + identity cols).
    src.prefix_columns_where(
        "src_",
        &[
            "image_basename",
            "width",
            "height",
            "chunk_id",
            "run_id",
            "image_path",
        ],
        |n| n.starts_with("feat_"),
    );

    // === STEP D: 3-way join ===
    // (a) cell-level inner join: omni × zsm on the full per-pair key. This is
    //     the join that Mode B corrupted; here it goes through the typed
    //     PairKey so a ref-only collapse is impossible.
    let cell = inner_join_on_pair_key(&omni, &zsm)?;
    eprintln!(
        "  joined_cell: {} rows × {} cols",
        cell.num_rows(),
        cell.num_columns()
    );

    // (b) per-source inner join × src on (run_id, image_basename). First add
    //     image_basename to the cell table from image_path. src is deduped to
    //     one row per (run_id, image_basename).
    let cell = add_basename(cell, "image_path", "image_basename")?;
    let src_dedup = dedupe_by(&src, &["run_id", "image_basename"])?;
    eprintln!("  src_dedupe: {} rows", src_dedup.num_rows());
    let joined = inner_join_on_keys(&cell, &src_dedup, &["run_id", "image_basename"])?;
    eprintln!(
        "  final joined: {} rows × {} cols",
        joined.num_rows(),
        joined.num_columns()
    );

    // === STEP E: split per codec + write (with integrity guards) ===
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| AssembleError::Io(format!("mkdir {}: {e}", args.out_dir.display())))?;
    let codecs_in_data = joined.distinct_str("codec")?;
    let codecs_to_write: Vec<String> = if args.codecs.is_empty() {
        codecs_in_data.clone()
    } else {
        args.codecs.clone()
    };
    for codec in &codecs_to_write {
        if !codecs_in_data.contains(codec) {
            eprintln!("  WARN: requested codec {codec:?} not in data");
            continue;
        }
        let codec_col = joined
            .column("codec")
            .ok_or_else(|| AssembleError::Schema("joined table lost 'codec' column".into()))?;
        let rows: Vec<usize> = (0..joined.num_rows())
            .filter(|&i| codec_col.key_at(i) == *codec)
            .collect();
        let sub = joined.take_rows(&rows)?;

        // Integrity gate 1 — leak detector: even if some upstream sidecar
        // smuggled a `*_mock*` column or a human-copied metric, the write
        // fails loudly (Mode A backstop).
        assert_no_leaked_columns(&format!("{codec}_training"), &sub)?;

        // Integrity gate 2 — ref-broadcast detector: if any per-cell metric
        // column collapsed to one value per reference (the Mode-B signature),
        // refuse to write. `ref_basename` is present in the omni passthrough
        // when the upstream carried it; skip the check when it isn't (nothing
        // to group by). The `score_ssim2_gpu` column is the one Mode B
        // corrupted, so it is the canonical column to guard.
        if sub.has_column("ref_basename") {
            for metric_col in ["score_ssim2_gpu", "score_iwssim_gpu"] {
                if sub.has_column(metric_col) {
                    assert_not_constant_per_ref(
                        &format!("{codec}_training"),
                        "ref_basename",
                        metric_col,
                        &sub,
                    )?;
                }
            }
        }

        let out_path = args.out_dir.join(format!("{codec}_training.parquet"));
        parquet_io::write_parquet(&sub, &out_path)?;
        let size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "  {codec}: {} rows → {} ({:.1} MB)",
            sub.num_rows(),
            out_path.display(),
            size as f64 / 1e6
        );
    }

    eprintln!("done");
    Ok(())
}

/// Sync (unless `--no-sync`) + union-by-name every parquet for one sidecar
/// `kind` across all `--runs`. This is STEP A + STEP B of the Python builder
/// folded together (DuckDB's `union_by_name` becomes [`Table::union_by_name`]).
fn load_kind(
    args: &AssembleArgs,
    sync: Option<&r2_sync::R2Sync>,
    kind: &str,
) -> Result<Table, AssembleError> {
    let mut all_tables: Vec<Table> = Vec::new();
    for run in &args.runs {
        let local = args.cache_dir.join(run).join(kind);
        if let Some(s) = sync {
            let prefix = format!("s3://{}/{}/{}/", args.bucket, run, kind);
            eprintln!("  syncing {prefix} → {}", local.display());
            let n = s.sync_prefix(&prefix, &local)?;
            eprintln!("    {n} parquets in cache");
        }
        if !local.exists() {
            return Err(AssembleError::Io(format!(
                "sidecar dir {} does not exist (run with R2 sync, or pre-populate)",
                local.display()
            )));
        }
        for f in r2_sync::list_local_parquets(&local)? {
            all_tables.push(parquet_io::read_parquet(&f)?);
        }
    }
    Table::union_by_name(all_tables)
}

/// Inner-join `left` × `right` on the typed [`key::PairKey`]. Right's per-pair
/// key columns + every other right column (except the duplicated key columns)
/// are appended to the matched left rows.
fn inner_join_on_pair_key(left: &Table, right: &Table) -> Result<Table, AssembleError> {
    key::PairKey::require_columns("inner_join_on_pair_key left", left)?;
    key::PairKey::require_columns("inner_join_on_pair_key right", right)?;
    join_generic(left, right, &CELL_JOIN_KEYS, OMNI_PASSTHROUGH_AS_SELECT)
}

/// Inner-join on an arbitrary key set (the per-source `(run_id,
/// image_basename)` join). The same row-index machinery as the pair-key join,
/// without the PairKey type (this key is NOT a per-pair key, so PairKey would
/// be the wrong type — the per-source join legitimately keys on
/// `image_basename`, a ref-level identifier, and is correct because the SOURCE
/// features genuinely are per-source).
fn inner_join_on_keys(left: &Table, right: &Table, keys: &[&str]) -> Result<Table, AssembleError> {
    for k in keys {
        if !left.has_column(k) {
            return Err(AssembleError::Schema(format!(
                "inner_join_on_keys: left missing key {k:?}"
            )));
        }
        if !right.has_column(k) {
            return Err(AssembleError::Schema(format!(
                "inner_join_on_keys: right missing key {k:?}"
            )));
        }
    }
    join_generic(left, right, keys, false)
}

/// `true` selects only the OMNI_PASSTHROUGH right-columns; `false` selects
/// every right column. Named for readability at the call site.
const OMNI_PASSTHROUGH_AS_SELECT: bool = true;

/// Shared inner-join machinery. Builds a multi-column string key, indexes the
/// right side (first-wins on dup — but the pair-key join's right side
/// (zensim features) is unique by construction; the per-source join dedupes
/// first), and emits matched left rows with selected right columns appended.
fn join_generic(
    left: &Table,
    right: &Table,
    keys: &[&str],
    omni_select: bool,
) -> Result<Table, AssembleError> {
    use std::collections::HashMap;
    let key_of = |t: &Table, i: usize| -> String {
        keys.iter()
            .map(|k| t.column(k).map(|c| c.key_at(i)).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\u{1f}") // unit separator — unambiguous compound key
    };
    let mut ridx: HashMap<String, usize> = HashMap::with_capacity(right.num_rows());
    for i in 0..right.num_rows() {
        ridx.entry(key_of(right, i)).or_insert(i);
    }

    let mut left_rows: Vec<usize> = Vec::new();
    let mut right_rows: Vec<usize> = Vec::new();
    for i in 0..left.num_rows() {
        if let Some(&ri) = ridx.get(&key_of(left, i)) {
            left_rows.push(i);
            right_rows.push(ri);
        }
    }

    // Start from the selected left rows (all left columns when omni_select is
    // false; the OMNI projection when joining omni × zsm).
    let left_sel = if omni_select {
        project_omni(left, &left_rows)?
    } else {
        left.take_rows(&left_rows)?
    };

    // Append right columns, excluding the join keys (already present on the
    // left) and any name collision (keep left's).
    let mut out = left_sel;
    for name in right.column_names() {
        if keys.contains(&name.as_str()) {
            continue;
        }
        if out.has_column(name) {
            continue; // collision → keep left's (handled by feat_ prefixing upstream)
        }
        let right_col = right.column(name).expect("name from column_names");
        let projected: Column = match right_col {
            Column::Str(v) => Column::Str(right_rows.iter().map(|&ri| v[ri].clone()).collect()),
            Column::F64(v) => Column::F64(right_rows.iter().map(|&ri| v[ri]).collect()),
            Column::I64(v) => Column::I64(right_rows.iter().map(|&ri| v[ri]).collect()),
        };
        out.set_column(name, projected)?;
    }
    Ok(out)
}

/// Project the per-pair key columns + the OMNI_PASSTHROUGH score/stat columns
/// (those present) from `left` over `rows`. Mirrors the explicit SELECT list
/// in build_per_codec_training.py's `joined_cell` query.
fn project_omni(left: &Table, rows: &[usize]) -> Result<Table, AssembleError> {
    let sub = left.take_rows(rows)?;
    let mut keep: Vec<&str> = CELL_JOIN_KEYS.to_vec();
    keep.push("image_basename"); // added later, but tolerate if present
    keep.extend_from_slice(OMNI_PASSTHROUGH);
    let mut out = Table::default();
    for name in sub.column_names() {
        if keep.contains(&name.as_str()) {
            let col = sub.column(name).expect("name");
            out.set_column(name, col.clone())?;
        }
    }
    // Guarantee the per-pair key columns survived (project keeps them by
    // construction, but a missing one would be a silent corruption).
    key::PairKey::require_columns("project_omni", &out)?;
    Ok(out)
}

/// Add a basename column derived from a path column (last `/`-separated
/// component). Rust port of `basename_col`.
fn add_basename(mut t: Table, src_col: &str, dst_col: &str) -> Result<Table, AssembleError> {
    if t.has_column(dst_col) {
        return Ok(t);
    }
    let col = t
        .column(src_col)
        .ok_or_else(|| AssembleError::Schema(format!("add_basename: no column {src_col:?}")))?;
    let Column::Str(paths) = col else {
        return Err(AssembleError::Schema(format!(
            "add_basename: {src_col:?} is not a string column"
        )));
    };
    let basenames: Vec<Option<String>> = paths
        .iter()
        .map(|p| {
            p.as_deref()
                .map(|s| s.rsplit('/').next().unwrap_or(s).to_string())
        })
        .collect();
    t.set_column(dst_col, Column::Str(basenames))?;
    Ok(t)
}

/// Keep one row per distinct `keys` tuple (first-wins), porting the
/// `ROW_NUMBER() … WHERE rn = 1` dedupe in build_per_codec_training.py.
fn dedupe_by(t: &Table, keys: &[&str]) -> Result<Table, AssembleError> {
    use std::collections::HashSet;
    for k in keys {
        if !t.has_column(k) {
            return Err(AssembleError::Schema(format!("dedupe_by: no key {k:?}")));
        }
    }
    let mut seen: HashSet<String> = HashSet::new();
    let mut keep: Vec<usize> = Vec::new();
    for i in 0..t.num_rows() {
        let k: String = keys
            .iter()
            .map(|kc| t.column(kc).map(|c| c.key_at(i)).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\u{1f}");
        if seen.insert(k) {
            keep.push(i);
        }
    }
    t.take_rows(&keep)
}

// The join primitives (`join::safe_join`, `join::attach_positional`,
// `join::assert_not_constant_per_ref`, `join::assert_no_leaked_columns`) are
// the public library API for the canonical-corpus migration and are exercised
// by `tests/assemble_join_safety.rs`. Callers reach them via the `pub mod
// join` path; there is no module-level re-export alias to keep the surface
// minimal (one canonical path per primitive).
