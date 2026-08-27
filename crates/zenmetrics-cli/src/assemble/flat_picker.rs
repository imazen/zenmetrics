#![forbid(unsafe_code)]

//! `--mode flat-picker`: turn a FLAT encode_sha-keyed scores parquet (the
//! LAN-era harvest shape: one row per encode, metric columns inline) into a
//! picker-SHAPE parquet the result browser's rollup can consume, with the
//! same typed-join safety the per-codec mode has:
//!
//! - every input is keyed on `encode_sha`; a DUPLICATE key in any input is a
//!   hard error (never a silent fan-out),
//! - encoded bytes come from a `sha\tbytes` TSV (blob listing) and every
//!   scores row MUST find its bytes (missing = hard error unless
//!   `--allow-missing-bytes`, which records NaN and prints the count),
//! - extra score sidecars (e.g. a model-fill like `zensim_c944_sidecar`)
//!   left-join with a printed miss count,
//! - identity columns are DERIVED, never guessed: `origin_id`/`width`/
//!   `height` parse from the strict `<origin>.scale<W>x<H>.png` rendition
//!   name (any other shape = error), `cell`/`knob_plan`/`fp` parse from
//!   `knob_tuple_json`.
//!
//! Registered consumer: the coefficient viewer rollup (BROWSER lane,
//! zensim plan doc 2026-08-27).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use super::parquet_io::{read_parquet, write_parquet};
use super::table::{AssembleError, Column, Table};

fn err(msg: String) -> AssembleError {
    AssembleError::Schema(msg)
}

fn str_col<'t>(t: &'t Table, name: &str, what: &str) -> Result<Vec<Option<String>>, AssembleError> {
    match t.column(name) {
        Some(Column::Str(v)) => Ok(v.clone()),
        Some(Column::I64(v)) => Ok(v.iter().map(|x| Some(x.to_string())).collect()),
        Some(Column::F64(v)) => Ok(v.iter().map(|x| Some(x.to_string())).collect()),
        None => Err(err(format!("{what}: missing column `{name}`"))),
    }
}

fn assert_unique(keys: &[Option<String>], what: &str) -> Result<(), AssembleError> {
    let mut seen: HashMap<&str, usize> = HashMap::with_capacity(keys.len());
    for (i, k) in keys.iter().enumerate() {
        let k = k
            .as_deref()
            .ok_or_else(|| err(format!("{what}: null encode_sha at row {i}")))?;
        if let Some(prev) = seen.insert(k, i) {
            return Err(err(format!(
                "{what}: DUPLICATE encode_sha `{k}` at rows {prev} and {i} — refusing the join"
            )));
        }
    }
    Ok(())
}

/// `<origin>.scale<W>x<H>.png` -> (origin, w, h). Strict: anything else errors.
fn parse_rendition(p: &str) -> Result<(String, i64, i64), AssembleError> {
    let base = p.rsplit('/').next().unwrap_or(p);
    let rest = base
        .strip_suffix(".png")
        .ok_or_else(|| err(format!("rendition `{base}`: no .png suffix")))?;
    let (origin, scale) = rest
        .split_once(".scale")
        .ok_or_else(|| err(format!("rendition `{base}`: no `.scale` token")))?;
    let (w, h) = scale
        .split_once('x')
        .ok_or_else(|| err(format!("rendition `{base}`: scale token not WxH")))?;
    let wi: i64 = w
        .parse()
        .map_err(|_| err(format!("rendition `{base}`: bad width")))?;
    let hi: i64 = h
        .parse()
        .map_err(|_| err(format!("rendition `{base}`: bad height")))?;
    Ok((origin.to_string(), wi, hi))
}

pub fn run_flat_picker(
    scores: &Path,
    sizes_tsv: &Path,
    extra_scores: &[std::path::PathBuf],
    rename: &[String],
    meta: &[String],
    enc_mode: &str,
    allow_missing_bytes: bool,
    out: &Path,
) -> Result<(), AssembleError> {
    let t = read_parquet(scores)?;
    let n = t.num_rows();
    // Row identity in the scores table is the CELL, not the sha: content
    // addressing legitimately maps many cells onto one byte-identical encode
    // (e.g. q-extremes on flat renditions), so `encode_sha` here is a
    // many-to-one JOIN KEY to attributes that are functions of the bytes
    // (size, model scores). Uniqueness IS enforced on the sha-keyed side
    // (sizes tsv, extra sidecars) below.
    let shas = str_col(&t, "encode_sha", "scores")?;
    let distinct: std::collections::HashSet<&str> =
        shas.iter().filter_map(|s| s.as_deref()).collect();
    if distinct.len() < n {
        eprintln!(
            "[flat-picker] note: {} cells share {} distinct encode_shas (byte-identical encodes; expected under content addressing)",
            n,
            distinct.len()
        );
    }
    if shas.iter().any(|s| s.is_none()) {
        return Err(err("scores: null encode_sha".into()));
    }

    // bytes from the blob listing
    let sizes_raw = std::fs::read_to_string(sizes_tsv)
        .map_err(|e| err(format!("sizes tsv {}: {e}", sizes_tsv.display())))?;
    let mut sizes: HashMap<&str, f64> = HashMap::new();
    for (ln, line) in sizes_raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (k, v) = line
            .split_once('\t')
            .ok_or_else(|| err(format!("sizes tsv line {}: not `sha\\tbytes`", ln + 1)))?;
        let b: f64 = v
            .trim()
            .parse()
            .map_err(|_| err(format!("sizes tsv line {}: bad bytes", ln + 1)))?;
        if sizes.insert(k.trim(), b).is_some() {
            return Err(err(format!("sizes tsv: DUPLICATE sha `{}`", k.trim())));
        }
    }
    let mut bytes = Vec::with_capacity(n);
    let mut missing_bytes = 0usize;
    for k in &shas {
        match sizes.get(k.as_deref().unwrap()) {
            Some(b) => bytes.push(*b),
            None => {
                missing_bytes += 1;
                bytes.push(f64::NAN);
            }
        }
    }
    if missing_bytes > 0 {
        let msg = format!("{missing_bytes}/{n} scores rows have NO bytes in the sizes tsv");
        if allow_missing_bytes {
            eprintln!("[flat-picker] WARNING {msg} (allowed by flag; NaN recorded)");
        } else {
            return Err(err(format!("{msg} — pass --allow-missing-bytes to accept")));
        }
    }

    // identity derivation
    let paths = str_col(&t, "image_path", "scores")?;
    let knobs = str_col(&t, "knob_tuple_json", "scores")?;
    let mut origin = Vec::with_capacity(n);
    let mut width = Vec::with_capacity(n);
    let mut height = Vec::with_capacity(n);
    let mut cell = Vec::with_capacity(n);
    let mut plan = Vec::with_capacity(n);
    let mut fp = Vec::with_capacity(n);
    for i in 0..n {
        let (o, w, h) = parse_rendition(paths[i].as_deref().unwrap_or(""))?;
        origin.push(Some(o));
        width.push(w);
        height.push(h);
        let kj = knobs[i].as_deref().unwrap_or("{}");
        let v: serde_json::Value = serde_json::from_str(kj)
            .map_err(|e| err(format!("row {i}: knob_tuple_json parse: {e}")))?;
        let g = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
        cell.push(g("cell"));
        plan.push(g("plan"));
        fp.push(g("fp"));
    }

    let mut out_t = t;
    out_t.set_column("origin_id", Column::Str(origin))?;
    out_t.set_column("width", Column::I64(width))?;
    out_t.set_column("height", Column::I64(height))?;
    out_t.set_column("cell", Column::Str(cell))?;
    out_t.set_column("knob_plan", Column::Str(plan))?;
    out_t.set_column("fp", Column::Str(fp))?;
    out_t.set_column("encoded_bytes", Column::F64(bytes))?;
    out_t.set_column("mode", Column::Str(vec![Some(enc_mode.to_string()); n]))?;
    // browser sidecar-join stand-in: content address IS the filename identity
    out_t.set_column("encoded_filename", Column::Str(shas.clone()))?;

    // extra encode_sha-keyed score sidecars (left join, loud miss count)
    for p in extra_scores {
        let s = read_parquet(p)?;
        let sk = str_col(&s, "encode_sha", "extra-scores")?;
        assert_unique(&sk, &format!("extra-scores {}", p.display()))?;
        let mut idx: HashMap<&str, usize> = HashMap::with_capacity(sk.len());
        for (i, k) in sk.iter().enumerate() {
            idx.insert(k.as_deref().unwrap(), i);
        }
        for cname in s.column_names().to_vec() {
            if cname == "encode_sha" {
                continue;
            }
            let src = match s.column(&cname) {
                Some(Column::F64(v)) => v.clone(),
                _ => {
                    return Err(err(format!(
                        "extra-scores {}: `{cname}` must be f64",
                        p.display()
                    )));
                }
            };
            let mut joined = Vec::with_capacity(n);
            let mut miss = 0usize;
            for k in &shas {
                match idx.get(k.as_deref().unwrap()) {
                    Some(&i) => joined.push(src[i]),
                    None => {
                        miss += 1;
                        joined.push(f64::NAN);
                    }
                }
            }
            eprintln!("[flat-picker] extra `{cname}`: {miss}/{n} rows unmatched (NaN)");
            out_t.set_column(&cname, Column::F64(joined))?;
        }
    }

    // renames LAST (so extra columns can be renamed too): `from=to`, repeatable
    let mut renames: BTreeMap<String, String> = BTreeMap::new();
    for r in rename {
        let (from, to) = r
            .split_once('=')
            .ok_or_else(|| err(format!("--rename `{r}`: expected from=to")))?;
        renames.insert(from.to_string(), to.to_string());
    }
    if !renames.is_empty() {
        out_t = out_t.rename_columns(&renames)?;
    }
    // constant metadata columns: `key=value`, repeatable
    for m in meta {
        let (k, v) = m
            .split_once('=')
            .ok_or_else(|| err(format!("--meta `{m}`: expected key=value")))?;
        out_t.set_column(k, Column::Str(vec![Some(v.to_string()); n]))?;
    }

    write_parquet(&out_t, out)?;
    println!(
        "flat-picker: {n} rows x {} cols -> {}",
        out_t.num_columns(),
        out.display()
    );
    Ok(())
}
