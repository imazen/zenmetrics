//! Untimed reconstruction replay of saved SVT measurement cells.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
};
use zenmetrics_av1_compare::{Backend, Config, verify_svt_reconstruction};

type Error = Box<dyn std::error::Error>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Verification {
    measurement_dir: String,
    output: String,
}
#[derive(Deserialize)]
struct Row {
    config: Config,
    source_sha256: String,
    input_sha256: String,
    output_sha256: String,
    binary_sha256: String,
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}

pub fn run(req: Verification) -> Result<(), Error> {
    let root = Path::new(&req.measurement_dir);
    let rows = BufReader::new(fs::File::open(root.join("rows.jsonl"))?);
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&req.output)?;
    let verifier = sha(&fs::read(std::env::current_exe()?)?);
    let mut seen = HashSet::new();
    let mut checked = 0;
    for line in rows.lines() {
        let row: Row = serde_json::from_str(&line?)?;
        let cfg = row.config;
        if !matches!(cfg.backend, Backend::Svt) {
            continue;
        }
        cfg.validate_configuration()?;
        if ![
            &row.source_sha256,
            &row.input_sha256,
            &row.output_sha256,
            &row.binary_sha256,
        ]
        .into_iter()
        .all(|h| valid_hash(h))
        {
            return Err("invalid measurement hash".into());
        }
        let key = serde_json::to_string(&(
            cfg,
            &row.input_sha256,
            &row.output_sha256,
            &row.binary_sha256,
        ))?;
        if !seen.insert(key) {
            continue;
        }
        let reference = image::open(root.join(format!(
            "{}-{}x{}-reference.png",
            row.source_sha256, cfg.width, cfg.height
        )))?
        .to_rgb8();
        if reference.dimensions() != (cfg.width, cfg.height) {
            return Err("reference dimensions differ from measured cell".into());
        }
        let pixels = crate::pixels::prepare(reference.as_raw(), cfg)?;
        if sha(&pixels) != row.input_sha256 {
            return Err("reprepared input differs from measured bytes".into());
        }
        let obu = fs::read(root.join("obu").join(format!("{}.obu", row.output_sha256)))?;
        if sha(&obu) != row.output_sha256 {
            return Err("measured OBU hash mismatch".into());
        }
        let result = verify_svt_reconstruction(cfg, &pixels, &obu);
        let record = serde_json::json!({
            "protocol": "svt-measured-reconstruction-v1", "config": cfg,
            "source_sha256": row.source_sha256, "input_sha256": row.input_sha256,
            "output_sha256": row.output_sha256, "measured_binary_sha256": row.binary_sha256,
            "verifier_binary_sha256": verifier, "ok": result.is_ok(), "error": result.as_ref().err(),
            "decoder": "libaom", "verification": "untimed replay; exact measured OBU and all reconstructed samples"
        });
        writeln!(output, "{record}")?;
        output.flush()?;
        result?;
        checked += 1;
    }
    if checked == 0 {
        return Err("no Rust SVT cells were verified".into());
    }
    println!(
        "{}",
        serde_json::json!({"verified_svt_cells": checked, "output": req.output})
    );
    Ok(())
}
