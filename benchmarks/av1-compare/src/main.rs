mod fleet;
mod measure;
mod pixels;
mod verify;
// One process, all five encoder implementations. Reads a request on stdin;
// emits one provenance/timing row after writing and independently checking OBU.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    time::Instant,
};
use zenmetrics_av1_compare::{Config, decode_planar, encode};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    config: Config,
    input: String,
    output: String,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command = std::env::args().nth(1);
    if command.as_deref() == Some("capabilities") {
        println!("av1-api-planar-v3");
        return Ok(());
    }
    let mut json = String::new();
    std::io::stdin().take(65537).read_to_string(&mut json)?;
    if json.len() > 65536 {
        return Err("request too large".into());
    }
    if command.as_deref() == Some("verify-measurement") {
        return verify::run(serde_json::from_str(&json)?);
    }
    if std::env::args().nth(1).as_deref() == Some("measure") {
        return measure::run(serde_json::from_str(&json)?, true);
    }
    if command.as_deref() == Some("declare") {
        return fleet::declare(&json);
    }
    if command.as_deref() == Some("jobexec")
        || serde_json::from_str::<serde_json::Value>(&json)?
            .get("kind")
            .is_some()
    {
        return fleet::execute(&json);
    }
    let req: Request = serde_json::from_str(&json)?;
    let pixels = std::fs::read(&req.input)?;
    req.config.validate(&pixels)?;
    if std::path::Path::new(&req.output).exists() {
        return Err("output already exists".into());
    }
    let source_sha = format!("{:x}", Sha256::digest(&pixels));
    let binary_sha = format!(
        "{:x}",
        Sha256::digest(std::fs::read(std::env::current_exe()?)?)
    );
    let start = Instant::now();
    let obu = encode(req.config, &pixels)?;
    let elapsed_ns = start.elapsed().as_nanos();
    decode_planar(&obu, req.config)?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&req.output)?
        .write_all(&obu)?;
    println!(
        "{}",
        serde_json::json!({"protocol":"av1-api-planar-v3", "config":req.config,
        "revision": req.config.revision(), "svt_reference": req.config.resolved_svt_reference(), "binary_sha256":binary_sha, "input_sha256": source_sha,
        "output_sha256": format!("{:x}", Sha256::digest(&obu)), "bytes":obu.len(),
        "api_elapsed_ns":elapsed_ns, "timing_scope":"fresh-lifecycle-including-plane-preparation",
        "decode_checked":"libaom", "c_fp_contract":"off", "aom_sb_size":64})
    );
    Ok(())
}
