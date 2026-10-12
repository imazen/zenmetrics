//! Native readback for benchmark adapter controls, including Windows ARM64.
#![forbid(unsafe_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args_os()
        .nth(1)
        .ok_or("expected ledger directory")?;
    let mut rows = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "parquet") {
            rows.extend(zenfleet_ledger::read_ledger(&path)?);
        }
    }
    println!("{}", serde_json::to_string(&rows)?);
    Ok(())
}
