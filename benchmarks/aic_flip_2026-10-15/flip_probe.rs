//! Standalone raw-map oracle driver: rustc --edition 2024 -O flip_probe.rs.
#[path = "../../crates/zenmetrics-cli/src/metrics/flip.rs"]
mod flip;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let w: usize = args[1].parse().unwrap();
    let h: usize = args[2].parse().unwrap();
    let r = std::fs::read(&args[3]).unwrap();
    let d = std::fs::read(&args[4]).unwrap();
    let map = flip::error_map(&r, &d, w, h, w * 3, w * 3).unwrap();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    use std::io::Write;
    out.write_all(&flip::mean(&map).to_le_bytes()).unwrap();
    for v in map {
        out.write_all(&v.to_le_bytes()).unwrap();
    }
}
