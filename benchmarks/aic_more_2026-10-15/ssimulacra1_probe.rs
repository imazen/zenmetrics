#[path = "../../crates/zenmetrics-cli/src/metrics/ssimulacra1.rs"]
mod ssimulacra1;
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let r = std::fs::read(&a[3]).unwrap();
    let d = std::fs::read(&a[4]).unwrap();
    println!(
        "{:.17}",
        ssimulacra1::score(&r, &d, w, h, w * 3, w * 3).unwrap()
    );
}
