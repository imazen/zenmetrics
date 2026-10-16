#[path = "../../crates/zenmetrics-cli/src/metrics/ws_hv.rs"]
mod ws_hv;
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let r = std::fs::read(&a[3]).unwrap();
    let d = std::fs::read(&a[4]).unwrap();
    println!(
        "{:.17}\t{:.17}",
        ws_hv::score(&r, &d, w, h, w * 3, w * 3).unwrap(),
        ws_hv::score_order(&r, &d, w, h, w * 3, w * 3, true).unwrap()
    );
}
