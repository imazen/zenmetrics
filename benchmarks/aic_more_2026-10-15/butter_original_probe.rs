#[path = "../../crates/zenmetrics-cli/src/metrics/butteraugli_original.rs"]
mod butteraugli_original;
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let r = std::fs::read(&a[3]).unwrap();
    let d = std::fs::read(&a[4]).unwrap();
    let map = butteraugli_original::error_map(&r, &d, w, h, w * 3, w * 3).unwrap();
    println!("{:.17}", map.into_iter().fold(0.0f32, f32::max));
}
