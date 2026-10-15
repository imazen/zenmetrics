//! Standalone raw-RGB CW-SSIM driver (no image decoder in the measured path).
#[path = "../../crates/zenmetrics-cli/src/metrics/cw_fft.rs"]
mod cw_fft;
#[path = "../../crates/zenmetrics-cli/src/metrics/cw_ssim.rs"]
mod cw_ssim;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let w: usize = args[1].parse().unwrap();
    let h: usize = args[2].parse().unwrap();
    let r = std::fs::read(&args[3]).unwrap();
    let d = std::fs::read(&args[4]).unwrap();
    println!(
        "{:.17}",
        cw_ssim::score(&r, &d, w, h, w * 3, w * 3).unwrap()
    );
}
