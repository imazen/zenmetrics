#[path = "../../crates/zenmetrics-cli/src/metrics/butteraugli_original.rs"]
mod butteraugli_original;
#[path = "../../crates/zenmetrics-cli/src/metrics/ssimulacra1.rs"]
mod ssimulacra1;
#[path = "../../crates/zenmetrics-cli/src/metrics/ws_hv.rs"]
mod ws_hv;
fn main() {
    for (w, h) in [(64, 64), (37, 41)] {
        let r: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    (0..3).map(move |c| ((x * 13 + y * 17 + c * 71 + (x * y) % 29) % 256) as u8)
                })
            })
            .collect();
        let mut d = r.clone();
        for p in d.iter_mut().step_by(17) {
            *p ^= 63;
        }
        let b = butteraugli_original::error_map(&r, &d, w, h, w * 3, w * 3)
            .unwrap()
            .into_iter()
            .fold(0.0_f32, f32::max);
        println!(
            "{w} {h} butter={b:.17} ssim1={:.17} ws={:.17}",
            ssimulacra1::score(&r, &d, w, h, w * 3, w * 3).unwrap(),
            ws_hv::score(&r, &d, w, h, w * 3, w * 3).unwrap()
        );
    }
}
