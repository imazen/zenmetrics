//! Where does `gmsd_rgb8`'s time go? Times the sRGB8 -> gray conversion and
//! the gray kernel separately (median of 31 calls, one size per argument).
//! Diagnostic only: the owner of published speed numbers is zensim's
//! `ssim2_speed_bar`.
fn main() {
    for arg in std::env::args().skip(1) {
        let n: usize = arg.parse().expect("size");
        let mut src = vec![0u8; n * n * 3];
        let mut dst = vec![0u8; n * n * 3];
        let mut s = 12345u32;
        for i in 0..src.len() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            src[i] = (s >> 24) as u8;
            dst[i] = src[i].wrapping_add((s >> 30) as u8);
        }
        let med = |f: &mut dyn FnMut()| {
            let mut t: Vec<f64> = (0..31)
                .map(|_| {
                    let t0 = std::time::Instant::now();
                    f();
                    t0.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[15]
        };
        let mut g1 = vec![0.0f32; n * n];
        let mut g2 = vec![0.0f32; n * n];
        let conv = med(&mut || {
            gmsd::rgb8_to_gray(&src, n, n, n * 3, &mut g1).unwrap();
            gmsd::rgb8_to_gray(&dst, n, n, n * 3, &mut g2).unwrap();
        });
        let kern = med(&mut || {
            std::hint::black_box(
                gmsd::gmsd(
                    gmsd::GrayImage::packed(&g1, n, n).unwrap(),
                    gmsd::GrayImage::packed(&g2, n, n).unwrap(),
                )
                .unwrap(),
            );
        });
        let whole = med(&mut || {
            std::hint::black_box(gmsd::gmsd_rgb8(&src, &dst, n, n, n * 3).unwrap());
        });
        println!(
            "{n}: convert(2 images) {conv:.3} ms  kernel {kern:.3} ms  gmsd_rgb8 {whole:.3} ms"
        );
    }
}
