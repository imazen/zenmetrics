use std::hint::black_box;
use std::time::Instant;

fn main() {
    let w = 128;
    let h = 128;
    let reference = (0..w * h * 3)
        .map(|i| ((i * 37) % 256) as u8)
        .collect::<Vec<_>>();
    let distorted = (0..w * h * 3)
        .map(|i| ((i * 37 + i % 7) % 256) as u8)
        .collect::<Vec<_>>();
    for _ in 0..10 {
        black_box(nlpd::score_rgb_u8(&reference, &distorted, w, h).unwrap());
    }
    let start = Instant::now();
    for _ in 0..100 {
        black_box(nlpd::score_rgb_u8(&reference, &distorted, w, h).unwrap());
    }
    println!(
        "NLPD Rust 128x128: {:.3} ms/pair",
        start.elapsed().as_secs_f64() * 10.0
    );
}
