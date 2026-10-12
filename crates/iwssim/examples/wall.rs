//! Wall-time sanity for `Iwssim::score` on a synthetic pair.
//! `cargo run --release -p iwssim --features parallel --example wall [w h reps]`

use std::time::Instant;

use iwssim::Iwssim;

fn main() {
    let mut args = std::env::args().skip(1);
    let w: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let h: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let n = w as usize * h as usize;

    let reference: Vec<f32> = (0..n)
        .map(|i| {
            (((i % w as usize) * 7 + (i / w as usize) * 13) ^ (i % 251)) as f32 % 255.0 / 255.0
        })
        .collect();
    let distorted: Vec<f32> = reference
        .iter()
        .map(|&v| (v * 255.0 + 9.0) % 255.0 / 255.0)
        .collect();

    let mut m = Iwssim::new(w, h).unwrap();
    let mut score = 0.0f64;
    let t0 = Instant::now();
    for _ in 0..reps {
        score = m.score_gray(&reference, &distorted).unwrap().score as f64;
    }
    let dt = t0.elapsed();
    println!(
        "{w}x{h} {reps} reps: {:.2} ms/call (score {score:.17}, bits {:016x})",
        dt.as_secs_f64() * 1e3 / reps as f64,
        score.to_bits()
    );
}
