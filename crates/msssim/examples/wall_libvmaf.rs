//! Wall-time sanity for the libvmaf float_ssim/float_ms_ssim ports.
//! `cargo run --release -p msssim --features parallel --example wall_libvmaf [w h reps]`

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let w: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let h: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);

    let reference: Vec<u16> = (0..w * h)
        .map(|i| (((i % w) * 7 + (i / w) * 13) ^ (i % 251)) as u16 % 255)
        .collect();
    let distorted: Vec<u16> = reference.iter().map(|&v| (v + 9) % 255).collect();

    let mut s = 0.0f64;
    let t0 = Instant::now();
    for _ in 0..reps {
        s = msssim::float_ssim(&reference, &distorted, w, h, w, 8).unwrap();
    }
    let dt = t0.elapsed();
    println!(
        "{w}x{h} {reps} reps float_ssim: {:.1} ms/call (score {s:.6})",
        dt.as_secs_f64() * 1e3 / reps as f64
    );

    let t0 = Instant::now();
    for _ in 0..reps {
        s = msssim::float_ms_ssim(&reference, &distorted, w, h, w, 8).unwrap();
    }
    let dt = t0.elapsed();
    println!(
        "{w}x{h} {reps} reps float_ms_ssim: {:.1} ms/call (score {s:.6})",
        dt.as_secs_f64() * 1e3 / reps as f64
    );
}
