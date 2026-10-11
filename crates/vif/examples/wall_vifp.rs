//! Wall-time sanity for `vif_plane_f32` (VIFp) on a synthetic pair.
//! `cargo run --release -p vif --features parallel --example wall_vifp [w h reps]`

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let w: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let h: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);

    let mut reference = vec![0f32; w * h];
    let mut distorted = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            reference[i] = (((x * 7 + y * 13) ^ (x * y % 251)) % 256) as f32;
            distorted[i] = (reference[i] + 9.0).min(255.0);
        }
    }

    let mut score = 0.0f64;
    let t0 = Instant::now();
    for _ in 0..reps {
        score = vif::vif_plane_f32(&reference, &distorted, w, h, w).unwrap();
    }
    let dt = t0.elapsed();
    println!(
        "{w}x{h} {reps} reps: {:.1} ms/call (score {score:.6})",
        dt.as_secs_f64() * 1e3 / reps as f64
    );
}
