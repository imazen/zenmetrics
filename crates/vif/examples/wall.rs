//! Wall-time sanity for `vifvec_plane_f32` on a synthetic pair.
//! `cargo run --release -p vif --features parallel --example wall [w h reps]`

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let w: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let h: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);

    let mut reference = vec![0.0f32; w * h];
    let mut distorted = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let v = ((x * 7 + y * 13) ^ (x * y % 251)) as f32;
            reference[i] = v;
            distorted[i] = v + 9.0;
        }
    }

    let mut score = 0.0f64;
    let t0 = Instant::now();
    for _ in 0..reps {
        score = vif::vifvec_plane_f32(&reference, &distorted, w, h, w).unwrap();
    }
    let dt = t0.elapsed();
    println!(
        "{w}x{h} {reps} reps: {:.1} ms/call (score {score:.9})",
        dt.as_secs_f64() * 1e3 / reps as f64
    );
}
