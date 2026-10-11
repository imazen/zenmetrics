//! Wall-time sanity for `mad_rgb8` at non-power-of-two dims (the Bluestein
//! path every real image takes).
//! `cargo run --release -p mad-iqa --features parallel --example wall [w h reps]`

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let w: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let h: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(767);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);

    let mut reference = vec![0u8; w * h * 3];
    let mut distorted = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let r = ((x * 7 + y * 13) ^ (x * y % 251)) as u8;
            let g = ((x * 11 + y * 3) % 253) as u8;
            let b = ((x * 5 + y * 17) % 251) as u8;
            reference[i * 3] = r;
            reference[i * 3 + 1] = g;
            reference[i * 3 + 2] = b;
            distorted[i * 3] = r.wrapping_add(9);
            distorted[i * 3 + 1] = g.wrapping_sub(7);
            distorted[i * 3 + 2] = b.wrapping_add(4);
        }
    }

    let mut score = (0.0f64, 0.0f64);
    let t0 = Instant::now();
    for _ in 0..reps {
        let s = mad_iqa::mad_rgb8(&reference, &distorted, w, h, w * 3).unwrap();
        score = (s.hi, s.lo);
    }
    let dt = t0.elapsed();
    println!(
        "{w}x{h} {reps} reps: {:.1} ms/call (hi {:.4} lo {:.4})",
        dt.as_secs_f64() * 1e3 / reps as f64,
        score.0,
        score.1
    );
}
