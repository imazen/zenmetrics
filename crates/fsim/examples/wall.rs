//! Wall-time sanity for `fsim_rgb8` on a synthetic pair.
//! `cargo run --release -p fsim --features parallel --example wall [w h reps]`

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let w: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let h: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);

    let mut reference = vec![0u8; w * h * 3];
    let mut distorted = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let v = ((x * 7 + y * 13) ^ (x * y % 251)) as u8;
            reference[i * 3] = v;
            reference[i * 3 + 1] = v.wrapping_add(3);
            reference[i * 3 + 2] = v;
            distorted[i * 3] = v.wrapping_add(9);
            distorted[i * 3 + 1] = v;
            distorted[i * 3 + 2] = v.wrapping_add(5);
        }
    }

    let mut s = fsim::Scores {
        fsim: 0.0,
        fsimc: 0.0,
    };
    let t0 = Instant::now();
    for _ in 0..reps {
        s = fsim::fsim_rgb8(&reference, &distorted, w, h, w * 3).unwrap();
    }
    let dt = t0.elapsed();
    println!(
        "{w}x{h} {reps} reps: {:.1} ms/call (fsim {:.6} fsimc {:.6})",
        dt.as_secs_f64() * 1e3 / reps as f64,
        s.fsim,
        s.fsimc
    );
}
