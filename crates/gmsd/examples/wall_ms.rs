//! Wall-clock probe for `ms_gmsd_rgb8`/`ms_gmsdc_rgb8` on a synthetic
//! WxH RGB8 pair. Usage: `wall_ms [w] [h] [reps]`.
fn main() {
    let mut args = std::env::args().skip(1);
    let w: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let h: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut r = vec![0u8; w * h * 3];
    let mut d = vec![0u8; w * h * 3];
    let mut s = 12345u32;
    for i in 0..r.len() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        r[i] = (s >> 24) as u8;
        d[i] = r[i].wrapping_add((s >> 30) as u8);
    }
    let mut times = Vec::with_capacity(reps);
    let mut score = 0.0;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        score =
            std::hint::black_box(gmsd::ms_gmsd_rgb8(&r, &d, w, h, w * 3).unwrap());
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{w}x{h} {reps} reps ms_gmsd: {:.1} ms/call median (score {score:.6})",
        times[reps / 2]
    );
}
