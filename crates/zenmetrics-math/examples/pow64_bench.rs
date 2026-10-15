// quick throughput check: vpow64_libm_into vs scalar libm::pow
fn main() {
    let n = 1usize << 20;
    let mut xs = vec![0.0f64; n];
    let mut st = 0x243f6a8885a308d3u64;
    for i in 0..n {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        xs[i] = 0.5 + (st >> 11) as f64 / (1u64 << 53) as f64;
    }
    let mut out = vec![0.0f64; n];
    let t = std::time::Instant::now();
    for _ in 0..20 {
        zenmetrics_math::vpow64_libm_into(&xs, &mut out, 0.03);
    }
    let lane = t.elapsed();
    let t = std::time::Instant::now();
    for _ in 0..20 {
        for (x, o) in xs.iter().zip(out.iter_mut()) {
            *o = libm::pow(*x, 0.03);
        }
    }
    let scalar = t.elapsed();
    println!(
        "lane: {:?} ({:.2} ns/elem)",
        lane,
        lane.as_nanos() as f64 / (20.0 * n as f64)
    );
    println!(
        "scalar: {:?} ({:.2} ns/elem)",
        scalar,
        scalar.as_nanos() as f64 / (20.0 * n as f64)
    );
}
