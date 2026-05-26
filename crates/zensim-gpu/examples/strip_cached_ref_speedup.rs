//! Task #75 perf check: measure the speedup from the device-cached
//! ref XYB pyramid vs the per-strip host-cached rebuild path in
//! Mode E (strip + cached-ref) zensim.
//!
//! Build:
//!   cargo build --release -p zensim-gpu --features cuda --example strip_cached_ref_speedup
//!
//! Run:
//!   ./target/release/examples/strip_cached_ref_speedup
//!
//! Override resolution / iter count:
//!   W=4096 H=4096 H_BODY=256 N_ITER=10 ./strip_cached_ref_speedup
//!
//! Output is informational — the values pinned in
//! `docs/STRIP_PROCESSING.md` come from this example.

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use std::time::Instant;
use zensim_gpu::Zensim;

fn main() {
    let n_iter: usize = std::env::var("N_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let w: u32 = std::env::var("W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let h: u32 = std::env::var("H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let h_body: u32 = std::env::var("H_BODY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    let client = CudaRuntime::client(&Default::default());

    let n = (w as usize) * (h as usize) * 3;
    let ref_bytes: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let dist_bytes: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();
    let mp = (w as f64 * h as f64) / 1e6;

    eprintln!(
        "zensim strip cached-ref speedup ({}x{}, {:.2} MP, h_body={}, n_iter={})",
        w, h, mp, h_body, n_iter
    );
    eprintln!("--------------------------------------------------------------------");

    // Path A: host-cached only (pre-task-#75 baseline).
    {
        let mut z = Zensim::<CudaRuntime>::new_strip(client.clone(), w, h, h_body)
            .expect("new_strip");
        z.set_reference_host_cached_only(&ref_bytes)
            .expect("set_reference_host_cached_only");
        // 2 warmup iters not counted.
        for _ in 0..2 {
            let _ = z.compute_with_reference(&dist_bytes).expect("warmup");
        }
        let mut total_ms = 0.0_f64;
        for i in 0..n_iter {
            let t = Instant::now();
            let _ = z
                .compute_with_reference(&dist_bytes)
                .expect("compute_with_reference");
            let dt = t.elapsed().as_secs_f64() * 1000.0;
            total_ms += dt;
            eprintln!("[host-cached] iter {i}: {:>7.2} ms", dt);
        }
        let mean_ms = total_ms / (n_iter as f64);
        eprintln!(
            "[host-cached] MEAN: {mean_ms:>7.2} ms / iter  =  {:>7.2} MP/s",
            mp / (mean_ms / 1000.0)
        );
    }

    // Path B: device-cached (task #75).
    {
        let mut z = Zensim::<CudaRuntime>::new_strip(client.clone(), w, h, h_body)
            .expect("new_strip");
        z.set_reference(&ref_bytes).expect("set_reference");
        for _ in 0..2 {
            let _ = z.compute_with_reference(&dist_bytes).expect("warmup");
        }
        let mut total_ms = 0.0_f64;
        for i in 0..n_iter {
            let t = Instant::now();
            let _ = z
                .compute_with_reference(&dist_bytes)
                .expect("compute_with_reference");
            let dt = t.elapsed().as_secs_f64() * 1000.0;
            total_ms += dt;
            eprintln!("[device-cached] iter {i}: {:>7.2} ms", dt);
        }
        let mean_ms = total_ms / (n_iter as f64);
        eprintln!(
            "[device-cached] MEAN: {mean_ms:>7.2} ms / iter  =  {:>7.2} MP/s",
            mp / (mean_ms / 1000.0)
        );
    }
}
