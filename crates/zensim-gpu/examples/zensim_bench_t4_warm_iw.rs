//! 12 MP zensim WithIw (372-feature) warm-reference bench. Used to
//! validate the perf impact of the strip-local masked + IW kernel
//! (`kernels::masked_iw_strip`) vs the pre-strip-local image-wide
//! variant in `kernels::masked_iw` (which is no longer used in the
//! production extended path but still compiled for reference).
//!
//! Build:
//!   cargo build --release -p zensim-gpu --features cuda --no-default-features \
//!     --example zensim_bench_t4_warm_iw
//!
//! Override iter count: `N_ITER=10 ./zensim_bench_t4_warm_iw`
//! Override resolution: `W=2048 H=1536 ./zensim_bench_t4_warm_iw`

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use std::time::Instant;
use zensim_gpu::{Zensim, ZensimFeatureRegime};

fn main() {
    let n_iter: usize = std::env::var("N_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let w: u32 = std::env::var("W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);
    let h: u32 = std::env::var("H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    let client = CudaRuntime::client(&Default::default());
    let mut z = Zensim::<CudaRuntime>::new_with_regime(client, w, h, ZensimFeatureRegime::WithIw)
        .expect("Zensim::new_with_regime(WithIw)");

    let n = (w as usize) * (h as usize) * 3;
    let ref_bytes: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let dist_bytes: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();

    let mp = (w as f64 * h as f64) / 1e6;
    eprintln!("warming WithIw reference ({}x{}, {:.2} MP) ...", w, h, mp);
    z.set_reference(&ref_bytes).expect("set_reference");

    // 2 warmup iters not counted (build kernels, fault pages, etc.).
    for _ in 0..2 {
        let _ = z.compute_with_reference_vec(&dist_bytes).expect("warmup");
    }

    eprintln!("running {} timed iters (372-feature) ...", n_iter);
    let mut total_ms = 0.0_f64;
    for i in 0..n_iter {
        let t = Instant::now();
        let features = z.compute_with_reference_vec(&dist_bytes).expect("compute");
        let dt = t.elapsed();
        let dt_ms = dt.as_secs_f64() * 1000.0;
        total_ms += dt_ms;
        eprintln!(
            "iter {i}: {:>6.2} ms  n_features={}  ({:.2} MP/s)",
            dt_ms,
            features.len(),
            mp / (dt_ms / 1000.0)
        );
    }
    let mean_ms = total_ms / (n_iter as f64);
    eprintln!(
        "MEAN: {mean_ms:.2} ms / iter  =  {:.2} MP/s",
        mp / (mean_ms / 1000.0)
    );
}
