//! Minimal harness for nsys profiling: 12 MP cvvdp warm-ref loop.
//! Build: cargo build --release -p cvvdp-gpu --features cuda --no-default-features --example bench_warm_ref
//! Profile: nsys profile -t cuda,nvtx -o /tmp/cvvdp-prof --stats=true ./target/release/examples/bench_warm_ref

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
use std::time::Instant;

fn main() {
    let n_iter: usize = std::env::var("N_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let w: u32 = 4000;
    let h: u32 = 3000;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();

    let client = CudaRuntime::client(&Default::default());
    let mut cvvdp =
        Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");

    let n = (w as usize) * (h as usize) * 3;
    let ref_bytes: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let dist_bytes: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();

    eprintln!(
        "warming reference ({}x{}, {:.2} MP) ...",
        w,
        h,
        (w as f64 * h as f64) / 1e6
    );
    cvvdp.warm_reference(&ref_bytes).expect("warm_reference");

    // 2 warmup iters not counted
    for _ in 0..2 {
        let _ = cvvdp
            .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
            .expect("warmup");
    }

    eprintln!("running {} timed iters ...", n_iter);
    for i in 0..n_iter {
        let t = Instant::now();
        let jod = cvvdp
            .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
            .expect("compute");
        eprintln!("iter {}: {:?}  jod={:.4}", i, t.elapsed(), jod);
    }
}
