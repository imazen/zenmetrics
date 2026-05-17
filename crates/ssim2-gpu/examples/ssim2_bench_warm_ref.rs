//! ssim2 warm-reference bench: measures the production cached-reference path
//! (`set_reference` once + `compute_with_reference` per call). This is the
//! hot path for encoder rate-distortion sweeps, and is the one the Fukushima
//! T_y.A skip-map should accelerate the most.

use std::time::Instant;
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use ssim2_gpu::Ssim2;

fn main() {
    let w: u32 = 4000;
    let h: u32 = 3000;
    let n = (w as usize) * (h as usize) * 3;
    let r: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let d: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();
    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, w, h).expect("new");
    s.set_reference(&r).expect("ref");
    // warmup
    for _ in 0..2 { let _ = s.compute_with_reference(&d).expect("warmup"); }
    eprintln!("ssim2 12 MP warm-ref timing:");
    for i in 0..5 {
        let t = Instant::now();
        let res = s.compute_with_reference(&d).expect("compute_with_ref");
        eprintln!("  iter {i}: {:?}  score={:.6}", t.elapsed(), res.score);
    }
}
