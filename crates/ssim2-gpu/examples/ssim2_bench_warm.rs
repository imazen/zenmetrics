//! ssim2-specific bench so multi-agent example builds don't overwrite
//! each other (a `bench_t4_warm` exists in every GPU metric crate; the
//! last `cargo build --example bench_t4_warm` writes to the same path).
//! Use this file when running ssim2-gpu performance comparisons.

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
    for _ in 0..2 { let _ = s.compute(&r, &d).expect("warmup"); }
    eprintln!("ssim2 12 MP timing:");
    for i in 0..5 {
        let t = Instant::now();
        let res = s.compute(&r, &d).expect("compute");
        eprintln!("  iter {i}: {:?}  score={:.6}", t.elapsed(), res.score);
    }
}
