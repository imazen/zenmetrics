//! GPU memory measurement driver for ssim2-gpu.
//! See `butteraugli-gpu/examples/mem_one_size.rs` for the protocol.

#![cfg(feature = "cuda")]

use std::io::Write;
use std::time::{Duration, Instant};

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use ssim2_gpu::Ssim2;

const CHILD_HOLD_MS: u64 = 400;

fn synth_srgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    use std::num::Wrapping;
    let n = (w as usize) * (h as usize) * 3;
    let mut v = Vec::with_capacity(n);
    let mut s = Wrapping(seed);
    for _ in 0..n {
        s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
        v.push(((s.0 >> 16) & 0xff) as u8);
    }
    v
}

fn main() {
    let mode = std::env::var("WORKER_MODE").unwrap_or_else(|_| "full".into());
    let w: u32 = std::env::var("WORKER_W").unwrap_or_else(|_| "1024".into()).parse().unwrap();
    let h: u32 = std::env::var("WORKER_H").unwrap_or_else(|_| "1024".into()).parse().unwrap();

    let r = synth_srgb(w, h, 42);
    let d = synth_srgb(w, h, 137);

    let client = Backend::client(&Default::default());
    let t0 = Instant::now();
    let score: f64 = match mode.as_str() {
        "full" => {
            let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
            s.set_reference(&r).expect("set_reference");
            let res = s.compute_with_reference(&d).expect("compute_with_reference");
            res.score
        }
        "strip" => {
            let mut s = Ssim2::<Backend>::new_strip(client, w, h, 256).expect("Ssim2::new_strip");
            let res = s.compute_stripped(&r, &d).expect("compute_stripped");
            res.score
        }
        other => panic!("unknown WORKER_MODE: {other}"),
    };
    let warm_dt = t0.elapsed();

    println!("READY {score:.6} warm_ms={:.2}", warm_dt.as_secs_f64() * 1e3);
    std::io::stdout().flush().expect("flush stdout");

    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
}
