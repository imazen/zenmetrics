//! GPU memory measurement driver for iwssim-gpu.
//! See `butteraugli-gpu/examples/mem_one_size.rs` for the protocol.

#![cfg(feature = "cuda")]

use std::io::Write;
use std::time::{Duration, Instant};

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use iwssim_gpu::Iwssim;

const CHILD_HOLD_MS: u64 = 400;

fn synth_gray(w: u32, h: u32, seed: u32) -> Vec<f32> {
    use std::num::Wrapping;
    let n = (w as usize) * (h as usize);
    let mut v = Vec::with_capacity(n);
    let mut s = Wrapping(seed);
    for _ in 0..n {
        s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
        v.push(((s.0 >> 16) & 0xff) as f32);
    }
    v
}

fn main() {
    let mode = std::env::var("WORKER_MODE").unwrap_or_else(|_| "full".into());
    let w: u32 = std::env::var("WORKER_W").unwrap_or_else(|_| "1024".into()).parse().unwrap();
    let h: u32 = std::env::var("WORKER_H").unwrap_or_else(|_| "1024".into()).parse().unwrap();

    let r = synth_gray(w, h, 42);
    let d = synth_gray(w, h, 137);

    let client = Backend::client(&Default::default());
    let t0 = Instant::now();
    let score: f64 = match mode.as_str() {
        "full" => {
            let mut iw = Iwssim::<Backend>::new(client, w, h).expect("Iwssim::new");
            iw.set_reference(&r).expect("set_reference");
            let res = iw.compute_with_reference(&d).expect("compute_with_reference");
            res.score
        }
        "strip" => {
            let mut iw = Iwssim::<Backend>::new_strip(client, w, h, 256).expect("Iwssim::new_strip");
            let res = iw.compute_gray_stripped(&r, &d).expect("compute_gray_stripped");
            res.score
        }
        other => panic!("unknown WORKER_MODE: {other}"),
    };
    let warm_dt = t0.elapsed();

    println!("READY {score:.6} warm_ms={:.2}", warm_dt.as_secs_f64() * 1e3);
    std::io::stdout().flush().expect("flush stdout");

    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
}
