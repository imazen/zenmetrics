//! GPU memory measurement driver for zensim-gpu.
//! See `butteraugli-gpu/examples/mem_one_size.rs` for the protocol.
//!
//! `compute_features` is the only entry — there is no strip-mode
//! ssim2-style stripped API; `set_reference` + `compute_with_reference`
//! is the cached-ref hot path used by every encoder.

#![cfg(feature = "cuda")]

use std::io::Write;
use std::time::{Duration, Instant};

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use zensim_gpu::Zensim;

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
    let _feat = match mode.as_str() {
        "full" => {
            let mut z = Zensim::<Backend>::new(client, w, h).expect("Zensim::new");
            z.set_reference(&r).expect("set_reference");
            z.compute_with_reference(&d).expect("compute_with_reference")
        }
        other => panic!("unknown WORKER_MODE for zensim: {other}"),
    };
    let warm_dt = t0.elapsed();

    println!("READY 0.0 warm_ms={:.2}", warm_dt.as_secs_f64() * 1e3);
    std::io::stdout().flush().expect("flush stdout");

    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
}
