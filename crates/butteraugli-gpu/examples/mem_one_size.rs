//! GPU memory measurement driver for butteraugli-gpu.
//!
//! Subprocess-per-cell pattern: the parent orchestrator
//! (`scripts/memory_audit/audit_gpu_metrics.py`) spawns this binary
//! once per (mode, width, height) cell, samples nvidia-smi
//! memory.used in a brief hold window, and reports the delta. The
//! child holds the GPU buffers alive after signaling READY so the
//! parent can sample at quiescent steady state — cubecl's memory
//! pool caches buffers across Drop, so the only way to see a clean
//! baseline is to let the OS reclaim the process.
//!
//! Modes:
//! - `full`     — `Butteraugli::new` + `set_reference` +
//!                `compute_with_reference` (whole-image, cached-ref).
//! - `strip`    — `Butteraugli::new_strip` (body 256) + `compute`
//!                (whole-image stripped under the hood).
//!
//! Output: one line on stdout `READY <score>` after the first
//! warm-up call returns. Stays alive for CHILD_HOLD_MS so the parent
//! can poll nvidia-smi. Exit code 0 on success.

#![cfg(feature = "cuda")]

use std::io::Write;
use std::time::{Duration, Instant};

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use butteraugli_gpu::Butteraugli;

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
    let score: f32 = match mode.as_str() {
        "full" => {
            let mut b = Butteraugli::<Backend>::new(client, w, h);
            b.set_reference(&r).expect("set_reference");
            let res = b.compute_with_reference(&d).expect("compute_with_reference");
            res.score
        }
        "strip" => {
            let mut b = Butteraugli::<Backend>::new_strip(client, w, h, 256);
            let res = b.compute_strip(&r, &d).expect("compute_strip");
            res.score
        }
        other => panic!("unknown WORKER_MODE: {other}"),
    };
    let warm_dt = t0.elapsed();

    println!("READY {score:.6} warm_ms={:.2}", warm_dt.as_secs_f64() * 1e3);
    std::io::stdout().flush().expect("flush stdout");

    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
}
