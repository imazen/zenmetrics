//! GPU memory measurement driver for cvvdp-gpu.
//! See `butteraugli-gpu/examples/mem_one_size.rs` for the protocol.
//!
//! Modes:
//! - `full`        — `Cvvdp::new` (whole-image)
//! - `strip_pair`  — `Cvvdp::new_strip_pair` (h_body=256, Mode B)

#![cfg(feature = "cuda")]

use std::io::Write;
use std::time::{Duration, Instant};

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

#[path = "../tests/common/mod.rs"]
mod common;

use common::Backend;

const CHILD_HOLD_MS: u64 = 400;

fn main() {
    let mode = std::env::var("WORKER_MODE").unwrap_or_else(|_| "full".into());
    let w: u32 = std::env::var("WORKER_W").unwrap_or_else(|_| "1024".into()).parse().unwrap();
    let h: u32 = std::env::var("WORKER_H").unwrap_or_else(|_| "1024".into()).parse().unwrap();

    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (r, d) = common::synth_pair_with_offset_dist(w as usize, h as usize);

    let client = Backend::client(&Default::default());
    let t0 = Instant::now();
    let jod: f32 = match mode.as_str() {
        "full" => {
            let mut c = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
                .expect("Cvvdp::new");
            c.compute_dkl_jod(&r, &d, ppd).expect("compute_dkl_jod")
        }
        "strip_pair" => {
            let mut c = Cvvdp::<Backend>::new_strip_pair(client, w, h, 256, CvvdpParams::PLACEHOLDER)
                .expect("Cvvdp::new_strip_pair");
            c.compute_dkl_jod(&r, &d, ppd).expect("compute_dkl_jod")
        }
        other => panic!("unknown WORKER_MODE: {other}"),
    };
    let warm_dt = t0.elapsed();

    println!("READY {jod:.6} warm_ms={:.2}", warm_dt.as_secs_f64() * 1e3);
    std::io::stdout().flush().expect("flush stdout");

    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
}
