//! Heaptrack driver: one DSSIM compute at 12 MP, mode selected by
//! the first CLI argument ("whole" or "strip").
//!
//! Run:
//! ```bash
//! cargo build --release -p dssim-gpu --example heaptrack_strip_12mp \
//!     --features cuda,cubecl-types
//! heaptrack -o /tmp/heaptrack_dssim_whole.zst \
//!     ./target/release/examples/heaptrack_strip_12mp whole
//! heaptrack -o /tmp/heaptrack_dssim_strip.zst \
//!     ./target/release/examples/heaptrack_strip_12mp strip
//! ```

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;

use dssim_gpu::Dssim;

fn make_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
    use std::num::Wrapping;
    let mut v = Vec::with_capacity(w * h * 3);
    let mut s = Wrapping(seed);
    for _ in 0..w * h {
        for _ in 0..3 {
            s = s * Wrapping(1664525u32) + Wrapping(1013904223u32);
            v.push(((s.0 >> 16) & 0xFF) as u8);
        }
    }
    v
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "strip".to_string());
    // 12 MP at 3464×3464 (close to a square 12 MP image — matches the
    // bench grid).
    let (w, h, h_body) = (3464_u32, 3464_u32, 256_u32);

    let ref_rgb = make_image(w as usize, h as usize, 42);
    let dis_rgb = make_image(w as usize, h as usize, 137);

    let client = Backend::client(&Default::default());
    let score = match mode.as_str() {
        "whole" => {
            let mut d = Dssim::<Backend>::new(client, w, h).unwrap();
            d.compute(&ref_rgb, &dis_rgb).unwrap().score
        }
        "strip" => {
            let mut d = Dssim::<Backend>::new_strip(client, w, h, h_body).unwrap();
            d.compute(&ref_rgb, &dis_rgb).unwrap().score
        }
        other => panic!("usage: heaptrack_strip_12mp <whole|strip>, got {other}"),
    };

    println!("mode={mode} score={score:.8}");
}
