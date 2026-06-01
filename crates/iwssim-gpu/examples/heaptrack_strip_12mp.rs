//! Minimal driver to feed heaptrack with a 12 MP whole-image vs
//! strip-mode run. Build:
//!
//! ```bash
//! cargo build --release --example heaptrack_strip_12mp -p iwssim-gpu \
//!     --features cubecl-types,cuda
//! heaptrack -o /tmp/heaptrack_iwssim_whole \
//!     ./target/release/examples/heaptrack_strip_12mp whole
//! heaptrack -o /tmp/heaptrack_iwssim_strip \
//!     ./target/release/examples/heaptrack_strip_12mp strip
//! ```
//!
//! `whole`: allocates the whole-image pipeline at 4000×3000.
//! `strip`: allocates `new_strip` with body=1024 at 4000×3000.
//! Both run 3 iters of compute. heaptrack records peak RSS over the
//! run; the delta between modes is the host-side memory cost of each
//! path.

use std::env;

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use iwssim_gpu::Iwssim;

fn make_gray(w: u32, h: u32, seed: u32) -> Vec<f32> {
    use std::num::Wrapping;
    let mut v = Vec::with_capacity((w * h) as usize);
    let mut s = Wrapping(seed);
    for _ in 0..w * h {
        s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
        v.push(((s.0 >> 16) & 0xFF) as f32);
    }
    v
}

fn main() {
    let mode = env::args().nth(1).unwrap_or_else(|| "whole".to_string());
    let w = 4000_u32;
    let h = 3000_u32;
    let ref_g = make_gray(w, h, 42);
    let dis_g = make_gray(w, h, 137);

    let client = Backend::client(&Default::default());
    match mode.as_str() {
        "whole" => {
            let mut iw = Iwssim::<Backend>::new(client.clone(), w, h).unwrap();
            for _ in 0..3 {
                let _ = iw.compute_gray(&ref_g, &dis_g).unwrap();
            }
            cubecl::future::block_on(client.sync()).expect("client.sync");
            println!("done whole {w}x{h}");
        }
        "strip" => {
            let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, 1024).unwrap();
            for _ in 0..3 {
                let _ = iw.compute_gray_stripped(&ref_g, &dis_g).unwrap();
            }
            cubecl::future::block_on(client.sync()).expect("client.sync");
            println!("done strip body=1024 {w}x{h}");
        }
        _ => {
            eprintln!("usage: heaptrack_strip_12mp <whole|strip>");
            std::process::exit(2);
        }
    }
}
