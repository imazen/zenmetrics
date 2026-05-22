//! Wall-clock timing comparison: capped vs uncapped Cvvdp.
//!
//! Measures `score()` latency at 4000×3000 (12 MP) and 1024×8192
//! (panorama) for `cap = None` and `cap = Some(8)` to validate that
//! the cap mechanism has negligible performance cost in Full mode.
//! (It SHOULD be neutral — the cap drops only the smallest coarse
//! band's launches.)
//!
//! Run with:
//!
//!     cargo run -p cvvdp-gpu --features cubecl-types \
//!         --example cap_perf_compare --release
//!
//! Output: lines of `size,cap,iter,ms`.

use cubecl::Runtime;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
use cvvdp_gpu::Cvvdp;
use std::time::Instant;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (w as usize) * (h as usize) * 3;
    let mut r = vec![0u8; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = (y * w as usize + x) * 3;
            r[i] = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            r[i + 1] = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            r[i + 2] = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
        }
    }
    let d: Vec<u8> = r
        .chunks_exact(3)
        .flat_map(|p| {
            [
                p[0].saturating_sub(8),
                p[1].saturating_sub(4),
                p[2].saturating_add(12),
            ]
        })
        .collect();
    (r, d)
}

fn time_case(label: &str, w: u32, h: u32, cap: Option<u32>) {
    let client = Backend::client(&Default::default());
    let geom = DisplayGeometry::STANDARD_4K;
    let mut cvvdp = Cvvdp::<Backend>::new_with_geometry_and_cap(
        client,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        geom,
        cap,
    )
    .expect("new Cvvdp");

    let (r, d) = synth_pair(w, h);

    // Warm-up.
    let _ = cvvdp.score(&r, &d).expect("score warm");

    // 3 timed iterations.
    for iter in 0..3 {
        let t = Instant::now();
        let _ = cvvdp.score(&r, &d).expect("score");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let cap_str = match cap {
            Some(c) => format!("cap{c}"),
            None => "uncapped".to_string(),
        };
        println!("{label},{cap_str},{iter},{ms:.2}");
    }
}

fn main() {
    println!("size,cap,iter,ms");
    let cases: &[(u32, u32, &str)] = &[
        (4000, 3000, "12mp_synth"),
        (1024, 8192, "panorama_1024x8192"),
        // 24 MP square — the canonical strip-processing motivator.
        // Fits the 8 GB VRAM cap via the existing Full pipeline
        // (4.5 GB raw per cap_memory_estimate); the cap variants
        // here exist to validate the cap mechanism doesn't regress
        // perf in Full mode.
        (4900, 4900, "24mp_square"),
    ];
    let caps: &[Option<u32>] = &[None, Some(8), Some(6)];
    for &(w, h, label) in cases {
        for &cap in caps {
            time_case(label, w, h, cap);
        }
    }
}
