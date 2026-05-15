//! Measure current cvvdp-gpu vs pycvvdp JOD drift on the
//! 256×256 `chroma_shift` synth fixture.
//!
//! This is the ongoing open finding tracked in
//! `docs/CHROMA_DRIFT_INVESTIGATION.md`. Tick 196 closed the
//! SRGB8_TO_LINEAR_LUT bug → DKL planes bit-identical with pycvvdp.
//! Tick 198 confirmed Weber bands bit-identical. Tick 199 found T_p
//! REF-side diverges 0.9% rel. Tick 200 switched host_scalar's
//! LOG_L_BKG_AXIS interp from binary-search to uniform-rescale to
//! match pycvvdp's `interp1q`. This probe re-measures.
//!
//! Run with:
//!     cargo run --release --example chroma_shift_drift_probe \
//!         -p cvvdp-gpu --features cuda
//!
//! pycvvdp golden (from `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`):
//! JOD = 9.664865 for `synth_256x256_chroma_shift`.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "hip", not(feature = "cuda"), not(feature = "wgpu")))]
type Backend = cubecl::hip::HipRuntime;

const W: u32 = 256;
const H: u32 = 256;
const PYCVVDP_GOLDEN: f32 = 9.664865;

fn synth_chroma_shift_pair() -> (Vec<u8>, Vec<u8>) {
    let n = (W * H * 3) as usize;
    let mut ref_b = vec![0u8; n];
    let mut dist_b = vec![0u8; n];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let r = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            let g = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            let b = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
            let i = (y * W as usize + x) * 3;
            ref_b[i] = r;
            ref_b[i + 1] = g;
            ref_b[i + 2] = b;
            dist_b[i] = r;
            dist_b[i + 1] = (g as i16 + 16).clamp(0, 255) as u8;
            dist_b[i + 2] = b;
        }
    }
    (ref_b, dist_b)
}

fn main() {
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_bytes, dist_bytes) = synth_chroma_shift_pair();

    let client = Backend::client(&Default::default());
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, W, H, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("compute_dkl_jod");

    let diff = gpu_jod - PYCVVDP_GOLDEN;
    let abs_diff = diff.abs();

    println!("chroma_shift JOD drift probe");
    println!("  cvvdp-gpu (current):  {gpu_jod:.6}");
    println!("  pycvvdp golden:       {PYCVVDP_GOLDEN:.6}");
    println!("  signed diff:          {diff:+.6}");
    println!("  abs diff:             {abs_diff:.6}");
    println!();
    if abs_diff < 0.005 {
        println!("STATUS: drift closed (< 0.005 tolerance — match other JOD fixtures)");
        println!("  → re-enable the chroma_shift JOD parity test in tests/pipeline_color.rs");
    } else if abs_diff < 0.05 {
        println!("STATUS: drift narrowed but above tight tolerance (0.005 < d < 0.05)");
    } else {
        println!("STATUS: drift still significant (>= 0.05)");
        println!("  → continue downstream investigation per docs/CHROMA_DRIFT_INVESTIGATION.md");
    }
}
