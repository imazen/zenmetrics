#![forbid(unsafe_code)]

//! ColorVideoVDP (still-image) score via the `cvvdp-gpu` crate.
//!
//! cvvdp's JOD is on a 0–10 scale where 10 = imperceptible. Currently
//! routes through `cvvdp_gpu::Cvvdp::score`, which is the
//! parity-locked host scalar path (matches pycvvdp v0.5.4 on the v1
//! R2 manifest within 0.006 JOD per `shadow_jod`). The full GPU
//! composition path also ships as `Cvvdp::compute_dkl_jod` (color →
//! weber → CSF → masking → GPU pool → host fold), parity-locked
//! against the host scalar at f32 precision for q ≥ 20 and against
//! the pycvvdp manifest values via `shadow_jod_gpu`. The CLI will
//! retarget once the GPU path's q=1 drift through `met2jod`'s steep
//! slope is resolved or absorbed into the test tolerance.

use cubecl::Runtime;

use crate::decode::Rgb8Image;
use crate::metrics::GpuRuntime;
use crate::metrics::gpu_runtime_dispatch::{auto_order, runtime_label};

pub fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    let candidates: Vec<GpuRuntime> = match runtime {
        GpuRuntime::Auto => auto_order().to_vec(),
        other => vec![other],
    };

    let mut last_error: Option<String> = None;
    for rt in candidates {
        match score_with_runtime(reference, distorted, rt) {
            Ok(value) => return Ok(value),
            Err(e) => {
                last_error = Some(format!("{}: {e}", runtime_label(rt)));
            }
        }
    }
    Err(format!(
        "cvvdp: no runtime succeeded; last error: {}",
        last_error.unwrap_or_else(|| "none".into())
    )
    .into())
}

fn score_with_runtime(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    match runtime {
        GpuRuntime::Cuda => {
            #[cfg(feature = "gpu-cuda")]
            {
                run::<cubecl::cuda::CudaRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-cuda"))]
            {
                Err("cuda runtime not compiled in (rebuild with `--features gpu-cuda`)".into())
            }
        }
        GpuRuntime::Wgpu => {
            #[cfg(feature = "gpu-wgpu")]
            {
                run::<cubecl::wgpu::WgpuRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            {
                Err("wgpu runtime not compiled in (rebuild with `--features gpu-wgpu`)".into())
            }
        }
        GpuRuntime::Hip => {
            #[cfg(feature = "gpu-hip")]
            {
                run::<cubecl::hip::HipRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-hip"))]
            {
                Err("hip runtime not compiled in (rebuild with `--features gpu-hip`)".into())
            }
        }
        GpuRuntime::Cpu => {
            #[cfg(feature = "gpu-cpu")]
            {
                // cvvdp-gpu's pool kernel uses Atomic<f32>::fetch_add,
                // which cubecl-cpu does not implement (panic at
                // cubecl-cpu/compiler/visitor/elem.rs: "atomic<f32>
                // not implemented"). The crate ships
                // Cvvdp::compute_dkl_jod_host_pool for exactly this
                // case — same JOD output, but pools the per-band
                // D values on the host instead of via the atomic
                // kernel. Without this routing the auto-dispatch
                // fall-through (Cuda → Wgpu → Hip → Cpu) bombs the
                // whole sweep on boxes where CUDA fails to init.
                run_cpu_host_pool::<cubecl::cpu::CpuRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-cpu"))]
            {
                Err("cpu runtime not compiled in (rebuild with `--features gpu-cpu`)".into())
            }
        }
        GpuRuntime::Auto => unreachable!("Auto is expanded by score()"),
    }
}

fn run<R: Runtime>(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    let client = R::client(&Default::default());
    let mut c = cvvdp_gpu::Cvvdp::<R>::new(
        client,
        reference.width,
        reference.height,
        cvvdp_gpu::CvvdpParams::PLACEHOLDER,
    )
    .map_err(|e| format!("Cvvdp::new: {e}"))?;
    let jod = c
        .score(&reference.pixels, &distorted.pixels)
        .map_err(|e| format!("Cvvdp::score: {e}"))?;
    if !jod.is_finite() {
        return Err(format!("cvvdp produced non-finite JOD: {jod}").into());
    }
    Ok(jod)
}

/// Sibling of `run` that routes through
/// [`cvvdp_gpu::Cvvdp::compute_dkl_jod_host_pool`] instead of
/// `Cvvdp::score`. Same JOD output (parity-locked at f32 noise
/// by `compute_dkl_jod_host_pool_matches_compute_dkl_jod`), but
/// pools per-band D values host-side so it runs on runtimes
/// without `Atomic<f32>::fetch_add` (`cubecl-cpu`, Metal via
/// `cubecl-wgpu` — see [`cvvdp_gpu::Cvvdp::compute_dkl_jod`]'s
/// Backend support section).
#[cfg(feature = "gpu-cpu")]
fn run_cpu_host_pool<R: Runtime>(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    let client = R::client(&Default::default());
    let mut c = cvvdp_gpu::Cvvdp::<R>::new(
        client,
        reference.width,
        reference.height,
        cvvdp_gpu::CvvdpParams::PLACEHOLDER,
    )
    .map_err(|e| format!("Cvvdp::new (cpu host_pool): {e}"))?;
    // Cvvdp::new uses DisplayGeometry::STANDARD_4K (per pipeline.rs:478);
    // mirror that here since `geometry` is a private field and there's
    // no public accessor. Cvvdp::score does the same internally.
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let jod = c
        .compute_dkl_jod_host_pool(&reference.pixels, &distorted.pixels, ppd)
        .map_err(|e| format!("Cvvdp::compute_dkl_jod_host_pool: {e}"))?;
    if !jod.is_finite() {
        return Err(format!("cvvdp (cpu host_pool) produced non-finite JOD: {jod}").into());
    }
    Ok(f64::from(jod))
}
