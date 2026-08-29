#![forbid(unsafe_code)]

//! Butteraugli GPU two-column emit.
//!
//! Our TSV / parquet schema emits **both** the max-norm
//! (`butteraugli_max_gpu`) and the libjxl 3-norm
//! (`butteraugli_pnorm3_gpu`, matching `butteraugli_main --pnorm` and
//! the Cloudinary CID22 paper), produced in one `compute()` call. The
//! 3-norm lives on `ButteraugliResult.pnorm_3` — a field of the typed
//! `butteraugli_gpu::Butteraugli<R>` pipeline result — so this module
//! reaches into the umbrella's `zenmetrics_api::butter` re-export to
//! drive the typed path directly. Same backend dispatch / `auto`
//! fall-through semantics as [`crate::metrics::run_gpu_via_umbrella`].
//!
//! The faithful HDR linear-planes path lived here too, but `--hdr` scoring
//! now routes through `zenmetrics_api::hdr::HdrScorer` (see
//! `crate::hdr::score_via_hdr_scorer`), so only the sRGB-u8 two-column emit
//! remains.
//!
//! ## This module is foldable — its original rationale is obsolete
//!
//! It used to open with "the umbrella's opaque
//! [`zenmetrics_api::Metric::compute_srgb_u8`] returns a single
//! `Score { value }` for butter — the max-norm only". **That has been false
//! since `a7fb1f35`**, which shipped
//! [`zenmetrics_api::Metric::compute_srgb_u8_multi`] returning
//! `Scores = [max, pnorm_3]` from the same single fused kernel. The other
//! reason to keep the typed path — preserving `Butteraugli<HipRuntime>`, which
//! the shared opaque `Backend` does not carry — went away when HIP was dropped
//! by owner decision on 2026-08-28.
//!
//! So this module can be replaced by one `compute_srgb_u8_multi` call at its
//! only caller (`crate::metrics::run_metric`'s `ButteraugliGpu` arm). What that
//! change is gated on is *measurement, not capability*: the umbrella builds
//! butteraugli in `MemoryMode::Auto` (strip-preferred) where this path is
//! whole-image, and `benchmarks/butter_strip_drift_2026-08-28.md` measured that
//! difference over 6,810 cells — the max-norm is **bit-identical** in
//! 6,810/6,810, but `pnorm_3` drifts up to **1.72e-6** relative. That is small,
//! and it is not zero, so folding shifts a column that already-harvested fleet
//! rows carry. Tracked as item 1 of imazen/zenmetrics#21.

use cubecl::Runtime;
use zenmetrics_api::butter;

use crate::decode::Rgb8Image;
use crate::metrics::{GpuRuntime, auto_order, runtime_label};

pub fn score_both(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "butteraugli-gpu: reference ({}×{}) and distorted ({}×{}) differ in size",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let candidates: Vec<GpuRuntime> = match runtime {
        GpuRuntime::Auto => auto_order().to_vec(),
        other => vec![other],
    };
    let mut last_error: Option<String> = None;
    for rt in candidates {
        match score_with_runtime(reference, distorted, rt) {
            Ok(value) => return Ok(value),
            Err(e) => last_error = Some(format!("{}: {e}", runtime_label(rt))),
        }
    }
    Err(format!(
        "butteraugli-gpu: no runtime succeeded; last error: {}",
        last_error.unwrap_or_else(|| "none".into())
    )
    .into())
}

fn score_with_runtime(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
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
                run::<cubecl::cpu::CpuRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-cpu"))]
            {
                Err("cpu runtime not compiled in (rebuild with `--features gpu-cpu`)".into())
            }
        }
        GpuRuntime::Auto => unreachable!("Auto is expanded by score_both()"),
    }
}

fn run<R: Runtime>(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let client = R::client(&Default::default());
    let mut b = butter::Butteraugli::<R>::new_multires(client, reference.width, reference.height);
    let result = b
        .compute(&reference.pixels, &distorted.pixels)
        .map_err(|e| format!("butteraugli-gpu: {e}"))?;
    let max = result.score as f64;
    let pnorm3 = result.pnorm_3 as f64;
    if !max.is_finite() {
        return Err(format!("butteraugli-gpu produced non-finite score (max-norm): {max}").into());
    }
    if !pnorm3.is_finite() {
        return Err(
            format!("butteraugli-gpu produced non-finite pnorm_3 (3-norm): {pnorm3}").into(),
        );
    }
    Ok((max, pnorm3))
}
