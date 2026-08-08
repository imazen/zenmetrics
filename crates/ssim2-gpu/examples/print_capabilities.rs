//! Diagnostic: print whichever cubecl runtime is selected, the GPU
//! adapter info (when wgpu), and the cubecl atomic-usage table (which
//! atomic ops are registered for f32, u32, etc).
//!
//! cubecl-wgpu silently no-ops `Atomic<f32>::fetch_add` when the
//! device doesn't expose `SHADER_FLOAT32_ATOMIC`, which makes our
//! reductions return zero and the score collapse to ~100 regardless
//! of distortion. This example dumps the registered atomic usages so
//! we can see in CI exactly what each runner supports.

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

use cubecl::Runtime;
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};

/// Which runtime the `Backend` alias above actually resolved to.
const SELECTED_RUNTIME: &str = if cfg!(feature = "cuda") {
    "CudaRuntime"
} else if cfg!(feature = "wgpu") {
    "WgpuRuntime"
} else {
    "<none — build with --features cuda or wgpu>"
};

/// Report the Vulkan adapter cubecl-wgpu would select.
///
/// This exists because a Mesa-equipped Linux box enumerates **two** Vulkan
/// devices — the real discrete GPU and `llvmpipe`, a CPU software rasterizer.
/// Landing on llvmpipe yields correct scores at CPU speed while still calling
/// itself "Vulkan", which silently invalidates any performance measurement.
/// Nothing in the tree used to print this (imazen/zenmetrics#42), so there was
/// no way to tell the two apart after the fact.
#[cfg(feature = "wgpu")]
fn report_wgpu_adapter() {
    use cubecl::wgpu::{RuntimeOptions, Vulkan, WgpuDevice, init_setup};

    println!("== wgpu Vulkan adapter ==");
    let setup = init_setup::<Vulkan>(&WgpuDevice::default(), RuntimeOptions::default());
    let info = setup.adapter.get_info();
    let dtype = format!("{:?}", info.device_type);
    println!("  name        = {:?}", info.name);
    println!("  device_type = {dtype}");
    println!("  backend     = {:?}", info.backend);
    println!("  driver      = {:?} {}", info.driver, info.driver_info);
    if !dtype.contains("DiscreteGpu") {
        println!(
            "  WARNING: this is NOT a discrete GPU. Timings taken on this adapter \
             measure a software rasterizer, not the GPU."
        );
    }
    println!();
}

#[cfg(not(feature = "wgpu"))]
fn report_wgpu_adapter() {
    println!("== wgpu Vulkan adapter ==\n  (not built with --features wgpu)\n");
}

fn main() {
    println!("== selected cubecl runtime ==\n  {SELECTED_RUNTIME}\n");
    report_wgpu_adapter();

    let client = Backend::client(&Default::default());
    let props = client.properties();

    println!("== cubecl atomic usages ==");

    let probes: &[(&str, StorageType)] = &[
        (
            "Atomic<f32>",
            StorageType::Atomic(ElemType::Float(FloatKind::F32)),
        ),
        (
            "Atomic<f64>",
            StorageType::Atomic(ElemType::Float(FloatKind::F64)),
        ),
        (
            "Atomic<u32>",
            StorageType::Atomic(ElemType::UInt(UIntKind::U32)),
        ),
        (
            "Atomic<i32>",
            StorageType::Atomic(ElemType::Int(IntKind::I32)),
        ),
    ];
    for (name, ty) in probes {
        let usage = props.atomic_type_usage(Type::Scalar(*ty));
        let mut flags: Vec<&str> = Vec::new();
        if usage.contains(AtomicUsage::LoadStore) {
            flags.push("LoadStore");
        }
        if usage.contains(AtomicUsage::Add) {
            flags.push("Add");
        }
        if usage.contains(AtomicUsage::MinMax) {
            flags.push("MinMax");
        }
        println!(
            "  {:14} = {}",
            name,
            if flags.is_empty() {
                "<none>".to_string()
            } else {
                flags.join("|")
            },
        );
    }
}
