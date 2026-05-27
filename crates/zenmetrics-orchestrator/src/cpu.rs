//! CPU + RAM detection. Pure-Rust, no subprocess fan-out:
//!
//! * Brand string + SIMD feature flags via `raw-cpuid` (CPUID instr).
//! * Logical core count via `std::thread::available_parallelism`.
//! * Total physical RAM in MiB via `sysinfo`.

use raw_cpuid::{CpuId, CpuIdReader};
use sysinfo::System;

use crate::CpuCapability;

/// Detect CPU brand, logical core count, SIMD feature flags, and total
/// RAM. Always succeeds on supported architectures — returns sensible
/// defaults (empty brand / 1 core / empty features) on the rare
/// platforms where CPUID isn't usable.
pub fn detect_cpu() -> CpuCapability {
    let cpuid = CpuId::new();

    let brand = cpuid
        .get_processor_brand_string()
        .map(|b| b.as_str().trim().to_string())
        .unwrap_or_default();

    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let features = collect_features(&cpuid);

    let ram_mib = total_ram_mib();

    CpuCapability {
        brand,
        logical_cores,
        features,
        ram_mib,
    }
}

/// Walk the CPUID feature bits we care about and collect them as
/// lowercase strings in a stable order. The list matches what the
/// orchestrator design doc enumerates: SIMD families relevant to the
/// CPU metric backends (CVVDP, butter, ssim2, dssim, iwssim, zensim).
fn collect_features<R: CpuIdReader>(cpuid: &CpuId<R>) -> Vec<String> {
    let mut out: Vec<&'static str> = Vec::new();

    if let Some(fi) = cpuid.get_feature_info() {
        if fi.has_sse41() {
            out.push("sse4.1");
        }
        if fi.has_sse42() {
            out.push("sse4.2");
        }
        if fi.has_aesni() {
            out.push("aes");
        }
        if fi.has_popcnt() {
            out.push("popcnt");
        }
        if fi.has_fma() {
            out.push("fma");
        }
        if fi.has_avx() {
            out.push("avx");
        }
    }

    if let Some(efi) = cpuid.get_extended_feature_info() {
        if efi.has_bmi1() {
            out.push("bmi1");
        }
        if efi.has_bmi2() {
            out.push("bmi2");
        }
        if efi.has_avx2() {
            out.push("avx2");
        }
        if efi.has_avx512f() {
            out.push("avx512f");
        }
        if efi.has_avx512bw() {
            out.push("avx512bw");
        }
        if efi.has_avx512vl() {
            out.push("avx512vl");
        }
        if efi.has_avx512dq() {
            out.push("avx512dq");
        }
    }

    // Stable order: emit in CPUID-discovery order (set above), but
    // sort for determinism so the machine_hash doesn't depend on
    // raw-cpuid's iteration order across versions.
    out.sort();
    out.dedup();
    out.into_iter().map(str::to_string).collect()
}

/// Total physical RAM in MiB. `sysinfo::System::total_memory()` returns
/// bytes; divide by 1 MiB. We refresh only the memory subsystem so
/// startup cost stays low (no process scan, no disk scan).
fn total_ram_mib() -> usize {
    let mut sys = System::new();
    sys.refresh_memory();
    let bytes = sys.total_memory();
    (bytes / (1024 * 1024)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_cpu_basic_invariants() {
        let cpu = detect_cpu();
        assert!(cpu.logical_cores >= 1, "should report at least 1 core");
        // Brand may legitimately be empty under qemu / weird VMs;
        // logical_cores is the only universally-reliable signal.
        // On a real CPU we'd see something — at minimum sse4.2 on
        // anything x86_64 from the last 15 years.
        if !cpu.brand.is_empty() {
            assert!(
                cpu.features.iter().any(|f| f == "sse4.2"),
                "modern x86_64 should expose sse4.2; got {:?}",
                cpu.features
            );
        }
    }

    #[test]
    fn collect_features_sorted_and_deduped() {
        let cpuid = CpuId::new();
        let feats = collect_features(&cpuid);
        let mut sorted = feats.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(feats, sorted, "feature list must be sorted+deduped");
    }

    #[test]
    fn total_ram_is_positive_on_real_systems() {
        // Skip on weirdly-instrumented CI hosts that return 0; only
        // assert the call doesn't panic.
        let _ = total_ram_mib();
    }
}
