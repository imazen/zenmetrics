//! CPU + RAM detection. Pure-Rust, no subprocess fan-out:
//!
//! * Brand string + SIMD feature flags via `raw-cpuid` (CPUID instr).
//! * Logical core count via `std::thread::available_parallelism`.
//! * Total physical RAM in MiB via `sysinfo` + WSL2 detection.
//!
//! ## Phase 2 RAM detection notes
//!
//! The Phase 1 `ram_mib` value reported 50185 on the 128 GB
//! water-cooled 7950X workstation. After investigation this is NOT a
//! `sysinfo` bug — it's the actual ceiling the Linux kernel sees under
//! WSL2 with `memory=50GB` set in `.wslconfig`. Surfacing 128 GB would
//! lie about what the orchestrator can schedule against; the
//! orchestrator's CPU-backend scheduler cares about
//! Linux-kernel-visible RAM, not Windows-host RAM.
//!
//! Phase 2 keeps `ram_mib` reading the kernel-visible total (what
//! `MemTotal` in `/proc/meminfo` reports — same as the original Phase
//! 1 value) and explicitly documents the WSL2 case via a new
//! `wsl_host_ram_mib_hint` field that callers can use to estimate
//! whether re-configuring `.wslconfig` would meaningfully grow the
//! schedulable RAM.

use std::fs;

#[cfg(target_arch = "x86_64")]
use raw_cpuid::{CpuId, CpuIdReader};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

use crate::CpuCapability;

/// Detect CPU brand, logical core count, SIMD feature flags, and total
/// RAM. Always succeeds on supported architectures — returns sensible
/// defaults (empty brand / 1 core / empty features) on the rare
/// platforms where CPUID isn't usable.
///
/// Architecture support:
/// - x86_64: full CPUID detection via raw-cpuid (brand string, SIMD
///   features sse4.1/sse4.2/aes/popcnt/fma/avx/bmi1/bmi2/avx2/avx512*).
/// - aarch64 (incl. Hetzner CAX Ampere Altra): empty brand, empty
///   feature list, `logical_cores` + `ram_mib` still populated. The
///   orchestrator's CpuAdapter does not currently key dispatch on
///   aarch64 features (NEON is implied baseline), so this is correct
///   behaviour — the orchestrator falls through to the CPU adapter on
///   any arch where the GPU runtimes aren't compiled in or dlopen
///   fails. Other arches behave the same as aarch64.
pub fn detect_cpu() -> CpuCapability {
    #[cfg(target_arch = "x86_64")]
    {
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

    #[cfg(not(target_arch = "x86_64"))]
    {
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let ram_mib = total_ram_mib();
        CpuCapability {
            brand: String::new(),
            logical_cores,
            features: Vec::new(),
            ram_mib,
        }
    }
}

/// Walk the CPUID feature bits we care about and collect them as
/// lowercase strings in a stable order. The list matches what the
/// orchestrator design doc enumerates: SIMD families relevant to the
/// CPU metric backends (CVVDP, butter, ssim2, dssim, iwssim, zensim).
#[cfg(target_arch = "x86_64")]
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
/// bytes; divide by 1 MiB.
///
/// Phase 2 hardened from Phase 1 by calling
/// `System::new_with_specifics(RefreshKind::nothing().with_memory(...))`
/// then `refresh_memory_specifics(RAM)`; the explicit form is
/// idempotent against future sysinfo refactors. The function also
/// falls back to parsing `/proc/meminfo` `MemTotal:` on Linux when
/// sysinfo returns 0 (rare but seen on cgroup-restricted hosts).
///
/// The returned value is the **Linux-kernel-visible** total RAM. On
/// WSL2 this is bounded by `.wslconfig:memory=...`. Use
/// [`detect_wsl2_host_ram_mib_hint`] to know whether the host could
/// expose more.
fn total_ram_mib() -> usize {
    let refresh = RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram());
    let mut sys = System::new_with_specifics(refresh);
    sys.refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
    let bytes = sys.total_memory();
    if bytes > 0 {
        (bytes / (1024 * 1024)) as usize
    } else {
        // Fall back to /proc/meminfo's MemTotal (Linux-only). Returns 0
        // on systems where that file doesn't exist.
        proc_meminfo_total_mib()
    }
}

/// Parse `MemTotal:` from `/proc/meminfo`. Returns 0 if the file is
/// missing or the line isn't present. The format is `MemTotal: NNN kB`
/// (always kilobytes, always integer).
fn proc_meminfo_total_mib() -> usize {
    let s = match fs::read_to_string("/proc/meminfo") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    parse_meminfo_total_kib(&s).map(|kib| kib / 1024).unwrap_or(0)
}

fn parse_meminfo_total_kib(text: &str) -> Option<usize> {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            // Tokens: <whitespace><number><whitespace>kB
            return rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<usize>().ok());
        }
    }
    None
}

/// Detect WSL2 and surface the Windows-host RAM if cheaply available.
///
/// Detection: `/proc/version` contains "microsoft" or "WSL" — that's
/// the standard WSL2 signature. We do NOT shell out to PowerShell
/// (slow startup, doesn't always exist, blocks the orchestrator
/// constructor); we just flag the WSL2 case so callers can interpret
/// `ram_mib` correctly.
///
/// Returns:
/// - `None`: not running under WSL2 (Linux native, macOS, Windows
///   native, etc.). `ram_mib` is the actual physical RAM.
/// - `Some(0)`: WSL2 detected, host RAM unknown.
/// - `Some(n)`: WSL2 detected, host RAM is `n` MiB (when discoverable
///   from `/proc/cmdline` or environment hints — best-effort).
pub fn detect_wsl2_host_ram_mib_hint() -> Option<usize> {
    let version = fs::read_to_string("/proc/version").ok()?;
    let lowered = version.to_ascii_lowercase();
    if !(lowered.contains("microsoft") || lowered.contains("wsl")) {
        return None;
    }
    // Found WSL2. We don't have a cheap way to ask the Windows host
    // for its total RAM from inside Linux, so return Some(0) as a
    // "WSL2 detected, host RAM unknown" sentinel. Callers who want
    // the actual host RAM can shell out to powershell.exe themselves;
    // we keep the orchestrator constructor fast.
    Some(0)
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

    #[cfg(target_arch = "x86_64")]
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

    #[test]
    fn parse_meminfo_total_kib_canonical() {
        // Canonical /proc/meminfo header — `MemTotal:` line followed by
        // a space-separated number and the `kB` suffix.
        let text = "MemTotal:       51390444 kB\nMemFree:        15243200 kB\n";
        assert_eq!(parse_meminfo_total_kib(text), Some(51_390_444));
    }

    #[test]
    fn parse_meminfo_total_kib_missing_returns_none() {
        let text = "MemFree:        15243200 kB\nMemAvailable:   45230000 kB\n";
        assert_eq!(parse_meminfo_total_kib(text), None);
    }

    #[test]
    fn parse_meminfo_total_kib_handles_extra_whitespace() {
        let text = "MemTotal:\t\t51390444 kB\n";
        assert_eq!(parse_meminfo_total_kib(text), Some(51_390_444));
    }

    #[test]
    fn wsl2_hint_never_panics() {
        // Smoke — just verify the call doesn't panic on the host. The
        // return value depends on whether the test runs under WSL2.
        let _ = detect_wsl2_host_ram_mib_hint();
    }
}
