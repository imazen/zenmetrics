//! GPU capability detection via `nvidia-smi` subprocess.
//!
//! Mirrors the pattern in `iwssim-gpu::memory_mode::query_nvidia_smi_memory_free`
//! (one short subprocess call, parse CSV, fall back gracefully on
//! failure). All failures collapse to `GpuCapability { present: false,
//! ..Default::default() }` so the rest of the orchestrator can keep
//! running on CPU-only machines (CI, AMD/Intel boxes, snap-docker).

use std::process::Command;

use crate::GpuCapability;

/// Best-effort GPU detection. Order of operations:
///
/// 1. `nvidia-smi --query-gpu=gpu_name,memory.total,driver_version` —
///    one row per GPU. We use the first row only (single-GPU is the
///    current scope per `ORCHESTRATOR_DESIGN.md` decision #5).
/// 2. `nvidia-smi --query-gpu=compute_cap` — optional, some older
///    drivers don't support this query and exit nonzero. Treat
///    failure as `None`.
/// 3. `nvcc --version` — optional, gives the CUDA toolkit version
///    text. We extract the `release X.Y` token if present.
///
/// Returns `GpuCapability { present: false, .. }` when step 1 fails.
pub fn detect_gpu() -> GpuCapability {
    let Some((model, total_vram_mib, driver_version)) = query_primary_gpu() else {
        return GpuCapability::default();
    };
    let compute_capability = query_compute_capability();
    let cuda_runtime = query_cuda_runtime();
    GpuCapability {
        present: true,
        model,
        total_vram_mib,
        driver_version,
        cuda_runtime,
        compute_capability,
    }
}

/// Run `nvidia-smi --query-gpu=gpu_name,memory.total,driver_version
/// --format=csv,noheader,nounits` and parse the first row.
fn query_primary_gpu() -> Option<(String, usize, String)> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=gpu_name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().next()?;
    parse_primary_gpu_row(line)
}

/// Pure parser for the CSV row that `nvidia-smi` emits. Public via
/// `pub(crate)` only — kept as a private function so the tests can
/// exercise it without spawning a subprocess.
fn parse_primary_gpu_row(line: &str) -> Option<(String, usize, String)> {
    let mut parts = line.split(',').map(str::trim);
    let name = parts.next()?.to_string();
    let mem_mib: usize = parts.next()?.parse().ok()?;
    let driver = parts.next()?.to_string();
    if name.is_empty() || driver.is_empty() {
        return None;
    }
    Some((name, mem_mib, driver))
}

/// Try `nvidia-smi --query-gpu=compute_cap`. Older drivers reject
/// this query (CUDA 11.0 and earlier shipped without it); treat any
/// nonzero exit or empty output as `None`.
fn query_compute_capability() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?.trim();
    if line.is_empty() {
        None
    } else {
        Some(line.to_string())
    }
}

/// Try `nvcc --version` and extract the "release X.Y" token. nvcc may
/// not be on PATH even when CUDA is installed (see this repo's
/// `CLAUDE.md` — `nvcc` lives under `/usr/local/cuda/bin/`); we don't
/// touch PATH ourselves and treat absence as `None`.
fn query_cuda_runtime() -> Option<String> {
    let out = Command::new("nvcc").arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    parse_nvcc_release(&s)
}

fn parse_nvcc_release(text: &str) -> Option<String> {
    // Typical output line:
    //   Cuda compilation tools, release 13.2, V13.2.1
    for line in text.lines() {
        if let Some(idx) = line.find("release ") {
            let rest = &line[idx + "release ".len()..];
            // Token ends at the next comma or whitespace.
            let end = rest
                .find(|c: char| c == ',' || c.is_whitespace())
                .unwrap_or(rest.len());
            let token = rest[..end].trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_primary_gpu_row_canonical() {
        let row = "NVIDIA GeForce RTX 5070, 12288, 596.21";
        let (name, mem, drv) = parse_primary_gpu_row(row).unwrap();
        assert_eq!(name, "NVIDIA GeForce RTX 5070");
        assert_eq!(mem, 12288);
        assert_eq!(drv, "596.21");
    }

    #[test]
    fn parse_primary_gpu_row_extra_whitespace() {
        let row = "  NVIDIA A100 ,  40960 ,  535.86.10 ";
        let (name, mem, drv) = parse_primary_gpu_row(row).unwrap();
        assert_eq!(name, "NVIDIA A100");
        assert_eq!(mem, 40960);
        assert_eq!(drv, "535.86.10");
    }

    #[test]
    fn parse_primary_gpu_row_rejects_empty() {
        assert!(parse_primary_gpu_row("").is_none());
        assert!(parse_primary_gpu_row(", 0, ").is_none());
    }

    #[test]
    fn parse_primary_gpu_row_rejects_non_numeric_mem() {
        assert!(parse_primary_gpu_row("FakeGPU, abc, 1.0").is_none());
    }

    #[test]
    fn parse_nvcc_release_canonical() {
        let txt = "\
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Mon_Mar_10_19:24:32_PDT_2026
Cuda compilation tools, release 13.2, V13.2.1
Build cuda_13.2.r13.2/compiler.34795729_0
";
        assert_eq!(parse_nvcc_release(txt).as_deref(), Some("13.2"));
    }

    #[test]
    fn parse_nvcc_release_missing() {
        assert!(parse_nvcc_release("not nvcc output").is_none());
    }

    #[test]
    fn detect_gpu_never_panics() {
        // Doesn't matter what the result is — we just want to be sure
        // the call doesn't blow up on hosts without nvidia-smi.
        let _ = detect_gpu();
    }
}
