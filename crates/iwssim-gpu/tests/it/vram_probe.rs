//! Tests for the live VRAM probe wired into [`vram_cap_bytes`].
//!
//! Hierarchy:
//!   1. `ZENMETRICS_VRAM_CAP_BYTES` env var (highest priority, hard cap).
//!   2. `nvidia-smi --query-gpu=memory.free` (cached process-wide,
//!      with a 10% safety factor).
//!   3. 8 GiB default fallback (CI / no GPU / AMD / Intel / etc.).
//!
//! These tests verify each layer of that hierarchy without depending
//! on whether the host machine has nvidia-smi — the probe's `Option`
//! return is the explicit signal.

use iwssim_gpu::{live_vram_probe_bytes, vram_cap_bytes};
use std::sync::{Mutex, OnceLock};

const VRAM_CAP_VAR: &str = "ZENMETRICS_VRAM_CAP_BYTES";

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn with_cap<R>(cap: Option<&str>, f: impl FnOnce() -> R) -> R {
    let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var(VRAM_CAP_VAR).ok();
    unsafe {
        match cap {
            Some(v) => std::env::set_var(VRAM_CAP_VAR, v),
            None => std::env::remove_var(VRAM_CAP_VAR),
        }
    }
    let out = f();
    unsafe {
        match prev {
            Some(p) => std::env::set_var(VRAM_CAP_VAR, p),
            None => std::env::remove_var(VRAM_CAP_VAR),
        }
    }
    out
}

#[test]
fn probe_returns_sensible_value_on_cuda_hosts() {
    // Probe layer. On a CUDA-capable host with nvidia-smi installed,
    // returns Some(bytes) with a sensible value (MB range to TB
    // range). On hosts without nvidia-smi, returns None — and that's
    // also valid (we can't fail this test on a non-NVIDIA machine).
    let probe = live_vram_probe_bytes();
    if let Some(bytes) = probe {
        // Must be at least 64 MiB (any modern GPU has more) and at
        // most 1 TiB (no consumer/datacenter GPU has more right now).
        let mib = 1024 * 1024;
        let tib = 1024 * mib * 1024;
        assert!(
            bytes >= 64 * mib,
            "live probe absurdly small: {bytes} bytes"
        );
        assert!(
            bytes <= tib as usize,
            "live probe absurdly large: {bytes} bytes"
        );
    }
    // No assertion when probe is None — that's the documented
    // best-effort behaviour. Verified more directly in
    // `fallback_to_8gb_when_probe_unavailable`.
}

#[test]
fn fallback_to_8gb_when_probe_unavailable() {
    // When the live probe is unavailable AND no env var is set, the
    // cap MUST be exactly 8 GiB. We can't synthetically force the
    // probe to fail (it's behind a process-wide OnceLock cache), so
    // this test is conditional: only assert when the cache happens
    // to be `None` (i.e. on a non-NVIDIA host or a host without
    // nvidia-smi).
    let probe = live_vram_probe_bytes();
    with_cap(None, || {
        let cap = vram_cap_bytes();
        match probe {
            None => assert_eq!(
                cap,
                8 * 1024 * 1024 * 1024,
                "no env, no probe → cap must be exactly 8 GiB"
            ),
            Some(p) => assert_eq!(
                cap, p,
                "no env, with probe → cap must equal probed value, got cap={cap} probe={p}"
            ),
        }
    });
}

#[test]
fn env_var_override_wins_over_probe_and_default() {
    // The env var path is the highest-priority layer. Even on a
    // machine with a working probe, an explicit cap wins.
    let explicit: usize = 4 * 1024 * 1024 * 1024; // 4 GiB
    with_cap(Some(&explicit.to_string()), || {
        assert_eq!(vram_cap_bytes(), explicit);
    });
    let explicit2: usize = 17 * 1024 * 1024 * 1024; // 17 GiB
    with_cap(Some(&explicit2.to_string()), || {
        assert_eq!(vram_cap_bytes(), explicit2);
    });
}

#[test]
fn probe_is_cached_process_wide() {
    // Multiple calls to `live_vram_probe_bytes` MUST return the same
    // value (the OnceLock is the contract). This matters because
    // nvidia-smi is slow (~50–200 ms) and a per-call probe would
    // destroy throughput on sweeps.
    let a = live_vram_probe_bytes();
    let b = live_vram_probe_bytes();
    let c = live_vram_probe_bytes();
    assert_eq!(a, b, "probe value drifted between calls A→B");
    assert_eq!(b, c, "probe value drifted between calls B→C");
}

#[test]
fn probe_value_includes_safety_factor() {
    // When the probe returns Some, the returned value should be
    // smaller than the GPU's RAW free memory reported by nvidia-smi
    // (we apply a 10% safety factor). We can verify the relationship
    // without depending on the absolute value: query nvidia-smi
    // directly, parse, compare.
    let probe = live_vram_probe_bytes();
    let Some(probed) = probe else {
        // No GPU / no nvidia-smi → can't test the safety factor.
        return;
    };
    // Query raw bytes directly. If it fails, just skip — the probe
    // function already returned Some, so this is purely a
    // verification, not a hard requirement.
    let out = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return,
    };
    let s = String::from_utf8_lossy(&out.stdout);
    let raw_mb: u64 = match s.lines().next().and_then(|l| l.trim().parse().ok()) {
        Some(v) => v,
        None => return,
    };
    let raw_bytes = (raw_mb as usize) * 1024 * 1024;
    // The probed value must be SMALLER than raw_bytes (safety factor
    // shaved off ~10%). Allow a generous band — the free memory can
    // also shift between the cached probe and this re-query, so we
    // just check `probed < raw_bytes` and `probed > raw_bytes / 2`
    // (probed shouldn't have lost more than half the memory).
    assert!(probed < raw_bytes, "probed {probed} not < raw {raw_bytes}");
    assert!(
        probed > raw_bytes / 2,
        "probed {probed} too far below raw {raw_bytes}"
    );
}
