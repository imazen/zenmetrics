//! VRAM no-leak regression guard for butteraugli-gpu (task #147).
//!
//! Asserts that the GPU working set does NOT grow without bound across
//! repeated use of a single `Butteraugli` instance. Three usage patterns
//! are covered, mirroring `examples/vram_leak_check.rs`:
//!
//!   * `many_scores`   — one cached reference, many `compute_with_reference`.
//!   * `many_new_refs` — many DISTINCT `set_reference` calls on one instance
//!                       (the in-place-plane-reuse path #144 measured).
//!   * `create_drop`   — construct → score → drop, repeated. (CubeCL pools
//!                       buffers across Drop, so this asserts a *plateau*,
//!                       not a return to baseline.)
//!
//! **Gating.** Compiled only with the `cuda` feature (a real NVIDIA GPU +
//! nvidia-smi at runtime). On CI without the cuda feature the whole file is
//! cfg'd out — the gate is at the build/feature level (visible in the
//! justfile / CI matrix), NOT a runtime file-existence skip. If the cuda
//! feature is on but no GPU is present, `CudaRuntime::client()` panics —
//! same contract as the other GPU integration tests in this crate
//! (`strip_parity.rs`, `reduction_parity.rs`).
//!
//! **Probe.** `nvidia-smi --query-gpu=memory.used` (global card MiB; per-PID
//! accounting is hidden under WSL2). Each sample is the MIN of several reads
//! so a *transient* allocation by a concurrent GPU process is filtered. The
//! test compares the post-warmup floor to the final floor; a genuine leak
//! pushes the floor up by hundreds-to-thousands of MiB, far above the
//! `MAX_GROWTH_MIB` gate. A concurrent process that allocates *and frees*
//! cannot push the min-floor up monotonically.

#![cfg(feature = "cuda")]

use std::process::Command;
use std::time::Duration;

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use cubecl::prelude::ComputeClient;

use butteraugli_gpu::Butteraugli;

/// Max permitted growth of the VRAM floor between the post-warmup sample
/// and the final sample, in MiB. Healthy reuse is ~0 (the working set is
/// allocated once). A real leak at these sizes adds tens-to-thousands of
/// MiB. 96 MiB leaves generous headroom for CubeCL pool settle + sampling
/// jitter on the shared card while still catching any genuine per-cycle
/// accumulation. Even a tiny ~1 MiB/cycle dist-upload-staging leak would
/// blow past this over N cycles.
const MAX_GROWTH_MIB: i64 = 96;

/// Images are 1 MP. Large enough that a leaked per-call buffer is visible
/// (a leaked dist-staging plane is ~4 MiB; a leaked full plane set is
/// ~12 MiB), small enough to run fast and not contend for VRAM on a busy
/// shared card.
const W: u32 = 1024;
const H: u32 = 1024;

/// Cycles. >= the brief's floor (100 for scores/refs). A per-cycle leak of
/// even 1 MiB accumulates to ~100 MiB over this window — caught by the gate.
const N_SCORES: usize = 120;
const N_NEW_REFS: usize = 120;
/// create_drop reconstructs the whole instance each cycle (expensive); the
/// brief's floor is 30. Keep it at 40 for a clear plateau read.
const N_CREATE_DROP: usize = 40;

fn synth_srgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    use std::num::Wrapping;
    let n = (w as usize) * (h as usize) * 3;
    let mut v = Vec::with_capacity(n);
    let mut s = Wrapping(seed.wrapping_mul(2_654_435_761).wrapping_add(1));
    for _ in 0..n {
        s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
        v.push(((s.0 >> 16) & 0xff) as u8);
    }
    v
}

fn vram_used_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Sync the GPU, let deferred frees land, then sample the card floor
/// (min of `samples` reads). Returns None if nvidia-smi is unavailable.
fn sample_floor(client: &ComputeClient<Backend>, samples: u32) -> Option<u64> {
    cubecl::future::block_on(client.sync()).ok()?;
    std::thread::sleep(Duration::from_millis(60));
    let mut lo = vram_used_mib()?;
    for _ in 1..samples.max(1) {
        std::thread::sleep(Duration::from_millis(8));
        if let Some(v) = vram_used_mib() {
            lo = lo.min(v);
        }
    }
    Some(lo)
}

/// Run `cycles` of `step`, sampling the VRAM floor each cycle. Asserts the
/// floor after warmup (cycle WARM) does not grow by more than
/// `MAX_GROWTH_MIB` by the final cycle. `step(i)` performs one unit of work.
fn assert_no_growth(
    label: &str,
    client: &ComputeClient<Backend>,
    cycles: usize,
    mut step: impl FnMut(usize),
) {
    const WARM: usize = 5;
    assert!(cycles > WARM + 5, "{label}: need more than warmup cycles");
    let samples = 5;
    let mut floors: Vec<u64> = Vec::with_capacity(cycles);
    for i in 0..cycles {
        step(i);
        match sample_floor(client, samples) {
            Some(v) => floors.push(v),
            None => {
                // nvidia-smi unavailable despite the cuda feature — can't
                // probe. The cuda feature implies a real GPU per this
                // crate's test contract, so treat a missing probe as a
                // hard environment error rather than silently passing.
                panic!("{label}: nvidia-smi memory.used query failed (no probe)");
            }
        }
    }
    // Lower envelope (rolling-min, window 5) removes additive transients
    // from a concurrent GPU consumer on the shared card.
    let env: Vec<u64> = (0..floors.len())
        .map(|i| *floors[i.saturating_sub(4)..=i].iter().min().unwrap())
        .collect();
    let post_warm = &env[WARM..];
    let floor_start = *post_warm.first().unwrap() as i64;
    let floor_end = *post_warm.last().unwrap() as i64;
    let floor_min = *post_warm.iter().min().unwrap() as i64;
    let floor_max = *post_warm.iter().max().unwrap() as i64;
    let growth = floor_end - floor_start;
    // Growth measured both end-vs-start and max-vs-min (the latter catches
    // a leak that saturates VRAM and stops climbing before the end).
    let span = floor_max - floor_min;
    eprintln!(
        "{label}: cycles={cycles} floor_start={floor_start} floor_end={floor_end} \
         floor_min={floor_min} floor_max={floor_max} growth={growth} span={span} MiB \
         (gate {MAX_GROWTH_MIB})"
    );
    assert!(
        growth <= MAX_GROWTH_MIB,
        "{label}: VRAM floor grew {growth} MiB over {cycles} cycles \
         (start={floor_start} end={floor_end}); exceeds {MAX_GROWTH_MIB} MiB gate — \
         suspected leak"
    );
    assert!(
        span <= MAX_GROWTH_MIB,
        "{label}: VRAM floor span {span} MiB over {cycles} cycles \
         (min={floor_min} max={floor_max}); exceeds {MAX_GROWTH_MIB} MiB gate — \
         suspected leak"
    );
}

#[test]
fn many_scores_one_instance_no_leak() {
    let client = Backend::client(&Default::default());
    let r = synth_srgb(W, H, 42);
    let mut b = Butteraugli::<Backend>::new(client.clone(), W, H);
    b.set_reference(&r).expect("set_reference");
    assert_no_growth("many_scores", &client, N_SCORES, |i| {
        let d = synth_srgb(W, H, 1000 + i as u32);
        let _ = b
            .compute_with_reference(&d)
            .expect("compute_with_reference");
    });
}

#[test]
fn many_new_references_one_instance_no_leak() {
    // The reuse path #144 flagged: distinct reference content every cycle,
    // on a single instance. set_reference overwrites the planes in place.
    let client = Backend::client(&Default::default());
    let d = synth_srgb(W, H, 137);
    let mut b = Butteraugli::<Backend>::new(client.clone(), W, H);
    assert_no_growth("many_new_refs", &client, N_NEW_REFS, |i| {
        let r = synth_srgb(W, H, 5000 + i as u32);
        b.set_reference(&r).expect("set_reference new ref");
        let _ = b
            .compute_with_reference(&d)
            .expect("compute_with_reference new ref");
    });
}

#[test]
fn create_drop_instances_plateau_no_leak() {
    // CubeCL pools across Drop, so VRAM does not return to baseline — but
    // it must PLATEAU (the pool is bounded), not climb every cycle.
    let client = Backend::client(&Default::default());
    let r = synth_srgb(W, H, 42);
    let d = synth_srgb(W, H, 137);
    assert_no_growth("create_drop", &client, N_CREATE_DROP, |_i| {
        let mut b = Butteraugli::<Backend>::new(client.clone(), W, H);
        let _ = b.compute(&r, &d).expect("compute");
        // b dropped at end of closure.
    });
}
