//! THROWAWAY FEASIBILITY SPIKE (task #152) — NOT part of the public API.
//!
//! Proves or refutes per-stream VRAM pool isolation in the cubecl CUDA
//! backend, which is the load-bearing question for the
//! "new context type owns an isolated stream+pool" design option
//! (`crates/zenmetrics-api/docs/VRAM_LIFECYCLE_DESIGN_2026-05-30.md`).
//!
//! Mechanism under test (file:line in the published `zenforks-cubecl-*`
//! 0.10.1 that this crate depends on):
//!   - `ComputeClient::set_stream(StreamId)` binds a client to an
//!     explicit stream (client.rs:97). An unbound client falls back to
//!     `StreamId::current()` (the calling thread's thread-local).
//!   - Each CUDA `Stream` owns its OWN `MemoryManagement<GpuStorage>`
//!     and a fresh `cuStreamCreate` stream (cuda/.../stream.rs:25,49).
//!   - Streams are stored in a fixed pool indexed by
//!     `stream_id.value % max_streams` (runtime stream/event.rs:143).
//!     `max_streams` default = 128 (config/streaming.rs:24).
//!   - `client.memory_cleanup()` -> `cleanup(explicit=true)` frees only
//!     pages where `amount_free == amount_total`, i.e. fully-free pages
//!     (sliced_pool.rs:118), then `sync()` flushes `cuMemFreeAsync`
//!     (cuda gpu.rs perform_deallocations).
//!
//! Run:
//!   cargo run --release -p cvvdp-gpu --features cuda,cubecl-types \
//!       --example vram_isolation_spike
//!
//! All VRAM samples are taken from `nvidia-smi --query-gpu=memory.used`
//! AFTER a `client.sync()` so the deferred free queue has drained.
//! No extrapolation: every number printed is a measured sample.

#![cfg(feature = "cuda")]

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::cuda::CudaRuntime;
use cubecl::stream_id::StreamId;

type Client = ComputeClient<CudaRuntime>;

/// Bind a CLONE of the default client to an explicit stream id so its
/// allocations + cleanup land on a private per-stream pool.
fn client_on_stream(value: u64) -> Client {
    let mut c = CudaRuntime::client(&Default::default());
    // SAFETY: documented unsafe in cubecl; we use a distinct `value`
    // per context, all < max_streams (128), so no stream_index alias.
    unsafe { c.set_stream(StreamId { value }) };
    c
}

/// nvidia-smi used-MiB for the first (only) GPU. Process-wide counter.
fn vram_used_mib() -> i64 {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .expect("nvidia-smi runs");
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next().unwrap().trim().parse().unwrap()
}

fn sample(c: &Client, label: &str, base: i64) -> i64 {
    // Drain the per-stream deferred-free queue before sampling.
    let _ = cubecl::future::block_on(c.sync());
    // Small settle; the driver updates nvidia-smi accounting async.
    std::thread::sleep(std::time::Duration::from_millis(150));
    let used = vram_used_mib();
    println!(
        "  [{label:<34}] used={used:>6} MiB  (Δbase {:+} MiB)",
        used - base
    );
    used
}

/// Allocate `n` device buffers of `chunk_bytes` each on `c`'s stream,
/// fill them (write forces real backing pages), return the handles so
/// the caller controls their lifetime (drop == return to pool free
/// list, NOT to the driver).
fn alloc_buffers(c: &Client, n: usize, chunk_bytes: usize) -> Vec<cubecl::server::Handle> {
    let payload = vec![0xA5u8; chunk_bytes];
    (0..n).map(|_| c.create_from_slice(&payload)).collect()
}

fn main() {
    println!("== cubecl CUDA per-stream VRAM isolation spike (task #152) ==");
    println!("   max_streams=128, each StreamId.value maps to its own pool\n");

    let base = vram_used_mib();
    println!("  baseline (no cubecl ctx)            used={base:>6} MiB");

    // ~1.5 GiB per context: 24 x 64 MiB chunks. Large multi-chunk so the
    // pool spans multiple fully-free-able pages on drop+cleanup.
    const CHUNK: usize = 64 * 1024 * 1024;
    const N: usize = 24;
    let per_ctx_mib = (CHUNK * N / (1024 * 1024)) as i64;
    println!("  per-context target ~= {per_ctx_mib} MiB ({N} x 64 MiB)\n");

    // ---- Context A on explicit stream 101 -------------------------------
    println!("-- create context A (stream value=101), allocate --");
    let ca = client_on_stream(101);
    let ha = alloc_buffers(&ca, N, CHUNK);
    let after_a = sample(&ca, "A allocated", base);

    // ---- Context B on explicit stream 202 -------------------------------
    println!("-- create context B (stream value=202), allocate --");
    let cb = client_on_stream(202);
    let hb = alloc_buffers(&cb, N, CHUNK);
    let after_b = sample(&cb, "A+B allocated", base);

    // ---- Drop A, cleanup A's stream -------------------------------------
    println!("-- drop A's handles + A.memory_cleanup() + sync --");
    drop(ha);
    ca.memory_cleanup();
    let after_drop_a = sample(&ca, "after drop+cleanup A", base);
    // B must be untouched: sample on B's client too.
    let b_still = sample(&cb, "  (B client view, B alive)", base);

    // ---- Drop B, cleanup B's stream -------------------------------------
    println!("-- drop B's handles + B.memory_cleanup() + sync --");
    drop(hb);
    cb.memory_cleanup();
    let after_drop_b = sample(&cb, "after drop+cleanup B", base);

    // ---- CONTROL: shared-page partial occupancy on ONE stream -----------
    // Two tiny allocations likely co-resident on a single pool page;
    // dropping one should NOT free the page (partial occupancy), so
    // cleanup returns ~nothing while the other handle is alive.
    println!("\n-- CONTROL: two small allocs share a page, drop one --");
    let cc = client_on_stream(303);
    let small = 1024 * 1024; // 1 MiB each; likely same pool page bucket
    let s1 = cc.create_from_slice(&vec![1u8; small]);
    let s2 = cc.create_from_slice(&vec![2u8; small]);
    let ctrl_both = sample(&cc, "control: both small alive", base);
    drop(s1);
    cc.memory_cleanup();
    let ctrl_after_drop_one = sample(&cc, "control: dropped 1of2 + cleanup", base);
    drop(s2);
    cc.memory_cleanup();
    let ctrl_after_drop_both = sample(&cc, "control: dropped 2of2 + cleanup", base);

    // ---- THREAD MOBILITY: explicit stream overrides thread_local --------
    // Alloc on stream 404 from a *worker thread*, hand the handles back
    // to the main thread, then drop + cleanup stream 404 FROM THE MAIN
    // THREAD. If the explicit stream_id overrides the thread-local, the
    // main thread reclaims the worker thread's allocations -> the
    // context is thread-mobile (Send-friendly).
    println!("\n-- THREAD MOBILITY: alloc on worker thread, reclaim on main --");
    let handles_from_worker = std::thread::spawn(|| {
        let cw = client_on_stream(404);
        let h = alloc_buffers(&cw, N, CHUNK);
        let _ = cubecl::future::block_on(cw.sync());
        // Move handles out of the worker thread; cw (client) is dropped
        // here but the device pages live in stream 404's pool.
        h
    })
    .join()
    .expect("worker thread ok");
    let after_worker_alloc = sample(&cc, "worker-thread alloc (stream 404)", base);
    // Main thread binds a fresh client to the SAME stream 404 and cleans.
    let c_main_404 = client_on_stream(404);
    drop(handles_from_worker);
    c_main_404.memory_cleanup();
    let after_main_reclaim = sample(&c_main_404, "main-thread reclaim of 404", base);
    let cross_thread_freed = after_worker_alloc - after_main_reclaim;
    println!(
        "  cross-thread reclaim freed {cross_thread_freed:+} MiB (target ~{per_ctx_mib}) -> {}",
        if cross_thread_freed > per_ctx_mib / 2 {
            "THREAD-MOBILE: explicit stream_id overrides thread-local"
        } else {
            "NOT thread-mobile"
        }
    );

    // ---- Verdict --------------------------------------------------------
    println!("\n== VERDICT ==");
    let a_grew = after_a - base;
    let b_added = after_b - after_a;
    let a_freed = after_b - after_drop_a;
    let b_freed = after_drop_a - after_drop_b;
    // After dropping+cleaning A, B's pool must remain resident. B's
    // resident footprint then is (after_drop_a - base); it should still
    // be ~B's allocation (large), proving cleanup(A) did NOT touch B.
    let b_resident_after_a_drop = after_drop_a - base;
    println!("  A alloc grew base by      {a_grew:+} MiB (target ~{per_ctx_mib})");
    println!("  B alloc added             {b_added:+} MiB (target ~{per_ctx_mib})");
    println!("  drop+cleanup A freed      {a_freed:+} MiB  <- isolated reclaim of A");
    println!(
        "  B still resident after A    {b_resident_after_a_drop:+} MiB (B untouched; B-client view {b_still} MiB)"
    );
    println!("  drop+cleanup B freed      {b_freed:+} MiB  <- isolated reclaim of B");
    // Isolation holds iff: (1) dropping A freed ~A's footprint, AND
    // (2) B stayed resident (≈ its own footprint) through A's cleanup,
    // AND (3) dropping B then freed ~B's footprint.
    let a_reclaimed = a_freed > per_ctx_mib / 2;
    let b_survived_a = b_resident_after_a_drop > per_ctx_mib / 2;
    let b_reclaimed = b_freed > per_ctx_mib / 2;
    println!(
        "  ISOLATION: {}",
        if a_reclaimed && b_survived_a && b_reclaimed {
            "CONFIRMED — A's pool freed to driver independently while B stayed resident, then B freed"
        } else {
            "NOT confirmed — see samples above"
        }
    );
    println!(
        "  CONTROL partial-page: drop-1of2 freed {} MiB, drop-2of2 freed {} MiB",
        ctrl_both - ctrl_after_drop_one,
        ctrl_after_drop_one - ctrl_after_drop_both
    );
    println!("    (expect ~0 after dropping only 1 of 2 co-resident allocs)");
}
