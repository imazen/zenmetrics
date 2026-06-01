//! THROWAWAY FEASIBILITY SPIKE (task #153) — NOT part of the public API.
//!
//! Proves or refutes per-stream VRAM pool isolation in the cubecl **wgpu**
//! backend (the backend behind Metal / Vulkan / DX12). This is the
//! load-bearing question for whether `MetricSession`'s per-stream pool
//! isolation — proven on CUDA in task #152
//! (`vram_isolation_spike.rs`) — also holds on wgpu/Metal, or whether
//! `MetricSession` must fall back to best-effort `release()` there.
//! See `crates/zenmetrics-api/docs/VRAM_LIFECYCLE_DESIGN_2026-05-30.md`
//! and issue imazen/zenmetrics#17 (open item).
//!
//! ## HARDWARE CAVEAT (read first)
//! This spike runs on a WSL2 / Windows host with an NVIDIA card. There is
//! **NO Apple GPU here**, so Metal cannot run. wgpu selects Vulkan or DX12
//! automatically on this host (the selected backend is printed at runtime
//! via `client.info()`). cubecl-wgpu's memory layer
//! (`WgpuMemManager` + the `SchedulerMultiStream` stream pool) is
//! **backend-agnostic within wgpu** — the exact same code serves
//! Metal/Vulkan/DX12. So a Vulkan/DX12 result here is the load-bearing
//! proxy for Metal: if isolation works through cubecl-wgpu's abstraction
//! it works on Metal; if it doesn't, Metal is out regardless.
//! **Metal *hardware* confirmation needs a Mac — this spike does NOT and
//! must NOT fabricate Metal numbers.**
//!
//! Mechanism under test (file:line in the published
//! `zenforks-cubecl-* 0.10.1` that this crate depends on — verified
//! against the fork source at `~/work/zenforks-cubecl-work/`):
//!   - `ComputeClient::set_stream(StreamId)` binds a client to an
//!     explicit stream (cubecl-runtime client.rs:97). This is the SAME
//!     backend-agnostic `ComputeClient<R>` the CUDA spike used — it is
//!     defined once in cubecl-runtime, not per backend. An unbound client
//!     falls back to `StreamId::current()` (the calling thread's
//!     thread-local; stream_id.rs).
//!   - Each wgpu `WgpuStream` owns its OWN `WgpuMemManager`
//!     (cubecl-wgpu stream.rs:30), which owns THREE
//!     `MemoryManagement<WgpuStorage>` pools (pool / staging / uniforms,
//!     cubecl-wgpu mem_manager.rs:20-22). So an isolated stream → an
//!     isolated set of pools.
//!   - Streams are stored in a fixed `StreamPool` indexed by
//!     `stream_id.value % max_streams`
//!     (cubecl-runtime stream/base.rs:101-103). `max_streams` default
//!     = 128 (cubecl-runtime config/streaming.rs:24). Distinct
//!     non-aliasing `StreamId` values → distinct `WgpuStream` → distinct
//!     pools.
//!   - `client.memory_cleanup()` -> server.rs:434 ->
//!     `stream.mem_manage.memory_cleanup(true)` -> `cleanup(explicit=true)`
//!     frees only fully-free pages (`amount_free == amount_total`,
//!     sliced_pool.rs:118) via `WgpuStorage::dealloc` (storage.rs:113),
//!     which drops the underlying `wgpu::Buffer`. NOTE: unlike CUDA there
//!     is no explicit `cuMemFreeAsync` + sync — wgpu reclaims the buffer
//!     lazily (storage.rs `flush`: "We don't wait for dealloc").
//!   - `client.memory_usage()` -> server.rs:428 ->
//!     `stream.mem_manage.memory_usage()` reports the PER-STREAM pool's
//!     `MemoryUsage { bytes_in_use, bytes_reserved, .. }`
//!     (memory_management/base.rs). This is the in-API truth of what THIS
//!     stream's pool holds.
//!
//! ## MEASUREMENT
//! The #133 GPU sweep found nvidia-smi cannot see wgpu/Vulkan allocations
//! per-PID (`--query-compute-apps` misses them). So the PRIMARY signal
//! here is cubecl-wgpu's own per-stream `memory_usage().bytes_reserved`
//! — the pool-level truth. We ALSO sample nvidia-smi *total* card
//! `memory.used` (global, all PIDs) and report whether it tracks wgpu
//! allocations at the card level. If only the pool-level number moves,
//! we say so explicitly: pool-level isolation (stream A's pool reports 0
//! reserved after cleanup while stream B's pool stays resident) is strong
//! evidence; driver-level VRAM return is the stronger claim we may not be
//! able to confirm on wgpu on this host.
//!
//! Run:
//!   cargo run --release -p cvvdp-gpu \
//!       --no-default-features --features wgpu,cubecl-types \
//!       --example wgpu_isolation_spike
//!
//! No extrapolation: every number printed is a measured sample.

#![cfg(feature = "wgpu")]

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::stream_id::StreamId;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

type Client = ComputeClient<WgpuRuntime>;

/// Bind a CLONE of the default client to an explicit stream id so its
/// allocations + cleanup land on a private per-stream pool.
fn client_on_stream(value: u64) -> Client {
    let mut c = WgpuRuntime::client(&WgpuDevice::default());
    // SAFETY: documented unsafe in cubecl; we use a distinct `value`
    // per context, all < max_streams (128), so no stream_index alias.
    unsafe { c.set_stream(StreamId { value }) };
    c
}

/// nvidia-smi *total* used-MiB for the first GPU. Process-wide, global
/// card counter. May or may not reflect wgpu/Vulkan allocations (see
/// #133); printed for transparency either way. Returns None if nvidia-smi
/// is unavailable (e.g. a non-NVIDIA host).
fn vram_used_mib() -> Option<i64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse().ok()
}

/// The per-stream POOL reserved bytes, in MiB. This is the in-API truth of
/// what `c`'s stream pool holds on the device, queried AFTER a sync so any
/// pending flush/cleanup has drained.
fn pool_reserved_mib(c: &Client) -> i64 {
    let _ = cubecl::future::block_on(c.sync());
    match c.memory_usage() {
        Ok(u) => (u.bytes_reserved / (1024 * 1024)) as i64,
        Err(e) => {
            eprintln!("  (memory_usage error: {e:?})");
            -1
        }
    }
}

/// In-use bytes (active slices, no padding/reserve), in MiB.
fn pool_in_use_mib(c: &Client) -> i64 {
    match c.memory_usage() {
        Ok(u) => (u.bytes_in_use / (1024 * 1024)) as i64,
        Err(_) => -1,
    }
}

struct Sample {
    pool_reserved: i64,
    smi_total: Option<i64>,
}

fn sample(c: &Client, label: &str, base_smi: Option<i64>) -> Sample {
    // Drain the per-stream queue + settle nvidia-smi accounting.
    let _ = cubecl::future::block_on(c.sync());
    std::thread::sleep(std::time::Duration::from_millis(150));
    let pool_reserved = pool_reserved_mib(c);
    let pool_in_use = pool_in_use_mib(c);
    let smi_total = vram_used_mib();
    let smi_delta = match (smi_total, base_smi) {
        (Some(t), Some(b)) => format!("{:+}", t - b),
        _ => "n/a".to_string(),
    };
    println!(
        "  [{label:<34}] pool_reserved={pool_reserved:>6} MiB  in_use={pool_in_use:>6} MiB  | smi_total_Δ={smi_delta} MiB"
    );
    Sample {
        pool_reserved,
        smi_total,
    }
}

/// Allocate `n` device buffers of `chunk_bytes` each on `c`'s stream,
/// fill them (write forces real backing pages), return the handles so the
/// caller controls their lifetime (drop == return to pool free list, NOT
/// necessarily to the driver).
fn alloc_buffers(c: &Client, n: usize, chunk_bytes: usize) -> Vec<cubecl::server::Handle> {
    let payload = vec![0xA5u8; chunk_bytes];
    (0..n).map(|_| c.create_from_slice(&payload)).collect()
}

fn main() {
    println!("== cubecl WGPU per-stream VRAM isolation spike (task #153) ==");

    // Identify which wgpu backend got selected on this host.
    // `info()` returns `&wgpu::Backend`; the enum is `Copy`, so deref-copy
    // it into an owned value before dropping the probe client.
    let probe = WgpuRuntime::client(&WgpuDevice::default());
    let backend = *probe.info();
    let name = WgpuRuntime::name(&probe);
    println!("   wgpu backend selected: {backend:?}   runtime name: {name}");
    println!("   max_streams=128, each StreamId.value maps to its own WgpuStream + pool");
    println!("   PRIMARY signal = per-stream pool memory_usage().bytes_reserved (in-API truth).");
    println!(
        "   SECONDARY signal = nvidia-smi TOTAL card memory.used (global; may miss Vulkan per #133).\n"
    );
    drop(probe);

    let base_smi = vram_used_mib();
    match base_smi {
        Some(b) => println!("  baseline nvidia-smi total used = {b} MiB (global card)"),
        None => println!("  nvidia-smi unavailable — pool-level signal only"),
    }

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
    let after_a = sample(&ca, "A allocated (A-pool view)", base_smi);

    // ---- Context B on explicit stream 202 -------------------------------
    println!("-- create context B (stream value=202), allocate --");
    let cb = client_on_stream(202);
    let hb = alloc_buffers(&cb, N, CHUNK);
    let after_b_on_b = sample(&cb, "A+B allocated (B-pool view)", base_smi);
    // Cross-check: A's pool should STILL report ~A's footprint (B's allocs
    // landed in B's pool, not A's). This is the per-stream isolation of
    // the *reserved* accounting.
    let a_pool_while_b_alive = sample(&ca, "  (A-pool view, both alive)", base_smi);

    // ---- Drop A, cleanup A's stream -------------------------------------
    println!("-- drop A's handles + A.memory_cleanup() + sync --");
    drop(ha);
    ca.memory_cleanup();
    let after_drop_a = sample(&ca, "after drop+cleanup A (A-pool)", base_smi);
    // B must be untouched: sample B's pool too.
    let b_still = sample(&cb, "  (B-pool view, B alive)", base_smi);

    // ---- Drop B, cleanup B's stream -------------------------------------
    println!("-- drop B's handles + B.memory_cleanup() + sync --");
    drop(hb);
    cb.memory_cleanup();
    let after_drop_b = sample(&cb, "after drop+cleanup B (B-pool)", base_smi);

    // ---- CONTROL: shared-page partial occupancy on ONE stream -----------
    // Two small allocations may be co-resident on a single pool page;
    // dropping one should NOT free the page (partial occupancy), so
    // cleanup returns ~nothing while the other handle is alive.
    println!("\n-- CONTROL: two small allocs may share a page, drop one --");
    let cc = client_on_stream(303);
    let small = 1024 * 1024; // 1 MiB each
    let s1 = cc.create_from_slice(&vec![1u8; small]);
    let s2 = cc.create_from_slice(&vec![2u8; small]);
    let ctrl_both = sample(&cc, "control: both small alive", base_smi);
    drop(s1);
    cc.memory_cleanup();
    let ctrl_after_drop_one = sample(&cc, "control: dropped 1of2 + cleanup", base_smi);
    drop(s2);
    cc.memory_cleanup();
    let ctrl_after_drop_both = sample(&cc, "control: dropped 2of2 + cleanup", base_smi);

    // ---- THREAD MOBILITY: explicit stream overrides thread_local --------
    // Alloc on stream 404 from a *worker thread*, hand the handles back to
    // the main thread, then drop + cleanup stream 404 FROM THE MAIN
    // THREAD. If the explicit stream_id overrides the thread-local, the
    // main thread reclaims the worker thread's allocations -> the context
    // is thread-mobile (Send-friendly). NOTE: wgpu StreamId is also
    // thread-derived by default (stream_id.rs), so this confirms the
    // explicit override path behaves the same as CUDA.
    println!("\n-- THREAD MOBILITY: alloc on worker thread, reclaim on main --");
    let handles_from_worker = std::thread::spawn(|| {
        let cw = client_on_stream(404);
        let h = alloc_buffers(&cw, N, CHUNK);
        let _ = cubecl::future::block_on(cw.sync());
        h
    })
    .join()
    .expect("worker thread ok");
    // Bind a main-thread client to stream 404 to observe its pool.
    let c_main_404 = client_on_stream(404);
    let after_worker_alloc = sample(&c_main_404, "worker-thread alloc (stream 404)", base_smi);
    drop(handles_from_worker);
    c_main_404.memory_cleanup();
    let after_main_reclaim = sample(&c_main_404, "main-thread reclaim of 404", base_smi);
    let cross_thread_freed = after_worker_alloc.pool_reserved - after_main_reclaim.pool_reserved;
    println!(
        "  cross-thread reclaim freed {cross_thread_freed:+} MiB pool-reserved (target ~{per_ctx_mib}) -> {}",
        if cross_thread_freed > per_ctx_mib / 2 {
            "THREAD-MOBILE: explicit stream_id overrides thread-local"
        } else {
            "NOT thread-mobile (or pool accounting differs)"
        }
    );

    // ---- Verdict --------------------------------------------------------
    println!("\n== VERDICT (pool-level, in-API) ==");
    let a_grew = after_a.pool_reserved; // A-pool reserved after A alloc
    let b_added_on_b = after_b_on_b.pool_reserved; // B-pool reserved after B alloc
    let a_pool_independent = a_pool_while_b_alive.pool_reserved; // A-pool while B alive
    let a_freed = a_pool_while_b_alive.pool_reserved - after_drop_a.pool_reserved;
    let b_resident_after_a_drop = b_still.pool_reserved;
    let b_freed = b_still.pool_reserved - after_drop_b.pool_reserved;
    println!("  A-pool reserved after A alloc        {a_grew:+} MiB (target ~{per_ctx_mib})");
    println!("  B-pool reserved after B alloc        {b_added_on_b:+} MiB (target ~{per_ctx_mib})");
    println!(
        "  A-pool reserved while B alive        {a_pool_independent:+} MiB (should ≈ A only, NOT A+B) -> per-stream accounting"
    );
    println!("  drop+cleanup A freed (A-pool)        {a_freed:+} MiB  <- isolated reclaim of A");
    println!(
        "  B-pool still reserved after A drop   {b_resident_after_a_drop:+} MiB (B untouched by A.cleanup)"
    );
    println!("  drop+cleanup B freed (B-pool)        {b_freed:+} MiB  <- isolated reclaim of B");

    // Pool-level isolation holds iff:
    //  (1) A's pool reported ~A's footprint while B was alive (NOT A+B) —
    //      i.e. B's allocs did not show up in A's pool accounting,
    //  (2) dropping+cleaning A freed ~A's footprint FROM A's pool,
    //  (3) B's pool stayed resident (~B's footprint) through A's cleanup,
    //  (4) dropping+cleaning B then freed ~B's footprint from B's pool.
    let a_accounting_isolated =
        a_pool_independent > per_ctx_mib / 2 && a_pool_independent < per_ctx_mib * 3 / 2;
    let a_reclaimed = a_freed > per_ctx_mib / 2;
    let b_survived_a = b_resident_after_a_drop > per_ctx_mib / 2;
    let b_reclaimed = b_freed > per_ctx_mib / 2;
    println!(
        "\n  POOL-LEVEL ISOLATION: {}",
        if a_accounting_isolated && a_reclaimed && b_survived_a && b_reclaimed {
            "CONFIRMED — A's pool freed independently while B's pool stayed resident, then B freed"
        } else {
            "NOT confirmed — see samples above"
        }
    );

    // Driver-level (nvidia-smi total) corroboration — best-effort.
    match (after_b_on_b.smi_total, base_smi, after_drop_b.smi_total) {
        (Some(peak), Some(b), Some(end)) => {
            let peak_delta = peak - b;
            let end_delta = end - b;
            println!(
                "  DRIVER-LEVEL (nvidia-smi total): peak Δ={peak_delta:+} MiB, after both freed Δ={end_delta:+} MiB"
            );
            if peak_delta > per_ctx_mib {
                println!(
                    "    -> nvidia-smi total DID track wgpu allocs at the card level (peak ≥ one context)."
                );
                if end_delta < peak_delta / 2 {
                    println!(
                        "    -> and the card returned most of it after cleanup (driver-level reclaim observable)."
                    );
                } else {
                    println!(
                        "    -> card did NOT visibly return it after cleanup (wgpu defers buffer free; lazy reclaim)."
                    );
                }
            } else {
                println!(
                    "    -> nvidia-smi total did NOT clearly track wgpu allocs (per-PID invisibility per #133 extends to total here). Pool-level signal is the load-bearing one."
                );
            }
        }
        _ => println!("  DRIVER-LEVEL: nvidia-smi unavailable; pool-level signal only."),
    }

    println!(
        "\n  CONTROL partial-page: drop-1of2 freed {} MiB, drop-2of2 freed {} MiB",
        ctrl_both.pool_reserved - ctrl_after_drop_one.pool_reserved,
        ctrl_after_drop_one.pool_reserved - ctrl_after_drop_both.pool_reserved
    );
    println!("    (expect ~0 after dropping only 1 of 2 co-resident allocs)");

    println!(
        "\n  METAL CAVEAT: this ran on {backend:?} (NO Apple GPU on this host). cubecl-wgpu's\n  WgpuMemManager + SchedulerMultiStream pool code is backend-agnostic within wgpu, so\n  this Vulkan/DX12 result is the load-bearing proxy for Metal. Metal HARDWARE confirmation\n  needs a Mac — no Metal numbers are fabricated here."
    );
}
