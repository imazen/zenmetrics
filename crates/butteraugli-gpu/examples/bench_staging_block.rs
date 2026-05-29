//! task142 — measure the cost of `reserve_staging`'s `submit_blocking`
//! round-trip in butteraugli-gpu's production upload path.
//!
//! The cubecl maintainer's PR #1334 review: "submit_blocking is
//! expensive and shouldn't be used for data transfer, only for data
//! fetching." Our `reserve_staging` (cubecl-runtime client.rs) wraps
//! `server.staging(...)` in `submit_blocking`, which blocks the caller
//! thread on a full round-trip to the device-runner thread per call.
//!
//! butteraugli-gpu uses it in production (pipeline.rs:782, 1332, 1657)
//! to pack sRGB bytes directly into pinned host memory. This bench
//! quantifies the per-upload block cost and whether it serializes
//! pipelined uploads.
//!
//! Run with:
//!   cargo run --release -p butteraugli-gpu --features cuda \
//!     --example bench_staging_block

use butteraugli_gpu::Butteraugli;
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use std::time::Instant;

const W: u32 = 4096;
const H: u32 = 4096; // 16 MP
const N_UPLOADS: usize = 8;

fn make_img(seed: usize, n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| ((i.wrapping_mul(17).wrapping_add(seed * 7 + 5)) & 0xff) as u8)
        .collect()
}

fn pct(times: &[std::time::Duration]) -> (f64, f64, f64) {
    let mut ms: Vec<f64> = times.iter().map(|t| t.as_secs_f64() * 1e3).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = ms.iter().sum();
    let mean = sum / ms.len() as f64;
    let median = ms[ms.len() / 2];
    let max = *ms.last().unwrap();
    (mean, median, max)
}

fn main() {
    let n_pixels = (W as usize) * (H as usize);
    let n_bytes = n_pixels * 3;
    println!(
        "task142 staging-block bench: {W}x{H} = {} MP, N={N_UPLOADS} uploads",
        n_pixels / 1_000_000
    );
    println!("pinned staging buffer per upload = {} bytes (u32-packed)", n_pixels * 4);

    let client = CudaRuntime::client(&Default::default());

    // Pre-make N distinct source images so the pack work isn't cache-trivial.
    let imgs: Vec<Vec<u8>> = (0..N_UPLOADS).map(|s| make_img(s, n_bytes)).collect();

    // ---------------------------------------------------------------
    // (A) Isolated reserve_staging round-trip latency.
    //     This is the pure cost of the submit_blocking round-trip:
    //     enqueue a shim on the runner thread + block on the oneshot
    //     until the server allocates a pinned Bytes and hands it back.
    // ---------------------------------------------------------------
    let pinned_len = n_pixels * 4;
    // warmup
    for _ in 0..3 {
        let _ = client.reserve_staging(&[pinned_len]);
    }
    let mut t_reserve = Vec::with_capacity(N_UPLOADS);
    for _ in 0..N_UPLOADS {
        let t = Instant::now();
        let staging = client.reserve_staging(&[pinned_len]);
        t_reserve.push(t.elapsed());
        drop(staging);
    }
    let (m, md, mx) = pct(&t_reserve);
    println!(
        "\n(A) reserve_staging() round-trip only: mean={m:.3}ms median={md:.3}ms max={mx:.3}ms"
    );

    // ---------------------------------------------------------------
    // (B) Full production upload path via pack_srgb_into_packed_u32_handle:
    //     reserve_staging (BLOCK) + pack u8x3->u32 + create (async submit).
    //     This is exactly what pipeline.rs does per image side.
    // ---------------------------------------------------------------
    let mut b = Butteraugli::<CudaRuntime>::new(client.clone(), W, H);
    // warmup
    for img in imgs.iter().take(2) {
        let _ = b.pack_srgb_into_packed_u32_handle(img).expect("pack");
    }
    cubecl::future::block_on(client.sync()).expect("sync");
    let mut t_pack = Vec::with_capacity(N_UPLOADS);
    let mut handles = Vec::with_capacity(N_UPLOADS);
    let t_pack_total = Instant::now();
    for img in imgs.iter() {
        let t = Instant::now();
        let h = b.pack_srgb_into_packed_u32_handle(img).expect("pack");
        t_pack.push(t.elapsed());
        handles.push(h);
    }
    let pack_wall = t_pack_total.elapsed();
    cubecl::future::block_on(client.sync()).expect("sync"); // ensure all uploads landed
    let pack_wall_synced = t_pack_total.elapsed();
    let (m, md, mx) = pct(&t_pack);
    println!(
        "\n(B) pack_srgb_into_packed_u32_handle (reserve_staging block + pack + create):"
    );
    println!("    per-call (caller-thread, pre-sync): mean={m:.3}ms median={md:.3}ms max={mx:.3}ms");
    println!("    {N_UPLOADS} uploads wall (pre-sync) = {:.3}ms", pack_wall.as_secs_f64() * 1e3);
    println!("    {N_UPLOADS} uploads wall (post-sync)= {:.3}ms", pack_wall_synced.as_secs_f64() * 1e3);
    drop(handles);

    // ---------------------------------------------------------------
    // (C) Standard create_from_slice path (no pinned helper):
    //     to_vec (pageable) + internal staging (BLOCK once) copies
    //     pageable->pinned + create. We u32-pack on the host first to
    //     match the byte volume of (B).
    // ---------------------------------------------------------------
    let packed: Vec<Vec<u8>> = imgs
        .iter()
        .map(|img| {
            let mut out = vec![0u8; pinned_len];
            for (c, t) in out.chunks_exact_mut(4).zip(img.chunks_exact(3)) {
                c[0] = t[0];
                c[1] = t[1];
                c[2] = t[2];
            }
            out
        })
        .collect();
    // warmup
    for p in packed.iter().take(2) {
        let _ = client.create_from_slice(p);
    }
    cubecl::future::block_on(client.sync()).expect("sync");
    let mut t_slice = Vec::with_capacity(N_UPLOADS);
    let mut handles2 = Vec::with_capacity(N_UPLOADS);
    let t_slice_total = Instant::now();
    for p in packed.iter() {
        let t = Instant::now();
        let h = client.create_from_slice(p);
        t_slice.push(t.elapsed());
        handles2.push(h);
    }
    let slice_wall = t_slice_total.elapsed();
    cubecl::future::block_on(client.sync()).expect("sync");
    let slice_wall_synced = t_slice_total.elapsed();
    let (m, md, mx) = pct(&t_slice);
    println!("\n(C) client.create_from_slice (pageable->pinned internal staging block + create):");
    println!("    per-call (caller-thread, pre-sync): mean={m:.3}ms median={md:.3}ms max={mx:.3}ms");
    println!("    {N_UPLOADS} uploads wall (pre-sync) = {:.3}ms", slice_wall.as_secs_f64() * 1e3);
    println!("    {N_UPLOADS} uploads wall (post-sync)= {:.3}ms", slice_wall_synced.as_secs_f64() * 1e3);
    drop(handles2);

    // ---------------------------------------------------------------
    // (D) Pipelined pattern: interleave upload + compute_handles so the
    //     runner thread is busy with compute when the next upload's
    //     reserve_staging block fires. If the block serializes, the
    //     caller stalls waiting for the runner to drain compute before
    //     it can hand back the pinned buffer.
    // ---------------------------------------------------------------
    let ref_h = b.pack_srgb_into_packed_u32_handle(&imgs[0]).expect("ref");
    // warmup compute
    for img in imgs.iter().take(2) {
        let dh = b.pack_srgb_into_packed_u32_handle(img).expect("dist");
        let _ = b.compute_handles(&ref_h, &dh).expect("compute");
    }
    cubecl::future::block_on(client.sync()).expect("sync");
    let t_pipe = Instant::now();
    let mut last = 0.0f32;
    for img in imgs.iter() {
        let dh = b.pack_srgb_into_packed_u32_handle(img).expect("dist");
        let res = b.compute_handles(&ref_h, &dh).expect("compute");
        last = res.score;
    }
    cubecl::future::block_on(client.sync()).expect("sync");
    let pipe_wall = t_pipe.elapsed();
    println!("\n(D) pipelined upload+compute_handles ({N_UPLOADS} pairs, post-sync):");
    println!("    total wall = {:.3}ms  ({:.3}ms/pair)  last_score={last:.4}",
        pipe_wall.as_secs_f64() * 1e3,
        pipe_wall.as_secs_f64() * 1e3 / N_UPLOADS as f64);

    // ---------------------------------------------------------------
    // (E) Runner-busy probe — the maintainer's real worst case.
    //     Queue a full compute (lots of kernels) WITHOUT syncing, then
    //     immediately fire reserve_staging. Because reserve_staging uses
    //     submit_blocking with SEND_FLUSH, the caller blocks on the
    //     oneshot until the runner thread finishes whatever it is doing
    //     and services the staging request. If the runner is mid-compute
    //     this measures the stall the maintainer warned about: the
    //     caller cannot proceed to pack the next image until the runner
    //     drains the queued compute up to the staging task.
    //
    //     NOTE: cubecl submits kernel *launches* async to the GPU but
    //     the runner thread itself processes the submit() closures
    //     quickly (it just enqueues GPU work); the GPU runs async. So
    //     "runner busy" means the CPU-side runner thread is busy
    //     building/dispatching launches, not the GPU being busy.
    // ---------------------------------------------------------------
    let dh = b.pack_srgb_into_packed_u32_handle(&imgs[1]).expect("dist");
    let mut t_busy = Vec::with_capacity(N_UPLOADS);
    for img in imgs.iter() {
        // Queue compute work (kernel launches enqueued on the runner)
        // but do NOT sync — the readback inside compute_handles DOES
        // sync, so instead we queue raw pack uploads to keep the runner
        // thread busy with submit() closures, then time a reserve.
        let _h0 = b.pack_srgb_into_packed_u32_handle(img).expect("busy-pack");
        let _h1 = b.pack_srgb_into_packed_u32_handle(img).expect("busy-pack");
        // Now fire a reserve_staging while submit() closures may still be
        // queued on the runner thread.
        let t = Instant::now();
        let s = client.reserve_staging(&[pinned_len]);
        t_busy.push(t.elapsed());
        drop(s);
    }
    let _ = &dh;
    cubecl::future::block_on(client.sync()).expect("sync");
    let (m, md, mx) = pct(&t_busy);
    println!("\n(E) reserve_staging() with runner-thread loaded by prior uploads:");
    println!("    round-trip: mean={m:.3}ms median={md:.3}ms max={mx:.3}ms");
}
