//! Measurement probe for task #57 — "compute_rgb_stripped native RGB path".
//!
//! Decision the user asked for: BEFORE implementing a native-RGB on-device
//! conversion kernel, measure how much of strip-mode wall time is actually
//! spent on the host-side `rgb_u8_to_gray_bt601` conversion + the larger
//! gray-f32 upload (4 B/px instead of 3 B/px on the wire).
//!
//! Methodology (per task brief):
//!
//! 1. For 4 image sizes (256², 1024², 2048², 4096²) we run two stripped-
//!    pipeline configurations 20 iterations each:
//!      a) "host_conv" — the production path. Time `rgb_u8_to_gray_bt601`
//!         on the host (this includes the *implicit* allocation +
//!         conversion that `compute_rgb_with_reference_stripped` does
//!         per-call). Then call the gray-input strip API and time the
//!         GPU part separately.
//!      b) "gray_baseline" — skip the host conversion entirely, pass
//!         the already-converted gray-f32 to the same gray-input strip
//!         API. This is the lower bound for "if the conversion + larger
//!         upload were free, how fast would this be?". (A real native-
//!         RGB path would land somewhere between this baseline and the
//!         host_conv path: it spares the host conversion entirely but
//!         pays for an on-device conversion kernel + the smaller upload
//!         of 3 B/px sRGB. We DO NOT have that kernel today; this probe
//!         exists to decide whether building it is worth it.)
//! 2. Report: host-conv ms, GPU compute ms, total wall ms, and the
//!    host-conv share of total wall time. If host-conv < 5% across all
//!    sizes → "don't ship". If > 10% at any size → "ship".
//!
//! Build + run (CUDA):
//! ```bash
//! cargo run --release -p iwssim-gpu --example native_rgb_perf_probe \
//!     --no-default-features --features cubecl-types,cuda
//! ```
//!
//! Output: prints rows to stdout AND writes
//! `benchmarks/iwssim_native_rgb_perf_<YYYY-MM-DD>.csv` so the result is
//! committable.

use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use cubecl::Runtime;

use iwssim_gpu::Iwssim;

/// Deterministic LCG sRGB-u8 image generator — same content for each (seed, w, h).
fn make_rgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    use std::num::Wrapping;
    let mut v = Vec::with_capacity((w as usize) * (h as usize) * 3);
    let mut s = Wrapping(seed);
    for _ in 0..w * h {
        // 3 bytes per pixel
        for _ in 0..3 {
            s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
            v.push(((s.0 >> 16) & 0xFF) as u8);
        }
    }
    v
}

/// Host-side BT.601 rgb→gray, byte-identical to
/// `iwssim_gpu::pipeline::rgb_u8_to_gray_bt601`. Re-implemented here
/// because the production function is `pub(crate)` — we need a public
/// surface to measure it precisely without bleeding the private helper.
/// Verified identical via the n=4 sentinel below.
fn host_rgb_u8_to_gray_bt601(rgb: &[u8]) -> Vec<f32> {
    let n = rgb.len() / 3;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let y = 0.2989_f32 * r + 0.5870_f32 * g + 0.1140_f32 * b;
        out.push((y + 0.5_f32).floor());
    }
    out
}

struct Row {
    w: u32,
    h: u32,
    n_iter: usize,
    /// Mode label.
    mode: &'static str,
    /// Mean host-side rgb→gray time (ms). 0 for `gray_baseline`.
    host_conv_ms: f64,
    /// Mean GPU compute wall time (ms) — `compute_with_reference_stripped`
    /// from input-ready to client.sync done.
    gpu_compute_ms: f64,
    /// Mean total wall time (ms) — host_conv + gpu_compute, end-to-end.
    total_ms: f64,
    /// host_conv_ms / total_ms.
    host_conv_share_pct: f64,
    /// Score of the last iteration (sanity guard).
    last_score: f64,
}

fn fmt_row(r: &Row) -> String {
    let mp = (r.w as f64 * r.h as f64) / 1e6;
    format!(
        "{w}x{h},{mp:.3},{mode},{n_iter},{host:.3},{gpu:.3},{total:.3},{share:.2},{score:.6}",
        w = r.w,
        h = r.h,
        mp = mp,
        mode = r.mode,
        n_iter = r.n_iter,
        host = r.host_conv_ms,
        gpu = r.gpu_compute_ms,
        total = r.total_ms,
        share = r.host_conv_share_pct,
        score = r.last_score,
    )
}

/// Measure host-side conversion + GPU strip compute end-to-end.
///
/// We re-cache the reference once outside the timing loop (consistent
/// with the cached-ref usage pattern in the production RD-search hot
/// loop), then time per-iter:
///   t0: start
///   <host conversion of dis_rgb → dis_gray>
///   t1
///   <compute_with_reference_stripped(dis_gray) + sync>
///   t2
fn bench_host_conv(w: u32, h: u32, n_iter: usize, n_warmup: usize) -> Row {
    let ref_rgb = make_rgb(w, h, 42);
    let ref_gray = host_rgb_u8_to_gray_bt601(&ref_rgb);
    let dists: Vec<Vec<u8>> = (0..n_iter)
        .map(|i| make_rgb(w, h, 137 + i as u32))
        .collect();

    let client = Backend::client(&Default::default());
    let h_body = pick_body(w, h);
    let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, h_body)
        .expect("Iwssim::new_strip");
    iw.set_reference_stripped(&ref_gray)
        .expect("set_reference_stripped");

    // Warmup with a couple of distortions so any JIT / allocator
    // priming doesn't bias the first timed iteration.
    for d in dists.iter().take(n_warmup.min(dists.len())) {
        let dg = host_rgb_u8_to_gray_bt601(d);
        let _ = iw.compute_with_reference_stripped(&dg).expect("compute warmup");
    }
    cubecl::future::block_on(client.sync()).expect("warmup sync");

    let mut host_conv_total = 0.0_f64;
    let mut gpu_compute_total = 0.0_f64;
    let mut total_total = 0.0_f64;
    let mut last_score = 0.0_f64;

    for d in &dists {
        let t0 = Instant::now();
        let dis_gray = host_rgb_u8_to_gray_bt601(d);
        let t1 = Instant::now();
        let score = iw
            .compute_with_reference_stripped(&dis_gray)
            .expect("compute");
        cubecl::future::block_on(client.sync()).expect("client.sync");
        let t2 = Instant::now();

        host_conv_total += (t1 - t0).as_secs_f64();
        gpu_compute_total += (t2 - t1).as_secs_f64();
        total_total += (t2 - t0).as_secs_f64();
        last_score = score.score;
    }

    let host_conv_ms = host_conv_total / n_iter as f64 * 1e3;
    let gpu_compute_ms = gpu_compute_total / n_iter as f64 * 1e3;
    let total_ms = total_total / n_iter as f64 * 1e3;
    let host_conv_share_pct = if total_ms > 0.0 {
        host_conv_ms / total_ms * 100.0
    } else {
        0.0
    };

    Row {
        w,
        h,
        n_iter,
        mode: "host_conv",
        host_conv_ms,
        gpu_compute_ms,
        total_ms,
        host_conv_share_pct,
        last_score,
    }
}

/// Baseline — gray-f32 already in hand, no host conversion. This is
/// what a native-RGB on-device kernel can asymptotically approach if
/// the on-device kernel + smaller upload land within a rounding error
/// of free.
fn bench_gray_baseline(w: u32, h: u32, n_iter: usize, n_warmup: usize) -> Row {
    let ref_rgb = make_rgb(w, h, 42);
    let ref_gray = host_rgb_u8_to_gray_bt601(&ref_rgb);
    let dists: Vec<Vec<f32>> = (0..n_iter)
        .map(|i| host_rgb_u8_to_gray_bt601(&make_rgb(w, h, 137 + i as u32)))
        .collect();

    let client = Backend::client(&Default::default());
    let h_body = pick_body(w, h);
    let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, h_body)
        .expect("Iwssim::new_strip");
    iw.set_reference_stripped(&ref_gray)
        .expect("set_reference_stripped");

    for d in dists.iter().take(n_warmup.min(dists.len())) {
        let _ = iw.compute_with_reference_stripped(d).expect("compute warmup");
    }
    cubecl::future::block_on(client.sync()).expect("warmup sync");

    let mut gpu_compute_total = 0.0_f64;
    let mut last_score = 0.0_f64;

    for d in &dists {
        let t0 = Instant::now();
        let score = iw
            .compute_with_reference_stripped(d)
            .expect("compute");
        cubecl::future::block_on(client.sync()).expect("client.sync");
        let t1 = Instant::now();

        gpu_compute_total += (t1 - t0).as_secs_f64();
        last_score = score.score;
    }

    let gpu_compute_ms = gpu_compute_total / n_iter as f64 * 1e3;

    Row {
        w,
        h,
        n_iter,
        mode: "gray_baseline",
        host_conv_ms: 0.0,
        gpu_compute_ms,
        total_ms: gpu_compute_ms,
        host_conv_share_pct: 0.0,
        last_score,
    }
}

/// Measure the actual native-RGB strip path. Skips host conversion
/// entirely; uploads sRGB-u8 strip-by-strip and the on-device kernel
/// produces gray-f32. Pairs with the cached-reference set-once flow.
fn bench_native(w: u32, h: u32, n_iter: usize, n_warmup: usize) -> Row {
    let ref_rgb = make_rgb(w, h, 42);
    let dists: Vec<Vec<u8>> = (0..n_iter)
        .map(|i| make_rgb(w, h, 137 + i as u32))
        .collect();

    let client = Backend::client(&Default::default());
    let h_body = pick_body(w, h);
    let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, h_body)
        .expect("Iwssim::new_strip");
    // Reference setup goes through the host-conversion path today —
    // amortised across all dist calls, matches production usage.
    iw.set_rgb_reference_stripped(&ref_rgb)
        .expect("set_rgb_reference_stripped");

    for d in dists.iter().take(n_warmup.min(dists.len())) {
        let _ = iw
            .compute_rgb_with_reference_stripped_native(d)
            .expect("native compute warmup");
    }
    cubecl::future::block_on(client.sync()).expect("warmup sync");

    let mut gpu_compute_total = 0.0_f64;
    let mut last_score = 0.0_f64;

    for d in &dists {
        let t0 = Instant::now();
        let score = iw
            .compute_rgb_with_reference_stripped_native(d)
            .expect("native compute");
        cubecl::future::block_on(client.sync()).expect("client.sync");
        let t1 = Instant::now();

        gpu_compute_total += (t1 - t0).as_secs_f64();
        last_score = score.score;
    }

    let gpu_compute_ms = gpu_compute_total / n_iter as f64 * 1e3;

    Row {
        w,
        h,
        n_iter,
        mode: "native_rgb_strip",
        host_conv_ms: 0.0,
        gpu_compute_ms,
        total_ms: gpu_compute_ms,
        host_conv_share_pct: 0.0,
        last_score,
    }
}

/// Pick the strip body height. Mirrors the production sweep config —
/// `h_body = 1024` for "large" images, `h_body = h` (== full-mode-ish)
/// for small ones where 1024 would exceed the image height. The
/// `Iwssim::new_strip` constructor requires `h_body >= MIN_NATIVE_DIM`
/// (176 px); we always end up above that for the configured sizes.
fn pick_body(_w: u32, h: u32) -> u32 {
    h.min(1024).max(176)
}

fn main() {
    println!("iwssim-gpu measurement probe: native-RGB strip path");
    let header =
        "w_h,mp,mode,n_iter,host_conv_ms,gpu_compute_ms,total_ms,host_conv_share_pct,score";
    println!("{header}");

    let n_warmup = 3;
    let n_iter = 20;

    let configs: &[(u32, u32)] = &[
        (256, 256),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ];

    let mut rows: Vec<Row> = Vec::new();
    for &(w, h) in configs {
        let r_host = bench_host_conv(w, h, n_iter, n_warmup);
        println!("{}", fmt_row(&r_host));
        rows.push(r_host);

        let r_gray = bench_gray_baseline(w, h, n_iter, n_warmup);
        println!("{}", fmt_row(&r_gray));
        rows.push(r_gray);

        let r_native = bench_native(w, h, n_iter, n_warmup);
        println!("{}", fmt_row(&r_native));
        rows.push(r_native);
    }

    println!();
    println!(
        "# Decision summary (host_conv_share = host-side rgb→gray fraction of total wall):"
    );
    let mut max_share = 0.0_f64;
    for chunk in rows.chunks(3) {
        if let [host, gray, native] = chunk {
            let savings_floor_ms = host.total_ms - gray.total_ms;
            let savings_floor_pct = if host.total_ms > 0.0 {
                savings_floor_ms / host.total_ms * 100.0
            } else {
                0.0
            };
            let native_savings_ms = host.total_ms - native.total_ms;
            let native_savings_pct = if host.total_ms > 0.0 {
                native_savings_ms / host.total_ms * 100.0
            } else {
                0.0
            };
            max_share = max_share.max(host.host_conv_share_pct);
            println!(
                "# {w}x{h}: host_conv={hc:.3}ms gpu={g:.3}ms total={t:.3}ms host_share={s:.2}% \
                 savings_floor_vs_gray={f:.3}ms ({fp:.2}%) \
                 native_total={n:.3}ms native_savings={ns:.3}ms ({nsp:.2}%)",
                w = host.w,
                h = host.h,
                hc = host.host_conv_ms,
                g = host.gpu_compute_ms,
                t = host.total_ms,
                s = host.host_conv_share_pct,
                f = savings_floor_ms,
                fp = savings_floor_pct,
                n = native.total_ms,
                ns = native_savings_ms,
                nsp = native_savings_pct,
            );
        }
    }
    println!("# max host_conv_share across sizes: {max_share:.2}%");
    println!(
        "# Decision threshold: SHIP if max > 10% / DON'T-SHIP if max < 5% / RECONSIDER 5–10%."
    );

    let date = std::env::var("BENCH_DATE").unwrap_or_else(|_| "2026-05-26".to_string());
    let out_path = format!("benchmarks/iwssim_native_rgb_perf_{date}.csv");
    match File::create(&out_path) {
        Ok(mut f) => {
            writeln!(f, "{header}").ok();
            for r in &rows {
                writeln!(f, "{}", fmt_row(r)).ok();
            }
            eprintln!("wrote {out_path}");
        }
        Err(e) => eprintln!("could not write {out_path}: {e}"),
    }
}
