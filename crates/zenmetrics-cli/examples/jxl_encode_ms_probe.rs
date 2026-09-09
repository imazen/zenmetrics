//! #34 probe: where does the sweep's zenjxl `encode_ms` time actually go?
//!
//! The canonical rollup shows zenjxl at a median ~31 s/MP, flat across effort
//! e1..e9, while the encoder measured directly spans 17-68x over that range.
//! The issue models it as `sweep ≈ measured + 31.3 s/MP` — an ADDITIVE,
//! effort-independent, pixel-proportional parasite — and does not identify the
//! mechanism. This harness splits the sweep's timed region into its parts and
//! A/Bs it against the direct `jxl_encoder` path the sweep uses for expert
//! knobs, so the parasite can be attributed rather than guessed at.
//!
//! Per the sweep-discipline rule, it sweeps SIZE (so the per-pixel slope and
//! the fixed intercept can be separated — a "s/MP" number alone is meaningless
//! without the intercept) and EFFORT (so an effort-independent term is
//! visible as such).
//!
//! Run:
//!   cargo run --release -p zenmetrics-cli --features sweep,jxl,png \
//!       --example jxl_encode_ms_probe
//!
//! Env: ZEN_PROBE_SIZES (default "64,128,256,384,512")
//!      ZEN_PROBE_EFFORTS (default "1,5,9")
//!      ZEN_PROBE_REPS (default 3; the MINIMUM wall is reported, which is the
//!      least contention-sensitive statistic)
//!      ZEN_PROBE_THREADS (default 0 = ambient rayon pool, what the knob-grid
//!      path gets via `apply_threads` with no ResourceLimits; set 1 to match the
//!      PLAN path, which pins `.with_threads(1)` per cell for content-addressing
//!      determinism). This is the #34 attribution knob.
//!      ZEN_PROBE_CONC (default 0 = off) — when >0, ALSO run that many encodes
//!      concurrently and report per-encode wall. This is the arm that matters
//!      for #34: `apply_threads` gives the wrapper `threads=0` (ambient rayon
//!      pool), so a sweep running cells in parallel has every in-flight cell
//!      contending for the same pool. Compare `conc_*` against the serial
//!      numbers above it.

use std::time::Instant;

fn synth(w: u32, h: u32) -> Vec<u8> {
    // Deterministic, non-degenerate content: a smooth gradient plus a
    // high-frequency term, so neither the DC path nor the AC path is
    // trivially skipped.
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    let mut s: u32 = 0x9E3779B9;
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = (s >> 24) as u8 / 8;
            v.push(((x * 255 / w.max(1)) as u8).wrapping_add(n));
            v.push(((y * 255 / h.max(1)) as u8).wrapping_add(n));
            v.push((((x ^ y) & 0xff) as u8).wrapping_add(n));
        }
    }
    v
}

fn env_list(key: &str, default: &str) -> Vec<u32> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect()
}

fn main() {
    use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
    use zenjxl::JxlEncoderConfig;
    use zenpixels::{PixelDescriptor, PixelSlice};

    let sizes = env_list("ZEN_PROBE_SIZES", "64,128,256,384,512");
    let efforts = env_list("ZEN_PROBE_EFFORTS", "1,5,9");
    let reps: u32 = std::env::var("ZEN_PROBE_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let threads: u8 = std::env::var("ZEN_PROBE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    println!(
        "# jxl_encode_ms_probe  threads_available={}  reps={reps} (min reported)  direct_threads={threads}",
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    println!(
        "size\tmp\teffort\tconstruct_ms\twrapper_encode_ms\twrapper_total_ms\tdirect_encode_ms\tbytes_wrapper\tbytes_direct"
    );

    for &n in &sizes {
        let px = synth(n, n);
        let mp = (n as f64) * (n as f64) / 1.0e6;
        for &e in &efforts {
            let (mut c_ms, mut w_ms, mut d_ms) = (f64::MAX, f64::MAX, f64::MAX);
            let (mut wb, mut db) = (0usize, 0usize);

            for _ in 0..reps {
                // ── the sweep's own path: zenjxl wrapper via zencodec ──
                let cfg = JxlEncoderConfig::new()
                    .with_generic_quality(75.0)
                    .with_generic_effort(e as i32);
                let slice =
                    PixelSlice::new(&px, n, n, (n as usize) * 3, PixelDescriptor::RGB8_SRGB)
                        .expect("pixel slice");
                let t0 = Instant::now();
                let encoder = cfg.job().encoder().expect("encoder");
                let t1 = Instant::now();
                let out = encoder.encode(slice).expect("wrapper encode");
                let t2 = Instant::now();
                c_ms = c_ms.min((t1 - t0).as_secs_f64() * 1e3);
                w_ms = w_ms.min((t2 - t1).as_secs_f64() * 1e3);
                wb = out.into_vec().len();

                // ── the direct jxl_encoder path (what the expert arm uses) ──
                let d_cfg = jxl_encoder::LossyConfig::new(jxl_encoder::quality_to_distance(
                    jxl_encoder::calibrated_jxl_quality(75.0),
                ))
                .with_effort(e as u8)
                .with_threads(threads as usize);
                let t3 = Instant::now();
                let dbytes = d_cfg
                    .encode(&px, n, n, jxl_encoder::PixelLayout::Rgb8)
                    .expect("direct encode");
                d_ms = d_ms.min(t3.elapsed().as_secs_f64() * 1e3);
                db = dbytes.len();
            }

            println!(
                "{n}\t{mp:.4}\t{e}\t{c_ms:.3}\t{w_ms:.3}\t{:.3}\t{d_ms:.3}\t{wb}\t{db}",
                c_ms + w_ms
            );
        }
    }

    let conc: usize = std::env::var("ZEN_PROBE_CONC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if conc == 0 {
        return;
    }

    println!();
    println!("# concurrency arm: {conc} simultaneous encodes, per-encode wall (mean/max)");
    println!(
        "size\tmp\teffort\tconc\tserial_ms\tconc_mean_ms\tconc_max_ms\tinflation_x\tconc_ms_per_mp"
    );
    for &n in &sizes {
        let px = synth(n, n);
        let mp = (n as f64) * (n as f64) / 1.0e6;
        for &e in &efforts {
            // serial baseline (min of reps)
            let mut serial = f64::MAX;
            for _ in 0..reps {
                let cfg = JxlEncoderConfig::new()
                    .with_generic_quality(75.0)
                    .with_generic_effort(e as i32);
                let slice =
                    PixelSlice::new(&px, n, n, (n as usize) * 3, PixelDescriptor::RGB8_SRGB)
                        .expect("slice");
                let t = Instant::now();
                let _ = cfg
                    .job()
                    .encoder()
                    .expect("enc")
                    .encode(slice)
                    .expect("encode");
                serial = serial.min(t.elapsed().as_secs_f64() * 1e3);
            }

            let walls: Vec<f64> = std::thread::scope(|sc| {
                let handles: Vec<_> = (0..conc)
                    .map(|_| {
                        let px = &px;
                        sc.spawn(move || {
                            let cfg = JxlEncoderConfig::new()
                                .with_generic_quality(75.0)
                                .with_generic_effort(e as i32);
                            let slice = PixelSlice::new(
                                px,
                                n,
                                n,
                                (n as usize) * 3,
                                PixelDescriptor::RGB8_SRGB,
                            )
                            .expect("slice");
                            let t = Instant::now();
                            let _ = cfg
                                .job()
                                .encoder()
                                .expect("enc")
                                .encode(slice)
                                .expect("encode");
                            t.elapsed().as_secs_f64() * 1e3
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("join"))
                    .collect()
            });

            let mean = walls.iter().sum::<f64>() / walls.len() as f64;
            let max = walls.iter().cloned().fold(0.0f64, f64::max);
            println!(
                "{n}\t{mp:.4}\t{e}\t{conc}\t{serial:.3}\t{mean:.3}\t{max:.3}\t{:.2}\t{:.1}",
                mean / serial,
                mean / mp
            );
        }
    }
}
