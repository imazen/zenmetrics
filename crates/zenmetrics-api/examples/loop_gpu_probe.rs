//! Task #163 — GPU side of the encoder-search-loop question, on the umbrella
//! API that production actually uses.
//!
//! Answers two things the CPU harness (`loop-wall`) cannot:
//!
//! 1. **Is `ButteraugliOpaque`'s cached reference real, or a replay?**
//!    `butteraugli-gpu`'s `resolve_auto` prefers Strip over Full whenever
//!    `height > MIN_STRIP_BODY + 2*HALO_ROWS` (= 64 + 160 = 224), *even when
//!    Full fits* (`butteraugli-gpu/src/memory_mode.rs:190-194`). In strip mode
//!    `set_reference_srgb_u8` only clones the bytes host-side
//!    (`opaque.rs:703`) and `compute_with_reference_srgb_u8` calls
//!    `compute_srgb_u8(&held, dist)` (`opaque.rs:726`) — the full one-shot.
//!    So the "warm" path should measure the SAME as the cold path under
//!    `Auto`, and strictly faster under `Full`. This example runs both and
//!    prints the ratio instead of asserting the source reading.
//!
//! 2. **What is the GPU setup bill and steady-state per-candidate cost on
//!    THIS backend?** Every committed GPU number in this repo is CUDA
//!    (`benchmarks/gpu_coldstart_2026-05-29.tsv`). The crossover in
//!    `benchmarks/metric_loop_2026-08-30.md` is stated against those; this
//!    prints the same four phases for whatever backend is selected so a
//!    non-CUDA host can be quoted on its own numbers.
//!
//! Phases, per metric:
//!   metric_new    — constructor (bulk device buffer allocation)
//!   first_compute — first dispatch: shader/NVRTC compile + upload + compute
//!   cold_warm     — steady-state `compute_srgb_u8(ref, dist)` (pair path)
//!   set_reference — one `set_reference_srgb_u8`
//!   warm_ref      — steady-state `compute_with_reference_srgb_u8(dist)`
//!
//! `warm_ref / cold_warm` is the number that matters: < 1 means the cached
//! reference is real; ~= 1 means it is a replay.
//!
//! Usage:
//!   LOOP_GPU_BACKEND=wgpu|cuda  LOOP_GPU_SIZES=256,512,1024  \
//!   LOOP_GPU_REPS=5  cargo run --release -p zenmetrics-api \
//!     --no-default-features --features wgpu,butter,ssim2,zensim,cubecl-types \
//!     --example loop_gpu_probe
//!
//! Output is TSV on stdout (header + rows), progress on stderr.

use std::time::Instant;

use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams};

/// Deterministic LCG sRGB bytes. Distinct `seed` => distinct content, so a
/// cached reference is never fed the same bytes twice and zensim-gpu's
/// identity short-circuit (`opaque.rs:1427`, returns 100.0 with zero GPU
/// work on `ref == dist`) can never fire.
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

fn median(t: &[f64]) -> f64 {
    if t.is_empty() {
        return f64::NAN;
    }
    let mut v = t.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n.is_multiple_of(2) {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn min_of(t: &[f64]) -> f64 {
    t.iter().cloned().fold(f64::INFINITY, f64::min)
}

struct Row {
    metric: &'static str,
    mode: &'static str,
    w: u32,
    h: u32,
    phase: &'static str,
    ms_median: f64,
    ms_min: f64,
    n: usize,
    score: f64,
}

fn emit(rows: &[Row]) {
    println!("metric\tmemory_mode\tw\th\tphase\tms_median\tms_min\tn\tscore");
    for r in rows {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}",
            r.metric, r.mode, r.w, r.h, r.phase, r.ms_median, r.ms_min, r.n, r.score
        );
    }
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn probe(
    kind: MetricKind,
    tag: &'static str,
    backend: Backend,
    mode: MemoryMode,
    mode_tag: &'static str,
    w: u32,
    h: u32,
    reps: usize,
    rows: &mut Vec<Row>,
) {
    eprintln!("[probe] {tag} {mode_tag} {w}x{h} reps={reps}");
    let params = match MetricParams::try_default_for(kind) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[probe] {tag}: params unavailable: {e} — SKIP");
            return;
        }
    };

    let t = Instant::now();
    let mut m = match Metric::new_with_memory_mode(kind, backend, w, h, params, mode) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[probe] {tag} {mode_tag} {w}x{h}: Metric::new failed: {e} — SKIP");
            return;
        }
    };
    let new_ms = t.elapsed().as_secs_f64() * 1e3;

    let refb = synth_srgb(w, h, 1);
    let dists: Vec<Vec<u8>> = (0..reps.max(1))
        .map(|i| synth_srgb(w, h, 100 + i as u32))
        .collect();

    // First dispatch pays the shader/NVRTC compile. Timed separately and
    // NEVER folded into the steady-state numbers.
    let t = Instant::now();
    let first = match m.compute_srgb_u8(&refb, &dists[0]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[probe] {tag} {mode_tag} {w}x{h}: first compute failed: {e} — SKIP");
            return;
        }
    };
    let first_ms = t.elapsed().as_secs_f64() * 1e3;

    // Steady-state COLD: full pair compute, reference re-uploaded each call.
    let mut cold = Vec::new();
    let mut last_cold = first.value;
    for d in &dists {
        let t = Instant::now();
        match m.compute_srgb_u8(&refb, d) {
            Ok(s) => {
                cold.push(t.elapsed().as_secs_f64() * 1e3);
                last_cold = s.value;
            }
            Err(e) => {
                eprintln!("[probe] {tag}: cold compute failed: {e}");
                return;
            }
        }
    }

    // set_reference, then steady-state WARM.
    let t = Instant::now();
    let setref_ok = m.set_reference_srgb_u8(&refb);
    let setref_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut warm = Vec::new();
    let mut last_warm = f64::NAN;
    if let Err(e) = setref_ok {
        eprintln!("[probe] {tag} {mode_tag}: set_reference failed: {e}");
    } else {
        for d in &dists {
            let t = Instant::now();
            match m.compute_with_reference_srgb_u8(d) {
                Ok(s) => {
                    warm.push(t.elapsed().as_secs_f64() * 1e3);
                    last_warm = s.value;
                }
                Err(e) => {
                    eprintln!("[probe] {tag} {mode_tag}: warm compute failed: {e}");
                    break;
                }
            }
        }
    }

    let push = |rows: &mut Vec<Row>, phase: &'static str, v: &[f64], score: f64| {
        rows.push(Row {
            metric: tag,
            mode: mode_tag,
            w,
            h,
            phase,
            ms_median: median(v),
            ms_min: min_of(v),
            n: v.len(),
            score,
        });
    };
    push(rows, "metric_new", &[new_ms], f64::NAN);
    push(rows, "first_compute", &[first_ms], first.value);
    push(rows, "cold_warm", &cold, last_cold);
    push(rows, "set_reference", &[setref_ms], f64::NAN);
    push(rows, "warm_ref", &warm, last_warm);

    let c = median(&cold);
    let wv = median(&warm);
    // The headline: is the cached reference real, or a replay of the cold path?
    eprintln!(
        "[probe] {tag} {mode_tag} {w}x{h}: cold={c:.2}ms warm={wv:.2}ms warm/cold={:.3} \
         (score cold={last_cold:.4} warm={last_warm:.4})",
        wv / c
    );
    if wv.is_finite() && c.is_finite() && wv / c > 0.9 {
        eprintln!("[probe]   ^ warm/cold >= 0.9 — the cached reference is NOT saving work here.");
    }
}

fn main() {
    let backend = match std::env::var("LOOP_GPU_BACKEND").as_deref() {
        Ok("cuda") => Backend::Cuda,
        _ => Backend::Wgpu,
    };
    let sizes: Vec<u32> = std::env::var("LOOP_GPU_SIZES")
        .unwrap_or_else(|_| "256,512,1024".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let reps: usize = std::env::var("LOOP_GPU_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    eprintln!("[probe] backend={backend:?} sizes={sizes:?} reps={reps}");
    let mut rows = Vec::new();

    for &s in &sizes {
        // butteraugli twice: Auto (what the umbrella builds by default, and
        // what resolves to Strip above 224 px tall) vs an explicit Full.
        probe(
            MetricKind::Butter,
            "butter",
            backend,
            MemoryMode::Auto,
            "auto",
            s,
            s,
            reps,
            &mut rows,
        );
        probe(
            MetricKind::Butter,
            "butter",
            backend,
            MemoryMode::Full,
            "full",
            s,
            s,
            reps,
            &mut rows,
        );
        probe(
            MetricKind::Ssim2,
            "ssim2",
            backend,
            MemoryMode::Auto,
            "auto",
            s,
            s,
            reps,
            &mut rows,
        );
        probe(
            MetricKind::Zensim,
            "zensim",
            backend,
            MemoryMode::Auto,
            "auto",
            s,
            s,
            reps,
            &mut rows,
        );
    }

    emit(&rows);
}
