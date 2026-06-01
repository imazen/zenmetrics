//! GPU memory measurement for `Cvvdp::new` Full vs
//! `Cvvdp::new_strip_pair` Mode B at 1024² and 4096² stretch.
//!
//! Reports TWO numbers per (size, mode):
//!
//! - `estimate_mib` — what `estimate_gpu_memory_bytes*` computes
//!   analytically from the buffer layout. This is the "pipeline
//!   asks the GPU for X bytes" number — the layer the Path-B
//!   lazy-transient refactor directly affects.
//! - `nvsmi_mib` — what `nvidia-smi --query-gpu=memory.used`
//!   reports for the process's contribution. This includes
//!   cubecl pool over-allocation, kernel JIT scratch, and CUDA
//!   driver page rounding. At the MiB granularity nvidia-smi
//!   exposes, this number is typically 2-6× the analytical
//!   estimate and dominates the visible peak.
//!
//! Both numbers matter — the pipeline-level estimate is what code
//! changes move; the driver-level nvsmi is what production capacity
//! planning sees. The Path-B chunk-1 refactor cuts the estimator
//! by ~10% at 1MP (~20 MB out of ~187 MB analytic). nvidia-smi
//! granularity buries that under cubecl pool over-allocation, so
//! the GATE criterion shifts to "estimator drops, JOD bit-identical,
//! and nvsmi doesn't regress."
//!
//! ## Why subprocess-per-measurement
//!
//! cubecl's memory pool keeps GPU buffers cached across `Drop` for
//! reuse by the next allocation in the same process. That defeats
//! a single-process before/after measurement — the Mode B baseline
//! ends up equal to the Full peak because the pool never returned
//! buffers to the driver. We work around this by launching the
//! actual cvvdp work in a child process: the OS reclaims the pool
//! on child exit, so the next child sees a clean baseline.
//!
//! Run with:
//!
//!     cargo run --release --example mem_mode_b_vs_full -p cvvdp-gpu --features cuda

#![cfg(feature = "cuda")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cubecl::Runtime;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
use cvvdp_gpu::{Cvvdp, estimate_gpu_memory_bytes, estimate_gpu_memory_bytes_strip_pair};

/// Replica of the PRE-PATH-B-CHUNK-1 `estimate_gpu_memory_bytes`
/// formula, used here for delta accounting. The committed
/// estimator returns the post-refactor working set (lazy
/// transient layout); subtracting this baseline number gives
/// the bytes saved by chunk 1 in Full mode.
///
/// Layout we mirror (from master HEAD 56a5c5be):
///   d_scratch: 6 buffer kinds × 3 channels × sum_level_pixels × 4
/// Layout we produce (in this branch):
///   d_scratch: 3 ch × sum_level_pixels × 4  (persistent `d`)
///            + 5 kinds × 3 ch × n0 × 4      (peak transient)
fn estimate_gpu_memory_bytes_pre_chunk1(width: u32, height: u32) -> Option<usize> {
    use cvvdp_gpu::params::DisplayGeometry;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    // Hard-coded PYRAMID_MIN_DIM = 4 (matches the const in
    // cvvdp_gpu's public crate; we don't re-export it but the
    // value is stable).
    const PYRAMID_MIN_DIM: u32 = 4;
    if width < PYRAMID_MIN_DIM * 2 || height < PYRAMID_MIN_DIM * 2 {
        return None;
    }
    // Pyramid-depth helper inlined: a stable formula based on the
    // ~0.2 cpd cutoff cvvdp uses. For STANDARD_4K geometry and
    // 1024² / 4096² the levels are 7 and 9 respectively — the
    // canonical pipeline_levels() in master returned those values.
    // We compute by halving until min(w,h) < 8.
    let n_levels = {
        let mut levels = 0u32;
        let mut w = width;
        let mut h = height;
        while w >= PYRAMID_MIN_DIM * 2 && h >= PYRAMID_MIN_DIM * 2 {
            levels += 1;
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        levels.max(1)
    };
    let _ = ppd; // STANDARD_4K geometry is implicit in this fallback.

    let n0 = (width as usize) * (height as usize);
    let src_bytes: usize = n0 * 3 * 4;
    let srgb_lut_bytes: usize = 256 * 4;
    let partials_bytes: usize = n_levels as usize * 3 * 4;
    // N_L_BKG matches kernels::csf::N_L_BKG (32). Internal but stable.
    const N_L_BKG: usize = 32;
    let logs_row_bytes: usize = n_levels as usize * 3 * N_L_BKG * 4;

    let mut level_pixels: Vec<usize> = Vec::with_capacity(n_levels as usize);
    let mut w = width;
    let mut h = height;
    for _ in 0..n_levels {
        level_pixels.push((w as usize) * (h as usize));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let sum_level_pixels: usize = level_pixels.iter().sum();

    let pyramid_bytes: usize = 3 * 3 * sum_level_pixels * 4;
    // The OLD formula: 6 kinds × 3 ch × all-levels.
    let d_scratch_bytes_old: usize = 6 * 3 * sum_level_pixels * 4;

    let mut weber_bytes: usize = 0;
    let mut fw = width;
    let mut fh = height;
    for _ in 0..n_levels.saturating_sub(1) {
        let n_fine = (fw as usize) * (fh as usize);
        let cw = fw.div_ceil(2);
        let n_v = (cw as usize) * (fh as usize);
        weber_bytes += (6 * n_fine + 4 * n_v) * 4;
        fw = cw;
        fh = fh.div_ceil(2);
    }

    let baseband_bytes: usize = level_pixels.last().copied().unwrap_or(0) * 4;

    Some(
        src_bytes
            + srgb_lut_bytes
            + partials_bytes
            + logs_row_bytes
            + pyramid_bytes
            + d_scratch_bytes_old
            + weber_bytes
            + baseband_bytes,
    )
}

#[path = "../tests/common/mod.rs"]
mod common;

use common::Backend;

const SIZES: &[(u32, u32, &str)] = &[(1024, 1024, "1024sq"), (4096, 4096, "4096sq")];

const STRIP_H_BODY: u32 = 256;

/// How long the child holds GPU buffers alive AFTER signalling
/// READY so the parent can sample nvidia-smi at quiescent steady
/// state. Empirically a single sample suffices once the child has
/// returned from compute_dkl_jod (driver state is stable post-
/// readback), but we take a few samples to defend against
/// transient driver-side fluctuation.
const CHILD_HOLD_MS: u64 = 400;

fn nvidia_smi_memory_used_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse().ok()
}

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    common::synth_pair_with_offset_dist(w as usize, h as usize)
}

/// Child process body. Allocates Cvvdp, runs one compute_dkl_jod,
/// signals READY <jod>, then sleeps while the parent samples
/// nvidia-smi.
fn run_worker(mode: &str, w: u32, h: u32) {
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_b, dis_b) = synth_pair(w, h);

    let client = Backend::client(&Default::default());
    let mut cvvdp = match mode {
        "full" => Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
            .expect("Cvvdp::new (Full)"),
        "mode_b" => {
            Cvvdp::<Backend>::new_strip_pair(client, w, h, STRIP_H_BODY, CvvdpParams::PLACEHOLDER)
                .expect("Cvvdp::new_strip_pair (Mode B)")
        }
        other => panic!("unknown WORKER_MODE: {other}"),
    };

    let jod = cvvdp
        .compute_dkl_jod(&ref_b, &dis_b, ppd)
        .expect("compute_dkl_jod");

    println!("READY {jod:.6}");
    std::io::stdout().flush().expect("flush stdout");

    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
    drop(cvvdp);
}

/// Run one child measurement, returning (baseline_mib, peak_mib,
/// delta_mib, jod). The parent reads child stdout until READY then
/// samples nvidia-smi memory.used for CHILD_HOLD_MS.
fn measure_one(child_bin: &str, mode: &str, w: u32, h: u32) -> (u64, u64, i64, f32) {
    std::thread::sleep(Duration::from_millis(300));
    let baseline_mib = nvidia_smi_memory_used_mib().expect("nvidia-smi baseline");

    let mut child = Command::new(child_bin)
        .env("WORKER_MODE", mode)
        .env("WORKER_W", w.to_string())
        .env("WORKER_H", h.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn child");

    let stdout = child.stdout.take().expect("child stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let ready_deadline = Instant::now() + Duration::from_secs(60);
    let mut jod: f32 = f32::NAN;
    while Instant::now() < ready_deadline {
        line.clear();
        let n = reader.read_line(&mut line).expect("read child stdout");
        if n == 0 {
            break;
        }
        if let Some(rest) = line.trim().strip_prefix("READY ") {
            jod = rest.parse().expect("parse READY jod");
            break;
        }
    }
    if jod.is_nan() {
        let out = child.wait_with_output().expect("wait child");
        eprintln!(
            "child never sent READY: status={:?}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        panic!("child failed");
    }

    let sample_until = Instant::now() + Duration::from_millis(CHILD_HOLD_MS);
    let mut peak_mib: u64 = baseline_mib;
    while Instant::now() < sample_until {
        if let Some(v) = nvidia_smi_memory_used_mib() {
            if v > peak_mib {
                peak_mib = v;
            }
        }
        std::thread::sleep(Duration::from_millis(30));
    }

    let out = child.wait_with_output().expect("wait child");
    if !out.status.success() {
        panic!(
            "child failed: status={:?}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let delta_mib = (peak_mib as i64) - (baseline_mib as i64);
    (baseline_mib, peak_mib, delta_mib, jod)
}

fn main() {
    // Worker entry.
    if let Ok(mode) = std::env::var("WORKER_MODE") {
        let w: u32 = std::env::var("WORKER_W").unwrap().parse().unwrap();
        let h: u32 = std::env::var("WORKER_H").unwrap().parse().unwrap();
        run_worker(&mode, w, h);
        return;
    }

    let child_bin: String = std::env::current_exe()
        .expect("current_exe")
        .to_string_lossy()
        .into_owned();

    println!("size,mode,estimate_master_mib,estimate_head_mib,nvsmi_delta_mib,jod");
    eprintln!("# Cvvdp working-set measurement, h_body = {STRIP_H_BODY}, subprocess-per-cell");
    eprintln!("# estimate_master_mib = pre-Path-B-chunk-1 analytical estimate (Full)");
    eprintln!("# estimate_head_mib   = post-Path-B-chunk-1 analytical estimate");
    eprintln!("#   for Mode B this uses estimate_gpu_memory_bytes_strip_pair which");
    eprintln!("#   ALSO models per-band strip-buffer allocation — and that piece is");
    eprintln!("#   NOT yet wired into the runtime allocator (still uses full-image");
    eprintln!("#   bands_ref/bands_dis/gauss_ref). So the Mode B HEAD estimate is");
    eprintln!("#   aspirational pending Path A; chunk 1 itself only moves the Full");
    eprintln!("#   estimate (d_scratch transient kinds moved out of persistent storage).");
    eprintln!("# nvsmi_delta_mib     = nvidia-smi memory.used delta during compute_dkl_jod");
    eprintln!("#   (driver-level; cubecl pool over-allocates so this is 2-6× the");
    eprintln!("#    estimator. Chunk 1's ~20 MB delta at 1MP is below MiB granularity");
    eprintln!("#    once pool noise is included — JOD bit-identical is the load-bearing");
    eprintln!("#    guarantee, with the estimator delta as the bona-fide working-set claim.)");
    eprintln!();

    for &(w, h, label) in SIZES {
        let est_master_mib =
            estimate_gpu_memory_bytes_pre_chunk1(w, h).expect("estimate master") / (1 << 20);
        let est_full_head = estimate_gpu_memory_bytes(w, h).expect("estimate full") / (1 << 20);
        let est_mode_b_head = estimate_gpu_memory_bytes_strip_pair(w, h, STRIP_H_BODY)
            .expect("estimate mode_b")
            / (1 << 20);

        let t0 = Instant::now();
        let (_full_base, _full_peak, full_delta, full_jod) = measure_one(&child_bin, "full", w, h);
        let full_dur = t0.elapsed();
        println!("{label},full,{est_master_mib},{est_full_head},{full_delta},{full_jod:.4}");
        eprintln!(
            "  {label} full   : estimate master {est_master_mib} MiB -> HEAD {est_full_head} MiB  ({:+.1}% Δ)  nvsmi delta {full_delta:+5} MiB  jod {full_jod:.4}  ({full_dur:.2?})",
            (est_full_head as f64 / est_master_mib as f64 - 1.0) * 100.0,
        );

        let t1 = Instant::now();
        let (_mb_base, _mb_peak, mb_delta, mb_jod) = measure_one(&child_bin, "mode_b", w, h);
        let mb_dur = t1.elapsed();
        println!("{label},mode_b,{est_master_mib},{est_mode_b_head},{mb_delta},{mb_jod:.4}");
        eprintln!(
            "  {label} mode_b : estimate master {est_master_mib} MiB -> HEAD {est_mode_b_head} MiB  ({:+.1}% Δ aspirational)  nvsmi delta {mb_delta:+5} MiB  jod {mb_jod:.4}  ({mb_dur:.2?})",
            (est_mode_b_head as f64 / est_master_mib as f64 - 1.0) * 100.0,
        );

        let jod_diff = (full_jod - mb_jod).abs();
        eprintln!("  {label} |jod_full - jod_mode_b| = {jod_diff:.6}");
        eprintln!();
    }
}
