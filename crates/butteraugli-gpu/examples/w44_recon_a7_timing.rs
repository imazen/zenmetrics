//! W44-RECON-DEEP/A7 timing harness: CPU vs GPU butteraugli wall-time
//! at three sizes on real corpus images.
//!
//! Single-shot per-call timing (no loop amortization), matching the
//! conditions the JXL encoder buttloop would face: each iteration
//! computes butteraugli once between the (current quantized recon)
//! and the (target reference) — there's no inner loop to warm up.
//!
//! Reports best-of-N for both CPU and GPU per (image, size).

use butteraugli::{ButteraugliParams as CpuParams, butteraugli as cpu_compute};
use butteraugli_gpu::Butteraugli;
use image::ImageReader;
use imgref::ImgVec;
use rgb::{RGB, RGB8};
use std::time::Instant;

type Backend = cubecl::cuda::CudaRuntime;

fn load_rgb_resized(path: &str, target: u32) -> (Vec<u8>, u32, u32) {
    let img = ImageReader::open(path)
        .expect("open png")
        .decode()
        .expect("decode png")
        .to_rgb8();
    let img = image::imageops::resize(
        &img,
        target,
        target,
        image::imageops::FilterType::Lanczos3,
    );
    let (w, h) = (img.width(), img.height());
    (img.into_raw(), w, h)
}

fn perturb_jpeg_like(src: &[u8], width: u32, mag: i32) -> Vec<u8> {
    let mut out = src.to_vec();
    let w = width as usize;
    let n = src.len() / 3;
    for i in 0..n {
        let x = i % w;
        let y = i / w;
        let bx = x / 8;
        let by = y / 8;
        let stripe = (((bx + by) % 3) as i32 - 1).signum();
        for c in 0..3 {
            let v = src[i * 3 + c] as i32 + stripe * mag;
            out[i * 3 + c] = v.clamp(0, 255) as u8;
        }
    }
    out
}

fn rgb_buf_to_imgvec(buf: &[u8], width: u32, height: u32) -> ImgVec<RGB8> {
    let pixels: Vec<RGB8> = buf
        .chunks_exact(3)
        .map(|c| RGB { r: c[0], g: c[1], b: c[2] })
        .collect();
    ImgVec::new(pixels, width as usize, height as usize)
}

fn main() {
    let images: Vec<String> = std::env::args().skip(1).collect();
    if images.is_empty() {
        eprintln!("usage: w44_recon_a7_timing <image1> [image2 ...]");
        std::process::exit(1);
    }

    let sizes: [(u32, u32); 3] = [(256, 256), (512, 512), (1024, 1024)];
    let mag: i32 = 8;
    let iters: usize = 5;

    let device = <Backend as cubecl::Runtime>::Device::default();
    let client = <Backend as cubecl::Runtime>::client(&device);

    println!(
        "image\twidth\theight\tmag\tmode\tcpu_best_ms\tcpu_med_ms\tgpu_best_ms\tgpu_med_ms\tcpu_score\tgpu_score\tabs_diff\trel_diff_pct"
    );

    for path in &images {
        for &(w, h) in &sizes {
            let (ref_rgb, ww, hh) = load_rgb_resized(path, w);
            assert_eq!(ww, w);
            assert_eq!(hh, h);
            let dist_rgb = perturb_jpeg_like(&ref_rgb, w, mag);
            let ref_img = rgb_buf_to_imgvec(&ref_rgb, w, h);
            let dist_img = rgb_buf_to_imgvec(&dist_rgb, w, h);

            for &(mode_name, single_res) in &[("multires", false), ("singleres", true)] {
                // --- CPU ---
                let params = if single_res {
                    CpuParams::default().with_single_resolution(true)
                } else {
                    CpuParams::default()
                };
                let mut cpu_times: Vec<f64> = Vec::with_capacity(iters);
                let mut cpu_score: f64 = 0.0;
                for _ in 0..iters {
                    let t = Instant::now();
                    let r = cpu_compute(ref_img.as_ref(), dist_img.as_ref(), &params).unwrap();
                    cpu_times.push(t.elapsed().as_secs_f64() * 1000.0);
                    cpu_score = r.score;
                }
                cpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let cpu_best = cpu_times[0];
                let cpu_med = cpu_times[iters / 2];

                // --- GPU (warm) ---
                let mut gpu = if single_res {
                    Butteraugli::<Backend>::new(client.clone(), w, h)
                } else {
                    Butteraugli::<Backend>::new_multires(client.clone(), w, h)
                };
                // 2 warmup runs (JIT, kernel cache)
                for _ in 0..2 {
                    let _ = gpu.compute(&ref_rgb, &dist_rgb).unwrap();
                }
                let mut gpu_times: Vec<f64> = Vec::with_capacity(iters);
                let mut gpu_score: f32 = 0.0;
                for _ in 0..iters {
                    let t = Instant::now();
                    let r = gpu.compute(&ref_rgb, &dist_rgb).unwrap();
                    gpu_times.push(t.elapsed().as_secs_f64() * 1000.0);
                    gpu_score = r.score;
                }
                gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let gpu_best = gpu_times[0];
                let gpu_med = gpu_times[iters / 2];

                let abs_diff = (gpu_score as f64 - cpu_score).abs();
                let rel_diff = abs_diff / cpu_score * 100.0;

                let stem = std::path::Path::new(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
                    stem, w, h, mag, mode_name,
                    cpu_best, cpu_med, gpu_best, gpu_med,
                    cpu_score, gpu_score, abs_diff, rel_diff
                );
            }
        }
    }
}
