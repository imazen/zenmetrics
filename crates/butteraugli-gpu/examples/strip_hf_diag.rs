//! Diagnostic: reproduce the HF-checkerboard Strip-vs-Full divergence
//! found by the mode_wall sweep (task #158).
//!
//! Content = the EXACT `structured_pair` from
//! `crates/zenmetrics-api/examples/mode_wall.rs`: a smooth RGB gradient
//! reference, and a distorted image = gradient + a period-8 (8×8 block)
//! ±mag checkerboard perturbation. At mag=12, 1024²/body-256 the
//! mode_wall sweep measured butter Strip diverging ~8% from Full.
//!
//! This harness prints Full vs Strip score + pnorm_3 + rel error across
//! a body-size sweep so we can see whether the divergence is a function
//! of the number of interior strip boundaries (more strips = more
//! boundaries = more error if the halo is insufficient).

use butteraugli_gpu::{Butteraugli, ButteraugliParams};
#[cfg(feature = "cuda")]
use butteraugli_gpu::{ButteraugliOpaque, MemoryMode};
use cubecl::Runtime;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

/// EXACT copy of mode_wall's `structured_pair`.
fn structured_pair(w: u32, h: u32, mag: u8) -> (Vec<u8>, Vec<u8>) {
    let (width, height) = (w as usize, h as usize);
    let mut a = vec![0u8; width * height * 3];
    let mut b = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 220 / width.max(1)) & 0xff) as u8;
            let g = ((y * 220 / height.max(1)) & 0xff) as u8;
            let bb = (((x + y) * 200 / (width + height).max(1)) & 0xff) as u8;
            let i = (y * width + x) * 3;
            a[i] = r;
            a[i + 1] = g;
            a[i + 2] = bb;
            let bx = x / 8;
            let by = y / 8;
            let pert = if (bx ^ by) & 1 == 0 {
                mag as i32
            } else {
                -(mag as i32)
            };
            b[i] = (r as i32 + pert).clamp(0, 255) as u8;
            b[i + 1] = (g as i32 + pert).clamp(0, 255) as u8;
            b[i + 2] = (bb as i32 + pert).clamp(0, 255) as u8;
        }
    }
    (a, b)
}

fn rel(want: f32, got: f32) -> f64 {
    let denom = (want as f64).abs().max(1e-12);
    (got as f64 - want as f64).abs() / denom
}

fn main() {
    // size, mag, list of body sizes to test
    let w = std::env::var("DIAG_W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024u32);
    let h = std::env::var("DIAG_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024u32);
    let mag: u8 = std::env::var("DIAG_MAG")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let bodies: Vec<u32> = std::env::var("DIAG_BODIES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![64, 128, 256, 512, h]);

    println!("# strip_hf_diag: {w}x{h} mag={mag} (HALO_ROWS see crate const)");
    println!(
        "{:>5}  {:>7}  {:>11}  {:>11}  {:>11}  {:>11}  {:>9}  {:>9}",
        "body",
        "nstrips",
        "whole_score",
        "strip_score",
        "whole_p3",
        "strip_p3",
        "rel_score",
        "rel_p3"
    );

    let (ref_buf, dis_buf) = structured_pair(w, h, mag);
    let client = Backend::client(&Default::default());

    // Single-res whole-image baseline (compute once).
    let mut whole = Butteraugli::<Backend>::new(client.clone(), w, h);
    let wr = whole
        .compute_with_options(&ref_buf, &dis_buf, &ButteraugliParams::default())
        .expect("whole");

    // Multires whole-image baseline — this is what the umbrella's
    // MemoryMode::Full path uses (new_multires). The benchmark's
    // "butter,full" number came from this.
    let mut mwhole = Butteraugli::<Backend>::new_multires(client.clone(), w, h);
    let mwr = mwhole
        .compute_with_options(&ref_buf, &dis_buf, &ButteraugliParams::default())
        .expect("multires whole");

    println!(
        "# single-res whole score={:.6} p3={:.6}   MULTIRES whole score={:.6} p3={:.6}",
        wr.score, wr.pnorm_3, mwr.score, mwr.pnorm_3
    );
    println!("# --- single-res strip (new_strip) vs single-res whole ---");
    for &body in &bodies {
        let nstrips = (h as f32 / body as f32).ceil() as u32;
        let mut strip = Butteraugli::<Backend>::new_strip(client.clone(), w, h, body);
        let sr = strip.compute_strip(&ref_buf, &dis_buf).expect("strip");
        println!(
            "{:>5}  {:>7}  {:>11.6}  {:>11.6}  {:>11.6}  {:>11.6}  {:>9.2e}  {:>9.2e}",
            body,
            nstrips,
            wr.score,
            sr.score,
            wr.pnorm_3,
            sr.pnorm_3,
            rel(wr.score, sr.score),
            rel(wr.pnorm_3, sr.pnorm_3),
        );
    }

    println!("# --- MULTIRES strip (new_multires_strip) vs MULTIRES whole ---");
    for &body in &bodies {
        // new_multires_strip enforces even body_h; round down to even.
        let body_even = body & !1;
        if body_even == 0 {
            continue;
        }
        let nstrips = (h as f32 / body_even as f32).ceil() as u32;
        let mut mstrip =
            Butteraugli::<Backend>::new_multires_strip(client.clone(), w, h, body_even);
        let msr = mstrip
            .compute_strip(&ref_buf, &dis_buf)
            .expect("multires strip");
        println!(
            "{:>5}  {:>7}  {:>11.6}  {:>11.6}  {:>11.6}  {:>11.6}  {:>9.2e}  {:>9.2e}",
            body_even,
            nstrips,
            mwr.score,
            msr.score,
            mwr.pnorm_3,
            msr.pnorm_3,
            rel(mwr.score, msr.score),
            rel(mwr.pnorm_3, msr.pnorm_3),
        );
    }

    // ── Umbrella opaque path (what zenmetrics-api Metric::Butter wraps,
    //    and what the mode_wall MW_PARITY harness measured). This is the
    //    EXACT comparison that found the 8% bug: Full vs Strip via the
    //    opaque shim. Both must now agree. ──
    #[cfg(feature = "cuda")]
    {
        use butteraugli_gpu::Backend as OBackend;
        println!("# --- UMBRELLA opaque (ButteraugliOpaque) Full vs Strip ---");
        let mut of = ButteraugliOpaque::new_with_memory_mode(
            OBackend::Cuda,
            w,
            h,
            ButteraugliParams::default(),
            MemoryMode::Full,
        )
        .expect("opaque full");
        let ofs = of
            .compute_srgb_u8(&ref_buf, &dis_buf)
            .expect("opaque full compute")
            .value;
        for &body in &bodies {
            let mut os = ButteraugliOpaque::new_with_memory_mode(
                OBackend::Cuda,
                w,
                h,
                ButteraugliParams::default(),
                MemoryMode::Strip { h_body: Some(body) },
            )
            .expect("opaque strip");
            let oss = os
                .compute_srgb_u8(&ref_buf, &dis_buf)
                .expect("opaque strip compute")
                .value;
            let r = (oss - ofs).abs() / ofs.abs().max(1e-12);
            println!(
                "body={body:>5} nstrips={:>3}  opaque_full={ofs:>11.6}  opaque_strip={oss:>11.6}  rel={r:>9.2e}",
                (h as f32 / body as f32).ceil() as u32,
            );
        }
    }
}
