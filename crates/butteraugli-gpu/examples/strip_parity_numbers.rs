//! Print whole-image vs strip-mode (score, pnorm_3) and their relative
//! diffs, for a couple of representative sizes. Used by the strip-mode
//! task report to back the "1e-4 rel" claim with measured numbers.

use butteraugli_gpu::{Butteraugli, ButteraugliParams};
use cubecl::Runtime;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let sx = ((x as f32 / 32.0).sin() * 50.0 + 128.0) as u8;
            let sy = ((y as f32 / 24.0).cos() * 40.0 + 128.0) as u8;
            let hf = (((x ^ y).wrapping_mul(seed.max(1)) ^ seed) & 0x3f) as u8;
            out.push(sx.wrapping_add(hf));
            out.push(sy.wrapping_add(hf));
            out.push(sx.wrapping_add(sy).wrapping_add(hf >> 1));
        }
    }
    out
}

fn rel(want: f32, got: f32) -> f64 {
    let denom = (want as f64).abs().max(1e-12);
    (got as f64 - want as f64).abs() / denom
}

fn main() {
    let cases: &[(u32, u32, u32)] = &[
        (1024, 1024, 128),
        (512, 512, 64),
        (768, 800, 96),
        (1024, 1024, 256),
        (1024, 1024, 512),
    ];

    println!(
        "{:>10}  {:>9}  {:>5}  {:>11}  {:>11}  {:>11}  {:>11}  {:>9}  {:>9}",
        "w",
        "h",
        "body",
        "whole_score",
        "strip_score",
        "whole_p3",
        "strip_p3",
        "rel_score",
        "rel_p3"
    );

    for &(w, h, body) in cases {
        let client = Backend::client(&Default::default());
        let ref_buf = make_image(w, h, 0);
        let dis_buf = make_image(w, h, 7);

        let mut whole = Butteraugli::<Backend>::new(client.clone(), w, h);
        let wr = whole
            .compute_with_options(&ref_buf, &dis_buf, &ButteraugliParams::default())
            .expect("whole");

        let mut strip = Butteraugli::<Backend>::new_strip(client, w, h, body);
        let sr = strip.compute_strip(&ref_buf, &dis_buf).expect("strip");

        println!(
            "{:>10}  {:>9}  {:>5}  {:>11.6}  {:>11.6}  {:>11.6}  {:>11.6}  {:>9.2e}  {:>9.2e}",
            w,
            h,
            body,
            wr.score,
            sr.score,
            wr.pnorm_3,
            sr.pnorm_3,
            rel(wr.score, sr.score),
            rel(wr.pnorm_3, sr.pnorm_3),
        );
    }
}
