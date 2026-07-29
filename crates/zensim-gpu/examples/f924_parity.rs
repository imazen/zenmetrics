//! CPU-vs-GPU parity harness for the 924-feature `Folded720Append` regime.
//!
//! zensim (CPU) emits 924 features: `[0..156) v1-basic`, `[156..372) = 0`
//! (deprecated), `[372..720) v2-348`, `[720..924) append-204`. zensim-gpu
//! currently tops out at the v1 372-feature layout, so this reports the gap
//! BLOCK BY BLOCK rather than as one number — a port of this size needs a gate
//! that says which part is wrong, not just that something is.
//!
//! Run: cargo run --release -p zensim-gpu --no-default-features \
//!        --features "wgpu,cubecl-types" --example f924_parity

use zensim::feature_v2::FeatureRegime;

fn synth(w: usize, h: usize, seed: u32, shift: f32) -> Vec<[u8; 3]> {
    let mut s = seed | 1;
    let mut v: Vec<[u8; 3]> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = ((s >> 24) as f32) / 255.0;
            // Structured content (gradients + edges) plus noise, so the
            // feature vector is not degenerate.
            let g = (x as f32 / w as f32) * 0.6 + (y as f32 / h as f32) * 0.3;
            let edge = if (x / 16 + y / 16) % 2 == 0 { 0.15 } else { 0.0 };
            let px = |c: usize| {
                let base = g + edge + n * 0.08 + c as f32 * 0.02 + shift;
                (base.clamp(0.0, 1.0) * 255.0) as u8
            };
            v.push([px(0), px(1), px(2)]);
        }
    }
    v
}

fn main() {
    const W: usize = 256;
    const H: usize = 256;
    let refe = synth(W, H, 7, 0.0);
    let dist = synth(W, H, 7, 0.03);

    // ---- CPU oracle at the 924 regime ----
    let z = zensim::Zensim::new(zensim::ZensimProfile::latest_preview());
    let src = zensim::RgbSlice::new(&refe, W, H);
    let dst = zensim::RgbSlice::new(&dist, W, H);
    // `compute_v2_features*` yields the 348-wide V2Bounded regime; the 924
    // layout comes from the FOLDED entry point.
    let cpu = z
        .compute_folded720_append_features(&src, &dst)
        .expect("cpu folded720+append");
    let f = cpu.features();
    println!("CPU regime      : {:?}", cpu.regime());
    println!("CPU feature len : {}", f.len());
    println!("CPU n_scales    : {}", cpu.n_scales());
    assert_eq!(
        cpu.regime(),
        FeatureRegime::Folded720Append,
        "expected the 924 regime"
    );

    let block = |name: &str, lo: usize, hi: usize| {
        let s = &f[lo..hi.min(f.len())];
        let nz = s.iter().filter(|v| **v != 0.0).count();
        let mx = s.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        println!("  {name:<22} [{lo:>4}..{hi:>4})  nonzero={nz:>4}/{:<4} max|v|={mx:.6}", s.len());
    };
    println!("\nCPU 924 layout:");
    block("v1-basic", 0, 156);
    block("deprecated-zeros", 156, 372);
    block("v2-348", 372, 720);
    block("append-204", 720, 924);

    // ---- GPU side: what does it actually produce today? ----
    use cubecl::wgpu::WgpuRuntime;
    use cubecl::prelude::*;
    type Backend = WgpuRuntime;

    let flat_ref: Vec<u8> = refe.iter().flatten().copied().collect();
    let flat_dst: Vec<u8> = dist.iter().flatten().copied().collect();
    let client = Backend::client(&Default::default());
    let mut zg = zensim_gpu::Zensim::<Backend>::new(client, W as u32, H as u32)
        .expect("zensim-gpu init");
    let gpu = zg.compute_features(&flat_ref, &flat_dst).expect("gpu features");

    println!("\nGPU output:");
    println!("  len = {} (regime = v1 layout)", gpu.len());

    // LAYOUT — established by measurement, not by reading the constants.
    //
    // `FEATURES_PER_SCALE = 19 * 3` in lib.rs describes a CONCEPTUAL
    // [scale][channel][13 basic ++ 6 peak] grouping, but the emitted array is
    // BLOCK-structured: [0..156) basic, then [156..228) peaks. Those are not
    // the same order, and the difference matters for the port.
    //
    // Proven at the boundary rather than assumed: indices 150..155 agree
    // between CPU and GPU to f32 precision, and at exactly 156 the CPU goes to
    // 0.0 (the folded layout's deprecated block) while the GPU continues with
    // nonzero peak values. An interleaved reading would have mismatched from
    // index 13 onward; it does not.
    //
    // Consequence: GPU [0..156) already IS the folded v1-basic block, and the
    // GPU peak block [156..228) is exactly what the folded layout discards.
    let n = 156.min(gpu.len());
    let mut worst = 0.0f64;
    let mut worst_i = 0usize;
    for i in 0..n {
        let d = (f[i] - gpu[i] as f64).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    println!("\nv1-basic block [0..156) CPU-vs-GPU (index-aligned):");
    println!("  max |diff| = {worst:.6e} at index {worst_i}   (f32 GPU vs f64 CPU)");

    let peak_nz = gpu[156..].iter().filter(|v| **v != 0.0).count();
    println!("  GPU [156..228) peak block: {peak_nz}/{} nonzero — discarded by", gpu.len() - 156);
    println!("  the folded layout, which emits 0.0 across [156..372).");

    println!("\nPORT REMAINING:");
    println!("  [ 372.. 720) v2-348    — not implemented on GPU");
    println!("  [ 720.. 924) append-204 — not implemented on GPU");
    println!("  [ 156.. 372) must emit 0 in the folded layout");
}
