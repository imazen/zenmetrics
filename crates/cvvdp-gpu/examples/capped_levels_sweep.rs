//! Capped-pyramid-levels parity sweep for `predict_jod_still_3ch_capped`.
//!
//! Measures the JOD drift introduced by truncating the cvvdp pyramid
//! at each cap depth (5..=9) on every fixture in
//! `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`. The natural
//! pyramid depth for each fixture is also printed for context.
//!
//! Purpose: feasibility check for `MemoryMode::Strip { capped_levels:
//! Option<u32> }` per the deferred-task plan. Strip processing of
//! 24 MP square images requires the halo per side to shrink, which
//! means dropping pyramid levels — but only ships if a cap depth fits
//! the canonical ≤ 0.005 JOD pycvvdp manifest parity gate. This
//! example produces the cap-depth-vs-JOD-diff data feeding that
//! ship/don't-ship decision.
//!
//! Run with:
//!
//!     cargo run -p cvvdp-gpu --features cubecl-types \
//!         --example capped_levels_sweep --release
//!
//! Output: CSV-formatted lines to stdout, schema
//! `fixture,natural_n_levels,cap,jod,golden,abs_diff`.

use cvvdp_gpu::host_scalar::predict_jod_still_3ch_capped;
use cvvdp_gpu::kernels::pyramid::band_frequencies;
use cvvdp_gpu::params::{DisplayGeometry, DisplayModel};

fn synth_pair_ref(w: usize, h: usize) -> Vec<u8> {
    let n = w * h * 3;
    let mut b = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let r = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            let g = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            let bb = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
            let i = (y * w + x) * 3;
            b[i] = r;
            b[i + 1] = g;
            b[i + 2] = bb;
        }
    }
    b
}

fn apply_offset_dist(ref_bytes: &[u8]) -> Vec<u8> {
    ref_bytes
        .chunks_exact(3)
        .flat_map(|p| {
            [
                p[0].saturating_sub(8),
                p[1].saturating_sub(4),
                p[2].saturating_add(12),
            ]
        })
        .collect()
}

fn synth_pair_odd_dim_ref(w: usize, h: usize) -> Vec<u8> {
    let n = w * h * 3;
    let mut b = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 8) % 256) as u8;
            let g = ((y * 16) % 256) as u8;
            let bb = (((x + y) * 12) % 256) as u8;
            let i = (y * w + x) * 3;
            b[i] = r;
            b[i + 1] = g;
            b[i + 2] = bb;
        }
    }
    b
}

fn fixtures() -> Vec<(&'static str, usize, usize, fn(usize, usize) -> Vec<u8>, fn(&[u8]) -> Vec<u8>, f32)> {
    // (name, w, h, ref_fn, dist_fn, pycvvdp_golden_jod)
    // pycvvdp_golden_jod values match scripts/cvvdp_goldens/pycvvdp_synth_goldens.json.
    // Only offset-distortion fixtures are included — they share the
    // (synth_pair_ref + apply_offset_dist) / (synth_pair_odd_dim_ref +
    // apply_offset_dist) construction. Chroma_shift, noise, and blur
    // fixtures use different ref/dist constructors not replicated here;
    // adding them later would extend the sweep but isn't required for
    // the cap-vs-JOD-drift trend (which is fixture-content-driven).
    vec![
        ("synth_128x128_offset", 128, 128, synth_pair_ref, apply_offset_dist, 9.456145286560059),
        ("synth_1024x1024_offset", 1024, 1024, synth_pair_ref, apply_offset_dist, 9.458330154418945),
        ("synth_1280x720_offset", 1280, 720, synth_pair_ref, apply_offset_dist, 9.454181671142578),
        ("synth_720x1280_offset", 720, 1280, synth_pair_ref, apply_offset_dist, 9.44536018371582),
        ("synth_4000x3000", 4000, 3000, synth_pair_ref, apply_offset_dist, 9.458027839660645),
        ("synth_73x91_odd", 73, 91, synth_pair_odd_dim_ref, apply_offset_dist, 9.39037036895752),
    ]
}

fn main() {
    let display = DisplayModel::STANDARD_4K;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();

    println!("fixture,natural_n_levels,cap,jod,golden,abs_diff");
    for (name, w, h, ref_fn, dist_fn, golden) in fixtures() {
        let ref_bytes = ref_fn(w, h);
        let dist_bytes = dist_fn(&ref_bytes);

        // Pyramid `band_frequencies` returns n+1 entries (the half-ppd
        // base + n reduces); `host_scalar` consumes `freqs.len()` as
        // the band count, so natural_n = freqs.len().
        let natural_n = band_frequencies(ppd, w, h).len();

        for cap in 5..=9 {
            let jod = predict_jod_still_3ch_capped(
                &ref_bytes,
                &dist_bytes,
                w,
                h,
                display,
                ppd,
                Some(cap),
            );
            let abs_diff = (jod - golden).abs();
            println!(
                "{name},{natural_n},{cap},{jod:.6},{golden:.6},{abs_diff:.6}"
            );
        }
        // Uncapped reference
        let jod_full = predict_jod_still_3ch_capped(
            &ref_bytes,
            &dist_bytes,
            w,
            h,
            display,
            ppd,
            None,
        );
        let abs_diff = (jod_full - golden).abs();
        println!("{name},{natural_n},NATURAL,{jod_full:.6},{golden:.6},{abs_diff:.6}");
    }
}
