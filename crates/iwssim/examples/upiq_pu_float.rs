//! Float-PU feeding A/B for IW-SSIM on the UPIQ HDR subset.
//!
//! Tests the fix identified in `benchmarks/pu_integrated_upiq_2026-06-09.md`:
//! feed `PU21(bt709-luma(nits)) · 255 / PU21(peak)` as **float** gray planes
//! into [`iwssim::Iwssim::score_gray`] — same 0..255 scale the u8 path
//! produces, but with no quantization round-trip. Runs each pair twice:
//! `iw_flag = true` (PU-IW-SSIM) and `iw_flag = false` (plain PU-MS-SSIM,
//! the first-party control). See imazen/zenmetrics#25.
//!
//!   cargo run --release -p iwssim --example upiq_pu_float -- \
//!     /mnt/v/output/zenmetrics/upiq-pu/upiq_pairs.tsv \
//!     /mnt/v/output/zenmetrics/upiq-pu/pu_iwssim_float.tsv

use std::collections::HashMap;
use std::io::Write;

use iwssim::{Iwssim, IwssimParams};

// gfxdisp/pu21 banding_glare — the parity-guarded coefficient set (see
// zenmetrics-api::hdr reference_parity_gfxdisp_goldens).
const P: [f32; 7] = [
    0.353_487_9,
    0.373_465_86,
    8.277_049e-5,
    0.906_256_26,
    0.091_503_03,
    0.909_951_7,
    596.314_8,
];

fn pu21(y: f32) -> f32 {
    let y = y.clamp(0.005, 10000.0);
    let yp = y.powf(P[3]);
    let inner = (P[0] + P[1] * yp) / (1.0 + P[2] * yp);
    (P[6] * (inner.powf(P[4]) - P[5])).max(0.0)
}

/// EXR (absolute cd/m²) → PU-encoded bt709-luma gray f32 in 0..255 scale
/// (255/PU(1000-nit peak), matching the production PuRescale shell minus
/// the u8 rounding).
fn load_pu_gray(path: &str, pu_max: f32) -> Result<(Vec<f32>, u32, u32), String> {
    let img = image::open(path)
        .map_err(|e| format!("{path}: {e}"))?
        .to_rgb32f();
    let (w, h) = (img.width(), img.height());
    let raw = img.into_raw();
    let scale = 255.0 / pu_max;
    let gray = raw
        .chunks_exact(3)
        .map(|c| pu21(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) * scale)
        .collect();
    Ok((gray, w, h))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pairs = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/mnt/v/output/zenmetrics/upiq-pu/upiq_pairs.tsv".into());
    let out_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/mnt/v/output/zenmetrics/upiq-pu/pu_iwssim_float.tsv".into());
    let pu_max = pu21(1000.0).max(1.0);

    let body = std::fs::read_to_string(&pairs).expect("read pairs tsv");
    let mut cache: HashMap<String, (Vec<f32>, u32, u32)> = HashMap::new();
    let mut pipes: HashMap<(u32, u32, bool), Iwssim> = HashMap::new();
    let mut out = String::from("ref_path\tdist_path\tpu_iwssim_float\tpu_msssim_float\n");
    let (mut ok, mut err) = (0usize, 0usize);

    for line in body.lines().skip(1) {
        let mut it = line.split('\t');
        let (Some(rp), Some(dp)) = (it.next(), it.next()) else {
            continue;
        };
        for p in [rp, dp] {
            if !cache.contains_key(p) {
                match load_pu_gray(p, pu_max) {
                    Ok(v) => {
                        cache.insert(p.to_string(), v);
                    }
                    Err(e) => eprintln!("LOAD FAIL {e}"),
                }
            }
        }
        let (Some((r, rw, rh)), Some((d, dw, dh))) = (cache.get(rp), cache.get(dp)) else {
            err += 1;
            continue;
        };
        if (rw, rh) != (dw, dh) {
            err += 1;
            continue;
        }
        let mut scores = [0.0f64; 2];
        let mut failed = false;
        for (slot, iw) in [(0usize, true), (1usize, false)] {
            let pipe = pipes.entry((*rw, *rh, iw)).or_insert_with(|| {
                let params = IwssimParams {
                    iw_flag: iw,
                    ..IwssimParams::default()
                };
                Iwssim::with_params(*rw, *rh, params).expect("Iwssim::with_params")
            });
            match pipe.score_gray(r, d) {
                Ok(s) => scores[slot] = s.score,
                Err(e) => {
                    eprintln!("SCORE FAIL iw={iw} {rp}|{dp}: {e:?}");
                    failed = true;
                    break;
                }
            }
        }
        if failed {
            err += 1;
            continue;
        }
        out.push_str(&format!("{rp}\t{dp}\t{}\t{}\n", scores[0], scores[1]));
        ok += 1;
        if ok % 50 == 0 {
            eprintln!("scored {ok}…");
        }
    }

    let mut f = std::fs::File::create(&out_path).expect("create out");
    f.write_all(out.as_bytes()).expect("write out");
    eprintln!("done: {ok} scored, {err} errored -> {out_path}");
}
