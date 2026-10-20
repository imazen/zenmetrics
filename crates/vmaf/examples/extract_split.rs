//! Per-extractor wall time for one v0.6.1 / v1 frame.
//!
//!   cargo run --release -p vmaf --features simd,avx512,parallel \
//!       --example extract_split -- <WxH> [v0|v1|both]

use std::hint::black_box;
use std::time::Instant;

fn synth(w: usize, h: usize, distorted: bool) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut y = vec![0u16; w * h];
    let mut u = vec![0u16; w * h / 4];
    let mut v = vec![0u16; w * h / 4];
    for r in 0..h {
        for c in 0..w {
            let mut val = 16 + ((c * 3 + r * 5 + (c / 17) * 9) % 220) as u16;
            if distorted {
                val = val.saturating_add_signed(((c % 7) as i16) - 3);
            }
            y[r * w + c] = val;
        }
    }
    let jitter = if distorted { 4u16 } else { 0u16 };
    for r in 0..h / 2 {
        for c in 0..w / 2 {
            let i = r * (w / 2) + c;
            u[i] = 100 + ((c * 2 + r * 3) % 56) as u16 + jitter;
            v[i] = 120 + ((c * 5 + r * 2) % 56) as u16 + jitter;
        }
    }
    (y, u, v)
}

fn t<F: FnMut() -> R, R>(name: &str, mut f: F) {
    // warm
    black_box(f());
    let t = Instant::now();
    black_box(f());
    println!("{name:>16}: {:8.2} ms", t.elapsed().as_secs_f64() * 1e3);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (w, h) = args
        .get(1)
        .and_then(|s| s.split_once('x'))
        .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))
        .unwrap_or((3354, 2516));
    let which = args.get(2).map_or("both", |s| s.as_str());
    let (ry, ru, rv) = synth(w, h, false);
    let (dy, du, dv) = synth(w, h, true);
    eprintln!(
        "{w}x{h} RAYON_NUM_THREADS={:?}",
        std::env::var("RAYON_NUM_THREADS")
    );

    if which != "v1" {
        t("v0 adm2", || {
            black_box(vmaf::adm2_v0_from_luma(
                &ry,
                &dy,
                w,
                h,
                8,
                vmaf::VmafV0Variant::Standard,
            ))
        });
        t("v0 vif", || {
            black_box(vmaf::vif_v0_from_luma(
                &ry,
                &dy,
                w,
                h,
                8,
                vmaf::VmafV0Variant::Standard,
            ))
        });
    }
    if which != "v0" {
        t("v1 cambi", || {
            black_box(vmaf::cambi_v1_from_luma(&dy, w, h, 8))
        });
        t("v1 speed_uv", || {
            black_box(vmaf::speed_v1_chroma_420(
                &ru,
                &rv,
                &du,
                &dv,
                w,
                h,
                8,
                vmaf::ModelVariant::Standard1080p,
            ))
        });
        t("v1 adm3", || {
            black_box(vmaf::adm3_v1_from_luma(
                &ry,
                &dy,
                w,
                h,
                8,
                vmaf::ModelVariant::Standard1080p,
            ))
        });
    }
}
