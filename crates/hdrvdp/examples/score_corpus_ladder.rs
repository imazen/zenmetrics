//! Score a **real** image pair ladder end to end with the HDR-VDP-2.2 port.
//!
//! Uses `zenmetrics-corpus`: one 256×256 source PNG and six JPEG
//! re-compressions of it (q1 … q90) — real content, real codec distortion,
//! spanning visually broken to near-transparent.
//!
//! Each pair is scored twice, to exercise both halves of the model:
//!
//! * **SDR** — the sRGB code values driven through HDR-VDP-2's own
//!   `srgb-display` model (99 cd/m² peak + 1 cd/m² black), i.e. the content as
//!   seen on an ordinary monitor.
//! * **HDR** — the *same* content presented on a 1000 cd/m² display: the sRGB
//!   EOTF gives display-linear values, which are then scaled to absolute
//!   cd/m² and fed as `rgb-bt.709`. This is not a synthetic HDR image, it is
//!   the same picture at a real HDR display's light levels — enough to prove
//!   the absolute-luminance path runs end to end on real pixels, and to show
//!   that the metric's verdict moves with the presentation luminance.
//!
//! Run with:
//! ```text
//! cargo run --release -p hdrvdp --example score_corpus_ladder
//! ```
//!
//! **This is not the UPIQ validation.** It demonstrates the pipeline on real
//! pixels and gives a monotonicity gate; it does not establish that the port
//! reproduces the reference implementation's numbers. See zenmetrics#50
//! chunk 4.

use hdrvdp::{ColorEncoding, Params, hdrvdp, pix_per_deg};

/// The sRGB EOTF, code value in `[0,1]` → display-linear in `[0,1]`.
fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn load_rgb(path: &std::path::Path) -> (Vec<f64>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("decoding {}: {e}", path.display()))
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data: Vec<f64> = img
        .into_raw()
        .into_iter()
        .map(|v| v as f64 / 255.0)
        .collect();
    (data, w, h)
}

fn main() {
    // A 256×256 image on a 24" 1920×1200 display at 0.5 m.
    let ppd = pix_per_deg(24.0, [1920.0, 1200.0], 0.5);
    let par = Params::new(ppd);

    let (reference, w, h) = load_rgb(&zenmetrics_corpus::source_png());
    // The same content presented at a 1000 cd/m² peak.
    const HDR_PEAK_NITS: f64 = 1000.0;
    let ref_nits: Vec<f64> = reference
        .iter()
        .map(|v| srgb_to_linear(*v) * HDR_PEAK_NITS)
        .collect();

    println!("# hdrvdp 0.0.1 — real-corpus JPEG ladder");
    println!("# source: zenmetrics-corpus source.png, {w}×{h}");
    println!("# pix_per_deg: {ppd:.3} (24in 1920x1200 at 0.5m)");
    println!("# sdr = srgb-display model (99/1 cd/m2); hdr = same content at a 1000 cd/m2 peak");
    println!("jpeg_q\tsdr_q_mos\tsdr_p_det\tsdr_c_max\thdr_q_mos\thdr_p_det\thdr_c_max");

    for q in [1u32, 5, 20, 45, 70, 90] {
        let (test, tw, th) = load_rgb(&zenmetrics_corpus::jpeg_at_quality(q));
        assert_eq!((tw, th), (w, h), "q{q} differs in size from the source");

        let sdr = hdrvdp(&test, &reference, w, h, ColorEncoding::SrgbDisplay, &par)
            .unwrap_or_else(|e| panic!("q{q} sdr: {e}"));

        let test_nits: Vec<f64> = test
            .iter()
            .map(|v| srgb_to_linear(*v) * HDR_PEAK_NITS)
            .collect();
        let hdr = hdrvdp(&test_nits, &ref_nits, w, h, ColorEncoding::RgbBt709, &par)
            .unwrap_or_else(|e| panic!("q{q} hdr: {e}"));
        assert!(
            !hdr.input_looks_relative,
            "the HDR feed should be in absolute cd/m²"
        );

        println!(
            "{q}\t{:.4}\t{:.6}\t{:.4}\t{:.4}\t{:.6}\t{:.4}",
            sdr.q_mos, sdr.p_det, sdr.c_max, hdr.q_mos, hdr.p_det, hdr.c_max
        );
    }
}
