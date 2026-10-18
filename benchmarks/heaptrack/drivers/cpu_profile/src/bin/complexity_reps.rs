//! Repeat one CPU metric on one prepared image pair, for hardware-counter deltas.
//!
//! `complexity-reps <W>x<H> <metric> <reps>` builds the zenmetrics-corpus photo
//! pair (source.png vs q20.jpg, mirror-tiled, as `new-metrics-wall` does with
//! NMW_CONTENT=photo), then scores it `reps` times through the CLI dispatch
//! (`run_metric_display`). Run it under `perf stat` at two rep counts: the
//! difference divided by the extra reps is the per-call cost with decode, image
//! preparation and process start removed. Prints nothing on the hot path; the
//! last score goes to stderr so reps can be checked for identical output.
//! Method: docs/IQA_COMPLEXITY.md.

use std::hint::black_box;
use std::sync::OnceLock;

use zenmetrics_cli::decode::{Rgb8Image, decode_image_to_rgb8};
use zenmetrics_cli::metrics::display::{CvvdpDisplay, parse_display};
use zenmetrics_cli::metrics::{GpuRuntime, MetricKind, run_metric_display};

fn mirror(i: usize, n: usize) -> usize {
    let m = i % (2 * n);
    if m < n { m } else { 2 * n - 1 - m }
}

fn tile(src: &Rgb8Image, w: u32, h: u32) -> Rgb8Image {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let (wu, hu) = (w as usize, h as usize);
    let mut pixels = vec![0u8; wu * hu * 3];
    for y in 0..hu {
        let sy = mirror(y, sh);
        for x in 0..wu {
            let si = (sy * sw + mirror(x, sw)) * 3;
            let di = (y * wu + x) * 3;
            pixels[di..di + 3].copy_from_slice(&src.pixels[si..si + 3]);
        }
    }
    Rgb8Image {
        pixels,
        width: w,
        height: h,
    }
}

fn display() -> &'static CvvdpDisplay {
    static DISPLAY: OnceLock<CvvdpDisplay> = OnceLock::new();
    DISPLAY.get_or_init(|| parse_display("standard_fhd").expect("standard_fhd display"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: complexity-reps <W>x<H> <metric> <reps>");
        std::process::exit(64);
    }
    let (w, h) = args[1]
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
        .unwrap_or_else(|| panic!("bad size {}", args[1]));
    let kind = *MetricKind::all()
        .iter()
        .find(|k| k.name() == args[2])
        .unwrap_or_else(|| panic!("unknown metric {}", args[2]));
    let reps: usize = args[3].parse().expect("reps");
    let src = decode_image_to_rgb8(&zenmetrics_corpus::source_png()).expect("decode source.png");
    let dist =
        decode_image_to_rgb8(&zenmetrics_corpus::jpeg_at_quality(20)).expect("decode q20.jpg");
    let (r, d) = (tile(&src, w, h), tile(&dist, w, h));
    if let Ok(dir) = std::env::var("COMPLEXITY_PAIR_OUT") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).expect("fixture directory");
        std::fs::write(dir.join("reference.rgb"), &r.pixels).expect("reference fixture");
        std::fs::write(dir.join("distorted.rgb"), &d.pixels).expect("distorted fixture");
        return;
    }
    let mut last = f64::NAN;
    for _ in 0..reps {
        let cols = run_metric_display(
            kind,
            black_box(&r),
            black_box(&d),
            GpuRuntime::Cpu,
            Some(display()),
        )
        .unwrap_or_else(|e| {
            eprintln!("REFUSED {}: {e}", args[2]);
            std::process::exit(2);
        });
        last = black_box(cols[0].1);
    }
    eprintln!("{} {}x{} reps={} score={}", args[2], w, h, reps, last);
}
