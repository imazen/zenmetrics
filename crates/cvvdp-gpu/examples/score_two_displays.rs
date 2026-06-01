//! Score one (reference, distorted) RGB image pair under two named
//! display presets and print both JODs + the delta. Used for the
//! phone-vs-desktop RD probe (sidesteps the zen-metrics CLI).
//!
//! Run:
//!     cargo run --release -p cvvdp-gpu \
//!         --features cuda,cubecl-types --no-default-features \
//!         --example score_two_displays -- <ref.png> <dist.png> [d1] [d2]
//!
//! Defaults: d1=standard_4k, d2=modern_oled_phone_indoor.
//! Output (one line, tab-separated): jod_d1<TAB>jod_d2<TAB>delta

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel};
use std::path::Path;

fn load_rgb8(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    (img.into_raw(), w, h)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <ref> <dist> [display1=standard_4k] [display2=modern_oled_phone_indoor]",
            args[0]
        );
        std::process::exit(2);
    }
    let ref_path = Path::new(&args[1]);
    let dist_path = Path::new(&args[2]);
    let d1 = args.get(3).map(String::as_str).unwrap_or("standard_4k");
    let d2 = args
        .get(4)
        .map(String::as_str)
        .unwrap_or("modern_oled_phone_indoor");

    let (ref_b, rw, rh) = load_rgb8(ref_path);
    let (dist_b, dw, dh) = load_rgb8(dist_path);
    assert_eq!((rw, rh), (dw, dh), "ref/dist size mismatch");

    let client = CudaRuntime::client(&Default::default());

    let mut score = |name: &str| -> f64 {
        let display =
            DisplayModel::by_name(name).unwrap_or_else(|| panic!("unknown display preset {name}"));
        let geom = DisplayGeometry::by_name(name)
            .unwrap_or_else(|| panic!("no geometry for preset {name}"));
        let params = CvvdpParams {
            display,
            ..CvvdpParams::PLACEHOLDER
        };
        let mut cvvdp =
            Cvvdp::<CudaRuntime>::new_with_geometry(client.clone(), rw, rh, params, geom)
                .expect("Cvvdp::new_with_geometry");
        cvvdp.score(&ref_b, &dist_b).expect("score")
    };

    let jod_d1 = score(d1);
    let jod_d2 = score(d2);

    // Machine-readable line for the probe harness.
    println!("{jod_d1:.6}\t{jod_d2:.6}\t{:.6}", jod_d2 - jod_d1);
}
