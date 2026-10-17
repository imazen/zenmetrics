//! Measure loading and two evaluations at an explicitly requested geometry.
use std::time::Instant;
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric};
use zenmetrics_nn::{Engine, PieApp, WeightsConsent};
fn main() {
    let a: Vec<_> = std::env::args().collect();
    assert_eq!(a.len(), 5, "MODEL_DIR ENGINE WIDTH HEIGHT");
    let w: usize = a[3].parse().unwrap();
    let h: usize = a[4].parse().unwrap();
    let engine = match a[2].as_str() {
        "rten" => Engine::Rten,
        #[cfg(feature = "ort")]
        "ort-cuda" => Engine::OrtCuda,
        #[cfg(feature = "ort-webgpu")]
        "ort-webgpu" => Engine::OrtWebgpu,
        _ => panic!("unbuilt engine"),
    };
    let start = Instant::now();
    let mut model = PieApp::from_dir(
        &a[1],
        engine,
        &WeightsConsent::new().accept(PieApp::WEIGHTS),
    )
    .unwrap();
    println!("load_seconds={:.6}", start.elapsed().as_secs_f64());
    let r: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i * 7 + i / w * 13) ^ (i % 251)) as u8)
        .collect();
    let d: Vec<u8> = r
        .iter()
        .enumerate()
        .map(|(i, &v)| v.wrapping_add((i % 11) as u8))
        .collect();
    let image = |pixels| {
        Image::from(
            PixelSlice::new(
                pixels,
                w as u32,
                h as u32,
                w * 3,
                PixelDescriptor::RGB8_SRGB,
            )
            .unwrap(),
        )
    };
    let mut first = None;
    for call in 0..2 {
        let now = Instant::now();
        let result = model
            .evaluate(image(&r), image(&d), EvalOpts::new())
            .unwrap();
        let score = result.scores().next().unwrap().1;
        println!(
            "call={call} seconds={:.6} score={score:.17} bits={:016x}",
            now.elapsed().as_secs_f64(),
            score.to_bits()
        );
        if let Some(bits) = first {
            assert_eq!(score.to_bits(), bits);
        }
        first = Some(score.to_bits());
    }
    println!("total_seconds={:.6}", start.elapsed().as_secs_f64());
}
