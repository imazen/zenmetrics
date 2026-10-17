//! `complexity-reps`-compatible driver for `scripts/complexity/perf_deltas.py`:
//!
//! ```text
//! pyiqa_ports_reps <WxH> <metric> <n>
//! ```
//!
//! `<metric>` is a CLI name (`wadiqam-pyiqa`, `ahiq-pyiqa`, `cw-ssim-pyiqa`,
//! `dmm-pyiqa`, `pieapp-pyiqa`). Loads the model and the pair once, then
//! scores the pair `n` times through `za::Metric::evaluate` (cold: both
//! images through the whole pipeline each call) and prints
//! `<metric> score=<s>`. Env: `PYIQA_PORTS_MODEL_DIR` (the metric's export
//! directory; unused by cw-ssim-pyiqa), `PYIQA_PORTS_ENGINE` (rten | ort-cpu
//! | ort-cuda, default rten), `PYIQA_PORTS_REF` / `PYIQA_PORTS_DIST` (8-bit
//! RGB PNGs of size `<WxH>`). Accepts the non-commercial weights' terms for
//! the run (a benchmark tool, not a consent path).

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric};
use zenmetrics_nn::WeightsConsent;
use zenmetrics_pyiqa_ports::{AhiqPyiqa, CwSsimPyiqa, DmmPyiqa, Engine, PieAppPyiqa, WadiqamPyiqa};

fn load(path: &str) -> (Vec<u8>, u32, u32) {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let out = zenpng::decode(
        &data,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .unwrap_or_else(|e| panic!("{path}: {e}"));
    let s = out.pixels.as_slice();
    assert_eq!(s.descriptor().bytes_per_pixel(), 3, "{path}: expected RGB8");
    let (w, h) = (s.width(), s.rows());
    let bytes = s.as_strided_bytes();
    let mut px = Vec::with_capacity(w as usize * h as usize * 3);
    for y in 0..h as usize {
        px.extend_from_slice(&bytes[y * s.stride()..][..w as usize * 3]);
    }
    (px, w, h)
}

fn image(px: &[u8], w: u32, h: u32) -> Image<'_> {
    Image::from(PixelSlice::new(px, w, h, w as usize * 3, PixelDescriptor::RGB8_SRGB).unwrap())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [shape, metric, n] = a.as_slice() else {
        panic!("usage: pyiqa_ports_reps <WxH> <metric> <n>");
    };
    let n: usize = n.parse().expect("n");
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} unset"));
    let engine = Engine::parse(&std::env::var("PYIQA_PORTS_ENGINE").unwrap_or("rten".into()))
        .expect("PYIQA_PORTS_ENGINE");
    let dir = std::env::var("PYIQA_PORTS_MODEL_DIR").unwrap_or(".".into());
    let consent = WeightsConsent::new()
        .accept(WadiqamPyiqa::WEIGHTS)
        .accept(AhiqPyiqa::WEIGHTS)
        .accept(PieAppPyiqa::WEIGHTS);
    let (mut m, col): (Box<dyn Metric>, &str) = match metric.as_str() {
        "wadiqam-pyiqa" => (
            Box::new(WadiqamPyiqa::from_dir(&dir, engine, &consent).unwrap()),
            "wadiqam_pyiqa",
        ),
        "ahiq-pyiqa" => (
            Box::new(AhiqPyiqa::from_dir(&dir, engine, &consent).unwrap()),
            "ahiq_pyiqa",
        ),
        "cw-ssim-pyiqa" => (Box::new(CwSsimPyiqa::new()), "cw_ssim_pyiqa"),
        "dmm-pyiqa" => (
            Box::new(DmmPyiqa::from_dir(&dir, engine).unwrap()),
            "dmm_pyiqa",
        ),
        "pieapp-pyiqa" => (
            Box::new(PieAppPyiqa::from_dir(&dir, engine, &consent).unwrap()),
            "pieapp_pyiqa",
        ),
        other => panic!("unknown metric {other}"),
    };
    let (rp, w, h) = load(&env("PYIQA_PORTS_REF"));
    let (dp, dw, dh) = load(&env("PYIQA_PORTS_DIST"));
    assert_eq!(format!("{w}x{h}"), *shape, "PYIQA_PORTS_REF size");
    assert_eq!((w, h), (dw, dh));
    let mut last = f64::NAN;
    for _ in 0..n {
        let e = m
            .evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
            .unwrap();
        last = e.scores().find(|(k, _)| *k == col).unwrap().1;
    }
    println!("{metric} score={last:?}");
}
