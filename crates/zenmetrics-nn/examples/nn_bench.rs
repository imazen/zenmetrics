//! Time one zenmetrics-nn model (`ZENMETRICS_NN_MODEL` = lpips_alex | lpips_vgg |
//! lpips_squeeze | stlpips_alex | stlpips_vgg | dists | deepdc, default lpips_vgg) on the ladder, in the modes `bar.py` uses for PyTorch:
//!
//! - `compute`: decoded RGB8 in memory → score (preprocess, two trunk runs, head)
//! - `warm`: reference taps cached once → score one candidate (one trunk run)
//! - `e2e`: read + PNG-decode both files → score
//!
//! ```text
//! cargo run --release -p zenmetrics-nn --example nn_bench -- \
//!     <model_dir> <ladder_dir> <out.tsv> [image=1415] [sizes=64,...,2048] [budget_s=4]
//! ```
//! Engine: `ZENMETRICS_NN_ENGINE` = `rten` (default) or `ort-cuda` (feature
//! `ort`, needs `ORT_DYLIB_PATH`). Thread count: rten's pool (`RTEN_NUM_THREADS`).

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Evaluation, Image, Metric};
use zenmetrics_nn::{Ckdn, DeepDc, Dists, Engine, Lpips, LpipsNet, WeightsConsent};

fn load_rgb8(path: &Path) -> (Vec<u8>, usize, usize) {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let out = zenpng::decode(
        &data,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .expect("png decode");
    let s = out.pixels.as_slice();
    let (w, h) = (s.width() as usize, s.rows() as usize);
    assert_eq!(
        s.descriptor().bytes_per_pixel(),
        3,
        "{}: expected RGB8",
        path.display()
    );
    let bytes = s.as_strided_bytes();
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        px.extend_from_slice(&bytes[y * s.stride()..y * s.stride() + w * 3]);
    }
    (px, w, h)
}

fn image(px: &[u8], w: usize, h: usize) -> Image<'_> {
    Image::from(PixelSlice::new(px, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap())
}

fn score(e: &dyn Evaluation, col: &str) -> f64 {
    e.scores().find(|(k, _)| k == col).expect("score").1
}

/// One warmup, then at least 3 and at most 50 timed runs within `budget_s`.
fn timed(budget_s: f64, mut f: impl FnMut() -> f64) -> (Vec<f64>, f64) {
    let mut score = f();
    let end = Instant::now() + std::time::Duration::from_secs_f64(budget_s);
    let mut ts = Vec::new();
    while ts.len() < 3 || (Instant::now() < end && ts.len() < 50) {
        let t = Instant::now();
        score = f();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    (ts, score)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (model, ladder, out) = (Path::new(&args[1]), Path::new(&args[2]), &args[3]);
    let stem = args.get(4).map_or("1415", String::as_str);
    let sizes: Vec<usize> = args
        .get(5)
        .map_or("64,128,256,512,1024,2048", String::as_str)
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let budget: f64 = args.get(6).map_or(4.0, |s| s.parse().unwrap());
    let host = std::fs::read_to_string("/etc/hostname")
        .unwrap_or_default()
        .trim()
        .to_string();
    let threads = std::env::var("RTEN_NUM_THREADS").unwrap_or_else(|_| "default".into());

    let engine_name = std::env::var("ZENMETRICS_NN_ENGINE").unwrap_or_else(|_| "rten".into());
    let engine = match engine_name.as_str() {
        "rten" => Engine::Rten,
        #[cfg(feature = "ort")]
        "ort-cuda" => Engine::OrtCuda,
        other => panic!("unknown or unbuilt engine {other}"),
    };
    let col: &'static str = match std::env::var("ZENMETRICS_NN_MODEL").as_deref() {
        Err(_) | Ok("lpips_vgg") => "lpips_vgg",
        Ok("lpips_alex") => "lpips_alex",
        Ok("lpips_squeeze") => "lpips_squeeze",
        Ok("dists") => "dists",
        Ok("deepdc") => "deepdc",
        Ok("ckdn") => "ckdn",
        Ok("stlpips_alex") => "stlpips_alex",
        Ok("stlpips_vgg") => "stlpips_vgg",
        Ok(other) => panic!("unknown ZENMETRICS_NN_MODEL {other}"),
    };
    let mut m: Box<dyn Metric> = if col == "dists" {
        Box::new(Dists::from_dir(model, engine).expect("load model"))
    } else if col == "deepdc" {
        Box::new(DeepDc::from_dir(model, engine).expect("load model"))
    } else if col == "ckdn" {
        let consent = WeightsConsent::new().accept(Ckdn::WEIGHTS);
        Box::new(Ckdn::from_dir(model, engine, &consent).expect("load model"))
    } else {
        let net = LpipsNet::ALL
            .into_iter()
            .find(|n| n.column() == col)
            .unwrap();
        Box::new(Lpips::from_dir(model, net, engine).expect("load model"))
    };
    let new = !Path::new(out).exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .unwrap();
    if new {
        writeln!(
            f,
            "host\tengine\trten_threads\tmetric\tmode\tsize\tn\tmedian_ms\tmin_ms\tscore"
        )
        .unwrap();
    }
    for &n in &sizes {
        let rp_path = ladder.join(format!("{stem}_{n}_ref.png"));
        let dp_path = ladder.join(format!("{stem}_{n}_dist.png"));
        let (rp, w, h) = load_rgb8(&rp_path);
        let (dp, _, _) = load_rgb8(&dp_path);
        let modes: [(
            &str,
            Box<dyn FnMut(&mut Box<dyn Metric>) -> (Vec<f64>, f64)>,
        ); 3] = [
            (
                "compute",
                Box::new(|m| {
                    timed(budget, || {
                        score(
                            &*m.evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                                .unwrap(),
                            col,
                        )
                    })
                }),
            ),
            (
                "warm",
                Box::new(|m| {
                    let mut warm = m.warm_reference(image(&rp, w, h), EvalOpts::new()).unwrap();
                    timed(budget, || {
                        score(
                            &*warm.evaluate(image(&dp, w, h), EvalOpts::new()).unwrap(),
                            col,
                        )
                    })
                }),
            ),
            (
                "e2e",
                Box::new(|m| {
                    timed(budget, || {
                        let (r, w, h) = load_rgb8(&rp_path);
                        let (d, _, _) = load_rgb8(&dp_path);
                        score(
                            &*m.evaluate(image(&r, w, h), image(&d, w, h), EvalOpts::new())
                                .unwrap(),
                            col,
                        )
                    })
                }),
            ),
        ];
        for (mode, mut run) in modes {
            let (mut ts, value) = run(&mut m);
            ts.sort_by(f64::total_cmp);
            let med = ts[ts.len() / 2];
            let row = format!(
                "{host}\t{engine_name}\t{threads}\t{col}\t{mode}\t{n}\t{}\t{med:.3}\t{:.3}\t{value:.7}",
                ts.len(),
                ts[0]
            );
            println!("{row}");
            writeln!(f, "{row}").unwrap();
        }
    }
}
