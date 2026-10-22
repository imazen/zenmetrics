//! Fit each metric map's placement on butteraugli's scale for jxl-encoder's
//! perceptual loop (`src/loop_map.rs`).
//!
//! ```text
//! cargo run --release -p zenmetrics-browser --example loop_norm_calibrate -- \
//!     out.tsv --fit <fit images…> --hold <held-out images…>
//! ```
//!
//! Each image (center-cropped to at most 512×512, even sides) is encoded
//! with jxl-encoder at effort 7 and each distance in `DISTANCES`, and
//! decoded. For every pair, butteraugli's diffmap and every metric map's
//! error reading (`loop_error_map`) give per-8×8-block distances the way the
//! loop computes them (`loop_tile_norms`: 16-norm × 1.2). For the metric's
//! map `e`, a block's distance after normalization is
//! `scale · tile(e^exponent)`. For each exponent in `EXPONENTS`, the scale
//! that best matches butteraugli in log space is the mean log ratio over
//! blocks where both are positive and butteraugli's is at least `FLOOR`;
//! the exponent with the smallest log RMSE wins. Reported per map: the fit,
//! its log RMSE and Pearson r on the fit images, and the same with the
//! fitted values on the held-out images.
//!
//! Prints the `DEFAULTS` table for `src/loop_map.rs` to stdout.
//!
//! `--save-pairs DIR` (before `--fit`) instead writes the encoded pairs and
//! butteraugli's block distances to `DIR` and stops, so maps of modules
//! built outside this workspace can be fitted against the same encodes:
//! `pairs.tsv` (index, set, image, distance, width, height) and per pair
//! `NNN.ref.rgba`, `NNN.dist.rgba` (packed RGBA8) and `NNN.butter.f32`
//! (little-endian per-block distances, `ceil(w/8) × ceil(h/8)` row-major).

use rayon::prelude::*;
use std::fmt::Write as _;

const DISTANCES: [f32; 7] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const EXPONENTS: [f64; 22] = [
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75,
    2.0, 2.5, 3.0, 4.0,
];
/// Blocks where butteraugli sees less than this are left out of the fit:
/// the loop never acts on them, and their logs are dominated by noise.
const FLOOR: f64 = 0.05;

struct Pair {
    image: String,
    distance: f32,
    width: u32,
    height: u32,
    reference: Vec<u8>,
    distorted: Vec<u8>,
    /// Butteraugli's per-block distances.
    butter: Vec<f32>,
}

fn load(path: &str) -> (Vec<u8>, u32, u32) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let packed = zenmetrics_browser::decode_image(&bytes, path)
        .unwrap_or_else(|_| panic!("{path}: decode failed"));
    let (w, h) = (packed[0], packed[1]);
    let (cw, ch) = (w.min(512) & !1, h.min(512) & !1);
    let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
    let mut rgba = Vec::with_capacity((cw * ch * 4) as usize);
    for y in y0..y0 + ch {
        for x in x0..x0 + cw {
            let p = packed[(2 + y * w + x) as usize];
            rgba.extend([p as u8, (p >> 8) as u8, (p >> 16) as u8, 255]);
        }
    }
    (rgba, cw, ch)
}

fn pairs(images: &[String]) -> Vec<Pair> {
    let jobs: Vec<(String, f32)> = images
        .iter()
        .flat_map(|p| DISTANCES.map(|d| (p.clone(), d)))
        .collect();
    jobs.par_iter()
        .map(|(path, distance)| {
            let (reference, width, height) = load(path);
            let rgb: Vec<u8> = reference
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect();
            let file = jxl_encoder::LossyConfig::new(*distance)
                .with_effort(7)
                .encode(&rgb, width, height, jxl_encoder::PixelLayout::Rgb8)
                .expect("encode");
            let packed = zenmetrics_browser::decode_image(&file, "candidate.jxl")
                .unwrap_or_else(|_| panic!("{path}: decode of the encode failed"));
            let distorted: Vec<u8> = packed[2..].iter().flat_map(|p| p.to_le_bytes()).collect();
            let errors = zenmetrics_browser::loop_error_map(
                "butteraugli",
                "",
                &reference,
                &distorted,
                width,
                height,
                width as usize * 4,
                "",
            )
            .expect("butteraugli");
            let butter =
                zenmetrics_browser::loop_tile_norms(&errors, width as usize, height as usize);
            Pair {
                image: path.clone(),
                distance: *distance,
                width,
                height,
                reference,
                distorted,
                butter,
            }
        })
        .collect()
}

/// Per block of each pair: butteraugli's distance and, per exponent, the
/// metric map's `tile(e^exponent)`.
fn samples(
    pairs: &[Pair],
    metric: &str,
    map: &str,
) -> Result<Vec<(f64, [f64; EXPONENTS.len()])>, String> {
    let per_pair: Vec<Result<Vec<_>, String>> = pairs
        .par_iter()
        .map(|p| {
            let e = zenmetrics_browser::loop_error_map(
                metric,
                map,
                &p.reference,
                &p.distorted,
                p.width,
                p.height,
                p.width as usize * 4,
                "standard_4k",
            )?;
            let (w, h) = (p.width as usize, p.height as usize);
            let tiles: Vec<Vec<f32>> = EXPONENTS
                .iter()
                .map(|&k| {
                    let powed: Vec<f32> =
                        e.iter().map(|&v| (f64::from(v).powf(k)) as f32).collect();
                    zenmetrics_browser::loop_tile_norms(&powed, w, h)
                })
                .collect();
            Ok(p.butter
                .iter()
                .enumerate()
                .map(|(b, &t)| {
                    (
                        f64::from(t),
                        std::array::from_fn(|k| f64::from(tiles[k][b])),
                    )
                })
                .collect())
        })
        .collect();
    let mut out = Vec::new();
    for r in per_pair {
        out.extend(r?);
    }
    Ok(out)
}

/// Log-space fit quality of `scale · m` against butteraugli: RMSE, Pearson r
/// and the number of blocks used.
fn quality(rows: &[(f64, f64)], scale: f64) -> (f64, f64, usize) {
    let used: Vec<(f64, f64)> = rows
        .iter()
        .filter(|(b, m)| *b >= FLOOR && *m > 0.0)
        .map(|(b, m)| (b.ln(), (scale * m).ln()))
        .collect();
    let n = used.len() as f64;
    if used.is_empty() {
        return (f64::NAN, f64::NAN, 0);
    }
    let rmse = (used.iter().map(|(b, m)| (b - m).powi(2)).sum::<f64>() / n).sqrt();
    let (mb, mm) = (
        used.iter().map(|u| u.0).sum::<f64>() / n,
        used.iter().map(|u| u.1).sum::<f64>() / n,
    );
    let cov: f64 = used.iter().map(|(b, m)| (b - mb) * (m - mm)).sum();
    let vb: f64 = used.iter().map(|(b, _)| (b - mb).powi(2)).sum();
    let vm: f64 = used.iter().map(|(_, m)| (m - mm).powi(2)).sum();
    (rmse, cov / (vb * vm).sqrt(), used.len())
}

/// Write `--save-pairs` output (see the module docs).
fn save_pairs(dir: &str, sets: [(&str, &[Pair]); 2]) {
    std::fs::create_dir_all(dir).expect("pairs directory");
    let mut index = String::from("index\tset\timage\tdistance\twidth\theight\n");
    let mut i = 0;
    for (set, pairs) in sets {
        for p in pairs {
            let file = |ext: &str| format!("{dir}/{i:03}.{ext}");
            std::fs::write(file("ref.rgba"), &p.reference).expect("write");
            std::fs::write(file("dist.rgba"), &p.distorted).expect("write");
            let butter: Vec<u8> = p.butter.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(file("butter.f32"), butter).expect("write");
            writeln!(
                index,
                "{i:03}\t{set}\t{}\t{}\t{}\t{}",
                p.image, p.distance, p.width, p.height
            )
            .unwrap();
            i += 1;
        }
    }
    std::fs::write(format!("{dir}/pairs.tsv"), index).expect("write pairs.tsv");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = args
        .first()
        .expect("usage: out.tsv [--save-pairs DIR] --fit <images…> --hold <images…>");
    let save_dir = args
        .iter()
        .position(|a| a == "--save-pairs")
        .map(|i| args.get(i + 1).expect("--save-pairs DIR").clone());
    let fit_at = args.iter().position(|a| a == "--fit").expect("--fit");
    let hold_at = args.iter().position(|a| a == "--hold").expect("--hold");
    let fit_images = &args[fit_at + 1..hold_at];
    let hold_images = &args[hold_at + 1..];
    assert!(!fit_images.is_empty() && !hold_images.is_empty());
    let started = std::time::Instant::now();
    let fit_pairs = pairs(fit_images);
    let hold_pairs = pairs(hold_images);
    eprintln!(
        "{} fit and {} held-out pairs encoded in {:.1} s",
        fit_pairs.len(),
        hold_pairs.len(),
        started.elapsed().as_secs_f64()
    );
    if let Some(dir) = save_dir {
        save_pairs(&dir, [("fit", &fit_pairs), ("hold", &hold_pairs)]);
        eprintln!("pairs written to {dir}");
        return;
    }
    let catalog: Vec<String> =
        serde_json::from_str(&zenmetrics_browser::available_spatial_maps()).unwrap();
    let p0 = &fit_pairs[0];
    let mut tsv = String::from(
        "metric\tmap\tscale\texponent\tfit_log_rmse\tfit_log_r\tfit_blocks\thold_log_rmse\thold_log_r\thold_blocks\n",
    );
    let mut defaults = String::new();
    for metric in catalog.iter().filter(|m| m.as_str() != "fsimc") {
        // The map names this metric publishes, default first.
        let Ok(first) = zenmetrics_browser::spatial_map(
            metric,
            "",
            &p0.reference,
            &p0.distorted,
            p0.width,
            p0.height,
            p0.width as usize * 4,
            "standard_4k",
        ) else {
            eprintln!("{metric}: no map at {}×{}", p0.width, p0.height);
            continue;
        };
        let mut names: Vec<String> = serde_json::from_str(&first.names()).unwrap();
        let default = first.name();
        names.retain(|n| *n != default);
        names.insert(0, default);
        for map in names {
            let t = std::time::Instant::now();
            let fit = match samples(&fit_pairs, metric, &map) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{metric}/{map}: {e}");
                    continue;
                }
            };
            let mut best: Option<(f64, f64, f64)> = None; // rmse, scale, exponent
            for (k, &exponent) in EXPONENTS.iter().enumerate() {
                let logs: Vec<f64> = fit
                    .iter()
                    .filter(|(b, m)| *b >= FLOOR && m[k] > 0.0)
                    .map(|(b, m)| b.ln() - m[k].ln())
                    .collect();
                if logs.is_empty() {
                    continue;
                }
                let scale = (logs.iter().sum::<f64>() / logs.len() as f64).exp();
                let rows: Vec<(f64, f64)> = fit.iter().map(|(b, m)| (*b, m[k])).collect();
                let (rmse, _, _) = quality(&rows, scale);
                if best.is_none_or(|(r, _, _)| rmse < r) {
                    best = Some((rmse, scale, exponent));
                }
            }
            let Some((_, scale, exponent)) = best else {
                eprintln!("{metric}/{map}: no positive blocks");
                continue;
            };
            let k = EXPONENTS.iter().position(|&e| e == exponent).unwrap();
            let rows = |s: &[(f64, [f64; EXPONENTS.len()])]| {
                s.iter().map(|(b, m)| (*b, m[k])).collect::<Vec<_>>()
            };
            let (fr, fp, fnn) = quality(&rows(&fit), scale);
            let hold = samples(&hold_pairs, metric, &map).expect("held-out maps");
            let (hr, hp, hn) = quality(&rows(&hold), scale);
            writeln!(
                tsv,
                "{metric}\t{map}\t{scale:.6e}\t{exponent}\t{fr:.4}\t{fp:.4}\t{fnn}\t{hr:.4}\t{hp:.4}\t{hn}"
            )
            .unwrap();
            writeln!(
                defaults,
                "    LoopNorm {{ metric: {metric:?}, map: {map:?}, scale: {scale:.6e}, exponent: {exponent:?}, r: {hp:.4} }},"
            )
            .unwrap();
            eprintln!(
                "{metric}/{map}: scale {scale:.4e} exponent {exponent} · fit rmse {fr:.3} r {fp:.3} · held-out rmse {hr:.3} r {hp:.3} ({:.1} s)",
                t.elapsed().as_secs_f64()
            );
        }
    }
    std::fs::write(out, tsv).expect("write tsv");
    println!("{defaults}");
    eprintln!("done in {:.1} s", started.elapsed().as_secs_f64());
}
