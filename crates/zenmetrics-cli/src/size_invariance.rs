//! Size-invariance validation harness.
//!
//! Property (user, 2026-06-06): a fixed (ref,dist) pair, downsampled and
//! rescored, must NOT fluctuate with size (≤ ~2pt), and must score to 1×1 with
//! no errors — for EVERY metric. Candidate small-image handling: reflect
//! (mirror) / tile pad up to the pyramid minimum (64px) before scoring.
//!
//! Two modes over the corpus at
//! `/mnt/v/zen/size-invariance-corpus/corpus/<image>/{ref,jpeg_q*,jxl_d*}/<W>x<H>.png`:
//!   * `downsample-pair` (DEFAULT, the invariance gate): take the base-size
//!     (256²) (ref,dist) pair, area-downsample BOTH to each size, rescore.
//!     Distortion is held fixed → drift IS the metric's size-artifact.
//!   * `encode-per-size`: score each independently-encoded size (proves
//!     error-free small-image scoring; drift conflates real per-size quality).
//!
//! Run: zenmetrics size-invariance [--corpus DIR] [--mode downsample-pair]

use crate::decode::{Rgb8Image, decode_image_to_rgb8};
use std::collections::BTreeMap;
use std::path::PathBuf;

const MIN_DIM: u32 = 64;

#[derive(clap::Args, Debug)]
pub struct SizeInvarianceArgs {
    #[arg(long, default_value = "/mnt/v/zen/size-invariance-corpus/corpus")]
    pub corpus: PathBuf,
    /// `downsample-pair` (default) or `encode-per-size`.
    #[arg(long, default_value = "downsample-pair")]
    pub mode: String,
    /// Min-dim floor for the invariant-region drift gate (smaller sizes are
    /// reported via min_px but excluded from the gate — degenerate).
    #[arg(long, default_value_t = 8)]
    pub floor: u32,
    /// Base square size to downsample from (downsample-pair mode).
    #[arg(long, default_value_t = 256)]
    pub base: u32,
}

#[derive(Clone, Copy, PartialEq)]
enum Pad {
    Raw,
    Mirror,
    Tile,
}
impl Pad {
    fn name(self) -> &'static str {
        match self {
            Pad::Raw => "raw",
            Pad::Mirror => "mirror",
            Pad::Tile => "tile",
        }
    }
}

fn refl(i: i64, n: i64) -> usize {
    if n <= 1 {
        return 0;
    }
    let p = 2 * (n - 1);
    let mut k = ((i % p) + p) % p;
    if k >= n {
        k = p - k;
    }
    k as usize
}

fn apply_pad(img: &Rgb8Image, strat: Pad) -> Rgb8Image {
    let (w, h) = (img.width, img.height);
    if strat == Pad::Raw || (w >= MIN_DIM && h >= MIN_DIM) {
        return Rgb8Image {
            pixels: img.pixels.clone(),
            width: w,
            height: h,
        };
    }
    let (bw, bh) = (w.max(MIN_DIM), h.max(MIN_DIM));
    let mut px = vec![0u8; (bw as usize) * (bh as usize) * 3];
    for y in 0..bh {
        for x in 0..bw {
            let (sx, sy) = match strat {
                Pad::Mirror => (refl(x as i64, w as i64), refl(y as i64, h as i64)),
                Pad::Tile => ((x % w) as usize, (y % h) as usize),
                Pad::Raw => unreachable!(),
            };
            let si = (sy * w as usize + sx) * 3;
            let di = ((y as usize) * bw as usize + x as usize) * 3;
            px[di..di + 3].copy_from_slice(&img.pixels[si..si + 3]);
        }
    }
    Rgb8Image {
        pixels: px,
        width: bw,
        height: bh,
    }
}

/// Area-average downscale to `n×n` (n ≤ source min dim).
fn area_resize_sq(img: &Rgb8Image, n: u32) -> Rgb8Image {
    let (sw, sh) = (img.width as usize, img.height as usize);
    let n = n as usize;
    let mut px = vec![0u8; n * n * 3];
    for oy in 0..n {
        let y0 = oy * sh / n;
        let y1 = ((oy + 1) * sh / n).max(y0 + 1);
        for ox in 0..n {
            let x0 = ox * sw / n;
            let x1 = ((ox + 1) * sw / n).max(x0 + 1);
            for c in 0..3 {
                let (mut s, mut cnt) = (0u32, 0u32);
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        s += img.pixels[(sy * sw + sx) * 3 + c] as u32;
                        cnt += 1;
                    }
                }
                px[(oy * n + ox) * 3 + c] = (s / cnt.max(1)) as u8;
            }
        }
    }
    Rgb8Image {
        pixels: px,
        width: n as u32,
        height: n as u32,
    }
}

type Scorer = fn(&Rgb8Image, &Rgb8Image) -> Result<f64, Box<dyn std::error::Error>>;

fn metric_table() -> Vec<(&'static str, Scorer)> {
    vec![
        ("zensim", crate::metrics::zensim::score as Scorer),
        ("ssim2", crate::metrics::ssim2::score as Scorer),
        (
            "butter_p3",
            (|r: &Rgb8Image, d: &Rgb8Image| {
                crate::metrics::butteraugli::score_both(r, d).map(|(_, p3)| p3)
            }) as Scorer,
        ),
    ]
}

fn fluct(v: &[(u32, f64)]) -> f64 {
    let (mn, mx) = v
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &(_, s)| (a.min(s), b.max(s)));
    mx - mn
}

fn square_sizes() -> Vec<u32> {
    let mut d: Vec<u32> = (1..=32).collect();
    d.extend((33..64).step_by(3));
    d.extend((64..=256).step_by(7));
    d.push(256);
    d.sort_unstable();
    d.dedup();
    d
}

pub fn run(args: &SizeInvarianceArgs) -> Result<(), Box<dyn std::error::Error>> {
    let images: Vec<String> = std::fs::read_dir(&args.corpus)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let metrics = metric_table();
    let pads = [Pad::Raw, Pad::Mirror, Pad::Tile];
    let downsample = args.mode != "encode-per-size";

    let mut drift: BTreeMap<(&str, &str), Vec<f64>> = BTreeMap::new();
    let mut errs: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut minpx: BTreeMap<(&str, &str), u32> = BTreeMap::new();

    for img in &images {
        let idir = args.corpus.join(img);
        let refdir = idir.join("ref");
        if !refdir.is_dir() {
            continue;
        }
        let dists: Vec<String> = std::fs::read_dir(&idir)?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "ref")
            .collect();

        for dist in &dists {
            // Build the (size, ref, dist) series.
            let mut pairs: Vec<(u32, Rgb8Image, Rgb8Image)> = Vec::new();
            if downsample {
                let b = args.base;
                let rb = decode_image_to_rgb8(&refdir.join(format!("{b}x{b}.png")));
                let db = decode_image_to_rgb8(&idir.join(dist).join(format!("{b}x{b}.png")));
                if let (Ok(rb), Ok(db)) = (rb, db) {
                    for t in square_sizes() {
                        if t > b {
                            continue;
                        }
                        let rt = if t == b {
                            Rgb8Image {
                                pixels: rb.pixels.clone(),
                                width: b,
                                height: b,
                            }
                        } else {
                            area_resize_sq(&rb, t)
                        };
                        let dt = if t == b {
                            Rgb8Image {
                                pixels: db.pixels.clone(),
                                width: b,
                                height: b,
                            }
                        } else {
                            area_resize_sq(&db, t)
                        };
                        pairs.push((t, rt, dt));
                    }
                }
            } else {
                let mut sizes: Vec<(u32, u32)> = std::fs::read_dir(&refdir)?
                    .filter_map(|e| e.ok())
                    .filter_map(|e| {
                        let n = e.file_name().to_string_lossy().replace(".png", "");
                        let (w, h) = n.split_once('x')?;
                        Some((w.parse().ok()?, h.parse().ok()?))
                    })
                    .collect();
                sizes.sort();
                for (w, h) in sizes {
                    let r = decode_image_to_rgb8(&refdir.join(format!("{w}x{h}.png")));
                    let d = decode_image_to_rgb8(&idir.join(dist).join(format!("{w}x{h}.png")));
                    if let (Ok(r), Ok(d)) = (r, d) {
                        pairs.push((w.min(h), r, d));
                    }
                }
            }

            for &pad in &pads {
                for (mname, scorer) in &metrics {
                    let key = (*mname, pad.name());
                    let mut series: Vec<(u32, f64)> = Vec::new();
                    for (mind, r, d) in &pairs {
                        let (pr, pd) = (apply_pad(r, pad), apply_pad(d, pad));
                        match scorer(&pr, &pd) {
                            Ok(s) => {
                                series.push((*mind, s));
                                let e = minpx.entry(key).or_insert(u32::MAX);
                                *e = (*e).min(*mind);
                            }
                            Err(_) => *errs.entry(key).or_insert(0) += 1,
                        }
                    }
                    let inv: Vec<(u32, f64)> = series
                        .iter()
                        .cloned()
                        .filter(|(m, _)| *m >= args.floor)
                        .collect();
                    if !inv.is_empty() {
                        drift.entry(key).or_default().push(fluct(&inv));
                    }
                }
            }
        }
    }

    println!(
        "Size-invariance [{}] over {} ({} images). Drift = max-min score across sizes (min-dim ≥ {}px) per (image,distortion) series.\n",
        if downsample {
            "downsample-pair (distortion FIXED — drift = metric artifact)"
        } else {
            "encode-per-size (drift conflates real quality change)"
        },
        args.corpus.display(),
        images.len(),
        args.floor
    );
    println!(
        "{:<12} {:<8} {:>7} {:>9} {:>9} {:>9} {:>7}",
        "metric", "pad", "errors", "median", "p90", "worst", "min_px"
    );
    for (mname, _) in &metrics {
        for pad in &pads {
            let key = (*mname, pad.name());
            let mut ds = drift.get(&key).cloned().unwrap_or_default();
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = if ds.is_empty() { 0.0 } else { ds[ds.len() / 2] };
            let p90 = if ds.is_empty() {
                0.0
            } else {
                ds[(ds.len() * 9 / 10).min(ds.len() - 1)]
            };
            let worst = ds.last().copied().unwrap_or(0.0);
            let mp = minpx.get(&key).copied().unwrap_or(0);
            println!(
                "{:<12} {:<8} {:>7} {:>9.3} {:>9.3} {:>9.3} {:>7}",
                mname,
                pad.name(),
                errs.get(&key).copied().unwrap_or(0),
                med,
                p90,
                worst,
                if mp == u32::MAX { 0 } else { mp }
            );
        }
        println!();
    }
    Ok(())
}
