//! Batch VMAF feature dump for the AIC2026 reproduction lane: emits the
//! columns the `batch`/`score-pairs` commands do not expose — `adm2` and
//! `vif_scales[0..3]` (whose sum is the published `VIF_vmaf` column) —
//! plus the `VMAF`/`VMAF-neg` scores for cross-checking the CLI's
//! `vmaf-studio601`/`vmaf-neg-studio601` route on the same ingress.
//!
//! Ingress: studio-swing BT.601 luma `round(16 + (65.481R + 128.553G +
//! 24.966B)/255)`, even crop (one column/row dropped on odd sizes), neutral
//! 128 chroma — identical to `metrics::vmaf::score(.., Ingress::Studio601)`
//! (the v0 features are luma-only, so chroma content does not matter).
//!
//! Usage: aic_vmaf_features <pairs.tsv> <out.tsv> [jobs]
//! pairs.tsv needs `ref_path` and `dist_path` columns; `distorted` is
//! passed through when present for joining.

use rayon::prelude::*;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use vmaf::{VmafV0Scorer, VmafV0Variant, Yuv420Frame};
use zenmetrics_cli::decode::decode_image_to_rgb8;

fn studio601_y(img: &zenmetrics_cli::decode::Rgb8Image) -> (Vec<u16>, usize, usize) {
    let (w, h) = (img.width as usize & !1, img.height as usize & !1);
    let stride = img.width as usize;
    let mut y = Vec::with_capacity(w * h);
    for row in 0..h {
        for x in 0..w {
            let p = (row * stride + x) * 3;
            let (r, g, b) = (
                f64::from(img.pixels[p]),
                f64::from(img.pixels[p + 1]),
                f64::from(img.pixels[p + 2]),
            );
            y.push(
                (16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u16,
            );
        }
    }
    (y, w, h)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <pairs.tsv> <out.tsv> [jobs]", args[0]);
        std::process::exit(2);
    }
    let jobs: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
    let text = fs::read_to_string(&args[1])?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("empty pairs tsv")?.split('\t').collect();
    let (ri, di) = (
        header
            .iter()
            .position(|&c| c == "ref_path")
            .ok_or("no ref_path")?,
        header
            .iter()
            .position(|&c| c == "dist_path")
            .ok_or("no dist_path")?,
    );
    let ni = header.iter().position(|&c| c == "distorted");
    let rows: Vec<(String, String, String)> = lines
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            (
                c[ri].to_string(),
                c[di].to_string(),
                ni.map(|i| c[i].to_string()).unwrap_or_default(),
            )
        })
        .collect();

    let pool = rayon::ThreadPoolBuilder::new().num_threads(jobs).build()?;
    let out: Vec<String> = pool.install(|| {
        rows.par_iter()
            .map(|(rp, dp, name)| {
                let r = decode_image_to_rgb8(Path::new(rp))
                    .unwrap_or_else(|e| panic!("decode {rp}: {e}"));
                let d = decode_image_to_rgb8(Path::new(dp))
                    .unwrap_or_else(|e| panic!("decode {dp}: {e}"));
                let (ry, w, h) = studio601_y(&r);
                let (dy, dw, dh) = studio601_y(&d);
                assert_eq!((w, h), (dw, dh), "size mismatch {rp} vs {dp}");
                let n = (w / 2) * (h / 2);
                let (ur, vr) = (vec![128u16; n], vec![128u16; n]);
                let (ud, vd) = (vec![128u16; n], vec![128u16; n]);
                let fr = [Yuv420Frame {
                    y: &ry,
                    u: &ur,
                    v: &vr,
                }];
                let fd = [Yuv420Frame {
                    y: &dy,
                    u: &ud,
                    v: &vd,
                }];
                let std_scorer = VmafV0Scorer::new(w, h, 8, VmafV0Variant::Standard)
                    .unwrap_or_else(|e| panic!("scorer {dp}: {e}"));
                let neg_scorer = VmafV0Scorer::new(w, h, 8, VmafV0Variant::StandardNeg)
                    .unwrap_or_else(|e| panic!("scorer {dp}: {e}"));
                let s = std_scorer
                    .score(&fr, &fd)
                    .unwrap_or_else(|e| panic!("{dp}: {e}"));
                let ng = neg_scorer
                    .score(&fr, &fd)
                    .unwrap_or_else(|e| panic!("{dp}: {e}"));
                let f = &s[0].features;
                let vif_vmaf: f64 = f.vif_scales.iter().sum();
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    name,
                    rp,
                    dp,
                    s[0].score,
                    ng[0].score,
                    f.adm2,
                    vif_vmaf,
                    f.vif_scales[0],
                    f.vif_scales[1],
                    f.vif_scales[2],
                    f.vif_scales[3]
                )
            })
            .collect()
    });

    let mut w = BufWriter::new(fs::File::create(&args[2])?);
    writeln!(
        w,
        "distorted\tref_path\tdist_path\tvmaf_studio601\tvmaf_neg_studio601\tadm2\tvif_vmaf\tvif_s0\tvif_s1\tvif_s2\tvif_s3"
    )?;
    for line in out {
        writeln!(w, "{line}")?;
    }
    Ok(())
}
