//! Validate `hdrvdp3` against the published AIC-2026 `HDR_VDP_3` column.
//!
//! The reference scores were produced by the `jpeg-ai-qaf` harness
//! (`metrics.py` `HDR_VDP.calc`), which feeds *whatever the decoder
//! produced* — for this corpus, 8-bit sRGB codes normalised to [0,1] —
//! through the HDR pipeline regardless of actual content:
//!
//! ```text
//! L = min(pq2lin(code) + L_refl, 10000 + L_refl)   cd/m²
//! L_refl = 0.005 · 100 / π                        (0.5 % ambient, 100 lx)
//! encoding = 'rgb-bt.709', task = 'quality'
//! ppd = pix_per_deg(64.5 in, [3840, 2160], 1.32528 m) ≈ 64.063
//! score = Q_JOD
//! ```
//!
//! Run with:
//! ```text
//! cargo run --release -p hdrvdp --example aic4_vdp3 -- /home/lilith/tmp/aic2026
//! ```

use hdrvdp::v3::{
    Emission, InputEncoding, Options, Params, Surround, Task, ViewingConditions, hdrvdp3,
    ingress::pq_to_linear,
};
use std::fs;
use std::path::{Path, PathBuf};

const L_REFL: f64 = 0.005 * 100.0 / std::f64::consts::PI;
const L_MAX: f64 = 10000.0;
/// `hdrvdp_pix_per_deg(64.5, [3840, 2160], 1.32528)`.
const PPD: f64 = 64.06340072695303;

/// `hdrvdp_pix_per_deg(diag_in, [w, h], dist_m)` — the upstream helper.
fn pix_per_deg(diag_in: f64, w: f64, h: f64, dist_m: f64) -> f64 {
    let ar = w / h;
    let height_mm = ((diag_in * 25.4).powi(2) / (1.0 + ar * ar)).sqrt();
    let height_deg =
        2.0 * (0.5 * height_mm / (dist_m * 1000.0)).atan() * 180.0 / std::f64::consts::PI;
    h / height_deg
}

fn load_srgb01(path: &Path) -> Option<(Vec<f64>, usize, usize)> {
    let data = fs::read(path).ok()?;
    let out = zenpng::decode(
        &data,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .ok()?;
    let img = out.pixels.try_as_imgref::<rgb::Rgb<u8>>()?;
    let (w, h) = (img.width(), img.height());
    let (buf, stride) = (img.buf(), img.stride());
    let mut px = Vec::with_capacity(w * h * 3);
    for row in buf.chunks(stride).take(h) {
        for p in row.iter().take(w) {
            px.extend_from_slice(&[
                f64::from(p.r) / 255.0,
                f64::from(p.g) / 255.0,
                f64::from(p.b) / 255.0,
            ]);
        }
    }
    Some((px, w, h))
}

fn main() {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/home/lilith/tmp/aic2026".into()),
    );
    let pairs_path = root.join("full/pairs.tsv");
    let table_path = root.join("metrics_fullres.tab");

    // Published table → map (dist_name) → HDR_VDP_3.
    let table = fs::read_to_string(&table_path).expect("metrics_fullres.tab");
    let mut lines = table.lines();
    let cols: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let i_dist = cols
        .iter()
        .position(|c| c.trim_matches('"') == "distorted")
        .unwrap_or(0);
    let col_name = if std::env::var("AIC4_VDP2").is_ok() {
        "HDR_VDP_2"
    } else {
        "HDR_VDP_3"
    };
    let i_col = cols
        .iter()
        .position(|c| c.trim_matches('"') == col_name)
        .unwrap_or_else(|| panic!("{col_name} column"));
    let mut want: std::collections::HashMap<String, f64> = Default::default();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        if f.len() <= i_col {
            continue;
        }
        want.insert(
            f[i_dist].trim_matches('"').to_string(),
            f[i_col].parse().unwrap_or(f64::NAN),
        );
    }

    let limit: usize = std::env::var("AIC4_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);

    let mut deltas = Vec::new();
    let mut skipped = 0usize;
    for l in fs::read_to_string(&pairs_path)
        .expect("pairs.tsv")
        .lines()
        .skip(1)
    {
        let mut f = l.split('\t');
        let (Some(rp), Some(dp)) = (f.next(), f.next()) else {
            continue;
        };
        let (r, t) = (Path::new(rp), Path::new(dp));
        if !r.exists() || !t.exists() {
            skipped += 1;
            continue;
        }
        let name = t.file_name().unwrap().to_string_lossy().to_string();
        let Some(&published) = want.get(&name) else {
            skipped += 1;
            continue;
        };
        let (rv, rw, rh) = load_srgb01(r).expect("ref decode");
        let (tv, tw, th) = load_srgb01(t).expect("dist decode");
        assert_eq!((rw, rh), (tw, th), "{name}: shape");

        // PPD env override sweeps the hypothesis that the published run
        // derived ppd from the image resolution rather than the 4K
        // display geometry (`image` → pix_per_deg on (w,h)).
        let ppd = match std::env::var("AIC4_PPD").as_deref() {
            Ok("image") => pix_per_deg(64.5, rw as f64, rh as f64, 1.32528),
            Ok(v) => v.parse().unwrap(),
            _ => PPD,
        };
        let task = match std::env::var("AIC4_TASK").as_deref() {
            Ok("flicker") => Task::Flicker,
            Ok("side-by-side" | "sbs") => Task::SideBySide,
            _ => Task::Quality,
        };
        let mut opts = Options::reference(task);
        // AIC4_BSC=<delta> probes sensitivity-scale hypotheses, e.g.
        // -0.203943775672 cancels the quality task's base correction.
        if let Ok(v) = std::env::var("AIC4_BSC") {
            opts.sensitivity_correction_delta = v.parse().unwrap();
        }
        let par = Params::new(
            task,
            ViewingConditions::new(ppd, Surround::None, 24),
            InputEncoding::RgbBt709,
            Emission::default_for(InputEncoding::RgbBt709),
            opts,
        )
        .unwrap();

        // AIC4_EOTF=none probes the hypothesis that the published run fed
        // raw [0,1] codes as absolute luminance (no PQ decode).
        let ingress = |v: &[f64]| -> Vec<f64> {
            // AIC4_NOREFL probes the hypothesis that the published run left
            // E_ambient at its default (no reflectance term).
            let refl = if std::env::var("AIC4_NOREFL").is_ok() {
                0.0
            } else {
                L_REFL
            };
            if std::env::var("AIC4_EOTF").as_deref() == Ok("none") {
                v.iter().map(|&l| (l + refl).min(L_MAX + refl)).collect()
            } else {
                pq_to_linear(v)
                    .iter()
                    .map(|&l| (l + refl).min(L_MAX + refl))
                    .collect()
            }
        };
        let (a, b) = if std::env::var("AIC4_SWAP").is_ok() {
            (rv.clone(), tv.clone())
        } else {
            (tv.clone(), rv.clone())
        };
        let ours = if std::env::var("AIC4_VDP2").is_ok() {
            // Protocol probe: run the *validated* v2 scorer on the same
            // ingress to identify the published run's ppd independently of
            // the v3 implementation.
            let p2 = hdrvdp::Params::new(ppd);
            hdrvdp::score(
                &ingress(&a),
                &ingress(&b),
                rw,
                rh,
                hdrvdp::ColorEncoding::RgbBt709,
                &p2,
            )
            .expect("hdrvdp2")
        } else {
            hdrvdp3(&ingress(&a), &ingress(&b), rw, rh, &par)
                .expect("hdrvdp3")
                .q_jod
        };
        if deltas.len() >= limit {
            break;
        }
        let d = ours - published;
        deltas.push(d.abs());
        println!(
            "{name}\tours {:.6}\tpublished {:.6}\tΔ{d:.2e}",
            ours, published
        );
    }
    deltas.sort_by(f64::total_cmp);
    let n = deltas.len();
    if n == 0 {
        eprintln!("no pairs decoded (skipped {skipped})");
        std::process::exit(2);
    }
    println!(
        "n={n} skipped={skipped} median|Δ|={:.3e} p95={:.3e} max|Δ|={:.3e}",
        deltas[n / 2],
        deltas[n * 95 / 100],
        deltas[n - 1]
    );
}
