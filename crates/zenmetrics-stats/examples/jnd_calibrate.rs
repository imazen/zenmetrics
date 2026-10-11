//! Calibrate a metric's score to a JND target, leaving one source out.
//!
//! A thin driver over `zenmetrics_stats::jnd` and `zenmetrics_stats::resample`:
//! no statistic is computed here that the crate does not own. For each declared
//! map family it fits leave-one-source-out, reports out-of-fold RMSE in JND, τ̂
//! (moment and maximum likelihood) and P.1401 rmse* against each stimulus' own
//! interval (d = 0, out of fold), and source-clustered bootstrap intervals with
//! the maps refitted inside every draw (copy-aware folds from `Draw::expand`).
//!
//! Input: a TSV with a header row and `#` comment lines, and columns `model`,
//! `source`, `score`, `jnd`, `jnd_se`, `ci95_lo`, `ci95_hi` (others ignored).
//!
//! ```text
//! cargo run --release -p zenmetrics-stats --example jnd_calibrate -- \
//!     --input scores.tsv --model A [--range-max 4] [--keep-anchors] \
//!     [--resamples 2000] [--seed 20261011]
//! ```
//!
//! Anchors (rows with `jnd == 0` and `jnd_se == 0`: the scale's fixed zero,
//! not a measurement) are excluded unless `--keep-anchors`. `--range-max J`
//! keeps rows with `jnd <= J`. Output is one JSON object per family on stdout.
//! Without finite `jnd_se` or `ci95_*` columns, τ̂ or rmse* are `null` while
//! the out-of-fold RMSE is still reported.

use std::collections::HashMap;
use zenmetrics_stats::jnd::{
    MapFamily, Shape, excess_error_sd, fit_map, out_of_fold, rmse, rmse_star,
};
use zenmetrics_stats::resample::{BootstrapConfig, Design, Interval, cluster_bootstrap};

/// Student t 97.5th percentiles for 1..=30 degrees of freedom (standard
/// tables), used only to widen the bootstrap SE into the small-G interval
/// `estimate ± t(G−1) · se · √(G/(G−1))` documented in `resample`.
const T975: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];

struct Rows {
    source: Vec<u32>,
    names: Vec<String>,
    score: Vec<f64>,
    jnd: Vec<f64>,
    se: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].clone())
}

fn load(path: &str, model: &str, range_max: f64, keep_anchors: bool) -> (Rows, usize) {
    let text = std::fs::read_to_string(path).expect("read input");
    let mut lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = |n: &str| {
        header
            .iter()
            .position(|h| *h == n)
            .unwrap_or_else(|| panic!("missing column {n}"))
    };
    let (cm, cs, csc, cj, cse, clo, chi) = (
        col("model"),
        col("source"),
        col("score"),
        col("jnd"),
        col("jnd_se"),
        col("ci95_lo"),
        col("ci95_hi"),
    );
    let mut ids: HashMap<String, u32> = HashMap::new();
    let mut r = Rows {
        source: vec![],
        names: vec![],
        score: vec![],
        jnd: vec![],
        se: vec![],
        lo: vec![],
        hi: vec![],
    };
    let mut anchors = 0;
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        if f[cm] != model {
            continue;
        }
        let num = |i: usize| -> f64 {
            f[i].parse()
                .unwrap_or_else(|_| panic!("bad number {:?}", f[i]))
        };
        let (j, se) = (num(cj), num(cse));
        if j == 0.0 && se == 0.0 && !keep_anchors {
            anchors += 1;
            continue;
        }
        if j > range_max {
            continue;
        }
        let next = ids.len() as u32;
        let id = *ids.entry(f[cs].to_string()).or_insert(next);
        if id == next {
            r.names.push(f[cs].to_string());
        }
        r.source.push(id);
        r.score.push(num(csc));
        r.jnd.push(j);
        r.se.push(se);
        r.lo.push(num(clo));
        r.hi.push(num(chi));
    }
    (r, anchors)
}

/// (out-of-fold RMSE, τ̂ ML, τ̂ moment, rmse*) for one row selection. `None`
/// only when the out-of-fold fit fails; a measure the inputs cannot support
/// (no per-stimulus SE or interval) is NaN on its own, so the others survive.
fn metrics(
    family: MapFamily,
    x: &[f64],
    y: &[f64],
    se: &[f64],
    lo: &[f64],
    hi: &[f64],
    g: &[u32],
) -> Option<[f64; 4]> {
    let oof = out_of_fold(family, x, y, g).ok()?;
    let (ml, moment) =
        excess_error_sd(&oof, y, se).map_or((f64::NAN, f64::NAN), |e| (e.ml, e.moment));
    Some([
        rmse(&oof, y).unwrap_or(f64::NAN),
        ml,
        moment,
        rmse_star(&oof, lo, hi, 0).unwrap_or(f64::NAN),
    ])
}

/// A JSON number with six decimals, or `null` when it is not finite (JSON has
/// no NaN or infinity).
fn num(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "null".into()
    }
}

fn interval_json(name: &str, iv: &Interval, groups: usize) -> String {
    let t = T975[(groups - 1).clamp(1, 30) - 1];
    let half = t * iv.se * (groups as f64 / (groups as f64 - 1.0)).sqrt();
    format!(
        "\"{name}\":{{\"estimate\":{},\"percentile95\":[{},{}],\"student95\":[{},{}],\"se\":{},\"n_finite\":{}}}",
        num(iv.estimate),
        num(iv.lo),
        num(iv.hi),
        num(iv.estimate - half),
        num(iv.estimate + half),
        num(iv.se),
        iv.n_finite
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = arg(&args, "--input").expect("--input");
    let model = arg(&args, "--model").expect("--model");
    let range_max: f64 =
        arg(&args, "--range-max").map_or(f64::INFINITY, |v| v.parse().expect("range"));
    let keep_anchors = args.iter().any(|a| a == "--keep-anchors");
    let resamples: usize =
        arg(&args, "--resamples").map_or(2000, |v| v.parse().expect("resamples"));
    let seed: u64 = arg(&args, "--seed").map_or(20261011, |v| v.parse().expect("seed"));
    let (r, anchors) = load(&input, &model, range_max, keep_anchors);
    let groups = r.names.len();
    assert!(groups >= 2, "need at least two sources, got {groups}");
    let cfg = BootstrapConfig {
        resamples,
        seed,
        level: 0.95,
    };
    let families = [
        ("KneePower", MapFamily::KneePower(Shape::Quality)),
        ("Logistic4", MapFamily::Logistic4),
        ("Linear", MapFamily::Linear),
        ("Isotonic", MapFamily::Isotonic(Shape::Quality)),
    ];
    for (name, fam) in families {
        let full = fit_map(fam, &r.score, &r.jnd)
            .map(|m| m.params().to_vec())
            .unwrap_or_default();
        let point = metrics(fam, &r.score, &r.jnd, &r.se, &r.lo, &r.hi, &r.source);
        let ivs = cluster_bootstrap(r.score.len(), &Design::OneWay(&r.source), &cfg, |d| {
            let e = d.expand();
            let pick = |v: &[f64]| -> Vec<f64> { e.rows.iter().map(|&i| v[i]).collect() };
            let g = e.groups_u32(0);
            metrics(
                fam,
                &pick(&r.score),
                &pick(&r.jnd),
                &pick(&r.se),
                &pick(&r.lo),
                &pick(&r.hi),
                &g,
            )
            .map_or(vec![f64::NAN; 4], |m| m.to_vec())
        })
        .expect("bootstrap");
        let p = point.map_or("null".to_string(), |m| {
            format!("[{},{},{},{}]", num(m[0]), num(m[1]), num(m[2]), num(m[3]))
        });
        println!(
            "{{\"model\":\"{model}\",\"family\":\"{name}\",\"rows\":{},\"sources\":{:?},\"anchors_excluded\":{anchors},\
\"range_max\":{},\"full_fit_params\":{:?},\"oof_rmse_tau_ml_tau_moment_rmse_star\":{p},{},{},{},{},\"resamples\":{resamples},\"seed\":{seed}}}",
            r.score.len(),
            r.names,
            if range_max.is_finite() {
                format!("{range_max}")
            } else {
                "null".into()
            },
            full,
            interval_json("oof_rmse", &ivs[0], groups),
            interval_json("tau_ml", &ivs[1], groups),
            interval_json("tau_moment", &ivs[2], groups),
            interval_json("rmse_star", &ivs[3], groups),
        );
    }
}
