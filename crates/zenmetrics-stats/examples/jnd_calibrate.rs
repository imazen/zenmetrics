//! Calibrate a metric's score to a JND target, leaving one source out.
//!
//! A thin driver over `zenmetrics_stats::jnd` and `zenmetrics_stats::resample`:
//! no statistic is computed here that the crate does not own. The primary
//! family is the knee power with its knee fixed at `--knee` (default 100, a
//! quality score's pixel-identity value); the fitted-knee knee power, the
//! 4-parameter logistic, the linear map and the isotonic map are reported
//! beside it. For each family it fits leave-one-source-out, reports out-of-fold RMSE in JND, τ̂
//! (moment and maximum likelihood) and P.1401 rmse* against each stimulus' own
//! interval (d = 0, out of fold), and source-clustered bootstrap intervals with
//! the maps refitted inside every draw (copy-aware folds from `Draw::expand`).
//!
//! Input: a TSV with a header row and `#` comment lines, and columns `model`,
//! `source`, `score`, `jnd`, `jnd_se`, `ci95_lo`, `ci95_hi` (others ignored).
//!
//! ```text
//! cargo run --release -p zenmetrics-stats --example jnd_calibrate -- \
//!     --input scores.tsv --model A [--knee 100] [--range-max 4] [--keep-anchors] \
//!     [--resamples 2000] [--seed 20261011]
//! ```
//!
//! After the families, one `support` object applies the join-point rule to the
//! primary map: its out-of-fold predictions are binned (`--bin-width`, default
//! 0.5 JND, up to `--support-max`, default 4). A bin is evaluable when it holds
//! at least `--min-stimuli` rows (20) from at least `min(--min-sources,
//! sources)` sources (10), and supported when it is evaluable with out-of-fold
//! RMSE at most `--max-rmse` (0.75 JND). Bins below the lowest evaluable bin
//! are skipped (the knee anchors the zero there); from it upward, `j_star` is
//! the upper edge of the last bin in the unbroken run of supported bins, or
//! `null` when the lowest evaluable bin is not supported or none is evaluable.
//! Binning on the prediction, not the label, keeps the held-out label out of
//! the selection; read `j_star` from a run without `--range-max`.
//!
//! Anchors (rows with `jnd == 0` and `jnd_se == 0`: the scale's fixed zero,
//! not a measurement) are excluded unless `--keep-anchors`. `--range-max J`
//! keeps rows with `jnd <= J`. Output is one JSON object per family on stdout.
//! Without finite `jnd_se` or `ci95_*` columns, τ̂ or rmse* are `null` while
//! the out-of-fold RMSE is still reported.

use std::collections::HashMap;
use zenmetrics_stats::jnd::{
    Knee, MapFamily, Shape, excess_error_sd, fit_map, out_of_fold, rmse, rmse_star,
};
use zenmetrics_stats::resample::{BootstrapConfig, Design, Interval, cluster_bootstrap};

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
    let (slo, shi) = iv.student(groups, 0.95).unwrap_or((f64::NAN, f64::NAN));
    format!(
        "\"{name}\":{{\"estimate\":{},\"percentile95\":[{},{}],\"student95\":[{},{}],\"se\":{},\"n_finite\":{}}}",
        num(iv.estimate),
        num(iv.lo),
        num(iv.hi),
        num(slo),
        num(shi),
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
    let knee = arg(&args, "--knee")
        .map_or(Some(100.0), |v| v.parse().ok())
        .and_then(Knee::new)
        .expect("--knee must be a finite number");
    let resamples: usize =
        arg(&args, "--resamples").map_or(2000, |v| v.parse().expect("resamples"));
    let seed: u64 = arg(&args, "--seed").map_or(20261011, |v| v.parse().expect("seed"));
    let num_arg = |k: &str, d: f64| -> f64 { arg(&args, k).map_or(d, |v| v.parse().expect(k)) };
    let bin_width = num_arg("--bin-width", 0.5);
    let support_max = num_arg("--support-max", 4.0);
    let min_stimuli = num_arg("--min-stimuli", 20.0) as usize;
    let min_sources = num_arg("--min-sources", 10.0) as usize;
    let max_rmse = num_arg("--max-rmse", 0.75);
    assert!(
        bin_width > 0.0 && support_max > 0.0,
        "bin width and support max must be positive"
    );
    let (r, anchors) = load(&input, &model, range_max, keep_anchors);
    let groups = r.names.len();
    assert!(groups >= 2, "need at least two sources, got {groups}");
    let cfg = BootstrapConfig {
        resamples,
        seed,
        level: 0.95,
    };
    let families = [
        (
            "KneePowerFixed",
            "primary",
            MapFamily::KneePowerFixed(Shape::Quality, knee),
        ),
        (
            "KneePower",
            "report-only",
            MapFamily::KneePower(Shape::Quality),
        ),
        ("Logistic4", "report-only", MapFamily::Logistic4),
        ("Linear", "report-only", MapFamily::Linear),
        (
            "Isotonic",
            "report-only",
            MapFamily::Isotonic(Shape::Quality),
        ),
    ];
    for (name, role, fam) in families {
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
            "{{\"model\":\"{model}\",\"family\":\"{name}\",\"role\":\"{role}\",\"rows\":{},\"sources\":{:?},\"anchors_excluded\":{anchors},\
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
    let fam = families[0].2;
    let Ok(oof) = out_of_fold(fam, &r.score, &r.jnd, &r.source) else {
        println!("{{\"model\":\"{model}\",\"support\":null,\"j_star\":null}}");
        return;
    };
    let need_sources = min_sources.min(groups);
    let bins = (support_max / bin_width).ceil() as usize;
    let mut cells = Vec::new();
    let mut j_star: Option<f64> = None;
    let (mut started, mut stopped) = (false, false);
    for b in 0..bins {
        let (lo, hi) = (
            b as f64 * bin_width,
            ((b + 1) as f64 * bin_width).min(support_max),
        );
        let rows: Vec<usize> = (0..oof.len())
            .filter(|&i| (oof[i] > lo || (b == 0 && oof[i] >= 0.0)) && oof[i] <= hi)
            .collect();
        let mut srcs: Vec<u32> = rows.iter().map(|&i| r.source[i]).collect();
        srcs.sort_unstable();
        srcs.dedup();
        let pick = |v: &[f64]| -> Vec<f64> { rows.iter().map(|&i| v[i]).collect() };
        let e = rmse(&pick(&oof), &pick(&r.jnd)).unwrap_or(f64::NAN);
        let evaluable = rows.len() >= min_stimuli && srcs.len() >= need_sources;
        let ok = evaluable && e <= max_rmse;
        started |= evaluable;
        if started && !stopped {
            if ok {
                j_star = Some(hi);
            } else {
                stopped = true;
            }
        }
        cells.push(format!(
            "{{\"lo\":{lo},\"hi\":{hi},\"n\":{},\"sources\":{},\"oof_rmse\":{},\"evaluable\":{evaluable},\"supported\":{ok}}}",
            rows.len(),
            srcs.len(),
            num(e)
        ));
    }
    println!(
        "{{\"model\":\"{model}\",\"support_family\":\"{}\",\"rule\":{{\"bin_width\":{bin_width},\"support_max\":{support_max},\"min_stimuli\":{min_stimuli},\"min_sources\":{need_sources},\"max_rmse\":{max_rmse},\"binned_on\":\"out_of_fold_prediction\"}},\"support\":[{}],\"j_star\":{}}}",
        families[0].0,
        cells.join(","),
        j_star.map_or("null".to_string(), |v| format!("{v}"))
    );
}
