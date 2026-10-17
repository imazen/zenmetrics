//! Parity of every zenmetrics-nn model against its upstream package's goldens.
//!
//! Inputs (set by `just nn-parity`, never defaulted here):
//! - `ZENMETRICS_NN_MODEL_DIR`: the `models/*/export.py` output directory
//!   (`<model>_goldens.tsv` next to each model's files).
//! - `ZENMETRICS_NN_MODELS`: comma-separated models to check:
//!   `lpips_alex`, `lpips_vgg`, `lpips_squeeze`, `stlpips_alex`, `stlpips_vgg`,
//!   `dists`, `deepdc`, `ckdn`, `afine`, `dreamsim`, `ahiq`, `arniqa`, `maniqa`,
//!   `liqe`, `musiq`, `hyperiqa`, `paq2piq`, `pieapp`, `wadiqam_fr_live`,
//!   `wadiqam_fr_tid`, `wadiqam_nr_live`, `wadiqam_nr_tid`, and `dmm` (its own
//!   gate — see `dmm_netlib_goldens`).
//! - `ZENMETRICS_NN_LADDER`: the ladder PNGs the goldens were computed on.
//! - `ZENMETRICS_NN_ENGINE`: `rten` or `ort-cuda` (needs the `ort` feature
//!   and `ORT_DYLIB_PATH`).
//!
//! PieAPP goldens use the official CUDA model with TF32 disabled; the CPU
//! control differs by at most 4.8e-7 native units. Other models use CPU fp32.

use std::path::{Path, PathBuf};

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Evaluation, Image, Metric, PairedNoReference};
use zenmetrics_nn::{
    Afine, Ahiq, Arniqa, Ckdn, DeepDc, Dists, Dmm, Dreamsim, Engine, Hyperiqa, Liqe, Lpips,
    LpipsNet, Maniqa, Musiq, Paq2piq, PieApp, WadiqamFr, WadiqamKind, WadiqamNr, WeightsConsent,
};

/// Tolerance in JND units (owner,
/// 2026-10-15: "should be in estimated jnd units, could be looser"): each
/// model may differ from upstream by 1/1000 of its estimated just-noticeable
/// difference. The engines differ from PyTorch in summation order, and
/// MANIQA also in its OpenCV-convention resize (a few ±1 u8 pixels);
/// measured worst on the 64²–2048² ladder is under 2e-4 JND across models.
const TOL_JND: f64 = 1e-3;

/// One estimated JND per score column, from `models/jnd_scale.tsv`
/// (KonJND-1k, pooled median at each image's JND threshold). A column
/// without an entry fails: every model needs its JND measured.
fn jnd(col: &str) -> f64 {
    const TABLE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/jnd_scale.tsv"));
    TABLE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("column\t"))
        .find_map(|l| {
            let mut f = l.split('\t');
            (f.next() == Some(col)).then(|| f.next().unwrap().parse::<f64>().unwrap())
        })
        .unwrap_or_else(|| panic!("models/jnd_scale.tsv has no JND for {col}"))
}

/// Absolute tolerance for `col`: `TOL_JND` JNDs.
fn tol(col: &str) -> f64 {
    TOL_JND * jnd(col)
}

fn env_dir(var: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(var)
            .unwrap_or_else(|| panic!("{var} must be set (run via `just nn-parity`)")),
    )
}

fn engine() -> Engine {
    match std::env::var("ZENMETRICS_NN_ENGINE").as_deref() {
        Ok("rten") => Engine::Rten,
        #[cfg(feature = "ort")]
        Ok("ort-cuda") => Engine::OrtCuda,
        other => panic!("ZENMETRICS_NN_ENGINE must name a built engine, got {other:?}"),
    }
}

/// The models named in `ZENMETRICS_NN_MODELS`, loaded, with their columns.
fn models() -> Vec<(Box<dyn Metric>, &'static str)> {
    let dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let list = std::env::var("ZENMETRICS_NN_MODELS")
        .expect("ZENMETRICS_NN_MODELS must be set (run via `just nn-parity`)");
    list.split(',')
        .map(|name| -> (Box<dyn Metric>, &'static str) {
            let name = name.trim();
            if name == "dists" {
                return (
                    Box::new(Dists::from_dir(&dir, engine()).expect("load dists")),
                    "dists",
                );
            }
            if name == "deepdc" {
                return (
                    Box::new(DeepDc::from_dir(&dir, engine()).expect("load deepdc")),
                    "deepdc",
                );
            }
            if name == "ckdn" {
                let consent = WeightsConsent::new().accept(Ckdn::WEIGHTS);
                return (
                    Box::new(Ckdn::from_dir(&dir, engine(), &consent).expect("load ckdn")),
                    "ckdn",
                );
            }
            if name == "afine" {
                let consent = WeightsConsent::new().accept(Afine::WEIGHTS);
                return (
                    Box::new(Afine::from_dir(&dir, engine(), &consent).expect("load afine")),
                    "afine",
                );
            }
            if name == "dreamsim" {
                let consent = WeightsConsent::new().accept(Dreamsim::WEIGHTS);
                return (
                    Box::new(Dreamsim::from_dir(&dir, engine(), &consent).expect("load dreamsim")),
                    "dreamsim",
                );
            }
            if name == "arniqa" {
                return (
                    Box::new(PairedNoReference::new(
                        Arniqa::from_dir(&dir, engine()).expect("load arniqa"),
                    )),
                    "arniqa",
                );
            }
            if name == "maniqa" {
                return (
                    Box::new(PairedNoReference::new(
                        Maniqa::from_dir(&dir, engine()).expect("load maniqa"),
                    )),
                    "maniqa",
                );
            }
            if name == "liqe" {
                let consent = WeightsConsent::new().accept(Liqe::WEIGHTS);
                return (
                    Box::new(PairedNoReference::new(
                        Liqe::from_dir(&dir, engine(), &consent).expect("load liqe"),
                    )),
                    "liqe",
                );
            }
            if name == "musiq" {
                return (
                    Box::new(PairedNoReference::new(
                        Musiq::from_dir(&dir, engine()).expect("load musiq"),
                    )),
                    "musiq",
                );
            }
            if name == "hyperiqa" {
                let consent = WeightsConsent::new().accept(Hyperiqa::WEIGHTS);
                return (
                    Box::new(PairedNoReference::new(
                        Hyperiqa::from_dir(&dir, engine(), &consent).expect("load hyperiqa"),
                    )),
                    "hyperiqa",
                );
            }
            if name == "pieapp" {
                let consent = WeightsConsent::new().accept(PieApp::WEIGHTS);
                return (
                    Box::new(PieApp::from_dir(&dir, engine(), &consent).expect("load pieapp")),
                    "pieapp",
                );
            }
            if name == "paq2piq" {
                let consent = WeightsConsent::new().accept(Paq2piq::WEIGHTS);
                return (
                    Box::new(PairedNoReference::new(
                        Paq2piq::from_dir(&dir, engine(), &consent).expect("load paq2piq"),
                    )),
                    "paq2piq",
                );
            }
            if let Some((kind, col)) = match name {
                "wadiqam_fr_live" => Some((WadiqamKind::FrLive, "wadiqam_fr_live")),
                "wadiqam_fr_tid" => Some((WadiqamKind::FrTid, "wadiqam_fr_tid")),
                "wadiqam_nr_live" => Some((WadiqamKind::NrLive, "wadiqam_nr_live")),
                "wadiqam_nr_tid" => Some((WadiqamKind::NrTid, "wadiqam_nr_tid")),
                _ => None,
            } {
                let m: Box<dyn Metric> = if kind.is_fr() {
                    Box::new(WadiqamFr::from_dir(&dir, engine(), kind).expect("load wadiqam fr"))
                } else {
                    Box::new(PairedNoReference::new(
                        WadiqamNr::from_dir(&dir, engine(), kind).expect("load wadiqam nr"),
                    ))
                };
                return (m, col);
            }
            if name == "ahiq" {
                let consent = WeightsConsent::new().accept(Ahiq::WEIGHTS);
                return (
                    Box::new(Ahiq::from_dir(&dir, engine(), &consent).expect("load ahiq")),
                    "ahiq",
                );
            }
            if name == "dmm" {
                return (
                    Box::new(Dmm::from_dir(&dir, engine()).expect("load dmm")),
                    "dmm",
                );
            }
            let net = *LpipsNet::ALL
                .iter()
                .find(|n| n.column() == name)
                .unwrap_or_else(|| panic!("unknown model {name:?}"));
            (
                Box::new(Lpips::from_dir(&dir, net, engine()).expect("load lpips")),
                net.column(),
            )
        })
        .collect()
}

/// Ladder PNG lookup: `ZENMETRICS_NN_LADDER` first, then the sibling
/// `ladder-nonsquare` directory (non-square goldens live there; the
/// shared square ladder must stay square).
fn ladder_png(ladder: &Path, name: &str) -> PathBuf {
    let p = ladder.join(name);
    if p.exists() {
        return p;
    }
    ladder.with_file_name("ladder-nonsquare").join(name)
}

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

fn score(e: &dyn Evaluation, col: &str) -> f64 {
    e.scores().find(|(k, _)| k == col).expect("score column").1
}

fn image(px: &[u8], w: usize, h: usize) -> Image<'_> {
    Image::from(PixelSlice::new(px, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap())
}

/// Whether `m` is a paired no-reference metric — its outputs carry the
/// distorted score under `col` and the reference's under `col_ref`, and the
/// goldens row gives both.
fn is_nr(m: &dyn Metric, col: &str) -> bool {
    m.outputs().iter().any(|s| *s.key == *format!("{col}_ref"))
}

#[test]
fn matches_upstream() {
    let model_dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let ladder = env_dir("ZENMETRICS_NN_LADDER");
    for (mut m, col) in models() {
        // DMM is gated in `dmm_netlib_goldens` below: its head consumes
        // the SVD solver's basis draw in degenerate clusters, which is
        // implementation-defined, so the MKL-draw upstream goldens are
        // structurally unreachable from our netlib-sgesdd transliteration.
        if col == "dmm" {
            continue;
        }
        let path = model_dir.join(format!("{col}_goldens.tsv"));
        let goldens =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let nr = is_nr(&*m, col);
        let mut checked = 0;
        let mut worst = 0.0f64;
        for line in goldens.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            let (stem, size, want): (&str, &str, f64) = (f[0], f[1], f[2].parse().unwrap());
            let (rp, w, h) = load_rgb8(&ladder_png(&ladder, &format!("{stem}_{size}_ref.png")));
            let (dp, _, _) = load_rgb8(&ladder_png(&ladder, &format!("{stem}_{size}_dist.png")));
            let eval = m
                .evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                .expect("evaluate");
            let got = score(&*eval, col);
            let err = (got - want).abs();
            worst = worst.max(err);
            eprintln!("{col} {stem} {size}: got {got:.9} want {want:.9} |d| {err:.3e}");
            // A mismatch on this CUDA host needs a diagnostic rerun. Preserve
            // and gate the first result even if the diagnostic happens to pass.
            if engine() != Engine::Rten && (!err.is_finite() || err > tol(col)) {
                let diagnostic = m
                    .evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                    .expect("diagnostic CUDA evaluation after first-score mismatch");
                eprintln!(
                    "{col} {stem} {size}: first mismatch={got:.17e} diagnostic={:.17e}",
                    score(&*diagnostic, col)
                );
            }
            assert!(
                err <= tol(col),
                "{col} {stem} {size}: |{got} - {want}| = {err:e} > {:e} ({TOL_JND} JND)",
                tol(col)
            );
            let repeat = if col == "pieapp" && engine() != Engine::Rten {
                let repeated = m
                    .evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                    .expect("repeat CUDA evaluation without replacing the first result");
                let value = score(&*repeated, col);
                eprintln!("{col} {stem} {size}: first={got:.17e} repeated={value:.17e}");
                Some(value)
            } else {
                None
            };
            if let Some(repeat) = repeat {
                assert_eq!(
                    got.to_bits(),
                    repeat.to_bits(),
                    "{col} {stem} {size}: CUDA repeat changed"
                );
            }
            if nr {
                // The reference column: golden field 3 under `<col>_ref`.
                let want_ref: f64 = f[3].parse().unwrap();
                let got_ref = score(&*eval, &format!("{col}_ref"));
                let err_ref = (got_ref - want_ref).abs();
                worst = worst.max(err_ref);
                eprintln!(
                    "{col}_ref {stem} {size}: got {got_ref:.9} want {want_ref:.9} |d| {err_ref:.3e}"
                );
                assert!(
                    err_ref <= tol(col),
                    "{col}_ref {stem} {size}: |{got_ref} - {want_ref}| = {err_ref:e} > {:e}",
                    tol(col)
                );
            }
            checked += 1;
        }
        assert!(checked > 0, "{} had no rows", path.display());
        eprintln!(
            "{col}: {checked} pairs, worst |d| {worst:.3e} = {:.2e} JND",
            worst / jnd(col)
        );
    }
}

/// Validate the entire reviewed panel before any inference starts.
fn validate_dmm_panel<'a>(
    netlib: &'a str,
    mkl: &std::collections::HashMap<(String, String), f64>,
) -> Vec<&'a str> {
    let mut lines = netlib
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    assert_eq!(lines.next(), Some("image\tsize\tdmm\ttol\tkind\tresidual"));
    let rows: Vec<_> = lines.collect();
    let mut seen = std::collections::HashSet::new();
    for &line in &rows {
        let f: Vec<_> = line.split('\t').collect();
        assert_eq!(f.len(), 6);
        let key = (f[0].to_string(), f[1].to_string());
        assert!(mkl.contains_key(&key), "extra netlib row {key:?}");
        assert!(seen.insert(key), "duplicate netlib row");
        let score: f64 = f[2].parse().unwrap();
        let tol: f64 = f[3].parse().unwrap();
        let residual: f64 = f[5].parse().unwrap();
        assert!(score.is_finite() && tol.is_finite() && tol >= 0.0);
        assert!(residual.is_finite() && residual >= 0.0);
        match f[4] {
            "jnd" => assert!(tol <= 1e-3 * jnd("dmm")),
            "pinned" => assert!(tol <= 1.05 * residual + 1e-12),
            other => panic!("unknown DMM golden kind {other}"),
        }
    }
    assert!(!rows.is_empty());
    assert_eq!(
        seen.len(),
        mkl.len(),
        "missing netlib rows from dmm_goldens.tsv"
    );
    rows
}

#[test]
fn dmm_panel_rejects_coverage_and_floor_changes() {
    let mkl = std::collections::HashMap::from([(("sample".into(), "128".into()), 1.0)]);
    let header = "image\tsize\tdmm\ttol\tkind\tresidual\n";
    let row = "sample\t128\t1.0\t0.0001\tjnd\t0.00001\n";
    assert_eq!(validate_dmm_panel(&format!("{header}{row}"), &mkl).len(), 1);
    for bad in [
        String::new(),         // missing row
        format!("{row}{row}"), // duplicate row
        row.replace("sample", "extra"),
        row.replace("0.0001", "0.001"), // JND budget growth
        row.replace("jnd", "pinned"),   // residual budget growth
        row.replace("0.0001", "NaN"),
        row.replace("0.0001", "inf"),
        row.replace("0.00001", "NaN"),
        row.replace("jnd", "unknown"),
    ] {
        assert!(
            std::panic::catch_unwind(|| {
                validate_dmm_panel(&format!("{header}{bad}"), &mkl);
            })
            .is_err(),
            "accepted invalid panel: {bad}"
        );
    }
}

/// DMM's gate: `dmm_goldens_netlib_<engine>.tsv` — scipy `sgesdd` f32 head
/// scored on this engine's own taps (`models/dmm/gen_netlib_goldens.py` +
/// `examples/dmm_tap_dump.rs`). The score's `wt` term consumes the
/// orthonormal basis the SVD draws inside degenerate singular clusters —
/// implementation-defined per LAPACK build. The same-tap goldens separate
/// host-solver residuals from trunk differences. The transliteration drifts past 1e-3 JND on
/// deep-null-heavy rows, which the golden file pins at their measured
/// residual (`kind=pinned`, tol = residual·1.05) as a regression floor.
/// The MKL-draw `dmm_goldens.tsv` oracle is printed per row as a recorded
/// measurement, not gated. Selected by listing `dmm` in
/// `ZENMETRICS_NN_MODELS` (`just nn-parity-dmm`).
#[test]
fn dmm_netlib_goldens() {
    let list = std::env::var("ZENMETRICS_NN_MODELS")
        .expect("ZENMETRICS_NN_MODELS must be set (run via `just nn-parity-dmm`)");
    if !list.split(',').any(|s| s.trim() == "dmm") {
        eprintln!("dmm_netlib_goldens: skipped (dmm not selected in ZENMETRICS_NN_MODELS)");
        return;
    }
    let model_dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let ladder = env_dir("ZENMETRICS_NN_LADDER");
    let ladder_ns = PathBuf::from(format!("{}-nonsquare", ladder.display()));
    let mut m = Dmm::from_dir(&model_dir, engine()).expect("load dmm");
    // Regression floors are reviewed source artifacts, not mutable model-dir files.
    let netlib = match engine() {
        Engine::Rten => include_str!("../models/dmm/dmm_goldens_netlib_rten.tsv"),
        #[cfg(feature = "ort")]
        Engine::OrtCuda => include_str!("../models/dmm/dmm_goldens_netlib_ort_cuda.tsv"),
        other => panic!("no reviewed DMM goldens for {other:?}"),
    };
    // Same row set as the netlib file: MKL draws are the recorded oracle.
    let mut mkl_keys = std::collections::HashSet::new();
    let mkl: std::collections::HashMap<(String, String), f64> =
        std::fs::read_to_string(model_dir.join("dmm_goldens.tsv"))
            .unwrap_or_else(|e| panic!("dmm_goldens.tsv: {e}"))
            .lines()
            .skip(1)
            .filter(|l| !l.is_empty())
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                let key = (f[0].to_string(), f[1].to_string());
                assert!(
                    mkl_keys.insert(key.clone()),
                    "duplicate upstream DMM golden row"
                );
                let value: f64 = f[2].parse().unwrap();
                assert!(value.is_finite(), "nonfinite upstream DMM golden");
                (key, value)
            })
            .collect();
    // Rows: image size dmm_netlib tol kind — `jnd` rows hold the true
    // 1e-3 JND gate vs the netlib score; `pinned` rows record the port's
    // measured solver-drift residual (+5% floor) because even same-tap
    // scipy-vs-transliteration draws diverge past the gate on
    // deep-null-heavy inputs.
    let mut seen = std::collections::HashSet::new();
    let mut checked = 0;
    let mut failures = Vec::new();
    let mut pinned = 0;
    let mut worst_jnd = 0.0f64;
    let mut within_jnd = 0;
    let mut mkl_errors = Vec::new();
    for line in validate_dmm_panel(netlib, &mkl) {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            f.len(),
            6,
            "DMM golden schema must include recorded residual"
        );
        let (stem, size, want, row_tol, kind): (&str, &str, f64, f64, &str) = (
            f[0],
            f[1],
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
            f[4],
        );
        let residual: f64 = f[5].parse().expect("recorded residual");
        assert!(want.is_finite() && row_tol.is_finite() && row_tol >= 0.0);
        assert!(residual.is_finite() && residual >= 0.0);
        match kind {
            "jnd" => assert!(
                row_tol <= 1e-3 * jnd("dmm"),
                "jnd tolerance exceeds canonical gate"
            ),
            "pinned" => assert!(
                row_tol <= 1.05 * residual + 1e-12,
                "pinned tolerance exceeds approved residual +5%"
            ),
            _ => panic!("unknown DMM golden kind {kind}"),
        }
        let key = (stem.to_string(), size.to_string());
        assert!(mkl.contains_key(&key), "extra netlib row {key:?}");
        assert!(seen.insert(key), "duplicate netlib row {stem} {size}");
        let dir = if size.contains('x') {
            &ladder_ns
        } else {
            &ladder
        };
        let (rp, w, h) = load_rgb8(&dir.join(format!("{stem}_{size}_ref.png")));
        let (dp, _, _) = load_rgb8(&dir.join(format!("{stem}_{size}_dist.png")));
        let got = score(
            &*m.evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                .expect("evaluate"),
            "dmm",
        );
        // The qualification host requires repeating CUDA disagreements. Keep
        // both observations visible; the first retains its frozen row limit.
        #[cfg(feature = "ort")]
        if matches!(engine(), Engine::OrtCuda) && (!got.is_finite() || (got - want).abs() > row_tol)
        {
            let repeated = score(
                &*m.evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                    .expect("repeat CUDA disagreement"),
                "dmm",
            );
            eprintln!(
                "dmm CUDA repeat {stem} {size}: first {got:.17e}, repeat {repeated:.17e}, target {want:.17e}, unchanged limit {row_tol:.17e}"
            );
        }
        let err = (got - want).abs();
        within_jnd += usize::from(err <= 1e-3 * jnd("dmm"));
        let mkl_err = mkl
            .get(&(stem.to_string(), size.to_string()))
            .map(|m| (got - m).abs());
        mkl_errors.push(mkl_err.expect("matching MKL golden"));
        if kind == "jnd" {
            worst_jnd = worst_jnd.max(err);
        } else {
            pinned += 1;
        }
        eprintln!(
            "dmm {stem} {size}: got {got:.17e} netlib {want:.17e} |d| {err:.3e} ({kind}, tol {row_tol:.3e}) vs-mkl {:?}",
            mkl_err
        );
        if !err.is_finite() || err > row_tol {
            failures.push(format!(
                "dmm {stem} {size}: |{got} - {want}| = {err:e} > {row_tol:e} ({kind})"
            ));
        }
        checked += 1;
    }
    assert!(checked > 0, "netlib goldens had no rows");
    assert_eq!(
        seen.len(),
        mkl.len(),
        "missing netlib rows from dmm_goldens.tsv"
    );
    mkl_errors.sort_by(f64::total_cmp);
    let median = (mkl_errors[(checked - 1) / 2] + mkl_errors[checked / 2]) / 2.0;
    eprintln!(
        "dmm vs MKL: median |d| {median:.9e}, max {:.9e}",
        mkl_errors[checked - 1]
    );
    eprintln!(
        "dmm: {checked} pairs vs netlib-sgesdd — {within_jnd} inside 1e-3 JND; {} classified jnd (worst {worst_jnd:.3e} = {:.3e} JND), {pinned} pinned solver-drift rows",
        checked - pinned,
        worst_jnd / jnd("dmm")
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn warm_reference_equals_cold() {
    let model_dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let ladder = env_dir("ZENMETRICS_NN_LADDER");
    let ladder_ns = PathBuf::from(format!("{}-nonsquare", ladder.display()));
    for (mut m, col) in models() {
        // First golden row plus every non-square (`WxH` size) row: the
        // warm path must not care about the aspect ratio (ahiq once swapped
        // h/w here).
        let goldens = std::fs::read_to_string(model_dir.join(format!("{col}_goldens.tsv")))
            .unwrap_or_else(|e| panic!("{col}: {e}"));
        let rows: Vec<&str> = goldens.lines().skip(1).collect();
        let mut picks = vec![rows[0]];
        picks.extend(
            rows.iter()
                .filter(|l| l.split('\t').nth(1).unwrap().contains('x')),
        );
        for pick in picks {
            let f: Vec<&str> = pick.split('\t').collect();
            let (stem, size) = (f[0], f[1]);
            let dir = if size.contains('x') {
                &ladder_ns
            } else {
                &ladder
            };
            let (rp, w, h) = load_rgb8(&dir.join(format!("{stem}_{size}_ref.png")));
            let (dp, _, _) = load_rgb8(&dir.join(format!("{stem}_{size}_dist.png")));
            let nr = is_nr(&*m, col);
            let cold = score(
                &*m.evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                    .unwrap(),
                col,
            );
            // CKDN is a quality score, not a distance: an identical pair's dr
            // is sigmoid(fc(csp(qse(x) - dte(x)))) — nonzero because the two
            // encoders differ. A-FINE's raw afine_all and AHIQ's quality score
            // are likewise image-dependent on identical pairs, WaDIQaM-FR's
            // concat on (r, r) is (0, h, h) — a nonzero "perfect pair"
            // quality — and an NR model's headline on (r, r) is the
            // reference's own quality score. For all of them the
            // identical-pair check below is warm == cold.
            let cold_same = (nr
                || matches!(col, "ckdn" | "afine" | "ahiq" | "pieapp")
                || col.starts_with("wadiqam_fr"))
            .then(|| {
                score(
                    &*m.evaluate(image(&rp, w, h), image(&rp, w, h), EvalOpts::new())
                        .unwrap(),
                    col,
                )
            });
            let mut warm = m.warm_reference(image(&rp, w, h), EvalOpts::new()).unwrap();
            // PieAPP caches only planar conversion on both engines. A trunk
            // engine caches the reference's features; a whole-graph engine
            // (ort-cuda) recomputes them on every candidate (ahiq's warm keeps
            // the nine converted crops, still a recomputation of features per
            // evaluate). DMM is the exception: its trunk runs on ort-cuda but
            // the SVD decompositions it caches are host-side, so it reuses
            // reference work on either engine. A paired no-reference metric
            // always scores its reference once, and WaDIQaM-FR's split
            // trunk+head caches the reference's patch features on every
            // engine.
            let expect_reuse = nr
                || (engine() == Engine::Rten && col != "pieapp")
                || col == "dmm"
                || col.starts_with("wadiqam_fr");
            assert_eq!(warm.reuses_reference_work(), expect_reuse, "{col}");
            for _ in 0..2 {
                let s = score(
                    &*warm.evaluate(image(&dp, w, h), EvalOpts::new()).unwrap(),
                    col,
                );
                assert_eq!(s.to_bits(), cold.to_bits(), "{col}");
            }
            let same = score(
                &*warm.evaluate(image(&rp, w, h), EvalOpts::new()).unwrap(),
                col,
            );
            if let Some(cold_same) = cold_same {
                assert_eq!(same.to_bits(), cold_same.to_bits(), "{col}");
                if col == "pieapp" {
                    // Recorded as parity.identical_patch_oracle in models/pieapp/card.toml.
                    const ORACLE_IDENTICAL: f64 = -3.725_290_298_461_914e-9;
                    assert!((same - ORACLE_IDENTICAL).abs() <= tol(col), "{col}");
                }
            } else if engine() == Engine::Rten || col != "dists" {
                assert_eq!(same, 0.0, "{col}");
            } else {
                // On ort-cuda the whole DISTS graph, including 1 - (dist1 + dist2),
                // runs in ONNX Runtime's f32 reduction order: measured 1.19e-7
                // (one f32 step at 1.0) for an identical pair. Bounded by the
                // parity tolerance.
                assert!(
                    same.abs() <= tol(col),
                    "{col}: identical pair scored {same}"
                );
            }
        }
    }
}
