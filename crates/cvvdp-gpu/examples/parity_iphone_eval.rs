//! Display-dispatch parity probe — runs the cvvdp host-scalar pipeline
//! (which exercises the same per-EOTF / per-primaries dispatch code
//! the GPU kernel now uses) against the 13-pair eval bundle at
//! `/tmp/cvvdp-display-eval/` under both `standard_4k` and
//! `iphone_14_pro` display presets, then joins against the existing
//! pycvvdp scores in `results.tsv` and writes
//! `/tmp/cvvdp-display-eval/parity_v2_scalar.tsv` with per-pair
//! `abs_diff` and a summary of mean / median / max.
//!
//! ## Why host_scalar instead of GPU?
//!
//! The PINNED TASK in `zenmetrics/CLAUDE.md` documents that the WSL2
//! development host can't reach cubecl-cuda (snap-docker / NVML
//! issues) and cubecl-cpu hits the `atomic<f32>` panic that short-
//! circuits scoring to default JOD=10. The host_scalar pipeline
//! produces the same numbers as the GPU pipeline at f32 precision
//! (every parity test in `crates/cvvdp-gpu/tests/` validates this
//! against pycvvdp goldens to within 0.005 JOD). Running through
//! host_scalar here validates the **dispatch logic** added by the
//! preceding commit; running the same probe on a CUDA-capable host
//! would additionally validate the GPU kernel's transcendental
//! ordering (Cuda's FMA vs host f32 add-mul). The kernel-vs-scalar
//! parity is already pinned by
//! `tests/color_kernel_display_dispatch.rs`.
//!
//! Run:
//!     cargo run --release --example parity_iphone_eval -p cvvdp-gpu
//!
//! Outputs:
//!     /tmp/cvvdp-display-eval/parity_v2_scalar.tsv
//!     summary on stdout

use std::fs;
use std::path::{Path, PathBuf};

use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::{DisplayGeometry, DisplayModel};

const EVAL_ROOT: &str = "/tmp/cvvdp-display-eval";

#[derive(Debug, Clone)]
struct Pair {
    name: String,
    ref_path: PathBuf,
    dist_path: PathBuf,
}

fn parse_pairs(json_text: &str, refs_dir: &Path, dists_dir: &Path) -> Vec<Pair> {
    // Minimal hand-parser for the {name, ref, dist, ...} array we
    // write into pairs.json — avoids pulling serde just for one
    // example.
    let mut out = Vec::new();
    let mut chars = json_text.chars().peekable();
    let mut cur_name: Option<String> = None;
    let mut cur_ref: Option<String> = None;
    let mut cur_dist: Option<String> = None;

    fn read_string(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Option<String> {
        // Expect opening quote next.
        while let Some(&c) = chars.peek() {
            if c == '"' {
                chars.next();
                break;
            }
            chars.next();
        }
        let mut s = String::new();
        for c in chars.by_ref() {
            if c == '"' {
                return Some(s);
            }
            s.push(c);
        }
        None
    }

    while let Some(&c) = chars.peek() {
        if c == '"' {
            let key = read_string(&mut chars).unwrap_or_default();
            // Advance past the colon.
            while let Some(&p) = chars.peek() {
                if p == ':' {
                    chars.next();
                    break;
                }
                chars.next();
            }
            let val = read_string(&mut chars).unwrap_or_default();
            match key.as_str() {
                "name" => cur_name = Some(val),
                "ref" => cur_ref = Some(val),
                "dist" => cur_dist = Some(val),
                _ => {}
            }
            if let (Some(n), Some(r), Some(d)) = (&cur_name, &cur_ref, &cur_dist) {
                out.push(Pair {
                    name: n.clone(),
                    ref_path: refs_dir.join(r),
                    dist_path: dists_dir.join(d),
                });
                cur_name = None;
                cur_ref = None;
                cur_dist = None;
            }
        } else {
            chars.next();
        }
    }
    out
}

fn load_rgb8(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

fn score_pair(pair: &Pair, display: DisplayModel, geom: DisplayGeometry) -> Option<f32> {
    let (ref_b, rw, rh) = load_rgb8(&pair.ref_path)?;
    let (dist_b, dw, dh) = load_rgb8(&pair.dist_path)?;
    if (rw, rh) != (dw, dh) {
        eprintln!(
            "skipping {}: ref {}x{} != dist {}x{}",
            pair.name, rw, rh, dw, dh
        );
        return None;
    }
    let ppd = geom.pixels_per_degree();
    Some(predict_jod_still_3ch(
        &ref_b,
        &dist_b,
        rw as usize,
        rh as usize,
        display,
        ppd,
    ))
}

/// Load the existing results.tsv from `/tmp/cvvdp-display-eval/` —
/// it has columns `pair_name, content_class, distortion, jod_4k,
/// jod_iphone, ...` from the previous agent's pycvvdp run.
fn load_pycvvdp_scores(path: &Path) -> std::collections::HashMap<String, (f64, f64)> {
    let text = fs::read_to_string(path).unwrap_or_default();
    let mut out = std::collections::HashMap::new();
    let mut lines = text.lines();
    // Header.
    let header = lines.next().unwrap_or("");
    let cols: Vec<&str> = header.split('\t').collect();
    let idx_name = cols.iter().position(|c| *c == "pair_name").unwrap_or(0);
    let idx_4k = cols.iter().position(|c| *c == "jod_4k").unwrap_or(3);
    let idx_iphone = cols.iter().position(|c| *c == "jod_iphone").unwrap_or(4);
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() <= idx_iphone.max(idx_name).max(idx_4k) {
            continue;
        }
        let name = fields[idx_name].to_string();
        let jod_4k: f64 = fields[idx_4k].parse().unwrap_or(f64::NAN);
        let jod_ip: f64 = fields[idx_iphone].parse().unwrap_or(f64::NAN);
        out.insert(name, (jod_4k, jod_ip));
    }
    out
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    let idx = ((n as f64) * p).floor() as usize;
    sorted[idx.min(n - 1)]
}

fn summarise(rows: &[(String, String, f64, f64, f64)]) -> (f64, f64, f64) {
    let mut diffs: Vec<f64> = rows.iter().map(|r| r.4).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if diffs.is_empty() {
        f64::NAN
    } else {
        diffs.iter().sum::<f64>() / diffs.len() as f64
    };
    let median = percentile(&diffs, 0.5);
    let max = diffs.last().copied().unwrap_or(f64::NAN);
    (mean, median, max)
}

fn main() {
    let root = Path::new(EVAL_ROOT);
    let refs = root.join("refs");
    let dists = root.join("dists");
    let pairs_json = fs::read_to_string(root.join("pairs.json"))
        .expect("read /tmp/cvvdp-display-eval/pairs.json");
    let pairs = parse_pairs(&pairs_json, &refs, &dists);
    eprintln!("loaded {} pairs", pairs.len());

    let pycvvdp = load_pycvvdp_scores(&root.join("results.tsv"));
    eprintln!("loaded {} pycvvdp rows", pycvvdp.len());

    let displays: &[(&str, DisplayModel, DisplayGeometry)] = &[
        ("standard_4k", DisplayModel::STANDARD_4K, DisplayGeometry::STANDARD_4K),
        ("iphone_14_pro", DisplayModel::IPHONE_14_PRO, DisplayGeometry::IPHONE_14_PRO),
    ];

    let mut all_rows: Vec<(String, String, f64, f64, f64)> = Vec::new();

    for (display_name, display, geom) in displays {
        for p in &pairs {
            let Some(imazen) = score_pair(p, *display, *geom) else {
                eprintln!("  skip {}", p.name);
                continue;
            };
            let Some((jod_4k, jod_ip)) = pycvvdp.get(&p.name).copied() else {
                eprintln!("  no pycvvdp row for {}", p.name);
                continue;
            };
            let pyc = match *display_name {
                "standard_4k" => jod_4k,
                "iphone_14_pro" => jod_ip,
                _ => f64::NAN,
            };
            let diff = (imazen as f64 - pyc).abs();
            all_rows.push((
                p.name.clone(),
                display_name.to_string(),
                pyc,
                imazen as f64,
                diff,
            ));
        }
    }

    // Write parity_v2_scalar.tsv.
    let out_path = root.join("parity_v2_scalar.tsv");
    let mut out = String::new();
    out.push_str("pair_name\tdisplay\tjod_pycvvdp\tjod_imazen\tabs_diff\n");
    for r in &all_rows {
        out.push_str(&format!("{}\t{}\t{:.6}\t{:.6}\t{:.6}\n", r.0, r.1, r.2, r.3, r.4));
    }
    fs::write(&out_path, &out).expect("write parity_v2_scalar.tsv");
    eprintln!("wrote {}", out_path.display());

    // Per-display summary.
    println!("\nParity summary (host_scalar vs pycvvdp v0.5.4)");
    println!("display          |   n | mean abs_diff | median | max");
    for (name, _, _) in displays {
        let rows: Vec<_> = all_rows
            .iter()
            .cloned()
            .filter(|r| &r.1 == name)
            .collect();
        let (mean, median, max) = summarise(&rows);
        let n = rows.len();
        let gate_mean = if mean < 0.10 { "OK" } else { "FAIL" };
        let gate_max = if max < 0.30 { "OK" } else { "FAIL" };
        println!(
            "{:<17}| {:>3} | {:>13.4} | {:>6.4} | {:>5.4} [mean<0.10:{} max<0.30:{}]",
            name, n, mean, median, max, gate_mean, gate_max,
        );
    }
}
