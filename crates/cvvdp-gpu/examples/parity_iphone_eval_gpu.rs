//! GPU variant of `parity_iphone_eval`. Runs the cubecl-cuda pipeline
//! against the 13-pair bundle at `/tmp/cvvdp-display-eval/` under both
//! `standard_4k` and `iphone_14_pro` display presets, joins against the
//! pycvvdp v0.5.4 reference scores in `results.tsv`, and writes
//! `/tmp/cvvdp-display-eval/parity_v2_gpu.tsv` plus a stdout summary.
//!
//! Build + run:
//!     cargo run --release -p cvvdp-gpu \
//!         --features cuda --no-default-features --example parity_iphone_eval_gpu
//!
//! All 13 pairs are 1024×768 sRGB, so one `Cvvdp` instance per display
//! preset is enough (geometry differs between standard_4k and
//! iphone_14_pro and is what drives ppd-dependent CSF queries).

use std::fs;
use std::path::{Path, PathBuf};

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel};

const EVAL_ROOT: &str = "/tmp/cvvdp-display-eval";

#[derive(Debug, Clone)]
struct Pair {
    name: String,
    ref_path: PathBuf,
    dist_path: PathBuf,
}

fn parse_pairs(json_text: &str, refs_dir: &Path, dists_dir: &Path) -> Vec<Pair> {
    let mut out = Vec::new();
    let mut chars = json_text.chars().peekable();
    let mut cur_name: Option<String> = None;
    let mut cur_ref: Option<String> = None;
    let mut cur_dist: Option<String> = None;

    fn read_string(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Option<String> {
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

fn load_pycvvdp_scores(path: &Path) -> std::collections::HashMap<String, (f64, f64)> {
    let text = fs::read_to_string(path).unwrap_or_default();
    let mut out = std::collections::HashMap::new();
    let mut lines = text.lines();
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

fn summarise(diffs: &[f64]) -> (f64, f64, f64) {
    let mut d = diffs.to_vec();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if d.is_empty() {
        f64::NAN
    } else {
        d.iter().sum::<f64>() / d.len() as f64
    };
    let median = percentile(&d, 0.5);
    let max = d.last().copied().unwrap_or(f64::NAN);
    (mean, median, max)
}

fn main() {
    let root = Path::new(EVAL_ROOT);
    let refs_dir = root.join("refs");
    let dists_dir = root.join("dists");
    let pairs_json =
        fs::read_to_string(root.join("pairs.json")).expect("read pairs.json");
    let pairs = parse_pairs(&pairs_json, &refs_dir, &dists_dir);
    eprintln!("loaded {} pairs", pairs.len());

    let pycvvdp = load_pycvvdp_scores(&root.join("results.tsv"));
    eprintln!("loaded {} pycvvdp rows", pycvvdp.len());

    // Pre-load all pairs; all 13 are 1024x768.
    let mut loaded: Vec<(Pair, Vec<u8>, Vec<u8>, u32, u32)> = Vec::new();
    for p in &pairs {
        let (Some((r, rw, rh)), Some((d, dw, dh))) =
            (load_rgb8(&p.ref_path), load_rgb8(&p.dist_path))
        else {
            eprintln!("  skip {} (image load)", p.name);
            continue;
        };
        if (rw, rh) != (dw, dh) {
            eprintln!("  skip {} (size mismatch {}x{} vs {}x{})", p.name, rw, rh, dw, dh);
            continue;
        }
        loaded.push((p.clone(), r, d, rw, rh));
    }
    if loaded.is_empty() {
        eprintln!("no usable pairs — aborting");
        std::process::exit(1);
    }
    let (w, h) = (loaded[0].3, loaded[0].4);
    eprintln!("using size {}x{} for Cvvdp instances", w, h);
    for (p, _, _, pw, ph) in &loaded {
        assert_eq!(
            (*pw, *ph),
            (w, h),
            "{}: mixed sizes not supported in this harness",
            p.name
        );
    }

    let displays: &[(&str, DisplayModel, DisplayGeometry)] = &[
        ("standard_4k", DisplayModel::STANDARD_4K, DisplayGeometry::STANDARD_4K),
        ("iphone_14_pro", DisplayModel::IPHONE_14_PRO, DisplayGeometry::IPHONE_14_PRO),
    ];

    let client = CudaRuntime::client(&Default::default());

    let mut all_rows: Vec<(String, String, f64, f64, f64)> = Vec::new();

    for (display_name, display, geom) in displays {
        let params = CvvdpParams {
            display: *display,
            ..CvvdpParams::PLACEHOLDER
        };
        let mut cvvdp = Cvvdp::<CudaRuntime>::new_with_geometry(
            client.clone(),
            w,
            h,
            params,
            *geom,
        )
        .expect("Cvvdp::new_with_geometry");

        for (p, ref_bytes, dist_bytes, _, _) in &loaded {
            let jod = match cvvdp.score(ref_bytes, dist_bytes) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("  {}/{} score error: {:?}", display_name, p.name, e);
                    continue;
                }
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
            let diff = (jod - pyc).abs();
            all_rows.push((
                p.name.clone(),
                display_name.to_string(),
                pyc,
                jod,
                diff,
            ));
        }
    }

    let out_path = root.join("parity_v2_gpu.tsv");
    let mut out = String::new();
    out.push_str("pair_name\tdisplay\tjod_pycvvdp\tjod_imazen\tabs_diff\n");
    for r in &all_rows {
        out.push_str(&format!(
            "{}\t{}\t{:.6}\t{:.6}\t{:.6}\n",
            r.0, r.1, r.2, r.3, r.4
        ));
    }
    fs::write(&out_path, &out).expect("write parity_v2_gpu.tsv");
    eprintln!("wrote {}", out_path.display());

    println!("\nParity summary (cvvdp-gpu CUDA backend vs pycvvdp v0.5.4)");
    println!("display          |   n | mean abs_diff | median |    max");
    for (name, _, _) in displays {
        let diffs: Vec<f64> = all_rows
            .iter()
            .filter(|r| &r.1 == name)
            .map(|r| r.4)
            .collect();
        let (mean, median, max) = summarise(&diffs);
        let n = diffs.len();
        let gate_mean = if mean < 0.10 { "OK" } else { "FAIL" };
        let gate_max = if max < 0.30 { "OK" } else { "FAIL" };
        println!(
            "{:<17}| {:>3} | {:>13.4} | {:>6.4} | {:>6.4} [mean<0.10:{} max<0.30:{}]",
            name, n, mean, median, max, gate_mean, gate_max,
        );
    }

    // Per-pair side-by-side print so the comparison vs the existing
    // host_scalar TSV is one glance.
    println!("\nPer-pair detail (jod_imazen_gpu | jod_pycvvdp | abs_diff):");
    println!(
        "{:<28} {:<14} {:>8} {:>8} {:>8}",
        "pair_name", "display", "gpu", "pycvvdp", "diff"
    );
    for r in &all_rows {
        println!(
            "{:<28} {:<14} {:>8.4} {:>8.4} {:>8.4}",
            r.0, r.1, r.3, r.2, r.4
        );
    }
}
