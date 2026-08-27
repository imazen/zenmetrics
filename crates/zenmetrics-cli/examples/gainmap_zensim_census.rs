//! 27-cell HDR zensim census for the jpeg-gainmap (Ultra HDR) fleet arm.
//!
//! Registration (design + gates FROZEN pre-run):
//! `benchmarks/gainmap_zensim_census_2026-08-27.md`.
//!
//! Encoder = the fleet's own [`sweep::hdr::encode_hdr`] with
//! [`HdrCodec::JpegGainmap`], in-process — same crate + commit as the fleet
//! executor, so the census measures exactly the code the fleet runs.
//! Judge = shelled sibling `zenmetrics score --metric zensim --hdr`
//! (the fleet-proven HDR route; output parsed at `zensim=`).
//! Search = blind-midpoint bracketed bisection on base JPEG q in [1,100]
//! (score assumed monotone non-decreasing in q), best-not-last, budget k.
//!
//! Usage:
//!   gainmap_zensim_census <refs.tsv> <refs_dir> <targets-csv> <k> <zm-bin> <out_dir>
//!
//! `refs.tsv` = the frozen instrument (scene\ttier\trendition, `#` comments).
//! Writes `<out_dir>/census_k<k>.tsv` + each cell's best `.jpg`.

use serde_json::Map;
use std::io::Write as _;
use zenmetrics_cli::sweep::hdr::{HdrCodec, HdrRef, decode_hdr_ref, encode_hdr};

fn judge(zm: &str, reference: &str, distorted: &str) -> Result<f64, String> {
    let o = std::process::Command::new(zm)
        .args([
            "score",
            "--metric",
            "zensim",
            "--hdr",
            "--reference",
            reference,
            "--distorted",
            distorted,
        ])
        .output()
        .map_err(|e| format!("spawn {zm}: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "judge rc={:?}: {}",
            o.status.code(),
            String::from_utf8_lossy(&o.stderr)
                .chars()
                .take(200)
                .collect::<String>()
        ));
    }
    let s = String::from_utf8_lossy(&o.stdout);
    s.split("zensim=")
        .nth(1)
        .and_then(|x| x.trim().parse::<f64>().ok())
        .ok_or_else(|| format!("unparsable judge output: {s}"))
}

struct CellResult {
    q_best: u8,
    encodes: u8,
    score_best: f64,
    bytes_best: Vec<u8>,
}

fn search_cell(
    source: &HdrRef,
    target: f64,
    k: u8,
    zm: &str,
    ref_path: &str,
    tmp_jpg: &str,
) -> Result<CellResult, String> {
    let knobs = Map::new(); // arm defaults: gm_quality=85, gm_scale=4
    let (mut lo, mut hi) = (1i32, 100i32);
    let mut best: Option<(f64, u8, f64, Vec<u8>)> = None; // (|err|, q, score, bytes)
    let mut used = 0u8;
    while used < k && lo <= hi {
        let q = (lo + hi) / 2;
        let cell = encode_hdr(HdrCodec::JpegGainmap, source, q as f64, &knobs)
            .map_err(|e| format!("encode q{q}: {e}"))?;
        std::fs::write(tmp_jpg, &cell.bytes).map_err(|e| format!("write {tmp_jpg}: {e}"))?;
        let s = judge(zm, ref_path, tmp_jpg)?;
        used += 1;
        let err = (s - target).abs();
        if best.as_ref().map(|b| err < b.0).unwrap_or(true) {
            best = Some((err, q as u8, s, cell.bytes));
        }
        if s < target {
            lo = q + 1;
        } else {
            hi = q - 1;
        }
    }
    let (_, q_best, score_best, bytes_best) = best.ok_or("no encodes ran")?;
    Ok(CellResult {
        q_best,
        encodes: used,
        score_best,
        bytes_best,
    })
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [refs_tsv, refs_dir, targets_csv, k_s, zm_bin, out_dir] = match a.as_slice() {
        [a1, a2, a3, a4, a5, a6] => [a1, a2, a3, a4, a5, a6].map(String::from),
        _ => {
            eprintln!(
                "usage: gainmap_zensim_census <refs.tsv> <refs_dir> <targets-csv> <k> <zm-bin> <out_dir>"
            );
            std::process::exit(2);
        }
    };
    let k: u8 = k_s.parse().expect("k");
    let targets: Vec<f64> = targets_csv
        .split(',')
        .map(|t| t.parse().expect("target"))
        .collect();
    std::fs::create_dir_all(&out_dir).expect("out_dir");
    let out_path = format!("{out_dir}/census_k{k}.tsv");
    let mut tsv = std::fs::File::create(&out_path).expect("tsv create");
    writeln!(
        tsv,
        "scene\ttier\ttarget\tq_best\tencodes_used\tscore_best\tabs_err\tbytes_best\tsecs"
    )
    .unwrap();

    let body = std::fs::read_to_string(&refs_tsv).expect("refs tsv");
    let mut rows: Vec<(String, String, f64, u8, u8, f64, f64, usize, f64)> = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("scene\t") {
            continue;
        }
        let mut f = line.split('\t');
        let (scene, tier, rendition) = (
            f.next().unwrap().to_string(),
            f.next().unwrap().to_string(),
            f.next().unwrap().to_string(),
        );
        let ref_path = format!("{refs_dir}/{rendition}");
        let source = decode_hdr_ref(std::path::Path::new(&ref_path))
            .unwrap_or_else(|e| panic!("{scene}: decode_hdr_ref: {e}"));
        for &t in &targets {
            let t0 = std::time::Instant::now();
            let best_jpg = format!("{out_dir}/{scene}_t{t:.0}_k{k}.jpg");
            let r = search_cell(&source, t, k, &zm_bin, &ref_path, &best_jpg)
                .unwrap_or_else(|e| panic!("{scene} t{t}: {e}"));
            // best-not-last: the tmp file holds the LAST trial; rewrite the BEST.
            std::fs::write(&best_jpg, &r.bytes_best).expect("write best");
            let secs = t0.elapsed().as_secs_f64();
            let err = (r.score_best - t).abs();
            writeln!(
                tsv,
                "{scene}\t{tier}\t{t:.0}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{secs:.1}",
                r.q_best,
                r.encodes,
                r.score_best,
                err,
                r.bytes_best.len(),
            )
            .unwrap();
            eprintln!(
                "{scene} t{t:.0}: q={} achieved={:.2} |err|={:.2} ({secs:.0}s)",
                r.q_best, r.score_best, err
            );
            rows.push((
                scene.clone(),
                tier.clone(),
                t,
                r.q_best,
                r.encodes,
                r.score_best,
                err,
                r.bytes_best.len(),
                secs,
            ));
        }
    }

    // Summary: median |err|, ±2 hits, per-target + per-tier medians.
    let mut all: Vec<f64> = rows.iter().map(|r| r.6).collect();
    let hits = rows.iter().filter(|r| r.6 <= 2.0).count();
    let med = median(&mut all);
    let mut per_t = String::new();
    for &t in &targets {
        let mut v: Vec<f64> = rows.iter().filter(|r| r.2 == t).map(|r| r.6).collect();
        per_t.push_str(&format!("t{t:.0}={:.2} ", median(&mut v)));
    }
    let mut per_tier = String::new();
    for tier in ["large", "mid", "small"] {
        let mut v: Vec<f64> = rows.iter().filter(|r| r.1 == tier).map(|r| r.6).collect();
        per_tier.push_str(&format!("{tier}={:.2} ", median(&mut v)));
    }
    let summary = format!(
        "k{k}: n={} median|err|={:.3} +-2hits={}/{} per-t: {per_t}per-tier: {per_tier}",
        rows.len(),
        med,
        hits,
        rows.len()
    );
    writeln!(tsv, "# {summary}").unwrap();
    eprintln!("{summary}");
    println!("{summary}");
}
