//! Score UPIQ HDR EXR pairs through the production [`HdrScorer`] route for
//! any metric — the same `hdr_feeding` path real HDR scoring takes
//! (cvvdp/butter → LinearPlanes + display model; iwssim → float PU(luma);
//! GPU ssim2 → integrated PU21; rest → u8 PU shell).
//!
//! Reads a TSV (`ref_path<TAB>dist_path`, header row) of absolute-luminance
//! EXRs; emits `ref_path<TAB>dist_path<TAB><score…>` per named score column.
//! Validation harness for `benchmarks/pu_integrated_upiq_2026-06-09.md` /
//! imazen/zenmetrics#25.
//!
//!   cargo run --release -p zenmetrics-api --features hdr,cuda,butter,cvvdp \
//!     --example upiq_hdr_score -- cvvdp \
//!     /mnt/v/output/zenmetrics/upiq-pu/upiq_pairs.tsv /tmp/cvvdp.tsv

use std::collections::HashMap;
use std::io::Write;
use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrScorer};
use zenmetrics_api::{Backend, MetricKind};

fn load_exr_nits(path: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let img = image::open(path)
        .map_err(|e| format!("{path}: {e}"))?
        .to_rgb32f();
    let (w, h) = (img.width(), img.height());
    Ok((img.into_raw(), w, h))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = match args.get(1).map(String::as_str).unwrap_or("cvvdp") {
        "cvvdp" => MetricKind::Cvvdp,
        "butter" => MetricKind::Butter,
        "ssim2" => MetricKind::Ssim2,
        "iwssim" => MetricKind::Iwssim,
        "dssim" => MetricKind::Dssim,
        "zensim" => MetricKind::Zensim,
        other => panic!("unknown metric kind: {other}"),
    };
    let pairs = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/mnt/v/output/zenmetrics/upiq-pu/upiq_pairs.tsv".into());
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("/tmp/upiq_hdr_{kind:?}.tsv").to_lowercase());
    let peak: f32 = args
        .get(4)
        .map(|v| v.parse().expect("peak nits"))
        .unwrap_or(HDR_PEAK_NITS);

    let body = std::fs::read_to_string(&pairs).expect("read pairs tsv");
    let mut cache: HashMap<String, (Vec<f32>, u32, u32)> = HashMap::new();
    let mut scorers: HashMap<(u32, u32), HdrScorer> = HashMap::new();
    let mut out = String::new();
    let mut header_written = false;
    let (mut ok, mut err) = (0usize, 0usize);

    for line in body.lines().skip(1) {
        let mut it = line.split('\t');
        let (Some(rp), Some(dp)) = (it.next(), it.next()) else {
            continue;
        };
        for p in [rp, dp] {
            if !cache.contains_key(p) {
                match load_exr_nits(p) {
                    Ok(v) => {
                        cache.insert(p.to_string(), v);
                    }
                    Err(e) => eprintln!("LOAD FAIL {e}"),
                }
            }
        }
        let (Some((r, rw, rh)), Some((d, dw, dh))) = (cache.get(rp), cache.get(dp)) else {
            err += 1;
            continue;
        };
        if (rw, rh) != (dw, dh) {
            err += 1;
            continue;
        }
        let scorer = match scorers.entry((*rw, *rh)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => v.insert(
                HdrScorer::new(kind, Backend::Auto, *rw, *rh, peak).expect("HdrScorer::new"),
            ),
        };
        match scorer.compute_multi(r, d) {
            Ok(s) => {
                if !header_written {
                    out.push_str("ref_path\tdist_path");
                    for ns in &s.scores {
                        out.push('\t');
                        out.push_str(ns.name);
                    }
                    out.push('\n');
                    header_written = true;
                }
                out.push_str(&format!("{rp}\t{dp}"));
                for ns in &s.scores {
                    out.push_str(&format!("\t{}", ns.value));
                }
                out.push('\n');
                ok += 1;
                if ok % 50 == 0 {
                    eprintln!("scored {ok}…");
                }
            }
            Err(e) => {
                eprintln!("SCORE FAIL {rp}|{dp}: {e:?}");
                err += 1;
            }
        }
    }

    let mut f = std::fs::File::create(&out_path).expect("create out");
    f.write_all(out.as_bytes()).expect("write out");
    eprintln!("done: {ok} scored, {err} errored -> {out_path}");
}
