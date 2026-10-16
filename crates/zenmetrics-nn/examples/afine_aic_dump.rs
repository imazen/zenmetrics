//! Dump every A-FINE sub-score per (pair, preprocessing variant) — the
//! data source for the AIC-4 `A-FINE` convention search in
//! `benchmarks/afine_aic_2026-10-16/` (and its `afine_aic_search.py`
//! evaluator).
//!
//! ```text
//! cargo run --release -p zenmetrics-nn --features ort \
//!   --example afine_aic_dump -- \
//!   <model_dir> <pairs.tsv> <out.tsv> [engine=rten|ort-cuda] [variants=native,...]
//! ```
//!
//! `pairs.tsv`: `ref_path<TAB>dist_path[<TAB>name]` rows (`#`/header
//! lines skipped); `name` defaults to the distorted filename. Output
//! TSV: one row per (pair, variant) with columns
//! `pair variant n_r n_d fid s_r s_d s_f lambda afine_all`. The fused
//! ort-cuda graph emits only `afine_all` (sub-score fields blank).
//! Engines: `rten` (default) or `ort-cuda` (feature `ort`, needs
//! `ORT_DYLIB_PATH` plus the CUDA/cuDNN libs on `LD_LIBRARY_PATH`).

use std::io::Write;

use zenmetrics_nn::{Afine, Engine, WeightsConsent};

const VARIANTS: &[&str] = &["native", "r224", "r384", "r512", "nonorm", "bgr"];

fn load_rgb8(path: &str) -> (Vec<u8>, usize, usize) {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let out = zenpng::decode(
        &data,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .unwrap_or_else(|e| panic!("{path}: {e}"));
    let s = out.pixels.as_slice();
    assert_eq!(s.descriptor().bytes_per_pixel(), 3, "RGB8 required: {path}");
    let (w, h) = (s.width() as usize, s.rows() as usize);
    let bytes = s.as_strided_bytes();
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        px.extend_from_slice(&bytes[y * s.stride() as usize..][..w * 3]);
    }
    (px, w, h)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if !(4..=6).contains(&args.len()) {
        eprintln!(
            "usage: {} MODEL_DIR PAIRS_TSV OUT_TSV [rten|ort-cuda] [VARIANTS,csv]",
            args[0]
        );
        std::process::exit(2);
    }
    let (dir, pairs, out) = (&args[1], &args[2], &args[3]);
    let engine = match args.get(4).map(String::as_str) {
        None | Some("rten") => Engine::Rten,
        #[cfg(feature = "ort")]
        Some("ort-cuda") => Engine::OrtCuda,
        Some(other) => {
            eprintln!("unknown engine {other:?} (rten | ort-cuda)");
            std::process::exit(2);
        }
    };
    let variants: Vec<String> = args
        .get(5)
        .filter(|v| !v.is_empty())
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_else(|| VARIANTS.iter().map(|s| s.to_string()).collect());

    let consent = WeightsConsent::new().accept(Afine::WEIGHTS);
    let af = Afine::from_dir(dir, engine, &consent).expect("load afine");

    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("create out"));
    writeln!(f, "{}", af.aic_dump_header()).unwrap();
    writeln!(
        f,
        "pair\tvariant\tn_r\tn_d\tfid\ts_r\ts_d\ts_f\tlambda\tafine_all"
    )
    .unwrap();

    for line in std::fs::read_to_string(pairs).expect("read pairs").lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("ref_path") {
            continue;
        }
        let mut fields = line.split('\t');
        let (rp, dp) = (fields.next().expect("ref"), fields.next().expect("dist"));
        let name = fields.next().map(str::to_string).unwrap_or_else(|| {
            std::path::Path::new(dp)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string()
        });
        let (rpx, rw0, rh0) = load_rgb8(rp);
        let (dpx, dw0, dh0) = load_rgb8(dp);
        assert_eq!((rw0, rh0), (dw0, dh0), "size mismatch: {rp} vs {dp}");
        for vname in &variants {
            match af.aic_dump_scores(&rpx, &dpx, rw0, rh0, vname) {
                Ok(row) => {
                    let (cells, all) = row.split_at(7);
                    let sub: Vec<String> = cells
                        .iter()
                        .map(|v| {
                            if v.is_nan() {
                                String::new()
                            } else {
                                format!("{v:.17e}")
                            }
                        })
                        .collect();
                    writeln!(f, "{name}\t{vname}\t{}\t{:.17e}", sub.join("\t"), all[0]).unwrap();
                }
                Err(e) => writeln!(f, "{name}\t{vname}\t\t\t\t\t\t\t\tERROR:{e}").unwrap(),
            }
        }
        f.flush().unwrap();
        eprintln!("done {name}");
    }
}
