//! CPU-side 372-feature extractor — mirror of `extract_acumen_features`
//! but uses `zensim::Zensim::compute_all_features` over a rayon worker
//! pool instead of the GPU pipeline.
//!
//! Why: GPU profiling (nsys, 2026-05-22) showed GPU compute is 2.4 ms/pair
//! but wall time is 220 ms/pair — GPU at 1-11% utilization, 99% of time
//! is decode + dispatch + sync overhead. On a 16-core 7950X w/ AVX-512,
//! pure CPU 372-feature extraction with rayon over pairs is competitive
//! (and sometimes faster) than GPU on this single-workstation workload.
//!
//! Schema-identical output to extract_acumen_features (372 cols), same
//! `--regime with_iw` semantics.
//!
//! Usage:
//!   extract_acumen_features_cpu \
//!     --pairs-tsv /tmp/safesyn_pairs.tsv \
//!     --out /tmp/safesyn_cpu.parquet \
//!     --acumen-mode-a --acumen-arch mode_b --mode-b-band-idx 3

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use zensim::{Zensim, ZensimProfile};
use zensim::source::RgbSlice;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AcumenArchSel {
    Off,
    ModeBLite,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pairs_tsv: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut acumen_mode_a = false;
    let mut acumen_ppd = 56.0_f32;
    let mut acumen_peak = 100.0_f32;
    let mut acumen_ambient = 5.0_f32;
    let mut acumen_arch = AcumenArchSel::Off;
    let mut mode_b_blur_sigma: usize = 8;
    let mut mode_b_band_idx: u32 = 3;
    let mut mode_b_clamp_lo: f32 = 0.1;
    let mut mode_b_clamp_hi: f32 = 4.0;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pairs-tsv" => pairs_tsv = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--acumen-mode-a" => acumen_mode_a = true,
            "--acumen-ppd" => acumen_ppd = args.next().unwrap().parse()?,
            "--acumen-peak-nits" => acumen_peak = args.next().unwrap().parse()?,
            "--acumen-ambient-nits" => acumen_ambient = args.next().unwrap().parse()?,
            "--acumen-arch" => {
                let v = args.next().unwrap();
                acumen_arch = match v.as_str() {
                    "off" | "none" => AcumenArchSel::Off,
                    "mode_b" | "mode-b" | "modeb" => AcumenArchSel::ModeBLite,
                    _ => return Err(format!("unknown --acumen-arch: {v}").into()),
                };
            }
            "--mode-b-blur-sigma" => mode_b_blur_sigma = args.next().unwrap().parse()?,
            "--mode-b-band-idx" => mode_b_band_idx = args.next().unwrap().parse()?,
            "--mode-b-clamp-lo" => mode_b_clamp_lo = args.next().unwrap().parse()?,
            "--mode-b-clamp-hi" => mode_b_clamp_hi = args.next().unwrap().parse()?,
            _ => return Err(format!("unknown arg: {arg}").into()),
        }
    }

    let pairs_tsv = pairs_tsv.ok_or("--pairs-tsv required")?;
    let out = out.ok_or("--out required")?;

    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let file = File::open(&pairs_tsv)?;
    let mut lines = BufReader::new(file).lines();
    let header = lines.next().ok_or("empty TSV")??;
    let cols: Vec<&str> = header.split('\t').collect();
    let ref_idx = cols.iter().position(|c| *c == "ref_path").ok_or("missing ref_path col")?;
    let dist_idx = cols.iter().position(|c| *c == "dist_path").ok_or("missing dist_path col")?;
    for line in lines {
        let line = line?;
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() <= ref_idx.max(dist_idx) {
            continue;
        }
        pairs.push((PathBuf::from(parts[ref_idx]), PathBuf::from(parts[dist_idx])));
    }
    eprintln!("[cpu] {} pairs from {pairs_tsv:?}", pairs.len());
    eprintln!("[cpu] threads: {}", rayon::current_num_threads());
    eprintln!("[cpu] acumen arch: {acumen_arch:?}");

    // 372 features = WithIw regime on CPU. Use compute_all_features.
    const N_FEATS: usize = 372;

    let mut fields = vec![Field::new("ref_basename", DataType::Utf8, false)];
    for i in 0..N_FEATS {
        fields.push(Field::new(&format!("f{i}"), DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));
    let writer_props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let out_file = File::create(&out)?;
    let mut writer = ArrowWriter::try_new(out_file, schema.clone(), Some(writer_props))?;

    // Build Mode B prerequisites once.
    let lut_bytes: &[u8] = include_bytes!("../data/castle_csf_v0_5_4_cvvdp.lut");
    let lut = zensim::acumen::castle_csf::CastleCsfLut::from_bytes(lut_bytes)
        .map_err(|e| format!("LUT: {e:?}"))?;
    let viewing = zensim::acumen::viewing::ViewingCondition::new(
        acumen_ppd, acumen_peak, acumen_ambient,
    );
    let mode_b_cfg = zensim::acumen::mode_b::ModeBConfig {
        blur_sigma: mode_b_blur_sigma,
        band_idx: mode_b_band_idx,
        clamp_lo: mode_b_clamp_lo,
        clamp_hi: mode_b_clamp_hi,
    };

    // Group pairs by reference so the Mode B ref preprocessor amortizes.
    // (For pairs.tsv that's already sorted by ref, this is no-op; for
    // random TSV input, this transparently improves cache behavior.)
    let mut groups: std::collections::BTreeMap<PathBuf, Vec<(usize, PathBuf)>> =
        std::collections::BTreeMap::new();
    for (idx, (r, d)) in pairs.iter().enumerate() {
        groups.entry(r.clone()).or_default().push((idx, d.clone()));
    }
    eprintln!("[cpu] {} unique refs across {} pairs", groups.len(), pairs.len());

    // Per-group: preprocess ref once, then rayon-parallel over distortions.
    // Result is collected into (idx, basename, feats) tuples so we can
    // reassemble in original TSV order at the end.
    let total_pairs = pairs.len();
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<(usize, String, Vec<f64>)> = groups
        .iter()
        .flat_map(|(ref_path, dists)| {
            // Decode ref once.
            let (r_rgb, w_r, h_r) = match decode_image(ref_path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("  decode ref failed for {ref_path:?}: {e}");
                    return Vec::new();
                }
            };

            // Mode B preprocessing of ref (once per ref).
            let mut mode_b_pre = if matches!(acumen_arch, AcumenArchSel::ModeBLite) && acumen_mode_a {
                let mut pre = zensim::acumen::mode_b::ModeBPreprocessor::new(
                    &lut, viewing, w_r, h_r, mode_b_cfg,
                );
                pre.set_reference(&r_rgb);
                Some(pre)
            } else {
                None
            };

            let r_for_zensim: Vec<u8> = if let Some(ref pre) = mode_b_pre {
                pre.apply_to_ref(&r_rgb)
            } else {
                r_rgb.clone()
            };

            let basename = ref_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();

            // Distortions: parallel over rayon thread pool. Each
            // worker decodes + preprocesses + zensim::compute.
            dists
                .par_iter()
                .filter_map(|(orig_idx, dist_path)| {
                    let (d_rgb, w_d, h_d) = decode_image(dist_path).ok()?;
                    if w_d != w_r || h_d != h_r {
                        eprintln!("  dim mismatch {dist_path:?}: ref={w_r}x{h_r} dist={w_d}x{h_d}");
                        return None;
                    }
                    let d_for_zensim: Vec<u8> = if let Some(ref pre) = mode_b_pre {
                        pre.apply_to_dist(&d_rgb)
                    } else {
                        d_rgb
                    };
                    // RgbSlice expects &[[u8; 3]]. Bytemuck-style reslice.
                    let r_chunks: &[[u8; 3]] =
                        bytemuck::cast_slice(&r_for_zensim[..r_for_zensim.len() / 3 * 3]);
                    let d_chunks: &[[u8; 3]] =
                        bytemuck::cast_slice(&d_for_zensim[..d_for_zensim.len() / 3 * 3]);
                    let r_src = RgbSlice::new(r_chunks, w_r as usize, h_r as usize);
                    let d_src = RgbSlice::new(d_chunks, w_d as usize, h_d as usize);
                    let z = Zensim::new(ZensimProfile::PreviewV0_1)
                        .with_parallel(false); // outer rayon already parallel
                    let result = z.compute_all_features(&r_src, &d_src).ok()?;
                    let feats = result.into_features();
                    let count = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if count % 100 == 0 {
                        eprintln!("  [{count}/{total_pairs}]");
                    }
                    Some((*orig_idx, basename.clone(), feats))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    eprintln!("[cpu] {} pairs computed", results.len());

    // Reassemble in original TSV order.
    let mut sorted = results;
    sorted.sort_by_key(|(idx, _, _)| *idx);
    let mut basenames = Vec::with_capacity(sorted.len());
    let mut feature_cols: Vec<Vec<Option<f64>>> = (0..N_FEATS).map(|_| Vec::with_capacity(sorted.len())).collect();
    for (_, base, feats) in sorted {
        basenames.push(base);
        for i in 0..N_FEATS {
            feature_cols[i].push(feats.get(i).copied());
        }
    }

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(N_FEATS + 1);
    arrays.push(Arc::new(StringArray::from(basenames)));
    for col in feature_cols {
        arrays.push(Arc::new(Float64Array::from(col)));
    }
    let batch = RecordBatch::try_new(schema, arrays)?;
    writer.write(&batch)?;
    writer.close()?;
    eprintln!("[cpu] wrote {out:?}");
    Ok(())
}

fn decode_image(path: &Path) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Ok((rgb.into_raw(), w, h))
}
