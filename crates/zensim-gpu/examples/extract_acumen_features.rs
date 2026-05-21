//! Extract 228-feature parquets via zensim-gpu, optionally with
//! castleCSF Mode A modulation (acumen).
//!
//! Output schema matches the trainer's expected shape:
//!   ref_basename: utf8
//!   human_score:  f64 (passed through from --human-score-column if
//!                 the input TSV has one; else 0.0)
//!   f0..f227:     f64 (zensim-gpu compute_features_srgb_u8 output)
//!
//! Usage:
//!   cargo run --release -p zensim-gpu --example extract_acumen_features \
//!     --features cuda -- \
//!     --pairs-tsv /tmp/cid22_pairs.tsv \
//!     --out /tmp/cid22_acumen_features.parquet \
//!     --acumen-mode-a
//!
//! Assumes per-pair dimensions vary; rebuilds the ZensimOpaque
//! whenever (W, H) changes (similar to how score-pairs handles
//! cubecl's per-(W,H) JIT cache).

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use std::env;
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use zensim_gpu::{AcumenArch, Backend, TOTAL_FEATURES, ZensimOpaque, ZensimParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pairs_tsv: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut acumen_mode_a = false;
    let mut acumen_ppd = 56.0_f32;
    let mut acumen_peak = 100.0_f32;
    let mut acumen_ambient = 5.0_f32;
    let mut acumen_arch = AcumenArch::HfPost;

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
                    "hf_post" | "hf-post" => AcumenArch::HfPost,
                    "wide_modulation" | "wide" => AcumenArch::WideModulation,
                    "aux_features" | "aux" => AcumenArch::AuxFeatures,
                    _ => return Err(format!("unknown --acumen-arch: {v}").into()),
                };
            }
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
    eprintln!("[extract_acumen_features] {} pairs from {pairs_tsv:?}", pairs.len());
    eprintln!("[extract_acumen_features] acumen-mode-a: {acumen_mode_a}, arch: {acumen_arch:?}");
    if acumen_mode_a {
        eprintln!(
            "[extract_acumen_features] viewing: ppd={acumen_ppd} peak={acumen_peak} ambient={acumen_ambient}"
        );
    }

    let backend = pick_backend();
    eprintln!("[extract_acumen_features] backend: {:?}", backend);

    let viewing = if acumen_mode_a {
        Some(zensim_gpu::ViewingCondition::new(
            acumen_ppd,
            acumen_peak,
            acumen_ambient,
        ))
    } else {
        None
    };

    // Output schema. Append 12 aux CSF weight columns when
    // arch == AuxFeatures (so the trainer sees f228..f239 as
    // additional per-pair context).
    let n_aux = if matches!(acumen_arch, AcumenArch::AuxFeatures) && acumen_mode_a {
        12usize
    } else {
        0
    };
    let n_features_total = TOTAL_FEATURES + n_aux;
    let mut fields = vec![Field::new("ref_basename", DataType::Utf8, false)];
    for i in 0..n_features_total {
        fields.push(Field::new(&format!("f{i}"), DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));
    let writer_props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let out_file = File::create(&out)?;
    let mut writer = ArrowWriter::try_new(out_file, schema.clone(), Some(writer_props))?;

    // Process in batches of (W, H) to amortize ZensimOpaque init.
    // Group pairs by ref-image dimensions so we only rebuild the
    // ZensimOpaque when dims change.
    let mut current_dims: Option<(u32, u32)> = None;
    let mut current_z: Option<ZensimOpaque> = None;
    let mut basenames = Vec::with_capacity(pairs.len());
    let mut feature_cols: Vec<Vec<Option<f64>>> =
        (0..n_features_total).map(|_| Vec::with_capacity(pairs.len())).collect();

    let mut ok = 0usize;
    let mut fail = 0usize;
    for (idx, (ref_path, dist_path)) in pairs.iter().enumerate() {
        let (r, w_r, h_r) = match decode_image(ref_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  decode ref failed [{}/{}]: {e}", idx + 1, pairs.len());
                fail += 1;
                continue;
            }
        };
        let (d, w_d, h_d) = match decode_image(dist_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  decode dist failed [{}/{}]: {e}", idx + 1, pairs.len());
                fail += 1;
                continue;
            }
        };
        if w_r != w_d || h_r != h_d {
            eprintln!(
                "  dim mismatch [{}/{}]: ref={}x{} dist={}x{}",
                idx + 1,
                pairs.len(),
                w_r,
                h_r,
                w_d,
                h_d
            );
            fail += 1;
            continue;
        }
        if current_dims != Some((w_r, h_r)) {
            // Rebuild ZensimOpaque for new dimensions.
            let mut params = ZensimParams::new().with_acumen_arch(acumen_arch);
            if let Some(v) = viewing {
                params = params.with_acumen_mode_a(v);
            }
            current_z = Some(ZensimOpaque::new(backend, w_r, h_r, params)?);
            current_dims = Some((w_r, h_r));
        }
        let z = current_z.as_mut().unwrap();
        let feats = match z.compute_features_srgb_u8(&r, &d) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  compute_features failed [{}/{}]: {e:?}", idx + 1, pairs.len());
                fail += 1;
                continue;
            }
        };
        let basename = ref_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        basenames.push(basename);
        for i in 0..TOTAL_FEATURES {
            feature_cols[i].push(Some(feats[i]));
        }
        // Aux columns: 12 castleCSF weights cached in ZensimOpaque
        // for the current reference. AuxFeatures arch only.
        if n_aux > 0 {
            let weights = z
                .acumen_band_weights_flat()
                .ok_or("acumen-mode-a + AuxFeatures but weights unavailable")?;
            for i in 0..12 {
                feature_cols[TOTAL_FEATURES + i].push(Some(weights[i] as f64));
            }
        }
        ok += 1;
        if (idx + 1) % 100 == 0 {
            eprintln!("  {}/{} ok={ok} fail={fail}", idx + 1, pairs.len());
        }
    }
    eprintln!("[extract_acumen_features] ok={ok} fail={fail} total={}", pairs.len());

    // Write parquet
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(TOTAL_FEATURES + 1);
    arrays.push(Arc::new(StringArray::from(basenames)));
    for col in feature_cols {
        arrays.push(Arc::new(Float64Array::from(col)));
    }
    let batch = RecordBatch::try_new(schema, arrays)?;
    writer.write(&batch)?;
    writer.close()?;
    eprintln!("[extract_acumen_features] wrote {out:?}");
    Ok(())
}

fn pick_backend() -> Backend {
    #[cfg(feature = "cuda")]
    {
        return Backend::Cuda;
    }
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        return Backend::Wgpu;
    }
    #[cfg(all(not(feature = "cuda"), not(feature = "wgpu")))]
    compile_error!("extract_acumen_features needs cuda or wgpu feature")
}

fn decode_image(path: &Path) -> Result<(Vec<u8>, u32, u32), Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Ok((rgb.into_raw(), w, h))
}
