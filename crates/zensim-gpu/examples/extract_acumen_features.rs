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
use zensim_gpu::{AcumenArch, Backend, ZensimFeatureRegime, ZensimOpaque, ZensimParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pairs_tsv: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut acumen_mode_a = false;
    let mut acumen_ppd = 56.0_f32;
    let mut acumen_peak = 100.0_f32;
    let mut acumen_ambient = 5.0_f32;
    let mut acumen_arch = AcumenArch::HfPost;
    // Mode B hyperparameters — defaults match the initial winning
    // run. The CLI exposes them so hyperparameter sweeps don't
    // require a rebuild.
    let mut mode_b_blur_sigma: usize = 8;
    // Default band_idx=3 matches the sweep winner from 2026-05-21 — lower
    // spatial frequencies give stronger Mode B signal. Original Path B
    // run used band_idx=2 (0.7328 CID22); band_idx=3 lifts to 0.7543.
    let mut mode_b_band_idx: u32 = 3;
    let mut mode_b_clamp_lo: f32 = 0.1;
    let mut mode_b_clamp_hi: f32 = 4.0;
    // Feature regime: Basic (228), Extended (300), WithIw (372).
    // Default = WithIw = production schema; pass --regime basic to
    // reproduce pre-2026-05-22 228-col outputs.
    let mut regime = ZensimFeatureRegime::WithIw;

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
                    "mode_b" | "mode-b" | "modeb" => AcumenArch::ModeB,
                    "mode_b_per_band" | "mode-b-per-band" | "perband" =>
                        AcumenArch::ModeBPerBand,
                    _ => return Err(format!("unknown --acumen-arch: {v}").into()),
                };
            }
            "--mode-b-blur-sigma" => mode_b_blur_sigma = args.next().unwrap().parse()?,
            "--mode-b-band-idx" => mode_b_band_idx = args.next().unwrap().parse()?,
            "--mode-b-clamp-lo" => mode_b_clamp_lo = args.next().unwrap().parse()?,
            "--mode-b-clamp-hi" => mode_b_clamp_hi = args.next().unwrap().parse()?,
            "--regime" => {
                let v = args.next().unwrap();
                regime = match v.as_str() {
                    "basic" => ZensimFeatureRegime::Basic,
                    "extended" | "ext" => ZensimFeatureRegime::Extended,
                    "with_iw" | "with-iw" | "withiw" | "372" => ZensimFeatureRegime::WithIw,
                    _ => return Err(format!("unknown --regime: {v}").into()),
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
    let base_features = regime.total_features();
    let n_features_total = base_features + n_aux;
    eprintln!(
        "[extract_acumen_features] regime: {regime:?} → {base_features} features + {n_aux} aux"
    );
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
    // Cache the Mode B preprocessor + last ref path so we reuse the
    // ref-side weight map across distortion variants of the same
    // reference. ~50% wall-time savings on safesyn (typically ~80
    // distortions per ref).
    let lut_bytes: &[u8] =
        include_bytes!("../data/castle_csf_v0_5_4_cvvdp.lut");
    let lut = zensim::acumen::castle_csf::CastleCsfLut::from_bytes(lut_bytes)
        .map_err(|e| format!("LUT: {e:?}"))?;
    let mode_b_cfg = zensim::acumen::mode_b::ModeBConfig {
        blur_sigma: mode_b_blur_sigma,
        band_idx: mode_b_band_idx,
        clamp_lo: mode_b_clamp_lo,
        clamp_hi: mode_b_clamp_hi,
    };
    let mut last_ref_path: Option<PathBuf> = None;
    let mut last_ref_dims: Option<(u32, u32)> = None;
    let mut mode_b_pre: Option<zensim::acumen::mode_b::ModeBPreprocessor> = None;
    let mut cached_ref_premul: Option<Vec<u8>> = None;

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
            let mut params = ZensimParams::new()
                .with_acumen_arch(acumen_arch)
                .with_regime(regime);
            if let Some(v) = viewing {
                params = params.with_acumen_mode_a(v);
            }
            current_z = Some(ZensimOpaque::new(backend, w_r, h_r, params)?);
            current_dims = Some((w_r, h_r));
        }
        let z = current_z.as_mut().unwrap();
        // Detect ref change (compares path + dims).
        let ref_changed = last_ref_path.as_ref() != Some(ref_path)
            || last_ref_dims != Some((w_r, h_r));

        // Mode B: pre-multiply BOTH ref and dist by the REFERENCE's
        // per-pixel achromatic CSF weight map. Cache the ref-side
        // pre-multiplied bytes across same-ref pairs.
        let d_used;
        if matches!(acumen_arch, AcumenArch::ModeB) && acumen_mode_a {
            let v = viewing.unwrap();
            if ref_changed {
                let mut pre = zensim::acumen::mode_b::ModeBPreprocessor::new(
                    &lut, v, w_r, h_r, mode_b_cfg,
                );
                pre.set_reference(&r);
                cached_ref_premul = Some(pre.apply_to_ref(&r));
                mode_b_pre = Some(pre);
            }
            let pre = mode_b_pre.as_ref().expect("preprocessor set above");
            d_used = pre.apply_to_dist(&d);
        } else {
            d_used = d.clone();
        }
        // Set the GPU reference ONCE per ref change. Subsequent
        // distortion pairs reuse the cached reference pyramid —
        // saves N-1 ref uploads (1MB each) and N-1 ref-pyramid
        // kernel launches for ~80 distortions per ref on safesyn.
        if ref_changed {
            let ref_for_gpu = if matches!(acumen_arch, AcumenArch::ModeB) && acumen_mode_a {
                cached_ref_premul.as_ref().unwrap().clone()
            } else {
                r.clone()
            };
            if let Err(e) = z.set_reference(&ref_for_gpu) {
                eprintln!("  set_reference failed [{}/{}]: {e:?}", idx + 1, pairs.len());
                fail += 1;
                continue;
            }
            last_ref_path = Some(ref_path.clone());
            last_ref_dims = Some((w_r, h_r));
        }
        let feats = match z.compute_with_reference_vec(&d_used) {
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
        for i in 0..base_features {
            feature_cols[i].push(Some(feats[i]));
        }
        // Aux columns: 12 castleCSF weights cached in ZensimOpaque
        // for the current reference. AuxFeatures arch only.
        if n_aux > 0 {
            let weights = z
                .acumen_band_weights_flat()
                .ok_or("acumen-mode-a + AuxFeatures but weights unavailable")?;
            for i in 0..12 {
                feature_cols[base_features + i].push(Some(weights[i] as f64));
            }
        }
        ok += 1;
        if (idx + 1) % 100 == 0 {
            eprintln!("  {}/{} ok={ok} fail={fail}", idx + 1, pairs.len());
        }
    }
    eprintln!("[extract_acumen_features] ok={ok} fail={fail} total={}", pairs.len());

    // Write parquet
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(n_features_total + 1);
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

// NOTE: Mode B preprocessing lives in `zensim::acumen::mode_b`.
// The example imports it inline via `use ... mode_b::{...}`. Keep
// a single source of truth for the algorithm so CPU and GPU
// pipelines stay bit-exact.
