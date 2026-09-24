//! Score the UPIQ HDR subset (380 absolute-luminance EXR pairs) with
//! [`hdrvdp::score`] and emit a TSV joined against the dataset's own
//! `HDRVDP2_2` column and JOD truth — the chunk-4 validation harness
//! (`imazen/zenmetrics#50`).
//!
//! `upiq_objective_scores.csv` carries the official per-condition
//! HDR-VDP-2.2 score — correlating our column against it is a per-image
//! implementation-parity check on real content (stronger than the pooled
//! SROCC-vs-JOD, which saturates at the JOD noise ceiling).
//!
//! **Reproduction protocol** (measured 2026-09-24, see
//! `benchmarks/hdrvdp_upiq_2026-09-24.md`): the released `HDRVDP2_2` column
//! was generated at a **fixed `pix_per_deg = 30`** on full-resolution
//! luminance — NOT at the subjective CSV's `pix_per_deg` column (~56.5 /
//! ~60.3, which lands ~12 points low). `HDRVDP_PPD=30 ... luminance`
//! reproduces the official scores to SROCC 0.996 / delta +0.05±0.36.
//! `HDRVDP_PPD_SCALE` (multiplies the CSV ppd), `HDRVDP_IMG_SCALE` and
//! `HDRVDP_SURROUND=geom` remain as the probe knobs that falsified the
//! alternatives; none is needed for the canonical run.
//!
//! ```bash
//! HDRVDP_PPD=30 cargo run -p hdrvdp --release \
//!     --example upiq_score -- <upiq_dir> <out.tsv> luminance
//! ```
//!
//! `<upiq_dir>` is the extracted `upiq_dataset/` (with `images/` and the two
//! CSVs). `n` optionally limits the row count for protocol probes. Output
//! columns: `condition_id test_file reference_file official_hdrvdp2_2 jod
//! hdrvdp` — `scripts/hdr/upiq_corr.py` reads the last column as the metric.

use std::collections::HashMap;
use std::io::Write;

use hdrvdp::{ColorEncoding, Params, score};

struct Condition {
    id: String,
    test: String,
    reference: String,
    ppd: f64,
    jod: f64,
    official: f64,
}

fn load_conditions(upiq_dir: &str) -> Vec<Condition> {
    let subj = std::fs::read_to_string(format!("{upiq_dir}/upiq_subjective_scores.csv"))
        .expect("read subjective csv");
    let obj = std::fs::read_to_string(format!("{upiq_dir}/upiq_objective_scores.csv"))
        .expect("read objective csv");

    let mut official: HashMap<String, f64> = HashMap::new();
    for line in obj.lines().skip(1) {
        let mut it = line.split(',');
        if let (Some(id), Some(_ds), Some(_pie), Some(_psnr), Some(_ssim), Some(hdrvdp)) = (
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
        ) && let Ok(v) = hdrvdp.parse()
        {
            official.insert(id.to_string(), v);
        }
    }

    let mut out = Vec::new();
    for line in subj.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        // condition_id,dataset,content_id,dist_id,dist_level,is_hdr,
        // test_file,reference_file,pix_per_deg,distortion_name,
        // repeating_content_id,JOD
        if f.len() < 12 || f[5] != "1" {
            continue;
        }
        out.push(Condition {
            id: f[0].to_string(),
            test: f[6].to_string(),
            reference: f[7].to_string(),
            ppd: f[8].parse().expect("pix_per_deg"),
            jod: f[11].parse().expect("JOD"),
            official: *official.get(f[0]).unwrap_or(&f64::NAN),
        });
    }
    out
}

/// Absolute-luminance linear RGB (cd/m²), interleaved `f32`.
fn load_exr(path: &str) -> (Vec<f32>, usize, usize) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .to_rgb32f();
    let (w, h) = (img.width() as usize, img.height() as usize);
    (img.into_raw(), w, h)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let upiq_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/mnt/v/datasets/upiq_extracted/upiq_dataset".into());
    let out_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/upiq_hdrvdp.tsv".into());
    let encoding = match args.get(3).map(String::as_str).unwrap_or("luminance") {
        "luminance" => ColorEncoding::Luminance,
        "rgb-bt.709" => ColorEncoding::RgbBt709,
        other => panic!("unknown encoding: {other}"),
    };
    let limit: usize = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    // Protocol probe knob: scales the CSV's `pix_per_deg` (e.g. 0.5 to test a
    // double viewing distance hypothesis). 1.0 for the nominal run.
    let ppd_scale: f64 = std::env::var("HDRVDP_PPD_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    // `surround_l` override: "geom" uses the reference's geometric mean
    // (upstream's surround_l=-1); unset keeps the 1e-5 default.
    let surround_geom = std::env::var("HDRVDP_SURROUND").is_ok_and(|s| s == "geom");
    // Protocol probe: downsample images by this factor before scoring.
    // FALSIFIED for UPIQ reproduction — half-res lands SROCC 0.944 vs
    // official where full-res lands 0.996; kept only for future probes.
    let ds_scale: f64 = std::env::var("HDRVDP_IMG_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    // Skip the first N conditions (probe disjoint subsets without editing).
    let skip: usize = std::env::var("HDRVDP_SKIP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // Absolute ppd override — UPIQ's released scores were generated at a
    // fixed 30 pix/deg for both HDR datasets (measured: per-image agreement
    // with the released HDRVDP2_2 column is SROCC 0.996 / delta +0.05±0.36
    // there, while the CSV's own `pix_per_deg` geometry is ~12 points off).
    let ppd_abs: Option<f64> = std::env::var("HDRVDP_PPD")
        .ok()
        .and_then(|s| s.parse().ok());

    let conds = load_conditions(&upiq_dir);
    eprintln!("{} HDR conditions", conds.len());

    // One output row per selected condition, filled by index — workers pull
    // indices from an atomic counter, so the result does not depend on
    // thread count or scheduling.
    let selected: Vec<&Condition> = conds.iter().skip(skip).take(limit).collect();
    let rows: Vec<std::sync::Mutex<Option<String>>> = selected
        .iter()
        .map(|_| std::sync::Mutex::new(None))
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::env::var("HDRVDP_JOBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        })
        .clamp(1, selected.len().max(1));

    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                // Reference images recur across conditions; test images are
                // used once, so only references are cached (bounded memory
                // per worker).
                let mut ref_cache: HashMap<String, (Vec<f32>, usize, usize)> = HashMap::new();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= selected.len() {
                        break;
                    }
                    let c = selected[i];
                    let mut planes: [Option<(Vec<f64>, usize, usize)>; 2] = [None, None];
                    for (slot, rel) in [(0, &c.reference), (1, &c.test)] {
                        let path = format!("{upiq_dir}/images/{rel}");
                        let decoded;
                        let rgb_ref = if slot == 0 {
                            if !ref_cache.contains_key(&path) {
                                match std::panic::catch_unwind(|| load_exr(&path)) {
                                    Ok(v) => {
                                        ref_cache.insert(path.clone(), v);
                                    }
                                    Err(_) => {
                                        eprintln!("LOAD FAIL {path}");
                                        continue;
                                    }
                                }
                            }
                            &ref_cache[&path]
                        } else {
                            decoded = std::panic::catch_unwind(|| load_exr(&path)).ok();
                            match &decoded {
                                Some(v) => v,
                                None => {
                                    eprintln!("LOAD FAIL {path}");
                                    continue;
                                }
                            }
                        };
                        let (rgb, cw, ch) = rgb_ref;
                        let (rgb, w, h): (Vec<f32>, usize, usize) = if ds_scale != 1.0 {
                            let nw = ((*cw as f64 * ds_scale) as usize).max(8);
                            let nh = ((*ch as f64 * ds_scale) as usize).max(8);
                            let img =
                                image::Rgb32FImage::from_raw(*cw as u32, *ch as u32, rgb.clone())
                                    .expect("img");
                            let sm = image::imageops::resize(
                                &img,
                                nw as u32,
                                nh as u32,
                                image::imageops::FilterType::CatmullRom,
                            );
                            (sm.into_raw(), nw, nh)
                        } else {
                            (rgb.clone(), *cw, *ch)
                        };
                        planes[slot] = Some((
                            match encoding {
                                ColorEncoding::RgbBt709 => {
                                    rgb.iter().map(|&v| f64::from(v)).collect()
                                }
                                _ => {
                                    rgb.as_chunks::<3>()
                                        .0
                                        .iter()
                                        .map(|p| {
                                            f64::from(p[0].mul_add(
                                                0.2126,
                                                p[1].mul_add(0.7152, p[2] * 0.0722),
                                            ))
                                        })
                                        .collect()
                                }
                            },
                            w,
                            h,
                        ));
                    }
                    let row = match (planes[1].take(), planes[0].take()) {
                        (Some((test, tw, th)), Some((reference, rw, rh))) => {
                            if (rw, rh) != (tw, th) {
                                eprintln!("SIZE MISMATCH {}", c.id);
                                None
                            } else {
                                let mut par = Params::new(ppd_abs.unwrap_or(c.ppd * ppd_scale));
                                if surround_geom {
                                    par.surround_l = None;
                                }
                                match score(&test, &reference, rw, rh, encoding, &par) {
                                    Ok(q) => Some(format!(
                                        "{}\t{}\t{}\t{}\t{}\t{}\n",
                                        c.id, c.test, c.reference, c.official, c.jod, q
                                    )),
                                    Err(e) => {
                                        eprintln!("SCORE FAIL {}: {e:?}", c.id);
                                        None
                                    }
                                }
                            }
                        }
                        _ => None,
                    };
                    *rows[i].lock().unwrap() = row;
                }
            });
        }
    });

    let mut out =
        String::from("condition_id\ttest_file\treference_file\tofficial_hdrvdp2_2\tjod\thdrvdp\n");
    let (mut done, mut failed) = (0usize, 0usize);
    for r in &rows {
        match r.lock().unwrap().as_ref() {
            Some(row) => {
                out.push_str(row);
                done += 1;
            }
            None => failed += 1,
        }
    }

    std::fs::File::create(&out_path)
        .and_then(|mut f| f.write_all(out.as_bytes()))
        .expect("write out");
    eprintln!("done: {done} scored, {failed} failed -> {out_path}");
}
