//! Video-path parity gate: `cvvdp::VideoScorer` / `score_video` vs the
//! committed pycvvdp v0.5.7 video goldens
//! (`scripts/cvvdp_goldens/video_goldens.json`, built by
//! `build_video_goldens.py` against the emitted PNG frames in
//! `/scratch/cvvdpvideo/`).
//!
//! The goldens JSON is committed (≈26 KB) so this test is fully
//! offline and needs no feature gate — it compiles and runs under
//! plain `cargo test -p cvvdp-conformance`.
//!
//! Two checks:
//!
//! 1. **JOD parity** — every (situation × display) cell within
//!    [`cvvdp_conformance::TOLERANCE_JOD`] (1e-3).
//! 2. **`Q_per_ch` stage parity** — for the two dumped fixtures,
//!    the per-(channel, frame, band) pooled differences must track
//!    pycvvdp's `stats["Q_per_ch"]` closely (diagnostic gate; the JOD
//!    gate is the hard one).

use std::collections::BTreeMap;

use cvvdp::params::{DisplayGeometry, DisplayModel};
use cvvdp::{CvvdpParams, VideoScorer};
use cvvdp_conformance::{TOLERANCE_JOD, all_video_situations};

const GOLDENS: &str = include_str!("../../../scripts/cvvdp_goldens/video_goldens.json");
/// Same cells scored with `temp_padding="symmetric"` (built by the
/// same script, `--temp-padding symmetric`).
const GOLDENS_SYMMETRIC: &str =
    include_str!("../../../scripts/cvvdp_goldens/video_goldens_symmetric.json");
/// u16 corpus + the committed real HDR10 clip, built with `--u16`.
const GOLDENS_U16: &str = include_str!("../../../scripts/cvvdp_goldens/video_goldens_u16.json");

/// Directory of the committed real HDR10 fixture (manifest
/// `real_clips` entry `vid16_hdr10_sky_real`).
const REAL_CLIP_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/hdr10_sky_192x108");

/// Decode a directory of RGB16 `f*.png` frames via zenpng into
/// `Vec<Vec<u16>>` interleaved — the same bytes the `--u16` golden
/// builder scored.
fn load_png16_clip(dir: &str) -> Vec<Vec<u16>> {
    use enough::Unstoppable;
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {dir}: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no PNG16 frames in {dir}");
    paths
        .iter()
        .map(|p| {
            let data = std::fs::read(p).expect("read png");
            let out = zenpng::decode(&data, &zenpng::PngDecodeConfig::default(), &Unstoppable)
                .unwrap_or_else(|e| panic!("decode {}: {e}", p.display()));
            assert_eq!(out.info.bit_depth, 16, "{} not 16-bit", p.display());
            let img = out
                .pixels
                .try_as_imgref::<rgb::Rgb<u16>>()
                .unwrap_or_else(|| panic!("{} not RGB16", p.display()));
            let (w, h, stride, buf) = (img.width(), img.height(), img.stride(), img.buf());
            let mut v = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                for px in &buf[y * stride..y * stride + w] {
                    v.extend_from_slice(&[px.r, px.g, px.b]);
                }
            }
            v
        })
        .collect()
}

/// Resolve a u16 clip for a goldens cell: the committed real HDR10
/// fixture, or a regenerated synthetic situation by name.
fn clip16_for(sit_name: &str) -> (Vec<Vec<u16>>, Vec<Vec<u16>>, u32, u32, f32) {
    if sit_name == "vid16_hdr10_sky_real" {
        let r = load_png16_clip(&format!("{REAL_CLIP_DIR}/ref"));
        let d = load_png16_clip(&format!("{REAL_CLIP_DIR}/dist_q8"));
        return (r, d, 96, 54, 30.0);
    }
    let s = cvvdp_conformance::all_video_situations_u16()
        .into_iter()
        .find(|s| s.name == sit_name)
        .unwrap_or_else(|| panic!("u16 situation {sit_name} not in registry"));
    (s.ref_frames, s.dist_frames, s.width, s.height, s.fps)
}

/// JOD parity for the u16/HDR input path — synthetic u16 situations
/// plus the real HDR10 clip, across SDR + PQ + HLG displays.
#[test]
fn video_jod_parity_u16() {
    let goldens: serde_json::Value =
        serde_json::from_str(GOLDENS_U16).expect("video_goldens_u16.json must parse");
    assert_eq!(goldens["bit_depth"].as_u64(), Some(16));
    let cells = goldens["cells"].as_object().expect("goldens .cells");
    let ref_version = goldens["reference_version"].as_str().unwrap_or("unknown");

    let mut max_delta = 0.0f64;
    let mut sum_delta = 0.0f64;
    let mut n = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (key, cell) in cells {
        let (sit_name, disp) = key.split_once('|').expect("cell key <sit>|<disp>");
        let jod_ref = cell["jod_ref"].as_f64().expect("cell jod_ref");
        let (params, geometry) = cell_params(disp);
        let (ref_frames, dist_frames, w, h, fps) = clip16_for(sit_name);
        let stats = cvvdp::score_video_u16_with_stats(
            &ref_frames,
            &dist_frames,
            w,
            h,
            fps,
            params,
            geometry,
            cvvdp::FrameLayout::Interleaved,
            cvvdp::TempPadding::Replicate,
        )
        .unwrap_or_else(|e| panic!("score_video_u16 {key}: {e:?}"));
        let delta = (f64::from(stats.jod) - jod_ref).abs();
        max_delta = max_delta.max(delta);
        sum_delta += delta;
        n += 1;
        if delta > TOLERANCE_JOD {
            failures.push(format!(
                "{key}: ref={jod_ref:.6} got={:.6} |delta|={delta:.6}",
                f64::from(stats.jod)
            ));
        }
    }

    eprintln!("=== cvvdp u16 video parity (pycvvdp {ref_version}) ===");
    eprintln!(
        "cells: {n}  max |Δ| = {max_delta:.6}  mean |Δ| = {:.6}",
        sum_delta / n as f64
    );
    assert!(
        failures.is_empty(),
        "{} u16 video cell(s) exceed {TOLERANCE_JOD:.0e} JOD:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Still-path parity for u16 input — `Cvvdp::score_u16` on the
/// frame-0 pair of every u16 clip, vs pycvvdp `predict(..., "HWC")`.
#[test]
fn still_jod_parity_u16() {
    let goldens: serde_json::Value =
        serde_json::from_str(GOLDENS_U16).expect("video_goldens_u16.json must parse");
    let cells = goldens["still_cells"]
        .as_object()
        .expect("goldens .still_cells");
    assert!(!cells.is_empty(), "expected still_cells in u16 goldens");

    let mut max_delta = 0.0f64;
    let mut sum_delta = 0.0f64;
    let mut n = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (key, cell) in cells {
        let (sit_name, disp) = key.split_once('|').expect("cell key <sit>|<disp>");
        let jod_ref = cell["jod_ref"].as_f64().expect("cell jod_ref");
        let (params, geometry) = cell_params(disp);
        let (ref_frames, dist_frames, w, h, _) = clip16_for(sit_name);
        let mut metric =
            cvvdp::Cvvdp::with_geometry(w, h, params, geometry).expect("Cvvdp::with_geometry");
        let jod = metric
            .score_u16(&ref_frames[0], &dist_frames[0])
            .unwrap_or_else(|e| panic!("score_u16 {key}: {e:?}"));
        let delta = (f64::from(jod) - jod_ref).abs();
        max_delta = max_delta.max(delta);
        sum_delta += delta;
        n += 1;
        if delta > TOLERANCE_JOD {
            failures.push(format!(
                "{key}: ref={jod_ref:.6} got={:.6} |delta|={delta:.6}",
                f64::from(jod)
            ));
        }
    }

    eprintln!("=== cvvdp u16 still parity ===");
    eprintln!(
        "cells: {n}  max |Δ| = {max_delta:.6}  mean |Δ| = {:.6}",
        sum_delta / n as f64
    );
    assert!(
        failures.is_empty(),
        "{} u16 still cell(s) exceed {TOLERANCE_JOD:.0e} JOD:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// f32 input must equal u16 input normalized by 65535 — pycvvdp
/// applies the same rule (`float32` taken as display-encoded
/// directly), so `score_video_f32(v/65535)` is bit-identical to
/// `score_video_u16(v)` when both feed `Eotf::forward` the same f32.
#[test]
fn video_f32_matches_u16_normalized() {
    let s = cvvdp_conformance::all_video_situations_u16()
        .into_iter()
        .find(|s| s.name == "vid16_static_texture_64")
        .expect("situation");
    for disp in ["standard_4k", "standard_hdr_pq", "standard_hdr_hlg"] {
        let (params, geometry) = cell_params(disp);
        let jod16 = cvvdp::score_video_u16(
            &s.ref_frames,
            &s.dist_frames,
            s.width,
            s.height,
            s.fps,
            params,
            geometry,
        )
        .expect("u16");
        let to_f32 = |frames: &[Vec<u16>]| -> Vec<Vec<f32>> {
            frames
                .iter()
                .map(|f| f.iter().map(|&v| f32::from(v) / 65535.0).collect())
                .collect()
        };
        let jodf = cvvdp::score_video_f32(
            &to_f32(&s.ref_frames),
            &to_f32(&s.dist_frames),
            s.width,
            s.height,
            s.fps,
            params,
            geometry,
        )
        .expect("f32");
        assert_eq!(
            jod16.to_bits(),
            jodf.to_bits(),
            "{disp}: u16 {jod16} vs f32 {jodf} must be bit-identical"
        );
    }
}

/// The u16 path must resolve differences *below* one u8 code — proof
/// the low bits actually reach the metric rather than being silently
/// truncated to 8-bit inside the pipeline. `Q_per_ch` is the probe:
/// a JOD of exactly 10.0 would be legitimate for a sub-JND delta, but
/// a truncated pipeline would emit identically-zero Q.
#[test]
fn u16_resolves_sub8bit_differences() {
    let s = cvvdp_conformance::all_video_situations_u16()
        .into_iter()
        .find(|s| s.name == "vid16_static_texture_64")
        .expect("situation");
    // Round-to-nearest u8 quantization, matching pycvvdp's uint8 path.
    let q8 = |v: u16| -> u8 { ((u32::from(v) * 255 + 32767) / 65535) as u8 };
    // Snap every sample to its u8 bin center (b*257 — exactly what
    // u8→u16 expansion yields), then perturb by ±120 while staying
    // inside the bin interior (b*257 ± 128, clamped for bin 255).
    let snap = |f: &[u16]| -> Vec<u16> {
        f.iter()
            .map(|&v| u16::from(q8(v)).saturating_mul(257))
            .collect()
    };
    let bump = |f: &[u16]| -> Vec<u16> {
        f.iter()
            .map(|&v| {
                if u32::from(v) + 120 <= 65535 {
                    v + 120
                } else {
                    v - 120
                }
            })
            .collect()
    };
    let ref16: Vec<Vec<u16>> = s.ref_frames.iter().map(|f| snap(f)).collect();
    let dist16: Vec<Vec<u16>> = ref16.iter().map(|f| bump(f)).collect();
    // Sanity: quantized u8 views are identical — the u8 API literally
    // cannot express this distortion.
    for (r, d) in ref16.iter().zip(dist16.iter()) {
        assert!(
            r.iter().zip(d.iter()).all(|(&a, &b)| q8(a) == q8(b)),
            "delta must stay inside u8 bins"
        );
    }

    let (params, geometry) = cell_params("standard_4k");
    let stats = cvvdp::score_video_u16_with_stats(
        &ref16,
        &dist16,
        s.width,
        s.height,
        s.fps,
        params,
        geometry,
        cvvdp::FrameLayout::Interleaved,
        cvvdp::TempPadding::Replicate,
    )
    .expect("score_video_u16_with_stats");
    let max_q = stats
        .q_per_ch
        .iter()
        .flat_map(|f| f.iter())
        .flat_map(|b| b.iter())
        .fold(0.0f32, |m, &q| m.max(q));
    assert!(
        max_q > 0.0,
        "sub-u8 delta produced all-zero Q — u16 low bits not consumed"
    );
}

/// `Planar` layout for u16 input must produce the same score as the
/// same samples `Interleaved`.
#[test]
fn u16_planar_matches_interleaved() {
    let s = cvvdp_conformance::all_video_situations_u16()
        .into_iter()
        .find(|s| s.name == "vid16_short_clip_odd")
        .expect("situation");
    let to_planar = |f: &[u16]| -> Vec<u16> {
        let n = f.len() / 3;
        let mut p = vec![0u16; f.len()];
        for i in 0..n {
            p[i] = f[i * 3];
            p[n + i] = f[i * 3 + 1];
            p[2 * n + i] = f[i * 3 + 2];
        }
        p
    };
    let ref_p: Vec<Vec<u16>> = s.ref_frames.iter().map(|f| to_planar(f)).collect();
    let dist_p: Vec<Vec<u16>> = s.dist_frames.iter().map(|f| to_planar(f)).collect();
    let (params, geometry) = cell_params("standard_hdr_pq");
    let ji = cvvdp::score_video_u16(
        &s.ref_frames,
        &s.dist_frames,
        s.width,
        s.height,
        s.fps,
        params,
        geometry,
    )
    .expect("interleaved");
    let stats_p = cvvdp::score_video_u16_with_stats(
        &ref_p,
        &dist_p,
        s.width,
        s.height,
        s.fps,
        params,
        geometry,
        cvvdp::FrameLayout::Planar,
        cvvdp::TempPadding::Replicate,
    )
    .expect("planar");
    assert_eq!(
        ji.to_bits(),
        stats_p.jod.to_bits(),
        "planar vs interleaved u16 must be bit-identical"
    );
}

/// A scorer that already consumed one sample type must reject pushes
/// of any other — mixing u8/u16/f32 windows would silently corrupt
/// the temporal ring.
#[test]
fn video_mixed_sample_types_rejected() {
    let (params, geometry) = cell_params("standard_4k");
    let (w, h) = (64u32, 64u32);
    let n = (w * h * 3) as usize;
    let u8f = vec![128u8; n];
    let u16f = vec![32768u16; n];
    let f32f = vec![0.5f32; n];

    // u8 first, then u16 / f32 — both rejected.
    let mut v = VideoScorer::new(w, h, 30.0, params, geometry).expect("scorer");
    v.push_frame(&u8f, &u8f).expect("u8 push");
    assert!(matches!(
        v.push_frame_u16(&u16f, &u16f),
        Err(cvvdp::Error::MixedSampleTypes)
    ));
    assert!(matches!(
        v.push_frame_f32(&f32f, &f32f),
        Err(cvvdp::Error::MixedSampleTypes)
    ));

    // u16 first, then u8 / f32.
    let mut v = VideoScorer::new(w, h, 30.0, params, geometry).expect("scorer");
    v.push_frame_u16(&u16f, &u16f).expect("u16 push");
    assert!(matches!(
        v.push_frame(&u8f, &u8f),
        Err(cvvdp::Error::MixedSampleTypes)
    ));
    assert!(matches!(
        v.push_frame_f32(&f32f, &f32f),
        Err(cvvdp::Error::MixedSampleTypes)
    ));

    // f32 first, then u8 / u16.
    let mut v = VideoScorer::new(w, h, 30.0, params, geometry).expect("scorer");
    v.push_frame_f32(&f32f, &f32f).expect("f32 push");
    assert!(matches!(
        v.push_frame(&u8f, &u8f),
        Err(cvvdp::Error::MixedSampleTypes)
    ));
    assert!(matches!(
        v.push_frame_u16(&u16f, &u16f),
        Err(cvvdp::Error::MixedSampleTypes)
    ));
}

fn cell_params(display_name: &str) -> (CvvdpParams, DisplayGeometry) {
    let display = DisplayModel::by_name(display_name)
        .unwrap_or_else(|| panic!("display {display_name} not in by_name registry"));
    let geometry = DisplayGeometry::by_name(display_name)
        .unwrap_or_else(|| panic!("geometry {display_name} not in by_name registry"));
    (
        CvvdpParams {
            display,
            ..Default::default()
        },
        geometry,
    )
}

#[test]
fn video_jod_parity_all_cells() {
    let goldens: serde_json::Value =
        serde_json::from_str(GOLDENS).expect("video_goldens.json must parse");
    let cells = goldens["cells"].as_object().expect("goldens .cells");
    let displays: Vec<&str> = goldens["displays"]
        .as_array()
        .expect("goldens .displays")
        .iter()
        .map(|d| d.as_str().unwrap())
        .collect();
    let ref_version = goldens["reference_version"].as_str().unwrap_or("unknown");

    let situations = all_video_situations();
    assert!(
        cells.len() >= situations.len() * displays.len(),
        "goldens cover {} cells, need {} x {}",
        cells.len(),
        situations.len(),
        displays.len()
    );

    let mut per_display: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
    let mut max_delta = 0.0f64;
    let mut sum_delta = 0.0f64;
    let mut n = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for s in &situations {
        for &disp in &displays {
            let key = format!("{}|{}", s.name, disp);
            let cell = cells
                .get(&key)
                .unwrap_or_else(|| panic!("missing golden cell {key}"));
            let jod_ref = cell["jod_ref"].as_f64().expect("cell jod_ref");

            let (params, geometry) = cell_params(disp);
            let jod = cvvdp::score_video(
                &s.ref_frames,
                &s.dist_frames,
                s.width,
                s.height,
                s.fps,
                params,
                geometry,
            )
            .unwrap_or_else(|e| panic!("score_video {key}: {e:?}"));

            let delta = (f64::from(jod) - jod_ref).abs();
            let entry = per_display.entry(disp.to_string()).or_insert((0.0, 0.0, 0));
            entry.0 = entry.0.max(delta);
            entry.1 += delta;
            entry.2 += 1;
            max_delta = max_delta.max(delta);
            sum_delta += delta;
            n += 1;
            if delta > TOLERANCE_JOD {
                failures.push(format!(
                    "{key}: ref={jod_ref:.6} got={:.6} |delta|={delta:.6}",
                    f64::from(jod)
                ));
            }
        }
    }

    eprintln!("=== cvvdp video parity (pycvvdp {ref_version}) ===");
    eprintln!(
        "cells: {n}  max |Δ| = {max_delta:.6}  mean |Δ| = {:.6}",
        sum_delta / n as f64
    );
    for (disp, (mx, sum, cnt)) in &per_display {
        eprintln!(
            "  {disp}: max |Δ| = {mx:.6}  mean |Δ| = {:.6}  (n={cnt})",
            sum / *cnt as f64
        );
    }
    assert!(
        failures.is_empty(),
        "{} video cell(s) exceed {TOLERANCE_JOD:.0e} JOD:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Per-stage parity: `Q_per_ch[channel][frame][band]` from pycvvdp
/// `stats` vs our `[frame][band][channel]` table.
#[test]
fn video_q_per_ch_stage_dumps() {
    run_stage_dumps(GOLDENS, cvvdp::TempPadding::Replicate);
}

/// Same JOD gate under `temp_padding="symmetric"` — includes the
/// `vid_short_clip_odd` (5 frames < fl=9) ping-pong path.
#[test]
fn video_jod_parity_symmetric() {
    let goldens: serde_json::Value =
        serde_json::from_str(GOLDENS_SYMMETRIC).expect("symmetric goldens must parse");
    assert_eq!(
        goldens["temp_padding"].as_str(),
        Some("symmetric"),
        "symmetric goldens file must record temp_padding"
    );
    let cells = goldens["cells"].as_object().expect("goldens .cells");
    let ref_version = goldens["reference_version"].as_str().unwrap_or("unknown");

    let situations = all_video_situations();
    let mut max_delta = 0.0f64;
    let mut sum_delta = 0.0f64;
    let mut n = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for s in &situations {
        for (key, cell) in cells {
            let (sit_name, disp) = key.split_once('|').expect("cell key <sit>|<disp>");
            if sit_name != s.name {
                continue;
            }
            let jod_ref = cell["jod_ref"].as_f64().expect("cell jod_ref");
            let (params, geometry) = cell_params(disp);
            let stats = cvvdp::score_video_with_stats(
                &s.ref_frames,
                &s.dist_frames,
                s.width,
                s.height,
                s.fps,
                params,
                geometry,
                cvvdp::FrameLayout::Interleaved,
                cvvdp::TempPadding::Symmetric,
            )
            .unwrap_or_else(|e| panic!("score_video_with_stats symmetric {key}: {e:?}"));
            assert_eq!(
                stats.q_per_ch.len(),
                s.ref_frames.len(),
                "{key}: symmetric must emit N output rows"
            );
            let delta = (f64::from(stats.jod) - jod_ref).abs();
            max_delta = max_delta.max(delta);
            sum_delta += delta;
            n += 1;
            if delta > TOLERANCE_JOD {
                failures.push(format!(
                    "{key}: ref={jod_ref:.6} got={:.6} |delta|={delta:.6}",
                    f64::from(stats.jod)
                ));
            }
        }
    }

    eprintln!("=== cvvdp video parity, symmetric padding (pycvvdp {ref_version}) ===");
    eprintln!(
        "cells: {n}  max |Δ| = {max_delta:.6}  mean |Δ| = {:.6}",
        sum_delta / n as f64
    );
    assert!(
        failures.is_empty(),
        "{} symmetric-padding cell(s) exceed {TOLERANCE_JOD:.0e} JOD:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// `Q_per_ch` stage dumps for the symmetric goldens (adds the
/// `vid_short_clip_odd` ping-pong fixture).
#[test]
fn video_q_per_ch_stage_dumps_symmetric() {
    run_stage_dumps(GOLDENS_SYMMETRIC, cvvdp::TempPadding::Symmetric);
}

fn run_stage_dumps(goldens_json: &str, padding: cvvdp::TempPadding) {
    let goldens: serde_json::Value =
        serde_json::from_str(goldens_json).expect("video goldens must parse");
    let dumps = goldens["stage_dumps"]
        .as_object()
        .expect("goldens .stage_dumps");
    assert!(!dumps.is_empty(), "expected at least one stage dump");

    let situations = all_video_situations();
    for (key, dump) in dumps {
        let (sit_name, disp) = key.split_once('|').expect("dump key <sit>|<disp>");
        let s = situations
            .iter()
            .find(|s| s.name == sit_name)
            .unwrap_or_else(|| panic!("dump situation {sit_name} not in registry"));

        let shape: Vec<usize> = dump["q_per_ch_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(shape.len(), 4, "{key}: expected [B,ch,F,bands]");
        let (n_ch, n_f, n_b) = (shape[1], shape[2], shape[3]);
        assert_eq!(n_ch, 4, "{key}: expected 4 channels");
        assert_eq!(n_f, s.ref_frames.len(), "{key}: frame count");
        let expected: Vec<f64> = dump["q_per_ch"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(expected.len(), n_ch * n_f * n_b, "{key} dump length");

        let (params, geometry) = cell_params(disp);
        let mut v = VideoScorer::with_layout_and_padding(
            s.width,
            s.height,
            s.fps,
            params,
            geometry,
            cvvdp::FrameLayout::Interleaved,
            padding,
        )
        .unwrap_or_else(|e| panic!("VideoScorer {key}: {e:?}"));
        for (rf, df) in s.ref_frames.iter().zip(s.dist_frames.iter()) {
            v.push_frame(rf, df)
                .unwrap_or_else(|e| panic!("push {key}: {e:?}"));
        }
        // Symmetric defers emissions (the lookahead drain happens in
        // `finish_with_stats`); replicate emits on every push, so the
        // live `q_per_ch_table` is already complete.
        let stats = v.finish_with_stats();
        let stats = stats.unwrap_or_else(|e| panic!("finish {key}: {e:?}"));
        let q = &stats.q_per_ch;
        assert_eq!(q.len(), n_f, "{key}: emitted frames");
        assert_eq!(q[0].len(), n_b, "{key}: bands");

        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut cnt = 0usize;
        for c in 0..n_ch {
            for f in 0..n_f {
                for b in 0..n_b {
                    let e = expected[(c * n_f + f) * n_b + b];
                    let g = f64::from(q[f][b][c]);
                    let d = (g - e).abs();
                    max_abs = max_abs.max(d);
                    sum_abs += d;
                    cnt += 1;
                    // Diagnostic gate — 2% relative or 5e-3 absolute.
                    // The hard gate is the JOD parity test.
                    let tol = (e.abs() * 0.02).max(5e-3);
                    assert!(
                        d < tol,
                        "{key} Q_per_ch[ch={c}][f={f}][b={b}]: {g} vs {e} (|Δ|={d})"
                    );
                }
            }
        }
        eprintln!(
            "{key}: Q_per_ch max |Δ| = {max_abs:.6}  mean |Δ| = {:.6}  (n={cnt})",
            sum_abs / cnt as f64
        );
    }
}
