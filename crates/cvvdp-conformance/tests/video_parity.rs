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
    let goldens: serde_json::Value =
        serde_json::from_str(GOLDENS).expect("video_goldens.json must parse");
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
        let mut v = VideoScorer::new(s.width, s.height, s.fps, params, geometry)
            .unwrap_or_else(|e| panic!("VideoScorer::new {key}: {e:?}"));
        for (rf, df) in s.ref_frames.iter().zip(s.dist_frames.iter()) {
            v.push_frame(rf, df)
                .unwrap_or_else(|e| panic!("push {key}: {e:?}"));
        }
        let q = v.q_per_ch_table();
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
