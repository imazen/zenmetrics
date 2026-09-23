//! Emit the conformance *video* corpus to disk as PNG frame
//! sequences + a manifest, for the pycvvdp golden builder to score.
//!
//! Usage:
//!   cargo run -p cvvdp-conformance --bin emit_video_situations -- <out_dir>
//!   cargo run -p cvvdp-conformance --bin emit_video_situations -- <out_dir> --u16
//!
//! Writes:
//!   <out_dir>/frames/<situation>/ref/f%04d.png     (8-bit PNG, u8 corpus)
//!   <out_dir>/frames/<situation>/dist/f%04d.png
//!   <out_dir>/video_manifest.json   { situations: [...], displays: [...] }
//!
//! With `--u16`: the 16-bit corpus emits RGB16 PNGs to
//! `frames16/<situation>/` and `video_manifest_u16.json`, plus a
//! `real_clips` section pointing at the committed HDR10 fixture in
//! `data/` (the golden builder resolves those repo-relative dirs).
//!
//! The manifest carries `fps` + `n_frames` per situation so the
//! Python builder (`scripts/cvvdp_goldens/build_video_goldens.py`)
//! can call `predict(dist, ref, "FHWC", frames_per_second)` per
//! (situation, display) cell. Frames are lossless PNG — what pycvvdp
//! scores is exactly what the Rust harness regenerates in-process.

use std::fs;
use std::path::{Path, PathBuf};

use cvvdp_conformance::{PYCVVDP_REFERENCE_VERSION, all_video_situations};
use enough::Unstoppable;
use imgref::ImgVec;
use rgb::Rgb;

fn save_png(path: &Path, rgb8: &[u8], w: u32, h: u32) {
    let pixels: Vec<Rgb<u8>> = rgb8
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    let img = ImgVec::new(pixels, w as usize, h as usize);
    let bytes = zenpng::encode_rgb8(
        img.as_ref(),
        None,
        &zenpng::EncodeConfig::default(),
        &Unstoppable,
        &Unstoppable,
    )
    .unwrap_or_else(|e| panic!("encode {}: {e}", path.display()));
    fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn save_png16(path: &Path, rgb16: &[u16], w: u32, h: u32) {
    let pixels: Vec<Rgb<u16>> = rgb16
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    let img = ImgVec::new(pixels, w as usize, h as usize);
    let bytes = zenpng::encode_rgb16(
        img.as_ref(),
        None,
        &zenpng::EncodeConfig::default(),
        &Unstoppable,
        &Unstoppable,
    )
    .unwrap_or_else(|e| panic!("encode {}: {e}", path.display()));
    fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let u16_mode = args.iter().any(|a| a == "--u16");
    let out_dir = PathBuf::from(args.get(1).map_or_else(
        || "/scratch/cvvdpvideo".to_string(),
        std::clone::Clone::clone,
    ));

    if u16_mode {
        emit_u16(&out_dir);
    } else {
        emit_u8(&out_dir);
    }
}

fn emit_u8(out_dir: &Path) {
    let frames_dir = out_dir.join("frames");

    let situations = all_video_situations();

    // The display list mirrors the video-parity gate in the work
    // order: the three SDR conformance displays + one HDR preset.
    let displays = [
        "standard_4k",
        "standard_fhd",
        "standard_phone",
        "standard_hdr_pq",
    ];

    let mut sit_entries = Vec::new();
    for s in &situations {
        let ref_dir = frames_dir.join(s.name).join("ref");
        let dist_dir = frames_dir.join(s.name).join("dist");
        fs::create_dir_all(&ref_dir).expect("mkdir ref");
        fs::create_dir_all(&dist_dir).expect("mkdir dist");
        for (k, f) in s.ref_frames.iter().enumerate() {
            save_png(&ref_dir.join(format!("f{k:04}.png")), f, s.width, s.height);
        }
        for (k, f) in s.dist_frames.iter().enumerate() {
            save_png(&dist_dir.join(format!("f{k:04}.png")), f, s.width, s.height);
        }
        sit_entries.push(serde_json::json!({
            "name": s.name,
            "class": s.class.as_str(),
            "width": s.width,
            "height": s.height,
            "fps": s.fps,
            "n_frames": s.ref_frames.len(),
            "ref_dir": format!("frames/{}/ref", s.name),
            "dist_dir": format!("frames/{}/dist", s.name),
        }));
    }

    let manifest = serde_json::json!({
        "reference": "gfxdisp/ColorVideoVDP",
        "reference_version": PYCVVDP_REFERENCE_VERSION,
        "kind": "video",
        "bit_depth": 8,
        "situations": sit_entries,
        "displays": displays,
        "cells": situations.len() * displays.len(),
    });

    let manifest_path = out_dir.join("video_manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .expect("write manifest");

    eprintln!(
        "wrote {} video situations x {} displays = {} cells to {}",
        situations.len(),
        displays.len(),
        situations.len() * displays.len(),
        out_dir.display()
    );
    eprintln!("manifest: {}", manifest_path.display());
}

fn emit_u16(out_dir: &Path) {
    let frames_dir = out_dir.join("frames16");

    let situations = cvvdp_conformance::all_video_situations_u16();

    // u16 cells cover one SDR display plus both HDR EOTFs — the PQ
    // and HLG paths have materially different input handling (HLG's
    // RGB-dependent OOTF), so both are gated.
    let displays = ["standard_4k", "standard_hdr_pq", "standard_hdr_hlg"];

    let mut sit_entries = Vec::new();
    for s in &situations {
        let ref_dir = frames_dir.join(s.name).join("ref");
        let dist_dir = frames_dir.join(s.name).join("dist");
        fs::create_dir_all(&ref_dir).expect("mkdir ref");
        fs::create_dir_all(&dist_dir).expect("mkdir dist");
        for (k, f) in s.ref_frames.iter().enumerate() {
            save_png16(&ref_dir.join(format!("f{k:04}.png")), f, s.width, s.height);
        }
        for (k, f) in s.dist_frames.iter().enumerate() {
            save_png16(&dist_dir.join(format!("f{k:04}.png")), f, s.width, s.height);
        }
        sit_entries.push(serde_json::json!({
            "name": s.name,
            "class": s.class.as_str(),
            "width": s.width,
            "height": s.height,
            "fps": s.fps,
            "n_frames": s.ref_frames.len(),
            "ref_dir": format!("frames16/{}/ref", s.name),
            "dist_dir": format!("frames16/{}/dist", s.name),
        }));
    }

    // Real-content fixture: the committed HDR10 clip under data/,
    // repo-relative so the golden builder can resolve it regardless
    // of the situations output dir. Ref frames are the source clip's
    // decoded+converted RGB16 (PQ/BT.2020, full-range); dist is the
    // 8-bit-roundtrip variant — a realistic banding distortion.
    let real_clips = serde_json::json!([{
        "name": "vid16_hdr10_sky_real",
        "class": "real_content",
        "width": 96,
        "height": 54,
        "fps": 30.0,
        "n_frames": 24,
        "ref_dir": "crates/cvvdp-conformance/data/hdr10_sky_192x108/ref",
        "dist_dir": "crates/cvvdp-conformance/data/hdr10_sky_192x108/dist_q8",
        "repo_relative": true,
        "displays": ["standard_hdr_pq"],
        "provenance": "JonaNorman/HDRSample hdr-pq-sky.mp4 (BT.2020 primaries, ST 2084 PQ, yuv420p10le, 30 fps); ffmpeg hevc decode -> rgb48le (BT.2020 NCL matrix, PQ-encoded samples), crop=96:54:760:80, 24 frames @30fps from t=2.0s; sample peak ~1030 nits; dist_q8 = per-sample 8-bit roundtrip (v/257 round *257)",
    }]);

    let manifest = serde_json::json!({
        "reference": "gfxdisp/ColorVideoVDP",
        "reference_version": PYCVVDP_REFERENCE_VERSION,
        "kind": "video",
        "bit_depth": 16,
        "sample_encoding": "display-encoded u16, normalized v/65535 (pycvvdp video_source_array semantics)",
        "situations": sit_entries,
        "real_clips": real_clips,
        "displays": displays,
        "cells": situations.len() * displays.len() + 1,
    });

    let manifest_path = out_dir.join("video_manifest_u16.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .expect("write manifest");

    eprintln!(
        "wrote {} u16 video situations x {} displays + 1 real clip to {}",
        situations.len(),
        displays.len(),
        out_dir.display()
    );
    eprintln!("manifest: {}", manifest_path.display());
}
