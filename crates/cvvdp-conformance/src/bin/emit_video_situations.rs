//! Emit the conformance *video* corpus to disk as PNG frame
//! sequences + a manifest, for the pycvvdp golden builder to score.
//!
//! Usage:
//!   cargo run -p cvvdp-conformance --bin emit_video_situations -- <out_dir>
//!
//! Writes:
//!   <out_dir>/frames/<situation>/ref/f%04d.png
//!   <out_dir>/frames/<situation>/dist/f%04d.png
//!   <out_dir>/video_manifest.json   { situations: [...], displays: [...] }
//!
//! The manifest carries `fps` + `n_frames` per situation so the
//! Python builder (`scripts/cvvdp_goldens/build_video_goldens.py`)
//! can call `predict(dist, ref, "FHWC", frames_per_second)` per
//! (situation, display) cell. Frames are lossless PNG — what pycvvdp
//! scores is exactly what the Rust harness regenerates in-process.

use std::fs;
use std::path::PathBuf;

use cvvdp_conformance::{PYCVVDP_REFERENCE_VERSION, all_video_situations};

fn save_png(path: &PathBuf, rgb: &[u8], w: u32, h: u32) {
    use image::{ImageBuffer, Rgb};
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(w, h, rgb.to_vec()).expect("rgb buffer");
    img.save(path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = PathBuf::from(args.get(1).map_or_else(
        || "/scratch/cvvdpvideo".to_string(),
        std::clone::Clone::clone,
    ));
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
