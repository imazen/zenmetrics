//! Emit the conformance situation corpus to disk as PNGs + a
//! manifest, for the pycvvdp golden builder to score.
//!
//! Usage:
//!   cargo run -p cvvdp-conformance --bin emit_situations -- <out_dir>
//!
//! Writes:
//!   <out_dir>/images/<situation>.ref.png
//!   <out_dir>/images/<situation>.dist.png
//!   <out_dir>/manifest.json   { situations: [...], displays: [...] }
//!
//! The manifest cross-products situations × displays so the Python
//! builder knows every cell to score. The Python builder
//! (`scripts/cvvdp_goldens/build_conformance_goldens.py`) scores each
//! (ref.png, dist.png, display_name) cell with pycvvdp v0.5.4 and
//! writes `conformance_goldens.json`.

use std::fs;
use std::path::PathBuf;

use cvvdp_conformance::{PYCVVDP_REFERENCE_VERSION, all_situations, conformance_displays};

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
        || "/tmp/cvvdp_conformance_situations".to_string(),
        std::clone::Clone::clone,
    ));
    let img_dir = out_dir.join("images");
    fs::create_dir_all(&img_dir).expect("mkdir images");

    let situations = all_situations();
    let displays = conformance_displays();

    let mut sit_entries = Vec::new();
    for s in &situations {
        let ref_name = format!("{}.ref.png", s.name);
        let dist_name = format!("{}.dist.png", s.name);
        save_png(&img_dir.join(&ref_name), &s.reference, s.width, s.height);
        save_png(&img_dir.join(&dist_name), &s.distorted, s.width, s.height);
        sit_entries.push(serde_json::json!({
            "name": s.name,
            "class": s.class.as_str(),
            "width": s.width,
            "height": s.height,
            "ref": format!("images/{ref_name}"),
            "dist": format!("images/{dist_name}"),
        }));
    }

    let disp_entries: Vec<_> = displays
        .iter()
        .map(|d| {
            serde_json::json!({
                "upstream_name": d.upstream_name,
                "role": d.role,
            })
        })
        .collect();

    let manifest = serde_json::json!({
        "reference": "gfxdisp/ColorVideoVDP",
        "reference_version": PYCVVDP_REFERENCE_VERSION,
        "situations": sit_entries,
        "displays": disp_entries,
        "cells": situations.len() * displays.len(),
    });

    let manifest_path = out_dir.join("manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .expect("write manifest");

    eprintln!(
        "wrote {} situations x {} displays = {} cells to {}",
        situations.len(),
        displays.len(),
        situations.len() * displays.len(),
        out_dir.display()
    );
    eprintln!("manifest: {}", manifest_path.display());
}
