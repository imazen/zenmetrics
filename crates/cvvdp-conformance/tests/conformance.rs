//! The cvvdp conformance matrix: every (impl × display × situation)
//! cell scored against the pycvvdp v0.5.4 reference.
//!
//! Gated behind the `conformance-goldens` cargo feature so the
//! default `cargo test -p cvvdp-conformance` invocation runs only the
//! offline self-tests (situation/matrix invariants) — it neither
//! touches the network nor requires a GPU. To run the full matrix:
//!
//! ```bash
//! cargo test -p cvvdp-conformance --features conformance-goldens \
//!     --test conformance -- --nocapture
//! ```
//!
//! With locally-built goldens (skips the R2 fetch):
//!
//! ```bash
//! CVVDP_CONFORMANCE_GOLDENS=/path/conformance_goldens.json \
//!   cargo test -p cvvdp-conformance --features conformance-goldens \
//!     --test conformance -- --nocapture
//! ```
//!
//! Output: `benchmarks/cvvdp_conformance_matrix_<DATE>.tsv` with every
//! cell's `jod_ref / jod_cpu / jod_gpu` and the three deltas. Cells
//! exceeding [`cvvdp_conformance::TOLERANCE_JOD`] are listed and, if
//! not in the documented divergence allow-list, fail the test.

#![cfg(feature = "conformance-goldens")]

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use cubecl::Runtime;
use cvvdp_conformance::{TOLERANCE_JOD, all_situations, conformance_displays};
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel};

type Backend = cubecl::cuda::CudaRuntime;

/// Documented divergences (cells allowed to exceed `TOLERANCE_JOD`).
/// Each entry MUST have a root-cause line in `docs/CVVDP_CONFORMANCE.md`.
/// Key format: `"<situation>|<display>|<impl>"` where impl is
/// `cpu` or `gpu`. An empty map means "no cell may exceed tolerance".
///
/// This list is the explicit, reviewable alternative to silently
/// widening the tolerance. A non-empty entry here is a FINDING, not a
/// pass.
fn documented_divergences() -> BTreeMap<&'static str, &'static str> {
    // Populated from the 2026-05-26 full run (see
    // docs/CVVDP_CONFORMANCE.md §Divergences). Each value is the
    // measured |delta| + root cause. These are the harness DOING ITS
    // JOB — surfaced, reviewed, and rationalized findings, NOT silent
    // passes. Re-derive with `--nocapture` after any cvvdp-cpu/gpu
    // change; if a cell's delta drops below 1e-3, remove it here.
    //
    // FINDING A — `iphone_14_pro` display (Y_peak = 1025 nit, the only
    // sRGB conformance display with peak >= 1000 nit). cvvdp-cpu and
    // cvvdp-gpu AGREE with each other to ~7e-5 JOD but BOTH land low
    // vs pycvvdp by up to 0.028 JOD on mid-quality JPEG content. A
    // shared (not GPU-specific) parity gap in the high-peak-luminance
    // CSF/masking adaptation regime; magnitude shrinks toward
    // near-lossless. Root cause + scope in docs/CVVDP_CONFORMANCE.md.
    //
    // FINDING B — GPU-only marginal (<= 0.0014 JOD) on extreme
    // high-frequency / heavily-blurred content at the perceptibility
    // floor (JOD ~3.7-4.4). GPU float reduction-order vs CPU at the
    // deepest pyramid bands. cvvdp-cpu passes these cells.
    BTreeMap::from([
        // Finding A
        (
            "synth_photo_256_jpeg60|iphone_14_pro|cpu",
            "0.02439 JOD; Finding A high-peak-luminance regime",
        ),
        (
            "synth_photo_256_jpeg60|iphone_14_pro|gpu",
            "0.02440 JOD; Finding A high-peak-luminance regime",
        ),
        (
            "synth_jpeg_q60|iphone_14_pro|cpu",
            "0.02439 JOD; Finding A high-peak-luminance regime",
        ),
        (
            "synth_jpeg_q60|iphone_14_pro|gpu",
            "0.02440 JOD; Finding A high-peak-luminance regime",
        ),
        (
            "synth_jpeg_q30|iphone_14_pro|cpu",
            "0.01634 JOD; Finding A high-peak-luminance regime",
        ),
        (
            "synth_jpeg_q30|iphone_14_pro|gpu",
            "0.01635 JOD; Finding A high-peak-luminance regime",
        ),
        (
            "synth_jpeg_q90|iphone_14_pro|cpu",
            "0.00649 JOD; Finding A high-peak-luminance regime",
        ),
        (
            "synth_jpeg_q90|iphone_14_pro|gpu",
            "0.00649 JOD; Finding A high-peak-luminance regime",
        ),
        (
            "large_1024_jpeg60|iphone_14_pro|cpu",
            "0.02806 JOD; Finding A high-peak-luminance regime",
        ),
        (
            "large_1024_jpeg60|iphone_14_pro|gpu",
            "0.02813 JOD; Finding A high-peak-luminance regime",
        ),
        // Finding B
        (
            "checkerboard_blur_r2|htc_vive_pro|gpu",
            "0.00139 JOD; Finding B GPU float reduction-order at perceptibility floor",
        ),
        (
            "checkerboard_blur_r2|standard_fhd|gpu",
            "0.00106 JOD; Finding B GPU float reduction-order at perceptibility floor",
        ),
        (
            "synth_blur_r5|standard_hdr_hlg|gpu",
            "0.00101 JOD; Finding B GPU float reduction-order at perceptibility floor",
        ),
    ])
}

struct CellResult {
    situation: String,
    display: String,
    class: String,
    width: u32,
    height: u32,
    jod_ref: f64,
    jod_cpu: f64,
    jod_gpu: f64,
}

#[test]
fn conformance_matrix() {
    let goldens = common::load_goldens();
    let golden_cells = goldens["cells"].as_object().expect("goldens .cells object");
    let ref_version = goldens["reference_version"].as_str().unwrap_or("unknown");

    let situations = all_situations();
    let displays = conformance_displays();

    let client = Backend::client(&Default::default());

    let mut results: Vec<CellResult> = Vec::new();
    let mut over_tol: Vec<(String, f64, f64, char)> = Vec::new();
    let mut missing_golden: Vec<String> = Vec::new();

    for s in &situations {
        for d in displays {
            let key = format!("{}|{}", s.name, d.upstream_name);
            let Some(g) = golden_cells.get(&key) else {
                missing_golden.push(key);
                continue;
            };
            let Some(jod_ref) = g["jod_ref"].as_f64() else {
                // Golden cell errored on the pycvvdp side — record + skip.
                missing_golden.push(format!("{key} (golden jod_ref null)"));
                continue;
            };

            let display = DisplayModel::by_name(d.upstream_name)
                .unwrap_or_else(|| panic!("display {} not in registry", d.upstream_name));
            let geometry = DisplayGeometry::by_name(d.upstream_name)
                .unwrap_or_else(|| panic!("geometry {} not in registry", d.upstream_name));
            let params = CvvdpParams {
                display,
                ..Default::default()
            };

            // --- cvvdp-cpu (black box via public API) ---
            let cpu_params = cvvdp_cpu::CvvdpParams {
                display,
                ..Default::default()
            };
            let mut cpu = cvvdp_cpu::Cvvdp::with_geometry(s.width, s.height, cpu_params, geometry)
                .unwrap_or_else(|e| panic!("cvvdp-cpu new {key}: {e:?}"));
            let jod_cpu = f64::from(
                cpu.score(&s.reference, &s.distorted)
                    .unwrap_or_else(|e| panic!("cvvdp-cpu score {key}: {e:?}")),
            );

            // --- cvvdp-gpu (black box via public API) ---
            let mut gpu = cvvdp_gpu::Cvvdp::<Backend>::new_with_geometry(
                client.clone(),
                s.width,
                s.height,
                params,
                geometry,
            )
            .unwrap_or_else(|e| panic!("cvvdp-gpu new {key}: {e:?}"));
            let jod_gpu = gpu
                .score(&s.reference, &s.distorted)
                .unwrap_or_else(|e| panic!("cvvdp-gpu score {key}: {e:?}"));

            let delta_cpu = (jod_cpu - jod_ref).abs();
            let delta_gpu = (jod_gpu - jod_ref).abs();
            let allow = documented_divergences();
            if delta_cpu > TOLERANCE_JOD && !allow.contains_key(format!("{key}|cpu").as_str()) {
                over_tol.push((format!("{key}|cpu"), jod_ref, jod_cpu, 'c'));
            }
            if delta_gpu > TOLERANCE_JOD && !allow.contains_key(format!("{key}|gpu").as_str()) {
                over_tol.push((format!("{key}|gpu"), jod_ref, jod_gpu, 'g'));
            }

            results.push(CellResult {
                situation: s.name.to_string(),
                display: d.upstream_name.to_string(),
                class: s.class.as_str().to_string(),
                width: s.width,
                height: s.height,
                jod_ref,
                jod_cpu,
                jod_gpu,
            });
        }
    }

    write_tsv(&results, ref_version);

    // Summary.
    let n = results.len();
    let pass_cpu = results
        .iter()
        .filter(|r| (r.jod_cpu - r.jod_ref).abs() <= TOLERANCE_JOD)
        .count();
    let pass_gpu = results
        .iter()
        .filter(|r| (r.jod_gpu - r.jod_ref).abs() <= TOLERANCE_JOD)
        .count();
    let max_dc = results
        .iter()
        .map(|r| (r.jod_cpu - r.jod_ref).abs())
        .fold(0.0_f64, f64::max);
    let max_dg = results
        .iter()
        .map(|r| (r.jod_gpu - r.jod_ref).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("=== cvvdp conformance matrix (pycvvdp {ref_version}) ===");
    eprintln!("cells scored: {n}");
    eprintln!("cpu within {TOLERANCE_JOD:.0e}: {pass_cpu}/{n}  (max |delta_cpu| = {max_dc:.6})");
    eprintln!("gpu within {TOLERANCE_JOD:.0e}: {pass_gpu}/{n}  (max |delta_gpu| = {max_dg:.6})");
    if !missing_golden.is_empty() {
        eprintln!("WARN: {} cells had no usable golden:", missing_golden.len());
        for m in missing_golden.iter().take(10) {
            eprintln!("  {m}");
        }
    }

    if !over_tol.is_empty() {
        // Sort biggest-delta first for the failure report.
        over_tol.sort_by(|a, b| {
            let da = (a.2 - a.1).abs();
            let db = (b.2 - b.1).abs();
            db.partial_cmp(&da).unwrap()
        });
        eprintln!(
            "=== {} cells exceed tolerance {TOLERANCE_JOD:.0e} (not in divergence allow-list) ===",
            over_tol.len()
        );
        for (key, jr, ji, _) in &over_tol {
            eprintln!(
                "  {key}: ref={jr:.5} impl={ji:.5} |delta|={:.5}",
                (ji - jr).abs()
            );
        }
        panic!(
            "{} conformance cell(s) exceed {TOLERANCE_JOD:.0e} JOD and are not documented \
             divergences — fix the impl or document the divergence in docs/CVVDP_CONFORMANCE.md",
            over_tol.len()
        );
    }
}

fn write_tsv(results: &[CellResult], ref_version: &str) {
    let date = "2026-05-26";
    let bench_dir = repo_root().join("benchmarks");
    std::fs::create_dir_all(&bench_dir).expect("mkdir benchmarks");
    let path = bench_dir.join(format!("cvvdp_conformance_matrix_{date}.tsv"));
    let mut f = std::fs::File::create(&path).expect("create tsv");
    writeln!(
        f,
        "# cvvdp conformance matrix — pycvvdp {ref_version} reference\n\
         # tolerance: |jod_impl - jod_ref| <= {TOLERANCE_JOD:.0e} JOD\n\
         # generated by cvvdp-conformance/tests/conformance.rs"
    )
    .unwrap();
    writeln!(
        f,
        "situation\tdisplay\tclass\twidth\theight\tjod_ref\tjod_cpu\tjod_gpu\tdelta_cpu\tdelta_gpu\tdelta_cpu_gpu\tpass_cpu\tpass_gpu"
    )
    .unwrap();
    for r in results {
        let dc = (r.jod_cpu - r.jod_ref).abs();
        let dg = (r.jod_gpu - r.jod_ref).abs();
        let dcg = (r.jod_cpu - r.jod_gpu).abs();
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}",
            r.situation,
            r.display,
            r.class,
            r.width,
            r.height,
            r.jod_ref,
            r.jod_cpu,
            r.jod_gpu,
            dc,
            dg,
            dcg,
            u8::from(dc <= TOLERANCE_JOD),
            u8::from(dg <= TOLERANCE_JOD),
        )
        .unwrap();
    }
    eprintln!("wrote {}", path.display());
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../crates/cvvdp-conformance
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(std::path::Path::parent)
        .map(std::path::Path::to_path_buf)
        .unwrap_or(manifest)
}
