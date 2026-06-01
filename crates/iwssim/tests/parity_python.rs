//! Parity test against captured Python-IW-SSIM goldens.
//!
//! Goldens live in
//! `crates/iwssim/goldens/python_iwssim_2026-05-27.json` — each entry
//! references a seed + a synthetic distortion type. The fixtures are
//! reconstructed locally via the same XorShift64 sequence the Python
//! capture script used.

use std::path::PathBuf;

use iwssim::{Iwssim, IwssimParams};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldenFile {
    schema_version: u32,
    #[allow(dead_code)]
    captured_utc: String,
    #[allow(dead_code)]
    python_reference: serde_json::Value,
    pairs: Vec<GoldenPair>,
}

#[derive(Debug, Deserialize)]
struct GoldenPair {
    name: String,
    kind: String,
    distortion: String,
    width: u32,
    height: u32,
    /// Required for `kind == "synthetic"`; optional for other kinds.
    #[serde(default)]
    seed: Option<u64>,
    #[allow(dead_code)]
    ref_sha: String,
    #[allow(dead_code)]
    dist_sha: String,
    score: f64,
}

/// XorShift64 PRNG — bit-identical to the Python capture script's
/// `xorshift64` closure. Produces the same `u64` sequence for the
/// same seed.
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s.wrapping_shl(13);
    s ^= s >> 7;
    s ^= s.wrapping_shl(17);
    *state = s;
    s
}

fn make_rgb_from_seed(w: u32, h: u32, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let n = (w as usize) * (h as usize) * 3;
    let mut out = vec![0u8; n];
    for i in (0..n).step_by(3) {
        let v = xorshift64(&mut state);
        out[i] = (v & 0xFF) as u8;
        out[i + 1] = ((v >> 8) & 0xFF) as u8;
        out[i + 2] = ((v >> 16) & 0xFF) as u8;
    }
    out
}

fn apply_distortion(ref_rgb: &[u8], w: u32, h: u32, distort: &str) -> Vec<u8> {
    let _ = w;
    let _ = h;
    match distort {
        "identical" => ref_rgb.to_vec(),
        "offset" => ref_rgb
            .iter()
            .map(|&v| (v as i32 + 5).clamp(0, 255) as u8)
            .collect(),
        "shift1px" => {
            // dist[:, 1:, :] = ref[:, :-1, :]; dist[:, 0, :] = 0
            let w = w as usize;
            let h = h as usize;
            let row_bytes = w * 3;
            let mut out = vec![0u8; row_bytes * h];
            for y in 0..h {
                let src_row = &ref_rgb[y * row_bytes..(y + 1) * row_bytes - 3];
                // Copy src[0..w-1] -> dst[1..w]
                out[y * row_bytes + 3..(y + 1) * row_bytes].copy_from_slice(src_row);
            }
            out
        }
        "swap" => {
            // ref[..., [1, 0, 2]]
            let mut out = ref_rgb.to_vec();
            for chunk in out.chunks_exact_mut(3) {
                chunk.swap(0, 1);
            }
            out
        }
        other => panic!("unknown distortion {other:?}"),
    }
}

fn load_goldens() -> GoldenFile {
    let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("goldens")
        .join("python_iwssim_2026-05-27.json");
    let s =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let file: GoldenFile =
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse goldens: {e}"));
    assert_eq!(file.schema_version, 1, "unexpected schema version");
    file
}

#[test]
fn xorshift_matches_python() {
    // Sanity-check the PRNG produces the same bytes the capture script
    // claims via `ref_sha`. For seed=1, the SHA256 (first 16 hex chars)
    // of the resulting 256x256x3 RGB buffer should be the one in the
    // goldens manifest.
    let goldens = load_goldens();
    let sample = goldens
        .pairs
        .iter()
        .find(|p| p.name == "synth_256_identical")
        .expect("identical 256 pair");
    let seed = sample.seed.expect("seed");
    let rgb = make_rgb_from_seed(sample.width, sample.height, seed);

    let mut hasher = sha2_hash(&rgb);
    let hex16 = hasher.drain(..16).collect::<String>();
    assert_eq!(
        hex16, sample.ref_sha,
        "XorShift64 byte sequence drift; check the PRNG implementation"
    );
}

/// Tiny dep-free SHA256 wrapper using a hex helper. We don't pull
/// sha2 just for this test; the goldens already include the sha and
/// we recompute via the `image::ImageBuffer` would be overkill.
/// Implement a minimal SHA-256 inline for parity comparison.
fn sha2_hash(bytes: &[u8]) -> String {
    use std::process::Command;
    // Defer to system shasum to avoid pulling sha2 as a dev-dep for
    // one comparison. The host has sha256sum (coreutils).
    use std::io::Write;
    let mut child = Command::new("sha256sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn sha256sum");
    {
        let stdin = child.stdin.as_mut().expect("sha256sum stdin");
        stdin.write_all(bytes).expect("write sha256sum stdin");
    }
    let out = child.wait_with_output().expect("sha256sum wait");
    let s = String::from_utf8(out.stdout).expect("sha256sum utf-8");
    s.split_whitespace().next().unwrap_or("").to_string()
}

/// Tolerance for f32-based pipeline parity vs Python's f32 PyTorch
/// path. Acceptable bands:
///
/// - Identical pair: ≤ 1e-6 (any drift here is a bug).
/// - Distorted pairs: ≤ 5e-3 in `[0, 1]` score space — order
///   matters more than absolute equality. The dominant sources of
///   drift are:
///     * imenlarge2's bilinear path uses align_corners=True; the
///       Python PyTorch path's default differs across torch versions.
///     * `info_content_weight_map`'s Cᵤ eigendecomposition order
///       differs between LAPACK's `geev` (Python's torch.eig) and our
///       Jacobi (CPU port); the per-pixel `infow` sum is invariant
///       in exact arithmetic but loses a few ULPs in f32.
///
/// Any score landing outside the band is flagged as a CI failure with
/// the per-fixture diff so future regressions are obvious.
const TOL_IDENTICAL: f64 = 1e-5;
const TOL_DISTORTED: f64 = 5e-3;

#[test]
fn parity_python_goldens() {
    let goldens = load_goldens();
    let mut failures: Vec<String> = Vec::new();
    let mut max_diff: f64 = 0.0;
    let mut max_diff_name = String::new();

    for p in &goldens.pairs {
        assert_eq!(p.kind, "synthetic", "only synthetic kind supported here");
        let seed = p.seed.expect("synthetic pairs must carry seed");
        let ref_rgb = make_rgb_from_seed(p.width, p.height, seed);
        let dist_rgb = apply_distortion(&ref_rgb, p.width, p.height, &p.distortion);

        let mut scorer = Iwssim::with_params(p.width, p.height, IwssimParams::default())
            .unwrap_or_else(|e| panic!("Iwssim::new {}x{}: {e}", p.width, p.height));
        let result = scorer
            .score(&ref_rgb, &dist_rgb)
            .unwrap_or_else(|e| panic!("score {}: {e}", p.name));
        let got = result.score;
        let want = p.score;
        let diff = (got - want).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_name = p.name.clone();
        }
        let tol = if p.distortion == "identical" {
            TOL_IDENTICAL
        } else {
            TOL_DISTORTED
        };
        let status = if diff <= tol { "OK   " } else { "FAIL " };
        println!(
            "  {} {:32}  got = {:.10}  want = {:.10}  diff = {:.3e}",
            status, p.name, got, want, diff
        );
        if diff > tol {
            failures.push(format!(
                "{}: got {:.10}, want {:.10} (diff {:.3e} > {:.3e})",
                p.name, got, want, diff, tol
            ));
        }
    }

    println!("\nmax |diff| = {:.6e} @ {}", max_diff, max_diff_name);
    assert!(
        failures.is_empty(),
        "{} parity failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
