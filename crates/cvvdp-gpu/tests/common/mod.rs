//! Fetch + cache helpers for cvvdp parity goldens.
//!
//! Goldens are produced by `scripts/cvvdp_goldens/build_goldens.py`
//! running the pinned pycvvdp v0.5.4 reference, and uploaded to R2
//! under `s3://coefficient/cvvdp-goldens/<version>/`. The same bucket
//! is exposed publicly at `https://coefficient.r2.imazen.org/`, so
//! tests fetch without credentials. Same pattern as zensim's
//! `zentrain-r2.imazen.org` corpus mirror.
//!
//! Tests using this module must be compiled with the
//! `parity-goldens` feature so the network code path isn't built into
//! the default `cargo test` run.
//!
//! Cache layout: `$XDG_CACHE_HOME/zenmetrics-cvvdp-goldens/<version>/<file>`.

// Each test file that includes this module uses a different subset of
// the helpers — fetch/cache for the R2 v1 manifest path, or the
// embedded JSON loader for synth-fixture goldens. The unused ones in
// any given test compile-unit aren't dead crate-wide.
#![allow(dead_code)]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Pin label for the current cvvdp golden set. Bump in lockstep with
/// the R2 prefix and the pycvvdp version pin in
/// `scripts/cvvdp_goldens/requirements.txt`.
pub const GOLDEN_VERSION: &str = "v1";

/// Public R2 URL for the manifest. The bucket is the same
/// `s3://coefficient/` that the sweep infrastructure uses; its public
/// mirror is configured at `coefficient.r2.imazen.org`.
pub const MANIFEST_URL: &str = "https://coefficient.r2.imazen.org/cvvdp-goldens/v1/manifest.json";

/// sha256 of the manifest, captured at upload time
/// (`2026-05-14`, v0.5.4 reference, zenmetrics-corpus 256×256 q-grid).
/// Bump alongside `GOLDEN_VERSION` when the goldens are regenerated.
pub const MANIFEST_SHA256: &str =
    "9b8638c4cee15b79240acd8116f54d417f2a641999ca0146567b1bea5aa594c5";

/// Returns the per-version cache directory, creating it if needed.
pub fn cache_dir() -> PathBuf {
    let base = if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(dir)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache")
    } else {
        std::env::temp_dir()
    };
    let dir = base.join("zenmetrics-cvvdp-goldens").join(GOLDEN_VERSION);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Fetch `name` (e.g. `"src_vs_q70.final.json"`) into the cache and
/// return the local path. Panics on failure — meant for use in tests
/// where the right behavior is loud failure, not silent skip.
pub fn fetch(name: &str, sha256: &str) -> PathBuf {
    let local = cache_dir().join(name);
    if local.exists() {
        if let Ok(hex) = file_sha256_hex(&local) {
            if hex == sha256 {
                return local;
            }
        }
        // Stale or corrupt — drop it and refetch.
        let _ = fs::remove_file(&local);
    }

    let base = MANIFEST_URL
        .rsplit_once('/')
        .map(|(b, _)| b)
        .expect("MANIFEST_URL must have at least one '/'");
    let url = format!("{base}/{name}");

    let mut resp = ureq::get(&url)
        .call()
        .unwrap_or_else(|e| panic!("GET {url}: {e}"));
    let mut bytes = Vec::new();
    resp.body_mut()
        .as_reader()
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("read {url}: {e}"));

    let got = sha256_hex(&bytes);
    assert_eq!(got, sha256, "sha256 mismatch for {name}");
    fs::write(&local, &bytes).unwrap_or_else(|e| panic!("write {local:?}: {e}"));
    local
}

fn file_sha256_hex(path: &Path) -> std::io::Result<String> {
    let bytes = fs::read(path)?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let out = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in out.iter() {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
    }
    s
}

/// Look up a pycvvdp synth golden JOD value from
/// `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json` (regenerated
/// by `bench_12mp_cuda.py`). The manifest is embedded at compile
/// time via `include_str!` so tests don't need filesystem access
/// at runtime; rebuild after regenerating the manifest to pick up
/// new golden values.
///
/// Panics if the fixture key is missing. Test authors should add
/// the fixture to `bench_12mp_cuda.py` first, regenerate, then
/// reference it here.
pub fn pycvvdp_synth_golden_jod(fixture: &str) -> f32 {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/pycvvdp_synth_goldens.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse pycvvdp_synth_goldens.json");
    let jod = v
        .get("fixtures")
        .and_then(|f| f.get(fixture))
        .and_then(|fx| fx.get("jod"))
        .and_then(|j| j.as_f64())
        .unwrap_or_else(|| {
            panic!(
                "pycvvdp_synth_goldens.json: fixture '{fixture}' not found or .jod is not a number"
            )
        });
    jod as f32
}

/// Look up the v1 corpus 256×256 per-pair JOD golden from
/// `scripts/cvvdp_goldens/v1_corpus_jods.json` (mirrors the
/// `pairs.*.jod` fields from the R2 v1 manifest). Embedded via
/// `include_str!` so tests don't need network or filesystem access.
///
/// `q` is the JPEG quality level (1, 5, 20, 45, 70, or 90). Panics
/// if the q is missing from the manifest.
/// Sentinel-pixel DKL value from
/// `scripts/cvvdp_goldens/pycvvdp_dkl_chroma_shift.json`.
/// Used by `compute_dkl_planes_matches_pycvvdp_dkl_at_chroma_shift_sentinels`
/// to localize where the 0.117 JOD chroma_shift drift starts.
pub struct DklSentinel {
    pub y: u32,
    pub x: u32,
    pub ref_dkl: [f32; 3],
    pub dist_dkl: [f32; 3],
}

/// Per-band per-channel pycvvdp Weber-contrast values at chroma_shift
/// sentinels. Used to localize where the chroma drift sits after the
/// DKL stage matches pycvvdp bit-identical (tick 196).
pub struct WeberSentinel {
    pub y0: u32,
    pub x0: u32,
    pub yk: u32,
    pub xk: u32,
    pub test_a: f32,
    pub ref_a: f32,
    pub test_rg: f32,
    pub ref_rg: f32,
    pub test_vy: f32,
    pub ref_vy: f32,
}

pub struct TpSentinel {
    pub y0: u32,
    pub x0: u32,
    pub yk: u32,
    pub xk: u32,
    pub t_p_test_a: f32,
    pub t_p_ref_a: f32,
    pub t_p_test_rg: f32,
    pub t_p_ref_rg: f32,
    pub t_p_test_vy: f32,
    pub t_p_ref_vy: f32,
}

/// Per-band T_p (post-CSF, pre-masking) values for chroma_shift.
/// T_p = weber · S · ch_gain. Used to localize whether the
/// downstream-of-weber chroma drift sits in the CSF apply.
pub fn pycvvdp_tp_chroma_shift_band(k: usize) -> Vec<TpSentinel> {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/pycvvdp_tp_chroma_shift.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse pycvvdp_tp_chroma_shift.json");
    let bands = v["bands"].as_array().expect(".bands missing");
    let samples = bands[k]["samples"].as_array().expect("band samples missing");
    samples
        .iter()
        .map(|s| TpSentinel {
            y0: s["y0"].as_u64().unwrap() as u32,
            x0: s["x0"].as_u64().unwrap() as u32,
            yk: s["yk"].as_u64().unwrap() as u32,
            xk: s["xk"].as_u64().unwrap() as u32,
            t_p_test_a:  s["t_p_test_a"].as_f64().unwrap() as f32,
            t_p_ref_a:   s["t_p_ref_a"].as_f64().unwrap() as f32,
            t_p_test_rg: s["t_p_test_rg"].as_f64().unwrap() as f32,
            t_p_ref_rg:  s["t_p_ref_rg"].as_f64().unwrap() as f32,
            t_p_test_vy: s["t_p_test_vy"].as_f64().unwrap() as f32,
            t_p_ref_vy:  s["t_p_ref_vy"].as_f64().unwrap() as f32,
        })
        .collect()
}

/// Returns (band_index → list of sentinels) for pycvvdp's Weber
/// pyramid at chroma_shift. Embedded via include_str! at compile time.
pub fn pycvvdp_weber_chroma_shift_band(k: usize) -> Vec<WeberSentinel> {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/pycvvdp_weber_chroma_shift.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse pycvvdp_weber_chroma_shift.json");
    let bands = v["bands"].as_array().expect(".bands missing");
    let samples = bands[k]["samples"].as_array().expect("band samples missing");
    samples
        .iter()
        .map(|s| WeberSentinel {
            y0: s["y0"].as_u64().unwrap() as u32,
            x0: s["x0"].as_u64().unwrap() as u32,
            yk: s["yk"].as_u64().unwrap() as u32,
            xk: s["xk"].as_u64().unwrap() as u32,
            test_a: s["test_A"].as_f64().unwrap() as f32,
            ref_a: s["ref_A"].as_f64().unwrap() as f32,
            test_rg: s["test_RG"].as_f64().unwrap() as f32,
            ref_rg: s["ref_RG"].as_f64().unwrap() as f32,
            test_vy: s["test_VY"].as_f64().unwrap() as f32,
            ref_vy: s["ref_VY"].as_f64().unwrap() as f32,
        })
        .collect()
}

pub fn pycvvdp_dkl_chroma_shift_sentinels() -> Vec<DklSentinel> {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/pycvvdp_dkl_chroma_shift.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse pycvvdp_dkl_chroma_shift.json");
    let sentinels = v
        .get("sentinels")
        .and_then(|s| s.as_array())
        .expect("missing .sentinels");
    sentinels
        .iter()
        .map(|s| {
            let y = s["y"].as_u64().unwrap() as u32;
            let x = s["x"].as_u64().unwrap() as u32;
            let ref_dkl = s["ref_dkl_f32"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            let dist_dkl = s["dist_dkl_f32"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            DklSentinel {
                y,
                x,
                ref_dkl: [ref_dkl[0], ref_dkl[1], ref_dkl[2]],
                dist_dkl: [dist_dkl[0], dist_dkl[1], dist_dkl[2]],
            }
        })
        .collect()
}

pub fn v1_corpus_jod_golden(q: u32) -> f32 {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/v1_corpus_jods.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse v1_corpus_jods.json");
    let pairs = v
        .get("pairs")
        .and_then(|p| p.as_object())
        .expect("v1_corpus_jods.json missing .pairs");
    for (_name, fx) in pairs {
        if fx.get("q").and_then(|n| n.as_u64()) == Some(q as u64) {
            return fx
                .get("jod")
                .and_then(|j| j.as_f64())
                .unwrap_or_else(|| panic!("v1_corpus_jods.json: q={q} has no .jod")) as f32;
        }
    }
    panic!("v1_corpus_jods.json: q={q} not found");
}
