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

/// Per-crate cache-dir subdirectory name. Lives in `~/.cache/<this>/<GOLDEN_VERSION>/`
/// (or `$XDG_CACHE_HOME/...` / `$TMPDIR/...` per the cache_dir() priority).
/// Tick 581: extracted as a pub const so it can be pinned in
/// `goldens_metadata.rs` rather than duplicated as a magic string
/// in two places.
pub const CACHE_DIR_SUBDIR: &str = "zenmetrics-cvvdp-goldens";

/// Returns the per-version cache directory, creating it if needed.
pub fn cache_dir() -> PathBuf {
    let base = if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(dir)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache")
    } else {
        std::env::temp_dir()
    };
    let dir = base.join(CACHE_DIR_SUBDIR).join(GOLDEN_VERSION);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Fetch `name` (e.g. `"src_vs_q70.final.json"`) into the cache and
/// return the local path. Panics on failure — meant for use in tests
/// where the right behavior is loud failure, not silent skip.
pub fn fetch(name: &str, sha256: &str) -> PathBuf {
    let local = cache_dir().join(name);
    if local.exists() {
        if let Ok(hex) = file_sha256_hex(&local)
            && hex == sha256
        {
            return local;
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
    fs::write(&local, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", local.display()));
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
    for b in &out {
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
        .and_then(serde_json::Value::as_f64)
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

/// Per-band per-channel pycvvdp Q_per_ch values at chroma_shift —
/// the output of the spatial pool (`lp_norm(D, beta=2, dim=spatial,
/// normalize=True)`). Used to localize whether the remaining 0.117
/// JOD drift sits in the spatial pool, band/channel pools, or
/// met2jod. See `dump_q_chroma.py`.
pub struct QSentinel {
    pub q_a: f32,
    pub q_rg: f32,
    pub q_vy: f32,
}

pub fn pycvvdp_q_chroma_shift_band(k: usize) -> QSentinel {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/pycvvdp_q_chroma_shift.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse pycvvdp_q_chroma_shift.json");
    let bands = v["bands"].as_array().expect(".bands missing");
    let b = &bands[k];
    QSentinel {
        q_a: b["q_a"].as_f64().unwrap() as f32,
        q_rg: b["q_rg"].as_f64().unwrap() as f32,
        q_vy: b["q_vy"].as_f64().unwrap() as f32,
    }
}

/// Per-band per-channel pycvvdp raw S (CSF sensitivity,
/// pre-sens_corr) values at chroma_shift sentinels. The struct
/// also carries the per-pixel `log_l_bkg_ref` pycvvdp computed,
/// so a parity test can feed THE SAME log_l_bkg into our
/// `sensitivity_scalar` for an apples-to-apples CSF lookup
/// comparison. See `dump_s_chroma.py`.
pub struct SSentinel {
    pub y0: u32,
    pub x0: u32,
    pub yk: u32,
    pub xk: u32,
    pub log_l_bkg_ref: f32,
    pub s_raw_a: f32,
    pub s_raw_rg: f32,
    pub s_raw_vy: f32,
}

/// Per-band raw S values + per-pixel log_l_bkg_ref for chroma_shift.
/// Index by pyramid level `k`. Embedded via include_str! at compile
/// time.
pub fn pycvvdp_s_chroma_shift_band(k: usize) -> Vec<SSentinel> {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/pycvvdp_s_chroma_shift.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse pycvvdp_s_chroma_shift.json");
    let bands = v["bands"].as_array().expect(".bands missing");
    let samples = bands[k]["samples"]
        .as_array()
        .expect("band samples missing");
    samples
        .iter()
        .map(|s| SSentinel {
            y0: s["y0"].as_u64().unwrap() as u32,
            x0: s["x0"].as_u64().unwrap() as u32,
            yk: s["yk"].as_u64().unwrap() as u32,
            xk: s["xk"].as_u64().unwrap() as u32,
            log_l_bkg_ref: s["log_l_bkg_ref"].as_f64().unwrap() as f32,
            s_raw_a: s["s_raw_a"].as_f64().unwrap() as f32,
            s_raw_rg: s["s_raw_rg"].as_f64().unwrap() as f32,
            s_raw_vy: s["s_raw_vy"].as_f64().unwrap() as f32,
        })
        .collect()
}

/// rho-per-band axis as pycvvdp's `lpyr.band_freqs` reports it on
/// the chroma_shift fixture. Single source of truth for the parity
/// test: same band index → same rho on both sides.
pub fn pycvvdp_s_chroma_shift_rho(k: usize) -> f32 {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/pycvvdp_s_chroma_shift.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse pycvvdp_s_chroma_shift.json");
    let bands = v["bands"].as_array().expect(".bands missing");
    bands[k]["rho"].as_f64().expect("band rho") as f32
}

/// Per-band per-channel pycvvdp D (post-masking, post-PU-blur,
/// pre-pool) values at chroma_shift sentinels. Used to localize
/// whether the 0.117 JOD drift sits in masking-and-earlier vs in
/// the pool / accumulation order. See `dump_d_chroma.py`.
pub struct DSentinel {
    pub y0: u32,
    pub x0: u32,
    pub yk: u32,
    pub xk: u32,
    pub d_a: f32,
    pub d_rg: f32,
    pub d_vy: f32,
}

/// Per-band D values for chroma_shift. Index by pyramid level
/// `k`. Embedded via include_str! at compile time.
pub fn pycvvdp_d_chroma_shift_band(k: usize) -> Vec<DSentinel> {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/pycvvdp_d_chroma_shift.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse pycvvdp_d_chroma_shift.json");
    let bands = v["bands"].as_array().expect(".bands missing");
    let samples = bands[k]["samples"]
        .as_array()
        .expect("band samples missing");
    samples
        .iter()
        .map(|s| DSentinel {
            y0: s["y0"].as_u64().unwrap() as u32,
            x0: s["x0"].as_u64().unwrap() as u32,
            yk: s["yk"].as_u64().unwrap() as u32,
            xk: s["xk"].as_u64().unwrap() as u32,
            d_a: s["d_a"].as_f64().unwrap() as f32,
            d_rg: s["d_rg"].as_f64().unwrap() as f32,
            d_vy: s["d_vy"].as_f64().unwrap() as f32,
        })
        .collect()
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
    let samples = bands[k]["samples"]
        .as_array()
        .expect("band samples missing");
    samples
        .iter()
        .map(|s| TpSentinel {
            y0: s["y0"].as_u64().unwrap() as u32,
            x0: s["x0"].as_u64().unwrap() as u32,
            yk: s["yk"].as_u64().unwrap() as u32,
            xk: s["xk"].as_u64().unwrap() as u32,
            t_p_test_a: s["t_p_test_a"].as_f64().unwrap() as f32,
            t_p_ref_a: s["t_p_ref_a"].as_f64().unwrap() as f32,
            t_p_test_rg: s["t_p_test_rg"].as_f64().unwrap() as f32,
            t_p_ref_rg: s["t_p_ref_rg"].as_f64().unwrap() as f32,
            t_p_test_vy: s["t_p_test_vy"].as_f64().unwrap() as f32,
            t_p_ref_vy: s["t_p_ref_vy"].as_f64().unwrap() as f32,
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
    let samples = bands[k]["samples"]
        .as_array()
        .expect("band samples missing");
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
        if fx.get("q").and_then(serde_json::Value::as_u64) == Some(u64::from(q)) {
            return fx
                .get("jod")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or_else(|| panic!("v1_corpus_jods.json: q={q} has no .jod"))
                as f32;
        }
    }
    panic!("v1_corpus_jods.json: q={q} not found");
}

/// Shared "first available GPU backend" type alias used by every
/// `#[cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]`
/// test file. Prefers cuda, falls back to wgpu, then hip. Tick 270
/// dedup — was hand-mirrored across 6 test files (color_kernel,
/// csf_kernel, masking_kernel, pyramid_kernel, pipeline_color,
/// pipeline_score) plus inline-in-fn copies in shadow_jod /
/// pool_scalar (kept local because they sit inside `fn` / `mod gpu`).
///
/// Not defined when no GPU feature is on — callers that use it are
/// already gated on the same `any(cuda, wgpu, hip)` cfg, so the
/// missing-symbol error if someone forgets the gate surfaces clearly.
#[cfg(feature = "cuda")]
pub type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "hip", not(feature = "cuda"), not(feature = "wgpu")))]
pub type Backend = cubecl::hip::HipRuntime;

/// Open a PNG/JPEG at `path`, decode to RGB8, and return the raw
/// bytes. Asserts the decoded dimensions match the expected
/// `(w, h)` — meant for test fixtures where a mismatch indicates
/// either a corrupted corpus file or a wrong test expectation.
/// Tick 267 dedup — was hand-mirrored across `pipeline_score.rs`
/// and `shadow_jod.rs`.
///
/// Accepts `&Path` so callers can pass either `&PathBuf` (auto-
/// derefs) or `&Path` directly. Tick 268 widened from `&PathBuf`.
pub fn load_rgb_bytes(path: &std::path::Path, w: u32, h: u32) -> Vec<u8> {
    let img = image::ImageReader::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
        .to_rgb8();
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    img.into_raw()
}

/// Deterministic synthetic reference image used by the
/// `synth_*_*` parity fixtures in `pycvvdp_synth_goldens.json`. Bit-
/// stable across pycvvdp's `scripts/cvvdp_goldens/bench_12mp_cuda.py`
/// `synth_pair_ref` and the Rust port — pure modular arithmetic, no
/// PRNG. Tick 255 dedup — was hand-inlined across 14 sites in
/// `tests/pipeline_color.rs` (chroma_shift, blur, noise, 12mp warm-ref,
/// 12mp cold, and stage probes). Callers that need a (ref, dist)
/// fixture pair pass this through their own per-fixture dist builder
/// (saturating_sub / clamp / pseudo-blur / etc.).
pub fn synth_pair_ref(w: usize, h: usize) -> Vec<u8> {
    let n = w * h * 3;
    let mut b = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let r = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            let g = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            let bb = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
            let i = (y * w + x) * 3;
            b[i] = r;
            b[i + 1] = g;
            b[i + 2] = bb;
        }
    }
    b
}

/// Apply the canonical `(-8, -4, +12)` per-channel saturating
/// offset distortion to a ref buffer. Matches the dist
/// construction `bench_12mp_cuda.py::synth_pair_12mp` uses (also
/// `synth_pair_odd_dim`). Tick 278 extracted this as a reusable
/// closure; tick 279 made it a standalone helper so callers can
/// pair it with either `synth_pair_ref` or
/// `synth_pair_odd_dim_ref`.
pub fn apply_offset_dist(ref_bytes: &[u8]) -> Vec<u8> {
    ref_bytes
        .chunks_exact(3)
        .flat_map(|p| {
            [
                p[0].saturating_sub(8),
                p[1].saturating_sub(4),
                p[2].saturating_add(12),
            ]
        })
        .collect()
}

/// Convenience: `(ref, dist)` pair from `synth_pair_ref` + the
/// canonical offset dist. Most call sites that want both halves
/// use this; sites that already hold a ref buffer use
/// `apply_offset_dist` directly. Tick 278.
pub fn synth_pair_with_offset_dist(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let r = synth_pair_ref(w, h);
    let d = apply_offset_dist(&r);
    (r, d)
}

/// Convenience: `(ref, dist)` pair from `synth_pair_odd_dim_ref` +
/// the canonical offset dist — the construction
/// `bench_12mp_cuda.py::synth_pair_odd_dim` uses for the 73×91
/// pycvvdp golden. Tick 280 — pairs with
/// `synth_pair_with_offset_dist` for the alternate ref pattern.
pub fn synth_pair_odd_dim_with_offset_dist(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let r = synth_pair_odd_dim_ref(w, h);
    let d = apply_offset_dist(&r);
    (r, d)
}

/// Alternate deterministic synthetic reference used by the 73×91
/// odd-dim parity fixture and several stage-probe tests. Bit-
/// stable across pycvvdp's `bench_12mp_cuda.py::synth_pair_odd_dim`
/// and the Rust port — modular arithmetic with all-channel x/y
/// linear patterns. Distinct from [`synth_pair_ref`] which uses
/// per-channel mixed coefficients; this one's coarser pattern is
/// easier to read in stage-probe debug dumps.
///
/// Tick 259 dedup — was hand-inlined across 10 sites in
/// `tests/pipeline_color.rs`, plus `tests/cpu_backend.rs` (synth_pair
/// helper) and `examples/manifest_parity_probe.rs` (synth_odd_pair).
pub fn synth_pair_odd_dim_ref(w: usize, h: usize) -> Vec<u8> {
    let n = w * h * 3;
    let mut b = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 8) % 256) as u8;
            let g = ((y * 8) % 256) as u8;
            let bb = (((x + y) * 4) % 256) as u8;
            let i = (y * w + x) * 3;
            b[i] = r;
            b[i + 1] = g;
            b[i + 2] = bb;
        }
    }
    b
}

/// All `q` values present in `scripts/cvvdp_goldens/v1_corpus_jods.json`,
/// sorted ascending. Tick 254 dedup — was `&[1, 5, 20, 45, 70, 90]`
/// hand-mirrored across 5 callers. A future
/// `scripts/cvvdp_goldens/build_goldens.py` rerun + JSON bump that
/// adds (e.g.) `q = 2` now propagates to every manifest-parity test
/// without hand-editing.
pub fn v1_corpus_qs() -> Vec<u32> {
    const MANIFEST_JSON: &str =
        include_str!("../../../../scripts/cvvdp_goldens/v1_corpus_jods.json");
    let v: serde_json::Value =
        serde_json::from_str(MANIFEST_JSON).expect("parse v1_corpus_jods.json");
    let pairs = v
        .get("pairs")
        .and_then(|p| p.as_object())
        .expect("v1_corpus_jods.json missing .pairs");
    let mut qs: Vec<u32> = pairs
        .values()
        .filter_map(|fx| {
            fx.get("q")
                .and_then(serde_json::Value::as_u64)
                .map(|q| q as u32)
        })
        .collect();
    qs.sort_unstable();
    qs.dedup();
    qs
}
