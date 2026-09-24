//! Golden validation against **official HDR-VDP-2.2.2** (the MATLAB release,
//! run under Octave 11.1). Inputs, `P_map`s, scalars, and per-plane quality
//! terms live in `validation/goldens/`; provenance, tolerances, and the
//! upstream-Octave `is_mex` workaround are documented in
//! `docs/VALIDATION.md`.
//!
//! This is the external-oracle complement to `bit_lock.rs`: the bit lock
//! proves the *optimised* code is identical to the frozen reference port;
//! this test proves the port matches the official implementation.

#![forbid(unsafe_code)]

use hdrvdp::{ColorEncoding, Params, hdrvdp};
use std::fs;
use std::path::{Path, PathBuf};

fn goldens_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/goldens")
}

fn read_f64(path: &Path) -> Vec<f64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 8, 0, "{path:?} not f64-aligned");
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

struct Golden {
    case: String,
    w: usize,
    h: usize,
    ppd: f64,
    q: f64,
    p_det: f64,
    c_max: f64,
}

fn cases() -> Vec<Golden> {
    let tsv = fs::read_to_string(goldens_dir().join("goldens.tsv")).expect("goldens.tsv");
    tsv.lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert!(f.len() >= 8, "bad goldens.tsv row: {l}");
            Golden {
                case: f[0].into(),
                w: f[2].parse().unwrap(),
                h: f[3].parse().unwrap(),
                ppd: f[4].parse().unwrap(),
                q: f[5].parse().unwrap(),
                p_det: f[6].parse().unwrap(),
                c_max: f[7].parse().unwrap(),
            }
        })
        .collect()
}

/// Tolerances: the measured worst-case deltas across the corpus are
/// |ΔP_det| = 5.3e-14, rel |ΔC_max| = 3.9e-12, |ΔP_map| = 3.9e-11,
/// |Δres.Q| = 9.9e-7 (VALIDATION.md). Bounds below carry ~3–10× headroom.
const TOL_PDET: f64 = 1e-12;
const TOL_CMAX_REL: f64 = 1e-9;
const TOL_PMAP: f64 = 1e-9;
const TOL_Q: f64 = 1e-5;

#[test]
fn all_cases_match_official_hdrvdp_2_2_2() {
    let dir = goldens_dir();
    let gs = cases();
    assert_eq!(gs.len(), 24, "expected 24 golden cases");

    for g in &gs {
        let ppd_i = g.ppd as usize;
        let test = read_f64(&dir.join(format!("{}_{}x{}_p{ppd_i}.f64", g.case, g.h, g.w)));
        let refr = read_f64(&dir.join(format!("ref_{}x{}.f64", g.h, g.w)));
        let pmap_want =
            read_f64(&dir.join(format!("{}_{}x{}_p{ppd_i}_pmap.f64", g.case, g.h, g.w)));
        assert_eq!(test.len(), g.w * g.h, "{}: input size", g.case);

        let par = Params {
            pix_per_deg: g.ppd,
            ..Default::default()
        };
        let res = hdrvdp(&test, &refr, g.w, g.h, ColorEncoding::Luminance, &par)
            .unwrap_or_else(|e| panic!("{}: {e:?}", g.case));

        let pmap_d = res
            .p_map
            .iter()
            .zip(&pmap_want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(pmap_d <= TOL_PMAP, "{}: P_map max|Δ| = {pmap_d:e}", g.case);
        assert!(
            (res.p_det - g.p_det).abs() <= TOL_PDET,
            "{}: P_det {} vs golden {}",
            g.case,
            res.p_det,
            g.p_det
        );
        assert!(
            (res.c_max - g.c_max).abs() / g.c_max.max(1.0) <= TOL_CMAX_REL,
            "{}: C_max {} vs golden {}",
            g.case,
            res.c_max,
            g.c_max
        );
        assert!(
            (res.q - g.q).abs() <= TOL_Q,
            "{}: res.Q {} vs golden {}",
            g.case,
            res.q,
            g.q
        );
    }
}
