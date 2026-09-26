//! Golden validation against the **HDR-VDP-3 reference port** — the
//! `VDP3/` tree of `jpeg-ai-qaf` (`feature/HDR-VDP2.2/main`, commit
//! `0628a6b`), a NumPy transliteration of HDR-VDP-3.0.7 run under
//! numpy 2.x / scipy 1.18 with shimmed `trapz`/`cumtrapz`/`np.e`.
//!
//! Cases live in `tests/data/v3/<name>/` (`test.f64le`, `ref.f64le`,
//! `pmap.f64le`, `meta.json`); `manifest.tsv` lists them with the
//! reference `Q`, `Q_JOD`, `P_map` mean/max.
//!
//! The reference runner is `/home/lilith/tmp/jpeg-ai-qaf-hdrvdp3/
//! gen_vdp3_goldens.py` — regenerating needs a Python with numpy+scipy.

#![forbid(unsafe_code)]

use hdrvdp::v3::{
    Emission, InputEncoding, Options, Params, Surround, Task, ViewingConditions, hdrvdp3,
};
use std::fs;
use std::path::{Path, PathBuf};

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/v3")
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

/// One column of a comma-separated `wavelength,ch0,ch1,…` emission CSV.
fn csv_col(path: &Path, col: usize) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path:?}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').nth(col).unwrap().trim().parse().unwrap())
        .collect()
}

struct Case {
    name: String,
    w: usize,
    h: usize,
    c: usize,
    encoding: InputEncoding,
    ppd: f64,
    task: Task,
    age: u8,
    surround: Surround,
    flags: Vec<String>,
    q: f64,
    q_jod: f64,
    pmap_mean: f64,
    pmap_max: f64,
}

fn parse_encoding(s: &str) -> InputEncoding {
    match s {
        "luminance" => InputEncoding::Luminance,
        "luma-display" => InputEncoding::LumaDisplay,
        // NB: upstream's docstring advertises 'sRGB-display' but the code
        // compares case-sensitively against 'srgb-display'; the manifest
        // uses the exact string that exercises the real path.
        "srgb-display" | "sRGB-display" => InputEncoding::SrgbDisplay,
        "rgb-bt.709" => InputEncoding::RgbBt709,
        "rgb-bt.2020" => InputEncoding::RgbBt2020,
        "rgb-native" => InputEncoding::RgbNative,
        "xyz" => InputEncoding::Xyz,
        "generic" => InputEncoding::Generic,
        _ => panic!("unknown encoding {s}"),
    }
}

fn parse_task(s: &str) -> Task {
    match s {
        "quality" => Task::Quality,
        "side-by-side" | "sbs" => Task::SideBySide,
        "flicker" => Task::Flicker,
        _ => panic!("unknown task {s}"),
    }
}

fn cases() -> Vec<Case> {
    let tsv = fs::read_to_string(data_dir().join("manifest.tsv")).expect("manifest.tsv");
    tsv.lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 14, "bad manifest row: {l}");
            let surround = match f[8] {
                "none" => Surround::None,
                "mean" => Surround::Mean,
                _ => panic!("unknown surround {}", f[8]),
            };
            Case {
                name: f[0].into(),
                w: f[1].parse().unwrap(),
                h: f[2].parse().unwrap(),
                c: f[3].parse().unwrap(),
                encoding: parse_encoding(f[4]),
                ppd: f[5].parse().unwrap(),
                task: parse_task(f[6]),
                age: f[7].parse().unwrap(),
                surround,
                flags: f[9]
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect(),
                q: f[10].parse().unwrap(),
                q_jod: f[11].parse().unwrap(),
                pmap_mean: f[12].parse().unwrap(),
                pmap_max: f[13].parse().unwrap(),
            }
        })
        .collect()
}

/// f64-vs-f64 agreement budget. Measured against the NumPy reference the
/// worst deviation over the 21-case corpus is |ΔQ_JOD| ≈ 1.6e-13 and
/// P_map max |Δ| ≈ 6.1e-13 — the remaining noise is FFT/transcendental
/// rounding order. 1e-9 keeps three orders of headroom for platform FP
/// variation while still failing on any *algorithmic* drift.
const TOL_Q_JOD: f64 = 1e-9;
const TOL_Q: f64 = 1e-9;
const TOL_PMAP: f64 = 1e-9;

#[test]
fn all_cases_match_reference_vdp3() {
    let dir = data_dir();
    let cs = cases();
    assert_eq!(cs.len(), 21, "expected 21 golden cases");

    for c in &cs {
        let d = dir.join(&c.name);
        let test = read_f64(&d.join("test.f64le"));
        let refr = read_f64(&d.join("ref.f64le"));
        let pmap_want = read_f64(&d.join("pmap.f64le"));
        assert_eq!(test.len(), c.w * c.h * c.c, "{}: test size", c.name);

        let mut opts = Options::reference(c.task);
        let mut emission = Emission::default_for(c.encoding);
        for fl in &c.flags {
            match fl.as_str() {
                "surround-mean" => {} // already carried by the surround column
                "pxthr" => opts.do_pixel_threshold = true,
                "no-mask" => opts.do_masking = false,
                "si-gauss" => opts.do_si_gauss = true,
                "custom-emission" => {
                    emission = Emission::Custom {
                        wavelengths_nm: csv_col(&dir.join("custom_emission_2ch.csv"), 0),
                        columns: vec![
                            csv_col(&dir.join("custom_emission_2ch.csv"), 1),
                            csv_col(&dir.join("custom_emission_2ch.csv"), 2),
                        ],
                    };
                }
                _ => panic!("{}: unknown flag {fl}", c.name),
            }
        }

        let par = Params::new(
            c.task,
            ViewingConditions::new(c.ppd, c.surround.clone(), c.age),
            c.encoding,
            emission,
            opts,
        )
        .unwrap_or_else(|e| panic!("{}: params: {e:?}", c.name));
        let res =
            hdrvdp3(&test, &refr, c.w, c.h, &par).unwrap_or_else(|e| panic!("{}: {e:?}", c.name));

        let pmap_d = res
            .p_map
            .iter()
            .zip(&pmap_want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let pmap_mean: f64 = res.p_map.iter().sum::<f64>() / res.p_map.len() as f64;
        let pmap_max: f64 = res.p_map.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        eprintln!(
            "{}: Q_JOD {} vs {} (Δ{:e}), P_map max|Δ| {pmap_d:e}",
            c.name,
            res.q_jod,
            c.q_jod,
            (res.q_jod - c.q_jod).abs()
        );
        assert!(
            pmap_d <= TOL_PMAP,
            "{}: P_map max|Δ| = {pmap_d:e} (ours mean {pmap_mean:e} vs {} , max {pmap_max:e} vs {})",
            c.name,
            c.pmap_mean,
            c.pmap_max
        );
        assert!(
            (res.q - c.q).abs() <= TOL_Q,
            "{}: Q {} vs golden {}",
            c.name,
            res.q,
            c.q
        );
        assert!(
            (res.q_jod - c.q_jod).abs() <= TOL_Q_JOD,
            "{}: Q_JOD {} vs golden {}",
            c.name,
            res.q_jod,
            c.q_jod
        );
    }
}

/// Every structurally-invalid call must return `Err`, never panic —
/// the caller's entire job is declaring the physical setup correctly.
#[test]
fn invalid_inputs_err_not_panic() {
    use hdrvdp::Error;

    let mk = |enc, em| {
        Params::new(
            Task::Quality,
            ViewingConditions::new(30.0, Surround::None, 24),
            enc,
            em,
            Options::reference(Task::Quality),
        )
    };
    let par = |enc| mk(enc, Emission::default_for(enc)).unwrap();

    // buffer-length vs encoding channel count
    let p = par(InputEncoding::Luminance);
    let im = vec![100.0; 64 * 64];
    assert!(matches!(
        hdrvdp3(&im[..im.len() - 1], &im, 64, 64, &p),
        Err(Error::SizeMismatch { .. }) | Err(Error::ChannelMismatch { .. })
    ));
    // 3-channel encoding fed 1-channel data
    let p3 = par(InputEncoding::RgbBt709);
    assert!(matches!(
        hdrvdp3(&im, &im, 64, 64, &p3),
        Err(Error::ChannelMismatch { .. })
    ));
    // generic encoding without a custom emission — rejected at Params
    assert!(matches!(
        mk(
            InputEncoding::Generic,
            Emission::Preset(hdrvdp::v3::DisplayPreset::D65)
        ),
        Err(Error::MissingEmission)
    ));
    // emission column count ≠ image channels: for `Generic` the channel
    // count derives *from* the table, so the mismatch is expressible only
    // as a fixed encoding + a Custom table of the wrong arity.
    let wl: Vec<f64> = (0..85).map(|i| 360.0 + i as f64 * 5.0).collect();
    let col = vec![1.0; wl.len()];
    let p2 = mk(
        InputEncoding::RgbBt709,
        Emission::Custom {
            wavelengths_nm: wl.clone(),
            columns: vec![col.clone(), col.clone()],
        },
    )
    .unwrap();
    let im3 = vec![50.0; 64 * 64 * 3];
    assert!(matches!(
        hdrvdp3(&im3, &im3, 64, 64, &p2),
        Err(Error::SpectralData(_))
    ));
    // ppd < 4 rejected at construction
    assert!(matches!(
        Params::new(
            Task::Quality,
            ViewingConditions::new(3.0, Surround::None, 24),
            InputEncoding::Luminance,
            Emission::default_for(InputEncoding::Luminance),
            Options::reference(Task::Quality),
        ),
        Err(Error::InvalidResolution(_))
    ));
    // non-increasing custom wavelengths rejected
    assert!(
        mk(
            InputEncoding::Generic,
            Emission::Custom {
                wavelengths_nm: vec![360.0, 500.0, 400.0],
                columns: vec![vec![1.0; 3]],
            },
        )
        .and_then(|p| hdrvdp3(&[1.0; 64 * 64], &[1.0; 64 * 64], 64, 64, &p).map(|_| p))
        .is_err()
    );
}
