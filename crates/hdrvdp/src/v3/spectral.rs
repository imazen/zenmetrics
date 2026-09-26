//! VDP3 spectral machinery — the tables under `data/v3/` loaded the way
//! upstream's `load_spectral_resp` does: every table's first column is the
//! wavelength in nm, the rest are data columns, and each column is
//! PCHIP-resampled onto a fixed 420-point grid spanning 360–780 nm
//! (`np.linspace(360, 780, 420)` — note the ~1.0024 nm step, reproduced
//! exactly).
//!
//! The cone fundamentals (`log_cone_smith_pokorny_1975.csv`) are **log10**
//! sensitivities and get `10**`; the scotopic curve
//! (`cie_scotopic_lum.txt`) is already linear and is used raw. Both have
//! their first 20 resampled rows zeroed before use — the reference's way of
//! killing the sub-380 nm extrapolation tail.
//!
//! `hdrvdp_iturgb2native` computes the `solve(native2xyz, itu2xyz)` matrix
//! that maps a standard RGB space onto the display's native channels.

use super::params::{DisplayPreset, Emission, InputEncoding};
use crate::interp::{interp1_pchip_extrap, trapz};
use crate::{Error, Result};

/// `np.linspace(360, 780, 420)` — the grid upstream resamples everything
/// onto: 420 points, endpoints inclusive, step 420/419 ≈ 1.0024 nm.
pub(crate) const LAMB_N: usize = 420;

/// The shared wavelength grid.
pub(crate) fn lamb() -> Vec<f64> {
    (0..LAMB_N)
        .map(|i| 360.0 + i as f64 * (780.0 - 360.0) / (LAMB_N - 1) as f64)
        .collect()
}

/// Parse a CSV table: first column = wavelength, remaining = data columns.
/// Returns `(wavelengths, columns)` — one `Vec<f64>` per data column.
fn parse_table(text: &str) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let mut wl = Vec::new();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() < 2 {
            return Err(Error::SpectralData(format!(
                "line {}: expected at least 2 columns",
                lineno + 1
            )));
        }
        let mut row = Vec::with_capacity(parts.len());
        for p in &parts {
            match p.parse::<f64>() {
                Ok(v) => row.push(v),
                Err(_) => {
                    return Err(Error::SpectralData(format!(
                        "line {}: unparseable value {p:?}",
                        lineno + 1
                    )));
                }
            }
        }
        if cols.is_empty() {
            cols = vec![Vec::new(); parts.len() - 1];
        } else if cols.len() != parts.len() - 1 {
            return Err(Error::SpectralData(format!(
                "line {}: {} data columns, expected {}",
                lineno + 1,
                parts.len() - 1,
                cols.len()
            )));
        }
        wl.push(row[0]);
        for (c, &v) in cols.iter_mut().zip(&row[1..]) {
            c.push(v);
        }
    }
    if wl.len() < 2 {
        return Err(Error::SpectralData("fewer than 2 rows".into()));
    }
    Ok((wl, cols))
}

/// `load_spectral_resp` — resample `src` columns onto the lamb grid with
/// extrapolating PCHIP.
fn resample(src: &str) -> Result<Vec<Vec<f64>>> {
    let (wl, cols) = parse_table(src)?;
    let grid = lamb();
    Ok(cols
        .iter()
        .map(|c| interp1_pchip_extrap(&wl, c, &grid))
        .collect())
}

/// The display's per-channel emission columns on the lamb grid —
/// `IMG_E` upstream.
pub(crate) fn emission_columns(emission: &Emission) -> Result<Vec<Vec<f64>>> {
    match emission {
        Emission::Preset(p) => resample(preset_text(*p)),
        Emission::Custom {
            wavelengths_nm,
            columns,
        } => {
            // Same resampling path as a CSV: caller supplies the raw table.
            if wavelengths_nm.len() < 2
                || columns.is_empty()
                || columns.iter().any(|c| c.len() != wavelengths_nm.len())
            {
                return Err(Error::SpectralData(
                    "custom emission: wavelength/column shape mismatch".into(),
                ));
            }
            if wavelengths_nm
                .windows(2)
                .any(|w| !(w[1] > w[0] && w[0].is_finite() && w[1].is_finite()))
            {
                return Err(Error::SpectralData(
                    "custom emission: wavelengths must be strictly increasing".into(),
                ));
            }
            let grid = lamb();
            Ok(columns
                .iter()
                .map(|c| interp1_pchip_extrap(wavelengths_nm, c, &grid))
                .collect())
        }
    }
}

fn preset_text(p: DisplayPreset) -> &'static str {
    match p {
        DisplayPreset::CcflLcd => include_str!("../../data/v3/emission_spectra_ccfl-lcd.csv"),
        DisplayPreset::Crt => include_str!("../../data/v3/emission_spectra_crt.csv"),
        DisplayPreset::LedLcd => include_str!("../../data/v3/emission_spectra_led-lcd.csv"),
        DisplayPreset::LedLcdSrgb => {
            include_str!("../../data/v3/emission_spectra_led-lcd-srgb.csv")
        }
        DisplayPreset::LedLcdWcg => {
            include_str!("../../data/v3/emission_spectra_led-lcd-wcg.csv")
        }
        DisplayPreset::Oled => include_str!("../../data/v3/emission_spectra_oled.csv"),
        DisplayPreset::D65 => include_str!("../../data/v3/d65.csv"),
    }
}

/// `LMSR_S` — the four receptor sensitivity columns (L, M, S cones, then
/// rods) on the lamb grid, after upstream's processing:
/// cones `10**`ed from the log table, zeros→min fill *before* the `10**`,
/// rods appended linear with their own first-20 zeroing.
pub(crate) fn receptor_sensitivities() -> Result<[Vec<f64>; 4]> {
    let mut lmsr = resample(include_str!(
        "../../data/v3/log_cone_smith_pokorny_1975.csv"
    ))?;
    let mut rod = resample(include_str!("../../data/v3/cie_scotopic_lum.txt"))?;
    if lmsr.len() != 3 || rod.len() != 1 {
        return Err(Error::SpectralData("vendored receptor table shape".into()));
    }
    for c in lmsr.iter_mut().take(3) {
        for v in c.iter_mut().take(20) {
            *v = 0.0;
        }
    }
    for v in rod[0].iter_mut().take(20) {
        *v = 0.0;
    }
    // `LMSR_S[LMSR_S==0] = np.min(LMSR_S)` — the minimum is taken over the
    // whole (log-domain) array, which is already strictly negative.
    let min_v = lmsr
        .iter()
        .flat_map(|c| c.iter().copied())
        .fold(f64::INFINITY, f64::min);
    for c in lmsr.iter_mut() {
        for v in c.iter_mut() {
            if *v == 0.0 {
                *v = min_v;
            }
            *v = 10f64.powf(*v);
        }
    }
    let r = rod.pop().unwrap();
    Ok([lmsr[0].clone(), lmsr[1].clone(), lmsr[2].clone(), r])
}

/// The CIE 1931 colour-matching functions on the lamb grid (linear).
pub(crate) fn xyz_cmf() -> Result<[Vec<f64>; 3]> {
    let c = resample(include_str!("../../data/v3/ciexyz31.csv"))?;
    if c.len() != 3 {
        return Err(Error::SpectralData("ciexyz31 table shape".into()));
    }
    Ok([c[0].clone(), c[1].clone(), c[2].clone()])
}

/// `hdrvdp_iturgb2native(standard, IMG_E)` — the `3×3` matrix mapping the
/// encoding's RGB (or XYZ) space onto the display's native channels.
/// Returns `None` for encodings that skip the transform upstream
/// (`luminance`, `luma-display`, `rgb-native`, `generic`).
pub(crate) fn itu2native(
    encoding: InputEncoding,
    img_e: &[Vec<f64>],
) -> Result<Option<[[f64; 3]; 3]>> {
    let standard = match encoding {
        InputEncoding::SrgbDisplay | InputEncoding::RgbBt709 => BT709_TO_XYZ,
        InputEncoding::RgbBt2020 => BT2020_TO_XYZ,
        InputEncoding::Xyz => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        _ => return Ok(None),
    };
    if img_e.len() != 3 {
        return Err(Error::SpectralData(
            "ITU→native transform requires a 3-channel emission table".into(),
        ));
    }
    let l = lamb();
    let cmf = xyz_cmf()?;
    let mut m_native2xyz = [[0.0f64; 3]; 3];
    for (rr, row) in m_native2xyz.iter_mut().enumerate() {
        for (cc, cell) in row.iter_mut().enumerate() {
            let prod: Vec<f64> = (0..LAMB_N)
                .map(|i| cmf[rr][i] * img_e[cc][i] * 683.002)
                .collect();
            *cell = trapz(&l, &prod);
        }
    }
    let norm = m_native2xyz[1].iter().sum::<f64>();
    for v in m_native2xyz.iter_mut().flat_map(|r| r.iter_mut()) {
        *v /= norm;
    }
    Ok(Some(solve3(m_native2xyz, standard)))
}

/// `solve(M_native2XYZ, M_iturgb2xyz)` — 3×3 dense solve.
#[allow(clippy::needless_range_loop)] // index form mirrors np.linalg.solve elimination
fn solve3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Gaussian elimination with partial pivoting; the matrix is a
    // well-conditioned colour transform, matching `np.linalg.solve`.
    let mut m = a;
    let mut x = b;
    for col in 0..3 {
        let mut piv = col;
        for r in col + 1..3 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        m.swap(col, piv);
        x.swap(col, piv);
        for r in col + 1..3 {
            let f = m[r][col] / m[col][col];
            for c in col..3 {
                m[r][c] -= f * m[col][c];
            }
            for c in 0..3 {
                x[r][c] -= f * x[col][c];
            }
        }
    }
    let mut out = [[0.0f64; 3]; 3];
    for c in 0..3 {
        for r in (0..3).rev() {
            let mut s = x[r][c];
            for k in r + 1..3 {
                s -= m[r][k] * out[k][c];
            }
            out[r][c] = s / m[r][r];
        }
    }
    out
}

const BT709_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
];
const BT2020_TO_XYZ: [[f64; 3]; 3] = [
    [0.6370, 0.1446, 0.1689],
    [0.2627, 0.6780, 0.0593],
    [0.0000, 0.0281, 1.0610],
];

/// The Pokorny aging-lens optical-density correction — upstream's `do_aod`
/// block. Returns the per-wavelength transmission `10^{-PCHIP(OD−OD_y)}`
/// evaluated at the lamb grid clipped to `[400, 650]`.
pub(crate) fn aging_optical_density_filter(age: u8) -> Vec<f64> {
    const LAM: [f64; 26] = [
        400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0, 510.0, 520.0,
        530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0, 610.0, 620.0, 630.0, 640.0, 650.0,
    ];
    const TL1: [f64; 26] = [
        0.600, 0.510, 0.433, 0.377, 0.327, 0.295, 0.267, 0.233, 0.207, 0.187, 0.167, 0.147, 0.133,
        0.120, 0.107, 0.093, 0.080, 0.067, 0.053, 0.040, 0.033, 0.027, 0.020, 0.013, 0.007, 0.000,
    ];
    const TL2_6: [f64; 6] = [1.000, 0.583, 0.300, 0.116, 0.033, 0.005];
    let tl2 = |i: usize| if i < 6 { TL2_6[i] } else { 0.0 };
    let age = f64::from(age);
    let od_minus_y: Vec<f64> = (0..26)
        .map(|i| {
            let od = if age <= 60.0 {
                TL1[i] * (1.0 + 0.02 * (age - 32.0)) + tl2(i)
            } else {
                TL1[i] * (1.56 + 0.0667 * (age - 60.0)) + tl2(i)
            };
            od - (TL1[i] + tl2(i))
        })
        .collect();
    let clipped: Vec<f64> = lamb().iter().map(|&v| v.clamp(LAM[0], LAM[25])).collect();
    interp1_pchip_extrap(&LAM, &od_minus_y, &clipped)
        .iter()
        .map(|&d| 10f64.powf(-d))
        .collect()
}
