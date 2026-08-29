//! Colour-encoding / display models: input pixels → absolute luminance
//! (cd/m²) per channel.
//!
//! Ported from the `color_encoding` switch and the `display_model*` /
//! `xyz2rgb` subfunctions of `hdrvdp.m` (MATLAB `hdrvdp-2.2.x`).
//!
//! Copyright (c) 2011, Rafal Mantiuk <mantiuk@gmail.com>
//!
//! Permission to use, copy, modify, and/or distribute this software for any
//! purpose with or without fee is hereby granted, provided that the above
//! copyright notice and this permission notice appear in all copies.
//!
//! THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//! WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//! MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//! ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//! WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//! ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//! OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

use crate::Error;
use crate::spectral::DisplaySpectra;

/// Peak luminance of the SDR display model, in cd/m².
pub const SDR_PEAK_NITS: f64 = 99.0;
/// Black level of the SDR display model, in cd/m².
pub const SDR_BLACK_NITS: f64 = 1.0;
/// Gamma of the `LumaDisplay` model.
pub const SDR_GAMMA: f64 = 2.2;

/// How the input pixel values are encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorEncoding {
    /// 1 channel, already absolute luminance in cd/m².
    Luminance,
    /// 1 channel of gamma-encoded luma in `[0, 1]`, driven through a
    /// `peak · V^2.2 + black` display model (99 cd/m² peak, 1 cd/m² black).
    LumaDisplay,
    /// 3 channels of sRGB in `[0, 1]`, driven through the sRGB EOTF and the
    /// same 99/1 display model.
    SrgbDisplay,
    /// 3 channels of linear BT.709-primary RGB, already in cd/m².
    RgbBt709,
    /// 3 channels of linear CIE XYZ in cd/m² (Y is luminance); converted to
    /// BT.709 RGB.
    Xyz,
}

impl ColorEncoding {
    /// Number of input channels this encoding requires.
    #[must_use]
    pub fn channels(self) -> usize {
        match self {
            Self::Luminance | Self::LumaDisplay => 1,
            Self::SrgbDisplay | Self::RgbBt709 | Self::Xyz => 3,
        }
    }

    /// Which display emission spectra this encoding implies.
    #[must_use]
    pub fn spectra(self) -> DisplaySpectra {
        match self {
            Self::Luminance | Self::LumaDisplay => DisplaySpectra::D65,
            Self::SrgbDisplay | Self::RgbBt709 | Self::Xyz => DisplaySpectra::CcflLcd,
        }
    }

    /// Whether values below 1 cd/m² across the whole image mean the caller
    /// almost certainly passed relative rather than absolute values.
    ///
    /// True for the encodings that take absolute input; false for the ones that
    /// take code values and apply a display model themselves.
    #[must_use]
    pub fn expects_absolute_input(self) -> bool {
        matches!(self, Self::Luminance | Self::RgbBt709 | Self::Xyz)
    }
}

/// `L = peak · V^gamma + black_level`.
#[must_use]
#[inline]
pub fn display_model(v: f64, gamma: f64, peak: f64, black_level: f64) -> f64 {
    peak * v.powf(gamma) + black_level
}

/// The sRGB EOTF followed by the 99/1 cd/m² display model.
#[must_use]
#[inline]
pub fn display_model_srgb(srgb: f64) -> f64 {
    const A: f64 = 0.055;
    const THR: f64 = 0.04045;
    let lin = if srgb <= THR {
        srgb / 12.92
    } else {
        ((srgb + A) / (1.0 + A)).powf(2.4)
    };
    SDR_PEAK_NITS * lin + SDR_BLACK_NITS
}

/// CIE XYZ → linear BT.709 RGB, using the matrix from `hdrvdp.m`.
#[must_use]
#[inline]
pub fn xyz_to_rgb(xyz: [f64; 3]) -> [f64; 3] {
    const M: [[f64; 3]; 3] = [
        [3.240708, -1.537259, -0.498570],
        [-0.969257, 1.875995, 0.041555],
        [0.055636, -0.203996, 1.057069],
    ];
    let mut out = [0.0; 3];
    for (o, row) in out.iter_mut().zip(M) {
        *o = row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2];
    }
    out
}

/// Apply `encoding` to an interleaved image, producing interleaved absolute
/// luminance in cd/m² with the same channel count.
///
/// `pixels.len()` must equal `width · height · encoding.channels()`.
///
/// # Errors
/// Returns [`Error::ChannelMismatch`] if the buffer length does not match the
/// encoding's channel count, and [`Error::ImpossibleValues`] if the result
/// contains a non-finite luminance.
pub fn to_nits(
    pixels: &[f64],
    width: usize,
    height: usize,
    encoding: ColorEncoding,
) -> Result<Vec<f64>, Error> {
    let ch = encoding.channels();
    let want = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(ch))
        .ok_or(Error::DimensionOverflow)?;
    if pixels.len() != want {
        return Err(Error::ChannelMismatch {
            expected: want,
            got: pixels.len(),
        });
    }

    let out: Vec<f64> = match encoding {
        ColorEncoding::Luminance | ColorEncoding::RgbBt709 => pixels.to_vec(),
        ColorEncoding::LumaDisplay => pixels
            .iter()
            .map(|&v| display_model(v, SDR_GAMMA, SDR_PEAK_NITS, SDR_BLACK_NITS))
            .collect(),
        ColorEncoding::SrgbDisplay => pixels.iter().map(|&v| display_model_srgb(v)).collect(),
        ColorEncoding::Xyz => {
            let mut o = Vec::with_capacity(pixels.len());
            let (triples, _) = pixels.as_chunks::<3>();
            for p in triples {
                o.extend_from_slice(&xyz_to_rgb([p[0], p[1], p[2]]));
            }
            o
        }
    };

    if let Some(bad) = out.iter().find(|v| !v.is_finite()) {
        return Err(Error::ImpossibleValues(*bad));
    }
    Ok(out)
}

/// Whether an image looks like it was passed in relative rather than absolute
/// units — upstream's `check_if_values_plausible` warning, returned as a fact
/// instead of printed.
///
/// Only meaningful for encodings where [`ColorEncoding::expects_absolute_input`]
/// is true. Upstream checks the green channel for 3-channel input.
#[must_use]
pub fn looks_relative(nits: &[f64], channels: usize, encoding: ColorEncoding) -> bool {
    if !encoding.expects_absolute_input() {
        return false;
    }
    let probe: f64 = if channels == 3 {
        nits.as_chunks::<3>()
            .0
            .iter()
            .map(|p| p[1])
            .fold(f64::MIN, f64::max)
    } else {
        nits.iter().copied().fold(f64::MIN, f64::max)
    };
    probe <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_display_model_endpoints_and_continuity() {
        // Black code value → the display's black level; white → peak + black.
        assert!((display_model_srgb(0.0) - 1.0).abs() < 1e-12);
        assert!((display_model_srgb(1.0) - 100.0).abs() < 1e-12);
        // The piecewise EOTF must be continuous at the 0.04045 breakpoint.
        let lo = display_model_srgb(0.04045 - 1e-9);
        let hi = display_model_srgb(0.04045 + 1e-9);
        assert!((lo - hi).abs() < 1e-6, "discontinuity: {lo} vs {hi}");
        // Mid grey (code 0.5) is ~21.4% linear → ~22.2 cd/m².
        let mid = display_model_srgb(0.5);
        assert!((mid - 22.24).abs() < 0.05, "{mid}");
        // Monotone.
        let mut prev = f64::MIN;
        for i in 0..=1000 {
            let v = display_model_srgb(i as f64 / 1000.0);
            assert!(v > prev);
            prev = v;
        }
    }

    #[test]
    fn luma_display_model_endpoints() {
        assert!((display_model(0.0, SDR_GAMMA, SDR_PEAK_NITS, SDR_BLACK_NITS) - 1.0).abs() < 1e-12);
        assert!(
            (display_model(1.0, SDR_GAMMA, SDR_PEAK_NITS, SDR_BLACK_NITS) - 100.0).abs() < 1e-12
        );
    }

    #[test]
    fn xyz_to_rgb_is_the_bt709_inverse() {
        // D65 white in XYZ at 100 cd/m² must map to near-equal RGB.
        let rgb = xyz_to_rgb([95.047, 100.0, 108.883]);
        for v in rgb {
            assert!((v - 100.0).abs() < 0.6, "{rgb:?}");
        }
        // Pure Y should stay grey-ish but is not a valid colour on its own;
        // what matters is linearity.
        let a = xyz_to_rgb([1.0, 2.0, 3.0]);
        let b = xyz_to_rgb([2.0, 4.0, 6.0]);
        for (x, y) in a.iter().zip(b) {
            assert!((2.0 * x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn to_nits_channel_mismatch_is_an_error() {
        let e = to_nits(&[0.0; 5], 2, 2, ColorEncoding::Luminance).unwrap_err();
        assert!(matches!(e, Error::ChannelMismatch { .. }), "{e:?}");
        let e = to_nits(&[0.0; 4], 2, 2, ColorEncoding::SrgbDisplay).unwrap_err();
        assert!(matches!(e, Error::ChannelMismatch { .. }), "{e:?}");
    }

    #[test]
    fn to_nits_passes_absolute_encodings_through_unchanged() {
        let px = [0.5, 12.0, 3000.0, 1e-4];
        let got = to_nits(&px, 2, 2, ColorEncoding::Luminance).unwrap();
        assert_eq!(got, px.to_vec());
    }

    #[test]
    fn to_nits_rejects_non_finite_results() {
        let e = to_nits(&[f64::NAN], 1, 1, ColorEncoding::Luminance).unwrap_err();
        assert!(matches!(e, Error::ImpossibleValues(_)), "{e:?}");
    }

    #[test]
    fn looks_relative_only_fires_for_absolute_encodings() {
        // A [0,1] image passed as absolute luminance is the classic misuse.
        assert!(looks_relative(
            &[0.1, 0.9, 0.3],
            1,
            ColorEncoding::Luminance
        ));
        assert!(!looks_relative(&[0.1, 300.0], 1, ColorEncoding::Luminance));
        // Code-value encodings legitimately live in [0,1].
        assert!(!looks_relative(&[0.1, 0.9], 1, ColorEncoding::LumaDisplay));
        // 3-channel probes the green channel.
        assert!(looks_relative(
            &[500.0, 0.4, 500.0],
            3,
            ColorEncoding::RgbBt709
        ));
    }
}
