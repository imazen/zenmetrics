//! The HDR-VDP-3 multi-scale decomposition: an **isotropic** steerable
//! pyramid on the `sp0Filters` kernel set.
//!
//! Unlike HDR-VDP-2's `sp3Filters` pyramid (4 oriented bands per level),
//! `sp0` has a single isotropic bandpass filter per level — the pyramid is
//! effectively a Laplacian-style band stack:
//!
//! ```text
//!   hi0  = corrDn(im,  hi0filt)          — residual high-pass, full res
//!   lo0  = corrDn(im,  lo0filt)
//!   per level ℓ: band_ℓ = corrDn(lo_ℓ, bfilt);  lo_{ℓ+1} = corrDn(lo_ℓ, lofilt, ↓2)
//!   final band = lo_H                        — residual low-pass
//! ```
//!
//! so `band_count = levels + 2` and band `b` is normalised by `2^-b` on read
//! (stored raw, scaled on access — upstream's `get_band`/`set_band`
//! `band_norm`). Band frequencies are `ppd/2^(b+1)` cycles/degree.
//!
//! Boundary handling is `reflect1` (mirror without duplicating the edge
//! sample), matching `corrDn`/`upConv` in the reference's `PyrTools`.
//!
//! Ported from `VDP3/hdrvdp_spyr.py` + `VDP3/PyrTools.py` (`sp0Filters`,
//! `buildSpyr`, `reconSpyr`) of the `jpeg-ai-qaf` HDR-VDP-3 port — itself a
//! transliteration of Simoncelli's matlabPyrTools (MIT). `f64` throughout.

use crate::spyr::{max_pyr_height, reflect1};

/// Pyramid low-pass pre-filter, 7×7 (upstream `sp0Filters` `lo0filt`).
#[allow(clippy::excessive_precision)]
const LO0FILT: [f64; 49] = [
    -4.514000000000000217e-04,
    -1.137099999999999938e-04,
    -3.725800000000000140e-04,
    -3.743859999999999896e-03,
    -3.725800000000000140e-04,
    -1.137099999999999938e-04,
    -4.514000000000000217e-04,
    -1.137099999999999938e-04,
    -6.119519999999999821e-03,
    -1.344159999999999973e-02,
    -7.563199999999999958e-03,
    -1.344159999999999973e-02,
    -6.119519999999999821e-03,
    -1.137099999999999938e-04,
    -3.725800000000000140e-04,
    -1.344159999999999973e-02,
    6.441487999999999381e-02,
    1.524935000000000040e-01,
    6.441487999999999381e-02,
    -1.344159999999999973e-02,
    -3.725800000000000140e-04,
    -3.743859999999999896e-03,
    -7.563199999999999958e-03,
    1.524935000000000040e-01,
    3.153017000000000181e-01,
    1.524935000000000040e-01,
    -7.563199999999999958e-03,
    -3.743859999999999896e-03,
    -3.725800000000000140e-04,
    -1.344159999999999973e-02,
    6.441487999999999381e-02,
    1.524935000000000040e-01,
    6.441487999999999381e-02,
    -1.344159999999999973e-02,
    -3.725800000000000140e-04,
    -1.137099999999999938e-04,
    -6.119519999999999821e-03,
    -1.344159999999999973e-02,
    -7.563199999999999958e-03,
    -1.344159999999999973e-02,
    -6.119519999999999821e-03,
    -1.137099999999999938e-04,
    -4.514000000000000217e-04,
    -1.137099999999999938e-04,
    -3.725800000000000140e-04,
    -3.743859999999999896e-03,
    -3.725800000000000140e-04,
    -1.137099999999999938e-04,
    -4.514000000000000217e-04,
];
/// Pyramid high-pass residual filter, 9×9 (`hi0filt`).
#[allow(clippy::excessive_precision)]
const HI0FILT: [f64; 81] = [
    5.997199999999999840e-04,
    -6.068000000000000179e-05,
    -3.324900000000000099e-04,
    -3.325599999999999737e-04,
    -2.406599999999999896e-04,
    -3.325599999999999737e-04,
    -3.324900000000000099e-04,
    -6.068000000000000179e-05,
    5.997199999999999840e-04,
    -6.068000000000000179e-05,
    1.263100000000000019e-04,
    4.927100000000000446e-04,
    1.459699999999999865e-04,
    -3.732100000000000131e-04,
    1.459699999999999865e-04,
    4.927100000000000446e-04,
    1.263100000000000019e-04,
    -6.068000000000000179e-05,
    -3.324900000000000099e-04,
    4.927100000000000446e-04,
    -1.616650000000000075e-03,
    -1.437358000000000038e-02,
    -2.420138000000000150e-02,
    -1.437358000000000038e-02,
    -1.616650000000000075e-03,
    4.927100000000000446e-04,
    -3.324900000000000099e-04,
    -3.325599999999999737e-04,
    1.459699999999999865e-04,
    -1.437358000000000038e-02,
    -6.300922999999999941e-02,
    -9.623594000000000592e-02,
    -6.300922999999999941e-02,
    -1.437358000000000038e-02,
    1.459699999999999865e-04,
    -3.325599999999999737e-04,
    -2.406599999999999896e-04,
    -3.732100000000000131e-04,
    -2.420138000000000150e-02,
    -9.623594000000000592e-02,
    8.554893000000000081e-01,
    -9.623594000000000592e-02,
    -2.420138000000000150e-02,
    -3.732100000000000131e-04,
    -2.406599999999999896e-04,
    -3.325599999999999737e-04,
    1.459699999999999865e-04,
    -1.437358000000000038e-02,
    -6.300922999999999941e-02,
    -9.623594000000000592e-02,
    -6.300922999999999941e-02,
    -1.437358000000000038e-02,
    1.459699999999999865e-04,
    -3.325599999999999737e-04,
    -3.324900000000000099e-04,
    4.927100000000000446e-04,
    -1.616650000000000075e-03,
    -1.437358000000000038e-02,
    -2.420138000000000150e-02,
    -1.437358000000000038e-02,
    -1.616650000000000075e-03,
    4.927100000000000446e-04,
    -3.324900000000000099e-04,
    -6.068000000000000179e-05,
    1.263100000000000019e-04,
    4.927100000000000446e-04,
    1.459699999999999865e-04,
    -3.732100000000000131e-04,
    1.459699999999999865e-04,
    4.927100000000000446e-04,
    1.263100000000000019e-04,
    -6.068000000000000179e-05,
    5.997199999999999840e-04,
    -6.068000000000000179e-05,
    -3.324900000000000099e-04,
    -3.325599999999999737e-04,
    -2.406599999999999896e-04,
    -3.325599999999999737e-04,
    -3.324900000000000099e-04,
    -6.068000000000000179e-05,
    5.997199999999999840e-04,
];
/// Pyramid level low-pass (decimation) filter, 13×13 (`lofilt`).
#[allow(clippy::excessive_precision)]
const LOFILT: [f64; 169] = [
    -2.257000000000000109e-04,
    -8.064399999999999563e-04,
    -5.686000000000000109e-05,
    8.741400000000000298e-04,
    -1.862800000000000122e-04,
    -1.031639999999999901e-03,
    -1.871920000000000008e-03,
    -1.031639999999999901e-03,
    -1.862800000000000122e-04,
    8.741400000000000298e-04,
    -5.686000000000000109e-05,
    -8.064399999999999563e-04,
    -2.257000000000000109e-04,
    -8.064399999999999563e-04,
    1.417620000000000078e-03,
    -1.903800000000000034e-04,
    -2.449059999999999866e-03,
    -4.596420000000000367e-03,
    -7.006740000000000017e-03,
    -6.948900000000000042e-03,
    -7.006740000000000017e-03,
    -4.596420000000000367e-03,
    -2.449059999999999866e-03,
    -1.903800000000000034e-04,
    1.417620000000000078e-03,
    -8.064399999999999563e-04,
    -5.686000000000000109e-05,
    -1.903800000000000034e-04,
    -3.059759999999999910e-03,
    -6.400999999999999572e-03,
    -6.720799999999999864e-03,
    -5.236180000000000001e-03,
    -3.781599999999999979e-03,
    -5.236180000000000001e-03,
    -6.720799999999999864e-03,
    -6.400999999999999572e-03,
    -3.059759999999999910e-03,
    -1.903800000000000034e-04,
    -5.686000000000000109e-05,
    8.741400000000000298e-04,
    -2.449059999999999866e-03,
    -6.400999999999999572e-03,
    -5.260019999999999800e-03,
    3.938620000000000315e-03,
    1.722078000000000150e-02,
    2.449600000000000041e-02,
    1.722078000000000150e-02,
    3.938620000000000315e-03,
    -5.260019999999999800e-03,
    -6.400999999999999572e-03,
    -2.449059999999999866e-03,
    8.741400000000000298e-04,
    -1.862800000000000122e-04,
    -4.596420000000000367e-03,
    -6.720799999999999864e-03,
    3.938620000000000315e-03,
    3.220743999999999690e-02,
    6.306261999999999979e-02,
    7.624673999999999341e-02,
    6.306261999999999979e-02,
    3.220743999999999690e-02,
    3.938620000000000315e-03,
    -6.720799999999999864e-03,
    -4.596420000000000367e-03,
    -1.862800000000000122e-04,
    -1.031639999999999901e-03,
    -7.006740000000000017e-03,
    -5.236180000000000001e-03,
    1.722078000000000150e-02,
    6.306261999999999979e-02,
    1.116387999999999964e-01,
    1.348998999999999893e-01,
    1.116387999999999964e-01,
    6.306261999999999979e-02,
    1.722078000000000150e-02,
    -5.236180000000000001e-03,
    -7.006740000000000017e-03,
    -1.031639999999999901e-03,
    -1.871920000000000008e-03,
    -6.948900000000000042e-03,
    -3.781599999999999979e-03,
    2.449600000000000041e-02,
    7.624673999999999341e-02,
    1.348998999999999893e-01,
    1.576508000000000076e-01,
    1.348998999999999893e-01,
    7.624673999999999341e-02,
    2.449600000000000041e-02,
    -3.781599999999999979e-03,
    -6.948900000000000042e-03,
    -1.871920000000000008e-03,
    -1.031639999999999901e-03,
    -7.006740000000000017e-03,
    -5.236180000000000001e-03,
    1.722078000000000150e-02,
    6.306261999999999979e-02,
    1.116387999999999964e-01,
    1.348998999999999893e-01,
    1.116387999999999964e-01,
    6.306261999999999979e-02,
    1.722078000000000150e-02,
    -5.236180000000000001e-03,
    -7.006740000000000017e-03,
    -1.031639999999999901e-03,
    -1.862800000000000122e-04,
    -4.596420000000000367e-03,
    -6.720799999999999864e-03,
    3.938620000000000315e-03,
    3.220743999999999690e-02,
    6.306261999999999979e-02,
    7.624673999999999341e-02,
    6.306261999999999979e-02,
    3.220743999999999690e-02,
    3.938620000000000315e-03,
    -6.720799999999999864e-03,
    -4.596420000000000367e-03,
    -1.862800000000000122e-04,
    8.741400000000000298e-04,
    -2.449059999999999866e-03,
    -6.400999999999999572e-03,
    -5.260019999999999800e-03,
    3.938620000000000315e-03,
    1.722078000000000150e-02,
    2.449600000000000041e-02,
    1.722078000000000150e-02,
    3.938620000000000315e-03,
    -5.260019999999999800e-03,
    -6.400999999999999572e-03,
    -2.449059999999999866e-03,
    8.741400000000000298e-04,
    -5.686000000000000109e-05,
    -1.903800000000000034e-04,
    -3.059759999999999910e-03,
    -6.400999999999999572e-03,
    -6.720799999999999864e-03,
    -5.236180000000000001e-03,
    -3.781599999999999979e-03,
    -5.236180000000000001e-03,
    -6.720799999999999864e-03,
    -6.400999999999999572e-03,
    -3.059759999999999910e-03,
    -1.903800000000000034e-04,
    -5.686000000000000109e-05,
    -8.064399999999999563e-04,
    1.417620000000000078e-03,
    -1.903800000000000034e-04,
    -2.449059999999999866e-03,
    -4.596420000000000367e-03,
    -7.006740000000000017e-03,
    -6.948900000000000042e-03,
    -7.006740000000000017e-03,
    -4.596420000000000367e-03,
    -2.449059999999999866e-03,
    -1.903800000000000034e-04,
    1.417620000000000078e-03,
    -8.064399999999999563e-04,
    -2.257000000000000109e-04,
    -8.064399999999999563e-04,
    -5.686000000000000109e-05,
    8.741400000000000298e-04,
    -1.862800000000000122e-04,
    -1.031639999999999901e-03,
    -1.871920000000000008e-03,
    -1.031639999999999901e-03,
    -1.862800000000000122e-04,
    8.741400000000000298e-04,
    -5.686000000000000109e-05,
    -8.064399999999999563e-04,
    -2.257000000000000109e-04,
];
/// Per-level isotropic bandpass filter, 9×9 (`bfilts`; one band per level).
#[allow(clippy::excessive_precision)]
const BFILT: [f64; 81] = [
    -9.066000000000000274e-05,
    -1.738640000000000064e-03,
    -4.942499999999999845e-03,
    -7.889389999999999598e-03,
    -1.009472999999999968e-02,
    -7.889389999999999598e-03,
    -4.942499999999999845e-03,
    -1.738640000000000064e-03,
    -9.066000000000000274e-05,
    -1.738640000000000064e-03,
    -4.625149999999999748e-03,
    -7.272540000000000081e-03,
    -7.623409999999999735e-03,
    -9.091949999999999685e-03,
    -7.623409999999999735e-03,
    -7.272540000000000081e-03,
    -4.625149999999999748e-03,
    -1.738640000000000064e-03,
    -4.942499999999999845e-03,
    -7.272540000000000081e-03,
    -2.129539999999999905e-02,
    -2.435661999999999897e-02,
    -3.487007999999999774e-02,
    -2.435661999999999897e-02,
    -2.129539999999999905e-02,
    -7.272540000000000081e-03,
    -4.942499999999999845e-03,
    -7.889389999999999598e-03,
    -7.623409999999999735e-03,
    -2.435661999999999897e-02,
    -1.730465999999999949e-02,
    -3.158604999999999746e-02,
    -1.730465999999999949e-02,
    -2.435661999999999897e-02,
    -7.623409999999999735e-03,
    -7.889389999999999598e-03,
    -1.009472999999999968e-02,
    -9.091949999999999685e-03,
    -3.487007999999999774e-02,
    -3.158604999999999746e-02,
    9.464194999999999691e-01,
    -3.158604999999999746e-02,
    -3.487007999999999774e-02,
    -9.091949999999999685e-03,
    -1.009472999999999968e-02,
    -7.889389999999999598e-03,
    -7.623409999999999735e-03,
    -2.435661999999999897e-02,
    -1.730465999999999949e-02,
    -3.158604999999999746e-02,
    -1.730465999999999949e-02,
    -2.435661999999999897e-02,
    -7.623409999999999735e-03,
    -7.889389999999999598e-03,
    -4.942499999999999845e-03,
    -7.272540000000000081e-03,
    -2.129539999999999905e-02,
    -2.435661999999999897e-02,
    -3.487007999999999774e-02,
    -2.435661999999999897e-02,
    -2.129539999999999905e-02,
    -7.272540000000000081e-03,
    -4.942499999999999845e-03,
    -1.738640000000000064e-03,
    -4.625149999999999748e-03,
    -7.272540000000000081e-03,
    -7.623409999999999735e-03,
    -9.091949999999999685e-03,
    -7.623409999999999735e-03,
    -7.272540000000000081e-03,
    -4.625149999999999748e-03,
    -1.738640000000000064e-03,
    -9.066000000000000274e-05,
    -1.738640000000000064e-03,
    -4.942499999999999845e-03,
    -7.889389999999999598e-03,
    -1.009472999999999968e-02,
    -7.889389999999999598e-03,
    -4.942499999999999845e-03,
    -1.738640000000000064e-03,
    -9.066000000000000274e-05,
];

/// `f64` `corrDn` — correlate with `reflect1` borders, keep every `step`-th
/// sample starting at 0. Filter origin is its centre `fs/2`.
#[must_use]
fn corr_dn64(
    image: &[f64],
    width: usize,
    height: usize,
    filter: &[f64],
    fs: usize,
    step: usize,
) -> Band3 {
    assert_eq!(image.len(), width * height, "corr_dn64: size");
    assert_eq!(filter.len(), fs * fs);
    let out_w = width.div_ceil(step);
    let out_h = height.div_ceil(step);
    let c = (fs / 2) as isize;
    let mut out = vec![0.0; out_w * out_h];
    for oy in 0..out_h {
        let iy0 = oy * step;
        for ox in 0..out_w {
            let ix0 = ox * step;
            let mut acc = 0.0;
            for ky in 0..fs {
                let sy = reflect1(iy0 as isize + ky as isize - c, height);
                let row = &image[sy * width..sy * width + width];
                for kx in 0..fs {
                    let sx = reflect1(ix0 as isize + kx as isize - c, width);
                    acc += filter[ky * fs + kx] * row[sx];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }
    Band3 {
        width: out_w,
        height: out_h,
        data: out,
    }
}

/// `f64` `upConv`: zero-upsample `band` by `step` onto the
/// `out_width × out_height` lattice and correlate with `filter` under
/// `reflect1` borders, accumulating into `res`.
fn up_conv64(
    band: &Band3,
    filter: &[f64],
    fs: usize,
    step: usize,
    out_width: usize,
    out_height: usize,
    res: &mut [f64],
) {
    assert_eq!(filter.len(), fs * fs);
    assert_eq!(res.len(), out_width * out_height);
    assert_eq!(
        (band.width, band.height),
        (out_width.div_ceil(step), out_height.div_ceil(step)),
        "up_conv64: band size vs out size"
    );
    let c = (fs / 2) as isize;
    let sample = |y: isize, x: isize| -> f64 {
        let y = reflect1(y, out_height);
        let x = reflect1(x, out_width);
        if !y.is_multiple_of(step) || !x.is_multiple_of(step) {
            return 0.0;
        }
        band.data[(y / step) * band.width + (x / step)]
    };
    for py in 0..out_height {
        for px in 0..out_width {
            let mut acc = 0.0;
            for ky in 0..fs {
                for kx in 0..fs {
                    acc += filter[ky * fs + kx]
                        * sample(py as isize + ky as isize - c, px as isize + kx as isize - c);
                }
            }
            res[py * out_width + px] += acc;
        }
    }
}

/// A single pyramid plane.
#[derive(Debug, Clone)]
pub struct Band3 {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row-major samples — the **stored** (un-normalised) coefficients.
    pub data: Vec<f64>,
}

/// The sp0 pyramid of one photoreceptor-response image.
///
/// `bands[0]` is the residual high-pass at full resolution, `bands[1..=H]`
/// the isotropic bandpass levels, `bands[H+1]` the low-pass baseband.
/// [`Self::band`] applies the `2^-b` normalisation upstream's `get_band`
/// uses; [`Self::set_band`] scales by `2^b` on write, matching `set_band`.
#[derive(Debug, Clone)]
pub struct Spyr0 {
    bands: Vec<Band3>,
    /// `2^-(b+1) · ppd` per band — upstream `get_freqs`.
    freqs: Vec<f64>,
}

impl Spyr0 {
    /// Decompose `image` (`width × height` row-major) into the sp0 pyramid.
    ///
    /// `levels = min(ceil(log2(ppd)) − 2, maxPyrHt(img, 13))` — upstream's
    /// rule, where 13 is the `lofilt` size. Returns `None` when the image is
    /// too small for even one level (min dimension < 13).
    #[must_use]
    pub fn decompose(image: &[f64], width: usize, height: usize, ppd: f64) -> Option<Self> {
        assert_eq!(image.len(), width * height, "spyr0: size");
        let max_ht = max_pyr_height(width, height, 13);
        let levels = (ppd.log2().ceil() as i64 - 2).min(max_ht as i64);
        if levels < 0 {
            return None;
        }
        let levels = levels as usize;

        let mut bands = Vec::with_capacity(levels + 2);
        bands.push(corr_dn64(image, width, height, &HI0FILT, 9, 1));
        let mut lo = corr_dn64(image, width, height, &LO0FILT, 7, 1);
        for _ in 0..levels {
            bands.push(corr_dn64(&lo.data, lo.width, lo.height, &BFILT, 9, 1));
            lo = corr_dn64(&lo.data, lo.width, lo.height, &LOFILT, 13, 2);
        }
        bands.push(lo);

        let freqs = (0..bands.len())
            .map(|b| ppd / 2f64.powi(b as i32 + 1))
            .collect();
        Some(Self { bands, freqs })
    }

    /// Number of bands (`levels + 2`).
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Band frequencies in cycles/degree, one per band.
    #[must_use]
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// `(width, height)` of band `b`.
    #[must_use]
    pub fn band_dims(&self, b: usize) -> (usize, usize) {
        (self.bands[b].width, self.bands[b].height)
    }

    /// Band `b`'s coefficients, normalised by `2^-b` (upstream `get_band`).
    #[must_use]
    pub fn band(&self, b: usize) -> Vec<f64> {
        let norm = 2f64.powi(b as i32);
        self.bands[b].data.iter().map(|&v| v / norm).collect()
    }

    /// Write `data` (already in the normalised domain) into band `b`,
    /// storing `data · 2^b` (upstream `set_band`).
    pub fn set_band(&mut self, b: usize, data: &[f64]) {
        assert_eq!(data.len(), self.bands[b].data.len(), "set_band size");
        let norm = 2f64.powi(b as i32);
        for (d, &v) in self.bands[b].data.iter_mut().zip(data) {
            *d = v * norm;
        }
    }

    /// `reconSpyr(pyr, pind, 'sp0Filters', 'reflect1', levs, 'all')`.
    ///
    /// `levs` names pyramid *levels*: `0` is the residual high-pass,
    /// `1..=H` the isotropic bands, `H+1` the low-pass. A level contributes
    /// only when present in `levs`; deeper levels are chained through the
    /// low-pass reconstruction whenever `levs` contains a level deeper than
    /// the current one — mirroring the reference recursion exactly.
    ///
    /// The metric only ever calls this with `levs = 0..H` (everything but
    /// the last bandpass level and the low-pass) to build the masking
    /// transform.
    #[must_use]
    pub fn reconstruct(&self, levs: &[usize]) -> Vec<f64> {
        let (w0, h0) = self.band_dims(0);
        let n_levels = self.bands.len() - 2; // H
        let lowpass_idx = self.bands.len() - 1;
        let has = |l: usize| levs.contains(&l);

        // reconSpyrLevs: `level` is the 1-based level number, equal to the
        // band index for sp0. Level `k` descends iff some requested level
        // is deeper than `k`; at the bottom call (`level == H`) the source
        // is the low-pass row.
        fn rec(me: &Spyr0, level: usize, has: &dyn Fn(usize) -> bool) -> Band3 {
            let (bw, bh) = me.band_dims(level);
            let mut acc = vec![0.0; bw * bh];
            if (level + 1..=me.bands.len() - 1).any(has) {
                let src = if level < me.bands.len() - 2 {
                    rec(me, level + 1, has)
                } else {
                    me.bands[me.bands.len() - 1].clone() // low-pass row
                };
                up_conv64(&src, &LOFILT, 13, 2, bw, bh, &mut acc);
            }
            if has(level) {
                up_conv64(&me.bands[level].clone(), &BFILT, 9, 1, bw, bh, &mut acc);
            }
            Band3 {
                width: bw,
                height: bh,
                data: acc,
            }
        }

        let res1 = if n_levels == 0 {
            // `spyrHt == 0`: `res1 = pyrBand(2)` (the low-pass row) iff
            // level 1 requested, else a zero map.
            if has(1) {
                self.bands[lowpass_idx].clone()
            } else {
                Band3 {
                    width: w0,
                    height: h0,
                    data: vec![0.0; w0 * h0],
                }
            }
        } else {
            rec(self, 1, &has)
        };

        let mut out = vec![0.0; w0 * h0];
        up_conv64(&res1, &LO0FILT, 7, 1, w0, h0, &mut out);
        if has(0) {
            up_conv64(&self.bands[0].clone(), &HI0FILT, 9, 1, w0, h0, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(w: usize, h: usize) -> Vec<f64> {
        (0..w * h)
            .map(|i| ((i / w + i % w) % 2) as f64 * 40.0 + (i % 7) as f64 * 0.5)
            .collect()
    }

    #[test]
    fn decompose_reconstruct_roundtrip() {
        let (w, h) = (96, 96);
        let im = checker(w, h);
        let p = Spyr0::decompose(&im, w, h, 30.0).unwrap();
        let levs: Vec<usize> = (0..p.band_count()).collect();
        let recon = p.reconstruct(&levs);
        // sp0 filters are not a perfect-reconstruction set at the borders,
        // but the interior should come back within a few % of a 40-unit
        // checker amplitude.
        let mut worst = 0.0f64;
        for y in 8..h - 8 {
            for x in 8..w - 8 {
                worst = worst.max((recon[y * w + x] - im[y * w + x]).abs());
            }
        }
        assert!(worst < 2.0, "interior reconstruction error {worst}");
    }

    #[test]
    fn freqs_halve_per_band() {
        let im = vec![1.0; 64 * 64];
        let p = Spyr0::decompose(&im, 64, 64, 30.0).unwrap();
        for (b, &f) in p.freqs().iter().enumerate() {
            assert!((f - 30.0 / 2f64.powi(b as i32 + 1)).abs() < 1e-12);
        }
    }
}
