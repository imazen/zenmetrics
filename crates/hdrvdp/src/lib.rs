#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! HDR-VDP-2.2 — a pure-Rust `f64` CPU reference port.
//!
//! HDR-VDP-2 (Mantiuk, Kim, Rempel & Heidrich, SIGGRAPH 2011; quality
//! recalibrated by Narwaria et al. 2015 as 2.2) predicts **visibility** — the
//! per-pixel probability that a human notices a difference — and **quality** —
//! a single mean-opinion-score correlate — for image pairs given in *absolute*
//! luminance, across the full range from starlight to sunlight. It is the
//! field's reference HDR metric: 0.936 SROCC on AIC-HDR2025 and 0.812 on UPIQ,
//! above every metric currently in this workspace.
//!
//! ## Status: chunk 2 of 6
//!
//! This crate is being landed in chunks (see imazen/zenmetrics#50). **What is
//! here now**, complete and unit-tested:
//!
//! | stage | module | status |
//! |---|---|---|
//! | display / colour encoding → cd/m² | [`display`] | ✅ |
//! | optical MTF | [`csf`] + [`fft`] | ✅ |
//! | photoreceptor spectral sensitivity | [`spectral`] | ✅ |
//! | photoreceptor non-linearity (JND space) | [`photoreceptor`] | ✅ |
//! | neural CSF | [`csf`] | ✅ |
//! | achromatic response (stages 1–5) | [`pathway`] | ✅ |
//! | steerable-pyramid decomposition | [`spyr`] | ✅ |
//! | contrast masking | — | ⏳ chunk 2b |
//! | probability pooling, `P_det` / `P_map` | — | ⏳ chunk 3 |
//! | quality correlate `Q` / `Q_MOS` | — | ⏳ chunk 3 |
//!
//! One boundary caveat is carried openly rather than papered over: the
//! pyramid's *synthesis* boundary rule is principled and self-consistent but
//! has not been compared against upstream's C implementation. It affects only
//! the visibility map's border pixels, never `Q` / `Q_MOS`. See [`spyr`].
//!
//! **There is no end-to-end score yet.** Nothing in this crate should be
//! reported as an HDR-VDP-2 number until chunk 4 lands and its UPIQ SROCC is
//! recorded in `benchmarks/`.
//!
//! ## Units
//!
//! Every luminance in this crate is **absolute, in cd/m² (nits)** — the same
//! currency `zenmetrics_api::hdr` speaks. Feeding relative or normalised values
//! silently scores at the wrong adaptation level; [`display::looks_relative`]
//! detects the common form of that mistake.
//!
//! ## Licensing
//!
//! Ported from the ISC-licensed HDR-VDP-2.2 MATLAB release, with three
//! upstream helpers deliberately **not** ported (they carry no permission
//! grant) and reimplemented in [`fft`] / [`spectral`] instead. The steerable
//! pyramid comes from the MIT-licensed matlabPyrTools. See
//! `THIRD-PARTY-NOTICES.md`.

pub mod csf;
pub mod display;
pub mod fft;
pub mod interp;
pub mod params;
pub mod pathway;
pub mod photoreceptor;
pub mod sp3_filters;
pub mod spectral;
pub mod spyr;

pub use display::ColorEncoding;
pub use params::{Params, pix_per_deg};
pub use pathway::{Pathway, visual_pathway};
pub use photoreceptor::Photoreceptor;
pub use spyr::{Band, SteerablePyramid};

/// Errors this crate can return.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The pixel buffer length does not match `width × height × channels`.
    ChannelMismatch {
        /// The length implied by the dimensions and colour encoding.
        expected: usize,
        /// The length actually supplied.
        got: usize,
    },
    /// `width × height × channels` overflowed `usize`.
    DimensionOverflow,
    /// The reference and distorted images differ in size.
    SizeMismatch {
        /// Reference dimensions, `(width, height)`.
        reference: (usize, usize),
        /// Distorted dimensions, `(width, height)`.
        distorted: (usize, usize),
    },
    /// A pixel value was not finite after the display model.
    ImpossibleValues(f64),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ChannelMismatch { expected, got } => write!(
                f,
                "pixel buffer has {got} values, expected {expected} for these dimensions and colour encoding"
            ),
            Self::DimensionOverflow => write!(f, "width × height × channels overflows usize"),
            Self::SizeMismatch {
                reference,
                distorted,
            } => write!(
                f,
                "reference is {}×{} but distorted is {}×{}",
                reference.0, reference.1, distorted.0, distorted.1
            ),
            Self::ImpossibleValues(v) => {
                write!(f, "non-finite luminance after the display model: {v}")
            }
        }
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_name_the_numbers() {
        let e = Error::ChannelMismatch {
            expected: 12,
            got: 7,
        };
        let s = e.to_string();
        assert!(s.contains("12") && s.contains('7'), "{s}");
        let e = Error::SizeMismatch {
            reference: (4, 3),
            distorted: (4, 5),
        };
        let s = e.to_string();
        assert!(s.contains("4×3") && s.contains("4×5"), "{s}");
    }
}
