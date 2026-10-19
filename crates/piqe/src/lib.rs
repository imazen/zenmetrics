// Port of pypiqe 1.2 (https://github.com/michael-rutherford/pypiqe), which is
// itself a port of MATLAB's `piqe`.
//
// pypiqe is MIT licensed, Copyright (c) 2023 Michael Rutherford; that notice
// is kept in `LICENSE-pypiqe` beside this crate's `Cargo.toml`.
//
// Reference:
//   Venkatanath N., Praneeth D., Maruthi Chandrasekhar Bh.,
//   Channappayya S. S., Medasani S. S., "Blind Image Quality Evaluation
//   Using Perception Based Features", Twenty First National Conference on
//   Communications (NCC), 2015, pp. 1-6. doi:10.1109/NCC.2015.7084843

//! PIQE — Perception-based Image Quality Evaluator (no-reference).
//!
//! A pure-Rust port of [`pypiqe`](https://github.com/michael-rutherford/pypiqe)
//! 1.2 (MIT), a Python conversion of MATLAB's `piqe`. PIQE scores an image on
//! `[0, 100]` — higher means worse — by dividing it into 16×16 blocks,
//! selecting the spatially active ones (normalized variance above 0.1), and
//! pooling a distortion estimate per active block (blockiness along the block
//! edges and Gaussian noise inside the block).
//!
//! Conventions matched to the oracle (all confirmed against
//! `pypiqe==1.2` + `opencv-python`):
//!
//! - RGB8 → gray is OpenCV's `cv2.cvtColor(COLOR_BGR2GRAY)` on the loaded
//!   BGR image, which computes `Y = (3735·B + 19235·G + 9798·R + 16384) >> 15`
//!   in 15-bit fixed point — bit-exact for all u8 inputs. GRAY8 input takes
//!   the plane verbatim (the oracle's 2-D path).
//! - Images whose side is not a multiple of 16 are bottom/right padded with
//!   `numpy`'s `symmetric` mode (edge-inclusive mirror).
//! - The gray plane is then rescaled to `[0, 255]` by `round(255·v/max)`
//!   (ties to even); an all-zero image normalizes to all-NaN and scores 100.
//! - Local mean/std come from a 7-tap Gaussian, σ = 7/6, replicate borders,
//!   evaluated in f64; `imnorm = (im − mu) / (sigma + 1)`.
//! - A block is spatially active when its 256-sample `ddof=1` variance of
//!   `imnorm` exceeds 0.1. It is *impaired* when any of the 44 overlapping
//!   6-element edge segments (11 per edge) has `ddof=1` std below 0.1. It is
//!   *noisy* when `blockSigma > 2·blockBeta` with
//!   `blockBeta = |σ − csd| / max(σ, csd)` and `csd` the center/surround std
//!   ratio; `csd = 0` when the ratio is NaN. The oracle's `centerSurDev`
//!   deletes column 7, then column 8 of the *already-deleted* block, so the
//!   surround keeps original column 8 and drops columns 7 and 9 — a quirk of
//!   the Python port reproduced here faithfully.
//! - Score `= (Σ distortion + 1) / (NHSA + 1) · 100`.
//!
//! One divergence is deliberate: the oracle accepts images of any size and
//! pads them up to 16 on a side; this port reports [`ErrorKind::TooSmall`]
//! below 16×16 (one real block) rather than scoring padding-manufactured
//! content.
//!
//! [`ErrorKind::TooSmall`]: zenmetrics_abstractions::ErrorKind::TooSmall

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use za::zenpixels::PixelDescriptor;
use zenmetrics_abstractions as za;

mod kernels;

/// Output revision of [`Piqe`] on the generic surface; bump when any output
/// can change.
const SCORE_REVISION: u32 = 1;

/// Analysis block size; the image is padded to a multiple of it.
const BLOCK: usize = 16;
/// Segment length along a block edge for the impairment test.
const SEGMENT: usize = 6;
/// Segments per edge: `BLOCK - SEGMENT + 1`.
const NSEG: usize = BLOCK - SEGMENT + 1;
/// `blockVar` gate for "high spatial activity" (`activityThreshold`).
const ACTIVITY_THRESHOLD: f64 = 0.1;
/// Per-segment std floor below which the block is impaired
/// (`blockImpairedThreshold`).
const IMPAIRED_THRESHOLD: f64 = 0.1;

/// 1-D Gaussian taps of `cv2.getGaussianKernel(7, 7.0/6.0)`, written as the
/// exact f64 bit patterns OpenCV produces (`exp(-(i−3)²/(2σ²))`, normalized
/// by reciprocal-multiply), so `mu`/`sigma` match the oracle's
/// `cv2.GaussianBlur(ksize=(7,7), sigmaX=7/6)` to within accumulation-order
/// ulps.
const KERNEL: [f64; 7] = [
    f64::from_bits(0x3f89_b929_91f2_487f),
    f64::from_bits(0x3fb4_2e11_ca51_7a5f),
    f64::from_bits(0x3fce_5fb7_c557_fad1),
    f64::from_bits(0x3fd5_edac_bc60_2377),
    f64::from_bits(0x3fce_5fb7_c557_fad1),
    f64::from_bits(0x3fb4_2e11_ca51_7a5f),
    f64::from_bits(0x3f89_b929_91f2_487f),
];

/// Estimated just-noticeable difference of the `piqe` score, measured on the
/// KonJND-1k corpus: `zenmetrics batch --metric piqe` over all 1,008 pairs
/// (`PairedNoReference` gives `piqe` and `piqe_ref` per pair), then JND =
/// median over pairs of `|piqe − piqe_ref|` = 12.486448. See
/// `benchmarks/nr_classic_2026-10-15/jnd.md` for the full distribution.
pub const JND: f64 = 12.486448;

/// The PIQE metric as a generic no-reference scorer — accepts packed
/// `RGB8_SRGB` and `GRAY8`. Headline = `piqe`
/// ([`ScoreKind::Distortion`](za::ScoreKind::Distortion): higher is worse).
#[derive(Debug, Default, Clone, Copy)]
pub struct Piqe;

impl Piqe {
    /// A new PIQE scorer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl za::NoReferenceMetric for Piqe {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("piqe", za::implementation!(SCORE_REVISION))
            .with_reference("pypiqe 1.2 (MATLAB piqe port)")
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new("piqe", za::ScoreKind::Distortion)];
        Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        let d = format.packed();
        d == Some(PixelDescriptor::RGB8_SRGB) || d == Some(PixelDescriptor::GRAY8)
    }

    fn evaluate_image(
        &mut self,
        image: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        opts.checkpoint()?;
        opts.refuse_map("piqe")?;
        let (gray, width, height) = gray_u8(&image)?;
        if width < BLOCK || height < BLOCK {
            return za::fail(
                za::MetricError::new(za::ErrorKind::TooSmall)
                    .with_metric("piqe")
                    .with_message(alloc::format!(
                        "image too small for PIQE: {width}x{height} \
                         (the analysis unit is a {BLOCK}x{BLOCK} block)"
                    )),
            );
        }
        let score = score_gray_u8(&gray, width, height);
        Ok(Box::new(za::ScalarEval::single("piqe", score)))
    }
}

/// Borrow the packed slice when its descriptor is `desc`.
fn packed_as<'a>(
    image: &'a za::Image<'a>,
    desc: PixelDescriptor,
) -> Option<(&'a [u8], usize, usize, usize)> {
    let za::Image::Packed(s) = image else {
        return None;
    };
    if s.descriptor() != desc {
        return None;
    }
    Some((
        s.as_strided_bytes(),
        s.width() as usize,
        s.rows() as usize,
        s.stride(),
    ))
}

/// The oracle's gray plane: OpenCV `COLOR_BGR2GRAY` on the BGR load of an
/// RGB8 image (15-bit fixed point, bit-exact vs `cv2.cvtColor`), or the
/// `GRAY8` plane verbatim.
fn gray_u8(image: &za::Image<'_>) -> za::MetricResult<(Vec<u8>, usize, usize)> {
    if let Some((px, w, h, stride)) = packed_as(image, PixelDescriptor::RGB8_SRGB) {
        let mut gray = Vec::with_capacity(w * h);
        for y in 0..h {
            let row = &px[y * stride..y * stride + w * 3];
            for x in 0..w {
                let (r, g, b) = (
                    u32::from(row[3 * x]),
                    u32::from(row[3 * x + 1]),
                    u32::from(row[3 * x + 2]),
                );
                gray.push(((3735 * b + 19235 * g + 9798 * r + 16384) >> 15) as u8);
            }
        }
        return Ok((gray, w, h));
    }
    if let Some((px, w, h, stride)) = packed_as(image, PixelDescriptor::GRAY8) {
        let mut gray = Vec::with_capacity(w * h);
        for y in 0..h {
            gray.extend_from_slice(&px[y * stride..y * stride + w]);
        }
        return Ok((gray, w, h));
    }
    za::fail(za::unsupported(
        &image.format(),
        "packed RGB8_SRGB or GRAY8",
    ))
}

/// `np.pad(ipImage, ((0, p), (0, p)), mode='symmetric')` index mapping:
/// the edge-inclusive mirror. `pad < size` always holds here (`pad ≤ 15`
/// and `size ≥ 16`), so one reflection suffices.
#[inline]
fn sym(i: usize, size: usize) -> usize {
    if i < size { i } else { 2 * size - 1 - i }
}

// `cv2.GaussianBlur(src, ksize=(7,7), sigmaX=7/6, BORDER_REPLICATE)` on a
// f64 plane: OpenCV's separable `sepFilter2D` — a row filter over columns
// then a column filter over rows, both sampling clamped indices.
// Implemented as `kernels::gauss7_pair` over edge-replicated pad planes:
// the k = 0..7 tap order and f64 mul+add two-rounding are preserved
// bitwise on every dispatch tier.

/// `np.var(v, ddof=1)`.
fn var1(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let dev = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>();
    dev / (n - 1.0)
}

// Correctly-rounded float ops routed through libm under `no_std` (sqrt and
// rint are exact in both implementations, so parity holds).
#[inline(always)]
fn sqrt64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}
/// `round_ties_even` = C `rint` under the default rounding mode.
#[inline(always)]
fn rint64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.round_ties_even()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::rint(x)
    }
}

/// `np.std(v, ddof=1)`.
fn std1(v: &[f64]) -> f64 {
    sqrt64(var1(v))
}

/// One block edge (16 elements) through `at`: impaired when any of its 11
/// overlapping 6-element segments has `std < IMPAIRED_THRESHOLD`.
fn edge_impaired(at: impl Fn(usize) -> f64) -> bool {
    let mut seg = [0.0f64; SEGMENT];
    for i in 0..NSEG {
        for (s, v) in seg.iter_mut().enumerate() {
            *v = at(i + s);
        }
        if std1(&seg) < IMPAIRED_THRESHOLD {
            return true;
        }
    }
    false
}

/// pypiqe's `noticeDistCriterion`: a block is impaired when any of the 44
/// overlapping 6-element segments on its four edges has std below
/// `blockImpairedThreshold`.
fn block_impaired(b: &[f64; 256]) -> bool {
    edge_impaired(|i| b[i])              // top:    row 0
        || edge_impaired(|i| b[i * BLOCK + 15]) // right:  col 15
        || edge_impaired(|i| b[15 * BLOCK + i]) // bottom: row 15
        || edge_impaired(|i| b[i * BLOCK]) // left:   col 0
}

/// pypiqe's `noiseCriterion`/`centerSurDev`: `(blockSigma, blockBeta)` where
/// `blockBeta = |σ − csd| / max(σ, csd)` and `csd` is the center/surround
/// std ratio (NaN → 0). The oracle's sequential `np.delete`s drop original
/// columns 7 and 9 — column 8 stays in the surround — which this reproduces.
fn noise_criterion(b: &[f64; 256], block_var: f64) -> (f64, f64) {
    let block_sigma = sqrt64(block_var);
    let mut center = [0.0f64; 32];
    let mut surround = [0.0f64; 224];
    for r in 0..BLOCK {
        center[2 * r] = b[r * BLOCK + 7];
        center[2 * r + 1] = b[r * BLOCK + 8];
        let mut i = r * 14;
        for c in 0..BLOCK {
            if c == 7 || c == 9 {
                continue;
            }
            surround[i] = b[r * BLOCK + c];
            i += 1;
        }
    }
    let csd = std1(&center) / std1(&surround);
    let csd = if csd.is_nan() { 0.0 } else { csd };
    let beta = (block_sigma - csd).abs() / block_sigma.max(csd);
    (block_sigma, beta)
}

/// PIQE score of a tightly packed gray u8 plane.
fn score_gray_u8(gray: &[u8], w: usize, h: usize) -> f64 {
    let (pw, ph) = (
        w + (BLOCK - w % BLOCK) % BLOCK,
        h + (BLOCK - h % BLOCK) % BLOCK,
    );

    // np.round(255 * (ipImage / np.max(ipImage))) on the symmetric-padded
    // plane (padding replicates edge values, so the max is the input max).
    let max = gray.iter().copied().max().unwrap_or(0);
    let mut im = vec![0.0f64; pw * ph];
    for (y, row) in im.chunks_exact_mut(pw).enumerate() {
        let sy = sym(y, h);
        for (x, v) in row.iter_mut().enumerate() {
            *v = rint64((f64::from(gray[sy * w + sym(x, w)]) / f64::from(max)) * 255.0);
        }
    }

    // mu = G*im, sigma = sqrt(|G*(im*im) − mu*mu|), imnorm = (im − mu)/(sigma+1)
    let (mu, e2) = kernels::gauss7_pair(&im, pw, ph, &KERNEL);
    let norm = kernels::normalize(&im, &mu, &e2, pw * ph);

    let mut nhsa = 0u64;
    let mut dist_scores = 0.0f64;
    for by in (0..ph).step_by(BLOCK) {
        for bx in (0..pw).step_by(BLOCK) {
            let mut b = [0.0f64; 256];
            for r in 0..BLOCK {
                for c in 0..BLOCK {
                    b[r * BLOCK + c] = norm[(by + r) * pw + bx + c];
                }
            }
            let bvar = var1(&b);
            if bvar > ACTIVITY_THRESHOLD {
                nhsa += 1;
                let wndc = f64::from(u8::from(block_impaired(&b)));
                let (bsig, bbeta) = noise_criterion(&b, bvar);
                let wnc = f64::from(u8::from(bsig > 2.0 * bbeta));
                dist_scores += wndc * (1.0 - bvar) + wnc * bvar;
            }
        }
    }
    (dist_scores + 1.0) / (1.0 + nhsa as f64) * 100.0
}
