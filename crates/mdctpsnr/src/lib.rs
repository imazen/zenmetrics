//! Pure-Rust CPU reimplementation of **mDCT-PSNR** — the masked-DCT
//! perceptual image metric of Thomas Richter (University of Stuttgart),
//! distribution `mDCTpsnr` (`dctpsnr` CLI). This port is derived from the
//! reference C++ implementation (zlib license, see below) in its default
//! build configuration: `LINEAR` color path, `KNEE_VALUE` gamma knee,
//! `AHUMADA` detection (`exp(-err^3.5)`), `EXTENDED_FILTER` DC low-pass
//! split, `BASE_VISIBILITY 0.08`, no `WEIGHT_MSE`/`WEIGHT_DELTA_E`/`USE_PSI`.
//!
//! > mDCTpsnr reference license (zlib), retained per its terms; this Rust
//! > file is an altered version and is not the original work:
//! >
//! > Copyright (C) 2009 Thomas Richter, University of Stuttgart.
//! > This software is provided 'as-is', without any express or implied
//! > warranty. Permission is granted to anyone to use this software for any
//! > purpose, including commercial applications, and to alter it and
//! > redistribute it freely, subject to: the origin of this software must
//! > not be misrepresented; altered source versions must be plainly marked
//! > as such and must not be misrepresented as being the original software;
//! > this notice may not be removed or altered.
//!
//! # Pipeline (matching `dctpsnr ref.ppm dist.ppm` default output)
//!
//! 1. sRGB-8 pixels → sRGB-linear RGB (exact EOTF LUT) → **linear** BT.601
//!    YCbCr; only Y is re-encoded with the sRGB gamma knee, Cb/Cr stay
//!    linear (and signed, ±0.5).
//! 2. Per component, sliding 8×8 windowed DCT at every position: the block
//!    is windowed toward its own mean, `b·w + avg·(1−w)`, with the 8×8
//!    `window_table`, then a two-pass scaled AAN float DCT. Band planes are
//!    `(W−8)×(H−7)`; the last possible column block (`i = W−8`) is skipped
//!    by the reference loop bound `i < w−8` and skipped here too.
//! 3. Per (band,component) a 5×5 separable mask: `mapped = |c·vis|^expon`,
//!    symmetric cosine-normalized kernel, `mask = 0.08/(0.08 + Σ5×5/25)`.
//!    The output coefficient is the centre coeff scaled by `vis` (a
//!    per-band constant folding in the AAN scale factors, the `norms`
//!    window-orthogonality correction, and the Ahumada/JPEG CSF tables).
//! 4. Per measured position, `errorline[j]` accumulates the product of
//!    per-band non-detection probabilities `exp(-(|ref−dst|·max(mask)·vb)^3.5)`
//!    over 195 terms (64 bands × 3 comps, with the Y DC band measured on a
//!    4-way 5×5 low/high split instead of its own coefficient and with a
//!    mask borrowed from the (0,1)/(1,0)/(1,1) bands for both images).
//! 5. `score_dB = −20·log10(error/(w·h·comps))` with `w = W−13`,
//!    `h = H−13`, `comps = 3` — note `h` in the denominator is one less
//!    than the number of measured lines (`H−12`); that is the reference's
//!    own normalization and is replicated deliberately.
//!
//! Validated against the built `dctpsnr` binary (goldens in `mod tests`)
//! and the published AIC-4 `mDCT-PSNR` column (dB scale).

#![cfg_attr(not(feature = "std"), no_std)]
// Bit-parity with the reference build requires the literal constants to be
// transcribed verbatim (their f32 values, not nearest-typed fractions), and
// several index-driven loops encode a specific accumulation order decoded
// from the compiled AVX2 binary — both look like clippy noise but are
// deliberate.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

extern crate alloc;

use alloc::vec::Vec;

/// Errors from [`mdct_psnr_srgb8`].
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// `reference.len()` or `distorted.len()` != `width * height * 3`.
    LengthMismatch,
    /// The metric needs `W−13 > 0` mask width and `H > 12` lines
    /// (12-line warmup + ≥1 measured line); we require `width >= 14` and
    /// `height >= 14`.
    ImageTooSmall,
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Stable column-name identifier for sweep sidecars:
/// `mdctpsnr_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time
/// via `MDCTPSNR_IMPL_TAG`). No `_cpu_` infix — no GPU twin exists.
pub const MDCTPSNR_COLUMN_NAME: &str = match option_env!("MDCTPSNR_IMPL_TAG") {
    Some(tag) => tag,
    None => concat!(
        "mdctpsnr_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::LengthMismatch => {
                f.write_str("interleaved RGB8 buffer length != width*height*3")
            }
            Error::ImageTooSmall => f.write_str("image must be at least 14x14"),
        }
    }
}

// ---------------------------------------------------------------------------
// Constants — transcribed from the reference (dct/component.cpp,
// measure/pooling.cpp).
// ---------------------------------------------------------------------------

/// Scaled-AAN DCT normalization factors (pooling.cpp `dct_scale`).
#[rustfmt::skip]
const DCT_SCALE: [f32; 8] = [
    1.0, 1.387039845, 1.306562965, 1.175875602,
    1.0, 0.785694958, 0.541196100, 0.275899379,
];

/// Per-band L2 norm squared of the windowed transform on constant-frequency
/// input (pooling.cpp `norms`). Transpose-symmetric.
#[rustfmt::skip]
const NORMS: [[f32; 8]; 8] = [
    [1.0,     0.742298, 1.22273,  1.24434,  1.2528,   1.26126,  1.28286,  1.76329],
    [0.742298,0.439822, 0.724487, 0.737285, 0.742298, 0.747311, 0.760109, 1.04477],
    [1.22273, 0.724487, 1.19339,  1.21448,  1.22273,  1.23099,  1.25207,  1.72098],
    [1.24434, 0.737285, 1.21448,  1.23593,  1.24434,  1.25274,  1.27419,  1.75139],
    [1.2528,  0.742298, 1.22273,  1.24434,  1.2528,   1.26126,  1.28286,  1.76329],
    [1.26126, 0.747311, 1.23099,  1.25274,  1.26126,  1.26977,  1.29152,  1.7752 ],
    [1.28286, 0.760109, 1.25207,  1.27419,  1.28286,  1.29152,  1.31364,  1.8056 ],
    [1.76329, 1.04477,  1.72098,  1.75139,  1.76329,  1.7752,   1.8056,   2.48181],
];

/// Ahumada/Watson luma quantization table (`#ifdef AHUMADA` branch of
/// pooling.cpp).
#[rustfmt::skip]
const LUMA_TBL: [u32; 64] = [
    25,  11,  11,  12,  15,  19,  25,  32,
    11,  13,  10,  10,  12,  15,  19,  24,
    11,  10,  14,  14,  16,  18,  22,  27,
    12,  10,  14,  18,  21,  24,  28,  33,
    15,  12,  16,  21,  26,  31,  36,  42,
    19,  15,  18,  24,  31,  38,  45,  53,
    25,  19,  22,  28,  36,  45,  55,  65,
    32,  24,  27,  33,  42,  53,  65,  77,
];

#[rustfmt::skip]
const CR_TBL: [u32; 64] = [
    21,  21,  41,  45,  55,  71,  92, 120,
    21,  37,  39,  38,  44,  55,  70,  89,
    41,  39,  51,  54,  59,  69,  83, 103,
    45,  38,  54,  69,  80,  91, 106, 126,
    55,  44,  59,  80, 100, 117, 136, 158,
    71,  55,  69,  91, 117, 144, 170, 198,
    92,  70,  83, 106, 136, 170, 206, 243,
    120,  89, 103, 126, 158, 198, 243, 290,
];

#[rustfmt::skip]
const CB_TBL: [u32; 64] = [
    45,  43, 103, 114, 141, 181, 236, 306,
    43,  78,  99,  97, 113, 140, 178, 228,
    103,  99, 130, 138, 150, 175, 212, 262,
    114,  97, 138, 176, 203, 232, 270, 321,
    141, 113, 150, 203, 254, 299, 347, 403,
    181, 140, 175, 232, 299, 367, 434, 505,
    236, 178, 212, 270, 347, 434, 525, 619,
    306, 228, 262, 321, 403, 505, 619, 739,
];

/// 8×8 Hamming window (dct/component.cpp `window_table` — the literal
/// table, not a recomputed window). The block is windowed *toward its own
/// mean*: `b·w + avg·(1−w)`.
#[rustfmt::skip]
const WINDOW: [[f32; 8]; 8] = [
    [0.2328, 0.4375, 0.5894, 0.6702, 0.6702, 0.5894, 0.4375, 0.2328],
    [0.4375, 0.8222, 1.1077, 1.2596, 1.2596, 1.1077, 0.8222, 0.4375],
    [0.5894, 1.1077, 1.4924, 1.6971, 1.6971, 1.4924, 1.1077, 0.5894],
    [0.6702, 1.2596, 1.6971, 1.9298, 1.9298, 1.6971, 1.2596, 0.6702],
    [0.6702, 1.2596, 1.6971, 1.9298, 1.9298, 1.6971, 1.2596, 0.6702],
    [0.5894, 1.1077, 1.4924, 1.6971, 1.6971, 1.4924, 1.1077, 0.5894],
    [0.4375, 0.8222, 1.1077, 1.2596, 1.2596, 1.1077, 0.8222, 0.4375],
    [0.2328, 0.4375, 0.5894, 0.6702, 0.6702, 0.5894, 0.4375, 0.2328],
];

// AAN DCT constants (dct/component.cpp).
const SQRT2_2: f32 = 0.707106781;
const C1: f32 = 0.382683433;
const C2: f32 = 0.541196100;
const C3: f32 = 1.306562965;

const MASK_SIZE: usize = 5;
const BASE_VISIBILITY: f32 = 0.08;

// ---------------------------------------------------------------------------
// libm shim: under `std` we call the platform libm (`f32::powf` → `powf`,
// `f32::exp` → `expf`, …) which is bit-identical to the reference binary's
// glibc calls on Linux. Under `no_std` we fall back to the `libm` crate —
// same algorithms to ≤1–2 ulp, but not bit-identical to glibc. This is the
// only intentional source of cross-platform score drift.
// ---------------------------------------------------------------------------
#[inline(always)]
fn pf(a: f32, b: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.powf(b)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::powf(a, b)
    }
}
#[inline(always)]
fn ef(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::expf(x)
    }
}
/// glibc libmvec `_ZGVdN8vv_powf` — the 8-lane `powf` the reference's
/// `-ffast-math` build vectorizes the `mapped` loop into. Scalar `powf`
/// disagrees with it on ~5% of inputs by 1 ulp; on x86_64+glibc we call the
/// real vector routine through `asm!` (stable ymm_reg operands). Elsewhere
/// we fall back to scalar `powf` — documented in DIVERGENCES.md.
#[cfg(all(
    feature = "std",
    target_arch = "x86_64",
    target_os = "linux",
    target_env = "gnu"
))]
mod libmvec {
    use super::sq;
    use core::arch::x86_64::__m256;

    #[link(name = "mvec")]
    unsafe extern "C" {
        fn _ZGVdN8vv_powf();
        fn _ZGVdN8v_expf();
    }

    /// `expf` on 8 lanes, bit-identical to the reference's vectorized call
    /// (glibc's real SVML-style kernel — differs from scalar `expf` by
    /// 1 ulp on a substantial fraction of inputs; verified via direct call).
    ///
    /// # Safety
    /// Requires AVX2 at runtime.
    #[target_feature(enable = "avx2")]
    unsafe fn expf8(x: &[f32; 8]) -> [f32; 8] {
        let r: __m256;
        unsafe {
            core::arch::asm!(
                "vmovups ymm0, ymmword ptr [{x}]",
                "call {f}",
                x = in(reg) x.as_ptr(),
                f = sym _ZGVdN8v_expf,
                out("ymm0") r,
                clobber_abi("C"),
            );
        }
        unsafe { core::mem::transmute::<__m256, [f32; 8]>(r) }
    }

    /// `Pooling::MeasureInBand` with the AVX2 vector `expf`: 8-lane chunks,
    /// `err³·√err` per lane, lanes with `err ≤ 0` keep their errorline value.
    pub fn measure_in_band(
        refline: &[f32],
        dstline: &[f32],
        refmask: &[f32],
        dstmask: &[f32],
        visbase: f32,
        errorline: &mut [f32],
    ) {
        if !std::arch::is_x86_feature_detected!("avx2") {
            super::measure_in_band(refline, dstline, refmask, dstmask, visbase, errorline);
            return;
        }
        let n = errorline.len();
        let chunks = n.div_ceil(8);
        unsafe {
            for ch in 0..chunks {
                let i = ch * 8;
                let mut neg = [0.0f32; 8];
                let mut keep = [false; 8];
                for j in 0..8 {
                    let k = i + j;
                    if k < n {
                        // Compiled order: err = (|r−d|·visbase)·max(rm,dm)
                        // (visbase multiplies FIRST), p = (err·√err)·err².
                        let err =
                            (refline[k] - dstline[k]).abs() * visbase * refmask[k].max(dstmask[k]);
                        if err > 0.0 {
                            let p = (err * sq(err)) * (err * err);
                            neg[j] = -p;
                            keep[j] = true;
                        }
                    }
                }
                let e = expf8(&neg);
                for j in 0..8 {
                    let k = i + j;
                    if k < n && keep[j] {
                        errorline[k] *= e[j];
                    }
                }
            }
        }
    }

    /// `powf` on 8 lanes, bit-identical to the reference's vectorized call.
    ///
    /// # Safety
    /// Requires AVX2 at runtime.
    #[target_feature(enable = "avx2")]
    unsafe fn powf8(x: &[f32; 8], y: &[f32; 8]) -> [f32; 8] {
        let r: __m256;
        unsafe {
            core::arch::asm!(
                "vmovups ymm0, ymmword ptr [{x}]",
                "vmovups ymm1, ymmword ptr [{y}]",
                "call {f}",
                x = in(reg) x.as_ptr(),
                y = in(reg) y.as_ptr(),
                f = sym _ZGVdN8vv_powf,
                out("ymm0") r,
                clobber_abi("C"),
            );
        }
        // SAFETY: `__m256` and `[f32; 8]` have identical layout.
        unsafe { core::mem::transmute::<__m256, [f32; 8]>(r) }
    }

    /// `mapped[i] = |in[i]|·vis` raised to `e`, matching the reference's
    /// loop structure: full 8-lane vector chunks, a 4-lane epilogue chunk
    /// (`_ZGVbN4vv_powf`, bit-identical per lane), scalar `powf` tail <4.
    pub fn powf_mapped(out: &mut [f32], inp: &[f32], vis: f32, e: f32) {
        let n = out.len();
        if !std::arch::is_x86_feature_detected!("avx2") {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = super::pf(v.abs() * vis, e);
            }
            return;
        }
        let ye = [e; 8];
        let full = n / 8;
        let rem = n % 8;
        // SAFETY: avx2 detected above.
        unsafe {
            for k in 0..full {
                let i = k * 8;
                let mut x = [0.0f32; 8];
                for j in 0..8 {
                    x[j] = inp[i + j].abs() * vis;
                }
                let r = powf8(&x, &ye);
                out[i..i + 8].copy_from_slice(&r);
            }
            if rem >= 4 {
                let i = n - rem;
                let mut x = [0.0f32; 8];
                for j in 0..rem.min(8) {
                    x[j] = inp[i + j].abs() * vis;
                }
                let r = powf8(&x, &ye);
                let take = rem.min(4).min(8);
                out[i..i + take.min(rem)].copy_from_slice(&r[..take.min(rem)]);
                // scalar tail for the final rem-4 lanes
                for j in 4..rem {
                    out[i + j] = super::pf(inp[i + j].abs() * vis, e);
                }
            } else {
                for j in 0..rem {
                    out[n - rem + j] = super::pf(inp[n - rem + j].abs() * vis, e);
                }
            }
        }
    }
}

/// Approximate reciprocal matching the x86 `vrcpps` the reference's
/// `-freciprocal-math` build emits (SSE `rcpss` returns the identical table
/// bits on the same CPU). Off x86 we fall back to true division — ≤1 ulp
/// difference post-NR, documented in DIVERGENCES.md.
#[inline(always)]
fn rcp_approx(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `_mm_rcp_ss` is SSE, baseline on x86_64.
        unsafe {
            core::arch::x86_64::_mm_cvtss_f32(core::arch::x86_64::_mm_rcp_ss(
                core::arch::x86_64::_mm_set_ss(x),
            ))
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1.0 / x
    }
}
#[inline(always)]
fn sf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sinf(x)
    }
}
/// `a·b + c` with a single rounding — `f32::mul_add` under `std`,
/// `libm::fmaf` under `no_std` (same single-rounded semantics; the host
/// FMA instruction where available).
#[inline(always)]
fn fm(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::fmaf(a, b, c)
    }
}
#[inline(always)]
fn sq(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(x)
    }
}
#[inline(always)]
fn lf(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.ln()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::logf(x)
    }
}
/// 5-tap masking kernel: `gy = cos((y−2)/3·π/2)`, normalized by
/// `sqrt(Σ_{x,y} gx·gy / 25)` — same accumulation order as
/// `Masking::Masking`.
fn mask_kernel() -> [f32; MASK_SIZE] {
    // The reference computes these taps from cosf()/sqrtf(), but its
    // `-ffast-math` build vectorizes the ctor loop through libmvec
    // `_ZGVbN4v_cosf`, which rounds two of the five taps 1 ulp differently
    // than scalar `cosf` — and the vectorized `norm` reduction shifts the
    // shared divisor another ulp. These bit patterns are the exact taps the
    // AVX2 reference binary emits (verified via its DUMPMASK dump); the
    // scalar-faithful formula would be `cosf((y-2)/3·pi/2)/sqrt(norm/25)`.
    [
        f32::from_bits(0x3f2b7ccb),
        f32::from_bits(0x3f948335),
        f32::from_bits(0x3fab7ccc),
        f32::from_bits(0x3f948335),
        f32::from_bits(0x3f2b7ccb),
    ]
}

/// Post-filter taps for the Y-DC extended split (`Masking::EnablePostFilter`):
/// `lo[i] = 1`, `hi[i] = sin((i−2)·π/4)`; `norm_lo = 1/5`,
/// `norm_hi = 1/√Σhi²`.
fn post_filter() -> ([f32; MASK_SIZE], [f32; MASK_SIZE], f32, f32) {
    let mut lo = [0.0f32; MASK_SIZE];
    let mut hi = [0.0f32; MASK_SIZE];
    let mut nhi = 0.0f32;
    for i in 0..MASK_SIZE {
        let xp = (i as i32 - ((MASK_SIZE - 1) >> 1) as i32) as f32 / ((MASK_SIZE - 1) >> 1) as f32;
        lo[i] = 1.0;
        hi[i] = sf(xp * core::f32::consts::PI * 0.5);
        nhi += hi[i] * hi[i];
    }
    (lo, hi, 1.0 / MASK_SIZE as f32, 1.0 / sq(nhi))
}

/// sRGB u8 → linear-light lookup (`CreateSRGBLinearLookup`, bits = 8).
/// The reference build's `-freciprocal-math` lowers `i/255`, `n/12.92` and
/// `x/1.055` to reciprocal multiplies; scalar `powf` (this loop was not
/// vectorized). Both details are load-bearing for bit parity.
fn srgb_linear_lut() -> [f32; 256] {
    const C255: f32 = 1.0 / 255.0;
    const C1292: f32 = 1.0 / 12.92;
    const C1055: f32 = 1.0 / 1.055;
    let mut lut = [0.0f32; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        let n = i as f32 * C255;
        *v = if n <= 0.04045 {
            n * C1292
        } else {
            pf((n + 0.055) * C1055, 2.4)
        };
    }
    lut
}

/// Per-band visibility constant (`Pooling::Measure` setup):
/// `1/(8·ds[x]·ds[y]·√norms[x][y]) · (255/16) · CSF factor`. The CSF factor
/// per component folds the Ahumada tables (luma) or the weighted chroma
/// tables (`0.088·45/cb_tbl`, `0.278·21/cr_tbl`). The reference build folds
/// `1/8 · 255/16` into the numerator constant `255/128` and emits two real
/// divisions — `(255/128)/(dsx·dsy·√n)` and `b/tbl` — then multiplies; that
/// op order is reproduced here (verified bitwise against the compiled
/// binary for all 192 bands).
fn band_vis(x: usize, y: usize, c: usize) -> f32 {
    const LC: f32 = 255.0 / 128.0;
    let (b, tbl) = match c {
        0 => (16.0f32, LUMA_TBL[x + (y << 3)]),
        1 => (0.088f32 * 45.0, CB_TBL[x + (y << 3)]),
        _ => (0.278f32 * 21.0, CR_TBL[x + (y << 3)]),
    };
    let vis = (LC / ((DCT_SCALE[x] * DCT_SCALE[y]) * sq(NORMS[x][y]))) * (b / tbl as f32);
    #[cfg(feature = "std")]
    if std::env::var_os("DUMPV").is_some() {
        eprintln!("VIS c{c} x{x} y{y} vis={:08x}", vis.to_bits());
    }
    vis
}

/// Frequency-dependent masking exponent (`switch(x+y)`).
fn band_expon(x: usize, y: usize) -> f32 {
    match x + y {
        0 => 0.7,
        1 | 2 => 0.8,
        3 => 0.9,
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Per-band masking pipeline (measure/masking.cpp `Masking`).
// ---------------------------------------------------------------------------

/// One (band, component) masking instance. `in_ring`/`add_ring` hold up to
/// the 5 most recent band rows oldest→newest, rotating exactly like the C++
/// pointer swap; `mask`/`coeff` are the reference's `m_pMask`/`m_pOutput`.
struct BandMask {
    vis: f32,
    expon: f32,
    /// Staging row the sliding DCT scatters into, `w8` wide.
    next_row: Vec<f32>,
    /// Last ≤5 band coefficient rows, each `w8` wide.
    in_ring: [Vec<f32>; MASK_SIZE],
    /// Horizontal 5-tap conv of `mapped`, per buffered row, `w13` wide.
    add_ring: [Vec<f32>; MASK_SIZE],
    /// Scratch for `mapped`, `w8` wide.
    mapped: Vec<f32>,
    /// Ring fill level (reference `m_ulY`: rises to 4, then stays).
    fill: usize,
    mask: Vec<f32>,
    coeff: Vec<f32>,
    /// Y-DC extended-filter outputs (`m_pFiltered[x][y]`; index meaning as
    /// in C++), `w13` wide, only for comp 0 band (0,0).
    filtered: Option<[[Vec<f32>; 2]; 2]>,
    /// Cached 5×5 coefficient products for the lowpass (`vis` folded in):
    /// `[row filter][col filter][i][j]` — `row/col` = 0 box, 1 sin-high.
    lp: Option<[[[[f32; MASK_SIZE]; MASK_SIZE]; 2]; 2]>,
    w13: usize,
}

impl BandMask {
    fn new(vis: f32, expon: f32, post: bool, w8: usize, w13: usize) -> Self {
        let lp = post.then(|| {
            let (lo, hi, nlo, nhi) = post_filter();
            let f = [lo, hi];
            let n = [nlo, nhi];
            // c[rf][cf][i][j] = f[rf][i] · (f[cf][j] · s[rf][cf]). The C++
            // computes the scale as `vis·NormLo·NormLo`, `vis·NormLo·NormHi`
            // (shared by LH and HL), `vis·NormHi·NormHi` — i.e.
            // `(vis · n[min(rf,cf)]) · n[max(rf,cf)]`, not `vis·n[rf]·n[cf]`.
            let mut c = [[[[0.0f32; MASK_SIZE]; MASK_SIZE]; 2]; 2];
            for rf in 0..2 {
                for cf in 0..2 {
                    let s = (vis * n[rf.min(cf)]) * n[rf.max(cf)];
                    for i in 0..MASK_SIZE {
                        for j in 0..MASK_SIZE {
                            // GCC reassociates f[i]·f[j]·s → f[i]·(f[j]·s).
                            c[rf][cf][i][j] = f[rf][i] * (f[cf][j] * s);
                        }
                    }
                }
            }
            c
        });
        BandMask {
            vis,
            expon,
            next_row: alloc::vec![0.0; w8],
            in_ring: [
                alloc::vec![0.0; w8],
                alloc::vec![0.0; w8],
                alloc::vec![0.0; w8],
                alloc::vec![0.0; w8],
                alloc::vec![0.0; w8],
            ],
            add_ring: [
                alloc::vec![0.0; w13],
                alloc::vec![0.0; w13],
                alloc::vec![0.0; w13],
                alloc::vec![0.0; w13],
                alloc::vec![0.0; w13],
            ],
            mapped: alloc::vec![0.0; w8],
            fill: 0,
            mask: alloc::vec![0.0; w13],
            coeff: alloc::vec![0.0; w13],
            filtered: post.then(|| {
                [
                    [alloc::vec![0.0; w13], alloc::vec![0.0; w13]],
                    [alloc::vec![0.0; w13], alloc::vec![0.0; w13]],
                ]
            }),
            lp,
            w13,
        }
    }

    /// `Masking::PushLine` on the staged `next_row`.
    fn push_row(&mut self, kernel: &[f32; MASK_SIZE]) {
        // Swap the staged row into the ring at `fill` (saturating at 4),
        // matching the C++ `m_pInput[m_ulY]->Swap(line)`.
        let slot = self.fill.min(MASK_SIZE - 1);
        self.in_ring[slot].copy_from_slice(&self.next_row);

        // mapped[i] = (|in[i]| · vis)^expon, full band width. For e≠1 the
        // reference's powf loop vectorizes through libmvec — use it when
        // available so the ~5% 1-ulp lanes match bit-exactly.
        let vis = self.vis;
        let e = self.expon;
        if e == 1.0 {
            for (m, &v) in self.mapped.iter_mut().zip(self.in_ring[slot].iter()) {
                *m = v.abs() * vis;
            }
        } else {
            #[cfg(all(
                feature = "std",
                target_arch = "x86_64",
                target_os = "linux",
                target_env = "gnu"
            ))]
            {
                libmvec::powf_mapped(&mut self.mapped, &self.in_ring[slot], vis, e);
            }
            #[cfg(not(all(
                feature = "std",
                target_arch = "x86_64",
                target_os = "linux",
                target_env = "gnu"
            )))]
            {
                for (m, &v) in self.mapped.iter_mut().zip(self.in_ring[slot].iter()) {
                    *m = pf(v.abs() * vis, e);
                }
            }
        }
        // Horizontal 5-tap symmetric conv → add_ring[slot].
        let (m0, m1, m2) = (kernel[0], kernel[1], kernel[2]);
        let mp = &self.mapped;
        let add = &mut self.add_ring[slot];
        for (i, a) in add.iter_mut().enumerate() {
            // GCC contracts `ColumnSumConvolution` to
            // fma(m2, mp2, fma(m0, s04, m1·s13)).
            *a = fm(
                m2,
                mp[i + 2],
                fm(m0, mp[i] + mp[i + 4], m1 * (mp[i + 1] + mp[i + 3])),
            );
        }

        if self.fill == MASK_SIZE - 1 {
            self.compute_mask(kernel);
            if self.lp.is_some() {
                self.compute_lowpass();
            }
            // Rotate: oldest buffer drops off the bottom, is reused on top.
            self.in_ring.rotate_left(1);
            self.add_ring.rotate_left(1);
        } else {
            self.fill += 1;
        }
    }

    /// `Masking::ComputeMask`:
    /// `mask = base/(base + Σ_{5×5} mapped · w / 25)`,
    /// `coeff[i] = in_ring[2][i+2] · vis`.
    fn compute_mask(&mut self, kernel: &[f32; MASK_SIZE]) {
        let (m0, m1, m2) = (kernel[0], kernel[1], kernel[2]);
        let base = BASE_VISIBILITY;
        let inv_ms2 = 1.0 / (MASK_SIZE * MASK_SIZE) as f32;
        for i in 0..self.w13 {
            // FMA chain order of `ComputeMaskLoop`: m0·s04, then
            // fma(m1, s13), fma(m2, d2), then fused fma(sum, 1/25, base).
            let sum = m0 * (self.add_ring[0][i] + self.add_ring[4][i]);
            let sum = fm(m1, self.add_ring[1][i] + self.add_ring[3][i], sum);
            let sum = fm(m2, self.add_ring[2][i], sum);
            let den = fm(sum, inv_ms2, base);
            // `-freciprocal-math` lowers base/den to vrcpps + one NR step:
            // mask = base·(2r − den·r²).
            let r = rcp_approx(den);
            self.mask[i] = base * ((r + r) - (den * r) * r);
            self.coeff[i] = self.in_ring[MASK_SIZE >> 1][i + (MASK_SIZE >> 1)] * self.vis;
        }
    }

    /// `Masking::ComputeLowpass` — the four 5×5 separable subbands of the
    /// DC coefficient plane (`filtered[x][y]` follows the C++ `m_pFiltered`
    /// indexing: `[x][y]` ↔ `[coeff pair]` per `GetLowpass(x,y)`).
    fn compute_lowpass(&mut self) {
        let cf4 = *self.lp.as_ref().unwrap();
        if let Some(filt) = &mut self.filtered {
            for x in 0..2 {
                for y in 0..2 {
                    // filtered[x][y] uses row-filter index `y`, col index
                    // `x`: m_pFiltered[1][0] = LH = lo_rows · hi_cols.
                    let cf = &cf4[y][x];
                    let out = &mut filt[x][y];
                    for (k, o) in out.iter_mut().enumerate() {
                        let mut acc = 0.0f32;
                        #[allow(clippy::needless_range_loop)]
                        for i in 0..MASK_SIZE {
                            let r = &self.in_ring[i][k..k + MASK_SIZE];
                            // GCC's unrolled+fma row term:
                            // (fma(v4,c4, fma(v0,c0, v1·c1))) + fma(v2,c2, v3·c3)
                            let t = fm(r[4], cf[i][4], fm(r[0], cf[i][0], r[1] * cf[i][1]))
                                + fm(r[2], cf[i][2], r[3] * cf[i][3]);
                            acc += t;
                        }
                        *o = acc;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-component sliding-window DCT (dct/component.cpp `Component`).
// ---------------------------------------------------------------------------

struct Component {
    /// Ring of 8 component rows (`w` wide), oldest first.
    ring: [Vec<f32>; 8],
    /// Ring fill level (saturates at 8 once full).
    fill: usize,
    /// Masking pipelines, indexed `[x][y]` like `m_Weights[x][y][c]`.
    bands: [[BandMask; 8]; 8],
    w8: usize,
    #[cfg(feature = "std")]
    first_call: bool,
    #[allow(dead_code)]
    first_call_w: bool,
}

impl Component {
    fn new(w: usize, w8: usize, w13: usize, c: usize) -> Self {
        Component {
            ring: [
                alloc::vec![0.0; w],
                alloc::vec![0.0; w],
                alloc::vec![0.0; w],
                alloc::vec![0.0; w],
                alloc::vec![0.0; w],
                alloc::vec![0.0; w],
                alloc::vec![0.0; w],
                alloc::vec![0.0; w],
            ],
            fill: 0,
            bands: core::array::from_fn(|x| {
                core::array::from_fn(|y| {
                    BandMask::new(
                        band_vis(x, y, c),
                        band_expon(x, y),
                        c == 0 && x == 0 && y == 0,
                        w8,
                        w13,
                    )
                })
            }),
            w8,
            #[cfg(feature = "std")]
            first_call: true,
            first_call_w: true,
        }
    }

    /// `Component::PushLine`: returns true once the ring holds 8 rows and
    /// the sliding DCT produces a row (i.e. from the 8th push onward).
    fn push_line(&mut self, row: &[f32]) -> bool {
        if self.fill < 8 {
            self.ring[self.fill].copy_from_slice(row);
            self.fill += 1;
            self.fill == 8
        } else {
            self.ring.rotate_left(1);
            self.ring[7].copy_from_slice(row);
            true
        }
    }

    /// `Component::Run` at `mod=1, offset=0`: sliding 8×8 windowed AAN DCT
    /// for `i in 0 .. w-8` — the last legal position `w−8` is skipped, as
    /// in the reference loop bound `i < w - 8`.
    fn run_dct(&mut self) {
        for i in 0..self.w8 {
            #[cfg(feature = "std")]
            let pcall = {
                static PCALL: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                PCALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            };
            let mut block = [[0.0f32; 8]; 8];
            for (y, r) in block.iter_mut().enumerate() {
                r.copy_from_slice(&self.ring[y][i..i + 8]);
            }
            #[cfg(feature = "std")]
            CUR_CALL.with(|c| c.set(pcall));
            #[cfg(feature = "std")]
            if let Some(v) = std::env::var_os("DUMPP") {
                let lo: usize = v.to_str().unwrap().parse().unwrap();
                let hi: usize = std::env::var("DUMPP2")
                    .map(|s| s.parse().unwrap())
                    .unwrap_or(lo);
                if pcall >= lo && pcall <= hi {
                    for (y, r) in block.iter().enumerate() {
                        for (x, &v) in r.iter().enumerate() {
                            eprintln!("PIN {pcall} {y} {x} {:#010x}", v.to_bits());
                        }
                    }
                }
            }
            #[cfg(feature = "std")]
            if std::env::var_os("DUMPI").is_some() && i == 0 && self.first_call {
                self.first_call = false;
                for (y, r) in self.ring.iter().enumerate() {
                    for (k, v) in r.iter().enumerate() {
                        eprintln!(
                            "ROW{:08x} {y} {k} {:08x}",
                            self as *const _ as usize & 0xffffff,
                            v.to_bits()
                        );
                    }
                }
            }
            // Window toward the block mean (not toward zero). The executed
            // (≥8-lane) reference path sums per-column with GCC's
            // `-fassociative-math` regrouping of the source tree — verified
            // bitwise from `N_AVX2 ProcessBlock_HWY` disassembly:
            // `((r0+r7)+(r5+r6)) + ((r1+r2)+(r3+r4))`; the scalar tail
            // `s0+..+s7` is then auto-vectorized into a tree reduction
            // `((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))` (vextractf128 +
            // vmovhlps + broadcast-lane1, verified bitwise), and the window
            // multiply is fused (`MulAdd`).
            let mut s = [0.0f32; 8];
            for (x, sv) in s.iter_mut().enumerate() {
                *sv = ((block[0][x] + block[7][x]) + (block[5][x] + block[6][x]))
                    + ((block[1][x] + block[2][x]) + (block[3][x] + block[4][x]));
            }
            let avg = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
            let avg = avg / 64.0;
            for (y, r) in block.iter_mut().enumerate() {
                for (x, v) in r.iter_mut().enumerate() {
                    let win = WINDOW[y];
                    *v = fm(*v, win[x], avg * (1.0 - win[x]));
                }
            }
            #[cfg(feature = "std")]
            if let Ok(t) = std::env::var("DUMPW")
                && pcall == t.parse::<usize>().unwrap_or(usize::MAX)
            {
                eprintln!("AVGHERE {:#010x}", avg.to_bits());
                for (y, r) in block.iter().enumerate() {
                    for (x, v) in r.iter().enumerate() {
                        eprintln!("WIN {} {} {:#010x}", y, x, v.to_bits());
                    }
                }
            }
            let coeff = dct_8x8(&block);
            for (x, cb) in coeff.iter().enumerate() {
                for (y, &v) in cb.iter().enumerate() {
                    self.bands[x][y].next_row[i] = v;
                }
            }
        }
    }
}

#[cfg(feature = "std")]
thread_local! {
    static CUR_CALL: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Scaled AAN 8×8 DCT matching the executed (≥8-lane SIMD) path of
/// `ProcessBlock`. The vector `in[y]` holds row `y` with the 8 columns in
/// the lanes, so the elementwise butterfly `in[0]+in[7]` pairs *rows*: the
/// first pass is a **vertical** (column) transform, the register file is
/// transposed, and the same butterfly runs again — a horizontal transform
/// of the original rows (despite the "process rows" comment). Returns
/// `out[u][v]` = coefficient (u = horizontal, v = vertical frequency),
/// i.e. what the reference writes to `outputLines[x][y]`.
///
/// (The reference's scalar fallback transforms rows first, producing the
/// transpose of this layout, which is observationally equivalent:
/// `dct_scale`, `norms`, and all three CSF tables are transpose-symmetric
/// — see `tables_are_symmetric`.)
fn dct_8x8(block: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    // Pass 1 (vertical): `aan_1d` per column — `v[k][x]` is the k-th
    // vertical frequency of column `x` (register `k`, lane `x`).
    let mut v = [[0.0f32; 8]; 8];
    for x in 0..8 {
        let col = [
            block[0][x],
            block[1][x],
            block[2][x],
            block[3][x],
            block[4][x],
            block[5][x],
            block[6][x],
            block[7][x],
        ];
        let cv = aan_1d(&col);
        for (k, &c) in cv.iter().enumerate() {
            v[k][x] = c;
        }
    }
    #[cfg(feature = "std")]
    if let Ok(t) = std::env::var("DUMPA") {
        let tgt: usize = t.parse().unwrap();
        let cur = CUR_CALL.with(|c| c.get());
        if cur == tgt {
            for (k, r) in v.iter().enumerate() {
                for (x, &c) in r.iter().enumerate() {
                    eprintln!("PASS1 {k} {x} {:#010x}", c.to_bits());
                }
            }
        }
    }
    // Transpose (register file) + pass 2 (same butterfly): a horizontal
    // transform of each vertical-frequency row — `hv[u]` is horizontal
    // freq `u` of row `v[vix]` → `out[u][vix]`.
    let mut out = [[0.0f32; 8]; 8];
    for (vix, row) in v.iter().enumerate() {
        let hv = aan_1d(row);
        for (u, &c) in hv.iter().enumerate() {
            out[u][vix] = c;
        }
    }
    #[cfg(feature = "std")]
    if let Ok(t) = std::env::var("DUMPB") {
        let tgt: usize = t.parse().unwrap();
        if CUR_CALL.with(|c| c.get()) == tgt {
            for (y, r) in out.iter().enumerate() {
                for (x, &c) in r.iter().enumerate() {
                    eprintln!("PASS2 {y} {x} {:#010x}", c.to_bits());
                }
            }
        }
    }
    out
}

/// The unscaled AAN 1-D DCT-II butterfly sequence (dct/component.cpp),
/// transcribed to reproduce the operation order — including `-ffast-math`
/// reassociations and FMA contractions — that the reference's compiled
/// Highway path executes. `mul_add` calls must compile to a single-rounded
/// FMA (they do on every Rust target with hardware FMA; software
/// `fma` is also single-rounded).
fn aan_1d(b: &[f32; 8]) -> [f32; 8] {
    let tmp0 = b[0] + b[7];
    let tmp7 = b[0] - b[7];
    let tmp1 = b[1] + b[6];
    let tmp6 = b[1] - b[6];
    let tmp2 = b[2] + b[5];
    let tmp5 = b[2] - b[5];
    let tmp3 = b[3] + b[4];

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;

    let mut out = [0.0f32; 8];
    out[0] = tmp10 + tmp11;
    out[4] = tmp10 - tmp11;

    // The reference's compiled code reassociates (tmp1-tmp2)+tmp13 into
    // tmp1+(tmp13-tmp2) and contracts the z1 output adds into FMAs.
    let z1n = tmp1 + (tmp13 - tmp2);
    out[2] = fm(z1n, SQRT2_2, tmp13);
    out[6] = fm(-z1n, SQRT2_2, tmp13);

    // tmp4+tmp5 is likewise reassociated to b[3]+(tmp5-b[4]); tmp4 is
    // never computed.
    let o10 = b[3] + (tmp5 - b[4]);
    let o11 = tmp6 + tmp5;
    let o12 = tmp6 + tmp7;

    let z5 = (o10 - o12) * C1;
    let z2 = fm(o10, C2, z5);
    let z4 = fm(o12, C3, z5);

    // z11/z13 contract the z3 multiply into fused forms.
    let z11 = fm(o11, SQRT2_2, tmp7);
    let z13 = fm(-o11, SQRT2_2, tmp7);

    out[5] = z13 + z2;
    out[3] = z13 - z2;
    out[1] = z11 + z4;
    out[7] = z11 - z4;
    out
}

// ---------------------------------------------------------------------------
// Pooling driver (measure/pooling.cpp `Pooling::Measure`).
// ---------------------------------------------------------------------------

struct MeteredImage {
    comps: [Component; 3],
}

impl MeteredImage {
    fn new(w: usize, w8: usize, w13: usize) -> Self {
        MeteredImage {
            comps: [
                Component::new(w, w8, w13, 0),
                Component::new(w, w8, w13, 1),
                Component::new(w, w8, w13, 2),
            ],
        }
    }
}

/// `Pooling::MeasureInBand`: the compiled AVX2 path computes
/// `err = (|ref−dst|·visbase)·max(mask_ref, mask_dst)` and
/// `p = (err·√err)·err²`, then `errorline *= exp(−p)` for `err > 0` —
/// reproduced here verbatim (multiply order verified from the
/// `N_AVX2` clone's disassembly). On x86_64+glibc the vectorized `expf`
/// (`_ZGVdN8v_expf`, which differs from scalar `expf` by ~1 ulp on many
/// inputs) is dispatched through `libmvec`.
#[allow(clippy::too_many_arguments)]
fn measure_in_band(
    refline: &[f32],
    dstline: &[f32],
    refmask: &[f32],
    dstmask: &[f32],
    visbase: f32,
    errorline: &mut [f32],
) {
    #[cfg(all(
        feature = "std",
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    ))]
    {
        libmvec::measure_in_band(refline, dstline, refmask, dstmask, visbase, errorline);
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..errorline.len() {
        let err = (refline[i] - dstline[i]).abs() * visbase * refmask[i].max(dstmask[i]);
        if err > 0.0 {
            let p = (err * sq(err)) * (err * err);
            errorline[i] *= ef(-p);
        }
    }
}

/// The Y-DC mask fixup (`Pooling::Measure`): comp 0 band (0,0) borrows the
/// masks of bands (0,1), (1,0), (1,1) — `0.5·m01 + 0.5·m10 + 0.25·m11` —
/// since DC artifacts would otherwise get a self-generated mask at block
/// boundaries.
fn dc_mask_fixup(img: &mut MeteredImage) {
    let bands = &mut img.comps[0].bands;
    let (x0, rest) = bands.split_at_mut(1);
    let (x1, _) = rest.split_at_mut(1);
    let (y00, rest0) = x0[0].split_at_mut(1);
    let (y01, _) = rest0.split_at_mut(1);
    let (y10, rest1) = x1[0].split_at_mut(1);
    let (y11, _) = rest1.split_at_mut(1);
    let m00 = &mut y00[0].mask;
    let m01 = &y01[0].mask;
    let m10 = &y10[0].mask;
    let m11 = &y11[0].mask;
    for i in 0..m00.len() {
        // GCC `-ffast-math` reassociates to fma(m11, 0.25, (m01+m10)·0.5).
        m00[i] = fm(m11[i], 0.25, (m01[i] + m10[i]) * 0.5);
    }
}

/// sRGB8 row → `(Y′, Cb, Cr)` float rows
/// (`ColorTransformer::ForwardsTransform`, `LINEAR` + `KNEE_VALUE` build).
/// Only Y passes the sRGB gamma knee; Cb/Cr stay linear and signed.
fn transform_row_into(row: &[u8], lut: &[f32; 256], out: &mut [Vec<f32>; 3]) {
    // GCC folds the constant kneeslope to 12.9199915; the expression below
    // evaluates to identical f32 bits.
    let kneeslope = (1.055 * pf(0.0031308, 1.0 / 2.4) - 0.055) / 0.0031308;
    for (i, px) in row.as_chunks::<3>().0.iter().enumerate() {
        let r = lut[px[0] as usize];
        let g = lut[px[1] as usize];
        let b = lut[px[2] as usize];
        // The reference binary fuses the matrix multiply into FMA chains:
        //   Y  = fma(0.114, b, fma(0.299, r, 0.587*g))
        //   Cb = fnmadd(0.33126, g, fma(-0.16875, r, 0.5*b))
        //   Cr = fmsub(0.5, r, fma(0.41869, g, 0.08131*b))
        // and the knee encode is fma(1.055, powf(y, 1/2.4), -0.055).
        let mut yv = fm(0.114f32, b, fm(0.299f32, r, 0.587 * g));
        let cb = fm(-0.33126f32, g, fm(-0.16875f32, r, 0.5 * b));
        let cr = fm(0.5f32, r, -(fm(0.41869f32, g, 0.08131 * b)));
        yv = if yv <= 0.0031308 {
            yv * kneeslope
        } else {
            fm(1.055f32, pf(yv, 1.0 / 2.4), -0.055)
        };
        out[0][i] = yv;
        out[1][i] = cb;
        out[2][i] = cr;
    }
}

/// mDCT-PSNR score in **dB** for two interleaved sRGB-8 RGB buffers of
/// `width`×`height` — exactly `dctpsnr ref.ppm dist.ppm` default output
/// (the AIC-4 `mDCT-PSNR` column). Identical images give `+inf`, matching
/// `−20·log10(0)` in the reference.
pub fn mdct_psnr_srgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<f32, Error> {
    if reference.len() != width * height * 3 || distorted.len() != width * height * 3 {
        return Err(Error::LengthMismatch);
    }
    if width < 14 || height < 14 {
        return Err(Error::ImageTooSmall);
    }

    let kernel = mask_kernel();
    let lut = srgb_linear_lut();
    let w8 = width - 8;
    let w13 = width - 13;

    let mut img_r = MeteredImage::new(width, w8, w13);
    let mut img_d = MeteredImage::new(width, w8, w13);

    let mut error = 0.0f32;
    #[cfg(feature = "std")]
    let mut error64 = 0.0f64;
    let mut errorline = alloc::vec![1.0f32; w13];
    let mut rows_r: [Vec<f32>; 3] = [
        alloc::vec![0.0; width],
        alloc::vec![0.0; width],
        alloc::vec![0.0; width],
    ];
    let mut rows_d: [Vec<f32>; 3] = [
        alloc::vec![0.0; width],
        alloc::vec![0.0; width],
        alloc::vec![0.0; width],
    ];

    for y in 0..height {
        // Image::ReadNextLine + ColorTransformer::ForwardsTransform.
        transform_row_into(
            &reference[y * width * 3..(y + 1) * width * 3],
            &lut,
            &mut rows_r,
        );
        transform_row_into(
            &distorted[y * width * 3..(y + 1) * width * 3],
            &lut,
            &mut rows_d,
        );

        let mut ready = false;
        for c in 0..3 {
            ready = img_r.comps[c].push_line(&rows_r[c]);
            img_d.comps[c].push_line(&rows_d[c]);
        }
        if ready {
            // SplitWork: run the sliding DCT (Component::Run) and push the
            // 64 band rows into their masking pipelines (Pooling::Run).
            for c in 0..3 {
                img_r.comps[c].run_dct();
                img_d.comps[c].run_dct();
            }
            for c in 0..3 {
                for x in 0..8 {
                    for yy in 0..8 {
                        img_r.comps[c].bands[x][yy].push_row(&kernel);
                        img_d.comps[c].bands[x][yy].push_row(&kernel);
                    }
                }
            }
        }

        // Main measure loop: active from image line 12 on (12-line warmup).
        if y >= 12 {
            for e in errorline.iter_mut() {
                *e = 1.0;
            }
            dc_mask_fixup(&mut img_r);
            dc_mask_fixup(&mut img_d);
            #[cfg(feature = "std")]
            let dline: Option<usize> = std::env::var("DUMPT").ok().and_then(|s| s.parse().ok());
            #[cfg(feature = "std")]
            let tgt_i: usize = std::env::var("DUMPTI")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            for c in 0..3 {
                for yy in 0..8 {
                    for x in 0..8 {
                        let br = &img_r.comps[c].bands[x][yy];
                        let bd = &img_d.comps[c].bands[x][yy];
                        if c == 0 && x == 0 && yy == 0 {
                            // EXTENDED_FILTER: the Y-DC band is measured on
                            // its four 5×5 low/high subbands instead of the
                            // raw coefficient, with these visbase factors.
                            let l0 = LUMA_TBL[0] as f32;
                            let subs: [(usize, usize, f32); 4] = [
                                (0, 0, l0 / 90.0),
                                (1, 0, l0 / 15.0),
                                (0, 1, l0 / 15.0),
                                (1, 1, l0 / 20.0),
                            ];
                            let fr = br.filtered.as_ref().unwrap();
                            let fd = bd.filtered.as_ref().unwrap();
                            for &(a, b, vb) in &subs {
                                measure_in_band(
                                    &fr[a][b],
                                    &fd[a][b],
                                    &br.mask,
                                    &bd.mask,
                                    vb,
                                    &mut errorline,
                                );
                            }
                            #[cfg(feature = "std")]
                            if dline == Some(y - 12) {
                                eprintln!("EL DC-EXT -> {:08x}", errorline[tgt_i].to_bits());
                            }
                        } else {
                            measure_in_band(
                                &br.coeff,
                                &bd.coeff,
                                &br.mask,
                                &bd.mask,
                                1.0,
                                &mut errorline,
                            );
                        }
                        #[cfg(feature = "std")]
                        if dline == Some(y - 12) {
                            eprintln!("EL c{c} x{x} y{yy} -> {:08x}", errorline[tgt_i].to_bits());
                        }
                    }
                }
            }
            // The reference's `error += 1.0f - errorline[i]` loop is
            // vectorized by GCC `-fassociative-math` into an 8-lane f32
            // accumulator summed per ymm chunk in address order, folded
            // high+low 128 to 4 lanes; when `w mod 8 >= 4` a 4-element
            // tail block is merged into the folded vector before the
            // collapse `(x0+x2)+(x1+x3)`, and elements past that block
            // are added scalar-sequentially. For `w mod 8 < 4` the fold
            // collapses directly and the tail is scalar. The f64 `err64`
            // accumulator mirrors this with 4 lanes (`cvtps2pd` pairs
            // `t[j]+t[j+4]`). Reproduced verbatim — the f32 result differs
            // by ~1e-1 absolute vs sequential summation over a 2K image.
            {
                let w = errorline.len();
                let n8 = w & !7;
                let rem = w & 7;
                let mut vacc = [0.0f32; 8];
                let mut dacc = [0.0f64; 4];
                for c in errorline[..n8].as_chunks::<8>().0 {
                    for j in 0..8 {
                        vacc[j] += 1.0 - c[j];
                    }
                    for j in 0..4 {
                        dacc[j] += (1.0 - c[j]) as f64 + (1.0 - c[j + 4]) as f64;
                    }
                }
                let mut x5 = [
                    vacc[0] + vacc[4],
                    vacc[1] + vacc[5],
                    vacc[2] + vacc[6],
                    vacc[3] + vacc[7],
                ];
                let mut x2 = [dacc[0] + dacc[2], dacc[1] + dacc[3]];
                if rem >= 4 {
                    // Merge the 4-wide tail into the folded vectors first,
                    // then collapse (the main-collapse result computed in
                    // the rem<4 path is dead on this path).
                    let mut t = [0.0f32; 4];
                    for j in 0..4 {
                        t[j] = 1.0 - errorline[n8 + j];
                        x5[j] += t[j];
                    }
                    for j in 0..2 {
                        x2[j] += t[j] as f64 + t[j + 2] as f64;
                    }
                }
                error += (x5[0] + x5[2]) + (x5[1] + x5[3]);
                #[cfg(feature = "std")]
                {
                    error64 += x2[0] + x2[1];
                }
                // Scalar remainder: elements n8..w (rem<4) or n8+4..w.
                let start = if rem >= 4 { n8 + 4 } else { n8 };
                for &e in &errorline[start..w] {
                    error += 1.0 - e;
                    #[cfg(feature = "std")]
                    {
                        error64 += (1.0 - e) as f64;
                    }
                }
            }
            #[cfg(feature = "std")]
            if std::env::var_os("DUMPEL").is_some() {
                eprintln!("ELL {}", y - 12);
                for di in (0..errorline.len()).step_by(16) {
                    eprintln!("ELPOS {di} {:.9}", 1.0 - errorline[di]);
                }
            }
        }
    }

    #[cfg(feature = "std")]
    if std::env::var_os("DUMPE").is_some() {
        let w13 = width - 13;
        let l = height - 12;
        eprintln!(
            "ERRSUM w={w13} h={} l={l} error={error:.9} error64={error64:.12}",
            height - 13
        );
    }
    // `return -20.0f * logf(error / (w * h * comps)) / logf(10.0f)` — with
    // the reference's denominator quirk: `h = H−13` while `H−12` lines are
    // measured.
    let w = (width - 13) as f32;
    let h = (height - 13) as f32;
    Ok(-20.0 * lf(error / (w * h * 3.0)) / lf(10.0))
}

/// Debug hook for parity-bisecting: prints the (0,0), (1,0), (0,1) Y-band
/// coeff/mask values for the first measured line, mirroring the patched
/// reference's `DUMP=1` output. Not API-stable.
#[cfg(feature = "std")]
pub fn dump_debug(reference: &[u8], distorted: &[u8], width: usize, height: usize) {
    let kernel = mask_kernel();
    let lut = srgb_linear_lut();
    let w8 = width - 8;
    let w13 = width - 13;
    let mut img_r = MeteredImage::new(width, w8, w13);
    let mut img_d = MeteredImage::new(width, w8, w13);
    let mut rows_r: [Vec<f32>; 3] = [
        alloc::vec![0.0; width],
        alloc::vec![0.0; width],
        alloc::vec![0.0; width],
    ];
    let mut rows_d: [Vec<f32>; 3] = [
        alloc::vec![0.0; width],
        alloc::vec![0.0; width],
        alloc::vec![0.0; width],
    ];
    for y in 0..height {
        transform_row_into(
            &reference[y * width * 3..(y + 1) * width * 3],
            &lut,
            &mut rows_r,
        );
        transform_row_into(
            &distorted[y * width * 3..(y + 1) * width * 3],
            &lut,
            &mut rows_d,
        );
        let mut ready = false;
        for c in 0..3 {
            ready = img_r.comps[c].push_line(&rows_r[c]);
            img_d.comps[c].push_line(&rows_d[c]);
        }
        if ready {
            for c in 0..3 {
                img_r.comps[c].run_dct();
                img_d.comps[c].run_dct();
            }
            for c in 0..3 {
                for x in 0..8 {
                    for yy in 0..8 {
                        img_r.comps[c].bands[x][yy].push_row(&kernel);
                        img_d.comps[c].bands[x][yy].push_row(&kernel);
                    }
                }
            }
        }
        if y >= 12 {
            dc_mask_fixup(&mut img_r);
            dc_mask_fixup(&mut img_d);
            let dline: usize = std::env::var("DUMP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if y == 12 + dline {
                for c in 0..3 {
                    for yy in 0..8 {
                        for x in 0..8 {
                            let br = &img_r.comps[c].bands[x][yy];
                            let bd = &img_d.comps[c].bands[x][yy];
                            for i in 0..w13 {
                                eprintln!(
                                    "BAND c{c} x{x} y{yy} i{i} ref={:08x} dst={:08x} rmask={:08x} dmask={:08x}",
                                    br.coeff[i].to_bits(),
                                    bd.coeff[i].to_bits(),
                                    br.mask[i].to_bits(),
                                    bd.mask[i].to_bits()
                                );
                            }
                        }
                    }
                }
                if let (Some(fr), Some(fd)) = (
                    img_r.comps[0].bands[0][0].filtered.as_ref(),
                    img_d.comps[0].bands[0][0].filtered.as_ref(),
                ) {
                    for a in 0..2 {
                        for b in 0..2 {
                            for i in 0..w13 {
                                eprintln!(
                                    "LP {a}{b} i{i} ref={:08x} dst={:08x}",
                                    fr[a][b][i].to_bits(),
                                    fd[a][b][i].to_bits()
                                );
                            }
                        }
                    }
                }
                return;
            }
        }
    }
}

/// The reference's `-jnd` remap (`main.cpp`): JND units from the dB score.
/// The AIC-4 `mDCT-PSNR` column is the plain dB score; this is only for
/// parity with the optional flag.
pub fn mdct_psnr_jnd(score_db: f32) -> f32 {
    let err = 80.0 - score_db;
    let err = if err < 0.0 { 0.0 } else { err };
    0.000475997802 * pf(err, 2.3182)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Identical images → detection error 0 → `−20·log10(0)` = +inf, as
    /// the reference computes it.
    #[test]
    fn identical_is_infinite() {
        let (w, h) = (32usize, 32usize);
        let img = alloc::vec![128u8; w * h * 3];
        let s = mdct_psnr_srgb8(&img, &img, w, h).unwrap();
        assert!(s.is_infinite() && s > 0.0);
    }

    /// A constant sRGB shift produces a finite, positive score.
    #[test]
    fn flat_shift_is_finite() {
        let (w, h) = (64usize, 48usize);
        let a = alloc::vec![120u8; w * h * 3];
        let b = alloc::vec![140u8; w * h * 3];
        let s = mdct_psnr_srgb8(&a, &b, w, h).unwrap();
        assert!(s.is_finite() && s > 0.0, "score {s}");
    }

    /// Golden vs the built reference `dctpsnr` (GCC `-O3 -ffast-math`,
    /// glibc 2.43, x86_64 AVX2) on a deterministic 64×48 texture pair —
    /// the binary prints 96.5176315 for it. The libmvec-dispatched paths
    /// make this near-exact on x86_64+GNU; off-platform scalar fallbacks
    /// drift by ~1ulp/step, so the tolerance widens there.
    #[test]
    fn golden_synthetic_64x48() {
        let (w, h) = (64usize, 48usize);
        let mut r = alloc::vec![0u8; w * h * 3];
        let mut d = alloc::vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                r[i] = ((x * 37 + y * 11) % 256) as u8;
                r[i + 1] = ((x * 3 + y * 29) % 256) as u8;
                r[i + 2] = ((x * 17 + y * 7) % 256) as u8;
                let dt = ((x * 13 + y * 5) % 7) as i32 - 3;
                for c in 0..3 {
                    d[i + c] = (r[i + c] as i32 + dt * [1, 2, -1][c]).clamp(0, 255) as u8;
                }
            }
        }
        let s = mdct_psnr_srgb8(&r, &d, w, h).unwrap();
        let expect = 96.5176315f32;
        #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
        let tol = 2e-4;
        #[cfg(not(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")))]
        let tol = 1e-2;
        assert!((s - expect).abs() < tol, "score {s} vs {expect}");
    }

    /// Goldens vs the built reference `dctpsnr` across all eight
    /// `(w-13) % 8` residue classes of the pooling line width — the
    /// vectorized f32 accumulator's tail path (4-wide block iff `rem ≥ 4`,
    /// then scalar) differs per class, and `rem == 3` was the worst
    /// historical divergence. Same deterministic texture as the 64×48
    /// golden; reference scores captured on x86_64+GNU/AVX2.
    #[test]
    fn golden_width_residues_48h() {
        // (width, height, reference dctpsnr score). w-13 = 32..39 covers
        // residues 0..=7 in order.
        let cases: &[(usize, usize, f32)] = &[
            (45, 48, 96.5051804),
            (46, 48, 96.4371643),
            (47, 48, 96.4117279),
            (48, 48, 96.4327469),
            (49, 48, 96.4426422),
            (50, 48, 96.4818649),
            (51, 48, 96.5563583),
            (52, 48, 96.6095505),
        ];
        for &(w, h, expect) in cases {
            let mut r = alloc::vec![0u8; w * h * 3];
            let mut d = alloc::vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    r[i] = ((x * 37 + y * 11) % 256) as u8;
                    r[i + 1] = ((x * 3 + y * 29) % 256) as u8;
                    r[i + 2] = ((x * 17 + y * 7) % 256) as u8;
                    let dt = ((x * 13 + y * 5) % 7) as i32 - 3;
                    for c in 0..3 {
                        d[i + c] = (r[i + c] as i32 + dt * [1, 2, -1][c]).clamp(0, 255) as u8;
                    }
                }
            }
            let s = mdct_psnr_srgb8(&r, &d, w, h).unwrap();
            #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
            let tol = 2e-4;
            #[cfg(not(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")))]
            let tol = 1e-2;
            assert!(
                (s - expect).abs() < tol,
                "{w}x{h} (w13%8={}): score {s} vs {expect}",
                (w - 13) % 8
            );
        }
    }

    /// The band tables are transpose-symmetric — this is what makes the
    /// reference's scalar-vs-SIMD band-ordering difference unobservable.
    #[test]
    fn tables_are_symmetric() {
        for x in 0..8 {
            for y in 0..8 {
                assert_eq!(LUMA_TBL[x + (y << 3)], LUMA_TBL[y + (x << 3)]);
                assert_eq!(CB_TBL[x + (y << 3)], CB_TBL[y + (x << 3)]);
                assert_eq!(CR_TBL[x + (y << 3)], CR_TBL[y + (x << 3)]);
                assert_eq!(NORMS[x][y], NORMS[y][x]);
            }
        }
    }
}
