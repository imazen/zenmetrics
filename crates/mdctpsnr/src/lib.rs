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

use zenmetrics_abstractions as za;

use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(any(all(feature = "std", target_arch = "aarch64"), target_arch = "x86_64"))]
use archmage::attune;
#[cfg(all(feature = "std", target_arch = "aarch64"))]
use archmage::{NeonToken, SimdToken as _};
#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, X64V3Token};
#[cfg(any(all(feature = "std", target_arch = "aarch64"), target_arch = "x86_64"))]
use magetypes::simd::generic::f32x8;

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

impl core::error::Error for Error {}

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
/// Vector helpers for the measured hot loops, built on magetypes'
/// mid-precision kernels (`pow_midp`, `exp_midp`) through the
/// safe `f32x8` API — no extern calls, no `asm!`, no `transmute`. These
/// kernels differ from glibc's `_ZGVdN8*` vector routines (and from scalar
/// `powf`/`expf`) by up to ~30 ulp per lane — within the golden tolerance —
/// and the `#[attune]` bodies dispatch on `X64V3Token` rather than probing
/// `is_x86_feature_detected!` directly.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
mod vecops {
    use archmage::{SimdToken, X64V3Token, attune};
    use core::arch::x86_64::__m256;
    use magetypes::simd::generic::f32x8;

    /// `Pooling::MeasureInBand` with the AVX2 vector `expf`: 8-lane chunks,
    /// `err³·√err` per lane, lanes with `err ≤ 0` keep their errorline value.
    pub(crate) fn measure_in_band(
        refline: &[f32],
        dstline: &[f32],
        refmask: &[f32],
        dstmask: &[f32],
        visbase: f32,
        errorline: &mut [f32],
    ) {
        if let Some(token) = X64V3Token::summon() {
            measure_in_band_v3(
                token, refline, dstline, refmask, dstmask, visbase, errorline,
            );
        } else {
            super::measure_in_band(refline, dstline, refmask, dstmask, visbase, errorline);
        }
    }

    /// One 8-lane chunk of [`measure_in_band`]: the same lane-local ops the
    /// scalar loop runs, so each output lane matches. `vsqrtps` and
    /// `vblendvps` are exact per lane; `exp_midp` supplies the
    /// vector `expf` (mid-precision exp2 kernel, ≤~2 ulp near zero).
    #[attune(v3, import_intrinsics)]
    fn measure_in_band_chunk8(
        token: X64V3Token,
        r: &[f32; 8],
        d: &[f32; 8],
        rm: &[f32; 8],
        dm: &[f32; 8],
        el: &mut [f32; 8],
        visv: __m256,
        absmask: __m256,
    ) {
        // Compiled order: err = (|r−d|·visbase)·max(rm,dm)
        // (visbase multiplies FIRST), p = (err·√err)·err².
        let err = _mm256_mul_ps(
            _mm256_mul_ps(
                _mm256_and_ps(
                    _mm256_sub_ps(_mm256_loadu_ps(r), _mm256_loadu_ps(d)),
                    absmask,
                ),
                visv,
            ),
            _mm256_max_ps(_mm256_loadu_ps(rm), _mm256_loadu_ps(dm)),
        );
        let keep = _mm256_cmp_ps::<_CMP_GT_OQ>(err, _mm256_setzero_ps());
        let p = _mm256_mul_ps(
            _mm256_mul_ps(err, _mm256_sqrt_ps(err)),
            _mm256_mul_ps(err, err),
        );
        let mut neg = [0.0f32; 8];
        _mm256_storeu_ps(&mut neg, _mm256_xor_ps(p, _mm256_set1_ps(-0.0)));
        // Checked `exp_midp` — `exp_midp_unchecked` wraps large `p` (err ≳
        // 3.6) into garbage through the unclamped exponent build, where the
        // scalar `expf` correctly returns 0.
        let e = _mm256_loadu_ps(
            &f32x8::<X64V3Token>::from_array_t(token, neg)
                .exp_midp()
                .to_array(),
        );
        let cur = _mm256_loadu_ps(el);
        _mm256_storeu_ps(el, _mm256_blendv_ps(cur, _mm256_mul_ps(cur, e), keep));
    }

    /// The AVX2 body of [`measure_in_band`]: whole-vector loop over
    /// 8-lane chunks plus a zero-padded tail chunk (err = 0 on padding
    /// lanes → keep-mask off → blended out).
    #[attune(wrap, import_intrinsics)]
    fn measure_in_band_v3(
        token: X64V3Token,
        refline: &[f32],
        dstline: &[f32],
        refmask: &[f32],
        dstmask: &[f32],
        visbase: f32,
        errorline: &mut [f32],
    ) {
        measure_in_band_inner_v3(
            token, refline, dstline, refmask, dstmask, visbase, errorline,
        );
    }

    /// The `#[attune(v3)]` inner — callable from inside other v3 regions
    /// (e.g. `measure_bands_span_v3_t`) with no per-band dispatch boundary.
    #[attune(v3, import_intrinsics)]
    pub(crate) fn measure_in_band_inner_v3(
        token: X64V3Token,
        refline: &[f32],
        dstline: &[f32],
        refmask: &[f32],
        dstmask: &[f32],
        visbase: f32,
        errorline: &mut [f32],
    ) {
        debug_assert_eq!(refline.len(), errorline.len());
        debug_assert_eq!(dstline.len(), errorline.len());
        debug_assert_eq!(refmask.len(), errorline.len());
        debug_assert_eq!(dstmask.len(), errorline.len());
        let visv = _mm256_set1_ps(visbase);
        let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let (rc, rt) = refline.as_chunks::<8>();
        let (dc, _) = dstline.as_chunks::<8>();
        let (rmc, _) = refmask.as_chunks::<8>();
        let (dmc, _) = dstmask.as_chunks::<8>();
        let (ec, et) = errorline.as_chunks_mut::<8>();
        for i in 0..ec.len() {
            measure_in_band_chunk8(
                token, &rc[i], &dc[i], &rmc[i], &dmc[i], &mut ec[i], visv, absmask,
            );
        }
        // Partial tail: zero-pad to 8 lanes and run the same vector chunk —
        // the previous chunked loop did exactly this.
        if !et.is_empty() {
            let t = et.len();
            let off = rc.len() * 8;
            let (mut r, mut d, mut rm, mut dm) =
                ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
            let mut el = [0.0f32; 8];
            r[..t].copy_from_slice(&rt[..t]);
            d[..t].copy_from_slice(&dstline[off..off + t]);
            rm[..t].copy_from_slice(&refmask[off..off + t]);
            dm[..t].copy_from_slice(&dstmask[off..off + t]);
            el[..t].copy_from_slice(et);
            measure_in_band_chunk8(token, &r, &d, &rm, &dm, &mut el, visv, absmask);
            et.copy_from_slice(&el[..t]);
        }
    }

    /// `mapped[i] = |in[i]|·vis` raised to `e`, matching the reference's
    /// loop structure: 8-lane vector chunks, then a scalar `powf` tail.
    pub(crate) fn powf_mapped(out: &mut [f32], inp: &[f32], vis: f32, e: f32) {
        if let Some(token) = X64V3Token::summon() {
            powf_mapped_v3_t(token, out, inp, vis, e);
        } else {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = super::pf(v.abs() * vis, e);
            }
        }
    }

    /// The AVX2 body of [`powf_mapped`]: mid-precision vector `pow`
    /// (`pow_midp` — edge-checked, so `0^e` stays exact 0) over contiguous
    /// 8-lane chunks; the remainder keeps scalar `powf`.
    #[attune]
    fn powf_mapped_v3_t(token: X64V3Token, out: &mut [f32], inp: &[f32], vis: f32, e: f32) {
        powf_mapped_v3(token, out, inp, vis, e);
    }

    /// The `#[attune(v3)]` inner — callable from inside other v3 regions
    /// (e.g. `push_row_v3_t`) with no extra dispatch boundary.
    #[attune(v3)]
    pub(crate) fn powf_mapped_v3(
        token: X64V3Token,
        out: &mut [f32],
        inp: &[f32],
        vis: f32,
        e: f32,
    ) {
        type V = f32x8<X64V3Token>;
        let visv = V::splat_t(token, vis);
        let (ic, it) = V::partition_slice_t(token, inp);
        let (oc, ot) = V::partition_slice_mut_t(token, out);
        for (i, o) in ic.iter().zip(oc.iter_mut()) {
            (V::load_t(token, i).abs() * visv).pow_midp(e).store(o);
        }
        for (o, &v) in ot.iter_mut().zip(it) {
            *o = super::pf(v.abs() * vis, e);
        }
    }
}

/// aarch64 mirrors of the `vecops` x86 kernels: `f32x4<NeonToken>` lanes run
/// the identical per-lane op order, and the same magetypes mid-precision
/// exp2/pow kernels the AVX2 bodies use — same ~2 ulp tolerance class vs
/// scalar `expf`/`powf`.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
mod vecops_neon {
    use archmage::{NeonToken, SimdToken, arcane};
    use magetypes::simd::generic::f32x4;

    pub(crate) fn measure_in_band(
        refline: &[f32],
        dstline: &[f32],
        refmask: &[f32],
        dstmask: &[f32],
        visbase: f32,
        errorline: &mut [f32],
    ) {
        if let Some(token) = NeonToken::summon() {
            measure_in_band_neon(
                token, refline, dstline, refmask, dstmask, visbase, errorline,
            );
        } else {
            super::measure_in_band(refline, dstline, refmask, dstmask, visbase, errorline);
        }
    }

    /// Per-element `err = (|r−d|·visbase)·max(rm,dm)`, `errorline *=
    /// exp(−(err·√err)·err²)` — `err == 0` lanes give `exp(0) = 1`, so the
    /// scalar `err > 0` guard is absorbed (multiplying by 1 is exact).
    #[arcane]
    fn measure_in_band_neon(
        token: NeonToken,
        refline: &[f32],
        dstline: &[f32],
        refmask: &[f32],
        dstmask: &[f32],
        visbase: f32,
        errorline: &mut [f32],
    ) {
        type V = f32x4<NeonToken>;
        debug_assert_eq!(refline.len(), errorline.len());
        debug_assert_eq!(dstline.len(), errorline.len());
        debug_assert_eq!(refmask.len(), errorline.len());
        debug_assert_eq!(dstmask.len(), errorline.len());
        let visv = V::splat_t(token, visbase);
        let (rc, rt) = refline.as_chunks::<4>();
        let (dc, _) = dstline.as_chunks::<4>();
        let (rmc, _) = refmask.as_chunks::<4>();
        let (dmc, _) = dstmask.as_chunks::<4>();
        let (ec, et) = errorline.as_chunks_mut::<4>();
        for i in 0..ec.len() {
            let err = ((V::load_t(token, &rc[i]) - V::load_t(token, &dc[i])).abs() * visv)
                * V::load_t(token, &rmc[i]).max(V::load_t(token, &dmc[i]));
            let p = (err * err.sqrt()) * (err * err);
            // checked `exp_midp`: large `err` underflows toward 0, which the
            // unchecked variant wraps into garbage through the exponent
            // bit-field build.
            let e = (-p).exp_midp();
            (V::load_t(token, &ec[i]) * e).store(&mut ec[i]);
        }
        // Scalar tail — same op order as the scalar reference loop.
        let off = ec.len() * 4;
        for (i, e) in et.iter_mut().enumerate() {
            let err =
                (rt[i] - dstline[off + i]).abs() * visbase * refmask[off + i].max(dstmask[off + i]);
            if err > 0.0 {
                let p = (err * super::sq(err)) * (err * err);
                *e *= super::ef(-p);
            }
        }
    }

    /// `mapped[i] = |in[i]|·vis` raised to `e` — same `pow_midp` the AVX2
    /// body uses (edge-checked: `0^e` stays exact 0).
    pub(crate) fn powf_mapped(out: &mut [f32], inp: &[f32], vis: f32, e: f32) {
        if let Some(token) = NeonToken::summon() {
            powf_mapped_neon(token, out, inp, vis, e);
        } else {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = super::pf(v.abs() * vis, e);
            }
        }
    }

    #[arcane]
    fn powf_mapped_neon(token: NeonToken, out: &mut [f32], inp: &[f32], vis: f32, e: f32) {
        type V = f32x4<NeonToken>;
        let visv = V::splat_t(token, vis);
        let (ic, it) = V::partition_slice_t(token, inp);
        let (oc, ot) = V::partition_slice_mut_t(token, out);
        for (i, o) in ic.iter().zip(oc.iter_mut()) {
            (V::load_t(token, i).abs() * visv).pow_midp(e).store(o);
        }
        for (o, &v) in ot.iter_mut().zip(it) {
            *o = super::pf(v.abs() * vis, e);
        }
    }
}

/// Elementwise body of `BandMask::compute_mask`. `#[inline(always)]` so the
/// `target_feature` clone below compiles the identical code under
/// `fma`/`avx2`, where `mul_add` lowers to `vfmadd` instead of a
/// correctly-rounded `fmaf` libcall — same bits, ~10× cheaper per op.
#[inline(always)]
fn compute_mask_impl(
    add: &[Vec<f32>; MASK_SIZE],
    mid_row: &[f32],
    m0: f32,
    m1: f32,
    m2: f32,
    vis: f32,
    w13: usize,
    mask: &mut [f32],
    coeff: &mut [f32],
) {
    let base = BASE_VISIBILITY;
    let inv_ms2 = 1.0 / (MASK_SIZE * MASK_SIZE) as f32;
    for i in 0..w13 {
        // FMA chain order of `ComputeMaskLoop`: m0·s04, then
        // fma(m1, s13), fma(m2, d2), then fused fma(sum, 1/25, base).
        let sum = m0 * (add[0][i] + add[4][i]);
        let sum = fm(m1, add[1][i] + add[3][i], sum);
        let sum = fm(m2, add[2][i], sum);
        let den = fm(sum, inv_ms2, base);
        // `-freciprocal-math` lowers base/den to vrcpps + one NR step:
        // mask = base·(2r − den·r²).
        let r = rcp_approx(den);
        mask[i] = base * ((r + r) - (den * r) * r);
        coeff[i] = mid_row[i + (MASK_SIZE >> 1)] * vis;
    }
}

/// `compute_mask_impl` compiled with hardware FMA/AVX2, with the per-lane
/// `rcpss` reciprocal widened to `vrcpps`. The scalar `rcpss` inside the
/// autovectorization path blocked every lane (LLVM can't vectorize around
/// the partial-register intrinsic), so the whole loop ran one element at
/// a time. `vrcpps` applies the same per-lane approximation, and the
/// FMA/abs chains preserve the serial per-element op order — bit-identical
/// to the safe impl (verified by the golden tests). Token proof wrapper:
/// callers summon `X64V3Token`, which guarantees `fma`/`avx2`.
#[cfg(target_arch = "x86_64")]
#[attune]
fn compute_mask_fma_v3_t(token: X64V3Token, mask_obj: &mut BandMask, kernel: &[f32; MASK_SIZE]) {
    compute_mask_v3(token, mask_obj, kernel);
}

/// The `#[attune(v3)]` inner — callable from inside other v3 regions
/// (e.g. `push_row_v3_t`) with no extra dispatch boundary.
#[cfg(target_arch = "x86_64")]
#[attune(v3, import_intrinsics)]
fn compute_mask_v3(_token: X64V3Token, mask_obj: &mut BandMask, kernel: &[f32; MASK_SIZE]) {
    let add = &mask_obj.add_ring;
    let mid_row = &mask_obj.in_ring[MASK_SIZE >> 1];
    let (m0, m1, m2) = (kernel[0], kernel[1], kernel[2]);
    let vis = mask_obj.vis;
    let w13 = mask_obj.w13;
    let mask = &mut mask_obj.mask;
    let coeff = &mut mask_obj.coeff;
    let base = BASE_VISIBILITY;
    let inv_ms2 = 1.0 / (MASK_SIZE * MASK_SIZE) as f32;

    let m0v = _mm256_set1_ps(m0);
    let m1v = _mm256_set1_ps(m1);
    let m2v = _mm256_set1_ps(m2);
    let inv_ms2v = _mm256_set1_ps(inv_ms2);
    let basev = _mm256_set1_ps(base);
    let visv = _mm256_set1_ps(vis);
    let w = w13 & !7;
    for i in (0..w).step_by(8) {
        // `i + 8 <= w <= w13`; `add`/`mask`/`coeff` are `w13` wide and
        // `mid_row` is `w8 = w13 + MASK_SIZE - 1` wide, so the
        // `i + 2 ..= i + 9` read ends at `w + 1 <= w13 + 1 < w8`.
        let s04 = _mm256_add_ps(
            _mm256_loadu_ps(add[0][i..].first_chunk().unwrap()),
            _mm256_loadu_ps(add[4][i..].first_chunk().unwrap()),
        );
        let sum = _mm256_mul_ps(m0v, s04);
        let s13 = _mm256_add_ps(
            _mm256_loadu_ps(add[1][i..].first_chunk().unwrap()),
            _mm256_loadu_ps(add[3][i..].first_chunk().unwrap()),
        );
        let sum = _mm256_fmadd_ps(m1v, s13, sum);
        let sum = _mm256_fmadd_ps(
            m2v,
            _mm256_loadu_ps(add[2][i..].first_chunk().unwrap()),
            sum,
        );
        let den = _mm256_fmadd_ps(sum, inv_ms2v, basev);
        // mask = base·(2r − den·r²): the scalar path computes `den*r`
        // and `(den*r)*r` as separate rounds — keep them unfused for
        // parity.
        let r = _mm256_rcp_ps(den);
        let dr = _mm256_mul_ps(den, r);
        let drr = _mm256_mul_ps(dr, r);
        let t = _mm256_sub_ps(_mm256_add_ps(r, r), drr);
        _mm256_storeu_ps(
            mask[i..].first_chunk_mut().unwrap(),
            _mm256_mul_ps(basev, t),
        );
        _mm256_storeu_ps(
            coeff[i..].first_chunk_mut().unwrap(),
            _mm256_mul_ps(
                _mm256_loadu_ps(mid_row[i + (MASK_SIZE >> 1)..].first_chunk().unwrap()),
                visv,
            ),
        );
    }
    // Scalar tail — same op order as `compute_mask_impl` on the
    // remainder (the impl takes whole slices; offsets stay inline here).
    for i in w..w13 {
        let sum = m0 * (add[0][i] + add[4][i]);
        let sum = fm(m1, add[1][i] + add[3][i], sum);
        let sum = fm(m2, add[2][i], sum);
        let den = fm(sum, inv_ms2, base);
        let r = rcp_approx(den);
        mask[i] = base * ((r + r) - (den * r) * r);
        coeff[i] = mid_row[i + (MASK_SIZE >> 1)] * vis;
    }
}

/// `compute_lowpass_scalar` with the 8 output lanes in an `f32x8` — the
/// same per-element FMA chain (identical bits per lane), and it removes
/// the baseline codegen's `fmaf` libcalls that made this the long-pole
/// task in the `push_row` par sweep.
#[cfg(target_arch = "x86_64")]
#[attune]
fn compute_lowpass_v3_t(token: X64V3Token, mask_obj: &mut BandMask) {
    compute_lowpass_v3(token, mask_obj);
}

/// The `#[attune(v3)]` inner — callable from inside other v3 regions.
#[cfg(target_arch = "x86_64")]
#[attune(v3)]
fn compute_lowpass_v3(token: X64V3Token, mask_obj: &mut BandMask) {
    type V = f32x8<X64V3Token>;
    let cf4 = *mask_obj.lp.as_ref().unwrap();
    let Some(filt) = &mut mask_obj.filtered else {
        return;
    };
    for x in 0..2 {
        for y in 0..2 {
            let cf = &cf4[y][x];
            let out = &mut filt[x][y];
            let n8 = out.len() & !7;
            for k0 in (0..n8).step_by(8) {
                let mut acc = V::zero_t(token);
                #[allow(clippy::needless_range_loop)]
                for i in 0..MASK_SIZE {
                    let ring = &mask_obj.in_ring[i];
                    let r = |j: usize| V::load_t(token, ring[k0 + j..][..8].first_chunk().unwrap());
                    let t = r(4).mul_add(V::splat_t(token, cf[i][4]), {
                        r(0).mul_add(V::splat_t(token, cf[i][0]), r(1) * cf[i][1])
                    }) + r(2).mul_add(V::splat_t(token, cf[i][2]), r(3) * cf[i][3]);
                    acc += t;
                }
                acc.store(out[k0..][..8].first_chunk_mut().unwrap());
            }
            // Scalar tail — same op order as `compute_lowpass_scalar`.
            for (k, o) in out[n8..].iter_mut().enumerate() {
                let k = k + n8;
                let mut acc = 0.0f32;
                #[allow(clippy::needless_range_loop)]
                for i in 0..MASK_SIZE {
                    let r = &mask_obj.in_ring[i][k..k + MASK_SIZE];
                    let t = fm(r[4], cf[i][4], fm(r[0], cf[i][0], r[1] * cf[i][1]))
                        + fm(r[2], cf[i][2], r[3] * cf[i][3]);
                    acc += t;
                }
                *o = acc;
            }
        }
    }
}

/// Horizontal 5-tap symmetric conv of `mp` into `add`
/// (`ColumnSumConvolution` order).
#[inline(always)]
fn conv_add_impl(mp: &[f32], m0: f32, m1: f32, m2: f32, add: &mut [f32]) {
    for (i, a) in add.iter_mut().enumerate() {
        // GCC contracts to fma(m2, mp2, fma(m0, s04, m1·s13)).
        *a = fm(
            m2,
            mp[i + 2],
            fm(m0, mp[i] + mp[i + 4], m1 * (mp[i + 1] + mp[i + 3])),
        );
    }
}

/// `conv_add_impl` under an `X64V3Token` proof wrapper — the token
/// guarantees `fma`/`avx2`, so `mul_add` lowers to `vfmadd`. Same op order
/// as the safe impl; lanes stay disjoint.
#[cfg(target_arch = "x86_64")]
#[attune]
fn conv_add_fma_v3_t(_token: X64V3Token, mp: &[f32], m0: f32, m1: f32, m2: f32, add: &mut [f32]) {
    conv_add_impl(mp, m0, m1, m2, add);
}

/// Window multiply of `run_dct`: `v = v·win + avg·(1−win)`, fused.
#[inline(always)]
fn window_impl(block: &mut [[f32; 8]; 8], avg: f32) {
    for (y, r) in block.iter_mut().enumerate() {
        let win = WINDOW[y];
        for (x, v) in r.iter_mut().enumerate() {
            *v = fm(*v, win[x], avg * (1.0 - win[x]));
        }
    }
}

/// `window_impl` under an `X64V3Token` proof wrapper — the token
/// guarantees `fma`/`avx2`. Same op order as the safe impl.
#[cfg(target_arch = "x86_64")]
#[attune]
fn window_fma_v3_t(_token: X64V3Token, block: &mut [[f32; 8]; 8], avg: f32) {
    window_impl(block, avg);
}

/// `dc_mask_fixup`'s `0.5·m01 + 0.5·m10 + 0.25·m11` merge (FMA order as the
/// GCC reassociation).
#[inline(always)]
fn dc_fixup_impl(m00: &mut [f32], m01: &[f32], m10: &[f32], m11: &[f32]) {
    for i in 0..m00.len() {
        m00[i] = fm(m11[i], 0.25, (m01[i] + m10[i]) * 0.5);
    }
}

/// `dc_fixup_impl` under an `X64V3Token` proof wrapper — the token
/// guarantees `fma`/`avx2`. Same op order as the safe impl.
#[cfg(target_arch = "x86_64")]
#[attune]
fn dc_fixup_fma_v3_t(_token: X64V3Token, m00: &mut [f32], m01: &[f32], m10: &[f32], m11: &[f32]) {
    dc_fixup_impl(m00, m01, m10, m11);
}

/// Approximate reciprocal matching the x86 `vrcpps` the reference's
/// `-freciprocal-math` build emits. We now use true division as the
/// Newton–Raphson seed on every platform — post-NR it stays within ≤1 ulp
/// of the `rcpss`-seeded iterate, as documented in DIVERGENCES.md (the
/// SSE intrinsic carried an otherwise-unneeded `unsafe`/feature edge).
#[inline(always)]
fn rcp_approx(x: f32) -> f32 {
    1.0 / x
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
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        if let Some(token) = X64V3Token::summon() {
            // One dispatch boundary for the whole map→conv→mask sequence —
            // the helpers inline into `push_row_v3_t`'s v3 region.
            push_row_v3_t(token, self, kernel);
            return;
        }
        self.push_row_scalar(kernel);
    }

    fn push_row_scalar(&mut self, kernel: &[f32; MASK_SIZE]) {
        // Swap the staged row into the ring at `fill` (saturating at 4),
        // matching the C++ `m_pInput[m_ulY]->Swap(line)` — a buffer swap,
        // not a copy: `next_row` is fully rewritten by `run_dct` before the
        // next push, so taking the slot's old buffer is safe.
        let slot = self.fill.min(MASK_SIZE - 1);
        core::mem::swap(&mut self.in_ring[slot], &mut self.next_row);

        // mapped[i] = (|in[i]| · vis)^expon, full band width. For e≠1 the
        // reference's powf loop vectorizes — `powf_mapped` runs it 8 lanes
        // at a time under v3 (within ~1 ulp of the libmvec result it
        // replaces, well inside tolerance).
        let vis = self.vis;
        let e = self.expon;
        if e == 1.0 {
            for (m, &v) in self.mapped.iter_mut().zip(self.in_ring[slot].iter()) {
                *m = v.abs() * vis;
            }
        } else {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            {
                vecops::powf_mapped(&mut self.mapped, &self.in_ring[slot], vis, e);
            }
            #[cfg(all(feature = "std", target_arch = "aarch64"))]
            {
                vecops_neon::powf_mapped(&mut self.mapped, &self.in_ring[slot], vis, e);
            }
            #[cfg(not(any(
                all(feature = "std", target_arch = "x86_64"),
                all(feature = "std", target_arch = "aarch64")
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
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = X64V3Token::summon() {
            conv_add_fma_v3_t(token, mp, m0, m1, m2, add);
        } else {
            conv_add_impl(mp, m0, m1, m2, add);
        }
        #[cfg(not(target_arch = "x86_64"))]
        conv_add_impl(mp, m0, m1, m2, add);

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
        // On x86_64 with FMA the `mul_add` calls in `mask_impl` lower to
        // `vfmadd` — bitwise identical to the correctly-rounded `fmaf`
        // libcall, just ~10× cheaper. Without it the libcall path is kept.
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = X64V3Token::summon() {
            compute_mask_fma_v3_t(token, self, kernel);
            return;
        }
        self.compute_mask_scalar(kernel);
    }

    fn compute_mask_scalar(&mut self, kernel: &[f32; MASK_SIZE]) {
        compute_mask_impl(
            &self.add_ring,
            &self.in_ring[MASK_SIZE >> 1],
            kernel[0],
            kernel[1],
            kernel[2],
            self.vis,
            self.w13,
            &mut self.mask,
            &mut self.coeff,
        );
    }

    /// `Masking::ComputeLowpass` — the four 5×5 separable subbands of the
    /// DC coefficient plane (`filtered[x][y]` follows the C++ `m_pFiltered`
    /// indexing: `[x][y]` ↔ `[coeff pair]` per `GetLowpass(x,y)`).
    fn compute_lowpass(&mut self) {
        #[cfg(target_arch = "x86_64")]
        if let Some(token) = X64V3Token::summon() {
            compute_lowpass_v3_t(token, self);
            return;
        }
        self.compute_lowpass_scalar();
    }

    fn compute_lowpass_scalar(&mut self) {
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

/// `push_row_scalar`'s whole sequence inside a single v3 region: the
/// `mapped` pow, `conv_add`, `compute_mask` and `compute_lowpass` helpers
/// are all `#[attune(v3)]`/`#[inline(always)]`, so the compiler sees one
/// uninterrupted vector body per band push — one dispatch boundary per
/// `push_row` instead of three, with kernel splats hoisted and `mapped`
/// kept hot between passes. Bit-identical to the scalar path.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[attune]
fn push_row_v3_t(token: X64V3Token, b: &mut BandMask, kernel: &[f32; MASK_SIZE]) {
    let slot = b.fill.min(MASK_SIZE - 1);
    core::mem::swap(&mut b.in_ring[slot], &mut b.next_row);

    let vis = b.vis;
    let e = b.expon;
    if e == 1.0 {
        for (m, &v) in b.mapped.iter_mut().zip(b.in_ring[slot].iter()) {
            *m = v.abs() * vis;
        }
    } else {
        vecops::powf_mapped_v3(token, &mut b.mapped, &b.in_ring[slot], vis, e);
    }
    let (m0, m1, m2) = (kernel[0], kernel[1], kernel[2]);
    // `conv_add_impl` is `#[inline(always)]` — it compiles with hardware
    // FMA inside this v3 region (`conv_add_fma_v3_t`'s body).
    conv_add_impl(&b.mapped, m0, m1, m2, &mut b.add_ring[slot]);

    if b.fill == MASK_SIZE - 1 {
        compute_mask_v3(token, b, kernel);
        if b.lp.is_some() {
            compute_lowpass_v3(token, b);
        }
        b.in_ring.rotate_left(1);
        b.add_ring.rotate_left(1);
    } else {
        b.fill += 1;
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
        // Hoist the forensic env lookups out of the per-position loop —
        // getenv is a strcmp scan of environ and was ~13% of program
        // instructions at 512px.
        #[cfg(feature = "std")]
        let dumpp: Option<(usize, usize)> = std::env::var_os("DUMPP").map(|v| {
            let lo: usize = v.to_str().unwrap().parse().unwrap();
            let hi = std::env::var("DUMPP2")
                .map(|s| s.parse().unwrap())
                .unwrap_or(lo);
            (lo, hi)
        });
        #[cfg(feature = "std")]
        let dumpi = std::env::var_os("DUMPI").is_some();
        #[cfg(feature = "std")]
        let dumpw: Option<usize> = std::env::var("DUMPW").ok().and_then(|s| s.parse().ok());
        // 8-positions-in-lanes DCT for the bulk; the forensic dump paths
        // and the `w8 % 8` tail keep the per-position loop (PCALL numbering
        // and dump order are preserved there).
        #[cfg(target_arch = "x86_64")]
        let start = {
            let no_dumps = {
                #[cfg(feature = "std")]
                {
                    dumpp.is_none() && !dumpi && dumpw.is_none() && dump_targets() == (None, None)
                }
                #[cfg(not(feature = "std"))]
                {
                    true
                }
            };
            let n8 = self.w8 & !7;
            if no_dumps && let Some(token) = X64V3Token::summon() {
                run_dct_soa_v3_t(token, self);
                n8
            } else {
                0
            }
        };
        #[cfg(all(feature = "std", target_arch = "aarch64"))]
        let start = {
            let no_dumps =
                dumpp.is_none() && !dumpi && dumpw.is_none() && dump_targets() == (None, None);
            let n8 = self.w8 & !7;
            if no_dumps && let Some(token) = NeonToken::summon() {
                run_dct_soa_neon_t(token, self);
                n8
            } else {
                0
            }
        };
        #[cfg(not(any(target_arch = "x86_64", all(feature = "std", target_arch = "aarch64"))))]
        let start = 0;
        for i in start..self.w8 {
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
            if let Some((lo, hi)) = dumpp
                && pcall >= lo
                && pcall <= hi
            {
                for (y, r) in block.iter().enumerate() {
                    for (x, &v) in r.iter().enumerate() {
                        eprintln!("PIN {pcall} {y} {x} {:#010x}", v.to_bits());
                    }
                }
            }
            #[cfg(feature = "std")]
            if dumpi && i == 0 && self.first_call {
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
            #[cfg(target_arch = "x86_64")]
            if let Some(token) = X64V3Token::summon() {
                window_fma_v3_t(token, &mut block, avg);
            } else {
                window_impl(&mut block, avg);
            }
            #[cfg(not(target_arch = "x86_64"))]
            window_impl(&mut block, avg);
            #[cfg(feature = "std")]
            if dumpw == Some(pcall) {
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
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        // Same op order as the safe impl — `mul_add` lowers to `vfmadd`.
        return dct_8x8_fma_v3_t(token, block);
    }
    dct_8x8_impl(block)
}

/// `dct_8x8` compiled with hardware FMA/AVX2. The butterfly's `fm` chains
/// lower to `vfmadd` instead of `fmaf` libcalls — bitwise identical
/// (single-rounded either way), ~10x cheaper per op.
#[cfg(target_arch = "x86_64")]
#[attune]
fn dct_8x8_fma_v3_t(_token: X64V3Token, block: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    dct_8x8_impl(block)
}

#[inline(always)]
fn dct_8x8_impl(block: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
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
    if let Some(tgt) = dump_targets().0 {
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
    if let Some(tgt) = dump_targets().1
        && CUR_CALL.with(|c| c.get()) == tgt
    {
        for (y, r) in out.iter().enumerate() {
            for (x, &c) in r.iter().enumerate() {
                eprintln!("PASS2 {y} {x} {:#010x}", c.to_bits());
            }
        }
    }
    out
}

/// `DUMPA`/`DUMPB` forensic targets, cached — the `aan_dct_2d` block loop
/// polled them per 8×8 block, which was ~11% of program instructions at
/// 512px. Same semantics as before: unset → `None`, malformed → panic.
#[cfg(feature = "std")]
fn dump_targets() -> (Option<usize>, Option<usize>) {
    static T: std::sync::OnceLock<(Option<usize>, Option<usize>)> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        (
            std::env::var("DUMPA").ok().map(|t| t.parse().unwrap()),
            std::env::var("DUMPB").ok().map(|t| t.parse().unwrap()),
        )
    })
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

/// `aan_1d` with 8 consecutive window positions in the `f32x8` lanes —
/// identical arithmetic per lane, so each position's coefficients are
/// bit-identical to the scalar butterfly. `#[inline(always)]`: called
/// 16× per `run_dct_soa8` — a dispatch boundary would materialize the
/// 256 B `&[V; 8]` argument per call.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn aan_1d_x8(token: X64V3Token, b: &[f32x8<X64V3Token>; 8]) -> [f32x8<X64V3Token>; 8] {
    type V = f32x8<X64V3Token>;
    let s22 = V::splat_t(token, SQRT2_2);
    let c2v = V::splat_t(token, C2);
    let c3v = V::splat_t(token, C3);
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

    let mut out = [V::zero_t(token); 8];
    out[0] = tmp10 + tmp11;
    out[4] = tmp10 - tmp11;

    let z1n = tmp1 + (tmp13 - tmp2);
    out[2] = z1n.mul_add(s22, tmp13);
    out[6] = (-z1n).mul_add(s22, tmp13);

    let o10 = b[3] + (tmp5 - b[4]);
    let o11 = tmp6 + tmp5;
    let o12 = tmp6 + tmp7;

    let z5 = (o10 - o12) * C1;
    let z2 = o10.mul_add(c2v, z5);
    let z4 = o12.mul_add(c3v, z5);

    let z11 = o11.mul_add(s22, tmp7);
    let z13 = (-o11).mul_add(s22, tmp7);

    out[5] = z13 + z2;
    out[3] = z13 - z2;
    out[1] = z11 + z4;
    out[7] = z11 - z4;
    out
}

/// One `#[attune]` boundary for the whole position loop — the per-8 body
/// is the `#[attune(v3)]` helper below so everything inlines into a
/// single v3 region.
#[cfg(target_arch = "x86_64")]
#[attune]
fn run_dct_soa_v3_t(token: X64V3Token, comp: &mut Component) {
    let n8 = comp.w8 & !7;
    for i0 in (0..n8).step_by(8) {
        run_dct_soa8(token, comp, i0);
    }
}

/// aarch64 twin of `aan_1d_x8` — `f32x8<NeonToken>` is a zero-cost
/// `[float32x4_t; 2]` pair, so the butterfly runs the identical per-lane
/// op order at half the per-instruction width. `mul_add` lowers to
/// `vfmaq` (single-rounded), same contract as the v3 `fm`.
/// `#[inline(always)]` — see `aan_1d_x8`.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[inline(always)]
fn aan_1d_x8_neon(token: NeonToken, b: &[f32x8<NeonToken>; 8]) -> [f32x8<NeonToken>; 8] {
    type V = f32x8<NeonToken>;
    let s22 = V::splat_t(token, SQRT2_2);
    let c2v = V::splat_t(token, C2);
    let c3v = V::splat_t(token, C3);
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

    let mut out = [V::zero_t(token); 8];
    out[0] = tmp10 + tmp11;
    out[4] = tmp10 - tmp11;

    let z1n = tmp1 + (tmp13 - tmp2);
    out[2] = z1n.mul_add(s22, tmp13);
    out[6] = (-z1n).mul_add(s22, tmp13);

    let o10 = b[3] + (tmp5 - b[4]);
    let o11 = tmp6 + tmp5;
    let o12 = tmp6 + tmp7;

    let z5 = (o10 - o12) * C1;
    let z2 = o10.mul_add(c2v, z5);
    let z4 = o12.mul_add(c3v, z5);

    let z11 = o11.mul_add(s22, tmp7);
    let z13 = (-o11).mul_add(s22, tmp7);

    out[5] = z13 + z2;
    out[3] = z13 - z2;
    out[1] = z11 + z4;
    out[7] = z11 - z4;
    out
}

/// aarch64 twin of `run_dct_soa8` — same eight-positions-in-lanes layout,
/// one contiguous 8-wide store per band row. Per-lane arithmetic order is
/// identical to the scalar loop, so results are bitwise equal.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[attune]
fn run_dct_soa8_neon(token: NeonToken, comp: &mut Component, i0: usize) {
    type V = f32x8<NeonToken>;
    // `i0 + x + 8 <= i0 + 15 <= w8 + 7 < w` whenever `i0 + 8 <= w8`.
    let mut b: [[V; 8]; 8] = core::array::from_fn(|y| {
        core::array::from_fn(|x| {
            V::load_t(token, comp.ring[y][i0 + x..][..8].first_chunk().unwrap())
        })
    });
    let s: [V; 8] = core::array::from_fn(|x| {
        ((b[0][x] + b[7][x]) + (b[5][x] + b[6][x])) + ((b[1][x] + b[2][x]) + (b[3][x] + b[4][x]))
    });
    let avg = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
    let avg = avg * (1.0f32 / 64.0);
    for y in 0..8 {
        for x in 0..8 {
            let w = WINDOW[y][x];
            b[y][x] = b[y][x].mul_add(V::splat_t(token, w), avg * (1.0 - w));
        }
    }
    // Pass 1 (vertical): `vx[x][k]` — built by `from_fn`, see the x86
    // twin for the rationale (a zero-init pass preceded every block).
    let vx: [[V; 8]; 8] = core::array::from_fn(|x| {
        aan_1d_x8_neon(
            token,
            &[
                b[0][x], b[1][x], b[2][x], b[3][x], b[4][x], b[5][x], b[6][x], b[7][x],
            ],
        )
    });
    for vix in 0..8 {
        let row: [V; 8] = core::array::from_fn(|x| vx[x][vix]);
        let hv = aan_1d_x8_neon(token, &row);
        for (u, &c) in hv.iter().enumerate() {
            c.store(
                comp.bands[u][vix].next_row[i0..][..8]
                    .first_chunk_mut()
                    .unwrap(),
            );
        }
    }
}

/// One `#[attune]` boundary for the whole position loop, aarch64 side.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[attune]
fn run_dct_soa_neon_t(token: NeonToken, comp: &mut Component) {
    let n8 = comp.w8 & !7;
    for i0 in (0..n8).step_by(8) {
        run_dct_soa8_neon(token, comp, i0);
    }
}

/// Eight-position sliding DCT: `run_dct` for `i in i0..i0+8` with the
/// window positions in the vector lanes — `b[y][x]` lane `j` is the
/// scalar path's `block[y][x]` at position `i0+j`. Both AAN passes stay
/// elementwise (no register-file transposes) and each band's staged row
/// gets one contiguous 8-wide store instead of eight scattered element
/// writes. Per-lane arithmetic order is identical to the scalar loop, so
/// results are bitwise equal.
#[cfg(target_arch = "x86_64")]
#[attune(v3)]
fn run_dct_soa8(token: X64V3Token, comp: &mut Component, i0: usize) {
    type V = f32x8<X64V3Token>;
    // `i0 + x + 8 <= i0 + 15 <= w8 + 7 < w` whenever `i0 + 8 <= w8`.
    let mut b: [[V; 8]; 8] = core::array::from_fn(|y| {
        core::array::from_fn(|x| {
            V::load_t(token, comp.ring[y][i0 + x..][..8].first_chunk().unwrap())
        })
    });
    // Per-position block mean — the same column-sum regrouping as the
    // scalar loop, evaluated on vectors so each lane sums its own block.
    let s: [V; 8] = core::array::from_fn(|x| {
        ((b[0][x] + b[7][x]) + (b[5][x] + b[6][x])) + ((b[1][x] + b[2][x]) + (b[3][x] + b[4][x]))
    });
    let avg = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
    // `/64.0` is exact (power of two) — the broadcast multiply is bitwise.
    let avg = avg * (1.0f32 / 64.0);
    // Window multiply, `v = v·win + avg·(1−win)` — same fm() order.
    for y in 0..8 {
        for x in 0..8 {
            let w = WINDOW[y][x];
            b[y][x] = b[y][x].mul_add(V::splat_t(token, w), avg * (1.0 - w));
        }
    }
    // Pass 1 (vertical): `aan_1d` per column — `vx[x][k]` lane `j` is
    // vertical frequency `k` of column `x` at position `i0+j`. Built by
    // `from_fn` so no zero-init pass precedes it (a 2 KB memset per
    // position block was ~10 % of this kernel's time).
    let vx: [[V; 8]; 8] = core::array::from_fn(|x| {
        aan_1d_x8(
            token,
            &[
                b[0][x], b[1][x], b[2][x], b[3][x], b[4][x], b[5][x], b[6][x], b[7][x],
            ],
        )
    });
    // Pass 2 (horizontal): `aan_1d` per vertical-frequency row —
    // `hv[u]` is horizontal frequency `u`, stored contiguously.
    for vix in 0..8 {
        let row: [V; 8] = core::array::from_fn(|x| vx[x][vix]);
        let hv = aan_1d_x8(token, &row);
        for (u, &c) in hv.iter().enumerate() {
            c.store(
                comp.bands[u][vix].next_row[i0..][..8]
                    .first_chunk_mut()
                    .unwrap(),
            );
        }
    }
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
/// `N_AVX2` clone's disassembly). On x86_64+std the 8-lane `expf` runs
/// through `vecops` (`exp_midp` — within ~2 ulp of glibc's
/// `_ZGVdN8v_expf`, and of scalar `expf`, on the inputs this loop sees).
#[allow(clippy::too_many_arguments)]
fn measure_in_band(
    refline: &[f32],
    dstline: &[f32],
    refmask: &[f32],
    dstmask: &[f32],
    visbase: f32,
    errorline: &mut [f32],
) {
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    {
        vecops::measure_in_band(refline, dstline, refmask, dstmask, visbase, errorline);
        return;
    }
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    {
        vecops_neon::measure_in_band(refline, dstline, refmask, dstmask, visbase, errorline);
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
    // GCC `-ffast-math` reassociates to fma(m11, 0.25, (m01+m10)·0.5).
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        dc_fixup_fma_v3_t(token, m00, m01, m10, m11);
    } else {
        dc_fixup_impl(m00, m01, m10, m11);
    }
    #[cfg(not(target_arch = "x86_64"))]
    dc_fixup_impl(m00, m01, m10, m11);
}

/// True when any of the bit-forensic `DUMPT`/`DUMPP`/`DUMPI`/`DUMPW` env
/// vars is set. The `parallel` path runs the serial code order then so
/// call numbering and errorline traces stay deterministic. The result is
/// frozen at first read (`OnceLock`): setting a DUMP var mid-process no
/// longer takes effect — forensic runs always export them upfront.
#[cfg(feature = "parallel")]
fn dump_env_set() -> bool {
    static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DUMP.get_or_init(|| {
        std::env::var_os("DUMPT").is_some()
            || std::env::var_os("DUMPP").is_some()
            || std::env::var_os("DUMPI").is_some()
            || std::env::var_os("DUMPW").is_some()
    })
}

/// The `(c, yy, x)` band-measure sweep restricted to `errorline[lo..hi]`.
/// Element `i`'s `*=` sequence follows the serial band order, so striping
/// `errorline` into contiguous per-thread ranges is bit-exact against the
/// serial loop (each element is folded by exactly one thread, in order).
#[cfg(feature = "parallel")]
fn measure_bands_span(
    comps_r: &[Component; 3],
    comps_d: &[Component; 3],
    lo: usize,
    hi: usize,
    errorline: &mut [f32],
) {
    for c in 0..3 {
        for yy in 0..8 {
            for x in 0..8 {
                let br = &comps_r[c].bands[x][yy];
                let bd = &comps_d[c].bands[x][yy];
                if c == 0 && x == 0 && yy == 0 {
                    // EXTENDED_FILTER: the Y-DC band is measured on its
                    // four 5×5 low/high subbands (same factors as the
                    // serial loop).
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
                            &fr[a][b][lo..hi],
                            &fd[a][b][lo..hi],
                            &br.mask[lo..hi],
                            &bd.mask[lo..hi],
                            vb,
                            errorline,
                        );
                    }
                } else {
                    measure_in_band(
                        &br.coeff[lo..hi],
                        &bd.coeff[lo..hi],
                        &br.mask[lo..hi],
                        &bd.mask[lo..hi],
                        1.0,
                        errorline,
                    );
                }
            }
        }
    }
}

/// `measure_bands_span` inside a single v3 region — the per-band
/// `measure_in_band` calls inline (`measure_in_band_inner_v3`) instead of
/// each paying a dispatch boundary. Same element/fold order → bit-exact.
#[cfg(all(feature = "std", feature = "parallel", target_arch = "x86_64"))]
#[attune]
fn measure_bands_span_v3_t(
    token: X64V3Token,
    comps_r: &[Component; 3],
    comps_d: &[Component; 3],
    lo: usize,
    hi: usize,
    errorline: &mut [f32],
) {
    for c in 0..3 {
        for yy in 0..8 {
            for x in 0..8 {
                let br = &comps_r[c].bands[x][yy];
                let bd = &comps_d[c].bands[x][yy];
                if c == 0 && x == 0 && yy == 0 {
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
                        vecops::measure_in_band_inner_v3(
                            token,
                            &fr[a][b][lo..hi],
                            &fd[a][b][lo..hi],
                            &br.mask[lo..hi],
                            &bd.mask[lo..hi],
                            vb,
                            errorline,
                        );
                    }
                } else {
                    vecops::measure_in_band_inner_v3(
                        token,
                        &br.coeff[lo..hi],
                        &bd.coeff[lo..hi],
                        &br.mask[lo..hi],
                        &bd.mask[lo..hi],
                        1.0,
                        errorline,
                    );
                }
            }
        }
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

        // `par` gates every rayon dispatch below: a 1-thread pool runs the
        // serial code verbatim — no per-row Vec collects, no par bridge,
        // identical bit order. (The `parallel` feature alone must not make
        // single-threaded scoring slower.)
        #[cfg(feature = "parallel")]
        let par = rayon::current_num_threads() > 1;
        #[cfg(not(feature = "parallel"))]
        let par = false;
        let _ = par;
        let ready = {
            #[cfg(feature = "parallel")]
            if par {
                // Components are independent ring buffers; the bit pattern
                // of each comp's output is unchanged by running them on
                // threads. `ready` keeps the serial semantics — the last
                // comp's flag.
                let vr: Vec<bool> = img_r
                    .comps
                    .par_iter_mut()
                    .zip(rows_r.par_iter())
                    .map(|(c, r)| c.push_line(r))
                    .collect();
                let _vd: Vec<bool> = img_d
                    .comps
                    .par_iter_mut()
                    .zip(rows_d.par_iter())
                    .map(|(c, r)| c.push_line(r))
                    .collect();
                vr[2]
            } else {
                let mut ready = false;
                for c in 0..3 {
                    ready = img_r.comps[c].push_line(&rows_r[c]);
                    img_d.comps[c].push_line(&rows_d[c]);
                }
                ready
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut ready = false;
                for c in 0..3 {
                    ready = img_r.comps[c].push_line(&rows_r[c]);
                    img_d.comps[c].push_line(&rows_d[c]);
                }
                ready
            }
        };
        if ready {
            // SplitWork: run the sliding DCT (Component::Run) and push the
            // 64 band rows into their masking pipelines (Pooling::Run).
            #[cfg(feature = "parallel")]
            {
                // Each (image, comp) runs its own sliding-window DCT —
                // disjoint state, identical results. The DUMPP/DUMPI/DUMPW
                // forensic env vars fall back to the serial order so PCALL
                // numbering stays deterministic for bit-level diffing.
                if par && !dump_env_set() {
                    // Each (image, comp) is a self-contained chain —
                    // `run_dct` then 64 band `push_row`s — on disjoint
                    // state. Six fat tasks per row beat ~390 tiny ones:
                    // no intra-phase rendezvous, and the per-thread
                    // working set stays inside one comp's 64 BandMask
                    // rings (cache-local) instead of interleaving all
                    // ~4600 scattered rows across the pool.
                    rayon::scope(|s| {
                        for comp in img_r.comps.iter_mut().chain(img_d.comps.iter_mut()) {
                            s.spawn(|_| {
                                comp.run_dct();
                                // `with_min_len(16)` on the 64 bands → 4
                                // chunks/comp = 24 stealable tasks total —
                                // still comp-local (same 64-ring working set).
                                comp.bands
                                    .as_flattened_mut()
                                    .into_par_iter()
                                    .with_min_len(16)
                                    .for_each(|b| b.push_row(&kernel));
                            });
                        }
                    });
                } else {
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
            }
            #[cfg(not(feature = "parallel"))]
            {
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
        }

        // Main measure loop: active from image line 12 on (12-line warmup).
        if y >= 12 {
            for e in errorline.iter_mut() {
                *e = 1.0;
            }
            dc_mask_fixup(&mut img_r);
            dc_mask_fixup(&mut img_d);
            #[cfg(feature = "std")]
            let (dline, tgt_i) = {
                static DT: std::sync::OnceLock<(Option<usize>, usize)> = std::sync::OnceLock::new();
                *DT.get_or_init(|| {
                    (
                        std::env::var("DUMPT").ok().and_then(|s| s.parse().ok()),
                        std::env::var("DUMPTI")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0),
                    )
                })
            };
            #[cfg(feature = "parallel")]
            let serial_measure = !par || dump_env_set();
            #[cfg(not(feature = "parallel"))]
            let serial_measure = true;
            if serial_measure {
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
                                eprintln!(
                                    "EL c{c} x{x} y{yy} -> {:08x}",
                                    errorline[tgt_i].to_bits()
                                );
                            }
                        }
                    }
                }
            }
            #[cfg(feature = "parallel")]
            if !serial_measure {
                // errorline[i] *= ef(-p) is a per-element multiplicative
                // fold, so striping the row into contiguous ranges keeps
                // every element's `*=` sequence in the serial (c,yy,x)
                // band order — bit-identical to the serial loop.
                let w = errorline.len();
                // Each stripe re-iterates all 384 band planes, so keep
                // spans ≥128 elems for the v3 kernel's 8-lane runs to
                // amortize per-band prologue/tail work.
                let stripes = (w / 128).max(1).min(rayon::current_num_threads() * 2);
                let stripe_len = w.div_ceil(stripes);
                let (cr, cd) = (&img_r.comps, &img_d.comps);
                errorline
                    .par_chunks_mut(stripe_len)
                    .enumerate()
                    .for_each(|(si, el)| {
                        let lo = si * stripe_len;
                        #[cfg(target_arch = "x86_64")]
                        if let Some(token) = X64V3Token::summon() {
                            measure_bands_span_v3_t(token, cr, cd, lo, lo + el.len(), el);
                            return;
                        }
                        measure_bands_span(cr, cd, lo, lo + el.len(), el);
                    });
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
/// The AIC2026 `mDCT-PSNR` column is the plain dB score; its
/// `proposal-mDCTPSNR` column is this remap of it (all 9,618 rows agree to
/// 5e-9, checked 2026-10-09).
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

    /// NEON `measure_in_band`/`powf_mapped` vs the scalar reference loops —
    /// `exp_midp`/`pow_midp` are ~2 ulp off libm, so the check is a
    /// per-element relative bound rather than bit equality (aarch64 carries
    /// no bit-locked golden anyway).
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    #[test]
    fn neon_vecops_match_scalar() {
        let n = 1029; // non-multiple of 4 for a real tail
        let rf: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 4.0).collect();
        let ds: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007).cos() * 3.0).collect();
        let rm: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 + 0.5) * 0.05).collect();
        let dm: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 + 0.25) * 0.04).collect();

        // measure_in_band: scalar reference computed inline.
        let mut a = alloc::vec![1.0f32; n];
        let mut b = a.clone();
        for i in 0..n {
            let err = (rf[i] - ds[i]).abs() * 1.37 * rm[i].max(dm[i]);
            if err > 0.0 {
                let p = (err * sq(err)) * (err * err);
                a[i] *= ef(-p);
            }
        }
        vecops_neon::measure_in_band(&rf, &ds, &rm, &dm, 1.37, &mut b);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                (x - y).abs() <= 4e-6 * x.abs().max(1.0),
                "measure index {i}: scalar {x} vs neon {y}"
            );
        }

        // powf_mapped (e = 0.8 band exponent).
        let inp: Vec<f32> = (0..n).map(|i| (i as f32 * 0.031).sin() * 2.0).collect();
        let mut a = alloc::vec![0.0f32; n];
        let mut b = a.clone();
        for (m, &v) in a.iter_mut().zip(&inp) {
            *m = pf(v.abs() * 1.9, 0.8);
        }
        vecops_neon::powf_mapped(&mut b, &inp, 1.9, 0.8);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                (x - y).abs() <= 1e-5 * x.abs().max(1.0),
                "powf index {i}: scalar {x} vs neon {y}"
            );
        }
    }
}

/// The mDCT-PSNR metric as a generic scorer — accepts `RGB8_SRGB`
/// `PixelSlice` inputs (strided repacked). Headline is dB PSNR — `+inf`
/// on identical inputs, matching `dctpsnr` default output.
#[derive(Debug, Default, Clone, Copy)]
pub struct MdctPsnr;

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::LengthMismatch => za::ErrorKind::InvalidInput,
            Error::ImageTooSmall => za::ErrorKind::TooSmall,
        }
    }
}

/// Output revision of the generic-surface scorer(s) in this crate.
const SCORE_REVISION: u32 = 1;

impl za::Metric for MdctPsnr {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("mdctpsnr", za::implementation!(SCORE_REVISION))
            .with_reference("mDCTpsnr dctpsnr (Richter 2009), default build")
            .with_config("color", "linear")
            .with_config("detection", "ahumada")
    }

    fn outputs(&self) -> alloc::borrow::Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] =
            &[za::ScoreSpec::new("mdctpsnr", za::ScoreKind::Decibels)];
        alloc::borrow::Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<alloc::boxed::Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map("mdctpsnr")?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let (rp, dp) = (za::contiguous_rgb8(&r), za::contiguous_rgb8(&d));
        let s = mdct_psnr_srgb8(&rp, &dp, r.width, r.height)
            .metric_err_with("mdctpsnr", Error::kind)?;
        Ok(alloc::boxed::Box::new(za::ScalarEval::single(
            "mdctpsnr",
            f64::from(s),
        )))
    }
}
