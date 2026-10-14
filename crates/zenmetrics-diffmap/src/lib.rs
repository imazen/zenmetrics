//! Per-pixel metric maps — the object-safe [`Diffmap`] trait.
//!
//! Every metric crate implements `Diffmap` on its own struct, so the
//! metric-specific extras (butteraugli's pnorm variants, HDR-VDP's
//! JOD/`P_pred`, FSIM's chroma score…) travel inside the impl while
//! callers hold `Box<dyn Diffmap>` / `Arc<dyn Diffmap>` uniformly.
//!
//! The pixel storage is a [`zenpixels::PixelCow`] of `Gray<f32>` —
//! implementations may borrow scorer scratch (zero-copy) or own a
//! `PixelBuffer`; width/height/stride ride inside the slice.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
extern crate alloc;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use core::iter::Iterator;
use zenpixels::PixelCow;

/// A metric's per-pixel map — "where is the difference/similarity".
///
/// Object-safe: score functions return `Box<dyn Diffmap>` (owned) when
/// callers need polymorphism, or a concrete `impl Diffmap` when they
/// want the metric's own fields too.
///
/// ## Semantics
///
/// `similarity()` is the map's polarity contract — canonical upstream
/// map conventions differ, so callers that show maps or fold them into
/// scores must consult it:
///
/// - `false` — higher = *more error/more visible difference*
///   (butteraugli diffmap, cvvdp masked-error map, HDR-VDP `P_map`,
///   PSNR-style error).
/// - `true` — higher = *more similar* (FSIM `FSIMmap`, HaarPSI
///   similarity map, SSIM/MS-SSIM `ssim_map`, GMSD's `gms` map,
///   VSI quality map, IW-SSIM).
///
/// ## Extending the trait
///
/// New methods always ship with a default impl — adding a method is
/// additive for implementors.
pub trait Diffmap {
    /// The map as `Gray<f32>` pixels, row-major
    /// (`pixels().as_slice().descriptor()` is
    /// `PixelDescriptor::GRAYF32_LINEAR`). Width, height and stride live
    /// in the slice — the map's resolution may legitimately differ from
    /// the input image (e.g. GMSD is `w/2 × h/2`); check
    /// `pixels().width()`/`rows()` instead of assuming.
    ///
    /// `PixelCow::Borrowed` means the view aliases scorer-owned scratch —
    /// cheap, but tied to the impl's borrow. `Owned` is standalone.
    ///
    /// The erased (`()`) representation avoids a `rgb`-crate dep in leaf
    /// metrics — consumers read `&[f32]` rows via `as_strided_bytes()` +
    /// `bytemuck`, or `try_typed::<rgb::Gray<f32>>()` when they have it.
    fn pixels(&self) -> PixelCow<'_, ()>;

    /// Polarity: `true` = higher is *more similar*, `false` = higher is
    /// *more error*.
    fn similarity(&self) -> bool;

    /// The scalar score computed alongside the map (lets consumers check
    /// map-vs-score coherence without re-running the metric).
    fn score(&self) -> f64;

    /// Metric-specific metadata as `(key, value)` pairs with stable
    /// `'static` keys — e.g. butteraugli's `pnorm`/`max_norm`, HDR-VDP's
    /// `p_det`/`jod`, per-scale labels. Default: empty.
    fn extras(&self) -> Box<dyn Iterator<Item = (Cow<'static, str>, Cow<'static, str>)> + '_> {
        Box::new(::core::iter::empty())
    }
}

/// View a packed `w × h` `f32` plane as a `GRAYF32_LINEAR` pixel slice —
/// the shape every leaf metric hands to its `Diffmap::pixels()`.
///
/// # Panics
/// If `data.len() != w·h`.
#[must_use]
pub fn gray_f32_slice(data: &[f32], w: u32, h: u32) -> zenpixels::PixelSlice<'_> {
    assert_eq!(
        data.len(),
        w as usize * h as usize,
        "diffmap slice len != w*h"
    );
    let bytes: &[u8] = bytemuck::cast_slice(data);
    zenpixels::PixelSlice::new(
        bytes,
        w,
        h,
        w as usize * core::mem::size_of::<f32>(),
        zenpixels::PixelDescriptor::GRAYF32_LINEAR,
    )
    .expect("packed f32 slice always validates as GRAYF32_LINEAR")
}
