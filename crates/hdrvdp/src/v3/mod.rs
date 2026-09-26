//! HDR-VDP-3 — the third-generation HDR visual difference predictor,
//! implemented natively in Rust (`f64` end to end).
//!
//! This is a **different metric** from the crate's HDR-VDP-2: a new
//! calibration (the `VDP3/` tree of `jpeg-ai-qaf`, a NumPy port of
//! HDR-VDP-3.0.7), an isotropic `sp0` pyramid instead of the oriented
//! `sp3`, a different masking model (`min(|T|,|R|)` pooled 3×3 with
//! neighbouring-band terms rather than the v2 GSM energy model), a
//! different JND non-linearity construction, and the published
//! `Q_JOD = 10 − 0.52·max(10−Q,0)^1.2812` correlate.
//!
//! ## The explicit-inputs contract
//!
//! HDR-VDP-3 is *sensitive to viewing conditions* — the same pixel pair at
//! a different `pixels_per_degree`, display model, surround or observer age
//! is a different problem and gives a different score. Where the reference
//! silently defaults these, this API requires them:
//!
//! * [`ViewingConditions::pixels_per_degree`] — required; derive it from
//!   geometry via [`pix_per_deg`].
//! * [`ViewingConditions::surround`] — required ([`Surround::None`] =
//!   upstream default `'none'`).
//! * [`ViewingConditions::observer_age`] — required; the three age models
//!   it feeds are on by default (upstream's hidden `24`).
//! * [`InputEncoding`] — how to read the pixels; no auto-detection.
//! * [`Emission`] — the display spectral model;
//!   [`Emission::default_for`] names upstream's per-encoding implicit
//!   choice but the field must be *set*, not inherited.
//! * [`Task`] — `quality`/`side-by-side`/`flicker` calibration bundles.
//!
//! ## Differences from upstream to be aware of
//!
//! * `pixels_per_degree < 4` is rejected at [`Params::new`] (upstream
//!   gives no usable pyramid below it).
//! * `min(w, h) < 13` yields the 0-level two-band pyramid (`maxPyrHt`
//!   clamps, as upstream does) rather than erroring; `ImageTooSmall` only
//!   guards the `ppd ≤ 2` degenerate case that `Params` already rejects.
//! * `Surround::PerChannel` is implemented correctly (upstream passes a
//!   per-channel vector to `np.pad`'s `constant_values`, which does not
//!   broadcast the way it needs — the option is effectively broken there).
//! * The HLG helper reproduces upstream's `TF_HLG.decode_float`; the
//!   harness's integer `decode` wrapper is noted in [`ingress`].

mod fft64;
mod gauss;
pub mod ingress;
mod metric;
pub mod params;
mod pathway;
mod spectral;
mod spyr0;
#[cfg(test)]
mod stage_test;

pub use metric::{HdrVdp3Result, hdrvdp3};
pub use params::{
    DisplayPreset, Emission, InputEncoding, Options, Params, Surround, Task, ViewingConditions,
    pix_per_deg,
};

/// Stable column-name identifier for sweep sidecars:
/// `hdrvdp3_jod_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time
/// via `HDRVDP3_IMPL_TAG`). The `jod` infix marks the [0,10] `Q_JOD` scale —
/// distinct from [`crate::HDRVDP_COLUMN_NAME`]'s `res.Q` [0,100]; these are
/// different metrics and must never share a column.
pub const HDRVDP3_COLUMN_NAME: &str = match option_env!("HDRVDP3_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "hdrvdp3_jod_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH"),
    ),
};
