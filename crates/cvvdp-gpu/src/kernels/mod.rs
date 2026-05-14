//! GPU kernels for the ColorVideoVDP still-image pipeline.
//!
//! Pipeline order (per reference/distorted side):
//!
//! 1. `color`   — sRGB packed-u8 → linear → DKL opponent planar f32
//!                via per-row display model (luminance + EOTF) and the
//!                cvvdp RGB→LMS→DKL matrix product.
//! 2. `pyramid` — per-channel Laplacian decomposition (downscale +
//!                upscale + subtract). Produces `n_levels` band buffers
//!                per channel.
//! 3. `csf`     — per-band sensitivity weighting (castleCSF for the
//!                achromatic channel, chrom variants for RG/VY).
//! 4. `masking` — within-channel + cross-channel contrast masking.
//!                Produces per-pixel masked differences per band.
//! 5. `pool`    — Minkowski accumulation per band, then per channel.
//!                Per-band partials are produced here; the host-side
//!                folder combines them into the scalar `D` and applies
//!                the JOD mapping.
//! 6. `reduce`  — common reduction utilities (per-column f64 accums,
//!                final host fold) shared by `pool` and any per-band
//!                debug taps.
//!
//! Numerical parity target: bit-stable to the Python `pycvvdp`
//! reference's float32 path. Per-thread accumulators stay in f64
//! where the reference uses f64 reductions; otherwise f32.

pub mod color;
pub mod csf;
pub mod masking;
pub mod pool;
pub mod pyramid;
pub mod reduce;
