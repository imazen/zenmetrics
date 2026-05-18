//! Currently empty — the per-crate `compute_pixels` paths already
//! handle `PixelSlice` conversion. The umbrella forwards a
//! caller-provided slice straight through to the matching variant in
//! [`crate::metric`].
//!
//! Reserved for future shared-conversion plumbing (e.g. converting
//! once to sRGB-RGB8 and reusing the buffer across multiple metric
//! invocations on the same pair).
