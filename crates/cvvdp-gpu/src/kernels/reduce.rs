//! Shared reduction utilities — per-band Minkowski accumulators and
//! the host fold that maps them to the scalar `D` distortion.
//!
//! Kept separate from `pool` so debug taps (per-band JOD contributions,
//! per-pixel masked-difference dumps) can reuse the same per-column
//! partial layout without re-implementing the host fold.
//!
//! Compiling stub — body lands alongside the per-band pooling kernel.

#![allow(unused_imports)]

use cubecl::prelude::*;
