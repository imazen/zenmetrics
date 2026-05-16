//! Neighborhood lookup helper — the 9 spatial offsets that define the
//! 3×3 block in the reference paper, indexed in the same order the
//! Python reference uses (the `(ny, nx)` outer loop ordering in
//! `info_content_weight_map`).
//!
//! The Python loop:
//!
//! ```python
//! for ny in (-1, 0, 1):
//!     for nx in (-1, 0, 1):
//!         # Y[:, n++] = roll(temp, [ny, nx])[Ly:Ly+nblv, Lx:Lx+nblh]
//! ```
//!
//! After cropping by `[Ly:Ly+nblv, Lx:Lx+nblh] = [1:1+nblv, 1:1+nblh]`
//! the value at central index `(py, px)` becomes `temp[py + 1 - ny,
//! px + 1 - nx]`. So in central-coordinates `(py, px)`, index `n` reads
//! offset `(1 - ny, 1 - nx)` from the original buffer. Iteration order:
//!
//! | n | (ny, nx) | offset (dy, dx) |
//! |---|----------|-----------------|
//! | 0 | (-1, -1) | ( 2,  2)        |
//! | 1 | (-1,  0) | ( 2,  1)        |
//! | 2 | (-1,  1) | ( 2,  0)        |
//! | 3 | ( 0, -1) | ( 1,  2)        |
//! | 4 | ( 0,  0) | ( 1,  1) ← center |
//! | 5 | ( 0,  1) | ( 1,  0)        |
//! | 6 | ( 1, -1) | ( 0,  2)        |
//! | 7 | ( 1,  0) | ( 0,  1)        |
//! | 8 | ( 1,  1) | ( 0,  0)        |
//!
//! Index 9 (when `parent = true`): reads `parent_band[py + 1, px + 1]`.

/// Spatial offsets (dy, dx) for the 9 neighborhood positions.
/// Crate-private constant — kernels expand the table inline.
pub const OFFS: [(i32, i32); 9] = [
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 2),
    (1, 1),
    (1, 0),
    (0, 2),
    (0, 1),
    (0, 0),
];

/// Maximum neighborhood size including parent.
pub const MAX_N: usize = 10;
