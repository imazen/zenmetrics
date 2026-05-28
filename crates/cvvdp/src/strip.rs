//! Mode B / Mode E K_SPLIT strip-walker helpers — CPU port.
//!
//! Ported from `cvvdp-gpu::pipeline::{mode_b_k_split,
//! mode_b_halo_at_level, mode_b_strip_h_at_level}` (see commits #81 + #84
//! on master log). The math is identical to the GPU helpers — the K_SPLIT
//! decision and per-level buffer sizing have to agree across both
//! implementations so the strip walker's parity gates against the
//! GPU reference are meaningful.
//!
//! ## Why K_SPLIT
//!
//! cvvdp's pyramid has 9 levels at the 4K standard viewing geometry. At
//! a typical strip body height `h_body = 512`, the *body at level k*
//! halves to `body_k = h_body >> k`:
//!
//! | k | body_k | PU blur halo at level k |
//! |---|--------|-------------------------|
//! | 0 | 512    | ±6 (negligible vs body) |
//! | 1 | 256    | ±6                       |
//! | 2 | 128    | ±6                       |
//! | 3 | 64     | ±6                       |
//! | 4 | 32     | ±6                       |
//! | 5 | 16     | ±6 (body ≈ 2·halo)       |
//! | 6 | 8      | ±6 (body < 2·halo)       |
//! | 7 | 4      | ±6 (body << halo)        |
//! | 8 | 2      | ±6 (body << halo)        |
//!
//! Once `body_k < 2·halo` the "strip" stops being a strip — the halo
//! covers more rows than the body, so the strip buffer is bigger than
//! the full image at that level. The K_SPLIT walker handles this by
//! using **per-strip storage for shallow bands `k < k_split`** and
//! **full-image storage for deep bands `k >= k_split`**. The deep bands
//! are tiny in absolute terms — level 8 at 4096² is 16×16 = 256
//! pixels per channel — so full-image storage there costs ~kB, not
//! the full-pyramid GB.
//!
//! ## API contract
//!
//! - [`mode_b_k_split`] picks the split point: largest k such that
//!   `body_k >= MODE_B_DEEP_THRESHOLD = 12` (twice the PU blur radius).
//! - [`mode_b_halo_at_level`] returns the per-level halo a single band
//!   needs at its own resolution for its own band-loop (8 = PU radius
//!   6 + 2-tap downscale slack).
//! - [`mode_b_strip_h_at_level`] returns the back-projected buffer
//!   height at level k accounting for the reduce chain (level k must
//!   feed level k+1 with enough source rows).
//!
//! These helpers are bit-identical to the GPU pipeline's versions so
//! sizing tables agree across implementations.

/// Threshold (rows at level k) below which a level falls into the
/// "deep" K_SPLIT band. Set to twice the σ=3 PU blur radius (6 rows)
/// so the strip body is at least 2 halos wide.
const MODE_B_DEEP_THRESHOLD: u32 = 12;

/// Per-level **band-resolution** halo for the Mode B strip walker.
///
/// Halo a single level needs *at its own resolution* for its own band
/// loop (PU blur reads ±6, pyramid downscale reads ±2, so 8 rows
/// covers both). Does NOT account for back-projection through the
/// reduce chain — see [`mode_b_strip_h_at_level`] for the correct
/// buffer height accounting cross-level reduce halos.
///
/// Kept as a separate helper because some callers (e.g., the band-
/// loop's own per-strip halo math) need the level-local value.
/// Allocator + estimator callers should prefer
/// [`mode_b_strip_h_at_level`].
#[doc(hidden)]
#[must_use]
pub fn mode_b_halo_at_level(k: u32, k_split: u32) -> u32 {
    if k >= k_split {
        0 // deep levels use full-image storage, no halo padding
    } else {
        // PU blur radius (6) + 2-tap downscale slack at this level.
        8
    }
}

/// Pick the K_SPLIT level for the Mode B walker at the given strip
/// body height `h_body` and pyramid depth `n_levels`.
///
/// Returns the largest `k_split` such that `h_body >> k_split >=
/// MODE_B_DEEP_THRESHOLD = 12`. For `h_body = 512, n_levels = 9` this
/// returns `k_split = 6` — bands 0..6 are strip-aware, bands 6..9 use
/// full-image storage at their (small) per-level resolution. For
/// smaller `h_body` the split shifts down accordingly.
///
/// Bit-identical to `cvvdp_gpu::pipeline::mode_b_k_split` so the GPU
/// and CPU walkers always agree on which levels are shallow vs deep.
#[doc(hidden)]
#[must_use]
pub fn mode_b_k_split(h_body: u32, n_levels: u32) -> u32 {
    let mut k_split = 0;
    while k_split < n_levels && (h_body >> k_split) >= MODE_B_DEEP_THRESHOLD {
        k_split += 1;
    }
    k_split.min(n_levels)
}

/// Strip buffer height at level `k` for the Mode B walker
/// (back-projected through the reduce chain).
///
/// `downscale_strip_kernel` reads `±2` source rows around `2·dy_logical`,
/// so producing `R_{k+1}` valid level-(k+1) output rows from level-k
/// source requires `2·R_{k+1} + 4` level-k source rows. The level-k
/// buffer must satisfy two constraints simultaneously:
///
/// 1. Its own band loop reads body+halo at level k:
///    `R_k >= body_k + 2·halo_k = (h_body >> k) + 16`.
/// 2. It must feed the level-(k+1) reduce:
///    `R_k >= 2·R_{k+1} + 4` for `k < k_split - 1`.
///
/// The recursion runs deepest-shallow → shallowest. At `h_body = 512,
/// k_split = 6`:
///
/// | k | body_k | R_k                              |
/// |---|--------|----------------------------------|
/// | 5 | 16     | 32                               |
/// | 4 | 32     | max(48, 2·32+4) = 68             |
/// | 3 | 64     | max(80, 2·68+4) = 140            |
/// | 2 | 128    | max(144, 2·140+4) = 284          |
/// | 1 | 256    | max(272, 2·284+4) = 572          |
/// | 0 | 512    | max(528, 2·572+4) = **1148**     |
///
/// Compared to the band-resolution-only model (which would give 528 at
/// level 0), back-projection roughly doubles the level-0 buffer at
/// h_body=512.
///
/// Returns 0 for `k >= k_split` since deep levels use full-image
/// storage (caller substitutes the level-k full image dim).
///
/// Bit-identical to `cvvdp_gpu::pipeline::mode_b_strip_h_at_level`.
#[doc(hidden)]
#[must_use]
pub fn mode_b_strip_h_at_level(k: u32, h_body: u32, k_split: u32) -> u32 {
    if k >= k_split {
        return 0; // caller uses full-image dim instead
    }
    // Iterate deepest-shallow → up. Track R_{k+1} as we walk down to k.
    // Deepest shallow level (k_split - 1): only the body+halo constraint
    // applies (no further reduce to feed).
    let halo_band = 8_u32;
    let deepest = k_split - 1;
    let mut r_deeper: u32 = (h_body >> deepest).saturating_add(2 * halo_band);
    if k == deepest {
        return r_deeper;
    }
    // Walk from k_split - 2 down to k.
    for ki in (k..deepest).rev() {
        let body_ki = h_body >> ki;
        let own = body_ki.saturating_add(2 * halo_band);
        let from_reduce = r_deeper.saturating_mul(2).saturating_add(4);
        r_deeper = own.max(from_reduce);
    }
    r_deeper
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical table from `mode_b_strip_h_at_level`'s doc comment.
    /// Pinned bit-identical to the GPU helper at `h_body = 512,
    /// n_levels = 9`. If this drifts, our K_SPLIT walker disagrees
    /// with the GPU reference's per-level sizing, and the parity
    /// invariant breaks.
    #[test]
    fn k_split_table_matches_gpu_doc() {
        let h_body = 512_u32;
        let n_levels = 9_u32;
        let k_split = mode_b_k_split(h_body, n_levels);
        // body_5 = 16 ≥ 12, body_6 = 8 < 12. So k_split == 6.
        assert_eq!(k_split, 6, "k_split must be 6 at h_body=512, n_levels=9");

        // R_5 = 32, R_4 = 68, R_3 = 140, R_2 = 284, R_1 = 572, R_0 = 1148.
        assert_eq!(mode_b_strip_h_at_level(5, h_body, k_split), 32);
        assert_eq!(mode_b_strip_h_at_level(4, h_body, k_split), 68);
        assert_eq!(mode_b_strip_h_at_level(3, h_body, k_split), 140);
        assert_eq!(mode_b_strip_h_at_level(2, h_body, k_split), 284);
        assert_eq!(mode_b_strip_h_at_level(1, h_body, k_split), 572);
        assert_eq!(mode_b_strip_h_at_level(0, h_body, k_split), 1148);

        // Deep levels return 0 (caller substitutes full-image dim).
        assert_eq!(mode_b_strip_h_at_level(6, h_body, k_split), 0);
        assert_eq!(mode_b_strip_h_at_level(7, h_body, k_split), 0);
        assert_eq!(mode_b_strip_h_at_level(8, h_body, k_split), 0);
    }

    #[test]
    fn k_split_smaller_h_body() {
        // h_body = 128: body_0 = 128, body_1 = 64, body_2 = 32,
        // body_3 = 16, body_4 = 8 < 12. So k_split == 4.
        let k_split = mode_b_k_split(128, 9);
        assert_eq!(k_split, 4);

        // Build R_k recursively for k_split=4:
        //   R_3 = body_3 + 2·halo = 16 + 16 = 32 (deepest shallow level)
        //   R_2 = max(body_2 + 16 = 32 + 16 = 48, 2·32 + 4 = 68) = 68
        //   R_1 = max(body_1 + 16 = 64 + 16 = 80, 2·68 + 4 = 140) = 140
        //   R_0 = max(body_0 + 16 = 128 + 16 = 144, 2·140 + 4 = 284) = 284
        assert_eq!(mode_b_strip_h_at_level(3, 128, k_split), 32);
        assert_eq!(mode_b_strip_h_at_level(2, 128, k_split), 68);
        assert_eq!(mode_b_strip_h_at_level(1, 128, k_split), 140);
        assert_eq!(mode_b_strip_h_at_level(0, 128, k_split), 284);
    }

    #[test]
    fn k_split_caps_at_n_levels() {
        // h_body so large that every level passes the threshold —
        // k_split caps at n_levels (no deep levels at all).
        let k_split = mode_b_k_split(8192, 9);
        // At level 8, body = 8192 / 256 = 32 ≥ 12 — still shallow.
        // Caps at n_levels = 9.
        assert_eq!(k_split, 9);
    }

    #[test]
    fn k_split_tiny_h_body() {
        // h_body = 8 < 12 — first level (k=0) already fails the
        // threshold, so k_split = 0 (no shallow levels at all).
        let k_split = mode_b_k_split(8, 9);
        assert_eq!(k_split, 0);
    }

    #[test]
    fn halo_at_level_returns_8_when_shallow() {
        let k_split = 6;
        for k in 0..k_split {
            assert_eq!(
                mode_b_halo_at_level(k, k_split),
                8,
                "halo for shallow level k={k} must be 8 (PU radius 6 + downscale slack 2)"
            );
        }
    }

    #[test]
    fn halo_at_level_returns_0_when_deep() {
        let k_split = 6;
        for k in k_split..9 {
            assert_eq!(
                mode_b_halo_at_level(k, k_split),
                0,
                "halo for deep level k={k} must be 0 (deep levels use full-image storage)"
            );
        }
    }

    /// Cross-check against the GPU helper's recurrence directly: the
    /// CPU port and GPU helper must produce bit-identical outputs for
    /// every (k, h_body, k_split) triple in a realistic range.
    /// Spot-checked at several `h_body` values used in practice.
    #[test]
    fn k_split_matches_recurrence_general() {
        for &h_body in &[16_u32, 32, 64, 128, 256, 512, 1024] {
            let n_levels = 9_u32;
            let k_split = mode_b_k_split(h_body, n_levels);
            for k in 0..k_split {
                let expected = brute_force_recurrence(k, h_body, k_split);
                let actual = mode_b_strip_h_at_level(k, h_body, k_split);
                assert_eq!(
                    actual, expected,
                    "mismatch at h_body={h_body}, k={k}, k_split={k_split}"
                );
            }
        }
    }

    /// Independent recurrence implementation — same math, written
    /// from-scratch to catch transcription bugs in the production
    /// helper. If they ever diverge that's a port bug.
    fn brute_force_recurrence(k: u32, h_body: u32, k_split: u32) -> u32 {
        if k >= k_split {
            return 0;
        }
        let halo_band = 8_u32;
        let deepest = k_split - 1;
        let mut r = h_body.checked_shr(deepest).unwrap_or(0) + 2 * halo_band;
        let mut ki = deepest;
        while ki > k {
            ki -= 1;
            let body_ki = h_body.checked_shr(ki).unwrap_or(0);
            let own = body_ki + 2 * halo_band;
            let from_reduce = 2 * r + 4;
            r = own.max(from_reduce);
        }
        r
    }
}
