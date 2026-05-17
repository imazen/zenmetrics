//! Skip-map dispatch for SSIMULACRA2 (Technique 2 of Kanetaka et al.
//! IWAIT 2026 / SPIE 14072-G).
//!
//! The 108-element WEIGHT vector in `pipeline::score_from_stats` has a
//! sparse distribution — about half its entries are exactly zero and
//! many of the rest are small enough that their contribution to the
//! final score is bounded under any plausible input. This module
//! tabulates the per-cell skip status under four user-selectable
//! modes and exposes the precomputed launch-skip masks the pipeline
//! consults at dispatch time.
//!
//! See `crates/ssim2-gpu/docs/SKIP_MAP_AUDIT.md` for the per-cell
//! audit and the rationale behind the thresholds.

use crate::NUM_SCALES;

/// Weight magnitude below which the cell is considered "small" under
/// the `Fast` mode.
pub const FAST_THRESHOLD: f64 = 1.0e-3;
/// Weight magnitude below which the cell is considered "small" under
/// the `Faster` mode.
pub const FASTER_THRESHOLD: f64 = 1.0e-2;

/// Skip-map mode selector. Default is `Faster`, matching the IWAIT
/// 2026 finding that all four modes hit identical SROCC on real
/// corpora (CID22) — there is no accuracy reason to pick anything
/// less aggressive.
///
/// - **Full**: no skipping. Bit-identical to the pre-skip-map output.
/// - **Lossless**: skip cells whose weight is literally `0.0`. The
///   output is bit-identical to Full because `WEIGHT[i].mul_add(v, s)`
///   with `WEIGHT[i] == 0` reduces to `s` exactly regardless of `v`.
/// - **Fast**: additionally skip cells with `|weight| < 1e-3`.
/// - **Faster**: additionally skip cells with `|weight| < 1e-2`.
///   Default. Paper §4: zero accuracy cost vs Full on CID22.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ssim2Mode {
    Full,
    Lossless,
    Fast,
    #[default]
    Faster,
}

impl Ssim2Mode {
    /// Threshold below which `|weight|` causes the cell to be skipped
    /// under this mode. `Full` returns a negative number so no cell is
    /// ever skipped (every `|w| >= 0` test fails as a skip).
    #[inline]
    pub fn threshold(self) -> f64 {
        match self {
            // -1.0 is below 0.0, so `|w| < -1.0` is always false → never skip.
            Ssim2Mode::Full => -1.0,
            // 0.0: only literal-zero weights satisfy `|w| <= 0.0` → exact skip.
            // We use the `<=` form for Lossless and `<` for Fast/Faster.
            Ssim2Mode::Lossless => 0.0,
            Ssim2Mode::Fast => FAST_THRESHOLD,
            Ssim2Mode::Faster => FASTER_THRESHOLD,
        }
    }
}

/// The 108-element SSIMULACRA2 score weights. Verbatim copy of the
/// `WEIGHT` array in `pipeline::score_from_stats`. Maintained as the
/// single source of truth — `pipeline.rs` re-exports this name in its
/// scoring function.
///
/// Index ordering (matches `score_from_stats`'s nested loop):
/// `i = ((c * NUM_SCALES + scale) * 2 + n) * 3 + map`, where
/// `c ∈ {0=X, 1=Y, 2=B}`, `scale ∈ 0..6`, `n ∈ {0=L1, 1=L4}`,
/// `map ∈ {0=DSSIM, 1=artifact, 2=detailloss}`.
pub const WEIGHTS: [f64; 108] = [
    0.0,
    0.000_737_660_670_740_658_6,
    0.0,
    0.0,
    0.000_779_348_168_286_730_9,
    0.0,
    0.0,
    0.000_437_115_573_010_737_9,
    0.0,
    1.104_172_642_665_734_6,
    0.000_662_848_341_292_71,
    0.000_152_316_327_837_187_52,
    0.0,
    0.001_640_643_745_659_975_4,
    0.0,
    1.842_245_552_053_929_8,
    11.441_172_603_757_666,
    0.0,
    0.000_798_910_943_601_516_3,
    0.000_176_816_438_078_653,
    0.0,
    1.878_759_497_954_638_7,
    10.949_069_906_051_42,
    0.0,
    0.000_728_934_699_150_807_2,
    0.967_793_708_062_683_3,
    0.0,
    0.000_140_034_242_854_358_84,
    0.998_176_697_785_496_7,
    0.000_319_497_559_344_350_53,
    0.000_455_099_211_379_206_3,
    0.0,
    0.0,
    0.001_364_876_616_324_339_8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.466_890_328_078_848,
    0.0,
    17.445_833_984_131_262,
    0.000_623_560_163_404_146_6,
    0.0,
    0.0,
    6.683_678_146_179_332,
    0.000_377_244_079_796_112_96,
    1.027_889_937_768_264,
    225.205_153_008_492_74,
    0.0,
    0.0,
    19.213_238_186_143_016,
    0.001_140_152_458_661_836_1,
    0.001_237_755_635_509_985,
    176.393_175_984_506_94,
    0.0,
    0.0,
    24.433_009_998_704_76,
    0.285_208_026_121_177_57,
    0.000_448_543_692_383_340_8,
    0.0,
    0.0,
    0.0,
    34.779_063_444_837_72,
    44.835_625_328_877_896,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_868_055_657_329_169_8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_531_319_187_435_874_7,
    0.0,
    0.000_165_338_141_613_791_12,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.000_417_917_180_325_133_6,
    0.001_729_082_823_472_283_3,
    0.0,
    0.002_082_700_584_663_643_7,
    0.0,
    0.0,
    8.826_982_764_996_862,
    23.192_433_439_989_26,
    0.0,
    95.108_049_881_108_6,
    0.986_397_803_440_068_2,
    0.983_438_279_246_535_3,
    0.001_228_640_504_827_849_3,
    171.266_725_589_730_7,
    0.980_785_887_243_537_9,
    0.0,
    0.0,
    0.0,
    0.000_513_006_458_899_067_9,
    0.0,
    0.000_108_540_578_584_115_37,
];

/// Compute the linear weight index for the cell
/// `(channel, scale, norm, map_type)` per the indexing convention in
/// `score_from_stats`. `channel ∈ 0..3`, `scale ∈ 0..NUM_SCALES`,
/// `norm ∈ 0..2` (0 = L1, 1 = L4), `map_type ∈ 0..3` (0 = DSSIM,
/// 1 = artifact, 2 = detailloss).
#[inline]
pub const fn weight_index(channel: usize, scale: usize, norm: usize, map_type: usize) -> usize {
    ((channel * NUM_SCALES + scale) * 2 + norm) * 3 + map_type
}

/// Returns `true` if both L1 and L4 weights for the given
/// `(scale, channel, map_type)` triple are below the mode's threshold
/// — i.e., the corresponding `launch_sum_p4` (reduction) can be
/// skipped entirely without changing the score by more than `2 *
/// threshold` worst-case.
///
/// This is the **launch-level** skip predicate used by `Ssim2::compute`
/// and `Ssim2::compute_with_reference`.
#[inline]
pub fn skip_reduction(mode: Ssim2Mode, scale: usize, channel: usize, map_type: usize) -> bool {
    if matches!(mode, Ssim2Mode::Full) {
        return false;
    }
    let w_l1 = WEIGHTS[weight_index(channel, scale, 0, map_type)].abs();
    let w_l4 = WEIGHTS[weight_index(channel, scale, 1, map_type)].abs();
    let max_w = w_l1.max(w_l4);
    if matches!(mode, Ssim2Mode::Lossless) {
        // Strict equality for Lossless — bit-identical.
        max_w == 0.0
    } else {
        max_w < mode.threshold()
    }
}

/// Returns `true` if all three error maps (DSSIM, artifact,
/// detailloss) for `(scale, channel)` would be reduction-skipped —
/// in which case the upstream `error_maps_kernel` for that channel
/// at that scale is also unneeded.
#[inline]
pub fn skip_error_map(mode: Ssim2Mode, scale: usize, channel: usize) -> bool {
    (0..3).all(|m| skip_reduction(mode, scale, channel, m))
}

/// Returns `true` if every cell at this scale (all 3 channels × all
/// 3 maps × both norms) is skip-eligible — in which case the entire
/// scale (XYB, products, blurs, transposes, error maps, reductions)
/// can be elided. Currently true only for scale 5 under Faster mode.
#[inline]
pub fn skip_scale(mode: Ssim2Mode, scale: usize) -> bool {
    (0..3).all(|c| skip_error_map(mode, scale, c))
}

/// Number of `launch_sum_p4` calls saved per `compute()` at this mode.
/// Used by tests and bench reports.
pub fn count_skipped_reductions(mode: Ssim2Mode, n_scales: usize) -> usize {
    let mut n = 0;
    for s in 0..n_scales {
        for c in 0..3 {
            for m in 0..3 {
                if skip_reduction(mode, s, c, m) {
                    n += 1;
                }
            }
        }
    }
    n
}

/// Number of `error_maps_kernel` launches saved per `compute()` at
/// this mode.
pub fn count_skipped_error_maps(mode: Ssim2Mode, n_scales: usize) -> usize {
    let mut n = 0;
    for s in 0..n_scales {
        for c in 0..3 {
            if skip_error_map(mode, s, c) {
                n += 1;
            }
        }
    }
    n
}

/// Number of scales fully skippable at this mode.
pub fn count_skipped_scales(mode: Ssim2Mode, n_scales: usize) -> usize {
    (0..n_scales).filter(|&s| skip_scale(mode, s)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The skip-map must NEVER skip a cell with a nonzero weight under
    /// Lossless mode — that's the entire correctness invariant.
    #[test]
    fn lossless_only_skips_literal_zero() {
        for c in 0..3 {
            for s in 0..NUM_SCALES {
                for m in 0..3 {
                    if skip_reduction(Ssim2Mode::Lossless, s, c, m) {
                        let w_l1 = WEIGHTS[weight_index(c, s, 0, m)];
                        let w_l4 = WEIGHTS[weight_index(c, s, 1, m)];
                        assert_eq!(
                            w_l1, 0.0,
                            "Lossless skipped (s={s},c={c},m={m}) with L1 weight {w_l1}"
                        );
                        assert_eq!(
                            w_l4, 0.0,
                            "Lossless skipped (s={s},c={c},m={m}) with L4 weight {w_l4}"
                        );
                    }
                }
            }
        }
    }

    /// Audit-table cross-check. If these numbers ever change, the
    /// audit doc + bench numbers in the commit message are stale.
    #[test]
    fn skip_counts_match_audit_doc() {
        assert_eq!(count_skipped_reductions(Ssim2Mode::Full, NUM_SCALES), 0);
        assert_eq!(count_skipped_reductions(Ssim2Mode::Lossless, NUM_SCALES), 17);
        assert_eq!(count_skipped_reductions(Ssim2Mode::Fast, NUM_SCALES), 30);
        assert_eq!(count_skipped_reductions(Ssim2Mode::Faster, NUM_SCALES), 34);

        assert_eq!(count_skipped_error_maps(Ssim2Mode::Full, NUM_SCALES), 0);
        assert_eq!(count_skipped_error_maps(Ssim2Mode::Lossless, NUM_SCALES), 1);
        assert_eq!(count_skipped_error_maps(Ssim2Mode::Fast, NUM_SCALES), 5);
        assert_eq!(count_skipped_error_maps(Ssim2Mode::Faster, NUM_SCALES), 7);

        assert_eq!(count_skipped_scales(Ssim2Mode::Full, NUM_SCALES), 0);
        assert_eq!(count_skipped_scales(Ssim2Mode::Lossless, NUM_SCALES), 0);
        assert_eq!(count_skipped_scales(Ssim2Mode::Fast, NUM_SCALES), 0);
        assert_eq!(count_skipped_scales(Ssim2Mode::Faster, NUM_SCALES), 1);
    }

    /// Cell-level skip counts (108 cells total).
    #[test]
    fn cell_level_skip_counts() {
        let mut cnt_lossless = 0;
        let mut cnt_fast = 0;
        let mut cnt_faster = 0;
        for &w in WEIGHTS.iter() {
            let aw = w.abs();
            if aw == 0.0 { cnt_lossless += 1; }
            if aw < FAST_THRESHOLD { cnt_fast += 1; }
            if aw < FASTER_THRESHOLD { cnt_faster += 1; }
        }
        assert_eq!(cnt_lossless, 56, "Lossless cell-skip count");
        assert_eq!(cnt_fast, 76, "Fast cell-skip count (cumulative incl. lossless)");
        assert_eq!(cnt_faster, 83, "Faster cell-skip count (cumulative incl. fast + lossless)");
    }
}
