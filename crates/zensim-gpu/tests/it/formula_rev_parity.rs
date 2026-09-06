//! **The revision-2 port's own gates** — `PLAN_REV2_WAVE_2026-09-06.md` §1.
//!
//! `zensim-gpu` is the fourth hand-copy of two per-pixel expressions revision 2
//! changes (`zensim/benchmarks/f4_arm_decision_2026-09-05.md` §10). These tests
//! hold the copy to the same two arms the CPU owner decided by measurement:
//!
//! * **F4 / `v1ssimcap`** — `SsimLumaForm::Clamp`, `num_m = max(0, 1 - D^2)`,
//!   in the four v1 SSIM kernels plus the diffmap kernel.
//! * **F17 / `v1hfgain`** — `HfGainForm::SaturatingExcess`, `g/(g+1)`, in the
//!   host-side finalize.
//!
//! # What each gate is, and what it is NOT
//!
//! **G-GPU.1 (revision 1 is untouched)** is measured by
//! `examples/formula_rev_dump.rs`, not here: byte-identity against the
//! PRE-PORT binary needs two builds, which a single test binary cannot do.
//! What lives here is the in-binary half —
//! [`pinning_the_active_revision_is_a_no_op`] — plus the whole existing
//! parity suite, which runs at revision 1 by default and must not move.
//!
//! **G-GPU.2 (revision 2 agrees with the CPU rev2 walk)** is the existing
//! CPU↔GPU suite run in a process with `ZENSIM_FORMULA_REV=2`: CPU `zensim` and
//! this crate then read the SAME variable with the same rules, so
//! `cpu_parity`, `extended_parity` and `cpu_gpu_feature_sweep` become a rev2
//! cross-check with no new tolerance invented anywhere. The tests here supply
//! the half that suite cannot: proof that the ported branches are **live** —
//! that revision 2 actually changes the output, and changes it into exactly
//! the arm the CPU owner picked.
//!
//! **G-GPU.3 (refuse what cannot be served)** is
//! [`a_diffmap_path_refuses_a_revision_it_cannot_serve`].
//!
//! # The anti-vacuity problem, and how it is solved here
//!
//! MEASURED (`examples/formula_rev_dump.rs` over 29,700 slot values): on 8-bit
//! sRGB fixtures, revision 2 moves **only** the twelve `hf_energy_gain` slots.
//! F4's `Clamp` moves **zero** cells there — which reproduces the CPU lane's
//! finding over 217,756 real rows exactly, and which means an SDR-only test of
//! the F4 port would pass whether the branch worked or was dead code.
//!
//! The PU-XYB (HDR) route is what makes it live: PU21 encodes absolute
//! luminance into ~[0, 600] before normalising by `PU_WHITE = 256.3`, so a
//! near-black reference against a 10,000 cd/m² distorted image drives
//! `|mu1 - mu2|` past 1 and the clamp fires. Both regimes are asserted below —
//! the SDR identity as a POSITIVE claim (it is the arm's defining property,
//! not an absence of evidence), and the HDR difference as the liveness proof.
//!
//! # Coverage note, stated rather than implied
//!
//! Four v1 kernels carry the F4 copy. Three of them —
//! `fused_features_kernel`, `fused_features_kernel_persist` and
//! `masked_iw_strip_kernel` — are exercised here (the PU-HDR case moves basic,
//! peak, masked and IW slots). The fourth, `masked_iw::masked_iw_kernel`, has
//! **no launch site anywhere in the crate**; it is ported for consistency and
//! is compile-checked only. `diffmap::per_scale_weighted_ssim_kernel` is
//! exercised by `cpu_gpu_diffmap_parity`, which sets `ZENSIM_GPU_DIFFMAP=1`.

use cubecl::Runtime;
use zensim_gpu::{FormulaRevision, Zensim, ZensimFeatureRegime};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "zensim-gpu formula_rev_parity test requires either the `cuda` or `wgpu` feature to \
     select a runtime"
);

macro_rules! make_client {
    () => {
        Backend::client(&Default::default())
    };
}

/// Basic block layout: 13 features per (scale, channel).
const BASIC_PER_CH: usize = 13;
/// `hf_energy_gain` — F17's slot, block-local index 12.
const HF_ENERGY_GAIN: usize = 12;
/// End of the basic block at 4 scales × 3 channels.
const BASIC_END: usize = 4 * 3 * BASIC_PER_CH;

fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            v.push(((x * 255) / w) as u8);
            v.push(((y * 255) / h) as u8);
            v.push((((x + y) * 255) / (w + h)) as u8);
        }
    }
    v
}

fn add_noise(data: &[u8], amount: i16) -> Vec<u8> {
    use std::num::Wrapping;
    let mut seed = Wrapping(12345_u32);
    data.iter()
        .map(|&v| {
            seed = seed * Wrapping(1103515245_u32) + Wrapping(12345_u32);
            let n = ((seed.0 >> 16) as i16 % (amount * 2 + 1)) - amount;
            (v as i16 + n).clamp(0, 255) as u8
        })
        .collect()
}

/// A flat-ish reference against a heavily noised copy — `var_dst >> var_src`,
/// i.e. F17's regime. The ramp keeps `var_src` above the `1e-10` gate, which a
/// perfectly flat source would not.
fn f17_pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let mut r = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let v = 120u8 + (((x + y) % 3) as u8);
            r.extend_from_slice(&[v, v, v]);
        }
    }
    let d = add_noise(&r, 60);
    (r, d)
}

fn features_at(
    rev: FormulaRevision,
    regime: ZensimFeatureRegime,
    w: usize,
    h: usize,
    r: &[u8],
    d: &[u8],
) -> Vec<f64> {
    let mut z = Zensim::<Backend>::new_with_regime(make_client!(), w as u32, h as u32, regime)
        .expect("construct")
        .with_formula_revision(rev);
    assert_eq!(z.formula_revision(), rev, "the override did not take");
    z.compute_features_vec(r, d).expect("compute")
}

/// PU-linear-nits planes: a near-black reference against a near-peak
/// distorted image, spatially patterned so the blur windows straddle the
/// swing. This is the only fixture family on this box that reaches F4.
fn pu_planes(w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let n = w * h;
    let mut lo = vec![0.005_f32; n];
    let mut hi = vec![0.005_f32; n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let alt = (x / 16 + y / 16) % 2 == 0;
            lo[i] = if alt { 0.005 } else { 0.02 };
            hi[i] = if alt { 10000.0 } else { 4000.0 };
        }
    }
    (lo, hi)
}

fn pu_features_at(
    rev: FormulaRevision,
    regime: ZensimFeatureRegime,
    w: usize,
    h: usize,
) -> Vec<f64> {
    let (lo, hi) = pu_planes(w, h);
    let mut z = Zensim::<Backend>::new_with_regime(make_client!(), w as u32, h as u32, regime)
        .expect("construct pu")
        .with_formula_revision(rev);
    z.compute_features_pu_linear_nits([&lo, &lo, &lo], [&hi, &hi, &hi])
}

// ───────────────────────── G-GPU.1, in-binary half ─────────────────────────

/// Pinning the ACTIVE revision must be indistinguishable from not pinning
/// anything — the mirror of
/// `zensim::ssim_form::tests::selecting_the_shipped_revision_is_a_no_op`.
///
/// Phrased against `active_revision()` rather than hard-coded to `Rev1` so the
/// SAME test is meaningful in a `ZENSIM_FORMULA_REV=2` process — which is how
/// G-GPU.2 runs this suite. Hard-coding `Rev1` would have made the rev2 run
/// report a failure that is really "the test assumed the default".
///
/// `to_bits()` equality, not a tolerance: the revision switch is not allowed
/// to spend a single bit of the CPU↔GPU parity budget the rest of this suite
/// lives inside.
///
/// This is the half of G-GPU.1 that fits in one binary. The other half —
/// byte-identity against the PRE-PORT build — is `examples/formula_rev_dump.rs`.
#[test]
fn pinning_the_active_revision_is_a_no_op() {
    let active = zensim_gpu::formula_rev::active_revision();
    eprintln!("formula_rev_parity: active revision = {}", active.as_str());
    let (w, h) = (64usize, 64usize);
    let r = gradient(w, h);
    let d = add_noise(&r, 8);
    for regime in [
        ZensimFeatureRegime::Basic,
        ZensimFeatureRegime::Extended,
        ZensimFeatureRegime::WithIw,
    ] {
        let mut z0 = Zensim::<Backend>::new_with_regime(make_client!(), w as u32, h as u32, regime)
            .expect("construct default");
        let unpinned = z0.compute_features_vec(&r, &d).expect("compute default");
        let pinned = features_at(active, regime, w, h, &r, &d);
        assert_eq!(unpinned.len(), pinned.len(), "{regime:?}: width");
        for (i, (a, b)) in unpinned.iter().zip(pinned.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{regime:?} f{i}: pinning revision {} moved the output ({a:.17e} vs {b:.17e})",
                active.as_str()
            );
        }
    }
}

// ───────────────────────── F17, the closed form ─────────────────────────

/// **F17's port, checked against its own closed form.**
///
/// Revision 2's arm is `g -> g/(g+1)`, an EXACT function of the revision-1
/// value — so this needs no second transcription of the arm and no tolerance:
/// run the same pixels at both revisions and assert the relation slot by slot,
/// in `f64`, on the twelve `hf_energy_gain` slots.
///
/// The `saw_positive` assertion is the anti-vacuity control. Without it the
/// test passes on an all-zero fixture (`0/(0+1) == 0`) and proves nothing —
/// which is exactly what a `var_src`-gated slot does on flat content.
///
/// Everything OUTSIDE those twelve slots must be bit-identical: F17 is a basic
/// block-local slot and revision 2 must not leak into its neighbours. That
/// second half is what would catch a `luma_clamp` accidentally wired to the
/// gain, or a gain arm applied to `hf_energy_loss`.
#[test]
fn f17_rev2_is_the_exact_saturating_map_of_rev1() {
    for &(w, h) in &[(64usize, 64usize), (128, 96)] {
        let (r, d) = f17_pair(w, h);
        for regime in [
            ZensimFeatureRegime::Basic,
            ZensimFeatureRegime::Extended,
            ZensimFeatureRegime::WithIw,
        ] {
            let v1 = features_at(FormulaRevision::Rev1, regime, w, h, &r, &d);
            let v2 = features_at(FormulaRevision::Rev2, regime, w, h, &r, &d);
            assert_eq!(v1.len(), v2.len());

            let mut saw_positive = false;
            for i in 0..v1.len() {
                let is_gain = i < BASIC_END && i % BASIC_PER_CH == HF_ENERGY_GAIN;
                if is_gain {
                    let g = v1[i];
                    assert!(
                        g >= 0.0,
                        "{w}x{h} {regime:?} f{i}: rev1 gain went negative: {g}"
                    );
                    if g > 0.0 {
                        saw_positive = true;
                    }
                    assert_eq!(
                        v2[i].to_bits(),
                        (g / (g + 1.0)).to_bits(),
                        "{w}x{h} {regime:?} f{i}: rev2 is not g/(g+1) of rev1 \
                         (rev1={g:.17e} rev2={:.17e})",
                        v2[i]
                    );
                    assert!(
                        (0.0..1.0).contains(&v2[i]),
                        "{w}x{h} {regime:?} f{i}: rev2 gain {} escaped [0, 1)",
                        v2[i]
                    );
                } else {
                    assert_eq!(
                        v1[i].to_bits(),
                        v2[i].to_bits(),
                        "{w}x{h} {regime:?} f{i}: revision 2 moved a slot that is not \
                         hf_energy_gain on SDR content ({:.17e} vs {:.17e}) — F4's Clamp arm \
                         is bit-identical to revision 1 wherever (mu1-mu2)^2 <= 1, which is \
                         everywhere on 8-bit sRGB",
                        v1[i],
                        v2[i]
                    );
                }
            }
            assert!(
                saw_positive,
                "{w}x{h} {regime:?}: no hf_energy_gain slot was positive — the fixture never \
                 entered F17's regime, so this cell proved nothing"
            );
        }
    }
}

// ───────────────────────── F4, liveness ─────────────────────────

/// **F4's ported branch is LIVE, and only where the arm says it should be.**
///
/// Two halves, and the first is the reason the second is necessary:
///
/// 1. On 8-bit sRGB the two revisions differ on NOTHING outside
///    `hf_energy_gain` (asserted in the F17 test above). `Clamp` is defined to
///    differ from revision 1 only where `(mu1 - mu2)^2 > 1`, and SDR XYB never
///    gets there. So an SDR-only test cannot tell a working clamp from dead
///    code.
/// 2. On the PU-XYB (HDR) route it MUST differ, and on SSIM-derived slots
///    specifically — `ssim_mean/4th/2nd` in the basic block, the peak block,
///    and (at the wider regimes) the masked and IW blocks, which is how the
///    persist kernel and the strip masked-IW kernel get covered too.
///
/// Rather than pin a magnitude (an f32 property of whichever rasteriser is
/// running), the HDR half asserts the arm's DEFINING property: with `num_m`
/// bounded in `[0, 1]` and `num_s/denom_s` in `[-1, 1]` by Cauchy-Schwarz,
/// the per-pixel `d` lands in `[0, 2]`, so every pooled SSIM slot whose
/// weight is itself bounded by 1 must be `<= 2`. Revision 1 has no such
/// bound, and this fixture proves it: MEASURED here, `ssim_max` reaches
/// **5.4275** at revision 1 and exactly **1.0** at revision 2, with basic
/// `ssim_4th` 3.3698 -> 1.0 and the masked block 3.3660 -> 0.9976.
///
/// The IW block is EXCLUDED from the `<= 2` half and only from that half:
/// its weight is `1 + k*activity`, which is unbounded above, so a bounded
/// `d` does not imply a bounded IW pool. It still participates in the
/// liveness count.
///
/// A direction assertion was tried first and is wrong in both directions:
/// clamping a negative `num_m` up to 0 LOWERS `d` when `num_s/denom_s > 0`
/// (the dominant case, and the one that produces F4's blow-up) and RAISES it
/// when the structure ratio is negative. Recorded because it cost a failing
/// run to learn.
#[test]
fn f4_rev2_branch_is_live_on_pu_hdr_and_inert_on_sdr() {
    let (w, h) = (128usize, 96usize);

    // Half 1: SDR — F4 changes nothing, so any difference here is a bug.
    let r = gradient(w, h);
    let d = add_noise(&r, 8);
    let s1 = features_at(
        FormulaRevision::Rev1,
        ZensimFeatureRegime::WithIw,
        w,
        h,
        &r,
        &d,
    );
    let s2 = features_at(
        FormulaRevision::Rev2,
        ZensimFeatureRegime::WithIw,
        w,
        h,
        &r,
        &d,
    );
    for i in 0..s1.len() {
        if i < BASIC_END && i % BASIC_PER_CH == HF_ENERGY_GAIN {
            continue; // F17's slots; covered by the test above.
        }
        assert_eq!(
            s1[i].to_bits(),
            s2[i].to_bits(),
            "SDR f{i} moved at revision 2 — Clamp must be bit-identical below the knee"
        );
    }

    // Half 2: PU-HDR — F4 must fire.
    for regime in [
        ZensimFeatureRegime::Basic,
        ZensimFeatureRegime::Extended,
        ZensimFeatureRegime::WithIw,
    ] {
        let p1 = pu_features_at(FormulaRevision::Rev1, regime, w, h);
        let p2 = pu_features_at(FormulaRevision::Rev2, regime, w, h);
        assert_eq!(p1.len(), p2.len());
        let mut moved_non_gain = 0usize;
        let (mut bounded1, mut bounded2) = (0.0_f64, 0.0_f64);
        let (mut worst_i, mut worst_v) = (usize::MAX, 0.0_f64);
        for i in 0..p1.len() {
            let is_gain = i < BASIC_END && i % BASIC_PER_CH == HF_ENERGY_GAIN;
            if p1[i].to_bits() != p2[i].to_bits() && !is_gain {
                moved_non_gain += 1;
            }
            if !is_bounded_ssim_slot(i) {
                continue;
            }
            bounded1 = bounded1.max(p1[i]);
            if p2[i] > worst_v {
                worst_v = p2[i];
                worst_i = i;
            }
            bounded2 = bounded2.max(p2[i]);
        }
        eprintln!(
            "PU {regime:?}: moved {moved_non_gain} non-gain slots; bounded-SSIM max \
             rev1={bounded1:.6} rev2={bounded2:.6}"
        );
        assert!(
            moved_non_gain > 0,
            "PU {regime:?}: revision 2 moved no SSIM-derived slot — F4's ported branch is \
             DEAD on this fixture, so the F4 half of the port is unmeasured"
        );
        // Anti-vacuity for the bound: revision 1 must actually exceed it here,
        // or `<= 2` at revision 2 would be a bound nothing was straining
        // against.
        assert!(
            bounded1 > 2.0,
            "PU {regime:?}: revision 1's bounded-SSIM max is only {bounded1:.6} — this fixture \
             does not reproduce F4's unboundedness, so the bound below proves nothing"
        );
        assert!(
            bounded2 <= 2.0 + 1e-6,
            "PU {regime:?}: revision 2 left f{worst_i} at {worst_v:.6} > 2 — Clamp puts the \
             per-pixel d in [0, 2], so a pool with weight <= 1 cannot exceed 2"
        );
    }
}

/// Slots whose value is a pool of the per-pixel `d` under a weight bounded by
/// 1, and which therefore inherit `Clamp`'s `d <= 2` bound directly:
/// the basic block's `ssim_mean/4th/2nd`, the peak block's `ssim_max` and
/// `ssim_l8`, and the masked block's `masked_ssim_mean/4th/2nd` (mask =
/// `1/(1 + k*a)`, in `(0, 1]`).
///
/// The IW block is excluded BY CONSTRUCTION, not by convenience: its weight is
/// `1 + k*a`, unbounded above, so a bounded `d` implies no bound on the pool.
/// The `art_*` / `det_*` / `mse` / `hf_*` slots are not pools of `d` at all.
fn is_bounded_ssim_slot(i: usize) -> bool {
    const PEAKS_END: usize = BASIC_END + 4 * 3 * 6;
    const MASKED_END: usize = PEAKS_END + 4 * 3 * 6;
    if i < BASIC_END {
        i % BASIC_PER_CH < 3
    } else if i < PEAKS_END {
        matches!((i - BASIC_END) % 6, 0 | 3)
    } else if i < MASKED_END {
        (i - PEAKS_END) % 6 < 3
    } else {
        false
    }
}

// ───────────────────────── G-GPU.3 ─────────────────────────

/// **G-GPU.3** — a path this crate cannot serve at a pinned revision refuses
/// loudly and names itself, instead of quietly mixing arithmetics.
///
/// The diffmap and linear-plane entry points are hybrids: the GPU produces the
/// map, the canonical CPU `zensim` produces the scalar score. CPU `zensim`
/// takes its revision from the process-wide `ZENSIM_FORMULA_REV` and has no
/// per-instance override, so a pipeline pinned to a different revision would
/// return a map and a score computed under two different arithmetics — the
/// exact silent-fallback defect this revision exists to remove.
///
/// The feature-only entry points are deliberately NOT refused: they emit no
/// CPU-derived scalar, so the override is unambiguous there. That asymmetry is
/// asserted too, because a refusal that swallowed the whole API would make the
/// two tests above unrunnable.
#[test]
fn a_diffmap_path_refuses_a_revision_it_cannot_serve() {
    let process = zensim_gpu::formula_rev::active_revision();
    let other = match process {
        FormulaRevision::Rev1 => FormulaRevision::Rev2,
        FormulaRevision::Rev2 => FormulaRevision::Rev1,
    };
    let (w, h) = (64usize, 64usize);
    let r = gradient(w, h);
    let d = add_noise(&r, 8);

    let mut z = Zensim::<Backend>::new(make_client!(), w as u32, h as u32)
        .expect("construct")
        .with_formula_revision(other);
    let mut map = Vec::new();
    let err = z
        .score_with_diffmap(&r, &d, &mut map)
        .expect_err("a mismatched revision on a hybrid path must be refused");
    match err {
        zensim_gpu::Error::FormulaRevisionMismatch {
            pipeline,
            cpu_process,
        } => {
            assert_eq!(pipeline, other);
            assert_eq!(cpu_process, process);
            let msg = format!(
                "{}",
                zensim_gpu::Error::FormulaRevisionMismatch {
                    pipeline,
                    cpu_process
                }
            );
            assert!(
                msg.contains("formula revision") && msg.contains("ZENSIM_FORMULA_REV"),
                "the refusal must name itself and say how to fix it, got: {msg}"
            );
        }
        other_err => panic!("wrong error for a revision mismatch: {other_err}"),
    }

    // Matching revision: the same call is served normally.
    let mut z_ok = Zensim::<Backend>::new(make_client!(), w as u32, h as u32)
        .expect("construct")
        .with_formula_revision(process);
    let mut map_ok = Vec::new();
    z_ok.score_with_diffmap(&r, &d, &mut map_ok)
        .expect("a matching revision must be served");
    assert_eq!(map_ok.len(), w * h);

    // And the feature-only path is NOT refused at the other revision.
    let mut z_feat = Zensim::<Backend>::new(make_client!(), w as u32, h as u32)
        .expect("construct")
        .with_formula_revision(other);
    z_feat
        .compute_features_vec(&r, &d)
        .expect("feature extraction must honour the override, not refuse it");
}
