//! **The CPU correctness oracle for the GPU kernels, with its walk made
//! selectable** (zensim fold-engine lane, 2026-08-30).
//!
//! Every CPU↔GPU parity test in this binary compares a CubeCL kernel's output
//! against the CPU `zensim` crate — `compute_extended_features` for the
//! 228/300/372 feature vectors, `Zensim::compute` for scores. That makes the
//! CPU crate's BUFFERED walk the reference the GPU is validated against, and
//! it is why `benchmarks/extraction_perf_and_buffered_removal_2026-08-30.md`
//! §5 lists "`zensim-gpu`'s only correctness oracle is
//! `compute_extended_features`" as a secondary blocker on retiring that walk:
//! deleting it would delete the kernels' reference.
//!
//! zensim now has a second walk that produces the same `ZensimResult`
//! **bit-for-bit** — the streaming fold, gated by
//! `zensim/tests/fold_engine_parity.rs` on score, `raw_distance`, every
//! feature and `mean_offset` across 18 geometries × {serial, rayon} × rayon
//! pool sizes 1/2/3/8/16. So the oracle's *function* survives buffered's
//! deletion; what remains is to say so here, and to be able to prove it on
//! this side of the repo boundary.
//!
//! [`cpu_oracle`] is that seam. It builds the CPU reference with a
//! **buffered** walk by default — no behaviour change to any existing test —
//! and switches to the fold when `ZENSIM_GPU_ORACLE_ENGINE=fold` is set:
//!
//! ```text
//! ZENSIM_GPU_ORACLE_ENGINE=fold cargo test -p zensim-gpu --test it
//! ```
//!
//! That run is the gate the zensim lane could not execute (this dev box has
//! no GPU): **the whole parity suite must pass unchanged with the fold as the
//! oracle.** Once it has, flipping the default here is a one-line change and
//! the secondary blocker is closed.
//!
//! [`cpu_oracle_engines_agree`] is the half that needs no GPU and therefore
//! runs anywhere: the two oracles must produce bit-identical feature vectors
//! at the sizes, shapes and profiles this suite actually asks them for. If
//! that fails, no GPU comparison is meaningful in either direction.

use zensim::{RgbSlice, Zensim as ZensimCpu, ZensimProfile};

/// Build the CPU reference. Buffered unless `ZENSIM_GPU_ORACLE_ENGINE=fold`.
///
/// A request the fold cannot serve bit-identically (weight-skipping linear
/// profiles, `num_scales != 4`, a live cancellation token) falls back to
/// buffered inside zensim — `zensim::fold_engine::is_fold_backable` owns that
/// predicate — so selecting `fold` can never silently change what a test
/// compares against; it either uses the fold or uses buffered.
pub(crate) fn cpu_oracle(profile: ZensimProfile) -> ZensimCpu {
    let z = ZensimCpu::new(profile);
    match std::env::var("ZENSIM_GPU_ORACLE_ENGINE").as_deref() {
        Ok("fold") => z.with_engine(zensim::fold_engine::ScoringEngine::Fold),
        _ => z,
    }
}

fn to_pix(buf: &[u8]) -> Vec<[u8; 3]> {
    buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Deterministic textured pair, same generator family the parity fixtures use.
fn pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let mut r = Vec::with_capacity(w * h * 3);
    let mut d = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 255) / w.max(1)) as u8;
            let tex = (((x * 7 + y * 13) % 32) * 3) as u8;
            let edge = if (y / 16) % 2 == 0 { 40u8 } else { 0 };
            let px = [
                base.wrapping_add(tex),
                base.wrapping_add(edge),
                (255 - base).wrapping_add(tex / 2),
            ];
            r.extend_from_slice(&px);
            let q = |v: u8| (v / 12) * 12;
            let mut dd = [q(px[0]), q(px[1]), q(px[2])];
            if x < w / 2 && y < h / 2 {
                dd[0] = dd[0].saturating_add(18);
            }
            d.extend_from_slice(&dd);
        }
    }
    (r, d)
}

/// **The GPU-free half of the oracle gate.** Both walks must produce
/// bit-identical CPU references at every shape this suite compares a kernel
/// against: the 372-wide `compute_extended_features` on the shipped profile,
/// the `Zensim::compute` score, and the sub-64 reflect-pad path. Sizes cover
/// the parity suite's own fixture geometries (64², 128², 192², 320×240) plus
/// the odd/non-16-aligned dims `odd_dim_pyramid_parity` exists for.
///
/// `to_bits()` equality, not a tolerance: the parity suite's tolerances exist
/// for CPU-vs-GPU f32 drift and are 2e-3..5e-2 wide. Two CPU walks computing
/// the same statistic over the same pixels have no licence to use any of that
/// budget, and if they ever did, every GPU budget in this crate would be
/// measuring against a moving reference.
#[test]
fn cpu_oracle_engines_agree() {
    const CELLS: &[(usize, usize)] = &[
        (64, 64),
        (128, 128),
        (192, 192),
        (320, 240),
        // the odd / non-16-aligned dims `odd_dim_pyramid_parity` covers
        (320, 241),
        (769, 513),
        // sub-64: the reflect-pad path `sub64_reflect_pad` covers
        (48, 40),
    ];
    let buffered = ZensimCpu::new(ZensimProfile::latest());
    let fold =
        ZensimCpu::new(ZensimProfile::latest()).with_engine(zensim::fold_engine::ScoringEngine::Fold);
    for &(w, h) in CELLS {
        let (rb, db) = pair(w, h);
        let (src, dst) = (to_pix(&rb), to_pix(&db));
        let s = RgbSlice::new(&src, w, h);
        let d = RgbSlice::new(&dst, w, h);

        let fb = buffered.compute_extended_features(&s, &d).unwrap();
        let ff = fold.compute_extended_features(&s, &d).unwrap();
        assert_eq!(
            fb.features().len(),
            ff.features().len(),
            "{w}x{h}: oracle feature width"
        );
        for (i, (&x, &y)) in fb.features().iter().zip(ff.features().iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{w}x{h}: oracle feature f{i} buffered {x:.17e} vs fold {y:.17e} — the CPU \
                 reference the GPU kernels are validated against must not depend on which \
                 CPU walk produced it"
            );
        }

        let sb = buffered.compute(&s, &d).unwrap();
        let sf = fold.compute(&s, &d).unwrap();
        assert_eq!(
            sb.score().to_bits(),
            sf.score().to_bits(),
            "{w}x{h}: oracle score buffered {:.17e} vs fold {:.17e}",
            sb.score(),
            sf.score()
        );
    }
}
