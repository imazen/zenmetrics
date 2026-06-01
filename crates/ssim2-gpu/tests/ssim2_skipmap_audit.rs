//! Skip-map mode parity tests.
//!
//! Verifies that the four [`Ssim2Mode`] values:
//! 1. `Full` matches the score the pipeline produced before skip-map
//!    dispatch landed (witnessed via direct comparison against
//!    `Ssim2Mode::Full` on the same fixture).
//! 2. `Lossless` is **bit-identical** to `Full` (skipping zero-weight
//!    cells changes the score by exactly zero — they contribute
//!    `0 * value` to the weighted sum).
//! 3. `Fast` and `Faster` agree with `Full` to better than `5e-4`
//!    relative on real-image fixtures (loose because cells with
//!    `|w| < threshold` still have nonzero contribution; the bound is
//!    `threshold * sum_of_values` which can be order `0.054` for
//!    Fast, `0.54` for Faster — but real data lands far below the
//!    worst case).
//!
//! Backend is selected at compile time (matches `parity_lock.rs`).

use cubecl::Runtime;
use ssim2_gpu::{Ssim2, Ssim2Batch, Ssim2Mode};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "ssim2-gpu integration tests require either the `cuda` or `wgpu` feature to select a runtime"
);

fn load_rgb8(path: &std::path::Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rgb8 = img.to_rgb8();
    let (w, h) = (rgb8.width(), rgb8.height());
    (rgb8.into_raw(), w, h)
}

fn corpus_dir() -> std::path::PathBuf {
    zenmetrics_corpus::corpus_dir()
}

/// Score the same JPEG distortion at each mode; check the four
/// invariants.
#[test]
fn modes_agree_on_jpeg_corpus() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");

    // 5 distortions, each at 4 modes
    let mut printed_header = false;
    for q in [5u32, 20, 45, 70, 90] {
        let path = dir.join(format!("q{q}.jpg"));
        let (dis, _, _) = load_rgb8(&path);

        let full = s
            .compute_with_mode(Ssim2Mode::Full, &src_bytes, &dis)
            .expect("Full")
            .score;
        let lossless = s
            .compute_with_mode(Ssim2Mode::Lossless, &src_bytes, &dis)
            .expect("Lossless")
            .score;
        let fast = s
            .compute_with_mode(Ssim2Mode::Fast, &src_bytes, &dis)
            .expect("Fast")
            .score;
        let faster = s
            .compute_with_mode(Ssim2Mode::Faster, &src_bytes, &dis)
            .expect("Faster")
            .score;

        if !printed_header {
            eprintln!(
                "{:>4} {:>14} {:>14} {:>14} {:>14}   |Δlossless|   |Δfast|/Δfull   |Δfaster|/Δfull",
                "q", "Full", "Lossless", "Fast", "Faster"
            );
            printed_header = true;
        }
        let dl = (lossless - full).abs();
        let dfa = (fast - full).abs();
        let dfx = (faster - full).abs();
        let rel_f = if full.abs() > 1e-3 {
            dfa / full.abs()
        } else {
            0.0
        };
        let rel_x = if full.abs() > 1e-3 {
            dfx / full.abs()
        } else {
            0.0
        };
        eprintln!(
            "{q:>4} {:>14.6} {:>14.6} {:>14.6} {:>14.6}   {:>10.3e}   {:>10.3e}   {:>10.3e}",
            full, lossless, fast, faster, dl, rel_f, rel_x
        );

        // Lossless skips only zero-weight cells, so the *algebraic* answer
        // is identical to Full (each skipped cell contributes 0 * v = 0
        // to the weighted sum). Whether the GPU's empirical result is
        // bit-identical depends on the reduction path:
        //
        // - With `fast-reduction` (default): the per-plane reduction uses
        //   `Atomic<f32>::fetch_add`, whose commit order across cubes
        //   varies across launches. Lossless skips some atomic adds
        //   entirely (zero-weight cells short-circuit), so the surviving
        //   add order is a different permutation than Full's — sub-ulp
        //   reorder drift on accumulators in the 0..100 band. Observed
        //   max |Δ| at q=5 on the JPEG corpus = 1.03e-5; the gate at
        //   5e-5 sits ~5× above the worst observed value and 4 orders of
        //   magnitude below any real miscoding signal (~1e-2+).
        // - Without `fast-reduction` (portable path): per-thread partials
        //   are written to a scratch buffer and the finalize kernel sums
        //   them in a deterministic `k = 0 .. n_threads` order. Lossless
        //   writes zeros where Full writes the real values, so the
        //   finalize sees the same algebraic input order with some terms
        //   replaced by their algebraic identity (0). The result is
        //   measured |Δ| = 0.000e0 across q ∈ {5, 20, 45, 70, 90} — gate
        //   at 1e-6 to catch any drift the determinism analysis missed.
        //
        // See `lossless_identical_pair_is_bit_exact` below for the
        // synthetic identical-pair check (independent of fast-reduction).
        // See: 2026-05-22 ssim2 flakes-warnings sweep + tighten-tolerances.
        #[cfg(feature = "fast-reduction")]
        let lossless_tol = 5e-5;
        #[cfg(not(feature = "fast-reduction"))]
        let lossless_tol = 1e-6;
        assert!(
            dl < lossless_tol,
            "Lossless not within reduction noise floor of Full at q={q}: full={full:.10}, lossless={lossless:.10}, Δ={dl:.3e}, tol={lossless_tol:.0e}"
        );
        // Fast / Faster bounded by `5e-4` relative.
        assert!(
            rel_f < 5e-4,
            "Fast diverged from Full at q={q}: full={full:.6}, fast={fast:.6}, rel={rel_f:.4e}"
        );
        assert!(
            rel_x < 5e-4,
            "Faster diverged from Full at q={q}: full={full:.6}, faster={faster:.6}, rel={rel_x:.4e}"
        );
    }
}

/// Identical-pair invariant: Full and Lossless modes on an identical
/// (ref == dis) input both produce score = 100 exactly. This is the
/// "bit-exact" test that complements the relaxed corpus gate above —
/// the corpus version absorbs GPU atomic-reduce reordering noise on
/// non-identical pairs, this one verifies the algebraic floor (no
/// distortion = perfect score, regardless of skip-map dispatch).
#[test]
fn lossless_identical_pair_is_bit_exact() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");

    let full = s
        .compute_with_mode(Ssim2Mode::Full, &src_bytes, &src_bytes)
        .expect("Full identical")
        .score;
    let lossless = s
        .compute_with_mode(Ssim2Mode::Lossless, &src_bytes, &src_bytes)
        .expect("Lossless identical")
        .score;

    let dl = (lossless - full).abs();
    // Identical-pair scores avoid the cross-call atomic-reorder noise
    // because every reduced value is the same algebraic constant — no
    // accumulator-ordering surface area to drift. Gate stays at 1e-6.
    assert!(
        dl < 1e-6,
        "Identical-pair Lossless not bit-identical to Full: full={full:.10}, lossless={lossless:.10}, Δ={dl:.3e}"
    );
}

/// Identical-image score under each mode. Should be ~100 in all four
/// (with backend-dependent tolerance).
#[test]
fn identical_image_all_modes() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");

    for mode in [
        Ssim2Mode::Full,
        Ssim2Mode::Lossless,
        Ssim2Mode::Fast,
        Ssim2Mode::Faster,
    ] {
        let r = s
            .compute_with_mode(mode, &src_bytes, &src_bytes)
            .expect("identical")
            .score;
        // CUDA's deterministic ordering gets very close to 100; cross-vendor
        // wgpu / Metal accumulates ulp noise through the IIR blur. Match
        // the tolerance in `parity_lock::identical_image_scores_100`.
        assert!(
            (r - 100.0).abs() < 1.0,
            "Identical image scored {r:.6} under mode {mode:?} (expected ≈100)"
        );
    }
}

/// Cached-reference path agrees with the direct path across all modes.
#[test]
fn cached_reference_matches_direct_all_modes() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));
    let (dis_bytes, _, _) = load_rgb8(&dir.join("q45.jpg"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");

    s.set_reference(&src_bytes).expect("set_reference");

    for mode in [
        Ssim2Mode::Full,
        Ssim2Mode::Lossless,
        Ssim2Mode::Fast,
        Ssim2Mode::Faster,
    ] {
        let direct = s
            .compute_with_mode(mode, &src_bytes, &dis_bytes)
            .expect("direct")
            .score;
        let cached = s
            .compute_with_reference_with_mode(mode, &dis_bytes)
            .expect("cached")
            .score;
        let d = (cached - direct).abs();
        assert!(
            d < 1e-3,
            "Mode {mode:?}: direct={direct:.6}, cached={cached:.6}, Δ={d:.3e}"
        );
    }
}

/// Batched path agrees with single-image path under each mode.
#[test]
fn batch_matches_single_image_all_modes() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));
    let (dis_bytes, _, _) = load_rgb8(&dir.join("q45.jpg"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client.clone(), w, h).expect("Ssim2::new");

    let mut b = Ssim2Batch::<Backend>::new(client, w, h, 4).expect("Ssim2Batch::new");
    b.set_reference(&src_bytes).expect("batch set_reference");

    for mode in [
        Ssim2Mode::Full,
        Ssim2Mode::Lossless,
        Ssim2Mode::Fast,
        Ssim2Mode::Faster,
    ] {
        let single = s
            .compute_with_mode(mode, &src_bytes, &dis_bytes)
            .expect("single")
            .score;
        let batch_inputs: Vec<Vec<u8>> = (0..4).map(|_| dis_bytes.clone()).collect();
        let batched = b
            .compute_batch_with_mode(mode, &batch_inputs)
            .expect("batch")[0]
            .score;
        let d = (batched - single).abs();
        assert!(
            d < 1e-2,
            "Mode {mode:?}: single={single:.6}, batched={batched:.6}, Δ={d:.3e}"
        );
    }
}
