//! The typed `Zensim<R>` public API must pad sub-64px images itself —
//! not just the `ZensimOpaque` wrapper — so no public interface returns
//! a silently-degenerate sub-floor-pyramid result. Verifies: (1) `new`
//! accepts sub-64 and `dimensions()` reports the logical size; (2) a
//! sub-64 feature vector equals the same image reflect-padded to 64 and
//! scored natively (definitive: same padded image ⇒ identical features);
//! (3) every size down to 1×1 yields finite features.

#![cfg(all(feature = "cubecl-types", feature = "cuda"))]

use cubecl::Runtime;
use zenmetrics_gpu_core::PadPlan;
use zensim_gpu::Zensim;

type Bt = cubecl::cuda::CudaRuntime;

fn img(w: u32, h: u32, s: u32) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let x = i % w;
            let y = i / w;
            [
                ((x + s) & 0xff) as u8,
                ((y * 3 + s) & 0xff) as u8,
                ((x ^ y ^ s) & 0xff) as u8,
            ]
        })
        .collect()
}

#[test]
fn typed_new_accepts_sub64_and_reports_logical_dims() {
    let client = Bt::client(&Default::default());
    let z = Zensim::<Bt>::new(client, 32, 32).expect("typed new must accept sub-64 (pads)");
    assert_eq!(
        z.dimensions(),
        (32, 32),
        "dims() must report the logical size"
    );
}

#[test]
fn typed_sub64_features_match_manual_pad_to_64() {
    let (w, h) = (32u32, 32u32);
    let r = img(w, h, 3);
    let d = img(w, h, 29);

    // Sub-64 through the typed pipeline (pads internally to 64).
    let mut z_sub = Zensim::<Bt>::new(Bt::client(&Default::default()), w, h).expect("sub new");
    let f_sub = z_sub.compute_features(&r, &d).expect("sub features");

    // The same image, reflect-padded to 64 by hand, scored natively at 64.
    let plan = PadPlan::to_min(w, h, 64);
    let r64 = plan.pad(&r, 3).into_owned();
    let d64 = plan.pad(&d, 3).into_owned();
    let mut z64 = Zensim::<Bt>::new(Bt::client(&Default::default()), 64, 64).expect("64 new");
    let f64v = z64.compute_features(&r64, &d64).expect("64 features");

    // Same padded image through the same pipeline ⇒ identical features.
    let max_abs = f_sub
        .iter()
        .zip(f64v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    eprintln!("typed sub-64 vs manual-pad-to-64: max |Δ feature| = {max_abs:.3e}");
    assert!(
        max_abs < 1e-9,
        "typed sub-64 features must equal the manually-padded-to-64 features; max Δ {max_abs}"
    );
}

#[test]
fn typed_scores_every_size_down_to_1px() {
    for n in [1u32, 2, 3, 4, 8, 16, 32, 48, 63, 64] {
        let r = img(n, n, 0);
        let d = img(n, n, 7);
        let mut z = Zensim::<Bt>::new(Bt::client(&Default::default()), n, n)
            .unwrap_or_else(|e| panic!("typed new at {n}px must succeed, got {e:?}"));
        assert_eq!(z.dimensions(), (n, n));
        let f = z
            .compute_features(&r, &d)
            .unwrap_or_else(|e| panic!("{n}px features: {e:?}"));
        assert!(
            f.iter().all(|v| v.is_finite()),
            "{n}px features must be finite"
        );
    }
}
