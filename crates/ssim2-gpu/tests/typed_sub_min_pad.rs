//! The typed `Ssim2<R>` public API pads sub-8px itself (not just the
//! `Ssim2Opaque` wrapper), so no public interface returns a degenerate
//! sub-floor result. Verifies: sub-8 `new` succeeds + `dimensions()`
//! reports logical; a sub-8 score equals the same image reflect-padded
//! to 8 and scored natively (definitive); scores down to 1×1.

#![cfg(all(feature = "cubecl-types", feature = "cuda"))]

use cubecl::Runtime;
use ssim2_gpu::Ssim2;
use zenmetrics_gpu_core::PadPlan;

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
fn typed_sub8_score_matches_manual_pad_to_8() {
    let (w, h) = (4u32, 4u32);
    let r = img(w, h, 3);
    let d = img(w, h, 29);

    let mut z = Ssim2::<Bt>::new(Bt::client(&Default::default()), w, h).expect("sub-8 pads");
    assert_eq!(z.dimensions(), (w, h), "dims() reports logical");
    let s_sub = z.compute(&r, &d).expect("sub-8 score").score;

    let plan = PadPlan::to_min(w, h, 8);
    let r8 = plan.pad(&r, 3).into_owned();
    let d8 = plan.pad(&d, 3).into_owned();
    let mut z8 = Ssim2::<Bt>::new(Bt::client(&Default::default()), 8, 8).expect("8 new");
    let s8 = z8.compute(&r8, &d8).expect("8 score").score;

    eprintln!("ssim2 typed sub-8={s_sub} manual-pad-8={s8} |Δ|={:.3e}", (s_sub - s8).abs());
    assert!(
        (s_sub - s8).abs() < 1e-9,
        "typed sub-8 score must equal manually-padded-to-8 score: {s_sub} vs {s8}"
    );
}

#[test]
fn typed_scores_down_to_1px() {
    for n in [1u32, 2, 4, 7, 8, 16] {
        let r = img(n, n, 0);
        let d = img(n, n, 7);
        let mut z = Ssim2::<Bt>::new(Bt::client(&Default::default()), n, n)
            .unwrap_or_else(|e| panic!("ssim2 new at {n}px must pad+succeed: {e:?}"));
        assert_eq!(z.dimensions(), (n, n));
        let s = z.compute(&r, &d).unwrap_or_else(|e| panic!("{n}px score: {e:?}")).score;
        assert!(s.is_finite(), "{n}px score must be finite, got {s}");
    }
}
