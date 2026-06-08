//! The typed `Cvvdp<R>` public API pads sub-8px itself (not just the
//! `CvvdpOpaque` wrapper), so no public interface returns a degenerate
//! sub-floor result. Verifies: sub-8 `new` succeeds + `dimensions()`
//! reports logical; a sub-8 score equals the same image reflect-padded
//! to 8 and scored natively at 8 (the definitive bit-identical check);
//! scores down to 1×1; the diffmap is cropped back to the logical
//! extent; and the linear-planes entry pads identically to the byte path.
//!
//! cvvdp is display-aware, but the PPD is a property of the configured
//! `DisplayGeometry`, NOT the image dims — both the sub-8 instance and
//! the native-8 instance use `STANDARD_4K`, so geometry is bit-identical
//! and the only difference (the reflect-padded pixels) is exactly what
//! the sub-8 instance produces internally. Hence the |Δ| < 1e-9 gate.

#![cfg(all(feature = "cubecl-types", feature = "cuda"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::CvvdpParams;
use zenmetrics_gpu_core::PadPlan;

type Bt = cubecl::cuda::CudaRuntime;

const MIN: u32 = 8;

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

/// sRGB-u8 → unit-scaled linear-light, one plane per primary. Mirrors the
/// LUT the byte path uses so the linear-planes path can be checked against
/// a manually-padded native-8 linear-planes score.
fn to_linear_planes(srgb: &[u8], n: usize) -> [Vec<f32>; 3] {
    let lin = |c: u8| -> f32 {
        let x = c as f32 / 255.0;
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for px in srgb.chunks_exact(3) {
        r.push(lin(px[0]));
        g.push(lin(px[1]));
        b.push(lin(px[2]));
    }
    [r, g, b]
}

/// The definitive check: a sub-8 typed score is bit-identical to the same
/// image reflect-padded to 8 and scored natively on an 8×8 instance.
#[test]
fn typed_sub8_score_matches_manual_pad_to_8() {
    let (w, h) = (4u32, 4u32);
    let r = img(w, h, 3);
    let d = img(w, h, 29);

    let mut z = Cvvdp::<Bt>::new(
        Bt::client(&Default::default()),
        w,
        h,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("sub-8 pads");
    assert_eq!(z.dimensions(), (w, h), "dims() reports logical");
    let s_sub = z.score(&r, &d).expect("sub-8 score");

    let plan = PadPlan::to_min(w, h, MIN);
    let r8 = plan.pad(&r, 3).into_owned();
    let d8 = plan.pad(&d, 3).into_owned();
    let mut z8 = Cvvdp::<Bt>::new(
        Bt::client(&Default::default()),
        MIN,
        MIN,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("8 new");
    let s8 = z8.score(&r8, &d8).expect("8 score");

    eprintln!(
        "cvvdp typed sub-8={s_sub} manual-pad-8={s8} |Δ|={:.3e}",
        (s_sub - s8).abs()
    );
    assert!(
        (s_sub - s8).abs() < 1e-9,
        "typed sub-8 score must equal manually-padded-to-8 score: {s_sub} vs {s8}"
    );
}

/// Non-square sub-8 (one axis already ≥8) also pads bit-identically.
#[test]
fn typed_nonsquare_sub8_score_matches_manual_pad_to_8() {
    let (w, h) = (3u32, 11u32);
    let r = img(w, h, 5);
    let d = img(w, h, 41);

    let mut z = Cvvdp::<Bt>::new(
        Bt::client(&Default::default()),
        w,
        h,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("sub-8 pads");
    assert_eq!(z.dimensions(), (w, h));
    let s_sub = z.score(&r, &d).expect("sub score");

    let plan = PadPlan::to_min(w, h, MIN);
    let (pw, ph) = plan.padded();
    let rp = plan.pad(&r, 3).into_owned();
    let dp = plan.pad(&d, 3).into_owned();
    let mut zp = Cvvdp::<Bt>::new(
        Bt::client(&Default::default()),
        pw,
        ph,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("padded new");
    let sp = zp.score(&rp, &dp).expect("padded score");

    eprintln!(
        "cvvdp {w}x{h}: sub={s_sub} pad-{pw}x{ph}={sp} |Δ|={:.3e}",
        (s_sub - sp).abs()
    );
    assert!(
        (s_sub - sp).abs() < 1e-9,
        "typed {w}x{h} score must equal manually-padded-to-{pw}x{ph}: {s_sub} vs {sp}"
    );
}

/// `score_with_diffmap` on a sub-8 typed instance returns a diffmap
/// cropped back to the LOGICAL extent (not the padded extent), and the
/// JOD matches the manually-padded native score bit-for-bit.
#[test]
fn typed_sub8_diffmap_cropped_to_logical_and_score_matches() {
    let (w, h) = (5u32, 5u32);
    let r = img(w, h, 11);
    let d = img(w, h, 200);

    let mut z = Cvvdp::<Bt>::new(
        Bt::client(&Default::default()),
        w,
        h,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("sub-8 pads");
    let mut dm = Vec::new();
    let s_sub = z
        .score_with_diffmap(&r, &d, &mut dm)
        .expect("sub diffmap score");
    assert_eq!(
        dm.len(),
        (w * h) as usize,
        "diffmap must be cropped to the logical {w}x{h} extent, got {} elems",
        dm.len()
    );
    assert!(
        dm.iter().all(|v| v.is_finite()),
        "diffmap values must be finite"
    );

    let plan = PadPlan::to_min(w, h, MIN);
    let r8 = plan.pad(&r, 3).into_owned();
    let d8 = plan.pad(&d, 3).into_owned();
    let mut z8 = Cvvdp::<Bt>::new(
        Bt::client(&Default::default()),
        MIN,
        MIN,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("8 new");
    let mut dm8 = Vec::new();
    let s8 = z8
        .score_with_diffmap(&r8, &d8, &mut dm8)
        .expect("8 diffmap score");

    eprintln!(
        "cvvdp sub-8 diffmap JOD={s_sub} pad-8 JOD={s8} |Δ|={:.3e} (dm8 padded len={})",
        (s_sub - s8).abs(),
        dm8.len()
    );
    assert!(
        (s_sub - s8).abs() < 1e-9,
        "sub-8 diffmap JOD must equal manually-padded-to-8 JOD: {s_sub} vs {s8}"
    );

    // The cropped sub-8 diffmap must equal the top-left logical
    // sub-rectangle of the padded native-8 diffmap (bit-for-bit) — the
    // crop is exactly that sub-rectangle.
    let pw = plan.padded().0 as usize;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let got = dm[y * w as usize + x];
            let want = dm8[y * pw + x];
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "cropped diffmap[{x},{y}]={got} must equal padded native diffmap[{x},{y}]={want}"
            );
        }
    }
}

/// The linear-planes entry pads identically to the byte path: a sub-8
/// `score_from_linear_planes` equals the same planes reflect-padded to 8
/// and scored natively.
#[test]
fn typed_sub8_linear_planes_matches_manual_pad_to_8() {
    let (w, h) = (6u32, 6u32);
    let n = (w * h) as usize;
    let r = img(w, h, 17);
    let d = img(w, h, 123);
    let [rr, rg, rb] = to_linear_planes(&r, n);
    let [dr, dg, db] = to_linear_planes(&d, n);

    let mut z = Cvvdp::<Bt>::new(
        Bt::client(&Default::default()),
        w,
        h,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("sub-8 pads");
    let s_sub = z
        .score_from_linear_planes(&rr, &rg, &rb, &dr, &dg, &db)
        .expect("sub linear-planes score");

    let plan = PadPlan::to_min(w, h, MIN);
    let p = |v: &[f32]| plan.pad(v, 1).into_owned();
    let mut z8 = Cvvdp::<Bt>::new(
        Bt::client(&Default::default()),
        MIN,
        MIN,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("8 new");
    let s8 = z8
        .score_from_linear_planes(&p(&rr), &p(&rg), &p(&rb), &p(&dr), &p(&dg), &p(&db))
        .expect("8 linear-planes score");

    eprintln!(
        "cvvdp linear-planes sub-8={s_sub} manual-pad-8={s8} |Δ|={:.3e}",
        (s_sub - s8).abs()
    );
    assert!(
        (s_sub - s8).abs() < 1e-9,
        "sub-8 linear-planes score must equal manually-padded-to-8 score: {s_sub} vs {s8}"
    );
}

/// Every size down to 1×1 scores finite with `dimensions()` logical.
#[test]
fn typed_scores_down_to_1px() {
    for n in [1u32, 2, 4, 7, 8, 16] {
        let r = img(n, n, 0);
        let d = img(n, n, 7);
        let mut z = Cvvdp::<Bt>::new(
            Bt::client(&Default::default()),
            n,
            n,
            CvvdpParams::PLACEHOLDER,
        )
        .unwrap_or_else(|e| panic!("cvvdp new at {n}px must pad+succeed: {e:?}"));
        assert_eq!(z.dimensions(), (n, n));
        let s = z
            .score(&r, &d)
            .unwrap_or_else(|e| panic!("{n}px score: {e:?}"));
        assert!(s.is_finite(), "{n}px score must be finite, got {s}");
    }
}
