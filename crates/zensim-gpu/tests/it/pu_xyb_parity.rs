//! CPU↔GPU PU-XYB (HDR) parity for
//! [`zensim_gpu::kernels::color::linear_nits_to_positive_pu_xyb_kernel`].
//!
//! The GPU HDR kernel mirrors CPU `zensim::color::pu_xyb_pixel` (exposed
//! for benches/tests as `zensim::bench_pu_xyb_scalar`): the shared opsin
//! mix, then PU21 encode `/ PU_WHITE` and the opponent shift, instead of
//! the SDR cube root. This builds the scale-0 PU-XYB plane on-device from
//! absolute-luminance linear-nits planes and compares the logical
//! (non-pad) region against the CPU scalar reference.
//!
//! The only divergence is the f32 `powf` transcendental (CPU libm vs the
//! cubecl device intrinsic), the same class the cube-root `cpu_parity`
//! test bounds for the SDR path. Like that test, this is a real
//! regression gate on CUDA (cubecl emits identical kernel source across
//! backends; the Metal `powf` may diverge more — see the diffmap test's
//! Metal note — but CUDA/Vulkan match closely).
//!
//! Requires a real GPU runtime (CUDA or wgpu) — the CpuRuntime JIT does
//! not implement every intrinsic these kernels use.

#![cfg(feature = "cubecl-types")]

use cubecl::Runtime;
use zensim_gpu::Zensim;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!("pu_xyb_parity requires the `cuda` or `wgpu` feature to select a GPU runtime");

#[cfg(feature = "cuda")]
const BACKEND_E: zensim_gpu::Backend = zensim_gpu::Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: zensim_gpu::Backend = zensim_gpu::Backend::Wgpu;

/// PU-XYB CPU↔GPU absolute tolerance per channel. The residual is the
/// f32 `powf` divergence between CPU libm and the cubecl device
/// intrinsic over the `[0.005, 10000]` cd/m² operating range, amplified
/// by `PU_X_SCALE` (4×) on the X channel. **Measured** max over the
/// cases below at land time (RTX 5070): CUDA `1.43e-6`, Vulkan
/// (cubecl-wgpu) `1.91e-6` — the device `powf` is near-bit-exact to
/// libm here. `5e-5` is a genuine regression gate with ~26× margin over
/// the measured Vulkan envelope. (Metal is excluded from CI — its
/// `powf`/naga translation diverges more, same as the cube-root diffmap
/// test's Metal note.)
const PU_XYB_ABS_TOL: f32 = 5e-5;

/// Absolute-luminance linear-RGB pixels (cd/m²) spanning the PU
/// operating range: log-spaced 0.005..4000 nits with a per-pixel chroma
/// tilt + LCG noise, so the opsin mix, the PU21 knee, and the HDR
/// highlight tail are all exercised.
fn make_nits_pixels(seed: u32, w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(w * h);
    let mut s = seed.wrapping_add(1);
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let noise = ((s >> 16) & 0xff) as f32 / 255.0;
            let t = ((x as f32 / w.max(1) as f32 + y as f32 / h.max(1) as f32) * 0.5 + noise * 0.1)
                .clamp(0.0, 1.0);
            let yb = 0.005f32 * (4000.0f32 / 0.005).powf(t);
            out.push([yb * 1.15, yb, yb * 0.8]);
        }
    }
    out
}

fn run_case(w: usize, h: usize) {
    let pixels = make_nits_pixels(7, w, h);
    let n = w * h;

    // CPU scalar reference (the exact spec the GPU kernel mirrors).
    let (mut cx, mut cy, mut cb) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    zensim::bench_pu_xyb_scalar(&pixels, &mut cx, &mut cy, &mut cb);

    // Split interleaved pixels into the three tight planes the GPU kernel reads.
    let (mut r, mut g, mut b) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    for (i, p) in pixels.iter().enumerate() {
        r[i] = p[0];
        g[i] = p[1];
        b[i] = p[2];
    }

    // GPU: build the scale-0 PU-XYB plane on-device, read it back.
    let client = Backend::client(&Default::default());
    let mut gpu = Zensim::<Backend>::new(client, w as u32, h as u32).expect("Zensim::new");
    gpu.debug_build_pu_xyb_scale0(true, &r, &g, &b);
    let (pw, _ph) = gpu.debug_scale_dims(0);
    let pw = pw as usize;
    let gx = gpu.debug_read_xyb(0, 0, true);
    let gy = gpu.debug_read_xyb(0, 1, true);
    let gb = gpu.debug_read_xyb(0, 2, true);

    // Compare only the logical (non-pad) columns — pad columns hold
    // mirror-reflected pixels whose CPU counterpart isn't in this buffer.
    let mut max_d = 0.0f32;
    for yy in 0..h {
        for xx in 0..w {
            let ci = yy * w + xx;
            let gi = yy * pw + xx;
            for (gg, cc) in [(gx[gi], cx[ci]), (gy[gi], cy[ci]), (gb[gi], cb[ci])] {
                max_d = max_d.max((gg - cc).abs());
            }
        }
    }
    eprintln!("pu_xyb_parity {w}x{h} (padded_w={pw}): max |GPU-CPU| = {max_d:.3e}");
    assert!(
        max_d < PU_XYB_ABS_TOL,
        "PU-XYB CPU↔GPU parity failed at {w}x{h}: max {max_d:.3e} >= tol {PU_XYB_ABS_TOL:.0e}"
    );
}

#[test]
fn pu_xyb_matches_cpu_64() {
    run_case(64, 64);
}

#[test]
fn pu_xyb_matches_cpu_padded_cols() {
    // 96 is not a SIMD-padded multiple for every tier → exercises the
    // mirror-offset pad-column path in the kernel.
    run_case(96, 72);
}

#[test]
fn pu_xyb_matches_cpu_256() {
    run_case(256, 256);
}

/// End-to-end public-API smoke for the canonical WithIw (372) regime —
/// what `ZensimParams::with_profile` selects for every V0_3+ profile
/// (Profile A included). The opaque `compute_features_pu_linear_nits`
/// runs pad → PU kernel → pyramid → feature reduction and returns 372
/// finite, non-trivial features for distinct HDR inputs.
#[test]
fn opaque_pu_features_withiw_smoke() {
    use zensim_gpu::{ZensimFeatureRegime, ZensimOpaque, ZensimParams};
    let (w, h) = (64usize, 64usize);
    let n = w * h;
    let split = |px: &[[f32; 3]]| {
        let (mut r, mut g, mut b) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        for (i, p) in px.iter().enumerate() {
            r[i] = p[0];
            g[i] = p[1];
            b[i] = p[2];
        }
        (r, g, b)
    };
    let (rr, rg, rb) = split(&make_nits_pixels(11, w, h));
    let (dr, dg, db) = split(&make_nits_pixels(29, w, h));

    let mut z = ZensimOpaque::new(
        BACKEND_E,
        w as u32,
        h as u32,
        ZensimParams::new().with_regime(ZensimFeatureRegime::WithIw),
    )
    .expect("opaque new WithIw");
    let feats = z
        .compute_features_pu_linear_nits([&rr, &rg, &rb], [&dr, &dg, &db])
        .expect("compute ok");
    assert_eq!(feats.len(), 372, "WithIw PU features = 372");
    assert!(
        feats.iter().all(|f| f.is_finite()),
        "all PU features finite"
    );
    assert!(
        feats.iter().any(|&f| f.abs() > 0.0),
        "distinct HDR inputs must produce non-trivial PU features"
    );
}

/// The PU feature entry is regime-aware like the SDR `compute_features_vec`
/// — no artificial WithIw gate. The legacy Basic regime returns 228
/// features (no IW-pool block), not `None`.
#[test]
fn opaque_pu_features_basic_returns_228() {
    use zensim_gpu::{ZensimFeatureRegime, ZensimOpaque, ZensimParams};
    let (w, h) = (64usize, 64usize);
    let n = w * h;
    let (mut r, mut g, mut b) = (vec![1.0f32; n], vec![1.0f32; n], vec![1.0f32; n]);
    for i in 0..n {
        let v = 0.005 + (i as f32);
        r[i] = v * 1.1;
        g[i] = v;
        b[i] = v * 0.9;
    }
    let mut z = ZensimOpaque::new(
        BACKEND_E,
        w as u32,
        h as u32,
        ZensimParams::new().with_regime(ZensimFeatureRegime::Basic),
    )
    .expect("opaque new Basic");
    let feats = z
        .compute_features_pu_linear_nits([&r, &g, &b], [&r, &g, &b])
        .expect("compute ok");
    assert_eq!(feats.len(), 228, "Basic regime PU features = 228");
    assert!(
        feats.iter().all(|f| f.is_finite()),
        "all PU features finite"
    );
}

// ───────────── integrated-PU score path (zenmetrics#25) ─────────────

/// Interleaved absolute-nits pair for the score tests: `make_nits_pixels`
/// as the reference, a uniformly 10%-darker copy as the distortion (the
/// same pair shape the umbrella's HDR tests use). Flattened `[R,G,B, …]`.
fn interleaved_pair(w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let r: Vec<f32> = make_nits_pixels(7, w, h).into_iter().flatten().collect();
    let d: Vec<f32> = r.iter().map(|&v| v * 0.9).collect();
    (r, d)
}

/// Score tolerance for the GPU integrated-PU path vs CPU
/// `Zensim::compute_pu_linear` on the same nits pair (profile B → the
/// BHdr bake on both sides).
///
/// **Measured on Metal (Apple aarch64, wgpu) at land time:** |Δ| =
/// 5.0e-3 at 128×96, 4.66e-1 at 200×150, 3.6e-1 at 256×256. The drift is
/// NOT the PU color stage (the scale-0 PU-XYB parity above is < 5e-5) and
/// not the mean-pooled features (block gate below, < 2e-3 per feature): it
/// is concentrated in the **peak** features — `ssim_max` / `ssim_p95`
/// (feature indices 156 + 18·scale + 6·ch + {0, 3}) — GPU 4.5e-3 vs CPU
/// 1.5e-3 at 256×256 on the Y channel. Both implementations form the local
/// variance as `E[x²] − μ²` in f32; on PU-XYB planes (Y up to ≈ 2.5 at
/// 4000 cd/m², vs ≲ 1 on the SDR cube-root planes) the cancellation noise
/// in smooth regions is ~6× larger, the two blur orderings round it
/// differently, and a `max` pool picks the single worst pixel — so the
/// same peak features that agree to ~4% on SDR content diverge 3× here.
/// Scaling the pair down (×0.002, all pixels < 10 cd/m²) collapses the
/// score drift to 5e-3, which pins it to magnitude, not to the PU kernel
/// or the routing. The SDR path is unaffected (its own parity locks
/// stand). Tracked as a Known Bug in the repo CLAUDE.md; CUDA/Vulkan
/// numbers are still to be recorded (the SDR peak drift is smaller
/// there).
///
/// Content caveat (measured the same day): a **pure linear gradient** pair
/// (the umbrella tests' `hdr_pair`, 50..650 cd/m² ramp × 0.9) is NOT a
/// valid GPU↔CPU parity input on either path — `hf_mag_loss`
/// (`max(0, 1 − Σ|dst−μ_dst| / Σ|src−μ_src|)`, basic feature 11) is
/// ill-conditioned there because a box mean reproduces a ramp exactly, so
/// both sums are rounding noise and the ratio is arbitrary: CPU 0.22 vs
/// GPU 0.07 (PU, scale 2), CPU 0.19 vs GPU 0.0 (SDR, same feature). The
/// SDR bake barely weights it (score Δ 1e-2) but BHdr does (Δ 16 points on
/// PU). Use textured content (`make_nits_pixels`) for cross-backend
/// checks; the gradient stays fine for identity / monotonicity tests.
///
/// `1.0` is therefore a coarse-but-real gate: the u8-shell score this path
/// replaces (`SdrU8(PuRescale)` → SDR bake) and the wrong-bake mutation
/// (`B` instead of `BHdr` on PU features, exact-checked below) both land
/// > 1 point away.
const PU_SCORE_ABS_TOL: f64 = 1.0;

/// Per-feature tolerance for the **mean-pooled basic block** (4 scales × 3
/// channels × 13 = features 0..156) of the GPU PU features vs CPU
/// `compute_pu_linear`. Measured max |Δ| on Metal at land time: 1.4e-4
/// (200×150), 1.8e-4 (256×256), 3.1e-4 (256×256 at ×0.002 gain) — the
/// same f32-vs-f64 reduction class the SDR `cpu_parity` family bounds. A
/// broken pyramid tail, mirror pad, or PU encode shifts these by orders
/// of magnitude.
const PU_MEAN_FEATURE_ABS_TOL: f64 = 2e-3;
/// Mean-pooled basic block extent: `n_scales(4) × 3 ch × 13`.
const PU_MEAN_BLOCK_END: usize = 156;

/// The opaque's integrated-PU score tracks CPU `compute_pu_linear` (profile
/// B → the BHdr bake on both sides) within the measured envelope, the
/// mean-pooled basic features match tightly, and the canonical
/// 372-feature vector comes out of the same pass.
#[test]
fn opaque_pu_score_matches_cpu_compute_pu_linear() {
    use zensim_gpu::{ZensimOpaque, ZensimParams};
    for (w, h) in [(128usize, 96usize), (200, 150), (256, 256)] {
        let (r, d) = interleaved_pair(w, h);
        let mut z = ZensimOpaque::new(
            BACKEND_E,
            w as u32,
            h as u32,
            ZensimParams::new().with_profile(zensim::ZensimProfile::B),
        )
        .expect("opaque new B");
        let (gpu, feats) = z
            .compute_pu_linear_nits_interleaved(&r, &d)
            .expect("gpu pu score");
        assert_eq!(feats.len(), 372, "profile B → WithIw → 372 PU features");
        assert!(feats.iter().all(|f| f.is_finite()));

        let cres = zensim::Zensim::new(zensim::ZensimProfile::B)
            .compute_pu_linear(&r, &d, w, h, 3 * w, 3 * w)
            .expect("cpu compute_pu_linear");
        let cpu = cres.score();
        let cf = cres.features();
        assert_eq!(cf.len(), 372);
        let (mut worst_i, mut worst_d) = (0usize, 0.0f64);
        for i in 0..PU_MEAN_BLOCK_END {
            let ad = (feats[i] - cf[i]).abs();
            if ad > worst_d {
                worst_d = ad;
                worst_i = i;
            }
        }
        eprintln!(
            "pu_score_parity {w}x{h}: gpu {} cpu {cpu} |Δ| {:.3e}; mean-block worst idx {worst_i} |Δ| {worst_d:.3e}",
            gpu.value,
            (gpu.value - cpu).abs()
        );
        assert!(gpu.value.is_finite(), "gpu score finite");
        assert!(
            (gpu.value - cpu).abs() < PU_SCORE_ABS_TOL,
            "{w}x{h}: gpu {} vs cpu {cpu} exceeds tol {PU_SCORE_ABS_TOL}",
            gpu.value
        );
        assert!(
            worst_d < PU_MEAN_FEATURE_ABS_TOL,
            "{w}x{h}: mean-pooled feature {worst_i} gpu {} cpu {} |Δ| {worst_d:.3e} >= {PU_MEAN_FEATURE_ABS_TOL:.0e}",
            feats[worst_i],
            cf[worst_i]
        );
    }
}

/// Routing gate: PU-linear features MUST be scored with the profile's
/// PU-linear bake (`B` → `BHdr`, mirroring CPU `params_pu_linear`), never
/// the SDR `B` bake. The opaque score equals the public
/// `score_features_with_profile_and_codec(BHdr, features)` forward pass
/// bit-for-bit, and differs materially from scoring the same features with
/// `B` — so dropping the mapping fails here.
#[test]
fn opaque_pu_score_routes_b_to_bhdr_bake() {
    use zensim_gpu::{ZensimOpaque, ZensimParams};
    let (w, h) = (128usize, 96usize);
    let (r, d) = interleaved_pair(w, h);
    let mut z = ZensimOpaque::new(
        BACKEND_E,
        w as u32,
        h as u32,
        ZensimParams::new().with_profile(zensim::ZensimProfile::B),
    )
    .expect("opaque new B");
    let (gpu, feats) = z
        .compute_pu_linear_nits_interleaved(&r, &d)
        .expect("gpu pu score");
    let (pw, ph) = (w as u32, h as u32);
    let via_bhdr = zensim::score_features_with_profile_and_codec(
        zensim::ZensimProfile::BHdr,
        &feats,
        pw,
        ph,
        None,
    )
    .expect("BHdr scores 372 PU features");
    let via_b = zensim::score_features_with_profile_and_codec(
        zensim::ZensimProfile::B,
        &feats,
        pw,
        ph,
        None,
    )
    .expect("B scores 372 features");
    assert_eq!(
        gpu.value, via_bhdr,
        "opaque PU score must be the BHdr forward pass"
    );
    assert!(
        (via_b - via_bhdr).abs() > 1.0,
        "SDR B bake on PU features ({via_b}) must differ from BHdr ({via_bhdr}) — otherwise this gate is void"
    );

    // An explicit BHdr profile scores identically (the mapping is idempotent).
    let mut zh = ZensimOpaque::new(
        BACKEND_E,
        w as u32,
        h as u32,
        ZensimParams::new().with_profile(zensim::ZensimProfile::BHdr),
    )
    .expect("opaque new BHdr");
    let (gpu_h, _) = zh
        .compute_pu_linear_nits_interleaved(&r, &d)
        .expect("gpu pu score BHdr");
    assert_eq!(gpu_h.value, gpu.value, "explicit BHdr == B routed to BHdr");
}

/// Identity scores exactly 100 (the `mark_identical` contract) with the
/// features still extracted; a wrong-length buffer is a loud
/// `DimensionMismatch`, not a mis-scored pair.
#[test]
fn opaque_pu_score_identity_100_and_length_check() {
    use zensim_gpu::{ZensimOpaque, ZensimParams};
    let (w, h) = (64usize, 64usize);
    let (r, d) = interleaved_pair(w, h);
    let mut z = ZensimOpaque::new(
        BACKEND_E,
        w as u32,
        h as u32,
        ZensimParams::new().with_profile(zensim::ZensimProfile::B),
    )
    .expect("opaque new B");
    let (id, feats) = z
        .compute_pu_linear_nits_interleaved(&r, &r)
        .expect("identity");
    assert_eq!(
        id.value, 100.0,
        "identity must short-circuit to exactly 100"
    );
    assert_eq!(feats.len(), 372);

    let (dist, _) = z
        .compute_pu_linear_nits_interleaved(&r, &d)
        .expect("distorted");
    assert!(
        dist.value < 100.0,
        "darkened copy {} must score below identity",
        dist.value
    );

    let err = z
        .compute_pu_linear_nits_interleaved(&r, &d[..d.len() - 3])
        .expect_err("short buffer must be rejected");
    assert!(
        matches!(err, zensim_gpu::Error::DimensionMismatch { .. }),
        "expected DimensionMismatch, got {err:?}"
    );
}
