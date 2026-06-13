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
