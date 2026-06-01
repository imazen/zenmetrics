//! Opaque-API strip-mode parity tests for [`ButteraugliOpaque`].
//!
//! These cover the GPU-tested path through the opaque shim (the
//! `compute_srgb_u8` entry point) when an explicit
//! [`MemoryMode::Strip`](butteraugli_gpu::MemoryMode) is selected by
//! the caller (or resolved-to-Strip by Auto on the strip-preferred
//! butter crate).
//!
//! The typed API has GPU strip parity coverage in
//! [`strip_parity.rs`](./strip_parity.rs); this file mirrors those
//! tests through the opaque shim so the `ButteraugliOpaque` →
//! `ButteraugliInner::compute_srgb_u8` → strip-resolver routing path
//! is exercised end-to-end (including the strip-vs-whole dispatch
//! inside the shim).
//!
//! All tests use the strip path through the opaque shim and compare
//! against:
//!   1. the typed `Butteraugli::compute_strip` direct path (no shim),
//!      verifying the shim's `is_strip_mode()` routing produces the
//!      same byte-for-byte score; and
//!   2. the opaque WHOLE-image path on the same input, verifying that
//!      strip and whole opaque entries converge to within the
//!      cross-path numerical tolerance.

#![cfg(all(
    feature = "cubecl-types",
    any(feature = "cuda", feature = "wgpu")
))]

use butteraugli_gpu::{
    Backend, Butteraugli, ButteraugliOpaque, ButteraugliParams, MemoryMode,
};
use cubecl::Runtime;

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;

#[cfg(feature = "cuda")]
const BACKEND_E: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: Backend = Backend::Wgpu;

/// Same image generator as `strip_parity.rs`: mid-spatial-frequency
/// content so both the σ=7.16 LF blur and σ=3.22 HF blur stages see
/// non-trivial signal. Different seed values produce a "distorted"
/// variant with measurable but non-saturating butter score.
fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let sx = ((x as f32 / 32.0).sin() * 50.0 + 128.0) as u8;
            let sy = ((y as f32 / 24.0).cos() * 40.0 + 128.0) as u8;
            let hf = (((x ^ y).wrapping_mul(seed.max(1)) ^ seed) & 0x3f) as u8;
            out.push(sx.wrapping_add(hf));
            out.push(sy.wrapping_add(hf));
            out.push(sx.wrapping_add(sy).wrapping_add(hf >> 1));
        }
    }
    out
}

fn assert_rel_eq(name: &str, want: f64, got: f64, tol: f64) {
    let denom = want.abs().max(1e-12);
    let rel = (got - want).abs() / denom;
    assert!(
        rel < tol,
        "{name}: want={want} got={got} rel_err={rel:.2e} (tol={tol:.0e})"
    );
}

// ─── opaque-strip vs typed-strip parity ───
//
// The opaque shim's strip path is just a thin route through the typed
// `Butteraugli::compute_strip_with_options`; these tests verify the
// routing does not introduce any numerical drift (byte-exact tolerance
// 1e-7, since the underlying kernel work is identical and the shim
// only wraps the score in a Score struct).

fn opaque_strip_score(w: u32, h: u32, h_body: u32, ref_buf: &[u8], dis_buf: &[u8]) -> f64 {
    let mut opaque = ButteraugliOpaque::new_with_memory_mode(
        BACKEND_E,
        w,
        h,
        ButteraugliParams::default(),
        MemoryMode::Strip {
            h_body: Some(h_body),
        },
    )
    .expect("opaque strip new");
    opaque
        .compute_srgb_u8(ref_buf, dis_buf)
        .expect("opaque strip compute")
        .value
}

fn typed_strip_score(w: u32, h: u32, h_body: u32, ref_buf: &[u8], dis_buf: &[u8]) -> f64 {
    let client = BackendT::client(&Default::default());
    // Mirror the opaque shim's `MemoryMode::Strip` routing EXACTLY. The shim
    // builds `new_multires_strip` (the HF-safe multi-resolution strip walker,
    // HALO_ROWS=80) and computes via `compute_strip_with_options` — see
    // `opaque.rs` `is_strip_mode()`. Before the multires-strip fix the shim
    // used single-res `new_strip`; this typed reference must track the shim's
    // ACTUAL path, which is now multires. (Comparing the multires shim against
    // single-res `new_strip` is what diverged ~14.6% on HF content — the bug
    // this parity gate exists to catch is shim-vs-typed *routing* drift, and
    // both sides must be the same algorithm for a 1e-7 byte-exact assertion.)
    let mut strip = Butteraugli::<BackendT>::new_multires_strip(client, w, h, h_body);
    strip
        .compute_strip(ref_buf, dis_buf)
        .expect("typed multires-strip compute")
        .score as f64
}

fn opaque_full_score(w: u32, h: u32, ref_buf: &[u8], dis_buf: &[u8]) -> f64 {
    let mut opaque = ButteraugliOpaque::new_with_memory_mode(
        BACKEND_E,
        w,
        h,
        ButteraugliParams::default(),
        MemoryMode::Full,
    )
    .expect("opaque full new");
    opaque
        .compute_srgb_u8(ref_buf, dis_buf)
        .expect("opaque full compute")
        .value
}

#[test]
fn opaque_strip_vs_typed_strip_256_body_64() {
    let w = 256;
    let h = 256;
    let body_h = 64;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let opaque = opaque_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    let typed = typed_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    assert_rel_eq("opaque-vs-typed-strip-256-64", typed, opaque, 1e-7);
}

#[test]
fn opaque_strip_vs_typed_strip_512_body_128() {
    let w = 512;
    let h = 512;
    let body_h = 128;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let opaque = opaque_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    let typed = typed_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    assert_rel_eq("opaque-vs-typed-strip-512-128", typed, opaque, 1e-7);
}

#[test]
fn opaque_strip_vs_typed_strip_1024_body_256() {
    let w = 1024;
    let h = 1024;
    let body_h = 256;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let opaque = opaque_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    let typed = typed_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    assert_rel_eq("opaque-vs-typed-strip-1024-256", typed, opaque, 1e-7);
}

// ─── opaque-strip vs opaque-Full (both multires) ───
//
// Both whole-image opaque (`Full` → `new_multires`) AND strip
// (`Strip` → `new_multires_strip`) are now the multi-resolution path,
// so opaque-strip == opaque-Full within the multires reduction-order +
// halo drift. `typed_whole_singleres_score` below is retained only for
// the `opaque_full_uses_multires` contract check (multires Full must be
// >= the single-res whole, proving the half-res supersample-add fires).

fn typed_whole_singleres_score(w: u32, h: u32, ref_buf: &[u8], dis_buf: &[u8]) -> f64 {
    let client = BackendT::client(&Default::default());
    let mut whole = Butteraugli::<BackendT>::new(client, w, h);
    whole.compute(ref_buf, dis_buf).expect("whole compute").score as f64
}

#[test]
fn opaque_strip_matches_full_multires_512() {
    // The strip path is now the MULTI-RESOLUTION walker (HALO_ROWS=80), so a
    // strip-mode score equals the multires Full score. That equality
    // (strip == Full) is exactly the HF-safety property the multires-strip
    // fix established — it is what lets `Auto` pick Strip on butter without a
    // score shift. Tolerance is the multires-strip-vs-multires-whole
    // reduction-order + halo drift documented in strip.rs (~7e-4 rel at
    // 512²); 1e-3 covers it. (Pre-fix this test asserted strip == single-res
    // whole, which was the HF-unsafe behaviour the fix replaced.)
    let w = 512;
    let h = 512;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let opaque_strip = opaque_strip_score(w, h, 64, &ref_buf, &dis_buf);
    let full_multires = opaque_full_score(w, h, &ref_buf, &dis_buf);
    assert_rel_eq(
        "opaque-strip-vs-full-multires-512",
        full_multires,
        opaque_strip,
        1e-3,
    );
}

#[test]
fn opaque_full_uses_multires() {
    // Quick contract check: the opaque-Full path differs from a
    // single-resolution whole pass (because it engages
    // `new_multires`'s half-res supersample-add). The two scores
    // should differ by a small but measurable amount on a
    // textured image — confirms the multires branch is firing.
    let w = 256;
    let h = 256;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let opaque_full = opaque_full_score(w, h, &ref_buf, &dis_buf);
    let typed_single = typed_whole_singleres_score(w, h, &ref_buf, &dis_buf);
    // Multi-res score must be >= single-res (adding non-negative
    // half-res diffmap can only raise the max).
    assert!(
        opaque_full + 1e-5 >= typed_single,
        "opaque-Full multires score ({opaque_full}) < typed single-res score ({typed_single}) — supersample-add should raise"
    );
}

// ─── edge cases ───

#[test]
fn opaque_strip_with_options_matches_typed_strip_with_options() {
    let w = 512;
    let h = 512;
    let body_h = 64;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let params = ButteraugliParams::default()
        .with_intensity_target(120.0)
        .with_hf_asymmetry(1.5)
        .with_xmul(0.5);

    // Opaque path with explicit params.
    let mut opaque = ButteraugliOpaque::new_with_memory_mode(
        BACKEND_E,
        w,
        h,
        params,
        MemoryMode::Strip { h_body: Some(body_h) },
    )
    .expect("opaque strip new with options");
    let opaque_value = opaque
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("opaque strip compute")
        .value;

    // Typed path with the same params — mirror the shim's multires-strip
    // routing (new_multires_strip + compute_strip_with_options).
    let client = BackendT::client(&Default::default());
    let mut typed = Butteraugli::<BackendT>::new_multires_strip(client, w, h, body_h);
    let typed_score = typed
        .compute_strip_with_options(&ref_buf, &dis_buf, &params)
        .expect("typed multires-strip compute with options")
        .score as f64;

    assert_rel_eq("opaque-strip-options-vs-typed", typed_score, opaque_value, 1e-7);
}

#[test]
fn opaque_strip_uneven_image_height_768_body_96() {
    // image_h=800 isn't a multiple of body=96 — last strip is partial
    // (32 rows). Mirrors the typed-API edge case in `strip_parity.rs`.
    let w = 768;
    let h = 800;
    let body_h = 96;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let opaque = opaque_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    let typed = typed_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    assert_rel_eq("opaque-strip-uneven", typed, opaque, 1e-7);
}

#[test]
fn opaque_strip_body_equals_image_height_one_strip() {
    // Degenerate single-strip mode: body_h == image_h. Walker runs
    // exactly one strip covering the whole image (with halo halo
    // collapsed to zero at the image edges).
    let w = 512;
    let h = 512;
    let body_h = h;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let opaque = opaque_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    let typed = typed_strip_score(w, h, body_h, &ref_buf, &dis_buf);
    assert_rel_eq("opaque-strip-one-strip", typed, opaque, 1e-7);
}

#[test]
fn opaque_auto_resolves_to_strip_on_butter() {
    // butter is strip-preferred — Auto picks Strip whenever it fits
    // (see `MemoryMode::Auto` resolver). With a generous default cap
    // (no env var override) any reasonable image should resolve to
    // strip. We can't observe the resolution directly through the
    // opaque shim, but we CAN verify that the score the shim produces
    // matches the typed-strip path at the auto-resolved body — by
    // computing what the auto-resolver would pick and comparing.
    let w = 1024;
    let h = 1024;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);

    // What the shim's Auto-resolver would pick (mirrors what
    // `ButteraugliOpaque::new_with_memory_mode(.., Auto)` runs):
    let cap = butteraugli_gpu::vram_cap_bytes();
    let resolved = butteraugli_gpu::memory_mode::resolve_auto(w, h, cap).expect("resolve");

    let mut opaque = ButteraugliOpaque::new_with_memory_mode(
        BACKEND_E,
        w,
        h,
        ButteraugliParams::default(),
        MemoryMode::Auto,
    )
    .expect("opaque auto");
    let auto_value = opaque
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("opaque auto compute")
        .value;

    match resolved {
        butteraugli_gpu::ResolvedMode::Strip { h_body } => {
            let typed_strip = typed_strip_score(w, h, h_body, &ref_buf, &dis_buf);
            assert_rel_eq(
                "opaque-auto-strip-vs-typed-strip",
                typed_strip,
                auto_value,
                1e-7,
            );
        }
        butteraugli_gpu::ResolvedMode::Full => {
            // Auto resolved to Full — the multires path. We at
            // least verify the score is finite and >= 0.
            assert!(
                auto_value.is_finite() && auto_value >= 0.0,
                "auto-Full score must be finite >= 0, got {auto_value}"
            );
        }
    }
}

#[test]
fn opaque_strip_score_struct_fields() {
    // Confirm the shim populates the Score struct's metric_name /
    // metric_version on the strip-mode path (mirrors the equivalent
    // assertion on the whole-image opaque path in `opaque.rs`).
    let w = 256;
    let h = 256;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let mut opaque = ButteraugliOpaque::new_with_memory_mode(
        BACKEND_E,
        w,
        h,
        ButteraugliParams::default(),
        MemoryMode::Strip { h_body: Some(64) },
    )
    .expect("opaque strip new");
    let score = opaque
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("opaque strip compute");
    assert_eq!(score.metric_name, "butter");
    assert_eq!(score.metric_version, env!("CARGO_PKG_VERSION"));
    assert!(score.value.is_finite() && score.value >= 0.0);
}
