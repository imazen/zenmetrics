//! Opaque-API unit tests for `dssim-gpu`.
//!
//! Verifies:
//!  - `opaque_srgb_u8_matches_typed`: same inputs through
//!    [`DssimOpaque`] vs the typed `Dssim<R>` produce bit-identical
//!    [`Score`].
//!  - `opaque_pixels_handles_stride`: a `PixelSlice` with stride
//!    padding gives the same score as the same image with tight
//!    stride.
//!
//! Both tests require `--features cubecl-types,cuda` (or `wgpu`).
//! They're scoped to the feature so default builds (which expose
//! only the opaque path) compile this test as a no-op.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use cubecl::Runtime;
use dssim_gpu::{Backend, Dssim, DssimOpaque, DssimParams};

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;

#[cfg(feature = "cuda")]
const BACKEND_E: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: Backend = Backend::Wgpu;

fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x + seed) & 0xff) as u8;
            let g = ((y + seed * 3) & 0xff) as u8;
            let b = ((x ^ y ^ seed) & 0xff) as u8;
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

#[test]
fn opaque_srgb_u8_matches_typed() {
    let w = 64_u32;
    let h = 64_u32;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);

    // Typed path.
    let client = BackendT::client(&Default::default());
    let mut typed = Dssim::<BackendT>::new(client, w, h).expect("typed new");
    let typed_score = typed.compute(&ref_buf, &dis_buf).expect("typed compute");

    // Opaque path.
    let mut opaque = DssimOpaque::new(BACKEND_E, w, h, DssimParams::DEFAULT).expect("opaque new");
    let opaque_score = opaque
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("opaque compute_srgb_u8");

    // Match within tight tolerance — same kernel, same input bytes,
    // but two separate clients / GPU buffer allocations so atomic-add
    // reduction order may differ by a few ULPs.
    let rel = (opaque_score.value - typed_score.score).abs() / typed_score.score.abs().max(1e-12);
    assert!(
        rel < 1e-5,
        "opaque {} vs typed {} differ by rel {}",
        opaque_score.value,
        typed_score.score,
        rel
    );
    assert_eq!(opaque_score.metric_name, "dssim");
    assert_eq!(opaque_score.metric_version, env!("CARGO_PKG_VERSION"));
}

#[cfg(feature = "pixels")]
#[test]
fn opaque_pixels_handles_stride() {
    use zenpixels::{PixelDescriptor, PixelSlice};

    let w = 64_u32;
    let h = 64_u32;
    let tight = make_image(w, h, 11);
    let dist_tight = make_image(w, h, 23);

    let descriptor = PixelDescriptor::RGB8_SRGB;
    let row_bytes = w as usize * 3;

    // Build a strided buffer with padding that keeps pixel alignment
    // (zenpixels validates stride % bpp == 0). 15 bytes = 5 RGB
    // pixels of padding per row, on top of `row_bytes`.
    let pad = 15usize;
    let stride = row_bytes + pad;
    assert_eq!(stride % 3, 0, "stride must be pixel-aligned");
    let mut padded_ref = vec![0xAA_u8; stride * h as usize];
    let mut padded_dis = vec![0xBB_u8; stride * h as usize];
    for y in 0..h as usize {
        padded_ref[y * stride..y * stride + row_bytes]
            .copy_from_slice(&tight[y * row_bytes..(y + 1) * row_bytes]);
        padded_dis[y * stride..y * stride + row_bytes]
            .copy_from_slice(&dist_tight[y * row_bytes..(y + 1) * row_bytes]);
    }

    let mut opaque = DssimOpaque::new(BACKEND_E, w, h, DssimParams::DEFAULT).expect("opaque new");

    // Tight slice path.
    let r_tight = PixelSlice::new(&tight, w, h, row_bytes, descriptor).expect("tight ref slice");
    let d_tight =
        PixelSlice::new(&dist_tight, w, h, row_bytes, descriptor).expect("tight dist slice");
    let tight_score = opaque
        .compute_pixels(r_tight, d_tight)
        .expect("tight compute_pixels");

    // Strided slice path.
    let r_padded =
        PixelSlice::new(&padded_ref, w, h, stride, descriptor).expect("padded ref slice");
    let d_padded =
        PixelSlice::new(&padded_dis, w, h, stride, descriptor).expect("padded dist slice");
    let padded_score = opaque
        .compute_pixels(r_padded, d_padded)
        .expect("padded compute_pixels");

    // The dssim-gpu reduction uses Atomic<f32>::fetch_add by default
    // (the `fast-reduction` feature) so two calls with the same input
    // bytes can differ at the few-ULP level. Verify the strided path
    // produces the same pixels as the tight path within a tight
    // tolerance.
    let rel = (tight_score.value - padded_score.value).abs() / tight_score.value.abs().max(1e-12);
    assert!(
        rel < 1e-5,
        "strided {} vs tight {} differ by rel {}",
        padded_score.value,
        tight_score.value,
        rel
    );
}
