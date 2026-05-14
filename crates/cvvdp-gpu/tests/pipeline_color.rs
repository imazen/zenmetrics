//! Integration test for `Cvvdp::compute_dkl_planes` — exercises the
//! upload + LUT-init + color-kernel path end-to-end through the
//! pipeline. Compares against the host scalar `srgb_byte_to_dkl_scalar`.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
use cvvdp_gpu::kernels::csf::precomputed_band_weights;
use cvvdp_gpu::kernels::pyramid::{gausspyr_reduce_scalar, laplacian_pyramid_dec_scalar};
use cvvdp_gpu::params::DisplayGeometry;
use cvvdp_gpu::params::{CvvdpParams, DisplayModel};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[test]
fn compute_dkl_planes_matches_host_scalar() {
    let client = Backend::client(&Default::default());
    let (w, h) = (16u32, 16u32);
    let n = (w * h) as usize;
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // Non-trivial RGB pattern.
    let mut srgb = Vec::with_capacity(n * 3);
    for i in 0..n {
        srgb.push((i % 251) as u8);
        srgb.push(((i * 7 + 13) % 251) as u8);
        srgb.push(((i * 19 + 41) % 251) as u8);
    }

    let [a, rg, vy] = cvvdp.compute_dkl_planes(&srgb).expect("compute_dkl_planes");
    assert_eq!(a.len(), n);
    assert_eq!(rg.len(), n);
    assert_eq!(vy.len(), n);

    let display = DisplayModel::STANDARD_4K;
    let mut max_err = 0.0_f32;
    for i in 0..n {
        let (ea, erg, evy) = srgb_byte_to_dkl_scalar(
            srgb[i * 3],
            srgb[i * 3 + 1],
            srgb[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        for d in [(a[i] - ea).abs(), (rg[i] - erg).abs(), (vy[i] - evy).abs()] {
            if d > max_err {
                max_err = d;
            }
        }
    }
    // 3e-5 absolute — same FMA-vs-non-FMA slack as the kernel-only
    // test in color_kernel.rs (DKL output magnitudes ~200, 1 ULP ≈ 1.5e-5).
    assert!(
        max_err < 3e-5,
        "compute_dkl_planes vs host scalar max-abs = {max_err}"
    );
}

#[test]
fn compute_dkl_gauss_pyramid_matches_host_scalar() {
    let client = Backend::client(&Default::default());
    let (w, h) = (16u32, 16u32);
    let n = (w * h) as usize;
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let mut srgb = Vec::with_capacity(n * 3);
    for i in 0..n {
        srgb.push((i % 251) as u8);
        srgb.push(((i * 7 + 13) % 251) as u8);
        srgb.push(((i * 19 + 41) % 251) as u8);
    }

    let pyramid = cvvdp
        .compute_dkl_gauss_pyramid(&srgb)
        .expect("compute_dkl_gauss_pyramid");
    assert!(!pyramid.is_empty(), "pyramid had no levels");

    // Host reference: compute level-0 via host scalar, then chain
    // gausspyr_reduce_scalar for each subsequent level. Each level
    // should match the GPU output within FMA-tolerance.
    let display = DisplayModel::STANDARD_4K;

    // Level 0: from host scalar color transform.
    let mut host_level: [Vec<f32>; 3] = [vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]];
    for i in 0..n {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            srgb[i * 3],
            srgb[i * 3 + 1],
            srgb[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        host_level[0][i] = a;
        host_level[1][i] = rg;
        host_level[2][i] = vy;
    }

    let mut prev_w = w as usize;
    let mut prev_h = h as usize;
    for (k, gpu_level) in pyramid.iter().enumerate() {
        let expected_w = if k == 0 { w as usize } else { prev_w / 2 };
        let expected_h = if k == 0 { h as usize } else { prev_h / 2 };
        let expected_n = expected_w * expected_h;
        for c in 0..3 {
            assert_eq!(
                gpu_level[c].len(),
                expected_n,
                "level {k} channel {c}: got {} elements, expected {expected_n}",
                gpu_level[c].len()
            );
        }

        if k > 0 {
            // Reduce each host channel from prev level into current.
            for c in 0..3 {
                let mut reduced = Vec::new();
                gausspyr_reduce_scalar(&host_level[c], prev_w, prev_h, &mut reduced);
                host_level[c] = reduced;
            }
        }

        let mut max_err = 0.0_f32;
        for c in 0..3 {
            for (a, b) in gpu_level[c].iter().zip(&host_level[c]) {
                let d = (a - b).abs();
                if d > max_err {
                    max_err = d;
                }
            }
        }
        assert!(
            max_err < 1e-3,
            "level {k} max-abs error vs host = {max_err}"
        );
        prev_w = expected_w;
        prev_h = expected_h;
    }
}

#[test]
fn compute_dkl_laplacian_pyramid_matches_host_scalar() {
    let client = Backend::client(&Default::default());
    let (w, h) = (16u32, 16u32);
    let n = (w * h) as usize;
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let mut srgb = Vec::with_capacity(n * 3);
    for i in 0..n {
        srgb.push((i % 251) as u8);
        srgb.push(((i * 7 + 13) % 251) as u8);
        srgb.push(((i * 19 + 41) % 251) as u8);
    }

    let gpu_bands = cvvdp
        .compute_dkl_laplacian_pyramid(&srgb)
        .expect("compute_dkl_laplacian_pyramid");

    let display = DisplayModel::STANDARD_4K;
    // Host reference: build DKL planes per channel, then call
    // laplacian_pyramid_dec_scalar on each channel separately.
    let mut planes: [Vec<f32>; 3] = [vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]];
    for i in 0..n {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            srgb[i * 3],
            srgb[i * 3 + 1],
            srgb[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        planes[0][i] = a;
        planes[1][i] = rg;
        planes[2][i] = vy;
    }

    let n_levels = gpu_bands.len();
    let mut host_bands_per_channel: [Vec<cvvdp_gpu::kernels::pyramid::Band>; 3] = [
        laplacian_pyramid_dec_scalar(&planes[0], w as usize, h as usize, n_levels),
        laplacian_pyramid_dec_scalar(&planes[1], w as usize, h as usize, n_levels),
        laplacian_pyramid_dec_scalar(&planes[2], w as usize, h as usize, n_levels),
    ];

    for k in 0..n_levels {
        for c in 0..3 {
            let host = std::mem::take(&mut host_bands_per_channel[c][k].data);
            assert_eq!(
                gpu_bands[k][c].len(),
                host.len(),
                "level {k} channel {c} size mismatch"
            );
            let max_err = gpu_bands[k][c]
                .iter()
                .zip(&host)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            // Loose tol — bands accumulate FMA + LUT-interp noise
            // across 3 stages (color, reduce, expand, subtract).
            assert!(
                max_err < 5e-3,
                "level {k} channel {c}: max-abs vs host scalar = {max_err}"
            );
        }
    }
}

#[test]
fn compute_dkl_csf_weighted_bands_matches_host() {
    let client = Backend::client(&Default::default());
    let (w, h) = (16u32, 16u32);
    let n = (w * h) as usize;
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let mut srgb = Vec::with_capacity(n * 3);
    for i in 0..n {
        srgb.push((i % 251) as u8);
        srgb.push(((i * 7 + 13) % 251) as u8);
        srgb.push(((i * 19 + 41) % 251) as u8);
    }

    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let l_bkg = 100.0_f32;

    let gpu = cvvdp
        .compute_dkl_csf_weighted_bands(&srgb, ppd, l_bkg)
        .expect("compute_dkl_csf_weighted_bands");

    // Host reference: same Laplacian path then per-channel × per-level
    // multiply by precomputed_band_weights.
    let display = DisplayModel::STANDARD_4K;
    let mut planes: [Vec<f32>; 3] = [vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]];
    for i in 0..n {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            srgb[i * 3],
            srgb[i * 3 + 1],
            srgb[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        planes[0][i] = a;
        planes[1][i] = rg;
        planes[2][i] = vy;
    }

    let n_levels = gpu.len();
    let host_bands: [Vec<cvvdp_gpu::kernels::pyramid::Band>; 3] = [
        laplacian_pyramid_dec_scalar(&planes[0], w as usize, h as usize, n_levels),
        laplacian_pyramid_dec_scalar(&planes[1], w as usize, h as usize, n_levels),
        laplacian_pyramid_dec_scalar(&planes[2], w as usize, h as usize, n_levels),
    ];
    let weights = precomputed_band_weights(ppd, w as usize, h as usize, l_bkg);

    for k in 0..n_levels {
        for c in 0..3 {
            let scale = weights[k][c];
            let host: Vec<f32> = host_bands[c][k].data.iter().map(|v| v * scale).collect();
            let max_err = gpu[k][c]
                .iter()
                .zip(&host)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            // Loose tol — weighted bands span a wider magnitude due
            // to the CSF scale (~1-100×) compounding with f32
            // multi-stage accumulated noise.
            assert!(
                max_err < 1e-1,
                "level {k} channel {c}: max-abs vs host weighted = {max_err}"
            );
        }
    }
}
