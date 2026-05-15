//! Integration test for `Cvvdp::compute_dkl_planes` — exercises the
//! upload + LUT-init + color-kernel path end-to-end through the
//! pipeline. Compares against the host scalar `srgb_byte_to_dkl_scalar`.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
use cvvdp_gpu::kernels::csf::precomputed_band_weights;
use cvvdp_gpu::kernels::pyramid::{
    gausspyr_reduce_scalar, laplacian_pyramid_dec_scalar, weber_contrast_pyr_dec_scalar,
};
use cvvdp_gpu::params::DisplayGeometry;
use cvvdp_gpu::params::{CvvdpParams, DisplayModel};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "hip", not(feature = "cuda"), not(feature = "wgpu")))]
type Backend = cubecl::hip::HipRuntime;

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
        for (c, gpu_plane) in gpu_level.iter().enumerate() {
            assert_eq!(
                gpu_plane.len(),
                expected_n,
                "level {k} channel {c}: got {} elements, expected {expected_n}",
                gpu_plane.len()
            );
        }

        if k > 0 {
            // Reduce each host channel from prev level into current.
            for plane in host_level.iter_mut() {
                let mut reduced = Vec::new();
                gausspyr_reduce_scalar(plane, prev_w, prev_h, &mut reduced);
                *plane = reduced;
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

#[test]
fn compute_dkl_weber_pyramid_matches_host_scalar() {
    // Build a deterministic sRGB pattern, run it through the GPU
    // Weber-contrast pyramid path, and compare bands + log_l_bkg
    // against host_scalar's weber_contrast_pyr_dec_scalar per channel.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let mut srgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let r = ((x * 8) % 256) as u8;
            let g = ((y * 8) % 256) as u8;
            let b = (((x + y) * 4) % 256) as u8;
            let i = (y * w as usize + x) * 3;
            srgb[i] = r;
            srgb[i + 1] = g;
            srgb[i + 2] = b;
        }
    }

    let (gpu_bands, gpu_log_l_bkg) = cvvdp.compute_dkl_weber_pyramid(&srgb).expect("gpu weber");

    // Host reference: replay color transform per pixel, then per
    // channel run weber_contrast_pyr_dec_scalar with the
    // achromatic-channel data as the L_bkg plane.
    let n_levels = gpu_bands.len();
    let n_px = (w * h) as usize;
    let mut planes: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
    let display = DisplayModel::STANDARD_4K;
    for (i, chunk) in srgb.chunks_exact(3).enumerate() {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            chunk[0],
            chunk[1],
            chunk[2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        planes[0][i] = a;
        planes[1][i] = rg;
        planes[2][i] = vy;
    }
    let host_per_ch = [
        weber_contrast_pyr_dec_scalar(&planes[0], &planes[0], w as usize, h as usize, n_levels),
        weber_contrast_pyr_dec_scalar(&planes[1], &planes[0], w as usize, h as usize, n_levels),
        weber_contrast_pyr_dec_scalar(&planes[2], &planes[0], w as usize, h as usize, n_levels),
    ];

    for k in 0..n_levels {
        // Compare bands per channel.
        for (c, gpu_plane) in gpu_bands[k].iter().enumerate() {
            let host_band = &host_per_ch[c].bands[k].data;
            assert_eq!(gpu_plane.len(), host_band.len(), "level {k} channel {c}");
            let max_err = gpu_plane
                .iter()
                .zip(host_band)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            // Tolerance: GPU uses single-call kernels, host uses the
            // unrolled scalar — both compute the same formula but
            // through different float accumulation orders.
            assert!(
                max_err < 5e-4,
                "weber band level {k} channel {c}: max-abs GPU vs host = {max_err}"
            );
        }
        // Compare log_l_bkg (taken from the achromatic channel).
        let host_log = &host_per_ch[0].log_l_bkg[k];
        assert_eq!(
            gpu_log_l_bkg[k].len(),
            host_log.len(),
            "log_l_bkg level {k}"
        );
        let max_err = gpu_log_l_bkg[k]
            .iter()
            .zip(host_log)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 1e-4,
            "weber log_l_bkg level {k}: max-abs GPU vs host = {max_err}"
        );
    }
}

#[test]
fn compute_dkl_t_p_bands_matches_host_scalar() {
    // GPU Weber pyramid + per-pixel CSF apply → T_p bands, compared
    // against the same formula computed entirely in host scalar
    // (sensitivity_corrected_scalar per pixel × CH_GAIN × band_mul).
    use cvvdp_gpu::kernels::csf::{CsfChannel, sensitivity_corrected_scalar};
    use cvvdp_gpu::kernels::masking::CH_GAIN;
    use cvvdp_gpu::kernels::pyramid::band_frequencies;

    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let mut srgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let r = ((x * 8) % 256) as u8;
            let g = ((y * 8) % 256) as u8;
            let b = (((x + y) * 4) % 256) as u8;
            let i = (y * w as usize + x) * 3;
            srgb[i] = r;
            srgb[i + 1] = g;
            srgb[i + 2] = b;
        }
    }

    // GPU T_p.
    let t_p_gpu = cvvdp
        .compute_dkl_t_p_bands(&srgb, ppd)
        .expect("compute_dkl_t_p_bands");

    // Host reference: build Weber pyramid + per-pixel CSF apply by hand.
    let n_px = (w * h) as usize;
    let display = DisplayModel::STANDARD_4K;
    let mut planes: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
    for (i, chunk) in srgb.chunks_exact(3).enumerate() {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            chunk[0],
            chunk[1],
            chunk[2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        planes[0][i] = a;
        planes[1][i] = rg;
        planes[2][i] = vy;
    }

    let n_levels = t_p_gpu.len();
    let host_per_ch = [
        cvvdp_gpu::kernels::pyramid::weber_contrast_pyr_dec_scalar(
            &planes[0], &planes[0], w as usize, h as usize, n_levels,
        ),
        cvvdp_gpu::kernels::pyramid::weber_contrast_pyr_dec_scalar(
            &planes[1], &planes[0], w as usize, h as usize, n_levels,
        ),
        cvvdp_gpu::kernels::pyramid::weber_contrast_pyr_dec_scalar(
            &planes[2], &planes[0], w as usize, h as usize, n_levels,
        ),
    ];

    let freqs = band_frequencies(ppd, w as usize, h as usize);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

    for k in 0..n_levels {
        let is_first = k == 0;
        let is_baseband = k == n_levels - 1;
        let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
        let bw = host_per_ch[0].bands[k].w;
        let bh = host_per_ch[0].bands[k].h;
        let n_band = bw * bh;
        let log_l_bkg_band = &host_per_ch[0].log_l_bkg[k];

        for c in 0..3 {
            let weber_c = &host_per_ch[c].bands[k].data;
            let mut host_t_p = vec![0.0_f32; n_band];
            for i in 0..n_band {
                let s = sensitivity_corrected_scalar(freqs[k], log_l_bkg_band[i], channels[c]);
                let ch_gain_eff = if is_baseband {
                    1.0
                } else {
                    band_mul * CH_GAIN[c]
                };
                host_t_p[i] = weber_c[i] * s * ch_gain_eff;
            }
            assert_eq!(
                t_p_gpu[k][c].len(),
                host_t_p.len(),
                "level {k} channel {c} size mismatch"
            );
            let max_err = t_p_gpu[k][c]
                .iter()
                .zip(&host_t_p)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            // T_p magnitudes span 10-100s once band_mul * CH_GAIN *
            // CSF S (typ. 10-300) multiply the Weber band. The GPU
            // path uses uniform-axis-step arithmetic for the
            // log_L_bkg interp while the host uses binary-search
            // interp1_clamped — bit-identical for interior values
            // but accumulating f32 noise through the chain. Use a
            // relative tolerance to absorb the magnitude variation.
            let max_t_p_mag = host_t_p
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max)
                .max(1.0);
            let rel_err = max_err / max_t_p_mag;
            assert!(
                rel_err < 5e-3,
                "T_p level {k} channel {c}: max-abs GPU vs host = {max_err}, \
                 relative = {rel_err:.4e}, max-abs |host T_p| = {max_t_p_mag}"
            );
        }
    }
}

#[test]
fn compute_dkl_d_bands_matches_host_scalar() {
    // GPU pipeline through T_p + host-scalar mult_mutual_band → D
    // bands, compared against the same composition in pure host
    // scalar (sensitivity_corrected_scalar + CH_GAIN + band_mul +
    // mult_mutual_band per non-baseband band; |T_p_dis - T_p_ref|
    // for the baseband).
    use cvvdp_gpu::kernels::csf::{CsfChannel, sensitivity_corrected_scalar};
    use cvvdp_gpu::kernels::masking::{CH_GAIN, mult_mutual_band};
    use cvvdp_gpu::kernels::pyramid::{band_frequencies, weber_contrast_pyr_dec_scalar};

    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // Deterministic ref + dist patterns (dist ≠ ref so masking step is exercised).
    let mut ref_srgb = vec![0u8; (w * h * 3) as usize];
    let mut dist_srgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let r = ((x * 8) % 256) as u8;
            let g = ((y * 8) % 256) as u8;
            let b = (((x + y) * 4) % 256) as u8;
            let i = (y * w as usize + x) * 3;
            ref_srgb[i] = r;
            ref_srgb[i + 1] = g;
            ref_srgb[i + 2] = b;
            dist_srgb[i] = r.saturating_sub(8);
            dist_srgb[i + 1] = g.saturating_sub(4);
            dist_srgb[i + 2] = b.saturating_add(12);
        }
    }

    let gpu_d = cvvdp
        .compute_dkl_d_bands(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_d_bands");

    // Host reference: replay color → Weber pyramid → per-pixel CSF
    // apply → mult_mutual_band (non-baseband) / |Δ| (baseband).
    let n_levels = gpu_d.len();
    let n_px = (w * h) as usize;
    let display = DisplayModel::STANDARD_4K;
    let mut ref_planes: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
    let mut dis_planes: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
    for i in 0..n_px {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            ref_srgb[i * 3],
            ref_srgb[i * 3 + 1],
            ref_srgb[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        ref_planes[0][i] = a;
        ref_planes[1][i] = rg;
        ref_planes[2][i] = vy;
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            dist_srgb[i * 3],
            dist_srgb[i * 3 + 1],
            dist_srgb[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        dis_planes[0][i] = a;
        dis_planes[1][i] = rg;
        dis_planes[2][i] = vy;
    }

    let ref_weber = [
        weber_contrast_pyr_dec_scalar(
            &ref_planes[0],
            &ref_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
        weber_contrast_pyr_dec_scalar(
            &ref_planes[1],
            &ref_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
        weber_contrast_pyr_dec_scalar(
            &ref_planes[2],
            &ref_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
    ];
    let dis_weber = [
        weber_contrast_pyr_dec_scalar(
            &dis_planes[0],
            &dis_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
        weber_contrast_pyr_dec_scalar(
            &dis_planes[1],
            &dis_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
        weber_contrast_pyr_dec_scalar(
            &dis_planes[2],
            &dis_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
    ];
    let freqs = band_frequencies(ppd, w as usize, h as usize);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

    for k in 0..n_levels {
        let is_first = k == 0;
        let is_baseband = k == n_levels - 1;
        let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
        let bw = ref_weber[0].bands[k].w;
        let bh = ref_weber[0].bands[k].h;
        let n_band = bw * bh;
        let log_l_bkg_band = &ref_weber[0].log_l_bkg[k];

        let mut t_p_dis: [Vec<f32>; 3] = [vec![0.0; n_band], vec![0.0; n_band], vec![0.0; n_band]];
        let mut t_p_ref: [Vec<f32>; 3] = [vec![0.0; n_band], vec![0.0; n_band], vec![0.0; n_band]];
        for c in 0..3 {
            for i in 0..n_band {
                let s = sensitivity_corrected_scalar(freqs[k], log_l_bkg_band[i], channels[c]);
                let ch_gain_eff = if is_baseband {
                    1.0
                } else {
                    band_mul * CH_GAIN[c]
                };
                t_p_dis[c][i] = dis_weber[c].bands[k].data[i] * s * ch_gain_eff;
                t_p_ref[c][i] = ref_weber[c].bands[k].data[i] * s * ch_gain_eff;
            }
        }

        let host_d = if is_baseband {
            let mut planes: [Vec<f32>; 3] =
                [vec![0.0; n_band], vec![0.0; n_band], vec![0.0; n_band]];
            for c in 0..3 {
                for i in 0..n_band {
                    planes[c][i] = (t_p_dis[c][i] - t_p_ref[c][i]).abs();
                }
            }
            planes
        } else {
            mult_mutual_band(&t_p_dis, &t_p_ref, bw, bh)
        };

        for c in 0..3 {
            assert_eq!(
                gpu_d[k][c].len(),
                host_d[c].len(),
                "level {k} channel {c} size mismatch"
            );
            let max_err = gpu_d[k][c]
                .iter()
                .zip(&host_d[c])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let max_d_mag = host_d[c]
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max)
                .max(1.0);
            let rel_err = max_err / max_d_mag;
            // D magnitudes after the soft clamp are in 1-100 range
            // for typical T_p deltas; rel tolerance absorbs the
            // accumulated f32 noise across color→weber→CSF→masking.
            assert!(
                rel_err < 1e-2,
                "D level {k} channel {c}: max-abs GPU vs host = {max_err}, \
                 relative = {rel_err:.4e}, max |host D| = {max_d_mag}"
            );
        }
    }
}

#[test]
fn compute_dkl_jod_matches_host_scalar() {
    // GPU-composed JOD (color + Weber pyramid + per-pixel CSF on
    // GPU; masking + pool + final fold on host) vs the all-host
    // host_scalar::predict_jod_still_3ch. Both should agree within
    // f32 accumulation noise.
    use cvvdp_gpu::host_scalar::predict_jod_still_3ch;

    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let mut ref_srgb = vec![0u8; (w * h * 3) as usize];
    let mut dist_srgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let r = ((x * 8) % 256) as u8;
            let g = ((y * 8) % 256) as u8;
            let b = (((x + y) * 4) % 256) as u8;
            let i = (y * w as usize + x) * 3;
            ref_srgb[i] = r;
            ref_srgb[i + 1] = g;
            ref_srgb[i + 2] = b;
            dist_srgb[i] = r.saturating_sub(8);
            dist_srgb[i + 1] = g.saturating_sub(4);
            dist_srgb[i + 2] = b.saturating_add(12);
        }
    }

    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");
    let host_jod =
        predict_jod_still_3ch(&ref_srgb, &dist_srgb, w as usize, h as usize, display, ppd);
    let diff = (gpu_jod - host_jod).abs();
    eprintln!("compute_dkl_jod = {gpu_jod:.6}, host_scalar = {host_jod:.6}, |diff| = {diff:.6}");
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    // f32 accumulation through 9 stages (color → reduce ×5 →
    // expand ×5 → subtract → weber contrast → per-pixel CSF interp
    // → mult_mutual + soft clamp → spatial-pool → 3-stage
    // Minkowski → met2jod) propagates a ~1% per-band Q delta into
    // a JOD shift whose magnitude scales with the slope of met2jod
    // at the operating point. The shadow_jod test pins
    // Cvvdp::score (the all-host path) to pycvvdp's v1 manifest
    // within 0.006 JOD; this test only measures the GPU-composed
    // path's drift from the all-host reference. Tolerance stays
    // loose until the masking + pool GPU kernels get wired into
    // compute_dkl_jod and the accumulated drift collapses.
    assert!(
        diff < 0.5,
        "GPU JOD {gpu_jod:.6} diverges from host scalar {host_jod:.6} by {diff:.6}"
    );
}

#[test]
fn compute_dkl_jod_with_warm_ref_matches_unwarm_path() {
    // Batch-scoring fast path: warm_reference dispatches the REF
    // weber pyramid once and caches the GPU state; subsequent
    // compute_dkl_jod_with_warm_ref calls skip REF weber. Output
    // JOD must match compute_dkl_jod(ref, dist, ppd) byte-for-byte
    // (same kernels, same data, just dispatched in two phases).
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let mut ref_srgb = vec![0u8; (w * h * 3) as usize];
    let mut dist_a = vec![0u8; (w * h * 3) as usize];
    let mut dist_b = vec![0u8; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let r = ((x * 8) % 256) as u8;
            let g = ((y * 8) % 256) as u8;
            let b = (((x + y) * 4) % 256) as u8;
            let i = (y * w as usize + x) * 3;
            ref_srgb[i] = r;
            ref_srgb[i + 1] = g;
            ref_srgb[i + 2] = b;
            dist_a[i] = r.saturating_sub(8);
            dist_a[i + 1] = g.saturating_sub(4);
            dist_a[i + 2] = b.saturating_add(12);
            dist_b[i] = r.saturating_add(5);
            dist_b[i + 1] = g.saturating_sub(10);
            dist_b[i + 2] = b.saturating_sub(3);
        }
    }

    // Reference values via the non-warm path. Each call rebuilds
    // both sides — exactly what the warm-ref fast path skips.
    let jod_a_unwarm = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_a, ppd)
        .expect("compute_dkl_jod a");
    let jod_b_unwarm = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_b, ppd)
        .expect("compute_dkl_jod b");

    // Warm-ref path: REF dispatched once, two DIST candidates.
    cvvdp.warm_reference(&ref_srgb).expect("warm_reference");
    let jod_a_warm = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_a, ppd)
        .expect("compute_dkl_jod_with_warm_ref a");
    let jod_b_warm = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_b, ppd)
        .expect("compute_dkl_jod_with_warm_ref b");

    // Same kernels, same data → same output, modulo f32 rounding
    // from any difference in scheduling order. 1e-5 absolute tol
    // catches any real divergence; numerical fuzz at the JOD scale
    // doesn't surface.
    assert!(
        (jod_a_warm - jod_a_unwarm).abs() < 1e-5,
        "warm vs unwarm JOD diverged for dist_a: warm={jod_a_warm:.6}, unwarm={jod_a_unwarm:.6}"
    );
    assert!(
        (jod_b_warm - jod_b_unwarm).abs() < 1e-5,
        "warm vs unwarm JOD diverged for dist_b: warm={jod_b_warm:.6}, unwarm={jod_b_unwarm:.6}"
    );

    // After two warm-ref scores the state should still be warm —
    // verify by scoring dist_a again and getting the same answer.
    let jod_a_warm2 = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_a, ppd)
        .expect("compute_dkl_jod_with_warm_ref a (second)");
    assert!(
        (jod_a_warm2 - jod_a_warm).abs() < 1e-5,
        "repeat warm-ref score diverged: first={jod_a_warm:.6}, second={jod_a_warm2:.6}"
    );

    // An intervening non-warm call must invalidate the warm state.
    let _ = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_a, ppd)
        .expect("intervening compute_dkl_jod");
    let err = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_a, ppd)
        .expect_err("warm state should be invalidated");
    match err {
        cvvdp_gpu::Error::NoWarmReference => {}
        other => panic!("expected NoWarmReference, got {other:?}"),
    }
}
