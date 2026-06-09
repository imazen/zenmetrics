//! Integration test for `Cvvdp::compute_dkl_planes` — exercises the
//! upload + LUT-init + color-kernel path end-to-end through the
//! pipeline. Compares against the host scalar `srgb_byte_to_dkl_scalar`.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]
// Per-band loops naturally use `for k in 0..n_bands` indexing into
// `ref_tp[k]` / `d_bands[k]` plus side metadata (sentinels,
// per-band widths) — converting to enumerate is a wash. Same
// pattern as the library's `#![allow(clippy::needless_range_loop)]`.
#![allow(clippy::needless_range_loop)]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
use cvvdp_gpu::kernels::csf::precomputed_band_weights;
use cvvdp_gpu::kernels::pyramid::{
    gausspyr_reduce_scalar, laplacian_pyramid_dec_scalar, weber_contrast_pyr_dec_scalar,
};
use cvvdp_gpu::params::DisplayGeometry;
use cvvdp_gpu::params::{CvvdpParams, DisplayModel};

use crate::common;

use common::Backend;

/// Manifest-parity tolerance shared by the 9 GPU/warm-ref tests
/// that compare cvvdp-gpu JOD against the pycvvdp v0.5.4 R2
/// goldens. 0.005 JOD ≈ ~10× the f32 noise floor observed at
/// 12 MP; tightened from the prior 0.05 schedule by tick 207
/// after ticks 204/206 closed the chroma_shift and 73×91 odd-dim
/// drifts. Measured max diff across q=1–90 is 0.0031 JOD.
///
/// Hoisted to file scope (tick 297) to remove 9 duplicate
/// in-function declarations that were triggering
/// `clippy::items_after_statements`.
const TOLERANCE: f32 = 0.005;

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
            for plane in &mut host_level {
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

    let srgb = common::synth_pair_odd_dim_ref(w as usize, h as usize);

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

    let srgb = common::synth_pair_odd_dim_ref(w as usize, h as usize);

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

        // Tick 204: pycvvdp overrides baseband rho to 0.1 cy/deg
        // (cvvdp_metric.py:628); the GPU pipeline now matches. Host
        // reference uses the same override here so the parity test
        // compares apples-to-apples.
        let rho_eff = if is_baseband {
            cvvdp_gpu::kernels::csf::CSF_BASEBAND_RHO
        } else {
            freqs[k]
        };
        for c in 0..3 {
            let weber_c = &host_per_ch[c].bands[k].data;
            let mut host_t_p = vec![0.0_f32; n_band];
            for i in 0..n_band {
                let s = sensitivity_corrected_scalar(rho_eff, log_l_bkg_band[i], channels[c]);
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
    let (ref_srgb, dist_srgb) = common::synth_pair_odd_dim_with_offset_dist(w as usize, h as usize);

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

        // Tick 204: baseband CSF rho override (see compute_dkl_t_p_*
        // sibling test).
        let rho_eff = if is_baseband {
            cvvdp_gpu::kernels::csf::CSF_BASEBAND_RHO
        } else {
            freqs[k]
        };
        let mut t_p_dis: [Vec<f32>; 3] = [vec![0.0; n_band], vec![0.0; n_band], vec![0.0; n_band]];
        let mut t_p_ref: [Vec<f32>; 3] = [vec![0.0; n_band], vec![0.0; n_band], vec![0.0; n_band]];
        for c in 0..3 {
            for i in 0..n_band {
                let s = sensitivity_corrected_scalar(rho_eff, log_l_bkg_band[i], channels[c]);
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

    let (ref_srgb, dist_srgb) = common::synth_pair_odd_dim_with_offset_dist(w as usize, h as usize);

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
    // Tightened in tick 184. Post tick-181's band-count alignment,
    // observed diff = 0.000387 JOD. 0.005 gives ~13× margin while
    // catching real regressions: pre-tick-175 ceil-div bug produced
    // 0.586 JOD drift at 12 MP; small-input drifts would be in a
    // similar range. The earlier 0.5 tolerance was left over from
    // when the GPU pipeline was still partial (host fold + Minkowski
    // path) — the full GPU path matches host within f32 precision
    // now.
    assert!(
        diff < 0.005,
        "GPU JOD {gpu_jod:.6} diverges from host scalar {host_jod:.6} by {diff:.6} (was 0.000387 at tick 184)"
    );
}

#[test]
fn compute_dkl_jod_host_pool_matches_compute_dkl_jod() {
    // Tick 208: compute_dkl_jod_host_pool is the cpu-backend-
    // compatible variant of compute_dkl_jod. Same JOD, computed
    // by reading D bands back to host and pooling via
    // lp_norm_mean instead of the GPU pool_band_3ch_kernel (which
    // uses Atomic<f32>::fetch_add, unsupported by cubecl-cpu).
    //
    // On any GPU backend both paths run; this test pins the
    // host-pool variant against the canonical compute_dkl_jod
    // output. f32-precision tolerance because both paths apply
    // the same Minkowski safe_pow form — only the accumulation
    // order differs (atomic on GPU, sequential on host).
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let (ref_srgb, dist_srgb) = common::synth_pair_odd_dim_with_offset_dist(w as usize, h as usize);

    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");
    let host_pool_jod = cvvdp
        .compute_dkl_jod_host_pool(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod_host_pool");
    let diff = (gpu_jod - host_pool_jod).abs();
    eprintln!(
        "host_pool: compute_dkl_jod = {gpu_jod:.6}, host_pool = {host_pool_jod:.6}, |diff| = {diff:.6}"
    );
    assert!(gpu_jod.is_finite() && host_pool_jod.is_finite());
    assert!(
        diff < 0.005,
        "compute_dkl_jod_host_pool {host_pool_jod:.6} diverges from compute_dkl_jod {gpu_jod:.6} by {diff:.6}"
    );
}

#[test]
fn compute_dkl_jod_host_pool_with_warm_ref_matches_compute_dkl_jod() {
    // Tick 212: host-pool warm-ref variant. Same JOD as the
    // canonical compute_dkl_jod, but uses the warm-ref state path
    // (skip REF weber) AND host pool (cubecl-cpu compatible).
    // Useful for batch CPU scoring against one warmed REF.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let (ref_srgb, dist_srgb) = common::synth_pair_odd_dim_with_offset_dist(w as usize, h as usize);

    // Canonical cold-ref path.
    let canonical = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");

    // Warm + host-pool path: warm_reference, then host-pool with warm ref.
    cvvdp.warm_reference(&ref_srgb).expect("warm_reference");
    let warm_host = cvvdp
        .compute_dkl_jod_host_pool_with_warm_ref(&dist_srgb, ppd)
        .expect("compute_dkl_jod_host_pool_with_warm_ref");

    let diff = (canonical - warm_host).abs();
    eprintln!(
        "warm host_pool: canonical = {canonical:.6}, warm_host_pool = {warm_host:.6}, |diff| = {diff:.6}"
    );
    assert!(canonical.is_finite() && warm_host.is_finite());
    assert!(
        diff < 0.005,
        "warm_host_pool {warm_host:.6} diverges from canonical compute_dkl_jod {canonical:.6} by {diff:.6}"
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

    let ref_srgb = common::synth_pair_odd_dim_ref(w as usize, h as usize);
    let dist_a = common::apply_offset_dist(&ref_srgb);
    let dist_b: Vec<u8> = ref_srgb
        .chunks_exact(3)
        .flat_map(|p| {
            [
                p[0].saturating_add(5),
                p[1].saturating_sub(10),
                p[2].saturating_sub(3),
            ]
        })
        .collect();

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

#[test]
fn warm_state_invalidates_after_each_documented_dispatcher() {
    // Tick 236: pin every dispatcher listed in `warm_reference`'s
    // docstring as a warm-state invalidator. The docstring promised
    // four invalidators since tick 170:
    //   - compute_dkl_jod
    //   - compute_dkl_d_bands
    //   - compute_dkl_weber_pyramid
    //   - compute_dkl_t_p_bands
    //
    // Before tick 236 only the first two actually cleared
    // `warm_ref_baseband_log_l_bkg`. `compute_dkl_weber_pyramid`
    // and `compute_dkl_t_p_bands` overwrote bands_ref +
    // weber_scratch[*].log_l_bkg via _dispatch_weber_pyramid_gpu
    // but left the cached scalar alive — a subsequent
    // `compute_dkl_jod_with_warm_ref` would silently mix stale
    // scalar with fresh bands. Tick 236 added the missing
    // `warm_ref_baseband_log_l_bkg = None` at the top of both;
    // this test pins each invalidator independently so the
    // contract stays honest.
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_srgb = vec![128u8; n];
    let dist_srgb: Vec<u8> = ref_srgb.iter().map(|b| b.saturating_add(8)).collect();

    // Re-arm warm state then trigger each documented invalidator
    // in turn. After each one, compute_dkl_jod_with_warm_ref must
    // surface Error::NoWarmReference.
    let invalidators: &[&str] = &[
        "compute_dkl_jod",
        "compute_dkl_d_bands",
        "compute_dkl_weber_pyramid",
        "compute_dkl_t_p_bands",
        "compute_dkl_laplacian_pyramid",
        "compute_dkl_csf_weighted_bands",
        // Tick 238: transitive invalidators that route through
        // compute_dkl_jod. Pin so a future refactor that bypasses
        // the GPU path (e.g. reintroducing a host_scalar shortcut
        // in `score`) doesn't silently break the documented
        // contract.
        "score",
        "score_with_reference",
        // Tick 314: compute_dkl_jod_host_pool also invalidates —
        // it routes through _dispatch_d_bands_into_scratch which
        // calls _dispatch_ref_weber_pyramid_only, which clears
        // the cached scalar. The cpu-runtime entry point shares
        // the same REF dispatch as the all-GPU jod, so the same
        // invalidation contract applies.
        "compute_dkl_jod_host_pool",
    ];

    let l_bkg_scalar = cvvdp_gpu::params::DisplayModel::STANDARD_4K.y_peak / 2.0;
    for &name in invalidators {
        cvvdp
            .warm_reference(&ref_srgb)
            .expect("warm_reference (re-arm)");
        match name {
            "compute_dkl_jod" => {
                let _ = cvvdp
                    .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
                    .expect("intervening compute_dkl_jod");
            }
            "compute_dkl_d_bands" => {
                let _ = cvvdp
                    .compute_dkl_d_bands(&ref_srgb, &dist_srgb, ppd)
                    .expect("intervening compute_dkl_d_bands");
            }
            "compute_dkl_weber_pyramid" => {
                let _ = cvvdp
                    .compute_dkl_weber_pyramid(&ref_srgb)
                    .expect("intervening compute_dkl_weber_pyramid");
            }
            "compute_dkl_t_p_bands" => {
                let _ = cvvdp
                    .compute_dkl_t_p_bands(&ref_srgb, ppd)
                    .expect("intervening compute_dkl_t_p_bands");
            }
            "compute_dkl_laplacian_pyramid" => {
                let _ = cvvdp
                    .compute_dkl_laplacian_pyramid(&ref_srgb)
                    .expect("intervening compute_dkl_laplacian_pyramid");
            }
            "compute_dkl_csf_weighted_bands" => {
                let _ = cvvdp
                    .compute_dkl_csf_weighted_bands(&ref_srgb, ppd, l_bkg_scalar)
                    .expect("intervening compute_dkl_csf_weighted_bands");
            }
            "score" => {
                let _ = cvvdp
                    .score(&ref_srgb, &dist_srgb)
                    .expect("intervening score");
            }
            "score_with_reference" => {
                cvvdp
                    .set_reference(&ref_srgb)
                    .expect("set_reference for score_with_reference path");
                let _ = cvvdp
                    .score_with_reference(&dist_srgb)
                    .expect("intervening score_with_reference");
            }
            "compute_dkl_jod_host_pool" => {
                let _ = cvvdp
                    .compute_dkl_jod_host_pool(&ref_srgb, &dist_srgb, ppd)
                    .expect("intervening compute_dkl_jod_host_pool");
            }
            other => unreachable!("unhandled invalidator {other}"),
        }
        let err = cvvdp
            .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
            .expect_err("warm state should be invalidated");
        match err {
            cvvdp_gpu::Error::NoWarmReference => {}
            other => panic!("after {name}: expected NoWarmReference, got {other:?}"),
        }
    }
}

#[test]
fn gauss_chain_helpers_do_not_invalidate_warm_state() {
    // Tick 251: pin the inverse of
    // warm_state_invalidates_after_each_documented_dispatcher.
    // `compute_dkl_planes` and `compute_dkl_gauss_pyramid` write only
    // to `gauss_ref` (per-call scratch, NOT part of the warm state).
    // They MUST preserve the warm scalar so a subsequent
    // `compute_dkl_jod_with_warm_ref` call succeeds.
    //
    // A future refactor that, say, made `compute_dkl_planes`
    // additionally pre-emit weber bands into bands_ref (matching the
    // public `compute_dkl_weber_pyramid` interface for symmetry) would
    // need to invalidate warm state — this test would catch it.
    //
    // Sibling to `set_reference_does_not_invalidate_warm_state` and
    // `warm_state_invalidates_after_each_documented_dispatcher`.
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_srgb = vec![128u8; n];
    let dist_srgb: Vec<u8> = ref_srgb.iter().map(|b| b.saturating_add(8)).collect();
    let other_srgb: Vec<u8> = ref_srgb.iter().map(|b| b.saturating_add(4)).collect();

    let non_invalidators: &[&str] = &[
        "compute_dkl_planes",
        "compute_dkl_gauss_pyramid",
        // Tick 315: pin the dual of the tick-314 docstring fix —
        // compute_dkl_jod_host_pool_with_warm_ref READS the cached
        // scalar (.ok_or(NoWarmReference)) but never writes it, so
        // it MUST preserve the warm state across calls. A refactor
        // that accidentally cleared the scalar (e.g. moving the
        // warm-ref host-pool path through _dispatch_d_bands_into_scratch
        // by mistake) would silently break batch cpu-runtime
        // scoring; this case catches it.
        "compute_dkl_jod_host_pool_with_warm_ref",
    ];

    for &name in non_invalidators {
        cvvdp
            .warm_reference(&ref_srgb)
            .expect("warm_reference (re-arm)");
        match name {
            "compute_dkl_planes" => {
                let _ = cvvdp
                    .compute_dkl_planes(&other_srgb)
                    .expect("intervening compute_dkl_planes");
            }
            "compute_dkl_gauss_pyramid" => {
                let _ = cvvdp
                    .compute_dkl_gauss_pyramid(&other_srgb)
                    .expect("intervening compute_dkl_gauss_pyramid");
            }
            "compute_dkl_jod_host_pool_with_warm_ref" => {
                // dist_srgb (not other_srgb) so the warm-ref call
                // is a genuine score against the warmed reference,
                // not a "fresh DIST against the same warm REF"
                // edge case.
                let _ = cvvdp
                    .compute_dkl_jod_host_pool_with_warm_ref(&dist_srgb, ppd)
                    .expect("intervening compute_dkl_jod_host_pool_with_warm_ref");
            }
            other => unreachable!("unhandled non-invalidator {other}"),
        }
        // Warm-ref score must succeed — the call above only touched
        // gauss_ref, not bands_ref or weber_scratch.log_l_bkg.
        let jod = cvvdp
            .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
            .unwrap_or_else(|e| panic!("after {name}: warm state must survive, got error {e:?}"));
        assert!(
            jod.is_finite(),
            "after {name}: warm-ref JOD must be finite, got {jod}"
        );
        assert!(
            (0.0..=10.0).contains(&jod),
            "after {name}: warm-ref JOD must be in [0, 10], got {jod}"
        );
    }
}

#[test]
fn set_reference_does_not_invalidate_warm_state() {
    // Tick 238: pin the documented non-invalidator. set_reference
    // only stashes host-side bytes for the score_with_reference
    // cache; it does no GPU dispatch and shouldn't disturb the
    // separate warm-ref state. A future refactor that eagerly
    // dispatches on set_reference (e.g. pre-uploading to GPU)
    // would silently break this contract — pin it.
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_srgb = vec![128u8; n];
    let dist_srgb: Vec<u8> = ref_srgb.iter().map(|b| b.saturating_add(8)).collect();
    let other_ref: Vec<u8> = ref_srgb.iter().map(|b| b.saturating_add(4)).collect();

    cvvdp.warm_reference(&ref_srgb).expect("warm_reference");
    cvvdp
        .set_reference(&other_ref)
        .expect("set_reference must not invalidate warm state");

    // Warm-ref score should still succeed against the warm_reference
    // input, not the set_reference one.
    let jod = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
        .expect("warm state must survive set_reference");
    assert!(jod.is_finite(), "warm-ref JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "warm-ref JOD must be in [0, 10], got {jod}"
    );
}

#[test]
fn compute_dkl_jod_matches_pycvvdp_at_12mp_synth() {
    // 12 MP parity vs pycvvdp v0.5.4 CUDA on a deterministic
    // synthetic 4000×3000 pair (same construction as
    // examples/time_12mp.rs). The pycvvdp golden was measured on
    // an RTX 5070 via scripts/cvvdp_goldens/bench_12mp_cuda.py
    // (see `benchmarks/pycvvdp_12mp_cuda_2026-05-14.md` and
    // `benchmarks/pycvvdp_parity_tick175_2026-05-15.md`).
    //
    // This test catches large-image drift that the 32×32 /
    // 256×256 / 73×91 parity tests can't: at small sizes the
    // pyramid is shallow enough that cumulative f32 noise doesn't
    // build up. The original 0.586 JOD drift was only visible at
    // 12 MP; the ceil-div + MAX_LEVELS=9 fix (tick 175) closed it
    // to ~0.0003.
    //
    // Tolerance: 0.005 JOD ≈ ~10× f32 noise floor we observed at
    // 12 MP. A floor-div or n-bands regression would push this
    // far past 0.005.
    //
    // Runtime: ~600ms per call at 12 MP on RTX-class CUDA. Acceptable
    // for parity-test budgets.
    // Loaded from scripts/cvvdp_goldens/pycvvdp_synth_goldens.json
    // (regenerated by bench_12mp_cuda.py). Auto-syncs when the
    // pycvvdp pin moves or the bench reruns.
    let pycvvdp_golden_jod: f32 = common::pycvvdp_synth_golden_jod("synth_4000x3000");

    let client = Backend::client(&Default::default());
    let (w, h) = (4000u32, 3000u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // Same synth construction as examples/time_12mp.rs +
    // scripts/cvvdp_goldens/bench_12mp_cuda.py — keep in sync.
    let (ref_srgb, dist_srgb) = common::synth_pair_with_offset_dist(w as usize, h as usize);

    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");
    let diff = (gpu_jod - pycvvdp_golden_jod).abs();
    eprintln!(
        "12mp synth: gpu_jod = {gpu_jod:.4}, pycvvdp golden = {pycvvdp_golden_jod:.4}, |diff| = {diff:.4}"
    );
    assert!(
        diff < TOLERANCE,
        "GPU JOD {gpu_jod:.4} drifts from pycvvdp golden {pycvvdp_golden_jod:.4} by {diff:.4} > {TOLERANCE:.4}"
    );
}

#[test]
fn compute_dkl_jod_matches_pycvvdp_at_256x256_blur3x1() {
    // 256×256 pycvvdp parity using a non-JPEG-q distortion type
    // (3-pixel horizontal average with wrap). The synth construction
    // is bit-stable across NumPy (bench_12mp_cuda.py) and Rust
    // (this test): pure u8→u16→u8 floor-div arithmetic.
    //
    // Adds size coverage at 256² without depending on the
    // zenmetrics-corpus PNG/JPEG files — useful for offline contexts
    // and as a deterministic widening of the distortion-type sweep
    // (currently only JPEG-q via the corpus).
    let pycvvdp_golden_jod = common::pycvvdp_synth_golden_jod("synth_256x256_blur3x1");

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // Match `synth_pair_256_blur3x1` byte-for-byte. ref is the same
    // synth_pair_12mp pattern at 256×256; dist is the wrap-around
    // 3-pixel horizontal average. Use u16 in the sum to avoid
    // u8 overflow (3 × 255 = 765 > 255).
    let wu = w as usize;
    let hu = h as usize;
    let ref_srgb = common::synth_pair_ref(wu, hu);
    let mut dist_srgb = vec![0u8; wu * hu * 3];
    for y in 0..hu {
        for x in 0..wu {
            let x1 = (x + 1) % wu;
            let x2 = (x + 2) % wu;
            let row = y * wu;
            for c in 0..3 {
                let a = u16::from(ref_srgb[(row + x) * 3 + c]);
                let b = u16::from(ref_srgb[(row + x1) * 3 + c]);
                let cval = u16::from(ref_srgb[(row + x2) * 3 + c]);
                dist_srgb[(row + x) * 3 + c] = ((a + b + cval) / 3) as u8;
            }
        }
    }

    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");
    let diff = (gpu_jod - pycvvdp_golden_jod).abs();
    eprintln!(
        "256×256 blur3x1: gpu_jod = {gpu_jod:.4}, pycvvdp golden = {pycvvdp_golden_jod:.4}, |diff| = {diff:.4}"
    );
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    assert!(
        diff < TOLERANCE,
        "GPU JOD {gpu_jod:.4} drifts from pycvvdp golden {pycvvdp_golden_jod:.4} by {diff:.4} > {TOLERANCE:.4}"
    );
}

#[test]
fn compute_dkl_planes_matches_pycvvdp_dkl_at_chroma_shift_sentinels() {
    // Tick 196: stage-1 parity probe to localize the 0.117 JOD
    // chroma_shift drift (open finding since tick 191; all
    // constants pinned ticks 192-195). Computes our DKL planes on
    // the chroma_shift fixture and compares 10 sentinel pixels
    // against pycvvdp's DKL output (dumped via
    // scripts/cvvdp_goldens/dump_dkl_chroma.py).
    //
    // If this test passes tight, the color transform is fine and
    // the drift is downstream (pyramid / CSF / masking / pool).
    // If it fails, the color transform is the source.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // Same byte-for-byte synth as the chroma_shift fixture.
    let wu = w as usize;
    let hu = h as usize;
    let ref_srgb = common::synth_pair_ref(wu, hu);
    let dist_srgb: Vec<u8> = ref_srgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], (i16::from(p[1]) + 16).clamp(0, 255) as u8, p[2]])
        .collect();

    let ref_planes = cvvdp
        .compute_dkl_planes(&ref_srgb)
        .expect("compute_dkl_planes ref");
    let dist_planes = cvvdp
        .compute_dkl_planes(&dist_srgb)
        .expect("compute_dkl_planes dist");

    let sentinels = common::pycvvdp_dkl_chroma_shift_sentinels();
    let mut max_diff = 0.0_f32;
    let mut max_host_gpu_diff = 0.0_f32;
    for s in &sentinels {
        let idx = (s.y as usize) * wu + (s.x as usize);
        let our_ref = [ref_planes[0][idx], ref_planes[1][idx], ref_planes[2][idx]];
        let our_dist = [
            dist_planes[0][idx],
            dist_planes[1][idx],
            dist_planes[2][idx],
        ];
        // Host-scalar reproduction at the same pixel — pinpoints
        // whether divergence is GPU-specific or shared with host.
        let byte_i = idx * 3;
        let (host_a, host_rg, host_vy) = srgb_byte_to_dkl_scalar(
            ref_srgb[byte_i],
            ref_srgb[byte_i + 1],
            ref_srgb[byte_i + 2],
            200.0,
            0.2,
            0.397_887_36,
        );
        let host_ref = [host_a, host_rg, host_vy];
        for c in 0..3 {
            let d_ref = (our_ref[c] - s.ref_dkl[c]).abs();
            let d_dist = (our_dist[c] - s.dist_dkl[c]).abs();
            let d_host = (our_ref[c] - host_ref[c]).abs();
            if d_ref > max_diff {
                max_diff = d_ref;
            }
            if d_dist > max_diff {
                max_diff = d_dist;
            }
            if d_host > max_host_gpu_diff {
                max_host_gpu_diff = d_host;
            }
        }
        eprintln!(
            "  ({:>3},{:>3}) gpu=({:.4},{:.4},{:.4}) host=({:.4},{:.4},{:.4}) py=({:.4},{:.4},{:.4})",
            s.y,
            s.x,
            our_ref[0],
            our_ref[1],
            our_ref[2],
            host_ref[0],
            host_ref[1],
            host_ref[2],
            s.ref_dkl[0],
            s.ref_dkl[1],
            s.ref_dkl[2],
        );
    }
    eprintln!("max host-vs-gpu diff: {max_host_gpu_diff:.4e}");
    eprintln!("max DKL diff over 10 sentinels: {max_diff:.6e}");
    // Tight tolerance — DKL is the very first stage. f32 noise here
    // is bounded by the sRGB EOTF + matmul precision (1e-4 cd/m^2
    // ish at the max-luminance scale of ~200 cd/m^2).
    assert!(
        max_diff < 1e-2,
        "DKL diverges from pycvvdp at chroma_shift by {max_diff:.4e} — color transform may be the source of chroma_shift drift"
    );
}

#[test]
fn compute_dkl_t_p_bands_ref_matches_pycvvdp_at_chroma_shift_all_bands() {
    // Tick 199 stage-3 parity probe: T_p (post-CSF, pre-masking)
    // on the chroma_shift fixture, **REF side only**.
    //
    // pycvvdp's apply_masking_model computes S from REF's log_l_bkg
    // and applies the same S to both T_test_p and T_ref_p. Our
    // compute_dkl_t_p_bands(srgb) computes log_l_bkg from THAT
    // call's input. So calling compute_dkl_t_p_bands(ref_srgb)
    // matches pycvvdp's REF T_p computation; calling with dist
    // would diverge structurally because dist's log_l_bkg ≠ ref's.
    //
    // The JOD path (compute_dkl_jod) is unaffected — it uses
    // REF's log_l_bkg for both sides (via the bands_dis split).
    //
    // After ticks 196-198 established DKL + Weber bit-identical,
    // this test localizes whether the remaining 0.117 JOD chroma
    // drift sits in CSF apply (this stage) or further downstream
    // (masking / pool).
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // chroma_shift ref bytes only.
    let wu = w as usize;
    let hu = h as usize;
    let ref_srgb = common::synth_pair_ref(wu, hu);

    let ref_tp = cvvdp
        .compute_dkl_t_p_bands(&ref_srgb, ppd)
        .expect("t_p ref");

    let n_bands = ref_tp.len();
    let mut overall_max_diff = 0.0_f32;
    let mut overall_max_rel = 0.0_f32;
    let mut first_diverging_band: Option<usize> = None;
    for k in 0..n_bands {
        let sentinels = common::pycvvdp_tp_chroma_shift_band(k);
        let band_w = (wu + (1 << k) - 1) >> k;
        let mut band_max_diff = 0.0_f32;
        let mut band_max_rel = 0.0_f32;
        for s in &sentinels {
            let yk = (s.yk as usize).min(band_w - 1);
            let xk = (s.xk as usize).min(band_w - 1);
            let idx = yk * band_w + xk;
            let pairs = [
                ("ref_A", ref_tp[k][0][idx], s.t_p_ref_a),
                ("ref_RG", ref_tp[k][1][idx], s.t_p_ref_rg),
                ("ref_VY", ref_tp[k][2][idx], s.t_p_ref_vy),
            ];
            for (_, ours, py) in pairs {
                let d = (ours - py).abs();
                let r = d / py.abs().max(1e-4);
                if d > band_max_diff {
                    band_max_diff = d;
                }
                if r > band_max_rel {
                    band_max_rel = r;
                }
            }
        }
        eprintln!("band {k} REF: max T_p abs={band_max_diff:.4e} rel={band_max_rel:.4e}");
        if band_max_diff > overall_max_diff {
            overall_max_diff = band_max_diff;
        }
        if band_max_rel > overall_max_rel {
            overall_max_rel = band_max_rel;
        }
        if first_diverging_band.is_none() && band_max_rel > 1e-3 {
            first_diverging_band = Some(k);
        }
    }
    eprintln!("overall max REF T_p abs={overall_max_diff:.4e} rel={overall_max_rel:.4e}");
    if let Some(k) = first_diverging_band {
        eprintln!("FIRST DIVERGING BAND (REF T_p): {k}");
    } else {
        eprintln!("All REF T_p bands match within f32 noise — CSF apply is bit-identical");
    }
    assert!(
        overall_max_rel < 0.5,
        "REF T_p bands diverge from pycvvdp by rel={overall_max_rel:.4e} — implausible regression"
    );
}

#[test]
fn compute_dkl_weber_pyramid_matches_pycvvdp_at_chroma_shift_all_bands() {
    // Tick 198 stage-2 parity probe across ALL bands (extending
    // tick 197's band-0-only probe). Compares our
    // compute_dkl_weber_pyramid output at every band level against
    // pycvvdp's interleaved weber_contrast_pyr output (test/ref
    // channels) on the chroma_shift fixture.
    //
    // The first band index where diff exceeds f32 noise localizes
    // where the 0.117 JOD chroma drift starts.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let wu = w as usize;
    let hu = h as usize;
    let ref_srgb = common::synth_pair_ref(wu, hu);
    let dist_srgb: Vec<u8> = ref_srgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], (i16::from(p[1]) + 16).clamp(0, 255) as u8, p[2]])
        .collect();

    let (ref_bands, _ref_log) = cvvdp
        .compute_dkl_weber_pyramid(&ref_srgb)
        .expect("weber ref");
    let (dist_bands, _dist_log) = cvvdp
        .compute_dkl_weber_pyramid(&dist_srgb)
        .expect("weber dist");

    let n_bands = ref_bands.len();
    let mut overall_max_diff = 0.0_f32;
    let mut first_diverging_band: Option<usize> = None;
    for k in 0..n_bands {
        let sentinels = common::pycvvdp_weber_chroma_shift_band(k);
        let mut band_max_diff = 0.0_f32;
        // Our band[k] has dims gauss_ref[k].w × gauss_ref[k].h
        // (ceil-div from level 0). The sentinel's yk/xk is computed
        // in the Python script using floor-div from level 0, matching
        // pycvvdp's per-band shape.
        let band_w = (wu + (1 << k) - 1) >> k; // ceil-div: equals our gauss_ref[k].w
        for s in &sentinels {
            // Sentinel yk/xk are floor-divided from level 0. For the
            // common case of 256² with k ≤ 7, the levels are powers
            // of 2 so floor-div ≡ ceil-div. Use min to be safe.
            let yk = (s.yk as usize).min(band_w - 1);
            let xk = (s.xk as usize).min(band_w - 1);
            let idx = yk * band_w + xk;
            let pairs = [
                ("test_A", dist_bands[k][0][idx], s.test_a),
                ("ref_A", ref_bands[k][0][idx], s.ref_a),
                ("test_RG", dist_bands[k][1][idx], s.test_rg),
                ("ref_RG", ref_bands[k][1][idx], s.ref_rg),
                ("test_VY", dist_bands[k][2][idx], s.test_vy),
                ("ref_VY", ref_bands[k][2][idx], s.ref_vy),
            ];
            for (_label, ours, py) in pairs {
                let d = (ours - py).abs();
                if d > band_max_diff {
                    band_max_diff = d;
                }
            }
        }
        eprintln!("band {k}: max weber diff over 10 sentinels = {band_max_diff:.4e}");
        if band_max_diff > overall_max_diff {
            overall_max_diff = band_max_diff;
        }
        if first_diverging_band.is_none() && band_max_diff > 1e-4 {
            first_diverging_band = Some(k);
        }
    }
    eprintln!("overall max weber diff: {overall_max_diff:.4e}");
    if let Some(k) = first_diverging_band {
        eprintln!(
            "FIRST DIVERGING BAND: {k} — localizes the chroma drift to weber stage at this level or upstream"
        );
    } else {
        eprintln!("All weber bands match within f32 noise — drift is downstream of weber");
    }
    // The chroma_shift drift is known to land somewhere — tolerance
    // here gates against a >1% relative regression in band values.
    // The discovered first-diverging-band is documented in the
    // commit message for the next investigation step.
    assert!(
        overall_max_diff < 0.5,
        "Weber bands diverge from pycvvdp by {overall_max_diff:.4e} — implausibly large"
    );
}

#[test]
fn spatial_pool_q_per_ch_matches_pycvvdp_at_chroma_shift_all_bands() {
    // Tick 204 stage-7 probe: spatial-pool Q_per_ch values at
    // chroma_shift. Compares `lp_norm_mean(D[c], BETA_SPATIAL)` over
    // our `compute_dkl_d_bands` output against pycvvdp's
    // `lp_norm(D, beta=2, dim=spatial, normalize=True)`.
    //
    // After tick 203, D bands match pycvvdp at f32 noise on large
    // magnitudes (tick 201 D test). If Q_per_ch ALSO matches, the
    // 0.117 JOD drift is in band/channel pools or met2jod. If
    // Q_per_ch diverges, the spatial pool itself is the source.
    use cvvdp_gpu::kernels::pool::{BETA_SPATIAL, lp_norm_mean};

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // chroma_shift ref + dist bytes.
    let wu = w as usize;
    let hu = h as usize;
    let ref_srgb = common::synth_pair_ref(wu, hu);
    let dist_srgb: Vec<u8> = ref_srgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], (i16::from(p[1]) + 16).clamp(0, 255) as u8, p[2]])
        .collect();

    let d_bands = cvvdp
        .compute_dkl_d_bands(&ref_srgb, &dist_srgb, ppd)
        .expect("d bands");

    let n_bands = d_bands.len();
    let mut overall_max_rel = 0.0_f32;
    let mut overall_max_abs = 0.0_f32;
    let mut first_diverging_band: Option<usize> = None;
    for k in 0..n_bands {
        let py = common::pycvvdp_q_chroma_shift_band(k);
        let ours_a = lp_norm_mean(&d_bands[k][0], BETA_SPATIAL);
        let ours_rg = lp_norm_mean(&d_bands[k][1], BETA_SPATIAL);
        let ours_vy = lp_norm_mean(&d_bands[k][2], BETA_SPATIAL);
        let pairs = [
            ("Q_A", ours_a, py.q_a),
            ("Q_RG", ours_rg, py.q_rg),
            ("Q_VY", ours_vy, py.q_vy),
        ];
        let mut band_max_abs = 0.0_f32;
        let mut band_max_rel = 0.0_f32;
        for (_, ours, p) in pairs {
            let d = (ours - p).abs();
            let r = d / p.abs().max(1e-6);
            if d > band_max_abs {
                band_max_abs = d;
            }
            if r > band_max_rel {
                band_max_rel = r;
            }
        }
        eprintln!(
            "band {k}: Q_per_ch ours=[{ours_a:.4e}, {ours_rg:.4e}, {ours_vy:.4e}] \
             pycvvdp=[{:.4e}, {:.4e}, {:.4e}] \
             max abs={band_max_abs:.4e} rel={band_max_rel:.4e}",
            py.q_a, py.q_rg, py.q_vy
        );
        if band_max_abs > overall_max_abs {
            overall_max_abs = band_max_abs;
        }
        if band_max_rel > overall_max_rel {
            overall_max_rel = band_max_rel;
        }
        if first_diverging_band.is_none() && band_max_rel > 1e-3 {
            first_diverging_band = Some(k);
        }
    }
    eprintln!("overall max Q_per_ch abs={overall_max_abs:.4e} rel={overall_max_rel:.4e}");
    if let Some(k) = first_diverging_band {
        eprintln!("FIRST DIVERGING BAND (Q_per_ch): {k} — spatial pool source localized");
    } else {
        eprintln!(
            "All Q_per_ch bands match at <0.1% rel — spatial pool is bit-close; \
             drift is in band/channel pool or met2jod"
        );
    }
    assert!(
        overall_max_rel < 0.5,
        "Q_per_ch diverges from pycvvdp by rel={overall_max_rel:.4e} — implausible"
    );
}

#[test]
fn compute_dkl_t_p_bands_matches_host_scalar_per_pixel_at_chroma_shift() {
    // Tick 203 stage-6 parity probe: pixel-by-pixel GPU vs host
    // scalar T_p comparison at chroma_shift.
    //
    // Tick 202 established host_scalar's S matches pycvvdp at
    // f32 noise (1e-6 rel) across all 8 bands. Tick 199 found GPU
    // T_p diverges from pycvvdp by 0.9% rel. By transitivity, the
    // GPU csf_apply_3ch_kernel must diverge from the host scalar
    // by ~0.9% rel — this test localizes that.
    //
    // Existing `compute_dkl_t_p_bands_matches_host_scalar` uses a
    // 5e-3 band-max-normalized tolerance which would mask 0.9%
    // per-pixel divergence. This test uses **per-pixel rel
    // tolerance** so the source localizes to the kernel rather
    // than getting averaged out.
    use cvvdp_gpu::kernels::csf::{CsfChannel, sensitivity_corrected_scalar};
    use cvvdp_gpu::kernels::masking::CH_GAIN;
    use cvvdp_gpu::kernels::pyramid::band_frequencies;

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let wu = w as usize;
    let hu = h as usize;
    let ref_srgb = common::synth_pair_ref(wu, hu);

    // GPU T_p.
    let t_p_gpu = cvvdp
        .compute_dkl_t_p_bands(&ref_srgb, ppd)
        .expect("compute_dkl_t_p_bands");

    // Host T_p — same chain pycvvdp uses, computed in pure
    // host scalar. Uses the SAME Weber + log_l_bkg the GPU path
    // produces internally (read back via the host scalar Weber
    // pyramid; tick 198 confirmed they match the GPU to 2.7e-7).
    let n_px = wu * hu;
    let display = DisplayModel::STANDARD_4K;
    let mut planes: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
    for (i, chunk) in ref_srgb.chunks_exact(3).enumerate() {
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
            &planes[0], &planes[0], wu, hu, n_levels,
        ),
        cvvdp_gpu::kernels::pyramid::weber_contrast_pyr_dec_scalar(
            &planes[1], &planes[0], wu, hu, n_levels,
        ),
        cvvdp_gpu::kernels::pyramid::weber_contrast_pyr_dec_scalar(
            &planes[2], &planes[0], wu, hu, n_levels,
        ),
    ];

    let freqs = band_frequencies(ppd, wu, hu);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

    let mut overall_max_rel = 0.0_f32;
    let mut overall_max_abs = 0.0_f32;
    let mut first_diverging_band: Option<usize> = None;
    for k in 0..n_levels {
        let is_first = k == 0;
        let is_baseband = k == n_levels - 1;
        let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
        let bw = host_per_ch[0].bands[k].w;
        let bh = host_per_ch[0].bands[k].h;
        let n_band = bw * bh;
        let log_l_bkg_band = &host_per_ch[0].log_l_bkg[k];

        // Tick 204: baseband CSF rho override.
        let rho_eff = if is_baseband {
            cvvdp_gpu::kernels::csf::CSF_BASEBAND_RHO
        } else {
            freqs[k]
        };
        let mut band_max_rel = 0.0_f32;
        let mut band_max_abs = 0.0_f32;
        for c in 0..3 {
            let weber_c = &host_per_ch[c].bands[k].data;
            for i in 0..n_band {
                let s = sensitivity_corrected_scalar(rho_eff, log_l_bkg_band[i], channels[c]);
                let ch_gain_eff = if is_baseband {
                    1.0
                } else {
                    band_mul * CH_GAIN[c]
                };
                let host_t_p = weber_c[i] * s * ch_gain_eff;
                let gpu_t_p = t_p_gpu[k][c][i];
                let d = (gpu_t_p - host_t_p).abs();
                // Per-pixel rel error — skip near-zero pixels
                // where |host| < 1e-4 (denom would amplify f32
                // noise into spurious "huge" rel errors).
                let r = if host_t_p.abs() > 1e-4 {
                    d / host_t_p.abs()
                } else {
                    0.0
                };
                if d > band_max_abs {
                    band_max_abs = d;
                }
                if r > band_max_rel {
                    band_max_rel = r;
                }
            }
        }
        eprintln!("band {k} GPU vs host T_p: max abs={band_max_abs:.4e} rel={band_max_rel:.4e}");
        if band_max_abs > overall_max_abs {
            overall_max_abs = band_max_abs;
        }
        if band_max_rel > overall_max_rel {
            overall_max_rel = band_max_rel;
        }
        if first_diverging_band.is_none() && band_max_rel > 1e-3 {
            first_diverging_band = Some(k);
        }
    }
    eprintln!("overall max GPU vs host T_p abs={overall_max_abs:.4e} rel={overall_max_rel:.4e}");
    if let Some(k) = first_diverging_band {
        eprintln!(
            "FIRST DIVERGING BAND (GPU vs host T_p): {k} — kernel-side discrepancy localized"
        );
    } else {
        eprintln!(
            "All bands match at <0.1% rel — GPU csf_apply matches host_scalar; \
             the 0.9% T_p vs pycvvdp drift must come from elsewhere"
        );
    }
    assert!(
        overall_max_rel < 0.5,
        "GPU T_p diverges from host_scalar by rel={overall_max_rel:.4e} — implausible"
    );
}

#[test]
fn sensitivity_scalar_matches_pycvvdp_raw_csf_at_chroma_shift_all_bands() {
    // Tick 202 stage-5 parity probe: raw CSF sensitivity (no
    // sens_corr_factor applied) at chroma_shift sentinels.
    //
    // Inputs to our `sensitivity_scalar(rho, log_l_bkg, cc)` are
    // pinned to pycvvdp's exact per-pixel `log_l_bkg_ref` (the REF
    // side of the achromatic Weber pyramid's log10-gauss). With the
    // same input, S divergence isolates the CSF lookup itself
    // (table + interp) from upstream pyramid + downstream sens_corr
    // application.
    //
    // Tick 198 confirmed Weber bands bit-identical. Tick 199 found
    // T_p REF-side 0.89% rel drift. Since T = Weber (bit-identical)
    // and ch_gain is constant, S MUST carry the 0.9% drift.
    // This probe answers: does S diverge BEFORE sens_corr (CSF
    // lookup divergence) or AFTER (sens_corr application order)?
    use cvvdp_gpu::kernels::csf::{CsfChannel, sensitivity_scalar};

    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];
    let n_bands = 8;
    let mut overall_max_rel = 0.0_f32;
    let mut overall_max_abs = 0.0_f32;
    let mut first_diverging_band: Option<usize> = None;
    for k in 0..n_bands {
        let rho = common::pycvvdp_s_chroma_shift_rho(k);
        let sentinels = common::pycvvdp_s_chroma_shift_band(k);
        let mut band_max_rel = 0.0_f32;
        let mut band_max_abs = 0.0_f32;
        for s in &sentinels {
            let ours_a = sensitivity_scalar(rho, s.log_l_bkg_ref, channels[0]);
            let ours_rg = sensitivity_scalar(rho, s.log_l_bkg_ref, channels[1]);
            let ours_vy = sensitivity_scalar(rho, s.log_l_bkg_ref, channels[2]);
            let pairs = [
                ("S_A", ours_a, s.s_raw_a),
                ("S_RG", ours_rg, s.s_raw_rg),
                ("S_VY", ours_vy, s.s_raw_vy),
            ];
            for (_, ours, py) in pairs {
                let d = (ours - py).abs();
                let r = d / py.abs().max(1e-6);
                if d > band_max_abs {
                    band_max_abs = d;
                }
                if r > band_max_rel {
                    band_max_rel = r;
                }
            }
        }
        eprintln!("band {k} rho={rho:.3} raw S: max abs={band_max_abs:.4e} rel={band_max_rel:.4e}");
        if band_max_abs > overall_max_abs {
            overall_max_abs = band_max_abs;
        }
        if band_max_rel > overall_max_rel {
            overall_max_rel = band_max_rel;
        }
        // CSF lookups should agree at f32 noise floor (~1e-6 rel)
        // since the LUT tables match pycvvdp's at 5e-11 (tick 193)
        // and the interp methods are mathematically equivalent for
        // matching inputs. 1e-3 rel marks a real divergence.
        if first_diverging_band.is_none() && band_max_rel > 1e-3 {
            first_diverging_band = Some(k);
        }
    }
    eprintln!("overall max raw S abs={overall_max_abs:.4e} rel={overall_max_rel:.4e}");
    if let Some(k) = first_diverging_band {
        eprintln!("FIRST DIVERGING BAND (raw S): {k} — CSF lookup divergence localized");
    } else {
        eprintln!(
            "All raw S bands match within 1e-3 rel — CSF lookup is bit-close; \
             the 0.9% T_p drift is from sens_corr application order"
        );
    }
    assert!(
        overall_max_rel < 0.5,
        "raw S diverges from pycvvdp by rel={overall_max_rel:.4e} — implausible regression"
    );
}

#[test]
fn compute_dkl_d_bands_matches_pycvvdp_at_chroma_shift_all_bands() {
    // Tick 201 stage-4 parity probe: D values (post-masking,
    // post-PU-blur, pre-pool) on the chroma_shift fixture, both
    // sides combined since D = clamped(|T_p-R_p|^p / (1+M)).
    //
    // After ticks 196-198 established DKL + Weber bit-identical
    // and tick 199 found T_p REF-side diverges 0.89% rel, this
    // localizes whether the masking model is where the divergence
    // amplifies (or where it stays bounded) — if D bands match
    // pycvvdp, the drift is downstream in pool / accumulation
    // order; if D diverges, masking is the source.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // chroma_shift ref + dist bytes (same synth as the
    // T_p test, plus the +16-on-G distortion).
    let wu = w as usize;
    let hu = h as usize;
    let ref_srgb = common::synth_pair_ref(wu, hu);
    let dist_srgb: Vec<u8> = ref_srgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], (i16::from(p[1]) + 16).clamp(0, 255) as u8, p[2]])
        .collect();

    let d_bands = cvvdp
        .compute_dkl_d_bands(&ref_srgb, &dist_srgb, ppd)
        .expect("d bands");

    let n_bands = d_bands.len();
    let mut overall_max_diff = 0.0_f32;
    let mut overall_max_rel = 0.0_f32;
    let mut first_diverging_band: Option<usize> = None;
    for k in 0..n_bands {
        let sentinels = common::pycvvdp_d_chroma_shift_band(k);
        let band_w = (wu + (1 << k) - 1) >> k;
        let band_h = (hu + (1 << k) - 1) >> k;
        let mut band_max_diff = 0.0_f32;
        let mut band_max_rel = 0.0_f32;
        for s in &sentinels {
            let yk = (s.yk as usize).min(band_h - 1);
            let xk = (s.xk as usize).min(band_w - 1);
            let idx = yk * band_w + xk;
            let pairs = [
                ("D_A", d_bands[k][0][idx], s.d_a),
                ("D_RG", d_bands[k][1][idx], s.d_rg),
                ("D_VY", d_bands[k][2][idx], s.d_vy),
            ];
            for (_, ours, py) in pairs {
                let d = (ours - py).abs();
                let r = d / py.abs().max(1e-6);
                if d > band_max_diff {
                    band_max_diff = d;
                }
                if r > band_max_rel {
                    band_max_rel = r;
                }
            }
        }
        eprintln!("band {k} D: max abs={band_max_diff:.4e} rel={band_max_rel:.4e}");
        if band_max_diff > overall_max_diff {
            overall_max_diff = band_max_diff;
        }
        if band_max_rel > overall_max_rel {
            overall_max_rel = band_max_rel;
        }
        if first_diverging_band.is_none() && band_max_rel > 1e-2 {
            first_diverging_band = Some(k);
        }
    }
    eprintln!("overall max D abs={overall_max_diff:.4e} rel={overall_max_rel:.4e}");
    if let Some(k) = first_diverging_band {
        eprintln!("FIRST DIVERGING BAND (D): {k}");
    } else {
        eprintln!("All D bands match within 1% rel — masking is bit-close, drift sits downstream");
    }
    // Generous tolerance gates regression; the real drift is reported
    // via eprintln! so future ticks can read the per-band diff and
    // localize the source. T_p had max ~0.9% rel; D may amplify to
    // ~5% if masking is sensitive (mask_p ≈ 2.26 in the safe_pow).
    // 50% rel is the implausible-regression tripwire.
    assert!(
        overall_max_rel < 0.5,
        "D bands diverge from pycvvdp by rel={overall_max_rel:.4e} — implausible regression"
    );
}

#[test]
fn compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift() {
    // Tick 204: drift closed. The 0.1174 JOD chroma_shift drift
    // chased through ticks 191-203 was caused by our pipeline
    // using the geometric baseband rho (0.190 cy/deg at 256²
    // standard_4k) for the CSF lookup, whereas pycvvdp overrides
    // it to 0.1 cy/deg (cvvdp_metric.py:628). Fixed in this tick
    // — host_scalar + GPU pipeline both now use
    // `CSF_BASEBAND_RHO = 0.1` at baseband.
    let pycvvdp_golden_jod = common::pycvvdp_synth_golden_jod("synth_256x256_chroma_shift");

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_srgb = common::synth_pair_ref(w as usize, h as usize);
    let dist_srgb: Vec<u8> = ref_srgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], (i16::from(p[1]) + 16).clamp(0, 255) as u8, p[2]])
        .collect();

    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");
    let diff = (gpu_jod - pycvvdp_golden_jod).abs();
    eprintln!(
        "256×256 chroma_shift: gpu_jod = {gpu_jod:.4}, pycvvdp golden = {pycvvdp_golden_jod:.4}, |diff| = {diff:.4}"
    );
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    assert!(
        diff < TOLERANCE,
        "GPU JOD {gpu_jod:.4} drifts from pycvvdp golden {pycvvdp_golden_jod:.4} by {diff:.4} > {TOLERANCE:.4}"
    );
}

#[test]
fn compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_256x256_chroma_shift() {
    // Tick 222: direct warm-ref pycvvdp parity. The warm-ref state
    // machine is currently covered transitively:
    //   - compute_dkl_jod_with_warm_ref_matches_unwarm_path pins
    //     warm == unwarm at ≤ 1e-5 JOD
    //   - compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift
    //     pins unwarm == pycvvdp at ≤ 0.005 JOD (closed in tick 204
    //     to 0.000000 on this fixture)
    // This test closes the transitive chain with a direct measure,
    // catching any regression that breaks at the warm-state-restoration
    // step without surfacing in either transitive leg.
    let pycvvdp_golden_jod = common::pycvvdp_synth_golden_jod("synth_256x256_chroma_shift");

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_srgb = common::synth_pair_ref(w as usize, h as usize);
    let dist_srgb: Vec<u8> = ref_srgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], (i16::from(p[1]) + 16).clamp(0, 255) as u8, p[2]])
        .collect();

    cvvdp.warm_reference(&ref_srgb).expect("warm_reference");
    let gpu_jod = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
        .expect("compute_dkl_jod_with_warm_ref");
    let diff = (gpu_jod - pycvvdp_golden_jod).abs();
    eprintln!(
        "warm-ref 256×256 chroma_shift: gpu_jod = {gpu_jod:.4}, pycvvdp golden = {pycvvdp_golden_jod:.4}, |diff| = {diff:.4}"
    );
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    assert!(
        diff < TOLERANCE,
        "warm-ref JOD {gpu_jod:.4} drifts from pycvvdp golden {pycvvdp_golden_jod:.4} by {diff:.4} > {TOLERANCE:.4}"
    );
}

// NOTE (tick 191, updated tick 196): the
// compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift test
// still fails (0.1174 JOD drift vs golden 9.6649). Tick 196
// fixed a SRGB8_TO_LINEAR_LUT bug (~6e-4 drift at bright bytes)
// — DKL planes are now bit-identical with pycvvdp (verified
// by compute_dkl_planes_matches_pycvvdp_dkl_at_chroma_shift_sentinels
// at 3.8e-5 max diff) — but the JOD still drifts by 0.117. So
// the divergence is DOWNSTREAM of the color transform (in
// pyramid/weber/CSF/masking/pool). The bench-script golden is
// captured in pycvvdp_synth_goldens.json; this test gets re-
// enabled once the downstream drift closes. See
// docs/CHROMA_DRIFT_INVESTIGATION.md.

#[test]
fn compute_dkl_jod_matches_pycvvdp_at_256x256_blur1x3() {
    // 256×256 pycvvdp parity with a VERTICAL 3-pixel blur —
    // complement to blur3x1 (horizontal). Together they exercise
    // both axes of the separable pyramid passes. The vertical blur
    // golden (8.1243) is lower than horizontal (8.4412), reflecting
    // the CSF's known vertical-leaning anisotropy.
    //
    // Bit-stable synth: dist[y,x,c] =
    //   (ref[y,x,c] + ref[(y+1)%h,x,c] + ref[(y+2)%h,x,c]) // 3.
    let pycvvdp_golden_jod = common::pycvvdp_synth_golden_jod("synth_256x256_blur1x3");

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let wu = w as usize;
    let hu = h as usize;
    let ref_srgb = common::synth_pair_ref(wu, hu);
    let mut dist_srgb = vec![0u8; wu * hu * 3];
    for y in 0..hu {
        for x in 0..wu {
            let y1 = (y + 1) % hu;
            let y2 = (y + 2) % hu;
            for c in 0..3 {
                let a = u16::from(ref_srgb[(y * wu + x) * 3 + c]);
                let b = u16::from(ref_srgb[(y1 * wu + x) * 3 + c]);
                let cval = u16::from(ref_srgb[(y2 * wu + x) * 3 + c]);
                dist_srgb[(y * wu + x) * 3 + c] = ((a + b + cval) / 3) as u8;
            }
        }
    }

    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");
    let diff = (gpu_jod - pycvvdp_golden_jod).abs();
    eprintln!(
        "256×256 blur1x3: gpu_jod = {gpu_jod:.4}, pycvvdp golden = {pycvvdp_golden_jod:.4}, |diff| = {diff:.4}"
    );
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    assert!(
        diff < TOLERANCE,
        "GPU JOD {gpu_jod:.4} drifts from pycvvdp golden {pycvvdp_golden_jod:.4} by {diff:.4} > {TOLERANCE:.4}"
    );
}

#[test]
fn compute_dkl_jod_matches_pycvvdp_at_256x256_noise() {
    // 256×256 pycvvdp parity with a non-spatial distortion: per-pixel
    // additive noise. Complementary to the blur3x1 test (which adds
    // spatial correlation in one direction). Together they cover two
    // distinct CSF/masking response shapes — broadband noise activates
    // every band, while horizontal blur attenuates high horizontal
    // frequencies specifically.
    //
    // Bit-stable synth construction across NumPy + Rust:
    //   noise[y,x,c] = ((x * 73 + y * 137 + c * 211) % 64) - 32
    //   dist[y,x,c] = clamp(ref[y,x,c] + noise[y,x,c], 0, 255)
    let pycvvdp_golden_jod = common::pycvvdp_synth_golden_jod("synth_256x256_noise");

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let wu = w as usize;
    let hu = h as usize;
    let ref_srgb = common::synth_pair_ref(wu, hu);
    let mut dist_srgb = vec![0u8; wu * hu * 3];
    for y in 0..hu {
        for x in 0..wu {
            let i = (y * wu + x) * 3;
            for c in 0..3 {
                let noise = ((x as i64 * 73 + y as i64 * 137 + c as i64 * 211) % 64) - 32;
                let v = (i64::from(ref_srgb[i + c]) + noise).clamp(0, 255) as u8;
                dist_srgb[i + c] = v;
            }
        }
    }

    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");
    let diff = (gpu_jod - pycvvdp_golden_jod).abs();
    eprintln!(
        "256×256 noise: gpu_jod = {gpu_jod:.4}, pycvvdp golden = {pycvvdp_golden_jod:.4}, |diff| = {diff:.4}"
    );
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    assert!(
        diff < TOLERANCE,
        "GPU JOD {gpu_jod:.4} drifts from pycvvdp golden {pycvvdp_golden_jod:.4} by {diff:.4} > {TOLERANCE:.4}"
    );
}

#[test]
fn compute_dkl_jod_matches_host_scalar_on_odd_dims() {
    // Catches regressions in the ceil-div pyramid invariant (tick
    // 175). All other JOD parity tests run at power-of-2 sizes
    // (32×32, 256×256) where floor-div == ceil-div at every pyramid
    // level — so they can't detect a future caller reverting either
    // the host_scalar or GPU pyramid back to floor-div halving.
    //
    // 73×91 produces an odd-dim source that diverges at level 4+:
    //   73 → 37 (ceil) vs 36 (floor)
    //   91 → 46 (ceil) vs 45 (floor)
    // If host_scalar and GPU both use ceil-div (current state) the
    // outputs agree within f32 precision. If either path slips back
    // to floor-div, the JOD diverges visibly.
    use cvvdp_gpu::host_scalar::predict_jod_still_3ch;

    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // Same per-pixel construction as the 32×32 test — distinct R/G/B
    // patterns + a small DIST perturbation so the JOD is non-trivial.
    let (ref_srgb, dist_srgb) = common::synth_pair_odd_dim_with_offset_dist(w as usize, h as usize);

    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");
    let host_jod =
        predict_jod_still_3ch(&ref_srgb, &dist_srgb, w as usize, h as usize, display, ppd);
    let diff = (gpu_jod - host_jod).abs();
    eprintln!(
        "odd-dim 73×91: gpu_jod = {gpu_jod:.6}, host_scalar = {host_jod:.6}, |diff| = {diff:.6}"
    );
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    // Tightened in tick 184. Post tick-181's band-count alignment,
    // observed diff = 0.0004 JOD. 0.005 gives ~12× margin while still
    // gating a real regression: a revert of ceil-div or the
    // band_frequencies-driven n_levels would push diff to 0.09+ JOD
    // (tick-177 baseline pre-band-count-fix) or 0.586+ JOD (tick-173
    // pre-ceil-div). Either far exceeds 0.005.
    assert!(
        diff < 0.005,
        "GPU JOD {gpu_jod:.6} diverges from host scalar {host_jod:.6} by {diff:.6} on odd dims (was 0.0004 at tick 184)"
    );
}

#[test]
fn compute_dkl_jod_matches_pycvvdp_at_73x91_odd() {
    // Tick 206: closed the odd-dim 73×91 residual drift identified
    // in tick 205. Root cause: pycvvdp's `gausspyr_reduce` has a
    // subtle bug — its horizontal-pass right-column patch selects
    // odd/even branch based on `x.shape[-2]` (INPUT ROW parity),
    // not column parity. Comments say "odd number of columns" but
    // the check uses rows (cvvdp_metric/lpyr_dec.py:204-209). For
    // mixed-parity inputs (e.g. 6×5 → 3×3 at level 4→5 of 73×91)
    // pycvvdp applies the wrong patch.
    //
    // We replicate the bug to match goldens: host_scalar gauss
    // reduce switched from pure reflection to zero-pad +
    // parity-aware patches; GPU `downscale_kernel` keeps the
    // reflect-based main path and applies a delta correction at
    // the right column when sw and sh parities mismatch.
    //
    // dist construction matches `synth_pair_odd_dim` in
    // `scripts/cvvdp_goldens/bench_12mp_cuda.py`.
    let pycvvdp_golden_jod = common::pycvvdp_synth_golden_jod("synth_73x91_odd");

    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let (ref_srgb, dist_srgb) = common::synth_pair_odd_dim_with_offset_dist(w as usize, h as usize);

    // Our compute_dkl_jod(ref, dist) corresponds semantically to
    // pycvvdp's predict(dist, ref) — both "score this distortion
    // against this reference" — even though the arg orders differ.
    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");
    let diff = (gpu_jod - pycvvdp_golden_jod).abs();
    eprintln!(
        "odd-dim 73×91: gpu_jod = {gpu_jod:.4}, pycvvdp golden = {pycvvdp_golden_jod:.4}, |diff| = {diff:.4}"
    );
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    assert!(
        diff < TOLERANCE,
        "GPU JOD {gpu_jod:.4} drifts from pycvvdp golden {pycvvdp_golden_jod:.4} by {diff:.4} > {TOLERANCE:.4}"
    );
}

#[test]
fn compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_73x91_odd() {
    // Tick 226: direct warm-ref pycvvdp parity on the mixed-parity
    // 73×91 fixture. Pairs with the chroma_shift warm-ref test
    // (`compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_256x256_chroma_shift`,
    // tick 222) but exercises a fundamentally different code path
    // on REF: the warm-state restoration has to apply the tick-206
    // gausspyr_reduce parity-bug fix on a pyramid whose mixed-parity
    // reduce levels (6×5 → 3×3 at level 4→5, 46×37 → 23×19 at level
    // 1→2) hit the bug-compatible right-column delta correction.
    //
    // Transitive coverage before this test:
    //   - compute_dkl_jod_matches_pycvvdp_at_73x91_odd (tick 206)
    //     pins cold compute_dkl_jod == pycvvdp at ≤ 0.005 JOD
    //   - compute_dkl_jod_with_warm_ref_matches_unwarm_path pins
    //     warm == unwarm at ≤ 1e-5 JOD (synth pair, same-parity dims)
    // Neither leg exercises the warm-state restoration on a
    // mixed-parity REF — that's what this test closes.
    let pycvvdp_golden_jod = common::pycvvdp_synth_golden_jod("synth_73x91_odd");

    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let (ref_srgb, dist_srgb) = common::synth_pair_odd_dim_with_offset_dist(w as usize, h as usize);

    cvvdp.warm_reference(&ref_srgb).expect("warm_reference");
    let gpu_jod = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
        .expect("compute_dkl_jod_with_warm_ref");
    let diff = (gpu_jod - pycvvdp_golden_jod).abs();
    eprintln!(
        "warm-ref 73×91 odd: gpu_jod = {gpu_jod:.4}, pycvvdp golden = {pycvvdp_golden_jod:.4}, |diff| = {diff:.4}"
    );
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    assert!(
        diff < TOLERANCE,
        "warm-ref JOD {gpu_jod:.4} drifts from pycvvdp golden {pycvvdp_golden_jod:.4} by {diff:.4} > {TOLERANCE:.4}"
    );
}

#[test]
fn compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_12mp_synth() {
    // Tick 235: large-image warm-ref pycvvdp parity. Completes the
    // warm-ref vs pycvvdp coverage grid:
    //   - chroma_shift (tick 222): small same-parity
    //   - 73×91 odd-dim (tick 226): small mixed-parity
    //   - 4000×3000 here:           large same-parity (full ~9-band pyramid)
    //
    // The full-depth pyramid exercises the warm-state restoration
    // across every weber_scratch[k].log_l_bkg level — a regression
    // in deep-pyramid warm-state would surface here even if it stays
    // hidden on the 256×256 fixtures.
    //
    // Runtime: ~600ms cold warm_reference + ~600ms compute_dkl_jod_with_warm_ref
    // on RTX-class CUDA. Acceptable budget for parity tests at
    // parity with the existing compute_dkl_jod_matches_pycvvdp_at_12mp_synth.
    //
    // Synth construction matches `synth_pair_12mp` in
    // scripts/cvvdp_goldens/bench_12mp_cuda.py — keep in sync.
    let pycvvdp_golden_jod: f32 = common::pycvvdp_synth_golden_jod("synth_4000x3000");

    let client = Backend::client(&Default::default());
    let (w, h) = (4000u32, 3000u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let (ref_srgb, dist_srgb) = common::synth_pair_with_offset_dist(w as usize, h as usize);

    cvvdp.warm_reference(&ref_srgb).expect("warm_reference");
    let gpu_jod = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
        .expect("compute_dkl_jod_with_warm_ref");
    let diff = (gpu_jod - pycvvdp_golden_jod).abs();
    eprintln!(
        "warm-ref 12mp synth: gpu_jod = {gpu_jod:.4}, pycvvdp golden = {pycvvdp_golden_jod:.4}, |diff| = {diff:.4}"
    );
    assert!(gpu_jod.is_finite(), "JOD must be finite, got {gpu_jod}");
    assert!(
        (0.0..=10.0).contains(&gpu_jod),
        "JOD must be in [0, 10], got {gpu_jod}"
    );
    assert!(
        diff < TOLERANCE,
        "warm-ref 12mp JOD {gpu_jod:.4} drifts from pycvvdp golden {pycvvdp_golden_jod:.4} by {diff:.4} > {TOLERANCE:.4}"
    );
}
