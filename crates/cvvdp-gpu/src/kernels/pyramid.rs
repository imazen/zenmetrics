//! Pyramid decomposition for still-image cvvdp (Weber contrast on
//! non-baseband levels, gaussian residual on the baseband).
//!
//! Per DKL channel, `weber_contrast_pyr_dec_scalar` produces
//! `n_levels` bands:
//!
//! - `band[k]` for `k < n_levels - 1` = Weber contrast of the
//!   `(gauss[k] - expand(gauss[k+1]))` layer relative to the
//!   per-pixel achromatic `L_bkg` plane (`expand(gauss_l_bkg[k+1])`,
//!   floored at 0.01, clipped to ±1000).
//! - `band[n_levels - 1]` = the coarsest gaussian (residual); the
//!   host bypasses Weber contrast for the baseband and feeds it
//!   directly into pooling.
//!
//! cvvdp v0.5.4 uses a 5-tap separable Gaussian (the "Burt-Adelson
//! kernel" with `a = 0.4`):
//!
//! ```text
//! K[a] = [0.25 - a/2, 0.25, a, 0.25, 0.25 - a/2]
//! ```
//!
//! At `a = 0.4` that's `[0.05, 0.25, 0.40, 0.25, 0.05]`. Applied
//! separably (vertical then horizontal) at stride 2 in each direction
//! for the reduce step; zero-interleaved at stride 2 + filtered for
//! the expand step (with the ×4 reconstruction gain split between the
//! two passes).
//!
//! Edge handling: **symmetric padding**. cvvdp's `gausspyr_reduce`
//! uses `F.conv2d` with `padding=2` (zero-pad) and then patches the
//! first/last rows/cols with explicit reflection terms. For the
//! scalar reference here we collapse those patches into a single
//! reflect-index helper; numerical equivalence is verified against
//! pycvvdp goldens in `tests/pyramid_scalar.rs`.
//!
//! Kernels in this module (all live, all parity-tested in
//! `tests/pyramid_kernel.rs`):
//!
//! - `downscale_kernel` — 5-tap separable Gaussian + 2× decimation
//!   (gauss-pyramid reduce step).
//! - `upscale_v_kernel` + `upscale_h_kernel` — separable vertical
//!   then horizontal 2× zero-insertion + 5-tap Gaussian (gauss-pyramid
//!   expand step), with reconstruction gain ×4 split as ×2 per pass.
//! - `subtract_kernel` — `band = fine - upscaled_coarse`. Still
//!   used by `compute_dkl_laplacian_pyramid` (vanilla Laplacian)
//!   and `compute_dkl_csf_weighted_bands` via the shared
//!   `_dispatch_laplacian_pyramid_gpu` helper; the Weber path
//!   went through the fused subtract+weber kernel below.
//! - `weber_contrast_compute_kernel` — per-pixel `layer / L_bkg`
//!   with cvvdp's clamps + `log10(L_bkg)` emission for the CSF
//!   lookup. Spec reference for the per-pixel math; production
//!   uses the 3-channel variants below.
//! - `weber_contrast_compute_3ch_kernel` — fused 3-channel weber
//!   compute with shared `log10(L_bkg)` write. One launch per
//!   non-baseband level instead of three.
//! - `subtract_weber_3ch_kernel` — further fuses the subtract step
//!   into the weber compute. Reads `fine[c]` + `upscaled[c]` for
//!   3 channels and writes `band[c] = clamp((fine[c] - upscaled[c])
//!   / L_bkg)` + shared `log_l_bkg`. Production weber kernel
//!   (replaces 3× `subtract_kernel` + 1× weber per level).

use cubecl::prelude::*;

/// Burt-Adelson kernel parameter `a` used by cvvdp v0.5.4.
pub const KERNEL_A: f32 = 0.4;

/// 5-tap separable Gaussian, evaluated from [`KERNEL_A`].
pub const GAUSS5: [f32; 5] = [
    0.25 - KERNEL_A / 2.0,
    0.25,
    KERNEL_A,
    0.25,
    0.25 - KERNEL_A / 2.0,
];

/// 2D separable 5-tap Gaussian + 2× decimation in each axis. Output
/// dimensions are `((sw + 1) / 2, (sh + 1) / 2)` — cvvdp rounds odd
/// dims up. Edge handling = symmetric reflection.
///
/// Two-pass: vertical pass decimates h by 2 into `sw × dh` scratch,
/// horizontal pass decimates w by 2 into the final `dw × dh` output.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pyramid::gausspyr_reduce_scalar;
///
/// // 8×8 input → 4×4 output (ceil-halving).
/// let src = vec![1.0_f32; 64];
/// let mut dst = Vec::new();
/// let (dw, dh) = gausspyr_reduce_scalar(&src, 8, 8, &mut dst);
/// assert_eq!((dw, dh), (4, 4));
/// assert_eq!(dst.len(), 16);
///
/// // Odd-dim ceil-halving: 7 → 4.
/// let src7 = vec![1.0_f32; 49];
/// let mut dst7 = Vec::new();
/// let (dw7, dh7) = gausspyr_reduce_scalar(&src7, 7, 7, &mut dst7);
/// assert_eq!((dw7, dh7), (4, 4));
/// ```
pub fn gausspyr_reduce_scalar(
    src: &[f32],
    sw: usize,
    sh: usize,
    dst: &mut Vec<f32>,
) -> (usize, usize) {
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    dst.clear();
    dst.resize(dw * dh, 0.0);
    let k = GAUSS5;

    // Bug-compatible with pycvvdp's `gausspyr_reduce` (lpyr_dec.py:186):
    // each pass uses zero-pad conv + an explicit edge patch that
    // emulates symmetric reflection. The horizontal-pass right-column
    // patch picks its odd/even branch from the INPUT row parity
    // (`x.shape[-2]`) — pycvvdp's source comments say "odd number of
    // columns" but the check uses rows. We replicate that quirk so
    // the goldens match. Tick 206 fix for the 73×91 odd-dim residual
    // (see docs/CHROMA_DRIFT_INVESTIGATION.md follow-up).

    // Vertical pass: zero-pad rows above/below, conv stride 2.
    let mut vscratch = vec![0.0_f32; sw * dh];
    for dy in 0..dh {
        let cy = 2 * dy as isize;
        for x in 0..sw {
            let read = |off: isize| -> f32 {
                let r = cy + off;
                if r < 0 || r >= sh as isize {
                    0.0
                } else {
                    src[r as usize * sw + x]
                }
            };
            vscratch[dy * sw + x] = k[0] * read(-2)
                + k[1] * read(-1)
                + k[2] * read(0)
                + k[3] * read(1)
                + k[4] * read(2);
        }
    }
    // Pycvvdp vertical first-row patch (always applied, no parity).
    if dh > 0 && sh >= 2 {
        for x in 0..sw {
            vscratch[x] += src[x] * k[1] + src[sw + x] * k[0];
        }
    }
    // Pycvvdp vertical last-row patch (parity = sh's parity).
    if dh > 0 {
        let last_dy = dh - 1;
        if sh % 2 == 1 && sh >= 2 {
            for x in 0..sw {
                vscratch[last_dy * sw + x] +=
                    src[(sh - 1) * sw + x] * k[3] + src[(sh - 2) * sw + x] * k[4];
            }
        } else if sh.is_multiple_of(2) {
            for x in 0..sw {
                vscratch[last_dy * sw + x] += src[(sh - 1) * sw + x] * k[4];
            }
        }
    }

    // Horizontal pass: zero-pad cols left/right, conv stride 2.
    for dy in 0..dh {
        for dx in 0..dw {
            let cx = 2 * dx as isize;
            let read = |off: isize| -> f32 {
                let c = cx + off;
                if c < 0 || c >= sw as isize {
                    0.0
                } else {
                    vscratch[dy * sw + c as usize]
                }
            };
            dst[dy * dw + dx] = k[0] * read(-2)
                + k[1] * read(-1)
                + k[2] * read(0)
                + k[3] * read(1)
                + k[4] * read(2);
        }
    }
    // Pycvvdp horizontal first-col patch (always applied).
    if dw > 0 && sw >= 2 {
        for dy in 0..dh {
            dst[dy * dw] += vscratch[dy * sw] * k[1] + vscratch[dy * sw + 1] * k[0];
        }
    }
    // Pycvvdp horizontal last-col patch. THE BUG: parity check uses
    // `x.shape[-2]` (the ORIGINAL input's row count, sh) instead of
    // the column count. When sw and sh have different parity, the
    // patch is mis-applied — but we replicate the bug to match
    // goldens. See pycvvdp/lpyr_dec.py:204-209 for the source bug.
    if dw > 0 {
        let last_dx = dw - 1;
        if sh % 2 == 1 && sw >= 2 {
            for dy in 0..dh {
                dst[dy * dw + last_dx] +=
                    vscratch[dy * sw + sw - 1] * k[3] + vscratch[dy * sw + sw - 2] * k[4];
            }
        } else if sh.is_multiple_of(2) {
            for dy in 0..dh {
                dst[dy * dw + last_dx] += vscratch[dy * sw + sw - 1] * k[4];
            }
        }
    }
    (dw, dh)
}

/// 2× upscale: zero-insert at stride 2 + 5-tap separable Gaussian.
///
/// Faithful to cvvdp's `gausspyr_expand` /
/// `interleave_zeros_and_pad`: each axis is expanded by building a
/// length-`(m+4)` buffer with the source values at even positions
/// starting at index 2, the input's first sample replicated at
/// index 0, and the input's last sample replicated at index
/// `m + 2 + (m & 1)`. A 5-tap conv with no padding then yields a
/// length-`m` row. Each axis multiplies output by 2; total
/// reconstruction gain is therefore ×4 across the separable pass.
///
/// `out_w`, `out_h` may be `2*sw`, `2*sh-1`, etc. depending on the
/// parity rule used by the matching reduce — pass the target size
/// explicitly.
pub fn gausspyr_expand_scalar(
    src: &[f32],
    sw: usize,
    sh: usize,
    out_w: usize,
    out_h: usize,
    dst: &mut Vec<f32>,
) {
    debug_assert!(out_w >= 2 * sw - 1 && out_w <= 2 * sw);
    debug_assert!(out_h >= 2 * sh - 1 && out_h <= 2 * sh);

    let k = GAUSS5;

    // Vertical pass: each column of `src` is expanded to `out_h`
    // samples via the zero-interleave + edge-replicate scheme.
    let mut vscratch = vec![0.0_f32; sw * out_h];
    let z_len_v = out_h + 4;
    let mut z_v = vec![0.0_f32; z_len_v];
    let odd_h = out_h & 1;
    let back_idx_v = out_h + 2 + odd_h;
    for x in 0..sw {
        z_v.fill(0.0);
        z_v[0] = src[x];
        for ky in 0..sh {
            z_v[2 + 2 * ky] = src[ky * sw + x];
        }
        z_v[back_idx_v] = src[(sh - 1) * sw + x];
        for y in 0..out_h {
            let sum = k[0] * z_v[y]
                + k[1] * z_v[y + 1]
                + k[2] * z_v[y + 2]
                + k[3] * z_v[y + 3]
                + k[4] * z_v[y + 4];
            vscratch[y * sw + x] = 2.0 * sum;
        }
    }

    // Horizontal pass: each row of vscratch is expanded to `out_w`
    // samples via the same scheme.
    dst.clear();
    dst.resize(out_w * out_h, 0.0);
    let z_len_h = out_w + 4;
    let mut z_h = vec![0.0_f32; z_len_h];
    let odd_w = out_w & 1;
    let back_idx_h = out_w + 2 + odd_w;
    for y in 0..out_h {
        z_h.fill(0.0);
        z_h[0] = vscratch[y * sw];
        for kx in 0..sw {
            z_h[2 + 2 * kx] = vscratch[y * sw + kx];
        }
        z_h[back_idx_h] = vscratch[y * sw + sw - 1];
        for x in 0..out_w {
            let sum = k[0] * z_h[x]
                + k[1] * z_h[x + 1]
                + k[2] * z_h[x + 2]
                + k[3] * z_h[x + 3]
                + k[4] * z_h[x + 4];
            dst[y * out_w + x] = 2.0 * sum;
        }
    }
}

/// One band of a Laplacian pyramid — a flat plane plus its
/// dimensions.
pub struct Band {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Compute the per-band spatial frequencies (cy/deg) for a cvvdp
/// pyramid, matching `lpyr_dec.get_freqs()` from pycvvdp v0.5.4.
///
/// The pyramid height is determined by:
/// - `max_levels = floor(log2(min(w, h))) - 1`
/// - the band index whose frequency drops to or below `min_freq = 0.2`
///   cy/deg (anything lower is below detectable threshold),
///   clamped to `max_levels`.
///
/// Returns a `Vec<f32>` of length `height + 1` (the "base" band plus
/// `height` subsequent reduces), each entry in cy/deg.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pyramid::band_frequencies;
/// use cvvdp_gpu::params::DisplayGeometry;
///
/// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
/// let freqs = band_frequencies(ppd, 1024, 1024);
///
/// // Frequencies are strictly decreasing (finer → coarser).
/// for i in 1..freqs.len() {
///     assert!(freqs[i] < freqs[i - 1]);
/// }
/// // All bands are positive (the MIN_FREQ = 0.2 cy/deg cutoff
/// // truncates the pyramid; entries may approach but stay > 0).
/// assert!(freqs.iter().all(|&f| f > 0.0));
/// // 1024² at standard 4K should produce ≥ 5 bands (the realistic
/// // pyramid depth for ~1 megapixel — actual count depends on PPD
/// // but is bounded by MAX_LEVELS = 9 and floored at 1).
/// assert!(freqs.len() >= 5);
/// ```
#[must_use]
pub fn band_frequencies(ppd: f32, width: usize, height: usize) -> Vec<f32> {
    const MIN_FREQ: f32 = 0.2;
    let min_dim = width.min(height);
    debug_assert!(min_dim >= 2, "pyramid needs at least 2px shortest side");
    let max_levels = (min_dim as f32).log2().floor() as usize - 1;
    let half_ppd = ppd / 2.0;

    // Build the candidate "bands" series cvvdp checks against
    // MIN_FREQ. 15 entries: [1.0, 0.3228, 0.3228/2, …, 0.3228/2^13]
    // each scaled by ppd/2.
    let mut candidate = Vec::with_capacity(15);
    candidate.push(half_ppd);
    for f in 0..14 {
        candidate.push(0.3228_f32 * 2.0_f32.powi(-f) * half_ppd);
    }
    let max_band = candidate
        .iter()
        .position(|&b| b <= MIN_FREQ)
        .unwrap_or(max_levels);
    let n_levels = (max_band + 1).min(max_levels);

    // Final frequencies: [ppd/2] (the base) + n_levels reduces.
    let mut freqs = Vec::with_capacity(n_levels + 1);
    freqs.push(half_ppd);
    for f in 0..n_levels {
        freqs.push(0.3228_f32 * 2.0_f32.powi(-(f as i32)) * half_ppd);
    }
    freqs
}

/// Multi-level Laplacian pyramid decomposition (host scalar). Matches
/// cvvdp's `lpyr_dec.laplacian_pyramid_dec` shape:
///
/// - `out[k] = gauss[k] - expand(gauss[k+1])` for `k < n_levels - 1`
/// - `out[n_levels - 1] = gauss[n_levels - 1]` (the coarsest gaussian)
///
/// `n_levels` defaults to `floor(log2(min(sw, sh)))` if the caller
/// passes `0`. cvvdp uses the same default.
///
/// The Gaussian pyramid is built by repeated `gausspyr_reduce_scalar`.
///
/// # Panics
///
/// Panics if the resolved level count is zero (e.g. `n_levels == 0`
/// together with `sw.min(sh).ilog2() as usize == 0`, which requires
/// `min(sw, sh) < 2`). Debug builds also trip the `n >= 1`
/// `debug_assert!`. Release builds reach the
/// `gauss.pop().expect("at least one level")` line after the
/// gaussian-pyramid loop produced nothing.
#[must_use]
pub fn laplacian_pyramid_dec_scalar(
    src: &[f32],
    sw: usize,
    sh: usize,
    n_levels: usize,
) -> Vec<Band> {
    let n = if n_levels == 0 {
        sw.min(sh).ilog2() as usize
    } else {
        n_levels
    };
    debug_assert!(n >= 1, "pyramid needs at least 1 level");

    // Build the Gaussian pyramid first.
    let mut gauss: Vec<Band> = Vec::with_capacity(n);
    gauss.push(Band {
        w: sw,
        h: sh,
        data: src.to_vec(),
    });
    for k in 1..n {
        let prev = &gauss[k - 1];
        let mut next_data = Vec::new();
        let (nw, nh) = gausspyr_reduce_scalar(&prev.data, prev.w, prev.h, &mut next_data);
        gauss.push(Band {
            w: nw,
            h: nh,
            data: next_data,
        });
    }

    // Build the Laplacian bands: band[k] = gauss[k] - expand(gauss[k+1]).
    let mut bands: Vec<Band> = Vec::with_capacity(n);
    let mut expanded = Vec::new();
    for k in 0..(n - 1) {
        let fine = &gauss[k];
        let coarse = &gauss[k + 1];
        gausspyr_expand_scalar(
            &coarse.data,
            coarse.w,
            coarse.h,
            fine.w,
            fine.h,
            &mut expanded,
        );
        let mut band_data = vec![0.0_f32; fine.w * fine.h];
        for (i, dst) in band_data.iter_mut().enumerate() {
            *dst = fine.data[i] - expanded[i];
        }
        bands.push(Band {
            w: fine.w,
            h: fine.h,
            data: band_data,
        });
    }
    // The coarsest band is the coarsest gaussian — no subtraction.
    let coarsest = gauss.pop().expect("at least one level");
    bands.push(coarsest);
    bands
}

/// Output of `weber_contrast_pyr_dec_scalar`: Weber-contrast bands
/// plus the per-band per-pixel log10 background luminance the CSF
/// stage consumes.
pub struct WeberPyramid {
    pub bands: Vec<Band>,
    /// `log10(L_bkg)` per band — shape matches each band's spatial
    /// dimensions for non-baseband levels, and is a 1×1 spatial
    /// mean for the baseband (cvvdp's `weber_contrast_pyr`).
    pub log_l_bkg: Vec<Vec<f32>>,
}

/// Single-channel Weber-contrast pyramid for cvvdp v0.5.4's
/// `contrast = "weber_g1"` path. Mirrors `weber_contrast_pyr.decompose`:
///
/// For each non-baseband level `k`:
/// 1. `expanded = expand(gauss[k+1])` — same dims as `gauss[k]`.
/// 2. `layer = gauss[k] - expanded` (Laplacian-style difference).
/// 3. `L_bkg = clamp(expanded, min=0.01)` (achromatic gauss; same
///    field used for all 3 DKL channels in the cvvdp pipeline).
/// 4. `contrast = clamp(layer / L_bkg, max=1000)`.
///
/// For the baseband (coarsest level):
/// 1. `layer = gauss[N-1]`.
/// 2. `L_bkg = mean(clamp(gauss_A[N-1], min=0.01))` — a SCALAR
///    (mean over spatial). Both test and ref end up dividing the
///    same image's gauss by its own mean: contrast would otherwise
///    be 1 everywhere.
/// 3. `contrast = layer / L_bkg`.
///
/// `log_l_bkg` stores `log10(L_bkg)` per band — per-pixel for
/// non-baseband, replicated scalar for baseband.
///
/// `l_bkg_channel_data` is the SEPARATE achromatic channel used to
/// compute L_bkg. cvvdp's weber_g1 path uses each image's own
/// achromatic gauss as its L_bkg (i.e. for ref-side bands, use
/// gauss_ref_A; for dist-side bands, use gauss_dist_A). For a
/// callee processing one image at a time, pass the image's own
/// achromatic Gaussian pyramid.
#[must_use]
pub fn weber_contrast_pyr_dec_scalar(
    image_plane: &[f32],
    l_bkg_plane: &[f32],
    sw: usize,
    sh: usize,
    n_levels: usize,
) -> WeberPyramid {
    // Build separate Gaussian pyramids for the image plane and the
    // L_bkg plane. They may be the same plane (single channel) but
    // are passed separately so the caller can use the achromatic
    // channel as L_bkg for chroma bands.
    fn build_pyr(src: &[f32], sw: usize, sh: usize, n: usize) -> Vec<Band> {
        let mut p = Vec::with_capacity(n);
        p.push(Band {
            w: sw,
            h: sh,
            data: src.to_vec(),
        });
        let mut w = sw;
        let mut h = sh;
        for _ in 1..n {
            let mut next = Vec::new();
            let (nw, nh) = gausspyr_reduce_scalar(&p.last().unwrap().data, w, h, &mut next);
            p.push(Band {
                w: nw,
                h: nh,
                data: next,
            });
            w = nw;
            h = nh;
        }
        p
    }

    let n = if n_levels == 0 {
        sw.min(sh).ilog2() as usize
    } else {
        n_levels
    };
    debug_assert!(n >= 1);

    let gauss_img = build_pyr(image_plane, sw, sh, n);
    let gauss_l = build_pyr(l_bkg_plane, sw, sh, n);

    let mut bands: Vec<Band> = Vec::with_capacity(n);
    let mut log_l_bkg: Vec<Vec<f32>> = Vec::with_capacity(n);

    let mut expanded_buf = Vec::new();
    for k in 0..n {
        let is_baseband = k == n - 1;
        let fine = &gauss_img[k];
        let l_fine = &gauss_l[k];
        let n_px = fine.w * fine.h;

        if is_baseband {
            // L_bkg = scalar mean over the achromatic baseband.
            let sum: f32 = l_fine.data.iter().map(|v| v.max(0.01)).sum();
            let l_bkg_mean = sum / l_fine.data.len() as f32;
            let log_l = l_bkg_mean.log10();
            let mut contrast = vec![0.0_f32; n_px];
            for i in 0..n_px {
                contrast[i] = fine.data[i] / l_bkg_mean;
            }
            bands.push(Band {
                w: fine.w,
                h: fine.h,
                data: contrast,
            });
            log_l_bkg.push(vec![log_l; n_px]);
        } else {
            // expanded gauss[k+1] → fine dims
            let coarse = &gauss_l[k + 1];
            gausspyr_expand_scalar(
                &coarse.data,
                coarse.w,
                coarse.h,
                fine.w,
                fine.h,
                &mut expanded_buf,
            );
            // Build the laplacian-style layer from the IMAGE's gauss,
            // not the l_bkg's gauss (the two differ for chroma channels).
            let img_coarse = &gauss_img[k + 1];
            let mut img_expanded = Vec::new();
            gausspyr_expand_scalar(
                &img_coarse.data,
                img_coarse.w,
                img_coarse.h,
                fine.w,
                fine.h,
                &mut img_expanded,
            );

            let mut contrast = vec![0.0_f32; n_px];
            let mut log_l = vec![0.0_f32; n_px];
            for i in 0..n_px {
                let l_bkg = expanded_buf[i].max(0.01);
                let layer = fine.data[i] - img_expanded[i];
                let c = (layer / l_bkg).clamp(-1000.0, 1000.0);
                contrast[i] = c;
                log_l[i] = l_bkg.log10();
            }
            bands.push(Band {
                w: fine.w,
                h: fine.h,
                data: contrast,
            });
            log_l_bkg.push(log_l);
        }
    }
    WeberPyramid { bands, log_l_bkg }
}

/// 2× downscale with the cvvdp 5-tap Gaussian. Per-output-pixel
/// thread; each thread reads 25 source pixels (5 × 5 reflected
/// taps) and emits one f32. Equivalent to two-pass separable conv
/// with symmetric reflection.
///
/// Bug-compatible with pycvvdp's `gausspyr_reduce` (lpyr_dec.py:186):
/// upstream uses zero-pad + parity-aware boundary patches. The
/// horizontal-pass right-column patch checks `x.shape[-2]` (INPUT
/// ROW parity) where the comments say "odd number of columns" —
/// the check is using rows. For mismatched-parity inputs (sw and
/// sh have different parity) the right column gets the wrong patch
/// from pycvvdp's perspective, but since we MATCH that pycvvdp
/// behavior our goldens align. Pure symmetric reflection (what
/// this kernel computes interior) matches pycvvdp's boundary
/// behavior for ALL same-parity inputs (256², 4000×3000 etc.);
/// for mixed-parity inputs we apply a delta correction at the
/// right column to switch from "reflect" to "pycvvdp's bug
/// branch". See `docs/CHROMA_DRIFT_INVESTIGATION.md` tick 206.
#[cube(launch)]
pub fn downscale_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let dy = idx / dw;
    let dx = idx - dy * dw;

    let cy = 2 * (dy as i32);
    let cx = 2 * (dx as i32);
    let sw = src_w as usize;
    let sh = src_h as usize;
    let sh_i = src_h as i32;
    let sw_i = src_w as i32;

    let k0 = f32::new(0.05);
    let k1 = f32::new(0.25);
    let k2 = f32::new(0.40);
    let k3 = f32::new(0.25);
    let k4 = f32::new(0.05);

    // Symmetric reflection at boundaries [0, n). For kernel-radius-2
    // accesses near the edge: one fold covers all cases (the most
    // extreme is cy-2 = -2 → 1, never re-reflects). Same for the
    // upper edge: cy+2 = sh+1 → sh-2.
    let r0_i = if cy - 2 < 0 { -(cy - 2) - 1 } else { cy - 2 };
    let r0 = r0_i as usize;
    let r1_i = if cy - 1 < 0 { -(cy - 1) - 1 } else { cy - 1 };
    let r1 = r1_i as usize;
    let r2 = cy as usize;
    let r3_i = if cy + 1 >= sh_i {
        2 * sh_i - (cy + 1) - 1
    } else {
        cy + 1
    };
    let r3 = r3_i as usize;
    let r4_i = if cy + 2 >= sh_i {
        2 * sh_i - (cy + 2) - 1
    } else {
        cy + 2
    };
    let r4 = r4_i as usize;

    let sx0_i = if cx - 2 < 0 { -(cx - 2) - 1 } else { cx - 2 };
    let sx0 = sx0_i as usize;
    let sx1_i = if cx - 1 < 0 { -(cx - 1) - 1 } else { cx - 1 };
    let sx1 = sx1_i as usize;
    let sx2 = cx as usize;
    let sx3_i = if cx + 1 >= sw_i {
        2 * sw_i - (cx + 1) - 1
    } else {
        cx + 1
    };
    let sx3 = sx3_i as usize;
    let sx4_i = if cx + 2 >= sw_i {
        2 * sw_i - (cx + 2) - 1
    } else {
        cx + 2
    };
    let sx4 = sx4_i as usize;

    let col0 = k0 * src[r0 * sw + sx0]
        + k1 * src[r1 * sw + sx0]
        + k2 * src[r2 * sw + sx0]
        + k3 * src[r3 * sw + sx0]
        + k4 * src[r4 * sw + sx0];
    let col1 = k0 * src[r0 * sw + sx1]
        + k1 * src[r1 * sw + sx1]
        + k2 * src[r2 * sw + sx1]
        + k3 * src[r3 * sw + sx1]
        + k4 * src[r4 * sw + sx1];
    let col2 = k0 * src[r0 * sw + sx2]
        + k1 * src[r1 * sw + sx2]
        + k2 * src[r2 * sw + sx2]
        + k3 * src[r3 * sw + sx2]
        + k4 * src[r4 * sw + sx2];
    let col3 = k0 * src[r0 * sw + sx3]
        + k1 * src[r1 * sw + sx3]
        + k2 * src[r2 * sw + sx3]
        + k3 * src[r3 * sw + sx3]
        + k4 * src[r4 * sw + sx3];
    let col4 = k0 * src[r0 * sw + sx4]
        + k1 * src[r1 * sw + sx4]
        + k2 * src[r2 * sw + sx4]
        + k3 * src[r3 * sw + sx4]
        + k4 * src[r4 * sw + sx4];

    let mut total_v = k0 * col0 + k1 * col1 + k2 * col2 + k3 * col3 + k4 * col4;

    // Tick 206 bug-compat delta. At the right column (dx = dw-1),
    // pycvvdp picks the horizontal patch branch by INPUT ROW
    // parity (sh) — its comment says "columns" but the code uses
    // rows. When sw and sh have the same parity the patch matches
    // what reflect computes; when they differ we add a delta to
    // switch from reflect to pycvvdp's bug branch. Closes the
    // 73×91 odd-dim residual.
    if dx == dw - 1 && sw >= 2 {
        // vscratch values at the right two columns. Use the same
        // reflect-based vertical conv (matches pycvvdp regardless
        // of sh parity for the vertical pass; see analysis in
        // docs/CHROMA_DRIFT_INVESTIGATION.md).
        let vs_last = k0 * src[r0 * sw + sw - 1]
            + k1 * src[r1 * sw + sw - 1]
            + k2 * src[r2 * sw + sw - 1]
            + k3 * src[r3 * sw + sw - 1]
            + k4 * src[r4 * sw + sw - 1];
        let vs_last2 = k0 * src[r0 * sw + sw - 2]
            + k1 * src[r1 * sw + sw - 2]
            + k2 * src[r2 * sw + sw - 2]
            + k3 * src[r3 * sw + sw - 2]
            + k4 * src[r4 * sw + sw - 2];

        let sw_odd = sw % 2 == 1;
        let sh_odd = sh % 2 == 1;
        if sw_odd && !sh_odd {
            // Reflect gave the "odd-W" patch result; pycvvdp picks
            // even-W (using sh's parity). Delta = pycvvdp_even -
            // reflect_odd = -0.05*vs_last2 - 0.20*vs_last.
            total_v += f32::new(-0.05) * vs_last2 + f32::new(-0.20) * vs_last;
        } else if !sw_odd && sh_odd {
            // Reflect gave "even-W"; pycvvdp picks odd-W.
            // Delta = +0.05*vs_last2 + 0.20*vs_last.
            total_v += f32::new(0.05) * vs_last2 + f32::new(0.20) * vs_last;
        }
    }

    dst[idx] = total_v;
}

/// Vertical pass of the cvvdp expand. Produces a `src_w × dst_h`
/// buffer from a `src_w × src_h` input. Each output pixel runs a
/// 5-tap conv of the implicit zero-interleaved column with cvvdp's
/// `interleave_zeros_and_pad` edge-replication scheme:
///
/// - `z = 0`                            → `src[0]` (front edge)
/// - `z = dst_h + 2 + (dst_h & 1)`      → `src[src_h - 1]` (back edge)
/// - `z = 2 + 2k` for `0 ≤ k < src_h`   → `src[k]`
/// - else                                → sparse zero
///
/// Output gain is ×2 here; the horizontal kernel applies the other
/// ×2 for the full ×4 reconstruction gain.
///
/// Validity branch is dodged by mask-multiplying the coefficient:
/// invalid taps contribute 0 to the sum, and the read index falls
/// back to 0 to avoid OOB.
//
// The `0u32.into()` calls in the body bridge between native `u32`
// literals and the cubecl IR type that the `#[cube(launch)]` macro
// expects on each branch of the `if`/`else` chain; clippy flags
// them as `useless_conversion` but removing them breaks the macro.
#[cube(launch)]
#[allow(clippy::useless_conversion)]
pub fn upscale_v_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (src_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let sw = src_w as usize;
    let y = idx / sw;
    let x = idx - y * sw;

    let k0 = f32::new(0.1);
    let k1 = f32::new(0.5);
    let k2 = f32::new(0.8);
    let k3 = f32::new(0.5);
    let k4 = f32::new(0.1);

    let back_v = (dst_h as i32) + 2 + ((dst_h as i32) & 1);
    let sh_i = src_h as i32;
    let zy_base = y as i32;

    let z0 = zy_base;
    let z1 = zy_base + 1;
    let z2 = zy_base + 2;
    let z3 = zy_base + 3;
    let z4 = zy_base + 4;

    let v0 = z0 == 0 || z0 == back_v || (z0 >= 2 && (z0 & 1) == 0 && ((z0 - 2) >> 1) < sh_i);
    let v1 = z1 == 0 || z1 == back_v || (z1 >= 2 && (z1 & 1) == 0 && ((z1 - 2) >> 1) < sh_i);
    let v2 = z2 == 0 || z2 == back_v || (z2 >= 2 && (z2 & 1) == 0 && ((z2 - 2) >> 1) < sh_i);
    let v3 = z3 == 0 || z3 == back_v || (z3 >= 2 && (z3 & 1) == 0 && ((z3 - 2) >> 1) < sh_i);
    let v4 = z4 == 0 || z4 == back_v || (z4 >= 2 && (z4 & 1) == 0 && ((z4 - 2) >> 1) < sh_i);

    let y0 = if z0 == 0 {
        0u32.into()
    } else if z0 == back_v {
        src_h - 1
    } else if z0 >= 2 && (z0 & 1) == 0 {
        ((z0 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let y1 = if z1 == 0 {
        0u32.into()
    } else if z1 == back_v {
        src_h - 1
    } else if z1 >= 2 && (z1 & 1) == 0 {
        ((z1 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let y2 = if z2 == 0 {
        0u32.into()
    } else if z2 == back_v {
        src_h - 1
    } else if z2 >= 2 && (z2 & 1) == 0 {
        ((z2 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let y3 = if z3 == 0 {
        0u32.into()
    } else if z3 == back_v {
        src_h - 1
    } else if z3 >= 2 && (z3 & 1) == 0 {
        ((z3 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let y4 = if z4 == 0 {
        0u32.into()
    } else if z4 == back_v {
        src_h - 1
    } else if z4 >= 2 && (z4 & 1) == 0 {
        ((z4 - 2) >> 1) as u32
    } else {
        0u32.into()
    };

    let m0 = if v0 { f32::new(1.0) } else { f32::new(0.0) };
    let m1 = if v1 { f32::new(1.0) } else { f32::new(0.0) };
    let m2 = if v2 { f32::new(1.0) } else { f32::new(0.0) };
    let m3 = if v3 { f32::new(1.0) } else { f32::new(0.0) };
    let m4 = if v4 { f32::new(1.0) } else { f32::new(0.0) };

    dst[idx] = (k0 * m0) * src[y0 as usize * sw + x]
        + (k1 * m1) * src[y1 as usize * sw + x]
        + (k2 * m2) * src[y2 as usize * sw + x]
        + (k3 * m3) * src[y3 as usize * sw + x]
        + (k4 * m4) * src[y4 as usize * sw + x];
}

/// Horizontal pass of the cvvdp expand. Consumes the vertical
/// kernel's output (`src_w × in_h`) and produces the full
/// `dst_w × in_h` result. The other ×2 of the ×4 reconstruction
/// gain lives here.
//
// See `upscale_v_kernel` for the rationale on the `useless_conversion`
// allow — the `0u32.into()` branches are required by the
// `#[cube(launch)]` macro.
#[cube(launch)]
#[allow(clippy::useless_conversion)]
pub fn upscale_h_kernel(src: &Array<f32>, dst: &mut Array<f32>, src_w: u32, dst_w: u32, in_h: u32) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * in_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let sw = src_w as usize;
    let y = idx / dw;
    let x = idx - y * dw;

    let k0 = f32::new(0.1);
    let k1 = f32::new(0.5);
    let k2 = f32::new(0.8);
    let k3 = f32::new(0.5);
    let k4 = f32::new(0.1);

    let back_h = (dst_w as i32) + 2 + ((dst_w as i32) & 1);
    let sw_i = src_w as i32;
    let zx_base = x as i32;

    let z0 = zx_base;
    let z1 = zx_base + 1;
    let z2 = zx_base + 2;
    let z3 = zx_base + 3;
    let z4 = zx_base + 4;

    let v0 = z0 == 0 || z0 == back_h || (z0 >= 2 && (z0 & 1) == 0 && ((z0 - 2) >> 1) < sw_i);
    let v1 = z1 == 0 || z1 == back_h || (z1 >= 2 && (z1 & 1) == 0 && ((z1 - 2) >> 1) < sw_i);
    let v2 = z2 == 0 || z2 == back_h || (z2 >= 2 && (z2 & 1) == 0 && ((z2 - 2) >> 1) < sw_i);
    let v3 = z3 == 0 || z3 == back_h || (z3 >= 2 && (z3 & 1) == 0 && ((z3 - 2) >> 1) < sw_i);
    let v4 = z4 == 0 || z4 == back_h || (z4 >= 2 && (z4 & 1) == 0 && ((z4 - 2) >> 1) < sw_i);

    let x0 = if z0 == 0 {
        0u32.into()
    } else if z0 == back_h {
        src_w - 1
    } else if z0 >= 2 && (z0 & 1) == 0 {
        ((z0 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x1 = if z1 == 0 {
        0u32.into()
    } else if z1 == back_h {
        src_w - 1
    } else if z1 >= 2 && (z1 & 1) == 0 {
        ((z1 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x2 = if z2 == 0 {
        0u32.into()
    } else if z2 == back_h {
        src_w - 1
    } else if z2 >= 2 && (z2 & 1) == 0 {
        ((z2 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x3 = if z3 == 0 {
        0u32.into()
    } else if z3 == back_h {
        src_w - 1
    } else if z3 >= 2 && (z3 & 1) == 0 {
        ((z3 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x4 = if z4 == 0 {
        0u32.into()
    } else if z4 == back_h {
        src_w - 1
    } else if z4 >= 2 && (z4 & 1) == 0 {
        ((z4 - 2) >> 1) as u32
    } else {
        0u32.into()
    };

    let m0 = if v0 { f32::new(1.0) } else { f32::new(0.0) };
    let m1 = if v1 { f32::new(1.0) } else { f32::new(0.0) };
    let m2 = if v2 { f32::new(1.0) } else { f32::new(0.0) };
    let m3 = if v3 { f32::new(1.0) } else { f32::new(0.0) };
    let m4 = if v4 { f32::new(1.0) } else { f32::new(0.0) };

    let base = y * sw;
    dst[idx] = (k0 * m0) * src[base + x0 as usize]
        + (k1 * m1) * src[base + x1 as usize]
        + (k2 * m2) * src[base + x2 as usize]
        + (k3 * m3) * src[base + x3 as usize]
        + (k4 * m4) * src[base + x4 as usize];
}

/// `band = fine - upscaled_coarse`.
#[cube(launch)]
pub fn subtract_kernel(
    fine: &Array<f32>,
    upscaled_coarse: &Array<f32>,
    band: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    band[idx] = fine[idx] - upscaled_coarse[idx];
}

/// Per-pixel finishing step of the Weber-contrast pyramid for one
/// non-baseband band of one channel. Mirrors the inner body of
/// `weber_contrast_pyr_dec_scalar`:
///
/// ```text
/// L_bkg     = max(expanded_lbkg, 0.01)
/// contrast  = clamp(layer / L_bkg, max = 1000)
/// log_l_bkg = log10(L_bkg)
/// ```
///
/// Inputs:
/// - `layer`         — `gauss_img[k] - expand(gauss_img[k+1])` for the
///                     channel of interest. Caller produces this via
///                     `upscale_v` + `upscale_h` + `subtract` kernels.
/// - `expanded_lbkg` — `expand(gauss_l_bkg[k+1])` (achromatic L_bkg
///                     plane, expanded to the band's spatial size).
///
/// Outputs:
/// - `contrast` — Weber-contrast band ready for CSF weighting +
///                masking.
/// - `log_l_bkg` — per-pixel log10 background luminance for the CSF
///                lookup. All 3 DKL channels share the same field
///                produced by the achromatic-channel run.
///
/// The baseband case (scalar mean L_bkg) is handled separately by
/// host code; the per-band per-pixel mean reduction wouldn't gain
/// from a per-pixel kernel.
#[cube(launch)]
pub fn weber_contrast_compute_kernel(
    layer: &Array<f32>,
    expanded_lbkg: &Array<f32>,
    contrast: &mut Array<f32>,
    log_l_bkg: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let l_min = f32::new(0.01);
    let l_max = f32::new(1000.0);
    let l_min_neg = f32::new(-1000.0);

    let raw_lbkg = expanded_lbkg[idx];
    let l = if raw_lbkg < l_min { l_min } else { raw_lbkg };

    let c_raw = layer[idx] / l;
    let c_hi = if c_raw > l_max { l_max } else { c_raw };
    let c_clamped = if c_hi < l_min_neg { l_min_neg } else { c_hi };

    contrast[idx] = c_clamped;
    // log10(x) via the natural log. cubecl 0.10's `f32::log` is
    // base-2 (per butteraugli-gpu's PORT_STATUS notes); `f32::ln` is
    // natural log. log10(x) = ln(x) * log10(e) = ln(x) * (1/ln(10)).
    log_l_bkg[idx] = f32::ln(l) * f32::new(core::f32::consts::LOG10_E);
}

/// 3-channel fused weber-contrast compute. Single launch produces
/// `contrast` for all three DKL channels plus the shared
/// `log_l_bkg`. Replaces three separate `weber_contrast_compute_kernel`
/// launches per non-baseband pyramid level — the per-pixel
/// `l_bkg_fine → log10` math is now computed once instead of three
/// times.
///
/// Inputs:
/// - `layer_a` / `layer_rg` / `layer_vy` — per-channel Laplacian
///   layers (`fine - upscaled_coarse`) for the three DKL channels.
/// - `expanded_lbkg` — per-pixel achromatic L_bkg (upscaled from
///   `gauss[k+1]` to fine resolution).
/// - `n` — pixel count (must match all four input arrays + outputs).
///
/// Outputs:
/// - `contrast_a` / `contrast_rg` / `contrast_vy` — per-channel
///   Weber-contrast bands.
/// - `log_l_bkg` — per-pixel `log10(max(L_bkg, 0.01))` shared by all
///   three channels in the downstream CSF lookup.
#[cube(launch)]
pub fn weber_contrast_compute_3ch_kernel(
    layer_a: &Array<f32>,
    layer_rg: &Array<f32>,
    layer_vy: &Array<f32>,
    expanded_lbkg: &Array<f32>,
    contrast_a: &mut Array<f32>,
    contrast_rg: &mut Array<f32>,
    contrast_vy: &mut Array<f32>,
    log_l_bkg: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let l_min = f32::new(0.01);
    let l_max = f32::new(1000.0);
    let l_min_neg = f32::new(-1000.0);

    let raw_lbkg = expanded_lbkg[idx];
    let l = if raw_lbkg < l_min { l_min } else { raw_lbkg };

    // Three layers / l with the same upper+lower clamp pattern.
    let c_a_raw = layer_a[idx] / l;
    let c_a_hi = if c_a_raw > l_max { l_max } else { c_a_raw };
    let c_a = if c_a_hi < l_min_neg {
        l_min_neg
    } else {
        c_a_hi
    };
    contrast_a[idx] = c_a;

    let c_rg_raw = layer_rg[idx] / l;
    let c_rg_hi = if c_rg_raw > l_max { l_max } else { c_rg_raw };
    let c_rg = if c_rg_hi < l_min_neg {
        l_min_neg
    } else {
        c_rg_hi
    };
    contrast_rg[idx] = c_rg;

    let c_vy_raw = layer_vy[idx] / l;
    let c_vy_hi = if c_vy_raw > l_max { l_max } else { c_vy_raw };
    let c_vy = if c_vy_hi < l_min_neg {
        l_min_neg
    } else {
        c_vy_hi
    };
    contrast_vy[idx] = c_vy;

    // log10(x) via the natural log — once per pixel rather than
    // three times.
    log_l_bkg[idx] = f32::ln(l) * f32::new(core::f32::consts::LOG10_E);
}

/// Fused subtract + 3-channel Weber-contrast compute.
///
/// Replaces three `subtract_kernel` launches and one
/// `weber_contrast_compute_3ch_kernel` launch per non-baseband
/// pyramid level — and eliminates the per-channel `layer_c`
/// intermediate buffer (the Laplacian-style layer never has to
/// materialize).
///
/// Per-pixel math, per channel `c ∈ {A, RG, VY}`:
///
/// ```text
/// L_bkg       = max(expanded_lbkg, 0.01)
/// contrast[c] = clamp((fine[c] - upscaled_coarse[c]) / L_bkg,
///                     [-1000, 1000])
/// log_l_bkg   = log10(L_bkg)   // shared across all three channels
/// ```
///
/// Inputs:
/// - `fine_a` / `fine_rg` / `fine_vy` — `gauss_ref[k]` planes (the
///   fine-resolution side of the Laplacian).
/// - `upsc_a` / `upsc_rg` / `upsc_vy` — upscaled coarse planes
///   produced by `upscale_v_kernel` + `upscale_h_kernel`.
/// - `expanded_lbkg` — per-pixel achromatic L_bkg from the
///   A-channel upscale.
/// - `n` — pixel count.
///
/// Outputs:
/// - `contrast_a` / `contrast_rg` / `contrast_vy` — per-channel
///   Weber-contrast bands ready for CSF weighting.
/// - `log_l_bkg` — per-pixel `log10(L_bkg)` shared by all three
///   channels.
#[cube(launch)]
pub fn subtract_weber_3ch_kernel(
    fine_a: &Array<f32>,
    fine_rg: &Array<f32>,
    fine_vy: &Array<f32>,
    upsc_a: &Array<f32>,
    upsc_rg: &Array<f32>,
    upsc_vy: &Array<f32>,
    expanded_lbkg: &Array<f32>,
    contrast_a: &mut Array<f32>,
    contrast_rg: &mut Array<f32>,
    contrast_vy: &mut Array<f32>,
    log_l_bkg: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let l_min = f32::new(0.01);
    let l_max = f32::new(1000.0);
    let l_min_neg = f32::new(-1000.0);

    let raw_lbkg = expanded_lbkg[idx];
    let l = if raw_lbkg < l_min { l_min } else { raw_lbkg };

    let layer_a = fine_a[idx] - upsc_a[idx];
    let c_a_raw = layer_a / l;
    let c_a_hi = if c_a_raw > l_max { l_max } else { c_a_raw };
    let c_a = if c_a_hi < l_min_neg {
        l_min_neg
    } else {
        c_a_hi
    };
    contrast_a[idx] = c_a;

    let layer_rg = fine_rg[idx] - upsc_rg[idx];
    let c_rg_raw = layer_rg / l;
    let c_rg_hi = if c_rg_raw > l_max { l_max } else { c_rg_raw };
    let c_rg = if c_rg_hi < l_min_neg {
        l_min_neg
    } else {
        c_rg_hi
    };
    contrast_rg[idx] = c_rg;

    let layer_vy = fine_vy[idx] - upsc_vy[idx];
    let c_vy_raw = layer_vy / l;
    let c_vy_hi = if c_vy_raw > l_max { l_max } else { c_vy_raw };
    let c_vy = if c_vy_hi < l_min_neg {
        l_min_neg
    } else {
        c_vy_hi
    };
    contrast_vy[idx] = c_vy;

    log_l_bkg[idx] = f32::ln(l) * f32::new(core::f32::consts::LOG10_E);
}

// Note (tick 159): I tried adding `upscale_v_3ch_kernel` and
// `upscale_h_3ch_kernel` that read/write 3 channels per thread with
// shared index/mask math. The intent was to halve the upscale
// launch count per level (6 → 2). Result: a ~4% jod regression at
// 12 MP on RTX-class CUDA — the 3ch kernel's per-thread work and
// register footprint reduced warp-level latency hiding more than
// launch overhead was costing us. Kept as a doc breadcrumb so this
// path isn't re-tried without a different angle (e.g. shared-memory
// tiling that actually changes the memory access pattern).

/// Baseband finishing step: scale each of the 3 coarsest Gaussian
/// planes by `inv_l_bkg_mean` (= 1 / mean(max(gauss_a, 0.01))) and
/// emit the 3 baseband bands. Replaces the host-side per-channel
/// read-back → divide → re-upload that the prior baseband path did
/// in `_dispatch_weber_pyramid_gpu`.
///
/// The host still reads back `gauss_a` once to compute the mean
/// (small buffer — ~192 pixels at MAX_LEVELS=9 / 12 MP, single
/// synchronous drain), but the 3 per-channel readbacks +
/// 3 per-channel reuploads become this single GPU launch.
#[cube(launch)]
pub fn baseband_divide_3ch_kernel(
    gauss_a: &Array<f32>,
    gauss_rg: &Array<f32>,
    gauss_vy: &Array<f32>,
    band_a: &mut Array<f32>,
    band_rg: &mut Array<f32>,
    band_vy: &mut Array<f32>,
    inv_l_bkg_mean: f32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    band_a[idx] = gauss_a[idx] * inv_l_bkg_mean;
    band_rg[idx] = gauss_rg[idx] * inv_l_bkg_mean;
    band_vy[idx] = gauss_vy[idx] * inv_l_bkg_mean;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss5_sums_to_one() {
        let s: f32 = GAUSS5.iter().sum();
        assert!((s - 1.0).abs() < 1e-7, "GAUSS5 sums to {s}, not 1.0");
    }

    #[test]
    fn reduce_halves_dimensions() {
        let src = vec![1.0_f32; 16 * 16];
        let mut dst = Vec::new();
        let (dw, dh) = gausspyr_reduce_scalar(&src, 16, 16, &mut dst);
        assert_eq!((dw, dh), (8, 8));
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn reduce_preserves_constant_signal() {
        // GAUSS5 sums to 1; on a constant input every output pixel
        // must equal the constant. Catches coefficient typos and
        // off-by-one edge errors simultaneously.
        let src = vec![2.5_f32; 16 * 16];
        let mut dst = Vec::new();
        gausspyr_reduce_scalar(&src, 16, 16, &mut dst);
        for &v in &dst {
            assert!(
                (v - 2.5).abs() < 1e-6,
                "constant-signal reduce produced {v} ≠ 2.5"
            );
        }
    }

    #[test]
    fn expand_preserves_constant_signal() {
        // With the cvvdp-style explicit edge extension (z[0] =
        // src[0], z[back] = src[-1]), every output sample's kernel
        // hits either the K[0]+K[2]+K[4] subset or the K[1]+K[3]
        // subset of the 5-tap, each summing to 0.5; the ×2 gain per
        // axis recovers full unity. So a constant input must produce
        // a constant output across the entire buffer — boundaries
        // included.
        let src = vec![7.5_f32; 8 * 8];
        let mut dst = Vec::new();
        gausspyr_expand_scalar(&src, 8, 8, 16, 16, &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert!(
                (v - 7.5).abs() < 1e-5,
                "constant-signal expand produced {v} ≠ 7.5 at index {i}"
            );
        }
    }

    #[test]
    fn reduce_then_expand_round_trips_constant() {
        let src = vec![2.0_f32; 16 * 16];
        let mut reduced = Vec::new();
        let (dw, dh) = gausspyr_reduce_scalar(&src, 16, 16, &mut reduced);
        let mut expanded = Vec::new();
        gausspyr_expand_scalar(&reduced, dw, dh, 16, 16, &mut expanded);
        for (i, &v) in expanded.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-5, "round-trip {v} ≠ 2.0 at index {i}");
        }
    }

    #[test]
    fn expand_preserves_constant_odd_target() {
        // Odd target dimension exercises the `out_h & 1` parity branch
        // in the edge-replication index. cvvdp uses div_ceil on
        // reduce, so the inverse target can be one less than 2*sh.
        let src = vec![4.0_f32; 4 * 4];
        let mut dst = Vec::new();
        gausspyr_expand_scalar(&src, 4, 4, 7, 7, &mut dst);
        for (i, &v) in dst.iter().enumerate() {
            assert!((v - 4.0).abs() < 1e-5, "odd-target expand {v} ≠ 4.0 at {i}");
        }
    }
}
