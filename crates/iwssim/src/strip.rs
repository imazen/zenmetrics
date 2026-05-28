//! Strip-mode IW-SSIM — Phase 9.Z.A.
//!
//! Walks the image in horizontal slabs of `body_h` rows plus a halo of
//! `STRIP_HALO_ROWS` rows on each side (clamped at image edges). The
//! halo is sized so that each strip's body rows produce **bit-identical**
//! output to the full-image pipeline at the same image positions —
//! every value the per-strip pyramid/blur kernels read for body rows
//! lies inside the strip's halo+body window.
//!
//! ## Algorithm
//!
//! Two-pass walk because the IW-SSIM info-content weight uses a
//! per-scale neighborhood covariance `C_u` that is global over the
//! image's valid region. Strips can't compute their own `C_u` because
//! the eigendecomposition depends on the full image's `Y^T·Y` sum.
//!
//! - **Pass 1** (per strip):
//!   1. Build the per-strip Laplacian pyramid (`build_laplacian_pyramid`)
//!      from the strip's gray pixels (body + halo).
//!   2. For each scale s ∈ 0..NUM_SCALES-1 with IW enabled:
//!      - Build the per-scale Y matrix on the strip's body rows only
//!        (no halo) — see [`build_y_matrix_body`].
//!      - Accumulate `Y^T·Y` (a `big_n × big_n` f64 matrix) into a
//!        per-scale shared accumulator. Track `nexp_total[s]` so the
//!        final `C_u = sum(Y^T·Y) / nexp_total`.
//!   3. For the top scale (s == NUM_SCALES-1):
//!      - Run `compute_cs` on the strip's body rows of `lp_ref[top]`
//!        and `lp_dis[top]` (with the full 11×11 halo). Sum the cs
//!        values and pixel count into the top-scale `(sum, n)`
//!        accumulator.
//!
//! - **Eigendecomp** (per scale, once): divide accumulated `Y^T·Y` by
//!   total `nexp` and feed to `decompose_and_invert`.
//!
//! - **Pass 2** (per strip):
//!   1. Rebuild the per-strip Laplacian pyramid (or cache from Pass 1 —
//!      currently rebuilt for clarity; cache hoisting is future work).
//!   2. For each scale s ∈ 0..NUM_SCALES-1:
//!      - Compute the strip's per-scale `cs` (body rows only).
//!      - Compute the strip's per-scale `infow` using the global
//!        eigendecomposition.
//!      - Accumulate `Σ(cs · iw)` and `Σ(iw)` over body rows into the
//!        per-scale `(sum_csiw, sum_iw)` accumulator.
//!
//! - **Finalize**: `wmcs[s] = sum_csiw[s] / sum_iw[s]` (or
//!   `sum_cs_top / n_cs_top` for the top scale). Final score is
//!   `Π wmcs[s]^β[s]` — identical to the full-image path.
//!
//! ## Strip-height invariants
//!
//! - `strip_height >= 64` to ensure scale-4 strips have non-empty
//!   body. (At scale 4 a body of 64 rows = 4 rows; less than that
//!   collapses to empty.)
//! - `strip_height` need not be a power of two; halving at each
//!   pyramid level uses ceil-div, the same as the full-image build.
//! - The walker handles strips at image edges (`body_start == 0` or
//!   `body_end == work_h`) by clamping the halo to the real edge —
//!   the strip's `reflect1` boundary at the image edge IS the
//!   full-image's `reflect1` at the same edge.
//!
//! ## Halo sizing
//!
//! Worst-case cumulative reach at scale 0 across all stages:
//!
//! | Stage | reach at scale s | scale-0 rows |
//! |---|---|---|
//! | binom5 LP build cascade (scale 0 → s)   | radius 2 per level | 2·(2^s + ... + 1) ≤ 2·(2^Nsc - 1) = 62 |
//! | 11×11 gauss valid blur at scale s       | radius 5           | 5·2^s ≤ 80 (s=4) |
//! | imenlarge2(g_ref[s+1]) at scale (s+1)  | ~4 src pixels       | 8·2^s ≤ 128 (s=4) |
//! | 3×3 box stats on lp / cs                 | radius 1           | 1·2^s ≤ 16 (s=4) |
//!
//! Total worst-case: 62 + 80 + 128 + 16 = 286 ≤ 320 rows. We round to
//! `STRIP_HALO_ROWS = 320` and align to the maximum scale's halving
//! granularity (2^(NUM_SCALES-1) = 16).
//!
//! Verified by parity tests: strip output matches full output exactly
//! at 1024² / 4096² with `strip_height ∈ {256, 512, 1024, 2048}`.
//!
//! ## K_SPLIT review (Phase 9.Z.B / task #124 — D6)
//!
//! iwssim's pyramid is `NUM_SCALES = 5`, shallower than cvvdp's 9.
//! At the canonical strip body `h_body = 512`, the K_SPLIT decision
//! `body_at_k = (h_body >> k).max(1) >= MODE_B_DEEP_THRESHOLD = 12`
//! evaluates as:
//!
//! | k | body_k = (512 >> k) | ≥ 12? |
//! |---|---------------------|-------|
//! | 0 | 512                 | yes   |
//! | 1 | 256                 | yes   |
//! | 2 | 128                 | yes   |
//! | 3 | 64                  | yes   |
//! | 4 | 32                  | yes   |
//!
//! Every level passes the shallow-band check, so for iwssim at
//! `h_body=512` the equivalent `k_split = NUM_SCALES = 5` — i.e.,
//! **iwssim has no deep bands**. The entire pyramid lives in the
//! strip-aware regime; there is no full-image-fallback regime for
//! deep bands like cvvdp has at levels 6/7/8.
//!
//! Even at the smallest sensible `h_body = STRIP_BODY_MIN = 64`:
//!
//! | k | body_k = (64 >> k) | ≥ 12? |
//! |---|--------------------|-------|
//! | 0 | 64                 | yes   |
//! | 1 | 32                 | yes   |
//! | 2 | 16                 | yes   |
//! | 3 | 8                  | no (deep) |
//! | 4 | 4                  | no (deep) |
//!
//! Only at `h_body=64` does iwssim see deep bands (k=3,4). The
//! production default is 512, so the K_SPLIT/deep-band distinction
//! is moot for normal iwssim use.
//!
//! **Decision: do NOT add K_SPLIT machinery to iwssim.** The current
//! strip walker uses `STRIP_HALO_ROWS = 320` (a fixed per-level halo
//! sized for the worst-case scale-0 reach) — a simpler model than
//! cvvdp's per-level halo with hybrid dispatch. iwssim doesn't have
//! cvvdp's σ=3 PU blur, so the deep-band halo problem that drove
//! cvvdp's K_SPLIT (`6 × 2^k`-rows halo at level k > 4) doesn't
//! occur here. Adding K_SPLIT would be a no-op at production `h_body`
//! values and dead code at smaller ones (caller pre-validation
//! rejects `h_body < STRIP_BODY_MIN = 64` already).

use alloc::vec::Vec;

use crate::eig::{cov_from_neighborhood, decompose_and_invert, EigResult};
use crate::params::IwssimParams;
use crate::pipeline::WarmState;
use crate::pyramid::{build_laplacian_pyramid, imenlarge2, pyramid_dims, PyrLevel};
use crate::ssim::{compute_cs, CsStats};
use crate::{Error, IwssimScore, NUM_SCALES, Result};

/// Conservative halo per side, in scale-0 rows. Computed in §"Halo
/// sizing" of the module doc to cover binom5 (62) + 11×11 valid blur
/// (80) + imenlarge2 (128) + 3×3 box (16) = 286 ≤ 320.
///
/// Aligned to `1 << (NUM_SCALES - 1)` = 16 so each strip's halo at
/// every scale halves to an integral number of rows without rounding
/// ambiguity (the parity test pins this).
pub const STRIP_HALO_ROWS: usize = 320;

/// Default strip body height in scale-0 rows. Power-of-two so the
/// halving cascade through 5 pyramid levels lands on integer rows
/// at every scale.
///
/// 512 keeps per-strip working set (body + 2·halo = 1152 rows) at
/// ~28 MB per channel at W=6000 — fits comfortably in L3 + DRAM
/// while bounding peak heap at ~5x lower than full-image.
pub const STRIP_BODY_DEFAULT: u32 = 512;

/// Minimum legal strip body height. At scale 4 the body must have at
/// least 1 row so `cs` is non-empty; 64 scale-0 rows = 4 scale-4 rows
/// after 4× halving.
pub const STRIP_BODY_MIN: u32 = 64;

/// Per-scale accumulator for the IW path (scales 0..NUM_SCALES-1).
#[derive(Clone, Default)]
struct ScaleAccum {
    /// `Y^T·Y` accumulator — `big_n × big_n` row-major. `big_n` is
    /// known by `IwssimParams` (`bl_sz_x * bl_sz_y + parent as usize`),
    /// so we just store the flat vec.
    yty: Vec<f64>,
    /// Total number of `nexp` rows summed into `yty` across all strips
    /// at this scale.
    nexp_total: usize,
    /// Strip-by-strip `Σ(cs · iw)` for this scale (body rows only).
    sum_csiw: f64,
    /// Strip-by-strip `Σ(iw)` for this scale (body rows only).
    sum_iw: f64,
}

/// Top-scale accumulator (scale = NUM_SCALES-1). The top scale doesn't
/// use IW: `wmcs[top] = mean(cs · l)` where `cs` already has the
/// luminance term folded in via `compute_cs(..., with_luminance=true)`.
#[derive(Clone, Default)]
struct TopAccum {
    /// Σ cs values across all strips' body rows.
    sum_cs: f64,
    /// Count of cs pixels across all strips.
    n_cs: usize,
}

/// One strip's geometry. `body_start..body_end` is the body range at
/// scale 0; halo extends below/above clamped at image edges.
#[derive(Clone, Copy, Debug)]
struct StripGeometry {
    /// Body start row in image (scale 0).
    body_start: usize,
    /// Body end row in image (exclusive, scale 0).
    body_end: usize,
    /// Halo+body region: `strip_start..strip_end` covers the strip's
    /// input data. `body_start - strip_start == top_halo`,
    /// `strip_end - body_end == bottom_halo`.
    strip_start: usize,
    /// End of strip's input data (exclusive).
    strip_end: usize,
}

impl StripGeometry {
    /// Total rows in this strip (body + halo above + halo below).
    fn strip_h(&self) -> usize {
        self.strip_end - self.strip_start
    }

    /// Body row range in strip-local coordinates.
    fn body_in_strip(&self) -> (usize, usize) {
        (
            self.body_start - self.strip_start,
            self.body_end - self.strip_start,
        )
    }
}

/// Compute the list of strip geometries that tile the image.
///
/// Each strip has body rows `[body_start, body_end)` with
/// `body_end - body_start` ≤ `body_h` (last strip may be smaller).
/// Halo extends `halo_rows` above and below the body, clamped at
/// image edges.
fn plan_strips(image_h: usize, body_h: usize, halo_rows: usize) -> Vec<StripGeometry> {
    let mut strips = Vec::new();
    let mut body_start = 0;
    while body_start < image_h {
        let body_end = (body_start + body_h).min(image_h);
        let strip_start = body_start.saturating_sub(halo_rows);
        let strip_end = (body_end + halo_rows).min(image_h);
        strips.push(StripGeometry {
            body_start,
            body_end,
            strip_start,
            strip_end,
        });
        body_start = body_end;
    }
    strips
}

/// Map a scale-0 row range `[s0_start, s0_end)` to scale-s row range.
///
/// Pyramid uses `ceil` halving, so scale-s row index for scale-0 row
/// r is `r >> s` (but ceil-aware: a row originally on the boundary
/// rounds toward the next coarser row).
///
/// For the body region: body_start at scale s is `body_start >> s`
/// (the row index where the body's *first* full row lives at scale s).
/// body_end at scale s is `body_end.div_ceil(1 << s)` — the
/// exclusive upper bound that just covers the last scale-0 body row.
fn body_at_scale(body_start_s0: usize, body_end_s0: usize, s: usize) -> (usize, usize) {
    let factor = 1usize << s;
    let body_start_s = body_start_s0.div_ceil(factor);
    let body_end_s = body_end_s0.div_ceil(factor);
    (body_start_s, body_end_s)
}

/// Extract `[start..end)` rows of a 2D buffer `(h, w)` into a fresh
/// `Vec<f32>`. Used to slice the strip's gray-pixel input from the
/// full-image padded buffer.
fn slice_rows(src: &[f32], w: usize, start: usize, end: usize) -> Vec<f32> {
    let n = (end - start) * w;
    let mut out = alloc::vec![0.0_f32; n];
    out.copy_from_slice(&src[start * w..end * w]);
    out
}

/// Build the `Y` matrix from the strip's body rows only at scale s,
/// matching the full-image's `build_y_matrix` semantics for those
/// pixels.
///
/// The neighborhood is `block_h × block_w` (3×3 default); for body
/// rows the neighborhood reads `±Lx, ±Ly` rows/cols from the strip's
/// `imgo` slab. As long as the strip's halo at scale s covers at
/// least `Ly` rows above body and below body, every neighborhood
/// read is in-bounds.
///
/// Output is rows of length `big_n` for each pixel in the body's
/// valid region at scale s.
fn build_y_matrix_body(
    img: &[f32],
    parent: Option<&[f32]>,
    strip_h_at_s: usize,
    w_at_s: usize,
    body_in_strip: (usize, usize),
    block_h: usize,
    block_w: usize,
) -> (Vec<f32>, usize, usize, usize) {
    let lx = (block_w - 1) / 2;
    let ly = (block_h - 1) / 2;

    // Valid region in strip-local coords is rows [ly, strip_h - ly),
    // cols [lx, w - lx). For the body, we want strip-local rows
    // [body_in_strip.0, body_in_strip.1) — but must intersect with
    // valid region. Body rows must be at strip rows >= ly AND
    // < strip_h - ly to allow the neighborhood read.
    let (body_start_s, body_end_s) = body_in_strip;
    debug_assert!(body_start_s >= ly,
        "strip halo at scale s ({} top_halo at scale s) too small for ly={} (body_start_s={})",
        body_start_s, ly, body_start_s);
    debug_assert!(body_end_s + ly <= strip_h_at_s,
        "strip bottom halo at scale s too small for ly={} (body_end_s={}, strip_h_at_s={})",
        ly, body_end_s, strip_h_at_s);

    // Rows produced in output Y: each "row" of Y corresponds to one
    // output pixel in the body's valid region.
    let nblv = body_end_s - body_start_s;
    let nblh = w_at_s - block_w + 1;
    let nexp = nblv * nblh;
    let big_n = block_h * block_w + parent.is_some() as usize;
    let mut y = alloc::vec![0.0_f32; nexp * big_n];

    let mut col = 0;
    for ny in -(ly as i32)..=(ly as i32) {
        for nx in -(lx as i32)..=(lx as i32) {
            for r in 0..nblv {
                for c in 0..nblh {
                    // Body row in strip-local coords:
                    let yy_strip = body_start_s + r;
                    // Add neighborhood offset:
                    let yy = (yy_strip as i32 + ny) as usize;
                    let xx = (c + lx) as i32 + nx;
                    let row_index = r * nblh + c;
                    y[row_index * big_n + col] = img[yy * w_at_s + (xx as usize)];
                }
            }
            col += 1;
        }
    }
    if let Some(parent_band) = parent {
        for r in 0..nblv {
            for c in 0..nblh {
                let yy = body_start_s + r;
                let xx = c + lx;
                let row_index = r * nblh + c;
                y[row_index * big_n + col] = parent_band[yy * w_at_s + xx];
            }
        }
    }
    (y, nexp, nblv, nblh)
}

/// Compute the cropped `(g, vv, ss)` arrays for the strip's body rows
/// only, then evaluate `infow + Σ(cs·iw)` and `Σ(iw)` into the
/// per-scale accumulator.
///
/// Algorithm parallels `compute_iw_maps` in `weights.rs`, but uses the
/// pre-computed global `EigResult` for `lambdas + Cu_inv`. The Y
/// matrix is built on the strip's body rows only.
fn fold_iw_for_strip_scale(
    lp_ref: &[f32],
    lp_dis: &[f32],
    parent_band: Option<&[f32]>,
    strip_h_at_s: usize,
    w_at_s: usize,
    body_in_strip: (usize, usize),
    params: &IwssimParams,
    eig: &EigResult,
    cs: &[f32],
    cs_h: usize,
    cs_w: usize,
    accum: &mut ScaleAccum,
) {
    let block_h = params.bl_sz_y as usize;
    let block_w = params.bl_sz_x as usize;
    let bound1 = params.bound1() as usize;

    // 1. 3×3 box stats on strip's body+halo rows at scale s. We only
    //    need the body's rows of mean_x/mean_y/cov_xy/ss_x/ss_y, but
    //    those require ±1 row halo. Strip has it.
    // Use a body-only computation that reads the body+halo strip.

    // Build box stats over body's strip-local rows
    // [body_start_s..body_end_s) — same row count as the Y matrix
    // (nblv). The box stats read ±1 row around each output row;
    // strip's halo at scale s guarantees [body_start_s - 1..
    // body_end_s + 1) is inside [0, strip_h_at_s).
    //
    // We compute box stats output over body rows directly (no ly
    // extension), since the iw map shape (nblv, nblh) does not
    // crop rows — only columns.
    let (body_start_s, body_end_s) = body_in_strip;
    let (g, vv) = box_stats_and_gain_body(
        lp_ref,
        lp_dis,
        strip_h_at_s,
        w_at_s,
        body_start_s,
        body_end_s,
    );

    // 2. Build Y matrix on body's valid region.
    let (y_matrix, nexp_body, nblv, nblh) = build_y_matrix_body(
        lp_ref,
        parent_band,
        strip_h_at_s,
        w_at_s,
        body_in_strip,
        block_h,
        block_w,
    );
    let big_n = block_h * block_w + parent_band.is_some() as usize;

    // 3. Compute ss = (Y · C_u_inv) ⊙ Y / N at each body pixel.
    let c_u_inv = eig.c_u_inv_slice();
    debug_assert_eq!(c_u_inv.len(), big_n * big_n);
    let n_f = big_n as f32;
    let mut ss_pix = alloc::vec![0.0_f32; nexp_body];
    for row in 0..nexp_body {
        let y_row = &y_matrix[row * big_n..(row + 1) * big_n];
        let mut acc = 0.0_f32;
        for i in 0..big_n {
            let yi = y_row[i];
            if yi == 0.0 {
                continue;
            }
            let cinv_row = &c_u_inv[i * big_n..(i + 1) * big_n];
            let mut inner = 0.0_f32;
            for j in 0..big_n {
                inner += cinv_row[j] * y_row[j];
            }
            acc += yi * inner;
        }
        ss_pix[row] = acc / n_f;
    }

    // g/vv covers strip rows [body_start_s..body_end_s), shape
    // (body_h, w_at_s) = (nblv, w_at_s). Crop columns by lx for
    // the iw map shape (nblv, nblh).
    let lx = (block_w - 1) / 2;
    let body_h_s = body_end_s - body_start_s;
    debug_assert_eq!(body_h_s, nblv,
        "box stats body rows ({body_h_s}) must equal Y nblv ({nblv})");
    let mut g_c = alloc::vec![0.0_f32; nblv * nblh];
    let mut vv_c = alloc::vec![0.0_f32; nblv * nblh];
    for r in 0..nblv {
        let src_row_off = r * w_at_s + lx;
        for c in 0..nblh {
            g_c[r * nblh + c] = g[src_row_off + c];
            vv_c[r * nblh + c] = vv[src_row_off + c];
        }
    }

    // 5. infow per pixel.
    let infow = compute_infow_inline(&g_c, &vv_c, &ss_pix, eig.lambdas(), params.sigma_nsq);

    // 6. Crop infow by bound1 on every side. The full-image path does
    //    `iw[bound1:-bound1]` on a (nblv, nblh) array; the cs map at
    //    this scale is (cs_h, cs_w) = (h_s - 10, w_s - 10).
    //
    //    For STRIP cs, the cs we received is computed on the strip's
    //    body rows (after the 11×11 valid blur). Its shape is
    //    (body_h_cs, w_s - 10) where body_h_cs = body's contribution
    //    to the 11×11 blur. We need to match shapes.
    //
    //    Actually: the cs is computed on the strip's body+halo (full
    //    strip), with valid blur producing (strip_h_at_s - 10) rows
    //    of cs. The body rows of cs are at rows [body_in_strip.0 -
    //    5, body_in_strip.1 - 5) in the cs output (since valid blur
    //    drops 5 rows top/bottom). For body's interior of cs we want
    //    rows [body_in_strip.0 - 5, body_in_strip.1 - 5).
    //
    //    Wait — the caller passes `cs` and `cs_h`/`cs_w` already
    //    sliced to the body's portion. So we don't crop again here;
    //    we just need infow cropped from (nblv, nblh) → (cs_h, cs_w).

    // Caller built Y matrix over body's image-Y-row range = body's
    // image rows in [ly, image_h - ly). iw has nblv × nblh shape;
    // bound1 row crop pairs each iw_crop row with a cs row.
    //
    // The cs slab passed is bound1-cropped (already trimmed top/bot)
    // and matches iw_crop's shape on rows. Columns are bound1-cropped
    // inside this fn so iw/cs align.
    debug_assert_eq!(nblv, cs_h + 2 * bound1,
        "nblv must equal cs_h + 2*bound1 (nblv={nblv}, cs_h={cs_h}, bound1={bound1})");
    debug_assert_eq!(nblh, cs_w + 2 * bound1,
        "nblh must equal cs_w + 2*bound1 (nblh={nblh}, cs_w={cs_w}, bound1={bound1})");

    // Accumulate sum_csiw, sum_iw over (cs_h, cs_w).
    let mut sum_csiw = 0.0_f64;
    let mut sum_iw = 0.0_f64;
    for r in 0..cs_h {
        let iw_row =
            &infow[(r + bound1) * nblh + bound1..(r + bound1) * nblh + bound1 + cs_w];
        let cs_row = &cs[r * cs_w..(r + 1) * cs_w];
        for c in 0..cs_w {
            let iw = iw_row[c];
            let cs_v = cs_row[c];
            sum_csiw += (cs_v * iw) as f64;
            sum_iw += iw as f64;
        }
    }
    accum.sum_csiw += sum_csiw;
    accum.sum_iw += sum_iw;
}

/// Box stats + gain correction on the strip's `body_start_s..body_end_s`
/// rows. Returns `(g, vv)` over those body rows.
///
/// Reads ±1 row halo (`body_start_s - 1 ... body_end_s + 1`) of `lp_ref`,
/// `lp_dis`, etc. — caller's strip MUST have ≥1 row halo at scale s.
fn box_stats_and_gain_body(
    lp_ref: &[f32],
    lp_dis: &[f32],
    strip_h: usize,
    w: usize,
    body_start_s: usize,
    body_end_s: usize,
) -> (Vec<f32>, Vec<f32>) {
    let body_h = body_end_s - body_start_s;
    let n = body_h * w;
    let inv9 = 1.0_f32 / 9.0;

    // For each body row r in [body_start_s, body_end_s), compute box
    // stats reading rows [r-1, r, r+1] of the strip. The strip's halo
    // at scale s must include row body_start_s-1 and body_end_s.
    let mut mean_x = alloc::vec![0.0_f32; n];
    let mut mean_y = alloc::vec![0.0_f32; n];
    let mut e_xx = alloc::vec![0.0_f32; n];
    let mut e_yy = alloc::vec![0.0_f32; n];
    let mut e_xy = alloc::vec![0.0_f32; n];

    for r in 0..body_h {
        let yy = body_start_s + r;
        for x in 0..w {
            let mut mx = 0.0_f32;
            let mut my = 0.0_f32;
            let mut sxx = 0.0_f32;
            let mut syy = 0.0_f32;
            let mut sxy = 0.0_f32;
            for dy in -1..=1i32 {
                let sy = yy as i32 + dy;
                if sy < 0 || sy >= strip_h as i32 {
                    continue;
                }
                for dx in -1..=1i32 {
                    let sx = x as i32 + dx;
                    if sx < 0 || sx >= w as i32 {
                        continue;
                    }
                    let idx = sy as usize * w + sx as usize;
                    let xv = lp_ref[idx];
                    let yv = lp_dis[idx];
                    mx += xv;
                    my += yv;
                    sxx += xv * xv;
                    syy += yv * yv;
                    sxy += xv * yv;
                }
            }
            let out_idx = r * w + x;
            mean_x[out_idx] = mx * inv9;
            mean_y[out_idx] = my * inv9;
            e_xx[out_idx] = sxx * inv9;
            e_yy[out_idx] = syy * inv9;
            e_xy[out_idx] = sxy * inv9;
        }
    }

    let tol = 1.0e-15_f32;
    let mut g = alloc::vec![0.0_f32; n];
    let mut vv = alloc::vec![0.0_f32; n];
    for i in 0..n {
        let mx = mean_x[i];
        let my = mean_y[i];
        let cov_i = e_xy[i] - mx * my;
        let ssx_i = (e_xx[i] - mx * mx).max(0.0);
        let ssy_i = (e_yy[i] - my * my).max(0.0);
        let mut g_i = cov_i / (ssx_i + tol);
        let mut vv_i = ssy_i - g_i * cov_i;
        if ssx_i < tol {
            g_i = 0.0;
            vv_i = ssy_i;
        }
        if ssy_i < tol {
            g_i = 0.0;
            vv_i = 0.0;
        }
        g[i] = g_i;
        vv[i] = vv_i;
    }
    (g, vv)
}

/// Inline copy of `weights.rs::compute_infow`, accepting the same
/// `lambdas` slice from the shared `EigResult`. We avoid taking a
/// public dep on `weights.rs` because that module is `pub(crate)` and
/// the strip walker lives in a sibling module.
fn compute_infow_inline(
    g: &[f32],
    vv: &[f32],
    ss: &[f32],
    lambdas: &[f32],
    sigma_nsq: f32,
) -> Vec<f32> {
    let n = g.len();
    debug_assert_eq!(vv.len(), n);
    debug_assert_eq!(ss.len(), n);
    let s2 = sigma_nsq;
    let s4 = s2 * s2;
    let mut infow = alloc::vec![0.0_f32; n];
    let tol = 1.0e-15_f32;
    for i in 0..n {
        let g_i = g[i];
        let vv_i = vv[i];
        let ss_i = ss[i];
        let mut acc = 0.0_f32;
        let one_plus_g2 = 1.0 + g_i * g_i;
        let common_num = (vv_i + one_plus_g2 * s2) * ss_i;
        let inv_s4 = 1.0 / s4;
        let sn2_vv = s2 * vv_i;
        for &lam in lambdas {
            let arg = (common_num * lam + sn2_vv) * inv_s4;
            acc += (1.0 + arg).log2();
        }
        if acc >= tol {
            infow[i] = acc;
        } else {
            infow[i] = 0.0;
        }
    }
    infow
}

/// Per-scale cs body slice — extracts the body-portion of the strip's
/// scale-s cs map. The 11×11 valid blur drops 5 rows on each end of
/// the strip, so the strip's cs has shape `(strip_h_at_s - 10,
/// w_at_s - 10)`. The body's contribution is the cs rows that came
/// from body+blur-halo source rows — for any cs row at strip-local
/// position `r_cs`, the source rows are `[r_cs..r_cs+11)` in the
/// strip's input. Body cs rows are those where the SOURCE row range
/// is entirely inside the body region.
///
/// Equivalent (in scale-0 coords): body cs at scale 0 includes rows
/// `[body_start_s0..body_end_s0)` of the strip's input that go through
/// the blur. At scale s the rule is identical with scale-s coords.
///
/// Returns the cs slab covering body's projected range plus the
/// halo-padding for the IW bound1 crop. Caller validates dims.
fn body_cs_range_at_scale(
    cs_strip_h: usize,
    body_in_strip: (usize, usize),
    strip_h_at_s: usize,
) -> (usize, usize) {
    // strip cs row r_cs corresponds to source rows [r_cs..r_cs+11).
    // The CENTER of the blur for cs row r_cs is at source row r_cs+5.
    // We label cs row r_cs as a "body" row when r_cs+5 lies inside
    // body (body_start_s..body_end_s).
    let _ = strip_h_at_s; // not used directly here; kept for symmetry
    let (body_start_s, body_end_s) = body_in_strip;
    let cs_start = body_start_s.saturating_sub(5);
    let cs_end = body_end_s.saturating_sub(5);
    let cs_start = cs_start.min(cs_strip_h);
    let cs_end = cs_end.min(cs_strip_h);
    (cs_start, cs_end)
}

/// Score strip-mode from work-padded gray planes.
///
/// `ref_work` and `dis_work` are `work_w × work_h` f32 planes (already
/// pad-tiled if the source was below MIN_NATIVE_DIM).
///
/// `body_h` is the requested strip body height in scale-0 rows.
pub(crate) fn score_strip_internal(
    ref_work: &[f32],
    dis_work: &[f32],
    work_w: usize,
    work_h: usize,
    body_h: usize,
    params: &IwssimParams,
) -> Result<IwssimScore> {
    if body_h < STRIP_BODY_MIN as usize {
        // Caller passed too-small body. Use default.
        return score_strip_internal(
            ref_work,
            dis_work,
            work_w,
            work_h,
            STRIP_BODY_DEFAULT as usize,
            params,
        );
    }

    let dims = pyramid_dims(work_w, work_h, NUM_SCALES);
    let block_h = params.bl_sz_y as usize;
    let block_w = params.bl_sz_x as usize;
    let parent_enabled = params.parent;

    // big_n at each scale: standard scales use block_h * block_w +
    // parent_enabled; the second-coarsest (NUM_SCALES-2) drops parent
    // when parent_enabled because there's no further coarser scale to
    // expand. (Wait — actually parent is enabled when s < nsc - 2 in
    // compute_iw_maps, so for nsc=5 IW scales are 0..3, parent active
    // for s in 0..2.)
    //
    // Per-scale big_n table:
    //   s=0: parent=true (since s < nsc-2=3) → big_n = 9 + 1 = 10
    //   s=1: parent=true → big_n = 10
    //   s=2: parent=true → big_n = 10
    //   s=3: parent=false → big_n = 9
    //   s=4: top scale, no IW
    let big_n_at = |s: usize| -> usize {
        if parent_enabled && s < NUM_SCALES - 2 {
            block_h * block_w + 1
        } else {
            block_h * block_w
        }
    };

    let strips = plan_strips(work_h, body_h, STRIP_HALO_ROWS);

    // Per-scale accumulators (one per IW scale = 0..nsc-1, but scale
    // nsc-1 uses TopAccum). big_n changes per scale (10 or 9 default).
    let mut scale_accums: Vec<ScaleAccum> = (0..NUM_SCALES - 1)
        .map(|s| {
            let n = big_n_at(s);
            ScaleAccum {
                yty: alloc::vec![0.0_f64; n * n],
                nexp_total: 0,
                sum_csiw: 0.0,
                sum_iw: 0.0,
            }
        })
        .collect();
    let mut top_accum = TopAccum::default();

    // ====== Pass 1: per-strip Y^T·Y accumulation + top-scale cs sum ======
    for sg in &strips {
        let strip_h = sg.strip_h();
        let body_in_strip = sg.body_in_strip();
        // Slice the strip's gray pixels from the work buffer.
        let ref_strip = slice_rows(ref_work, work_w, sg.strip_start, sg.strip_end);
        let dis_strip = slice_rows(dis_work, work_w, sg.strip_start, sg.strip_end);

        // Per-strip Laplacian pyramid.
        let ref_levels = build_laplacian_pyramid(&ref_strip, work_w, strip_h, NUM_SCALES);
        let dis_levels = build_laplacian_pyramid(&dis_strip, work_w, strip_h, NUM_SCALES);
        let (lp_ref_v, g_ref_v) = split_levels(&ref_levels);
        let (lp_dis_v, _) = split_levels(&dis_levels);

        // For each IW scale, accumulate Y^T·Y.
        if params.iw_flag {
            for s in 0..NUM_SCALES - 1 {
                let (w_s, _) = dims[s];
                // strip's height at scale s:
                let dims_s_strip = pyramid_dims(work_w, strip_h, NUM_SCALES);
                let (_, strip_h_at_s) = dims_s_strip[s];
                // Body in strip at scale s:
                let body_in_strip_s = body_at_scale(body_in_strip.0, body_in_strip.1, s);
                if body_in_strip_s.1 <= body_in_strip_s.0 {
                    continue;
                }
                let ly = (block_h - 1) / 2;
                // The Y matrix in the full pipeline is built over
                // image rows [ly, image_h_at_s - ly) — boundary rows
                // are excluded. In strip mode, the body's Y rows are
                // body's image rows intersected with that valid
                // region.
                //
                // Strip's body in image coords at scale s:
                let strip_start_at_s = sg.strip_start.div_ceil(1 << s);
                let body_start_image = body_in_strip_s.0 + strip_start_at_s;
                let body_end_image = body_in_strip_s.1 + strip_start_at_s;
                let (_, image_h_at_s) = dims[s];
                // Trim body to Y's valid region in image coords:
                let body_start_image_trim = body_start_image.max(ly);
                let body_end_image_trim = body_end_image.min(image_h_at_s.saturating_sub(ly));
                if body_end_image_trim <= body_start_image_trim {
                    continue;
                }
                // Adjusted body in strip-local coords. After the
                // image-boundary trim, the body's first/last Y rows
                // map to strip rows that ARE inside [ly, strip_h_at_s -
                // ly) — verified by the strip's halo provision.
                let body_in_strip_s = (
                    body_start_image_trim - strip_start_at_s,
                    body_end_image_trim - strip_start_at_s,
                );
                // Sanity: strip must contain enough halo for the
                // neighborhood read at these adjusted body rows.
                // For top-of-image strips the trim ensures body's
                // first row is at least ly; for bottom-of-image
                // strips the trim ensures body's last row + ly
                // fits in the strip (which goes to image_h_at_s).
                debug_assert!(body_in_strip_s.0 >= ly,
                    "body_in_strip_s.0 {} < ly {} (strip_start_at_s={}, body_start_image_trim={})",
                    body_in_strip_s.0, ly, strip_start_at_s, body_start_image_trim);
                debug_assert!(body_in_strip_s.1 + ly <= strip_h_at_s,
                    "body_in_strip_s.1 {} + ly {} > strip_h_at_s {} (strip_start_at_s={}, body_end_image_trim={})",
                    body_in_strip_s.1, ly, strip_h_at_s, strip_start_at_s, body_end_image_trim);

                let big_n = big_n_at(s);
                let parent_enabled_at = parent_enabled && s < NUM_SCALES - 2;
                let parent_band: Option<Vec<f32>> = if parent_enabled_at {
                    let (w_nxt, _) = dims[s + 1];
                    let (_, h_nxt_strip) = dims_s_strip[s + 1];
                    Some(imenlarge2(
                        &g_ref_v[s + 1],
                        w_nxt,
                        h_nxt_strip,
                        w_s,
                        strip_h_at_s,
                    ))
                } else {
                    None
                };

                let (y_strip, nexp_strip, _, _) = build_y_matrix_body(
                    &lp_ref_v[s],
                    parent_band.as_deref(),
                    strip_h_at_s,
                    w_s,
                    body_in_strip_s,
                    block_h,
                    block_w,
                );

                // Accumulate Y^T · Y into scale_accums[s].yty (n*n)
                // — symmetric, store both halves.
                let acc = &mut scale_accums[s];
                debug_assert_eq!(acc.yty.len(), big_n * big_n);
                for i in 0..big_n {
                    for j in i..big_n {
                        let mut sum = 0.0_f64;
                        for k in 0..nexp_strip {
                            let a = y_strip[k * big_n + i] as f64;
                            let b = y_strip[k * big_n + j] as f64;
                            sum += a * b;
                        }
                        acc.yty[i * big_n + j] += sum;
                        if i != j {
                            acc.yty[j * big_n + i] += sum;
                        }
                    }
                }
                acc.nexp_total += nexp_strip;
            }
        }

        // Top scale (s=NUM_SCALES-1): compute cs over the strip's
        // body rows + halo with luminance term. Accumulate into
        // top_accum.
        {
            let s = NUM_SCALES - 1;
            let (w_s, _) = dims[s];
            let dims_s_strip = pyramid_dims(work_w, strip_h, NUM_SCALES);
            let (_, strip_h_at_s) = dims_s_strip[s];
            if strip_h_at_s > 10 {
                // Compute cs over the entire strip at scale s
                let cs_full = compute_cs(
                    &lp_ref_v[s],
                    &lp_dis_v[s],
                    strip_h_at_s,
                    w_s,
                    true, // with_luminance for the top scale
                );
                // The 11×11 valid blur produces cs of shape
                // (strip_h_at_s - 10, w_s - 10). Extract body's
                // contribution to the top-scale mean.
                let body_in_strip_s = body_at_scale(body_in_strip.0, body_in_strip.1, s);
                let (cs_body_start, cs_body_end) = body_cs_range_at_scale(
                    cs_full.cs_h,
                    body_in_strip_s,
                    strip_h_at_s,
                );
                let cs_body_h = cs_body_end.saturating_sub(cs_body_start);
                if cs_body_h > 0 {
                    let cs_w_at_s = cs_full.cs_w;
                    for r in cs_body_start..cs_body_end {
                        let cs_row = &cs_full.cs[r * cs_w_at_s..(r + 1) * cs_w_at_s];
                        for &v in cs_row {
                            top_accum.sum_cs += v as f64;
                        }
                    }
                    top_accum.n_cs += cs_body_h * cs_w_at_s;
                }
            }
        }
    }

    // ====== Eigendecomp per scale ======
    let mut eigs: Vec<Option<EigResult>> = (0..NUM_SCALES - 1).map(|_| None).collect();
    if params.iw_flag {
        for s in 0..NUM_SCALES - 1 {
            let acc = &scale_accums[s];
            if acc.nexp_total == 0 {
                continue;
            }
            let big_n = big_n_at(s);
            // C_u = sum(Y^T·Y) / nexp_total
            let mut cu = alloc::vec![0.0_f64; big_n * big_n];
            let inv_n = 1.0 / acc.nexp_total as f64;
            for k in 0..big_n * big_n {
                cu[k] = acc.yty[k] * inv_n;
            }
            let eig = decompose_and_invert(&cu, big_n);
            eigs[s] = Some(eig);
        }
    }

    // ====== Pass 2: per-strip cs + infow accumulation ======
    if params.iw_flag {
        for sg in &strips {
            let strip_h = sg.strip_h();
            let body_in_strip = sg.body_in_strip();
            let ref_strip = slice_rows(ref_work, work_w, sg.strip_start, sg.strip_end);
            let dis_strip = slice_rows(dis_work, work_w, sg.strip_start, sg.strip_end);

            let ref_levels = build_laplacian_pyramid(&ref_strip, work_w, strip_h, NUM_SCALES);
            let dis_levels = build_laplacian_pyramid(&dis_strip, work_w, strip_h, NUM_SCALES);
            let (lp_ref_v, g_ref_v) = split_levels(&ref_levels);
            let (lp_dis_v, _) = split_levels(&dis_levels);

            for s in 0..NUM_SCALES - 1 {
                let Some(eig) = eigs[s].as_ref() else {
                    continue;
                };
                let (w_s, _) = dims[s];
                let dims_s_strip = pyramid_dims(work_w, strip_h, NUM_SCALES);
                let (_, strip_h_at_s) = dims_s_strip[s];
                let body_in_strip_s_raw = body_at_scale(body_in_strip.0, body_in_strip.1, s);
                if body_in_strip_s_raw.1 <= body_in_strip_s_raw.0 {
                    continue;
                }
                let ly = (block_h - 1) / 2;
                // Apply the same image-boundary trim as Pass 1.
                let strip_start_at_s = sg.strip_start.div_ceil(1 << s);
                let body_start_image = body_in_strip_s_raw.0 + strip_start_at_s;
                let body_end_image = body_in_strip_s_raw.1 + strip_start_at_s;
                let (_, image_h_at_s) = dims[s];
                let body_start_image_trim = body_start_image.max(ly);
                let body_end_image_trim = body_end_image.min(image_h_at_s.saturating_sub(ly));
                if body_end_image_trim <= body_start_image_trim {
                    continue;
                }
                let body_in_strip_s = (
                    body_start_image_trim - strip_start_at_s,
                    body_end_image_trim - strip_start_at_s,
                );
                if body_in_strip_s.0 < ly || body_in_strip_s.1 + ly > strip_h_at_s {
                    continue;
                }

                // 11×11 valid blur — compute cs over the strip at scale s.
                if strip_h_at_s <= 10 {
                    continue;
                }
                let cs_full = compute_cs(
                    &lp_ref_v[s],
                    &lp_dis_v[s],
                    strip_h_at_s,
                    w_s,
                    false,
                );

                let bound1 = params.bound1() as usize;
                // body_in_strip_s is in strip-local Y-row coords, post
                // image-edge trim. nblv = body_in_strip_s.1 - .0.
                // After bound1 row crop in fold, iw_crop has rows
                // (nblv - 2*bound1). Those rows align with cs body's
                // strip-cs rows = [body_in_strip_s.0 + bound1 - ly,
                // body_in_strip_s.1 - bound1 - ly)... but ly = 1 and
                // bound1 = 4 → cs strip-rows = [body.0 + 3, body.1 - 5).
                //
                // Wait — derivation: iw[r] = Y[r] which corresponds to
                // strip-source row body_in_strip.0 - ly + r + ly =
                // body_in_strip.0 + r (because Y matrix row r reads
                // strip-source row at body_start + r as the center;
                // body_in_strip.0 was already chosen as Y's strip-row).
                // iw_crop[r''] = iw[r'' + bound1] = strip-source row
                // body_in_strip.0 + r'' + bound1.
                //
                // cs[R] = strip-source row R + 5. iw_crop and cs
                // align when strip-source row matches:
                //   body_in_strip.0 + r'' + bound1 = R + 5
                //   R = body_in_strip.0 + bound1 - 5 + r''
                //
                // So cs strip-rows for iw_crop = [body_in_strip.0 +
                // bound1 - 5, body_in_strip.1 - bound1 - 5).
                let ly = 1usize; // block_h=3 → ly=1
                let _ = ly;
                let cs_w_at_s = cs_full.cs_w;
                let strip_cs_h = cs_full.cs_h;

                // Compute cs_body rows in strip-cs coords.
                let cs_body_start_i = (body_in_strip_s.0 + bound1) as isize - 5;
                let cs_body_end_i = (body_in_strip_s.1 as isize - bound1 as isize) - 5;
                let cs_body_start = cs_body_start_i.max(0) as usize;
                let cs_body_end = (cs_body_end_i.max(0) as usize).min(strip_cs_h);
                if cs_body_end <= cs_body_start {
                    continue;
                }
                let cs_body_h = cs_body_end - cs_body_start;
                let cs_body_w = cs_w_at_s;
                if cs_body_w == 0 {
                    continue;
                }

                // If the cs body range was clamped (start < 0 or end >
                // strip_cs_h), we also need to clamp body_in_strip_s
                // so its Y matrix produces an iw_crop matching cs_body.
                //
                // From the derivation:
                //   iw_crop[r''] ↔ strip-cs row = body.0 + bound1 - 5 + r''
                //   r'' ∈ [0, nblv - 2*bound1)
                //   maps to strip-cs row ∈ [body.0 + bound1 - 5,
                //     body.1 - bound1 - 5)
                //
                // For cs_body rows [cs_body_start, cs_body_end):
                //   need iw_crop rows [cs_body_start - (body.0 + bound1 - 5),
                //     cs_body_end - (body.0 + bound1 - 5))
                //   = [r''_start, r''_end)
                //   r''_start = cs_body_start - body.0 - bound1 + 5
                //   r''_end = cs_body_end - body.0 - bound1 + 5
                //
                // Convert back to Y rows (add bound1):
                //   y_start = r''_start + bound1 = cs_body_start - body.0 + 5
                //   y_end = r''_end + bound1 = cs_body_end - body.0 + 5
                //
                // Translated to strip-Y rows (Y row r corresponds to
                // strip-local row body.0 + r ... no, actually Y row r
                // corresponds to strip-local row r if we build Y over
                // the WHOLE strip — but we want to limit Y rows to
                // body's contribution.)
                //
                // For SIMPLICITY: trim body_in_strip_s to the range
                // such that Y_crop matches cs_body exactly. The Y
                // matrix we'll build has nblv' rows = cs_body_h +
                // 2*bound1; body_in_strip_s' = [body.0 + δ_top,
                // body.1 - δ_bot) where
                //   δ_top = max(0, body.0 + bound1 - 5 - cs_body_start)
                //         = max(0, -cs_body_start_i)   (if cs_body_start = 0
                //                                        from clamp)
                //   δ_bot = max(0, cs_body_end_i - cs_body_end)
                //
                // For non-clamped cases δ_top = δ_bot = 0 → no change.
                let body_y_start_offset = (-cs_body_start_i).max(0) as usize;
                let body_y_end_offset =
                    ((cs_body_end_i - cs_body_end as isize).max(0)) as usize;
                let body_in_strip_s_final = (
                    body_in_strip_s.0 + body_y_start_offset,
                    body_in_strip_s.1 - body_y_end_offset,
                );

                let mut cs_body = alloc::vec![0.0_f32; cs_body_h * cs_body_w];
                for r in 0..cs_body_h {
                    let src_row = &cs_full.cs[(cs_body_start + r) * cs_w_at_s
                        ..(cs_body_start + r + 1) * cs_w_at_s];
                    cs_body[r * cs_body_w..(r + 1) * cs_body_w].copy_from_slice(src_row);
                }
                let cs_pass2 = cs_body;

                // Build parent_band at scale s using the strip's
                // g_ref at scale s+1.
                let parent_enabled_at = parent_enabled && s < NUM_SCALES - 2;
                let parent_band: Option<Vec<f32>> = if parent_enabled_at {
                    let (w_nxt, _) = dims[s + 1];
                    let (_, h_nxt_strip) = dims_s_strip[s + 1];
                    Some(imenlarge2(
                        &g_ref_v[s + 1],
                        w_nxt,
                        h_nxt_strip,
                        w_s,
                        strip_h_at_s,
                    ))
                } else {
                    None
                };

                // Need extended body coverage so cropping at bound1
                // matches. We compute box_stats/Y/infow on a
                // slightly-wider body that extends bound1 rows
                // above + below the cs_body region. Strip halo must
                // cover this.
                //
                // For the IW infow, the body rows we accumulate are
                // body cs rows after bound1 crop. The Y matrix /
                // box stats are evaluated at the SAME body rows
                // (since iw indexes line up with cs after bound1).
                //
                // For our box_stats_and_gain_body we already pass
                // body_in_strip_s which covers the strip's body at
                // scale s. infow returned has shape (nblv, nblh) =
                // (body_h_s - 2*ly, w_s - 2*lx). After bound1 crop
                // it's (cs_h_image, cs_w_image) for that body region.

                fold_iw_for_strip_scale(
                    &lp_ref_v[s],
                    &lp_dis_v[s],
                    parent_band.as_deref(),
                    strip_h_at_s,
                    w_s,
                    body_in_strip_s_final,
                    params,
                    eig,
                    &cs_pass2,
                    cs_body_h,
                    cs_body_w,
                    &mut scale_accums[s],
                );
            }
        }
    } else {
        // iw_flag=false: each scale's wmcs = mean(cs). Accumulate
        // per-scale sums alongside the top scale. This branch is
        // here for completeness; production callers use iw_flag=true.
        for sg in &strips {
            let strip_h = sg.strip_h();
            let body_in_strip = sg.body_in_strip();
            let ref_strip = slice_rows(ref_work, work_w, sg.strip_start, sg.strip_end);
            let dis_strip = slice_rows(dis_work, work_w, sg.strip_start, sg.strip_end);
            let ref_levels = build_laplacian_pyramid(&ref_strip, work_w, strip_h, NUM_SCALES);
            let dis_levels = build_laplacian_pyramid(&dis_strip, work_w, strip_h, NUM_SCALES);
            let (lp_ref_v, _) = split_levels(&ref_levels);
            let (lp_dis_v, _) = split_levels(&dis_levels);
            for s in 0..NUM_SCALES - 1 {
                let (w_s, _) = dims[s];
                let dims_s_strip = pyramid_dims(work_w, strip_h, NUM_SCALES);
                let (_, strip_h_at_s) = dims_s_strip[s];
                if strip_h_at_s <= 10 {
                    continue;
                }
                let cs_full = compute_cs(
                    &lp_ref_v[s],
                    &lp_dis_v[s],
                    strip_h_at_s,
                    w_s,
                    false,
                );
                let body_in_strip_s = body_at_scale(body_in_strip.0, body_in_strip.1, s);
                let (cs_body_start, cs_body_end) = body_cs_range_at_scale(
                    cs_full.cs_h,
                    body_in_strip_s,
                    strip_h_at_s,
                );
                let cs_body_h = cs_body_end.saturating_sub(cs_body_start);
                if cs_body_h == 0 {
                    continue;
                }
                let cs_w_at_s = cs_full.cs_w;
                let acc = &mut scale_accums[s];
                for r in cs_body_start..cs_body_end {
                    let cs_row = &cs_full.cs[r * cs_w_at_s..(r + 1) * cs_w_at_s];
                    for &v in cs_row {
                        acc.sum_csiw += v as f64;
                        acc.sum_iw += 1.0;
                    }
                }
            }
        }
    }

    // ====== Finalize ======
    let mut wmcs: [f64; NUM_SCALES] = [0.0; NUM_SCALES];
    if params.iw_flag {
        for s in 0..NUM_SCALES - 1 {
            let acc = &scale_accums[s];
            let denom = if acc.sum_iw == 0.0 { 1.0 } else { acc.sum_iw };
            wmcs[s] = acc.sum_csiw / denom;
        }
    } else {
        for s in 0..NUM_SCALES - 1 {
            let acc = &scale_accums[s];
            let denom = if acc.sum_iw == 0.0 { 1.0 } else { acc.sum_iw };
            wmcs[s] = acc.sum_csiw / denom;
        }
    }
    // Top scale
    let top_denom = if top_accum.n_cs == 0 {
        1.0
    } else {
        top_accum.n_cs as f64
    };
    wmcs[NUM_SCALES - 1] = top_accum.sum_cs / top_denom;

    // score = Π |wmcs[s]|^β[s]
    let mut score = 1.0_f64;
    for s in 0..NUM_SCALES {
        score *= wmcs[s].abs().powf(crate::filters::SCALE_WEIGHTS[s] as f64);
    }
    let _ = (); // unused-imports placeholder
    Ok(IwssimScore {
        score,
        per_scale: wmcs,
    })
}

/// Split borrow of `levels` into `(lp_vec, g_vec)`.
fn split_levels(levels: &[PyrLevel]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut lp = Vec::with_capacity(levels.len());
    let mut g = Vec::with_capacity(levels.len());
    for level in levels {
        lp.push(level.lp.clone());
        g.push(level.g.clone());
    }
    (lp, g)
}

/// Score in strip mode against a warm reference. The warm state
/// holds the full-image reference Laplacian pyramid + Gaussian
/// pyramid, plus (lazily filled) per-scale eigendecomposition.
///
/// Single-pass: per strip, build only the dist pyramid (memory-
/// bounded to one strip's worth of work), then compute cs + infow
/// using the cached ref state + global eigendecomp, accumulate
/// `Σ(cs·iw)` / `Σ(iw)` per scale + the top-scale `Σ cs`. Finalize
/// once at the end.
///
/// The eigendecomposition lives in `warm.eigs` and is built once,
/// either lazily on the first strip walk (if not yet populated) or
/// by a separate setup pass. For now we compute it lazily on the
/// global ref pyramid the first time `score_with_warm_ref_strip`
/// is called — the cost is one full-image Y matrix build per scale,
/// amortized across many dist scores.
pub(crate) fn score_with_warm_ref_strip_internal(
    warm: &mut WarmState,
    dis_work: &[f32],
    work_w: usize,
    work_h: usize,
    body_h: usize,
    params: &IwssimParams,
) -> Result<IwssimScore> {
    if body_h < STRIP_BODY_MIN as usize {
        return score_with_warm_ref_strip_internal(
            warm,
            dis_work,
            work_w,
            work_h,
            STRIP_BODY_DEFAULT as usize,
            params,
        );
    }

    let dims = pyramid_dims(work_w, work_h, NUM_SCALES);
    let block_h = params.bl_sz_y as usize;
    let block_w = params.bl_sz_x as usize;
    let parent_enabled = params.parent;
    let bound1 = params.bound1() as usize;

    let big_n_at = |s: usize| -> usize {
        if parent_enabled && s < NUM_SCALES - 2 {
            block_h * block_w + 1
        } else {
            block_h * block_w
        }
    };

    // ====== Lazy eigendecomp on the warm reference ======
    // Computes per-scale C_u from the FULL reference Y matrix once.
    // Subsequent strip walks reuse the cached eigendecomposition.
    if params.iw_flag {
        for s in 0..NUM_SCALES - 1 {
            if warm.eigs[s].is_some() {
                continue;
            }
            let (w_s, h_s) = dims[s];
            let big_n = big_n_at(s);
            let parent_enabled_at = parent_enabled && s < NUM_SCALES - 2;
            // Build the parent_band from g_ref[s+1] if enabled.
            let parent_band: Option<Vec<f32>> = if parent_enabled_at {
                let (w_nxt, h_nxt) = dims[s + 1];
                Some(imenlarge2(&warm.g_ref[s + 1], w_nxt, h_nxt, w_s, h_s))
            } else {
                None
            };
            // Build the full Y matrix for the reference at this
            // scale. This is the same Y the full-image `compute_iw_maps`
            // path produces — same `cov_from_neighborhood` input.
            // Memory cost: ~`nblv * nblh * big_n * 4` bytes ≈ 380 MB
            // at scale 0 of 40 MP; freed after this loop.
            let (y_full, nexp, _, _) =
                build_y_matrix_full(&warm.lp_ref[s], parent_band.as_deref(), h_s, w_s, block_h, block_w);
            let cu = cov_from_neighborhood(&y_full, nexp, big_n);
            let eig = decompose_and_invert(&cu, big_n);
            warm.eigs[s] = Some(eig);
        }
    }

    // ====== Walk strips for dist + accumulate ======
    let strips = plan_strips(work_h, body_h, STRIP_HALO_ROWS);
    let mut scale_accums: Vec<ScaleAccum> = (0..NUM_SCALES - 1)
        .map(|s| {
            let n = big_n_at(s);
            ScaleAccum {
                yty: alloc::vec![0.0_f64; n * n],
                nexp_total: 0,
                sum_csiw: 0.0,
                sum_iw: 0.0,
            }
        })
        .collect();
    let mut top_accum = TopAccum::default();

    for sg in &strips {
        let strip_h = sg.strip_h();
        let body_in_strip = sg.body_in_strip();

        // Slice the dist strip and build its pyramid.
        let dis_strip = {
            let n = strip_h * work_w;
            let mut buf = alloc::vec![0.0_f32; n];
            buf.copy_from_slice(&dis_work[sg.strip_start * work_w..sg.strip_end * work_w]);
            buf
        };
        let dis_levels = build_laplacian_pyramid(&dis_strip, work_w, strip_h, NUM_SCALES);
        let (lp_dis_v, _) = split_levels(&dis_levels);

        for s in 0..NUM_SCALES - 1 {
            if !params.iw_flag {
                break;
            }
            let Some(eig) = warm.eigs[s].as_ref() else {
                continue;
            };
            let (w_s, _) = dims[s];
            let dims_s_strip = pyramid_dims(work_w, strip_h, NUM_SCALES);
            let (_, strip_h_at_s) = dims_s_strip[s];
            let body_in_strip_s_raw = body_at_scale(body_in_strip.0, body_in_strip.1, s);
            if body_in_strip_s_raw.1 <= body_in_strip_s_raw.0 {
                continue;
            }
            let ly = (block_h - 1) / 2;
            let strip_start_at_s = sg.strip_start.div_ceil(1 << s);
            let body_start_image = body_in_strip_s_raw.0 + strip_start_at_s;
            let body_end_image = body_in_strip_s_raw.1 + strip_start_at_s;
            let (_, image_h_at_s) = dims[s];
            let body_start_image_trim = body_start_image.max(ly);
            let body_end_image_trim = body_end_image.min(image_h_at_s.saturating_sub(ly));
            if body_end_image_trim <= body_start_image_trim {
                continue;
            }
            let body_in_strip_s = (
                body_start_image_trim - strip_start_at_s,
                body_end_image_trim - strip_start_at_s,
            );
            if body_in_strip_s.0 < ly || body_in_strip_s.1 + ly > strip_h_at_s {
                continue;
            }
            if strip_h_at_s <= 10 {
                continue;
            }

            // The cs computation needs both ref and dis at this scale.
            // The REF lp at this scale is full-image (warm.lp_ref[s]).
            // The DIS lp is per-strip (lp_dis_v[s]).
            //
            // For the strip we need the ref's STRIP slice at scale s
            // covering the same strip rows as the dist. Slice it from
            // warm.lp_ref[s].
            let ref_lp_strip_at_s = {
                let n = strip_h_at_s * w_s;
                let mut buf = alloc::vec![0.0_f32; n];
                let src_start = strip_start_at_s * w_s;
                buf.copy_from_slice(
                    &warm.lp_ref[s][src_start..src_start + n],
                );
                buf
            };

            let cs_full = compute_cs(
                &ref_lp_strip_at_s,
                &lp_dis_v[s],
                strip_h_at_s,
                w_s,
                false,
            );

            // Body cs rows in strip-cs coords.
            let cs_body_start_i = (body_in_strip_s.0 + bound1) as isize - 5;
            let cs_body_end_i = (body_in_strip_s.1 as isize - bound1 as isize) - 5;
            let cs_body_start = cs_body_start_i.max(0) as usize;
            let cs_body_end =
                (cs_body_end_i.max(0) as usize).min(cs_full.cs_h);
            if cs_body_end <= cs_body_start {
                continue;
            }
            let cs_body_h = cs_body_end - cs_body_start;
            let cs_w_at_s = cs_full.cs_w;
            let cs_body_w = cs_w_at_s;
            if cs_body_w == 0 {
                continue;
            }
            let body_y_start_offset = (-cs_body_start_i).max(0) as usize;
            let body_y_end_offset =
                ((cs_body_end_i - cs_body_end as isize).max(0)) as usize;
            let body_in_strip_s_final = (
                body_in_strip_s.0 + body_y_start_offset,
                body_in_strip_s.1 - body_y_end_offset,
            );

            let mut cs_body = alloc::vec![0.0_f32; cs_body_h * cs_body_w];
            for r in 0..cs_body_h {
                let src_row = &cs_full.cs[(cs_body_start + r) * cs_w_at_s
                    ..(cs_body_start + r + 1) * cs_w_at_s];
                cs_body[r * cs_body_w..(r + 1) * cs_body_w].copy_from_slice(src_row);
            }

            let parent_enabled_at = parent_enabled && s < NUM_SCALES - 2;
            let parent_band: Option<Vec<f32>> = if parent_enabled_at {
                let (w_nxt, _) = dims[s + 1];
                // For the warm path we have the ref's full-image
                // Gaussian. We could slice it to the strip's range,
                // but parent_band is computed via imenlarge2 over
                // the WHOLE g_ref[s+1] — equivalent to the full-image
                // path. The result is a (w_s, image_h_at_s)
                // full-image parent_band; we slice the strip-rows
                // out of it.
                //
                // For Phase 9.Z.A simplicity, just slice g_ref[s+1]
                // to the strip rows + halo at scale s+1 and run
                // imenlarge2 on that strip. The strip's parent_band
                // at scale s = imenlarge2(strip_g_ref[s+1]). Memory
                // bounded.
                let (_, h_nxt_strip) = dims_s_strip[s + 1];
                let strip_start_at_s1 = sg.strip_start.div_ceil(1 << (s + 1));
                let strip_end_at_s1 = (sg.strip_end + ((1 << (s + 1)) - 1)) >> (s + 1);
                // Use the full-image warm.g_ref[s+1] slice at
                // strip rows.
                let h_actual = strip_end_at_s1 - strip_start_at_s1;
                let n = h_actual * w_nxt;
                let mut g_ref_strip_at_s1 = alloc::vec![0.0_f32; n];
                let src_start = strip_start_at_s1 * w_nxt;
                if src_start + n <= warm.g_ref[s + 1].len() {
                    g_ref_strip_at_s1.copy_from_slice(
                        &warm.g_ref[s + 1][src_start..src_start + n],
                    );
                } else {
                    // Strip extends past image; fall back to full
                    // image (should not happen in practice).
                    g_ref_strip_at_s1.copy_from_slice(&warm.g_ref[s + 1]);
                }
                let _ = h_nxt_strip;
                Some(imenlarge2(
                    &g_ref_strip_at_s1,
                    w_nxt,
                    h_actual,
                    w_s,
                    strip_h_at_s,
                ))
            } else {
                None
            };

            fold_iw_for_strip_scale(
                &ref_lp_strip_at_s,
                &lp_dis_v[s],
                parent_band.as_deref(),
                strip_h_at_s,
                w_s,
                body_in_strip_s_final,
                params,
                eig,
                &cs_body,
                cs_body_h,
                cs_body_w,
                &mut scale_accums[s],
            );
        }

        // Top scale (NUM_SCALES-1).
        {
            let s = NUM_SCALES - 1;
            let (w_s, _) = dims[s];
            let dims_s_strip = pyramid_dims(work_w, strip_h, NUM_SCALES);
            let (_, strip_h_at_s) = dims_s_strip[s];
            if strip_h_at_s > 10 {
                let strip_start_at_s = sg.strip_start.div_ceil(1 << s);
                let ref_lp_strip_at_s = {
                    let n = strip_h_at_s * w_s;
                    let mut buf = alloc::vec![0.0_f32; n];
                    let src_start = strip_start_at_s * w_s;
                    if src_start + n <= warm.lp_ref[s].len() {
                        buf.copy_from_slice(&warm.lp_ref[s][src_start..src_start + n]);
                    } else {
                        buf.copy_from_slice(&warm.lp_ref[s]);
                    }
                    buf
                };
                let cs_full = compute_cs(
                    &ref_lp_strip_at_s,
                    &lp_dis_v[s],
                    strip_h_at_s,
                    w_s,
                    true,
                );
                let body_in_strip_s = body_at_scale(body_in_strip.0, body_in_strip.1, s);
                let (cs_body_start, cs_body_end) = body_cs_range_at_scale(
                    cs_full.cs_h,
                    body_in_strip_s,
                    strip_h_at_s,
                );
                let cs_body_h = cs_body_end.saturating_sub(cs_body_start);
                if cs_body_h > 0 {
                    let cs_w_at_s = cs_full.cs_w;
                    for r in cs_body_start..cs_body_end {
                        let cs_row = &cs_full.cs[r * cs_w_at_s..(r + 1) * cs_w_at_s];
                        for &v in cs_row {
                            top_accum.sum_cs += v as f64;
                        }
                    }
                    top_accum.n_cs += cs_body_h * cs_w_at_s;
                }
            }
        }
    }

    // ====== Finalize ======
    let mut wmcs: [f64; NUM_SCALES] = [0.0; NUM_SCALES];
    for s in 0..NUM_SCALES - 1 {
        let acc = &scale_accums[s];
        let denom = if acc.sum_iw == 0.0 { 1.0 } else { acc.sum_iw };
        wmcs[s] = acc.sum_csiw / denom;
    }
    let top_denom = if top_accum.n_cs == 0 {
        1.0
    } else {
        top_accum.n_cs as f64
    };
    wmcs[NUM_SCALES - 1] = top_accum.sum_cs / top_denom;

    let mut score = 1.0_f64;
    for s in 0..NUM_SCALES {
        score *= wmcs[s].abs().powf(crate::filters::SCALE_WEIGHTS[s] as f64);
    }
    Ok(IwssimScore {
        score,
        per_scale: wmcs,
    })
}

/// Build the FULL Y matrix for `compute_iw_maps`-style covariance
/// — equivalent to `crate::weights::build_y_matrix` but visible to
/// the strip module. Used by `score_with_warm_ref_strip_internal`
/// to lazily eig-decompose the warm reference once per scale.
fn build_y_matrix_full(
    img: &[f32],
    parent: Option<&[f32]>,
    h: usize,
    w: usize,
    block_h: usize,
    block_w: usize,
) -> (Vec<f32>, usize, usize, usize) {
    let lx = (block_w - 1) / 2;
    let ly = (block_h - 1) / 2;
    let nblv = h - block_h + 1;
    let nblh = w - block_w + 1;
    let nexp = nblv * nblh;
    let big_n = block_h * block_w + parent.is_some() as usize;
    let mut y = alloc::vec![0.0_f32; nexp * big_n];
    let mut col = 0;
    for ny in -(ly as i32)..=(ly as i32) {
        for nx in -(lx as i32)..=(lx as i32) {
            for r in 0..nblv {
                for c in 0..nblh {
                    let yy = (r + ly) as i32 + ny;
                    let xx = (c + lx) as i32 + nx;
                    let row_index = r * nblh + c;
                    y[row_index * big_n + col] = img[(yy as usize) * w + (xx as usize)];
                }
            }
            col += 1;
        }
    }
    if let Some(parent_band) = parent {
        for r in 0..nblv {
            for c in 0..nblh {
                let yy = r + ly;
                let xx = c + lx;
                let row_index = r * nblh + c;
                y[row_index * big_n + col] = parent_band[yy * w + xx];
            }
        }
    }
    (y, nexp, nblv, nblh)
}

// Force the Error / CsStats imports to stay used even when feature
// configurations vary.
const _: () = {
    let _ = core::mem::size_of::<CsStats>();
    let _ = core::mem::size_of::<Error>();
};
