// Cosmetic doc-list lints (column-aligned mode tables); allowed per the CI clippy
// policy in .github/workflows/ci.yml.
#![allow(clippy::doc_overindented_list_items, clippy::doc_lazy_continuation)]

//! Task #163 — ENCODER-SEARCH-LOOP wall harness for zensim / SSIMULACRA2 /
//! butteraugli.
//!
//! # Why this exists next to `cpu-wall`
//!
//! `cpu-wall` (task #139/#141) times the six metrics through their **packed
//! sRGB u8** entry points. That is the sweep/fleet-scoring shape. It is NOT the
//! shape a codec quality-search loop has.
//!
//! `jxl-encoder`'s three search loops
//! (`vardct/perceptual_loop.rs`, `vardct/zensim_loop.rs`, `vardct/ssim2_loop.rs`)
//! all hold the reconstruction as **planar linear f32 at a padded stride**
//! (`recon_r/g/b`, stride `padded_width`) and score the full frame. There is no
//! u8 anywhere in those loops. Timing the u8 path therefore answers a different
//! question than "what does one more candidate cost this loop".
//!
//! So this harness measures the loop's real shape, decomposed so the cost model
//!
//!     T(N) = precompute + N * (ingress + warm_compare)
//!
//! can be FIT rather than assumed — one cell per term:
//!
//! | cell                  | what it times                                        |
//! |-----------------------|------------------------------------------------------|
//! | `oneshot`             | build reference + one compare (a cold loop's per-cand)|
//! | `precompute`          | reference-side only — the one-time term               |
//! | `warm_strided`        | one warm compare, distorted read AT PADDED STRIDE     |
//! | `warm_tight_plus_copy`| strided->tight copy of the distorted, then warm compare|
//! | `copy_only`           | the strided->tight copy alone (ingress in isolation)  |
//! | `warm_zeroalloc`      | warm compare through the scratch-recycling entry point |
//! | `loop_n5`             | precompute + 5 warm compares (additivity check)        |
//! | `warm_imgref_packed`  | (ssim2) warm compare, TIGHT interleaved `ImgRef`       |
//! | `warm_imgref_strided` | (ssim2) warm compare, PADDED interleaved `ImgRef`      |
//!
//! `warm_strided` vs `warm_tight_plus_copy` is the load-bearing pair: both
//! butteraugli and zensim accept an arbitrary `stride` on BOTH sides, so the
//! copy is avoidable — but `jxl-encoder`'s `zensim_backend.rs:363-366` performs
//! it on every candidate anyway (and its `padded_width == width` fast path still
//! does a full `copy_from_slice` of all three planes). This measures the bill.
//!
//! fast-ssim2 has no planar entry point, so a PLANAR caller's distorted side must
//! be interleaved into `Vec<[f32; 3]>` every candidate — that is `copy_only` for
//! ssim2, and it is not avoidable through the current API.
//!
//! It DOES, however, take a stride: `impl ToLinearRgb for ImgRef<'_, [f32; 3]>`
//! walks `self.pixels()` (`fast-ssim2/src/input.rs:261-266`), which honours
//! `imgref`'s stride. `warm_imgref_packed` vs `warm_imgref_strided` is the pair
//! that proves it — the same entry point handed a tight and a padded interleaved
//! buffer, with no caller-side repack in either. Measured 2026-08-30 on 0.8.2:
//! ratio 0.984–1.008 over 64²–4096², and the scores are bit-identical
//! (`PARITY:imgref_stride_parity_abs_delta` = 0).
//!
//! These two cells exist because the FIRST pass of this study concluded from
//! reading `ToLinearRgb` that fast-ssim2 could not take stride, and published
//! that. It never ran a strided input — every ssim2 cell pre-flattened planar
//! data into a packed `Img::new(...)`. Reading an implementation is not
//! measuring it; if a claim is about runtime behaviour, there must be a cell.
//!
//! # Usage
//!   loop-wall <size_label> <out_tsv> [metric_filter]
//!   size_label ∈ { 64 128 256 512 1024 2K 4096 }
//!   metric_filter ∈ { zensim ssim2 butter }
//!
//! Rows: size_label metric mode w h stride mean_ns mean_ms n_rounds score
//!
//! Release build, NO `-C target-cpu=native` (runtime SIMD dispatch is what
//! callers actually get — CLAUDE.md).

use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Duration;

use zenbench::prelude::*;

/// Extra pixels of row padding, mimicking the encoder's SIMD-padded planes.
/// `jxl-encoder` pads rows out to a block multiple, so `padded_width > width`
/// is the NORMAL case in the loops we are modelling — measuring only the tight
/// case would hide the ingress cost entirely.
const PAD: usize = 16;

/// Candidate count for the additivity check. `jxl-encoder`'s effort ladder
/// (`effort.rs:1428-1447` x `lossy_search_seeds_for`) yields 3 compares at e8
/// and 5 at e9/e10 — the production-typical band. 5 is the e9/e10 point.
const LOOP_N: usize = 5;

// ---------------------------------------------------------------------------
// Synthetic planar linear-f32 input.
//
// Values are kept in (0, 1] and finite: butteraugli's planar entry points run
// `check_finite_f32` over every plane and reject NaN/Inf, and a zero-variance
// plane would make the SSIM cross terms degenerate.
// ---------------------------------------------------------------------------
fn synth_planar(width: usize, height: usize, stride: usize) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let n = stride * height;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for y in 0..height {
        for x in 0..width {
            let i = y * stride + x;
            // Same procedural pattern as `cpu-wall`'s `synth_pair`, mapped to
            // linear [0,1] — so the two harnesses see the same content shape.
            r[i] = ((((x * 17 + y * 5) % 251) as f32) + 40.0) / 291.0;
            g[i] = ((((x * 11 + y * 13) % 247) as f32) + 40.0) / 287.0;
            b[i] = ((((x * 7 + y * 19) % 241) as f32) + 40.0) / 281.0;
        }
        // Padding columns get a benign finite value; they are never read by a
        // correct implementation, but butteraugli's finiteness scan walks
        // `stride * height`, so they must not be NaN.
        for x in width..stride {
            let i = y * stride + x;
            r[i] = 0.5;
            g[i] = 0.5;
            b[i] = 0.5;
        }
    }
    let mut dr = r.clone();
    let mut dg = g.clone();
    let mut db = b.clone();
    for y in 0..height {
        for x in 0..width {
            let i = y * stride + x;
            dr[i] = (dr[i] - 0.027).clamp(1.0e-4, 1.0);
            dg[i] = (dg[i] - 0.014).clamp(1.0e-4, 1.0);
            db[i] = (db[i] + 0.042).clamp(1.0e-4, 1.0);
        }
    }
    ([r, g, b], [dr, dg, db])
}

/// Strided -> tight copy of one plane. Byte-for-byte the shape
/// `jxl-encoder`'s `zensim_backend.rs:251` / `cvvdp_backend.rs` use, including
/// the `padded_width == width` whole-buffer fast path.
fn copy_strided_into(dst: &mut [f32], src: &[f32], stride: usize, width: usize, height: usize) {
    if stride == width {
        dst.copy_from_slice(&src[..width * height]);
        return;
    }
    for y in 0..height {
        dst[y * width..y * width + width].copy_from_slice(&src[y * stride..y * stride + width]);
    }
}

/// Strided planar -> packed interleaved `[f32; 3]`. The repack `ssim2_loop.rs`
/// does per candidate (`:319-325`) because fast-ssim2 has no planar entry.
fn interleave_into(
    dst: &mut Vec<[f32; 3]>,
    p: &[Vec<f32>; 3],
    stride: usize,
    width: usize,
    height: usize,
) {
    dst.clear();
    for y in 0..height {
        let row = y * stride;
        for x in 0..width {
            dst.push([p[0][row + x], p[1][row + x], p[2][row + x]]);
        }
    }
}

fn size_dims(label: &str) -> Option<(usize, usize)> {
    match label {
        "64" => Some((64, 64)),
        "128" => Some((128, 128)),
        "256" => Some((256, 256)),
        "512" => Some((512, 512)),
        "1024" => Some((1024, 1024)),
        "2K" => Some((2048, 2048)),
        "4096" => Some((4096, 4096)),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "usage: loop-wall <size_label> <out_tsv> [metric_filter]\n  size_label: 64 128 256 512 1024 2K 4096\n  metric_filter: zensim ssim2 butter"
        );
        std::process::exit(64);
    }
    let label = args[1].clone();
    let out_tsv = args[2].clone();
    let metric_filter: Option<String> = args.get(3).cloned();
    let want = |m: &str| metric_filter.as_deref().map(|f| f == m).unwrap_or(true);
    let (w, h) = match size_dims(&label) {
        Some(d) => d,
        None => {
            eprintln!("bad size label: {label}");
            std::process::exit(64);
        }
    };
    let stride = w + PAD;

    let (refp, distp) = synth_planar(w, h, stride);

    let (group_wall, per_cell_max_time, min_rounds) = match label.as_str() {
        "64" | "128" => (Duration::from_secs(240), Duration::from_secs(6), 32usize),
        "256" => (Duration::from_secs(300), Duration::from_secs(8), 24),
        "512" => (Duration::from_secs(420), Duration::from_secs(10), 20),
        "1024" => (Duration::from_secs(600), Duration::from_secs(14), 16),
        "2K" => (Duration::from_secs(900), Duration::from_secs(25), 12),
        "4096" => (Duration::from_secs(1500), Duration::from_secs(60), 8),
        _ => (Duration::from_secs(420), Duration::from_secs(10), 16),
    };

    let mut scores: Vec<(String, f64)> = Vec::new();

    let build = |suite: &mut zenbench::prelude::Suite| {
        suite.group(format!("loop_wall_{label}"), |g| {
            g.config().max_wall_time(group_wall);
            g.config().max_time(per_cell_max_time);
            g.config().min_rounds(min_rounds);

            // ---------------- butteraugli ----------------
            if want("butter") {
                use butteraugli::{ButteraugliParams, ButteraugliReference};
                let p = ButteraugliParams::new();

                g.bench("butter__oneshot", {
                    let (rp, dp, p) = (refp.clone(), distp.clone(), p.clone());
                    move |b| {
                        b.iter(|| {
                            let pre = ButteraugliReference::new_linear_planar(
                                &rp[0],
                                &rp[1],
                                &rp[2],
                                w,
                                h,
                                stride,
                                p.clone(),
                            )
                            .unwrap();
                            zenbench::black_box(
                                pre.compare_linear_planar(&dp[0], &dp[1], &dp[2], stride)
                                    .unwrap()
                                    .score,
                            )
                        })
                    }
                });
                g.bench("butter__precompute", {
                    let (rp, p) = (refp.clone(), p.clone());
                    move |b| {
                        b.iter(|| {
                            zenbench::black_box(
                                ButteraugliReference::new_linear_planar(
                                    &rp[0],
                                    &rp[1],
                                    &rp[2],
                                    w,
                                    h,
                                    stride,
                                    p.clone(),
                                )
                                .unwrap()
                                .width(),
                            )
                        })
                    }
                });
                g.bench("butter__warm_strided", {
                    let (rp, dp, p) = (refp.clone(), distp.clone(), p.clone());
                    let pre = ButteraugliReference::new_linear_planar(
                        &rp[0], &rp[1], &rp[2], w, h, stride, p,
                    )
                    .unwrap();
                    move |b| {
                        b.iter(|| {
                            zenbench::black_box(
                                pre.compare_linear_planar(&dp[0], &dp[1], &dp[2], stride)
                                    .unwrap()
                                    .score,
                            )
                        })
                    }
                });
                // The zero-allocation steady-state entry: diffmap recycled into
                // a caller-owned Vec instead of freshly allocated per compare.
                g.bench("butter__warm_zeroalloc", {
                    let (rp, dp, p) = (refp.clone(), distp.clone(), p.clone());
                    let pre = ButteraugliReference::new_linear_planar(
                        &rp[0], &rp[1], &rp[2], w, h, stride, p,
                    )
                    .unwrap();
                    let mut dm: Vec<f32> = Vec::new();
                    move |b| {
                        b.iter(|| {
                            zenbench::black_box(
                                pre.compare_linear_planar_into(
                                    &dp[0], &dp[1], &dp[2], stride, &mut dm,
                                )
                                .unwrap()
                                .0,
                            )
                        })
                    }
                });
                g.bench("butter__loop_n5", {
                    let (rp, dp, p) = (refp.clone(), distp.clone(), p.clone());
                    move |b| {
                        b.iter(|| {
                            let pre = ButteraugliReference::new_linear_planar(
                                &rp[0],
                                &rp[1],
                                &rp[2],
                                w,
                                h,
                                stride,
                                p.clone(),
                            )
                            .unwrap();
                            let mut dm: Vec<f32> = Vec::new();
                            let mut acc = 0.0f64;
                            for _ in 0..LOOP_N {
                                acc += pre
                                    .compare_linear_planar_into(
                                        &dp[0], &dp[1], &dp[2], stride, &mut dm,
                                    )
                                    .unwrap()
                                    .0;
                            }
                            zenbench::black_box(acc)
                        })
                    }
                });
            }

            // ---------------- zensim ----------------
            if want("zensim") {
                use zensim::{DiffmapOptions, Zensim, ZensimProfile};
                g.bench("zensim__oneshot", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    move |b| {
                        let z = Zensim::new(ZensimProfile::latest_preview());
                        b.iter(|| {
                            let pre = z
                                .precompute_reference_linear_planar(
                                    [&rp[0], &rp[1], &rp[2]],
                                    w,
                                    h,
                                    stride,
                                )
                                .unwrap();
                            zenbench::black_box(
                                z.compute_with_ref_and_diffmap_linear_planar(
                                    &pre,
                                    [&dp[0], &dp[1], &dp[2]],
                                    w,
                                    h,
                                    stride,
                                    DiffmapOptions::default(),
                                )
                                .unwrap()
                                .score(),
                            )
                        })
                    }
                });
                g.bench("zensim__precompute", {
                    let rp = refp.clone();
                    move |b| {
                        let z = Zensim::new(ZensimProfile::latest_preview());
                        b.iter(|| {
                            zenbench::black_box(
                                z.precompute_reference_linear_planar(
                                    [&rp[0], &rp[1], &rp[2]],
                                    w,
                                    h,
                                    stride,
                                )
                                .unwrap()
                                .width(),
                            )
                        })
                    }
                });
                // What zensim_loop.rs actually does: pass the padded stride
                // straight through, no copy.
                g.bench("zensim__warm_strided", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    let z = Zensim::new(ZensimProfile::latest_preview());
                    let pre = z
                        .precompute_reference_linear_planar([&rp[0], &rp[1], &rp[2]], w, h, stride)
                        .unwrap();
                    move |b| {
                        b.iter(|| {
                            zenbench::black_box(
                                z.compute_with_ref_and_diffmap_linear_planar(
                                    &pre,
                                    [&dp[0], &dp[1], &dp[2]],
                                    w,
                                    h,
                                    stride,
                                    DiffmapOptions::default(),
                                )
                                .unwrap()
                                .score(),
                            )
                        })
                    }
                });
                // What zensim_backend.rs does: copy all three distorted planes
                // strided -> tight into persistent scratch, then call with
                // stride == width. The copy is pure overhead: the callee took
                // a stride argument all along.
                g.bench("zensim__warm_tight_plus_copy", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    let z = Zensim::new(ZensimProfile::latest_preview());
                    // Reference built TIGHT here, matching the backend.
                    let mut tr = [
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                    ];
                    for c in 0..3 {
                        copy_strided_into(&mut tr[c], &rp[c], stride, w, h);
                    }
                    let pre = z
                        .precompute_reference_linear_planar([&tr[0], &tr[1], &tr[2]], w, h, w)
                        .unwrap();
                    let mut sc = [
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                    ];
                    move |b| {
                        b.iter(|| {
                            for c in 0..3 {
                                copy_strided_into(&mut sc[c], &dp[c], stride, w, h);
                            }
                            zenbench::black_box(
                                z.compute_with_ref_and_diffmap_linear_planar(
                                    &pre,
                                    [&sc[0], &sc[1], &sc[2]],
                                    w,
                                    h,
                                    w,
                                    DiffmapOptions::default(),
                                )
                                .unwrap()
                                .score(),
                            )
                        })
                    }
                });
                g.bench("zensim__copy_only", {
                    let dp = distp.clone();
                    let mut sc = [
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                        vec![0.0f32; w * h],
                    ];
                    move |b| {
                        b.iter(|| {
                            for c in 0..3 {
                                copy_strided_into(&mut sc[c], &dp[c], stride, w, h);
                            }
                            zenbench::black_box(sc[0][0] + sc[1][0] + sc[2][0])
                        })
                    }
                });
                g.bench("zensim__loop_n5", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    move |b| {
                        let z = Zensim::new(ZensimProfile::latest_preview());
                        b.iter(|| {
                            let pre = z
                                .precompute_reference_linear_planar(
                                    [&rp[0], &rp[1], &rp[2]],
                                    w,
                                    h,
                                    stride,
                                )
                                .unwrap();
                            let mut acc = 0.0f64;
                            for _ in 0..LOOP_N {
                                acc += z
                                    .compute_with_ref_and_diffmap_linear_planar(
                                        &pre,
                                        [&dp[0], &dp[1], &dp[2]],
                                        w,
                                        h,
                                        stride,
                                        DiffmapOptions::default(),
                                    )
                                    .unwrap()
                                    .score();
                            }
                            zenbench::black_box(acc)
                        })
                    }
                });
            }

            // ---------------- ssim2 ----------------
            if want("ssim2") {
                use fast_ssim2::Ssimulacra2Reference;
                use imgref::Img;
                g.bench("ssim2__oneshot", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    move |b| {
                        let mut ri: Vec<[f32; 3]> = Vec::new();
                        let mut di: Vec<[f32; 3]> = Vec::new();
                        b.iter(|| {
                            interleave_into(&mut ri, &rp, stride, w, h);
                            interleave_into(&mut di, &dp, stride, w, h);
                            let pre =
                                Ssimulacra2Reference::new(Img::new(ri.as_slice(), w, h)).unwrap();
                            zenbench::black_box(pre.compare(Img::new(di.as_slice(), w, h)).unwrap())
                        })
                    }
                });
                g.bench("ssim2__precompute", {
                    let rp = refp.clone();
                    move |b| {
                        let mut ri: Vec<[f32; 3]> = Vec::new();
                        b.iter(|| {
                            interleave_into(&mut ri, &rp, stride, w, h);
                            zenbench::black_box(
                                Ssimulacra2Reference::new(Img::new(ri.as_slice(), w, h))
                                    .unwrap()
                                    .width(),
                            )
                        })
                    }
                });
                // ssim2_loop.rs's exact per-candidate shape: repack the strided
                // planar reconstruction into packed [f32;3], then `compare`.
                g.bench("ssim2__warm_strided", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    let mut ri: Vec<[f32; 3]> = Vec::new();
                    interleave_into(&mut ri, &rp, stride, w, h);
                    let pre = Ssimulacra2Reference::new(Img::new(ri.as_slice(), w, h)).unwrap();
                    move |b| {
                        let mut di: Vec<[f32; 3]> = Vec::new();
                        b.iter(|| {
                            interleave_into(&mut di, &dp, stride, w, h);
                            zenbench::black_box(pre.compare(Img::new(di.as_slice(), w, h)).unwrap())
                        })
                    }
                });
                // ---- CORRECTION 2026-08-30: the stride A/B the first pass missed.
                //
                // The cells above pre-flatten planar -> a PACKED interleaved Vec and
                // hand fast-ssim2 a tight `Img::new(...)`. So they never exercised a
                // strided input at all, and the first pass wrongly concluded from
                // reading `ToLinearRgb` that fast-ssim2 cannot take stride. It CAN:
                // `impl ToLinearRgb for ImgRef<'_, [f32; 3]>` walks `self.pixels()`,
                // which honours the stride (`input.rs:261-266`).
                //
                // These two cells isolate the real question the workspace stride rule
                // asks — "does accepting a stride cost anything ON THE PACKED PATH?" —
                // by handing the SAME entry point an interleaved buffer twice: once
                // tight, once padded. The caller does NO repack in either; the buffer
                // is built outside the timed region, exactly as a caller holding
                // interleaved pixels would have it.
                g.bench("ssim2__warm_imgref_packed", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    let mut ri: Vec<[f32; 3]> = Vec::new();
                    interleave_into(&mut ri, &rp, stride, w, h);
                    let pre = Ssimulacra2Reference::new(Img::new(ri.as_slice(), w, h)).unwrap();
                    // Tight interleaved distorted: stride == width.
                    let mut di: Vec<[f32; 3]> = Vec::new();
                    interleave_into(&mut di, &dp, stride, w, h);
                    move |b| {
                        b.iter(|| {
                            zenbench::black_box(pre.compare(Img::new(di.as_slice(), w, h)).unwrap())
                        })
                    }
                });
                g.bench("ssim2__warm_imgref_strided", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    let mut ri: Vec<[f32; 3]> = Vec::new();
                    interleave_into(&mut ri, &rp, stride, w, h);
                    let pre = Ssimulacra2Reference::new(Img::new(ri.as_slice(), w, h)).unwrap();
                    // PADDED interleaved distorted: a `stride`-wide row of [f32; 3]
                    // holding `w` real pixels, then PAD columns of filler that a
                    // correct stride-aware ingress must never read.
                    let mut di_pad: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; stride * h];
                    for y in 0..h {
                        for x in 0..w {
                            let s = y * stride + x;
                            di_pad[y * stride + x] = [dp[0][s], dp[1][s], dp[2][s]];
                        }
                    }
                    move |b| {
                        b.iter(|| {
                            zenbench::black_box(
                                pre.compare(imgref::Img::new_stride(
                                    di_pad.as_slice(),
                                    w,
                                    h,
                                    stride,
                                ))
                                .unwrap(),
                            )
                        })
                    }
                });
                // Same, through the scratch-recycling entry (zero Vec allocs
                // after the first call).
                g.bench("ssim2__warm_zeroalloc", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    let mut ri: Vec<[f32; 3]> = Vec::new();
                    interleave_into(&mut ri, &rp, stride, w, h);
                    let pre = Ssimulacra2Reference::new(Img::new(ri.as_slice(), w, h)).unwrap();
                    let mut ctx = pre.compare_context();
                    move |b| {
                        let mut di: Vec<[f32; 3]> = Vec::new();
                        b.iter(|| {
                            interleave_into(&mut di, &dp, stride, w, h);
                            zenbench::black_box(
                                pre.compare_with(&mut ctx, Img::new(di.as_slice(), w, h))
                                    .unwrap(),
                            )
                        })
                    }
                });
                // The repack alone — fast-ssim2 has no planar entry point, so
                // unlike zensim's copy this one is NOT avoidable today.
                g.bench("ssim2__copy_only", {
                    let dp = distp.clone();
                    move |b| {
                        let mut di: Vec<[f32; 3]> = Vec::new();
                        b.iter(|| {
                            interleave_into(&mut di, &dp, stride, w, h);
                            zenbench::black_box(di[0][0])
                        })
                    }
                });
                g.bench("ssim2__loop_n5", {
                    let (rp, dp) = (refp.clone(), distp.clone());
                    move |b| {
                        let mut ri: Vec<[f32; 3]> = Vec::new();
                        let mut di: Vec<[f32; 3]> = Vec::new();
                        b.iter(|| {
                            interleave_into(&mut ri, &rp, stride, w, h);
                            let pre =
                                Ssimulacra2Reference::new(Img::new(ri.as_slice(), w, h)).unwrap();
                            let mut ctx = pre.compare_context();
                            let mut acc = 0.0f64;
                            for _ in 0..LOOP_N {
                                interleave_into(&mut di, &dp, stride, w, h);
                                acc += pre
                                    .compare_with(&mut ctx, Img::new(di.as_slice(), w, h))
                                    .unwrap();
                            }
                            zenbench::black_box(acc)
                        })
                    }
                });
            }
        });
    };

    let no_gate = std::env::var("LOOP_WALL_NO_GATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let result = if no_gate {
        eprintln!(
            "[loop-wall] LOOP_WALL_NO_GATE=1 — resource gate DISABLED (caller guarantees a quiet machine)"
        );
        zenbench::run_gated(zenbench::GateConfig::disabled(), build)
    } else {
        zenbench::run(build)
    };

    record_scores(
        &mut scores,
        w,
        h,
        stride,
        &refp,
        &distp,
        metric_filter.as_deref(),
    );
    write_tsv(&out_tsv, &label, w, h, stride, &result, &scores);
}

/// One score per metric, and — critically — a STRIDED-vs-TIGHT parity check.
///
/// The recommendation this harness feeds says "drop the per-candidate copy and
/// pass the padded stride through". That is only sound if the two produce the
/// same number, so the parity delta is recorded next to the wall time rather
/// than asserted in a comment.
fn record_scores(
    scores: &mut Vec<(String, f64)>,
    w: usize,
    h: usize,
    stride: usize,
    rp: &[Vec<f32>; 3],
    dp: &[Vec<f32>; 3],
    metric_filter: Option<&str>,
) {
    let want = |m: &str| metric_filter.map(|f| f == m).unwrap_or(true);

    let tight = |src: &[Vec<f32>; 3]| {
        let mut out = [
            vec![0.0f32; w * h],
            vec![0.0f32; w * h],
            vec![0.0f32; w * h],
        ];
        for c in 0..3 {
            copy_strided_into(&mut out[c], &src[c], stride, w, h);
        }
        out
    };
    let tr = tight(rp);
    let td = tight(dp);

    if want("butter") {
        use butteraugli::{ButteraugliParams, ButteraugliReference};
        let p = ButteraugliParams::new();
        let pre = ButteraugliReference::new_linear_planar(
            &rp[0],
            &rp[1],
            &rp[2],
            w,
            h,
            stride,
            p.clone(),
        )
        .unwrap();
        let s_strided = pre
            .compare_linear_planar(&dp[0], &dp[1], &dp[2], stride)
            .unwrap()
            .score;
        let pre_t =
            ButteraugliReference::new_linear_planar(&tr[0], &tr[1], &tr[2], w, h, w, p).unwrap();
        let s_tight = pre_t
            .compare_linear_planar(&td[0], &td[1], &td[2], w)
            .unwrap()
            .score;
        scores.push(("butter".into(), s_strided));
        scores.push((
            "butter__stride_parity_abs_delta".into(),
            (s_strided - s_tight).abs(),
        ));
    }
    if want("zensim") {
        use zensim::{DiffmapOptions, Zensim, ZensimProfile};
        let z = Zensim::new(ZensimProfile::latest_preview());
        let pre = z
            .precompute_reference_linear_planar([&rp[0], &rp[1], &rp[2]], w, h, stride)
            .unwrap();
        let s_strided = z
            .compute_with_ref_and_diffmap_linear_planar(
                &pre,
                [&dp[0], &dp[1], &dp[2]],
                w,
                h,
                stride,
                DiffmapOptions::default(),
            )
            .unwrap()
            .score();
        let pre_t = z
            .precompute_reference_linear_planar([&tr[0], &tr[1], &tr[2]], w, h, w)
            .unwrap();
        let s_tight = z
            .compute_with_ref_and_diffmap_linear_planar(
                &pre_t,
                [&td[0], &td[1], &td[2]],
                w,
                h,
                w,
                DiffmapOptions::default(),
            )
            .unwrap()
            .score();
        scores.push(("zensim".into(), s_strided));
        scores.push((
            "zensim__stride_parity_abs_delta".into(),
            (s_strided - s_tight).abs(),
        ));

        // Cold vs warm parity: does the warm-reference path score the same as
        // a cold one-shot? (The GPU side documents such a claim; verify the
        // CPU side rather than assume it.)
        let s_cold = z
            .compute_with_ref_and_diffmap_linear_planar(
                &pre,
                [&dp[0], &dp[1], &dp[2]],
                w,
                h,
                stride,
                DiffmapOptions::default(),
            )
            .unwrap()
            .score();
        scores.push((
            "zensim__warm_repeat_abs_delta".into(),
            (s_strided - s_cold).abs(),
        ));
    }
    if want("ssim2") {
        use fast_ssim2::Ssimulacra2Reference;
        use imgref::Img;
        let mut ri: Vec<[f32; 3]> = Vec::new();
        let mut di: Vec<[f32; 3]> = Vec::new();
        interleave_into(&mut ri, rp, stride, w, h);
        interleave_into(&mut di, dp, stride, w, h);
        let pre = Ssimulacra2Reference::new(Img::new(ri.as_slice(), w, h)).unwrap();
        let s_warm = pre.compare(Img::new(di.as_slice(), w, h)).unwrap();
        let s_cold = fast_ssim2::compute_ssimulacra2(
            Img::new(ri.as_slice(), w, h),
            Img::new(di.as_slice(), w, h),
        )
        .unwrap();
        let mut ctx = pre.compare_context();
        let s_ctx = pre
            .compare_with(&mut ctx, Img::new(di.as_slice(), w, h))
            .unwrap();
        scores.push(("ssim2".into(), s_warm));
        // Does the warm-reference path agree with the cold one-shot?
        scores.push((
            "ssim2__warm_vs_cold_abs_delta".into(),
            (s_warm - s_cold).abs(),
        ));
        // Does the scratch-recycling path agree with the allocating one?
        scores.push(("ssim2__zeroalloc_abs_delta".into(), (s_warm - s_ctx).abs()));

        // CORRECTION 2026-08-30: does a STRIDED ImgRef score the same as a packed
        // one? If the ingress read the padding columns this would be non-zero.
        // This is the actual test of "supports stride", as opposed to merely
        // accepting the type — and it is the check the first pass never ran.
        let mut di_pad: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; stride * h];
        for y in 0..h {
            for x in 0..w {
                let s = y * stride + x;
                di_pad[y * stride + x] = [dp[0][s], dp[1][s], dp[2][s]];
            }
        }
        let s_strided_imgref = pre
            .compare(imgref::Img::new_stride(di_pad.as_slice(), w, h, stride))
            .unwrap();
        scores.push((
            "ssim2__imgref_stride_parity_abs_delta".into(),
            (s_warm - s_strided_imgref).abs(),
        ));
    }
}

fn lookup(scores: &[(String, f64)], key: &str) -> String {
    scores
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| format!("{v}"))
        .unwrap_or_else(|| "-".to_string())
}

fn write_tsv(
    out_tsv: &str,
    label: &str,
    w: usize,
    h: usize,
    stride: usize,
    result: &SuiteResult,
    scores: &[(String, f64)],
) {
    let need_header = !std::path::Path::new(out_tsv).exists();
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_tsv)
        .expect("open out tsv");
    if need_header {
        writeln!(
            f,
            "size_label\tmetric\tmode\tw\th\tstride\tmean_ns\tmean_ms\tn_rounds\tscore"
        )
        .unwrap();
    }
    for comp in &result.comparisons {
        for bm in &comp.benchmarks {
            let parts: Vec<&str> = bm.name.split("__").collect();
            let metric = parts.first().copied().unwrap_or("?");
            let mode = parts.get(1).copied().unwrap_or("?");
            let mean_ns = bm.summary.mean;
            writeln!(
                f,
                "{label}\t{metric}\t{mode}\t{w}\t{h}\t{stride}\t{mean_ns:.1}\t{:.5}\t{}\t{}",
                mean_ns / 1.0e6,
                comp.completed_rounds,
                lookup(scores, metric)
            )
            .unwrap();
        }
    }
    // Parity rows carry no wall time — they are correctness evidence for the
    // recommendation, emitted alongside it so the two can never drift apart.
    for (k, v) in scores {
        if k.contains("delta") {
            let metric = k.split("__").next().unwrap_or("?");
            writeln!(
                f,
                "{label}\t{metric}\tPARITY:{}\t{w}\t{h}\t{stride}\t-\t-\t-\t{v}",
                k.split_once("__").map(|x| x.1).unwrap_or(k)
            )
            .unwrap();
        }
    }
    eprintln!("wrote loop-wall rows for size {label} to {out_tsv}");
}
