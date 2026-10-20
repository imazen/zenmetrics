//! SWE-2 PSNR-optimization experiment harness (2026-10-20).
//!
//! Modes:
//!   psnr_opt_wall exhaustive
//!       For every RGB8 triple, compare f32 candidate conversions against
//!       the f64 production formulas (studio-601 Y/Cb/Cr, libvmaf Y) and
//!       the production f32 studio-601 Y code against its reference.
//!       A conversion that differs on any triple cannot be bit-exact.
//!   psnr_opt_wall bench <w> <h> <reps>
//!       Times the production metric paths (via `run_metric`) and the f32
//!       candidate kernels on the synthetic AIC wall pair.
//!
//! The f32 candidates are measurement-only: a mismatch under
//! `exhaustive` means the variant can never produce identical score bits.

use archmage::SimdToken;
use clap::ValueEnum;
use std::time::Instant;
use zenmetrics_cli::{
    decode::Rgb8Image,
    metrics::{GpuRuntime, MetricKind, run_metric},
};

// ---- scalar reference conversions (copied from classical.rs) ----------

fn code_y_f64(x: &[u8]) -> f64 {
    (16.0
        + (65.481 * f64::from(x[0]) + 128.553 * f64::from(x[1]) + 24.966 * f64::from(x[2])) / 255.0)
        .round()
        .clamp(0.0, 255.0)
}
fn code_cb_f64(x: &[u8]) -> f64 {
    (128.0
        + (-37.797 * f64::from(x[0]) - 74.203 * f64::from(x[1]) + 112.0 * f64::from(x[2])) / 255.0)
        .round()
        .clamp(0.0, 255.0)
}
fn code_cr_f64(x: &[u8]) -> f64 {
    (128.0
        + (112.0 * f64::from(x[0]) - 93.786 * f64::from(x[1]) - 18.214 * f64::from(x[2])) / 255.0)
        .round()
        .clamp(0.0, 255.0)
}
fn code_libvmaf_f64(x: &[u8]) -> u8 {
    let lum = 0.2126 * (f64::from(x[0]) / 255.0)
        + 0.7152 * (f64::from(x[1]) / 255.0)
        + 0.0722 * (f64::from(x[2]) / 255.0);
    (16.0 + 219.0 * lum).round().clamp(16.0, 235.0) as u8
}
fn code_studio_y_f32(x: &[u8]) -> u8 {
    (16.0 + (65.481 * x[0] as f32 + 128.553 * x[1] as f32 + 24.966 * x[2] as f32) / 255.0)
        .round()
        .clamp(0.0, 255.0) as u8
}

// ---- f32 candidate conversions ----------------------------------------

fn code_y_f32(x: &[u8]) -> f64 {
    f64::from(
        (16.0f32 + (65.481 * x[0] as f32 + 128.553 * x[1] as f32 + 24.966 * x[2] as f32) / 255.0)
            .round()
            .clamp(0.0, 255.0),
    )
}
fn code_cb_f32(x: &[u8]) -> f64 {
    f64::from(
        (128.0f32 + (-37.797 * x[0] as f32 - 74.203 * x[1] as f32 + 112.0 * x[2] as f32) / 255.0)
            .round()
            .clamp(0.0, 255.0),
    )
}
fn code_cr_f32(x: &[u8]) -> f64 {
    f64::from(
        (128.0f32 + (112.0 * x[0] as f32 - 93.786 * x[1] as f32 - 18.214 * x[2] as f32) / 255.0)
            .round()
            .clamp(0.0, 255.0),
    )
}
fn code_libvmaf_f32(x: &[u8]) -> u8 {
    let lum = 0.2126f32 * (x[0] as f32 / 255.0)
        + 0.7152f32 * (x[1] as f32 / 255.0)
        + 0.0722f32 * (x[2] as f32 / 255.0);
    (16.0f32 + 219.0 * lum).round().clamp(16.0, 235.0) as u8
}

/// The kernel's `floor(x + 0.5)` half-away rounding, scalar form — checked
/// against `f32::round`/`f64::round` too.
fn code_y_f32_floor(x: &[u8]) -> u8 {
    ((16.0f32 + (65.481 * x[0] as f32 + 128.553 * x[1] as f32 + 24.966 * x[2] as f32) / 255.0)
        + 0.5)
        .floor()
        .clamp(0.0, 255.0) as u8
}
fn code_plane_f64_floor(x: &[u8], plane: usize) -> f64 {
    let lin = match plane {
        1 => {
            (-37.797 * f64::from(x[0]) - 74.203 * f64::from(x[1]) + 112.0 * f64::from(x[2])) / 255.0
        }
        2 => {
            (112.0 * f64::from(x[0]) - 93.786 * f64::from(x[1]) - 18.214 * f64::from(x[2])) / 255.0
        }
        _ => {
            (65.481 * f64::from(x[0]) + 128.553 * f64::from(x[1]) + 24.966 * f64::from(x[2]))
                / 255.0
        }
    };
    let off = if plane == 0 { 16.0 } else { 128.0 };
    (off + lin + 0.5).floor().clamp(0.0, 255.0)
}
fn code_libvmaf_f64_floor(x: &[u8]) -> u8 {
    let lum = 0.2126 * (f64::from(x[0]) / 255.0)
        + 0.7152 * (f64::from(x[1]) / 255.0)
        + 0.0722 * (f64::from(x[2]) / 255.0);
    ((16.0 + 219.0 * lum) + 0.5).floor().clamp(16.0, 235.0) as u8
}

fn exhaustive() {
    let mut n_y_f32 = 0u64;
    let mut n_cb_f32 = 0u64;
    let mut n_cr_f32 = 0u64;
    let mut n_vmaf_f32 = 0u64;
    let mut n_yfloor = 0u64;
    let mut n_pfloor = [0u64; 3];
    let mut n_vmaf_floor = 0u64;
    let mut first: [Option<(u8, u8, u8)>; 4] = [None; 4];
    for r in 0..=255u32 {
        for g in 0..=255u32 {
            for b in 0..=255u32 {
                let x = [r as u8, g as u8, b as u8];
                if code_y_f32(&x) != code_y_f64(&x) {
                    n_y_f32 += 1;
                    first[0].get_or_insert((r as u8, g as u8, b as u8));
                }
                if code_cb_f32(&x) != code_cb_f64(&x) {
                    n_cb_f32 += 1;
                    first[1].get_or_insert((r as u8, g as u8, b as u8));
                }
                if code_cr_f32(&x) != code_cr_f64(&x) {
                    n_cr_f32 += 1;
                    first[2].get_or_insert((r as u8, g as u8, b as u8));
                }
                if code_libvmaf_f32(&x) != code_libvmaf_f64(&x) {
                    n_vmaf_f32 += 1;
                    first[3].get_or_insert((r as u8, g as u8, b as u8));
                }
                // floor(x+0.5) == round() on the production paths:
                if code_y_f32_floor(&x) != code_studio_y_f32(&x) {
                    n_yfloor += 1;
                }
                for p in 0..3 {
                    let refcode = match p {
                        1 => code_cb_f64(&x),
                        2 => code_cr_f64(&x),
                        _ => code_y_f64(&x),
                    };
                    if code_plane_f64_floor(&x, p) != refcode {
                        n_pfloor[p] += 1;
                    }
                }
                if code_libvmaf_f64_floor(&x) != code_libvmaf_f64(&x) {
                    n_vmaf_floor += 1;
                }
            }
        }
    }
    let t = 1u64 << 24;
    println!("exhaustive over {t} RGB8 triples:");
    println!(
        "  f32 y-studio601  vs f64: {n_y_f32} mismatches (first {:?})",
        first[0]
    );
    println!(
        "  f32 cb           vs f64: {n_cb_f32} mismatches (first {:?})",
        first[1]
    );
    println!(
        "  f32 cr           vs f64: {n_cr_f32} mismatches (first {:?})",
        first[2]
    );
    println!(
        "  f32 libvmaf-y    vs f64: {n_vmaf_f32} mismatches (first {:?})",
        first[3]
    );
    println!("  floor(+0.5) f32 y-kernel vs f32::round: {n_yfloor} mismatches");
    println!(
        "  floor(+0.5) f64 plane kernels vs f64::round: y={} cb={} cr={} mismatches",
        n_pfloor[0], n_pfloor[1], n_pfloor[2]
    );
    println!("  floor(+0.5) f64 libvmaf kernel vs f64::round: {n_vmaf_floor} mismatches");
}

// ---- f32 candidate SIMD kernel (measurement only) ---------------------

/// f32 variant of the chroma/libvmaf code + SSE shape — timed to size the
/// win the candidate would give IF it were bit-exact (see `exhaustive`).
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x16, i32x16))]
fn plane_sse_f32_inner(
    token: Token,
    a: &[u8],
    b: &[u8],
    c0: f32,
    c1: f32,
    c2: f32,
    off: f32,
) -> u64 {
    let n = a.len() / 3;
    let vc0 = f32x16::splat_t(token, c0);
    let vc1 = f32x16::splat_t(token, c1);
    let vc2 = f32x16::splat_t(token, c2);
    let inv255 = f32x16::splat_t(token, 255.0);
    let voff = f32x16::splat_t(token, off);
    let half = f32x16::splat_t(token, 0.5);
    let lo = f32x16::splat_t(token, 0.0);
    let hi = f32x16::splat_t(token, 255.0);
    let mut total = 0u64;
    let mut block = i32x16::zero_t(token);
    let mut px = 0usize;
    while px + 16 <= n {
        let base = px * 3;
        let lin = |p: &[u8]| {
            (vc0 * f32x16::from_array_t(token, core::array::from_fn(|j| p[base + 3 * j] as f32))
                + vc1
                    * f32x16::from_array_t(
                        token,
                        core::array::from_fn(|j| p[base + 3 * j + 1] as f32),
                    )
                + vc2
                    * f32x16::from_array_t(
                        token,
                        core::array::from_fn(|j| p[base + 3 * j + 2] as f32),
                    ))
                / inv255
        };
        let code = |l: f32x16| (voff + l + half).floor().clamp(lo, hi).to_i32();
        let d = code(lin(a)) - code(lin(b));
        block = block + d * d;
        px += 16;
        if px % (16384 * 16) == 0 {
            for v in block.to_array() {
                total += v as u64;
            }
            block = i32x16::zero_t(token);
        }
    }
    for v in block.to_array() {
        total += v as u64;
    }
    while px < n {
        let lin = |p: &[u8]| {
            (c0 * p[3 * px] as f32 + c1 * p[3 * px + 1] as f32 + c2 * p[3 * px + 2] as f32) / 255.0
        };
        let code = |p: &[u8]| (off + lin(p) + 0.5).floor().clamp(0.0, 255.0) as i64;
        let d = code(a) - code(b);
        total += (d * d) as u64;
        px += 1;
    }
    total
}

fn plane_sse_f32(a: &[u8], b: &[u8], c0: f32, c1: f32, c2: f32, off: f32) -> u64 {
    archmage::attuned!(
        plane_sse_f32_inner(a, b, c0, c1, c2, off),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// f32 candidate for the unrounded luma routes — the conversion in f32
/// differs in the last bits, so this can never be exact; timed to size
/// the trade.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x16))]
fn luma_sqerr_f32_inner(token: Token, a: &[u8], b: &[u8], cr: f32, cg: f32, cb: f32) -> f64 {
    let n = a.len() / 3;
    let vcr = f32x16::splat_t(token, cr);
    let vcg = f32x16::splat_t(token, cg);
    let vcb = f32x16::splat_t(token, cb);
    let mut acc = f32x16::zero_t(token);
    let mut px = 0usize;
    while px + 16 <= n {
        let base = px * 3;
        let ya = vcr
            * f32x16::from_array_t(token, core::array::from_fn(|j| a[base + 3 * j] as f32))
            + vcg
                * f32x16::from_array_t(token, core::array::from_fn(|j| a[base + 3 * j + 1] as f32))
            + vcb
                * f32x16::from_array_t(token, core::array::from_fn(|j| a[base + 3 * j + 2] as f32));
        let yb = vcr
            * f32x16::from_array_t(token, core::array::from_fn(|j| b[base + 3 * j] as f32))
            + vcg
                * f32x16::from_array_t(token, core::array::from_fn(|j| b[base + 3 * j + 1] as f32))
            + vcb
                * f32x16::from_array_t(token, core::array::from_fn(|j| b[base + 3 * j + 2] as f32));
        let d = ya - yb;
        acc = acc + d * d;
        px += 16;
    }
    // NOT bit-exact: f32 lane accumulation reorders and rounds at f32
    // precision — this is the measurement candidate, not a production path.
    let mut sum = f64::from(acc.reduce_add());
    while px < n {
        let ya = cr * a[3 * px] as f32 + cg * a[3 * px + 1] as f32 + cb * a[3 * px + 2] as f32;
        let yb = cr * b[3 * px] as f32 + cg * b[3 * px + 1] as f32 + cb * b[3 * px + 2] as f32;
        let d = f64::from(ya - yb);
        sum += d * d;
        px += 1;
    }
    sum
}

fn luma_sqerr_f32(a: &[u8], b: &[u8], cr: f32, cg: f32, cb: f32) -> f64 {
    archmage::attuned!(
        luma_sqerr_f32_inner(a, b, cr, cg, cb),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

fn bench(w: usize, h: usize, reps: usize) {
    let mut reference = vec![0u8; w * h * 3];
    let mut distorted = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let v = ((x * 7 + y * 13) ^ (x * y % 251)) as u8;
            reference[i..i + 3].copy_from_slice(&[v, v.wrapping_add(3), v]);
            distorted[i..i + 3].copy_from_slice(&[v.wrapping_add(9), v, v.wrapping_add(5)]);
        }
    }
    let r = Rgb8Image {
        pixels: reference,
        width: w as u32,
        height: h as u32,
    };
    let d = Rgb8Image {
        pixels: distorted,
        width: w as u32,
        height: h as u32,
    };
    let n = (w * h) as f64;

    for name in [
        "psnr",
        "psnr-y",
        "psnr-y601",
        "psnr-y-studio601",
        "psnr-y-libvmaf",
        "psnr-cb",
        "psnr-cr",
        "psnr-ycbcr611",
    ] {
        let kind = MetricKind::from_str(name, false).unwrap();
        let t = Instant::now();
        let mut out = Vec::new();
        for _ in 0..reps {
            out = run_metric(kind, &r, &d, GpuRuntime::Cpu).unwrap();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!(
            "{name}\t{w}x{h}\t{ms:.3} ms/call\t{}",
            out[0].1.to_bits() as u64
        );
    }

    // f32 candidate kernels (timings only — see `exhaustive` for the
    // bit-exactness verdicts).
    let a = &r.pixels;
    let b = &d.pixels;
    let time = |label: &str, f: &mut dyn FnMut() -> f64| {
        let t = Instant::now();
        let mut v = 0.0;
        for _ in 0..reps {
            v = f();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("{label}\t{w}x{h}\t{ms:.3} ms/call\t(res={v:.6})");
    };
    time("f32-cand psnr-y-studio601-eq", &mut || {
        plane_sse_f32(a, b, 65.481, 128.553, 24.966, 16.0) as f64 / n
    });
    time("f32-cand psnr-cb", &mut || {
        plane_sse_f32(a, b, -37.797, -74.203, 112.0, 128.0) as f64 / n
    });
    time("f32-cand psnr-cr", &mut || {
        plane_sse_f32(a, b, 112.0, -93.786, -18.214, 128.0) as f64 / n
    });
    time("f32-cand luma709 (inexact)", &mut || {
        luma_sqerr_f32(a, b, 0.2126, 0.7152, 0.0722) / n
    });
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("exhaustive") => exhaustive(),
        Some("tiers") => {
            for (name, t) in [
                ("v4x", archmage::X64V4xToken::summon().is_some()),
                ("v4", archmage::X64V4Token::summon().is_some()),
                ("v3", archmage::X64V3Token::summon().is_some()),
                ("v2", archmage::X64V2Token::summon().is_some()),
                ("v1", archmage::X64V1Token::summon().is_some()),
            ] {
                println!("{name}: {t}");
            }
        }
        Some("bench") => {
            let w: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1024);
            let h: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1024);
            let reps: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(5);
            bench(w, h, reps);
        }
        _ => {
            eprintln!("usage: psnr_opt_wall exhaustive | tiers | bench <w> <h> <reps>");
            std::process::exit(64);
        }
    }
}
