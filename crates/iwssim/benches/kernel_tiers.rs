//! Per-kernel NEON-vs-forced-scalar for iwssim's SIMD kernels.
//!
//! iwssim had 15 dispatch sites and no tier benchmark. Its kernels are
//! `pub(crate)`, so nothing outside could measure them — a kernel slower than
//! the scalar tier it dispatches away from would have been invisible. That
//! failure mode was real elsewhere in the 2026-07-29 aarch64 sweep: zenquant
//! 0.58x, linear-srgb 0.93x, zenresize 0.94x.
//!
//! NEON is BASELINE on aarch64, so the "scalar" arm is autovectorized too:
//! ~1.00x means LLVM already matched the dispatched path; BELOW 1.00 is a bug.
//!
//! Run: `cargo bench -p iwssim --features _dev --bench kernel_tiers`

use iwssim::simd_kernels as k;
use zenbench::prelude::*;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") { "neon" } else { "v3(avx2)" };

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(on: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!on).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_on: bool) -> bool { false }

const W: usize = 1024;
const H: usize = 1024;

fn ramp(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / 16_777_216.0
        })
        .collect()
}

fn bench(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    let n = W * H;
    let a: &'static [f32] = Box::leak(ramp(n, 3).into_boxed_slice());
    let b_: &'static [f32] = Box::leak(ramp(n, 7).into_boxed_slice());
    let c: &'static [f32] = Box::leak(ramp(n, 11).into_boxed_slice());
    let d: &'static [f32] = Box::leak(ramp(n, 13).into_boxed_slice());
    let e: &'static [f32] = Box::leak(ramp(n, 17).into_boxed_slice());

    macro_rules! ab {
        ($name:expr, $out:expr, $call:expr) => {
            suite.compare($name, |g| {
                g.throughput(Throughput::Bytes((n * 4) as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |bch| {
                        bch.with_input(move || { set_simd(simd); $out })
                            .run(move |mut o| { $call(&mut o); o })
                    });
                }
            });
        };
    }

    ab!("square_into", vec![0f32; n], |o: &mut Vec<f32>| k::square_into(a, o));
    ab!("mul_into", vec![0f32; n], |o: &mut Vec<f32>| k::mul_into(a, b_, o));
    ab!("cs_combine_into", vec![0f32; n],
        |o: &mut Vec<f32>| k::cs_combine_into(a, b_, c, d, e, o, true));
    // Valid-mode 11-tap: the output is 10 shorter along the filtered axis.
    const DW: usize = W - 10;
    const DH: usize = H - 10;
    ab!("ssim_gauss_h_pass", vec![0f32; H * DW],
        |o: &mut Vec<f32>| k::ssim_gauss_h_pass(a, H, W, DW, o));
    ab!("ssim_gauss_v_pass", vec![0f32; DH * W],
        |o: &mut Vec<f32>| k::ssim_gauss_v_pass(a, H, DH, W, o));

    // Reduction, no output buffer.
    suite.compare("weighted_sum_pair", |g| {
        g.throughput(Throughput::Bytes((n * 4) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |bch| {
                bch.with_input(move || set_simd(simd))
                    .run(move |_| k::weighted_sum_pair(a, b_))
            });
        }
    });

    set_simd(true);
}

zenbench::main!(bench);
