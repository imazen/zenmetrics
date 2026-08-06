//! Per-kernel NEON-vs-forced-scalar for cvvdp's transcendental kernels.
//!
//! vexp/vlog/vpow are the hottest inner loops of the metric. Whole-metric
//! benches cannot reveal one of them being SLOWER than its own scalar
//! fallback — that failure mode was real in this sweep (three zenfilters NEON
//! kernels lost to their scalar tier), and transcendentals are exactly where
//! it hides, because a hand-written polynomial competes against whatever LLVM
//! autovectorizes from the scalar body.
//!
//! NOTE: on aarch64 NEON is BASELINE, so the "scalar" arm is the magetypes
//! scalar tier WITH autovectorization. ~1.00x means both compiled to
//! equivalent work; BELOW 1.00 is the bug this exists to catch.
//!
//! Run: `cargo bench -p cvvdp --bench kernel_tiers`

use cvvdp::__bench_math as k;
use zenbench::prelude::*;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_e: bool) -> bool {
    false
}

const N: usize = 1 << 20;

/// Positive inputs spanning several decades — log and pow are undefined or
/// degenerate at/below zero, and a narrow range would let a polynomial look
/// better than it is.
fn xs() -> Vec<f32> {
    let mut s = 0x9e37_79b9u32;
    (0..N)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (s >> 8) as f32 / 16_777_216.0;
            1e-3 + u * 40.0
        })
        .collect()
}

fn bench_math(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    let src: &'static [f32] = Box::leak(xs().into_boxed_slice());

    macro_rules! pair {
        ($name:expr, $call:expr) => {
            suite.compare($name, |g| {
                g.throughput(Throughput::Elements(N as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        let mut out = vec![0f32; N];
                        b.iter(move || {
                            set_simd(simd);
                            #[allow(clippy::redundant_closure_call)]
                            ($call)(src, &mut out)
                        })
                    });
                }
            });
        };
    }

    pair!("vexp_into", |s: &[f32], o: &mut [f32]| k::vexp_into(s, o));
    pair!("vlog_into", |s: &[f32], o: &mut [f32]| k::vlog_into(s, o));
    pair!("vpow_into/p=2.4", |s: &[f32], o: &mut [f32]| k::vpow_into(
        s, o, 2.4
    ));
    pair!("vpow_into/p=0.42", |s: &[f32], o: &mut [f32]| k::vpow_into(
        s, o, 0.42
    ));

    set_simd(true);
}

zenbench::main!(bench_math);
