//! Brute-force math-equivalence harness for cvvdp-cpu's SIMD kernels.
//!
//! Proves (blur kernels) or bounds (transcendental approximations) that
//! each SIMD kernel matches its scalar reference across exhaustive +
//! randomized inputs and adversarial edge cases. The end-to-end 1e-4
//! JOD parity gate and the ~5 per-chunk fixture tests can MASK a
//! per-element kernel divergence the Minkowski pooling absorbs — this
//! harness measures the per-element envelope directly.
//!
//! Kernels under test (all on cvvdp-cpu master):
//!   1. sigma3 13-tap Gaussian blur (`gaussian_blur_sigma3_simd` /
//!      `pu_blur_horizontal_pass` / `pu_blur_vertical_pass`)
//!   2. pyramid 5-tap reduce/expand (`reduce_*_pass` / `expand_*_pass`)
//!   3. masking vexp / vlog / vpow / safe_pow_with_offset
//!
//! Assertion strategy:
//!   - BLUR kernels: near-exact (FMA-grouping differences only). The
//!     SIMD interior accumulates the 5/13-tap dot in source order
//!     `v0*k0 + v1*k1 + ...`, which is the SAME left-to-right order as
//!     the scalar reference. LLVM may contract `a + b*c` into one FMA
//!     differently per path, so we expect a tiny ULP envelope. Asserted
//!     at a TIGHT bound (4 ULP). Anything larger = boundary-handling bug.
//!   - vexp/vlog/vpow: magetypes `*_midp_unchecked` are APPROXIMATIONS.
//!     We MEASURE the ULP envelope and commit it as a regression gate
//!     (max + p99 + max-rel), NOT bit-exactness. The scalar TAIL of
//!     these kernels also goes through `f32::exp`/`ln`/`powf` (libm),
//!     so the divergence is purely the SIMD-lane approximation vs libm.
//!
//! Access: the `pub(crate)` kernel entry points are re-exported through
//! `cvvdp_cpu::__simd_equiv_test_api` under the `__simd_equiv_test`
//! cargo feature (a `#[doc(hidden)]` visibility shim, no logic change).
//!
//! Run: `cargo test -p cvvdp-cpu --features __simd_equiv_test --test
//! simd_equivalence -- --nocapture` to see the printed envelopes.

#![cfg(feature = "__simd_equiv_test")]
// The scalar reference kernels here index the tap arrays `k[t]` /
// `GAUSS5[i]` by an explicit loop counter ON PURPOSE — they are
// line-by-line ports of the source/upstream scalar contract (which uses
// `for t in 0..13 { ... k[t] ... }`). Rewriting them as `.enumerate()`
// iterators would obscure the 1:1 correspondence with the code under
// test. The production crate carries the same `#![allow]` for the same
// reason; mirror it in this test crate.
#![allow(clippy::needless_range_loop)]

use cvvdp_cpu::__simd_equiv_test_api as api;
use cvvdp_cpu::__simd_equiv_test_api::{GAUSS5, PU_BLUR_KERNEL_1D};

// ===========================================================================
// ULP + error metric machinery
// ===========================================================================

/// Monotonic ordering transform for f32 bit patterns: maps the f32 line
/// to a total order on i64 such that adjacent representable values
/// differ by 1. Handles +/- zero and signed magnitudes. NaN is not
/// expected from these kernels (asserted separately).
///
/// Canonical IEEE-754 total-order key (the same trick `f32::total_cmp`
/// uses): take the raw bits as i32; if the sign bit is set (negative),
/// flip ALL bits, else flip only the sign bit. The result is a strictly
/// monotone i32 over the float line. We then widen to i64 so the
/// subtraction in `ulp_diff` cannot overflow at the line extremes.
fn ord_key(x: f32) -> i64 {
    let raw = x.to_bits() as i32;
    // `raw >> 31` is 0 for non-negative, -1 (all ones) for negative.
    // For negative: XOR with all-ones flips every bit.
    // For non-negative: XOR with 0x8000_0000 flips just the sign bit.
    let mask = ((raw >> 31) as u32) | 0x8000_0000;
    let ordered_u32 = x.to_bits() ^ mask;
    // Widen the MONOTONE u32 to i64 unsigned (NOT via i32 — that would
    // wrap the high half to negative and break adjacency across zero).
    // In u32 space the line runs 0 (most-negative) .. u32::MAX
    // (most-positive); adjacent representable floats differ by 1.
    i64::from(ordered_u32)
}

/// ULP distance between two finite f32 values. `+0.0` and `-0.0` are
/// treated as 0 ULP apart.
fn ulp_diff(a: f32, b: f32) -> u64 {
    if a == b {
        return 0; // covers +0 vs -0
    }
    debug_assert!(a.is_finite() && b.is_finite(), "ulp_diff on non-finite");
    let ka = ord_key(a);
    let kb = ord_key(b);
    (ka - kb).unsigned_abs()
}

#[derive(Default, Clone)]
struct Envelope {
    n: u64,
    max_abs: f64,
    max_rel: f64,
    // ULP histogram buckets: [0, 1, 2-4, 5-16, 17-128, >128]
    hist: [u64; 6],
    ulps: Vec<u64>, // collected for percentiles (capped sample)
    max_ulp: u64,
    // worst-case context for reporting
    worst_ulp_a: f32,
    worst_ulp_b: f32,
    nan_or_inf: u64,
    // For the transcendental APPROXIMATIONS: values where |want| is
    // below a magnitude floor (catastrophic-cancellation residue or
    // underflow) make ULP/rel meaningless (dividing a tiny error by ~0).
    // We track them separately rather than letting them dominate the
    // measured envelope — the masking pipeline doesn't care about
    // sub-floor magnitudes. ZERO floor (the default) counts everything.
    abs_floor: f64,
    below_floor: u64,
}

impl Envelope {
    fn with_floor(abs_floor: f64) -> Self {
        Envelope {
            abs_floor,
            ..Default::default()
        }
    }

    fn observe(&mut self, got: f32, want: f32) {
        self.n += 1;
        if !got.is_finite() || !want.is_finite() {
            // Permit matching non-finite (e.g. both inf) but flag any
            // mismatch where one is finite and the other isn't.
            if got.is_finite() != want.is_finite() || (got.is_nan() != want.is_nan()) {
                self.nan_or_inf += 1;
            }
            return;
        }
        // Skip sub-floor reference magnitudes from the ULP/rel envelope
        // (still counted in `below_floor` for honest reporting).
        if (want.abs() as f64) < self.abs_floor && (got.abs() as f64) < self.abs_floor {
            self.below_floor += 1;
            return;
        }
        let abs = (got as f64 - want as f64).abs();
        if abs > self.max_abs {
            self.max_abs = abs;
        }
        let denom = (want.abs() as f64).max(self.abs_floor.max(1e-30));
        let rel = abs / denom;
        if rel > self.max_rel {
            self.max_rel = rel;
        }
        let u = ulp_diff(got, want);
        if u > self.max_ulp {
            self.max_ulp = u;
            self.worst_ulp_a = got;
            self.worst_ulp_b = want;
        }
        let bucket = match u {
            0 => 0,
            1 => 1,
            2..=4 => 2,
            5..=16 => 3,
            17..=128 => 4,
            _ => 5,
        };
        self.hist[bucket] += 1;
        // Cap the percentile sample to bound memory on large sweeps.
        if self.ulps.len() < 5_000_000 {
            self.ulps.push(u);
        }
    }

    fn percentile(&mut self, p: f64) -> u64 {
        if self.ulps.is_empty() {
            return 0;
        }
        self.ulps.sort_unstable();
        let idx = ((self.ulps.len() as f64 - 1.0) * p).round() as usize;
        self.ulps[idx]
    }

    fn report(&mut self, name: &str) {
        let p50 = self.percentile(0.50);
        let p99 = self.percentile(0.99);
        let total: u64 = self.hist.iter().sum();
        let pct = |c: u64| -> f64 {
            if total == 0 {
                0.0
            } else {
                100.0 * c as f64 / total as f64
            }
        };
        println!("---- {name} ----");
        println!("  elements compared : {}", self.n);
        if self.abs_floor > 0.0 {
            println!(
                "  abs floor         : {:.1e}  (below-floor skipped: {} of {})",
                self.abs_floor, self.below_floor, self.n
            );
        }
        println!("  max abs err       : {:.6e}", self.max_abs);
        println!("  max rel err       : {:.6e}", self.max_rel);
        println!(
            "  ULP: p50={p50} p99={p99} max={} (got={} want={})",
            self.max_ulp, self.worst_ulp_a, self.worst_ulp_b
        );
        println!(
            "  ULP hist: 0:{:.2}% 1:{:.2}% 2-4:{:.2}% 5-16:{:.2}% 17-128:{:.2}% >128:{:.2}%",
            pct(self.hist[0]),
            pct(self.hist[1]),
            pct(self.hist[2]),
            pct(self.hist[3]),
            pct(self.hist[4]),
            pct(self.hist[5]),
        );
        assert_eq!(
            self.nan_or_inf, 0,
            "{name}: {} elements had finite/non-finite mismatch",
            self.nan_or_inf
        );
    }
}

// ===========================================================================
// Deterministic RNG + input distributions
// ===========================================================================

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64.
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Approximate standard normal via sum of 4 uniforms (Irwin-Hall).
    fn gaussian(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.7320508 // scale to ~unit variance
    }
}

#[derive(Clone, Copy)]
enum Dist {
    Unit,       // [0, 1)
    Large,      // [0, 1e4)
    Gaussian,   // ~N(0,1), can be negative
    LogUniform, // 10^U(-6, 4): spans magnitudes, strictly positive
    PowerOf2,   // exact powers of 2 in a wide range
}

fn gen_values(rng: &mut Rng, n: usize, dist: Dist) -> Vec<f32> {
    (0..n)
        .map(|_| match dist {
            Dist::Unit => rng.unit(),
            Dist::Large => rng.unit() * 1e4,
            Dist::Gaussian => rng.gaussian(),
            Dist::LogUniform => {
                let e = rng.unit() * 10.0 - 6.0; // [-6, 4)
                10f32.powf(e)
            }
            Dist::PowerOf2 => {
                let e = (rng.next_u64() % 40) as i32 - 20; // 2^[-20, 20)
                2f32.powi(e)
            }
        })
        .collect()
}

// ===========================================================================
// Scalar references (pass-level contracts).
//
// CRITICAL: the SIMD pyramid passes implement ONLY the bulk zero-pad
// conv (reduce) / zero-insert conv (expand). They do NOT apply the
// pycvvdp edge patches — those are applied by the cvvdp-cpu
// `pyramid.rs` wrapper AFTER the SIMD pass. So the per-pass scalar
// reference here is the documented per-pass contract (re-ported from
// the inline source comments + inline tests), NOT the full
// `gausspyr_reduce_scalar` (which folds in the patches).
//
// The 13-tap blur SIMD path IS the full blur (interior SIMD + scalar
// boundary patches), so its reference is the upstream
// `gaussian_blur_sigma3` scalar — already accessible via cvvdp-gpu.
// ===========================================================================

fn reflect_idx_for_blur(i: isize, n: usize) -> usize {
    let n_i = n as isize;
    let mut j = i;
    while j < 0 || j >= n_i {
        if j < 0 {
            j = -j;
        }
        if j >= n_i {
            j = 2 * n_i - 2 - j;
        }
    }
    j as usize
}

/// Scalar reference for the 13-tap blur HORIZONTAL pass (per-row,
/// reflect-padded). Line-by-line with the source contract.
fn blur_h_scalar_ref(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let k = PU_BLUR_KERNEL_1D;
    let mut out = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0_f32;
            for t in 0..13 {
                let sx = reflect_idx_for_blur(x as isize + t as isize - 6, w);
                s += k[t] * src[y * w + sx];
            }
            out[y * w + x] = s;
        }
    }
    out
}

/// Scalar reference for the 13-tap blur VERTICAL pass.
fn blur_v_scalar_ref(h_pass: &[f32], w: usize, h: usize) -> Vec<f32> {
    let k = PU_BLUR_KERNEL_1D;
    let mut out = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0_f32;
            for t in 0..13 {
                let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                s += k[t] * h_pass[sy * w + x];
            }
            out[y * w + x] = s;
        }
    }
    out
}

/// Scalar reference for the pyramid reduce VERTICAL pass (zero-pad
/// conv, stride 2 in y). NO edge patches.
fn reduce_v_scalar_ref(src: &[f32], sw: usize, sh: usize, dh: usize) -> Vec<f32> {
    let k = GAUSS5;
    let mut out = vec![0.0_f32; sw * dh];
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
            out[dy * sw + x] = k[0] * read(-2)
                + k[1] * read(-1)
                + k[2] * read(0)
                + k[3] * read(1)
                + k[4] * read(2);
        }
    }
    out
}

/// Scalar reference for the pyramid reduce HORIZONTAL pass (zero-pad
/// conv, stride 2 in x). NO edge patches.
fn reduce_h_scalar_ref(vscratch: &[f32], sw: usize, dw: usize, dh: usize) -> Vec<f32> {
    let k = GAUSS5;
    let mut out = vec![0.0_f32; dw * dh];
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
            out[dy * dw + dx] = k[0] * read(-2)
                + k[1] * read(-1)
                + k[2] * read(0)
                + k[3] * read(1)
                + k[4] * read(2);
        }
    }
    out
}

/// Scalar reference for the pyramid expand VERTICAL pass (zero-insert).
fn expand_v_scalar_ref(src: &[f32], sw: usize, sh: usize, out_h: usize) -> Vec<f32> {
    let k = GAUSS5;
    let mut out = vec![0.0_f32; sw * out_h];
    let z_len_v = out_h + 4;
    let odd_h = out_h & 1;
    let back_idx_v = out_h + 2 + odd_h;
    let mut z_v = vec![0.0_f32; z_len_v];
    for x in 0..sw {
        for v in z_v.iter_mut() {
            *v = 0.0;
        }
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
            out[y * sw + x] = 2.0 * sum;
        }
    }
    out
}

/// Scalar reference for the pyramid expand HORIZONTAL pass (zero-insert).
fn expand_h_scalar_ref(vscratch: &[f32], sw: usize, out_w: usize, out_h: usize) -> Vec<f32> {
    let k = GAUSS5;
    let mut out = vec![0.0_f32; out_w * out_h];
    let z_len_h = out_w + 4;
    let odd_w = out_w & 1;
    let back_idx_h = out_w + 2 + odd_w;
    let mut z_h = vec![0.0_f32; z_len_h];
    for y in 0..out_h {
        for v in z_h.iter_mut() {
            *v = 0.0;
        }
        let row_off = y * sw;
        z_h[0] = vscratch[row_off];
        for kx in 0..sw {
            z_h[2 + 2 * kx] = vscratch[row_off + kx];
        }
        z_h[back_idx_h] = vscratch[row_off + sw - 1];
        for x in 0..out_w {
            let sum = k[0] * z_h[x]
                + k[1] * z_h[x + 1]
                + k[2] * z_h[x + 2]
                + k[3] * z_h[x + 3]
                + k[4] * z_h[x + 4];
            out[y * out_w + x] = 2.0 * sum;
        }
    }
    out
}

// ===========================================================================
// Size grids + adversarial fillers
// ===========================================================================

/// (w, h) grid stressing boundaries, remainder lanes, odd/prime dims.
/// The blur caller guards w>6 && h>6, but the SIMD passes themselves
/// are exercised directly — we still keep w,h >= 7 so reflect indices
/// stay well-defined for the 13-tap radius-6 window across small dims.
fn blur_sizes() -> Vec<(usize, usize)> {
    vec![
        (7, 7),
        (7, 13),
        (13, 7),
        (8, 8),
        (9, 9),
        (12, 12),
        (15, 15),
        (16, 16),
        (17, 17),
        (16, 17),
        (17, 16),
        (23, 23), // prime
        (24, 24),
        (31, 29), // primes
        (32, 32),
        (33, 35),
        (40, 24),
        (47, 53), // primes
        (64, 64),
        (97, 101), // primes
        (100, 100),
        (128, 128),
        (129, 127),
        (256, 256),
        (257, 259),
    ]
}

/// (sw, sh) grid for reduce. dh = ceil(sh/2), dw = ceil(sw/2).
fn reduce_sizes() -> Vec<(usize, usize)> {
    vec![
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (8, 8),
        (15, 15),
        (16, 16),
        (17, 19),
        (23, 29), // primes
        (24, 24),
        (32, 32),
        (33, 35),
        (40, 24),
        (47, 53),
        (64, 64),
        (73, 91),
        (97, 101),
        (128, 128),
        (129, 127),
        (256, 256),
        (257, 259),
    ]
}

/// (sw, sh, out_w, out_h) grid for expand. The scalar contract asserts
/// out_w ∈ [2*sw-1, 2*sw]; we cover both parities.
fn expand_cases() -> Vec<(usize, usize, usize, usize)> {
    vec![
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        (2, 2, 4, 4),
        (2, 2, 3, 3),
        (4, 4, 8, 8),
        (4, 4, 7, 7),
        (8, 6, 16, 12),
        (8, 6, 15, 11),
        (16, 12, 32, 24),
        (16, 12, 31, 23),
        (23, 17, 46, 34), // odd source
        (24, 16, 48, 32),
        (33, 17, 65, 33),
        (47, 29, 94, 58),
        (64, 32, 128, 64),
        (97, 51, 194, 101),
        (129, 65, 257, 129),
    ]
}

/// Fill an adversarial buffer of the given kind into `buf` (sized to n).
/// `max_v` controls the saturation magnitude.
fn fill_adversarial(buf: &mut [f32], kind: usize, w: usize, max_v: f32) {
    match kind {
        0 => buf.iter_mut().for_each(|v| *v = 0.0),   // all-zero
        1 => buf.iter_mut().for_each(|v| *v = max_v), // all-max
        2 => buf.iter_mut().for_each(|v| *v = 0.5),   // all-equal mid
        3 => {
            // single-pixel spike at center
            buf.iter_mut().for_each(|v| *v = 0.0);
            if !buf.is_empty() {
                buf[buf.len() / 2] = max_v;
            }
        }
        4 => {
            // checkerboard 0 / max (alternating extremes)
            for (i, v) in buf.iter_mut().enumerate() {
                let x = i % w.max(1);
                let y = i / w.max(1);
                *v = if (x + y) & 1 == 0 { 0.0 } else { max_v };
            }
        }
        5 => {
            // spike at the very first element (left/top reflect boundary)
            buf.iter_mut().for_each(|v| *v = 0.0);
            if !buf.is_empty() {
                buf[0] = max_v;
            }
        }
        6 => {
            // spike at the very last element (right/bottom reflect boundary)
            buf.iter_mut().for_each(|v| *v = 0.0);
            if !buf.is_empty() {
                let n = buf.len();
                buf[n - 1] = max_v;
            }
        }
        7 => {
            // denormals + subnormals mixed with a few normals
            for (i, v) in buf.iter_mut().enumerate() {
                *v = if i & 3 == 0 {
                    f32::from_bits(1) // smallest subnormal
                } else if i & 3 == 1 {
                    f32::MIN_POSITIVE * 0.5 // subnormal
                } else if i & 3 == 2 {
                    1.0
                } else {
                    f32::from_bits(0x0080_0001) // just above MIN_POSITIVE
                };
            }
        }
        _ => unreachable!(),
    }
}

const N_ADVERSARIAL: usize = 8;

// ===========================================================================
// 1. sigma3 13-tap Gaussian blur
// ===========================================================================
//
// Reference: line-by-line scalar (interior + reflect boundary). The
// SIMD path's interior accumulates the same source-order dot, so the
// only divergence is per-op FMA contraction. Tight bound: <= 4 ULP.

const BLUR_ULP_BOUND: u64 = 4;

#[test]
fn blur_sigma3_horizontal_pass_equiv() {
    let mut env = Envelope::default();
    let mut rng = Rng::new(0x5151_5151_0001);
    let dists = [
        Dist::Unit,
        Dist::Large,
        Dist::Gaussian,
        Dist::LogUniform,
        Dist::PowerOf2,
    ];

    // Randomized cases: every size × every distribution × a few seeds.
    let sizes = blur_sizes();
    for &(w, h) in &sizes {
        for &dist in &dists {
            for _seed in 0..12 {
                let src = gen_values(&mut rng, w * h, dist);
                let want = blur_h_scalar_ref(&src, w, h);
                let mut got = vec![0.0_f32; w * h];
                api::pu_blur_horizontal_pass(&src, w, h, &mut got);
                for i in 0..want.len() {
                    env.observe(got[i], want[i]);
                }
            }
        }
    }

    // Adversarial edge cases at sizes that exercise lane boundaries.
    for &(w, h) in &[
        (7usize, 7usize),
        (16, 16),
        (17, 17),
        (32, 32),
        (33, 35),
        (64, 64),
    ] {
        for kind in 0..N_ADVERSARIAL {
            let mut src = vec![0.0_f32; w * h];
            fill_adversarial(&mut src, kind, w, 1000.0);
            let want = blur_h_scalar_ref(&src, w, h);
            let mut got = vec![0.0_f32; w * h];
            api::pu_blur_horizontal_pass(&src, w, h, &mut got);
            for i in 0..want.len() {
                env.observe(got[i], want[i]);
            }
        }
    }

    env.report("blur sigma3 HORIZONTAL pass (SIMD vs scalar)");
    assert!(
        env.max_ulp <= BLUR_ULP_BOUND,
        "blur H pass ULP {} exceeds tight bound {} (got={} want={}) — \
         a blur kernel above a few ULP is a boundary-handling bug, NOT \
         hand-wavable FMA noise. Isolate the failing input.",
        env.max_ulp,
        BLUR_ULP_BOUND,
        env.worst_ulp_a,
        env.worst_ulp_b
    );
}

#[test]
fn blur_sigma3_vertical_pass_equiv() {
    let mut env = Envelope::default();
    let mut rng = Rng::new(0x5151_5151_0002);
    let dists = [
        Dist::Unit,
        Dist::Large,
        Dist::Gaussian,
        Dist::LogUniform,
        Dist::PowerOf2,
    ];

    let sizes = blur_sizes();
    for &(w, h) in &sizes {
        for &dist in &dists {
            for _seed in 0..12 {
                let h_pass = gen_values(&mut rng, w * h, dist);
                let want = blur_v_scalar_ref(&h_pass, w, h);
                let mut got = vec![0.0_f32; w * h];
                api::pu_blur_vertical_pass(&h_pass, w, h, &mut got);
                for i in 0..want.len() {
                    env.observe(got[i], want[i]);
                }
            }
        }
    }

    for &(w, h) in &[
        (7usize, 7usize),
        (16, 16),
        (17, 17),
        (32, 32),
        (33, 35),
        (64, 64),
    ] {
        for kind in 0..N_ADVERSARIAL {
            let mut h_pass = vec![0.0_f32; w * h];
            fill_adversarial(&mut h_pass, kind, w, 1000.0);
            let want = blur_v_scalar_ref(&h_pass, w, h);
            let mut got = vec![0.0_f32; w * h];
            api::pu_blur_vertical_pass(&h_pass, w, h, &mut got);
            for i in 0..want.len() {
                env.observe(got[i], want[i]);
            }
        }
    }

    env.report("blur sigma3 VERTICAL pass (SIMD vs scalar)");
    assert!(
        env.max_ulp <= BLUR_ULP_BOUND,
        "blur V pass ULP {} exceeds tight bound {} (got={} want={}) — \
         boundary-handling bug, not FMA noise. Isolate the failing input.",
        env.max_ulp,
        BLUR_ULP_BOUND,
        env.worst_ulp_a,
        env.worst_ulp_b
    );
}

#[test]
fn blur_sigma3_full_equiv_vs_upstream() {
    // Full two-pass blur vs the canonical upstream scalar
    // `gaussian_blur_sigma3` (cvvdp-gpu). This is the end-to-end blur
    // contract — both passes composed.
    use cvvdp_gpu::kernels::masking::gaussian_blur_sigma3;
    let mut env = Envelope::default();
    let mut rng = Rng::new(0x5151_5151_0003);
    let dists = [Dist::Unit, Dist::Large, Dist::Gaussian, Dist::LogUniform];

    for &(w, h) in &blur_sizes() {
        for &dist in &dists {
            let src = gen_values(&mut rng, w * h, dist);
            let want = gaussian_blur_sigma3(&src, w, h);
            let mut h_pass = Vec::new();
            let mut got = Vec::new();
            api::gaussian_blur_sigma3_simd(&src, w, h, &mut h_pass, &mut got);
            for i in 0..want.len() {
                env.observe(got[i], want[i]);
            }
        }
    }

    for &(w, h) in &[(7usize, 7usize), (16, 16), (33, 35), (64, 64), (128, 128)] {
        for kind in 0..N_ADVERSARIAL {
            let mut src = vec![0.0_f32; w * h];
            fill_adversarial(&mut src, kind, w, 1000.0);
            let want = gaussian_blur_sigma3(&src, w, h);
            let mut h_pass = Vec::new();
            let mut got = Vec::new();
            api::gaussian_blur_sigma3_simd(&src, w, h, &mut h_pass, &mut got);
            for i in 0..want.len() {
                env.observe(got[i], want[i]);
            }
        }
    }

    env.report("blur sigma3 FULL two-pass (SIMD vs upstream scalar)");
    // Two passes compound the per-pass envelope; allow a slightly looser
    // (but still TIGHT) bound. If this exceeds 8 ULP it is a bug.
    const FULL_BLUR_ULP_BOUND: u64 = 8;
    assert!(
        env.max_ulp <= FULL_BLUR_ULP_BOUND,
        "full blur ULP {} exceeds bound {} (got={} want={}) — investigate.",
        env.max_ulp,
        FULL_BLUR_ULP_BOUND,
        env.worst_ulp_a,
        env.worst_ulp_b
    );
}

// ===========================================================================
// 2. pyramid 5-tap reduce / expand
// ===========================================================================

const PYR_ULP_BOUND: u64 = 4;

#[test]
fn pyramid_reduce_vertical_equiv() {
    let mut env = Envelope::default();
    let mut rng = Rng::new(0x7272_0001);
    let dists = [
        Dist::Unit,
        Dist::Large,
        Dist::Gaussian,
        Dist::LogUniform,
        Dist::PowerOf2,
    ];

    for &(sw, sh) in &reduce_sizes() {
        let dh = sh.div_ceil(2);
        for &dist in &dists {
            for _s in 0..12 {
                let src = gen_values(&mut rng, sw * sh, dist);
                let want = reduce_v_scalar_ref(&src, sw, sh, dh);
                let mut got = vec![0.0_f32; sw * dh];
                api::reduce_vertical_pass(&src, sw, sh, dh, &mut got);
                for i in 0..want.len() {
                    env.observe(got[i], want[i]);
                }
            }
        }
    }
    for &(sw, sh) in &[
        (8usize, 8usize),
        (16, 16),
        (17, 19),
        (32, 32),
        (33, 35),
        (64, 64),
    ] {
        let dh = sh.div_ceil(2);
        for kind in 0..N_ADVERSARIAL {
            let mut src = vec![0.0_f32; sw * sh];
            fill_adversarial(&mut src, kind, sw, 1000.0);
            let want = reduce_v_scalar_ref(&src, sw, sh, dh);
            let mut got = vec![0.0_f32; sw * dh];
            api::reduce_vertical_pass(&src, sw, sh, dh, &mut got);
            for i in 0..want.len() {
                env.observe(got[i], want[i]);
            }
        }
    }
    env.report("pyramid REDUCE vertical (SIMD vs scalar)");
    assert!(
        env.max_ulp <= PYR_ULP_BOUND,
        "reduce-v ULP {} exceeds {} (got={} want={}) — boundary bug.",
        env.max_ulp,
        PYR_ULP_BOUND,
        env.worst_ulp_a,
        env.worst_ulp_b
    );
}

#[test]
fn pyramid_reduce_horizontal_equiv() {
    let mut env = Envelope::default();
    let mut rng = Rng::new(0x7272_0002);
    let dists = [
        Dist::Unit,
        Dist::Large,
        Dist::Gaussian,
        Dist::LogUniform,
        Dist::PowerOf2,
    ];

    for &(sw, sh) in &reduce_sizes() {
        let dh = sh.div_ceil(2);
        let dw = sw.div_ceil(2);
        for &dist in &dists {
            for _s in 0..12 {
                let vs = gen_values(&mut rng, sw * dh, dist);
                let want = reduce_h_scalar_ref(&vs, sw, dw, dh);
                let mut got = vec![0.0_f32; dw * dh];
                api::reduce_horizontal_pass(&vs, sw, dw, dh, &mut got);
                for i in 0..want.len() {
                    env.observe(got[i], want[i]);
                }
            }
        }
    }
    for &(sw, sh) in &[
        (8usize, 8usize),
        (16, 16),
        (17, 19),
        (32, 32),
        (33, 35),
        (64, 64),
    ] {
        let dh = sh.div_ceil(2);
        let dw = sw.div_ceil(2);
        for kind in 0..N_ADVERSARIAL {
            let mut vs = vec![0.0_f32; sw * dh];
            fill_adversarial(&mut vs, kind, sw, 1000.0);
            let want = reduce_h_scalar_ref(&vs, sw, dw, dh);
            let mut got = vec![0.0_f32; dw * dh];
            api::reduce_horizontal_pass(&vs, sw, dw, dh, &mut got);
            for i in 0..want.len() {
                env.observe(got[i], want[i]);
            }
        }
    }
    env.report("pyramid REDUCE horizontal (SIMD vs scalar)");
    assert!(
        env.max_ulp <= PYR_ULP_BOUND,
        "reduce-h ULP {} exceeds {} (got={} want={}) — boundary bug.",
        env.max_ulp,
        PYR_ULP_BOUND,
        env.worst_ulp_a,
        env.worst_ulp_b
    );
}

#[test]
fn pyramid_expand_vertical_equiv() {
    let mut env = Envelope::default();
    let mut rng = Rng::new(0x7272_0003);
    let dists = [
        Dist::Unit,
        Dist::Large,
        Dist::Gaussian,
        Dist::LogUniform,
        Dist::PowerOf2,
    ];

    for &(sw, sh, _ow, oh) in &expand_cases() {
        for &dist in &dists {
            for _s in 0..12 {
                let src = gen_values(&mut rng, sw * sh, dist);
                let want = expand_v_scalar_ref(&src, sw, sh, oh);
                let mut got = vec![0.0_f32; sw * oh];
                api::expand_vertical_pass(&src, sw, sh, oh, &mut got);
                for i in 0..want.len() {
                    env.observe(got[i], want[i]);
                }
            }
        }
    }
    for &(sw, sh, _ow, oh) in &[
        (8usize, 6usize, 16usize, 12usize),
        (16, 12, 32, 24),
        (33, 17, 65, 33),
    ] {
        for kind in 0..N_ADVERSARIAL {
            let mut src = vec![0.0_f32; sw * sh];
            fill_adversarial(&mut src, kind, sw, 1000.0);
            let want = expand_v_scalar_ref(&src, sw, sh, oh);
            let mut got = vec![0.0_f32; sw * oh];
            api::expand_vertical_pass(&src, sw, sh, oh, &mut got);
            for i in 0..want.len() {
                env.observe(got[i], want[i]);
            }
        }
    }
    env.report("pyramid EXPAND vertical (SIMD vs scalar)");
    assert!(
        env.max_ulp <= PYR_ULP_BOUND,
        "expand-v ULP {} exceeds {} (got={} want={}) — boundary bug.",
        env.max_ulp,
        PYR_ULP_BOUND,
        env.worst_ulp_a,
        env.worst_ulp_b
    );
}

#[test]
fn pyramid_expand_horizontal_equiv() {
    let mut env = Envelope::default();
    let mut rng = Rng::new(0x7272_0004);
    let dists = [
        Dist::Unit,
        Dist::Large,
        Dist::Gaussian,
        Dist::LogUniform,
        Dist::PowerOf2,
    ];

    for &(sw, _sh, ow, oh) in &expand_cases() {
        for &dist in &dists {
            for _s in 0..12 {
                let vs = gen_values(&mut rng, sw * oh, dist);
                let want = expand_h_scalar_ref(&vs, sw, ow, oh);
                let mut got = vec![0.0_f32; ow * oh];
                let mut z = Vec::new();
                api::expand_horizontal_pass(&vs, sw, ow, oh, &mut got, &mut z);
                for i in 0..want.len() {
                    env.observe(got[i], want[i]);
                }
            }
        }
    }
    for &(sw, _sh, ow, oh) in &[
        (8usize, 6usize, 16usize, 12usize),
        (16, 12, 32, 24),
        (33, 17, 65, 33),
    ] {
        for kind in 0..N_ADVERSARIAL {
            let mut vs = vec![0.0_f32; sw * oh];
            fill_adversarial(&mut vs, kind, sw, 1000.0);
            let want = expand_h_scalar_ref(&vs, sw, ow, oh);
            let mut got = vec![0.0_f32; ow * oh];
            let mut z = Vec::new();
            api::expand_horizontal_pass(&vs, sw, ow, oh, &mut got, &mut z);
            for i in 0..want.len() {
                env.observe(got[i], want[i]);
            }
        }
    }
    env.report("pyramid EXPAND horizontal (SIMD vs scalar)");
    assert!(
        env.max_ulp <= PYR_ULP_BOUND,
        "expand-h ULP {} exceeds {} (got={} want={}) — boundary bug.",
        env.max_ulp,
        PYR_ULP_BOUND,
        env.worst_ulp_a,
        env.worst_ulp_b
    );
}

// ===========================================================================
// 3. masking transcendentals: vexp / vlog / vpow / safe_pow
//
// These wrap magetypes' `*_midp_unchecked` APPROXIMATIONS. We do NOT
// assert bit-exactness. We MEASURE the ULP envelope vs the scalar libm
// reference (`f32::exp/ln/powf`) and assert it does not REGRESS beyond
// the committed bound + margin.
//
// The committed bounds below were measured on this AVX2 (`v3`) host on
// 2026-05-26 (see docs/SIMD_EQUIVALENCE.md). The scalar TAIL of these
// kernels also goes through libm, so on the tail elements ULP == 0; the
// SIMD lanes carry the approximation error. We deliberately use sizes
// that are NOT lane-aligned so both regimes are exercised.
//
// SAFETY of domain: vlog/vpow/safe_pow require strictly positive input.
// `pow_midp_unchecked` is `exp2(p*log2(x))` with no domain guard, so we
// only feed positive values to those (the masking pipeline guarantees
// `|magnitude| + SAFE_EPS > 0`). vexp accepts any finite input.
// ===========================================================================

// Committed regression gates (MEASURED 2026-05-26 on AVX2 `v3` host;
// see docs/SIMD_EQUIVALENCE.md). These are NOT bit-exactness assertions
// — `*_midp_unchecked` are approximations. The gate = measured envelope
// rounded up to the next clean bucket so legitimate cross-host jitter
// doesn't flake while a real degradation (e.g. dropping to a `*_lowp`
// tier) still trips. Each kernel asserts BOTH a ULP cap and a max-rel
// cap (rel is the load-bearing one for ln, where ULP near 0 explodes).
//
// `ABS_FLOOR` excludes sub-floor reference magnitudes (catastrophic-
// cancellation residue / underflow) from the ULP/rel envelope: dividing
// a ~1e-16 absolute error by a ~0 reference is meaningless and the
// masking pipeline never consumes sub-floor magnitudes. Skipped counts
// are REPORTED (not hidden). Floor = 1e-12: well below the smallest
// magnitude the masking stage carries (post-offset `>= (1e-5)^3.7 ~
// 1e-18` exists numerically but contributes nothing to the JOD pool).
const MATH_ABS_FLOOR: f64 = 1e-12;

const VEXP_MAX_ULP_GATE: u64 = 24; // measured max 14
const VEXP_MAX_REL_GATE: f64 = 2e-6; // measured 9.67e-7
const VLOG_MAX_ULP_GATE: u64 = 8; // measured max 3 (in-domain)
const VLOG_MAX_REL_GATE: f64 = 1e-6; // measured 2.82e-7
const VPOW_MAX_ULP_GATE: u64 = 64; // measured max 40 in-contract
const VPOW_MAX_REL_GATE: f64 = 1e-5; // measured 2.52e-6 in-contract
// safe_pow composes pow_midp with a subtraction; ULP runs a bit higher
// than bare vpow. Measured max 92 — well inside magetypes' documented
// ~128 ULP budget; gate at 128 (the contract) so a real tier drop trips.
const SAFEPOW_MAX_ULP_GATE: u64 = 128; // measured max 92 in-contract
const SAFEPOW_MAX_REL_GATE: f64 = 2e-5; // measured 8.28e-6

/// Non-lane-aligned sizes so both the SIMD body and the scalar tail run.
/// (lane = 8 on the `v3`/scalar tiers these kernels dispatch to; sizes
/// like 9, 17, 33 force a 1/8-wide remainder; 16/256/4096 are aligned.)
fn math_sizes() -> Vec<usize> {
    vec![
        0, 1, 3, 7, 8, 9, 13, 15, 16, 17, 31, 33, 100, 257, 1000, 4096, 4099,
    ]
}

#[test]
fn vexp_envelope() {
    let mut env = Envelope::with_floor(MATH_ABS_FLOOR);
    let mut rng = Rng::new(0xE5_0001);
    // exp domain in the masking/CSF chain is roughly [-20, 20].
    for &n in &math_sizes() {
        // Random spread across [-20, 20].
        let xs: Vec<f32> = (0..n).map(|_| (rng.unit() - 0.5) * 40.0).collect();
        let mut got = vec![0.0_f32; n];
        api::vexp_into(&xs, &mut got);
        for i in 0..n {
            env.observe(got[i], xs[i].exp());
        }
        // Gaussian-distributed too.
        let xs2: Vec<f32> = (0..n).map(|_| rng.gaussian() * 5.0).collect();
        let mut got2 = vec![0.0_f32; n];
        api::vexp_into(&xs2, &mut got2);
        for i in 0..n {
            env.observe(got2[i], xs2[i].exp());
        }
    }
    // In-contract adversarial: exactly 0, ±integers, ±near-domain-edge,
    // tiny near-0. exp of these is always finite + well-scaled.
    let adv: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 10.0, -10.0, 15.0, -15.0, 19.9, -19.9, 1e-7,
        -1e-7,
    ];
    let mut adv_got = vec![0.0_f32; adv.len()];
    api::vexp_into(&adv, &mut adv_got);
    for i in 0..adv.len() {
        env.observe(adv_got[i], adv[i].exp());
    }

    env.report("vexp (SIMD exp_midp_unchecked vs f32::exp) [in-contract]");
    assert!(
        env.max_ulp <= VEXP_MAX_ULP_GATE && env.max_rel <= VEXP_MAX_REL_GATE,
        "vexp envelope regressed: max_ulp={} (gate {}), max_rel={:.3e} (gate {:.3e}). \
         If the approximation tier changed (e.g. exp_lowp), update the gate WITH \
         measurement + a docs note; do not silently loosen.",
        env.max_ulp,
        VEXP_MAX_ULP_GATE,
        env.max_rel,
        VEXP_MAX_REL_GATE
    );
}

#[test]
fn vlog_envelope() {
    let mut env = Envelope::with_floor(MATH_ABS_FLOOR);
    let mut rng = Rng::new(0xE5_0002);
    // log domain: positive, spanning many orders of magnitude.
    for &n in &math_sizes() {
        // log-uniform 10^[-6, 4].
        let xs: Vec<f32> = (0..n)
            .map(|_| 10f32.powf(rng.unit() * 10.0 - 6.0))
            .collect();
        let mut got = vec![0.0_f32; n];
        api::vlog_into(&xs, &mut got);
        for i in 0..n {
            env.observe(got[i], xs[i].ln());
        }
        // Near 1.0 (where ln→0, relative error is hardest).
        let xs2: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit()).collect();
        let mut got2 = vec![0.0_f32; n];
        api::vlog_into(&xs2, &mut got2);
        for i in 0..n {
            env.observe(got2[i], xs2[i].ln());
        }
    }
    // In-contract adversarial positives, including exact powers of 2,
    // near-1 (hardest rel), and the magnitude span the CSF stage sees.
    // We DELIBERATELY include exactly-1.0 (ln=0): the floor mechanism
    // routes it to below_floor, so it doesn't corrupt the rel envelope
    // but is still counted + reported.
    let adv: Vec<f32> = vec![
        1.0, 2.0, 4.0, 0.5, 0.25, 1e-6, 1e6, 1.0001, 0.9999, 3.0, 100.0, 1024.0,
    ];
    let mut adv_got = vec![0.0_f32; adv.len()];
    api::vlog_into(&adv, &mut adv_got);
    for i in 0..adv.len() {
        env.observe(adv_got[i], adv[i].ln());
    }

    env.report("vlog (SIMD ln_midp_unchecked vs f32::ln) [in-contract]");
    assert!(
        env.max_rel <= VLOG_MAX_REL_GATE,
        "vlog REL envelope regressed: max_rel={:.3e} (gate {:.3e}).",
        env.max_rel,
        VLOG_MAX_REL_GATE
    );
    assert!(
        env.max_ulp <= VLOG_MAX_ULP_GATE,
        "vlog ULP envelope regressed: max_ulp={} (gate {}) at got={} want={} \
         (near-zero ln values would inflate ULP, but the abs floor excludes \
         them — check the rel gate first).",
        env.max_ulp,
        VLOG_MAX_ULP_GATE,
        env.worst_ulp_a,
        env.worst_ulp_b
    );
}

#[test]
fn vpow_envelope() {
    let mut env = Envelope::with_floor(MATH_ABS_FLOOR);
    let mut rng = Rng::new(0xE5_0003);
    // Typical masking exponents (MASK_Q ∈ [1.3, 3.7], MASK_P ≈ 2.26)
    // plus 0.5 / 1.0 / integers. IN-CONTRACT base domain is
    // [SAFE_EPS=1e-5, ~200] (the masking magnitudes + offset). We sweep
    // a slightly wider [1e-3, 200] + log-uniform [1e-4, 1e4] so the
    // tail of the documented domain is covered, but NOT subnormals (out
    // of contract for `pow_midp_unchecked` — see the separate probe).
    let exps: &[f32] = &[0.5, 1.0, 1.3, 1.8, 2.0, 2.26, 3.0, 3.7];
    for &n in &math_sizes() {
        let xs: Vec<f32> = (0..n).map(|_| 1e-3 + rng.unit() * 200.0).collect();
        for &p in exps {
            let mut got = vec![0.0_f32; n];
            api::vpow_into(&xs, &mut got, p);
            for i in 0..n {
                env.observe(got[i], xs[i].powf(p));
            }
        }
        // log-uniform positive (spans magnitudes within range).
        let xs2: Vec<f32> = (0..n).map(|_| 10f32.powf(rng.unit() * 8.0 - 4.0)).collect();
        for &p in exps {
            let mut got = vec![0.0_f32; n];
            api::vpow_into(&xs2, &mut got, p);
            for i in 0..n {
                env.observe(got[i], xs2[i].powf(p));
            }
        }
    }
    // In-contract adversarial positives (no subnormals) × exponents.
    let adv: Vec<f32> = vec![1.0, 2.0, 4.0, 0.5, 1e-4, 1e4, 100.0, 0.001, 255.0];
    for &p in exps {
        let mut adv_got = vec![0.0_f32; adv.len()];
        api::vpow_into(&adv, &mut adv_got, p);
        for i in 0..adv.len() {
            env.observe(adv_got[i], adv[i].powf(p));
        }
    }

    env.report("vpow (SIMD pow_midp_unchecked vs f32::powf) [in-contract]");
    assert!(
        env.max_ulp <= VPOW_MAX_ULP_GATE && env.max_rel <= VPOW_MAX_REL_GATE,
        "vpow envelope regressed: max_ulp={} (gate {}), max_rel={:.3e} (gate {:.3e}). \
         The masking parity claim (~128 ULP / ~1e-5 rel) depends on this — if it's \
         WORSE, surface it; the masking chunk needs revisiting.",
        env.max_ulp,
        VPOW_MAX_ULP_GATE,
        env.max_rel,
        VPOW_MAX_REL_GATE
    );
}

#[test]
fn safe_pow_with_offset_envelope() {
    let mut env = Envelope::with_floor(MATH_ABS_FLOOR);
    let mut rng = Rng::new(0xE5_0004);
    let offset = 1e-5_f32; // SAFE_EPS
    let exps: &[f32] = &[1.3, 1.8, 2.26, 3.0, 3.7];
    for &n in &math_sizes() {
        // masking-stage magnitudes [0, ~200] (non-negative; the pipeline
        // pre-offsets by SAFE_EPS so input + offset > 0). x=0 yields
        // `eps^p - eps^p`; the SIMD path subtracts the EXACT precomputed
        // `eps^p` from the APPROXIMATE `pow_midp(eps)`, so the residue is
        // a ~1e-16-magnitude value vs a 0 reference — the abs floor
        // routes those to below_floor (reported, not asserted).
        let xs: Vec<f32> = (0..n).map(|_| rng.unit() * 200.0).collect();
        for &p in exps {
            let offset_pow_p = offset.powf(p);
            let mut got = vec![0.0_f32; n];
            api::safe_pow_with_offset_into(&xs, &mut got, offset, p, offset_pow_p);
            for i in 0..n {
                let want = (xs[i] + offset).powf(p) - offset_pow_p;
                env.observe(got[i], want);
            }
        }
    }
    // In-contract adversarial: exactly 0 (→ residue near 0, below floor),
    // tiny, mid, large — all non-negative as the pipeline guarantees.
    let adv: Vec<f32> = vec![0.0, 1e-6, 1e-3, 1.0, 10.0, 100.0, 200.0, 0.5];
    for &p in exps {
        let offset_pow_p = offset.powf(p);
        let mut adv_got = vec![0.0_f32; adv.len()];
        api::safe_pow_with_offset_into(&adv, &mut adv_got, offset, p, offset_pow_p);
        for i in 0..adv.len() {
            let want = (adv[i] + offset).powf(p) - offset_pow_p;
            env.observe(adv_got[i], want);
        }
    }

    env.report("safe_pow_with_offset (SIMD vs scalar (x+eps)^p - eps^p) [in-contract]");
    assert!(
        env.max_ulp <= SAFEPOW_MAX_ULP_GATE && env.max_rel <= SAFEPOW_MAX_REL_GATE,
        "safe_pow envelope regressed: max_ulp={} (gate {}), max_rel={:.3e} (gate {:.3e}).",
        env.max_ulp,
        SAFEPOW_MAX_ULP_GATE,
        env.max_rel,
        SAFEPOW_MAX_REL_GATE
    );
}

// ===========================================================================
// Out-of-contract probe: DOCUMENT (not hard-assert) the behavior of the
// `*_midp_unchecked` approximations on inputs the masking pipeline NEVER
// produces — subnormals, exact zero into pow/log, etc. The kernels are
// documented as "positive, in-range inputs only" (see simd_math.rs). The
// brute-force harness surfaced that feeding `f32::MIN_POSITIVE` to
// `pow_midp_unchecked` produces a large WRONG-SIGN value (the unchecked
// log2 has no subnormal guard). This is NOT a bug to fix — it is the
// contract. This test RECORDS the divergence so future agents see it was
// measured and understood, and FAILS only if the kernels start producing
// NaN/Inf (which would propagate into the JOD pool) on these inputs.
// ===========================================================================

#[test]
fn transcendentals_out_of_contract_probe() {
    // Subnormals + exact-zero are OUT of the documented domain for
    // vpow/vlog/safe_pow. We assert only that the SIMD output stays
    // FINITE (no NaN/Inf leak) — the magnitude may be arbitrarily wrong,
    // which is expected for `*_unchecked` on out-of-domain inputs.
    let ooc: Vec<f32> = vec![
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE * 0.5, // subnormal
        f32::from_bits(1),       // smallest subnormal
        1e-30,
        1e-20,
    ];

    // vpow on out-of-contract bases.
    for &p in &[1.3f32, 2.26, 3.7] {
        let mut got = vec![0.0_f32; ooc.len()];
        api::vpow_into(&ooc, &mut got, p);
        for (i, &g) in got.iter().enumerate() {
            assert!(
                g.is_finite(),
                "vpow produced non-finite {g} on out-of-contract input {} p={p} \
                 — would leak NaN/Inf into JOD pool. (Magnitude error on subnormals \
                 is EXPECTED for pow_midp_unchecked; NaN/Inf is not.)",
                ooc[i]
            );
        }
        // Document the actual values (visible under --nocapture).
        println!("  vpow OOC p={p}: {got:?}");
    }

    // vlog on out-of-contract (subnormal) inputs.
    let mut log_got = vec![0.0_f32; ooc.len()];
    api::vlog_into(&ooc, &mut log_got);
    for (i, &g) in log_got.iter().enumerate() {
        assert!(
            g.is_finite(),
            "vlog produced non-finite {g} on out-of-contract input {}",
            ooc[i]
        );
    }
    println!("  vlog OOC subnormals: {log_got:?}");

    // safe_pow with exact-zero input (the cancellation residue).
    let offset = 1e-5_f32;
    let zeros = vec![0.0_f32; 16];
    for &p in &[1.3f32, 2.26, 3.7] {
        let opp = offset.powf(p);
        let mut got = vec![0.0_f32; zeros.len()];
        api::safe_pow_with_offset_into(&zeros, &mut got, offset, p, opp);
        for &g in &got {
            assert!(
                g.is_finite(),
                "safe_pow produced non-finite {g} on x=0 p={p}"
            );
            // The residue must be NEGLIGIBLE in absolute terms (the
            // scalar reference is exactly 0). Bound it well below the
            // masking floor so it can't matter to the JOD pool.
            assert!(
                g.abs() < 1e-9,
                "safe_pow x=0 residue {g} exceeds 1e-9 — the (x+eps)^p approximation \
                 error at x=0 should be far below the masking magnitude floor."
            );
        }
        println!("  safe_pow x=0 p={p}: residue={}", got[0]);
    }
}

// ===========================================================================
// Sanity: ulp_diff machinery itself
// ===========================================================================

#[test]
fn ulp_diff_self_check() {
    assert_eq!(ulp_diff(1.0, 1.0), 0);
    assert_eq!(ulp_diff(0.0, -0.0), 0);
    assert_eq!(ulp_diff(1.0, f32::from_bits(1.0_f32.to_bits() + 1)), 1);
    assert_eq!(ulp_diff(1.0, f32::from_bits(1.0_f32.to_bits() + 4)), 4);
    // Across zero under the IEEE total-order key, the line is
    // ...,-subnormal, -0.0, +0.0, +subnormal,... so smallest positive
    // subnormal to smallest negative subnormal spans 3 steps (through
    // both signed zeros).
    let pos = f32::from_bits(1);
    let neg = -f32::from_bits(1);
    assert_eq!(ulp_diff(pos, neg), 3);
    // +0.0 to +smallest-subnormal is exactly 1 step.
    assert_eq!(ulp_diff(0.0, f32::from_bits(1)), 1);
    // Symmetry on an arbitrary mid-magnitude value (not a named const).
    let a = 12_345.678_f32;
    let b = f32::from_bits(a.to_bits() + 7);
    assert_eq!(ulp_diff(a, b), 7);
    assert_eq!(ulp_diff(b, a), 7);
}
