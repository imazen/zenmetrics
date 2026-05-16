//! Parity test for `kernels::pool::do_pooling_and_jod_still_3ch`
//! against pycvvdp v0.5.4's `cvvdp.do_pooling_and_jods()`.
//!
//! Three Q_per_ch fixtures covering the JOD curve:
//! - near-perfect (~10 JOD)
//! - middling (~9.99 JOD)
//! - strongly distorted (~9.93 JOD)
#![allow(clippy::excessive_precision)]

use cvvdp_gpu::kernels::pool::{
    do_pooling_and_jod_still_3ch, lp_norm_mean, lp_norm_sum, met2jod, pool_band_finalize,
};

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]
#[path = "common/mod.rs"]
mod common;

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]
mod gpu {
    use cubecl::Runtime;
    use cubecl::prelude::*;
    use cvvdp_gpu::kernels::pool::{
        fill_f32_kernel, lp_norm_mean, pool_band_3ch_kernel, pool_band_finalize, pool_band_kernel,
    };

    use super::common::Backend;

    #[test]
    fn pool_band_kernel_matches_host_lp_norm_mean() {
        let client = Backend::client(&Default::default());

        // Deterministic input spanning sign + magnitude range so the
        // safe_pow (|v|+eps)^p - eps^p form actually exercises the
        // epsilon shift on the small values.
        let n = 256usize;
        let band: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 * 0.0123;
                x.sin() * 5.0 + 0.0005 * if i.is_multiple_of(7) { -1.0 } else { 1.0 }
            })
            .collect();
        let beta = 2.0_f32;

        // GPU path: kernel accumulates safe_pow per pixel into a
        // single-slot Atomic<f32> partial; host finalises with
        // pool_band_finalize.
        let band_h = client.create_from_slice(f32::as_bytes(&band));
        let partial_h = client.create_from_slice(f32::as_bytes(&[0.0_f32; 1]));

        let cube_dim = CubeDim::new_1d(64);
        let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

        unsafe {
            pool_band_kernel::launch::<Backend>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(band_h.clone(), n),
                ArrayArg::from_raw_parts(partial_h.clone(), 1),
                beta,
                0_u32,
                n as u32,
            );
        }

        let bytes = client.read_one(partial_h.clone()).expect("read partial");
        let partial: &[f32] = f32::from_bytes(&bytes);
        let gpu_q = pool_band_finalize(partial[0], n, beta);

        let cpu_q = lp_norm_mean(&band, beta);
        let rel = ((gpu_q - cpu_q) / cpu_q.abs().max(1e-6)).abs();
        assert!(
            rel < 5e-4,
            "GPU pool Q = {gpu_q}, CPU lp_norm_mean = {cpu_q}, rel = {rel:.4e}"
        );
    }

    #[test]
    fn pool_band_3ch_kernel_matches_per_channel_kernel() {
        // Each channel's partial must match what the single-channel
        // kernel would produce for that channel alone. Three distinct
        // signal shapes catch a stray cross-channel mix.
        let client = Backend::client(&Default::default());

        let n = 256usize;
        let band_a: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 * 0.0123;
                x.sin() * 5.0
            })
            .collect();
        let band_rg: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 * 0.05;
                x.cos() * 3.0 - 1.5
            })
            .collect();
        let band_vy: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.08).collect();
        let beta = 2.0_f32;

        let band_a_h = client.create_from_slice(f32::as_bytes(&band_a));
        let band_rg_h = client.create_from_slice(f32::as_bytes(&band_rg));
        let band_vy_h = client.create_from_slice(f32::as_bytes(&band_vy));
        // Layout partials so 3ch writes to [3, 5, 7] — non-contiguous
        // indices catch a stray slot-index bug that would have passed
        // with contiguous [0, 1, 2].
        let partials_init = vec![0.0_f32; 10];
        let partials_h = client.create_from_slice(f32::as_bytes(&partials_init));

        let cube_dim = CubeDim::new_1d(64);
        let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

        unsafe {
            pool_band_3ch_kernel::launch::<Backend>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(band_a_h, n),
                ArrayArg::from_raw_parts(band_rg_h, n),
                ArrayArg::from_raw_parts(band_vy_h, n),
                ArrayArg::from_raw_parts(partials_h.clone(), partials_init.len()),
                beta,
                3_u32,
                5_u32,
                7_u32,
                n as u32,
            );
        }

        let bytes = client.read_one(partials_h).expect("read partials");
        let partials: &[f32] = f32::from_bytes(&bytes);

        let q_a = pool_band_finalize(partials[3], n, beta);
        let q_rg = pool_band_finalize(partials[5], n, beta);
        let q_vy = pool_band_finalize(partials[7], n, beta);

        let cpu_a = lp_norm_mean(&band_a, beta);
        let cpu_rg = lp_norm_mean(&band_rg, beta);
        let cpu_vy = lp_norm_mean(&band_vy, beta);

        for (name, gpu, cpu) in [
            ("a", q_a, cpu_a),
            ("rg", q_rg, cpu_rg),
            ("vy", q_vy, cpu_vy),
        ] {
            let rel = ((gpu - cpu) / cpu.abs().max(1e-6)).abs();
            assert!(
                rel < 5e-4,
                "channel {name}: GPU Q = {gpu}, CPU lp_norm_mean = {cpu}, rel = {rel:.4e}"
            );
        }

        // Sanity: untouched partial slots stayed at zero — proves the
        // kernel didn't accidentally write outside its target indices.
        // Bit-pattern equality is the right test: the slots are
        // initialized to 0.0 and never written, so they should retain
        // the all-zero IEEE-754 bit pattern. `.to_bits()` form
        // sidesteps clippy::float_cmp's conservative warning.
        for i in [0, 1, 2, 4, 6, 8, 9] {
            assert_eq!(
                partials[i].to_bits(),
                0.0_f32.to_bits(),
                "untouched partial slot {i} got written ({})",
                partials[i]
            );
        }
    }

    #[test]
    fn fill_f32_kernel_writes_uniform_value() {
        // Used by the baseband CSF path to fill log_l_bkg with the
        // scalar log_l_bkg_baseband. Simple but a regression gate:
        // a stray + 1 in the kernel body or a wrong index would
        // show up as an off-by-one fill or a zero-trail.
        let client = Backend::client(&Default::default());

        let n = 128usize;
        // Pre-seed with a sentinel so any unwritten slot would
        // surface as `7.0` in the assertion.
        let dest_init = vec![7.0_f32; n];
        let dest_h = client.create_from_slice(f32::as_bytes(&dest_init));

        let value = -1.23456_f32;
        let cube_dim = CubeDim::new_1d(64);
        let cube_count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);

        unsafe {
            fill_f32_kernel::launch::<Backend>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(dest_h.clone(), n),
                value,
                n as u32,
            );
        }

        let bytes = client.read_one(dest_h).expect("read dest");
        let dest: &[f32] = f32::from_bytes(&bytes);
        // fill_f32_kernel writes `value` byte-for-byte; bit-pattern
        // equality is the right test (clippy::float_cmp would
        // otherwise complain about ==).
        for (i, &v) in dest.iter().enumerate() {
            assert_eq!(
                v.to_bits(),
                value.to_bits(),
                "slot {i} = {v}, expected {value} (sentinel was 7.0 → would be visible if unwritten)"
            );
        }
    }
}

#[test]
fn pool_near_perfect_matches_pycvvdp() {
    let q_per_ch = vec![[0.01_f32; 3]; 8];
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    assert!(
        (jod - 10.0).abs() < 1e-3,
        "near-perfect: got {jod}, expected ~10.0"
    );
}

#[test]
fn pool_middling_matches_pycvvdp() {
    // ch0..2 rows, 8 bands each. Layout: q[k] = [ch0, ch1, ch2].
    let ch = [
        [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.04],
        [0.4, 0.25, 0.18, 0.12, 0.08, 0.06, 0.04, 0.03],
        [0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02],
    ];
    let q_per_ch: Vec<[f32; 3]> = (0..8).map(|k| [ch[0][k], ch[1][k], ch[2][k]]).collect();
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    let expected = 9.987_316_f32;
    assert!(
        (jod - expected).abs() < 1e-3,
        "middling: got {jod}, expected {expected}"
    );
}

#[test]
fn pool_strong_matches_pycvvdp() {
    let ch = [
        [2.5, 1.5, 1.0, 0.8, 0.5, 0.4],
        [2.0, 1.2, 0.8, 0.6, 0.4, 0.3],
        [1.5, 0.9, 0.6, 0.5, 0.3, 0.2],
    ];
    let q_per_ch: Vec<[f32; 3]> = (0..6).map(|k| [ch[0][k], ch[1][k], ch[2][k]]).collect();
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    let expected = 9.931_840_f32;
    assert!(
        (jod - expected).abs() < 1e-3,
        "strong: got {jod}, expected {expected}"
    );
}

#[test]
fn met2jod_continuous_at_kink() {
    // The piecewise transform is C0 at Q=0.1; verify the two
    // branches agree there to within f32 epsilon.
    let q = 0.1_f32;
    let from_low = met2jod(q);
    let from_high = met2jod(q + 1e-6);
    assert!(
        (from_low - from_high).abs() < 1e-3,
        "discontinuity at Q=0.1: low={from_low}, high={from_high}"
    );
}

#[test]
fn met2jod_clamps_at_origin() {
    // Q=0 should give JOD=10 (no perceptible difference).
    let jod = met2jod(0.0);
    assert!((jod - 10.0).abs() < 1e-6, "met2jod(0) = {jod}, expected 10");
}

#[test]
#[should_panic(expected = "need at least one pyramid level")]
fn do_pooling_and_jod_panics_on_empty_q_per_ch() {
    // `do_pooling_and_jod_still_3ch` has a documented `# Panics`
    // section: "Panics if `q_per_ch` is empty (`n_levels == 0`)".
    // Pin the contract — a refactor that silently returns 0.0 or
    // NaN on empty input would mask upstream bugs (e.g. a pyramid
    // that built zero bands due to a min-dim regression).
    let empty: Vec<[f32; 3]> = Vec::new();
    let _ = do_pooling_and_jod_still_3ch(&empty);
}

// `lp_norm_sum` is a public API but had no direct unit tests — its
// behaviour was only exercised transitively through
// `do_pooling_and_jod_still_3ch`. Pin a few hand-computable cases
// so a refactor of the `safe_pow_lp` shift (which lives inside
// pool.rs and is the source of subtle bit-level drift) trips here
// before propagating into the JOD parity tests.

#[test]
fn lp_norm_sum_pythagorean_triple_at_p2() {
    // Classic L2: lp_norm_sum([3, 4], 2) returns
    //   sqrt(sum_i((|x_i| + eps)^p - eps^p) + eps) - eps^(1/p)
    // which simplifies to ~sqrt(25) - sqrt(eps) at this scale.
    // The outer eps^(1/p) is NOT negligible — for p=2 with
    // eps=1e-5 it's ~0.00316, large enough to be the dominant
    // error term against the naive L2 result of 5.0. The form
    // is documented as cvvdp's `safe_pow` shape — it's the
    // differentiable-at-zero family that the pool kernels match
    // exactly so the cvvdp parity stays bit-stable.
    let got = lp_norm_sum(&[3.0, 4.0], 2.0);
    let eps_tail = (1e-5_f32).sqrt(); // ≈ 0.0031623
    let expected = 5.0 - eps_tail;
    assert!(
        (got - expected).abs() < 1e-3,
        "lp_norm_sum([3, 4], 2) = {got}, expected ~{expected} (naive L2=5 minus eps^(1/2)={eps_tail})",
    );
}

#[test]
fn lp_norm_sum_handles_negative_signs_via_abs() {
    // safe_pow_lp takes |x| first — so sign of inputs must not
    // change the output. Pin so a refactor that drops the .abs()
    // (and silently squares-negatives-back-positive at p=2 but
    // breaks at odd p) trips immediately.
    let pos = lp_norm_sum(&[3.0, 4.0], 2.0);
    let mixed = lp_norm_sum(&[-3.0, 4.0], 2.0);
    let neg = lp_norm_sum(&[-3.0, -4.0], 2.0);
    assert_eq!(
        pos.to_bits(),
        mixed.to_bits(),
        "sign of one input changed lp_norm_sum: pos={pos}, mixed={mixed}",
    );
    assert_eq!(
        pos.to_bits(),
        neg.to_bits(),
        "sign of both inputs changed lp_norm_sum: pos={pos}, neg={neg}",
    );
}

#[test]
fn lp_norm_sum_zero_input_returns_zero() {
    // safe_pow_lp(0, p) = (0 + eps)^p - eps^p = 0, so any all-zero
    // input vector accumulates zero before the outer safe_pow_lp,
    // which itself produces 0 at any p. The empty case is also
    // zero (sum-of-empty is 0).
    for n in [0_usize, 1, 5, 64] {
        let v = vec![0.0_f32; n];
        let got = lp_norm_sum(&v, 2.0);
        assert!(
            got.abs() < 1e-6,
            "lp_norm_sum([0; {n}], 2) = {got}, expected 0",
        );
    }
}

// `pool_band_finalize` is a public API that wraps the host-side
// fold for the GPU `pool_band_kernel` / `pool_band_3ch_kernel`
// atomic partials. Its closed-form algebra is
// `((partial / n).max(0) + eps)^(1/β) - eps^(1/β)`. Until this
// section it was exercised only INDIRECTLY: the kernel produced
// a partial, finalize was called, and the result was compared to
// `lp_norm_mean`. A refactor that drops the `.max(0)` clamp
// (atomic-noise negatives), the `- eps^(1/β)` tail (silently
// inflating Q by ~3e-3 at β=2), or the `partial / n` division
// (inflates Q by ~n^(1/β)) would not have surfaced in the
// indirect tests because the kernel happened to never produce
// those inputs. Pin the algebra directly so each invariant
// trips its own test, not a downstream parity gate where the
// root cause is harder to find.

#[test]
fn pool_band_finalize_zero_partial_returns_zero() {
    // With partial=0 the formula is
    //   ((0 / n).max(0) + eps)^(1/β) - eps^(1/β) = 0
    // i.e. the eps^(1/β) tail subtraction must exactly cancel
    // the eps^(1/β) head term. Pin so a refactor that drops the
    // tail subtraction silently floors every band's Q at
    // eps^(1/β) (~0.003 at β=2, ~0.056 at β=4) and inflates
    // downstream JOD.
    for &beta in &[1.0_f32, 2.0, 4.0, 8.0] {
        for &n in &[1_usize, 64, 1024, 65536] {
            let q = pool_band_finalize(0.0, n, beta);
            assert!(
                q.abs() < 1e-6,
                "pool_band_finalize(0, {n}, {beta}) = {q}, expected 0 (eps^(1/β) tail must cancel head)",
            );
        }
    }
}

#[test]
fn pool_band_finalize_negative_partial_clamps_to_zero() {
    // Atomic-f32 fetch_add on tiny opposing values can produce a
    // small negative partial through rounding even when the
    // theoretical sum is non-negative. The `.max(0)` guard keeps
    // the (partial/n) input to safe_pow non-negative — without it,
    // (negative + eps)^(1/β) returns NaN at non-integer β and
    // propagates through the pool fold into the JOD score. Pin
    // the guard explicitly.
    for &beta in &[2.0_f32, 4.0] {
        for &partial in &[-1e-3_f32, -1.0, -1e6] {
            let q = pool_band_finalize(partial, 1024, beta);
            assert!(
                q.is_finite() && q.abs() < 1e-6,
                "pool_band_finalize({partial}, 1024, {beta}) = {q}, expected 0 (negative-clamp safety)",
            );
        }
    }
}

#[test]
fn pool_band_finalize_matches_lp_norm_mean_on_synth_signal() {
    // The kernel-finalize contract is: feeding the kernel's per-
    // pixel-accumulated safe_pow partial into pool_band_finalize
    // reproduces `lp_norm_mean(values, β)` to f32 precision. This
    // is the exact identity the GPU pool kernels rely on; the
    // existing `pool_band_kernel_matches_host_lp_norm_mean` test
    // confirms it through the GPU path (which requires a backend
    // and an atomic-f32-capable runtime), but a scalar version of
    // the same identity belongs in pool_scalar.rs so a CPU-only
    // test run (e.g. cubecl-cpu CI without GPU) still trips on
    // host-side regressions.
    let eps: f32 = 1e-5;
    let signal: Vec<f32> = (0..256_usize)
        .map(|i| {
            let x = i as f32 * 0.0123;
            x.sin() * 5.0 + 0.0005 * if i.is_multiple_of(7) { -1.0 } else { 1.0 }
        })
        .collect();
    let beta = 2.0_f32;
    // Build the kernel's per-pixel-accumulated partial in scalar
    // form: sum over pixels of (|v| + eps)^β - eps^β.
    let partial: f32 = signal
        .iter()
        .map(|&v| (v.abs() + eps).powf(beta) - eps.powf(beta))
        .sum();
    let from_finalize = pool_band_finalize(partial, signal.len(), beta);
    let from_mean = lp_norm_mean(&signal, beta);
    let rel = ((from_finalize - from_mean) / from_mean.abs().max(1e-6)).abs();
    assert!(
        rel < 1e-5,
        "pool_band_finalize from partial = {from_finalize}, lp_norm_mean = {from_mean}, rel = {rel:.4e}",
    );
}

#[test]
fn pool_band_finalize_eps_tail_is_substantial_at_low_beta() {
    // Same observation as `lp_norm_sum_pythagorean_triple_at_p2`:
    // the outer `- eps^(1/β)` tail is NOT a rounding-error term.
    // At β=1 eps^(1/β) = eps = 1e-5 (negligible), but at β=2 it
    // becomes ~3.16e-3 and at β=4 ~0.056. Pin both so a refactor
    // that drops the tail at β=2 (which "looks safe" because 1e-5
    // is small) gets caught — the actual tail is 316× larger at
    // β=2 than β=1.
    for (beta, expected_tail) in [(1.0_f32, 1e-5_f32), (2.0, 3.162_28e-3), (4.0, 5.623_4e-2)] {
        // partial=0 isolates the tail magnitude: result is
        // (eps).powf(1/β) - (eps).powf(1/β) = 0, so we instead
        // measure indirectly by computing what pool_band_finalize
        // returns for partial = eps*n. That gives
        //   ((eps*n/n) + eps)^(1/β) - eps^(1/β)
        //   = (2*eps)^(1/β) - eps^(1/β)
        //   = eps^(1/β) * (2^(1/β) - 1)
        let n = 16usize;
        let partial = (1e-5_f32) * n as f32;
        let q = pool_band_finalize(partial, n, beta);
        let expected = expected_tail * ((2.0_f32).powf(1.0 / beta) - 1.0);
        assert!(
            (q - expected).abs() < expected.abs() * 1e-3 + 1e-8,
            "β={beta}: pool_band_finalize(eps*n, n, β) = {q}, expected ~{expected} (tail eps^(1/β)={expected_tail})",
        );
    }
}

#[test]
fn pool_band_finalize_divides_by_n() {
    // The `partial / n` step is what makes this lp_norm_*MEAN*
    // (not sum). A refactor that drops the division (or uses
    // n=1) would inflate Q by ~n^(1/β). Pin by feeding identical
    // partials at different n: the result must shrink as n grows
    // (because partial/n shrinks). With partial = K and varying
    // n, expected: (K/n + eps)^(1/β) - eps^(1/β), which is
    // strictly monotonic decreasing in n for K > 0, β > 0.
    let partial = 100.0_f32;
    let beta = 2.0_f32;
    let q_n1 = pool_band_finalize(partial, 1, beta);
    let q_n100 = pool_band_finalize(partial, 100, beta);
    let q_n10000 = pool_band_finalize(partial, 10000, beta);
    assert!(
        q_n1 > q_n100 && q_n100 > q_n10000,
        "pool_band_finalize must be strictly decreasing in n_pixels under fixed partial: q_n1={q_n1}, q_n100={q_n100}, q_n10000={q_n10000}",
    );
    // Exact algebra check at n=100, β=2:
    //   ((100/100) + 1e-5)^(1/2) - (1e-5)^(1/2)
    //   = sqrt(1.00001) - sqrt(1e-5)
    //   ≈ 1.000005 - 0.00316228
    let eps: f32 = 1e-5;
    let expected_n100 = (1.0_f32 + eps).sqrt() - eps.sqrt();
    assert!(
        (q_n100 - expected_n100).abs() < 1e-5,
        "pool_band_finalize(100, 100, 2) = {q_n100}, expected closed-form ~{expected_n100}",
    );
}

#[test]
fn lp_norm_sum_scales_with_count_under_uniform_input() {
    // For uniform input [a; n] at exponent p, the eps-shifted form
    // produces ~(n)^(1/p) * |a| MINUS the outer eps^(1/p) tail.
    // Pin two specific counts so a regression that breaks the
    // sum/mean split (e.g. accidentally dividing by n inside
    // lp_norm_sum, turning it into lp_norm_mean) trips here. The
    // eps_tail term at p=4 is ~0.0562 — substantial enough to
    // matter, so we subtract it explicitly rather than loosening
    // the tolerance to mask it.
    let a = 2.5_f32;
    let p = 4.0_f32;
    let eps_tail = (1e-5_f32).powf(1.0 / p); // ≈ 0.0562
    let got_n1 = lp_norm_sum(&[a], p);
    let got_n16 = lp_norm_sum(&[a; 16], p);
    let expected_n1 = a - eps_tail; // 1^(1/4) * a - eps^(1/4)
    let expected_n16 = (16f32).powf(1.0 / p) * a - eps_tail; // = 2 * a - eps_tail = 4.944
    assert!(
        (got_n1 - expected_n1).abs() < 1e-3,
        "lp_norm_sum([{a}], 4) = {got_n1}, expected ~{expected_n1} (a={a} minus eps^(1/4)={eps_tail})",
    );
    assert!(
        (got_n16 - expected_n16).abs() < 1e-3,
        "lp_norm_sum([{a}; 16], 4) = {got_n16}, expected ~{expected_n16} (n^(1/p)*a={} minus eps^(1/4)={eps_tail})",
        (16f32).powf(1.0 / p) * a,
    );
}

// `lp_norm_mean` is the sibling of `lp_norm_sum` (cvvdp's
// `lp_norm` with `normalize=True`). It's exercised through the
// GPU-gated pool_band_kernel parity test and via the
// `pool_band_finalize_matches_lp_norm_mean_on_synth_signal` test
// (one synthetic input), but had no direct unit tests pinning its
// individual algebra invariants. Same gap-shape as the
// `lp_norm_sum_*` closure (tick 351) — pin closed-form cases so
// the eps-shift algebra is locked at this entry point too.

#[test]
fn lp_norm_mean_empty_input_returns_zero() {
    // Documented contract: `if values.is_empty() { return 0.0 }`
    // — early-return before the division-by-zero in `acc / n`.
    // Pin so a refactor that drops the guard surfaces here
    // immediately (without it, n=0 yields NaN, which then
    // poisons the JOD pipeline).
    for p in [1.0_f32, 2.0, 4.0, 8.0] {
        let got = lp_norm_mean(&[], p);
        assert_eq!(
            got.to_bits(),
            0.0_f32.to_bits(),
            "lp_norm_mean([], p={p}) = {got}, expected exactly 0",
        );
    }
}

#[test]
fn lp_norm_mean_uniform_input_returns_a() {
    // For any uniform input [a; n] at any p > 0:
    //   sum_i((|a|+eps)^p - eps^p) = n * ((|a|+eps)^p - eps^p)
    //   ÷ n = (|a|+eps)^p - eps^p
    //   safe_pow(..., 1/p) = (|a|+eps+eps - eps^(1/p))... actually:
    //   ((|a|+eps)^p - eps^p + eps)^(1/p) - eps^(1/p)
    // For |a| ≫ eps^(1/p), this reduces to ≈ |a|. Pin two
    // (a, p) shapes spanning small/mid magnitudes. Catches a
    // refactor that drops the divide-by-n step (which would
    // turn lp_norm_mean into lp_norm_sum and overestimate by
    // ~n^(1/p)).
    for &(a, p) in &[(0.5_f32, 2.0), (2.5, 4.0)] {
        let eps_tail = (1e-5_f32).powf(1.0 / p);
        for n in [1_usize, 4, 16, 64] {
            let v = vec![a; n];
            let got = lp_norm_mean(&v, p);
            // Account for the outer eps^(1/p) tail subtraction;
            // the inner term ((|a|+eps)^p - eps^p) is dominated
            // by |a|^p for |a| ≫ eps so we use |a| as the lead.
            let rel_err = ((got - (a - eps_tail)).abs()) / a;
            assert!(
                rel_err < 1e-3,
                "lp_norm_mean([{a}; {n}], {p}) = {got}, expected ≈ {} (uniform-invariance + eps tail = {eps_tail})",
                a - eps_tail,
            );
        }
    }
}

#[test]
fn lp_norm_mean_handles_negative_signs_via_abs() {
    // safe_pow_lp takes |x|, so sign of inputs must not change
    // the output. Same property as lp_norm_sum's
    // `_handles_negative_signs_via_abs`, pinned separately
    // because lp_norm_mean has its own copy of the call site —
    // a refactor that drops `.abs()` from one but not the other
    // surfaces here vs the sibling test.
    let pos = lp_norm_mean(&[3.0, 4.0, 5.0], 2.0);
    let mixed = lp_norm_mean(&[-3.0, 4.0, -5.0], 2.0);
    let neg = lp_norm_mean(&[-3.0, -4.0, -5.0], 2.0);
    assert_eq!(
        pos.to_bits(),
        mixed.to_bits(),
        "sign of inputs changed lp_norm_mean: pos={pos}, mixed={mixed}",
    );
    assert_eq!(
        pos.to_bits(),
        neg.to_bits(),
        "sign of inputs changed lp_norm_mean: pos={pos}, neg={neg}",
    );
}

#[test]
fn lp_norm_mean_relates_to_lp_norm_sum_by_n_root() {
    // The defining identity:
    //   lp_norm_sum(v, p) ≈ n^(1/p) * lp_norm_mean(v, p)
    // (exact in the limit eps → 0). Both functions share the
    // safe_pow_lp eps-shift; the only structural difference is
    // the `/ n` division in lp_norm_mean before the outer
    // safe_pow. Pin so a refactor that changes the eps shift
    // in only one of them surfaces immediately.
    //
    // Tolerance: the outer `- eps^(1/p)` tail is the same
    // constant in both functions, but the asymmetric shift on
    // a smaller `mean` magnitude creates a small relative
    // mismatch. At p=2 the tail is ≈ 3.16e-3; at p=4 it's
    // ≈ 5.62e-2 — proportionally bigger. Use 1.5e-2 to absorb
    // p=4 at signal magnitudes of a few; this still catches a
    // structural divergence (where `/ n` is dropped or applied
    // twice, which would be order-1 relative error).
    let signal: Vec<f32> = (0..8_usize).map(|i| (i as f32 + 1.0) * 0.7).collect();
    let n = signal.len() as f32;
    for p in [2.0_f32, 4.0] {
        let s = lp_norm_sum(&signal, p);
        let m = lp_norm_mean(&signal, p);
        let scale = n.powf(1.0 / p);
        let predicted = scale * m;
        let rel = ((s - predicted) / s).abs();
        assert!(
            rel < 1.5e-2,
            "lp_norm_sum vs n^(1/p) * lp_norm_mean diverged at p={p}: \
             sum={s}, mean={m}, n^(1/p)={scale}, predicted={predicted} (rel={rel:.4e})",
        );
    }
}
