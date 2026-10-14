//! Differential tests: every `fdlibm64` lane result must equal scalar
//! `libm::pow` bit for bit (NaN payloads and signs included) on every tier
//! this host supports.
use super::*;
use alloc::string::String;
use archmage::SimdToken;

/// Random pairs in the seeded sweep. CI and `just` may raise it with
/// `ZENMETRICS_POW_DIFF_PAIRS`; the default is the acceptance floor.
const DEFAULT_RANDOM_PAIRS: usize = 2_000_000;

/// Every tier available on this host, as `(name, pairs-fn)`.
fn tiers() -> Vec<(&'static str, fn(&[f64], &[f64], &mut [f64]))> {
    let mut t: Vec<(&'static str, fn(&[f64], &[f64], &mut [f64]))> = vec![
        ("scalar", |x, y, o| {
            fdlibm64::vpow64_libm_pairs_kernel_4::<{ fdlibm::coverage::EXACT }, _>(
                ScalarToken,
                x,
                y,
                o,
            )
        }),
        ("dispatch", |x, y, o| vpow64_libm_pairs_into(x, y, o)),
    ];
    #[cfg(target_arch = "x86_64")]
    if archmage::X64V3Token::summon().is_some() {
        t.push(("v3", |x, y, o| {
            v3::vpow64_libm_pairs_into_gated_v3_t::<{ fdlibm::coverage::EXACT }>(
                archmage::X64V3Token::summon().unwrap(),
                x,
                y,
                o,
            )
        }));
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if archmage::X64V4xToken::summon().is_some() {
        t.push(("v4x", |x, y, o| {
            v4x::vpow64_libm_pairs_into_gated_v4x_t::<{ fdlibm::coverage::EXACT }>(
                archmage::X64V4xToken::summon().unwrap(),
                x,
                y,
                o,
            )
        }));
    }
    #[cfg(target_arch = "aarch64")]
    if archmage::NeonToken::summon().is_some() {
        t.push(("neon", |x, y, o| {
            neon::vpow64_libm_pairs_into_gated_neon_t::<{ fdlibm::coverage::EXACT }>(
                archmage::NeonToken::summon().unwrap(),
                x,
                y,
                o,
            )
        }));
    }
    #[cfg(target_arch = "wasm32")]
    if archmage::Wasm128Token::summon().is_some() {
        t.push(("wasm128", |x, y, o| {
            wasm128::vpow64_libm_pairs_into_gated_wasm128_t::<{ fdlibm::coverage::EXACT }>(
                archmage::Wasm128Token::summon().unwrap(),
                x,
                y,
                o,
            )
        }));
    }
    t
}

/// Run `(xs, ys)` through every tier and compare against scalar libm.
/// Returns the number of lanes checked per tier.
fn check(label: &str, xs: &[f64], ys: &[f64]) -> usize {
    assert_eq!(xs.len(), ys.len());
    let want: Vec<u64> = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| libm::pow(x, y).to_bits())
        .collect();
    let mut got = vec![0.0f64; xs.len()];
    for (name, f) in tiers() {
        got.fill(f64::from_bits(0x7ff8_dead_beef_0000));
        f(xs, ys, &mut got);
        let mut bad = 0usize;
        let mut first = String::new();
        for i in 0..xs.len() {
            if got[i].to_bits() != want[i] {
                if bad < 8 {
                    first += &alloc::format!(
                        "\n  x={:#018x} ({:e}) y={:#018x} ({:e}) libm={:#018x} lane={:#018x}",
                        xs[i].to_bits(),
                        xs[i],
                        ys[i].to_bits(),
                        ys[i],
                        want[i],
                        got[i].to_bits()
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(
            bad,
            0,
            "{label}/{name}: {bad} of {} lanes differ:{first}",
            xs.len()
        );
    }
    xs.len()
}

/// SplitMix64: a fixed, seedable stream.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
}

fn step(v: f64, ulps: i64) -> f64 {
    f64::from_bits((v.to_bits() as i64).wrapping_add(ulps) as u64)
}

/// Special and boundary values for `x`, both signs.
fn x_edges() -> Vec<f64> {
    let mut v = vec![
        0.0,
        f64::from_bits(1),
        f64::from_bits(2),
        f64::from_bits(0x000f_ffff_ffff_ff00),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        step(f64::MIN_POSITIVE, 1),
        1e-300,
        1e-30,
        1e-10,
        1e-5,
        0.01,
        0.25,
        0.5,
        0.75,
        0.999_999_999_999,
        step(1.0, -2),
        step(1.0, -1),
        1.0,
        step(1.0, 1),
        step(1.0, 2),
        step(1.0, 7),
        step(1.0, 8),
        step(1.0, -8),
        step(1.0, -9),
        1.5,
        2.0,
        3.0,
        10.0,
        255.0,
        65535.0,
        1e10,
        1e100,
        1e300,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
        f64::from_bits(0x7ff4_0000_0000_0001), // signalling NaN
        f64::from_bits(0x7fff_ffff_ffff_ffff),
    ];
    // Interval-select boundaries: mantissa head j at sqrt(3/2) and sqrt(3),
    // with empty and full low words.
    for j in [0x3988du64, 0x3988e, 0x3988f, 0xbb679, 0xbb67a, 0xbb67b] {
        for e in [0x3feu64, 0x3ff, 0x400, 0x001, 0x7fe] {
            for lo in [0u64, 1, 0xffff_ffff] {
                v.push(f64::from_bits((e << 52) | (j << 32) | lo));
            }
        }
        v.push(f64::from_bits((j << 32) | 0xabcd_1234)); // subnormal, same head
    }
    let neg: Vec<f64> = v.iter().map(|x| -x).collect();
    v.extend(neg);
    v
}

/// Special and boundary values for `y`, both signs.
fn y_edges() -> Vec<f64> {
    let mut v = vec![
        0.0,
        f64::from_bits(1),
        1e-300,
        0.02,
        0.25,
        0.4,
        step(0.5, -1),
        0.5,
        step(0.5, 1),
        1.0 / 3.0,
        step(1.0, -1),
        1.0,
        step(1.0, 1),
        1.5,
        step(2.0, -1),
        2.0,
        step(2.0, 1),
        2.2,
        2.4,
        3.0,
        4.0,
        5.0,
        7.0,
        8.0,
        100.0,
        1023.0,
        1024.0,
        1074.0,
        1075.0,
        1076.0,
        1e5,
        step(2_147_483_648.0, -2),
        2_147_483_647.0,
        2_147_483_648.0,
        step(2_147_483_648.0, 1),
        step(9_007_199_254_740_992.0, -1),
        9_007_199_254_740_992.0,
        1e10,
        1e100,
        1e300,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
        f64::from_bits(0x7ff4_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_0001),
    ];
    let neg: Vec<f64> = v.iter().map(|x| -x).collect();
    v.extend(neg);
    v
}

#[test]
fn pow64_libm_edge_matrix_bit_exact() {
    let (xe, ye) = (x_edges(), y_edges());
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for &x in &xe {
        for &y in &ye {
            xs.push(x);
            ys.push(y);
        }
    }
    // Shift lane alignment so every value visits every lane slot.
    for rot in 0..9 {
        check("edge-matrix", &xs[rot..], &ys[rot..]);
    }
}

/// `z = y·log2(x)` around the overflow (1024), subnormal-output (-1022),
/// underflow (-1075) thresholds and the |z| = 0.5 `n`-extract boundary,
/// ±64 ulp in `y`.
#[test]
fn pow64_libm_range_thresholds_bit_exact() {
    let mut rng = Rng(0x7051_7f00_9e37_babe);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..600 {
        let x = f64::from_bits(rng.next() & 0x7fff_ffff_ffff_ffff);
        if !x.is_finite() || x == 0.0 || x == 1.0 {
            continue;
        }
        let l2 = x.log2();
        for target in [
            1024.0, 1023.5, -0.49, -0.5, -0.51, 0.49, 0.5, 0.51, -1021.0, -1022.0, -1023.0,
            -1074.0, -1075.0, -1075.4, -1076.0,
        ] {
            let y0 = target / l2;
            if !y0.is_finite() {
                continue;
            }
            for d in -64..=64 {
                xs.push(x);
                ys.push(step(y0, d));
            }
        }
    }
    check("thresholds", &xs, &ys);
}

/// Exact half-integer `z` (the `n = [z + 0.5]` rounding ties), plus
/// integer `z`: powers of two raised to `(k + 0.5) / m` and `k / m`.
#[test]
fn pow64_libm_exact_ties_bit_exact() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for m in -1074i32..=1023 {
        if m == 0 {
            continue;
        }
        let x = libm::ldexp(1.0, m.clamp(-1074, 1023) as i32);
        if !x.is_finite() || x == 0.0 {
            continue;
        }
        for k in -2100i32..=2100 {
            for num in [k as f64 + 0.5, k as f64] {
                let y = num / m as f64;
                xs.push(x);
                ys.push(y);
                xs.push(x);
                ys.push(step(y, 1));
                xs.push(x);
                ys.push(step(y, -1));
            }
        }
    }
    check("ties", &xs, &ys);
}

/// Dense positive-normal and subnormal `x` over every exponent with a
/// mantissa stride, crossed with moderate `y` (the metric domain).
#[test]
fn pow64_libm_dense_positive_bit_exact() {
    let ys_grid: Vec<f64> = (-40..=40)
        .map(|k| k as f64 * 0.1)
        .chain([
            0.02,
            -0.02,
            0.4,
            2.4,
            1.0 / 2.4,
            1.0 / 3.0,
            0.6,
            1.3,
            2.26,
            3.7,
        ])
        .collect();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut bits = 1u64;
    while bits < 0x7ff0_0000_0000_0000 {
        let x = f64::from_bits(bits);
        for &y in &ys_grid {
            xs.push(x);
            ys.push(y);
        }
        bits += 0x0008_0000_01a2_3b7d; // odd stride: every exponent, varied mantissas
    }
    check("dense", &xs, &ys);
}

/// Seeded random sweep across four distributions: raw bit patterns (all
/// classes), positive finite `x` with `|y| < 8`, the metric domain
/// `x ∈ (0, 2]`, `y ∈ [-4, 4]`, and `x` near 1 with raw `y`.
#[test]
fn pow64_libm_random_sweep_bit_exact() {
    let total = std::env::var("ZENMETRICS_POW_DIFF_PAIRS")
        .ok()
        .map(|s| s.parse::<usize>().expect("ZENMETRICS_POW_DIFF_PAIRS"))
        .unwrap_or(DEFAULT_RANDOM_PAIRS);
    const CHUNK: usize = 1 << 19;
    let mut rng = Rng(0x5eed_0ff1_1bc0_c0de);
    let names: Vec<&str> = tiers().iter().map(|t| t.0).collect();
    std::eprintln!("pow64_libm random sweep: {total} pairs on tiers {names:?}");
    let mut checked = 0usize;
    let mut xs = vec![0.0f64; CHUNK];
    let mut ys = vec![0.0f64; CHUNK];
    while checked < total {
        let n = CHUNK.min(total - checked);
        for i in 0..n {
            let (x, y) = match rng.next() % 4 {
                0 => (f64::from_bits(rng.next()), f64::from_bits(rng.next())),
                1 => (
                    f64::from_bits(rng.next() % 0x7ff0_0000_0000_0000),
                    rng.range(-8.0, 8.0),
                ),
                2 => (rng.range(0.0, 2.0), rng.range(-4.0, 4.0)),
                _ => (
                    step(1.0, (rng.next() % 8192) as i64 - 4096),
                    f64::from_bits(rng.next()),
                ),
            };
            xs[i] = x;
            ys[i] = y;
        }
        checked += check("random", &xs[..n], &ys[..n]);
    }
    assert!(checked >= total);
}

/// The constant-exponent slice driver, every length class (chunks + tail).
#[test]
fn vpow64_libm_into_matches_scalar() {
    let mut rng = Rng(42);
    for n in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 257, 1000] {
        let xs: Vec<f64> = (0..n).map(|_| rng.range(0.0, 2.0)).collect();
        for p in [0.02f64, 0.4, 2.4, -1.5, 1.0 / 3.0, 0.5, 2.0, 1.0, 0.0] {
            let mut out = vec![0.0; n];
            vpow64_libm_into(&xs, &mut out, p);
            for i in 0..n {
                assert_eq!(
                    out[i].to_bits(),
                    libm::pow(xs[i], p).to_bits(),
                    "n={n} p={p}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// log_libm / log2_libm (e_log.c / e_log2.c ports) — same rigor vs the
// scalar `libm::log`/`libm::log2` oracles.
// ---------------------------------------------------------------------------

/// Every tier available on this host, as `(name, unary-f64 fn)`. `LOG2`
/// selects the e_log2 oracles; `LOG2 = false` the e_log ones.
fn tiers_log<const LOG2: bool>() -> Vec<(&'static str, fn(&[f64], &mut [f64]))> {
    let mut t: Vec<(&'static str, fn(&[f64], &mut [f64]))> = vec![
        ("scalar", |x, o| {
            if LOG2 {
                fdlibm64::vlog2_libm_kernel_4(ScalarToken, x, o)
            } else {
                fdlibm64::vlog_libm_kernel_4(ScalarToken, x, o)
            }
        }),
        ("dispatch", |x, o| {
            if LOG2 {
                vlog2_libm_into(x, o)
            } else {
                vlog_libm_into(x, o)
            }
        }),
    ];
    #[cfg(target_arch = "x86_64")]
    if archmage::X64V3Token::summon().is_some() {
        t.push(("v3", |x, o| {
            if LOG2 {
                v3::vlog2_libm_into_v3_t(archmage::X64V3Token::summon().unwrap(), x, o)
            } else {
                v3::vlog_libm_into_v3_t(archmage::X64V3Token::summon().unwrap(), x, o)
            }
        }));
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if archmage::X64V4xToken::summon().is_some() {
        t.push(("v4x", |x, o| {
            if LOG2 {
                v4x::vlog2_libm_into_v4x_t(archmage::X64V4xToken::summon().unwrap(), x, o)
            } else {
                v4x::vlog_libm_into_v4x_t(archmage::X64V4xToken::summon().unwrap(), x, o)
            }
        }));
    }
    #[cfg(target_arch = "aarch64")]
    if archmage::NeonToken::summon().is_some() {
        t.push(("neon", |x, o| {
            if LOG2 {
                neon::vlog2_libm_into_neon_t(archmage::NeonToken::summon().unwrap(), x, o)
            } else {
                neon::vlog_libm_into_neon_t(archmage::NeonToken::summon().unwrap(), x, o)
            }
        }));
    }
    #[cfg(target_arch = "wasm32")]
    if archmage::Wasm128Token::summon().is_some() {
        t.push(("wasm128", |x, o| {
            if LOG2 {
                wasm128::vlog2_libm_into_wasm128_t(archmage::Wasm128Token::summon().unwrap(), x, o)
            } else {
                wasm128::vlog_libm_into_wasm128_t(archmage::Wasm128Token::summon().unwrap(), x, o)
            }
        }));
    }
    t
}

fn check_log<const LOG2: bool>(label: &str, xs: &[f64]) -> usize {
    let want: Vec<u64> = xs
        .iter()
        .map(|&x| {
            if LOG2 {
                libm::log2(x).to_bits()
            } else {
                libm::log(x).to_bits()
            }
        })
        .collect();
    let mut got = vec![0.0f64; xs.len()];
    for (name, f) in tiers_log::<LOG2>() {
        got.fill(f64::from_bits(0x7ff8_dead_beef_f00d));
        f(xs, &mut got);
        let mut bad = 0usize;
        let mut first = String::new();
        for i in 0..xs.len() {
            if got[i].to_bits() != want[i] {
                if bad < 8 {
                    first += &alloc::format!(
                        "\n  x={:#018x} ({:e}) libm={:#018x} lane={:#018x}",
                        xs[i].to_bits(),
                        xs[i],
                        want[i],
                        got[i].to_bits()
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(
            bad,
            0,
            "{label}/{name}: {bad} of {} lanes differ:{first}",
            xs.len()
        );
    }
    xs.len()
}

/// Edge matrix for both e_log and e_log2: every branch boundary — the
/// subnormal gate (hx < 0x00100000), the x==1 early-out, the
/// inf/NaN gate, the reduce pivots around 0x3fe6a09e, ±ulp each.
#[test]
fn log64_libm_edge_matrix_bit_exact() {
    let mut xs: Vec<f64> = Vec::new();
    for bits in [
        0x0000_0000_0000_0000u64, // +0
        0x0000_0000_0000_0001,    // smallest subnormal
        0x000f_ffff_ffff_ffff,    // largest subnormal
        0x0010_0000_0000_0000,    // smallest normal
        0x3fe6_a09d_ffff_ffff,    // just below the reduce pivot
        0x3fe6_a09e_0000_0000,    // ~sqrt(2)/2 — reduce pivot
        0x3fe6_a09e_667f_3bcd,    // sqrt(2)/2 exactly
        0x3ff0_0000_0000_0000,    // 1.0
        0x3ff0_0000_0000_0001,    // 1+ulp
        0x3fef_ffff_ffff_ffff,    // 1-ulp
        0x4000_0000_0000_0000,    // 2.0
        0x3fe0_0000_0000_0000,    // 0.5
        0x7fe0_0000_0000_0000,    // huge normal
        0x7fef_ffff_ffff_ffff,    // max finite
        0x7ff0_0000_0000_0000,    // +inf
        0x7ff8_0000_0000_0000,    // +NaN
    ] {
        for d in -2i64..=2 {
            let b = (bits as i64).wrapping_add(d);
            if b > 0 {
                xs.push(f64::from_bits(b as u64));
            }
        }
    }
    // Negatives, -0, -inf, -NaN — all take the scalar fallback.
    xs.extend_from_slice(&[
        -0.0f64,
        -1.0,
        -f64::MIN_POSITIVE,
        f64::NEG_INFINITY,
        f64::from_bits(0xfff8_0000_0000_0001),
        f64::NAN,
        f64::INFINITY,
        f64::from_bits(0x7ff0_0000_0000_0001), // sNaN-ish positive
    ]);
    check_log::<false>("log edges", &xs);
    check_log::<true>("log2 edges", &xs);
}

/// Dense sweep across the full positive domain: subnormals through
/// max-finite (ulp-strided bit patterns), plus the live [1e-300, 1e300]
/// value range.
#[test]
fn log64_libm_dense_domain_bit_exact() {
    let mut xs: Vec<f64> = (0..2_000_000u64)
        .map(|i| f64::from_bits(0x0010_0000_0000_0000 + i * 0x10_0000_0001))
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    let mut rng = Rng(0x10c6);
    xs.extend((0..500_000).map(|_| rng.range(1e-300, 1e300)));
    xs.extend((0..300_000).map(|_| rng.range(0.5, 1.5))); // k≈0 band
    check_log::<false>("log dense", &xs);
    check_log::<true>("log2 dense", &xs);
}

#[test]
fn log64_libm_random_sweep_bit_exact() {
    let mut rng = Rng(0x10c);
    let mut xs = vec![0.0f64; 4096];
    for _ in 0..32 {
        for v in xs.iter_mut() {
            *v = match rng.next() % 5 {
                0 => rng.range(0.0, 4.0),
                1 => rng.range(1e-320, 1e-280), // subnormal-heavy
                2 => f64::from_bits(rng.next() & 0x000f_ffff_ffff_ffff), // subnormal
                3 => -rng.range(0.0, 1e300),    // negatives → fallback
                _ => f64::from_bits(rng.next()), // anything incl NaN/inf
            };
        }
        check_log::<false>("log random", &xs[..2048]);
        check_log::<true>("log2 random", &xs[..2048]);
    }
}

/// Slice drivers at every length class.
#[test]
fn vlog64_libm_into_matches_scalar() {
    let mut rng = Rng(11);
    for n in [0usize, 1, 3, 4, 5, 7, 8, 9, 15, 31, 257, 1000] {
        let xs: Vec<f64> = (0..n).map(|_| rng.range(1e-8, 1e8)).collect();
        let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
        vlog_libm_into(&xs, &mut a);
        vlog2_libm_into(&xs, &mut b);
        for i in 0..n {
            assert_eq!(a[i].to_bits(), libm::log(xs[i]).to_bits(), "n={n}");
            assert_eq!(b[i].to_bits(), libm::log2(xs[i]).to_bits(), "n={n}");
        }
    }
}

/// f64 coverage gates must be bit-identical to scalar on their promised
/// domain. `SUBN` is folded into EDGE for this port.
#[test]
fn pow64_libm_gated_tiers_bit_exact_on_domain() {
    use fdlibm::coverage::*;
    let mut rng = Rng(0xBEEF);
    for &p in &[2.4f64, 0.4, -0.02, 6.0, -3.5] {
        let xs: Vec<f64> = (0..2048)
            .map(|i| match i % 4 {
                0 => f64::from_bits(1 + (rng.next() & 0x000f_ffff_ffff_fffe)), // subnormal
                1 => rng.range(1e-307, 1.0), // small normal → underflow result
                2 => rng.range(1.0, 16.0),
                _ => rng.range(1e-150, 1e150),
            })
            .collect();
        let (mut a, mut b) = (vec![0.0; xs.len()], vec![0.0; xs.len()]);
        vpow64_libm_into_gated::<{ EXACT }>(&xs, &mut a, p);
        vpow64_libm_into_gated::<{ POSITIVE_FINITE }>(&xs, &mut b, p);
        for (i, &x) in xs.iter().enumerate() {
            let want = libm::pow(x, p).to_bits();
            assert_eq!(a[i].to_bits(), want, "EXACT p={p} i={i} x={x:e}");
            assert_eq!(b[i].to_bits(), want, "POSITIVE_FINITE p={p} i={i} x={x:e}");
        }
    }
    for &p in &[2.4f64, 0.02, 1.3] {
        let xs: Vec<f64> = (0..1024).map(|_| rng.range(0.25, 4.0)).collect();
        let (mut a, mut b) = (vec![0.0; xs.len()], vec![0.0; xs.len()]);
        vpow64_libm_into_gated::<{ POSITIVE_NORMAL }>(&xs, &mut a, p);
        vpow64_libm_into_gated::<{ BULK }>(&xs, &mut b, p);
        for (i, &x) in xs.iter().enumerate() {
            let want = libm::pow(x, p).to_bits();
            assert_eq!(a[i].to_bits(), want, "POSITIVE_NORMAL p={p} i={i}");
            assert_eq!(b[i].to_bits(), want, "BULK p={p} i={i}");
        }
    }
}
