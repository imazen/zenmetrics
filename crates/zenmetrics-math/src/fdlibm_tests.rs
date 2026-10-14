//! Differential tests: every `fdlibm` lane result must equal scalar
//! `libm::powf` bit for bit (NaN payloads and signs included) on every tier
//! this host supports.
use super::*;
use alloc::string::String;
use archmage::SimdToken;

/// Random pairs in the seeded sweep. CI and `just` may raise it with
/// `ZENMETRICS_POWF_DIFF_PAIRS`; the default is the acceptance floor.
const DEFAULT_RANDOM_PAIRS: usize = 10_000_000;

/// Every tier available on this host, as `(name, pairs-fn)`.
fn tiers() -> Vec<(&'static str, fn(&[f32], &[f32], &mut [f32]))> {
    let mut t: Vec<(&'static str, fn(&[f32], &[f32], &mut [f32]))> = vec![
        ("scalar", |x, y, o| {
            fdlibm::vpowf_libm_pairs_kernel_8::<{ fdlibm::coverage::EXACT }, _>(ScalarToken, x, y, o)
        }),
        ("dispatch", |x, y, o| vpowf_libm_pairs_into(x, y, o)),
    ];
    #[cfg(target_arch = "x86_64")]
    if archmage::X64V3Token::summon().is_some() {
        t.push(("v3", |x, y, o| {
            v3::vpowf_libm_pairs_into_gated_v3_t::<{ fdlibm::coverage::EXACT }>(archmage::X64V3Token::summon().unwrap(), x, y, o)
        }));
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if archmage::X64V4xToken::summon().is_some() {
        t.push(("v4x", |x, y, o| {
            v4x::vpowf_libm_pairs_into_gated_v4x_t::<{ fdlibm::coverage::EXACT }>(archmage::X64V4xToken::summon().unwrap(), x, y, o)
        }));
    }
    #[cfg(target_arch = "aarch64")]
    if archmage::NeonToken::summon().is_some() {
        t.push(("neon", |x, y, o| {
            neon::vpowf_libm_pairs_into_gated_neon_t::<{ fdlibm::coverage::EXACT }>(archmage::NeonToken::summon().unwrap(), x, y, o)
        }));
    }
    #[cfg(target_arch = "wasm32")]
    if archmage::Wasm128Token::summon().is_some() {
        t.push(("wasm128", |x, y, o| {
            wasm128::vpowf_libm_pairs_into_gated_wasm128_t::<{ fdlibm::coverage::EXACT }>(archmage::Wasm128Token::summon().unwrap(), x, y, o)
        }));
    }
    t
}

/// Run `(xs, ys)` through every tier and compare against scalar libm.
/// Returns the number of lanes checked per tier.
fn check(label: &str, xs: &[f32], ys: &[f32]) -> usize {
    assert_eq!(xs.len(), ys.len());
    let want: Vec<u32> = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| libm::powf(x, y).to_bits())
        .collect();
    let mut got = vec![0.0f32; xs.len()];
    for (name, f) in tiers() {
        got.fill(f32::from_bits(0x7fc0_dead));
        f(xs, ys, &mut got);
        let mut bad = 0usize;
        let mut first = String::new();
        for i in 0..xs.len() {
            if got[i].to_bits() != want[i] {
                if bad < 8 {
                    first += &alloc::format!(
                        "\n  x={:#010x} ({:e}) y={:#010x} ({:e}) libm={:#010x} lane={:#010x}",
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
    fn bits(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
}

fn step(v: f32, ulps: i32) -> f32 {
    f32::from_bits((v.to_bits() as i32).wrapping_add(ulps) as u32)
}

/// Special and boundary values for `x`, both signs.
fn x_edges() -> Vec<f32> {
    let mut v = vec![
        0.0,
        f32::from_bits(1),
        f32::from_bits(2),
        f32::from_bits(0x0000_0fff),
        f32::from_bits(0x0040_0000),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        step(f32::MIN_POSITIVE, 1),
        1e-30,
        1e-10,
        1e-5,
        0.01,
        0.25,
        0.5,
        0.75,
        0.999_999,
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
        1e30,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0x7fa0_0001), // signalling NaN
        f32::from_bits(0x7fff_ffff),
    ];
    // Interval-select boundaries: mantissa j at sqrt(3/2) and sqrt(3).
    for j in [
        0x1c_c470u32,
        0x1c_c471,
        0x1c_c472,
        0x5d_b3d6,
        0x5d_b3d7,
        0x5d_b3d8,
    ] {
        for e in [
            0x3f00_0000u32,
            0x3f80_0000,
            0x4000_0000,
            0x0080_0000,
            0x7f00_0000,
        ] {
            v.push(f32::from_bits(e | j));
        }
        v.push(f32::from_bits(j)); // subnormal with the same mantissa
    }
    let neg: Vec<f32> = v.iter().map(|x| -x).collect();
    v.extend(neg);
    v
}

/// Special and boundary values for `y`, both signs.
fn y_edges() -> Vec<f32> {
    let mut v = vec![
        0.0,
        f32::from_bits(1),
        1e-30,
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
        127.0,
        128.0,
        149.0,
        150.0,
        151.0,
        1e5,
        16_777_215.0,
        16_777_216.0,
        16_777_218.0,
        step(134_217_728.0, -1),
        134_217_728.0,
        step(134_217_728.0, 1),
        1e10,
        1e30,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0x7fa0_0001),
        f32::from_bits(0xffc0_0001),
    ];
    let neg: Vec<f32> = v.iter().map(|x| -x).collect();
    v.extend(neg);
    v
}

#[test]
fn powf_libm_edge_matrix_bit_exact() {
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
    for rot in 0..17 {
        check("edge-matrix", &xs[rot..], &ys[rot..]);
    }
}

/// `z = y·log2(x)` around the overflow (128), subnormal-output (-126),
/// underflow (-150) thresholds and their neighbours, ±64 ulp in `y`.
#[test]
fn powf_libm_range_thresholds_bit_exact() {
    let mut rng = Rng(0x7051_7f00);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..600 {
        let x = f32::from_bits(rng.bits() & 0x7fff_ffff);
        if !x.is_finite() || x == 0.0 || x == 1.0 {
            continue;
        }
        let l2 = (x as f64).log2();
        for target in [
            128.0, 127.5, -125.0, -126.0, -127.0, -140.0, -149.0, -150.0, -151.0,
        ] {
            let y0 = (target / l2) as f32;
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
fn powf_libm_exact_ties_bit_exact() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for m in -149i32..=127 {
        if m == 0 {
            continue;
        }
        let x = libm::ldexpf(1.0, m);
        for k in -300i32..=300 {
            for num in [k as f32 + 0.5, k as f32] {
                let y = num / m as f32;
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
fn powf_libm_dense_positive_bit_exact() {
    let ys_grid: Vec<f32> = (-40..=40)
        .map(|k| k as f32 * 0.1)
        .chain([0.02, -0.02, 0.4, 2.4, 1.0 / 2.4, 0.6, 1.3, 2.26, 3.7])
        .collect();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut bits = 1u32;
    while bits < 0x7f80_0000 {
        let x = f32::from_bits(bits);
        for &y in &ys_grid {
            xs.push(x);
            ys.push(y);
        }
        bits += 0x0000_7f3b; // odd stride: every exponent, varied mantissas
    }
    check("dense", &xs, &ys);
}

/// Seeded random sweep, ≥ 10 M pairs across four distributions:
/// raw bit patterns (all classes), positive finite `x` with `|y| < 8`,
/// the metric domain `x ∈ (0, 2]`, `y ∈ [-4, 4]`, and `x` near 1 with
/// raw `y`.
#[test]
fn powf_libm_random_10m_bit_exact() {
    let total = std::env::var("ZENMETRICS_POWF_DIFF_PAIRS")
        .ok()
        .map(|s| s.parse::<usize>().expect("ZENMETRICS_POWF_DIFF_PAIRS"))
        .unwrap_or(DEFAULT_RANDOM_PAIRS);
    const CHUNK: usize = 1 << 20;
    let mut rng = Rng(0x5eed_0ff1_1bc0_c0de);
    let names: Vec<&str> = tiers().iter().map(|t| t.0).collect();
    std::eprintln!("powf_libm random sweep: {total} pairs on tiers {names:?}");
    let mut checked = 0usize;
    let mut xs = vec![0.0f32; CHUNK];
    let mut ys = vec![0.0f32; CHUNK];
    while checked < total {
        let n = CHUNK.min(total - checked);
        for i in 0..n {
            let (x, y) = match rng.next() % 4 {
                0 => (f32::from_bits(rng.bits()), f32::from_bits(rng.bits())),
                1 => (
                    f32::from_bits(rng.bits() % 0x7f80_0000),
                    rng.range(-8.0, 8.0),
                ),
                2 => (rng.range(0.0, 2.0), rng.range(-4.0, 4.0)),
                _ => (
                    step(1.0, (rng.bits() % 4096) as i32 - 2048),
                    f32::from_bits(rng.bits()),
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
fn vpow_libm_into_matches_scalar() {
    let mut rng = Rng(42);
    for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 33, 257, 1000] {
        let xs: Vec<f32> = (0..n).map(|_| rng.range(0.0, 2.0)).collect();
        for p in [0.02f32, 0.4, 2.4, -1.5, 0.5, 2.0, 1.0, 0.0] {
            let mut out = vec![0.0; n];
            vpow_libm_into(&xs, &mut out, p);
            for i in 0..n {
                assert_eq!(
                    out[i].to_bits(),
                    libm::powf(xs[i], p).to_bits(),
                    "n={n} p={p}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// expf_libm (e_expf port) — same differential rigor vs scalar `libm::expf`.
// ---------------------------------------------------------------------------

/// Every tier available on this host, as `(name, unary-fn)`.
fn tiers_expf() -> Vec<(&'static str, fn(&[f32], &mut [f32]))> {
    let mut t: Vec<(&'static str, fn(&[f32], &mut [f32]))> = vec![
        ("scalar", |x, o| {
            fdlibm::vexpf_libm_kernel_8(ScalarToken, x, o)
        }),
        ("dispatch", |x, o| vexpf_libm_into(x, o)),
    ];
    #[cfg(target_arch = "x86_64")]
    if archmage::X64V3Token::summon().is_some() {
        t.push(("v3", |x, o| {
            v3::vexpf_libm_into_v3_t(archmage::X64V3Token::summon().unwrap(), x, o)
        }));
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if archmage::X64V4xToken::summon().is_some() {
        t.push(("v4x", |x, o| {
            v4x::vexpf_libm_into_v4x_t(archmage::X64V4xToken::summon().unwrap(), x, o)
        }));
    }
    #[cfg(target_arch = "aarch64")]
    if archmage::NeonToken::summon().is_some() {
        t.push(("neon", |x, o| {
            neon::vexpf_libm_into_neon_t(archmage::NeonToken::summon().unwrap(), x, o)
        }));
    }
    #[cfg(target_arch = "wasm32")]
    if archmage::Wasm128Token::summon().is_some() {
        t.push(("wasm128", |x, o| {
            wasm128::vexpf_libm_into_wasm128_t(
                archmage::Wasm128Token::summon().unwrap(),
                x,
                o,
            )
        }));
    }
    t
}

/// Run `xs` through every tier and compare against scalar `libm::expf`.
fn check_expf(label: &str, xs: &[f32]) -> usize {
    let want: Vec<u32> = xs.iter().map(|&x| libm::expf(x).to_bits()).collect();
    let mut got = vec![0.0f32; xs.len()];
    for (name, f) in tiers_expf() {
        got.fill(f32::from_bits(0x7fc0_dead));
        f(xs, &mut got);
        let mut bad = 0usize;
        let mut first = String::new();
        for i in 0..xs.len() {
            if got[i].to_bits() != want[i] {
                if bad < 8 {
                    first += &alloc::format!(
                        "\n  x={:#010x} ({:e}) libm={:#010x} lane={:#010x}",
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

#[test]
fn expf_libm_edge_matrix_bit_exact() {
    let mut xs: Vec<f32> = Vec::new();
    // Every e_expf branch boundary, both signs, ±3 ulp each side.
    for bits in [
        0x0000_0000u32,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x3900_0000, // 2^-14: tiny-result boundary
        0x3eb1_7218, // 0.5 ln2
        0x3f85_1592, // 1.5 ln2
        0x42ae_ac50, // underflow boundary (|x| >= 87.33655)
        0x42b1_7218, // overflow boundary (x >= 88.722839)
        0x42cf_f1b5, // result-zero boundary (x <= -103.972084)
        0x7f80_0000, // inf
    ] {
        for d in -3i32..=3 {
            for &s in &[1u32, 0x8000_0000] {
                let b = (bits as i32).wrapping_add(d) as u32 | s;
                if (b & 0x7f80_0000) != 0 {
                    xs.push(f32::from_bits(b));
                }
            }
        }
    }
    // NaN payload/sign variants + canonical.
    xs.extend_from_slice(&[
        f32::NAN,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0x7fa0_0000),
        f32::from_bits(0xffc0_0001),
        f32::from_bits(0x7f80_0001),
    ]);
    check_expf("edges", &xs);
}

#[test]
fn expf_libm_dense_domain_bit_exact() {
    // Dense coverage of the live domain [-105, 90]: crosses the k
    // boundaries (±0.5ln2, ±1.5ln2), the subnormal-output band
    // [-103.97, -87.34], the result-zero boundary, and overflow.
    let xs: Vec<f32> = (0..600_000u32)
        .map(|i| -105.0f32 + (i as f32) * (195.0 / 600_000.0))
        .collect();
    check_expf("dense", &xs);
}

#[test]
fn expf_libm_random_sweep_bit_exact() {
    let mut rng = Rng(0x5eed_e1);
    let mut xs = vec![0.0f32; 4096];
    for _rep in 0..64 {
        for v in xs.iter_mut() {
            *v = match rng.next() % 6 {
                0 => rng.range(-110.0, 100.0),
                1 => rng.range(-104.5, -86.5), // subnormal-output band
                2 => rng.range(-1.5, 1.5),     // k-boundary straddle
                3 => rng.range(87.0, 89.5),    // overflow edge
                4 => f32::from_bits(rng.bits() & 0x007f_ffff), // subnormal x
                _ => f32::from_bits(rng.bits()), // anything incl NaN/inf
            };
        }
        check_expf("random", &xs[..2048]);
    }
}

#[test]
fn vexpf_libm_into_matches_scalar() {
    let mut rng = Rng(7);
    for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 33, 257, 1000] {
        let xs: Vec<f32> = (0..n).map(|_| rng.range(-110.0, 100.0)).collect();
        let mut out = vec![0.0; n];
        vexpf_libm_into(&xs, &mut out);
        for i in 0..n {
            assert_eq!(
                out[i].to_bits(),
                libm::expf(xs[i]).to_bits(),
                "n={n} i={i} x={:e}",
                xs[i]
            );
        }
    }
}

/// The `coverage` gates must be bit-identical to scalar on their promised
/// domain — including the scalar-fallback lanes that RANGE still services.
#[test]
fn powf_libm_gated_tiers_bit_exact_on_domain() {
    use fdlibm::coverage::*;
    let mut rng = Rng(0xC0FFEE);
    // Domain: x positive-normal OR subnormal (POSITIVE_FINITE covers both),
    // p moderate — x^p ranges over normal + underflow lanes.
    for gate_name in ["EXACT", "POSITIVE_FINITE"] {
        for &p in &[2.4f32, 0.4, -0.02, 6.0, -3.5] {
            let xs: Vec<f32> = (0..4096)
                .map(|i| match i % 4 {
                    0 => f32::from_bits(0x0000_0001 + (rng.bits() & 0x007f_fffe)), // subnormal
                    1 => rng.range(1e-37, 1.0),  // small normal → underflow result
                    2 => rng.range(1.0, 16.0),   // mid
                    _ => rng.range(1e-30, 1e30), // wide normal
                })
                .collect();
            let mut out = vec![0.0; xs.len()];
            match gate_name {
                "EXACT" => vpow_libm_into_gated::<{ EXACT }>(&xs, &mut out, p),
                _ => vpow_libm_into_gated::<{ POSITIVE_FINITE }>(&xs, &mut out, p),
            }
            for (i, (&x, &o)) in xs.iter().zip(&out).enumerate() {
                assert_eq!(
                    o.to_bits(),
                    libm::powf(x, p).to_bits(),
                    "{gate_name} p={p} i={i} x={x:e}"
                );
            }
        }
    }
    // POSITIVE_NORMAL / BULK: x normal, results normal.
    for &p in &[2.4f32, 0.02, 1.3] {
        let xs: Vec<f32> = (0..2048).map(|_| rng.range(0.25, 4.0)).collect();
        let mut a = vec![0.0; xs.len()];
        let mut b = vec![0.0; xs.len()];
        vpow_libm_into_gated::<{ POSITIVE_NORMAL }>(&xs, &mut a, p);
        vpow_libm_into_gated::<{ BULK }>(&xs, &mut b, p);
        for (i, &x) in xs.iter().enumerate() {
            let want = libm::powf(x, p).to_bits();
            assert_eq!(a[i].to_bits(), want, "POSITIVE_NORMAL p={p} i={i} x={x:e}");
            assert_eq!(b[i].to_bits(), want, "BULK p={p} i={i} x={x:e}");
        }
    }
}
