//! Tests: golden vectors from the reference implementation (Octave,
//! `psnrhvsm.m`), API contract checks, and per-tier parity.

use alloc::vec::Vec;

use crate::*;

/// Golden rows produced by running the reference `psnrhvsm.m` under
/// GNU Octave on the `gen` patterns below (same integer arithmetic, so
/// the inputs are bit-identical). Columns: `(name, w, h, step, hvs_m,
/// hvs)`. Regenerate with `validation/gen_goldens.m`.
struct Golden {
    name: &'static str,
    width: usize,
    height: usize,
    step: usize,
    hvs_m: f64,
    hvs: f64,
    make: fn() -> (Vec<f32>, Vec<f32>),
}

/// `gen` kind 1: `mod(37x + 91y + x·y, 256)`.
fn gen1(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(((37 * x + 91 * y + x * y) % 256) as f32);
        }
    }
    v
}

/// `gen` kind 2: `mod((17x XOR 29y) + 3x + 5y, 256)`.
fn gen2(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(((((17 * x) ^ (29 * y)) + 3 * x + 5 * y) % 256) as f32);
        }
    }
    v
}

/// `gen` kind 3: `mod(x + 2y, 256)` — DC-dominated gradient.
fn gen3(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(((x + 2 * y) % 256) as f32);
        }
    }
    v
}

/// `gen` kind 4: gen1 with the `20<=x<40, 10<=y<30` patch replaced by
/// gen2 — mixed smooth/texture content.
fn gen4(w: usize, h: usize) -> Vec<f32> {
    let mut v = gen1(w, h);
    let g2 = gen2(w, h);
    for y in 0..h {
        for x in 0..w {
            if (20..40).contains(&x) && (10..30).contains(&y) {
                v[y * w + x] = g2[y * w + x];
            }
        }
    }
    v
}

/// `gen` kind 5: constant 128.
fn gen5(w: usize, h: usize) -> Vec<f32> {
    alloc::vec![128.0f32; w * h]
}

/// `mod(img + c, 256)` — the Octave goldens' modular offset.
fn off(v: &[f32], c: u32) -> Vec<f32> {
    v.iter()
        .map(|&p| ((p as u32) + c) % 256)
        .map(|p| p as f32)
        .collect()
}

/// `min(img + c, 255)` — saturating offset.
fn clamp_add(v: &[f32], c: u32) -> Vec<f32> {
    v.iter()
        .map(|&p| ((p as u32) + c).min(255))
        .map(|p| p as f32)
        .collect()
}

fn goldens() -> Vec<Golden> {
    alloc::vec![
        Golden {
            name: "g1_vs_g1p7",
            width: 64,
            height: 64,
            step: 8,
            hvs_m: 17.900065595784,
            hvs: 15.676050069368,
            make: || (gen1(64, 64), off(&gen1(64, 64), 7)),
        },
        Golden {
            name: "g1_vs_g2",
            width: 64,
            height: 64,
            step: 8,
            hvs_m: 8.512479065136,
            hvs: 7.064879559932,
            make: || (gen1(64, 64), gen2(64, 64)),
        },
        Golden {
            name: "g2_vs_g2p3",
            width: 64,
            height: 64,
            step: 8,
            hvs_m: 26.264640103374,
            hvs: 22.152643396635,
            make: || (gen2(64, 64), off(&gen2(64, 64), 3)),
        },
        Golden {
            name: "g3_24x16",
            width: 24,
            height: 16,
            step: 8,
            hvs_m: 20.480864927892,
            hvs: 20.480864927892,
            make: || (gen3(24, 16), off(&gen3(24, 16), 15)),
        },
        Golden {
            name: "g4_100x52",
            width: 100,
            height: 52,
            step: 8,
            hvs_m: 20.346664089806,
            hvs: 17.908566392242,
            make: || (gen4(100, 52), off(&gen4(100, 52), 5)),
        },
        Golden {
            name: "g1_8x8",
            width: 8,
            height: 8,
            step: 8,
            hvs_m: 15.592246791711,
            hvs: 13.187506187539,
            make: || (gen1(8, 8), off(&gen1(8, 8), 30)),
        },
        Golden {
            name: "identical",
            width: 64,
            height: 64,
            step: 8,
            hvs_m: 100000.0,
            hvs: 100000.0,
            make: || (gen1(64, 64), gen1(64, 64)),
        },
        Golden {
            name: "g1_step4",
            width: 64,
            height: 64,
            step: 4,
            hvs_m: 17.850562673526,
            hvs: 15.634189958272,
            make: || (gen1(64, 64), off(&gen1(64, 64), 7)),
        },
        Golden {
            name: "const_patch",
            width: 64,
            height: 64,
            step: 8,
            hvs_m: 26.688943764399,
            hvs: 25.522332255989,
            make: || {
                let mut d = gen5(64, 64);
                let g = gen2(8, 8);
                for y in 0..8 {
                    for x in 0..8 {
                        d[y * 64 + x] = g[y * 8 + x];
                    }
                }
                (gen5(64, 64), d)
            },
        },
        Golden {
            name: "g2_dcshift",
            width: 64,
            height: 64,
            step: 8,
            hvs_m: 18.690858018536,
            hvs: 15.043492701114,
            make: || (gen2(64, 64), off(&gen2(64, 64), 10)),
        },
        Golden {
            name: "g1_clamp15",
            width: 64,
            height: 64,
            step: 8,
            hvs_m: 20.748175133551,
            hvs: 20.717671469471,
            make: || (gen1(64, 64), clamp_add(&gen1(64, 64), 15)),
        },
    ]
}

/// The f32 block pipeline vs the f64 reference goldens. The per-block
/// f32 terms hold the score within ~1e-3 dB; identical images land
/// exactly on 100000 (zero energy, no log).
#[test]
fn golden_scores() {
    for g in goldens() {
        let (r, d) = (g.make)();
        let s = psnrhvs_plane_f32_step(&r, &d, g.width, g.height, g.width, g.step).unwrap();
        let tol = if g.name == "identical" { 0.0 } else { 1e-3 };
        assert!(
            (s.psnr_hvs_m - g.hvs_m).abs() <= tol,
            "{}: psnr_hvs_m {} vs golden {}",
            g.name,
            s.psnr_hvs_m,
            g.hvs_m
        );
        assert!(
            (s.psnr_hvs - g.hvs).abs() <= tol,
            "{}: psnr_hvs {} vs golden {}",
            g.name,
            s.psnr_hvs,
            g.hvs
        );
    }
}

/// PSNR-HVS-M is never lower than PSNR-HVS on the same pair (masking
/// only ever shrinks the non-DC distortion terms).
#[test]
fn hvs_m_bounds_hvs() {
    let r = gen4(64, 64);
    for c in [1u32, 5, 20, 60] {
        let d = off(&r, c);
        let s = psnrhvs_plane_f32(&r, &d, 64, 64, 64).unwrap();
        assert!(s.psnr_hvs_m >= s.psnr_hvs, "c={c}: {s:?}");
    }
}

#[test]
fn errors() {
    let tiny = alloc::vec![0.0f32; 64];
    assert!(matches!(
        psnrhvs_plane_f32(&tiny, &tiny, 8, 7, 8),
        Err(Error::TooSmall { .. })
    ));
    let p = alloc::vec![0.0f32; 64 * 64];
    assert!(matches!(
        psnrhvs_plane_f32(&p[..100], &p, 64, 64, 64),
        Err(Error::BufferLength { .. })
    ));
    assert!(matches!(
        psnrhvs_plane_f32(&p, &p, 64, 64, 32),
        Err(Error::InvalidStride { .. })
    ));
    assert!(matches!(
        psnrhvs_plane_f32_step(&p, &p, 64, 64, 64, 0),
        Err(Error::InvalidStep)
    ));
    let rgb = alloc::vec![0u8; 3 * 8 * 8];
    assert!(matches!(
        psnrhvs_rgb8(&rgb, &rgb, 8, 7, 24),
        Err(Error::TooSmall { .. })
    ));
}

/// Identical images give exactly 100000 on every entry point.
#[test]
fn identical_is_100000() {
    let p = gen4(64, 64);
    let s = psnrhvs_plane_f32(&p, &p, 64, 64, 64).unwrap();
    assert_eq!(s.psnr_hvs, 100000.0);
    assert_eq!(s.psnr_hvs_m, 100000.0);
    let rgb: Vec<u8> = gen2(48, 16).iter().map(|&v| v as u8).collect();
    let s = psnrhvs_rgb8(&rgb, &rgb, 16, 16, 48).unwrap();
    assert_eq!(s.psnr_hvs, 100000.0);
    let s = psnrhvs_luma8(&rgb, &rgb, 16, 16, 48).unwrap();
    assert_eq!(s.psnr_hvs, 100000.0);
}

/// rgb8/luma8 against per-channel reference goldens (the reference is
/// single-channel; each channel plane was run through `psnrhvsm.m` under
/// Octave — `validation/gen_goldens_rgb.m` — and the mean is the
/// multi-plane convention codec harnesses report).
#[test]
fn rgb8_luma8_goldens() {
    let w = 32usize;
    let h = 24usize;
    let g = gen2(w, h);
    let mut rgb_r = alloc::vec![0u8; 3 * w * h];
    let mut rgb_d = alloc::vec![0u8; 3 * w * h];
    for i in 0..w * h {
        rgb_r[3 * i] = g[i] as u8;
        rgb_r[3 * i + 1] = (g[i] as u8).wrapping_add(40);
        rgb_r[3 * i + 2] = (g[i] as u8) / 2;
        rgb_d[3 * i] = (g[i] as u8).wrapping_add(9);
        rgb_d[3 * i + 1] = (g[i] as u8).wrapping_add(31);
        rgb_d[3 * i + 2] = ((g[i] as u8) / 2).wrapping_add(7);
    }
    let s = psnrhvs_rgb8(&rgb_r, &rgb_d, w, h, 3 * w).unwrap();
    // Octave per-channel goldens (hvs_m, hvs).
    let tol = 1e-3;
    assert!(
        (s.red.psnr_hvs_m - 21.173839444519).abs() <= tol,
        "R m: {}",
        s.red.psnr_hvs_m
    );
    assert!(
        (s.red.psnr_hvs - 17.070955425789).abs() <= tol,
        "R h: {}",
        s.red.psnr_hvs
    );
    assert!(
        (s.green.psnr_hvs_m - 17.008519407711).abs() <= tol,
        "G m: {}",
        s.green.psnr_hvs_m
    );
    assert!(
        (s.green.psnr_hvs - 13.779868540738).abs() <= tol,
        "G h: {}",
        s.green.psnr_hvs
    );
    assert!(
        (s.blue.psnr_hvs_m - 27.100729308721).abs() <= tol,
        "B m: {}",
        s.blue.psnr_hvs_m
    );
    assert!(
        (s.blue.psnr_hvs - 27.100729308721).abs() <= tol,
        "B h: {}",
        s.blue.psnr_hvs
    );
    let mean_m = (s.red.psnr_hvs_m + s.green.psnr_hvs_m + s.blue.psnr_hvs_m) / 3.0;
    let mean_h = (s.red.psnr_hvs + s.green.psnr_hvs + s.blue.psnr_hvs) / 3.0;
    assert!((s.psnr_hvs_m - mean_m).abs() < 1e-12);
    assert!((s.psnr_hvs - mean_h).abs() < 1e-12);
    // BT.601 luma of the same pair through `psnrhvsm.m`:
    //   lum = round(0.299R + 0.587G + 0.114B) per channel-pair
    //   (computed in Octave with double-cast channels — `gen` returns
    //   uint64, and `0.299*uint64` would silently do integer math).
    let s = psnrhvs_luma8(&rgb_r, &rgb_d, w, h, 3 * w).unwrap();
    assert!(
        (s.psnr_hvs_m - 22.564507318756).abs() <= tol,
        "Y m: {}",
        s.psnr_hvs_m
    );
    assert!(
        (s.psnr_hvs - 18.116854289998).abs() <= tol,
        "Y h: {}",
        s.psnr_hvs
    );
}

/// Every compiled-in SIMD tier must produce bit-identical band sums
/// (fixed-order per-block f32 sums + f64 raster fold).
#[test]
fn tier_parity() {
    let (r, d) = (gen4(48, 40), off(&gen4(48, 40), 11));
    let scalar = kernel::psnrhvs_band_scalar_t(archmage::ScalarToken, &r, &d, 48, 0, 5, 6, 8, None);
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            let got = kernel::psnrhvs_band_v3_t(t, &r, &d, 48, 0, 5, 6, 8, None);
            assert_eq!(got, scalar, "v3 vs scalar");
        }
        #[cfg(feature = "avx512")]
        if let Some(t) = archmage::X64V4xToken::summon() {
            let got = kernel::psnrhvs_band_v4x_t(t, &r, &d, 48, 0, 5, 6, 8, None);
            assert_eq!(got, scalar, "v4x vs scalar");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::NeonToken::summon() {
            let got = kernel::psnrhvs_band_neon_t(t, &r, &d, 48, 0, 5, 6, 8, None);
            assert_eq!(got, scalar, "neon vs scalar");
        }
    }
}

/// Interleave three planes into packed RGB8 (values are 0..=255 integers).
fn rgb_of(r: &[f32], g: &[f32], b: &[f32]) -> Vec<u8> {
    r.iter()
        .zip(g)
        .zip(b)
        .flat_map(|((&r, &g), &b)| [r as u8, g as u8, b as u8])
        .collect()
}

/// The score's own fold over a channel's block map: the same band split
/// as `plane_step_impl`, per-band raster f64 sums of `64·v` (exact: `v`
/// is `S/64`), then `finalize`.
fn fold_exact(masked: &[f32], unmasked: &[f32], nb_x: usize, nb_y: usize) -> PlaneScore {
    let nb = kernel::n_bands(nb_y);
    let sz = nb_y.div_ceil(nb);
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for b in 0..nb {
        let (lo, hi) = (b * sz, ((b + 1) * sz).min(nb_y));
        let (mut a, mut c) = (0.0f64, 0.0f64);
        for i in lo * nb_x..hi.max(lo) * nb_x {
            a += masked[i] as f64 * 64.0;
            c += unmasked[i] as f64 * 64.0;
        }
        s1 += a;
        s2 += c;
    }
    finalize(s1, s2, nb_x * nb_y)
}

/// Each channel's block maps fold back to that channel's scores bit for
/// bit, and requesting maps changes no score.
#[test]
fn block_maps_fold_to_channel_scores_exactly() {
    for (w, h) in [(64usize, 48usize), (37, 29), (8, 8), (130, 75)] {
        let (r, g, b) = (gen1(w, h), gen2(w, h), gen4(w, h));
        let a = rgb_of(&r, &g, &b);
        let d = rgb_of(&off(&r, 9), &clamp_add(&g, 30), &off(&gen3(w, h), 1));
        let plain = psnrhvs_rgb8(&a, &d, w, h, w * 3).unwrap();
        let (score, maps) = psnrhvs_rgb8_impl(&a, &d, w, h, w * 3, true).unwrap();
        assert_eq!(score, plain, "{w}x{h}");
        let (nb_x, nb_y) = ((w - 8) / 8 + 1, (h - 8) / 8 + 1);
        for (m, want) in maps
            .unwrap()
            .iter()
            .zip([plain.red, plain.green, plain.blue])
        {
            assert_eq!(
                (m.masked.len(), m.unmasked.len()),
                (nb_x * nb_y, nb_x * nb_y)
            );
            let got = fold_exact(&m.masked, &m.unmasked, nb_x, nb_y);
            assert_eq!(got.psnr_hvs.to_bits(), want.psnr_hvs.to_bits(), "{w}x{h}");
            assert_eq!(
                got.psnr_hvs_m.to_bits(),
                want.psnr_hvs_m.to_bits(),
                "{w}x{h}"
            );
        }
    }
}

/// An independent f64 transcription of one block of `psnrhvsm.m` (DCT by
/// definition, `maskeff`, the two energies) agrees with the maps' blocks
/// to the f32 pipeline's precision.
#[test]
fn block_maps_match_an_f64_block_oracle() {
    fn dct(x: &[f64; 64]) -> [f64; 64] {
        let c = |k: usize| {
            if k == 0 {
                (1.0f64 / 8.0).sqrt()
            } else {
                (2.0f64 / 8.0).sqrt()
            }
        };
        let mut out = [0.0; 64];
        for k in 0..8 {
            for l in 0..8 {
                let mut acc = 0.0;
                for m in 0..8 {
                    for n in 0..8 {
                        acc += x[m * 8 + n]
                            * (core::f64::consts::PI * (2 * m + 1) as f64 * k as f64 / 16.0).cos()
                            * (core::f64::consts::PI * (2 * n + 1) as f64 * l as f64 / 16.0).cos();
                    }
                }
                out[k * 8 + l] = c(k) * c(l) * acc;
            }
        }
        out
    }
    fn var(v: &[f64]) -> f64 {
        let mu = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / (v.len() - 1) as f64
    }
    fn maskeff(x: &[f64; 64], d: &[f64; 64]) -> f64 {
        let e: f64 = (1..64)
            .map(|c| d[c] * d[c] * kernel::MASK_COF[c] as f64)
            .sum();
        let vari = var(x) * 64.0;
        let q = |r0: usize, c0: usize| {
            let v: Vec<f64> = (0..16).map(|i| x[(r0 + i / 4) * 8 + c0 + i % 4]).collect();
            var(&v) * 16.0
        };
        let pop = if vari == 0.0 {
            0.0
        } else {
            (q(0, 0) + q(0, 4) + q(4, 4) + q(4, 0)) / vari
        };
        (e * pop).sqrt() / 32.0
    }
    let (w, h) = (48, 40);
    let (r, d) = (gen4(w, h), clamp_add(&gen1(w, h), 17));
    let (_, maps) = plane_step_impl(&r, &d, w, h, w, 8, true).unwrap();
    let maps = maps.unwrap();
    let mut worst = 0.0f64;
    for by in 0..h / 8 {
        for bx in 0..w / 8 {
            let blk = |p: &[f32]| {
                core::array::from_fn(|i| p[(by * 8 + i / 8) * w + bx * 8 + i % 8] as f64)
            };
            let (a, b): ([f64; 64], [f64; 64]) = (blk(&r), blk(&d));
            let (da, db) = (dct(&a), dct(&b));
            let mask = maskeff(&a, &da).max(maskeff(&b, &db));
            let (mut s1, mut s2) = (0.0f64, 0.0f64);
            for c in 0..64 {
                let u = (da[c] - db[c]).abs();
                let t = u * kernel::CSF[c] as f64;
                s2 += t * t;
                let um = if c == 0 {
                    u
                } else {
                    (u - mask / kernel::MASK_COF[c] as f64).max(0.0)
                };
                let t1 = um * kernel::CSF[c] as f64;
                s1 += t1 * t1;
            }
            let i = by * (w / 8) + bx;
            for (got, want) in [(maps.masked[i], s1 / 64.0), (maps.unmasked[i], s2 / 64.0)] {
                let rel = (got as f64 - want).abs() / want.abs().max(1e-3);
                worst = worst.max(rel);
            }
        }
    }
    #[cfg(feature = "std")]
    std::eprintln!("worst per-block relative difference vs the f64 oracle: {worst:e}");
    assert!(worst < 1e-4, "{worst}");
}

/// Every compiled-in tier writes bit-identical block maps.
#[test]
fn tier_parity_of_block_maps() {
    let (r, d) = (gen4(48, 40), off(&gen4(48, 40), 11));
    let run = |f: &dyn Fn(Option<(&mut [f32], &mut [f32])>) -> (f64, f64)| {
        let (mut m, mut u) = (alloc::vec![0.0f32; 30], alloc::vec![0.0f32; 30]);
        let sums = f(Some((&mut m, &mut u)));
        (sums, m, u)
    };
    let scalar =
        run(&|b| kernel::psnrhvs_band_scalar_t(archmage::ScalarToken, &r, &d, 48, 0, 5, 6, 8, b));
    assert_eq!(
        scalar.0,
        kernel::psnrhvs_band_scalar_t(archmage::ScalarToken, &r, &d, 48, 0, 5, 6, 8, None)
    );
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            assert_eq!(
                run(&|b| kernel::psnrhvs_band_v3_t(t, &r, &d, 48, 0, 5, 6, 8, b)),
                scalar
            );
        }
        #[cfg(feature = "avx512")]
        if let Some(t) = archmage::X64V4xToken::summon() {
            assert_eq!(
                run(&|b| kernel::psnrhvs_band_v4x_t(t, &r, &d, 48, 0, 5, 6, 8, b)),
                scalar
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::NeonToken::summon() {
            assert_eq!(
                run(&|b| kernel::psnrhvs_band_neon_t(t, &r, &d, 48, 0, 5, 6, 8, b)),
                scalar
            );
        }
    }
}

/// A change confined to one block moves only that block's cells; a change
/// in the unscored right/bottom strip moves none.
#[test]
fn block_maps_are_block_local() {
    let (w, h) = (45, 37); // 5×4 blocks plus 5- and 5-pixel strips
    let base = gen2(w, h);
    let mut inside = base.clone();
    for y in 9..14 {
        for x in 17..23 {
            inside[y * w + x] = 255.0 - inside[y * w + x];
        }
    }
    let (_, m) = plane_step_impl(&base, &inside, w, h, w, 8, true).unwrap();
    let m = m.unwrap();
    for (i, (&a, &b)) in m.masked.iter().zip(&m.unmasked).enumerate() {
        if i == 5 + 2 {
            assert!(b > 0.0, "the changed block must respond");
        } else {
            assert_eq!((a, b), (0.0, 0.0), "block {i}");
        }
    }
    let mut strip = base.clone();
    for y in 0..h {
        for x in 40..w {
            strip[y * w + x] = 0.0;
        }
    }
    let (s, m) = plane_step_impl(&base, &strip, w, h, w, 8, true).unwrap();
    let m = m.unwrap();
    assert!(m.masked.iter().chain(&m.unmasked).all(|&v| v == 0.0));
    assert_eq!(s.psnr_hvs, 100000.0);
}
