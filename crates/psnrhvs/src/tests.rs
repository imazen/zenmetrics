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
    let scalar = kernel::psnrhvs_band_scalar(archmage::ScalarToken, &r, &d, 48, 0, 5, 6, 8);
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            let got = kernel::psnrhvs_band_v3(t, &r, &d, 48, 0, 5, 6, 8);
            assert_eq!(got, scalar, "v3 vs scalar");
        }
        #[cfg(feature = "avx512")]
        if let Some(t) = archmage::X64V4Token::summon() {
            let got = kernel::psnrhvs_band_v4(t, &r, &d, 48, 0, 5, 6, 8);
            assert_eq!(got, scalar, "v4 vs scalar");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::NeonToken::summon() {
            let got = kernel::psnrhvs_band_neon(t, &r, &d, 48, 0, 5, 6, 8);
            assert_eq!(got, scalar, "neon vs scalar");
        }
    }
}
