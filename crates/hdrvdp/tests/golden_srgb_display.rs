//! Golden validation of the `sRGB-display` colour encoding against **official HDR-VDP-2.2.2**
//! (MATLAB release under Octave), the route 8-bit sRGB images take.
//!
//! Upstream documents `'sRGB-display'` as the encoding "for standard (LDR) color images": an sRGB
//! display with a 100 cd/m² peak and 1 cd/m² black. `validation/gen_srgb_goldens.m` builds every
//! image with integer arithmetic only, so this test regenerates the same 8-bit codes bit for bit
//! and only the expected `res.Q` values are committed (`validation/goldens_srgb/goldens_srgb.tsv`).

#![forbid(unsafe_code)]

use hdrvdp::{ColorEncoding, Params, score};
use std::path::Path;

/// Same bound as the luminance goldens (`golden_2_2_2.rs`, measured worst case 7.8e-4 there).
const TOL_Q: f64 = 5e-3;

/// Interleaved row-major RGB codes, `img[(y·w + x)·3 + c]`, matching `reference_codes` in the generator.
fn reference_codes(w: usize, h: usize) -> Vec<u32> {
    let mut img = vec![0u32; w * h * 3];
    for y in 0..h as u64 {
        for x in 0..w as u64 {
            for c in 0..3u64 {
                let v = 31 * x + 17 * y + 59 * c + x * y / 13 + (x + 2 * c).pow(2) / 29;
                img[((y as usize) * w + x as usize) * 3 + c as usize] = (v % 256) as u32;
            }
        }
    }
    img
}

fn distort(name: &str, r: &[u32], w: usize, h: usize) -> Vec<u32> {
    match name {
        "identical" => r.to_vec(),
        "noise" => {
            // MINSTD, advanced in the generator's y, x, c loop order (= interleaved order).
            let mut state: u64 = 12345;
            r.iter()
                .map(|&v| {
                    state = state * 48271 % 2_147_483_647;
                    (i64::from(v) + (state % 21) as i64 - 10).clamp(0, 255) as u32
                })
                .collect()
        }
        "blur" => {
            let mut t = vec![0u32; r.len()];
            for y in 0..h as isize {
                for x in 0..w as isize {
                    for c in 0..3 {
                        let mut s = 0u32;
                        for dy in -1..=1isize {
                            for dx in -1..=1isize {
                                let yy = (y + dy).clamp(0, h as isize - 1) as usize;
                                let xx = (x + dx).clamp(0, w as isize - 1) as usize;
                                s += r[(yy * w + xx) * 3 + c];
                            }
                        }
                        t[(y as usize * w + x as usize) * 3 + c] = s / 9;
                    }
                }
            }
            t
        }
        "dark" => r.iter().map(|&v| v * 55 / 100).collect(),
        "bright" => r.iter().map(|&v| (v * 130 / 100 + 8).min(255)).collect(),
        "chroma" => r
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 3 == 0 { (v + 20).min(255) } else { v })
            .collect(),
        other => panic!("unknown case {other}"),
    }
}

fn unit(codes: &[u32]) -> Vec<f64> {
    codes.iter().map(|&v| f64::from(v) / 255.0).collect()
}

#[test]
fn srgb_display_matches_official_hdrvdp_2_2_2() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("validation/goldens_srgb/goldens_srgb.tsv");
    let tsv = std::fs::read_to_string(&path).expect("goldens_srgb.tsv");
    let mut n = 0;
    let mut worst = (0.0f64, 0.0f64);
    let mut failures = Vec::new();
    for line in tsv.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 6, "bad row: {line}");
        let (case, w, h): (&str, usize, usize) =
            (f[0], f[2].parse().unwrap(), f[3].parse().unwrap());
        let (ppd, want): (f64, f64) = (f[4].parse().unwrap(), f[5].parse().unwrap());
        let r = reference_codes(w, h);
        let t = distort(case, &r, w, h);
        let (rf, tf) = (unit(&r), unit(&t));
        let base = Params {
            pix_per_deg: ppd,
            ..Default::default()
        };
        let strict = score(
            &tf,
            &rf,
            w,
            h,
            ColorEncoding::SrgbDisplay,
            &base.clone().with_strict(),
        )
        .unwrap();
        let fast = score(&tf, &rf, w, h, ColorEncoding::SrgbDisplay, &base).unwrap();
        eprintln!(
            "{case:9} {w:3}x{h:<3} ppd {ppd:2}: official {want:.6} strict {strict:.6} (Δ {:+.2e}) default {fast:.6} (Δ {:+.2e})",
            strict - want,
            fast - want
        );
        if (strict - want).abs() > TOL_Q || (fast - want).abs() > TOL_Q {
            failures.push(format!(
                "{case} {w}x{h} ppd {ppd}: strict {strict} default {fast} vs official {want}"
            ));
        }
        worst = (
            worst.0.max((strict - want).abs()),
            worst.1.max((fast - want).abs()),
        );
        n += 1;
    }
    assert_eq!(n, 24, "expected 24 golden cases");
    eprintln!(
        "sRGB-display: worst |ΔQ| strict {:.3e}, default {:.3e} over {n} cases",
        worst.0, worst.1
    );
    assert!(
        failures.is_empty(),
        "{} of {n} cases above |ΔQ| {TOL_Q}:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
