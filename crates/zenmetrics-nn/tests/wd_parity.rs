//! External Cool-Chic WD goldens, five KonJND-normalized columns. Caller
//! explicitly supplies models, ladder and engine via `nn-wd-parity`.
use std::path::{Path, PathBuf};
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric};
use zenmetrics_nn::{Engine, WassersteinDistortion};
fn env(name: &str) -> PathBuf {
    std::env::var_os(name).expect(name).into()
}
fn png(ladder: &Path, name: &str) -> PathBuf {
    let p = ladder.join(name);
    if p.exists() {
        p
    } else {
        ladder.with_file_name("ladder-nonsquare").join(name)
    }
}
fn load(path: &Path) -> (Vec<u8>, usize, usize) {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let decoded = zenpng::decode(
        &raw,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .unwrap();
    let s = decoded.pixels.as_slice();
    assert_eq!(s.descriptor().bytes_per_pixel(), 3);
    let (w, h) = (s.width() as usize, s.rows() as usize);
    let bytes = s.as_strided_bytes();
    let mut out = Vec::new();
    for y in 0..h {
        out.extend_from_slice(&bytes[y * s.stride()..y * s.stride() + 3 * w]);
    }
    (out, w, h)
}
fn image(px: &[u8], w: usize, h: usize, stride: usize) -> Image<'_> {
    Image::from(
        PixelSlice::new(px, w as u32, h as u32, stride, PixelDescriptor::RGB8_SRGB).unwrap(),
    )
}
fn jnd(column: &str) -> f64 {
    include_str!("../models/jnd_scale.tsv")
        .lines()
        .find_map(|line| {
            let f: Vec<_> = line.split('\t').collect();
            (f.first() == Some(&column)).then(|| f[1].parse().unwrap())
        })
        .unwrap_or_else(|| panic!("missing KonJND measurement for {column}"))
}
#[test]
fn coolchic_five_column_ladder_and_warm_stride() {
    let models = env("ZENMETRICS_NN_MODEL_DIR");
    let ladder = env("ZENMETRICS_NN_LADDER");
    let engine = match std::env::var("ZENMETRICS_NN_ENGINE").as_deref() {
        Ok("rten") => Engine::Rten,
        #[cfg(feature = "ort")]
        Ok("ort-cuda") => Engine::OrtCuda,
        other => panic!("unbuilt engine {other:?}"),
    };
    let mut wd = WassersteinDistortion::from_dir(&models, engine).unwrap();
    let goldens = std::fs::read_to_string(models.join("wd_goldens.tsv")).unwrap();
    let mut lines = goldens.lines();
    assert_eq!(
        lines.next(),
        Some("image\tsize\twd_s1\twd_s2\twd_s3\twd_s4\twd_s5")
    );
    let mut worst = [0.0f64; 5];
    let mut count = 0;
    for line in lines {
        let f: Vec<_> = line.split('\t').collect();
        assert_eq!(f.len(), 7);
        let (r, w, h) = load(&png(&ladder, &format!("{}_{}_ref.png", f[0], f[1])));
        let (d, dw, dh) = load(&png(&ladder, &format!("{}_{}_dist.png", f[0], f[1])));
        assert_eq!((w, h), (dw, dh));
        let want: Vec<f64> = f[2..].iter().map(|x| x.parse().unwrap()).collect();
        let evaluate = |wd: &mut WassersteinDistortion| -> Vec<f64> {
            wd.evaluate(
                image(&r, w, h, w * 3),
                image(&d, w, h, w * 3),
                EvalOpts::new(),
            )
            .unwrap()
            .scores()
            .map(|(_, v)| v)
            .collect()
        };
        let mut actual = evaluate(&mut wd);
        let mismatches = |values: &[f64]| {
            values.iter().zip(&want).enumerate().any(|(k, (a, b))| {
                !a.is_finite() || (a - b).abs() > 1e-3 * jnd(&format!("wd_s{}", k + 1))
            })
        };
        if mismatches(&actual) {
            let retry = evaluate(&mut wd);
            eprintln!(
                "WD mismatch {} {} initial={actual:?} retry={retry:?}",
                f[0], f[1]
            );
            actual = retry;
        }
        for k in 0..5 {
            let error = (actual[k] - want[k]).abs();
            worst[k] = worst[k].max(error);
            eprintln!(
                "wd_s{} {} {} actual={:.17e} want={:.17e} delta={:.9e}",
                k + 1,
                f[0],
                f[1],
                actual[k],
                want[k],
                error
            );
            assert!(actual[k].is_finite() && error <= 1e-3 * jnd(&format!("wd_s{}", k + 1)));
        }
        if f[0] == "1415" && f[1] == "64" {
            let stride = w * 3 + 15;
            let mut padded = vec![203u8; stride * h];
            for y in 0..h {
                padded[y * stride..y * stride + w * 3]
                    .copy_from_slice(&d[y * w * 3..(y + 1) * w * 3]);
            }
            let mut warm = wd
                .warm_reference(image(&r, w, h, w * 3), EvalOpts::new())
                .unwrap();
            assert!(warm.reuses_reference_work());
            let value = warm
                .evaluate(image(&padded, w, h, stride), EvalOpts::new())
                .unwrap();
            let scores: Vec<_> = value.scores().map(|(_, v)| v.to_bits()).collect();
            assert_eq!(
                scores,
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let same = warm
                .evaluate(image(&r, w, h, w * 3), EvalOpts::new())
                .unwrap();
            assert!(same.scores().all(|(_, v)| v == 0.0));
        }
        count += 1;
    }
    assert!(count > 0);
    for (k, error) in worst.into_iter().enumerate() {
        eprintln!(
            "WD_s{}: {count} pairs worst {error:.9e} = {:.9e} JND",
            k + 1,
            error / jnd(&format!("wd_s{}", k + 1))
        );
    }
}
