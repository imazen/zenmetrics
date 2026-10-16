//! Cool-Chic Wasserstein Distortion on raw sRGB VGG16 pre-pool features.
//!
//! Port of Orange's BSD-3-Clause implementation (see `LICENSE-wd`), based
//! on Qiu et al., CISS 2024. This convention is distinct from the AIC site's
//! unpublished WD preprocessing. Statistics and the sigma pyramid run here;
//! the exported VGG16 graph supplies ReLU outputs 3, 8, 15 and 22.

use std::borrow::Cow;
use std::path::Path;

use crate::common::read_meta;
use crate::engine::{self, Tap, Trunk};
use crate::preprocess::rgb8_to_planar_01;
use crate::{Engine, Error, za};

const COLUMNS: [&str; 5] = ["wd_s1", "wd_s2", "wd_s3", "wd_s4", "wd_s5"];
const CHANNELS: [usize; 4] = [64, 128, 256, 512];
const MAX_SIDE: usize = 2048;
const VAR_MIN: f32 = 5e-7;

/// Wasserstein Distortion at log2(sigma) 1 through 5, in Cool-Chic's
/// convention. Lower is closer; identical inputs score zero in all columns.
/// Input is strided sRGB8, with each dimension in 8..=2048.
///
/// VGG16 IMAGENET1K_V1 weights are permissive and are not bundled.
pub struct WassersteinDistortion {
    trunk: Box<dyn Trunk>,
}

// Zero padding, including on the sigma map, is part of this convention.
fn lowpass(x: &[f32], h: usize, w: usize, stride: usize) -> Vec<f32> {
    let (oh, ow) = (h.div_ceil(stride), w.div_ceil(stride));
    let mut out = vec![0.0; oh * ow];
    let kernel = [0.25f32, 0.5, 0.25];
    for y in 0..oh {
        for xout in 0..ow {
            let mut sum = 0.0;
            for (ky, &wy) in kernel.iter().enumerate() {
                let sy = (y * stride + ky).wrapping_sub(1);
                if sy >= h {
                    continue;
                }
                for (kx, &wx) in kernel.iter().enumerate() {
                    let sx = (xout * stride + kx).wrapping_sub(1);
                    if sx < w {
                        sum += x[sy * w + sx] * (wy * wx);
                    }
                }
            }
            out[y * ow + xout] = sum;
        }
    }
    out
}

fn decimate(x: &[f32], h: usize, w: usize) -> Vec<f32> {
    (0..h)
        .step_by(2)
        .flat_map(|y| (0..w).step_by(2).map(move |xpos| x[y * w + xpos]))
        .collect()
}

/// Five sigma weights at each of the five statistic resolutions. A constant
/// sigma becomes nonconstant at the border after zero-padded downsampling.
fn weights(h: usize, w: usize) -> Vec<[Vec<f32>; 5]> {
    let mut sigma: [Vec<f32>; 5] = core::array::from_fn(|k| vec![(k + 1) as f32; h * w]);
    let (mut h, mut w) = (h, w);
    let mut result = Vec::with_capacity(5);
    for level in 1..=5 {
        result.push(core::array::from_fn(|k| {
            sigma[k]
                .iter()
                .map(|&s| (1.0 - (s - level as f32).abs()).max(0.0))
                .collect()
        }));
        sigma = sigma.map(|s| lowpass(&s, h, w, 2));
        h = h.div_ceil(2);
        w = w.div_ceil(2);
    }
    result
}

fn channel(a: &[f32], b: &[f32], h: usize, w: usize, weights: &[[Vec<f32>; 5]]) -> [[f64; 5]; 5] {
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    let mut aa: Vec<_> = a.iter().map(|x| x * x).collect();
    let mut bb: Vec<_> = b.iter().map(|x| x * x).collect();
    let (mut h, mut w) = (h, w);
    let mut result = [[0.0; 5]; 5];
    for (level, weights) in weights.iter().enumerate() {
        let ma = lowpass(&a, h, w, 1);
        let mb = lowpass(&b, h, w, 1);
        let pa = lowpass(&aa, h, w, 1);
        let pb = lowpass(&bb, h, w, 1);
        for i in 0..h * w {
            let sa = (pa[i] - ma[i] * ma[i]).max(VAR_MIN).sqrt();
            let sb = (pb[i] - mb[i] * mb[i]).max(VAR_MIN).sqrt();
            let dm = ma[i] - mb[i];
            let ds = sa - sb;
            let distance = dm * dm + ds * ds;
            for (k, weight) in weights.iter().enumerate() {
                result[level][k] += f64::from(weight[i] * distance);
            }
        }
        for value in &mut result[level] {
            *value /= (h * w) as f64;
        }
        a = decimate(&ma, h, w);
        b = decimate(&mb, h, w);
        aa = decimate(&pa, h, w);
        bb = decimate(&pb, h, w);
        h = h.div_ceil(2);
        w = w.div_ceil(2);
    }
    result
}

fn head(a: &[Tap], b: &[Tap]) -> Result<[f64; 5], Error> {
    if a.len() != CHANNELS.len() || b.len() != a.len() {
        return Err(Error::BadModel("WD requires four VGG taps".into()));
    }
    let mut total = [0.0f32; 5];
    for ((a, b), c) in a.iter().zip(b).zip(CHANNELS) {
        if (a.c, a.h, a.w) != (b.c, b.h, b.w) {
            return Err(Error::SizeMismatch);
        }
        if a.c != c
            || a.h == 0
            || a.w == 0
            || a.data.len() != c * a.h * a.w
            || b.data.len() != a.data.len()
        {
            return Err(Error::BadModel("WD feature shape mismatch".into()));
        }
        let weights = weights(a.h, a.w);
        let plane = a.plane();
        // Bound scratch concurrency while respecting the configured thread cap.
        let channels = crate::par::par_map(c, crate::par::threads_for(c * plane).min(4), |ch| {
            channel(
                &a.data[ch * plane..(ch + 1) * plane],
                &b.data[ch * plane..(ch + 1) * plane],
                a.h,
                a.w,
                &weights,
            )
        });
        let mut layer = [0.0f32; 5];
        for level in 0..5 {
            for (k, value) in layer.iter_mut().enumerate() {
                *value += (channels.iter().map(|ch| ch[level][k]).sum::<f64>() / c as f64) as f32;
            }
        }
        for (total, layer) in total.iter_mut().zip(layer) {
            *total += layer;
        }
    }
    Ok(total.map(f64::from))
}

fn evaluation(values: [f64; 5]) -> Box<dyn za::Evaluation> {
    Box::new(za::ScalarEval {
        named: COLUMNS
            .into_iter()
            .zip(values)
            .map(|(k, v)| (Cow::Borrowed(k), v))
            .collect(),
    })
}

impl WassersteinDistortion {
    /// Load `wd_trunk.onnx` and `wd.meta` produced by the WD export recipe.
    /// Both engines run the same trunk and the Rust statistics head.
    ///
    /// # Errors
    /// Returns an error for missing/malformed files or unavailable engines.
    pub fn from_dir(dir: impl AsRef<Path>, engine: Engine) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let (channels, min) = read_meta(&dir.join("wd.meta"))?;
        if channels != CHANNELS || min != 8 {
            return Err(Error::BadModel(
                "WD metadata must declare 64,128,256,512 channels and min_size 8".into(),
            ));
        }
        let path = dir.join("wd_trunk.onnx");
        let trunk = match engine {
            Engine::Rten => engine::load(&path, 4)?,
            #[cfg(feature = "ort")]
            Engine::OrtCuda => {
                Box::new(engine::ort_engine::OrtCudaTrunk::load(&path)?) as Box<dyn Trunk>
            }
        };
        Ok(Self { trunk })
    }

    fn check_size(img: &za::Rgb8In<'_>) -> Result<(), Error> {
        if img.width < 8 || img.height < 8 {
            return Err(Error::TooSmall { min: 8 });
        }
        if img.width > MAX_SIDE || img.height > MAX_SIDE {
            return Err(Error::TooLarge { limit: MAX_SIDE });
        }
        Ok(())
    }

    fn features(&self, img: &za::Rgb8In<'_>) -> Result<Vec<Tap>, Error> {
        Self::check_size(img)?;
        self.trunk
            .run(rgb8_to_planar_01(img), img.height, img.width)
    }
}

impl za::Metric for WassersteinDistortion {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("wd", za::implementation!(1)).with_reference(format!(
            "Cool-Chic WD, raw RGB VGG16 pre-pool, sigma 2^1..2^5; engine {}",
            self.trunk.engine()
        ))
    }
    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        Cow::Owned(
            COLUMNS
                .map(|c| za::ScoreSpec::new(c, za::ScoreKind::Difference))
                .to_vec(),
        )
    }
    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }
    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map("wd")?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch).metric_err_with("wd", Error::kind);
        }
        let a = self.features(&r).metric_err_with("wd", Error::kind)?;
        opts.checkpoint()?;
        let b = self.features(&d).metric_err_with("wd", Error::kind)?;
        opts.checkpoint()?;
        Ok(evaluation(head(&a, &b).metric_err_with("wd", Error::kind)?))
    }
    fn warm_reference(
        &mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + '_>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map("wd")?;
        let view = za::rgb8_view(&r)?;
        let reference = self.features(&view).metric_err_with("wd", Error::kind)?;
        Ok(Box::new(WarmWd {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}
struct WarmWd<'a> {
    metric: &'a WassersteinDistortion,
    reference: Vec<Tap>,
    size: (usize, usize),
}
impl za::WarmReference for WarmWd<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map("wd")?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != self.size {
            return Err(Error::SizeMismatch).metric_err_with("wd", Error::kind);
        }
        let b = self
            .metric
            .features(&d)
            .metric_err_with("wd", Error::kind)?;
        Ok(evaluation(
            head(&self.reference, &b).metric_err_with("wd", Error::kind)?,
        ))
    }
    fn reuses_reference_work(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padding_and_odd_decimation() {
        assert_eq!(lowpass(&[1.0], 1, 1, 1), vec![0.25]);
        assert_eq!(lowpass(&[1.0; 9], 3, 3, 2), vec![0.5625; 4]);
        assert_eq!(
            decimate(&[0., 1., 2., 3., 4., 5., 6., 7., 8.], 3, 3),
            vec![0., 2., 6., 8.]
        );
    }
    #[test]
    fn identical_features_are_zero() {
        let x: Vec<_> = (0..35).map(|i| (i * 13 % 17) as f32 / 7.).collect();
        assert_eq!(channel(&x, &x, 5, 7, &weights(5, 7)), [[0.; 5]; 5]);
    }
    /// Pinned from Cool-Chic wasserstein_distortion on two 5x7 channels.
    #[test]
    fn statistics_match_independent_upstream_golden() {
        let a: Vec<f32> = (0..70).map(|i| (i % 17) as f32 / 7.0).collect();
        let b: Vec<f32> = a.iter().map(|&x| x * 0.91 + 0.13).collect();
        let weights = weights(5, 7);
        let channels: Vec<_> = (0..2)
            .map(|c| {
                channel(
                    &a[c * 35..(c + 1) * 35],
                    &b[c * 35..(c + 1) * 35],
                    5,
                    7,
                    &weights,
                )
            })
            .collect();
        let mut actual = [0.0f32; 5];
        for level in 0..5 {
            for (k, value) in actual.iter_mut().enumerate() {
                *value += (channels.iter().map(|c| c[level][k]).sum::<f64>() / 2.0) as f32;
            }
        }
        let expected = [
            0.0030416005756706,
            0.0006262084934860468,
            0.000398605945520103,
            0.00011972583888564259,
            0.00010031183046521619,
        ];
        for (a, b) in actual.into_iter().zip(expected) {
            assert!((f64::from(a) - b).abs() < 1e-7, "{a} vs {b}");
        }
    }

    #[test]
    fn size_errors_precede_pixel_access() {
        for (w, h) in [(0, 8), (8, 0), (7, 8)] {
            let img = za::Rgb8In {
                pixels: &[],
                width: w,
                height: h,
                stride: 0,
            };
            assert!(matches!(
                WassersteinDistortion::check_size(&img),
                Err(Error::TooSmall { .. })
            ));
        }
        let img = za::Rgb8In {
            pixels: &[],
            width: 16384,
            height: 16384,
            stride: 0,
        };
        assert!(matches!(
            WassersteinDistortion::check_size(&img),
            Err(Error::TooLarge { .. })
        ));
    }
}
