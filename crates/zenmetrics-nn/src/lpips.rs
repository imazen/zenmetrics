//! LPIPS v0.1, VGG16 trunk (Zhang, Isola, Efros, Shechtman, Wang, "The
//! Unreasonable Effectiveness of Deep Features as a Perceptual Metric",
//! CVPR 2018). Parity target: the upstream `lpips` 0.1.4 package.
//!
//! The exported graph (`models/lpips-vgg/export.py`) is LPIPS's
//! `ScalingLayer` plus VGG16 features through `relu5_3`, returning the five
//! taps. The head below is the rest of `lpips.LPIPS.forward` with
//! `spatial=False`: per tap, unit-normalise over channels
//! (`x / (sqrt(Σc x²) + 1e-10)`), square the difference, weight the channels
//! by the tap's 1×1 `lin` layer, average over space; sum over taps.

use std::borrow::Cow;
use std::path::Path;

use crate::engine::{self, Tap, Trunk};
use crate::preprocess::rgb8_to_planar_pm1;
use crate::{Error, za};

const TRUNK_FILE: &str = "lpips_vgg_trunk.onnx";
const LIN_FILE: &str = "lpips_vgg_lin.f32";
const CHANNELS: [usize; 5] = [64, 128, 256, 512, 512];
const EPS: f32 = 1e-10;
/// VGG16 pools four times before `relu5_3`; below 16 px the last tap is empty.
const MIN_SIZE: usize = 16;
/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const FAMILY: &str = "lpips_vgg";

/// LPIPS v0.1 with the VGG16 trunk. Lower is closer; `0` for identical
/// inputs. Input: packed sRGB8 (any stride).
pub struct LpipsVgg {
    trunk: Box<dyn Trunk>,
    lin: [Vec<f32>; 5],
}

impl LpipsVgg {
    /// Load an exported model directory: `lpips_vgg_trunk.onnx` and
    /// `lpips_vgg_lin.f32`, as written by `models/lpips-vgg/export.py`.
    ///
    /// # Errors
    /// [`Error::Io`] when a file can't be read, [`Error::BadModel`] when
    /// either file doesn't have the expected shape.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let lin_path = dir.join(LIN_FILE);
        let raw = std::fs::read(&lin_path).map_err(|source| Error::Io {
            path: lin_path.clone(),
            source,
        })?;
        let total: usize = CHANNELS.iter().sum();
        if raw.len() != total * 4 {
            return Err(Error::BadModel(format!(
                "{LIN_FILE}: {} bytes, expected {}",
                raw.len(),
                total * 4
            )));
        }
        let all: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if all.iter().any(|v| !v.is_finite()) {
            return Err(Error::BadModel(format!("{LIN_FILE}: non-finite weight")));
        }
        let mut at = 0;
        let lin = CHANNELS.map(|c| {
            let v = all[at..at + c].to_vec();
            at += c;
            v
        });
        let trunk = engine::load(&dir.join(TRUNK_FILE), CHANNELS.len())?;
        Ok(Self { trunk, lin })
    }

    fn taps(&self, img: &za::Rgb8In<'_>) -> Result<Vec<Tap>, Error> {
        check_size(img)?;
        let taps = self
            .trunk
            .run(rgb8_to_planar_pm1(img), img.height, img.width)?;
        check_taps(&taps)?;
        Ok(taps)
    }

    /// Reference and distorted taps, the two trunk runs concurrent.
    fn tap_pair(
        &self,
        r: &za::Rgb8In<'_>,
        d: &za::Rgb8In<'_>,
    ) -> Result<(Vec<Tap>, Vec<Tap>), Error> {
        check_size(r)?;
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        let (a, b) = self.trunk.run_pair(
            rgb8_to_planar_pm1(r),
            rgb8_to_planar_pm1(d),
            r.height,
            r.width,
        )?;
        check_taps(&a)?;
        check_taps(&b)?;
        Ok((a, b))
    }

    /// The score of `dist` against reference taps already normalised.
    fn score_against(&self, reference: &[Tap], dist: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let taps = self.taps(dist)?;
        self.score_taps(reference, &taps)
    }

    fn score_taps(&self, reference: &[Tap], taps: &[Tap]) -> Result<f64, Error> {
        let mut total = 0.0f64;
        for ((r, d), lin) in reference.iter().zip(taps).zip(&self.lin) {
            if (r.h, r.w) != (d.h, d.w) {
                return Err(Error::SizeMismatch);
            }
            total += tap_distance(r, d, lin);
        }
        Ok(total)
    }

    fn id(&self) -> za::MetricId {
        za::MetricId::new(FAMILY, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "lpips 0.1.4 (net=vgg, version=0.1); engine {}",
            self.trunk.engine()
        ))
    }
}

fn check_size(img: &za::Rgb8In<'_>) -> Result<(), Error> {
    if img.width < MIN_SIZE || img.height < MIN_SIZE {
        return Err(Error::TooSmall { min: MIN_SIZE });
    }
    Ok(())
}

fn check_taps(taps: &[Tap]) -> Result<(), Error> {
    if taps.len() != CHANNELS.len() {
        return Err(Error::BadModel(format!(
            "{} taps, expected {}",
            taps.len(),
            CHANNELS.len()
        )));
    }
    for (t, &c) in taps.iter().zip(&CHANNELS) {
        if t.c != c || t.data.len() != t.c * t.plane() || t.plane() == 0 {
            return Err(Error::BadModel(format!(
                "tap shape [{}, {}, {}], expected {c} channels",
                t.c, t.h, t.w
            )));
        }
    }
    Ok(())
}

/// Unit-normalise a tap over channels, in place (LPIPS `normalize_tensor`).
fn normalize(t: &mut Tap) {
    let plane = t.plane();
    let mut norm = vec![0.0f32; plane];
    for ch in t.data.chunks_exact(plane) {
        for (n, &v) in norm.iter_mut().zip(ch) {
            *n += v * v;
        }
    }
    for n in &mut norm {
        *n = n.sqrt() + EPS;
    }
    for ch in t.data.chunks_exact_mut(plane) {
        for (v, &n) in ch.iter_mut().zip(&norm) {
            *v /= n;
        }
    }
}

/// `mean_p Σc lin[c]·(r[c,p] − d̂[c,p])²` with `r` already normalised and
/// `d` normalised here on the fly.
fn tap_distance(r: &Tap, d: &Tap, lin: &[f32]) -> f64 {
    let plane = d.plane();
    let mut norm = vec![0.0f32; plane];
    for ch in d.data.chunks_exact(plane) {
        for (n, &v) in norm.iter_mut().zip(ch) {
            *n += v * v;
        }
    }
    for n in &mut norm {
        *n = n.sqrt() + EPS;
    }
    let mut acc = vec![0.0f32; plane];
    for ((rc, dc), &w) in r
        .data
        .chunks_exact(plane)
        .zip(d.data.chunks_exact(plane))
        .zip(lin)
    {
        for (((a, &rv), &dv), &n) in acc.iter_mut().zip(rc).zip(dc).zip(&norm) {
            let diff = rv - dv / n;
            *a += w * diff * diff;
        }
    }
    acc.iter().map(|&v| f64::from(v)).sum::<f64>() / plane as f64
}

impl za::Metric for LpipsVgg {
    fn id(&self) -> za::MetricId {
        LpipsVgg::id(self)
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new(FAMILY, za::ScoreKind::Difference)];
        Cow::Borrowed(OUTPUTS)
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
        opts.refuse_map(FAMILY)?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let (mut rt, dt) = self.tap_pair(&r, &d).metric_err_with(FAMILY, Error::kind)?;
        rt.iter_mut().for_each(normalize);
        opts.checkpoint()?;
        let s = self
            .score_taps(&rt, &dt)
            .metric_err_with(FAMILY, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(FAMILY, s)))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let view = za::rgb8_view(&r)?;
        let mut taps = self.taps(&view).metric_err_with(FAMILY, Error::kind)?;
        taps.iter_mut().for_each(normalize);
        Ok(Box::new(WarmLpips {
            metric: self,
            taps,
            size: (view.width, view.height),
        }))
    }
}

/// A reference whose normalised taps are computed once.
struct WarmLpips<'m> {
    metric: &'m LpipsVgg,
    taps: Vec<Tap>,
    size: (usize, usize),
}

impl za::WarmReference for WarmLpips<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(FAMILY)?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != self.size {
            return Err(Error::SizeMismatch).metric_err_with(FAMILY, Error::kind);
        }
        let s = self
            .metric
            .score_against(&self.taps, &d)
            .metric_err_with(FAMILY, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(FAMILY, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tap(c: usize, h: usize, w: usize, f: impl Fn(usize) -> f32) -> Tap {
        Tap {
            c,
            h,
            w,
            data: (0..c * h * w).map(f).collect(),
        }
    }

    #[test]
    fn identical_taps_score_zero() {
        let mut r = tap(4, 3, 5, |i| (i as f32 * 0.37).sin());
        let d = r.clone();
        normalize(&mut r);
        assert_eq!(tap_distance(&r, &d, &[1.0, 2.0, 3.0, 4.0]), 0.0);
    }

    #[test]
    fn matches_direct_formula() {
        let r0 = tap(3, 2, 2, |i| i as f32 - 4.0);
        let d0 = tap(3, 2, 2, |i| (i as f32 * 1.3).cos());
        let lin = [0.5f32, 1.5, 2.0];
        let mut r = r0.clone();
        normalize(&mut r);
        let got = tap_distance(&r, &d0, &lin);
        let plane = 4;
        let norm = |t: &Tap, p: usize| {
            let s: f32 = (0..3).map(|c| t.data[c * plane + p].powi(2)).sum();
            s.sqrt() + EPS
        };
        let mut want = 0.0f64;
        for p in 0..plane {
            let mut acc = 0.0f32;
            for c in 0..3 {
                let a = r0.data[c * plane + p] / norm(&r0, p);
                let b = d0.data[c * plane + p] / norm(&d0, p);
                acc += lin[c] * (a - b) * (a - b);
            }
            want += f64::from(acc);
        }
        want /= plane as f64;
        assert!(
            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }
}
