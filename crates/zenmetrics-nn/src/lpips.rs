//! LPIPS v0.1 (Zhang, Isola, Efros, Shechtman, Wang, "The Unreasonable
//! Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018), with
//! the AlexNet, VGG16 or SqueezeNet trunk. Parity target: the upstream
//! `lpips` 0.1.4 package.
//!
//! The exported trunk graph (`models/lpips/export.py`) is LPIPS's
//! `ScalingLayer` plus every slice of the upstream network, returning one tap
//! per slice. The head below is the rest of `lpips.LPIPS.forward` with
//! `spatial=False`: per tap, unit-normalise over channels
//! (`x / (sqrt(Σc x²) + 1e-10)`), square the difference, weight the channels
//! by the tap's 1×1 `lin` layer, average over space; sum over taps.

use std::borrow::Cow;
use std::path::Path;

use crate::engine::{self, PairGraph, Tap, Trunk};
use crate::preprocess::rgb8_to_planar_pm1;
use crate::{Engine, Error, za};

const EPS: f32 = 1e-10;
/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;

/// Which upstream network an [`Lpips`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LpipsNet {
    /// AlexNet (`net='alex'`, upstream's default and recommended variant).
    Alex,
    /// VGG16 (`net='vgg'`).
    Vgg,
    /// SqueezeNet 1.1 (`net='squeeze'`).
    Squeeze,
}

impl LpipsNet {
    /// Upstream's `net=` name, also the file prefix `lpips_<name>_*`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Alex => "alex",
            Self::Vgg => "vgg",
            Self::Squeeze => "squeeze",
        }
    }

    /// The score column: `lpips_alex`, `lpips_vgg`, `lpips_squeeze`.
    pub const fn column(self) -> &'static str {
        match self {
            Self::Alex => "lpips_alex",
            Self::Vgg => "lpips_vgg",
            Self::Squeeze => "lpips_squeeze",
        }
    }

    /// Every network, in upstream's order.
    pub const ALL: [Self; 3] = [Self::Alex, Self::Vgg, Self::Squeeze];
}

/// LPIPS v0.1. Lower is closer; `0` for identical inputs. Input: packed
/// sRGB8 (any stride).
pub struct Lpips {
    net: LpipsNet,
    backend: Backend,
    lin: Vec<Vec<f32>>,
    min_size: usize,
}

enum Backend {
    /// Trunk graph on the engine, head here: taps come back to the host.
    Trunk(Box<dyn Trunk>),
    /// The whole metric in one graph; only the score comes back.
    #[cfg_attr(not(feature = "ort"), allow(dead_code))]
    Pair(Box<dyn PairGraph>),
}

/// `P.meta`: `channels c0,c1,...` and `min_size n`.
fn read_meta(path: &Path) -> Result<(Vec<usize>, usize), Error> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let bad = |m: &str| Error::BadModel(format!("{}: {m}", path.display()));
    let (mut channels, mut min_size) = (None, None);
    for line in text.lines() {
        match line.split_once(' ') {
            Some(("channels", v)) => {
                channels = Some(
                    v.split(',')
                        .map(|c| {
                            c.trim()
                                .parse::<usize>()
                                .map_err(|_| bad("bad channel count"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            Some(("min_size", v)) => {
                min_size = Some(v.trim().parse::<usize>().map_err(|_| bad("bad min_size"))?)
            }
            _ => {}
        }
    }
    let channels = channels
        .filter(|c| !c.is_empty())
        .ok_or_else(|| bad("no channels line"))?;
    Ok((
        channels,
        min_size.ok_or_else(|| bad("no min_size line"))?.max(1),
    ))
}

fn read_f32(path: &Path, expected: usize) -> Result<Vec<f32>, Error> {
    let raw = std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if raw.len() != expected * 4 {
        return Err(Error::BadModel(format!(
            "{}: {} bytes, expected {}",
            path.display(),
            raw.len(),
            expected * 4
        )));
    }
    let v: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::BadModel(format!(
            "{}: non-finite weight",
            path.display()
        )));
    }
    Ok(v)
}

impl Lpips {
    /// Load `net` from an exported model directory (`lpips_<net>_trunk.onnx`,
    /// `lpips_<net>_lin.f32`, `lpips_<net>.meta`, as written by
    /// `models/lpips/export.py`) on `engine`. The ONNX Runtime engines read
    /// `lpips_<net>_full.onnx` instead of the trunk.
    ///
    /// # Errors
    /// [`Error::Io`] when a file can't be read, [`Error::BadModel`] when a
    /// file doesn't have the expected shape, [`Error::Engine`] when the engine
    /// can't start (for ONNX Runtime: library or CUDA provider not loadable).
    pub fn from_dir(dir: impl AsRef<Path>, net: LpipsNet, engine: Engine) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let p = format!("lpips_{}", net.name());
        let (channels, min_size) = read_meta(&dir.join(format!("{p}.meta")))?;
        let all = read_f32(&dir.join(format!("{p}_lin.f32")), channels.iter().sum())?;
        let mut at = 0;
        let lin = channels
            .iter()
            .map(|&c| {
                let v = all[at..at + c].to_vec();
                at += c;
                v
            })
            .collect();
        let backend = match engine {
            Engine::Rten => Backend::Trunk(engine::load(
                &dir.join(format!("{p}_trunk.onnx")),
                channels.len(),
            )?),
            #[cfg(feature = "ort")]
            Engine::OrtCuda => Backend::Pair(Box::new(engine::ort_engine::OrtCudaPair::load(
                &dir.join(format!("{p}_full.onnx")),
                "lpips",
            )?)),
        };
        Ok(Self {
            net,
            backend,
            lin,
            min_size,
        })
    }

    /// The network this instance runs.
    pub fn net(&self) -> LpipsNet {
        self.net
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk(t) => t.engine(),
            Backend::Pair(p) => p.engine(),
        }
    }

    fn trunk(&self) -> Result<&dyn Trunk, Error> {
        match &self.backend {
            Backend::Trunk(t) => Ok(t.as_ref()),
            Backend::Pair(_) => Err(Error::Engine("no trunk graph on this engine".into())),
        }
    }

    fn check_size(&self, img: &za::Rgb8In<'_>) -> Result<(), Error> {
        if img.width < self.min_size || img.height < self.min_size {
            return Err(Error::TooSmall { min: self.min_size });
        }
        Ok(())
    }

    fn check_taps(&self, taps: &[Tap]) -> Result<(), Error> {
        if taps.len() != self.lin.len() {
            return Err(Error::BadModel(format!(
                "{} taps, expected {}",
                taps.len(),
                self.lin.len()
            )));
        }
        for (t, lin) in taps.iter().zip(&self.lin) {
            if t.c != lin.len() || t.data.len() != t.c * t.plane() || t.plane() == 0 {
                return Err(Error::BadModel(format!(
                    "tap shape [{}, {}, {}], expected {} channels",
                    t.c,
                    t.h,
                    t.w,
                    lin.len()
                )));
            }
        }
        Ok(())
    }

    fn taps(&self, img: &za::Rgb8In<'_>) -> Result<Vec<Tap>, Error> {
        self.check_size(img)?;
        let taps = self
            .trunk()?
            .run(rgb8_to_planar_pm1(img), img.height, img.width)?;
        self.check_taps(&taps)?;
        Ok(taps)
    }

    /// A cold pair on whichever backend is loaded. On a trunk engine the two
    /// trunk runs are concurrent.
    fn score_pair(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        self.check_size(r)?;
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        match &self.backend {
            Backend::Trunk(t) => {
                let (mut rt, dt) = t.run_pair(
                    rgb8_to_planar_pm1(r),
                    rgb8_to_planar_pm1(d),
                    r.height,
                    r.width,
                )?;
                self.check_taps(&rt)?;
                self.check_taps(&dt)?;
                rt.iter_mut().for_each(normalize);
                self.score_taps(&rt, &dt)
            }
            Backend::Pair(p) => p.score(
                rgb8_to_planar_pm1(r),
                rgb8_to_planar_pm1(d),
                r.height,
                r.width,
            ),
        }
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

impl za::Metric for Lpips {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(self.net.column(), za::implementation!(SCORE_REVISION)).with_reference(
            format!(
                "lpips 0.1.4 (net={}, version=0.1); engine {}",
                self.net.name(),
                self.engine_name()
            ),
        )
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const ALEX: &[za::ScoreSpec] =
            &[za::ScoreSpec::new("lpips_alex", za::ScoreKind::Difference)];
        const VGG: &[za::ScoreSpec] = &[za::ScoreSpec::new("lpips_vgg", za::ScoreKind::Difference)];
        const SQUEEZE: &[za::ScoreSpec] = &[za::ScoreSpec::new(
            "lpips_squeeze",
            za::ScoreKind::Difference,
        )];
        Cow::Borrowed(match self.net {
            LpipsNet::Alex => ALEX,
            LpipsNet::Vgg => VGG,
            LpipsNet::Squeeze => SQUEEZE,
        })
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
        let col = self.net.column();
        opts.checkpoint()?;
        opts.refuse_map(col)?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let s = self.score_pair(&r, &d).metric_err_with(col, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(col, s)))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        use za::ResultMetricExt as _;
        let col = self.net.column();
        opts.checkpoint()?;
        let view = za::rgb8_view(&r)?;
        let reference = match &self.backend {
            Backend::Trunk(_) => {
                let mut taps = self.taps(&view).metric_err_with(col, Error::kind)?;
                taps.iter_mut().for_each(normalize);
                WarmState::Taps(taps)
            }
            Backend::Pair(_) => {
                self.check_size(&view).metric_err_with(col, Error::kind)?;
                WarmState::Pixels(rgb8_to_planar_pm1(&view))
            }
        };
        Ok(Box::new(WarmLpips {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}

/// A reference prepared once: normalised taps on a trunk engine, the
/// converted input on a whole-graph engine (which recomputes its features).
struct WarmLpips<'m> {
    metric: &'m Lpips,
    reference: WarmState,
    size: (usize, usize),
}

enum WarmState {
    Taps(Vec<Tap>),
    Pixels(Vec<f32>),
}

impl za::WarmReference for WarmLpips<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        let col = self.metric.net.column();
        opts.checkpoint()?;
        opts.refuse_map(col)?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != self.size {
            return Err(Error::SizeMismatch).metric_err_with(col, Error::kind);
        }
        let s = match &self.reference {
            WarmState::Taps(taps) => self.metric.score_against(taps, &d),
            WarmState::Pixels(px) => match &self.metric.backend {
                Backend::Pair(p) => p.score(px.clone(), rgb8_to_planar_pm1(&d), d.height, d.width),
                Backend::Trunk(_) => Err(Error::Engine("warm state does not match engine".into())),
            },
        }
        .metric_err_with(col, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(col, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        matches!(self.reference, WarmState::Taps(_))
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
