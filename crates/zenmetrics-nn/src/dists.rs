//! DISTS (Ding, Ma, Wang, Simoncelli, "Image Quality Assessment: Unifying
//! Structure and Texture Similarity", TPAMI 2020). Parity target: the
//! upstream `DISTS_pytorch` 0.1 package, without its optional resize.
//!
//! The exported trunk (`models/dists/export.py`) is the package's
//! `forward_once` minus its first output: ImageNet normalisation and VGG16
//! with Hanning L2 pooling, returning the five stage outputs. The sixth tap,
//! listed first upstream, is the raw `[0, 1]` input, taken here from the
//! pixels. The head is the rest of `DISTS.forward`: per tap and channel,
//! texture term `S1 = (2·μx·μy + c1) / (μx² + μy² + c1)` and structure term
//! `S2 = (2·σxy + c2) / (σx² + σy² + c2)` with `c1 = c2 = 1e-6`, weighted by
//! the learned α and β (pre-divided by their total at export), and
//! `DISTS = 1 − Σ(α·S1 + β·S2)`. Spatial statistics and per-tap channel sums
//! are accumulated in f64; the cross-tap total is f32, as upstream's is (see
//! `score_stats`).

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_f32, read_meta};
use crate::engine::{self, Tap};
use crate::preprocess::rgb8_to_planar_01;
use crate::{Engine, Error, za};

const C1: f64 = 1e-6;
const C2: f64 = 1e-6;
/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const COLUMN: &str = "dists";

/// DISTS. Lower is closer; `0` for identical inputs. Input: packed sRGB8
/// (any stride).
pub struct Dists {
    backend: Backend,
    /// Per tap: α and β, already divided by their joint sum.
    alpha: Vec<Vec<f32>>,
    beta: Vec<Vec<f32>>,
    min_size: usize,
}

/// Lane count for the plane accumulators below. `plane_dot(x, x)` and the
/// sum-of-squares lane of `plane_sums(x)` accumulate the same f64 products in
/// the same lane order, so they return identical bits; `cov == var` then
/// holds bitwise for an identical pair and `s2 == 1` exactly.
const LANES: usize = 8;

/// `Σ x` and `Σ x²` of one channel plane in a single pass, in f64.
fn plane_sums(ch: &[f32]) -> (f64, f64) {
    let mut sx = [0.0f64; LANES];
    let mut sxx = [0.0f64; LANES];
    let (body, tail) = ch.as_chunks::<LANES>();
    for c in body {
        for i in 0..LANES {
            let x = f64::from(c[i]);
            sx[i] += x;
            sxx[i] += x * x;
        }
    }
    let (mut tx, mut txx) = (0.0f64, 0.0f64);
    for &v in tail {
        let x = f64::from(v);
        tx += x;
        txx += x * x;
    }
    let (mut sum, mut sq) = (tx, txx);
    for i in 0..LANES {
        sum += sx[i];
        sq += sxx[i];
    }
    (sum, sq)
}

/// `Σ x·y` over two channel planes, in f64. Same lane order as
/// [`plane_sums`]' square term: `plane_dot(x, x)` is bitwise equal to it.
fn plane_dot(a: &[f32], b: &[f32]) -> f64 {
    let mut lanes = [0.0f64; LANES];
    let n = a.len().min(b.len());
    let (body, tail) = a[..n].as_chunks::<LANES>();
    let (bbody, btail) = b[..n].as_chunks::<LANES>();
    for (ca, cb) in body.iter().zip(bbody) {
        for i in 0..LANES {
            lanes[i] += f64::from(ca[i]) * f64::from(cb[i]);
        }
    }
    let mut t = 0.0f64;
    for (&va, &vb) in tail.iter().zip(btail) {
        t += f64::from(va) * f64::from(vb);
    }
    let mut sum = t;
    for l in lanes {
        sum += l;
    }
    sum
}

/// One tap's per-channel statistics, plus the map itself for the covariance.
struct TapStats {
    tap: Tap,
    mean: Vec<f64>,
    var: Vec<f64>,
}

impl TapStats {
    fn new(tap: Tap) -> Self {
        let plane = tap.plane();
        let n = plane as f64;
        let mut mean = Vec::with_capacity(tap.c);
        let mut var = Vec::with_capacity(tap.c);
        for ch in tap.data.chunks_exact(plane) {
            let (sx, sxx) = plane_sums(ch);
            let m = sx / n;
            mean.push(m);
            // E[x²] − m², accumulated in the same order as the covariance's
            // `exy − mx·my` in `score_stats`, so an identical pair gets
            // `cov == var` bitwise (see LANES).
            var.push(sxx / n - m * m);
        }
        Self { tap, mean, var }
    }
}

impl Dists {
    /// Load an exported model directory (`dists_trunk.onnx`,
    /// `dists_alpha.f32`, `dists_beta.f32`, `dists.meta`, as written by
    /// `models/dists/export.py`) on `engine`. The ONNX Runtime engines read
    /// `dists_full.onnx` instead of the trunk.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`].
    pub fn from_dir(dir: impl AsRef<Path>, engine: Engine) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let (channels, min_size) = read_meta(&dir.join("dists.meta"))?;
        if channels.first() != Some(&3) {
            return Err(Error::BadModel(
                "dists.meta: first tap must be the 3-channel input".into(),
            ));
        }
        let total = channels.iter().sum();
        let split = |all: Vec<f32>| {
            let mut at = 0;
            channels
                .iter()
                .map(|&c| {
                    let v = all[at..at + c].to_vec();
                    at += c;
                    v
                })
                .collect::<Vec<_>>()
        };
        let alpha = split(read_f32(&dir.join("dists_alpha.f32"), total)?);
        let beta = split(read_f32(&dir.join("dists_beta.f32"), total)?);
        let backend = match engine {
            Engine::Rten => Backend::Trunk(engine::load(
                &dir.join("dists_trunk.onnx"),
                channels.len() - 1,
            )?),
            #[cfg(feature = "ort")]
            Engine::OrtCuda => Backend::Pair(Box::new(engine::ort_engine::OrtCudaPair::load(
                &dir.join("dists_full.onnx"),
                "dists",
            )?)),
        };
        Ok(Self {
            backend,
            alpha,
            beta,
            min_size,
        })
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk(t) => t.engine(),
            Backend::Pair(p) => p.engine(),
            Backend::Image(g) => g.engine(),
        }
    }

    fn check_size(&self, img: &za::Rgb8In<'_>) -> Result<(), Error> {
        if img.width < self.min_size || img.height < self.min_size {
            return Err(Error::TooSmall { min: self.min_size });
        }
        Ok(())
    }

    /// The six taps (input first), validated against the exported channels.
    fn assemble(
        &self,
        input: Vec<f32>,
        h: usize,
        w: usize,
        trunk_taps: Vec<Tap>,
    ) -> Result<Vec<TapStats>, Error> {
        let mut taps = Vec::with_capacity(trunk_taps.len() + 1);
        taps.push(Tap {
            c: 3,
            h,
            w,
            data: input,
        });
        taps.extend(trunk_taps);
        if taps.len() != self.alpha.len() {
            return Err(Error::BadModel(format!(
                "{} taps, expected {}",
                taps.len(),
                self.alpha.len()
            )));
        }
        for (t, a) in taps.iter().zip(&self.alpha) {
            if t.c != a.len() || t.data.len() != t.c * t.plane() || t.plane() == 0 {
                return Err(Error::BadModel(format!(
                    "tap shape [{}, {}, {}], expected {} channels",
                    t.c,
                    t.h,
                    t.w,
                    a.len()
                )));
            }
        }
        Ok(taps.into_iter().map(TapStats::new).collect())
    }

    fn stats(&self, img: &za::Rgb8In<'_>) -> Result<Vec<TapStats>, Error> {
        self.check_size(img)?;
        let Backend::Trunk(t) = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        let x = rgb8_to_planar_01(img);
        let taps = t.run(x.clone(), img.height, img.width)?;
        self.assemble(x, img.height, img.width, taps)
    }

    /// Per tap, the channel sums `Σ α·S1` and `Σ β·S2` in f64; across taps
    /// and in the final `1 − (dist1 + dist2)`, f32 as upstream's tensors
    /// are. The f32 steps matter: the f32-normalised α and β sum to
    /// 1 + 6.5e-8, which upstream's f32 total rounds away (an identical pair
    /// scores exactly 0 there) and an f64 total would not.
    fn score_stats(&self, r: &[TapStats], d: &[TapStats]) -> Result<f64, Error> {
        let (mut dist1, mut dist2) = (0.0f32, 0.0f32);
        for (((rs, ds), alpha), beta) in r.iter().zip(d).zip(&self.alpha).zip(&self.beta) {
            let (mut tap1, mut tap2) = (0.0f64, 0.0f64);
            if (rs.tap.h, rs.tap.w) != (ds.tap.h, ds.tap.w) {
                return Err(Error::SizeMismatch);
            }
            let plane = rs.tap.plane();
            let n = plane as f64;
            for (c, (rc, dc)) in rs
                .tap
                .data
                .chunks_exact(plane)
                .zip(ds.tap.data.chunks_exact(plane))
                .enumerate()
            {
                let (mx, my) = (rs.mean[c], ds.mean[c]);
                let s1 = (2.0 * mx * my + C1) / (mx * mx + my * my + C1);
                let exy = plane_dot(rc, dc) / n;
                let cov = exy - mx * my;
                let s2 = (2.0 * cov + C2) / (rs.var[c] + ds.var[c] + C2);
                tap1 += f64::from(alpha[c]) * s1;
                tap2 += f64::from(beta[c]) * s2;
            }
            dist1 += tap1 as f32;
            dist2 += tap2 as f32;
        }
        Ok(f64::from(1.0f32 - (dist1 + dist2)))
    }

    fn score_pair(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        self.check_size(r)?;
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        let (h, w) = (r.height, r.width);
        let (xr, xd) = (rgb8_to_planar_01(r), rgb8_to_planar_01(d));
        match &self.backend {
            Backend::Trunk(t) => {
                let (tr, td) = t.run_pair(xr.clone(), xd.clone(), h, w)?;
                let rs = self.assemble(xr, h, w, tr)?;
                let ds = self.assemble(xd, h, w, td)?;
                self.score_stats(&rs, &ds)
            }
            Backend::Pair(p) => p.score(xr, xd, h, w),
            Backend::Image(_) => Err(Error::Engine("image graph on a pair model".into())),
        }
    }
}

impl za::Metric for Dists {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "DISTS_pytorch 0.1, no resize; engine {}",
            self.engine_name()
        ))
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new(COLUMN, za::ScoreKind::Difference)];
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
        opts.refuse_map(COLUMN)?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let s = self
            .score_pair(&r, &d)
            .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let view = za::rgb8_view(&r)?;
        let reference = match &self.backend {
            Backend::Trunk(_) => {
                WarmState::Stats(self.stats(&view).metric_err_with(COLUMN, Error::kind)?)
            }
            Backend::Pair(_) => {
                self.check_size(&view)
                    .metric_err_with(COLUMN, Error::kind)?;
                WarmState::Pixels(rgb8_to_planar_01(&view))
            }
            Backend::Image(_) => {
                return Err(Error::Engine("image graph on a pair model".into()))
                    .metric_err_with(COLUMN, Error::kind);
            }
        };
        Ok(Box::new(WarmDists {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}

/// A reference prepared once: tap statistics on a trunk engine, the
/// converted input on a whole-graph engine (which recomputes its features).
struct WarmDists<'m> {
    metric: &'m Dists,
    reference: WarmState,
    size: (usize, usize),
}

enum WarmState {
    Stats(Vec<TapStats>),
    Pixels(Vec<f32>),
}

impl za::WarmReference for WarmDists<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != self.size {
            return Err(Error::SizeMismatch).metric_err_with(COLUMN, Error::kind);
        }
        let s = match &self.reference {
            WarmState::Stats(rs) => self
                .metric
                .stats(&d)
                .and_then(|ds| self.metric.score_stats(rs, &ds)),
            WarmState::Pixels(px) => match &self.metric.backend {
                Backend::Pair(p) => p.score(px.clone(), rgb8_to_planar_01(&d), d.height, d.width),
                Backend::Trunk(_) | Backend::Image(_) => {
                    Err(Error::Engine("warm state does not match engine".into()))
                }
            },
        }
        .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        matches!(self.reference, WarmState::Stats(_))
    }
}
