//! CKDN (Zheng, Fu, Zeng, Zha, Luo, "Learning Conditional Knowledge
//! Distillation for Degraded-Reference Image Quality Assessment", ICCV 2021;
//! upstream `researchmm/CKDN`, MIT, trained checkpoint epoch 14). Parity
//! target: that model's `predict_score` output `[:, 0]` (`dr`).
//!
//! CKDN is a degraded-reference metric: its upstream input is a 12-channel
//! stack `(restored, degraded_ref, second_restored, pristine_ref)`. For plain
//! full-reference scoring the exported graphs feed
//! `cat(dist, ref, dist, ref)`, under which `fr == dr`, `nr` is a constant
//! and `diff == 0`, so the metric is just `dr`.
//!
//! `models/ckdn/export.py` splits that into three graphs: `ckdn_qse` and
//! `ckdn_dte` are the two encoder branches (ResNet conv1+maxpool+layer1) of
//! the distorted and reference image respectively, each a `[1, 256, H/4,
//! W/4]` tap; `ckdn_tail` is upstream's `csp` (layer2) + avgpool + `fc_` +
//! sigmoid on their difference. The Rust head is the difference itself, so
//! `warm_reference` caches `dte(ref)`. On whole-graph engines
//! `ckdn_full.onnx` runs the upstream forward end to end. Inputs are u8
//! pixels as f32 (0..255); the graphs normalise by `mean*255`/`std*255`
//! inside, as upstream's PrefetchLoader does.
//!
//! Before the graphs, each image goes through the eval transform upstream's
//! `predict_score.py` applies via its timm loader (`img_size=288`,
//! `crop_pct=0.875`): resize so the short side is 329, centre-crop 288. The
//! AIC-4 published column — which pyiqa's wrapper produces — uses an
//! antialiased bicubic resize for the distorted image but **nearest** for
//! the reference (verified by bit-exact input capture); this port follows
//! that convention, since it is what the scored data was generated with.

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_meta};
use crate::engine::{self, Tap, Trunk};
use crate::preprocess::rgb8_to_planar_255;
use crate::resize::{center_crop, resize_bicubic_aa, resize_nearest};
use crate::{Engine, Error, WeightsConsent, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 2;
const COLUMN: &str = "ckdn";

/// Eval transform: `Resize(329)` on the short side (long side floored to keep
/// aspect) then `CenterCrop(288)` — upstream's `img_size=288, crop_pct=0.875`.
const RESIZE_SHORT: usize = 329;
const CROP: usize = 288;

/// CKDN. Higher is closer (a sigmoid quality score in `(0, 1)`). Input:
/// packed sRGB8 (any stride).
pub struct Ckdn {
    backend: Backend,
    /// On trunk engines: the reference-side encoder and the diff→dr tail.
    parts: Option<Parts>,
    channels: usize,
}

struct Parts {
    dte: Box<dyn Trunk>,
    tail: Box<dyn Trunk>,
}

impl Ckdn {
    /// This model's weights' published terms: the trained checkpoint is
    /// distributed through a Google Drive link in upstream's README with no
    /// licence stated for it (`licence_class = "unknown"` in the card), so
    /// loading needs a [`WeightsConsent`] accepting these terms.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::Unknown,
        terms_url: "https://github.com/researchmm/CKDN",
    };

    /// Load an exported model directory (`ckdn_qse.onnx`, `ckdn_dte.onnx`,
    /// `ckdn_tail.onnx`, `ckdn.meta`, as written by `models/ckdn/export.py`)
    /// on `engine`. The ONNX Runtime engines read `ckdn_full.onnx` instead of
    /// the trunk graphs.
    ///
    /// `consent` must accept [`Self::WEIGHTS`] (the weights state no licence);
    /// it is checked before any file is read.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`], plus [`Error::WeightsNotAccepted`] when
    /// `consent` does not accept [`Self::WEIGHTS`].
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        consent: &WeightsConsent,
    ) -> Result<Self, Error> {
        consent.check(&Self::WEIGHTS)?;
        let dir = dir.as_ref();
        let (channels, _min_size) = read_meta(&dir.join("ckdn.meta"))?;
        if channels.len() != 1 {
            return Err(Error::BadModel(
                "ckdn.meta: expected one channel count".into(),
            ));
        }
        let (backend, parts) = match engine {
            Engine::Rten => (
                Backend::Trunk(engine::load(&dir.join("ckdn_qse.onnx"), 1)?),
                Some(Parts {
                    dte: engine::load(&dir.join("ckdn_dte.onnx"), 1)?,
                    tail: engine::load(&dir.join("ckdn_tail.onnx"), 1)?,
                }),
            ),
            #[cfg(feature = "ort")]
            Engine::OrtCuda => (
                Backend::Pair(Box::new(engine::ort_engine::OrtCudaPair::load(
                    &dir.join("ckdn_full.onnx"),
                    "ckdn",
                )?)),
                None,
            ),
        };
        Ok(Self {
            backend,
            parts,
            channels: channels[0],
        })
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk(t) => t.engine(),
            Backend::Pair(p) => p.engine(),
        }
    }

    /// The eval transform: resize so the short side is [`RESIZE_SHORT`]
    /// (long side floored to keep aspect, torchvision's `int()`), then
    /// centre-crop [`CROP`]. `nearest` selects the reference side's kernel —
    /// the published convention resizes the distorted image with antialiased
    /// bicubic but the reference with nearest. Returns planar f32 `0..255`,
    /// `[3, CROP, CROP]`.
    fn prep(img: &za::Rgb8In<'_>, nearest: bool) -> Vec<f32> {
        let (h, w) = (img.height, img.width);
        let (nh, nw) = if w <= h {
            (RESIZE_SHORT * h / w, RESIZE_SHORT)
        } else {
            (RESIZE_SHORT, RESIZE_SHORT * w / h)
        };
        let planar = rgb8_to_planar_255(img);
        let resized = if nearest {
            resize_nearest(&planar, 3, h, w, nh, nw)
        } else {
            resize_bicubic_aa(&planar, 3, h, w, nh, nw)
        };
        center_crop(&resized, 3, nh, nw, CROP)
    }

    /// A single-tap run's `[c, h, w]` output, checked against `channels`.
    fn take(taps: Vec<Tap>, channels: usize) -> Result<Tap, Error> {
        let mut it = taps.into_iter();
        let (Some(t), None) = (it.next(), it.next()) else {
            return Err(Error::BadModel("expected exactly one tap".into()));
        };
        if t.c != channels || t.plane() == 0 || t.data.len() != t.c * t.plane() {
            return Err(Error::BadModel(format!(
                "tap shape [{}, {}, {}], expected {channels} channels",
                t.c, t.h, t.w
            )));
        }
        Ok(t)
    }

    /// `qse(img)` on a trunk engine — the distorted-image encoder, fed the
    /// antialiased-bicubic crop.
    fn encode_qse(&self, img: &za::Rgb8In<'_>) -> Result<Tap, Error> {
        let Backend::Trunk(qse) = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        Self::take(qse.run(Self::prep(img, false), CROP, CROP)?, self.channels)
    }

    /// `dte(img)` on a trunk engine — the reference-side encoder, fed the
    /// nearest-neighbour crop.
    fn encode_dte(&self, img: &za::Rgb8In<'_>) -> Result<Tap, Error> {
        let Some(parts) = &self.parts else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        Self::take(
            parts.dte.run(Self::prep(img, true), CROP, CROP)?,
            self.channels,
        )
    }

    /// `tail(qse(d) - dte(r))`: `dr`.
    fn dr(&self, qse_d: &Tap, dte_r: &Tap) -> Result<f64, Error> {
        let Some(parts) = &self.parts else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        if (qse_d.h, qse_d.w, qse_d.c) != (dte_r.h, dte_r.w, dte_r.c) {
            return Err(Error::SizeMismatch);
        }
        let diff: Vec<f32> = qse_d
            .data
            .iter()
            .zip(&dte_r.data)
            .map(|(&a, &b)| a - b)
            .collect();
        // The tail emits the score as a [1, 1, 1, 1] map.
        let t = Self::take(parts.tail.run_c(diff, qse_d.c, qse_d.h, qse_d.w)?, 1)?;
        Ok(f64::from(t.data[0]))
    }

    fn score_pair(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        match &self.backend {
            Backend::Trunk(qse) => {
                let parts = self.parts.as_ref().expect("trunk parts");
                let (xr, xd) = (Self::prep(r, true), Self::prep(d, false));
                let (qd, rr) = std::thread::scope(|s| {
                    let hj = s.spawn(|| qse.run(xd, CROP, CROP));
                    let rr = parts.dte.run(xr, CROP, CROP);
                    (
                        hj.join()
                            .map_err(|_| Error::Engine("encoder run panicked".into()))
                            .and_then(|r| r),
                        rr,
                    )
                });
                self.dr(
                    &Self::take(qd?, self.channels)?,
                    &Self::take(rr?, self.channels)?,
                )
            }
            Backend::Pair(p) => p.score(Self::prep(r, true), Self::prep(d, false), CROP, CROP),
        }
    }
}

impl za::Metric for Ckdn {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "researchmm/CKDN checkpoint epoch 14, cat(dist, ref, dist, ref)[:,0], resize short->329 + cc288 (dist bicubic-aa, ref nearest); engine {}",
            self.engine_name()
        ))
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new(COLUMN, za::ScoreKind::Similarity)];
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
            Backend::Trunk(_) => WarmState::Tap(
                self.encode_dte(&view)
                    .metric_err_with(COLUMN, Error::kind)?,
            ),
            Backend::Pair(_) => WarmState::Pixels(Self::prep(&view, true)),
        };
        Ok(Box::new(WarmCkdn {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}

/// A reference prepared once: its dte encoding on a trunk engine, the
/// converted input on a whole-graph engine (which recomputes its features).
struct WarmCkdn<'m> {
    metric: &'m Ckdn,
    reference: WarmState,
    size: (usize, usize),
}

enum WarmState {
    Tap(Tap),
    Pixels(Vec<f32>),
}

impl za::WarmReference for WarmCkdn<'_> {
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
            WarmState::Tap(dte_r) => self
                .metric
                .encode_qse(&d)
                .and_then(|qd| self.metric.dr(&qd, dte_r)),
            WarmState::Pixels(px) => match &self.metric.backend {
                Backend::Pair(p) => p.score(px.clone(), Ckdn::prep(&d, false), CROP, CROP),
                Backend::Trunk(_) => Err(Error::Engine("warm state does not match engine".into())),
            },
        }
        .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        matches!(self.reference, WarmState::Tap(_))
    }
}
