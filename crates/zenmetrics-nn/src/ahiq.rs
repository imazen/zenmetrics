//! AHIQ (CVPRW 2022, upstream `IIGROUP/AHIQ`, MIT): a Vit-B/8 trunk (blocks
//! 0–4, CLS dropped → 3840 channels at 28×28) plus ResNet-50 layer-1
//! Bottleneck taps (3×256 → 768 at 56×56); the head deforms the CNN taps by
//! offsets learned from the *reference* ViT map, then a pixel-score regressor
//! pools `f·w / Σw`. torchvision's `DeformConv2d` has no ONNX op: the export
//! decomposes it as nine bilinear `grid_sample` taps + a 1×1 conv (verified
//! ≤4e-7 against torchvision on random inputs).
//!
//! Upstream's eval protocol scores an image as the mean over nine
//! deterministic 224×224 crops (`five_point_crop` idx 0–8) — no resizing —
//! so inputs below 224 are refused. `ahiq_trunk.onnx` emits the two taps,
//! `ahiq_head.onnx` takes `(vit_d, vit_r, cnn_d, cnn_r)` → score on a trunk
//! engine, and `ahiq_full.onnx` is the whole pair graph on ort-cuda.

use std::borrow::Cow;
use std::path::Path;

use crate::common::read_meta;
use crate::engine::{self, AuxInput, PairGraph, Tap, Trunk};
use crate::preprocess::rgb8_to_planar_pm1;
use crate::{Engine, Error, WeightsConsent, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const COLUMN: &str = "ahiq";
/// Upstream's fixed crop side (the ViT consumes 224×224 tokens).
const CROP: usize = 224;
const NCROPS: usize = 9;

/// A learned IQA metric: nine-crop AHIQ, mean over the ensemble.
pub struct Ahiq {
    backend: Backend,
    /// `[vit_c, cnn_c]` from `ahiq.meta`.
    channels: [usize; 2],
    min_size: usize,
}

enum Backend {
    /// `ahiq_trunk.onnx` (x → vit, cnn taps) + `ahiq_head.onnx`
    /// ((vit_d, vit_r, cnn_d, cnn_r) → score).
    Trunk {
        trunk: Box<dyn Trunk>,
        head: Box<dyn Trunk>,
    },
    /// `ahiq_full.onnx` (r, d) → score, one crop at a time.
    #[cfg_attr(not(feature = "ort"), allow(dead_code))]
    Pair(Box<dyn PairGraph>),
}

/// `five_point_crop` indices 0–8 as (top, left) pairs — the four corners,
/// the centre, and the four edge centres. Needs `h, w >= CROP`.
fn crops(h: usize, w: usize) -> [(usize, usize); NCROPS] {
    let (ch, cw, k) = (h / 2, w / 2, CROP);
    [
        (0, 0),
        (0, w - k),
        (h - k, 0),
        (h - k, w - k),
        (ch - k / 2, cw - k / 2),
        (ch - k / 2, 0),
        (ch - k / 2, w - k),
        (0, cw - k / 2),
        (h - k, cw - k / 2),
    ]
}

/// Planar `[c, h, w]` → `[c, k, k]` at (top, left).
fn crop_at(p: &[f32], c: usize, h: usize, w: usize, t: usize, l: usize, k: usize) -> Vec<f32> {
    debug_assert_eq!(p.len(), c * h * w);
    let mut out = Vec::with_capacity(c * k * k);
    for ch in p.chunks_exact(h * w).take(c) {
        for row in ch[t * w..(t + k) * w].chunks_exact(w) {
            out.extend_from_slice(&row[l..l + k]);
        }
    }
    out
}

impl Ahiq {
    /// This model's weights' published terms: the epoch-32 checkpoint
    /// (`1Nk-IpjnDNXbWacoh3T69wkSYhYYWip2W.pth` from upstream's Google Drive
    /// link) states no licence, so loading needs a [`WeightsConsent`]
    /// accepting these terms.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::Unknown,
        terms_url: "https://github.com/IIGROUP/AHIQ",
    };

    /// Load an exported model directory (`ahiq_trunk.onnx`, `ahiq_head.onnx`,
    /// `ahiq_full.onnx`, `ahiq.meta`, as written by `models/ahiq/export.py`)
    /// on `engine`. The ONNX Runtime engines read `ahiq_full.onnx` instead
    /// of the trunk/head graphs.
    ///
    /// `consent` must accept [`Self::WEIGHTS`]; it is checked before any
    /// file is read.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`], plus [`Error::WeightsNotAccepted`]
    /// when `consent` does not accept [`Self::WEIGHTS`].
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        consent: &WeightsConsent,
    ) -> Result<Self, Error> {
        consent.check(&Self::WEIGHTS)?;
        let dir = dir.as_ref();
        let (channels, min_size) = read_meta(&dir.join("ahiq.meta"))?;
        let channels: [usize; 2] = channels
            .as_slice()
            .try_into()
            .map_err(|_| Error::BadModel("ahiq.meta: expected 2 tap widths".into()))?;
        let backend = match engine {
            Engine::Rten => Backend::Trunk {
                trunk: engine::load(&dir.join("ahiq_trunk.onnx"), 2)?,
                head: engine::load(&dir.join("ahiq_head.onnx"), 1)?,
            },
            #[cfg(feature = "ort")]
            Engine::OrtCuda => Backend::Pair(Box::new(engine::ort_engine::OrtCudaPair::load(
                &dir.join("ahiq_full.onnx"),
                "ahiq",
            )?)),
        };
        Ok(Self {
            backend,
            channels,
            min_size,
        })
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk { trunk, .. } => trunk.engine(),
            Backend::Pair(p) => p.engine(),
        }
    }

    /// Upstream `ToTensor + Normalize(0.5)` — planar f32 in `[-1, 1]`. No
    /// resizing anywhere in the protocol.
    fn prep(img: &za::Rgb8In<'_>) -> Vec<f32> {
        rgb8_to_planar_pm1(img)
    }

    /// Check a trunk tap's channel width against the meta.
    fn check_taps(&self, taps: &[Tap]) -> Result<(), Error> {
        if taps.len() != 2 || taps[0].c != self.channels[0] || taps[1].c != self.channels[1] {
            return Err(Error::BadModel(format!(
                "trunk taps [{:?}], expected [vit {}, cnn {}]",
                taps.iter().map(|t| t.c).collect::<Vec<_>>(),
                self.channels[0],
                self.channels[1]
            )));
        }
        Ok(())
    }

    /// `trunk(crop)` → `[vit, cnn]`.
    fn taps_of(&self, trunk: &dyn Trunk, crop: Vec<f32>) -> Result<Vec<Tap>, Error> {
        let taps = trunk.run(crop, CROP, CROP)?;
        self.check_taps(&taps)?;
        Ok(taps)
    }

    /// `head(vit_d, vit_r, cnn_d, cnn_r)` → scalar score.
    fn head_score(&self, td: &[Tap], tr: &[Tap]) -> Result<f64, Error> {
        let Backend::Trunk { head, .. } = &self.backend else {
            return Err(Error::Engine("no head graph on this engine".into()));
        };
        let (vd, cd) = (&td[0], &td[1]);
        let (vr, cr) = (&tr[0], &tr[1]);
        let aux = [
            AuxInput {
                index: 1,
                shape: [1, vr.c, vr.h, vr.w],
                data: vr.data.clone(),
            },
            AuxInput {
                index: 2,
                shape: [1, cd.c, cd.h, cd.w],
                data: cd.data.clone(),
            },
            AuxInput {
                index: 3,
                shape: [1, cr.c, cr.h, cr.w],
                data: cr.data.clone(),
            },
        ];
        let out = head.run_c_aux(vd.data.clone(), vd.c, vd.h, vd.w, &aux)?;
        let score = out
            .first()
            .and_then(|t| t.data.first())
            .ok_or_else(|| Error::BadModel("ahiq head produced no scalar".into()))?;
        Ok(f64::from(*score))
    }

    /// The nine-crop ensemble score on a trunk engine, both sides fresh.
    fn score_trunk(
        &self,
        trunk: &dyn Trunk,
        pr: &[f32],
        pd: &[f32],
        h: usize,
        w: usize,
    ) -> Result<f64, Error> {
        let mut s = 0.0;
        for (t, l) in crops(h, w) {
            let rc = crop_at(pr, 3, h, w, t, l, CROP);
            let dc = crop_at(pd, 3, h, w, t, l, CROP);
            let (tr, td) = trunk.run_pair(rc, dc, CROP, CROP)?;
            self.check_taps(&tr)?;
            self.check_taps(&td)?;
            s += self.head_score(&td, &tr)?;
        }
        Ok(s / NCROPS as f64)
    }

    /// The nine-crop score against cached reference taps (warm path).
    fn score_trunk_warm(
        &self,
        trunk: &dyn Trunk,
        ref_taps: &[Vec<Tap>],
        pd: &[f32],
        h: usize,
        w: usize,
    ) -> Result<f64, Error> {
        let mut s = 0.0;
        for (i, (t, l)) in crops(h, w).into_iter().enumerate() {
            let td = self.taps_of(trunk, crop_at(pd, 3, h, w, t, l, CROP))?;
            s += self.head_score(&td, &ref_taps[i])?;
        }
        Ok(s / NCROPS as f64)
    }

    /// The reference-side trunk taps for all nine crops (warm cache).
    fn ref_taps(&self, img: &za::Rgb8In<'_>) -> Result<Vec<Vec<Tap>>, Error> {
        let Backend::Trunk { trunk, .. } = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        let pr = Self::prep(img);
        let (h, w) = (img.height, img.width);
        crops(h, w)
            .into_iter()
            .map(|(t, l)| self.taps_of(&**trunk, crop_at(&pr, 3, h, w, t, l, CROP)))
            .collect()
    }

    fn score_pair(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        if r.width.min(r.height) < self.min_size {
            return Err(Error::TooSmall { min: self.min_size });
        }
        match &self.backend {
            Backend::Trunk { trunk, .. } => {
                let (pr, pd) = (Self::prep(r), Self::prep(d));
                self.score_trunk(&**trunk, &pr, &pd, r.height, r.width)
            }
            Backend::Pair(p) => {
                let (pr, pd) = (Self::prep(r), Self::prep(d));
                Self::score_pair_graph(&**p, &pr, &pd, r.height, r.width)
            }
        }
    }

    /// The nine-crop ensemble score through the whole-pair graph.
    fn score_pair_graph(
        p: &dyn PairGraph,
        pr: &[f32],
        pd: &[f32],
        h: usize,
        w: usize,
    ) -> Result<f64, Error> {
        let mut s = 0.0;
        for (t, l) in crops(h, w) {
            s += p.score(
                crop_at(pr, 3, h, w, t, l, CROP),
                crop_at(pd, 3, h, w, t, l, CROP),
                CROP,
                CROP,
            )?;
        }
        Ok(s / NCROPS as f64)
    }
}

impl za::Metric for Ahiq {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "IIGROUP/AHIQ epoch-32 checkpoint (vit_base_patch8_224 blocks 0-4 + resnet50 layer1 taps, deformable fusion + pixel regressor), mean of nine 224x224 crops; engine {}",
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
        if view.width.min(view.height) < self.min_size {
            return Err(Error::TooSmall { min: self.min_size })
                .metric_err_with(COLUMN, Error::kind);
        }
        let reference = match &self.backend {
            Backend::Trunk { .. } => {
                WarmState::Taps(self.ref_taps(&view).metric_err_with(COLUMN, Error::kind)?)
            }
            Backend::Pair(_) => WarmState::Pixels(Self::prep(&view)),
        };
        Ok(Box::new(WarmAhiq {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}

/// A reference prepared once: its nine crop taps on a trunk engine, the
/// converted pixels on a whole-graph engine (which recomputes features).
struct WarmAhiq<'m> {
    metric: &'m Ahiq,
    reference: WarmState,
    size: (usize, usize),
}

enum WarmState {
    /// Nine `[vit_r, cnn_r]` tap pairs, one per crop.
    Taps(Vec<Vec<Tap>>),
    Pixels(Vec<f32>),
}

impl za::WarmReference for WarmAhiq<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let view = za::rgb8_view(&d)?;
        if (view.width, view.height) != self.size {
            return Err(Error::SizeMismatch).metric_err_with(COLUMN, Error::kind);
        }
        let (h, w) = self.size;
        let s = match (&self.metric.backend, &self.reference) {
            (Backend::Trunk { trunk, .. }, WarmState::Taps(ref_taps)) => {
                let pd = Ahiq::prep(&view);
                self.metric.score_trunk_warm(&**trunk, ref_taps, &pd, h, w)
            }
            (Backend::Pair(p), WarmState::Pixels(pr)) => {
                let pd = Ahiq::prep(&view);
                Ahiq::score_pair_graph(&**p, pr, &pd, h, w)
            }
            _ => Err(Error::Engine("warm state does not match engine".into())),
        }
        .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        matches!(self.reference, WarmState::Taps(_))
    }
}
