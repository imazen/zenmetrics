//! AHIQ (CVPRW 2022, upstream `IIGROUP/AHIQ`, MIT): a ViT-B/8 trunk
//! (`vit_base_patch8_224.augreg_in21k_ft_in1k` — the augreg checkpoint
//! upstream's pinned timm 0.5.4 resolves; blocks 0–4, CLS dropped → 3840
//! channels at 28×28) plus ResNet-50 layer-1 Bottleneck taps
//! (`resnet50.a1_in1k`; 3×256 → 768 at 56×56); the head deforms the CNN taps
//! by offsets learned from the *reference* ViT map, then a pixel-score
//! regressor pools `f·w / Σw`. torchvision's `DeformConv2d` has no ONNX op:
//! the export decomposes it as nine bilinear `grid_sample` taps + a 1×1 conv
//! (verified ≤4e-7 against torchvision on random inputs).
//!
//! The port's eval protocol scores an image as the mean over nine
//! deterministic 224×224 crops (`five_point_crop` idx 0–8) — a deliberate
//! deterministic deviation: upstream `test.py`'s default is `n_ensemble=20`
//! *seeded random* crops (`options/base_options.py:29`, seed 1919), and its
//! `n_ensemble <= 9` branch is this protocol — `n=1` scores the whole
//! image, `n=2..8` means the first `n` of the nine crops, `n=9` all nine.
//! No resizing anywhere, so inputs below 224 are refused. `ahiq_trunk.onnx`
//! emits the two taps, `ahiq_head.onnx` takes `(vit_d, vit_r, cnn_d, cnn_r)`
//! → score
//! on a trunk engine, and `ahiq_full.onnx` is the whole pair graph on
//! ort-cuda. Only the nine crop windows are ever read — the input is never
//! materialised as a whole-image f32 plane.

use std::borrow::Cow;
use std::path::Path;

use crate::common::read_meta;
use crate::engine::{self, AuxInput, PairGraph, Tap, Trunk};
use crate::{Engine, Error, WeightsConsent, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 2;
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

/// `crops()` cannot serve below the ViT's fixed crop side, whatever the
/// meta says.
fn effective_min_size(meta_min: usize) -> usize {
    meta_min.max(CROP)
}

/// The `[3, CROP, CROP]` f32 plane of the window at (top, left), in
/// `[-1, 1]` — `(v/255)·2 − 1` per pixel, bit-identical to converting the
/// whole image with `rgb8_to_planar_pm1` and slicing. Bounded at 150,528
/// floats whatever the image size: the protocol reads only these windows,
/// so the full image is never materialised.
fn crop_pm1(img: &za::Rgb8In<'_>, t: usize, l: usize) -> Vec<f32> {
    debug_assert!(t + CROP <= img.height && l + CROP <= img.width);
    let k = CROP;
    let mut out = Vec::with_capacity(3 * k * k);
    for c in 0..3usize {
        for y in t..t + k {
            let base = y * img.stride + l * 3 + c;
            for x in 0..k {
                out.push((f32::from(img.pixels[base + x * 3]) / 255.0) * 2.0 - 1.0);
            }
        }
    }
    out
}

impl Ahiq {
    /// This model's weights' published terms: the epoch-32 checkpoint
    /// (`1Nk-IpjnDNXbWacoh3T69wkSYhYYWip2W.pth` from upstream's Google Drive
    /// link) states no licence, so loading needs a [`WeightsConsent`]
    /// accepting these terms. (The two ImageNet backbone checkpoints —
    /// timm `vit_base_patch8_224.augreg_in21k_ft_in1k` and
    /// `resnet50.a1_in1k` — are Apache-2.0 via timm.)
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
        // crops() cannot serve below the ViT's fixed crop side, whatever
        // the meta says.
        let min_size = effective_min_size(min_size);
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
        r: &za::Rgb8In<'_>,
        d: &za::Rgb8In<'_>,
    ) -> Result<f64, Error> {
        let mut s = 0.0;
        for (t, l) in crops(r.height, r.width) {
            let rc = crop_pm1(r, t, l);
            let dc = crop_pm1(d, t, l);
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
        d: &za::Rgb8In<'_>,
    ) -> Result<f64, Error> {
        let mut s = 0.0;
        for (i, (t, l)) in crops(d.height, d.width).into_iter().enumerate() {
            let td = self.taps_of(trunk, crop_pm1(d, t, l))?;
            s += self.head_score(&td, &ref_taps[i])?;
        }
        Ok(s / NCROPS as f64)
    }

    /// The reference-side trunk taps for all nine crops (warm cache).
    fn ref_taps(&self, img: &za::Rgb8In<'_>) -> Result<Vec<Vec<Tap>>, Error> {
        let Backend::Trunk { trunk, .. } = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        crops(img.height, img.width)
            .into_iter()
            .map(|(t, l)| self.taps_of(&**trunk, crop_pm1(img, t, l)))
            .collect()
    }

    /// The nine crop windows of an image, converted once — the whole-graph
    /// engine's warm state. 9 × 602 KB whatever the image size.
    fn ref_crops(img: &za::Rgb8In<'_>) -> Vec<Vec<f32>> {
        crops(img.height, img.width)
            .into_iter()
            .map(|(t, l)| crop_pm1(img, t, l))
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
            Backend::Trunk { trunk, .. } => self.score_trunk(&**trunk, r, d),
            Backend::Pair(p) => Self::score_pair_graph(&**p, r, d),
        }
    }

    /// The nine-crop ensemble score through the whole-pair graph.
    fn score_pair_graph(
        p: &dyn PairGraph,
        r: &za::Rgb8In<'_>,
        d: &za::Rgb8In<'_>,
    ) -> Result<f64, Error> {
        let mut s = 0.0;
        for (t, l) in crops(r.height, r.width) {
            s += p.score(crop_pm1(r, t, l), crop_pm1(d, t, l), CROP, CROP)?;
        }
        Ok(s / NCROPS as f64)
    }

    /// [`Self::score_pair_graph`] against cached reference crops.
    fn score_pair_graph_warm(
        p: &dyn PairGraph,
        ref_crops: &[Vec<f32>],
        d: &za::Rgb8In<'_>,
    ) -> Result<f64, Error> {
        let mut s = 0.0;
        for (i, (t, l)) in crops(d.height, d.width).into_iter().enumerate() {
            s += p.score(ref_crops[i].clone(), crop_pm1(d, t, l), CROP, CROP)?;
        }
        Ok(s / NCROPS as f64)
    }
}

impl za::Metric for Ahiq {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "IIGROUP/AHIQ epoch-32 checkpoint (vit_base_patch8_224.augreg_in21k_ft_in1k blocks 0-4 + resnet50.a1_in1k layer1 taps, deformable fusion + pixel regressor), mean of nine 224x224 crops; engine {}",
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
            Backend::Pair(_) => WarmState::Crops(Self::ref_crops(&view)),
        };
        Ok(Box::new(WarmAhiq {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}

/// A reference prepared once: its nine crop taps on a trunk engine, the nine
/// converted crop windows on a whole-graph engine (which recomputes
/// features).
struct WarmAhiq<'m> {
    metric: &'m Ahiq,
    reference: WarmState,
    /// `(width, height)`.
    size: (usize, usize),
}

enum WarmState {
    /// Nine `[vit_r, cnn_r]` tap pairs, one per crop.
    Taps(Vec<Vec<Tap>>),
    /// Nine `[3, CROP, CROP]` `[-1, 1]` windows.
    Crops(Vec<Vec<f32>>),
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
        if view.width.min(view.height) < self.metric.min_size {
            return Err(Error::TooSmall {
                min: self.metric.min_size,
            })
            .metric_err_with(COLUMN, Error::kind);
        }
        let s = match (&self.metric.backend, &self.reference) {
            (Backend::Trunk { trunk, .. }, WarmState::Taps(ref_taps)) => {
                self.metric.score_trunk_warm(&**trunk, ref_taps, &view)
            }
            (Backend::Pair(p), WarmState::Crops(ref_crops)) => {
                Ahiq::score_pair_graph_warm(&**p, ref_crops, &view)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::za::Metric as _;

    /// A trunk that echoes two tiny content-derived taps; the head stub
    /// combines them. Channels match `stub()`.
    struct StubTrunk;
    impl Trunk for StubTrunk {
        fn engine(&self) -> &'static str {
            "stub"
        }
        fn run(&self, input: Vec<f32>, h: usize, w: usize) -> Result<Vec<Tap>, Error> {
            // only single crops may ever reach the engine — a whole-image
            // input or a transposed window fails loudly here.
            assert_eq!((h, w), (CROP, CROP));
            assert_eq!(input.len(), 3 * CROP * CROP);
            let m = input.iter().sum::<f32>() / input.len() as f32;
            let f = input[0];
            Ok(vec![
                Tap {
                    c: 4,
                    h: 1,
                    w: 1,
                    data: vec![m, -m, f, 1.0],
                },
                Tap {
                    c: 2,
                    h: 1,
                    w: 1,
                    data: vec![m + 1.0, -1.0],
                },
            ])
        }
    }

    /// A head: score = vit_d·vit_r + cnn_d·cnn_r.
    struct StubHead;
    impl Trunk for StubHead {
        fn engine(&self) -> &'static str {
            "stub"
        }
        fn run(&self, _input: Vec<f32>, _h: usize, _w: usize) -> Result<Vec<Tap>, Error> {
            unreachable!()
        }
        fn run_c_aux(
            &self,
            input: Vec<f32>,
            _c: usize,
            _h: usize,
            _w: usize,
            aux: &[AuxInput],
        ) -> Result<Vec<Tap>, Error> {
            let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
            let s = dot(&input, &aux[0].data) + dot(&aux[1].data, &aux[2].data);
            Ok(vec![Tap {
                c: 1,
                h: 1,
                w: 1,
                data: vec![s],
            }])
        }
    }

    struct StubPair;
    impl PairGraph for StubPair {
        fn engine(&self) -> &'static str {
            "stub"
        }
        fn score(&self, r: Vec<f32>, d: Vec<f32>, h: usize, w: usize) -> Result<f64, Error> {
            assert_eq!((h, w), (CROP, CROP));
            assert_eq!(r.len(), 3 * CROP * CROP);
            assert_eq!(d.len(), 3 * CROP * CROP);
            let m = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
            Ok(m(&r) * m(&d))
        }
    }

    fn stub_trunk() -> Ahiq {
        Ahiq {
            backend: Backend::Trunk {
                trunk: Box::new(StubTrunk),
                head: Box::new(StubHead),
            },
            channels: [4, 2],
            min_size: CROP,
        }
    }

    fn stub_pair() -> Ahiq {
        Ahiq {
            backend: Backend::Pair(Box::new(StubPair)),
            channels: [4, 2],
            min_size: CROP,
        }
    }

    /// A deterministic packed-RGB frame with content varying in both axes.
    fn frame(w: usize, h: usize, seed: u8) -> Vec<u8> {
        (0..h * w)
            .flat_map(|i| {
                let (y, x) = (i / w, i % w);
                [
                    (x as u8).wrapping_add(seed),
                    (y as u8).wrapping_add(seed),
                    (x.wrapping_mul(y) as u8).wrapping_add(seed),
                ]
            })
            .collect()
    }

    fn view(px: &[u8], w: usize, h: usize) -> za::Rgb8In<'_> {
        za::Rgb8In {
            pixels: px,
            width: w,
            height: h,
            stride: w * 3,
        }
    }

    fn image(px: &[u8], w: usize, h: usize) -> za::Image<'_> {
        za::Image::from(
            za::zenpixels::PixelSlice::new(
                px,
                w as u32,
                h as u32,
                w * 3,
                za::zenpixels::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap(),
        )
    }

    fn score(e: &dyn za::Evaluation) -> f64 {
        e.scores().find(|(k, _)| k == COLUMN).unwrap().1
    }

    #[test]
    fn undersized_is_refused() {
        let m = stub_trunk();
        let rp = frame(300, 200, 7);
        let err = m
            .score_pair(&view(&rp, 300, 200), &view(&rp, 300, 200))
            .unwrap_err();
        assert!(matches!(err, Error::TooSmall { min: CROP }), "{err:?}");
        let mut m = stub_pair();
        let err = m
            .evaluate(
                image(&rp, 300, 200),
                image(&rp, 300, 200),
                za::EvalOpts::new(),
            )
            .err()
            .expect("200px < 224 must fail");
        assert!(err.to_string().contains("too_small"), "{err}");
    }

    #[test]
    fn crop_windows_match_full_plane() {
        // crop_pm1 must equal rgb8_to_planar_pm1 + window slice bit for bit.
        let (w, h) = (320, 240);
        let px = frame(w, h, 3);
        let img = view(&px, w, h);
        let full = crate::preprocess::rgb8_to_planar_pm1(&img);
        for (t, l) in crops(h, w) {
            let got = crop_pm1(&img, t, l);
            for c in 0..3 {
                for y in 0..CROP {
                    for x in 0..CROP {
                        let want = full[c * h * w + (t + y) * w + l + x];
                        assert_eq!(got[c * CROP * CROP + y * CROP + x], want);
                    }
                }
            }
        }
    }

    /// Non-square images must reach the engine with h/w in the right slots:
    /// the warm path used to swap them. warm == cold, both orientations,
    /// both backends, through the public `warm_reference`/`evaluate` API.
    #[test]
    fn warm_equals_cold_nonsquare() {
        for (w, h) in [(320usize, 240usize), (240, 320)] {
            let rp = frame(w, h, 11);
            let dp = frame(w, h, 29);
            for mut m in [stub_trunk(), stub_pair()] {
                let cold = score(
                    &*m.evaluate(image(&rp, w, h), image(&dp, w, h), za::EvalOpts::new())
                        .unwrap(),
                );
                let mut warm = m
                    .warm_reference(image(&rp, w, h), za::EvalOpts::new())
                    .unwrap();
                let ws = score(
                    &*warm
                        .evaluate(image(&dp, w, h), za::EvalOpts::new())
                        .unwrap(),
                );
                assert_eq!(ws.to_bits(), cold.to_bits(), "{w}x{h}");
            }
        }
    }

    /// A 16384-long-side frame in both orientations scores through bounded
    /// 224² inputs only — the stubs assert no larger buffer ever reaches
    /// the engine.
    #[test]
    fn extreme_aspect_is_bounded() {
        for (w, h) in [(16384usize, CROP), (CROP, 16384)] {
            let rp = frame(w, h, 5);
            let dp = frame(w, h, 9);
            for m in [stub_trunk(), stub_pair()] {
                let s = m.score_pair(&view(&rp, w, h), &view(&dp, w, h));
                assert!(s.is_ok(), "{w}x{h}: {s:?}");
            }
        }
    }

    /// `Rgb8In.stride` may exceed `width·3`; the same logical image in a
    /// padded buffer must feed identical crop pixels to the engine.
    #[test]
    fn padded_stride_is_honored() {
        let (w, h) = (320, 240);
        let tight = frame(w, h, 7);
        let pad = 19;
        let stride = w * 3 + pad;
        let mut padded = vec![0u8; stride * h];
        for y in 0..h {
            padded[y * stride..y * stride + w * 3]
                .copy_from_slice(&tight[y * w * 3..(y + 1) * w * 3]);
        }
        let pv = za::Rgb8In {
            pixels: &padded,
            width: w,
            height: h,
            stride,
        };
        let tv = view(&tight, w, h);
        for (t, l) in crops(h, w) {
            assert_eq!(crop_pm1(&pv, t, l), crop_pm1(&tv, t, l), "{t},{l}");
        }
        for m in [stub_trunk(), stub_pair()] {
            let a = m.score_pair(&pv, &pv).unwrap();
            let b = m.score_pair(&tv, &tv).unwrap();
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// `ahiq.meta` may declare any `min_size` (or none meaningful); the
    /// metric refuses below the fixed crop side regardless.
    #[test]
    fn meta_min_size_is_floored_at_crop() {
        let dir = std::env::temp_dir().join(format!("ahiq-meta-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let meta = dir.join("ahiq.meta");
        std::fs::write(&meta, "channels 3840,768\nmin_size 0\n").unwrap();
        let (channels, declared) = read_meta(&meta).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(channels, [3840, 768]);
        assert_eq!(effective_min_size(declared), CROP);
        assert_eq!(effective_min_size(300), 300);
        let m = Ahiq {
            backend: Backend::Trunk {
                trunk: Box::new(StubTrunk),
                head: Box::new(StubHead),
            },
            channels: [4, 2],
            min_size: effective_min_size(declared),
        };
        let rp = frame(300, 200, 7);
        let err = m.score_pair(&view(&rp, 300, 200), &view(&rp, 300, 200));
        assert!(matches!(err, Err(Error::TooSmall { min: CROP })), "{err:?}");
    }

    #[test]
    fn crop_positions_cover_the_frame() {
        for (h, w) in [(224, 224), (240, 320), (320, 240), (1000, 224)] {
            for (t, l) in crops(h, w) {
                assert!(t + CROP <= h && l + CROP <= w, "{t},{l} in {h}x{w}");
            }
        }
    }
}
