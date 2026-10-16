//! PaQ-2-PiQ (Y. Ying, H. Niu, P. Gupta, D. Mahajan, D. Ghadiyaram,
//! A. Bovik, "From Patches to Pictures (PaQ-2-PiQ): Mapping the
//! Perceptual Space of Picture Quality", CVPR 2020; upstream
//! `baidut/paq2piq`, MIT code — **the released checkpoint is CC-BY-NC
//! 4.0**: `WeightsLicence::NonCommercial`, consent-gated). Parity
//! target: upstream `InferenceModel.predict` on CPU fp32,
//! `RoIPoolModel-fit.10.bs.120.pth`.
//!
//! The eval transform is upstream's `ToTensor` (/255) at native
//! resolution — no resize, no ImageNet normalisation. The model is
//! ResNet-18 (cut before the classifier) feeding `RoIPool((2,2),
//! spatial_scale=1/32)` over 401 ROIs — the whole image plus a 20×20
//! `linspace` block grid — then the head (`AdaptiveConcatPool`,
//! BatchNorm–Linear–ReLU–BatchNorm–Linear). The score is `t[0]`, the
//! whole-image ROI's prediction; the 400 block scores (the quality
//! map) are discarded.
//!
//! `models/paq2piq/export.py` loads upstream's `paq2piq.model
//! .RoIPoolModel` verbatim and exports two graphs: `paq2piq_body.onnx`
//! `x[1,3,h,w] -> feats[1,512,hf,wf]` and `paq2piq_head.onnx`
//! `pooled[401,512,2,2] -> score[1]`. The torchvision RoIPool between
//! them runs host-side in [`roi_pool`] — rten has no `MaxRoiPool`
//! operator — with upstream's exact quantisation: `std::round` (half
//! away) on `roi·1/32`, `+1` sizes, floor/ceil bins, clamped, empty
//! bins 0.

use std::borrow::Cow;
use std::path::Path;

use crate::common::read_meta;
use crate::engine::{self, ImageGraph, NamedInput};
use crate::par;
use crate::preprocess::rgb8_to_planar_01;
use crate::{Engine, Error, WeightsConsent, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const COLUMN: &str = "paq2piq";
/// Block grid: 1 whole-image ROI + 20×20 blocks = 401 RoIPool inputs.
const BLOCKS: usize = 20;
const N_ROIS: usize = 1 + BLOCKS * BLOCKS;
/// resnet18 children[:-2] output channels.
const C: usize = 512;
/// RoIPool's `spatial_scale`.
const SCALE: f32 = 1.0 / 32.0;
/// Native-resolution tensor bound, applied before any image allocation.
const MAX_SIDE: usize = 4096;

/// PaQ-2-PiQ, authors' release checkpoint. Higher is better —
/// upstream's ~0–100 MOS-scale score. Input: packed sRGB8 (any
/// stride); width and height must be in 1..=4096. The fully convolutional
/// graph runs at native resolution.
pub struct Paq2piq {
    body: Box<dyn ImageGraph>,
    head: Box<dyn ImageGraph>,
    min_size: usize,
}

impl Paq2piq {
    /// This model's weights' published terms: the release lives in
    /// `baidut/PaQ-2-PiQ`, whose `CC-BY-NC-4.0.txt` and README badge
    /// (CC BY-NC-SA 4.0) are noncommercial. Consent is required to
    /// load.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::NonCommercial,
        terms_url: "https://github.com/baidut/PaQ-2-PiQ/blob/master/CC-BY-NC-4.0.txt",
    };

    /// Load an exported model directory (`paq2piq_body.onnx`,
    /// `paq2piq_head.onnx`, `paq2piq.meta`, as written by
    /// `models/paq2piq/export.py`) on `engine`.
    ///
    /// `consent` must accept [`Self::WEIGHTS`] (CC-BY-NC weights).
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`], plus `consent` does not accept
    /// [`Self::WEIGHTS`].
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        consent: &WeightsConsent,
    ) -> Result<Self, Error> {
        consent.check(&Self::WEIGHTS)?;
        let dir = dir.as_ref();
        let (_channels, min_size) = read_meta(&dir.join("paq2piq.meta"))?;
        let body = dir.join("paq2piq_body.onnx");
        let head = dir.join("paq2piq_head.onnx");
        let (body, head): (Box<dyn ImageGraph>, Box<dyn ImageGraph>) = match engine {
            Engine::Rten => (engine::load_image(&body)?, engine::load_image(&head)?),
            #[cfg(feature = "ort")]
            Engine::OrtCuda => (
                Box::new(engine::ort_engine::OrtCudaImage::load(&body, "feats")?),
                Box::new(engine::ort_engine::OrtCudaImage::load(&head, "score")?),
            ),
        };
        Ok(Self {
            body,
            head,
            min_size,
        })
    }

    /// Upstream `get_blockwise_rois(20,20)`: the whole image first,
    /// then row-major `[x1, y1, x2, y2]` over a `linspace(0,dim,21)`
    /// grid, in input-pixel coordinates (RoIPool's `spatial_scale`
    /// maps them into feature space).
    fn rois(h: usize, w: usize) -> [[f32; 4]; N_ROIS] {
        let mut out = [[0.0; 4]; N_ROIS];
        out[0] = [0.0, 0.0, w as f32, h as f32];
        // linspace: f64 division then the f32 cast upstream's
        // `torch.tensor(a).float()` applies.
        let f = |i: usize, dim: usize| (i as f64 * dim as f64 / BLOCKS as f64) as f32;
        for n in 0..BLOCKS {
            for m in 0..BLOCKS {
                out[1 + n * BLOCKS + m] = [f(m, w), f(n, h), f(m + 1, w), f(n + 1, h)];
            }
        }
        out
    }

    /// `ToTensor` only: `x [1,3,h,w]` = planar u8 /255 (upstream
    /// applies no resize and no ImageNet normalisation).
    fn prep(img: &za::Rgb8In<'_>, min_size: usize) -> Result<Vec<f32>, Error> {
        let (h, w) = (img.height, img.width);
        if h < min_size || w < min_size {
            return Err(Error::TooSmall { min: min_size });
        }
        if h > MAX_SIDE || w > MAX_SIDE {
            return Err(Error::TooLarge { limit: MAX_SIDE });
        }
        Ok(rgb8_to_planar_01(img))
    }

    /// torchvision `RoIPool((2,2), 1/32)` on `feats [1,C,hf,wf]` →
    /// `pooled [N,C,2,2]` — same quantisation as the upstream kernel:
    /// `std::round` (half-away) ROI bounds scaled by 1/32, ROI size
    /// `max(end−start+1, 1)`, floor/ceil bin edges clamped to the map,
    /// empty bins 0.
    ///
    /// Each ROI's four bins read only `feats` and write only its own `C·4`
    /// slice of the output, so the ROIs run on scoped worker threads — every
    /// output element keeps the original scan order (its max over the same
    /// cells), so the result is unchanged.
    fn roi_pool(feats: &[f32], hf: usize, wf: usize, rois: &[[f32; 4]]) -> Vec<f32> {
        let mut out = vec![0.0f32; rois.len() * C * 4];
        // One max over a feature cell per bin per channel is the work that
        // decides the thread count.
        let work: usize = rois
            .iter()
            .map(|&r| {
                Self::roi_bins(hf, wf, r)
                    .iter()
                    .filter(|(h0, h1, w0, w1)| h1 > h0 && w1 > w0)
                    .map(|(h0, h1, w0, w1)| ((h1 - h0) * (w1 - w0)) as usize)
                    .sum::<usize>()
            })
            .sum::<usize>()
            * C;
        let threads = par::threads_for(work);
        par::for_each_row(&mut out, C * 4, threads, |i, orow| {
            for (bin, &(h0, h1, w0, w1)) in Self::roi_bins(hf, wf, rois[i]).iter().enumerate() {
                if h1 > h0 && w1 > w0 {
                    let (h0, h1, w0, w1) = (h0 as usize, h1 as usize, w0 as usize, w1 as usize);
                    for c in 0..C {
                        let mut m = f32::NEG_INFINITY;
                        for y in h0..h1 {
                            for x in w0..w1 {
                                m = m.max(feats[(c * hf + y) * wf + x]);
                            }
                        }
                        orow[c * 4 + bin] = m;
                    }
                }
            }
        });
        out
    }

    /// The four `RoIPool` bins of one ROI as clamped `(h0, h1, w0, w1)`
    /// half-open feature-map ranges (an empty bin has `h1 == h0` or
    /// `w1 == w0`).
    fn roi_bins(hf: usize, wf: usize, roi: [f32; 4]) -> [(i64, i64, i64, i64); 4] {
        let [x1, y1, x2, y2] = roi;
        let (xs, ys) = ((x1 * SCALE).round() as i64, (y1 * SCALE).round() as i64);
        let rw = ((x2 * SCALE).round() as i64 - xs + 1).max(1);
        let rh = ((y2 * SCALE).round() as i64 - ys + 1).max(1);
        let mut bins = [(0i64, 0i64, 0i64, 0i64); 4];
        for ph in 0..2i64 {
            for pw in 0..2i64 {
                let h0 = ((ph as f64 * rh as f64 / 2.0).floor() as i64 + ys).clamp(0, hf as i64);
                let h1 =
                    (((ph + 1) as f64 * rh as f64 / 2.0).ceil() as i64 + ys).clamp(0, hf as i64);
                let w0 = ((pw as f64 * rw as f64 / 2.0).floor() as i64 + xs).clamp(0, wf as i64);
                let w1 =
                    (((pw + 1) as f64 * rw as f64 / 2.0).ceil() as i64 + xs).clamp(0, wf as i64);
                bins[(ph * 2 + pw) as usize] = (h0, h1, w0, w1);
            }
        }
        bins
    }

    /// Score one prepared image: body → host RoIPool → head → `t[0]`.
    fn score_image(&self, img: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let x = Self::prep(img, self.min_size)?;
        let (fshape, feats) = self.body.tensor(&[NamedInput {
            name: "x",
            shape: vec![1, 3, img.height, img.width],
            data: x,
        }])?;
        let &[1, c, hf, wf] = fshape.as_slice() else {
            return Err(Error::BadModel(format!(
                "paq2piq feats: want [1,C,hf,wf], got {fshape:?}"
            )));
        };
        if c != C {
            return Err(Error::BadModel(format!(
                "paq2piq feats channels {c} != {C}"
            )));
        }
        let rois = Self::rois(img.height, img.width);
        let pooled = Self::roi_pool(&feats, hf, wf, &rois);
        self.head.score(&[NamedInput {
            name: "pooled",
            shape: vec![N_ROIS, C, 2, 2],
            data: pooled,
        }])
    }
}

impl za::NoReferenceMetric for Paq2piq {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "baidut/paq2piq RoIPoolModel-fit.10.bs.120, native res + 1+20x20 RoIPool grid; engine {}",
            self.body.engine()
        ))
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new(COLUMN, za::ScoreKind::Quality)];
        Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate_image(
        &mut self,
        image: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let img = za::rgb8_view(&image)?;
        let s = self
            .score_image(&img)
            .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(px: &'a [u8], w: usize, h: usize) -> za::Rgb8In<'a> {
        za::Rgb8In {
            pixels: px,
            width: w,
            height: h,
            stride: w * 3,
        }
    }

    /// Refuse empty and oversized images before touching their pixel data.
    #[test]
    fn prep_size_bounds() {
        for (w, h) in [(0, 5), (5, 0)] {
            assert!(matches!(
                Paq2piq::prep(&view(&[], w, h), 1),
                Err(Error::TooSmall { .. })
            ));
        }
        assert!(Paq2piq::prep(&view(&[0u8; 3], 1, 1), 1).is_ok());
        for (w, h) in [(4097, 1), (1, 4097), (16384, 16384), (usize::MAX / 3, 1)] {
            assert!(matches!(
                Paq2piq::prep(&view(&[], w, h), 1),
                Err(Error::TooLarge { limit: 4096 })
            ));
        }
    }

    #[test]
    fn refuses_without_weights_consent() {
        let err = Paq2piq::from_dir(
            "/nonexistent-zenmetrics-paq2piq",
            Engine::Rten,
            &WeightsConsent::new(),
        )
        .err()
        .expect("must refuse");
        assert!(matches!(err, Error::WeightsNotAccepted(_)), "{err}");
    }

    /// ToTensor is a plain /255, no normalisation, native size.
    #[test]
    fn prep_is_div255() {
        let px = vec![128u8; 4 * 4 * 3];
        let x = Paq2piq::prep(&view(&px, 4, 4), 1).unwrap();
        assert_eq!(x.len(), 3 * 4 * 4);
        assert!((x[0] - 128.0 / 255.0).abs() < 1e-7, "{}", x[0]);
    }

    /// The ROI grid: ROI 0 is the whole image; ROI 1 is the top-left
    /// block; ROIs are `[x1, y1, x2, y2]` in input pixels.
    #[test]
    fn rois_grid() {
        let r = Paq2piq::rois(40, 60);
        assert_eq!(r[0], [0.0, 0.0, 60.0, 40.0]);
        assert_eq!(r[1], [0.0, 0.0, 3.0, 2.0]);
        assert_eq!(r[20], [57.0, 0.0, 60.0, 2.0]);
    }

    /// The torchvision quantization on hand-computed cases. A 64×64
    /// image → feat map 2×2; the whole-image ROI rounds to
    /// `round(64/32)=2`, and the `+1` in `roi_width` makes `rw=rh=3`,
    /// so the 2×2 bins overlap: `[0,2)` and `[1,3)` (clamped to the
    /// map) — every tap sees the max of the overlapping region = 4.
    #[test]
    fn roi_pool_quantization() {
        let mut f = vec![0.0f32; C * 2 * 2];
        f[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let out = Paq2piq::roi_pool(&f, 2, 2, &[[0.0, 0.0, 64.0, 64.0]]);
        assert_eq!(&out[0..4], &[4.0, 4.0, 4.0, 4.0]);
        // 32x32 -> round(32/32)=1 -> roi 1x1 on a 1x1 map: only the
        // first bin is non-empty ([0,1)x[0,1)); the rest clamp to
        // empty -> 0.
        let f1 = {
            let mut v = vec![0.0f32; C];
            v[0] = 7.0;
            v
        };
        let out = Paq2piq::roi_pool(&f1, 1, 1, &[[0.0, 0.0, 32.0, 32.0]]);
        assert_eq!(&out[0..4], &[7.0, 0.0, 0.0, 0.0]);
        // Half-away rounding: roi x2 = 16 -> round(0.5) = 1 (not 0) —
        // same 1x1 roi, first bin only.
        let out2 = Paq2piq::roi_pool(&f1, 1, 1, &[[0.0, 0.0, 16.0, 16.0]]);
        assert_eq!(&out2[0..4], &[7.0, 0.0, 0.0, 0.0]);
        // A zero-area roi in the middle: all bins empty -> 0s.
        let f0 = vec![0.0f32; C * 4];
        let out3 = Paq2piq::roi_pool(&f0, 2, 2, &[[64.0, 64.0, 64.0, 64.0]]);
        assert_eq!(&out3[0..4], &[0.0, 0.0, 0.0, 0.0]);
    }

    /// The ROI-parallel pool is bit-identical to the sequential path and a
    /// one-worker cap launches no helper thread.
    #[test]
    fn roi_pool_thread_bit_identical() {
        let (hf, wf) = (64, 64);
        let feats: Vec<f32> = (0..C * hf * wf)
            .map(|i| (i as f32 * 0.371).sin() * 5.0)
            .collect();
        let rois = Paq2piq::rois(2048, 2048);
        let (a, spawns) =
            par::observe_workers(Some(1), || Paq2piq::roi_pool(&feats, hf, wf, &rois));
        assert_eq!(spawns, 0);
        let (b, parallel_spawns) =
            par::observe_workers(Some(16), || Paq2piq::roi_pool(&feats, hf, wf, &rois));
        assert!(parallel_spawns > 0);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// Strided input rows are honoured.
    #[test]
    fn prep_strided_rows() {
        let (w, h, stride) = (4, 4, 4 * 3 + 4);
        let mut px = vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                px[y * stride + x * 3] = 200;
            }
        }
        let img = za::Rgb8In {
            pixels: &px,
            width: w,
            height: h,
            stride,
        };
        let x = Paq2piq::prep(&img, 1).unwrap();
        assert!((x[0] - 200.0 / 255.0).abs() < 1e-7, "{}", x[0]);
    }
}
