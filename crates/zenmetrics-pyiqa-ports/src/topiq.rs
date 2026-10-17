//! TOPIQ-FR (Chen, Mo, Hou, Wu, Liao, Sun, Yan, Lin, "TOPIQ: A Top-down
//! Approach from Semantics to Distortions for Image Quality Assessment",
//! IEEE TIP 2024), as pyiqa 0.1.16 runs it: `CFANet` with a ResNet50 trunk,
//! the KADID-10k checkpoint (pyiqa `topiq_fr`, AIC2026 column `TOPIQ`) or the
//! PIPAL checkpoint (pyiqa `topiq_fr-pipal`, column `TOPIQ-PIPAL`). Higher is
//! better; the identical-pair score is image-dependent (about 0.93), not 1.
//!
//! The exported graphs (`models/topiq/export.py`) are the trunk (ImageNet
//! normalisation and the five `features_only` stages; each checkpoint
//! fine-tunes its own trunk) and the head (gated local pooling, self-attention
//! per level, cross-scale attention, the score MLP). The host does what is
//! size-dependent and not expressible in rten: the adaptive-average-pool
//! matrices that bring every level to the coarsest level's size (pyiqa pools
//! only when a level is larger in both dimensions; otherwise identity), and
//! the learned positional embedding bicubic-resized to each level (torch
//! `F.interpolate(mode='bicubic', align_corners=False)`, computed in f32 as
//! torch's CPU kernel does). pyiqa's eval path for these models has no resize
//! and one crop. Every engine runs the same two graphs, so `warm_reference`
//! caches the reference taps on all of them.

use std::borrow::Cow;
use std::path::Path;

use zenmetrics_nn::{Error, WeightsConsent, WeightsLicence, WeightsTerms};

use crate::engine::{self, Graph, Tensor, read_f32, read_meta, rgb8_to_planar_01};
use crate::{Engine, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const EMB: usize = 256;
const EMB_GRID: usize = 32;
const LEVELS: usize = 5;
const CHANNELS: [usize; LEVELS] = [64, 256, 512, 1024, 2048];
/// Bound on `W·H`. The head's self-attention over the coarsest level has
/// `W·H/1024` tokens, so its 4-head score matrix is `4·(W·H/1024)²` f32:
/// about 1 GiB at this bound (8 Mi pixels).
const MAX_PIXELS: usize = 8 << 20;

/// Which TOPIQ-FR checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TopiqVariant {
    /// `cfanet_fr_kadid_res50` (pyiqa `topiq_fr`; AIC2026 `TOPIQ`).
    Kadid,
    /// `cfanet_fr_pipal_res50` (pyiqa `topiq_fr-pipal`; AIC2026 `TOPIQ-PIPAL`).
    Pipal,
}

impl TopiqVariant {
    /// Both variants.
    pub const ALL: [Self; 2] = [Self::Kadid, Self::Pipal];

    /// Score column, also the exported files' prefix.
    pub const fn column(self) -> &'static str {
        match self {
            Self::Kadid => "topiq_fr",
            Self::Pipal => "topiq_fr_pipal",
        }
    }

    /// The checkpoint's published terms: pyiqa distributes both under its
    /// non-commercial terms.
    pub const fn weights(self) -> WeightsTerms {
        WeightsTerms {
            model: self.column(),
            licence: WeightsLicence::NonCommercial,
            terms_url: "https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE",
        }
    }
}

/// TOPIQ-FR. Higher is closer. Input: packed sRGB8 (any stride).
pub struct Topiq {
    variant: TopiqVariant,
    trunk: Box<dyn Graph>,
    head: Box<dyn Graph>,
    /// `[256, 32, 32]` positional embedding.
    pos_emb: Vec<f32>,
    min_size: usize,
}

impl Topiq {
    /// Load `<col>_trunk.onnx`, `<col>_head.onnx`, `<col>_posemb.f32` and
    /// `topiq.meta` (as written by `models/topiq/export.py`) from `dir`.
    ///
    /// `consent` must accept [`TopiqVariant::weights`]; it is checked before
    /// any file is read.
    ///
    /// # Errors
    /// [`Error::WeightsNotAccepted`], [`Error::Io`] for a missing file,
    /// [`Error::BadModel`] for a malformed one, [`Error::Engine`] when the
    /// engine is not built.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        variant: TopiqVariant,
        engine: Engine,
        consent: &WeightsConsent,
    ) -> Result<Self, Error> {
        engine::accept(consent, &variant.weights())?;
        let dir = dir.as_ref();
        let col = variant.column();
        let (channels, min_size) = read_meta(&dir.join("topiq.meta"))?;
        if channels != CHANNELS {
            return Err(Error::BadModel(format!(
                "topiq.meta channels {channels:?}, expected {CHANNELS:?}"
            )));
        }
        Ok(Self {
            variant,
            trunk: engine::load(&dir.join(format!("{col}_trunk.onnx")), engine)?,
            head: engine::load(&dir.join(format!("{col}_head.onnx")), engine)?,
            pos_emb: read_f32(
                &dir.join(format!("{col}_posemb.f32")),
                EMB * EMB_GRID * EMB_GRID,
            )?,
            min_size,
        })
    }

    /// The checkpoint.
    pub fn variant(&self) -> TopiqVariant {
        self.variant
    }

    fn taps(&self, img: &za::Rgb8In<'_>) -> Result<Vec<Tensor>, Error> {
        let (w, h) = (img.width, img.height);
        if w < self.min_size || h < self.min_size {
            return Err(Error::TooSmall { min: self.min_size });
        }
        if w.saturating_mul(h) > MAX_PIXELS {
            return Err(Error::TooLarge { limit: MAX_PIXELS });
        }
        let x = Tensor::new(vec![1, 3, h, w], rgb8_to_planar_01(img));
        let taps = self.trunk.run(vec![x])?;
        let ok = taps.len() == LEVELS
            && taps
                .iter()
                .zip(CHANNELS)
                .all(|(t, c)| t.shape.len() == 4 && t.shape[0] == 1 && t.shape[1] == c);
        if !ok {
            return Err(Error::BadModel(
                "trunk must return five [1, c, h, w] taps".into(),
            ));
        }
        Ok(taps)
    }

    fn head(&self, d: Vec<Tensor>, r: Vec<Tensor>) -> Result<f64, Error> {
        let (th, tw) = (d[LEVELS - 1].shape[2], d[LEVELS - 1].shape[3]);
        let mut ph = Vec::with_capacity(LEVELS);
        let mut pw = Vec::with_capacity(LEVELS);
        let mut pos = Vec::with_capacity(LEVELS);
        for t in &d {
            let (lh, lw) = (t.shape[2], t.shape[3]);
            // pyiqa: `if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw`.
            let pooled = lh > th && lw > tw;
            let (oh, ow) = if pooled { (th, tw) } else { (lh, lw) };
            ph.push(Tensor::new(vec![oh, lh], pool_matrix(lh, oh)));
            pw.push(Tensor::new(
                vec![lw, ow],
                transpose(&pool_matrix(lw, ow), ow, lw),
            ));
            pos.push(Tensor::new(vec![oh * ow, 1, EMB], self.pos_tokens(oh, ow)));
        }
        let mut inputs = d;
        inputs.extend(r);
        inputs.extend(ph);
        inputs.extend(pw);
        inputs.extend(pos);
        let out = self.head.run(inputs)?;
        out.first()
            .and_then(|t| t.data.first())
            .map(|&s| f64::from(s))
            .ok_or_else(|| Error::Engine("empty head output".into()))
    }

    /// The embedding resized to `oh × ow`, as `[oh·ow, 256]` tokens.
    fn pos_tokens(&self, oh: usize, ow: usize) -> Vec<f32> {
        let planes = bicubic_resize(&self.pos_emb, EMB, EMB_GRID, EMB_GRID, oh, ow);
        let n = oh * ow;
        let mut out = vec![0.0f32; n * EMB];
        for c in 0..EMB {
            for (i, &v) in planes[c * n..(c + 1) * n].iter().enumerate() {
                out[i * EMB + c] = v;
            }
        }
        out
    }

    fn score_pair(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        let rt = self.taps(r)?;
        self.head(self.taps(d)?, rt)
    }
}

impl za::Metric for Topiq {
    fn id(&self) -> za::MetricId {
        let col = self.variant.column();
        za::MetricId::new(col, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "pyiqa 0.1.16 {} (CFANet, ResNet50), no resize, one crop; engine {}",
            match self.variant {
                TopiqVariant::Kadid => "topiq_fr",
                TopiqVariant::Pipal => "topiq_fr-pipal",
            },
            self.trunk.engine()
        ))
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const KADID: &[za::ScoreSpec] = &[za::ScoreSpec::new(
            TopiqVariant::Kadid.column(),
            za::ScoreKind::Similarity,
        )];
        const PIPAL: &[za::ScoreSpec] = &[za::ScoreSpec::new(
            TopiqVariant::Pipal.column(),
            za::ScoreKind::Similarity,
        )];
        Cow::Borrowed(match self.variant {
            TopiqVariant::Kadid => KADID,
            TopiqVariant::Pipal => PIPAL,
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
        let col = self.variant.column();
        opts.checkpoint()?;
        opts.refuse_map(col)?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let s = self.score_pair(&r, &d).metric_err_with(col, engine::kind)?;
        Ok(Box::new(za::ScalarEval::single(col, s)))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let view = za::rgb8_view(&r)?;
        let taps = self
            .taps(&view)
            .metric_err_with(self.variant.column(), engine::kind)?;
        Ok(Box::new(WarmTopiq {
            metric: self,
            taps,
            size: (view.width, view.height),
        }))
    }
}

/// A reference's five trunk taps, computed once.
struct WarmTopiq<'m> {
    metric: &'m Topiq,
    taps: Vec<Tensor>,
    size: (usize, usize),
}

impl za::WarmReference for WarmTopiq<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        let col = self.metric.variant.column();
        opts.checkpoint()?;
        opts.refuse_map(col)?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != self.size {
            return Err(Error::SizeMismatch).metric_err_with(col, engine::kind);
        }
        let s = self
            .metric
            .taps(&d)
            .and_then(|dt| self.metric.head(dt, self.taps.clone()))
            .metric_err_with(col, engine::kind)?;
        Ok(Box::new(za::ScalarEval::single(col, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        true
    }
}

/// `[n_out, n_in]` row-major matrix of torch `adaptive_avg_pool` windows
/// `[⌊i·n/o⌋, ⌈(i+1)·n/o⌉)`, each weight `1 / len` (f32). `n_out == n_in`
/// gives the identity.
fn pool_matrix(n_in: usize, n_out: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n_out * n_in];
    for i in 0..n_out {
        let s = i * n_in / n_out;
        let e = ((i + 1) * n_in).div_ceil(n_out);
        let v = 1.0f32 / (e - s) as f32;
        m[i * n_in + s..i * n_in + e].fill(v);
    }
    m
}

fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = m[r * cols + c];
        }
    }
    t
}

/// torch's cubic convolution coefficients for fractional offset `t`,
/// A = −0.75 (`get_cubic_upsample_coefficients`, f32).
fn cubic_coeffs(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let x2 = 1.0 - t;
    [c2(t + 1.0), c1(t), c1(x2), c2(x2 + 1.0)]
}

/// One axis of torch's non-antialiased bicubic: clamped source indices and
/// coefficients per output. `scale = in/out` and `src = scale·(i+0.5) − 0.5`
/// in f32 (`area_pixel_compute_source_index`, cubic: no clamp at 0).
fn bicubic_axis(n_in: usize, n_out: usize) -> Vec<([usize; 4], [f32; 4])> {
    let scale = n_in as f32 / n_out as f32;
    (0..n_out)
        .map(|i| {
            let src = scale * (i as f32 + 0.5) - 0.5;
            let base = src.floor();
            let t = src - base;
            let base = base as i64;
            let idx =
                std::array::from_fn(|k| (base - 1 + k as i64).clamp(0, n_in as i64 - 1) as usize);
            (idx, cubic_coeffs(t))
        })
        .collect()
}

/// Planar `[c, h, w]` → `[c, oh, ow]`, torch `F.interpolate(mode='bicubic',
/// align_corners=False)` on CPU: per output, `Σ_y wy·(Σ_x wx·src)`
/// accumulated in f32 in tap order, as the generic kernel's
/// `Interpolate<2>` over `Interpolate<1>` does.
fn bicubic_resize(img: &[f32], c: usize, h: usize, w: usize, oh: usize, ow: usize) -> Vec<f32> {
    let ys = bicubic_axis(h, oh);
    let xs = bicubic_axis(w, ow);
    let mut out = vec![0.0f32; c * oh * ow];
    for ch in 0..c {
        let plane = &img[ch * h * w..(ch + 1) * h * w];
        for (oy, (yi, yw)) in ys.iter().enumerate() {
            for (ox, (xi, xw)) in xs.iter().enumerate() {
                let row = |r: usize| {
                    let s = &plane[r * w..(r + 1) * w];
                    let mut acc = s[xi[0]] * xw[0];
                    for k in 1..4 {
                        acc += s[xi[k]] * xw[k];
                    }
                    acc
                };
                let mut acc = row(yi[0]) * yw[0];
                for k in 1..4 {
                    acc += row(yi[k]) * yw[k];
                }
                out[ch * oh * ow + oy * ow + ox] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_matrix_rows_sum_to_one_and_cover_windows() {
        for (n, o) in [(472, 30), (30, 30), (7, 3), (5, 4)] {
            let m = pool_matrix(n, o);
            for i in 0..o {
                let row = &m[i * n..(i + 1) * n];
                let s: f32 = row.iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "{n}->{o} row {i}: {s}");
                let first = row.iter().position(|&v| v != 0.0).unwrap();
                assert_eq!(first, i * n / o);
            }
        }
        let id = pool_matrix(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(id[i * 4 + j], if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn bicubic_same_size_is_identity() {
        let img: Vec<f32> = (0..2 * 5 * 6).map(|i| (i as f32).sin()).collect();
        assert_eq!(bicubic_resize(&img, 2, 5, 6, 5, 6), img);
    }

    #[test]
    fn refuses_without_consent_before_reading() {
        for v in TopiqVariant::ALL {
            let err = match Topiq::from_dir(
                "/nonexistent-topiq",
                v,
                Engine::Rten,
                &WeightsConsent::new(),
            ) {
                Err(e) => e,
                Ok(_) => panic!("loaded without consent"),
            };
            assert!(matches!(err, Error::WeightsNotAccepted(t) if t == v.weights()));
        }
    }
}
