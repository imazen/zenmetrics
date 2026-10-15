//! A-FINE (Du et al., "A-FINE: Adaptive Fidelity-Naturalness Evaluation for
//! AI-Generated Image Quality Assessment", TMM 2024; upstream
//! `ChrisDud0257/AFINE`, Apache-2.0). Parity target: upstream `afine.py`'s
//! `afine_all` output — the raw adapter score, which is what the AIC-4
//! `A-FINE` column publishes (range ≈ −2..2, **lower is better**).
//!
//! Pipeline (upstream QuickInference/afine.py + utils/network_archs.py):
//!   1. CLIP-normalize the 0..1 sRGB image with the CLIP ViT stats.
//!   2. `F.interpolate` bicubic-upsample (a = −0.75, no antialias, index
//!      clamp) to the next multiple of 32 per axis — ViT-B/32 patch grid.
//!   3. The finetuned CLIP visual trunk yields 12 token maps `[1, N, 768]`.
//!   4. `AFINEQhead`: per feature (de-normalized pixels + relu(tap)) take
//!      token mean/variance, project, MLP → naturalness scalar.
//!   5. `AFINEDhead`: SSIM-style mean/var/cov terms between ref and dist
//!      statistics, channel-weighted → fidelity scalar.
//!   6. NLM logistic scalers + `AFINELearnLambda` adapter → `afine_all`.
//!
//! `models/afine/export.py` emits `afine_trunk.onnx` (the network of step 3:
//! normalized image → 12 taps `[1,768,N,1]`), `afine_full.onnx` (the whole
//! pipeline for whole-graph engines) and `afine_heads.f32` (the step 4-6
//! weights, layout documented in the export script). The heads run in Rust:
//! they are per-channel statistics plus two small linears, so on a trunk
//! engine they need only each tap's channel moments — computed on the host
//! where the data already is — rather than a multi-input head graph.

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_f32, read_meta};
use crate::engine::{self, Tap};
use crate::preprocess::rgb8_to_planar_01;
use crate::resize::resize_bicubic;
use crate::{Engine, Error, WeightsConsent, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const COLUMN: &str = "afine";

/// CLIP normalization stats (upstream `afine.py`).
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// ViT-B/32: one token per 32×32 patch, 768 channels, 12 tapped layers.
const TAPS: usize = 12;
const TAP_C: usize = 768;
const HIDDEN: usize = 128;
/// proj_head input: pixel mean‖var (3·2) + 12 projected features.
const HEAD_IN: usize = 3 * 2 + HIDDEN * TAPS; // 1542
/// proj_head hidden width (`hidden_dim * 6`).
const HEAD_HIDDEN: usize = 768;
/// dhead channel weights: 3 pixel + 12·768 feature channels.
const DHEAD_CH: usize = 3 + TAPS * TAP_C; // 9219
/// Positional embedding: 768 channels × 50 base positions, resized to
/// `N+1` positions on the host (rten's `Resize` has no cubic mode).
const POS_EMB: usize = TAP_C * 50;

/// Expected `afine_heads.f32` element count — layout documented in
/// `models/afine/export.py`.
const HEADS_LEN: usize = HIDDEN * 2 * TAP_C // proj_feat 128×1536
    + HIDDEN
    + HEAD_HIDDEN * HEAD_IN
    + HEAD_HIDDEN
    + HEAD_HIDDEN // proj_head.2 row
    + 1
    + 2 * DHEAD_CH
    + 2
    + 2
    + 1
    + POS_EMB;

/// Per-channel moments of one plane of one feature — the only reductions
/// the heads ever need.
#[derive(Clone, Copy)]
struct Stats {
    mean: f64,
    var: f64,
}

impl Stats {
    fn of(it: impl Iterator<Item = f64> + Clone) -> Self {
        let n = it.clone().count() as f64;
        let mean = it.clone().sum::<f64>() / n;
        let var = it.map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        Self { mean, var }
    }
}

/// Per-image statistics the heads consume.
struct ImgStats {
    /// Pixel planes, de-normalized to 0..1: `[3]`.
    pix: [Stats; 3],
    /// Trunk taps, relu'd: `12` × `[768]` per channel over `N` tokens.
    taps: Vec<Vec<Stats>>,
}

/// A-FINE. Lower is better (a distortion/naturalness score ≈ −2..2, the
/// `afine_all` tap). Input: packed sRGB8 (any stride).
pub struct Afine {
    backend: Backend,
    heads: Heads,
    min_size: usize,
}

/// The step 4-6 weights, in `afine_heads.f32` order.
struct Heads {
    /// proj_feat `[128, 1536]` row-major.
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    /// proj_head.0 `[768, 1542]` row-major.
    head0_w: Vec<f32>,
    head0_b: Vec<f32>,
    /// proj_head.2 `[1, 768]`, then bias.
    head2_w: Vec<f32>,
    head2_b: f64,
    /// softplus(alpha)/Σ · softplus(alpha‖beta) — dhead's normalized
    /// channel weights, in `chns` order: pixel block of 3, then 12×768.
    alpha_n: Vec<f64>,
    beta_n: Vec<f64>,
    /// NLM `(yita3, yita4)` for naturalness and fidelity.
    nr: [f64; 2],
    fr: [f64; 2],
    /// adapter `k` (pre-softplus).
    k: f64,
    /// CLIP visual positional embedding, seq-major `[50][768]`.
    pos_emb: Vec<f32>,
}

impl Heads {
    fn parse(w: &[f32]) -> Result<Self, Error> {
        let bad = |m: &str| Error::BadModel(format!("afine_heads.f32: {m}"));
        let mut it = w.iter();
        let mut take = |n: usize| -> Result<Vec<f32>, Error> {
            let v: Vec<f32> = it.by_ref().take(n).copied().collect();
            if v.len() != n {
                return Err(bad("truncated"));
            }
            Ok(v)
        };
        let (proj_w, proj_b) = (take(HIDDEN * 2 * TAP_C)?, take(HIDDEN)?);
        let (head0_w, head0_b) = (take(HEAD_HIDDEN * HEAD_IN)?, take(HEAD_HIDDEN)?);
        let (head2_w, hb2) = (take(HEAD_HIDDEN)?, take(1)?);
        let (alpha, beta) = (take(DHEAD_CH)?, take(DHEAD_CH)?);
        let (nr, fr, k) = (take(2)?, take(2)?, take(1)?);
        let pos_emb = take(POS_EMB)?;
        if it.next().is_some() {
            return Err(bad("trailing values"));
        }
        // upstream: w_sum = softplus(alpha).sum() + softplus(beta).sum() + ε,
        // then each block divides by w_sum.
        let w_sum = alpha
            .iter()
            .chain(&beta)
            .map(|&v| softplus(f64::from(v)))
            .sum::<f64>()
            + 1e-10;
        let alpha_n = alpha
            .iter()
            .map(|&v| softplus(f64::from(v)) / w_sum)
            .collect();
        let beta_n = beta
            .iter()
            .map(|&v| softplus(f64::from(v)) / w_sum)
            .collect();
        Ok(Self {
            proj_w,
            proj_b,
            head0_w,
            head0_b,
            head2_w,
            head2_b: f64::from(hb2[0]),
            alpha_n,
            beta_n,
            nr: [f64::from(nr[0]), f64::from(nr[1])],
            fr: [f64::from(fr[0]), f64::from(fr[1])],
            k: f64::from(k[0]),
            pos_emb,
        })
    }

    /// proj_feat on a `[2·768]` mean‖var vector → `[128]`.
    fn proj_feat(&self, mv: &[f64]) -> Vec<f64> {
        self.proj_b
            .iter()
            .zip(self.proj_w.as_chunks::<{ 2 * TAP_C }>().0)
            .map(|(&b, row)| {
                f64::from(b)
                    + row
                        .iter()
                        .zip(mv)
                        .map(|(&w, &x)| f64::from(w) * x)
                        .sum::<f64>()
            })
            .collect()
    }

    /// proj_head: `[1542]` → Linear→GELU→Linear → scalar.
    fn mlp(&self, x: &[f64]) -> f64 {
        let h = self
            .head0_b
            .iter()
            .zip(self.head0_w.as_chunks::<HEAD_IN>().0)
            .map(|(&b, row)| {
                gelu(
                    f64::from(b)
                        + row
                            .iter()
                            .zip(x)
                            .map(|(&w, &x)| f64::from(w) * x)
                            .sum::<f64>(),
                )
            })
            .collect::<Vec<f64>>();
        self.head2_b
            + h.iter()
                .zip(&self.head2_w)
                .map(|(&v, &w)| f64::from(w) * v)
                .sum::<f64>()
    }

    /// `AFINEQhead(img, feats)` — naturalness of one image.
    fn qhead(&self, s: &ImgStats) -> f64 {
        let mut x = Vec::with_capacity(HEAD_IN);
        for t in &s.pix {
            x.push(t.mean);
        }
        for t in &s.pix {
            x.push(t.var);
        }
        for tap in &s.taps {
            let mut mv = Vec::with_capacity(2 * TAP_C);
            mv.extend(tap.iter().map(|s| s.mean));
            mv.extend(tap.iter().map(|s| s.var));
            x.extend(self.proj_feat(&mv));
        }
        self.mlp(&x)
    }

    /// `AFINEDhead(d, r, fd, fr)` — fidelity. `cov` carries the per-channel
    /// cross-moment `E[x·y]` for each feature (pixels first, then 12 taps).
    fn dhead(&self, d: &ImgStats, r: &ImgStats, cov: &[Vec<f64>]) -> f64 {
        const C1: f64 = 1e-10;
        const C2: f64 = 1e-10;
        let (mut d1, mut d2) = (0.0, 0.0);
        let mut off = 0usize;
        for k in 0..=TAPS {
            let ch = if k == 0 { 3 } else { TAP_C };
            let (sx, sy, xy): (&[Stats], &[Stats], &[f64]) = if k == 0 {
                (&d.pix, &r.pix, &cov[0])
            } else {
                (&d.taps[k - 1], &r.taps[k - 1], &cov[k])
            };
            for c in 0..ch {
                let (mx, my) = (sx[c].mean, sy[c].mean);
                let s1 = (2.0 * mx * my + C1) / (mx * mx + my * my + C1);
                d1 += self.alpha_n[off + c] * s1;
                let cxy = xy[c] - mx * my;
                let s2 = (2.0 * cxy + C2) / (sx[c].var + sy[c].var + C2);
                d2 += self.beta_n[off + c] * s2;
            }
            off += ch;
        }
        1.0 - (d1 + d2)
    }
}

/// logistic NLM `(y1−y2)/(1+e^e)+y2`, y1=2, y2=−2 — the `e ≥ 10` branch
/// coincides with this expression there and beyond (4·e⁻ᵉ/(1+e⁻ᵉ) equals it
/// to fp64 precision once e ≥ ~17, and differs by <2e-4 at the switch), and
/// `models/afine/export.py` emits the branch-free equivalent for ONNX.
fn nlm(x: f64, [y3, y4]: [f64; 2]) -> f64 {
    let e = -(x - y3) / (y4.abs() + 1e-10);
    if e >= 10.0 {
        4.0 * (-e).exp() / (1.0 + (-e).exp()) - 2.0
    } else {
        4.0 / (1.0 + e.exp()) - 2.0
    }
}

/// NLM_FR_Fit_with_limit: parameter clamps, then the same logistic.
fn nlm_fr(x: f64, [y3, y4]: [f64; 2]) -> f64 {
    nlm(x, [y3.clamp(0.05, 0.95), y4.clamp(0.01, 0.70)])
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 { x } else { (1.0 + x.exp()).ln() }
}

/// `nn.GELU` default (erf form).
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// `erf(x)` via the Numerical Recipes `erfcc` Chebyshev fit (fractional
/// error < 1.2e-7; zenmetrics-stats' `special.rs` carries the same form).
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let p = -z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let erfc = t * p.exp(); // erfc(|x|)
    if x >= 0.0 { 1.0 - erfc } else { erfc - 1.0 }
}

impl Afine {
    /// This model's weights' published terms: `afine.pth` is distributed
    /// through a Google Drive link in upstream's README with no licence
    /// stated for it, but it was trained on DiffIQA data whose terms are
    /// noncommercial — derived weights inherit them, so loading needs a
    /// [`WeightsConsent`] accepting these terms. The OpenAI CLIP ViT-B/32
    /// it overlays ships a model card, not a licence — the consent covers
    /// the fine-tuned product as distributed.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::NonCommercial,
        terms_url: "https://github.com/ChrisDud0257/AFINE",
    };

    /// Load an exported model directory (`afine_trunk.onnx`,
    /// `afine_heads.f32`, `afine.meta`, as written by
    /// `models/afine/export.py`) on `engine`. The ONNX Runtime engines read
    /// `afine_full.onnx` instead of the trunk graph.
    ///
    /// `consent` must accept [`Self::WEIGHTS`] (the weights state no
    /// licence); it is checked before any file is read.
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
        let (channels, min_size) = read_meta(&dir.join("afine.meta"))?;
        if channels != [TAP_C] {
            return Err(Error::BadModel(format!(
                "afine.meta: expected channels {TAP_C}, got {channels:?}"
            )));
        }
        let heads = Heads::parse(&read_f32(&dir.join("afine_heads.f32"), HEADS_LEN)?)?;
        let backend = match engine {
            Engine::Rten => Backend::Trunk(engine::load(&dir.join("afine_trunk.onnx"), TAPS)?),
            #[cfg(feature = "ort")]
            Engine::OrtCuda => Backend::Pair(Box::new(engine::ort_engine::OrtCudaPair::load(
                &dir.join("afine_full.onnx"),
                "afine",
            )?)),
        };
        Ok(Self {
            backend,
            heads,
            min_size,
        })
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk(t) => t.engine(),
            Backend::Pair(p) => p.engine(),
        }
    }

    /// Upstream preprocessing: RGB8 → planar 0..1 → CLIP normalize →
    /// bicubic upsample to the next multiple of 32 per axis. Returns the
    /// normalized `[3, nh, nw]` tensor and the positional embedding
    /// resampled to `nh/32·nw/32 + 1` tokens (seq-major `[N+1, 768]`,
    /// the trunk graph's `pe` input — the resample happens here because
    /// rten's `Resize` op cannot do bicubic).
    fn prep(&self, img: &za::Rgb8In<'_>) -> (Vec<f32>, usize, usize, Vec<f32>) {
        let (h, w) = (img.height, img.width);
        let (nh, nw) = (h + (32 - h % 32) % 32, w + (32 - w % 32) % 32);
        let mut planar = rgb8_to_planar_01(img);
        for ch in 0..3 {
            for v in &mut planar[ch * h * w..(ch + 1) * h * w] {
                *v = (*v - MEAN[ch]) / STD[ch];
            }
        }
        let norm = resize_bicubic(&planar, 3, h, w, nh, nw);
        let n_tok = (nh / 32) * (nw / 32) + 1;
        let pe = resize_bicubic(&self.heads.pos_emb, 1, 50, TAP_C, n_tok, TAP_C);
        (norm, nh, nw, pe)
    }

    /// The trunk graph's `pe` aux input for a prepped `(nh, nw)` image.
    fn pe_aux(pe: Vec<f32>, nh: usize, nw: usize) -> engine::AuxInput {
        engine::AuxInput {
            index: 1,
            shape: [1, 1, (nh / 32) * (nw / 32) + 1, TAP_C],
            data: pe,
        }
    }

    /// Pixel-plane stats of a normalized `[3, h, w]` tensor, de-normalized
    /// (as the heads' `x * std + mean` does).
    fn pix_stats(norm: &[f32], plane: usize) -> [Stats; 3] {
        let mut out = [Stats {
            mean: 0.0,
            var: 0.0,
        }; 3];
        for (c, s) in out.iter_mut().enumerate() {
            let (m, sd) = (f64::from(MEAN[c]), f64::from(STD[c]));
            *s = Stats::of(
                norm[c * plane..(c + 1) * plane]
                    .iter()
                    .map(|&v| f64::from(v) * sd + m),
            );
        }
        out
    }

    /// Per-channel `E[x·y]` of two normalized planes, de-normalized.
    fn pix_cov(x: &[f32], y: &[f32], plane: usize) -> Vec<f64> {
        (0..3)
            .map(|c| {
                let (m, sd) = (f64::from(MEAN[c]), f64::from(STD[c]));
                x[c * plane..(c + 1) * plane]
                    .iter()
                    .zip(&y[c * plane..(c + 1) * plane])
                    .map(|(&a, &b)| (f64::from(a) * sd + m) * (f64::from(b) * sd + m))
                    .sum::<f64>()
                    / plane as f64
            })
            .collect()
    }

    /// Per-channel stats of one tap, relu'd — `[768]` over `N` tokens.
    fn tap_stats(t: &Tap) -> Vec<Stats> {
        let n = t.plane();
        (0..t.c)
            .map(|c| {
                Stats::of(
                    t.data[c * n..(c + 1) * n]
                        .iter()
                        .map(|&v| f64::from(v.max(0.0))),
                )
            })
            .collect()
    }

    /// Per-channel `E[relu(x)·relu(y)]` over a tap's tokens.
    fn tap_cov(x: &Tap, y: &Tap) -> Vec<f64> {
        let n = x.plane();
        (0..x.c)
            .map(|c| {
                x.data[c * n..(c + 1) * n]
                    .iter()
                    .zip(&y.data[c * n..(c + 1) * n])
                    .map(|(&a, &b)| f64::from(a.max(0.0)) * f64::from(b.max(0.0)))
                    .sum::<f64>()
                    / n as f64
            })
            .collect()
    }

    /// Feature taps of a prepped (normalized) image on the trunk engine.
    fn taps(&self, norm: Vec<f32>, h: usize, w: usize, pe: Vec<f32>) -> Result<Vec<Tap>, Error> {
        let Backend::Trunk(t) = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        let taps = t.run_aux(norm, h, w, &[Self::pe_aux(pe, h, w)])?;
        Self::check_taps(&taps)?;
        Ok(taps)
    }

    fn check_taps(taps: &[Tap]) -> Result<(), Error> {
        if taps.len() != TAPS {
            return Err(Error::BadModel(format!(
                "{} taps, expected {TAPS}",
                taps.len()
            )));
        }
        for t in taps {
            if t.c != TAP_C || t.w != 1 || t.plane() == 0 || t.data.len() != t.c * t.plane() {
                return Err(Error::BadModel(format!(
                    "tap shape [{}, {}, {}], expected [{TAP_C}, N, 1]",
                    t.c, t.h, t.w
                )));
            }
        }
        Ok(())
    }

    /// `ImgStats` of a prepped image: pixel stats + per-tap relu stats.
    fn img_stats(norm: &[f32], plane: usize, taps: &[Tap]) -> ImgStats {
        ImgStats {
            pix: Self::pix_stats(norm, plane),
            taps: taps.iter().map(Self::tap_stats).collect(),
        }
    }

    /// Steps 4-6 on a prepped pair.
    fn score_stats(&self, d: &ImgStats, r: &ImgStats, cov: &[Vec<f64>]) -> f64 {
        let n_d = self.heads.qhead(d);
        let n_r = self.heads.qhead(r);
        let fid = self.heads.dhead(d, r, cov);
        let (s_d, s_r, s_f) = (
            nlm(n_d, self.heads.nr),
            nlm(n_r, self.heads.nr),
            nlm_fr(fid, self.heads.fr),
        );
        // AFINELearnLambda: u = exp(softplus(k)·(n_r − n_d))·n_d + f.
        (softplus(self.heads.k) * (s_r - s_d)).exp() * s_d + s_f
    }

    /// Pixel+tap cross-moments, `[13][ch]` — `cov[0]` pixels, then taps.
    fn cross_covs(
        d_norm: &[f32],
        r_norm: &[f32],
        plane: usize,
        f_d: &[Tap],
        f_r: &[Tap],
    ) -> Vec<Vec<f64>> {
        let mut v = Vec::with_capacity(TAPS + 1);
        v.push(Self::pix_cov(d_norm, r_norm, plane));
        for (x, y) in f_d.iter().zip(f_r) {
            v.push(Self::tap_cov(x, y));
        }
        v
    }

    fn score_pair(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        let (rn, rh, rw, rpe) = self.prep(r);
        let (dn, dh, dw, dpe) = self.prep(d);
        match &self.backend {
            Backend::Trunk(t) => {
                let (ft_r, ft_d) = t.run_pair_aux(
                    rn.clone(),
                    dn.clone(),
                    rh,
                    rw,
                    &[Self::pe_aux(rpe, rh, rw)],
                    &[Self::pe_aux(dpe, dh, dw)],
                )?;
                Self::check_taps(&ft_r)?;
                Self::check_taps(&ft_d)?;
                let plane = rh * rw;
                let (s_r, s_d) = (
                    Self::img_stats(&rn, plane, &ft_r),
                    Self::img_stats(&dn, plane, &ft_d),
                );
                let cov = Self::cross_covs(&dn, &rn, plane, &ft_d, &ft_r);
                Ok(self.score_stats(&s_d, &s_r, &cov))
            }
            // equal input sizes ⇒ equal padded sizes; the dims here apply
            // to both graph inputs.
            Backend::Pair(p) => p.score(rn, dn, dh, dw),
        }
    }
}

impl za::Metric for Afine {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "ChrisDud0257/AFINE afine_all (raw adapter output), CLIP-norm + bicubic upsample to mult-32; engine {}",
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
        if r.width.min(r.height) < self.min_size || d.width.min(d.height) < self.min_size {
            return Err(Error::TooSmall { min: self.min_size })
                .metric_err_with(COLUMN, Error::kind);
        }
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
        let (rn, rh, rw, rpe) = self.prep(&view);
        let reference = match &self.backend {
            Backend::Trunk(_) => {
                let taps = self
                    .taps(rn.clone(), rh, rw, rpe)
                    .metric_err_with(COLUMN, Error::kind)?;
                WarmState::Trunk {
                    stats: Self::img_stats(&rn, rh * rw, &taps),
                    norm: rn,
                    taps,
                }
            }
            Backend::Pair(_) => WarmState::Pixels(rn),
        };
        Ok(Box::new(WarmAfine {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}

/// A reference prepared once: its stats and taps on a trunk engine, the
/// prepped input on a whole-graph engine (which recomputes its features).
struct WarmAfine<'m> {
    metric: &'m Afine,
    reference: WarmState,
    size: (usize, usize),
}

enum WarmState {
    /// Reference tap tensors (needed for the covariances) + their stats and
    /// the normalized pixels (for the pixel covariance).
    Trunk {
        stats: ImgStats,
        norm: Vec<f32>,
        taps: Vec<Tap>,
    },
    Pixels(Vec<f32>),
}

impl za::WarmReference for WarmAfine<'_> {
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
        if d.width.min(d.height) < self.metric.min_size {
            return Err(Error::TooSmall {
                min: self.metric.min_size,
            })
            .metric_err_with(COLUMN, Error::kind);
        }
        let (dn, dh, dw, dpe) = self.metric.prep(&d);
        let s = match &self.reference {
            WarmState::Trunk { stats, norm, taps } => {
                let ft_d = self
                    .metric
                    .taps(dn.clone(), dh, dw, dpe)
                    .metric_err_with(COLUMN, Error::kind)?;
                let plane = dh * dw;
                let s_d = Afine::img_stats(&dn, plane, &ft_d);
                let cov = Afine::cross_covs(&dn, norm, plane, &ft_d, taps);
                self.metric.score_stats(&s_d, stats, &cov)
            }
            WarmState::Pixels(px) => match &self.metric.backend {
                Backend::Pair(p) => p
                    .score(px.clone(), dn, dh, dw)
                    .metric_err_with(COLUMN, Error::kind)?,
                Backend::Trunk(_) => {
                    return Err(Error::Engine("warm state does not match engine".into()))
                        .metric_err_with(COLUMN, Error::kind);
                }
            },
        };
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        matches!(self.reference, WarmState::Trunk { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_gelu_sign_and_values() {
        // torch.special.erf / F.gelu references (fp64).
        let erf_cases = [
            (-2.0, -0.9953222650189527),
            (-0.5, -0.5204998778130465),
            (0.0, 0.0),
            (0.5, 0.5204998778130465),
            (2.0, 0.9953222650189527),
            (6.0, 1.0),
        ];
        for (x, want) in erf_cases {
            assert!((erf(x) - want).abs() <= 1e-7, "erf({x}) = {}", erf(x));
        }
        let gelu_cases = [
            (-0.9704703041139714, -0.16100690451172137),
            (-1.0, -0.15865525693163705),
            (-0.40589515, -0.13898247432738543),
            (0.0, 0.0),
            (0.5, 0.34573123062451056),
            (1.0, 0.8413447460685429),
            (4.0, 3.99987316105496),
        ];
        for (x, want) in gelu_cases {
            assert!((gelu(x) - want).abs() <= 1e-7, "gelu({x}) = {}", gelu(x));
        }
    }

    #[test]
    fn softplus_edges() {
        assert_eq!(softplus(0.0), std::f64::consts::LN_2);
        assert!((softplus(5.0) - 5.006715348489118).abs() <= 1e-12);
        assert_eq!(softplus(50.0), 50.0);
    }
}
