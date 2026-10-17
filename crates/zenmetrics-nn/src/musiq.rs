//! MUSIQ (J. Ke, Q. Wang, Y. Wang, P. Milanfar, F. Yang, "MUSIQ:
//! Multi-scale Image Quality Transformer", ICCV 2021; upstream
//! `google-research/google-research/musiq` at `3ef5211`, Apache-2.0,
//! official KonIQ checkpoint `koniq_ckpt.npz` from the project's GCS
//! bucket). Parity target: upstream's `run_predict_image.py` on the
//! legacy JAX/Flax stack (jax 0.3.25 + flax 0.3.3, CPU fp32).
//!
//! The eval transform is upstream's `get_multiscale_patches`: the sRGB8
//! image is normalised `[0,255] → [−1,1]` (`(x/255)·2 − 1`), then
//! aspect-ratio-resized to longer side 224 and 384 with TensorFlow's
//! GAUSSIAN resampler (`scale_and_translate`, σ = 0.5, radius 1.5,
//! `antialias = false` — [`resize_gaussian_tf_f32`]). Each scale and the
//! original-resolution image are cut into 32×32 stride-32 patches
//! (zero-padded SAME); every patch row is `[pixels (h,w,c), spatial hash,
//! scale id, mask]`, the two rescaled sequences are zero-padded to
//! `⌈longer/32⌉²` rows, and all three concatenated. The graph consumes
//! `[1, l, 3075]` and emits one MOS-scale score — higher is better.
//!
//! `models/musiq/export.py` loads `opt/target/*` from the checkpoint NPZ,
//! mirrors the 5-layer ResNet patch stem + 14-block transformer in
//! PyTorch, and exports `musiq.onnx` with a dynamic patch-axis `l`.
//! rten runs it through [`engine::load_image`], ONNX Runtime engines
//! through the named-input whole-image graph.

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_meta};
use crate::engine::{self, NamedInput};
use crate::preprocess::rgb8_to_planar_u8;
use crate::tfresize::resize_gaussian_tf_f32;
use crate::{Engine, Error, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 2;
const COLUMN: &str = "musiq";
/// Patch size/stride of `get_multiscale_patches`.
const PATCH: usize = 32;
/// `hse_grid_size`: the spatial hash grid is GRID² cells.
const GRID: usize = 10;
/// `longer_side_lengths`: aspect-ratio-preserving rescale targets.
const LONGER: [usize; 2] = [224, 384];
/// Row layout: `32·32·3` patch pixels + spatial + scale + mask.
const ROW: usize = PATCH * PATCH * 3 + 3;
/// Patch-sequence cap: upstream leaves the native-resolution group
/// unbounded, but attention over `l` rows is O(l²) — 8192 rows (≈ a
/// 2800px-square input) already allocates ~1.6 GB for `q·kᵀ` alone.
const MAX_ROWS: usize = 8192;

/// MUSIQ, KonIQ checkpoint. Higher is better — upstream's MOS-scale
/// quality score (the ladder goldens span 15.9–76.2). Input: packed sRGB8
/// (any stride). Extreme aspect ratios whose rescaled side rounds to 0
/// and inputs whose patch sequence would exceed [`MAX_ROWS`] rows are
/// refused (`TooSmall` / `TooManyPatches` / `DegenerateAspectRatio`).
pub struct Musiq {
    backend: Backend,
    min_size: usize,
}

impl Musiq {
    /// This model's weights' published terms: `koniq_ckpt.npz` is
    /// published on the project's GCS bucket and covered by the
    /// repository's Apache-2.0 licence; the same checkpoint is on
    /// Kaggle Models (`google/musiq`, koniq-10k variation, licenseName
    /// Apache-2.0).
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::Permissive,
        terms_url: "https://www.kaggle.com/models/google/musiq/tensorFlow2/koniq-10k",
    };

    /// Load an exported model directory (`musiq.onnx`, `musiq.meta`, as
    /// written by `models/musiq/export.py`) on `engine`.
    ///
    /// The weights are permissively licensed; `from_dir` takes no consent.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`].
    pub fn from_dir(dir: impl AsRef<Path>, engine: Engine) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let (_channels, min_size) = read_meta(&dir.join("musiq.meta"))?;
        let backend = match engine {
            Engine::Rten => Backend::Image(engine::load_image(&dir.join("musiq.onnx"))?),
            #[cfg(feature = "ort")]
            gpu => Backend::Image(Box::new(engine::ort_engine::OrtGpuImage::load(
                &dir.join("musiq.onnx"),
                "musiq",
                gpu,
            )?)),
        };
        Ok(Self { backend, min_size })
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk(t) => t.engine(),
            Backend::Pair(p) => p.engine(),
            Backend::Image(g) => g.engine(),
        }
    }

    /// The patch sequence for one `[c, h, w]` f32 image at `scale_id`,
    /// appended to `out` as `ceil(h/32)·ceil(w/32)` rows of 3075 floats
    /// (zero-padded SAME window, hashed spatial index, scale, mask 1),
    /// then zero rows up to `cap` when `cap >= 0` (upstream's
    /// `_pad_or_cut_to_max_seq_len`; our caps can only pad, never cut).
    fn extract(img: &[f32], h: usize, w: usize, cap: i64, scale_id: f32, out: &mut Vec<f32>) {
        let (ch, cw) = (h.div_ceil(PATCH), w.div_ceil(PATCH));
        let (pt, pl) = ((ch * PATCH - h) / 2, (cw * PATCH - w) / 2);
        let pos = |count: usize| -> Vec<i64> {
            // tf.compat.v1 resize NEAREST of arange(GRID): floor(i·GRID/count).
            let sc = GRID as f32 / count as f32;
            (0..count)
                .map(|i| ((i as f32 * sc).floor() as i64).min(GRID as i64 - 1))
                .collect()
        };
        let (ph, pw) = (pos(ch), pos(cw));
        for dy in 0..ch {
            for dx in 0..cw {
                for py in 0..PATCH {
                    let sy = dy * PATCH + py;
                    for px in 0..PATCH {
                        let sx = dx * PATCH + px;
                        for c in 0..3 {
                            let v = if sy >= pt && sy - pt < h && sx >= pl && sx - pl < w {
                                img[c * h * w + (sy - pt) * w + sx - pl]
                            } else {
                                0.0
                            };
                            out.push(v);
                        }
                    }
                }
                out.push((ph[dy] * GRID as i64 + pw[dx]) as f32);
                out.push(scale_id);
                out.push(1.0);
            }
        }
        if cap >= 0 {
            out.resize(
                out.len() + (cap as usize).saturating_sub(ch * cw) * ROW,
                0.0,
            );
        }
    }

    /// `get_multiscale_patches` on the normalised image: the two rescaled
    /// sequences plus the full-resolution one — `[1, L, 3075]` flattened.
    fn prep(img: &za::Rgb8In<'_>, min_size: usize) -> Result<Vec<f32>, Error> {
        let (h, w) = (img.height, img.width);
        if h == 0 || w == 0 {
            return Err(Error::TooSmall { min: 1 });
        }
        if h < min_size || w < min_size {
            return Err(Error::TooSmall { min: min_size });
        }
        // L^2 attention bound: the full-resolution group is uncapped
        // (upstream's -1), so the patch sequence is 193 + ceil(h/32)^2ish
        // rows; refuse before allocating when it would exceed MAX_ROWS
        // (a [1,6,8192,8192] f32 attention map is ~1.6 GB).
        if 193 + h.div_ceil(PATCH) * w.div_ceil(PATCH) > MAX_ROWS {
            return Err(Error::TooManyPatches { limit: MAX_ROWS });
        }
        let full = rgb8_to_planar_u8(img);
        // (x − 0)/255 then −1 + t·2 — f32 element order as upstream's
        // normalize_value_range.
        let norm: Vec<f32> = full
            .iter()
            .map(|&v| -1.0f32 + (v as f32 / 255.0) * 2.0)
            .collect();
        let mut out = Vec::new();
        for (scale_id, longer) in LONGER.iter().enumerate() {
            let ratio = *longer as f32 / h.max(w) as f32;
            // tf.round is ties-to-even (`rint`), not half-away.
            let rh = (h as f32 * ratio).round_ties_even() as usize;
            let rw = (w as f32 * ratio).round_ties_even() as usize;
            // Upstream errors "output dimensions must be positive" here —
            // an extreme aspect ratio can round a resized side to zero.
            if rh == 0 || rw == 0 {
                return Err(Error::DegenerateAspectRatio);
            }
            let r = resize_gaussian_tf_f32(&norm, 3, h, w, rh, rw);
            let cap = (longer.div_ceil(PATCH) * longer.div_ceil(PATCH)) as i64;
            Self::extract(&r, rh, rw, cap, scale_id as f32, &mut out);
        }
        Self::extract(&norm, h, w, -1, LONGER.len() as f32, &mut out);
        Ok(out)
    }

    /// Score one prepared image: `x [1, L, 3075]` in, scalar MOS out.
    fn score_image(&self, img: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let x = Self::prep(img, self.min_size)?;
        let l = x.len() / ROW;
        match &self.backend {
            Backend::Image(g) => g.score(&[NamedInput {
                name: "x",
                shape: vec![1, l, ROW],
                data: x,
            }]),
            Backend::Trunk(_) | Backend::Pair(_) => {
                Err(Error::Engine("musiq needs an image-graph engine".into()))
            }
        }
    }
}

impl za::NoReferenceMetric for Musiq {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "google-research/musiq @3ef5211 koniq_ckpt, 3-scale TF-gaussian patches; engine {}",
            self.engine_name()
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

    /// Zero dimensions are refused before any arithmetic, extreme
    /// aspect ratios whose rescaled side rounds to zero fail like
    /// upstream's "output dimensions must be positive", and inputs
    /// whose patch sequence would exceed MAX_ROWS are refused.
    #[test]
    fn prep_size_bounds() {
        for (w, h) in [(0, 5), (5, 0)] {
            assert!(matches!(
                Musiq::prep(&view(&[], w, h), 1),
                Err(Error::TooSmall { min: 1 })
            ));
        }
        assert_eq!(
            Error::TooManyPatches { limit: 8192 }.to_string(),
            "image exceeds the metric's 8192-patch-row intermediate bound"
        );
        assert!(
            Error::DegenerateAspectRatio
                .to_string()
                .contains("degenerate aspect ratio")
        );
        // 1x1000: 224/1000 -> rh 0 (upstream errors identically).
        assert!(matches!(
            Musiq::prep(&view(&[0u8; 3000], 1000, 1), 1),
            Err(Error::DegenerateAspectRatio)
        ));
        // 2x1000: 448/1000 rounds to 0 at scale 224; scale 384 would give 1.
        assert!(matches!(
            Musiq::prep(&view(&[0u8; 6000], 1000, 2), 1),
            Err(Error::DegenerateAspectRatio)
        ));
        // 4096x4096: 193 + 128^2 = 16577 rows > MAX_ROWS.
        let big = vec![0u8; 4096 * 4096 * 3];
        assert!(matches!(
            Musiq::prep(&view(&big, 4096, 4096), 1),
            Err(Error::TooManyPatches { limit: 8192 })
        ));
        assert!(Musiq::prep(&view(&[0u8; 3], 1, 1), 1).is_ok());
    }

    /// The sequence length is `49 + 144 + ceil(h/32)·ceil(w/32)`.
    #[test]
    fn prep_seq_len() {
        for (h, w, l) in [(256, 256, 257), (64, 64, 197), (240, 320, 273)] {
            let px = vec![128u8; h * w * 3];
            let x = Musiq::prep(&view(&px, w, h), 1).unwrap();
            assert_eq!(x.len(), l * ROW, "{h}x{w}");
        }
    }

    /// Hash grid: tf v1 nearest of `arange(10)` — `floor(i·10/count)`.
    #[test]
    fn hash_is_v1_nearest() {
        let px = vec![0u8; 256 * 256 * 3];
        let x = Musiq::prep(&view(&px, 256, 256), 1).unwrap();
        // Original-resolution group starts at row 49+144=193; 256 has an
        // 8x8 patch grid, hash row 0..8 per axis.
        let idx = |dy: usize, dx: usize| x[(193 + dy * 8 + dx) * ROW + 3072] as i64;
        // ph for count 8: floor(i·10/8) = 0,1,2,3,5,6,7,8
        assert_eq!(idx(0, 0), 0);
        assert_eq!(idx(1, 1), 11);
        assert_eq!(idx(4, 4), 5 * 10 + 5);
        assert_eq!(idx(7, 7), 8 * 10 + 8);
    }

    /// Row metadata: hashed position, scale id, mask 1 for real rows;
    /// a tiny image's rescaled groups pad with all-zero rows.
    #[test]
    fn rows_meta_and_padding() {
        let px = vec![200u8; 8 * 8 * 3];
        let x = Musiq::prep(&view(&px, 8, 8), 1).unwrap();
        let l = x.len() / ROW;
        assert_eq!(l, 49 + 144 + 1);
        // First row of each group.
        assert_eq!(x[3073], 0.0);
        assert_eq!(x[49 * ROW + 3073], 1.0);
        assert_eq!(x[193 * ROW + 3073], 2.0);
        assert_eq!(x[3074], 1.0);
    }

    /// A row-stride larger than `w*3` reads the same pixels.
    #[test]
    fn prep_strided_rows() {
        let (w, h, stride) = (5, 4, 5 * 3 + 7);
        let mut packed = vec![0u8; w * h * 3];
        let mut buf = vec![0xEEu8; stride * h];
        for (i, v) in packed.iter_mut().enumerate() {
            *v = (i * 13 + 7) as u8;
        }
        for y in 0..h {
            buf[y * stride..y * stride + w * 3]
                .copy_from_slice(&packed[y * w * 3..(y + 1) * w * 3]);
        }
        let s = za::Rgb8In {
            pixels: &buf,
            width: w,
            height: h,
            stride,
        };
        assert_eq!(
            Musiq::prep(&s, 1).unwrap(),
            Musiq::prep(&view(&packed, w, h), 1).unwrap()
        );
    }

    /// Normalisation is `(x/255)·2 − 1` in f32: 255 → 1, 0 → −1.
    #[test]
    fn prep_normalises() {
        let px = vec![255u8; 32 * 32 * 3];
        let x = Musiq::prep(&view(&px, 32, 32), 1).unwrap();
        // Original-res group is the last row; its first pixel is 1.0.
        assert_eq!(x[193 * ROW], 1.0);
        let px = vec![0u8; 32 * 32 * 3];
        let x = Musiq::prep(&view(&px, 32, 32), 1).unwrap();
        assert_eq!(x[193 * ROW], -1.0);
    }
}
