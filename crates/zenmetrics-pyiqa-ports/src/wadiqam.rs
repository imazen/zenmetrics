//! WaDIQaM-FR as pyiqa 0.1.16 scores it (`pyiqa.create_metric('wadiqam_fr')`,
//! checkpoint `WaDIQaM-kadid-f7541ea5.pth`, KADID-10k-trained; the AIC2026
//! `WaDIQaM` column). The public `wadiqam-fr-*` metrics in zenmetrics-nn use
//! upstream deepIQA's LIVE/TID weights and protocol instead.
//!
//! Protocol (`pyiqa/archs/wadiqam_arch.py`, eval mode):
//! - `ToTensor` input, `u8 / 255` in f32.
//! - 32×32 patches at stride 32 with origins `range(0, h - 32, 32)` ×
//!   `range(0, w - 32, 32)`, row-major. Unlike deepIQA's `⌊(d − 32)/32⌋ + 1`
//!   grid, a dimension that is a multiple of 32 drops its last full patch,
//!   and a dimension ≤ 32 has none (pyiqa then fails; this port refuses it).
//! - Shared trunk (5 × [conv3×3+ReLU ×2, maxpool2]) → 512 features per patch
//!   (`wadiqam_pyiqa_trunk.onnx`).
//! - `feat = cat(f_ref, f_dist, f_dist − f_ref)`; `q = fc_q(feat)`,
//!   `w = fc_w(feat) + 1e-8` (`wadiqam_pyiqa_head.onnx`).
//! - score `Σ q·w / Σ w` (pooled here in f64; pyiqa sums in f32).
//!
//! Higher is better; identical pairs score an image-dependent value.
//! The warm reference caches the reference's patch features.

use std::path::Path;

use zenmetrics_abstractions as za;
use zenmetrics_nn::{Error, WeightsConsent, WeightsLicence, WeightsTerms};

use crate::Engine;
use crate::engine::{self, Graph, Tensor, read_meta};
use crate::port::{Port, check_buffer, impl_metric};

const COLUMN: &str = "wadiqam_pyiqa";
/// Patch side and grid stride.
const PATCH: usize = 32;
/// Trunk features per patch.
const FEAT: usize = 512;
/// Patches per graph call: bounds the first conv's activation at
/// 512 · 32 · 32² · 4 B = 64 MiB. Per-patch outputs are independent of the
/// batch they run in.
const CHUNK: usize = 512;
/// Patch-count bound, checked before any patch tensor allocates (a 4096²
/// image has 127² = 16,129).
const MAX_PATCHES: usize = 32_768;

/// pyiqa's patch origins along one axis: `range(0, d - 32, 32)`.
fn origins(d: usize) -> usize {
    if d <= PATCH {
        0
    } else {
        (d - PATCH).div_ceil(PATCH)
    }
}

/// WaDIQaM-FR, pyiqa convention (see the module docs).
pub struct WadiqamPyiqa {
    trunk: Box<dyn Graph>,
    head: Box<dyn Graph>,
    min_size: usize,
}

/// A reference image's patch features, `[P, 512]`.
pub(crate) struct WarmRef {
    w: usize,
    h: usize,
    feat: Vec<f32>,
}

impl WadiqamPyiqa {
    /// The checkpoint's terms: distributed by pyiqa's weight repository,
    /// which states no licence of its own; pyiqa's terms are noncommercial.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::NonCommercial,
        terms_url: "https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE",
    };

    /// Load `wadiqam_pyiqa_{trunk,head}.onnx` and `wadiqam_pyiqa.meta` from
    /// `dir` (written by `models/wadiqam-pyiqa/export.py`) on `engine`.
    ///
    /// # Errors
    /// [`Error::WeightsNotAccepted`] when `consent` does not accept
    /// [`Self::WEIGHTS`] (checked before any file is read); otherwise I/O,
    /// malformed exports or engine load failures.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        consent: &WeightsConsent,
    ) -> Result<Self, Error> {
        engine::accept(consent, &Self::WEIGHTS)?;
        let dir = dir.as_ref();
        let (channels, min_size) = read_meta(&dir.join(format!("{COLUMN}.meta")))?;
        if channels != [FEAT] {
            return Err(Error::BadModel(format!(
                "{COLUMN}.meta: channels {channels:?}, expected [{FEAT}]"
            )));
        }
        Ok(Self {
            trunk: engine::load(&dir.join(format!("{COLUMN}_trunk.onnx")), engine)?,
            head: engine::load(&dir.join(format!("{COLUMN}_head.onnx")), engine)?,
            min_size: min_size.max(PATCH + 1),
        })
    }

    /// Patch features `[P, 512]` of a packed RGB8 image.
    fn features(&self, img: &za::Rgb8In<'_>) -> Result<Vec<f32>, Error> {
        check_buffer(img)?;
        let (w, h, stride) = (img.width, img.height, img.stride);
        if w < self.min_size || h < self.min_size {
            return Err(Error::TooSmall { min: self.min_size });
        }
        let (gh, gw) = (origins(h), origins(w));
        let n = gh * gw;
        if n > MAX_PATCHES {
            return Err(Error::TooManyPatches { limit: MAX_PATCHES });
        }
        let mut feat = Vec::with_capacity(n * FEAT);
        let per = 3 * PATCH * PATCH;
        let mut first = 0;
        while first < n {
            let count = CHUNK.min(n - first);
            let mut x = vec![0.0f32; count * per];
            for (k, d) in x.chunks_exact_mut(per).enumerate() {
                let (gy, gx) = ((first + k) / gw, (first + k) % gw);
                for py in 0..PATCH {
                    let row =
                        &img.pixels[(gy * PATCH + py) * stride + gx * PATCH * 3..][..PATCH * 3];
                    for (px, p) in row.as_chunks::<3>().0.iter().enumerate() {
                        for c in 0..3 {
                            d[c * PATCH * PATCH + py * PATCH + px] = f32::from(p[c]) / 255.0;
                        }
                    }
                }
            }
            let out = self
                .trunk
                .run(vec![Tensor::new(vec![count, 3, PATCH, PATCH], x)])?;
            match out.as_slice() {
                [t] if t.shape == [count, FEAT] => feat.extend_from_slice(&t.data),
                _ => return Err(Error::BadModel("trunk must return h[P,512]".into())),
            }
            first += count;
        }
        Ok(feat)
    }
}

impl Port for WadiqamPyiqa {
    type Warm = WarmRef;
    const COLUMN: &'static str = COLUMN;
    const KIND: za::ScoreKind = za::ScoreKind::Similarity;
    const REVISION: u32 = 1;

    fn reference(&self) -> String {
        format!(
            "pyiqa 0.1.16 wadiqam_fr (WaDIQaM-kadid-f7541ea5); engine {}",
            self.trunk.engine()
        )
    }

    fn warm(&self, r: &za::Rgb8In<'_>) -> Result<WarmRef, Error> {
        Ok(WarmRef {
            w: r.width,
            h: r.height,
            feat: self.features(r)?,
        })
    }

    fn warm_size(w: &WarmRef) -> (usize, usize) {
        (w.w, w.h)
    }

    fn score_warm(&self, r: &WarmRef, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let dfeat = self.features(d)?;
        let n = dfeat.len() / FEAT;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        let mut first = 0;
        while first < n {
            let count = CHUNK.min(n - first);
            let mut f = Vec::with_capacity(count * 3 * FEAT);
            for p in first..first + count {
                let fr = &r.feat[p * FEAT..][..FEAT];
                let fd = &dfeat[p * FEAT..][..FEAT];
                f.extend_from_slice(fr);
                f.extend_from_slice(fd);
                f.extend(fd.iter().zip(fr).map(|(&a, &b)| a - b));
            }
            let out = self.head.run(vec![Tensor::new(vec![count, 3 * FEAT], f)])?;
            let [q, wt] = out.as_slice() else {
                return Err(Error::BadModel("head must return q and w".into()));
            };
            if q.data.len() != count || wt.data.len() != count {
                return Err(Error::BadModel("head outputs must be [P,1]".into()));
            }
            // pyiqa multiplies q·w in f32 before summing.
            for (&q, &a) in q.data.iter().zip(&wt.data) {
                num += f64::from(q * a);
                den += f64::from(a);
            }
            first += count;
        }
        Ok(num / den)
    }
}

impl_metric!(WadiqamPyiqa);

#[cfg(test)]
mod tests {
    use super::origins;

    #[test]
    fn origins_match_python_range() {
        for d in 0..300usize {
            let py = (0..d.saturating_sub(32)).step_by(32).count();
            assert_eq!(origins(d), py, "d={d}");
        }
    }
}
