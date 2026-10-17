//! PieAPP as pyiqa 0.1.16 scores it (`pyiqa.create_metric('pieapp')`,
//! `pyiqa/archs/pieapp_arch.py`; the AIC2026 `PieAPP` column).
//!
//! The network and checkpoint are zenmetrics-nn's `pieapp` (the same
//! `PieAPPv0.1` file, sha256 `0937b014…`), and this port loads that export
//! unchanged (`pieapp.onnx` from `crates/zenmetrics-nn/models/pieapp/export.py`:
//! inputs `reference`, `distorted` `[n,3,64,64]` RGB in `[0,1]`, output
//! `[n,2]` = per-patch error and weight + 1e-6). The convention that differs
//! is the patch grid: pyiqa unfolds 64×64 patches at **stride 27** with no
//! padding (`extract_2d_patches(x, 64, 27, padding='none')`: origins 0, 27,
//! … ≤ d − 64), where zenmetrics-nn uses the official release's dense
//! stride 6. Score `Σ e·w / Σ w` over the patches (pooled in f64; pyiqa in
//! f32). Lower is better.
//!
//! The warm reference keeps the reference's planar copy only; the network
//! runs on both patches of every pair, so it saves no network work.

use std::path::Path;

use zenmetrics_abstractions as za;
use zenmetrics_nn::{Error, WeightsConsent, WeightsLicence, WeightsTerms};

use crate::Engine;
use crate::engine::{self, Graph, Tensor, read_meta};
use crate::port::{Port, check_buffer, impl_metric};

const COLUMN: &str = "pieapp_pyiqa";
const PATCH: usize = 64;
/// pyiqa's `PieAPP(stride=27)` default.
const STRIDE: usize = 27;
/// Patch pairs per graph call.
const BATCH: usize = 32;
const MAX_SIDE: usize = 16_384;

/// PieAPP, pyiqa convention (see the module docs).
pub struct PieAppPyiqa {
    graph: Box<dyn Graph>,
}

/// Planar `[3, h, w]` f32 in `[0,1]`.
pub(crate) struct Planes {
    data: Vec<f32>,
    w: usize,
    h: usize,
}

fn planes(img: &za::Rgb8In<'_>) -> Result<Planes, Error> {
    check_buffer(img)?;
    let (w, h) = (img.width, img.height);
    if w < PATCH || h < PATCH {
        return Err(Error::TooSmall { min: PATCH });
    }
    if w > MAX_SIDE || h > MAX_SIDE {
        return Err(Error::TooLarge {
            limit: MAX_SIDE * MAX_SIDE,
        });
    }
    Ok(Planes {
        data: crate::engine::rgb8_to_planar_01(img),
        w,
        h,
    })
}

/// `[n, 3, 64, 64]` patches of `p` at `positions` (top, left).
fn patches(p: &Planes, positions: &[(usize, usize)]) -> Vec<f32> {
    let mut out = Vec::with_capacity(positions.len() * 3 * PATCH * PATCH);
    for &(t, l) in positions {
        for c in 0..3 {
            for y in t..t + PATCH {
                out.extend_from_slice(&p.data[c * p.w * p.h + y * p.w + l..][..PATCH]);
            }
        }
    }
    out
}

/// pyiqa's unfold origins along one axis.
fn origins(d: usize) -> impl Iterator<Item = usize> {
    (0..=d - PATCH).step_by(STRIDE)
}

impl PieAppPyiqa {
    /// The PieAPP v0.1 checkpoint's noncommercial terms (as zenmetrics-nn's
    /// `PieApp::WEIGHTS`, under this column's name).
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::NonCommercial,
        terms_url: "https://github.com/prashnani/PerceptualImageError/blob/master/LICENSE.txt",
    };

    /// Load zenmetrics-nn's PieAPP export (`pieapp.onnx`, `pieapp.meta`)
    /// from `dir` on `engine`.
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
        let (channels, min_size) = read_meta(&dir.join("pieapp.meta"))?;
        if channels != [3] || min_size != PATCH {
            return Err(Error::BadModel(
                "pieapp.meta: expected channels 3 and min_size 64".into(),
            ));
        }
        Ok(Self {
            graph: engine::load(&dir.join("pieapp.onnx"), engine)?,
        })
    }
}

impl Port for PieAppPyiqa {
    type Warm = Planes;
    const COLUMN: &'static str = COLUMN;
    const KIND: za::ScoreKind = za::ScoreKind::Difference;
    const REVISION: u32 = 1;

    fn reference(&self) -> String {
        format!(
            "pyiqa 0.1.16 pieapp (PieAPPv0.1, 64x64 patches at stride 27); engine {}",
            self.graph.engine()
        )
    }

    fn warm(&self, r: &za::Rgb8In<'_>) -> Result<Planes, Error> {
        planes(r)
    }

    fn warm_size(w: &Planes) -> (usize, usize) {
        (w.w, w.h)
    }

    fn reuses_reference_work() -> bool {
        false
    }

    fn score_warm(&self, r: &Planes, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let d = planes(d)?;
        let positions: Vec<(usize, usize)> = origins(r.h)
            .flat_map(|t| origins(r.w).map(move |l| (t, l)))
            .collect();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for chunk in positions.chunks(BATCH) {
            let n = chunk.len();
            let out = self.graph.run(vec![
                Tensor::new(vec![n, 3, PATCH, PATCH], patches(r, chunk)),
                Tensor::new(vec![n, 3, PATCH, PATCH], patches(&d, chunk)),
            ])?;
            let [t] = out.as_slice() else {
                return Err(Error::BadModel(
                    "pieapp graph must return one output".into(),
                ));
            };
            if t.shape != [n, 2] {
                return Err(Error::BadModel(format!(
                    "pieapp patch output {:?}, expected [{n},2]",
                    t.shape
                )));
            }
            for pair in t.data.as_chunks::<2>().0 {
                let (e, w) = (f64::from(pair[0]), f64::from(pair[1]));
                if !e.is_finite() || !w.is_finite() {
                    return Err(Error::Engine("nonfinite PieAPP prediction".into()));
                }
                num += e * w;
                den += w;
            }
        }
        if den == 0.0 {
            return Err(Error::Engine("PieAPP patch weights sum to zero".into()));
        }
        Ok(num / den)
    }
}

impl_metric!(PieAppPyiqa);

#[cfg(test)]
mod tests {
    use super::origins;

    #[test]
    fn grid_is_torch_unfold() {
        // floor((d - 64) / 27) + 1 origins.
        for d in 64..400usize {
            assert_eq!(origins(d).count(), (d - 64) / 27 + 1, "d={d}");
        }
        assert_eq!(origins(840).last(), Some(756));
    }
}
