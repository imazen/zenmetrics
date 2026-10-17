//! AHIQ as pyiqa 0.1.16 scores it (`pyiqa.create_metric('ahiq')`,
//! checkpoint `AHIQ_vit_p8_epoch33-da3ea303.pth`; the AIC2026 `AHIQ` column).
//!
//! What differs from zenmetrics-nn's `ahiq` (upstream `test.py` nine-crop
//! protocol), all from `pyiqa/archs/ahiq_arch.py`:
//! - backbones as timm 1.0.30 resolves pyiqa's bare names:
//!   `vit_base_patch8_224.augreg2_in21k_ft_in1k` (not augreg) and
//!   `resnet50.a1_in1k`; the epoch-33 head checkpoint (not epoch 32);
//! - ImageNet mean/std input normalisation (not `[-1, 1]`);
//! - `uniform_crop(…, 224, 20)`: step `(d − 224) // 4` per axis, crops at
//!   `(i·step_h, j·step_w)`, `i < 4`, `j < 5`; images with
//!   `min(h, w) ≤ 224` are first upscaled by `F.interpolate(scale_factor =
//!   225 / min(h, w), mode='bilinear')` (align_corners False, no
//!   antialias), so there is no minimum size;
//! - score = mean of the 20 crop scores.
//!
//! Graphs (`models/ahiq-pyiqa/export.py`): `ahiq_pyiqa_trunk.onnx` (crop in
//! `[0,1]` → ViT blocks 0–4 and ResNet layer-1 taps), `ahiq_pyiqa_ref.onnx`
//! (reference taps → deform offsets and the reduced reference features
//! `f_ref`), `ahiq_pyiqa_dist.onnx` (distorted taps + offsets + `f_ref` →
//! crop score). The split is exact: pyiqa computes the offsets from the
//! reference ViT map only and reduces each side separately.
//!
//! Higher is better. The warm reference caches the 20 crops' offsets and
//! `f_ref` (≈ 8.5 MB).

use std::path::Path;

use zenmetrics_abstractions as za;
use zenmetrics_nn::{Error, WeightsConsent, WeightsLicence, WeightsTerms};

use crate::Engine;
use crate::engine::{self, Graph, Tensor};
use crate::port::{Port, check_buffer, impl_metric};

const COLUMN: &str = "ahiq_pyiqa";
const CROP: usize = 224;
const NCROP: usize = 20;
/// Largest side accepted (the f32 planar copy is 12 B per pixel).
const MAX_SIDE: usize = 16_384;

/// AHIQ, pyiqa convention (see the module docs).
pub struct AhiqPyiqa {
    trunk: Box<dyn Graph>,
    refh: Box<dyn Graph>,
    disth: Box<dyn Graph>,
}

/// One reference crop's head inputs.
struct RefCrop {
    offset: Tensor,
    f_ref: Tensor,
}

/// A reference image's per-crop offsets and reduced features.
pub(crate) struct WarmRef {
    w: usize,
    h: usize,
    crops: Vec<RefCrop>,
}

/// Planar `[3, h, w]` f32 in `[0,1]` (`u8 / 255`), upscaled like pyiqa's
/// `uniform_crop` when `min(h, w) ≤ 224`. Returns `(planes, h, w)`.
fn prepare(rgb: &[u8], w: usize, h: usize, stride: usize) -> (Vec<f32>, usize, usize) {
    let x = crate::engine::rgb8_to_planar_01(&za::Rgb8In {
        pixels: rgb,
        width: w,
        height: h,
        stride,
    });
    let m = w.min(h);
    if m > CROP {
        return (x, h, w);
    }
    let sf = (CROP + 1) as f64 / m as f64;
    let (oh, ow) = (
        (h as f64 * sf).floor() as usize,
        (w as f64 * sf).floor() as usize,
    );
    (bilinear(&x, h, w, oh, ow, sf), oh, ow)
}

/// One axis of torch's `compute_indices_weights_linear` (align_corners
/// False, explicit scale factor): `(i0, i1, w0, w1)` per output index.
fn axis(input: usize, output: usize, sf: f64) -> Vec<(usize, usize, f32, f32)> {
    let scale = (1.0 / sf) as f32;
    (0..output)
        .map(|i| {
            let src = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
            let i0 = (src.floor() as usize).min(input - 1);
            let l1 = (src - i0 as f32).clamp(0.0, 1.0);
            let i1 = if i0 < input - 1 { i0 + 1 } else { i0 };
            (i0, i1, 1.0 - l1, l1)
        })
        .collect()
}

/// `F.interpolate(x, scale_factor=sf, mode='bilinear')` on planar f32:
/// width taps inside, height taps outside, as ATen's generic CPU kernel
/// nests them.
fn bilinear(x: &[f32], h: usize, w: usize, oh: usize, ow: usize, sf: f64) -> Vec<f32> {
    let (ay, ax) = (axis(h, oh, sf), axis(w, ow, sf));
    let mut out = Vec::with_capacity(3 * oh * ow);
    for c in 0..3 {
        let p = &x[c * h * w..][..h * w];
        for &(y0, y1, wy0, wy1) in &ay {
            let (r0, r1) = (&p[y0 * w..][..w], &p[y1 * w..][..w]);
            for &(x0, x1, wx0, wx1) in &ax {
                let t0 = r0[x0] * wx0 + r0[x1] * wx1;
                let t1 = r1[x0] * wx0 + r1[x1] * wx1;
                out.push(t0 * wy0 + t1 * wy1);
            }
        }
    }
    out
}

/// The 20 crop origins `(top, left)`.
fn origins(h: usize, w: usize) -> Vec<(usize, usize)> {
    let (sh, sw) = ((h - CROP) / 4, (w - CROP) / 4);
    (0..4)
        .flat_map(|i| (0..5).map(move |j| (i * sh, j * sw)))
        .take(NCROP)
        .collect()
}

fn crop(x: &[f32], h: usize, w: usize, top: usize, left: usize) -> Tensor {
    let mut out = Vec::with_capacity(3 * CROP * CROP);
    for c in 0..3 {
        for y in top..top + CROP {
            out.extend_from_slice(&x[c * h * w + y * w + left..][..CROP]);
        }
    }
    Tensor::new(vec![1, 3, CROP, CROP], out)
}

impl AhiqPyiqa {
    /// The epoch-33 checkpoint's terms (pyiqa's weight repository states no
    /// licence; pyiqa's terms are noncommercial). The ImageNet backbones are
    /// Apache-2.0 via timm.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::NonCommercial,
        terms_url: "https://github.com/chaofengc/IQA-PyTorch/blob/main/LICENSE",
    };

    /// Load `ahiq_pyiqa_{trunk,ref,dist}.onnx` from `dir` (written by
    /// `models/ahiq-pyiqa/export.py`) on `engine`.
    ///
    /// # Errors
    /// [`Error::WeightsNotAccepted`] when `consent` does not accept
    /// [`Self::WEIGHTS`] (checked before any file is read); otherwise I/O or
    /// engine load failures.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        consent: &WeightsConsent,
    ) -> Result<Self, Error> {
        engine::accept(consent, &Self::WEIGHTS)?;
        let dir = dir.as_ref();
        let g = |n: &str| engine::load(&dir.join(format!("{COLUMN}_{n}.onnx")), engine);
        Ok(Self {
            trunk: g("trunk")?,
            refh: g("ref")?,
            disth: g("dist")?,
        })
    }

    /// The prepared image's 20 crops' trunk taps, one crop at a time.
    fn for_crops(
        &self,
        img: &za::Rgb8In<'_>,
        mut f: impl FnMut(usize, Vec<Tensor>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        check_buffer(img)?;
        let (w, h) = (img.width, img.height);
        if w > MAX_SIDE || h > MAX_SIDE {
            return Err(Error::TooLarge {
                limit: MAX_SIDE * MAX_SIDE,
            });
        }
        let (x, ph, pw) = prepare(img.pixels, w, h, img.stride);
        if ph < CROP || pw < CROP {
            return Err(Error::TooSmall { min: CROP });
        }
        for (k, (t, l)) in origins(ph, pw).into_iter().enumerate() {
            let taps = self.trunk.run(vec![crop(&x, ph, pw, t, l)])?;
            if taps.len() != 2 {
                return Err(Error::BadModel("trunk must return vit and cnn".into()));
            }
            f(k, taps)?;
        }
        Ok(())
    }
}

impl Port for AhiqPyiqa {
    type Warm = WarmRef;
    const COLUMN: &'static str = COLUMN;
    const KIND: za::ScoreKind = za::ScoreKind::Similarity;
    const REVISION: u32 = 1;

    fn reference(&self) -> String {
        format!(
            "pyiqa 0.1.16 ahiq (AHIQ_vit_p8_epoch33-da3ea303, timm 1.0.30 backbones, 20 uniform crops); engine {}",
            self.trunk.engine()
        )
    }

    fn warm(&self, r: &za::Rgb8In<'_>) -> Result<WarmRef, Error> {
        let mut crops = Vec::with_capacity(NCROP);
        self.for_crops(r, |_, taps| {
            let mut o = self.refh.run(taps)?;
            if o.len() != 2 {
                return Err(Error::BadModel(
                    "ref head must return offset and f_ref".into(),
                ));
            }
            let f_ref = o.pop().expect("two outputs");
            let offset = o.pop().expect("two outputs");
            crops.push(RefCrop { offset, f_ref });
            Ok(())
        })?;
        Ok(WarmRef {
            w: r.width,
            h: r.height,
            crops,
        })
    }

    fn warm_size(w: &WarmRef) -> (usize, usize) {
        (w.w, w.h)
    }

    fn score_warm(&self, r: &WarmRef, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let mut sum = 0.0f64;
        self.for_crops(d, |k, mut taps| {
            let cnn = taps.pop().expect("two taps");
            let vit = taps.pop().expect("two taps");
            let rc = &r.crops[k];
            let out = self
                .disth
                .run(vec![vit, cnn, rc.offset.clone(), rc.f_ref.clone()])?;
            match out.as_slice() {
                [s] if s.data.len() == 1 => sum += f64::from(s.data[0]),
                _ => return Err(Error::BadModel("dist head must return one score".into())),
            }
            Ok(())
        })?;
        Ok(sum / NCROP as f64)
    }
}

impl_metric!(AhiqPyiqa);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crops_are_pyiqa_uniform_crop() {
        // 840 × 944 (h × w): step 154 × 180.
        let o = origins(840, 944);
        assert_eq!(o.len(), 20);
        assert_eq!(o[0], (0, 0));
        assert_eq!(o[4], (0, 720));
        assert_eq!(o[19], (462, 720));
    }

    #[test]
    fn small_images_upscale_to_225() {
        let rgb = vec![128u8; 64 * 64 * 3];
        let (x, h, w) = prepare(&rgb, 64, 64, 192);
        assert_eq!((h, w), (225, 225));
        assert!(x.iter().all(|&v| (v - 128.0 / 255.0).abs() < 1e-6));
    }
}
