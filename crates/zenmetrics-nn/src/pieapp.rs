//! PieAPP (Prashnani et al., CVPR 2018), independently reconstructed from
//! paper section 3 / figure 4 and checkpoint tensor metadata. The official
//! implementation is a black-box oracle only; see `models/pieapp/PROVENANCE.md`.

use std::borrow::Cow;
use std::path::Path;

use crate::common::read_meta;
use crate::engine::{self, ImageGraph, NamedInput};
use crate::preprocess::rgb8_to_planar_01;
use crate::{Engine, Error, WeightsConsent, WeightsLicence, WeightsTerms, za};

const COLUMN: &str = "pieapp";
const PATCH: usize = 64;
const STEP: usize = 6;
const BATCH: usize = 32;
const MAX_SIDE: usize = 4096;
const MAX_PIXELS: usize = if usize::BITS >= 64 {
    9_000_000
} else {
    1_048_576
};

/// PieAPP v0.1 with deterministic dense 64×64 patches at stride six.
/// Weights require explicit noncommercial consent. The graph predicts
/// patch error and weight; bounded batches are pooled on the host.
pub struct PieApp {
    graph: Box<dyn ImageGraph>,
}

impl PieApp {
    /// The official checkpoint's noncommercial terms.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::NonCommercial,
        terms_url: "https://github.com/prashnani/PerceptualImageError/blob/master/LICENSE.txt",
    };

    /// Load a paper/tensor-metadata reconstruction exported by
    /// `models/pieapp/export.py`. Consent is checked before reading files.
    ///
    /// # Errors
    /// Refuses missing consent, malformed exports or engine load failure.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        consent: &WeightsConsent,
    ) -> Result<Self, Error> {
        consent.check(&Self::WEIGHTS)?;
        let dir = dir.as_ref();
        let (channels, minimum) = read_meta(&dir.join("pieapp.meta"))?;
        if channels != [3] || minimum != PATCH {
            return Err(Error::BadModel(
                "PieAPP requires channels 3 and min_size 64".into(),
            ));
        }
        let path = dir.join("pieapp.onnx");
        let graph: Box<dyn ImageGraph> = match engine {
            Engine::Rten => engine::load_image(&path)?,
            #[cfg(feature = "ort")]
            Engine::OrtCuda => Box::new(engine::ort_engine::OrtCudaImage::load(&path, "patches")?),
        };
        Ok(Self { graph })
    }

    fn prepare(img: &za::Rgb8In<'_>) -> Result<Prepared, Error> {
        validate_geometry(img.width, img.height)?;
        Ok(Prepared {
            data: rgb8_to_planar_01(img),
            w: img.width,
            h: img.height,
        })
    }

    fn score(&self, r: &Prepared, d: &Prepared, opts: za::EvalOpts<'_>) -> za::MetricResult<f64> {
        use za::ResultMetricExt as _;
        if (r.w, r.h) != (d.w, d.h) {
            return Err(Error::SizeMismatch).metric_err_with(COLUMN, Error::kind);
        }
        let mut positions = Vec::with_capacity(BATCH);
        let mut sum = 0.0f64;
        let mut weights = 0.0f64;
        let mut consume = |positions: &[(usize, usize)]| -> za::MetricResult<()> {
            opts.checkpoint()?;
            let n = positions.len();
            let (shape, result) = self
                .graph
                .tensor(&[
                    NamedInput {
                        name: "reference",
                        shape: vec![n, 3, PATCH, PATCH],
                        data: patches(r, positions),
                    },
                    NamedInput {
                        name: "distorted",
                        shape: vec![n, 3, PATCH, PATCH],
                        data: patches(d, positions),
                    },
                ])
                .metric_err_with(COLUMN, Error::kind)?;
            if shape != [n, 2] || result.len() != 2 * n {
                return Err(Error::BadModel(format!(
                    "PieAPP patch output {shape:?}, expected [{n},2]"
                )))
                .metric_err_with(COLUMN, Error::kind);
            }
            for pair in result.as_chunks::<2>().0 {
                let [error, weight] = [f64::from(pair[0]), f64::from(pair[1])];
                if !error.is_finite() || !weight.is_finite() {
                    return Err(Error::Engine("nonfinite PieAPP prediction".into()))
                        .metric_err_with(COLUMN, Error::kind);
                }
                sum += error * weight;
                weights += weight;
            }
            Ok(())
        };
        let ys = axis_positions(r.h);
        let xs = axis_positions(r.w);
        for y in ys {
            for &x in &xs {
                positions.push((y, x));
                if positions.len() == BATCH {
                    consume(&positions)?;
                    positions.clear();
                }
            }
        }
        if !positions.is_empty() {
            consume(&positions)?;
        }
        if !sum.is_finite() || !weights.is_finite() || weights == 0.0 {
            return Err(Error::Engine("invalid PieAPP weighted pool".into()))
                .metric_err_with(COLUMN, Error::kind);
        }
        Ok(sum / weights)
    }
}

fn validate_geometry(width: usize, height: usize) -> Result<(), Error> {
    if width < PATCH || height < PATCH {
        return Err(Error::TooSmall { min: PATCH });
    }
    if width > MAX_SIDE || height > MAX_SIDE {
        return Err(Error::TooLarge { limit: MAX_SIDE });
    }
    if width.checked_mul(height).is_none_or(|n| n > MAX_PIXELS) {
        return Err(Error::TooLarge { limit: MAX_PIXELS });
    }
    Ok(())
}

struct Prepared {
    data: Vec<f32>,
    w: usize,
    h: usize,
}

/// Dense stride-six grid; incomplete trailing strides are not sampled.
fn axis_positions(side: usize) -> Vec<usize> {
    (0..=side - PATCH).step_by(STEP).collect()
}

fn patches(image: &Prepared, positions: &[(usize, usize)]) -> Vec<f32> {
    let plane = image.w * image.h;
    let mut out = Vec::with_capacity(positions.len() * 3 * PATCH * PATCH);
    for &(y, x) in positions {
        for c in 0..3 {
            for row in y..y + PATCH {
                let start = c * plane + row * image.w + x;
                out.extend_from_slice(&image.data[start..start + PATCH]);
            }
        }
    }
    out
}

impl za::Metric for PieApp {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(3)).with_reference(format!(
            "PieAPP v0.1 paper/weights reconstruction, dense stride 6; engine {}",
            self.graph.engine()
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
        let r = Self::prepare(&r).metric_err_with(COLUMN, Error::kind)?;
        let d = Self::prepare(&d).metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(
            COLUMN,
            self.score(&r, &d, opts)?,
        )))
    }
    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let r = Self::prepare(&za::rgb8_view(&r)?).metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(WarmPieApp {
            metric: self,
            reference: r,
        }))
    }
}

/// Cache only planar conversion; dense reference feature caches would grow
/// with patch count times 120832 and defeat the bounded-batch schedule.
struct WarmPieApp<'m> {
    metric: &'m PieApp,
    reference: Prepared,
}
impl za::WarmReference for WarmPieApp<'_> {
    fn reuses_reference_work(&self) -> bool {
        false
    }
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let d = PieApp::prepare(&za::rgb8_view(&d)?).metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(
            COLUMN,
            self.metric.score(&self.reference, &d, opts)?,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SignedWeights([f32; 2]);
    impl ImageGraph for SignedWeights {
        fn engine(&self) -> &'static str {
            "controlled-oracle-weights"
        }
        fn score(&self, _: &[NamedInput]) -> Result<f64, Error> {
            Err(Error::Engine("tensor-only test graph".into()))
        }
        fn tensor(&self, inputs: &[NamedInput]) -> Result<(Vec<usize>, Vec<f32>), Error> {
            assert_eq!(inputs[0].shape[0], 2);
            Ok((vec![2, 2], vec![1.0, self.0[0], 1.0, self.0[1]]))
        }
    }

    #[test]
    fn signed_public_weights_pool_and_zero_denominator_is_refused() {
        let image = Prepared {
            data: vec![0.0; 70 * 64 * 3],
            w: 70,
            h: 64,
        };
        // Official opaque bias/projection controls, forced raw heads -1/+1.
        // benchmarks/pieapp_review_2026-10-16/weight-identify.json.
        let model = PieApp {
            graph: Box::new(SignedWeights([-0.999_999, 1.000_001])),
        };
        assert_eq!(
            model.score(&image, &image, za::EvalOpts::new()).unwrap(),
            1.0
        );
        let model = PieApp {
            graph: Box::new(SignedWeights([-1.0, 1.0])),
        };
        assert!(model.score(&image, &image, za::EvalOpts::new()).is_err());
    }

    #[test]
    fn dense_grid_matches_small_blackbox_cases() {
        assert_eq!(axis_positions(64), [0]);
        assert_eq!(axis_positions(65), [0]);
        assert_eq!(
            axis_positions(128),
            [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        );
    }

    #[test]
    fn normalized_ingress_preserves_every_code_value() {
        let pixels: Vec<u8> = (0u8..=255)
            .flat_map(|v| [v, v.wrapping_add(73), v.wrapping_add(151)])
            .collect();
        let input = za::Rgb8In {
            pixels: &pixels,
            width: 256,
            height: 1,
            stride: 768,
        };
        let planar = rgb8_to_planar_01(&input);
        for (channel_index, channel) in planar.as_chunks::<256>().0.iter().enumerate() {
            for (position, &value) in channel.iter().enumerate() {
                let code = f32::from(pixels[position * 3 + channel_index]);
                assert_eq!(value.to_bits(), (code / 255.0).to_bits());
                assert_eq!(value * 255.0, code);
            }
        }
    }

    #[test]
    fn geometry_bounds_report_the_relevant_limit() {
        assert!(matches!(
            validate_geometry(4097, 64),
            Err(Error::TooLarge { limit: MAX_SIDE })
        ));
        assert!(matches!(
            validate_geometry(3001, 3000),
            Err(Error::TooLarge { limit: MAX_PIXELS })
        ));
        if usize::BITS >= 64 {
            assert!(validate_geometry(3000, 3000).is_ok());
        } else {
            assert!(validate_geometry(1024, 1024).is_ok());
        }
    }

    #[test]
    fn consent_precedes_file_access() {
        let result = PieApp::from_dir(
            "absent-model-directory",
            Engine::Rten,
            &WeightsConsent::new(),
        );
        assert!(matches!(result, Err(Error::WeightsNotAccepted(_))));
        assert_eq!(PieApp::WEIGHTS.licence, WeightsLicence::NonCommercial);
    }

    #[test]
    fn rejects_geometry_before_reading_pixels() {
        for (w, h) in [(0, 64), (64, 0), (63, 70), (70, 63)] {
            let image = za::Rgb8In {
                pixels: &[],
                width: w,
                height: h,
                stride: w * 3,
            };
            assert!(matches!(
                PieApp::prepare(&image),
                Err(Error::TooSmall { min: 64 })
            ));
        }
        let image = za::Rgb8In {
            pixels: &[],
            width: 4096,
            height: 4096,
            stride: 4096 * 3,
        };
        assert!(matches!(
            PieApp::prepare(&image),
            Err(Error::TooLarge { .. })
        ));
    }

    #[test]
    fn patch_extraction_preserves_strided_rows_and_channels() {
        let (w, h, stride) = (66, 67, 66 * 3 + 15);
        let mut tight = vec![0u8; w * h * 3];
        let mut padded = vec![219u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let value = ((x + 3 * y + 71 * c) % 251) as u8;
                    tight[(y * w + x) * 3 + c] = value;
                    padded[y * stride + x * 3 + c] = value;
                }
            }
        }
        let a = PieApp::prepare(&za::Rgb8In {
            pixels: &tight,
            width: w,
            height: h,
            stride: w * 3,
        })
        .unwrap();
        let b = PieApp::prepare(&za::Rgb8In {
            pixels: &padded,
            width: w,
            height: h,
            stride,
        })
        .unwrap();
        let positions = [(0, 0), (3, 2)];
        let actual = patches(&a, &positions);
        assert_eq!(actual, patches(&b, &positions));
        let mut expected = Vec::new();
        for (y, x) in positions {
            for c in 0..3 {
                for row in y..y + 64 {
                    for col in x..x + 64 {
                        expected.push(f32::from(((col + 3 * row + 71 * c) % 251) as u8) / 255.0);
                    }
                }
            }
        }
        assert_eq!(actual, expected);
    }
}
