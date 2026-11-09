//! Native linear-planar Butteraugli and explicit reusable packing for metrics
//! whose current public input API is packed. No color conversion is implicit.
use super::*;
use iqa::zenpixels::{
    Cicp, ColorContext,
    hdr::DiffuseWhite,
    plane::{FloatEncoding, PlanarSlice, PlaneEncoding, PlaneRole},
};
use iqa::{EvaluationRef, PlanarMetric, PlanarWarmReference, PreparePlanarReference};

fn unsupported<T>() -> MetricResult<T> {
    iqa::fail(iqa::MetricError::new(ErrorKind::UnsupportedInput))
}
fn linear<'a>(image: &PlanarSlice<'a>) -> MetricResult<[butteraugli::ImgRef<'a, f32>; 3]> {
    let expected = ColorContext::from_cicp(Cicp::new(1, 8, 0, true))
        .with_diffuse_white(DiffuseWhite::new(80.0));
    if image.color_context() != &expected
        || image.alpha_mode().is_some()
        || image.layout().planes().len() != 3
    {
        return unsupported();
    }
    let mut out = [None; 3];
    for (spec, plane) in image.layout().planes().iter().zip(image.planes()) {
        if spec.encoding() != PlaneEncoding::Float(FloatEncoding::NOMINAL) {
            return unsupported();
        }
        let c = match spec.role() {
            PlaneRole::Red => 0,
            PlaneRole::Green => 1,
            PlaneRole::Blue => 2,
            _ => return unsupported(),
        };
        let AnyPlane::F32(p) = plane else {
            return unsupported();
        };
        out[c] = Some(butteraugli::Img::new_stride(
            p.into_strided_samples(),
            p.width(),
            p.height(),
            p.stride(),
        ));
    }
    let [Some(r), Some(g), Some(b)] = out else {
        return unsupported();
    };
    Ok([r, g, b])
}
impl PlanarMetric for Butter {
    fn evaluate_planar(
        &mut self,
        r: &PlanarSlice<'_>,
        d: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        self.warm_planar_reference(r, opts)?
            .evaluate_planar(d, opts)
    }
}
impl PreparePlanarReference for Butter {
    type PlanarPrepared = ButterWarm;
    fn warm_planar_reference(
        &mut self,
        r: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<ButterWarm> {
        opts.checkpoint()?;
        butteraugli::ButteraugliReference::new_linear_planes(
            linear(r)?,
            butteraugli::ButteraugliParams::default().with_compute_diffmap(opts.diagnostics()),
        )
        .metric_err_with("butteraugli", butter_error)
        .map(ButterWarm)
    }
}
static BUTTER_SPECS: [ScoreSpec; 2] = [
    ScoreSpec::new("max", Direction::Lower).with_perfect(0.0),
    ScoreSpec::new("pnorm3", Direction::Lower).with_perfect(0.0),
];
// These descriptors borrow static names/units without per-call heap allocation.
struct BorrowedButter<'a> {
    values: [f64; 2],
    specs: &'a [ScoreSpec; 2],
    map: Option<butteraugli::ImgRef<'a, f32>>,
    map_spec: &'a ScoreSpec,
    grid: iqa::MapGrid,
}
impl Evaluation for BorrowedButter<'_> {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        self.specs
            .iter()
            .zip(self.values)
            .map(|(s, v)| Score::new(s, v))
    }
    fn maps(&self) -> impl Iterator<Item = iqa::MapView<'_>> {
        self.map.into_iter().map(|m| {
            iqa::MapView::new(
                self.map_spec,
                plane(m.buf(), m.width(), m.height(), m.stride()),
                self.grid,
            )
        })
    }
}
impl PlanarWarmReference for ButterWarm {
    type Output = Assessment<butteraugli::ButteraugliResult, 2>;
    fn evaluate_planar(
        &mut self,
        d: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        self.0
            .compare_linear_planes_with(linear(d)?, opts.stop(), |score, pnorm_3, map| {
                let diffmap = opts.diagnostics().then(|| {
                    butteraugli::Img::new(
                        map.rows().flatten().copied().collect(),
                        map.width(),
                        map.height(),
                    )
                });
                butter_result(butteraugli::ButteraugliResult::new(score, pnorm_3, diffmap))
            })
            .metric_err_with("butteraugli", butter_error)
    }
    fn with_evaluation(
        &mut self,
        d: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
        consume: &mut dyn FnMut(EvaluationRef<'_>),
    ) -> MetricResult<()> {
        let specs = BUTTER_SPECS.clone().map(|s| s.with_unit("butteraugli"));
        let map_spec = error_map("diffmap").with_unit("butteraugli");
        let grid = iqa::MapGrid::new(d.width(), d.height())?;
        self.0
            .compare_linear_planes_with(linear(d)?, opts.stop(), |score, p3, map| {
                let result = BorrowedButter {
                    values: [score, p3],
                    specs: &specs,
                    map: opts.diagnostics().then_some(map),
                    map_spec: &map_spec,
                    grid,
                };
                consume(EvaluationRef::new(&result));
            })
            .metric_err_with("butteraugli", butter_error)
    }
}
fn fill(image: &PlanarSlice<'_>, scratch: &mut Vec<[f32; 4]>) -> MetricResult<()> {
    let p = linear(image)?;
    let (w, h) = (image.width() as usize, image.height() as usize);
    let n = w
        .checked_mul(h)
        .ok_or_else(|| iqa::whereat::at(iqa::MetricError::new(ErrorKind::Resource)))?;
    if n > scratch.len() {
        scratch
            .try_reserve(n - scratch.len())
            .metric_err("planar-packing", ErrorKind::Resource)?;
    }
    scratch.resize(n, [0.0, 0.0, 0.0, 1.0]);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                scratch[y * w + x][c] = p[c].buf()[y * p[c].stride() + x];
            }
        }
    }
    Ok(())
}
fn packed<'a>(scratch: &'a [[f32; 4]], image: &PlanarSlice<'_>) -> PixelSlice<'a> {
    PixelSlice::new(
        bytemuck::cast_slice(scratch),
        image.width(),
        image.height(),
        image.width() as usize * 16,
        PixelDescriptor::RGBAF32_LINEAR,
    )
    .expect("checked linear RGB storage")
}
/// Prepared native packed work with an explicit reusable interleave buffer.
pub struct PackedPlanar<P> {
    inner: P,
    scratch: Vec<[f32; 4]>,
}
impl<P: WarmReference> PlanarWarmReference for PackedPlanar<P> {
    type Output = P::Output;
    fn evaluate_planar(
        &mut self,
        d: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        fill(d, &mut self.scratch)?;
        self.inner.evaluate(&packed(&self.scratch, d), opts)
    }
}
macro_rules! packed_planar {
    ($metric:ty,$warm:ty,$supported:expr) => {
        impl PlanarMetric for $metric {
            fn evaluate_planar(
                &mut self,
                r: &PlanarSlice<'_>,
                d: &PlanarSlice<'_>,
                opts: EvalOpts<'_>,
            ) -> MetricResult<Self::Output> {
                if !($supported)(self) {
                    return unsupported();
                }
                let mut a = Vec::new();
                let mut b = Vec::new();
                fill(r, &mut a)?;
                fill(d, &mut b)?;
                self.evaluate(&packed(&a, r), &packed(&b, d), opts)
            }
        }
        impl PreparePlanarReference for $metric {
            type PlanarPrepared = PackedPlanar<$warm>;
            fn warm_planar_reference(
                &mut self,
                r: &PlanarSlice<'_>,
                opts: EvalOpts<'_>,
            ) -> MetricResult<Self::PlanarPrepared> {
                if !($supported)(self) {
                    return unsupported();
                }
                let mut scratch = Vec::new();
                fill(r, &mut scratch)?;
                let inner = self.warm_reference(&packed(&scratch, r), opts)?;
                Ok(PackedPlanar { inner, scratch })
            }
        }
    };
}
packed_planar!(Ssim2, SsimWarm, |_: &Ssim2| true);
packed_planar!(Zen, ZenWarm, |m: &Zen| m.hdr.is_none());
