//! Native implementation adapters used only to validate the unpublished IQA contract.
#![forbid(unsafe_code)]
use iqa::zenpixels::{PixelBuffer, PixelSlice};
use iqa::{
    ErrorKind, EvalOpts, Evaluation, Image, Implementation, InputFormat, Metric, MetricId,
    MetricResult, ResultMetricExt, ScoreKind, ScoreSpec, WarmReference,
};
use std::borrow::Cow;
use zenpixels_iqa as iqa;

/// Native evaluation with stable owned scalar/map data.
struct NativeEval {
    scores: Vec<(Cow<'static, str>, f64)>,
    map: Option<PixelBuffer>,
}
impl Evaluation for NativeEval {
    fn score(&self) -> f64 {
        self.scores[0].1
    }
    fn scores(&self) -> iqa::Scores<'_> {
        Box::new(self.scores.iter().cloned())
    }
    fn map(&self) -> Option<iqa::MapView<'_>> {
        self.map.as_ref().map(|m| {
            iqa::MapView::new(
                "difference",
                m.as_slice(),
                iqa::MapPolarity::Difference,
                iqa::MapGeometry::identity(m.width(), m.height()),
            )
            .unwrap()
        })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
/// Bind native outputs and optional native map to a requested serving geometry.
fn result(
    scores: &[(&str, f64)],
    map: Option<(&[f32], u32, u32)>,
    opts: EvalOpts<'_>,
) -> MetricResult<Box<dyn Evaluation>> {
    let map = match map {
        Some((m, w, h)) => Some(iqa::gray_f32_buffer(m, w, h)?),
        None => None,
    };
    let e: Box<dyn Evaluation> = Box::new(NativeEval {
        scores: scores
            .iter()
            .map(|(k, v)| (Cow::Owned((*k).into()), *v))
            .collect(),
        map,
    });
    if let Some(request) = opts.map {
        iqa::apply_map_request(e, request)
    } else {
        Ok(e)
    }
}
fn id(family: &'static str, version: &'static str, source: &'static str) -> MetricId {
    MetricId::new(family, Implementation::new(family, version, 1))
        .with_reference(source)
        .with_config("adapter", "native-sdr-control-v1")
}
fn butter_error(e: &butteraugli::ButteraugliError) -> ErrorKind {
    match e {
        butteraugli::ButteraugliError::Cancelled(r) => ErrorKind::Stopped(*r),
        _ => ErrorKind::Backend,
    }
}
fn ssim_error(e: &fast_ssim2::Ssimulacra2Error) -> ErrorKind {
    match e {
        fast_ssim2::Ssimulacra2Error::Cancelled(r) => ErrorKind::Stopped(*r),
        _ => ErrorKind::Backend,
    }
}
fn zensim_error(e: &zensim::ZensimError) -> ErrorKind {
    match e {
        zensim::ZensimError::Cancelled { reason: r } => ErrorKind::Stopped(*r),
        _ => ErrorKind::Backend,
    }
}
fn pixel<'a>(image: &'a Image<'_>) -> MetricResult<&'a PixelSlice<'a>> {
    iqa::rgb8_view(image)?;
    Ok(image.as_packed().expect("packed view validated"))
}
fn rgb<'a>(p: &'a iqa::Rgb8In<'_>) -> butteraugli::ImgRef<'a, butteraugli::RGB8> {
    // PixelSlice already enforces a pixel-aligned row stride. Cast alignment is 1.
    butteraugli::Img::new_stride(
        bytemuck::cast_slice(p.pixels),
        p.width,
        p.height,
        p.stride / 3,
    )
}
/// Butteraugli native scalar pair and optional full-resolution difference map.
pub struct Butter;
impl Metric for Butter {
    fn id(&self) -> MetricId {
        id("butteraugli", "0.9.4", "31687022")
            .with_config("intensity_target", 80)
            .with_config("hf_asymmetry", 1)
            .with_config("xmul", 1)
    }
    fn outputs(&self) -> Cow<'static, [ScoreSpec]> {
        Cow::Owned(vec![
            ScoreSpec::new("max", ScoreKind::Difference),
            ScoreSpec::new("pnorm3", ScoreKind::Difference),
        ])
    }
    fn accepts(&self, f: &InputFormat) -> bool {
        iqa::is_rgb8_srgb(f)
    }
    fn evaluate(
        &mut self,
        r: Image<'_>,
        d: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn Evaluation>> {
        let (r, d) = iqa::rgb8_pair(&r, &d)?;
        let native = butteraugli::butteraugli_with_stop(
            rgb(&r),
            rgb(&d),
            &butteraugli::ButteraugliParams::default().with_compute_diffmap(opts.map.is_some()),
            opts.stop(),
        )
        .metric_err_with("butteraugli", butter_error)?;
        butter_result(native, opts)
    }
    fn warm_reference<'m>(
        &'m mut self,
        r: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn WarmReference + 'm>> {
        opts.checkpoint()?;
        let r = iqa::rgb8_view(&r)?;
        let native = butteraugli::ButteraugliReference::new(
            &iqa::contiguous_rgb8(&r),
            r.width,
            r.height,
            butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
        )
        .metric_err_with("butteraugli", butter_error)?;
        Ok(Box::new(ButterWarm(native)))
    }
}
fn butter_result(
    native: butteraugli::ButteraugliResult,
    opts: EvalOpts<'_>,
) -> MetricResult<Box<dyn Evaluation>> {
    let vals = native
        .diffmap
        .as_ref()
        .map(|m| m.pixels().collect::<Vec<f32>>());
    let map = native
        .diffmap
        .as_ref()
        .zip(vals.as_ref())
        .map(|(m, v)| (v.as_slice(), m.width() as u32, m.height() as u32));
    result(
        &[("max", native.score), ("pnorm3", native.pnorm_3)],
        map,
        opts,
    )
}
struct ButterWarm(butteraugli::ButteraugliReference);
impl WarmReference for ButterWarm {
    fn evaluate(&mut self, d: Image<'_>, opts: EvalOpts<'_>) -> MetricResult<Box<dyn Evaluation>> {
        let d = iqa::rgb8_view(&d)?;
        let native = self
            .0
            .compare_with_stop(&iqa::contiguous_rgb8(&d), opts.stop())
            .metric_err_with("butteraugli", butter_error)?;
        butter_result(native, opts)
    }
    fn reuses_reference_work(&self) -> bool {
        true
    }
}
/// SSIMULACRA2 native whole-image scalar; no spatial map is exposed by this API.
pub struct Ssim2;
impl Metric for Ssim2 {
    fn id(&self) -> MetricId {
        id("ssim2", "0.9.0", "aaf7705b")
            .with_config("kernel", "simd")
            .with_config("strips", false)
    }
    fn outputs(&self) -> Cow<'static, [ScoreSpec]> {
        Cow::Owned(vec![ScoreSpec::new("score", ScoreKind::Quality100)])
    }
    fn accepts(&self, f: &InputFormat) -> bool {
        iqa::is_rgb8_srgb(f)
    }
    fn evaluate(
        &mut self,
        r: Image<'_>,
        d: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn Evaluation>> {
        opts.refuse_map("ssim2")?;
        let value = fast_ssim2::compute_ssimulacra2_with_config(
            pixel(&r)?,
            pixel(&d)?,
            &fast_ssim2::Ssimulacra2Config::default().with_stop(opts.stop()),
        )
        .metric_err_with("ssim2", ssim_error)?;
        result(&[("score", value)], None, opts)
    }
    fn warm_reference<'m>(
        &'m mut self,
        r: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn WarmReference + 'm>> {
        let native = fast_ssim2::Ssimulacra2Reference::new_with_config(
            pixel(&r)?,
            &fast_ssim2::Ssimulacra2Config::default().with_stop(opts.stop()),
        )
        .metric_err_with("ssim2", ssim_error)?;
        Ok(Box::new(SsimWarm(native)))
    }
}
struct SsimWarm(fast_ssim2::Ssimulacra2Reference);
impl WarmReference for SsimWarm {
    fn evaluate(&mut self, d: Image<'_>, opts: EvalOpts<'_>) -> MetricResult<Box<dyn Evaluation>> {
        opts.refuse_map("ssim2")?;
        let value = self
            .0
            .compare_with_config(
                pixel(&d)?,
                &fast_ssim2::Ssimulacra2Config::default().with_stop(opts.stop()),
            )
            .metric_err_with("ssim2", ssim_error)?;
        result(&[("score", value)], None, opts)
    }
    fn reuses_reference_work(&self) -> bool {
        true
    }
}
/// The serving repository's adopted A profile, not the research checkout.
pub struct Zen;
fn zensim(opts: EvalOpts<'_>) -> MetricResult<zensim::Zensim> {
    // Replace operation-local control on each call; prepared work retains no pulse.
    let shared = opts
        .pulse
        .share()
        .metric_err("zensim", ErrorKind::UnsupportedRequest)?;
    Ok(zensim::Zensim::new(zensim::ZensimProfile::latest())
        .with_parallel(false)
        .with_stop(shared))
}
fn source<'a>(image: &'a Image<'_>) -> MetricResult<zensim::ZenpixelsSource<'a>> {
    zensim::ZenpixelsSource::try_from_slice(pixel(image)?)
        .metric_err("zensim", ErrorKind::UnsupportedInput)
}
impl Metric for Zen {
    fn id(&self) -> MetricId {
        id("zensim", "0.3.0", "5475929d")
            .with_config("profile", zensim::ZensimProfile::latest().name())
            .with_config("parallel", false)
            .with_config("diffmap", "default")
    }
    fn outputs(&self) -> Cow<'static, [ScoreSpec]> {
        Cow::Owned(vec![ScoreSpec::new("score", ScoreKind::Quality100)])
    }
    fn accepts(&self, f: &InputFormat) -> bool {
        iqa::is_rgb8_srgb(f)
    }
    fn evaluate(
        &mut self,
        r: Image<'_>,
        d: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn Evaluation>> {
        let z = zensim(opts)?;
        let r = source(&r)?;
        let d = source(&d)?;
        if opts.map.is_some() {
            let native = z
                .compute_with_diffmap(&r, &d, zensim::DiffmapOptions::default())
                .metric_err_with("zensim", zensim_error)?;
            result(
                &[("score", native.score())],
                Some((
                    native.diffmap(),
                    native.width() as u32,
                    native.height() as u32,
                )),
                opts,
            )
        } else {
            let native = z.compute(&r, &d).metric_err_with("zensim", zensim_error)?;
            result(&[("score", native.score())], None, opts)
        }
    }
    fn warm_reference<'m>(
        &'m mut self,
        r: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn WarmReference + 'm>> {
        let native = zensim(opts)?
            .precompute_reference(&source(&r)?)
            .metric_err_with("zensim", zensim_error)?;
        Ok(Box::new(ZenWarm(native)))
    }
}
struct ZenWarm(zensim::PrecomputedReference);
impl WarmReference for ZenWarm {
    fn evaluate(&mut self, d: Image<'_>, opts: EvalOpts<'_>) -> MetricResult<Box<dyn Evaluation>> {
        let z = zensim(opts)?;
        let d = source(&d)?;
        if opts.map.is_some() {
            let n = z
                .compute_with_ref_and_diffmap(&self.0, &d, zensim::DiffmapOptions::default())
                .metric_err_with("zensim", zensim_error)?;
            result(
                &[("score", n.score())],
                Some((n.diffmap(), n.width() as u32, n.height() as u32)),
                opts,
            )
        } else {
            let n = z
                .compute_with_ref(&self.0, &d)
                .metric_err_with("zensim", zensim_error)?;
            result(&[("score", n.score())], None, opts)
        }
    }
    fn reuses_reference_work(&self) -> bool {
        true
    }
}
/// Deterministic textured native samples; padding is a distinct poison value.
pub fn fixture(w: u32, h: u32, padding: usize, distorted: bool) -> Vec<u8> {
    let stride = w as usize * 3 + padding;
    let mut p = vec![0xe7; (h as usize - 1) * stride + w as usize * 3];
    for y in 0..h as usize {
        for x in 0..w as usize {
            for c in 0..3 {
                let v = ((x * 31 + y * 47 + c * 63 + (x * y) % 71) % 190 + 25) as u8;
                p[y * stride + x * 3 + c] = if distorted && (x + 3 * y + c) % 7 == 0 {
                    v.saturating_add(17)
                } else {
                    v
                };
            }
        }
    }
    p
}
