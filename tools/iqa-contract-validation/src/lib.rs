//! Source-pinned native adapters for the minimal typed IQA contract.
#![forbid(unsafe_code)]
use iqa::zenpixels::{PixelDescriptor, PixelSlice};
use iqa::{
    Direction, ErrorKind, EvalOpts, Evaluation, Implementation, Metric, MetricId, MetricResult,
    PrepareReference, ResultMetricExt, Score, ScoreSpec, WarmReference,
};
use zenpsy_api as iqa;

/// Typed native assessment, retaining the original result without map copies.
pub struct Assessment<T, const N: usize> {
    pub native: T,
    pub values: [f64; N],
    specs: [ScoreSpec; N],
}
impl<T, const N: usize> std::fmt::Debug for Assessment<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Assessment")
            .field("values", &self.values)
            .finish_non_exhaustive()
    }
}
impl<T, const N: usize> Evaluation for Assessment<T, N> {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        self.specs
            .iter()
            .zip(self.values)
            .map(|(s, v)| Score::new(s, v))
    }
}
fn one<T>(native: T, value: f64) -> Assessment<T, 1> {
    Assessment {
        native,
        values: [value],
        specs: [ScoreSpec::new("score", Direction::Higher).with_perfect(100.0)],
    }
}
fn id(family: &'static str, version: &'static str, source: &'static str) -> MetricId {
    MetricId::new(family, Implementation::new(family, version, 1)).with_reference(source)
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
fn zen_error(e: &zensim::ZensimError) -> ErrorKind {
    match e {
        zensim::ZensimError::Cancelled { reason: r } => ErrorKind::Stopped(*r),
        _ => ErrorKind::Backend,
    }
}
fn rgb8<'a>(p: &'a PixelSlice<'_>) -> MetricResult<butteraugli::ImgRef<'a, butteraugli::RGB8>> {
    if p.descriptor() != PixelDescriptor::RGB8_SRGB || p.color_context().is_some() {
        return iqa::fail(iqa::MetricError::new(ErrorKind::UnsupportedInput));
    }
    Ok(butteraugli::Img::new_stride(
        bytemuck::cast_slice(p.as_strided_bytes()),
        p.width() as usize,
        p.rows() as usize,
        p.stride() / 3,
    ))
}
fn packed_rgb8(p: &PixelSlice<'_>) -> MetricResult<Vec<u8>> {
    rgb8(p)?;
    Ok((0..p.rows())
        .flat_map(|y| p.row(y).iter().copied())
        .collect())
}
/// Butteraugli's two scores and its original optional diagnostic result.
pub struct Butter;
pub struct ButterWarm(butteraugli::ButteraugliReference);
fn butter_result(
    native: butteraugli::ButteraugliResult,
) -> Assessment<butteraugli::ButteraugliResult, 2> {
    Assessment {
        values: [native.score, native.pnorm_3],
        native,
        specs: [
            ScoreSpec::new("max", Direction::Lower).with_perfect(0.0),
            ScoreSpec::new("pnorm3", Direction::Lower).with_perfect(0.0),
        ],
    }
}
impl Metric for Butter {
    type Output = Assessment<butteraugli::ButteraugliResult, 2>;
    fn id(&self) -> MetricId {
        id("butteraugli", "0.9.4", "31687022")
            .with_config("intensity_target", 80)
            .with_config("hf_asymmetry", 1)
            .with_config("xmul", 1)
    }
    fn evaluate(
        &mut self,
        r: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        butteraugli::butteraugli_with_stop(
            rgb8(r)?,
            rgb8(d)?,
            &butteraugli::ButteraugliParams::default().with_compute_diffmap(opts.diagnostics()),
            opts.stop(),
        )
        .metric_err_with("butteraugli", butter_error)
        .map(butter_result)
    }
}
impl PrepareReference for Butter {
    type Prepared<'a> = ButterWarm;
    fn warm_reference(
        &mut self,
        r: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<ButterWarm> {
        opts.checkpoint()?;
        let n = butteraugli::ButteraugliReference::new(
            &packed_rgb8(r)?,
            r.width() as usize,
            r.rows() as usize,
            butteraugli::ButteraugliParams::default().with_compute_diffmap(opts.diagnostics()),
        )
        .metric_err_with("butteraugli", butter_error)?;
        Ok(ButterWarm(n))
    }
}
impl WarmReference for ButterWarm {
    type Output = Assessment<butteraugli::ButteraugliResult, 2>;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Self::Output> {
        let mut n = self
            .0
            .compare_with_stop(&packed_rgb8(d)?, opts.stop())
            .metric_err_with("butteraugli", butter_error)?;
        // The native warm API returns its pooling map even when compute_diffmap
        // is false. It is algorithm scratch; do not retain it as diagnostics.
        if !opts.diagnostics() {
            n.diffmap = None;
        }
        Ok(butter_result(n))
    }
}
/// SSIMULACRA2 keeps its scalar or complete native maps, without regridding.
pub enum SsimNative {
    Scalar(f64),
    Maps(fast_ssim2::Ssimulacra2Maps),
}
pub struct Ssim2;
pub struct SsimWarm(fast_ssim2::Ssimulacra2Reference);
impl Metric for Ssim2 {
    type Output = Assessment<SsimNative, 1>;
    fn id(&self) -> MetricId {
        id("ssim2", fast_ssim2::VERSION, "6d27c728")
            .with_config("kernel", "simd")
            .with_config("strips", false)
    }
    fn evaluate(
        &mut self,
        r: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let config = fast_ssim2::Ssimulacra2Config::default().with_stop(opts.stop());
        if opts.diagnostics() {
            let n = fast_ssim2::compute_ssimulacra2_maps_with_config(r, d, &config)
                .metric_err_with("ssim2", ssim_error)?;
            let v = n.score;
            Ok(one(SsimNative::Maps(n), v))
        } else {
            let v = fast_ssim2::compute_ssimulacra2_with_config(r, d, &config)
                .metric_err_with("ssim2", ssim_error)?;
            Ok(one(SsimNative::Scalar(v), v))
        }
    }
}
impl PrepareReference for Ssim2 {
    type Prepared<'a> = SsimWarm;
    fn warm_reference(&mut self, r: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<SsimWarm> {
        fast_ssim2::Ssimulacra2Reference::new_with_config(
            r,
            &fast_ssim2::Ssimulacra2Config::default().with_stop(opts.stop()),
        )
        .metric_err_with("ssim2", ssim_error)
        .map(SsimWarm)
    }
}
impl WarmReference for SsimWarm {
    type Output = Assessment<SsimNative, 1>;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Self::Output> {
        let config = fast_ssim2::Ssimulacra2Config::default().with_stop(opts.stop());
        if opts.diagnostics() {
            let n = self
                .0
                .compare_with_maps_with_config(d, &config)
                .metric_err_with("ssim2", ssim_error)?;
            let v = n.score;
            Ok(one(SsimNative::Maps(n), v))
        } else {
            let v = self
                .0
                .compare_with_config(d, &config)
                .metric_err_with("ssim2", ssim_error)?;
            Ok(one(SsimNative::Scalar(v), v))
        }
    }
}
/// Serving Zensim's original structured scalar/diagnostic result.
pub enum ZenNative {
    Scalar(zensim::ZensimResult),
    Map(zensim::DiffmapResult),
}
#[derive(Default)]
pub struct Zen {
    pub hdr: Option<zensim::HdrEncoding>,
}
pub struct ZenWarm(zensim::PrecomputedReference);
fn zensim(opts: EvalOpts<'_>) -> MetricResult<zensim::Zensim> {
    let shared = opts
        .pulse
        .share()
        .metric_err("zensim", ErrorKind::UnsupportedRequest)?;
    Ok(zensim::Zensim::new(zensim::ZensimProfile::latest())
        .with_parallel(false)
        .with_stop(shared))
}
impl Metric for Zen {
    type Output = Assessment<ZenNative, 1>;
    fn id(&self) -> MetricId {
        id("zensim", "0.3.0", "5475929d")
            .with_config("profile", zensim::ZensimProfile::latest().name())
            .with_config("parallel", false)
            .with_config("hdr", format!("{:?}", self.hdr))
    }
    fn evaluate(
        &mut self,
        r: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let z = zensim(opts)?;
        if let Some(encoding) = self.hdr {
            if opts.diagnostics() {
                return iqa::fail(iqa::MetricError::new(ErrorKind::NoMap));
            }
            let n = z
                .compute_hdr(
                    &hdr_source(r, encoding)?,
                    &hdr_source(d, encoding)?,
                    encoding,
                )
                .metric_err_with("zensim", zen_error)?;
            let v = n.score();
            return Ok(one(ZenNative::Scalar(n), v));
        }
        let r = zensim::ZenpixelsSource::try_from_slice(r)
            .metric_err("zensim", ErrorKind::UnsupportedInput)?;
        let d = zensim::ZenpixelsSource::try_from_slice(d)
            .metric_err("zensim", ErrorKind::UnsupportedInput)?;
        if opts.diagnostics() {
            let n = z
                .compute_with_diffmap(&r, &d, zensim::DiffmapOptions::default())
                .metric_err_with("zensim", zen_error)?;
            let v = n.score();
            Ok(one(ZenNative::Map(n), v))
        } else {
            let n = z.compute(&r, &d).metric_err_with("zensim", zen_error)?;
            let v = n.score();
            Ok(one(ZenNative::Scalar(n), v))
        }
    }
}
impl PrepareReference for Zen {
    type Prepared<'a> = ZenWarm;
    fn warm_reference(&mut self, r: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<ZenWarm> {
        if self.hdr.is_some() {
            return iqa::fail(iqa::MetricError::new(ErrorKind::UnsupportedRequest));
        }
        let source = zensim::ZenpixelsSource::try_from_slice(r)
            .metric_err("zensim", ErrorKind::UnsupportedInput)?;
        zensim(opts)?
            .precompute_reference(&source)
            .metric_err_with("zensim", zen_error)
            .map(ZenWarm)
    }
}
impl WarmReference for ZenWarm {
    type Output = Assessment<ZenNative, 1>;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<Self::Output> {
        let z = zensim(opts)?;
        let d = zensim::ZenpixelsSource::try_from_slice(d)
            .metric_err("zensim", ErrorKind::UnsupportedInput)?;
        if opts.diagnostics() {
            let n = z
                .compute_with_ref_and_diffmap(&self.0, &d, zensim::DiffmapOptions::default())
                .metric_err_with("zensim", zen_error)?;
            let v = n.score();
            Ok(one(ZenNative::Map(n), v))
        } else {
            let n = z
                .compute_with_ref(&self.0, &d)
                .metric_err_with("zensim", zen_error)?;
            let v = n.score();
            Ok(one(ZenNative::Scalar(n), v))
        }
    }
}
/// Explicit HDR ingress. Linear samples are absolute cd/m², never normalized
/// relative light. The validation adapter refuses attached override contexts.
pub fn hdr_source<'a>(
    p: &'a PixelSlice<'_>,
    encoding: zensim::HdrEncoding,
) -> MetricResult<zensim::StridedBytes<'a>> {
    use iqa::zenpixels::{AlphaMode, ColorPrimaries, PixelFormat, SignalRange, TransferFunction};
    let desc = p.descriptor();
    let transfer = match encoding {
        zensim::HdrEncoding::Linear => TransferFunction::Linear,
        zensim::HdrEncoding::Pq { .. } => TransferFunction::Pq,
        zensim::HdrEncoding::Hlg { .. } => TransferFunction::Hlg,
        _ => return iqa::fail(iqa::MetricError::new(ErrorKind::UnsupportedInput)),
    };
    if desc.pixel_format() != PixelFormat::RgbaF32
        || desc.transfer() != transfer
        || desc.alpha() != Some(AlphaMode::Opaque)
        || desc.signal_range != SignalRange::Full
        || p.color_context().is_some()
    {
        return iqa::fail(iqa::MetricError::new(ErrorKind::UnsupportedInput));
    }
    let primaries = match desc.primaries {
        ColorPrimaries::Bt709 => zensim::ColorPrimaries::Srgb,
        ColorPrimaries::DisplayP3 => zensim::ColorPrimaries::DisplayP3,
        ColorPrimaries::Bt2020 => zensim::ColorPrimaries::Bt2020,
        _ => return iqa::fail(iqa::MetricError::new(ErrorKind::UnsupportedInput)),
    };
    zensim::StridedBytes::try_with_alpha_mode(
        p.as_strided_bytes(),
        p.width() as usize,
        p.rows() as usize,
        p.stride(),
        zensim::PixelFormat::LinearF32Rgba,
        zensim::AlphaMode::Opaque,
    )
    .map(|s| s.with_color_primaries(primaries))
    .metric_err_with("zensim", zen_error)
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
