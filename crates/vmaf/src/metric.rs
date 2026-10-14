//! VMAF v1.0.16 on the zenmetrics-abstractions surface: a temporal metric.
//!
//! Each frame's `motion3` depends on the next frame (and, for HFR models,
//! the one after), so [`VmafV1Stream::push`] returns results late: zero,
//! one or two per push. [`Vmaf::sequence`](za::Metric::sequence) exposes
//! that directly; [`za::Metric::evaluate`] scores one pair as a one-frame
//! sequence (motion 0, as libvmaf scores a single frame).
//!
//! Input is planar YCbCr 4:2:0 (`PlaneLayout::ycbcr_420`) with U8 samples,
//! or U16 samples carrying 10 significant bits. VMAF is defined on code
//! values: transfer, primaries and signal range are used as given, exactly
//! as libvmaf does, and are not checked.

use std::borrow::Cow;
use std::boxed::Box;
use std::vec::Vec;

use zenmetrics_abstractions as za;
use za::zenpixels::{ChannelType, PixelSlice, PlaneSemantic};

use crate::{Error, ModelVariant, PoolingMethod, VmafFrameResult, VmafV1Stream, Yuv420Frame};

/// Output revision of [`Vmaf`] on the generic surface.
const SCORE_REVISION: u32 = 1;

const OUTPUTS: &[za::ScoreSpec] = &[
    za::ScoreSpec::new("vmaf", za::ScoreKind::Quality100),
    za::ScoreSpec::new("cambi", za::ScoreKind::Feature),
    za::ScoreSpec::new("speed_chroma_uv", za::ScoreKind::Feature),
    za::ScoreSpec::new("adm3", za::ScoreKind::Feature),
    za::ScoreSpec::new("motion3", za::ScoreKind::Feature),
];

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::InvalidModel(_) => za::ErrorKind::InvalidConfig,
            // The scorers' size floors (SPEED's operating size, CAMBI, ADM).
            Error::InvalidInput(m) if m.contains("too small") || m.contains("below") => za::ErrorKind::TooSmall,
            Error::NonFiniteInput(_) | Error::InvalidInput(_) => za::ErrorKind::InvalidInput,
        }
    }
}

/// VMAF v1.0.16 as a generic, temporal scorer.
#[derive(Clone, Copy, Debug)]
pub struct Vmaf {
    variant: ModelVariant,
    pooling: PoolingMethod,
}

impl Vmaf {
    /// Score with `variant`'s model, pooling sequences by their mean.
    pub fn new(variant: ModelVariant) -> Self {
        Self {
            variant,
            pooling: PoolingMethod::Mean,
        }
    }

    /// Pool sequences with `pooling` instead of the mean.
    #[must_use]
    pub fn with_pooling(mut self, pooling: PoolingMethod) -> Self {
        self.pooling = pooling;
        self
    }

    fn short_name(&self) -> &'static str {
        match self.variant {
            ModelVariant::Standard1080p => "1080p",
            ModelVariant::Phone => "phone",
            ModelVariant::Default4k => "4k",
            ModelVariant::Consumer4k => "4k_consumer",
            ModelVariant::HfrStandard1080p => "hfr_1080p",
            ModelVariant::HfrPhone => "hfr_phone",
            ModelVariant::HfrDefault4k => "hfr_4k",
            ModelVariant::HfrConsumer4k => "hfr_4k_consumer",
        }
    }

    fn distance(&self) -> u32 {
        match self.variant {
            ModelVariant::HfrStandard1080p
            | ModelVariant::HfrPhone
            | ModelVariant::HfrDefault4k
            | ModelVariant::HfrConsumer4k => 2,
            _ => 1,
        }
    }
}

/// Whether `format` is YCbCr 4:2:0 at a bit depth VMAF scores: U8 with 8
/// bits, or U16 with 10.
fn accepts_format(format: &za::InputFormat) -> bool {
    let za::InputFormat::Planar(p) = format else {
        return false;
    };
    let planes = p.layout.planes();
    let shape = [
        (PlaneSemantic::Luma, 1, 1),
        (PlaneSemantic::ChromaCb, 2, 2),
        (PlaneSemantic::ChromaCr, 2, 2),
    ];
    planes.len() == 3
        && planes
            .iter()
            .zip(shape)
            .all(|(d, (s, h, v))| d.semantic == s && d.h_subsample == h && d.v_subsample == v)
        && planes.iter().all(|d| d.channel_type == planes[0].channel_type)
        && matches!(
            (planes[0].channel_type, p.significant_bits),
            (ChannelType::U8, 8) | (ChannelType::U16, 10)
        )
}

/// One plane's samples as packed `u16`.
fn plane_u16(s: &PixelSlice<'_>) -> Vec<u16> {
    let (w, h) = (s.width() as usize, s.rows() as usize);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h as u32 {
        let row = s.row(y);
        match s.descriptor().channel_type() {
            ChannelType::U8 => out.extend(row[..w].iter().map(|&v| u16::from(v))),
            _ => out.extend(row[..w * 2].chunks_exact(2).map(|c| u16::from_ne_bytes([c[0], c[1]]))),
        }
    }
    out
}

struct Frame {
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

impl Frame {
    fn as_yuv(&self) -> Yuv420Frame<'_> {
        Yuv420Frame {
            y: &self.y,
            u: &self.u,
            v: &self.v,
        }
    }
}

/// Validate one image and widen its planes.
#[track_caller]
fn frame(img: &za::Image<'_>) -> za::MetricResult<(Frame, (u32, u32, u8))> {
    if !accepts_format(&img.format()) {
        return za::fail(
            za::unsupported(&img.format(), "planar YCbCr 4:2:0, U8 (8-bit) or U16 (10-bit)").with_metric("vmaf"),
        );
    }
    let Some(p) = img.as_planar() else {
        return za::fail(za::MetricError::new(za::ErrorKind::Internal).with_metric("vmaf"));
    };
    let s = p.planes();
    Ok((
        Frame {
            y: plane_u16(&s[0]),
            u: plane_u16(&s[1]),
            v: plane_u16(&s[2]),
        },
        (p.width(), p.height(), p.significant_bits()),
    ))
}

fn evaluation(r: &VmafFrameResult) -> Box<dyn za::Evaluation> {
    let f = &r.features;
    Box::new(
        za::ScalarEval::single("vmaf", r.score)
            .with("cambi", f.cambi)
            .with("speed_chroma_uv", f.speed_chroma_uv)
            .with("adm3", f.adm3)
            .with("motion3", f.motion3),
    )
}

impl za::Metric for Vmaf {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("vmaf", za::implementation!(SCORE_REVISION))
            .with_variant(self.short_name())
            .with_reference("libvmaf v3.2.1")
            .with_config("model", self.variant.built_in_name())
            .with_config("pooling", format!("{:?}", self.pooling))
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        accepts_format(format)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        opts.checkpoint()?;
        opts.refuse_map("vmaf")?;
        let mut seq = za::Metric::sequence(self, za::SequenceInfo::new().with_frame_count(1))?;
        let mut frames = seq.push(r, d, opts)?;
        frames.extend(seq.finish(opts)?.frames);
        match frames.pop() {
            Some(f) if frames.is_empty() && f.index == 0 => Ok(f.evaluation),
            _ => za::fail(za::MetricError::new(za::ErrorKind::Internal).with_metric("vmaf")),
        }
    }

    fn temporal(&self) -> za::Temporal {
        let distance = self.distance();
        za::Temporal::Stateful {
            history: distance,
            latency: distance,
        }
    }

    fn sequence<'m>(&'m mut self, _info: za::SequenceInfo) -> za::MetricResult<Box<dyn za::Sequence + 'm>> {
        Ok(Box::new(VmafSequence {
            vmaf: *self,
            stream: None,
            shape: None,
            pushed: 0,
            emitted: 0,
            scores: Vec::new(),
            failed: false,
        }))
    }
}

/// A [`Vmaf`] sequence session over [`VmafV1Stream`].
struct VmafSequence {
    vmaf: Vmaf,
    stream: Option<VmafV1Stream>,
    shape: Option<(u32, u32, u8)>,
    pushed: u64,
    emitted: u64,
    scores: Vec<f64>,
    failed: bool,
}

impl VmafSequence {
    fn ready(&mut self, results: Vec<VmafFrameResult>) -> Vec<za::FrameEvaluation> {
        results
            .iter()
            .map(|r| {
                self.scores.push(r.score);
                let f = za::FrameEvaluation::new(self.emitted, evaluation(r));
                self.emitted += 1;
                f
            })
            .collect()
    }

    fn push_inner(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Vec<za::FrameEvaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map("vmaf")?;
        let (rf, shape) = frame(&r)?;
        let (df, dshape) = frame(&d)?;
        if dshape != shape {
            return za::fail(za::MetricError::new(za::ErrorKind::DimensionMismatch).with_metric("vmaf").with_message(
                format!("reference {shape:?}, distorted {dshape:?} (width, height, bits)"),
            ));
        }
        match self.shape {
            Some(s) if s != shape => {
                return za::fail(za::MetricError::new(za::ErrorKind::DimensionMismatch).with_metric("vmaf").with_message(
                    format!("frame {} is {shape:?}, the sequence started {s:?}", self.pushed),
                ));
            }
            Some(_) => {}
            None => {
                let (w, h, bits) = shape;
                let (wu, hu) = (w as usize, h as usize);
                if wu < crate::adm::ADM_MIN_DIM
                    || hu < crate::adm::ADM_MIN_DIM
                    || (wu < crate::cambi::CAMBI_MIN_WIDTH_HEIGHT && hu < crate::cambi::CAMBI_MIN_WIDTH_HEIGHT)
                {
                    return za::fail(za::MetricError::new(za::ErrorKind::TooSmall).with_metric("vmaf").with_message(format!(
                        "{w}x{h}: VMAF needs both sides >= {} and one side >= {}",
                        crate::adm::ADM_MIN_DIM,
                        crate::cambi::CAMBI_MIN_WIDTH_HEIGHT
                    )));
                }
                self.stream = Some(
                    VmafV1Stream::new(w as usize, h as usize, bits, self.vmaf.variant).metric_err_with("vmaf", Error::kind)?,
                );
                self.shape = Some(shape);
            }
        }
        let Some(stream) = self.stream.as_mut() else {
            return za::fail(za::MetricError::new(za::ErrorKind::Internal).with_metric("vmaf"));
        };
        let results = stream.push(rf.as_yuv(), df.as_yuv()).metric_err_with("vmaf", Error::kind)?;
        self.pushed += 1;
        Ok(self.ready(results))
    }
}

impl za::Sequence for VmafSequence {
    fn push(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Vec<za::FrameEvaluation>> {
        if self.failed {
            return za::fail(
                za::MetricError::new(za::ErrorKind::SequenceOrder)
                    .with_metric("vmaf")
                    .with_message("push after a failed push"),
            );
        }
        let out = self.push_inner(r, d, opts);
        self.failed = out.is_err();
        out
    }

    fn finish(mut self: Box<Self>, opts: za::EvalOpts<'_>) -> za::MetricResult<za::SequenceSummary> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        if self.failed {
            return za::fail(
                za::MetricError::new(za::ErrorKind::SequenceOrder)
                    .with_metric("vmaf")
                    .with_message("finish after a failed push"),
            );
        }
        let Some(stream) = self.stream.take() else {
            return za::fail(
                za::MetricError::new(za::ErrorKind::SequenceOrder).with_metric("vmaf").with_message("finish with no frames"),
            );
        };
        let rest = stream.finish().metric_err_with("vmaf", Error::kind)?;
        let frames = self.ready(rest);
        let pooled = crate::pool_v1_scores(&self.scores, self.vmaf.pooling).metric_err_with("vmaf", Error::kind)?;
        Ok(za::SequenceSummary::new(
            frames,
            Some(Box::new(za::ScalarEval::single("vmaf", pooled))),
            self.pushed,
        ))
    }

    fn frames_pushed(&self) -> u64 {
        self.pushed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use za::Metric as _;
    use za::zenpixels::{PixelDescriptor, PlaneLayout};

    // The libvmaf parity tests' size: above every feature's floor.
    const W: usize = 640;
    const H: usize = 360;

    fn gray8(d: &[u8], w: usize, h: usize) -> PixelSlice<'_> {
        PixelSlice::new(d, w as u32, h as u32, w, PixelDescriptor::GRAY8).unwrap()
    }

    fn gray16(d: &[u8], w: usize, h: usize) -> PixelSlice<'_> {
        PixelSlice::new(d, w as u32, h as u32, w * 2, PixelDescriptor::GRAY16).unwrap()
    }

    fn yuv(p: &[Vec<u16>; 3]) -> Yuv420Frame<'_> {
        Yuv420Frame { y: &p[0], u: &p[1], v: &p[2] }
    }

    /// One synthetic 8-bit 4:2:0 frame: a moving gradient plus texture.
    fn yuv8(t: usize, noise: u8) -> [Vec<u8>; 3] {
        let y = (0..W * H)
            .map(|i| {
                let (x, yy) = (i % W, i / W);
                (((x + 3 * t) * 3 + yy * 2) % 220 + (i * 7) % 11 + usize::from(i % 5 == 0) * usize::from(noise)).min(255) as u8
            })
            .collect();
        let c = |k: usize| (0..(W / 2) * (H / 2)).map(|i| (100 + (i * k + t) % 50) as u8).collect();
        [y, c(3), c(5)]
    }

    fn image(p: &[Vec<u8>; 3]) -> za::Image<'_> {
        za::Planes::new(
            PlaneLayout::ycbcr_420(ChannelType::U8),
            vec![gray8(&p[0], W, H), gray8(&p[1], W / 2, H / 2), gray8(&p[2], W / 2, H / 2)],
        )
        .unwrap()
        .into()
    }

    fn to_u16(p: &[Vec<u8>; 3]) -> [Vec<u16>; 3] {
        p.clone().map(|v| v.into_iter().map(u16::from).collect())
    }

    #[test]
    fn single_pair_conforms() {
        let (r, d) = (yuv8(0, 0), yuv8(0, 40));
        za::conformance::assert_conforms(&mut Vmaf::new(ModelVariant::Standard1080p), &image(&r), &image(&d));
    }

    /// Every pushed frame comes back once, in order, within the declared
    /// latency, bit-identical to driving `VmafV1Stream` directly; the pooled
    /// score is `pool_v1_scores` over them.
    #[test]
    fn sequence_matches_the_stream_and_respects_latency() {
        for variant in [ModelVariant::Standard1080p, ModelVariant::HfrStandard1080p] {
            let frames: Vec<_> = (0..6).map(|t| (yuv8(t, 0), yuv8(t, 30 + t as u8))).collect();
            let images: Vec<_> = frames.iter().map(|(r, d)| (image(r), image(d))).collect();
            let mut m = Vmaf::new(variant);
            let report = za::conformance::check_sequence(&mut m, &images);
            assert!(report.violations.is_empty(), "{:?}", report.violations);

            let mut stream = VmafV1Stream::new(W, H, 8, variant).unwrap();
            let mut want = Vec::new();
            for (r, d) in &frames {
                let (r, d) = (to_u16(r), to_u16(d));
                want.extend(stream.push(yuv(&r), yuv(&d)).unwrap().into_iter().map(|x| x.score));
            }
            want.extend(stream.finish().unwrap().into_iter().map(|x| x.score));
            let got: Vec<f64> = report.frames.iter().map(|(_, s)| s[0].1).collect();
            assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            let pooled = crate::pool_v1_scores(&want, PoolingMethod::Mean).unwrap();
            assert_eq!(report.pooled.unwrap()[0].1.to_bits(), pooled.to_bits());
        }
    }

    /// 10-bit content rides U16 planes with `significant_bits = 10`; 16
    /// significant bits is refused rather than rescaled.
    #[test]
    fn ten_bit_planes_are_accepted_sixteen_refused() {
        let p = to_u16(&yuv8(0, 0)).map(|v| v.into_iter().map(|s| s * 4).collect::<Vec<u16>>());
        let bytes: Vec<Vec<u8>> = p.iter().map(|v| v.iter().flat_map(|s| s.to_ne_bytes()).collect()).collect();
        let planes = || {
            za::Planes::new(
                PlaneLayout::ycbcr_420(ChannelType::U16),
                vec![gray16(&bytes[0], W, H), gray16(&bytes[1], W / 2, H / 2), gray16(&bytes[2], W / 2, H / 2)],
            )
            .unwrap()
        };
        let mut m = Vmaf::new(ModelVariant::Standard1080p);
        let ten: za::Image<'_> = planes().with_significant_bits(10).unwrap().into();
        assert!(m.accepts(&ten.format()));
        let e = m.evaluate(ten.clone(), ten, za::EvalOpts::SCORE_ONLY).unwrap();
        assert!(e.score().is_finite());
        let sixteen: za::Image<'_> = planes().into();
        let err = m.evaluate(sixteen.clone(), sixteen, za::EvalOpts::SCORE_ONLY).err().unwrap();
        assert_eq!(err.error().kind(), za::ErrorKind::UnsupportedInput);
    }

    #[test]
    fn mid_sequence_size_change_fails_and_poisons_the_session() {
        let mut m = Vmaf::new(ModelVariant::Standard1080p);
        let mut seq = m.sequence(za::SequenceInfo::new()).unwrap();
        let (r, d) = (yuv8(0, 0), yuv8(0, 9));
        seq.push(image(&r), image(&d), za::EvalOpts::SCORE_ONLY).unwrap();
        fn small(p: &[Vec<u8>; 3]) -> za::Image<'_> {
            za::Image::from(
                za::Planes::new(
                    PlaneLayout::ycbcr_420(ChannelType::U8),
                    vec![gray8(&p[0], W, H - 2), gray8(&p[1], W / 2, H / 2 - 1), gray8(&p[2], W / 2, H / 2 - 1)],
                )
                .unwrap(),
            )
        }
        let e = seq.push(small(&r), small(&d), za::EvalOpts::SCORE_ONLY).err().unwrap();
        assert_eq!(e.error().kind(), za::ErrorKind::DimensionMismatch);
        let e = seq.push(image(&r), image(&d), za::EvalOpts::SCORE_ONLY).err().unwrap();
        assert_eq!(e.error().kind(), za::ErrorKind::SequenceOrder);
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;
    use za::Metric as _;
    use za::zenpixels::{PixelDescriptor, PlaneLayout};

    #[test]
    fn below_cambi_minimum_is_too_small() {
        let (w, h) = (64usize, 48usize);
        let (y, c) = (vec![128u8; w * h], vec![128u8; (w / 2) * (h / 2)]);
        fn g(d: &[u8], w: usize, h: usize) -> PixelSlice<'_> {
            PixelSlice::new(d, w as u32, h as u32, w, PixelDescriptor::GRAY8).unwrap()
        }
        let img: za::Image<'_> =
            za::Planes::new(PlaneLayout::ycbcr_420(ChannelType::U8), vec![g(&y, w, h), g(&c, w / 2, h / 2), g(&c, w / 2, h / 2)])
                .unwrap()
                .into();
        let e = Vmaf::new(ModelVariant::Standard1080p).evaluate(img.clone(), img, za::EvalOpts::SCORE_ONLY).err().unwrap();
        assert_eq!(e.error().kind(), za::ErrorKind::TooSmall);
    }
}
