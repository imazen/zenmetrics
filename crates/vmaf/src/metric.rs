//! VMAF v1.0.16 on the zenmetrics-abstractions surface: a temporal metric.
//!
//! Each frame's `motion3` depends on the next frame (and, for HFR models,
//! the one after), so [`VmafV1Stream::push`] returns results late: zero,
//! one or two per push. [`Vmaf::sequence`](za::Metric::sequence) exposes
//! that directly; [`za::Metric::evaluate`] scores one pair as a one-frame
//! sequence (motion 0, as libvmaf scores a single frame).
//!
//! Maps, on request, for a single pair ([`za::Metric::evaluate`]; sequences
//! refuse map requests), each one feature's components, not a per-pixel
//! VMAF; SPEED and AIM maps are [`za::MapPolarity::Difference`]:
//!
//! - SPEED's chroma block maps `speed_u` (primary) and `speed_v`, each
//!   block's `|S_ref − S_dist|` entropy-weighted difference, the terms whose
//!   block mean is that channel's SPEED score before the U/V combine and the
//!   45 clamp. A block is 5×5 operating samples (every 16th sample of the
//!   prescaled chroma plane): 160 luma pixels at prescale 1, 320 at 0.5. The
//!   phone models' 0.6 prescale has no whole-pixel grid, so they serve no
//!   SPEED maps. Support is global (frame-wide covariance eigenvalues).
//! - ADM3's AIM maps `aim_{s}_{h,v,d}` for scales 0–3 and the horizontal,
//!   vertical and diagonal bands: per band sample, the cube of the additive
//!   impairment left visible after masking by the restored signal, the
//!   integer terms whose cube-rooted band sums form AIM's numerator (`adm3`
//!   blends `1 − AIM` with the detail-loss ratio). Values are in the units of
//!   the band's cube-root argument. Scale `s` samples `2^(s+1)`-pixel blocks
//!   from the origin, cropped to the pooled inner region; support is global.
//!   They are the primary maps of the phone models.
//! - CAMBI's `cambi_0` … `cambi_4`: per-scale c-values of the distorted
//!   frame, the values its top-k pooling reads and libvmaf's `heatmaps_path`
//!   dumps. CAMBI scores banding in the distorted frame alone, so these carry
//!   no reference comparison ([`za::MapPolarity::Feature`]; higher = more
//!   visible banding).
//!
//! Motion is not exposed as maps.
//!
//! Input is planar YCbCr 4:2:0 (`PlaneLayout::ycbcr_420`) with U8 samples,
//! or U16 samples carrying 10 significant bits. VMAF is defined on code
//! values: transfer, primaries and signal range are used as given, exactly
//! as libvmaf does, and are not checked.

use std::borrow::Cow;
use std::boxed::Box;
use std::vec::Vec;

use za::PlaneRole;
use za::zenpixels::{ChannelType, PixelSlice};
use zenmetrics_abstractions as za;

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
            Error::InvalidInput(m) if m.contains("too small") || m.contains("below") => {
                za::ErrorKind::TooSmall
            }
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
        (PlaneRole::Luma, 1, 1),
        (PlaneRole::Cb, 2, 2),
        (PlaneRole::Cr, 2, 2),
    ];
    planes.len() == 3
        && planes
            .iter()
            .zip(shape)
            .all(|(d, (s, h, v))| d.role == s && d.h_subsample == h && d.v_subsample == v)
        && planes
            .iter()
            .all(|d| d.channel_type == planes[0].channel_type)
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
            _ => out.extend(
                row[..w * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_ne_bytes(*c)),
            ),
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
            za::unsupported(
                &img.format(),
                "planar YCbCr 4:2:0, U8 (8-bit) or U16 (10-bit)",
            )
            .with_metric("vmaf"),
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

fn scalars(r: &VmafFrameResult) -> za::ScalarEval {
    let f = &r.features;
    za::ScalarEval::single("vmaf", r.score)
        .with("cambi", f.cambi)
        .with("speed_chroma_uv", f.speed_chroma_uv)
        .with("adm3", f.adm3)
        .with("motion3", f.motion3)
}

fn evaluation(r: &VmafFrameResult) -> Box<dyn za::Evaluation> {
    Box::new(scalars(r))
}

/// A single scored pair with SPEED's chroma block maps.
struct VmafMaps {
    scalars: za::ScalarEval,
    maps: Vec<(&'static str, Vec<f32>, za::MapGeometry, za::MapPolarity)>,
}

/// CAMBI c-value map names, one per scale.
const CAMBI_NAMES: [&str; 5] = ["cambi_0", "cambi_1", "cambi_2", "cambi_3", "cambi_4"];

/// AIM map names by scale and band (horizontal, vertical, diagonal).
const AIM_NAMES: [[&str; 3]; 4] = [
    ["aim_0_h", "aim_0_v", "aim_0_d"],
    ["aim_1_h", "aim_1_v", "aim_1_d"],
    ["aim_2_h", "aim_2_v", "aim_2_d"],
    ["aim_3_h", "aim_3_v", "aim_3_d"],
];

/// ADM detail-loss numerator map names by scale and band.
const ADM_NUM_NAMES: [[&str; 3]; 4] = [
    ["adm_num_0_h", "adm_num_0_v", "adm_num_0_d"],
    ["adm_num_1_h", "adm_num_1_v", "adm_num_1_d"],
    ["adm_num_2_h", "adm_num_2_v", "adm_num_2_d"],
    ["adm_num_3_h", "adm_num_3_v", "adm_num_3_d"],
];

/// ADM detail-loss denominator map names by scale and band.
const ADM_DEN_NAMES: [[&str; 3]; 4] = [
    ["adm_den_0_h", "adm_den_0_v", "adm_den_0_d"],
    ["adm_den_1_h", "adm_den_1_v", "adm_den_1_d"],
    ["adm_den_2_h", "adm_den_2_v", "adm_den_2_d"],
    ["adm_den_3_h", "adm_den_3_v", "adm_den_3_d"],
];

impl VmafMaps {
    fn view(&self, i: usize) -> Option<za::MapView<'_>> {
        let (name, data, g, polarity) = &self.maps[i];
        let px = za::gray_f32_slice(data, g.cells_x, g.cells_y).ok()?;
        za::MapView::new(name, px, *polarity, *g).ok()
    }
}

impl za::Evaluation for VmafMaps {
    fn score(&self) -> f64 {
        za::Evaluation::score(&self.scalars)
    }

    fn scores(&self) -> za::Scores<'_> {
        self.scalars.scores()
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        self.view(0)
    }

    fn maps(&self) -> Vec<za::MapView<'_>> {
        (0..self.maps.len()).filter_map(|i| self.view(i)).collect()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
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
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let request = opts.map;
        let scoring = opts.without_map();
        let mut seq = za::Metric::sequence(self, za::SequenceInfo::new().with_frame_count(1))?;
        let mut frames = seq.push(r.clone(), d.clone(), scoring)?;
        frames.extend(seq.finish(scoring)?.frames);
        let evaluation = match frames.pop() {
            Some(f) if frames.is_empty() && f.index == 0 => f.evaluation,
            _ => {
                return za::fail(za::MetricError::new(za::ErrorKind::Internal).with_metric("vmaf"));
            }
        };
        let Some(req) = request else {
            return Ok(evaluation);
        };
        let Some(scalars) = evaluation.as_any().downcast_ref::<za::ScalarEval>() else {
            return za::fail(za::MetricError::new(za::ErrorKind::Internal).with_metric("vmaf"));
        };
        // SPEED's block maps come from the same computation as its
        // feature; the frames were validated by the push above.
        let ((rf, (w, h, bits)), (df, _)) = (frame(&r)?, frame(&d)?);
        let (_, speed) = crate::speed::speed_v1_chroma_420_maps(
            &rf.u,
            &rf.v,
            &df.u,
            &df.v,
            w as usize,
            h as usize,
            bits,
            self.variant,
        )
        .metric_err_with("vmaf", Error::kind)?;
        opts.checkpoint()?;
        let (_, adm) =
            crate::adm::adm3_v1_terms(&rf.y, &df.y, w as usize, h as usize, bits, self.variant)
                .metric_err_with("vmaf", Error::kind)?;
        opts.checkpoint()?;
        let (_, cambi) = crate::cambi::cambi_v1_maps(&df.y, w as usize, h as usize, bits)
            .metric_err_with("vmaf", Error::kind)?;
        opts.checkpoint()?;
        let mut maps = Vec::new();
        // The phone models' 0.6 prescale puts SPEED's blocks on a fractional
        // pixel grid; there is no exact geometry to serve, so they get none.
        if let Some((cell, origin)) = speed.luma_grid {
            let g = za::MapGeometry::grid(
                w,
                h,
                origin,
                origin,
                cell,
                cell,
                speed.blocks.0 as u32,
                speed.blocks.1 as u32,
            )
            .with_support(za::Support::Global);
            let [u, v] = speed.maps;
            maps.push(("speed_u", u, g, za::MapPolarity::Difference));
            maps.push(("speed_v", v, g, za::MapPolarity::Difference));
        }
        // ADM's DWT output `k` reads inputs `2k − 1 .. 2k + 2`, so a scale-`s`
        // band sample owns the `2^(s+1)`-pixel block at `k·2^(s+1)`. The
        // masking and decoupling chain makes support global. The detail-loss
        // terms have no reference-comparison reading of their own: the
        // numerator is restored detail left visible, the denominator the
        // reference's detail.
        let adm_grid = |s: usize, t: &crate::adm::BandTerms| {
            let cell = 2u32 << s;
            za::MapGeometry::grid(
                w,
                h,
                t.left as i64 * i64::from(cell),
                t.top as i64 * i64::from(cell),
                cell,
                cell,
                t.cols as u32,
                t.rows as u32,
            )
            .with_support(za::Support::Global)
        };
        for (names, polarity, pick) in [
            (
                &AIM_NAMES,
                za::MapPolarity::Difference,
                (|t| &t.aim) as fn(&crate::adm::AdmScaleTerms) -> &crate::adm::BandTerms,
            ),
            (&ADM_NUM_NAMES, za::MapPolarity::Feature, |t| &t.num),
            (&ADM_DEN_NAMES, za::MapPolarity::Feature, |t| &t.den),
        ] {
            for (s, scale) in adm.iter().enumerate() {
                let t = pick(scale);
                let g = adm_grid(s, t);
                for (b, name) in names[s].into_iter().enumerate() {
                    maps.push((name, t.map(b), g, polarity));
                }
            }
        }
        // CAMBI point-samples even positions per scale (`decimate`), so
        // scale sample k is input pixel `factor·k`; cells are centred on it.
        // A value reads its scale's window (`window_size/2` samples) of the
        // mode-filtered image and spatial mask: the mode-filter chain adds
        // under `2·factor`, the full-resolution mask and anti-dithering 5
        // pixels.
        let pad = (cambi.window_size / 2) as u32;
        for (s, scale) in cambi.scales.into_iter().enumerate() {
            let f = scale.factor as u32;
            let halo = (pad + 2) * f + 5;
            let g = za::MapGeometry::grid(
                w,
                h,
                -i64::from(f / 2),
                -i64::from(f / 2),
                f,
                f,
                scale.w as u32,
                scale.h as u32,
            )
            .with_halo(halo, halo);
            maps.push((CAMBI_NAMES[s], scale.c, g, za::MapPolarity::Feature));
        }
        let eval = VmafMaps {
            scalars: scalars.clone(),
            maps,
        };
        za::apply_map_request(Box::new(eval), req)
    }

    fn temporal(&self) -> za::Temporal {
        let distance = self.distance();
        za::Temporal::Stateful {
            history: distance,
            latency: distance,
        }
    }

    fn sequence<'m>(
        &'m mut self,
        _info: za::SequenceInfo,
    ) -> za::MetricResult<Box<dyn za::Sequence + 'm>> {
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
            return za::fail(
                za::MetricError::new(za::ErrorKind::DimensionMismatch)
                    .with_metric("vmaf")
                    .with_message(format!(
                        "reference {shape:?}, distorted {dshape:?} (width, height, bits)"
                    )),
            );
        }
        match self.shape {
            Some(s) if s != shape => {
                return za::fail(
                    za::MetricError::new(za::ErrorKind::DimensionMismatch)
                        .with_metric("vmaf")
                        .with_message(format!(
                            "frame {} is {shape:?}, the sequence started {s:?}",
                            self.pushed
                        )),
                );
            }
            Some(_) => {}
            None => {
                let (w, h, bits) = shape;
                let (wu, hu) = (w as usize, h as usize);
                if wu < crate::adm::ADM_MIN_DIM
                    || hu < crate::adm::ADM_MIN_DIM
                    || (wu < crate::cambi::CAMBI_MIN_WIDTH_HEIGHT
                        && hu < crate::cambi::CAMBI_MIN_WIDTH_HEIGHT)
                {
                    return za::fail(
                        za::MetricError::new(za::ErrorKind::TooSmall)
                            .with_metric("vmaf")
                            .with_message(format!(
                                "{w}x{h}: VMAF needs both sides >= {} and one side >= {}",
                                crate::adm::ADM_MIN_DIM,
                                crate::cambi::CAMBI_MIN_WIDTH_HEIGHT
                            )),
                    );
                }
                self.stream = Some(
                    VmafV1Stream::new(w as usize, h as usize, bits, self.vmaf.variant)
                        .metric_err_with("vmaf", Error::kind)?,
                );
                self.shape = Some(shape);
            }
        }
        let Some(stream) = self.stream.as_mut() else {
            return za::fail(za::MetricError::new(za::ErrorKind::Internal).with_metric("vmaf"));
        };
        let results = stream
            .push(rf.as_yuv(), df.as_yuv())
            .metric_err_with("vmaf", Error::kind)?;
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

    fn finish(
        mut self: Box<Self>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<za::SequenceSummary> {
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
                za::MetricError::new(za::ErrorKind::SequenceOrder)
                    .with_metric("vmaf")
                    .with_message("finish with no frames"),
            );
        };
        let rest = stream.finish().metric_err_with("vmaf", Error::kind)?;
        let frames = self.ready(rest);
        let pooled = crate::pool_v1_scores(&self.scores, self.vmaf.pooling)
            .metric_err_with("vmaf", Error::kind)?;
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
    use za::PlaneLayout;
    use za::zenpixels::PixelDescriptor;

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
        Yuv420Frame {
            y: &p[0],
            u: &p[1],
            v: &p[2],
        }
    }

    /// One synthetic 8-bit 4:2:0 frame: a moving gradient plus texture.
    fn yuv8(t: usize, noise: u8) -> [Vec<u8>; 3] {
        let y = (0..W * H)
            .map(|i| {
                let (x, yy) = (i % W, i / W);
                (((x + 3 * t) * 3 + yy * 2) % 220
                    + (i * 7) % 11
                    + usize::from(i % 5 == 0) * usize::from(noise))
                .min(255) as u8
            })
            .collect();
        let c = |k: usize| {
            (0..(W / 2) * (H / 2))
                .map(|i| (100 + (i * k + t) % 50) as u8)
                .collect()
        };
        [y, c(3), c(5)]
    }

    fn image(p: &[Vec<u8>; 3]) -> za::Image<'_> {
        za::Planes::new(
            PlaneLayout::ycbcr_420(ChannelType::U8),
            vec![
                gray8(&p[0], W, H),
                gray8(&p[1], W / 2, H / 2),
                gray8(&p[2], W / 2, H / 2),
            ],
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
        za::conformance::assert_conforms(
            &mut Vmaf::new(ModelVariant::Standard1080p),
            &image(&r),
            &image(&d),
        );
    }

    /// Every pushed frame comes back once, in order, within the declared
    /// latency, bit-identical to driving `VmafV1Stream` directly; the pooled
    /// score is `pool_v1_scores` over them.
    #[test]
    fn sequence_matches_the_stream_and_respects_latency() {
        for variant in [ModelVariant::Standard1080p, ModelVariant::HfrStandard1080p] {
            let frames: Vec<_> = (0..6)
                .map(|t| (yuv8(t, 0), yuv8(t, 30 + t as u8)))
                .collect();
            let images: Vec<_> = frames.iter().map(|(r, d)| (image(r), image(d))).collect();
            let mut m = Vmaf::new(variant);
            let report = za::conformance::check_sequence(&mut m, za::SequenceInfo::new(), &images);
            assert!(report.violations.is_empty(), "{:?}", report.violations);

            let mut stream = VmafV1Stream::new(W, H, 8, variant).unwrap();
            let mut want = Vec::new();
            for (r, d) in &frames {
                let (r, d) = (to_u16(r), to_u16(d));
                want.extend(
                    stream
                        .push(yuv(&r), yuv(&d))
                        .unwrap()
                        .into_iter()
                        .map(|x| x.score),
                );
            }
            want.extend(stream.finish().unwrap().into_iter().map(|x| x.score));
            let got: Vec<f64> = report.frames.iter().map(|(_, s)| s[0].1).collect();
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let pooled = crate::pool_v1_scores(&want, PoolingMethod::Mean).unwrap();
            assert_eq!(report.pooled.unwrap()[0].1.to_bits(), pooled.to_bits());
        }
    }

    /// 10-bit content rides U16 planes with `significant_bits = 10`; 16
    /// significant bits is refused rather than rescaled.
    #[test]
    fn ten_bit_planes_are_accepted_sixteen_refused() {
        let p = to_u16(&yuv8(0, 0)).map(|v| v.into_iter().map(|s| s * 4).collect::<Vec<u16>>());
        let bytes: Vec<Vec<u8>> = p
            .iter()
            .map(|v| v.iter().flat_map(|s| s.to_ne_bytes()).collect())
            .collect();
        let planes = || {
            za::Planes::new(
                PlaneLayout::ycbcr_420(ChannelType::U16),
                vec![
                    gray16(&bytes[0], W, H),
                    gray16(&bytes[1], W / 2, H / 2),
                    gray16(&bytes[2], W / 2, H / 2),
                ],
            )
            .unwrap()
        };
        let mut m = Vmaf::new(ModelVariant::Standard1080p);
        let ten: za::Image<'_> = planes().with_significant_bits(10).unwrap().into();
        assert!(m.accepts(&ten.format()));
        let e = m
            .evaluate(ten.clone(), ten, za::EvalOpts::SCORE_ONLY)
            .unwrap();
        assert!(e.score().is_finite());
        let sixteen: za::Image<'_> = planes().into();
        let err = m
            .evaluate(sixteen.clone(), sixteen, za::EvalOpts::SCORE_ONLY)
            .err()
            .unwrap();
        assert_eq!(err.error().kind(), za::ErrorKind::UnsupportedInput);
    }

    #[test]
    fn mid_sequence_size_change_fails_and_poisons_the_session() {
        let mut m = Vmaf::new(ModelVariant::Standard1080p);
        let mut seq = m.sequence(za::SequenceInfo::new()).unwrap();
        let (r, d) = (yuv8(0, 0), yuv8(0, 9));
        seq.push(image(&r), image(&d), za::EvalOpts::SCORE_ONLY)
            .unwrap();
        fn small(p: &[Vec<u8>; 3]) -> za::Image<'_> {
            za::Image::from(
                za::Planes::new(
                    PlaneLayout::ycbcr_420(ChannelType::U8),
                    vec![
                        gray8(&p[0], W, H - 2),
                        gray8(&p[1], W / 2, H / 2 - 1),
                        gray8(&p[2], W / 2, H / 2 - 1),
                    ],
                )
                .unwrap(),
            )
        }
        let e = seq
            .push(small(&r), small(&d), za::EvalOpts::SCORE_ONLY)
            .err()
            .unwrap();
        assert_eq!(e.error().kind(), za::ErrorKind::DimensionMismatch);
        let e = seq
            .push(image(&r), image(&d), za::EvalOpts::SCORE_ONLY)
            .err()
            .unwrap();
        assert_eq!(e.error().kind(), za::ErrorKind::SequenceOrder);
    }
}

#[cfg(test)]
mod map_tests {
    use super::*;
    use za::Metric as _;
    use za::PlaneLayout;
    use za::zenpixels::PixelDescriptor;

    const W: usize = 640;
    const H: usize = 360;

    fn gray8(d: &[u8], w: usize, h: usize) -> PixelSlice<'_> {
        PixelSlice::new(d, w as u32, h as u32, w, PixelDescriptor::GRAY8).unwrap()
    }

    /// A textured 4:2:0 frame with varied chroma (SPEED reads chroma).
    fn frame8(seed: usize) -> [Vec<u8>; 3] {
        let y = (0..W * H)
            .map(|i| (((i % W) * 3 + (i / W) * 5 + (i * 7 + seed) % 13) % 230) as u8)
            .collect();
        let c = |k: usize| {
            (0..(W / 2) * (H / 2))
                .map(|i| {
                    let (x, yy) = (i % (W / 2), i / (W / 2));
                    (40 + (x * k + yy * 3 + (i * 31 + seed * k) % 23) % 170) as u8
                })
                .collect()
        };
        [y, c(3), c(5)]
    }

    fn image(p: &[Vec<u8>; 3]) -> za::Image<'_> {
        za::Planes::new(
            PlaneLayout::ycbcr_420(ChannelType::U8),
            vec![
                gray8(&p[0], W, H),
                gray8(&p[1], W / 2, H / 2),
                gray8(&p[2], W / 2, H / 2),
            ],
        )
        .unwrap()
        .into()
    }

    fn bits(e: &dyn za::Evaluation) -> Vec<u64> {
        e.scores().map(|(_, v)| v.to_bits()).collect()
    }

    /// Map requests leave every output bit-identical, and SPEED's block maps
    /// fold back to the `speed_chroma_uv` feature bit for bit: each channel
    /// is its blocks' f32 mean, combined as SPEED combines them.
    #[test]
    fn speed_maps_keep_the_scores_and_fold_to_the_feature() {
        for variant in [ModelVariant::Standard1080p, ModelVariant::Consumer4k] {
            let (r, d) = (frame8(0), frame8(9));
            let mut d2 = r.clone();
            for v in d2[1].iter_mut().step_by(7) {
                *v = v.saturating_add(30);
            }
            for (rr, dd) in [(&r, &d), (&r, &r), (&r, &d2)] {
                let mut m = Vmaf::new(variant);
                let plain = m
                    .evaluate(image(rr), image(dd), za::EvalOpts::SCORE_ONLY)
                    .unwrap();
                let mapped = m
                    .evaluate(
                        image(rr),
                        image(dd),
                        za::EvalOpts::SCORE_ONLY.with_native_map(),
                    )
                    .unwrap();
                assert_eq!(bits(&*mapped), bits(&*plain), "{variant:?}");
                let all = mapped.maps();
                assert_eq!(
                    all.iter().map(|m| m.name).collect::<Vec<_>>(),
                    ["speed_u", "speed_v"]
                        .into_iter()
                        .chain(AIM_NAMES.into_iter().flatten())
                        .chain(ADM_NUM_NAMES.into_iter().flatten())
                        .chain(ADM_DEN_NAMES.into_iter().flatten())
                        .chain(CAMBI_NAMES)
                        .collect::<Vec<_>>()
                );
                let maps = &all[..2];
                let (cell, origin) = match variant {
                    ModelVariant::Consumer4k => (320, -30),
                    _ => (160, -15),
                };
                let mut channel = [0.0f32; 2];
                for (c, map) in maps.iter().enumerate() {
                    assert_eq!(map.polarity, za::MapPolarity::Difference);
                    let g = map.geometry;
                    assert_eq!(
                        (g.cell_width, g.origin_x, g.support),
                        (cell, origin, za::Support::Global)
                    );
                    let v = map.values();
                    let mut sum = 0.0f32;
                    for &x in v.iter() {
                        assert!(x.is_finite() && x >= 0.0);
                        sum += x;
                    }
                    channel[c] = sum / v.len() as f32;
                }
                // Neither fixture is singular, so SPEED averages U and V.
                let uv = ((channel[0] + channel[1]) / 2.0).min(45.0) as f64;
                let feature = plain
                    .scores()
                    .find(|(k, _)| k == "speed_chroma_uv")
                    .unwrap()
                    .1;
                assert_eq!(
                    uv.to_bits(),
                    feature.to_bits(),
                    "{variant:?}: {uv} vs {feature}"
                );
                if std::ptr::eq(rr, dd) {
                    // SPEED evaluates `log2(1 + σ²)` in f32 for the
                    // reference and in f64 for the distorted side, as
                    // libvmaf does, so identical frames leave only that
                    // rounding in each block term.
                    let worst = maps
                        .iter()
                        .flat_map(|m| m.values().into_owned())
                        .fold(0.0f32, f32::max);
                    eprintln!("{variant:?}: identical-frame SPEED block terms reach {worst:e}");
                    assert!(worst < 1e-4, "{worst}");
                }
            }
        }
    }

    /// A chroma change confined to one block peaks in that block's cell.
    #[test]
    fn a_chroma_change_peaks_in_its_block() {
        let r = frame8(4);
        let mut d = r.clone();
        // Chroma rows/cols 90..110 × 170..190 → luma (340..380, 180..220),
        // inside block (2, 1) of the 160-pixel grid that starts at -15.
        for yy in 90..110 {
            for x in 170..190 {
                let i = yy * (W / 2) + x;
                d[2][i] = d[2][i].wrapping_add(90);
            }
        }
        let mut m = Vmaf::new(ModelVariant::Standard1080p);
        let e = m
            .evaluate(
                image(&r),
                image(&d),
                za::EvalOpts::SCORE_ONLY.with_native_map(),
            )
            .unwrap();
        let v_map = &e.maps()[1];
        let v = v_map.values();
        let k = (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32;
        let cell = (k % v_map.width(), k / v_map.width());
        assert_eq!(Some(cell), v_map.geometry.cell_at(360, 200));
    }

    /// ADM maps (AIM, and the detail-loss numerator and denominator) keep the
    /// scores and sit on their scale's block grid over the pooled inner
    /// region; AIM vanishes for identical frames and responds where the luma
    /// changed. Phone models serve them without SPEED.
    #[test]
    fn adm_maps_are_adm_terms_on_their_band_grids() {
        let (r, d) = (frame8(2), frame8(5));
        for variant in [ModelVariant::Standard1080p, ModelVariant::Phone] {
            let mut m = Vmaf::new(variant);
            let plain = m
                .evaluate(image(&r), image(&d), za::EvalOpts::SCORE_ONLY)
                .unwrap();
            let mapped = m
                .evaluate(
                    image(&r),
                    image(&d),
                    za::EvalOpts::SCORE_ONLY.with_native_map(),
                )
                .unwrap();
            assert_eq!(bits(&*mapped), bits(&*plain), "{variant:?}");
            let maps = mapped.maps();
            if matches!(variant, ModelVariant::Phone) {
                assert_eq!(
                    maps.iter().map(|m| m.name).collect::<Vec<_>>(),
                    AIM_NAMES
                        .into_iter()
                        .flatten()
                        .chain(ADM_NUM_NAMES.into_iter().flatten())
                        .chain(ADM_DEN_NAMES.into_iter().flatten())
                        .chain(CAMBI_NAMES)
                        .collect::<Vec<_>>(),
                    "phone models have no SPEED grid"
                );
                assert_eq!(mapped.map().unwrap().name, "aim_0_h");
            }
            let (luma, _) = frame(&image(&r)).unwrap();
            let y: Vec<u16> = luma.y;
            let (_, terms) =
                crate::adm::adm3_v1_terms(&y, &frame(&image(&d)).unwrap().0.y, W, H, 8, variant)
                    .unwrap();
            let kinds: [(
                &[[&str; 3]; 4],
                za::MapPolarity,
                fn(&crate::adm::AdmScaleTerms) -> &crate::adm::BandTerms,
            ); 3] = [
                (&AIM_NAMES, za::MapPolarity::Difference, |t| &t.aim),
                (&ADM_NUM_NAMES, za::MapPolarity::Feature, |t| &t.num),
                (&ADM_DEN_NAMES, za::MapPolarity::Feature, |t| &t.den),
            ];
            for (names, polarity, pick) in kinds {
                let prefix = &names[0][0][..names[0][0].len() - 3];
                let kind: Vec<_> = maps.iter().filter(|m| m.name.starts_with(prefix)).collect();
                assert_eq!(kind.len(), 12, "{prefix}");
                for (k, map) in kind.iter().enumerate() {
                    let (s, b) = (k / 3, k % 3);
                    assert_eq!(map.name, names[s][b]);
                    assert_eq!(map.polarity, polarity);
                    let g = map.geometry;
                    let (cell, t) = (2u32 << s, pick(&terms[s]));
                    assert_eq!(
                        g,
                        za::MapGeometry::grid(
                            W as u32,
                            H as u32,
                            t.left as i64 * i64::from(cell),
                            t.top as i64 * i64::from(cell),
                            cell,
                            cell,
                            t.cols as u32,
                            t.rows as u32
                        )
                        .with_support(za::Support::Global)
                    );
                    // The pooled region is the inner 80% of the band.
                    assert!(g.origin_x > 0 && g.origin_x < W as i64 / 5);
                    assert_eq!(map.values().as_ref(), t.map(b).as_slice());
                    assert!(map.values().iter().all(|&v| v >= 0.0));
                }
                assert!(kind.iter().any(|m| m.values().iter().any(|&v| v > 0.0)));
            }
        }
        // Identical frames have no additive impairment.
        let mapped = Vmaf::new(ModelVariant::Standard1080p)
            .evaluate(
                image(&r),
                image(&r),
                za::EvalOpts::SCORE_ONLY.with_native_map(),
            )
            .unwrap();
        for m in mapped.maps().iter().filter(|m| m.name.starts_with("aim_")) {
            assert!(m.values().iter().all(|&v| v == 0.0), "{}", m.name);
        }
        // A luma change confined to one region peaks in a scale-0 cell over it.
        let mut d = r.clone();
        for yy in 150..170 {
            for x in 300..330 {
                d[0][yy * W + x] = d[0][yy * W + x].wrapping_add(70);
            }
        }
        let mapped = Vmaf::new(ModelVariant::Standard1080p)
            .evaluate(
                image(&r),
                image(&d),
                za::EvalOpts::SCORE_ONLY.with_native_map(),
            )
            .unwrap();
        for name in AIM_NAMES[0] {
            let maps = mapped.maps();
            let m = maps.iter().find(|m| m.name == name).unwrap();
            let v = m.values();
            let k = (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32;
            let g = m.geometry;
            let (cx, cy) = (k % m.width(), k / m.width());
            let (x0, y0) = (
                g.origin_x + i64::from(cx * g.cell_width),
                g.origin_y + i64::from(cy * g.cell_height),
            );
            assert!(
                (296..334).contains(&x0) && (146..172).contains(&y0),
                "{name}: peak cell at ({x0}, {y0})"
            );
        }
    }

    #[test]
    fn sequences_refuse_maps() {
        let (r, d) = (frame8(0), frame8(1));
        let mut m = Vmaf::new(ModelVariant::Standard1080p);
        {
            let mut seq = m.sequence(za::SequenceInfo::new()).unwrap();
            let e = seq
                .push(
                    image(&r),
                    image(&d),
                    za::EvalOpts::SCORE_ONLY.with_native_map(),
                )
                .err()
                .unwrap();
            assert_eq!(e.error().kind(), za::ErrorKind::NoMap);
        }
        let stop = za::how_far::StopOnly::new(za::AlwaysStop(za::how_far::StopReason::Cancelled));
        let e = m
            .evaluate(
                image(&r),
                image(&d),
                za::EvalOpts::SCORE_ONLY.with_native_map().with_pulse(&stop),
            )
            .err()
            .unwrap();
        assert!(za::is_stopped(&e));
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;
    use za::Metric as _;
    use za::PlaneLayout;
    use za::zenpixels::PixelDescriptor;

    #[test]
    fn below_cambi_minimum_is_too_small() {
        let (w, h) = (64usize, 48usize);
        let (y, c) = (vec![128u8; w * h], vec![128u8; (w / 2) * (h / 2)]);
        fn g(d: &[u8], w: usize, h: usize) -> PixelSlice<'_> {
            PixelSlice::new(d, w as u32, h as u32, w, PixelDescriptor::GRAY8).unwrap()
        }
        let img: za::Image<'_> = za::Planes::new(
            PlaneLayout::ycbcr_420(ChannelType::U8),
            vec![g(&y, w, h), g(&c, w / 2, h / 2), g(&c, w / 2, h / 2)],
        )
        .unwrap()
        .into();
        let e = Vmaf::new(ModelVariant::Standard1080p)
            .evaluate(img.clone(), img, za::EvalOpts::SCORE_ONLY)
            .err()
            .unwrap();
        assert_eq!(e.error().kind(), za::ErrorKind::TooSmall);
    }
}
