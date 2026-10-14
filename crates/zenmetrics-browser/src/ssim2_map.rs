//! SSIMULACRA2 on the unified metric surface, serving fast-ssim2's native
//! spatial terms as maps.
//!
//! SSIMULACRA2 has no per-pixel score map. At each of up to six pyramid
//! scales and each positive-XYB channel it computes three per-pixel error
//! terms, takes the mean and 4-norm of each, and turns the 108 aggregates
//! into the score through a weighted sum and a nonlinear transform. The maps
//! are those terms ([`fast_ssim2::Ssimulacra2ScaleMaps`]):
//! `{ssim,artifact,detail_lost}_{x,y,b}_{scale}`, 54 at most, with
//! `ssim_y_0` (full-resolution luma-like channel) first. All are
//! [`za::MapPolarity::Difference`] (`0` = identical).
//!
//! Scale `s` samples `2^s × 2^s` pixel blocks from the origin; a value reads
//! the scale within the Gaussian blur's 5-sample radius, so the halo is
//! `5·2^s` (fast-ssim2 measures rounding residue past it, ≤ 1.7e-4 on a
//! 0.66 peak). Inputs under 8×8 are reflect-padded, so their cells beyond
//! the image are clipped and their support is global. With alpha, the maps
//! come from the background composite that set the score.

use std::borrow::Cow;

use zenmetrics_abstractions as za;
use zenmetrics_abstractions::zenpixels::PixelDescriptor;

/// Output revision of [`Ssimulacra2`] on the generic surface.
const SCORE_REVISION: u32 = 1;

/// `NAMES[scale][term][channel]`.
const NAMES: [[[&str; 3]; 3]; 6] = [
    [
        ["ssim_x_0", "ssim_y_0", "ssim_b_0"],
        ["artifact_x_0", "artifact_y_0", "artifact_b_0"],
        ["detail_lost_x_0", "detail_lost_y_0", "detail_lost_b_0"],
    ],
    [
        ["ssim_x_1", "ssim_y_1", "ssim_b_1"],
        ["artifact_x_1", "artifact_y_1", "artifact_b_1"],
        ["detail_lost_x_1", "detail_lost_y_1", "detail_lost_b_1"],
    ],
    [
        ["ssim_x_2", "ssim_y_2", "ssim_b_2"],
        ["artifact_x_2", "artifact_y_2", "artifact_b_2"],
        ["detail_lost_x_2", "detail_lost_y_2", "detail_lost_b_2"],
    ],
    [
        ["ssim_x_3", "ssim_y_3", "ssim_b_3"],
        ["artifact_x_3", "artifact_y_3", "artifact_b_3"],
        ["detail_lost_x_3", "detail_lost_y_3", "detail_lost_b_3"],
    ],
    [
        ["ssim_x_4", "ssim_y_4", "ssim_b_4"],
        ["artifact_x_4", "artifact_y_4", "artifact_b_4"],
        ["detail_lost_x_4", "detail_lost_y_4", "detail_lost_b_4"],
    ],
    [
        ["ssim_x_5", "ssim_y_5", "ssim_b_5"],
        ["artifact_x_5", "artifact_y_5", "artifact_b_5"],
        ["detail_lost_x_5", "detail_lost_y_5", "detail_lost_b_5"],
    ],
];

/// SSIMULACRA2 (fast-ssim2) as a generic scorer: packed 8-bit sRGB RGB or
/// RGBA, or linear f32 RGB, strided rows accepted.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Ssimulacra2;

fn error_kind(e: &fast_ssim2::Ssimulacra2Error) -> za::ErrorKind {
    use fast_ssim2::Ssimulacra2Error as E;
    match e {
        E::InvalidConfiguration(_) => za::ErrorKind::InvalidConfig,
        E::UnsupportedInput(_) => za::ErrorKind::UnsupportedInput,
        E::NonMatchingImageDimensions => za::ErrorKind::DimensionMismatch,
        E::InvalidImageSize => za::ErrorKind::TooSmall,
        E::ImageTooLarge { .. } => za::ErrorKind::Resource,
        E::Cancelled(r) => za::ErrorKind::Stopped(*r),
        _ => za::ErrorKind::Internal,
    }
}

#[track_caller]
fn fail<T>(e: fast_ssim2::Ssimulacra2Error) -> za::MetricResult<T> {
    use za::ResultMetricExt as _;
    Err::<T, _>(e).metric_err_with("ssim2", error_kind)
}

#[track_caller]
fn packed<'a, 'b>(img: &'a za::Image<'b>) -> za::MetricResult<&'a za::zenpixels::PixelSlice<'b>> {
    match img.as_packed() {
        Some(s) if accepts_descriptor(s.descriptor()) => Ok(s),
        _ => za::fail(
            za::unsupported(&img.format(), "packed RGB8/RGBA8 sRGB or linear RGBF32")
                .with_metric("ssim2"),
        ),
    }
}

fn accepts_descriptor(d: PixelDescriptor) -> bool {
    [
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGBF32_LINEAR,
    ]
    .contains(&d)
}

/// A score with its maps, in serving order.
struct Ssim2Maps {
    score: f64,
    maps: Vec<(&'static str, Vec<f32>, za::MapGeometry)>,
}

impl Ssim2Maps {
    fn new(m: fast_ssim2::Ssimulacra2Maps, width: u32, height: u32) -> Self {
        let padded = width < 8 || height < 8;
        let mut maps = Vec::with_capacity(m.scales.len() * 9);
        for (s, scale) in m.scales.into_iter().enumerate() {
            let cell = 1u32 << s;
            let mut g = za::MapGeometry::grid(
                width,
                height,
                0,
                0,
                cell,
                cell,
                scale.width as u32,
                scale.height as u32,
            );
            g = if padded {
                g.with_support(za::Support::Global)
            } else {
                g.with_halo(5 * cell, 5 * cell)
            };
            let fast_ssim2::Ssimulacra2ScaleMaps {
                ssim,
                artifact,
                detail_lost,
                ..
            } = scale;
            for (t, planes) in [ssim, artifact, detail_lost].into_iter().enumerate() {
                for (c, plane) in planes.into_iter().enumerate() {
                    maps.push((NAMES[s][t][c], plane, g));
                }
            }
        }
        // Serve `ssim_y_0` first.
        if let Some(i) = maps.iter().position(|(n, ..)| *n == "ssim_y_0") {
            let primary = maps.remove(i);
            maps.insert(0, primary);
        }
        Self {
            score: m.score,
            maps,
        }
    }

    fn view(&self, i: usize) -> Option<za::MapView<'_>> {
        let (name, data, g) = &self.maps[i];
        let px = za::gray_f32_slice(data, g.cells_x, g.cells_y).ok()?;
        za::MapView::new(name, px, za::MapPolarity::Difference, *g).ok()
    }
}

impl za::Evaluation for Ssim2Maps {
    fn score(&self) -> f64 {
        self.score
    }

    fn scores(&self) -> za::Scores<'_> {
        Box::new(std::iter::once((Cow::Borrowed("ssim2"), self.score)))
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        self.view(0)
    }

    fn maps(&self) -> Vec<za::MapView<'_>> {
        (0..self.maps.len()).filter_map(|i| self.view(i)).collect()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn evaluate_pair(
    r: &za::zenpixels::PixelSlice<'_>,
    d: &za::zenpixels::PixelSlice<'_>,
    opts: za::EvalOpts<'_>,
) -> za::MetricResult<Box<dyn za::Evaluation>> {
    let stop = opts.stop();
    let config = fast_ssim2::Ssimulacra2Config::default().with_stop(stop);
    let Some(req) = opts.map else {
        return match fast_ssim2::compute_ssimulacra2_with_config(r, d, &config) {
            Ok(s) => Ok(Box::new(za::ScalarEval::single("ssim2", s))),
            Err(e) => fail(e),
        };
    };
    let maps = match fast_ssim2::compute_ssimulacra2_maps_with_config(r, d, &config) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    opts.checkpoint()?;
    za::apply_map_request(Box::new(Ssim2Maps::new(maps, r.width(), r.rows())), req)
}

impl za::Metric for Ssimulacra2 {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(
            "ssim2",
            za::Implementation::new("fast-ssim2", fast_ssim2::VERSION, SCORE_REVISION)
                .with_build(option_env!("ZENMETRICS_BUILD_ID")),
        )
        .with_reference("SSIMULACRA 2.1 (libjxl ssimulacra2 tool)")
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new("ssim2", za::ScoreKind::Quality100)];
        Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        format.packed().is_some_and(accepts_descriptor)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        opts.checkpoint()?;
        let (rp, dp) = (packed(&r)?, packed(&d)?);
        za::check_same_size(&r, &d)?;
        evaluate_pair(rp, dp, opts)
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        opts.checkpoint()?;
        let rp = packed(&r)?;
        let config = fast_ssim2::Ssimulacra2Config::default().with_stop(opts.stop());
        let prepared = match fast_ssim2::Ssimulacra2Reference::new_with_config(rp, &config) {
            Ok(p) => p,
            Err(e) => return fail(e),
        };
        Ok(Box::new(Ssim2Warm {
            prepared,
            reference: r.to_owned_image(),
        }))
    }
}

/// A prepared reference: score-only candidates reuse its pyramid
/// ([`fast_ssim2::Ssimulacra2Reference`], bit-identical to one-shot
/// scoring); map requests rerun the pair from the retained reference copy.
struct Ssim2Warm {
    prepared: fast_ssim2::Ssimulacra2Reference,
    reference: za::OwnedImage,
}

impl za::WarmReference for Ssim2Warm {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        opts.checkpoint()?;
        let r = self.reference.as_image();
        za::check_same_size(&r, &d)?;
        let dp = packed(&d)?;
        if opts.map.is_some() {
            return evaluate_pair(packed(&r)?, dp, opts);
        }
        let config = fast_ssim2::Ssimulacra2Config::default().with_stop(opts.stop());
        match self.prepared.compare_with_config(dp, &config) {
            Ok(s) => Ok(Box::new(za::ScalarEval::single("ssim2", s))),
            Err(e) => fail(e),
        }
    }

    fn reuses_reference_work(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use za::Metric as _;
    use za::zenpixels::PixelSlice;

    fn image(px: &[u8], w: usize, h: usize, stride: usize) -> za::Image<'_> {
        PixelSlice::new(px, w as u32, h as u32, stride, PixelDescriptor::RGB8_SRGB)
            .unwrap()
            .into()
    }

    fn bits(e: &dyn za::Evaluation) -> Vec<u64> {
        e.scores().map(|(_, v)| v.to_bits()).collect()
    }

    #[test]
    fn conforms_to_the_generic_surface() {
        za::conformance::assert_conforms_rgb8(&mut Ssimulacra2, 64, 48);
        za::conformance::assert_conforms_rgb8(&mut Ssimulacra2, 5, 7);
    }

    /// The headline is fast-ssim2's score bit for bit, with and without
    /// maps, and the maps are its native terms on their pyramid grids.
    #[test]
    fn scores_match_fast_ssim2_and_maps_describe_the_pyramid() {
        for (w, h) in [(64usize, 48usize), (37, 23), (8, 8), (5, 3), (130, 9)] {
            let a = za::conformance::textured_rgb8(w, h, 1);
            let b = za::conformance::textured_rgb8(w, h, 2);
            let (ra, rb) = (image(&a, w, h, w * 3), image(&b, w, h, w * 3));
            let direct =
                fast_ssim2::compute_ssimulacra2(ra.as_packed().unwrap(), rb.as_packed().unwrap())
                    .unwrap();
            let plain = Ssimulacra2
                .evaluate(ra.clone(), rb.clone(), za::EvalOpts::SCORE_ONLY)
                .unwrap();
            let mapped = Ssimulacra2
                .evaluate(ra, rb, za::EvalOpts::SCORE_ONLY.with_native_map())
                .unwrap();
            assert_eq!(plain.score().to_bits(), direct.to_bits(), "{w}x{h}");
            assert_eq!(bits(&*mapped), bits(&*plain), "{w}x{h}");
            let maps = mapped.maps();
            assert_eq!(maps[0].name, "ssim_y_0");
            assert_eq!(maps.len() % 9, 0);
            let (mut sw, mut sh) = (w.max(8), h.max(8));
            for s in 0..maps.len() / 9 {
                let at_scale: Vec<_> = maps
                    .iter()
                    .filter(|m| m.name.ends_with(&format!("_{s}")))
                    .collect();
                assert_eq!(at_scale.len(), 9, "{w}x{h} scale {s}");
                for m in at_scale {
                    let g = m.geometry;
                    assert_eq!((g.cells_x as usize, g.cells_y as usize), (sw, sh));
                    assert_eq!((g.input_width as usize, g.input_height as usize), (w, h));
                    assert_eq!(g.cell_width, 1 << s);
                    assert_eq!(m.polarity, za::MapPolarity::Difference);
                    let padded = w < 8 || h < 8;
                    assert_eq!(g.support == za::Support::Global, padded);
                }
                (sw, sh) = (sw.div_ceil(2), sh.div_ceil(2));
            }
        }
    }

    /// Warm candidates reuse the prepared reference for scores and serve the
    /// same maps as cold evaluation.
    #[test]
    fn warm_candidates_match_cold() {
        let (w, h) = (48usize, 40usize);
        let a = za::conformance::textured_rgb8(w, h, 5);
        let cands = [
            za::conformance::textured_rgb8(w, h, 6),
            za::conformance::textured_rgb8(w, h, 7),
        ];
        let mut m = Ssimulacra2;
        let cold: Vec<_> = cands
            .iter()
            .map(|d| {
                let e = Ssimulacra2
                    .evaluate(
                        image(&a, w, h, w * 3),
                        image(d, w, h, w * 3),
                        za::EvalOpts::SCORE_ONLY.with_native_map(),
                    )
                    .unwrap();
                (bits(&*e), e.map().unwrap().values().into_owned())
            })
            .collect();
        let mut warm = m
            .warm_reference(image(&a, w, h, w * 3), za::EvalOpts::SCORE_ONLY)
            .unwrap();
        assert!(warm.reuses_reference_work());
        for (d, (sb, map)) in cands.iter().zip(&cold) {
            let s = warm
                .evaluate(image(d, w, h, w * 3), za::EvalOpts::SCORE_ONLY)
                .unwrap();
            assert_eq!(&bits(&*s), sb);
            let e = warm
                .evaluate(
                    image(d, w, h, w * 3),
                    za::EvalOpts::SCORE_ONLY.with_native_map(),
                )
                .unwrap();
            assert_eq!(&bits(&*e), sb);
            assert_eq!(&e.map().unwrap().values().into_owned(), map);
        }
    }

    #[test]
    fn strided_input_and_cancellation() {
        let (w, h) = (33usize, 29usize);
        let (a, b) = (
            za::conformance::textured_rgb8(w, h, 3),
            za::conformance::textured_rgb8(w, h, 4),
        );
        let stride = (w + 3) * 3;
        let pad = |src: &[u8]| {
            let mut out = vec![0x99u8; stride * h];
            for y in 0..h {
                out[y * stride..y * stride + w * 3]
                    .copy_from_slice(&src[y * w * 3..(y + 1) * w * 3]);
            }
            out
        };
        let (pa, pb) = (pad(&a), pad(&b));
        let opts = za::EvalOpts::SCORE_ONLY.with_native_map();
        let packed = Ssimulacra2
            .evaluate(image(&a, w, h, w * 3), image(&b, w, h, w * 3), opts)
            .unwrap();
        let strided = Ssimulacra2
            .evaluate(image(&pa, w, h, stride), image(&pb, w, h, stride), opts)
            .unwrap();
        assert_eq!(bits(&*packed), bits(&*strided));
        for (p, s) in packed.maps().iter().zip(strided.maps()) {
            assert_eq!(p.values(), s.values(), "{}", p.name);
        }
        let stop = za::how_far::StopOnly::new(za::AlwaysStop(za::how_far::StopReason::Cancelled));
        let err = Ssimulacra2
            .evaluate(
                image(&a, w, h, w * 3),
                image(&b, w, h, w * 3),
                opts.with_pulse(&stop),
            )
            .err()
            .unwrap();
        assert!(za::is_stopped(&err));
    }
}
