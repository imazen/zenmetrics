//! Validation-only bridge to the selected native kernels; never a published API.
#![forbid(unsafe_code)]
#![allow(dead_code, deprecated)]
#![allow(clippy::manual_midpoint, clippy::chunks_exact_to_as_chunks)]
include!("kernel.rs");
use iqa::zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice};
use iqa::{
    ErrorKind, EvalOpts, Evaluation, Image, Implementation, InputFormat, Metric, MetricError,
    MetricId, MetricResult, ScoreKind, ScoreSpec,
};
use std::borrow::Cow;
use zenpixels_iqa as iqa;

struct Margarine;
struct Native {
    scores: Vec<(Cow<'static, str>, f64)>,
    map: PixelBuffer,
}
impl Evaluation for Native {
    fn score(&self) -> f64 {
        self.scores[0].1
    }
    fn scores(&self) -> iqa::Scores<'_> {
        Box::new(self.scores.iter().cloned())
    }
    fn map(&self) -> Option<iqa::MapView<'_>> {
        Some(
            iqa::MapView::new(
                "difference",
                self.map.as_slice(),
                iqa::MapPolarity::Difference,
                iqa::MapGeometry::identity(self.map.width(), self.map.height()),
            )
            .unwrap(),
        )
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
fn rows<'a>(v: &'a iqa::Rgb8In<'_>) -> ingress::EncodedRows<'a> {
    ingress::EncodedRows::new(
        ingress::Samples::U8(v.pixels),
        v.width,
        v.height,
        v.stride,
        3,
    )
    .unwrap()
}
fn native(r: &Image<'_>, d: &Image<'_>) -> diff::InternalResult {
    let (r, d) = iqa::rgb8_pair(r, d).unwrap();
    row_psycho::compute_geometry(
        &rows(&r),
        &rows(&d),
        128,
        512,
        &ButteraugliParams::default(),
    )
    .unwrap()
}
impl Metric for Margarine {
    fn id(&self) -> MetricId {
        MetricId::new("margarine", Implementation::new("margarine", "0.0.0", 1))
            .with_reference("zenmetrics b50fea22 selected simd-row-malta")
            .with_config("rows", 128)
            .with_config("columns", 512)
            .with_config("pooling", "max,p1,p2,p3,p6")
            .with_config("native_uhf", false)
    }
    fn outputs(&self) -> Cow<'static, [ScoreSpec]> {
        Cow::Owned(
            ["max", "p1", "p2", "p3", "p6"]
                .map(|key| ScoreSpec::new(key, ScoreKind::Difference))
                .to_vec(),
        )
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
        opts.checkpoint()?;
        let (r, d) = iqa::rgb8_pair(&r, &d)?;
        let r = ingress::EncodedRows::new(
            ingress::Samples::U8(r.pixels),
            r.width,
            r.height,
            r.stride,
            3,
        )
        .expect("shared RGB input admission validated row geometry");
        let d = ingress::EncodedRows::new(
            ingress::Samples::U8(d.pixels),
            d.width,
            d.height,
            d.stride,
            3,
        )
        .expect("shared RGB input admission validated row geometry");
        // The selected kernel has no Pulse argument. Boundaries remain cooperative;
        // no claim of cancellation within its native call is made.
        let n = row_psycho::compute_geometry_send(&r, &d, 128, 512, &ButteraugliParams::default())
            .map_err(|e| {
                iqa::whereat::at(MetricError::new(ErrorKind::Backend).with_boxed_detail(e))
            })?;
        opts.checkpoint()?;
        let map = n.diffmap.as_ref().unwrap();
        let scores = [
            ("max", n.score),
            ("p1", pnorm(map, 1.0)),
            ("p2", pnorm(map, 2.0)),
            ("p3", n.pnorm_3),
            ("p6", pnorm(map, 6.0)),
        ]
        .map(|(k, v)| (Cow::Borrowed(k), v))
        .to_vec();
        let values: Vec<f32> = (0..map.height())
            .flat_map(|y| map.row(y).iter().copied())
            .collect();
        let e: Box<dyn Evaluation> = Box::new(Native {
            scores,
            map: iqa::gray_f32_buffer(&values, map.width() as u32, map.height() as u32)?,
        });
        match opts.map {
            Some(request) => iqa::apply_map_request(e, request),
            None => Ok(e),
        }
    }
}
fn fixture(w: u32, h: u32, padding: usize, distorted: bool) -> Vec<u8> {
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
fn image(bytes: &[u8], w: u32, h: u32, padding: usize) -> Image<'_> {
    PixelSlice::new(
        bytes,
        w,
        h,
        w as usize * 3 + padding,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap()
    .into()
}
fn main() {
    assert_eq!(CANDIDATE, "simd-row-malta");
    // The selected private kernel now exposes a Send+Sync cause so the IQA
    // envelope can retain its allocation, rather than flattening it to text.
    let sample = fixture(64, 64, 0, false);
    let sample_image = image(&sample, 64, 64, 0);
    let sample_view = iqa::rgb8_view(&sample_image).unwrap();
    let sample_rows = rows(&sample_view);
    let error = row_psycho::compute_geometry_send(
        &sample_rows,
        &sample_rows,
        128,
        0,
        &ButteraugliParams::default(),
    )
    .err()
    .expect("zero tile columns must refuse");
    let original = &*error as *const (dyn std::error::Error + Send + Sync) as *const ();
    let envelope = MetricError::new(ErrorKind::InvalidConfig).with_boxed_detail(error);
    let retained =
        std::error::Error::source(&envelope).unwrap() as *const dyn std::error::Error as *const ();
    assert_eq!(original, retained);

    for (w, h) in [(64, 64), (97, 73), (131, 129)] {
        let r = fixture(w, h, 0, false);
        let d = fixture(w, h, 0, true);
        let rp = fixture(w, h, 9, false);
        let dp = fixture(w, h, 15, true);
        let ri = image(&r, w, h, 0);
        let di = image(&d, w, h, 0);
        iqa::conformance::assert_conforms(&mut Margarine, &ri, &di);
        let direct = native(&ri, &di);
        let map = direct.diffmap.as_ref().unwrap();
        let out = iqa::score(&mut Margarine, ri, di, EvalOpts::new().with_native_map()).unwrap();
        let expected = [
            direct.score,
            pnorm(map, 1.0),
            pnorm(map, 2.0),
            direct.pnorm_3,
            pnorm(map, 6.0),
        ]
        .map(f64::to_bits);
        assert_eq!(
            out.result
                .scores()
                .map(|(_, v)| v.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
        let padded = iqa::score(
            &mut Margarine,
            image(&rp, w, h, 9),
            image(&dp, w, h, 15),
            EvalOpts::new().with_native_map(),
        )
        .unwrap();
        assert_eq!(
            padded
                .result
                .scores()
                .map(|(_, v)| v.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
        let expected_map: Vec<_> = (0..map.height())
            .flat_map(|y| map.row(y).iter().map(|v| v.to_bits()))
            .collect();
        for e in [&out, &padded] {
            assert_eq!(
                e.result
                    .map()
                    .unwrap()
                    .values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected_map
            );
        }
        assert!(
            iqa::prepare_reference(&mut Margarine, image(&r, w, h, 0), EvalOpts::new()).is_err()
        );
        println!(
            "PASS selected Margarine {w}x{h}: five native scalar bits, map bits, stride, conformance; preparation explicitly unsupported"
        );
    }
}
