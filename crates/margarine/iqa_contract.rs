//! Validation bridge for the selected native kernels, preserving native data.
#![forbid(unsafe_code)]
#![allow(dead_code, deprecated)]
#![allow(clippy::manual_midpoint, clippy::chunks_exact_to_as_chunks)]
include!("kernel.rs");
use iqa::zenpixels::{PixelDescriptor, PixelSlice};
use iqa::{
    Direction, ErrorKind, EvalOpts, Evaluation, Implementation, Metric, MetricError, MetricId,
    MetricResult, Score, ScoreSpec,
};
use zenpsy_api as iqa;
struct Margarine;
struct Native {
    original: diff::InternalResult,
    values: [f64; 5],
    specs: [ScoreSpec; 5],
}
impl std::fmt::Debug for Native {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Native")
            .field("values", &self.values)
            .finish_non_exhaustive()
    }
}
impl Evaluation for Native {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        self.specs
            .iter()
            .zip(self.values)
            .map(|(s, v)| Score::new(s, v))
    }
    fn maps(&self) -> impl Iterator<Item = iqa::MapView<'_>> {
        static SPEC: ScoreSpec = ScoreSpec::new("diffmap", Direction::Lower).with_perfect(0.0);
        self.original.diffmap.iter().map(|m| {
            let plane =
                iqa::zenpixels::plane::PlaneRef::new(m.data(), m.width(), m.height(), m.stride())
                    .unwrap();
            iqa::MapView::new(
                &SPEC,
                iqa::zenpixels::plane::AnyPlane::F32(plane),
                iqa::MapGrid::new(m.width() as u32, m.height() as u32).unwrap(),
            )
        })
    }
}
fn rows<'a>(p: &'a PixelSlice<'_>) -> MetricResult<ingress::EncodedRows<'a>> {
    if p.descriptor() != PixelDescriptor::RGB8_SRGB || p.color_context().is_some() {
        return iqa::fail(MetricError::new(ErrorKind::UnsupportedInput));
    }
    Ok(ingress::EncodedRows::new(
        ingress::Samples::U8(p.as_strided_bytes()),
        p.width() as usize,
        p.rows() as usize,
        p.stride(),
        3,
    )
    .unwrap())
}
impl Metric for Margarine {
    type Output = Native;
    fn id(&self) -> MetricId {
        MetricId::new("margarine", Implementation::new("margarine", "0.0.0", 1))
            .with_reference("zenmetrics b50fea22 selected simd-row-malta")
            .with_config("rows", 128)
            .with_config("columns", 512)
            .with_config("pooling", "max,p1,p2,p3,p6")
            .with_config("native_uhf", false)
    }
    fn evaluate(
        &mut self,
        r: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Native> {
        opts.checkpoint()?;
        let mut original = row_psycho::compute_geometry_send(
            &rows(r)?,
            &rows(d)?,
            128,
            512,
            &ButteraugliParams::default(),
        )
        .map_err(|e| iqa::whereat::at(MetricError::new(ErrorKind::Backend).with_boxed_detail(e)))?;
        opts.checkpoint()?;
        // The selected kernel pools these scalars from its internal map. Do not
        // copy or retain that scratch as a diagnostic result unless requested.
        let map = original.diffmap.as_ref().unwrap();
        let values = [
            original.score,
            pnorm(map, 1.0),
            pnorm(map, 2.0),
            original.pnorm_3,
            pnorm(map, 6.0),
        ];
        if !opts.diagnostics() {
            original.diffmap = None;
        }
        Ok(Native {
            original,
            values,
            specs: ["max", "p1", "p2", "p3", "p6"]
                .map(|k| ScoreSpec::new(k, Direction::Lower).with_perfect(0.0)),
        })
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
fn image(bytes: &[u8], w: u32, h: u32, padding: usize) -> PixelSlice<'_> {
    PixelSlice::new(
        bytes,
        w,
        h,
        w as usize * 3 + padding,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap()
}
fn main() {
    assert_eq!(CANDIDATE, "simd-row-malta");
    let sample = fixture(64, 64, 0, false);
    let sample_image = image(&sample, 64, 64, 0);
    let sample_rows = rows(&sample_image).unwrap();
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
        let direct = row_psycho::compute_geometry(
            &rows(&ri).unwrap(),
            &rows(&di).unwrap(),
            128,
            512,
            &ButteraugliParams::default(),
        )
        .unwrap();
        let map = direct.diffmap.as_ref().unwrap();
        let expected = [
            direct.score,
            pnorm(map, 1.0),
            pnorm(map, 2.0),
            direct.pnorm_3,
            pnorm(map, 6.0),
        ]
        .map(f64::to_bits);
        let scalar = iqa::score(&mut Margarine, &ri, &di, EvalOpts::new()).unwrap();
        assert_eq!(scalar.values.map(f64::to_bits), expected);
        assert!(scalar.original.diffmap.is_none());
        let mut executor = zenmetrics_orchestrator::psy::ContractExecutor::default();
        executor.register("selected-margarine", || Ok(Margarine));
        let report = executor
            .run(
                &ri,
                &[di.clone(), di.clone()],
                EvalOpts::new(),
                zenmetrics_orchestrator::psy::Fallback::Refuse,
            )
            .ok()
            .unwrap();
        assert!(report.is_complete());
        assert!(!report.attempts[0].prepared);
        for result in report.completed {
            assert_eq!(
                result
                    .assessment
                    .scores
                    .iter()
                    .map(|s| s.value.to_bits())
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(
                result
                    .assessment
                    .native::<Native>()
                    .unwrap()
                    .original
                    .diffmap
                    .is_none()
            );
            assert_eq!(result.identity, Margarine.id());
        }

        let expected_map = (0..map.height())
            .flat_map(|y| map.row(y).iter().map(|v| v.to_bits()))
            .collect::<Vec<_>>();
        for (r, d) in [(ri, di), (image(&rp, w, h, 9), image(&dp, w, h, 15))] {
            let out =
                iqa::score(&mut Margarine, &r, &d, EvalOpts::new().with_native_map()).unwrap();
            assert_eq!(
                out.scores().map(|s| s.value.to_bits()).collect::<Vec<_>>(),
                expected
            );
            let generic = out.maps().next().unwrap();
            let iqa::zenpixels::plane::AnyPlane::F32(plane) = generic.values else {
                panic!("native f32 map")
            };
            assert_eq!(
                plane.strided_samples().as_ptr(),
                out.original.diffmap.as_ref().unwrap().data().as_ptr()
            );
            assert_eq!(generic.grid, iqa::MapGrid::new(w, h).unwrap());
            assert_eq!(
                plane
                    .rows()
                    .flatten()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected_map
            );
            let map = out.original.diffmap.unwrap();
            assert_eq!((map.width(), map.height()), (w as usize, h as usize));
            assert_eq!(
                (0..map.height())
                    .flat_map(|y| map.row(y).iter().map(|v| v.to_bits()))
                    .collect::<Vec<_>>(),
                expected_map
            );
        }
        println!(
            "PASS selected Margarine {w}x{h}: five scalar bits, opt-in native map bits, stride; score-only results retain no map"
        );
    }
}
