//! Native output adapters; no IO or display conversion inside these calls.
use za::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions as za;
use zenmetrics_cli::decode::Rgb8Image;
#[path = "../../../../../crates/zenmetrics-cli/src/metrics/butteraugli_original.rs"]
mod butter_original;
#[path = "../../../../../crates/zenmetrics-cli/src/metrics/flip.rs"]
mod flip;
pub(super) type Call = Box<dyn FnMut() -> Result<f64, String> + Send>;
pub(super) fn cvvdp_score(
    r: &'static Rgb8Image,
    d: &'static Rgb8Image,
    map: bool,
) -> Result<Call, String> {
    let display = zenmetrics_cli::metrics::display::parse_display("standard_fhd")?;
    let mut scorer = cvvdp::Cvvdp::with_geometry(
        r.width,
        r.height,
        cvvdp::CvvdpParams {
            display: display.model(),
            ..Default::default()
        },
        display.geometry(),
    )
    .map_err(|e| e.to_string())?;
    let mut values = Vec::new();
    Ok(Box::new(move || {
        let value = if map {
            scorer.score_with_diffmap(&r.pixels, &d.pixels, &mut values)
        } else {
            scorer.score(&r.pixels, &d.pixels)
        }
        .map_err(|e| e.to_string())?;
        std::hint::black_box(&values);
        Ok(f64::from(value))
    }))
}
pub(super) fn native(name: &str, r: &'static Rgb8Image, d: &'static Rgb8Image) -> Option<Call> {
    let (w, h) = (r.width as usize, r.height as usize);
    if name == "cvvdp" {
        return cvvdp_score(r, d, true).ok();
    }
    if matches!(name, "flip" | "flip-v17" | "butteraugli-original") {
        let name = name.to_owned();
        return Some(Box::new(move || {
            let map = if name == "butteraugli-original" {
                butter_original::error_map(&r.pixels, &d.pixels, w, h, w * 3, w * 3)
            } else {
                flip::error_map(&r.pixels, &d.pixels, w, h, w * 3, w * 3)
            }
            .map_err(str::to_owned)?;
            let score = match name.as_str() {
                "flip" => flip::mean(&map),
                "flip-v17" => flip::mean_v17(&map),
                _ => map.iter().copied().fold(0.0_f32, f32::max) as f64,
            };
            std::hint::black_box(&map);
            Ok(score)
        }));
    }
    if name == "butteraugli" {
        return Some(Box::new(move || {
            let result = butteraugli::butteraugli(
                imgref::ImgRef::new(bytemuck::cast_slice::<u8, rgb::RGB<u8>>(&r.pixels), w, h),
                imgref::ImgRef::new(bytemuck::cast_slice::<u8, rgb::RGB<u8>>(&d.pixels), w, h),
                &butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
            )
            .map_err(|e| e.to_string())?;
            std::hint::black_box(&result.diffmap);
            Ok(result.score)
        }));
    }
    if name == "ssim2" {
        return Some(Box::new(move || {
            let pixel = |p: &'static Rgb8Image| {
                PixelSlice::new(
                    &p.pixels,
                    p.width,
                    p.height,
                    p.width as usize * 3,
                    PixelDescriptor::RGB8_SRGB,
                )
                .unwrap()
            };
            let result = fast_ssim2_current::compute_ssimulacra2_maps(&pixel(r), &pixel(d))
                .map_err(|e| e.to_string())?;
            std::hint::black_box(&result);
            Ok(result.score)
        }));
    }
    let mut metric: Box<dyn za::Metric> = match name {
        "gmsd" => Box::new(gmsd::Gmsd),
        "haarpsi" => Box::new(haarpsi::Haarpsi),
        "fsim-nodownscale" => Box::new(fsim::Fsim::with_downscale(fsim::Downscale::Factor(
            std::num::NonZeroU32::new(1).unwrap(),
        ))),
        "fsim" | "fsimc" => Box::new(fsim::Fsim::default()),
        "fsim-k2" | "fsimc-k2" | "fsim-k3" | "fsimc-k3" | "fsim-k4" | "fsimc-k4" => {
            Box::new(fsim::Fsim::with_downscale(fsim::Downscale::Factor(
                std::num::NonZeroU32::new(name.as_bytes()[name.len() - 1] as u32 - b'0' as u32)
                    .unwrap(),
            )))
        }
        "vsi-nodownscale" => Box::new(vsi::Vsi::with_downscale(vsi::Downscale::Factor(
            std::num::NonZeroU32::new(1).unwrap(),
        ))),
        "vsi" => Box::new(vsi::Vsi::default()),
        "vsi-k2" | "vsi-k3" | "vsi-k4" => {
            Box::new(vsi::Vsi::with_downscale(vsi::Downscale::Factor(
                std::num::NonZeroU32::new(name.as_bytes()[name.len() - 1] as u32 - b'0' as u32)
                    .unwrap(),
            )))
        }
        "msssim" => Box::new(msssim::Msssim),
        "nlpd" => Box::new(nlpd::Nlpd),
        "psnrhvs" => Box::new(psnrhvs::Psnrhvs),
        "mdctpsnr" => Box::new(mdctpsnr::MdctPsnr),
        "mad" => Box::new(mad_iqa::Mad),
        "iwssim" => Box::new(iwssim::Iwssim::new(r.width, r.height).ok()?),
        "vif" => Box::new(vif::Vif),
        "vifvec" => Box::new(vif::Vifvec),
        _ => return None,
    };
    let image = |p: &'static Rgb8Image| {
        za::Image::from(
            PixelSlice::new(
                &p.pixels,
                p.width,
                p.height,
                p.width as usize * 3,
                PixelDescriptor::RGB8_SRGB,
            )
            .unwrap(),
        )
    };
    let ri = image(r);
    let di = image(d);
    let column = if name.starts_with("fsimc") {
        "fsimc"
    } else if name.starts_with("fsim") {
        "fsim"
    } else {
        name
    };
    let column = column.to_owned();
    Some(Box::new(move || {
        let eval = metric
            .evaluate(
                ri.clone(),
                di.clone(),
                za::EvalOpts::SCORE_ONLY.with_native_map(),
            )
            .map_err(|e| e.to_string())?;
        if eval.maps().is_empty() {
            return Err("native evaluation returned no maps".into());
        }
        let score = eval
            .scores()
            .find(|(key, _)| *key == column)
            .map_or_else(|| eval.score(), |(_, v)| v);
        std::hint::black_box(&eval);
        Ok(score)
    }))
}
