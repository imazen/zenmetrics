//! Native output adapters; no IO or display conversion inside these calls.
use za::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions as za;
#[cfg(iqa_compete_ingress)]
use zenmetrics_abstractions::Metric;
use zenmetrics_cli::decode::Rgb8Image;
#[path = "../../../../../crates/zenmetrics-cli/src/metrics/butteraugli_original.rs"]
mod butter_original;
#[path = "../../../../../crates/zenmetrics-cli/src/metrics/flip.rs"]
mod flip;
#[cfg(iqa_compete_ingress)]
#[allow(dead_code, unexpected_cfgs, unreachable_pub)]
#[path = "../../../../../crates/zenmetrics-cli/src/metrics/vmaf.rs"]
mod ingress;
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
    if name == "zensim" {
        let scorer = zensim::Zensim::new(zensim::ZensimProfile::latest_preview());
        return Some(Box::new(move || {
            let result = scorer
                .compute_with_diffmap(
                    &zensim::RgbSlice::new(bytemuck::cast_slice::<u8, [u8; 3]>(&r.pixels), w, h),
                    &zensim::RgbSlice::new(bytemuck::cast_slice::<u8, [u8; 3]>(&d.pixels), w, h),
                    zensim::DiffmapWeighting::default(),
                )
                .map_err(|e| e.to_string())?;
            std::hint::black_box(&result);
            Ok(result.score())
        }));
    }
    if name == "dssim" {
        let mut scorer = dssim_core::Dssim::new();
        scorer.set_save_ssim_maps(5);
        return Some(Box::new(move || {
            let rr = scorer
                .create_image_rgb(bytemuck::cast_slice::<u8, rgb::RGB<u8>>(&r.pixels), w, h)
                .ok_or("DSSIM reference input")?;
            let dd = scorer
                .create_image_rgb(bytemuck::cast_slice::<u8, rgb::RGB<u8>>(&d.pixels), w, h)
                .ok_or("DSSIM distorted input")?;
            let (score, maps) = scorer.compare(&rr, dd);
            if maps.is_empty() {
                return Err("DSSIM produced no native maps".into());
            }
            std::hint::black_box(&maps);
            Ok(f64::from(score))
        }));
    }
    #[cfg(iqa_compete_ingress)]
    if name == "vmaf-v1" {
        let mut scorer = vmaf::Vmaf::new(vmaf::ModelVariant::Standard1080p);
        return Some(Box::new(move || {
            let eval = vmaf_evaluate(&mut scorer, r, d)?;
            std::hint::black_box(&eval);
            Ok(eval.score())
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

#[cfg(iqa_compete_ingress)]
fn vmaf_image(
    planes: &(Vec<u8>, Vec<u8>, Vec<u8>),
    w: usize,
    h: usize,
) -> Result<za::Image<'_>, String> {
    use za::zenpixels::{ChannelType, PlaneLayout};
    fn gray(p: &[u8], w: usize, h: usize) -> Result<PixelSlice<'_>, String> {
        PixelSlice::new(p, w as u32, h as u32, w, PixelDescriptor::GRAY8).map_err(|e| e.to_string())
    }
    za::Planes::new(
        PlaneLayout::ycbcr_420(ChannelType::U8),
        vec![
            gray(&planes.0, w, h)?,
            gray(&planes.1, w / 2, h / 2)?,
            gray(&planes.2, w / 2, h / 2)?,
        ],
    )
    .map(za::Image::from)
    .map_err(|e| e.to_string())
}

#[cfg(iqa_compete_ingress)]
fn vmaf_evaluate(
    scorer: &mut vmaf::Vmaf,
    r: &Rgb8Image,
    d: &Rgb8Image,
) -> Result<Box<dyn za::Evaluation>, String> {
    let (w, h) = (r.width as usize & !1, r.height as usize & !1);
    let rp = ingress::to_yuv420_even(r, ingress::Ingress::Bt709).map_err(|e| e.to_string())?;
    let dp = ingress::to_yuv420_even(d, ingress::Ingress::Bt709).map_err(|e| e.to_string())?;
    scorer
        .evaluate(
            vmaf_image(&rp, w, h)?,
            vmaf_image(&dp, w, h)?,
            za::EvalOpts::SCORE_ONLY.with_native_map(),
        )
        .map_err(|e| e.to_string())
}
#[cfg(iqa_compete_ingress)]
pub(super) fn vmaf_probe_dump(
    r: &Rgb8Image,
    d: &Rgb8Image,
    path: &std::path::Path,
) -> Result<serde_json::Value, String> {
    std::fs::create_dir(path).map_err(|e| e.to_string())?;
    let eval = vmaf_evaluate(
        &mut vmaf::Vmaf::new(vmaf::ModelVariant::Standard1080p),
        r,
        d,
    )?;
    let mut manifest = Vec::new();
    for (i, m) in eval.maps().into_iter().enumerate() {
        let bytes: Vec<u8> = m
            .values()
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        let name = format!("map-{i}.f32");
        std::fs::write(path.join(&name), bytes).map_err(|e| e.to_string())?;
        manifest.push(serde_json::json!({"file":name,"name":m.name,"width":m.width(),"height":m.height(),"geometry":format!("{:?}",m.geometry)}));
    }
    Ok(serde_json::json!({"score_bits":format!("{:016x}",eval.score().to_bits()),"maps":manifest}))
}

/// Export native component buffers once, before competitor timing.
#[cfg(iqa_compete_ingress)]
pub(super) fn competitor_planes(
    image: &Rgb8Image,
    convention: &str,
) -> Result<(usize, usize, Vec<u8>), Box<dyn std::error::Error>> {
    let (w, h) = (image.width as usize, image.height as usize);
    match convention {
        "709-420" | "601-420" => {
            let matrix = if convention == "709-420" {
                ingress::Ingress::Bt709
            } else {
                ingress::Ingress::Studio601
            };
            let (y, u, v) = ingress::to_yuv420_even(image, matrix)?;
            Ok((w & !1, h & !1, [y, u, v].concat()))
        }
        "601-y" => {
            // Public converter crops odd sizes. Replicate to even first, then
            // extract the original rectangle: the per-pixel f32 formula is unchanged.
            let pw = (w + 1) & !1;
            let ph = (h + 1) & !1;
            let mut padded = Vec::with_capacity(pw * ph * 3);
            for y in 0..ph {
                for x in 0..pw {
                    let i = (y.min(h - 1) * w + x.min(w - 1)) * 3;
                    padded.extend_from_slice(&image.pixels[i..i + 3]);
                }
            }
            let gray = zenmetrics_cli::metrics::studio601_gray(&Rgb8Image {
                width: pw as u32,
                height: ph as u32,
                pixels: padded,
            });
            let mut plane = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    plane.push(gray.pixels[(y * pw + x) * 3]);
                }
            }
            Ok((w, h, plane))
        }
        "709-y" => {
            let mut plane = Vec::with_capacity(w * h);
            for rgb in image.pixels.as_chunks::<3>().0 {
                let lum = 0.2126 * (f64::from(rgb[0]) / 255.0)
                    + 0.7152 * (f64::from(rgb[1]) / 255.0)
                    + 0.0722 * (f64::from(rgb[2]) / 255.0);
                plane.push((16.0 + 219.0 * lum).round().clamp(16.0, 235.0) as u8);
            }
            Ok((w, h, plane))
        }
        _ => Err("unknown competitor component convention".into()),
    }
}
