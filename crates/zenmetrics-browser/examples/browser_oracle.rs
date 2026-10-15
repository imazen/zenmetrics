//! Native library oracle, independent of the browser binding and worker ingress.
use zenmetrics_abstractions::zenpixels;
use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};
fn main() {
    let (w, h) = (64, 64);
    let reference: Vec<[u8; 3]> = (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| std::array::from_fn(|c| ((x * 17 + y * 13 + c * 41) % 256) as u8))
        })
        .collect();
    let distorted: Vec<_> = reference
        .iter()
        .map(|p| p.map(|v| v.saturating_sub(7)))
        .collect();
    let result = Zensim::new(ZensimProfile::codec_target())
        .compute_with_diffmap(
            &RgbSlice::new(&reference, w, h),
            &RgbSlice::new(&distorted, w, h),
            DiffmapWeighting::default(),
        )
        .unwrap();
    let r: &[rgb::RGB<u8>] = bytemuck::cast_slice(&reference);
    let d: &[rgb::RGB<u8>] = bytemuck::cast_slice(&distorted);
    let mut maps = serde_json::Map::new();
    let butter = butteraugli::butteraugli(
        butteraugli::ImgRef::new(r, w, h),
        butteraugli::ImgRef::new(d, w, h),
        &butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
    )
    .unwrap()
    .diffmap
    .unwrap();
    maps.insert(
        "butteraugli".into(),
        serde_json::json!({"width":w,"height":h,"similarity":false,"values":butter.buf()}),
    );
    let mut dssim = dssim_core::Dssim::new();
    dssim.set_save_ssim_maps(1);
    let dr = dssim.create_image_rgb(r, w, h).unwrap();
    let dd = dssim.create_image_rgb(d, w, h).unwrap();
    let (_, ssim) = dssim.compare(&dr, dd);
    let ssim = &ssim[0].map;
    maps.insert("dssim".into(),serde_json::json!({"width":ssim.width(),"height":ssim.height(),"similarity":true,"values":ssim.buf()}));
    let mut cv = cvvdp::Cvvdp::new(w as u32, h as u32, Default::default()).unwrap();
    let mut cvmap = Vec::new();
    cv.score_with_diffmap(
        bytemuck::cast_slice(&reference),
        bytemuck::cast_slice(&distorted),
        &mut cvmap,
    )
    .unwrap();
    maps.insert(
        "cvvdp".into(),
        serde_json::json!({"width":w,"height":h,"similarity":false,"values":cvmap}),
    );
    let mut rg = vec![0.; w * h];
    let mut dg = vec![0.; w * h];
    gmsd::rgb8_to_gray(bytemuck::cast_slice(&reference), w, h, w * 3, &mut rg).unwrap();
    gmsd::rgb8_to_gray(bytemuck::cast_slice(&distorted), w, h, w * 3, &mut dg).unwrap();
    let (mw, mh) = gmsd::map_dims(w, h);
    let mut gmap = vec![0.; mw * mh];
    gmsd::gmsd_with_map(
        gmsd::GrayImage::packed(&rg, w, h).unwrap(),
        gmsd::GrayImage::packed(&dg, w, h).unwrap(),
        &mut gmap,
    )
    .unwrap();
    maps.insert(
        "gmsd".into(),
        serde_json::json!({"width":mw,"height":mh,"similarity":true,"values":gmap}),
    );
    let psnr: Vec<f32> = reference
        .iter()
        .zip(&distorted)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| {
                    let delta = (*x as f32 - *y as f32) / 255.;
                    delta * delta
                })
                .sum::<f32>()
                / 3.
        })
        .collect();
    maps.insert(
        "psnr".into(),
        serde_json::json!({"width":w,"height":h,"similarity":false,"values":psnr}),
    );
    let (_, fs) = fsim::fsim_rgb8_with_diffmap(
        bytemuck::cast_slice(&reference),
        bytemuck::cast_slice(&distorted),
        w,
        h,
        w * 3,
    )
    .unwrap();
    let (_, ha) = haarpsi::haarpsi_rgb8_with_diffmap(
        bytemuck::cast_slice(&reference),
        bytemuck::cast_slice(&distorted),
        w,
        h,
        w * 3,
    )
    .unwrap();
    let (_, vs) = vsi::vsi_rgb8_with_diffmap(
        bytemuck::cast_slice(&reference),
        bytemuck::cast_slice(&distorted),
        w,
        h,
        w * 3,
    )
    .unwrap();
    let (_, ms) = msssim::msssim_rgb8_with_diffmap(
        bytemuck::cast_slice(&reference),
        bytemuck::cast_slice(&distorted),
        w,
        h,
        w * 3,
    )
    .unwrap();
    for (name, eval) in [
        ("fsim", &fs as &dyn zenmetrics_abstractions::Evaluation),
        ("fsimc", &fs),
        ("haarpsi", &ha),
        ("vsi", &vs),
        ("msssim", &ms),
    ] {
        let map = eval.map().unwrap();
        let g = map.geometry;
        maps.insert(name.into(),serde_json::json!({"width":map.width(),"height":map.height(),"similarity":true,"values":map.values(),"geometry":{
            "inputWidth":g.input_width,"inputHeight":g.input_height,"originX":g.origin_x,"originY":g.origin_y,
            "cellWidth":g.cell_width,"cellHeight":g.cell_height,"cellsX":g.cells_x,"cellsY":g.cells_y
        }}));
    }
    // SSIMULACRA2 straight from fast-ssim2, independent of the browser's
    // adapter: the fused diffmap, one cell per input pixel.
    let ssim2 = fast_ssim2::compute_ssimulacra2_maps(
        &zenpixels::PixelSlice::new(
            bytemuck::cast_slice(&reference),
            w as u32,
            h as u32,
            w * 3,
            zenpixels::PixelDescriptor::RGB8_SRGB,
        )
        .unwrap(),
        &zenpixels::PixelSlice::new(
            bytemuck::cast_slice(&distorted),
            w as u32,
            h as u32,
            w * 3,
            zenpixels::PixelDescriptor::RGB8_SRGB,
        )
        .unwrap(),
    )
    .unwrap();
    maps.insert("ssim2".into(),serde_json::json!({"width":w,"height":h,"similarity":false,"values":ssim2.diffmap(),"geometry":{
        "inputWidth":w,"inputHeight":h,"originX":0,"originY":0,"cellWidth":1,"cellHeight":1,"cellsX":w,"cellsY":h
    }}));
    // Maps served through the unified metric interface itself.
    fn packed(px: &[u8], w: usize, h: usize) -> zenmetrics_abstractions::Image<'_> {
        zenmetrics_abstractions::Image::from(
            zenpixels::PixelSlice::new(
                px,
                w as u32,
                h as u32,
                w * 3,
                zenpixels::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap(),
        )
    }
    let mut native = |name: &str,
                      metric: &mut dyn zenmetrics_abstractions::Metric,
                      r: zenmetrics_abstractions::Image<'_>,
                      d: zenmetrics_abstractions::Image<'_>| {
        let opts = zenmetrics_abstractions::EvalOpts::SCORE_ONLY.with_native_map();
        let eval = metric.evaluate(r, d, opts).unwrap();
        let map = eval.map().unwrap();
        let g = map.geometry;
        let similarity = map.polarity == zenmetrics_abstractions::MapPolarity::Similarity;
        maps.insert(name.into(),serde_json::json!({"width":map.width(),"height":map.height(),"similarity":similarity,"values":map.values(),"geometry":{
            "inputWidth":g.input_width,"inputHeight":g.input_height,"originX":g.origin_x,"originY":g.origin_y,
            "cellWidth":g.cell_width,"cellHeight":g.cell_height,"cellsX":g.cells_x,"cellsY":g.cells_y
        }}));
    };
    // One larger pair for all of them: NLPD needs both sides >= 96, vifvec 72,
    // IW-SSIM 176, and VMAF one side >= 216. Horizontal bands lose contrast
    // (MAD's high-quality strategy ignores masked errors); elsewhere SPEED's
    // chroma sees the blue channel halved in vertical stripes.
    let (lw, lh) = (480, 352);
    let large: Vec<u8> = (0..lh)
        .flat_map(|y| {
            (0..lw).flat_map(move |x| {
                (0..3).map(move |c| ((x * 7 + y * 5 + c * 61 + (x * y) % 23) % 256) as u8)
            })
        })
        .collect();
    let striped: Vec<u8> = large
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let (x, y, c) = ((i / 3) % lw, i / 3 / lw, i % 3);
            if (y / 24) % 3 == 0 {
                v / 4 * 3 + 32
            } else if c == 2 && (x / 32) % 3 == 0 {
                v >> 1
            } else {
                v
            }
        })
        .collect();
    let mut iw = iwssim::Iwssim::new(lw as u32, lh as u32).unwrap();
    let rgb: [(&str, &mut dyn zenmetrics_abstractions::Metric); 7] = [
        ("nlpd", &mut nlpd::Nlpd),
        ("psnrhvs", &mut psnrhvs::Psnrhvs),
        ("mdctpsnr", &mut mdctpsnr::MdctPsnr),
        ("mad", &mut mad_iqa::Mad),
        ("iwssim", &mut iw),
        ("vif", &mut vif::Vif),
        ("vifvec", &mut vif::Vifvec),
    ];
    for (name, metric) in rgb {
        native(
            name,
            metric,
            packed(&large, lw, lh),
            packed(&striped, lw, lh),
        );
    }
    // BT.709 limited-range 8-bit 4:2:0 with 2×2 box-averaged chroma, the
    // `vmaf-v1` ingress contract, restated here independently of the CLI.
    let yuv420 = |rgb: &[u8]| {
        let (mut y, mut cb, mut cr) = (
            vec![0u8; lw * lh],
            vec![0u8; lw * lh / 4],
            vec![0u8; lw * lh / 4],
        );
        for by in 0..lh / 2 {
            for bx in 0..lw / 2 {
                let (mut u, mut v) = (0.0f64, 0.0f64);
                for p in [0, 1]
                    .into_iter()
                    .flat_map(|dy| [0, 1].map(|dx| (2 * by + dy) * lw + 2 * bx + dx))
                {
                    let [r, g, b] = [0, 1, 2].map(|c| f64::from(rgb[3 * p + c]) / 255.0);
                    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    y[p] = (16.0_f64 + 219.0 * lum).round().clamp(16.0, 235.0) as u8;
                    u += -0.114572 * r - 0.385428 * g + 0.5 * b;
                    v += 0.5 * r - 0.454153 * g - 0.045847 * b;
                }
                cb[by * (lw / 2) + bx] =
                    (128.0_f64 + 224.0 * u / 4.0).round().clamp(16.0, 240.0) as u8;
                cr[by * (lw / 2) + bx] =
                    (128.0_f64 + 224.0 * v / 4.0).round().clamp(16.0, 240.0) as u8;
            }
        }
        [y, cb, cr]
    };
    fn planes(
        [y, cb, cr]: &[Vec<u8>; 3],
        w: usize,
        h: usize,
    ) -> zenmetrics_abstractions::Image<'_> {
        let plane = |p, w: usize, h: usize| {
            zenpixels::PixelSlice::new(p, w as u32, h as u32, w, zenpixels::PixelDescriptor::GRAY8)
                .unwrap()
        };
        zenmetrics_abstractions::Image::from(
            zenmetrics_abstractions::Planes::new(
                zenpixels::PlaneLayout::ycbcr_420(zenpixels::ChannelType::U8),
                vec![
                    plane(y, w, h),
                    plane(cb, w / 2, h / 2),
                    plane(cr, w / 2, h / 2),
                ],
            )
            .unwrap(),
        )
    }
    let (ly, sy) = (yuv420(&large), yuv420(&striped));
    native(
        "vmaf-v1",
        &mut vmaf::Vmaf::new(vmaf::ModelVariant::Standard1080p),
        planes(&ly, lw, lh),
        planes(&sy, lw, lh),
    );
    let (ow, oh) = (257, 253);
    let odd: Vec<u8> = (0..ow * oh * 4)
        .filter(|i| i % 4 != 3)
        .map(|i| ((i * 17 + (i / (ow * 4)) * 13) % 256) as u8)
        .collect();
    let (_, identity_vsi) = vsi::vsi_rgb8_with_diffmap(&odd, &odd, ow, oh, ow * 3).unwrap();
    let odd_vsi = identity_vsi.map().to_vec();
    let mut profiles = serde_json::Map::new();
    #[allow(deprecated)]
    let named = [
        ("b", ZensimProfile::B),
        ("c", ZensimProfile::C),
        ("d", ZensimProfile::D),
        ("a", ZensimProfile::A),
        ("preview-v0.1", ZensimProfile::PreviewV0_1),
        ("preview-v0.2", ZensimProfile::PreviewV0_2),
    ];
    for (name, profile) in named {
        let scorer = Zensim::new(profile);
        let score = scorer
            .compute(
                &RgbSlice::new(&reference, w, h),
                &RgbSlice::new(&distorted, w, h),
            )
            .unwrap()
            .score();
        let map = scorer.compute_with_diffmap(
            &RgbSlice::new(&reference, w, h),
            &RgbSlice::new(&distorted, w, h),
            DiffmapWeighting::default(),
        );
        if name == "c" {
            assert!(
                map.is_err(),
                "Update C map support when its native route is fixed"
            );
            profiles.insert(name.into(), serde_json::json!({"score":score,"map":null}));
        } else {
            let map = map.unwrap();
            profiles.insert(
                name.into(),
                serde_json::json!({"score":score,"map":map.diffmap()}),
            );
        }
    }
    println!(
        "{}",
        serde_json::json!({"score":result.score(),"map":result.diffmap(),"spatial_maps":maps,"profiles":profiles,"odd_vsi_identity":odd_vsi})
    );
}
