//! Native library oracle, independent of the browser binding and worker ingress.
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
