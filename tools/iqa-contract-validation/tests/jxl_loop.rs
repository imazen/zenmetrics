//! The existing JXL-shaped native route: borrowed planar f32 ingress and a
//! reusable export buffer. This is not a claim that the kernel exports no copy.
use zenpsy_api::{
    Direction, MapGrid, MapView, ScoreSpec,
    zenpixels::plane::{AnyPlane, PlaneRef},
};

#[test]
fn jxl_shape_keeps_float_inputs_and_reuses_map_export_storage() {
    let (w, h, stride) = (97, 73, 104);
    let reference: [Vec<f32>; 3] = core::array::from_fn(|c| {
        (0..w * h)
            .map(|i| ((i * 31 + (i / w) * 47 + c * 63) % 190 + 25) as f32 / 255.0)
            .collect()
    });
    // This fixture declares relative linear BT.709/D65, with Butteraugli's
    // native SDR 80 cd/m² target. It is not an HDR or wide-gamut qualification.
    let prepared = butteraugli::ButteraugliReference::new_linear_planar(
        &reference[0],
        &reference[1],
        &reference[2],
        w,
        h,
        w,
        butteraugli::ButteraugliParams::default()
            .with_intensity_target(80.0)
            .with_compute_diffmap(true),
    )
    .unwrap();
    // JXL retains full padded reconstruction buffers, including the last row's
    // padding. The current native entry requires stride*height backing storage.
    let mut candidate: [Vec<f32>; 3] = core::array::from_fn(|c| {
        let mut p = vec![0.0; stride * h];
        for y in 0..h {
            for x in 0..w {
                p[y * stride + x] = reference[c][y * w + x];
            }
        }
        p
    });
    let pointers = candidate.each_ref().map(|p| p.as_ptr());
    let mut map = vec![0.0; w * h];
    let allocation = map.as_ptr();
    static SPEC: ScoreSpec = ScoreSpec::new("diffmap", Direction::Lower).with_perfect(0.0);
    for delta in [0.01, 0.03] {
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    candidate[c][y * stride + x] = reference[c][y * w + x]
                        + if (x + 3 * y + c) % 7 == 0 { delta } else { 0.0 };
                }
            }
        }
        let original = candidate.clone(); // Test oracle only; never an adapter copy.
        let expected = prepared
            .compare_linear_planar(&candidate[0], &candidate[1], &candidate[2], stride)
            .unwrap();
        let (score, p3) = prepared
            .compare_linear_planar_into(
                &candidate[0],
                &candidate[1],
                &candidate[2],
                stride,
                &mut map,
            )
            .unwrap();
        assert_eq!(
            (score.to_bits(), p3.to_bits()),
            (expected.score.to_bits(), expected.pnorm_3.to_bits())
        );
        assert_eq!(candidate, original);
        assert_eq!(candidate.each_ref().map(|p| p.as_ptr()), pointers);
        assert_eq!(map.as_ptr(), allocation);
        let borrowed = MapView::new(
            &SPEC,
            AnyPlane::F32(PlaneRef::new(&map[..], w, h, w).unwrap()),
            MapGrid::new(w as u32, h as u32).expect("native map geometry"),
        );
        let AnyPlane::F32(view) = borrowed.values else {
            panic!("native f32 map")
        };
        assert_eq!(view.strided_samples().as_ptr(), allocation);
        assert_eq!(
            view.rows()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .diffmap
                .as_ref()
                .unwrap()
                .pixels()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
    }
}
