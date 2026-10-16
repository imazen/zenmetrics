//! Behaviour gates for `brisque`: strided input is bit-identical to
//! packed, `PairedNoReference<Brisque>` conforms to the generic contract,
//! a warm reference evaluates the reference once and cold == warm
//! bitwise, images too small for a second scale fail with
//! `ErrorKind::TooSmall`, and content the oracle rejects (flat GGD sigma)
//! fails with `ErrorKind::Degenerate` instead of a panic/NaN.

use brisque::Brisque;
use za::conformance::{assert_conforms, distort_rgb8, rgb8_image, scores, textured_rgb8};
use za::zenpixels::{PixelDescriptor, PixelSlice};
use za::{EvalOpts, Image, Metric, NoReferenceMetric, PairedNoReference};
use zenmetrics_abstractions as za;

fn evaluate(img: Image<'_>) -> Vec<(std::borrow::Cow<'static, str>, f64)> {
    scores(&*Brisque::new().evaluate_image(img, EvalOpts::new()).unwrap())
}

#[test]
fn strided_equals_packed_bit_for_bit() {
    let (w, h) = (197usize, 101usize);
    let packed = textured_rgb8(w, h, 7);
    // Same pixels with 21 bytes (7 pixels) of stride padding per row.
    let stride = w * 3 + 21;
    let mut buf = vec![0u8; stride * h];
    for y in 0..h {
        buf[y * stride..y * stride + w * 3].copy_from_slice(&packed[y * w * 3..(y + 1) * w * 3]);
    }
    let strided: Image<'_> =
        PixelSlice::new(&buf, w as u32, h as u32, stride, PixelDescriptor::RGB8_SRGB)
            .unwrap()
            .into();
    let a = evaluate(rgb8_image(&packed, w, h));
    let b = evaluate(strided);
    assert_eq!(a.len(), b.len());
    for ((ka, va), (kb, vb)) in a.iter().zip(b.iter()) {
        assert_eq!(ka, kb);
        assert_eq!(va.to_bits(), vb.to_bits(), "{ka} differs strided vs packed");
    }
}

#[test]
fn paired_adapter_conforms() {
    let mut m = PairedNoReference::new(Brisque::new());
    let a = textured_rgb8(288, 192, 1);
    let b = distort_rgb8(&a, 288, 192);
    assert_conforms(&mut m, &rgb8_image(&a, 288, 192), &rgb8_image(&b, 288, 192));
}

#[test]
fn warm_reference_equals_cold_bit_for_bit() {
    let (w, h) = (192usize, 100usize);
    let a = textured_rgb8(w, h, 3);
    let b = distort_rgb8(&a, w, h);
    let (ra, rb) = (rgb8_image(&a, w, h), rgb8_image(&b, w, h));
    let mut m = PairedNoReference::new(Brisque::new());
    let cold = scores(&*m.evaluate(ra.clone(), rb.clone(), EvalOpts::new()).unwrap());
    let mut warm = m.warm_reference(ra, EvalOpts::new()).unwrap();
    assert!(warm.reuses_reference_work());
    let hot = scores(&*warm.evaluate(rb, EvalOpts::new()).unwrap());
    assert_eq!(cold.len(), hot.len());
    for ((ka, va), (kb, vb)) in cold.iter().zip(hot.iter()) {
        assert_eq!(ka, kb);
        assert_eq!(va.to_bits(), vb.to_bits(), "{ka} differs warm vs cold");
    }
}

#[test]
fn too_small_is_too_small_not_panic() {
    // All unsupported widths/heights must refuse before reflect padding.
    let cases = (1..=8)
        .flat_map(|w| (1..=8).map(move |h| (w, h)))
        .filter(|&(w, h)| w < 8 || h < 8)
        .chain([(1usize, 100), (100, 1)]);
    for (w, h) in cases {
        let data = vec![128u8; w * h * 3];
        match Brisque::new().evaluate_image(rgb8_image(&data, w, h), EvalOpts::new()) {
            Err(e) => assert_eq!(e.error().kind(), za::ErrorKind::TooSmall, "{w}x{h}"),
            Ok(_) => panic!("{w}x{h}: expected TooSmall"),
        }
    }
}

#[test]
fn eight_by_eight_has_supported_geometry() {
    let data = textured_rgb8(8, 8, 7);
    match Brisque::new().evaluate_image(rgb8_image(&data, 8, 8), EvalOpts::new()) {
        Ok(e) => assert!(e.score().is_finite()),
        Err(e) => assert_eq!(e.error().kind(), za::ErrorKind::Degenerate),
    }
}

#[test]
fn gray8_accepted() {
    // GRAY8 planes score — a contentful luma plane.
    let (w, h) = (192usize, 192usize);
    let rgb = textured_rgb8(w, h, 5);
    let lum: Vec<u8> = rgb
        .chunks_exact(3)
        .map(|p| ((77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8) as u8)
        .collect();
    let gimg: Image<'_> = PixelSlice::new(&lum, w as u32, h as u32, w, PixelDescriptor::GRAY8)
        .unwrap()
        .into();
    let v = evaluate(gimg);
    assert!(v[0].1.is_finite(), "GRAY8 brisque should be finite: {v:?}");
}

#[test]
fn degenerate_content_is_degenerate_not_nan() {
    // Flat plane: GGD sigma ~0 makes the oracle's gamfit raise (and the
    // single-sign AGGD products give a zero denominator); the port
    // reports ErrorKind::Degenerate.
    let (w, h) = (192usize, 192usize);
    let data = vec![128u8; w * h * 3];
    match Brisque::new().evaluate_image(rgb8_image(&data, w, h), EvalOpts::new()) {
        Err(e) => assert_eq!(e.error().kind(), za::ErrorKind::Degenerate, "{w}x{h} flat"),
        Ok(v) => panic!("{w}x{h} flat: expected Degenerate, got {:?}", scores(&*v)),
    }
}
