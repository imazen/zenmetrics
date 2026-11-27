//! Metric inputs: one packed [`PixelView`] or a [`PlanarView`].
//!
//! Most metrics take packed RGB, but video metrics (VMAF) are defined on
//! YCbCr planes with subsampled chroma, which a packed view can't describe.
//! [`Image`] carries either using `zenpixels 0.5`'s native [`PixelView`] and
//! [`PlanarView`]; [`InputFormat`] is its pixel-free description for
//! [`crate::Metric::accepts`].

use alloc::borrow::Cow;
use alloc::format;
use alloc::vec::Vec;

pub use zenpixels::{
    BitPacking, ChannelOrder, ColorSpace, PixelBuf, PixelFormat, PixelView, PlanarBuf, PlanarView,
    PlaneBuf, PlaneFormat, PlaneRole, PlaneView, Primaries, SampleLayout, SampleType, SignalRange,
    Transfer, formats,
};

use crate::error::{ErrorKind, MetricError, MetricResult, fail};

/// One image a metric scores: a packed [`PixelView`] or a [`PlanarView`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Image<'a> {
    /// Interleaved pixels; layout, depth and color ride in [`PixelFormat`].
    Packed(PixelView<'a>),
    /// Separate planes (YCbCr, planar RGB) carried directly as a [`PlanarView`].
    Planar(PlanarView<'a>),
}

impl<'a> From<PixelView<'a>> for Image<'a> {
    fn from(s: PixelView<'a>) -> Self {
        Image::Packed(s)
    }
}

impl<'a> From<PlanarView<'a>> for Image<'a> {
    fn from(p: PlanarView<'a>) -> Self {
        Image::Planar(p)
    }
}

impl<'a> Image<'a> {
    /// Luma-resolution width.
    pub fn width(&self) -> u32 {
        match self {
            Image::Packed(s) => s.width(),
            Image::Planar(p) => p.width(),
        }
    }

    /// Luma-resolution height.
    pub fn height(&self) -> u32 {
        match self {
            Image::Packed(s) => s.height(),
            Image::Planar(p) => p.height(),
        }
    }

    /// The pixel-free description of this input.
    pub fn format(&self) -> InputFormat {
        match self {
            Image::Packed(s) => InputFormat::Packed(s.format().clone()),
            Image::Planar(p) => InputFormat::Planar(PlanarFormat::from_view(p)),
        }
    }

    /// The packed view, if this is one.
    pub fn as_packed(&self) -> Option<&PixelView<'a>> {
        match self {
            Image::Packed(s) => Some(s),
            Image::Planar(_) => None,
        }
    }

    /// The planar view, if this is planar.
    pub fn as_planar(&self) -> Option<&PlanarView<'a>> {
        match self {
            Image::Planar(p) => Some(p),
            Image::Packed(_) => None,
        }
    }

    /// Fallibly copy the pixels into an [`OwnedImage`] (for warm references
    /// and sessions that must outlive the caller's buffers).
    ///
    /// # Errors
    /// [`ErrorKind::Resource`] when allocation fails.
    #[track_caller]
    pub fn try_to_owned_image(&self) -> MetricResult<OwnedImage> {
        match self {
            Image::Packed(s) => s
                .to_tight_buf()
                .map(OwnedImage::Packed)
                .map_err(|e| e.map_error(|b| MetricError::from_detail(ErrorKind::Resource, b))),
            Image::Planar(p) => copy_planar_to_buf(p).map(OwnedImage::Planar),
        }
    }

    /// Copy the pixels into an [`OwnedImage`].
    ///
    /// # Panics
    /// Panics if backing allocation fails (prefer [`Self::try_to_owned_image`]
    /// in fallible code).
    pub fn to_owned_image(&self) -> OwnedImage {
        self.try_to_owned_image()
            .expect("owned image allocation failed")
    }
}

#[track_caller]
fn copy_planar_to_buf(src: &PlanarView<'_>) -> MetricResult<PlanarBuf> {
    let formats: Vec<PlaneFormat> = src.planes().iter().map(PlaneView::format).collect();
    let mut dst = PlanarBuf::new(src.width(), src.height(), &formats)
        .map_err(|e| e.map_error(|b| MetricError::from_detail(ErrorKind::Resource, b)))?;
    dst = dst.with_chroma_siting(src.chroma_siting());
    if let Some(c) = src.color() {
        dst = dst.with_color(c.clone());
    }
    if let Some(m) = src.matrix_coefficients() {
        dst = dst.with_matrix_coefficients(m);
    }
    if let Some(r) = src.effective_range() {
        dst = dst.with_range_override(r);
    }
    if let Some(a) = src.effective_alpha() {
        dst = dst.with_alpha_override(a);
    }
    for (idx, sp) in src.planes().iter().enumerate() {
        if let Some(mut dp) = dst.plane_view_mut(idx) {
            for y in 0..sp.height() {
                dp.row_mut(y).copy_from_slice(sp.row(y));
            }
        }
    }
    Ok(dst)
}

/// Pixel-free description of a [`PlanarView`] input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanarFormat {
    /// Plane specifications, in plane order.
    pub planes: Vec<PlaneFormat>,
    /// Optional color space attached to the planar image.
    pub color: Option<ColorSpace>,
    /// Effective signal range, if declared.
    pub signal_range: Option<SignalRange>,
}

impl PlanarFormat {
    /// Extract the pixel-free description from a [`PlanarView`].
    pub fn from_view(view: &PlanarView<'_>) -> Self {
        Self {
            planes: view.planes().iter().map(PlaneView::format).collect(),
            color: view.color().cloned(),
            signal_range: view.effective_range(),
        }
    }

    /// Standard 8-bit YCbCr 4:2:0 plane specifications.
    pub fn ycbcr_420_u8() -> Self {
        Self {
            planes: alloc::vec![
                PlaneFormat::new(PlaneRole::Y, SampleType::U8),
                PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
                PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
            ],
            color: None,
            signal_range: None,
        }
    }

    /// Standard 10-bit-in-U16 (LSB-aligned) YCbCr 4:2:0 plane specifications.
    pub fn ycbcr_420_u10() -> Self {
        let pack = BitPacking::Lsb { bits: 10 };
        Self {
            planes: alloc::vec![
                PlaneFormat::new(PlaneRole::Y, SampleType::U16).with_packing(pack),
                PlaneFormat::new(PlaneRole::Cb, SampleType::U16)
                    .with_packing(pack)
                    .with_subsampling(2, 2),
                PlaneFormat::new(PlaneRole::Cr, SampleType::U16)
                    .with_packing(pack)
                    .with_subsampling(2, 2),
            ],
            color: None,
            signal_range: None,
        }
    }
}

/// What an input looks like, without its pixels — what
/// [`crate::Metric::accepts`] inspects.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputFormat {
    /// A packed view with this format.
    Packed(PixelFormat),
    /// A planar view with these plane formats and color signaling.
    Planar(PlanarFormat),
}

impl InputFormat {
    /// The packed [`PixelFormat`], if packed.
    pub fn packed(&self) -> Option<&PixelFormat> {
        match self {
            InputFormat::Packed(d) => Some(d),
            InputFormat::Planar(_) => None,
        }
    }

    /// The planar description, if planar.
    pub fn planar(&self) -> Option<&PlanarFormat> {
        match self {
            InputFormat::Planar(p) => Some(p),
            InputFormat::Packed(_) => None,
        }
    }
}

/// An owned copy of an [`Image`].
#[non_exhaustive]
#[derive(Debug)]
pub enum OwnedImage {
    /// Packed pixels.
    Packed(PixelBuf),
    /// Planar pixels.
    Planar(PlanarBuf),
}

impl OwnedImage {
    /// Borrow as an [`Image`].
    pub fn as_image(&self) -> Image<'_> {
        match self {
            OwnedImage::Packed(b) => Image::Packed(b.as_view()),
            OwnedImage::Planar(p) => Image::Planar(p.as_view()),
        }
    }
}

/// A borrowed strided sRGB8 image — what [`rgb8_pair`] returns.
#[derive(Clone, Copy, Debug)]
pub struct Rgb8In<'a> {
    /// `RGBRGB…` bytes, rows `stride` apart.
    pub pixels: &'a [u8],
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row stride in bytes (≥ `width·3`).
    pub stride: usize,
}

/// Whether `format` is packed `RGB8_SRGB` — the native input of most
/// in-tree metrics.
pub fn is_rgb8_srgb(format: &InputFormat) -> bool {
    format.packed() == Some(&formats::RGB8_SRGB)
}

/// Borrow a reference/distorted pair as strided packed sRGB8.
///
/// # Errors
/// [`ErrorKind::UnsupportedInput`] when either image isn't packed
/// `RGB8_SRGB`; [`ErrorKind::DimensionMismatch`] when their sizes differ.
#[track_caller]
pub fn rgb8_pair<'a>(r: &Image<'a>, d: &Image<'a>) -> MetricResult<(Rgb8In<'a>, Rgb8In<'a>)> {
    let a = rgb8_view(r)?;
    let b = rgb8_view(d)?;
    check_same_size(r, d)?;
    Ok((a, b))
}

/// Borrow one image as strided packed sRGB8.
///
/// # Errors
/// [`ErrorKind::UnsupportedInput`] when it isn't packed `RGB8_SRGB`.
#[track_caller]
pub fn rgb8_view<'a>(img: &Image<'a>) -> MetricResult<Rgb8In<'a>> {
    match img {
        Image::Packed(s) if s.format() == &formats::RGB8_SRGB => Ok(Rgb8In {
            pixels: s.as_strided_bytes(),
            width: s.width() as usize,
            height: s.height() as usize,
            stride: s.stride_bytes(),
        }),
        other => fail(unsupported(&other.format(), "packed RGB8_SRGB")),
    }
}

/// The [`ErrorKind::UnsupportedInput`] error for `format`, naming what the
/// metric wanted.
pub fn unsupported(format: &InputFormat, wanted: &'static str) -> MetricError {
    MetricError::new(ErrorKind::UnsupportedInput)
        .with_message(format!("wanted {wanted}, got {format:?}"))
}

/// [`ErrorKind::DimensionMismatch`] unless both images have one size.
///
/// # Errors
/// As described.
#[track_caller]
pub fn check_same_size(r: &Image<'_>, d: &Image<'_>) -> MetricResult<()> {
    if (r.width(), r.height()) != (d.width(), d.height()) {
        return fail(
            MetricError::new(ErrorKind::DimensionMismatch).with_message(format!(
                "reference {}x{}, distorted {}x{}",
                r.width(),
                r.height(),
                d.width(),
                d.height()
            )),
        );
    }
    Ok(())
}

/// Repack a strided rgb8 view to packed `RGBRGB…` — a borrow when the
/// stride is already `width·3`, a copy otherwise. For metrics whose native
/// entry takes packed input only.
#[must_use]
pub fn contiguous_rgb8<'a>(r: &Rgb8In<'a>) -> Cow<'a, [u8]> {
    let row = r.width * 3;
    if r.stride == row {
        Cow::Borrowed(&r.pixels[..row * r.height])
    } else {
        let mut out = Vec::with_capacity(row * r.height);
        for y in 0..r.height {
            out.extend_from_slice(&r.pixels[y * r.stride..y * r.stride + row]);
        }
        Cow::Owned(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn planar_420_validate_and_round_trip() {
        let (y, u, v) = (vec![1u8; 5 * 3], vec![2u8; 3 * 2], vec![3u8; 3 * 2]);
        let py =
            PlaneView::new(&y, 5, 3, 5, PlaneFormat::new(PlaneRole::Y, SampleType::U8)).unwrap();
        let pu = PlaneView::new(
            &u,
            3,
            2,
            3,
            PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
        )
        .unwrap();
        let pv = PlaneView::new(
            &v,
            3,
            2,
            3,
            PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
        )
        .unwrap();
        let p = PlanarView::new(5, 3, &[py, pu, pv]).unwrap();
        assert_eq!((p.width(), p.height()), (5, 3));
        let img = Image::from(p);
        let owned = img.to_owned_image();
        let back = owned.as_image();
        assert_eq!(back.format(), img.format());
        assert_eq!(back.as_planar().unwrap().planes()[2].row(1), &[3, 3, 3]);
    }

    #[test]
    fn planar_rejects_wrong_chroma_size() {
        let (y, u) = (vec![0u8; 16], vec![0u8; 4]);
        let py =
            PlaneView::new(&y, 4, 4, 4, PlaneFormat::new(PlaneRole::Y, SampleType::U8)).unwrap();
        let pu = PlaneView::new(
            &u,
            2,
            2,
            2,
            PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
        )
        .unwrap();
        let pv = PlaneView::new(
            &u,
            1,
            4,
            1,
            PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
        )
        .unwrap();
        assert!(PlanarView::new(4, 4, &[py, pu, pv]).is_err());
    }

    #[test]
    fn rgb8_pair_distinguishes_format_from_size() {
        let a = vec![0u8; 2 * 2 * 3];
        let b = vec![0u8; 3 * 2 * 3];
        let ra = Image::from(PixelView::new(&a, 2, 2, 6, &formats::RGB8_SRGB).unwrap());
        let rb = Image::from(PixelView::new(&b, 3, 2, 9, &formats::RGB8_SRGB).unwrap());
        assert_eq!(
            rgb8_pair(&ra, &rb).unwrap_err().error().kind(),
            ErrorKind::DimensionMismatch
        );
        let g = vec![0u8; 4];
        let rg = Image::from(PixelView::new(&g, 2, 2, 2, &formats::GRAY8_SRGB).unwrap());
        assert_eq!(
            rgb8_pair(&ra, &rg).unwrap_err().error().kind(),
            ErrorKind::UnsupportedInput
        );
        assert!(rgb8_pair(&ra, &ra).is_ok());
    }

    #[test]
    fn contiguous_rgb8_repacks_strided_rows() {
        let px = [1u8, 2, 3, 9, 4, 5, 6, 9];
        let v = Rgb8In {
            pixels: &px,
            width: 1,
            height: 2,
            stride: 4,
        };
        assert_eq!(&*contiguous_rgb8(&v), &[1, 2, 3, 4, 5, 6]);
    }
}
