//! Metric inputs: one packed [`PixelSlice`] or a set of planes.
//!
//! Most metrics take packed RGB, but video metrics (VMAF) are defined on
//! YCbCr planes with subsampled chroma, which a packed slice can't
//! describe. [`Image`] carries either; [`InputFormat`] is its
//! pixel-free description for [`crate::Metric::accepts`].

use alloc::borrow::Cow;
use alloc::format;
use alloc::vec::Vec;

use zenpixels::{
    ChannelLayout, ColorPrimaries, PixelBuffer, PixelCow, PixelDescriptor, PixelSlice, PlaneLayout,
    SignalRange, TransferFunction,
};

use crate::error::{ErrorKind, MetricError, MetricResult, fail};

/// One image a metric scores: a packed slice or a planar set.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Image<'a> {
    /// Interleaved pixels; layout, depth and color ride in the descriptor.
    Packed(PixelSlice<'a>),
    /// Separate planes (YCbCr, planar RGB).
    Planar(Planes<'a>),
}

impl<'a> From<PixelSlice<'a>> for Image<'a> {
    fn from(s: PixelSlice<'a>) -> Self {
        Image::Packed(s)
    }
}

impl<'a> From<Planes<'a>> for Image<'a> {
    fn from(p: Planes<'a>) -> Self {
        Image::Planar(p)
    }
}

impl<'a> Image<'a> {
    /// Luma-resolution width.
    pub fn width(&self) -> u32 {
        match self {
            Image::Packed(s) => s.width(),
            Image::Planar(p) => p.width,
        }
    }

    /// Luma-resolution height.
    pub fn height(&self) -> u32 {
        match self {
            Image::Packed(s) => s.rows(),
            Image::Planar(p) => p.height,
        }
    }

    /// The pixel-free description of this input.
    pub fn format(&self) -> InputFormat {
        match self {
            Image::Packed(s) => InputFormat::Packed(s.descriptor()),
            Image::Planar(p) => InputFormat::Planar(p.format()),
        }
    }

    /// The packed slice, if this is one.
    pub fn as_packed(&self) -> Option<&PixelSlice<'a>> {
        match self {
            Image::Packed(s) => Some(s),
            Image::Planar(_) => None,
        }
    }

    /// The planes, if this is planar.
    pub fn as_planar(&self) -> Option<&Planes<'a>> {
        match self {
            Image::Planar(p) => Some(p),
            Image::Packed(_) => None,
        }
    }

    /// Copy the pixels into an [`OwnedImage`] (for warm references and
    /// sessions that must outlive the caller's buffers).
    pub fn to_owned_image(&self) -> OwnedImage {
        match self {
            Image::Packed(s) => OwnedImage::Packed(PixelCow::Borrowed(s.clone()).into_owned()),
            Image::Planar(p) => OwnedImage::Planar {
                layout: p.layout.clone(),
                planes: p
                    .planes
                    .iter()
                    .map(|s| PixelCow::Borrowed(s.clone()).into_owned())
                    .collect(),
                width: p.width,
                height: p.height,
                significant_bits: p.significant_bits,
            },
        }
    }
}

/// A planar image: one single-channel [`PixelSlice`] per plane of a
/// [`PlaneLayout`], validated against it.
#[derive(Clone, Debug)]
pub struct Planes<'a> {
    layout: PlaneLayout,
    planes: Vec<PixelSlice<'a>>,
    width: u32,
    height: u32,
    significant_bits: u8,
}

/// Pixel-free description of a [`Planes`] input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanarFormat {
    /// Plane semantics, sample types and subsampling.
    pub layout: PlaneLayout,
    /// Transfer function shared by every plane.
    pub transfer: TransferFunction,
    /// Signal range shared by every plane.
    pub signal_range: SignalRange,
    /// Primaries shared by every plane.
    pub primaries: ColorPrimaries,
    /// Bits in use per sample: `8` for U8, `10`/`12`/`16` for U16
    /// containers.
    pub significant_bits: u8,
}

impl<'a> Planes<'a> {
    /// Validate `planes` against `layout`.
    ///
    /// Requirements: `layout` is planar with one descriptor per slice;
    /// every slice is single-channel gray of its plane's sample type; a
    /// full-resolution plane exists and fixes the image size; every other
    /// plane has exactly `plane_width`/`plane_height` of it; all planes
    /// share transfer, signal range and primaries. `significant_bits`
    /// defaults to the full container width.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidInput`] naming the first violated requirement.
    #[track_caller]
    pub fn new(layout: PlaneLayout, planes: Vec<PixelSlice<'a>>) -> MetricResult<Self> {
        let invalid = |m: Cow<'static, str>| fail(MetricError::new(ErrorKind::InvalidInput).with_message(m));
        if !layout.is_planar() {
            return invalid("plane layout is interleaved".into());
        }
        let descs = layout.planes();
        if descs.len() != planes.len() {
            return invalid(format!("layout has {} planes, got {}", descs.len(), planes.len()).into());
        }
        let Some(full) = descs.iter().position(|d| !d.is_subsampled()) else {
            return invalid("no full-resolution plane".into());
        };
        let (width, height) = (planes[full].width(), planes[full].rows());
        let first = planes[0].descriptor();
        for (i, (d, s)) in descs.iter().zip(&planes).enumerate() {
            let sd = s.descriptor();
            if sd.layout() != ChannelLayout::Gray || sd.channel_type() != d.channel_type {
                return invalid(format!("plane {i}: expected single-channel {:?}, got {:?}", d.channel_type, sd).into());
            }
            if (s.width(), s.rows()) != (d.plane_width(width), d.plane_height(height)) {
                return invalid(
                    format!(
                        "plane {i}: {}x{} but layout implies {}x{}",
                        s.width(),
                        s.rows(),
                        d.plane_width(width),
                        d.plane_height(height)
                    )
                    .into(),
                );
            }
            if (sd.transfer, sd.signal_range, sd.primaries) != (first.transfer, first.signal_range, first.primaries) {
                return invalid(format!("plane {i}: color signaling differs from plane 0").into());
            }
        }
        let significant_bits = (first.channel_type().byte_size() * 8) as u8;
        Ok(Self {
            layout,
            planes,
            width,
            height,
            significant_bits,
        })
    }

    /// Declare how many low bits of each sample are in use (10-bit video
    /// in U16 containers). Samples must not exceed `2^bits − 1`; metrics
    /// may check.
    ///
    /// # Errors
    /// [`ErrorKind::InvalidInput`] when `bits` is 0 or wider than the
    /// sample container.
    #[track_caller]
    pub fn with_significant_bits(mut self, bits: u8) -> MetricResult<Self> {
        let container = (self.planes[0].descriptor().channel_type().byte_size() * 8) as u8;
        if bits == 0 || bits > container {
            return fail(
                MetricError::new(ErrorKind::InvalidInput)
                    .with_message(format!("{bits} significant bits in a {container}-bit sample")),
            );
        }
        self.significant_bits = bits;
        Ok(self)
    }

    /// The layout.
    pub fn layout(&self) -> &PlaneLayout {
        &self.layout
    }

    /// The planes, in layout order.
    pub fn planes(&self) -> &[PixelSlice<'a>] {
        &self.planes
    }

    /// Full-resolution width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Full-resolution height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Bits in use per sample.
    pub fn significant_bits(&self) -> u8 {
        self.significant_bits
    }

    /// The pixel-free description.
    pub fn format(&self) -> PlanarFormat {
        let d = self.planes[0].descriptor();
        PlanarFormat {
            layout: self.layout.clone(),
            transfer: d.transfer,
            signal_range: d.signal_range,
            primaries: d.primaries,
            significant_bits: self.significant_bits,
        }
    }
}

/// What an input looks like, without its pixels — what
/// [`crate::Metric::accepts`] inspects.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputFormat {
    /// A packed slice with this descriptor.
    Packed(PixelDescriptor),
    /// A planar set.
    Planar(PlanarFormat),
}

impl InputFormat {
    /// The packed descriptor, if packed.
    pub fn packed(&self) -> Option<PixelDescriptor> {
        match self {
            InputFormat::Packed(d) => Some(*d),
            InputFormat::Planar(_) => None,
        }
    }
}

/// An owned copy of an [`Image`].
#[non_exhaustive]
#[derive(Debug)]
pub enum OwnedImage {
    /// Packed pixels.
    Packed(PixelBuffer),
    /// Planes, in layout order.
    Planar {
        /// The layout.
        layout: PlaneLayout,
        /// One buffer per plane.
        planes: Vec<PixelBuffer>,
        /// Full-resolution width.
        width: u32,
        /// Full-resolution height.
        height: u32,
        /// Bits in use per sample.
        significant_bits: u8,
    },
}

impl OwnedImage {
    /// Borrow as an [`Image`].
    pub fn as_image(&self) -> Image<'_> {
        match self {
            OwnedImage::Packed(b) => Image::Packed(b.as_slice()),
            OwnedImage::Planar {
                layout,
                planes,
                width,
                height,
                significant_bits,
            } => Image::Planar(Planes {
                layout: layout.clone(),
                planes: planes.iter().map(PixelBuffer::as_slice).collect(),
                width: *width,
                height: *height,
                significant_bits: *significant_bits,
            }),
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
    format.packed() == Some(PixelDescriptor::RGB8_SRGB)
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
        Image::Packed(s) if s.descriptor() == PixelDescriptor::RGB8_SRGB => Ok(Rgb8In {
            pixels: s.as_strided_bytes(),
            width: s.width() as usize,
            height: s.rows() as usize,
            stride: s.stride(),
        }),
        other => fail(unsupported(&other.format(), "packed RGB8_SRGB")),
    }
}

/// The [`ErrorKind::UnsupportedInput`] error for `format`, naming what the
/// metric wanted.
pub fn unsupported(format: &InputFormat, wanted: &'static str) -> MetricError {
    MetricError::new(ErrorKind::UnsupportedInput).with_message(format!("wanted {wanted}, got {format:?}"))
}

/// [`ErrorKind::DimensionMismatch`] unless both images have one size.
///
/// # Errors
/// As described.
#[track_caller]
pub fn check_same_size(r: &Image<'_>, d: &Image<'_>) -> MetricResult<()> {
    if (r.width(), r.height()) != (d.width(), d.height()) {
        return fail(MetricError::new(ErrorKind::DimensionMismatch).with_message(format!(
            "reference {}x{}, distorted {}x{}",
            r.width(),
            r.height(),
            d.width(),
            d.height()
        )));
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
    use zenpixels::ChannelType;

    fn gray8(data: &[u8], w: u32, h: u32) -> PixelSlice<'_> {
        PixelSlice::new(data, w, h, w as usize, PixelDescriptor::GRAY8).unwrap()
    }

    #[test]
    fn planes_420_validate_and_round_trip() {
        let (y, u, v) = (vec![1u8; 5 * 3], vec![2u8; 3 * 2], vec![3u8; 3 * 2]);
        let p = Planes::new(
            PlaneLayout::ycbcr_420(ChannelType::U8),
            vec![gray8(&y, 5, 3), gray8(&u, 3, 2), gray8(&v, 3, 2)],
        )
        .unwrap();
        assert_eq!((p.width(), p.height(), p.significant_bits()), (5, 3, 8));
        let img = Image::from(p);
        let owned = img.to_owned_image();
        let back = owned.as_image();
        assert_eq!(back.format(), img.format());
        assert_eq!(back.as_planar().unwrap().planes()[2].row(1), &[3, 3, 3]);
    }

    #[test]
    fn planes_reject_wrong_chroma_size() {
        let (y, u) = (vec![0u8; 16], vec![0u8; 4]);
        let err = Planes::new(
            PlaneLayout::ycbcr_420(ChannelType::U8),
            vec![gray8(&y, 4, 4), gray8(&u, 2, 2), gray8(&u, 1, 4)],
        )
        .unwrap_err();
        assert_eq!(err.error().kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn significant_bits_bounded_by_container() {
        let y = vec![0u8; 4];
        let p = Planes::new(PlaneLayout::gray(ChannelType::U8), vec![gray8(&y, 2, 2)]).unwrap();
        assert!(p.clone().with_significant_bits(9).is_err());
        assert_eq!(p.with_significant_bits(6).unwrap().significant_bits(), 6);
    }

    #[test]
    fn rgb8_pair_distinguishes_format_from_size() {
        let a = vec![0u8; 2 * 2 * 3];
        let b = vec![0u8; 3 * 2 * 3];
        let ra = Image::from(PixelSlice::new(&a, 2, 2, 6, PixelDescriptor::RGB8_SRGB).unwrap());
        let rb = Image::from(PixelSlice::new(&b, 3, 2, 9, PixelDescriptor::RGB8_SRGB).unwrap());
        assert_eq!(rgb8_pair(&ra, &rb).unwrap_err().error().kind(), ErrorKind::DimensionMismatch);
        let g = vec![0u8; 4];
        let rg = Image::from(gray8(&g, 2, 2));
        assert_eq!(rgb8_pair(&ra, &rg).unwrap_err().error().kind(), ErrorKind::UnsupportedInput);
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
