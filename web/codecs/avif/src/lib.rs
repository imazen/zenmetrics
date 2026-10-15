//! AVIF module of the zenmetrics browser viewer: zenavif encode (zenravif,
//! single-threaded), then decode the file back for scoring. Fetched by the
//! worker only when AVIF is chosen.
#![forbid(unsafe_code)]

use wasm_bindgen::prelude::*;

/// An encode-decode round trip: the file and its decoded pixels.
#[wasm_bindgen]
pub struct Coded {
    file: Vec<u8>,
    rgba: Vec<u8>,
}

#[wasm_bindgen]
impl Coded {
    /// The encoded AVIF file.
    #[wasm_bindgen(getter)]
    pub fn file(&self) -> Vec<u8> {
        self.file.clone()
    }
    /// The decoded image, RGBA8 with alpha 255, row-major.
    #[wasm_bindgen(getter)]
    pub fn rgba(&self) -> Vec<u8> {
        self.rgba.clone()
    }
}

/// Encode opaque RGBA8 rows (alpha dropped) as AVIF at `quality` (1–100,
/// zenavif's scale) and `speed` (1 slowest … 10 fastest), then decode the
/// file with zenavif.
#[wasm_bindgen]
pub fn roundtrip(
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    speed: u8,
) -> Result<Coded, JsError> {
    console_error_panic_hook::set_once();
    let (w, h) = (width as usize, height as usize);
    if rgba.len() != w * h * 4 {
        return Err(JsError::new("pixel buffer does not match the image size"));
    }
    let rgb: Vec<rgb::Rgb<u8>> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| rgb::Rgb {
            r: p[0],
            g: p[1],
            b: p[2],
        })
        .collect();
    let config = zenavif::EncoderConfig::new()
        .quality(quality)
        .speed(speed)
        .threads(Some(1));
    let file = zenavif::encode_rgb8(
        imgref::ImgRef::new(&rgb, w, h),
        &config,
        almost_enough::StopToken::new(enough::Unstoppable),
    )
    .map_err(|e| JsError::new(&format!("{e:?}")))?
    .avif_file;
    let decoded = zenavif::decode(&file).map_err(|e| JsError::new(&format!("{e:?}")))?;
    let pixels = decoded.as_slice();
    if (pixels.width(), pixels.rows()) != (width, height) {
        return Err(JsError::new("decoded size differs from the source"));
    }
    let mut out = vec![0; w * h * 4];
    zenpixels_convert::converter::RowConverter::new(
        pixels.descriptor(),
        zenpixels::PixelDescriptor::RGBA8_SRGB,
    )
    .map_err(|e| JsError::new(&e.to_string()))?
    .convert_rows(
        pixels.as_strided_bytes(),
        pixels.stride(),
        &mut out,
        w * 4,
        width,
        height,
    )
    .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(Coded { file, rgba: out })
}
