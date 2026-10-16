//! WebP module of the zenmetrics browser viewer: zenwebp encode, then decode
//! the file back for scoring. Fetched by the worker only when WebP is chosen.
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
    /// The encoded WebP file.
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

/// Encode opaque RGBA8 rows (alpha dropped) as lossy WebP at `quality`
/// (0–100, zenwebp's scale) and decode the file with zenwebp.
#[wasm_bindgen]
pub fn roundtrip(rgba: &[u8], width: u32, height: u32, quality: f32) -> Result<Coded, JsError> {
    console_error_panic_hook::set_once();
    if rgba.len() != width as usize * height as usize * 4 {
        return Err(JsError::new("pixel buffer does not match the image size"));
    }
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    let config = zenwebp::EncoderConfig::new_lossy().with_quality(quality);
    let file =
        zenwebp::EncodeRequest::new(&config, &rgb, zenwebp::PixelLayout::Rgb8, width, height)
            .encode()
            .map_err(|e| JsError::new(&format!("{e:?}")))?;
    let (rgba, w, h) =
        zenwebp::decoder::decode_rgba(&file).map_err(|e| JsError::new(&format!("{e:?}")))?;
    if (w, h) != (width, height) {
        return Err(JsError::new("decoded size differs from the source"));
    }
    Ok(Coded { file, rgba })
}
