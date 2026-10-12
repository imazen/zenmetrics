#![forbid(unsafe_code)]
//! Opaque sRGB8 rows, a cached reference and raw spatial error maps.
use wasm_bindgen::prelude::*;
use zensim::{DiffmapWeighting, PrecomputedReference, RgbSlice, Zensim, ZensimProfile};

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

fn rgb(rgba: &[u8], width: u32, height: u32, stride: usize) -> Result<Vec<[u8; 3]>, String> {
    let row = (width as usize).checked_mul(4).ok_or("width overflow")?;
    let required = (height as usize)
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(row))
        .ok_or("empty image or dimension overflow")?;
    if width == 0 || stride < row || rgba.len() < required {
        return Err("invalid dimensions, stride or buffer length".into());
    }
    let count = (width as usize)
        .checked_mul(height as usize)
        .ok_or("image size overflow")?;
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(count).map_err(|e| e.to_string())?;
    for y in 0..height as usize {
        for p in rgba[y * stride..y * stride + row].as_chunks::<4>().0 {
            if p[3] != 255 {
                return Err(
                    "opaque sRGB8 required; choose an explicit alpha composite before scoring"
                        .into(),
                );
            }
            pixels.push([p[0], p[1], p[2]]);
        }
    }
    Ok(pixels)
}

/// Reference preparation retained by the coordinator worker across candidates.
#[wasm_bindgen]
pub struct Reference {
    scorer: Zensim,
    reference: PrecomputedReference,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl Reference {
    /// Accept declared, opaque sRGB8 rows with a byte stride.
    #[wasm_bindgen(constructor)]
    pub fn new(rgba: &[u8], width: u32, height: u32, stride: usize) -> Result<Reference, JsError> {
        let pixels = rgb(rgba, width, height, stride).map_err(|e| JsError::new(&e))?;
        let scorer = Zensim::new(ZensimProfile::codec_target());
        let source = RgbSlice::new(&pixels, width as usize, height as usize);
        let reference = scorer
            .precompute_reference(&source)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            scorer,
            reference,
            width,
            height,
        })
    }

    /// Score once, retaining both the scalar and the unnormalized map.
    pub fn compare(&self, rgba: &[u8], stride: usize) -> Result<Comparison, JsError> {
        let pixels = rgb(rgba, self.width, self.height, stride).map_err(|e| JsError::new(&e))?;
        let source = RgbSlice::new(&pixels, self.width as usize, self.height as usize);
        let result = self
            .scorer
            .compute_with_ref_and_diffmap(&self.reference, &source, DiffmapWeighting::default())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Comparison {
            score: result.score(),
            map: result.diffmap().to_vec(),
        })
    }
}

/// Scalar and raw map from the same comparison.
#[wasm_bindgen]
pub struct Comparison {
    score: f64,
    map: Vec<f32>,
}

#[wasm_bindgen]
impl Comparison {
    /// The codec-target zensim score.
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f64 {
        self.score
    }
    /// Raw row-major error values; no visualization normalization.
    #[wasm_bindgen(getter)]
    pub fn map(&self) -> Vec<f32> {
        self.map.clone()
    }
}

/// Actual initialized Rayon pool size, for browser qualification.
#[wasm_bindgen]
pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::rgb;
    #[test]
    fn ingress_preserves_strided_rows_and_refuses_alpha() {
        let packed = [1, 2, 3, 255, 4, 5, 6, 255];
        let padded = [1, 2, 3, 255, 99, 99, 4, 5, 6, 255];
        assert_eq!(
            rgb(&packed, 1, 2, 4).unwrap(),
            rgb(&padded, 1, 2, 6).unwrap()
        );
        assert_eq!(rgb(&packed, 1, 2, 4).unwrap(), vec![[1, 2, 3], [4, 5, 6]]);
        assert!(rgb(&[1, 2, 3, 0], 1, 1, 4).is_err());
        assert!(rgb(&packed, 1, 2, 3).is_err());
        assert!(rgb(&packed, 1, 0, 4).is_err());
        assert!(rgb(&packed, u32::MAX, u32::MAX, usize::MAX).is_err());
    }
}
