//! JPEG XL module of the zenmetrics browser viewer. Fetched by the worker
//! only when JPEG XL or the encoder loop player is used.
//!
//! [`roundtrip`] encodes and decodes a candidate. [`loop_encode`] runs
//! jxl-encoder's perceptual quantization loop, scored either by its own
//! butteraugli or by a JavaScript callback that returns any viewer metric's
//! map on butteraugli's scale, and returns the loop's trace in the `.ltrace`
//! format of jxl-encoder's `loop_trace_dump` example.
#![forbid(unsafe_code)]

use jxl_encoder::{
    ExternalLoopMetric, ExternalMetric, LoopFrame, LoopTrace, LossyConfig, PixelLayout,
};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

/// An encode-decode round trip: the file and its decoded pixels.
#[wasm_bindgen]
pub struct Coded {
    file: Vec<u8>,
    rgba: Vec<u8>,
}

#[wasm_bindgen]
impl Coded {
    /// The JPEG XL codestream.
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

/// One traced loop encode.
#[wasm_bindgen]
pub struct LoopRun {
    coded: Coded,
    trace: Vec<u8>,
}

#[wasm_bindgen]
impl LoopRun {
    /// The JPEG XL codestream.
    #[wasm_bindgen(getter)]
    pub fn file(&self) -> Vec<u8> {
        self.coded.file.clone()
    }
    /// The decoded image, RGBA8 with alpha 255, row-major.
    #[wasm_bindgen(getter)]
    pub fn rgba(&self) -> Vec<u8> {
        self.coded.rgba.clone()
    }
    /// The loop trace in `.ltrace` format.
    #[wasm_bindgen(getter)]
    pub fn trace(&self) -> Vec<u8> {
        self.trace.clone()
    }
}

fn rgb(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    if rgba.len() != width as usize * height as usize * 4 {
        return Err("pixel buffer does not match the image size".into());
    }
    Ok(rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect())
}

/// Decode a codestream with zenjxl-decoder to RGBA8 (alpha 255).
pub fn decode(file: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let img = zenjxl_decoder::decode(file).map_err(|e| e.to_string())?;
    if img.width != width as usize || img.height != height as usize {
        return Err("decoded size differs from the source".into());
    }
    match img.channels {
        4 => Ok(img
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect()),
        3 => Ok(img
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect()),
        2 => Ok(img
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[0], p[0], 255])
            .collect()),
        1 => Ok(img.data.iter().flat_map(|&v| [v, v, v, 255]).collect()),
        c => Err(format!("unexpected channel count {c}")),
    }
}

/// Encode opaque RGBA8 rows (alpha dropped) with jxl-encoder at `quality`
/// on the libjpeg scale (mapped to a distance by `calibrated_jxl_quality`
/// and `quality_to_distance`) and `effort`, then decode the codestream.
#[wasm_bindgen]
pub fn roundtrip(
    rgba: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    effort: u8,
) -> Result<Coded, JsError> {
    console_error_panic_hook::set_once();
    let rgb = rgb(rgba, width, height).map_err(|e| JsError::new(&e))?;
    let distance = jxl_encoder::quality_to_distance(jxl_encoder::calibrated_jxl_quality(quality));
    let file = LossyConfig::new(distance)
        .with_effort(effort)
        .encode(&rgb, width, height, PixelLayout::Rgb8)
        .map_err(|e| JsError::new(&e.to_string()))?;
    let rgba = decode(&file, width, height).map_err(|e| JsError::new(&e))?;
    Ok(Coded { file, rgba })
}

/// The distance [`roundtrip`] encodes `quality` at.
#[wasm_bindgen]
pub fn quality_distance(quality: f32) -> f32 {
    jxl_encoder::quality_to_distance(jxl_encoder::calibrated_jxl_quality(quality))
}

thread_local! {
    /// The JavaScript scorer of the running [`loop_encode`].
    static SCORER: RefCell<Option<js_sys::Function>> = const { RefCell::new(None) };
}

/// sRGB8 of a linear value in `[0, 1]`, as the trace stores reconstructions.
fn srgb8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5) as u8
}

/// The loop's metric: each reconstruction goes to the JavaScript scorer as
/// opaque sRGB8 RGBA, which returns `{values: Float32Array, score: number}`
/// with `width × height` values on butteraugli's scale.
struct JsScorer;

impl ExternalLoopMetric for JsScorer {
    fn set_reference(
        &mut self,
        _: &[f32],
        _: &[f32],
        _: &[f32],
        _: usize,
        _: usize,
    ) -> Result<(), String> {
        // The scorer already holds the source the encode was given.
        Ok(())
    }

    fn compare(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
        width: usize,
        height: usize,
        diffmap: &mut Vec<f32>,
    ) -> Result<f64, String> {
        let mut rgba = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in y * stride..y * stride + width {
                rgba.extend([srgb8(r[x]), srgb8(g[x]), srgb8(b[x]), 255]);
            }
        }
        SCORER.with(|slot| {
            let slot = slot.borrow();
            let scorer = slot.as_ref().ok_or("no scorer installed")?;
            let out = scorer
                .call1(&JsValue::NULL, &js_sys::Uint8Array::from(&rgba[..]))
                .map_err(|e| format!("scorer failed: {e:?}"))?;
            let values: js_sys::Float32Array = js_sys::Reflect::get(&out, &"values".into())
                .map_err(|e| format!("{e:?}"))?
                .dyn_into()
                .map_err(|_| "scorer values must be a Float32Array")?;
            if values.length() as usize != width * height {
                return Err(format!(
                    "scorer returned {} values for {width}×{height}",
                    values.length()
                ));
            }
            diffmap.resize(width * height, 0.0);
            values.copy_to(diffmap);
            js_sys::Reflect::get(&out, &"score".into())
                .map_err(|e| format!("{e:?}"))?
                .as_f64()
                .ok_or_else(|| "scorer score must be a number".into())
        })
    }
}

/// Settings recorded in a trace's header.
pub struct TraceHeader<'a> {
    /// Target distance.
    pub distance: f32,
    /// Encoder effort.
    pub effort: u8,
    /// Loop iterations per seed.
    pub iters: u32,
    /// Encoded size in bytes.
    pub bytes: usize,
    /// What scored the loop, for display.
    pub steering: &'a str,
}

/// Pack a loop's frames as `.ltrace` (see jxl-encoder's
/// `examples/loop_trace_dump.rs`): magic, `u32` JSON header length, header,
/// source sRGB8 RGB, then per frame the quant field, the next quant field
/// when there is one, the tile distances and the diffmap as `f32`, and the
/// reconstruction as sRGB8 RGB.
pub fn pack_trace(
    frames: &[LoopFrame],
    source_rgb: &[u8],
    width: u32,
    height: u32,
    head: &TraceHeader<'_>,
) -> Result<Vec<u8>, String> {
    let first = frames.first().ok_or("the loop recorded no iterations")?;
    let header = serde_json::json!({
        "width": width,
        "height": height,
        "distance": head.distance,
        "effort": head.effort,
        "iters": head.iters,
        "bytes": head.bytes,
        "steering": head.steering,
        "xsize_blocks": first.xsize_blocks,
        "ysize_blocks": first.ysize_blocks,
        "frames": frames.iter().map(|f| serde_json::json!({
            "seed": f.seed,
            "iter": f.iter,
            "metric": f.metric,
            "score": f.score,
            "target": f.target,
            "has_next": f.next_quant_field.is_some(),
        })).collect::<Vec<_>>(),
    })
    .to_string();
    let mut out = Vec::new();
    out.extend_from_slice(b"JXLTRACE1\n");
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(source_rgb);
    let f32s = |out: &mut Vec<u8>, v: &[f32]| out.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    for f in frames {
        f32s(&mut out, &f.quant_field);
        if let Some(next) = &f.next_quant_field {
            f32s(&mut out, next);
        }
        f32s(&mut out, &f.tile_dist);
        f32s(&mut out, &f.diffmap);
        let [r, g, b] = &f.recon;
        out.extend((0..r.len()).flat_map(|i| [srgb8(r[i]), srgb8(g[i]), srgb8(b[i])]));
    }
    Ok(out)
}

/// Encode with the perceptual loop traced: `iters` iterations per seed at
/// `distance` and `effort`. With `scorer` (a function from a reconstruction's
/// sRGB8 RGBA to `{values, score}`), the loop is scored by it instead of
/// butteraugli; `steering` names it in the trace.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn loop_encode(
    rgba: &[u8],
    width: u32,
    height: u32,
    distance: f32,
    effort: u8,
    iters: u32,
    scorer: Option<js_sys::Function>,
    steering: &str,
) -> Result<LoopRun, JsError> {
    console_error_panic_hook::set_once();
    let rgb = rgb(rgba, width, height).map_err(|e| JsError::new(&e))?;
    let external = scorer.is_some();
    SCORER.with(|slot| *slot.borrow_mut() = scorer);
    let trace = LoopTrace::new();
    let metric = ExternalMetric::new(JsScorer);
    let config = LossyConfig::new(distance)
        .with_effort(effort)
        .with_butteraugli_iters(iters);
    let mut request = config
        .encode_request(width, height, PixelLayout::Rgb8)
        .with_loop_trace(&trace);
    if external {
        request = request.with_external_metric(&metric);
    }
    let encoded = request.encode(&rgb);
    SCORER.with(|slot| *slot.borrow_mut() = None);
    let file = encoded.map_err(|e| JsError::new(&e.to_string()))?;
    let decoded = decode(&file, width, height).map_err(|e| JsError::new(&e))?;
    let head = TraceHeader {
        distance,
        effort,
        iters,
        bytes: file.len(),
        steering,
    };
    let trace =
        pack_trace(&trace.take(), &rgb, width, height, &head).map_err(|e| JsError::new(&e))?;
    Ok(LoopRun {
        coded: Coded {
            file,
            rgba: decoded,
        },
        trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed trace has exactly the layout the viewer's parser reads.
    #[test]
    fn packed_trace_layout() {
        let (w, h) = (40u32, 24u32);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let trace = LoopTrace::new();
        let file = LossyConfig::new(1.0)
            .with_effort(7)
            .with_butteraugli_iters(2)
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_loop_trace(&trace)
            .encode(&rgb)
            .unwrap();
        let frames = trace.take();
        let head = TraceHeader {
            distance: 1.0,
            effort: 7,
            iters: 2,
            bytes: file.len(),
            steering: "butteraugli",
        };
        let packed = pack_trace(&frames, &rgb, w, h, &head).unwrap();
        assert_eq!(&packed[..10], b"JXLTRACE1\n");
        let n = u32::from_le_bytes(packed[10..14].try_into().unwrap()) as usize;
        let json: serde_json::Value = serde_json::from_slice(&packed[14..14 + n]).unwrap();
        assert_eq!(json["frames"].as_array().unwrap().len(), frames.len());
        let blocks = (w.div_ceil(8) * h.div_ceil(8)) as usize;
        let pixels = (w * h) as usize;
        let expected = 14
            + n
            + pixels * 3
            + frames
                .iter()
                .map(|f| {
                    (blocks * (2 + usize::from(f.next_quant_field.is_some())) + pixels) * 4
                        + pixels * 3
                })
                .sum::<usize>();
        assert_eq!(packed.len(), expected);
        assert_eq!(decode(&file, w, h).unwrap().len(), pixels * 4);
    }
}
