//! Input tensors for the learned metrics the browser runs with
//! onnxruntime-web, built with zenmetrics-nn's own conversion and resize
//! code (`preprocess.rs`, `resize.rs`, `tfresize.rs`, included by path at
//! the crate root). Each function restates one model's eval transform from
//! its zenmetrics-nn source, cited per model; the graphs are the models'
//! `*_full` exports, which take these tensors and return the score.

use wasm_bindgen::prelude::*;

use crate::preprocess::{rgb8_to_planar_01, rgb8_to_planar_pm1, rgb8_to_planar_u8};
use crate::resize::{resize_bicubic_cv2_u8, resize_bilinear_pil_u8};
use crate::tfresize::resize_gaussian_tf_f32;
use crate::za;

/// One graph input: name, shape and row-major f32 data.
struct Tensor {
    name: &'static str,
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// The tensors one image contributes to its graph.
#[wasm_bindgen]
pub struct NnPrepared {
    tensors: Vec<Tensor>,
}

#[wasm_bindgen]
impl NnPrepared {
    /// `[{name, shape, offset, length}]` into [`Self::data`], as JSON.
    #[wasm_bindgen(getter)]
    pub fn layout(&self) -> String {
        let mut offset = 0;
        serde_json::json!(
            self.tensors
                .iter()
                .map(|t| {
                    let entry = serde_json::json!({"name":t.name,"shape":t.shape,"offset":offset,"length":t.data.len()});
                    offset += t.data.len();
                    entry
                })
                .collect::<Vec<_>>()
        )
        .to_string()
    }
    /// Every tensor's data, concatenated in [`Self::layout`] order.
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.tensors
            .iter()
            .flat_map(|t| t.data.iter().copied())
            .collect()
    }
}

/// zenmetrics-nn `lpips.rs` `check_size` and the size checks in `dists.rs`
/// and `deepdc.rs` (`check_size`): both sides at least the model's
/// `min_size` from its `.meta`.
fn check_min(img: &za::Rgb8In<'_>, min: usize) -> Result<(), String> {
    if img.width < min || img.height < min {
        return Err(format!("image is too small: needs at least {min}×{min}"));
    }
    Ok(())
}

/// MANIQA (`maniqa.rs` `prep`): OpenCV INTER_CUBIC on u8 to 224×224, then
/// `(v/255 − 0.5)/0.5`.
fn maniqa(img: &za::Rgb8In<'_>) -> Vec<Tensor> {
    const SIDE: usize = 224;
    let full = rgb8_to_planar_u8(img);
    let r = resize_bicubic_cv2_u8(&full, 3, img.height, img.width, SIDE, SIDE);
    let data = r
        .iter()
        .map(|&v| (f32::from(v) / 255.0 - 0.5) / 0.5)
        .collect();
    vec![Tensor {
        name: "x",
        shape: vec![1, 3, SIDE, SIDE],
        data,
    }]
}

/// ARNIQA (`arniqa.rs` `crop01` and `prep`): five zero-padded 224 crops
/// (centre, TL, BL, TR, BR) of the image and of its PIL-bilinear half-size
/// copy, `v/255`.
fn arniqa(img: &za::Rgb8In<'_>) -> Result<Vec<Tensor>, String> {
    const CROP: usize = 224;
    let (h, w) = (img.height, img.width);
    if h < 2 || w < 2 {
        return Err("image is too small: needs at least 2×2".into());
    }
    let crop01 = |img: &[u8], h: usize, w: usize, top: i64, left: i64| {
        let mut out = vec![0.0f32; 3 * CROP * CROP];
        for c in 0..3 {
            let src = &img[c * h * w..(c + 1) * h * w];
            let dst = &mut out[c * CROP * CROP..(c + 1) * CROP * CROP];
            for y in 0..CROP as i64 {
                let sy = top + y;
                if sy < 0 || sy >= h as i64 {
                    continue;
                }
                for x in 0..CROP as i64 {
                    let sx = left + x;
                    if sx >= 0 && sx < w as i64 {
                        dst[y as usize * CROP + x as usize] =
                            src[sy as usize * w + sx as usize] as f32 / 255.0;
                    }
                }
            }
        }
        out
    };
    let full = rgb8_to_planar_u8(img);
    let half = resize_bilinear_pil_u8(&full, 3, h, w, h / 2, w / 2);
    let crops = |img: &[u8], h: usize, w: usize| -> Vec<f32> {
        let (cx, cy) = (w / 2, h / 2);
        let (k, h64, w64) = (CROP as i64, h as i64, w as i64);
        let origins = [
            (cy as i64 - k / 2, cx as i64 - k / 2),
            (0, 0),
            (h64 - k, 0),
            (0, w64 - k),
            (h64 - k, w64 - k),
        ];
        let mut out = Vec::with_capacity(5 * 3 * CROP * CROP);
        for &(top, left) in &origins {
            out.extend_from_slice(&crop01(img, h, w, top, left));
        }
        out
    };
    Ok(vec![
        Tensor {
            name: "x",
            shape: vec![5, 3, CROP, CROP],
            data: crops(&full, h, w),
        },
        Tensor {
            name: "x_ds",
            shape: vec![5, 3, CROP, CROP],
            data: crops(&half, h / 2, w / 2),
        },
    ])
}

/// MUSIQ (`musiq.rs` constants, `extract` and `prep`): `[-1, 1]`
/// normalisation, TF Gaussian resizes to longer side 224 and 384, and the
/// three 32×32 patch sequences with hashed positions, scale ids and masks.
fn musiq(img: &za::Rgb8In<'_>) -> Result<Vec<Tensor>, String> {
    const PATCH: usize = 32;
    const GRID: usize = 10;
    const LONGER: [usize; 2] = [224, 384];
    const ROW: usize = PATCH * PATCH * 3 + 3;
    const MAX_ROWS: usize = 8192;
    fn extract(img: &[f32], h: usize, w: usize, cap: i64, scale_id: f32, out: &mut Vec<f32>) {
        let (ch, cw) = (h.div_ceil(PATCH), w.div_ceil(PATCH));
        let (pt, pl) = ((ch * PATCH - h) / 2, (cw * PATCH - w) / 2);
        let pos = |count: usize| -> Vec<i64> {
            let sc = GRID as f32 / count as f32;
            (0..count)
                .map(|i| ((i as f32 * sc).floor() as i64).min(GRID as i64 - 1))
                .collect()
        };
        let (ph, pw) = (pos(ch), pos(cw));
        for dy in 0..ch {
            for dx in 0..cw {
                for py in 0..PATCH {
                    let sy = dy * PATCH + py;
                    for px in 0..PATCH {
                        let sx = dx * PATCH + px;
                        for c in 0..3 {
                            let v = if sy >= pt && sy - pt < h && sx >= pl && sx - pl < w {
                                img[c * h * w + (sy - pt) * w + sx - pl]
                            } else {
                                0.0
                            };
                            out.push(v);
                        }
                    }
                }
                out.push((ph[dy] * GRID as i64 + pw[dx]) as f32);
                out.push(scale_id);
                out.push(1.0);
            }
        }
        if cap >= 0 {
            out.resize(
                out.len() + (cap as usize).saturating_sub(ch * cw) * ROW,
                0.0,
            );
        }
    }
    let (h, w) = (img.height, img.width);
    if h == 0 || w == 0 {
        return Err("image is too small: needs at least 1×1".into());
    }
    if 193 + h.div_ceil(PATCH) * w.div_ceil(PATCH) > MAX_ROWS {
        return Err(format!(
            "image has too many patches for MUSIQ (more than {MAX_ROWS} rows)"
        ));
    }
    let full = rgb8_to_planar_u8(img);
    let norm: Vec<f32> = full
        .iter()
        .map(|&v| -1.0f32 + (v as f32 / 255.0) * 2.0)
        .collect();
    let mut out = Vec::new();
    for (scale_id, longer) in LONGER.iter().enumerate() {
        let ratio = *longer as f32 / h.max(w) as f32;
        let rh = (h as f32 * ratio).round_ties_even() as usize;
        let rw = (w as f32 * ratio).round_ties_even() as usize;
        if rh == 0 || rw == 0 {
            return Err("degenerate aspect ratio for MUSIQ".into());
        }
        let r = resize_gaussian_tf_f32(&norm, 3, h, w, rh, rw);
        let cap = (longer.div_ceil(PATCH) * longer.div_ceil(PATCH)) as i64;
        extract(&r, rh, rw, cap, scale_id as f32, &mut out);
    }
    extract(&norm, h, w, -1, LONGER.len() as f32, &mut out);
    let rows = out.len() / ROW;
    Ok(vec![Tensor {
        name: "x",
        shape: vec![1, rows, ROW],
        data: out,
    }])
}

/// WaDIQaM (`wadiqam.rs` `grid` and `extract_patches`): the
/// non-overlapping 32×32 stride-32 grid in row-major patch order,
/// `[P, 3, 32, 32]` f32 from raw u8, at most 32768 patches.
fn wadiqam(img: &za::Rgb8In<'_>, min_size: usize) -> Result<Vec<Tensor>, String> {
    const PATCH: usize = 32;
    const MAX_PATCHES: usize = 32768;
    let grid = |dim: usize| {
        if dim < PATCH {
            0
        } else {
            (dim - PATCH) / PATCH + 1
        }
    };
    let (h, w) = (img.height, img.width);
    let (gh, gw) = (grid(h), grid(w));
    let n = gh * gw;
    if n == 0 {
        return Err(format!(
            "image is too small: needs at least {min_size}×{min_size}"
        ));
    }
    if n > MAX_PATCHES {
        return Err(format!("image has more than {MAX_PATCHES} WaDIQaM patches"));
    }
    let mut out = vec![0.0f32; n * 3 * PATCH * PATCH];
    let stride = img.stride;
    for gy in 0..gh {
        for gx in 0..gw {
            let d = &mut out[(gy * gw + gx) * 3 * PATCH * PATCH..][..3 * PATCH * PATCH];
            for py in 0..PATCH {
                let row = &img.pixels[(gy * PATCH + py) * stride + (gx * PATCH) * 3..];
                for px in 0..PATCH {
                    let base = py * PATCH + px;
                    d[base] = row[px * 3] as f32;
                    d[PATCH * PATCH + base] = row[px * 3 + 1] as f32;
                    d[2 * PATCH * PATCH + base] = row[px * 3 + 2] as f32;
                }
            }
        }
    }
    Ok(vec![Tensor {
        name: "x",
        shape: vec![n, 3, PATCH, PATCH],
        data: out,
    }])
}

/// The input tensors of one image for `model` (`lpips`, `planar01`,
/// `maniqa`, `arniqa`, `musiq`, `wadiqam`), from opaque RGBA rows with a byte stride.
/// `min_size` is the model's `.meta` value; models without one pass 1.
pub fn prepare(
    model: &str,
    rgba: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    min_size: usize,
) -> Result<Vec<f32>, String> {
    prepare_tensors(model, rgba, width, height, stride, min_size)
        .map(|t| t.into_iter().flat_map(|t| t.data).collect())
}

/// CKDN's asymmetric eval transform (`ckdn.rs::prep`): resize short side
/// to 329, then centre-crop 288; reference nearest, distorted bicubic AA.
fn ckdn(img: &za::Rgb8In<'_>, nearest: bool) -> Result<Vec<Tensor>, String> {
    use crate::preprocess::rgb8_to_planar_255;
    use crate::resize::{center_crop, resize_bicubic_aa, resize_nearest};
    let (h, w) = (img.height, img.width);
    let (nh, nw) = if w <= h {
        ((329 * h as u64 / w as u64) as usize, 329)
    } else {
        (329, (329 * w as u64 / h as u64) as usize)
    };
    if nh.max(nw) > 32_768 {
        return Err("image is too large for CKDN: resized long side exceeds 32768".into());
    }
    let planar = rgb8_to_planar_255(img);
    let resized = if nearest {
        resize_nearest(&planar, 3, h, w, nh, nw)
    } else {
        resize_bicubic_aa(&planar, 3, h, w, nh, nw)
    };
    Ok(vec![Tensor {
        name: "x",
        shape: vec![1, 3, 288, 288],
        data: center_crop(&resized, 3, nh, nw, 288),
    }])
}

/// HyperIQA's documented deterministic 25-crop protocol (`hyperiqa.rs::prep`).
/// The graph has fixed batch 25; this does not emulate random crop sampling.
fn hyperiqa(img: &za::Rgb8In<'_>) -> Vec<Tensor> {
    let planar = rgb8_to_planar_u8(img);
    let r = resize_bilinear_pil_u8(&planar, 3, img.height, img.width, 512, 384);
    let mut data = Vec::with_capacity(25 * 3 * 224 * 224);
    for y in [0, 72, 144, 216, 288] {
        for x in [0, 40, 80, 120, 160] {
            for c in 0..3 {
                for yy in 0..224 {
                    for &v in &r[c * 512 * 384 + (y + yy) * 384 + x..][..224] {
                        data.push(
                            (v as f32 / 255.0 - [0.485, 0.456, 0.406][c])
                                / [0.229, 0.224, 0.225][c],
                        );
                    }
                }
            }
        }
    }
    vec![Tensor {
        name: "x",
        shape: vec![25, 3, 224, 224],
        data,
    }]
}

fn prepare_tensors(
    model: &str,
    rgba: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    min_size: usize,
) -> Result<Vec<Tensor>, String> {
    let pixels = crate::rgb(rgba, width, height, stride)?;
    let img = za::Rgb8In {
        pixels: pixels.as_flattened(),
        width: width as usize,
        height: height as usize,
        stride: width as usize * 3,
    };
    check_min(&img, min_size)?;
    let planar = |data| {
        vec![Tensor {
            name: "x",
            shape: vec![1, 3, img.height, img.width],
            data,
        }]
    };
    match model {
        // lpips.rs `score_pair`: `rgb8_to_planar_pm1` (LPIPS and ST-LPIPS).
        "lpips" => Ok(planar(rgb8_to_planar_pm1(&img))),
        // dists.rs and deepdc.rs `score_pair`: `rgb8_to_planar_01`.
        "planar01" => Ok(planar(rgb8_to_planar_01(&img))),
        // wd.rs `check_size` and `features`: 8..=4096 per side, at most
        // 1,048,576 pixels on 32-bit targets, then `rgb8_to_planar_01`.
        "wd" => {
            let max_pixels = if usize::BITS == 32 {
                1024 * 1024
            } else {
                9_000_000
            };
            if img.width > 4096 || img.height > 4096 || img.width * img.height > max_pixels {
                return Err(format!(
                    "image is too large for WD: at most 4096 per side and {max_pixels} pixels"
                ));
            }
            Ok(planar(rgb8_to_planar_01(&img)))
        }
        "ckdn-reference" => ckdn(&img, true),
        "ckdn-distorted" => ckdn(&img, false),
        "hyperiqa" => Ok(hyperiqa(&img)),
        "maniqa" => Ok(maniqa(&img)),
        "arniqa" => arniqa(&img),
        "musiq" => musiq(&img),
        "wadiqam" => wadiqam(&img, min_size),
        other => Err(format!("no browser preprocessing for model {other}")),
    }
}

/// [`prepare`] for the worker: the tensors and their layout.
#[wasm_bindgen]
pub fn nn_prepare(
    model: &str,
    rgba: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    min_size: usize,
) -> Result<NnPrepared, JsError> {
    prepare_tensors(model, rgba, width, height, stride, min_size)
        .map(|tensors| NnPrepared { tensors })
        .map_err(|e| JsError::new(&e))
}

/// Wasserstein Distortion's statistics head for one VGG16 tap, restated
/// from zenmetrics-nn `wd.rs` (`lowpass`, `decimate`, `weights`, `channel`
/// and the per-tap part of `head`): the tap's five per-level values, which
/// the caller adds in f32 in tap order (`head`'s `total += layer`). One
/// thread: `head` sums its per-channel results in channel order whatever
/// its split, so the value does not depend on it.
mod wd {
    const VAR_MIN: f32 = 5e-7;

    fn lowpass(x: &[f32], h: usize, w: usize, stride: usize) -> Vec<f32> {
        let (oh, ow) = (h.div_ceil(stride), w.div_ceil(stride));
        let mut out = vec![0.0; oh * ow];
        let kernel = [0.25f32, 0.5, 0.25];
        for y in 0..oh {
            for xout in 0..ow {
                let mut sum = 0.0;
                for (ky, &wy) in kernel.iter().enumerate() {
                    let sy = (y * stride + ky).wrapping_sub(1);
                    if sy >= h {
                        continue;
                    }
                    for (kx, &wx) in kernel.iter().enumerate() {
                        let sx = (xout * stride + kx).wrapping_sub(1);
                        if sx < w {
                            sum += x[sy * w + sx] * (wy * wx);
                        }
                    }
                }
                out[y * ow + xout] = sum;
            }
        }
        out
    }

    fn decimate(x: &[f32], h: usize, w: usize) -> Vec<f32> {
        (0..h)
            .step_by(2)
            .flat_map(|y| (0..w).step_by(2).map(move |xpos| x[y * w + xpos]))
            .collect()
    }

    fn weights(h: usize, w: usize) -> Vec<[Vec<f32>; 5]> {
        let mut sigma: [Vec<f32>; 5] = core::array::from_fn(|k| vec![(k + 1) as f32; h * w]);
        let (mut h, mut w) = (h, w);
        let mut result = Vec::with_capacity(5);
        for level in 1..=5 {
            result.push(core::array::from_fn(|k| {
                sigma[k]
                    .iter()
                    .map(|&s| (1.0 - (s - level as f32).abs()).max(0.0))
                    .collect()
            }));
            sigma = sigma.map(|s| lowpass(&s, h, w, 2));
            h = h.div_ceil(2);
            w = w.div_ceil(2);
        }
        result
    }

    fn channel(
        a: &[f32],
        b: &[f32],
        h: usize,
        w: usize,
        weights: &[[Vec<f32>; 5]],
    ) -> [[f64; 5]; 5] {
        let (mut a, mut b) = (a.to_vec(), b.to_vec());
        let mut aa: Vec<_> = a.iter().map(|x| x * x).collect();
        let mut bb: Vec<_> = b.iter().map(|x| x * x).collect();
        let (mut h, mut w) = (h, w);
        let mut result = [[0.0; 5]; 5];
        for (level, weights) in weights.iter().enumerate() {
            let ma = lowpass(&a, h, w, 1);
            let mb = lowpass(&b, h, w, 1);
            let pa = lowpass(&aa, h, w, 1);
            let pb = lowpass(&bb, h, w, 1);
            for i in 0..h * w {
                let sa = (pa[i] - ma[i] * ma[i]).max(VAR_MIN).sqrt();
                let sb = (pb[i] - mb[i] * mb[i]).max(VAR_MIN).sqrt();
                let dm = ma[i] - mb[i];
                let ds = sa - sb;
                let distance = dm * dm + ds * ds;
                for (k, weight) in weights.iter().enumerate() {
                    result[level][k] += f64::from(weight[i] * distance);
                }
            }
            for value in &mut result[level] {
                *value /= (h * w) as f64;
            }
            a = decimate(&ma, h, w);
            b = decimate(&mb, h, w);
            aa = decimate(&pa, h, w);
            bb = decimate(&pb, h, w);
            h = h.div_ceil(2);
            w = w.div_ceil(2);
        }
        result
    }

    /// `head`'s `layer` for one tap of `c` channels of `h × w`.
    pub(super) fn layer(a: &[f32], b: &[f32], c: usize, h: usize, w: usize) -> [f32; 5] {
        let weights = weights(h, w);
        let plane = h * w;
        let channels: Vec<_> = (0..c)
            .map(|ch| {
                channel(
                    &a[ch * plane..(ch + 1) * plane],
                    &b[ch * plane..(ch + 1) * plane],
                    h,
                    w,
                    &weights,
                )
            })
            .collect();
        let mut layer = [0.0f32; 5];
        for level in 0..5 {
            for (k, value) in layer.iter_mut().enumerate() {
                *value += (channels.iter().map(|ch| ch[level][k]).sum::<f64>() / c as f64) as f32;
            }
        }
        layer
    }
}

/// One VGG16 tap's Wasserstein Distortion contribution (see [`wd`]):
/// reference and distorted features `[c, h, w]`, `c` the tap's channel
/// count in `wd.rs`'s order 64, 128, 256, 512.
pub fn wd_layer(a: &[f32], b: &[f32], c: usize, h: usize, w: usize) -> Result<Vec<f32>, String> {
    if ![64, 128, 256, 512].contains(&c)
        || h == 0
        || w == 0
        || a.len() != c * h * w
        || b.len() != a.len()
    {
        return Err("WD feature shape mismatch".into());
    }
    Ok(wd::layer(a, b, c, h, w).to_vec())
}

/// [`wd_layer`] for the worker.
#[wasm_bindgen]
pub fn nn_wd_layer(
    a: &[f32],
    b: &[f32],
    c: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>, JsError> {
    wd_layer(a, b, c, h, w).map_err(|e| JsError::new(&e))
}
