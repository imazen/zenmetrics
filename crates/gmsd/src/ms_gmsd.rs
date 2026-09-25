//! Paper-derived MS-GMSD/MS-GMSDc. This is not author-software parity.
//! Numerical conventions are frozen in the gmsd-chroma preregistration.
use alloc::vec::Vec;
#[cfg(target_arch = "aarch64")]
use archmage::NeonToken;
#[cfg(target_arch = "wasm32")]
use archmage::Wasm128Token;
#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use archmage::X64V4Token;
use archmage::magetypes;

pub(crate) struct Planes {
    // Y reference/distortion, I reference/distortion, Q reference/distortion.
    data: [Vec<f64>; 6],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pitch: usize,
}

impl Planes {
    fn empty(width: usize, height: usize) -> Self {
        let pitch = super::kernel::padded_pitch(width);
        Self {
            data: core::array::from_fn(|_| alloc::vec![0.0; pitch*(height+2)]),
            width,
            height,
            pitch,
        }
    }

    pub(crate) fn half(&self) -> Self {
        let mut out = Self::empty(self.width.div_ceil(2), self.height.div_ceil(2));
        for (dest, a) in out.data.iter_mut().zip(&self.data) {
            for y in 0..out.height {
                for x in 0..out.width {
                    let x0 = 2 * x + 1;
                    let x1 = (2 * x + 2).min(self.width);
                    let y0 = 2 * y + 1;
                    let y1 = (2 * y + 2).min(self.height);
                    dest[(y + 1) * out.pitch + x + 1] = (((a[y0 * self.pitch + x0]
                        + a[y0 * self.pitch + x1])
                        + a[y1 * self.pitch + x0])
                        + a[y1 * self.pitch + x1])
                        * 0.25;
                }
            }
        }
        out
    }
}

pub(crate) fn prepare(r: &[u8], d: &[u8], w: usize, h: usize, stride: usize) -> Planes {
    let mut p = Planes::empty(w, h);
    for y in 0..h {
        for x in 0..w {
            for (side, rgb) in [r, d].into_iter().enumerate() {
                let i = y * stride + 3 * x;
                let (r, g, b) = (
                    f64::from(rgb[i]),
                    f64::from(rgb[i + 1]),
                    f64::from(rgb[i + 2]),
                );
                let j = (y + 1) * p.pitch + x + 1;
                p.data[side][j] = 0.299 * r + 0.587 * g + 0.114 * b;
                p.data[2 + side][j] = 0.595716 * r - 0.274453 * g - 0.321263 * b;
                p.data[4 + side][j] = 0.211456 * r - 0.522591 * g + 0.311135 * b;
            }
        }
    }
    p
}

#[magetypes(rite, define(f64x8), v4, v3, neon, wasm128, scalar)]
fn similarity_row(token: Token, p: &Planes, y: usize, stabilizer: f64, out: &mut [f64]) {
    let t = f64x8::splat(token, 1.0 / 3.0);
    let two = f64x8::splat(token, 2.0);
    let alpha = f64x8::splat(token, 0.5);
    let c = f64x8::splat(token, stabilizer);
    for x in (0..p.width).step_by(8) {
        macro_rules! sample {
            ($plane:expr,$dy:expr,$dx:expr) => {
                f64x8::from_array(
                    token,
                    core::array::from_fn(|k| p.data[$plane][(y + $dy) * p.pitch + x + k + $dx]),
                )
            };
        }
        let r = chroma_prewitt!(sample, 0, t);
        let d = chroma_prewitt!(sample, 1, t);
        let mask = (alpha * r) * d;
        let q = (((two * r) * d - mask) + c) / (((r * r + d * d) - mask) + c);
        let n = (p.width - x).min(8);
        out[x..x + n].copy_from_slice(&q.to_array()[..n]);
    }
}

macro_rules! band {
    ($name:ident,$token:ident,$row:ident) => {
        #[archmage::arcane]
        pub(crate) fn $name(token: $token, p: &Planes, y0: usize, c: f64, out: &mut [f64]) {
            for (y, row) in out.chunks_mut(p.width).enumerate() {
                $row(token, p, y0 + y, c, row);
            }
        }
    };
}
#[cfg(target_arch = "x86_64")]
band!(ms_band_v3, X64V3Token, similarity_row_v3);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
band!(ms_band_v4, X64V4Token, similarity_row_v4);
#[cfg(target_arch = "aarch64")]
band!(ms_band_neon, NeonToken, similarity_row_neon);
#[cfg(target_arch = "wasm32")]
band!(ms_band_wasm128, Wasm128Token, similarity_row_wasm128);
pub(crate) fn ms_band_scalar(
    token: archmage::ScalarToken,
    p: &Planes,
    y0: usize,
    c: f64,
    out: &mut [f64],
) {
    for (y, row) in out.chunks_mut(p.width).enumerate() {
        similarity_row_scalar(token, p, y0 + y, c, row);
    }
}

pub(crate) fn map(p: &Planes, c: f64) -> Vec<f64> {
    let mut out = alloc::vec![0.0; p.width*p.height];
    let run = |i: usize, out: &mut [f64]| {
        archmage::incant!(
            ms_band(p, i * super::BAND_ROWS, c, out),
            [v4, v3, neon, wasm128, scalar]
        );
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(p.width * super::BAND_ROWS)
            .enumerate()
            .for_each(|(i, q)| run(i, q));
    }
    #[cfg(not(feature = "parallel"))]
    for (i, q) in out.chunks_mut(p.width * super::BAND_ROWS).enumerate() {
        run(i, q);
    }
    out
}

pub(crate) fn variance(q: &[f64]) -> f64 {
    let mean = q.iter().sum::<f64>() / q.len() as f64;
    q.iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / q.len() as f64
}

pub(crate) fn finish(variances: [f64; 4], p: &Planes) -> (f64, f64) {
    let weights = [0.096, 0.596, 0.289, 0.019];
    let ms = super::sqrt_f64(variances.into_iter().zip(weights).map(|(v, w)| v * w).sum());
    let mut error = 0.0;
    for y in 0..p.height {
        for x in 0..p.width {
            let j = (y + 1) * p.pitch + x + 1;
            let di = p.data[2][j] - p.data[3][j];
            let dq = p.data[4][j] - p.data[5][j];
            error += di * di + dq * dq;
        }
    }
    let chroma = super::sqrt_f64(error / (p.width * p.height) as f64);
    let gamma = 2.0 / (1.0 + 0.32 * libm::exp(-15.0 * ms)) - 1.0;
    (ms, gamma * ms + (1.0 - gamma) * (0.01 * chroma))
}

pub(crate) fn run(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    stride: usize,
) -> super::Result<(f64, f64)> {
    super::check_rgb8(r, w, h, stride)?;
    super::check_rgb8(d, w, h, stride)?;
    if w == 0 || h == 0 {
        return Err(super::Error::TooSmall {
            width: w,
            height: h,
        });
    }
    let mut p = prepare(r, d, w, h, stride);
    let mut variances = [0.0; 4];
    for (scale, v) in variances.iter_mut().enumerate() {
        *v = variance(&map(&p, 170.0));
        if scale != 3 {
            p = p.half();
        }
    }
    Ok(finish(variances, &p))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_and_colour_shift() {
        for (w, h) in [(1, 1), (3, 5), (67, 31)] {
            let r = alloc::vec![128;w*h*3];
            assert_eq!(run(&r, &r, w, h, w * 3).unwrap(), (0.0, 0.0));
            let mut d = r.clone();
            for rgb in d.chunks_mut(3) {
                rgb[0] = 200;
                rgb[2] = 20;
            }
            let (_, colour) = run(&r, &d, w, h, w * 3).unwrap();
            assert!(colour > 0.0);
        }
    }
}
