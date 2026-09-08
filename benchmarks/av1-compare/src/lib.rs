//! In-process comparison arms for the zenfleet executor. The initial protocol
//! deliberately specifies 8-bit, limited-range I420 and one still per call.
use serde::{Deserialize, Serialize};
use std::{ffi::c_int, ptr};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum Backend {
    #[serde(rename = "libaom")]
    Libaom,
    #[serde(rename = "c-svt-av1")]
    CSvt,
    #[serde(rename = "zenav1-svt")]
    Svt,
    #[serde(rename = "zenav1-aom")]
    Aom,
    #[serde(rename = "zenrav1e")]
    Rav1e,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub backend: Backend,
    pub width: u32,
    pub height: u32,
    /// Native quantizer: 0..63 for AOM/SVT, 0..255 for zenrav1e.
    pub quantizer: u32,
    pub speed: u32,
    /// Current common comparison arm requires 1. C SVT interprets this as lp.
    pub threads: u32,
}
impl Config {
    pub fn validate(&self, pixels: &[u8]) -> Result<(), String> {
        if !(64..=16384).contains(&self.width)
            || !(64..=16384).contains(&self.height)
            || !self.width.is_multiple_of(2)
            || !self.height.is_multiple_of(2)
        {
            return Err("protocol v1 requires even dimensions in 64..=16384".into());
        }
        let size = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_add(n / 2))
            .ok_or("image length overflow")?;
        if pixels.len() != size {
            return Err(format!(
                "expected {size} packed I420 bytes, got {}",
                pixels.len()
            ));
        }
        if self.threads != 1 {
            return Err("protocol v1 has only the single-thread/lp1 arm".into());
        }
        let (qmax, smin, smax) = match self.backend {
            Backend::Libaom | Backend::Aom => (63, 0, 9),
            Backend::CSvt => (63, 0, 13),
            Backend::Svt => (63, 1, 10),
            Backend::Rav1e => (255, 0, 10),
        };
        if self.quantizer > qmax || !(smin..=smax).contains(&self.speed) {
            return Err("quantizer or speed outside this backend's native range".into());
        }
        Ok(())
    }
    pub fn revision(&self) -> &'static str {
        match self.backend {
            Backend::Libaom => env!("LIBAOM_REV"),
            Backend::CSvt => env!("C_SVT_REV"),
            Backend::Svt => env!("ZENAV1_SVT_REV"),
            Backend::Aom => env!("ZENAV1_AOM_REV"),
            Backend::Rav1e => env!("ZENRAV1E_REV"),
        }
    }
}
#[repr(C)]
struct COutput {
    data: *mut u8,
    len: usize,
}
unsafe extern "C" {
    fn zm_libaom(
        p: *const u8,
        w: u32,
        h: u32,
        q: u32,
        speed: u32,
        threads: u32,
        out: *mut COutput,
    ) -> c_int;
    fn zm_c_svt(
        p: *const u8,
        w: u32,
        h: u32,
        q: u32,
        speed: u32,
        threads: u32,
        out: *mut COutput,
    ) -> c_int;
    fn zm_av1_free(p: *mut u8);
    fn zm_check_decode(p: *const u8, len: usize, w: u32, h: u32) -> c_int;
}
impl Drop for COutput {
    fn drop(&mut self) {
        // SAFETY: C allocates this pointer with malloc/realloc; it is owned only
        // by this guard, including on failure. free(NULL) is permitted.
        unsafe { zm_av1_free(self.data) }
    }
}

/// Fresh encoder lifecycle including setup, input preparation and teardown.
/// Callers time this API only; file reads, hashing, decoding and scoring are outside.
pub fn encode(cfg: Config, pixels: &[u8]) -> Result<Vec<u8>, String> {
    cfg.validate(pixels)?;
    let w = cfg.width as usize;
    let h = cfg.height as usize;
    let (y, uv) = pixels.split_at(w * h);
    let (u, v) = uv.split_at(w * h / 4);
    match cfg.backend {
        Backend::Libaom | Backend::CSvt => {
            let mut out = COutput {
                data: ptr::null_mut(),
                len: 0,
            };
            // SAFETY: exact validated I420 lengths and bounded integer dimensions;
            // C borrows pixels only until its synchronous encoder drain completes.
            let rc = unsafe {
                let f = if matches!(cfg.backend, Backend::Libaom) {
                    zm_libaom
                } else {
                    zm_c_svt
                };
                f(
                    pixels.as_ptr(),
                    cfg.width,
                    cfg.height,
                    cfg.quantizer,
                    cfg.speed,
                    cfg.threads,
                    &mut out,
                )
            };
            if rc != 0 {
                return Err(format!(
                    "{:?} public C API failed at stage {rc}",
                    cfg.backend
                ));
            }
            if out.data.is_null() || out.len == 0 || out.len > isize::MAX as usize {
                return Err("invalid C output buffer".into());
            }
            // SAFETY: successful C call owns out.len initialized bytes until Drop.
            Ok(unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec())
        }
        Backend::Svt => {
            // Invert this wrapper's documented integer QP mapping, then verify
            // round-trip rather than silently compare a neighbouring quantizer.
            let quality = 100.0 - cfg.quantizer as f32 * 99.0 / 63.0;
            if svtav1::avif::AvifEncoder::quality_to_qp_static(quality) != cfg.quantizer as u8 {
                return Err("SVT quality-to-QP round-trip failed".into());
            }
            svtav1::avif::AvifEncoder::new()
                .with_quality(quality)
                .with_speed(cfg.speed as u8)
                .with_num_threads(Some(1))
                .encode_yuv420(y, u, v, cfg.width, cfg.height, cfg.width)
                .map(|o| o.data)
                .map_err(|e| e.to_string())
        }
        Backend::Aom => {
            let mut k = aom_encode::key_frame::KeyFrameConfig::allintra_speed0(
                w,
                h,
                8,
                false,
                1,
                1,
                cfg.quantizer as i32,
            );
            k.cpu_used = cfg.speed as i32;
            k.enable_restoration = true;
            let planes = [y, u, v].map(|p| p.iter().map(|&v| u16::from(v)).collect::<Vec<_>>());
            aom_encode::key_frame::encode_key_frame(
                aom_encode::key_frame::KeyFramePlanes {
                    y: &planes[0],
                    u: &planes[1],
                    v: &planes[2],
                },
                &k,
            )
            .map_err(|e| e.to_string())
        }
        Backend::Rav1e => {
            use zenrav1e::prelude::*;
            let e = EncoderConfig {
                width: w,
                height: h,
                bit_depth: 8,
                chroma_sampling: ChromaSampling::Cs420,
                pixel_range: PixelRange::Limited,
                still_picture: true,
                quantizer: cfg.quantizer as usize,
                min_quantizer: cfg.quantizer as u8,
                speed_settings: SpeedSettings::from_preset(cfg.speed as u8),
                ..Default::default()
            };
            let mut ctx: Context<u8> = zenrav1e::Config::new()
                .with_encoder_config(e)
                .with_threads(1)
                .new_context()
                .map_err(|e| format!("{e:?}"))?;
            let mut f = ctx.new_frame();
            for (i, p) in [y, u, v].into_iter().enumerate() {
                f.planes[i].copy_from_raw_u8(p, if i == 0 { w } else { w / 2 }, 1);
            }
            ctx.send_frame(f).map_err(|e| e.to_string())?;
            ctx.flush();
            let mut bytes = Vec::new();
            loop {
                match ctx.receive_packet() {
                    Ok(p) => bytes.extend(p.data),
                    Err(EncoderStatus::Encoded) => continue,
                    Err(EncoderStatus::LimitReached) => break,
                    Err(e) => return Err(e.to_string()),
                }
            }
            if bytes.is_empty() {
                return Err("zenrav1e returned no frame".into());
            }
            Ok(bytes)
        }
    }
}
/// Independent libaom decode and format check, always outside the encode timer.
pub fn check_decode(obu: &[u8], width: u32, height: u32) -> Result<(), String> {
    // SAFETY: decoder borrows the complete immutable slice only for this call.
    let rc = unsafe { zm_check_decode(obu.as_ptr(), obu.len(), width, height) };
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("libaom decode/format check failed: {rc}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_five_arms_decode() {
        // Correctness witness, not a performance corpus or calibration result.
        let mut pixels = vec![128; 64 * 64 * 3 / 2];
        for (i, y) in pixels[..64 * 64].iter_mut().enumerate() {
            *y = if (i / 8 + i / 512).is_multiple_of(2) {
                32
            } else {
                220
            };
        }
        for backend in [
            Backend::Libaom,
            Backend::CSvt,
            Backend::Svt,
            Backend::Aom,
            Backend::Rav1e,
        ] {
            let cfg = Config {
                backend,
                width: 64,
                height: 64,
                quantizer: 40,
                speed: 6,
                threads: 1,
            };
            let obu = encode(cfg, &pixels).unwrap_or_else(|e| panic!("{backend:?}: {e}"));
            check_decode(&obu, 64, 64).unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        }
    }
    #[test]
    fn invalid_requests_fail_before_ffi() {
        let cfg = Config {
            backend: Backend::Libaom,
            width: 64,
            height: 64,
            quantizer: 40,
            speed: 6,
            threads: 1,
        };
        assert!(encode(cfg, &[]).is_err());
        assert!(
            encode(
                Config {
                    width: u32::MAX,
                    ..cfg
                },
                &[]
            )
            .is_err()
        );
        assert!(
            encode(
                Config {
                    quantizer: 64,
                    ..cfg
                },
                &vec![0; 6144]
            )
            .is_err()
        );
        assert!(encode(Config { threads: 2, ..cfg }, &vec![0; 6144]).is_err());
        assert!(check_decode(&[0xff; 16], 64, 64).is_err());
    }
}
