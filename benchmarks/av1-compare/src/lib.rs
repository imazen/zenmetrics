//! In-process comparison arms for the zenfleet executor. The initial protocol
//! covers limited-range stills at each backend's native precision and chroma.
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum Chroma {
    #[default]
    #[serde(rename = "420")]
    Cs420,
    #[serde(rename = "422")]
    Cs422,
    #[serde(rename = "444")]
    Cs444,
    #[serde(rename = "mono")]
    Mono,
}
impl Chroma {
    pub fn shifts(self) -> (usize, usize) {
        match self {
            Self::Cs420 | Self::Mono => (1, 1),
            Self::Cs422 => (1, 0),
            Self::Cs444 => (0, 0),
        }
    }
}
/// Reference used for SVT decisions, independent of the Rust build revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum SvtSource {
    #[serde(rename = "svt-mainline-4.2.0-9292ec8e32bce26f781f277ec8739b53426c4300")]
    Mainline420,
    #[serde(rename = "svt-hybrid-3115c0c1b23e860dfd75c94f6740e0298182dd13")]
    Hybrid3115,
}
impl SvtSource {
    fn reference(self) -> svtav1::avif::SvtReference {
        match self {
            Self::Mainline420 => svtav1::avif::SvtReference::Mainline420,
            Self::Hybrid3115 => svtav1::avif::SvtReference::Hybrid3115,
        }
    }
}
fn eight() -> u8 {
    8
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub backend: Backend,
    pub width: u32,
    pub height: u32,
    /// Native quantizer: 0..63 for AOM/SVT, 0..255 for zenrav1e.
    pub quantizer: u32,
    pub speed: i32,
    /// Requested threads (C SVT interprets this as lp). Rust AOM currently requires 1.
    pub threads: u32,
    #[serde(default = "eight")]
    pub bit_depth: u8,
    #[serde(default)]
    pub chroma: Chroma,
    #[serde(default)]
    pub tune: Option<u8>,
    #[serde(default)]
    pub scm: Option<u8>,
    #[serde(default)]
    pub sb128: bool,
    #[serde(default)]
    pub svt_reference: Option<SvtSource>,
    #[serde(default)]
    pub zen_intra_edge_filter: bool,
}
impl Config {
    /// Legacy requests used the hybrid source. Always resolve that identity in
    /// output records, including requests which omit the explicit selector.
    pub fn resolved_svt_reference(&self) -> Option<&'static str> {
        matches!(self.backend, Backend::CSvt | Backend::Svt).then(|| {
            self.svt_reference
                .unwrap_or(SvtSource::Hybrid3115)
                .reference()
                .id()
        })
    }

    pub fn plane_lengths(&self) -> (usize, usize) {
        let (sx, sy) = self.chroma.shifts();
        (
            self.width as usize * self.height as usize,
            if self.chroma == Chroma::Mono {
                0
            } else {
                (self.width as usize).div_ceil(1 << sx) * (self.height as usize).div_ceil(1 << sy)
            },
        )
    }
    pub fn validate_configuration(&self) -> Result<(), String> {
        if !(64..=16384).contains(&self.width) || !(64..=16384).contains(&self.height) {
            return Err("comparison protocol requires dimensions in 64..=16384".into());
        }
        if !matches!(self.bit_depth, 8 | 10 | 12) {
            return Err("bit depth must be 8, 10 or 12".into());
        }
        if !(1..=16).contains(&self.threads) {
            return Err("thread setting must be 1..=16".into());
        }
        let (qmax, smax) = match self.backend {
            Backend::Rav1e => (255, 10),
            Backend::CSvt => (63, 13),
            _ => (63, 9),
        };
        let smin = if matches!(self.backend, Backend::CSvt | Backend::Svt) {
            -1
        } else {
            0
        };
        if self.quantizer > qmax || !(smin..=smax).contains(&self.speed) {
            return Err("native quantizer or preset out of range".into());
        }
        if matches!(self.backend, Backend::CSvt | Backend::Svt) {
            if self.bit_depth == 12 || !matches!(self.chroma, Chroma::Cs420 | Chroma::Mono) {
                return Err("SVT supports 8/10-bit 420; Rust additionally supports mono".into());
            }
            if matches!(self.backend, Backend::CSvt)
                && (self.chroma == Chroma::Mono
                    || !self.width.is_multiple_of(2)
                    || !self.height.is_multiple_of(2))
            {
                return Err("C SVT requires even 420; monochrome is a Rust extension".into());
            }
        } else if self.tune.is_some() || self.scm.is_some() {
            return Err("explicit SVT tools require an SVT backend".into());
        }
        if self.tune.is_some_and(|v| v > 4) || self.scm.is_some_and(|v| v > 2) {
            return Err("SVT tune/SCM outside supported range".into());
        }
        if let Some(reference) = self.svt_reference {
            if !matches!(self.backend, Backend::CSvt | Backend::Svt) {
                return Err("SVT reference requires an SVT backend".into());
            }
            if reference == SvtSource::Mainline420
                && (matches!(self.backend, Backend::CSvt) || self.chroma == Chroma::Mono)
            {
                return Err("pristine reference requires Rust SVT 420; linked C is hybrid".into());
            }
        }
        if self.zen_intra_edge_filter
            && (!matches!(self.backend, Backend::Svt)
                || self.speed != -1
                || self.chroma != Chroma::Cs420)
        {
            return Err("AOM intra-edge experiment requires Rust SVT native -1 and 420".into());
        }
        if self.sb128 && !matches!(self.backend, Backend::Libaom | Backend::Aom) {
            return Err("explicit SB size is currently an AOM arm".into());
        }
        if matches!(self.backend, Backend::Aom) && self.threads != 1 {
            return Err("standalone Rust AOM has no threaded encoder API".into());
        }
        Ok(())
    }
    pub fn validate(&self, pixels: &[u8]) -> Result<(), String> {
        self.validate_configuration()?;
        let (y, c) = self.plane_lengths();
        let size = (y + 2 * c) * if self.bit_depth == 8 { 1 } else { 2 };
        if pixels.len() != size {
            return Err(format!(
                "expected {size} packed planar bytes, got {}",
                pixels.len()
            ));
        }
        if self.bit_depth > 8
            && pixels
                .as_chunks::<2>()
                .0
                .iter()
                .any(|p| u16::from_le_bytes([p[0], p[1]]) >= (1 << self.bit_depth))
        {
            return Err("input sample exceeds coded bit depth".into());
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

/// Replay an SVT measurement outside its timed interval and verify the exact
/// measured stream against the encoder's final reconstruction. Requiring the
/// same bytes prevents a differently configured replay from passing the gate.
pub fn verify_svt_reconstruction(
    cfg: Config,
    pixels: &[u8],
    expected: &[u8],
) -> Result<(), String> {
    cfg.validate(pixels)?;
    if !matches!(cfg.backend, Backend::Svt) {
        return Err("encoder reconstruction verification requires Rust SVT".into());
    }
    let (yn, cn) = cfg.plane_lengths();
    let bps = if cfg.bit_depth == 8 { 1 } else { 2 };
    let (y, uv) = pixels.split_at(yn * bps);
    let (u, v) = uv.split_at(cn * bps);
    encode_svt(cfg, y, u, v, Some(expected)).map(|_| ())
}

fn crop_svt_recon<T: Copy + Into<u16>>(
    planes: &(Vec<T>, Vec<T>, Vec<T>),
    stride: usize,
    cfg: Config,
) -> Vec<u16> {
    let mut result = Vec::new();
    for (plane, data) in [&planes.0, &planes.1, &planes.2].into_iter().enumerate() {
        if plane > 0 && cfg.chroma == Chroma::Mono {
            break;
        }
        let shift = usize::from(plane > 0);
        let width = (cfg.width as usize).div_ceil(1 << shift);
        let height = (cfg.height as usize).div_ceil(1 << shift);
        let stride = stride.div_ceil(1 << shift);
        for row in 0..height {
            result.extend(
                data[row * stride..row * stride + width]
                    .iter()
                    .copied()
                    .map(Into::into),
            );
        }
    }
    result
}

fn encode_svt(
    cfg: Config,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    expected_obu: Option<&[u8]>,
) -> Result<Vec<u8>, String> {
    let w = cfg.width as usize;
    use svtav1::encoder::{
        pipeline::EncodePipeline,
        rate_control::{RcConfig, RcMode},
        speed_config::NativePreset,
    };
    let rc = RcConfig {
        mode: RcMode::Cqp,
        qp: cfg.quantizer as u8,
        ..Default::default()
    };
    let preset = NativePreset::new(i8::try_from(cfg.speed).map_err(|e| e.to_string())?)
        .ok_or("native SVT preset out of range")?;
    let mut p = EncodePipeline::new_with_preset(cfg.width, cfg.height, preset, rc, 0, 1)
        .with_chroma_420(cfg.chroma == Chroma::Cs420)
        .with_thread_count(cfg.threads as usize)
        .with_bit_depth(cfg.bit_depth)
        .with_recon_output(expected_obu.is_some());
    p.reference = cfg
        .svt_reference
        .unwrap_or(SvtSource::Hybrid3115)
        .reference();
    if cfg.zen_intra_edge_filter {
        p.enhancements = p
            .enhancements
            .with(svtav1::avif::ZenEnhancement::AomIntraEdgeFilter);
    }
    if let Some(tune) = cfg.tune {
        p.hdr.tune = tune;
    }
    if let Some(scm) = cfg.scm {
        p.hdr.screen_content_mode = Some(scm);
    }
    let r = if cfg.bit_depth == 8 {
        if cfg.chroma == Chroma::Mono {
            p.try_encode_frame(y, w)
        } else {
            p.try_encode_frame_420(y, u, v, w)
        }
    } else {
        let s = [y, u, v].map(|p| {
            p.as_chunks::<2>()
                .0
                .iter()
                .map(|p| u16::from_le_bytes([p[0], p[1]]))
                .collect::<Vec<_>>()
        });
        if cfg.chroma == Chroma::Mono {
            p.try_encode_frame_hbd(&s[0], w)
        } else {
            p.try_encode_frame_420_hbd(&s[0], &s[1], &s[2], w)
        }
    };
    let obu = r.map_err(|e| e.to_string())?;
    if let Some(expected) = expected_obu {
        if obu != expected {
            return Err("reconstruction verification did not reproduce measured OBU bytes".into());
        }
        let recon = if cfg.bit_depth == 8 {
            crop_svt_recon(
                p.last_recon
                    .as_ref()
                    .ok_or("missing native8 reconstruction")?,
                p.width as usize,
                cfg,
            )
        } else {
            crop_svt_recon(
                p.last_recon10_final
                    .as_ref()
                    .ok_or("missing native10 reconstruction")?,
                p.width as usize,
                cfg,
            )
        };
        let decoded = decode_planar(&obu, cfg)?;
        if recon != decoded {
            return Err(format!(
                "encoder/decoder reconstruction mismatch: first sample {:?}, expected {} samples, decoded {}",
                recon.iter().zip(&decoded).position(|(a, b)| a != b),
                recon.len(),
                decoded.len()
            ));
        }
    }
    Ok(obu)
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
        speed: i32,
        threads: u32,
        bd: u32,
        sx: u32,
        sy: u32,
        mono: u32,
        tune: i32,
        scm: i32,
        sb128: u32,
        out: *mut COutput,
    ) -> c_int;
    fn zm_c_svt(
        p: *const u8,
        w: u32,
        h: u32,
        q: u32,
        speed: i32,
        threads: u32,
        bd: u32,
        sx: u32,
        sy: u32,
        mono: u32,
        tune: i32,
        scm: i32,
        sb128: u32,
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
    let (yn, cn) = cfg.plane_lengths();
    let bps = if cfg.bit_depth == 8 { 1 } else { 2 };
    let (y, uv) = pixels.split_at(yn * bps);
    let (u, v) = uv.split_at(cn * bps);
    let samples = || {
        [y, u, v].map(|p| {
            if cfg.bit_depth == 8 {
                p.iter().map(|&v| u16::from(v)).collect::<Vec<_>>()
            } else {
                p.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u16::from_le_bytes([p[0], p[1]]))
                    .collect()
            }
        })
    };
    let (sx, sy) = cfg.chroma.shifts();
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
                    cfg.bit_depth as u32,
                    sx as u32,
                    sy as u32,
                    u32::from(cfg.chroma == Chroma::Mono),
                    cfg.tune.map_or(-1, i32::from),
                    cfg.scm.map_or(-1, i32::from),
                    u32::from(cfg.sb128),
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
        Backend::Svt => encode_svt(cfg, y, u, v, None),
        Backend::Aom => {
            let mut k = aom_encode::key_frame::KeyFrameConfig::allintra_speed0(
                w,
                h,
                cfg.bit_depth,
                cfg.chroma == Chroma::Mono,
                sx,
                sy,
                cfg.quantizer as i32,
            );
            k.cpu_used = cfg.speed;
            k.enable_restoration = true;
            k.sb_size_128 = cfg.sb128;
            let s = samples();
            aom_encode::key_frame::encode_key_frame(
                aom_encode::key_frame::KeyFramePlanes {
                    y: &s[0],
                    u: &s[1],
                    v: &s[2],
                },
                &k,
            )
            .map_err(|e| e.to_string())
        }
        Backend::Rav1e => {
            if cfg.bit_depth == 8 {
                rav_encode::<u8>(cfg, [y, u, v])
            } else {
                rav_encode::<u16>(cfg, [y, u, v])
            }
        }
    }
}
fn rav_encode<T: zenrav1e::prelude::Pixel>(
    cfg: Config,
    planes: [&[u8]; 3],
) -> Result<Vec<u8>, String> {
    use zenrav1e::prelude::*;
    let chroma_sampling = match cfg.chroma {
        Chroma::Cs420 => ChromaSampling::Cs420,
        Chroma::Cs422 => ChromaSampling::Cs422,
        Chroma::Cs444 => ChromaSampling::Cs444,
        Chroma::Mono => ChromaSampling::Cs400,
    };
    let e = EncoderConfig {
        width: cfg.width as usize,
        height: cfg.height as usize,
        bit_depth: cfg.bit_depth as usize,
        chroma_sampling,
        pixel_range: PixelRange::Limited,
        still_picture: true,
        quantizer: cfg.quantizer as usize,
        min_quantizer: cfg.quantizer as u8,
        speed_settings: SpeedSettings::from_preset(cfg.speed as u8),
        ..Default::default()
    };
    let mut ctx: Context<T> = zenrav1e::Config::new()
        .with_encoder_config(e)
        .with_threads(cfg.threads as usize)
        .new_context()
        .map_err(|e| format!("{e:?}"))?;
    let mut f = ctx.new_frame();
    let bps = if cfg.bit_depth == 8 { 1 } else { 2 };
    let (sx, _) = cfg.chroma.shifts();
    for (i, p) in planes.into_iter().enumerate() {
        if !p.is_empty() {
            f.planes[i].copy_from_raw_u8(
                p,
                if i == 0 {
                    cfg.width as usize * bps
                } else {
                    (cfg.width as usize).div_ceil(1 << sx) * bps
                },
                bps,
            );
        }
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
        Err("zenrav1e returned no frame".into())
    } else {
        Ok(bytes)
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
                bit_depth: 8,
                chroma: Chroma::Cs420,
                tune: None,
                scm: None,
                sb128: false,
                svt_reference: None,
                zen_intra_edge_filter: false,
            };
            let obu = encode(cfg, &pixels).unwrap_or_else(|e| panic!("{backend:?}: {e}"));
            check_decode(&obu, 64, 64).unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        }
    }
    #[test]
    fn both_svt_research_presets_match_and_other_backends_refuse_it() {
        let cfg = Config {
            backend: Backend::CSvt,
            width: 64,
            height: 64,
            quantizer: 25,
            speed: -1,
            threads: 1,
            bit_depth: 8,
            chroma: Chroma::Cs420,
            tune: None,
            scm: None,
            sb128: false,
            svt_reference: None,
            zen_intra_edge_filter: false,
        };
        let pixels: Vec<u8> = (0..64 * 64 * 3 / 2)
            .map(|i| ((i * 37 + i / 13 * 53) % 220 + 16) as u8)
            .collect();
        for bit_depth in [8, 10] {
            let cfg = Config { bit_depth, ..cfg };
            // Exercise native samples with nonzero low bits in the 10-bit case.
            let input = if bit_depth == 8 {
                pixels.clone()
            } else {
                pixels
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &v)| (u16::from(v) * 4 + (i % 4) as u16).to_le_bytes())
                    .collect()
            };
            let c = encode(cfg, &input).expect("C public research preset -1");
            let rust = encode(
                Config {
                    backend: Backend::Svt,
                    ..cfg
                },
                &input,
            )
            .expect("Rust native research preset -1");
            decode_planar(&c, cfg).unwrap();
            decode_planar(&rust, cfg).unwrap();
            verify_svt_reconstruction(
                Config {
                    backend: Backend::Svt,
                    ..cfg
                },
                &input,
                &rust,
            )
            .unwrap();
            assert_eq!(c, rust, "research preset parity at {bit_depth} bits");
        }
        for backend in [Backend::Libaom, Backend::Aom, Backend::Rav1e] {
            assert!(Config { backend, ..cfg }.validate_configuration().is_err());
        }
        // Enum entries -2/-3 exist, but this C build's public minimum is -1.
        for backend in [Backend::CSvt, Backend::Svt] {
            for speed in [-2, -3] {
                assert!(
                    Config {
                        backend,
                        speed,
                        ..cfg
                    }
                    .validate_configuration()
                    .is_err()
                );
            }
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
            bit_depth: 8,
            chroma: Chroma::Cs420,
            tune: None,
            scm: None,
            sb128: false,
            svt_reference: None,
            zen_intra_edge_filter: false,
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
        assert!(encode(Config { threads: 17, ..cfg }, &vec![0; 6144]).is_err());
        assert!(check_decode(&[0xff; 16], 64, 64).is_err());
    }
}

/// Copy independently decoded 8-bit limited-range I420 planes for scoring.
pub fn decode_i420(obu: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    if !(64..=16384).contains(&width)
        || !(64..=16384).contains(&height)
        || !width.is_multiple_of(2)
        || !height.is_multiple_of(2)
    {
        return Err("invalid decode dimensions".into());
    }
    let mut out = vec![0; width as usize * height as usize * 3 / 2];
    unsafe extern "C" {
        fn zm_decode_i420(p: *const u8, len: usize, w: u32, h: u32, dst: *mut u8) -> c_int;
    }
    // SAFETY: the shim verifies exact dimensions/format before writing this
    // checked-size output. Both input and output live throughout the call.
    let rc = unsafe { zm_decode_i420(obu.as_ptr(), obu.len(), width, height, out.as_mut_ptr()) };
    if rc == 0 {
        Ok(out)
    } else {
        Err(format!("decode I420 failed: {rc}"))
    }
}

/// Independently decode to native-precision planes, verifying every format field.
pub fn decode_planar(obu: &[u8], cfg: Config) -> Result<Vec<u16>, String> {
    cfg.validate_configuration()?;
    let (y, c) = cfg.plane_lengths();
    let (sx, sy) = cfg.chroma.shifts();
    let mut out = vec![0; y + 2 * c];
    unsafe extern "C" {
        fn zm_decode_planar(
            p: *const u8,
            len: usize,
            w: u32,
            h: u32,
            bd: u32,
            sx: u32,
            sy: u32,
            mono: u32,
            dst: *mut u16,
        ) -> c_int;
    }
    // SAFETY: validated bounded dimensions determine the output allocation. The
    // C shim verifies the decoded format before writing and retains no pointers.
    let rc = unsafe {
        zm_decode_planar(
            obu.as_ptr(),
            obu.len(),
            cfg.width,
            cfg.height,
            cfg.bit_depth as u32,
            sx as u32,
            sy as u32,
            u32::from(cfg.chroma == Chroma::Mono),
            out.as_mut_ptr(),
        )
    };
    if rc == 0 {
        Ok(out)
    } else {
        Err(format!("native decode/format check failed: {rc}"))
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;
    fn config(backend: Backend, bit_depth: u8, chroma: Chroma) -> Config {
        Config {
            backend,
            width: 96,
            height: 80,
            quantizer: 0,
            speed: 6,
            threads: 1,
            bit_depth,
            chroma,
            tune: None,
            scm: None,
            sb128: false,
            svt_reference: None,
            zen_intra_edge_filter: false,
        }
    }
    #[test]
    fn explicit_reference_and_intra_edge_are_encoded_and_recorded() {
        let raw = include_bytes!(
            "../../../../zenav1-svt/rust/svtav1/tests/fixtures/reference_chroma/diag64-8.yuv"
        );
        let mut cfg = config(Backend::Svt, 8, Chroma::Cs420);
        cfg.width = 64;
        cfg.height = 64;
        cfg.quantizer = 48;
        cfg.speed = 0;
        cfg.svt_reference = Some(SvtSource::Mainline420);
        let pristine = encode(cfg, raw).unwrap();
        assert_eq!(
            pristine,
            include_bytes!(
                "../../../../zenav1-svt/rust/svtav1/tests/fixtures/reference_chroma/diag64-8-p0-mainline.obu"
            )
        );
        let hybrid = encode(
            Config {
                svt_reference: Some(SvtSource::Hybrid3115),
                ..cfg
            },
            raw,
        )
        .unwrap();
        assert_ne!(pristine, hybrid);
        cfg.speed = -1;
        let native = encode(cfg, raw).unwrap();
        cfg.zen_intra_edge_filter = true;
        let enhanced = encode(cfg, raw).unwrap();
        verify_svt_reconstruction(cfg, raw, &enhanced).unwrap();
        assert!(verify_svt_reconstruction(cfg, raw, &native).is_err());
        assert_ne!(
            decode_planar(&native, cfg).unwrap(),
            decode_planar(&enhanced, cfg).unwrap()
        );
        let replay: Config = serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(
            replay.resolved_svt_reference(),
            cfg.resolved_svt_reference()
        );
        assert_eq!(encode(replay, raw).unwrap(), enhanced);
        for backend in [Backend::CSvt, Backend::Libaom, Backend::Aom, Backend::Rav1e] {
            assert!(Config { backend, ..cfg }.validate_configuration().is_err());
        }
        assert!(Config { speed: 0, ..cfg }.validate_configuration().is_err());
        assert!(
            Config {
                chroma: Chroma::Mono,
                ..cfg
            }
            .validate_configuration()
            .is_err()
        );
        assert!(
            Config {
                backend: Backend::CSvt,
                zen_intra_edge_filter: false,
                ..cfg
            }
            .validate_configuration()
            .is_err()
        );
    }
    #[test]
    fn native_lossless_format_matrix() {
        let mut count = 0;
        for backend in [
            Backend::Libaom,
            Backend::Aom,
            Backend::Rav1e,
            Backend::Svt,
            Backend::CSvt,
        ] {
            for depth in [8, 10, 12] {
                for chroma in [Chroma::Cs420, Chroma::Cs422, Chroma::Cs444, Chroma::Mono] {
                    let cfg = config(backend, depth, chroma);
                    let expected = match backend {
                        Backend::CSvt => depth != 12 && chroma == Chroma::Cs420,
                        Backend::Svt => {
                            depth != 12 && matches!(chroma, Chroma::Cs420 | Chroma::Mono)
                        }
                        _ => true,
                    };
                    assert_eq!(cfg.validate_configuration().is_ok(), expected, "{cfg:?}");
                    if !expected {
                        continue;
                    } // Explicitly tested unsupported matrix cell.
                    let (y, c) = cfg.plane_lengths();
                    // Nonzero low bits at 10/12 bits catch accidental 8-bit truncation.
                    let samples = (0..y + 2 * c)
                        .map(|i| ((i * 37 + i / 13 * 19 + 7) % ((1usize << depth) - 1)) as u16)
                        .collect::<Vec<_>>();
                    let packed: Vec<u8> = if depth == 8 {
                        samples.iter().map(|&v| v as u8).collect()
                    } else {
                        samples.iter().flat_map(|v| v.to_le_bytes()).collect()
                    };
                    let obu = encode(cfg, &packed).unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
                    let decoded =
                        decode_planar(&obu, cfg).unwrap_or_else(|e| panic!("{cfg:?}: {e}"));
                    assert!(
                        decoded == samples,
                        "lossless native-plane mismatch: {cfg:?}"
                    );
                    if matches!(backend, Backend::Svt) {
                        verify_svt_reconstruction(cfg, &packed, &obu).unwrap();
                    }
                    count += 1;
                }
            }
        }
        assert_eq!(count, 42);
    }
    #[test]
    fn format_mismatch_is_refused_by_decoder() {
        let cfg = config(Backend::Libaom, 8, Chroma::Cs444);
        let obu = encode(cfg, &vec![128; 96 * 80 * 3]).unwrap();
        assert!(
            decode_planar(
                &obu,
                Config {
                    chroma: Chroma::Cs420,
                    ..cfg
                }
            )
            .is_err()
        );
        assert!(
            decode_planar(
                &obu,
                Config {
                    bit_depth: 10,
                    ..cfg
                }
            )
            .is_err()
        );
    }
}
