use yuv::{YuvChromaSubsampling as Cs, YuvRange as Range, YuvStandardMatrix as Matrix};
use zenmetrics_av1_compare::{Chroma, Config};
type Error = Box<dyn std::error::Error>;
fn sampling(c: Chroma) -> Cs {
    match c {
        Chroma::Cs420 | Chroma::Mono => Cs::Yuv420,
        Chroma::Cs422 => Cs::Yuv422,
        Chroma::Cs444 => Cs::Yuv444,
    }
}
pub fn prepare(rgb: &[u8], cfg: Config) -> Result<Vec<u8>, Error> {
    let (w, h) = (cfg.width, cfg.height);
    let samples = if cfg.bit_depth == 8 {
        let mut p = yuv::YuvPlanarImageMut::<u8>::alloc(w, h, sampling(cfg.chroma));
        let f = match cfg.chroma {
            Chroma::Cs420 | Chroma::Mono => yuv::rgb_to_yuv420,
            Chroma::Cs422 => yuv::rgb_to_yuv422,
            Chroma::Cs444 => yuv::rgb_to_yuv444,
        };
        f(
            &mut p,
            rgb,
            w * 3,
            Range::Limited,
            Matrix::Bt709,
            yuv::YuvConversionMode::Balanced,
        )?;
        let p = p.to_fixed();
        let data = if cfg.chroma == Chroma::Mono {
            p.y_plane.to_vec()
        } else {
            [p.y_plane, p.u_plane, p.v_plane].concat()
        };
        return Ok(data);
    } else {
        let max = (1u32 << cfg.bit_depth) - 1;
        let native = rgb
            .iter()
            .map(|&v| ((u32::from(v) * max + 127) / 255) as u16)
            .collect::<Vec<_>>();
        let mut p = yuv::YuvPlanarImageMut::<u16>::alloc(w, h, sampling(cfg.chroma));
        let f = match (cfg.bit_depth, cfg.chroma) {
            (10, Chroma::Cs420 | Chroma::Mono) => yuv::rgb10_to_i010,
            (10, Chroma::Cs422) => yuv::rgb10_to_i210,
            (10, Chroma::Cs444) => yuv::rgb10_to_i410,
            (12, Chroma::Cs420 | Chroma::Mono) => yuv::rgb12_to_i012,
            (12, Chroma::Cs422) => yuv::rgb12_to_i212,
            (12, Chroma::Cs444) => yuv::rgb12_to_i412,
            _ => return Err("unsupported conversion precision".into()),
        };
        f(&mut p, &native, w * 3, Range::Limited, Matrix::Bt709)?;
        let p = p.to_fixed();
        if cfg.chroma == Chroma::Mono {
            p.y_plane.to_vec()
        } else {
            [p.y_plane, p.u_plane, p.v_plane].concat()
        }
    };
    Ok(samples.into_iter().flat_map(u16::to_le_bytes).collect())
}
pub fn unpack(bytes: &[u8], depth: u8) -> Vec<u16> {
    if depth == 8 {
        bytes.iter().map(|&v| u16::from(v)).collect()
    } else {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect()
    }
}
pub fn rgb_from_samples(data: &[u16], cfg: Config) -> Result<Vec<u8>, Error> {
    let (w, h) = (cfg.width, cfg.height);
    let (yn, cn) = cfg.plane_lengths();
    let (sx, _) = cfg.chroma.shifts();
    if cfg.chroma == Chroma::Mono {
        let offset = 16i32 << (cfg.bit_depth - 8);
        let span = 219i32 << (cfg.bit_depth - 8);
        return Ok(data
            .iter()
            .flat_map(|&v| {
                let v = (((i32::from(v) - offset) * 255 + span / 2) / span).clamp(0, 255) as u8;
                [v, v, v]
            })
            .collect());
    }
    let p = yuv::YuvPlanarImage {
        y_plane: &data[..yn],
        y_stride: w,
        u_plane: &data[yn..yn + cn],
        u_stride: w.div_ceil(1 << sx),
        v_plane: &data[yn + cn..],
        v_stride: w.div_ceil(1 << sx),
        width: w,
        height: h,
    };
    if cfg.bit_depth == 8 {
        let bytes = data.iter().map(|&v| v as u8).collect::<Vec<_>>();
        let p = yuv::YuvPlanarImage {
            y_plane: &bytes[..yn],
            y_stride: w,
            u_plane: &bytes[yn..yn + cn],
            u_stride: p.u_stride,
            v_plane: &bytes[yn + cn..],
            v_stride: p.v_stride,
            width: w,
            height: h,
        };
        let mut rgb = vec![0; yn * 3];
        let f = match cfg.chroma {
            Chroma::Cs420 => yuv::yuv420_to_rgb,
            Chroma::Cs422 => yuv::yuv422_to_rgb,
            Chroma::Cs444 => yuv::yuv444_to_rgb,
            _ => unreachable!(),
        };
        f(&p, &mut rgb, w * 3, Range::Limited, Matrix::Bt709)?;
        return Ok(rgb);
    }
    let f = match (cfg.bit_depth, cfg.chroma) {
        (10, Chroma::Cs420) => yuv::i010_to_rgb10,
        (10, Chroma::Cs422) => yuv::i210_to_rgb10,
        (10, Chroma::Cs444) => yuv::i410_to_rgb10,
        (12, Chroma::Cs420) => yuv::i012_to_rgb12,
        (12, Chroma::Cs422) => yuv::i212_to_rgb12,
        (12, Chroma::Cs444) => yuv::i412_to_rgb12,
        _ => return Err("unsupported decode conversion".into()),
    };
    let mut rgb = vec![0; yn * 3];
    f(&p, &mut rgb, w * 3, Range::Limited, Matrix::Bt709)?;
    let max = (1u32 << cfg.bit_depth) - 1;
    Ok(rgb
        .into_iter()
        .map(|v| ((u32::from(v) * 255 + max / 2) / max) as u8)
        .collect())
}
