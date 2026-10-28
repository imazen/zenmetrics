//! Natural SSIMULACRA2: native PQ -> BT.709 absolute linear -> nits/100.
#![forbid(unsafe_code)]
mod pq;
use std::{
    fs,
    io::{self, BufRead},
    path::Path,
};
use zenpixels::{ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor};
use zenpixels_convert::{RowConverter, policy::ConvertOptions};
type Error = Box<dyn std::error::Error>;
struct Image {
    data: Vec<[f32; 3]>,
    width: usize,
    height: usize,
}
// E26 common-bt709-pq-native-v3 owner, zenmetrics 348bde5b hdr.rs:
// decode PQ at native precision, convert absolute linear primaries without clipping.
fn normalize(rgb: &[f32], width: u32, cp: u8) -> Result<Vec<f32>, Error> {
    let mut out = if cp == 1 {
        rgb.to_vec()
    } else {
        let primaries = match cp {
            9 => ColorPrimaries::Bt2020,
            12 => ColorPrimaries::DisplayP3,
            _ => return Err("unsupported HDR primaries".into()),
        };
        let options = ConvertOptions::permissive().with_clip_out_of_gamut(false);
        let mut converter = RowConverter::new_explicit(
            PixelDescriptor::RGBF32_LINEAR.with_primaries(primaries),
            PixelDescriptor::RGBF32_LINEAR,
            &options,
        )?;
        let mut out = vec![0.0f32; rgb.len()];
        for (src, dst) in rgb
            .chunks_exact(width as usize * 3)
            .zip(out.chunks_exact_mut(width as usize * 3))
        {
            converter.convert_row(
                bytemuck::cast_slice(src),
                bytemuck::cast_slice_mut(dst),
                width,
            );
        }
        out
    };
    for v in &mut out {
        *v /= 100.0;
        if !v.is_finite() {
            return Err("nonfinite natural HDR input".into());
        }
    }
    Ok(out)
}
fn load(path: &Path) -> Result<Image, Error> {
    let decoded = zenpng::decode(
        &fs::read(path)?,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )?;
    let info = &decoded.info;
    if info.icc_profile.is_some()
        || info.srgb_intent.is_some()
        || info.source_gamma.is_some()
        || info.chromaticities.is_some()
    {
        return Err("conflicting HDR metadata".into());
    }
    let cp = info.cicp.ok_or("native PQ CICP required")?;
    if cp.transfer_characteristics != 16 || cp.matrix_coefficients != 0 || !cp.full_range {
        return Err("full-range RGB PQ required".into());
    }
    let s = decoded.pixels.as_slice();
    let d = s.descriptor();
    if d.channel_type() != ChannelType::U16 || d.layout() != ChannelLayout::Rgb {
        return Err("native RGB16 PQ required".into());
    }
    let (width, height) = (s.width() as usize, s.rows() as usize);
    if width * height >= 1_000_000 {
        return Err("HDR probe must be below 1MP".into());
    }
    let mut nits = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for sample in s.row(y as u32)[..width * 6].as_chunks::<2>().0 {
            nits.push(pq::pq_eotf(
                f32::from(u16::from_ne_bytes(*sample)) / 65535.0,
            ));
        }
    }
    let normalized = normalize(&nits, width as u32, cp.color_primaries)?;
    Ok(Image {
        data: normalized.as_chunks::<3>().0.to_vec(),
        width,
        height,
    })
}
fn main() -> Result<(), Error> {
    for line in io::stdin().lock().lines() {
        let pair: serde_json::Value = serde_json::from_str(&line?)?;
        let r = load(Path::new(
            pair["reference"].as_str().ok_or("reference missing")?,
        ))?;
        let d = load(Path::new(
            pair["distorted"].as_str().ok_or("distorted missing")?,
        ))?;
        if (r.width, r.height) != (d.width, d.height) {
            return Err("geometry differs".into());
        }
        let score = fast_ssim2::compute_ssimulacra2(
            imgref::Img::new(r.data.as_slice(), r.width, r.height),
            imgref::Img::new(d.data.as_slice(), d.width, d.height),
        )?;
        if !score.is_finite() {
            return Err("nonfinite natural HDR score".into());
        }
        println!(
            "{}",
            serde_json::json!({"reference":pair["reference"],"distorted":pair["distorted"],"value":score,"value_bits":format!("{:016x}",score.to_bits()),"implementation":"fast-ssim2@b7c2b4b33f3a26b753caabd5264a7fef9c4f4545;natural-cube-root;linear-BT709;nits/100;unclipped;E26-common-ingress@348bde5b"})
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_white_and_superwhite_stay_linear() {
        assert_eq!(
            normalize(&[100., 200., 1000.], 1, 1).unwrap(),
            vec![1., 2., 10.]
        );
    }
    #[test]
    fn primary_conversion_retains_out_of_gamut_light() {
        let out = normalize(&[100., 0., 0.], 1, 9).unwrap();
        assert!(out[0] > 1. && out[1] < 0. && out[2] < 0.);
    }
    #[test]
    fn pq_absolute_endpoints_match_e26() {
        assert_eq!(pq::pq_eotf(0.), 0.);
        assert_eq!(pq::pq_eotf(1.), 10000.);
    }
}
