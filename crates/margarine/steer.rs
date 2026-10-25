//! Separate native-linear steering candidate; the selected CLI is unchanged.
#![forbid(unsafe_code)]
#![allow(dead_code, deprecated)]
#![allow(clippy::manual_midpoint, clippy::chunks_exact_to_as_chunks)]
include!("kernel.rs");
use std::io::{BufWriter, Read, Write};

fn geometry(w: usize, h: usize) -> Result<usize, Box<dyn Error>> {
    if w == 0 || h == 0 {
        return Err("empty linear image".into());
    }
    w.checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| "linear image geometry overflow".into())
}

fn read_linear(input: &mut impl Read, samples: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut values = Vec::new();
    values.try_reserve_exact(samples)?;
    let mut bytes = [0u8; 12288];
    while values.len() < samples {
        let count = (samples - values.len()).min(bytes.len() / 4);
        input.read_exact(&mut bytes[..count * 4])?;
        for b in bytes[..count * 4].chunks_exact(4) {
            let v = f32::from_le_bytes(b.try_into()?);
            if !v.is_finite() {
                return Err("nonfinite linear sample".into());
            }
            values.push(v);
        }
    }
    Ok(values)
}

fn compute_linear(
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
) -> Result<diff::InternalResult, Box<dyn Error>> {
    let a = ingress::EncodedRows::new(ingress::Samples::Linear(a), w, h, w * 3, 3)?;
    let b = ingress::EncodedRows::new(ingress::Samples::Linear(b), w, h, w * 3, 3)?;
    row_psycho::compute_geometry(&a, &b, 128, 512, &ButteraugliParams::default())
}

fn write_map(out: &mut impl Write, map: &image::ImageF) -> Result<(), Box<dyn Error>> {
    for y in 0..map.height() {
        for v in map.row(y) {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn serve(input: &mut impl Read, out: &mut impl Write) -> Result<(), Box<dyn Error>> {
    writeln!(
        out,
        "{{\"metric\":\"margarine-linear\",\"candidate\":\"simd-row-malta-linear\",\"input\":\"linear-rgb-f32\",\"map\":\"diffmap\",\"scale\":1.0,\"exponent\":1.0}}"
    )?;
    out.flush()?;
    let mut reference = None;
    loop {
        let mut tag = [0u8; 1];
        if input.read(&mut tag)? == 0 {
            return Ok(());
        }
        if !matches!(tag[0], b'R' | b'D') {
            return Err("unknown linear map protocol tag".into());
        }
        let mut dims = [0u8; 8];
        input.read_exact(&mut dims)?;
        let w = u32::from_le_bytes(dims[..4].try_into()?) as usize;
        let h = u32::from_le_bytes(dims[4..].try_into()?) as usize;
        let pixels = read_linear(input, geometry(w, h)?)?;
        if tag[0] == b'R' {
            reference = Some((pixels, w, h));
            continue;
        }
        let (a, rw, rh) = reference
            .as_ref()
            .ok_or("distorted image before reference")?;
        if (*rw, *rh) != (w, h) {
            return Err("linear image dimensions differ".into());
        }
        let result = compute_linear(a, &pixels, w, h)?;
        let map = result.diffmap.as_ref().ok_or("missing map")?;
        if !result.score.is_finite() || (0..h).any(|y| map.row(y).iter().any(|v| !v.is_finite())) {
            return Err("nonfinite linear metric result".into());
        }
        out.write_all(&result.score.to_le_bytes())?;
        write_map(out, map)?;
        out.flush()?;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if CANDIDATE != "simd-row-malta"
        || cfg!(any(
            feature = "coarse-gaussian",
            feature = "native-gaussian",
            feature = "native-mask",
            feature = "phase-rows",
            feature = "tiles",
            feature = "compact",
            feature = "sparse",
            feature = "pooled",
            feature = "perceptual",
            feature = "physical",
            feature = "simd-opsin",
            feature = "wide-malta"
        ))
    {
        return Err("native-linear steering requires the selected row-Malta kernels without additional approximation features".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--server"] {
        return serve(
            &mut std::io::stdin().lock(),
            &mut BufWriter::new(std::io::stdout().lock()),
        );
    }
    if args.len() != 7 || args[0] != "--linear" || args[5] != "--diffmap" {
        return Err("usage: margarine-steer --server | --linear WIDTH HEIGHT REF.f32 DIST.f32 --diffmap NEW.f32\nLinear sRGB interleaved RGB f32le; finite values outside [0,1] are preserved.".into());
    }
    let (w, h) = (args[1].parse()?, args[2].parse()?);
    let n = geometry(w, h)?;
    let mut files = [
        std::fs::File::open(&args[3])?,
        std::fs::File::open(&args[4])?,
    ];
    let bytes = n.checked_mul(4).ok_or("linear file length overflow")? as u64;
    if files
        .iter()
        .any(|f| f.metadata().map_or(true, |m| m.len() != bytes))
    {
        return Err("linear file length does not match dimensions".into());
    }
    let a = read_linear(&mut files[0], n)?;
    let b = read_linear(&mut files[1], n)?;
    let result = compute_linear(&a, &b, w, h)?;
    let mut out = BufWriter::new(std::fs::File::create_new(&args[6])?);
    write_map(&mut out, result.diffmap.as_ref().ok_or("missing map")?)?;
    out.flush()?;
    println!(
        "{{\"width\":{w},\"height\":{h},\"score\":{:.17},\"p3\":{:.17}}}",
        result.score, result.pnorm_3
    );
    Ok(())
}

#[cfg(test)]
mod linear_tests {
    use super::*;

    #[test]
    fn strided_linear_rows_preserve_extended_range_and_ignore_padding() {
        let data = [
            -0.01,
            1.02,
            0.5,
            0.25,
            0.0,
            1.0,
            f32::NAN,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ];
        let rows = ingress::EncodedRows::new(ingress::Samples::Linear(&data), 2, 2, 7, 3).unwrap();
        assert_eq!(rows.linear_strip(0, 2), [&data[..6], &data[7..]].concat());
        assert_eq!(
            rows.columns(1, 2).linear_strip(0, 2),
            [&data[3..6], &data[10..]].concat()
        );
        assert!(ingress::EncodedRows::new(ingress::Samples::Linear(&data), 2, 2, 6, 3).is_err());
    }

    #[test]
    fn same_linear_values_reproduce_encoded_candidate_map() {
        let (w, h) = (39, 37);
        let data: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 17 + i / 13) % 256) as u8)
            .collect();
        let mut dist = data.clone();
        for v in dist.iter_mut().step_by(7) {
            *v = v.saturating_add(3);
        }
        let a = ingress::EncodedRows::new(ingress::Samples::U8(&data), w, h, w * 3, 3).unwrap();
        let b = ingress::EncodedRows::new(ingress::Samples::U8(&dist), w, h, w * 3, 3).unwrap();
        let expected =
            row_psycho::compute_geometry(&a, &b, 128, 512, &ButteraugliParams::default()).unwrap();
        let actual = compute_linear(&a.linear_strip(0, h), &b.linear_strip(0, h), w, h).unwrap();
        assert_eq!(actual.score.to_bits(), expected.score.to_bits());
        let (a, b) = (actual.diffmap.unwrap(), expected.diffmap.unwrap());
        for y in 0..h {
            assert_eq!(a.row(y), b.row(y));
        }
    }

    #[test]
    fn linear_maps_support_strided_rows_across_strip_and_column_boundaries() {
        let (w, h) = (769, 263);
        let stride = w * 3 + 7;
        let a: Vec<f32> = (0..w * h * 3)
            .map(|i| ((i * 17 + i / 13) % 256) as f32 / 255.0 * 1.02 - 0.01)
            .collect();
        let b: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, v)| v + if i % 7 == 0 { 0.002 } else { 0.0 })
            .collect();
        let mut ap = vec![f32::NAN; h * stride];
        let mut bp = ap.clone();
        for y in 0..h {
            ap[y * stride..y * stride + w * 3].copy_from_slice(&a[y * w * 3..(y + 1) * w * 3]);
            bp[y * stride..y * stride + w * 3].copy_from_slice(&b[y * w * 3..(y + 1) * w * 3]);
        }
        let av = ingress::EncodedRows::new(ingress::Samples::Linear(&ap), w, h, stride, 3).unwrap();
        let bv = ingress::EncodedRows::new(ingress::Samples::Linear(&bp), w, h, stride, 3).unwrap();
        let strided =
            row_psycho::compute_geometry(&av, &bv, 128, 512, &ButteraugliParams::default())
                .unwrap();
        let packed = compute_linear(&a, &b, w, h).unwrap();
        assert_eq!(strided.score.to_bits(), packed.score.to_bits());
        let (a, b) = (strided.diffmap.unwrap(), packed.diffmap.unwrap());
        for y in 0..h {
            assert_eq!(a.row(y), b.row(y));
        }
    }

    #[test]
    fn server_protocol_returns_native_map_bits_and_maximum() {
        let (w, h) = (16, 17);
        let a = vec![0.5; w * h * 3];
        let mut b = a.clone();
        b[3 * 23] = 0.502;
        let expected = compute_linear(&a, &b, w, h).unwrap();
        let mut messages = Vec::new();
        for (tag, pixels) in [(b'R', &a), (b'D', &b)] {
            messages.push(tag);
            messages.extend_from_slice(&(w as u32).to_le_bytes());
            messages.extend_from_slice(&(h as u32).to_le_bytes());
            for p in pixels {
                messages.extend_from_slice(&p.to_le_bytes());
            }
        }
        let mut reply = Vec::new();
        serve(&mut &messages[..], &mut reply).unwrap();
        let start = reply.iter().position(|&b| b == b'\n').unwrap() + 1;
        assert_eq!(
            f64::from_le_bytes(reply[start..start + 8].try_into().unwrap()).to_bits(),
            expected.score.to_bits()
        );
        let mut expected_bytes = Vec::new();
        write_map(&mut expected_bytes, expected.diffmap.as_ref().unwrap()).unwrap();
        assert_eq!(&reply[start + 8..], expected_bytes);
    }

    #[test]
    fn server_rejects_truncated_messages_and_nonfinite_pixels() {
        let mut out = Vec::new();
        assert!(serve(&mut &b"R\x01\x00"[..], &mut out).is_err());
        assert!(read_linear(&mut &f32::NAN.to_le_bytes()[..], 1).is_err());
        assert!(geometry(usize::MAX, 2).is_err());
    }
}
