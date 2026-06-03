//! Decode an UltraHDR JPEG → HDR, re-encode at low JPEG quality to produce a
//! *distorted* UltraHDR JPEG. Used to give the HDR-metric smoke test a
//! discrimination data point on the UltraHDR-JPEG decode path (original vs
//! low-quality re-encode should score well below identity).
//!
//! Usage: make_distorted <in.jpg> <out.jpg> [base_q] [gainmap_q]

use std::path::PathBuf;

use ultrahdr_rs::{Decoder, Encoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let inp = PathBuf::from(
        args.next()
            .ok_or("usage: make_distorted <in.jpg> <out.jpg> [base_q] [gainmap_q]")?,
    );
    let out = PathBuf::from(args.next().ok_or("missing <out.jpg>")?);
    let base_q: u8 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(20);
    let gain_q: u8 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(20);

    let bytes = std::fs::read(&inp)?;
    let dec = Decoder::new(&bytes).map_err(|e| format!("{e:?}"))?;
    if !dec.is_ultrahdr() {
        return Err("input is not a (Google-format) UltraHDR JPEG".into());
    }
    let hdr = dec.decode_hdr(4.0).map_err(|e| format!("{e:?}"))?;

    let mut enc = Encoder::new();
    enc.set_hdr_image(hdr).set_quality(base_q, gain_q);
    let encoded = enc.encode().map_err(|e| format!("{e:?}"))?;
    std::fs::write(&out, &encoded)?;
    eprintln!(
        "wrote {} ({} bytes, base_q={base_q} gainmap_q={gain_q})",
        out.display(),
        encoded.len()
    );
    Ok(())
}
