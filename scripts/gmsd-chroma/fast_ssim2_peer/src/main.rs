//! Frozen TRAIN RGB8 pair scoring through fast-ssim2's public CPU API.
//! TSV: pair_key, width, height, ref_rgb, dist_rgb. RGB is packed sRGB u8,
//! row-major, three channels, stride = width pixels. No labels are consumed.
use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::io::{BufWriter, Write};

fn read_rgb(path: &str, width: usize, height: usize) -> Result<imgref::ImgVec<[u8; 3]>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let expected = width.checked_mul(height).and_then(|n| n.checked_mul(3)).ok_or("dimension overflow")?;
    if width == 0 || height == 0 || bytes.len() != expected {
        return Err("invalid packed RGB8 dimensions or length".into());
    }
    let pixels = bytes.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
    Ok(imgref::ImgVec::new(pixels, width, height))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 { return Err("usage: peer <raw_pairs.tsv> <new_scores.tsv>".into()); }
    let input = fs::read_to_string(&args[0])?;
    let mut lines = input.lines();
    if lines.next() != Some("pair_key\twidth\theight\tref_rgb\tdist_rgb") {
        return Err("unexpected raw-pair TSV schema".into());
    }
    let mut out = BufWriter::new(fs::OpenOptions::new().write(true).create_new(true).open(&args[1])?);
    writeln!(out, "pair_key\tfast_ssim2")?;
    let mut seen = HashSet::new();
    for (index, line) in lines.enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 5 || !seen.insert(fields[0]) { return Err("invalid or duplicate pair row".into()); }
        let (width, height) = (fields[1].parse()?, fields[2].parse()?);
        let reference = read_rgb(fields[3], width, height)?;
        let distorted = read_rgb(fields[4], width, height)?;
        let score = fast_ssim2::compute_ssimulacra2(reference.as_ref(), distorted.as_ref())?;
        if !score.is_finite() { return Err("non-finite fast-ssim2 score".into()); }
        writeln!(out, "{}\t{score:.17e}", fields[0])?;
        if (index + 1) % 100 == 0 { eprintln!("peer_rows {}", index + 1); }
    }
    out.flush()?;
    println!("peer_rows {}", seen.len());
    Ok(())
}
