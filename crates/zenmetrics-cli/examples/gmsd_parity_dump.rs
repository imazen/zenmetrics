#![forbid(unsafe_code)]
//! GMSD port parity harness, Rust half. For each `ref<TAB>dist` pair in a
//! TSV (header row skipped; the first two columns are read):
//!
//! 1. decode both images through the CLI's zen-codec decode path,
//! 2. convert to BT.601 8-bit luma with `gmsd::rgb8_to_gray` (libgmsd's
//!    command-line conversion),
//! 3. crop to even width/height — libgmsd's `downsample_2x2` writes past its
//!    allocation on odd input, so only even sizes are a valid comparison,
//! 4. write `<out>/<i>.ref.f32`, `<i>.dist.f32` (packed little-endian f32),
//!    `<i>.rust_map.f32`, and one line `i  w  h  rust_gmsd  ref  dist` to
//!    `<out>/rust.tsv`.
//!
//! The C half (`gmsd_raw` built from libgmsd, outside this repo) reads the
//! same raw planes; the comparison is done by the parity script recorded in
//! `benchmarks/gmsd_parity_2026-09-22.md`.
//!
//! Usage: `gmsd_parity_dump <pairs.tsv> <out-dir> [max-pairs]`

use std::io::Write;
use std::path::Path;

fn write_f32(path: &Path, v: &[f32]) {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write raw plane");
}

fn gray_even(path: &str) -> (Vec<f32>, usize, usize) {
    let img = zenmetrics_cli::decode::decode_image_to_rgb8(Path::new(path))
        .unwrap_or_else(|e| panic!("decode {path}: {e}"));
    let (w, h) = (img.width as usize, img.height as usize);
    let mut g = vec![0.0f32; w * h];
    gmsd::rgb8_to_gray(&img.pixels, w, h, w * 3, &mut g).expect("gray");
    let (we, he) = (w & !1, h & !1);
    let mut out = Vec::with_capacity(we * he);
    for y in 0..he {
        out.extend_from_slice(&g[y * w..y * w + we]);
    }
    (out, we, he)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pairs = args.next().expect("pairs.tsv");
    let out = std::path::PathBuf::from(args.next().expect("out dir"));
    let max: usize = args
        .next()
        .map(|s| s.parse().expect("max"))
        .unwrap_or(usize::MAX);
    std::fs::create_dir_all(&out).expect("mkdir");
    let text = std::fs::read_to_string(&pairs).expect("read pairs");
    let mut log = std::fs::File::create(out.join("rust.tsv")).expect("rust.tsv");
    writeln!(log, "i\tw\th\trust_gmsd\tref\tdist").unwrap();
    for (i, line) in text.lines().skip(1).take(max).enumerate() {
        let mut cols = line.split('\t');
        let (r, d) = (cols.next().expect("ref"), cols.next().expect("dist"));
        let (rg, w, h) = gray_even(r);
        let (dg, w2, h2) = gray_even(d);
        assert_eq!((w, h), (w2, h2), "pair {i}: size mismatch");
        let (mw, mh) = gmsd::map_dims(w, h);
        let mut map = vec![0.0f32; mw * mh];
        let s = gmsd::gmsd_with_map(
            gmsd::GrayImage::packed(&rg, w, h).unwrap(),
            gmsd::GrayImage::packed(&dg, w, h).unwrap(),
            &mut map,
        )
        .expect("gmsd");
        write_f32(&out.join(format!("{i}.ref.f32")), &rg);
        write_f32(&out.join(format!("{i}.dist.f32")), &dg);
        write_f32(&out.join(format!("{i}.rust_map.f32")), &map);
        writeln!(log, "{i}\t{w}\t{h}\t{:.17e}\t{r}\t{d}", s.gmsd).unwrap();
    }
}
