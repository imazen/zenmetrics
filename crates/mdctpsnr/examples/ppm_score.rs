//! `ppm_score <ref.ppm> <dist.ppm>` — score two binary P6/P5 PNM files
//! with `mdct_psnr_srgb8`, for side-by-side checks against the reference
//! `dctpsnr` binary.

use std::io::Read;

fn next_token(d: &[u8], pos: &mut usize) -> Vec<u8> {
    loop {
        while *pos < d.len() && (d[*pos] as char).is_ascii_whitespace() {
            *pos += 1;
        }
        if *pos < d.len() && d[*pos] == b'#' {
            while *pos < d.len() && d[*pos] != b'\n' {
                *pos += 1;
            }
            continue;
        }
        break;
    }
    let start = *pos;
    while *pos < d.len() && !(d[*pos] as char).is_ascii_whitespace() {
        *pos += 1;
    }
    d[start..*pos].to_vec()
}

fn read_pnm(path: &str) -> (Vec<u8>, usize, usize, usize) {
    let mut f = std::fs::File::open(path).unwrap();
    let mut data = Vec::new();
    f.read_to_end(&mut data).unwrap();
    let mut pos = 0usize;
    let magic = next_token(&data, &mut pos);
    let comps = match magic.as_slice() {
        b"P6" => 3,
        b"P5" => 1,
        _ => panic!("not a binary PNM"),
    };
    let w: usize = String::from_utf8(next_token(&data, &mut pos))
        .unwrap()
        .parse()
        .unwrap();
    let h: usize = String::from_utf8(next_token(&data, &mut pos))
        .unwrap()
        .parse()
        .unwrap();
    let maxv: usize = String::from_utf8(next_token(&data, &mut pos))
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(maxv, 255, "8-bit only");
    pos += 1; // single whitespace after maxval
    let px = data[pos..pos + w * h * comps].to_vec();
    (px, w, h, comps)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (ra, w1, h1, c1) = read_pnm(&args[1]);
    let (rb, w2, h2, c2) = read_pnm(&args[2]);
    assert_eq!((w1, h1, c1), (w2, h2, c2));
    assert_eq!(c1, 3, "P6 color only");
    if std::env::var_os("DUMP").is_some() {
        mdctpsnr::dump_debug(&ra, &rb, w1, h1);
        return;
    }
    let score = mdctpsnr::mdct_psnr_srgb8(&ra, &rb, w1, h1).unwrap();
    println!("{score:.9}");
}
