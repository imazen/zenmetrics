//! Debug dump: band coeff/mask lines for the first measured line, to
//! bisect against the reference's `DUMP=1` stderr output.

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

fn read_pnm(path: &str) -> (Vec<u8>, usize, usize) {
    let mut f = std::fs::File::open(path).unwrap();
    let mut data = Vec::new();
    f.read_to_end(&mut data).unwrap();
    let mut pos = 0usize;
    let magic = next_token(&data, &mut pos);
    assert_eq!(magic.as_slice(), b"P6");
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
    assert_eq!(maxv, 255);
    pos += 1;
    (data[pos..pos + w * h * 3].to_vec(), w, h)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (ra, w, h) = read_pnm(&args[1]);
    let (rb, _, _) = read_pnm(&args[2]);
    mdctpsnr::dump_debug(&ra, &rb, w, h);
}
