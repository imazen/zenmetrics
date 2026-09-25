//! `batch_score <pairs.tsv>` — score many P6 PNM pairs in parallel threads.
//! Each line: `name<TAB>ref.ppm<TAB>dst.ppm`. Prints `name<TAB>score`.

use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    let arg = std::env::args().nth(1).unwrap();
    let pairs: Vec<(String, String, String)> = std::fs::read_to_string(&arg)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut it = l.split('\t');
            (
                it.next().unwrap().to_string(),
                it.next().unwrap().to_string(),
                it.next().unwrap().to_string(),
            )
        })
        .collect();
    let n = pairs.len();
    let next = Arc::new(AtomicUsize::new(0));
    let out: Vec<AtomicUsize> = Vec::new();
    let _ = out;
    let results: Vec<std::sync::Mutex<Option<f32>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    let pairs = Arc::new(pairs);
    let results = Arc::new(results);
    let threads = std::env::var("JOBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let mut handles = Vec::new();
    for _ in 0..threads {
        let pairs = Arc::clone(&pairs);
        let results = Arc::clone(&results);
        let next = Arc::clone(&next);
        handles.push(std::thread::spawn(move || {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= pairs.len() {
                    break;
                }
                let (name, rp, dp) = &pairs[i];
                let (ra, w, h) = read_pnm(rp);
                let (rb, w2, h2) = read_pnm(dp);
                assert_eq!((w, h), (w2, h2), "{name}");
                let score = mdctpsnr::mdct_psnr_srgb8(&ra, &rb, w, h).unwrap();
                *results[i].lock().unwrap() = Some(score);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    for (i, (name, _, _)) in pairs.iter().enumerate() {
        println!("{}\t{:.9}", name, results[i].lock().unwrap().unwrap());
    }
}
