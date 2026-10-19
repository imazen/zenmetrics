use brisque::Brisque;
use std::hint::black_box;
use std::time::Instant;
use za::zenpixels::{PixelDescriptor, PixelSlice};
use za::{EvalOpts, Image, NoReferenceMetric};
use zenmetrics_abstractions as za;

fn mirror(i: usize, n: usize) -> usize {
    let m = i % (2 * n);
    if m < n { m } else { 2 * n - 1 - m }
}

fn tile(src: &[u8], sw: usize, sh: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        let sy = mirror(y, sh);
        for x in 0..w {
            let si = (sy * sw + mirror(x, sw)) * 3;
            let di = (y * w + x) * 3;
            out[di..di + 3].copy_from_slice(&src[si..si + 3]);
        }
    }
    out
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let reps: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "7".into())
        .parse()
        .unwrap();
    let shape = std::env::args().nth(3); // optional WxH mirror-tile
    let img = image::open(&path).unwrap().to_rgb8();
    let (mut w, mut h) = (img.width() as usize, img.height() as usize);
    let owned;
    let px: &[u8] = if let Some(s) = shape {
        let (nw, nh) = s
            .split_once('x')
            .map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()))
            .unwrap();
        owned = tile(&img, w, h, nw, nh);
        w = nw;
        h = nh;
        &owned
    } else {
        &img
    };
    let slice = PixelSlice::new(px, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let image = Image::from(slice);
    let mut times = Vec::new();
    let mut score = f64::NAN;
    for _ in 0..reps {
        let t = Instant::now();
        score = match Brisque::new().evaluate_image(black_box(image.clone()), EvalOpts::new()) {
            Ok(e) => e.score(),
            Err(e) => {
                eprintln!("ERR {:?}", e.error().kind());
                return;
            }
        };
        times.push(t.elapsed());
    }
    times.sort();
    eprintln!(
        "{w}x{h} reps={reps} median={:.3?} min={:.3?} score={score}",
        times[times.len() / 2],
        times[0]
    );
}
