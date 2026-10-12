//! Lean benchmark of the CLI's RGB SSIM source, using its RGB8 input layout.
//! Requires `parallel` for the classical module's Rayon dependency.
#![forbid(unsafe_code)]

// Same three fields as zenmetrics-cli/src/decode.rs. The scoring module is
// compiled directly below; no scoring logic is copied into this harness.
mod decode {
    pub(crate) struct Rgb8Image {
        pub(crate) pixels: Vec<u8>,
        pub(crate) width: u32,
        pub(crate) height: u32,
    }
}
#[path = "../../zenmetrics-cli/src/metrics/classical.rs"]
mod classical;

// The included module also tests an unrelated PSNR ingress. Supply its
// studio-601 fixture here; native CLI tests exercise the production ingress.
#[cfg(test)]
mod metrics {
    use super::decode::Rgb8Image;
    pub(crate) fn studio601_gray(img: &Rgb8Image) -> Rgb8Image {
        let w = (img.width & !1) as usize;
        let h = (img.height & !1) as usize;
        let mut pixels = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for px in img.pixels[y * img.width as usize * 3..][..w * 3]
                .as_chunks::<3>()
                .0
            {
                let luma = (16.0
                    + (65.481 * px[0] as f32 + 128.553 * px[1] as f32 + 24.966 * px[2] as f32)
                        / 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                pixels.extend_from_slice(&[luma; 3]);
            }
        }
        Rgb8Image {
            pixels,
            width: w as u32,
            height: h as u32,
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let w: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let h: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(2);
    let kind = match args.next().as_deref().unwrap_or("ssim") {
        "ssim" => classical::Kind::Ssim,
        "psnr" => classical::Kind::Psnr,
        "psnr-y" => classical::Kind::PsnrY,
        "psnr-y601" => classical::Kind::PsnrY601,
        "psnr-y-studio601" => classical::Kind::PsnrYStudio601,
        "psnr-y-libvmaf" => classical::Kind::PsnrYLibvmaf,
        other => panic!("unknown classical metric: {other}"),
    };
    let mut reference = vec![0; w * h * 3];
    let mut distorted = vec![0; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let v = ((x * 7 + y * 13) ^ (x * y % 251)) as u8;
            reference[i..i + 3].copy_from_slice(&[v, v.wrapping_add(3), v]);
            distorted[i..i + 3].copy_from_slice(&[v.wrapping_add(9), v, v.wrapping_add(5)]);
        }
    }
    let r = decode::Rgb8Image {
        pixels: reference,
        width: w as u32,
        height: h as u32,
    };
    let d = decode::Rgb8Image {
        pixels: distorted,
        width: w as u32,
        height: h as u32,
    };
    let mut score = 0.0;
    let start = std::time::Instant::now();
    for _ in 0..reps {
        score = classical::score(kind, &r, &d).unwrap();
    }
    println!(
        "{w}x{h} {reps} reps RGB SSIM: {:.1} ms/call (score {score:.17}, bits {:016x})",
        start.elapsed().as_secs_f64() * 1e3 / reps as f64,
        score.to_bits()
    );
}
