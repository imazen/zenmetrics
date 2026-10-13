//! Wall-time sanity for `psnrhvs_rgb8` on a synthetic pair.
//! `cargo run --release -p psnrhvs --features parallel --example wall [w h reps]`

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let w: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let h: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);

    let mut reference = vec![0u8; w * h * 3];
    let mut distorted = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let v = ((x * 7 + y * 13) ^ (x * y % 251)) as u8;
            reference[i * 3] = v;
            reference[i * 3 + 1] = v.wrapping_add(3);
            reference[i * 3 + 2] = v;
            distorted[i * 3] = v.wrapping_add(9);
            distorted[i * 3 + 1] = v;
            distorted[i * 3 + 2] = v.wrapping_add(5);
        }
    }

    let mut s = (0.0f64, 0.0f64);
    let t0 = Instant::now();
    for _ in 0..reps {
        let r = psnrhvs::psnrhvs_rgb8(&reference, &distorted, w, h, w * 3).unwrap();
        s = (r.psnr_hvs, r.psnr_hvs_m);
    }
    let dt = t0.elapsed();
    println!(
        "{w}x{h} {reps} reps: {:.2} ms/call (psnr_hvs {:.17} psnr_hvs_m {:.17}, bits {:016x}/{:016x})",
        dt.as_secs_f64() * 1e3 / reps as f64,
        s.0,
        s.1,
        s.0.to_bits(),
        s.1.to_bits()
    );

    // Daala variant: same planes as u8 YUV420 (cw×ch chroma).
    let (cw, ch) = (w / 2, h / 2);
    let mut y_ref = vec![0u8; w * h];
    let mut y_dis = vec![0u8; w * h];
    let mut c_ref = vec![0u8; cw * ch];
    let mut c_dis = vec![0u8; cw * ch];
    for (i, (r, d)) in y_ref.iter_mut().zip(y_dis.iter_mut()).enumerate() {
        *r = (i * 37 % 251) as u8;
        *d = (i * 53 % 241) as u8;
    }
    for (i, (r, d)) in c_ref.iter_mut().zip(c_dis.iter_mut()).enumerate() {
        *r = (i * 29 % 239) as u8;
        *d = (i * 41 % 233) as u8;
    }
    let t0 = Instant::now();
    let mut sd = 0.0f64;
    for _ in 0..reps {
        let r = psnrhvs::daala::psnr_hvs_daala_yuv420(
            &y_ref, &y_dis, &c_ref, &c_dis, &c_ref, &c_dis, w, h, w, cw,
        );
        sd = r.psnr_hvs;
    }
    let dt = t0.elapsed();
    println!(
        "daala yuv420 {w}x{h} {reps} reps: {:.2} ms/call (psnr_hvs {:.17}, bits {:016x})",
        dt.as_secs_f64() * 1e3 / reps as f64,
        sd,
        sd.to_bits()
    );
}
