// Probe: VIFp on candidate luma/channel conventions for AIC-4 column ID.
use std::env;
use std::fs;
use vif::vif_plane_f32;

fn luma(
    rgb: &[u8],
    stride_px: usize,
    w: usize,
    h: usize,
    f: impl Fn(f32, f32, f32) -> f32,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for row in rgb.chunks_exact(stride_px * 3).take(h) {
        for px in row[..w * 3].chunks_exact(3) {
            out.push(f(px[0] as f32, px[1] as f32, px[2] as f32));
        }
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (w0, h): (usize, usize) = (args[3].parse().unwrap(), args[4].parse().unwrap());
    let w = w0; // metrics don't need even dims
    let rgb_r = fs::read(&args[1]).unwrap();
    let rgb_d = fs::read(&args[2]).unwrap();

    let run = |name: &str, fr: &[f32], fd: &[f32]| {
        let v = vif_plane_f32(fr, fd, w, h, w).unwrap();
        println!("{name:24} {v:.6}");
    };

    // studio-swing JPEG 601 Y (unrounded)
    run(
        "studio601-Y",
        &luma(&rgb_r, w0, w, h, |r, g, b| {
            16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
        }),
        &luma(&rgb_d, w0, w, h, |r, g, b| {
            16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
        }),
    );
    // full-range BT.601
    run(
        "full601-Y",
        &luma(&rgb_r, w0, w, h, |r, g, b| {
            0.299 * r + 0.587 * g + 0.114 * b
        }),
        &luma(&rgb_d, w0, w, h, |r, g, b| {
            0.299 * r + 0.587 * g + 0.114 * b
        }),
    );
    // per-channel mean VIF
    let mut acc = 0.0;
    for (ci, cn) in [(0usize, "R"), (1, "G"), (2, "B")] {
        let a = luma(&rgb_r, w0, w, h, |r, g, b| [r, g, b][ci]);
        let b = luma(&rgb_d, w0, w, h, |r, g, b| [r, g, b][ci]);
        let v = vif_plane_f32(&a, &b, w, h, w).unwrap();
        println!("{cn}-channel VIF          {v:.6}");
        acc += v;
    }
    println!("mean(RGB) VIF            {:.6}", acc / 3.0);
    // mean plane (R+G+B)/3
    run(
        "mean-plane",
        &luma(&rgb_r, w0, w, h, |r, g, b| (r + g + b) / 3.0),
        &luma(&rgb_d, w0, w, h, |r, g, b| (r + g + b) / 3.0),
    );
}
