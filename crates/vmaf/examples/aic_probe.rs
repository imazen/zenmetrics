// Probe: score a raw-RGB pair through VmafV0Scorer (studio-swing BT.601 Y,
// neutral chroma — v0 features are luma-only).
use std::env;
use std::fs;
use vmaf::{VmafV0Scorer, VmafV0Variant, Yuv420Frame};

fn to_yuv420(
    rgb: &[u8],
    stride: usize,
    w: usize,
    h: usize,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut y = Vec::with_capacity(w * h);
    for row in rgb.chunks_exact(stride * 3).take(h) {
        for px in row[..w * 3].chunks_exact(3) {
            let (r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64);
            let yy = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0;
            y.push(yy.round().clamp(0.0, 255.0) as u16);
        }
    }
    let n = (w / 2) * (h / 2);
    (y, vec![128u16; n], vec![128u16; n])
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (w_src, h_src): (usize, usize) = (args[3].parse().unwrap(), args[4].parse().unwrap());
    let (w, h) = (w_src & !1, h_src & !1);
    let rgb_r = fs::read(&args[1]).unwrap();
    let rgb_d = fs::read(&args[2]).unwrap();
    assert_eq!(rgb_r.len(), w_src * h_src * 3);
    assert_eq!(rgb_d.len(), w_src * h_src * 3);
    let (yr, ur, vr) = to_yuv420(&rgb_r, w_src, w, h);
    let (yd, ud, vd) = to_yuv420(&rgb_d, w_src, w, h);
    let fr = Yuv420Frame {
        y: &yr,
        u: &ur,
        v: &vr,
    };
    let fd = Yuv420Frame {
        y: &yd,
        u: &ud,
        v: &vd,
    };
    for variant in [VmafV0Variant::Standard, VmafV0Variant::StandardNeg] {
        let scorer = VmafV0Scorer::new(w, h, 8, variant).unwrap();
        let out = scorer.score(&[fr], &[fd]).unwrap();
        let f = &out[0].features;
        println!(
            "{:?}: score={:.6} adm2={:.6} vif=[{:.6} {:.6} {:.6} {:.6}] mean_vif={:.6} motion2={:.6}",
            variant,
            out[0].score,
            f.adm2,
            f.vif_scales[0],
            f.vif_scales[1],
            f.vif_scales[2],
            f.vif_scales[3],
            f.vif_scales.iter().sum::<f64>() / 4.0,
            f.motion2
        );
    }
}
