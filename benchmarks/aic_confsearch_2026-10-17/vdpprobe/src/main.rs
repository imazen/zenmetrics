//! AIC HDR-VDP ingress probe: score a pairs TSV with the in-tree hdrvdp (v2 = official 2.2.2
//! port, v3 = 3.0.7 port) under a configurable display model.
//! usage: vdpprobe <pairs.tsv> <out.tsv> [key=value ...]
//! keys: eotf=srgb|pq|g22 peak black amb rho ppd task=quality|flicker|side-by-side
//!       bits=10|0 v=2|3|23 lmax
use std::collections::HashMap;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut c: HashMap<String, String> = [
        ("eotf", "srgb"), ("peak", "200"), ("black", "0.2"), ("amb", "250"), ("rho", "0.005"),
        ("ppd", "37.84"), ("task", "quality"), ("emis", "default"), ("age", "24"), ("surround", "none"), ("bsc", "0"), ("aesl", "1"), ("masking", "1"), ("enc2", "rgb"), ("enc", "rgb709"), ("bits", "10"), ("v", "23"), ("lmax", "10000"),
    ].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    for kv in &a[3..] {
        let (k, v) = kv.split_once('=').expect("key=value");
        assert!(c.contains_key(k), "unknown key {k}");
        c.insert(k.into(), v.into());
    }
    let f = |k: &str| c[k].parse::<f64>().unwrap();
    let (peak, black, lmax, ppd) = (f("peak"), f("black"), f("lmax"), f("ppd"));
    let refl = f("rho") * f("amb") / std::f64::consts::PI;
    let bits: u32 = c["bits"].parse().unwrap();
    let eotf = c["eotf"].clone();
    let lum = |path: &str| -> (Vec<f64>, usize, usize) {
        let img = image::open(path).expect(path).to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let q = if bits > 0 { ((1u32 << bits) - 1) as f64 } else { 0.0 };
        let px = img.into_raw().into_iter().map(|b| {
            let mut v = b as f64 / 255.0;
            if bits > 0 { v = (v * q).round() / q; }
            let l = match eotf.as_str() {
                "srgb" => {
                    let lin = if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
                    (peak - black) * lin + black
                }
                "g22" => (peak - black) * v.powf(2.2) + black,
                "pq" => hdrvdp::v3::ingress::pq_to_linear(&[v])[0],
                e => panic!("eotf {e}"),
            };
            (l + refl).min(lmax + refl)
        }).collect();
        (px, w, h)
    };
    let task = match c["task"].as_str() {
        "flicker" => hdrvdp::v3::Task::Flicker,
        "side-by-side" => hdrvdp::v3::Task::SideBySide,
        _ => hdrvdp::v3::Task::Quality,
    };
    // ingress colour space: the nits feed is linear BT.709; optionally converted (D65)
    const M_XYZ: [[f64; 3]; 3] = [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]];
    const M_2020: [[f64; 3]; 3] = [[0.627403895934699, 0.329283038377884, 0.043313065687417], [0.069097289358232, 0.919540395075459, 0.011362315566309], [0.016391438875150, 0.088013307877226, 0.895595253247624]];
    let encs = c["enc"].clone();
    let conv = move |v: Vec<f64>| -> Vec<f64> {
        let m = match encs.as_str() { "xyz" => M_XYZ, "rgb2020" => M_2020, _ => return v };
        v.chunks_exact(3).flat_map(|p| (0..3).map(move |i| m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2])).collect()
    };
    let enc = match c["enc"].as_str() { "xyz" => hdrvdp::v3::InputEncoding::Xyz, "rgb2020" => hdrvdp::v3::InputEncoding::RgbBt2020, _ => hdrvdp::v3::InputEncoding::RgbBt709 };
    use hdrvdp::v3::{DisplayPreset as D, Emission as E, Surround as S};
    let emis = match c["emis"].as_str() {
        "default" => E::default_for(enc),
        "ccfl-lcd" => E::Preset(D::CcflLcd), "crt" => E::Preset(D::Crt), "led-lcd" => E::Preset(D::LedLcd),
        "led-lcd-srgb" => E::Preset(D::LedLcdSrgb), "led-lcd-wcg" => E::Preset(D::LedLcdWcg),
        "oled" => E::Preset(D::Oled), "d65" => E::Preset(D::D65), e => panic!("emis {e}"),
    };
    let sur = match c["surround"].as_str() { "none" => S::None, "mean" => S::Mean, v => S::Uniform(v.parse().unwrap()) };
    let mut opts = hdrvdp::v3::Options::reference(task);
    opts.sensitivity_correction_delta = f("bsc");
    opts.do_aesl = c["aesl"] == "1";
    opts.do_masking = c["masking"] == "1";
    let p3 = hdrvdp::v3::Params::new(
        task,
        hdrvdp::v3::ViewingConditions::new(ppd, sur, c["age"].parse().unwrap()),
        enc, emis, opts,
    ).unwrap();
    let p2 = hdrvdp::Params::new(ppd);
    let pairs = std::fs::read_to_string(&a[1]).unwrap();
    let mut out = std::fs::File::create(&a[2]).unwrap();
    writeln!(out, "distorted\tv2_q\tv2_qmos\tv3_jod\tv3_q").unwrap();
    for line in pairs.lines().skip(1) {
        let fld: Vec<&str> = line.split('\t').collect();
        let (r0, w, h) = lum(fld[0]);
        let (t0, _, _) = lum(fld[1]);
        let (r, t) = (conv(r0.clone()), conv(t0.clone()));
        let (mut q2, mut m2, mut j3, mut q3) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        if c["v"].contains('2') {
            let res = if c["enc2"] == "lum" {
                // linear BT.709 luminance of the nits feed, as one channel
                let y = |v: &[f64]| -> Vec<f64> { v.chunks_exact(3).map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).collect() };
                hdrvdp::hdrvdp(&y(&t), &y(&r), w, h, hdrvdp::ColorEncoding::Luminance, &p2).unwrap()
            } else if c["enc"] == "xyz" {
                hdrvdp::hdrvdp(&t, &r, w, h, hdrvdp::ColorEncoding::Xyz, &p2).unwrap()
            } else {
                hdrvdp::hdrvdp(&t0, &r0, w, h, hdrvdp::ColorEncoding::RgbBt709, &p2).unwrap()
            };
            q2 = res.q; m2 = res.q_mos;
        }
        if c["v"].contains('3') { let r3 = hdrvdp::v3::hdrvdp3(&t, &r, w, h, &p3).unwrap(); j3 = r3.q_jod; q3 = r3.q; }
        writeln!(out, "{}\t{q2:?}\t{m2:?}\t{j3:?}\t{q3:?}", fld[2]).unwrap();
        out.flush().unwrap();
    }
}
