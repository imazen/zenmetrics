//! Probe: which libvmaf v1 model produced the AIC2026 online `VMAF v1`
//! column? Scores each pair with `VmafV1Scorer` under every
//! `ModelVariant`, on studio-601 YUV420 planes (the AIC convention;
//! ingress choice moved the delta <0.02 median, so it is fixed here).
//!
//! Usage: v1_variant_probe <pairs.tsv> <image_dir>
//! pairs.tsv: `ref_path<TAB>dist_path<TAB>published_v1` lines.

use std::env;
use vmaf::{ModelVariant, VmafV1Scorer, Yuv420Frame};
use zenmetrics_cli::decode::decode_image_to_rgb8;

/// Studio-swing BT.601 YUV420, same math as
/// `metrics::vmaf::to_yuv420_even(.., Ingress::Studio601)`.
fn studio601_420(img: &zenmetrics_cli::decode::Rgb8Image) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let stride = img.width as usize;
    let (w, h) = (img.width as usize & !1, img.height as usize & !1);
    let (mut y, mut cb, mut cr) = (vec![0u8; w * h], vec![0u8; w * h / 4], vec![0u8; w * h / 4]);
    for by in 0..h / 2 {
        for bx in 0..w / 2 {
            let (mut u_sum, mut v_sum) = (0.0f64, 0.0f64);
            for dy in 0..2 {
                for dx in 0..2 {
                    let (row, col) = (2 * by + dy, 2 * bx + dx);
                    let p = (row * stride + col) * 3;
                    let (r, g, b) = (
                        f64::from(img.pixels[p]),
                        f64::from(img.pixels[p + 1]),
                        f64::from(img.pixels[p + 2]),
                    );
                    let lum = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0;
                    y[row * w + col] = lum.round().clamp(0.0, 255.0) as u8;
                    u_sum += (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0;
                    v_sum += (112.0 * r - 93.786 * g - 18.214 * b) / 255.0;
                }
            }
            cb[by * w / 2 + bx] = (128.0 + u_sum / 4.0).round().clamp(16.0, 240.0) as u8;
            cr[by * w / 2 + bx] = (128.0 + v_sum / 4.0).round().clamp(16.0, 240.0) as u8;
        }
    }
    (
        y.into_iter().map(u16::from).collect(),
        cb.into_iter().map(u16::from).collect(),
        cr.into_iter().map(u16::from).collect(),
    )
}

/// Full-swing BT.601 (JPEG convention) YUV444->420 2x2-mean chroma.
fn fullrange601_420(img: &zenmetrics_cli::decode::Rgb8Image) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let stride = img.width as usize;
    let (w, h) = (img.width as usize & !1, img.height as usize & !1);
    let (mut y, mut cb, mut cr) = (vec![0u8; w * h], vec![0u8; w * h / 4], vec![0u8; w * h / 4]);
    for by in 0..h / 2 {
        for bx in 0..w / 2 {
            let (mut u_sum, mut v_sum) = (0.0f64, 0.0f64);
            for dy in 0..2 {
                for dx in 0..2 {
                    let (row, col) = (2 * by + dy, 2 * bx + dx);
                    let p = (row * stride + col) * 3;
                    let (r, g, b) = (
                        f64::from(img.pixels[p]),
                        f64::from(img.pixels[p + 1]),
                        f64::from(img.pixels[p + 2]),
                    );
                    y[row * w + col] = (0.299 * r + 0.587 * g + 0.114 * b)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    u_sum += -0.168736 * r - 0.331264 * g + 0.5 * b;
                    v_sum += 0.5 * r - 0.418688 * g - 0.081312 * b;
                }
            }
            cb[by * w / 2 + bx] = (128.0 + u_sum / 4.0).round().clamp(0.0, 255.0) as u8;
            cr[by * w / 2 + bx] = (128.0 + v_sum / 4.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    (
        y.into_iter().map(u16::from).collect(),
        cb.into_iter().map(u16::from).collect(),
        cr.into_iter().map(u16::from).collect(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <pairs.tsv> <image_dir>", args[0]);
        std::process::exit(2);
    }
    let dir = std::path::Path::new(&args[2]);
    let variants = [
        ("3d0h", ModelVariant::Standard1080p),
        ("5d0h", ModelVariant::Phone),
        ("1d5h_2160", ModelVariant::Default4k),
        ("3d0h_2160", ModelVariant::Consumer4k),
        ("hfr_3d0h", ModelVariant::HfrStandard1080p),
        ("hfr_5d0h", ModelVariant::HfrPhone),
        ("hfr_1d5h_2160", ModelVariant::HfrDefault4k),
        ("hfr_3d0h_2160", ModelVariant::HfrConsumer4k),
    ];
    print!("distorted\tpublished");
    for (name, _) in &variants {
        if *name == "3d0h" {
            print!(
                "\t3d0h\tcambi\tspeed_uv\tadm3\tmotion3\t3d0h_neutraluv\tneutral_speed_uv\t3d0h_fullrange\t3d0h_bd10\t3d0h_m2f1\t3d0h_m2mean\t3d0h_m2dd\t3d0h_sub_adm2"
            );
        } else {
            print!("\t{name}");
        }
    }
    println!();
    for line in std::fs::read_to_string(&args[1])?.lines() {
        let mut f = line.split('\t');
        let (r, d, p) = (f.next().unwrap(), f.next().unwrap(), f.next().unwrap_or(""));
        let ri = decode_image_to_rgb8(&dir.join(r))?;
        let di = decode_image_to_rgb8(&dir.join(d))?;
        let (ry, ru, rv) = studio601_420(&ri);
        let (dy, du, dv) = studio601_420(&di);
        let neutral = vec![128u16; ru.len()];
        let model3d0h = vmaf::VmafModel::new(ModelVariant::Standard1080p).unwrap();
        let (fy, fu, fv) = fullrange601_420(&ri);
        let (gy, gu, gv) = fullrange601_420(&di);
        let (ry10, ru10, rv10) = (
            ry.iter().map(|v| v << 2).collect::<Vec<u16>>(),
            ru.iter().map(|v| v << 2).collect::<Vec<u16>>(),
            rv.iter().map(|v| v << 2).collect::<Vec<u16>>(),
        );
        let (dy10, du10, dv10) = (
            dy.iter().map(|v| v << 2).collect::<Vec<u16>>(),
            du.iter().map(|v| v << 2).collect::<Vec<u16>>(),
            dv.iter().map(|v| v << 2).collect::<Vec<u16>>(),
        );
        let (w, h) = (ri.width as usize & !1, ri.height as usize & !1);
        let adm2v = vmaf::adm2_v0_from_luma(&ry, &dy, w, h, 8, vmaf::VmafV0Variant::Standard)?;
        print!("{d}\t{p}");
        for (name, v) in &variants {
            let scorer = VmafV1Scorer::new(w, h, 8, *v)?;
            let rr = [Yuv420Frame {
                y: &ry,
                u: &ru,
                v: &rv,
            }];
            let dd = [Yuv420Frame {
                y: &dy,
                u: &du,
                v: &dv,
            }];
            let res = &scorer.score(&rr, &dd)?[0];
            if name == &"3d0h" {
                let (nres, nfeat) = {
                    let rr = [Yuv420Frame {
                        y: &ry,
                        u: &neutral,
                        v: &neutral,
                    }];
                    let dd = [Yuv420Frame {
                        y: &dy,
                        u: &neutral,
                        v: &neutral,
                    }];
                    let r0 = &scorer.score(&rr, &dd)?[0];
                    (r0.score, r0.features.speed_chroma_uv)
                };
                let fr_score = {
                    let rr = [Yuv420Frame {
                        y: &fy,
                        u: &fu,
                        v: &fv,
                    }];
                    let dd = [Yuv420Frame {
                        y: &gy,
                        u: &gu,
                        v: &gv,
                    }];
                    scorer.score(&rr, &dd)?[0].score
                };
                let bd10_score = {
                    let s10 = VmafV1Scorer::new(w, h, 10, ModelVariant::Standard1080p)?;
                    let rr = [Yuv420Frame {
                        y: &ry10,
                        u: &ru10,
                        v: &rv10,
                    }];
                    let dd = [Yuv420Frame {
                        y: &dy10,
                        u: &du10,
                        v: &dv10,
                    }];
                    s10.score(&rr, &dd)?[0].score
                };
                // 2-frame sequences: ref=[r,r], dist=[r,d] — frame1 gets
                // real motion = SAD(dist,ref); also try dist=[d,d].
                let (m_f1, m_mean, m2) = {
                    let rr = [
                        Yuv420Frame {
                            y: &ry,
                            u: &ru,
                            v: &rv,
                        },
                        Yuv420Frame {
                            y: &ry,
                            u: &ru,
                            v: &rv,
                        },
                    ];
                    let dd = [
                        Yuv420Frame {
                            y: &ry,
                            u: &ru,
                            v: &rv,
                        },
                        Yuv420Frame {
                            y: &dy,
                            u: &du,
                            v: &dv,
                        },
                    ];
                    let s = scorer.score(&rr, &dd)?;
                    let dd2 = [
                        Yuv420Frame {
                            y: &dy,
                            u: &du,
                            v: &dv,
                        },
                        Yuv420Frame {
                            y: &dy,
                            u: &du,
                            v: &dv,
                        },
                    ];
                    let s2 = scorer.score(&rr, &dd2)?;
                    (s[1].score, (s[0].score + s[1].score) / 2.0, s2[0].score)
                };
                let f = &res.features;
                print!(
                    "\t{}\t{}\t{}\t{}",
                    res.score, f.cambi, f.speed_chroma_uv, f.adm3
                );
                let sub_adm2 = model3d0h.predict(vmaf::VmafFeatures {
                    cambi: f.cambi,
                    speed_chroma_uv: f.speed_chroma_uv,
                    adm3: adm2v,
                    motion3: 0.0,
                })?;
                print!(
                    "\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    f.motion3, nres, nfeat, fr_score, bd10_score, m_f1, m_mean, m2, sub_adm2
                );
            } else {
                print!("\t{}", res.score);
            }
        }
        println!();
    }
    Ok(())
}
