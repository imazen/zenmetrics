//! Per-stage wall-clock timing for the pure-Rust scoring path — no libvmaf
//! FFI. Used to measure aarch64 NEON coverage: each stage times the public
//! entry point, so a NEON kernel landing inside `adm3_v1_from_luma` shows up
//! as a delta on the `adm` row.
//!
//! Run: `cargo bench -p vmaf --features simd --bench arm_stages`

use std::hint::black_box;
use std::time::Instant;
use vmaf::{
    ModelVariant, VmafV0Variant, Yuv420Frame, adm3_v1_from_luma, cambi_v1_from_luma,
    motion3_from_luma, score_v1_420, speed_v1_chroma_420, vif_v0_from_luma,
};

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const FRAMES: usize = 2;
const ROUNDS: usize = 5;

struct Frame {
    planes: [Vec<u16>; 3],
}

fn fixture(index: usize, distorted: bool) -> Frame {
    let mut planes = [
        vec![0; WIDTH * HEIGHT],
        vec![0; WIDTH * HEIGHT / 4],
        vec![0; WIDTH * HEIGHT / 4],
    ];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = 16 + ((x * 3 + y * 5 + index * 7 + (x / 17) * 9) % 220);
            planes[0][y * WIDTH + x] = if distorted && x < WIDTH / 2 {
                (value / 12 * 12) as u16
            } else {
                value as u16
            };
        }
    }
    for y in 0..HEIGHT / 2 {
        for x in 0..WIDTH / 2 {
            let slot = y * WIDTH / 2 + x;
            let u = 16 + ((x * 7 + y * 3 + (x / 9) * (y / 9) * 13 + index * 11) % 224);
            let v = 16 + ((x * 2 + y * 9 + (x / 13) * (y / 7) * 5 + index * 13) % 224);
            planes[1][slot] = if distorted {
                (u + (x + y + index) % 17).min(240) as u16
            } else {
                u as u16
            };
            planes[2][slot] = if distorted {
                v.saturating_sub((x + 2 * y + index) % 19) as u16
            } else {
                v as u16
            };
        }
    }
    Frame { planes }
}

fn main() {
    let reference: Vec<Frame> = (0..FRAMES).map(|i| fixture(i, false)).collect();
    let distorted: Vec<Frame> = (0..FRAMES).map(|i| fixture(i, true)).collect();
    let ref_y: Vec<&[u16]> = reference.iter().map(|f| f.planes[0].as_slice()).collect();
    let dis_y: Vec<&[u16]> = distorted.iter().map(|f| f.planes[0].as_slice()).collect();
    let ref_yuv: Vec<Yuv420Frame> = reference
        .iter()
        .map(|f| Yuv420Frame {
            y: &f.planes[0],
            u: &f.planes[1],
            v: &f.planes[2],
        })
        .collect();
    let dis_yuv: Vec<Yuv420Frame> = distorted
        .iter()
        .map(|f| Yuv420Frame {
            y: &f.planes[0],
            u: &f.planes[1],
            v: &f.planes[2],
        })
        .collect();

    println!("stage\tmedian_ms\tmin_ms\trounds");
    let report = |name: &str, mut f: Box<dyn FnMut()>| {
        let mut times = Vec::with_capacity(ROUNDS);
        f(); // warm
        for _ in 0..ROUNDS {
            let start = Instant::now();
            f();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{name}\t{:.3}\t{:.3}\t{ROUNDS}",
            times[ROUNDS / 2],
            times[0]
        );
    };

    report(
        "motion",
        Box::new(|| {
            black_box(motion3_from_luma(&ref_y, WIDTH, HEIGHT, 8, false).unwrap());
        }),
    );
    report(
        "cambi",
        Box::new(|| {
            for f in &dis_y {
                black_box(cambi_v1_from_luma(f, WIDTH, HEIGHT, 8).unwrap());
            }
        }),
    );
    report(
        "speed",
        Box::new(|| {
            for i in 0..FRAMES {
                black_box(
                    speed_v1_chroma_420(
                        ref_yuv[i].u,
                        ref_yuv[i].v,
                        dis_yuv[i].u,
                        dis_yuv[i].v,
                        WIDTH,
                        HEIGHT,
                        8,
                        ModelVariant::Standard1080p,
                    )
                    .unwrap(),
                );
            }
        }),
    );
    report(
        "adm",
        Box::new(|| {
            for i in 0..FRAMES {
                black_box(
                    adm3_v1_from_luma(
                        ref_y[i],
                        dis_y[i],
                        WIDTH,
                        HEIGHT,
                        8,
                        ModelVariant::Standard1080p,
                    )
                    .unwrap(),
                );
            }
        }),
    );
    report(
        "vif",
        Box::new(|| {
            for i in 0..FRAMES {
                black_box(
                    vif_v0_from_luma(
                        ref_y[i],
                        dis_y[i],
                        WIDTH,
                        HEIGHT,
                        8,
                        VmafV0Variant::Standard,
                    )
                    .unwrap(),
                );
            }
        }),
    );
    report(
        "vmaf_v1",
        Box::new(|| {
            black_box(
                score_v1_420(
                    &ref_yuv,
                    &dis_yuv,
                    WIDTH,
                    HEIGHT,
                    8,
                    ModelVariant::Standard1080p,
                )
                .unwrap(),
            );
        }),
    );
}
