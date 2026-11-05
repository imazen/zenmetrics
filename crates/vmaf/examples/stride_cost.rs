//! Reproducible whole-call workload for perf/heaptrack; no in-kernel timers.
//! Arguments: model (v0/v1), width, height, calls. Emits the exact score bits.
use std::{env, hint::black_box};
use vmaf::{ModelVariant, VmafV0Scorer, VmafV0Variant, VmafV1Scorer, Yuv420Frame};
fn main() {
    let args: Vec<_> = env::args().collect();
    let model = args.get(1).expect("v0 or v1");
    let width: usize = args.get(2).expect("width").parse().unwrap();
    let height: usize = args.get(3).expect("height").parse().unwrap();
    let calls: usize = args.get(4).expect("calls").parse().unwrap();
    assert!(width >= 64 && height >= 64 && width % 2 == 0 && height % 2 == 0 && calls > 0);
    let y: Vec<_> = (0..width * height)
        .map(|i| (16 + ((i % width) * 3 + (i / width) * 5) % 220) as u16)
        .collect();
    let dis: Vec<_> = y
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if i % 7 == 0 {
                v.saturating_add(4).min(255)
            } else {
                *v
            }
        })
        .collect();
    let chroma = vec![128; width * height / 4];
    let reference = [Yuv420Frame {
        y: &y,
        u: &chroma,
        v: &chroma,
        strides: [width, width / 2, width / 2],
    }];
    let distorted = [Yuv420Frame {
        y: &dis,
        u: &chroma,
        v: &chroma,
        strides: [width, width / 2, width / 2],
    }];
    let mut bits = None;
    let mut observe = |value: f64| {
        if let Some(previous) = bits {
            assert_eq!(previous, value.to_bits());
        }
        bits = Some(value.to_bits());
    };
    match model.as_str() {
        "v1" => {
            let scorer = VmafV1Scorer::new(width, height, 8, ModelVariant::Standard1080p).unwrap();
            for _ in 0..calls {
                observe(
                    black_box(
                        scorer
                            .score(black_box(&reference), black_box(&distorted))
                            .unwrap(),
                    )[0]
                    .score,
                );
            }
        }
        "v0" => {
            let scorer = VmafV0Scorer::new(width, height, 8, VmafV0Variant::Standard).unwrap();
            for _ in 0..calls {
                observe(
                    black_box(
                        scorer
                            .score(black_box(&reference), black_box(&distorted))
                            .unwrap(),
                    )[0]
                    .score,
                );
            }
        }
        _ => panic!("model must be v0 or v1"),
    }
    println!("score_bits={:016x} calls={calls}", bits.unwrap());
}
