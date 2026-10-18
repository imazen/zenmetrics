//! Prepared-pair repetition adapter; decode/tiling are outside the hot loop.
//! COMPLEXITY_PAIR_ROOT/<WxH>/{reference,distorted}.rgb comes from the existing
//! cpu-profile complexity-reps fixture export. RAYON_NUM_THREADS sets workers.
use fsim::{Downscale, Fsim};
use std::hint::black_box;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert_eq!(a.len(), 4, "shape metric reps");
    let (w, h) = a[1].split_once('x').unwrap();
    let (w, h): (usize, usize) = (w.parse().unwrap(), h.parse().unwrap());
    let factor: u32 = a[2]
        .strip_prefix("fsim-y-k")
        .expect("luma metric")
        .parse()
        .unwrap();
    assert!((2..=4).contains(&factor));
    let dir =
        std::path::PathBuf::from(std::env::var("COMPLEXITY_PAIR_ROOT").expect("fixture root"))
            .join(&a[1]);
    let r = std::fs::read(dir.join("reference.rgb")).unwrap();
    let d = std::fs::read(dir.join("distorted.rgb")).unwrap();
    let scorer = Fsim::with_downscale(Downscale::Factor(
        core::num::NonZeroU32::new(factor).unwrap(),
    ));
    let reps: usize = a[3].parse().unwrap();
    let mut last = f64::NAN;
    for _ in 0..reps {
        last = black_box(
            scorer
                .score_luma8(black_box(&r), black_box(&d), w, h, w * 3)
                .unwrap(),
        );
    }
    eprintln!("{} {} reps={} score={}", a[2], a[1], reps, last);
}
