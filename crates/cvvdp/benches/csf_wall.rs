//! Single-size end-to-end CSF optimization wall check. Texture is deterministic.
use cvvdp::{Cvvdp, CvvdpParams};
use zenbench::prelude::*;
fn bench(suite: &mut Suite) {
    let n = 1024 * 1024 * 3;
    let mut seed = 1234_u32;
    let reference: Vec<u8> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        })
        .collect();
    let mut seed = 9876_u32;
    let distorted: Vec<u8> = reference
        .iter()
        .map(|&v| {
            seed = seed.wrapping_mul(48271);
            (v as i32 + ((seed >> 24) as i32 - 128) / 8).clamp(0, 255) as u8
        })
        .collect();
    let mut cv = Cvvdp::new(1024, 1024, CvvdpParams::default()).unwrap();
    eprintln!("score={:.9}", cv.score(&reference, &distorted).unwrap());
    suite.group("cvvdp/1024x1024", move |g| {
        // Declare the internal Rayon pool so zenbench does not count its work
        // as background load. No extra scorer copies or external worker threads.
        let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "1".into());
        g.bench_tagged("score", &[("threads", &threads)], move |b| {
            b.iter(|| std::hint::black_box(cv.score(&reference, &distorted).unwrap()))
        });
    });
}
zenbench::main!(bench);
