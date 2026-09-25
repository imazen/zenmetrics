use iai_callgrind::{library_benchmark, library_benchmark_group, main};

#[library_benchmark]
fn score_128() -> f64 {
    let w = 128;
    let h = 128;
    let reference = (0..w * h * 3)
        .map(|i| ((i * 37) % 256) as u8)
        .collect::<Vec<_>>();
    let distorted = (0..w * h * 3)
        .map(|i| ((i * 37 + i % 7) % 256) as u8)
        .collect::<Vec<_>>();
    nlpd::score_rgb_u8(&reference, &distorted, w, h).unwrap()
}

library_benchmark_group!(name = nlpd_group; benchmarks = score_128);
main!(library_benchmark_groups = nlpd_group);
