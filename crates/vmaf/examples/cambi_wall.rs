use std::{hint::black_box, time::Instant};
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let reps: usize = a[3].parse().unwrap();
    let r: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i * 7 + i / w * 13) ^ (i % 251)) as u8)
        .collect();
    let d: Vec<u8> = r
        .iter()
        .enumerate()
        .map(|(i, &v)| v.wrapping_add((i % 11) as u8))
        .collect();
    let y: Vec<u16> = d
        .chunks_exact(3)
        .map(|p| 16 + (u16::from(p[0]) * 219 / 255))
        .collect();
    let start = Instant::now();
    let mut score = 0.0f64;
    for _ in 0..reps {
        score = black_box(vmaf::cambi_from_luma(&y, w, h, w, 8).unwrap());
    }
    println!(
        "{w}x{h} {reps} reps: {:.6} ms/call score {score:.17} bits {:016x}",
        start.elapsed().as_secs_f64() * 1000.0 / reps as f64,
        score.to_bits()
    );
}
