//! Dump per-scale wmcs for two raw f32 gray planes.
//! Usage: dump_scales W H ref.f32 dis.f32

use iwssim::Iwssim;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let w: u32 = args[1].parse().unwrap();
    let h: u32 = args[2].parse().unwrap();
    let read = |p: &str| -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let r = read(&args[3]);
    let d = read(&args[4]);
    let mut m = Iwssim::new(w, h).unwrap();
    let s = m.score_gray(&r, &d).unwrap();
    for (i, v) in s.per_scale.iter().enumerate() {
        println!("scale {}: wmcs={:.9}", i + 1, v);
    }
    println!("score={:.9}", s.score);
}
