extern crate alloc;
#[path = "../src/lib.rs"]
#[allow(dead_code, unreachable_pub)]
mod port;
use port::pipeline_debug;

fn main() {
    let p = std::env::args().nth(1).unwrap();
    let img = image::open(&p).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let mut y = Vec::with_capacity((w * h) as usize);
    for px in img.pixels() {
        let m01 = px[0] as f32 * 0.299 + px[1] as f32 * 0.587;
        y.push((px[2] as f32).mul_add(0.114, m01).round_ties_even());
    }
    let (f, s, d, din, score) = pipeline_debug(&y, w as usize, h as usize);
    for (i, v) in f.iter().enumerate() {
        println!("f{i:2} {:08x} {v:.9e}", v.to_bits());
    }
    for (i, v) in s.iter().enumerate() {
        println!("s{i:2} {:08x} {v:.9e}", v.to_bits());
    }
    for (i, v) in d.iter().enumerate() {
        println!("d{i:3} {:08x} {v:.9e}", v.to_bits());
    }
    for (i, v) in din.iter().enumerate() {
        println!("q{i:6} {:08x}", v.to_bits());
    }
    for (i, v) in d.iter().enumerate() {
        let k = libm::expf(-(v * 0.05));
        println!("k{i:3} {:08x} {k:.9e}", k.to_bits());
    }
    println!("SCORE {score:.9}");
}
