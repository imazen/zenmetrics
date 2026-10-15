//! Internal numerical review driver, compiling the same private implementation.
extern crate alloc;
#[path = "../src/lib.rs"]
#[allow(dead_code, unreachable_pub)]
mod port;
use std::io::{BufRead, Write};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args[1] == "ratios" {
        let data = std::fs::read(&args[2]).unwrap();
        assert_eq!(data.len() % 24, 0);
        let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3]).unwrap());
        for row in data.chunks_exact(24) {
            let a: Vec<f32> = row
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            let (g, gi) = port::ggd_ratio_argmin(a[0], a[1]);
            let (r, ri) = port::aggd_ratio_argmin(a[2], a[3], a[4], a[5]);
            out.write_all(&g.to_le_bytes()).unwrap();
            out.write_all(&(gi as u32).to_le_bytes()).unwrap();
            out.write_all(&r.to_le_bytes()).unwrap();
            out.write_all(&(ri as u32).to_le_bytes()).unwrap();
        }
    } else {
        assert_eq!(args[1], "panel");
        for line in std::io::BufReader::new(std::fs::File::open(&args[2]).unwrap()).lines() {
            let line = line.unwrap();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let img = image::open(&line).unwrap().to_rgb8();
            let (w, h) = img.dimensions();
            let y: Vec<f32> = img
                .pixels()
                .map(|p| {
                    f32::from(p[2])
                        .mul_add(0.114, f32::from(p[0]) * 0.299 + f32::from(p[1]) * 0.587)
                        .round_ties_even()
                })
                .collect();
            let result =
                std::panic::catch_unwind(|| port::pipeline_debug(&y, w as usize, h as usize));
            match result {
                Ok((f, _, _, _, score)) => {
                    print!("{line}\t{score:.17}\t");
                    for v in f {
                        print!("{:08x},", v.to_bits());
                    }
                    println!();
                }
                Err(_) => println!("{line}\tNaN\tdegenerate"),
            }
        }
    }
}
