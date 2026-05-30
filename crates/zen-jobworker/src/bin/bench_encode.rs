#![forbid(unsafe_code)]
//! CPU-bound throughput meter for the encode workload — used to compare cost/result across box
//! types (ARM vs x86, core counts). Loads an image once, then jpeg-encodes it as fast as possible
//! across N threads for D seconds and prints encodes/sec. No I/O in the hot loop → measures the
//! box's CPU on the actual encode kernel, normalized later by the box's €/hr.
//!
//! Usage: bench_encode <image> [threads=all cores] [secs=5]

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use image::codecs::jpeg::JpegEncoder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "/tmp/zensrc/source.png".to_string());
    let threads: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let secs: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);

    let img = Arc::new(
        image::ImageReader::open(&path)
            .expect("open image")
            .with_guessed_format()
            .expect("guess format")
            .decode()
            .expect("decode image"),
    );
    let (w, h) = (img.width(), img.height());
    let count = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));

    let start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..threads {
        let img = img.clone();
        let count = count.clone();
        let stop = stop.clone();
        handles.push(std::thread::spawn(move || {
            let mut local = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let mut out = Vec::new();
                JpegEncoder::new_with_quality(&mut out, 80).encode_image(&*img).expect("encode");
                black_box(&out);
                local += 1;
            }
            count.fetch_add(local, Ordering::Relaxed);
        }));
    }
    std::thread::sleep(Duration::from_secs(secs));
    stop.store(true, Ordering::Relaxed);
    for hnd in handles {
        let _ = hnd.join();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let total = count.load(Ordering::Relaxed);
    println!(
        "image={w}x{h} threads={threads} secs={elapsed:.2} encodes={total} encodes_per_sec={:.1}",
        total as f64 / elapsed
    );
}
