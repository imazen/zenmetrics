//! Allocation profiler for cvvdp using a counting global
//! allocator wrapper. Counts (alloc, free) calls + total bytes for
//! a single `Cvvdp::score` invocation and a single
//! `score_with_warm_ref` invocation at each measurement size.
//!
//! Run:
//! ```bash
//! cargo run -p cvvdp --release --example alloc_profile -- --output benchmarks/cvvdp_cpu_alloc_profile.tsv
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use cvvdp::{Cvvdp, CvvdpParams};

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREES.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

const SIZES: &[(u32, u32)] = &[(256, 256), (512, 512), (1024, 1024)];

fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let n = (w as usize) * (h as usize);
    let mut out = vec![0u8; n * 3];
    let mut s = seed;
    for i in 0..n * 3 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        out[i] = (s >> 16) as u8;
    }
    out
}

fn snap() -> (u64, u64, u64) {
    (
        ALLOCS.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

fn run_one(w: u32, h: u32, writer: &mut File) -> std::io::Result<()> {
    let r = make_image(w, h, 1234);
    let d = make_image(w, h, 9876);

    // Setup outside measurement (Cvvdp::new owns the scratch).
    let mut cv = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
    // Warmup so pool/JIT effects settle.
    let _ = cv.score(&r, &d).unwrap();
    let _ = cv.score(&r, &d).unwrap();

    // COLD path: one fresh score call.
    let (a0, f0, b0) = snap();
    let _ = cv.score(&r, &d).unwrap();
    let (a1, f1, b1) = snap();
    writeln!(
        writer,
        "{w}\t{h}\t{}\tcold\t{}\t{}\t{}",
        (w as u64) * (h as u64),
        a1 - a0,
        f1 - f0,
        b1 - b0
    )?;
    eprintln!(
        "  {w}x{h} cold: allocs={} frees={} bytes={}",
        a1 - a0,
        f1 - f0,
        b1 - b0
    );

    // WARM path: warm_reference then score_with_warm_ref.
    cv.warm_reference(&r).unwrap();
    let _ = cv.score_with_warm_ref(&d).unwrap();

    let (a2, f2, b2) = snap();
    let _ = cv.score_with_warm_ref(&d).unwrap();
    let (a3, f3, b3) = snap();
    writeln!(
        writer,
        "{w}\t{h}\t{}\twarm\t{}\t{}\t{}",
        (w as u64) * (h as u64),
        a3 - a2,
        f3 - f2,
        b3 - b2
    )?;
    eprintln!(
        "  {w}x{h} warm: allocs={} frees={} bytes={}",
        a3 - a2,
        f3 - f2,
        b3 - b2
    );

    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut output_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            output_path = Some(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }
    let path = output_path.unwrap_or_else(|| "benchmarks/cvvdp_cpu_alloc_profile.tsv".to_string());

    let parent = std::path::Path::new(&path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent).ok();
    let mut out = File::create(&path)?;
    writeln!(out, "size_w\tsize_h\tpixels\tpath\tallocs\tfrees\tbytes")?;
    eprintln!("Writing alloc profile TSV to {path}");

    for &(w, h) in SIZES {
        run_one(w, h, &mut out)?;
    }
    eprintln!("done");
    Ok(())
}
