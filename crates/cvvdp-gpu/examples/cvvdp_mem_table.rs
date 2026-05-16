//! Print `estimate_gpu_memory_bytes` + `recommend_parallel` for
//! common image sizes × common GPU memory sizes.
//!
//! Useful for sizing PARALLEL workers on a sweep deploy. Run with:
//!
//! ```text
//! cargo run --release --example cvvdp_mem_table -p cvvdp-gpu --no-default-features
//! ```

use cvvdp_gpu::{estimate_gpu_memory_bytes, recommend_parallel};

fn main() {
    println!("estimate_gpu_memory_bytes table:");
    for &(w, h) in &[
        (64_u32, 64_u32),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 3072),
    ] {
        if let Some(b) = estimate_gpu_memory_bytes(w, h) {
            println!("  {w:>4}×{h:<4}  {b:>15} bytes  {:>8.1} MB", b as f64 / 1e6);
        }
    }

    // Routes through the same `recommend_parallel` helper that
    // production callers use, including the PARALLEL_SAFETY_FACTOR
    // constant — no inline duplication of the math.
    println!("\nPARALLEL caps (recommend_parallel, 1.5× safety):");
    for &(mem_gb, mem_name) in &[
        (8.0_f64, "8 GB (RTX 3070)"),
        (12.0, "12 GB (RTX 3060)"),
        (16.0, "16 GB (RTX 4070 Ti S)"),
        (24.0, "24 GB (RTX 3090/4090)"),
    ] {
        let mem_bytes = (mem_gb * 1e9) as u64;
        print!("  {mem_name:<24} ");
        for &(w, h) in &[(256_u32, 256_u32), (1024, 1024), (2048, 2048), (4096, 3072)] {
            let p = recommend_parallel(mem_bytes, w, h);
            print!("{w}²={p:>3}  ");
        }
        println!();
    }
}
