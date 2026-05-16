use cvvdp_gpu::estimate_gpu_memory_bytes;
fn main() {
    println!("estimate_gpu_memory_bytes table:");
    for &(w, h) in &[
        (64u32, 64u32),
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
    println!("\nPARALLEL caps on common GPU sizes (1.5x safety):");
    for &(mem_gb, mem_name) in &[
        (8.0_f64, "8 GB (RTX 3070)"),
        (12.0, "12 GB (RTX 3060)"),
        (16.0, "16 GB (RTX 4070 Ti S)"),
        (24.0, "24 GB (RTX 3090/4090)"),
    ] {
        print!("  {mem_name:<24} ");
        for &(w, h) in &[(256u32, 256u32), (1024, 1024), (2048, 2048), (4096, 3072)] {
            let est = estimate_gpu_memory_bytes(w, h).unwrap();
            let p = (mem_gb * 1e9 / (1.5 * est as f64)).floor() as u32;
            print!("{w}²={p:>3}  ");
        }
        println!();
    }
}
