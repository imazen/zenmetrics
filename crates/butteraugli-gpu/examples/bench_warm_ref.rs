use butteraugli_gpu::Butteraugli;
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use std::time::Instant;

fn main() {
    let w: u32 = 4000;
    let h: u32 = 3000;
    let n = (w as usize) * (h as usize) * 3;
    let r: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let d: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();
    let client = CudaRuntime::client(&Default::default());
    let mut b = Butteraugli::<CudaRuntime>::new(client, w, h);
    b.set_reference(&r).expect("ref");
    // warmup
    for _ in 0..2 {
        let _ = b.compute_with_reference(&d).expect("warmup");
    }
    eprintln!("butter 12 MP warm-ref timing:");
    for i in 0..5 {
        let t = Instant::now();
        let res = b.compute_with_reference(&d).expect("compute");
        eprintln!(
            "  iter {i}: {:?}  score={:.6} pnorm3={:.6}",
            t.elapsed(),
            res.score,
            res.pnorm_3
        );
    }
}
