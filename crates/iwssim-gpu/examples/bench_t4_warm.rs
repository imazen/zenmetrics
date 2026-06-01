use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use iwssim_gpu::Iwssim;
use std::time::Instant;
fn main() {
    let w: u32 = 4000;
    let h: u32 = 3000;
    let n = (w as usize) * (h as usize) * 3;
    let r: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let d: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();
    let client = CudaRuntime::client(&Default::default());
    let mut s = Iwssim::<CudaRuntime>::new(client, w, h).expect("new");
    for _ in 0..2 {
        let _ = s.compute_rgb(&r, &d).expect("warmup");
    }
    eprintln!("iwssim 12 MP timing:");
    for i in 0..5 {
        let t = Instant::now();
        let res = s.compute_rgb(&r, &d).expect("compute");
        eprintln!("  iter {i}: {:?}  score={:.6}", t.elapsed(), res.score);
    }
}
