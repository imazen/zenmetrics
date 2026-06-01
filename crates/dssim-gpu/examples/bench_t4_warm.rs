use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use dssim_gpu::Dssim;
use std::time::Instant;
fn main() {
    let w: u32 = 4000;
    let h: u32 = 3000;
    let n = (w as usize) * (h as usize) * 3;
    let r: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let d: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();
    let client = CudaRuntime::client(&Default::default());
    let mut s = Dssim::<CudaRuntime>::new(client, w, h).expect("new");
    s.set_reference(&r).expect("ref");
    for _ in 0..2 {
        let _ = s.compute(&r, &d).expect("warmup");
    }
    eprintln!("dssim 12 MP timing:");
    for i in 0..5 {
        let t = Instant::now();
        let res = s.compute(&r, &d).expect("compute");
        eprintln!("  iter {i}: {:?}  score={:.6}", t.elapsed(), res.score);
    }
}
