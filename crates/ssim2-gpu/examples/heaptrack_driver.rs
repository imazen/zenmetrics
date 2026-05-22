//! Heaptrack-friendly single 12 MP `Ssim2::compute` invocation.
//!
//! Use this binary to measure host RSS for the Phase 1 plane-aliasing
//! change. Two warmup calls (so heaptrack sees the steady-state
//! allocations, not the first-call kernel compile dance) then one
//! score call.
//!
//! ```sh
//! cargo build --release -p ssim2-gpu \
//!     --no-default-features --features cuda,fast-reduction,cubecl-types,pixels \
//!     --example heaptrack_driver
//! heaptrack ./target/release/examples/heaptrack_driver
//! ```

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use ssim2_gpu::Ssim2;

const W: u32 = 4000;
const H: u32 = 3000;

fn synthetic(w: usize, h: usize, mag: u8) -> (Vec<u8>, Vec<u8>) {
    let mut a = vec![0u8; w * h * 3];
    let mut b = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 220 / w.max(1)) & 0xff) as u8;
            let g = ((y * 220 / h.max(1)) & 0xff) as u8;
            let bb = (((x + y) * 200 / (w + h).max(1)) & 0xff) as u8;
            let i = (y * w + x) * 3;
            a[i] = r;
            a[i + 1] = g;
            a[i + 2] = bb;
            let bx = x / 8;
            let by = y / 8;
            let pert = if (bx ^ by) & 1 == 0 {
                mag as i32
            } else {
                -(mag as i32)
            };
            b[i] = (r as i32 + pert).clamp(0, 255) as u8;
            b[i + 1] = (g as i32 + pert).clamp(0, 255) as u8;
            b[i + 2] = (bb as i32 + pert).clamp(0, 255) as u8;
        }
    }
    (a, b)
}

fn main() {
    let (a, b) = synthetic(W as usize, H as usize, 6);

    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, W, H).expect("Ssim2::new");

    // Two warmup calls so heaptrack captures the steady-state allocation
    // profile, not the one-time JIT/kernel-compile work.
    let _ = s.compute(&a, &b).expect("warmup 1");
    let _ = s.compute(&a, &b).expect("warmup 2");
    let r = s.compute(&a, &b).expect("compute");

    println!("12MP score = {:.6}", r.score);
}
