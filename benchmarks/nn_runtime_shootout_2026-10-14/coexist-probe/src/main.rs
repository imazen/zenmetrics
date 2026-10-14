// Probe: burn 0.22 (upstream cubecl 0.11) and zenmetrics' ssim2-gpu (zenforks-cubecl 0.10.2)
// in ONE process, interleaving work on the same GPU. Host data crosses between them, never types.
use burn::tensor::{Device, DeviceKind, Distribution, Tensor};
use cubecl::Runtime;
use ssim2_gpu::Ssim2;

fn pair(w: usize, h: usize, mag: i32) -> (Vec<u8>, Vec<u8>) {
    let mut a = vec![0u8; w * h * 3];
    let mut b = vec![0u8; w * h * 3];
    for y in 0..h { for x in 0..w { for c in 0..3 {
        let i = (y * w + x) * 3 + c;
        let v = ((x * 7 + y * 3 + c * 50) & 0xff) as i32;
        a[i] = v as u8;
        let p = if ((x / 8) ^ (y / 8)) & 1 == 0 { mag } else { -mag };
        b[i] = (v + p).clamp(0, 255) as u8;
    }}}
    (a, b)
}

fn burn_work(dev: &Device) -> f32 {
    let a = Tensor::<2>::random([512, 512], Distribution::Default, dev);
    let s = a.clone().matmul(a).sum();
    s.into_data().to_vec::<f32>().unwrap()[0]
}

fn main() {
    let (a, b) = pair(256, 256, 12);
    let which = std::env::args().nth(1).unwrap_or("cuda".into());
    let dev = match which.as_str() { "cuda" => Device::cuda(0), _ => Device::vulkan(DeviceKind::DiscreteGpu(0)) };
    let mut scores = vec![];
    if which == "cuda" {
        let client = cubecl::cuda::CudaRuntime::client(&Default::default());
        let mut s = Ssim2::<cubecl::cuda::CudaRuntime>::new(client, 256, 256).expect("new");
        for i in 0..5 {
            let bsum = burn_work(&dev);
            let sc = s.compute(&a, &b).expect("compute").score;
            println!("cuda round {i}: burn matmul-sum={bsum:.4e} ssim2={sc:.6}");
            scores.push(sc);
        }
    } else {
        let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
        let mut s = Ssim2::<cubecl::wgpu::WgpuRuntime>::new(client, 256, 256).expect("new");
        for i in 0..5 {
            let bsum = burn_work(&dev);
            let sc = s.compute(&a, &b).expect("compute").score;
            println!("wgpu round {i}: burn matmul-sum={bsum:.4e} ssim2={sc:.6}");
            scores.push(sc);
        }
    }
    println!("ssim2 identical across rounds: {}", scores.windows(2).all(|w| w[0] == w[1]));
}
