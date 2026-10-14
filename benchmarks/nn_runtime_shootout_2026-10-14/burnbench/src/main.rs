// Times burn-onnx generated ResNet50 / VGG16-LPIPS on flex (CPU), cuda, vulkan devices.
use burn::tensor::{Device, DeviceKind, Distribution, Tensor};
use std::time::Instant;
mod model {
    pub mod resnet50_224 { include!(concat!(env!("OUT_DIR"), "/model/resnet50_224.rs")); }
    pub mod vgg16lpips_256 { include!(concat!(env!("OUT_DIR"), "/model/vgg16lpips_256.rs")); }
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dev, net, hw, iters): (&str, &str, usize, usize) = (&a[1], &a[2], a[3].parse().unwrap(), a[4].parse().unwrap());
    let device = match dev {
        "flex" => Device::flex(),
        "cuda" => Device::cuda(0),
        "vulkan" => Device::vulkan(DeviceKind::DiscreteGpu(0)),
        _ => panic!("device"),
    };
    let fresh = || { let x = Tensor::<4>::random([1, 3, hw, hw], Distribution::Default, &device); let _ = x.to_data(); x };
    let mut checksum = 0f64;
    let mut times = vec![];
    match net {
        "resnet50" => {
            let m = model::resnet50_224::Model::from_file(concat!(env!("OUT_DIR"), "/model/resnet50_224.bpk"), &device);
            for i in 0..iters + 3 {
                let x = fresh();
                let t = Instant::now();
                let y = m.forward(x);
                let v = y.into_data().to_vec::<f32>().unwrap();
                if i >= 3 { times.push(t.elapsed().as_secs_f64() * 1e3); }
                checksum += v.iter().map(|v| *v as f64).sum::<f64>();
            }
        }
        _ => {
            let m = model::vgg16lpips_256::Model::from_file(concat!(env!("OUT_DIR"), "/model/vgg16lpips_256.bpk"), &device);
            for i in 0..iters + 3 {
                let t0 = Instant::now();
                let x = fresh();
                let t = Instant::now();
                let (a, b, c, d, e) = m.forward(x);
                // device-side scalar reduction per tap (what an LPIPS score does), tiny readback forces completion
                let s: Vec<f32> = [a, b, c, d, e].into_iter().map(|o| o.sum().into_data().to_vec::<f32>().unwrap()[0]).collect();
                if i >= 3 { times.push(t.elapsed().as_secs_f64() * 1e3); }
                checksum += s.iter().map(|v| *v as f64).sum::<f64>();
                let _ = t0;
            }
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("burn {dev}\t{net}_{hw}\tmin={:.2}ms\tmed={:.2}ms\tchecksum={checksum:.3e}", times[0], times[times.len() / 2]);
}
