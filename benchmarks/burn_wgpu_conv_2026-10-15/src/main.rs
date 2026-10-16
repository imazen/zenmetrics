//! 3x3 conv on burn 0.22: default conv2d (autotune) vs explicit im2col + matmul,
//! CUDA vs Vulkan, f32 and f16, at the VGG16 layer shapes LPIPS uses.
use burn::tensor::module::{conv2d, unfold4d};
use burn::tensor::ops::{ConvOptions, UnfoldOptions};
use burn::tensor::{Device, DeviceKind, Distribution, FloatDType, Tensor};
use std::time::Instant;

fn sync(t: Tensor<4>) -> f32 {
    t.sum().into_data().convert::<f32>().to_vec::<f32>().unwrap()[0]
}

fn conv_im2col(x: Tensor<4>, w: Tensor<4>) -> Tensor<4> {
    let [n, _c, h, wd] = x.dims();
    let [co, ci, kh, kw] = w.dims();
    let cols = unfold4d(x, [kh, kw], UnfoldOptions::new([1, 1], [1, 1], [1, 1])); // [n, ci*9, h*w]
    let wm = w.reshape([1, co, ci * kh * kw]);
    wm.matmul(cols).reshape([n, co, h, wd])
}

fn time(iters: usize, mut f: impl FnMut() -> f32) -> (f64, f32) {
    let mut v = 0.0;
    for _ in 0..3 { v = f(); }
    let mut ts = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        v = f();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(f64::total_cmp);
    (ts[ts.len() / 2], v)
}

fn main() {
    let shapes: [(usize, usize, usize); 6] = [(3, 64, 256), (64, 64, 256), (128, 128, 128), (256, 256, 64), (512, 512, 32), (512, 512, 16)];
    for dev_name in ["cuda", "vulkan"] {
        let device = match dev_name {
            "cuda" => Device::cuda(0),
            _ => Device::vulkan(DeviceKind::DiscreteGpu(0)),
        };
        for dt in ["f32", "f16"] {
            for &(ci, co, hw) in &shapes {
                let mut x = Tensor::<4>::random([1, ci, hw, hw], Distribution::Default, &device);
                let mut w = Tensor::<4>::random([co, ci, 3, 3], Distribution::Default, &device);
                if dt == "f16" {
                    x = x.cast(FloatDType::F16);
                    w = w.cast(FloatDType::F16);
                }
                let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
                let (td, vd) = time(20, || sync(conv2d(x.clone(), w.clone(), None, opts.clone())));
                let (ti, vi) = time(20, || sync(conv_im2col(x.clone(), w.clone())));
                let gflop = 2.0 * (ci * 9 * co * hw * hw) as f64 / 1e9;
                println!(
                    "{dev_name}\t{dt}\t{ci}->{co}@{hw}\tconv2d {td:.3} ms ({:.0} GFLOP/s)\tim2col+mm {ti:.3} ms ({:.0} GFLOP/s)\tsum {vd:.4e} vs {vi:.4e}",
                    gflop / td * 1e3,
                    gflop / ti * 1e3
                );
            }
        }
    }
}
