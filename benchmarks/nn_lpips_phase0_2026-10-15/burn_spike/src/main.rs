//! Spike: LPIPS-vgg trunk hand-written in burn 0.22, weights from the raw dump,
//! LPIPS head on device. Parity vs upstream goldens and timing on CUDA / Vulkan.
//! usage: burnlpips <spike_dir> <goldens.tsv> <device cuda|vulkan> <dtype f32|f16> <sizes> <reps>
use burn::tensor::activation::relu;
use burn::tensor::module::{conv2d, max_pool2d};
use burn::tensor::ops::{ConvOptions, MaxPoolOptions};
use burn::tensor::{Device, DeviceKind, FloatDType, Tensor, TensorData};
use std::time::Instant;

const CHANNELS: [usize; 5] = [64, 128, 256, 512, 512];
// conv layer count per VGG16 stage (taps after the last conv of each stage)
const STAGES: [usize; 5] = [2, 2, 3, 3, 3];

struct Vgg {
    shift: Tensor<4>,
    scale: Tensor<4>,
    convs: Vec<(Tensor<4>, Tensor<1>)>,
    lin: Vec<Tensor<4>>,
}

impl Vgg {
    fn load(path: &str, device: &Device, f16: bool) -> Self {
        let raw = std::fs::read(path).unwrap();
        let all: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let mut at = 0usize;
        let mut take = |n: usize| {
            let v = all[at..at + n].to_vec();
            at += n;
            v
        };
        let cast = |t: Tensor<4>| if f16 { t.cast(FloatDType::F16) } else { t };
        let shift = cast(Tensor::<4>::from_data(TensorData::new(take(3), [1, 3, 1, 1]), device));
        let scale = cast(Tensor::<4>::from_data(TensorData::new(take(3), [1, 3, 1, 1]), device));
        let mut convs = Vec::new();
        let mut cin = 3;
        for (s, &n) in STAGES.iter().enumerate() {
            for _ in 0..n {
                let co = CHANNELS[s];
                let w = cast(Tensor::<4>::from_data(TensorData::new(take(co * cin * 9), [co, cin, 3, 3]), device));
                let b = Tensor::<1>::from_data(TensorData::new(take(co), [co]), device);
                let b = if f16 { b.cast(FloatDType::F16) } else { b };
                convs.push((w, b));
                cin = co;
            }
        }
        // lin stays f32: the head runs in f32 regardless.
        let lin = CHANNELS.iter().map(|&c| Tensor::<4>::from_data(TensorData::new(take(c), [1, c, 1, 1]), device)).collect();
        assert_eq!(at, all.len());
        Vgg { shift, scale, convs, lin }
    }

    fn taps(&self, x: Tensor<4>) -> Vec<Tensor<4>> {
        let mut x = (x - self.shift.clone()) / self.scale.clone();
        let mut out = Vec::new();
        let mut i = 0;
        for (s, &n) in STAGES.iter().enumerate() {
            if s > 0 {
                let mut o = MaxPoolOptions::new([2, 2]);
                o.stride = [2, 2];
                x = max_pool2d(x, o);
            }
            for _ in 0..n {
                let (w, b) = &self.convs[i];
                x = relu(conv2d(x, w.clone(), Some(b.clone()), ConvOptions::new([1, 1], [1, 1], [1, 1], 1)));
                i += 1;
            }
            out.push(x.clone());
        }
        out
    }

    /// LPIPS: per tap unit-normalise over channels, squared diff, lin-weighted
    /// channel sum, spatial mean; sum over taps. Head in f32.
    fn lpips(&self, r: Tensor<4>, d: Tensor<4>) -> f32 {
        let rt = self.taps(r);
        let dt = self.taps(d);
        let mut total: Option<Tensor<1>> = None;
        for ((a, b), lin) in rt.into_iter().zip(dt).zip(&self.lin) {
            let (a, b) = (a.cast(FloatDType::F32), b.cast(FloatDType::F32));
            let na = a.clone() / (a.clone().powi_scalar(2).sum_dim(1).sqrt() + 1e-10);
            let nb = b.clone() / (b.clone().powi_scalar(2).sum_dim(1).sqrt() + 1e-10);
            let diff = (na - nb).powi_scalar(2);
            let v = (diff * lin.clone()).sum_dim(1).mean();
            total = Some(match total { Some(t) => t + v, None => v });
        }
        total.unwrap().into_data().convert::<f32>().to_vec::<f32>().unwrap()[0]
    }
}

fn load_rgb(path: &str, n: usize, device: &Device, f16: bool) -> Tensor<4> {
    let raw = std::fs::read(path).unwrap();
    assert_eq!(raw.len(), n * n * 3);
    let mut planar = vec![0f32; 3 * n * n];
    for (i, px) in raw.chunks_exact(3).enumerate() {
        for c in 0..3 {
            planar[c * n * n + i] = (f32::from(px[c]) / 255.0) * 2.0 - 1.0;
        }
    }
    let t = Tensor::<4>::from_data(TensorData::new(planar, [1, 3, n, n]), device);
    if f16 { t.cast(FloatDType::F16) } else { t }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, goldens, dev, dt) = (&a[1], &a[2], a[3].as_str(), a[4].as_str());
    let sizes: Vec<usize> = a[5].split(',').map(|s| s.parse().unwrap()).collect();
    let reps: usize = a[6].parse().unwrap();
    let device = match dev {
        "cuda" => Device::cuda(0),
        _ => Device::vulkan(DeviceKind::DiscreteGpu(0)),
    };
    let f16 = dt == "f16";
    let m = Vgg::load(&format!("{dir}/lpips_vgg_all.f32"), &device, f16);
    let gold = std::fs::read_to_string(goldens).unwrap();
    for line in gold.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let (stem, n, want): (&str, usize, f64) = (f[0], f[1].parse().unwrap(), f[2].parse().unwrap());
        if !sizes.contains(&n) {
            continue;
        }
        let r = load_rgb(&format!("{dir}/{stem}_{n}_ref.rgb"), n, &device, f16);
        let d = load_rgb(&format!("{dir}/{stem}_{n}_dist.rgb"), n, &device, f16);
        let mut s = m.lpips(r.clone(), d.clone());
        let mut ts = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            s = m.lpips(r.clone(), d.clone());
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(f64::total_cmp);
        let med = if ts.is_empty() { f64::NAN } else { ts[ts.len() / 2] };
        println!("{dev}\t{dt}\t{stem}\t{n}\tgot {s:.9}\twant {want:.9}\t|d| {:.3e}\tmedian {med:.3} ms", (f64::from(s) - want).abs());
    }
}
