//! Quick measurement: actual GPU memory used by Strip vs Full mode at
//! 4096×4096 across all 3 regimes. Polls nvidia-smi at construction +
//! after first compute, computes the delta vs baseline.
//!
//! Run as one of:
//!     cargo run --release -p zensim-gpu --example strip_measure_actual --features cuda -- full basic
//!     cargo run --release -p zensim-gpu --example strip_measure_actual --features cuda -- strip basic
//!     cargo run --release -p zensim-gpu --example strip_measure_actual --features cuda -- strip iw

use cubecl::Runtime;
use std::process::Command;
use std::time::Instant;
use zensim_gpu::{Zensim, ZensimFeatureRegime};

type Backend = cubecl::cuda::CudaRuntime;

fn nvidia_smi_used() -> u64 {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next().unwrap().trim().parse::<u64>().unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_else(|| "full".into());
    let regime_str = args.get(2).cloned().unwrap_or_else(|| "basic".into());
    let regime = match regime_str.as_str() {
        "ext" | "extended" => ZensimFeatureRegime::Extended,
        "iw" | "withiw" => ZensimFeatureRegime::WithIw,
        _ => ZensimFeatureRegime::Basic,
    };
    let w_str = args.get(3).cloned().unwrap_or_else(|| "4096".into());
    let h_str = args.get(4).cloned().unwrap_or_else(|| "4096".into());
    let w: u32 = w_str.parse().unwrap();
    let h: u32 = h_str.parse().unwrap();

    let pixels = (w as usize) * (h as usize);
    let ref_img: Vec<u8> = (0..pixels * 3).map(|i| (i & 0xFF) as u8).collect();
    let dist_img: Vec<u8> = (0..pixels * 3).map(|i| ((i ^ 0xA5) & 0xFF) as u8).collect();

    eprintln!("# mode={mode} regime={regime:?} w={w} h={h}");
    let baseline = nvidia_smi_used();
    eprintln!("baseline_used_mb={baseline}");

    let client = Backend::client(&Default::default());
    let t0 = Instant::now();
    let mut z = match mode.as_str() {
        "strip" => Zensim::<Backend>::new_strip_with_halo_and_regime(client, w, h, 256, 40, regime)
            .unwrap(),
        _ => Zensim::<Backend>::new_with_regime(client, w, h, regime).unwrap(),
    };
    let ctor_ms = t0.elapsed().as_millis();
    let after_ctor = nvidia_smi_used();
    eprintln!("after_ctor_used_mb={after_ctor} ctor_ms={ctor_ms}");

    let t1 = Instant::now();
    let _features = z.compute_features_vec(&ref_img, &dist_img).unwrap();
    let cf_ms = t1.elapsed().as_millis();
    let after_compute = nvidia_smi_used();
    eprintln!("after_compute_used_mb={after_compute} compute_ms={cf_ms}");

    let peak_delta_mb = after_compute.saturating_sub(baseline);
    eprintln!("peak_delta_mb={peak_delta_mb}");
    println!("{mode},{regime:?},{w}x{h},peak_delta_mb={peak_delta_mb},compute_ms={cf_ms}");
}
