//! Raw RGB driver for AIC metric oracle checks; run from the pinned snapshot.
use clap::ValueEnum;
use zenmetrics_cli::{
    decode::Rgb8Image,
    metrics::{GpuRuntime, MetricKind, run_metric},
};
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let kind = MetricKind::from_str(&a[1], false).unwrap();
    let w: u32 = a[2].parse().unwrap();
    let h: u32 = a[3].parse().unwrap();
    let r = Rgb8Image {
        pixels: std::fs::read(&a[4]).unwrap(),
        width: w,
        height: h,
    };
    let d = Rgb8Image {
        pixels: std::fs::read(&a[5]).unwrap(),
        width: w,
        height: h,
    };
    for (name, score) in run_metric(kind, &r, &d, GpuRuntime::Auto).unwrap() {
        println!("{name}\t{score:.17}");
    }
}
