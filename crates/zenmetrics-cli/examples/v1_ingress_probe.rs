//! Probe: does the AIC2026 online `VMAF v1` column equal libvmaf
//! `vmaf_v1.0.16_3d0h` under a different ingress? Scores each pair as
//! `Model::V1` under both `Ingress::Studio601` (the AIC BT.601
//! studio-swing convention used by every other VMAF column) and
//! `Ingress::Bt709` (what the `vmaf-v1` metric currently runs), and
//! prints both next to the published column for joining.
//!
//! Usage: v1_ingress_probe <pairs.tsv> <image_dir>
//! pairs.tsv: `ref_path<TAB>dist_path<TAB>published_v1` lines.

use std::env;
use zenmetrics_cli::decode::decode_image_to_rgb8;
use zenmetrics_cli::metrics::vmaf::{self, Ingress, Model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <pairs.tsv> <image_dir>", args[0]);
        std::process::exit(2);
    }
    let dir = std::path::Path::new(&args[2]);
    println!("distorted\tpublished\tv1_601\tv1_709");
    for line in std::fs::read_to_string(&args[1])?.lines() {
        let mut f = line.split('\t');
        let (r, d, p) = (
            f.next().unwrap(),
            f.next().unwrap(),
            f.next().unwrap_or(""),
        );
        let ri = decode_image_to_rgb8(&dir.join(r))?;
        let di = decode_image_to_rgb8(&dir.join(d))?;
        let s601 = vmaf::score(&ri, &di, Model::V1, Ingress::Studio601)?;
        let s709 = vmaf::score(&ri, &di, Model::V1, Ingress::Bt709)?;
        println!("{d}\t{p}\t{s601}\t{s709}");
    }
    Ok(())
}
