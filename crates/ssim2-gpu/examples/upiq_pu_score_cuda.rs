//! Score UPIQ HDR EXR pairs with the PU21-native GPU path
//! (`Ssim2::compute_linear_nits`) on CUDA, for SROCC validation against
//! the CPU prototype (fast-ssim2 `hdr-pu`) and the UPIQ baselines.
//!
//! Reads a TSV (`ref_path<TAB>dist_path`, header row) of absolute-luminance
//! EXRs; writes `ref_path<TAB>dist_path<TAB>pu_ssim2_gpu`. Pipelines are
//! cached per image dimension. See imazen/zenmetrics#25.
//!
//!   cargo run --release -p ssim2-gpu --features cuda \
//!     --example upiq_pu_score_cuda -- /tmp/upiq_pairs.tsv /tmp/pu_gpu_scores.tsv

use std::collections::HashMap;
use std::io::Write;

use cubecl::Runtime;
use ssim2_gpu::Ssim2;

type Backend = cubecl::cuda::CudaRuntime;

fn load_exr_nits(path: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let img = image::open(path)
        .map_err(|e| format!("{path}: {e}"))?
        .to_rgb32f();
    let (w, h) = (img.width(), img.height());
    Ok((img.into_raw(), w, h))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pairs = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/upiq_pairs.tsv".into());
    let out_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/pu_gpu_scores.tsv".into());

    let body = std::fs::read_to_string(&pairs).expect("read pairs tsv");
    let mut img_cache: HashMap<String, (Vec<f32>, u32, u32)> = HashMap::new();
    let mut pipes: HashMap<(u32, u32), Ssim2<Backend>> = HashMap::new();
    let mut out = String::from("ref_path\tdist_path\tpu_ssim2_gpu\n");
    let (mut ok, mut err) = (0usize, 0usize);

    for line in body.lines().skip(1) {
        let mut it = line.split('\t');
        let (Some(rp), Some(dp)) = (it.next(), it.next()) else {
            continue;
        };
        for p in [rp, dp] {
            if !img_cache.contains_key(p) {
                match load_exr_nits(p) {
                    Ok(v) => {
                        img_cache.insert(p.to_string(), v);
                    }
                    Err(e) => eprintln!("LOAD FAIL {e}"),
                }
            }
        }
        let (Some((r, rw, rh)), Some((d, dw, dh))) = (img_cache.get(rp), img_cache.get(dp)) else {
            err += 1;
            continue;
        };
        if (rw, rh) != (dw, dh) {
            err += 1;
            continue;
        }
        let pipe = pipes.entry((*rw, *rh)).or_insert_with(|| {
            let client = Backend::client(&Default::default());
            Ssim2::<Backend>::new(client, *rw, *rh).expect("Ssim2::new")
        });
        match pipe.compute_linear_nits(r, d) {
            Ok(res) => {
                out.push_str(&format!("{rp}\t{dp}\t{}\n", res.score));
                ok += 1;
                if ok % 50 == 0 {
                    eprintln!("scored {ok}…");
                }
            }
            Err(e) => {
                eprintln!("SCORE FAIL {rp}|{dp}: {e:?}");
                err += 1;
            }
        }
    }

    let mut f = std::fs::File::create(&out_path).expect("create out");
    f.write_all(out.as_bytes()).expect("write out");
    eprintln!("done: {ok} scored, {err} errored -> {out_path}");
}
