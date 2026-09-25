//! Interleaved GMSD/MDSI/MS-GMSDc matrix; raw zenbench samples stay external.
use std::{sync::Arc, time::Duration};
use zenbench::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output =
        std::path::PathBuf::from(std::env::args().nth(1).ok_or("output directory required")?);
    std::fs::create_dir_all(&output)?;
    // Keep resource observations, but do not occupy the shared heavy lock
    // waiting for unrelated activity to stop. Unclean rounds are reported.
    let gate = zenbench::GateConfig::default().max_wait(Duration::ZERO);
    let result = zenbench::run_gated(gate, |suite| {
        for size in [64, 256, 1024, 4096] {
            let mut state = 20260924_u32;
            let r: Arc<Vec<u8>> = Arc::new(
                (0..size * size * 3)
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        (state >> 24) as u8
                    })
                    .collect(),
            );
            let d: Arc<Vec<u8>> = Arc::new(
                r.iter()
                    .enumerate()
                    .map(|(i, &v)| if i % 3 == 0 { v / 2 } else { v })
                    .collect(),
            );
            for threads in [1, 8] {
                let pool = Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap(),
                );
                suite.group(format!("rgb8_{size}_t{threads}"), |group| {
                    group.throughput(Throughput::Elements((size * size) as u64));
                    group
                        .config()
                        .max_rounds(30)
                        .min_rounds(10)
                        .max_time(Duration::from_secs(3));
                    for metric in ["gmsd", "mdsi", "ms_gmsdc"] {
                        let (r, d, pool) = (r.clone(), d.clone(), pool.clone());
                        group.bench(metric, move |b| {
                            b.iter(|| {
                                pool.install(|| match metric {
                                    "gmsd" => {
                                        gmsd::gmsd_rgb8(&r, &d, size, size, size * 3).unwrap().gmsd
                                    }
                                    "mdsi" => {
                                        gmsd::mdsi_rgb8(&r, &d, size, size, size * 3).unwrap()
                                    }
                                    _ => gmsd::ms_gmsdc_rgb8(&r, &d, size, size, size * 3).unwrap(),
                                })
                            })
                        });
                    }
                });
            }
        }
    });
    result.save(output.join("zenbench.json"))?;
    std::fs::write(output.join("zenbench.csv"), result.to_csv())?;
    println!("raw samples: {}", output.display());
    Ok(())
}
