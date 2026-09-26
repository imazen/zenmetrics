//! Old-vs-new spatial pool, interleaved (zenbench).
//!
//! `old_atomic` replays the pool `Cvvdp::compute_dkl_jod` dispatched
//! before 2026-09-26: zero the partials, then per level either
//! `pool_band_3ch_lds_kernel` (≥ 16 384 px, one `Atomic<f32>` add per
//! workgroup) or `pool_band_3ch_kernel` (one atomic add per pixel).
//! `new_fixed_order` is the current two-pass pool: zero the row
//! partials, `pool_rows_3ch_kernel` per level, one
//! `pool_rows_finalize_kernel`. Both read the `n_levels × 3` partials
//! back, so each iteration includes the GPU sync, like the real call.
//!
//! Inputs: an N×N pyramid (the pipeline's level count and ceil-halved
//! dims) of fixed pseudo-random D values, shared by both benches. The
//! rest of the JOD pipeline is untouched by the change, so this is the
//! whole of the cost difference.
//!
//! Before timing, each size prints an accuracy line: every (level,
//! channel) partial from 20 `old_atomic` runs and from `new_fixed_order`
//! against an f64 host sum of the same contributions
//! (`(|x| + 1e-5)² − 1e-5²` of the same f32 D values).
//!
//! Run: `cargo bench -p cvvdp-gpu --no-default-features --features wgpu,cubecl-types --bench pool_reduction`
//! Sizes: `POOL_BENCH_SIZES` (default `256,1024,4096`). On wgpu the old
//! LDS dispatch at 4096² asks for 65 536 workgroups in x, over the
//! 65 535 limit, so run 4096² with `--features cuda`.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cubecl::prelude::*;

#[path = "../tests/it/common/mod.rs"]
mod common;
use common::Backend;
use cvvdp_gpu::kernels::pool::{
    BETA_SPATIAL, POOL_LDS_BLOCK_DIM, POOL_ROW_LANES, fill_f32_kernel, pool_band_3ch_kernel,
    pool_band_3ch_lds_kernel, pool_rows_3ch_kernel, pool_rows_finalize_kernel,
};
use cvvdp_gpu::kernels::pyramid::band_frequencies;
use cvvdp_gpu::params::DisplayGeometry;
use zenbench::prelude::*;

const N_CH: usize = 3;

#[derive(Clone)]
struct Pyramid {
    /// f64 host sum of the per-pixel contributions, per (level, channel).
    reference: Vec<f64>,
    client: ComputeClient<Backend>,
    /// (width, height, D planes) per level.
    levels: Vec<(usize, usize, [cubecl::server::Handle; N_CH])>,
    partials: cubecl::server::Handle,
    rows: cubecl::server::Handle,
    rows_len: usize,
    meta: cubecl::server::Handle,
}

fn build(n: usize) -> Pyramid {
    let client = Backend::client(&Default::default());
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let n_levels = band_frequencies(ppd, n, n).len().min(cvvdp_gpu::MAX_LEVELS);
    let mut levels = Vec::with_capacity(n_levels);
    let (mut w, mut h) = (n, n);
    let mut seed = 0x1234_5678_u32;
    let mut reference = Vec::with_capacity(n_levels * N_CH);
    let eps = 1e-5_f64;
    for _ in 0..n_levels {
        let planes = std::array::from_fn(|_| {
            let v: Vec<f32> = (0..w * h)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    (seed as f32 / u32::MAX as f32) * 4.0 - 2.0
                })
                .collect();
            reference.push(
                v.iter()
                    .map(|&x| (f64::from(x.abs()) + eps).powi(2) - eps * eps)
                    .sum::<f64>(),
            );
            client.create_from_slice(f32::as_bytes(&v))
        });
        levels.push((w, h, planes));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let rows_len = N_CH * levels.iter().map(|l| l.1).sum::<usize>();
    let mut meta = Vec::new();
    let mut base = 0usize;
    for &(_, bh, _) in &levels {
        for c in 0..N_CH {
            meta.push((base + c * bh) as u32);
            meta.push(bh as u32);
        }
        base += N_CH * bh;
    }
    Pyramid {
        partials: client.create_from_slice(f32::as_bytes(&vec![0.0; n_levels * N_CH])),
        rows: client.create_from_slice(f32::as_bytes(&vec![0.0; rows_len])),
        meta: client.create_from_slice(u32::as_bytes(&meta)),
        reference,
        rows_len,
        levels,
        client,
    }
}

fn read_partials(p: &Pyramid) -> Vec<f32> {
    let bytes = p
        .client
        .read_one(p.partials.clone())
        .expect("read partials");
    f32::from_bytes(&bytes).to_vec()
}

fn fill(p: &Pyramid, h: &cubecl::server::Handle, n: usize) {
    unsafe {
        fill_f32_kernel::launch::<Backend>(
            &p.client,
            CubeCount::Static((n as u32).div_ceil(64), 1, 1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(h.clone(), n),
            0.0,
            n as u32,
        );
    }
}

/// The pre-2026-09-26 `_pool_q_per_ch` dispatch, verbatim in shape.
fn old_atomic(p: &Pyramid) -> Vec<f32> {
    let n_partials = p.levels.len() * N_CH;
    fill(p, &p.partials, n_partials);
    for (k, (bw, bh, d)) in p.levels.iter().enumerate() {
        let n_px = bw * bh;
        let (ia, irg, ivy) = (
            (k * N_CH) as u32,
            (k * N_CH + 1) as u32,
            (k * N_CH + 2) as u32,
        );
        unsafe {
            if n_px >= 16_384 {
                pool_band_3ch_lds_kernel::launch::<Backend>(
                    &p.client,
                    CubeCount::Static((n_px as u32).div_ceil(POOL_LDS_BLOCK_DIM), 1, 1),
                    CubeDim::new_1d(POOL_LDS_BLOCK_DIM),
                    ArrayArg::from_raw_parts(d[0].clone(), n_px),
                    ArrayArg::from_raw_parts(d[1].clone(), n_px),
                    ArrayArg::from_raw_parts(d[2].clone(), n_px),
                    ArrayArg::from_raw_parts(p.partials.clone(), n_partials),
                    BETA_SPATIAL,
                    ia,
                    irg,
                    ivy,
                    n_px as u32,
                );
            } else {
                pool_band_3ch_kernel::launch::<Backend>(
                    &p.client,
                    CubeCount::Static((n_px as u32).div_ceil(64), 1, 1),
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(d[0].clone(), n_px),
                    ArrayArg::from_raw_parts(d[1].clone(), n_px),
                    ArrayArg::from_raw_parts(d[2].clone(), n_px),
                    ArrayArg::from_raw_parts(p.partials.clone(), n_partials),
                    BETA_SPATIAL,
                    ia,
                    irg,
                    ivy,
                    n_px as u32,
                );
            }
        }
    }
    read_partials(p)
}

/// The current `_pool_q_per_ch` dispatch (see `pipeline.rs`).
fn new_fixed_order(p: &Pyramid) -> Vec<f32> {
    fill(p, &p.rows, p.rows_len);
    let mut base = 0usize;
    for (bw, bh, d) in &p.levels {
        let (bw, bh) = (*bw, *bh);
        let gx = bh.clamp(1, 32_768);
        unsafe {
            pool_rows_3ch_kernel::launch::<Backend>(
                &p.client,
                CubeCount::Static(gx as u32, bh.div_ceil(gx) as u32, 1),
                CubeDim::new_1d(POOL_ROW_LANES),
                ArrayArg::from_raw_parts(d[0].clone(), bw * bh),
                ArrayArg::from_raw_parts(d[1].clone(), bw * bh),
                ArrayArg::from_raw_parts(d[2].clone(), bw * bh),
                ArrayArg::from_raw_parts(p.rows.clone(), p.rows_len),
                BETA_SPATIAL,
                bw as u32,
                bh as u32,
                0,
                base as u32,
                (base + bh) as u32,
                (base + 2 * bh) as u32,
                (bw as u32).div_ceil(POOL_ROW_LANES),
            );
        }
        base += N_CH * bh;
    }
    let n_slots = p.levels.len() * N_CH;
    let max_rows = p.levels.iter().map(|l| l.1).max().unwrap_or(1) as u32;
    unsafe {
        pool_rows_finalize_kernel::launch::<Backend>(
            &p.client,
            CubeCount::Static(n_slots as u32, 1, 1),
            CubeDim::new_1d(POOL_ROW_LANES),
            ArrayArg::from_raw_parts(p.rows.clone(), p.rows_len),
            ArrayArg::from_raw_parts(p.meta.clone(), 2 * n_slots),
            ArrayArg::from_raw_parts(p.partials.clone(), n_slots),
            max_rows.div_ceil(POOL_ROW_LANES),
        );
    }
    read_partials(p)
}

fn bench_pool(suite: &mut Suite) {
    let sizes = std::env::var("POOL_BENCH_SIZES").unwrap_or_else(|_| "256,1024,4096".into());
    for n in sizes
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("size"))
    {
        let p = build(n);
        // Accuracy record: relative error of each partial vs the f64 sum.
        let rel = |got: &[f32]| -> Vec<f64> {
            got.iter()
                .zip(&p.reference)
                .map(|(&g, &r)| (f64::from(g) - r) / r)
                .collect()
        };
        let olds: Vec<Vec<f64>> = (0..20).map(|_| rel(&old_atomic(&p))).collect();
        let new_a = new_fixed_order(&p);
        let new_b = new_fixed_order(&p);
        let new = rel(&new_a);
        let max_abs = |v: &[f64]| v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let old_max = olds.iter().map(|v| max_abs(v)).fold(0.0_f64, f64::max);
        let old_mean = olds.iter().map(|v| mean(v)).sum::<f64>() / olds.len() as f64;
        let old_runs_distinct = {
            let mut bits: Vec<Vec<u64>> = olds
                .iter()
                .map(|v| v.iter().map(|x| x.to_bits()).collect())
                .collect();
            bits.sort();
            bits.dedup();
            bits.len()
        };
        eprintln!(
            "pool_reduction {n}x{n}: levels={} slots={} | old_atomic x20: distinct partial vectors={old_runs_distinct} \
             max|rel err|={old_max:.3e} mean rel err={old_mean:+.3e} | new_fixed_order: repeat bit-identical={} \
             max|rel err|={:.3e} mean rel err={:+.3e}",
            p.levels.len(),
            p.reference.len(),
            new_a
                .iter()
                .zip(&new_b)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            max_abs(&new),
            mean(&new),
        );
        let (p_old, p_new) = (p.clone(), p);
        suite.group(format!("pool_{n}x{n}"), move |g| {
            g.throughput(Throughput::Elements((n * n) as u64));
            g.config()
                .min_rounds(30)
                .max_time(std::time::Duration::from_secs(60))
                .max_wall_time(std::time::Duration::from_secs(300));
            let p_old = p_old.clone();
            g.bench("old_atomic", move |bch| bch.iter(|| old_atomic(&p_old)));
            let p_new = p_new.clone();
            g.bench("new_fixed_order", move |bch| {
                bch.iter(|| new_fixed_order(&p_new))
            });
            g.baseline("old_atomic");
        });
    }
}

zenbench::main!(bench_pool);
