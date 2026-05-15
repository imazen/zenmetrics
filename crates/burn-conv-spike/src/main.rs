//! Perf spike: cvvdp's hand-written 5-tap separable Gaussian
//! downscale kernel vs. cubek-convolution's matmul-routed conv2d
//! (two passes for separability).
//!
//! Question being answered: should we replace `downscale_kernel`
//! with two `cubek_convolution::launch_ref` calls (im2col -> GEMM)?
//!
//! Runs at 4000x3000 f32 on CUDA. 100 iterations, reports median
//! nanoseconds per op for each path plus the ratio.
//!
//! Standalone crate; pins cubecl 0.10.0 stable so cubek-convolution
//! (which excludes 0.10.0-pre.4) resolves cleanly.

use std::time::Instant;

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::cuda::CudaRuntime;
use cubecl::ir::AddressType;
use cubecl::prelude::*;
use cubecl::server::Bindings;
use cubecl::zspace::{Shape, Strides};

mod kernel;
use kernel::downscale_kernel;

const SRC_W: u32 = 4000;
const SRC_H: u32 = 3000;
const DST_W: u32 = SRC_W.div_ceil(2); // 2000
const DST_H: u32 = SRC_H.div_ceil(2); // 1500
const ITERS: usize = 100;

fn synth_input() -> Vec<f32> {
    let n = (SRC_W * SRC_H) as usize;
    let mut v = Vec::with_capacity(n);
    // Smooth pattern with some variation so 0 isn't a degenerate
    // case for the matmul tiles.
    for y in 0..SRC_H {
        for x in 0..SRC_W {
            let fx = x as f32 / SRC_W as f32;
            let fy = y as f32 / SRC_H as f32;
            v.push(0.5 + 0.25 * (fx * 6.28).sin() * (fy * 6.28).cos());
        }
    }
    v
}

// =====================================================================
// Path A: hand-written downscale_kernel from cvvdp-gpu/pyramid.rs
// =====================================================================

fn run_handwritten(client: &ComputeClient<CudaRuntime>, src_bytes: &[u8]) -> (f64, Vec<f32>) {
    let n_src = (SRC_W * SRC_H) as usize;
    let n_dst = (DST_W * DST_H) as usize;

    let src_handle = client.create_from_slice(src_bytes);
    let dst_handle = client.empty(n_dst * core::mem::size_of::<f32>());

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((n_dst as u32).div_ceil(64), 1, 1);

    // Warm-up: 5 launches + sync. JIT compile happens here on first
    // launch, and tile/block layout caches populate.
    for _ in 0..5 {
        unsafe {
            downscale_kernel::launch::<CudaRuntime>(
                client,
                cube_count.clone(),
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&src_handle, n_src, 1),
                ArrayArg::from_raw_parts::<f32>(&dst_handle, n_dst, 1),
                ScalarArg::new(SRC_W),
                ScalarArg::new(SRC_H),
                ScalarArg::new(DST_W),
                ScalarArg::new(DST_H),
            );
        }
    }
    client.sync();

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        unsafe {
            downscale_kernel::launch::<CudaRuntime>(
                client,
                cube_count.clone(),
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&src_handle, n_src, 1),
                ArrayArg::from_raw_parts::<f32>(&dst_handle, n_dst, 1),
                ScalarArg::new(SRC_W),
                ScalarArg::new(SRC_H),
                ScalarArg::new(DST_W),
                ScalarArg::new(DST_H),
            );
        }
        client.sync();
        samples.push(t0.elapsed().as_nanos() as f64);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ns = samples[samples.len() / 2];

    let dst_bytes = client.read_one(dst_handle.binding());
    let mut dst = vec![0f32; n_dst];
    let dst_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut dst);
    dst_slice.copy_from_slice(&dst_bytes);

    (median_ns, dst)
}

// =====================================================================
// Path B: cubek-convolution two-pass (5x1 then 1x5)
// =====================================================================

use cubek_convolution::{
    AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy,
    launch_ref,
};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek_std::InputBinding;

fn run_cubek_pass(
    client: &ComputeClient<CudaRuntime>,
    in_handle: cubecl::server::Handle,
    weight_handle: cubecl::server::Handle,
    out_handle: cubecl::server::Handle,
    in_shape: [usize; 4],       // [N, H, W, C]
    weight_shape: [usize; 4],   // [O, kH, kW, C]
    out_shape_arr: [usize; 4],  // [N, oH, oW, O]
    stride: [usize; 2],
    padding: [usize; 2],
    strategy: &Strategy,
    dtypes: &MatmulElems,
) -> Result<(), String> {
    let f32_size = core::mem::size_of::<f32>();
    let in_size = in_shape.iter().product::<usize>();
    let weight_size = weight_shape.iter().product::<usize>();
    let out_size = out_shape_arr.iter().product::<usize>();

    let in_strides = row_major_strides(&in_shape);
    let weight_strides = row_major_strides(&weight_shape);
    let out_strides = row_major_strides(&out_shape_arr);

    let in_binding = TensorBinding::<CudaRuntime> {
        handle: in_handle.binding(),
        shape: Shape::new(in_shape),
        strides: Strides::new(&in_strides),
        runtime: core::marker::PhantomData,
    };
    let weight_binding = TensorBinding::<CudaRuntime> {
        handle: weight_handle.binding(),
        shape: Shape::new(weight_shape),
        strides: Strides::new(&weight_strides),
        runtime: core::marker::PhantomData,
    };
    let out_binding = TensorBinding::<CudaRuntime> {
        handle: out_handle.binding(),
        shape: Shape::new(out_shape_arr),
        strides: Strides::new(&out_strides),
        runtime: core::marker::PhantomData,
    };
    let _ = (in_size, weight_size, out_size, f32_size); // silence unused

    let inputs = ConvolutionInputs::Forward {
        input: InputBinding::new(in_binding, dtypes.lhs_global),
        weight: InputBinding::new(weight_binding, dtypes.rhs_global),
        bias: None,
        out: out_binding,
    };
    let args = ConvolutionArgs::<2> {
        stride,
        padding,
        dilation: [1, 1],
    };
    launch_ref(strategy, client, inputs, args, dtypes.clone())
        .map_err(|e| format!("{e:?}"))
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn run_cubek_separable(
    client: &ComputeClient<CudaRuntime>,
    src_bytes: &[u8],
    strategy: &Strategy,
) -> Result<(f64, Vec<f32>), String> {
    // Kernel: 5-tap Gaussian
    let k: [f32; 5] = [0.05, 0.25, 0.40, 0.25, 0.05];

    // V pass: input [1, 3000, 4000, 1] x weight [1, 5, 1, 1] -> [1, 1500, 4000, 1]
    //   stride [2, 1], padding [2, 0] (kH=5 needs pad 2)
    // H pass: vscratch [1, 1500, 4000, 1] x weight [1, 1, 5, 1] -> [1, 1500, 2000, 1]
    //   stride [1, 2], padding [0, 2]

    let f32_size = core::mem::size_of::<f32>();
    let n_src = (SRC_W * SRC_H) as usize;
    let n_v = (SRC_W * DST_H) as usize; // 4000 * 1500
    let n_dst = (DST_W * DST_H) as usize; // 2000 * 1500

    // Allocate
    let src_handle = client.create_from_slice(src_bytes);
    let vscratch_handle = client.empty(n_v * f32_size);
    let dst_handle = client.empty(n_dst * f32_size);

    let weight_v_bytes = bytemuck::cast_slice::<f32, u8>(&k);
    let weight_h_bytes = bytemuck::cast_slice::<f32, u8>(&k);
    let weight_v_handle = client.create_from_slice(weight_v_bytes);
    let weight_h_handle = client.create_from_slice(weight_h_bytes);

    // Dtypes: f32 all the way (no f16 globals on this 1-channel test).
    let f32_storage = f32::as_type_native_unchecked().storage_type();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: f32_storage,
        rhs: f32_storage,
        out: f32_storage,
    });

    // Warm-up: 5 iterations.
    for _ in 0..5 {
        run_cubek_pass(
            client,
            src_handle.clone(),
            weight_v_handle.clone(),
            vscratch_handle.clone(),
            [1, SRC_H as usize, SRC_W as usize, 1],
            [1, 5, 1, 1],
            [1, DST_H as usize, SRC_W as usize, 1],
            [2, 1],
            [2, 0],
            strategy,
            &dtypes,
        )?;
        run_cubek_pass(
            client,
            vscratch_handle.clone(),
            weight_h_handle.clone(),
            dst_handle.clone(),
            [1, DST_H as usize, SRC_W as usize, 1],
            [1, 1, 5, 1],
            [1, DST_H as usize, DST_W as usize, 1],
            [1, 2],
            [0, 2],
            strategy,
            &dtypes,
        )?;
    }
    client.sync();

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        run_cubek_pass(
            client,
            src_handle.clone(),
            weight_v_handle.clone(),
            vscratch_handle.clone(),
            [1, SRC_H as usize, SRC_W as usize, 1],
            [1, 5, 1, 1],
            [1, DST_H as usize, SRC_W as usize, 1],
            [2, 1],
            [2, 0],
            strategy,
            &dtypes,
        )?;
        run_cubek_pass(
            client,
            vscratch_handle.clone(),
            weight_h_handle.clone(),
            dst_handle.clone(),
            [1, DST_H as usize, SRC_W as usize, 1],
            [1, 1, 5, 1],
            [1, DST_H as usize, DST_W as usize, 1],
            [1, 2],
            [0, 2],
            strategy,
            &dtypes,
        )?;
        client.sync();
        samples.push(t0.elapsed().as_nanos() as f64);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ns = samples[samples.len() / 2];

    let dst_bytes = client.read_one(dst_handle.binding());
    let mut dst = vec![0f32; n_dst];
    let dst_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut dst);
    dst_slice.copy_from_slice(&dst_bytes);

    let _ = Bindings::default;
    Ok((median_ns, dst))
}

fn main() {
    println!("=== burn-conv-spike: cvvdp downscale vs cubek separable conv2d ===");
    println!("Hardware: AMD Ryzen 9 7950X + RTX 5070 (sm_120)");
    println!("Input: {}x{} f32, output: {}x{} f32", SRC_W, SRC_H, DST_W, DST_H);
    println!("Iterations per measurement: {}", ITERS);
    println!();

    let device = Default::default();
    let client = CudaRuntime::client(&device);

    let src = synth_input();
    let src_bytes: &[u8] = bytemuck::cast_slice(&src);

    // --- Path A: hand-written ---
    println!("[A] hand-written downscale_kernel (cvvdp)...");
    let (hand_ns, hand_out) = run_handwritten(&client, src_bytes);
    println!("    median: {:.0} ns/op ({:.3} ms)", hand_ns, hand_ns / 1e6);

    // --- Path B: cubek separable, multiple algorithms ---
    let algorithms = &[
        ("SimpleSyncCyclic+Cmma", ConvAlgorithm::SimpleSyncCyclic, AcceleratedTileKind::Cmma),
        ("SimpleSyncStrided+Cmma", ConvAlgorithm::SimpleSyncStrided, AcceleratedTileKind::Cmma),
        ("SimpleSyncTilewise+Cmma", ConvAlgorithm::SimpleSyncTilewise, AcceleratedTileKind::Cmma),
        ("SimpleAsyncCyclic+Cmma", ConvAlgorithm::SimpleAsyncCyclic, AcceleratedTileKind::Cmma),
        ("SimpleSyncCyclic+Mma", ConvAlgorithm::SimpleSyncCyclic, AcceleratedTileKind::Mma),
    ];

    let mut best_cubek: Option<(String, f64, Vec<f32>)> = None;
    for (label, algo, tile) in algorithms {
        let strategy = Strategy::Inferred {
            algorithm: *algo,
            tile_kind: *tile,
        };
        println!("[B] cubek separable ({})...", label);
        match run_cubek_separable(&client, src_bytes, &strategy) {
            Ok((ns, out)) => {
                println!("    median: {:.0} ns/op ({:.3} ms)", ns, ns / 1e6);
                if best_cubek.as_ref().is_none_or(|(_, prev, _)| ns < *prev) {
                    best_cubek = Some(((*label).to_string(), ns, out));
                }
            }
            Err(e) => {
                println!("    FAILED: {}", e);
            }
        }
    }

    // --- Parity check + verdict ---
    println!();
    println!("=== Results ===");
    println!("hand-written downscale_kernel: {:.0} ns/op", hand_ns);

    match best_cubek {
        None => {
            println!("cubek separable: ALL ALGORITHMS FAILED");
            println!("Verdict: cannot evaluate — cubek path did not run.");
        }
        Some((label, cubek_ns, cubek_out)) => {
            println!("cubek separable (best: {}): {:.0} ns/op", label, cubek_ns);
            let ratio = cubek_ns / hand_ns;
            println!("delta (cubek/hand): {:.2}x", ratio);

            // Sentinel parity check: middle pixel
            let mid_idx = (DST_H / 2 * DST_W + DST_W / 2) as usize;
            let h_mid = hand_out[mid_idx];
            let c_mid = cubek_out[mid_idx];
            let abs_diff = (h_mid - c_mid).abs();
            let rel_diff = abs_diff / h_mid.abs().max(1e-12);
            println!(
                "parity (mid pixel): hand={:.6} cubek={:.6} abs_diff={:.6} rel_diff={:.6}",
                h_mid, c_mid, abs_diff, rel_diff
            );
            let parity_ok = abs_diff < 1e-3 || rel_diff < 1e-3;
            println!("parity_ok (tolerance 1e-3): {}", parity_ok);
            if !parity_ok {
                println!("NOTE: Parity mismatch is expected — cvvdp's hand-written kernel");
                println!("uses symmetric reflection padding + pycvvdp's bug-compat edge");
                println!("delta; cubek uses zero padding. Interior pixels (far from edges)");
                println!("should still match approximately.");

                // Try a more central pixel
                let center_idx = ((DST_H / 2) * DST_W + DST_W / 2) as usize;
                let n_center_samples = 5;
                let mut max_diff: f32 = 0.0;
                for dy in 0..n_center_samples {
                    for dx in 0..n_center_samples {
                        let idx = ((DST_H / 2 + dy) * DST_W + DST_W / 2 + dx) as usize;
                        let d = (hand_out[idx] - cubek_out[idx]).abs();
                        if d > max_diff { max_diff = d; }
                    }
                }
                let _ = center_idx;
                println!("max abs_diff over central 5x5 patch: {:.6}", max_diff);
            }

            println!();
            println!("=== Verdict ===");
            if ratio < 1.10 {
                println!("PASS: cubek path is within 10% of hand-written. Burn port is viable.");
            } else if ratio <= 1.5 {
                println!("MARGINAL: cubek path is {:.2}x slower (10-50% regression).", ratio);
                println!("Flag for user judgement — the API simplification may be worth the cost,");
                println!("but it's not a free win.");
            } else {
                println!("FAIL: cubek path is {:.2}x slower (>50% regression).", ratio);
                println!("The Burn port via cubek is a perf regression at this problem size.");
                println!("Recommend abandoning the rewrite; keep the hand-written kernel.");
            }
        }
    }
}
