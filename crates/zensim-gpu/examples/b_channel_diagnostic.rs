//! B-channel multi-strip parity diagnostic for the strip-local masked-IW
//! kernel.
//!
//! ## What this does
//!
//! Reproduces the CPU's `process_strip_channel` per-strip processing
//! AND the GPU's `masked_iw_strip_kernel` per-strip processing as
//! shadow CPU implementations in pure Rust, on the SAME positive XYB
//! input planes. Dumps per-row diagnostic values for every
//! (scale, strip, channel) triple to TSV files in `/tmp/`.
//!
//! Why both as CPU shadows: we already know the GPU kernel ships a
//! particular formula (read from `kernels::masked_iw_strip.rs`); we
//! need to know what the CPU is ACTUALLY computing to locate the gap.
//! Re-deriving both on the same XYB inputs and the same Rust types
//! eliminates GPU-side f32 rounding / async-launch confounders.
//!
//! ## Output
//!
//! Per (scale, strip, channel) one TSV per side at:
//!   /tmp/zensim_diag_cpu_scale{N}_strip{S}_chan{C}.tsv
//!   /tmp/zensim_diag_gpu_scale{N}_strip{S}_chan{C}.tsv
//! With columns:
//!   sy, gy, mu1_h, mu1_after_swap, mask_step1, mu_blur_inner,
//!   activity, mask_w, iw_w
//!
//! And a master diff at /tmp/zensim_diag_summary.tsv with
//! per-(scale,strip,channel) max-abs-rel divergence per column.
//!
//! Run with:
//!   PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
//!     cargo run --release -p zensim-gpu --features cuda --no-default-features \
//!     --example b_channel_diagnostic

use cubecl::Runtime;
use zensim_gpu::{Zensim, ZensimFeatureRegime};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!("requires cuda or wgpu feature");

// CPU shadow constants — must match zensim/src/streaming.rs.
const STRIP_INNER: usize = 32;
const R: usize = 5; // blur radius
const DIAM: usize = 2 * R + 1;
const K_MASK: f32 = 4.0;

fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w) as u8;
            let g = ((y * 255) / h) as u8;
            let b = (((x + y) * 255) / (w + h)) as u8;
            v.push(r);
            v.push(g);
            v.push(b);
        }
    }
    v
}

fn add_noise(data: &[u8], amount: i16) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = Vec::with_capacity(data.len());
    let mut seed = Wrapping(12345_u32);
    for &v in data {
        seed = seed * Wrapping(1103515245_u32) + Wrapping(12345_u32);
        let noise = ((seed.0 >> 16) as i16 % (amount * 2 + 1)) - amount;
        out.push((v as i16 + noise).clamp(0, 255) as u8);
    }
    out
}

// === Boundary handling (matches CPU + GPU) ===

#[inline]
fn mirror_idx(i: usize, r: usize, height: usize) -> usize {
    if i <= r {
        (r - i).min(height - 1)
    } else {
        (i - r).min(height - 1)
    }
}

#[inline]
fn vblur_add_idx(y: usize, r: usize, height: usize) -> usize {
    let add_raw = y + r + 1;
    if add_raw < height {
        add_raw
    } else {
        let reflected = 2 * (height as isize - 1) - add_raw as isize;
        reflected.unsigned_abs().min(height - 1)
    }
}

#[inline]
fn vblur_rem_idx(y: usize, r: usize, height: usize) -> usize {
    let rem_i = y as isize - r as isize;
    let idx = if rem_i < 0 {
        rem_i.unsigned_abs()
    } else {
        rem_i as usize
    };
    idx.min(height - 1)
}

#[inline]
fn h_mirror_idx(x: isize, width: usize) -> usize {
    let w_i = width as isize;
    let period = 2 * (w_i - 1);
    let mut x = ((x % period) + period) % period;
    if x >= w_i {
        x = period - x;
    }
    (x as usize).min(width - 1)
}

// === Box blur (1D) for H-blur of a strip-row ===

fn h_blur_row(src: &[f32], width: usize, radius: usize, out: &mut [f32]) {
    let r = radius;
    let diam = 2 * r + 1;
    let inv = 1.0_f32 / diam as f32;
    for x in 0..width {
        let mut sum = 0.0_f32;
        for k in 0..diam as isize {
            let kx = x as isize - r as isize + k;
            sum += src[h_mirror_idx(kx, width)];
        }
        out[x] = sum * inv;
    }
}

// === H-blur of a full strip plane (rows × cols), one row at a time ===

fn h_blur_plane(src: &[f32], width: usize, height: usize, radius: usize, out: &mut [f32]) {
    for y in 0..height {
        let off = y * width;
        let src_row = &src[off..off + width];
        let out_row = &mut out[off..off + width];
        h_blur_row(src_row, width, radius, out_row);
    }
}

// === Per-column V-blur (strip-local mirror) for a plane ===

fn v_blur_plane(src: &[f32], width: usize, height: usize, radius: usize, out: &mut [f32]) {
    let r = radius;
    let diam = 2 * r + 1;
    let inv = 1.0_f32 / diam as f32;
    for x in 0..width {
        let mut sum = 0.0_f32;
        // Prefix init: y in 0..diam, mirror_idx
        for i in 0..diam {
            let idx = mirror_idx(i, r, height);
            sum += src[idx * width + x];
        }
        for y in 0..height {
            out[y * width + x] = sum * inv;
            let add = vblur_add_idx(y, r, height);
            let rem = vblur_rem_idx(y, r, height);
            sum = sum + src[add * width + x] - src[rem * width + x];
        }
    }
}

// === SSIM math (single per-pixel, matches fused.rs / GPU kernel) ===
// Currently unused — diagnostic uses summary metrics rather than per-pixel.
// Kept around because the next b-channel divergence investigation will
// likely want this verbatim.

#[allow(dead_code)]
const C2: f32 = 0.0009;

#[allow(dead_code)]
fn ssim_sd(mu1: f32, mu2: f32, ssq: f32, s12: f32) -> f32 {
    let mu_diff = mu1 - mu2;
    let num_m = mu_diff.mul_add(-mu_diff, 1.0);
    let num_s = 2.0_f32.mul_add((-mu1).mul_add(mu2, s12), C2);
    let denom_s = (-mu2).mul_add(mu2, (-mu1).mul_add(mu1, ssq)) + C2;
    1.0 - (num_m * num_s) / denom_s
}

// === Strip layout (matches CPU streaming.rs) ===

fn strip_layout(strip_idx: usize, height: usize) -> (usize, usize, usize, usize, usize, usize) {
    // Returns (strip_top, strip_bot, strip_h, inner_start_global, inner_h, inner_off)
    let inner_start_g = strip_idx * STRIP_INNER;
    let inner_end_g = ((strip_idx + 1) * STRIP_INNER).min(height);
    let inner_h = inner_end_g - inner_start_g;
    let strip_top = inner_start_g.saturating_sub(R);
    let strip_bot = (inner_end_g + R).min(height);
    let strip_h = strip_bot - strip_top;
    let inner_off = inner_start_g - strip_top;
    (
        strip_top,
        strip_bot,
        strip_h,
        inner_start_g,
        inner_h,
        inner_off,
    )
}

fn cpu_strip_count(height: usize) -> usize {
    height.div_ceil(STRIP_INNER)
}

/// Per-(channel) shadow CPU state — replays `process_strip_channel`
/// for one scale, dumping every per-row buffer to TSVs.
///
/// Inputs: full-image ref XYB plane [width × height] for each of 3
/// channels, full-image dist plane likewise.
fn cpu_shadow_dump(
    label: &str,
    scale: usize,
    width: usize,
    height: usize,
    ref_planes: &[Vec<f32>; 3],
    dis_planes: &[Vec<f32>; 3],
) -> Vec<(usize, usize, ChannelDump)> {
    let mut out = Vec::new();
    let n_strips = cpu_strip_count(height);
    // Persistent buffers across strips AND channels — matches CPU.
    let max_strip_h = STRIP_INNER + 2 * R;
    let max_strip_n = max_strip_h * width;
    let mut bufs = ScaleBufs::new(max_strip_n);

    for strip_idx in 0..n_strips {
        let (strip_top, strip_bot, strip_h, _, inner_h, inner_off) =
            strip_layout(strip_idx, height);
        let strip_n = strip_h * width;
        bufs.resize(strip_n);

        for ch in 0..3 {
            let src_strip = ref_planes[ch][strip_top * width..strip_bot * width].to_vec();
            let dst_strip = dis_planes[ch][strip_top * width..strip_bot * width].to_vec();
            let dump = cpu_one_channel_one_strip(
                width, strip_h, inner_off, inner_h, &src_strip, &dst_strip, &mut bufs,
            );
            // Write per-row TSV for the dump.
            let path = format!(
                "/tmp/zensim_diag_{}_scale{}_strip{}_chan{}.tsv",
                label, scale, strip_idx, ch
            );
            write_dump_tsv(&path, strip_top, &dump);
            out.push((strip_idx, ch, dump));
        }
    }
    out
}

struct ScaleBufs {
    mu1: Vec<f32>,
    mu2: Vec<f32>,
    sigma1_sq: Vec<f32>,
    sigma12: Vec<f32>,
    mask: Vec<f32>,
    mul_buf: Vec<f32>,
    temp_blur: Vec<f32>,
    iw_weight: Vec<f32>,
}
impl ScaleBufs {
    fn new(size: usize) -> Self {
        Self {
            mu1: vec![0.0; size],
            mu2: vec![0.0; size],
            sigma1_sq: vec![0.0; size],
            sigma12: vec![0.0; size],
            mask: vec![0.0; size],
            mul_buf: vec![0.0; size],
            temp_blur: vec![0.0; size],
            iw_weight: vec![0.0; size],
        }
    }
    fn resize(&mut self, size: usize) {
        self.mu1.resize(size, 0.0);
        self.mu2.resize(size, 0.0);
        self.sigma1_sq.resize(size, 0.0);
        self.sigma12.resize(size, 0.0);
        self.mask.resize(size, 0.0);
        self.mul_buf.resize(size, 0.0);
        self.temp_blur.resize(size, 0.0);
        self.iw_weight.resize(size, 0.0);
    }
}

#[derive(Clone)]
#[allow(dead_code)] // some fields are dump-target context, set but not read
struct ChannelDump {
    width: usize,
    strip_h: usize,
    inner_off: usize,
    inner_h: usize,
    /// Per-row average of mu1 (after H-blur, before V-blur).
    mu1_h_avg: Vec<f32>,
    /// Per-row average of mu1 at the row AFTER swap-from-V-blur, i.e.
    /// the actual mu1 the activity computation uses. At inner rows this is
    /// V-blurred mu1; at overlap rows this is whatever stale state happens
    /// to be in `bufs.mask` from earlier processing.
    mu1_post_swap_avg: Vec<f32>,
    /// Per-row average of mask after Step 1 (|src - mu1_post_swap|).
    mask_step1_avg: Vec<f32>,
    /// Per-row average of mul_buf after Step 2 (V-blur of mask_step1).
    activity_avg: Vec<f32>,
    /// Per-row sample at column = width/2 for spot-checks.
    sample_col: usize,
    mu1_h_sample: Vec<f32>,
    mu1_post_swap_sample: Vec<f32>,
    mask_step1_sample: Vec<f32>,
    activity_sample: Vec<f32>,
}

fn cpu_one_channel_one_strip(
    width: usize,
    strip_h: usize,
    inner_off: usize,
    inner_h: usize,
    src: &[f32],
    dst: &[f32],
    bufs: &mut ScaleBufs,
) -> ChannelDump {
    // === Step A: H-blur src,dst → mu1/mu2/sigma1_sq/sigma12 (all rows) ===
    // Matches `fused_blur_h_ssim`.
    let strip_n = strip_h * width;
    h_blur_plane(src, width, strip_h, R, &mut bufs.mu1);
    h_blur_plane(dst, width, strip_h, R, &mut bufs.mu2);
    // sigma1_sq holds H-blur of src² + dst²
    let mut sq_combined = vec![0.0_f32; strip_n];
    for i in 0..strip_n {
        sq_combined[i] = src[i] * src[i] + dst[i] * dst[i];
    }
    h_blur_plane(&sq_combined, width, strip_h, R, &mut bufs.sigma1_sq);
    let mut prod = vec![0.0_f32; strip_n];
    for i in 0..strip_n {
        prod[i] = src[i] * dst[i];
    }
    h_blur_plane(&prod, width, strip_h, R, &mut bufs.sigma12);

    // Snapshot mu1 H-blur (per-row averages) for the dump.
    let mu1_h_avg = row_averages(&bufs.mu1, width, strip_h);
    let sample_col = width / 2;
    let mu1_h_sample = row_samples(&bufs.mu1, width, strip_h, sample_col);

    // === Step B: fused V-blur features ===
    // Matches `fused_vblur_features_ssim`: only writes mask/mul_buf at
    // inner rows. Overlap rows keep their previous values.
    //
    // For our purposes we don't need the per-pixel accumulators, just
    // the side-effect on `bufs.mask` and `bufs.mul_buf`.
    let mut v_mu1_out = bufs.mask.clone(); // simulate the overlap-stale path
    let mut v_mu2_out = bufs.mul_buf.clone();
    {
        let r = R;
        let diam = DIAM;
        let inv = 1.0_f32 / diam as f32;
        for x in 0..width {
            let mut sum_m1 = 0.0_f32;
            let mut sum_m2 = 0.0_f32;
            for i in 0..diam {
                let idx = mirror_idx(i, r, strip_h);
                sum_m1 += bufs.mu1[idx * width + x];
                sum_m2 += bufs.mu2[idx * width + x];
            }
            for y in 0..strip_h {
                if y >= inner_off && y < inner_off + inner_h {
                    v_mu1_out[y * width + x] = sum_m1 * inv;
                    v_mu2_out[y * width + x] = sum_m2 * inv;
                }
                let add = vblur_add_idx(y, r, strip_h);
                let rem = vblur_rem_idx(y, r, strip_h);
                sum_m1 = sum_m1 + bufs.mu1[add * width + x] - bufs.mu1[rem * width + x];
                sum_m2 = sum_m2 + bufs.mu2[add * width + x] - bufs.mu2[rem * width + x];
            }
        }
    }
    // Commit V-blur outputs into mask / mul_buf.
    bufs.mask = v_mu1_out;
    bufs.mul_buf = v_mu2_out;

    // === Swap mu1↔mask, mu2↔mul_buf (matches CPU) ===
    std::mem::swap(&mut bufs.mu1, &mut bufs.mask);
    std::mem::swap(&mut bufs.mu2, &mut bufs.mul_buf);

    // Snapshot mu1 AFTER swap. At inner rows this is V-blurred; at
    // overlap rows it's whatever was in bufs.mask before the V-blur.
    let mu1_post_swap_avg = row_averages(&bufs.mu1, width, strip_h);
    let mu1_post_swap_sample = row_samples(&bufs.mu1, width, strip_h, sample_col);

    // === Step 1: mask[..strip_n] = |src - bufs.mu1| ===
    for i in 0..strip_n {
        bufs.mask[i] = (src[i] - bufs.mu1[i]).abs();
    }
    let mask_step1_avg = row_averages(&bufs.mask, width, strip_h);
    let mask_step1_sample = row_samples(&bufs.mask, width, strip_h, sample_col);

    // === Step 2: mul_buf = box_blur(mask) via temp_blur ===
    // CPU uses `box_blur_1pass_into` which is H-blur followed by V-blur
    // (the standard separable box blur). We mirror that.
    h_blur_plane(&bufs.mask, width, strip_h, R, &mut bufs.temp_blur);
    v_blur_plane(&bufs.temp_blur, width, strip_h, R, &mut bufs.mul_buf);

    let activity_avg = row_averages(&bufs.mul_buf, width, strip_h);
    let activity_sample = row_samples(&bufs.mul_buf, width, strip_h, sample_col);

    // === Step 3: bufs.mask[inner] = mask_weight ===
    let inner_off_n = inner_off * width;
    let inner_n = inner_h * width;
    for i in 0..inner_n {
        let a = bufs.mul_buf[inner_off_n + i];
        bufs.mask[inner_off_n + i] = 1.0 / (1.0 + K_MASK * a);
    }

    // === Step 4: V-blur of sigma1_sq via temp_blur, then swap ===
    v_blur_plane(&bufs.sigma1_sq, width, strip_h, R, &mut bufs.temp_blur);
    std::mem::swap(&mut bufs.sigma1_sq, &mut bufs.temp_blur);
    v_blur_plane(&bufs.sigma12, width, strip_h, R, &mut bufs.temp_blur);
    std::mem::swap(&mut bufs.sigma12, &mut bufs.temp_blur);

    ChannelDump {
        width,
        strip_h,
        inner_off,
        inner_h,
        mu1_h_avg,
        mu1_post_swap_avg,
        mask_step1_avg,
        activity_avg,
        sample_col,
        mu1_h_sample,
        mu1_post_swap_sample,
        mask_step1_sample,
        activity_sample,
    }
}

fn row_averages(buf: &[f32], width: usize, h: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(h);
    for y in 0..h {
        let off = y * width;
        let s: f32 = buf[off..off + width].iter().sum();
        out.push(s / width as f32);
    }
    out
}
fn row_samples(buf: &[f32], width: usize, h: usize, col: usize) -> Vec<f32> {
    (0..h).map(|y| buf[y * width + col]).collect()
}

fn write_dump_tsv(path: &str, strip_top: usize, dump: &ChannelDump) {
    use std::fmt::Write as _;
    let mut s = String::new();
    writeln!(
        s,
        "sy\tgy\tis_inner\tmu1_h_avg\tmu1_post_swap_avg\tmask_step1_avg\tactivity_avg\tmu1_h_s\tmu1_post_swap_s\tmask_step1_s\tactivity_s"
    )
    .unwrap();
    for sy in 0..dump.strip_h {
        let gy = strip_top + sy;
        let is_inner = sy >= dump.inner_off && sy < dump.inner_off + dump.inner_h;
        writeln!(
            s,
            "{}\t{}\t{}\t{:+.6e}\t{:+.6e}\t{:+.6e}\t{:+.6e}\t{:+.6e}\t{:+.6e}\t{:+.6e}\t{:+.6e}",
            sy,
            gy,
            is_inner as u8,
            dump.mu1_h_avg[sy],
            dump.mu1_post_swap_avg[sy],
            dump.mask_step1_avg[sy],
            dump.activity_avg[sy],
            dump.mu1_h_sample[sy],
            dump.mu1_post_swap_sample[sy],
            dump.mask_step1_sample[sy],
            dump.activity_sample[sy],
        )
        .unwrap();
    }
    std::fs::write(path, s).expect("write tsv");
    println!("wrote {}", path);
}

// === GPU shadow: replays the masked_iw_strip_kernel logic, with the
//     same per-row dumps. The GPU computes activity differently at
//     overlap rows: it uses on-the-fly H-blur of the previous channel's
//     source (rather than CPU's nested-abs-diff stale state). ===

fn gpu_shadow_dump(
    label: &str,
    scale: usize,
    width: usize,
    height: usize,
    ref_planes: &[Vec<f32>; 3],
    // Per-channel V-blurred mu1 plane (image-wide), exactly as the GPU
    // reads from `persist_planes_ref`.
    persist_mu1: &[Vec<f32>; 3],
) -> Vec<(usize, usize, ChannelDump)> {
    let mut out = Vec::new();
    let n_strips = cpu_strip_count(height);
    for strip_idx in 0..n_strips {
        let (strip_top, _strip_bot, strip_h, _, inner_h, inner_off) =
            strip_layout(strip_idx, height);
        for ch in 0..3 {
            // GPU's mu1 at strip-row sy:
            //   inner row → persist_mu1[ch][gy]
            //   overlap row + ch==0 → 0
            //   overlap row + ch>=1 → H-blur of ref_planes[ch-1] at gy
            let mut mu1_strip = vec![0.0_f32; strip_h * width];
            let mut prev_h_blur_row = vec![0.0_f32; width];
            for sy in 0..strip_h {
                let gy = strip_top + sy;
                let is_inner = sy >= inner_off && sy < inner_off + inner_h;
                if is_inner {
                    let src_row = &persist_mu1[ch][gy * width..(gy + 1) * width];
                    mu1_strip[sy * width..(sy + 1) * width].copy_from_slice(src_row);
                } else if ch == 0 {
                    // zeros
                } else {
                    let prev_ref_row = &ref_planes[ch - 1][gy * width..(gy + 1) * width];
                    h_blur_row(prev_ref_row, width, R, &mut prev_h_blur_row);
                    mu1_strip[sy * width..(sy + 1) * width].copy_from_slice(&prev_h_blur_row);
                }
            }
            // src in strip-local coords.
            let src_strip =
                ref_planes[ch][strip_top * width..(strip_top + strip_h) * width].to_vec();

            // Step 1: |src - mu1|
            let mut mask = vec![0.0_f32; strip_h * width];
            for i in 0..mask.len() {
                mask[i] = (src_strip[i] - mu1_strip[i]).abs();
            }
            // Step 2: H-blur then V-blur (strip-local mirror).
            let mut temp = vec![0.0_f32; strip_h * width];
            let mut activity = vec![0.0_f32; strip_h * width];
            h_blur_plane(&mask, width, strip_h, R, &mut temp);
            v_blur_plane(&temp, width, strip_h, R, &mut activity);

            let mu1_h_avg = row_averages(&mu1_strip, width, strip_h); // same as mu1
            let mu1_post_swap_avg = mu1_h_avg.clone();
            let mask_step1_avg = row_averages(&mask, width, strip_h);
            let activity_avg = row_averages(&activity, width, strip_h);
            let sample_col = width / 2;
            let mu1_h_sample = row_samples(&mu1_strip, width, strip_h, sample_col);
            let mu1_post_swap_sample = mu1_h_sample.clone();
            let mask_step1_sample = row_samples(&mask, width, strip_h, sample_col);
            let activity_sample = row_samples(&activity, width, strip_h, sample_col);

            let dump = ChannelDump {
                width,
                strip_h,
                inner_off,
                inner_h,
                mu1_h_avg,
                mu1_post_swap_avg,
                mask_step1_avg,
                activity_avg,
                sample_col,
                mu1_h_sample,
                mu1_post_swap_sample,
                mask_step1_sample,
                activity_sample,
            };
            let path = format!(
                "/tmp/zensim_diag_{}_scale{}_strip{}_chan{}.tsv",
                label, scale, strip_idx, ch
            );
            write_dump_tsv(&path, strip_top, &dump);
            out.push((strip_idx, ch, dump));
        }
    }
    out
}

fn diff_dumps(
    cpu: &[(usize, usize, ChannelDump)],
    gpu: &[(usize, usize, ChannelDump)],
) -> Vec<(usize, usize, &'static str, usize, f32, f32, f32)> {
    let mut diffs = Vec::new();
    for ((sc, cc, cd), (sg, cg, gd)) in cpu.iter().zip(gpu.iter()) {
        assert_eq!(sc, sg);
        assert_eq!(cc, cg);
        // Compare per-row avgs for activity, mask_step1, mu1_post_swap.
        let cols: &[(&str, &Vec<f32>, &Vec<f32>)] = &[
            (
                "mu1_post_swap_avg",
                &cd.mu1_post_swap_avg,
                &gd.mu1_post_swap_avg,
            ),
            ("mask_step1_avg", &cd.mask_step1_avg, &gd.mask_step1_avg),
            ("activity_avg", &cd.activity_avg, &gd.activity_avg),
        ];
        for &(name, ca, ga) in cols {
            for sy in 0..ca.len() {
                let cv = ca[sy];
                let gv = ga[sy];
                let abs = (cv - gv).abs();
                let rel = abs / cv.abs().max(1e-6);
                diffs.push((*sc, *cc, name, sy, cv, gv, rel));
            }
        }
    }
    diffs
}

fn main() {
    let w = 64;
    let h = 64;
    let ref_rgb = gradient(w, h);
    let dist_rgb = add_noise(&ref_rgb, 8);

    let client = <Backend as Runtime>::client(&Default::default());
    let mut z = Zensim::<Backend>::new_with_regime(
        client,
        w as u32,
        h as u32,
        ZensimFeatureRegime::Extended,
    )
    .expect("create zensim");
    let _gpu_features = z
        .compute_features_vec(&ref_rgb, &dist_rgb)
        .expect("gpu run");

    // Read back the GPU's ref XYB planes and persist mu1 planes for scale 0.
    let scale = 0usize;
    let (padded_w_u32, h_u32) = z.debug_scale_dims(scale);
    let pw = padded_w_u32 as usize;
    let sh = h_u32 as usize;
    println!("scale 0: padded_w={}, h={}", pw, sh);

    // Read 3-channel ref_xyb for scale 0 (this is the SAME data the GPU
    // operated on).
    let ref_planes_padded: [Vec<f32>; 3] = [
        z.debug_read_xyb(scale, 0, true),
        z.debug_read_xyb(scale, 1, true),
        z.debug_read_xyb(scale, 2, true),
    ];
    let dis_planes_padded: [Vec<f32>; 3] = [
        z.debug_read_xyb(scale, 0, false),
        z.debug_read_xyb(scale, 1, false),
        z.debug_read_xyb(scale, 2, false),
    ];
    // persist mu1 planes (V-blurred, image-wide).
    let persist_mu1_padded: [Vec<f32>; 3] = [
        z.debug_read_persist_plane(scale, 0, 0),
        z.debug_read_persist_plane(scale, 1, 0),
        z.debug_read_persist_plane(scale, 2, 0),
    ];

    // The CPU's process_strip_channel works on the LOGICAL width
    // (since the CPU doesn't pad like GPU). At scale 0 of a 64×64 image,
    // padded_w = simd_padded_width(64). If pw > w, the CPU side won't
    // include those padding columns. For 64×64 we expect pw == 64.
    assert_eq!(pw, w, "expected scale 0 padded_w == w for this fixture");

    let cpu_dumps = cpu_shadow_dump("cpu", scale, w, sh, &ref_planes_padded, &dis_planes_padded);
    let gpu_dumps = gpu_shadow_dump("gpu", scale, w, sh, &ref_planes_padded, &persist_mu1_padded);

    let diffs = diff_dumps(&cpu_dumps, &gpu_dumps);

    // Summarize: max-abs-rel per (strip, channel, column).
    use std::collections::HashMap;
    let mut max_rel: HashMap<(usize, usize, &str), (usize, f32, f32, f32)> = HashMap::new();
    for (strip, ch, name, sy, cv, gv, rel) in &diffs {
        let key = (*strip, *ch, *name);
        let entry = max_rel.entry(key).or_insert((*sy, *cv, *gv, *rel));
        if *rel > entry.3 {
            *entry = (*sy, *cv, *gv, *rel);
        }
    }
    println!("\n=== Per-(strip,chan,column) MAX REL DIVERGENCE ===");
    let mut keys: Vec<_> = max_rel.keys().collect();
    keys.sort();
    for k in keys {
        let v = &max_rel[k];
        println!(
            "strip={} chan={} col={:24} sy={:2} cpu={:+.6e} gpu={:+.6e} rel={:.3e}",
            k.0, k.1, k.2, v.0, v.1, v.2, v.3
        );
    }

    // First diverging row per (strip, channel, column).
    println!("\n=== FIRST DIVERGENCE (rel > 1e-3) ===");
    let mut seen: std::collections::HashSet<(usize, usize, &str)> =
        std::collections::HashSet::new();
    for (strip, ch, name, sy, cv, gv, rel) in &diffs {
        let key = (*strip, *ch, *name);
        if seen.contains(&key) {
            continue;
        }
        if *rel > 1e-3 {
            println!(
                "strip={} chan={} col={:24} sy={:2} cpu={:+.6e} gpu={:+.6e} rel={:.3e}",
                strip, ch, name, sy, cv, gv, rel
            );
            seen.insert(key);
        }
    }
}
