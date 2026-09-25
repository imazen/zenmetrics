use crate::{Error, ModelVariant};
#[cfg(feature = "simd")]
use archmage::autoversion;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use archmage::{NeonToken, arcane};
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::{X64V3Token, arcane};

const BLOCK_SIZE: usize = 5;
const NUM_SCALES: u32 = 4;
const EIGENVALUE_EPS: f32 = 1e-6;
const EIGENVALUE_MAX_ITERS: usize = 500;
const SIGMA_NN: f32 = 0.19;
const NN_FLOOR: f32 = 0.1;
const WEIGHT_VAR_MODE: i32 = 5;
const SPEED_MAX_VAL: f32 = 45.0;
const KERNELSCALE: f32 = 1.0;
fn log2_2pi_e() -> f64 {
    (2.0 * std::f64::consts::PI * std::f64::consts::E).log2()
}

#[derive(Clone, Copy)]
struct Dims {
    original_height: usize,
    original_width: usize,
    scaled_height: usize,
    scaled_width: usize,
    alloc_height: usize,
    alloc_width: usize,
    block_size: usize,
    truncated_width: usize,
    truncated_height: usize,
    num_blocks_horizontal: usize,
    num_blocks_vertical: usize,
    num_blocks: usize,
    elements_in_block: usize,
    submatrix_width: usize,
    submatrix_height: usize,
}

impl Dims {
    fn new(w: usize, h: usize, prescale: f64) -> Result<Self, Error> {
        let mut d = Dims {
            original_height: h,
            original_width: w,
            scaled_height: (h as f64 * prescale + 0.5) as usize,
            scaled_width: (w as f64 * prescale + 0.5) as usize,
            alloc_height: 0,
            alloc_width: 0,
            block_size: BLOCK_SIZE,
            truncated_width: 0,
            truncated_height: 0,
            num_blocks_horizontal: 0,
            num_blocks_vertical: 0,
            num_blocks: 0,
            elements_in_block: 0,
            submatrix_width: 0,
            submatrix_height: 0,
        };
        d.alloc_height = d.original_height.max(d.scaled_height);
        d.alloc_width = d.original_width.max(d.scaled_width);
        let operating_height = d.scaled_height >> NUM_SCALES;
        let operating_width = d.scaled_width >> NUM_SCALES;
        d.truncated_width = (operating_width / d.block_size) * d.block_size;
        d.truncated_height = (operating_height / d.block_size) * d.block_size;
        d.num_blocks_horizontal = d.truncated_width / d.block_size;
        d.num_blocks_vertical = d.truncated_height / d.block_size;
        d.num_blocks = d.num_blocks_horizontal * d.num_blocks_vertical;
        d.elements_in_block = d.block_size * d.block_size;
        d.submatrix_width = d.truncated_width.wrapping_sub(d.block_size - 1);
        d.submatrix_height = d.truncated_height.wrapping_sub(d.block_size - 1);
        if d.truncated_height == 0 || d.truncated_width == 0 {
            return Err(Error::InvalidInput("operating dimensions too small"));
        }
        Ok(d)
    }
}

fn get_sign(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

fn pythagoras(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

fn compute_column_norm(a: &[f32], col: usize, start_row: usize, size: usize) -> f32 {
    let mut norm = 0.0f32;
    for i in start_row..size {
        norm += a[i * size + col] * a[i * size + col];
    }
    norm.sqrt()
}

fn compute_householder_transform(a: &mut [f32], col: usize, start_row: usize, size: usize) -> f32 {
    if size - start_row == 1 {
        return 0.0;
    }
    let xnorm = compute_column_norm(a, col, start_row + 1, size);
    if xnorm == 0.0 {
        return 0.0;
    }
    let alpha = a[start_row * size + col];
    let beta = -get_sign(alpha) * pythagoras(alpha, xnorm);
    let tau = (beta - alpha) / beta;
    let s = alpha - beta;
    if s != 0.0 {
        for i in start_row..size {
            a[i * size + col] /= s;
        }
        a[start_row * size + col] = beta;
    }
    tau
}

fn tridiagonal_multiply(
    a: &[f32],
    v: &[f32],
    x: &mut [f32],
    tau_i: f32,
    start: usize,
    size: usize,
) {
    for i in start..size {
        x[i] = 0.0;
        for j in start..size {
            x[i] += tau_i * a[i * size + j] * v[j];
        }
    }
}

fn tridiagonal_dot_product(x: &[f32], v: &[f32], start: usize, size: usize) -> f32 {
    let mut res = 0.0f32;
    for i in start..size {
        res += x[i] * v[i];
    }
    res
}

fn tridiagonal_axpy(x: &mut [f32], v: &[f32], alpha: f32, start: usize, size: usize) {
    for i in start..size {
        x[i] += alpha * v[i];
    }
}

fn tridiagonal_syr2(a: &mut [f32], x: &[f32], v: &[f32], start: usize, size: usize) {
    for i in start..size {
        for j in start..size {
            a[i * size + j] -= x[i] * v[j] + v[i] * x[j];
        }
    }
}

fn convert_to_tridiagonal(
    a: &mut [f32],
    size: usize,
    d: &mut [f32],
    sd: &mut [f32],
    buffer: &mut [f32],
) {
    let (v, rest) = buffer.split_at_mut(size);
    let x = &mut rest[..size];

    for i in 0..size.saturating_sub(2) {
        let tau_i = compute_householder_transform(a, i, i + 1, size);
        for j in i + 1..size {
            v[j] = a[j * size + i];
        }
        if tau_i != 0.0 {
            a[(i + 1) * size + i] = v[i + 1];
            v[i + 1] = 1.0;
            tridiagonal_multiply(a, v, x, tau_i, i + 1, size);
            let xv = tridiagonal_dot_product(x, v, i + 1, size);
            let alpha = -0.5 * tau_i * xv;
            tridiagonal_axpy(x, v, alpha, i + 1, size);
            tridiagonal_syr2(a, x, v, i + 1, size);
        }
    }

    for i in 0..size {
        d[i] = a[i * size + i];
    }
    for i in 0..size - 1 {
        sd[i] = a[(i + 1) * size + i];
    }
}

fn chop_small_elements(d: &[f32], sd: &mut [f32], size: usize) {
    for i in 0..size - 1 {
        if sd[i].abs() < EIGENVALUE_EPS * (d[i].abs() + d[i + 1].abs()) {
            sd[i] = 0.0;
        }
    }
}

fn trailing_eigenvalue(d: &[f32], sd: &[f32], n: usize) -> f32 {
    let ta = d[n - 2];
    let tb = d[n - 1];
    let tab = sd[n - 2];
    let dt = (ta - tb) / 2.0;
    if dt > 0.0 {
        tb - tab * (tab / (dt + pythagoras(dt, tab)))
    } else if dt == 0.0 {
        tb - tab.abs()
    } else {
        tb + tab * (tab / (-dt + pythagoras(dt, tab)))
    }
}

fn create_givens(a: f32, b: f32) -> (f32, f32) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let t = -a / b;
        let s1 = 1.0 / (1.0 + t * t).sqrt();
        (s1 * t, s1)
    } else {
        let t = -b / a;
        let c1 = 1.0 / (1.0 + t * t).sqrt();
        (c1, c1 * t)
    }
}

fn qr_step(d: &mut [f32], sd: &mut [f32], n: usize) {
    let mut mu = trailing_eigenvalue(d, sd, n);
    if EIGENVALUE_EPS * mu.abs() > d[0].abs() + sd[0].abs() {
        mu = 0.0;
    }
    let mut x = d[0] - mu;
    let mut z = sd[0];

    let mut ak: f32;
    let mut bk = 0.0f32;
    let mut zk = 0.0f32;

    let mut ap = d[0];
    let mut bp = sd[0];
    let mut aq = d[1];

    if n == 2 {
        let (c, s) = create_givens(x, z);
        let ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        let bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        let aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);
        ak = ap1;
        bk = bp1;
        ap = aq1;
        d[0] = ak;
        sd[0] = bk;
        d[1] = ap;
        return;
    }

    let mut bq = sd[1];
    for k in 0..n - 1 {
        let (c, s) = create_givens(x, z);
        let bk1 = c * bk - s * zk;
        let ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        let bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        let zp1 = -s * bq;
        let aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);
        let bq1 = c * bq;

        ak = ap1;
        bk = bp1;
        zk = zp1;
        ap = aq1;
        bp = bq1;
        if k < n - 2 {
            aq = d[k + 2];
        }
        if k < n - 3 {
            bq = sd[k + 2];
        }
        d[k] = ak;
        if k > 0 {
            sd[k - 1] = bk1;
        }
        if k < n - 2 {
            sd[k + 1] = bp;
        }
        x = bk;
        z = zk;
    }
    d[n - 1] = ap;
    sd[n - 2] = bk;
}

fn compute_eigenvalues_tridiagonal(
    d: &mut [f32],
    sd: &mut [f32],
    eigenvalues: &mut [f32],
    size: usize,
) {
    chop_small_elements(d, sd, size);
    let mut b = size - 1;
    let mut iter = 0;
    while b > 0 && iter < EIGENVALUE_MAX_ITERS {
        if sd[b - 1] == 0.0 {
            b -= 1;
            continue;
        }
        let mut a = b - 1;
        while a > 0 {
            if sd[a - 1] == 0.0 {
                break;
            }
            a -= 1;
        }
        let n_block = b - a + 1;
        qr_step(&mut d[a..], &mut sd[a..], n_block);
        chop_small_elements(&d[a..], &mut sd[a..], n_block);
        iter += 1;
    }
    eigenvalues[..size].copy_from_slice(&d[..size]);
}

fn compute_eigenvalues(
    a_immutable: &[f32],
    eigenvalues: &mut [f32],
    size: usize,
    buffer: &mut [f32],
) {
    let (a, rest) = buffer.split_at_mut(size * size);
    a.copy_from_slice(&a_immutable[..size * size]);
    let (d, rest) = rest.split_at_mut(size);
    let (sd, tmp) = rest.split_at_mut(size);
    if size == 1 {
        eigenvalues[0] = a[0];
        return;
    }
    convert_to_tridiagonal(a, size, d, sd, &mut tmp[..2 * size]);
    compute_eigenvalues_tridiagonal(d, sd, eigenvalues, size);
}

fn matrix_transpose(m: &mut [f32], size: usize) {
    for i in 0..size {
        for j in 0..i {
            m.swap(i * size + j, j * size + i);
        }
    }
}

fn matrix_identity(m: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            m[i * cols + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
}

fn matrix_mul(dst: &mut [f32], x: &[f32], y: &[f32], xrows: usize, xcols: usize, ycols: usize) {
    for v in dst.iter_mut() {
        *v = 0.0;
    }
    for i in 0..xrows {
        for k in 0..xcols {
            for j in 0..ycols {
                dst[i * ycols + j] += x[i * xcols + k] * y[k * ycols + j];
            }
        }
    }
}

fn matrix_minor(m: &mut [f32], size: usize, d: usize) {
    for i in 0..size {
        for j in 0..size {
            if i < d || j < d {
                m[i * size + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
}

fn matrix_identity_minus_v_vt(dst: &mut [f32], v: &[f32], size: usize) {
    for i in 0..size {
        for j in 0..size {
            dst[i * size + j] = -2.0 * v[i] * v[j];
        }
    }
    for i in 0..size {
        dst[i * size + i] += 1.0;
    }
}

fn vector_norm(x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    sum.sqrt()
}

fn matrix_qr_decomposition(
    a: &[f32],
    q: &mut [f32],
    r: &mut [f32],
    tmp_q: &mut [f32],
    tmp_z: &mut [f32],
    size: usize,
) {
    tmp_z.copy_from_slice(&a[..size * size]);
    matrix_identity(q, size, size);
    let mut vec = vec![0.0f32; size];
    let mut tmp_mul = vec![0.0f32; size * size];

    for k in 0..size - 1 {
        matrix_minor(tmp_z, size, k);
        for i in 0..size {
            vec[i] = tmp_z[i * size + k];
        }
        let norm = vector_norm(&vec);
        let sign = get_sign(a[k * size + k]);
        vec[k] += sign * norm;
        let vnorm = vector_norm(&vec);
        for v in vec.iter_mut() {
            *v /= vnorm;
        }
        matrix_identity_minus_v_vt(tmp_q, &vec, size);
        matrix_mul(&mut tmp_mul, tmp_q, tmp_z, size, size, size);
        tmp_z.copy_from_slice(&tmp_mul);
        matrix_mul(&mut tmp_mul, tmp_q, q, size, size, size);
        q.copy_from_slice(&tmp_mul);
    }

    matrix_mul(r, q, a, size, size, size);
    matrix_transpose(q, size);
}

fn solve_triangular_system(r: &[f32], x: &mut [f32], b: &[f32], size: usize, bcols: usize) -> bool {
    for i in (0..size).rev() {
        let denominator = r[i * size + i];
        if denominator.abs() < EIGENVALUE_EPS {
            return false;
        }
        for j in 0..bcols {
            let mut independent_term = b[i * bcols + j];
            for k in i + 1..size {
                independent_term -= x[k * bcols + j] * r[i * size + k];
            }
            x[i * bcols + j] = independent_term / denominator;
        }
    }
    true
}

fn solve_linear_system(
    a_data: &[f32],
    a_size: usize,
    b_data: &[f32],
    b_cols: usize,
    output: &mut [f32],
    buffer: &mut [f32],
) -> bool {
    let (a, rest) = buffer.split_at_mut(a_size * a_size);
    a.copy_from_slice(&a_data[..a_size * a_size]);
    let (q, rest) = rest.split_at_mut(a_size * a_size);
    let (r, rest) = rest.split_at_mut(a_size * a_size);
    let (tmp1, tmp2) = rest.split_at_mut(a_size * a_size);
    let mut tmp_rect = vec![0.0f32; a_size * b_cols];

    matrix_qr_decomposition(a, q, r, tmp1, tmp2, a_size);
    matrix_transpose(q, a_size);
    matrix_mul(&mut tmp_rect, q, b_data, a_size, a_size, b_cols);
    solve_triangular_system(r, output, &tmp_rect, a_size, b_cols)
}

fn compute_mean(
    dim: &Dims,
    data: &[f32],
    stride_px: usize,
    start_row: usize,
    start_col: usize,
) -> f32 {
    let mut result = 0.0f32;
    for i in 0..dim.submatrix_height {
        for j in 0..dim.submatrix_width {
            result += data[(start_row + i) * stride_px + (start_col + j)];
        }
    }
    result / (dim.submatrix_width * dim.submatrix_height) as f32
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
fn a8<T, const N: usize>(s: &[T]) -> &[T; N] {
    s.try_into().unwrap()
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
fn a8m<T, const N: usize>(s: &mut [T]) -> &mut [T; N] {
    s.try_into().unwrap()
}

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]
static FORCE_SCALAR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn v3_token() -> Option<X64V3Token> {
    if FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    <X64V3Token as archmage::SimdToken>::summon()
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
fn neon_token() -> Option<NeonToken> {
    if FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    <NeonToken as archmage::SimdToken>::summon()
}

/// Direct port of `convolution_f32_avx_s_1d_v_scanline`: 8-wide vertical
/// convolution, `tmp[j] = sum_k f[k] * src[k*stride + j]` over `0..wfloor8`.
/// `src` starts at the first tap row (caller offsets by `i - radius`).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn vif_filter1d_vrow_v3(
    _token: X64V3Token,
    f: &[f32],
    src: &[f32],
    stride_px: usize,
    tmp: &mut [f32],
    wfloor8: usize,
) {
    for j in (0..wfloor8).step_by(8) {
        let mut sum = _mm256_setzero_ps();
        for (k, &fk) in f.iter().enumerate() {
            let g = _mm256_loadu_ps(a8(&src[k * stride_px + j..k * stride_px + j + 8]));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_set1_ps(fk), g));
        }
        _mm256_storeu_ps(a8m(&mut tmp[j..j + 8]), sum);
    }
}

/// Direct port of `convolution_f32_avx_s_1d_h_scanline`: 8-wide horizontal
/// convolution, `dst[j + radius] = sum_k f[k] * tmp[j + k]` over `0..j_end`.
/// The loads reach `j_end + 2*radius - 2`; `tmp` must have that much slack.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn vif_filter1d_hrow_v3(
    _token: X64V3Token,
    f: &[f32],
    tmp: &[f32],
    dst: &mut [f32],
    radius: usize,
    j_end: usize,
) {
    for j in (0..j_end).step_by(8) {
        let mut sum = _mm256_setzero_ps();
        for (k, &fk) in f.iter().enumerate() {
            let g = _mm256_loadu_ps(a8(&tmp[j + k..j + k + 8]));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_set1_ps(fk), g));
        }
        _mm256_storeu_ps(a8m(&mut dst[j + radius..j + radius + 8]), sum);
    }
}

/// NEON port of `vif_filter1d_vrow_v3`: 8-wide vertical convolution as two
/// f32x4 halves. Separate mul+add keeps the scalar loop's two-rounding
/// arithmetic bit-exact (an FMA would be single-rounding).
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[arcane(import_intrinsics)]
fn vif_filter1d_vrow_neon(
    _token: NeonToken,
    f: &[f32],
    src: &[f32],
    stride_px: usize,
    tmp: &mut [f32],
    wfloor8: usize,
) {
    for j in (0..wfloor8).step_by(8) {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        for (k, &fk) in f.iter().enumerate() {
            let w = vdupq_n_f32(fk);
            let g0 = vld1q_f32(a8(&src[k * stride_px + j..k * stride_px + j + 4]));
            let g1 = vld1q_f32(a8(&src[k * stride_px + j + 4..k * stride_px + j + 8]));
            sum0 = vaddq_f32(sum0, vmulq_f32(w, g0));
            sum1 = vaddq_f32(sum1, vmulq_f32(w, g1));
        }
        vst1q_f32(a8m(&mut tmp[j..j + 4]), sum0);
        vst1q_f32(a8m(&mut tmp[j + 4..j + 8]), sum1);
    }
}

/// NEON port of `vif_filter1d_hrow_v3`: `dst[j + radius] = sum_k f[k] *
/// tmp[j + k]` over `0..j_end`, two f32x4 halves, mul+add (see vrow note).
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[arcane(import_intrinsics)]
fn vif_filter1d_hrow_neon(
    _token: NeonToken,
    f: &[f32],
    tmp: &[f32],
    dst: &mut [f32],
    radius: usize,
    j_end: usize,
) {
    for j in (0..j_end).step_by(8) {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        for (k, &fk) in f.iter().enumerate() {
            let w = vdupq_n_f32(fk);
            let g0 = vld1q_f32(a8(&tmp[j + k..j + k + 4]));
            let g1 = vld1q_f32(a8(&tmp[j + k + 4..j + k + 8]));
            sum0 = vaddq_f32(sum0, vmulq_f32(w, g0));
            sum1 = vaddq_f32(sum1, vmulq_f32(w, g1));
        }
        vst1q_f32(a8m(&mut dst[j + radius..j + radius + 4]), sum0);
        vst1q_f32(a8m(&mut dst[j + radius + 4..j + radius + 8]), sum1);
    }
}

/// NEON port of `compute_covariance_v3`: same two-chain f64 accumulator
/// structure, two f64x2 lanes per chain step (4 f64 lanes per iteration vs
/// AVX2's 8, then 2-wide and scalar tails). FMA rounding matches.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn compute_covariance_neon(
    _token: NeonToken,
    data: &[f32],
    mean_x: f64,
    mean_y: f64,
    stride_px: usize,
    srx: usize,
    scx: usize,
    sry: usize,
    scy: usize,
    sub_w: usize,
    sub_h: usize,
) -> f64 {
    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mx = vdupq_n_f64(mean_x);
    let my = vdupq_n_f64(mean_y);
    let mut scalar_tail = 0.0f64;
    for i in 0..sub_h {
        let xb = (srx + i) * stride_px + scx;
        let yb = (sry + i) * stride_px + scy;
        let mut j = 0usize;
        while j + 4 <= sub_w {
            let cx0 = vsubq_f64(vcvt_f64_f32(vld1_f32(a8(&data[xb + j..xb + j + 2]))), mx);
            let cx1 = vsubq_f64(
                vcvt_f64_f32(vld1_f32(a8(&data[xb + j + 2..xb + j + 4]))),
                mx,
            );
            let cy0 = vsubq_f64(vcvt_f64_f32(vld1_f32(a8(&data[yb + j..yb + j + 2]))), my);
            let cy1 = vsubq_f64(
                vcvt_f64_f32(vld1_f32(a8(&data[yb + j + 2..yb + j + 4]))),
                my,
            );
            acc0 = vfmaq_f64(acc0, cx0, cy0);
            acc1 = vfmaq_f64(acc1, cx1, cy1);
            j += 4;
        }
        while j + 2 <= sub_w {
            let cx = vsubq_f64(vcvt_f64_f32(vld1_f32(a8(&data[xb + j..xb + j + 2]))), mx);
            let cy = vsubq_f64(vcvt_f64_f32(vld1_f32(a8(&data[yb + j..yb + j + 2]))), my);
            acc0 = vfmaq_f64(acc0, cx, cy);
            j += 2;
        }
        while j < sub_w {
            scalar_tail += (data[xb + j] as f64 - mean_x) * (data[yb + j] as f64 - mean_y);
            j += 1;
        }
    }
    let acc = vaddq_f64(acc0, acc1);
    let mut tmp = [0.0f64; 2];
    vst1q_f64(a8m(&mut tmp), acc);
    tmp[0] + tmp[1] + scalar_tail
}

/// Direct port of `compute_cov_kernel_avx2`: f32 -> f64 widening, two
/// parallel f64 accumulator chains hiding FMA latency, 8-wide then 4-wide
/// then scalar per row. Returns the unnormalized sum (caller divides).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
#[allow(clippy::too_many_arguments)]
fn compute_covariance_v3(
    _token: X64V3Token,
    data: &[f32],
    mean_x: f64,
    mean_y: f64,
    stride_px: usize,
    srx: usize,
    scx: usize,
    sry: usize,
    scy: usize,
    sub_w: usize,
    sub_h: usize,
) -> f64 {
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mx = _mm256_set1_pd(mean_x);
    let my = _mm256_set1_pd(mean_y);
    let mut scalar_tail = 0.0f64;
    for i in 0..sub_h {
        let xb = (srx + i) * stride_px + scx;
        let yb = (sry + i) * stride_px + scy;
        let mut j = 0usize;
        while j + 8 <= sub_w {
            let cx0 = _mm256_sub_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a8(&data[xb + j..xb + j + 4]))),
                mx,
            );
            let cx1 = _mm256_sub_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a8(&data[xb + j + 4..xb + j + 8]))),
                mx,
            );
            let cy0 = _mm256_sub_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a8(&data[yb + j..yb + j + 4]))),
                my,
            );
            let cy1 = _mm256_sub_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a8(&data[yb + j + 4..yb + j + 8]))),
                my,
            );
            acc0 = _mm256_fmadd_pd(cx0, cy0, acc0);
            acc1 = _mm256_fmadd_pd(cx1, cy1, acc1);
            j += 8;
        }
        while j + 4 <= sub_w {
            let cx = _mm256_sub_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a8(&data[xb + j..xb + j + 4]))),
                mx,
            );
            let cy = _mm256_sub_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a8(&data[yb + j..yb + j + 4]))),
                my,
            );
            acc0 = _mm256_fmadd_pd(cx, cy, acc0);
            j += 4;
        }
        while j < sub_w {
            scalar_tail += (data[xb + j] as f64 - mean_x) * (data[yb + j] as f64 - mean_y);
            j += 1;
        }
    }
    let acc = _mm256_add_pd(acc0, acc1);
    let mut tmp = [0.0f64; 4];
    _mm256_storeu_pd(&mut tmp, acc);
    tmp[0] + tmp[1] + tmp[2] + tmp[3] + scalar_tail
}

fn compute_covariance(
    dim: &Dims,
    data: &[f32],
    means: &[f32],
    stride_px: usize,
    srx: usize,
    scx: usize,
    sry: usize,
    scy: usize,
) -> f32 {
    let mean_x = means[srx * dim.block_size + scx] as f64;
    let mean_y = means[sry * dim.block_size + scy] as f64;
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if let Some(token) = <X64V3Token as archmage::SimdToken>::summon() {
        let result = compute_covariance_v3(
            token,
            data,
            mean_x,
            mean_y,
            stride_px,
            srx,
            scx,
            sry,
            scy,
            dim.submatrix_width,
            dim.submatrix_height,
        );
        return (result / (dim.submatrix_width * dim.submatrix_height) as f64) as f32;
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if let Some(token) = neon_token() {
        let result = compute_covariance_neon(
            token,
            data,
            mean_x,
            mean_y,
            stride_px,
            srx,
            scx,
            sry,
            scy,
            dim.submatrix_width,
            dim.submatrix_height,
        );
        return (result / (dim.submatrix_width * dim.submatrix_height) as f64) as f32;
    }
    let mut result = 0.0f64;
    for i in 0..dim.submatrix_height {
        for j in 0..dim.submatrix_width {
            let val_x = data[(srx + i) * stride_px + scx + j] as f64;
            let val_y = data[(sry + i) * stride_px + scy + j] as f64;
            result += (val_x - mean_x) * (val_y - mean_y);
        }
    }
    (result / (dim.submatrix_width * dim.submatrix_height) as f64) as f32
}

fn compute_covariance_matrix(
    dim: &Dims,
    data: &[f32],
    cov_mat: &mut [f32],
    means: &mut [f32],
    stride_px: usize,
) {
    for sr in 0..dim.block_size {
        for sc in 0..dim.block_size {
            means[sr * dim.block_size + sc] = compute_mean(dim, data, stride_px, sr, sc);
        }
    }
    let e = dim.elements_in_block;
    for x in 0..e {
        for y in 0..=x {
            let cov = compute_covariance(
                dim,
                data,
                means,
                stride_px,
                x / dim.block_size,
                x % dim.block_size,
                y / dim.block_size,
                y % dim.block_size,
            );
            cov_mat[x * e + y] = cov;
            cov_mat[y * e + x] = cov;
        }
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn compute_independent_term(
    dim: &Dims,
    data: &[f32],
    independent_term: &mut [f32],
    stride_px: usize,
) {
    for si in 0..dim.block_size {
        for sj in 0..dim.block_size {
            let mut i = si;
            while i < dim.truncated_height {
                let mut j = sj;
                while j < dim.truncated_width {
                    let out_row = si * dim.block_size + sj;
                    let out_col = ((i - si) / dim.block_size) * dim.num_blocks_horizontal
                        + (j - sj) / dim.block_size;
                    independent_term[out_row * dim.num_blocks + out_col] = data[i * stride_px + j];
                    j += dim.block_size;
                }
                i += dim.block_size;
            }
        }
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn update_entropy(dim: &Dims, entropy: &mut [f32], s: &[f32], l: f32, sigma_nn: f32) {
    for i in 0..dim.num_blocks_vertical {
        for j in 0..dim.num_blocks_horizontal {
            let idx = i * dim.num_blocks_horizontal + j;
            entropy[idx] = (entropy[idx] as f64
                + ((l * s[idx] + sigma_nn) as f64).log2()
                + log2_2pi_e()) as f32;
        }
    }
}

struct EstResult {
    entropies: Vec<f32>,
    variances: Vec<f32>,
    singular: bool,
}

fn est_params(dim: &Dims, data: &[f32], stride_px: usize, sigma_nn: f32) -> EstResult {
    let e = dim.elements_in_block;
    let mut cov_mat = vec![0.0f32; e * e];
    let mut eigenvalues = vec![0.0f32; e];
    let mut means = vec![0.0f32; e];
    compute_covariance_matrix(dim, data, &mut cov_mat, &mut means, stride_px);

    let mut eig_buffer = vec![0.0f32; e * e + 4 * e];
    compute_eigenvalues(&cov_mat, &mut eigenvalues, e, &mut eig_buffer);

    let mut independent_term = vec![0.0f32; e * dim.num_blocks];
    compute_independent_term(dim, data, &mut independent_term, stride_px);

    let mut linear_system_sol = vec![0.0f32; e * dim.num_blocks];
    let regular = eigenvalues.iter().all(|&v| v >= EIGENVALUE_EPS);
    let mut solved = false;
    if regular {
        let mut qr_buffer = vec![0.0f32; 5 * e * e];
        solved = solve_linear_system(
            &cov_mat,
            e,
            &independent_term,
            dim.num_blocks,
            &mut linear_system_sol,
            &mut qr_buffer,
        );
    }
    let cannot_invert = !regular || !solved;
    if cannot_invert {
        linear_system_sol.fill(0.0);
    }

    for i in 0..e {
        for j in 0..dim.num_blocks {
            linear_system_sol[i * dim.num_blocks + j] = (linear_system_sol[i * dim.num_blocks + j]
                * independent_term[i * dim.num_blocks + j])
                / e as f32;
        }
    }
    for i in 1..e {
        for j in 0..dim.num_blocks {
            linear_system_sol[j] += linear_system_sol[i * dim.num_blocks + j];
        }
    }

    let mut entropies = vec![0.0f32; dim.num_blocks];
    for &ev in eigenvalues[..e].iter() {
        let l = ev.max(0.0);
        update_entropy(dim, &mut entropies, &linear_system_sol, l, sigma_nn);
    }
    let variances = linear_system_sol[..dim.num_blocks].to_vec();

    EstResult {
        entropies,
        variances,
        singular: cannot_invert,
    }
}

fn get_speed_score(dim: &Dims, ref_r: &EstResult, dis_r: &EstResult) -> f32 {
    let base_entropy = (dim.elements_in_block as f64
        * ((((1.0f32 + NN_FLOOR) * SIGMA_NN) as f64).log2() + log2_2pi_e()))
        as f32;
    let mut score = 0.0f32;
    for i in 0..dim.num_blocks {
        if ref_r.entropies[i] < base_entropy && dis_r.entropies[i] < base_entropy {
            continue;
        }
        let (spatial_ref, spatial_dis) = if WEIGHT_VAR_MODE == 5 {
            (
                (ref_r.entropies[i] as f64 * ((1.0f32 + ref_r.variances[i]) as f64).log2()) as f32,
                (dis_r.entropies[i] as f64
                    * (1.0 + (0.75 * ref_r.variances[i] as f64 + 0.25 * dis_r.variances[i] as f64))
                        .log2()) as f32,
            )
        } else {
            return f32::NAN;
        };
        score += (spatial_ref - spatial_dis).abs();
    }
    score / dim.num_blocks as f32
}

#[cfg_attr(feature = "simd", autoversion)]
fn subtract_image(im1: &mut [f32], im2: &[f32], w: usize, h: usize, stride_px: usize) {
    for i in 0..h {
        for j in 0..w {
            im1[i * stride_px + j] -= im2[i * stride_px + j];
        }
    }
}

fn round_up_to_odd(f: f32) -> i32 {
    let ceiling = f.ceil() as i32;
    if ceiling % 2 == 0 {
        ceiling + 1
    } else {
        ceiling
    }
}

fn gaussian_kernel(size: usize, stdev: f32) -> Vec<f32> {
    let k = (size - 1) / 2;
    let mut out = vec![0.0f32; size];
    let mut sum = 0.0f32;
    for (i, v) in out.iter_mut().enumerate() {
        let x = i as f64 - k as f64;
        let num = (-0.5 * x / stdev as f64 * x / stdev as f64).exp() as f32;
        let den = (1.0 / (stdev as f64 * (2.0 * std::f64::consts::PI).sqrt())) as f32;
        *v = num / den;
        sum += *v;
    }
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

fn vif_get_filter_size(scale: u32, kernelscale: f32) -> usize {
    let n = ((1u32 << (4 - scale)) + 1) as f32;
    (round_up_to_odd(n * kernelscale)).max(3) as usize
}

fn mirror_i32(idx: i32, size: usize) -> usize {
    let size = size as i32;
    if idx < 0 {
        (-idx) as usize
    } else if idx >= size {
        (2 * size - idx - 2) as usize
    } else {
        idx as usize
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn vif_filter1d(f: &[f32], src: &[f32], dst: &mut [f32], w: usize, h: usize, stride_px: usize) {
    let fwidth = f.len();
    let radius = fwidth / 2;
    // +2*radius slack: the AVX2 horizontal scanline's 8-wide loads reach
    // tmp[j + fwidth + 6] with j up to floorn(w - radius, 8) - 8, the same
    // headroom libvmaf's ceil(w, 8)-strided tmp provides.
    let mut tmp = vec![0.0f32; w + 2 * radius];
    for i in 0..h {
        #[cfg(feature = "simd")]
        #[allow(unused_mut)]
        let mut j = 0usize;
        #[cfg(not(feature = "simd"))]
        let j = 0usize;
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if i >= radius && i + radius < h {
            if let Some(t) = v3_token() {
                let wfloor8 = w / 8 * 8;
                let base = (i - radius) * stride_px;
                vif_filter1d_vrow_v3(
                    t,
                    f,
                    &src[base..base + (fwidth - 1) * stride_px + wfloor8],
                    stride_px,
                    &mut tmp,
                    wfloor8,
                );
                j = wfloor8;
            }
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if i >= radius && i + radius < h {
            if let Some(t) = neon_token() {
                let wfloor8 = w / 8 * 8;
                let base = (i - radius) * stride_px;
                vif_filter1d_vrow_neon(
                    t,
                    f,
                    &src[base..base + (fwidth - 1) * stride_px + wfloor8],
                    stride_px,
                    &mut tmp,
                    wfloor8,
                );
                j = wfloor8;
            }
        }
        for j in j..w {
            let mut accum = 0.0f32;
            for (fi, &fc) in f.iter().enumerate() {
                let ii = mirror_i32(i as i32 - radius as i32 + fi as i32, h);
                accum += fc * src[ii * stride_px + j];
            }
            tmp[j] = accum;
        }
        for j in 0..radius.min(w) {
            let mut accum = 0.0f32;
            for (fj, &fc) in f.iter().enumerate() {
                let jj = mirror_i32(j as i32 - radius as i32 + fj as i32, w);
                accum += fc * tmp[jj];
            }
            dst[i * stride_px + j] = accum;
        }
        #[cfg(feature = "simd")]
        #[allow(unused_mut)]
        let mut jj = 0usize;
        #[cfg(not(feature = "simd"))]
        let jj = 0usize;
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if let Some(t) = v3_token() {
            let j_end = w.saturating_sub(radius) / 8 * 8;
            vif_filter1d_hrow_v3(
                t,
                f,
                &tmp,
                &mut dst[i * stride_px..i * stride_px + w],
                radius,
                j_end,
            );
            jj = j_end;
        }
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if let Some(t) = neon_token() {
            let j_end = w.saturating_sub(radius) / 8 * 8;
            vif_filter1d_hrow_neon(
                t,
                f,
                &tmp,
                &mut dst[i * stride_px..i * stride_px + w],
                radius,
                j_end,
            );
            jj = j_end;
        }
        for j in jj.max(radius.min(w))..w {
            let mut accum = 0.0f32;
            for (fj, &fc) in f.iter().enumerate() {
                let jm = mirror_i32(j as i32 - radius as i32 + fj as i32, w);
                accum += fc * tmp[jm];
            }
            dst[i * stride_px + j] = accum;
        }
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn vif_dec16(src: &[f32], dst: &mut [f32], w: usize, h: usize, stride_px: usize) {
    for i in 0..h / 16 {
        for j in 0..w / 16 {
            dst[i * stride_px + j] = src[(i * 16) * stride_px + j * 16];
        }
    }
}

fn mirror_f32(i: f32, left: f32, right: f32) -> f32 {
    if i < left {
        -i
    } else if i > right {
        2.0 * right - i
    } else {
        i
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn bilinear_scale(
    src: &[f32],
    dst: &mut [f32],
    src_w: usize,
    src_h: usize,
    stride_px: usize,
    dst_w: usize,
    dst_h: usize,
) {
    if src_w == dst_w && src_h == dst_h {
        dst[..dst_h * stride_px].copy_from_slice(&src[..dst_h * stride_px]);
        return;
    }
    let ratio_x = src_w as f32 / dst_w as f32;
    let ratio_y = src_h as f32 / dst_h as f32;
    let mut x1a = vec![0usize; dst_w];
    let mut x2a = vec![0usize; dst_w];
    let mut dxa = vec![0.0f32; dst_w];
    for x in 0..dst_w {
        let xx = (x as f32 + 0.5) * ratio_x - 0.5;
        x1a[x] = mirror_f32(xx.floor(), 0.0, (src_w - 1) as f32) as usize;
        x2a[x] = mirror_f32(xx.ceil(), 0.0, (src_w - 1) as f32) as usize;
        dxa[x] = xx - x1a[x] as f32;
    }
    for y in 0..dst_h {
        let yy = (y as f32 + 0.5) * ratio_y - 0.5;
        let y1 = mirror_f32(yy.floor(), 0.0, (src_h - 1) as f32) as usize;
        let y2 = mirror_f32(yy.ceil(), 0.0, (src_h - 1) as f32) as usize;
        let dy = yy - y1 as f32;
        let r1 = &src[y1 * stride_px..];
        let r2 = &src[y2 * stride_px..];
        for x in 0..dst_w {
            let dx = dxa[x];
            dst[y * stride_px + x] = (1.0 - dy) * (1.0 - dx) * r1[x1a[x]]
                + (1.0 - dy) * dx * r1[x2a[x]]
                + dy * (1.0 - dx) * r2[x1a[x]]
                + dy * dx * r2[x2a[x]];
        }
    }
}

#[cfg_attr(feature = "simd", autoversion)]
fn filter_and_downscale(dim: &Dims, prescale: f64, frame_buffer: &mut [f32], stride_px: usize) {
    let frame_size = stride_px * dim.alloc_height;
    let mut tmp = vec![0.0f32; 2 * frame_size];
    let (curr_scale, tmpbuf) = tmp.split_at_mut(frame_size);

    if (prescale - 1.0).abs() >= 1.0e-3 {
        tmpbuf[..frame_size].copy_from_slice(&frame_buffer[..frame_size]);
        bilinear_scale(
            tmpbuf,
            frame_buffer,
            dim.original_width,
            dim.original_height,
            stride_px,
            dim.scaled_width,
            dim.scaled_height,
        );
    }

    let filter_width_antialias = vif_get_filter_size(1, KERNELSCALE);
    let filter_antialias = gaussian_kernel(
        filter_width_antialias,
        (NUM_SCALES as f32).sqrt() * filter_width_antialias as f32 / 5.0,
    );
    vif_filter1d(
        &filter_antialias,
        frame_buffer,
        curr_scale,
        dim.scaled_width,
        dim.scaled_height,
        stride_px,
    );

    vif_dec16(
        curr_scale,
        frame_buffer,
        dim.scaled_width,
        dim.scaled_height,
        stride_px,
    );

    let downscaled_w = dim.scaled_width >> NUM_SCALES;
    let downscaled_h = dim.scaled_height >> NUM_SCALES;
    let filter_width = vif_get_filter_size(NUM_SCALES, KERNELSCALE);
    let filter = gaussian_kernel(filter_width, filter_width as f32 / 5.0);
    vif_filter1d(
        &filter,
        frame_buffer,
        curr_scale,
        downscaled_w,
        downscaled_h,
        stride_px,
    );
    subtract_image(
        frame_buffer,
        curr_scale,
        downscaled_w,
        downscaled_h,
        stride_px,
    );
}

fn channel_score(
    dim: &Dims,
    prescale: f64,
    ref_plane: &[u16],
    dis_plane: &[u16],
    cw: usize,
    ch: usize,
    bit_depth: u8,
) -> (f32, bool) {
    let stride_px = dim.alloc_width;
    let npix = stride_px * dim.alloc_height;
    let mut ref_buf = vec![0.0f32; npix];
    let mut dis_buf = vec![0.0f32; npix];
    let (scaler, offset): (f32, f32) = match bit_depth {
        10 => (4.0, -128.0),
        12 => (16.0, -128.0),
        16 => (256.0, -128.0),
        _ => (1.0, -128.0),
    };
    for i in 0..ch {
        for j in 0..cw {
            ref_buf[i * stride_px + j] = ref_plane[i * cw + j] as f32 / scaler + offset;
            dis_buf[i * stride_px + j] = dis_plane[i * cw + j] as f32 / scaler + offset;
        }
    }

    filter_and_downscale(dim, prescale, &mut ref_buf, stride_px);
    let ref_r = est_params(dim, &ref_buf, stride_px, SIGMA_NN);
    filter_and_downscale(dim, prescale, &mut dis_buf, stride_px);
    let dis_r = est_params(dim, &dis_buf, stride_px, SIGMA_NN);

    let score = if (ref_r.singular && !dis_r.singular) || (!ref_r.singular && dis_r.singular) {
        0.0
    } else {
        get_speed_score(dim, &ref_r, &dis_r)
    };
    (score, ref_r.singular || dis_r.singular)
}

fn prescale_for(variant: ModelVariant) -> f64 {
    match variant {
        ModelVariant::Phone | ModelVariant::HfrPhone => 0.6,
        ModelVariant::Consumer4k | ModelVariant::HfrConsumer4k => 0.5,
        _ => 1.0,
    }
}

pub fn speed_v1_chroma_420(
    reference_u: &[u16],
    reference_v: &[u16],
    distorted_u: &[u16],
    distorted_v: &[u16],
    width: usize,
    height: usize,
    bit_depth: u8,
    variant: ModelVariant,
) -> Result<f64, Error> {
    if !matches!(bit_depth, 8 | 10 | 12 | 16) {
        return Err(Error::InvalidInput("unsupported bit depth"));
    }
    if width == 0 || height == 0 {
        return Err(Error::InvalidInput("zero dimension"));
    }
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(Error::InvalidInput("odd luma dimension"));
    }
    let cw = width / 2;
    let ch = height / 2;
    let npix = cw
        .checked_mul(ch)
        .ok_or(Error::InvalidInput("dimension overflow"))?;
    for plane in [reference_u, reference_v, distorted_u, distorted_v] {
        if plane.len() != npix {
            return Err(Error::InvalidInput("plane length mismatch"));
        }
        let max_sample = (1u32 << bit_depth) - 1;
        if plane.iter().any(|&v| v as u32 > max_sample) {
            return Err(Error::InvalidInput("sample exceeds bit depth"));
        }
    }

    let prescale = prescale_for(variant);
    let dim = Dims::new(cw, ch, prescale)?;
    dim.alloc_width
        .checked_mul(dim.alloc_height)
        .ok_or(Error::InvalidInput("dimension overflow"))?;

    let (score_u, err_u) =
        channel_score(&dim, prescale, reference_u, distorted_u, cw, ch, bit_depth);
    let (score_v, err_v) =
        channel_score(&dim, prescale, reference_v, distorted_v, cw, ch, bit_depth);

    let score_uv = if err_u && !err_v {
        score_v
    } else if err_v && !err_u {
        score_u
    } else {
        (score_u + score_v) / 2.0
    };

    Ok((score_uv.min(SPEED_MAX_VAL)) as f64)
}

#[cfg(all(
    test,
    feature = "simd",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod tests {
    use super::*;

    /// Scalar reference of the covariance kernel (two-rounding mul+add),
    /// compared against the FMA intrinsic within tight relative tolerance —
    /// FMA is single-rounding so bit-exact equality is not expected.
    fn cov_scalar(
        data: &[f32],
        mean_x: f64,
        mean_y: f64,
        stride: usize,
        srx: usize,
        scx: usize,
        sry: usize,
        scy: usize,
        w: usize,
        h: usize,
    ) -> f64 {
        let mut result = 0.0f64;
        for i in 0..h {
            for j in 0..w {
                result += (data[(srx + i) * stride + scx + j] as f64 - mean_x)
                    * (data[(sry + i) * stride + scy + j] as f64 - mean_y);
            }
        }
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_covariance_matches_scalar() {
        let Some(token) = <X64V3Token as archmage::SimdToken>::summon() else {
            return;
        };
        let stride = 64usize;
        let data: Vec<f32> = (0..stride * 40)
            .map(|i| ((i * 37 + (i / stride) * 11) % 997) as f32 * 0.125 - 60.0)
            .collect();
        for (w, h) in [(8usize, 4usize), (9, 5), (13, 7), (16, 8), (21, 9), (5, 11)] {
            for (srx, scx, sry, scy) in [(0usize, 0usize, 1usize, 2usize), (2, 3, 4, 0)] {
                let mx = 12.5f64;
                let my = -7.25f64;
                let avx2 =
                    compute_covariance_v3(token, &data, mx, my, stride, srx, scx, sry, scy, w, h);
                let scalar = cov_scalar(&data, mx, my, stride, srx, scx, sry, scy, w, h);
                let denom = scalar.abs().max(1.0);
                assert!(
                    (avx2 - scalar).abs() / denom <= 1e-12,
                    "w={w} h={h}: avx2={avx2} scalar={scalar}"
                );
            }
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn neon_covariance_matches_scalar() {
        let Some(token) = <NeonToken as archmage::SimdToken>::summon() else {
            return;
        };
        let stride = 64usize;
        let data: Vec<f32> = (0..stride * 40)
            .map(|i| ((i * 37 + (i / stride) * 11) % 997) as f32 * 0.125 - 60.0)
            .collect();
        for (w, h) in [(8usize, 4usize), (9, 5), (13, 7), (16, 8), (21, 9), (5, 11)] {
            for (srx, scx, sry, scy) in [(0usize, 0usize, 1usize, 2usize), (2, 3, 4, 0)] {
                let mx = 12.5f64;
                let my = -7.25f64;
                let neon =
                    compute_covariance_neon(token, &data, mx, my, stride, srx, scx, sry, scy, w, h);
                let scalar = cov_scalar(&data, mx, my, stride, srx, scx, sry, scy, w, h);
                let denom = scalar.abs().max(1.0);
                assert!(
                    (neon - scalar).abs() / denom <= 1e-12,
                    "w={w} h={h}: neon={neon} scalar={scalar}"
                );
            }
        }
    }

    /// `vif_filter1d_vrow_v3`/`vif_filter1d_hrow_v3` (AVX mul+add, no FMA)
    /// and the `_neon` ports must bit-match the scalar mirrored-edge
    /// separable filter. Runs the whole `vif_filter1d` with and without the
    /// vector tier so dispatch coverage equals production.
    #[test]
    fn simd_vif_filter1d_matches_scalar_for_tails_and_edges() {
        #[cfg(target_arch = "x86_64")]
        if <X64V3Token as archmage::SimdToken>::summon().is_none() {
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if <NeonToken as archmage::SimdToken>::summon().is_none() {
            return;
        }
        use std::sync::atomic::Ordering;
        let run = |f: &[f32], w: usize, h: usize, stride: usize, scalar: bool| {
            FORCE_SCALAR.store(scalar, Ordering::Relaxed);
            let src: Vec<f32> = (0..stride * h)
                .map(|i| ((i * 7919 + (i / stride) * 313) % 1021) as f32 * 0.5 - 255.0)
                .collect();
            let mut dst = vec![0f32; stride * h];
            vif_filter1d(f, &src, &mut dst, w, h, stride);
            FORCE_SCALAR.store(false, Ordering::Relaxed);
            dst
        };
        // fwidth 5/9 are the SpEED antialias/downscale filter sizes; 3 covers
        // an odd-small radius; 1 is the degenerate no-op tap.
        for fwidth in [1usize, 3, 5, 9] {
            let f: Vec<f32> = gaussian_kernel(fwidth, fwidth as f32 / 5.0 + 0.25);
            for (w, h, stride) in [
                (8usize, 4usize, 8usize),
                (9, 5, 9),
                (17, 11, 20),
                (21, 3, 21),
                (40, 40, 40),
                (34, 7, 40),
                (33, 33, 33),
                (16, 2, 16),
            ] {
                if f.len() / 2 >= w.min(h) {
                    continue;
                }
                let s = run(&f, w, h, stride, true);
                let v = run(&f, w, h, stride, false);
                assert_eq!(s, v, "fwidth={fwidth} {w}x{h} stride={stride}");
            }
        }
    }
}
