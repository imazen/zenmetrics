// One definition for the native eight- and sixteen-lane families.
macro_rules! kernels {
    ($Vector:ident, $Convert:ident, $lanes:literal, $exp:ident, $ln:ident, $pow:ident) => {
        use super::MathPolicy;
        use magetypes::simd::backends::$Convert;
        use magetypes::simd::generic::$Vector;
        /// `out[i] = (xs[i] + offset)^p - offset_pow_p` for every element.
        ///
        /// All inputs must satisfy `xs[i] + offset > 0` so the unchecked
        /// `pow_midp_unchecked` path is sound. The caller passes
        /// `offset_pow_p = offset.powf(p)` (loop-invariant, computed once).
        #[inline(always)]
        pub fn safe_pow_with_offset_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            xs: &[f32],
            out: &mut [f32],
            offset: f32,
            p: f32,
            offset_pow_p: f32,
        ) {
            debug_assert_eq!(xs.len(), out.len());

            type F32x8<T> = $Vector<T>;
            let offset_v = F32x8::<T>::splat(token, offset);
            let offset_pow_p_v = F32x8::<T>::splat(token, offset_pow_p);
            let exact_offset = F32x8::<T>::splat(token, P::scalar_pow(offset, p) - offset_pow_p);

            let (in_chunks, in_tail) = F32x8::<T>::partition_slice(token, xs);
            let (out_chunks, out_tail) = F32x8::<T>::partition_slice_mut(token, out);
            debug_assert_eq!(in_chunks.len(), out_chunks.len());

            for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                let x = F32x8::<T>::load(token, in_chunk);
                let shifted = x + offset_v;
                // pow_midp_unchecked = exp2_midp_unchecked(p * log2_midp_unchecked(x)).
                // Input is guaranteed > 0 because the caller pre-offsets by
                // `offset = SAFE_EPS > 0` and the magnitudes the masking
                // pipeline produces never go below 0.
                let raised = P::$pow(shifted, p);
                let result = raised - offset_pow_p_v;
                // Preserve the scalar offset result when the addition loses x.
                F32x8::<T>::blend(shifted.simd_eq(offset_v), exact_offset, result).store(out_chunk);
            }

            // Scalar tail uses the policy scalar implementation:
            // same offset / p / subtraction order, just element-wise.
            for (xi, oi) in in_tail.iter().zip(out_tail.iter_mut()) {
                *oi = P::scalar_pow(xi + offset, p) - offset_pow_p;
            }
        }

        /// `out[i] = exp(xs[i])`.
        #[inline(always)]
        pub fn vexp_kernel<T: $Convert, P: MathPolicy>(token: T, xs: &[f32], out: &mut [f32]) {
            debug_assert_eq!(xs.len(), out.len());
            type F32x8<T> = $Vector<T>;

            let (in_chunks, in_tail) = F32x8::<T>::partition_slice(token, xs);
            let (out_chunks, out_tail) = F32x8::<T>::partition_slice_mut(token, out);
            for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                P::$exp(F32x8::<T>::load(token, in_chunk)).store(out_chunk);
            }
            for (xi, oi) in in_tail.iter().zip(out_tail.iter_mut()) {
                *oi = P::scalar_exp(*xi);
            }
        }

        /// `out[i] = ln(xs[i])`. Inputs must be `> 0`.
        #[inline(always)]
        pub fn vlog_kernel<T: $Convert, P: MathPolicy>(token: T, xs: &[f32], out: &mut [f32]) {
            debug_assert_eq!(xs.len(), out.len());
            type F32x8<T> = $Vector<T>;

            let (in_chunks, in_tail) = F32x8::<T>::partition_slice(token, xs);
            let (out_chunks, out_tail) = F32x8::<T>::partition_slice_mut(token, out);
            for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                P::$ln(F32x8::<T>::load(token, in_chunk)).store(out_chunk);
            }
            for (xi, oi) in in_tail.iter().zip(out_tail.iter_mut()) {
                *oi = P::scalar_ln(*xi);
            }
        }

        /// `out[i] = xs[i]^p`. Inputs must be `> 0`.
        #[inline(always)]
        pub fn vpow_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            xs: &[f32],
            out: &mut [f32],
            p: f32,
        ) {
            debug_assert_eq!(xs.len(), out.len());
            type F32x8<T> = $Vector<T>;

            let (in_chunks, in_tail) = F32x8::<T>::partition_slice(token, xs);
            let (out_chunks, out_tail) = F32x8::<T>::partition_slice_mut(token, out);
            for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
                P::$pow(F32x8::<T>::load(token, in_chunk), p).store(out_chunk);
            }
            for (xi, oi) in in_tail.iter().zip(out_tail.iter_mut()) {
                *oi = P::scalar_pow(*xi, p);
            }
        }

        /// Weber-contrast band fill (pyramid non-baseband levels):
        /// `band[i] = clamp((fine[i] - img_exp[i]) / l, -1000, 1000)`,
        /// `log[i] = log10(l)`, where `l = max(expanded_l[i], 0.01)`.
        /// Replaces a scalar loop whose per-pixel `log10f` dominated the
        /// video profile; the vector `ln * LOG10_E` differs from `log10f` by
        /// ~1 ulp, far below the 1e-3 JOD gate.
        #[inline(always)]
        pub fn vweber_band_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            band: &mut [f32],
            log: &mut [f32],
            fine: &[f32],
            img_exp: &[f32],
            exp_l: &[f32],
        ) {
            debug_assert_eq!(band.len(), log.len());
            debug_assert_eq!(band.len(), fine.len());
            debug_assert_eq!(band.len(), img_exp.len());
            debug_assert_eq!(band.len(), exp_l.len());
            type F32x8<T> = $Vector<T>;
            let floor_v = F32x8::<T>::splat(token, 0.01);
            let log10e = F32x8::<T>::splat(token, core::f32::consts::LOG10_E);
            let hi = F32x8::<T>::splat(token, 1000.0);
            let lo = F32x8::<T>::splat(token, -1000.0);
            let (b_chunks, b_tail) = F32x8::<T>::partition_slice_mut(token, band);
            let (l_chunks, l_tail) = F32x8::<T>::partition_slice_mut(token, log);
            let (f_chunks, f_tail) = F32x8::<T>::partition_slice(token, fine);
            let (e_chunks, e_tail) = F32x8::<T>::partition_slice(token, img_exp);
            let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, exp_l);
            for ((((b, lg), f), e), x) in b_chunks
                .iter_mut()
                .zip(l_chunks.iter_mut())
                .zip(f_chunks.iter())
                .zip(e_chunks.iter())
                .zip(x_chunks.iter())
            {
                let l = F32x8::<T>::load(token, x).max(floor_v);
                let c = (F32x8::<T>::load(token, f) - F32x8::<T>::load(token, e)) / l;
                c.min(hi).max(lo).store(b);
                (P::$ln(l) * log10e).store(lg);
            }
            for ((((b, lg), f), e), x) in b_tail
                .iter_mut()
                .zip(l_tail.iter_mut())
                .zip(f_tail.iter())
                .zip(e_tail.iter())
                .zip(x_tail.iter())
            {
                let l = x.max(0.01);
                *b = ((f - e) / l).clamp(-1000.0, 1000.0);
                *lg = super::scalar_log10(l);
            }
        }

        // ---------------------------------------------------------------------------
        // Elementwise arithmetic kernels (video path — temporal FIR, masking glue,
        // spatial pooling). Same conventions as the transcendental kernels above:
        // caller-owned `out`, non-fused mul+add so the lane semantics mirror the
        // scalar loops they replace.
        // ---------------------------------------------------------------------------

        /// `dst[i] += a * src[i]` — scalar-broadcast multiply-accumulate
        /// (the temporal FIR inner loop).
        #[inline(always)]
        pub fn vaxpy_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            dst: &mut [f32],
            src: &[f32],
            a: f32,
        ) {
            debug_assert_eq!(dst.len(), src.len());
            type F32x8<T> = $Vector<T>;
            let av = F32x8::<T>::splat(token, a);
            let (d_chunks, d_tail) = F32x8::<T>::partition_slice_mut(token, dst);
            let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, src);
            for (d_chunk, s_chunk) in d_chunks.iter_mut().zip(s_chunks.iter()) {
                let acc = F32x8::<T>::load(token, d_chunk);
                let v = F32x8::<T>::load(token, s_chunk);
                (acc + v * av).store(d_chunk);
            }
            for (di, si) in d_tail.iter_mut().zip(s_tail.iter()) {
                *di += a * *si;
            }
        }

        /// `dst[i] = a * src[i]` — scalar-broadcast multiply into a distinct
        /// buffer (the post-blur `mask_c_lin` scaling).
        #[inline(always)]
        pub fn vscale_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            dst: &mut [f32],
            src: &[f32],
            a: f32,
        ) {
            debug_assert_eq!(dst.len(), src.len());
            type F32x8<T> = $Vector<T>;
            let av = F32x8::<T>::splat(token, a);
            let (d_chunks, d_tail) = F32x8::<T>::partition_slice_mut(token, dst);
            let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, src);
            for (d_chunk, s_chunk) in d_chunks.iter_mut().zip(s_chunks.iter()) {
                (F32x8::<T>::load(token, s_chunk) * av).store(d_chunk);
            }
            for (di, si) in d_tail.iter_mut().zip(s_tail.iter()) {
                *di = *si * a;
            }
        }

        /// `out[i] = |x[i] - y[i]|`.
        #[inline(always)]
        pub fn vabs_diff_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            out: &mut [f32],
            x: &[f32],
            y: &[f32],
        ) {
            debug_assert_eq!(out.len(), x.len());
            debug_assert_eq!(out.len(), y.len());
            type F32x8<T> = $Vector<T>;
            let (o_chunks, o_tail) = F32x8::<T>::partition_slice_mut(token, out);
            let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, x);
            let (y_chunks, y_tail) = F32x8::<T>::partition_slice(token, y);
            for ((o_chunk, x_chunk), y_chunk) in o_chunks
                .iter_mut()
                .zip(x_chunks.iter())
                .zip(y_chunks.iter())
            {
                let xv = F32x8::<T>::load(token, x_chunk);
                let yv = F32x8::<T>::load(token, y_chunk);
                (xv - yv).abs().store(o_chunk);
            }
            for ((oi, xi), yi) in o_tail.iter_mut().zip(x_tail.iter()).zip(y_tail.iter()) {
                *oi = (*xi - *yi).abs();
            }
        }

        /// `out[i] = (|x[i] − y[i]| + offset)^p − offset_pow_p` — the
        /// `vabs_diff` + `safe_pow_with_offset` pair fused into one pass
        /// (the |diff| plane is never materialised). Same per-element op
        /// order as the two-pass chain → bit-identical.
        #[inline(always)]
        pub fn vabs_diff_pow_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            out: &mut [f32],
            x: &[f32],
            y: &[f32],
            offset: f32,
            p: f32,
            offset_pow_p: f32,
        ) {
            debug_assert_eq!(out.len(), x.len());
            debug_assert_eq!(x.len(), y.len());
            type F32x8<T> = $Vector<T>;
            let offset_v = F32x8::<T>::splat(token, offset);
            let offset_pow_p_v = F32x8::<T>::splat(token, offset_pow_p);
            let exact_offset = F32x8::<T>::splat(token, P::scalar_pow(offset, p) - offset_pow_p);
            let (o_chunks, o_tail) = F32x8::<T>::partition_slice_mut(token, out);
            let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, x);
            let (y_chunks, y_tail) = F32x8::<T>::partition_slice(token, y);
            for ((o_chunk, x_chunk), y_chunk) in o_chunks
                .iter_mut()
                .zip(x_chunks.iter())
                .zip(y_chunks.iter())
            {
                let d = (F32x8::<T>::load(token, x_chunk) - F32x8::<T>::load(token, y_chunk)).abs();
                let shifted = d + offset_v;
                let result = P::$pow(shifted, p) - offset_pow_p_v;
                F32x8::<T>::blend(shifted.simd_eq(offset_v), exact_offset, result).store(o_chunk);
            }
            for ((oi, xi), yi) in o_tail.iter_mut().zip(x_tail.iter()).zip(y_tail.iter()) {
                *oi = P::scalar_pow((*xi - *yi).abs() + offset, p) - offset_pow_p;
            }
        }

        /// `out[i] = min(|x[i]|, |y[i]|)` — the mutual-mask raw term.
        #[inline(always)]
        pub fn vmin_abs_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            out: &mut [f32],
            x: &[f32],
            y: &[f32],
        ) {
            debug_assert_eq!(out.len(), x.len());
            debug_assert_eq!(out.len(), y.len());
            type F32x8<T> = $Vector<T>;
            let (o_chunks, o_tail) = F32x8::<T>::partition_slice_mut(token, out);
            let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, x);
            let (y_chunks, y_tail) = F32x8::<T>::partition_slice(token, y);
            for ((o_chunk, x_chunk), y_chunk) in o_chunks
                .iter_mut()
                .zip(x_chunks.iter())
                .zip(y_chunks.iter())
            {
                let xv = F32x8::<T>::load(token, x_chunk);
                let yv = F32x8::<T>::load(token, y_chunk);
                xv.abs().min(yv.abs()).store(o_chunk);
            }
            for ((oi, xi), yi) in o_tail.iter_mut().zip(x_tail.iter()).zip(y_tail.iter()) {
                *oi = xi.abs().min(yi.abs());
            }
        }

        /// Weber-contrast non-baseband fill WITHOUT the `log_l_bkg` output —
        /// identical `band` values to [`vweber_band_kernel`]; only the
        /// reference achromatic pyramid's `log_l_bkg` is ever read, so the
        /// other pyramids skip the `log10` and the plane write entirely.
        #[inline(always)]
        pub fn vweber_band_nolog_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            band: &mut [f32],
            fine: &[f32],
            img_exp: &[f32],
            exp_l: &[f32],
        ) {
            debug_assert_eq!(band.len(), fine.len());
            debug_assert_eq!(band.len(), img_exp.len());
            debug_assert_eq!(band.len(), exp_l.len());
            type F32x8<T> = $Vector<T>;
            let floor_v = F32x8::<T>::splat(token, 0.01);
            let hi = F32x8::<T>::splat(token, 1000.0);
            let lo = F32x8::<T>::splat(token, -1000.0);
            let (b_chunks, b_tail) = F32x8::<T>::partition_slice_mut(token, band);
            let (f_chunks, f_tail) = F32x8::<T>::partition_slice(token, fine);
            let (e_chunks, e_tail) = F32x8::<T>::partition_slice(token, img_exp);
            let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, exp_l);
            for (((b, f), e), x) in b_chunks
                .iter_mut()
                .zip(f_chunks.iter())
                .zip(e_chunks.iter())
                .zip(x_chunks.iter())
            {
                let l = F32x8::<T>::load(token, x).max(floor_v);
                let c = (F32x8::<T>::load(token, f) - F32x8::<T>::load(token, e)) / l;
                c.min(hi).max(lo).store(b);
            }
            for (((b, f), e), x) in b_tail
                .iter_mut()
                .zip(f_tail.iter())
                .zip(e_tail.iter())
                .zip(x_tail.iter())
            {
                let l = x.max(0.01);
                *b = ((f - e) / l).clamp(-1000.0, 1000.0);
            }
        }

        /// `d1[i] = a1 * src[i]`, `d2[i] = a2 * src[i]` — one read of `src`
        /// feeding two scale outputs. The transient channel's FIR reads the
        /// same sustained-A planes as channel 0 with different taps.
        #[inline(always)]
        pub fn vscale2_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            d1: &mut [f32],
            d2: &mut [f32],
            src: &[f32],
            a1: f32,
            a2: f32,
        ) {
            debug_assert_eq!(d1.len(), src.len());
            debug_assert_eq!(d2.len(), src.len());
            type F32x8<T> = $Vector<T>;
            let a1v = F32x8::<T>::splat(token, a1);
            let a2v = F32x8::<T>::splat(token, a2);
            let (d1c, d1t) = F32x8::<T>::partition_slice_mut(token, d1);
            let (d2c, d2t) = F32x8::<T>::partition_slice_mut(token, d2);
            let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, src);
            for ((o1, o2), s_chunk) in d1c.iter_mut().zip(d2c.iter_mut()).zip(s_chunks.iter()) {
                let v = F32x8::<T>::load(token, s_chunk);
                (v * a1v).store(o1);
                (v * a2v).store(o2);
            }
            for ((o1, o2), &sv) in d1t.iter_mut().zip(d2t.iter_mut()).zip(s_tail.iter()) {
                *o1 = sv * a1;
                *o2 = sv * a2;
            }
        }

        /// `d1[i] += a1 * src[i]`, `d2[i] += a2 * src[i]` — dual accumulator
        /// for the shared sustained-A FIR input (channels 0 and 3).
        #[inline(always)]
        pub fn vaxpy2_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            d1: &mut [f32],
            d2: &mut [f32],
            src: &[f32],
            a1: f32,
            a2: f32,
        ) {
            debug_assert_eq!(d1.len(), src.len());
            debug_assert_eq!(d2.len(), src.len());
            type F32x8<T> = $Vector<T>;
            let a1v = F32x8::<T>::splat(token, a1);
            let a2v = F32x8::<T>::splat(token, a2);
            let (d1c, d1t) = F32x8::<T>::partition_slice_mut(token, d1);
            let (d2c, d2t) = F32x8::<T>::partition_slice_mut(token, d2);
            let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, src);
            for ((o1, o2), s_chunk) in d1c.iter_mut().zip(d2c.iter_mut()).zip(s_chunks.iter()) {
                let v = F32x8::<T>::load(token, s_chunk);
                let acc1 = F32x8::<T>::load(token, o1);
                let acc2 = F32x8::<T>::load(token, o2);
                (acc1 + v * a1v).store(o1);
                (acc2 + v * a2v).store(o2);
            }
            for ((o1, o2), &sv) in d1t.iter_mut().zip(d2t.iter_mut()).zip(s_tail.iter()) {
                *o1 += a1 * sv;
                *o2 += a2 * sv;
            }
        }

        /// `dst[i] = Σ_k coeffs[k]·srcs[k][off + i]` — the temporal FIR as a
        /// single fused pass: every source plane is read once per output
        /// element instead of once per `vaxpy` tap call (~3× less plane
        /// traffic). The per-element add sequence is `c0·s0` then
        /// `acc + ck·sk` in ascending k — the identical order the
        /// `vscale`-first-tap + `vaxpy` chain produces, so results are
        /// bit-identical to the unfused loop.
        #[inline(always)]
        pub fn vfir_into_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            dst: &mut [f32],
            srcs: &[&[f32]],
            coeffs: &[f32],
            off: usize,
        ) {
            debug_assert_eq!(srcs.len(), coeffs.len());
            debug_assert!(!srcs.is_empty());
            type F32x8<T> = $Vector<T>;
            let (d_chunks, d_tail) = F32x8::<T>::partition_slice_mut(token, dst);
            let c0 = F32x8::<T>::splat(token, coeffs[0]);
            let mut i = 0usize;
            for dc in d_chunks.iter_mut() {
                let b = off + i;
                let mut acc =
                    F32x8::<T>::load(token, srcs[0][b..b + $lanes].try_into().unwrap()) * c0;
                for (s, &c) in srcs[1..].iter().zip(coeffs[1..].iter()) {
                    let v = F32x8::<T>::load(token, s[b..b + $lanes].try_into().unwrap());
                    acc += v * F32x8::<T>::splat(token, c);
                }
                acc.store(dc);
                i += $lanes;
            }
            for (j, d) in d_tail.iter_mut().enumerate() {
                let b = off + i + j;
                let mut acc = srcs[0][b] * coeffs[0];
                for (s, &c) in srcs[1..].iter().zip(coeffs[1..].iter()) {
                    acc += s[b] * c;
                }
                *d = acc;
            }
        }

        /// Two fused FIR outputs over the same source planes —
        /// `d0[i] = Σ_k c0[k]·srcs[k][off+i]`, `d1[i] = Σ_k c1[k]·srcs[k][off+i]`.
        /// Channel 0 and channel 3 both filter the sustained-A plane with
        /// different taps; sharing the loads halves the source traffic
        /// again. Per-element add order matches the unfused tap chain.
        #[inline(always)]
        pub fn vfir2_into_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            d0: &mut [f32],
            d1: &mut [f32],
            srcs: &[&[f32]],
            coeffs0: &[f32],
            coeffs1: &[f32],
            off: usize,
        ) {
            debug_assert_eq!(d0.len(), d1.len());
            debug_assert_eq!(srcs.len(), coeffs0.len());
            debug_assert_eq!(srcs.len(), coeffs1.len());
            debug_assert!(!srcs.is_empty());
            type F32x8<T> = $Vector<T>;
            let (d0c, d0t) = F32x8::<T>::partition_slice_mut(token, d0);
            let (d1c, d1t) = F32x8::<T>::partition_slice_mut(token, d1);
            let ca0 = F32x8::<T>::splat(token, coeffs0[0]);
            let cb0 = F32x8::<T>::splat(token, coeffs1[0]);
            let mut i = 0usize;
            for (o0, o1) in d0c.iter_mut().zip(d1c.iter_mut()) {
                let b = off + i;
                let v0 = F32x8::<T>::load(token, srcs[0][b..b + $lanes].try_into().unwrap());
                let mut acc0 = v0 * ca0;
                let mut acc1 = v0 * cb0;
                for ((s, &c0k), &c1k) in srcs[1..]
                    .iter()
                    .zip(coeffs0[1..].iter())
                    .zip(coeffs1[1..].iter())
                {
                    let v = F32x8::<T>::load(token, s[b..b + $lanes].try_into().unwrap());
                    acc0 += v * F32x8::<T>::splat(token, c0k);
                    acc1 += v * F32x8::<T>::splat(token, c1k);
                }
                acc0.store(o0);
                acc1.store(o1);
                i += $lanes;
            }
            for (j, (o0, o1)) in d0t.iter_mut().zip(d1t.iter_mut()).enumerate() {
                let b = off + i + j;
                let mut acc0 = srcs[0][b] * coeffs0[0];
                let mut acc1 = srcs[0][b] * coeffs1[0];
                for ((s, &c0k), &c1k) in srcs[1..]
                    .iter()
                    .zip(coeffs0[1..].iter())
                    .zip(coeffs1[1..].iter())
                {
                    acc0 += s[b] * c0k;
                    acc1 += s[b] * c1k;
                }
                *o0 = acc0;
                *o1 = acc1;
            }
        }

        /// `o1[i] = ((x1[i]·a)·w[i])·b`, `o2[i] = ((x2[i]·a)·w[i])·b` — the
        /// `vmul2_scale2` pair for test and reference CSF weighting in one
        /// pass (the sensitivity map `w` is loaded once).
        #[allow(clippy::too_many_arguments)]
        #[inline(always)]
        pub fn vmul2_scale2_pair_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            o1: &mut [f32],
            o2: &mut [f32],
            x1: &[f32],
            x2: &[f32],
            w: &[f32],
            a: f32,
            b: f32,
        ) {
            debug_assert_eq!(o1.len(), x1.len());
            debug_assert_eq!(o2.len(), x2.len());
            debug_assert_eq!(o1.len(), w.len());
            type F32x8<T> = $Vector<T>;
            let av = F32x8::<T>::splat(token, a);
            let bv = F32x8::<T>::splat(token, b);
            let (o1c, o1t) = F32x8::<T>::partition_slice_mut(token, o1);
            let (o2c, o2t) = F32x8::<T>::partition_slice_mut(token, o2);
            let (x1c, x1t) = F32x8::<T>::partition_slice(token, x1);
            let (x2c, x2t) = F32x8::<T>::partition_slice(token, x2);
            let (w_chunks, w_tail) = F32x8::<T>::partition_slice(token, w);
            for ((((p1, p2), v1), v2), wv) in o1c
                .iter_mut()
                .zip(o2c.iter_mut())
                .zip(x1c.iter())
                .zip(x2c.iter())
                .zip(w_chunks.iter())
            {
                let wv = F32x8::<T>::load(token, wv);
                ((F32x8::<T>::load(token, v1) * av * wv) * bv).store(p1);
                ((F32x8::<T>::load(token, v2) * av * wv) * bv).store(p2);
            }
            for ((((p1, p2), &v1), &v2), &wv) in o1t
                .iter_mut()
                .zip(o2t.iter_mut())
                .zip(x1t.iter())
                .zip(x2t.iter())
                .zip(w_tail.iter())
            {
                *p1 = ((v1 * a) * wv) * b;
                *p2 = ((v2 * a) * wv) * b;
            }
        }

        /// Fused `|t[i]−r[i]|·s[i]` followed by the `lp_norm_mean_p2`
        /// accumulation — identical chunk/lane order to running
        /// `vabs_diff_mul_into` then `vlp_norm_mean_p2`, minus the
        /// intermediate plane write+read. Returns the final scalar.
        #[inline(always)]
        pub fn vabs_diff_mul_lp2_sum_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            t: &[f32],
            r: &[f32],
            s: &[f32],
        ) -> f32 {
            const LP_SAFE_EPS: f32 = 1e-5;
            let n = t.len();
            debug_assert_eq!(r.len(), n);
            debug_assert_eq!(s.len(), n);
            if n == 0 {
                return 0.0;
            }
            type F32x8<T> = $Vector<T>;
            let e = F32x8::<T>::splat(token, LP_SAFE_EPS);
            let e2 = F32x8::<T>::splat(token, LP_SAFE_EPS * LP_SAFE_EPS);
            let (t_chunks, t_tail) = F32x8::<T>::partition_slice(token, t);
            let (r_chunks, r_tail) = F32x8::<T>::partition_slice(token, r);
            let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, s);
            let mut acc = F32x8::<T>::zero(token);
            for ((tc, rc), sc) in t_chunks.iter().zip(r_chunks.iter()).zip(s_chunks.iter()) {
                let d = (F32x8::<T>::load(token, tc) - F32x8::<T>::load(token, rc)).abs()
                    * F32x8::<T>::load(token, sc);
                let u = d.abs() + e;
                acc += u * u - e2;
            }
            let mut sum = acc.reduce_add();
            for ((&tv, &rv), &sv) in t_tail.iter().zip(r_tail.iter()).zip(s_tail.iter()) {
                let d = (tv - rv).abs() * sv;
                let u = d.abs() + LP_SAFE_EPS;
                sum += u * u - LP_SAFE_EPS * LP_SAFE_EPS;
            }
            sum
        }

        /// Shared `(mean + eps)^0.5 − eps^0.5` tail for the `*_lp2` reduce
        /// kernels — applied once to the (possibly band-folded) total sum.

        /// `vxcm_pool_clamp_4ch` without the `d` write — reads the `pow`
        /// planes, computes the clamped diff in-register, and accumulates the
        /// `lp_norm_mean_p2` sum per channel in the identical chunk/lane
        /// order, returning the four final norms.
        #[inline(always)]
        pub fn vxcm_pool_clamp_4ch_sqsum_partial_kernel<T: $Convert, P: MathPolicy>(
            token: T,
            d: &[&[f32]; 4],
            t: &[&[f32]; 4],
            w: &[[f32; 4]; 4],
            d_max: f32,
        ) -> [f32; 4] {
            const LP_SAFE_EPS: f32 = 1e-5;
            let n = d[0].len();
            debug_assert!(d.iter().all(|c| c.len() == n));
            debug_assert!(t.iter().all(|c| c.len() == n));
            type F32x8<T> = $Vector<T>;
            let d_max_v = F32x8::<T>::splat(token, d_max);
            let one = F32x8::<T>::splat(token, 1.0);
            let e = F32x8::<T>::splat(token, LP_SAFE_EPS);
            let e2 = F32x8::<T>::splat(token, LP_SAFE_EPS * LP_SAFE_EPS);
            let mut wv = [[F32x8::<T>::zero(token); 4]; 4];
            for (k, wv_row) in wv.iter_mut().enumerate() {
                for (cc, wv_e) in wv_row.iter_mut().enumerate() {
                    *wv_e = F32x8::<T>::splat(token, w[k][cc]);
                }
            }

            let (t0c, t0t) = F32x8::<T>::partition_slice(token, t[0]);
            let (t1c, t1t) = F32x8::<T>::partition_slice(token, t[1]);
            let (t2c, t2t) = F32x8::<T>::partition_slice(token, t[2]);
            let (t3c, t3t) = F32x8::<T>::partition_slice(token, t[3]);
            let t_chunks = [t0c, t1c, t2c, t3c];
            let t_tails = [t0t, t1t, t2t, t3t];
            let (d0c, d0t) = F32x8::<T>::partition_slice(token, d[0]);
            let (d1c, d1t) = F32x8::<T>::partition_slice(token, d[1]);
            let (d2c, d2t) = F32x8::<T>::partition_slice(token, d[2]);
            let (d3c, d3t) = F32x8::<T>::partition_slice(token, d[3]);
            let d_chunks = [d0c, d1c, d2c, d3c];
            let d_tails = [d0t, d1t, d2t, d3t];

            let mut acc = [F32x8::<T>::zero(token); 4];
            for i in 0..t_chunks[0].len() {
                let tv = [
                    F32x8::<T>::load(token, &t_chunks[0][i]),
                    F32x8::<T>::load(token, &t_chunks[1][i]),
                    F32x8::<T>::load(token, &t_chunks[2][i]),
                    F32x8::<T>::load(token, &t_chunks[3][i]),
                ];
                for cc in 0..4 {
                    let m = wv[0][cc] * tv[0]
                        + wv[1][cc] * tv[1]
                        + wv[2][cc] * tv[2]
                        + wv[3][cc] * tv[3];
                    let dv = F32x8::<T>::load(token, &d_chunks[cc][i]);
                    let du = dv / (one + m);
                    let fin = (d_max_v * du) / (d_max_v + du);
                    let u = fin.abs() + e;
                    acc[cc] += u * u - e2;
                }
            }
            let mut sums = [
                acc[0].reduce_add(),
                acc[1].reduce_add(),
                acc[2].reduce_add(),
                acc[3].reduce_add(),
            ];
            for i in 0..t_tails[0].len() {
                for cc in 0..4 {
                    let m = w[0][cc] * t_tails[0][i]
                        + w[1][cc] * t_tails[1][i]
                        + w[2][cc] * t_tails[2][i]
                        + w[3][cc] * t_tails[3][i];
                    let du = d_tails[cc][i] / (1.0 + m);
                    let fin = d_max * du / (d_max + du);
                    let u = fin.abs() + LP_SAFE_EPS;
                    sums[cc] += u * u - LP_SAFE_EPS * LP_SAFE_EPS;
                }
            }
            sums
        }

        /// Uniform-axis LUT interpolation, optionally followed by exp((value+offset)*scale).
        /// The bracket is bounded by `max_index < lut.len()-1`; finite x inputs only.
        #[inline(always)]
        pub fn gather_lerp_kernel<T: $Convert, P: MathPolicy, const EXP: bool>(
            token: T,
            xs: &[f32],
            lut: &[f32],
            min: f32,
            inv_step: f32,
            max_index: f32,
            offset: f32,
            scale: f32,
            out: &mut [f32],
        ) {
            assert_eq!(xs.len(), out.len());
            assert!(lut.len() >= 2 && max_index >= 0.0 && max_index < (lut.len() - 1) as f32);
            type V<T> = $Vector<T>;
            let min_v = V::<T>::splat(token, min);
            let step_v = V::<T>::splat(token, inv_step);
            let zero = V::<T>::zero(token);
            let max_v = V::<T>::splat(token, max_index);
            let offset_v = V::<T>::splat(token, offset);
            let scale_v = V::<T>::splat(token, scale);
            let (xc, xt) = V::<T>::partition_slice(token, xs);
            let (oc, ot) = V::<T>::partition_slice_mut(token, out);
            for (x, o) in xc.iter().zip(oc) {
                let idx = ((V::<T>::load(token, x) - min_v) * step_v)
                    .max(zero)
                    .min(max_v);
                let floor = idx.floor();
                let lanes = floor.to_array();
                // The safe hardware gather currently exists only on V4's f32x16.
                // A lane gather keeps this single body valid on every tier.
                let lo =
                    V::<T>::from_array(token, core::array::from_fn(|j| lut[lanes[j] as usize]));
                let hi =
                    V::<T>::from_array(token, core::array::from_fn(|j| lut[lanes[j] as usize + 1]));
                let value = lo + (idx - floor) * (hi - lo);
                if EXP {
                    P::$exp((value + offset_v) * scale_v).store(o);
                } else {
                    value.store(o);
                }
            }
            for (x, o) in xt.iter().zip(ot) {
                let idx = ((*x - min) * inv_step).clamp(0.0, max_index);
                let floor = super::scalar_floor(idx);
                let lo = floor as usize;
                let value = lut[lo] + (idx - floor) * (lut[lo + 1] - lut[lo]);
                *o = if EXP {
                    P::scalar_exp((value + offset) * scale)
                } else {
                    value
                };
            }
        }
    };
}
