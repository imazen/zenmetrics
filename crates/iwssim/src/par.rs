//! Deterministic plane banding for `parallel` builds — same contract
//! as `cvvdp::par`: band boundaries are a pure function of the plane
//! length, never of `RAYON_NUM_THREADS`, so single- and
//! multi-threaded runs execute the identical partition. Elementwise
//! and per-row maps are bit-identical by construction; reduce kernels
//! fold per-band partials in band order, which is deterministic
//! (though re-associated vs the flat accumulation — ulp-level drift,
//! inside the crate's 1e-4 parity band).
//!
//! With `parallel` off, every helper runs bands sequentially in
//! order — same code path, same result.

use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Plane samples below which `rayon` dispatch is not worth its
/// overhead (matches `cvvdp::par::PAR_MIN_SAMPLES` — one octave below
/// fast-ssim2's crossover).
pub(crate) const PAR_MIN_SAMPLES: usize = 1 << 17;

/// Band count for `n` samples — pure function of `n`, capped so join
/// overhead stays noise relative to ~64k-sample tasks.
#[inline]
pub(crate) fn n_bands(n: usize) -> usize {
    if n < PAR_MIN_SAMPLES {
        1
    } else {
        (n / (1 << 16)).min(32)
    }
}

/// `f(b)` over `0..nb` collected in band order — parallel under
/// `parallel`, sequential otherwise. Reduce kernels use this so the
/// partial fold order is fixed regardless of thread count.
pub(crate) fn collect_bands<T: Send>(nb: usize, f: impl Fn(usize) -> T + Send + Sync) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        (0..nb).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..nb).map(f).collect()
    }
}

/// Row bands over `dst` (row-major, row length `w`): `f(y0,
/// dst_band)` receives the band's absolute first row and its rows.
/// `nb == 1` calls `f(0, dst)` — the pre-banding path.
#[inline]
pub(crate) fn map_rows(dst: &mut [f32], w: usize, f: impl Fn(usize, &mut [f32]) + Send + Sync) {
    #[cfg(feature = "parallel")]
    {
        let n = dst.len();
        let nb = n_bands(n);
        if nb > 1 {
            debug_assert!(w > 0 && n.is_multiple_of(w));
            let rows = (n / w).div_ceil(nb);
            dst.par_chunks_mut(rows * w)
                .enumerate()
                .for_each(|(i, band)| f(i * rows, band));
            return;
        }
    }
    let _ = w;
    f(0, dst);
}

/// Row bands over two aligned `dst` planes: `f(y0, a_band, b_band)`
/// where both bands cover the same rows.
#[inline]
pub(crate) fn map_rows2(
    a: &mut [f32],
    b: &mut [f32],
    w: usize,
    f: impl Fn(usize, &mut [f32], &mut [f32]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        let n = a.len();
        let nb = n_bands(n);
        if nb > 1 {
            debug_assert!(w > 0 && n.is_multiple_of(w));
            let rows = (n / w).div_ceil(nb);
            a.par_chunks_mut(rows * w)
                .zip(b.par_chunks_mut(rows * w))
                .enumerate()
                .for_each(|(i, (ab, bb))| f(i * rows, ab, bb));
            return;
        }
    }
    let _ = w;
    f(0, a, b);
}

/// Flat bands over one `dst` and three aligned `src` slices:
/// `f(d_band, s0_band, s1_band, s2_band)`.
#[inline]
pub(crate) fn map1_3(
    dst: &mut [f32],
    s0: &[f32],
    s1: &[f32],
    s2: &[f32],
    f: impl Fn(&mut [f32], &[f32], &[f32], &[f32]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        let n = dst.len();
        let nb = n_bands(n);
        if nb > 1 {
            let sz = n.div_ceil(nb);
            dst.par_chunks_mut(sz)
                .zip(s0.par_chunks(sz))
                .zip(s1.par_chunks(sz))
                .zip(s2.par_chunks(sz))
                .for_each(|(((d, a), b), c)| f(d, a, b, c));
            return;
        }
    }
    f(dst, s0, s1, s2);
}
