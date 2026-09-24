//! Deterministic plane banding for `parallel` builds.
//!
//! fast-ssim2's `Tuning` lesson, applied: scheduling decides *how*
//! work is divided across the pool, never *what* is computed. Band
//! boundaries below are a pure function of the plane length — never
//! of `RAYON_NUM_THREADS` — so single- and multi-threaded runs
//! execute the identical partition. Elementwise maps are then
//! bit-identical by construction; reduce kernels fold per-band
//! partial sums in band order, which is deterministic (though it
//! re-associates vs the old flat accumulation — ulp-level drift,
//! inside the 1e-3 JOD parity gate).
//!
//! With `parallel` off, every helper below runs its bands
//! sequentially in order — same code path, same result.

use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Plane samples below which `rayon` dispatch is not worth its
/// overhead. fast-ssim2 measured ~2× SLOWER when splitting every
/// small stage at 320×240 (`PAR_MIN_SAMPLES = 1 << 18` there);
/// pyramid tails are tiny planes and at 512² even level 0 sits right
/// at that boundary, so the threshold here is one octave lower.
pub(crate) const PAR_MIN_SAMPLES: usize = 1 << 17;

/// Band count for `n` samples — a pure function of `n`, capped so
/// join overhead stays noise relative to ~64k-sample tasks while
/// still filling an 8-way pool with headroom.
#[inline]
pub(crate) fn n_bands(n: usize) -> usize {
    if n < PAR_MIN_SAMPLES {
        1
    } else {
        (n / (1 << 16)).min(32)
    }
}

/// Contiguous chunk length for band splits over `n` samples
/// (`n` itself when the plane is below the parallel threshold).
#[cfg(feature = "parallel")]
#[inline]
pub(crate) fn band_size(n: usize) -> usize {
    let nb = n_bands(n);
    if nb <= 1 { n.max(1) } else { n.div_ceil(nb) }
}

/// `f(b)` over `0..nb` collected in band order — parallel under
/// `parallel`, sequential otherwise. Reduce kernels use this so the
/// partial-sum fold order is fixed regardless of thread count.
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

/// `rayon::join` with a sequential fallback — for two independent
/// equal-cost builds.
#[inline]
pub(crate) fn join2<A: Send, B: Send>(
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    #[cfg(feature = "parallel")]
    {
        rayon::join(a, b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (a(), b())
    }
}

/// Row bands with an absolute row index: `f(y0, dst_band)` where
/// `dst_band` covers rows `y0..y0 + dst_band.len()/w`. For
/// neighbourhood ops (blur vertical pass) that read a halo OUTSIDE
/// the band — the closure keeps the full source and indexes by
/// absolute row while writing the band-relative slice. Serial
/// (`nb == 1`) calls `f(0, dst)` — the exact pre-banding path.
#[inline]
pub(crate) fn map_rows(dst: &mut [f32], w: usize, f: impl Fn(usize, &mut [f32]) + Send + Sync) {
    #[cfg(feature = "parallel")]
    {
        let n = dst.len();
        let nb = n_bands(n);
        if nb > 1 {
            debug_assert!(w > 0 && n.is_multiple_of(w));
            let rows = (n / w).div_ceil(nb);
            use rayon::prelude::*;
            dst.par_chunks_mut(rows * w)
                .enumerate()
                .for_each(|(i, band)| f(i * rows, band));
            return;
        }
    }
    let _ = w;
    f(0, dst);
}

/// `f(dst_band, src_band)` over aligned bands.
#[inline]
pub(crate) fn map1(dst: &mut [f32], src: &[f32], f: impl Fn(&mut [f32], &[f32]) + Send + Sync) {
    #[cfg(feature = "parallel")]
    {
        let n = dst.len();
        let sz = band_size(n);
        if sz < n {
            dst.par_chunks_mut(sz)
                .zip(src.par_chunks(sz))
                .for_each(|(d, s)| f(d, s));
            return;
        }
    }
    f(dst, src);
}

/// `f(dst_band, s0_band, s1_band)` over aligned bands.
#[inline]
pub(crate) fn map1_2(
    dst: &mut [f32],
    s0: &[f32],
    s1: &[f32],
    f: impl Fn(&mut [f32], &[f32], &[f32]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        let n = dst.len();
        let sz = band_size(n);
        if sz < n {
            dst.par_chunks_mut(sz)
                .zip(s0.par_chunks(sz))
                .zip(s1.par_chunks(sz))
                .for_each(|((d, a), b)| f(d, a, b));
            return;
        }
    }
    f(dst, s0, s1);
}

/// `f(d0_band, d1_band, src_band)` over aligned bands — the
/// dual-accumulator FIR shape.
#[inline]
pub(crate) fn map2_1(
    d0: &mut [f32],
    d1: &mut [f32],
    src: &[f32],
    f: impl Fn(&mut [f32], &mut [f32], &[f32]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        let n = d0.len();
        let sz = band_size(n);
        if sz < n {
            d0.par_chunks_mut(sz)
                .zip(d1.par_chunks_mut(sz))
                .zip(src.par_chunks(sz))
                .for_each(|((a, b), s)| f(a, b, s));
            return;
        }
    }
    f(d0, d1, src);
}

/// `f(d0_band, d1_band, s0_band, s1_band, s2_band)` — the paired
/// CSF-scaling shape.
#[inline]
pub(crate) fn map2_3(
    d0: &mut [f32],
    d1: &mut [f32],
    s0: &[f32],
    s1: &[f32],
    s2: &[f32],
    f: impl Fn(&mut [f32], &mut [f32], &[f32], &[f32], &[f32]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        let n = d0.len();
        let sz = band_size(n);
        if sz < n {
            d0.par_chunks_mut(sz)
                .zip(d1.par_chunks_mut(sz))
                .zip(s0.par_chunks(sz))
                .zip(s1.par_chunks(sz))
                .zip(s2.par_chunks(sz))
                .for_each(|((((a, b), x), y), z)| f(a, b, x, y, z));
            return;
        }
    }
    f(d0, d1, s0, s1, s2);
}

/// Three independent dsts, no src — the RGB→DKL conversion shape.
/// `f(base, d0_band, d1_band, d2_band)`; `base` is the band's first
/// sample index so global `fetch(i)` closures stay correct.
#[inline]
pub(crate) fn map3_base(
    d0: &mut [f32],
    d1: &mut [f32],
    d2: &mut [f32],
    f: impl Fn(usize, &mut [f32], &mut [f32], &mut [f32]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        let n = d0.len();
        let sz = band_size(n);
        if sz < n {
            d0.par_chunks_mut(sz)
                .zip(d1.par_chunks_mut(sz))
                .zip(d2.par_chunks_mut(sz))
                .enumerate()
                .for_each(|(b, ((a, bb), c))| f(b * sz, a, bb, c));
            return;
        }
    }
    f(0, d0, d1, d2);
}

/// `f(base, dst_band)` — caller slices its own sources by the band's
/// first sample index (for N-source fused kernels like the FIR
/// dot-product).
#[inline]
pub(crate) fn map_base(dst: &mut [f32], f: impl Fn(usize, &mut [f32]) + Send + Sync) {
    #[cfg(feature = "parallel")]
    {
        let n = dst.len();
        let sz = band_size(n);
        if sz < n {
            use rayon::prelude::*;
            dst.par_chunks_mut(sz)
                .enumerate()
                .for_each(|(b, band)| f(b * sz, band));
            return;
        }
    }
    f(0, dst);
}

/// `f(base, d0_band, d1_band)` — two dsts sharing the same base
/// (paired FIR accumulators).
#[inline]
pub(crate) fn map2_base(
    d0: &mut [f32],
    d1: &mut [f32],
    f: impl Fn(usize, &mut [f32], &mut [f32]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        let n = d0.len();
        let sz = band_size(n);
        if sz < n {
            use rayon::prelude::*;
            d0.par_chunks_mut(sz)
                .zip(d1.par_chunks_mut(sz))
                .enumerate()
                .for_each(|(b, (a, c))| f(b * sz, a, c));
            return;
        }
    }
    f(0, d0, d1);
}

/// `f(base, d0_band, d1_band, d2_band, d3_band)` — four dsts sharing
/// the same base (the four-channel sensitivity maps).
#[inline]
pub(crate) fn map4_base(
    d0: &mut [f32],
    d1: &mut [f32],
    d2: &mut [f32],
    d3: &mut [f32],
    f: impl Fn(usize, &mut [f32], &mut [f32], &mut [f32], &mut [f32]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        let n = d0.len();
        let sz = band_size(n);
        if sz < n {
            use rayon::prelude::*;
            d0.par_chunks_mut(sz)
                .zip(d1.par_chunks_mut(sz))
                .zip(d2.par_chunks_mut(sz))
                .zip(d3.par_chunks_mut(sz))
                .enumerate()
                .for_each(|(b, (((a, c), e), g))| f(b * sz, a, c, e, g));
            return;
        }
    }
    f(0, d0, d1, d2, d3);
}

/// Row-aligned bands over three outputs: `f(y0, b0, b1, b2)` — for
/// strided-source converts (padded-row linear planes) that index
/// source rows absolutely.
#[inline]
pub(crate) fn map3_rows(
    d0: &mut [f32],
    d1: &mut [f32],
    d2: &mut [f32],
    w: usize,
    f: impl Fn(usize, &mut [f32], &mut [f32], &mut [f32]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        let n = d0.len();
        let nb = n_bands(n);
        if nb > 1 {
            debug_assert!(w > 0 && n.is_multiple_of(w));
            let rows = (n / w).div_ceil(nb);
            use rayon::prelude::*;
            d0.par_chunks_mut(rows * w)
                .zip(d1.par_chunks_mut(rows * w))
                .zip(d2.par_chunks_mut(rows * w))
                .enumerate()
                .for_each(|(i, ((a, b), c))| f(i * rows, a, b, c));
            return;
        }
    }
    let _ = w;
    f(0, d0, d1, d2);
}

/// `f(t_band, r_band, s_band)` per band → partials folded in band
/// order. `nb == 1` calls `f` once on the full slices — the exact
/// pre-banding path, bit-identical.
#[inline]
pub(crate) fn reduce3(
    t: &[f32],
    r: &[f32],
    s: &[f32],
    f: impl Fn(&[f32], &[f32], &[f32]) -> f32 + Send + Sync,
) -> f32 {
    let n = t.len();
    let nb = n_bands(n);
    if nb == 1 {
        return f(t, r, s);
    }
    let sz = n.div_ceil(nb);
    let parts = collect_bands(nb, |b| {
        let lo = b * sz;
        let hi = (lo + sz).min(n);
        if lo >= hi {
            0.0
        } else {
            f(&t[lo..hi], &r[lo..hi], &s[lo..hi])
        }
    });
    parts.iter().fold(0.0, |a, &p| a + p)
}

/// `f(d_bands, t_bands)` per band over 4+4 channel planes → `[f32;
/// 4]` partials folded per channel in band order. `nb == 1` calls
/// `f` once on the full slices.
#[inline]
pub(crate) fn reduce8_4(
    d: &[&[f32]; 4],
    t: &[&[f32]; 4],
    f: impl Fn(&[&[f32]; 4], &[&[f32]; 4]) -> [f32; 4] + Send + Sync,
) -> [f32; 4] {
    let n = d[0].len();
    let nb = n_bands(n);
    if nb == 1 {
        return f(d, t);
    }
    let sz = n.div_ceil(nb);
    let parts = collect_bands(nb, |b| {
        let lo = b * sz;
        let hi = (lo + sz).min(n);
        if lo >= hi {
            return [0.0; 4];
        }
        f(
            &[&d[0][lo..hi], &d[1][lo..hi], &d[2][lo..hi], &d[3][lo..hi]],
            &[&t[0][lo..hi], &t[1][lo..hi], &t[2][lo..hi], &t[3][lo..hi]],
        )
    });
    let mut total = [0.0_f32; 4];
    for p in &parts {
        for c in 0..4 {
            total[c] += p[c];
        }
    }
    total
}
