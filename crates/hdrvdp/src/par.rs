//! Deterministic task-level parallelism for `parallel` builds — same
//! contract as `iwssim::par`/`cvvdp::par`: the partition is structural
//! (independent images, independent band planes), never a function of
//! `RAYON_NUM_THREADS`, so results are bit-identical at any thread count.
//! Reduction order is preserved by collecting per-task results in band
//! order rather than racing them into shared accumulators.
//!
//! With `parallel` off every helper runs sequentially in order — same
//! code path, same result.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// `(f(), g())` — parallel under `parallel`, sequential otherwise.
/// The two tasks are independent by construction at the call site, so the
/// result cannot depend on scheduling.
#[inline]
pub(crate) fn join2<A: Send, B: Send>(
    f: impl FnOnce() -> A + Send,
    g: impl FnOnce() -> B + Send,
) -> (A, B) {
    #[cfg(feature = "parallel")]
    {
        rayon::join(f, g)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (f(), g())
    }
}

/// `f(i)` over `0..n` collected in index order — parallel under
/// `parallel`, sequential otherwise. Used for per-plane work where each
/// output must land at its own index regardless of scheduling.
pub(crate) fn collect_indexed<T: Send>(n: usize, f: impl Fn(usize) -> T + Send + Sync) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

/// `f(offset, dst_chunk)` over `n`-element disjoint chunks of `dst` —
/// parallel under `parallel`, sequential otherwise. For element-wise
/// kernels where `dst[i]` depends only on `src[i]`: each chunk is
/// independent, so the result cannot depend on scheduling. The
/// sequential build calls `f` once on the whole slice — same code path
/// as an unchunked call.
pub(crate) fn for_each_dst_chunk(
    dst: &mut [f64],
    n: usize,
    f: impl Fn(usize, &mut [f64]) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        dst.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, d)| f(i * n, d));
    }
    #[cfg(not(feature = "parallel"))]
    {
        f(0, dst);
    }
}

/// 8-aligned chunk length for [`for_each_dst_chunk`] — ~256 tasks so the
/// pool balances, sized so small inputs get one chunk.
pub(crate) fn chunk8(len: usize) -> usize {
    (len / 8).div_ceil(256).max(1) * 8
}
