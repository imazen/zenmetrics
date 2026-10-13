//! Thread-local reusable `f32` plane buffers.
//!
//! A full IW-SSIM pass allocates ~20 plane-sized scratch slabs per
//! score call (pyramid intermediates, cs moments, IW weight maps).
//! Fresh `alloc::vec!` pays malloc + zero-fill + page-fault churn on
//! every call; pooling reuses the allocations across calls. Buffers
//! carry **unspecified contents** — every pooled site must write each
//! element before it is read (all our producer kernels do; a site that
//! needs zeros must `vec![0.0; n]` instead).
//!
//! `no_std` builds get plain allocation (no TLS to hang the pool on).

use alloc::vec::Vec;

/// Enough for one call's live set: pyramid scratch (~6), five scales of
/// cs intermediates (~7 each) and IW maps (~7 each), plus slack for the
/// strip path's per-strip bands.
const POOL_CAP: usize = 80;

/// Take a length-`n` buffer, reusing a pooled allocation when one fits.
/// Contents are unspecified.
pub(crate) fn buf_take(n: usize) -> Vec<f32> {
    #[cfg(feature = "std")]
    {
        let mut v = BUF_POOL.with(|p| {
            let mut p = p.borrow_mut();
            let idx = p
                .iter()
                .enumerate()
                .filter(|(_, v)| v.capacity() >= n)
                // Prefer `len >= n` so `resize` truncates rather than
                // zero-filling an extension; then smallest capacity.
                .min_by_key(|(_, v)| (v.len() < n, v.capacity()))
                .map(|(i, _)| i);
            idx.map(|i| p.swap_remove(i)).unwrap_or_default()
        });
        v.resize(n, 0.0);
        v
    }
    #[cfg(not(feature = "std"))]
    {
        alloc::vec![0.0_f32; n]
    }
}

/// Return a buffer to the pool for reuse by a later call.
pub(crate) fn buf_give(v: Vec<f32>) {
    #[cfg(feature = "std")]
    {
        if v.capacity() > 0 {
            BUF_POOL.with(|p| {
                let mut p = p.borrow_mut();
                if p.len() < POOL_CAP {
                    p.push(v);
                }
            });
        }
    }
    #[cfg(not(feature = "std"))]
    {
        drop(v);
    }
}

#[cfg(feature = "std")]
std::thread_local! {
    static BUF_POOL: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Bisect shim: drops instead of pooling (no reuse from these sites).
#[allow(dead_code)]
pub(crate) fn buf_give_never(v: Vec<f32>) {
    drop(v);
}
