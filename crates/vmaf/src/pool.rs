//! Thread-local scratch-buffer recycling.
//!
//! Fresh `vec![0; n]` allocations of multi-MB buffers come from mmap'd zero
//! pages: every 4 KiB page faults on first touch, so allocating ~40 MB of
//! band arrays per frame costs ~10K page faults plus the memset — several
//! milliseconds per frame at 720p. Pooling keeps the pages faulted, so the
//! per-frame cost disappears after the first call.
//!
//! Recycled vecs may contain **stale data** — callers must write every
//! element before reading it (the same invariant the band consumers already
//! rely on when arrays are reused across scales). Buffers larger than
//! [`POOL_CAP`] are dropped rather than pooled.

use std::cell::RefCell;

const POOL_CAP: usize = 1 << 24;

macro_rules! pool {
    ($static_name:ident, $take:ident, $give:ident, $ty:ty) => {
        thread_local! {
            static $static_name: RefCell<Vec<Vec<$ty>>> = const { RefCell::new(Vec::new()) };
        }
        /// Pooled vec of length `n`; recycled contents may be stale.
        pub(crate) fn $take(n: usize) -> Vec<$ty> {
            let mut v = $static_name
                .with(|p| p.borrow_mut().pop())
                .unwrap_or_default();
            if v.len() >= n {
                v.truncate(n);
            } else {
                v.resize(n, <$ty>::default());
            }
            v
        }
        /// Return a vec to the pool (keeps its allocation for reuse).
        pub(crate) fn $give(v: Vec<$ty>) {
            if v.capacity() <= POOL_CAP {
                $static_name.with(|p| p.borrow_mut().push(v));
            }
        }
    };
}

pool!(POOL_I32, take_i32, give_i32, i32);
pool!(POOL_I16, take_i16, give_i16, i16);
pool!(POOL_F32, take_f32, give_f32, f32);
pool!(POOL_U32, take_u32, give_u32, u32);
pool!(POOL_U16, take_u16, give_u16, u16);
