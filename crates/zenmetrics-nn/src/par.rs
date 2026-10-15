//! Scoped work-splitting for the metric heads. The trunk pair leaves every
//! worker thread free once both runs return, while the head is the biggest
//! serial block left (measured ~18% of `compute` for DISTS at 256² on i265).
//!
//! Every use is a `par_map` over items with independent outputs — channel
//! statistics, per-tap distances — reassembled in index order, so each
//! item's arithmetic and every later merge order is unchanged. A run with
//! `threads == 1` is byte-for-byte the sequential code.

/// Threads for a pass over estimated independent work units (scalar element
/// conversions or multiply-accumulates, depending on the caller): roughly one
/// worker per 256K units, capped by the process's configured worker count.
pub(crate) fn threads_for(elems: usize) -> usize {
    #[cfg(test)]
    if let Some(cap) = TEST_WORKERS.with(|v| v.get()) {
        return (elems / (256 * 1024)).clamp(1, cap);
    }
    (elems / (256 * 1024)).clamp(1, crate::engine::workers())
}

/// `f(i)` for `0..len`, split into contiguous ranges across up to `threads`
/// scoped threads, results in index order. Falls back to a plain loop when
/// `threads <= 1`.
pub(crate) fn par_map<T: Send>(
    len: usize,
    threads: usize,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let threads = threads.min(len);
    if threads <= 1 {
        return (0..len).map(&f).collect();
    }
    let per = len.div_ceil(threads);
    std::thread::scope(|s| {
        let f = &f;
        let mut handles = Vec::new();
        for start in (0..len).step_by(per) {
            let end = (start + per).min(len);
            #[cfg(test)]
            SPAWNS.with(|v| v.set(v.get() + 1));
            handles.push(s.spawn(move || (start..end).map(f).collect::<Vec<T>>()));
        }
        handles
            .into_iter()
            .flat_map(|h| match h.join() {
                Ok(v) => v,
                Err(e) => std::panic::resume_unwind(e),
            })
            .collect()
    })
}

/// `f(&mut item)` over a slice, split into at most `threads` contiguous
/// chunks on scoped threads (same splitting as `par_map`).
pub(crate) fn par_each_mut<T: Send>(items: &mut [T], threads: usize, f: impl Fn(&mut T) + Sync) {
    let threads = threads.min(items.len());
    if threads <= 1 {
        items.iter_mut().for_each(f);
        return;
    }
    let per = items.len().div_ceil(threads);
    std::thread::scope(|s| {
        let f = &f;
        for chunk in items.chunks_mut(per) {
            #[cfg(test)]
            SPAWNS.with(|v| v.set(v.get() + 1));
            s.spawn(move || chunk.iter_mut().for_each(f));
        }
    });
}

/// Indexed rows without a descriptor allocation on the sequential path.
/// Empty-width buffers have no rows.
pub(crate) fn for_each_row<T: Send>(
    out: &mut [T],
    width: usize,
    threads: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    if width == 0 {
        return;
    }
    if threads <= 1 {
        for (i, row) in out.chunks_mut(width).enumerate() {
            f(i, row);
        }
    } else {
        let mut rows: Vec<_> = out.chunks_mut(width).enumerate().collect();
        par_each_mut(&mut rows, threads, |row| f(row.0, row.1));
    }
}

#[cfg(test)]
thread_local! {
    static TEST_WORKERS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static SPAWNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn observe_workers<T>(cap: Option<usize>, f: impl FnOnce() -> T) -> (T, usize) {
    struct Reset(Option<usize>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_WORKERS.with(|v| v.set(self.0));
        }
    }
    let _reset = Reset(TEST_WORKERS.with(|v| v.replace(cap)));
    SPAWNS.with(|v| v.set(0));
    let value = f();
    (value, SPAWNS.with(|v| v.get()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `par_each_mut` with more items than threads must never run more than
    /// `threads` calls at once (a high-water mark over a shared counter),
    /// and must visit every item exactly once.
    #[test]
    fn each_mut_respects_thread_cap() {
        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let mut items = vec![0usize; 16];
        par_each_mut(&mut items, 3, |v| {
            let n = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(n, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(10));
            *v += 1;
            live.fetch_sub(1, Ordering::SeqCst);
        });
        let peak = peak.load(Ordering::SeqCst);
        assert!(peak <= 3, "peak concurrency {peak} exceeded the cap");
        assert!(peak >= 2, "items serialised under a cap of 3 (peak {peak})");
        assert!(items.iter().all(|&v| v == 1), "every item visited once");
    }

    /// `par_map` returns results in index order and `threads <= 1` collapses
    /// both helpers to the plain sequential loop.
    #[test]
    fn map_order_and_sequential_fallback() {
        let got = par_map(12, 4, |i| i * i);
        assert_eq!(got, (0..12).map(|i| i * i).collect::<Vec<_>>());
        let got = par_map(12, 1, |i| i * i);
        assert_eq!(got, (0..12).map(|i| i * i).collect::<Vec<_>>());
        let mut items = vec![1usize; 8];
        par_each_mut(&mut items, 1, |v| *v += 1);
        assert_eq!(items, [2; 8]);
    }
}
