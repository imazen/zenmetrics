//! Scoped work-splitting for the metric heads. The trunk pair leaves every
//! worker thread free once both runs return, while the head is the biggest
//! serial block left (measured ~18% of `compute` for DISTS at 256² on i265).
//!
//! Every use is a `par_map` over items with independent outputs — channel
//! statistics, per-tap distances — reassembled in index order, so each
//! item's arithmetic and every later merge order is unchanged. A run with
//! `threads == 1` is byte-for-byte the sequential code.

/// Threads for a head pass over `elems` f32 elements: roughly one worker
/// per 256K elements, capped by the process's configured worker count.
pub(crate) fn threads_for(elems: usize) -> usize {
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

/// `f(&mut item)` over a slice, one item per task on up to `threads` scoped
/// threads.
pub(crate) fn par_each_mut<T: Send>(items: &mut [T], threads: usize, f: impl Fn(&mut T) + Sync) {
    let threads = threads.min(items.len());
    if threads <= 1 {
        items.iter_mut().for_each(f);
        return;
    }
    std::thread::scope(|s| {
        let f = &f;
        for it in items.iter_mut() {
            s.spawn(move || f(it));
        }
    });
}
