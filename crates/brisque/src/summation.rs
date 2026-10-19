//! Independent balanced binary-tree summation of f32 values.
//!
//! Each sum splits its index range in half until one value remains. This
//! specifies a deterministic addition order without any platform kernel code.
use alloc::vec::Vec;

/// One leaf of the [`aggd_tree_sums`]/[`ggd_tree_sums`] fused walk.
/// Lane order is fixed: [neg², pos², |x|, x²]; lanes 4..8 stay +0.
pub(crate) const SUM_NEG2: usize = 0;
pub(crate) const SUM_POS2: usize = 1;
pub(crate) const SUM_ABS: usize = 2;
pub(crate) const SUM_SQ: usize = 3;

pub(crate) fn inner_sum(values: &[f32]) -> f32 {
    match values.len() {
        0 => 0.0,
        1 => values[0],
        n => {
            let (left, right) = values.split_at(n / 2);
            inner_sum(left) + inner_sum(right)
        }
    }
}

/// Four masked/derived reductions over `x` in one tree walk.
///
/// The scalar code computed each of `inner_sum` over materialized
/// `(x*mask_l)²`, `(x*mask_r)²`, `|x|`, `x²` arrays — four identical
/// halving trees. Here lane `k` replays the `k`-th tree: the leaf for
/// index `i` is `[xl², xr², |x|, x²]` evaluated from `x[i]` alone, so
/// every lane sees exactly the same association as its scalar tree and
/// results are bit-identical.
///
/// Scalar leaf equivalents (must stay in sync with callers):
/// - `xl²`: `(if x < 0 { x } else { 0.0 }).pow(2)` — `v<0` is false for
///   ±0 and NaN, matching `x < 0.0` mask semantics.
/// - `xr²`: same with `x > 0`.
/// - `|x|`, `x²`: elementwise.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
pub(crate) fn quad_tree_sums(token: Token, x: &[f32]) -> [f32; 4] {
    // Iterative post-order replay of the recursive halving tree: the
    // stack carries segments plus merge markers, so every internal add
    // combines the same (left,right) subtree sums as the scalar
    // `inner_sum`, per lane.
    const MERGE: usize = usize::MAX;
    let z = f32x8::zero_t(token);
    let mut segs: Vec<(usize, usize)> = Vec::with_capacity(32);
    let mut vals: Vec<f32x8> = Vec::with_capacity(32);
    // Pairwise reduce matching the split-half tree on a len-8 segment:
    // `((v0+v1)+(v2+v3)) + ((v4+v5)+(v6+v7))`.
    #[inline(always)]
    fn tree8(a: &[f32; 8]) -> f32 {
        ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
    }
    segs.push((0, x.len()));
    while let Some((lo, hi)) = segs.pop() {
        if lo == MERGE {
            let r = vals.pop().unwrap();
            let l = vals.pop().unwrap();
            vals.push(l + r);
        } else if hi - lo == 8 {
            // Batch a whole len-8 subtree: one vector computes all four
            // statistics elementwise (bit-identical to the per-element
            // leaf values — `min/max(x,0)²` ≡ the `<0`/`>0` masks,
            // including NaN and ±0), then each stat's 8 positions are
            // folded with the scalar pairwise order the recursive tree
            // would use. Replaces eight per-leaf `from_array` builds.
            let xv = f32x8::load_t(token, x[lo..hi].try_into().unwrap());
            let neg = xv.min(z);
            let pos = xv.max(z);
            let leaf = f32x8::from_array_t(
                token,
                [
                    tree8(&(neg * neg).to_array()),
                    tree8(&(pos * pos).to_array()),
                    tree8(&xv.abs().to_array()),
                    tree8(&(xv * xv).to_array()),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            );
            vals.push(leaf);
        } else if hi - lo <= 1 {
            let leaf = if hi > lo {
                let v = x[lo];
                let sq = v * v;
                f32x8::from_array_t(
                    token,
                    [
                        if v < 0.0 { sq } else { 0.0 },
                        if v > 0.0 { sq } else { 0.0 },
                        v.abs(),
                        sq,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                )
            } else {
                f32x8::zero_t(token)
            };
            vals.push(leaf);
        } else {
            let mid = lo + (hi - lo) / 2;
            segs.push((MERGE, MERGE));
            segs.push((mid, hi));
            segs.push((lo, mid));
        }
    }
    let t = vals.pop().unwrap();
    let a = t.to_array();
    [a[0], a[1], a[2], a[3]]
}

/// Dispatched [`quad_tree_sums`]. See it for the parity contract.
pub(crate) fn quad_sums(x: &[f32]) -> [f32; 4] {
    archmage::attuned!(
        quad_tree_sums(x),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `outer_sum` per-column trees, 8 columns per lane group.
///
/// Lane `l` owns output column `i + l`: leaf loads are the contiguous
/// `values[r * columns + i .. i + 8]` and every internal node is one
/// vector add that replays each column's balanced tree — bit-identical
/// to the scalar `column` recursion on every tier. The tail group
/// (`columns % 8 != 0`) clamps lane indices to `columns - 1`; those
/// lanes' results are discarded.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn outer_sum_core(
    token: Token,
    values: &[f32],
    rows: usize,
    columns: usize,
    out: &mut [f32],
) {
    const MERGE: usize = usize::MAX;
    let mut segs: Vec<(usize, usize)> = Vec::with_capacity(32);
    let mut vals: Vec<f32x8> = Vec::with_capacity(32);
    let mut i = 0usize;
    while i < columns {
        let tail = columns - i < 8;
        segs.push((0, rows));
        while let Some((lo, hi)) = segs.pop() {
            if lo == MERGE {
                let r = vals.pop().unwrap();
                let l = vals.pop().unwrap();
                vals.push(l + r);
            } else if hi - lo <= 1 {
                let leaf = if hi > lo {
                    let base = lo * columns + i;
                    if tail {
                        f32x8::from_array_t(
                            token,
                            core::array::from_fn(|l| {
                                values[base + l.min(columns - i - 1)]
                            }),
                        )
                    } else {
                        f32x8::from_slice_t(token, &values[base..base + 8])
                    }
                } else {
                    f32x8::zero_t(token)
                };
                vals.push(leaf);
            } else {
                let mid = lo + (hi - lo) / 2;
                segs.push((MERGE, MERGE));
                segs.push((mid, hi));
                segs.push((lo, mid));
            }
        }
        let a = vals.pop().unwrap().to_array();
        let n = (columns - i).min(8);
        out[i..i + n].copy_from_slice(&a[..n]);
        i += 8;
    }
}

pub(crate) fn outer_sum(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    assert_eq!(values.len(), rows * columns);
    let mut out = vec![0f32; columns];
    archmage::attuned!(
        outer_sum_core(values, rows, columns, &mut out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    out
}
