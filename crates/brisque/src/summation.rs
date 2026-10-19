//! Independent balanced binary-tree summation of f32 values.
//!
//! Each sum splits its index range in half until one value remains. This
//! specifies a deterministic addition order without any platform kernel code.
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

// The old `outer_sum` column tree is kept as the lazy `kernels::tap_tree`:
// the same `mid = lo + (hi−lo)/2` split over a leaf sequence, evaluated
// per output element instead of over a materialized prod matrix.

/// Fixed-order f64 accumulators for the big map statistics. Element i of
/// a contiguous scan adds to lane `i & 7`; lanes combine in the gmsd
/// convention `t = [a0+a4, a1+a5, a2+a6, a3+a7]; t0+t1+t2+t3`. The order
/// is identical on every backend and CPU tier, and f64 accumulation is
/// near-exact — replacing the recursive f32 pairwise trees (whose own
/// ~1e-7-relative order error the JND/1000 gate tolerates with margin).
pub(crate) struct LaneSums8(pub [f64; 8]);

impl LaneSums8 {
    pub(crate) const fn new() -> Self {
        Self([0.0; 8])
    }

    #[inline(always)]
    pub(crate) fn add(&mut self, i: usize, v: f64) {
        self.0[i & 7] += v;
    }

    pub(crate) fn total(self) -> f64 {
        let a = self.0;
        let t = [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]];
        t[0] + t[1] + t[2] + t[3]
    }
}
