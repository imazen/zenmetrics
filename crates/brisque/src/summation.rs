//! Independent balanced binary-tree summation of f32 values.
//!
//! Each sum splits its index range in half until one value remains. This
//! specifies a deterministic addition order without any platform kernel code.
use alloc::vec::Vec;

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

pub(crate) fn inner_mean(values: &[f32]) -> f32 {
    inner_sum(values) / values.len() as f32
}

pub(crate) fn outer_sum(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    assert_eq!(values.len(), rows * columns);
    fn column(values: &[f32], stride: usize, index: usize, first: usize, end: usize) -> f32 {
        match end - first {
            0 => 0.0,
            1 => values[first * stride + index],
            n => {
                let middle = first + n / 2;
                column(values, stride, index, first, middle)
                    + column(values, stride, index, middle, end)
            }
        }
    }
    (0..columns)
        .map(|i| column(values, columns, i, 0, rows))
        .collect()
}
