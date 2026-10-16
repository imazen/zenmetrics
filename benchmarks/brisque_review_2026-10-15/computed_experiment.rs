//! Review experiment: f64 libm gamma functions, rounded once per table value.
use core::ops::Deref;
use std::sync::OnceLock;
pub(crate) struct Grid {
    kind: u8,
    data: OnceLock<[f32; super::NGAM]>,
}
impl Grid {
    const fn new(kind: u8) -> Self {
        Self {
            kind,
            data: OnceLock::new(),
        }
    }
}
impl Deref for Grid {
    type Target = [f32; super::NGAM];
    fn deref(&self) -> &Self::Target {
        self.data.get_or_init(|| {
            let gam = super::gam_grid();
            core::array::from_fn(|i| {
                let a = f64::from(gam[i]);
                let l1 = libm::lgamma(1.0 / a);
                let l2 = libm::lgamma(2.0 / a);
                let l3 = libm::lgamma(3.0 / a);
                libm::exp(match self.kind {
                    0 => l1 + l3 - 2.0 * l2,
                    1 => 2.0 * l2 - l1 - l3,
                    _ => l2 - (l1 + l3) / 2.0,
                }) as f32
            })
        })
    }
}
pub(crate) static R_GGD: Grid = Grid::new(0);
pub(crate) static R_AGGD: Grid = Grid::new(1);
pub(crate) static ETA_FACTOR: Grid = Grid::new(2);
