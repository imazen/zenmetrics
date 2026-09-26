// f64 Prewitt arithmetic for the MS-GMSD paper variant.
// The caller supplies its lane-vector loader and a vector containing 1/3.
// Preserve the operation order: no FMA or horizontal lane reduction.
macro_rules! chroma_prewitt {
    ($sample:ident, $plane:expr, $t:ident) => {{
        let u0 = $sample!($plane, 0, 0);
        let m0 = $sample!($plane, 1, 0);
        let d0 = $sample!($plane, 2, 0);
        let u1 = $sample!($plane, 0, 1);
        let d1 = $sample!($plane, 2, 1);
        let u2 = $sample!($plane, 0, 2);
        let m2 = $sample!($plane, 1, 2);
        let d2 = $sample!($plane, 2, 2);
        let gx = ((((-$t * u0 - $t * m0) - $t * d0) + $t * u2) + $t * m2) + $t * d2;
        let gy = ((((-$t * u0 + $t * d0) - $t * u1) + $t * d1) - $t * u2) + $t * d2;
        (gx * gx + gy * gy).sqrt()
    }};
}
