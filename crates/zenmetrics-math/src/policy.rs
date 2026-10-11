//! Policies select numerical approximations without adding SIMD backends.
use magetypes::simd::{backends::F32x8Convert, generic::f32x8};
#[cfg(feature = "avx512")]
use magetypes::simd::{backends::F32x16Convert, generic::f32x16};

/// A transcendental policy with lazy per-token monomorphization.
/// Implementations must document their domains and numerical error contract.
pub trait MathPolicy {
    /// Scalar exponential for short tails.
    fn scalar_exp(x: f32) -> f32;
    /// Scalar natural logarithm for short tails.
    fn scalar_ln(x: f32) -> f32;
    /// Scalar power for short tails.
    fn scalar_pow(x: f32, p: f32) -> f32;
    /// Natural exponential on eight lanes.
    fn exp8<T: F32x8Convert>(x: f32x8<T>) -> f32x8<T>;
    /// Natural logarithm on eight lanes.
    fn ln8<T: F32x8Convert>(x: f32x8<T>) -> f32x8<T>;
    /// Constant-exponent power on eight lanes.
    fn pow8<T: F32x8Convert>(x: f32x8<T>, p: f32) -> f32x8<T>;
    /// Natural exponential on sixteen lanes.
    #[cfg(feature = "avx512")]
    fn exp16<T: F32x16Convert>(x: f32x16<T>) -> f32x16<T>;
    /// Natural logarithm on sixteen lanes.
    #[cfg(feature = "avx512")]
    fn ln16<T: F32x16Convert>(x: f32x16<T>) -> f32x16<T>;
    /// Constant-exponent power on sixteen lanes.
    #[cfg(feature = "avx512")]
    fn pow16<T: F32x16Convert>(x: f32x16<T>, p: f32) -> f32x16<T>;
}

/// magetypes' unchecked medium-precision polynomial policy.
/// Positive normal log/pow inputs and finite normal-range exp results only.
#[derive(Clone, Copy, Debug, Default)]
pub struct Midp;
impl MathPolicy for Midp {
    #[inline(always)]
    fn scalar_exp(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.exp()
        }
        #[cfg(not(feature = "std"))]
        {
            f32x8::splat(archmage::ScalarToken, x).exp_midp().to_array()[0]
        }
    }
    #[inline(always)]
    fn scalar_ln(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.ln()
        }
        #[cfg(not(feature = "std"))]
        {
            f32x8::splat(archmage::ScalarToken, x).ln_midp().to_array()[0]
        }
    }
    #[inline(always)]
    fn scalar_pow(x: f32, p: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.powf(p)
        }
        #[cfg(not(feature = "std"))]
        {
            f32x8::splat(archmage::ScalarToken, x)
                .pow_midp(p)
                .to_array()[0]
        }
    }
    #[inline(always)]
    fn exp8<T: F32x8Convert>(x: f32x8<T>) -> f32x8<T> {
        x.exp_midp_unchecked()
    }
    #[inline(always)]
    fn ln8<T: F32x8Convert>(x: f32x8<T>) -> f32x8<T> {
        x.ln_midp_unchecked()
    }
    #[inline(always)]
    fn pow8<T: F32x8Convert>(x: f32x8<T>, p: f32) -> f32x8<T> {
        x.pow_midp_unchecked(p)
    }
    #[cfg(feature = "avx512")]
    #[inline(always)]
    fn exp16<T: F32x16Convert>(x: f32x16<T>) -> f32x16<T> {
        x.exp_midp_unchecked()
    }
    #[cfg(feature = "avx512")]
    #[inline(always)]
    fn ln16<T: F32x16Convert>(x: f32x16<T>) -> f32x16<T> {
        x.ln_midp_unchecked()
    }
    #[cfg(feature = "avx512")]
    #[inline(always)]
    fn pow16<T: F32x16Convert>(x: f32x16<T>, p: f32) -> f32x16<T> {
        x.pow_midp_unchecked(p)
    }
}
