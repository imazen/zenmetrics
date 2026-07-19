//! A deterministic, dependency-free PRNG for seeded fault schedules.
//!
//! xorshift64* — good enough to spread fault decisions, and identical on every
//! platform for a given seed, so a chaos test that fails reproduces byte-for-byte
//! from its seed. Never use this for anything cryptographic; it exists only to
//! make "inject a transient error ~5% of the time" reproducible.

/// Seeded xorshift64* generator.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// New generator from `seed`. A zero seed is remapped (xorshift can't leave
    /// the all-zero state) so `Rng::new(0)` still produces a stream.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    /// Next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform float in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        // Top 53 bits → f64 mantissa, the standard trick.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// `true` with probability `p`. `p <= 0` never fires; `p >= 1` always does.
    pub fn chance(&mut self, p: f64) -> bool {
        if p <= 0.0 {
            false
        } else if p >= 1.0 {
            true
        } else {
            self.unit() < p
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_for_a_seed() {
        let a: Vec<u64> = (0..8).scan(Rng::new(42), |r, _| Some(r.next_u64())).collect();
        let b: Vec<u64> = (0..8).scan(Rng::new(42), |r, _| Some(r.next_u64())).collect();
        assert_eq!(a, b, "same seed → same stream");
        let c: Vec<u64> = (0..8).scan(Rng::new(43), |r, _| Some(r.next_u64())).collect();
        assert_ne!(a, c, "different seed → different stream");
    }

    #[test]
    fn chance_bounds_are_exact() {
        let mut r = Rng::new(1);
        assert!(!r.chance(0.0));
        assert!(r.chance(1.0));
    }

    #[test]
    fn chance_is_roughly_calibrated() {
        let mut r = Rng::new(7);
        let hits = (0..10_000).filter(|_| r.chance(0.25)).count();
        // Deterministic stream, so this is a fixed number — just assert it's in
        // a sane band around 2500 (catches a broken `unit()`).
        assert!((2000..3000).contains(&hits), "got {hits} hits for p=0.25");
    }
}
