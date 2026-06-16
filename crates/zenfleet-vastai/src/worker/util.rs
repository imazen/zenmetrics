//! Worker-internal utilities.

use std::pin::Pin;

use sha2::{Digest, Sha256};

/// Deterministic shuffle keyed by `seed`. Same seed -> same output.
/// We use sha256(seed) as the rng seed so the shuffle is stable
/// across reruns (resumability) and unique per worker (boot-time
/// claim contention).
///
/// Implementation: a Fisher-Yates shuffle driven by a sha256-based
/// keyed PRNG. We do NOT pull `rand`/`rand_chacha` into the binary
/// just for this — sha256 of (seed || index) gives 256 bits we
/// can chop into u64s for swaps.
pub fn seeded_shuffle(items: &[String], seed: &str) -> Vec<String> {
    let mut out = items.to_vec();
    if out.len() <= 1 {
        return out;
    }
    let mut hasher_seed = Sha256::new();
    hasher_seed.update(seed.as_bytes());
    let seed_hash = hasher_seed.finalize();

    let n = out.len();
    for i in (1..n).rev() {
        // hash(seed_hash || i) -> 8 bytes -> u64 -> reduce mod (i+1)
        let mut h = Sha256::new();
        h.update(seed_hash);
        h.update(i.to_le_bytes());
        let digest = h.finalize();
        let chunk: [u8; 8] = digest[..8].try_into().expect("sha256 has 8+ bytes");
        let r = (u64::from_le_bytes(chunk) % (i as u64 + 1)) as usize;
        out.swap(i, r);
    }
    out
}

/// Wait for SIGTERM or SIGINT. The returned future is pinned so
/// callers can borrow it inside a `tokio::select!` without
/// repeatedly re-arming the signal.
pub fn shutdown_signal() -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(async {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
        tokio::select! {
            _ = sigterm.recv() => tracing::warn!("SIGTERM"),
            _ = sigint.recv()  => tracing::warn!("SIGINT"),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_order() {
        let items: Vec<String> = (0..10).map(|i| format!("c{i}")).collect();
        let a = seeded_shuffle(&items, "worker-A");
        let b = seeded_shuffle(&items, "worker-A");
        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_different_order() {
        let items: Vec<String> = (0..100).map(|i| format!("c{i}")).collect();
        let a = seeded_shuffle(&items, "worker-A");
        let b = seeded_shuffle(&items, "worker-B");
        assert_ne!(a, b, "different seeds should produce different orderings");
    }

    #[test]
    fn preserves_set() {
        let items: Vec<String> = (0..20).map(|i| format!("c{i}")).collect();
        let shuffled = seeded_shuffle(&items, "seed");
        let mut a = items.clone();
        let mut b = shuffled.clone();
        a.sort();
        b.sort();
        assert_eq!(a, b, "shuffle must be a permutation");
    }
}
