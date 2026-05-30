//! Garbage collection: a pure reachability mark-sweep over the blob index + the referenced set.
//!
//! Never a per-object crawl — the caller scans the columnar **blob-index** (a Parquet inventory) and
//! supplies `referenced` (a `SELECT DISTINCT output_sha FROM ledger WHERE reachable_from_roots`, one
//! columnar scan). This function only *decides*; the caller applies grace + tombstone + Tower-mirror
//! before any delete, and can render the plan as a dry-run preview (goal C).
//!
//! The two fears, designed out (goal G):
//! - **Overzealous** is impossible: a referenced blob is always `Keep` (structural), and an
//!   unreferenced-but-**not-regenerable** blob is `RefuseSurface` — never auto-deleted, escalated for
//!   an explicit pin/archive decision.
//! - **Poor / unbounded** is handled: unreferenced cheap-regenerable blobs are an LRU cache
//!   (`EvictCheap`), and expensive-regenerable ones evict only under budget pressure.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::content::Sha256Hex;
use crate::job::Regenerability;

/// One row of the Parquet blob inventory the GC scans.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobIndexEntry {
    pub sha: Sha256Hex,
    pub size: u64,
    pub regenerability: Regenerability,
    /// Unix seconds of last reference — the LRU key among cheap-regenerable evictions.
    pub last_ref_secs: u64,
}

/// The decision for one blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GcVerdict {
    /// Referenced or pinned — always kept.
    Keep,
    /// Unreferenced + cheap to rebuild → safe to drop (cache miss = cheap recompute).
    EvictCheap,
    /// Unreferenced + expensive to rebuild → drop only when over the storage budget.
    EvictUnderPressure,
    /// Unreferenced + not regenerable → never auto-delete; surface for a human pin/archive decision.
    RefuseSurface,
}

/// The full plan, partitioned by verdict. Caller sums sizes for the dry-run "how much would this
/// free" preview, then applies grace/tombstone/mirror to the eviction sets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GcPlan {
    pub keep: Vec<Sha256Hex>,
    pub evict_cheap: Vec<Sha256Hex>,
    pub evict_under_pressure: Vec<Sha256Hex>,
    pub refuse_surface: Vec<Sha256Hex>,
}

/// Compute the GC plan. `referenced` = shas reachable from roots (from the ledger); `roots` =
/// explicitly pinned shas always kept even if unreferenced. Pure; no I/O.
pub fn gc_plan(
    index: &[BlobIndexEntry],
    referenced: &HashSet<Sha256Hex>,
    roots: &HashSet<Sha256Hex>,
) -> GcPlan {
    let mut plan = GcPlan::default();
    for e in index {
        if referenced.contains(&e.sha) || roots.contains(&e.sha) {
            plan.keep.push(e.sha.clone());
            continue;
        }
        match e.regenerability {
            Regenerability::CheapRegenerable => plan.evict_cheap.push(e.sha.clone()),
            Regenerability::ExpensiveRegenerable => plan.evict_under_pressure.push(e.sha.clone()),
            Regenerability::NotRegenerable => plan.refuse_surface.push(e.sha.clone()),
        }
    }
    plan
}

/// LRU-capped eviction of the cheap-regenerable cache (goal G: "regenerable-cheap blobs = LRU-capped
/// cache, bounded, lossless rebuild"). Among unreferenced cheap blobs, keep the most-recently-used up
/// to `cheap_cap_bytes` and return the oldest ones to evict (smallest `last_ref_secs` first) until the
/// retained cheap bytes fit the cap. If the cache already fits, evicts nothing. Pure; no I/O.
pub fn lru_cap_evict(
    index: &[BlobIndexEntry],
    referenced: &HashSet<Sha256Hex>,
    roots: &HashSet<Sha256Hex>,
    cheap_cap_bytes: u64,
) -> Vec<Sha256Hex> {
    let mut cheap: Vec<&BlobIndexEntry> = index
        .iter()
        .filter(|e| {
            !referenced.contains(&e.sha)
                && !roots.contains(&e.sha)
                && e.regenerability == Regenerability::CheapRegenerable
        })
        .collect();
    let total: u64 = cheap.iter().map(|e| e.size).sum();
    if total <= cheap_cap_bytes {
        return Vec::new();
    }
    // Newest-first: keep MRU until the cap is reached; everything after is evicted (the LRU tail).
    cheap.sort_by(|a, b| b.last_ref_secs.cmp(&a.last_ref_secs));
    let mut kept = 0u64;
    let mut evict = Vec::new();
    for e in cheap {
        if kept + e.size <= cheap_cap_bytes {
            kept += e.size;
        } else {
            evict.push(e.sha.clone());
        }
    }
    evict
}

/// An audit + recovery record written *before* a blob is deleted (goal G: "grace + tombstone +
/// Tower-mirror-verify before any non-regenerable delete"). Persisted to the tombstone store so a
/// delete is always traceable and (within the grace window) reversible.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tombstone {
    pub sha: Sha256Hex,
    pub size: u64,
    pub regenerability: Regenerability,
    /// Why it was deleted: `lru_evict` / `under_pressure` / `force_irreplaceable`.
    pub reason: String,
    /// Unix seconds the tombstone was written (injected — pure code has no clock).
    pub deleted_at: u64,
    /// True iff a Tower-mirror copy was sha-verified present before deletion (required for any
    /// non-regenerable delete; trivially true for regenerable blobs, which can be rebuilt).
    pub mirror_verified: bool,
}

/// Classify a single blob (used by the dashboard's per-blob "why kept/evictable" explainer).
pub fn verdict(
    e: &BlobIndexEntry,
    referenced: &HashSet<Sha256Hex>,
    roots: &HashSet<Sha256Hex>,
) -> GcVerdict {
    if referenced.contains(&e.sha) || roots.contains(&e.sha) {
        return GcVerdict::Keep;
    }
    match e.regenerability {
        Regenerability::CheapRegenerable => GcVerdict::EvictCheap,
        Regenerability::ExpensiveRegenerable => GcVerdict::EvictUnderPressure,
        Regenerability::NotRegenerable => GcVerdict::RefuseSurface,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::sha256;

    fn entry(bytes: &[u8], regen: Regenerability) -> BlobIndexEntry {
        BlobIndexEntry {
            sha: sha256(bytes),
            size: bytes.len() as u64,
            regenerability: regen,
            last_ref_secs: 0,
        }
    }

    #[test]
    fn referenced_is_always_kept_even_if_cheap() {
        let e = entry(b"jpeg", Regenerability::CheapRegenerable);
        let referenced: HashSet<_> = [e.sha.clone()].into_iter().collect();
        assert_eq!(verdict(&e, &referenced, &HashSet::new()), GcVerdict::Keep);
    }

    #[test]
    fn root_pin_keeps_unreferenced() {
        let e = entry(b"source", Regenerability::NotRegenerable);
        let roots: HashSet<_> = [e.sha.clone()].into_iter().collect();
        assert_eq!(verdict(&e, &HashSet::new(), &roots), GcVerdict::Keep);
    }

    #[test]
    fn unreferenced_cheap_is_lru_evictable() {
        let e = entry(b"jpeg", Regenerability::CheapRegenerable);
        assert_eq!(verdict(&e, &HashSet::new(), &HashSet::new()), GcVerdict::EvictCheap);
    }

    #[test]
    fn unreferenced_expensive_only_under_pressure() {
        let e = entry(b"avif", Regenerability::ExpensiveRegenerable);
        assert_eq!(
            verdict(&e, &HashSet::new(), &HashSet::new()),
            GcVerdict::EvictUnderPressure
        );
    }

    #[test]
    fn unreferenced_irreplaceable_is_refused_not_deleted() {
        let e = entry(b"only-copy-source", Regenerability::NotRegenerable);
        assert_eq!(
            verdict(&e, &HashSet::new(), &HashSet::new()),
            GcVerdict::RefuseSurface,
            "never auto-delete an irreplaceable blob; escalate for a human decision"
        );
    }

    fn cheap_aged(bytes: &[u8], last_ref: u64) -> BlobIndexEntry {
        let mut e = entry(bytes, Regenerability::CheapRegenerable);
        e.last_ref_secs = last_ref;
        e
    }

    #[test]
    fn lru_cap_keeps_mru_evicts_oldest_until_under_cap() {
        // three 4-byte cheap blobs (12 bytes), cap 8 → evict the single oldest (keep 8 = 2 newest).
        let index = vec![
            cheap_aged(b"aaaa", 100), // newest
            cheap_aged(b"bbbb", 50),
            cheap_aged(b"cccc", 10), // oldest → evicted
        ];
        let evict = lru_cap_evict(&index, &HashSet::new(), &HashSet::new(), 8);
        assert_eq!(evict, vec![sha256(b"cccc")], "only the LRU tail is evicted");
    }

    #[test]
    fn lru_cap_evicts_nothing_when_cache_fits() {
        let index = vec![cheap_aged(b"aaaa", 100), cheap_aged(b"bbbb", 50)];
        assert!(lru_cap_evict(&index, &HashSet::new(), &HashSet::new(), 1000).is_empty());
    }

    #[test]
    fn lru_cap_never_evicts_referenced_or_irreplaceable() {
        let referenced_blob = cheap_aged(b"hot", 10);
        let referenced: HashSet<_> = [referenced_blob.sha.clone()].into_iter().collect();
        let index = vec![
            referenced_blob,                                       // referenced cheap — never evicted
            entry(b"source", Regenerability::NotRegenerable),      // irreplaceable — never in cheap LRU
            cheap_aged(b"cold", 1),                                // the only LRU candidate
        ];
        let evict = lru_cap_evict(&index, &referenced, &HashSet::new(), 0);
        assert_eq!(evict, vec![sha256(b"cold")], "only unreferenced cheap blobs are LRU-evictable");
    }

    #[test]
    fn plan_partitions_the_index() {
        let referenced_blob = entry(b"keep-me", Regenerability::CheapRegenerable);
        let referenced: HashSet<_> = [referenced_blob.sha.clone()].into_iter().collect();
        let index = vec![
            referenced_blob,
            entry(b"jpeg-orphan", Regenerability::CheapRegenerable),
            entry(b"avif-orphan", Regenerability::ExpensiveRegenerable),
            entry(b"source-orphan", Regenerability::NotRegenerable),
        ];
        let plan = gc_plan(&index, &referenced, &HashSet::new());
        assert_eq!(plan.keep.len(), 1);
        assert_eq!(plan.evict_cheap.len(), 1);
        assert_eq!(plan.evict_under_pressure.len(), 1);
        assert_eq!(plan.refuse_surface.len(), 1);
        // total accounted-for == index size (nothing lost or double-counted)
        let total = plan.keep.len()
            + plan.evict_cheap.len()
            + plan.evict_under_pressure.len()
            + plan.refuse_surface.len();
        assert_eq!(total, index.len());
    }
}
