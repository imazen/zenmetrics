//! Discoverability catalog (goal I): address a *result set* by its semantic identity — corpus,
//! codec, quality grid, metric, config — so an agent finds work by *what it is*, not by remembering
//! a `run_id`. The catalog key is content-addressed, so "does this result already exist?" is a
//! lookup and declaring the same result set twice is a structural no-op. Entries are *derived from
//! the ledger* (the caller materializes them), so the catalog can't drift from reality.

use serde::{Deserialize, Serialize};

use crate::content::{Sha256Hex, sha256};

/// Semantic identity of a result set — the human-meaningful description of "what work this is".
/// `q_grid`/`config` are *descriptors* (e.g. `"0..=100step5"`), not enumerated lists, so the same
/// logical sweep has one stable identity regardless of how it was launched.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticId {
    pub corpus: String,
    pub codec: String,
    pub q_grid: String,
    pub metric: String,
    pub config: String,
}

impl SemanticId {
    /// Content-addressed catalog key. Same description → same key (find-by-description, goal I).
    /// serde serializes struct fields in declaration order, so this is deterministic.
    pub fn key(&self) -> Sha256Hex {
        let canon = serde_json::to_vec(self).expect("SemanticId is serializable");
        sha256(&canon)
    }
}

/// A catalog row: what exists, plus provenance for the staleness-proof audit. `date` is passed in
/// (pure code has no clock); `build_commit` is the source SHA that produced the rows (the
/// `_MANIFEST.json` / DATA_PROVENANCE discipline, enforced as a field rather than prose).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: SemanticId,
    pub build_commit: String,
    pub date: String,
    pub rows: u64,
    pub r2_path: String,
}

impl CatalogEntry {
    pub fn key(&self) -> Sha256Hex {
        self.id.key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(metric: &str) -> SemanticId {
        SemanticId {
            corpus: "cid22".into(),
            codec: "zenjpeg".into(),
            q_grid: "0..=100step5".into(),
            metric: metric.into(),
            config: "xyb=on,trellis=on".into(),
        }
    }

    #[test]
    fn same_description_same_key() {
        assert_eq!(id("cvvdp").key(), id("cvvdp").key());
    }

    #[test]
    fn different_description_different_key() {
        assert_ne!(id("cvvdp").key(), id("ssim2").key());
    }

    #[test]
    fn find_by_description_not_by_runid() {
        // An agent that wants cvvdp over cid22 zenjpeg can compute the key with no run_id in hand.
        let wanted = id("cvvdp").key();
        let existing = CatalogEntry {
            id: id("cvvdp"),
            build_commit: "abc1234".into(),
            date: "2026-05-29".into(),
            rows: 513_570,
            r2_path: "s3://zentrain/cvvdp-v15rc/omni/".into(),
        };
        assert_eq!(
            existing.key(),
            wanted,
            "the wanted result set is discoverable by description"
        );
    }
}
