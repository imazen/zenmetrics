//! Stable identities: the human/training identity tuple (kept for compatibility with the existing
//! Parquet ledger), and the content-addressed [`JobId`] that makes enqueue idempotent and lets
//! agents discover and avoid duplicate work (goals A & I).

use serde::{Deserialize, Serialize};

use crate::content::{Sha256Hex, sha256};
use crate::job::JobKind;

/// The existing per-cell identity tuple. Retained as the human/training key alongside content
/// hashes; the Parquet `assemble` join and all training data depend on it. Content addressing is
/// *additive* — this tuple is not going away.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellId {
    pub image_path: String,
    pub codec: String,
    pub q: i64,
    pub knob_tuple_json: String,
}

impl CellId {
    /// Stable string key matching the Parquet identity tuple (US-separated to avoid delimiter
    /// collisions with paths/JSON).
    pub fn tuple_key(&self) -> String {
        format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}",
            self.image_path, self.codec, self.q, self.knob_tuple_json
        )
    }
}

/// Content-addressed job identity: SHA-256 over the canonical (kind, sorted+deduped input shas).
///
/// Same logical job → same `JobId` → enqueue is a no-op (idempotent by construction, goal A).
/// Different inputs or kind → different `JobId`. This is the anti-duplicate-work primitive (goal I):
/// an agent can compute the `JobId` for the work it *wants*, ask whether that result exists, and skip
/// if so — without remembering any `run_id`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JobId(pub Sha256Hex);

/// Canonical wire form for hashing — a plain **struct**, deliberately not a `serde_json::Value`
/// built via the `json!` macro. A struct's `#[derive(Serialize)]` always emits fields in
/// declaration order, which is fixed at compile time and independent of any cargo feature; a
/// `json!({...})` map is a `serde_json::Value::Object`, backed by `serde_json::Map`, whose
/// iteration/serialization order flips from alphabetical (the default `BTreeMap`) to
/// macro-literal insertion order (`IndexMap`) the instant ANY crate anywhere in the same build
/// enables `serde_json/preserve_order` — cargo unifies feature flags workspace-wide, so this was
/// sensitive to what ELSE got compiled alongside `zenfleet-core`, not just this crate's own
/// `Cargo.toml`. That was zenmetrics#38 (see `CLAUDE.md`'s "Known Bugs": `zenmetrics-cli` enables
/// `preserve_order` for its own `compare`-output ordering, and `zenfleet-sweep`/`zenfleet-vastai`
/// pull it in transitively through default features, so `cargo test -p zenfleet-core
/// -p zenfleet-sweep` computed a DIFFERENT `JobId` for the same logical job than
/// `cargo test -p zenfleet-core` alone — reproduced live 2026-08-24, golden hash
/// `6e79bec221...` (plain) vs `797c1300af...` (co-compiled with zenmetrics-cli)).
///
/// Field order below (`inputs` before `kind`) is chosen to reproduce the historical
/// alphabetically-sorted-by-key byte output exactly (`i` < `k`), so this fix changes NO existing
/// content-addressed `JobId`, ledger row, or manifest computed by a plain (non-`preserve_order`)
/// build — verified against `job::tests::scorefile_sdr_serialization_and_job_id_are_golden_stable`
/// under both a plain build and one co-compiled with `zenmetrics-cli`.
#[derive(Serialize)]
struct JobIdCanonForm<'a> {
    inputs: &'a [&'a str],
    kind: &'a JobKind,
}

impl JobId {
    pub fn of(kind: &JobKind, input_shas: &[Sha256Hex]) -> Self {
        // Canonicalize: sort + dedup inputs so input order can't change the identity.
        let mut inputs: Vec<&str> = input_shas.iter().map(Sha256Hex::as_str).collect();
        inputs.sort_unstable();
        inputs.dedup();
        let canon = JobIdCanonForm {
            inputs: &inputs,
            kind,
        };
        let bytes = serde_json::to_vec(&canon).expect("JobKind is always serializable");
        JobId(sha256(&bytes))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobid_is_idempotent() {
        let k = JobKind::Metric {
            metric: "ssim2".into(),
        };
        let a = sha256(b"encodeA");
        assert_eq!(
            JobId::of(&k, std::slice::from_ref(&a)),
            JobId::of(&k, std::slice::from_ref(&a)),
            "same logical job must produce the same JobId (no-op enqueue)"
        );
    }

    #[test]
    fn jobid_is_input_order_independent() {
        let k = JobKind::Bake {
            view: "train".into(),
        };
        let a = sha256(b"a");
        let b = sha256(b"b");
        assert_eq!(
            JobId::of(&k, &[a.clone(), b.clone()]),
            JobId::of(&k, &[b, a])
        );
    }

    #[test]
    fn jobid_differs_on_kind_and_inputs() {
        let a = sha256(b"a");
        let cvvdp = JobKind::Metric {
            metric: "cvvdp".into(),
        };
        let ssim2 = JobKind::Metric {
            metric: "ssim2".into(),
        };
        assert_ne!(
            JobId::of(&cvvdp, std::slice::from_ref(&a)),
            JobId::of(&ssim2, std::slice::from_ref(&a))
        );
        assert_ne!(
            JobId::of(&cvvdp, std::slice::from_ref(&a)),
            JobId::of(&cvvdp, &[sha256(b"other")])
        );
    }

    #[test]
    fn cell_tuple_key_is_stable() {
        let c = CellId {
            image_path: "x.png".into(),
            codec: "zenjpeg".into(),
            q: 80,
            knob_tuple_json: "{}".into(),
        };
        assert_eq!(c.tuple_key(), c.clone().tuple_key());
        assert!(c.tuple_key().contains("zenjpeg"));
    }
}
