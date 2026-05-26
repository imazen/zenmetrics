#![forbid(unsafe_code)]

//! The typed per-pair join key — the type-level corruption fix.
//!
//! # The bug this prevents (DATA_INTEGRITY_root_cause_2026-05-25.md, Mode B)
//!
//! A per-pair metric (`ssim2_gpu`, one value per `(image_path, codec, q,
//! knob_tuple_json)`) was joined onto a per-corpus features table that
//! carried **only `ref_basename`**. The Python join helper computed its
//! effective key as the intersection of requested keys with columns
//! actually present, so the key silently collapsed to `["ref_basename"]`.
//! It then did `groupby("ref_basename").mean()` and broadcast one mean
//! score onto all ~125 distortions of each reference — destroying the
//! per-distortion signal it was supposed to carry. The result trained on
//! garbage.
//!
//! `join_safety.py` defends this **at runtime** with `REF_ONLY_KEYS` +
//! an explicit refusal. This module defends it **at compile time**: the
//! join API ([`super::join::safe_join`]) takes a [`PairKey`], and a
//! `PairKey` cannot be constructed without all four per-pair fields. There
//! is no `PairKey { ref_basename }` constructor. A caller physically cannot
//! express the ref-only collapse — the type system rejects it before any
//! data is touched.
//!
//! This is the type-level equivalent of `REF_ONLY_KEYS`: the runtime guard
//! still exists ([`super::join::safe_join`] re-checks at the data layer for
//! the dynamic-column case), but the *primary* defense is that the bad call
//! is unrepresentable.

use super::table::{AssembleError, Table};

/// The columns that, ON THEIR OWN, are NOT a per-pair key. Joining a
/// per-pair metric on any subset of these alone is the Mode-B misjoin.
/// Mirrors `join_safety.REF_ONLY_KEYS`. Used by the dynamic-key runtime
/// guard in [`super::join`]; the static `PairKey` makes the common path
/// unable to reach it.
pub const REF_ONLY_KEYS: &[&str] = &["ref_basename", "ref", "source"];

/// The canonical per-pair key column names, in the order the score sidecars
/// (`scores/{ssim2,iwssim,cvvdp}_imazen*.parquet`) store them. `image_path`
/// is the per-pair distortion identifier; the other three disambiguate the
/// encode that produced it.
pub const IMAGE_PATH: &str = "image_path";
pub const CODEC: &str = "codec";
pub const Q: &str = "q";
pub const KNOB_TUPLE_JSON: &str = "knob_tuple_json";

/// The per-pair join key. **Its existence as a struct with four required
/// fields is the corruption fix.**
///
/// A `PairKey` value is the materialised key for one row: the four field
/// strings, concatenated by the canonical join. You build a `PairKey` for a
/// row of a [`Table`] only via [`PairKey::for_row`], which requires every
/// per-pair column to be present on the table — so a features table that
/// dropped `codec`/`q`/`knob_tuple_json` (leaving only `ref_basename`)
/// cannot produce `PairKey`s, and [`super::join::safe_join`] refuses it.
///
/// Contrast with the Python `available_keys = [k for k in join_keys if k in
/// target.columns]` pattern, which *silently* shrank the key to whatever was
/// present. There is no shrinking here: all four or nothing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PairKey {
    pub image_path: String,
    pub codec: String,
    /// `q` is keyed as its canonical string form so an `Int64` `q` on the
    /// metric side and a textual `q` on the features side compare equal.
    pub q: String,
    pub knob_tuple_json: String,
}

impl PairKey {
    /// The four key column names. There is intentionally NO accessor that
    /// returns a subset — callers cannot ask for "just ref_basename".
    pub const COLUMNS: [&'static str; 4] = [IMAGE_PATH, CODEC, Q, KNOB_TUPLE_JSON];

    /// Construct the key for row `i` of `table`. Returns
    /// [`AssembleError::JoinSafety`] if the table lacks ANY of the four
    /// per-pair columns — this is where a ref-only table is rejected.
    ///
    /// The check is performed once up front by [`PairKey::require_columns`];
    /// `for_row` assumes the columns exist (it is only called after that
    /// check passes) and reads them positionally.
    ///
    /// `#[allow(dead_code)]`: called by [`super::join::safe_join`], itself a
    /// public primitive the lib build doesn't invoke (see its note).
    #[allow(dead_code)]
    pub fn for_row(table: &Table, i: usize) -> PairKey {
        // Safe to unwrap: callers must call `require_columns` first.
        let col = |name: &str| table.column(name).expect("PairKey column missing");
        PairKey {
            image_path: col(IMAGE_PATH).key_at(i),
            codec: col(CODEC).key_at(i),
            q: col(Q).key_at(i),
            knob_tuple_json: col(KNOB_TUPLE_JSON).key_at(i),
        }
    }

    /// Verify `table` carries every per-pair key column. Call once before a
    /// loop of [`PairKey::for_row`]. Returns a [`AssembleError::JoinSafety`]
    /// naming the missing columns and pointing at the alternative
    /// (`attach_positional`) — the same guidance `join_safety.py` gives.
    pub fn require_columns(label: &str, table: &Table) -> Result<(), AssembleError> {
        let missing: Vec<&str> = Self::COLUMNS
            .iter()
            .copied()
            .filter(|c| !table.has_column(c))
            .collect();
        if !missing.is_empty() {
            // If the ONLY keys the table carries are ref-only ones, name the
            // Mode-B bug explicitly — that's the diagnostic operators search
            // for.
            let present_ref_only: Vec<&str> = REF_ONLY_KEYS
                .iter()
                .copied()
                .filter(|k| table.has_column(k))
                .collect();
            return Err(AssembleError::JoinSafety(format!(
                "{label}: target table is missing per-pair key column(s) {missing:?} \
                 (it carries {present_ref_only:?}). Joining a per-pair metric on \
                 ref-only keys would broadcast one value per reference onto all its \
                 distortions (the ssim2_gpu ref-misjoin bug, Mode B). Carry the full \
                 per-pair key {cols:?} on the features table, or attach the metric \
                 POSITIONALLY via attach_positional.",
                cols = Self::COLUMNS
            )));
        }
        Ok(())
    }
}
