//! Resolved-state dedup for the `--knob-grid` path.
//!
//! `--plan` cells are deduplicated by the codec's own planner: `cross()`
//! resolves each stratum into a real encoder config and merges any two whose
//! byte-identity fingerprint matches (`zenjpeg/docs/VARIANT_GENERATION.md` §4 —
//! 46 % of the naive cross product on zenjpeg `rd_core × Step5`). The
//! `--knob-grid` path had no such step: `grid::KnobGrid::iter_tuples` is a
//! face-value cartesian product, so two spellings that resolve to the same
//! encoder state each got their own encode job.
//!
//! That is not hypothetical. The AVIF svt-rs / aom-rs subsample sweep declared
//! `32 sources × 10 speeds × 30 q` per backend and **17.8 % of it is byte-wise
//! duplicate work** (`benchmarks/avif_sweep_permutation_retrofit_2026-09-01.md`):
//!
//! * svt-rs speeds 7, 8, 9 and 10 all resolve to SVT preset 9 (C remaps every
//!   all-intra preset above M9 down to M9), and
//! * q 98 and q 100 resolve to the same quantizer on BOTH backends (each
//!   backend clamps its lossy dial away from the lossless quantizer).
//!
//! This module gives the knob-grid path the same resolved-state identity the
//! plan path has, so `--dry-run --emit-cells` declares one cell per distinct
//! encode instead of one per spelling.
//!
//! # The rule for adding a codec
//!
//! [`knob_cell_identity`] returns `None` for anything without a registered
//! resolver, and `None` means **no dedup** — one cell per spelling, the
//! row-safe baseline. Never guess: a resolver must route through the same code
//! the encoder runs (that is why the AVIF arm calls
//! `encode::avif_config_from_knobs` and `encode::aom_rs_cq_level` rather than
//! mirroring their arithmetic), and every merge it induces must be proven by
//! encode, not by reading code.

use super::encode::CodecKind;
use serde_json::{Map, Value};

/// The resolved-state identity of one `--knob-grid` cell: two cells with equal
/// identities produce byte-identical output for the same source image.
///
/// `Ok(None)` = this codec/knobset has no registered resolver, so the caller
/// must NOT dedup it.
pub fn knob_cell_identity(
    codec: CodecKind,
    q: f64,
    knobs: &Map<String, Value>,
) -> Result<Option<u64>, String> {
    match codec {
        #[cfg(all(feature = "sweep", feature = "avif"))]
        CodecKind::Zenavif => avif_identity(q, knobs),
        _ => Ok(None),
    }
}

#[cfg(all(feature = "sweep", feature = "avif"))]
fn avif_identity(q: f64, knobs: &Map<String, Value>) -> Result<Option<u64>, String> {
    match knobs.get("backend").and_then(Value::as_str) {
        // The zenav1-aom port. Its bitstream is a function of the ALLINTRA
        // config, which this arm drives from exactly two knobs: `--cq-level`
        // (via `aom_rs_cq_level`, the owner of the q mapping) and `--cpu-used`.
        // Anything else is rejected by `encode_avif_aom_rs` itself, so a knob
        // this identity cannot see cannot reach the encoder either.
        Some("aom-rs") => {
            #[cfg(feature = "avif-aom")]
            {
                if let Some(unknown) = knobs
                    .keys()
                    .find(|k| !["backend", "speed"].contains(&k.as_str()))
                {
                    return Err(format!(
                        "zenavif aom-rs backend: knob '{unknown}' is not wired (supported: \
                         backend, speed); refusing to dedup a cell whose identity it cannot see"
                    ));
                }
                let speed = knobs.get("speed").and_then(Value::as_u64).unwrap_or(6);
                let cq = super::encode::aom_rs_cq_level(q);
                // FNV-64, the same shape every codec's `sweep::fingerprint`
                // uses. The leading tag keeps aom-rs identities disjoint from
                // zenavif's fingerprint space.
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for b in b"aom-rs"
                    .iter()
                    .copied()
                    .chain(speed.to_le_bytes())
                    .chain((cq as u32).to_le_bytes())
                {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
                Ok(Some(h))
            }
            #[cfg(not(feature = "avif-aom"))]
            {
                let _ = q;
                Ok(None)
            }
        }
        // zenravif and svt-rs are both `zenavif::EncoderConfig` backends, so
        // the codec's OWN fingerprint is the identity — it resolves the
        // quality curve, the speed-derived search settings and (since the
        // 2026-09-01 backend axis) the svt-rs preset/QP mediators.
        _ => {
            let cfg = super::encode::avif_config_from_knobs(q, knobs)
                .map_err(|e| format!("zenavif knob tuple does not resolve: {e}"))?;
            Ok(Some(zenavif::sweep::fingerprint(&cfg)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knobs(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    /// The exact alias class the live AVIF subsample sweep was burning CPU on:
    /// 21 svt-rs + 7 aom-rs identical-`output_sha` groups in its ledger, every
    /// one of them `{98, 100}` at a fixed speed.
    #[cfg(feature = "avif-aom")]
    #[test]
    fn aom_rs_q98_and_q100_share_one_identity() {
        let k = knobs(&[
            ("backend", Value::from("aom-rs")),
            ("speed", Value::from(4)),
        ]);
        let a = knob_cell_identity(CodecKind::Zenavif, 98.0, &k).unwrap();
        let b = knob_cell_identity(CodecKind::Zenavif, 100.0, &k).unwrap();
        assert_eq!(a, b, "q98 and q100 both resolve to cq_level 1");
        // Control: the neighbouring grid point must NOT merge.
        let c = knob_cell_identity(CodecKind::Zenavif, 96.0, &k).unwrap();
        assert_ne!(a, c, "q96 resolves to cq_level 3 and is a distinct encode");
    }

    #[cfg(feature = "avif-aom")]
    #[test]
    fn aom_rs_speed_dial_is_injective() {
        // Unlike svt-rs, every --cpu-used value is a distinct encode.
        let mut seen = std::collections::HashSet::new();
        for s in 0..=9u64 {
            let k = knobs(&[
                ("backend", Value::from("aom-rs")),
                ("speed", Value::from(s)),
            ]);
            assert!(
                seen.insert(knob_cell_identity(CodecKind::Zenavif, 50.0, &k).unwrap()),
                "aom-rs speed {s} collided"
            );
        }
    }

    #[cfg(feature = "avif-svt")]
    #[test]
    fn svt_rs_m9_speed_class_shares_one_identity() {
        let id = |s: u64, q: f64| {
            knob_cell_identity(
                CodecKind::Zenavif,
                q,
                &knobs(&[
                    ("backend", Value::from("svt-rs")),
                    ("speed", Value::from(s)),
                ]),
            )
            .unwrap()
        };
        let m9 = id(7, 50.0);
        for s in [8u64, 9, 10] {
            assert_eq!(id(s, 50.0), m9, "svt-rs speed {s} must merge into preset 9");
        }
        // Control: speed 6 is preset 7 and must stay distinct.
        assert_ne!(id(6, 50.0), m9);
        // And the q alias holds on this backend too.
        assert_eq!(id(4, 98.0), id(4, 100.0));
        assert_ne!(id(4, 96.0), id(4, 98.0));
    }

    #[cfg(feature = "avif-svt")]
    #[test]
    fn svt_rs_and_zenravif_never_merge() {
        let svt = knob_cell_identity(
            CodecKind::Zenavif,
            50.0,
            &knobs(&[
                ("backend", Value::from("svt-rs")),
                ("speed", Value::from(4)),
            ]),
        )
        .unwrap();
        let zen = knob_cell_identity(
            CodecKind::Zenavif,
            50.0,
            &knobs(&[("speed", Value::from(4))]),
        )
        .unwrap();
        assert_ne!(svt, zen);
    }

    /// A codec with no registered resolver must return `None` — no dedup, one
    /// cell per spelling. Silence here is the row-safe default.
    #[test]
    fn unregistered_codecs_are_not_deduped() {
        for codec in [CodecKind::Zenjpeg, CodecKind::Zenwebp, CodecKind::Zenjxl] {
            assert_eq!(
                knob_cell_identity(codec, 50.0, &Map::new()).unwrap(),
                None,
                "{} must not dedup without a registered resolver",
                codec.name()
            );
        }
    }
}
