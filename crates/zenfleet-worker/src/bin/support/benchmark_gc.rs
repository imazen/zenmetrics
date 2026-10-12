//! Retain every benchmark attempt and its inputs before latest-wins folding.
use std::collections::{BTreeSet, HashSet};
use std::io;
use zenfleet_core::{JobKind, LedgerRow, Sha256Hex};

#[derive(Default)]
pub(super) struct Evidence {
    bundles: BTreeSet<Sha256Hex>,
    receipts: BTreeSet<Sha256Hex>,
    roots: HashSet<Sha256Hex>,
}
impl Evidence {
    pub(super) fn observe(&mut self, row: &LedgerRow) -> Result<(), Box<dyn std::error::Error>> {
        if let JobKind::Benchmark {
            bundle_sha,
            runner_sha,
            ..
        } = &row.kind
        {
            self.bundles.insert(Sha256Hex::parse(bundle_sha)?);
            self.roots.insert(Sha256Hex::parse(runner_sha)?);
            if let Some(output) = &row.output_sha {
                self.receipts.insert(Sha256Hex::parse(output.as_str())?);
            }
        }
        Ok(())
    }
    pub(super) fn expand(
        mut self,
        mut read: impl FnMut(&Sha256Hex) -> Result<Vec<u8>, Box<dyn std::error::Error>>,
    ) -> Result<HashSet<Sha256Hex>, Box<dyn std::error::Error>> {
        for (sha, receipt) in self
            .bundles
            .iter()
            .map(|s| (s, false))
            .chain(self.receipts.iter().map(|s| (s, true)))
        {
            let bytes = read(sha)?;
            if bytes.len() > 8 * 1024 * 1024 || zenfleet_core::content::sha256(&bytes) != *sha {
                return Err(
                    io::Error::other("benchmark metadata size/hash mismatch; GC refused").into(),
                );
            }
            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
            if value["schema"] != 1 {
                return Err(
                    io::Error::other("unknown benchmark evidence schema; GC refused").into(),
                );
            }
            let entries = if receipt {
                if value["kind"] != "zenfleet_benchmark_observation"
                    || value["retention"]["regenerability"] != "not_regenerable"
                {
                    return Err(io::Error::other("invalid benchmark receipt; GC refused").into());
                }
                let refs = value["retention"]["references"]
                    .as_array()
                    .ok_or_else(|| io::Error::other("missing evidence references"))?;
                for reference in refs {
                    self.roots.insert(Sha256Hex::parse(
                        reference
                            .as_str()
                            .ok_or_else(|| io::Error::other("invalid evidence reference"))?,
                    )?);
                }
                &value["artifacts"]
            } else {
                &value["files"]
            };
            for entry in entries
                .as_object()
                .ok_or_else(|| io::Error::other("missing benchmark artifacts"))?
                .values()
            {
                self.roots.insert(Sha256Hex::parse(
                    entry["sha256"]
                        .as_str()
                        .ok_or_else(|| io::Error::other("invalid artifact hash"))?,
                )?);
            }
            self.roots.insert(sha.clone());
        }
        Ok(self.roots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use zenfleet_core::{
        BlobIndexEntry, CellId, JobId, JobStatus, Regenerability, content::sha256, gc_plan,
    };

    #[test]
    fn old_attempt_inputs_and_nested_artifacts_survive_zero_budget_gc() {
        let input = sha256(b"input");
        let trace = sha256(b"trace");
        let old_trace = sha256(b"earlier attempt trace");
        let runner = sha256(b"runner");
        let bundle =
            serde_json::to_vec(&json!({"schema":1,"files":{"input":{"sha256":input}}})).unwrap();
        let bundle_sha = sha256(&bundle);
        let receipt = |trace: &Sha256Hex| {
            serde_json::to_vec(&json!({"schema":1,"kind":"zenfleet_benchmark_observation",
            "artifacts":{"trace":{"sha256":trace}},"retention":{"regenerability":"not_regenerable","references":[trace]}})).unwrap()
        };
        let (old, new) = (receipt(&old_trace), receipt(&trace));
        let blobs = [
            (bundle_sha.clone(), bundle),
            (sha256(&old), old),
            (sha256(&new), new),
        ];
        let kind = JobKind::Benchmark {
            bundle_sha: bundle_sha.to_string(),
            runner_sha: runner.to_string(),
            experiment: "control".into(),
            machine: "cpu".into(),
            domain: "unrestricted".into(),
            replicate: 0,
            backend: "wall".into(),
            timeout_seconds: 1,
            admission_seconds: 0,
        };
        let mut evidence = Evidence::default();
        for (i, (output, _)) in blobs[1..].iter().enumerate() {
            let row = LedgerRow {
                job_id: JobId::of(&kind, &[]),
                kind: kind.clone(),
                cell: CellId {
                    image_path: "input".into(),
                    codec: "benchmark".into(),
                    q: 0,
                    knob_tuple_json: "{}".into(),
                },
                output_sha: Some(output.clone()),
                status: JobStatus::Done,
                error_class: None,
                attempts: i as u32 + 1,
                ts: i as u64,
                worker: "control".into(),
                provider: "local".into(),
            };
            evidence.observe(&row).unwrap();
        }
        let roots = evidence
            .expand(|s| Ok(blobs.iter().find(|(hash, _)| hash == s).unwrap().1.clone()))
            .unwrap();
        let expected = [
            input,
            trace,
            old_trace,
            runner,
            blobs[0].0.clone(),
            blobs[1].0.clone(),
            blobs[2].0.clone(),
        ];
        for sha in &expected {
            assert!(roots.contains(sha));
        }
        let orphan = sha256(b"unrelated cache");
        let index: Vec<_> = expected
            .iter()
            .chain([&orphan])
            .map(|sha| BlobIndexEntry {
                sha: sha.clone(),
                size: 1,
                regenerability: Regenerability::CheapRegenerable,
                last_ref_secs: 0,
            })
            .collect();
        let plan = gc_plan(&index, &roots, &HashSet::new());
        assert_eq!(plan.keep.len(), expected.len());
        assert_eq!(plan.evict_cheap, [orphan]);
    }

    #[test]
    fn changed_or_unknown_metadata_refuses_collection() {
        for bytes in [b"changed".as_slice(), br#"{"schema":2,"files":{}}"#] {
            let mut evidence = Evidence::default();
            evidence.bundles.insert(if bytes == b"changed" {
                sha256(b"expected")
            } else {
                sha256(bytes)
            });
            assert!(evidence.expand(|_| Ok(bytes.to_vec())).is_err());
        }
    }
}
