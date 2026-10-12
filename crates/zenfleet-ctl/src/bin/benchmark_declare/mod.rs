use serde::Deserialize;
use zenfleet_core::{CellId, DesiredJob, JobKind, ResourceHint, Sha256Hex};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    schema: u32,
    observations: Vec<Observation>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    job: JobKind,
    peak_mem_bytes: u64,
    threads: u32,
}
pub(super) fn declare(bytes: &[u8]) -> Result<Vec<DesiredJob>, String> {
    let spec: Spec = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if spec.schema != 1 || spec.observations.is_empty() {
        return Err("expected schema 1 and nonempty observations".into());
    }
    let mut jobs = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for o in spec.observations {
        let JobKind::Benchmark {
            bundle_sha,
            runner_sha,
            experiment,
            machine,
            domain,
            backend,
            timeout_seconds,
            admission_seconds,
            ..
        } = &o.job
        else {
            return Err("declare-benchmarks accepts benchmark jobs only".into());
        };
        let bundle = Sha256Hex::parse(bundle_sha.clone()).map_err(|e| e.to_string())?;
        let runner = Sha256Hex::parse(runner_sha.clone()).map_err(|e| e.to_string())?;
        for s in [experiment, machine, domain] {
            if s.is_empty()
                || s.len() > 256
                || s.chars()
                    .any(|c| !c.is_ascii_alphanumeric() && !"._-".contains(c))
            {
                return Err(
                    "identities require 1..256 ASCII letters/digits/dot/underscore/dash".into(),
                );
            }
        }
        if !["wall", "os_peak", "heaptrack", "driver"].contains(&backend.as_str())
            || *timeout_seconds == 0
            || *timeout_seconds > 86400
            || *admission_seconds > 86400
            || o.peak_mem_bytes == 0
            || o.threads == 0
        {
            return Err("invalid backend, budget or resource hint".into());
        }
        let requires = o.job.required_capabilities();
        let job = DesiredJob::new(
            o.job,
            vec![bundle, runner],
            CellId {
                image_path: "benchmark".into(),
                codec: "benchmark".into(),
                q: 0,
                knob_tuple_json: "{}".into(),
            },
        )
        .with_requires(requires)
        .with_hint(ResourceHint {
            peak_mem_bytes: o.peak_mem_bytes,
            threads: o.threads,
            vram_bytes: None,
        });
        if !ids.insert(job.job_id()) {
            return Err("duplicate planned observation; use a distinct replicate".into());
        }
        jobs.push(job);
    }
    Ok(jobs)
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn duplicate_replication_is_rejected_and_machine_is_routed() {
        let job = json!({"kind":"benchmark","bundle_sha":"a".repeat(64),"runner_sha":"b".repeat(64),"experiment":"test","machine":"cpu-a","domain":"p","replicate":0,"backend":"wall","timeout_seconds":10,"admission_seconds":0});
        let obs = json!({"job":job,"threads":1,"peak_mem_bytes":1048576});
        let spec = json!({"schema":1,"observations":[obs]});
        let jobs = declare(&serde_json::to_vec(&spec).unwrap()).unwrap();
        assert!(jobs[0].requires.contains(&"benchmark-machine:cpu-a".into()));
        let spec = json!({"schema":1,"observations":[obs,obs]});
        assert!(declare(&serde_json::to_vec(&spec).unwrap()).is_err());
    }
}
