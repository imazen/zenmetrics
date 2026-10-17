//! Strict private adapter for content-addressed derived-data views.
use serde::Deserialize;
use zenfleet_core::{CellId, DesiredJob, JobKind, ResourceHint, Sha256Hex};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    schema: u32,
    views: Vec<View>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct View {
    view: String,
    cell: CellId,
    inputs: Vec<String>,
    hint: ResourceHint,
    requires: Vec<String>,
}

pub(super) fn declare(bytes: &[u8]) -> Result<Vec<DesiredJob>, String> {
    let spec: Spec = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if spec.schema != 1 || spec.views.is_empty() {
        return Err("expected schema 1 and nonempty views".into());
    }
    let mut ids = std::collections::HashSet::new();
    let mut jobs = Vec::with_capacity(spec.views.len());
    for v in spec.views {
        if v.view.is_empty()
            || v.inputs.is_empty()
            || v.requires.is_empty()
            || v.hint.peak_mem_bytes == 0
            || v.hint.threads == 0
        {
            return Err(
                "view, hashed inputs, capability and nonzero resource hints are required".into(),
            );
        }
        let inputs = v
            .inputs
            .into_iter()
            .map(|s| Sha256Hex::parse(s).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let job = DesiredJob::new(JobKind::Bake { view: v.view }, inputs, v.cell)
            .with_hint(v.hint)
            .with_requires(v.requires);
        if !ids.insert(job.job_id()) {
            return Err(
                "duplicate derived view identity; cell metadata is not part of JobId".into(),
            );
        }
        jobs.push(job);
    }
    Ok(jobs)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn item(view: &str) -> serde_json::Value {
        serde_json::json!({"view":view,"cell":{"image_path":"ref.png","codec":"zenjpeg","q":-1,"knob_tuple_json":"{}"},
            "inputs":["a".repeat(64)],"hint":{"peak_mem_bytes":1024,"threads":1,"vram_bytes":null},
            "requires":["derived-data-v1"]})
    }
    #[test]
    fn derived_inputs_bind_identity_and_reject_cell_only_aliases() {
        let a = item("one");
        let mut b = a.clone();
        b["cell"]["q"] = 2.into();
        assert!(
            declare(
                &serde_json::to_vec(&serde_json::json!({"schema":1,"views":[a.clone(),b]}))
                    .unwrap()
            )
            .is_err()
        );
        let jobs = declare(
            &serde_json::to_vec(&serde_json::json!({"schema":1,"views":[a,item("two")]})).unwrap(),
        )
        .unwrap();
        assert_ne!(jobs[0].job_id(), jobs[1].job_id());
        assert_eq!(jobs[0].hint.unwrap().threads, 1);
    }
    #[test]
    fn derived_refuses_unhashed_inputs_and_zero_resource_hints() {
        for field in ["input", "memory", "threads"] {
            let mut a = item("one");
            match field {
                "input" => a["inputs"][0] = "relative/file.png".into(),
                "memory" => a["hint"]["peak_mem_bytes"] = 0.into(),
                _ => a["hint"]["threads"] = 0.into(),
            }
            assert!(
                declare(&serde_json::to_vec(&serde_json::json!({"schema":1,"views":[a]})).unwrap())
                    .is_err()
            );
        }
    }
}
