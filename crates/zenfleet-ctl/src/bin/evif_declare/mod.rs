//! One prepared view and an explicit, hash-bound union of EVIF parameter vectors.
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use zenfleet_core::{CellId, DesiredJob, JobKind, ResourceHint, Sha256Hex, sha256};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Spec {
    pub sources: Vec<Source>,
    pub families: Vec<Family>,
    pub implementation_shas: Vec<String>,
    pub peak_mem_bytes: u64,
    pub diagnostics: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub image_path: String,
    pub source_sha256: String,
    pub pixels_sha256: String,
    pub identity: String,
    pub width: u32,
    pub height: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Family {
    pub label: String,
    pub codec: String,
    pub params: Vec<String>,
}
#[derive(Serialize)]
struct GridRow<'a> {
    family: &'a str,
    label: &'a str,
    rung: usize,
    params_json: &'a str,
}
#[derive(Serialize)]
struct Knobs<'a> {
    grid_path: &'a str,
    grid_sha256: &'a str,
    reference_pixels_sha256: &'a str,
    width: u32,
    height: u32,
    diagnostics: &'a str,
}
pub(super) fn grid(spec: &Spec) -> Result<(Vec<u8>, i64), String> {
    let mut bytes = Vec::new();
    let mut seen = HashSet::new();
    let mut quality = None;
    for family in &spec.families {
        let backend = family
            .codec
            .strip_prefix("evif:")
            .ok_or("EVIF codec prefix required")?;
        if !matches!(backend, "zenjpeg-expert" | "zenjpeg-color")
            || family.label.is_empty()
            || family.params.is_empty()
        {
            return Err("unsupported/empty EVIF family".into());
        }
        for (rung, param) in family.params.iter().enumerate() {
            let value: serde_json::Value =
                serde_json::from_str(param).map_err(|e| e.to_string())?;
            let q = value["quality"].as_f64().ok_or("EVIF quality missing")?;
            if !q.is_finite() || q.fract() != 0. || !(0.0..=100.0).contains(&q) {
                return Err("integer quality in0..100 required".into());
            }
            if quality.is_some_and(|prior| prior != q as i64) {
                return Err("one luma quality per union required".into());
            }
            quality = Some(q as i64);
            if !seen.insert((&family.label, param)) {
                return Err("duplicate EVIF rung".into());
            }
            bytes.extend(
                serde_json::to_vec(&GridRow {
                    family: backend,
                    label: &family.label,
                    rung,
                    params_json: param,
                })
                .map_err(|e| e.to_string())?,
            );
            bytes.push(b'\n');
        }
    }
    if seen.len() > 128 {
        return Err("union exceeds128 bounded rungs".into());
    }
    Ok((bytes, quality.ok_or("empty EVIF grid")?))
}
pub(super) fn declare(
    spec: Spec,
    grid_path: &str,
    grid_bytes: &[u8],
    q: i64,
) -> Result<Vec<DesiredJob>, String> {
    if spec.sources.is_empty()
        || spec.implementation_shas.len() < 3
        || spec.peak_mem_bytes == 0
        || !std::path::Path::new(grid_path).is_absolute()
        || !matches!(spec.diagnostics.as_str(), "none" | "all")
    {
        return Err(
            "sources, absolute grid, pins and resource/diagnostic admission required".into(),
        );
    }
    let pins: Vec<_> = spec
        .implementation_shas
        .into_iter()
        .map(|s| Sha256Hex::parse(s).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let grid_sha = sha256(grid_bytes);
    let mut jobs = Vec::new();
    let mut seen = HashSet::new();
    for source in spec.sources {
        if source.width == 0
            || source.height == 0
            || u64::from(source.width) * u64::from(source.height) >= 1_000_000
            || source.identity.is_empty()
        {
            return Err("invalid source geometry/identity".into());
        }
        Sha256Hex::parse(source.pixels_sha256.clone()).map_err(|e| e.to_string())?;
        let source_sha = Sha256Hex::parse(source.source_sha256).map_err(|e| e.to_string())?;
        let knobs = serde_json::to_string(&Knobs {
            grid_path,
            grid_sha256: grid_sha.as_str(),
            reference_pixels_sha256: &source.pixels_sha256,
            width: source.width,
            height: source.height,
            diagnostics: &spec.diagnostics,
        })
        .map_err(|e| e.to_string())?;
        let mut inputs = vec![
            source_sha,
            sha256(source.identity.as_bytes()),
            grid_sha.clone(),
            sha256(knobs.as_bytes()),
        ];
        inputs.extend(pins.iter().cloned());
        let job = DesiredJob::new(
            JobKind::Metric {
                metric: "evif-chroma-batch-v1".into(),
            },
            inputs,
            CellId {
                image_path: source.image_path,
                codec: "evif:batch".into(),
                q,
                knob_tuple_json: knobs,
            },
        )
        .with_requires([
            "evif-view-batch-v1".to_owned(),
            "ssim2-safesyn-b7c2b4b3".to_owned(),
        ])
        .with_hint(ResourceHint {
            peak_mem_bytes: spec.peak_mem_bytes,
            threads: 1,
            vram_bytes: Some(0),
        });
        if !seen.insert(job.job_id()) {
            return Err("duplicate source job identity".into());
        }
        jobs.push(job);
    }
    Ok(jobs)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> Spec {
        serde_json::from_value(serde_json::json!({"sources":[{"image_path":"x.png","source_sha256":"11".repeat(32),"pixels_sha256":"55".repeat(32),"identity":"TRAIN/x","width":32,"height":32}],"families":[{"label":"ycc444","codec":"evif:zenjpeg-expert","params":["{\"quality\":95,\"xyb\":false}","{\"quality\":95,\"xyb\":true}"]}],"implementation_shas":["22".repeat(32),"33".repeat(32),"44".repeat(32)],"peak_mem_bytes":268435456,"diagnostics":"none"})).unwrap()
    }
    #[test]
    fn all_rungs_share_one_view_job_and_source_identity_never_collapses() {
        let mut s = spec();
        s.sources.push(Source {
            image_path: "alias.png".into(),
            source_sha256: "11".repeat(32),
            pixels_sha256: "55".repeat(32),
            identity: "TRAIN/alias".into(),
            width: 32,
            height: 32,
        });
        let (g, q) = grid(&s).unwrap();
        assert_eq!(
            g.split(|b| *b == b'\n').filter(|l| !l.is_empty()).count(),
            2
        );
        let jobs = declare(
            s,
            &std::env::temp_dir().join("grid.jsonl").to_string_lossy(),
            &g,
            q,
        )
        .unwrap();
        assert_eq!(jobs.len(), 2);
        assert_ne!(jobs[0].job_id(), jobs[1].job_id());
        assert!(
            jobs.iter()
                .all(|j| j.requires.contains(&"evif-view-batch-v1".into()))
        );
    }
    #[test]
    fn duplicates_and_megapixel_boundary_refused() {
        let mut s = spec();
        let duplicate = s.families[0].params[0].clone();
        s.families[0].params.push(duplicate);
        assert!(grid(&s).is_err());
        let mut s = spec();
        s.sources[0].width = 1000;
        s.sources[0].height = 1000;
        let (g, q) = grid(&s).unwrap();
        assert!(
            declare(
                s,
                &std::env::temp_dir().join("grid.jsonl").to_string_lossy(),
                &g,
                q
            )
            .is_err()
        );
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticSpec {
    sources: Vec<DiagnosticSource>,
    implementation_shas: Vec<String>,
    peak_mem_bytes: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticSource {
    source: Source,
    evif_shas: Vec<String>,
}

pub(super) fn diagnostics(bytes: &[u8]) -> Result<Vec<DesiredJob>, String> {
    let spec: DiagnosticSpec = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if spec.sources.is_empty() || spec.implementation_shas.len() < 3 || spec.peak_mem_bytes == 0 {
        return Err("diagnostic sources, pins and memory hint required".into());
    }
    let mut jobs = Vec::new();
    let mut seen = HashSet::new();
    for entry in spec.sources {
        let source = entry.source;
        if source.width == 0
            || source.height == 0
            || u64::from(source.width) * u64::from(source.height) >= 1_000_000
            || entry.evif_shas.is_empty()
            || entry.evif_shas.len() > 128
        {
            return Err("invalid diagnostic geometry or selection".into());
        }
        let mut unique = HashSet::new();
        for sha in &entry.evif_shas {
            Sha256Hex::parse(sha.clone()).map_err(|e| e.to_string())?;
            if !unique.insert(sha) {
                return Err("duplicate diagnostic container".into());
            }
        }
        let knobs=serde_json::json!({"source_sha256":source.source_sha256,"reference_pixels_sha256":source.pixels_sha256,"width":source.width,"height":source.height,"evif_shas":entry.evif_shas}).to_string();
        let mut inputs = vec![
            Sha256Hex::parse(source.source_sha256).map_err(|e| e.to_string())?,
            sha256(source.identity.as_bytes()),
            sha256(knobs.as_bytes()),
        ];
        for sha in entry.evif_shas.iter().chain(&spec.implementation_shas) {
            inputs.push(Sha256Hex::parse(sha.clone()).map_err(|e| e.to_string())?);
        }
        let job = DesiredJob::new(
            JobKind::ScoreFile {
                metrics: vec!["evif-diagnostics-v1".into()],
                hdr: false,
                hdr_transfer: None,
            },
            inputs,
            CellId {
                image_path: source.image_path,
                codec: "evif:diagnostics".into(),
                q: 95,
                knob_tuple_json: knobs,
            },
        )
        .with_requires(["evif-diagnostics-v1".to_owned()])
        .with_hint(ResourceHint {
            peak_mem_bytes: spec.peak_mem_bytes,
            threads: 1,
            vram_bytes: Some(0),
        });
        if !seen.insert(job.job_id()) {
            return Err("duplicate diagnostic view identity".into());
        }
        jobs.push(job);
    }
    Ok(jobs)
}

#[cfg(test)]
mod diagnostic_tests {
    #[test]
    fn selected_containers_and_pins_are_bound_without_encoding() {
        let source = serde_json::json!({"image_path":"ref.png","source_sha256":"11".repeat(32),"pixels_sha256":"22".repeat(32),"identity":"TRAIN/ref","width":32,"height":32});
        let mut spec = serde_json::json!({"sources":[{"source":source,"evif_shas":["33".repeat(32)]}],"implementation_shas":["44".repeat(32),"55".repeat(32),"66".repeat(32)],"peak_mem_bytes":268435456});
        let jobs = super::diagnostics(&serde_json::to_vec(&spec).unwrap()).unwrap();
        assert_eq!(jobs.len(), 1);
        assert!(matches!(
            jobs[0].kind,
            zenfleet_core::JobKind::ScoreFile { .. }
        ));
        assert!(jobs[0].inputs.iter().any(|s| s.as_str() == "33".repeat(32)));
        spec["sources"][0]["evif_shas"][0] = "not-a-content-sha".into();
        assert!(super::diagnostics(&serde_json::to_vec(&spec).unwrap()).is_err());
    }
}
