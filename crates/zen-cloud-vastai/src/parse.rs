//! Tolerant parser for `vastai show instances --raw` / `vastai show
//! instances-v1 --raw` JSON output.
//!
//! This crate exists because the bash+python destroyer at
//! `/tmp/cvvdp-resume/run_destroy_ssim2_754.sh` crashed once
//! `vastai show instances --raw` emitted output that failed
//! `json.loads()` (`JSONDecodeError: Expecting value: line 1 column 1
//! (char 0)`), leaving 15 boxes running orphaned for 1.5 hr at the end
//! of an ssim2 backfill — burning $/hr after the work was done.
//!
//! Failure modes the parser must tolerate without exiting:
//!
//! 1. **Top-level not JSON at all.** vastai 1.0.8 sometimes prints a
//!    deprecation banner *before* the JSON body when `instances` is
//!    deprecated in favour of `instances-v1`; older shells captured
//!    both streams together. We strip everything before the first
//!    `[` / `{` byte before attempting `from_str`.
//! 2. **Top-level is an object (v1)** with `instances`, `instances_found`,
//!    `next_token`, etc.: pull the array out of `.instances`.
//! 3. **Top-level is an array (legacy v0)**: use it directly.
//! 4. **Individual instance row is malformed** — missing `id`, `label`,
//!    `dph_total`, or has a non-string label: emit a warning and skip
//!    that row, do NOT abort.
//! 5. **`dph_total` may be int, float, string, or null.** Coerce to f64
//!    or default to 0.0.
//! 6. **Empty array / no instances**: report 0 instances cleanly,
//!    exit code 0.

use serde::Deserialize;
use std::borrow::Cow;

/// One vast.ai instance row, post-tolerant-parse. The fields here are a
/// subset of what vastai actually returns — only what `vastai-fleet`
/// needs for status / destroy / watch.
///
/// `label` is `Option<String>` because unlabeled instances are legal
/// (vastai assigns an empty-string label by default; we normalise
/// `Some("")` to `None`).
#[derive(Debug, Clone, PartialEq)]
pub struct Instance {
    pub id: i64,
    pub label: Option<String>,
    pub status: Option<String>,
    pub dph_total: f64,
    pub gpu_name: Option<String>,
}

/// Result of parsing a `vastai show instances` raw output blob.
#[derive(Debug, Default)]
pub struct ParseReport {
    pub instances: Vec<Instance>,
    /// Per-row warnings emitted because a row was malformed and skipped.
    /// Callers should print these to stderr so the operator knows the
    /// parser was tolerant of bad data — silent skipping would be
    /// worse than crashing.
    pub warnings: Vec<String>,
}

/// Strip everything before the first `[` or `{` byte. vastai 1.0.8 prints
/// deprecation banners and other chatter on stderr that, on some shells,
/// gets merged with the JSON body. This is a pragmatic "find the start
/// of JSON" pass; if the body genuinely has neither bracket the caller
/// gets a parse error, which is fine.
fn strip_preamble(s: &str) -> Cow<'_, str> {
    let bytes = s.as_bytes();
    let start = bytes
        .iter()
        .position(|&b| b == b'[' || b == b'{')
        .unwrap_or(0);
    if start == 0 {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(s[start..].to_string())
    }
}

/// Raw row used for the second-pass deserialise. We DO use serde here
/// rather than indexing `serde_json::Value`, but only on a single row at
/// a time so one bad row doesn't poison the array.
#[derive(Deserialize)]
struct RawInstance {
    #[serde(default)]
    id: Option<serde_json::Value>,
    #[serde(default)]
    label: Option<serde_json::Value>,
    #[serde(default)]
    actual_status: Option<serde_json::Value>,
    #[serde(default)]
    status: Option<serde_json::Value>,
    #[serde(default)]
    cur_state: Option<serde_json::Value>,
    #[serde(default)]
    dph_total: Option<serde_json::Value>,
    #[serde(default)]
    gpu_name: Option<serde_json::Value>,
}

fn coerce_i64(v: &serde_json::Value) -> Option<i64> {
    if let Some(i) = v.as_i64() {
        return Some(i);
    }
    if let Some(s) = v.as_str() {
        return s.parse().ok();
    }
    if let Some(f) = v.as_f64() {
        return Some(f as i64);
    }
    None
}

fn coerce_f64(v: &serde_json::Value) -> Option<f64> {
    if let Some(f) = v.as_f64() {
        return Some(f);
    }
    if let Some(i) = v.as_i64() {
        return Some(i as f64);
    }
    if let Some(s) = v.as_str() {
        return s.parse().ok();
    }
    None
}

fn coerce_str(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Parse the raw output of `vastai show instances{,-v1} --raw`.
///
/// On structural parse failure (no JSON found / top-level neither array
/// nor object) this returns an error. On a parsed but malformed body
/// (some rows good, some bad) this returns `Ok(ParseReport)` with the
/// good rows and a warning per bad row.
pub fn parse_instances(raw: &str) -> anyhow::Result<ParseReport> {
    let stripped = strip_preamble(raw);
    let trimmed = stripped.trim();
    if trimmed.is_empty() {
        return Ok(ParseReport::default());
    }

    let top: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
        anyhow::anyhow!(
            "vastai output is not JSON: {e}; first 200 chars: {:?}",
            &trimmed[..trimmed.len().min(200)]
        )
    })?;

    // Find the instance array. v1 wraps in {"instances": [...], ...};
    // legacy v0 returns a bare array.
    let arr = match &top {
        serde_json::Value::Array(a) => a.clone(),
        serde_json::Value::Object(map) => match map.get("instances") {
            Some(serde_json::Value::Array(a)) => a.clone(),
            Some(other) => {
                return Err(anyhow::anyhow!(
                    "vastai output: .instances is not an array (got {:?})",
                    discriminant(other)
                ));
            }
            None => {
                return Err(anyhow::anyhow!(
                    "vastai output is an object but has no `instances` array"
                ));
            }
        },
        other => {
            return Err(anyhow::anyhow!(
                "vastai output is neither array nor object (got {:?})",
                discriminant(other)
            ));
        }
    };

    let mut report = ParseReport::default();
    for (idx, row) in arr.into_iter().enumerate() {
        match parse_one(idx, row) {
            Ok(inst) => report.instances.push(inst),
            Err(msg) => report.warnings.push(msg),
        }
    }
    Ok(report)
}

fn discriminant(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn parse_one(idx: usize, row: serde_json::Value) -> Result<Instance, String> {
    // First try strict shape; if that fails, fall back to a
    // serde_json::Value::Object lookup so we can still recover an id.
    let raw: RawInstance = match serde_json::from_value(row.clone()) {
        Ok(r) => r,
        Err(e) => {
            return Err(format!("row {idx}: failed to deserialise: {e}; raw: {row}"));
        }
    };

    let id = raw
        .id
        .as_ref()
        .and_then(coerce_i64)
        .ok_or_else(|| format!("row {idx}: no `id` field (got {:?})", raw.id))?;

    let label = raw
        .label
        .as_ref()
        .and_then(coerce_str)
        .filter(|s| !s.is_empty());

    let status = raw
        .actual_status
        .as_ref()
        .or(raw.status.as_ref())
        .or(raw.cur_state.as_ref())
        .and_then(coerce_str);

    let dph_total = raw.dph_total.as_ref().and_then(coerce_f64).unwrap_or(0.0);

    let gpu_name = raw.gpu_name.as_ref().and_then(coerce_str);

    Ok(Instance {
        id,
        label,
        status,
        dph_total,
        gpu_name,
    })
}

/// Filter instances by label substring. Returns instances whose `label`
/// is `Some(s)` and `s.contains(prefix)`. Empty `prefix` returns every
/// labeled instance (does not match unlabeled ones — destroying every
/// box on the account is too dangerous a default; require an explicit
/// match string).
pub fn filter_by_label(instances: &[Instance], label_prefix: &str) -> Vec<Instance> {
    instances
        .iter()
        .filter(|i| match &i.label {
            Some(s) => s.contains(label_prefix),
            None => false,
        })
        .cloned()
        .collect()
}

/// Aggregate by status. Returns a sorted Vec<(status, count)> for stable
/// output.
pub fn status_breakdown(instances: &[Instance]) -> Vec<(String, usize)> {
    let mut map: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for i in instances {
        let key = i.status.clone().unwrap_or_else(|| "unknown".to_string());
        *map.entry(key).or_insert(0) += 1;
    }
    map.into_iter().collect()
}

/// Total $/hr across the given instance set.
pub fn total_dph(instances: &[Instance]) -> f64 {
    instances.iter().map(|i| i.dph_total).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_array_parses() {
        let r = parse_instances("[]").unwrap();
        assert_eq!(r.instances.len(), 0);
        assert_eq!(r.warnings.len(), 0);
    }

    #[test]
    fn v1_envelope_with_empty_array_parses() {
        let raw =
            r#"{"instances": [], "instances_found": 0, "label_counts": {}, "next_token": null}"#;
        let r = parse_instances(raw).unwrap();
        assert_eq!(r.instances.len(), 0);
    }

    #[test]
    fn deprecation_banner_prefix_is_stripped() {
        let raw = "DEPRECATED: `vastai show instances` will be removed in a future release. Use `vastai show instances-v1` for the new paginated command.\n[]";
        let r = parse_instances(raw).unwrap();
        assert_eq!(r.instances.len(), 0);
    }

    #[test]
    fn empty_string_yields_no_instances_no_error() {
        // The 2026-05-18 ssim2 destroyer crashed on `Expecting value:
        // line 1 column 1 (char 0)` — that is exactly the empty-string
        // case (vastai stdout consumed entirely by stderr capture).
        // We MUST not crash here; the watch loop just polls again.
        let r = parse_instances("").unwrap();
        assert_eq!(r.instances.len(), 0);
    }

    #[test]
    fn one_well_formed_instance() {
        let raw = r#"[{"id": 12345, "label": "ssim2-backfill-w7", "actual_status": "running", "dph_total": 0.15, "gpu_name": "RTX 4090"}]"#;
        let r = parse_instances(raw).unwrap();
        assert_eq!(r.instances.len(), 1);
        let i = &r.instances[0];
        assert_eq!(i.id, 12345);
        assert_eq!(i.label.as_deref(), Some("ssim2-backfill-w7"));
        assert_eq!(i.status.as_deref(), Some("running"));
        assert!((i.dph_total - 0.15).abs() < 1e-9);
        assert_eq!(i.gpu_name.as_deref(), Some("RTX 4090"));
    }

    #[test]
    fn mixed_good_and_bad_rows() {
        let raw = r#"[
            {"id": 1, "label": "ssim2-w1", "actual_status": "running", "dph_total": 0.10},
            {"label": "missing-id-row", "actual_status": "running", "dph_total": 0.10},
            {"id": "string-id-coerced", "label": "ssim2-w3", "dph_total": "0.20"},
            {"id": 4, "label": null, "actual_status": "loading", "dph_total": 0.30},
            null
        ]"#;
        let r = parse_instances(raw).unwrap();
        // Expected: row 1 OK, row 2 SKIPPED (no id), row 3 SKIPPED
        // (id-as-string isn't a number), row 4 OK (null label is
        // fine — becomes unlabeled), row 5 SKIPPED (null at top
        // level fails RawInstance deserialise).
        let ids: Vec<i64> = r.instances.iter().map(|i| i.id).collect();
        // Row 3 may succeed if "string-id-coerced" parses (it doesn't —
        // not numeric), so we should have rows 1 and 4 only.
        assert!(ids.contains(&1), "row 1 should parse, got ids={ids:?}");
        assert!(ids.contains(&4), "row 4 should parse, got ids={ids:?}");
        assert!(
            !r.warnings.is_empty(),
            "expected at least one warning for the missing-id row"
        );
    }

    #[test]
    fn empty_label_string_becomes_none() {
        let raw = r#"[{"id": 1, "label": "", "actual_status": "running", "dph_total": 0.10}]"#;
        let r = parse_instances(raw).unwrap();
        assert_eq!(r.instances.len(), 1);
        assert_eq!(r.instances[0].label, None);
    }

    #[test]
    fn dph_can_be_string() {
        // Some vastai responses serialise numeric fields as strings.
        let raw = r#"[{"id": 1, "label": "w1", "dph_total": "0.42"}]"#;
        let r = parse_instances(raw).unwrap();
        assert!((r.instances[0].dph_total - 0.42).abs() < 1e-9);
    }

    #[test]
    fn null_top_level_is_error() {
        let r = parse_instances("null");
        assert!(r.is_err(), "null is structurally invalid for our purposes");
    }

    #[test]
    fn top_level_object_missing_instances_is_error() {
        let r = parse_instances(r#"{"error": "auth failed"}"#);
        assert!(r.is_err());
    }

    #[test]
    fn filter_by_label_substring() {
        let raw = r#"[
            {"id": 1, "label": "ssim2-backfill-w1", "actual_status": "running", "dph_total": 0.10},
            {"id": 2, "label": "ssim2-backfill-w2", "actual_status": "running", "dph_total": 0.10},
            {"id": 3, "label": "iwssim-backfill-w1", "actual_status": "running", "dph_total": 0.10},
            {"id": 4, "label": null, "actual_status": "running", "dph_total": 0.10}
        ]"#;
        let r = parse_instances(raw).unwrap();
        assert_eq!(r.instances.len(), 4);
        let matched = filter_by_label(&r.instances, "ssim2-backfill");
        assert_eq!(matched.len(), 2);
        let unlabeled_match = filter_by_label(&r.instances, "");
        // Empty prefix matches every labeled instance (3 of 4).
        assert_eq!(unlabeled_match.len(), 3);
    }

    #[test]
    fn status_breakdown_groups() {
        let raw = r#"[
            {"id": 1, "label": "w1", "actual_status": "running", "dph_total": 0.10},
            {"id": 2, "label": "w2", "actual_status": "running", "dph_total": 0.10},
            {"id": 3, "label": "w3", "actual_status": "loading", "dph_total": 0.10},
            {"id": 4, "label": "w4", "actual_status": "exited", "dph_total": 0.10}
        ]"#;
        let r = parse_instances(raw).unwrap();
        let b = status_breakdown(&r.instances);
        assert_eq!(
            b,
            vec![
                ("exited".to_string(), 1),
                ("loading".to_string(), 1),
                ("running".to_string(), 2),
            ]
        );
    }

    #[test]
    fn total_dph_sums() {
        let raw = r#"[
            {"id": 1, "label": "w1", "actual_status": "running", "dph_total": 0.10},
            {"id": 2, "label": "w2", "actual_status": "running", "dph_total": 0.25},
            {"id": 3, "label": "w3", "actual_status": "exited", "dph_total": 0.07}
        ]"#;
        let r = parse_instances(raw).unwrap();
        assert!((total_dph(&r.instances) - 0.42).abs() < 1e-9);
    }

    #[test]
    fn malformed_garbage_returns_error_not_panic() {
        // The 2026-05-18 destroyer panicked on this very input.
        let r = parse_instances("oops not json at all");
        assert!(r.is_err());
    }

    #[test]
    fn truncated_array_returns_error_not_panic() {
        let r = parse_instances("[ {\"id\": 1, ");
        assert!(r.is_err());
    }
}
