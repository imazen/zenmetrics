//! Fleet actuation (goal C "kill") — the dashboard's only side-effecting capability.
//!
//! A minimal Hetzner Cloud client (list + delete servers, scoped by label) so the KILL controls
//! actually tear paid boxes down — not just queue an intent. Inlined here rather than depending on
//! `zencloud-hetzner` so the Railway deploy workspace stays tiny (no fleet/salad/orchestrator stack);
//! the API surface mirrors `zencloud-hetzner::api`.
//!
//! **Safety: kill is scoped by a label *existence* selector** (default `group`). Fleet-launched boxes
//! carry `group=<launch-group>`; persistent dev boxes (e.g. `zen-arm-dev`) carry no `group` label and
//! are therefore never matched by a `KillFleet`. A `KillTier`/`KillRun` narrows to `group=<value>`.

use serde::Serialize;

use crate::ControlIntent;

const HETZNER_API_BASE: &str = "https://api.hetzner.cloud/v1";

/// A live fleet box (subset of the Hetzner server object the dashboard renders / kills).
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct FleetBox {
    pub id: i64,
    pub name: String,
    pub status: String,
    pub server_type: String,
    pub datacenter: String,
    pub ipv4: Option<String>,
    /// Value of the `group` label (the launch-group name), if present.
    pub group: Option<String>,
}

/// Result of a kill action: what was actually deleted, plus any per-server errors.
#[derive(Serialize, Debug, Default, PartialEq)]
pub struct KillResult {
    pub selector: String,
    pub killed: Vec<FleetBox>,
    pub errors: Vec<String>,
    /// Set when nothing was actuated (e.g. token missing) — the intent is still recorded.
    pub note: Option<String>,
}

/// Hetzner API token from env (`HETZNER_API_TOKEN`, falling back to `ZEN_HCLOUD_TOKEN`). `None` ⇒ the
/// dashboard can observe/record kill intents but cannot actuate — it says so in [`KillResult::note`].
pub fn fleet_token() -> Option<String> {
    std::env::var("HETZNER_API_TOKEN")
        .or_else(|_| std::env::var("ZEN_HCLOUD_TOKEN"))
        .ok()
        .filter(|t| !t.is_empty())
}

/// The label key that marks fleet-managed boxes (default `group`). Override with `ZEN_FLEET_LABEL`.
pub fn fleet_label_key() -> String {
    std::env::var("ZEN_FLEET_LABEL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "group".to_string())
}

/// Map a kill intent to a Hetzner `label_selector`. `KillFleet` ⇒ existence selector on the fleet
/// label (all managed boxes); `KillTier`/`KillRun` ⇒ `<label>=<value>` (one launch group).
pub fn selector_for(intent: &ControlIntent, label_key: &str) -> Option<String> {
    match intent {
        ControlIntent::KillFleet => Some(label_key.to_string()),
        ControlIntent::KillTier { tier } => Some(format!("{label_key}={tier}")),
        ControlIntent::KillRun { run } => Some(format!("{label_key}={run}")),
        _ => None,
    }
}

// --- Hetzner JSON (parse only the fields we render) ---

#[derive(serde::Deserialize)]
struct ServersResp {
    servers: Vec<RawServer>,
}
#[derive(serde::Deserialize)]
struct RawServer {
    id: i64,
    name: String,
    status: String,
    #[serde(default)]
    server_type: Named,
    #[serde(default)]
    datacenter: Named,
    #[serde(default)]
    public_net: PublicNet,
    #[serde(default)]
    labels: std::collections::HashMap<String, String>,
}
#[derive(serde::Deserialize, Default)]
struct Named {
    #[serde(default)]
    name: String,
}
#[derive(serde::Deserialize, Default)]
struct PublicNet {
    #[serde(default)]
    ipv4: Option<Ipv4>,
}
#[derive(serde::Deserialize)]
struct Ipv4 {
    #[serde(default)]
    ip: String,
}

impl RawServer {
    fn into_box(self, label_key: &str) -> FleetBox {
        FleetBox {
            id: self.id,
            name: self.name,
            status: self.status,
            server_type: self.server_type.name,
            datacenter: self.datacenter.name,
            ipv4: self.public_net.ipv4.map(|v| v.ip).filter(|s| !s.is_empty()),
            group: self.labels.get(label_key).cloned(),
        }
    }
}

/// `GET /servers?label_selector=<selector>` — live fleet boxes matching the selector.
pub async fn list_fleet(
    client: &reqwest::Client,
    token: &str,
    selector: &str,
    label_key: &str,
) -> Result<Vec<FleetBox>, String> {
    let url = format!(
        "{HETZNER_API_BASE}/servers?label_selector={}&per_page=50",
        urlencode(selector)
    );
    let resp = client
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("GET /servers {status}: {text}"));
    }
    let parsed: ServersResp = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    Ok(parsed
        .servers
        .into_iter()
        .map(|s| s.into_box(label_key))
        .collect())
}

/// `DELETE /servers/{id}`.
async fn delete_server(client: &reqwest::Client, token: &str, id: i64) -> Result<(), String> {
    let url = format!("{HETZNER_API_BASE}/servers/{id}");
    let resp = client
        .delete(&url)
        .bearer_auth(token)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status();
    if status.is_success() {
        Ok(())
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(format!("DELETE /servers/{id} {status}: {body}"))
    }
}

/// List boxes matching `selector`, then DELETE each. Best-effort: per-server errors are collected so
/// one failure doesn't abort the rest. Returns the boxes actually deleted.
pub async fn kill_fleet(
    client: &reqwest::Client,
    token: &str,
    selector: &str,
    label_key: &str,
) -> KillResult {
    let mut out = KillResult {
        selector: selector.to_string(),
        ..Default::default()
    };
    let boxes = match list_fleet(client, token, selector, label_key).await {
        Ok(b) => b,
        Err(e) => {
            out.errors.push(format!("list: {e}"));
            return out;
        }
    };
    if boxes.is_empty() {
        out.note = Some("no boxes matched the selector".to_string());
        return out;
    }
    for b in boxes {
        match delete_server(client, token, b.id).await {
            Ok(()) => out.killed.push(b),
            Err(e) => out.errors.push(e),
        }
    }
    out
}

/// Idle / orphan boxes (goal F: "idle boxes reaped"): running fleet boxes with **no matching worker
/// heartbeat** — a box that's billing but doing no work (its worker died or never started). These are
/// the reap targets. Matching is by name (a fleet box `arm-iter3-001` vs a `WorkerReport.worker` of
/// the same name). A caller reaps with [`kill_named`] on the returned names.
pub fn idle_boxes(
    boxes: &[FleetBox],
    worker_names: &std::collections::HashSet<String>,
) -> Vec<FleetBox> {
    boxes
        .iter()
        .filter(|b| b.status == "running" && !worker_names.contains(&b.name))
        .cloned()
        .collect()
}

/// Kill specific boxes by name (goal C/F: stop-spend tears down named paid workers). Only deletes
/// boxes that carry the fleet label AND whose name is in `names` — so an over-budget teardown can
/// never touch an unlabeled box (e.g. a persistent dev box) even if a worker name happened to match.
pub async fn kill_named(
    client: &reqwest::Client,
    token: &str,
    names: &[String],
    label_key: &str,
) -> KillResult {
    let mut out = KillResult {
        selector: format!("{label_key} ∩ names={names:?}"),
        ..Default::default()
    };
    if names.is_empty() {
        out.note = Some("no named workers to tear down".to_string());
        return out;
    }
    let wanted: std::collections::HashSet<&str> = names.iter().map(|s| s.as_str()).collect();
    let boxes = match list_fleet(client, token, label_key, label_key).await {
        Ok(b) => b,
        Err(e) => {
            out.errors.push(format!("list: {e}"));
            return out;
        }
    };
    let targets: Vec<FleetBox> = boxes
        .into_iter()
        .filter(|b| wanted.contains(b.name.as_str()))
        .collect();
    if targets.is_empty() {
        out.note = Some("no labeled fleet box matched the named paid workers".to_string());
        return out;
    }
    for b in targets {
        match delete_server(client, token, b.id).await {
            Ok(()) => out.killed.push(b),
            Err(e) => out.errors.push(e),
        }
    }
    out
}

/// Minimal URL-encoder for `label_selector` values: preserve `=` (selector separator) and `,`, encode
/// the rest conservatively. Mirrors `zencloud-hetzner::api::urlencode`.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'=' | b',' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_maps_intents() {
        assert_eq!(
            selector_for(&ControlIntent::KillFleet, "group").as_deref(),
            Some("group")
        );
        assert_eq!(
            selector_for(
                &ControlIntent::KillTier {
                    tier: "arm-iter3".into()
                },
                "group"
            )
            .as_deref(),
            Some("group=arm-iter3")
        );
        assert_eq!(
            selector_for(
                &ControlIntent::KillRun {
                    run: "sweep-42".into()
                },
                "group"
            )
            .as_deref(),
            Some("group=sweep-42")
        );
        assert!(selector_for(&ControlIntent::GcDryRun, "group").is_none());
    }

    #[test]
    fn existence_selector_excludes_unlabeled_boxes() {
        // KillFleet uses the bare label key (existence selector) — a box with no `group` label
        // (e.g. the persistent zen-arm-dev) is never matched server-side.
        let s = selector_for(&ControlIntent::KillFleet, "group").unwrap();
        assert_eq!(s, "group", "existence selector, not a value match");
    }

    #[test]
    fn urlencode_preserves_selector_syntax() {
        assert_eq!(urlencode("group=arm,role=worker"), "group=arm,role=worker");
        assert_eq!(urlencode("a b"), "a%20b");
    }

    #[test]
    fn idle_boxes_are_running_without_a_worker() {
        let b = |name: &str, status: &str| FleetBox {
            id: 1,
            name: name.into(),
            status: status.into(),
            server_type: "cax21".into(),
            datacenter: "fsn1".into(),
            ipv4: None,
            group: Some("g".into()),
        };
        let boxes = vec![
            b("w-001", "running"),
            b("w-002", "running"),
            b("w-003", "off"),
        ];
        let workers: std::collections::HashSet<String> =
            ["w-001".to_string()].into_iter().collect();
        let idle = idle_boxes(&boxes, &workers);
        // w-002 is running but has no worker heartbeat → idle; w-001 has one; w-003 isn't running.
        assert_eq!(idle.len(), 1);
        assert_eq!(idle[0].name, "w-002");
    }

    #[test]
    fn parse_servers_resp() {
        let json = r#"{"servers":[{"id":42,"name":"arm-iter3-001","status":"running",
            "server_type":{"name":"cax21"},"datacenter":{"name":"fsn1-dc14"},
            "public_net":{"ipv4":{"ip":"1.2.3.4"}},"labels":{"group":"arm-iter3"}}]}"#;
        let parsed: ServersResp = serde_json::from_str(json).unwrap();
        let b = parsed.servers.into_iter().next().unwrap().into_box("group");
        assert_eq!(b.id, 42);
        assert_eq!(b.server_type, "cax21");
        assert_eq!(b.ipv4.as_deref(), Some("1.2.3.4"));
        assert_eq!(b.group.as_deref(), Some("arm-iter3"));
    }
}
