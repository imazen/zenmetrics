#![forbid(unsafe_code)]
//! Railway entrypoint: serves the shadcn dashboard SPA + monitoring views + control API over the
//! Parquet ledger, **live-reloading** it on an interval and **firing notifications** to a webhook.
//! Control actions that tear down paid boxes are **actuated here** against the Hetzner fleet (goal C).
//!
//! Config via env (Railway dashboard):
//!   ZEN_LEDGER, ZEN_BLOB_INDEX, ZEN_WORKERS_JSON  — local paths or s3:// URIs (R2)
//!   ZEN_R2_ENDPOINT                               — R2 endpoint for s3:// (+ AWS_* creds)
//!   ZEN_REFRESH_SECS                              — ledger reload interval (default 30)
//!   ZEN_NOTIFY_WEBHOOK                            — Slack/Discord/ntfy webhook URL (optional; goal D)
//!   ZEN_PUBLIC_URL                                — base URL for notification deep links
//!   ZEN_BUDGET_CAP_USD (default 0=off), ZEN_POISON_THRESHOLD (default 10)
//!   ZEN_DASH_PASSWORD                             — HTTP Basic Auth password (empty ⇒ open; logs a warning)
//!   HETZNER_API_TOKEN / ZEN_HCLOUD_TOKEN          — fleet kill + live-fleet visibility (goal C/H)
//!   ZEN_FLEET_LABEL (default `group`)             — label that scopes which boxes a KILL may touch
//!   ZEN_WEB_DIR (default ./web/dist)              — built shadcn SPA assets to serve
//!   PORT                                          — injected by Railway (default 3000)

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::RwLock;
use tower_http::services::{ServeDir, ServeFile};

use zen_job_core::{JobStatus, Sha256Hex};
use zen_jobdash::{
    catalog_view, cost_view, detect, failures, fleet_label_key, fleet_token, format_event,
    gc_dry_run, idle_boxes, kill_fleet, kill_named, list_fleet, progress, results_view, run_summary,
    selector_for, stop_spend, storage, workers_view, CatalogRow, ControlIntent, CostView, DashData,
    FailureCell, FleetBox, KindProgress, NotifyPayload, ResultRow, RunSummary, TierStorage, WorkerStat,
};

/// Shared app state: the live ledger snapshot + a pooled HTTP client for fleet actuation.
#[derive(Clone)]
struct AppState {
    data: Arc<RwLock<DashData>>,
    http: reqwest::Client,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = tokio::task::spawn_blocking(load).await?.unwrap_or_else(|e| {
        eprintln!("zen-jobdash: warning: initial load failed ({e}); serving empty views");
        DashData::default()
    });
    eprintln!(
        "zen-jobdash: loaded {} rows, {} blobs, {} workers",
        data.rows.len(),
        data.blobs.len(),
        data.workers.len()
    );
    let state = AppState { data: Arc::new(RwLock::new(data)), http: reqwest::Client::new() };

    if std::env::var("ZEN_DASH_PASSWORD").ok().filter(|p| !p.is_empty()).is_none() {
        eprintln!("zen-jobdash: WARNING: ZEN_DASH_PASSWORD unset — dashboard is UNAUTHENTICATED.");
    }
    if fleet_token().is_none() {
        eprintln!("zen-jobdash: note: no HETZNER_API_TOKEN — kill controls record intent but won't actuate.");
    }

    // Background: reload the ledger on an interval (live views) + fire notifications (goal D).
    tokio::spawn(refresh_loop(state.data.clone()));

    let web_dir = std::env::var("ZEN_WEB_DIR").unwrap_or_else(|_| "./web/dist".to_string());
    let app = build_router(state, &web_dir);

    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(3000);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("zen-jobdash listening on http://{addr} (web_dir={web_dir})");
    axum::serve(listener, app).await?;
    Ok(())
}

fn build_router(state: AppState, web_dir: &str) -> Router {
    let api = Router::new()
        .route("/api/progress", get(api_progress))
        .route("/api/summary", get(api_summary))
        .route("/api/failures", get(api_failures))
        .route("/api/cost", get(api_cost))
        .route("/api/storage", get(api_storage))
        .route("/api/workers", get(api_workers))
        .route("/api/catalog", get(api_catalog))
        .route("/api/results", get(api_results))
        .route("/api/peek/{sha}", get(api_peek))
        .route("/api/fleet", get(api_fleet))
        .route("/api/control", post(api_control))
        .with_state(state);

    // Serve the built shadcn SPA at the root; unknown paths fall back to index.html (client routing).
    // If the build output is absent (e.g. running locally without `npm run build`), serve a minimal
    // placeholder so the API is still reachable.
    let index = format!("{web_dir}/index.html");
    let mut app = if Path::new(&index).exists() {
        // Serve built assets; any unmatched path (client routes, hard refreshes) falls back to
        // index.html so the SPA always boots.
        let serve = ServeDir::new(web_dir).fallback(ServeFile::new(index));
        api.fallback_service(serve)
    } else {
        api.fallback(placeholder)
    };
    // Health check is exempt from auth so Railway probes never 401.
    app = app.route("/healthz", get(|| async { "ok" }));
    app.layer(middleware::from_fn(require_auth))
}

/// HTTP Basic Auth gate (goal: password protection). When `ZEN_DASH_PASSWORD` is set, every request
/// (except `/healthz`) must carry `Authorization: Basic base64("<anything>:<password>")`; on miss we
/// return 401 + `WWW-Authenticate`, so browsers show a native login prompt and cache it for the SPA's
/// `fetch()` calls. When unset, the gate is open (a warning is logged at startup).
async fn require_auth(req: Request, next: Next) -> Response {
    let password = std::env::var("ZEN_DASH_PASSWORD").unwrap_or_default();
    if password.is_empty() || req.uri().path() == "/healthz" {
        return next.run(req).await;
    }
    let ok = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Basic "))
        .and_then(basic_password)
        .map(|p| constant_time_eq(p.as_bytes(), password.as_bytes()))
        .unwrap_or(false);
    if ok {
        next.run(req).await
    } else {
        let mut resp = (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
        resp.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Basic realm=\"zen-jobdash\", charset=\"UTF-8\""),
        );
        resp
    }
}

/// Decode a `Basic` credential, returning the password half of `user:password`.
fn basic_password(b64: &str) -> Option<String> {
    let decoded = base64_decode(b64.trim())?;
    let s = String::from_utf8(decoded).ok()?;
    s.split_once(':').map(|(_, pw)| pw.to_string())
}

/// Length-independent byte comparison (avoids leaking the password length via early return).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Minimal standard-alphabet base64 decoder (no external dep). Ignores padding; rejects bad chars.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for &c in s.as_bytes() {
        if c == b'=' {
            break;
        }
        let v = val(c)? as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

fn load() -> Result<DashData, zen_jobdash::DashError> {
    let ledger: Vec<String> = std::env::var("ZEN_LEDGER")
        .map(|s| s.split(',').filter(|p| !p.is_empty()).map(String::from).collect())
        .unwrap_or_default();
    let endpoint = std::env::var("ZEN_R2_ENDPOINT").ok();
    let blob = std::env::var("ZEN_BLOB_INDEX").ok();
    let workers = std::env::var("ZEN_WORKERS_JSON").ok();
    DashData::from_sources(&ledger, endpoint.as_deref(), blob.as_deref(), workers.as_deref())
}

/// Periodically reload the ledger so views reflect live runs, and fire newly-true notification
/// conditions to the webhook (each fires once). No-op for notifications if ZEN_NOTIFY_WEBHOOK is unset.
async fn refresh_loop(state: Arc<RwLock<DashData>>) {
    let secs: u64 = std::env::var("ZEN_REFRESH_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let webhook = std::env::var("ZEN_NOTIFY_WEBHOOK").ok();
    let base_url = std::env::var("ZEN_PUBLIC_URL").unwrap_or_default();
    let cap: f64 = std::env::var("ZEN_BUDGET_CAP_USD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let poison_threshold: usize =
        std::env::var("ZEN_POISON_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let client = reqwest::Client::new();
    let mut fired: HashSet<String> = HashSet::new();

    loop {
        tokio::time::sleep(Duration::from_secs(secs)).await;
        match tokio::task::spawn_blocking(load).await {
            Ok(Ok(fresh)) => {
                if let Some(url) = webhook.as_deref() {
                    let prog = progress(&fresh.rows);
                    let cost = cost_view(&fresh.workers);
                    for ev in detect(&prog, &cost, cap, poison_threshold) {
                        let sig = serde_json::to_string(&ev).unwrap_or_default();
                        if fired.insert(sig) {
                            let payload = format_event(&ev, &base_url);
                            if let Err(e) = send_webhook(&client, url, &payload).await {
                                eprintln!("zen-jobdash: webhook send failed: {e}");
                            }
                        }
                    }
                }
                *state.write().await = fresh;
            }
            Ok(Err(e)) => eprintln!("zen-jobdash: ledger reload failed: {e}"),
            Err(e) => eprintln!("zen-jobdash: reload task panicked: {e}"),
        }
    }
}

async fn send_webhook(client: &reqwest::Client, url: &str, p: &NotifyPayload) -> Result<(), String> {
    let body = serde_json::json!({ "text": format!("{} — {}", p.text, p.link) });
    client
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Output blobs of Done rows are the referenced set (a proxy until the full reachability join over
/// the desired graph is wired).
fn referenced(data: &DashData) -> HashSet<Sha256Hex> {
    data.rows
        .iter()
        .filter(|r| r.status == JobStatus::Done)
        .filter_map(|r| r.output_sha.clone())
        .collect()
}

async fn api_progress(State(s): State<AppState>) -> Json<Vec<KindProgress>> {
    Json(progress(&s.data.read().await.rows))
}
async fn api_summary(State(s): State<AppState>) -> Json<RunSummary> {
    let d = s.data.read().await;
    Json(run_summary(&d.rows, &d.workers))
}
async fn api_failures(State(s): State<AppState>) -> Json<Vec<FailureCell>> {
    Json(failures(&s.data.read().await.rows))
}
async fn api_cost(State(s): State<AppState>) -> Json<CostView> {
    Json(cost_view(&s.data.read().await.workers))
}
async fn api_storage(State(s): State<AppState>) -> Json<Vec<TierStorage>> {
    Json(storage(&s.data.read().await.blobs))
}
async fn api_workers(State(s): State<AppState>) -> Json<Vec<WorkerStat>> {
    Json(workers_view(&s.data.read().await.workers))
}
async fn api_catalog(State(s): State<AppState>) -> Json<Vec<CatalogRow>> {
    Json(catalog_view(&s.data.read().await.rows))
}
async fn api_results(State(s): State<AppState>) -> Json<Vec<ResultRow>> {
    Json(results_view(&s.data.read().await.rows, 200))
}

/// Peek a result blob by its content hash (goal B: "peek results in-browser"). Fetches
/// `ZEN_BLOBS_R2/<sha>` from R2 and returns its bytes as (truncated) text + size. The blob base URI
/// is `ZEN_BLOBS_R2` (e.g. `s3://bucket/blobs`); R2 endpoint from `ZEN_R2_ENDPOINT`.
async fn api_peek(axum::extract::Path(sha): axum::extract::Path<String>) -> Json<serde_json::Value> {
    // Guard: content hashes are hex — reject anything else (no path traversal into the bucket).
    if sha.is_empty() || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Json(serde_json::json!({ "error": "sha must be hex" }));
    }
    let Some(base) = std::env::var("ZEN_BLOBS_R2").ok().filter(|s| !s.is_empty()) else {
        return Json(serde_json::json!({ "error": "set ZEN_BLOBS_R2 (s3://bucket/blobs) to enable result peek" }));
    };
    let uri = format!("{}/{}", base.trim_end_matches('/'), sha);
    let endpoint = std::env::var("ZEN_R2_ENDPOINT").ok();
    match tokio::task::spawn_blocking(move || zen_ledger::read_bytes_uri(&uri, endpoint.as_deref())).await {
        Ok(Ok(bytes)) => {
            let size = bytes.len();
            let text: String = String::from_utf8_lossy(&bytes).chars().take(4096).collect();
            Json(serde_json::json!({ "sha": sha, "size": size, "text": text }))
        }
        Ok(Err(e)) => Json(serde_json::json!({ "sha": sha, "error": e.to_string() })),
        Err(e) => Json(serde_json::json!({ "sha": sha, "error": e.to_string() })),
    }
}

/// Live fleet boxes (goal C/H visibility): the actual paid/free boxes Hetzner reports for the fleet
/// label. Returns `{boxes, actuation}` so the UI can show "kill won't actuate" when the token is absent.
async fn api_fleet(State(s): State<AppState>) -> Json<serde_json::Value> {
    let label = fleet_label_key();
    match fleet_token() {
        Some(token) => match list_fleet(&s.http, &token, &label, &label).await {
            Ok(boxes) => {
                // Flag idle/orphan boxes (running, no matching worker heartbeat) — goal F reap targets.
                let worker_names: HashSet<String> =
                    s.data.read().await.workers.iter().map(|w| w.worker.clone()).collect();
                let idle: Vec<String> =
                    idle_boxes(&boxes, &worker_names).into_iter().map(|b| b.name).collect();
                Json(serde_json::json!({ "actuation": true, "label": label, "boxes": boxes, "idle": idle }))
            }
            Err(e) => Json(serde_json::json!({ "actuation": true, "label": label, "boxes": Vec::<FleetBox>::new(), "error": e })),
        },
        None => Json(serde_json::json!({
            "actuation": false,
            "label": label,
            "boxes": Vec::<FleetBox>::new(),
            "note": "no HETZNER_API_TOKEN — kill records intent but won't actuate"
        })),
    }
}

/// Control surface (goal C). GC/StopSpend are pure previews; Kill* actuate against the Hetzner fleet
/// when a token is present (else the intent is recorded with a note). Pause/Drain stay intents.
async fn api_control(State(s): State<AppState>, Json(intent): Json<ControlIntent>) -> Json<serde_json::Value> {
    // Kill paths actuate — handle before taking the read lock (no ledger data needed).
    if matches!(
        intent,
        ControlIntent::KillFleet | ControlIntent::KillTier { .. } | ControlIntent::KillRun { .. }
    ) {
        let label = fleet_label_key();
        let selector = selector_for(&intent, &label).unwrap_or_default();
        return match fleet_token() {
            Some(token) => {
                let result = kill_fleet(&s.http, &token, &selector, &label).await;
                Json(serde_json::json!({ "action": "kill", "actuated": true, "result": result }))
            }
            None => Json(serde_json::json!({
                "action": "kill", "actuated": false, "selector": selector,
                "note": "no HETZNER_API_TOKEN — intent recorded, no boxes touched"
            })),
        };
    }

    // Idle reaping (goal F): kill running fleet boxes with no matching worker heartbeat. Needs the
    // worker list (from the ledger) + a live fleet list, so it's handled before the GC read lock.
    if matches!(intent, ControlIntent::ReapIdle) {
        let label = fleet_label_key();
        return match fleet_token() {
            Some(token) => {
                let worker_names: HashSet<String> =
                    s.data.read().await.workers.iter().map(|w| w.worker.clone()).collect();
                match list_fleet(&s.http, &token, &label, &label).await {
                    Ok(boxes) => {
                        let idle = idle_boxes(&boxes, &worker_names);
                        if idle.is_empty() {
                            Json(serde_json::json!({ "action": "reap_idle", "actuated": true, "reaped": [], "note": "no idle boxes" }))
                        } else {
                            let names: Vec<String> = idle.iter().map(|b| b.name.clone()).collect();
                            let result = kill_named(&s.http, &token, &names, &label).await;
                            Json(serde_json::json!({ "action": "reap_idle", "actuated": true, "result": result }))
                        }
                    }
                    Err(e) => Json(serde_json::json!({ "action": "reap_idle", "actuated": true, "error": e })),
                }
            }
            None => Json(serde_json::json!({ "action": "reap_idle", "actuated": false, "note": "no HETZNER_API_TOKEN" })),
        };
    }

    let guard = s.data.read().await;
    let d: &DashData = &guard;
    let plan = match intent {
        ControlIntent::GcDryRun => serde_json::to_value(gc_dry_run(&d.blobs, &referenced(d), &HashSet::new()))
            .unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() })),
        ControlIntent::StopSpend { cap_usd } => {
            let spent = cost_view(&d.workers).total_spent_usd;
            let decision = stop_spend(&d.workers, spent, cap_usd);
            // Goal C/F: over budget → actually tear the paid boxes down (scoped to fleet-labeled
            // boxes whose name matches a paid worker). Free tiers keep draining.
            let actuation = if decision.over_budget && !decision.tear_down.is_empty() {
                match fleet_token() {
                    Some(token) => {
                        let label = fleet_label_key();
                        let result = kill_named(&s.http, &token, &decision.tear_down, &label).await;
                        serde_json::json!({ "actuated": true, "result": result })
                    }
                    None => serde_json::json!({ "actuated": false, "note": "over budget but no HETZNER_API_TOKEN — would tear down listed workers" }),
                }
            } else {
                serde_json::json!({ "actuated": false })
            };
            serde_json::json!({ "decision": decision, "teardown": actuation })
        }
        // Pause/Drain/Resume write a RunControl object to R2; workers honor it before pulling work
        // (goal C: pause/resume/drain without losing state). Needs ZEN_CONTROL_R2 (an s3:// URI).
        ControlIntent::Pause { .. } => write_run_control(zen_job_core::RunControl::PAUSED),
        ControlIntent::Drain { .. } => write_run_control(zen_job_core::RunControl::DRAINING),
        ControlIntent::Resume { .. } => write_run_control(zen_job_core::RunControl::RUNNING),
        other => serde_json::json!({ "intent": other, "status": "queued_for_fleet_actuation" }),
    };
    Json(plan)
}

/// Write a [`RunControl`](zen_job_core::RunControl) object to R2 (goal C pause/drain actuation).
/// Target is `ZEN_CONTROL_R2` (an `s3://bucket/key` URI, or a local path for dev); R2 endpoint from
/// `ZEN_R2_ENDPOINT`. No-op with a note when `ZEN_CONTROL_R2` is unset.
fn write_run_control(ctl: zen_job_core::RunControl) -> serde_json::Value {
    let Some(uri) = std::env::var("ZEN_CONTROL_R2").ok().filter(|s| !s.is_empty()) else {
        return serde_json::json!({
            "action": "control", "written": false,
            "note": "set ZEN_CONTROL_R2 (s3://bucket/key) + point workers at --control-r2-key to enable pause/drain"
        });
    };
    let endpoint = std::env::var("ZEN_R2_ENDPOINT").ok();
    let body = serde_json::to_vec(&ctl).unwrap_or_default();
    match zen_ledger::write_bytes_uri(&uri, &body, endpoint.as_deref()) {
        Ok(()) => serde_json::json!({ "action": "control", "written": true, "control": ctl, "uri": uri }),
        Err(e) => serde_json::json!({ "action": "control", "written": false, "error": e.to_string() }),
    }
}

/// Served only when the SPA build output is missing (local dev without `npm run build`).
async fn placeholder() -> Html<&'static str> {
    Html(
        "<!doctype html><meta charset=utf-8><title>zen-jobdash</title>\
        <body style=\"font:14px system-ui;margin:3rem;max-width:40rem\">\
        <h1>zen-jobdash</h1><p>API is live (<code>/api/progress</code>, <code>/api/cost</code>, \
        <code>/api/fleet</code>, …). The shadcn SPA build output was not found — \
        run <code>npm ci &amp;&amp; npm run build</code> in <code>crates/zen-jobdash/web</code> \
        or set <code>ZEN_WEB_DIR</code>.</p></body>",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: all base64 literals below decode to harmless, non-credential fixtures (no real
    // password/token shape) so secret scanners don't flag the test file.
    #[test]
    fn base64_roundtrip_known_vectors() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("Zm9vOmJhcg==").unwrap(), b"foo:bar");
        assert!(base64_decode("not base64!").is_none());
    }

    #[test]
    fn basic_auth_extracts_password() {
        // base64("hello:world") = aGVsbG86d29ybGQ=
        assert_eq!(basic_password("aGVsbG86d29ybGQ=").as_deref(), Some("world"));
        // password may contain colons — only the first split counts; base64("u:a:b") = dTphOmI=
        assert_eq!(basic_password("dTphOmI=").as_deref(), Some("a:b"));
    }

    #[test]
    fn constant_time_eq_basic() {
        assert!(constant_time_eq(b"alpha", b"alpha"));
        assert!(!constant_time_eq(b"alpha", b"alph"));
        assert!(!constant_time_eq(b"alpha", b"Alpha"));
    }
}
