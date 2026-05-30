#![forbid(unsafe_code)]
//! Railway entrypoint: serves monitoring views + control API over the Parquet ledger, **live-reloading**
//! it on an interval (so views reflect the fleet's real runs) and **firing notifications** to a webhook.
//!
//! Config via env (Railway dashboard):
//!   ZEN_LEDGER, ZEN_BLOB_INDEX, ZEN_WORKERS_JSON  — local paths or s3:// URIs (R2)
//!   ZEN_R2_ENDPOINT                               — R2 endpoint for s3:// (+ AWS_* creds)
//!   ZEN_REFRESH_SECS                              — ledger reload interval (default 30)
//!   ZEN_NOTIFY_WEBHOOK                            — Slack/Discord/ntfy webhook URL (optional; goal D)
//!   ZEN_PUBLIC_URL                                — base URL for notification deep links
//!   ZEN_BUDGET_CAP_USD (default 0=off), ZEN_POISON_THRESHOLD (default 10)
//!   PORT                                          — injected by Railway (default 3000)
//!
//! The dashboard never runs workers — it reads the ledger and emits control intents/plans.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::RwLock;

use zen_job_core::{JobStatus, Sha256Hex};
use zen_jobdash::{
    cost_view, detect, failures, format_event, gc_dry_run, progress, stop_spend, storage,
    ControlIntent, CostView, DashData, FailureCell, KindProgress, NotifyPayload, TierStorage,
};

type Shared = Arc<RwLock<DashData>>;

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
    let state: Shared = Arc::new(RwLock::new(data));

    // Background: reload the ledger on an interval (live views) + fire notifications (goal D).
    tokio::spawn(refresh_loop(state.clone()));

    let app = Router::new()
        .route("/", get(index))
        .route("/api/progress", get(api_progress))
        .route("/api/failures", get(api_failures))
        .route("/api/cost", get(api_cost))
        .route("/api/storage", get(api_storage))
        .route("/api/control", post(api_control))
        .with_state(state);

    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(3000);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("zen-jobdash listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn load() -> Result<DashData, zen_jobdash::DashError> {
    // ZEN_LEDGER/BLOB_INDEX/WORKERS may be local paths or s3:// URIs; ZEN_R2_ENDPOINT (+ AWS_* creds)
    // staging is used for s3://. This is how the deployed dashboard reads the live R2 ledger.
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
async fn refresh_loop(state: Shared) {
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

/// POST a notification to a webhook. Body is `{"text": "..."}` — accepted by Slack incoming webhooks
/// and ntfy; Discord/email gateways take the same shape via most relays. Best-effort.
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

async fn api_progress(State(s): State<Shared>) -> Json<Vec<KindProgress>> {
    Json(progress(&s.read().await.rows))
}
async fn api_failures(State(s): State<Shared>) -> Json<Vec<FailureCell>> {
    Json(failures(&s.read().await.rows))
}
async fn api_cost(State(s): State<Shared>) -> Json<CostView> {
    Json(cost_view(&s.read().await.workers))
}
async fn api_storage(State(s): State<Shared>) -> Json<Vec<TierStorage>> {
    Json(storage(&s.read().await.blobs))
}

async fn api_control(State(s): State<Shared>, Json(intent): Json<ControlIntent>) -> Json<serde_json::Value> {
    let guard = s.read().await;
    let d: &DashData = &guard;
    let plan = match intent {
        ControlIntent::GcDryRun => serde_json::to_value(gc_dry_run(&d.blobs, &referenced(d), &HashSet::new()))
            .unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() })),
        ControlIntent::StopSpend { cap_usd } => {
            let spent = cost_view(&d.workers).total_spent_usd;
            serde_json::to_value(stop_spend(&d.workers, spent, cap_usd))
                .unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() }))
        }
        // Kill/Pause/Drain are actuated by the fleet layer; the dashboard records the intent.
        other => serde_json::json!({ "intent": other, "status": "queued_for_fleet_actuation" }),
    };
    Json(plan)
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

const INDEX_HTML: &str = r#"<!doctype html>
<html><head><meta charset="utf-8"><title>zen-jobdash</title>
<style>body{font:14px system-ui,monospace;margin:2rem;max-width:60rem}
h1{font-size:1.2rem}table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #ccc;padding:.2rem .5rem}
pre{background:#f4f4f4;padding:.5rem;overflow:auto}.t{color:#888;font-size:.8rem}</style></head>
<body><h1>zen-jobdash</h1>
<p>Control plane for the zen job system. Live (auto-refreshing) views below; control via <code>POST /api/control</code>.</p>
<div class="t" id="ts"></div>
<div id="cost"></div><div id="progress"></div><div id="failures"></div><div id="storage"></div>
<script>
async function j(u){return (await fetch(u)).json()}
// escape ledger-derived values (worker/provider/path/codec) before inserting into the DOM
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function tbl(rows){if(!rows.length)return '<p>(none)</p>';const k=Object.keys(rows[0]);
 return '<table><tr>'+k.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>'+
 rows.map(r=>'<tr>'+k.map(x=>'<td>'+esc(r[x]??'')+'</td>').join('')+'</tr>').join('')+'</table>'}
function setHtml(id,heading,body){const el=document.getElementById(id);el.innerHTML='';
 const h=document.createElement('h2');h.textContent=heading;el.appendChild(h);
 const div=document.createElement('div');div.innerHTML=body;el.appendChild(div)}
async function render(){
 try{
  const c=await j('/api/cost');
  const cd=document.getElementById('cost');cd.innerHTML='';
  const h=document.createElement('h2');h.textContent='cost';cd.appendChild(h);
  const pre=document.createElement('pre');pre.textContent=JSON.stringify(c,null,1);cd.appendChild(pre);
  setHtml('progress','progress per kind',tbl(await j('/api/progress')));
  setHtml('failures','failures',tbl(await j('/api/failures')));
  setHtml('storage','storage per tier',tbl(await j('/api/storage')));
  document.getElementById('ts').textContent='updated '+new Date().toLocaleTimeString();
 }catch(e){document.getElementById('ts').textContent='refresh error: '+e}
}
render(); setInterval(render, 15000);
</script></body></html>"#;
