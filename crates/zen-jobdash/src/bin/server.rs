#![forbid(unsafe_code)]
//! Railway entrypoint: serves the monitoring views + control API over the Parquet ledger.
//!
//! Config via env (set in the Railway dashboard):
//!   ZEN_LEDGER        comma-separated parquet ledger paths (or an R2-synced dir's files)
//!   ZEN_BLOB_INDEX    blob-index parquet path (optional)
//!   ZEN_WORKERS_JSON  worker-heartbeat JSON array path (optional)
//!   PORT              injected by Railway; defaults to 3000 locally (per the 3000-3999 rule)
//!
//! The dashboard never runs workers — it reads the ledger and emits control intents/plans.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};

use zen_job_core::{JobStatus, Sha256Hex};
use zen_jobdash::{
    cost_view, failures, gc_dry_run, progress, stop_spend, storage, ControlIntent, CostView,
    DashData, FailureCell, KindProgress, TierStorage,
};

type Shared = Arc<DashData>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = match load() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("zen-jobdash: warning: could not load ledger ({e}); serving empty views");
            DashData::default()
        }
    };
    eprintln!(
        "zen-jobdash: loaded {} ledger rows, {} blobs, {} workers",
        data.rows.len(),
        data.blobs.len(),
        data.workers.len()
    );
    let state: Shared = Arc::new(data);

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
    // ZEN_LEDGER entries may be local paths or s3:// URIs; ZEN_R2_ENDPOINT (+ AWS_* creds) is used
    // for the s3:// ones. This is how the deployed dashboard reads the live R2 ledger.
    let ledger: Vec<String> = std::env::var("ZEN_LEDGER")
        .map(|s| s.split(',').filter(|p| !p.is_empty()).map(String::from).collect())
        .unwrap_or_default();
    let endpoint = std::env::var("ZEN_R2_ENDPOINT").ok();
    let blob = std::env::var("ZEN_BLOB_INDEX").ok().map(PathBuf::from);
    let workers = std::env::var("ZEN_WORKERS_JSON").ok().map(PathBuf::from);
    DashData::from_sources(&ledger, endpoint.as_deref(), blob.as_deref(), workers.as_deref())
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

async fn api_progress(State(d): State<Shared>) -> Json<Vec<KindProgress>> {
    Json(progress(&d.rows))
}
async fn api_failures(State(d): State<Shared>) -> Json<Vec<FailureCell>> {
    Json(failures(&d.rows))
}
async fn api_cost(State(d): State<Shared>) -> Json<CostView> {
    Json(cost_view(&d.workers))
}
async fn api_storage(State(d): State<Shared>) -> Json<Vec<TierStorage>> {
    Json(storage(&d.blobs))
}

async fn api_control(State(d): State<Shared>, Json(intent): Json<ControlIntent>) -> Json<serde_json::Value> {
    let plan = match intent {
        ControlIntent::GcDryRun => {
            serde_json::to_value(gc_dry_run(&d.blobs, &referenced(&d), &HashSet::new()))
                .unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() }))
        }
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
pre{background:#f4f4f4;padding:.5rem;overflow:auto}</style></head>
<body><h1>zen-jobdash</h1>
<p>Control plane for the zen job system. Live views below; control via <code>POST /api/control</code>.</p>
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
(async()=>{
 const c=await j('/api/cost');
 const cd=document.getElementById('cost');cd.innerHTML='';
 const h=document.createElement('h2');h.textContent='cost';cd.appendChild(h);
 const pre=document.createElement('pre');pre.textContent=JSON.stringify(c,null,1);cd.appendChild(pre);
 setHtml('progress','progress per kind',tbl(await j('/api/progress')));
 setHtml('failures','failures',tbl(await j('/api/failures')));
 setHtml('storage','storage per tier',tbl(await j('/api/storage')));
})();
</script></body></html>"#;
