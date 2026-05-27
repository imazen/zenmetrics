//! SaladCloud `JobQueue` — a local HTTP job receiver fed by the baked-in
//! `salad-http-job-queue-worker` sidecar.
//!
//! ## How Salad's managed queue reaches the app
//!
//! Salad's queue is *managed + push* (spec §1.9): the
//! `salad-http-job-queue-worker` Go sidecar (Apache-2.0,
//! github.com/SaladTechnologies/salad-cloud-job-queue-worker) runs
//! inside the container, holds the gRPC stream to Salad's managed queue
//! backend, and for each assigned job does:
//!
//! ```text
//! POST http://localhost:<job.port><job.path>
//! body = <job.input>            # opaque per-job bytes from the queue
//! ```
//!
//! then reads the **HTTP response body** back as the job *output* and
//! returns it to the queue (gRPC `CompleteJob`); a non-2xx / connection
//! error makes the sidecar reject the job so Salad re-queues it. (Source
//! of truth: the sidecar's `internal/workers/workers.go`
//! `http://localhost:%d%s` + `Output: responseBody`, and the
//! `service_v1alpha.proto` `Job { job_id, port, path, input }` /
//! `CompleteJob` / `RejectJob` messages.)
//!
//! So the app side speaks **plain HTTP**, not gRPC: the gRPC contract is
//! entirely between the sidecar and Salad's queue backend, internal to
//! the sidecar. This crate therefore serves a tiny HTTP endpoint rather
//! than embedding a tonic gRPC server — it matches the only sidecar
//! binary Salad actually ships (`salad-http-job-queue-worker`, used by
//! the upstream `samples/mandelbrot` `with-shell-script` /
//! `with-s6-overlay` integrations) and avoids a redundant gRPC layer.
//! See `SALAD.md` for the deviation note vs spec §1.9's "tonic" wording.
//!
//! ## Mapping onto the [`zen_cloud_core::JobQueue`] trait
//!
//! - [`SaladJobQueue::next_chunk`] blocks until the sidecar POSTs the
//!   next job, then returns it as a [`Chunk`] (`id` = `job_id` header if
//!   present, else a synthesised counter; `payload` = the request body).
//! - [`SaladJobQueue::ack_chunk`] hands the outcome back to the parked
//!   HTTP handler, which turns it into the HTTP response the sidecar
//!   reads: `Done`/`Skipped` → `200 OK`, `Failed`/`Retryable` → `5xx`
//!   so the sidecar rejects + Salad re-queues. One job at a time — the
//!   sidecar will not POST the next job until this one's response is
//!   written, which exactly matches the single-flight `run_worker` loop.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::mpsc as std_mpsc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use zen_cloud_core::{Chunk, ChunkId, ChunkOutcome, CloudError};

/// Header the sidecar's job-forward request carries the job id in, when
/// present. The HTTP `Job` payload itself does not put the id in the
/// body, so we surface it via a header the launcher's `path` can route
/// (Salad's job definition controls `path`; we standardise on this
/// header). Absent → we synthesise a monotonic id.
const JOB_ID_HEADER: &str = "x-salad-job-id";

/// One received job, plus the reply channel the HTTP handler is parked
/// on. The worker fills `reply` (via `ack_chunk`) with the bytes +
/// status the sidecar should read back as the job output.
struct ReceivedJob {
    id: ChunkId,
    body: Bytes,
    reply: oneshot::Sender<JobReply>,
}

/// What `ack_chunk` sends back to the parked HTTP handler.
struct JobReply {
    status: StatusCode,
    body: Bytes,
}

/// Configuration for the local job-receiver server.
#[derive(Clone, Debug)]
pub struct SaladQueueConfig {
    /// Address to bind the local HTTP job receiver on. Must match the
    /// `port` Salad's job definition targets (and the `path` is matched
    /// loosely — every POST is treated as a job). Defaults to
    /// `0.0.0.0:80` mirroring the upstream mandelbrot sample.
    pub bind_addr: SocketAddr,
}

impl Default for SaladQueueConfig {
    fn default() -> Self {
        Self {
            // 0.0.0.0:80 — the sidecar POSTs to localhost:<port>; the
            // mandelbrot sample binds the app on :80.
            bind_addr: SocketAddr::from(([0, 0, 0, 0], 80)),
        }
    }
}

/// Pull-shaped facade over Salad's push queue.
///
/// Internally a background tokio runtime runs the HTTP receiver; the
/// sync trait methods bridge to it over a std channel (for received
/// jobs) + a per-job oneshot (for the reply). This keeps the
/// [`zen_cloud_core::run_worker`] loop's single-flight, fully-sync
/// contract intact.
pub struct SaladJobQueue {
    /// Receives jobs the HTTP server accepted, in arrival order.
    job_rx: std_mpsc::Receiver<ReceivedJob>,
    /// The reply channel for the job currently in flight (set by
    /// `next_chunk`, consumed by `ack_chunk`).
    in_flight: Option<oneshot::Sender<JobReply>>,
    /// Keeps the background runtime alive for the queue's lifetime.
    _runtime: tokio::runtime::Runtime,
    /// Signals the server to shut down on drop.
    _shutdown: oneshot::Sender<()>,
}

impl SaladJobQueue {
    /// Bind the local job receiver and start serving. The returned queue
    /// is ready for `run_worker` to pull from.
    pub fn bind(config: SaladQueueConfig) -> Result<Self, CloudError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(|e| CloudError::Queue(format!("build salad queue runtime: {e}")))?;

        let (job_tx, job_rx) = std_mpsc::channel::<ReceivedJob>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        // Bind synchronously on the runtime so a bind failure (port in
        // use, permission) surfaces from `bind()` rather than being lost
        // in a detached task.
        let bind_addr = config.bind_addr;
        let listener = runtime
            .block_on(async move { TcpListener::bind(bind_addr).await })
            .map_err(|e| CloudError::Queue(format!("bind salad job receiver {bind_addr}: {e}")))?;
        tracing::info!("salad job receiver listening on {bind_addr}");

        runtime.spawn(async move {
            if let Err(e) = serve(listener, job_tx, shutdown_rx).await {
                tracing::error!("salad job receiver stopped: {e}");
            }
        });

        Ok(Self {
            job_rx,
            in_flight: None,
            _runtime: runtime,
            _shutdown: shutdown_tx,
        })
    }

    /// Convenience: bind on the default `0.0.0.0:80`.
    pub fn bind_default() -> Result<Self, CloudError> {
        Self::bind(SaladQueueConfig::default())
    }
}

impl zen_cloud_core::JobQueue for SaladJobQueue {
    fn next_chunk(&mut self) -> Result<Option<Chunk>, CloudError> {
        // A job left un-acked would strand the HTTP handler; the
        // run_worker loop always acks before pulling again, but guard
        // anyway by replying with a 500 if a prior reply was never sent.
        if let Some(reply) = self.in_flight.take() {
            let _ = reply.send(JobReply {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: Bytes::from_static(b"chunk not acknowledged"),
            });
        }
        match self.job_rx.recv() {
            Ok(job) => {
                self.in_flight = Some(job.reply);
                Ok(Some(Chunk {
                    id: job.id,
                    payload: String::from_utf8_lossy(&job.body).into_owned(),
                }))
            }
            // All senders dropped → the server stopped → queue drained.
            Err(_) => Ok(None),
        }
    }

    fn ack_chunk(&mut self, _id: &ChunkId, outcome: ChunkOutcome) -> Result<(), CloudError> {
        let reply = self
            .in_flight
            .take()
            .ok_or_else(|| CloudError::Queue("ack_chunk without an in-flight job".into()))?;
        let (status, body) = outcome_to_http(&outcome);
        // If the handler is gone (sidecar hung up), that is benign —
        // Salad will time the job out and re-queue it.
        let _ = reply.send(JobReply { status, body });
        Ok(())
    }
}

/// Map a [`ChunkOutcome`] to the HTTP response the sidecar reads back.
///
/// `Done`/`Skipped` → `200 OK` (the sidecar `CompleteJob`s the job; the
/// real artifacts are already in BlobStorage, so the HTTP body is just a
/// small status JSON). `Failed`/`Retryable` → `500` so the sidecar
/// rejects the job and Salad re-queues it for another instance.
fn outcome_to_http(outcome: &ChunkOutcome) -> (StatusCode, Bytes) {
    match outcome {
        ChunkOutcome::Done => (StatusCode::OK, Bytes::from_static(b"{\"status\":\"done\"}")),
        ChunkOutcome::Skipped { reason } => (
            StatusCode::OK,
            Bytes::from(format!(
                "{{\"status\":\"skipped\",\"reason\":{}}}",
                json_str(reason)
            )),
        ),
        ChunkOutcome::Retryable { error } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Bytes::from(format!(
                "{{\"status\":\"retryable\",\"error\":{}}}",
                json_str(error)
            )),
        ),
        ChunkOutcome::Failed { error } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Bytes::from(format!(
                "{{\"status\":\"failed\",\"error\":{}}}",
                json_str(error)
            )),
        ),
    }
}

/// Minimal JSON string escaping for the small status bodies above.
fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}

/// Run the local HTTP job receiver until `shutdown` fires or the
/// listener errors. Each accepted connection serves one job at a time
/// (the sidecar opens a fresh request per job). The listener is bound by
/// the caller so bind failures surface synchronously from `bind()`.
async fn serve(
    listener: TcpListener,
    job_tx: std_mpsc::Sender<ReceivedJob>,
    mut shutdown: oneshot::Receiver<()>,
) -> Result<(), String> {
    loop {
        let accept = tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("salad job receiver shutting down");
                return Ok(());
            }
            a = listener.accept() => a,
        };
        let (stream, _peer) = match accept {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("salad job receiver accept error: {e}");
                continue;
            }
        };
        let io = TokioIo::new(stream);
        let job_tx = job_tx.clone();
        tokio::spawn(async move {
            let service = service_fn(move |req| handle(req, job_tx.clone()));
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                tracing::debug!("salad job connection ended: {e}");
            }
        });
    }
}

/// Handle one inbound HTTP request from the sidecar: forward the body as
/// a job, park on the reply, then write it back as the HTTP response.
/// A `GET /health` (or any GET) returns `200 OK` so platform health
/// checks pass without consuming a job slot.
async fn handle(
    req: Request<Incoming>,
    job_tx: std_mpsc::Sender<ReceivedJob>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    if req.method() == hyper::Method::GET {
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .body(Full::new(Bytes::from_static(b"{\"status\":\"OK\"}")))
            .expect("static health response"));
    }

    // Pull the optional job id header before consuming the request.
    let job_id = req
        .headers()
        .get(JOB_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| ChunkId(s.to_owned()));

    let body = match req.into_body().collect().await {
        Ok(c) => c.to_bytes(),
        Err(e) => {
            return Ok(error_response(
                StatusCode::BAD_REQUEST,
                &format!("read job body: {e}"),
            ));
        }
    };

    let id = job_id.unwrap_or_else(|| ChunkId(format!("salad-{}", next_synthetic_id())));
    let (reply_tx, reply_rx) = oneshot::channel::<JobReply>();

    if job_tx
        .send(ReceivedJob {
            id,
            body,
            reply: reply_tx,
        })
        .is_err()
    {
        // The worker side is gone — nothing will ack. Tell the sidecar
        // to reject (re-queue) rather than mark the job complete.
        return Ok(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker not accepting jobs",
        ));
    }

    match reply_rx.await {
        Ok(JobReply { status, body }) => Ok(Response::builder()
            .status(status)
            .body(Full::new(body))
            .unwrap_or_else(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "bad reply"))),
        // Worker dropped the reply channel without acking — reject so
        // Salad re-queues.
        Err(_) => Ok(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "worker dropped job without acknowledgement",
        )),
    }
}

fn error_response(status: StatusCode, msg: &str) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .body(Full::new(Bytes::from(format!(
            "{{\"status\":\"error\",\"error\":{}}}",
            json_str(msg)
        ))))
        .expect("error response builds")
}

/// Monotonic fallback ids for jobs that arrive without an
/// `x-salad-job-id` header.
fn next_synthetic_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_done_is_200() {
        let (s, b) = outcome_to_http(&ChunkOutcome::Done);
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8_lossy(&b).contains("done"));
    }

    #[test]
    fn outcome_failed_is_500_and_carries_error() {
        let (s, b) = outcome_to_http(&ChunkOutcome::Failed {
            error: "boom \"quoted\"".into(),
        });
        assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
        let body = String::from_utf8_lossy(&b);
        assert!(body.contains("failed"));
        // The error string is JSON-escaped (quotes preserved as \").
        assert!(body.contains("\\\"quoted\\\""));
    }

    #[test]
    fn outcome_retryable_is_500() {
        let (s, _b) = outcome_to_http(&ChunkOutcome::Retryable {
            error: "net blip".into(),
        });
        assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn outcome_skipped_is_200() {
        let (s, b) = outcome_to_http(&ChunkOutcome::Skipped {
            reason: "exists".into(),
        });
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8_lossy(&b).contains("skipped"));
    }

    #[test]
    fn synthetic_ids_are_monotonic() {
        let a = next_synthetic_id();
        let b = next_synthetic_id();
        assert!(b > a);
    }

    #[test]
    fn default_config_binds_port_80() {
        assert_eq!(SaladQueueConfig::default().bind_addr.port(), 80);
    }
}
