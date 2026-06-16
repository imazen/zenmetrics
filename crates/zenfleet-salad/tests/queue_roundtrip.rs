//! End-to-end test of the Salad `JobQueue` HTTP receiver.
//!
//! Simulates the `salad-http-job-queue-worker` sidecar: bind the queue
//! on an ephemeral port, POST a job body to it (as the sidecar would),
//! and assert that `next_chunk` surfaces the body as a `Chunk` and that
//! `ack_chunk(Done)` makes the receiver return `200 OK` with the
//! done-status body the sidecar reads back as the job output. This is
//! the only part of the Salad glue testable off a Salad node (no IMDS,
//! no real sidecar); the `compute` closure itself is backend-agnostic
//! and is covered by the worker's own tests.

use std::net::SocketAddr;

use zenfleet_cloud::{ChunkOutcome, JobQueue};
use zenfleet_salad::queue::{SaladJobQueue, SaladQueueConfig};

#[test]
fn sidecar_post_roundtrips_through_next_chunk_and_ack() {
    // Bind on an OS-assigned port (port 0) so the test never collides
    // with port 80 or a peer test.
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    // We need the actual bound port; bind a probe listener to discover a
    // free port, drop it, then bind the queue there. (The queue binds
    // internally, so we pick a port that was just free.)
    let probe = std::net::TcpListener::bind(addr).unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let mut queue = SaladJobQueue::bind(SaladQueueConfig {
        bind_addr: SocketAddr::from(([127, 0, 0, 1], port)),
    })
    .expect("bind salad queue");

    // Simulate the sidecar POSTing a job on a background thread (the
    // POST blocks until the worker acks, exactly like the real sidecar).
    let url = format!("http://127.0.0.1:{port}/job");
    let poster = std::thread::spawn(move || {
        // Give the server a beat to be ready to accept.
        let client = reqwest::blocking::Client::new();
        // Retry briefly in case the listener isn't accepting yet.
        let mut last_err = None;
        for _ in 0..50 {
            match client
                .post(&url)
                .header("x-salad-job-id", "job-42")
                .body(r#"{"chunk_id":"job-42","codec":"jpeg"}"#)
                .send()
            {
                Ok(resp) => {
                    let status = resp.status();
                    let text = resp.text().unwrap_or_default();
                    return Ok((status, text));
                }
                Err(e) => {
                    last_err = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
        }
        Err(last_err.unwrap())
    });

    // The worker pulls the job.
    let chunk = queue
        .next_chunk()
        .expect("next_chunk ok")
        .expect("a chunk arrives");
    assert_eq!(chunk.id.as_str(), "job-42");
    assert!(chunk.payload.contains("\"codec\":\"jpeg\""));

    // The worker acks Done — the receiver replies 200 to the sidecar.
    queue
        .ack_chunk(&chunk.id, ChunkOutcome::Done)
        .expect("ack ok");

    let (status, text) = poster.join().unwrap().expect("sidecar POST completed");
    assert_eq!(status.as_u16(), 200);
    assert!(text.contains("done"), "body was: {text}");
}

#[test]
fn failed_outcome_makes_receiver_return_5xx() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let probe = std::net::TcpListener::bind(addr).unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let mut queue = SaladJobQueue::bind(SaladQueueConfig {
        bind_addr: SocketAddr::from(([127, 0, 0, 1], port)),
    })
    .expect("bind salad queue");

    let url = format!("http://127.0.0.1:{port}/job");
    let poster = std::thread::spawn(move || {
        let client = reqwest::blocking::Client::new();
        for _ in 0..50 {
            if let Ok(resp) = client.post(&url).body("payload").send() {
                return resp.status().as_u16();
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        0
    });

    let chunk = queue.next_chunk().unwrap().unwrap();
    queue
        .ack_chunk(
            &chunk.id,
            ChunkOutcome::Failed {
                error: "boom".into(),
            },
        )
        .unwrap();

    let status = poster.join().unwrap();
    assert_eq!(
        status, 500,
        "failed outcome should be 5xx so salad re-queues"
    );
}
