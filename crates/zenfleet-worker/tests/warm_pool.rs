//! Warm executor pool tests (#45 — the warm-process fix): reuse, crash isolation, recycle
//! policy, error-class frames, and the one-shot-vs-warm byte-identity A/B — all against the
//! scriptable `fake-serve-exec` bin (see its module docs for the behavior table).
//!
//! Unix-only: each test hands the pool a tiny wrapper script that exports a PER-TEST
//! `FAKE_SERVE_LOG` (spawn-count file) — no process-global env mutation, fully parallel-safe.
#![cfg(unix)]

use std::sync::atomic::{AtomicU64, Ordering};
use zenfleet_core::{CellId, DesiredJob, ErrorClass, JobKind, sha256};
use zenfleet_worker::{WarmExecPool, WarmPoolCfg, exec_command};

const FAKE: &str = env!("CARGO_BIN_EXE_fake-serve-exec");

static N: AtomicU64 = AtomicU64::new(0);

fn tmpdir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "warm_pool_{}_{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A ScoreFile job whose `cell.codec` selects the fake executor's behavior.
fn job(codec: &str, seed: &[u8]) -> DesiredJob {
    DesiredJob::new(
        JobKind::ScoreFile {
            metrics: vec!["ssim2".into()],
            hdr: false,
            hdr_transfer: None,
        },
        vec![sha256(seed)],
        CellId {
            image_path: "img/ref.png".into(),
            codec: codec.into(),
            q: 0,
            knob_tuple_json: "{}".into(),
        },
    )
}

fn starts_logged(log: &std::path::Path) -> usize {
    std::fs::read_to_string(log)
        .map(|s| s.lines().filter(|l| l.starts_with("start ")).count())
        .unwrap_or(0)
}

fn cfg_unbounded() -> WarmPoolCfg {
    WarmPoolCfg {
        rss_max_bytes: 0,
        max_jobs_per_child: 0,
    }
}

/// Build a per-test wrapper script that exports a private `FAKE_SERVE_LOG` and execs the fake
/// executor — no process-global env mutation, so tests parallelize freely. Returns
/// `(wrapper_path, log_path, tempdir)`.
fn wrapped_fake() -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let dir = tmpdir();
    let log = dir.join("starts.log");
    let wrapper = dir.join("fake_wrapped.sh");
    std::fs::write(
        &wrapper,
        format!(
            "#!/bin/sh\nFAKE_SERVE_LOG={} exec {} \"$@\"\n",
            log.display(),
            FAKE
        ),
    )
    .unwrap();
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
    (wrapper, log, dir)
}

fn with_logged_pool<T>(
    cfg: WarmPoolCfg,
    body: impl FnOnce(&WarmExecPool, &std::path::Path) -> T,
) -> T {
    let (wrapper, log, dir) = wrapped_fake();
    let pool = WarmExecPool::new(wrapper.to_str().unwrap(), cfg);
    let out = body(&pool, &log);
    drop(pool);
    std::fs::remove_dir_all(&dir).ok();
    out
}

/// One warm child serves many jobs (the whole point): 3 jobs → 1 process start, and every
/// response echoes its own job JSON.
#[test]
fn warm_pool_reuses_one_child_across_jobs() {
    with_logged_pool(cfg_unbounded(), |pool, log| {
        for i in 0..3u8 {
            let j = job("ok", &[i]);
            let out = pool.run_job(&j).expect("ok job succeeds");
            assert_eq!(out, serde_json::to_vec(&j).unwrap());
        }
        assert_eq!(starts_logged(log), 1, "one warm child, reused");
    });
}

/// A/B byte-identity gate: the warm path's output bytes are IDENTICAL to the one-shot path's for
/// the same job (both echo the job JSON — the fake's one-shot mode mirrors `/bin/cat`). This is
/// process topology, not math.
#[test]
fn warm_vs_oneshot_outputs_are_byte_identical() {
    with_logged_pool(cfg_unbounded(), |pool, _| {
        let j = job("ok", b"ab");
        let warm = pool.run_job(&j).expect("warm ok");
        let oneshot = exec_command(FAKE, &j).expect("one-shot ok");
        assert_eq!(warm, oneshot, "same bytes from both process topologies");
    });
}

/// Crash isolation: a child dying MID-JOB fails that job as transient `WorkerLost`, the next job
/// spawns a fresh child and succeeds, and the pass never dies.
#[test]
fn child_death_is_transient_and_respawns() {
    with_logged_pool(cfg_unbounded(), |pool, log| {
        pool.run_job(&job("ok", b"a")).expect("warm-up ok");
        let err = pool
            .run_job(&job("die", b"b"))
            .expect_err("dead child fails the job");
        assert_eq!(err.class, ErrorClass::WorkerLost, "msg: {}", err.msg);
        assert!(err.class.is_transient());
        let out = pool
            .run_job(&job("ok", b"c"))
            .expect("respawned child serves");
        assert!(!out.is_empty());
        assert_eq!(starts_logged(log), 2, "death → exactly one respawn");
    });
}

/// A framed per-job error (status 1) does NOT kill the warm child — deterministic failure for
/// that job, same child serves the next.
#[test]
fn framed_job_error_keeps_child_warm() {
    with_logged_pool(cfg_unbounded(), |pool, log| {
        pool.run_job(&job("ok", b"a")).expect("ok");
        let err = pool
            .run_job(&job("job-error", b"b"))
            .expect_err("framed error");
        assert_eq!(err.class, ErrorClass::EncoderPanic);
        assert_eq!(err.msg, "boom");
        pool.run_job(&job("ok", b"c"))
            .expect("same child still alive");
        assert_eq!(
            starts_logged(log),
            1,
            "framed error must not recycle the child"
        );
    });
}

/// A status-3 CLASSIFIED error frame maps to the real ErrorClass — the serve-mode half of the
/// error-class fidelity fix (CUDA OOM must not poison as encoder_panic).
#[test]
fn classified_frame_carries_real_error_class() {
    with_logged_pool(cfg_unbounded(), |pool, _| {
        let err = pool
            .run_job(&job("classified-oom", b"a"))
            .expect_err("classified failure");
        assert_eq!(err.class, ErrorClass::Oom, "msg: {}", err.msg);
        assert!(
            err.class.is_transient(),
            "OOM retries, never poisons directly"
        );
        assert_eq!(err.msg, "vram exhausted");
    });
}

/// `max_jobs_per_child = 1` recycles after every job: 3 jobs → 3 process starts.
#[test]
fn recycles_at_job_cap() {
    let cfg = WarmPoolCfg {
        rss_max_bytes: 0,
        max_jobs_per_child: 1,
    };
    with_logged_pool(cfg, |pool, log| {
        for i in 0..3u8 {
            pool.run_job(&job("ok", &[i])).expect("ok");
        }
        assert_eq!(starts_logged(log), 3, "job cap 1 → fresh child per job");
    });
}

/// RSS watermark: a child that ballooned past the watermark (the `grow` job leaks+touches
/// ~96 MiB) is recycled after that job; the next job runs on a fresh child. Linux-only
/// (`/proc/<pid>/status`).
#[test]
#[cfg(target_os = "linux")]
fn recycles_at_rss_watermark() {
    let cfg = WarmPoolCfg {
        rss_max_bytes: 48 << 20, // 48 MiB — well under the ~96 MiB the grow job materializes
        max_jobs_per_child: 0,
    };
    with_logged_pool(cfg, |pool, log| {
        pool.run_job(&job("grow", b"a")).expect("grow job succeeds");
        pool.run_job(&job("ok", b"b")).expect("next job ok");
        assert_eq!(
            starts_logged(log),
            2,
            "over-watermark child recycled after its job"
        );
    });
}

/// Concurrent callers each get their own child (the chunked path's admission concurrency),
/// and results stay per-job correct.
#[test]
fn concurrent_jobs_get_distinct_children() {
    with_logged_pool(cfg_unbounded(), |pool, log| {
        std::thread::scope(|s| {
            for i in 0..4u8 {
                s.spawn(move || {
                    // Every thread runs several jobs so children interleave through the pool.
                    for k in 0..5u8 {
                        let j = job("ok", &[i, k]);
                        let out = pool.run_job(&j).expect("ok");
                        assert_eq!(out, serde_json::to_vec(&j).unwrap());
                    }
                });
            }
        });
        let n = starts_logged(log);
        assert!(
            (1..=4).contains(&n),
            "children bounded by peak concurrency (got {n})"
        );
    });
}
