#![forbid(unsafe_code)]
//! Runnable single-node instance of the job system — proves the engine is *operational* (real
//! processes, real files), not just unit-tested. Local FS stands in for R2; this is exactly the
//! `zen-cloud-local` dev mode. Demonstrates, with real I/O:
//!   declare → reconcile(gap) → execute (real content-addressed blob writes + a fail + a transient) →
//!   persist Parquet ledger → reconcile (retry/poison) → compact → reconcile to CONVERGENCE →
//!   build a real blob index → GC (real delete of an orphan, tombstone, refuse the irreplaceable).
//!
//! Run: `cargo run -p zen-jobdash --bin local_demo`
//! Then serve it: `ZEN_LEDGER=<printed path> cargo run -p zen-jobdash` and open http://localhost:3000

use std::collections::HashSet;
use std::path::PathBuf;

use zen_job_core::{
    BlobIndexEntry, CellId, DesiredJob, ErrorClass, JobKind, JobStatus, LedgerRow, LedgerView,
    Regenerability, ResourceClass, RetryPolicy, Sha256Hex, WorkerReport, blob_key, gc_plan,
    reconcile, sha256,
};
use zen_ledger::{compact_ledger, read_ledger, write_blob_index, write_ledger};

fn cell(i: usize) -> CellId {
    CellId {
        image_path: format!("img/{i}.png"),
        codec: "zenjpeg".into(),
        q: 80,
        knob_tuple_json: "{}".into(),
    }
}

fn desired_set() -> Vec<DesiredJob> {
    (0..5)
        .map(|i| DesiredJob {
            kind: JobKind::Metric {
                metric: "cvvdp".into(),
            },
            inputs: vec![sha256(format!("encode-{i}").as_bytes())],
            cell: cell(i),
        })
        .collect()
}

fn row(
    d: &DesiredJob,
    status: JobStatus,
    err: Option<ErrorClass>,
    attempts: u32,
    ts: u64,
    out: Option<Sha256Hex>,
) -> LedgerRow {
    LedgerRow {
        job_id: d.job_id(),
        kind: d.kind.clone(),
        cell: d.cell.clone(),
        output_sha: out,
        status,
        error_class: err,
        attempts,
        ts,
        worker: "local-1".into(),
        provider: "local".into(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("zenjobsim_{}", std::process::id()));
    let blobs_dir = dir.join("blobs");
    std::fs::create_dir_all(&blobs_dir)?;
    let blob_path = |sha: &Sha256Hex| blobs_dir.join(sha.as_str()); // local stand-in for blobs/<sha>

    let desired = desired_set();
    let policy = RetryPolicy::default();
    println!(
        "== local single-node run ({} jobs) in {} ==",
        desired.len(),
        dir.display()
    );

    // Round 1 — empty ledger: the whole set is the gap.
    let p0 = reconcile(&desired, &LedgerView::new(), policy);
    println!("round1: enqueue={} done={}", p0.enqueue.len(), p0.done);

    // Execute: jobs 0-2 succeed (real blob written), job3 deterministic-fails, job4 transient-fails.
    let mut written: Vec<(Sha256Hex, Regenerability)> = Vec::new();
    let mut rows = Vec::new();
    for (i, d) in desired.iter().enumerate() {
        match i {
            3 => rows.push(row(
                d,
                JobStatus::Failed,
                Some(ErrorClass::DecodeError),
                1,
                100,
                None,
            )),
            4 => rows.push(row(
                d,
                JobStatus::Failed,
                Some(ErrorClass::Timeout),
                1,
                100,
                None,
            )),
            _ => {
                let bytes = format!("score-blob-{i}").into_bytes();
                let sha = sha256(&bytes);
                std::fs::write(blob_path(&sha), &bytes)?; // REAL content-addressed write
                written.push((sha.clone(), Regenerability::CheapRegenerable));
                rows.push(row(d, JobStatus::Done, None, 1, 100, Some(sha)));
            }
        }
    }
    let sc1 = dir.join("ledger-chunk1.parquet");
    write_ledger(&sc1, &rows)?;
    println!(
        "executed: wrote {} real blobs to {}",
        written.len(),
        blobs_dir.display()
    );

    // Round 2 — read the persisted ledger back: done=3, transient retried, deterministic poisoned.
    let v1 = LedgerView::from_rows(read_ledger(&sc1)?);
    let p1 = reconcile(&desired, &v1, policy);
    println!(
        "round2: done={} retry={} poison={}",
        p1.done,
        p1.enqueue.len(),
        p1.poison.len()
    );

    // Retry job4 → success (real blob); record job3 POISON.
    let bytes4 = b"score-blob-4-retry".to_vec();
    let sha4 = sha256(&bytes4);
    std::fs::write(blob_path(&sha4), &bytes4)?;
    written.push((sha4.clone(), Regenerability::CheapRegenerable));
    let rows2 = vec![
        row(&desired[4], JobStatus::Done, None, 2, 200, Some(sha4)),
        row(
            &desired[3],
            JobStatus::Poison,
            Some(ErrorClass::DecodeError),
            1,
            200,
            None,
        ),
    ];
    let sc2 = dir.join("ledger-chunk2.parquet");
    write_ledger(&sc2, &rows2)?;

    // Compact the two sidecars → latest-wins consolidated ledger.
    let ledger = dir.join("ledger.parquet");
    let n = compact_ledger(&[&sc1, &sc2], &ledger)?;
    println!("compacted 2 sidecars -> {n} rows in {}", ledger.display());

    // Round 3 — converged: nothing left to enqueue.
    let v2 = LedgerView::from_rows(read_ledger(&ledger)?);
    let p2 = reconcile(&desired, &v2, policy);
    println!(
        "round3: done={} enqueue={} poison={}  => CONVERGED={}",
        p2.done,
        p2.enqueue.len(),
        p2.poison.len(),
        p2.enqueue.is_empty()
    );
    assert!(p2.enqueue.is_empty(), "must converge");

    // Inject an irreplaceable orphan + a cheap orphan, then GC for real.
    let cheap_orphan = sha256(b"orphan-jpeg-bytes");
    std::fs::write(blob_path(&cheap_orphan), b"orphan-jpeg-bytes")?;
    written.push((cheap_orphan.clone(), Regenerability::CheapRegenerable));
    let src_orphan = sha256(b"only-copy-source");
    std::fs::write(blob_path(&src_orphan), b"only-copy-source")?;
    written.push((src_orphan.clone(), Regenerability::NotRegenerable));

    // Real blob index from on-disk files (size from metadata).
    let mut index = Vec::new();
    for (sha, regen) in &written {
        let size = std::fs::metadata(blob_path(sha))?.len();
        index.push(BlobIndexEntry {
            sha: sha.clone(),
            size,
            regenerability: *regen,
            last_ref_secs: 0,
        });
    }
    // Referenced = output blobs of Done rows in the compacted ledger.
    let referenced: HashSet<Sha256Hex> = read_ledger(&ledger)?
        .into_iter()
        .filter(|r| r.status == JobStatus::Done)
        .filter_map(|r| r.output_sha)
        .collect();

    let plan = gc_plan(&index, &referenced, &HashSet::new());
    // Actually delete the cheap orphans; tombstone them; never touch refused.
    let mut tombstones = Vec::new();
    for sha in &plan.evict_cheap {
        std::fs::remove_file(blob_path(sha))?; // REAL delete
        tombstones.push(serde_json::json!({ "sha": sha.as_str(), "reason": "gc_evict_cheap", "regenerable": true }));
    }
    std::fs::write(
        dir.join("tombstones.json"),
        serde_json::to_vec_pretty(&tombstones)?,
    )?;

    println!(
        "gc: kept={} evicted_cheap={} (REAL deletes) refused_surface={} (kept, escalated)",
        plan.keep.len(),
        plan.evict_cheap.len(),
        plan.refuse_surface.len()
    );
    // Prove it: referenced + refused blobs still exist; cheap orphan is gone.
    assert!(
        blob_path(&src_orphan).exists(),
        "irreplaceable orphan must NOT be deleted"
    );
    assert!(
        !blob_path(&cheap_orphan).exists(),
        "cheap orphan must be deleted"
    );
    for sha in &referenced {
        assert!(blob_path(sha).exists(), "referenced blob must survive GC");
    }
    println!(
        "verified on disk: referenced + irreplaceable survive; cheap orphan deleted; tombstone written"
    );

    // Materialize the dashboard's other inputs so B's storage/cost panes + C's GC dry-run show real
    // numbers over HTTP: a blob index of the survivors + a sample worker-heartbeat file.
    let survivors: Vec<BlobIndexEntry> = index
        .iter()
        .filter(|e| blob_path(&e.sha).exists())
        .cloned()
        .collect();
    let blob_index = dir.join("blob_index.parquet");
    write_blob_index(&blob_index, &survivors)?;

    let workers = vec![
        WorkerReport {
            worker: "oracle-arm-1".into(),
            provider: "oracle".into(),
            class: ResourceClass::CpuArm,
            rate_usd_per_hr: 0.0,
            uptime_secs: 3600,
            jobs_done: 2,
        },
        WorkerReport {
            worker: "vast-gpu-1".into(),
            provider: "vast".into(),
            class: ResourceClass::Gpu,
            rate_usd_per_hr: 0.35,
            uptime_secs: 1800,
            jobs_done: 2,
        },
    ];
    let workers_json = dir.join("workers.json");
    std::fs::write(&workers_json, serde_json::to_vec_pretty(&workers)?)?;

    println!("\nOPERATIONAL. Serve this run's dashboard with:");
    let lp: PathBuf = std::fs::canonicalize(&ledger)?;
    let bp: PathBuf = std::fs::canonicalize(&blob_index)?;
    let wp: PathBuf = std::fs::canonicalize(&workers_json)?;
    println!(
        "  ZEN_LEDGER={} ZEN_BLOB_INDEX={} ZEN_WORKERS_JSON={} PORT=3137 cargo run -p zen-jobdash",
        lp.display(),
        bp.display(),
        wp.display()
    );
    // machine-readable lines for scripting
    println!("LEDGER_PATH={}", lp.display());
    println!("BLOB_INDEX_PATH={}", bp.display());
    println!("WORKERS_JSON={}", wp.display());
    let _ = blob_key(&src_orphan); // (blob_key is the R2 key form; unused locally but part of the contract)
    Ok(())
}
