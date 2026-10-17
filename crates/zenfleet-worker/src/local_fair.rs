//! Opt-in single-owner local execution: one memory-bounded gap, no chunk barrier.
//! Uses the native reconciler, handler, blob store and durable ledger writer.

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

#[allow(clippy::too_many_arguments)]
pub(super) fn execute<H, B, F>(
    desired: &[DesiredJob],
    view: &LedgerView,
    policy: RetryPolicy,
    handler: &H,
    store: &B,
    params: &ChunkParams,
    flush: &mut F,
    ctx: WorkerCtx<'_>,
) -> Result<ExecOutcome, WorkerRunError>
where
    H: Fn(&DesiredJob) -> Result<Vec<u8>, HandlerError> + Sync,
    B: BlobStore + Sync,
    F: FnMut(&str, &[LedgerRow]) -> Result<(), String>,
{
    let plan = reconcile_at(desired, view, policy, ctx.now, policy.stale_claim_after);
    let by_id: HashMap<_, _> = desired.iter().map(|d| (d.job_id(), d)).collect();
    let gap: Vec<_> = plan
        .enqueue
        .iter()
        .filter_map(|id| by_id.get(id).copied())
        .collect();
    for job in &gap {
        let h = job.hint.unwrap_or(params.fallback_hint);
        let vram = h
            .vram_bytes
            .unwrap_or_else(|| default_vram_estimate(&job.kind));
        if h.peak_mem_bytes > params.budget.ram_budget_bytes
            || h.threads.max(1) > params.budget.cores
            || params
                .budget
                .vram_budget_bytes
                .is_some_and(|cap| vram > cap)
        {
            return Err(WorkerRunError::Config(format!(
                "local fair job {} cannot fit its fixed budget: hint={h:?} budget={:?}",
                job.job_id().as_str(),
                params.budget
            )));
        }
    }
    let mut small = Vec::new();
    let mut large = Vec::new();
    for (i, job) in gap.iter().enumerate() {
        let h = job.hint.unwrap_or(params.fallback_hint);
        if h.peak_mem_bytes >= 1024 * 1024 * 1024 {
            large.push(i);
        } else {
            small.push(i);
        }
    }
    small.sort_by_key(|&i| gap[i].hint.unwrap_or(params.fallback_hint).peak_mem_bytes);
    large.sort_by_key(|&i| {
        std::cmp::Reverse(gap[i].hint.unwrap_or(params.fallback_hint).peak_mem_bytes)
    });
    // Start a large job, then admit seven small/mid jobs before the next large
    // opportunity. Backfill scans past a temporarily unaffordable candidate.
    let mut ordered = Vec::with_capacity(gap.len());
    let mut small = small.into_iter();
    let mut large = large.into_iter();
    loop {
        let before = ordered.len();
        if let Some(i) = large.next() {
            ordered.push(i);
        }
        ordered.extend(small.by_ref().take(7));
        if before == ordered.len() {
            break;
        }
    }
    struct Shared {
        pending: Vec<usize>,
        running: InFlight,
    }
    let shared = Mutex::new(Shared {
        pending: ordered,
        running: InFlight::default(),
    });
    let cv = Condvar::new();
    let slots = params.budget.cores.clamp(1, 24) as usize;
    let reserve = std::env::var("ZEN_LOCAL_RESERVE_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let (tx, rx) = mpsc::channel();
    let halted = AtomicBool::new(false);
    let mut publication_error = None;
    let mut out = ExecOutcome {
        rows: Vec::new(),
        done: 0,
        failed: 0,
        poisoned: 0,
        skipped: 0,
        already_poison: plan.already_poison,
    };
    let make = |d: &DesiredJob, status, output_sha, error_class| LedgerRow {
        job_id: d.job_id(),
        kind: d.kind.clone(),
        cell: d.cell.clone(),
        output_sha,
        status,
        error_class,
        attempts: view.get(&d.job_id()).map_or(1, |r| r.attempts + 1),
        ts: ctx.now,
        worker: ctx.worker.to_owned(),
        provider: ctx.provider.to_owned(),
    };
    eprintln!(
        "zenfleet-worker: local fair gap={} slots={slots} ram_budget_bytes={} per-cell durable ledger",
        gap.len(),
        params.budget.ram_budget_bytes
    );
    std::thread::scope(|scope| {
        for _ in 0..slots.min(gap.len()) {
            let tx = tx.clone();
            let gap = &gap;
            let shared = &shared;
            let cv = &cv;
            let halted = &halted;
            scope.spawn(move || loop {
                let (gi, h, vram) = {
                    let mut g = shared.lock().unwrap_or_else(|p| p.into_inner());
                    loop {
                        if halted.load(Ordering::Acquire) || g.pending.is_empty() { return; }
                        let live = if reserve == 0 { u64::MAX } else {
                            std::fs::read_to_string("/proc/meminfo").ok().and_then(|s|
                                s.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v|v.parse::<u64>().ok()))
                                .unwrap_or(0).saturating_mul(1024).saturating_sub(reserve)
                        };
                        let candidate = g.pending.iter().position(|&i| {
                            let h = gap[i].hint.unwrap_or(params.fallback_hint);
                            let vram = h.vram_bytes.unwrap_or_else(|| default_vram_estimate(&gap[i].kind));
                            g.running.mem_bytes.saturating_add(h.peak_mem_bytes) <= live && h.peak_mem_bytes <= params.budget.ram_budget_bytes && params.budget.can_admit(&g.running, h.peak_mem_bytes, h.threads.max(1), vram)
                        });
                        if let Some(pos) = candidate {
                            let gi = g.pending.remove(pos);
                            let h = gap[gi].hint.unwrap_or(params.fallback_hint);
                            let vram = h.vram_bytes.unwrap_or_else(|| default_vram_estimate(&gap[gi].kind));
                            g.running.add(h.peak_mem_bytes, h.threads.max(1), vram);
                            eprintln!("zenfleet-worker[local-fair] admit gi={gi} count={} ram_bytes={} job_mem_bytes={}", g.running.count, g.running.mem_bytes, h.peak_mem_bytes);
                            break (gi, h, vram);
                        }
                        g = cv.wait_timeout(g, std::time::Duration::from_secs(1)).unwrap_or_else(|p| p.into_inner()).0;
                    }
                };
                let result = handler(gap[gi]).and_then(|bytes| store.put(&bytes)
                    .map_err(|e| HandlerError::new(ErrorClass::UploadFail, e.to_string())));
                if let Err(e) = &result { eprintln!("zenfleet-worker: local cell failed [{}]: {}", gap[gi].job_id().as_str(), failure_tail(&e.msg)); }
                {
                    let mut g = shared.lock().unwrap_or_else(|p| p.into_inner());
                    g.running.remove(h.peak_mem_bytes, h.threads.max(1), vram);
                }
                cv.notify_all();
                // Do not admit another producer until this row is durably
                // acknowledged or publication has closed the queue.
                let (ack, wait) = mpsc::sync_channel(0);
                if tx.send((gi, result, ack)).is_err() || wait.recv() != Ok(true) { return; }
            });
        }
        drop(tx);
        for (gi, result, ack) in rx {
            let d = gap[gi];
            let row = match result {
                Ok(sha) => make(d, JobStatus::Done, Some(sha), None),
                Err(e) => make(d, JobStatus::Failed, None, Some(e.class)),
            };
            // Blob is durable before its native one-cell ledger sidecar.
            let published = publish(flush, &chunk_id(&[d.job_id()]), &row);
            if let Err(error) = &published {
                {
                    let _closed = shared.lock().unwrap_or_else(|p| p.into_inner());
                    halted.store(true, Ordering::Release);
                }
                cv.notify_all();
                eprintln!("zenfleet-worker: unresolved local publication: {error}");
                publication_error.get_or_insert_with(|| error.clone());
            } else if row.status == JobStatus::Done {
                out.done += 1;
            } else {
                out.failed += 1;
            }
            let _ = ack.send(published.is_ok());
            out.rows.push(row);
        }
    });
    for id in &plan.poison {
        if let Some(d) = by_id.get(id) {
            let row = make(
                d,
                JobStatus::Poison,
                None,
                view.get(id).and_then(|r| r.error_class),
            );
            if let Err(error) = publish(flush, &chunk_id(std::slice::from_ref(id)), &row) {
                eprintln!("zenfleet-worker: unresolved local poison publication: {error}");
                publication_error.get_or_insert(error);
            } else {
                out.poisoned += 1;
            }
            out.rows.push(row);
        }
    }
    match publication_error {
        Some(error) => Err(WorkerRunError::Ledger(error)),
        None => Ok(out),
    }
}

fn publish<F>(flush: &mut F, id: &str, row: &LedgerRow) -> Result<(), String>
where
    F: FnMut(&str, &[LedgerRow]) -> Result<(), String>,
{
    let mut error = String::new();
    for attempt in 0..3 {
        match flush(id, std::slice::from_ref(row)) {
            Ok(()) => return Ok(()),
            Err(e) => error = e,
        }
        if attempt < 2 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    Err(error)
}

// Keep the full singleton digest. Existing eight-hex chunk files remain valid
// ledger inputs; new local fair rows cannot overwrite a truncated namespace.
pub(super) fn ledger_uri(ledger_out: &str, id: &str) -> String {
    let tag = id.strip_prefix("chunk-").unwrap_or(id);
    match ledger_out.rsplit_once('.') {
        Some((stem, ext)) if !ext.contains('/') => format!("{stem}.cell-{tag}.{ext}"),
        _ => format!("{ledger_out}.cell-{tag}.parquet"),
    }
}

// A failed sidecar retains the exact unresolved row in a synced diagnostic.
// Restart refuses it until the native ledger explicitly accounts for that row;
// retrying publication never calls the producer a second time.
pub(super) fn flush_checked(ledger_out: &Path, id: &str, rows: &[LedgerRow]) -> Result<(), String> {
    let uri = ledger_uri(&ledger_out.to_string_lossy(), id);
    match atomic_write(&uri, rows) {
        Ok(()) => archive_marker(ledger_out, id, rows),
        Err(error) => {
            let bytes = serde_json::to_vec(rows).map_err(|e| e.to_string())?;
            let marker = publication_marker(ledger_out, id, rows[0].attempts);
            let retained = (|| -> io::Result<()> {
                match std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&marker)
                {
                    Ok(mut file) => {
                        file.write_all(&bytes)?;
                        file.sync_all()?;
                    }
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                        if std::fs::read(&marker)? != bytes {
                            return Err(io::Error::other(
                                "unresolved publication identity changed",
                            ));
                        }
                        std::fs::File::open(&marker)?.sync_all()?;
                    }
                    Err(e) => return Err(e),
                }
                if let Some(parent) = marker.parent() {
                    std::fs::File::open(parent)?.sync_all()?;
                }
                Ok(())
            })();
            // Include the full row identity in the caller's retained command log
            // if the underlying filesystem also refuses the diagnostic write.
            Err(format!(
                "local fair ledger write to {uri} failed: {error}; unresolved marker {}: {retained:?}; rows={}",
                marker.display(),
                String::from_utf8_lossy(&bytes)
            ))
        }
    }
}

fn atomic_write(uri: &str, rows: &[LedgerRow]) -> Result<(), String> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let temporary = reserve_temporary(uri, std::process::id(), &NEXT)?;
    zenfleet_ledger::write_ledger_uri(&temporary, rows, None).map_err(|e| e.to_string())?;
    std::fs::File::open(&temporary)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())?;
    std::fs::rename(&temporary, uri).map_err(|e| e.to_string())?;
    let parent = Path::new(uri)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::File::open(parent)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())
}

fn reserve_temporary(
    uri: &str,
    pid: u32,
    next: &std::sync::atomic::AtomicU64,
) -> Result<String, String> {
    loop {
        let path = format!(
            "{uri}.partial-{pid}-{}",
            next.fetch_add(1, Ordering::Relaxed)
        );
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.to_string()),
        }
    }
}

fn archive_marker(ledger_out: &Path, id: &str, rows: &[LedgerRow]) -> Result<(), String> {
    let marker = publication_marker(ledger_out, id, rows[0].attempts);
    if marker.exists()
        && std::fs::read(&marker).map_err(|e| e.to_string())?
            != serde_json::to_vec(rows).map_err(|e| e.to_string())?
    {
        return Err(format!(
            "malformed or mismatched publication diagnostic requires reconciliation: {}",
            marker.display()
        ));
    }
    archive_marker_path(&marker)
}

fn archive_marker_path(marker: &Path) -> Result<(), String> {
    if !marker.exists() {
        return Ok(());
    }
    // Preserve diagnostics, but retire their admission veto only after their
    // exact row is committed or explicitly supplied by the native ledger.
    std::fs::File::open(marker)
        .and_then(|f| f.sync_all())
        .map_err(|e| e.to_string())?;
    let name = marker
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .replace(".unpublished-", ".reconciled-");
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let archived = marker.with_file_name(format!("{name}.{stamp}"));
    std::fs::rename(marker, archived).map_err(|e| e.to_string())?;
    if let Some(parent) = marker.parent() {
        std::fs::File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn publication_marker(ledger_out: &Path, id: &str, attempt: u32) -> PathBuf {
    let name = ledger_out.file_name().unwrap_or_default().to_string_lossy();
    ledger_out.with_file_name(format!("{name}.unpublished-{id}-{attempt}.json"))
}

pub(super) fn require_reconciled_publication(
    ledger_out: &Path,
    view: &LedgerView,
) -> Result<(), WorkerRunError> {
    let parent = ledger_out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.exists() {
        return Ok(());
    }
    let prefix = format!(
        "{}.unpublished-",
        ledger_out.file_name().unwrap_or_default().to_string_lossy()
    );
    for entry in std::fs::read_dir(parent).map_err(|e| WorkerRunError::Io(e.to_string()))? {
        let entry = entry.map_err(|e| WorkerRunError::Io(e.to_string()))?;
        if !entry.file_name().to_string_lossy().starts_with(&prefix) {
            continue;
        }
        let bytes =
            std::fs::read(entry.path()).map_err(|e| WorkerRunError::Ledger(e.to_string()))?;
        let rows: Vec<LedgerRow> =
            serde_json::from_slice(&bytes).map_err(|e| WorkerRunError::Ledger(e.to_string()))?;
        if rows.is_empty() || rows.iter().any(|row| view.get(&row.job_id) != Some(row)) {
            return Err(WorkerRunError::Ledger(format!(
                "unresolved local publication requires native reconciliation before producer restart: {}",
                entry.path().display()
            )));
        }
        archive_marker_path(&entry.path()).map_err(WorkerRunError::Ledger)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn long_job_does_not_hold_a_chunk_barrier_and_done_rows_resume_exactly() {
        let store = LocalBlobStore::new(super::super::tests::tmp()).unwrap();
        let jobs: Vec<_> = (0..12).map(super::super::tests::cheap_cell).collect();
        let peak = AtomicUsize::new(0);
        let active = AtomicUsize::new(0);
        let first = jobs[0].job_id();
        let flushed_small = AtomicUsize::new(0);
        let handler = |job: &DesiredJob| {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            if job.job_id() == first {
                while flushed_small.load(Ordering::SeqCst) < 11 {
                    std::thread::yield_now();
                }
            }
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(job.job_id().as_str().as_bytes().to_vec())
        };
        let params = super::super::tests::test_params();
        let mut rows = Vec::new();
        let mut first_was_last = false;
        let out = execute(
            &jobs,
            &LedgerView::new(),
            RetryPolicy::default(),
            &handler,
            &store,
            &params,
            &mut |_, r: &[LedgerRow]| {
                assert_eq!(r.len(), 1);
                rows.extend_from_slice(r);
                if r[0].job_id == first {
                    first_was_last = rows.len() == jobs.len();
                } else {
                    flushed_small.fetch_add(1, Ordering::SeqCst);
                }
                Ok(())
            },
            WorkerCtx {
                worker: "invented",
                provider: "local",
                now: 100,
            },
        )
        .unwrap();
        assert_eq!(out.done, 12);
        assert!(first_was_last);
        assert!(peak.load(Ordering::SeqCst) > 1);
        let mut view = LedgerView::new();
        for row in rows {
            view.apply(row);
        }
        let resumed = execute(
            &jobs,
            &view,
            RetryPolicy::default(),
            &|_| panic!("completed job re-executed"),
            &store,
            &params,
            &mut |_, _| panic!("completed job reflushed"),
            WorkerCtx {
                worker: "invented",
                provider: "local",
                now: 101,
            },
        )
        .unwrap();
        assert_eq!(resumed.done, 0);
    }

    #[test]
    fn publication_retries_the_row_without_repeating_the_producer() {
        let store = LocalBlobStore::new(super::super::tests::tmp()).unwrap();
        let jobs = vec![super::super::tests::cheap_cell(0)];
        let calls = AtomicUsize::new(0);
        let mut publications = 0;
        let out = execute(
            &jobs,
            &LedgerView::new(),
            RetryPolicy::default(),
            &|_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(vec![1])
            },
            &store,
            &super::super::tests::test_params(),
            &mut |_, _| {
                publications += 1;
                if publications < 3 {
                    Err("invented transient publication failure".into())
                } else {
                    Ok(())
                }
            },
            WorkerCtx {
                worker: "invented",
                provider: "local",
                now: 100,
            },
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(publications, 3);
        assert_eq!(out.done, 1);
    }

    #[test]
    fn unrecoverable_publication_closes_admission_before_another_producer() {
        let store = LocalBlobStore::new(super::super::tests::tmp()).unwrap();
        let jobs: Vec<_> = (0..8).map(super::super::tests::cheap_cell).collect();
        let calls = AtomicUsize::new(0);
        let mut params = super::super::tests::test_params();
        params.budget.cores = 1;
        let result = execute(
            &jobs,
            &LedgerView::new(),
            RetryPolicy::default(),
            &|_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(vec![1])
            },
            &store,
            &params,
            &mut |_, _| Err("invented persistent publication failure".into()),
            WorkerCtx {
                worker: "invented",
                provider: "local",
                now: 100,
            },
        );
        assert!(matches!(result, Err(WorkerRunError::Ledger(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn oversized_hint_is_a_terminal_configuration_error() {
        let store = LocalBlobStore::new(super::super::tests::tmp()).unwrap();
        let params = super::super::tests::test_params();
        let mut job = super::super::tests::cheap_cell(0);
        job.hint = Some(ResourceHint {
            peak_mem_bytes: params.budget.ram_budget_bytes + 1,
            threads: 1,
            vram_bytes: None,
        });
        let result = execute(
            &[job],
            &LedgerView::new(),
            RetryPolicy::default(),
            &|_| panic!("oversized job executed"),
            &store,
            &params,
            &mut |_, _| panic!("oversized job published"),
            WorkerCtx {
                worker: "invented",
                provider: "local",
                now: 100,
            },
        );
        assert!(matches!(result, Err(WorkerRunError::Config(_))));
    }

    #[test]
    fn full_sidecar_names_and_unpublished_rows_require_explicit_reconciliation() {
        let root = super::super::tests::tmp();
        std::fs::create_dir_all(&root).unwrap();
        let ledger = root.join("ledger.parquet");
        let id = format!("chunk-{}", "a".repeat(64));
        let other = format!("chunk-{}{}", "a".repeat(8), "b".repeat(56));
        assert_ne!(
            ledger_uri(&ledger.to_string_lossy(), &id),
            ledger_uri(&ledger.to_string_lossy(), &other)
        );
        // A directory at the sidecar path forces the real Parquet writer to
        // fail while its adjacent retained diagnostic remains writable.
        std::fs::create_dir(ledger_uri(&ledger.to_string_lossy(), &id)).unwrap();
        let job = super::super::tests::cheap_cell(0);
        let row = LedgerRow {
            job_id: job.job_id(),
            kind: job.kind,
            cell: job.cell,
            output_sha: Some(sha256(&[1])),
            status: JobStatus::Done,
            error_class: None,
            attempts: 1,
            ts: 100,
            worker: "invented".into(),
            provider: "local".into(),
        };
        assert!(flush_checked(&ledger, &id, std::slice::from_ref(&row)).is_err());
        assert!(require_reconciled_publication(&ledger, &LedgerView::new()).is_err());
        let view = LedgerView::from_rows([row]);
        require_reconciled_publication(&ledger, &view).unwrap();
        // The diagnostic is retained outside the veto namespace, so later
        // legitimate native state progression cannot resurrect its veto.
        assert!(!publication_marker(&ledger, &id, 1).exists());
        require_reconciled_publication(&ledger, &LedgerView::new()).unwrap();
        assert!(
            std::fs::read_dir(&root)
                .unwrap()
                .filter_map(Result::ok)
                .any(|e| e.file_name().to_string_lossy().contains(".reconciled-"))
        );
    }

    #[test]
    fn fresh_partial_reservation_preserves_a_previous_process_diagnostic() {
        let root = super::super::tests::tmp();
        std::fs::create_dir_all(&root).unwrap();
        let uri = root.join("ledger.parquet").to_string_lossy().into_owned();
        let prior = format!("{uri}.partial-1234-0");
        std::fs::write(&prior, b"retained diagnostic").unwrap();
        let next = std::sync::atomic::AtomicU64::new(0);
        let fresh = reserve_temporary(&uri, 1234, &next).unwrap();
        assert_eq!(fresh, format!("{uri}.partial-1234-1"));
        assert_eq!(std::fs::read(prior).unwrap(), b"retained diagnostic");
        assert!(std::fs::read(fresh).unwrap().is_empty());
    }
}
