//! Live verification of safe GC against R2 (goal G). Puts a referenced blob, two cheap-regenerable
//! blobs (one old, one new), and an irreplaceable orphan; then runs `gc_execute` with a cap that fits
//! only the newest cheap blob and asserts:
//!   - the referenced blob is KEPT (never evicted),
//!   - the LRU tail (oldest cheap) is EVICTED with a tombstone,
//!   - the newest cheap blob is KEPT (fits the cap),
//!   - the unreferenced irreplaceable blob is REFUSED (surfaced, never deleted).
//!
//! Needs R2_ACCOUNT_ID / R2_BUCKET + AWS_* creds in env. Cleans up its prefix.

use std::collections::HashSet;

use zen_job_core::{BlobIndexEntry, Regenerability, Sha256Hex};
use zen_jobworker::{BlobStore, GcExecCfg, R2BlobStore, gc_execute};

fn main() {
    let acct = std::env::var("R2_ACCOUNT_ID").expect("R2_ACCOUNT_ID");
    let bucket = std::env::var("R2_BUCKET").expect("R2_BUCKET");
    let ep = format!("https://{acct}.r2.cloudflarestorage.com");
    let pfx = format!("jobsys-gc-{}", std::process::id());
    let store = R2BlobStore::new(ep.clone(), bucket.clone(), format!("{pfx}/blobs"));

    // distinct contents (same 24-byte length for the two LRU candidates → clean cap math)
    let referenced = store.put(b"REFERENCED-cheap-blob-01").expect("put ref");
    let cheap_old = store.put(b"cheap-OLD-lru-tail-aaaaaa").expect("put old");
    let cheap_new = store.put(b"cheap-NEW-keep-me-bbbbbbb").expect("put new");
    let irreplaceable = store.put(b"IRREPLACEABLE-source-only").expect("put irr");
    let csize = 24u64;

    let entry = |sha: &Sha256Hex, regen, last_ref: u64| BlobIndexEntry {
        sha: sha.clone(),
        size: csize,
        regenerability: regen,
        last_ref_secs: last_ref,
    };
    let index = vec![
        entry(&referenced, Regenerability::CheapRegenerable, 100),
        entry(&cheap_old, Regenerability::CheapRegenerable, 10),
        entry(&cheap_new, Regenerability::CheapRegenerable, 90),
        entry(&irreplaceable, Regenerability::NotRegenerable, 5),
    ];
    let refset: HashSet<Sha256Hex> = [referenced.clone()].into_iter().collect();
    let roots = HashSet::new();

    let mut cfg = GcExecCfg {
        endpoint: &ep,
        blobs_base_uri: &format!("s3://{bucket}/{pfx}/blobs"),
        tombstones_base_uri: Some(&format!("s3://{bucket}/{pfx}/tombstones")),
        cheap_cap_bytes: csize, // fits exactly one cheap blob → evict the LRU tail
        now: 1000,
        execute: false,
    };

    // dry-run
    let dry = gc_execute(&index, &refset, &roots, &cfg);
    println!(
        "[dry-run] kept={} would_evict={:?} refused={}",
        dry.kept,
        dry.lru_evicted,
        dry.refused.len()
    );
    assert_eq!(
        dry.lru_evicted,
        vec![cheap_old.to_string()],
        "dry-run: only the LRU tail"
    );
    assert_eq!(
        dry.refused.len(),
        1,
        "dry-run: irreplaceable orphan surfaced"
    );

    // execute
    cfg.execute = true;
    let cleanup = |store: &R2BlobStore, bucket: &str, ep: &str, pfx: &str| {
        let _ = std::process::Command::new("s5cmd")
            .args([
                "--endpoint-url",
                ep,
                "rm",
                &format!("s3://{bucket}/{pfx}/*"),
            ])
            .status();
        let _ = store; // (store kept for type)
    };
    let rep = gc_execute(&index, &refset, &roots, &cfg);
    println!(
        "[execute] kept={} evicted={:?} freed={}B tombstones={} refused={} errors={:?}",
        rep.kept,
        rep.lru_evicted,
        rep.freed_bytes,
        rep.tombstones_written,
        rep.refused.len(),
        rep.errors
    );

    // verify R2 state
    let old_gone = !store.exists(&cheap_old);
    let new_kept = store.exists(&cheap_new);
    let ref_kept = store.exists(&referenced);
    let irr_kept = store.exists(&irreplaceable);
    println!(
        "verify: old_evicted={old_gone} new_kept={new_kept} referenced_kept={ref_kept} irreplaceable_refused={irr_kept}"
    );

    let ok = old_gone
        && new_kept
        && ref_kept
        && irr_kept
        && rep.lru_evicted == vec![cheap_old.to_string()]
        && rep.tombstones_written == 1
        && rep.errors.is_empty();
    cleanup(&store, &bucket, &ep, &pfx);
    assert!(ok, "GC safety guarantees violated");
    println!("### GC LIVE PASS — LRU tail evicted + tombstoned; referenced + irreplaceable kept.");
}
