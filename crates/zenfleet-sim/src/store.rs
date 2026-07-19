//! [`FaultStore`] — an in-memory object store that misbehaves the way R2/s5cmd
//! actually does.
//!
//! It implements [`zenfleet_cloud::BlobStorage`] so it drops straight into
//! [`zenfleet_cloud::run_worker`] and anything else written against the trait,
//! and it adds the two **strongly-consistent conditional** primitives R2 exposes
//! but the trait doesn't — [`FaultStore::put_if_absent`] (`If-None-Match: *`) and
//! [`FaultStore::cas`] (`If-Match: <etag>`) — which the atomic claim relies on.
//!
//! The store is a cheap shared handle: `clone()` shares the same objects, clock,
//! fault spec, RNG, and op counters, so every simulated worker races over one
//! substrate. It is single-threaded and deterministic — a "race" is a controlled
//! interleaving of calls, which is what makes it a reproducible test.
//!
//! ## Consistency model
//! A `put` records an object with `visible_at = now + consistency_delay_secs`.
//! Trait reads (`get`/`head`/`list`) only see an object once `visible_at <= now`
//! — read-after-write / list-after-write staleness. With
//! [`FaultSpec::read_your_writes`], a worker sees *its own* latest write
//! immediately but still not its peers' — the asymmetry that lets a token-race
//! claim double-acquire. The conditional ops ([`put_if_absent`](FaultStore::put_if_absent)
//! / [`cas`](FaultStore::cas)) evaluate their condition against *actual* existence
//! atomically (strongly consistent), which is exactly why an `If-None-Match`
//! claim needs no read-back and can't double-acquire.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use zenfleet_cloud::{ArtifactKey, BlobMeta, BlobStorage, CloudError};

use crate::clock::SimClock;
use crate::fault::FaultSpec;
use crate::rng::Rng;

/// Per-op counters — the "how much did the substrate cost / hurt" ledger a
/// scenario asserts against (e.g. "the flaky run still converged with N retries",
/// or "the tarball path issued 1 GET, not 10 000").
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpCounts {
    pub puts: u64,
    pub gets: u64,
    pub heads: u64,
    pub lists: u64,
    pub deletes: u64,
    /// Ops that failed with a transient (retryable) storage error.
    pub transient_errors: u64,
    /// Ops that failed with a credentials error (bad/expired creds).
    pub auth_errors: u64,
    /// `put`s that silently stored truncated bytes.
    pub partial_writes: u64,
    /// `delete`s that failed (couldn't clean up).
    pub delete_failures: u64,
}

struct Obj {
    bytes: Vec<u8>,
    etag: String,
    /// Which worker wrote this (`""` for anonymous trait puts). Drives
    /// read-your-writes visibility.
    writer: String,
    /// Clock time at/after which trait reads can see this object.
    visible_at: u64,
    /// The stored bytes are a truncated partial upload.
    corrupt: bool,
}

struct Inner {
    clock: SimClock,
    spec: RefCell<FaultSpec>,
    rng: RefCell<Rng>,
    objs: RefCell<HashMap<String, Obj>>,
    ops: RefCell<OpCounts>,
    etag_seq: RefCell<u64>,
}

/// A breakable in-memory [`BlobStorage`]. Clone to hand the same substrate to
/// several simulated workers.
#[derive(Clone)]
pub struct FaultStore {
    inner: Rc<Inner>,
}

impl FaultStore {
    /// A store sharing `clock`, running `spec`, seeded with `seed`.
    pub fn new(clock: SimClock, spec: FaultSpec, seed: u64) -> Self {
        Self {
            inner: Rc::new(Inner {
                clock,
                spec: RefCell::new(spec),
                rng: RefCell::new(Rng::new(seed)),
                objs: RefCell::new(HashMap::new()),
                ops: RefCell::new(OpCounts::default()),
                etag_seq: RefCell::new(0),
            }),
        }
    }

    /// Swap the fault schedule mid-scenario (e.g. "creds expire now", "the R2
    /// blip is over"). Returns the previous spec.
    pub fn set_spec(&self, spec: FaultSpec) -> FaultSpec {
        self.inner.spec.replace(spec)
    }

    /// A snapshot of the op counters.
    pub fn counts(&self) -> OpCounts {
        self.inner.ops.borrow().clone()
    }

    /// The shared clock.
    pub fn clock(&self) -> SimClock {
        self.inner.clock.clone()
    }

    fn next_etag(&self) -> String {
        let mut s = self.inner.etag_seq.borrow_mut();
        *s += 1;
        format!("etag-{s}")
    }

    /// Latency + credential + transient faults common to every op. Advances the
    /// clock, then fails the op if creds are bad/expired or a transient blip
    /// fires. Returns `Ok(())` if the op should proceed.
    fn precheck(&self) -> Result<(), CloudError> {
        let latency = self.inner.spec.borrow().op_latency_secs;
        self.inner.clock.advance(latency);
        let now = self.inner.clock.now();

        let (creds_invalid, creds_expire_at, transient_rate) = {
            let s = self.inner.spec.borrow();
            (s.creds_invalid, s.creds_expire_at, s.transient_rate)
        };
        if creds_invalid || (creds_expire_at != 0 && now >= creds_expire_at) {
            self.inner.ops.borrow_mut().auth_errors += 1;
            return Err(CloudError::Credentials(
                "403 forbidden (bad or expired scoped credentials)".into(),
            ));
        }
        if self.inner.rng.borrow_mut().chance(transient_rate) {
            self.inner.ops.borrow_mut().transient_errors += 1;
            return Err(CloudError::Storage("503 slow down (transient)".into()));
        }
        Ok(())
    }

    fn is_visible(&self, obj: &Obj, reader: Option<&str>) -> bool {
        if obj.visible_at <= self.inner.clock.now() {
            return true;
        }
        // Not globally visible yet — but a worker can read its own fresh write.
        if self.inner.spec.borrow().read_your_writes
            && let Some(w) = reader
            && obj.writer == w
        {
            return true;
        }
        false
    }

    fn insert(&self, key: &str, bytes: &[u8], writer: &str, allow_partial: bool) {
        let now = self.inner.clock.now();
        let delay = self.inner.spec.borrow().consistency_delay_secs;
        let partial = allow_partial && {
            let rate = self.inner.spec.borrow().partial_write_rate;
            self.inner.rng.borrow_mut().chance(rate)
        };
        let stored: Vec<u8> = if partial {
            self.inner.ops.borrow_mut().partial_writes += 1;
            bytes[..bytes.len() / 2].to_vec() // truncated "successful" upload
        } else {
            bytes.to_vec()
        };
        let etag = self.next_etag();
        self.inner.objs.borrow_mut().insert(
            key.to_string(),
            Obj {
                bytes: stored,
                etag,
                writer: writer.to_string(),
                visible_at: now.saturating_add(delay),
                corrupt: partial,
            },
        );
    }

    // ---- worker-aware reads (used by the claim layer for read-your-writes) ----

    /// `get` as a named worker (read-your-writes aware).
    pub fn get_as(&self, worker: &str, key: &str) -> Result<Vec<u8>, CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().gets += 1;
        let objs = self.inner.objs.borrow();
        match objs.get(key) {
            Some(o) if self.is_visible(o, Some(worker)) => Ok(o.bytes.clone()),
            _ => Err(CloudError::Storage(format!("no such key (or not yet visible): {key}"))),
        }
    }

    /// `head` as a named worker (read-your-writes aware). `None` = not present
    /// or not yet visible.
    pub fn head_as(&self, worker: &str, key: &str) -> Result<Option<BlobMeta>, CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().heads += 1;
        let objs = self.inner.objs.borrow();
        Ok(match objs.get(key) {
            Some(o) if self.is_visible(o, Some(worker)) => Some(BlobMeta {
                size: o.bytes.len() as u64,
                etag: Some(o.etag.clone()),
            }),
            _ => None,
        })
    }

    /// `put` as a named worker (records the writer for read-your-writes).
    pub fn put_as(&self, worker: &str, key: &str, bytes: &[u8]) -> Result<(), CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().puts += 1;
        self.insert(key, bytes, worker, true);
        Ok(())
    }

    // ---- strongly-consistent conditional ops (R2 If-None-Match / If-Match) ----

    /// Atomic create-if-absent — models `PUT If-None-Match: *`. Returns `Ok(true)`
    /// if THIS call created the object, `Ok(false)` if it already existed. The
    /// existence check is strongly consistent (actual presence, not visibility),
    /// so two racers can never both get `true`. This is the primitive that makes
    /// the conditional claim exactly-once with no read-back.
    pub fn put_if_absent(&self, worker: &str, key: &str, bytes: &[u8]) -> Result<bool, CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().puts += 1;
        if self.inner.objs.borrow().contains_key(key) {
            return Ok(false);
        }
        self.insert(key, bytes, worker, false);
        Ok(true)
    }

    /// Atomic compare-and-swap on the ETag — models `PUT If-Match: <etag>`.
    /// Returns `Ok(true)` if the current ETag matched and the write landed,
    /// `Ok(false)` otherwise. This is how a stale claim is safely stolen: two
    /// reclaimers racing on the same observed ETag can't both win.
    pub fn cas(
        &self,
        worker: &str,
        key: &str,
        expected_etag: &str,
        bytes: &[u8],
    ) -> Result<bool, CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().puts += 1;
        let matches = self
            .inner
            .objs
            .borrow()
            .get(key)
            .map(|o| o.etag == expected_etag)
            .unwrap_or(false);
        if matches {
            self.insert(key, bytes, worker, false);
        }
        Ok(matches)
    }

    /// Read bytes + current ETag (global visibility). `None` = absent / not yet
    /// visible. The ETag feeds a subsequent [`cas`](Self::cas) for a steal.
    pub fn get_with_etag(&self, key: &str) -> Result<Option<(Vec<u8>, String)>, CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().gets += 1;
        let objs = self.inner.objs.borrow();
        Ok(match objs.get(key) {
            Some(o) if self.is_visible(o, None) => Some((o.bytes.clone(), o.etag.clone())),
            _ => None,
        })
    }

    /// Whether the stored bytes for `key` are a truncated partial upload — a test
    /// hook so a scenario can assert content-addressing catches corruption.
    pub fn is_corrupt(&self, key: &str) -> bool {
        self.inner
            .objs
            .borrow()
            .get(key)
            .map(|o| o.corrupt)
            .unwrap_or(false)
    }
}

impl BlobStorage for FaultStore {
    fn put(&self, key: &ArtifactKey, bytes: &[u8]) -> Result<(), CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().puts += 1;
        self.insert(key.as_str(), bytes, "", true);
        Ok(())
    }

    fn get(&self, key: &ArtifactKey) -> Result<Vec<u8>, CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().gets += 1;
        let objs = self.inner.objs.borrow();
        match objs.get(key.as_str()) {
            Some(o) if self.is_visible(o, None) => Ok(o.bytes.clone()),
            _ => Err(CloudError::Storage(format!("missing {key}"))),
        }
    }

    fn head(&self, key: &ArtifactKey) -> Result<Option<BlobMeta>, CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().heads += 1;
        let objs = self.inner.objs.borrow();
        Ok(match objs.get(key.as_str()) {
            Some(o) if self.is_visible(o, None) => Some(BlobMeta {
                size: o.bytes.len() as u64,
                etag: Some(o.etag.clone()),
            }),
            _ => None,
        })
    }

    fn list(&self, prefix: &str) -> Result<Vec<ArtifactKey>, CloudError> {
        self.precheck()?;
        self.inner.ops.borrow_mut().lists += 1;
        let drop_rate = self.inner.spec.borrow().list_drop_rate;
        let objs = self.inner.objs.borrow();
        let mut out = Vec::new();
        for (k, o) in objs.iter() {
            if !k.starts_with(prefix) || !self.is_visible(o, None) {
                continue;
            }
            // List is the weakest-consistency op — a visible key may still be
            // omitted from any single LIST.
            if self.inner.rng.borrow_mut().chance(drop_rate) {
                continue;
            }
            out.push(ArtifactKey(k.clone()));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    fn delete(&self, key: &ArtifactKey) -> Result<(), CloudError> {
        self.precheck()?;
        let fail_rate = self.inner.spec.borrow().delete_fail_rate;
        if self.inner.rng.borrow_mut().chance(fail_rate) {
            self.inner.ops.borrow_mut().delete_failures += 1;
            return Err(CloudError::Storage(format!(
                "delete failed (object stuck): {key}"
            )));
        }
        self.inner.ops.borrow_mut().deletes += 1;
        self.inner.objs.borrow_mut().remove(key.as_str());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_store_round_trips() {
        let clock = SimClock::new(0);
        let s = FaultStore::new(clock, FaultSpec::perfect(), 1);
        s.put(&ArtifactKey("k".into()), b"hello").unwrap();
        assert_eq!(s.get(&ArtifactKey("k".into())).unwrap(), b"hello");
        assert!(s.head(&ArtifactKey("k".into())).unwrap().is_some());
        assert_eq!(s.list("").unwrap().len(), 1);
        s.delete(&ArtifactKey("k".into())).unwrap();
        assert!(s.head(&ArtifactKey("k".into())).unwrap().is_none());
    }

    #[test]
    fn eventual_consistency_hides_a_fresh_put() {
        let clock = SimClock::new(100);
        let s = FaultStore::new(clock.clone(), FaultSpec::eventual_consistency(3), 1);
        // Anonymous put → not even read-your-writes helps.
        s.put(&ArtifactKey("late".into()), b"x").unwrap();
        assert!(
            s.head(&ArtifactKey("late".into())).unwrap().is_none(),
            "just-written object is not yet visible (read-after-write window)"
        );
        clock.advance(3);
        assert!(
            s.head(&ArtifactKey("late".into())).unwrap().is_some(),
            "visible after the consistency delay elapses"
        );
    }

    #[test]
    fn put_if_absent_is_atomic() {
        let s = FaultStore::new(SimClock::new(0), FaultSpec::eventual_consistency(10), 1);
        assert!(s.put_if_absent("w1", "claim", b"a").unwrap(), "first wins");
        assert!(
            !s.put_if_absent("w2", "claim", b"b").unwrap(),
            "second loses — strongly consistent even under a 10s consistency delay"
        );
    }

    #[test]
    fn bad_credentials_fail_every_op() {
        let s = FaultStore::new(SimClock::new(0), FaultSpec::bad_credentials(), 1);
        assert!(matches!(
            s.put(&ArtifactKey("k".into()), b"x"),
            Err(CloudError::Credentials(_))
        ));
        assert_eq!(s.counts().auth_errors, 1);
    }
}
