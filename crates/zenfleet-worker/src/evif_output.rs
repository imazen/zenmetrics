//! EVIF job output: persist the native bitstream before publishing its container.
//! Wire: magic, two little-endian u64 lengths, EVIF bytes, native bitstream bytes.
use super::{BlobStore, DesiredJob, Sha256Hex};
use std::io;
const MAGIC: &[u8] = b"ZFEVIF1\n";
const LIMIT: usize = 32 * 1024 * 1024;
fn split(bytes: &[u8]) -> io::Result<(&[u8], &[u8])> {
    let bad = || io::Error::new(io::ErrorKind::InvalidData, "invalid EVIF output envelope");
    if !bytes.starts_with(MAGIC) || bytes.len() < 24 || bytes.len() > LIMIT {
        return Err(bad());
    }
    let a =
        usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap())).map_err(|_| bad())?;
    let b = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().unwrap()))
        .map_err(|_| bad())?;
    if a == 0 || b == 0 || a.checked_add(b).and_then(|n| n.checked_add(24)) != Some(bytes.len()) {
        return Err(bad());
    }
    Ok((&bytes[24..24 + a], &bytes[24 + a..]))
}
pub(super) fn persist(
    store: &dyn BlobStore,
    job: &DesiredJob,
    bytes: &[u8],
) -> io::Result<Sha256Hex> {
    if !job.cell.codec.starts_with("evif:") && !job.cell.codec.starts_with("hdr-chroma:") {
        return store.put(bytes);
    }
    if bytes.starts_with(super::staged_output::MAGIC) {
        return super::staged_output::persist(store, job, bytes);
    }
    let (container, bits) = split(bytes)?;
    // A DONE ledger row can only refer to a container after both objects land.
    store.put(bits)?;
    store.put(container)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::LocalBlobStore;
    use zenfleet_core::{CellId, JobKind, sha256};
    #[test]
    fn hdr_staging_persists_avif_and_pq16_before_the_receipt() {
        let dir = std::env::temp_dir().join(format!("hdr-output-{}", std::process::id()));
        let stage = dir.join("stage");
        std::fs::create_dir_all(&stage).unwrap();
        let store = LocalBlobStore::new(dir.join("store")).unwrap();
        let job = DesiredJob::new(
            JobKind::Bake {
                view: "hdr-chroma-view-v1".into(),
            },
            vec![],
            CellId {
                image_path: "view.pq.png".into(),
                codec: "hdr-chroma:avif-svt".into(),
                q: -1,
                knob_tuple_json: "{}".into(),
            },
        );
        std::fs::write(stage.join("sample.avif"), b"avif").unwrap();
        std::fs::write(stage.join("sample.pq.png"), b"pq16").unwrap();
        std::fs::write(
            stage.join("result.json"),
            br#"{"image_path":"view.pq.png","knob_tuple_json":"{}"}"#,
        )
        .unwrap();
        let envelope = serde_json::json!({"root":stage,"primary":"result.json","objects":[
            {"path":"sample.avif","sha256":sha256(b"avif").as_str()},
            {"path":"sample.pq.png","sha256":sha256(b"pq16").as_str()}]});
        let mut bytes = crate::staged_output::MAGIC.to_vec();
        bytes.extend(serde_json::to_vec(&envelope).unwrap());
        let output = persist(&store, &job, &bytes).unwrap();
        assert!(store.exists(&sha256(b"avif")));
        assert!(store.exists(&sha256(b"pq16")));
        assert!(store.exists(&output));
        assert!(!stage.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn both_artifacts_persist_and_truncation_fails() {
        let dir = std::env::temp_dir().join(format!("evif-output-{}", std::process::id()));
        let store = LocalBlobStore::new(&dir).unwrap();
        let job = DesiredJob::new(
            JobKind::Metric {
                metric: "ssim2-safesyn-b7c2b4b3".into(),
            },
            vec![],
            CellId {
                image_path: "r.png".into(),
                codec: "evif:zenjpeg-expert".into(),
                q: 95,
                knob_tuple_json: "{}".into(),
            },
        );
        let mut wire = MAGIC.to_vec();
        wire.extend(4u64.to_le_bytes());
        wire.extend(3u64.to_le_bytes());
        wire.extend(b"evifjpg");
        assert_eq!(persist(&store, &job, &wire).unwrap(), sha256(b"evif"));
        assert!(store.exists(&sha256(b"jpg")));
        assert!(persist(&store, &job, &wire[..wire.len() - 1]).is_err());
        let mut overflow = wire.clone();
        overflow[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(persist(&store, &job, &overflow).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
