//! Content addressing: every blob (source image, encoded variant, diffmap) is named by the
//! lowercase-hex SHA-256 of its bytes. This is the keystone the rest of the system rests on —
//! idempotent enqueue (goal A), don't-lose-data GC (goal G), and discoverability (goal I) all key
//! off the hash. Identical bytes → identical key → free dedup.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Lowercase-hex SHA-256 digest (64 chars). Storage key + dedup key + GC handle.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Sha256Hex(String);

impl Sha256Hex {
    /// Validate and wrap an existing 64-char lowercase-hex string.
    pub fn parse(s: impl Into<String>) -> Result<Self, ContentError> {
        let s = s.into();
        if s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            Ok(Self(s))
        } else {
            Err(ContentError::BadDigest(s))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Sha256Hex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::fmt::Debug for Sha256Hex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sha256Hex({})", &self.0)
    }
}

/// Hash bytes into a content address.
pub fn sha256(bytes: &[u8]) -> Sha256Hex {
    let mut h = Sha256::new();
    h.update(bytes);
    Sha256Hex(hex::encode(h.finalize()))
}

/// Canonical object-store key for a blob: `blobs/<sha256>`. Listing is never needed (a Parquet
/// blob-index is the inventory), so a flat keyspace is fine.
pub fn blob_key(sha: &Sha256Hex) -> String {
    format!("blobs/{sha}")
}

/// A reference to a stored blob: its content hash and byte length.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    pub sha: Sha256Hex,
    pub len: u64,
}

impl BlobRef {
    /// Hash `bytes` and record their length.
    pub fn of(bytes: &[u8]) -> Self {
        Self {
            sha: sha256(bytes),
            len: bytes.len() as u64,
        }
    }

    pub fn key(&self) -> String {
        blob_key(&self.sha)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ContentError {
    #[error("not a 64-char lowercase-hex sha256: {0:?}")]
    BadDigest(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // NIST SHA-256 test vectors.
        assert_eq!(
            sha256(b"").as_str(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc").as_str(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn dedup_is_content_based() {
        assert_eq!(sha256(b"same bytes"), sha256(b"same bytes"));
        assert_ne!(sha256(b"a"), sha256(b"b"));
    }

    #[test]
    fn blob_key_format() {
        let s = sha256(b"abc");
        assert_eq!(blob_key(&s), format!("blobs/{s}"));
        assert!(blob_key(&s).starts_with("blobs/"));
        assert_eq!(blob_key(&s).len(), "blobs/".len() + 64);
    }

    #[test]
    fn parse_validates() {
        assert!(Sha256Hex::parse(sha256(b"x").as_str().to_string()).is_ok());
        assert!(Sha256Hex::parse("xyz").is_err()); // too short
        assert!(Sha256Hex::parse("A".repeat(64)).is_err()); // uppercase rejected
        assert!(Sha256Hex::parse("g".repeat(64)).is_err()); // non-hex rejected
    }

    #[test]
    fn blobref_roundtrips_and_measures_len() {
        let b = BlobRef::of(b"hello");
        assert_eq!(b.len, 5);
        assert_eq!(b.key(), blob_key(&b.sha));
        let j = serde_json::to_string(&b).unwrap();
        assert_eq!(serde_json::from_str::<BlobRef>(&j).unwrap(), b);
    }
}
