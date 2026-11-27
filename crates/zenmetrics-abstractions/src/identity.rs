//! Metric identity and provenance, unified on [`zenpsy_api`].
//!
//! [`Backend`], [`Implementation`], [`MetricId`], and [`implementation!`] are
//! re-exported directly from [`zenpsy_api`]. Stable hashing and column-naming
//! helpers ([`MetricIdExt::fingerprint`], [`MetricIdExt::semantic_column`],
//! [`MetricIdExt::column`], [`MetricIdExt::canonical`], [`BackendExt::token`])
//! live here as extension traits preserving the exact `zenmetrics-id-v1`
//! FNV-1a fingerprints and column names.

use alloc::string::{String, ToString};
use core::fmt::Write as _;

pub use zenpsy_api::{Backend, Implementation, MetricId, implementation};

const FP_DOMAIN: &str = "zenmetrics-id-v1";

/// Column-token formatting for [`Backend`].
pub trait BackendExt {
    /// Column token: `cpu`, `gpu_cuda`, …
    fn token(&self) -> String;
}

impl BackendExt for Backend {
    fn token(&self) -> String {
        match self {
            Backend::Cpu => "cpu".into(),
            Backend::Gpu { api, .. } => {
                let mut s = String::from("gpu_");
                push_sanitized(&mut s, api);
                s
            }
            _ => "unknown".into(),
        }
    }
}

/// Stable hashing, column naming, and canonical provenance rendering for
/// [`MetricId`].
pub trait MetricIdExt {
    /// 64-bit FNV-1a over every identity field except the crate version
    /// and build id, each field length-prefixed so no two field lists
    /// collide by concatenation. Stable across platforms and releases.
    fn fingerprint(&self) -> u64;

    /// `{family}[_{variant}]_{key}`, sanitized to `[a-z0-9_]`.
    fn semantic_column(&self, key: &str) -> String;

    /// The fully qualified column for output `key`:
    /// `{semantic}__{crate}_r{revision}_{backend}_{fp8}`.
    fn column(&self, key: &str) -> String;

    /// Every field, provenance included, as one `;`-separated
    /// `key=value` line (`\`, `;` and `=` escaped with `\`). Store it next
    /// to persisted scores; it reconstructs what produced them.
    fn canonical(&self) -> String;
}

impl MetricIdExt for MetricId {
    fn fingerprint(&self) -> u64 {
        let imp = self.implementation();
        let mut h = Fnv64::new();
        h.field(FP_DOMAIN);
        h.field(self.family());
        h.field(self.variant().unwrap_or(""));
        h.field(self.reference());
        h.field(imp.crate_name());
        h.write(&imp.score_revision().to_le_bytes());
        h.field(&imp.backend().token());
        h.write(&(self.config().len() as u64).to_le_bytes());
        for (k, v) in self.config() {
            h.field(k);
            h.field(v);
        }
        h.finish()
    }

    fn semantic_column(&self, key: &str) -> String {
        let mut s = String::new();
        push_sanitized(&mut s, self.family());
        if let Some(v) = self.variant() {
            s.push('_');
            push_sanitized(&mut s, v);
        }
        s.push('_');
        push_sanitized(&mut s, key);
        s
    }

    fn column(&self, key: &str) -> String {
        let imp = self.implementation();
        let mut s = self.semantic_column(key);
        s.push_str("__");
        push_sanitized(&mut s, imp.crate_name());
        let _ = write!(
            s,
            "_r{}_{}_{:08x}",
            imp.score_revision(),
            imp.backend().token(),
            self.fingerprint() >> 32
        );
        s
    }

    fn canonical(&self) -> String {
        let mut s = String::new();
        let mut kv = |k: &str, v: &str| {
            if !s.is_empty() {
                s.push(';');
            }
            push_escaped(&mut s, k);
            s.push('=');
            push_escaped(&mut s, v);
        };
        let imp = self.implementation();
        kv("family", self.family());
        kv("variant", self.variant().unwrap_or(""));
        kv("reference", self.reference());
        let mut crate_id = String::from(imp.crate_name());
        crate_id.push('@');
        crate_id.push_str(imp.crate_version());
        kv("crate", &crate_id);
        kv("rev", &imp.score_revision().to_string());
        kv("backend", &imp.backend().token());
        kv("build", imp.build().unwrap_or(""));
        let mut fp = String::new();
        let _ = write!(fp, "{:016x}", self.fingerprint());
        kv("fingerprint", &fp);
        for (k, v) in self.config() {
            let mut key = String::from("cfg.");
            key.push_str(k);
            kv(&key, v);
        }
        s
    }
}

/// Lowercase ASCII alphanumerics; every other run of characters becomes
/// one `_`; no leading or trailing `_` from the run rule.
fn push_sanitized(out: &mut String, s: &str) {
    let start = out.len();
    let mut pending = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            if pending && out.len() > start {
                out.push('_');
            }
            pending = false;
            out.push(c.to_ascii_lowercase());
        } else {
            pending = true;
        }
    }
}

/// Whether `s` is already a sanitized column component or full column:
/// non-empty, `[a-z0-9_]` only.
pub fn is_sanitized_column(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

fn push_escaped(out: &mut String, s: &str) {
    for c in s.chars() {
        if matches!(c, '\\' | ';' | '=') {
            out.push('\\');
        }
        out.push(c);
    }
}

struct Fnv64(u64);

impl Fnv64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn field(&mut self, s: &str) {
        self.write(&(s.len() as u64).to_le_bytes());
        self.write(s.as_bytes());
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> MetricId {
        MetricId::new("cvvdp", Implementation::new("cvvdp", "0.1.0", 1))
            .with_reference("pycvvdp 0.5.7")
    }

    #[test]
    fn fnv_known_vector() {
        let mut h = Fnv64::new();
        h.write(b"a");
        assert_eq!(h.finish(), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn version_and_build_are_provenance_not_identity() {
        let a = base();
        let b = MetricId::new(
            "cvvdp",
            Implementation::new("cvvdp", "0.2.0", 1).with_build(Some("deadbeef")),
        )
        .with_reference("pycvvdp 0.5.7");
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.column("jod"), b.column("jod"));
        assert_ne!(a.canonical(), b.canonical());
    }

    #[test]
    fn revision_backend_config_variant_change_the_column() {
        let a = base();
        let rev = MetricId::new("cvvdp", Implementation::new("cvvdp", "0.1.0", 2))
            .with_reference("pycvvdp 0.5.7");
        let gpu = base().with_backend(Backend::gpu("cuda"));
        let cfg = base().with_config("ppd", 60.0f32);
        let var = base().with_variant("hdr");
        let cols = [&a, &rev, &gpu, &cfg, &var].map(|m| m.column("jod"));
        for i in 0..cols.len() {
            for j in i + 1..cols.len() {
                assert_ne!(cols[i], cols[j]);
            }
        }
        assert!(gpu.column("jod").contains("_gpu_cuda_"));
        assert!(rev.column("jod").contains("_r2_"));
    }

    #[test]
    fn config_order_does_not_matter_and_last_write_wins() {
        let a = base().with_config("b", 2).with_config("a", 1);
        let b = base()
            .with_config("a", 9)
            .with_config("b", 2)
            .with_config("a", 1);
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.config_value("a"), Some("1"));
        assert_eq!(a.config()[0].0, "a");
    }

    #[test]
    fn field_boundaries_cannot_collide() {
        let a = MetricId::new("ab", Implementation::new("c", "0", 1));
        let b = MetricId::new("a", Implementation::new("bc", "0", 1));
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn column_shapes() {
        let m =
            MetricId::new("mad-iqa", Implementation::new("mad-iqa", "0.1.0", 1)).with_variant("V3");
        assert_eq!(m.semantic_column("Hi Index"), "mad_iqa_v3_hi_index");
        let col = m.column("hi");
        assert!(col.starts_with("mad_iqa_v3_hi__mad_iqa_r1_cpu_"), "{col}");
        assert_eq!(col.len(), "mad_iqa_v3_hi__mad_iqa_r1_cpu_".len() + 8);
        assert!(is_sanitized_column(&col));
    }

    #[test]
    fn canonical_escapes_and_lists_config() {
        let m = base().with_config("eotf", "a;b=c");
        let c = m.canonical();
        assert!(
            c.starts_with(
                "family=cvvdp;variant=;reference=pycvvdp 0.5.7;crate=cvvdp@0.1.0;rev=1;backend=cpu;build=;fingerprint="
            ),
            "{c}"
        );
        assert!(c.ends_with(r"cfg.eotf=a\;b\=c"), "{c}");
    }

    #[test]
    fn macro_reads_cargo() {
        let imp = crate::implementation!(4);
        assert_eq!(imp.crate_name(), "zenmetrics-abstractions");
        assert_eq!(imp.crate_version(), env!("CARGO_PKG_VERSION"));
        assert_eq!(imp.score_revision(), 4);
        assert_eq!(imp.backend(), Backend::Cpu);
    }
}
