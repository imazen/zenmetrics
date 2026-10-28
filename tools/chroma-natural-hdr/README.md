# Natural HDR SSIMULACRA2 probe

Owner ruling, 2026-10-28: candidate HDR teacher uses standard SSIMULACRA2
cube-root opsin. Native RGB16 PQ PNG is decoded, transformed to absolute
linear BT.709 using E26's unclipped common ingress, then divided by 100
(100 cd/m² maps to 1.0). Superwhite and out-of-gamut values are retained.
No PU21 path, tone mapping, peak rescaling, clipping or eight-bit conversion.

Metric: fast-ssim2=0.8.2, source b7c2b4b33f3a26b753caabd5264a7fef9c4f4545.
Ingress owner: zenmetrics 348bde5bf96d635383965d80d9adb8f9c195fd14,
`crates/zenmetrics-cli/src/hdr.rs` and `crates/zenmetrics-api/src/hdr.rs`;
PQ EOTF copied unchanged, linear primary conversion retains its pinned
zenpixels 1d3ddd2131b22e9e3754c765d4ce45ce3901de59 implementation.

The isolated lock keeps this graph separate from the frozen SDR teacher.
JSONL pairs enter stdin; finite score doubles and implementation receipts
leave stdout. `cargo test --manifest-path tools/chroma-natural-hdr/Cargo.toml`
checks reference white, superwhite, negative gamut values and PQ endpoints.
