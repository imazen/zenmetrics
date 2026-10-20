# DSSIM benchmark dependency identity

Verified 2026-10-20 against repository source `c10bdade` and the three frozen
canonical build snapshots. The selected dssim-core 3.5.1 dependency is
`331b56cf8998d89c3c89444880f8a0f51f046fb8`, the exact head of
[DSSIM PR200](https://github.com/kornelski/dssim/pull/200). It includes the
preceding SIMD, fused input preparation and streamed-moments changes.
The temporary request to select PR199 was corrected before any replacement
worker was built or deployed; the committed dependency pin remains unchanged.

| Host | Frozen canonical worker SHA256 | Saved successful DSSIM cells at inspection |
|---|---|---:|
| i265 | `5169daef303c5592a8a183c54b24bcb2f4ae0a9ec198edb91324af607a6c7a1e` | 32 of 48 |
| mac | `537e46df048b6be5121a84c58265db9f5975d889dc78d8853e6a13eba9937df1` | 24 of 48 |
| r7900x | `9375bebc7c684fa5ad1eec3682ff73946a90e9d2ae3e2f16217001020724dbc8` | 40 of 48 |

These are the observed partial frontiers at 2026-10-20 00:41 UTC, not a full
completion or qualification claim. Identity was checked from each frozen
`build.json`, its snapshot `Cargo.lock`, and the saved per-attempt result's
`impl_binary_sha`. Progress reports may lag individual saved attempts.

The existing canonical cohort covers six non-8K geometries, scalar mode,
one/eight workers, three fresh-process constructor-inclusive cold repeats
and ten warm rounds. Existing matching PR200 rows should remain in that
cohort; no PR199 replacement or extra DSSIM rerun is needed solely to select
PR200. Earlier unrelated binaries retain their own provenance.

The adapter in `crates/zenmetrics-api/src/cpu_dispatch.rs` uses the ordinary
`create_image_rgb` constructor. Thus these are PR200's default compact/streaming
images, not an additional benchmark of its opt-in cached-reference policy.
Benchmark process warming is distinct from caching DSSIM reference statistics.
