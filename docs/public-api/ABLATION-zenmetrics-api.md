# ABLATION-zenmetrics-api.md
<!-- date: 2026-06-11 | snapshot-commit: 2524d81f | grep-template: ugrep -r --include="*.rs" SYMBOL /home/lilith/work --exclude-dir=target --exclude-dir=.jj --exclude-dir=zenmetrics -->

## Scope
Snapshot: `docs/public-api/zenmetrics-api.txt` — 464 items (default), 514 items (all-features, adds `context::MetricContext<R>` + `context::PairHandles` from `cubecl-types` feature).

This is the umbrella crate. All 9 metric crates are optional deps; the pub surface is additive per feature.

Known consumers: `zenmetrics-cli` (primary driver — uses nearly the entire surface), `zenmetrics-orchestrator` (session/batch API), `jxl-encoder` (uses via `zenmetrics-api::iwssim::IwssimParams::allow_small` path documented in tests).

---

## Analysis notes

The `hdr` module (HDR surface: `HdrFeeding`, `HdrTransfer`, `HdrScorer`, `DisplayModel`, `pu21_encode`, `pq_eotf`, `srgb_eotf`, etc.) is the June 2026 PU program API. Per mission instructions: **KEEP wholesale** — recent deliberate design. The free functions (`pu21_encode`, `pu21_decode`, `pq_eotf`, `pq_inverse_eotf`, `srgb_eotf`, `hlg_inverse_oetf`, `hlg_system_gamma`, `nits_interleaved_to_pu_luma_gray`, `hdr_feeding`) are used by `zenmetrics-cli/src/hdr.rs` (confirmed `pu21_encode` call at line 397). KEEP.

The `context` module (`MetricContext<R>`, `PairHandles`) is a `cubecl-types`-gated low-level typed upload-once API used by `zenmetrics-api/tests/it/compute_handles.rs` and `zenmetrics-api/src/metric.rs:1510` (`pair.ref_handle`, `pair.dist_handle`). The pub fields (`client`, `width`, `height` on `MetricContext`; `ref_handle`, `dist_handle`, `generation` on `PairHandles`) serve the batch-scoring pattern. KEEP.

---

## Summary

| Class | Count | % of 464 |
|-------|-------|---------|
| Total items (default) | 464 | 100 % |
| Flagged (A) | 0 | 0 % |
| Flagged (B) | 1 | < 1 % |
| KEEP (used or deliberate) | 463 | >99 % |

---

## Flagged items

### B — `pub(crate)` / remove candidates

| Item | Reason |
|------|--------|
| `zenmetrics_api::pixels::*` (the entire `pixels` module if present in default features) | Needs verification: check whether `zenmetrics_api::pixels` re-exports `zenpixels` types or defines its own. If it re-exports a wrapper of wrapper types with no direct external consumer, candidate B. **NOTE: not confirmed — further source check required before acting.** Leave KEEP pending that check. (This item is listed as a caution, not a confirmed flag.) |

*No items confirmed flagged in this crate with sufficient evidence. The surface is large but well-consumed.*

---

## Workspace-wide top-10 digest (from all 9 crates)

This digest summarizes the highest-confidence findings across the workspace. Ordered by confidence and impact.

| Rank | Item | Crate | Class | Evidence |
|------|------|-------|-------|---------|
| 1 | `auto_strip_body_for(…)` (per-crate memory_mode) | ssim2-gpu, zensim-gpu, iwssim-gpu, dssim-gpu, butteraugli-gpu, cvvdp-gpu | B×6 | Zero cross-crate consumers in org grep; internal to each crate's `memory_mode.rs`. Consistent pattern across all 6 GPU crates. |
| 2 | `cvvdp::diffmap` (empty-pub module) | cvvdp | B | Module body is entirely `pub(crate)`; snapshot shows the module name but zero child items. Zero external refs. |
| 3 | `cvvdp::kernels::csf::csf_lut_v0_5_4` (sub-module) | cvvdp | B | LUT constants re-exported at the parent `csf::` level; the sub-module path has zero external consumer hits. |
| 4 | `iwssim::IwssimParams` fields `bl_sz_x`, `bl_sz_y`, `sigma_nsq` | iwssim | A | Never set via struct literal outside crate; only `allow_small` and `iw_flag` have external use. Three algorithmic tunables with no documented stable contract. |
| 5 | `butteraugli_gpu::ButteraugliOpaque::pack_srgb_into_packed_u32_handle` | butteraugli-gpu | B | Used only in `examples/bench_staging_block.rs` (diagnostic bench, not production path). No production callers. |
| 6 | `iwssim::rgb_u8_to_gray_bt601(&[u8], &mut [f32])` (in-place) | iwssim | B | Only the `_vec` variant is called cross-crate (by `iwssim-gpu`). The in-place form has zero external consumers. |
| 7 | `iwssim::STRIP_BODY_MIN`, `iwssim::STRIP_HALO_ROWS` | iwssim | B | Self-test only; callers receive an error on invalid `h_body`, don't need the bound constant. |
| 8 | `ssim2_gpu::memory_mode::STRIP_HALO_ROWS`, `STRIP_H_BODY_DEFAULT` | ssim2-gpu | B | Self-test only; not referenced by `zenmetrics-api`, CLI, or orchestrator. |
| 9 | `cvvdp_gpu::memory_mode::estimate_gpu_memory_bytes_for_mode`, `estimate_gpu_memory_bytes_usize` | cvvdp-gpu | B | Self-test only; stable external surface uses crate-root re-exports. |
| 10 | `zensim_gpu::memory_mode::estimate_strip_gpu_memory_bytes` (regime-blind) | zensim-gpu | B | Legacy stub; `estimate_strip_gpu_memory_bytes_with_regime` is the correct function for this crate. Zero external consumers of the blind form. |

---

## Items confirmed KEEP (representative)

- All of `zenmetrics_api::hdr::*` — June 2026 PU program API, keep wholesale.
- `Metric`, `MetricKind`, `MetricParams`, `Score`, `Scores`, `Backend` — primary umbrella API.
- `reclaim_pooled_vram`, `resolve_memory_mode`, `score_pair` — used by CLI and orchestrator.
- `MemoryMode`, `CachedRefStripPolicy` — umbrella policy enums.
- `context::MetricContext<R>`, `context::PairHandles` (`cubecl-types` feature) — batch upload-once API.
- Re-exports `zenmetrics_api::cvvdp`, `::butter`, `::ssim2`, `::dssim`, `::iwssim`, `::zensim` — per-metric namespaces consumed by CLI.
