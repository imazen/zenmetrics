# ABLATION-butteraugli-gpu.md
<!-- date: 2026-06-11 | snapshot-commit: 2524d81f | grep-template: ugrep -r --include="*.rs" SYMBOL /home/lilith/work --exclude-dir=target --exclude-dir=.jj --exclude-dir=zenmetrics -->

## Scope
Snapshot: `docs/public-api/butteraugli-gpu.txt` — 207 items (default), 210 items (all-features).
The 3 extra items in all-features are `Butteraugli<R>` + its `compute_handles` + a trait impl, gated behind `cubecl-types`.

Known consumers: `zenmetrics-api` (`reclaim_pooled_vram`), `zen-metrics-cli`, `benchmarks/butteraugli-gpu-*` examples (use `ButteraugliBatch`, `pack_srgb_into_packed_u32_handle`, `reduce_diffmap_to_score`).

---

## Summary

| Class | Count | % of 207 |
|-------|-------|---------|
| Total items (default) | 207 | 100 % |
| Flagged (A) | 0 | 0 % |
| Flagged (B) | 2 | < 1 % |
| KEEP (used or deliberate) | 205 | >99 % |

---

## Flagged items

### B — `pub(crate)` / remove candidates

| Item | Reason |
|------|--------|
| `butteraugli_gpu::memory_mode::auto_strip_body_for(u32, u32, usize) -> u32` | Consistent with the pattern across all GPU crates. Zero cross-crate references outside self-tests. B-class. |
| `butteraugli_gpu::ButteraugliOpaque::pack_srgb_into_packed_u32_handle(&self, &[u8]) -> Result<Handle>` | Only used in `butteraugli-gpu/examples/bench_staging_block.rs` (internal benchmark). No production callers in org-wide grep. This is a staging-path diagnostic tool that leaked into the public API. Candidate B — or at minimum A (`#[doc(hidden)]`). Given publish=false, B is cheap. |

---

## Items confirmed KEEP (representative)

- `ButteraugliOpaque`, `ButteraugliParams`, `GpuButteraugliResult`, `Error`, `Result` — primary API.
- `ButteraugliBatch<R>` — used by `examples/batch_parity.rs` and `options_smoke.rs`; intentional batch API for jxl-encoder use case.
- `reduce_diffmap_to_score` — used by `examples/reduction_parity.rs`; intentional for butteraugli reduction.
- `memory_mode::reclaim_pooled_vram` — called by `zenmetrics-api`.
- `memory_mode::resolve_auto`, `estimate_gpu_memory_bytes`, `estimate_strip_gpu_memory_bytes`, `live_vram_probe_bytes`, `vram_cap_bytes` — orchestrator / opaque consumers.
- `memory_mode::MemoryMode`, `ResolvedMode` — typed mode API.
- `ButteraugliOpaque::compute_srgb_u8_with_pnorm3`, `compute_from_linear_interleaved`, `compute_from_linear_planes` — HDR path consumers.
