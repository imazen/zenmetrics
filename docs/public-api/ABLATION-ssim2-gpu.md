# ABLATION-ssim2-gpu.md
<!-- date: 2026-06-11 | snapshot-commit: 2524d81f | grep-template: ugrep -r --include="*.rs" SYMBOL /home/lilith/work --exclude-dir=target --exclude-dir=.jj --exclude-dir=zenmetrics -->

## Scope
Snapshot: `docs/public-api/ssim2-gpu.txt` — 282 items (default), 314 items (all-features).
The 32 extra items in all-features are from the `cubecl-types` feature: `Ssim2<R>` typed API + `Ssim2Batch<R>` batch API + `compute_handles*`.

Known consumers: `zenmetrics-api` (opaque path, `reclaim_pooled_vram`), `zenmetrics-cli` (SSIM2_FIR_COLUMN_NAME, SSIM2_IIR_COLUMN_NAME, `column_name_for_blur`), `benchmarks/ssim2-gpu-*`, in-repo tests.

---

## Summary

| Class | Count | % of 282 |
|-------|-------|---------|
| Total items (default) | 282 | 100 % |
| Flagged (A) | 0 | 0 % |
| Flagged (B) | 3 | ~1 % |
| KEEP (used or deliberate) | 279 | ~99 % |

---

## Flagged items

### B — `pub(crate)` / remove candidates (default-features surface)

| Item | Reason |
|------|--------|
| `ssim2_gpu::memory_mode::auto_strip_body_for(u32, u32, usize) -> u32` | Internal helper used only within `ssim2-gpu/src/memory_mode.rs` + `ssim2-gpu/tests/it/memory_mode.rs`. No cross-crate references found. The public `resolve_auto` and `estimate_*` functions are the stable external surface. B-class. |
| `ssim2_gpu::memory_mode::STRIP_HALO_ROWS: u32` | Constant used only in self-tests (`ssim2-gpu/tests/it/memory_mode.rs`). No cross-crate consumers. B-class. |
| `ssim2_gpu::memory_mode::STRIP_H_BODY_DEFAULT: u32` | Same pattern — only in self-tests and `ssim2-gpu/src/`. Not referenced by `zenmetrics-api`, CLI, or orchestrator. `cvvdp-gpu` has its own `STRIP_H_BODY_DEFAULT`. B-class. |

---

## Memory mode module assessment
The `memory_mode` module is broadly pub because `zenmetrics-api::metric.rs` calls `ssim2_gpu::memory_mode::reclaim_pooled_vram(b)` directly, and `butteraugli-gpu/tests/it/opaque_strip_parity.rs` calls `resolve_auto`. These are legitimate cross-crate consumers. The module must remain pub; only the three items above are over-exposed.

## Items confirmed KEEP (representative)

- `Ssim2Opaque`, `Error`, `Result` — primary opaque API.
- `Ssim2Blur`, `Ssim2Mode`, `XybFlavor`, `GpuSsim2Result` — algorithm controls consumed by CLI.
- `Ssim2Params` — consumed by opaque path.
- `SSIM2_FIR_COLUMN_NAME`, `SSIM2_IIR_COLUMN_NAME`, `column_name_for_blur` — consumed by CLI parquet writer.
- `memory_mode::reclaim_pooled_vram` — called by `zenmetrics-api`.
- `memory_mode::resolve_auto`, `estimate_gpu_memory_bytes`, `estimate_strip_gpu_memory_bytes`, `live_vram_probe_bytes`, `vram_cap_bytes` — consumed by orchestrator / opaque init.
- `memory_mode::MemoryMode`, `ResolvedMode` — used in typed API path.
- `Ssim2<R>` + `Ssim2Batch<R>` (cubecl-types) — intentional low-level typed API for benchmarks.
