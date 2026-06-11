# ABLATION-iwssim-gpu.md
<!-- date: 2026-06-11 | snapshot-commit: 2524d81f | grep-template: ugrep -r --include="*.rs" SYMBOL /home/lilith/work --exclude-dir=target --exclude-dir=.jj --exclude-dir=zenmetrics -->

## Scope
Snapshot: `docs/public-api/iwssim-gpu.txt` — 232 items (default == all-features).
`iwssim-gpu` is the reference crate for the shared `memory_mode` pattern — other GPU crates doc-comment "see `iwssim_gpu::memory_mode::live_vram_probe_bytes`" as the canonical description. This crate's memory_mode surface is the most complete and most consumed.

Known consumers: `zenmetrics-api` (`reclaim_pooled_vram`), `zen-metrics-cli` (`IWSSIM_COLUMN_NAME`, `IwssimParams`), `iwssim-gpu/tests/it/vram_probe.rs` (live VRAM tests using `live_vram_probe_bytes`, `vram_cap_bytes`).

---

## Summary

| Class | Count | % of 232 |
|-------|-------|---------|
| Total items | 232 | 100 % |
| Flagged (A) | 0 | 0 % |
| Flagged (B) | 1 | < 1 % |
| KEEP (used or deliberate) | 231 | >99 % |

---

## Flagged items

### B — `pub(crate)` / remove candidates

| Item | Reason |
|------|--------|
| `iwssim_gpu::memory_mode::auto_strip_body_for(u32, u32, usize) -> u32` | Internal helper, same pattern as the other GPU crates. Used only within `iwssim-gpu/src/memory_mode.rs`. No cross-crate references. `resolve_auto` is the external entry point. B-class. |

---

## Items confirmed KEEP (representative)

- `IwssimOpaque` (called `Iwssim` via re-export), `IwssimParams`, `Error`, `Result` — primary API.
- `IwssimParams::allow_small` — consumed by CLI.
- `IWSSIM_COLUMN_NAME`, `MIN_NATIVE_DIM`, `NUM_SCALES` — consumed by CLI / orchestrator.
- `memory_mode::live_vram_probe_bytes`, `vram_cap_bytes` — consumed by `iwssim-gpu/tests/it/vram_probe.rs` (verified by explicit test suite).
- `memory_mode::reclaim_pooled_vram` — called by `zenmetrics-api`.
- `memory_mode::resolve_auto`, `estimate_gpu_memory_bytes`, `estimate_strip_gpu_memory_bytes` — used by opaque init / orchestrator.
- `memory_mode::MemoryMode`, `ResolvedMode` — typed mode API.
