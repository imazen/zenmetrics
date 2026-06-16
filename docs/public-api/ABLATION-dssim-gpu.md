# ABLATION-dssim-gpu.md
<!-- date: 2026-06-11 | snapshot-commit: 2524d81f | grep-template: ugrep -r --include="*.rs" SYMBOL /home/lilith/work --exclude-dir=target --exclude-dir=.jj --exclude-dir=zenmetrics -->

## Scope
Snapshot: `docs/public-api/dssim-gpu.txt` — 220 items (default == all-features).

Known consumers: `zenmetrics-api` (`reclaim_pooled_vram`), `zenmetrics-cli`, `dssim-gpu/tests/it/*`.

---

## Summary

| Class | Count | % of 220 |
|-------|-------|---------|
| Total items | 220 | 100 % |
| Flagged (A) | 0 | 0 % |
| Flagged (B) | 1 | < 1 % |
| KEEP (used or deliberate) | 219 | >99 % |

---

## Flagged items

### B — `pub(crate)` / remove candidates

| Item | Reason |
|------|--------|
| `dssim_gpu::memory_mode::auto_strip_body_for(u32, u32, usize) -> u32` | Internal helper consistent with the pattern across all GPU crates. Zero cross-crate consumers found. `resolve_auto` is the external surface. B-class. |

---

## Items confirmed KEEP (representative)

- `DssimOpaque`, `Error`, `Result` — primary API.
- `memory_mode::reclaim_pooled_vram` — called by `zenmetrics-api`.
- `memory_mode::resolve_auto`, `estimate_gpu_memory_bytes`, `estimate_strip_gpu_memory_bytes`, `live_vram_probe_bytes`, `vram_cap_bytes` — orchestrator / opaque consumers.
- `memory_mode::MemoryMode`, `ResolvedMode` — typed mode API.
- `DssimParams`, `DssimScore` — consumed by tests and CLI.
