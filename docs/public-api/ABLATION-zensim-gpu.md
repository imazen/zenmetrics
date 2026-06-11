# ABLATION-zensim-gpu.md
<!-- date: 2026-06-11 | snapshot-commit: 2524d81f | grep-template: ugrep -r --include="*.rs" SYMBOL /home/lilith/work --exclude-dir=target --exclude-dir=.jj --exclude-dir=zenmetrics -->

## Scope
Snapshot: `docs/public-api/zensim-gpu.txt` — 234 items (default == all-features).

Known consumers: `zenmetrics-api` (opaque path, `reclaim_pooled_vram`), `zen-metrics-cli` (`ZensimFeatureRegime`, `ZensimParams`), `zensim-gpu/tests/it/*`.

---

## Summary

| Class | Count | % of 234 |
|-------|-------|---------|
| Total items | 234 | 100 % |
| Flagged (A) | 0 | 0 % |
| Flagged (B) | 2 | < 1 % |
| KEEP (used or deliberate) | 232 | >99 % |

---

## Flagged items

### B — `pub(crate)` / remove candidates

| Item | Reason |
|------|--------|
| `zensim_gpu::memory_mode::auto_strip_body_for(u32, u32, ZensimFeatureRegime, usize) -> u32` | Internal helper. Used only within `zensim-gpu/src/memory_mode.rs` itself (no cross-crate hits in org grep). `resolve_auto` is the external-facing function. B-class. |
| `zensim_gpu::memory_mode::estimate_strip_gpu_memory_bytes(u32, u32) -> Option<usize>` | This is the regime-blind variant; `estimate_strip_gpu_memory_bytes_with_regime` is the full one. The regime-blind form is a legacy stub with zero external consumers. The regime-aware `estimate_strip_gpu_memory_bytes_with_regime` is correct for this crate. B-class. |

---

## Notable: `CUBECL_OVERHEAD_BYTES` const
Used by `zensim-gpu/tests/it/memory_mode.rs` for budget calculation verification. This is a self-test consumer only but the constant's purpose (documenting cubecl allocation overhead) makes it reasonable to keep public for diagnostic use. KEEP.

## Items confirmed KEEP (representative)

- `ZensimOpaque`, `ZensimParams`, `Error`, `Result` — primary API.
- `ZensimFeatureRegime` enum + `needs_extended_kernel()` — consumed by `zen-metrics-cli` (`ZensimFeatureRegime::WithIw.total_features()`).
- `Zensim<R>` typed API — used in `zensim-gpu/tests/it/cpu_gpu_feature_sweep.rs`.
- `TOTAL_FEATURES`, `TOTAL_FEATURES_EXTENDED`, `TOTAL_FEATURES_WITH_IW` — consumed by cli tests.
- `memory_mode::reclaim_pooled_vram` — called by `zenmetrics-api`.
- `memory_mode::resolve_auto`, `estimate_gpu_memory_bytes`, `estimate_strip_gpu_memory_bytes_with_regime` — used by orchestrator / opaque init.
- `memory_mode::MemoryMode`, `ResolvedMode`, `CUBECL_OVERHEAD_BYTES` — used in tests.
