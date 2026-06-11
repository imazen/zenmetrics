# ABLATION-cvvdp-gpu.md
<!-- date: 2026-06-11 | snapshot-commit: 2524d81f | grep-template: ugrep -r --include="*.rs" SYMBOL /home/lilith/work --exclude-dir=target --exclude-dir=.jj --exclude-dir=zenmetrics -->

## Scope
Snapshot: `docs/public-api/cvvdp-gpu.txt` — 174 items (default == all-features).
`cvvdp-gpu` has the smallest per-crate surface because the canonical scalar items live in `cvvdp` and are re-exported here; the GPU-specific API is just the `CvvdpOpaque` + memory mode wrapper.

Known consumers: `zenmetrics-api` (`reclaim_pooled_vram`, `live_vram_probe_bytes`), `zenmetrics-orchestrator` (`live_vram_probe_bytes`, `chooser.rs`), `jxl-encoder` (`CvvdpOpaque`, `CvvdpParams`, `params::DisplayModel`), `benchmarks/gpu_vram_profile` (`estimate_gpu_memory_bytes`, `estimate_gpu_memory_bytes_capped`, `estimate_gpu_memory_bytes_strip_pair`), `cvvdp-gpu/tests/it/*` (conformance + strip parity).

---

## Summary

| Class | Count | % of 174 |
|-------|-------|---------|
| Total items | 174 | 100 % |
| Flagged (A) | 0 | 0 % |
| Flagged (B) | 2 | ~1 % |
| KEEP (used or deliberate) | 172 | ~99 % |

---

## Flagged items

### B — `pub(crate)` / remove candidates

| Item | Reason |
|------|--------|
| `cvvdp_gpu::memory_mode::estimate_gpu_memory_bytes_for_mode(u32, u32, MemoryMode) -> usize` | Used only in `cvvdp-gpu/tests/it/memory_mode.rs` (self-test). Not referenced in `zenmetrics-api`, `zenmetrics-orchestrator`, `benchmarks/`, or `jxl-encoder`. The public `estimate_gpu_memory_bytes` / `estimate_gpu_memory_bytes_capped` / `estimate_gpu_memory_bytes_strip_pair` functions at the crate root are the stable surface. B-class. |
| `cvvdp_gpu::memory_mode::estimate_gpu_memory_bytes_usize(u32, u32) -> usize` | Used only in `cvvdp-gpu/tests/it/memory_mode.rs` (self-test for the `usize` variant). The crate-root re-export `estimate_gpu_memory_bytes_usize` is also a test helper. The stable user-facing functions are `estimate_gpu_memory_bytes` (returns `Option<usize>`) + `estimate_gpu_memory_bytes_capped`. B-class. |

---

## Items confirmed KEEP (representative)

- `CvvdpOpaque`, `CvvdpParams`, `PerfMode`, `Cvvdp` (typed), `Error`, `Result` — primary API.
- `cvvdp_gpu::params::*` re-exports — consumed by `jxl-encoder/src/api.rs`.
- `cvvdp_gpu::PARALLEL_SAFETY_FACTOR`, `recommend_parallel` — pinned by `lib_reexports.rs` test; used in doctest math.
- `memory_mode::live_vram_probe_bytes` — called by `zenmetrics-api` and `zenmetrics-orchestrator`.
- `memory_mode::reclaim_pooled_vram` — called by `zenmetrics-api`.
- `estimate_gpu_memory_bytes`, `estimate_gpu_memory_bytes_capped`, `estimate_gpu_memory_bytes_strip_pair` (crate root) — consumed by `benchmarks/gpu_vram_profile/src/main.rs`.
- `memory_mode::STRIP_H_BODY_DEFAULT` — used by `cvvdp-gpu/tests/it/strip_mode_b_parity.rs` and self-test; reasonable to keep for the strip-mode calibration use case.
- `memory_mode::MemoryMode`, `ResolvedMode`, `STRIP_ALIGN` — typed mode API.
