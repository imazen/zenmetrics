# ABLATION-cvvdp.md
<!-- date: 2026-06-11 | snapshot-commit: 2524d81f | grep-template: ugrep -r --include="*.rs" SYMBOL /home/lilith/work --exclude-dir=target --exclude-dir=.jj --exclude-dir=zenmetrics -->

## Scope
Snapshot: `docs/public-api/cvvdp.txt` — 586 items (default), 587 items (all-features).
The single extra item in all-features is from the `__simd_equiv_test` feature, which is underscore-prefixed and excluded by convention.

Known consumers: `cvvdp-gpu` (cross-crate re-exports from `cvvdp::kernels::*`), `jxl-encoder` (via `cvvdp_cpu` alias — uses `Cvvdp`, `CvvdpParams`, `Error`), `zenmetrics-api` (CPU metric path), `cvvdp-conformance` tests. The `cvvdp-gpu` crate explicitly tests `cvvdp_gpu::kernels::*` surface via `lib_reexports.rs` pinning test, so every re-exported kernel item has at least that consumer.

---

## Analysis approach
The `cvvdp::kernels` module (180+ items) was deliberately moved from `cvvdp-gpu` to `cvvdp` in Phase 8c.1-B so the CPU crate owns the canonical scalar implementations. `cvvdp-gpu` re-exports them verbatim, and `cvvdp-gpu/tests/it/lib_reexports.rs` pins the re-export surface. Therefore **all `cvvdp::kernels::*` items are legitimate public API** despite looking like implementation details — they are the source-of-truth for the GPU crate's own public surface.

The module-level items in `cvvdp::kernels::csf`, `cvvdp::kernels::color`, `cvvdp::kernels::masking`, `cvvdp::kernels::diffmap`, `cvvdp::kernels::pyramid`, `cvvdp::kernels::pool` are used directly by conformance tests, the GPU-side kernel tests, and the `jxl-encoder` perceptual loop. KEEP wholesale.

---

## Summary

| Class | Count | % of 586 |
|-------|-------|---------|
| Total items | 586 | 100 % |
| Flagged (A) | 0 | 0 % |
| Flagged (B) | 2 | < 1 % |
| KEEP (used or deliberate) | 584 | >99 % |

---

## Flagged items

### B — `pub(crate)` / remove candidates

| Item | Reason |
|------|--------|
| `cvvdp::diffmap` (the module, not `cvvdp::kernels::diffmap`) | The module itself is `pub mod diffmap` but contains zero public items — `DiffmapAccum`, `accumulate_band_diffmap`, and `finalize_diffmap` are all `pub(crate)`. The snapshot shows `pub mod cvvdp::diffmap` with no child items. No consumers reference `cvvdp::diffmap::` anywhere. The module is a documentation namespace stub. Candidate B: make `pub(crate) mod diffmap`. Cost is zero since publish=false. |
| `cvvdp::kernels::csf::csf_lut_v0_5_4` (sub-module) | The LUT constants are also re-exported at `cvvdp::kernels::csf::GE_SIGMA`, `LOG_L_BKG_AXIS`, etc. (without the version submodule). No cross-crate consumer references `csf_lut_v0_5_4::*` directly (grep finds zero external hits). The re-export at the parent level is the stable path. Candidate B: make `pub(crate) mod csf_lut_v0_5_4`. The re-exports at `csf::` level remain. |

---

## Items confirmed KEEP (representative)

- `cvvdp::Cvvdp`, `Error`, `Result`, `PlaneShapeMismatch`, `DimensionMismatch` — primary API.
- `cvvdp::params::*`, `cvvdp::presets::*` — re-exported by `cvvdp-gpu`; used by jxl-encoder via `cvvdp_gpu::params::DisplayModel`, `Eotf`, `Primaries`.
- `cvvdp::host_scalar::predict_jod_still_3ch*` — used by `cvvdp-gpu::host_scalar` re-export.
- All `cvvdp::kernels::**` items — canonical source for `cvvdp-gpu::kernels::*` re-exports; pinned by `lib_reexports.rs`.
- `cvvdp::MAX_LEVELS`, `PYRAMID_MIN_DIM`, `N_CHANNELS`, `CVVDP_COLUMN_NAME` — pinned by `lib_reexports.rs` test.
- `cvvdp::Cvvdp::score_*` methods, `warm_reference*`, `score_from_linear_planes*` — jxl-encoder buttloop consumers.
