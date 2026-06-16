# ABLATION-iwssim.md
<!-- date: 2026-06-11 | snapshot-commit: 2524d81f | grep-template: ugrep -r --include="*.rs" SYMBOL /home/lilith/work --exclude-dir=target --exclude-dir=.jj --exclude-dir=zenmetrics -->

## Scope
Snapshot: `docs/public-api/iwssim.txt` — 99 items (default == all-features).

Known consumers checked: `zenmetrics-cli` (primary driver), `zenmetrics-api`, `zenmetrics-orchestrator`, in-repo tests.

---

## Summary

| Class | Count | % of 99 |
|-------|-------|---------|
| Total items | 99 | 100 % |
| Flagged (A) | 2 | 2 % |
| Flagged (B) | 3 | 3 % |
| KEEP (used or deliberate) | 94 | 95 % |

---

## Flagged items

### A — `#[doc(hidden)]` / `#[deprecated]` candidates

| Item | Reason |
|------|--------|
| `iwssim::IwssimParams` pub fields: `bl_sz_x`, `bl_sz_y`, `sigma_nsq` | These three tunables are never set via struct literals outside the crate — all external use goes through `IwssimParams::new()` / `IwssimParams::allow_small(bool)` constructors. The fields are pub but have no cross-crate assignment hits in the org-wide grep (only `iw_flag` and `parent` appear in test files within this repo; `bl_sz_x`/`bl_sz_y`/`sigma_nsq` have zero external references). Candidate for `#[doc(hidden)]` (A) — note publish=false so B is equally cheap. |
| `iwssim::IwssimParams::parent: bool` | Same pattern — pub field accessed only from within `iwssim/src/` and `tests/`. `parent` is used structurally in tests via `IwssimParams { parent: true, ..Default::default() }` to enable parent-mode SSIM, so it IS intentional algorithm control. However it leaks the algorithm's internal semantics with no stable name in the literature. Candidate A (`#[doc(hidden)]`) — worth adding a named constructor instead. |

### B — `pub(crate)` / remove candidates

| Item | Reason |
|------|--------|
| `iwssim::rgb_u8_to_gray_bt601(&[u8], &mut [f32])` | The in-place variant is not called anywhere outside `iwssim/src/` or `iwssim-gpu/src/`. The `_vec` variant below is used by the GPU crate's pipeline. The in-place form is a helper that leaked through. No consumers in org grep. B-class (cheaply removable — workspace-only). |
| `iwssim::STRIP_BODY_MIN: u32` | Used only in `iwssim/tests/strip_parity.rs` (self-test) and one `benchmarks/` driver. Callers who pass an invalid `h_body` get an error; they don't need to replicate the constant. B-class — could be crate-private with the error message carrying the bound. |
| `iwssim::STRIP_HALO_ROWS: usize` | Same pattern. Mentioned in `cpu_profile` benchmarks for documentation, but no algorithmic dependency — callers don't need to predict the halo size. B-class. |

---

## Items confirmed KEEP (representative)

- `Iwssim`, `IwssimParams`, `IwssimScore`, `Error` — primary API, heavily used.
- `IwssimParams::allow_small()` constructor — used by `zenmetrics-cli`.
- `IwssimParams::iw_flag` — used in parity tests to toggle IW off; algorithmic toggle that users do need.
- `STRIP_BODY_DEFAULT` — used in `benchmarks/cpu_profile` and `iwssim/tests/strip_parity.rs`.
- `rgb_u8_to_gray_bt601_vec` — used by `iwssim-gpu/src/pipeline.rs`.
- `IWSSIM_COLUMN_NAME`, `MIN_NATIVE_DIM`, `NUM_SCALES` — all consumed by CLI / orchestrator.
- All `Iwssim::score_*` and `warm_reference_*` methods — used by CLI + orchestrator.
