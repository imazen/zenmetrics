# cvvdp-gpu port status

Tracking faithful-port progress against the Python reference
(`gfxdisp/ColorVideoVDP`). One row per pipeline stage.

| Stage              | Module                 | Status                                   | Parity check                              |
|--------------------|------------------------|------------------------------------------|-------------------------------------------|
| sRGB → linear      | `kernels/color`        | host scalar + cubecl kernel body         | 2e-3 vs pycvvdp scalar goldens            |
| Display model      | `kernels/color`        | fused into host scalar + kernel          | same                                      |
| RGB → DKL          | `kernels/color`        | fused into host scalar + kernel          | same                                      |
| Laplacian pyramid  | `kernels/pyramid`      | host scalar (reduce/expand) interior-OK  | constant-signal interior + GAUSS5 sum     |
| CSF weighting      | `kernels/csf`          | scaffold                                 | none                                      |
| Contrast masking   | `kernels/masking`      | scaffold                                 | none                                      |
| Per-band pooling   | `kernels/pool`         | scaffold                                 | none                                      |
| Host fold / JOD    | `pipeline`             | scaffold                                 | none                                      |

## Reference version pin

`gfxdisp/ColorVideoVDP` **v0.5.4** (latest tag as of 2026-05-14).
Driver script in `scripts/cvvdp_goldens/` runs `pycvvdp==0.5.4` to
produce parity goldens. When bumping: also bump the R2 prefix
(`v1` → `v2`), the `GOLDEN_VERSION` const in `tests/common/mod.rs`,
and the version assertion in `tests/parity.rs`.

The cvvdp parameter JSON gets vendored into
`crates/cvvdp-gpu/data/cvvdp_v0.5.4.json` once the script lands (small
~5 KB file, safe to commit) and loaded through `params::CvvdpParams`.

## Out of scope (v0)

- Video / temporal channels (sustained + transient).
- Foveation / gaze maps.
- HDR display models — sRGB-std only for the initial parity pass.

## Open questions

- **Edge fix-ups for pyramid reduce/expand**: cvvdp v0.5.4 uses
  `F.conv2d(padding=2)` + explicit row/col patches that reproduce
  symmetric reflection (see `lpyr_dec.gausspyr_reduce`/`gausspyr_expand`).
  The Rust port currently uses pure symmetric reflection via a
  reflect-index helper, which matches the interior exactly but
  diverges on the outer 2-pixel ring (for expand) and the top/bottom
  rows of odd-height inputs (for reduce). Need to port the explicit
  patches before any whole-image JOD parity test can pass.
- **Per-band CSF weight precomputation**: should the host upload one
  flat `f32` array (`n_levels × N_CHANNELS`) or one tensor per band?
  Single flat upload is simpler; keep unless a per-band variant becomes
  necessary.
- **Atomics for pooling**: cubecl-cpu doesn't yet support
  `Atomic<f32>::fetch_add` (per zensim-gpu's lib.rs). Use per-band
  per-block partials with a tree reduction, same shape as zensim-gpu's
  fused features kernel, so the CPU runtime works.
