# cvvdp-gpu port status

Tracking faithful-port progress against the Python reference
(`gfxdisp/ColorVideoVDP`). One row per pipeline stage.

| Stage              | Module                 | Status     | Goldens? |
|--------------------|------------------------|------------|----------|
| sRGB → linear      | `kernels/color`        | scaffold   | no       |
| Display model      | `kernels/color`        | scaffold   | no       |
| RGB → DKL          | `kernels/color`        | scaffold   | no       |
| Laplacian pyramid  | `kernels/pyramid`      | scaffold   | no       |
| CSF weighting      | `kernels/csf`          | scaffold   | no       |
| Contrast masking   | `kernels/masking`      | scaffold   | no       |
| Per-band pooling   | `kernels/pool`         | scaffold   | no       |
| Host fold / JOD    | `pipeline`             | scaffold   | no       |

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

- **Edge handling in the pyramid filters**: cvvdp's reference uses
  `replicate` border in PyTorch's `F.conv2d(padding_mode='replicate')`.
  Confirm during port.
- **Per-band CSF weight precomputation**: should the host upload one
  flat `f32` array (`n_levels × N_CHANNELS`) or one tensor per band?
  Single flat upload is simpler; keep unless a per-band variant becomes
  necessary.
- **Atomics for pooling**: cubecl-cpu doesn't yet support
  `Atomic<f32>::fetch_add` (per zensim-gpu's lib.rs). Use per-band
  per-block partials with a tree reduction, same shape as zensim-gpu's
  fused features kernel, so the CPU runtime works.
