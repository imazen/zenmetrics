# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES

None.

### Added

- `fdlibm::powf_libm_8x` / `powf_libm_16x` (avx512), `vpow_libm_into` and `vpowf_libm_pairs_into`: lane-vectorized `libm::powf`, bit-identical per lane on every input (`d747a7b6`).
- Shared caller-owned math extracted from CVVDP, with native f32x16 v4/v4x dispatch, lazy policy/token kernels, bounded LUT interpolation, and deterministic parallel helpers (`01e28ca1`).

### Fixed

- Offset-power lanes whose addition rounds to the offset use the scalar policy offset result, preserving exact-zero std behavior (`01e28ca1`).
