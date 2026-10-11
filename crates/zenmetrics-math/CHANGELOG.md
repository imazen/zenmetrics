# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES

None.

### Added

- Shared caller-owned math extracted from CVVDP, with native f32x16 v4/v4x dispatch, lazy policy/token kernels, bounded LUT interpolation, and deterministic parallel helpers (`01e28ca1`).

### Fixed

- Offset-power lanes whose addition rounds to the offset use the scalar policy offset result, preserving exact-zero std behavior (`01e28ca1`).
