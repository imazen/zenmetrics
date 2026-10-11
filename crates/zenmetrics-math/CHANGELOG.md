# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES

None.

### Added

- Shared caller-owned math extracted from CVVDP, with native f32x16 v4/v4x
  dispatch, lazy policy/token kernels, bounded LUT interpolation, and the union
  of deterministic parallel helpers.

### Fixed

- Offset-power lanes whose addition rounds to the offset use the scalar policy
  offset result, preserving exact-zero behavior with the std Midp policy.
