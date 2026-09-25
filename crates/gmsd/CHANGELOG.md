# Changelog

## [Unreleased]

- Add quarantined paper-derived `ms_gmsd_rgb8` / `ms_gmsdc_rgb8`, pending
  qualification against an independent NumPy transcription. No
  author-software equivalence is claimed for those variants. Conventions
  were preregistered; implementation `03cf564f`.
- Add the quarantined `mdsi_rgb8` default-summation MDSI implementation
  and validation/speed examples. RGB8 ingress has explicit dimensions
  and byte stride. The implementation is written from the paper alone
  (clean room; readings of underspecified details in
  `docs/MDSI_CHOICES.md`) and validated against scores from the authors'
  reference software on 116 pairs (relative difference at most 1e-9; a wrong
  constant fails 109 of 116). The box average runs in exact integer
  arithmetic; the score is bit-identical to a straight-line evaluation of
  the paper's equations at every tier and thread count. Existing GMSD
  arithmetic is unchanged. `03cf564f`.
