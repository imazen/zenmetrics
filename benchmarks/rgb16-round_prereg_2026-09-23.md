# RGB16 decode preregistration — 2026-09-23

This is a deterministic decode correctness study. No human labels, sample selection, bootstrap CI, or random seeds apply.

## Inputs and decision rule

- Base: `master@origin` at workspace creation (`e2a5e68a`). The committed source tree is the input; its commit id is the content hash.
- Use the committed `tests/fixtures/ref_64.avif`; record its SHA256 before evaluation. Patch only its CICP transfer field to codes 1, 4, 6, 8, 13, 14, and 15.
- For each route, compare RGB8 against exact integer rounding of the native `decode_full` samples. No tolerance and no captured pixel hash oracle.
- Accept an AVIF descriptor tag only if all seven SDR transfer patches preserve current decoded RGB8 and all five unmodified HDR tripwire tests pass.
- Exercise every other available >8-bit/f32 decode fixture; report absent fixtures as MISSING.
- The required release CLI suite, touched suites, and clippy must pass before proposing merge.
