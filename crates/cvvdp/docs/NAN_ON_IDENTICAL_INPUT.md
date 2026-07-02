# cvvdp NaN on identical input pairs (2026-07-02)

Full investigation + evidence trail: `../../cvvdp-gpu/docs/NAN_ON_IDENTICAL_INPUT.md`
(the production bug was reported against the GPU `cvvdp-gpu` path; this crate's CPU
implementation was live-tested extensively as part of that investigation and never
reproduced a non-finite result, but got the same defensive fix for consistency and
because it shares the "identical images" input contract).

`Cvvdp::score` / `Cvvdp::score_with_diffmap` now short-circuit to `10.0` (zero
diffmap) when `reference_srgb == distorted_srgb` byte-for-byte, before any
DKL/pyramid/CSF/masking work runs. Regression test:
`tests/it/diffmap_invariants.rs::identical_solid_colors_yield_exact_max_jod_and_zero_diffmap`
(covers solid black/white/mid-gray/near-black/near-white — the prior identical-input
tests in this crate only used ramp or PRNG-noise patterns, never flat/solid colors).
