# zenmetrics-math

Internal (`publish = false`) caller-owned SIMD slice math shared by CPU IQA
metrics. Runtime dispatch selects v4x/v4 (native f32x16, opt-in `avx512`),
v3, NEON, wasm128, or scalar (f32x8). Arithmetic uses separate multiply/add;
reductions keep a fixed order within a width, so cross-width totals can differ.

`vexp_into`, `vlog_into`, `vpow_into` (also exported as `_into_slice`), masking, Weber contrast, temporal FIR,
and pooling helpers allocate no scratch. `gather_lerp_into` interpolates a
uniform-axis LUT with bounds-safe per-lane gathers; `gather_lerp_exp_into`
fuses interpolation and exponential scaling. All slice lengths must match.
For strided planes, call these primitives on each active row slice.

The default `Midp` policy uses magetypes' unchecked medium-precision math.
Use finite inputs and positive normal logarithm/power bases; exponential
outputs must remain normal and finite. NaNs and subnormal log/pow bases are
outside that contract. Offset power accepts subnormal magnitudes because the
positive offset moves the base into the valid domain. Arithmetic propagates
NaNs. Short tails use scalar libm with `std`; allocator-only builds use
magetypes' checked scalar approximation.

Consumers choose a token and policy through `width8`/`width16` generic kernels.
`MathPolicy` selects transcendental implementations and scalar tails without
implementing sealed backend traits. Only instantiated policy/token pairs
compile. The dispatch convenience functions use `Midp`.

`par` contains deterministic band and row helpers, with optional Rayon via
`parallel`. Scheduling does not change partition boundaries or fold order.
The metric crates retain their existing local `par` modules in this migration.

Validation recipes in the workspace justfile use CI-pinned sibling snapshots:
`just solmath-lock`, `just solmath-test`, and `just solmath-csf-bench`.

Offset-power kernels use the scalar policy result when adding an input rounds
to the offset itself. This keeps exact-zero and subnormal inputs consistent
across vector widths without leaving a transcendental approximation residual.

The [CVVDP migration measurements](../../benchmarks/cvvdp_shared_math_2026-10-11.md)
record parity, the single-size wall check, source provenance, and unmeasured tiers.
