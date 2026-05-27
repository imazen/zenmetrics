# SIMD-vs-Scalar Kernel Equivalence (brute-force harness)

`crates/cvvdp/tests/simd_equivalence.rs` brute-force compares every
cvvdp SIMD kernel against its scalar reference across thousands of
randomized inputs + adversarial edge cases, and measures the per-element
ULP / relative-error envelope.

**Why this exists.** The end-to-end 1e-4 JOD parity gate
(`tests/parity_against_host_scalar.rs`) and the ~5 fixture tests inline
in `simd_pyramid.rs` / `simd_math.rs` can MASK a per-element kernel
divergence: the Minkowski spatial/band/channel pooling absorbs isolated
per-pixel errors before they reach the scalar JOD. This harness checks
each kernel in isolation, at the element level, so a regression that the
pool would hide still trips a test.

**Run it:**

```bash
cargo test -p cvvdp --features __simd_equiv_test --test simd_equivalence -- --nocapture
```

The `__simd_equiv_test` cargo feature enables a `#[doc(hidden)]`
visibility shim (`lib.rs::__simd_equiv_test_api`) that re-exports the
`pub(crate)` kernel entry points so an external test crate can drive
them. The shim carries NO logic — it is thin `pub fn` forwarders (a
`pub use` of a `pub(crate)` item is rejected by the compiler, E0364).
The feature is OFF by default; no production path touches it.

## Kernels covered

1. **σ=3 13-tap Gaussian blur** — `gaussian_blur_sigma3_simd` plus its
   horizontal/vertical passes in isolation
   (`pu_blur_horizontal_pass` / `pu_blur_vertical_pass`).
2. **Pyramid 5-tap reduce/expand** — `reduce_vertical_pass`,
   `reduce_horizontal_pass`, `expand_vertical_pass`,
   `expand_horizontal_pass`.
3. **Masking transcendentals** — `vexp_into`, `vlog_into`, `vpow_into`,
   `safe_pow_with_offset_into` (wrap magetypes `*_midp_unchecked`
   approximations).

## Coverage

Per kernel:
- **≥ 1000 randomized input cases** (blur passes: 1500 = 25 sizes × 5
  distributions × 12 seeds; pyramid reduce: 1260; pyramid expand: 1020;
  the transcendentals process tens of thousands of array elements across
  17 sizes × multiple distributions × all masking exponents). Element
  comparison counts: 12 M (blur passes), 2–6 M (pyramid), 19 k–155 k
  (transcendentals).
- **Sizes** stress boundaries + remainder lanes + odd/prime dims:
  1×1, 7×7, 7×13, 13×7, prime pairs (23×23, 31×29, 47×53, 97×101),
  lane-aligned (8/16/32/64/256), lane±1 (9, 17, 33, 257, 259).
- **Value distributions:** uniform [0,1), uniform [0, 1e4), Gaussian
  (signed), log-uniform 10^[-6,4] (spans magnitudes), exact powers of 2.
- **Adversarial fills:** all-zero, all-max, all-equal, single-pixel
  spike (center / first / last — stresses reflect boundary), checkerboard
  0/max (high-freq response), denormal+subnormal mix.

## Metrics

Per kernel the harness reports (printed under `--nocapture`):
max absolute error, max relative error, ULP distribution
(histogram 0 / 1 / 2-4 / 5-16 / 17-128 / >128), p50 / p99 / max ULP, and
worst-case `(got, want)` pair. ULP distance uses the IEEE-754 total-order
key (the `f32::total_cmp` transform widened to u64 so the difference
can't overflow at the line extremes); `ulp_diff_self_check` validates it.

## Measured envelopes (2026-05-26, AVX2 `v3` host, 7950X)

### Blur + pyramid — BIT-IDENTICAL (or sub-2-ULP)

| Kernel                     | max ULP | max rel    | note |
|---                         |---      |---         |---   |
| blur H pass                | **0**   | 0          | bit-identical |
| blur V pass                | **0**   | 0          | bit-identical |
| blur full two-pass         | **0**   | 0          | bit-identical vs upstream `gaussian_blur_sigma3` |
| pyramid reduce V           | **0**   | 0          | bit-identical |
| pyramid reduce H           | **0**   | 0          | bit-identical |
| pyramid expand V           | **2**   | 2.8e-15    | 0.01% of elements; subnormal tail only |
| pyramid expand H           | **2**   | 2.8e-15    | 0.01% of elements; subnormal tail only |

**Finding (better than expected).** The source comments anticipated a
"FMA-grouping difference (well below 1e-5 abs)" — in practice the blur
and reduce kernels are **bit-identical** to scalar across every input
tested. The SIMD interior accumulates the 5/13-tap dot in the SAME
left-to-right source order as the scalar reference (`v0*k0 + v1*k1 +
…`), and on this target LLVM contracts both paths to the same FMA
sequence. The expand passes show 2 ULP on ~0.01% of elements, exclusively
on subnormal magnitudes near the zero-insertion holes — numerically
negligible (2.8e-15 relative). All blur/pyramid tests assert a **tight
≤ 4 ULP bound** (full two-pass ≤ 8 ULP); the measured headroom is large.

These kernels are NOT approximations — any future change pushing them
above a few ULP indicates a boundary-handling or reassociation bug, not
hand-wavable "FMA noise", and the test will fail loudly.

### Transcendentals — APPROXIMATION envelopes (committed regression gates)

These wrap magetypes' `*_midp_unchecked` (`exp2(p·log2(x))` for pow,
etc.) which is documented at ~128 ULP / ~1e-5 relative. The harness
measures the actual envelope on the **in-contract domain** (the input
ranges the masking pipeline actually produces) and commits it as a
regression gate — NOT bit-exactness.

| Kernel    | max ULP | p99 ULP | max rel    | gate (ULP / rel) | in-contract domain |
|---        |---      |---      |---         |---               |---                 |
| `vexp`    | **14**  | 11      | 9.67e-7    | 24 / 2e-6        | x ∈ [-20, 20] |
| `vlog`    | **3**   | 2       | 2.82e-7    | 8 / 1e-6         | x > 0, 10^[-6,4] + near-1 |
| `vpow`    | **40**  | 16      | 2.52e-6    | 64 / 1e-5        | base ∈ [1e-3, 200] ∪ 10^[-4,4], p ∈ [0.5, 3.7] |
| `safe_pow`| **92**  | 14      | 8.28e-6   | 128 / 2e-5       | (x+1e-5)^p - 1e-5^p, x ∈ [0, 200] |

Gates = measured envelope rounded up to the next clean bucket, with
margin so legitimate cross-host jitter doesn't flake while a real
degradation (e.g. dropping to a `*_lowp` tier) still trips. `safe_pow`
runs ULP a bit higher than bare `vpow` because it composes the pow with
a subtraction; its measured max 92 sits comfortably inside the
documented ~128 ULP budget. **The masking parity claim (~128 ULP /
~1e-5 rel) is CONFIRMED, not exceeded** — `vpow`'s in-contract max rel is
2.5e-6, an order of magnitude better than the ~1e-5 budget.

#### Abs floor on the transcendental envelope

ULP / relative error against a near-zero reference is meaningless
(dividing a ~1e-16 absolute error by ~0 explodes both metrics). The
transcendental tests apply an **absolute magnitude floor of 1e-12**: when
both `got` and `want` fall below it, the comparison is excluded from the
ULP/rel envelope but **counted and reported** in `below_floor`. These are
catastrophic-cancellation residues and underflow products that the
masking pipeline never consumes (its contributions live well above
1e-12). The harness prints e.g. `below-floor skipped: 930 of 155352` for
`vpow` so the exclusion is auditable, not hidden.

## Out-of-contract behavior (DOCUMENTED, not a bug)

`transcendentals_out_of_contract_probe` records — and asserts only
finiteness on — the behavior of the `*_midp_unchecked` approximations on
inputs the masking pipeline **never** produces:

- **`pow_midp_unchecked` / `safe_pow` on subnormals** (e.g.
  `f32::MIN_POSITIVE`): the unchecked `log2` has no subnormal guard, so
  the result can be a large WRONG-MAGNITUDE value. This is the documented
  "positive, in-range inputs only" contract (`simd_math.rs`), not a bug
  to fix. The masking pipeline guarantees `|magnitude| + SAFE_EPS ≥ 1e-5`,
  so subnormals never reach the kernel.
- **`safe_pow` at x = 0**: scalar `want = (eps)^p − eps^p = 0` exactly;
  the SIMD path subtracts the EXACT precomputed `eps^p` from the
  APPROXIMATE `pow_midp(eps)`, leaving a ~1e-16 residue. The probe
  asserts the residue stays below 1e-9 (far below the masking floor) and
  finite.

The probe's hard assertion is **no NaN/Inf** — those would propagate into
the JOD pool. Magnitude error on out-of-domain inputs is expected and
recorded (visible under `--nocapture`), so a future agent sees the
behavior was measured and understood rather than re-discovering it.

## What a failure means

- **Blur / pyramid test fails** (> 4 ULP, or > 8 for full two-pass): a
  boundary-handling or reassociation bug. The failure message prints the
  worst `(got, want)`. Isolate the failing size/distribution — do NOT
  loosen the bound.
- **Transcendental test fails:** the approximation envelope regressed
  past the committed gate. If the magetypes tier intentionally changed
  (e.g. `pow_lowp`), re-measure and update the gate constant + this doc
  WITH the new numbers. If `vpow`'s envelope is WORSE than the documented
  ~128 ULP / ~1e-5 rel, the masking chunk's parity claim needs revisiting
  — surface it, don't paper over it.
