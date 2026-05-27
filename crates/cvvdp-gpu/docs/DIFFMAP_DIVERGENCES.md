# Diffmap divergences vs the canonical CVVDP scalar path

**Scope**: this document covers the per-pixel diffmap exposed via
`Cvvdp::score_with_diffmap` and the `from_linear_planes` family
(added in the Phase 1 chunk of the CVVDP-fork RFC at
`~/work/zen/jxl-encoder/docs/RFC_CVVDP_FORK.md`). The scalar JOD
path is unchanged; this is purely about the *new* per-pixel signal.

## The RFC §3 invariant cannot hold strictly

The RFC sketches a contract:

```
JOD_scalar  ==  10.0 - minkowski_p_norm(diffmap_flat, p = params.beta_sch)
```

This is achievable for codecs whose JOD-style scalar is produced by
a **single Minkowski reduction** of a per-pixel signal. cvvdp v0.5.4
isn't shaped that way: the scalar JOD comes from a **three-stage
Minkowski pool with three different exponents** —

| Stage | Exponent | Source | Reduction |
|---|---|---|---|
| Spatial (per band, per channel) | `BETA_SPATIAL = 2` | `lp_norm_mean(D_kc, β=2)` | mean |
| Across bands (per channel) | `BETA_BAND = 4` | `lp_norm_sum({w_kc * q_kc : k}, β=4)` | sum (weighted) |
| Across channels | `BETA_CH = 4` | `lp_norm_sum({q_c : c}, β=4)` | sum |
| Image integration | scalar | `Q_tc * IMAGE_INT` | × |
| JOD map | piecewise | `met2jod(Q)` | smooth |

There is no closed-form single per-pixel signal whose Minkowski-p
norm collapses all three reductions identically. Two different
exponents (β=2 then β=4) at consecutive stages mean the pool is
non-commutative across stages — any per-pixel construction either
preserves the β=2 spatial mean (and breaks the β=4 channel sum) or
the β=4 channel sum (and breaks the β=2 spatial mean).

## What we implement instead

The recipe both `cvvdp-gpu::kernels::diffmap` (this crate, GPU path)
and `cvvdp::diffmap` (CPU port at master `da816947`) use:

1. **Bilinear-upsample** each per-(band, channel) masked-difference
   plane `D[k][c]` to base resolution `W × H`. Convention: OpenCV
   INTER_LINEAR / PyTorch align_corners=False
   (`fx = (x + 0.5) * src_w / dst_w − 0.5`, then clamp to
   `[0, src_w−1]`).
2. **Weighted sum across bands** per channel:
   `per_ch[c][i] = Σ_k per_sband_w[k][c] * PER_CH_W[c] * D_up[k][c][i]`,
   where `per_sband_w[k][c] = 1` for `k < n_levels − 1` and
   `BASEBAND_W[c]` at the baseband. Identical to the band weights
   used in `kernels::pool::do_pooling_and_jod_still_3ch`.
3. **Per-pixel Minkowski-p across DKL channels**:
   `diffmap[i] = (max(per_ch[0][i], 0)^β + max(per_ch[1][i], 0)^β
                + max(per_ch[2][i], 0)^β)^(1/β)` at `β = BETA_CH = 4`.
   The `max(., 0)` clamp masks f32 rounding noise (the analytical
   sum is non-negative by construction — masking output is
   non-negative). No `safe_pow` epsilon — the differentiability
   hack the scalar pool uses doesn't help the buttloop's per-block
   median+MAD reducer.

## What invariants the diffmap DOES satisfy

| Property | Tolerance | Source |
|---|---|---|
| `diffmap.len() == width * height` | exact | shape contract |
| Identity inputs → all-zero diffmap | 1e-7 absolute | masking stage produces D=0 for identical inputs; bilinear upsample of 0 is 0; sum of 0s is 0; pow(0, β) = 0 |
| Non-negative | exact | max(., 0) clamp + Minkowski |
| Monotone in distortion magnitude | exact at α-scale | scaling per-band D by α scales diffmap by α |
| GPU↔CPU per-pixel parity | 1e-5 absolute | shared recipe, see `tests/diffmap_invariants.rs::channel_pool_matches_cpu_recipe_pointwise` |

## What invariants the diffmap does NOT satisfy

- `JOD_scalar == 10.0 - minkowski_p_norm(diffmap_flat, p = BETA_CH)`
  — see "The RFC §3 invariant cannot hold strictly" above. The
  scalar JOD is unchanged; the diffmap is a separate side-channel
  output.
- Single-image-integration scale factor (`IMAGE_INT`) — applied to
  the scalar at the final step, NOT distributed per-pixel.
- pycvvdp `get_diff_map=True` compatibility — pycvvdp v0.5.4 does
  not export a per-pixel diffmap. If a future pycvvdp version adds
  one, this RFC's recipe is the contract (per RFC §3 "if upstream
  differs, RFC's recipe is the contract").

## Consumers

The jxl-encoder buttloop's per-block 8×8 median+MAD reducer
(`vardct/butteraugli_loop.rs::compute_per_block_distance`) is
metric-agnostic — it just needs a per-pixel signal where larger =
worse. The diffmap satisfies that contract regardless of the
strict-invariant divergence.

## See also

- `crates/cvvdp-gpu/src/kernels/diffmap.rs` module docstring — the
  recipe lives in the implementation comments too.
- `crates/cvvdp/src/diffmap.rs` — the CPU port's matching
  implementation. The two impls share the recipe byte-for-byte.
- `~/work/zen/jxl-encoder/docs/RFC_CVVDP_FORK.md` §3 — original
  RFC sketch of the contract.
