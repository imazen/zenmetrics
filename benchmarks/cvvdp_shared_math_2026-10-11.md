# CVVDP shared math and CSF wall check — 2026-10-11

The shared `zenmetrics-math` crate replaces CVVDP's private slice math. Native
v4/v4x entries use f32x16; the other tiers use f32x8. One macro defines both
width families. `MathPolicy` and explicit-token kernels keep policy/token
instantiations lazy. Feature entry points use `arcane`; their generic inner
bodies inline without another trampoline. No sealed backend implementations
were added.

CSF sensitivity evaluation now uses bounded LUT lane gathers, vector lerp, and
Midp exponential evaluation in 256-element stack tiles. All six full-plane
sites and both strip sites use it. Scalar tails remain. The scalar CSF helper
remains as a test oracle. Other metrics retain their local parallel helpers.

The user approved one numerical correction beyond extraction: offset-power
lanes whose addition rounds to the offset now use the scalar policy's offset
result. The new strict tier tests exposed the inherited approximation residue;
the correction preserves their assertions, including exact-zero std results.

## Measured wall time

[Raw summaries](cvvdp_shared_math_2026-10-11.tsv) and
[commands, source hashes, and provenance](cvvdp_shared_math_2026-10-11.meta.json).

One deterministic textured 1024×1024 sRGB pair, reused scorer, 30 zenbench
samples per run, CPU i265, cores 2–9, Rayon 8, normal bench profile without
`target-cpu=native`. The benchmark declares its internal thread count to
zenbench so its own workers do not trigger the background-load gate.

| Variant | Mean ms | Median ms |
|---|---:|---:|
| Scalar-CSF control, with corrected shared math | 44.128728430 | 44.166687000 |
| Vector CSF, with the same corrected shared math | 36.921654770 | 36.933281500 |

The measured mean ratio is 1.195199097; elapsed time decreased 16.331931412%.
Both initial scores were 9.861638069. Earlier complete runs are also retained
in the TSV; they were separate samples under different ambient load.
This measures one pair and one host, not a corpus-wide or AVX-512 speedup.

`run-heavy` control: `rc=0 13s | peak-RSS 0.35GiB | min-avail 25846MiB | peak-load 2.27`.
Vector run: `rc=0 12s | peak-RSS 0.35GiB | min-avail 25881MiB | peak-load 0.82`.
These RSS numbers cover the guarded command, including compilation.

Disassembly of the AVX2 gather/exp entry shows YMM interpolation and the
vector polynomial exponential. LUT loads are bounds-checked per-lane scalar
gathers; arithmetic around them is vectorized.

## Validation and limits

Existing CVVDP JOD tolerances and strip/full bit-equality assertions were
unchanged. Measured maximum JOD deviation from the scalar reference:
9.536743e-7 across 11 synthetic pairs (16²–512²), 4.7683716e-7 across JPEG
qualities 1, 5, 20, 45, 70, 90. Direct CSF sensitivity maximum relative error:
4.7464144e-7 across all three channels, six frequencies, LUT endpoints,
bracket edges, and vector/tile boundary lengths.

Default and avx512+parallel math/CVVDP tests pass. Scalar and AVX2 execute on
this host. Native v4/v4x compile, but the CPU cannot summon their tokens;
NEON/wasm runtime checks were not run. The logical sixteen-lane scalar test
also passes. Reductions retain fixed order within a width; cross-width totals
can differ in low bits.

Default and parallel workspace builds pass with existing warnings. The full
workspace AVX-512 build fails in unchanged `iwssim/src/simd_kernels.rs`: missing
`infow_map_into_v4`, plus unsupported `f64x4<X64V4Token>`. Its source matches
the base commit byte for byte, and an isolated IWSSIM AVX-512 build reproduces
the same nine errors. This gate remains blocked independently of this change.

Changed-crate all-target Clippy passes with `-D warnings`; scoped fmt and the
CI-pinned lock check pass. The advisory quality kit is absent in this checkout.
The lock also includes the three dependency entries required by concurrently
landed MDCTPSNR and NLPD changes; dev-tree resolution upgrades were discarded as directed.

Follow-ups: execute native v4/v4x and NEON/wasm parity on matching hardware;
resolve the separate IWSSIM workspace build failure; evaluate additional math
policies and a measured native gather specialization. The parallel-helper
migrations remain separate work.
