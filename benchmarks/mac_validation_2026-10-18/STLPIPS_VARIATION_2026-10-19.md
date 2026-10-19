# Retained STLPIPS-Alex Metal variation

Read-only capture on 2026-10-19. No additional benchmark, runtime probe,
worker rebuild, service mutation or gate change was performed for this audit.
The [structured receipt](stlpips-metal-variation-2026-10-19.json) binds the
copied attempt files, result seals, model/runtime identities and inputs.

Both first attempts below score the same 2048×1080 RGB8 pair with the frozen
ORT WebGPU/Metal worker. Each controller result is `failed`, with reason
`unqualified IQA_ROW status: variant_mismatch` and verified process cleanup.
Each raw row has ten zenbench samples, all ten `gate_clean=true`, and
`gpu_call_bit_variation_count=3`. Neither supplies qualified timing evidence.

| Cell | Workers | Baseline bits | Observed alternate bits |
|---|---:|---|---|
| `000315-af09ebacba16767a` | 1 | `3f82881020000000` | `3f857d5180000000` |
| `000319-e3b4d6fb2a3b641e` | 8 | `3f82881020000000` | `3f857d5180000000` |

The decoded raw scores are 0.009048582054674625 and 0.010492932051420212,
an absolute raw difference of 0.0014443499967455864. This is not a JND
conversion or a native-oracle comparison. The cause remains unproven.

The counter covers every zenbench warmup, calibration and timed result; checks
run in deferred output Drop after timing. The retained logs do not identify
which calls varied, so three variations must not be described as three of the
ten timed rounds. `abs_delta=0` and `map_score_matches_scalar_bits=true` in the
headline row do not override the separate per-call variation refusal.

At capture both cells had only `attempt-0001` under a frozen two-attempt cap:
one retry remained per cell. The normal queue continues its existing bounded
retry policy. These failed packets remain unqualified even if a later attempt
passes. No claim is made here about the later retry outcome.

Worker source base is `7310d24c889ff56dbc6c41ba8460e436d233f549`, binary
SHA256 `0773cd747da14a2f535719810cbfe9b0f93e9ee01dc1491c330e0755661bd4f9`.
ORT core is 1.31.0, official WebGPU plugin 0.4.0; full recorded hashes appear
in the receipt. These identities are copied from sealed retained evidence;
models and the binary were not rehashed during an active measurement.

Original attempts remain on mac under
`~/benchmark-results/iqa-mac-20261019-gpu-first.noindex/larger-nn-metal-attempts2-attempts2/cells/<cell>/attempt-0001/`.
Byte-verified copies of all attempt files, original context and remote SHA256s
are on i265 at `~/benchmark-artifacts/iqa-mac-stlpips-metal-20261019/`.
Only the small diagnostic projection is committed; no graphs, weights or
compiled outputs are included. This finding is separate from the known strict
CPU-assignment refusals documented in [the node audit](WEBGPU_NODE_AUDIT.md).

The later owner-authorized [finite-only timing policy](finite-metal-policy-2026-10-19.json)
uses a separate worker and immutable successor. Both remaining production
retries pass resource gates and are accepted with numerical-instability flags.
The original attempts above retain their original failed classification;
native-oracle parity and a cause for the variation remain separate questions.
