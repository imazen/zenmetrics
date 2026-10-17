# NN-port landing checks, 2026-10-17

The restored score-revision-1 DMM solver passes both frozen engine panels. No golden limit or classification changed. The independent arithmetic study remains incomplete; see [its per-change table](dmm_bisect_results_2026-10-17.md).

`~/tmp/nnport_land_gate.sh` exits 0 with every `GATE_PASS` and `GATE_DONE`. The original compile group preceded the arithmetic rollback, so `just nn-port-compile-gate` was rerun on the restored source and also exits 0. Captured trees for that repeat: `53721b154818`, then `d52b68695085` (the latter adds the completed epsilon measurement). It passes lock resolution, formatting, workspace CI clippy, strict CLI clippy, NN all-target clippy, browser check/registration, the default NN suite, benchmark-study controls, portable Python syntax, all nine arithmetic-isolation controls, and all twelve CLI feature shapes. No test expectation was relaxed.

| Native parity | Captured tree | Result |
|---|---|---|
| AHIQ rten | `b9cdc747f74d` | 17 rows; worst absolute error 5.295e-8 = 2.13e-6 JND; identity and exact warm/cold controls pass |
| DMM rten | `16af89c1fb81` | 24 frozen limits pass; 6 canonical JND, 18 pinned; identity and exact warm/cold controls pass |
| DMM ort-cuda | `27f91990c2a9` | 24 frozen limits pass; 7 canonical JND, 17 pinned; identity and exact warm/cold controls pass |
| AHIQ ort-cuda | `53721b154818` | 17 rows; worst absolute error 2.582e-7 = 1.04e-5 JND; identity and exact warm/cold controls pass |

DMM numerical arithmetic is the frozen baseline `66bb98605c83` in both native runs; the later source differences are tooling/docs. The complete independent baseline run also repeats all 24 CUDA scores bit-identically.

Resource wrapper receipts (parity/check runs, not speed benchmarks):

```text
Restored compile group: run-heavy: done rc=0 46s | peak-RSS 1.47GiB | min-avail 21531MiB | peak-load 1.81
AHIQ rten: run-heavy: done rc=0 149s | peak-RSS 1.38GiB | min-avail 20172MiB | peak-load 14.12
DMM rten: run-heavy: done rc=0 300s | peak-RSS 8.30GiB | min-avail 13409MiB | peak-load 5.94
DMM ort-cuda: run-heavy: done rc=0 276s | peak-RSS 3.01GiB | min-avail 19775MiB | peak-load 1.53
AHIQ ort-cuda: run-heavy: done rc=0 14s | peak-RSS 2.01GiB | min-avail 20141MiB | peak-load 1.64
```

Advisory quality comparison with the same pinned siblings/tool kit: original tree `9c69ab54` reports 4114 warnings across 195 files and two compile errors; restored tree `d5dc81ba` reports 4112 warnings across 194 files and no compile errors. Both report clean formatting. Deny, complexity and shellcheck tooling remain unavailable and are explicitly reported by the advisory runner; unused-dependency, typo and stale-doc findings remain advisory debt. Required strict CLI/NN clippy passes separately.

```text
Original quality: run-heavy: done rc=0 26s | peak-RSS 0.52GiB | min-avail 21316MiB | peak-load 1.34
Restored quality: run-heavy: done rc=0 29s | peak-RSS 0.57GiB | min-avail 20979MiB | peak-load 2.72
```

Full logs on the qualification host: `~/tmp/nnport_land_gate.log`, `~/tmp/nn-port-scratch/fix-final-compile.log`, `~/tmp/nn-port-scratch/fix-quality-{original,restored}.log`. The optional `just fmt` API-snapshot subprocess failed building the HIP dependency; scoped formatting and the required `fmt-check` pass. Its original failure remains at `~/tmp/nn-port-scratch/fmt-apidoc-blocker.txt`.
