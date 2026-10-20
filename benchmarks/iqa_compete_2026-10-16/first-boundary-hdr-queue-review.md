# HDR per-arm source review, 2026-10-20

Missing: retained-only selection does not receive the admitted worker's artifact
audit, and its summary validator does not require finite explicit score bits.
These findings concern reuse of retained observations. They do not establish a
failure in newly executed arms or invalidate retained numerical observations.
Active frozen cohorts and helpers must remain unchanged.

The tracked successor now fixes both paths. Every retained arm dispatches a
read-only admitted `audit` operation, including terminal single-arm reuse.
That operation runs `context`, `checked_seal` and `verified_observation` on the
retained prefix. A terminal audit also reconstructs the complete ABBA case and
requires its retained receipt to agree. Source and preflight closure are checked
again before writing a new audit receipt. Passed summaries must contain finite
explicit f64 bits. Existing evaluator execution and numerical limits are
unchanged. This tracked successor is separate from the active frozen v2 sources
whose hashes appear below.

## Actionable findings

1. `first_boundary_hdrvdp_arm_queue.py:242` accepts an existing arm after checking
   summary identity fields. `checked_case`, at line 199, hashes arm result JSON
   but does not validate its artifact map. A wholly retained selection, including
   a terminal single-arm selection, never calls the admitted worker. Changed raw
   logs, snapshots or preflight artifacts therefore need not prevent successful
   completion. Fresh arms call `context`, `checked_seal` and
   `verified_observation` inside admission; a subsequent fresh arm also checks
   its predecessors. A future successor should dispatch an admitted read-only
   audit for retained selections, reusing those same validators and recomputing
   ABBA from original stdout. No scorer rerun is necessary for that audit.
2. `checked_case:215` collects any `score_bits` associated with `passed=True`.
   Four null values satisfy its equality gate. Fresh worker `scalar_bits` parses
   finite decimal values and checks their exact f64 hexadecimal encoding, so this
   is a retained-summary validation defect. The successor must require a
   16-character hexadecimal finite f64 value for every passed arm, recomputed
   from raw output, before applying exact equality.

The new `first_boundary_hdrvdp_queue_review.py` exercises these two paths with
synthetic files only. It never starts a worker, Octave or a native scorer. Its
admitted execution receipt belongs with the private review output, separately
from metric or timing qualification.

That old-source control passed: an altered synthetic raw artifact and four null
bit fields were accepted by the old summary, reproducing both defects. All
38 selected commands formatted explicit 1/8 budgets. Wrapper receipt: rc=0,
0 seconds, peak RSS 0.02 GiB, minimum available 15238 MiB, peak load 2.41.
The result SHA256 is
`50cb763c6c62f3f0fbc09f643a22a4b72813463c69f10caaf85a5557e5468006`.

`first_boundary_hdrvdp_queue_audit_check.py` supplies a separate file-only
successor control: valid retained ABBA, changed raw log, null bits, NaN bits,
changed preflight artifact and changed seal. It calls the actual audit function
with a synthetic complete source/plan/receipt topology and never invokes an
evaluator. Rejections must occur on the successor while the old summary accepts
the same damaged fixtures; these controls do not qualify real metric scores.

## Fresh-arm admission and source stack

No concrete fresh-arm admission/hash/exact-scalar defect was found by this
source review. Each dispatch acquires both shared locks and enters run-heavy
before the worker checks parent sources, input bytes, sealed preflight closure
and earlier arm artifacts. Before/after runtime snapshots and post-call source
checks remain admitted. The wrapper returns before the coordinator submits the
next arm. Missing terminal output or a source/runtime error refuses continuation;
there is no automatic retry. This is source evidence, not an OS admission test.

The reviewed source commit is `d5fb47e3ae846f3d82af85c5fccc31d75704fcd7`:

| File | SHA256, matching active v2 frozen copy |
| --- | --- |
| `first_boundary_hdrvdp_arm_queue.py` | `3f4c9b057fd179a4e253f4f51b700733cc9698fe49a1d66e6e784d3dcffc2fcb` |
| `first_boundary_hdrvdp_arm_worker.py` | `a85c29b7659df87a21fdc85a76f87f3ee00dbd508ba026eb5d658c19eff6e6fd` |

Predecessor commits: admitted golden driver `b98cec13`, resident design
`4d84bef4`, byte-preserved resident fixture `20dcbafb`. The current40 native
binary remains source `ea767ba1a4029963bc56a74fb824e64455e4599e`, SHA256
`19898e546917c9b62badcf1733ec74aacb3e02b4e14d88af1e31923a88a54005`.
No build, inference or numerical acceptance follows from this review.

## Comparative thread budgets

The primary `prepare_ordered_canonical.py` selects a minimum-thread source case
then changes top-level `threads`. The 38 selected current reference commands
each contain one explicit standalone `{threads}` argument. The frozen current40
`compare_canonical.py:64` uses the case count for native config, environment and
canonical format substitutions. Thus declared 1/8 budgets agree for this
selection; actual runtime thread use was not measured here. Missing exact-thread
source rows are not a defect for an explicitly templated callable.

A future preparer can prefer exact matching geometry/input/phase/source/thread
rows while allowing declared `{threads}` callables. A fixed command or a command
whose budget lives in a bound config must require that exact budget and matching
config rather than changing only the outer case. No primary preparer, justfile,
current default or existing numerical gate was changed.

## Separate overhead observation

Root reports a 901-second preflight wrapper with 75–96-second child elapsed
times and 0.07–0.13-second child CPU times. This remains unprofiled fixture
overhead. No attribution to Windows/network paths or metric cost is established.
Startup source investigation is secondary to admission/correctness review.
