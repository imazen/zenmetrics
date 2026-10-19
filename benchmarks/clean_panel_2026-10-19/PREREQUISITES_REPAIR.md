# Prerequisite repair candidate: source and synthetic review only

This candidate remains unqualified. No scorer, metric build, real reseal,
acquisition, public push or landing was performed. The old pilot remains
rejected: 0/19 qualified, with no real timing ratio or confidence interval.
Previously accepted containment and archive-prefix repairs are preserved.

## Snapshot attribution

`gate_trace.py` schema 2 separates unique monitor snapshots from unique checks.
A snapshot may serve multiple checks with independently verified ages. The
reviewer counterexample uses one snapshot at 100 and checks at 120/130.
Duplicate checks, snapshot redefinition, missing joins and hash/count/identity/
age disagreement refuse. `verify_artifact_join` checks actual file bytes and
exact emitted-event membership. Neither validator grants attribution readiness.

Future instrumentation must assign an ID at each monitor refresh, emit its
immutable snapshot once, and record each check's own ID, checked time and the
ID/hash of the cached snapshot it actually read. It must not force refreshes,
resample to reconstruct historical state, or discard reuse. Runtime source,
binary, configuration and outer ownership receipts must join those events.
No shared monitor/runtime instrumentation is changed here.

## Matched observer boundary

The new protocol puts paired result extraction, finite/column checks, TLS state
lookup, bit-stability recording and count updates inside both timing clocks.
Serialization and final stability refusal occur outside. This is a matched
semantic boundary, not a claim of identical overhead; no subtraction is used.
Synthetic routing checks exercise the actual patched native closure and distinct
7/19 outputs with one metric call; the canonical helper checks clock ordering.

`observer_cohort.py` stages an isolated source overlay; build receipt generation
follows successful compilation and binds source, command, features, toolchain
and binary. The sealer only declares the v2 native protocol with that receipt.
Old binary markers cannot qualify as this cohort. Full metric compilation and
runtime qualification remain prohibited and unperformed. Existing PyIQA adapter
wiring is a prototype, not the declared PIQ oracle: see
[oracle/native contract](ORACLE_NATIVE_CONTRACT.md).

## Fixed-look implementation

`fixed_look_policy.json` freezes 120 accepted blocks, 60 per order, 72 available
slots per order, at most 12 resource rejections per order, and a 24-hour cap.
`fixed_look.py` is disabled by default and accepts only closed synthetic runner,
clock and study-book classes. There is no live runner. Synthetic accepted rows
represent already classified fixtures, not actual scalar/resource admission.

The schedule uses seed 20261019, xorshift64* shifts right12/left25/right27,
64-bit wrapping, multiplier 2685821657736338717; each low bit chooses an AB/BA
pair. Diagnostic seeds 20261020/20261021 use rejection-sampled Fisher-Yates
permutations, 99,999 permutations, >= ties and (b+1)/(B+1) p-values. Order
median-log difference and absolute lag-one correlation reject at p<=0.025.
Passing screens does not establish independence or a common median.

Only a complete balanced cohort receives one synthetic fixed look: exact
binomial ranks R49/R72 at n=120, inclusive ties, 3% relative half-width and
inclusive crossing of 1.08. Completion does not guarantee precision. Incomplete
cohorts emit no interval. Duplicate blocks, malformed/nonpositive timing, fatal
failures and exhausted budgets stop without salvage. STOP and elapsed>=cap
apply before launch, during admission and after in-flight cleanup. Unknown
cleanup never emits a release event. The study book preserves every study,
including incomplete ones, and refuses overwriting its full-report export.
It is an in-memory synthetic ledger, not a durable production admission registry.

The new schedule fixture has a known failing hand-written expectation. It is
intentionally unchanged pending explicit owner approval. A separate immutable
proposed patch and independently computed bit-vector derivation are retained
outside this candidate; they are not applied. The entire suite remains red.

IID/common-median assumptions, dependence, rejection, completion and precision
selection require independent model review before collection. Conditional
coverage after these screens is not established by the unconditional binomial
calculation. No selected-subset 95% claim is authorized.

## Lock and native-route prerequisites

`admission_receipt.py` records existing caller-held kernel flock proofs and a
verified direct child identity/argv. It never acquires/releases locks or launches
children. Integration must occur inside existing admission and preserve the
accepted PID-safe containment-before-release path. Owned-child controls do not
prove old lock ownership retrospectively.

Native proposals separately bind the original Butteraugli and libjxl executable
roots, inputs, source/build/controller receipts and loader configuration. They
do not filter dependencies out of the Python graph or grant scope qualification.
Libjxl ingress/lifecycle and Butteraugli norm pairing remain unresolved; current
source/binary observations do not bind historical rows. No metric route changes.

The corrected supervisor MLIR note establishes both filenames' presence with
distinct inodes, each link count two. It does not establish byte equality or
native loading closure. The general census remains supervisor-owned.
