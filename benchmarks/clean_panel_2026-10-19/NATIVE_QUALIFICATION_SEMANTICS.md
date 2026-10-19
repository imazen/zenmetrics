# Native qualification semantics — proposed contract, no execution authorization

The current fixed_look.py is byte-identical to the archived
 diagnostic-proposal-e1301b1e/fixed_look.py: SHA256
ace89cf287467dfdf0d1de1d6c61ad6052ef9118da1052091b05826747967991.
Source identity was checked by SHA256 and cmp. This document changes no code,
expectation, policy, seed, threshold or authorization.

## Assumption declaration and independent review

Before native qualification, bind a versioned, reviewed statistical contract
to the exact cohort, sampling unit, estimand, sampling mechanism, order
assignment, stopping rule, admissible resource rejection and selection rules.
Explain the independence/exchangeability and common-distribution assumptions
needed by the rank interval and permutation diagnostics, and whether the
sampling/order mechanism supports them. Review must address time drift,
within-block versus between-block dependence, order effects, rejection and
completion selection, and precision screening. Unsupported assumptions block
the relevant inference; they are not repaired by a passing diagnostic.

The target is an explicit justified model, not an empirical proof of IID.
Unconditional model coverage must not be relabeled conditional coverage of
completed/selected/precision-passing studies. Declare the population and scope
of any coverage statement. Retain all incomplete and rejected studies.

## Empirical diagnostic refusal

The frozen diagnostic function always emits iid_established=False and
selection_model_qualified=False. These describe what diagnostics establish;
neither is read as a decision input by the frozen synthetic controller.
They must not become an accidental impossible empirical pass gate or be
flipped true to authorize a native run. Assumption review is a separate record.

Preserve both permutation seeds, 99,999 permutations, inclusive >= observed
ties, plus-one probability calculation, and refusal when either p <= 0.025.
A diagnostic refusal is adverse evidence under the declared diagnostic model.
No refusal is not proof of IID, exchangeability, absent drift, or valid
selection. The tied-case proposal remains unapplied pending owner approval.

## Finite protocol and implementation readiness

Future native implementation must separately enforce and bind actual source,
binaries, input/config/scope and numerical qualification, ordered existing-lock
admission, PID/start ownership, monitor/gate-age decisions, rejection reasons,
cleanup and full immutable attempt ledger. Compile receipts alone do not
satisfy these gates. Its implementation and evidence require independent
review and explicit execution authorization before live acquisition.

Keep seed 20261019 and the existing schedule: 144 slots, 72 per order;
120 accepted blocks, 60 AB and 60 BA; at most 12 resource rejections per
order; 24-hour elapsed deadline including admission/inflight handling;
single final look with no salvage/repeated finalization. Keep the block-ratio
estimator and exact rank interval, relative halfwidth <= 0.03, and inclusive
1.08 crossing rule. Preserve every diagnostic threshold and schedule byte.

Readiness, assumptions reviewed, diagnostic refusal, protocol completion and
interval conclusion are distinct facts, not interchangeable booleans.
The current controller is synthetic-only and independently initializes
qualified=False and production_enabled=False; it has no live runner and
cannot qualify a native measurement. Plan724f7175 is build-only and its
unresolved statistical-model gate means the assumption contract still needs
review, not that IID must be empirically proved. No readiness promotion is
made here. Generic59/60 red, 0/19 and attribution_ready=false remain.
