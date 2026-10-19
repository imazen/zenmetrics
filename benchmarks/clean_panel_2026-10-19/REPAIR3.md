# Archive-prefix import provenance

Candidate for independent review; no landing or acquisition. The prior P1
exceptional-cleanup repair on 3be52cc1 is independently accepted within its
tested bounds. Containment source is unchanged in this repair.

Python zipimport accepts an archive plus internal directory as a search root,
although the complete pseudo-path does not exist on the filesystem. Inventory
now finds the existing file ancestor, checks that it is a ZIP container, and
binds its bytes, resolved target and exact internal prefix. Unsupported file
ancestors refuse capture. Genuinely absent default archive paths remain explicit
absent roots; later creation changes their binding.

Actual owned venvs use executable .pth entries to discover external .zip and
.egg roots with nested internal directories. Fresh interpreters demonstrate
successful imports before and after bytecode replacement, with unchanged .pth
bytes. Verification rejects container mutation, deletion and symlink retargeting,
including retargeting to identical bytes. Ordinary root/cache assertions remain.

Before evidence is the attributed independent reviewer counterexample against
3be52cc1: VALUE changed from 1 to 2 while the archive-prefix binding still verified.
The reviewer script/log are preserved unchanged in the private evidence archive;
they are not represented as a new author baseline run. After: all 36 acceptance
tests pass via `just clean-panel-check`, including both archive extensions and
all prior containment tests. run-heavy: rc0, 21s, peak RSS 0.03 GiB, minimum
available 51351 MiB, peak load 0.46, 2 GiB hard cap, one job. These are control
resources, not metric performance measurements.

Evidence: /mnt/v/aic2026-evidence-private/clean-panel-repair3-2026-10-19/
SHA256SUMS binds the acceptance log and attributed prior counterexample;
final-review binds the immutable candidate identity and diff.

The installed canonical environment, shared producers and metric code were not
changed. No scorer, real reseal, trace or benchmark ran. Historical artifacts
remain preserved. Coverage stays 0/19, the old pilot rejected, no ratio or CI.
Runtime validation, native second-result parity, HeavyProcesses(4), remaining
18 adapters and production stopping/order/dependence remain separate blocked
gates. Existing platform and pre/post verification limits in REPAIR2.md remain.
