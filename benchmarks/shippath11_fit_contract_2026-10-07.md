# SHIPPATH11 qualified-fit harvest contract — 2026-10-07

The D1 profile carries a content-bound `shippath-qualified-fit-contract-v1`
record from zensim. Harvest requires the program archive named by the trusted
manifest, verifies its SHA-256, and reads its contract and binary inventory.
The declared data SHA must match the contract. The supplied canonical model
inspector must match the program's binary hash.

Expected budget, final epoch, exact four-source route, consumed IDs, seeds,
receipt/freeze/decision hashes and seven table admissions come from that pinned
contract. Result claims must match. The canonical model loader decodes selected
and production-packed metadata; requested/actual epochs, checkpoint epoch,
pair budget, sampler, seed streams, input table hashes and per-table feature,
formula and decoder declarations must agree. Lexical admission paths are mapped
only through the executor's manifest-hash extraction identity; a different data
hash refuses.

Explicit `--local-smoke-budget 2:128` changes argv/job identities and results
carry `execution_contract=local-smoke`. These jobs have separate smoke cell
names. `--allow-local-smoke` permits verification only: CLI installation and the
lower install owner refuse such results. Registered E30 and production manifests
retain 120 epochs, 50,000 pairs/epoch and final epoch 119.

For the new D1 jobs, use `harvest_fit_cells.py --program-archive <program.tar.gz>
--checkpoint-inspector <matching inspector>`, in addition to the existing
manifest/IDs/ledger/blob arguments. Do not use `--allow-local-smoke` for registered
fits. The preparation artifacts include the exact tools and inspector pins.

The E28 runtime driver at `/var/tmp/fitv2/harvest_driver_v2.py` delegates to
`/var/tmp/fitv2/tools/scripts/jobsys/harvest_fit_cells.py`. Inspection confirmed
that existing verifier checks receipts/file hashes and prediction counts but
has no registered epoch/pair-budget or decoded checkpoint-admission binding.
The driver's already-installed fast path compares identity fields only. Neither
runtime file was changed or executed by this lane; E28 harvest needs separate
coordinator authorization before changing its running owner.

Local artifacts: `/mnt/v/output/zensim/shippath11-2026-10-07`. The worker is built
from master `f26c61cb` in the CI-pinned 23-sibling snapshot, including the reviewed
claim-ownership fix. Its byte-derived build ID is `925f9783329f`;
`WORKER_BUILD.json` records the full binary/source/lock identities. The image is
`ghcr.io/imazen/zenfleet-worker:fit-d1-e30-v39-w925f9783329f`.

Validation includes actual installed `fit-cell-exec` entry invocations on fresh
scratch, explicit two-epoch/128-pair fits, checkpoint decoding, canonical
production densify/f16/TRAIN calibration, and reviewer negative controls with
recomputed receipt hashes. Full registered fits remain unrun; nothing was
published or enqueued. E28 remains first and launch authorization is required.
