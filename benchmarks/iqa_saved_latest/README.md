# Latest synchronized IQA results

The five-minute result synchronization service refreshes this directory only
from finalized saved observations on mac, i265 and r7900x. Git history retains
each published snapshot. The timestamped initial capture remains under
`benchmarks/iqa_saved_2026-10-20T0056Z/`.

Select `preferred=True` qualified rows. Preserve exact binary, input, variant,
geometry, worker, engine and cold/warm identities; never pool revisions or
protocols. Pending, refused and gate-flagged attempts remain explicit.
Only canonical scalar, one/eight-worker, non-8K observations are published.
Private metrics use the existing aggregate allowlist; no scores, private code,
model assets or raw capture paths are projected.

`sync.json` records the latest capture window and evidence identity. Hashes,
coverage and protocol records accompany the three host CSVs. A cycle with no
new measured evidence makes no commit. Source checkout ownership or uncommitted
work defers publication without changing that work. Network/validation failure
retains the previous verified export and writes a local error receipt.

See `docs/IQA_RESULT_SYNC.md` for service operation and logs.
