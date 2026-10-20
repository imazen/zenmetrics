# HDR resident timing-scope promotion

`first_boundary_hdrvdp_promote.py` verifies retained files under caller-owned
shared admission. It launches no scorer, Octave or subprocess. Use a NEW output
directory; retain all candidate plans, controls and frozen predecessor bundles.

Required evidence: nine passed headless resize-2.20.1 preflights; all four warm
photo512 ABBA cases (both base metrics at threads 1/8); four actual resident-cell
protocol controls; two complete current40 native observation roots at threads
1/8. Every retained raw log, runtime before/after snapshot, scalar bit pattern,
source/input binding, terminal result and derived-batch audit is reverified.
A derived cohort is accepted by explicit path/SHA after its parent, case/catalog
bytes and required helper copies agree. The reviewed batch controller is bound
by SHA; no directory name substitutes for provenance. The batch producer needs
caller admission around its complete execution; it does not acquire locks itself.

The root producer emits one envelope per actual cell, outside its artifact root:

```json
{
  "schema": 1,
  "kind": "hdr-resident-cell-protocol",
  "passed": true,
  "returncode": 0,
  "case_id": "<candidate plan row id>",
  "metric": "hdrvdp",
  "threads": 1,
  "samples": 2,
  "command": ["<exact fully formatted candidate canonical argv>"],
  "root": "<absolute retained control directory>",
  "cell_root": "<root>/cell",
  "stdout": "<root>/adapter-stdout.log",
  "stderr": "<root>/adapter-stderr.log",
  "artifacts": {"<every recursive regular file under root>": "<sha256>"}
}
```

Set adapter TMPDIR to cell_root. Keep admission/wrapper logs outside root;
stdout/stderr above contain only adapter-forwarded process output. Include all
adapter scripts, identities, exit records, runtime/ldd snapshots and original
process logs in artifacts. Symlinks and nonregular artifacts refuse. Samples
must equal the actual requested positive count; threads must match the explicit
candidate row. An inner successful exit cannot replace the outer exit receipt.

The request is JSON schema 1 with `cohort`, `seal`, two `candidate_plans`, four
`cell_controls`, and (for a derived batch) `batch_receipt`, each a
`{"path":"<absolute path>","sha256":"<explicit SHA>"}` record. Its
`native_controls` contains two objects with `threads` (1 or 8), `root` and the
complete recursive `artifacts` map. The request itself is supplied by explicit
SHA:

```sh
python3 -B first_boundary_hdrvdp_promote.py REQUEST NEW_OUTPUT --request-sha REQUEST_SHA
```

Run this file audit through the shared admission policy. The frozen source
manifest must pass before executing the promoter. Use a NEW cell config prepared
against the exact headless parent and derived batch seal; the older non-headless
candidate config cannot satisfy this proof.

Promotion writes a new receipt and two reference plans with `matched_scope=true`
and `timing_scope_qualified=true`. This describes the clock boundary: whole RGB8
conversion, unchanged original evaluator call and temporary release inside;
input reads, original-helper parameter preload and script setup outside. Actual
cell results must match original/default/resident ABBA bits exactly.

Numerical qualification remains separate. V2 retains its existing 0.005 absolute
bound. V3 retains null bound and `native_parity_qualified=false`, with actual
native/reference scores and absolute difference recorded. An absent V3 bound
alone does not block descriptive matched-scope timing. Native resource flags are
retained verbatim as numerical observations; zero timing cells are qualified by
this promoter. Subsequent comparison attempts retain their own resource gates.

`first_boundary_hdrvdp_promote_check.py` creates synthetic receipts using real
frozen evaluator slices and seal/audit validators. Its synthetic scores exercise
acceptance and rejection, including unbounded V3 timing scope and preserved
failed native resource flags. These controls execute no metric and do not
establish numerical or performance acceptance.
