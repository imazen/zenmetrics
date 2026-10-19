# CLI reference-source inventory

Runtime parity and timing qualification remain **0/19**. This is a source-only
census: a route, source file or documented gate does not establish successful
execution. No metric calls, scorer imports or builds were performed.

## Missing bindings

- Every unmerged harness route remains runtime **UNQUALIFIED**. Model, weights,
  binary and source joins marked `UNKNOWN` need their stated resolver.
- `ms-gmsd`/`ms-gmsdc`: author multi-scale reference unbound; upstream search
  is not exhausted. `ws-hv`: notebook origin URL unbound.
- HDR-VDP author MATLAB releases are not mirrored. Configured jpeg-ai-qaf
  Python routes do not establish author-MATLAB equivalence.
- Staged binaries do not identify their source revisions. DSSIM's resolved
  dependency version needs its lockfile; SSIMULACRA2's libjxl route remains a
  diagnostic alternative pending standalone Cloudinary CMS binding.
- Fixed-factor SSIM k5/k6 recipes remain invalid: the reviewed
  [source patcher](../iqa_compete_2026-10-16/fixed_factor_oracle.py) accepts
  factors 1–4 and makes no metric calls. The unmerged port_spotcheck
  monkeypatch with the same basename is a different file.
- MAD helper selection remains open. GMSD's C reference has no completed
  runtime control here. The metrix MS-SSIM route still needs equivalence to
  the author's `ssim_mscale_new.m` checked.
- Missing numeric gates remain `UNKNOWN`, including CAMBI. Internal SIMD
  bit parity and observed errors are not external-reference acceptance gates.
- STLPIPS and DeepDC package pins do not establish origin-repository joins.

## Definitions

`claimed_canonical` identifies the primary reference claimed for this name.
`claimed_reference_language` describes that reference;
`harness_language_unmerged_3494d8af` describes the actual unmerged runner.
`build_or_runner_binding` and `harness_route_unmerged_3494d8af` name that
runner separately. Neither language nor route presence proves equivalence.

`role` records the relation between those identities: `primary` means the
runner claims the named reference, with its source join still unqualified;
`substitute` means the runner uses another implementation; `owner-variant`
means owner-defined semantics on a stated base. `diagnostic`, `definition`,
`self` and `excluded` retain their explicit scope. A MATLAB primary can have
a Python substitute. PIQE's primary is pypiqe 1.2 while its harness uses
PyIQA; the four WaDIQaM primary routes name Chainer deepIQA while their harness
uses PyIQA. The separately named `wadiqam-pyiqa` declares PyIQA as primary.
DSSIM's primary is Rust, with a C API shim in the runner.

`parity_gate_documented` contains existing doc/test gates only. Harness bounds
are separately labeled defaults. Evidence summaries and observed errors are
not thresholds. No numerical gate or expectation was changed.

## Method and source identities

All current-tree doc/test references bind to
`27350d07f2c3692a9d0665a81a3d0beffaa8c0e1`; the complete path list is
`gate_sources` in [inventory-source.meta](inventory-source.meta), including
actual crate tests and the NN README. The registry is
[`MetricKind` at that revision](https://github.com/imazen/zenmetrics/blob/27350d07f2c3692a9d0665a81a3d0beffaa8c0e1/crates/zenmetrics-cli/src/metrics/mod.rs).
Only the enum block, lines 97–1362, supplies names: 126 unique names from
133 file-wide value names; two LumaIngress and five other-enum values are
excluded. The retained list-metrics capture has the same 126-name set; its
full SHA256 is in the metadata.

There are **127 rows**: 124 data rows, two excluded CLI aliases (`pieapp`,
`pieapp-pyiqa`), and one non-CLI AIMOS exclusion (proprietary, unreleased).
`fsimc` is an additional FeatureSIM output, not a CLI name.

The base table is a byte-pinned evidence copy from unmerged revision
`3494d8af34839c21ff0b97f787ffd15c8a4d4ac3`. Its harness sources are evidence
from that revision, not master coverage. The generator validates the base
SHA256 before reading its rows. Source and binary hashes are distinct.

## Artifacts and verification

[make_inventory_source.py](make_inventory_source.py) emits three deterministic
TSV shards in registry order, each at most 30,000 bytes.
[inventory-source.tsv](inventory-source.tsv) records full hashes, sizes and
row ranges. Schema `inventory-source/4` adds the separate language identities.

The shared exclusive-write helper preserves differing existing artifacts and
refuses unexpected shards; it never deletes glob matches. Generation belongs
in a fresh directory containing the pinned base table. `--verify` reads only,
compares every shard, index and metadata byte, checks the exact shard set and
requires each CLI name exactly once, with AIMOS exactly once. These integrity
checks establish the inventory artifact contract, not metric qualification.
