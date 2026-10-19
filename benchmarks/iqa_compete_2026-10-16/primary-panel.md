# Serialized primary FSIM/VSI parity panel

[run_octave_panel.py](run_octave_panel.py) runs the existing primary-reference
and frozen-Rust adapters over an explicit input/reference manifest. This is
numerical parity and process-memory acquisition. It produces **no qualified
speed ratios** and does not complete the broader CLI study.

The current input grid has three pairs at four geometries. Factors 1–4 for
FSIM/VSI and budgets 1/28 produce a planned 192 cases. On this host, 28 is the
resource policy's available-thread budget (32 eligible CPUs minus four).
The controller resolves and records that limit; it does not label it 32-thread
execution or prove observed parallel utilization.

```sh
just --justfile benchmarks/iqa_compete_2026-10-16/oracle.just octave-panel \
  <input-pairs.json> <reference-manifests.json> <frozen-rust-build> \
  <fresh-output-directory> 1,all > <full-log> 2>&1
```

The controller locks and resource-wraps **each child**. Do not hold either
heavy-job lock around the controller itself. It runs the canonical process,
then Rust, retaining the unchanged documented gates. A numerical failure
does not prevent the other requested cases from running. If the canonical
process fails or returns nonfinite scores, an independent Rust probe still
records that arm's outcome. A deliberately failing oracle control verifies
this path without pretending it is a method's numerical result.

Sources, helper hashes, input identities, executable identities and budgets
form a frozen panel identity. Every case retains its full logs, commands,
comparison and artifact hashes. `octave-panel-resume` requires the same
identity and verifies completed artifacts before retaining them. Partial
attempts move intact under `partials/` before a new attempt; no earlier data
is overwritten. One-case controls preserve the known VSI failure, retain a
completed case without rerunning it, and preserve a copied partial attempt
byte-for-byte before retrying. Their overall exit remains nonzero because the
unchanged numerical gate fails.

## Deadline and RSS layering

The shared `supervision.scoring_command` makes GNU time the observer outside
the deadline, with the scorer directly inside `timeout`. The existing timeout
and 20-second kill grace remain unchanged. GNU time then retains the command's
peak-RSS receipt, including the observed scorer child, while the deadline can
terminate that child directly. No timing sample includes wrapper startup.

`oracle.just oracle-memory-timeout-check <fresh-evidence>` uses a real,
signal-resistant nonmetric child. The deliberately reversed ordering returns
with that child alive; the test cleans only its exact PID/start identity.
The production ordering returns after the existing kill grace with no live
child. The control reports
`done rc=0 22s | peak-RSS 0.01GiB | min-avail 14134MiB | peak-load 1.29`.
All five existing supervision tests also pass unchanged. These checks cover
the new primary adapters, not generic acquisition-runner qualification.

An earlier pipe-based diagnostic kept its observation open until a child
closed the pipe. It is retained as confounded evidence and supplies no
containment acceptance. The file-backed control above measures parent return
and checks child liveness separately. Exact controls and hashes are in
[the panel metadata](../port_spotcheck_2026-10-18/primary-panel.meta).
