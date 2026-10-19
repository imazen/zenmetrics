# Primary Octave FSIM/VSI adapter

[octave_primary.py](octave_primary.py) executes the authors' MATLAB FSIM/VSI
releases through Octave. Its default selects verified scratch copies produced
by [the fixed-factor patcher](fixed-factor-oracle.md), replacing only the factor
expression with k in 1–4. `--automatic-factor` instead selects the registered
original source. This is a benchmark reference, outside shipped crates.

Missing: qualified paired ratios and full runtime call-graph closure.
The completed numerical matrices are recorded in the
[result report](../port_spotcheck_2026-10-18/vsi-resize-runtime.md).
Successful Octave execution does not establish equivalence
to every MATLAB/toolbox release. No numerical gate changes here.

The adapter checks manifests and exact RGB8 input sizes before launching a fresh
process. It hashes inputs, source, adapter and executable, asserts the selected
function path, and records the direct image-package/compiled-builtin bindings.
Before/after source, input and executable hashes must agree. Nonfinite reference
scores are preserved as findings. The caller supplies both heavy-job locks and
`run-heavy`; output goes to a new evidence directory, never over an old attempt.

File reads, package loading and reusable raw buffers precede the clock. Each
call reshapes resident RGB8 into the canonical channel layout, calls the author
function, and releases the per-call images before stopping the clock. The first
call emits `COLD`; later calls emit `SAMPLE`. FSIM returns both FSIM and FSIMc.
These clocks are diagnostic until scope and acquisition qualification completes.

This headless cohort clears inherited display variables and passes
`--no-window-system`. OpenMP/BLAS environment budgets and FFTW's explicit thread
budget are recorded. Requested budgets do not prove parallel utilization.

```sh
just --justfile benchmarks/iqa_compete_2026-10-16/oracle.just octave-primary \
  <patched-source.manifest.json> <reference.rgb> <distorted.rgb> \
  <width> <height> <threads> <warm-samples> <fresh-evidence-directory>
```

## Diagnostic execution, 2026-10-19

Host: lilith-wsl. Octave 6.4.0; image package 2.12.0. The pinned synthetic
64×64 RGB8 reference/distorted pair is shared with the primary SSIMULACRA2
ingress controls. Input SHA256 values:

- Reference: `87ea49834ed53138995f3ac6845891f7693b0e279dc8195ce63a30edb551e1c4`.
- Distorted: `bb6fdb0f4aa7299e311278ac892d220798068806560767bc67c43732bb1af3d6`.

Each factor uses its own verified source copy and fresh process at one thread.
First-call and warm scalar strings agree within every case:

| Factor | VSI | FSIM | FSIMc |
|---|---|---|---|
| 1 | 0.99670084799370584 | 0.99505540469866183 | 0.99231289308386017 |
| 2 | 0.99745523062338892 | 0.99572052901144292 | 0.99463919221490527 |
| 3 | 0.99936543709759651 | 0.99993836123854674 | 0.99946167058291235 |
| 4 | 0.99974683281214294 | 0.99996854152686365 | 0.99968696922506262 |

The serialized eight-case panel reports
`run-heavy: done rc=0 4s | peak-RSS 0.07GiB | min-avail 12852MiB | peak-load 3.41`.
These are whole-panel resources, not per-metric memory measurements or ratios.
Full commands, source manifests and per-case records live under
`~/tmp/portcheck-octave-primary-20261019T1001/`; the complete panel log is
`~/tmp/portcheck-octave-primary-20261019T1001-factor-panel.log`.

An earlier display-inheriting VSI attempt took 103 seconds as a whole process,
with the same scalar `0.99670084799370584` on its first and warm call. It
reported `peak-RSS 0.06GiB | min-avail 12918MiB | peak-load 2.85`, rc0.
Startup was observed waiting on a socket before scoring. The explicit headless
attempt finishes in one second, with the same scalar; no metric speed ratio is
inferred from those process durations. A failed help query and its termination
workspace are preserved outside the repository. Scratch directories must exist
before Octave starts; its help formatter needs them even for metadata queries.

## Partial Rust parity

The [frozen-Rust probe](probe_octave_parity.py) joins an actual executable
inventory, build receipt, canonical execution record, factor manifest and input
hashes before comparison. It calls the existing harness's untimed `probe_only`
path, which checks repeated Rust values for bit equality. It inherits the
existing base gates (FSIM 5e-5, VSI 1e-4) for the owner-defined fixed variants.
It does not acquire timing samples or infer a ratio from canonical clocks.

The eight 64×64 factor cases pass at one thread on this control pair, including
both FSIM output columns. Maximum absolute differences are
9.231485020855246e-9 for FSIM/FSIMc and 8.306581955475068e-5 for VSI.
The [partial scalar table](../port_spotcheck_2026-10-18/parity-fsim-vsi-64.tsv)
retains each observed delta. This is one pair, not the requested content grid.
The whole serialized probe panel reports
`done rc=0 1s | peak-RSS 0.02GiB | min-avail 13122MiB | peak-load 3.82`.
Run `oracle.just octave-rust-parity` with the eight explicit arguments listed
by `just --show`; preserve full output under both locks and the resource wrapper.
This initial eight-case control remains a separate historical cohort.

The subsequent [real-photo crop failure](../port_spotcheck_2026-10-18/vsi-photo-finding.md)
shows why source dependencies need explicit bindings: image 2.12 interpolation
differs from the port's claimed antialiased convention. Selecting unchanged
upstream image-2.20.1 resize M-files resolves the focused photo discrepancy
within the unchanged gate; see the linked report for its full panel and runtime
limits. The optional `--measure-memory` flag retains GNU time process peak RSS
for each canonical/Rust child; explicit `oracle.just` memory recipes enable it.
These process peaks do not supply matched speed measurements.

## Automatic factors

The automatic panel uses the factor-1 manifest as a registry/integrity carrier,
but executes its **original** source path and checks the registered before-patch
SHA256. The unused fixed copy remains checked and is explicitly labeled as
unused. Records distinguish `factor_mode=automatic`, the selected original path,
and the resolved factor. At 3840×2160 that factor is 8. Rust probes select the
unsuffixed `fsim` or `vsi`, and retain their existing numerical bounds.

`oracle.just octave-automatic-panel` accepts inputs, reference map, frozen Rust
build, fresh output, thread selection, and the explicit image archive. The
controller supplies both heavy locks and run-heavy. `octave-factor-mode-check`
rejects source-mode confusion, unknown modes and wrong resolved factors before
the Rust executable or evidence directory is opened. No automatic mode changes
a reference source or an existing metric implementation.
