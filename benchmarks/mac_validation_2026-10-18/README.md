# Mac native/Metal validation — 2026-10-18

MISSING: ARM64 Fast frozen-main fingerprints in butteraugli (the test is x86-only).
Butteraugli's whole-workspace command fails its CLI harness with an external target directory.
CVVDP's existing 4096² Mode B test remains ignored by its ≥16 GB GPU gate; it was not forced.
Zensim has six failing tests; see below. DISTS completed only 2/18 golden pairs:
the resource guard stopped both combined and isolated runs at 2048².

Host=mac, Apple M4 Pro, macOS 27.0 (26A428), ARM64, rustc/cargo 1.99.0.
Zenmetrics source=`a3bc3a98`; includes fixes `c4bfe0fd`, `f29a8371`, `9b0e827c`.
A fresh clone provided an exact `git archive` build snapshot, with all sibling trees
archived at `ci/sibling-pins.tsv` revisions. `cargo metadata --locked --no-deps
--format-version 1` resolved; every Cargo test used `--locked`. No manifests,
lockfiles, metric implementations or assertions changed.

One owned target directory, incremental compilation and debug symbols disabled;
`WGPU_BACKEND=metal`, `RUSTC_WRAPPER=`, `RUST_TEST_THREADS=1`.
All heavy commands ran at nice 19 through the macOS resource wrapper with
`--mem 12G --jobs 2`. This is a monitored RSS guard, not a cgroup hard cap.
The host had concurrent work: timings are not speed evidence.

## CVVDP

Exact command for each module (in its own process):

```
cargo test --locked -p cvvdp-gpu --no-default-features --features wgpu --test it <module> -- --nocapture --test-threads=1
```

| module | pass / executed | existing ignored |
| --- | --- | --- |
| mode_b_walker_parity | 8/8 | 1 (4096²) |
| strip_mode_b_csf_halo_parity | 5/5 | 0 |
| strip_mode_e_parity | 12/12 | 0 |
| dispatch_limit | 2/2 | 0 |
| cancel | 4/4 | 0 |

All five processes exited 0, with no `cubecl-wgpu:` or DSD-thread panic lines.
Measured Full = StripPair JOD: 256² `9.457393646240234`, 512²
`9.457931518554688`, 1024² `9.458330154418945`; all printed differences 0.
At 128², Full and Mode B print `9.456145`, difference 0.
At 1000x750, Mode E and Full print identical JODs: noise 5.008305,
offset 9.458392; both differences 0.000e0, with 219 strip dispatches.
The regression gate is 1e-4 absolute JOD, rather than a bitwise assertion. Dispatch-limit JOD:
2048² `1.854582`, 3000x2000 `3.804538`.

## Zensim

Build command:

```
cargo test --locked -p zensim-gpu --no-default-features --features wgpu --test it --no-run
```

The resulting `it-99c7acf43a861a3d` binary's `--list` yielded 113 tests in
21 modules. Each module was run in a fresh process with `<module>::
--nocapture --test-threads=1` to avoid cumulative GPU exhaustion.
Result: **107/113 pass, six fail, zero ignored**. No `cubecl-wgpu:` or DSD-thread panics.

| failing test | measured result |
| --- | --- |
| diffmap_invariants::invariant_1_identity_yields_near_zero_diffmap | identity 2.3330367, expected near zero (documented failure persists) |
| cached_ref_slot_rebuild::gpu_score_correct_after_foreign_size_rebuild | CPU 48.827715234845805; GPU one-shot 51.24123255864118, cached prints 51.2412; difference 2.413517323795375 JOD vs 0.10 bound (documented failure persists) |
| cpu_gpu_diffmap_parity::gpu_diffmap_matches_cpu_canonical_pointwise | three fixture/delta cells fail; worst fixture 4, 200x160, delta 30: 0.008055568 vs 0.001 tolerance; max score error prints 0.000000 |
| odd_dim_pyramid_parity::odd_1022x818_matches_reported_corpus_dims | warm variant 0, feature 12: CPU 2.619536e2, GPU 2.608041e2, abs 1.1495, rel 0.0043883 |
| odd_dim_pyramid_parity::odd_769x513_matches_reported_corpus_dims | warm variant 0, feature 12: CPU 2.478096e2, GPU 2.452202e2, abs 2.5894, rel 0.010449 |
| odd_dim_pyramid_parity::odd_height_power_of_two_plus_one_compounds_every_scale | 1025x513 warm variant 0, feature 12: CPU 2.498342e2, GPU 2.485936e2, abs 1.2406, rel 0.0049656 |

The odd-dimension gates use absolute and relative budgets both 0.004.
These four additional failing tests were not listed in the older two-failure note;
this run does not establish when they began failing. No fix or threshold change was attempted.

[Per-process commands, status and resource-wrapper records](results.tsv).
[Butteraugli native evidence](https://github.com/imazen/butteraugli/tree/02b570b490dda0f2593aca99260ed34ab062d5b3/benchmarks/mac_validation_2026-10-18).

## ORT WebGPU / Metal

Published libraries were obtainable; no ORT source build was used:
[plugin 0.4.0](https://pypi.org/project/onnxruntime-ep-webgpu/0.4.0/#files),
[ORT core 1.31.0](https://pypi.org/project/onnxruntime/1.31.0/#files).
The plugin wheel is named macOS universal2; its extracted dylib reports ARM64.
The core dylib is ARM64. [Exact wheel URLs and dylib hashes](providers.tsv).
`ORT_DYLIB_PATH` selected `libonnxruntime.1.31.0.dylib`;
`ZENMETRICS_NN_WEBGPU_EP_LIB` selected `libonnxruntime_providers_webgpu.dylib`.
Only LPIPS-Alex/DISTS model files were copied from the source host; the ladder
was copied with its image files. Input hashes are retained with the raw logs.
The engine's source forbids CPU fallback; the native plugin ran against Metal.

Command:

```
just nn-parity "$MODEL_DIR" "$LADDER" ort-webgpu model-tests,ort-webgpu lpips_alex,dists
```

The exact pinned source/dependency snapshot was already prepared, so its
scratch-only `snap-cargo` recipe was replaced with `cargo --locked {{args}}`.
This avoids a second snapshot/target tree; `nn-parity`, the test source and all
model goldens remained unchanged. `CARGO_PROFILE_RELEASE_DEBUG=0` bounded disk use.

The combined command exited 137 during DISTS 2048² after all 18 LPIPS pairs
passed. Each model was then run in its own process via the same command with
its single model name, under the same 12 GiB guard:

- **LPIPS-Alex: PASS**, all 18/18 golden pairs (64²–2048², three references),
  worst absolute error 1.787e-7 = 2.74e-6 JND, plus warm-reference/cold parity.
  The four-test binary exits 0; one test is the unrelated DMM not-selected
  path, and one is a negative-control panel test with intentionally caught panics.
- **DISTS: INCOMPLETE**, 2/18 golden pairs completed in both attempts:
  1415/1024 score 0.180054665 vs 0.180055797, error 1.132e-6;
  1415/128 score 0.227918506 vs 0.227918565, error 5.933e-8.
  The isolated process again stopped at the next pair, 1415/2048, with exit 137.
  The wrapper reported min-avail 3712 MiB (its free+inactive+speculative-page
  proxy fell below 4096 MiB). Peak RSS was unavailable because it killed the
  process group. No claim of an ORT error return, full DISTS parity, or warm/cold
  DISTS parity follows from this interrupted run. The guard was not raised.

[ORT results](ort-results.tsv). LPIPS's wrapper record:
`run-heavy: done rc=0 3s | peak-RSS 0.34GiB | min-avail 13312MiB | peak-load 30.41`.
Elapsed values are retained as raw resource records, not speed comparisons.

## Retention and cleanup

Raw logs, status files, model/library hashes, pinned sibling revisions and runner
commands were copied off mac. Resolving command on host **dev**:
`ls /mnt/v/output/zenmetrics/mac_validation_2026-10-18/raw` (checked 2026-10-18).
Three retained log hashes:

```
959edf324004048ca97dbf95b0cf50f0e0f566ede462068cd8380f2b61bcdc2b  butteraugli-exact.log
eee76c57107c42615cc174556648cbfbb5f6fda1cfcfa03494c88d003dfa62b0  zensim-cached_ref_slot_rebuild.log
3f10f2620ce8634be8d2c9fe03a8a51d07faea5baaa8def592cda3bc260c2619  ort-dists.log
```

The owned Cargo target directory was removed after all validation processes
finished. `test ! -e "$HOME/tmp/mac-validate/target"` passed. mac's `df -k /`
then reported **13,335,312 KiB available**. No existing checkout or other
session's targets/data were removed.

## DISTS isolated one-job retry (2026-10-18)

At the user's request after reported reduced host load, retried only DISTS,
with the same source snapshot `a3bc3a98`, ORT core 1.31.0, plugin 0.4.0,
model files, goldens and Metal backend. No other metric was rerun.

```
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
run-heavy --mem 12G --jobs 1 -- \
just nn-parity "$MODEL_DIR" "$LADDER" ort-webgpu model-tests,ort-webgpu dists
```

The existing pinned scratch snapshot's direct `cargo --locked` delegation
was reused. Compilation and test execution shared one resource-wrapper run.
**Result: exit 137 at the next 1415/2048 pair, 2/18 pairs complete.**
Largest measured passing size: **1024x1024**, score `0.180054665` vs golden
`0.180055797`, absolute error `1.132e-6`. The other completed pair was
1415/128, score `0.227918506` vs `0.227918565`, error `5.933e-8`.
Warm/cold parity was not reached. No further tests were attempted.
The intentionally caught DMM panel negative controls print panics and pass;
they are unrelated to the DISTS memory-guard termination.

Resource record:
`run-heavy: done rc=137 146s | peak-RSS unavailableGiB | min-avail 3696MiB | peak-load 44.60`.
The monitored available-page proxy again fell below the 4096 MiB floor.
Peak RSS is unavailable after the process-group kill; no memory peak or
speed claim is inferred. The 12 GiB guard was not raised.

Raw log: `ort-dists-retry-one-job.log` in the retained raw-log directory
on dev described above; SHA256 `edafe22e1a657d767101425a1088ddedfe5026dc2314405f63ed4681e24290a3`.
The owned Mac target was removed again and confirmed absent. After cleanup,
`df -k /` reported **10,119,616 KiB available**.
