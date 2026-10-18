# Durable Mac CPU runner qualification — 2026-10-18

The Mac ARM64 worker and standalone bundles are built and runtime-checked. The
complete 82-route performance matrix has **not** been run by this qualification.
The runner accepts a finished measurement only after validating its frozen cell,
input and binary identities, timing phase, finite score bits, samples and process
cleanup. This is protocol qualification, not independent metric-oracle parity.

## Frozen artifacts

Two local bundles are retained on the validation Mac and copied to the coordinator
under `~/benchmark-bundles/`:

| Bundle | CPU routes | Cells | Manifest SHA-256 |
|---|---:|---:|---|
| `iqa-mac-20261018-all` | 82 | 9184 | `64c2791d30900922026899988b1c505b2b53411ad947b3862d665db91355e9e1` |
| `iqa-mac-20261018-core` | 14 | 1568 | `d8588fc41d242f3cf8bbf3dc44874d138af01f264688de8157d5f95713cc3bfc` |

Each includes the ARM64 binary, scripts, exact input files, build receipt,
resource wrapper, input provenance and checksum manifest. `select_durable.py`
creates other subsets without Cargo or a repository checkout. The core set is
PSNR, SSIM, SSIMULACRA2, Butteraugli, Zensim, CVVDP, IW-SSIM, GMSD, MDSI,
HaarPSI, FSIM, VSI, MS-SSIM and NLPD. NN routes, including PieAPP, are absent from
this build. Named variants remain separate entries in the full inventory.

Worker logical source: `4b4f5b16b7063448819f5176cc654c98bcdaf277`; features
`compete,metrics-parallel`, release build, unchanged lock and sibling pins.
`build.json` records compiler identity, exact source-file hashes, binary SHA-256,
actual Mac snapshot and dynamic-library inventory. `otool -L` lists only system
libraries/frameworks. No `target-cpu=native` flag was supplied.

The S67 scaling inputs retain the existing `inputs.meta.json` hashes. 4K/8K are
mirrored derivatives. Thread budgets are explicitly 1 and 8. Each mode has three
fresh-process cold cells and one ten-round warm cell. This panel does not provide
content diversity or a perceptual ranking study for AIC2026.

## Executed gates

- Private Rust timing-boundary controls: 2/2; scorer construction occurs inside
  the cold clock and only one first evaluation is made.
- Actual Mac worker controls: 7/7; PSNR and GMSD cold/warm score bits agree,
  GMSD scalar/map modes run independently, and unsupported/refused/nonfinite
  results have no timing. Explicit warm phase requires ten rounds.
- Linux integrated Python suite: 150/150 before the final subset helper;
  the final bundle/service/subset suite separately passes 16/16.
- Final actual Mac recovery suite: 21/21. Includes controller SIGKILL, SIGSTOP,
  watchdog SIGKILL, worker deadline despite continuing output, grandchild cleanup,
  nested-session wrapper, bounded retries, untouched-before-retry scheduling,
  SIGTERM during admission, changed inputs/binary/results, malformed/partial rows,
  duplicate controllers, retained completed output, and live-group refusal.
- Real macOS launchd SIGKILL control: controller killed during GMSD execution;
  launchd restarted it; all eight cells completed. The interrupted first cell
  retained two attempts. No manual restart was issued.
- Real macOS launchd SIGSTOP control: controller frozen; missing heartbeat caused
  its service to kill and restart it; all eight cells completed without SIGCONT
  or manual restart. A finished first attempt was retained without remeasurement.
- Final packaged subset: 16 cells terminal, 12 successful timing rows and four
  explicit unsupported PSNR-map rows. Nine cold and three warm rows; each
  metric/mode's cold and warm scores are bit-identical.

The launchd controls used controller source `b52aee59`, content-equivalent to
`8fb573aa`. The final packaged smoke and 21-test Mac suite include subsequent
resume-priority changes `6dcbc122`. Both test services were unloaded and their
owned test plists removed. No long performance run was left active.

Complete logs, attempts, test-service receipts, build metadata and original
control bundle are retained at `~/benchmark-artifacts/iqa-mac-20261018/` on the
coordinator. Original Mac receipts are under `~/tmp/iqa-mac-durable/`. These are
local artifacts, not committed model weights, images or generated dumps.

## Sequential live run, observed 2026-10-18 10:27 UTC

The user started the core bundle after the qualification above. Remaining
families are now installed as dependent launchd services, using the shared Mac
heavy lock. All seven sizes and both thread budgets remain selected.

| Order | Bundle suffix | Methods | Cells | Manifest SHA-256 |
|---|---|---:|---:|---|
| 1 | `core` | 14 | 1568 | `d8588fc41d242f3cf8bbf3dc44874d138af01f264688de8157d5f95713cc3bfc` |
| 2 | `rest-v2` | 71 | 7952 | `e74f2e4d8cf4a80855b0986db9b224ded1631c2542e170a03866e53f1ba63da2` |
| 3 | `nn-cpu-v3` | 23 | 1288 | `785ac3c6c01d2528e5ccf792fe1f132a42aea867b897b093016660eec7b9bdd5` |
| 4 | `nn-metal-v3` | 23 | 1288 | `1b6e5c1b62e1826ff1b94837b5610ce10e5340d8e6636049e5a4943ade0a1954` |

Bundle names begin `iqa-mac-20261018-` under `~/benchmark-bundles` on Mac.
Outputs are `~/benchmark-results/iqa-core`, `iqa-rest-v2`, `iqa-nn-cpu-v3`
and `iqa-nn-metal-v3`. Successor services were observed waiting on the correct
predecessor, with no measurement output created early. Superseded waiting
services were disabled and unloaded; their bundles and receipts remain retained.
Recheck state with each bundle's `mac_service.py status BUNDLE OUTPUT` and the
corresponding `.service/queue-wait.json`; this table records installation, not
completion of these panels.

The second build enables `compete-nn,metrics-parallel,zenmetrics-cli/nn-webgpu`
at worker source `7310d24c`; binary SHA-256 is
`0773cd747da14a2f535719810cbfe9b0f93e9ee01dc1491c330e0755661bd4f9`.
It includes PIQE, NIQE and BRISQUE in the non-neural set, bringing that family
to 85 methods. AHIQ and DMM are assigned to the neural family. PieAPP is excluded
by the existing explicit-opt-in rule, so NN coverage is **23/24**, with scalar
cold/warm cells and no invented native maps. Exports, heads, external tensors,
ORT core 1.31.0 and WebGPU provider 0.4.0 are supplied locally and hash-frozen.
Runner hashes are also bound into the new manifest identities (`82161fc3`).

An interrupted NLPD warm attempt left a stale zenbench running reservation.
Its worker and both retained process groups were absent; recovery acquired the
ticket guard, archived the original bytes and kept a sibling backup. Subsequent
warm cells completed. Failed attempts remain in the original output, including
those awaiting retry. Exhausted-cell counts alone do not count failed attempts.
The queued controllers include worker identity recorded before exec and
post-cleanup recovery (`dcf93f35`). Linux controls pass 23/23 for that change,
bundle/service controls 17/17 and admission controls 14/14; the added frozen
runner rejection control passes separately. The 14 admission controls also
pass on Mac. No acceptance threshold changed.

Real LPIPS-Alex 512x512 controls on the updated controller complete one cold and
one ten-round warm cell per engine. Within-engine score bits match exactly:
RTEN `3f6f725bf4bbb08c`, ORT WebGPU/Metal `3f6f725c80000000`. These controls do
not establish parity or timing acceptance for the other neural models. Original
blocked controls are retained beside their successful replacements.

Installation commands, manifests, original failed controls, successful control
reports and the admission archive remain under `~/tmp/iqa-nn-sequence` on Mac;
operator receipts are copied to `~/benchmark-artifacts/iqa-mac-sequence-20261018`
on i265. Source commands are reproducible through `just bench-iqa-mac ...
install --after ...` and `just iqa-nn-timing-check BUNDLE NEW_OUTPUT`.

Resource wrapper receipts (scope observations, not metric-memory comparisons):

```
Mac timing build: run-heavy: done rc=0 15s | peak-RSS 0.97GiB | min-avail 15653MiB | peak-load 4.90
Mac final recovery controls: run-heavy: done rc=0 78s | peak-RSS 0.03GiB | min-avail 17230MiB | peak-load 2.85
Linux integrated controls: run-heavy: done rc=0 79s | peak-RSS 0.03GiB | min-avail 24686MiB | peak-load 0.44
```

Mac `min-avail` is the wrapper's reclaimable-page proxy. Its RSS guard is
monitored; it is not a cgroup hard limit. See the [execution contract and commands](../../docs/IQA_DURABLE_MAC.md).
