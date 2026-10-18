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

Resource wrapper receipts (scope observations, not metric-memory comparisons):

```
Mac timing build: run-heavy: done rc=0 15s | peak-RSS 0.97GiB | min-avail 15653MiB | peak-load 4.90
Mac final recovery controls: run-heavy: done rc=0 78s | peak-RSS 0.03GiB | min-avail 17230MiB | peak-load 2.85
Linux integrated controls: run-heavy: done rc=0 79s | peak-RSS 0.03GiB | min-avail 24686MiB | peak-load 0.44
```

Mac `min-avail` is the wrapper's reclaimable-page proxy. Its RSS guard is
monitored; it is not a cgroup hard limit. See the [execution contract and commands](../../docs/IQA_DURABLE_MAC.md).
