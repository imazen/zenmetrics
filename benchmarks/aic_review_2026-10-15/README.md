# Round 5 review fixes

The review fixes are on `aic-more`, with no rebase or master merge.
Published-column discrepancies remain those measured in the
[Round 4 evidence](../aic_more_2026-10-15/README.md); this review does not establish
new published conventions.

CAMBI's zero score is normal for an unbanded photo. Its bogus-output check now
has no identity bounds and does not reject a constant batch. Empty and nonfinite
checks remain active. The all-zero sweep regression failed before the fix; all
11 bogus-output tests pass with `--features sweep`. Without that feature the
binary test module does not compile into the test harness.

The standalone CAMBI stride test now runs with no features and with
`simd,avx512`. Active row samples go straight into the existing normalized image
buffer, with no intermediate packing Vec. Nonzero 10/12/16-bit controls and the
8-bit control assert packed/strided bit equality; padding deliberately contains
out-of-range samples to catch accidental padding reads.

## Original-author pins

[pin_oracles.py](pin_oracles.py) generates the exact RGB64 fixture and banded
luma ramps used by the Rust tests. Its references are the pinned Google,
Cloudinary, Netflix and IVP-Lab sources from Round 4. Google and Cloudinary CLI
formatting is increased to 17 significant digits; arithmetic is unchanged.
The C++ SSIMULACRA build uses OpenCV 4.10.0. WS-HV uses cv2 5.0 and PyWavelets
1.10.0. No pyiqa source, package or output is used.

| Metric | Oracle score | Absolute tolerance | Reason |
|---|---:|---:|---|
| Butteraugli-original | 2.201789379119873 | 1e-7 | below one f32 ULP at this score |
| SSIMULACRA1 | 0.01399088125957948 | 1e-12 | complete 4×4 blocks and SIMD rows; final f64 rounding only |
| WS-HV | 0.7486863524749465 | 1e-15 | a few f64 Haar/pool rounding ULPs |
| CAMBI, step1 every8 samples | 0.824363696451263 | 1e-12 | nonzero original C score; below f32 reduction drift |
| CAMBI, step4 every16 samples | 15.471107781777791 | 1e-12 | same, with stronger visible banding |

The RGB fixture is `(13*x + 17*y + 71*c + (x*y)%29)%256`; every seventeenth
packed byte in the distortion is XOR63. CAMBI is 256×256, 10-bit limited-range
luma `256 + (x/block)*step`. No image/binary fixture is committed.
[oracle_pins.json](oracle_pins.json) also preserves a 37×41 diagnostic: its
SSIMULACRA1/WS-HV residuals are larger and not fully attributed, so it is not used
as a tight unit pin. [pins_rust.rs](pins_rust.rs) scores both RGB geometries.

## SSIMULACRA1 block-order experiment

Four 4-value row partial sums replace the sequential 16-value reduction for
complete 4×4 blocks. Incomplete blocks keep their original clipped behavior.
Against the original C++ on all61 pairs, native median/max improves from
6.243039321618404e-9 / 1.2795374011980581e-7 to
5.758771265407248e-9 / 1.2784454301965897e-7. Oracle JND median/max improves from
1.411077636959135e-7 / 3.692812030942072e-6 to
1.3870521731185192e-7 / 3.6896605500602586e-6.

The ladder native maximum decreases slightly, but its median and JND maximum
increase slightly. Published AIC maximum also increases slightly. The change is
retained for the smaller original-oracle AIC residual and reference operation
order; no acceptance threshold changes. Every value in both units is preserved
in [ssim_order_summary.json](ssim_order_summary.json).

## Reproduction

Use the disk-backed Round 4 Python environment and oracle sources. Commands run
under the shared heavy lock and `run-heavy`; set
`LOCK_SNAP_DIR=$HOME/tmp/zenmetrics-lock-snap-aicvariants`,
`TMPDIR=$HOME/tmp/aic-variants-scratch`, and `CARGO_INCREMENTAL=0`.

- `just aic-review-pins <python>` generates original-author pins.
- `just aic-review-rust-pins` scores the Rust fixtures.
- `just aic-review-pin-sensitivity` tests intentional scratch-only mutations.
- `just aic-review-libs` tests CLI and vmaf without/with SIMD.
- `just aic-review-bogus` enables sweep and runs all bogus-output regressions.
- Build the release `aic_more_probe` via `just snap-cargo @ build -p zenmetrics-cli --example aic_more_probe --release`.
- `just aic-review-oracles <python>` reruns all four original-author panels.
- `python3 benchmarks/aic_review_2026-10-15/compare_panels.py` checks score bits
  against Round 4 (SSIMULACRA1 against the retained block-order candidate).
- `just aic-review-ci` runs the library/sweep tests and CI's exact all-member
  workspace fmt and workspace clippy command.

Existing results are not overwritten; move an earlier run to a retained directory
before regenerating. Full logs, formatter-only build copies and binaries stay
outside git under `~/tmp/aic-variants-scratch/round5`; compact tables and hashed
receipts are committed here. The disk-full failed attempts are retained and do
not count as gate results. Only inactive Cargo incremental cache was removed.

## Post-incident validation

The coordinator marked 17:10–17:25 UTC as an ENOSPC risk window. Results from that
window are provisional. A clean rebuild after 17:25 passes CLI library tests
(81 pass, two existing timing tests ignored), vmaf without features (17 pass),
vmaf with `simd,avx512` (38 pass, one existing ignored), all11 sweep bogus-output
tests, CI workspace fmt and the exact workspace clippy command. The wrapper
reports rc0,61s, peak-RSS3.19GiB, min-avail20298MiB, peak-load6.74.

Original-author pin sources were rebuilt after the window. The new oracle pin
capture is byte-identical to the committed capture; the 64×64 Rust residuals are
0 / 4.40619762898109e-16 / 6.661338147750939e-16 for Butteraugli / SSIMULACRA1 /
WS-HV. [pin_residuals.json](pin_residuals.json) retains the odd-size diagnostic too.

Scratch-only negative controls prove the pins reject a changed Butteraugli
mask-response constant, the previous sequential SSIMULACRA1 4×4 reduction, and
f32 WS-HV pooling. The mutation checker itself succeeds only when each selected
pin fails. The standalone CAMBI pins are nonzero and run in both feature builds.

The branch remains stacked on the original `aic-more` base. Inventory is still
87 locally and88 after the coordinator's merge with master's `flip-v17`; Round 5
does not alter that expectation or merge master into this lane.

The complete fresh oracle run reports rc0,567s, peak-RSS2.97GiB,
min-avail20362MiB, peak-load8.23. [panel_invariance.json](panel_invariance.json)
verifies exact score bits: Butteraugli and WS-HV retain all82 scores each; CAMBI
retains all61 AIC scores and12 zero ladder scores, with9 matched refusals.
SSIMULACRA1 retains all82 scores of the measured row-partial candidate.
[summary.json](summary.json) and the eight per-metric TSVs report every residual
in native and fitted-JND units. No published-column discrepancy is relabeled
as exact reproduction.

A fresh C CAMBI run also reproduces every stored default/v1/uncapped-v1 value
and geometry refusal across all61+21 cases exactly; see
[cambi_reference_invariance.json](cambi_reference_invariance.json). Its wrapper
reports rc0,11s, peak-RSS0.67GiB, min-avail22452MiB, peak-load12.23. Full successful,
failed and interrupted captures are retained under
`~/tmp/aic-variants-scratch/retained-round5` (i265), with file hashes and production
source hashes in [validation.meta](validation.meta).
