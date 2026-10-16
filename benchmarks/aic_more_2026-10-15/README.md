# Four AIC-4 metric columns

All four CLI implementations are present. Exact reproduction of the published
CAMBI, SSIMULACRA1 and WS-HV columns remains unconfirmed; the tables retain the
measured residuals. Original Butteraugli agrees to the published rounding.

| CLI | Column | Convention |
|---|---|---|
| `cambi` | `cambi` | libvmaf standalone defaults; distorted studio-swing BT.601 luma, crop odd dimensions to even |
| `butteraugli-original` | `butteraugli_original` | Google pre-libjxl Butteraugli, default asymmetry, maximum error map value |
| `ssimulacra1` | `ssimulacra1` | Original Cloudinary RGB→linear→Lab, six scales, asymmetric edge and worst-grid/block pooling |
| `ws-hv` | `ws_hv` | Rounded RGB BT.601 gray, three symmetric Haar levels, H/V similarity product, diagonal saliency |

The three private RGB implementations support independent row strides. The new
`vmaf::cambi_from_luma` entry point supports strided luma and keeps a packing-free
tight path. Existing `cambi_v1_from_luma` and VMAF-v1 maps retain their preset.

## Sources and identification

- CAMBI: [Netflix libvmaf v3.2.1](https://github.com/Netflix/vmaf/tree/f85a853692a8c730d0270cd733c8bb30b5b93b7c),
  commit `f85a853692a8c730d0270cd733c8bb30b5b93b7c`.
- Original Butteraugli: [Google reference](https://github.com/google/butteraugli/tree/71b18b636b9c7d1ae0c1d3730b85b3c127eb4511),
  commit `71b18b636b9c7d1ae0c1d3730b85b3c127eb4511` (2019-03-19), Apache-2.0.
  The existing sibling crate is the libjxl algorithm; its available parameters do
  not select this earlier opsin response, masking or Malta weights. Its measured
  61-pair published median/max errors are 1.0871057609558106/11.723514236450196,
  or 0.23395578783422333/2.032673033167208 JND. This justified a separate port.
- SSIMULACRA1: [Cloudinary reference](https://github.com/cloudinary/ssimulacra/tree/1fa99a464b571b31311938c7fb6a773407454312),
  commit `1fa99a464b571b31311938c7fb6a773407454312`, Apache-2.0.
  Copyright notices are in `crates/zenmetrics-cli/NOTICE-ssimulacra1`.
  The C++ build substitutes three minimal OpenCV includes for the umbrella header;
  algorithm source is unchanged. Rust follows the generic eleven-tap row filter
  and symmetric column filter with fused multiply-add, matching OpenCV's arithmetic
  much more closely than a symmetric horizontal reduction. The remaining low-bit
  residual has not been fully attributed.
- WS-HV: Cheraaqee, Maviz, Mansouri and Mahmoudi-Aznaveh,
  [Quality Assessment of Screen Content Images in Wavelet Domain](https://doi.org/10.1109/TCSVT.2021.3067627),
  IEEE TCSVT 32(2), 566–578 (2022).
  Identified through [bibliography entry 13](https://www.aimspress.com/aimspress-data/mbe/2023/5/PDF/mbe-20-05-412.pdf),
  the publisher record and the [author repository](https://github.com/IVP-Lab/WS_HV/tree/e31cb50fb8c0e12f7dc3eb185092b23e60ab1bc6).
  Commit `e31cb50fb8c0e12f7dc3eb185092b23e60ab1bc6` has no declared repository license;
  its notebook is only a local oracle. No author code is vendored. Rust independently
  implements the mathematical Haar, similarity, interpolation and pooling operations.
  The notebook's BGR-loading/RGB-conversion combination was tested explicitly;
  proper RGB gives smaller published errors. Flat zero-saliency pairs return zero,
  as the author oracle does. JPEG's CTC and CFP links were resolved but PDF retrieval
  failed; identification does not rely on reading those PDFs.

No pyiqa source, installation, execution or model output was used.

## CAMBI settings and geometry

Compared standalone defaults (VLT=0, no high-resolution speedup, cap=1000), the
VMAF-v1 preset (VLT=0.06, speedup from 1080p, cap=17), and uncapped v1. All use
window=65, topk=0.6, TVI=0.019, maximum log contrast=2, BT.1886.
Standalone defaults give the smaller published median/JND errors and are selected.
The published maximum 17.100557 also rules out an exact cap-17 match.

The native reciprocal-table restriction is preserved: both dimensions below 216
refuse, and the resolution-adjusted histogram window must fit 4226 entries.
Nine of 21 ladder pairs refuse in both Rust and C (64², 128², 4096² for each of
three images); the other twelve produce scores. No refused case is called a pass
or silently skipped. The v1 speedup supports the three 4096² cases but changes the
selected standalone convention.

## Reproduction and evidence

Run the recipes under the shared heavy-job lock and `run-heavy`, with
`LOCK_SNAP_DIR=$HOME/tmp/zenmetrics-lock-snap-aicvariants` and disk-backed TMPDIR.
Use `just snap-cargo @ build -p zenmetrics-cli --example aic_more_probe --release`,
then copy the example to the oracle scratch directory as `aic-more-final-probe`.
`just aic-more-verify <python>` runs the four oracle panels; `just aic-more-ci` runs
CLI/vmaf tests, all-member fmt, CI's exact workspace clippy flags and inventory.
The Python environment needs NumPy, Pillow, OpenCV and PyWavelets; oracle versions,
build commands and hashes are retained in `validation.meta`.

Input identities are shared with [Round 3 input hashes](../aic_flip_2026-10-15/input_hashes.tsv).
All 61 published rows and all 21 ladder cases are retained in the per-metric TSVs.
Absolute errors are reported in native published units and fitted JND units using
`a * max(0, sign * (x-b)) ** c` from the supplied site config. These are measured
residuals, not newly invented acceptance thresholds. C++ Butteraugli prints six
decimals and SSIMULACRA1 prints eight decimals, limiting scalar oracle precision.

Full logs, old arithmetic experiments, oracle sources/binaries and JSON captures
remain in the external scratch directory recorded in `validation.meta`; generated
files over 30 KB are not committed. Per-metric compact TSVs preserve the results.

## Measured errors

All tables show median / maximum absolute error.

| Metric | 61-pair published units | 61-pair JND units |
|---|---:|---:|
| cambi | 0.000131816381678 / 0.17806128161 | 0.000157035404803 / 0.0648476588305 |
| butter | 2.69897461003e-06 / 4.9473571778e-05 | 5.93321182407e-07 / 8.94251708994e-06 |
| ssim1 | 2.284971353e-07 / 1.86120462599e-05 | 5.58530569972e-06 / 0.000537036728972 |
| ws | 0.000894208359779 / 0.0207583157283 | 0.00492011604994 / 0.139318362305 |

| Metric/panel | Oracle units | Oracle JND units |
|---|---:|---:|
| cambi/ladder | 0 / 0 | 0 / 0 |
| cambi/aic | 0 / 6.93889390391e-18 | 0 / 6.43929354283e-15 |
| butter/ladder | 2.41424560343e-07 / 4.79095459305e-07 | 4.80664223979e-08 / 9.64021704686e-08 |
| butter/aic | 2.74688720658e-07 / 4.73571777349e-07 | 5.83466626125e-08 / 1.26883100249e-07 |
| ssim1/ladder | 2.14297105061e-09 / 4.60295007665e-09 | 4.28218243087e-08 / 8.97034562186e-08 |
| ssim1/aic | 6.24303932162e-09 / 1.2795374012e-07 | 1.41107763696e-07 / 3.69281203094e-06 |
| ws/ladder | 1.11022302463e-16 / 2.77555756156e-16 | 8.881784197e-16 / 2.6645352591e-15 |
| ws/aic | 1.11022302463e-16 / 2.32782102483e-08 | 8.881784197e-16 / 1.59105298714e-07 |

CAMBI ladder statistics cover 12 numeric pairs, all exactly 0.0 in both
implementations, and 9 matched refusals. Those zero scores do not establish
banding sensitivity: the 61-pair panel is the substantive parity evidence, with
58 non-zero scores and a maximum oracle error of 6.94e-18.
CAMBI C JSON formatting was changed from six decimals to 17 significant digits
without changing metric arithmetic; the earlier rounded-output captures are retained.

Final validation: CLI library 78 pass, two pre-existing ignored; VMAF library
37 pass. CI all-member fmt and exact workspace clippy pass. Inventory has 9/10
tests passing and the expected 87 != 88 failure on this stacked base; no assertion
was relaxed. `inventory_union.json` verifies master has 84, only `flip-v17` is
missing here, and the union with these four names is exactly 88. The coordinator
merges this branch and retains master’s reviewed FLIP/CW-SSIM changes.

Pair tables: [CAMBI](cambi_aic.tsv), [Butteraugli-original](butter_aic.tsv),
[SSIMULACRA1](ssim1_aic.tsv), [WS-HV](ws_aic.tsv). Full numeric summaries are in
[summary.json](summary.json); source/log/binary hashes are in
[validation.meta](validation.meta).

## Alpha boundary

The CLI decodes to RGB8 and drops alpha. These ports therefore score those RGB
channels. Upstream SSIMULACRA1 instead blends transparent inputs onto gray 128
and includes an alpha term. Upstream original Butteraugli evaluates black and
white backgrounds and takes the worse result. The RGB oracle panels here do not
qualify equivalence for transparent inputs.
