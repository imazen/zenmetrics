# AOM sweep color-signaling regression

Tracking: [zenmetrics #112](https://github.com/imazen/zenmetrics/issues/112).

The SDR `aom-rs` sweep arm converts RGB to **full-range BT.601 YCbCr**, but
its oracle-bootstrap sequence OBU declares limited range and omits the color
description. The AVIF container is separately tagged BT.709/sRGB/BT.601/full.
The input conversion, bootstrap copy, and container-only declarations remain
in `crates/zenmetrics-cli/src/sweep/encode.rs` at `2fb351fe`.
This is an encode-wrapper finding; it does not establish a general defect in
the standalone zenav1-aom encoder.

The retained 256×256 speed-8 q100 AOM witness was reproduced on 2026-11-07
with FFmpeg/ffprobe 8.0.1. It declares `color_range=tv`, with no reported
primaries, transfer or matrix. Normal RGB decode and explicitly full-range
BT.601 decode differ in **115265 of 196608 RGB components**, maximum delta
**21**. The same-source SVT speed-8 q100 control reports
`pc/smpte170m/iec61966-2-1/bt709` and has **0** differing components.
These are comparisons between two interpretations of the same encoded file,
not lossless comparisons against the original image.

## Reproduce the two failing assertions

Requires Python 3, `just`, and FFmpeg/ffprobe with an AV1 decoder. No Python
packages or Rust build are needed. Files remain in the existing object store;
no binary fixtures are committed. Supply the configured endpoint and
credentials for the store holding this run, then fetch the immutable witnesses:

```sh
mkdir -p "$HOME/tmp/avif-color-regression"
aws --endpoint-url "$ZEN_S3_ENDPOINT" s3 cp \
  s3://zentrain/jobs/smallav1-enc-20261028/blobs/301f0ff8e62bb367e49fcf8f84617f480ac9ec6458b8d4eb550679d647b90b9e \
  "$HOME/tmp/avif-color-regression/aom-q100.avif"
aws --endpoint-url "$ZEN_S3_ENDPOINT" s3 cp \
  s3://zentrain/jobs/smallav1-enc-20261028/blobs/018c8b771bad4185a56a1f54ca969190d539684d9fb083b898741fe3fcc69515 \
  "$HOME/tmp/avif-color-regression/svt-q100.avif"

# Passing control: both tests pass, exit 0.
just avif-color-regression "$HOME/tmp/avif-color-regression/svt-q100.avif" \
  --sha256 018c8b771bad4185a56a1f54ca969190d539684d9fb083b898741fe3fcc69515

# Historical defective output: both tests fail, exit 1.
just avif-color-regression "$HOME/tmp/avif-color-regression/aom-q100.avif" \
  --sha256 301f0ff8e62bb367e49fcf8f84617f480ac9ec6458b8d4eb550679d647b90b9e
```

`test_av1_declares_input_color` checks all four stream color fields against
the encoder's actual input conversion and container contract.
`test_normal_decode_matches_explicit_input_color` requires exact RGB equality
between normal decode and a decode explicitly using full-range BT.601. It
uses the same decoder and output format in both paths, avoiding a lossy-codec
quality threshold. FFmpeg is an independent test oracle only.

The caller explicitly selects this artifact regression via the just recipe;
it is not automatically run against an absent private fixture in CI. Missing
inputs, tools, bad checksums, invalid streams and decode failures are errors,
never successful skips. The test runner accepts sub-1MP SDR AV1 images.

## Acceptance for a repair

Run the **same tests**, without the historical checksum, on newly generated
AOM sweep output (including the witnessed settings):

```sh
just avif-color-regression /path/to/fresh-aom-q100.avif
```

Both tests must pass. Preserve the historical blob and its failing result.
Add the gate to the fresh-encode validation path when repairing the wrapper;
then retain correctly signaled encodes and recompute quality/feature sidecars
under their new content hashes. Do not relabel historical quality scores as
repaired, and do not remove the existing payload-parity checks.

This change supplies reproduction/acceptance tests, **not a production fix**.
The separate s8 oracle-payload divergence is not explained by this finding.
