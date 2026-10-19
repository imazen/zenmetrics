# Local canonical rechecks

Missing: the full per-name grid, remaining classical and learned references,
map comparisons and qualified regression intervals. The first six-name panel
is a scalar check on one 512×512 AIC photo pair, at one worker, on Ryzen 9
7950X / WSL2. It does not replace multi-size or content validation.

The frozen Rust build is upstream master `efae6b8fed9e57346157d92dffc3cdb57c44560e`
(clean exported tree `f82b3614b385`). Binary SHA256:
`7859f91207fceeb45243dff3955ef4b276a885e35d5e1ca93ca49bd983a02cc0`.
It enables `compete,metrics-parallel,zenmetrics-cli/nr`, runtime SIMD dispatch,
and the existing immutable-model preload/map benchmark adapters. No native-CPU
compiler flag was used. The build receipt is rc=0, 126s, peak-RSS 0.87GiB,
minimum available 10651MiB, peak load 3.42.

[results.json](results.json) supplies the README's exact-name observations.
`pass` uses the existing NR JND/1000 bound. A displayed delta without `pass`
means no external acceptance bound has been assigned. No golden or threshold
was changed. The first attempt retained only canonical observations because a
verified orphan zenbench reservation blocked the Rust arms. Its process and
former process groups had exited, its guard was exclusively acquired, and the
reservation was preserved under a `.bak` name before the successful new attempt.

Speed ratios are Rust/canonical (lower is faster), from the median of three
alternating acquisition-block ratios, with five warm observations per arm.
Both sides begin with resident RGB8 and include their callable conversions;
immutable model loading is excluded, and owned parameter clones remain timed.
The NR references evaluate both images. Ratios are descriptive approximations:
there is no confidence interval or 1.08 regression acceptance claim. The
SSIMULACRA1 adapter preconverts RGB to BGR, so its ratio is withheld.
[Resource receipts](resource-receipts.txt) retain every acquisition block's
wrapper peak-RSS/minimum-available/load line. Process `/usr/bin/time -v`
receipts and all samples remain in the raw evidence.

Raw evidence on the local WSL2 machine:
`~/tmp/local-canonical-20261019T2330/retry-run/`.
The adjacent `retry-run-SHA256SUMS.json` binds all files; SHA256
`949f92cd45e2c421cad703a01a06b91b3d3fbb80c553227c345d0e06c49c70f5`.
The plan binds inputs, binaries, explicit helper sources and reference parameters.
This is exploratory executable evidence, not a complete runtime-closure audit.

## Reproduce

Use Python 3.12 for the installed scientific references. First build a current
frozen Rust worker with the existing snapshot build recipe. The ordinary
`just bench-iqa` follows the shared cold/warm CPU protocol; the command below
adds explicit original-implementation comparisons.
Prepare a plan with explicit installed reference paths:

```sh
just bench-iqa-canonical-plan \
  --binary /path/to/iqa-compete --pairs /path/to/pairs.json \
  --output /path/to/new-plan.json --source COMMIT \
  --python /path/to/reference-venv/bin/python \
  --extras /path/to/basicsr-and-pypiqe --brisque-weights /path/to/weights.pt \
  --ssim1 /path/to/ssimulacra-canonical --ssim2 /path/to/iqa-ssim2-primary \
  --sizes cid22 --threads 1
just bench-iqa-canonical /path/to/new-plan.json /path/to/new-results
```

The initial registry includes BRISQUE/PIQ 0.8.0, NIQE/BasicSR 1.4.2,
PIQE/pypiqe 1.2, original SSIMULACRA1, primary SSIMULACRA2 v2.1 and NVIDIA
FLIP 1.7. Missing adapters remain explicit in the plan. `--metrics` on either
command restricts the named cohort. The comparison runner serializes heavy
work under both existing local admission locks and the 12GiB resource wrapper.
Existing outputs are refused rather than overwritten. Run
`just bench-iqa-canonical-check` for parser, failure and scope controls.
