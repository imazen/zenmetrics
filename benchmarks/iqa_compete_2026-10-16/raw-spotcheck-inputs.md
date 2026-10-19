# Raw input grid for primary port checks

[prepare_spotcheck_raw.py](prepare_spotcheck_raw.py) center-crops retained
RGB8 derivatives. It performs no decoding, resampling or color arithmetic.
Sources may have strided rows; outputs contain only active crop pixels.
Every source SHA256 is checked before output creation and again after copying.
New preparations require a fresh directory. Source and output identities,
geometry, crop coordinates and generator hash are retained in `pairs.json`.

The grid is 64×64, 255×257, 1024×1024 and 3840×2160 for each of three pairs:

- S67 reference/CCM15: landscape photograph, the committed AIC input pair.
- S68 reference/JXL15: stained-glass artwork with strong outlines.
- S58 reference/JXL15: digital illustration.

These use existing port-check raw 4K derivatives. The S67 bytes match the
committed [input metadata](inputs.meta.json). Historical B/C preparation
commands survive, but their original executable/source binding has not been
independently re-established. The new crops bind the actual consumed raw
bytes; this is not a new claim about historical decoding equivalence.
Smaller cells are center crops, not resized versions of the whole picture.

```sh
just --justfile benchmarks/iqa_compete_2026-10-16/oracle.just \
  spotcheck-raw-prepare <source-plan.json> <fresh-output-directory>
```

The October 19 preparation creates 12 pairs with manifest SHA256
`83d61ef3ce1877a20e07a2c8f8ac0df0a761a0a77903a5784b011a74b17f0712`.
It lives on the host's ext4 filesystem under
`~/tmp/portcheck-primary-grid-20261019T1020/inputs/`.
The plan, original files and generated pixels are preserved outside git.
Full command/output: `~/tmp/portcheck-primary-grid-20261019T1020-prepare.log`.
Both heavy-job locks and `run-heavy --mem 12G --jobs 1` report
`done rc=0 1s | peak-RSS 0.07GiB | min-avail 12858MiB | peak-load 3.03`.

`oracle.just spotcheck-raw-check` exercises coordinate-coded packed/strided
pixels, untouched padding and sources, the packed full-row path, short inputs
and invalid geometry. Both tests pass. This checks preparation behavior only;
no scoring or timing comparison is performed.
