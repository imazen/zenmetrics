# Fresh AIC61 fixed-factor verification

Fresh numerical execution on 2026-10-20, Ryzen 9 7950X / Ubuntu WSL2, on all 61 staged RGB8 840×944 AIC2026 crop pairs. This is a separate cohort from the [retained-evidence aggregation](../README.md). Source is current `origin/master` at build start: `287921882c7d47c8ffcaa13226e20794be2408b0`; the build ran in a dedicated jj workspace and CI-pinned sibling snapshot. No metric arithmetic or numerical bound changed.

1952 finite scalar comparisons, 122 matching size refusals, 0 above-bound or unmatched outcomes across 34 scalar variants (29 CLI names). FSIMc is included because the author function returns it alongside FSIM. Each scored variant has 61 comparisons. MS-SSIM k5/k6 each refuse all 61 pairs: their final pooled views cannot accommodate the unchanged 11×11 scoring window. Refusals do not contribute to medians or correlations. Spearman compares Rust and oracle scores across these 61 pairs, using average ranks for ties; it is not correlation with subjective ratings.

| Variant | Scored | Matching refusals | Median abs Δ | Max abs Δ | Spearman | Above bound |
|---|---:|---:|---:|---:|---:|---:|
| `fsim` | 61 | 0 | 1.945281491e-09 | 4.532695186e-08 | 1 | 0 |
| `fsim-k2` | 61 | 0 | 1.834910446e-09 | 9.921470463e-07 | 1 | 0 |
| `fsim-k3` | 61 | 0 | 1.945281491e-09 | 4.532695186e-08 | 1 | 0 |
| `fsim-k4` | 61 | 0 | 1.475267131e-09 | 1.37987739e-08 | 1 | 0 |
| `fsim-nodownscale` | 61 | 0 | 3.739201171e-09 | 1.079870822e-06 | 1 | 0 |
| `fsimc` | 61 | 0 | 1.98164174e-09 | 4.501693585e-08 | 1 | 0 |
| `fsimc-k2` | 61 | 0 | 1.860761212e-09 | 9.920724756e-07 | 1 | 0 |
| `fsimc-k3` | 61 | 0 | 1.98164174e-09 | 4.501693585e-08 | 1 | 0 |
| `fsimc-k4` | 61 | 0 | 1.505025549e-09 | 1.231888147e-08 | 1 | 0 |
| `fsimc-nodownscale` | 61 | 0 | 3.989125363e-09 | 1.07937192e-06 | 1 | 0 |
| `ms-ssim-pyiqa-f32` | 61 | 0 | 1.537675842e-08 | 1.006509572e-07 | 1 | 0 |
| `ms-ssim-pyiqa-f32-k2` | 61 | 0 | 4.089996231e-09 | 5.701840278e-08 | 1 | 0 |
| `ms-ssim-pyiqa-f32-k3` | 61 | 0 | 1.601388355e-09 | 3.470096754e-08 | 1 | 0 |
| `ms-ssim-pyiqa-f32-k4` | 61 | 0 | 7.856432171e-10 | 3.554067529e-08 | 1 | 0 |
| `ms-ssim-pyiqa-f32-k5` | 0 | 61 | — | — | — | 0 |
| `ms-ssim-pyiqa-f32-k6` | 0 | 61 | — | — | — | 0 |
| `ssim-pyiqa` | 61 | 0 | 6.553225523e-07 | 1.294691057e-05 | 1 | 0 |
| `ssim-pyiqa-f32` | 61 | 0 | 8.311299937e-08 | 3.417382702e-07 | 1 | 0 |
| `ssim-pyiqa-f32-k2` | 61 | 0 | 2.935776e-08 | 1.815311341e-07 | 1 | 0 |
| `ssim-pyiqa-f32-k3` | 61 | 0 | 1.398658345e-08 | 1.044854222e-07 | 1 | 0 |
| `ssim-pyiqa-f32-k4` | 61 | 0 | 7.82340237e-09 | 9.453135574e-08 | 1 | 0 |
| `ssim-pyiqa-f32-k5` | 61 | 0 | 4.59980054e-09 | 6.376740525e-08 | 1 | 0 |
| `ssim-pyiqa-f32-k6` | 61 | 0 | 2.95592173e-09 | 5.50087671e-08 | 1 | 0 |
| `ssimc-pyiqa-f32` | 61 | 0 | 1.090858281e-07 | 3.602111662e-07 | 1 | 0 |
| `ssimc-pyiqa-f32-k2` | 61 | 0 | 4.275112453e-08 | 1.854551329e-07 | 1 | 0 |
| `ssimc-pyiqa-f32-k3` | 61 | 0 | 1.976597674e-08 | 1.341168125e-07 | 1 | 0 |
| `ssimc-pyiqa-f32-k4` | 61 | 0 | 1.307356856e-08 | 1.295400835e-07 | 1 | 0 |
| `ssimc-pyiqa-f32-k5` | 61 | 0 | 8.913112248e-09 | 9.471959339e-08 | 1 | 0 |
| `ssimc-pyiqa-f32-k6` | 61 | 0 | 6.402382624e-09 | 8.772012761e-08 | 1 | 0 |
| `vsi` | 61 | 0 | 3.859942921e-09 | 1.083987233e-08 | 1 | 0 |
| `vsi-k2` | 61 | 0 | 3.666916659e-09 | 1.141070183e-08 | 1 | 0 |
| `vsi-k3` | 61 | 0 | 3.859942921e-09 | 1.083987233e-08 | 1 | 0 |
| `vsi-k4` | 61 | 0 | 3.78539311e-09 | 9.993059269e-09 | 1 | 0 |
| `vsi-nodownscale` | 61 | 0 | 3.166062634e-09 | 1.323239984e-08 | 1 | 0 |

## Oracle contracts and bounds

PyIQA 0.1.16 / Torch 2.14.1 / NumPy 2.5.4 / Pillow 13.0.0, CPU float64 input. `ssim-pyiqa` calls the original default rounded-Y class with no downsampling. Unrounded Y uses f64 0.299/0.587/0.114 coefficients; the color variant uses RGB. Fixed-factor direct calls use `downsample=True` and only replace the shared `ssim()` factor expression with k. MS-SSIM therefore pools each scoring view independently while advancing the unpooled twofold pyramid. k1 is the unsuffixed unrounded route. PyIQA retains its original float32-generated Gaussian and MS weights promoted to float64.

The supplied scratch oracle was copied, with package context supplied for its relative import, explicit 16/1 intra/inter-op thread caps, and full exception messages retained. Arithmetic and factor semantics were unchanged. The existing verified 0.1.16 environment was reused instead of reinstalling dependencies. Original and executed script hashes are recorded below.

FSIM/FSIMc and VSI use the authors' [FeatureSIM.m](https://cslinzhang.github.io/home/FSIM/Files/FeatureSIM.m) and [VSI.m](https://cslinzhang.github.io/home/VSI/Files/VSI.m) under Octave 6.4.0 / system image 2.12.0. Automatic controls execute the original sources. Fixed k1–k4 copies pass the repository patcher and independent checker before and after execution; only automatic F becomes k. VSI selects the [unchanged upstream image-2.20.1 resize overlay](../../iqa_compete_2026-10-16/image_resize_source.py), containing imresize.m and private/interp_method.m. This is a two-file overlay, not installation of the newer package. FFTW and BLAS/OpenMP use one thread. Rust uses one Rayon thread. No PyIQA FSIM/VSI substitute is used.

Existing inclusive absolute bounds: rounded SSIM 5e-5; unrounded SSIM family 1e-6; FSIM/FSIMc 5e-5; VSI 1e-4. See [prior contract](../README.md#bounds-and-limitations). These measured results do not establish 1e-9 relative agreement. This panel qualifies only these inputs and scalar routes; maps, other sizes, timings, GPU paths and all MATLAB releases are not verified.

Independent SciPy Spearman recomputation agrees for all 32 scored variants. Automatic-factor Rust scores equal k3 bit-for-bit for every pair in FSIM, FSIMc and VSI (183 controls); the geometric automatic factor is 3. All source, binary and decoded input bindings remain unchanged. The first Rust harness attempt assumed a single FSIM output; its records are preserved separately, and the corrected full run extracts both columns.

## Commands and identities

Scratch root: `$HOME/tmp/ffverify`; full per-pair outputs, patched sources, input bindings, worst-case paired values and logs are retained there. Only this aggregate and summary are committed.

```sh
# Dedicated jj workspace; origin/master fetched before the build.
TMPDIR=$HOME/tmp/ffverify/scratch LOCK_SNAP_DIR=$HOME/tmp/ffverify/build \
  ~/work/claudehints/scripts/run-heavy --mem 12G --jobs 16 -- \
  just snap-cargo @ build --release -p zenmetrics-cli
# Each of these ran serially under the same TMPDIR and run-heavy cap:
<verified-pyiqa-env>/bin/python ~/tmp/ffverify/oracle-aic61.py \
  ~/tmp/ffverify/pairs.tsv ~/tmp/ffverify/aic61/pyiqa.tsv
python3 ~/tmp/ffverify/octave_aic61.py
python3 ~/tmp/ffverify/rust_aic61.py
python3 ~/tmp/ffverify/aggregate_aic61.py
# Rust driver: one JSON score command per pair and CLI name; FSIM emits two columns.
~/tmp/ffverify/build/target/release/zenmetrics score --metric <variant> \
  --reference <ref.png> --distorted <dist.png> --output json
```

Whole-step resource receipts (not per-metric memory or speed measurements):

`run-heavy: done rc=0 110s | peak-RSS 1.45GiB | min-avail 7793MiB | peak-load 9.73`  
`run-heavy: done rc=0 356s | peak-RSS 1.67GiB | min-avail 9379MiB | peak-load 11.10`  
`run-heavy: done rc=0 369s | peak-RSS 0.60GiB | min-avail 10072MiB | peak-load 7.28`  
`run-heavy: done rc=0 167s | peak-RSS 0.21GiB | min-avail 10585MiB | peak-load 4.30`  

SHA256 bindings:

| Artifact | SHA256 |
|---|---|
| Rust CLI | `41478902ef31ae910f05942009e4c0adfed7b30c0dde89a31320eb55fb38b084` |
| Pairs TSV | `7a65924bf71cb358185da915f2f0820b948be6e96deee7c7ce6e58ca5e4bff08` |
| RGB8 input bindings | `fcd57b11879aac594d2b5891a3781d07e40ad70a35627cc131df48a82febcf4e` |
| Original supplied oracle | `19ae1683999b5d77c4e3f7f3b51fb5a2136383188b9689ac9fb80b27891c9ed3` |
| Executed PyIQA oracle | `4f3226c9ab0fbf50eaa701b10e69469a38bd9c1c9a0469062d3916ea73b5c5c9` |
| PyIQA source | `a0eb8e0bedfae99d9d014371a77b7d9afb110c71a22b6cda06a4992da0d951e8` |
| Author FeatureSIM.m | `4b23dc2881c71dfa996e40fd84b9b946f3934b36ecfb3ee4dfbafa5505bcbba9` |
| Author VSI.m | `5939b9053966f58377e08052b5e2d0e0e3f954040fe732625591363285433954` |
| Resize archive | `b4b6865e7f65ee5b9c254a69128bf10e93b38bc27f5bc5b973fef300217fc725` |
| Octave driver | `5d711a08b66b8d070ccac8987f2ffb494356723928542e333155755b86648375` |
| Rust driver | `deb5d21622bd3a3b3ffb827651716100ed617ab5be1b0d2a72c3d9dc7f75720a` |
| Aggregator | `de5ce5c605cd5b0328bd17e373e067f66d9e77c4aedd409eb700301fe8ac0c01` |
| PyIQA results | `7a2b2fa89be308195be62a9b64bd16ca6923b316869e7e647f6e8bf7283840ee` |
| Octave results | `0de345624ee505b0d43c8ae682f91d67e6dfe933f94bcc8812534db178d93733` |
| Rust results | `c88572de98c430030a3135e6379a3e9f660fba41299b77b51ce035a2c0baa4ae` |
| Aggregate TSV | `95fc08756b5cd79227157c24a6fbede44c4f9e9c3c2c36f3e3b6c27395b274cf` |
