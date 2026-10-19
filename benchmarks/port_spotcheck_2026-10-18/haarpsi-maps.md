# HaarPSI native map observations, 2026-10-19

Missing: an agreed external map bound and an explanation of the nonfinite
differences. These observations do not qualify map parity or timings.

The [24-case map grid](haarpsi-maps.json) covers the retained three RGB pairs,
four sizes and 1/28 threads on the Ryzen 9 7950X. Every returned scalar exactly
matches the earlier scalar study on its own implementation. The original
[HaarPSI.m](https://github.com/rgcda/haarpsi/blob/3b9599b0c1911ee08f1bb991bb19e19d86727819/HaarPSI.m)
returns separate orientation/chroma similarity and weight arrays. We retain
both arrays and compare their weighted mean with the Rust native map; no
logistic transform is applied to either map.

The largest finite pixel difference is `1.1233741833871491e-5`.
The illustration pair at 3840×2160 has 41 positions at each thread count
where Rust returns NaN and the weighted primary map is finite. The
[82 observations](haarpsi-map-nonfinite.tsv) retain the coordinates, values
and each primary weight. Primary weight sums there range from
`2.1316282072803006e-14` to `3.410605131648481e-13`; primary similarities
range from `0.9997549215060189` to `1.0`. Tiny nonzero primary weights are
an observation, not proof of the cause of Rust's NaNs. Other positions with
matching NaNs are counted separately in the per-pair tables.

The existing `5e-5` scalar bound is unchanged and is not assigned to maps.
The map grid uses the frozen `e2d414c33a6d3c7da50a818fa36e56654d3ebeef`
metric rlibs; source bytes are verified against that commit before compiling
the standalone probe. This is a separate cohort from newer master binaries.

Raw arrays and logs remain at
`/home/lilith/tmp/portcheck-haarpsi-map-panel-20261019T1443/`, with per-file
hashes in each raw result and the completion hash in the committed index.
The map run receipt was `rc=0 38s | peak-RSS 1.12GiB | min-avail 12414MiB |
peak-load 4.09`. Publication verified every recorded file hash and recomputed
the mismatched positions: `rc=0 1s | peak-RSS 0.20GiB | min-avail 13471MiB |
peak-load 4.07`. These are process resource observations, not per-metric
memory or latency claims.

Commands are the existing `haarpsi-primary-map-check` recipe and
`haarpsi-primary-maps-publish` in
[`oracle.just`](../iqa_compete_2026-10-16/oracle.just). Both run under the two
shared-host locks and `run-heavy --mem 12G`; inputs are the retained
`portcheck-primary-grid-20261019T1020/inputs/pairs.json` manifest.
