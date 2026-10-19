# VSI real-photo crop: reference resize mismatch

Reference-runtime finding, 2026-10-19, host lilith-wsl. A separately bound
resize control explains the focused photo mismatch without changing metric
code or the numerical bound. The initial observations below remain retained.

The 64×64 center crop of the retained S67 reference/CCM15 4K pair fails the
documented VSI absolute bound of **1e-4** against the image-2.12 runner:

| Route | Score |
|---|---|
| Primary original `VSI.m` through Octave 6.4.0/image 2.12.0 | 0.94679144074028154 |
| Primary factor-1 scratch copy | 0.94679144074028154 |
| Frozen Rust `vsi` | 0.9567964257437082 |
| Frozen Rust `vsi-nodownscale` | 0.9567964257437082 |
| Same original author source, unchanged image-2.20.1 resize M-file overlay | 0.95679610895892431 |

Maximum absolute difference: **0.010004985003426614**. Both canonical
first/warm calls repeat, and the existing Rust probe checks repeated score
bits. The unsuffixed Rust score bits are `3fee9e1389aff725`.
The new reference's absolute Rust difference is **3.1678478384922215e-7**,
within the unchanged bound. The selected upstream resize implements the
antialiased convention claimed by the port; system image 2.12 uses different
interpolation. This is a two-M-file overlay on Octave 6.4, not a full newer
package installation. [Source bindings, focused controls, and limits](vsi-resize-runtime.md).

An independent invocation selects the original, unchanged author release
`VSI.m`, SHA256 `5939b9053966f58377e08052b5e2d0e0e3f954040fe732625591363285433954`.
It asserts that raw RGB reshape and independent `imread` of PPM files produce
exactly equal arrays for both images, then obtains the same scalar through
both ingress routes. This reproduces the failure without the factor patch
and checks pixel layout independently. The primary release's
[origin and terms](../iqa_compete_2026-10-16/fixed-factor-oracle.md) are preserved.

Inputs (packed RGB8, sRGB, 64×64):

- Reference SHA256: `f682e292a598fa00b40ac14dcfb7aced34923a2db7f11c81480f72246ac579aa`.
- Distorted SHA256: `2b5079a0f6573a58d9072c73626f511023347ac07e95544c09f2e49c12de8837`.

Rust source is unchanged revision `e2d414c33a6d3c7da50a818fa36e56654d3ebeef`,
CI-pinned siblings, `compete,metrics-parallel`, thin LTO, native flag variables
unset. Executable SHA256:
`b9279b4af7032a35a11b7bd6da2ddb9d258ec044475ea97eb7d99e69aaf8ea6d`.
This is a different input from the synthetic pair whose eight fixed-factor
cases passed; those partial results do not establish general VSI parity.

`/usr/bin/time -v` records canonical process peak RSS **69508 kbytes** and
Rust probe process peak RSS **8956 kbytes**. These are complete process peaks,
including loaded inputs and first/warm calls, not allocation estimates.
Both host locks and the 12 GiB wrapper cover the failed comparison:
`done rc=1 1s | peak-RSS 0.07GiB | min-avail 12782MiB | peak-load 1.71`.
The separate ingress/original-source reproduction succeeds as a diagnostic
control: `done rc=0 1s | peak-RSS 0.07GiB | min-avail 13070MiB | peak-load 1.36`.
No speed ratio is acquired or qualified by these controls.

Complete argv, records and memory receipts:
`~/tmp/portcheck-primary-memory-control-20261019T1025/` and its sibling `.log`.
Independent source, ingress and base-route reproduction:
`~/tmp/portcheck-vsi-photo-independent-20261019T1027/` and its sibling `.log`.
[review.meta](review.meta) binds the reviewed external files by SHA256.
The raw grid and generation recipe are documented
[here](../iqa_compete_2026-10-16/raw-spotcheck-inputs.md).
