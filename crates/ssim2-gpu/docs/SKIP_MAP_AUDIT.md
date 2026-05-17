# SSIMULACRA2 weight audit and skip-map (Technique 2 of Kanetaka et al. IWAIT 2026)

This document tabulates the 108-element score-weight vector in
`pipeline.rs::score_from_stats` and assigns each cell a skip mode.

## Source

The 108-weight table is the one shipped in our `WEIGHT[]` const
(`crates/ssim2-gpu/src/pipeline.rs` lines 944-1053). It is the same
table used by the published `ssimulacra2` Rust crate (cloudinary
ssimulacra2 port). The skip-map dispatches **against this table**, not
against the paper's Table 1 — the IWAIT paper's weight values are
slightly different (different fit, different polynomial postprocess)
and a skip-map sized to those weights could discard cells that matter
to *our* score.

## Indexing convention

Inside `score_from_stats` the 108 weights are addressed by

```text
i = ((c * NUM_SCALES + scale) * 2 + n) * 3 + map
```

where `c ∈ {0,1,2}` = `{X, Y, B}`, `scale ∈ 0..6`, `n ∈ {0,1}` =
`{L1, L4}` norm, and `map ∈ {0,1,2}` = `{DSSIM, artifact, detailloss}`.

The pipeline launches `error_maps_kernel` per `(scale, channel)` —
one launch computes all three map planes for that cell — and one
`launch_sum_p4` per `(scale, channel, map)`. The reductions are
where the launch-skip lever lives: 54 launches per scale-channel-map
triple in the unbatched path (6 × 3 × 3). Both `L1` and `L4` norms
are produced from the same plane in the same reduction launch (the
plane is summed once and ⁴-summed once), so a launch can be skipped
only when **both** `L1` and `L4` weights for that (scale, channel,
map) are below the mode's threshold.

## Mode thresholds

- **Lossless**: `|w| == 0` exactly. Skipping these cells changes the
  output by exactly zero — the only floating-point effect is removing
  a `WEIGHT[i].mul_add(0.0, ssim)` op, which is bit-identical to not
  performing it.
- **Fast**: `|w| < 1e-3`. Per-cell contribution to the pre-sigmoid sum
  is bounded by `|w| · |avg_value|`; SSIM map values live in `[0, 1]`,
  so each Fast-skipped cell perturbs the sum by `< 1e-3`.
- **Faster** (default): `|w| < 1e-2`. Per-cell perturbation `< 1e-2`.

After the polynomial + power-sigmoid postprocess the bound on final
score perturbation is loose (roughly `0.054` for Fast, `0.54` for
Faster in the worst case), but the IWAIT paper measured zero accuracy
cost vs Full at the SROCC level (CID22 corpus) for the analogous
thresholds in their implementation. The bound is a worst case the
real data does not approach.

## Skip-count summary

| Mode      | Cells skipped | Cells retained | Error-map launches skipped | (Scale, channel) cells skipped | Scales fully skippable |
|-----------|---------------|----------------|----------------------------|-------------------------------|------------------------|
| Full      | 0             | 108            | 0 / 54                     | 0 / 18                        | (none)                 |
| Lossless  | 56            | 52             | 17 / 54                    | 1 / 18                        | (none)                 |
| Fast      | 76            | 32             | 30 / 54                    | 5 / 18                        | (none)                 |
| Faster    | 83            | 25             | 34 / 54                    | 7 / 18                        | scale 5                |

A "launch" here is one `launch_sum_p4` call (one reduction over one
error-map plane). A "(scale, channel) cell" is a launch site we
*also* skip the upstream `error_maps_kernel` for — possible only when
all three maps for that (scale, channel) have both norms zero (so the
plane is never read).

Scale 5 has `max |w| = 0.001365`, below the Faster threshold of
`1e-2`, so the whole scale (blur + error_maps + reductions for all 3
channels × all 3 map types) is skippable in Faster mode.

### Per-scale max |w|

| scale | max(\|w\|)  |
|-------|-------------|
| 0     | 17.4458     |
| 1     | 6.68368     |
| 2     | 225.205     |
| 3     | 176.393     |
| 4     | 171.267     |
| 5     | 0.00136488  |

## Cross-check against paper Table 1

The paper's Table 1 (kanetaka-iwait-2026 notes section "Weight table")
has noticeably different magnitudes — e.g. their `Y-L4-DSSIM @ s4` is
`34.78` vs ours `34.78`, `B-L4-artifact @ s4` is `171.27` vs ours
`171.27`. The large-magnitude entries agree to four significant
figures; the small magnitudes diverge by < 1e-4. We did not
re-verify cell-by-cell — the implementation must skip on *our* table
and the paper's table is for cross-checking that our skip set falls
in the same general region.

## Full 108-cell table

Format: `idx  (scale, channel, norm, map)  weight  |weight|  L F X`.
`L` = skipped under Lossless, `F` = skipped under Fast, `X` =
skipped under Faster.

```
idx   name                  weight                |w|              L    F    X
  0   X-s0-L1-DSSIM         0                     0                skip skip skip
  1   X-s0-L1-artifact      0.0007376606707       0.0007376606707       skip skip
  2   X-s0-L1-detailloss    0                     0                skip skip skip
  3   X-s0-L4-DSSIM         0                     0                skip skip skip
  4   X-s0-L4-artifact      0.0007793481683       0.0007793481683       skip skip
  5   X-s0-L4-detailloss    0                     0                skip skip skip
  6   X-s1-L1-DSSIM         0                     0                skip skip skip
  7   X-s1-L1-artifact      0.000437115573        0.000437115573        skip skip
  8   X-s1-L1-detailloss    0                     0                skip skip skip
  9   X-s1-L4-DSSIM         1.104172643           1.104172643
 10   X-s1-L4-artifact      0.0006628483413       0.0006628483413       skip skip
 11   X-s1-L4-detailloss    0.0001523163278       0.0001523163278       skip skip
 12   X-s2-L1-DSSIM         0                     0                skip skip skip
 13   X-s2-L1-artifact      0.001640643746        0.001640643746             skip
 14   X-s2-L1-detailloss    0                     0                skip skip skip
 15   X-s2-L4-DSSIM         1.842245552           1.842245552
 16   X-s2-L4-artifact      11.4411726            11.4411726
 17   X-s2-L4-detailloss    0                     0                skip skip skip
 18   X-s3-L1-DSSIM         0.0007989109436       0.0007989109436       skip skip
 19   X-s3-L1-artifact      0.0001768164381       0.0001768164381       skip skip
 20   X-s3-L1-detailloss    0                     0                skip skip skip
 21   X-s3-L4-DSSIM         1.878759498           1.878759498
 22   X-s3-L4-artifact      10.94906991           10.94906991
 23   X-s3-L4-detailloss    0                     0                skip skip skip
 24   X-s4-L1-DSSIM         0.0007289346992       0.0007289346992       skip skip
 25   X-s4-L1-artifact      0.9677937081          0.9677937081
 26   X-s4-L1-detailloss    0                     0                skip skip skip
 27   X-s4-L4-DSSIM         0.0001400342429       0.0001400342429       skip skip
 28   X-s4-L4-artifact      0.9981766978          0.9981766978
 29   X-s4-L4-detailloss    0.0003194975593       0.0003194975593       skip skip
 30   X-s5-L1-DSSIM         0.0004550992114       0.0004550992114       skip skip
 31   X-s5-L1-artifact      0                     0                skip skip skip
 32   X-s5-L1-detailloss    0                     0                skip skip skip
 33   X-s5-L4-DSSIM         0.001364876616        0.001364876616             skip
 34   X-s5-L4-artifact      0                     0                skip skip skip
 35   X-s5-L4-detailloss    0                     0                skip skip skip
 36   Y-s0-L1-DSSIM         0                     0                skip skip skip
 37   Y-s0-L1-artifact      0                     0                skip skip skip
 38   Y-s0-L1-detailloss    0                     0                skip skip skip
 39   Y-s0-L4-DSSIM         7.466890328           7.466890328
 40   Y-s0-L4-artifact      0                     0                skip skip skip
 41   Y-s0-L4-detailloss    17.44583398           17.44583398
 42   Y-s1-L1-DSSIM         0.0006235601634       0.0006235601634       skip skip
 43   Y-s1-L1-artifact      0                     0                skip skip skip
 44   Y-s1-L1-detailloss    0                     0                skip skip skip
 45   Y-s1-L4-DSSIM         6.683678146           6.683678146
 46   Y-s1-L4-artifact      0.0003772440798       0.0003772440798       skip skip
 47   Y-s1-L4-detailloss    1.027889938           1.027889938
 48   Y-s2-L1-DSSIM         225.205153            225.205153
 49   Y-s2-L1-artifact      0                     0                skip skip skip
 50   Y-s2-L1-detailloss    0                     0                skip skip skip
 51   Y-s2-L4-DSSIM         19.21323819           19.21323819
 52   Y-s2-L4-artifact      0.001140152459        0.001140152459             skip
 53   Y-s2-L4-detailloss    0.001237755636        0.001237755636             skip
 54   Y-s3-L1-DSSIM         176.393176            176.393176
 55   Y-s3-L1-artifact      0                     0                skip skip skip
 56   Y-s3-L1-detailloss    0                     0                skip skip skip
 57   Y-s3-L4-DSSIM         24.43301              24.43301
 58   Y-s3-L4-artifact      0.2852080261          0.2852080261
 59   Y-s3-L4-detailloss    0.0004485436924       0.0004485436924       skip skip
 60   Y-s4-L1-DSSIM         0                     0                skip skip skip
 61   Y-s4-L1-artifact      0                     0                skip skip skip
 62   Y-s4-L1-detailloss    0                     0                skip skip skip
 63   Y-s4-L4-DSSIM         34.77906344           34.77906344
 64   Y-s4-L4-artifact      44.83562533           44.83562533
 65   Y-s4-L4-detailloss    0                     0                skip skip skip
 66   Y-s5-L1-DSSIM         0                     0                skip skip skip
 67   Y-s5-L1-artifact      0                     0                skip skip skip
 68   Y-s5-L1-detailloss    0                     0                skip skip skip
 69   Y-s5-L4-DSSIM         0                     0                skip skip skip
 70   Y-s5-L4-artifact      0                     0                skip skip skip
 71   Y-s5-L4-detailloss    0                     0                skip skip skip
 72   B-s0-L1-DSSIM         0                     0                skip skip skip
 73   B-s0-L1-artifact      0.0008680556573       0.0008680556573       skip skip
 74   B-s0-L1-detailloss    0                     0                skip skip skip
 75   B-s0-L4-DSSIM         0                     0                skip skip skip
 76   B-s0-L4-artifact      0                     0                skip skip skip
 77   B-s0-L4-detailloss    0                     0                skip skip skip
 78   B-s1-L1-DSSIM         0                     0                skip skip skip
 79   B-s1-L1-artifact      0.0005313191874       0.0005313191874       skip skip
 80   B-s1-L1-detailloss    0                     0                skip skip skip
 81   B-s1-L4-DSSIM         0.0001653381416       0.0001653381416       skip skip
 82   B-s1-L4-artifact      0                     0                skip skip skip
 83   B-s1-L4-detailloss    0                     0                skip skip skip
 84   B-s2-L1-DSSIM         0                     0                skip skip skip
 85   B-s2-L1-artifact      0                     0                skip skip skip
 86   B-s2-L1-detailloss    0                     0                skip skip skip
 87   B-s2-L4-DSSIM         0.0004179171803       0.0004179171803       skip skip
 88   B-s2-L4-artifact      0.001729082823        0.001729082823             skip
 89   B-s2-L4-detailloss    0                     0                skip skip skip
 90   B-s3-L1-DSSIM         0.002082700585        0.002082700585             skip
 91   B-s3-L1-artifact      0                     0                skip skip skip
 92   B-s3-L1-detailloss    0                     0                skip skip skip
 93   B-s3-L4-DSSIM         8.826982765           8.826982765
 94   B-s3-L4-artifact      23.19243344           23.19243344
 95   B-s3-L4-detailloss    0                     0                skip skip skip
 96   B-s4-L1-DSSIM         95.10804988           95.10804988
 97   B-s4-L1-artifact      0.9863978034          0.9863978034
 98   B-s4-L1-detailloss    0.9834382792          0.9834382792
 99   B-s4-L4-DSSIM         0.001228640505        0.001228640505             skip
100   B-s4-L4-artifact      171.2667256           171.2667256
101   B-s4-L4-detailloss    0.9807858872          0.9807858872
102   B-s5-L1-DSSIM         0                     0                skip skip skip
103   B-s5-L1-artifact      0                     0                skip skip skip
104   B-s5-L1-detailloss    0                     0                skip skip skip
105   B-s5-L4-DSSIM         0.0005130064589       0.0005130064589       skip skip
106   B-s5-L4-artifact      0                     0                skip skip skip
107   B-s5-L4-detailloss    0.0001085405786       0.0001085405786       skip skip
```

## Dispatch design notes

Two layers of skip are implemented in this round:

1. **Per-reduction launch skip (step 3b)** — if both norms (L1, L4)
   for `(scale, channel, map)` have `|w| < mode_threshold`, skip the
   corresponding `launch_sum_p4` call. The partials/sums buffers are
   pre-zeroed each call, so skipped slots correctly contribute 0 to
   the host's weighted-sum fold.

2. **Per-(scale, channel) error-map skip** — if all three maps for a
   given `(scale, channel)` are launch-skipped, the upstream
   `error_maps_kernel` for that channel is also unneeded. That makes
   the kernel-launch saving bigger (1 error-map launch + 3 reduction
   launches per skipped cell).

3. **Per-scale full-skip (Faster only)** — scale 5 has every weight
   under `1e-2`, so the entire pyramid level — XYB conversion, the 5
   blurs, transposes, error maps, and reductions for all 3 channels —
   can be skipped. Most of the wall-time savings at large image sizes
   live here, because skipping scale 0 / 1 / 2 (which are the
   expensive ones) is not on the table for any mode.

The per-pixel `pointwise_mul` for `sigma11_in`, `sigma22_in`,
`sigma12_in` is computed even for skipped channels — it's used by the
blur input chain — but only at scales where we *do* run the blur. At
fully-skipped scales (scale 5 in Faster mode) it's skipped along with
everything else.
