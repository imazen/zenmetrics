# HDR-VDP canonical-configuration search: data provenance (2026-10-19)

Step 1 of the `hdrvdpres` lane: which files hold the published `HDR_VDP_2` and
`HDR_VDP_3` columns, and which pixels the 61-pair panel scores. Checked on
r5900xt and dev, 2026-10-19 02:20–02:40 UTC. Scripts: [`scripts/`](scripts/).
No per-image published values are committed here.

## Sources

| copy | origin | retrieved | bytes | sha256 |
|---|---|---|---|---|
| `metrics_cropped.csv` (DaRUS v2.0 original) | [datafile 603368](https://darus.uni-stuttgart.de/api/access/datafile/603368?format=original) | 2026-10-19 | 11,501,265 | see below |
| `metrics_fullres.csv` (DaRUS v2.0 original) | [datafile 603369](https://darus.uni-stuttgart.de/api/access/datafile/603369?format=original) | 2026-10-19 | 11,425,824 | see below |
| `metrics_cropped.csv` (site) | [`/aic2026/api/metrics-csv/cropped`](https://merkur242.inf.uni-konstanz.de/aic2026/api/metrics-csv/cropped), header `x-local-updated-at: 2026-10-05T17:29:52.399Z` | 2026-10-19 | 11,571,496 | see below |
| `metrics_fullres.csv` (site) | [`/aic2026/api/metrics-csv/fullres`](https://merkur242.inf.uni-konstanz.de/aic2026/api/metrics-csv/fullres), header `x-local-updated-at: 2026-10-05T17:29:51.213Z` | 2026-10-19 | 11,496,884 | see below |
| site metric config | [`/aic2026/api/metric-eval/config`](https://merkur242.inf.uni-konstanz.de/aic2026/api/metric-eval/config) | 2026-10-19 | 21,676 | see below |
| dataset metadata | [DaRUS dataset API](https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/DARUS-6156), [DOI 10.18419/DARUS-6156](https://doi.org/10.18419/DARUS-6156) | 2026-10-19 | 26,389 | see below |
| dataset README | [datafile 639751](https://darus.uni-stuttgart.de/api/access/datafile/639751) | 2026-10-19 | 6,199 | see below |
| encoding recipes | [datafile 565360](https://darus.uni-stuttgart.de/api/access/datafile/565360) | 2026-10-19 | 17,907 | see below |
| cropped stimuli zip | DaRUS file `AIC2026-dataset-cropped.zip` (id 554517, MD5 `f9010c61602bf631a049d167d3f4cf18`), copy on dev `/mnt/v/datasets/aic2026/` fetched 2026-09-19 | 2026-09-19 | 9,295,978,208 | MD5 matches DaRUS |
| July label table | `aic2026_public_split_preliminary_jnd_reconstruction.csv`, private; on r5900xt `~/tmp/iqa-cx-lead/private-aic/` | file mtime 2026-10-17 | 240,567 | see below |
| paper | [arXiv:2607.22783](https://arxiv.org/abs/2607.22783) | 2026-09-25 | — | — |

DaRUS reported dataset version 2.0, released 2026-07-30T08:21:58Z, last
updated 2026-07-30T10:28:43Z; both metrics files were created there on
2026-07-02. Every downloaded file's MD5 equals the checksum DaRUS publishes.

Full sha256 values:

```
3fb7208165c039cdc4bdea2e1244ea52020b60019c9c96f4b00a9b9967a7ad99  metrics_cropped.csv (DaRUS v2.0)
786e1647d2a8618672b5ed2336b4892a21ba278da47d12722d662e22ba010951  metrics_fullres.csv (DaRUS v2.0)
1bd59115b54c7ca420f8a1bb4b839c56a7025ac351da3baa1a99aed1ac8ce06d  metrics_cropped.csv (site, updated 2026-10-05)
067290cadbbf87e7ac7dbeb06e68d42a6345a018e7a5f287b72a13decf0fb75c  metrics_fullres.csv (site, updated 2026-10-05)
19184a8e26411478fc1d29bc9748c0c24725b5366b23d00729a7b7191e821687  config.json (site, fetched 2026-10-19)
fd2ef91a38b400c0012f38145be92f8bbc289e83f85ef1f54701a5767d5a5447  config.json (site, fetched 2026-10-15, used by aic_confsearch)
3ec56822e0ebac8e032f413e8a7ed0878783211720f80064360f43ce9d4fab5a  DaRUS dataset metadata v2.0 JSON
017a28447c465a9786dfdddaab28f7b2d14b73023134bd84c5a6ec9d9aade5f1  readme_AIC2026.md
39804bbfca9b4804a1f8d8d6a5bbcfd8d1d56fce5e6363f7ed1a9ede67b98b16  encoding_recipes.md
fc30508ae397f615193555b7aeaf820c0dde5e02a2f2ede8032b4d3819ce9e29  metrics_subset.csv (61 rows, built 2026-10-15)
9a017bcaa1b9ffc127a3daea1cc854a303708bd2091f4ba45ef78c9f2bbf82fa  pairs.tsv (aicfix 61-pair list)
bd5ea3c55e3eb3ff1b9e316ae63d0e84ce6c5b521058643df05f9d9de5c6b877  aic2026_public_split_preliminary_jnd_reconstruction.csv
```

The CSVs the earlier lanes used (`aic_confsearch_2026-10-17`,
`aic_refusals_2026-10-18`; r5900xt `~/tmp/aicfix-data/data/aic2026/`, dev
`/mnt/v/datasets/aic2026/`) are byte-identical to the DaRUS v2.0 originals.

## What the HDR columns are, across copies

`scripts/cmp_hdr_cols.py`, keyed on the distorted name without `PTC_`:

- **`HDR_VDP_2` and `HDR_VDP_3` are identical**, string for string on all
  9,618 rows, in DaRUS v2.0 `metrics_cropped.csv`, the site's
  `metrics_cropped.csv` (updated 2026-10-05), the Dataverse `.tab` export
  downloaded on 2026-09-25, and the 61-row `metrics_subset.csv`. The site
  CSVs carry 83 columns against DaRUS's 80; the three extra columns do not
  touch the HDR values.
- **`metrics_fullres.csv` differs from `metrics_cropped.csv` on 4,368 rows,
  exactly the rows of 32 sources** (largest differences: 9.26 Q for
  `HDR_VDP_2`, 1.35 for `HDR_VDP_3`). The other 38 sources agree on every
  row. 32 and 38 are the README's category counts: category 3 (larger than
  840×944, cropped only for the subjective test) against categories 1 and 2
  (already 840×944 before compression). So the fullres columns were scored
  on the full decoded images and the cropped columns on the crops.
- **The July reconstruction CSV has no HDR columns.** Its columns are
  `split, source, distorted, JND_score, JND_pointwise` (5,768 rows): it
  supplies the subjective labels, not metric values. Its use of the HDR
  columns can only be by join on `distorted`.
- Published HDR values print 6 decimals, which sets the exact-reproduction
  gate.

## The 61 crop pairs

The 61 pairs (61 sources, 16 codecs; one row in every 160 of
`metrics_fullres.csv`, built 2026-10-15 for
[nn_aic2026_crosscheck](../nn_aic2026_crosscheck_2026-10-15.md)) are 122 PNGs.
Their sha256 list is [`crop_img_sha256.txt`](crop_img_sha256.txt).

- `scripts/zip_member_sha.py` hashed the same names inside dev's
  `AIC2026-dataset-cropped.zip`: **all 122 are byte-identical to the members
  under `PTC-crops-all/`** (zip timestamps 2026-04-30 to 2026-05-29).
- All 122 are 840×944, 8-bit RGB (colour type 2), non-interlaced
  (`scripts/png_chunks.py`). 103 carry an `sRGB` chunk; 19 carry no
  ancillary chunk at all (17 references plus `PTC_S40_JAIB_05.png` and
  `PTC_S48_JAI_13.png`). None has `iCCP`, `gAMA` or `cHRM`, so a decoder
  that honours colour chunks has nothing to apply beyond the sRGB flag.
- 27 of the 61 pairs come from category-3 sources (their fullres and cropped
  HDR values differ), 34 from categories 1 and 2. For the 34, the crop is the
  whole image, so cropped and fullres scores must agree; for the 27, the
  published cropped value was scored on the crop.

## Not done in this step

- The `AIC2026-dataset-cropped.zip` hash was checked against DaRUS's MD5,
  not re-hashed with sha256.
- Steps 2–6 of the lane (upstream HDR-VDP sources, baseline reproduction,
  configuration search, port change) did not start: the lane was wound down
  by the owner after this step. The remaining work is listed in the lane
  report.

## Successor upstream evidence

Official release archives, hashes, CTC conditions, and the retained Octave
compatibility audit are recorded in [UPSTREAM.md](UPSTREAM.md).
