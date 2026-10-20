# Per-pair calculations

These seven CSVs retain all 2,074 observations from the [fresh AIC61 panel](../README.md): 1,952 scored comparisons and 122 matching size refusals. Each CSV has 61 pairs and grouped columns `<variant>__ours`, `__oracle`, `__abs_delta`, and `__status`. Numeric cells use 17 significant digits, preserving binary64 round trips. Refusals have empty numeric cells and `matching_size_refusal` status; they are not zero scores. Pair identifiers come from distorted-image stems in the bound input manifest.

No scoring was repeated for this export. The original Rust JSON and PyIQA/Octave TSV hashes are recorded in the parent report. `export_csv.py RAW_DIRECTORY OUTPUT_DIRECTORY` reconstructs these files from `rust.jsonl`, `pyiqa.tsv`, and `octave.tsv`, verifies unique observations and expected refusal reasons, and writes a size/hash manifest. Gzip sizes below were measured using Python gzip level 9 with mtime=0; the repository keeps plaintext CSVs.

| File | Variants | CSV bytes | Gzip bytes | SHA256 |
|---|---:|---:|---:|---|
| [ssim-control.csv](ssim-control.csv) | 1 | 5274 | 2218 | `649f114e8f4e1305596b11bb52785b125afe25d85392c57eea1d4569eb0fd6ca` |
| [ssim-y.csv](ssim-y.csv) | 6 | 27086 | 10790 | `6f056dc366a36183fb98e7fcf58d901ff3b8d186371500b6ad9ee0bd7792c2f4` |
| [ssim-rgb.csv](ssim-rgb.csv) | 6 | 27094 | 10890 | `84ff951812d2e473e62edbe78924676cb3713cdb6e7f949dba315f6cec6c000e` |
| [ms-ssim.csv](ms-ssim.csv) | 6 | 21692 | 7390 | `ee5afb3ff4bf25befdc056d4b142f1092e1fa62757fbb6183a871ca38c965357` |
| [fsim.csv](fsim.csv) | 5 | 22527 | 7574 | `0c54588761f05e6073ed72ecb29b98f51eb3c4b0bb09065bbaca1ecc5a9e2005` |
| [fsimc.csv](fsimc.csv) | 5 | 22587 | 7651 | `016019261c71f610f0dccd7654a2a4e73d91a25f1cc9c37d36fdf34e8d43e7ac` |
| [vsi.csv](vsi.csv) | 5 | 22511 | 7469 | `603dfb20c738f572055689282f534ead58f66dfd92656217c68b80582653cbeb` |

Total plaintext: **148,771 bytes**; individually gzipped total: **53,982 bytes**. Every tracked generated CSV is below 30,000 bytes. A single long-form CSV prepared outside git is 200,879 bytes, or 56,535 bytes gzipped.

## Decoding and future reruns

The original Rust panel invoked `score` separately per pair/metric and therefore decoded each input repeatedly. Its driver did not use the existing resident-bitmap path. PyIQA loaded each pair once before its variant loop. The Octave factor-major panel reread the prepared raw RGB8 files for each factor; it did not repeatedly decode PNG.

The CLI `compare` command decodes each unique path once, then passes those decoded RGB8 images to every requested metric. Invoke it once per pair to bound resident image memory before moving to the next pair:

```sh
zenmetrics compare --reference ref.png --variant distorted.png \
  --metric ssim-pyiqa-f32 --metric ssim-pyiqa-f32-k2 \
  --metric fsim --metric vsi --output json
```

Select the required metrics explicitly for the new panel. Use JSON to retain full precision; the display TSV rounds to six decimals. FSIM emits both FSIM and FSIMc columns. Keep new results and their executable/source identities separate from this frozen panel; expected refusals must remain explicit. Reusing decoded RGB8 does not imply all metric-specific luma conversions or reference features are shared.
