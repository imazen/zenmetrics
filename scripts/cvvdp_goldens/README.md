# cvvdp goldens

Produces per-stage golden tensors and final JOD scores from the pinned
ColorVideoVDP Python reference, for `cvvdp-gpu`'s parity tests.

The repo intentionally does NOT carry the golden binaries — they live
in R2 under `s3://coefficient/cvvdp-goldens/<version>/`, fetched on
first test run and cached under `$XDG_CACHE_HOME/zenmetrics-cvvdp-goldens/`.

## Pin

ColorVideoVDP v0.5.4 (gfxdisp/ColorVideoVDP).

When updating the pin: bump `requirements.txt`, rerun `build_goldens.py`,
upload to a new R2 prefix (`v2`, `v3`, ...), and update `GOLDEN_VERSION`
+ `MANIFEST_URL` in `crates/cvvdp-gpu/tests/common/mod.rs`. Old prefixes
stay live so older commits keep building.

## Local one-shot

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 build_goldens.py \
    --pairs pairs.json \
    --image-root ../../crates/zenmetrics-corpus/data \
    --out ./out

# inspect ./out/manifest.json, then:
./upload_to_r2.sh ./out v1
```

`pairs.json` is a list of `{name, ref, dist}` entries pointing at PNGs
relative to `--image-root`. Default corpus is `zenmetrics-corpus`'s
`source.png` paired with each `q*.jpg`.

## What gets uploaded

- `manifest.json` — names, hashes, shapes for every artifact.
- `<name>.final.json` — final JOD plus pycvvdp's internal stats.
- `<name>.stage_*.bin` — per-stage f32 tensors (added when per-stage
  taps land in `build_goldens.py`).
