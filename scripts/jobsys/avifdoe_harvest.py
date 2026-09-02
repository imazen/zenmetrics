#!/usr/bin/env python3
"""avifdoe_harvest.py — reduce the AVIF-DOE score blobs + encode ledgers to ONE tidy table.

The DOE's four encode runs (a0r/a1/a2/ag) and its score run
(avifdoe-svt-sf-cpu-20260902) are joined here into a single parquet keyed on
`encode_sha`, which is the content address of the encoded AVIF bitstream and
therefore the only stable identity across the two sides.

WHAT THIS DOES NOT DO: statistics. Every SROCC/PLCC/CI in the Stage-A report
comes from the canonical owner (`zenstats` via `panel` / `scripts/lib/zen_stats.py`).
This script only reshapes bytes into rows.

HETEROGENEOUS ROWS ARE FIRST-CLASS. The score run changed metric sets mid-flight
(DOE plan §13.5: 3-metric before the 907-blob boundary at 2026-09-02T05:42Z,
2-metric after), so a row carries `metrics_present` and every metric column is
nullable. NEVER filter on a fixed metric set or a fixed row count.

`zensim` is emitted as a 720-wide FEATURE vector (`kind:"feature"`), not a
scalar — so `ssim2` is the only corpus-wide scalar quality response, exactly as
the plan's §7.1 designates it. `--with-features` keeps the vectors.

Usage:
  avifdoe_harvest.py --score-dir DIR --sizes sizes.tsv --pairs a1=P1.tsv ... --out out.parquet
"""
import argparse, collections, glob, json, os, sys

def parse_label(label):
    """'s4-svt-420-acb1-mtx32' -> (speed=4, chroma='420', devs=['acb1','mtx32'])."""
    parts = label.split("-")
    if len(parts) < 3 or not parts[0].startswith("s"):
        return None, None, None
    try:
        speed = int(parts[0][1:])
    except ValueError:
        return None, None, None
    return speed, parts[2], [p for p in parts[3:] if p]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", required=True, help="dir of score blobs (NDJSON, one per chunk)")
    ap.add_argument("--sizes", required=True, help="TSV: <encode_sha_uri>\\t<bytes>")
    ap.add_argument("--pairs", action="append", default=[], help="run=path.tsv (zenfleet-ctl pairs output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-features", action="store_true", help="retain the 720-wide zensim vectors")
    a = ap.parse_args()

    # --- cell metadata, from the canonical `zenfleet-ctl pairs` tables -------
    cell = {}                      # encode_sha_uri -> dict
    for spec in a.pairs:
        run, path = spec.split("=", 1)
        with open(path) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            ix = {c: i for i, c in enumerate(hdr)}
            for line in f:
                r = line.rstrip("\n").split("\t")
                kt = json.loads(r[ix["knob_tuple_json"]])
                lbl = kt.get("cell")
                speed, chroma, devs = parse_label(lbl)
                cell[r[ix["encode_sha"]]] = dict(
                    run=run, image=r[ix["image_path"]], q=int(r[ix["q"]]),
                    arm=lbl, plan=kt.get("plan"), fp=kt.get("fp"),
                    speed=speed, chroma=chroma,
                    devs="|".join(sorted(devs)) if devs is not None else None,
                    n_dev=len(devs) if devs is not None else None,
                )
    print(f"cells from pairs tables: {len(cell)}", file=sys.stderr)

    # --- encoded bytes, from the object listing ------------------------------
    size = {}
    with open(a.sizes) as f:
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            size[k] = int(v)
    print(f"encode sizes: {len(size)}", file=sys.stderr)

    # --- scores -------------------------------------------------------------
    rows = {}                      # encode_sha -> row
    nblob = nline = 0
    dropped_no_cell = collections.Counter()
    for p in sorted(glob.glob(os.path.join(a.score_dir, "*"))):
        if not os.path.isfile(p):
            continue
        nblob += 1
        with open(p, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nline += 1
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                esha = d.get("encode_sha")
                if not esha:
                    continue
                r = rows.get(esha)
                if r is None:
                    meta = cell.get(esha)
                    if meta is None:
                        dropped_no_cell[esha.rsplit("/", 3)[1] if "/" in esha else "?"] += 1
                        continue
                    r = dict(meta)
                    r["encode_sha"] = esha
                    r["bytes"] = size.get(esha)
                    r["_metrics"] = set()
                    rows[esha] = r
                if d.get("kind") == "metric":
                    m = d.get("metric")
                    if m:
                        r["_metrics"].add(m)
                        r[f"m_{m}"] = d.get("score")
                    for k, v in (d.get("scores") or {}).items():
                        r["_metrics"].add(k)
                        r.setdefault(f"m_{k}", v)
                elif d.get("kind") == "feature":
                    r["_metrics"].add("zensim_features")
                    r["feature_regime"] = d.get("regime")
                    if a.with_features:
                        r["features"] = d.get("features")
    print(f"score blobs read: {nblob}  ndjson lines: {nline}  joined rows: {len(rows)}", file=sys.stderr)
    if dropped_no_cell:
        print(f"WARNING dropped (no cell metadata) by run: {dict(dropped_no_cell)}", file=sys.stderr)

    metric_cols = sorted({k for r in rows.values() for k in r if k.startswith("m_")})
    for r in rows.values():
        r["metrics_present"] = "|".join(sorted(r.pop("_metrics")))
        for c in metric_cols:
            r.setdefault(c, None)
        r.setdefault("feature_regime", None)
        if a.with_features:
            r.setdefault("features", None)

    import pyarrow as pa, pyarrow.parquet as pq
    tbl = pa.Table.from_pylist(list(rows.values()))
    pq.write_table(tbl, a.out, compression="zstd")
    print(f"wrote {a.out}: {tbl.num_rows} rows x {tbl.num_columns} cols", file=sys.stderr)
    print("metrics_present histogram:", file=sys.stderr)
    for k, v in collections.Counter(r["metrics_present"] for r in rows.values()).most_common():
        print(f"   {v:>7}  {k}", file=sys.stderr)
    print("rows per run:", dict(collections.Counter(r["run"] for r in rows.values())), file=sys.stderr)
    print("rows missing bytes:", sum(1 for r in rows.values() if r["bytes"] is None), file=sys.stderr)

if __name__ == "__main__":
    main()
