#!/usr/bin/env python3
"""WaDIQaM score-direction check on a monotone JPEG ladder.

Builds `<ref>.jpeg` distortions at q95..q3 for a set of reference PNGs,
scores every `wadiqam_{fr,nr}_{live,tid}` column with `zenmetrics batch`
(one call per column over a shared pairs TSV), then reports per-source
Spearman rho of score vs degradation. All four columns are expected to
RISE as quality drops: fr_* are distance-like, nr_* are DMOS-like —
measured, not labelled (TID2013's MOS labels do not transfer: nr_tid
still rises, median rho +0.79 over 12 KonJND references).

usage: direction.py --zenmetrics BIN --refs 'DIR/*.png' --work DIR

The JPEG ladder needs PIL (`Image.save(quality=q)`). Receipt from the
port branch: `benchmarks/wadiqam_direction_2026-10-16.txt`.
"""
import argparse
import csv
import statistics
import subprocess
import sys
from glob import glob
from pathlib import Path

QS = [95, 80, 60, 40, 25, 15, 8, 3]
COLUMNS = ["wadiqam_fr_live", "wadiqam_fr_tid", "wadiqam_nr_live", "wadiqam_nr_tid"]


def spearman(a, b):
    def rk(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for k, i in enumerate(order):
            r[i] = k
        return r

    ra, rb = rk(a), rk(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenmetrics", required=True, help="built zenmetrics binary")
    ap.add_argument("--refs", required=True, help="glob of reference PNGs")
    ap.add_argument("--work", required=True, help="scratch dir for the ladder + TSVs")
    ap.add_argument("--jobs", default="1")
    a = ap.parse_args()

    from PIL import Image  # noqa: E402

    work = Path(a.work)
    ladder = work / "ladder"
    ladder.mkdir(parents=True, exist_ok=True)
    rows = []
    for ref in sorted(glob(a.refs)):
        ref = Path(ref)
        src = ref.stem.split("_")[0]
        img = Image.open(ref).convert("RGB")
        for q in QS:
            dist = ladder / f"d_{src}_{q}.png"
            if not dist.exists():
                jpeg = ladder / f".{src}_{q}.jpg"
                img.save(jpeg, quality=int(q))
                Image.open(jpeg).save(dist)  # decode then re-encode PNG
                jpeg.unlink()
            rows.append({"ref_path": str(ref), "dist_path": str(dist),
                         "src": src, "q": q})
    pairs = work / "pairs.tsv"
    with pairs.open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["ref_path", "dist_path", "src", "q"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    for col in COLUMNS:
        out = work / f"out_{col}.tsv"
        if not out.exists():
            subprocess.run(
                [a.zenmetrics, "batch", "--metric", col.replace("_", "-"),
                 "--pairs", str(pairs), "--output", str(out),
                 "--jobs", a.jobs],
                check=True)
        scored = {r["dist_path"]: r for r in
                  csv.DictReader(out.open(), delimiter="\t")}
        for r in rows:
            r[col] = scored[r["dist_path"]][col]
            if col.startswith("wadiqam_nr"):
                r[col + "_ref"] = scored[r["dist_path"]][col + "_ref"]

    for col in COLUMNS:
        rhos = []
        print(col)
        for src in sorted({r["src"] for r in rows}):
            rs = sorted((r for r in rows if r["src"] == src), key=lambda r: -int(r["q"]))
            sc = [float(r[col]) for r in rs]
            rho = spearman([-int(r["q"]) for r in rs], sc)
            rhos.append(rho)
            ref_note = f" | ref {rs[0][col + '_ref']}" if col + "_ref" in rs[0] else ""
            print(f"  {src} rho={rho:+.2f} "
                  + " ".join(f"{v:.1f}" for v in sc) + ref_note)
        pos = sum(r > 0 for r in rhos)
        print(f"  median rho {statistics.median(rhos):+.2f}; "
              f"srcs rho>0: {pos}/{len(rhos)}")
        if pos < len(rhos) * 2 // 3:
            sys.exit(f"{col}: direction check failed "
                     f"(only {pos}/{len(rhos)} sources rise with distortion)")


if __name__ == "__main__":
    main()
