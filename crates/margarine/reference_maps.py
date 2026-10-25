"""Compare retained reference-audit maps and the encoder's saved block targets.

Usage: python3 reference_maps.py PAIRS_DIR MAPS_DIR OUTPUT.json
Only pixel payloads and split/path metadata are read; no human labels.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def tiles(v, w, h):
    bx, by = (w + 7) // 8, (h + 7) // 8
    p = np.zeros((by * 8, bx * 8), dtype=np.float64)
    n = np.zeros_like(p)
    p[:h, :w] = v.reshape(h, w)
    n[:h, :w] = 1
    s = (p ** 16).reshape(by, 8, bx, 8).sum((1, 3))
    c = n.reshape(by, 8, bx, 8).sum((1, 3))
    return (1.2 * (s / c) ** (1 / 16)).ravel()


def stats(x, y):
    x, y = np.log(x), np.log(y)
    return dict(n=len(x), log_bias=float(np.mean(x-y)),
                log_rmse=float(np.sqrt(np.mean((x-y)**2))),
                log_r=float(np.corrcoef(x, y)[0, 1]))


def main():
    pairs, maps, out = map(Path, sys.argv[1:])
    groups = {}
    comparisons = {k: dict(pairs=0, differing_pixels=0, max_abs=0.)
                   for k in ("teacher", "fast", "exact")}
    for r in csv.DictReader((pairs / "pairs.tsv").open(), delimiter="\t"):
        if "validation" in r["image"]:
            continue
        i, w, h = r["index"], int(r["width"]), int(r["height"])
        target = np.fromfile(pairs / f"{i}.butter.f32", "<f4").astype(np.float64)
        loop = np.fromfile(maps / f"{i}.loop.f32", "<f4")
        for mode in ("loop", "teacher", "fast", "exact"):
            v = np.fromfile(maps / f"{i}.{mode}.f32", "<f4")
            if mode != "loop":
                c = comparisons[mode]
                c["pairs"] += 1
                c["differing_pixels"] += int(np.count_nonzero(loop.view("u4") != v.view("u4")))
                c["max_abs"] = max(c["max_abs"], float(abs(loop-v).max()))
            t = tiles(v.astype(np.float64), w, h)
            for pool in ("bulk", "tail"):
                k = (t > 0) & (target >= .05)
                if pool == "tail" and not (t.max() > 0 and target.max() > 0):
                    continue
                x, y = (t[k], target[k]) if pool == "bulk" else ([t.max()], [target.max()])
                key = f"{mode}/{r['set']}/{pool}"
                a, b = groups.setdefault(key, ([], []))
                a.extend(x)
                b.extend(y)
    result = dict(comparisons=comparisons, calibration={k: stats(*v) for k, v in groups.items()})
    with out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
