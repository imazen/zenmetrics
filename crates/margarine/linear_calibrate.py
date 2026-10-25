"""Score unrounded calibration reconstructions with the native-linear candidate.

Usage: python3 linear_calibrate.py SAVED_PAIRS REPLAY_DIR BINARY NEW_OUTPUT_DIR
Retains every map and compares replayed targets to the frozen block targets.
"""
import csv
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from reference_maps import stats, tiles


def read_exact(stream, count):
    data = bytearray()
    while len(data) < count:
        part = stream.read(count - len(data))
        if not part:
            raise RuntimeError("linear map server ended during reply")
        data.extend(part)
    return data


def main():
    pairs, replay, binary, output = map(Path, sys.argv[1:])
    output.mkdir()
    groups = {}
    target_max_abs = 0.
    max_pixel_error = 0.
    rows = list(csv.DictReader((pairs / "pairs.tsv").open(), delimiter="\t"))
    child = subprocess.Popen([str(binary), "--server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        header = json.loads(child.stdout.readline())
        if header["input"] != "linear-rgb-f32":
            raise RuntimeError("server lacks native-linear ingress")
        for row in rows:
            if "validation" in row["image"]:
                continue
            i, w, h = row["index"], int(row["width"]), int(row["height"])
            for tag, side in ((b"R", "ref"), (b"D", "dist")):
                pixels = (replay / f"{i}.{side}.linear.f32").read_bytes()
                if len(pixels) != w*h*12:
                    raise RuntimeError(f"{i}: invalid native pixel length")
                child.stdin.write(tag + struct.pack("<II", w, h) + pixels)
            child.stdin.flush()
            score = struct.unpack("<d", read_exact(child.stdout, 8))[0]
            raw = read_exact(child.stdout, w*h*4)
            (output / f"{i}.margarine.f32").write_bytes(raw)
            v = np.frombuffer(raw, "<f4").astype(np.float64)
            if not np.isfinite(v).all() or score != v.max():
                raise RuntimeError(f"{i}: invalid map or max")
            loop = np.fromfile(replay / f"{i}.loop.f32", "<f4").astype(np.float64)
            target = np.fromfile(pairs / f"{i}.butter.f32", "<f4").astype(np.float64)
            target_max_abs = max(target_max_abs, float(abs(tiles(loop, w, h)-target).max()))
            max_pixel_error = max(max_pixel_error, float(abs(loop-v).max()))
            t = tiles(v, w, h)
            for pool in ("bulk", "tail"):
                if pool == "bulk":
                    k = (t > 0) & (target >= .05)
                    x, y = t[k], target[k]
                else:
                    if not (t.max() > 0 and target.max() > 0):
                        continue
                    x, y = [t.max()], [target.max()]
                a, b = groups.setdefault(f"{row['set']}/{pool}", ([], []))
                a.extend(x)
                b.extend(y)
            print(i, "native map retained", flush=True)
    finally:
        child.stdin.close()
        rc = child.wait()
    if rc:
        raise RuntimeError(f"server exited {rc}")
    # f64 numpy reduction versus the producer's f32 final rounding.
    if target_max_abs > 1e-5:
        raise RuntimeError(f"replay changes target blocks: {target_max_abs}")
    summary = dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   header=header, replay_target_max_abs=target_max_abs,
                   max_pixel_error=max_pixel_error,
                   calibration={k: stats(*v) for k, v in groups.items()})
    with (output / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
