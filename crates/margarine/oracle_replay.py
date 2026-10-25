"""Independent djxl readback of the calibration replay's retained JXL bytes.

Usage: python3 oracle_replay.py PAIRS_DIR REPLAY_DIR NEW_OUTPUT.json
The decoder's PPM output stays in memory; only hashes and pixel diffs persist.
The saved calibration pixels are iteration zero; the bitstream is encoded after
the quantization update. Their difference is diagnostic, not a parity oracle.
"""
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    pairs, replay, output = map(Path, sys.argv[1:])
    results = []
    decoder_arguments = ["-", "--output_format=ppm", "--bits_per_sample=8",
                         "--num_threads=1", "--quiet", "--color_space=RGB_D65_SRG_Rel_SRG"]
    for r in csv.DictReader((pairs / "pairs.tsv").open(), delimiter="\t"):
        if "validation" in r["image"]:
            continue
        i, w, h = r["index"], int(r["width"]), int(r["height"])
        command = ["djxl", str(replay / f"{i}.jxl"), *decoder_arguments]
        out = subprocess.run(command, check=True, capture_output=True)
        with Image.open(io.BytesIO(out.stdout)) as image:
            if image.size != (w, h) or image.mode != "RGB":
                raise RuntimeError(f"{i}: unexpected decoded pixel format or geometry")
            decoded = np.asarray(image).copy()
        expected = np.frombuffer((pairs / f"{i}.dist.rgba").read_bytes(), np.uint8).reshape(h, w, 4)[:, :, :3]
        diff = abs(decoded.astype(np.int16)-expected.astype(np.int16))
        results.append(dict(index=i, width=w, height=h,
                            jxl_sha256=hashlib.sha256((replay / f"{i}.jxl").read_bytes()).hexdigest(),
                            rgb_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
                            iteration_zero_differing_samples=int(np.count_nonzero(diff)),
                            iteration_zero_max_abs=int(diff.max())))
        print(i, "independent decode complete", flush=True)
    version = subprocess.run(["djxl", "--version"], check=True, capture_output=True, text=True)
    result = dict(decoder=(version.stdout+version.stderr).strip(),
                  decoder_sha256=hashlib.sha256(Path(shutil.which("djxl")).read_bytes()).hexdigest(),
                  decoder_arguments=decoder_arguments,
                  validation="all bitstreams decode as expected RGB8 geometry",
                  comparison="diagnostic: final bitstream versus iteration-zero reconstruction",
                  pairs=len(results),
                  iteration_zero_exact_pairs=sum(r["iteration_zero_differing_samples"] == 0 for r in results),
                  iteration_zero_max_abs=max(r["iteration_zero_max_abs"] for r in results), rows=results)
    with output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
