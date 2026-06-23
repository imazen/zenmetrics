#!/usr/bin/env python3
"""Hash + index the ORIGINAL source images via the imazen-26 manifest.

Completes the rendition->original provenance chain: renditions_index.source_stem
joins to sources_index.stem, whose row carries the original's sha256 + dims +
content_class + path. Reads the manifest (stem -> original path) so we index
exactly the referenced originals.

Manifest schema (imazen26_manifest.tsv): stem  split  content_class  source  path
  (col1 'stem' is the rendition source_stem; col5 'path' is the original .sdr.png)

Usage:
  index_sources.py --manifest <manifest.tsv> --out <sources_index.tsv> [--jobs 8]

Columns: stem  sha256  bytes  width  height  px  content_class  source  split  original_path
"""
import argparse, hashlib, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_file(p: Path):
    h = hashlib.sha256(); n = 0
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk); n += len(chunk)
        return h.hexdigest(), n
    except OSError:
        return "MISSING", 0


def dims(p: Path):
    try:
        out = subprocess.run(["identify", "-format", "%w %h", str(p)],
                             capture_output=True, text=True, timeout=60)
        w, h = out.stdout.split()[:2]
        return int(w), int(h)
    except Exception:
        return 0, 0


def row(rec):
    stem, split, content_class, source, path = rec
    p = Path(path)
    sha, nbytes = sha256_file(p)
    w, h = dims(p) if sha != "MISSING" else (0, 0)
    return "\t".join(str(x) for x in [stem, sha, nbytes, w, h, w * h, content_class, source, split, path])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    recs = []
    for ln in args.manifest.read_text().splitlines():
        c = ln.rstrip("\n").split("\t")
        if len(c) < 5 or c[0] == "sha256" or c[0] == "stem":  # skip header
            continue
        recs.append((c[0], c[1], c[2], c[3], c[4]))
    if not recs:
        print("no manifest rows parsed", file=sys.stderr); sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rows = list(ex.map(row, recs))
    with open(args.out, "w") as f:
        f.write("stem\tsha256\tbytes\twidth\theight\tpx\tcontent_class\tsource\tsplit\toriginal_path\n")
        f.write("\n".join(rows) + "\n")

    missing = sum(1 for r in rows if "\tMISSING\t" in r)
    big = sum(1 for r in rows if int(r.split("\t")[5] or 0) > 1_048_576)
    print(f"indexed {len(rows)} originals -> {args.out}  (missing={missing}, >1MP={big})")


if __name__ == "__main__":
    main()
