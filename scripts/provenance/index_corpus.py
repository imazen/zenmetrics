#!/usr/bin/env python3
"""Content-address + index an image corpus into a TSV (provenance foundation).

Per the ML Data Pipeline Discipline: every corpus dir gets a hash index so
"where did this come from / is it still valid / what is its original" is a TSV
lookup, not a forensic hunt. Emits one row per image with sha256 + geometry +
size_class (+ optional R2 key and original linkage).

Usage:
  index_corpus.py --dir <corpus> --out <index.tsv>
                  [--r2-prefix s3://bucket/path]      # canonical R2 location
                  [--name-dims]                       # parse WxH from <stem>.scale<W>x<H>.png
                  [--jobs N]                          # parallel hashing (default 8)

Columns: name  sha256  bytes  source_stem  width  height  px  size_class  r2_key
size_class bins match select_corpus.py / the trainer: tiny<=4096, small<=65536,
medium<=1048576, else large.
"""
import argparse, hashlib, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCALE_RE = re.compile(r"scale(\d+)x(\d+)", re.IGNORECASE)
TINY, SMALL, MEDIUM = 4096, 65536, 1048576


def size_class(px: int) -> str:
    if px <= TINY: return "tiny"
    if px <= SMALL: return "small"
    if px <= MEDIUM: return "medium"
    return "large"


def source_stem(name: str) -> str:
    return name.split(".scale")[0]


def sha256_file(p: Path) -> tuple[str, int]:
    h = hashlib.sha256(); n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk); n += len(chunk)
    return h.hexdigest(), n


def dims_from_name(name: str):
    m = SCALE_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def dims_from_image(p: Path):
    # lazy: only used when --name-dims can't parse; needs Pillow
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except Exception:
        return (None, None)


def row(p: Path, name_dims: bool, r2_prefix: str | None) -> str:
    name = p.name
    sha, nbytes = sha256_file(p)
    w, h = dims_from_name(name) if name_dims else (None, None)
    if w is None:
        w, h = dims_from_image(p)
    px = (w * h) if (w and h) else 0
    sc = size_class(px) if px else "unknown"
    r2 = f"{r2_prefix.rstrip('/')}/{name}" if r2_prefix else ""
    return "\t".join(str(x) for x in [name, sha, nbytes, source_stem(name), w or "", h or "", px or "", sc, r2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--r2-prefix", default=None)
    ap.add_argument("--name-dims", action="store_true", help="parse WxH from scale<W>x<H> in the filename (fast, no decode)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--glob", default="*.png")
    args = ap.parse_args()

    files = sorted(args.dir.glob(args.glob))
    if not files:
        print(f"no files match {args.dir}/{args.glob}", file=sys.stderr); sys.exit(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rows = list(ex.map(lambda p: row(p, args.name_dims, args.r2_prefix), files))
    with open(args.out, "w") as f:
        f.write("name\tsha256\tbytes\tsource_stem\twidth\theight\tpx\tsize_class\tr2_key\n")
        f.write("\n".join(rows) + "\n")

    # summary
    from collections import Counter
    sc = Counter(r.split("\t")[7] for r in rows)
    stems = len({r.split("\t")[3] for r in rows})
    print(f"indexed {len(rows)} files ({stems} source stems) -> {args.out}")
    print("  size_class:", dict(sc))


if __name__ == "__main__":
    main()
