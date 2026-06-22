#!/usr/bin/env python3
"""Select a size-stratified, content-spread corpus subset for picker sweeps.

The rendition corpus (train_renditions_2026-06-14) holds ~1482 SDR PNGs that are
multiple scales of ~370 source images (filename `<source>.scale<W>x<H>.png`).
A picker needs the four size classes (tiny/small/medium/large) each populated
with enough distinct images, plus content spread across source classes.

This bins renditions by pixel count, then within each bin samples up to N,
preferring distinct source stems for content diversity. Symlinks the picks into
<out> (a flat dir the sweep's --sources consumes). Deterministic (fixed seed).

Usage:
  select_corpus.py --src <renditions-dir> --out <corpus-dir> --per-class 40
"""
import argparse
import os
import random
import re
from pathlib import Path

# pixel-count bin upper bounds (inclusive), matching the trainer's size_class
TINY = 64 * 64          # <= 4096
SMALL = 256 * 256       # <= 65536
MEDIUM = 1024 * 1024    # <= ~1.05M ; above = large

SCALE_RE = re.compile(r"scale(\d+)x(\d+)", re.IGNORECASE)


def size_class(px: int) -> str:
    if px <= TINY:
        return "tiny"
    if px <= SMALL:
        return "small"
    if px <= MEDIUM:
        return "medium"
    return "large"


def source_stem(name: str) -> str:
    # everything before the first '.scale' — the source image identity
    return name.split(".scale")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument(
        "--max-mp",
        type=float,
        default=4.0,
        help="skip renditions larger than this many megapixels (the "
        "non-orchestrator GPU metric path OOMs/thrashes on >~6 MP); the "
        "'large' size class stays 1..max-mp MP, plenty for the size axis",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    buckets: dict[str, list[Path]] = {"tiny": [], "small": [], "medium": [], "large": []}
    for p in sorted(args.src.glob("*.png")):
        m = SCALE_RE.search(p.name)
        if not m:
            continue
        px = int(m.group(1)) * int(m.group(2))
        if args.max_mp and px > args.max_mp * 1_000_000:
            continue
        buckets[size_class(px)].append(p)

    args.out.mkdir(parents=True, exist_ok=True)
    # clear stale symlinks
    for old in args.out.glob("*.png"):
        if old.is_symlink():
            old.unlink()

    picked_total = 0
    for cls, files in buckets.items():
        # group by source stem so we can prefer distinct sources
        by_src: dict[str, list[Path]] = {}
        for f in files:
            by_src.setdefault(source_stem(f.name), []).append(f)
        stems = list(by_src)
        rng.shuffle(stems)
        picks: list[Path] = []
        # round-robin one rendition per distinct source until we hit per-class
        ri = 0
        while len(picks) < args.per_class and stems:
            progressed = False
            for s in stems:
                lst = by_src[s]
                if ri < len(lst):
                    picks.append(lst[ri])
                    progressed = True
                    if len(picks) >= args.per_class:
                        break
            if not progressed:
                break
            ri += 1
        for f in picks:
            link = args.out / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
        n_src = len({source_stem(f.name) for f in picks})
        print(f"{cls:>7}: {len(picks):4d} renditions  ({n_src} distinct sources, pool={len(files)})")
        picked_total += len(picks)
    print(f"total: {picked_total} renditions -> {args.out}")


if __name__ == "__main__":
    main()
