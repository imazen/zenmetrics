#!/usr/bin/env python3
"""Generate a size-dense, tier-balanced rendition corpus by DOWNSCALING sources.

The picker rendition pool is size-skewed (86% tiny: only a few dozen small/medium
exist), which starves the trainer's small/medium (size_class, zq) cells and caps the
oracle gap (DATA_STARVED / PER_SIZE_TAIL gates). This tool fixes that by taking the
largest rendition per source and emitting a DENSE log-spaced ladder of smaller sizes
(Lanczos, **never upscaling**), so every size tier gets dense coverage from the same
content — exactly what the dense-sampling discipline (global CLAUDE.md) prescribes.

Naming matches the existing convention `<source>.scale<W>x<H>.png` so the downstream
sweep + imazen26 feature join + omni_to_pareto work unchanged.

    gen_dense_corpus.py --src <renditions-dir> --out <corpus-dir> \
        [--max-per-source 11] [--min-px 64] [--seed 20260625]

Picks, per source, the rendition with the largest longest-side as the resize source
(no upscaling — only sizes <= that side are emitted). Writes a provenance TSV
(rendition -> source-file + sha256 + WxH + tier) next to the corpus.
"""
import argparse
import collections
import hashlib
import re
from pathlib import Path

from PIL import Image

SCALE_RE = re.compile(r"scale(\d+)x(\d+)", re.IGNORECASE)
# Log-spaced longest-side targets spanning tiny -> large (<=1 MP square ~ 1024).
# Dense in the starved small/medium band; only emit those <= the source side.
LADDER = [64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 1024]


def source_id(name: str) -> str:
    """Identity = everything before the first '.scale' (matches the corpus convention)."""
    return name.split(".scale")[0]


def longest_side(path: Path) -> int:
    m = SCALE_RE.search(path.name)
    if m:
        return max(int(m.group(1)), int(m.group(2)))
    try:
        with Image.open(path) as im:
            return max(im.size)
    except Exception:
        return 0


def tier(px: int) -> str:
    mp = px * px / 1e6  # square-equivalent upper bound for the longest side
    return "tiny" if px < 224 else "small" if px < 448 else "medium" if px < 840 else "large"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="dir of source renditions")
    ap.add_argument("--out", required=True, type=Path, help="output corpus dir")
    ap.add_argument("--max-per-source", type=int, default=len(LADDER),
                    help="cap ladder steps per source (default: full ladder)")
    ap.add_argument("--min-px", type=int, default=64, help="skip targets below this longest-side")
    ap.add_argument("--seed", type=int, default=20260625)
    ap.add_argument("--sizes", type=str, default="",
                    help="comma-sep longest-side targets; overrides LADDER (e.g. to add a dense small-end below/between the default ladder)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # Largest rendition per source = the resize source (no upscaling above it).
    best: dict[str, Path] = {}
    best_side: dict[str, int] = {}
    for p in sorted(args.src.glob("*.png")):
        sid = source_id(p.name)
        side = longest_side(p)
        if side > best_side.get(sid, 0):
            best_side[sid] = side
            best[sid] = p

    prov_path = args.out / "_provenance.tsv"
    tiers = collections.Counter()
    n_rend = 0
    with open(prov_path, "w") as prov:
        prov.write("rendition\tsource_file\tsource_sha256\twidth\theight\tlongest\ttier\n")
        for sid, src_path in sorted(best.items()):
            try:
                im = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"skip {src_path.name}: {e}")
                continue
            src_side = max(im.size)
            sha = hashlib.sha256(src_path.read_bytes()).hexdigest()
            ladder = [int(x) for x in args.sizes.split(",") if x.strip()] if args.sizes else LADDER
            steps = [t for t in ladder if args.min_px <= t <= src_side][: args.max_per_source]
            for target in steps:
                scale = target / src_side
                w = max(1, round(im.width * scale))
                h = max(1, round(im.height * scale))
                out_name = f"{sid}.scale{w}x{h}.png"
                out_path = args.out / out_name
                if not out_path.exists():
                    im.resize((w, h), Image.LANCZOS).save(out_path)
                t = tier(max(w, h))
                tiers[t] += 1
                n_rend += 1
                prov.write(f"{out_name}\t{src_path.name}\t{sha}\t{w}\t{h}\t{max(w, h)}\t{t}\n")

    print(f"sources: {len(best)}  renditions: {n_rend}")
    print(f"tier counts: {dict(tiers)}")
    print(f"provenance: {prov_path}")


if __name__ == "__main__":
    main()
