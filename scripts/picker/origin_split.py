#!/usr/bin/env python3
"""CANONICAL train/val/test split for picker work — the ONE source of truth.

Rule (set by the user 2026-06-26): the split is by ORIGIN image, by the last
digit of the origin's numeric id, and EVERY sizing/crop/encode derivative of an
origin inherits the origin's bucket — so no derivative ever leaks across the
split:

    last digit of origin id in {0,2,4,6,8} -> TRAIN
                              in {1,3,5}     -> VALIDATION
                              in {7,9}       -> TEST

Train only ever sees even-origin content. Validation = ids ending 1/3/5; test =
7/9. This is origin-level (NOT per-rendition) and deterministic (NOT a seeded
shuffle) — reproducible across blind sessions with zero state.

`origin_stem(name)`  : rendition/variant filename -> origin stem
   o_1004.scale48x64.png            -> o_1004
   v2_src0001.png.scale256x250.png  -> v2_src0001.png
   1004 (bare manifest stem)        -> 1004
`origin_id_last_digit(name)` : the parity digit (or None if no numeric id)
`split_of(name)`     : 'train' | 'val' | 'test' | None

Import this everywhere (train_hybrid, omni_to_pareto, corpus segmentation) — do
not re-implement the rule.

⚠ GOTCHA — feed it PIPELINE rendition names, NOT raw descriptive originals.
The id is the TRAILING numeric token of the origin stem. That is correct for the
pipeline's rendition naming (`o_<stem>`, `v2_src<NNNN>`, bare manifest stem
`1003`). It is WRONG for raw imazen-26 originals like
`1003_general_oceanfront_..._4000x3000.sdr.png` — the trailing number there is the
DIMENSION (3000 → digit 0 → "train") not the stem (1003 → "val"). So the clean
re-sweep MUST stem-map imazen-26 → `o_<stem>.png` BEFORE rendition generation (the
dense corpus already does this: `o_1004.scale…`). bare manifest stems (`1003`) are
fine because the stem IS the only number.
"""
import re

TRAIN_DIGITS = frozenset("02468")
VAL_DIGITS = frozenset("135")
TEST_DIGITS = frozenset("79")

_REND = re.compile(r"\.scale\d+x\d+(\.png)?$", re.IGNORECASE)
_TRAILING_NUM = re.compile(r"(\d+)\D*$")


def origin_stem(name: str) -> str:
    """Strip the directory + the `.scaleWxH(.png)` rendition suffix → origin stem."""
    base = name.rsplit("/", 1)[-1]
    return _REND.sub("", base)


def origin_id_last_digit(name: str):
    """Last digit of the origin's trailing numeric id, or None."""
    m = _TRAILING_NUM.search(origin_stem(name))
    return m.group(1)[-1] if m else None


def split_of(name: str):
    """'train' | 'val' | 'test' | None (None = unsplittable: no numeric origin id)."""
    d = origin_id_last_digit(name)
    if d is None:
        return None
    if d in TRAIN_DIGITS:
        return "train"
    if d in VAL_DIGITS:
        return "val"
    if d in TEST_DIGITS:
        return "test"
    return None


if __name__ == "__main__":
    # Self-test the rule against canonical examples (asserted).
    cases = {
        "o_1004.scale48x64.png": "train",               # id 1004, last digit 4 -> train
        "/data/o_1004.scale48x64.png": "train",
        "o_1003.scale72x96.png": "val",                 # 3 -> val
        "o_1007.scale72x96.png": "test",                # 7 -> test
        "v2_src0001.png.scale256x250.png": "val",       # 1 -> val
        "v2_src0002.png.scale49x64.png": "train",       # 2 -> train
        "v2_src0009.png.scale512x500.png": "test",      # 9 -> test
        "1000": "train",
        "1005": "val",
        "1009": "test",
    }
    bad = 0
    for name, want in cases.items():
        got = split_of(name)
        flag = "" if got == want else f"  ‼ EXPECTED {want}"
        if got != want:
            bad += 1
        print(f"  {name:38s} stem={origin_stem(name):20s} digit={origin_id_last_digit(name)} -> {got}{flag}")
    assert bad == 0, f"{bad} self-test mismatches"
    print("origin_split self-test OK")
