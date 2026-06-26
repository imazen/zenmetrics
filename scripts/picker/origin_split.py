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

The id is the LEADING stem (after an optional `o_`/`v2_src` prefix), so it is
correct for ALL of: dense renditions `o_1004.scale…`, crops `o_1004_c25_tl.scale…`,
`v2_src0001.png.scale…`, raw imazen-26 descriptive originals
`1003_general_…_4000x3000.sdr.png` (→ 1003, NOT the trailing dimension 3000), and
bare manifest stems `1003`. The only requirement: the origin stem must LEAD the
name (the imazen-26 + dense conventions all satisfy this), so no stem-mapping step
is needed before rendition generation. A name with no leading numeric stem → None.
"""
import re

TRAIN_DIGITS = frozenset("02468")
VAL_DIGITS = frozenset("135")
TEST_DIGITS = frozenset("79")

_REND = re.compile(r"\.scale\d+x\d+(\.png)?$", re.IGNORECASE)
# The origin id is the LEADING stem — after an optional `o_` / `v2_src` prefix —
# NOT a trailing number. Leading-stem is robust to crop labels (`o_1004_c25_tl`),
# trailing dimensions (`1003_..._4000x3000`), and descriptive suffixes, all of
# which a trailing-number rule would wrongly grab. Patterns tried in order:
_STEM_PATS = (
    re.compile(r"^o_(\d+)"),       # dense-corpus rendition:  o_1004[...]
    re.compile(r"^v2_src(\d+)"),   # imazen-26-png-v2:        v2_src0001[...]
    re.compile(r"^(\d+)"),         # bare manifest stem / descriptive: 1003[_general...]
)


def origin_stem(name: str) -> str:
    """Strip the directory + the `.scaleWxH(.png)` rendition suffix → origin stem."""
    base = name.rsplit("/", 1)[-1]
    return _REND.sub("", base)


def origin_id(name: str):
    """The origin's numeric id — the LEADING stem (after an optional o_/v2_src
    prefix), or None if there's no leading numeric stem."""
    base = origin_stem(name)
    for pat in _STEM_PATS:
        m = pat.match(base)
        if m:
            return m.group(1)
    return None


def origin_id_last_digit(name: str):
    """Last digit of the origin's (leading) numeric id, or None."""
    oid = origin_id(name)
    return oid[-1] if oid else None


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
        # leading-stem robustness (the cases the old trailing-number rule got WRONG):
        "o_1004_c25_tl.scale48x64.png": "train",        # crop label — leading o_1004, not crop 25
        "1003_general_oceanfront_4000x3000.sdr.png": "val",   # descriptive — 1003, NOT dim 3000
        "9736_gen_products_brass-clock_p0497_1024x1024.sdr.png": "train",  # 9736 -> 6, NOT 1024
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
