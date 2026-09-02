#!/usr/bin/env python3
"""HDR corpus precheck — gates G0.2 and G0.5 of the HBD AVIF arm.

Registered by ``benchmarks/avif_hdr_arm_plan_2026-09-02.md``. Reads the PNG
``IHDR`` and ``cICP`` chunks of every candidate HDR reference and answers the two
questions that must be settled BEFORE any encode cell is declared:

* **G0.2 — will ``decode_hdr_ref`` accept this corpus?**  That function
  (``zenmetrics-cli/src/sweep/hdr.rs:97``) takes PQ PNGs only: 16-bit with
  ``cICP`` transfer characteristics 16.  Anything else is refused at encode time,
  so finding out here costs seconds instead of a wave.
* **G0.5 — is colour volume confounded with content?**  Measured 2026-09-02 on
  ``imazen-26-png-v2``: the 76 HDR references split **33 BT.709 / 43 Display-P3**,
  and primaries track content class almost exactly (interiors 19/20 BT.709,
  nature 39/47 P3).  A content-only k-means over that corpus yields picks whose
  gamut is effectively determined by their content, which makes every "content
  effect" a content-or-gamut ambiguity with no post-hoc fix.  The cross-tab this
  prints is the input to the plan's >=5-per-primaries balance constraint and must
  be published with the picks.

Reads the file header only -- no pixel decode, no image library, no network.

Exit status is the gate: **0 = G0.2 PASS**, **1 = at least one file fails**,
**2 = no files matched** (a silent empty run is how a precheck lies).

    python3 scripts/hdr_corpus_precheck.py /mnt/v/output/imazen-26-png-v2
    python3 scripts/hdr_corpus_precheck.py DIR --glob '**/*.hdr.png' --tsv picks_precheck.tsv
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import struct
import sys

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# cICP transfer characteristics (ITU-T H.273). 16 = PQ (SMPTE ST 2084) is the
# only value decode_hdr_ref accepts; 18 = HLG is HDR but is refused there too.
TRANSFER_NAMES = {1: "BT.709", 8: "linear", 13: "sRGB", 16: "PQ", 18: "HLG"}
PRIMARIES_NAMES = {1: "BT.709", 9: "BT.2020", 11: "DCI-P3", 12: "Display-P3"}
REQUIRED_TRANSFER = 16
REQUIRED_BIT_DEPTH = 16
REQUIRED_COLOR_TYPE = 2  # truecolour RGB


class BadPng(Exception):
    """The file is not a PNG we can read a header out of."""


def read_png_header(path: pathlib.Path) -> dict:
    """Return {bit_depth, color_type, cicp} from a PNG's header chunks.

    Stops at the first ``IDAT``/``IEND`` -- every chunk we care about is
    required by the spec to precede the image data, so this never reads pixels.
    ``cicp`` is None when the chunk is absent.
    """
    with path.open("rb") as fh:
        if fh.read(8) != PNG_MAGIC:
            raise BadPng("not a PNG (bad magic)")
        info: dict = {"bit_depth": None, "color_type": None, "cicp": None}
        while True:
            head = fh.read(8)
            if len(head) < 8:
                break
            length, ctype = struct.unpack(">I4s", head)
            if length > (1 << 31):
                raise BadPng(f"implausible chunk length {length} in {ctype!r}")
            data = fh.read(length)
            fh.read(4)  # CRC — not verified; we are reading, not validating
            if ctype == b"IHDR":
                if len(data) < 10:
                    raise BadPng("short IHDR")
                info["bit_depth"] = data[8]
                info["color_type"] = data[9]
            elif ctype == b"cICP":
                if len(data) < 4:
                    raise BadPng("short cICP")
                info["cicp"] = tuple(data[:4])  # primaries, transfer, matrix, range
            elif ctype in (b"IDAT", b"IEND"):
                break
        if info["bit_depth"] is None:
            raise BadPng("no IHDR")
        return info


def gate_g02(info: dict) -> list[str]:
    """Reasons this file would be refused by decode_hdr_ref. Empty = accepted."""
    reasons = []
    if info["bit_depth"] != REQUIRED_BIT_DEPTH:
        reasons.append(f"bit_depth={info['bit_depth']} (need {REQUIRED_BIT_DEPTH})")
    if info["color_type"] != REQUIRED_COLOR_TYPE:
        reasons.append(f"color_type={info['color_type']} (need {REQUIRED_COLOR_TYPE}=RGB)")
    cicp = info["cicp"]
    if cicp is None:
        reasons.append("no cICP chunk (transfer unsignalled)")
    elif cicp[1] != REQUIRED_TRANSFER:
        name = TRANSFER_NAMES.get(cicp[1], "?")
        reasons.append(f"transfer={cicp[1]} ({name}) (need {REQUIRED_TRANSFER}=PQ)")
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=pathlib.Path, help="corpus root to walk")
    ap.add_argument("--glob", default="**/*.hdr.png", help="pattern under root (default: %(default)s)")
    ap.add_argument("--tsv", type=pathlib.Path, help="write a per-file TSV here (the picks-TSV input)")
    args = ap.parse_args()

    files = sorted(args.root.glob(args.glob))
    if not files:
        print(f"G0.2 ERROR: no files matched {args.glob!r} under {args.root}", file=sys.stderr)
        return 2

    rows, failures = [], []
    combos: collections.Counter = collections.Counter()
    cross: collections.Counter = collections.Counter()

    for f in files:
        try:
            info = read_png_header(f)
        except (BadPng, OSError) as e:
            failures.append((f, [str(e)]))
            continue
        reasons = gate_g02(info)
        if reasons:
            failures.append((f, reasons))
        cicp = info["cicp"] or (None, None, None, None)
        # Category = the immediate parent dir, which is how imazen-26 is organised.
        category = f.parent.name
        combos[(info["bit_depth"], info["color_type"], cicp)] += 1
        cross[(category, cicp[0])] += 1
        rows.append((str(f), category, info["bit_depth"], info["color_type"], *cicp))

    print(f"files scanned: {len(files)}")
    print("\n== header combinations ==")
    for (bd, ct, cicp), n in sorted(combos.items(), key=lambda kv: -kv[1]):
        tname = TRANSFER_NAMES.get(cicp[1], "?") if cicp[1] is not None else "-"
        pname = PRIMARIES_NAMES.get(cicp[0], "?") if cicp[0] is not None else "-"
        print(f"  bit_depth={bd} color_type={ct} cICP={cicp} [{pname}/{tname}]  n={n}")

    # G0.5 — primaries x content. Read this table before clustering, not after.
    prims = sorted({p for _, p in cross if p is not None})
    cats = sorted({c for c, _ in cross})
    print("\n== G0.5: primaries x content class ==")
    header = f"{'category':<36}" + "".join(f"{PRIMARIES_NAMES.get(p, p)!s:>14}" for p in prims)
    print(header)
    for c in cats:
        print(f"{c:<36}" + "".join(f"{cross[(c, p)]:>14}" for p in prims))
    print(f"{'TOTAL':<36}" + "".join(f"{sum(cross[(c, p)] for c in cats):>14}" for p in prims))
    if len(prims) > 1:
        print(
            "\n  NOTE: more than one primaries value is present. Per the arm plan's G0.5, the\n"
            "  K-selection must carry >=5 of each, this cross-tab must ship with the picks, and\n"
            "  no content-class conclusion may be stated without it (colour volume and content\n"
            "  are confounded in imazen-26: interiors 19/20 BT.709, nature 39/47 P3)."
        )

    if args.tsv:
        with args.tsv.open("w") as fh:
            fh.write("path\tcategory\tbit_depth\tcolor_type\tprimaries\ttransfer\tmatrix\trange\n")
            for r in rows:
                fh.write("\t".join("" if v is None else str(v) for v in r) + "\n")
        print(f"\nwrote {len(rows)} rows -> {args.tsv}")

    print(f"\n== G0.2: {len(files) - len(failures)}/{len(files)} accepted by decode_hdr_ref ==")
    if failures:
        print(f"G0.2 FAIL — {len(failures)} file(s) would be refused at encode time:", file=sys.stderr)
        for f, reasons in failures[:20]:
            print(f"  {f.name}: {'; '.join(reasons)}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
        return 1
    print("G0.2 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
