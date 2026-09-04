#!/usr/bin/env python3
"""Regression test for the S1c pixel-count trap in avif_speed_analyze.py.

Why this file exists. `pixels_of()` parsed pixel count only from a
`.crop(N).png` filename tag. The S1c corpus (per
benchmarks/avif_speed_instrument_2026-09-03.md) breaks that: the BUDGET
corpus deliberately keeps native-looking `NNNN.scaleWxH.png` names on files
that are actually 1024^2 crops, and those names don't match `.crop(N).png`
either -- so `pixels_of` silently returned None for every S1c row and the
alpha+beta fit went empty/degenerate with no visible error. The fix reads
real dimensions from each PNG's own IHDR chunk via an optional
`--sources-dir` pixel map, and refuses (SystemExit) if two dirs disagree on
one basename's pixel count -- exactly the native-vs-budget same-name
collision this corpus creates on purpose.

Run standalone (`python3 scripts/jobsys/test_avif_speed_analyze.py`) or
under pytest. Hermetic: writes tiny synthetic PNG headers to a tempdir, no
real corpus, no network, ~instant.
"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avif_speed_analyze as S  # noqa: E402

FAILS, RAN = [], []


def case(name):
    def deco(fn):
        RAN.append(name)
        try:
            fn()
        except AssertionError as e:
            FAILS.append(f"{name}: {e}")
        except Exception as e:  # a crash is a failure too
            FAILS.append(f"{name}: unexpected {type(e).__name__}: {e}")
        return fn
    return deco


def _write_png(path, width, height):
    """Minimal PNG: signature + IHDR chunk only (no IDAT/IEND needed --
    read_png_dimensions never reads past IHDR). CRC bytes are dummy; the
    reader (like hdr_corpus_precheck.py's) doesn't validate them."""
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as fh:
        fh.write(S.PNG_MAGIC)
        fh.write(struct.pack(">I4s", len(ihdr_data), b"IHDR"))
        fh.write(ihdr_data)
        fh.write(b"\x00\x00\x00\x00")  # dummy CRC


@case("1. read_png_dimensions reads the real IHDR width/height")
def _():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "x.png")
        _write_png(p, 1024, 683)
        assert S.read_png_dimensions(p) == (1024, 683)


@case("2. read_png_dimensions returns None on a non-PNG file")
def _():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "not_a_png.png")
        with open(p, "wb") as fh:
            fh.write(b"not a png at all")
        assert S.read_png_dimensions(p) is None


@case("3. pixels_of still uses the crop-regex when no pixmap is given (S1a/S1b unaffected)")
def _():
    assert S.pixels_of("foo.crop512.png") == 512 * 512
    assert S.pixels_of("foo.crop512.png", pixmap=None) == 512 * 512


@case("4. pixels_of on an S1c-style name with NO pixmap is unresolved (the pre-fix bug, pinned)")
def _():
    # This is the exact class of name the resume doc flags: scale-tagged,
    # not crop-tagged. Without a pixel map it must stay None, not guess.
    assert S.pixels_of("1220.scale3000x4000.png") is None
    assert S.pixels_of("1008.scale3000x4000.png", pixmap={}) is None


@case("5. pixels_of resolves an S1c-style name via the pixmap (the fix)")
def _():
    with tempfile.TemporaryDirectory() as native_dir:
        _write_png(os.path.join(native_dir, "1220.scale3000x4000.png"), 3000, 4000)
        pixmap = S.load_pixel_map([native_dir])
        assert S.pixels_of("1220.scale3000x4000.png", pixmap) == 3000 * 4000


@case("6. THE TRAP: budget corpus reuses the native basename at a different pixel count")
def _():
    # This is the concrete S1c hazard: avifsvt-subsample (native) and
    # avif-doe-1024 (budget) both contain a file literally named
    # "1220.scale3000x4000.png" -- native is really 3000x4000, budget is a
    # 1024^2-class crop. A single analyzer run is scoped to ONE corpus by
    # its --s1a glob, so load_pixel_map() over the CORRECT single
    # --sources-dir must return that corpus's own true size...
    with tempfile.TemporaryDirectory() as native_dir:
        _write_png(os.path.join(native_dir, "1220.scale3000x4000.png"), 3000, 4000)
        pixmap = S.load_pixel_map([native_dir])
        assert pixmap["1220.scale3000x4000.png"] == 3000 * 4000

    with tempfile.TemporaryDirectory() as budget_dir:
        _write_png(os.path.join(budget_dir, "1220.scale3000x4000.png"), 1024, 1024)
        pixmap = S.load_pixel_map([budget_dir])
        assert pixmap["1220.scale3000x4000.png"] == 1024 * 1024


@case("7. load_pixel_map REFUSES to silently pool native+budget under one basename")
def _():
    # ...and if someone accidentally passes BOTH corpora's dirs to one
    # invocation, load_pixel_map must fail loud rather than pick one
    # arbitrarily (which would silently conflate two different physical
    # images sharing a name).
    with tempfile.TemporaryDirectory() as native_dir, \
         tempfile.TemporaryDirectory() as budget_dir:
        _write_png(os.path.join(native_dir, "1220.scale3000x4000.png"), 3000, 4000)
        _write_png(os.path.join(budget_dir, "1220.scale3000x4000.png"), 1024, 1024)
        try:
            S.load_pixel_map([native_dir, budget_dir])
        except SystemExit as e:
            assert e.code not in (0, None), "must exit non-zero on a conflict"
            return
        raise AssertionError("load_pixel_map pooled conflicting basenames silently")


@case("8. unresolved cells are NAMED, never silently dropped (main()'s accounting)")
def _():
    # Regression pin for the actual failure mode this bug produced: rows
    # with no resolvable pixel count must be visible in the summary, not
    # just vanish into an empty arms table with no explanation.
    with tempfile.TemporaryDirectory() as t:
        tsv = os.path.join(t, "s1c_pass1.tsv")
        with open(tsv, "w") as fh:
            fh.write("image_path\tcodec\tq\tknob_tuple_json\tencoded_bytes\tencode_ms\tencoded_filename\tdecode_ms\n")
            fh.write('1220.scale3000x4000.png\tavif\t50\t{"backend":"zenravif","speed":1}\t1000\t42.0\tout.avif\t3.0\n')
        acc, _ = S.parse_rows([tsv])
        assert len(acc) == 1
        (img, backend, speed, q), v = next(iter(acc.items()))
        assert S.pixels_of(img, pixmap=None) is None, "must stay unresolved with no pixmap"


for n in RAN:
    print(("FAIL " if any(f.startswith(n) for f in FAILS) else "ok   ") + n)
for f in FAILS:
    print("  " + f, file=sys.stderr)
print(f"\nRESULT: {'PASS' if not FAILS else 'FAIL'} ({len(RAN) - len(FAILS)}/{len(RAN)})")
sys.exit(1 if FAILS else 0)
